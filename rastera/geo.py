from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from affine import Affine
from async_geotiff import Window
from async_geotiff._transform import HasTransform
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError


@dataclass(frozen=True, slots=True)
class BBox:
    """A 2D axis-aligned bounding box in world coordinates.

    Stored as (minx, miny, maxx, maxy) using the dataset's CRS units.
    """

    minx: float
    miny: float
    maxx: float
    maxy: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.minx, self.miny, self.maxx, self.maxy))

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    def intersect(self, other: BBox) -> BBox | None:
        """Return the intersection with `other`, or None if there is no overlap."""
        inter = BBox(
            minx=max(self.minx, other.minx),
            miny=max(self.miny, other.miny),
            maxx=min(self.maxx, other.maxx),
            maxy=min(self.maxy, other.maxy),
        )
        # Empty/invalid intersection (no area).
        if inter.minx >= inter.maxx or inter.miny >= inter.maxy:
            return None
        return inter


def ensure_bbox(bbox: BBox | tuple[float, float, float, float]) -> BBox:
    """Normalize a bbox argument to a BBox instance."""
    return bbox if isinstance(bbox, BBox) else BBox(*bbox)


def normalize_band_indices(
    band_indices: Sequence[int] | None, n_bands: int
) -> list[int]:
    """Return a concrete list of 0-based band indices for internal use.

    Args:
        band_indices: 1-based band indices (matching rasterio convention).
            ``None`` selects all bands.
        n_bands: Total number of bands in the dataset.

    Returns:
        0-based indices suitable for NumPy indexing.
    """
    if band_indices is None:
        return list(range(n_bands))
    if len(band_indices) == 0:
        raise ValueError("band_indices must not be empty (use None for all bands)")
    for b in band_indices:
        if b < 1:
            raise ValueError(
                f"Band indices are 1-based (got {b}). Use 1 for the first band."
            )
        if b > n_bands:
            raise ValueError(
                f"Band index {b} out of range for dataset with {n_bands} band(s)."
            )
    return [b - 1 for b in band_indices]


def bounds_from_transform(transform: Affine, width: int, height: int) -> BBox:
    """Compute bounding box in world coordinates from an affine transform.

    Takes the hull of all four corners, so a rotated transform yields its
    true envelope rather than a degenerate box.
    """
    corners = [
        _affine_apply(transform, c, r)
        for c, r in ((0, 0), (width, 0), (0, height), (width, height))
    ]
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    return BBox(minx=min(xs), miny=min(ys), maxx=max(xs), maxy=max(ys))


class WindowOutOfRangeError(ValueError):
    """A bbox rounds to a zero-sized pixel window."""


def window_from_bbox(
    meta: HasTransform,
    bbox: BBox | tuple[float, float, float, float],
    *,
    snap_to_grid: bool = True,
) -> Window:
    """Return the pixel window for a world-space bbox.

    ``snap_to_grid=True`` (default) rounds outward onto the source grid, so the
    window never holds fewer pixels than *bbox* covers — one short leaves a row
    or column with nothing behind it. ``False`` gives ``round(span)`` pixels
    like rasterio, for callers that re-anchor the transform on *bbox*.
    """
    bbox = ensure_bbox(bbox)
    inv = ~meta.transform
    minx, miny, maxx, maxy = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy

    # top-left and bottom-right in pixel coords
    col_min_f, row_max_f = _affine_apply(inv, minx, maxy)
    col_max_f, row_min_f = _affine_apply(inv, maxx, miny)

    # The interval is clipped to the image first.  Clamping only the offset
    # (`max(0, floor(lo))`) leaves the span positive for a bbox lying entirely
    # left of or above the image, which yields a plausible window over the wrong
    # pixels.  For a bbox inside the image the clip is a no-op, so the sizing
    # rules below are unaffected.
    col_lo = max(0.0, min(col_min_f, col_max_f))
    col_hi = min(float(meta.width), max(col_min_f, col_max_f))
    row_lo = max(0.0, min(row_min_f, row_max_f))
    row_hi = min(float(meta.height), max(row_min_f, row_max_f))

    if col_hi <= col_lo or row_hi <= row_lo:
        raise WindowOutOfRangeError("BBox does not intersect image")

    if snap_to_grid:
        # A bare ceil would buy a whole extra column off ~transform's ULP error;
        # the else branch takes a difference, which cancels it.
        col_lo, col_hi = _denoise(col_lo), _denoise(col_hi)
        row_lo, row_hi = _denoise(row_lo), _denoise(row_hi)
        col_off, row_off = math.floor(col_lo), math.floor(row_lo)
        # col_hi/row_hi are already clipped to the image, so ceil stays in range.
        width = math.ceil(col_hi) - col_off
        height = math.ceil(row_hi) - row_off
    else:
        # rasterio passes float windows to GDALRasterIOEx (e.g. offset=5539.5,
        # height=1800.6); GDAL starts at floor(offset) and produces round(span)
        # pixels.  Replicating it makes native reads match rasterio's shape AND
        # pixel values (confirmed RMSE=0).
        col_off, row_off = math.floor(col_lo), math.floor(row_lo)
        width = min(meta.width, col_off + math.floor(col_hi - col_lo + 0.5)) - col_off
        height = min(meta.height, row_off + math.floor(row_hi - row_lo + 0.5)) - row_off

    # Sub-pixel sliver: the bbox overlaps but rounds to nothing.
    if width <= 0 or height <= 0:
        raise WindowOutOfRangeError("BBox does not intersect image")

    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def compute_paste_slices(
    *,
    src: HasTransform,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
) -> tuple[slice, slice, slice, slice] | None:
    """
    Compute aligned source/target slices for pasting a read window into a mosaic.

    This is used when you have already read a window (described by `src`)
    and want to paste it into a destination array whose pixel grid is described
    by `dst_transform` (pixel -> world for the destination).

    `src` must have `.transform`, `.width`, `.height` attributes.

    Returns (dst_rows, dst_cols, src_rows, src_cols) or None if there is no
    overlap after clipping to destination bounds.
    """
    dst_inv_transform = ~dst_transform

    # Top-left world coordinate of the source window.
    wx0, wy0 = _affine_apply(src.transform, 0, 0)

    # Map into destination pixel coordinates.
    dst_c0_f, dst_r0_f = _affine_apply(dst_inv_transform, wx0, wy0)
    dst_c0 = math.floor(dst_c0_f + 0.5)
    dst_r0 = math.floor(dst_r0_f + 0.5)

    # Initial (unclipped) target indices in destination pixel coordinates.
    dst_c1 = dst_c0 + src.width
    dst_r1 = dst_r0 + src.height

    # Clip to destination bounds. NumPy will silently clip slice endpoints that
    # exceed the array shape; we clip explicitly so we can also crop the source
    # window to keep source/target shapes aligned.
    clipped_dst_c0 = max(0, dst_c0)
    clipped_dst_r0 = max(0, dst_r0)
    clipped_dst_c1 = min(dst_width, dst_c1)
    clipped_dst_r1 = min(dst_height, dst_r1)

    if clipped_dst_c0 >= clipped_dst_c1 or clipped_dst_r0 >= clipped_dst_r1:
        return None

    # Corresponding crop on the source window.
    src_c0 = clipped_dst_c0 - dst_c0
    src_r0 = clipped_dst_r0 - dst_r0
    src_c1 = src_c0 + (clipped_dst_c1 - clipped_dst_c0)
    src_r1 = src_r0 + (clipped_dst_r1 - clipped_dst_r0)

    return (
        slice(clipped_dst_r0, clipped_dst_r1),
        slice(clipped_dst_c0, clipped_dst_c1),
        slice(src_r0, src_r1),
        slice(src_c0, src_c1),
    )


def transform_bbox(
    bbox: BBox, from_crs: int, to_crs: int, densify_pts: int = 21
) -> BBox:
    """Transform a BBox between CRS (EPSG codes).

    Delegates to pyproj's ``transform_bounds``, which densifies each edge with
    *densify_pts* samples to capture projected curvature *and* accounts for a
    pole or antimeridian falling in the box's interior. Hulling densified edges
    alone misses those: an EPSG:3413 sea-ice grid reaches lat 90 through its
    interior, and edge sampling tops out around 56.
    """
    if from_crs == to_crs:
        return bbox
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    try:
        minx, miny, maxx, maxy = transformer.transform_bounds(
            *bbox, densify_pts=densify_pts, errcheck=True
        )
    except ProjError as exc:
        raise ValueError(
            f"Cannot transform bbox {tuple(bbox)} from EPSG:{from_crs} to "
            f"EPSG:{to_crs}; the bbox reaches outside the target CRS's area "
            f"of use. Clip it first. ({exc})"
        ) from exc
    # errcheck only fires on hard PROJ errors; a bbox that merely leaves the
    # domain can still come back inf/nan, and no finite envelope is correct.
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        raise ValueError(
            f"Transforming bbox {tuple(bbox)} from EPSG:{from_crs} to "
            f"EPSG:{to_crs} produced inf/nan bounds; the bbox reaches outside "
            f"the target CRS's area of use. Clip it first."
        )
    # transform_bounds signals an antimeridian crossing by returning minx > maxx.
    # No axis-aligned BBox represents that, and silently hulling it would span
    # the wrong 340 degrees of the globe.
    if minx > maxx:
        raise ValueError(
            f"Bbox {tuple(bbox)} crosses the antimeridian in EPSG:{to_crs} "
            f"(wrapped bounds {minx} > {maxx}); no single axis-aligned bbox "
            f"covers it. Split the request at the antimeridian."
        )
    return BBox(float(minx), float(miny), float(maxx), float(maxy))


def _affine_apply(t: Affine, x: float, y: float) -> tuple[float, float]:
    """Apply an affine transform to a point, with correct typing."""
    rx, ry = t * (x, y)
    return float(rx), float(ry)


def _denoise(pixel_coord: float, tol: float = 1e-6) -> float:
    """Collapse a pixel coordinate onto an exact boundary it all but sits on.

    ``~transform`` carries a few ULPs of error through the divide, so a bbox
    edge that is exactly on the source grid arrives as 12105.000000000004.
    *tol* is in pixels — orders of magnitude above that error and far below
    any sub-pixel offset a caller could mean.
    """
    nearest = round(pixel_coord)
    return float(nearest) if abs(pixel_coord - nearest) < tol else pixel_coord


def _normalize_crs(crs: int | CRS) -> int:
    """Convert an EPSG integer or ``pyproj.CRS`` to an EPSG integer."""
    if isinstance(crs, int):
        return crs
    epsg = crs.to_epsg()
    if epsg is None:
        raise ValueError(
            f"CRS {crs.name!r} has no EPSG code; pass an integer EPSG code instead."
        )
    return epsg
