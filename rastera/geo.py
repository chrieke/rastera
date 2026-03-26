from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import numpy as np
from affine import Affine
from async_geotiff import Window
from pyproj import Transformer

_WARP_GRID_STEP = 16


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


def window_from_bbox(
    meta,
    bbox: BBox | tuple[float, float, float, float],
) -> Window:
    """Return pixel window for a world-space bbox."""

    # Map a world-space bbox to a clamped pixel window.
    bbox = ensure_bbox(bbox)
    inv = ~meta.transform
    minx, miny, maxx, maxy = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy

    # top-left and bottom-right in pixel coords
    col_min_f, row_max_f = _affine_apply(inv, minx, maxy)
    col_max_f, row_min_f = _affine_apply(inv, maxx, miny)

    # Match rasterio/GDAL window sizing: floor(offset) + round(span).
    #
    # rasterio passes float windows to GDALRasterIOEx (e.g. offset=5539.5,
    # height=1800.6).  GDAL starts at floor(offset) and produces
    # round(span) pixels.  We replicate that here so that native-
    # resolution reads return the same shape AND pixel values as
    # rasterio (confirmed RMSE=0).
    #
    # An alternative (floor/ceil on individual bounds) includes every
    # pixel partially covered by the bbox, but produces +1 pixel vs
    # rasterio whenever the bbox doesn't align to the source grid.
    col_lo = min(col_min_f, col_max_f)
    col_hi = max(col_min_f, col_max_f)
    row_lo = min(row_min_f, row_max_f)
    row_hi = max(row_min_f, row_max_f)

    col_off = max(0, math.floor(col_lo))
    row_off = max(0, math.floor(row_lo))
    width = min(meta.width, col_off + math.floor(col_hi - col_lo + 0.5)) - col_off
    height = min(meta.height, row_off + math.floor(row_hi - row_lo + 0.5)) - row_off

    if width <= 0 or height <= 0:
        msg = "BBox does not intersect image"
        raise ValueError(msg)

    return Window(col_off=col_off, row_off=row_off, width=width, height=height)



def compute_paste_slices(
    *,
    src,
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
    dst_c0 = int(round(dst_c0_f))
    dst_r0 = int(round(dst_r0_f))

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


def transform_bbox(bbox: BBox, from_crs: int, to_crs: int, densify_pts: int = 21) -> BBox:
    """Transform a BBox between CRS (EPSG codes).

    Densifies all 4 edges with *densify_pts* samples each (default 21,
    matching rasterio's ``transform_bounds``). This captures the curvature
    of projected edges at high latitudes and near polar singularities.
    """
    if from_crs == to_crs:
        return bbox
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    xs, ys = [], []
    for t in np.linspace(0, 1, densify_pts):
        dx = t * (bbox.maxx - bbox.minx)
        dy = t * (bbox.maxy - bbox.miny)
        xs.extend([bbox.minx + dx, bbox.maxx, bbox.maxx - dx, bbox.minx])
        ys.extend([bbox.miny, bbox.miny + dy, bbox.maxy, bbox.maxy - dy])
    xs_out, ys_out = transformer.transform(xs, ys)
    xs_valid = [x for x in xs_out if np.isfinite(x)]
    ys_valid = [y for y in ys_out if np.isfinite(y)]
    if not xs_valid or not ys_valid:
        raise ValueError(f"All coordinates became inf/nan transforming bbox from EPSG:{from_crs} to EPSG:{to_crs}")
    return BBox(min(xs_valid), min(ys_valid), max(xs_valid), max(ys_valid))


def resample_nearest(
    src_array: np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    nodata: int | float | None = None,
    transformer: Transformer | None = None,
) -> np.ndarray:
    """Resample src_array to a target grid using nearest-neighbor.

    Nearest-neighbor is the default resampling method in rasterio
    (``Resampling.nearest`` in both ``DatasetReader.read()`` and
    ``WarpedVRT``).

    Args:
        src_array: (bands, h, w) source data.
        src_transform: Affine pixel→world for source.
        dst_transform: Affine pixel→world for destination.
        dst_width: Output width in pixels.
        dst_height: Output height in pixels.
        nodata: Fill value for out-of-bounds pixels.
        transformer: pyproj Transformer (target CRS → source CRS). None if same CRS.
    """
    h, w = src_array.shape[1], src_array.shape[2]

    if transformer is None:
        # Same CRS: compose affines and use 1D index arrays (no meshgrid).
        combined = ~src_transform * dst_transform
        src_col_1d = np.floor(
            float(combined.a) * (np.arange(dst_width, dtype=np.float64) + 0.5)
            + float(combined.c)
        ).astype(np.intp)
        src_row_1d = np.floor(
            float(combined.e) * (np.arange(dst_height, dtype=np.float64) + 0.5)
            + float(combined.f)
        ).astype(np.intp)

        valid_col = (src_col_1d >= 0) & (src_col_1d < w)
        valid_row = (src_row_1d >= 0) & (src_row_1d < h)

        col_safe = np.clip(src_col_1d, 0, w - 1)
        row_safe = np.clip(src_row_1d, 0, h - 1)
        out = src_array[:, row_safe[:, np.newaxis], col_safe[np.newaxis, :]]

        if nodata is not None and not (np.all(valid_col) and np.all(valid_row)):
            fill = np.array(nodata, dtype=src_array.dtype)
            invalid = ~(valid_row[:, np.newaxis] & valid_col[np.newaxis, :])
            out[:, invalid] = fill
    else:
        # Coarse-grid + interpolation: transform sparse grid through pyproj,
        # bilinearly interpolate to full resolution.
        src_col_f, src_row_f = _coarse_grid_transform(
            dst_width,
            dst_height,
            dst_transform,
            src_transform,
            transformer,
        )
        src_col = np.floor(src_col_f).astype(np.intp)
        src_row = np.floor(src_row_f).astype(np.intp)

        valid = (src_col >= 0) & (src_col < w) & (src_row >= 0) & (src_row < h)

        src_col_safe = np.clip(src_col, 0, w - 1)
        src_row_safe = np.clip(src_row, 0, h - 1)
        out = src_array[:, src_row_safe, src_col_safe]

        if nodata is not None and not np.all(valid):
            fill = np.array(nodata, dtype=src_array.dtype)
            out[:, ~valid] = fill

    return out


# ---- Small public helpers ----


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
    """Compute bounding box in world coordinates from an affine transform."""
    x0, y0 = _affine_apply(transform, 0, 0)
    x1, y1 = _affine_apply(transform, width, height)
    return BBox(minx=min(x0, x1), miny=min(y0, y1), maxx=max(x0, x1), maxy=max(y0, y1))


def ensure_bbox(bbox: BBox | tuple[float, float, float, float]) -> BBox:
    """Normalize a bbox argument to a BBox instance."""
    return bbox if isinstance(bbox, BBox) else BBox(*bbox)



# ---- Private helpers ----


def _coarse_grid_transform(
    dst_width: int,
    dst_height: int,
    dst_transform: Affine,
    src_transform: Affine,
    transformer: Transformer,
    step: int = _WARP_GRID_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform destination pixels to source pixel coords via coarse-grid interpolation.

    Instead of transforming every destination pixel through pyproj, transforms a
    coarse grid (every ``step`` pixels) and bilinearly interpolates the rest.

    Returns ``(src_col_f, src_row_f)`` as float arrays of shape
    ``(dst_height, dst_width)``.
    """
    # Build coarse grid nodes, always including the last pixel.
    coarse_cols = np.arange(0, dst_width, step, dtype=np.float64)
    if coarse_cols[-1] < dst_width - 1:
        coarse_cols = np.append(coarse_cols, dst_width - 1)
    coarse_rows = np.arange(0, dst_height, step, dtype=np.float64)
    if coarse_rows[-1] < dst_height - 1:
        coarse_rows = np.append(coarse_rows, dst_height - 1)

    # Transform coarse grid: dst pixel centers → world → source CRS → source pixels
    cc, cr = np.meshgrid(coarse_cols + 0.5, coarse_rows + 0.5)
    cwx = float(dst_transform.a) * cc + float(dst_transform.c)
    cwy = float(dst_transform.e) * cr + float(dst_transform.f)
    cwx, cwy = transformer.transform(cwx, cwy)

    src_inv = ~src_transform
    coarse_src_col = float(src_inv.a) * cwx + float(src_inv.c)
    coarse_src_row = float(src_inv.e) * cwy + float(src_inv.f)

    n_coarse_rows = len(coarse_rows)
    coarse_col_centers = coarse_cols + 0.5
    coarse_row_centers = coarse_rows + 0.5
    full_col_centers = np.arange(dst_width, dtype=np.float64) + 0.5

    # Pass 1: interpolate along columns for each coarse row.
    temp_col = np.empty((n_coarse_rows, dst_width), dtype=np.float64)
    temp_row = np.empty((n_coarse_rows, dst_width), dtype=np.float64)
    for i in range(n_coarse_rows):
        temp_col[i] = np.interp(full_col_centers, coarse_col_centers, coarse_src_col[i])
        temp_row[i] = np.interp(full_col_centers, coarse_col_centers, coarse_src_row[i])

    # Pass 2: vectorised interpolation along rows.
    full_row_centers = np.arange(dst_height, dtype=np.float64) + 0.5
    row_idx = np.interp(
        full_row_centers, coarse_row_centers, np.arange(n_coarse_rows, dtype=np.float64)
    )
    row_lo = np.clip(np.floor(row_idx).astype(int), 0, n_coarse_rows - 2)
    row_frac = (row_idx - row_lo)[:, np.newaxis]  # (dst_height, 1)

    src_col_f = temp_col[row_lo] + row_frac * (temp_col[row_lo + 1] - temp_col[row_lo])
    src_row_f = temp_row[row_lo] + row_frac * (temp_row[row_lo + 1] - temp_row[row_lo])

    return src_col_f, src_row_f


def _affine_apply(t: Affine, x: float, y: float) -> tuple[float, float]:
    """Apply an affine transform to a point, with correct typing."""
    rx, ry = t * (x, y)  # type: ignore[misc]
    return float(rx), float(ry)
