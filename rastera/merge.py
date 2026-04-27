from __future__ import annotations

import math
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

import numpy as np
from affine import Affine
from async_geotiff import RasterArray
from pyproj import CRS, Transformer

from . import config
from .geo import (
    BBox,
    WindowOutOfRangeError,
    _affine_apply,
    _normalize_crs,
    bounds_from_transform,
    compute_paste_slices,
    ensure_bbox,
    normalize_band_indices,
    resample_nearest,
    transform_bbox,
)
from .reader import AsyncGeoTIFF, _CrsNodata, _grid_for_bbox, _make_output_array


async def merge(
    cogs: Sequence[AsyncGeoTIFF],
    *,
    bbox: BBox | tuple[float, float, float, float],
    bbox_crs: int | CRS,
    band_indices: Sequence[int] | None = None,
    fill_value: int | float = 0,
    target_crs: int | CRS | None = None,
    target_resolution: float,
    mosaic_method: Literal["first", "last"] = "first",
    crs_method: Literal["most_common", "first"] = "most_common",
    snap_to_grid: bool = True,
    use_overviews: bool = False,
) -> RasterArray:
    """
    Merge a bbox that may span multiple GeoTIFFs and return a single stitched array.

    Args:
        cogs: Sequence of opened AsyncGeoTIFF instances
        bbox: Bounding box of the merged image
        bbox_crs: EPSG code or ``pyproj.CRS`` of the bbox coordinate
            system. The bbox is transformed to the COGs' native CRS
            automatically.
        band_indices: 1-based band indices to read
        fill_value: Value used for pixels in `bbox` that aren't covered by any
            input GeoTIFF (i.e. a "no data" fill; not always 0).
        target_crs: Output EPSG code or ``pyproj.CRS``. Each COG is
            reprojected into this CRS before merging when it differs
            from the source. When ``None``, the CRS is inferred from
            the inputs using *crs_method*.
        target_resolution: Output pixel size in target CRS units.
        mosaic_method: Overlap strategy when multiple COGs cover the same pixel.
            ``"first"`` keeps the first valid pixel (matching rasterio.merge
            default). ``"last"`` lets later COGs overwrite earlier ones.
        crs_method: Strategy for choosing the output CRS when *target_crs*
            is ``None``. ``"most_common"`` picks the CRS shared by the most
            inputs; ``"first"`` uses the CRS of the first input.
        snap_to_grid: When True (default), the output grid snaps to the
            source pixel grid for exact 1:1 copies (no resampling); the
            bbox may shift by up to 1 pixel. When False, the output bbox
            matches the requested bbox exactly and nearest-neighbor
            resampling selects source pixels, matching rasterio/GDAL
            behaviour.
        use_overviews: When True, reads from pre-computed COG overview
            levels to save bandwidth. Overview pixels are resampled
            aggregates, not original measurements — expect reduced
            variance, dampened extremes, and altered spectral ratios
            compared to full-resolution data. Suitable for thumbnails
            or coarse segmentation; avoid for tasks requiring precise
            pixel values such as spectral index computation or
            per-pixel regression.

    Returns:
        An ``async_geotiff.RasterArray`` containing the merged mosaic.
    """
    if not cogs:
        raise ValueError("merge requires at least one AsyncGeoTIFF")

    bbox_crs = _normalize_crs(bbox_crs)
    if target_crs is not None:
        target_crs = _normalize_crs(target_crs)

    if target_crs is None:
        target_crs = _resolve_target_crs(cogs, crs_method)

    bbox = ensure_bbox(bbox)
    base = cogs[0]
    base_gt = base._geotiff

    # Validate + resolve count; keep original band_indices for cog.read() calls.
    n_out_bands = len(normalize_band_indices(band_indices, base.count))

    # Decide whether we need the reprojected merge path.
    all_same_crs = all(cog._crs_epsg == base._crs_epsg for cog in cogs[1:])
    all_same_res = all(
        math.isclose(float(cog._geotiff.transform.a), float(base_gt.transform.a))
        for cog in cogs[1:]
    )
    crs_matches_target = target_crs == base._crs_epsg
    res_matches_target = math.isclose(target_resolution, base_gt.res[0], rel_tol=1e-6)

    # Note: use_overviews is intentionally NOT included here.  The native
    # fast path is only reached when res_matches_target is True, meaning
    # the target resolution equals the native resolution.  Since overviews
    # are always coarser than native, _best_overview_for_resolution would
    # return None anyway — so use_overviews is correctly a no-op here.
    needs_reproject = (
        not all_same_crs
        or not all_same_res
        or not crs_matches_target
        or not res_matches_target
        or not snap_to_grid
    )

    if needs_reproject:
        return await _merge_reprojected(
            cogs,
            bbox=bbox,
            bbox_crs=bbox_crs,
            band_indices=band_indices,
            n_out_bands=n_out_bands,
            fill_value=fill_value,
            target_crs=target_crs,
            target_resolution=target_resolution,
            mosaic_method=mosaic_method,
            use_overviews=use_overviews,
        )

    # --- Native merge fast path (no resampling needed) ---
    # All COGs share the same CRS and resolution as the target.
    # snap_to_grid=True: output grid snaps to source pixel grid (1:1 copy).
    # snap_to_grid=False: output grid matches bbox exactly; paste rounds to
    #   nearest pixel, avoiding the resample_nearest overhead.
    _require_compatible_merge_inputs(cogs)

    native_crs = base._crs_epsg
    assert native_crs is not None
    native_bbox = transform_bbox(bbox, bbox_crs, native_crs)

    if snap_to_grid:
        window_transform, win_width, win_height = _mosaic_grid_from_bbox(
            base_transform=base_gt.transform,
            bbox=native_bbox,
        )
    else:
        window_transform, win_width, win_height = _grid_for_bbox(
            native_bbox,
            target_resolution,
        )

    # Get sub bboxes specific to the contributing image
    sub_bboxes: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        sub_bbox = native_bbox.intersect(BBox(*cog._geotiff.bounds))
        if sub_bbox is not None:
            sub_bboxes.append((cog, sub_bbox))

    async def _read_native_bands(cog: AsyncGeoTIFF, sb: BBox) -> RasterArray:
        indices = normalize_band_indices(band_indices, cog.count)
        return await cog._read_native(bbox=sb, band_indices=indices)

    out_data = await _gather_and_paste(
        contributing=sub_bboxes,
        dst_transform=window_transform,
        dst_width=win_width,
        dst_height=win_height,
        n_bands=n_out_bands,
        dtype=base_gt.dtype,
        nodata=base._nodata,
        fill_value=fill_value,
        read_fn=_read_native_bands,
        mosaic_method=mosaic_method,
    )
    return _make_output_array(
        out_data, window_transform, win_width, win_height, base_gt
    )


async def _merge_reprojected(
    cogs: Sequence[AsyncGeoTIFF],
    *,
    bbox: BBox,
    bbox_crs: int,
    band_indices: Sequence[int] | None,
    n_out_bands: int,
    fill_value: int | float,
    target_crs: int,
    target_resolution: float,
    mosaic_method: Literal["first", "last"] = "first",
    use_overviews: bool = False,
) -> RasterArray:
    """Path B: merge with reprojection — supports mixed-CRS inputs."""
    base = cogs[0]
    base_gt = base._geotiff
    out_crs = target_crs

    # Transform bbox into the output CRS
    target_bbox = transform_bbox(bbox, bbox_crs, out_crs)
    res = target_resolution

    # Build output grid
    out_transform, out_w, out_h = _grid_for_bbox(target_bbox, res)

    # Find contributing COGs by intersecting bounds (in target CRS) with output bbox
    contributing: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        assert cog._crs_epsg is not None
        cog_bounds_in_target = transform_bbox(
            BBox(*cog._geotiff.bounds), cog._crs_epsg, out_crs
        )
        sub_bbox = target_bbox.intersect(cog_bounds_in_target)
        if sub_bbox is not None:
            contributing.append((cog, sub_bbox))

    async def _read_and_reproject(cog: AsyncGeoTIFF, sb: BBox) -> RasterArray:
        assert cog._crs_epsg is not None
        # Compute an output-aligned sub-grid for this COG's contribution.
        subgrid = _output_subgrid(out_transform, out_w, out_h, sb)
        if subgrid is None:
            return _make_output_array(
                np.full((n_out_bands, 0, 0), 0, dtype=base_gt.dtype),
                out_transform,
                0,
                0,
                _CrsNodata(CRS.from_epsg(out_crs), cog._nodata),
            )
        sub_transform, sub_w, sub_h = subgrid

        # Geographic extent of the aligned sub-grid (may be slightly
        # larger than sb due to integer pixel rounding).
        read_bbox = bounds_from_transform(sub_transform, sub_w, sub_h)

        needs_reproject = cog._crs_epsg != out_crs
        if needs_reproject:
            read_bbox = transform_bbox(read_bbox, out_crs, cog._crs_epsg)

        # Pad by one native pixel so the resampler has source data for
        # every output pixel even when grids are offset by a fraction.
        pad = float(cog._geotiff.res[0])
        read_bbox = BBox(
            read_bbox.minx - pad,
            read_bbox.miny - pad,
            read_bbox.maxx + pad,
            read_bbox.maxy + pad,
        )

        # Select best overview for the target resolution.
        overview = None
        if use_overviews:
            src_res = res
            if needs_reproject:
                cog_bounds_target = transform_bbox(
                    BBox(*cog._geotiff.bounds), cog._crs_epsg, out_crs
                )
                cog_bounds_native = BBox(*cog._geotiff.bounds)
                src_res = res * (cog_bounds_native.width / cog_bounds_target.width)
            overview = cog._best_overview_for_resolution(src_res)

        indices = normalize_band_indices(band_indices, cog.count)
        native = await cog._read_native(
            bbox=read_bbox,
            band_indices=indices,
            overview=overview,
        )

        transformer = None
        if needs_reproject:
            transformer = Transformer.from_crs(out_crs, cog._crs_epsg, always_xy=True)

        out_data = resample_nearest(
            native.data,  # type: ignore[reportUnknownMemberType]
            src_transform=native.transform,
            dst_transform=sub_transform,
            dst_width=sub_w,
            dst_height=sub_h,
            nodata=cog._nodata,
            transformer=transformer,
        )

        geotiff_ref = _CrsNodata(CRS.from_epsg(out_crs), cog._nodata)
        return _make_output_array(out_data, sub_transform, sub_w, sub_h, geotiff_ref)

    out_data = await _gather_and_paste(
        contributing=contributing,
        dst_transform=out_transform,
        dst_width=out_w,
        dst_height=out_h,
        n_bands=n_out_bands,
        dtype=base_gt.dtype,
        nodata=base._nodata,
        fill_value=fill_value,
        read_fn=_read_and_reproject,
        mosaic_method=mosaic_method,
    )

    geotiff_ref = _CrsNodata(CRS.from_epsg(out_crs), base._nodata)
    return _make_output_array(out_data, out_transform, out_w, out_h, geotiff_ref)


async def _gather_and_paste(
    *,
    contributing: list[tuple[AsyncGeoTIFF, BBox]],
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    n_bands: int,
    dtype: np.dtype[Any] | None,
    nodata: int | float | None,
    fill_value: int | float,
    read_fn: Callable[[AsyncGeoTIFF, BBox], Awaitable[RasterArray]],
    mosaic_method: Literal["first", "last"] = "first",
) -> np.ndarray:
    """Read contributing COGs and paste into a single output array.

    Results are pasted in input order. Overlap is resolved by ``mosaic_method``:
    ``"first"`` keeps the first valid pixel, ``"last"`` lets later COGs
    overwrite earlier ones.

    Sequential by default: async-geotiff already parallelizes COG-block reads
    within each contributing TIFF, so an outer fan-out here multiplies the
    in-flight HTTP request count without adding throughput on a saturated
    link. Set ``rastera.set_concurrency(merge=N>1)`` to opt into outer
    parallelism. For ``mosaic_method="first"`` reads run in batches of N so
    the ``filled.all()`` early exit still triggers between batches.
    """
    out_array = np.full(
        (n_bands, dst_height, dst_width),
        fill_value,
        dtype=dtype,
    )

    if not contributing:
        return out_array

    filled = (
        np.zeros((dst_height, dst_width), dtype=bool)
        if mosaic_method == "first"
        else None
    )

    n = config._merge_concurrency
    for i in range(0, len(contributing), n):
        batch = contributing[i : i + n]
        arrays = await config._gather_bounded(
            n, [_read_or_skip(read_fn, cog, sub) for cog, sub in batch]
        )
        for (cog, sub_bbox), arr in zip(batch, arrays):
            if arr is None:
                continue

            slices = compute_paste_slices(
                src=arr,
                dst_transform=dst_transform,
                dst_width=dst_width,
                dst_height=dst_height,
            )
            if slices is None:
                continue
            dst_rows, dst_cols, src_rows, src_cols = slices
            src_data: np.ndarray[Any, Any] = arr.data[:, src_rows, src_cols]  # type: ignore[reportUnknownMemberType]

            if nodata is not None:
                if isinstance(nodata, float) and math.isnan(nodata):
                    valid = ~np.isnan(src_data)
                else:
                    valid = src_data != nodata
                src_valid = np.any(valid, axis=0)
            else:
                src_valid = None

            if mosaic_method == "first":
                assert filled is not None
                unfilled = ~filled[dst_rows, dst_cols]
                if src_valid is not None:
                    paste_mask = unfilled & src_valid
                else:
                    paste_mask = unfilled
                np.copyto(out_array[:, dst_rows, dst_cols], src_data, where=paste_mask)
                filled[dst_rows, dst_cols] |= paste_mask
            else:
                if src_valid is not None:
                    np.copyto(
                        out_array[:, dst_rows, dst_cols], src_data, where=src_valid
                    )
                else:
                    out_array[:, dst_rows, dst_cols] = src_data

        if mosaic_method == "first" and filled is not None and filled.all():
            return out_array

    return out_array


def _output_subgrid(
    out_transform: Affine, out_w: int, out_h: int, sub_bbox: BBox
) -> tuple[Affine, int, int] | None:
    """Compute the portion of the output grid covering *sub_bbox*.

    Returns ``(sub_transform, sub_width, sub_height)`` where
    *sub_transform* is an integer-pixel-offset window of *out_transform*,
    guaranteeing pixel-perfect alignment with the output grid.
    Returns ``None`` if the sub-bbox doesn't overlap.
    """
    inv = ~out_transform
    c0, r0 = _affine_apply(inv, sub_bbox.minx, sub_bbox.maxy)
    c1, r1 = _affine_apply(inv, sub_bbox.maxx, sub_bbox.miny)

    col_min = max(0, math.floor(min(c0, c1)))
    row_min = max(0, math.floor(min(r0, r1)))
    col_max = min(out_w, math.ceil(max(c0, c1)))
    row_max = min(out_h, math.ceil(max(r0, r1)))

    sub_w = col_max - col_min
    sub_h = row_max - row_min
    if sub_w <= 0 or sub_h <= 0:
        return None

    res = out_transform.a
    sub_transform = Affine(
        res,
        0,
        out_transform.c + col_min * res,
        0,
        -res,
        out_transform.f - row_min * res,
    )
    return sub_transform, sub_w, sub_h


def _mosaic_grid_from_bbox(
    *, base_transform: Affine, bbox: BBox
) -> tuple[Affine, int, int]:
    """
    Create a pixel-aligned mosaic grid for `bbox` on the `base_transform` grid.

    Unlike `window_from_bbox`, this does NOT clamp to any particular image size.
    """
    inv = ~base_transform

    # top-left and bottom-right in pixel coords (may be outside any single image)
    col_min_f, row_max_f = _affine_apply(inv, bbox.minx, bbox.maxy)
    col_max_f, row_min_f = _affine_apply(inv, bbox.maxx, bbox.miny)

    # Match rasterio/GDAL sizing: floor(offset) + round(span).
    col_lo = min(col_min_f, col_max_f)
    col_hi = max(col_min_f, col_max_f)
    row_lo = min(row_min_f, row_max_f)
    row_hi = max(row_min_f, row_max_f)

    col_min = math.floor(col_lo)
    row_min = math.floor(row_lo)
    col_max = col_min + max(1, math.floor(col_hi - col_lo + 0.5))
    row_max = row_min + max(1, math.floor(row_hi - row_lo + 0.5))

    width = int(col_max - col_min)
    height = int(row_max - row_min)
    if width <= 0 or height <= 0:
        raise ValueError("bbox does not cover any pixels on the base grid")

    transform = base_transform * Affine.translation(col_min, row_min)
    return transform, width, height


def _require_compatible_merge_inputs(cogs: Sequence[AsyncGeoTIFF]) -> None:
    """
    Validate that all inputs can be pasted onto a single shared pixel grid.

    Current merge assumes a north-up, non-rotated Affine grid (b=d=0) and that
    all sources are aligned to that grid (origins differ by whole pixels).
    """
    base = cogs[0]
    base_t = base._geotiff.transform
    scale_x = float(base_t.a)
    scale_y = float(-base_t.e)

    if not math.isclose(float(base_t.b), 0.0) or not math.isclose(float(base_t.d), 0.0):
        raise NotImplementedError(
            "merge currently requires a north-up (non-rotated) grid"
        )

    for cog in cogs[1:]:
        t = cog._geotiff.transform
        if cog._crs_epsg != base._crs_epsg:
            raise ValueError("All GeoTIFFs must share the same CRS EPSG")
        if cog.count != base.count:
            raise ValueError("All GeoTIFFs must share the same band count")
        if not math.isclose(float(t.a), scale_x):
            raise ValueError("All GeoTIFFs must share the same pixel width")
        if not math.isclose(float(-t.e), scale_y):
            raise ValueError("All GeoTIFFs must share the same pixel height")
        if not math.isclose(float(t.b), 0.0) or not math.isclose(float(t.d), 0.0):
            raise NotImplementedError(
                "merge currently requires a north-up (non-rotated) grid"
            )

        # Ensure origins line up on the same pixel grid (integer pixel offsets).
        off_x = (float(t.c) - float(base_t.c)) / scale_x
        off_y = (float(base_t.f) - float(t.f)) / scale_y
        if not math.isclose(off_x, round(off_x), abs_tol=1e-6) or not math.isclose(
            off_y, round(off_y), abs_tol=1e-6
        ):
            raise ValueError(
                "All GeoTIFFs must be aligned to the same pixel grid "
                "(origins differ by whole pixels)"
            )


def _resolve_target_crs(
    cogs: Sequence[AsyncGeoTIFF],
    crs_method: Literal["most_common", "first"],
) -> int:
    """Pick a target CRS from the input COGs."""
    if crs_method == "first":
        for cog in cogs:
            if cog._crs_epsg is not None:
                return cog._crs_epsg
    else:  # most_common
        counts = Counter(cog._crs_epsg for cog in cogs if cog._crs_epsg is not None)
        if counts:
            return counts.most_common(1)[0][0]
    msg = "No CRS found in any input GeoTIFF; pass target_crs explicitly."
    raise ValueError(msg)


async def _read_or_skip(
    read_fn: Callable[[AsyncGeoTIFF, BBox], Awaitable[RasterArray]],
    cog: AsyncGeoTIFF,
    sub_bbox: BBox,
) -> RasterArray | None:
    """Run *read_fn* and return ``None`` on the sub-pixel sliver case
    (BBox.intersect accepted the overlap but the rounded read window
    is zero), so callers can filter rather than wrap each call in
    try/except."""
    try:
        return await read_fn(cog, sub_bbox)
    except WindowOutOfRangeError:
        return None
