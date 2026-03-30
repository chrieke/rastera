from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

import numpy as np
from affine import Affine
from async_geotiff import Array
from pyproj import CRS

from .reader import AsyncGeoTIFF, _CrsNodata, _grid_for_bbox, _make_output_array
from .geo import (
    BBox,
    _affine_apply,
    bounds_from_transform,
    compute_paste_slices,
    ensure_bbox,
    normalize_band_indices,
    transform_bbox,
)


async def merge_cogs(
    cogs: Sequence[AsyncGeoTIFF],
    *,
    bbox: BBox | tuple[float, float, float, float],
    bbox_crs: int | None = None,
    band_indices: Sequence[int] | None = None,
    fill_value: int | float = 0,
    target_crs: int | None = None,
    target_resolution: float | None = None,
    method: Literal["first", "last"] = "first",
) -> Array:
    """
    Merge a bbox that may span multiple GeoTIFFs and return a single stitched array.

    Args:
        cogs: Sequence of opened AsyncGeoTIFF instances
        bbox: Bounding box of the merged image
        bbox_crs: EPSG code of the bbox coordinate system. When set, the bbox
            is transformed to the COGs' native CRS automatically.
        band_indices: 1-based band indices to read
        fill_value: Value used for pixels in `bbox` that aren't covered by any
            input GeoTIFF (i.e. a "no data" fill; not always 0).
        target_crs: Output EPSG code. When set, each COG is reprojected into
            this CRS before merging. Allows merging COGs from different CRS
            (e.g. adjacent UTM zones).
        target_resolution: Output pixel size in target CRS units.
        method: Overlap strategy when multiple COGs cover the same pixel.
            ``"first"`` keeps the first valid pixel (matching rasterio.merge
            default). ``"last"`` lets later COGs overwrite earlier ones.

    Returns:
        An ``async_geotiff.Array`` containing the merged mosaic.
    """
    if not cogs:
        raise ValueError("merge requires at least one AsyncGeoTIFF")

    if bbox_crs is None:
        raise ValueError("bbox_crs is required when bbox is provided")

    bbox = ensure_bbox(bbox)
    base = cogs[0]
    base_gt = base._geotiff

    # Validate + resolve count; keep original band_indices for cog.read() calls.
    n_out_bands = len(normalize_band_indices(band_indices, base_gt.count))

    # Decide whether we need the reprojected merge path.
    all_same_crs = all(
        cog._crs_epsg == base._crs_epsg for cog in cogs[1:]
    )
    all_same_res = all(
        math.isclose(float(cog._geotiff.transform.a), float(base_gt.transform.a))
        for cog in cogs[1:]
    )
    crs_matches_target = target_crs is None or target_crs == base._crs_epsg
    res_matches_target = target_resolution is None or math.isclose(
        target_resolution, base_gt.res[0], rel_tol=1e-6
    )

    needs_reproject = (
        not all_same_crs
        or not all_same_res
        or not crs_matches_target
        or not res_matches_target
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
            method=method,
        )

    # --- Path A: native merge (existing fast path) ---
    # The output grid is snapped to the source COG's pixel grid so that
    # pixels are copied 1:1 without resampling.  This differs from
    # rasterio.merge / WarpedVRT which place the origin at the exact
    # bbox corner, introducing a sub-pixel shift and nearest-neighbour
    # resampling even when CRS and resolution already match.
    _require_compatible_merge_inputs(cogs)

    native_crs = base._crs_epsg
    native_bbox = transform_bbox(bbox, bbox_crs, native_crs)

    # Get grid for bbox image mosaic
    window_transform, win_width, win_height, out_bounds = _mosaic_grid_from_bbox(
        base_transform=base_gt.transform,
        bbox=native_bbox,
    )

    # Get sub bboxes specific to the contributing image
    sub_bboxes: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        sub_bbox = native_bbox.intersect(BBox(*cog._geotiff.bounds))
        if sub_bbox is not None:
            sub_bboxes.append((cog, sub_bbox))

    async def _read_native_bands(
        cog: AsyncGeoTIFF, sb: BBox
    ) -> Array:
        indices = normalize_band_indices(band_indices, cog._geotiff.count)
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
        method=method,
    )
    return _make_output_array(out_data, window_transform, win_width, win_height, base_gt)


async def _merge_reprojected(
    cogs: Sequence[AsyncGeoTIFF],
    *,
    bbox: BBox,
    bbox_crs: int,
    band_indices: Sequence[int] | None,
    n_out_bands: int,
    fill_value: int | float,
    target_crs: int | None,
    target_resolution: float | None,
    method: Literal["first", "last"] = "first",
) -> Array:
    """Path B: merge with reprojection — supports mixed-CRS inputs."""
    base = cogs[0]
    base_gt = base._geotiff
    out_crs = target_crs if target_crs is not None else base._crs_epsg

    # Transform bbox into the output CRS
    target_bbox = transform_bbox(bbox, bbox_crs, out_crs)

    # Determine output resolution
    if target_resolution is not None:
        res = target_resolution
    else:
        # Preserve native pixel density of the first COG
        native_res = base_gt.res[0]
        src_bbox = transform_bbox(target_bbox, out_crs, base._crs_epsg)
        n_cols = max(1, round(src_bbox.width / native_res))
        n_rows = max(1, round(src_bbox.height / native_res))
        res_x = target_bbox.width / n_cols
        res_y = target_bbox.height / n_rows
        res = min(res_x, res_y)

    # Build output grid
    out_transform, out_w, out_h = _grid_for_bbox(target_bbox, res)

    # Find contributing COGs by intersecting their bounds (in target CRS) with output bbox
    contributing: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        cog_bounds_in_target = transform_bbox(
            BBox(*cog._geotiff.bounds), cog._crs_epsg, out_crs
        )
        sub_bbox = target_bbox.intersect(cog_bounds_in_target)
        if sub_bbox is not None:
            contributing.append((cog, sub_bbox))

    async def _read_and_reproject(
        cog: AsyncGeoTIFF, sb: BBox
    ) -> Array:
        return await cog.read(
            bbox=sb,
            bbox_crs=out_crs,
            target_crs=out_crs,
            target_resolution=res,
            band_indices=band_indices,
        )

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
        method=method,
    )

    geotiff_ref = _CrsNodata(CRS.from_epsg(out_crs), base._nodata) if target_crs else base_gt
    return _make_output_array(out_data, out_transform, out_w, out_h, geotiff_ref)


async def _gather_and_paste(
    *,
    contributing: list[tuple[AsyncGeoTIFF, BBox]],
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    n_bands: int,
    dtype: np.dtype,
    nodata: int | float | None,
    fill_value: int | float,
    read_fn: Callable[[AsyncGeoTIFF, BBox], Awaitable[Array]],
    method: Literal["first", "last"] = "first",
) -> np.ndarray:
    """Read contributing COGs sequentially and paste into a single output array.

    Results are pasted in input order. Overlap is resolved by ``method``:
    ``"first"`` keeps the first valid pixel, ``"last"`` lets later COGs
    overwrite earlier ones.
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
        if method == "first"
        else None
    )

    for cog, sub_bbox in contributing:
        arr = await read_fn(cog, sub_bbox)

        slices = compute_paste_slices(
            src=arr,
            dst_transform=dst_transform,
            dst_width=dst_width,
            dst_height=dst_height,
        )
        if slices is None:
            continue
        dst_rows, dst_cols, src_rows, src_cols = slices
        src_data = arr.data[:, src_rows, src_cols]

        if nodata is not None:
            if isinstance(nodata, float) and math.isnan(nodata):
                valid = ~np.isnan(src_data)
            else:
                valid = src_data != nodata
            src_valid = np.any(valid, axis=0)
        else:
            src_valid = None

        if method == "first":
            assert filled is not None
            unfilled = ~filled[dst_rows, dst_cols]
            if src_valid is not None:
                paste_mask = unfilled & src_valid
            else:
                paste_mask = unfilled
            np.copyto(out_array[:, dst_rows, dst_cols], src_data, where=paste_mask)
            filled[dst_rows, dst_cols] |= paste_mask
            if filled.all():
                break
        else:
            if src_valid is not None:
                np.copyto(out_array[:, dst_rows, dst_cols], src_data, where=src_valid)
            else:
                out_array[:, dst_rows, dst_cols] = src_data

    return out_array


def _mosaic_grid_from_bbox(
    *, base_transform: Affine, bbox: BBox
) -> tuple[Affine, int, int, BBox]:
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
    bounds = bounds_from_transform(transform, width, height)
    return transform, width, height, bounds


def _require_compatible_merge_inputs(cogs: Sequence[AsyncGeoTIFF]) -> None:
    """
    Validate that all inputs can be pasted onto a single shared pixel grid.

    Current merge assumes a north-up, non-rotated Affine grid (b=d=0) and that
    all sources are aligned to that grid (origins differ by whole pixels).
    """
    base = cogs[0]
    base_transform = base._geotiff.transform
    scale_x = float(base_transform.a)
    scale_y = float(-base_transform.e)

    if not math.isclose(float(base_transform.b), 0.0) or not math.isclose(
        float(base_transform.d),
        0.0,
    ):
        raise NotImplementedError(
            "merge currently requires a north-up (non-rotated) grid"
        )

    for cog in cogs[1:]:
        if cog._crs_epsg != base._crs_epsg:
            raise ValueError("All GeoTIFFs must share the same CRS EPSG")
        if cog._geotiff.count != base._geotiff.count:
            raise ValueError("All GeoTIFFs must share the same band count")
        if not math.isclose(float(cog._geotiff.transform.a), scale_x):
            raise ValueError("All GeoTIFFs must share the same pixel width")
        if not math.isclose(float(-cog._geotiff.transform.e), scale_y):
            raise ValueError("All GeoTIFFs must share the same pixel height")
        if not math.isclose(float(cog._geotiff.transform.b), 0.0) or not math.isclose(
            float(cog._geotiff.transform.d),
            0.0,
        ):
            raise NotImplementedError(
                "merge currently requires a north-up (non-rotated) grid"
            )

        # Ensure origins line up on the same pixel grid (integer pixel offsets).
        off_x = (float(cog._geotiff.transform.c) - float(base_transform.c)) / scale_x
        off_y = (float(base_transform.f) - float(cog._geotiff.transform.f)) / scale_y
        if not math.isclose(off_x, round(off_x), abs_tol=1e-6) or not math.isclose(
            off_y, round(off_y), abs_tol=1e-6
        ):
            raise ValueError(
                "All GeoTIFFs must be aligned to the same pixel grid (origins differ by whole pixels)"
            )
