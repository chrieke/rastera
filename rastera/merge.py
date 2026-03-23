from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

import numpy as np
from affine import Affine

from .reader import AsyncGeoTIFF
from .geo import (
    BBox,
    _affine_apply,
    bounds_from_transform,
    compute_paste_slices,
    ensure_bbox,
    normalize_band_indices,
    transform_bbox,
)
from .meta import Profile

DEFAULT_CONCURRENCY = 4

# Per-COG tile fetch batch size during merges. Limits concurrent HTTP requests
# per COG to prevent connection failures when multiple COGs are read in parallel.
# Empirically determined: without batching, max_concurrency >= 4 fails on
# high-resolution COGs (~150+ tiles each).
DEFAULT_MERGE_BATCH_SIZE = 48


async def merge_cogs(
    cogs: Sequence[AsyncGeoTIFF],
    *,
    bbox: BBox | tuple[float, float, float, float],
    bbox_crs: int | None = None,
    band_indices: Sequence[int] | None = None,
    fill_value: int | float = 0,
    target_crs: int | None = None,
    target_resolution: float | None = None,
    max_concurrency: int = DEFAULT_CONCURRENCY,
    tile_batch_size: int = DEFAULT_MERGE_BATCH_SIZE,
    method: Literal["first", "last"] = "first",
) -> tuple[np.ndarray, Profile]:
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
        max_concurrency: Maximum number of COGs to read concurrently.
            Limits peak memory to ~max_concurrency tile arrays.
        method: Overlap strategy when multiple COGs cover the same pixel.
            ``"first"`` keeps the first valid pixel (matching rasterio.merge
            default). ``"last"`` lets later COGs overwrite earlier ones.

    Returns:
        Tuple of numpy array and window metadata
    """
    if not cogs:
        raise ValueError("merge requires at least one AsyncGeoTIFF")

    if bbox_crs is None:
        raise ValueError("bbox_crs is required when bbox is provided")

    bbox = ensure_bbox(bbox)
    base = cogs[0]

    # Validate + resolve count; keep original band_indices for cog.read() calls.
    n_out_bands = len(normalize_band_indices(band_indices, base.ifd.samples_per_pixel))

    # Decide whether we need the reprojected merge path.
    all_same_crs = all(
        cog.profile.crs_epsg == base.profile.crs_epsg for cog in cogs[1:]
    )
    all_same_res = all(
        math.isclose(float(cog.profile.transform.a), float(base.profile.transform.a))
        for cog in cogs[1:]
    )
    crs_matches_target = target_crs is None or target_crs == base.profile.crs_epsg
    res_matches_target = target_resolution is None or math.isclose(
        target_resolution, base.profile.res[0], rel_tol=1e-6
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
            max_concurrency=max_concurrency,
            tile_batch_size=tile_batch_size,
            method=method,
        )

    # --- Path A: native merge (existing fast path) ---
    # The output grid is snapped to the source COG's pixel grid so that
    # pixels are copied 1:1 without resampling.  This differs from
    # rasterio.merge / WarpedVRT which place the origin at the exact
    # bbox corner, introducing a sub-pixel shift and nearest-neighbour
    # resampling even when CRS and resolution already match.
    _require_compatible_merge_inputs(cogs)

    native_crs = base.profile.crs_epsg
    native_bbox = transform_bbox(bbox, bbox_crs, native_crs)

    # Get profile for bbox image mosaic
    window_transform, win_width, win_height, out_bounds = _mosaic_grid_from_bbox(
        base_transform=base.profile.transform,
        bbox=native_bbox,
    )
    out_profile = Profile(
        width=win_width,
        height=win_height,
        count=base.profile.count,
        dtype=base.profile.dtype,
        transform=window_transform,
        res=base.profile.res,
        crs_epsg=base.profile.crs_epsg,
        bounds=out_bounds,
        nodata=base.profile.nodata,
        tile_width=base.profile.tile_width,
        tile_height=base.profile.tile_height,
    )

    # Get sub bboxes specific to the contributing image
    sub_bboxes: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        sub_bbox = native_bbox.intersect(cog.profile.bounds)
        if sub_bbox is not None:
            sub_bboxes.append((cog, sub_bbox))

    async def _read_native_bands(
        cog: AsyncGeoTIFF, sb: BBox
    ) -> tuple[np.ndarray, Profile]:
        indices = normalize_band_indices(band_indices, cog.ifd.samples_per_pixel)
        return await cog._read_native(
            bbox=sb, band_indices=indices, _tile_batch_size=tile_batch_size
        )

    out_array = await _gather_and_paste(
        contributing=sub_bboxes,
        out_profile=out_profile,
        n_bands=n_out_bands,
        fill_value=fill_value,
        read_fn=_read_native_bands,
        max_concurrency=max_concurrency,
        method=method,
    )
    return out_array, out_profile


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
    max_concurrency: int = DEFAULT_CONCURRENCY,
    tile_batch_size: int = DEFAULT_MERGE_BATCH_SIZE,
    method: Literal["first", "last"] = "first",
) -> tuple[np.ndarray, Profile]:
    """Path B: merge with reprojection — supports mixed-CRS inputs."""
    base = cogs[0]
    out_crs = target_crs if target_crs is not None else base.profile.crs_epsg

    # Transform bbox into the output CRS
    target_bbox = transform_bbox(bbox, bbox_crs, out_crs)

    # Determine output resolution
    if target_resolution is not None:
        res = target_resolution
    else:
        # Preserve native pixel density of the first COG
        native_res = base.profile.res[0]
        src_bbox = transform_bbox(target_bbox, out_crs, base.profile.crs_epsg)
        n_cols = max(1, round(src_bbox.width / native_res))
        n_rows = max(1, round(src_bbox.height / native_res))
        res_x = target_bbox.width / n_cols
        res_y = target_bbox.height / n_rows
        res = min(res_x, res_y)

    # Build output grid
    out_profile = Profile.for_bbox(
        target_bbox,
        res,
        out_crs,
        count=n_out_bands,
        dtype=base.profile.dtype,
        nodata=base.profile.nodata,
        tile_width=base.profile.tile_width,
        tile_height=base.profile.tile_height,
    )

    # Find contributing COGs by intersecting their bounds (in target CRS) with output bbox
    contributing: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        cog_bounds_in_target = transform_bbox(
            cog.profile.bounds, cog.profile.crs_epsg, out_crs
        )
        sub_bbox = target_bbox.intersect(cog_bounds_in_target)
        if sub_bbox is not None:
            contributing.append((cog, sub_bbox))

    async def _read_and_unwrap(
        cog: AsyncGeoTIFF, sb: BBox
    ) -> tuple[np.ndarray, Profile]:
        return await cog.read(
            bbox=sb,
            bbox_crs=out_crs,
            target_crs=out_crs,
            target_resolution=res,
            band_indices=band_indices,
            _tile_batch_size=tile_batch_size,
        )

    out_array = await _gather_and_paste(
        contributing=contributing,
        out_profile=out_profile,
        n_bands=n_out_bands,
        fill_value=fill_value,
        read_fn=_read_and_unwrap,
        max_concurrency=max_concurrency,
        method=method,
    )
    return out_array, out_profile


async def _gather_and_paste(
    *,
    contributing: list[tuple[AsyncGeoTIFF, BBox]],
    out_profile: Profile,
    n_bands: int,
    fill_value: int | float,
    read_fn: Callable[[AsyncGeoTIFF, BBox], Awaitable[tuple[np.ndarray, Profile]]],
    max_concurrency: int = DEFAULT_CONCURRENCY,
    method: Literal["first", "last"] = "first",
) -> np.ndarray:
    """Read contributing COGs concurrently and paste into a single output array.

    Uses a semaphore to keep at most ``max_concurrency`` reads in flight.
    Unlike chunked processing, a new read starts as soon as any previous
    read finishes — no waiting for a whole batch to complete.

    Results are pasted in input order. Overlap is resolved by ``method``:
    ``"first"`` keeps the first valid pixel, ``"last"`` lets later COGs
    overwrite earlier ones.
    """
    out_array = np.full(
        (n_bands, out_profile.height, out_profile.width),
        fill_value,
        dtype=out_profile.dtype,
    )

    if not contributing:
        return out_array

    # For "first" semantics we track which pixels have been filled so
    # later COGs don't overwrite them.  One bool per pixel — negligible
    # compared to the output array itself.
    # TODO: With "first" semantics, we could skip reading COGs whose bbox
    # is fully covered by already-pasted data. Would require lazy task
    # launching (instead of all-upfront) so reads aren't already in-flight
    # when we discover the output is full. Only helps the temporal overlap
    # case (same tile, many dates); spatial mosaics rarely overlap.
    filled = (
        np.zeros((out_profile.height, out_profile.width), dtype=bool)
        if method == "first"
        else None
    )

    sem = asyncio.Semaphore(max_concurrency)

    async def _read(idx: int, cog: AsyncGeoTIFF, sub_bbox: BBox):
        async with sem:
            result = await read_fn(cog, sub_bbox)
        return idx, result

    tasks = [
        asyncio.create_task(_read(i, cog, sb))
        for i, (cog, sb) in enumerate(contributing)
    ]

    pending: dict[int, tuple[np.ndarray, Profile]] = {}
    next_idx = 0

    for coro in asyncio.as_completed(tasks):
        idx, (sub_arr, sub_profile) = await coro
        pending[idx] = (sub_arr, sub_profile)

        # Drain the pending buffer in input order so overlap semantics
        # are deterministic and earlier results get freed from memory.
        while next_idx in pending:
            arr, prof = pending.pop(next_idx)
            next_idx += 1
            slices = compute_paste_slices(
                src_profile=prof,
                dst_transform=out_profile.transform,
                dst_width=out_profile.width,
                dst_height=out_profile.height,
            )
            if slices is None:
                continue
            dst_rows, dst_cols, src_rows, src_cols = slices
            src_data = arr[:, src_rows, src_cols]
            nodata = prof.nodata

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
                out_array[:, dst_rows, dst_cols] = np.where(
                    paste_mask, src_data, out_array[:, dst_rows, dst_cols]
                )
                filled[dst_rows, dst_cols] |= paste_mask
            else:
                # "last" — later COGs overwrite earlier ones
                if src_valid is not None:
                    out_array[:, dst_rows, dst_cols] = np.where(
                        src_valid, src_data, out_array[:, dst_rows, dst_cols]
                    )
                else:
                    out_array[:, dst_rows, dst_cols] = src_data

    return out_array


def _mosaic_grid_from_bbox(
    *, base_transform: Affine, bbox: BBox
) -> tuple[Affine, int, int, BBox]:
    """
    Create a pixel-aligned mosaic grid for `bbox` on the `base_transform` grid.

    Unlike `Window.from_bbox`, this does NOT clamp to any particular image size.
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
    base_transform = base.profile.transform
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
        if cog.profile.crs_epsg != base.profile.crs_epsg:
            raise ValueError("All GeoTIFFs must share the same CRS EPSG")
        if cog.ifd.samples_per_pixel != base.ifd.samples_per_pixel:
            raise ValueError("All GeoTIFFs must share the same samples_per_pixel")
        if not math.isclose(float(cog.profile.transform.a), scale_x):
            raise ValueError("All GeoTIFFs must share the same pixel width")
        if not math.isclose(float(-cog.profile.transform.e), scale_y):
            raise ValueError("All GeoTIFFs must share the same pixel height")
        if not math.isclose(float(cog.profile.transform.b), 0.0) or not math.isclose(
            float(cog.profile.transform.d),
            0.0,
        ):
            raise NotImplementedError(
                "merge currently requires a north-up (non-rotated) grid"
            )

        # Ensure origins line up on the same pixel grid (integer pixel offsets).
        off_x = (float(cog.profile.transform.c) - float(base_transform.c)) / scale_x
        off_y = (float(base_transform.f) - float(cog.profile.transform.f)) / scale_y
        if not math.isclose(off_x, round(off_x), abs_tol=1e-6) or not math.isclose(
            off_y, round(off_y), abs_tol=1e-6
        ):
            raise ValueError(
                "All GeoTIFFs must be aligned to the same pixel grid (origins differ by whole pixels)"
            )
