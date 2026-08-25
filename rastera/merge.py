from __future__ import annotations

import math
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

import numpy as np
from affine import Affine
from async_geotiff import RasterArray
from pyproj import CRS

from . import config
from .geo import (
    BBox,
    WindowOutOfRangeError,
    _affine_apply,
    _denoise,
    _is_on_res_grid,
    _normalize_crs,
    compute_paste_slices,
    ensure_bbox,
    normalize_band_indices,
    snapped_grid_for_bbox,
    transform_bbox,
    validate_resolution,
)
from .reader import (
    AsyncGeoTIFF,
    _CrsNodata,
    _grid_for_bbox,
    _make_output_array,
)
from .resampling import ResamplingMethod, validate_resampling


async def merge(
    cogs: Sequence[AsyncGeoTIFF],
    *,
    bbox: BBox | tuple[float, float, float, float],
    bbox_crs: int | CRS,
    band_indices: Sequence[int] | None = None,
    nodata: int | float | None = None,
    target_crs: int | CRS | None = None,
    target_resolution: float,
    mosaic_method: Literal["first", "last"] = "first",
    crs_method: Literal["most_common", "first"] = "most_common",
    snap_to_grid: bool = True,
    use_overviews: bool = False,
    resampling: ResamplingMethod = "nearest",
) -> RasterArray:
    """Merge a bbox that may span multiple GeoTIFFs into one stitched array.

    Args:
        bbox_crs: The bbox is transformed to the COGs' native CRS automatically.
        band_indices: 1-based.
        nodata: The value that means "no data" in the output: merge fills
            every uncovered pixel with it and reports it as ``arr.nodata``,
            so a file written from the result can be tagged with it.
            Defaults to the first input's nodata, like
            ``rasterio.merge.merge``; if that input has none, gaps get 0 and
            ``arr.nodata`` stays ``None``, since 0 is usually real data.
            The inputs are unaffected — each is still read through its own
            nodata — and real pixels can hold this value too, so
            ``RasterArray.mask`` is what actually marks the gaps.
        target_crs: Each COG is reprojected into this CRS before merging when
            it differs from the source. When ``None``, inferred from the
            inputs using *crs_method*.
        mosaic_method: Overlap strategy when multiple COGs cover the same pixel.
            ``"first"`` keeps the first valid pixel (matching rasterio.merge
            default). ``"last"`` lets later COGs overwrite earlier ones.
        crs_method: How to choose the output CRS when *target_crs* is ``None``.
            ``"most_common"`` picks the CRS shared by the most inputs;
            ``"first"`` uses the CRS of the first input.
        snap_to_grid: When True (default), the output grid is rounded
            outward onto multiples of ``target_resolution`` (GDAL's
            ``-tap``), on both merge paths. The output transform and shape
            are then a pure function of the bbox and resolution — never of
            source grid phase, tile count, or tile order — and each edge not
            already on that grid grows by less than one pixel. The inputs
            are copied 1:1 (no resampling) only when *every* one of them is
            already on that grid at the target CRS and resolution; a single
            off-grid input sends them all through ``resampling``. When
            False, the output grid is anchored at the requested
            ``(minx, maxy)`` with width/height rounded to whole pixels (so
            the max edges can drift by <0.5 px), matching rasterio/GDAL
            merge behaviour. Each input is always read on its own pixel
            grid; this flag does not change that.
        use_overviews: Trades accuracy for bandwidth; see
            :meth:`rastera.AsyncGeoTIFF.read` for what overview pixels cost.
        resampling: Used when reprojecting or changing resolution; see
            :meth:`rastera.AsyncGeoTIFF.read` for the per-method trade-offs.
    """
    if not cogs:
        raise ValueError("merge requires at least one AsyncGeoTIFF")

    # Up front, and before any read: a misspelled method silently selected the
    # other branch's semantics, and the grid arguments failed several frames
    # deep with an error naming neither the argument nor this call.
    if mosaic_method not in ("first", "last"):
        raise ValueError(
            f"mosaic_method must be 'first' or 'last', got {mosaic_method!r}"
        )
    if crs_method not in ("most_common", "first"):
        raise ValueError(
            f"crs_method must be 'most_common' or 'first', got {crs_method!r}"
        )
    validate_resampling(resampling)
    validate_resolution(target_resolution)
    # Before any read: an unusable sentinel should fail here rather than
    # several frames deep in np.full.
    fill_value, out_nodata = _resolve_output_nodata(nodata, cogs[0])

    bbox_crs = _normalize_crs(bbox_crs)
    if target_crs is not None:
        target_crs = _normalize_crs(target_crs)

    if target_crs is None:
        target_crs = _resolve_target_crs(cogs, crs_method)

    bbox = ensure_bbox(bbox)
    base = cogs[0]

    # Validate + resolve count; keep original band_indices for cog.read() calls.
    n_out_bands = len(normalize_band_indices(band_indices, base.count))

    # Both paths paste every contributor into one array of cogs[0]'s dtype with
    # n_out_bands rows, so those must line up before either dispatches.
    _require_stackable_bands(cogs, band_indices)

    base_gt = base._geotiff
    all_same_crs = all(cog._crs_epsg == base._crs_epsg for cog in cogs[1:])
    all_same_res = all(
        math.isclose(float(cog._geotiff.transform.a), float(base_gt.transform.a))
        for cog in cogs[1:]
    )
    crs_matches_target = target_crs == base._crs_epsg
    # Default isclose tolerance: the output grid's pixel scale is exactly
    # target_resolution, so a looser match would let the block copy accumulate
    # a whole pixel of paste drift over a large enough offset.
    res_matches_target = math.isclose(target_resolution, base_gt.res[0])

    # The native path is a straight block copy onto the snapped output grid,
    # which sits on multiples of target_resolution — exact only when every
    # source grid is on those multiples too, north-up and square at that
    # resolution (res_matches_target checks the x axis only; a negative -e
    # can never isclose a positive resolution). Anything else is resampled.
    srcs_on_res_grid = math.isclose(
        target_resolution, -float(base_gt.transform.e)
    ) and all(
        _is_on_res_grid(float(cog._geotiff.transform.c), target_resolution)
        and _is_on_res_grid(float(cog._geotiff.transform.f), target_resolution)
        for cog in cogs
    )

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
        or not srcs_on_res_grid
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
            out_nodata=out_nodata,
            target_crs=target_crs,
            target_resolution=target_resolution,
            mosaic_method=mosaic_method,
            snap_to_grid=snap_to_grid,
            use_overviews=use_overviews,
            resampling=resampling,
        )

    # --- Native merge fast path (no resampling needed) ---
    # Reached only when all COGs share the target CRS and resolution AND
    # snap_to_grid is True — every other case routes to _merge_reprojected.
    _require_compatible_merge_inputs(cogs)

    native_crs = base._crs_epsg
    assert native_crs is not None
    native_bbox = transform_bbox(bbox, bbox_crs, native_crs)

    window_transform, win_width, win_height = snapped_grid_for_bbox(
        native_bbox, target_resolution
    )

    sub_bboxes: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        sub_bbox = native_bbox.intersect(BBox(*cog._geotiff.bounds))
        if sub_bbox is not None:
            sub_bboxes.append((cog, sub_bbox))

    async def _read_native_bands(cog: AsyncGeoTIFF, sb: BBox) -> RasterArray:
        indices = normalize_band_indices(band_indices, cog.count)
        # *sb* is clipped to this COG's bounds, so an unsnapped window would
        # drop the tile's last row/column at the seam and let the next COG
        # fill it with its own pixels.
        return await cog._read_native(bbox=sb, band_indices=indices)

    out_data, coverage = await _gather_and_paste(
        contributing=sub_bboxes,
        dst_transform=window_transform,
        dst_width=win_width,
        dst_height=win_height,
        n_bands=n_out_bands,
        dtype=base_gt.dtype,
        fill_value=fill_value,
        read_fn=_read_native_bands,
        mosaic_method=mosaic_method,
    )
    geotiff_ref = _CrsNodata(CRS.from_epsg(native_crs), out_nodata)
    return _make_output_array(
        out_data, window_transform, win_width, win_height, geotiff_ref, mask=coverage
    )


async def _merge_reprojected(
    cogs: Sequence[AsyncGeoTIFF],
    *,
    bbox: BBox,
    bbox_crs: int,
    band_indices: Sequence[int] | None,
    n_out_bands: int,
    fill_value: int | float,
    out_nodata: int | float | None,
    target_crs: int,
    target_resolution: float,
    mosaic_method: Literal["first", "last"] = "first",
    snap_to_grid: bool = True,
    use_overviews: bool = False,
    resampling: ResamplingMethod = "nearest",
) -> RasterArray:
    base = cogs[0]
    base_gt = base._geotiff
    out_crs = target_crs

    target_bbox = transform_bbox(bbox, bbox_crs, out_crs)
    res = target_resolution

    out_transform, out_w, out_h = (
        snapped_grid_for_bbox(target_bbox, res)
        if snap_to_grid
        else _grid_for_bbox(target_bbox, res)
    )

    # Find contributing COGs by intersecting bounds (in target CRS) with output
    # bbox.
    contributing: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        assert cog._crs_epsg is not None
        sub_bbox = target_bbox.intersect(
            transform_bbox(BBox(*cog._geotiff.bounds), cog._crs_epsg, out_crs)
        )
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

        return await cog._read_to_grid(
            dst_transform=sub_transform,
            dst_width=sub_w,
            dst_height=sub_h,
            out_crs=out_crs,
            band_indices=normalize_band_indices(band_indices, cog.count),
            resampling=resampling,
            use_overviews=use_overviews,
        )

    out_data, coverage = await _gather_and_paste(
        contributing=contributing,
        dst_transform=out_transform,
        dst_width=out_w,
        dst_height=out_h,
        n_bands=n_out_bands,
        dtype=base_gt.dtype,
        fill_value=fill_value,
        read_fn=_read_and_reproject,
        mosaic_method=mosaic_method,
    )

    geotiff_ref = _CrsNodata(CRS.from_epsg(out_crs), out_nodata)
    return _make_output_array(
        out_data, out_transform, out_w, out_h, geotiff_ref, mask=coverage
    )


async def _gather_and_paste(
    *,
    contributing: list[tuple[AsyncGeoTIFF, BBox]],
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    n_bands: int,
    dtype: np.dtype[Any] | None,
    fill_value: int | float,
    read_fn: Callable[[AsyncGeoTIFF, BBox], Awaitable[RasterArray]],
    mosaic_method: Literal["first", "last"] = "first",
) -> tuple[np.ndarray, np.ndarray]:
    """Read contributing COGs and paste into a single output array.

    Returns the array and the ``(dst_height, dst_width)`` coverage it was
    pasted under: True where an output pixel holds a real source value, False
    where it is still *fill_value* because no contributor reached it or every
    one that did was its own nodata there.

    Results are pasted in input order, each masked by its own nodata. Overlap
    is resolved by ``mosaic_method``: ``"first"`` keeps the first valid pixel,
    ``"last"`` lets later COGs overwrite earlier ones.

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

    # Allocated for both methods, and returned even when it ends up all True:
    # a caller handed mask=None falls back to masked_equal(data, nodata), which
    # cannot tell a gap from a real pixel that happens to hold the same value —
    # a contributor declaring no sentinel of its own is full of them.
    filled = np.zeros((dst_height, dst_width), dtype=bool)

    if not contributing:
        return out_array, filled

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

            # Geometry first: which destination pixels this COG reaches. A
            # source that declares no sentinel still has an edge, and without
            # this the warp's edge-clamped pixels paste as data and mark the
            # ground filled, locking out the neighbour that covers it.
            src_valid = None if arr.mask is None else arr.mask[src_rows, src_cols]

            # Then each contributor's own sentinel: using cogs[0]'s would paste
            # another COG's nodata as real data.
            cog_nodata = cog._nodata
            if cog_nodata is not None:
                if isinstance(cog_nodata, float) and math.isnan(cog_nodata):
                    valid = ~np.isnan(src_data)
                else:
                    valid = src_data != cog_nodata
                sentinel_valid = np.any(valid, axis=0)
                src_valid = (
                    sentinel_valid if src_valid is None else src_valid & sentinel_valid
                )

            if mosaic_method == "first":
                unfilled = ~filled[dst_rows, dst_cols]
                if src_valid is not None:
                    paste_mask = unfilled & src_valid
                else:
                    paste_mask = unfilled
                np.copyto(out_array[:, dst_rows, dst_cols], src_data, where=paste_mask)
                filled[dst_rows, dst_cols] |= paste_mask
            else:
                # ``|=``, not ``=``: "last" overwrites pixels by design, but an
                # earlier contributor's coverage still stands where this one is
                # invalid. Accumulating never gates what "last" pastes.
                if src_valid is not None:
                    np.copyto(
                        out_array[:, dst_rows, dst_cols], src_data, where=src_valid
                    )
                    filled[dst_rows, dst_cols] |= src_valid
                else:
                    out_array[:, dst_rows, dst_cols] = src_data
                    filled[dst_rows, dst_cols] = True

        if mosaic_method == "first" and filled.all():
            return out_array, filled

    return out_array, filled


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

    # A bare ceil would buy a spurious row/column off ~transform's ULP error;
    # with the output grid on resolution multiples, contributor bounds landing
    # exactly on a grid line are the common case, not the exception.
    col_min = max(0, math.floor(_denoise(min(c0, c1))))
    row_min = max(0, math.floor(_denoise(min(r0, r1))))
    col_max = min(out_w, math.ceil(_denoise(max(c0, c1))))
    row_max = min(out_h, math.ceil(_denoise(max(r0, r1))))

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


def _resolve_output_nodata(
    nodata: int | float | None, base: AsyncGeoTIFF
) -> tuple[int | float, int | float | None]:
    """The value the gaps hold, and the sentinel the output reports.

    One value in two roles, so a file written from the result can be tagged
    with what its own pixels use. Taken from *nodata*, else from the first
    input's own declaration — ``base._nodata``, not ``base._geotiff.nodata``:
    the two differ for a VRT, whose declared <NoDataValue> overrides its
    source's, and the compositing already keys off the former. With neither,
    the gaps hold 0 but the output reports nothing, because 0 is a real
    measurement in most rasters and claiming it would blank them.

    An inherited sentinel needs no validation: ``_coerce_nodata`` already
    dropped it if this dtype could not carry it.
    """
    if nodata is None:
        return (0, None) if base._nodata is None else (base._nodata, base._nodata)
    _validate_nodata(nodata, base._geotiff.dtype)
    return nodata, nodata


def _validate_nodata(nodata: int | float, dtype: np.dtype[Any] | None) -> None:
    """Reject a sentinel the output dtype cannot carry.

    ``np.full`` is inconsistent about these: out-of-range integers raise a bare
    ``OverflowError`` naming neither the argument nor the dtype, while a
    fractional or NaN fill is truncated to something the caller never asked for
    (0.5 and NaN both land on 0 in an integer mosaic).
    """
    if dtype is None:
        return
    if not isinstance(nodata, int | float | np.number) or isinstance(nodata, bool):
        raise ValueError(f"nodata must be a number, got {nodata!r}")
    if dtype.kind not in ("i", "u", "b"):
        return
    if math.isnan(nodata) or math.isinf(nodata):
        raise ValueError(
            f"nodata={nodata!r} cannot be represented in {dtype}; "
            f"pass a finite integer sentinel"
        )
    if nodata != int(nodata):
        raise ValueError(
            f"nodata={nodata!r} is not an integer and would be "
            f"truncated in a {dtype} mosaic"
        )
    if dtype.kind == "b":
        return  # np.iinfo has no bool entry, and there is no range to check
    info = np.iinfo(dtype)
    if not info.min <= nodata <= info.max:
        raise ValueError(
            f"nodata={nodata!r} is outside the range of {dtype} "
            f"[{info.min}, {info.max}]"
        )


def _require_stackable_bands(
    cogs: Sequence[AsyncGeoTIFF], band_indices: Sequence[int] | None
) -> None:
    """Validate dtype and band availability across inputs.

    Applies to both merge paths: without this, a mismatched dtype surfaces as an
    opaque ``TypeError`` from ``np.copyto`` (or truncates silently when the cast
    is narrowing), and a missing band as a broadcast error.

    With explicit *band_indices* the counts need not agree — each read resolves
    the indices against its own COG, so only the requested bands have to exist.
    """
    base = cogs[0]
    base_dtype = base._geotiff.dtype
    highest = max(band_indices) if band_indices else None
    for cog in cogs[1:]:
        if cog._geotiff.dtype != base_dtype:
            raise ValueError(
                f"All GeoTIFFs must share the same dtype; {base.uri!r} is "
                f"{base_dtype} but {cog.uri!r} is {cog._geotiff.dtype}"
            )
        if highest is None:
            if cog.count != base.count:
                raise ValueError(
                    f"All GeoTIFFs must share the same band count; {base.uri!r} "
                    f"has {base.count} but {cog.uri!r} has {cog.count}"
                )
        elif cog.count < highest:
            raise ValueError(
                f"All GeoTIFFs must carry the requested bands; band_indices "
                f"asks for band {highest} but {cog.uri!r} has {cog.count}"
            )


def _require_compatible_merge_inputs(cogs: Sequence[AsyncGeoTIFF]) -> None:
    """
    Validate that all inputs can be pasted onto a single shared pixel grid.

    Native-path only: assumes a north-up, non-rotated Affine grid (b=d=0) and
    that all sources are aligned to it (origins differ by whole pixels).
    Cross-input dtype/band-count checks live in :func:`_require_stackable_bands`.
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
