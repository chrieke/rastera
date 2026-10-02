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
    _grid_bounds,
    _is_on_res_grid,
    _normalize_crs,
    bounds_from_transform,
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
    _require_epsg,
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
        nodata: Fills the uncovered pixels, and is reported as ``arr.nodata``.
            Defaults to the first input's, as in ``rasterio.merge.merge``;
            without one, gaps get 0 and ``arr.nodata`` is None. Each input is
            still read through its own nodata. Real pixels can hold this value
            too, so ``RasterArray.mask`` is what marks the gaps.
        target_crs: Each COG is reprojected into this CRS before merging when
            it differs from the source. When ``None``, inferred from the
            inputs using *crs_method*.
        mosaic_method: ``"first"`` keeps the first valid value in each band, as
            ``rasterio.merge`` does; ``"last"`` lets later inputs overwrite.
        crs_method: How to choose the output CRS when *target_crs* is ``None``.
            ``"most_common"`` picks the CRS shared by the most inputs;
            ``"first"`` uses the CRS of the first input.
        snap_to_grid: Round the output grid outward onto multiples of
            ``target_resolution`` (GDAL's ``-tap``), so it depends only on the
            bbox and resolution. Inputs are copied without resampling only
            when every one is on that grid. When False, the grid is anchored
            at the bbox's ``(minx, maxy)`` with rounded pixel counts, as
            ``rasterio.merge`` does.
        use_overviews: Trades accuracy for bandwidth; see
            :meth:`rastera.AsyncGeoTIFF.read` for what overview pixels cost.
        resampling: Used when reprojecting or changing resolution; see
            :meth:`rastera.AsyncGeoTIFF.read` for the per-method trade-offs.
    """
    if not cogs:
        raise ValueError("merge requires at least one AsyncGeoTIFF")

    # Before any read, and naming the argument.
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
    for cog in cogs:
        _require_epsg(cog)  # every input is placed by its EPSG code
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
    # which sits on multiples of target_resolution. That is exact only when
    # every input, not just the first, is on those multiples too: north-up,
    # running east, and square at that resolution (a negative -e can never
    # isclose a positive resolution). With the CRS and resolution checks
    # above, that is all the block copy needs. Anything else is resampled.
    srcs_on_res_grid = all(
        math.isclose(target_resolution, -float(t.e))
        and float(t.a) > 0
        and float(t.b) == 0
        and float(t.d) == 0
        and _is_on_res_grid(float(t.c), target_resolution)
        and _is_on_res_grid(float(t.f), target_resolution)
        for t in (cog._geotiff.transform for cog in cogs)
    )

    # Not use_overviews: the native path reads at the native resolution, which
    # no overview serves.
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
    native_crs = base._crs_epsg
    assert native_crs is not None
    native_bbox = transform_bbox(bbox, bbox_crs, native_crs)

    window_transform, win_width, win_height = snapped_grid_for_bbox(
        native_bbox, target_resolution
    )

    sub_bboxes: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        sub_bbox = native_bbox.intersect(_grid_bounds(cog._geotiff))
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

    # Against the grid, not the bbox: snapping grows the grid by up to a pixel,
    # and a source covering only that margin was left out, leaving it empty.
    grid_bbox = bounds_from_transform(out_transform, out_w, out_h)
    contributing: list[tuple[AsyncGeoTIFF, BBox]] = []
    for cog in cogs:
        sub_bbox = _covered_bbox(cog, grid_bbox, out_crs)
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
    """Read the contributors and paste them, in input order and each masked
    by its own nodata per band, into one array.

    Returns the array and its ``(dst_height, dst_width)`` coverage: True where
    some band holds a real source value. Reads run in batches of
    ``set_concurrency(merge=N)``.
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
    # Per band, for "first" only, and only once a contributor's sentinel covers
    # some bands of a pixel but not others. Until then every band of a pixel is
    # filled together and ``filled`` says which.
    band_filled: np.ndarray | None = None

    if not contributing:
        return out_array, filled

    n = config._merge_concurrency
    for i in range(0, len(contributing), n):
        batch = contributing[i : i + n]
        if mosaic_method == "first":
            # "first" pastes only where nothing is yet, so a contributor whose
            # cells are all filled, in every band, would be read for nothing.
            taken = filled if band_filled is None else band_filled
            batch = [
                (cog, sub)
                for cog, sub in batch
                if not _cells_filled(taken, dst_transform, sub)
            ]
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

            # Then each contributor's own sentinel, per band like gdalwarp and
            # rasterio.merge: using cogs[0]'s would paste another COG's nodata
            # as real data, and one band's sentinel would hide the next COG's
            # real value in that band.
            band_valid = None
            cog_nodata = cog._nodata
            if cog_nodata is not None:
                if isinstance(cog_nodata, float) and math.isnan(cog_nodata):
                    valid = ~np.isnan(src_data)
                else:
                    valid = src_data != cog_nodata
                if src_valid is not None:
                    valid &= src_valid
                src_valid = valid.any(axis=0)
                if not np.array_equal(src_valid, valid.all(axis=0)):
                    band_valid = valid
            paste_where = band_valid if band_valid is not None else src_valid

            if mosaic_method == "first":
                if band_valid is not None and band_filled is None:
                    band_filled = np.repeat(filled[np.newaxis], n_bands, axis=0)
                taken = filled if band_filled is None else band_filled
                paste_mask = ~taken[..., dst_rows, dst_cols]
                if paste_where is not None:
                    paste_mask &= paste_where
                np.copyto(out_array[:, dst_rows, dst_cols], src_data, where=paste_mask)
                taken[..., dst_rows, dst_cols] |= paste_mask
                if band_filled is not None:
                    filled[dst_rows, dst_cols] |= paste_mask.any(axis=0)
            else:
                # ``|=``, not ``=``: "last" overwrites pixels by design, but an
                # earlier contributor's coverage still stands where this one is
                # invalid. Accumulating never gates what "last" pastes.
                if paste_where is not None:
                    np.copyto(
                        out_array[:, dst_rows, dst_cols], src_data, where=paste_where
                    )
                    filled[dst_rows, dst_cols] |= src_valid
                else:
                    out_array[:, dst_rows, dst_cols] = src_data
                    filled[dst_rows, dst_cols] = True

        if (
            mosaic_method == "first"
            and (filled if band_filled is None else band_filled).all()
        ):
            return out_array, filled

    return out_array, filled


def _output_subgrid(
    out_transform: Affine, out_w: int, out_h: int, sub_bbox: BBox
) -> tuple[Affine, int, int] | None:
    """The whole-pixel window of the output grid *sub_bbox* reaches, as
    ``(transform, width, height)``, or None."""
    cells = _output_cells(out_transform, out_w, out_h, sub_bbox)
    if cells is None:
        return None
    rows, cols = cells
    res = out_transform.a
    sub_transform = Affine(
        res,
        0,
        out_transform.c + cols.start * res,
        0,
        -res,
        out_transform.f - rows.start * res,
    )
    return sub_transform, cols.stop - cols.start, rows.stop - rows.start


def _output_cells(
    out_transform: Affine, out_w: int, out_h: int, sub_bbox: BBox
) -> tuple[slice, slice] | None:
    """The rows and columns of the output grid *sub_bbox* reaches, or None."""
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

    if col_max <= col_min or row_max <= row_min:
        return None
    return slice(row_min, row_max), slice(col_min, col_max)


def _cells_filled(filled: np.ndarray, dst_transform: Affine, sub_bbox: BBox) -> bool:
    """Whether every output cell *sub_bbox* reaches is filled already.

    *filled* is ``(rows, cols)``, or ``(bands, rows, cols)`` once bands fill
    apart.
    """
    h, w = filled.shape[-2:]
    cells = _output_cells(dst_transform, w, h, sub_bbox)
    return cells is not None and bool(filled[(..., *cells)].all())


def _resolve_output_nodata(
    nodata: int | float | None, base: AsyncGeoTIFF
) -> tuple[int | float, int | float | None]:
    """The value the gaps hold, and the sentinel the output reports: *nodata*,
    else the first input's resolved one (a VRT's ``<NoDataValue>``, not its
    source's). With neither, gaps hold 0 and the output reports none, since 0
    is real data in most rasters.
    """
    if nodata is None:
        return (0, None) if base._nodata is None else (base._nodata, base._nodata)
    _validate_nodata(nodata, base._geotiff.dtype)
    return nodata, nodata


def _validate_nodata(nodata: int | float, dtype: np.dtype[Any] | None) -> None:
    """Reject a sentinel the output dtype cannot carry, which ``np.full``
    would raise on unhelpfully or truncate (0.5 and NaN both to 0)."""
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
    """Require one dtype across the inputs, and the requested bands in each,
    or one band count when *band_indices* is None. A mismatch otherwise
    surfaces deep in ``np.copyto``, or truncates silently."""
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


def _covered_bbox(cog: AsyncGeoTIFF, target_bbox: BBox, out_crs: int) -> BBox | None:
    """The part of *target_bbox* (in *out_crs*) that *cog* covers.

    Clipped in the source CRS first, the direction ``read()`` transforms in.
    Transforming the source's whole extent instead under-reports a wide one (a
    continental EPSG:4326 source loses ~1 km at its edge in UTM) and fails
    outright for a global one. The whole extent is still the fallback when
    either transform fails: an output bbox the source CRS cannot hold (a
    world-wide mosaic of UTM tiles), or a clip off a polar source that crosses
    the antimeridian on its way back to EPSG:4326.
    """
    src_crs = cog._crs_epsg
    assert src_crs is not None
    src_bounds = _grid_bounds(cog._geotiff)
    try:
        in_src = transform_bbox(target_bbox, out_crs, src_crs)
        # A wide output bbox comes back short in the source CRS too, by up to
        # ~0.5% of its size, which would cut a tile on its edge. The margin is
        # free: the result is cut back to target_bbox below.
        dx, dy = in_src.width * 0.05, in_src.height * 0.05
        clipped = BBox(
            in_src.minx - dx, in_src.miny - dy, in_src.maxx + dx, in_src.maxy + dy
        ).intersect(src_bounds)
        if clipped is None:
            return None
        covered = transform_bbox(clipped, src_crs, out_crs)
    except ValueError:
        covered = transform_bbox(src_bounds, src_crs, out_crs)
    return target_bbox.intersect(covered)


async def _read_or_skip(
    read_fn: Callable[[AsyncGeoTIFF, BBox], Awaitable[RasterArray]],
    cog: AsyncGeoTIFF,
    sub_bbox: BBox,
) -> RasterArray | None:
    """*read_fn*, or None for a sub-pixel sliver that rounds to no window."""
    try:
        return await read_fn(cog, sub_bbox)
    except WindowOutOfRangeError:
        return None
