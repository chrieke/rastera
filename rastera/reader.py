from __future__ import annotations

import asyncio
import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any, Protocol, TypedDict, cast, overload

import numpy as np
from affine import Affine
from async_geotiff import GeoTIFF, Overview, RasterArray, Window
from pyproj import CRS, Transformer

from .config import _gather_bounded
from .geo import (
    BBox,
    Picks,
    WindowOutOfRangeError,
    _affine_apply,
    _denoise,
    _grid_bounds,
    _is_on_res_grid,
    _mercator_clamped_bounds,
    _normalize_crs,
    _require_north_up,
    _transformer,
    bounds_from_transform,
    ensure_bbox,
    normalize_band_indices,
    snapped_grid_for_bbox,
    transform_bbox,
    unsnapped_window,
    validate_resolution,
    window_from_bbox,
)
from .predictor import read_window
from .profile import RasterProfile, _build_profile
from .resampling import (
    ResamplingMethod,
    _fill_uncovered,
    _kernel_halo,
    _resample_impl,
    validate_resampling,
)
from .store import (
    _build_store,
    _extract_key,
    _is_local_descriptor,
    _parse_uri,
    _require_same_bucket,
    _resolve_local_path,
    _shared_store,
)

# LRU cache for parsed GeoTIFF objects, keyed by URI (see ``_cache_key``).
# Avoids re-fetching headers on repeated opens of the same file.
_CacheKey = str | tuple[str, int, int, int, int]
_geotiff_cache: OrderedDict[_CacheKey, GeoTIFF] = OrderedDict()
_cache_max_size: int = 128


class AsyncGeoTIFF:
    """AsyncGeoTIFF instance for a single GeoTIFF file.

    Wraps ``async_geotiff.GeoTIFF`` with bbox-based reading, reprojection,
    resampling, and overview selection.
    """

    def __init__(
        self,
        uri: str,
        geotiff: _GeoTIFFLike,
        *,
        meta_overrides: MetaOverrides | None = None,
    ):
        self.uri = uri
        self._geotiff = geotiff
        resolved = _resolve_meta_overrides(meta_overrides)
        # Kept apart from ``_crs_epsg``, which conflates "the caller declared
        # this" with "the file resolved to this". ``_resolved_crs`` needs the
        # difference: an override is the one case where reading ``geotiff.crs``
        # is both lossy and liable to raise.
        self._crs_override: int | None = resolved.get("crs")
        self._crs_epsg: int | None = (
            self._crs_override
            if self._crs_override is not None
            else geotiff.crs.to_epsg()
        )
        # The tag's text, not ``geotiff.nodata``'s float() of it (see
        # ``_coerce_nodata``). ``getattr``: only a real ``GeoTIFF`` has ``ifd``.
        ifd = getattr(geotiff, "ifd", None)
        declared = ifd.gdal_nodata if ifd is not None else geotiff.nodata
        self._nodata: int | float | None = _coerce_nodata(declared, geotiff.dtype)

        self.overviews: list[tuple[int, int]] = [
            (o.width, o.height) for o in geotiff.overviews
        ]

    def _override_nodata(self, nodata: float) -> None:
        """Replace the sentinel this dataset's *pixels* use with *nodata*.

        A value this dataset's dtype cannot carry is ignored rather than
        treated as "no nodata", which would discard a sentinel the file does
        declare. Subclasses that wrap other datasets override this to push the
        value down to them (see ``_VRTDataset``).
        """
        coerced = _coerce_nodata(nodata, self._geotiff.dtype)
        if coerced is not None:
            self._nodata = coerced

    def _internal_mask_uri(self) -> str | None:
        """The URI of the file behind this dataset with an internal mask, if any.

        ``getattr``: only a real ``GeoTIFF`` header has ``mask_ifd``. Datasets
        over other files override this to ask them.
        """
        if getattr(self._geotiff, "mask_ifd", None) is not None:
            return self.uri
        return None

    @property
    def count(self) -> int:
        return self._geotiff.count

    @property
    def profile(self) -> RasterProfile:
        """Everything this dataset's header says about it, as one dict.

        See ``RasterProfile`` for the keys and their caveats.
        """
        return _build_profile(self)

    @property
    def _resolved_crs(self) -> CRS:
        """``profile["crs"]``: the file's own CRS, not one rebuilt from its
        EPSG code, since ``to_epsg()`` matches at 70% confidence and would
        relabel a near-miss WKT. An override wins and is read without touching
        the file's — geo keys that don't parse are what it exists for."""
        if self._crs_override is not None:
            return CRS.from_epsg(self._crs_override)
        return self._geotiff.crs

    def _best_overview_for_resolution(self, dst_res: tuple[float, float]):
        """The coarsest overview no coarser than *dst_res* on either axis, or
        None to read full resolution.

        A level counts at its nominal factor, ``round(width / o.width)``. GDAL's
        COG driver sizes levels by floor division, so an odd-sized file's level
        is a hair coarser than its factor, 20.009 m for 2301 px at 10 m, and an
        exact comparison skipped it at the very targets overviews serve.

        Reads the pyramid off ``_geotiff``, not ``self.overviews``: this needs
        readable ``Overview`` objects, while ``self.overviews`` holds (width,
        height) pairs and is emptied by both VRT flavours.
        """
        gt = self._geotiff
        # The largest factor each axis allows; the slack absorbs float noise in
        # an exact request such as 20 m from 10 m.
        max_fx = dst_res[0] / gt.res[0] * (1 + 1e-9)
        max_fy = dst_res[1] / gt.res[1] * (1 + 1e-9)
        best, best_factor = None, 0
        for o in gt.overviews:
            fx = round(gt.width / o.width)
            fy = round(gt.height / o.height)
            if fx <= max_fx and fy <= max_fy and fx * fy > best_factor:
                best, best_factor = o, fx * fy
        return best

    @classmethod
    async def open(
        cls,
        uri: str,
        *,
        store: Any = None,
        prefetch: int = 32768,
        cache: bool = True,
        meta_overrides: MetaOverrides | None = None,
        **store_kwargs: Any,
    ) -> AsyncGeoTIFF:
        """Open a GeoTIFF from a URI.

        Supports s3://, https://, gs://, az://, and local file paths.

        Args:
            uri: Any URI supported by object_store
                (s3://, https://, gs://, file://, etc.).
            store: Optional pre-constructed store. When provided,
                the key is extracted from the URI and used as the
                path within the store. If no store is provided, one is
                built for the URI's bucket or host: obstore's for a local
                path, async-tiff's otherwise.
            prefetch: Number of bytes to prefetch when opening the TIFF.
            cache: When True, cache the parsed GeoTIFF object in memory so that
                subsequent opens of the same URI skip the header fetch. A
                local file rewritten since is read afresh; a remote object is
                assumed unchanged until :func:`rastera.clear_cache`. A remote
                URI is cached with the store it was first opened through, so
                a later open with another store or other credentials reads
                through that first one; pass ``cache=False`` where it matters.
            meta_overrides: Optional header overrides applied at construction.
                Currently supports ``{"crs": int | CRS}`` for TIFFs missing
                or carrying incorrect georeferencing. Overrides always
                replace the file's reported value.
            **store_kwargs: Extra keyword arguments forwarded to ``from_url``
                (e.g. ``region``, ``skip_signature``, ``request_payer``).
        """
        if uri.lower().endswith(".vrt"):
            from .vrt import _open_vrt

            return await _open_vrt(
                uri,
                store=store,
                prefetch=prefetch,
                cache=cache,
                meta_overrides=meta_overrides,
                **store_kwargs,
            )

        if uri.lower().endswith(".xml"):
            from .formats.dimap import _maybe_open_dimap

            dimap_ds = await _maybe_open_dimap(
                uri,
                store=store,
                prefetch=prefetch,
                cache=cache,
                meta_overrides=meta_overrides,
                **store_kwargs,
            )
            if dimap_ds is not None:
                return dimap_ds
            # Non-DIMAP .xml falls through — the normal TIFF open below
            # will surface the "unexpected magic bytes" error.

        # Keyed before the fetch: a file rewritten during it must not be cached
        # under its new version.
        key = _cache_key(uri) if cache and _cache_max_size > 0 else None
        if key is not None:
            gt = _cache_get(key)
            if gt is not None:
                return cls(uri, gt, meta_overrides=meta_overrides)

        if store is None:
            store = _build_store(uri, **store_kwargs)
        geotiff = await GeoTIFF.open(_extract_key(uri), store=store, prefetch=prefetch)

        if key is not None:
            _cache_put(key, geotiff)

        return cls(uri, geotiff, meta_overrides=meta_overrides)

    async def read(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        bbox_crs: int | CRS | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        target_crs: int | CRS | None = None,
        target_resolution: float | None = None,
        snap_to_grid: bool = True,
        use_overviews: bool = False,
        resampling: ResamplingMethod = "nearest",
    ) -> RasterArray:
        """Read image data, optionally reprojecting and resampling.

        Args:
            bbox: Must be in *bbox_crs*, which must equal *target_crs* if set,
                else the dataset CRS. Defaults to the dataset's extent,
                clamped to ±85.0511° when reprojecting a geographic dataset to
                Mercator, as gdalwarp does.
            window: In full-resolution pixels. Combines with
                *target_resolution* but not with *target_crs*. Naming pixels
                the dataset does not have raises
                :class:`rastera.WindowOutOfRangeError` rather than padding —
                unlike *bbox*, which clips. A window is an exact pixel range,
                so overhanging one is a mistake and not a partial request.
            band_indices: 1-based. Each GeoTIFF read still fetches and decodes
                all of its bands, even when they are stored apart
                (``INTERLEAVE=BAND``); the subset is taken afterwards.
            target_resolution: Without it, a reprojecting read takes the
                resolution gdalwarp picks: ``gdalwarp -te`` with a bbox, and
                ``gdalwarp -t_srs`` for the whole dataset without one. The
                grid is then laid out as for a given resolution, square and
                rounded out, so it can have a row or column more than
                gdalwarp's, which rounds the pixel count and stretches each
                axis's pixel size to fit.
            snap_to_grid: When True (default) and *target_resolution* is
                given with a bbox, the output grid is rounded outward onto
                multiples of ``target_resolution`` — see
                :func:`rastera.snapped_grid_for_bbox`. Transform and shape
                are then a pure function of bbox and resolution, before the
                clip to the dataset described below; sources
                already on that grid are copied 1:1, anything else is
                resampled onto it, shifting values by up to half a pixel.
                Without *target_resolution* the window snaps outward on the
                source grid instead — a 1:1 copy of the stored pixels. When
                False, the transform is anchored at ``bbox``. A native read
                then matches its extent to within half a pixel and returns
                the pixels ``rasterio.read(window=from_bounds(...))`` does:
                GDAL's nearest picks, which on a span that is not whole
                pixels repeat or drop one row or column. A resampled one is
                ceil-sized, so the max edges can overhang ``bbox`` by up to
                a pixel.

                Within one CRS the result is clipped to the dataset either
                way, so a bbox reaching past the edge comes back smaller
                rather than padded — ``rasterio.read``'s default, where
                padding is ``boundless=True``. A reprojecting read is not
                clipped, matching ``gdalwarp -te``: it returns the whole grid
                the bbox names, and the pixels with no source behind them
                carry the dataset's ``nodata``, or 0 where it declares none.
                Reprojecting leaves some of those regardless, since the grid
                is the envelope of a footprint that arrives rotated. Which
                pixels they were is on ``RasterArray.mask`` when the dataset
                declares no sentinel — 0 is real data in most rasters, so
                comparing against it would blank them; when it does declare
                one the value is in the pixels and ``as_masked()`` finds it.
                A NaN sentinel is the exception: ``as_masked()`` compares by
                equality, which NaN never meets, so find it with
                ``np.isnan``.
            use_overviews: When True, reads from pre-computed COG overview
                levels to save bandwidth, and only when the read actually
                changes resolution — a native-resolution or purely
                reprojecting read ignores it, since every overview is coarser
                than what such a read asks for. Overview pixels are resampled
                aggregates, not original measurements — expect reduced
                variance, dampened extremes, and altered spectral ratios
                compared to full-resolution data. Suitable for thumbnails
                or coarse segmentation; avoid for tasks requiring precise
                pixel values such as spectral index computation or
                per-pixel regression.
            resampling: Used when reprojecting or changing resolution.
                ``"nearest"`` (default) is fast, exact and blocky;
                ``"bilinear"`` is smooth with no overshoot; ``"cubic"`` is
                sharper but can overshoot the source value range. Both
                kernels widen when downsampling, to anti-alias as GDAL's
                warp does, and renormalize around nodata GDAL-style — see
                :func:`rastera.resampling.resample` for the precise rules.
                A file with an internal mask raises ``NotImplementedError``
                here, since the warp would read the pixels it hides as data.
        """
        gt = self._geotiff
        band_indices = normalize_band_indices(band_indices, self.count)
        if window is not None and bbox is not None:
            raise ValueError("Cannot specify both bbox and window")
        if bbox is not None and bbox_crs is None:
            raise ValueError("bbox_crs is required when bbox is provided")
        if window is not None and target_crs is not None:
            raise ValueError("Cannot combine window with target_crs")
        # async-geotiff reads anything that is not its own Window, a rasterio
        # Window included, as no window at all and returns the whole image.
        if window is not None and not isinstance(window, Window):
            raise TypeError(
                f"window must be a rastera.Window, got "
                f"{type(window).__module__}.{type(window).__qualname__}"
            )
        if window is not None:
            _validate_window(gt, window)
        # ``resampling`` is checked here rather than left to ``resample()``: the
        # native path never calls it, so an unknown method was silently ignored
        # on exactly the reads where it looked like it had been honoured.
        validate_resampling(resampling)
        if target_resolution is not None:
            validate_resolution(target_resolution)

        if bbox_crs is not None:
            bbox_crs = _normalize_crs(bbox_crs)
        if target_crs is not None:
            target_crs = _normalize_crs(target_crs)
        if bbox_crs is not None or target_crs is not None:
            _require_epsg(self)

        needs_reproject = target_crs is not None and target_crs != self._crs_epsg
        # Both axes: a source with non-square pixels matching *target_resolution*
        # on x alone still has to be resampled, or the rows come back at the
        # source's y resolution while the caller is told they are square.
        needs_resample = target_resolution is not None and not (
            math.isclose(target_resolution, gt.res[0])
            and math.isclose(target_resolution, gt.res[1])
        )
        # bbox + explicit resolution names an output lattice; only then does
        # snap_to_grid mean "snap the output onto resolution multiples".
        snap = snap_to_grid and bbox is not None and target_resolution is not None

        # Native fast path: no reprojection or resampling needed, so read
        # directly from the source without an extra copy through resample().
        use_native = not needs_reproject and not needs_resample
        if use_native and snap:
            # A 1:1 window copy lands on the lattice only if the source grid
            # is on it: origin on multiples of the resolution, unrotated and
            # north-up. ``needs_resample`` already matched both axes' *sizes*;
            # the a and -e tests here are about their *signs*: a grid whose
            # columns run west came back mirrored, and a south-up grid's
            # positive e can never isclose a positive resolution.
            assert target_resolution is not None
            t = gt.transform
            use_native = (
                _is_on_res_grid(float(t.c), target_resolution)
                and _is_on_res_grid(float(t.f), target_resolution)
                and float(t.b) == 0
                and float(t.d) == 0
                and float(t.a) > 0
                and math.isclose(target_resolution, -float(t.e))
            )

        if bbox is not None and use_native:
            if bbox_crs != self._crs_epsg:
                raise ValueError(
                    f"bbox_crs ({bbox_crs}) does not match "
                    f"target CRS ({self._crs_epsg}). "
                    f"Please provide bbox in the target CRS."
                )
            bbox = ensure_bbox(bbox)

        if use_native:
            result = await self._read_native(
                bbox=bbox,
                window=window,
                band_indices=band_indices,
                snap_to_grid=snap_to_grid,
            )
            if snap:
                # Stamp the promised lattice: the file origin may carry float
                # noise the gate tolerates, and int*res is the exact
                # arithmetic snapped_grid_for_bbox uses.
                assert target_resolution is not None
                res, t = target_resolution, result.transform
                result = dc_replace(
                    result,
                    transform=Affine(
                        res,
                        0,
                        round(t.c / res) * res,
                        0,
                        -res,
                        round(t.f / res) * res,
                    ),
                )
            return result

        # Window + resample (window + reproject is rejected above)
        if window is not None:
            assert target_resolution is not None
            return await self._read_window_resampled(
                window=window,
                band_indices=band_indices,
                target_resolution=target_resolution,
                use_overviews=use_overviews,
                resampling=resampling,
            )

        return await self._read_resampled(
            bbox=ensure_bbox(bbox) if bbox is not None else None,
            bbox_crs=bbox_crs,
            band_indices=band_indices,
            target_crs=target_crs,
            target_resolution=target_resolution,
            needs_reproject=needs_reproject,
            needs_resample=needs_resample,
            snap=snap,
            use_overviews=use_overviews,
            resampling=resampling,
        )

    async def _read_window_resampled(
        self,
        window: Window,
        band_indices: Sequence[int] | None,
        target_resolution: float,
        use_overviews: bool,
        resampling: ResamplingMethod,
    ) -> RasterArray:
        """Read a pixel window and resample to *target_resolution*.

        Resolves *window* to world coordinates up front: it is in full-
        resolution pixels, so handing it to an overview would read a different
        region entirely.
        """
        gt = self._geotiff
        target_bbox = bounds_from_transform(
            gt.transform * Affine.translation(window.col_off, window.row_off),
            window.width,
            window.height,
        )
        out_transform, out_w, out_h = _grid_for_bbox(
            target_bbox, target_resolution, use_ceil=True
        )
        # read() rejects window together with target_crs, so this grid is
        # already in the dataset's own CRS.
        return await self._read_to_grid(
            dst_transform=out_transform,
            dst_width=out_w,
            dst_height=out_h,
            out_crs=self._crs_epsg,
            band_indices=band_indices,
            resampling=resampling,
            use_overviews=use_overviews,
        )

    async def _read_resampled(
        self,
        bbox: BBox | None,
        bbox_crs: int | None,
        band_indices: Sequence[int] | None,
        target_crs: int | None,
        target_resolution: float | None,
        needs_reproject: bool,
        needs_resample: bool,
        snap: bool,
        use_overviews: bool,
        resampling: ResamplingMethod,
    ) -> RasterArray:
        gt = self._geotiff
        src_crs = self._crs_epsg
        out_crs = target_crs or src_crs

        if bbox is not None:
            target_bbox = bbox
            if bbox_crs is not None and bbox_crs != out_crs:
                raise ValueError(
                    f"bbox_crs ({bbox_crs}) does not match target CRS ({out_crs}). "
                    f"Please provide bbox in the target CRS."
                )
        elif needs_reproject:
            # needs_reproject implies target_crs was given, so out_crs is set.
            assert src_crs is not None and out_crs is not None
            src_bounds = _grid_bounds(gt)
            clamped = _mercator_clamped_bounds(src_bounds, src_crs, out_crs)
            target_bbox = transform_bbox(clamped or src_bounds, src_crs, out_crs)
            if clamped is not None:
                # gdalwarp then reads as if -te named the clamped extent, which
                # also picks the -te resolution below.
                bbox = target_bbox
        else:
            target_bbox = _grid_bounds(gt)

        # Clip to the dataset, matching the native path (see read()'s docstring
        # for the semantics). The *bbox* and not the grid, so the result stays an
        # integer-pixel sub-window of the unclipped grid and
        # ``snapped_grid_for_bbox``'s lattice still describes it.
        #
        # Same CRS only. The native path this agrees with is itself unreachable
        # when reprojecting, so that is the whole of the disagreement — and
        # clipping a warp would mean transforming the dataset's extent into
        # *out_crs*, which ``transform_bounds`` under-reports badly for a wide
        # source: a global EPSG:4326 extent comes back as x ∈ [500000, 1505647]
        # in UTM32N, rejecting any AOI west of the central meridian.
        if bbox is not None and not needs_reproject:
            clipped = target_bbox.intersect(_grid_bounds(gt))
            if clipped is None:
                raise WindowOutOfRangeError("BBox does not intersect image")
            target_bbox = clipped

        if target_resolution is not None:
            res = target_resolution
        elif needs_reproject:
            assert src_crs is not None and out_crs is not None
            res = (
                _default_resolution(gt, src_crs, out_crs)
                if bbox is None
                else _default_resolution_for_bbox(gt, target_bbox, src_crs, out_crs)
            )
        else:
            res = gt.res[0]

        out_transform, out_w, out_h = (
            snapped_grid_for_bbox(target_bbox, res)
            if snap
            else _grid_for_bbox(target_bbox, res, use_ceil=True)
        )
        return await self._read_to_grid(
            dst_transform=out_transform,
            dst_width=out_w,
            dst_height=out_h,
            out_crs=out_crs,
            band_indices=band_indices,
            resampling=resampling,
            # gdalwarp's default *res* is not a resolution anyone asked for.
            use_overviews=use_overviews and needs_resample,
        )

    async def _read_to_grid(
        self,
        *,
        dst_transform: Affine,
        dst_width: int,
        dst_height: int,
        out_crs: int | None,
        band_indices: Sequence[int] | None,
        resampling: ResamplingMethod,
        use_overviews: bool,
    ) -> RasterArray:
        """Fill a caller-chosen destination grid from this dataset.

        The caller owns the grid; this owns the warp — which source pixels to
        fetch, from which overview, with how much halo, through which
        transformer.

        *dst_transform* must be north-up. *out_crs* is its EPSG; pass
        ``self._crs_epsg`` when the grid is already in this dataset's own CRS.
        """
        # The warp takes validity from nodata alone, so it would resample the
        # pixels an internal mask hides as data.
        masked_uri = self._internal_mask_uri()
        if masked_uri is not None:
            raise NotImplementedError(
                f"{masked_uri} has an internal mask, which resampled and "
                "reprojected reads do not support. Read it at native "
                "resolution, where the mask comes back on RasterArray.mask."
            )
        src_crs = self._crs_epsg
        needs_reproject = out_crs != src_crs
        # Destination pixel size, expressed in *source* units once reprojected,
        # so the halo and the overview are chosen against the grid the kernel
        # actually walks: the spacing picks the overview, the reach sizes the
        # halo (see ``_src_units_per_pixel``). In one CRS the two are the same.
        spacing = reach = (float(dst_transform.a), -float(dst_transform.e))

        read_bbox = bounds_from_transform(dst_transform, dst_width, dst_height)
        transformer = None
        if needs_reproject:
            assert src_crs is not None and out_crs is not None
            transformer = _transformer(out_crs, src_crs)
            spacing, reach = _src_units_per_pixel(transformer, read_bbox, spacing)
            read_bbox = transform_bbox(read_bbox, out_crs, src_crs)

        # Per axis: the coarsest overview no coarser than *either* axis wants,
        # so neither upsamples from a level that already lost the detail.
        overview = (
            self._best_overview_for_resolution(spacing) if use_overviews else None
        )
        readable = overview if overview is not None else self._geotiff
        read_bbox = _halo_bbox(
            read_bbox,
            method=resampling,
            dst_res=reach,
            src_res=(float(readable.res[0]), float(readable.res[1])),
            reprojecting=needs_reproject,
        )

        native = await self._read_native(
            bbox=read_bbox,
            band_indices=band_indices,
            overview=overview,
        )

        def _warp() -> tuple[np.ndarray, np.ndarray | None]:
            out, covered = _resample_impl(
                native.data,  # type: ignore[reportUnknownMemberType]
                src_transform=native.transform,
                dst_transform=dst_transform,
                dst_width=dst_width,
                dst_height=dst_height,
                nodata=self._nodata,
                transformer=transformer,
                method=resampling,
            )
            return _fill_uncovered(out, covered, self._nodata), covered

        # CPU-bound, and seconds long for a large warp: on the event loop it
        # stalled every other task for that long, and set_concurrency could not
        # overlap two of them.
        out_data, coverage = await asyncio.to_thread(_warp)

        # Coverage stands in as the mask only when the dataset declares no
        # sentinel. With one, the warp already wrote it outside the footprint
        # and ``as_masked()`` finds it by value; handing over coverage instead
        # would unmask every *interior* nodata pixel, which it knows nothing of.
        mask = coverage if self._nodata is None else None

        return _make_output_array(
            out_data,
            dst_transform,
            dst_width,
            dst_height,
            self._output_geotiff_ref(out_crs),
            mask=mask,
        )

    async def _read_native(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        overview: Any | None = None,
        snap_to_grid: bool = True,
    ) -> RasterArray:
        """Read at native resolution/CRS, optionally from an overview."""
        # async_geotiff's Window has no stride/step support, so reads always
        # pull every pixel in the requested window at the chosen overview
        # level; any further downsampling happens post-fetch in `resample`.
        readable = overview if overview is not None else self._geotiff
        unsnapped = bbox is not None and not snap_to_grid
        if unsnapped:
            _require_north_up(readable.transform)  # before any I/O

        rows = cols = None
        if bbox is None and window is None:
            bbox = _grid_bounds(readable)
        if window is None:
            assert bbox is not None
            if unsnapped:
                window, rows, cols = unsnapped_window(readable, bbox)
            else:
                window = window_from_bbox(readable, bbox)

        # ``_GeoTIFFLike`` carries no ``read``. The datasets that synthesize
        # their ``_geotiff`` override ``_read_native``, so this is always real.
        result = await read_window(cast("GeoTIFF | Overview", readable), window)

        # rastera reads an alpha band as a plain band. The file's tag is the only
        # sign of one, and GDAL puts it on band 4 of any 4-band Byte file it
        # creates, NIR or not. Left set, ``as_masked()`` masks every band by it,
        # and a ``band_indices`` subset leaves it naming the wrong band, or none.
        result = dc_replace(result, _alpha_band_idx=None)
        # Every band in order, the default read, is left alone: the fancy index
        # would copy it all, and the read then held twice its size.
        if band_indices is not None and list(band_indices) != list(range(result.count)):
            result = dc_replace(
                result,
                data=result.data[band_indices],  # type: ignore[reportUnknownMemberType]
                count=len(band_indices),
            )

        # Anchor on the requested bbox rather than the pixel-snapped window
        # origin — rasterio's fractional-window behaviour.  Clamped to the
        # image, as the window was: an edge the bbox overhangs would otherwise
        # label the pixels somewhere they are not.
        if unsnapped:
            assert bbox is not None
            if rows is not None and cols is not None:
                result = _take_picks(result, rows, cols)
            bbox = ensure_bbox(bbox)
            res_x, res_y = readable.res
            img = _grid_bounds(readable)
            result = dc_replace(
                result,
                transform=Affine(
                    res_x,
                    0,
                    max(bbox.minx, img.minx),
                    0,
                    -res_y,
                    min(bbox.maxy, img.maxy),
                ),
            )

        geotiff_ref = self._output_geotiff_ref(self._crs_epsg)
        if isinstance(geotiff_ref, _CrsNodata):
            result = dc_replace(result, _geotiff=geotiff_ref)
        return result

    def _output_geotiff_ref(self, out_crs: int | None) -> _GeoTIFFLike | _CrsNodata:
        """What ``RasterArray.crs``/``.nodata`` should read off for our output.

        The real GeoTIFF whenever it already agrees, so ``arr._geotiff`` stays
        a live handle. A stub otherwise, carrying what this dataset actually
        resolved: a ``meta_overrides`` CRS, a reprojection's target, or a
        sentinel ``_coerce_nodata`` dropped as unrepresentable. Labelling the
        output with the file's values instead is how a uint16 array comes back
        reporting ``nodata=-9999`` and crashes ``as_masked()``.
        """
        gt = self._geotiff
        # An override skips the agreement check rather than running it: asking
        # the file would raise on exactly the unparseable geo keys the
        # override replaces. An override that happens to match the file just
        # gets the stub, which carries everything the output reads.
        if (
            self._crs_override is None
            and out_crs == gt.crs.to_epsg()
            and _same_nodata(self._nodata, gt.nodata)
        ):
            return gt
        # No EPSG to build from leaves whatever this dataset resolved to —
        # possibly a WKT that has no code.
        crs = CRS.from_epsg(out_crs) if out_crs is not None else self._resolved_crs
        return _CrsNodata(crs, self._nodata)

    def __repr__(self) -> str:
        gt = self._geotiff
        return (
            f"AsyncGeoTIFF({self.uri}, "
            f"width={gt.width}, height={gt.height}, "
            f"crs={self._crs_epsg})"
        )


@overload
async def open(
    uri: str | os.PathLike[str],
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> AsyncGeoTIFF: ...


@overload
async def open(
    uri: Sequence[str | os.PathLike[str]],
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> list[AsyncGeoTIFF]: ...


async def open(
    uri: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> AsyncGeoTIFF | list[AsyncGeoTIFF]:
    """Open one or more GeoTIFFs from any supported URI.

    When a list of URIs is passed, files are opened concurrently with a
    shared object store for connection reuse. A URL with a query (presigned)
    and a local VRT or DIMAP each build their own.

    Args:
        uri: A single URI or a list of URIs. A local path may also be a
            ``pathlib.Path``, as in ``rasterio.open``.
        store: Optional pre-constructed store for connection reuse.
        prefetch: Number of bytes to prefetch when opening the TIFF.
        cache: When True, cache parsed TIFF headers in memory so that
            subsequent opens of the same URI skip the header fetch.
        meta_overrides: Optional header overrides (e.g. ``{"crs": 3006}``)
            for TIFFs missing or carrying incorrect georeferencing. The
            same override is applied to every URI when a list is passed.
        **store_kwargs: Extra kwargs forwarded to ``async_tiff.store.from_url``
            (e.g. ``skip_signature``, ``region``, ``request_payer``).
    """
    if isinstance(uri, str | os.PathLike):
        return await AsyncGeoTIFF.open(
            os.fspath(uri),
            store=store,
            prefetch=prefetch,
            cache=cache,
            meta_overrides=meta_overrides,
            **store_kwargs,
        )
    return await _open_many(
        [os.fspath(u) for u in uri],
        store=store,
        prefetch=prefetch,
        cache=cache,
        meta_overrides=meta_overrides,
        **store_kwargs,
    )


async def _open_many(
    uris: Sequence[str],
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> list[AsyncGeoTIFF]:
    """Open multiple GeoTIFFs concurrently with a shared store."""
    uris = list(uris)
    if not uris:
        return []
    if store is None:
        shared = [u for u in uris if not _needs_own_store(u)]
        if shared:
            _require_same_bucket(shared, "using a shared store")
        # Only when some header is not cached: the opens of cached ones never
        # touch the store, and building it blocks for 80-160 ms.
        if any(not cache or get_cached_geotiff(u) is None for u in shared):
            store = _build_store(shared[0], **store_kwargs)
        stores = [None if _needs_own_store(u) else store for u in uris]
    else:
        stores = [store] * len(uris)
    # store_kwargs is forwarded as well as consumed above: a URI that gets its
    # own store (`None` here) builds it from them, and the VRT and DIMAP
    # branches need them to build their own obstore for the descriptor fetch
    # (the async-tiff and obstore store types are not interchangeable). Plain
    # TIFF opens on the shared store ignore them.
    return await _gather_bounded(
        len(uris),
        [
            AsyncGeoTIFF.open(
                u,
                store=s,
                prefetch=prefetch,
                cache=cache,
                meta_overrides=meta_overrides,
                **store_kwargs,
            )
            for u, s in zip(uris, stores)
        ],
    )


# ---- Public cache API ----


def get_cached_geotiff(uri: str) -> GeoTIFF | None:
    """Return the cached parsed ``GeoTIFF`` for *uri*, or ``None`` on miss.

    The cache is the module-level LRU populated by ``AsyncGeoTIFF.open``.
    A hit moves *uri* to the most-recently-used position. Returns ``None``
    when caching is disabled (``set_cache_size(0)``) or the URI is absent.
    """
    if _cache_max_size > 0:
        key = _cache_key(uri)
        if key is not None:
            return _cache_get(key)
    return None


def clear_cache() -> None:
    """Drop all entries from the in-memory GeoTIFF header cache.

    Does not change the configured cache size; subsequent opens repopulate
    it up to the current limit. Also drops the cached pyproj Transformers, so
    a later change to PROJ's network or grid settings takes effect.
    """
    _geotiff_cache.clear()
    _transformer.cache_clear()


def set_cache_size(n: int) -> None:
    """Set the maximum number of parsed GeoTIFF headers held in memory.

    The cache is a process-wide LRU shared by all callers of
    ``AsyncGeoTIFF.open``; the default capacity is 128. Passing ``n=0``
    disables caching entirely and evicts everything currently held.
    Shrinking below the current population evicts least-recently-used
    entries until the new bound is satisfied.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"cache size must be int >= 0, got {n!r}")
    global _cache_max_size
    _cache_max_size = n
    while len(_geotiff_cache) > _cache_max_size:
        _geotiff_cache.popitem(last=False)


# ---- Internal helpers for constructing output Arrays ----


class _Readable(Protocol):
    """The one member ``_GeoTIFFLike`` withholds: the pixel fetch."""

    async def read(self, *, window: Window | None = None) -> RasterArray: ...


class _OverviewLike(_Readable, Protocol):
    """One level of a real file's pyramid — readable, unlike its parent header."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def res(self) -> tuple[float, float]: ...
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...
    @property
    def transform(self) -> Affine: ...


class _GeoTIFFLike(Protocol):
    """The ``self._geotiff`` contract: header metadata, no I/O.

    Only plain files hold a real ``async_geotiff.GeoTIFF``; the VRT and DIMAP
    datasets synthesize theirs (``_VirtualGeoTIFF``). Reading anything wider
    than this raises ``AttributeError`` on those, and annotating the attribute
    ``GeoTIFF`` is what let that pass unchecked — so widening this Protocol
    means every synthesized dataset must supply the new field.
    """

    @property
    def count(self) -> int: ...
    @property
    def crs(self) -> CRS: ...
    @property
    def nodata(self) -> float | None: ...
    @property
    def dtype(self) -> np.dtype[Any] | None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def res(self) -> tuple[float, float]: ...
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...
    @property
    def transform(self) -> Affine: ...
    @property
    def overviews(self) -> Sequence[_OverviewLike]: ...


@dataclass(frozen=True, slots=True)
class _CrsNodata:
    """Stub standing in for ``_geotiff`` on constructed RasterArray objects."""

    crs: CRS
    nodata: float | None


def _grid_for_bbox(
    bbox: BBox, res: float, *, use_ceil: bool = False
) -> tuple[Affine, int, int]:
    """Compute (transform, width, height) for a regular grid covering *bbox*.

    Uses ``round()`` by default to match rasterio/GDAL merge behaviour.
    When *use_ceil* is True, uses ``math.ceil()`` to match rasterio read
    behaviour (always covers the full bbox). Denoised first: on a degree grid,
    600 pixels of 0.0001 divide by 0.0002 to 300.00000000000244, which ceil
    alone makes 301.
    """
    fn = math.ceil if use_ceil else round
    width = max(1, fn(_denoise(bbox.width / res)))
    height = max(1, fn(_denoise(bbox.height / res)))
    transform = Affine(res, 0, bbox.minx, 0, -res, bbox.maxy)
    return transform, width, height


def _make_output_array(
    data: np.ndarray,
    transform: Affine,
    width: int,
    height: int,
    geotiff: _GeoTIFFLike | _CrsNodata,
    mask: np.ndarray | None = None,
) -> RasterArray:
    return RasterArray(
        data=data,
        mask=mask,
        width=width,
        height=height,
        count=data.shape[0],
        transform=transform,
        _alpha_band_idx=None,
        _geotiff=geotiff,  # type: ignore[reportArgumentType]
    )


def _take_picks(arr: RasterArray, rows: Picks, cols: Picks) -> RasterArray:
    """The *rows* and *cols* of *arr* that :func:`~rastera.geo.unsnapped_window`
    picked.

    An axis with as many picks as pixels is left alone: picks step by 0-1 or
    by 1-2, never both, so those are all of them, in order.
    """
    if (len(rows), len(cols)) == (arr.height, arr.width):
        return arr
    idx: tuple[Any, Any]
    if len(rows) == arr.height:
        idx = (slice(None), cols)
    elif len(cols) == arr.width:
        idx = (rows, slice(None))
    else:
        # One index for both axes: one axis after the other held a third
        # full-size copy at the peak.
        idx = (rows[:, None], cols)
    data: np.ndarray[Any, Any] = arr.data[(slice(None), *idx)]  # type: ignore[reportUnknownMemberType]
    mask = None if arr.mask is None else arr.mask[idx]
    return dc_replace(arr, data=data, mask=mask, height=len(rows), width=len(cols))


def _src_units_per_pixel(
    transformer: Transformer, bbox: BBox, dst_res: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """One destination pixel of *dst_res* in source-CRS units, per axis: its
    spacing, then its reach.

    The spacing is the length of one destination step, and picks the overview.
    The reach adds both destination steps along each source axis, since the
    CRSs may be rotated against each other. The kernel is widened by the reach
    (``resampling._footprint``), so the halo sized from it covers the kernel.
    The overview is not picked by the reach: on a grid rotated by θ it is
    cos θ + sin θ times the spacing, so a level that much coarser than the
    request was read and resampled up.

    A one-pixel finite difference at *bbox*'s centre, not a ratio of the bbox
    extents: ``transform_bbox`` returns a densified *envelope*, so for a thin
    grid — merge hands us 1-px-wide edge contributors — the envelope's width is
    set by the projection's curvature over the long axis rather than by the
    grid's own width, inflating the ratio by 100x and with it the halo.
    Falls back to *dst_res* for both if the probe leaves the transform's
    domain; ``transform_bbox`` on the same rectangle raises loudly right after.
    """
    cx = (bbox.minx + bbox.maxx) / 2.0
    cy = (bbox.miny + bbox.maxy) / 2.0
    rx, ry = dst_res
    xs, ys = transformer.transform([cx, cx + rx, cx], [cy, cy, cy + ry])
    if not all(math.isfinite(v) for v in (*xs, *ys)):
        return dst_res, dst_res
    spacing = (
        math.hypot(xs[1] - xs[0], ys[1] - ys[0]) or rx,
        math.hypot(xs[2] - xs[0], ys[2] - ys[0]) or ry,
    )
    reach = (
        abs(xs[1] - xs[0]) + abs(xs[2] - xs[0]) or rx,
        abs(ys[1] - ys[0]) + abs(ys[2] - ys[0]) or ry,
    )
    return spacing, reach


def _default_resolution(gt: _GeoTIFFLike, src_crs: int, dst_crs: int) -> float:
    """What ``gdalwarp -t_srs`` picks for *gt* in *dst_crs* without ``-te``.

    GDAL's ``GDALSuggestedWarpOutput2``: the distance in *dst_crs* between the
    grid's first and last corner, over the same diagonal in pixels. When the
    corners share an x or a y, as on a grid centred on a pole in EPSG:4326, the
    transformed extent's diagonal instead.
    """
    t = gt.transform
    x1, y1 = _affine_apply(t, gt.width, gt.height)
    xs, ys = _transformer(src_crs, dst_crs).transform([t.c, x1], [t.f, y1])
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    if dx == 0 or dy == 0 or not math.isfinite(dx + dy):
        extent = transform_bbox(_grid_bounds(gt), src_crs, dst_crs)
        dx, dy = extent.width, extent.height
    return math.hypot(dx, dy) / math.hypot(gt.width, gt.height)


def _default_resolution_for_bbox(
    gt: _GeoTIFFLike, bbox: BBox, src_crs: int, dst_crs: int
) -> float:
    """What ``gdalwarp -t_srs -te`` picks for *gt* in *dst_crs* over *bbox*.

    The finest local scale on a 10x10 grid of points across *bbox*: at each,
    a small step in x and one in y, each over the source pixels it spans,
    averaged. GDAL also drops points under a tenth of the median, which no
    ordinary grid has, and falls back to ``_default_resolution`` when no point
    transforms.
    """
    n = 10
    eps = min(bbox.width, bbox.height) / 1000
    x, y = np.meshgrid(
        np.linspace(bbox.minx, bbox.maxx, n), np.linspace(bbox.miny, bbox.maxy, n)
    )
    step = np.where(np.arange(n) == n - 1, -eps, eps)  # inward at the far edge
    sx, sy = _transformer(dst_crs, src_crs).transform(
        np.stack([x, x + step, x]), np.stack([y, y, y + step[:, None]])
    )
    inv = ~gt.transform
    col = inv.a * sx + inv.b * sy + inv.c
    row = inv.d * sx + inv.e * sy + inv.f
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (eps / np.hypot(col[1:] - col[0], row[1:] - row[0])).mean(axis=0)
    res = res[np.isfinite(res)]
    if not res.size:
        return _default_resolution(gt, src_crs, dst_crs)
    return float(res.min())


def _halo_bbox(
    bbox: BBox,
    *,
    method: ResamplingMethod,
    dst_res: tuple[float, float],
    src_res: tuple[float, float],
    reprojecting: bool,
) -> BBox:
    """Widen a source-read bbox by the reach of the resampling kernel.

    Sized to the output extent alone, the outermost pixels come out of a
    truncated, renormalised kernel — a biased ring, and two adjacent AOIs
    disagreeing along their shared edge. Per axis, because a kernel widened for
    a 10x downsample in x is not wide enough for a 2x one in y. Reprojecting,
    one pixel is the floor: nearest needs no kernel halo, but a cross-CRS
    ``read_bbox`` is a densified envelope, so the slack absorbs any curvature
    it under-states. In one CRS that pixel only reached into the neighbouring
    tiles.
    """
    floor = 1 if reprojecting else 0
    pad_x = max(floor, _kernel_halo(method, dst_res[0] / src_res[0])) * src_res[0]
    pad_y = max(floor, _kernel_halo(method, dst_res[1] / src_res[1])) * src_res[1]
    return BBox(
        bbox.minx - pad_x, bbox.miny - pad_y, bbox.maxx + pad_x, bbox.maxy + pad_y
    )


def _coerce_nodata(
    nodata: float | str | None, dtype: np.dtype[Any] | None
) -> int | float | None:
    """Coerce nodata, a number or the GDAL_NODATA tag's text, to match the
    raster dtype.

    The text goes through ``int()`` before ``float()``: a float rounds past
    2**53, so uint64's maximum, 2**64 - 1, would come back as 2**64 and out
    of range.

    Returns None when *dtype* cannot carry the value — NaN or a fraction on an
    integer band, or an integer outside the dtype's range. All mean "this
    raster has no representable sentinel": no pixel can ever equal it, which
    is also how GDAL reads it. Carrying it anyway makes ``np.array(nodata,
    dtype=...)`` inside ``resample`` raise ``OverflowError``, and truncating
    3.7 to 3 masked every real 3. A VRT declaring
    ``<NoDataValue>-9999</NoDataValue>`` over a uint16 source is the usual case
    (GDAL clamps the value when it fills, so its masked copy is a no-op there
    too).
    """
    if nodata is None or dtype is None:
        return None
    if isinstance(nodata, str):
        try:
            nodata = int(nodata)
        except ValueError:
            nodata = float(nodata)
    dt = np.dtype(dtype)
    if dt.kind in ("i", "u"):
        if math.isnan(nodata) or not float(nodata).is_integer():
            return None
        info = np.iinfo(dt)
        return None if not info.min <= nodata <= info.max else int(nodata)
    return float(nodata)


def _same_nodata(resolved: int | float | None, declared: float | None) -> bool:
    """``==``, but NaN equals itself: a float raster's NaN sentinel comes
    through ``_coerce_nodata`` unchanged and is still the file's own value."""
    if resolved is None or declared is None:
        return resolved is declared
    return resolved == declared or (math.isnan(resolved) and math.isnan(declared))


class MetaOverrides(TypedDict, total=False):
    """Header metadata overrides for ``open()``.

    Values replace what the GeoTIFF reports, even when already set.
    Useful when a TIFF is missing georeferencing that you know
    out-of-band (e.g. a sidecar-less file known to be EPSG:3006).

    Fields:
        crs: EPSG code (``int``) or ``pyproj.CRS`` declaring the
            dataset's coordinate reference system. Always *replaces*
            the file's reported CRS — there is no fallback semantics.
            Only relabels the data; it does not reproject. The override
            is what subsequent ``read()`` calls see as ``bbox_crs`` /
            ``target_crs`` source.
    """

    crs: int | CRS


_META_OVERRIDE_KEYS: frozenset[str] = frozenset({"crs"})


def _resolve_meta_overrides(
    overrides: MetaOverrides | None,
) -> dict[str, Any]:
    """Validate *overrides* and normalize values to their stored form."""
    if not overrides:
        return {}
    unknown = set(overrides) - _META_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"Unknown meta_overrides key(s): {sorted(unknown)}. "
            f"Allowed: {sorted(_META_OVERRIDE_KEYS)}."
        )
    resolved: dict[str, Any] = {}
    if "crs" in overrides:
        resolved["crs"] = _normalize_crs(overrides["crs"])
    return resolved


def _validate_window(gt: _GeoTIFFLike, window: Window) -> None:
    """Reject a window naming pixels the dataset does not have.

    Checked in ``read`` ahead of the native/resampled split, because neither
    branch sees it reliably on its own. The native path hands the window to
    whatever backs ``_geotiff`` and inherits that backend's answer: a real file
    raises async-geotiff's ``WindowError``, while a synthesized dataset pads
    instead — DIMAP pre-fills nodata and ``_tile_decomposition`` simply omits
    the tiles that do not exist. The resampled path converts the window to
    world coordinates first, so nothing downstream ever sees it. One check here
    means one answer for every dataset type.
    """
    if (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + window.width > gt.width
        or window.row_off + window.height > gt.height
    ):
        raise WindowOutOfRangeError(
            f"Window extends outside image bounds. Window: "
            f"cols={window.col_off}:{window.col_off + window.width}, "
            f"rows={window.row_off}:{window.row_off + window.height}. "
            f"Image: {gt.width}x{gt.height}."
        )


def _require_epsg(ds: AsyncGeoTIFF) -> None:
    """Raise unless *ds* has an EPSG code, which every CRS argument and every
    reprojection is matched against."""
    if ds._crs_epsg is None:
        raise ValueError(
            f"{ds.uri} has a CRS with no EPSG code, and rastera takes CRSs as "
            f"EPSG codes. If it is EPSG:N, open it with meta_overrides="
            f"{{'crs': N}}."
        )


def _cache_get(key: _CacheKey) -> GeoTIFF | None:
    gt = _geotiff_cache.get(key)
    if gt is not None:
        _geotiff_cache.move_to_end(key)
    return gt


def _cache_put(key: _CacheKey, gt: GeoTIFF) -> None:
    """Insert first, then trim: two opens of one uncached URI both land here,
    and trimming first evicted an unrelated entry for the second."""
    _geotiff_cache[key] = gt
    _geotiff_cache.move_to_end(key)
    while len(_geotiff_cache) > _cache_max_size:
        _geotiff_cache.popitem(last=False)


def _cache_key(uri: str) -> _CacheKey | None:
    """Where *uri*'s header is cached, or ``None`` when it cannot be.

    A local file is keyed on its resolved path, modification and change times,
    inode and size, so a file rewritten in place, or a relative path opened
    again from another directory, misses. The change time catches a rewrite
    that restores the modification time, as ``cp -p`` and ``tar -x`` do. A
    remote URI is keyed as given: checking that would cost a request per open,
    which is what the cache saves.
    """
    path = _resolve_local_path(uri)
    if path is None:
        return uri
    try:
        st = path.stat()
    except OSError:
        return None  # the open itself says why
    return (str(path), st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_size)


def _needs_own_store(uri: str) -> bool:
    """Whether *uri* in an ``open()`` list builds its own store rather than
    sharing the list's.

    A URL with a query is its own store root (see ``_parse_http_uri``), so two
    presigned URLs on one host were refused as two buckets, with both
    signatures in the error. And a local VRT or DIMAP, see
    ``_is_local_descriptor``.
    """
    return _is_local_descriptor(uri) or _parse_uri(uri).root == uri


def _source_store(
    uri: str,
    stores: dict[tuple[str, str | None], Any],
    cache: bool,
    **store_kwargs: Any,
) -> Any | None:
    """The store a VRT or DIMAP opens one of its sources with, shared per
    bucket across them (see ``_shared_store``).

    ``None`` when the header is cached: the open then returns before it needs
    one, and a warm descriptor open built none before stores were shared. And
    ``None`` for a nested VRT or DIMAP, whose own URI is never in the header
    cache: it shares stores among its sources itself, for the ones not cached.
    """
    if uri.lower().endswith((".vrt", ".xml")):
        return None
    if cache and get_cached_geotiff(uri) is not None:
        return None
    return _shared_store(uri, stores, **store_kwargs)
