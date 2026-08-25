from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any, Protocol, TypedDict, cast, overload

import numpy as np
from affine import Affine
from async_geotiff import GeoTIFF, RasterArray, Window
from pyproj import CRS, Transformer

from .geo import (
    BBox,
    WindowOutOfRangeError,
    _is_on_res_grid,
    _normalize_crs,
    bounds_from_transform,
    ensure_bbox,
    normalize_band_indices,
    snapped_grid_for_bbox,
    transform_bbox,
    validate_resolution,
    window_from_bbox,
)
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
    _require_same_bucket,
)

# LRU cache for parsed GeoTIFF objects, keyed by URI.
# Avoids re-fetching headers on repeated opens of the same file.
_geotiff_cache: OrderedDict[str, GeoTIFF] = OrderedDict()
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
        self._nodata: int | float | None = _coerce_nodata(geotiff.nodata, geotiff.dtype)

        self.overviews: list[tuple[int, int]] = [
            (o.width, o.height) for o in geotiff.overviews
        ]

    def _override_nodata(self, nodata: float) -> None:
        """Replace the sentinel this dataset's *pixels* use with *nodata*.

        A value this dataset's dtype cannot carry is ignored rather than
        treated as "no nodata", which would discard a sentinel the file does
        declare. Subclasses that wrap other datasets override this to push the
        value down to whoever actually resamples (see ``_VRTDataset``).
        """
        coerced = _coerce_nodata(nodata, self._geotiff.dtype)
        if coerced is not None:
            self._nodata = coerced

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

    def _best_overview_for_resolution(self, target_resolution: float):
        """Return the Overview whose resolution is closest to *target_resolution*
        without being coarser. Returns None to use full resolution.

        Reads the pyramid off ``_geotiff``, not ``self.overviews``: this needs
        readable ``Overview`` objects, while ``self.overviews`` holds (width,
        height) pairs and is emptied by both VRT flavours.
        """
        native_res = self._geotiff.res[0]
        valid = [
            (o, native_res * (self._geotiff.width / o.width))
            for o in self._geotiff.overviews
            if native_res * (self._geotiff.width / o.width) <= target_resolution
        ]
        return max(valid, key=lambda x: x[1])[0] if valid else None

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
                path within the store. If no store is provided, it
                is auto-constructed via ``async_tiff.store.from_url``.
            prefetch: Number of bytes to prefetch when opening the TIFF.
            cache: When True, cache the parsed GeoTIFF object in memory so that
                subsequent opens of the same URI skip the header fetch.
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

        if cache:
            gt = get_cached_geotiff(uri)
            if gt is not None:
                return cls(uri, gt, meta_overrides=meta_overrides)

        if store is None:
            store = _build_store(uri, **store_kwargs)
        geotiff = await GeoTIFF.open(_extract_key(uri), store=store, prefetch=prefetch)

        if cache and _cache_max_size > 0:
            if len(_geotiff_cache) >= _cache_max_size:
                _geotiff_cache.popitem(last=False)
            _geotiff_cache[uri] = geotiff

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
                else the dataset CRS.
            window: In full-resolution pixels. Combines with
                *target_resolution* but not with *target_crs*. Naming pixels
                the dataset does not have raises
                :class:`rastera.WindowOutOfRangeError` rather than padding —
                unlike *bbox*, which clips. A window is an exact pixel range,
                so overhanging one is a mistake and not a partial request.
            band_indices: 1-based.
            snap_to_grid: When True (default) and *target_resolution* is
                given with a bbox, the output grid is rounded outward onto
                multiples of ``target_resolution`` — see
                :func:`rastera.snapped_grid_for_bbox`. Transform and shape
                are then a pure function of bbox and resolution; sources
                already on that grid are copied 1:1, anything else is
                resampled onto it, shifting values by up to half a pixel.
                Without *target_resolution* the window snaps outward on the
                source grid instead — a 1:1 copy of the stored pixels. When
                False, the transform is anchored at ``bbox``. A native read
                then matches its extent to within half a pixel —
                ``rasterio.read(window=from_bounds(...))`` behaviour; a
                resampled one is ceil-sized, so the max edges can overhang
                ``bbox`` by up to a pixel.

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
        """
        gt = self._geotiff
        band_indices = normalize_band_indices(band_indices, self.count)
        if window is not None and bbox is not None:
            raise ValueError("Cannot specify both bbox and window")
        if bbox is not None and bbox_crs is None:
            raise ValueError("bbox_crs is required when bbox is provided")
        if window is not None and target_crs is not None:
            raise ValueError("Cannot combine window with target_crs")
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
            # the -e test here is about the y axis' *sign*, since a south-up
            # grid's positive e can never isclose a positive resolution.
            assert target_resolution is not None
            t = gt.transform
            use_native = (
                _is_on_res_grid(float(t.c), target_resolution)
                and _is_on_res_grid(float(t.f), target_resolution)
                and float(t.b) == 0
                and float(t.d) == 0
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
            target_bbox = transform_bbox(BBox(*gt.bounds), src_crs, out_crs)
        else:
            target_bbox = BBox(*gt.bounds)

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
            clipped = target_bbox.intersect(BBox(*gt.bounds))
            if clipped is None:
                raise WindowOutOfRangeError("BBox does not intersect image")
            target_bbox = clipped

        if target_resolution is not None:
            res = target_resolution
        elif needs_reproject:
            # Preserve native pixel density across the CRS change.
            assert src_crs is not None and out_crs is not None
            src_bbox = transform_bbox(target_bbox, out_crs, src_crs)
            native_res = gt.res[0]
            n_cols = max(1, round(src_bbox.width / native_res))
            n_rows = max(1, round(src_bbox.height / native_res))
            res = min(target_bbox.width / n_cols, target_bbox.height / n_rows)
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
            # Density-preserving *res* is not a resolution anyone asked for.
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
        src_crs = self._crs_epsg
        needs_reproject = out_crs != src_crs
        # Destination pixel size, expressed in *source* units once reprojected,
        # so the halo and the overview are chosen against the grid the kernel
        # actually walks.
        dst_res = (float(dst_transform.a), -float(dst_transform.e))

        read_bbox = bounds_from_transform(dst_transform, dst_width, dst_height)
        transformer = None
        if needs_reproject:
            assert src_crs is not None and out_crs is not None
            transformer = Transformer.from_crs(out_crs, src_crs, always_xy=True)
            dst_res = _src_units_per_pixel(transformer, read_bbox, dst_res)
            read_bbox = transform_bbox(read_bbox, out_crs, src_crs)

        # min(): the coarsest overview no coarser than *either* axis wants, so
        # neither upsamples from a level that already lost the detail.
        overview = (
            self._best_overview_for_resolution(min(dst_res)) if use_overviews else None
        )
        readable = overview if overview is not None else self._geotiff
        read_bbox = _halo_bbox(
            read_bbox,
            method=resampling,
            dst_res=dst_res,
            src_res=(float(readable.res[0]), float(readable.res[1])),
        )

        native = await self._read_native(
            bbox=read_bbox,
            band_indices=band_indices,
            overview=overview,
        )

        out_data, coverage = _resample_impl(
            native.data,  # type: ignore[reportUnknownMemberType]
            src_transform=native.transform,
            dst_transform=dst_transform,
            dst_width=dst_width,
            dst_height=dst_height,
            nodata=self._nodata,
            transformer=transformer,
            method=resampling,
        )
        out_data = _fill_uncovered(out_data, coverage, self._nodata)

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

        if bbox is None and window is None:
            bbox = BBox(*readable.bounds)
        if window is None:
            assert bbox is not None
            window = window_from_bbox(readable, bbox, snap_to_grid=snap_to_grid)

        # ``_GeoTIFFLike`` carries no ``read``. The datasets that synthesize
        # their ``_geotiff`` override ``_read_native``, so this is always real.
        result = await cast("_Readable", readable).read(window=window)

        if band_indices is not None:
            result = dc_replace(
                result,
                data=result.data[band_indices],  # type: ignore[reportUnknownMemberType]
                count=len(band_indices),
            )

        # Anchor on the requested bbox rather than the pixel-snapped window
        # origin — rasterio's fractional-window behaviour.  Clamped to the
        # image, as the window was: an edge the bbox overhangs would otherwise
        # label the pixels somewhere they are not.
        if bbox is not None and not snap_to_grid:
            bbox = ensure_bbox(bbox)
            res_x, res_y = readable.res
            img = BBox(*readable.bounds)
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
    uri: str,
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> AsyncGeoTIFF: ...


@overload
async def open(
    uri: Sequence[str],
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> list[AsyncGeoTIFF]: ...


async def open(
    uri: str | Sequence[str],
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> AsyncGeoTIFF | list[AsyncGeoTIFF]:
    """Open one or more GeoTIFFs from any supported URI.

    When a list of URIs is passed, files are opened concurrently with a
    shared object store for connection reuse.

    Args:
        uri: A single URI or a list of URIs.
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
    if isinstance(uri, str):
        return await AsyncGeoTIFF.open(
            uri,
            store=store,
            prefetch=prefetch,
            cache=cache,
            meta_overrides=meta_overrides,
            **store_kwargs,
        )
    return await _open_many(
        uri,
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
        _require_same_bucket(uris, "using a shared store")
        store = _build_store(uris[0], **store_kwargs)
    # store_kwargs is forwarded as well as consumed above: plain TIFF opens
    # ignore it once `store` is set, but the VRT and DIMAP branches need it to
    # build their own obstore for the descriptor fetch (the async-tiff and
    # obstore store types are not interchangeable).
    return list(
        await asyncio.gather(
            *(
                AsyncGeoTIFF.open(
                    u,
                    store=store,
                    prefetch=prefetch,
                    cache=cache,
                    meta_overrides=meta_overrides,
                    **store_kwargs,
                )
                for u in uris
            )
        )
    )


# ---- Public cache API ----


def get_cached_geotiff(uri: str) -> GeoTIFF | None:
    """Return the cached parsed ``GeoTIFF`` for *uri*, or ``None`` on miss.

    The cache is the module-level LRU populated by ``AsyncGeoTIFF.open``.
    A hit moves *uri* to the most-recently-used position. Returns ``None``
    when caching is disabled (``set_cache_size(0)``) or the URI is absent.
    """
    if _cache_max_size > 0:
        gt = _geotiff_cache.get(uri)
        if gt is not None:
            _geotiff_cache.move_to_end(uri)
        return gt
    return None


def clear_cache() -> None:
    """Drop all entries from the in-memory GeoTIFF header cache.

    Does not change the configured cache size; subsequent opens repopulate
    it up to the current limit.
    """
    _geotiff_cache.clear()


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
    behaviour (always covers the full bbox).
    """
    fn = math.ceil if use_ceil else round
    width = max(1, fn(bbox.width / res))
    height = max(1, fn(bbox.height / res))
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


def _src_units_per_pixel(
    transformer: Transformer, bbox: BBox, dst_res: tuple[float, float]
) -> tuple[float, float]:
    """*dst_res* re-expressed in source-CRS units, per axis.

    A one-pixel finite difference at *bbox*'s centre, not a ratio of the bbox
    extents: ``transform_bbox`` returns a densified *envelope*, so for a thin
    grid — merge hands us 1-px-wide edge contributors — the envelope's width is
    set by the projection's curvature over the long axis rather than by the
    grid's own width, inflating the ratio by 100x and with it the halo.
    Falls back to *dst_res* if the probe leaves the transform's domain;
    ``transform_bbox`` on the same rectangle raises loudly right after.
    """
    cx = (bbox.minx + bbox.maxx) / 2.0
    cy = (bbox.miny + bbox.maxy) / 2.0
    rx, ry = dst_res
    xs, ys = transformer.transform([cx, cx + rx, cx], [cy, cy, cy + ry])
    if not all(math.isfinite(v) for v in (*xs, *ys)):
        return dst_res
    # Hypotenuse, not the x/y component: the two CRSs may be rotated relative to
    # each other, so a step along dst x moves in both source axes.
    step_x = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
    step_y = math.hypot(xs[2] - xs[0], ys[2] - ys[0])
    return (step_x or rx, step_y or ry)


def _halo_bbox(
    bbox: BBox,
    *,
    method: ResamplingMethod,
    dst_res: tuple[float, float],
    src_res: tuple[float, float],
) -> BBox:
    """Widen a source-read bbox by the reach of the resampling kernel.

    Sized to the output extent alone, the outermost pixels come out of a
    truncated, renormalised kernel — a biased ring, and two adjacent AOIs
    disagreeing along their shared edge. Per axis, because a kernel widened for
    a 10x downsample in x is not wide enough for a 2x one in y. One pixel is the
    floor: nearest needs no kernel halo, but a cross-CRS ``read_bbox`` is a
    densified envelope, so the slack absorbs any curvature it under-states.
    """
    pad_x = max(1, _kernel_halo(method, dst_res[0] / src_res[0])) * src_res[0]
    pad_y = max(1, _kernel_halo(method, dst_res[1] / src_res[1])) * src_res[1]
    return BBox(
        bbox.minx - pad_x, bbox.miny - pad_y, bbox.maxx + pad_x, bbox.maxy + pad_y
    )


def _coerce_nodata(
    nodata: float | None, dtype: np.dtype[Any] | None
) -> int | float | None:
    """Coerce nodata from async-geotiff (always float) to match the raster dtype.

    Returns None when *dtype* cannot carry the value — NaN on an integer band,
    or an integer outside the dtype's range. Both mean "this raster has no
    representable sentinel": no pixel can ever equal it, and carrying it
    anyway makes ``np.array(nodata, dtype=...)`` inside ``resample`` raise
    ``OverflowError``. A VRT declaring ``<NoDataValue>-9999</NoDataValue>``
    over a uint16 source is the case that reaches this (GDAL clamps the value
    when it fills, so its masked copy is a no-op there too).
    """
    if nodata is None or dtype is None:
        return None
    dt = np.dtype(dtype)
    if dt.kind in ("i", "u"):
        if math.isnan(nodata):
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
