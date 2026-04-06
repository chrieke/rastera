from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any, overload

import numpy as np
from affine import Affine
from async_geotiff import GeoTIFF, RasterArray, Window
from async_tiff.store import from_url
from pyproj import CRS, Transformer

from .geo import (
    BBox,
    _normalize_crs,
    ensure_bbox,
    normalize_band_indices,
    resample_nearest,
    transform_bbox,
    window_from_bbox,
)
from .store import (
    _apply_s3_defaults,
    _bucket_url,
    _build_store,
    _extract_key,
    _resolve_local_path,
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

    def __init__(self, uri: str, geotiff: GeoTIFF):
        self.uri = uri
        self._geotiff = geotiff
        self._crs_epsg: int | None = geotiff.crs.to_epsg()
        self._nodata: int | float | None = _coerce_nodata(geotiff.nodata, geotiff.dtype)

        self.overviews: list[tuple[int, int]] = [
            (o.width, o.height) for o in geotiff.overviews
        ]

    def _best_overview_for_resolution(self, target_resolution: float):
        """Return the Overview whose resolution is closest to *target_resolution*
        without being coarser. Returns None to use full resolution."""
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
            **store_kwargs: Extra keyword arguments forwarded to ``from_url``
                (e.g. ``region``, ``skip_signature``, ``request_payer``).
        """
        if cache:
            gt = get_cached_geotiff(uri)
            if gt is not None:
                return cls(uri, gt)

        if store is not None:
            key = _extract_key(uri)
            geotiff = await GeoTIFF.open(key, store=store, prefetch=prefetch)
        else:
            local_path = _resolve_local_path(uri)
            if local_path is not None:
                store = from_url(local_path.parent.as_uri(), **store_kwargs)
                geotiff = await GeoTIFF.open(
                    local_path.name, store=store, prefetch=prefetch
                )
            else:
                _apply_s3_defaults(store_kwargs, uri)
                store = from_url(uri, **store_kwargs)
                geotiff = await GeoTIFF.open("", store=store, prefetch=prefetch)

        if cache and _cache_max_size > 0:
            if len(_geotiff_cache) >= _cache_max_size:
                _geotiff_cache.popitem(last=False)
            _geotiff_cache[uri] = geotiff

        return cls(uri, geotiff)

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
    ) -> RasterArray:
        """Read image data, optionally reprojecting and resampling.

        Args:
            bbox: (minx, miny, maxx, maxy). Must be in target_crs if set,
                else dataset CRS.
            bbox_crs: EPSG code or ``pyproj.CRS`` of the bbox coordinate
                system. Must match target_crs (or the dataset CRS when
                target_crs is not set).
            window: Pixel window (col_off, row_off, width, height). Can
                combine with target_resolution for resampling but not with
                target_crs.
            band_indices: 1-based band indices to read.
            target_crs: Output EPSG code or ``pyproj.CRS``. When set,
                data is reprojected.
            target_resolution: Output pixel size in target CRS units.
            snap_to_grid: When True (default), the output grid snaps to
                the source pixel grid for exact 1:1 copies (no resampling);
                the bbox may shift by up to 1 pixel. When False, the output
                bbox matches the requested bbox exactly and nearest-neighbor
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
            An ``async_geotiff.RasterArray`` containing pixel data and spatial metadata.
        """
        gt = self._geotiff
        band_indices = normalize_band_indices(band_indices, gt.count)
        if window is not None and bbox is not None:
            raise ValueError("Cannot specify both bbox and window")
        if bbox is not None and bbox_crs is None:
            raise ValueError("bbox_crs is required when bbox is provided")
        if window is not None and target_crs is not None:
            raise ValueError("Cannot combine window with target_crs")

        if bbox_crs is not None:
            bbox_crs = _normalize_crs(bbox_crs)
        if target_crs is not None:
            target_crs = _normalize_crs(target_crs)

        needs_reproject = target_crs is not None and target_crs != self._crs_epsg
        needs_resample = target_resolution is not None and not math.isclose(
            target_resolution, gt.res[0], rel_tol=1e-6
        )

        # Native fast path: no reprojection or resampling needed, so read
        # directly from the source without an extra copy through resample_nearest.
        use_native = not needs_reproject and not needs_resample

        if bbox is not None and use_native:
            if bbox_crs != self._crs_epsg:
                raise ValueError(
                    f"bbox_crs ({bbox_crs}) does not match "
                    f"target CRS ({self._crs_epsg}). "
                    f"Please provide bbox in the target CRS."
                )
            bbox = ensure_bbox(bbox)

        if use_native:
            return await self._read_native(
                bbox=bbox,
                window=window,
                band_indices=band_indices,
                snap_to_grid=snap_to_grid,
            )

        # Window + resample (window + reproject is rejected above)
        if window is not None:
            return await self._read_window_resampled(
                window=window,
                band_indices=band_indices,
                target_resolution=target_resolution,
                use_overviews=use_overviews,
            )

        return await self._read_resampled(
            bbox=ensure_bbox(bbox) if bbox is not None else None,
            bbox_crs=bbox_crs,
            band_indices=band_indices,
            target_crs=target_crs,
            target_resolution=target_resolution,
            needs_reproject=needs_reproject,
            needs_resample=needs_resample,
            use_overviews=use_overviews,
        )

    async def _read_window_resampled(
        self,
        window: Window,
        band_indices: Sequence[int] | None,
        target_resolution: float,
        use_overviews: bool,
    ) -> RasterArray:
        """Read a pixel window and resample to *target_resolution*."""
        overview = (
            self._best_overview_for_resolution(target_resolution)
            if use_overviews
            else None
        )
        native = await self._read_native(
            window=window,
            band_indices=band_indices,
            overview=overview,
        )
        target_bbox = BBox(*native.bounds)
        out_transform, out_w, out_h = _grid_for_bbox(
            target_bbox, target_resolution, use_ceil=True
        )
        out_data = resample_nearest(
            native.data,
            src_transform=native.transform,
            dst_transform=out_transform,
            dst_width=out_w,
            dst_height=out_h,
            nodata=self._nodata,
        )
        return _make_output_array(out_data, out_transform, out_w, out_h, self._geotiff)

    async def _read_resampled(
        self,
        bbox: BBox | None,
        bbox_crs: int | None,
        band_indices: Sequence[int] | None,
        target_crs: int | None,
        target_resolution: float | None,
        needs_reproject: bool,
        needs_resample: bool,
        use_overviews: bool,
    ) -> RasterArray:
        """Read with reprojection and/or resampling."""
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
            target_bbox = transform_bbox(BBox(*gt.bounds), src_crs, out_crs)
        else:
            target_bbox = BBox(*gt.bounds)

        src_bbox = (
            transform_bbox(target_bbox, out_crs, src_crs)
            if needs_reproject
            else target_bbox
        )

        # Pick the best overview for the target resolution.
        # For cross-CRS reads, convert target resolution to source CRS units
        # using the bbox width ratio (e.g. 0.001° → ~83m).
        overview = None
        if needs_resample and use_overviews:
            src_res = target_resolution
            if needs_reproject:
                src_res = target_resolution * (src_bbox.width / target_bbox.width)
            overview = self._best_overview_for_resolution(src_res)

        # Determine output resolution
        if target_resolution is not None:
            res = target_resolution
        elif needs_reproject:
            # Preserve native pixel density across the CRS change.
            native_res = gt.res[0]
            n_cols = max(1, round(src_bbox.width / native_res))
            n_rows = max(1, round(src_bbox.height / native_res))
            res = min(target_bbox.width / n_cols, target_bbox.height / n_rows)
        else:
            res = gt.res[0]

        out_transform, out_w, out_h = _grid_for_bbox(target_bbox, res, use_ceil=True)

        # ceil() in _grid_for_bbox may extend the output grid beyond
        # target_bbox.  Expand the source read bbox to cover the full
        # output extent so the resampler has source data for every
        # destination pixel (avoids nodata/black edges).
        read_bbox = BBox(
            target_bbox.minx,
            target_bbox.maxy - out_h * res,
            target_bbox.minx + out_w * res,
            target_bbox.maxy,
        )
        if needs_reproject:
            read_bbox = transform_bbox(read_bbox, out_crs, src_crs)

        native = await self._read_native(
            bbox=read_bbox,
            band_indices=band_indices,
            overview=overview,
        )

        transformer = None
        if needs_reproject:
            transformer = Transformer.from_crs(out_crs, src_crs, always_xy=True)

        out_data = resample_nearest(
            native.data,
            src_transform=native.transform,
            dst_transform=out_transform,
            dst_width=out_w,
            dst_height=out_h,
            nodata=self._nodata,
            transformer=transformer,
        )

        geotiff_ref = (
            _CrsNodata(CRS.from_epsg(out_crs), self._nodata) if needs_reproject else gt
        )
        return _make_output_array(out_data, out_transform, out_w, out_h, geotiff_ref)

    async def _read_native(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        overview: Any | None = None,
        snap_to_grid: bool = True,
    ) -> RasterArray:
        """Read at native resolution/CRS, optionally from an overview."""
        # TODO: async_geotiff's Window has no stride/step support, so we
        # always read at full tile resolution even when the output is much
        # coarser. If async_geotiff adds stride support we could do
        # decimated reads here to skip unnecessary pixels and reduce I/O.

        # Determine which readable to use (full-res GeoTIFF or an Overview)
        if overview is not None:
            readable = overview
        else:
            readable = self._geotiff

        if bbox is None and window is None:
            bbox = BBox(*readable.bounds)
        if window is None:
            window = window_from_bbox(readable, bbox)

        # Use async-geotiff's built-in read (handles tile fetching + stitching)
        result = await readable.read(window=window)

        # Select requested bands
        if band_indices is not None:
            result = dc_replace(
                result, data=result.data[band_indices], count=len(band_indices)
            )

        # When reading by bbox, override the transform origin to match the
        # exact requested bbox (not the pixel-snapped window origin).  This
        # mirrors rasterio's fractional-window behaviour.
        if bbox is not None and not snap_to_grid:
            bbox = ensure_bbox(bbox)
            res = readable.res[0]
            result = dc_replace(
                result,
                transform=Affine(res, 0, bbox.minx, 0, -res, bbox.maxy),
            )

        return result

    def __repr__(self) -> str:
        gt = self._geotiff
        return (
            f"AsyncGeoTIFF({self.uri}, "
            f"width={gt.width}, height={gt.height}, "
            f"crs={self._crs_epsg})"
        )


async def _open_many(
    uris: Sequence[str],
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    **store_kwargs: Any,
) -> list[AsyncGeoTIFF]:
    """Open multiple GeoTIFFs concurrently with a shared store."""
    uris = list(uris)
    if not uris:
        return []
    if store is None:
        bucket = _bucket_url(uris[0])
        mismatched = [u for u in uris[1:] if _bucket_url(u) != bucket]
        if mismatched:
            raise ValueError(
                f"All URIs must belong to the same bucket/host when using a "
                f"shared store. First URI resolves to {bucket!r}, but these "
                f"do not: {mismatched}"
            )
        store = _build_store(uris[0], **store_kwargs)
    return list(
        await asyncio.gather(
            *(
                AsyncGeoTIFF.open(u, store=store, prefetch=prefetch, cache=cache)
                for u in uris
            )
        )
    )


@overload
async def open(
    uri: str,
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    **store_kwargs,
) -> AsyncGeoTIFF: ...


@overload
async def open(
    uri: Sequence[str],
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    **store_kwargs,
) -> list[AsyncGeoTIFF]: ...


async def open(
    uri: str | Sequence[str],
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    **store_kwargs,
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
        **store_kwargs: Extra kwargs forwarded to ``async_tiff.store.from_url``
            (e.g. ``skip_signature``, ``region``, ``request_payer``).
    """
    if isinstance(uri, str):
        return await AsyncGeoTIFF.open(
            uri, store=store, prefetch=prefetch, cache=cache, **store_kwargs
        )
    return await _open_many(
        uri, store=store, prefetch=prefetch, cache=cache, **store_kwargs
    )


# ---- Public cache API ----


def get_cached_geotiff(uri: str) -> GeoTIFF | None:
    """Return a cached GeoTIFF object for *uri*, or None on cache miss."""
    if _cache_max_size > 0:
        gt = _geotiff_cache.get(uri)
        if gt is not None:
            _geotiff_cache.move_to_end(uri)
        return gt
    return None


def clear_cache() -> None:
    """Clear the in-memory GeoTIFF header cache."""
    _geotiff_cache.clear()


def set_cache_size(n: int) -> None:
    """Set max cached GeoTIFF objects (LRU eviction). 0 disables."""
    global _cache_max_size
    _cache_max_size = n
    while len(_geotiff_cache) > _cache_max_size:
        _geotiff_cache.popitem(last=False)


# ---- Internal helpers for constructing output Arrays ----


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
    geotiff,
    mask: np.ndarray | None = None,
) -> RasterArray:
    """Construct a RasterArray for rastera output."""
    return RasterArray(
        data=data,
        mask=mask,
        width=width,
        height=height,
        count=data.shape[0],
        transform=transform,
        _alpha_band_idx=None,
        _geotiff=geotiff,
    )


def _coerce_nodata(nodata: float | None, dtype: np.dtype) -> int | float | None:
    """Coerce nodata from async-geotiff (always float) to match the raster dtype."""
    if nodata is None:
        return None
    kind = np.dtype(dtype).kind
    if kind in ("i", "u"):
        return None if math.isnan(nodata) else int(nodata)
    return float(nodata)
