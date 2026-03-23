from __future__ import annotations

import asyncio
import math
import os
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

import numpy as np
from affine import Affine
from async_tiff import TIFF
from async_tiff.store import from_url
from pyproj import Transformer

from .geo import (
    BBox,
    Window,
    compute_tile_paste_slices,
    ensure_bbox,
    get_intersecting_image_tiles,
    normalize_band_indices,
    resample_nearest,
    transform_bbox,
)
from .meta import Profile

_DEFAULT_REGION = (
    os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"
)

# In-memory cache for parsed TIFF objects, keyed by URI.
# Avoids re-fetching headers on repeated opens of the same file.
_tiff_cache: dict[str, TIFF] = {}
_cache_max_size: int = 128

# Maximum number of tiles to fetch in a single fetch_tiles() call.
# Prevents HTTP connection pool exhaustion when reading large bboxes
# from high-resolution COGs (which can require 100-200+ tiles).
# Keep DEFAULT_CONCURRENCY (merge.py) × _TILE_FETCH_BATCH_SIZE < 200
# to stay within safe HTTP connection pool limits (reqwest/async_tiff).
_TILE_FETCH_BATCH_SIZE: int = 48


def clear_cache() -> None:
    """Clear the in-memory TIFF header cache."""
    _tiff_cache.clear()


def set_cache_size(n: int) -> None:
    """Set the maximum number of cached TIFF objects. 0 disables caching."""
    global _cache_max_size
    _cache_max_size = n
    while len(_tiff_cache) > _cache_max_size:
        _tiff_cache.pop(next(iter(_tiff_cache)))


def set_tile_fetch_batch_size(n: int) -> None:
    """Set the maximum number of tiles per fetch_tiles() call.

    Controls the peak number of concurrent HTTP range requests per COG read.
    Total peak concurrent requests = max_concurrency × batch_size.
    """
    global _TILE_FETCH_BATCH_SIZE
    _TILE_FETCH_BATCH_SIZE = n


class AsyncGeoTIFF:
    """AsyncGeoTIFF instance for a single GeoTIFF file.

    IFD: Image File Directory. One image can contain multiple IFDs, e.g. overview, masks etc.,
    all stored in tiff.ifds.
    """

    def __init__(self, uri: str, tiff: TIFF, ifd_index: int = 0):
        self.uri = uri
        self.tiff = tiff
        self.ifd_index = ifd_index
        self.ifd = tiff.ifds[self.ifd_index]
        self.profile: Profile = Profile.from_ifd(self.ifd)

        # Attach overview sizes (already parsed during open — no extra requests)
        base_w, base_h = self.ifd.image_width, self.ifd.image_height
        self.overviews: list[tuple[int, int]] = [
            (ifd.image_width, ifd.image_height)
            for ifd in self.tiff.ifds[1:]
            if ifd.image_width < base_w and ifd.image_height < base_h
        ]
        self.profile.overviews = self.overviews

    def _best_ifd_for_resolution(self, target_resolution: float) -> int:
        """Return the IFD index whose resolution is closest to *target_resolution*
        without being coarser. Falls back to the full-resolution IFD (0)."""
        native_res = self.profile.res[0]
        best_idx = self.ifd_index  # default: full res
        best_res = native_res

        for i, ifd in enumerate(self.tiff.ifds[1:], start=1):
            if ifd.image_width >= self.ifd.image_width:
                continue  # not an overview
            ovr_res = native_res * (self.ifd.image_width / ifd.image_width)
            if ovr_res <= target_resolution and ovr_res >= best_res:
                best_res = ovr_res
                best_idx = i

        return best_idx

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
            uri: Any URI supported by object_store (s3://, https://, gs://, file://, etc.).
            store: Optional pre-constructed store. When provided, the key is
                extracted from the URI and used as the path within the store. If no store is
                provided, it is auto-constructed from the URI via ``async_tiff.store.from_url``.
            prefetch: Number of bytes to prefetch when opening the TIFF.
            cache: When True, cache the parsed TIFF object in memory so that
                subsequent opens of the same URI skip the header fetch.
            **store_kwargs: Extra keyword arguments forwarded to ``from_url``
                (e.g. ``region``, ``skip_signature``, ``request_payer``).
        """
        if cache and _cache_max_size > 0 and uri in _tiff_cache:
            return cls(uri, _tiff_cache[uri])

        if store is not None:
            key = _extract_key(uri)
            tiff = await TIFF.open(key, store=store, prefetch=prefetch)
        else:
            local_path = _resolve_local_path(uri)
            if local_path is not None:
                store = from_url(local_path.parent.as_uri(), **store_kwargs)
                tiff = await TIFF.open(local_path.name, store=store, prefetch=prefetch)
            else:
                if _is_s3_uri(uri):
                    store_kwargs.setdefault("skip_signature", True)
                    store_kwargs.setdefault("region", _detect_region(uri))
                store = from_url(uri, **store_kwargs)
                tiff = await TIFF.open("", store=store, prefetch=prefetch)

        if cache and _cache_max_size > 0:
            if len(_tiff_cache) >= _cache_max_size:
                _tiff_cache.pop(next(iter(_tiff_cache)))
            _tiff_cache[uri] = tiff

        return cls(uri, tiff)

    async def read(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        bbox_crs: int | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        target_crs: int | None = None,
        target_resolution: float | None = None,
    ) -> tuple[np.ndarray, Profile]:
        """Read image data, optionally reprojecting and resampling.

        Args:
            bbox: (minx, miny, maxx, maxy). In bbox_crs if set, else
                target_crs if set, else dataset CRS.
            bbox_crs: EPSG code of the bbox coordinate system. When set, the
                bbox is transformed to the appropriate CRS automatically.
            window: Pixel window (col_min, col_max, row_min, row_max). Can
                combine with target_resolution for resampling but not with
                target_crs.
            band_indices: 1-based band indices to read.
            target_crs: Output EPSG code. When set, data is reprojected.
            target_resolution: Output pixel size in target CRS units.

        Returns:
            Tuple of (numpy array, Profile) containing pixel data and spatial metadata.
        """
        band_indices = normalize_band_indices(band_indices, self.ifd.samples_per_pixel)
        if window is not None and bbox is not None:
            raise ValueError("Cannot specify both bbox and window")
        if bbox is not None and bbox_crs is None:
            raise ValueError("bbox_crs is required when bbox is provided")
        if window is not None and target_crs is not None:
            raise ValueError("Cannot combine window with target_crs")

        needs_reproject = target_crs is not None and target_crs != self.profile.crs_epsg
        needs_resample = target_resolution is not None and not math.isclose(
            target_resolution, self.profile.res[0], rel_tol=1e-6
        )

        # When bbox_crs is given with no reprojection/resampling, transform
        # the bbox into the dataset's native CRS for the native read path.
        if (
            bbox is not None
            and bbox_crs is not None
            and not needs_reproject
            and not needs_resample
        ):
            bbox = transform_bbox(ensure_bbox(bbox), bbox_crs, self.profile.crs_epsg)

        if not needs_reproject and not needs_resample:
            data, profile = await self._read_native(
                bbox=bbox, window=window, band_indices=band_indices
            )
            return data, profile

        # Pick the best overview IFD for the target resolution
        ovr_idx = (
            self._best_ifd_for_resolution(target_resolution)
            if needs_resample
            else self.ifd_index
        )

        # Window + resample: read native pixels for the window, then resample
        if window is not None and needs_resample:
            native_arr, native_profile = await self._read_native(
                window=window, band_indices=band_indices, ifd_index=ovr_idx
            )
            target_bbox = native_profile.bounds
            res = target_resolution
            out_profile = Profile.for_bbox(
                target_bbox,
                res,
                self.profile.crs_epsg,
                count=len(band_indices),
                dtype=self.profile.dtype,
                nodata=self.profile.nodata,
            )
            out_array = resample_nearest(
                native_arr,
                src_transform=native_profile.transform,
                dst_transform=out_profile.transform,
                dst_width=out_profile.width,
                dst_height=out_profile.height,
                nodata=self.profile.nodata,
            )
            return out_array, out_profile

        # Target bbox in target CRS (or source CRS if no reprojection)
        src_crs = self.profile.crs_epsg
        out_crs = target_crs or src_crs

        if bbox is not None:
            target_bbox = ensure_bbox(bbox)
            # Transform bbox from bbox_crs into the output CRS
            if bbox_crs is not None and bbox_crs != out_crs:
                target_bbox = transform_bbox(target_bbox, bbox_crs, out_crs)
        else:
            target_bbox = None

        # Transform bbox to source CRS for tile fetching
        if target_bbox is None:
            if needs_reproject:
                target_bbox = transform_bbox(self.profile.bounds, src_crs, out_crs)
            else:
                target_bbox = self.profile.bounds

        if needs_reproject:
            src_bbox = transform_bbox(target_bbox, out_crs, src_crs)
        else:
            src_bbox = target_bbox

        # Read from best overview (or full res if no suitable overview)
        native_arr, native_profile = await self._read_native(
            bbox=src_bbox, band_indices=band_indices, ifd_index=ovr_idx
        )

        # Build target grid
        if target_resolution is not None:
            res = target_resolution
        elif needs_reproject:
            # Derive a resolution in the target CRS that preserves the native
            # pixel density: use the native pixel count across the source bbox.
            native_res = self.profile.res[0]
            src_bbox_width = src_bbox.width
            src_bbox_height = src_bbox.height
            n_cols = max(1, round(src_bbox_width / native_res))
            n_rows = max(1, round(src_bbox_height / native_res))
            res_x = target_bbox.width / n_cols
            res_y = target_bbox.height / n_rows
            res = min(res_x, res_y)
        else:
            res = self.profile.res[0]
        out_profile = Profile.for_bbox(
            target_bbox,
            res,
            out_crs,
            count=len(band_indices),
            dtype=self.profile.dtype,
            nodata=self.profile.nodata,
        )

        # Reproject/resample
        transformer = None
        if needs_reproject:
            transformer = Transformer.from_crs(out_crs, src_crs, always_xy=True)

        out_array = resample_nearest(
            native_arr,
            src_transform=native_profile.transform,
            dst_transform=out_profile.transform,
            dst_width=out_profile.width,
            dst_height=out_profile.height,
            nodata=self.profile.nodata,
            transformer=transformer,
        )
        return out_array, out_profile

    async def _read_native(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        ifd_index: int | None = None,
    ) -> tuple[np.ndarray, Profile]:
        """Read at native resolution/CRS, optionally from an overview IFD."""
        ifd_index = ifd_index if ifd_index is not None else self.ifd_index
        ifd = self.tiff.ifds[ifd_index]

        if ifd_index == self.ifd_index:
            ifd_profile = self.profile
        else:
            # Overview IFDs lack geo tags — derive profile from base profile
            scale_x = self.profile.width / ifd.image_width
            scale_y = self.profile.height / ifd.image_height
            ovr_res = (self.profile.res[0] * scale_x, self.profile.res[1] * scale_y)
            ovr_transform = Affine(
                ovr_res[0],
                0,
                float(self.profile.transform.c),
                0,
                -ovr_res[1],
                float(self.profile.transform.f),
            )
            ifd_profile = Profile(
                width=ifd.image_width,
                height=ifd.image_height,
                count=self.profile.count,
                dtype=self.profile.dtype,
                transform=ovr_transform,
                res=ovr_res,
                crs_epsg=self.profile.crs_epsg,
                bounds=self.profile.bounds,
                nodata=self.profile.nodata,
                tile_width=ifd.tile_width,
                tile_height=ifd.tile_height,
            )

        # band_indices are already 0-based (converted by the caller).
        if bbox is None and window is None:
            bbox = ifd_profile.bounds
        if window is None:
            window = Window.from_bbox(ifd_profile, bbox)

        tile_coords = get_intersecting_image_tiles(
            window, ifd.tile_width, ifd.tile_height
        )
        tiles = []
        for i in range(0, len(tile_coords), _TILE_FETCH_BATCH_SIZE):
            batch = tile_coords[i : i + _TILE_FETCH_BATCH_SIZE]
            tiles.extend(
                await self.tiff.fetch_tiles(batch, ifd_index)
            )

        out_array = np.zeros(
            (len(band_indices), window.win_height, window.win_width),
            dtype=ifd_profile.dtype,
        )

        decoded_results = await asyncio.gather(
            *(_decode_tile_concurrently(t) for t in tiles)
        )

        for tx, ty, decoded in decoded_results:
            slices = compute_tile_paste_slices(
                tx=tx,
                ty=ty,
                tile_width=ifd.tile_width,
                tile_height=ifd.tile_height,
                window=window,
            )
            if slices is None:
                continue
            out_rows, out_cols, tile_rows, tile_cols = slices
            arr = np.asarray(decoded).transpose(2, 0, 1)  # (bands, H, W)
            out_array[:, out_rows, out_cols] = arr[band_indices, tile_rows, tile_cols]

        out_profile = ifd_profile.adjust_to_window(window)
        return out_array, out_profile

    def __repr__(self) -> str:
        return f"AsyncGeoTIFF({self.uri}, profile={self.profile})"


def _is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://") or ".s3." in uri or ".s3-" in uri


def _detect_region(uri: str) -> str:
    """Try to extract the AWS region from an S3 HTTPS URL, fall back to env/default.

    Handles virtual-hosted style:
        https://<bucket>.s3.<region>.amazonaws.com/...
        https://<bucket>.s3-<region>.amazonaws.com/...
    And path style:
        https://s3.<region>.amazonaws.com/...
    """
    m = re.search(r"[./]s3[.-]([a-z0-9-]+)\.amazonaws\.com", uri)
    if m:
        return m.group(1)
    return _DEFAULT_REGION


def _extract_key(uri: str) -> str:
    """Extract the object key from a URI, for use with a pre-constructed store."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc
        if ".s3." in host or ".s3-" in host:
            return parsed.path.lstrip("/")
        # Path-style: https://s3.<region>.amazonaws.com/<bucket>/<key>
        parts = parsed.path.lstrip("/").split("/", 1)
        return parts[1] if len(parts) == 2 else ""
    # Local path or other scheme — use path as-is
    return parsed.path or uri


def _build_store(uri: str, **store_kwargs: Any) -> Any:
    """Build an object store rooted at the bucket/host level."""
    local_path = _resolve_local_path(uri)
    if local_path is not None:
        return from_url(local_path.parent.as_uri(), **store_kwargs)
    if _is_s3_uri(uri):
        store_kwargs.setdefault("skip_signature", True)
        store_kwargs.setdefault("region", _detect_region(uri))
    # Root the store at the bucket, not the full object path
    bucket_url = _bucket_url(uri)
    return from_url(bucket_url, **store_kwargs)


def _bucket_url(uri: str) -> str:
    """Extract the bucket-level URL from a full object URI."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        # s3://bucket/key -> s3://bucket
        return f"s3://{parsed.netloc}"
    if parsed.scheme in {"gs", "az"}:
        return f"{parsed.scheme}://{parsed.netloc}"
    if parsed.scheme in {"http", "https"}:
        # https://bucket.s3.region.amazonaws.com/key -> https://bucket.s3.region.amazonaws.com
        return f"{parsed.scheme}://{parsed.netloc}"
    return uri


def _resolve_local_path(uri: str):
    """Return resolved Path if uri is local, else None."""
    from pathlib import Path

    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file") or _is_s3_uri(uri):
        return None
    return Path(parsed.path if parsed.scheme == "file" else uri).resolve()


async def _decode_tile_concurrently(tile: Any) -> tuple[int, int, Any]:
    tx, ty = tile.x, tile.y
    try:
        decoded = await tile.decode()
        return tx, ty, decoded
    except Exception as exc:
        raise RuntimeError(f"Failed to decode tile ({tx}, {ty}): {exc}") from exc
