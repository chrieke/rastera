from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

import geopandas as gpd
import obstore
import pyarrow.parquet as pq
from obstore.store import from_url as obstore_from_url
from pyproj import Transformer
from shapely import ops
from shapely.geometry import box

from .reader import (
    AsyncGeoTIFF,
    _bucket_url,
    _detect_region,
    _extract_key,
    _is_s3_uri,
    _resolve_local_path,
    get_cached_geotiff,
)




class HeaderCacheStore:
    """Obspec-compatible store wrapper that serves pre-fetched header bytes from cache.

    For byte ranges that fall within the cached region, data is served from memory.
    For ranges beyond the cache (tile data), requests are delegated to the inner store
    via ``obstore`` (which can call both native Rust stores and Python stores).
    """

    def __init__(self, inner: Any, cache: dict[str, bytes]):
        self._inner = inner
        self._cache = cache

    async def get_range_async(
        self,
        path: str,
        *,
        start: int,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes:
        if end is not None:
            actual_end = end
        elif length is not None:
            actual_end = start + length
        else:
            actual_end = None

        cached = self._cache.get(path)
        if cached is not None and actual_end is not None and actual_end <= len(cached):
            return cached[start:actual_end]
        return bytes(await obstore.get_range_async(
            self._inner, path, start=start, end=end, length=length,
        ))

    async def get_ranges_async(
        self,
        path: str,
        *,
        starts: Sequence[int],
        ends: Sequence[int] | None = None,
        lengths: Sequence[int] | None = None,
    ) -> list[bytes]:
        cached = self._cache.get(path)
        results: list[bytes | None] = [None] * len(starts)
        uncached_indices: list[int] = []
        uncached_starts: list[int] = []
        uncached_ends: list[int] = []

        for i, s in enumerate(starts):
            e = ends[i] if ends is not None else s + lengths[i]
            if cached is not None and e <= len(cached):
                results[i] = cached[s:e]
            else:
                uncached_indices.append(i)
                uncached_starts.append(s)
                uncached_ends.append(e)

        if uncached_indices:
            fetched = await obstore.get_ranges_async(
                self._inner, path, starts=uncached_starts, ends=uncached_ends,
            )
            for idx, data in zip(uncached_indices, fetched):
                results[idx] = bytes(data)

        return results  # type: ignore[return-value]


async def build_index(
    uris: Sequence[str],
    *,
    store: Any = None,
    prefetch: int = 32768,
    concurrency: int = 100,
    **store_kwargs: Any,
) -> gpd.GeoDataFrame:
    """Build a geoparquet-ready index from a list of COG URIs.

    Opens each COG to extract structured metadata and fetches the raw header
    bytes needed for zero-network reconstruction via ``open_from_index``.

    Args:
        uris: COG URIs to index.
        store: Optional pre-constructed object store for connection reuse.
        prefetch: Number of header bytes to store per file (default 32KB).
        concurrency: Maximum number of concurrent file opens (default 100).
        **store_kwargs: Extra kwargs forwarded to ``async_tiff.store.from_url``.

    Returns:
        A GeoDataFrame with geometry in EPSG:4326.
        Write with ``gdf.to_parquet(path)`` for geoparquet.
    """
    uris = list(uris)
    if not uris:
        return _empty_geodataframe()
    obs = _build_obstore(uris[0], **store_kwargs)
    sem = asyncio.Semaphore(concurrency)

    # Fetch header bytes once, then open COGs through cache so async-geotiff
    # reads from memory instead of making a second network request.
    async def _fetch_header(uri: str) -> tuple[str, str, bytes]:
        async with sem:
            key = _obstore_key(uri)
            hdr = bytes(await obstore.get_range_async(obs, key, start=0, end=prefetch))
            return uri, key, hdr

    fetched = await asyncio.gather(*(_fetch_header(u) for u in uris))
    cache = {key: hdr for _, key, hdr in fetched}
    cached_store = HeaderCacheStore(obs, cache)

    async def _open_one(uri: str, hdr: bytes) -> tuple[AsyncGeoTIFF, bytes]:
        async with sem:
            try:
                src = await AsyncGeoTIFF.open(uri, store=cached_store, prefetch=prefetch)
                return src, hdr
            except Exception as exc:
                hint = ""
                if _resolve_local_path(uri) is not None:
                    hint = " (local files are not supported, use remote URIs)"
                raise RuntimeError(f"Failed to index {uri!r}{hint}") from exc

    results = await asyncio.gather(*(_open_one(u, hdr) for u, _, hdr in fetched))

    rows: dict[str, list] = {
        "uri": [],
        "header_bytes": [],
        "crs_epsg": [],
        "width": [],
        "height": [],
        "count": [],
        "res_x": [],
        "res_y": [],
        "dtype": [],
        "nodata": [],
        "overviews": [],
    }
    geometries = []

    for src, hdr in results:
        p = src.profile
        rows["uri"].append(src.uri)
        rows["header_bytes"].append(hdr)
        rows["crs_epsg"].append(p.crs_epsg)
        rows["width"].append(p.width)
        rows["height"].append(p.height)
        rows["count"].append(p.count)
        rows["res_x"].append(p.res[0])
        rows["res_y"].append(p.res[1])
        rows["dtype"].append(str(p.dtype))
        rows["nodata"].append(p.nodata)
        rows["overviews"].append(json.dumps(p.overviews or []))
        # Reproject bounds to EPSG:4326 for a consistent geometry column
        b = p.bounds
        geom = box(b.minx, b.miny, b.maxx, b.maxy)
        if p.crs_epsg is not None and p.crs_epsg != 4326:
            t = Transformer.from_crs(p.crs_epsg, 4326, always_xy=True)
            geom = ops.transform(t.transform, geom)
        geometries.append(geom)

    return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")


async def open_from_index(
    gdf_or_path: gpd.GeoDataFrame | str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    bbox_crs: int | None = None,
    store: Any = None,
    prefetch: int = 32768,
    concurrency: int = 100,
    **store_kwargs: Any,
) -> list[AsyncGeoTIFF]:
    """Open COGs using pre-fetched headers from a geoparquet index.

    When *bbox* is provided and *gdf_or_path* is a file path, only the
    matching rows are loaded into memory — header bytes for non-matching
    files are never read.

    Args:
        gdf_or_path: A GeoDataFrame or path to a ``.parquet`` geoparquet file.
        bbox: Optional (minx, miny, maxx, maxy) spatial filter.
        bbox_crs: EPSG code of the bbox. When omitted, the bbox is assumed
            to be in the same CRS as the index geometry column (EPSG:4326).
        store: Optional pre-constructed object store.
        prefetch: Must match the prefetch value used when building the index.
        concurrency: Maximum number of concurrent file opens (default 100).
        **store_kwargs: Extra kwargs forwarded to ``obstore.store.from_url``.

    Returns:
        List of AsyncGeoTIFF instances ready for ``.read()`` calls.
    """
    if isinstance(gdf_or_path, str):
        gdf = _read_geoparquet(gdf_or_path, bbox=bbox, bbox_crs=bbox_crs)
    else:
        gdf = gdf_or_path
        if bbox is not None:
            gdf = _filter_gdf(gdf, bbox, bbox_crs)

    if len(gdf) == 0:
        return []

    uris = gdf["uri"].tolist()
    headers = gdf["header_bytes"].tolist()

    if store is not None:
        shared_store = store
        keys = [_extract_key(u) for u in uris]
    else:
        shared_store = _build_obstore(uris[0], **store_kwargs)
        keys = [_obstore_key(u) for u in uris]

    cache = dict(zip(keys, headers))
    cached_store = HeaderCacheStore(shared_store, cache)
    sem = asyncio.Semaphore(concurrency)

    async def _open_one(uri: str) -> AsyncGeoTIFF:
        async with sem:
            cached_gt = get_cached_geotiff(uri)
            if cached_gt is not None:
                return AsyncGeoTIFF(uri, cached_gt)
            return await AsyncGeoTIFF.open(uri, store=cached_store, prefetch=prefetch)

    return list(await asyncio.gather(*(_open_one(u) for u in uris)))


def _read_geoparquet(
    path: str,
    bbox: tuple[float, float, float, float] | None = None,
    bbox_crs: int | None = None,
) -> gpd.GeoDataFrame:
    """Read a geoparquet index, optionally filtering spatially.

    When *bbox* is provided, reads only metadata columns first for spatial
    filtering, then loads ``header_bytes`` only for matched rows.
    """
    if bbox is None:
        return gpd.read_parquet(path)

    meta_cols = [c for c in pq.read_schema(path).names if c != "header_bytes"]
    gdf_meta = gpd.read_parquet(path, columns=meta_cols).reset_index(drop=True)

    filtered = _filter_gdf(gdf_meta, bbox, bbox_crs)
    if len(filtered) == 0:
        return filtered

    row_indices = filtered.index.tolist()
    header_col = pq.read_table(path, columns=["header_bytes"]).column("header_bytes")
    filtered = filtered.copy()
    filtered["header_bytes"] = header_col.take(row_indices).to_pylist()

    return filtered


def _filter_gdf(
    gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    bbox_crs: int | None = None,
) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame by bounding box intersection."""
    minx, miny, maxx, maxy = bbox
    query_geom = box(minx, miny, maxx, maxy)

    if bbox_crs is not None and gdf.crs is not None and gdf.crs.to_epsg() != bbox_crs:
        transformer = Transformer.from_crs(bbox_crs, gdf.crs.to_epsg(), always_xy=True)
        query_geom = ops.transform(transformer.transform, query_geom)

    return gdf[gdf.intersects(query_geom)]


def _obstore_key(uri: str) -> str:
    """Extract the object key for use with an obstore rooted at bucket level.

    Unlike ``_extract_key`` (used with async-tiff stores), this does not
    distinguish virtual-hosted from path-style S3 HTTP URLs because
    obstore handles that internally when the store is rooted via
    ``_bucket_url``.
    """
    local_path = _resolve_local_path(uri)
    if local_path is not None:
        return local_path.name
    parsed = urlparse(uri)
    if parsed.scheme in ("s3", "gs", "az"):
        return parsed.path.lstrip("/")
    if parsed.scheme in ("http", "https"):
        host = parsed.netloc
        path = parsed.path.lstrip("/")
        # Path-style: https://s3.<region>.amazonaws.com/<bucket>/<key>
        # The store is rooted at the host, so strip the bucket prefix to
        # match what _extract_key returns (and what AsyncGeoTIFF.open uses).
        if ".s3." not in host and ".s3-" not in host:
            parts = path.split("/", 1)
            return parts[1] if len(parts) == 2 else ""
        return path
    return parsed.path or uri


def _build_obstore(uri: str, **store_kwargs: Any) -> Any:
    """Build an obstore-compatible object store for the given URI.

    For S3/GCS/AZ URIs, the store is rooted at the bucket level.
    For HTTP and local paths, the store is rooted at the file URL.
    """
    local_path = _resolve_local_path(uri)
    if local_path is not None:
        return obstore_from_url(local_path.parent.as_uri(), **store_kwargs)
    if _is_s3_uri(uri):
        store_kwargs.setdefault("skip_signature", True)
        store_kwargs.setdefault("region", _detect_region(uri))
    return obstore_from_url(_bucket_url(uri), **store_kwargs)


def _empty_geodataframe() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "uri": [],
            "header_bytes": [],
            "crs_epsg": [],
            "width": [],
            "height": [],
            "count": [],
            "res_x": [],
            "res_y": [],
            "dtype": [],
            "nodata": [],
            "overviews": [],
        },
        geometry=[],
        crs="EPSG:4326",
    )
