from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, cast

import geopandas as gpd
import obstore
import pyarrow.parquet as pq
import shapely
from obstore.store import HTTPStore as ObstoreHTTPStore
from obstore.store import from_url as obstore_from_url
from pyproj import CRS
from shapely import ops
from shapely.geometry import MultiPolygon, box

from .config import _gather_bounded
from .geo import _transformer
from .reader import (
    AsyncGeoTIFF,
    get_cached_geotiff,
)
from .store import (
    _build_store_with,
    _extract_key,
    _is_local_descriptor,
    _require_same_bucket,
    _resolve_local_path,
)

# Written by build_index, read back by open_from_index; the geometry column is
# added separately by GeoDataFrame.
_INDEX_COLUMNS = (
    "uri",
    "header_bytes",
    "crs_epsg",
    "width",
    "height",
    "count",
    "res_x",
    "res_y",
    "dtype",
    "nodata",
    "overviews",
)


async def build_index(
    uris: Sequence[str],
    *,
    store: Any = None,
    prefetch: int = 32768,
    concurrency: int = 100,
    **store_kwargs: Any,
) -> gpd.GeoDataFrame:
    """Build a geoparquet-ready index of COG headers, for ``open_from_index``.

    Args:
        store: A pre-constructed *obstore* store for connection reuse — not the
            async-tiff store ``rastera.open`` takes.
        prefetch: Header bytes stored per file.
        **store_kwargs: Forwarded to ``obstore.store.from_url``.

    Returns:
        A GeoDataFrame with geometry in EPSG:4326. Write with
        ``gdf.to_parquet(path)`` for geoparquet.

    Raises:
        ValueError: If *uris* span more than one bucket/host. Index each
            bucket separately and concatenate the frames.
    """
    _check_concurrency(concurrency)
    uris = list(uris)
    if not uris:
        return _empty_geodataframe()
    # Even with the caller's store: the header cache is keyed by object key.
    _require_same_bucket(uris, "building an index")
    obs = store if store is not None else _build_obstore(uris[0], **store_kwargs)

    # Fetch each header once, then open through it.
    async def _fetch_header(uri: str) -> tuple[str, str, bytes]:
        key = _extract_key(uri)
        try:
            hdr = await obstore.get_range_async(obs, key, start=0, end=prefetch)
        except Exception as exc:
            raise RuntimeError(f"Failed to index {uri!r}") from exc
        return uri, key, bytes(hdr)

    fetched = await _gather_bounded(concurrency, [_fetch_header(u) for u in uris])
    cache = {key: hdr for _, key, hdr in fetched}
    cached_store = HeaderCacheStore(obs, cache)

    async def _open_one(uri: str, hdr: bytes) -> tuple[AsyncGeoTIFF, bytes]:
        try:
            # store_kwargs too: a VRT or DIMAP fetches its descriptor with
            # them, not through the store.
            src = await AsyncGeoTIFF.open(
                uri,
                store=None if _is_local_descriptor(uri) else cached_store,
                prefetch=prefetch,
                cache=_resolve_local_path(uri) is None,
                **store_kwargs,
            )
            return src, hdr
        except Exception as exc:
            raise RuntimeError(f"Failed to index {uri!r}") from exc

    try:
        results = await _gather_bounded(
            concurrency, [_open_one(u, hdr) for u, _, hdr in fetched]
        )
    finally:
        cache.clear()  # see open_from_index

    rows: dict[str, list[Any]] = {c: [] for c in _INDEX_COLUMNS}
    geometries: list[Any] = []

    for src, hdr in results:
        p = src.profile
        rows["uri"].append(src.uri)
        rows["header_bytes"].append(hdr)
        rows["crs_epsg"].append(p["crs_epsg"])
        rows["width"].append(p["width"])
        rows["height"].append(p["height"])
        rows["count"].append(p["count"])
        rows["res_x"].append(p["res"][0])
        rows["res_y"].append(p["res"][1])
        rows["dtype"].append(p["dtype"])
        rows["nodata"].append(p["nodata"])
        rows["overviews"].append(json.dumps(p["overviews"]))
        b = p["bounds"]
        geom = box(b.minx, b.miny, b.maxx, b.maxy)
        crs = p["crs_epsg"] if p["crs_epsg"] is not None else p["crs"]
        if crs != 4326:
            geom = _reproject(geom, crs, 4326)
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

    The index pins each file's header, so rebuild it when a file is rewritten
    in place: an old index reads the new pixels at the old georeference.

    Args:
        bbox_crs: When omitted, the bbox is assumed to be in the same CRS as
            the index geometry column (EPSG:4326).
        prefetch: Must match the value used when building the index.
        **store_kwargs: Forwarded to ``obstore.store.from_url``.

    Raises:
        ValueError: If the selected rows span more than one bucket/host.
            Narrow the selection with *bbox* or open each bucket separately.
    """
    _check_concurrency(concurrency)
    if isinstance(gdf_or_path, str):
        gdf = _read_geoparquet(gdf_or_path, bbox=bbox, bbox_crs=bbox_crs)
    else:
        gdf = gdf_or_path
        if bbox is not None:
            gdf = _filter_gdf(gdf, bbox, bbox_crs)

    if len(gdf) == 0:
        return []

    uris: list[str] = gdf["uri"].tolist()  # type: ignore[reportUnknownMemberType]
    headers: list[bytes] = gdf["header_bytes"].tolist()  # type: ignore[reportUnknownMemberType]

    # One store and a cache keyed by object key, though an index may span
    # buckets.
    _require_same_bucket(uris, "opening from an index")

    shared_store = (
        store if store is not None else _build_obstore(uris[0], **store_kwargs)
    )
    keys = [_extract_key(u) for u in uris]

    cache = dict(zip(keys, headers))
    cached_store = HeaderCacheStore(shared_store, cache)

    async def _open_one(uri: str) -> AsyncGeoTIFF:
        cached_gt = get_cached_geotiff(uri)
        if cached_gt is not None:
            return AsyncGeoTIFF(uri, cached_gt)
        # Not cached for a local file: the header comes from the index, and
        # the file may have been rewritten since, so it would land under
        # the new version's cache key.
        return await AsyncGeoTIFF.open(
            uri,
            store=None if _is_local_descriptor(uri) else cached_store,
            prefetch=prefetch,
            cache=_resolve_local_path(uri) is None,
            **store_kwargs,
        )

    try:
        return await _gather_bounded(concurrency, [_open_one(u) for u in uris])
    finally:
        # Parsed now, and later reads fall through to the inner store. Left in
        # place, every header the reader's LRU keeps held this store, and with
        # it every row's bytes: about 3 GB for 100k rows at the default
        # prefetch. On a failed open too, as the opens before it are cached.
        cache.clear()


class HeaderCacheStore:
    """An obspec store serving byte ranges inside a cached header from memory,
    and the rest from *inner*."""

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
        return bytes(
            await obstore.get_range_async(
                self._inner,
                path,
                start=start,
                end=end,
                length=length,
            )
        )

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

        if ends is None:
            if lengths is None:
                raise ValueError("Either ends or lengths must be provided")
            resolved_ends = [s + length for s, length in zip(starts, lengths)]
        else:
            resolved_ends = list(ends)

        for i, s in enumerate(starts):
            e = resolved_ends[i]
            if cached is not None and e <= len(cached):
                results[i] = cached[s:e]
            else:
                uncached_indices.append(i)
                uncached_starts.append(s)
                uncached_ends.append(e)

        if uncached_indices:
            fetched = await obstore.get_ranges_async(
                self._inner,
                path,
                starts=uncached_starts,
                ends=uncached_ends,
            )
            for idx, data in zip(uncached_indices, fetched):
                results[idx] = bytes(data)

        return cast(list[bytes], results)


# ---- Internal helpers ----


def _read_geoparquet(
    path: str,
    bbox: tuple[float, float, float, float] | None = None,
    bbox_crs: int | None = None,
) -> gpd.GeoDataFrame:
    """Read a geoparquet index, optionally only the rows meeting *bbox*.

    With a bbox, the metadata columns are filtered first and ``header_bytes``,
    one prefetch window per COG, is streamed in batches keeping only the
    matched rows: 307 MB peak RSS against 429 MB for a 131 MB column. That
    bounds what is resident, not what is read, since geopandas writes one row
    group.
    """
    if bbox is None:
        return gpd.read_parquet(path)  # type: ignore[reportUnknownMemberType]

    with pq.ParquetFile(path) as pf:
        all_names: list[str] = pf.schema_arrow.names  # type: ignore[reportUnknownMemberType]
        meta_cols = [c for c in all_names if c != "header_bytes"]
        gdf_meta = gpd.read_parquet(  # type: ignore[reportUnknownMemberType]
            path, columns=meta_cols
        ).reset_index(drop=True)

        filtered = _filter_gdf(gpd.GeoDataFrame(gdf_meta), bbox, bbox_crs)
        if len(filtered) == 0:
            return filtered

        row_indices: list[int] = filtered.index.tolist()  # type: ignore[reportUnknownMemberType]
        filtered = filtered.copy()
        filtered["header_bytes"] = _take_header_bytes(pf, row_indices)

    return filtered


# ~32 MB a batch at the default prefetch; pyarrow's 65536 rows would be 2 GB.
_HEADER_BATCH_ROWS = 1024


def _take_header_bytes(pf: pq.ParquetFile, row_indices: Sequence[int]) -> list[bytes]:
    """The ``header_bytes`` of *row_indices*, positions in the file's row
    order, read in batches."""
    wanted = set(row_indices)
    found: dict[int, bytes] = {}
    offset = 0
    batches = pf.iter_batches(  # type: ignore[reportUnknownMemberType]
        batch_size=_HEADER_BATCH_ROWS, columns=["header_bytes"]
    )
    for batch in batches:
        n: int = batch.num_rows  # type: ignore[reportUnknownMemberType]
        # intersection() over a range iterates it without materialising a set.
        for i in wanted.intersection(range(offset, offset + n)):
            found[i] = batch.column(0)[i - offset].as_py()  # type: ignore[reportUnknownMemberType]
        offset += n
        if len(found) == len(wanted):
            break
    return [found[i] for i in row_indices]


def _filter_gdf(
    gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    bbox_crs: int | None = None,
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox
    query_geom = box(minx, miny, maxx, maxy)

    if bbox_crs is not None and gdf.crs is not None and gdf.crs.to_epsg() != bbox_crs:
        query_geom = _reproject(query_geom, bbox_crs, gdf.crs)

    result = gdf[gdf.intersects(query_geom)]
    assert isinstance(result, gpd.GeoDataFrame)
    return result


def _check_concurrency(concurrency: int) -> None:
    if (
        not isinstance(concurrency, int)
        or isinstance(concurrency, bool)
        or concurrency < 1
    ):
        raise ValueError(f"concurrency must be int >= 1, got {concurrency!r}")


def _build_obstore(uri: str, **store_kwargs: Any) -> Any:
    return _build_store_with(uri, obstore_from_url, ObstoreHTTPStore, **store_kwargs)


def _empty_geodataframe() -> gpd.GeoDataFrame:
    # A fresh list per column: one shared list would alias all eleven.
    return gpd.GeoDataFrame(
        {c: [] for c in _INDEX_COLUMNS}, geometry=[], crs="EPSG:4326"
    )


# Segments per side when densifying a box before reprojecting it. At 20, a
# 110 km UTM tile's densified footprint misses 0.001% of its true area.
_EDGE_SEGMENTS = 20


def _reproject(geom: Any, from_crs: int | CRS, to_crs: int | CRS) -> Any:
    """*geom*, a box, reprojected with its edges densified, since a straight
    UTM edge is a curve in lon/lat: corners alone cut ~400 m off a 110 km tile
    at 60N.

    Into lon/lat, a box across the antimeridian or around a pole gets its
    envelope instead, split at 180° or over every longitude; reprojected as a
    ring it matched no query inside it.
    """
    t = _transformer(from_crs, to_crs)
    if t.target_crs is not None and t.target_crs.is_geographic:
        bounds = t.transform_bounds(*geom.bounds, densify_pts=_EDGE_SEGMENTS + 1)
        minx, miny, maxx, maxy = bounds
        if all(math.isfinite(v) for v in bounds):
            if miny <= -90 or maxy >= 90:
                return box(-180, miny, 180, maxy)
            if minx > maxx:
                east, west = box(minx, miny, 180, maxy), box(-180, miny, maxx, maxy)
                return MultiPolygon([east, west])
    # A point or zero-height query box has no edge to bend: segmentize rejects
    # the zero step, or empties the flat polygon, which then matches nothing.
    if geom.area > 0:
        minx, miny, maxx, maxy = geom.bounds
        step = max(maxx - minx, maxy - miny) / _EDGE_SEGMENTS
        geom = shapely.segmentize(geom, step)
    return ops.transform(t.transform, geom)
