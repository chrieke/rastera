from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, cast

import geopandas as gpd
import obstore
import pyarrow.parquet as pq
from obstore.store import from_url as obstore_from_url
from pyproj import Transformer
from shapely import ops
from shapely.geometry import box

from .reader import (
    AsyncGeoTIFF,
    get_cached_geotiff,
)
from .store import (
    _build_store_with,
    _extract_key,
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
    """Build a geoparquet-ready index from a list of COG URIs.

    Opens each COG to extract structured metadata and fetches the raw header
    bytes needed for zero-network reconstruction via ``open_from_index``.

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
    uris = list(uris)
    if not uris:
        return _empty_geodataframe()
    # Checked even when the caller supplies a store: the header cache below is
    # keyed by object key, so two buckets mirroring a key path would collapse.
    _require_same_bucket(uris, "building an index")
    obs = store if store is not None else _build_obstore(uris[0], **store_kwargs)
    sem = asyncio.Semaphore(concurrency)

    # Fetch header bytes once, then open COGs through cache so async-geotiff
    # reads from memory instead of making a second network request.
    async def _fetch_header(uri: str) -> tuple[str, str, bytes]:
        async with sem:
            key = _extract_key(uri)
            hdr = bytes(await obstore.get_range_async(obs, key, start=0, end=prefetch))
            return uri, key, hdr

    fetched = await asyncio.gather(*(_fetch_header(u) for u in uris))
    cache = {key: hdr for _, key, hdr in fetched}
    cached_store = HeaderCacheStore(obs, cache)

    async def _open_one(uri: str, hdr: bytes) -> tuple[AsyncGeoTIFF, bytes]:
        async with sem:
            try:
                src = await AsyncGeoTIFF.open(
                    uri, store=cached_store, prefetch=prefetch
                )
                return src, hdr
            except Exception as exc:
                hint = ""
                if _resolve_local_path(uri) is not None:
                    hint = " (local files are not supported, use remote URIs)"
                raise RuntimeError(f"Failed to index {uri!r}{hint}") from exc

    results = await asyncio.gather(*(_open_one(u, hdr) for u, _, hdr in fetched))

    rows: dict[str, list[Any]] = {c: [] for c in _INDEX_COLUMNS}
    geometries: list[Any] = []

    for src, hdr in results:
        # Every column but uri/header_bytes is a profile key, so read them off
        # the profile: hand-copying them is how this came to record a
        # band-stack VRT's first source's band count rather than the stack's.
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
        if p["crs_epsg"] is not None and p["crs_epsg"] != 4326:
            t = Transformer.from_crs(p["crs_epsg"], 4326, always_xy=True)
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
        bbox_crs: When omitted, the bbox is assumed to be in the same CRS as
            the index geometry column (EPSG:4326).
        prefetch: Must match the value used when building the index.
        **store_kwargs: Forwarded to ``obstore.store.from_url``.

    Raises:
        ValueError: If the selected rows span more than one bucket/host.
            Narrow the selection with *bbox* or open each bucket separately.
    """
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

    # An index may legitimately span buckets (rows concatenated from several
    # builds), but a single open pass cannot: one store, and a header cache
    # keyed by object key. Reject rather than serve one file's bytes as another's.
    _require_same_bucket(uris, "opening from an index")

    shared_store = (
        store if store is not None else _build_obstore(uris[0], **store_kwargs)
    )
    keys = [_extract_key(u) for u in uris]

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
    """Read a geoparquet index, optionally filtering spatially.

    When *bbox* is provided, the metadata columns are read and filtered first,
    then ``header_bytes`` is streamed in batches and only the matched rows are
    kept. That column dominates the file — one prefetch window per COG, 32 KiB
    by default — so materializing it whole costs roughly its own size on top of
    the result, however few rows the bbox selects. Measured on a 131 MB header
    column selecting 2 rows: 429 MB peak RSS reading the column at once against
    307 MB streaming it, so the saving tracks the column and reaches GBs on a
    100k-COG index.

    The bytes still have to come off disk. geopandas writes a single row group
    at any realistic index size, so there is no row-group boundary to skip past;
    this bounds what is resident, not what is read.
    """
    if bbox is None:
        return gpd.read_parquet(path)  # type: ignore[reportUnknownMemberType]

    # Closed rather than left to the GC: a process opening index after index
    # would otherwise sit on a file handle per call until collection.
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


# Rows of ``header_bytes`` held in memory at once while picking out the matched
# ones. pyarrow's own default is 65536, which at a 32 KiB prefetch window is 2 GB
# a batch — the thing this streaming exists to avoid. 1024 puts a batch at ~32 MB.
_HEADER_BATCH_ROWS = 1024


def _take_header_bytes(pf: pq.ParquetFile, row_indices: Sequence[int]) -> list[bytes]:
    """The ``header_bytes`` of *row_indices*, in that order, read in batches.

    *row_indices* are positions into the file's row order, which is what
    ``_read_geoparquet``'s ``reset_index(drop=True)`` makes the filtered frame's
    index. Batches arrive in that same order, so each one covers a contiguous
    span of positions and only the wanted rows are retained.
    """
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
        transformer = Transformer.from_crs(bbox_crs, gdf.crs.to_epsg(), always_xy=True)
        query_geom = ops.transform(transformer.transform, query_geom)

    result = gdf[gdf.intersects(query_geom)]
    assert isinstance(result, gpd.GeoDataFrame)
    return result


def _build_obstore(uri: str, **store_kwargs: Any) -> Any:
    return _build_store_with(uri, obstore_from_url, **store_kwargs)


def _empty_geodataframe() -> gpd.GeoDataFrame:
    # A fresh list per column: one shared list would alias all eleven.
    return gpd.GeoDataFrame(
        {c: [] for c in _INDEX_COLUMNS}, geometry=[], crs="EPSG:4326"
    )
