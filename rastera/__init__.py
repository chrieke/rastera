from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, overload

from async_tiff.store import S3Store

from .reader import (
    AsyncGeoTIFF,
    _build_store,
    clear_cache,
    set_cache_size,
)
from .merge import merge_cogs

__all__ = [
    "AsyncGeoTIFF",
    "S3Store",
    "clear_cache",
    "set_cache_size",
    "open",
    "merge",
]

try:
    import geopandas  # noqa: F401
    import obstore  # noqa: F401
    import pyarrow  # noqa: F401
except ImportError:
    pass
else:
    from .index import build_index, open_from_index
    __all__ += ["build_index", "open_from_index"]


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

    uris = list(uri)
    if not uris:
        return []

    # Build a shared store from the first URI if none provided
    if store is None:
        store = _build_store(uris[0], **store_kwargs)

    return list(await asyncio.gather(
        *(AsyncGeoTIFF.open(u, store=store, prefetch=prefetch, cache=cache)
          for u in uris)
    ))


async def merge(
    sources: Sequence[AsyncGeoTIFF],
    *,
    bbox: tuple[float, float, float, float],
    bbox_crs: int | None = None,
    band_indices: Sequence[int] | None = None,
    fill_value: int | float = 0,
    target_crs: int | None = None,
    target_resolution: float | None = None,
    method: Literal["first", "last"] = "first",
) -> tuple:
    """
    Rasterio-style helper: read a bbox mosaic from multiple already-open sources.

    Pass in the `AsyncGeoTIFF` instances returned by :func:`open`. Any caching
    of headers/metadata happens on those `AsyncGeoTIFF` instances.

    When *target_crs* or *target_resolution* is set (or when the inputs have
    different CRS / resolution), each COG is individually reprojected into the
    target grid before merging, allowing cross-CRS merges (e.g. adjacent UTM
    zones).

    Args:
        method: Overlap strategy when multiple sources cover the same pixel.
            ``"first"`` (default) keeps the first valid pixel, matching
            ``rasterio.merge`` behaviour. ``"last"`` lets later sources
            overwrite earlier ones.
    """
    return await merge_cogs(
        sources,
        bbox=bbox,
        bbox_crs=bbox_crs,
        band_indices=band_indices,
        fill_value=fill_value,
        target_crs=target_crs,
        target_resolution=target_resolution,
        method=method,
    )
