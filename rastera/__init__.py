from __future__ import annotations

from async_geotiff import RasterArray, Window
from async_tiff.store import S3Store  # type: ignore[import-untyped]

from .merge import merge
from .reader import AsyncGeoTIFF, clear_cache, open, set_cache_size

__all__ = [
    "RasterArray",
    "AsyncGeoTIFF",
    "S3Store",
    "Window",
    "clear_cache",
    "set_cache_size",
    "open",
    "merge",
]

try:
    import geopandas  # noqa: F401
    import pyarrow  # noqa: F401
except ImportError:
    pass
else:
    from .index import build_index, open_from_index

    __all__ += ["build_index", "open_from_index"]
