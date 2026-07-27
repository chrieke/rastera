from __future__ import annotations

from typing import TYPE_CHECKING, Any

from async_geotiff import RasterArray, Window
from async_tiff.store import S3Store  # type: ignore[import-untyped]

from .config import set_concurrency, set_warp_strategy
from .merge import merge
from .reader import AsyncGeoTIFF, clear_cache, open, set_cache_size

if TYPE_CHECKING:
    from .index import build_index, open_from_index

__all__ = [
    "RasterArray",
    "AsyncGeoTIFF",
    "S3Store",
    "Window",
    "clear_cache",
    "set_cache_size",
    "set_concurrency",
    "set_warp_strategy",
    "open",
    "merge",
    "build_index",
    "open_from_index",
]

_INDEX_EXPORTS = ("build_index", "open_from_index")

# Hidden from type checkers on purpose: a visible module-level __getattr__ makes
# them accept *any* rastera attribute, which would cost every downstream caller
# typo detection on the public namespace. The TYPE_CHECKING import above already
# declares the lazy names statically.
if not TYPE_CHECKING:
    # The index extra drags in geopandas/pyarrow, which cost more to import than
    # the rest of rastera put together. Resolve them on first attribute access
    # so reader-only callers never pay for them.
    def __getattr__(name: str) -> Any:
        if name in _INDEX_EXPORTS:
            try:
                from . import index
            except ImportError as exc:
                raise ImportError(
                    f"rastera.{name} requires the optional index dependencies; "
                    'install them with `pip install "rastera[index]"`.'
                ) from exc
            return getattr(index, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
