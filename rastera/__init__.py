from __future__ import annotations

from typing import TYPE_CHECKING, Any

from async_geotiff import RasterArray, Window
from async_tiff.store import S3Store  # type: ignore[import-untyped]

from .config import set_concurrency, set_warp_strategy
from .geo import BBox, snapped_grid_for_bbox
from .merge import merge
from .reader import AsyncGeoTIFF, clear_cache, open, set_cache_size

if TYPE_CHECKING:
    # ``as`` marks these re-exported without __all__, which would make
    # `from rastera import *` force the very import they exist to defer.
    from .index import build_index as build_index
    from .index import open_from_index as open_from_index

__all__ = [
    "BBox",
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
    "snapped_grid_for_bbox",
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
                # ImportError over AttributeError: a missing extra is an
                # install problem. Trade-off: `hasattr` propagates it, not False.
                raise ImportError(
                    f"rastera.{name} requires the optional index dependencies; "
                    'install them with `pip install "rastera[index]"`.'
                ) from exc
            return getattr(index, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
