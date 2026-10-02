from __future__ import annotations

from typing import TYPE_CHECKING, Any

from async_geotiff import RasterArray, Window
from async_tiff.store import S3Store  # type: ignore[import-untyped]

from .config import set_concurrency, set_warp_strategy
from .geo import BBox, WindowOutOfRangeError, snapped_grid_for_bbox
from .merge import merge
from .profile import RasterProfile
from .reader import AsyncGeoTIFF, clear_cache, open, set_cache_size
from .resampling import ResamplingMethod

if TYPE_CHECKING:
    # ``as`` marks these re-exported without __all__, which would make
    # `from rastera import *` force the very import they exist to defer.
    from .index import build_index as build_index
    from .index import open_from_index as open_from_index

__all__ = [
    "BBox",
    "RasterArray",
    "AsyncGeoTIFF",
    "RasterProfile",
    "ResamplingMethod",
    "S3Store",
    "Window",
    # A bbox missing the image is ordinary over a tile set, so it can be caught
    # apart from the argument errors, which are ValueErrors too.
    "WindowOutOfRangeError",
    "clear_cache",
    "set_cache_size",
    "set_concurrency",
    "set_warp_strategy",
    "open",
    "merge",
    "snapped_grid_for_bbox",
]

_INDEX_EXPORTS = ("build_index", "open_from_index")

# The index extra imports geopandas and pyarrow, which cost more than the rest
# of rastera, so those names resolve on first access. Hidden from type checkers,
# which would otherwise accept any rastera attribute.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name in _INDEX_EXPORTS:
            try:
                from . import index
            except ImportError as exc:
                # An install problem, so not AttributeError, though `hasattr`
                # then raises rather than returning False.
                raise ImportError(
                    f"rastera.{name} requires the optional index dependencies; "
                    'install them with `pip install "rastera[index]"`.'
                ) from exc
            return getattr(index, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
