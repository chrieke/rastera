from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from affine import Affine


def make_meta(
    width: int = 100, height: int = 100, scale: float = 10.0
) -> SimpleNamespace:
    """Duck-typed object with transform/width/height for window_from_bbox etc."""
    transform = Affine(scale, 0, 0, 0, -scale, height * scale)
    return SimpleNamespace(width=width, height=height, transform=transform)


def make_mock_geotiff(
    width: int = 100,
    height: int = 100,
    scale: float = 10.0,
    count: int = 3,
    tile_width: int = 256,
    tile_height: int = 256,
    dtype: np.dtype[Any] = np.dtype("u2"),
    nodata: float | None = None,
    crs_epsg: int = 32632,
) -> MagicMock:
    """Build a mock async_geotiff.GeoTIFF."""
    gt = MagicMock()
    gt.width = width
    gt.height = height
    gt.count = count
    gt.dtype = dtype
    gt.nodata = nodata
    gt.tile_width = tile_width
    gt.tile_height = tile_height

    transform = Affine(scale, 0, 0, 0, -scale, height * scale)
    gt.transform = transform
    gt.res = (scale, scale)

    crs_mock = MagicMock()
    crs_mock.to_epsg.return_value = crs_epsg
    gt.crs = crs_mock

    gt.bounds = (0, 0, width * scale, height * scale)
    gt.overviews = []

    return gt


def slicing_read(geotiff: Any, full: np.ndarray[Any, Any]):
    """An async ``read(window=...)`` that really slices *full*.

    ``AsyncMock(return_value=...)`` ignores the window it was handed, which
    hides every bug about *which* pixels a read asked for.
    """
    from async_geotiff import RasterArray

    async def _read(window: Any) -> Any:
        data = full[
            :,
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ]
        return RasterArray(
            data=data,
            mask=None,
            width=data.shape[2],
            height=data.shape[1],
            count=data.shape[0],
            transform=geotiff.transform
            * Affine.translation(window.col_off, window.row_off),
            _alpha_band_idx=None,
            _geotiff=geotiff,
        )

    return _read


def spy_read_native(obj: Any) -> list[dict[str, Any]]:
    """Record the kwargs of every ``_read_native`` call, then delegate.

    Which bbox and which overview reach ``_read_native`` is the contract
    between the grid-choosing callers and the warp; nothing else observes it.
    """
    calls: list[dict[str, Any]] = []
    real = obj._read_native

    async def _wrapped(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return await real(**kwargs)

    obj._read_native = _wrapped
    return calls
