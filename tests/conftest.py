from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray

# The attribute surface a real GeoTIFF offers rastera, plus ``read``. Spec'ing
# the mock to it turns a read outside that contract into an AttributeError
# instead of a Mock that compares unequal, iterates empty and floats to 1.0.
_GEOTIFF_ATTRS = (
    "bounds",
    "count",
    "crs",
    "dtype",
    "height",
    "nodata",
    "overviews",
    "read",
    "res",
    "tile_height",
    "tile_width",
    "transform",
    "width",
)


@pytest.fixture(autouse=True)
def _clear_aws_region(monkeypatch: pytest.MonkeyPatch):
    """Region resolution consults the environment, so a developer's AWS_REGION
    would otherwise decide what the offline tests assert."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)


@pytest.fixture(autouse=True)
def _reset_concurrency():
    """Concurrency is process-wide, so a test that raises it would otherwise
    decide how the next one fans out."""
    yield
    import rastera

    rastera.set_concurrency(merge=1, vrt=1, dimap=1)


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
    crs_epsg: int | None = 32632,
    origin_x: float = 0.0,
    origin_y: float | None = None,
) -> MagicMock:
    """Build a mock async_geotiff.GeoTIFF.

    *origin_y* defaults to the north edge of a scene whose south edge sits on
    zero, so the default grid is the familiar (0, 0, w*scale, h*scale) box.
    """
    if origin_y is None:
        origin_y = height * scale

    gt = MagicMock(spec=_GEOTIFF_ATTRS)
    gt.width = width
    gt.height = height
    gt.count = count
    gt.dtype = dtype
    gt.nodata = nodata
    gt.tile_width = tile_width
    gt.tile_height = tile_height

    gt.transform = Affine(scale, 0, origin_x, 0, -scale, origin_y)
    gt.res = (scale, scale)

    crs_mock = MagicMock()
    crs_mock.to_epsg.return_value = crs_epsg
    gt.crs = crs_mock

    gt.bounds = (
        origin_x,
        origin_y - height * scale,
        origin_x + width * scale,
        origin_y,
    )
    gt.overviews = []

    return gt


def make_raster_array(
    data: np.ndarray[Any, Any], transform: Affine, geotiff: Any
) -> RasterArray:
    """RasterArray over *data*, with width/height/count taken from its shape."""
    return RasterArray(
        data=data,
        mask=None,
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        transform=transform,
        _alpha_band_idx=None,
        _geotiff=geotiff,
    )


def slicing_read(geotiff: Any, full: np.ndarray[Any, Any]):
    """An async ``read(window=...)`` that really slices *full*.

    ``AsyncMock(return_value=...)`` ignores the window it was handed, which
    hides every bug about *which* pixels a read asked for.
    """

    async def _read(window: Any) -> Any:
        data = full[
            :,
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ]
        return make_raster_array(
            data,
            geotiff.transform * Affine.translation(window.col_off, window.row_off),
            geotiff,
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


def spy_read_to_grid(obj: Any) -> list[dict[str, Any]]:
    """Record every ``_read_to_grid`` call, then delegate.

    Which path ran is otherwise invisible from the output — the whole point
    of the snapped grid is that both paths return the same one.
    """
    calls: list[dict[str, Any]] = []
    real = obj._read_to_grid

    async def _wrapped(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return await real(**kwargs)

    obj._read_to_grid = _wrapped
    return calls
