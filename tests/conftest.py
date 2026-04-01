from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from affine import Affine


def make_meta(width=100, height=100, scale=10.0):
    """Duck-typed object with transform/width/height for window_from_bbox etc."""
    transform = Affine(scale, 0, 0, 0, -scale, height * scale)
    return SimpleNamespace(width=width, height=height, transform=transform)


def make_mock_geotiff(
    width=100,
    height=100,
    scale=10.0,
    count=3,
    tile_width=256,
    tile_height=256,
    dtype=np.dtype("u2"),
    nodata=None,
    crs_epsg=32632,
):
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
