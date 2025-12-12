from unittest.mock import MagicMock

import numpy as np
from affine import Affine

from rastera.geo import BBox
from rastera.meta import Profile


def make_profile(width=100, height=100, scale=10.0, count=1, dtype=np.uint8):
    """North-up profile with origin at (0, height*scale)."""
    transform = Affine(scale, 0, 0, 0, -scale, height * scale)
    return Profile(
        width=width,
        height=height,
        count=count,
        dtype=np.dtype(dtype),
        transform=transform,
        res=(scale, scale),
        crs_epsg=32632,
        bounds=BBox(0, 0, width * scale, height * scale),
    )


def make_mock_ifd(
    width=100, height=100, scale=10.0, bands=3, tile_size=256,
    sample_format=1, bits_per_sample=16, nodata_tag=None,
    crs_epsg=32632, geographic_type=None,
):
    """Build a mock IFD suitable for Profile.from_ifd() and AsyncGeoTIFF tests."""
    ifd = MagicMock()
    ifd.image_width = width
    ifd.image_height = height
    ifd.samples_per_pixel = bands
    ifd.tile_width = tile_size
    ifd.tile_height = tile_size
    ifd.model_pixel_scale = [scale, scale, 0]
    ifd.model_tiepoint = [0, 0, 0, 0, height * scale, 0]
    ifd.bits_per_sample = [bits_per_sample]
    ifd.sample_format = [sample_format]
    ifd.geo_key_directory = MagicMock(
        projected_type=crs_epsg, geographic_type=geographic_type,
    )
    ifd.other_tags = {}
    if nodata_tag is not None:
        from rastera.meta import _GDAL_NODATA_TAG
        ifd.other_tags[_GDAL_NODATA_TAG] = nodata_tag
    return ifd
