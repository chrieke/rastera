"""Unit tests for the module-level IFD parsing helpers in meta.py."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from affine import Affine

from rastera.meta import (
    Profile,
    _GDAL_NODATA_TAG,
    _dtype_from_ifd,
    _nodata_from_ifd,
    _transform_from_ifd,
)
from tests.conftest import make_mock_ifd


# ── _transform_from_ifd ─────────────────────────────────────────────────


class TestTransformFromIfd:
    def test_standard_north_up(self):
        ifd = make_mock_ifd(width=100, height=200, scale=10.0)
        t = _transform_from_ifd(ifd)
        assert isinstance(t, Affine)
        assert t.a == 10.0  # scale_x
        assert t.e == -10.0  # -scale_y
        assert t.c == 0.0  # origin x
        assert t.f == 2000.0  # origin y = height * scale

    def test_nonzero_tiepoint_offset(self):
        ifd = make_mock_ifd()
        # Tiepoint at pixel (5, 10) maps to geo (500, 1500)
        ifd.model_tiepoint = [5, 10, 0, 500, 1500, 0]
        ifd.model_pixel_scale = [10.0, 10.0, 0]
        t = _transform_from_ifd(ifd)
        # origin_x = x0 - i * scale_x = 500 - 5*10 = 450
        assert t.c == 450.0
        # origin_y = y0 + j * scale_y = 1500 + 10*10 = 1600
        assert t.f == 1600.0

    def test_missing_pixel_scale_raises(self):
        ifd = MagicMock()
        ifd.model_pixel_scale = None
        ifd.model_tiepoint = [0, 0, 0, 0, 0, 0]
        with pytest.raises(ValueError, match="Missing"):
            _transform_from_ifd(ifd)

    def test_missing_tiepoint_raises(self):
        ifd = MagicMock()
        ifd.model_pixel_scale = [10, 10, 0]
        ifd.model_tiepoint = None
        with pytest.raises(ValueError, match="Missing"):
            _transform_from_ifd(ifd)

    def test_short_tiepoint_raises(self):
        ifd = MagicMock()
        ifd.model_pixel_scale = [10, 10, 0]
        ifd.model_tiepoint = [0, 0, 0]  # too short
        with pytest.raises(ValueError, match="Missing"):
            _transform_from_ifd(ifd)


# ── _dtype_from_ifd ──────────────────────────────────────────────────────


class TestDtypeFromIfd:
    @pytest.mark.parametrize(
        "sf, bits, expected",
        [
            (1, 8, np.dtype("u1")),  # Uint8
            (1, 16, np.dtype("u2")),  # Uint16
            (1, 32, np.dtype("u4")),  # Uint32
            (2, 16, np.dtype("i2")),  # Int16
            (2, 32, np.dtype("i4")),  # Int32
            (3, 32, np.dtype("f4")),  # Float32
            (3, 64, np.dtype("f8")),  # Float64
        ],
    )
    def test_integer_sample_format(self, sf, bits, expected):
        ifd = make_mock_ifd(sample_format=sf, bits_per_sample=bits)
        assert _dtype_from_ifd(ifd) == expected

    @pytest.mark.parametrize(
        "sf_int, bits, expected",
        [
            (1, 16, np.dtype("u2")),
            (2, 32, np.dtype("i4")),
            (3, 32, np.dtype("f4")),
        ],
    )
    def test_enum_like_sample_format(self, sf_int, bits, expected):
        """SampleFormat may be an enum-like object convertible via int()."""
        ifd = make_mock_ifd(bits_per_sample=bits)
        sf_obj = MagicMock()
        sf_obj.__int__ = lambda self: sf_int
        sf_obj.__eq__ = lambda self, other: False  # not a plain int
        ifd.sample_format = [sf_obj]
        assert _dtype_from_ifd(ifd) == expected

    def test_unsupported_format_raises(self):
        ifd = make_mock_ifd()
        ifd.sample_format = [99]
        with pytest.raises(NotImplementedError, match="Unsupported sample_format"):
            _dtype_from_ifd(ifd)


# ── _nodata_from_ifd ─────────────────────────────────────────────────────


class TestNodataFromIfd:
    def test_no_other_tags(self):
        ifd = MagicMock()
        ifd.other_tags = None
        assert _nodata_from_ifd(ifd, np.dtype("f4")) is None

    def test_no_nodata_tag(self):
        ifd = make_mock_ifd()
        assert _nodata_from_ifd(ifd, np.dtype("f4")) is None

    def test_numeric_string(self):
        ifd = make_mock_ifd(nodata_tag="-9999")
        result = _nodata_from_ifd(ifd, np.dtype("f4"))
        assert result == -9999.0

    def test_nan_string(self):
        ifd = make_mock_ifd(nodata_tag="nan")
        result = _nodata_from_ifd(ifd, np.dtype("f4"))
        assert math.isnan(result)

    def test_nan_string_variants(self):
        for s in ["NaN", "+nan", "-nan", "NAN"]:
            ifd = make_mock_ifd(nodata_tag=s)
            result = _nodata_from_ifd(ifd, np.dtype("f4"))
            assert math.isnan(result), f"Failed for {s!r}"

    def test_nan_for_integer_dtype_returns_none(self):
        ifd = make_mock_ifd(nodata_tag="nan")
        assert _nodata_from_ifd(ifd, np.dtype("u2")) is None

    def test_integer_value(self):
        ifd = make_mock_ifd(nodata_tag=0)
        result = _nodata_from_ifd(ifd, np.dtype("u1"))
        assert result == 0
        assert isinstance(result, int)

    def test_float_value(self):
        ifd = make_mock_ifd(nodata_tag=-9999.0)
        result = _nodata_from_ifd(ifd, np.dtype("f4"))
        assert result == -9999.0
        assert isinstance(result, float)

    def test_float_coerced_to_int_for_int_dtype(self):
        ifd = make_mock_ifd(nodata_tag=255.0)
        result = _nodata_from_ifd(ifd, np.dtype("u1"))
        assert result == 255
        assert isinstance(result, int)

    def test_nested_single_element_list(self):
        ifd = make_mock_ifd(nodata_tag=[[["-9999"]]])
        result = _nodata_from_ifd(ifd, np.dtype("f4"))
        assert result == -9999.0

    def test_rational_tuple(self):
        ifd = make_mock_ifd(nodata_tag=(1, 2))
        result = _nodata_from_ifd(ifd, np.dtype("f4"))
        assert result == 0.5

    def test_null_byte_string_returns_none(self):
        ifd = make_mock_ifd(nodata_tag="\x00")
        assert _nodata_from_ifd(ifd, np.dtype("f4")) is None

    def test_empty_string_returns_none(self):
        ifd = make_mock_ifd(nodata_tag="  ")
        assert _nodata_from_ifd(ifd, np.dtype("f4")) is None

    def test_unsupported_type_returns_none(self):
        ifd = make_mock_ifd(nodata_tag=object())
        assert _nodata_from_ifd(ifd, np.dtype("f4")) is None


# ── Profile.from_ifd end-to-end ─────────────────────────────────────────


class TestProfileFromIfd:
    def test_basic_profile(self):
        ifd = make_mock_ifd(width=256, height=512, scale=10.0, bands=3)
        p = Profile.from_ifd(ifd)
        assert p.width == 256
        assert p.height == 512
        assert p.count == 3
        assert p.dtype == np.dtype("u2")
        assert p.crs_epsg == 32632
        assert p.res == (10.0, 10.0)

    def test_geographic_crs_fallback(self):
        ifd = make_mock_ifd()
        ifd.geo_key_directory = MagicMock(projected_type=None, geographic_type=4326)
        p = Profile.from_ifd(ifd)
        assert p.crs_epsg == 4326

    def test_no_geo_key_directory(self):
        ifd = make_mock_ifd()
        ifd.geo_key_directory = None
        p = Profile.from_ifd(ifd)
        assert p.crs_epsg is None

    def test_nodata_populated(self):
        ifd = make_mock_ifd(nodata_tag="-9999")
        p = Profile.from_ifd(ifd)
        assert p.nodata == -9999
