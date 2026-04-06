"""Unit tests for _coerce_nodata."""

import math

import numpy as np

from rastera.reader import _coerce_nodata


class TestCoerceNodata:
    def test_none_returns_none(self):
        assert _coerce_nodata(None, np.dtype("f4")) is None

    def test_float_for_float_dtype(self):
        result = _coerce_nodata(-9999.0, np.dtype("f4"))
        assert result == -9999.0
        assert isinstance(result, float)

    def test_float_coerced_to_int_for_int_dtype(self):
        result = _coerce_nodata(255.0, np.dtype("u1"))
        assert result == 255
        assert isinstance(result, int)

    def test_nan_returns_none_for_int_dtype(self):
        assert _coerce_nodata(float("nan"), np.dtype("u2")) is None

    def test_nan_preserved_for_float_dtype(self):
        result = _coerce_nodata(float("nan"), np.dtype("f4"))
        assert result is not None
        assert math.isnan(result)

    def test_zero_nodata_int(self):
        result = _coerce_nodata(0.0, np.dtype("u1"))
        assert result == 0
        assert isinstance(result, int)

    def test_negative_nodata_int(self):
        result = _coerce_nodata(-32768.0, np.dtype("i2"))
        assert result == -32768
        assert isinstance(result, int)
