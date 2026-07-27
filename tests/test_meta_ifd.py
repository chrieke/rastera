"""Unit tests for _coerce_nodata."""

import math

import numpy as np
import pytest

from rastera.reader import _coerce_nodata


class TestCoerceNodata:
    def test_none_returns_none(self):
        assert _coerce_nodata(None, np.dtype("f4")) is None

    @pytest.mark.parametrize(
        ("nodata", "dtype", "expected", "expected_type"),
        [
            (-9999.0, "f4", -9999.0, float),
            (255.0, "u1", 255, int),
            (0.0, "u1", 0, int),
            # The last two are the dtype bounds, which are inclusive.
            (-32768.0, "i2", -32768, int),
            (65535.0, "u2", 65535, int),
        ],
    )
    def test_coerced_to_the_dtype_family(
        self, nodata: float, dtype: str, expected: float, expected_type: type
    ):
        result = _coerce_nodata(nodata, np.dtype(dtype))
        assert result == expected
        assert isinstance(result, expected_type)

    def test_nan_returns_none_for_int_dtype(self):
        assert _coerce_nodata(float("nan"), np.dtype("u2")) is None

    def test_nan_preserved_for_float_dtype(self):
        result = _coerce_nodata(float("nan"), np.dtype("f4"))
        assert result is not None
        assert math.isnan(result)

    @pytest.mark.parametrize(
        ("nodata", "dtype"),
        [(-9999.0, "u2"), (65535.0, "u1"), (-32769.0, "i2")],
    )
    def test_out_of_range_returns_none_for_int_dtype(self, nodata: float, dtype: str):
        """No pixel of that dtype can hold the value, and carrying it anyway
        makes ``np.array(nodata, dtype=...)`` inside resample raise
        OverflowError. A VRT declaring <NoDataValue>-9999</NoDataValue> over a
        uint16 source is how this arises."""
        assert _coerce_nodata(nodata, np.dtype(dtype)) is None
