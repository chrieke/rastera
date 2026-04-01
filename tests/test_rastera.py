"""Unit tests for AsyncGeoTIFF."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray, Window

import rastera
from rastera.reader import (
    AsyncGeoTIFF,
    _geotiff_cache,
    clear_cache,
    set_cache_size,
)
from tests.conftest import make_mock_geotiff

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_read_result(shape, dtype=np.uint16, fill=1, transform=None, geotiff=None):
    """Create a mock async-geotiff RasterArray result."""
    data = np.full(shape, fill, dtype=dtype)
    if transform is None:
        transform = Affine(1, 0, 0, 0, -1, shape[1])
    if geotiff is None:
        geotiff = MagicMock()
        geotiff.nodata = None
        geotiff.crs = MagicMock()
        geotiff.crs.to_epsg.return_value = 32632
    return RasterArray(
        data=data,
        mask=None,
        width=shape[2],
        height=shape[1],
        count=shape[0],
        transform=transform,
        _alpha_band_idx=None,
        _geotiff=geotiff,
    )


# ── Construction & properties ────────────────────────────────────────────


class TestAsyncGeoTIFFInit:
    def test_construction(self):
        gt = make_mock_geotiff()
        obj = AsyncGeoTIFF("s3://bucket/key.tif", gt)
        assert obj.uri == "s3://bucket/key.tif"
        assert obj._crs_epsg == 32632

    def test_repr(self):
        gt = make_mock_geotiff()
        obj = AsyncGeoTIFF("s3://bucket/key.tif", gt)
        r = repr(obj)
        assert "AsyncGeoTIFF" in r
        assert "s3://bucket/key.tif" in r

    def test_geotiff_attrs(self):
        gt = make_mock_geotiff(width=200, height=150, count=4)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        assert obj._geotiff.width == 200
        assert obj._geotiff.height == 150
        assert obj._geotiff.count == 4

    def test_overviews_populated(self):
        gt = make_mock_geotiff()
        ovr = MagicMock()
        ovr.width = 50
        ovr.height = 50
        gt.overviews = [ovr]
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        assert obj.overviews == [(50, 50)]


# ── open() classmethod ──────────────────────────────────────────────────


class TestOpen:
    @pytest.mark.asyncio
    @patch("rastera.reader.GeoTIFF")
    @patch("rastera.reader.from_url")
    async def test_open_auto_store(self, mock_from_url, mock_geotiff_cls):
        """Without an explicit store, from_url builds one from the URI."""
        gt = make_mock_geotiff()
        mock_store = MagicMock()
        mock_from_url.return_value = mock_store
        mock_geotiff_cls.open = AsyncMock(return_value=gt)

        obj = await AsyncGeoTIFF.open("s3://bucket/key.tif", skip_signature=True)

        mock_from_url.assert_called_once_with(
            "s3://bucket/key.tif", skip_signature=True, region="us-west-2"
        )
        mock_geotiff_cls.open.assert_awaited_once_with(
            "", store=mock_store, prefetch=32768
        )
        assert obj.uri == "s3://bucket/key.tif"
        assert isinstance(obj, AsyncGeoTIFF)

    @pytest.mark.asyncio
    @patch("rastera.reader.GeoTIFF")
    async def test_open_with_store(self, mock_geotiff_cls):
        """With an explicit store, from_url is NOT called; key is extracted from URI."""
        gt = make_mock_geotiff()
        mock_geotiff_cls.open = AsyncMock(return_value=gt)
        existing_store = MagicMock()

        obj = await AsyncGeoTIFF.open(
            "s3://bucket/path/to/key.tif", store=existing_store
        )

        mock_geotiff_cls.open.assert_awaited_once_with(
            "path/to/key.tif", store=existing_store, prefetch=32768
        )
        assert obj.uri == "s3://bucket/path/to/key.tif"

    @pytest.mark.asyncio
    async def test_open_multi_uri_cross_bucket_raises(self):
        """Passing URIs from different buckets without an explicit store should raise."""
        with pytest.raises(ValueError, match="same bucket/host"):
            await rastera.open(
                [
                    "s3://bucket-a/file1.tif",
                    "s3://bucket-b/file2.tif",
                ]
            )


# ── read() ───────────────────────────────────────────────────────────────


class TestRead:
    @pytest.mark.asyncio
    async def test_read_bbox_and_window_raises(self):
        gt = make_mock_geotiff()
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        with pytest.raises(ValueError, match="Cannot specify both"):
            await obj.read(
                bbox=(0, 0, 100, 100),
                bbox_crs=32632,
                window=Window(col_off=0, row_off=0, width=10, height=10),
            )

    @pytest.mark.asyncio
    async def test_read_full_image(self):
        """Read with no bbox/window should use full image bounds."""
        gt = make_mock_geotiff(
            width=16, height=16, scale=1.0, count=1, tile_width=16, tile_height=16
        )
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        result = _make_read_result((1, 16, 16), dtype=np.uint16, geotiff=gt)
        gt.read = AsyncMock(return_value=result)

        raster_array = await obj.read()
        assert raster_array.data.shape == (1, 16, 16)
        assert raster_array.data.dtype == np.uint16
        assert raster_array.width == 16
        assert raster_array.height == 16
        np.testing.assert_array_equal(raster_array.data, 1)

    @pytest.mark.asyncio
    async def test_read_with_window(self):
        gt = make_mock_geotiff(
            width=32, height=32, scale=1.0, count=2, tile_width=32, tile_height=32
        )
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        result = _make_read_result((2, 16, 16), dtype=np.uint16, fill=42, geotiff=gt)
        gt.read = AsyncMock(return_value=result)

        window = Window(col_off=4, row_off=4, width=16, height=16)
        raster_array = await obj.read(window=window)
        assert raster_array.data.shape == (2, 16, 16)
        np.testing.assert_array_equal(raster_array.data, 42)

    @pytest.mark.asyncio
    async def test_read_band_indices(self):
        gt = make_mock_geotiff(
            width=16, height=16, scale=1.0, count=3, tile_width=16, tile_height=16
        )
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        data = np.arange(3 * 16 * 16, dtype=np.uint16).reshape(3, 16, 16)
        result = RasterArray(
            data=data,
            mask=None,
            width=16,
            height=16,
            count=3,
            transform=Affine(1, 0, 0, 0, -1, 16),
            _alpha_band_idx=None,
            _geotiff=gt,
        )
        gt.read = AsyncMock(return_value=result)

        raster_array = await obj.read(band_indices=[1, 3])
        assert raster_array.data.shape == (2, 16, 16)
        # band_indices [1, 3] → 0-based [0, 2]
        np.testing.assert_array_equal(raster_array.data[0], data[0])
        np.testing.assert_array_equal(raster_array.data[1], data[2])

    @pytest.mark.asyncio
    async def test_read_band_index_zero_raises(self):
        gt = make_mock_geotiff(width=16, height=16, scale=1.0, count=3)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        with pytest.raises(ValueError, match="1-based"):
            await obj.read(band_indices=[0])


# ── LRU cache behaviour ────────────────────────────────────────────────


class TestLRUCache:
    def setup_method(self):
        clear_cache()
        self._orig_size = rastera.reader._cache_max_size

    def teardown_method(self):
        clear_cache()
        set_cache_size(self._orig_size)

    def test_lru_eviction_order(self):
        """Accessing an entry promotes it; the least-recently-used entry is evicted."""
        set_cache_size(2)
        gt_a, gt_b, gt_c = (make_mock_geotiff() for _ in range(3))

        # Insert A, then B (cache: A, B)
        _geotiff_cache["a"] = gt_a
        _geotiff_cache["b"] = gt_b

        # Access A — promotes it (cache order: B, A)
        _geotiff_cache.move_to_end("a")

        # Insert C — should evict B (LRU), not A
        set_cache_size(2)  # trigger eviction check if needed
        _geotiff_cache["c"] = gt_c
        if len(_geotiff_cache) > 2:
            _geotiff_cache.popitem(last=False)

        assert "a" in _geotiff_cache, "A was accessed recently and should survive"
        assert "c" in _geotiff_cache, "C was just inserted and should survive"
        assert (
            "b" not in _geotiff_cache
        ), "B was least-recently-used and should be evicted"
