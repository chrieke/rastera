"""Unit tests for AsyncGeoTIFF."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from affine import Affine

from rastera.reader import AsyncGeoTIFF, _decode_tile_concurrently, _extract_key
from rastera.geo import BBox, Window
from rastera.meta import Profile
from tests.conftest import make_mock_ifd


# ── Helpers ──────────────────────────────────────────────────────────────

_make_mock_ifd = make_mock_ifd


def _make_mock_tiff(ifd):
    tiff = MagicMock()
    tiff.ifds = [ifd]
    return tiff


# ── Construction & properties ────────────────────────────────────────────


class TestAsyncGeoTIFFInit:
    def test_construction(self):
        ifd = _make_mock_ifd()
        tiff = _make_mock_tiff(ifd)
        gt = AsyncGeoTIFF("s3://bucket/key.tif", tiff)
        assert gt.uri == "s3://bucket/key.tif"
        assert gt.ifd_index == 0
        assert gt.ifd is ifd
        assert isinstance(gt.profile, Profile)

    def test_repr(self):
        ifd = _make_mock_ifd()
        tiff = _make_mock_tiff(ifd)
        gt = AsyncGeoTIFF("s3://bucket/key.tif", tiff)
        r = repr(gt)
        assert "AsyncGeoTIFF" in r
        assert "s3://bucket/key.tif" in r

    def test_profile_matches_ifd(self):
        ifd = _make_mock_ifd(width=200, height=150, bands=4)
        tiff = _make_mock_tiff(ifd)
        gt = AsyncGeoTIFF("s3://b/k.tif", tiff)
        assert gt.profile.width == 200
        assert gt.profile.height == 150
        assert gt.profile.count == 4


# ── open() classmethod ──────────────────────────────────────────────────


class TestOpen:
    @pytest.mark.asyncio
    @patch("rastera.reader.TIFF")
    @patch("rastera.reader.from_url")
    async def test_open_auto_store(self, mock_from_url, mock_tiff_cls):
        """Without an explicit store, from_url builds one from the URI."""
        ifd = _make_mock_ifd()
        tiff_inst = _make_mock_tiff(ifd)
        mock_store = MagicMock()
        mock_from_url.return_value = mock_store
        mock_tiff_cls.open = AsyncMock(return_value=tiff_inst)

        gt = await AsyncGeoTIFF.open(
            "s3://bucket/key.tif", skip_signature=True
        )

        mock_from_url.assert_called_once_with(
            "s3://bucket/key.tif", skip_signature=True, region="us-west-2"
        )
        mock_tiff_cls.open.assert_awaited_once_with(
            "", store=mock_store, prefetch=32768
        )
        assert gt.uri == "s3://bucket/key.tif"
        assert isinstance(gt, AsyncGeoTIFF)

    @pytest.mark.asyncio
    @patch("rastera.reader.TIFF")
    async def test_open_with_store(self, mock_tiff_cls):
        """With an explicit store, from_url is NOT called; key is extracted from URI."""
        ifd = _make_mock_ifd()
        tiff_inst = _make_mock_tiff(ifd)
        mock_tiff_cls.open = AsyncMock(return_value=tiff_inst)
        existing_store = MagicMock()

        gt = await AsyncGeoTIFF.open(
            "s3://bucket/path/to/key.tif", store=existing_store
        )

        mock_tiff_cls.open.assert_awaited_once_with(
            "path/to/key.tif", store=existing_store, prefetch=32768
        )
        assert gt.uri == "s3://bucket/path/to/key.tif"


# ── Context manager ──────────────────────────────────────────────────────



# ── read() ───────────────────────────────────────────────────────────────


def _make_tile_mock(tx, ty, data):
    """Create a mock tile with .x, .y and async .decode()."""
    tile = MagicMock()
    tile.x = tx
    tile.y = ty
    tile.decode = AsyncMock(return_value=data)
    return tile


class TestRead:
    @pytest.mark.asyncio
    async def test_read_bbox_and_window_raises(self):
        ifd = _make_mock_ifd()
        tiff = _make_mock_tiff(ifd)
        gt = AsyncGeoTIFF("s3://b/k.tif", tiff)
        with pytest.raises(ValueError, match="Cannot specify both"):
            await gt.read(bbox=(0, 0, 100, 100), bbox_crs=32633, window=Window(0, 10, 0, 10))

    @pytest.mark.asyncio
    async def test_read_full_image(self):
        """Read with no bbox/window should use full image bounds."""
        ifd = _make_mock_ifd(width=16, height=16, scale=1.0, bands=1, tile_size=16)
        tiff = _make_mock_tiff(ifd)
        gt = AsyncGeoTIFF("s3://b/k.tif", tiff)

        # Single tile covering the whole image: shape (H, W, bands)
        tile_data = np.ones((16, 16, 1), dtype=np.uint16)
        tile_mock = _make_tile_mock(0, 0, tile_data)
        tiff.fetch_tiles = AsyncMock(return_value=[tile_mock])

        data, profile = await gt.read()
        assert data.shape == (1, 16, 16)
        assert data.dtype == np.uint16
        assert profile.width == 16
        assert profile.height == 16
        np.testing.assert_array_equal(data, 1)

    @pytest.mark.asyncio
    async def test_read_with_window(self):
        ifd = _make_mock_ifd(width=32, height=32, scale=1.0, bands=2, tile_size=32)
        tiff = _make_mock_tiff(ifd)
        gt = AsyncGeoTIFF("s3://b/k.tif", tiff)

        tile_data = np.full((32, 32, 2), 42, dtype=np.uint16)
        tile_mock = _make_tile_mock(0, 0, tile_data)
        tiff.fetch_tiles = AsyncMock(return_value=[tile_mock])

        window = Window(4, 20, 4, 20)
        data, profile = await gt.read(window=window)
        assert data.shape == (2, 16, 16)
        np.testing.assert_array_equal(data, 42)

    @pytest.mark.asyncio
    async def test_read_band_indices(self):
        ifd = _make_mock_ifd(width=16, height=16, scale=1.0, bands=3, tile_size=16)
        tiff = _make_mock_tiff(ifd)
        gt = AsyncGeoTIFF("s3://b/k.tif", tiff)

        tile_data = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3)
        tile_mock = _make_tile_mock(0, 0, tile_data)
        tiff.fetch_tiles = AsyncMock(return_value=[tile_mock])

        data, profile = await gt.read(band_indices=[1, 3])
        assert data.shape == (2, 16, 16)
        np.testing.assert_array_equal(data[0], tile_data[:, :, 0])
        np.testing.assert_array_equal(data[1], tile_data[:, :, 2])

    @pytest.mark.asyncio
    async def test_read_band_index_zero_raises(self):
        ifd = _make_mock_ifd(width=16, height=16, scale=1.0, bands=3, tile_size=16)
        tiff = _make_mock_tiff(ifd)
        gt = AsyncGeoTIFF("s3://b/k.tif", tiff)

        with pytest.raises(ValueError, match="1-based"):
            await gt.read(band_indices=[0])


# ── _decode_tile_concurrently ────────────────────────────────────────────


class TestDecodeTileConcurrently:
    @pytest.mark.asyncio
    async def test_success(self):
        data = np.zeros((8, 8, 1))
        tile = _make_tile_mock(2, 3, data)
        tx, ty, decoded = await _decode_tile_concurrently(tile)
        assert tx == 2
        assert ty == 3
        np.testing.assert_array_equal(decoded, data)

    @pytest.mark.asyncio
    async def test_failure_wraps_exception(self):
        tile = MagicMock()
        tile.x = 1
        tile.y = 2
        tile.decode = AsyncMock(side_effect=ValueError("bad tile"))
        with pytest.raises(RuntimeError, match="Failed to decode tile"):
            await _decode_tile_concurrently(tile)
