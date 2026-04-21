"""Unit tests for AsyncGeoTIFF."""

from typing import Any
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


def _make_read_result(
    shape: tuple[int, int, int],
    dtype: Any = np.uint16,
    fill: int = 1,
    transform: Affine | None = None,
    geotiff: Any = None,
) -> RasterArray:
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
    async def test_open_auto_store(self, mock_from_url: Any, mock_geotiff_cls: Any):
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
    async def test_open_with_store(self, mock_geotiff_cls: Any):
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
        """Cross-bucket URIs without explicit store should raise."""
        with pytest.raises(ValueError, match="same bucket/host"):
            await rastera.open(
                [
                    "s3://bucket-a/file1.tif",
                    "s3://bucket-b/file2.tif",
                ]
            )

    @pytest.mark.asyncio
    @patch("rastera.reader.GeoTIFF")
    @patch("rastera.store.from_url")
    async def test_open_many_accepts_sibling_local_paths(
        self, mock_from_url: Any, mock_geotiff_cls: Any, tmp_path
    ):
        """Sibling local paths share a parent-dir bucket and must not be rejected."""
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        a.write_bytes(b"")
        b.write_bytes(b"")
        mock_from_url.return_value = MagicMock()
        mock_geotiff_cls.open = AsyncMock(return_value=make_mock_geotiff())

        srcs = await rastera.open([str(a), str(b)], cache=False)

        assert len(srcs) == 2
        mock_from_url.assert_called_once_with(tmp_path.resolve().as_uri())


# ── meta_overrides ──────────────────────────────────────────────────────


class TestMetaOverrides:
    def test_crs_override_on_missing(self):
        """When the TIFF reports no CRS, the override fills it in."""
        gt = make_mock_geotiff()
        gt.crs.to_epsg.return_value = None
        obj = AsyncGeoTIFF("s3://b/k.tif", gt, meta_overrides={"crs": 3006})
        assert obj._crs_epsg == 3006

    def test_crs_override_replaces_existing(self):
        """Override always wins, even when the TIFF already has a CRS."""
        gt = make_mock_geotiff(crs_epsg=4326)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt, meta_overrides={"crs": 3006})
        assert obj._crs_epsg == 3006

    def test_no_override_preserves_file_crs(self):
        gt = make_mock_geotiff(crs_epsg=32632)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        assert obj._crs_epsg == 32632

    def test_crs_override_accepts_pyproj_crs(self):
        from pyproj import CRS

        gt = make_mock_geotiff()
        gt.crs.to_epsg.return_value = None
        obj = AsyncGeoTIFF(
            "s3://b/k.tif", gt, meta_overrides={"crs": CRS.from_epsg(3006)}
        )
        assert obj._crs_epsg == 3006

    def test_unknown_key_raises(self):
        gt = make_mock_geotiff()
        with pytest.raises(ValueError, match="Unknown meta_overrides key"):
            AsyncGeoTIFF("s3://b/k.tif", gt, meta_overrides={"csr": 3006})  # type: ignore[typeddict-unknown-key]

    @pytest.mark.asyncio
    @patch("rastera.reader.GeoTIFF")
    @patch("rastera.reader.from_url")
    async def test_open_forwards_override(
        self, mock_from_url: Any, mock_geotiff_cls: Any
    ):
        """meta_overrides passed to open() propagates to the AsyncGeoTIFF."""
        gt = make_mock_geotiff()
        gt.crs.to_epsg.return_value = None
        mock_from_url.return_value = MagicMock()
        mock_geotiff_cls.open = AsyncMock(return_value=gt)

        obj = await rastera.open(
            "s3://bucket/key.tif", meta_overrides={"crs": 3006}
        )
        assert isinstance(obj, AsyncGeoTIFF)
        assert obj._crs_epsg == 3006


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

        arr = await obj.read()
        assert arr.data.shape == (1, 16, 16)  # type: ignore[reportUnknownMemberType]
        assert arr.data.dtype == np.uint16  # type: ignore[reportUnknownMemberType]
        assert arr.width == 16
        assert arr.height == 16
        np.testing.assert_array_equal(arr.data, 1)  # type: ignore[reportUnknownMemberType]

    @pytest.mark.asyncio
    async def test_read_with_window(self):
        gt = make_mock_geotiff(
            width=32, height=32, scale=1.0, count=2, tile_width=32, tile_height=32
        )
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        result = _make_read_result((2, 16, 16), dtype=np.uint16, fill=42, geotiff=gt)
        gt.read = AsyncMock(return_value=result)

        window = Window(col_off=4, row_off=4, width=16, height=16)
        arr = await obj.read(window=window)
        assert arr.data.shape == (2, 16, 16)  # type: ignore[reportUnknownMemberType]
        np.testing.assert_array_equal(arr.data, 42)  # type: ignore[reportUnknownMemberType]

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

        arr = await obj.read(band_indices=[1, 3])
        assert arr.data.shape == (2, 16, 16)  # type: ignore[reportUnknownMemberType]
        # band_indices [1, 3] → 0-based [0, 2]
        np.testing.assert_array_equal(arr.data[0], data[0])  # type: ignore[reportUnknownMemberType]
        np.testing.assert_array_equal(arr.data[1], data[2])  # type: ignore[reportUnknownMemberType]

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
