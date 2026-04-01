"""Unit tests for build_index, open_from_index, and HeaderCacheStore."""

from unittest.mock import AsyncMock, MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from rastera.index import HeaderCacheStore, _obstore_key, build_index, open_from_index
from rastera.reader import AsyncGeoTIFF

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_mock_async_geotiff(
    uri="s3://bucket/key.tif",
    crs_epsg=32632,
    width=100,
    height=100,
    scale=10.0,
    count=3,
    dtype=np.dtype("u2"),
    nodata=None,
):
    """Build a mock AsyncGeoTIFF with _geotiff, _crs_epsg, _nodata."""
    transform = Affine(scale, 0, 0, 0, -scale, height * scale)
    bounds = (0, 0, width * scale, height * scale)

    gt = MagicMock()
    gt.width = width
    gt.height = height
    gt.count = count
    gt.dtype = dtype
    gt.nodata = float(nodata) if nodata is not None else None
    gt.transform = transform
    gt.res = (scale, scale)
    gt.bounds = bounds
    gt.tile_width = 256
    gt.tile_height = 256
    crs_mock = MagicMock()
    crs_mock.to_epsg.return_value = crs_epsg
    gt.crs = crs_mock
    gt.overviews = []

    obj = MagicMock(spec=AsyncGeoTIFF)
    obj.uri = uri
    obj._geotiff = gt
    obj._crs_epsg = crs_epsg
    obj._nodata = nodata
    obj.overviews = []
    return obj


def _make_index_gdf(entries):
    """Build a GeoDataFrame matching the build_index schema.

    Each entry is a dict with keys: uri, crs_epsg, minx, miny, maxx, maxy.
    Missing keys get sensible defaults.
    """
    rows = {
        "uri": [],
        "header_bytes": [],
        "crs_epsg": [],
        "width": [],
        "height": [],
        "count": [],
        "res_x": [],
        "res_y": [],
        "dtype": [],
        "nodata": [],
        "overviews": [],
    }
    geometries = []
    for e in entries:
        rows["uri"].append(e["uri"])
        rows["header_bytes"].append(e.get("header_bytes", b"\x00" * 100))
        rows["crs_epsg"].append(e.get("crs_epsg", 32632))
        rows["width"].append(e.get("width", 100))
        rows["height"].append(e.get("height", 100))
        rows["count"].append(e.get("count", 3))
        rows["res_x"].append(e.get("res_x", 10.0))
        rows["res_y"].append(e.get("res_y", 10.0))
        rows["dtype"].append(e.get("dtype", "uint16"))
        rows["nodata"].append(e.get("nodata", None))
        rows["overviews"].append(e.get("overviews", "[]"))
        geometries.append(box(e["minx"], e["miny"], e["maxx"], e["maxy"]))
    return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")


# ── HeaderCacheStore ─────────────────────────────────────────────────────


class TestHeaderCacheStore:
    @pytest.mark.asyncio
    @patch("rastera.index.obstore.get_range_async", new_callable=AsyncMock)
    async def test_get_range_served_from_cache(self, mock_get_range):
        cached_bytes = b"ABCDEFGHIJ"  # 10 bytes
        store = HeaderCacheStore(MagicMock(), {"file.tif": cached_bytes})

        result = await store.get_range_async("file.tif", start=2, end=6)

        assert result == b"CDEF"
        mock_get_range.assert_not_called()

    @pytest.mark.asyncio
    @patch("rastera.index.obstore.get_range_async", new_callable=AsyncMock)
    async def test_get_range_delegates_beyond_cache(self, mock_get_range):
        cached_bytes = b"ABCDE"  # 5 bytes
        inner = MagicMock()
        store = HeaderCacheStore(inner, {"file.tif": cached_bytes})
        mock_get_range.return_value = b"REMOTE_DATA"

        result = await store.get_range_async("file.tif", start=3, end=10)

        assert result == b"REMOTE_DATA"
        mock_get_range.assert_awaited_once_with(
            inner,
            "file.tif",
            start=3,
            end=10,
            length=None,
        )

    @pytest.mark.asyncio
    @patch("rastera.index.obstore.get_ranges_async", new_callable=AsyncMock)
    async def test_get_ranges_mixed(self, mock_get_ranges):
        cached_bytes = b"0123456789"  # 10 bytes
        inner = MagicMock()
        store = HeaderCacheStore(inner, {"file.tif": cached_bytes})
        mock_get_ranges.return_value = [b"REMOTE"]

        result = await store.get_ranges_async(
            "file.tif",
            starts=[0, 8],
            ends=[4, 20],
        )

        assert result[0] == b"0123"  # from cache
        assert result[1] == b"REMOTE"  # delegated
        mock_get_ranges.assert_awaited_once_with(
            inner,
            "file.tif",
            starts=[8],
            ends=[20],
        )


# ── build_index ──────────────────────────────────────────────────────────


class TestBuildIndex:
    @pytest.mark.asyncio
    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.obstore.get_range_async", new_callable=AsyncMock)
    async def test_single_uri(self, mock_get_range, mock_open, mock_build_obs):
        mock_build_obs.return_value = MagicMock()
        mock_get_range.return_value = b"\x00" * 32768
        mock_cog = _make_mock_async_geotiff(
            uri="s3://bucket/key.tif",
            crs_epsg=32632,
            width=100,
            height=100,
            scale=10.0,
            count=3,
        )
        mock_open.return_value = mock_cog

        gdf = await build_index(["s3://bucket/key.tif"])

        assert len(gdf) == 1
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
        row = gdf.iloc[0]
        assert row["uri"] == "s3://bucket/key.tif"
        assert row["crs_epsg"] == 32632
        assert row["width"] == 100
        assert row["height"] == 100
        assert row["count"] == 3
        assert row["res_x"] == 10.0
        assert row["dtype"] == "uint16"
        expected_cols = {
            "uri",
            "header_bytes",
            "crs_epsg",
            "width",
            "height",
            "count",
            "res_x",
            "res_y",
            "dtype",
            "nodata",
            "overviews",
            "geometry",
        }
        assert set(gdf.columns) == expected_cols

    @pytest.mark.asyncio
    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.obstore.get_range_async", new_callable=AsyncMock)
    async def test_reprojects_bounds_to_4326(
        self, mock_get_range, mock_open, mock_build_obs
    ):
        """A UTM COG's geometry in the index should be in EPSG:4326, not UTM."""
        mock_build_obs.return_value = MagicMock()
        mock_get_range.return_value = b"\x00" * 100
        mock_cog = _make_mock_async_geotiff(
            uri="s3://bucket/utm.tif",
            crs_epsg=32632,
            width=100,
            height=100,
            scale=10.0,
        )
        mock_open.return_value = mock_cog

        gdf = await build_index(["s3://bucket/utm.tif"])

        geom = gdf.geometry.iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        # UTM bounds (0,0)-(1000,1000) → WGS84 should be small lon/lat values
        assert -180 <= minx <= 180
        assert -90 <= miny <= 90
        assert maxx > minx
        assert maxy > miny

    @pytest.mark.asyncio
    async def test_empty_uris(self):
        gdf = await build_index([])

        assert len(gdf) == 0
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
        assert "uri" in gdf.columns
        assert "header_bytes" in gdf.columns


# ── open_from_index ──────────────────────────────────────────────────────


class TestOpenFromIndex:
    @pytest.mark.asyncio
    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.get_cached_geotiff", return_value=None)
    async def test_returns_cogs(self, mock_cache, mock_open, mock_build_obs):
        mock_build_obs.return_value = MagicMock()
        mock_open.return_value = MagicMock(spec=AsyncGeoTIFF)

        gdf = _make_index_gdf(
            [
                {"uri": "s3://b/a.tif", "minx": 0, "miny": 0, "maxx": 1, "maxy": 1},
                {"uri": "s3://b/b.tif", "minx": 1, "miny": 0, "maxx": 2, "maxy": 1},
            ]
        )

        result = await open_from_index(gdf)

        assert len(result) == 2
        assert mock_open.await_count == 2

    @pytest.mark.asyncio
    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.get_cached_geotiff", return_value=None)
    async def test_bbox_filter(self, mock_cache, mock_open, mock_build_obs):
        mock_build_obs.return_value = MagicMock()
        mock_open.return_value = MagicMock(spec=AsyncGeoTIFF)

        gdf = _make_index_gdf(
            [
                {"uri": "s3://b/a.tif", "minx": 0, "miny": 0, "maxx": 1, "maxy": 1},
                {"uri": "s3://b/b.tif", "minx": 10, "miny": 10, "maxx": 11, "maxy": 11},
                {"uri": "s3://b/c.tif", "minx": 20, "miny": 20, "maxx": 21, "maxy": 21},
            ]
        )

        result = await open_from_index(gdf, bbox=(0, 0, 1, 1), bbox_crs=4326)

        assert len(result) == 1
        mock_open.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_after_filter(self):
        gdf = _make_index_gdf(
            [
                {"uri": "s3://b/a.tif", "minx": 10, "miny": 10, "maxx": 11, "maxy": 11},
            ]
        )

        result = await open_from_index(gdf, bbox=(0, 0, 1, 1), bbox_crs=4326)

        assert result == []


# ── _obstore_key ─────────────────────────────────────────────────────────


class TestObstoreKey:
    def test_s3_uri(self):
        assert _obstore_key("s3://bucket/path/file.tif") == "path/file.tif"

    def test_virtual_hosted_url(self):
        url = "https://bucket.s3.us-east-1.amazonaws.com/path/file.tif"
        assert _obstore_key(url) == "path/file.tif"

    def test_path_style_url(self):
        url = "https://s3.us-east-1.amazonaws.com/bucket/path/file.tif"
        assert _obstore_key(url) == "path/file.tif"

    def test_path_style_url_matches_extract_key(self):
        """_obstore_key and _extract_key must agree for all S3 URL styles."""
        from rastera.reader import _extract_key

        urls = [
            "s3://bucket/path/file.tif",
            "https://bucket.s3.us-east-1.amazonaws.com/path/file.tif",
            "https://s3.us-east-1.amazonaws.com/bucket/path/file.tif",
        ]
        for url in urls:
            assert _obstore_key(url) == _extract_key(url), f"Mismatch for {url}"
