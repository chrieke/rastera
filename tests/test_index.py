"""Unit tests for build_index, open_from_index, and HeaderCacheStore."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
box = pytest.importorskip("shapely.geometry").box

from rastera.index import (  # noqa: E402
    HeaderCacheStore,
    build_index,
    open_from_index,
)
from rastera.reader import AsyncGeoTIFF  # noqa: E402
from tests.conftest import make_mock_geotiff  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_mock_async_geotiff(
    uri: str = "s3://bucket/key.tif",
    crs_epsg: int = 32632,
    width: int = 100,
    height: int = 100,
    scale: float = 10.0,
    count: int = 3,
    dtype: np.dtype[Any] = np.dtype("u2"),
    nodata: float | None = None,
) -> MagicMock:
    """Build a mock AsyncGeoTIFF with _geotiff, _crs_epsg, _nodata."""
    gt = make_mock_geotiff(
        width=width,
        height=height,
        scale=scale,
        count=count,
        dtype=dtype,
        nodata=float(nodata) if nodata is not None else None,
        crs_epsg=crs_epsg,
    )

    obj = MagicMock(spec=AsyncGeoTIFF)
    obj.uri = uri
    obj._geotiff = gt
    obj._crs_epsg = crs_epsg
    obj._nodata = nodata
    obj.overviews = []
    return obj


# Return type is Any, not gpd.GeoDataFrame: gpd comes from importorskip, so it
# is a variable rather than a module symbol usable in a type expression.
def _make_index_gdf(entries: list[dict[str, Any]]) -> Any:
    """Build a GeoDataFrame matching the build_index schema.

    Each entry is a dict with keys: uri, crs_epsg, minx, miny, maxx, maxy.
    Missing keys get sensible defaults.
    """
    rows: dict[str, list[Any]] = {
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
    geometries: list[Any] = []
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
    @patch("rastera.index.obstore.get_range_async", new_callable=AsyncMock)
    async def test_get_range_served_from_cache(self, mock_get_range: Any) -> None:
        cached_bytes = b"ABCDEFGHIJ"  # 10 bytes
        store = HeaderCacheStore(MagicMock(), {"file.tif": cached_bytes})

        result = await store.get_range_async("file.tif", start=2, end=6)

        assert result == b"CDEF"
        mock_get_range.assert_not_called()

    @patch("rastera.index.obstore.get_range_async", new_callable=AsyncMock)
    async def test_get_range_delegates_beyond_cache(self, mock_get_range: Any) -> None:
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

    @patch("rastera.index.obstore.get_ranges_async", new_callable=AsyncMock)
    async def test_get_ranges_mixed(self, mock_get_ranges: Any) -> None:
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
    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.obstore.get_range_async", new_callable=AsyncMock)
    async def test_single_uri(
        self, mock_get_range: Any, mock_open: Any, mock_build_obs: Any
    ) -> None:
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
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326
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

    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.obstore.get_range_async", new_callable=AsyncMock)
    async def test_reprojects_bounds_to_4326(
        self,
        mock_get_range: Any,
        mock_open: Any,
        mock_build_obs: Any,
    ) -> None:
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
        minx, miny, maxx, maxy = geom.bounds  # type: ignore[reportUnknownMemberType]
        # UTM bounds (0,0)-(1000,1000) → WGS84 should be small lon/lat values
        assert -180 <= minx <= 180
        assert -90 <= miny <= 90
        assert maxx > minx
        assert maxy > miny

    async def test_cross_bucket_raises(self) -> None:
        with pytest.raises(ValueError, match="same bucket/host"):
            await build_index(["s3://bucket-a/key.tif", "s3://bucket-b/key.tif"])

    async def test_cross_bucket_raises_with_explicit_store(self) -> None:
        """A caller-supplied store does not rescue mirrored key paths: the
        header cache is keyed by object key, so the two rows would collapse."""
        with pytest.raises(ValueError, match="same bucket/host"):
            await build_index(
                ["s3://bucket-a/tiles/x.tif", "s3://bucket-b/tiles/x.tif"],
                store=MagicMock(),
            )

    async def test_empty_uris(self) -> None:
        gdf = await build_index([])

        assert len(gdf) == 0
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326
        assert "uri" in gdf.columns
        assert "header_bytes" in gdf.columns


# ── open_from_index ──────────────────────────────────────────────────────


class TestOpenFromIndex:
    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.get_cached_geotiff", return_value=None)
    async def test_returns_cogs(
        self, mock_cache: Any, mock_open: Any, mock_build_obs: Any
    ) -> None:
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

    async def test_cross_bucket_raises(self) -> None:
        """Mirrored buckets sharing a key path would collapse in the header
        cache, serving one file's header for the other's URI."""
        gdf = _make_index_gdf(
            [
                {
                    "uri": "s3://bucket-a/tiles/x.tif",
                    "header_bytes": b"\xaa" * 100,
                    "minx": 0,
                    "miny": 0,
                    "maxx": 1,
                    "maxy": 1,
                },
                {
                    "uri": "s3://bucket-b/tiles/x.tif",
                    "header_bytes": b"\xbb" * 100,
                    "minx": 0,
                    "miny": 0,
                    "maxx": 1,
                    "maxy": 1,
                },
            ]
        )

        with pytest.raises(ValueError, match="same bucket/host"):
            await open_from_index(gdf)

    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.get_cached_geotiff", return_value=None)
    async def test_bbox_narrowing_to_one_bucket_is_allowed(
        self, mock_cache: Any, mock_open: Any, mock_build_obs: Any
    ) -> None:
        """A multi-bucket index is fine as long as the selected rows agree."""
        mock_build_obs.return_value = MagicMock()
        mock_open.return_value = MagicMock(spec=AsyncGeoTIFF)

        gdf = _make_index_gdf(
            [
                {"uri": "s3://a/x.tif", "minx": 0, "miny": 0, "maxx": 1, "maxy": 1},
                {"uri": "s3://b/y.tif", "minx": 5, "miny": 5, "maxx": 6, "maxy": 6},
            ]
        )

        result = await open_from_index(gdf, bbox=(0, 0, 1, 1), bbox_crs=4326)

        assert len(result) == 1

    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.get_cached_geotiff", return_value=None)
    async def test_bbox_filter(
        self, mock_cache: Any, mock_open: Any, mock_build_obs: Any
    ) -> None:
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

    async def test_empty_after_filter(self) -> None:
        gdf = _make_index_gdf(
            [
                {"uri": "s3://b/a.tif", "minx": 10, "miny": 10, "maxx": 11, "maxy": 11},
            ]
        )

        result = await open_from_index(gdf, bbox=(0, 0, 1, 1), bbox_crs=4326)

        assert result == []


class TestBuildIndexStore:
    @patch("rastera.index._build_obstore")
    @patch("rastera.index.AsyncGeoTIFF.open", new_callable=AsyncMock)
    @patch("rastera.index.obstore.get_range_async", new_callable=AsyncMock)
    async def test_honours_explicit_store(
        self, mock_get_range: Any, mock_open: Any, mock_build_obs: Any
    ) -> None:
        """`store` was declared and documented but never read, so a caller's
        authenticated store was silently replaced by a default unsigned one.
        `open_from_index` already honoured it."""
        mock_get_range.return_value = b"\x00" * 32768
        mock_open.return_value = _make_mock_async_geotiff(uri="s3://bucket/key.tif")
        sentinel = MagicMock(name="caller_store")

        await build_index(["s3://bucket/key.tif"], store=sentinel)

        mock_build_obs.assert_not_called()
        # The header fetch goes through the caller's store, not a rebuilt one.
        assert mock_get_range.await_args is not None
        assert mock_get_range.await_args.args[0] is sentinel
