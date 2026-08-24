"""Unit tests for AsyncGeoTIFF."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray, Window
from pyproj import CRS

import rastera
from rastera.formats.dimap import _DIMAPDataset
from rastera.geo import BBox, snapped_grid_for_bbox
from rastera.profile import RasterProfile
from rastera.reader import (
    AsyncGeoTIFF,
    _geotiff_cache,
    clear_cache,
    set_cache_size,
)
from rastera.resampling import resample
from tests.conftest import (
    make_mock_geotiff,
    make_raster_array,
    slicing_read,
    spy_read_native,
    spy_read_to_grid,
)

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
    return make_raster_array(data, transform, geotiff)


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

    def test_overviews_populated(self):
        gt = make_mock_geotiff()
        ovr = MagicMock()
        ovr.width = 50
        ovr.height = 50
        gt.overviews = [ovr]
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        assert obj.overviews == [(50, 50)]


# ── profile ─────────────────────────────────────────────────────────────


class TestProfile:
    def test_grid_keys(self):
        src = AsyncGeoTIFF(
            "s3://b/k.tif",
            make_mock_geotiff(width=1024, height=1024, count=3, nodata=0.0),
        )
        gt = src._geotiff
        profile = src.profile
        assert set(profile) == set(RasterProfile.__required_keys__)
        assert profile["width"] == 1024
        assert profile["height"] == 1024
        assert profile["count"] == 3
        assert profile["dtype"] == "uint16"
        assert profile["crs"] is gt.crs
        assert profile["crs_epsg"] == 32632
        assert profile["transform"] == gt.transform
        assert profile["res"] == (10.0, 10.0)
        assert profile["nodata"] == 0
        assert profile["overviews"] == []

    def test_bounds_is_a_bbox(self):
        """Not the raw 4-tuple: BBox is rastera's own spatial type, so a bbox
        out of the profile goes straight back into read() or merge()."""
        profile = AsyncGeoTIFF("s3://b/k.tif", make_mock_geotiff()).profile
        assert profile["bounds"] == BBox(0.0, 0.0, 1000.0, 1000.0)
        assert isinstance(profile["bounds"], BBox)

    def test_crs_is_the_files_own_object(self):
        """Not CRS.from_epsg(crs_epsg): to_epsg() matches at 70% confidence, so
        rebuilding from the code can relabel a WKT as a CRS it isn't."""
        gt = make_mock_geotiff(crs_epsg=32632)
        assert AsyncGeoTIFF("s3://b/k.tif", gt).profile["crs"] is gt.crs

    def test_codeless_wkt_keeps_the_crs(self):
        gt = make_mock_geotiff()
        gt.crs.to_epsg.return_value = None
        profile = AsyncGeoTIFF("s3://b/k.tif", gt).profile
        assert profile["crs_epsg"] is None
        assert profile["crs"] is gt.crs

    def test_crs_override_wins(self):
        gt = make_mock_geotiff(crs_epsg=32632)
        profile = AsyncGeoTIFF("s3://b/k.tif", gt, meta_overrides={"crs": 3006}).profile
        assert profile["crs"] == CRS.from_epsg(3006)
        assert profile["crs_epsg"] == 3006

    async def test_crs_override_never_touches_an_unparseable_crs(self):
        """The whole reason _crs_override is stored apart from _crs_epsg: geo
        keys that raise are exactly what an override exists to replace. Reads
        have to survive it too, not just the profile — ``_output_geotiff_ref``
        asks the file for its EPSG on every read path."""
        gt = make_mock_geotiff(width=20, height=20, count=1)
        gt.read = slicing_read(gt, np.ones((1, 20, 20), np.uint16))
        type(gt).crs = PropertyMock(side_effect=ValueError("bad geo keys"))
        try:
            src = AsyncGeoTIFF("s3://b/k.tif", gt, meta_overrides={"crs": 3006})
            assert src.profile["crs"] == CRS.from_epsg(3006)
            arr = await src._read_native()
            assert arr.crs == CRS.from_epsg(3006)
        finally:
            del type(gt).crs

    def test_nodata_is_the_coerced_sentinel(self):
        """A uint16 band cannot hold -9999, so the dataset has no sentinel even
        though the file declares one."""
        gt = make_mock_geotiff(dtype=np.dtype("u2"), nodata=-9999.0)
        assert AsyncGeoTIFF("s3://b/k.tif", gt).profile["nodata"] is None
        assert gt.nodata == -9999.0

    def test_representable_sentinel_passes_through(self):
        gt = make_mock_geotiff(dtype=np.dtype("u2"), nodata=0.0)
        assert AsyncGeoTIFF("s3://b/k.tif", gt).profile["nodata"] == 0

    def test_dtype_is_a_string(self):
        """rasterio takes the dtype by name, not as a numpy object."""
        gt = make_mock_geotiff()
        assert isinstance(AsyncGeoTIFF("s3://b/k.tif", gt).profile["dtype"], str)

    def test_unmapped_dtype_is_none_not_the_string_none(self):
        """str(None) would put "None" in the profile, which rasterio accepts
        as far as the key check and then fails on obscurely."""
        gt = make_mock_geotiff(width=1024, height=1024)
        gt.dtype = None
        assert AsyncGeoTIFF("s3://b/k.tif", gt).profile["dtype"] is None


# ── profile reachability across the synthesized datasets ───────────────


def _synthesized_datasets() -> dict[str, AsyncGeoTIFF]:
    """One of each dataset whose ``_geotiff`` is not a real GeoTIFF."""
    from tests.formats.test_dimap import _two_group_layout
    from tests.test_vrt import _make_rgbnir_ds
    from tests.test_vrt_processed import _make_processed_ds

    return {
        "vrt": _make_rgbnir_ds(),
        "processed": _make_processed_ds()[0],
        "dimap": _DIMAPDataset("s3://b/DIM.xml", _two_group_layout()),
    }


@pytest.mark.parametrize("flavour", ["vrt", "processed", "dimap"])
def test_profile_is_reachable_on_synthesized_datasets(flavour: str) -> None:
    """A VRT or DIMAP dataset builds its ``_geotiff`` from a ten-field stand-in,
    so a profile key reading anything wider raises AttributeError only for the
    users on those formats. Every required key must resolve on all three."""
    profile = _synthesized_datasets()[flavour].profile
    assert set(RasterProfile.__required_keys__) <= set(profile)


# ── the public namespace ────────────────────────────────────────────────


class TestPublicExports:
    def test_all_names_resolve(self):
        for name in rastera.__all__:
            assert getattr(rastera, name, None) is not None, name

    async def test_missed_bbox_is_catchable_from_the_top_level(self):
        """A bbox that misses a tile is ordinary when walking an AOI over a tile
        set, so it must be catchable by name rather than as a bare ValueError
        alongside every argument-validation error."""
        gt = make_mock_geotiff(width=4, height=4, scale=10.0, count=1, tile_width=4)
        gt.read = slicing_read(gt, np.zeros((1, 4, 4), np.uint16))
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        with pytest.raises(rastera.WindowOutOfRangeError):
            await obj.read(bbox=(1e6, 1e6, 1e6 + 10, 1e6 + 10), bbox_crs=32632)


# ── open() classmethod ──────────────────────────────────────────────────


class TestOpen:
    @patch("rastera.reader.GeoTIFF")
    @patch("rastera.store.from_url")
    async def test_open_auto_store(self, mock_from_url: Any, mock_geotiff_cls: Any):
        """Without an explicit store, one is built rooted at the bucket."""
        gt = make_mock_geotiff()
        mock_store = MagicMock()
        mock_from_url.return_value = mock_store
        mock_geotiff_cls.open = AsyncMock(return_value=gt)

        obj = await AsyncGeoTIFF.open("s3://bucket/key.tif", skip_signature=True)

        mock_from_url.assert_called_once_with(
            "s3://bucket", skip_signature=True, region="us-west-2"
        )
        mock_geotiff_cls.open.assert_awaited_once_with(
            "key.tif", store=mock_store, prefetch=32768
        )
        assert obj.uri == "s3://bucket/key.tif"
        assert isinstance(obj, AsyncGeoTIFF)

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

    async def test_open_multi_uri_cross_bucket_raises(self):
        """Cross-bucket URIs without explicit store should raise."""
        with pytest.raises(ValueError, match="same bucket/host"):
            await rastera.open(
                [
                    "s3://bucket-a/file1.tif",
                    "s3://bucket-b/file2.tif",
                ]
            )

    @patch("rastera.reader.GeoTIFF")
    @patch("rastera.store.from_url")
    async def test_open_many_accepts_sibling_local_paths(
        self, mock_from_url: Any, mock_geotiff_cls: Any, tmp_path: Path
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

    @patch("rastera.reader.GeoTIFF")
    @patch("rastera.store.from_url")
    async def test_open_forwards_override(
        self, mock_from_url: Any, mock_geotiff_cls: Any
    ):
        """meta_overrides passed to open() propagates to the AsyncGeoTIFF."""
        gt = make_mock_geotiff()
        gt.crs.to_epsg.return_value = None
        mock_from_url.return_value = MagicMock()
        mock_geotiff_cls.open = AsyncMock(return_value=gt)

        obj = await rastera.open("s3://bucket/key.tif", meta_overrides={"crs": 3006})
        assert isinstance(obj, AsyncGeoTIFF)
        assert obj._crs_epsg == 3006


# ── read() ───────────────────────────────────────────────────────────────


class TestReadArgumentValidation:
    """read() shares merge()'s validators, so the same bad argument is rejected
    the same way on both entry points instead of failing deep in NumPy."""

    @staticmethod
    def _obj():
        gt = make_mock_geotiff(width=16, height=16, scale=1.0, count=1)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        # No case here should reach I/O.
        gt.read = AsyncMock(side_effect=AssertionError("read was issued"))
        return obj

    async def test_unknown_resampling_rejected_on_native_path(self):
        """The native path never calls resample(), so this argument was
        accepted and then silently ignored."""
        with pytest.raises(ValueError, match="Unknown resampling method"):
            await self._obj().read(resampling="lanczos")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    async def test_bad_target_resolution_rejected(self, bad: float):
        with pytest.raises(ValueError, match="target_resolution"):
            await self._obj().read(target_resolution=bad)

    async def test_inverted_bbox_rejected(self):
        with pytest.raises(ValueError, match="minx < maxx"):
            await self._obj().read(bbox=(10, 0, 0, 10), bbox_crs=32632)

    async def test_float_band_index_rejected(self):
        with pytest.raises(ValueError, match="must be integers"):
            await self._obj().read(band_indices=[1.5])  # type: ignore[list-item]

    async def test_native_resolution_is_not_rejected(self):
        """target_resolution equal to the source res is a no-op, not an error."""
        gt = make_mock_geotiff(
            width=16, height=16, scale=1.0, count=1, tile_width=16, tile_height=16
        )
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        gt.read = AsyncMock(
            return_value=_make_read_result((1, 16, 16), dtype=np.uint16, geotiff=gt)
        )
        arr = await obj.read(target_resolution=1.0, resampling="bilinear")
        assert arr.data.shape == (1, 16, 16)  # type: ignore[reportUnknownMemberType]


class TestRead:
    async def test_read_bbox_and_window_raises(self):
        gt = make_mock_geotiff()
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        with pytest.raises(ValueError, match="Cannot specify both"):
            await obj.read(
                bbox=(0, 0, 100, 100),
                bbox_crs=32632,
                window=Window(col_off=0, row_off=0, width=10, height=10),
            )

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

    async def test_read_band_index_zero_raises(self):
        gt = make_mock_geotiff(width=16, height=16, scale=1.0, count=3)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        with pytest.raises(ValueError, match="1-based"):
            await obj.read(band_indices=[0])

    async def test_window_resample_reads_past_the_window(self):
        """The output grid is ceil-sized, so reading only the window left the
        trailing row/column with nothing behind it — nodata, even mid-image
        where the source has plenty."""
        gt = make_mock_geotiff(
            width=20, height=20, scale=1.0, count=1, tile_width=20, nodata=0
        )
        full = (np.arange(400, dtype=np.uint16) + 1).reshape(1, 20, 20)
        gt.read = slicing_read(gt, full)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        window = Window(col_off=5, row_off=5, width=10, height=10)
        arr = await obj.read(window=window, target_resolution=3.0)
        assert not np.any(np.asarray(arr.data) == 0)  # type: ignore[reportUnknownMemberType]
        # A window is a pixel-space request: its extent anchors the output.
        assert arr.transform.c == 5.0

    async def test_window_resample_matches_equivalent_bbox(self):
        """Window and bbox reads of the same region agree when the window's
        extent lies on the target-resolution lattice (row_off ≡ 20 mod 3
        here); a bbox read snaps onto that lattice, a window read stays
        anchored on the window's own extent."""
        gt = make_mock_geotiff(
            width=20, height=20, scale=1.0, count=1, tile_width=20, nodata=0
        )
        full = (np.arange(400, dtype=np.uint16) + 1).reshape(1, 20, 20)
        gt.read = slicing_read(gt, full)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        window = Window(col_off=6, row_off=2, width=9, height=9)
        arr = await obj.read(window=window, target_resolution=3.0)

        equivalent = await obj.read(
            bbox=(6.0, 9.0, 15.0, 18.0), bbox_crs=32632, target_resolution=3.0
        )
        np.testing.assert_array_equal(arr.data, equivalent.data)  # type: ignore[reportUnknownMemberType]
        assert arr.bounds == equivalent.bounds

        offgrid = await obj.read(
            bbox=(5.0, 5.0, 15.0, 15.0), bbox_crs=32632, target_resolution=3.0
        )
        assert offgrid.transform == Affine(3, 0, 3.0, 0, -3, 15.0)
        assert (offgrid.width, offgrid.height) == (4, 4)

    async def test_window_resample_with_overview_reads_right_region(self):
        """*window* is in full-resolution pixels; it used to be handed to the
        overview unchanged, which reads a different region entirely."""
        gt = make_mock_geotiff(width=400, height=400, scale=1.0, count=1)
        ov = make_mock_geotiff(width=100, height=100, scale=4.0, count=1)
        ov.transform = Affine(4, 0, 0, 0, -4, 400)
        ov.res = (4.0, 4.0)
        ov.bounds = (0, 0, 400, 400)
        gt.overviews = [ov]
        gt.read = slicing_read(gt, np.zeros((1, 400, 400), np.uint16))
        ov.read = slicing_read(ov, np.zeros((1, 100, 100), np.uint16))

        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        obj._best_overview_for_resolution = lambda r: ov if r >= 4.0 else None  # type: ignore[method-assign]

        arr = await obj.read(
            window=Window(col_off=300, row_off=0, width=80, height=80),
            target_resolution=8.0,
            use_overviews=True,
        )
        assert (arr.bounds[0], arr.bounds[2]) == (300.0, 380.0)

    @pytest.mark.parametrize("method", ["bilinear", "cubic"])
    # res 1x1 downsamples 10x on both axes; 2x20 downsamples 5x in x but
    # *up*samples in y, so an x-derived halo is too narrow for the y kernel.
    @pytest.mark.parametrize(
        ("res_x", "res_y"), [(1.0, 1.0), (2.0, 20.0)], ids=["square", "tall-pixels"]
    )
    async def test_downsample_edge_matches_full_source(
        self, method: Any, res_x: float, res_y: float
    ):
        """resample widens its kernel with the downsample factor; without a
        matching source halo the outer ring came from a truncated kernel.
        Full-source resample is the oracle."""
        size = 200
        full = (np.arange(size * size, dtype=np.uint16) % 4093).reshape(1, size, size)
        gt = make_mock_geotiff(
            width=size, height=size, scale=1.0, count=1, tile_width=size
        )
        gt.transform = src_transform = Affine(res_x, 0, 0, 0, -res_y, size * res_y)
        gt.res = (res_x, res_y)
        gt.bounds = (0, 0, size * res_x, size * res_y)
        gt.read = slicing_read(gt, full)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        # Interior on both axes, so a halo is available to be got wrong.
        bbox = BBox(50.0 * res_x, 50.0 * res_y, 150.0 * res_x, 150.0 * res_y)
        got = await obj.read(
            bbox=bbox, bbox_crs=32632, target_resolution=10.0, resampling=method
        )
        out_transform, out_w, out_h = snapped_grid_for_bbox(bbox, 10.0)
        want = resample(
            full,
            src_transform=src_transform,
            dst_transform=out_transform,
            dst_width=out_w,
            dst_height=out_h,
            nodata=None,
            method=method,
        )
        np.testing.assert_array_equal(got.data, want)  # type: ignore[reportUnknownMemberType]

    async def test_unsnapped_transform_clamped_to_image(self):
        """The window is clipped to the image, so anchoring on a bbox edge that
        overhangs it labelled the pixels where they are not."""
        gt = make_mock_geotiff(
            width=16, height=16, scale=10.0, count=1, tile_width=16, tile_height=16
        )  # world x/y in [0, 160]
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        gt.read = AsyncMock(
            return_value=_make_read_result((1, 16, 8), dtype=np.uint16, geotiff=gt)
        )

        arr = await obj.read(
            bbox=(-500, 0, 80, 160), bbox_crs=32632, snap_to_grid=False
        )
        assert arr.transform.c == 0.0  # not -500
        assert arr.transform.f == 160.0

    async def test_unsnapped_transform_non_square_pixels(self):
        """res[0] was used for both axes, so tall pixels came back short."""
        gt = make_mock_geotiff(
            width=16, height=16, scale=10.0, count=1, tile_width=16, tile_height=16
        )
        gt.res = (10.0, 20.0)
        gt.transform = Affine(10, 0, 0, 0, -20, 320)
        gt.bounds = (0, 0, 160, 320)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        gt.read = AsyncMock(
            return_value=_make_read_result((1, 8, 8), dtype=np.uint16, geotiff=gt)
        )

        arr = await obj.read(bbox=(0, 160, 80, 320), bbox_crs=32632, snap_to_grid=False)
        assert arr.transform.a == 10.0
        assert arr.transform.e == -20.0


# ── read: output grid is a pure function of the arguments ───────────────


class TestReadGridInvariance:
    """With bbox + explicit target_resolution, the output grid matches
    snapped_grid_for_bbox regardless of source grid phase or path."""

    # Off-grid on every edge at res 1.0, and its edges rounded outward.
    BBOX = (0.8, 0.3, 13.8, 10.3)
    GRID = (Affine(1, 0, 0.0, 0, -1, 11.0), 14, 11)

    @staticmethod
    def _obj(origin_x: float = 0.0, origin_y: float = 20.0) -> AsyncGeoTIFF:
        gt = make_mock_geotiff(
            width=20,
            height=20,
            scale=1.0,
            count=1,
            tile_width=20,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        full = (np.arange(400, dtype=np.uint16) + 1).reshape(1, 20, 20)
        gt.read = slicing_read(gt, full)
        return AsyncGeoTIFF("s3://b/k.tif", gt)

    @staticmethod
    def _grid(result: RasterArray) -> tuple[Affine, int, int]:
        return result.transform, result.width, result.height

    @pytest.mark.parametrize("phase", [0.0, 0.25, 0.5])
    async def test_grid_ignores_source_phase(self, phase: float):
        obj = self._obj(origin_x=phase, origin_y=20.0 + phase)
        result = await obj.read(bbox=self.BBOX, bbox_crs=32632, target_resolution=1.0)
        assert self._grid(result) == self.GRID

    async def test_aligned_bbox_is_exact_and_copied_natively(self):
        obj = self._obj()
        calls = spy_read_to_grid(obj)
        result = await obj.read(
            bbox=(1.0, 2.0, 12.0, 9.0), bbox_crs=32632, target_resolution=1.0
        )
        assert self._grid(result) == (Affine(1, 0, 1.0, 0, -1, 9.0), 11, 7)
        assert not calls

    async def test_native_transform_is_stamped_onto_the_lattice(self):
        # Origin 1e-7 off the lattice: inside the gate's tolerance, so the
        # window copy runs — but the promised transform is the exact lattice.
        obj = self._obj(origin_x=1e-7, origin_y=20.0 + 1e-7)
        result = await obj.read(
            bbox=(1.0, 2.0, 12.0, 9.0), bbox_crs=32632, target_resolution=1.0
        )
        assert result.transform == Affine(1, 0, 1.0, 0, -1, 9.0)

    async def test_offgrid_source_is_resampled_to_the_lattice(self):
        obj = self._obj(origin_x=0.5, origin_y=20.5)
        calls = spy_read_to_grid(obj)
        result = await obj.read(bbox=self.BBOX, bbox_crs=32632, target_resolution=1.0)
        assert self._grid(result) == self.GRID
        assert calls

    async def test_non_square_source_is_resampled_to_the_lattice(self):
        gt = make_mock_geotiff(width=20, height=20, scale=1.0, count=1, tile_width=20)
        gt.transform = Affine(1.0, 0, 0.0, 0, -2.0, 40.0)
        gt.res = (1.0, 2.0)
        gt.bounds = (0.0, 0.0, 20.0, 40.0)
        gt.read = slicing_read(gt, np.ones((1, 20, 20), np.uint16))
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        calls = spy_read_to_grid(obj)
        result = await obj.read(bbox=self.BBOX, bbox_crs=32632, target_resolution=1.0)
        assert self._grid(result) == self.GRID
        assert calls

    @staticmethod
    def _non_square_obj() -> AsyncGeoTIFF:
        """20x10 px at 1.0 x 2.0, so world bounds are the square (0, 0, 20, 20)."""
        gt = make_mock_geotiff(
            width=20, height=10, scale=1.0, y_scale=2.0, count=1, tile_width=20
        )
        gt.read = slicing_read(gt, np.ones((1, 10, 20), np.uint16))
        return AsyncGeoTIFF("s3://b/k.tif", gt)

    async def test_non_square_source_resamples_for_a_window_read(self):
        # target_resolution was matched against res[0] alone, so a source
        # already 1.0 wide took the native path and kept its 2.0-tall rows.
        obj = self._non_square_obj()
        result = await obj.read(
            window=Window(col_off=0, row_off=0, width=20, height=10),
            target_resolution=1.0,
        )
        assert self._grid(result) == (Affine(1, 0, 0.0, 0, -1, 20.0), 20, 20)

    async def test_non_square_source_resamples_when_unsnapped(self):
        obj = self._non_square_obj()
        result = await obj.read(
            bbox=(0.0, 0.0, 20.0, 20.0),
            bbox_crs=32632,
            target_resolution=1.0,
            snap_to_grid=False,
        )
        assert self._grid(result) == (Affine(1, 0, 0.0, 0, -1, 20.0), 20, 20)

    async def test_square_source_still_copies_natively(self):
        # The guard against over-correcting: res[1] now participates, so a
        # square source must not be pushed off its own fast path.
        obj = self._obj()
        calls = spy_read_to_grid(obj)
        await obj.read(
            window=Window(col_off=0, row_off=0, width=20, height=20),
            target_resolution=1.0,
        )
        assert not calls

    async def test_bare_bbox_read_keeps_the_source_window(self):
        # No target_resolution names no lattice: the read stays a 1:1 copy
        # on the source grid, phase and all (clipped at the tile's bottom
        # edge, which sits 0.2 above the bbox's miny).
        obj = self._obj(origin_x=0.5, origin_y=20.5)
        result = await obj.read(bbox=self.BBOX, bbox_crs=32632)
        assert self._grid(result) == (Affine(1, 0, 0.5, 0, -1, 10.5), 14, 10)


# ── The resampling seam's contract ──────────────────────────────────────


class TestWarpSeam:
    """The bbox/overview handed to ``_read_native`` by the resampled paths."""

    @staticmethod
    def _utm_obj() -> tuple[AsyncGeoTIFF, MagicMock]:
        """A 400x400 @10m COG at a realistic UTM 33N origin."""
        gt = make_mock_geotiff(
            width=400, height=400, scale=10.0, count=1, tile_width=400, crs_epsg=32633
        )
        gt.transform = Affine(10, 0, 300000, 0, -10, 5700000)
        gt.bounds = (300000, 5696000, 304000, 5700000)
        gt.read = slicing_read(gt, np.zeros((1, 400, 400), np.uint16))
        return AsyncGeoTIFF("s3://b/k.tif", gt), gt

    @pytest.mark.parametrize(
        ("method", "pad"),
        [
            # bilinear at 2x downsample reaches 2 source px; nearest needs no
            # kernel halo and falls back to the 1 px floor.
            ("bilinear", 20.0),
            ("nearest", 10.0),
        ],
    )
    async def test_same_crs_halo_is_kernel_sized(self, method: str, pad: float):
        obj, _ = self._utm_obj()
        calls = spy_read_native(obj)

        await obj.read(
            bbox=(301000, 5697000, 302000, 5698000),
            bbox_crs=32633,
            target_resolution=20.0,
            resampling=method,  # type: ignore[arg-type]
        )

        assert len(calls) == 1
        assert tuple(calls[0]["bbox"]) == (
            301000.0 - pad,
            5697000.0 - pad,
            302000.0 + pad,
            5698000.0 + pad,
        )
        assert calls[0]["overview"] is None

    async def test_cross_crs_halo_is_per_axis(self):
        """UTM->4326 compresses x and y by different factors, so a scalar
        source-resolution ratio under-pads one axis."""
        obj, _ = self._utm_obj()
        calls = spy_read_native(obj)

        await obj.read(
            bbox=(12.1386, 51.3893, 12.1595, 51.4042),
            bbox_crs=4326,
            target_crs=4326,
            target_resolution=0.0002,
            resampling="cubic",
        )

        assert len(calls) == 1
        # 0.0002 deg is 14.53 m of easting but 23.00 m of northing here, so
        # cubic reaches 3 source px in x and 5 in y: pad_x=30 m, pad_y=50 m.
        # A single x-derived ratio would under-pad y by 0.87 output rows.
        np.testing.assert_allclose(
            tuple(calls[0]["bbox"]),
            (
                300889.40178267437,
                5696885.89827398,
                302474.86487110157,
                5698710.452795458,
            ),
            atol=1e-6,
        )

    async def test_reproject_without_target_resolution_skips_overviews(self):
        """Overviews are all coarser than native, so a density-preserving
        reprojection must not silently read one."""
        obj, gt = self._utm_obj()
        ov = make_mock_geotiff(
            width=200, height=200, scale=20.0, count=1, tile_width=200, crs_epsg=32633
        )
        ov.transform = Affine(20, 0, 300000, 0, -20, 5700000)
        ov.bounds = gt.bounds
        ov.read = slicing_read(ov, np.zeros((1, 200, 200), np.uint16))
        gt.overviews = [ov]
        calls = spy_read_native(obj)

        await obj.read(
            bbox=(12.1386, 51.3893, 12.1595, 51.4042),
            bbox_crs=4326,
            target_crs=4326,
            use_overviews=True,
        )

        assert len(calls) == 1
        assert calls[0]["overview"] is None

    async def test_same_crs_resample_reports_source_geotiff(self):
        """Not a _CrsNodata stub: RasterArray.crs/.nodata read straight off
        this, and vrt._dispatch_source_reads keys its nodata swap off a
        sub-read reporting the *file's* value."""
        obj, gt = self._utm_obj()

        arr = await obj.read(
            bbox=(301000, 5697000, 302000, 5698000),
            bbox_crs=32633,
            target_resolution=20.0,
        )
        assert arr._geotiff is gt

        reprojected = await obj.read(
            bbox=(12.1386, 51.3893, 12.1595, 51.4042),
            bbox_crs=4326,
            target_crs=4326,
            target_resolution=0.0002,
        )
        assert reprojected._geotiff is not gt


# ── The CRS and nodata the output reports ───────────────────────────────


class TestOutputLabels:
    """``RasterArray.crs``/``.nodata`` read straight off ``_geotiff``, so every
    read path has to attach one that agrees with what the *dataset* resolved.
    Logically equivalent reads disagreeing is the bug these guard."""

    @staticmethod
    def _obj(
        meta_overrides: Any = None, **gt_kwargs: Any
    ) -> tuple[AsyncGeoTIFF, MagicMock]:
        gt = make_mock_geotiff(count=1, **gt_kwargs)
        gt.read = slicing_read(gt, np.zeros((1, 100, 100), gt.dtype))
        return AsyncGeoTIFF("s3://b/k.tif", gt, meta_overrides=meta_overrides), gt

    @pytest.mark.parametrize(
        ("read_kwargs", "expected"),
        [
            ({}, 3006),
            ({"target_resolution": 20.0}, 3006),
            # Reprojection reads *from* the override and labels with the target.
            ({"target_crs": 4326}, 4326),
        ],
    )
    async def test_crs_override_reaches_the_output(
        self, read_kwargs: dict[str, Any], expected: int
    ):
        """meta_overrides governed windowing but never the label it exists to fix."""
        obj, _ = self._obj(meta_overrides={"crs": 3006}, crs_epsg=32632)
        arr = await obj.read(**read_kwargs)
        assert arr.crs.to_epsg() == expected

    @pytest.mark.parametrize(
        "read_kwargs",
        [{}, {"target_resolution": 20.0}, {"target_crs": 4326}],
    )
    async def test_unrepresentable_nodata_is_not_reported(
        self, read_kwargs: dict[str, Any]
    ):
        """A uint16 band declaring -9999 has no sentinel its dtype can carry, so
        the output must report none — reporting it makes as_masked() raise."""
        obj, _ = self._obj(nodata=-9999.0, dtype=np.dtype("u2"))
        assert obj._nodata is None

        arr = await obj.read(**read_kwargs)
        assert arr.nodata is None
        arr.as_masked()  # would raise TypeError on an unconvertible fill_value

    @pytest.mark.parametrize(
        ("nodata", "dtype"),
        [
            (0.0, np.dtype("u2")),
            # NaN survives _coerce_nodata untouched, so it is still the file's.
            (float("nan"), np.dtype("f4")),
        ],
    )
    async def test_agreeing_file_keeps_the_live_geotiff(
        self, nodata: float, dtype: np.dtype[Any]
    ):
        """The substitution is conditional: nothing to correct, nothing to stub.
        vrt._dispatch_source_reads keys its nodata swap off what a sub-read
        reports, and merge reaches through _geotiff for dtype and transform."""
        obj, gt = self._obj(nodata=nodata, dtype=dtype)
        assert (await obj.read())._geotiff is gt
        assert (await obj.read(target_resolution=20.0))._geotiff is gt


# ── Coverage on the output ──────────────────────────────────────────────


class TestReadCoverage:
    """A read whose grid runs past the image reports where the image stopped.

    Without a declared sentinel nothing did: the warp clamps its source
    indices, so the overhang came back as a copy of the border column.
    """

    @staticmethod
    def _ramp(nodata: float | None) -> AsyncGeoTIFF:
        """20x20 @1m over (0.3, 0.3)-(20.3, 20.3), values 1..20 across."""
        gt = make_mock_geotiff(
            width=20,
            height=20,
            scale=1.0,
            count=1,
            origin_x=0.3,
            origin_y=20.3,
            crs_epsg=32632,
            dtype=np.dtype("u2"),
            nodata=nodata,
        )
        gt.read = slicing_read(
            gt, np.tile(np.arange(1, 21, dtype=np.uint16), (20, 1))[None]
        )
        return AsyncGeoTIFF("s3://b/k.tif", gt)

    async def _read(self, nodata: float | None) -> RasterArray:
        return await self._ramp(nodata).read(
            bbox=BBox(0.3, 0.3, 25.3, 20.3), bbox_crs=32632, target_resolution=1.0
        )

    async def test_overhang_is_masked_and_blank(self):
        arr = await self._read(None)
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert arr.mask is not None
        # The snapped grid rounds 0.3..20.3 outward, so row 0 (centre y=20.5)
        # clears the image's north edge and columns from 20 on clear its east.
        assert not arr.mask[0].any()
        assert arr.mask[1:, :20].all()
        assert not arr.mask[:, 20:].any()
        assert (data[0, :, 20:] == 0).all(), "border column replicated outward"
        # The covered part is untouched: blanking must not eat a real column.
        assert data[0, 10, :20].tolist() == list(range(1, 21))
        assert np.ma.is_masked(arr.as_masked()[0, 10, 20])

    async def test_a_declared_sentinel_keeps_the_masked_equal_fallback(self):
        """Coverage knows nothing about *interior* nodata, so handing it over as
        the mask would unmask every pixel the file itself marked invalid."""
        arr = await self._read(7.0)
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert arr.mask is None
        assert (data[0, :, 20:] == 7).all()
        masked = arr.as_masked()
        assert np.ma.is_masked(masked[0, 10, 20])  # outside the image
        assert np.ma.is_masked(masked[0, 10, 6])  # the ramp's own 7

    async def test_a_read_inside_the_image_has_no_mask(self):
        arr = await self._ramp(None).read(
            bbox=BBox(2.0, 2.0, 18.0, 18.0), bbox_crs=32632, target_resolution=1.0
        )
        assert arr.mask is None


# ── LRU cache behaviour ────────────────────────────────────────────────


class TestLRUCache:
    def setup_method(self):
        clear_cache()
        self._orig_size = rastera.reader._cache_max_size

    def teardown_method(self):
        clear_cache()
        set_cache_size(self._orig_size)

    @pytest.mark.parametrize("bad", [-1, -100, 1.5, "x", True, False])
    def test_rejects_invalid_size(self, bad: Any):
        """A negative size popped past an empty cache: KeyError, not ValueError.
        bool is an int subclass, so reject it explicitly (as set_concurrency does).
        """
        with pytest.raises(ValueError, match="cache size must be int >= 0"):
            set_cache_size(bad)

    def test_zero_size_disables_and_evicts(self):
        set_cache_size(2)
        _geotiff_cache["a"] = make_mock_geotiff()
        set_cache_size(0)
        assert len(_geotiff_cache) == 0
