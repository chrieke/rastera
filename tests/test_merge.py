"""Unit tests for merge_cogs and helpers."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray

from rastera.geo import BBox, bounds_from_transform
from rastera.merge import (
    _mosaic_grid_from_bbox,
    _require_compatible_merge_inputs,
    merge_cogs,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_geotiff_stub(
    width=100,
    height=100,
    scale=10.0,
    count=1,
    origin_x=0.0,
    origin_y=None,
    crs_epsg=32632,
    dtype: np.dtype[Any] = np.dtype("u2"),
    nodata=None,
):
    """Build a MagicMock that quacks like async_geotiff.GeoTIFF."""
    if origin_y is None:
        origin_y = height * scale
    transform = Affine(scale, 0, origin_x, 0, -scale, origin_y)
    bounds = (origin_x, origin_y - height * scale, origin_x + width * scale, origin_y)

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
    return gt


def _make_cog(
    width=100,
    height=100,
    scale=10.0,
    bands=1,
    origin_x=0.0,
    origin_y=None,
    crs=32632,
    dtype: np.dtype[Any] = np.dtype("u2"),
    nodata=None,
):
    """Build a mock AsyncGeoTIFF."""
    gt = _make_geotiff_stub(
        width=width,
        height=height,
        scale=scale,
        count=bands,
        origin_x=origin_x,
        origin_y=origin_y,
        crs_epsg=crs,
        dtype=dtype,
        nodata=nodata,
    )
    cog = MagicMock()
    cog._geotiff = gt
    cog._crs_epsg = crs
    cog._nodata = nodata
    cog.overviews = []
    cog.read = AsyncMock()
    return cog


def _make_array(data, transform, geotiff=None, nodata=None):
    """Build a RasterArray for test returns."""
    if geotiff is None:
        geotiff = MagicMock()
        geotiff.nodata = float(nodata) if nodata is not None else None
        geotiff.crs = MagicMock()
        geotiff.crs.to_epsg.return_value = 32632
    return RasterArray(
        data=data,
        mask=None,
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        transform=transform,
        _alpha_band_idx=None,
        _geotiff=geotiff,
    )


# ── _mosaic_grid_from_bbox ───────────────────────────────────────────────


class TestMosaicGridFromBbox:
    def test_aligned_bbox(self):
        base_transform = Affine(10, 0, 0, 0, -10, 1000)
        bbox = BBox(100, 500, 300, 800)
        transform, w, h = _mosaic_grid_from_bbox(
            base_transform=base_transform,
            bbox=bbox,
        )
        assert w == 20
        assert h == 30
        bounds = bounds_from_transform(transform, w, h)
        assert bounds.minx == 100.0
        assert bounds.maxy == 800.0

    def test_subpixel_bbox_still_produces_grid(self):
        base_transform = Affine(10, 0, 0, 0, -10, 1000)
        # A tiny bbox within a single pixel still produces a 1x1 grid
        bbox = BBox(5, 5, 6, 6)
        _, w, h = _mosaic_grid_from_bbox(base_transform=base_transform, bbox=bbox)
        assert w >= 1
        assert h >= 1


# ── _require_compatible_merge_inputs ─────────────────────────────────────


class TestRequireCompatibleMergeInputs:
    def test_single_cog_passes(self):
        _require_compatible_merge_inputs([_make_cog()])

    def test_mismatched_crs_raises(self):
        cog1 = _make_cog(crs=32632)
        cog2 = _make_cog(crs=32633)
        with pytest.raises(ValueError, match="same CRS"):
            _require_compatible_merge_inputs([cog1, cog2])

    def test_mismatched_resolution_raises(self):
        cog1 = _make_cog(scale=10.0)
        cog2 = _make_cog(scale=20.0)
        with pytest.raises(ValueError, match="same pixel width"):
            _require_compatible_merge_inputs([cog1, cog2])

    def test_aligned_cogs_pass(self):
        # Two COGs with different origins but aligned to the same grid
        cog1 = _make_cog(origin_x=0.0, origin_y=1000.0)
        cog2 = _make_cog(origin_x=1000.0, origin_y=1000.0)
        _require_compatible_merge_inputs([cog1, cog2])


# ── merge_cogs ───────────────────────────────────────────────────────────


class TestMergeCogs:
    async def test_single_cog(self):
        cog = _make_cog(width=10, height=10, scale=1.0, bands=1)
        # Mock read returns a 1-band 5x5 array
        read_arr = np.ones((1, 5, 5), dtype=np.uint16) * 42
        read_result = _make_array(
            read_arr,
            transform=Affine(1, 0, 2, 0, -1, 8),
            geotiff=cog._geotiff,
        )
        cog._read_native = AsyncMock(return_value=read_result)

        result = await merge_cogs(
            [cog],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            snap_to_grid=True,
        )
        assert result.data.shape[0] == 1  # 1 band
        cog._read_native.assert_called_once()

    async def test_no_cogs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            await merge_cogs(
                [],
                bbox=BBox(0, 0, 10, 10),
                bbox_crs=32632,
                target_crs=32632,
                target_resolution=1.0,
            )

    async def test_fill_value_used(self):
        """When a COG doesn't intersect the bbox, the output should be fill_value."""
        cog = _make_cog(width=10, height=10, scale=1.0, origin_x=100, origin_y=110)
        # bbox is at (0,0)-(10,10), cog is at (100,100)-(110,110) — no overlap
        result = await merge_cogs(
            [cog],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            fill_value=9999,
            target_crs=32632,
            target_resolution=1.0,
        )
        assert np.all(result.data == 9999)

    async def test_two_cogs_overlap(self):
        """Two overlapping COGs: second one wins in overlap region."""
        cog1 = _make_cog(width=10, height=10, scale=1.0, bands=1)
        cog2 = _make_cog(width=10, height=10, scale=1.0, bands=1, origin_x=5.0)

        arr1 = np.ones((1, 10, 10), dtype=np.uint16) * 1
        arr2 = np.ones((1, 10, 10), dtype=np.uint16) * 2
        cog1._read_native = AsyncMock(
            return_value=_make_array(
                arr1,
                Affine(1, 0, 0, 0, -1, 10),
                geotiff=cog1._geotiff,
            )
        )
        cog2._read_native = AsyncMock(
            return_value=_make_array(
                arr2,
                Affine(1, 0, 5, 0, -1, 10),
                geotiff=cog2._geotiff,
            )
        )

        result = await merge_cogs(
            [cog1, cog2],
            bbox=BBox(0, 0, 15, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            method="last",
            snap_to_grid=True,
        )
        # The overlap region (cols 5-9) should have cog2's value (last writer wins with method="last")
        assert result.data.shape == (1, 10, 15)
        assert np.all(result.data[0, :, :5] == 1)  # cog1 only
        assert np.all(result.data[0, :, 10:] == 2)  # cog2 only
        assert np.all(result.data[0, :, 5:10] == 2)  # overlap -> cog2 wins

    async def test_nodata_skipped_in_overlap(self):
        """Nodata pixels in a later COG should not overwrite valid data from earlier COGs."""
        NODATA = 0
        cog1 = _make_cog(width=10, height=10, scale=1.0, bands=1, nodata=NODATA)
        cog2 = _make_cog(
            width=10, height=10, scale=1.0, bands=1, origin_x=5.0, nodata=NODATA
        )

        arr1 = np.ones((1, 10, 10), dtype=np.uint16) * 42
        # cog2: left half is nodata, right half is valid
        arr2 = np.zeros((1, 10, 10), dtype=np.uint16)  # all nodata
        arr2[:, :, 5:] = 99  # right half is valid
        cog1._read_native = AsyncMock(
            return_value=_make_array(
                arr1,
                Affine(1, 0, 0, 0, -1, 10),
                nodata=NODATA,
            )
        )
        cog2._read_native = AsyncMock(
            return_value=_make_array(
                arr2,
                Affine(1, 0, 5, 0, -1, 10),
                nodata=NODATA,
            )
        )

        result = await merge_cogs(
            [cog1, cog2],
            bbox=BBox(0, 0, 15, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            snap_to_grid=True,
        )
        assert result.data.shape == (1, 10, 15)
        # cog1-only region: value 42
        assert np.all(result.data[0, :, :5] == 42)
        # overlap where cog2 has nodata: cog1's value (42) preserved
        assert np.all(result.data[0, :, 5:10] == 42)
        # cog2-only valid region: value 99
        assert np.all(result.data[0, :, 10:] == 99)

    async def test_nan_nodata_skipped(self):
        """NaN nodata pixels should be transparent during merge."""
        cog1 = _make_cog(
            width=10,
            height=10,
            scale=1.0,
            bands=1,
            dtype=np.dtype("f4"),
            nodata=float("nan"),
        )
        cog2 = _make_cog(
            width=10,
            height=10,
            scale=1.0,
            bands=1,
            dtype=np.dtype("f4"),
            nodata=float("nan"),
        )

        arr1 = np.full((1, 10, 10), 5.0, dtype=np.float32)
        # cog2: top half is NaN, bottom half is valid
        arr2 = np.full((1, 10, 10), np.nan, dtype=np.float32)
        arr2[:, 5:, :] = 77.0
        cog1._read_native = AsyncMock(
            return_value=_make_array(
                arr1,
                Affine(1, 0, 0, 0, -1, 10),
                nodata=float("nan"),
            )
        )
        cog2._read_native = AsyncMock(
            return_value=_make_array(
                arr2,
                Affine(1, 0, 0, 0, -1, 10),
                nodata=float("nan"),
            )
        )

        result = await merge_cogs(
            [cog1, cog2],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            method="last",
            snap_to_grid=True,
        )
        # top half: cog2 is NaN so cog1's value (5.0) preserved
        assert np.all(result.data[0, :5, :] == 5.0)
        # bottom half: cog2 has valid data (77.0) which overwrites
        assert np.all(result.data[0, 5:, :] == 77.0)

    async def test_nodata_none_still_overwrites(self):
        """When nodata is None, later COGs overwrite earlier ones with method='last'."""
        cog1 = _make_cog(width=10, height=10, scale=1.0, bands=1)
        cog2 = _make_cog(width=10, height=10, scale=1.0, bands=1)

        arr1 = np.ones((1, 10, 10), dtype=np.uint16) * 42
        arr2 = np.zeros((1, 10, 10), dtype=np.uint16)  # all zeros
        cog1._read_native = AsyncMock(
            return_value=_make_array(
                arr1,
                Affine(1, 0, 0, 0, -1, 10),
            )
        )
        cog2._read_native = AsyncMock(
            return_value=_make_array(
                arr2,
                Affine(1, 0, 0, 0, -1, 10),
            )
        )

        result = await merge_cogs(
            [cog1, cog2],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            method="last",
            snap_to_grid=True,
        )
        # nodata=None with method="last", so cog2's zeros overwrite cog1's 42s
        assert np.all(result.data == 0)


# ── merge_cogs: reprojected path ────────────────────────────────────────


class TestMergeReprojected:
    """Tests for _merge_reprojected, triggered by target_crs or target_resolution."""

    async def test_merge_with_target_crs(self):
        """Setting target_crs different from native CRS triggers the reprojected path."""
        cog = _make_cog(width=10, height=10, scale=1.0, bands=1, crs=32632)

        # Return native-CRS data; the merge code resamples into the target grid.
        native_arr = np.ones((1, 10, 10), dtype=np.uint16) * 42
        native_result = _make_array(native_arr, Affine(1, 0, 0, 0, -1, 10))
        cog._read_native = AsyncMock(return_value=native_result)

        await merge_cogs(
            [cog],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=4326,
            target_resolution=1.0,
        )
        cog._read_native.assert_called()

    async def test_merge_with_target_resolution(self):
        """Setting target_resolution different from native triggers the reprojected path."""
        cog = _make_cog(width=10, height=10, scale=1.0, bands=1)

        # Return native-resolution data; merge code resamples to target_resolution.
        native_arr = np.ones((1, 10, 10), dtype=np.uint16) * 7
        native_result = _make_array(native_arr, Affine(1.0, 0, 0, 0, -1.0, 10))
        cog._read_native = AsyncMock(return_value=native_result)

        result = await merge_cogs(
            [cog],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=2.0,
        )
        # Output should use the requested resolution
        assert result.res[0] == pytest.approx(2.0)
        cog._read_native.assert_called()

    async def test_merge_method_first_reprojected(self):
        """method='first' in reprojected path keeps the first COG's pixels."""
        # Two fully overlapping COGs, different values. method="first" should
        # keep cog1's value everywhere.
        cog1 = _make_cog(width=10, height=10, scale=1.0, bands=1, crs=32632)
        cog2 = _make_cog(width=10, height=10, scale=1.0, bands=1, crs=32632)

        # Return native-resolution data; merge code resamples to target_resolution=2.0.
        arr1 = np.ones((1, 10, 10), dtype=np.uint16) * 1
        arr2 = np.ones((1, 10, 10), dtype=np.uint16) * 2
        native_transform = Affine(1.0, 0, 0, 0, -1.0, 10)
        cog1._read_native = AsyncMock(return_value=_make_array(arr1, native_transform))
        cog2._read_native = AsyncMock(return_value=_make_array(arr2, native_transform))

        result = await merge_cogs(
            [cog1, cog2],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=2.0,
            method="first",
        )
        # method="first": cog1's values should take precedence everywhere
        assert np.all(result.data == 1)
