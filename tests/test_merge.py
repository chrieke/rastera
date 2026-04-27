"""Unit tests for merge and helpers."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray

from rastera.geo import BBox, WindowOutOfRangeError, bounds_from_transform
from rastera.merge import (
    _mosaic_grid_from_bbox,
    _require_compatible_merge_inputs,
    _resolve_target_crs,
    merge,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_geotiff_stub(
    width: int = 100,
    height: int = 100,
    scale: float = 10.0,
    count: int = 1,
    origin_x: float = 0.0,
    origin_y: float | None = None,
    crs_epsg: int | None = 32632,
    dtype: np.dtype[Any] = np.dtype("u2"),
    nodata: float | None = None,
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
    width: int = 100,
    height: int = 100,
    scale: float = 10.0,
    bands: int = 1,
    origin_x: float = 0.0,
    origin_y: float | None = None,
    crs: int | None = 32632,
    dtype: np.dtype[Any] = np.dtype("u2"),
    nodata: float | None = None,
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
    cog.count = bands
    cog.read = AsyncMock()
    return cog


def _make_array(
    data: np.ndarray[Any, Any],
    transform: Affine,
    geotiff: Any = None,
    nodata: float | None = None,
):
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


# ── merge ───────────────────────────────────────────────────────────


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

        result = await merge(
            [cog],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            snap_to_grid=True,
        )
        assert result.data.shape[0] == 1  # type: ignore[reportUnknownMemberType]  # 1 band
        cog._read_native.assert_called_once()

    async def test_no_cogs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            await merge(
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
        result = await merge(
            [cog],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            fill_value=9999,
            target_crs=32632,
            target_resolution=1.0,
        )
        assert np.all(result.data == 9999)  # type: ignore[reportUnknownMemberType]

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

        result = await merge(
            [cog1, cog2],
            bbox=BBox(0, 0, 15, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            mosaic_method="last",
            snap_to_grid=True,
        )
        # Overlap region (cols 5-9): cog2 wins (mosaic_method="last")
        assert result.data.shape == (1, 10, 15)  # type: ignore[reportUnknownMemberType]
        assert np.all(result.data[0, :, :5] == 1)  # type: ignore[reportUnknownMemberType]  # cog1 only
        assert np.all(result.data[0, :, 10:] == 2)  # type: ignore[reportUnknownMemberType]  # cog2 only
        assert np.all(result.data[0, :, 5:10] == 2)  # type: ignore[reportUnknownMemberType]  # overlap -> cog2 wins

    async def test_nodata_skipped_in_overlap(self):
        """Nodata pixels should not overwrite valid data."""
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

        result = await merge(
            [cog1, cog2],
            bbox=BBox(0, 0, 15, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            snap_to_grid=True,
        )
        assert result.data.shape == (1, 10, 15)  # type: ignore[reportUnknownMemberType]
        # cog1-only region: value 42
        assert np.all(result.data[0, :, :5] == 42)  # type: ignore[reportUnknownMemberType]
        # overlap where cog2 has nodata: cog1's value (42) preserved
        assert np.all(result.data[0, :, 5:10] == 42)  # type: ignore[reportUnknownMemberType]
        # cog2-only valid region: value 99
        assert np.all(result.data[0, :, 10:] == 99)  # type: ignore[reportUnknownMemberType]

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

        result = await merge(
            [cog1, cog2],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            mosaic_method="last",
            snap_to_grid=True,
        )
        # top half: cog2 is NaN so cog1's value (5.0) preserved
        assert np.all(result.data[0, :5, :] == 5.0)  # type: ignore[reportUnknownMemberType]
        # bottom half: cog2 has valid data (77.0) which overwrites
        assert np.all(result.data[0, 5:, :] == 77.0)  # type: ignore[reportUnknownMemberType]

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

        result = await merge(
            [cog1, cog2],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            mosaic_method="last",
            snap_to_grid=True,
        )
        # nodata=None with mosaic_method="last", so cog2's zeros overwrite cog1's 42s
        assert np.all(result.data == 0)  # type: ignore[reportUnknownMemberType]

    async def test_subpixel_sliver_contributor_does_not_raise(self):
        """A COG that overlaps the request bbox by < 0.5 px must not abort
        the merge. Repro for the production failure where _read_native raises
        ValueError("BBox does not intersect image") on a sub-pixel sliver.
        """
        # Two aligned COGs side by side: main covers x=[0,10], right covers
        # x=[10,20]. Request bbox spills 0.1 m into the right COG, so
        # BBox.intersect accepts the (10, 0, 10.1, 10) sliver but
        # window_from_bbox rounds it to a 0-width window and raises.
        main = _make_cog(width=10, height=10, scale=1.0, bands=1)
        right = _make_cog(width=10, height=10, scale=1.0, bands=1, origin_x=10.0)

        main_arr = np.ones((1, 10, 10), dtype=np.uint16) * 7
        main._read_native = AsyncMock(
            return_value=_make_array(
                main_arr,
                transform=Affine(1, 0, 0, 0, -1, 10),
                geotiff=main._geotiff,
            )
        )
        right._read_native = AsyncMock(
            side_effect=WindowOutOfRangeError("BBox does not intersect image")
        )

        # Put `right` first so the sliver is read before main fills the output
        # (otherwise mosaic_method="first" short-circuits and never reads right).
        result = await merge(
            [right, main],
            bbox=BBox(0, 0, 10.1, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            snap_to_grid=True,
        )
        assert result.data.shape[0] == 1  # type: ignore[reportUnknownMemberType]
        assert result.data.shape[1] == 10  # type: ignore[reportUnknownMemberType]
        assert np.all(result.data[0, :, :10] == 7)  # type: ignore[reportUnknownMemberType]
        right._read_native.assert_called_once()
        main._read_native.assert_called_once()


# ── merge: reprojected path ────────────────────────────────────────


class TestMergeReprojected:
    """Tests for _merge_reprojected, triggered by target_crs or target_resolution."""

    async def test_merge_with_target_crs(self):
        """target_crs != native CRS triggers reprojected path."""
        cog = _make_cog(width=10, height=10, scale=1.0, bands=1, crs=32632)

        # Return native-CRS data; the merge code resamples into the target grid.
        native_arr = np.ones((1, 10, 10), dtype=np.uint16) * 42
        native_result = _make_array(native_arr, Affine(1, 0, 0, 0, -1, 10))
        cog._read_native = AsyncMock(return_value=native_result)

        await merge(
            [cog],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=4326,
            target_resolution=1.0,
        )
        cog._read_native.assert_called()

    async def test_merge_with_target_resolution(self):
        """target_resolution != native triggers reprojected path."""
        cog = _make_cog(width=10, height=10, scale=1.0, bands=1)

        # Return native-resolution data; merge code resamples to target_resolution.
        native_arr = np.ones((1, 10, 10), dtype=np.uint16) * 7
        native_result = _make_array(native_arr, Affine(1.0, 0, 0, 0, -1.0, 10))
        cog._read_native = AsyncMock(return_value=native_result)

        result = await merge(
            [cog],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=2.0,
        )
        # Output should use the requested resolution
        assert result.res[0] == pytest.approx(2.0)  # type: ignore[reportUnknownMemberType]
        cog._read_native.assert_called()

    async def test_merge_method_first_reprojected(self):
        """mosaic_method='first' in reprojected path keeps the first COG's pixels."""
        # Two fully overlapping COGs, different values. mosaic_method="first" should
        # keep cog1's value everywhere.
        cog1 = _make_cog(width=10, height=10, scale=1.0, bands=1, crs=32632)
        cog2 = _make_cog(width=10, height=10, scale=1.0, bands=1, crs=32632)

        # Return native-resolution data; merge code resamples to target_resolution=2.0.
        arr1 = np.ones((1, 10, 10), dtype=np.uint16) * 1
        arr2 = np.ones((1, 10, 10), dtype=np.uint16) * 2
        native_transform = Affine(1.0, 0, 0, 0, -1.0, 10)
        cog1._read_native = AsyncMock(return_value=_make_array(arr1, native_transform))
        cog2._read_native = AsyncMock(return_value=_make_array(arr2, native_transform))

        result = await merge(
            [cog1, cog2],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=2.0,
            mosaic_method="first",
        )
        # mosaic_method="first": cog1's values should take precedence everywhere
        assert np.all(result.data == 1)  # type: ignore[reportUnknownMemberType]


# ── _resolve_target_crs ────────────────────────────────────────────────


class TestResolveTargetCrs:
    def test_most_common_picks_majority(self):
        cogs = [_make_cog(crs=32632), _make_cog(crs=32633), _make_cog(crs=32632)]
        assert _resolve_target_crs(cogs, "most_common") == 32632

    def test_first_picks_first(self):
        cogs = [_make_cog(crs=32633), _make_cog(crs=32632), _make_cog(crs=32632)]
        assert _resolve_target_crs(cogs, "first") == 32633

    def test_first_skips_none_crs(self):
        cogs = [_make_cog(crs=None), _make_cog(crs=32632)]
        assert _resolve_target_crs(cogs, "first") == 32632

    def test_most_common_skips_none_crs(self):
        cogs = [_make_cog(crs=None), _make_cog(crs=32632)]
        assert _resolve_target_crs(cogs, "most_common") == 32632

    def test_all_none_raises(self):
        cogs = [_make_cog(crs=None), _make_cog(crs=None)]
        with pytest.raises(ValueError, match="No CRS found"):
            _resolve_target_crs(cogs, "most_common")

    def test_single_cog(self):
        cogs = [_make_cog(crs=4326)]
        assert _resolve_target_crs(cogs, "most_common") == 4326
        assert _resolve_target_crs(cogs, "first") == 4326
