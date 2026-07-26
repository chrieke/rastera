"""Unit tests for merge and helpers."""

from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray

from rastera.geo import (
    BBox,
    WindowOutOfRangeError,
    bounds_from_transform,
    window_from_bbox,
)
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

    def test_offgrid_bbox_is_contained(self):
        """Rounding the span rather than the far edge stopped the mosaic a
        pixel short of a bbox its own reads had already covered."""
        bbox = BBox(0.8, 0.0, 11.3, 10.0)
        transform, w, h = _mosaic_grid_from_bbox(
            base_transform=Affine(1, 0, 0, 0, -1, 10), bbox=bbox
        )
        bounds = bounds_from_transform(transform, w, h)
        assert bounds.minx <= bbox.minx and bounds.maxx >= bbox.maxx
        assert bounds.miny <= bbox.miny and bounds.maxy >= bbox.maxy


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

    async def test_merge_resampling_bilinear(self):
        """merge with resampling='bilinear' produces expected shape and dtype."""
        cog = _make_cog(width=10, height=10, scale=1.0, bands=1)
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
            resampling="bilinear",
        )
        assert result.res[0] == pytest.approx(2.0)  # type: ignore[reportUnknownMemberType]
        assert result.data.dtype == np.uint16  # type: ignore[reportUnknownMemberType]
        # Constant input → bilinear output is the same constant.
        assert np.all(result.data == 7)  # type: ignore[reportUnknownMemberType]

    async def test_merge_resampling_cubic(self):
        """merge with resampling='cubic' produces expected shape and dtype."""
        cog = _make_cog(width=10, height=10, scale=1.0, bands=1)
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
            resampling="cubic",
        )
        assert result.res[0] == pytest.approx(2.0)  # type: ignore[reportUnknownMemberType]
        assert result.data.dtype == np.uint16  # type: ignore[reportUnknownMemberType]
        # Constant input → cubic output is the same constant (kernel sums to 1).
        assert np.all(result.data == 7)  # type: ignore[reportUnknownMemberType]

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


# ── merge: seam between adjacent tiles ─────────────────────────────────


def _make_windowed_cog(origin_x: float, width: int, value: int):
    """A COG whose ``_read_native`` honours the real window arithmetic.

    The other merge tests mock ``_read_native`` with a fixed array, which hides
    sizing bugs in ``window_from_bbox``.
    """
    cog = _make_cog(
        width=width,
        height=20,
        scale=1.0,
        bands=1,
        origin_x=origin_x,
        origin_y=15.0,
        nodata=0,
    )
    gt = cog._geotiff

    async def _read_native(
        *, bbox: Any = None, snap_to_grid: bool = True, **_: Any
    ) -> RasterArray:
        win = window_from_bbox(gt, bbox, snap_to_grid=snap_to_grid)
        data = np.full((1, win.height, win.width), value, dtype=np.uint16)
        transform = gt.transform * Affine.translation(win.col_off, win.row_off)
        return _make_array(data, transform, geotiff=gt)

    cog._read_native = _read_native
    return cog


class TestMergeSeam:
    """A bbox that doesn't land on the source grid used to lose the left
    tile's last column at the seam: rasterio's floor-offset/round-span window
    sizing dropped it once the interval was clipped to the tile's right edge,
    so the pixel fell through to the *next* tile — silently violating
    mosaic_method="first".
    """

    # Left tile covers x < 10, right tile starts at x = 6. The bbox origin
    # sits 0.8 px off the shared grid, and output column 8 (x 8.8-9.8, centre
    # 9.3) falls in the left tile's final column.
    BBOX = BBox(0.8, 0.0, 13.8, 10.0)
    SEAM_COL = 8

    @pytest.mark.parametrize("snap_to_grid", [True, False])
    async def test_first_wins_at_seam(self, snap_to_grid: bool):
        left = _make_windowed_cog(origin_x=-10.0, width=20, value=1)
        right = _make_windowed_cog(origin_x=6.0, width=20, value=2)

        result = await merge(
            [left, right],
            bbox=self.BBOX,
            bbox_crs=32632,
            target_crs=32632,
            target_resolution=1.0,
            snap_to_grid=snap_to_grid,
        )
        data: np.ndarray[Any, Any] = result.data  # type: ignore[reportUnknownMemberType]
        # Snapping shifts the grid onto the source pixels, moving the last
        # left-tile column one to the right.
        col = self.SEAM_COL + 1 if snap_to_grid else self.SEAM_COL
        assert data[0, :, col].tolist() == [1] * data.shape[1]
        assert data[0, :, col + 1].tolist() == [2] * data.shape[1]


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


# ── concurrency: merge ─────────────────────────────────────────────


@pytest.fixture
def _reset_merge_concurrency():
    yield
    import rastera

    rastera.set_concurrency(merge=1, vrt=1, dimap=1)


def _make_strip_cog(origin_x: float, value: int):
    """A 10×10 single-band COG at (origin_x, 0..10) returning *value*."""
    cog = _make_cog(width=10, height=10, scale=1.0, bands=1, origin_x=origin_x)
    arr = np.ones((1, 10, 10), dtype=np.uint16) * value
    cog._read_native = AsyncMock(
        return_value=_make_array(
            arr, Affine(1, 0, origin_x, 0, -1, 10), geotiff=cog._geotiff
        )
    )
    return cog


class TestMergeConcurrencyInvariance:
    @pytest.mark.parametrize("n", [1, 2, 8])
    @pytest.mark.parametrize("mosaic_method", ["first", "last"])
    async def test_pixel_equal_across_n(
        self,
        n: int,
        mosaic_method: Literal["first", "last"],
        _reset_merge_concurrency: None,
    ):
        """Output must match the n=1 baseline pixel-for-pixel for any n."""
        import rastera

        # 5 side-by-side strips with distinct values, no overlap.
        rastera.set_concurrency(merge=1)
        cogs = [_make_strip_cog(i * 10.0, value=i + 1) for i in range(5)]
        baseline = await merge(
            cogs,
            bbox=BBox(0, 0, 50, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            mosaic_method=mosaic_method,
            snap_to_grid=True,
        )

        rastera.set_concurrency(merge=n)
        cogs = [_make_strip_cog(i * 10.0, value=i + 1) for i in range(5)]
        result = await merge(
            cogs,
            bbox=BBox(0, 0, 50, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            mosaic_method=mosaic_method,
            snap_to_grid=True,
        )
        result_data: np.ndarray[Any, Any] = result.data  # type: ignore[reportUnknownMemberType]
        baseline_data: np.ndarray[Any, Any] = baseline.data  # type: ignore[reportUnknownMemberType]
        assert np.array_equal(result_data, baseline_data)

    async def test_first_mode_still_early_exits(self, _reset_merge_concurrency: None):
        """With mosaic_method='first', first batch fully fills output → later
        batches should not be read at all."""
        import rastera

        rastera.set_concurrency(merge=4)

        # 12 fully-overlapping COGs at the same location with distinct values.
        # The first batch (size 4) covers the full output, so the early-exit
        # check between batches should prevent later batches from being read.
        cogs = [_make_strip_cog(0.0, value=i + 1) for i in range(12)]

        await merge(
            cogs,
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            target_crs=32632,
            target_resolution=1.0,
            mosaic_method="first",
            snap_to_grid=True,
        )

        # First batch (4 COGs) should be read; subsequent 8 should not.
        called = [c._read_native.await_count for c in cogs]
        assert called[:4] == [1, 1, 1, 1]
        assert called[4:] == [0] * 8


# ── cross-input compatibility ────────────────────────────────────────────


class TestMixedInputs:
    def _cog_at(self, origin_x: float, value: int, **kw: Any):
        """A 10x10 COG at 10m/px filled with *value*, ready for merge."""
        cog = _make_cog(width=10, height=10, bands=1, origin_x=origin_x, **kw)
        data = np.full((1, 10, 10), value, dtype=cog._geotiff.dtype)
        arr = _make_array(
            data,
            Affine(10, 0, origin_x, 0, -10, 100),
            geotiff=cog._geotiff,
        )
        cog._read_native = AsyncMock(return_value=arr)
        return cog

    async def test_mixed_nodata_masks_per_contributor(self):
        """_gather_and_paste applied cogs[0]'s sentinel to every contributor, so
        a second COG's nodata was pasted as real data."""
        a = self._cog_at(0.0, 7, nodata=0, dtype=np.dtype("u1"))
        b = self._cog_at(100.0, 255, nodata=255, dtype=np.dtype("u1"))

        out = await merge(
            [a, b],
            bbox=(0, 0, 200, 100),
            bbox_crs=32632,
            target_resolution=10.0,
            fill_value=0,
        )
        # A is fully valid; B is entirely its own nodata, so it contributes
        # nothing and its half stays at fill_value.
        out_data: np.ndarray[Any, Any] = out.data  # type: ignore[reportUnknownMemberType]
        assert np.all(out_data[0, :, :10] == 7)
        assert np.all(out_data[0, :, 10:] == 0)

    async def test_mixed_dtype_rejected(self):
        """A float32 contribution into a uint16 output raised an opaque
        TypeError from np.copyto; a narrowing cast truncated silently."""
        a = self._cog_at(0.0, 7, dtype=np.dtype("u2"))
        b = self._cog_at(100.0, 1, dtype=np.dtype("f4"))
        with pytest.raises(ValueError, match="same dtype"):
            await merge(
                [a, b], bbox=(0, 0, 200, 100), bbox_crs=32632, target_resolution=10.0
            )

    async def test_mixed_band_count_rejected(self):
        a = _make_cog(width=10, height=10, bands=1)
        b = _make_cog(width=10, height=10, bands=3, origin_x=100.0)
        with pytest.raises(ValueError, match="same band count"):
            await merge(
                [a, b], bbox=(0, 0, 200, 100), bbox_crs=32632, target_resolution=10.0
            )

    def _cog_with_bands(self, origin_x: float, n_bands: int, n_read_bands: int):
        """A COG advertising *n_bands* whose mocked read returns *n_read_bands*."""
        cog = _make_cog(width=10, height=10, bands=n_bands, origin_x=origin_x)
        data = np.full((n_read_bands, 10, 10), 5, dtype=cog._geotiff.dtype)
        arr = _make_array(
            data, Affine(10, 0, origin_x, 0, -10, 100), geotiff=cog._geotiff
        )
        cog._read_native = AsyncMock(return_value=arr)
        return cog

    async def test_band_subset_allows_mixed_counts(self):
        """Explicit band_indices are resolved against each COG separately, so
        the counts need not agree — only the requested bands must exist."""
        a = self._cog_with_bands(0.0, n_bands=4, n_read_bands=3)
        b = self._cog_with_bands(100.0, n_bands=3, n_read_bands=3)
        out = await merge(
            [a, b],
            bbox=(0, 0, 200, 100),
            bbox_crs=32632,
            target_resolution=20.0,
            band_indices=[1, 2, 3],
        )
        out_data: np.ndarray[Any, Any] = out.data  # type: ignore[reportUnknownMemberType]
        assert out_data.shape[0] == 3

    async def test_band_subset_missing_band_rejected(self):
        a = self._cog_with_bands(0.0, n_bands=3, n_read_bands=3)
        b = self._cog_with_bands(100.0, n_bands=2, n_read_bands=3)
        with pytest.raises(ValueError, match="requested bands"):
            await merge(
                [a, b],
                bbox=(0, 0, 200, 100),
                bbox_crs=32632,
                target_resolution=20.0,
                band_indices=[1, 2, 3],
            )

    async def test_mixed_dtype_rejected_on_reprojected_path(self):
        """The reprojected path ran no compatibility check at all."""
        a = self._cog_at(0.0, 7, dtype=np.dtype("u2"))
        b = self._cog_at(100.0, 1, dtype=np.dtype("f4"))
        with pytest.raises(ValueError, match="same dtype"):
            await merge(
                [a, b],
                bbox=(0, 0, 200, 100),
                bbox_crs=32632,
                # Non-native resolution forces _merge_reprojected.
                target_resolution=20.0,
            )
