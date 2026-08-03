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
    transform_bbox,
    window_from_bbox,
)
from rastera.merge import (
    _require_compatible_merge_inputs,
    _resolve_target_crs,
    _snapped_grid_for_bbox,
    merge,
)
from rastera.reader import AsyncGeoTIFF
from rastera.resampling import ResamplingMethod
from tests.conftest import (
    make_mock_geotiff,
    make_raster_array,
    slicing_read,
    spy_read_native,
)

# ── Helpers ──────────────────────────────────────────────────────────────


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
    gt = make_mock_geotiff(
        width=width,
        height=height,
        scale=scale,
        count=bands,
        origin_x=origin_x,
        origin_y=origin_y,
        crs_epsg=crs,
        dtype=dtype,
        nodata=float(nodata) if nodata is not None else None,
    )
    cog = MagicMock()
    cog._geotiff = gt
    cog._crs_epsg = crs
    cog._nodata = nodata
    cog.overviews = []
    cog.count = bands
    cog.read = AsyncMock()
    # Bind the real warp seam: merge routes its reprojected reads through it, so
    # auto-mocking it would skip the code under test and swallow the
    # ``_read_native`` mocks these tests install.
    cog._read_to_grid = AsyncGeoTIFF._read_to_grid.__get__(cog)
    cog._best_overview_for_resolution = (
        AsyncGeoTIFF._best_overview_for_resolution.__get__(cog)
    )
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
    return make_raster_array(data, transform, geotiff)


# ── _snapped_grid_for_bbox ───────────────────────────────────────────────


class TestSnappedGridForBbox:
    def test_aligned_bbox_is_exact(self):
        transform, w, h = _snapped_grid_for_bbox(BBox(100, 500, 300, 800), 10.0)
        assert (w, h) == (20, 30)
        bounds = bounds_from_transform(transform, w, h)
        assert (bounds.minx, bounds.miny, bounds.maxx, bounds.maxy) == (
            100.0,
            500.0,
            300.0,
            800.0,
        )

    def test_subpixel_bbox_still_produces_grid(self):
        # A tiny bbox within a single pixel still produces a 1x1 grid
        _, w, h = _snapped_grid_for_bbox(BBox(5, 5, 6, 6), 10.0)
        assert (w, h) == (1, 1)

    def test_offgrid_bbox_is_contained(self):
        """Rounding the span rather than the far edge stopped the mosaic a
        pixel short of a bbox its own reads had already covered."""
        bbox = BBox(0.8, 0.0, 11.3, 10.0)
        transform, w, h = _snapped_grid_for_bbox(bbox, 1.0)
        bounds = bounds_from_transform(transform, w, h)
        assert bounds.minx <= bbox.minx and bounds.maxx >= bbox.maxx
        assert bounds.miny <= bbox.miny and bounds.maxy >= bbox.maxy
        # Each off-grid edge grows outward by less than one pixel, no further.
        assert (w, h) == (12, 10)

    def test_negative_coordinates(self):
        transform, w, h = _snapped_grid_for_bbox(BBox(-25.0, -14.0, -4.0, -3.0), 10.0)
        assert (w, h) == (3, 2)
        bounds = bounds_from_transform(transform, w, h)
        assert (bounds.minx, bounds.miny, bounds.maxx, bounds.maxy) == (
            -30.0,
            -20.0,
            0.0,
            0.0,
        )

    def test_denoises_utm_magnitude_edges(self):
        """An edge exactly on the grid arrives with ULP error from the divide;
        without _denoise, ceil would buy a spurious column."""
        minx = 499999.9999999996
        transform, w, h = _snapped_grid_for_bbox(
            BBox(minx, 0.0, minx + 20.0, 10.0), 10.0
        )
        assert (w, h) == (2, 1)
        assert transform.c == 500000.0

    def test_thin_bbox_on_grid_line_still_names_a_pixel(self):
        _, w, h = _snapped_grid_for_bbox(
            BBox(9.9999999999, 0.0, 10.0000000001, 10.0), 1.0
        )
        assert (w, h) == (1, 10)


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


# ── merge argument validation ────────────────────────────────────────────


class TestMergeArgumentValidation:
    """These arguments used to fail either silently — by selecting the other
    branch's semantics — or several frames deep in NumPy with an error naming
    neither the argument nor this call. All are rejected before any read."""

    @staticmethod
    def _merge(**kwargs: Any):
        cog = _make_cog(width=10, height=10, scale=1.0, bands=1)
        # No read is expected to happen: every case here must fail before I/O.
        cog._read_native = AsyncMock(side_effect=AssertionError("read was issued"))
        defaults = dict(
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            target_crs=32632,
            target_resolution=1.0,
        )
        return merge([cog], **{**defaults, **kwargs})  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["fisrt", "min", "MAX", "First"])
    async def test_unknown_mosaic_method_rejected(self, bad: str):
        """Anything that wasn't exactly "first" fell through to last-wins."""
        with pytest.raises(ValueError, match="mosaic_method must be"):
            await self._merge(mosaic_method=bad)

    @pytest.mark.parametrize("bad", ["First", "common", "most-common"])
    async def test_unknown_crs_method_rejected(self, bad: str):
        """Anything that wasn't exactly "first" fell through to most_common,
        which for mixed inputs reprojects the whole mosaic into another zone."""
        with pytest.raises(ValueError, match="crs_method must be"):
            await self._merge(crs_method=bad)

    async def test_unknown_resampling_rejected(self):
        """The native fast path never calls resample(), so its method argument
        was never validated at all on this route."""
        with pytest.raises(ValueError, match="Unknown resampling method"):
            await self._merge(resampling="lanczos")

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    async def test_bad_target_resolution_rejected(self, bad: float):
        with pytest.raises(ValueError, match="target_resolution"):
            await self._merge(target_resolution=bad)

    @pytest.mark.parametrize("bad", [-1, 70000])
    async def test_fill_value_outside_dtype_range_rejected(self, bad: int):
        """np.full raised a bare OverflowError naming neither uint16 nor the
        fill_value argument."""
        with pytest.raises(ValueError, match="outside the range"):
            await self._merge(fill_value=bad)

    async def test_fractional_fill_value_rejected(self):
        """0.5 was truncated to 0 in a uint16 mosaic — indistinguishable from
        a deliberate fill_value=0."""
        with pytest.raises(ValueError, match="not an integer"):
            await self._merge(fill_value=0.5)

    async def test_nan_fill_value_on_integer_rejected(self):
        """NaN also landed on 0, with only a RuntimeWarning."""
        with pytest.raises(ValueError, match="cannot be represented"):
            await self._merge(fill_value=float("nan"))

    async def test_nan_fill_value_on_float_dtype_allowed(self):
        cog = _make_cog(width=10, height=10, scale=1.0, dtype=np.dtype("f4"))
        result = await merge(
            [cog],
            bbox=BBox(100, 100, 110, 110),  # disjoint from the cog: pure fill
            bbox_crs=32632,
            target_crs=32632,
            target_resolution=1.0,
            fill_value=float("nan"),
        )
        assert np.all(np.isnan(result.data))  # type: ignore[reportUnknownMemberType]

    async def test_inverted_bbox_rejected(self):
        """min()/max() swapped this into the complementary extent."""
        with pytest.raises(ValueError, match="minx < maxx"):
            await self._merge(bbox=BBox(10, 0, 0, 10))

    async def test_float_band_index_rejected(self):
        with pytest.raises(ValueError, match="must be integers"):
            await self._merge(band_indices=[1.5])

    async def test_valid_arguments_still_reach_the_read(self):
        """Guards the guard: the checks above must not reject a normal call."""
        cog = _make_cog(width=10, height=10, scale=1.0, bands=1)
        cog._read_native = AsyncMock(
            return_value=_make_array(
                np.ones((1, 10, 10), dtype=np.uint16),
                transform=Affine(1, 0, 0, 0, -1, 10),
                geotiff=cog._geotiff,
            )
        )
        await merge(
            [cog],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            band_indices=[1],
            fill_value=0,
            target_crs=32632,
            target_resolution=1.0,
            mosaic_method="last",
            crs_method="first",
            resampling="nearest",
        )
        cog._read_native.assert_called_once()


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
        assert result.res[0] == pytest.approx(2.0)
        cog._read_native.assert_called()

    @pytest.mark.parametrize("resampling", ["bilinear", "cubic"])
    async def test_merge_resampling_preserves_a_constant(
        self, resampling: ResamplingMethod
    ):
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
            resampling=resampling,
        )
        assert result.res[0] == pytest.approx(2.0)
        assert result.data.dtype == np.uint16  # type: ignore[reportUnknownMemberType]
        # Both kernels sum to 1, so a constant input survives downsampling.
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


def _make_windowed_cog(
    origin_x: float, width: int, value: int, *, origin_y: float = 15.0
):
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
        origin_y=origin_y,
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


# ── merge: output grid is a pure function of the arguments ─────────────


def _spy_read_to_grid(cog: Any) -> list[dict[str, Any]]:
    """Record every ``_read_to_grid`` call, then delegate.

    Which merge path ran is otherwise invisible from the output — the whole
    point of the snapped grid is that both paths return the same one.
    """
    calls: list[dict[str, Any]] = []
    real = cog._read_to_grid

    async def _wrapped(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return await real(**kwargs)

    cog._read_to_grid = _wrapped
    return calls


class TestMergeGridInvariance:
    """The output transform and shape are a pure function of
    (bbox, target_resolution, snap_to_grid) — never of source grid phase,
    merge path, tile count, or tile order."""

    # Off-grid on every edge at res 1.0.
    BBOX = BBox(0.8, 0.3, 13.8, 10.3)
    # Its edges rounded outward onto the resolution grid.
    GRID = (Affine(1, 0, 0.0, 0, -1, 11.0), 14, 11)

    @staticmethod
    def _grid(result: RasterArray) -> tuple[Affine, int, int]:
        return result.transform, result.width, result.height

    async def _merge(self, cogs: list[Any], **kwargs: Any) -> RasterArray:
        return await merge(
            cogs,
            bbox=self.BBOX,
            bbox_crs=32632,
            target_crs=32632,
            target_resolution=1.0,
            **kwargs,
        )

    @pytest.mark.parametrize("phase", [0.0, 0.25, 0.5])
    async def test_grid_ignores_source_phase(self, phase: float):
        left = _make_windowed_cog(
            origin_x=-10.0 + phase, width=20, value=1, origin_y=15.0 + phase
        )
        right = _make_windowed_cog(
            origin_x=6.0 + phase, width=20, value=2, origin_y=15.0 + phase
        )
        result = await self._merge([left, right])
        assert self._grid(result) == self.GRID

    async def test_native_and_warp_paths_agree(self):
        aligned = _make_windowed_cog(origin_x=-10.0, width=30, value=1)
        shifted = _make_windowed_cog(origin_x=-10.5, width=30, value=1, origin_y=15.5)
        native = await self._merge([aligned])
        warped = await self._merge([shifted])
        assert self._grid(native) == self._grid(warped) == self.GRID

    async def test_grid_ignores_tile_count_and_order(self):
        a = _make_windowed_cog(origin_x=-10.0, width=20, value=1)
        b = _make_windowed_cog(origin_x=6.0, width=20, value=2)
        for cogs in ([a], [b], [a, b], [b, a]):
            result = await self._merge(list(cogs))
            assert self._grid(result) == self.GRID

    async def test_snap_to_grid_false_stays_bbox_anchored(self):
        cog = _make_windowed_cog(origin_x=-10.0, width=30, value=1)
        result = await self._merge([cog], snap_to_grid=False)
        assert (result.width, result.height) == (13, 10)
        assert (result.transform.c, result.transform.f) == (
            self.BBOX.minx,
            self.BBOX.maxy,
        )

    async def test_aligned_bbox_is_exact_and_copied_natively(self):
        cog = _make_windowed_cog(origin_x=-10.0, width=30, value=1)
        native_calls = spy_read_native(cog)
        to_grid_calls = _spy_read_to_grid(cog)
        result = await merge(
            [cog],
            bbox=BBox(1.0, 2.0, 12.0, 9.0),
            bbox_crs=32632,
            target_crs=32632,
            target_resolution=1.0,
        )
        assert self._grid(result) == (Affine(1, 0, 1.0, 0, -1, 9.0), 11, 7)
        assert native_calls and not to_grid_calls

    async def test_offgrid_source_is_resampled_to_the_snapped_grid(self):
        cog = _make_windowed_cog(origin_x=-10.5, width=30, value=1, origin_y=15.5)
        to_grid_calls = _spy_read_to_grid(cog)
        result = await self._merge([cog])
        assert self._grid(result) == self.GRID
        assert to_grid_calls

    async def test_mixed_phase_inputs_merge_in_either_order(self):
        aligned = _make_windowed_cog(origin_x=-10.0, width=20, value=1)
        shifted = _make_windowed_cog(origin_x=6.5, width=20, value=2, origin_y=15.5)
        first = await self._merge([aligned, shifted])
        second = await self._merge([shifted, aligned])
        assert self._grid(first) == self._grid(second) == self.GRID

    async def test_uncovered_margin_gets_fill_value(self):
        cog = _make_windowed_cog(origin_x=0.0, width=10, value=7)
        result = await merge(
            [cog],
            bbox=BBox(0.3, 0.0, 10.3, 10.0),
            bbox_crs=32632,
            target_crs=32632,
            target_resolution=1.0,
            fill_value=9,
        )
        data: np.ndarray[Any, Any] = result.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (1, 10, 11)
        # The tile ends at x=10; the snapped grid's last column (x 10..11) is
        # covered by no input and must carry the fill.
        assert (data[0, :, 10] == 9).all()
        assert (data[0, :, :10] == 7).all()


# ── _resolve_target_crs ────────────────────────────────────────────────


class TestResolveTargetCrs:
    @pytest.mark.parametrize(
        ("crs_list", "method", "expected"),
        [
            ([32632, 32633, 32632], "most_common", 32632),
            ([32633, 32632, 32632], "first", 32633),
            ([None, 32632], "first", 32632),
            ([None, 32632], "most_common", 32632),
            ([4326], "most_common", 4326),
            ([4326], "first", 4326),
        ],
    )
    def test_picks(
        self,
        crs_list: list[int | None],
        method: Literal["most_common", "first"],
        expected: int,
    ):
        cogs = [_make_cog(crs=c) for c in crs_list]
        assert _resolve_target_crs(cogs, method) == expected

    def test_all_none_raises(self):
        cogs = [_make_cog(crs=None), _make_cog(crs=None)]
        with pytest.raises(ValueError, match="No CRS found"):
            _resolve_target_crs(cogs, "most_common")


# ── concurrency: merge ─────────────────────────────────────────────


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

    async def test_first_mode_still_early_exits(self):
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


# ── The resampling seam, shared with read() ─────────────────────────────


def _real_utm_cog(with_overview: bool = False) -> tuple[AsyncGeoTIFF, Any]:
    """A real AsyncGeoTIFF over a stub GeoTIFF at a UTM 33N origin.

    Real, not a MagicMock, so ``read()`` and ``merge()`` can be compared on the
    same object and both go through the production seam.
    """
    gt = make_mock_geotiff(
        width=400,
        height=400,
        scale=10.0,
        count=1,
        origin_x=300000.0,
        origin_y=5700000.0,
        crs_epsg=32633,
    )
    gt.read = slicing_read(gt, np.zeros((1, 400, 400), np.uint16))
    ov = None
    if with_overview:
        ov = make_mock_geotiff(
            width=200,
            height=200,
            scale=20.0,
            count=1,
            origin_x=300000.0,
            origin_y=5700000.0,
            crs_epsg=32633,
        )
        ov.read = slicing_read(ov, np.zeros((1, 200, 200), np.uint16))
        gt.overviews = [ov]
    return AsyncGeoTIFF("s3://b/k.tif", gt), ov


# An AOI inside the tile's 4326 footprint.
_AOI = BBox(12.1386, 51.3893, 12.1595, 51.4042)


class TestMergeSharesTheWarpSeam:
    async def test_halo_is_per_axis(self):
        """merge sized the halo from an x-only resolution ratio and applied it
        to both axes, so the y edges came from a truncated kernel."""
        cog, _ = _real_utm_cog()
        calls = spy_read_native(cog)
        res = 0.0002

        result = await merge(
            [cog],
            bbox=_AOI,
            bbox_crs=4326,
            target_crs=4326,
            target_resolution=res,
            resampling="cubic",
        )

        assert len(calls) == 1
        got = BBox(*calls[0]["bbox"])
        # Reference frame: the result's own extent, unpadded, so the assertion
        # measures the halo alone rather than halo plus grid convention.
        unpadded = transform_bbox(
            bounds_from_transform(result.transform, result.width, result.height),
            4326,
            32633,
        )
        pad_x = unpadded.minx - got.minx
        pad_y = unpadded.miny - got.miny
        # 0.0002 deg is ~14.5 m of easting but ~23 m of northing, so y needs the
        # wider kernel and therefore the deeper halo.
        assert pad_y > pad_x

    async def test_read_and_merge_agree_on_overview(self):
        """Both derived "target resolution in source units" from a bbox width
        ratio, but from *different* bboxes, so the same arguments could select
        different overviews — and overview pixels are pre-averaged aggregates.
        """
        # Chosen to straddle the 20 m overview under the two old denominators:
        # merge's whole-image ratio gave 19.6 m (no overview), a read-region
        # ratio gives 21.2 m (overview selected).
        res = 2.9233e-04

        reader_cog, _ = _real_utm_cog(with_overview=True)
        reader_calls = spy_read_native(reader_cog)
        await reader_cog.read(
            bbox=_AOI,
            bbox_crs=4326,
            target_crs=4326,
            target_resolution=res,
            use_overviews=True,
            resampling="bilinear",
        )

        merge_cog, _ = _real_utm_cog(with_overview=True)
        merge_calls = spy_read_native(merge_cog)
        await merge(
            [merge_cog],
            bbox=_AOI,
            bbox_crs=4326,
            target_crs=4326,
            target_resolution=res,
            use_overviews=True,
            resampling="bilinear",
        )

        # Output grids differ by design (read ceils to cover the bbox, merge
        # rounds to match GDAL), so compare the decision, not the pixels.
        # Assert the decision itself, not just agreement: "both None" would
        # otherwise pass while use_overviews silently stopped working.
        assert reader_calls[0]["overview"] is not None
        assert (reader_calls[0]["overview"] is not None) == (
            merge_calls[0]["overview"] is not None
        )

    async def test_thin_edge_contributor_reads_a_sane_halo(self):
        """A contributor clipping a thin column of the output gets a 1-px-wide
        sub-grid. Sizing the source resolution from that sub-grid's *extents*
        read the projection's curvature over the tall axis as apparent width,
        inflating the halo ~250x (30 m -> 7330 m) and picking a wrong overview.
        """
        cog, _ = _real_utm_cog()
        calls = spy_read_native(cog)
        res = 0.0002
        tile = transform_bbox(BBox(*cog._geotiff.bounds), 32633, 4326)

        # An output bbox overlapping only a sliver of the tile's left edge, but
        # spanning its full height.
        await merge(
            [cog],
            bbox=BBox(tile.minx - 20 * res, tile.miny, tile.minx + res, tile.maxy),
            bbox_crs=4326,
            target_crs=4326,
            target_resolution=res,
            resampling="cubic",
        )

        assert len(calls) == 1
        got = BBox(*calls[0]["bbox"])
        # 10 m source, cubic, ~1.4x downsample -> 3 source px each way. Allow
        # generous slack; the bug produced hundreds of pixels, not a few.
        assert got.width < 500.0, f"halo blew up: read {got.width:.0f} m wide"
        assert got.height < cog._geotiff.bounds[3] - cog._geotiff.bounds[1] + 500.0
