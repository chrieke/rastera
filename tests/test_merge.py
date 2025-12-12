"""Unit tests for merge_cogs and helpers."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from affine import Affine

from rastera.geo import BBox
from rastera.merge import merge_cogs, _mosaic_grid_from_bbox, _require_compatible_merge_inputs
from rastera.meta import Profile
from tests.conftest import make_mock_ifd, make_profile


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_cog(
    width=100, height=100, scale=10.0, bands=1, origin_x=0.0, origin_y=None, crs=32632,
):
    """Build a mock AsyncGeoTIFF with a realistic Profile."""
    if origin_y is None:
        origin_y = height * scale
    transform = Affine(scale, 0, origin_x, 0, -scale, origin_y)
    bounds = BBox(origin_x, origin_y - height * scale, origin_x + width * scale, origin_y)

    profile = Profile(
        width=width, height=height, count=bands,
        dtype=np.dtype("u2"), transform=transform,
        res=(scale, scale), crs_epsg=crs, bounds=bounds,
    )

    ifd = make_mock_ifd(width=width, height=height, scale=scale, bands=bands, crs_epsg=crs)
    cog = MagicMock()
    cog.profile = profile
    cog.ifd = ifd
    cog.read = AsyncMock()
    return cog


# ── _mosaic_grid_from_bbox ───────────────────────────────────────────────


class TestMosaicGridFromBbox:
    def test_aligned_bbox(self):
        base_transform = Affine(10, 0, 0, 0, -10, 1000)
        bbox = BBox(100, 500, 300, 800)
        transform, w, h, bounds = _mosaic_grid_from_bbox(
            base_transform=base_transform, bbox=bbox,
        )
        assert w == 20
        assert h == 30
        assert bounds.minx == 100.0
        assert bounds.maxy == 800.0

    def test_subpixel_bbox_still_produces_grid(self):
        base_transform = Affine(10, 0, 0, 0, -10, 1000)
        # A tiny bbox within a single pixel still produces a 1x1 grid
        bbox = BBox(5, 5, 6, 6)
        _, w, h, _ = _mosaic_grid_from_bbox(base_transform=base_transform, bbox=bbox)
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
        read_profile = Profile(
            width=5, height=5, count=1, dtype=np.dtype("u2"),
            transform=Affine(1, 0, 2, 0, -1, 8),
            res=(1.0, 1.0), crs_epsg=32632,
            bounds=BBox(2, 3, 7, 8),
        )
        cog._read_native = AsyncMock(return_value=(read_arr, read_profile))

        result_arr, result_profile = await merge_cogs(
            [cog], bbox=BBox(0, 0, 10, 10), bbox_crs=32632, band_indices=[1],
        )
        assert result_arr.shape[0] == 1  # 1 band
        assert result_profile.crs_epsg == 32632
        cog._read_native.assert_called_once()

    async def test_no_cogs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            await merge_cogs([], bbox=BBox(0, 0, 10, 10), bbox_crs=32632)

    async def test_no_bbox_crs_raises(self):
        cog = _make_cog()
        with pytest.raises(ValueError, match="bbox_crs is required"):
            await merge_cogs([cog], bbox=BBox(0, 0, 10, 10))

    async def test_fill_value_used(self):
        """When a COG doesn't intersect the bbox, the output should be fill_value."""
        cog = _make_cog(width=10, height=10, scale=1.0, origin_x=100, origin_y=110)
        # bbox is at (0,0)-(10,10), cog is at (100,100)-(110,110) — no overlap
        result_arr, _ = await merge_cogs(
            [cog], bbox=BBox(0, 0, 10, 10), bbox_crs=32632,
            band_indices=[1], fill_value=9999,
        )
        assert np.all(result_arr == 9999)

    async def test_two_cogs_overlap(self):
        """Two overlapping COGs: second one wins in overlap region."""
        cog1 = _make_cog(width=10, height=10, scale=1.0, bands=1)
        cog2 = _make_cog(width=10, height=10, scale=1.0, bands=1, origin_x=5.0)

        arr1 = np.ones((1, 10, 10), dtype=np.uint16) * 1
        prof1 = Profile(
            width=10, height=10, count=1, dtype=np.dtype("u2"),
            transform=Affine(1, 0, 0, 0, -1, 10),
            res=(1.0, 1.0), crs_epsg=32632, bounds=BBox(0, 0, 10, 10),
        )
        arr2 = np.ones((1, 10, 10), dtype=np.uint16) * 2
        prof2 = Profile(
            width=10, height=10, count=1, dtype=np.dtype("u2"),
            transform=Affine(1, 0, 5, 0, -1, 10),
            res=(1.0, 1.0), crs_epsg=32632, bounds=BBox(5, 0, 15, 10),
        )
        cog1._read_native = AsyncMock(return_value=(arr1, prof1))
        cog2._read_native = AsyncMock(return_value=(arr2, prof2))

        result_arr, result_profile = await merge_cogs(
            [cog1, cog2], bbox=BBox(0, 0, 15, 10), bbox_crs=32632, band_indices=[1],
        )
        # The overlap region (cols 5-9) should have cog2's value (last writer wins)
        assert result_arr.shape == (1, 10, 15)
        assert np.all(result_arr[0, :, :5] == 1)  # cog1 only
        assert np.all(result_arr[0, :, 10:] == 2)  # cog2 only
        assert np.all(result_arr[0, :, 5:10] == 2)  # overlap -> cog2 wins

    async def test_nodata_skipped_in_overlap(self):
        """Nodata pixels in a later COG should not overwrite valid data from earlier COGs."""
        NODATA = 0
        cog1 = _make_cog(width=10, height=10, scale=1.0, bands=1)
        cog1.profile = replace(cog1.profile,nodata=NODATA)
        cog2 = _make_cog(width=10, height=10, scale=1.0, bands=1, origin_x=5.0)
        cog2.profile = replace(cog2.profile,nodata=NODATA)

        arr1 = np.ones((1, 10, 10), dtype=np.uint16) * 42
        prof1 = Profile(
            width=10, height=10, count=1, dtype=np.dtype("u2"),
            transform=Affine(1, 0, 0, 0, -1, 10),
            res=(1.0, 1.0), crs_epsg=32632, bounds=BBox(0, 0, 10, 10),
            nodata=NODATA,
        )
        # cog2: left half is nodata, right half is valid
        arr2 = np.zeros((1, 10, 10), dtype=np.uint16)  # all nodata
        arr2[:, :, 5:] = 99  # right half is valid
        prof2 = Profile(
            width=10, height=10, count=1, dtype=np.dtype("u2"),
            transform=Affine(1, 0, 5, 0, -1, 10),
            res=(1.0, 1.0), crs_epsg=32632, bounds=BBox(5, 0, 15, 10),
            nodata=NODATA,
        )
        cog1._read_native = AsyncMock(return_value=(arr1, prof1))
        cog2._read_native = AsyncMock(return_value=(arr2, prof2))

        result_arr, _ = await merge_cogs(
            [cog1, cog2], bbox=BBox(0, 0, 15, 10), bbox_crs=32632, band_indices=[1],
        )
        assert result_arr.shape == (1, 10, 15)
        # cog1-only region: value 42
        assert np.all(result_arr[0, :, :5] == 42)
        # overlap where cog2 has nodata: cog1's value (42) preserved
        assert np.all(result_arr[0, :, 5:10] == 42)
        # cog2-only valid region: value 99
        assert np.all(result_arr[0, :, 10:] == 99)

    async def test_nan_nodata_skipped(self):
        """NaN nodata pixels should be transparent during merge."""
        cog1 = _make_cog(width=10, height=10, scale=1.0, bands=1)
        cog1.profile = replace(cog1.profile,dtype=np.dtype("f4"), nodata=float("nan"))
        cog2 = _make_cog(width=10, height=10, scale=1.0, bands=1)
        cog2.profile = replace(cog2.profile,dtype=np.dtype("f4"), nodata=float("nan"))

        arr1 = np.full((1, 10, 10), 5.0, dtype=np.float32)
        prof1 = Profile(
            width=10, height=10, count=1, dtype=np.dtype("f4"),
            transform=Affine(1, 0, 0, 0, -1, 10),
            res=(1.0, 1.0), crs_epsg=32632, bounds=BBox(0, 0, 10, 10),
            nodata=float("nan"),
        )
        # cog2: top half is NaN, bottom half is valid
        arr2 = np.full((1, 10, 10), np.nan, dtype=np.float32)
        arr2[:, 5:, :] = 77.0
        prof2 = Profile(
            width=10, height=10, count=1, dtype=np.dtype("f4"),
            transform=Affine(1, 0, 0, 0, -1, 10),
            res=(1.0, 1.0), crs_epsg=32632, bounds=BBox(0, 0, 10, 10),
            nodata=float("nan"),
        )
        cog1._read_native = AsyncMock(return_value=(arr1, prof1))
        cog2._read_native = AsyncMock(return_value=(arr2, prof2))

        result_arr, _ = await merge_cogs(
            [cog1, cog2], bbox=BBox(0, 0, 10, 10), bbox_crs=32632, band_indices=[1],
        )
        # top half: cog2 is NaN so cog1's value (5.0) preserved
        assert np.all(result_arr[0, :5, :] == 5.0)
        # bottom half: cog2 has valid data (77.0) which overwrites
        assert np.all(result_arr[0, 5:, :] == 77.0)

    async def test_nodata_none_still_overwrites(self):
        """When nodata is None, later COGs overwrite earlier ones (original behavior)."""
        cog1 = _make_cog(width=10, height=10, scale=1.0, bands=1)
        cog2 = _make_cog(width=10, height=10, scale=1.0, bands=1)

        arr1 = np.ones((1, 10, 10), dtype=np.uint16) * 42
        prof1 = Profile(
            width=10, height=10, count=1, dtype=np.dtype("u2"),
            transform=Affine(1, 0, 0, 0, -1, 10),
            res=(1.0, 1.0), crs_epsg=32632, bounds=BBox(0, 0, 10, 10),
        )
        arr2 = np.zeros((1, 10, 10), dtype=np.uint16)  # all zeros
        prof2 = Profile(
            width=10, height=10, count=1, dtype=np.dtype("u2"),
            transform=Affine(1, 0, 0, 0, -1, 10),
            res=(1.0, 1.0), crs_epsg=32632, bounds=BBox(0, 0, 10, 10),
        )
        cog1._read_native = AsyncMock(return_value=(arr1, prof1))
        cog2._read_native = AsyncMock(return_value=(arr2, prof2))

        result_arr, _ = await merge_cogs(
            [cog1, cog2], bbox=BBox(0, 0, 10, 10), bbox_crs=32632, band_indices=[1],
        )
        # nodata=None, so cog2's zeros overwrite cog1's 42s
        assert np.all(result_arr == 0)
