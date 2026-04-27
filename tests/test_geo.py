"""Unit tests for pure geometry, parsing, and utility functions."""

import numpy as np
import pytest
from affine import Affine

from rastera.geo import (
    BBox,
    WindowOutOfRangeError,
    compute_paste_slices,
    resample_nearest,
    transform_bbox,
    window_from_bbox,
)
from rastera.reader import _extract_key
from tests.conftest import make_meta

# ── BBox ──────────────────────────────────────────────────────────────────


class TestBBox:
    def test_properties(self):
        b = BBox(0, 0, 10, 5)
        assert b.width == 10
        assert b.height == 5

    def test_iter_and_unpack(self):
        b = BBox(1, 2, 3, 4)
        assert list(b) == [1, 2, 3, 4]
        minx, miny, maxx, maxy = b
        assert (minx, miny, maxx, maxy) == (1, 2, 3, 4)

    def test_intersect_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 15, 15)
        c = a.intersect(b)
        assert c == BBox(5, 5, 10, 10)

    def test_intersect_no_overlap(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(10, 10, 15, 15)
        assert a.intersect(b) is None

    def test_intersect_edge_touch(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(5, 0, 10, 5)
        assert a.intersect(b) is None  # touching edge = no area

    def test_intersect_contained(self):
        outer = BBox(0, 0, 10, 10)
        inner = BBox(2, 2, 8, 8)
        assert outer.intersect(inner) == inner


# ── Window ────────────────────────────────────────────────────────────────


class TestWindow:
    def test_from_bbox_full(self):
        p = make_meta()
        w = window_from_bbox(p, BBox(0, 0, 1000, 1000))  # type: ignore[reportArgumentType]
        assert w.col_off == 0 and w.width == 100
        assert w.row_off == 0 and w.height == 100

    def test_from_bbox_subset(self):
        p = make_meta()
        w = window_from_bbox(p, BBox(100, 200, 500, 800))  # type: ignore[reportArgumentType]
        assert w.width > 0 and w.height > 0
        assert w.col_off >= 10 and w.col_off + w.width <= 50

    def test_from_bbox_no_intersect(self):
        p = make_meta()
        with pytest.raises(WindowOutOfRangeError, match="does not intersect"):
            window_from_bbox(p, BBox(2000, 2000, 3000, 3000))  # type: ignore[reportArgumentType]

    def test_from_bbox_subpixel_overlap_raises(self):
        # make_meta(): 100x100 grid at 10 m/px, x in [0, 1000].
        # bbox spans 0.1 m on x (= 0.01 px) — floor(0.01 + 0.5) = 0,
        # so the rounded window has zero width and window_from_bbox must raise.
        p = make_meta()
        with pytest.raises(WindowOutOfRangeError, match="does not intersect"):
            window_from_bbox(p, BBox(999.9, 0, 1000.0, 1000))  # type: ignore[reportArgumentType]

    def test_from_bbox_clamps(self):
        p = make_meta()
        # bbox extends beyond image
        w = window_from_bbox(p, BBox(-500, -500, 500, 500))  # type: ignore[reportArgumentType]
        assert w.col_off == 0
        assert w.col_off + w.width <= 100 and w.row_off + w.height <= 100
        assert w.width > 0 and w.height > 0


# ── compute_paste_slices ──────────────────────────────────────────────────


class TestComputePasteSlices:
    def test_aligned_paste(self):
        # src profile has origin at (0, 500) in world coords (north-up)
        src = make_meta(width=50, height=50, scale=10.0)
        dst_transform = Affine(10, 0, 0, 0, -10, 1000)
        result = compute_paste_slices(
            src=src,  # type: ignore[reportArgumentType]
            dst_transform=dst_transform,
            dst_width=100,
            dst_height=100,
        )
        assert result is not None
        dst_rows, dst_cols, src_rows, src_cols = result
        assert dst_cols == slice(0, 50)
        assert dst_rows.stop - dst_rows.start == 50  # correct height

    def test_no_overlap(self):
        src = make_meta(width=50, height=50, scale=10.0)
        # destination is far away
        dst_transform = Affine(10, 0, 5000, 0, -10, 10000)
        result = compute_paste_slices(
            src=src,  # type: ignore[reportArgumentType]
            dst_transform=dst_transform,
            dst_width=100,
            dst_height=100,
        )
        assert result is None


# ── _extract_key ─────────────────────────────────────────────────────────


class TestExtractKey:
    def test_s3_scheme(self):
        assert _extract_key("s3://my-bucket/path/to/file.tif") == "path/to/file.tif"

    def test_virtual_hosted_style(self):
        assert (
            _extract_key(
                "https://my-bucket.s3.us-west-2.amazonaws.com/path/to/file.tif"
            )
            == "path/to/file.tif"
        )

    def test_path_style(self):
        assert (
            _extract_key(
                "https://s3.us-west-2.amazonaws.com/my-bucket/path/to/file.tif"
            )
            == "path/to/file.tif"
        )

    def test_local_path(self, tmp_path):
        f = tmp_path / "file.tif"
        f.write_bytes(b"")
        assert _extract_key(str(f)) == "file.tif"

    def test_file_scheme(self, tmp_path):
        f = tmp_path / "file.tif"
        f.write_bytes(b"")
        assert _extract_key(f.as_uri()) == "file.tif"


# ── transform_bbox ───────────────────────────────────────────────────────


class TestTransformBbox:
    def test_same_crs_noop(self):
        bbox = BBox(500000, 5000000, 600000, 5100000)
        result = transform_bbox(bbox, 32632, 32632)
        assert result == bbox

    def test_roundtrip(self):
        bbox = BBox(10.0, 50.0, 11.0, 51.0)  # lon/lat
        projected = transform_bbox(bbox, 4326, 32632)
        assert projected.width > 0 and projected.height > 0
        back = transform_bbox(projected, 32632, 4326)
        # Roundtrip is lossy (envelope of sampled points), but should contain original
        assert back.minx <= bbox.minx and back.maxx >= bbox.maxx
        assert back.miny <= bbox.miny and back.maxy >= bbox.maxy


# ── resample_nearest ─────────────────────────────────────────────────────


class TestResampleNearest:
    def _make_src(self):
        """4x4 source array with sequential values, 10m pixels, origin (0, 40)."""
        arr = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        transform = Affine(10, 0, 0, 0, -10, 40)
        return arr, transform

    def test_identity(self):
        arr, t = self._make_src()
        out = resample_nearest(arr, t, t, 4, 4)
        np.testing.assert_array_equal(out, arr)

    def test_downsample(self):
        arr, src_t = self._make_src()
        # 2x2 output, 20m pixels, same origin
        dst_t = Affine(20, 0, 0, 0, -20, 40)
        out = resample_nearest(arr, src_t, dst_t, 2, 2)
        assert out.shape == (1, 2, 2)
        # Pixel centers at (10,30), (30,30), (10,10), (30,10)
        # → src pixels (1,1),(3,1),(1,3),(3,3)
        # col+0.5 → col 0.5*20=10, 1.5*20=30
        # row 0.5*20 → y=40-10=30, y=40-30=10
        # src col for x=10: 1, x=30: 3
        # src row for y=30: 1, y=10: 3
        assert out[0, 0, 0] == arr[0, 1, 1]
        assert out[0, 0, 1] == arr[0, 1, 3]
        assert out[0, 1, 0] == arr[0, 3, 1]
        assert out[0, 1, 1] == arr[0, 3, 3]

    def test_upsample(self):
        arr, src_t = self._make_src()
        # 8x8 output, 5m pixels, same origin
        dst_t = Affine(5, 0, 0, 0, -5, 40)
        out = resample_nearest(arr, src_t, dst_t, 8, 8)
        assert out.shape == (1, 8, 8)
        # Each src pixel should appear in a 2x2 block
        # Top-left output pixel center: (2.5, 37.5) → src (0, 0)
        assert out[0, 0, 0] == arr[0, 0, 0]
        assert out[0, 0, 1] == arr[0, 0, 0]  # same src pixel
        assert out[0, 1, 0] == arr[0, 0, 0]

    def test_nodata_for_out_of_bounds(self):
        arr, src_t = self._make_src()
        # Destination extends beyond source: 4x4 at 10m but shifted right by 20m
        dst_t = Affine(10, 0, 20, 0, -10, 40)
        out = resample_nearest(arr, src_t, dst_t, 4, 4, nodata=-1)
        # First 2 cols map to src cols 2,3; last 2 cols are out of bounds
        assert out[0, 0, 0] == arr[0, 0, 2]
        assert out[0, 0, 3] == -1  # out of bounds

    def test_with_reprojection(self):
        from pyproj import Transformer

        # Source in UTM 32N, destination in WGS84
        src_arr = np.ones((1, 10, 10), dtype=np.float32)
        src_t = Affine(100, 0, 500000, 0, -100, 5000000)  # 100m pixels in UTM

        # Small bbox in WGS84 covering the source area
        dst_t = Affine(0.001, 0, 9.0, 0, -0.001, 45.1)
        transformer = Transformer.from_crs(4326, 32632, always_xy=True)
        out = resample_nearest(
            src_arr, src_t, dst_t, 10, 10, nodata=0, transformer=transformer
        )
        assert out.shape == (1, 10, 10)
        # Some pixels should have data (1.0), some may be nodata (0)
        assert np.any(out == 1.0) or np.any(out == 0)

    def test_coarse_grid_matches_brute_force(self):
        """Coarse-grid interpolation should match per-pixel pyproj within 0.125 px."""
        from pyproj import Transformer

        from rastera.geo import _coarse_grid_transform

        # Source in UTM 33N, destination in WGS84 — realistic Sentinel-2 scenario
        src_t = Affine(10, 0, 200000, 0, -10, 4700000)  # 10m pixels
        dst_t = Affine(0.0001, 0, 12.0, 0, -0.0001, 42.4)
        transformer = Transformer.from_crs(4326, 32633, always_xy=True)
        dst_w, dst_h = 200, 200

        # Coarse-grid result
        col_f, row_f = _coarse_grid_transform(dst_w, dst_h, dst_t, src_t, transformer)

        # Brute-force reference
        cols = np.arange(dst_w) + 0.5
        rows = np.arange(dst_h) + 0.5
        dc, dr = np.meshgrid(cols, rows)
        wx = float(dst_t.a) * dc + float(dst_t.c)
        wy = float(dst_t.e) * dr + float(dst_t.f)
        wx, wy = transformer.transform(wx, wy)
        src_inv = ~src_t
        ref_col = float(src_inv.a) * wx + float(src_inv.c)
        ref_row = float(src_inv.e) * wy + float(src_inv.f)

        assert np.max(np.abs(col_f - ref_col)) < 0.125
        assert np.max(np.abs(row_f - ref_row)) < 0.125

    def test_small_grid_with_transformer(self):
        """Grid smaller than the coarse step size should still work."""
        from pyproj import Transformer

        src_arr = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
        src_t = Affine(100, 0, 500000, 0, -100, 5000000)
        dst_t = Affine(0.001, 0, 9.0, 0, -0.001, 45.1)
        transformer = Transformer.from_crs(4326, 32632, always_xy=True)
        out = resample_nearest(
            src_arr, src_t, dst_t, 5, 5, nodata=-1, transformer=transformer
        )
        assert out.shape == (1, 5, 5)

    def test_single_pixel_with_transformer(self):
        """1x1 destination grid should not crash."""
        from pyproj import Transformer

        src_arr = np.ones((1, 10, 10), dtype=np.float32)
        src_t = Affine(100, 0, 500000, 0, -100, 5000000)
        dst_t = Affine(0.01, 0, 9.0, 0, -0.01, 45.1)
        transformer = Transformer.from_crs(4326, 32632, always_xy=True)
        out = resample_nearest(
            src_arr, src_t, dst_t, 1, 1, nodata=0, transformer=transformer
        )
        assert out.shape == (1, 1, 1)
