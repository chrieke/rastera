"""Unit tests for the resample() function and its kernel/coord helpers."""

import numpy as np
from affine import Affine

from rastera.resampling import resample

# ── resample (nearest) ───────────────────────────────────────────────────


class TestResampleNearest:
    def _make_src(self):
        """4x4 source array with sequential values, 10m pixels, origin (0, 40)."""
        arr = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        transform = Affine(10, 0, 0, 0, -10, 40)
        return arr, transform

    def test_identity(self):
        arr, t = self._make_src()
        out = resample(arr, t, t, 4, 4)
        np.testing.assert_array_equal(out, arr)

    def test_downsample(self):
        arr, src_t = self._make_src()
        # 2x2 output, 20m pixels, same origin
        dst_t = Affine(20, 0, 0, 0, -20, 40)
        out = resample(arr, src_t, dst_t, 2, 2)
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
        out = resample(arr, src_t, dst_t, 8, 8)
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
        out = resample(arr, src_t, dst_t, 4, 4, nodata=-1)
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
        out = resample(
            src_arr, src_t, dst_t, 10, 10, nodata=0, transformer=transformer
        )
        assert out.shape == (1, 10, 10)
        # Some pixels should have data (1.0), some may be nodata (0)
        assert np.any(out == 1.0) or np.any(out == 0)

    def test_coarse_grid_matches_brute_force(self):
        """Coarse-grid interpolation should match per-pixel pyproj within 0.125 px."""
        from pyproj import Transformer

        from rastera.resampling import _coarse_grid_transform

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
        out = resample(
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
        out = resample(
            src_arr, src_t, dst_t, 1, 1, nodata=0, transformer=transformer
        )
        assert out.shape == (1, 1, 1)


# ── resample (bilinear) ──────────────────────────────────────────────────


class TestResampleBilinear:
    def _make_src(self):
        """4x4 source array with sequential values, 10m pixels, origin (0, 40)."""
        arr = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        transform = Affine(10, 0, 0, 0, -10, 40)
        return arr, transform

    def test_identity(self):
        """Same grid, frac=0 at every pixel center → output equals source."""
        arr, t = self._make_src()
        out = resample(arr, t, t, 4, 4, method="bilinear")
        np.testing.assert_allclose(out, arr)

    def test_midpoint_average(self):
        """Sample halfway between two pixel centers → mean of the two."""
        # Two horizontal pixels: values 10, 20.  Pixel centers at world x=5, 15.
        # Sample at world (10, 5), exactly between them → expect 15.
        arr = np.array([[[10.0, 20.0]]], dtype=np.float32)  # (1, 1, 2)
        src_t = Affine(10, 0, 0, 0, -10, 10)
        # Single dst pixel whose center is at world (10, 5):
        # center_x = a*0.5 + c, center_y = e*0.5 + f.
        # With dst pixel width 1 and origin (9.5, 5.5): center at (10, 5).
        dst_t = Affine(1, 0, 9.5, 0, -1, 5.5)
        out = resample(arr, src_t, dst_t, 1, 1, method="bilinear")
        np.testing.assert_allclose(out, [[[15.0]]])

    def test_quarter_offset(self):
        """Sample 25% of the way from one center to the next → 0.75 * a + 0.25 * b."""
        arr = np.array([[[10.0, 20.0]]], dtype=np.float32)
        src_t = Affine(10, 0, 0, 0, -10, 10)
        # Sample at world (7.5, 5): between center 0 (x=5) and center 1 (x=15),
        # at 25% from center 0 → expect 0.75 * 10 + 0.25 * 20 = 12.5
        dst_t = Affine(1, 0, 7.0, 0, -1, 5.5)
        out = resample(arr, src_t, dst_t, 1, 1, method="bilinear")
        np.testing.assert_allclose(out, [[[12.5]]])

    def test_no_overshoot(self):
        """Bilinear output is always within [src_min, src_max] (convex combo)."""
        rng = np.random.default_rng(42)
        arr = rng.random((1, 8, 8), dtype=np.float32) * 100
        src_t = Affine(10, 0, 0, 0, -10, 80)
        # Upsample 2x with arbitrary phase
        dst_t = Affine(5, 0, 2.3, 0, -5, 78.7)
        out = resample(arr, src_t, dst_t, 16, 16, method="bilinear")
        assert out.min() >= arr.min() - 1e-5
        assert out.max() <= arr.max() + 1e-5

    def test_nodata_center_gate(self):
        """If the src pixel under the dst center is nodata, output is nodata."""
        arr = np.array(
            [[[1.0, 1.0, 1.0], [1.0, -9999.0, 1.0], [1.0, 1.0, 1.0]]],
            dtype=np.float32,
        )
        src_t = Affine(10, 0, 0, 0, -10, 30)
        # Dst pixel center at world (15, 15) → src pixel (1, 1) (the nodata one).
        dst_t = Affine(1, 0, 14.5, 0, -1, 15.5)
        out = resample(arr, src_t, dst_t, 1, 1, nodata=-9999.0, method="bilinear")
        np.testing.assert_array_equal(out, [[[-9999.0]]])

    def test_nodata_renormalize(self):
        """Partial nodata in 2×2 window → output uses renormalized survivors.

        The center pixel (under the dst center) must be valid for renormalize
        to fire — otherwise the GDAL center-gate would set the output to
        nodata regardless.
        """
        # Source: nodata at (0, 1); all other pixels = 10.
        # Dst center at world (10, 10) → src corner (1, 1); floor picks
        # pixel (1, 1) which is valid (= 10).  The 2×2 kernel samples
        # all four pixels equally (weight 0.25 each); the one nodata sample
        # is dropped and the rest renormalize to 10.
        arr = np.array([[[10.0, -9999.0], [10.0, 10.0]]], dtype=np.float32)
        src_t = Affine(10, 0, 0, 0, -10, 20)
        dst_t = Affine(1, 0, 9.5, 0, -1, 10.5)
        out = resample(arr, src_t, dst_t, 1, 1, nodata=-9999.0, method="bilinear")
        np.testing.assert_allclose(out, [[[10.0]]])

    def test_nodata_all_invalid(self):
        """Fully nodata 2x2 window → output is nodata."""
        arr = np.full((1, 2, 2), -9999.0, dtype=np.float32)
        src_t = Affine(10, 0, 0, 0, -10, 20)
        dst_t = Affine(1, 0, 9.5, 0, -1, 10.5)
        out = resample(arr, src_t, dst_t, 1, 1, nodata=-9999.0, method="bilinear")
        np.testing.assert_array_equal(out, [[[-9999.0]]])

    def test_oob_fill(self):
        """Dst pixel whose center is outside source extent → nodata."""
        arr, src_t = self._make_src()
        # Shift dst origin so two columns fall outside the source.
        dst_t = Affine(10, 0, 20, 0, -10, 40)
        out = resample(arr, src_t, dst_t, 4, 4, nodata=-1, method="bilinear")
        # First 2 cols map to src cols 2, 3; last 2 cols are OOB.
        np.testing.assert_array_equal(out[0, :, 2], -1)
        np.testing.assert_array_equal(out[0, :, 3], -1)

    def test_with_reprojection(self):
        from pyproj import Transformer

        src_arr = np.ones((1, 10, 10), dtype=np.float32)
        src_t = Affine(100, 0, 500000, 0, -100, 5000000)
        dst_t = Affine(0.001, 0, 9.0, 0, -0.001, 45.1)
        transformer = Transformer.from_crs(4326, 32632, always_xy=True)
        out = resample(
            src_arr,
            src_t,
            dst_t,
            10,
            10,
            nodata=0,
            transformer=transformer,
            method="bilinear",
        )
        assert out.shape == (1, 10, 10)
        assert not np.any(np.isnan(out))

    def test_anti_aliasing_spreads_delta_on_downsample(self):
        """4× downsample of a single bright pixel: GDAL-style anti-aliasing
        expands the kernel so the peak is spread over many output pixels,
        each receiving a small fraction of the source value.

        Regression: without the anti-aliasing expansion, a strict 2×2
        bilinear kernel would deposit nearly the entire 1000.0 in a
        single output pixel.  With expansion, the max output is bounded
        by the peak tent-kernel weight (~0.22 × 0.22 ≈ 0.05) × 1000 ≈ 50.
        """
        arr = np.zeros((1, 16, 16), dtype=np.float32)
        arr[0, 8, 8] = 1000.0
        src_t = Affine(1, 0, 0, 0, -1, 16)
        dst_t = Affine(4, 0, 0, 0, -4, 16)  # 4× downsample → 4×4
        out = resample(arr, src_t, dst_t, 4, 4, method="bilinear")
        assert out.max() < 100, (
            f"anti-aliased bilinear 4× downsample should spread delta "
            f"peak; got max={out.max()} (strict 2×2 would deposit ~1000)"
        )

    def test_nodata_nan_sentinel(self):
        """NaN-sentinel nodata: NaN samples must be excluded from kernel
        renormalization (NaN != NaN, so naive `sample != nodata` would
        treat them as valid) and must not leak through `NaN * 0 = NaN`.

        Same setup as :meth:`test_nodata_renormalize` but with NaN
        instead of -9999.  The codebase already supports NaN nodata in
        ``merge.py``'s paste loop; the resampling kernel must too.
        """
        arr = np.array(
            [[[10.0, np.nan], [10.0, 10.0]]],
            dtype=np.float32,
        )
        src_t = Affine(10, 0, 0, 0, -10, 20)
        dst_t = Affine(1, 0, 9.5, 0, -1, 10.5)
        out = resample(
            arr, src_t, dst_t, 1, 1, nodata=float("nan"), method="bilinear"
        )
        # Renormalized over the three valid 10.0 samples → 10.0 exactly,
        # and no NaN leaks through.
        assert not np.any(np.isnan(out)), f"output should be NaN-free, got {out}"
        np.testing.assert_allclose(out, [[[10.0]]])

    def test_nodata_nan_center_gate(self):
        """NaN sample under the dst center → output is NaN (the GDAL
        center-pixel gate must fire on NaN sentinels too)."""
        arr = np.array(
            [[[1.0, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]]],
            dtype=np.float32,
        )
        src_t = Affine(10, 0, 0, 0, -10, 30)
        dst_t = Affine(1, 0, 14.5, 0, -1, 15.5)  # center → src pixel (1, 1)
        out = resample(
            arr, src_t, dst_t, 1, 1, nodata=float("nan"), method="bilinear"
        )
        assert np.isnan(out[0, 0, 0]), f"expected NaN output, got {out}"


# ── resample (cubic) ─────────────────────────────────────────────────────


class TestResampleCubic:
    def _make_src(self):
        """8x8 source array with sequential values, 10m pixels, origin (0, 80)."""
        arr = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
        transform = Affine(10, 0, 0, 0, -10, 80)
        return arr, transform

    def test_identity(self):
        """Cubic at the same grid returns the source array (frac=0 at centers)."""
        arr, t = self._make_src()
        out = resample(arr, t, t, 8, 8, method="cubic")
        np.testing.assert_allclose(out, arr, atol=1e-5)

    def test_smooth_downsample_vs_nearest(self):
        """On a gradient, cubic downsample differs from nearest snapping."""
        arr, src_t = self._make_src()  # values 0..63
        # 2x2 downsample (4x reduction); use phase that doesn't align with grid
        dst_t = Affine(40, 0, 0, 0, -40, 80)
        out_cubic = resample(arr, src_t, dst_t, 2, 2, method="cubic")
        out_nearest = resample(arr, src_t, dst_t, 2, 2, method="nearest")
        # Cubic should give different values than nearest in general
        assert not np.allclose(out_cubic, out_nearest)
        # Cubic output should still be bounded by source range (with small
        # overshoot tolerance).
        assert out_cubic.min() >= arr.min() - 5
        assert out_cubic.max() <= arr.max() + 5

    def test_kernel_sums_to_one(self):
        """On a constant source, cubic output equals the constant (kernel
        partition of unity)."""
        arr = np.full((1, 8, 8), 42.0, dtype=np.float32)
        src_t = Affine(10, 0, 0, 0, -10, 80)
        # Arbitrary dst grid that requires interpolation
        dst_t = Affine(7.3, 0, 1.7, 0, -7.3, 79.2)
        out = resample(arr, src_t, dst_t, 6, 6, method="cubic")
        np.testing.assert_allclose(out, 42.0, atol=1e-4)

    def test_nodata_center_gate(self):
        """If the src pixel under the dst center is nodata, output is nodata."""
        arr = np.ones((1, 5, 5), dtype=np.float32)
        arr[0, 2, 2] = -9999.0
        src_t = Affine(10, 0, 0, 0, -10, 50)
        # Dst pixel center at world (25, 25) → src pixel (2, 2) (the nodata one).
        dst_t = Affine(1, 0, 24.5, 0, -1, 25.5)
        out = resample(arr, src_t, dst_t, 1, 1, nodata=-9999.0, method="cubic")
        np.testing.assert_array_equal(out, [[[-9999.0]]])

    def test_nodata_renormalize(self):
        """Partial nodata in 4x4 window → valid output from renormalized
        survivors (not propagated to nodata)."""
        # 5x5 source with one nodata in the kernel window but NOT at center.
        arr = np.full((1, 5, 5), 10.0, dtype=np.float32)
        arr[0, 0, 0] = -9999.0  # corner, far from center
        src_t = Affine(10, 0, 0, 0, -10, 50)
        # Dst pixel center at world (25, 25) → src pixel (2, 2); the 4x4
        # window covers src rows [1..4] and cols [1..4] (after the -0.5 shift,
        # base_row=1, base_col=1, taps at -1..2 → src rows 0..3).
        # Source pixel (0, 0) IS in the window (at tap (-1, -1)).
        dst_t = Affine(1, 0, 24.5, 0, -1, 25.5)
        out = resample(arr, src_t, dst_t, 1, 1, nodata=-9999.0, method="cubic")
        # Output should be a finite value close to 10 (the renormalized mean
        # of surviving samples, all of which are 10).
        assert np.isfinite(out[0, 0, 0])
        np.testing.assert_allclose(out, 10.0, atol=1e-4)

    def test_nodata_all_invalid(self):
        """Fully nodata 4x4 window → output is nodata."""
        arr = np.full((1, 4, 4), -9999.0, dtype=np.float32)
        src_t = Affine(10, 0, 0, 0, -10, 40)
        dst_t = Affine(1, 0, 19.5, 0, -1, 20.5)
        out = resample(arr, src_t, dst_t, 1, 1, nodata=-9999.0, method="cubic")
        np.testing.assert_array_equal(out, [[[-9999.0]]])

    def test_nodata_per_dimension_safety(self):
        """No row OR no column of the 4×4 window has ≥2 valid samples →
        output is nodata (GDAL cubic per-dim safety gate)."""
        # 5×5 source, all nodata except center pixel (1, 2) which is valid.
        # Dst center at world (25, 35) → src_col_f=2.5, src_row_f=1.5 →
        # center pixel = (1, 2), valid (no center-gate fire).
        # Cubic base = (1, 2), 4×4 window covers rows 0..3, cols 1..4.
        # Only one valid sample in the window (at (1, 2)), so every row
        # has ≤1 valid and every column has ≤1 valid → per-dim gate fires.
        arr = np.full((1, 5, 5), -9999.0, dtype=np.float32)
        arr[0, 1, 2] = 5.0
        src_t = Affine(10, 0, 0, 0, -10, 50)
        dst_t = Affine(1, 0, 24.5, 0, -1, 35.5)
        out = resample(arr, src_t, dst_t, 1, 1, nodata=-9999.0, method="cubic")
        np.testing.assert_array_equal(out, [[[-9999.0]]])

    def test_oob_fill(self):
        """Dst pixel whose center is outside source extent → nodata."""
        arr, src_t = self._make_src()
        # Shift dst origin so columns fall outside.
        dst_t = Affine(10, 0, 100, 0, -10, 80)
        out = resample(arr, src_t, dst_t, 4, 4, nodata=-1, method="cubic")
        # All output is OOB.
        np.testing.assert_array_equal(out, -1)

    def test_integer_dtype_clip_no_wrap(self):
        """Cubic overshoot on uint8 input near 255 doesn't wrap to 0."""
        # Construct a sharp step that maximises cubic overshoot.
        arr = np.zeros((1, 1, 8), dtype=np.uint8)
        arr[0, 0, :4] = 0
        arr[0, 0, 4:] = 255
        src_t = Affine(1, 0, 0, 0, -1, 1)
        # Upsample 4x; any overshoot beyond 255 should clip to 255 (not wrap).
        dst_t = Affine(0.25, 0, 0, 0, -0.25, 1)
        out = resample(arr, src_t, dst_t, 32, 1, method="cubic")
        assert out.dtype == np.uint8
        # No wrap-around: all values must be in [0, 255], not garbage.
        assert out.min() >= 0
        assert out.max() <= 255

    def test_with_reprojection(self):
        from pyproj import Transformer

        src_arr = np.ones((1, 20, 20), dtype=np.float32)
        src_t = Affine(100, 0, 500000, 0, -100, 5000000)
        dst_t = Affine(0.001, 0, 9.0, 0, -0.001, 45.1)
        transformer = Transformer.from_crs(4326, 32632, always_xy=True)
        out = resample(
            src_arr,
            src_t,
            dst_t,
            10,
            10,
            nodata=0,
            transformer=transformer,
            method="cubic",
        )
        assert out.shape == (1, 10, 10)
        assert not np.any(np.isnan(out))

    def test_anti_aliasing_spreads_delta_on_downsample(self):
        """4× downsample of a single bright pixel: GDAL-style anti-aliasing
        expands the cubic kernel from 4×4 to 16×16 so the peak is spread
        over many output pixels.  Without expansion, a strict 4×4 cubic
        kernel would deposit nearly all of 1000.0 in one output pixel.
        """
        arr = np.zeros((1, 16, 16), dtype=np.float32)
        arr[0, 8, 8] = 1000.0
        src_t = Affine(1, 0, 0, 0, -1, 16)
        dst_t = Affine(4, 0, 0, 0, -4, 16)
        out = resample(arr, src_t, dst_t, 4, 4, method="cubic")
        assert out.max() < 100, (
            f"anti-aliased cubic 4× downsample should spread delta peak; "
            f"got max={out.max()} (strict 4×4 would deposit ~1000)"
        )
