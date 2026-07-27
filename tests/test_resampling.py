"""Unit tests for the resample() function and its kernel/coord helpers."""

from typing import Any

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

from rastera.resampling import ResamplingMethod, resample


def _src_grid(n: int):
    """n×n source of sequential values, 10m pixels, origin (0, n*10)."""
    arr = np.arange(n * n, dtype=np.float32).reshape(1, n, n)
    return arr, Affine(10, 0, 0, 0, -10, n * 10)


# ── resample (nearest) ───────────────────────────────────────────────────


class TestResampleNearest:
    def test_identity(self):
        arr, t = _src_grid(4)
        out = resample(arr, t, t, 4, 4)
        np.testing.assert_array_equal(out, arr)

    def test_downsample(self):
        arr, src_t = _src_grid(4)
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
        arr, src_t = _src_grid(4)
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
        arr, src_t = _src_grid(4)
        # Destination extends beyond source: 4x4 at 10m but shifted right by 20m
        dst_t = Affine(10, 0, 20, 0, -10, 40)
        out = resample(arr, src_t, dst_t, 4, 4, nodata=-1)
        # First 2 cols map to src cols 2,3; last 2 cols are out of bounds
        assert out[0, 0, 0] == arr[0, 0, 2]
        assert out[0, 0, 3] == -1  # out of bounds

    def test_with_reprojection(self):
        # Source in UTM 32N, destination in WGS84
        src_arr = np.ones((1, 10, 10), dtype=np.float32)
        src_t = Affine(100, 0, 500000, 0, -100, 5000000)  # 100m pixels in UTM

        # Small bbox in WGS84 covering the source area
        dst_t = Affine(0.001, 0, 9.0, 0, -0.001, 45.1)
        transformer = Transformer.from_crs(4326, 32632, always_xy=True)
        out = resample(src_arr, src_t, dst_t, 10, 10, nodata=0, transformer=transformer)
        assert out.shape == (1, 10, 10)
        # Some pixels should have data (1.0), some may be nodata (0)
        assert np.any(out == 1.0) or np.any(out == 0)

    def test_coarse_grid_matches_brute_force(self):
        """Coarse-grid interpolation should match per-pixel pyproj within 0.125 px."""
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
        src_arr = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
        src_t = Affine(100, 0, 500000, 0, -100, 5000000)
        dst_t = Affine(0.001, 0, 9.0, 0, -0.001, 45.1)
        transformer = Transformer.from_crs(4326, 32632, always_xy=True)
        out = resample(src_arr, src_t, dst_t, 5, 5, nodata=-1, transformer=transformer)
        assert out.shape == (1, 5, 5)

    def test_single_pixel_with_transformer(self):
        """1x1 destination grid should not crash."""
        src_arr = np.ones((1, 10, 10), dtype=np.float32)
        src_t = Affine(100, 0, 500000, 0, -100, 5000000)
        dst_t = Affine(0.01, 0, 9.0, 0, -0.01, 45.1)
        transformer = Transformer.from_crs(4326, 32632, always_xy=True)
        out = resample(src_arr, src_t, dst_t, 1, 1, nodata=0, transformer=transformer)
        assert out.shape == (1, 1, 1)


# ── resample (bilinear) ──────────────────────────────────────────────────


class TestResampleBilinear:
    def test_identity(self):
        """Same grid, frac=0 at every pixel center → output equals source."""
        arr, t = _src_grid(4)
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
        arr, src_t = _src_grid(4)
        # Shift dst origin so two columns fall outside the source.
        dst_t = Affine(10, 0, 20, 0, -10, 40)
        out = resample(arr, src_t, dst_t, 4, 4, nodata=-1, method="bilinear")
        # First 2 cols map to src cols 2, 3; last 2 cols are OOB.
        np.testing.assert_array_equal(out[0, :, 2], -1)
        np.testing.assert_array_equal(out[0, :, 3], -1)

    def test_with_reprojection(self):
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
        out = resample(arr, src_t, dst_t, 1, 1, nodata=float("nan"), method="bilinear")
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
        out = resample(arr, src_t, dst_t, 1, 1, nodata=float("nan"), method="bilinear")
        assert np.isnan(out[0, 0, 0]), f"expected NaN output, got {out}"


# ── resample (cubic) ─────────────────────────────────────────────────────


class TestResampleCubic:
    def test_identity(self):
        """Cubic at the same grid returns the source array (frac=0 at centers)."""
        arr, t = _src_grid(8)
        out = resample(arr, t, t, 8, 8, method="cubic")
        np.testing.assert_allclose(out, arr, atol=1e-5)

    def test_smooth_downsample_vs_nearest(self):
        """On a gradient, cubic downsample differs from nearest snapping."""
        arr, src_t = _src_grid(8)  # values 0..63
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
        arr, src_t = _src_grid(8)
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


# ── separable (same-CRS) two-pass accumulator ────────────────────────────


def _bruteforce_kernel(
    src: np.ndarray[Any, Any],
    src_t: Affine,
    dst_t: Affine,
    dw: int,
    dh: int,
    method: ResamplingMethod,
    nodata: float | None,
) -> np.ndarray[Any, Any]:
    """Non-separable per-tap reference for the same-CRS kernel.

    A straightforward 2-D-loop transcription of the documented algorithm
    (the implementation pre-dating the separable rewrite): shared geometry
    and weight functions, full outer-product kernel, GDAL-style nodata
    renormalization with the center gate and cubic per-dimension gate.  Used
    to prove the separable two-pass produces identical output.
    """
    import math

    from rastera.resampling import _bilinear_weights, _cubic_weights

    nb, h, w = src.shape
    nan = nodata is not None and nodata != nodata
    c = ~src_t * dst_t
    col_f = float(c.a) * (np.arange(dw) + 0.5) + float(c.c)
    row_f = float(c.e) * (np.arange(dh) + 0.5) + float(c.f)
    ccol = np.floor(col_f).astype(np.intp)
    crow = np.floor(row_f).astype(np.intp)
    bcol = np.floor(col_f - 0.5).astype(np.intp)
    brow = np.floor(row_f - 0.5).astype(np.intp)
    fcol = (col_f - 0.5) - bcol
    frow = (row_f - 0.5) - brow
    xf = max(1.0, abs(float(c.a)))
    yf = max(1.0, abs(float(c.e)))
    rad = 1 if method == "bilinear" else 2
    xoff = tuple(range(1 - math.ceil(rad * xf), math.ceil(rad * xf) + 1))
    yoff = tuple(range(1 - math.ceil(rad * yf), math.ceil(rad * yf) + 1))
    wfn = _bilinear_weights if method == "bilinear" else _cubic_weights
    wx = wfn(fcol, xoff, xf)
    wy = wfn(frow, yoff, yf)

    acc = np.zeros((nb, dh, dw))
    wt = np.zeros((dh, dw))
    rvc = np.zeros((len(yoff), dh, dw))
    cvc = np.zeros((len(xoff), dh, dw))
    for i, dy in enumerate(yoff):
        sr = brow + dy
        safer = np.clip(sr, 0, h - 1)
        ibr = (sr >= 0) & (sr < h)
        for j, dx in enumerate(xoff):
            sc = bcol + dx
            safec = np.clip(sc, 0, w - 1)
            ibc = (sc >= 0) & (sc < w)
            sample = src[:, safer[:, None], safec[None, :]]
            wxy = wy[i][:, None] * wx[j][None, :]
            ib = ibr[:, None] & ibc[None, :]
            if nodata is not None:
                isnd = np.isnan(sample) if nan else (sample == nodata)
                if nan:
                    sample = np.where(isnd, 0.0, sample)
                valid = ~isnd.any(0) & ib
                contrib = wxy * valid
                acc += sample * contrib
                wt += contrib
                rvc[i] += valid
                cvc[j] += valid
            else:
                acc += sample * wxy

    if nodata is not None:
        out = np.zeros_like(acc)
        hw = wt > 0
        np.divide(acc, wt, out=out, where=hw)
        cs = src[:, np.clip(crow, 0, h - 1)[:, None], np.clip(ccol, 0, w - 1)[None, :]]
        ibc2 = ((crow >= 0) & (crow < h))[:, None] & ((ccol >= 0) & (ccol < w))[None, :]
        cisnd = np.isnan(cs).any(0) if nan else (cs == nodata).any(0)
        invalid = (cisnd | ~ibc2) | ~hw
        if method == "cubic":
            invalid |= ~((rvc >= 2).any(0) & (cvc >= 2).any(0))
        out[:, invalid] = float(nodata)
    else:
        out = acc
    if np.issubdtype(src.dtype, np.integer):
        info = np.iinfo(src.dtype)
        np.clip(out, info.min, info.max, out=out)
        np.round(out, out=out)
    return out.astype(src.dtype)


class TestSeparableEquivalence:
    """The same-CRS path is a separable two-pass rewrite of the non-separable
    2-D loop; it must produce identical output for integer dtypes (the float
    accumulators round-trip to the same int).  Sizes are chosen so dst_h
    spans multiple row-blocks, exercising the chunked accumulator.
    """

    @pytest.mark.parametrize("method", ["bilinear", "cubic"])
    @pytest.mark.parametrize("scale", [0.5, 2.0, 4.0])
    @pytest.mark.parametrize("nodata", [None, 0])
    def test_matches_bruteforce(
        self, method: ResamplingMethod, scale: float, nodata: int | None
    ):
        rng = np.random.default_rng(7)
        # src_h tall enough that several scales push dst_h past the 256-row
        # block size and trigger multi-block chunking.
        src = rng.integers(1, 5000, size=(3, 700, 48)).astype(np.uint16)
        if nodata is not None:
            src[:, rng.random((700, 48)) < 0.2] = nodata
        src_t = Affine(1, 0, 0, 0, -1, 700)
        dh = int(round(700 / scale))
        dw = int(round(48 / scale))
        dst_t = Affine(scale, 0, 0, 0, -scale, 700)
        out = resample(src, src_t, dst_t, dw, dh, nodata=nodata, method=method)
        ref = _bruteforce_kernel(src, src_t, dst_t, dw, dh, method, nodata)
        # Separable reorders the float summation, so integer output may differ
        # by ≤1 LSB at rounding boundaries; a nodata-mask flip would differ by
        # far more and still fail.
        np.testing.assert_allclose(out, ref, atol=1)
        if nodata is not None:
            np.testing.assert_array_equal(out == nodata, ref == nodata)

    @pytest.mark.parametrize("method", ["bilinear", "cubic"])
    def test_anisotropic_downsample_no_int8_overflow(self, method: ResamplingMethod):
        """One axis downsampled hard enough to exceed 127 cubic kernel taps:
        the per-dimension valid-sample count must not overflow (int32, not
        int8), or pixels with plenty of valid neighbours are wrongly gated to
        nodata.  Compared against the non-separable reference, which counts in
        float64 (overflow-free)."""
        rng = np.random.default_rng(11)
        src = rng.integers(1, 2000, size=(3, 600, 80)).astype(np.uint16)
        src[:, rng.random((600, 80)) < 0.18] = 0  # nodata holes
        src_t = Affine(1, 0, 0, 0, -1, 600)
        # 60× vertical downsample → 240 y-taps for cubic (> int8 max 127).
        dst_t = Affine(8, 0, 0, 0, -8, 600)
        out = resample(src, src_t, dst_t, 10, 75, nodata=0, method=method)
        ref = _bruteforce_kernel(src, src_t, dst_t, 10, 75, method, 0)
        np.testing.assert_array_equal(out == 0, ref == 0)  # identical nodata mask
        np.testing.assert_allclose(out, ref, atol=1)

    def test_block_size_invariance(self, monkeypatch: pytest.MonkeyPatch):
        """Chunking is purely an implementation detail: output must not depend
        on the row-block size (catches block-boundary indexing bugs)."""
        import rastera.resampling as r

        rng = np.random.default_rng(3)
        src = rng.integers(1, 5000, size=(2, 500, 64)).astype(np.uint16)
        src[:, rng.random((500, 64)) < 0.2] = 0
        src_t = Affine(1, 0, 0, 0, -1, 500)
        dst_t = Affine(2, 0, 0, 0, -2, 500)
        kw: dict[str, Any] = dict(nodata=0, method="cubic")
        monkeypatch.setattr(r, "_SEPARABLE_ROW_BLOCK", 1_000_000)
        whole = resample(src, src_t, dst_t, 32, 250, **kw)
        monkeypatch.setattr(r, "_SEPARABLE_ROW_BLOCK", 7)
        tiny = resample(src, src_t, dst_t, 32, 250, **kw)
        np.testing.assert_array_equal(whole, tiny)


class TestTwoPassReproject:
    """The two-pass cross-CRS strategy (downsample in source CRS, then
    reproject near-unit-scale) is an opt-in faster alternative to the single
    non-separable warp.  It is not bit-exact — two kernels apply — but must
    stay close on the interior, be a strict no-op where it does not apply
    (upsample / same-CRS / nearest / single_pass), and not introduce an edge
    fringe under nodata.
    """

    def _cross_setup(
        self,
        src_res: float,
        dst_res: float,
        *,
        H: int = 200,
        W: int = 200,
        dtype: np.typing.DTypeLike = np.float32,
        seed: int = 0,
    ) -> tuple[np.ndarray[Any, Any], Affine, Affine, int, int, Transformer]:
        """A smooth EPSG:3006 raster + a target UTM33N grid at ``dst_res``.

        Returns ``(arr, src_t, dst_t, dw, dh, transformer)`` ready for
        ``resample(..., transformer=transformer)``.  ``scale ≈ dst_res/src_res``.
        """
        yy, xx = np.mgrid[0:H, 0:W]
        arr = (100 + 50 * np.sin(xx / 15.0) + 40 * np.cos(yy / 12.0)).astype(dtype)
        arr = arr[None]  # (1, H, W); strictly positive so 0 means nodata only
        x0, y0 = 650000.0, 6580000.0
        src_t = Affine(src_res, 0, x0, 0, -src_res, y0)
        fwd = Transformer.from_crs(3006, 32633, always_xy=True)
        gx, gy = np.meshgrid([x0, x0 + W * src_res], [y0 - H * src_res, y0])
        ux, uy = fwd.transform(gx.ravel(), gy.ravel())
        pad = 4 * dst_res  # keep dst footprint inside the source extent
        dw = int((ux.max() - ux.min() - 2 * pad) // dst_res)
        dh = int((uy.max() - uy.min() - 2 * pad) // dst_res)
        dst_t = Affine(
            dst_res, 0, ux.min() + pad, 0, -dst_res, uy.min() + pad + dh * dst_res
        )
        T = Transformer.from_crs(32633, 3006, always_xy=True)
        return arr, src_t, dst_t, dw, dh, T

    @pytest.mark.parametrize("method", ["bilinear", "cubic"])
    def test_matches_single_pass_on_interior(self, method: ResamplingMethod):
        arr, st, dt, dw, dh, T = self._cross_setup(0.16, 0.5)
        kw: dict[str, Any] = dict(transformer=T, method=method)
        sp = resample(arr, st, dt, dw, dh, warp_strategy="single_pass", **kw)
        tp = resample(arr, st, dt, dw, dh, warp_strategy="auto", **kw)
        b = 3  # drop the edge band where the two methods legitimately differ
        d = np.abs(sp[:, b:-b, b:-b].astype(float) - tp[:, b:-b, b:-b].astype(float))
        assert float(np.sqrt((d**2).mean())) < 1.0
        assert float(d.max()) < 5.0

    def test_auto_dispatch_threshold(self):
        # "auto" takes the two-pass route only above its conservative downsample
        # scale (> 2.0).  Below it, "auto" is identical to single-pass.
        arr, st, dt, dw, dh, T = self._cross_setup(0.16, 0.28)  # scale ~1.75
        kw: dict[str, Any] = dict(transformer=T, method="cubic")
        sp = resample(arr, st, dt, dw, dh, warp_strategy="single_pass", **kw)
        auto = resample(arr, st, dt, dw, dh, warp_strategy="auto", **kw)
        np.testing.assert_array_equal(auto, sp)
        # Well above the cutoff, "auto" engages the two-pass branch and diverges.
        arr, st, dt, dw, dh, T = self._cross_setup(0.16, 0.5)  # scale ~3.1
        kw: dict[str, Any] = dict(transformer=T, method="cubic")
        sp = resample(arr, st, dt, dw, dh, warp_strategy="single_pass", **kw)
        auto = resample(arr, st, dt, dw, dh, warp_strategy="auto", **kw)
        assert not np.array_equal(auto, sp)

    def test_upsample_is_noop(self):
        # scale < 1: two-pass split has no benefit and must not engage.
        arr, st, dt, dw, dh, T = self._cross_setup(0.5, 0.16)
        kw: dict[str, Any] = dict(transformer=T, method="cubic")
        sp = resample(arr, st, dt, dw, dh, warp_strategy="single_pass", **kw)
        out = resample(arr, st, dt, dw, dh, warp_strategy="auto", **kw)
        np.testing.assert_array_equal(sp, out)

    def test_same_crs_is_noop(self):
        # transformer=None: the separable same-CRS path handles it; the
        # cross-CRS two-pass branch is never reached.
        arr = (np.arange(160 * 160, dtype=np.float32) % 97).reshape(1, 160, 160)
        st = Affine(0.16, 0, 0, 0, -0.16, 0)
        dt = Affine(0.5, 0, 0, 0, -0.5, 0)
        kw: dict[str, Any] = dict(transformer=None, method="cubic")
        sp = resample(arr, st, dt, 50, 50, warp_strategy="single_pass", **kw)
        out = resample(arr, st, dt, 50, 50, warp_strategy="auto", **kw)
        np.testing.assert_array_equal(sp, out)

    def test_nearest_ignores_strategy(self):
        arr, st, dt, dw, dh, T = self._cross_setup(0.16, 0.5)
        kw: dict[str, Any] = dict(transformer=T, method="nearest")
        sp = resample(arr, st, dt, dw, dh, warp_strategy="single_pass", **kw)
        tp = resample(arr, st, dt, dw, dh, warp_strategy="auto", **kw)
        np.testing.assert_array_equal(sp, tp)

    def test_integer_no_double_rounding(self):
        # uint16 two-pass must equal the float two-pass rounded once — i.e. the
        # intermediate is not re-quantized between passes.
        arr, st, dt, dw, dh, T = self._cross_setup(0.16, 0.5)
        arr_u = arr.astype(np.uint16)
        kw: dict[str, Any] = dict(transformer=T, method="cubic", warp_strategy="auto")
        tp_u = resample(arr_u, st, dt, dw, dh, **kw)
        tp_f = resample(arr_u.astype(np.float32), st, dt, dw, dh, **kw)
        b = 3
        np.testing.assert_allclose(
            tp_u[:, b:-b, b:-b], np.round(tp_f[:, b:-b, b:-b]), atol=1
        )

    @pytest.mark.parametrize("method", ["bilinear", "cubic"])
    def test_nodata_no_edge_fringe(self, method: ResamplingMethod):
        # The halo must keep Pass A's edge erosion off the region Pass B reads:
        # two-pass should not turn many single-pass-valid pixels into nodata.
        arr, st, dt, dw, dh, T = self._cross_setup(0.16, 0.5)
        kw: dict[str, Any] = dict(transformer=T, nodata=0, method=method)
        sp = resample(arr, st, dt, dw, dh, warp_strategy="single_pass", **kw)
        tp = resample(arr, st, dt, dw, dh, warp_strategy="auto", **kw)
        extra_nodata = (tp == 0) & (sp != 0)
        assert extra_nodata.mean() < 0.01

    def test_global_setter_and_validation(self):
        import rastera
        import rastera.config as config

        arr, st, dt, dw, dh, T = self._cross_setup(0.16, 0.5)
        kw: dict[str, Any] = dict(transformer=T, method="cubic")
        explicit = resample(arr, st, dt, dw, dh, warp_strategy="auto", **kw)
        prev = config._warp_strategy
        try:
            rastera.set_warp_strategy("auto")
            via_global = resample(arr, st, dt, dw, dh, **kw)  # no explicit arg
            np.testing.assert_array_equal(explicit, via_global)
        finally:
            rastera.set_warp_strategy(prev)
        with pytest.raises(ValueError):
            rastera.set_warp_strategy("nope")  # type: ignore[reportArgumentType]


# ── input validation ─────────────────────────────────────────────────────


class TestResampleValidation:
    """Guards against silently-wrong output for grids/dtypes/sentinels the
    kernels cannot represent. Each of these previously either produced
    plausible wrong pixels or disagreed between methods."""

    SRC = np.arange(64, dtype=np.uint8).reshape(1, 8, 8)
    SRC_T = Affine(1, 0, 0, 0, -1, 8)
    # Destination partly outside the source, so nodata fill is exercised.
    DST_T = Affine(1, 0, -4, 0, -1, 8)

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_rotated_dst_rejected(self, method: ResamplingMethod):
        """Only the a/c/e/f terms are read, so a rotated grid used to be
        resampled as if it were north-up."""
        rotated = self.SRC_T * Affine.rotation(30)
        with pytest.raises(NotImplementedError, match="north-up"):
            resample(self.SRC, self.SRC_T, rotated, 8, 8, method=method)

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_rotated_src_rejected(self, method: ResamplingMethod):
        rotated = self.SRC_T * Affine.rotation(30)
        with pytest.raises(NotImplementedError, match="north-up"):
            resample(self.SRC, rotated, self.DST_T, 8, 8, method=method)

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_nodata_outside_dtype_range_rejected(self, method: ResamplingMethod):
        """nearest raised OverflowError while bilinear/cubic clipped -9999 to 0,
        making nodata indistinguishable from a real 0-valued pixel."""
        with pytest.raises(ValueError, match="outside the range"):
            resample(
                self.SRC, self.SRC_T, self.DST_T, 8, 8, nodata=-9999, method=method
            )

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_nan_nodata_on_integer_rejected(self, method: ResamplingMethod):
        """np.round(NaN).astype(int16) is undefined; nearest raised but
        bilinear/cubic emitted a RuntimeWarning and cast garbage."""
        src = self.SRC.astype(np.int16)
        with pytest.raises(ValueError, match="NaN cannot be represented"):
            resample(
                src, self.SRC_T, self.DST_T, 8, 8, nodata=float("nan"), method=method
            )

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_fractional_nodata_on_integer_rejected(self, method: ResamplingMethod):
        """No integer pixel can equal 1.5, and the methods disagreed on what to
        write for it: nearest truncated to 1 on the cast, the kernels rounded
        to 2. Either way the caller's mask never matched."""
        with pytest.raises(ValueError, match="not an integer"):
            resample(self.SRC, self.SRC_T, self.DST_T, 8, 8, nodata=1.5, method=method)

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_whole_float_nodata_on_integer_allowed(self, method: ResamplingMethod):
        """3.0 is representable, so it stays legal — rasterio hands nodata over
        as a float and every real sentinel arrives this way."""
        out = resample(
            self.SRC, self.SRC_T, self.DST_T, 8, 8, nodata=3.0, method=method
        )
        assert (out == 3).any()

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_nodata_past_float64_mantissa_rejected(self, method: ResamplingMethod):
        """The kernels accumulate in float64 and write the sentinel back through
        float(nodata), so 2**53+1 came out as 2**53 — bilinear marked nodata
        with a value nearest never produces and nothing downstream matches."""
        src = self.SRC.astype(np.int64)
        with pytest.raises(ValueError, match="exactly representable"):
            resample(src, self.SRC_T, self.DST_T, 8, 8, nodata=2**53 + 1, method=method)

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_large_but_exact_nodata_allowed(self, method: ResamplingMethod):
        src = self.SRC.astype(np.int64)
        out = resample(
            src, self.SRC_T, self.DST_T, 8, 8, nodata=-(2**40), method=method
        )
        assert (out == -(2**40)).any()

    def test_fractional_nodata_on_float_allowed(self):
        src = self.SRC.astype(np.float32)
        out = resample(src, self.SRC_T, self.DST_T, 8, 8, nodata=1.5, method="bilinear")
        assert (out == np.float32(1.5)).any()

    def test_nan_nodata_on_float_allowed(self):
        src = self.SRC.astype(np.float32)
        out = resample(
            src, self.SRC_T, self.DST_T, 8, 8, nodata=float("nan"), method="bilinear"
        )
        assert np.isnan(out[0, 0, 0])

    @pytest.mark.parametrize("method", ["bilinear", "cubic"])
    def test_complex_rejected(self, method: ResamplingMethod):
        src = self.SRC.astype(np.complex128)
        with pytest.raises(NotImplementedError, match="complex"):
            resample(src, self.SRC_T, self.DST_T, 8, 8, method=method)

    def test_complex_allowed_for_nearest(self):
        """Only the kernels do arithmetic on samples; nearest is a pure gather,
        so it copies complex values through unharmed."""
        src = self.SRC.astype(np.complex128) * (1 + 2j)
        out = resample(src, self.SRC_T, self.SRC_T, 8, 8, method="nearest")
        np.testing.assert_array_equal(out, src)

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    @pytest.mark.parametrize("transformer", [None, "cross"])
    def test_zero_size_output(self, method: ResamplingMethod, transformer: str | None):
        """The cross-CRS path raised IndexError on an empty coarse grid while
        the same-CRS path returned cleanly."""
        t = None
        if transformer == "cross":
            t = Transformer.from_crs(32632, 32633, always_xy=True)
        out = resample(
            self.SRC, self.SRC_T, self.DST_T, 0, 0, transformer=t, method=method
        )
        assert out.shape == (1, 0, 0)
        assert out.dtype == self.SRC.dtype

    def test_negative_size_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            resample(self.SRC, self.SRC_T, self.DST_T, -1, 8)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown resampling method"):
            resample(self.SRC, self.SRC_T, self.DST_T, 8, 8, method="lanczos")  # type: ignore[arg-type]


class TestBoolMask:
    """bool is not an np.integer, so the clip+round was skipped and any
    non-zero kernel accumulation became True — dilating the mask."""

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_downsample_does_not_dilate(self, method: ResamplingMethod):
        mask = np.zeros((1, 4, 4), dtype=bool)
        mask[0, 0, 0] = True
        out = resample(
            mask,
            Affine(1, 0, 0, 0, -1, 4),
            Affine(2, 0, 0, 0, -2, 4),
            2,
            2,
            method=method,
        )
        assert out.dtype == np.bool_
        # A single True out of the 2x2 footprint is a 0.25 weight -> False.
        assert not out.any()

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_majority_true_survives(self, method: ResamplingMethod):
        mask = np.ones((1, 4, 4), dtype=bool)
        out = resample(
            mask,
            Affine(1, 0, 0, 0, -1, 4),
            Affine(2, 0, 0, 0, -2, 4),
            2,
            2,
            method=method,
        )
        assert out.all()

    def test_two_pass_cross_crs_bool(self):
        """_two_pass_work_dtype called np.iinfo(np.bool_), which raises."""

        mask = np.ones((1, 64, 64), dtype=bool)
        t = Transformer.from_crs(32632, 32632, always_xy=True)
        out = resample(
            mask,
            Affine(1, 0, 0, 0, -1, 64),
            Affine(4, 0, 0, 0, -4, 64),
            16,
            16,
            transformer=t,
            method="cubic",
            warp_strategy="auto",
        )
        assert out.dtype == np.bool_
