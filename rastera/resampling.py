"""Pixel resampling for resolution changes and reprojection.

Exposes a single public entry point :func:`resample`, which dispatches on a
``method`` argument to one of three implementations:

- ``"nearest"`` — nearest-neighbor, memory-tight 1D/2D index path.
- ``"bilinear"`` — separable linear kernel; 2×2 at upsampling/identity,
  widened proportionally when downsampling to act as an anti-aliasing
  low-pass filter (matches GDAL's warp behaviour).
- ``"cubic"`` — Keys cubic convolution (a = -0.5); 4×4 at
  upsampling/identity, similarly widened when downsampling.

Bilinear and cubic use GDAL-style nodata handling: kernel weights are
renormalized over valid samples, with a center-pixel nodata gate and (for
cubic) a per-dimension ≥2-valid safety gate to avoid overshoot from negative
cubic weights at data/nodata boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from affine import Affine
from pyproj import Transformer

ResamplingMethod = Literal["nearest", "bilinear", "cubic"]


def resample(
    src_array: np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    nodata: int | float | None = None,
    transformer: Transformer | None = None,
    method: ResamplingMethod = "nearest",
) -> np.ndarray:
    """Resample src_array to a target grid.

    Three methods are supported, matching GDAL / rasterio conventions:

    - ``"nearest"`` (default): nearest-neighbor. Fast, exact, no smoothing.
      Matches ``Resampling.nearest`` in rasterio.
    - ``"bilinear"``: separable linear kernel. 2×2 at upsampling and
      identity; expanded to ``2·⌈scale⌉ × 2·⌈scale⌉`` when downsampling
      so the kernel acts as a low-pass anti-aliasing filter (where
      ``scale = max(1, dst_res / src_res)``).  Matches
      ``Resampling.bilinear`` / ``gdalwarp -r bilinear``. No overshoot.
    - ``"cubic"``: Keys cubic convolution (a = -0.5). 4×4 at
      upsampling/identity; expanded to ``4·⌈scale⌉ × 4·⌈scale⌉`` when
      downsampling.  Matches ``Resampling.cubic`` / ``gdalwarp -r cubic``.
      Can overshoot the source value range (for integer dtypes, output
      is clipped to the dtype range and rounded).

    For ``"bilinear"`` and ``"cubic"`` with ``nodata`` set, nodata is
    handled GDAL-style: kernel weights are renormalized over valid
    samples (invalid samples are dropped from the kernel). A target
    pixel is set to ``nodata`` when the source pixel under the target
    center is nodata, when every kernel sample is nodata, or — for
    cubic only — when fewer than 2 valid samples exist along each axis
    of the kernel window (negative cubic weights cause severe overshoot
    when valid/invalid samples alternate).

    ``nodata`` may be a finite sentinel (e.g. -9999, 0) or NaN; NaN is
    detected via ``np.isnan`` so the center gate and renormalization
    behave identically across sentinel types.

    Args:
        src_array: (bands, h, w) source data.
        src_transform: Affine pixel→world for source.
        dst_transform: Affine pixel→world for destination.
        dst_width: Output width in pixels.
        dst_height: Output height in pixels.
        nodata: Fill value for out-of-bounds pixels. Also drives kernel
            renormalization for bilinear/cubic when set.
        transformer: pyproj Transformer (target CRS → source CRS).
            ``None`` if same CRS.
        method: One of ``"nearest"``, ``"bilinear"``, ``"cubic"``.
    """
    if method == "nearest":
        return _resample_nearest(
            src_array,
            src_transform,
            dst_transform,
            dst_width,
            dst_height,
            nodata,
            transformer,
        )
    if method in ("bilinear", "cubic"):
        return _resample_kernel(
            src_array,
            src_transform,
            dst_transform,
            dst_width,
            dst_height,
            nodata,
            transformer,
            method,
        )
    raise ValueError(
        f"Unknown resampling method {method!r}; "
        "expected 'nearest', 'bilinear', or 'cubic'."
    )


def _resample_nearest(
    src_array: np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    nodata: int | float | None,
    transformer: Transformer | None,
) -> np.ndarray:
    """Nearest-neighbor resampling.

    Memory-tight: same-CRS uses 1D index arrays, cross-CRS uses the
    coarse-grid transform with in-place ops.
    """
    h, w = src_array.shape[1], src_array.shape[2]

    if transformer is None:
        # Same CRS: compose affines and use 1D index arrays (no meshgrid).
        combined = cast(Affine, ~src_transform * dst_transform)
        src_col_1d = np.floor(
            float(combined.a) * (np.arange(dst_width, dtype=np.float64) + 0.5)
            + float(combined.c)
        ).astype(np.intp)
        src_row_1d = np.floor(
            float(combined.e) * (np.arange(dst_height, dtype=np.float64) + 0.5)
            + float(combined.f)
        ).astype(np.intp)

        valid_col = (src_col_1d >= 0) & (src_col_1d < w)
        valid_row = (src_row_1d >= 0) & (src_row_1d < h)

        col_safe = np.clip(src_col_1d, 0, w - 1)
        row_safe = np.clip(src_row_1d, 0, h - 1)
        out = src_array[:, row_safe[:, np.newaxis], col_safe[np.newaxis, :]]

        if nodata is not None and not (np.all(valid_col) and np.all(valid_row)):
            fill = np.array(nodata, dtype=src_array.dtype)
            invalid = ~(valid_row[:, np.newaxis] & valid_col[np.newaxis, :])
            out[:, invalid] = fill
    else:
        # Coarse-grid + interpolation: transform sparse grid through pyproj,
        # bilinearly interpolate to full resolution.  In-place ops and eager
        # deletion keep peak memory to 2 full-size index arrays instead of 6.
        src_col_f, src_row_f = _coarse_grid_transform(
            dst_width,
            dst_height,
            dst_transform,
            src_transform,
            transformer,
        )
        np.floor(src_col_f, out=src_col_f)
        np.floor(src_row_f, out=src_row_f)
        src_col = src_col_f.astype(np.intp)
        del src_col_f
        src_row = src_row_f.astype(np.intp)
        del src_row_f

        valid = (src_col >= 0) & (src_col < w) & (src_row >= 0) & (src_row < h)
        np.clip(src_col, 0, w - 1, out=src_col)
        np.clip(src_row, 0, h - 1, out=src_row)

        out = src_array[:, src_row, src_col]
        del src_col, src_row

        if nodata is not None and not np.all(valid):
            out[:, ~valid] = np.array(nodata, dtype=src_array.dtype)

    return out


def _resample_kernel(
    src_array: np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    nodata: int | float | None,
    transformer: Transformer | None,
    method: Literal["bilinear", "cubic"],
) -> np.ndarray:
    """Bilinear or cubic resampling with GDAL-style nodata renormalization
    and anti-aliasing kernel expansion for downsampling.

    See :func:`resample` for the user-facing semantics summary.
    Implementation notes:

    - The neighbourhood is computed via a small nested Python loop with
      vectorised numpy inside.  Kernel half-width per axis is
      ``base_radius · max(1, |dst_res / src_res|)`` (rounded up), where
      ``base_radius`` is 1 for bilinear and 2 for cubic.  Upsampling
      and identity reads use the default radii; downsampling expands.
    - Weights are separable and computed once outside the loop, then
      pre-normalized along the tap axis so the kernel sums to 1.
    - Out-of-bounds taps (kernel reach beyond the source extent for
      output pixels near an edge) are treated as nodata for
      renormalization when ``nodata`` is set, and clamped (edge
      replicated) otherwise.
    - Accumulation is in float64; integer output dtypes are
      clip+round-cast at the end (cubic can overshoot the source range).
    """
    n_bands, h, w = src_array.shape

    # NaN-sentinel nodata needs `np.isnan` for detection (NaN != NaN means
    # `==` and `!=` both miss it) and zeroing-out before multiply (NaN * 0
    # propagates NaN into the accumulator).  This mirrors the NaN path in
    # `merge.py`'s paste loop.
    nodata_is_nan = nodata is not None and nodata != nodata

    # --- Compute float source coordinates for every destination pixel.
    # Same-CRS: keep coords 1D ``(W,)`` / ``(H,)`` — base/frac/center and
    # the separable kernel weights stay 1D, with the kernel loop forming
    # 2D arrays only on demand.  For 4K×4K cubic this avoids materialising
    # ``(4, H, W)`` weight tensors (~4 GB).  Cross-CRS reprojection is
    # not separable, so the coarse-grid path returns full 2D coords.
    if transformer is None:
        combined = cast(Affine, ~src_transform * dst_transform)
        src_col_f = float(combined.a) * (
            np.arange(dst_width, dtype=np.float64) + 0.5
        ) + float(combined.c)
        src_row_f = float(combined.e) * (
            np.arange(dst_height, dtype=np.float64) + 0.5
        ) + float(combined.f)
        coords_2d = False
        # Local pixel scale = src pixels per dst pixel (= dst_res / src_res
        # along the axis-aligned same-CRS case).
        x_scale_local = abs(float(combined.a))
        y_scale_local = abs(float(combined.e))
    else:
        src_col_f, src_row_f = _coarse_grid_transform(
            dst_width, dst_height, dst_transform, src_transform, transformer
        )
        coords_2d = True
        # Approximate local pixel scale from the median absolute gradient
        # of src coords along each dst axis.  Median is robust against
        # outliers near the source extent boundary.  A global (not
        # per-pixel) scale matches GDAL's warp behaviour.
        if dst_width >= 2:
            x_scale_local = float(np.median(np.abs(np.diff(src_col_f, axis=1))))
        else:
            x_scale_local = 1.0
        if dst_height >= 2:
            y_scale_local = float(np.median(np.abs(np.diff(src_row_f, axis=0))))
        else:
            y_scale_local = 1.0

    # Source pixel containing the dst center (pixel-corner convention).
    # Used for the OOB gate and the GDAL-style center-pixel nodata gate.
    center_col = np.floor(src_col_f).astype(np.intp)
    center_row = np.floor(src_row_f).astype(np.intp)

    # Kernel base/frac: shift by -0.5 so the kernel interpolates between
    # source pixel CENTERS (at integer + 0.5 in src pixel-corner space).
    # Without this shift, a dst pixel landing exactly on a src pixel
    # center would be a 50/50 blend with the neighbour instead of the
    # exact src value.
    shifted_col_f = src_col_f - 0.5
    shifted_row_f = src_row_f - 0.5
    base_col = np.floor(shifted_col_f).astype(np.intp)
    base_row = np.floor(shifted_row_f).astype(np.intp)
    frac_col = shifted_col_f - base_col
    frac_row = shifted_row_f - base_row

    # --- Anti-aliasing: GDAL expands the kernel radius when downsampling
    # (scale > 1) so that bilinear/cubic act as proper low-pass filters
    # over the wider source footprint covered by each dst pixel.  When
    # upsampling (scale < 1) the kernel keeps its default radius.
    x_filter = max(1.0, x_scale_local)
    y_filter = max(1.0, y_scale_local)
    base_radius = 1 if method == "bilinear" else 2
    n_x_radius = math.ceil(base_radius * x_filter)
    n_y_radius = math.ceil(base_radius * y_filter)
    x_offsets = tuple(range(1 - n_x_radius, n_x_radius + 1))
    y_offsets = tuple(range(1 - n_y_radius, n_y_radius + 1))

    weights_fn = _bilinear_weights if method == "bilinear" else _cubic_weights
    wx = weights_fn(frac_col, x_offsets, x_filter)
    wy = weights_fn(frac_row, y_offsets, y_filter)

    # --- Accumulators.
    acc_val = np.zeros((n_bands, dst_height, dst_width), dtype=np.float64)
    acc_wt: np.ndarray | None = None
    row_valid_counts: np.ndarray | None = None
    col_valid_counts: np.ndarray | None = None
    if nodata is not None:
        acc_wt = np.zeros((dst_height, dst_width), dtype=np.float64)
        if method == "cubic":
            row_valid_counts = np.zeros(
                (len(y_offsets), dst_height, dst_width), dtype=np.int8
            )
            col_valid_counts = np.zeros(
                (len(x_offsets), dst_height, dst_width), dtype=np.int8
            )

    # --- Accumulate kernel contributions.
    for i, dy in enumerate(y_offsets):
        src_row_idx = base_row + dy
        safe_row = np.clip(src_row_idx, 0, h - 1)
        in_bounds_row = (src_row_idx >= 0) & (src_row_idx < h)
        wy_i = wy[i]
        for j, dx in enumerate(x_offsets):
            src_col_idx = base_col + dx
            safe_col = np.clip(src_col_idx, 0, w - 1)
            in_bounds_col = (src_col_idx >= 0) & (src_col_idx < w)
            wx_j = wx[j]

            if coords_2d:
                sample = src_array[:, safe_row, safe_col]  # (B, H, W)
                w_xy = wy_i * wx_j  # (H, W)
                in_bounds = in_bounds_row & in_bounds_col  # (H, W)
            else:
                sample = src_array[:, safe_row[:, None], safe_col[None, :]]
                w_xy = wy_i[:, None] * wx_j[None, :]  # outer product → (H, W)
                in_bounds = in_bounds_row[:, None] & in_bounds_col[None, :]

            if nodata is not None:
                # Pixel is valid only if all bands are non-nodata AND the
                # tap is in-bounds.  Per-band-uniform validity matches the
                # single dataset-level nodata convention used throughout
                # rastera.  NaN-sentinel: use `np.isnan` (NaN != NaN means
                # `!=` would mark every NaN as valid) and zero-out NaN
                # samples before the multiply.
                if nodata_is_nan:
                    is_nodata = np.isnan(sample)
                    sample = np.where(is_nodata, 0.0, sample)
                else:
                    is_nodata = sample == nodata
                valid = ~is_nodata.any(axis=0) & in_bounds  # (H, W)
                contrib = w_xy * valid  # (H, W), bool→float promotion
                acc_val += sample * contrib  # broadcast (B,H,W) * (H,W)
                assert acc_wt is not None
                acc_wt += contrib
                if method == "cubic":
                    assert row_valid_counts is not None
                    assert col_valid_counts is not None
                    row_valid_counts[i] += valid
                    col_valid_counts[j] += valid
            else:
                # No nodata: kernel uses clamped (edge-replicated) samples
                # without renormalization.
                acc_val += sample * w_xy

    # --- Finalize: renormalize, apply gates, cast to source dtype.
    if nodata is not None:
        assert acc_wt is not None
        out_f = np.zeros_like(acc_val)
        has_weight = acc_wt > 0
        np.divide(acc_val, acc_wt, out=out_f, where=has_weight)

        # Center gate: source pixel under the dst center is nodata or OOB.
        center_safe_row = np.clip(center_row, 0, h - 1)
        center_safe_col = np.clip(center_col, 0, w - 1)
        if coords_2d:
            center_sample = src_array[:, center_safe_row, center_safe_col]
            in_bounds_center = (
                (center_row >= 0)
                & (center_row < h)
                & (center_col >= 0)
                & (center_col < w)
            )
        else:
            center_sample = src_array[
                :, center_safe_row[:, None], center_safe_col[None, :]
            ]
            in_bounds_center = (
                ((center_row >= 0) & (center_row < h))[:, None]
                & ((center_col >= 0) & (center_col < w))[None, :]
            )
        if nodata_is_nan:
            center_is_nodata = np.isnan(center_sample).any(axis=0)
        else:
            center_is_nodata = (center_sample == nodata).any(axis=0)
        center_bad = center_is_nodata | ~in_bounds_center

        invalid = center_bad | ~has_weight

        if method == "cubic":
            assert row_valid_counts is not None
            assert col_valid_counts is not None
            per_dim_ok = (row_valid_counts >= 2).any(axis=0) & (
                col_valid_counts >= 2
            ).any(axis=0)
            invalid |= ~per_dim_ok

        if invalid.any():
            out_f[:, invalid] = float(nodata)
    else:
        out_f = acc_val

    src_dtype = src_array.dtype
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        np.clip(out_f, info.min, info.max, out=out_f)
        np.round(out_f, out=out_f)
    return out_f.astype(src_dtype)


# ---- Private helpers ----

_WARP_GRID_STEP = 16


def _coarse_grid_transform(
    dst_width: int,
    dst_height: int,
    dst_transform: Affine,
    src_transform: Affine,
    transformer: Transformer,
    step: int = _WARP_GRID_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform dst pixels to src pixel coords via coarse-grid interpolation.

    Instead of transforming every destination pixel through pyproj, transforms a
    coarse grid (every ``step`` pixels) and bilinearly interpolates the rest.

    Returns ``(src_col_f, src_row_f)`` as float arrays of shape
    ``(dst_height, dst_width)``.
    """
    # Build coarse grid nodes, always including the last pixel.
    coarse_cols = np.arange(0, dst_width, step, dtype=np.float64)
    if coarse_cols[-1] < dst_width - 1:
        coarse_cols = np.append(coarse_cols, dst_width - 1)
    coarse_rows = np.arange(0, dst_height, step, dtype=np.float64)
    if coarse_rows[-1] < dst_height - 1:
        coarse_rows = np.append(coarse_rows, dst_height - 1)

    # Transform coarse grid: dst pixel centers → world → source CRS → source pixels
    cc, cr = np.meshgrid(coarse_cols + 0.5, coarse_rows + 0.5)
    cwx = float(dst_transform.a) * cc + float(dst_transform.c)
    cwy = float(dst_transform.e) * cr + float(dst_transform.f)
    cwx, cwy = transformer.transform(cwx, cwy)

    src_inv = ~src_transform
    coarse_src_col = float(src_inv.a) * cwx + float(src_inv.c)
    coarse_src_row = float(src_inv.e) * cwy + float(src_inv.f)

    n_coarse_rows = len(coarse_rows)
    coarse_col_centers = coarse_cols + 0.5
    coarse_row_centers = coarse_rows + 0.5
    full_col_centers = np.arange(dst_width, dtype=np.float64) + 0.5

    # Pass 1: interpolate along columns for each coarse row.
    temp_col = np.empty((n_coarse_rows, dst_width), dtype=np.float64)
    temp_row = np.empty((n_coarse_rows, dst_width), dtype=np.float64)
    for i in range(n_coarse_rows):
        temp_col[i] = np.interp(full_col_centers, coarse_col_centers, coarse_src_col[i])
        temp_row[i] = np.interp(full_col_centers, coarse_col_centers, coarse_src_row[i])

    # Pass 2: vectorised interpolation along rows.
    full_row_centers = np.arange(dst_height, dtype=np.float64) + 0.5
    row_idx = np.interp(
        full_row_centers, coarse_row_centers, np.arange(n_coarse_rows, dtype=np.float64)
    )
    row_lo = np.clip(np.floor(row_idx).astype(int), 0, n_coarse_rows - 2)
    row_frac = (row_idx - row_lo)[:, np.newaxis]  # (dst_height, 1)

    src_col_f = temp_col[row_lo] + row_frac * (temp_col[row_lo + 1] - temp_col[row_lo])
    src_row_f = temp_row[row_lo] + row_frac * (temp_row[row_lo + 1] - temp_row[row_lo])

    return src_col_f, src_row_f


def _bilinear_weights(
    frac: np.ndarray, offsets: Sequence[int], scale: float
) -> np.ndarray:
    """Bilinear (tent) weights, GDAL-style anti-aliased.

    For each tap at offset ``k``, the distance from the sample point is
    ``k - frac``.  Weight = ``max(0, 1 - |k - frac| / scale)``.  When
    ``scale = 1`` and ``offsets = (0, 1)`` this reduces to the standard
    2-tap tent ``(1 - frac, frac)``; with ``scale > 1`` it widens to act
    as an anti-aliasing low-pass filter for downsampling.

    Returns shape ``(len(offsets), *frac.shape)``, normalized so weights
    sum to 1 along the first axis (handles the kernel-truncation case
    where the support extends beyond the integer offsets).
    """
    weights = np.stack(
        [np.maximum(0.0, 1.0 - np.abs(k - frac) / scale) for k in offsets]
    )
    return weights / weights.sum(axis=0, keepdims=True)


def _cubic_weights(
    frac: np.ndarray, offsets: Sequence[int], scale: float
) -> np.ndarray:
    """Keys cubic (a = -0.5) weights, GDAL-style anti-aliased.

    Like :func:`_bilinear_weights` but evaluated against the Keys cubic
    function (matching GDAL's ``GWKCubic``):

    - ``|d| < 1``: ``1.5|d|³ - 2.5|d|² + 1``
    - ``1 ≤ |d| < 2``: ``-0.5|d|³ + 2.5|d|² - 4|d| + 2``
    - ``|d| ≥ 2``: 0

    with ``d = (k - frac) / scale``.  At ``scale = 1`` and standard
    4-tap offsets ``{-1, 0, 1, 2}`` this is the partition-of-unity Keys
    kernel.  Normalization handles the rare case where summed weights
    drift from 1 due to scaling.
    """
    weights = []
    for k in offsets:
        d = np.abs(k - frac) / scale
        d2 = d * d
        d3 = d2 * d
        w_inner = 1.5 * d3 - 2.5 * d2 + 1.0
        w_outer = -0.5 * d3 + 2.5 * d2 - 4.0 * d + 2.0
        weights.append(np.where(d < 1.0, w_inner, np.where(d < 2.0, w_outer, 0.0)))
    out = np.stack(weights)
    return out / out.sum(axis=0, keepdims=True)
