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

from . import config
from .config import WarpStrategy

ResamplingMethod = Literal["nearest", "bilinear", "cubic"]

# Local downsample scale above which the ``"auto"`` strategy takes the two-pass
# cross-CRS route.  Set conservatively: benchmarking cross-CRS warps across
# image sizes (512²–4096²) and both kernels, scale 2.0 is the lowest threshold
# that is a speed win everywhere; below it two-pass is erratic and often slower
# (its fixed intermediate-allocation + near-unit reproject cost is not repaid —
# worst for cheap-kernel bilinear at large sizes, where it can run ~1.5x
# slower).  Below the threshold (and at scale <= 1 — upsampling — where the
# two-pass split has no benefit) the single-pass warp is used.
_AUTO_SCALE_THRESHOLD = 2.0


def resample(
    src_array: np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    nodata: int | float | None = None,
    transformer: Transformer | None = None,
    method: ResamplingMethod = "nearest",
    *,
    warp_strategy: WarpStrategy | None = None,
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
        warp_strategy: How a cross-CRS bilinear/cubic warp is carried out.
            ``None`` (default) reads the process-wide setting from
            :func:`rastera.set_warp_strategy`; pass an explicit value to
            override it for this call (useful in tests). See that function for
            the ``"auto"`` / ``"single_pass"`` semantics. No effect on nearest
            (any CRS/scale), same-CRS, or upsampling.
    """
    if method not in ("nearest", "bilinear", "cubic"):
        raise ValueError(
            f"Unknown resampling method {method!r}; "
            "expected 'nearest', 'bilinear', or 'cubic'."
        )
    if warp_strategy is None:
        warp_strategy = config._warp_strategy

    _validate_grids(src_transform, dst_transform, dst_width, dst_height)
    _validate_dtype_nodata(src_array.dtype, nodata)
    if dst_width == 0 or dst_height == 0:
        return np.empty(
            (src_array.shape[0], dst_height, dst_width), dtype=src_array.dtype
        )

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
    return _resample_kernel(
        src_array,
        src_transform,
        dst_transform,
        dst_width,
        dst_height,
        nodata,
        transformer,
        method,
        warp_strategy,
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
    warp_strategy: WarpStrategy = "single_pass",
) -> np.ndarray:
    """Bilinear or cubic resampling with GDAL-style nodata renormalization
    and anti-aliasing kernel expansion for downsampling.

    See :func:`resample` for the user-facing semantics summary.
    Implementation notes:

    - Same-CRS reads use a separable two-pass accumulation
      (:func:`_accumulate_separable`): ``O(taps_x + taps_y)`` instead of
      ``O(taps_x · taps_y)``.  Cross-CRS warps are not separable (the
      source sampling grid is not axis-aligned with the output) so they
      keep the non-separable 2-D loop.  The two paths are numerically
      equivalent: the separable reorder changes float summation only at
      the ULP level, so integer output may differ by at most 1 LSB at
      rounding boundaries.
    - Kernel half-width per axis is
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
    # Only the kernels do arithmetic on samples; nearest is a pure gather and
    # copies complex values through unharmed, so the rejection lives here
    # rather than in `resample`.
    if np.issubdtype(src_array.dtype, np.complexfloating):
        raise NotImplementedError(
            f"{method} resampling does not support complex dtype {src_array.dtype}"
        )
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

        # Cross-CRS downsample: optionally split into a same-CRS downsample
        # (fast separable path) + a near-unit-scale reproject.  Gated on the
        # local scale we just computed, so no extra coarse-grid work.
        threshold = _two_pass_threshold(warp_strategy)
        if threshold is not None and max(x_scale_local, y_scale_local) > threshold:
            return _resample_two_pass(
                src_array,
                src_transform,
                dst_transform,
                dst_width,
                dst_height,
                nodata,
                transformer,
                method,
                x_scale_local,
                y_scale_local,
            )

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

    # --- Accumulate kernel contributions.
    per_dim_ok: np.ndarray | None = None
    if coords_2d:
        # Non-separable 2-D loop for the cross-CRS warp: the source sampling
        # grid is not axis-aligned with the output, so weights and indices
        # are full 2-D and cannot be reused across rows/columns.
        acc_val = np.zeros((n_bands, dst_height, dst_width), dtype=np.float64)
        acc_wt: np.ndarray | None = None
        row_valid_counts: np.ndarray | None = None
        col_valid_counts: np.ndarray | None = None
        if nodata is not None:
            acc_wt = np.zeros((dst_height, dst_width), dtype=np.float64)
            if method == "cubic":
                # int32 (not int8): a single axis can have >127 kernel taps
                # under heavy anisotropic downsampling, which would overflow
                # int8 and spuriously fail the >=2 gate.
                row_valid_counts = np.zeros(
                    (len(y_offsets), dst_height, dst_width), dtype=np.int32
                )
                col_valid_counts = np.zeros(
                    (len(x_offsets), dst_height, dst_width), dtype=np.int32
                )

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

                sample = src_array[:, safe_row, safe_col]  # (B, H, W)
                w_xy = wy_i * wx_j  # (H, W)
                in_bounds = in_bounds_row & in_bounds_col  # (H, W)

                if nodata is not None:
                    # Pixel is valid only if all bands are non-nodata AND the
                    # tap is in-bounds.  NaN-sentinel: use `np.isnan` and
                    # zero-out NaN samples before the multiply.
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
                    # No nodata: clamped (edge-replicated) samples, no renorm.
                    acc_val += sample * w_xy

        if nodata is not None and method == "cubic":
            assert row_valid_counts is not None
            assert col_valid_counts is not None
            per_dim_ok = np.asarray(
                (row_valid_counts >= 2).any(axis=0)
                & (col_valid_counts >= 2).any(axis=0)
            )
    else:
        # Same-CRS: separable two-pass, O(taps_x + taps_y).
        acc_val, acc_wt, per_dim_ok = _accumulate_separable(
            src_array,
            base_col,
            base_row,
            wx,
            wy,
            x_offsets,
            y_offsets,
            nodata,
            nodata_is_nan,
            method,
        )

    return _finalize_kernel(
        acc_val,
        acc_wt,
        per_dim_ok,
        src_array,
        center_row,
        center_col,
        coords_2d,
        nodata,
        nodata_is_nan,
    )


# ---- Private helpers ----


def _validate_grids(
    src_transform: Affine, dst_transform: Affine, dst_width: int, dst_height: int
) -> None:
    """Reject grids this module cannot represent.

    Every path below reads only the ``a, c, e, f`` terms of the affines, so a
    rotated or sheared grid would be resampled as if it were north-up — wrong
    pixels, no error.  ``merge`` rejects rotation for the same reason.
    """
    for name, t in (("src_transform", src_transform), ("dst_transform", dst_transform)):
        b, d = float(t.b), float(t.d)
        if not math.isclose(b, 0.0) or not math.isclose(d, 0.0):
            raise NotImplementedError(
                f"resample requires a north-up (non-rotated) grid; {name} has "
                f"b={b!r}, d={d!r}"
            )
    if dst_width < 0 or dst_height < 0:
        raise ValueError(
            f"dst_width/dst_height must be >= 0, got {dst_width}x{dst_height}"
        )


def _validate_dtype_nodata(dtype: np.dtype, nodata: int | float | None) -> None:
    """Reject nodata sentinels the dtype cannot carry.

    Such a sentinel used to behave differently per method within one call:
    ``nearest`` raised ``OverflowError`` while bilinear/cubic clipped it into
    the valid range, making nodata indistinguishable from real data.
    """
    if nodata is None or dtype.kind not in ("i", "u", "b"):
        return
    if math.isnan(nodata):
        raise ValueError(
            f"nodata=NaN cannot be represented in {dtype}; pass a finite "
            f"sentinel or nodata=None"
        )
    if dtype.kind == "b":
        return  # np.iinfo has no bool entry, and there is no range to check
    info = np.iinfo(dtype)
    if not info.min <= nodata <= info.max:
        raise ValueError(
            f"nodata={nodata!r} is outside the range of {dtype} "
            f"[{info.min}, {info.max}]; no pixel can equal it"
        )


_WARP_GRID_STEP = 16

# Output-row block size for the separable two-pass accumulator.  Bounds the
# (bands, src_rows, dst_w) intermediate and keeps each pass cache-resident.
_SEPARABLE_ROW_BLOCK = 256


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
    # PROJ returns inf for out-of-domain input.  np.interp below would smear a
    # single inf node across a whole `step`-wide cell as NaN, and the index
    # casts after that are undefined — so fail loudly instead.
    if not (np.all(np.isfinite(cwx)) and np.all(np.isfinite(cwy))):
        raise ValueError(
            "Reprojecting the destination grid produced inf/nan coordinates; "
            "it reaches outside the source CRS's area of use."
        )

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
    weights: list[np.ndarray] = []
    for k in offsets:
        d = np.abs(k - frac) / scale
        d2 = d * d
        d3 = d2 * d
        w_inner = 1.5 * d3 - 2.5 * d2 + 1.0
        w_outer = -0.5 * d3 + 2.5 * d2 - 4.0 * d + 2.0
        weights.append(np.where(d < 1.0, w_inner, np.where(d < 2.0, w_outer, 0.0)))
    out = np.stack(weights)
    return out / out.sum(axis=0, keepdims=True)


def _accumulate_separable(
    src_array: np.ndarray,
    base_col: np.ndarray,
    base_row: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    x_offsets: Sequence[int],
    y_offsets: Sequence[int],
    nodata: int | float | None,
    nodata_is_nan: bool,
    method: Literal["bilinear", "cubic"],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Separable two-pass kernel accumulation for the same-CRS path.

    Equivalent to the non-separable 2-D loop in :func:`_resample_kernel`, but
    ``O(taps_x + taps_y)`` instead of ``O(taps_x · taps_y)``: convolve along
    columns into a ``(bands, src_rows, dst_w)`` intermediate, then along rows.
    Processed in output-row blocks so the intermediate stays bounded and each
    pass is cache-resident.

    ``base_col``/``base_row`` are 1-D ``(dst_w,)``/``(dst_h,)`` and ``wx``/``wy``
    are ``(taps, dst_w)``/``(taps, dst_h)`` (separable, same-CRS only).

    With ``nodata`` set, GDAL-style renormalization is two separable
    convolutions: a masked-value numerator and a valid-weight denominator
    over the per-source-pixel validity mask.  Without ``nodata``, samples are
    edge-replicated (clamped) with no renormalization.

    Returns ``(acc_val, acc_wt, per_dim_ok)`` for :func:`_finalize_kernel`.
    """
    n_bands, h, w = src_array.shape
    dst_w = base_col.shape[0]
    dst_h = base_row.shape[0]

    # Per-x-tap source columns (edge-clamped) and in-bounds, reused per block.
    safe_cols = [np.clip(base_col + dx, 0, w - 1) for dx in x_offsets]
    inb_cols = [(base_col + dx >= 0) & (base_col + dx < w) for dx in x_offsets]

    acc_val = np.empty((n_bands, dst_h, dst_w), dtype=np.float64)
    acc_wt = np.empty((dst_h, dst_w), dtype=np.float64) if nodata is not None else None

    # Source-pixel validity (all bands non-nodata), shared by every block.
    valid_src: np.ndarray | None = None
    if nodata is not None:
        if nodata_is_nan:
            valid_src = np.asarray(~np.isnan(src_array).any(axis=0))
        else:
            valid_src = np.asarray(~(src_array == nodata).any(axis=0))

    for r0 in range(0, dst_h, _SEPARABLE_ROW_BLOCK):
        r1 = min(r0 + _SEPARABLE_ROW_BLOCK, dst_h)
        br = base_row[r0:r1]
        # Source-row span this block touches (clamped into [0, h-1]).
        smin = int(np.clip(int(br.min()) + y_offsets[0], 0, h - 1))
        smax = int(np.clip(int(br.max()) + y_offsets[-1], 0, h - 1))
        nrows = smax - smin + 1

        if nodata is not None:
            assert valid_src is not None and acc_wt is not None
            vs_blk = valid_src[smin : smax + 1]  # (nrows, w)
            # Zero out invalid (incl. NaN) source values before the multiply.
            ms_blk = np.where(vs_blk, src_array[:, smin : smax + 1, :], 0.0)
            # Pass 1 (columns): masked-value numerator + valid-weight denom.
            inter_num = np.zeros((n_bands, nrows, dst_w), dtype=np.float64)
            inter_den = np.zeros((nrows, dst_w), dtype=np.float64)
            for j in range(len(x_offsets)):
                weff = wx[j] * inb_cols[j]  # (dst_w,)
                inter_num += ms_blk[:, :, safe_cols[j]] * weff
                inter_den += vs_blk[:, safe_cols[j]] * weff
            # Pass 2 (rows).
            val_blk = np.zeros((n_bands, r1 - r0, dst_w), dtype=np.float64)
            wt_blk = np.zeros((r1 - r0, dst_w), dtype=np.float64)
            for i, dy in enumerate(y_offsets):
                src_row_idx = br + dy
                sr = np.clip(np.clip(src_row_idx, 0, h - 1) - smin, 0, nrows - 1)
                weff_y = wy[i][r0:r1] * ((src_row_idx >= 0) & (src_row_idx < h))
                val_blk += inter_num[:, sr, :] * weff_y[None, :, None]
                wt_blk += inter_den[sr, :] * weff_y[:, None]
            acc_val[:, r0:r1, :] = val_blk
            acc_wt[r0:r1, :] = wt_blk
        else:
            src_blk = src_array[:, smin : smax + 1, :]
            inter = np.zeros((n_bands, nrows, dst_w), dtype=np.float64)
            for j in range(len(x_offsets)):
                inter += src_blk[:, :, safe_cols[j]] * wx[j]
            val_blk = np.zeros((n_bands, r1 - r0, dst_w), dtype=np.float64)
            for i, dy in enumerate(y_offsets):
                sr = np.clip(np.clip(br + dy, 0, h - 1) - smin, 0, nrows - 1)
                val_blk += inter[:, sr, :] * wy[i][r0:r1][None, :, None]
            acc_val[:, r0:r1, :] = val_blk

    per_dim_ok: np.ndarray | None = None
    if nodata is not None and method == "cubic":
        assert valid_src is not None
        per_dim_ok = _separable_cubic_per_dim_ok(
            valid_src, base_row, safe_cols, inb_cols, x_offsets, y_offsets, h, w
        )
    return acc_val, acc_wt, per_dim_ok


def _separable_cubic_per_dim_ok(
    valid_src: np.ndarray,
    base_row: np.ndarray,
    safe_cols: list[np.ndarray],
    inb_cols: list[np.ndarray],
    x_offsets: Sequence[int],
    y_offsets: Sequence[int],
    h: int,
    w: int,
) -> np.ndarray:
    """Separable form of the cubic per-dimension ≥2-valid safety gate.

    Reproduces ``(row_valid_counts >= 2).any(0) & (col_valid_counts >= 2).any(0)``
    from the 2-D loop without materializing the per-tap count tensors:
    ``xcount[sr, c]`` counts valid in-bounds x-taps at source row ``sr`` /
    dst col ``c``; ``ycount[r, sc]`` the y-analogue.  Returns a
    ``(dst_h, dst_w)`` boolean mask (True where the pixel passes the gate).
    """
    dst_w = safe_cols[0].shape[0]
    dst_h = base_row.shape[0]
    safe_rows = [np.clip(base_row + dy, 0, h - 1) for dy in y_offsets]
    inb_rows = [(base_row + dy >= 0) & (base_row + dy < h) for dy in y_offsets]

    xcount = np.zeros((h, dst_w), dtype=np.int32)
    for j in range(len(x_offsets)):
        xcount += valid_src[:, safe_cols[j]] & inb_cols[j]
    ycount = np.zeros((dst_h, w), dtype=np.int32)
    for i in range(len(y_offsets)):
        ycount += valid_src[safe_rows[i], :] & inb_rows[i][:, None]

    part_a = np.zeros((dst_h, dst_w), dtype=bool)
    for i in range(len(y_offsets)):
        part_a |= (xcount[safe_rows[i], :] >= 2) & inb_rows[i][:, None]
    part_b = np.zeros((dst_h, dst_w), dtype=bool)
    for j in range(len(x_offsets)):
        part_b |= (ycount[:, safe_cols[j]] >= 2) & inb_cols[j][None, :]
    return part_a & part_b


def _finalize_kernel(
    acc_val: np.ndarray,
    acc_wt: np.ndarray | None,
    per_dim_ok: np.ndarray | None,
    src_array: np.ndarray,
    center_row: np.ndarray,
    center_col: np.ndarray,
    coords_2d: bool,
    nodata: int | float | None,
    nodata_is_nan: bool,
) -> np.ndarray:
    """Renormalize, apply nodata gates, and cast to the source dtype.

    Shared by both the separable (same-CRS) and 2-D (cross-CRS) paths.
    ``center_row``/``center_col`` are 1-D for same-CRS and 2-D for cross-CRS;
    ``per_dim_ok`` is the precomputed cubic safety mask (or ``None``).
    """
    h, w = src_array.shape[1], src_array.shape[2]
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
            in_bounds_center = ((center_row >= 0) & (center_row < h))[:, None] & (
                (center_col >= 0) & (center_col < w)
            )[None, :]
        if nodata_is_nan:
            center_is_nodata = np.isnan(center_sample).any(axis=0)
        else:
            center_is_nodata = (center_sample == nodata).any(axis=0)

        invalid = (center_is_nodata | ~in_bounds_center) | ~has_weight
        if per_dim_ok is not None:
            invalid |= ~per_dim_ok
        if invalid.any():
            out_f[:, invalid] = float(nodata)
    else:
        out_f = acc_val

    src_dtype = src_array.dtype
    # Bool is not an np.integer, so it would skip the round below and let any
    # non-zero accumulation become True — dilating the mask.
    if src_dtype.kind == "b":
        return out_f >= 0.5
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        np.clip(out_f, info.min, info.max, out=out_f)
        np.round(out_f, out=out_f)
    return out_f.astype(src_dtype)


def _two_pass_threshold(strategy: WarpStrategy) -> float | None:
    """Local downsample scale above which two-pass applies, or None to disable.

    ``"auto"`` triggers only on the stronger downsamples where the
    widened-kernel cost clearly dominates; ``"single_pass"`` never triggers.
    """
    if strategy == "auto":
        return _AUTO_SCALE_THRESHOLD
    return None


def _two_pass_work_dtype(dtype: np.dtype) -> np.dtype:
    """Float dtype for the two-pass intermediate that avoids double rounding.

    Running both passes in float (the intermediate is never cast back to an
    integer dtype between them) means a single clip+round at the end instead
    of one per pass.  float32 is used only when it represents the source
    integer range exactly (mantissa is 24 bits); otherwise float64.  Float
    sources keep their own width (kernel accumulation is float64 regardless).
    """
    if np.issubdtype(dtype, np.floating):
        return dtype
    if dtype.kind == "b":
        return np.dtype(np.float32)
    info = np.iinfo(dtype)
    if info.min >= -(2**24) and info.max <= 2**24:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _resample_two_pass(
    src_array: np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    nodata: int | float | None,
    transformer: Transformer | None,
    method: Literal["bilinear", "cubic"],
    x_scale: float,
    y_scale: float,
) -> np.ndarray:
    """Cross-CRS downsample as two cheap passes instead of one wide warp.

    Pass A downsamples ``src_array`` in its own CRS to an intermediate grid at
    ~target resolution (``transformer=None`` → fast separable path).  Pass B
    reprojects that smaller intermediate to the final grid at near-unit scale
    (a narrow kernel).  Both passes run in float so the integer clip+round
    happens once, at the end.

    The intermediate is built with a halo beyond the source extent: Pass A's
    widened kernel and (for cubic) the ≥2-valid gate erode the outermost
    intermediate pixels under nodata, so the halo keeps that erosion off the
    region Pass B samples — avoiding an edge fringe single-pass would not have.

    See :func:`resample` / :func:`rastera.set_warp_strategy` for when this
    runs and how its output relates to the single-pass warp.
    """
    n_bands, h, w = src_array.shape
    base_radius = 1 if method == "bilinear" else 2

    # Per-axis: only ever downsample (scale <= 1 axes keep source resolution).
    sx = max(1.0, x_scale)
    sy = max(1.0, y_scale)
    core_w = max(1, round(w / sx))
    core_h = max(1, round(h / sy))

    # Intermediate pixel size that spans the source extent exactly with
    # ``core_w``/``core_h`` pixels, preserving the source axis signs.
    inter_a = float(src_transform.a) * w / core_w
    inter_e = float(src_transform.e) * h / core_h

    # Halo (intermediate pixels) >= Pass A's edge reach, plus one for the
    # cubic gate's slightly longer reach at nodata boundaries.
    halo = base_radius + 1
    inter_w = core_w + 2 * halo
    inter_h = core_h + 2 * halo
    origin_x = float(src_transform.c) - halo * inter_a
    origin_y = float(src_transform.f) - halo * inter_e
    inter_transform = Affine(inter_a, 0.0, origin_x, 0.0, inter_e, origin_y)

    orig_dtype = src_array.dtype
    work_dtype = _two_pass_work_dtype(orig_dtype)
    work = src_array.astype(work_dtype, copy=False)

    inter = resample(
        work,
        src_transform=src_transform,
        dst_transform=inter_transform,
        dst_width=inter_w,
        dst_height=inter_h,
        nodata=nodata,
        transformer=None,
        method=method,
        warp_strategy="single_pass",
    )
    out = resample(
        inter,
        src_transform=inter_transform,
        dst_transform=dst_transform,
        dst_width=dst_width,
        dst_height=dst_height,
        nodata=nodata,
        transformer=transformer,
        method=method,
        warp_strategy="single_pass",
    )

    # Both passes ran in float; clip+round+cast back to the source dtype once.
    if orig_dtype.kind == "b":
        return out >= 0.5
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = out.astype(np.float64, copy=False)
        np.clip(out, info.min, info.max, out=out)
        np.round(out, out=out)
    return out.astype(orig_dtype, copy=False)
