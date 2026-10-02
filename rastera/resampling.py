"""Pixel resampling for resolution changes and reprojection.

:func:`resample` is the entry point, dispatching on ``method`` to one of three
implementations. :func:`validate_resampling` is public alongside it so callers
with a fast path that never reaches :func:`resample` — ``AsyncGeoTIFF.read``'s
native read, ``merge`` — can still reject an unknown method at their own
boundary rather than silently ignoring it.

- ``"nearest"`` — nearest-neighbor, memory-tight 1D/2D index path.
- ``"bilinear"`` — separable linear kernel; 2×2 at upsampling/identity,
  widened proportionally when downsampling to act as an anti-aliasing
  low-pass filter (matches GDAL's warp behaviour).
- ``"cubic"`` — Keys cubic convolution (a = -0.5); 4×4 at
  upsampling/identity, similarly widened when downsampling.

Bilinear and cubic use GDAL-style nodata handling: kernel weights are
renormalized over valid samples, with a center-pixel nodata gate.  Cubic adds
a per-dimension ≥2-valid gate of rastera's own against overshoot from
negative weights at data/nodata boundaries; where gdalwarp samples cubic 4x4,
it instead falls back to bilinear for any pixel whose taps touch nodata or
the source edge, so the two differ there.  As in gdalwarp, each band's kernel
is judged on its own: a sentinel in one band leaves the others' kernels alone.
The center gate is per band as well.  gdalwarp gates a pixel only where the
center is nodata in every band, and fills a band that is nodata there on its
own from the valid neighbours.  So a multi-band read differs from gdalwarp at
those pixels; a single-band read does not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from affine import Affine
from pyproj import Transformer

from . import config
from .config import WarpStrategy
from .geo import _DENOISE_TOL

ResamplingMethod = Literal["nearest", "bilinear", "cubic"]
_RESAMPLING_METHODS = ("nearest", "bilinear", "cubic")

# Local downsample scale above which the ``"auto"`` strategy takes the two-pass
# cross-CRS route.  Set conservatively: benchmarking cross-CRS warps across
# image sizes (512²–4096²) and both kernels, scale 2.0 is the lowest threshold
# that is a speed win everywhere; below it two-pass is erratic and often slower
# (its fixed intermediate-allocation + near-unit reproject cost is not repaid —
# worst for cheap-kernel bilinear at large sizes, where it can run ~1.5x
# slower).  Below the threshold (and at scale <= 1 — upsampling — where the
# two-pass split has no benefit) the single-pass warp is used.
_AUTO_SCALE_THRESHOLD = 2.0

# Kernel half-width in source pixels at unit scale: bilinear samples 2x2, cubic
# 4x4.  Downsampling widens it (see the anti-aliasing expansion below).
_BASE_RADIUS = {"bilinear": 1, "cubic": 2}


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
      ``scale = max(1, dst_res / src_res)``, rounded to a whole factor
      within 0.05 of one, as gdalwarp does).  Matches
      ``Resampling.bilinear`` / ``gdalwarp -r bilinear``. No overshoot.
    - ``"cubic"``: Keys cubic convolution (a = -0.5). 4×4 at
      upsampling/identity; expanded to ``4·⌈scale⌉ × 4·⌈scale⌉`` when
      downsampling.  Matches ``Resampling.cubic`` / ``gdalwarp -r cubic``.
      Can overshoot the source value range (for integer dtypes, output
      is clipped to the dtype range and rounded).

    Where the destination grid is the source grid shifted by whole pixels,
    ``"bilinear"`` and ``"cubic"`` copy the source pixels as ``"nearest"``
    does, like gdalwarp.

    For ``"bilinear"`` and ``"cubic"`` with ``nodata`` set, nodata is
    handled GDAL-style: kernel weights are renormalized over valid
    samples (invalid samples are dropped from the kernel). A target
    pixel is set to ``nodata`` when the source pixel under the target
    center is nodata, when every kernel sample is nodata, or — for
    cubic only, and rastera's own rule — when fewer than 2 valid samples
    exist along each axis of the kernel window (negative cubic weights
    cause severe overshoot when valid/invalid samples alternate).  All of
    this is per band: a band comes out the same whether or not it is
    resampled together with others.  gdalwarp renormalizes per band too,
    but its center gate is not per band (see the module docstring).

    ``nodata`` may be a finite sentinel (e.g. -9999, 0) or NaN; NaN is
    detected via ``np.isnan`` so the center gate and renormalization
    behave identically across sentinel types.  A real value the kernel
    rounds or clips onto the sentinel is moved one step off it, as
    gdalwarp does, so it does not read as missing.

    A destination pixel with no source pixel under its center is blanked
    regardless of ``nodata`` — to the sentinel when one is declared, to zero
    when none is, which is what GDAL leaves where its warper never writes.
    ``AsyncGeoTIFF.read`` also reports which pixels those were, on
    ``RasterArray.mask``.

    Args:
        src_array: ``(bands, h, w)``.
        src_transform: Pixel→world for the source; *dst_transform* likewise for
            the destination.
        nodata: The sentinel for invalid pixels; the bilinear/cubic
            renormalization keys off it, and it fills out-of-bounds pixels
            (which are zeroed when no sentinel is declared).
        transformer: Target CRS → source CRS; ``None`` if same CRS.
        warp_strategy: How a cross-CRS bilinear/cubic warp is carried out.
            ``None`` (default) reads the process-wide setting from
            :func:`rastera.set_warp_strategy`; pass an explicit value to
            override it for this call (useful in tests). No effect on nearest
            (any CRS/scale), same-CRS, or upsampling.
    """
    out, coverage = _resample_impl(
        src_array,
        src_transform,
        dst_transform,
        dst_width,
        dst_height,
        nodata,
        transformer,
        method,
        warp_strategy=warp_strategy,
    )
    return _fill_uncovered(out, coverage, nodata)


def validate_resampling(method: str) -> None:
    """Reject an unknown resampling method."""
    if method not in _RESAMPLING_METHODS:
        expected = ", ".join(repr(m) for m in _RESAMPLING_METHODS)
        raise ValueError(
            f"Unknown resampling method {method!r}; expected one of {expected}."
        )


def _resample_impl(
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
    src_coverage: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """:func:`resample`, plus the coverage mask it computes along the way.

    Coverage is ``(dst_height, dst_width)``, True where a source pixel lies
    under the destination pixel's center, and ``None`` when every destination
    pixel is covered — the common case then allocates nothing.  It is computed
    whether or not the source declares a sentinel: every path below clamps its
    source indices, so an uncovered destination pixel is indistinguishable from
    a real one without it.

    *src_coverage* is ``(h, w)`` over *src_array* and names which source pixels
    are themselves real.  Only :func:`_resample_two_pass` passes it, to carry
    pass A's coverage through pass B.
    """
    validate_resampling(method)
    if warp_strategy is None:
        warp_strategy = config._warp_strategy

    _validate_grids(src_transform, dst_transform, dst_width, dst_height)
    _validate_dtype_nodata(src_array.dtype, nodata)
    if dst_width == 0 or dst_height == 0:
        empty = np.empty(
            (src_array.shape[0], dst_height, dst_width), dtype=src_array.dtype
        )
        return empty, None

    if method == "nearest":
        return _resample_nearest(
            src_array,
            src_transform,
            dst_transform,
            dst_width,
            dst_height,
            nodata,
            transformer,
            src_coverage,
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
        src_coverage,
    )


def _resample_nearest(
    src_array: np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    nodata: int | float | None,
    transformer: Transformer | None,
    src_coverage: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Nearest-neighbor resampling, and the coverage mask (see
    :func:`_resample_impl`).

    Memory-tight: same-CRS uses 1D index arrays, cross-CRS uses the
    coarse-grid transform with in-place ops.
    """
    h, w = src_array.shape[1], src_array.shape[2]

    if transformer is None:
        # Same CRS: compose affines and use 1D index arrays (no meshgrid).
        combined = ~src_transform @ dst_transform
        src_col_1d = _pixel_index(
            float(combined.a) * (np.arange(dst_width, dtype=np.float64) + 0.5)
            + float(combined.c)
        )
        src_row_1d = _pixel_index(
            float(combined.e) * (np.arange(dst_height, dtype=np.float64) + 0.5)
            + float(combined.f)
        )

        valid_col = (src_col_1d >= 0) & (src_col_1d < w)
        valid_row = (src_row_1d >= 0) & (src_row_1d < h)

        col_safe = np.clip(src_col_1d, 0, w - 1)
        row_safe = np.clip(src_row_1d, 0, h - 1)
        out = src_array[:, row_safe[:, np.newaxis], col_safe[np.newaxis, :]]

        if np.all(valid_col) and np.all(valid_row):
            coverage = None
        else:
            coverage = valid_row[:, np.newaxis] & valid_col[np.newaxis, :]
        if src_coverage is not None:
            coverage = _and_coverage(
                coverage, src_coverage[row_safe[:, np.newaxis], col_safe[np.newaxis, :]]
            )
        if nodata is not None and coverage is not None:
            out[:, ~coverage] = np.array(nodata, dtype=src_array.dtype)
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

        coverage = None if np.all(valid) else valid
        if src_coverage is not None:
            coverage = _and_coverage(coverage, src_coverage[src_row, src_col])
        del src_col, src_row

        if nodata is not None and coverage is not None:
            out[:, ~coverage] = np.array(nodata, dtype=src_array.dtype)

    return out, coverage


def _resample_kernel(
    src_array: np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_width: int,
    dst_height: int,
    nodata: int | float | None,
    transformer: Transformer | None,
    method: Literal["bilinear", "cubic"],
    warp_strategy: WarpStrategy,
    src_coverage: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
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
      ``base_radius · max(1, _kernel_scale(|dst_res / src_res|))`` (rounded
      up), where ``base_radius`` is 1 for bilinear and 2 for cubic.
      Upsampling and identity reads use the default radii; downsampling
      expands.
    - Weights are separable and computed once outside the loop, then
      pre-normalized along the tap axis so the kernel sums to 1.
    - Out-of-bounds taps (kernel reach beyond the source extent for
      output pixels near an edge) are treated as nodata for
      renormalization when ``nodata`` is set, and clamped (edge
      replicated) otherwise.  Either way the returned coverage mask marks
      the destination pixels whose *center* fell outside the source.
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
        combined = ~src_transform @ dst_transform
        src_col_f = float(combined.a) * (
            np.arange(dst_width, dtype=np.float64) + 0.5
        ) + float(combined.c)
        src_row_f = float(combined.e) * (
            np.arange(dst_height, dtype=np.float64) + 0.5
        ) + float(combined.f)
        coords_2d = False
        # Every destination center on a source center, one source pixel apart:
        # each kernel is a copy of that pixel, with weight 0 on its neighbours.
        # A NaN or inf neighbour still spreads through the 0 weight, and cubic's
        # ≥2-valid gate drops one-pixel-wide features, so copy instead, as
        # gdalwarp's "translation-on-pixel-boundaries" optimization does.
        if _on_src_centers(src_col_f, float(combined.a)) and _on_src_centers(
            src_row_f, float(combined.e)
        ):
            return _resample_nearest(
                src_array,
                src_transform,
                dst_transform,
                dst_width,
                dst_height,
                nodata,
                None,
                src_coverage,
            )
        # Local pixel scale = src pixels per dst pixel (= dst_res / src_res
        # along the axis-aligned same-CRS case).
        x_scale_local = _kernel_scale(abs(float(combined.a)))
        y_scale_local = _kernel_scale(abs(float(combined.e)))
    else:
        src_col_f, src_row_f = _coarse_grid_transform(
            dst_width, dst_height, dst_transform, src_transform, transformer
        )
        coords_2d = True
        # A global (not per-pixel) scale matches GDAL's warp behaviour.
        probe_col, probe_row = src_col_f, src_row_f
        if dst_width < 2 or dst_height < 2:
            # A 1-pixel axis has no step to measure; one pixel past the output
            # gives it one.
            probe_col, probe_row = _coarse_grid_transform(
                max(2, dst_width),
                max(2, dst_height),
                dst_transform,
                src_transform,
                transformer,
            )
        x_scale_local = _kernel_scale(_footprint(probe_col))
        y_scale_local = _kernel_scale(_footprint(probe_row))

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
                src_coverage,
            )

    h, w = src_array.shape[1], src_array.shape[2]

    # --- Anti-aliasing: GDAL expands the kernel radius when downsampling
    # (scale > 1) so that bilinear/cubic act as proper low-pass filters
    # over the wider source footprint covered by each dst pixel.  When
    # upsampling (scale < 1) the kernel keeps its default radius.
    x_filter = max(1.0, x_scale_local)
    y_filter = max(1.0, y_scale_local)
    n_x_radius = math.ceil(_BASE_RADIUS[method] * x_filter)
    n_y_radius = math.ceil(_BASE_RADIUS[method] * y_filter)
    x_offsets = tuple(range(1 - n_x_radius, n_x_radius + 1))
    y_offsets = tuple(range(1 - n_y_radius, n_y_radius + 1))
    weights_fn = _bilinear_weights if method == "bilinear" else _cubic_weights

    # The kernels drop a pixel from every band they are handed when any of
    # those bands holds the sentinel.  gdalwarp weights each band's kernel on
    # its own, so bands whose sentinel footprints differ are handed over one at
    # a time.
    if _sentinel_differs_across_bands(src_array, nodata, nodata_is_nan):
        band_groups = [src_array[b : b + 1] for b in range(src_array.shape[0])]
    else:
        band_groups = [src_array]
    accumulate = _accumulate_2d if coords_2d else _accumulate_separable

    def _kernel(col_f: np.ndarray, row_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The output at these destination coordinates, and which of those
        pixels have a source pixel under their center."""
        # Source pixel containing the dst center (pixel-corner convention).
        # Used for the OOB gate and the GDAL-style center-pixel nodata gate.
        center_col = _pixel_index(col_f)
        center_row = _pixel_index(row_f)

        # The same in-bounds-center test `_finalize_kernel` applies under
        # nodata, taken here because coverage holds with or without a sentinel.
        if coords_2d:
            covered = (
                (center_row >= 0)
                & (center_row < h)
                & (center_col >= 0)
                & (center_col < w)
            )
        else:
            covered = ((center_row >= 0) & (center_row < h))[:, np.newaxis] & (
                (center_col >= 0) & (center_col < w)
            )[np.newaxis, :]
        if src_coverage is not None:
            safe_row = np.clip(center_row, 0, h - 1)
            safe_col = np.clip(center_col, 0, w - 1)
            covered &= (
                src_coverage[safe_row, safe_col]
                if coords_2d
                else src_coverage[safe_row[:, np.newaxis], safe_col[np.newaxis, :]]
            )

        # Kernel base/frac: shift by -0.5 so the kernel interpolates between
        # source pixel CENTERS (at integer + 0.5 in src pixel-corner space).
        # Without this shift, a dst pixel landing exactly on a src pixel
        # center would be a 50/50 blend with the neighbour instead of the
        # exact src value.
        shifted_col = col_f - 0.5
        shifted_row = row_f - 0.5
        base_col = np.floor(shifted_col).astype(np.intp)
        base_row = np.floor(shifted_row).astype(np.intp)
        wx = weights_fn(shifted_col - base_col, x_offsets, x_filter)
        wy = weights_fn(shifted_row - base_row, y_offsets, y_filter)

        outs: list[np.ndarray] = []
        for bands in band_groups:
            acc_val, acc_wt, per_dim_ok = accumulate(
                bands,
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
            outs.append(
                _finalize_kernel(
                    acc_val,
                    acc_wt,
                    per_dim_ok,
                    bands,
                    center_row,
                    center_col,
                    coords_2d,
                    nodata,
                    nodata_is_nan,
                )
            )
        return (outs[0] if len(outs) == 1 else np.concatenate(outs)), covered

    # A block of destination rows at a time: the sums are (bands, rows, W) in
    # float64, plus (taps, rows, W) weights cross-CRS.  Over the whole grid they
    # peaked at 15x a uint8 output same-CRS and 49x cross-CRS for cubic.
    out = np.empty((src_array.shape[0], dst_height, dst_width), dtype=src_array.dtype)
    covered = np.empty((dst_height, dst_width), dtype=bool)
    for r0 in range(0, dst_height, _ROW_BLOCK):
        rows = slice(r0, r0 + _ROW_BLOCK)
        col_f = src_col_f[rows] if coords_2d else src_col_f
        out[:, rows], covered[rows] = _kernel(col_f, src_row_f[rows])
    return out, None if covered.all() else covered


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
    if math.isinf(nodata) or nodata != int(nodata):
        # A fractional sentinel matches no pixel and lands differently per
        # method — nearest truncates it on the cast, the kernels round it.
        raise ValueError(
            f"nodata={nodata!r} is not an integer and so cannot be represented "
            f"in {dtype}; no pixel can equal it"
        )
    if dtype.kind == "b":
        return  # np.iinfo has no bool entry, and there is no range to check
    info = np.iinfo(dtype)
    if not info.min <= nodata <= info.max:
        raise ValueError(
            f"nodata={nodata!r} is outside the range of {dtype} "
            f"[{info.min}, {info.max}]; no pixel can equal it"
        )
    # The kernels accumulate in float64 and write the sentinel back through
    # ``float(nodata)``, so a 64-bit sentinel past the mantissa comes back
    # altered — bilinear marks nodata with a value nearest never produces.
    if abs(int(nodata)) > 2**53:
        raise ValueError(
            f"nodata={nodata!r} exceeds 2**53 and is not exactly representable "
            f"in the float64 the resampling kernels use; pick a smaller sentinel"
        )


_WARP_GRID_STEP = 16
# The interpolation error, in source pixels summed over both axes, above which
# a coarse-grid cell is transformed per pixel. gdalwarp's default ``-et``.
_WARP_MAX_ERROR = 0.125

# Destination rows the kernels handle at a time.  Bounds their float64 sums,
# the separable accumulator's (bands, src_rows, dst_w) intermediate and the
# cross-CRS path's 2-D weights, and keeps each pass cache-resident.
_ROW_BLOCK = 256


def _coarse_grid_transform(
    dst_width: int,
    dst_height: int,
    dst_transform: Affine,
    src_transform: Affine,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform dst pixels to src pixel coords via coarse-grid interpolation.

    Instead of transforming every destination pixel through pyproj, transforms a
    coarse grid (every ``_WARP_GRID_STEP`` pixels) and bilinearly interpolates
    the rest.  Where the grid bends, near a pole or across the source CRS's 180°
    seam, each cell is also transformed at its centre and edge midpoints, and
    per pixel if any of them is off by more than ``_WARP_MAX_ERROR``.  The
    centre alone misses the bend around a pole, where the errors along the two
    axes cancel out.
    """
    src_inv = ~src_transform

    def _to_src(cols: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wx = float(dst_transform.a) * (cols + 0.5) + float(dst_transform.c)
        wy = float(dst_transform.e) * (rows + 0.5) + float(dst_transform.f)
        wx, wy = transformer.transform(wx, wy)
        # PROJ returns inf for out-of-domain input.  np.interp below would smear
        # a single inf node across a whole `step`-wide cell as NaN, and the index
        # casts after that are undefined — so fail loudly instead.
        if not (np.all(np.isfinite(wx)) and np.all(np.isfinite(wy))):
            raise ValueError(
                "Reprojecting the destination grid produced inf/nan coordinates; "
                "it reaches outside the source CRS's area of use."
            )
        return (
            float(src_inv.a) * wx + float(src_inv.c),
            float(src_inv.e) * wy + float(src_inv.f),
        )

    probe_cols = _probe_positions(dst_width)
    probe_rows = _probe_positions(dst_height)
    coarse_cols, coarse_rows = probe_cols[::2], probe_rows[::2]
    coarse_src_col, coarse_src_row = _to_src(*np.meshgrid(coarse_cols, coarse_rows))

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

    # In place, so each coordinate peaks at two full-size arrays rather than
    # four: this is the largest allocation of a cross-CRS kernel read.
    def _along_rows(temp: np.ndarray) -> np.ndarray:
        out = temp[row_lo]
        step = temp[row_lo + 1]
        step -= out
        step *= row_frac
        out += step
        return out

    src_col, src_row = _along_rows(temp_col), _along_rows(temp_row)

    # Measuring costs four times the nodes' transforms, so only where the
    # nodes' curvature says it may be needed.  On random CRS pairs and grids the
    # measured error came within 1.25x of that estimate, hence the margin.  It
    # misses where PROJ switches datum transformation at an area-of-use edge
    # (e.g. around EPSG:27700 or 2056): a jump of J px reads as J/8, so jumps
    # under about 0.5 px stay interpolated.
    bend = _bend(coarse_cols, coarse_rows, coarse_src_col, coarse_src_row)
    if bend <= _WARP_MAX_ERROR / 2:
        return src_col, src_row

    probe_src_col, probe_src_row = _to_src(*np.meshgrid(probe_cols, probe_rows))
    probes = np.ix_(probe_rows, probe_cols)
    off = (
        np.abs(src_col[probes] - probe_src_col)
        + np.abs(src_row[probes] - probe_src_row)
    ) > _WARP_MAX_ERROR
    # Per cell, whether any of its nodes, edge midpoints or centre is off.
    bent = _any_per_cell(_any_per_cell(off).T).T
    # One row of cells at a time, so a grid bent throughout holds at most one
    # row of cells' exact coordinates at once.
    cell_of_row = np.minimum(
        np.arange(dst_height) // _WARP_GRID_STEP, bent.shape[0] - 1
    )
    cell_of_col = np.minimum(np.arange(dst_width) // _WARP_GRID_STEP, bent.shape[1] - 1)
    for i in np.flatnonzero(bent.any(axis=1)):
        rows = np.flatnonzero(cell_of_row == i)
        cols = np.flatnonzero(bent[i, cell_of_col])
        cells = np.ix_(rows, cols)
        src_col[cells], src_row[cells] = _to_src(*np.meshgrid(cols, rows))

    return src_col, src_row


def _probe_positions(n: int) -> np.ndarray:
    """Pixel indices along one axis, alternating coarse-grid node and the probe
    midway to the next one.  Nodes sit every ``_WARP_GRID_STEP`` pixels and on
    the last pixel, so interpolation never extrapolates."""
    nodes = np.arange(0, n, _WARP_GRID_STEP)
    if nodes[-1] < n - 1:
        nodes = np.append(nodes, n - 1)
    out = np.empty(2 * len(nodes) - 1, dtype=nodes.dtype)
    out[::2] = nodes
    out[1::2] = (nodes[:-1] + nodes[1:]) // 2
    return out


def _bend(
    coarse_cols: np.ndarray,
    coarse_rows: np.ndarray,
    coarse_src_col: np.ndarray,
    coarse_src_row: np.ndarray,
) -> float:
    """The interpolation error the coarse grid's own curvature implies, in
    source pixels summed over both axes: a parabola through three nodes misses
    the chord at its midpoint by f''·h²/8.  Infinite along an axis with two
    nodes, which has no curvature to read."""
    total = 0.0
    for axis, nodes in ((1, coarse_cols), (0, coarse_rows)):
        if len(nodes) == 1:
            continue  # nothing is interpolated along this axis
        if len(nodes) == 2:
            return math.inf
        h = np.diff(nodes).astype(np.float64)
        shape = (1, -1) if axis == 1 else (-1, 1)
        for coord in (coarse_src_col, coarse_src_row):
            slope = np.diff(coord, axis=axis) / h.reshape(shape)
            curvature = 2 * np.diff(slope, axis=axis) / (h[:-1] + h[1:]).reshape(shape)
            total += float(np.abs(curvature).max())
    return total * _WARP_GRID_STEP**2 / 8


def _any_per_cell(flags: np.ndarray) -> np.ndarray:
    """Per cell along axis 0 of *flags* (laid out as :func:`_probe_positions`),
    whether its two nodes or its midpoint are set.  A one-node axis is one cell."""
    if len(flags) == 1:
        return flags
    return flags[:-1:2] | flags[1::2] | flags[2::2]


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


def _accumulate_2d(
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
    """Non-separable 2-D kernel accumulation for the cross-CRS warp.

    The source sampling grid is not axis-aligned with the output, so weights
    and indices are full 2-D and cannot be reused across rows or columns —
    hence ``O(taps_x · taps_y)`` here against :func:`_accumulate_separable`'s
    ``O(taps_x + taps_y)``.

    ``base_col``/``base_row`` and ``wx``/``wy`` are 2-D over the output grid,
    unlike the 1-D per-axis arrays the separable path takes.

    Returns ``(acc_val, acc_wt, per_dim_ok)`` for :func:`_finalize_kernel`.
    """
    n_bands, h, w = src_array.shape
    dst_height, dst_width = base_row.shape

    per_dim_ok: np.ndarray | None = None
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
                # tap is in-bounds; the bands here share one sentinel
                # footprint (see `_resample_kernel`).  NaN-sentinel: use
                # `np.isnan` and zero-out NaN samples before the multiply.
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
            (row_valid_counts >= 2).any(axis=0) & (col_valid_counts >= 2).any(axis=0)
        )

    return acc_val, acc_wt, per_dim_ok


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
    Handed one block of destination rows at a time, it reads only the source
    rows their taps reach.

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

    # Per-x-tap source columns (edge-clamped) and in-bounds.
    safe_cols = [np.clip(base_col + dx, 0, w - 1) for dx in x_offsets]
    inb_cols = [(base_col + dx >= 0) & (base_col + dx < w) for dx in x_offsets]

    # Only the source rows the y-taps reach, clamped into the source.  A tap
    # outside them is outside the source too, so on this slice the edge clamp
    # and the in-bounds test below give the same answer as on the full source.
    smin = int(np.clip(int(base_row.min()) + y_offsets[0], 0, h - 1))
    smax = int(np.clip(int(base_row.max()) + y_offsets[-1], 0, h - 1))
    src_blk = src_array[:, smin : smax + 1, :]
    br = base_row - smin
    nrows = smax - smin + 1

    acc_wt: np.ndarray | None = None
    valid_src: np.ndarray | None = None
    if nodata is not None:
        # Source-pixel validity (all bands non-nodata).
        if nodata_is_nan:
            valid_src = np.asarray(~np.isnan(src_blk).any(axis=0))
        else:
            valid_src = np.asarray(~(src_blk == nodata).any(axis=0))
        # Zero out invalid (incl. NaN) source values before the multiply.
        # A typed zero keeps the source dtype; a 0.0 fill would promote an
        # integer block to float64, 8x the size of a uint8 one.
        ms_blk = np.where(valid_src, src_blk, src_array.dtype.type(0))
        # Pass 1 (columns): masked-value numerator + valid-weight denom.
        inter_num = np.zeros((n_bands, nrows, dst_w), dtype=np.float64)
        inter_den = np.zeros((nrows, dst_w), dtype=np.float64)
        for j in range(len(x_offsets)):
            weff = wx[j] * inb_cols[j]  # (dst_w,)
            inter_num += ms_blk[:, :, safe_cols[j]] * weff
            inter_den += valid_src[:, safe_cols[j]] * weff
        # Pass 2 (rows).
        acc_val = np.zeros((n_bands, dst_h, dst_w), dtype=np.float64)
        acc_wt = np.zeros((dst_h, dst_w), dtype=np.float64)
        for i, dy in enumerate(y_offsets):
            src_row_idx = br + dy
            sr = np.clip(src_row_idx, 0, nrows - 1)
            weff_y = wy[i] * ((src_row_idx >= 0) & (src_row_idx < nrows))
            acc_val += inter_num[:, sr, :] * weff_y[None, :, None]
            acc_wt += inter_den[sr, :] * weff_y[:, None]
    else:
        inter = np.zeros((n_bands, nrows, dst_w), dtype=np.float64)
        for j in range(len(x_offsets)):
            inter += src_blk[:, :, safe_cols[j]] * wx[j]
        acc_val = np.zeros((n_bands, dst_h, dst_w), dtype=np.float64)
        for i, dy in enumerate(y_offsets):
            sr = np.clip(br + dy, 0, nrows - 1)
            acc_val += inter[:, sr, :] * wy[i][None, :, None]

    per_dim_ok: np.ndarray | None = None
    if nodata is not None and method == "cubic":
        assert valid_src is not None
        per_dim_ok = _separable_cubic_per_dim_ok(
            valid_src, br, safe_cols, inb_cols, x_offsets, y_offsets, nrows, w
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
        # In place: every pixel without weight is invalid and overwritten with
        # nodata below, so what the divide leaves there never shows.
        out_f = acc_val
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
        invalid = None
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
    out = out_f.astype(src_dtype)
    if nodata is not None and invalid is not None:
        _avoid_nodata(out, nodata, ~invalid)
    return out


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
    src_coverage: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
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

    Coverage composes across the passes: Pass A reports the halo ring as
    uncovered and Pass B takes that as its ``src_coverage``.  Both go through
    ``_resample_impl``, which does not fill — blanking the halo would leave
    Pass B's kernel averaging real data against the fill value.

    See :func:`resample` / :func:`rastera.set_warp_strategy` for when this
    runs and how its output relates to the single-pass warp.
    """
    n_bands, h, w = src_array.shape

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
    halo = _BASE_RADIUS[method] + 1
    inter_w = core_w + 2 * halo
    inter_h = core_h + 2 * halo
    origin_x = float(src_transform.c) - halo * inter_a
    origin_y = float(src_transform.f) - halo * inter_e
    inter_transform = Affine(inter_a, 0.0, origin_x, 0.0, inter_e, origin_y)

    orig_dtype = src_array.dtype
    work_dtype = _two_pass_work_dtype(orig_dtype)
    work = src_array.astype(work_dtype, copy=False)

    inter, inter_coverage = _resample_impl(
        work,
        src_transform=src_transform,
        dst_transform=inter_transform,
        dst_width=inter_w,
        dst_height=inter_h,
        nodata=nodata,
        transformer=None,
        method=method,
        warp_strategy="single_pass",
        src_coverage=src_coverage,
    )
    out, coverage = _resample_impl(
        inter,
        src_transform=inter_transform,
        dst_transform=dst_transform,
        dst_width=dst_width,
        dst_height=dst_height,
        nodata=nodata,
        transformer=transformer,
        method=method,
        warp_strategy="single_pass",
        src_coverage=inter_coverage,
    )

    # Both passes ran in float; clip+round+cast back to the source dtype once.
    if orig_dtype.kind == "b":
        return out >= 0.5, coverage
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = out.astype(np.float64, copy=False)
        # Before the round: Pass B wrote the sentinel exactly into the pixels it
        # gated, and moved any real value off it.
        gated = None if nodata is None else out == nodata
        np.clip(out, info.min, info.max, out=out)
        np.round(out, out=out)
        out = out.astype(orig_dtype)
        if nodata is not None and gated is not None:
            _avoid_nodata(out, nodata, ~gated)
        return out, coverage
    return out.astype(orig_dtype, copy=False), coverage


def _footprint(coord: np.ndarray) -> float:
    """How many source pixels a destination pixel spans along the source axis
    *coord* holds (source columns or rows), as the median over the grid.

    Once the CRSs are rotated against each other, a step along either
    destination axis moves along this source axis, so both count, as gdalwarp
    counts them. Counting only the matching one, a quarter turn read as no
    downsample at all. The median is robust against outliers near the source
    extent boundary.
    """
    extent = np.diff(coord[:-1], axis=1)
    np.abs(extent, out=extent)
    step = np.diff(coord[:, :-1], axis=0)
    np.abs(step, out=step)
    extent += step
    del step
    return float(np.median(extent, overwrite_input=True))


def _kernel_scale(scale: float) -> float:
    """The downsample factor a kernel is widened by: *scale*, rounded to a whole
    factor when within 0.05 of one, as gdalwarp does.

    Unrounded, a factor just above 1 doubled the taps for next to no weight: at
    1.005 a cross-CRS bilinear read sampled 4x4 where GDAL samples 2x2, and ran
    3.5x slower.

    ``_kernel_halo`` keeps the unrounded factor. Its ceiling is never below the
    rounded kernel's reach, and the halo sizes its factor separately (see
    ``reader._halo_bbox``), so rounding it too could cut the kernel short.
    """
    whole = round(scale)
    if scale > 1 and abs(scale - whole) < 0.05:
        return float(whole)
    return scale


def _pixel_index(coord: np.ndarray) -> np.ndarray:
    """The source pixel each source pixel coordinate falls in.

    ``floor``, except that a coordinate within ``_DENOISE_TOL`` of a pixel edge
    counts as on it and takes the pixel past it, as GDAL's nearest does. A
    same-CRS read of a half-pixel-phase source puts every destination center
    on an edge, and ``~transform``'s float noise broke those ties differently
    for each bbox origin: overlapping reads disagreed by a whole column.
    """
    nearest = np.rint(coord)
    on_edge = np.abs(coord - nearest) < _DENOISE_TOL
    return np.floor(np.where(on_edge, nearest, coord)).astype(np.intp)


def _on_src_centers(coord: np.ndarray, step: float) -> bool:
    """Whether 1-D source pixel coordinates *coord*, *step* apart, are one
    source pixel apart and each within ``_DENOISE_TOL`` of a pixel center.

    An odd downsample factor puts every center on a source center too; *step*
    rules it out.
    """
    if abs(step - 1.0) >= _DENOISE_TOL:
        return False
    off = coord - 0.5
    return bool(np.all(np.abs(off - np.rint(off)) < _DENOISE_TOL))


def _kernel_halo(method: ResamplingMethod, scale: float) -> int:
    """Source pixels a *method* kernel reaches beyond the one it samples.

    Widens with the downsample factor (see the anti-aliasing note in
    :func:`_resample_kernel`); *scale* is ``dst_res / src_res``.
    """
    if method == "nearest":
        return 0
    return math.ceil(_BASE_RADIUS[method] * max(1.0, scale))


def _sentinel_differs_across_bands(
    src_array: np.ndarray, nodata: int | float | None, nodata_is_nan: bool
) -> bool:
    """Whether the sentinel sits on different pixels in different bands."""
    if nodata is None or src_array.shape[0] == 1:
        return False
    is_nodata = np.isnan(src_array) if nodata_is_nan else src_array == nodata
    return not np.array_equal(is_nodata.any(axis=0), is_nodata.all(axis=0))


def _avoid_nodata(out: np.ndarray, nodata: int | float, valid: np.ndarray) -> None:
    """Move real pixels of *out* that landed on *nodata* one step off it.

    A kernel can round or clip a real value onto the sentinel — cubic undershoot
    next to a bright edge clips to a nodata of 0 — and everything downstream
    then reads it as missing. gdalwarp moves it the same way, warning "changed
    to ... to avoid being treated as NoData": an integer steps down, or up from
    the dtype minimum; a float moves to the next value up, or down from the
    dtype maximum or +inf. *valid* is ``(h, w)``: the pixels not deliberately
    nodata.
    """
    hit = (out == nodata) & valid
    if not hit.any():
        return
    if np.issubdtype(out.dtype, np.integer):
        step = 1 if nodata == np.iinfo(out.dtype).min else -1
        out[hit] = int(nodata) + step
    else:
        # From +inf, which a value overflowing the dtype lands on, too.
        toward = -np.inf if nodata >= np.finfo(out.dtype).max else np.inf
        out[hit] = np.nextafter(out.dtype.type(nodata), out.dtype.type(toward))


def _and_coverage(
    coverage: np.ndarray | None, covered: np.ndarray
) -> np.ndarray | None:
    """AND *covered* into *coverage*, collapsing an all-covered result to None."""
    merged = covered if coverage is None else coverage & covered
    return None if merged.all() else merged


def _fill_uncovered(
    out: np.ndarray, coverage: np.ndarray | None, nodata: int | float | None
) -> np.ndarray:
    """Blank the destination pixels no source pixel reaches.

    Only for the no-sentinel case; with *nodata* set the resamplers already
    wrote it there.  Without one nothing means "invalid", so zero — what GDAL
    leaves outside the source footprint — is the honest stand-in.  The mask is
    what actually distinguishes them; this keeps a caller that ignores it from
    being handed a replicated border pixel instead.
    """
    if coverage is None or nodata is not None:
        return out
    out[:, ~coverage] = 0
    return out
