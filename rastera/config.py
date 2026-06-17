from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Literal, TypeVar

_merge_concurrency: int = 1
_vrt_concurrency: int = 1
_dimap_concurrency: int = 1

# How a cross-CRS warp (bilinear/cubic reprojection) is carried out.  See
# ``set_warp_strategy`` for semantics.  Default is "auto": heavy cross-CRS
# downsamples take the faster two-pass route, everything else (nearest of
# any kind, and all same-CRS resamples) is unchanged.
WarpStrategy = Literal["auto", "single_pass"]
_warp_strategy: WarpStrategy = "auto"


def set_concurrency(
    *,
    merge: int | None = None,
    vrt: int | None = None,
    dimap: int | None = None,
) -> None:
    """Configure outer-loop concurrency for ``merge``, VRT source dispatch,
    and DIMAP tile reads. Default for all three is 1 (sequential).

    Inner concurrency is always on: async-geotiff already issues the
    per-tile range requests inside a single COG concurrently, regardless
    of these settings. Setting n>1 here stacks an *outer* fan-out on top
    of that — multiplying the in-flight HTTP request count by roughly
    n × inner_fanout. This can help when the inner read is small or
    latency-bound, but on saturated links or rate-limited buckets it
    risks connection-pool exhaustion and 429/SlowDown errors. Tune
    conservatively.

    Behavior per variant:

    - ``merge``: fan-out across contributing COGs in ``rastera.merge``.
      For ``mosaic_method="last"``, all contributors are read in one
      bounded gather. For ``mosaic_method="first"`` (the default),
      contributors are read in batches of ``merge`` and the early-exit
      check (``filled.all()``) runs between batches — so n>1 may
      over-fetch up to one batch worth of contributors compared to n=1.
    - ``vrt``: fan-out across distinct underlying sources for one VRT
      read. Bands are grouped by source first so each unique source is
      read once per call; n>1 reads multiple sources in parallel.
    - ``dimap``: fan-out across (band-group, tile) pairs inside a
      single DIMAP read. Each pair writes to a disjoint output region.
      Already-opened tiles are deduped via the single-flight tile
      cache, so n>1 only multiplies in-flight *block* reads, not tile
      header fetches.

    Pass ``None`` to leave a value unchanged. Values must be int >= 1.
    """
    global _merge_concurrency, _vrt_concurrency, _dimap_concurrency
    for name, val in (("merge", merge), ("vrt", vrt), ("dimap", dimap)):
        if val is None:
            continue
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(f"{name} concurrency must be int >= 1, got {val!r}")
    if merge is not None:
        _merge_concurrency = merge
    if vrt is not None:
        _vrt_concurrency = vrt
    if dimap is not None:
        _dimap_concurrency = dimap


def set_warp_strategy(strategy: WarpStrategy) -> None:
    """Select how a cross-CRS warp (a reprojecting resample) is carried out.

    This is a process-wide setting read by :func:`rastera.resampling.resample`.
    **It applies only to bilinear and cubic, and only when reprojecting
    (different source/target CRS) while downsampling.**  It has NO effect on:

    - ``nearest`` resampling — any CRS, any scale (it reads a single source
      pixel, so there is no kernel to widen and nothing to speed up);
    - same-CRS resamples (already use the fast separable path);
    - upsampling.

    A cross-CRS downsample widens the resampling kernel to anti-alias (e.g. a
    16cm→50cm reproject uses a ~14-tap cubic / ~7-tap bilinear kernel per axis)
    and runs a non-separable 2-D loop, which is ``O(taps_x · taps_y)``.  The
    two-pass strategy instead downsamples in the *source* CRS first (the fast
    separable same-CRS path), then reprojects the smaller intermediate at
    near-unit scale (a narrow kernel) — far cheaper for heavy downsamples
    (~3-4x for cubic, ~3x for bilinear at scale 3).

    Values:

    - ``"auto"`` (default): take the two-pass route only above a conservative
      downsample scale (> 2.0), where the kernel blow-up clearly dominates the
      small quality difference; single-pass otherwise.  Below ~1.75 two-pass is
      break-even-to-slower (its fixed intermediate-allocation + near-unit
      reproject cost is not yet repaid), so the conservative cutoff also avoids
      a needless quality hit where there is no speed win.
    - ``"single_pass"``: always the single non-separable warp.  Bit-exact with
      releases before two-pass existed.  Use this when exact reproducibility
      against older output matters.

    Two-pass output is *not* bit-identical to single-pass: it applies two
    resampling kernels, so it is marginally softer at the highest spatial
    frequencies (measured RMS < 1 DN for typical imagery).  Low-frequency
    content is unchanged.
    """
    valid = ("auto", "single_pass")
    if strategy not in valid:
        raise ValueError(f"warp strategy must be one of {valid}, got {strategy!r}")
    global _warp_strategy
    _warp_strategy = strategy


T = TypeVar("T")


async def _gather_bounded(n: int, coros: list[Awaitable[T]]) -> list[T]:
    """Run *coros* with at most n in flight. Returns results in input order."""
    if n <= 1 or len(coros) <= 1:
        return [await c for c in coros]
    sem = asyncio.Semaphore(n)

    async def _run(c: Awaitable[T]) -> T:
        async with sem:
            return await c

    return await asyncio.gather(*(_run(c) for c in coros))
