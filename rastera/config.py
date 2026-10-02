from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Coroutine
from typing import Any, Literal

_merge_concurrency: int = 1
_vrt_concurrency: int = 1
_dimap_concurrency: int = 1

# How a cross-CRS warp (bilinear/cubic reprojection) is carried out.  See
# ``set_warp_strategy`` for semantics.
WarpStrategy = Literal["auto", "single_pass"]
_warp_strategy: WarpStrategy = "single_pass"


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

    - ``merge``: fan-out across contributing COGs in ``rastera.merge``,
      in batches of ``merge``: a batch is read and pasted before the next
      starts, which bounds the arrays held at once. For
      ``mosaic_method="first"`` (the default) the early exit
      (``filled.all()``) and the skip of contributors whose cells are all
      filled run between batches, so n>1 may read contributors that n=1
      would skip.
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
    # Validate all three before assigning any, so one bad argument leaves the
    # process-wide settings as they were rather than half-applied.
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

    Process-wide setting read by :func:`rastera.resampling.resample`. Applies
    only to bilinear/cubic when reprojecting *and* downsampling — nearest,
    same-CRS resamples, and upsampling are unaffected.

    - ``"single_pass"`` (default): always the single warp, within 0.2 DN RMS
      of gdalwarp.
    - ``"auto"``: above a downsample scale of 2.0, downsample in the source
      CRS first (fast separable path) then reproject the smaller intermediate
      at near-unit scale: about 2x faster than the single warp for bilinear,
      3-5x for cubic. Below that threshold two-pass is break-even-to-slower,
      so single-pass is used.

    Two-pass is an approximation. It applies two kernels, on an intermediate
    grid sized from the read window, so its result is softer and depends on
    the bbox. On textured 8-bit imagery it is about 3 DN RMS from gdalwarp,
    and two overlapping reads disagree by up to 11 DN where they meet, which
    can show as seams between tiles. With scattered nodata, about 3.5% of
    pixels change between nodata and valid.
    """
    valid = ("auto", "single_pass")
    if strategy not in valid:
        raise ValueError(f"warp strategy must be one of {valid}, got {strategy!r}")
    global _warp_strategy
    _warp_strategy = strategy


async def _gather_bounded[T](n: int, coros: list[Awaitable[T]]) -> list[T]:
    """Run *coros* with at most n in flight. Returns results in input order.

    The first failure cancels the rest before it propagates. ``asyncio.gather``
    leaves them running: a ``build_index`` that hit one missing file kept
    issuing every other GET after the caller had its error.
    """
    if n <= 1 or len(coros) <= 1:
        results: list[T] = []
        try:
            for c in coros:
                results.append(await c)
        except BaseException:
            _close_unstarted(coros[len(results) + 1 :])
            raise
        return results
    sem = asyncio.Semaphore(n)

    async def _run(c: Awaitable[T]) -> T:
        async with sem:
            return await c

    tasks = [asyncio.ensure_future(_run(c)) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _close_unstarted(coros)
        raise


def _close_unstarted(coros: list[Awaitable[Any]]) -> None:
    """Close the coroutines a failure left unawaited. The caller built every
    one up front, so Python would warn that they were never awaited."""
    for c in coros:
        if isinstance(c, Coroutine):
            c.close()
