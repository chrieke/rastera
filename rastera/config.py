from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Coroutine
from typing import Any, Literal

_merge_concurrency: int = 1
_vrt_concurrency: int = 1
_dimap_concurrency: int = 1

WarpStrategy = Literal["auto", "single_pass"]
_warp_strategy: WarpStrategy = "single_pass"


def set_concurrency(
    *,
    merge: int | None = None,
    vrt: int | None = None,
    dimap: int | None = None,
) -> None:
    """How many reads ``merge`` contributors, VRT sources and DIMAP tiles run
    at once. All default to 1. ``None`` leaves a value unchanged.

    async-geotiff already fetches the blocks inside one file concurrently, so
    n > 1 multiplies the requests in flight. That helps small, latency-bound
    reads, and on a saturated link or a rate-limited bucket risks
    connection-pool exhaustion and 429/SlowDown errors.

    - ``merge``: contributors in batches of n, each pasted before the next is
      read. With ``mosaic_method="first"`` the skip of contributors whose
      cells are filled runs between batches, so n > 1 may read ones n = 1
      skips.
    - ``vrt``: the distinct sources of one VRT read.
    - ``dimap``: the (band group, tile) pairs of one DIMAP read.
    """
    global _merge_concurrency, _vrt_concurrency, _dimap_concurrency
    # All three first, so one bad argument changes none.
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
    """Set how a bilinear or cubic read warps when it reprojects and
    downsamples.

    - ``"single_pass"`` (default): one warp, within 0.2 DN RMS of gdalwarp.
    - ``"auto"``: above a 2x downsample, downsample in the source CRS first,
      then reproject the smaller result. About 2x faster for bilinear and 3-5x
      for cubic.

    Two-pass applies two kernels on an intermediate grid sized from the read
    window, so it is softer and depends on the bbox: about 3 DN RMS from
    gdalwarp on textured 8-bit imagery, up to 11 DN between two overlapping
    reads (seams between tiles), and 3.5% of pixels flipping between nodata
    and valid with scattered nodata.
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
