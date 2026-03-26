"""Single-run benchmark runner for rastera or rasterio.

Spawned as a fresh subprocess by the harness to avoid in-process caching.

Usage:
    python runner.py --library rastera --uri <cog_uri> \
        --bbox minx,miny,maxx,maxy --bbox-crs 32633 \
        [--target-crs 4326] [--target-resolution 100] \
        [--save-array /tmp/out.npy]
        [--mode read|merge] [--uri2 <second_cog_uri>]
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np


def read_rastera(uri: str, bbox: tuple, bbox_crs: int,
                 target_crs: int | None, target_resolution: float | None) -> np.ndarray:
    import asyncio
    import rastera

    async def _run():
        src = await rastera.open(uri)
        result = await src.read(
            bbox=bbox, bbox_crs=bbox_crs,
            target_crs=target_crs, target_resolution=target_resolution,
        )
        return result.data

    return asyncio.run(_run())


def merge_rastera(uris: list[str], bbox: tuple, bbox_crs: int,
                  target_crs: int | None, target_resolution: float | None) -> np.ndarray:
    import asyncio
    import rastera

    async def _run():
        sources = await rastera.open(uris)
        result = await rastera.merge(
            sources, bbox=bbox, bbox_crs=bbox_crs,
            target_crs=target_crs, target_resolution=target_resolution,
        )
        return result.data

    return asyncio.run(_run())


def read_rasterio(uri: str, bbox: tuple, bbox_crs: int,
                  target_crs: int | None, target_resolution: float | None) -> np.ndarray:
    import os
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling
    from rasterio.crs import CRS
    from rasterio.windows import from_bounds
    from affine import Affine
    import math

    # Match rastera's skip_signature=True for public S3 buckets
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

    out_crs = CRS.from_epsg(target_crs) if target_crs else CRS.from_epsg(bbox_crs)

    with rasterio.open(uri) as src:
        src_crs_epsg = src.crs.to_epsg()

        # Transform bbox into the output CRS for grid construction
        minx, miny, maxx, maxy = bbox
        if bbox_crs and target_crs and bbox_crs != target_crs:
            from pyproj import Transformer as ProjTransformer
            t = ProjTransformer.from_crs(bbox_crs, target_crs, always_xy=True)
            xs = [minx, maxx, minx, maxx]
            ys = [miny, miny, maxy, maxy]
            txs, tys = t.transform(xs, ys)
            minx, maxx = min(txs), max(txs)
            miny, maxy = min(tys), max(tys)
        elif bbox_crs and not target_crs and bbox_crs != src_crs_epsg:
            from pyproj import Transformer as ProjTransformer
            t = ProjTransformer.from_crs(bbox_crs, src_crs_epsg, always_xy=True)
            xs = [minx, maxx, minx, maxx]
            ys = [miny, miny, maxy, maxy]
            txs, tys = t.transform(xs, ys)
            minx, maxx = min(txs), max(txs)
            miny, maxy = min(tys), max(tys)

        if target_crs or target_resolution:
            vrt_kwargs = {
                "crs": out_crs,
                "resampling": Resampling.nearest,
            }
            if target_resolution:
                width = max(1, math.ceil((maxx - minx) / target_resolution))
                height = max(1, math.ceil((maxy - miny) / target_resolution))
                dst_transform = Affine(target_resolution, 0, minx,
                                       0, -target_resolution, maxy)
                vrt_kwargs["transform"] = dst_transform
                vrt_kwargs["width"] = width
                vrt_kwargs["height"] = height

            with WarpedVRT(src, **vrt_kwargs) as vrt:
                data = vrt.read(resampling=Resampling.nearest)
        else:
            # Same CRS, same resolution — just read the bbox window
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            data = src.read(window=win, resampling=Resampling.nearest)

    return data


def merge_rasterio(uris: list[str], bbox: tuple, bbox_crs: int,
                   target_crs: int | None, target_resolution: float | None) -> np.ndarray:
    """Merge using rasterio.merge.merge — matches the notebook pattern."""
    import os
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling

    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

    res = target_resolution or 10
    datasets = [rasterio.open(u) for u in uris]
    try:
        array, transform = merge(
            datasets,
            bounds=tuple(bbox),
            res=res,
        )
    finally:
        for ds in datasets:
            ds.close()

    return array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, choices=["rastera", "rasterio"])
    parser.add_argument("--mode", default="read", choices=["read", "merge"])
    parser.add_argument("--uri", required=True)
    parser.add_argument("--uri2", default=None, help="Second URI for merge mode")
    parser.add_argument("--bbox", required=True, help="minx,miny,maxx,maxy")
    parser.add_argument("--bbox-crs", required=True, type=int)
    parser.add_argument("--target-crs", type=int, default=None)
    parser.add_argument("--target-resolution", type=float, default=None)
    parser.add_argument("--save-array", default=None, help="Path to save output .npy")
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))

    t0 = time.perf_counter()
    if args.mode == "merge":
        uris = [args.uri]
        if args.uri2:
            uris.append(args.uri2)
        if args.library == "rastera":
            data = merge_rastera(uris, bbox, args.bbox_crs,
                                 args.target_crs, args.target_resolution)
        else:
            data = merge_rasterio(uris, bbox, args.bbox_crs,
                                  args.target_crs, args.target_resolution)
    else:
        if args.library == "rastera":
            data = read_rastera(args.uri, bbox, args.bbox_crs,
                                args.target_crs, args.target_resolution)
        else:
            data = read_rasterio(args.uri, bbox, args.bbox_crs,
                                 args.target_crs, args.target_resolution)
    elapsed = time.perf_counter() - t0

    # Peak RSS in MB (macOS reports bytes, Linux reports KB)
    import platform
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_bytes = ru.ru_maxrss if platform.system() == "Darwin" else ru.ru_maxrss * 1024
    peak_rss_mb = round(peak_rss_bytes / (1024 * 1024), 1)

    result = {
        "library": args.library,
        "mode": args.mode,
        "elapsed_s": round(elapsed, 4),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "mean": round(float(np.mean(data)), 4),
        "peak_rss_mb": peak_rss_mb,
    }
    print(json.dumps(result))

    if args.save_array:
        np.save(args.save_array, data)


if __name__ == "__main__":
    main()
