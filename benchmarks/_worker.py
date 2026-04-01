"""Single-run benchmark worker for rastera or rasterio.

Internal: spawned as a fresh subprocess by run.py to avoid in-process caching.
Not intended to be called directly.
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np


def read_rastera(
    uri: str,
    bbox: tuple,
    bbox_crs: int,
    target_crs: int | None,
    target_resolution: float | None,
    snap_to_grid: bool = True,
    use_overviews: bool = True,
) -> tuple[np.ndarray, list]:
    import asyncio

    import rastera

    async def _run():
        src = await rastera.open(uri)
        result = await src.read(
            bbox=bbox,
            bbox_crs=bbox_crs,
            target_crs=target_crs,
            target_resolution=target_resolution,
            snap_to_grid=snap_to_grid,
            use_overviews=use_overviews,
        )
        t = result.transform
        return result.data, [t.a, t.b, t.c, t.d, t.e, t.f]

    return asyncio.run(_run())


def merge_rastera(
    uris: list[str],
    bbox: tuple,
    bbox_crs: int,
    target_crs: int | None,
    target_resolution: float | None,
    snap_to_grid: bool = True,
    use_overviews: bool = True,
) -> tuple[np.ndarray, list]:
    import asyncio

    import rastera

    async def _run():
        sources = await rastera.open(uris)
        result = await rastera.merge(
            sources,
            bbox=bbox,
            bbox_crs=bbox_crs,
            target_crs=target_crs,
            target_resolution=target_resolution,
            snap_to_grid=snap_to_grid,
            use_overviews=use_overviews,
        )
        t = result.transform
        return result.data, [t.a, t.b, t.c, t.d, t.e, t.f]

    return asyncio.run(_run())


def read_rasterio(
    uri: str,
    bbox: tuple,
    bbox_crs: int,
    target_crs: int | None,
    target_resolution: float | None,
) -> tuple[np.ndarray, list]:
    import math
    import os

    import rasterio
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling
    from rasterio.windows import from_bounds

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
                dst_transform = Affine(
                    target_resolution, 0, minx, 0, -target_resolution, maxy
                )
                vrt_kwargs["transform"] = dst_transform
                vrt_kwargs["width"] = width
                vrt_kwargs["height"] = height

            with WarpedVRT(src, **vrt_kwargs) as vrt:
                data = vrt.read(resampling=Resampling.nearest)
                t = vrt.transform
                transform = [t.a, t.b, t.c, t.d, t.e, t.f]
        else:
            # Same CRS, same resolution — just read the bbox window
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            data = src.read(window=win, resampling=Resampling.nearest)
            t = src.window_transform(win)
            transform = [t.a, t.b, t.c, t.d, t.e, t.f]

    return data, transform


def merge_rasterio(
    uris: list[str],
    bbox: tuple,
    bbox_crs: int,
    target_crs: int | None,
    target_resolution: float | None,
) -> tuple[np.ndarray, list]:
    """Merge using rasterio.merge.merge — matches the notebook pattern."""
    import os

    import rasterio
    from rasterio.crs import CRS
    from rasterio.merge import merge

    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

    out_crs = CRS.from_epsg(target_crs) if target_crs else None

    # Transform bbox into the output CRS so merge() gets correct bounds
    merge_bounds = tuple(bbox)
    if out_crs and bbox_crs and target_crs != bbox_crs:
        from pyproj import Transformer as ProjTransformer

        t = ProjTransformer.from_crs(bbox_crs, target_crs, always_xy=True)
        minx, miny, maxx, maxy = bbox
        xs = [minx, maxx, minx, maxx]
        ys = [miny, miny, maxy, maxy]
        txs, tys = t.transform(xs, ys)
        merge_bounds = (min(txs), min(tys), max(txs), max(tys))

    datasets = [rasterio.open(u) for u in uris]
    vrts = []
    try:
        # Only use WarpedVRT when actual reprojection is needed.
        # Wrapping same-CRS datasets in VRT adds an unnecessary GDAL warp
        # layer that resamples through an intermediate grid.
        needs_vrt = out_crs and any(ds.crs != out_crs for ds in datasets)
        if needs_vrt:
            from rasterio.vrt import WarpedVRT
            from rasterio.warp import Resampling

            vrts = [
                WarpedVRT(ds, crs=out_crs, resampling=Resampling.nearest)
                for ds in datasets
            ]
            sources = vrts
        else:
            sources = datasets

        merge_kwargs = {"bounds": merge_bounds}
        if target_resolution is not None:
            merge_kwargs["res"] = target_resolution
        array, out_transform = merge(sources, **merge_kwargs)
    finally:
        for v in vrts:
            v.close()
        for ds in datasets:
            ds.close()

    t = out_transform
    return array, [t.a, t.b, t.c, t.d, t.e, t.f]


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
    parser.add_argument(
        "--no-snap-to-grid",
        action="store_true",
        default=False,
        help="Disable snap-to-grid (rastera only)",
    )
    parser.add_argument(
        "--no-overviews",
        action="store_true",
        default=False,
        help="Disable COG overview usage (rastera only)",
    )
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))

    t0 = time.perf_counter()
    if args.mode == "merge":
        uris = [args.uri]
        if args.uri2:
            uris.append(args.uri2)
        if args.library == "rastera":
            data, transform = merge_rastera(
                uris,
                bbox,
                args.bbox_crs,
                args.target_crs,
                args.target_resolution,
                snap_to_grid=not args.no_snap_to_grid,
                use_overviews=not args.no_overviews,
            )
        else:
            data, transform = merge_rasterio(
                uris, bbox, args.bbox_crs, args.target_crs, args.target_resolution
            )
    else:
        if args.library == "rastera":
            data, transform = read_rastera(
                args.uri,
                bbox,
                args.bbox_crs,
                args.target_crs,
                args.target_resolution,
                snap_to_grid=not args.no_snap_to_grid,
                use_overviews=not args.no_overviews,
            )
        else:
            data, transform = read_rasterio(
                args.uri, bbox, args.bbox_crs, args.target_crs, args.target_resolution
            )
    elapsed = time.perf_counter() - t0

    # Peak RSS in MB (macOS reports bytes, Linux reports KB)
    import platform

    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_bytes = (
        ru.ru_maxrss if platform.system() == "Darwin" else ru.ru_maxrss * 1024
    )
    peak_rss_mb = round(peak_rss_bytes / (1024 * 1024), 1)

    result = {
        "library": args.library,
        "mode": args.mode,
        "elapsed_s": round(elapsed, 4),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "mean": round(float(np.mean(data)), 4),
        "peak_rss_mb": peak_rss_mb,
        "transform": [round(v, 10) for v in transform],
    }
    print(json.dumps(result))

    if args.save_array:
        import rasterio as _rio
        from rasterio.crs import CRS as _CRS
        from rasterio.transform import Affine as _Affine

        out_crs = _CRS.from_epsg(args.target_crs or args.bbox_crs)
        out_transform = _Affine(*transform)
        bands, height, width = data.shape
        with _rio.open(
            args.save_array,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=bands,
            dtype=data.dtype,
            crs=out_crs,
            transform=out_transform,
            compress="lzw",
        ) as dst:
            dst.write(data)


if __name__ == "__main__":
    main()
