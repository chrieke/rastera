"""Benchmark harness: spawns fresh subprocesses for fair comparison.

Usage:
    python benchmarks/run.py [--runs 5]

Each run is a fresh Python process so neither library benefits from
in-process caching (rastera TIFF header cache, GDAL VSI cache).

Measures wall-clock time, peak RSS, output accuracy (pixel comparison),
result consistency (mean, dtype, shape), and spatial alignment (transform,
pixel size, bounds).

Scenarios and expected accuracy
-------------------------------
1.  Read: same CRS, native resolution, not snapped — pixel-exact match
    (0% differ).  Transform differs from rasterio (sub-pixel origin shift).
2.  Read: same CRS, native resolution, snapped — pixel-exact match.
    Transform matches rasterio (grid-aligned origin).
3.  Read: same CRS, 60 m, no overviews (default) — same pipeline as rasterio.
4.  Read: same CRS, 60 m downsample via overviews — ~97% differ.  rastera
    reads from the 40 m COG overview (pre-averaged pixels); rasterio/GDAL
    WarpedVRT reads full 10 m resolution.  Different source data, not a bug.
5.  Read: cross-CRS reproject to EPSG:4326 (default) — ~2% differ.
    Different reprojection implementations (pyproj + numpy NN vs GDAL
    warp kernel).
6.  Read: cross-CRS reproject to EPSG:4326 via overviews — larger diff
    expected due to overview source pixels + different reprojection.
7.  Read: large bbox at 120 m — ~97% differ.  Same overview explanation
    as scenario 4 (rastera uses the 80 m overview).
8.  Merge: 2 adjacent UTM tiles, same CRS, 10 m, not snapped —
    near-exact (<0.1%).  Transform differs (sub-pixel origin shift).
9.  Merge: same as 8 but snapped — near-exact, transform matches.
10. Merge: cross-CRS (32632 + 32633), reproject to 32633, 10 m.
11. Merge: cross-CRS (32632 + 32633), reproject to 4326 — ~98% differ.
    Combination of overview use and different reprojection implementations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(_PROJECT_ROOT / ".venv" / "bin" / "python")
RUNNER = str(Path(__file__).parent / "_worker.py")

# Sentinel-2 B03 over Rome
URI = "s3://e84-earth-search-sentinel-data/sentinel-2-c1-l2a/33/T/TG/2025/7/S2B_T33TTG_20250703T100029_L2A/B03.tif"
# Adjacent tile in same UTM zone (EPSG:32633)
URI_33TUG = "s3://e84-earth-search-sentinel-data/sentinel-2-c1-l2a/33/T/UG/2025/7/S2B_T33TUG_20250703T100029_L2A/B03.tif"
# Adjacent tile in different UTM zone (EPSG:32632) — overlaps 33TTG across zone boundary
URI_32TQM = "s3://e84-earth-search-sentinel-data/sentinel-2-c1-l2a/32/T/QM/2025/7/S2B_T32TQM_20250703T100029_L2A/B03.tif"

SCENARIOS = [
    # --- Single-file reads ---
    {
        "name": "Read: same CRS, native resolution (bbox subset), not snapped - raster matches bbox exactly (default)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
    },
    {
        "name": "Read: same CRS, native resolution (bbox subset), snapped to raster grid (rastera)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "snap_to_grid": True,
    },
    {
        "name": "Read: same CRS, downsampled to 60m, no overviews (default)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "target_resolution": 60.0,
    },
    {
        "name": "Read: same CRS, downsampled to 60m via overviews (rastera)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "target_resolution": 60.0,
        "use_overviews": True,
    },
    {
        "name": "Read: cross-CRS reproject to EPSG:4326, 0.001 deg, no overviews (default)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "target_crs": 4326,
        "target_resolution": 0.001,
        "reproject_bbox": True,
    },
    {
        "name": "Read: cross-CRS reproject to EPSG:4326, 0.001 deg, via overviews (rastera)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "target_crs": 4326,
        "target_resolution": 0.001,
        "reproject_bbox": True,
        "use_overviews": True,
    },
    {
        "name": "Read: large bbox, overview resolution (120m)",
        "mode": "read",
        "bbox": "200000.0,4600000.0,300000.0,4700000.0",
        "bbox_crs": 32633,
        "target_resolution": 120.0,
    },
    # --- Multi-file merges ---
    {
        "name": "Merge: 2 tiles, same CRS, 10m resolution, not snapped - raster matches bbox exactly (default)",
        "mode": "merge",
        "bbox": "283838.0,4629464.7,326626.2,4648263.2",
        "bbox_crs": 32633,
        "target_resolution": 10.0,
    },
    {
        "name": "Merge: 2 tiles, same CRS, 10m resolution, snapped to raster grid (rastera)",
        "mode": "merge",
        "bbox": "283838.0,4629464.7,326626.2,4648263.2",
        "bbox_crs": 32633,
        "target_resolution": 10.0,
        "snap_to_grid": True,
    },
    {
        "name": "Merge: 2 tiles, cross-CRS (32632+32633), reproject to 32633, 10m",
        "mode": "merge",
        "uri": URI,
        "uri2": URI_32TQM,
        "bbox": "11.8,41.7,12.5,42.2",
        "bbox_crs": 4326,
        "target_crs": 32633,
        "target_resolution": 10.0,
    },
    {
        "name": "Merge: 2 tiles, cross-CRS (32632+32633), reproject to 4326, 0.001 deg",
        "mode": "merge",
        "uri": URI,
        "uri2": URI_32TQM,
        "bbox": "11.8,41.7,12.5,42.2",
        "bbox_crs": 4326,
        "target_crs": 4326,
        "target_resolution": 0.001,
    },
]


def purge_page_cache():
    """Drop OS page cache for cold-cache benchmarks. Requires sudo on macOS."""
    import platform

    if platform.system() == "Darwin":
        subprocess.run(["sudo", "-n", "purge"], capture_output=True)
    else:
        # Linux: drop page cache
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True,
        )


def run_once(
    scenario: dict,
    library: str,
    save_array: str | None = None,
    cold_cache: bool = False,
    snap_to_grid: bool = False,
    no_overviews: bool = False,
) -> dict:
    if cold_cache:
        purge_page_cache()
    mode = scenario.get("mode", "read")
    uri = scenario.get("uri", URI)
    uri2 = scenario.get("uri2", URI_33TUG)

    bbox_str = scenario["bbox"]
    bbox_crs = scenario["bbox_crs"]

    # rastera requires bbox_crs == target_crs; reproject the bbox upfront
    if (
        scenario.get("reproject_bbox")
        and library == "rastera"
        and "target_crs" in scenario
    ):
        from pyproj import Transformer as ProjTransformer

        target_crs = scenario["target_crs"]
        minx, miny, maxx, maxy = (float(x) for x in bbox_str.split(","))
        t = ProjTransformer.from_crs(bbox_crs, target_crs, always_xy=True)
        xs = [minx, maxx, minx, maxx]
        ys = [miny, miny, maxy, maxy]
        txs, tys = t.transform(xs, ys)
        bbox_str = f"{min(txs)},{min(tys)},{max(txs)},{max(tys)}"
        bbox_crs = target_crs

    cmd = [
        PYTHON,
        RUNNER,
        "--library",
        library,
        "--mode",
        mode,
        "--uri",
        uri,
        "--bbox",
        bbox_str,
        "--bbox-crs",
        str(bbox_crs),
    ]
    if mode == "merge":
        cmd += ["--uri2", uri2]
    if "target_crs" in scenario:
        cmd += ["--target-crs", str(scenario["target_crs"])]
    if "target_resolution" in scenario:
        cmd += ["--target-resolution", str(scenario["target_resolution"])]
    if save_array:
        cmd += ["--save-array", save_array]
    if snap_to_grid and library == "rastera":
        cmd += ["--snap-to-grid"]
    if no_overviews and library == "rastera":
        cmd += ["--no-overviews"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        print(f"  FAILED ({library}): {result.stderr.strip()}", file=sys.stderr)
        return {
            "library": library,
            "elapsed_s": float("inf"),
            "error": result.stderr.strip(),
        }

    return json.loads(result.stdout.strip())


def compare_arrays(path_a: str, path_b: str) -> dict:
    import rasterio

    with rasterio.open(path_a) as src:
        a_raw = src.read()
    with rasterio.open(path_b) as src:
        b_raw = src.read()
    a = a_raw.astype(np.float64)
    b = b_raw.astype(np.float64)

    # Crop to overlapping region (off-by-one from rounding differences is expected)
    exact_match = a.shape == b.shape
    min_bands = min(a.shape[0], b.shape[0])
    min_h = min(a.shape[1], b.shape[1])
    min_w = min(a.shape[2], b.shape[2])
    a = a[:min_bands, :min_h, :min_w]
    b = b[:min_bands, :min_h, :min_w]

    diff = np.abs(a - b)
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    nonzero = diff[diff > 0]

    result = {
        "shapes_exact_match": exact_match,
        "shape_rastera": list(a_raw.shape),
        "shape_rasterio": list(b_raw.shape),
        "compared_shape": [min_bands, min_h, min_w],
        "rmse": round(float(np.sqrt(np.mean(diff**2))), 4),
        "max_abs_error": round(float(np.max(diff)), 4),
        "pct_pixels_differ": round(float(np.mean(diff > 0) * 100), 2),
        "data_range": round(data_range, 1),
    }
    if len(nonzero) > 0:
        result["median_diff_where_nonzero"] = round(float(np.median(nonzero)), 1)
        result["rmse_pct_of_range"] = (
            round(result["rmse"] / data_range * 100, 2) if data_range > 0 else 0.0
        )
    return result


def print_accuracy(accuracy: dict):
    rmse_pct = accuracy.get("rmse_pct_of_range", 0)
    pct_diff = accuracy["pct_pixels_differ"]

    if not accuracy["shapes_exact_match"]:
        print(
            f"    ⚠️  Shapes differ: "
            f"rastera={accuracy['shape_rastera']} "
            f"rasterio={accuracy['shape_rasterio']}"
        )
        print(f"    Comparing overlap: {accuracy['compared_shape']}")
    else:
        print(f"    ✅ Shape: {accuracy['shape_rastera']}")
    print(
        f"    {'✅' if rmse_pct < 1 else '⚠️' if rmse_pct < 5 else '❌'} "
        f"RMSE: {accuracy['rmse']}  ({rmse_pct}% of data range)"
    )
    print(
        f"    {'✅' if pct_diff < 1 else '⚠️' if pct_diff < 10 else '❌'} "
        f"Pixels that differ: {pct_diff}%"
    )
    if pct_diff > 50:
        print(
            "    ↳ Expected: different overview strategies (rastera reads "
            "pre-built COG overviews; rasterio/GDAL downsamples from "
            "full-resolution data). Different source pixels."
        )


def print_spatial_alignment(r: dict, rio: dict):
    """Compare transforms (origin, pixel size, bounds) between two results."""
    t_r = r["transform"]  # [a, b, c, d, e, f]
    t_rio = rio["transform"]
    s_r, s_rio = r["shape"], rio["shape"]

    # Pixel size
    res_r = (t_r[0], t_r[4])  # (pixel_width, -pixel_height)
    res_rio = (t_rio[0], t_rio[4])
    res_match = res_r[0] == res_rio[0] and res_r[1] == res_rio[1]

    # Origin (top-left corner)
    origin_r = (t_r[2], t_r[5])  # (x, y)
    origin_rio = (t_rio[2], t_rio[5])
    origin_dx = origin_r[0] - origin_rio[0]
    origin_dy = origin_r[1] - origin_rio[1]
    # Shift in pixels
    px_shift_x = origin_dx / t_r[0] if t_r[0] else 0
    px_shift_y = origin_dy / t_r[4] if t_r[4] else 0

    # Bounds: bottom-right = origin + shape * pixel_size
    def bounds(t, s):
        minx = t[2]
        maxy = t[5]
        maxx = minx + s[2] * t[0]
        miny = maxy + s[1] * t[4]
        return (minx, miny, maxx, maxy)

    b_r = bounds(t_r, s_r)
    b_rio = bounds(t_rio, s_rio)

    shift_ok = abs(px_shift_x) < 0.01 and abs(px_shift_y) < 0.01
    bounds_ok = b_r == b_rio

    print("\n  Spatial alignment:")
    print(
        f"    {'✅' if res_match else '❌'} "
        f"pixel size: rastera={res_r}  rasterio={res_rio}"
    )
    print(
        f"    {'✅' if shift_ok else '⚠️' if abs(px_shift_x) < 1 and abs(px_shift_y) < 1 else '❌'} "
        f"origin shift: dx={origin_dx:.6f}  dy={origin_dy:.6f}  "
        f"({px_shift_x:.3f} px, {px_shift_y:.3f} px)"
    )
    print(
        f"    {'✅' if bounds_ok else '⚠️'} bounds match: {'yes' if bounds_ok else 'NO'}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="Purge OS page cache before each run (requires sudo)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip saving output arrays to benchmarks/data/",
    )
    args = parser.parse_args()

    export_dir = None if args.no_export else Path(__file__).parent / "data"
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting arrays to {export_dir}\n")

    if args.cold_cache:
        # Verify sudo works without password
        r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
        if r.returncode != 0:
            print("ERROR: --cold-cache requires passwordless sudo for 'purge'.")
            print("Run: sudo -v   (then re-run this script)")
            sys.exit(1)
        print("Cold-cache mode: purging OS page cache before each run\n")

    for scenario_idx, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{'=' * 60}")
        print(f"Scenario {scenario_idx}: {scenario['name']}")
        print(f"{'=' * 60}")

        # Slug for export filenames: "1_read_same_crs_native_resolution"
        slug = f"{scenario_idx}_{scenario['name'].lower()}"
        slug = slug.replace(":", "").replace(",", "")
        slug = slug.replace("(", "").replace(")", "")
        slug = "_".join(slug.split())

        no_overviews = not scenario.get("use_overviews", False)
        snap_to_grid = scenario.get("snap_to_grid", False)
        timings = {"rastera": [], "rasterio": []}
        memory = {"rastera": [], "rasterio": []}

        # First run: save arrays for accuracy comparison
        first_results = {}
        saved_paths = {}
        if export_dir:
            for library in ["rastera", "rasterio"]:
                saved_paths[library] = str(export_dir / f"{slug}_{library}.tif")
        else:
            for library, suffix in [("rastera", "_ra.tif"), ("rasterio", "_rio.tif")]:
                f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                saved_paths[library] = f.name
                f.close()

        try:
            for library in ["rastera", "rasterio"]:
                save_path = saved_paths[library]
                result = run_once(
                    scenario,
                    library,
                    save_array=save_path,
                    cold_cache=args.cold_cache,
                    snap_to_grid=snap_to_grid,
                    no_overviews=no_overviews,
                )
                if "error" not in result:
                    first_results[library] = result
                    timings[library].append(result["elapsed_s"])
                    memory[library].append(result.get("peak_rss_mb", 0))
                    print(
                        f"  {library} run 1: {result['elapsed_s']:.3f}s  "
                        f"mem={result.get('peak_rss_mb', '?')}MB  shape={result['shape']}"
                    )
                else:
                    print(f"  {library} run 1: FAILED")

            # Result consistency check
            if "rastera" in first_results and "rasterio" in first_results:
                r, rio = first_results["rastera"], first_results["rasterio"]
                mean_diff = abs(r["mean"] - rio["mean"])
                dtype_ok = r["dtype"] == rio["dtype"]
                shape_ok = r["shape"] == rio["shape"]
                print("\n  Result consistency:")
                print(
                    f"    {'✅' if mean_diff < 5 else '⚠️' if mean_diff < 50 else '❌'} "
                    f"mean:  rastera={r['mean']}  rasterio={rio['mean']}  "
                    f"diff={mean_diff:.4f}"
                )
                print(
                    f"    {'✅' if dtype_ok else '❌'} "
                    f"dtype: rastera={r['dtype']}  rasterio={rio['dtype']}"
                )
                print(
                    f"    {'✅' if shape_ok else '❌'} "
                    f"shape: rastera={r['shape']}  rasterio={rio['shape']}"
                )

                # Spatial alignment check
                if "transform" in r and "transform" in rio:
                    print_spatial_alignment(r, rio)

            # Accuracy comparison
            try:
                accuracy = compare_arrays(
                    saved_paths["rastera"],
                    saved_paths["rasterio"],
                )
                print("\n  Accuracy comparison:")
                print_accuracy(accuracy)
            except Exception as e:
                print(f"  Accuracy comparison failed: {e}")

            if export_dir:
                for lib, path in saved_paths.items():
                    print(f"  Saved: {path}")
        finally:
            if not export_dir:
                for path in saved_paths.values():
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        # Remaining runs for timing
        for run_idx in range(2, args.runs + 1):
            for library in ["rastera", "rasterio"]:
                result = run_once(
                    scenario,
                    library,
                    cold_cache=args.cold_cache,
                    snap_to_grid=snap_to_grid,
                    no_overviews=no_overviews,
                )
                if "error" not in result:
                    timings[library].append(result["elapsed_s"])
                    memory[library].append(result.get("peak_rss_mb", 0))
                    print(
                        f"  {library} run {run_idx}: {result['elapsed_s']:.3f}s  "
                        f"mem={result.get('peak_rss_mb', '?')}MB"
                    )
                else:
                    print(f"  {library} run {run_idx}: FAILED")

        # Summary
        print(f"\n  Summary ({args.runs} runs):")
        for library in ["rastera", "rasterio"]:
            t = timings[library]
            m = memory[library]
            if t:
                med = median(t)
                mem_med = median(m) if m else 0
                print(
                    f"    {library}: median={med:.3f}s  range=[{min(t):.3f}, {max(t):.3f}]  "
                    f"mem={mem_med:.0f}MB (peak RSS)"
                )
            else:
                print(f"    {library}: all runs failed")

        if timings["rastera"] and timings["rasterio"]:
            speedup = median(timings["rasterio"]) / median(timings["rastera"])
            icon = "🟢" if speedup > 1.5 else "🟡" if speedup > 1.0 else "🔴"
            print(f"    {icon} rastera speedup: {speedup:.2f}x")
        if memory["rastera"] and memory["rasterio"]:
            mem_ratio = (
                median(memory["rasterio"]) / median(memory["rastera"])
                if median(memory["rastera"]) > 0
                else 0
            )
            print(f"    memory ratio (rasterio/rastera): {mem_ratio:.2f}x")


if __name__ == "__main__":
    main()
