"""Benchmark harness: spawns fresh subprocesses for fair comparison.

Usage:
    python benchmarks/benchmark.py [--runs 5]

Each run is a fresh Python process so neither library benefits from
in-process caching (rastera TIFF header cache, GDAL VSI cache).

Scenarios
---------
1. Read: same CRS, native resolution (bbox subset)
2. Read: same CRS, downsampled to 60 m
3. Read: cross-CRS reproject to EPSG:4326, 0.001 deg
4. Merge: 2 adjacent UTM tiles, same CRS, 10 m resolution

Each scenario measures wall-clock time, peak RSS, output accuracy
(array comparison), and result consistency (mean, dtype, shape).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(_PROJECT_ROOT / ".venv" / "bin" / "python")
RUNNER = str(Path(__file__).parent / "runner.py")

# Sentinel-2 B03 over Rome — two adjacent UTM tiles
URI = "s3://e84-earth-search-sentinel-data/sentinel-2-c1-l2a/33/T/TG/2025/7/S2B_T33TTG_20250703T100029_L2A/B03.tif"
URI2 = "s3://e84-earth-search-sentinel-data/sentinel-2-c1-l2a/33/T/UG/2025/7/S2B_T33TUG_20250703T100029_L2A/B03.tif"

SCENARIOS = [
    # --- Single-file reads ---
    {
        "name": "Read: same CRS, native resolution (bbox subset)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
    },
    {
        "name": "Read: same CRS, downsampled to 60m",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "target_resolution": 60.0,
    },
    {
        "name": "Read: cross-CRS reproject to EPSG:4326, 0.001 deg",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "target_crs": 4326,
        "target_resolution": 0.001,
    },
    # --- Multi-file merges ---
    {
        "name": "Merge: 2 tiles, same CRS, 10m resolution",
        "mode": "merge",
        "bbox": "283838.0,4629464.7,326626.2,4648263.2",
        "bbox_crs": 32633,
        "target_resolution": 10.0,
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
) -> dict:
    if cold_cache:
        purge_page_cache()
    mode = scenario.get("mode", "read")
    cmd = [
        PYTHON,
        RUNNER,
        "--library",
        library,
        "--mode",
        mode,
        "--uri",
        URI,
        "--bbox",
        scenario["bbox"],
        "--bbox-crs",
        str(scenario["bbox_crs"]),
    ]
    if mode == "merge":
        cmd += ["--uri2", URI2]
    if "target_crs" in scenario:
        cmd += ["--target-crs", str(scenario["target_crs"])]
    if "target_resolution" in scenario:
        cmd += ["--target-resolution", str(scenario["target_resolution"])]
    if save_array:
        cmd += ["--save-array", save_array]

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
    a_raw = np.load(path_a)
    b_raw = np.load(path_b)
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
    if not accuracy["shapes_exact_match"]:
        print(
            f"    Shapes differ (off-by-one rounding): "
            f"rastera={accuracy['shape_rastera']} "
            f"rasterio={accuracy['shape_rasterio']}"
        )
        print(f"    Comparing overlap: {accuracy['compared_shape']}")
    else:
        print(f"    Shape: {accuracy['shape_rastera']}")
    print(
        f"    RMSE: {accuracy['rmse']}  "
        f"({accuracy.get('rmse_pct_of_range', 0)}% of data range)"
    )
    print(
        f"    Max abs error: {accuracy['max_abs_error']}  "
        f"(data range: {accuracy['data_range']})"
    )
    if "median_diff_where_nonzero" in accuracy:
        print(
            f"    Median diff (where nonzero): {accuracy['median_diff_where_nonzero']}"
        )
    print(f"    Pixels that differ: {accuracy['pct_pixels_differ']}%")
    if accuracy["pct_pixels_differ"] > 50:
        print(f"    Note: High pixel difference is expected for nearest-neighbor")
        print(
            f"    resampling — different grid alignment picks different source pixels."
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="Purge OS page cache before each run (requires sudo)",
    )
    args = parser.parse_args()

    if args.cold_cache:
        # Verify sudo works without password
        r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
        if r.returncode != 0:
            print("ERROR: --cold-cache requires passwordless sudo for 'purge'.")
            print("Run: sudo -v   (then re-run this script)")
            sys.exit(1)
        print("Cold-cache mode: purging OS page cache before each run\n")

    for scenario in SCENARIOS:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'=' * 60}")

        timings = {"rastera": [], "rasterio": []}
        memory = {"rastera": [], "rasterio": []}

        # First run: save arrays for accuracy comparison
        first_results = {}
        with (
            tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f_rastera,
            tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f_rasterio,
        ):
            for library in ["rastera", "rasterio"]:
                save_path = f_rastera.name if library == "rastera" else f_rasterio.name
                result = run_once(
                    scenario, library, save_array=save_path, cold_cache=args.cold_cache
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
                print(f"\n  Result consistency:")
                print(
                    f"    mean:  rastera={r['mean']}  rasterio={rio['mean']}  "
                    f"diff={abs(r['mean'] - rio['mean']):.4f}"
                )
                print(
                    f"    dtype: rastera={r['dtype']}  rasterio={rio['dtype']}  "
                    f"match={'yes' if r['dtype'] == rio['dtype'] else 'NO'}"
                )
                print(
                    f"    shape: rastera={r['shape']}  rasterio={rio['shape']}  "
                    f"match={'yes' if r['shape'] == rio['shape'] else 'NO'}"
                )

            # Accuracy comparison
            try:
                accuracy = compare_arrays(f_rastera.name, f_rasterio.name)
                print("\n  Accuracy comparison:")
                print_accuracy(accuracy)
            except Exception as e:
                print(f"  Accuracy comparison failed: {e}")

        # Remaining runs for timing
        for run_idx in range(2, args.runs + 1):
            for library in ["rastera", "rasterio"]:
                result = run_once(scenario, library, cold_cache=args.cold_cache)
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
            print(f"    rastera speedup: {speedup:.2f}x")
        if memory["rastera"] and memory["rasterio"]:
            mem_ratio = (
                median(memory["rasterio"]) / median(memory["rastera"])
                if median(memory["rastera"]) > 0
                else 0
            )
            print(f"    memory ratio (rasterio/rastera): {mem_ratio:.2f}x")


if __name__ == "__main__":
    main()
