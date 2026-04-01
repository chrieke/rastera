"""Benchmark harness: shared infrastructure for read and merge benchmarks.

Each run is a fresh Python process so neither library benefits from
in-process caching (rastera TIFF header cache, GDAL VSI cache).

Measures wall-clock time, peak RSS, output accuracy (pixel comparison),
result consistency (mean, dtype, shape), and spatial alignment (transform,
pixel size, bounds).

Usage:
    python -m benchmarks.read [--runs 5]
    python -m benchmarks.merge [--runs 5]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
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
    snap_to_grid: bool = True,
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
    if not snap_to_grid and library == "rastera":
        cmd += ["--no-snap-to-grid"]
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


def check_borders(path: str, threshold: float = 0.5) -> dict:
    """Check that no border row/column is majority a single value.

    Returns a dict with per-edge results and an overall ``ok`` flag.
    ``threshold`` is the fraction above which a border is considered bad
    (default 0.5 = majority).
    """
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)

    edges = {
        "top": arr[:, 0, :],
        "bottom": arr[:, -1, :],
        "left": arr[:, :, 0],
        "right": arr[:, :, -1],
    }

    results = {}
    ok = True
    for name, edge in edges.items():
        vals, counts = np.unique(edge, return_counts=True)
        dominant_frac = float(counts.max()) / edge.size
        bad = dominant_frac > threshold
        if bad:
            ok = False
        results[name] = {
            "dominant_value": float(vals[counts.argmax()]),
            "dominant_frac": round(dominant_frac, 3),
            "bad": bad,
        }

    return {"ok": ok, "edges": results}


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


def format_accuracy(accuracy: dict) -> list[str]:
    lines = []
    rmse_pct = accuracy.get("rmse_pct_of_range", 0)
    pct_diff = accuracy["pct_pixels_differ"]

    if not accuracy["shapes_exact_match"]:
        lines.append(
            f"    ⚠️  Shapes differ: "
            f"rastera={accuracy['shape_rastera']} "
            f"rasterio={accuracy['shape_rasterio']}"
        )
        lines.append(f"    Comparing overlap: {accuracy['compared_shape']}")
    else:
        lines.append(f"    ✅ Shape: {accuracy['shape_rastera']}")
    lines.append(
        f"    {'✅' if rmse_pct < 1 else '⚠️' if rmse_pct < 5 else '❌'} "
        f"RMSE: {accuracy['rmse']}  ({rmse_pct}% of data range)"
    )
    lines.append(
        f"    {'✅' if pct_diff < 1 else '⚠️' if pct_diff < 10 else '❌'} "
        f"Pixels that differ: {pct_diff}%"
    )
    return lines


def format_spatial_alignment(
    r: dict, rio: dict, snap_to_grid: bool = True
) -> list[str]:
    """Compare transforms (origin, pixel size, bounds) between two results."""
    lines = []
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

    lines.append(
        f"    {'✅' if res_match else '❌'} "
        f"pixel size: rastera={res_r}  rasterio={res_rio}"
    )
    snap_reason = "  (due to snapping)" if snap_to_grid and not shift_ok else ""
    lines.append(
        f"    {'✅' if shift_ok else '⚠️' if abs(px_shift_x) < 1 and abs(px_shift_y) < 1 else '❌'} "
        f"origin shift: dx={origin_dx:.6f}  dy={origin_dy:.6f}  "
        f"({px_shift_x:.3f} px, {px_shift_y:.3f} px){snap_reason}"
    )
    bounds_reason = "  (due to snapping)" if snap_to_grid and not bounds_ok else ""
    lines.append(
        f"    {'✅' if bounds_ok else '⚠️'} bounds match: {'yes' if bounds_ok else 'NO'}{bounds_reason}"
    )
    return lines


def _assess_result(
    scenario: dict,
    accuracy: dict | None,
    consistency: dict | None,
    border_check: dict | None = None,
) -> tuple[bool, str]:
    """Determine overall pass/fail against the scenario's ``expect`` spec.

    Each scenario should declare an ``expect`` dict with:
        shape_match:  bool   — shapes must be identical (default True)
        dtype_match:  bool   — dtypes must be identical (default True)
        max_pct_differ: float — max % pixels that differ (default 0)
        max_rmse_pct:   float — max RMSE as % of data range (default 0)
        note:         str    — explanation shown when expected differences occur
    """
    if "expect" not in scenario:
        return False, "missing 'expect' in scenario definition"
    expect = scenario["expect"]
    expect_shape = expect.get("shape_match", True)
    expect_dtype = expect.get("dtype_match", True)
    max_pct = expect.get("max_pct_differ", 0)
    max_rmse = expect.get("max_rmse_pct", 0)
    note = expect.get("note", "")

    problems = []

    if consistency:
        if expect_dtype and not consistency["dtype_ok"]:
            problems.append("dtype mismatch")
        if expect_shape and not consistency["shape_ok"]:
            problems.append("shape mismatch")

    if accuracy:
        pct_diff = accuracy["pct_pixels_differ"]
        rmse_pct = accuracy.get("rmse_pct_of_range", 0)
        if pct_diff > max_pct:
            problems.append(f"{pct_diff}% pixels differ (limit {max_pct}%)")
        if rmse_pct > max_rmse:
            problems.append(f"RMSE {rmse_pct}% of range (limit {max_rmse}%)")

    if border_check and not border_check["ok"]:
        bad_edges = [
            name for name, info in border_check["edges"].items() if info["bad"]
        ]
        problems.append(f"suspect border edges: {', '.join(bad_edges)}")

    if not problems:
        if note:
            return True, f"As expected: {note}"
        return True, "Results match."

    return False, "; ".join(problems)


def run_benchmarks(scenarios: list[dict]):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
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

    # Derive subdir from first scenario's mode (read / merge) + timestamp
    mode = scenarios[0].get("mode", "read")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    export_dir = (
        None
        if args.no_export
        else Path(__file__).parent / "data" / f"{mode}_{timestamp}"
    )
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)

    if args.cold_cache:
        # Verify sudo works without password
        r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
        if r.returncode != 0:
            print("ERROR: --cold-cache requires passwordless sudo for 'purge'.")
            print("Run: sudo -v   (then re-run this script)")
            sys.exit(1)

    report = []

    def out(line: str = ""):
        """Print and capture a line for the markdown report."""
        print(line)
        report.append(line)

    for scenario_idx, scenario in enumerate(scenarios, 1):
        # Slug for export filenames
        slug = f"{scenario_idx}_{scenario['name'].lower()}"
        slug = slug.replace(":", "").replace(",", "")
        slug = slug.replace("(", "").replace(")", "")
        slug = "_".join(slug.split())

        no_overviews = not scenario.get("use_overviews", False)
        snap_to_grid = scenario.get("snap_to_grid", True)
        timings = {"rastera": [], "rasterio": []}
        mem = {"rastera": [], "rasterio": []}

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

        accuracy = None
        consistency = None
        spatial_lines = None

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
                    mem[library].append(result.get("peak_rss_mb", 0))

            # Result consistency
            if "rastera" in first_results and "rasterio" in first_results:
                ra, rio = first_results["rastera"], first_results["rasterio"]
                consistency = {
                    "mean_ra": ra["mean"],
                    "mean_rio": rio["mean"],
                    "mean_diff": abs(ra["mean"] - rio["mean"]),
                    "dtype_ok": ra["dtype"] == rio["dtype"],
                    "dtype_ra": ra["dtype"],
                    "dtype_rio": rio["dtype"],
                    "shape_ok": ra["shape"] == rio["shape"],
                    "shape_ra": ra["shape"],
                    "shape_rio": rio["shape"],
                }
                if "transform" in ra and "transform" in rio:
                    spatial_lines = format_spatial_alignment(
                        ra, rio, snap_to_grid=snap_to_grid
                    )

            # Accuracy comparison
            try:
                accuracy = compare_arrays(
                    saved_paths["rastera"],
                    saved_paths["rasterio"],
                )
            except Exception:
                pass

            # Border sanity check (rastera output)
            border_check = None
            try:
                border_check = check_borders(saved_paths["rastera"])
            except Exception:
                pass

            if export_dir:
                export_paths = [saved_paths[lib] for lib in ["rastera", "rasterio"]]
            else:
                export_paths = None
        finally:
            if not export_dir:
                for path in saved_paths.values():
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        # Remaining runs for timing
        for _ in range(2, args.runs + 1):
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
                    mem[library].append(result.get("peak_rss_mb", 0))

        # ── Print structured report ──────────────────────────────
        out(f"\n{'=' * 60}")
        out(f"Scenario {scenario_idx}: {scenario['name']}")
        out(f"{'=' * 60}")

        # Overall verdict
        passed, reason = _assess_result(scenario, accuracy, consistency, border_check)
        out(f"\n  Result: {'✅ AS EXPECTED' if passed else '❌ UNEXPECTED DIFFERENCE'}")
        out(f"  Reason: {reason}")

        # Consistency
        if consistency:
            c = consistency
            out("\n  Result consistency:")
            out(
                f"    {'✅' if c['mean_diff'] < 5 else '⚠️' if c['mean_diff'] < 50 else '❌'} "
                f"mean:  rastera={c['mean_ra']}  rasterio={c['mean_rio']}  "
                f"diff={c['mean_diff']:.4f}"
            )
            out(
                f"    {'✅' if c['dtype_ok'] else '❌'} "
                f"dtype: rastera={c['dtype_ra']}  rasterio={c['dtype_rio']}"
            )
            out(
                f"    {'✅' if c['shape_ok'] else '❌'} "
                f"shape: rastera={c['shape_ra']}  rasterio={c['shape_rio']}"
            )

        # Spatial alignment
        if spatial_lines:
            out("\n  Spatial alignment:")
            for line in spatial_lines:
                out(line)

        # Border sanity
        if border_check:
            if border_check["ok"]:
                out("\n  Border sanity: ✅ no suspect edges")
            else:
                out("\n  Border sanity:")
                for edge_name, info in border_check["edges"].items():
                    if info["bad"]:
                        out(
                            f"    ❌ {edge_name}: {info['dominant_frac']*100:.0f}% "
                            f"is value {info['dominant_value']}"
                        )

        # Accuracy
        if accuracy:
            out("\n  Accuracy:")
            for line in format_accuracy(accuracy):
                out(line)

        # Speed summary
        out(f"\n  Speed ({args.runs} run{'s' if args.runs > 1 else ''}):")
        for library in ["rastera", "rasterio"]:
            t = timings[library]
            m = mem[library]
            if t:
                med = median(t)
                mem_med = median(m) if m else 0
                out(
                    f"    {library}: median={med:.3f}s  range=[{min(t):.3f}, {max(t):.3f}]  "
                    f"mem={mem_med:.0f}MB"
                )
            else:
                out(f"    {library}: all runs failed")
        if timings["rastera"] and timings["rasterio"]:
            speedup = median(timings["rasterio"]) / median(timings["rastera"])
            icon = "🟢" if speedup > 1.5 else "🟡" if speedup > 1.0 else "🔴"
            out(f"    {icon} rastera speedup: {speedup:.2f}x")
        if mem["rastera"] and mem["rasterio"]:
            mem_ratio = (
                median(mem["rasterio"]) / median(mem["rastera"])
                if median(mem["rastera"]) > 0
                else 0
            )
            out(f"    memory ratio (rasterio/rastera): {mem_ratio:.2f}x")

        # Export paths
        if export_dir:
            out("\n  Exported to:")
            for path in export_paths:
                out(f"    {path}")

    if export_dir:
        report_path = export_dir / "report.md"
        with open(report_path, "w") as f:
            f.write("```\n")
            f.write("\n".join(report))
            f.write("\n```\n")
        out(f"\n{'─' * 60}")
        out(f"All arrays exported to: {export_dir}")
        out(f"Report written to:      {report_path}")
