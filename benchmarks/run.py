"""Benchmark harness: shared infrastructure for read and merge benchmarks.

Each run is a fresh Python process so neither library benefits from
in-process caching (rastera TIFF header cache, GDAL VSI cache).

Measures wall-clock time, peak RSS, output accuracy (pixel comparison),
result consistency (mean, dtype, shape), and spatial alignment (transform,
pixel size, bounds).

Timings and spatial alignment describe each library on its own defaults — its own
grid anchoring, and GDAL free to pick an overview. Accuracy cannot: on two grids a
pixel apart it would measure the offset rather than the pixels. So rasterio is
re-run on rastera's terms and only those pixels are compared — see
:func:`check_shared_grid` for what that proves.

Usage (as scripts, not ``-m``: run_read/run_merge import this module as a
sibling, which needs their own directory on sys.path):
    python benchmarks/run_read.py [--runs 5]
    python benchmarks/run_merge.py [--runs 5]
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
from _worker import _corner_hull

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(_PROJECT_ROOT / ".venv" / "bin" / "python")
RUNNER = str(Path(__file__).parent / "_worker.py")

# Sentinel-2 B03 over Rome
URI = "s3://e84-earth-search-sentinel-data/sentinel-2-c1-l2a/33/T/TG/2025/7/S2B_T33TTG_20250703T100029_L2A/B03.tif"
# Adjacent tile in same UTM zone (EPSG:32633)
URI_33TUG = "s3://e84-earth-search-sentinel-data/sentinel-2-c1-l2a/33/T/UG/2025/7/S2B_T33TUG_20250703T100029_L2A/B03.tif"
# Adjacent tile in different UTM zone (EPSG:32632) — overlaps 33TTG across zone boundary
URI_32TQM = "s3://e84-earth-search-sentinel-data/sentinel-2-c1-l2a/32/T/QM/2025/7/S2B_T32TQM_20250703T100029_L2A/B03.tif"

# How far a densified bbox hull may outgrow a four-corner one before the difference
# stops being edge curvature and starts being a bug (under a pixel per axis on this
# suite's UTM/4326 bboxes).
_HULL_SLACK_PX = 2


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
    *,
    # Required: which library gets which value is the whole basis of the
    # comparison, so there is no default that is right for both.
    snap_to_grid: bool,
    use_overviews: bool,
) -> dict:
    if cold_cache:
        purge_page_cache()
    mode = scenario.get("mode", "read")
    uri = scenario.get("uri", URI)
    uri2 = scenario.get("uri2", URI_33TUG)

    bbox_str = scenario["bbox"]
    bbox_crs = scenario["bbox_crs"]

    if library == "rastera":
        bbox, bbox_crs = _rastera_bbox(scenario)
        if bbox_crs != scenario["bbox_crs"]:
            bbox_str = ",".join(str(v) for v in bbox)

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
    cmd += ["--snap-to-grid" if snap_to_grid else "--no-snap-to-grid"]
    cmd += ["--overviews" if use_overviews else "--no-overviews"]

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


def compare_arrays(path_a: str, path_b: str, *, by_transform: bool = True) -> dict:
    """Compare two rasters over the ground they share.

    With *by_transform* both must be on one grid — same pixel size, origins a whole
    number of pixels apart — which is what the aligned re-run guarantees; out of
    phase raises rather than being reported as a resampling difference. Extents may
    still differ by outward rounding, so each array is sliced to the overlap via its
    transform: cropping from ``[0, 0]`` would compare ground a row apart as soon as
    one grid rounds north rather than south.

    A native-resolution read passes ``by_transform=False``. Both sides read the same
    source pixels from ``floor(offset)``, but an unsnapped output reports the *bbox*
    as its origin — a sub-pixel fiction rastera mirrors from rasterio deliberately
    (``geo.window_from_bbox``) that says nothing about the pixels.
    """
    import rasterio

    with rasterio.open(path_a) as src:
        a_raw, t_a = src.read(), src.transform
    with rasterio.open(path_b) as src:
        b_raw, t_b = src.read(), src.transform

    if (t_a.a, t_a.e) != (t_b.a, t_b.e):
        raise ValueError(
            f"pixel sizes differ: {(t_a.a, t_a.e)} vs {(t_b.a, t_b.e)}; nothing to compare"
        )
    off_x, off_y = _pixel_offset(t_a, t_b) if by_transform else (0, 0)

    # b's origin sits off_x cols / off_y rows into a's grid; negative means a's
    # origin sits inside b's instead, so the leading pixels to drop swap sides.
    a_col, b_col = max(0, off_x), max(0, -off_x)
    a_row, b_row = max(0, off_y), max(0, -off_y)
    min_bands = min(a_raw.shape[0], b_raw.shape[0])
    min_h = min(a_raw.shape[1] - a_row, b_raw.shape[1] - b_row)
    min_w = min(a_raw.shape[2] - a_col, b_raw.shape[2] - b_col)
    if min_h <= 0 or min_w <= 0:
        raise ValueError("the two rasters cover no common ground")

    exact_match = a_raw.shape == b_raw.shape and (off_x, off_y) == (0, 0)
    a = a_raw[:min_bands, a_row : a_row + min_h, a_col : a_col + min_w].astype(
        np.float64
    )
    b = b_raw[:min_bands, b_row : b_row + min_h, b_col : b_col + min_w].astype(
        np.float64
    )

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


def format_accuracy(accuracy: dict, expect: dict) -> list[str]:
    """Render the pixel comparison against *expect*'s own limits, so a scenario
    whose point is a bounded difference does not print ❌ under an ✅ verdict.
    """
    lines = []
    rmse_pct = accuracy.get("rmse_pct_of_range", 0)
    pct_diff = accuracy["pct_pixels_differ"]
    max_rmse = expect.get("max_rmse_pct", 0)
    max_pct = expect.get("max_pct_differ", 0)

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
        f"    {'✅' if rmse_pct <= max_rmse else '❌'} "
        f"RMSE: {accuracy['rmse']}  ({rmse_pct}% of data range, limit {max_rmse}%)"
    )
    lines.append(
        f"    {'✅' if pct_diff <= max_pct else '❌'} "
        f"Pixels that differ: {pct_diff}%  (limit {max_pct}%)"
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
    grid_check: dict | None = None,
    accuracy_error: str | None = None,
) -> tuple[bool, str]:
    """Determine overall pass/fail against the scenario's ``expect`` spec.

    Each scenario should declare an ``expect`` dict with:
        shape_match:  bool   — shapes must be identical (default True)
        dtype_match:  bool   — dtypes must be identical (default True)
        max_pct_differ: float — max % pixels that differ (default 0)
        max_rmse_pct:   float — max RMSE as % of data range (default 0)
        note:         str    — explanation shown when expected differences occur

    ``shape_match`` judges the libraries' *default* grids; the pixel limits judge a
    shared one (:func:`compare_arrays`), so a nonzero limit there means the
    resampling differs, not the grids.
    """
    if "expect" not in scenario:
        return False, "missing 'expect' in scenario definition"
    expect = scenario["expect"]
    expect_shape = expect.get("shape_match", True)
    expect_dtype = expect.get("dtype_match", True)
    max_pct = expect.get("max_pct_differ", 0)
    max_rmse = expect.get("max_rmse_pct", 0)
    note = expect.get("note", "")

    # A missing comparison has to fail, or a scenario that never produced a pixel
    # reports "as expected".
    problems = []
    if accuracy is None:
        problems.append(f"no accuracy comparison ({accuracy_error or 'a run failed?'})")
    if consistency is None:
        problems.append("no result consistency (a run failed?)")

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

    if grid_check and not grid_check["ok"]:
        problems.append(f"shared grid not reached: {grid_check['detail']}")

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

        use_overviews = scenario.get("use_overviews", False)
        snap_to_grid = scenario.get("snap_to_grid", True)
        timings = {"rastera": [], "rasterio": []}
        mem = {"rastera": [], "rasterio": []}

        # First run: save arrays for accuracy comparison
        first_results = {}
        saved_paths = {}
        names = ["rastera", "rasterio", "rasterio_aligned"]
        if export_dir:
            for name in names:
                saved_paths[name] = str(export_dir / f"{slug}_{name}.tif")
        else:
            for name in names:
                f = tempfile.NamedTemporaryFile(suffix=f"_{name}.tif", delete=False)
                saved_paths[name] = f.name
                f.close()

        accuracy = None
        accuracy_error = None
        consistency = None
        spatial_lines = None
        grid_check = None
        border_check = None
        # rasterio's own output is what gets compared unless a re-run aligns it.
        compare_path = saved_paths["rasterio"]

        try:
            for library in ["rastera", "rasterio"]:
                save_path = saved_paths[library]
                result = run_once(
                    scenario,
                    library,
                    save_array=save_path,
                    cold_cache=args.cold_cache,
                    # rasterio's timed run keeps its own defaults, so timings and
                    # spatial alignment describe each library as it ships; the
                    # matched re-run below is what gets compared.
                    snap_to_grid=snap_to_grid if library == "rastera" else False,
                    use_overviews=use_overviews if library == "rastera" else True,
                )
                if "error" not in result:
                    first_results[library] = result
                    timings[library].append(result["elapsed_s"])
                    mem[library].append(result.get("peak_rss_mb", 0))

            if _needs_matched_rerun(scenario, first_results):
                aligned = run_once(
                    scenario,
                    "rasterio",
                    save_array=saved_paths["rasterio_aligned"],
                    cold_cache=False,  # its timing is discarded, so don't pay for a purge
                    snap_to_grid=snap_to_grid,
                    use_overviews=use_overviews,
                )
                if "error" in aligned:
                    compare_path = None
                    accuracy_error = "the aligned rasterio re-run failed"
                else:
                    compare_path = saved_paths["rasterio_aligned"]
                    grid_check = check_shared_grid(
                        scenario, first_results["rastera"], aligned
                    )

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
            if compare_path:
                try:
                    accuracy = compare_arrays(
                        saved_paths["rastera"],
                        compare_path,
                        by_transform=_lands_on_res_grid(scenario),
                    )
                except Exception as exc:
                    accuracy_error = str(exc) or type(exc).__name__

            # Border sanity check (rastera output)
            try:
                border_check = check_borders(saved_paths["rastera"])
            except Exception:
                pass

            if export_dir:
                export_paths = [
                    saved_paths[name]
                    for name in names
                    if os.path.exists(saved_paths[name])
                ]
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
                    # Same split as the first run: a timing series that changed
                    # configuration halfway through would have no median to take.
                    snap_to_grid=snap_to_grid if library == "rastera" else False,
                    use_overviews=use_overviews if library == "rastera" else True,
                )
                if "error" not in result:
                    timings[library].append(result["elapsed_s"])
                    mem[library].append(result.get("peak_rss_mb", 0))

        # ── Print structured report ──────────────────────────────
        out(f"\n{'=' * 60}")
        out(f"Scenario {scenario_idx}: {scenario['name']}")
        out(f"{'=' * 60}")

        # Overall verdict
        passed, reason = _assess_result(
            scenario,
            accuracy,
            consistency,
            border_check,
            grid_check,
            accuracy_error=accuracy_error,
        )
        out(f"\n  Result: {'✅ AS EXPECTED' if passed else '❌ UNEXPECTED DIFFERENCE'}")
        out(f"  Reason: {reason}")

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

        if spatial_lines:
            out("\n  Spatial alignment:")
            for line in spatial_lines:
                out(line)

        if grid_check:
            out(
                f"\n  Shared grid (rasterio re-run on rastera's terms): "
                f"{'✅' if grid_check['ok'] else '❌'} {grid_check['detail']}"
            )

        if border_check:
            if border_check["ok"]:
                out("\n  Border sanity: ✅ no suspect edges")
            else:
                out("\n  Border sanity:")
                for edge_name, info in border_check["edges"].items():
                    if info["bad"]:
                        out(
                            f"    ❌ {edge_name}: {info['dominant_frac'] * 100:.0f}% "
                            f"is value {info['dominant_value']}"
                        )

        if accuracy:
            out("\n  Accuracy:")
            for line in format_accuracy(accuracy, scenario.get("expect", {})):
                out(line)
        elif accuracy_error:
            out(f"\n  Accuracy: ❌ not compared ({accuracy_error})")

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


def _rastera_bbox(scenario: dict) -> tuple[tuple[float, ...], int]:
    """The bbox and CRS rastera is called with, after any upfront reprojection.

    ``read()`` requires ``bbox_crs == target_crs`` (``rastera/reader.py``), so read
    scenarios flagged ``reproject_bbox`` hand it a bbox already moved into the target
    CRS. ``merge()`` takes the two apart and reprojects the bbox itself, which is the
    case :func:`_rastera_reprojects_bbox` picks out.
    """
    bbox = tuple(float(x) for x in scenario["bbox"].split(","))
    bbox_crs = scenario["bbox_crs"]
    if scenario.get("reproject_bbox") and "target_crs" in scenario:
        target_crs = scenario["target_crs"]
        return _corner_hull(bbox, bbox_crs, target_crs), target_crs
    return bbox, bbox_crs


def check_shared_grid(scenario: dict, ra: dict, rio: dict) -> dict:
    """Whether the re-run really landed on the same grid rastera chose.

    How much that proves depends on the mode. On the merge path rasterio derives the
    grid with its own ``-tap`` (``merge(target_aligned_pixels=True)``), so agreement
    is the suite's only independent evidence that rastera's rounding matches GDAL's.
    On the read path rasterio has no bounds-based tap to call, so the worker
    reimplements the arithmetic (``_worker._tap_grid``) and agreement confirms the
    plumbing, not the formula — ``tests/test_merge.py`` covers that.

    Phase (pixel size, whole-pixel origin offset) and extent must both match, except
    where rastera reprojects the bbox itself: it densifies each edge where the worker
    takes the four corners, so that hull may be a pixel or two larger and only has
    to *contain* rasterio's within ``_HULL_SLACK_PX``.
    """
    from affine import Affine

    t_ra, t_rio = Affine(*ra["transform"]), Affine(*rio["transform"])
    if (t_ra.a, t_ra.e) != (t_rio.a, t_rio.e):
        return {
            "ok": False,
            "detail": f"pixel size {(t_rio.a, t_rio.e)} vs rastera's {(t_ra.a, t_ra.e)}",
        }
    try:
        off = _pixel_offset(t_ra, t_rio)
    except ValueError as exc:
        return {"ok": False, "detail": str(exc)}

    if off == (0, 0) and ra["shape"] == rio["shape"]:
        return {"ok": True, "detail": "identical transform and shape"}
    mismatch = {
        "ok": False,
        "detail": f"same phase but offset {off} px, "
        f"shape {rio['shape']} vs rastera's {ra['shape']}",
    }
    if not _rastera_reprojects_bbox(scenario):
        return mismatch

    off_x, off_y = off
    _, ra_h, ra_w = ra["shape"]
    _, rio_h, rio_w = rio["shape"]
    contained = (
        0 <= off_x <= _HULL_SLACK_PX
        and 0 <= off_y <= _HULL_SLACK_PX
        and off_x + rio_w <= ra_w
        and off_y + rio_h <= ra_h
    )
    if not contained:
        return mismatch | {
            "detail": mismatch["detail"]
            + f" — outside the {_HULL_SLACK_PX}px a densified bbox hull can explain",
        }
    return {
        "ok": True,
        "detail": f"same phase, rastera's grid contains rasterio's with {off} px "
        f"to spare ({ra['shape']} vs {rio['shape']}) — rastera densifies the "
        "bbox hull, the worker takes its corners",
    }


def _needs_matched_rerun(scenario: dict, results: dict) -> bool:
    """Whether rasterio has to be re-run on rastera's terms before comparing pixels.

    Both flags only bite where there is a target resolution — the only branch with a
    grid to snap, and the only one where GDAL is asked to decimate and so gets to
    choose a pyramid level. Deliberately not conditioned on the two grids happening
    to disagree: where the bbox already sits on resolution multiples they agree
    anyway, and skipping the re-run there would leave rasterio on a different
    pyramid level.
    """
    if "rastera" not in results or "rasterio" not in results:
        return False
    if not _lands_on_res_grid(scenario):
        return False
    return scenario.get("snap_to_grid", True) or not scenario.get(
        "use_overviews", False
    )


def _lands_on_res_grid(scenario: dict) -> bool:
    """Whether the output sits on resolution multiples rather than source pixels.

    Without a target resolution both libraries read the source's own pixels, so
    there is no grid to snap rasterio onto and the reported origins are not
    comparable — see :func:`compare_arrays`.
    """
    return "target_resolution" in scenario


def _rastera_reprojects_bbox(scenario: dict) -> bool:
    """Whether rastera, not the harness, moves the bbox into the target CRS."""
    target_crs = scenario.get("target_crs")
    return (
        target_crs is not None
        and target_crs != scenario["bbox_crs"]
        and not scenario.get("reproject_bbox")
    )


def _pixel_offset(t_a, t_b) -> tuple[int, int]:
    """``(cols, rows)`` from *t_a*'s origin to *t_b*'s. Raises if not a whole number.

    Origins a fractional pixel apart cannot be compared by array index at all, so
    this is a hard error rather than something to round away.
    """
    off_x = (t_b.c - t_a.c) / t_a.a
    off_y = (t_b.f - t_a.f) / t_a.e
    if abs(off_x - round(off_x)) > 1e-6 or abs(off_y - round(off_y)) > 1e-6:
        raise ValueError(f"grids are out of phase by ({off_x:.4f}, {off_y:.4f}) px")
    return round(off_x), round(off_y)
