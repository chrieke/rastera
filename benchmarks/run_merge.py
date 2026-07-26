"""Merge benchmarks: multi-file merge scenarios.

Usage:
    python benchmarks/run_merge.py [--runs 5]
"""

from __future__ import annotations

from run import URI, URI_32TQM, run_benchmarks

SCENARIOS = [
    {
        "name": "Merge: 2 tiles, same CRS, 10m resolution, snapped to raster grid (rastera default)",
        "mode": "merge",
        "bbox": "283838.0,4629464.7,326626.2,4648263.2",
        "bbox_crs": 32633,
        "target_crs": 32633,
        "target_resolution": 10.0,
        "expect": {
            "shape_match": False,
            "max_pct_differ": 100,
            "max_rmse_pct": 2,
            "note": "Snapping shifts origin ~0.8 px and rounds outward: +1 row/col vs rasterio.",
        },
    },
    {
        "name": "Merge: 2 tiles, same CRS, 10m resolution, not snapped - raster matches bbox exactly (rasterio default)",
        "mode": "merge",
        "bbox": "283838.0,4629464.7,326626.2,4648263.2",
        "bbox_crs": 32633,
        "target_crs": 32633,
        "target_resolution": 10.0,
        "snap_to_grid": False,
        "expect": {"max_pct_differ": 0, "max_rmse_pct": 0},
    },
    {
        "name": "Merge: 2 tiles, same CRS, downsampled to 60m, no overviews (default)",
        "mode": "merge",
        "bbox": "283838.0,4629464.7,326626.2,4648263.2",
        "bbox_crs": 32633,
        "target_crs": 32633,
        "target_resolution": 60.0,
        "expect": {
            "max_pct_differ": 100,
            "max_rmse_pct": 2,
            "note": "Snapping + 6x downsample: snapped origin + rasterio ratio drift (~6.005 vs 6.0).",
        },
    },
    {
        "name": "Merge: 2 tiles, same CRS, downsampled to 60m, via overviews (rastera)",
        "mode": "merge",
        "bbox": "283838.0,4629464.7,326626.2,4648263.2",
        "bbox_crs": 32633,
        "target_crs": 32633,
        "target_resolution": 60.0,
        "use_overviews": True,
        "expect": {
            "max_pct_differ": 100,
            "max_rmse_pct": 2,
            "note": "Snapping + 6x downsample + overview source: snapped origin + rasterio ratio drift.",
        },
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
        "expect": {
            "max_pct_differ": 100,
            "max_rmse_pct": 2,
            "note": "Snapping + cross-CRS warp: different warp math shifts most pixels.",
        },
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
        "expect": {
            "max_pct_differ": 100,
            "max_rmse_pct": 2,
            "note": "Snapping + cross-CRS warp: different warp math shifts most pixels.",
        },
    },
    {
        "name": "Merge: 2 tiles, cross-CRS (32632+32633), reproject to 4326, 0.001 deg, via overviews (rastera)",
        "mode": "merge",
        "uri": URI,
        "uri2": URI_32TQM,
        "bbox": "11.8,41.7,12.5,42.2",
        "bbox_crs": 4326,
        "target_crs": 4326,
        "target_resolution": 0.001,
        "use_overviews": True,
        "expect": {
            "max_pct_differ": 100,
            "max_rmse_pct": 5,
            "note": "Snapping + cross-CRS warp + overview source.",
        },
    },
]

if __name__ == "__main__":
    run_benchmarks(SCENARIOS)
