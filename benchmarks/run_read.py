"""Read benchmarks: single-file read scenarios.

Usage:
    python benchmarks/run_read.py [--runs 5]
"""

from __future__ import annotations

from run import run_benchmarks

SCENARIOS = [
    {
        "name": "Read: same CRS, native resolution (bbox subset), snapped to raster grid (rastera default)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "expect": {
            "shape_match": False,
            "note": "snap_to_grid rounds outward: +1 row and shifted bounds vs rasterio; overlapping pixels identical.",
        },
    },
    {
        "name": "Read: same CRS, native resolution (bbox subset), not snapped - raster matches bbox exactly (rasterio default)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "snap_to_grid": False,
        "expect": {"max_pct_differ": 0, "max_rmse_pct": 0},
    },
    {
        "name": "Read: same CRS, downsampled to 60m, no overviews (both default)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "target_resolution": 60.0,
        "expect": {"max_pct_differ": 0, "max_rmse_pct": 0},
    },
    {
        "name": "Read: same CRS, downsampled to 60m via overviews (rastera)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "target_resolution": 60.0,
        "use_overviews": True,
        "expect": {
            "max_pct_differ": 100,
            "max_rmse_pct": 2,
            "note": "~97% differ: rastera reads 40m COG overview; rasterio reads full 10m and downsamples.",
        },
    },
    {
        "name": "Read: cross-CRS reproject to EPSG:4326, 0.001 deg, no overviews (default)",
        "mode": "read",
        "bbox": "255804.0,4626619.0,274330.0,4644625.0",
        "bbox_crs": 32633,
        "target_crs": 4326,
        "target_resolution": 0.001,
        "reproject_bbox": True,
        "expect": {
            "max_pct_differ": 5,
            "max_rmse_pct": 1,
            "note": "~2% differ: coarse-grid warp (step=16) interpolates coords, shifting some pixels across NN boundaries.",
        },
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
        "expect": {
            "max_pct_differ": 100,
            "max_rmse_pct": 5,
            "note": "~97% differ: COG overview source + coarse-grid warp interpolation.",
        },
    },
]

if __name__ == "__main__":
    run_benchmarks(SCENARIOS)
