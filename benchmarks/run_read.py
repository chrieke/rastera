"""Read benchmarks: single-file read scenarios.

Usage:
    python benchmarks/run_read.py [--runs 5]
"""

from __future__ import annotations

from run import run_benchmarks

# UTM 33N subset over Rome, inside the B03 tile's footprint.
BBOX = "255804.0,4626619.0,274330.0,4644625.0"

SCENARIOS = [
    {
        "name": "Read: same CRS, native resolution (bbox subset), snapped to raster grid (rastera default)",
        "mode": "read",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "expect": {
            "shape_match": False,
            "note": "snap_to_grid rounds outward: +1 row and shifted bounds vs rasterio; overlapping pixels identical.",
        },
    },
    {
        "name": "Read: same CRS, native resolution (bbox subset), not snapped - raster matches bbox exactly (rasterio default)",
        "mode": "read",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "snap_to_grid": False,
        "expect": {"max_pct_differ": 0, "max_rmse_pct": 0},
    },
    {
        "name": "Read: same CRS, downsampled to 60m, no overviews (both default)",
        "mode": "read",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_resolution": 60.0,
        "expect": {
            "shape_match": False,
            "max_pct_differ": 100,
            "max_rmse_pct": 2,
            "note": "Grid snaps outward onto 60m multiples (bbox is off-grid at "
            "60m): origin shifts vs rasterio's bbox-anchored grid.",
        },
    },
    {
        "name": "Read: same CRS, downsampled to 60m via overviews (rastera)",
        "mode": "read",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_resolution": 60.0,
        "use_overviews": True,
        "expect": {
            "shape_match": False,
            "max_pct_differ": 100,
            "max_rmse_pct": 2,
            "note": "Grid snaps outward onto 60m multiples; rastera reads the "
            "40m COG overview while rasterio downsamples full 10m.",
        },
    },
    {
        "name": "Read: cross-CRS reproject to EPSG:4326, 0.001 deg, no overviews (default)",
        "mode": "read",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_crs": 4326,
        "target_resolution": 0.001,
        "reproject_bbox": True,
        "expect": {
            "shape_match": False,
            "max_pct_differ": 100,
            "max_rmse_pct": 2,
            "note": "Grid snaps outward onto 0.001 deg multiples (reprojected "
            "bbox is off-grid): sub-pixel origin shift moves NN picks.",
        },
    },
    {
        "name": "Read: cross-CRS reproject to EPSG:4326, 0.001 deg, via overviews (rastera)",
        "mode": "read",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_crs": 4326,
        "target_resolution": 0.001,
        "reproject_bbox": True,
        "use_overviews": True,
        "expect": {
            "shape_match": False,
            "max_pct_differ": 100,
            "max_rmse_pct": 5,
            "note": "Grid snaps outward onto 0.001 deg multiples + COG overview "
            "source + coarse-grid warp interpolation.",
        },
    },
]

if __name__ == "__main__":
    run_benchmarks(SCENARIOS)
