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
            "note": "Grid snaps outward onto 60m multiples (bbox is off-grid at "
            "60m): origin shifts vs rasterio's bbox-anchored grid. On one grid "
            "the decimation is identical.",
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
            "max_rmse_pct": 3,
            "note": "Grid snaps outward onto 60m multiples; rastera reads the 40m "
            "COG overview while rasterio's pinned-transform WarpedVRT warps from "
            "full 10m. Nearly every pixel is a different source sample, so the "
            "percentage carries no signal; RMSE is the check.",
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
            "max_pct_differ": 5,
            "max_rmse_pct": 1,
            "note": "Grid snaps outward onto 0.001 deg multiples (reprojected bbox "
            "is off-grid). ~2% of pixels still differ on a shared grid: the "
            "coarse-grid warp (step=16) interpolates coords, moving some picks "
            "across a nearest-neighbour boundary.",
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
            "max_rmse_pct": 4,
            "note": "Grid snaps outward onto 0.001 deg multiples + COG overview "
            "source + coarse-grid warp interpolation. As in scenario 4 the "
            "percentage carries no signal; RMSE is the check.",
        },
    },
]

if __name__ == "__main__":
    run_benchmarks(SCENARIOS)
