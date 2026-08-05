"""Merge benchmarks: multi-file merge scenarios.

Usage:
    python benchmarks/run_merge.py [--runs 5]
"""

from __future__ import annotations

from run import URI, URI_32TQM, run_benchmarks

# UTM 33N strip spanning the seam between the two tiles.
BBOX = "283838.0,4629464.7,326626.2,4648263.2"

SCENARIOS = [
    {
        "name": "Merge: 2 tiles, same CRS, 10m resolution, snapped to raster grid (rastera default)",
        "mode": "merge",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_crs": 32633,
        "target_resolution": 10.0,
        "expect": {
            "shape_match": False,
            "note": "Snapping shifts origin ~0.8 px and rounds outward: +1 row/col "
            "vs rasterio. On one grid both tiles and the seam match exactly.",
        },
    },
    {
        "name": "Merge: 2 tiles, same CRS, 10m resolution, not snapped - raster matches bbox exactly (rasterio default)",
        "mode": "merge",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_crs": 32633,
        "target_resolution": 10.0,
        "snap_to_grid": False,
        "expect": {"max_pct_differ": 0, "max_rmse_pct": 0},
    },
    {
        "name": "Merge: 2 tiles, same CRS, downsampled to 60m, no overviews (default)",
        "mode": "merge",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_crs": 32633,
        "target_resolution": 60.0,
        "expect": {
            "shape_match": False,
            "note": "Grid snaps outward onto 60m multiples (bbox is off-grid at "
            "60m, so +1-2 px per axis vs rasterio). On one grid, and with GDAL "
            "held to full resolution like rastera, the decimation is identical — "
            "left to itself GDAL auto-selects the 40m overview here.",
        },
    },
    {
        "name": "Merge: 2 tiles, same CRS, downsampled to 60m, via overviews (rastera)",
        "mode": "merge",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_crs": 32633,
        "target_resolution": 60.0,
        "use_overviews": True,
        "expect": {
            "shape_match": False,
            "note": "Grid snaps outward onto 60m multiples. Both sides read the "
            "same 40m overview, so on one grid the pixels are identical.",
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
            "shape_match": False,
            "note": "Grid snaps outward onto 10m multiples (transformed bbox is "
            "off-grid). Both warps agree pixel for pixel on one grid, across the "
            "32632/32633 zone boundary included.",
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
            "max_pct_differ": 5,
            "max_rmse_pct": 0.5,
            "note": "The bbox is already on 0.001 deg multiples, so both grids "
            "agree without a re-run. ~3% of pixels differ where the coarse-grid "
            "warp interpolates coordinates across a nearest-neighbour boundary. "
            "rasterio is the slower side because each WarpedVRT carries the final "
            "grid and so warps from full 10m in one step.",
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
            "max_rmse_pct": 2,
            "note": "rastera warps from the 40m overview while rasterio warps from "
            "full 10m — a pinned-grid WarpedVRT leaves GDAL no window to pick an "
            "overview for. Nearly every pixel is therefore a different source "
            "sample and the percentage carries no signal; RMSE is the check.",
        },
    },
]

if __name__ == "__main__":
    run_benchmarks(SCENARIOS)
