"""Read benchmarks: single-file read scenarios.

Usage:
    python benchmarks/run_read.py [--runs 5]
"""

from __future__ import annotations

from pathlib import Path

from run import run_benchmarks

# UTM 33N subset over Rome, inside the B03 tile's footprint.
BBOX = "255804.0,4626619.0,274330.0,4644625.0"

# Synthetic 10-band, band-interleaved files with a predictor, written on first
# run under the gitignored data dir.
_DATA = Path(__file__).parent / "data"
BAND_PRED2 = str(_DATA / "band_interleaved_int16_pred2.tif")
BAND_PRED3 = str(_DATA / "band_interleaved_float32_pred3.tif")
# 1800x1800 px, starting and ending mid-tile.
BAND_BBOX = "501000.0,5001480.0,519000.0,5019480.0"

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
    {
        "name": "Read: same CRS, downsampled to 60m, bilinear, no overviews",
        "mode": "read",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_resolution": 60.0,
        "resampling": "bilinear",
        "expect": {
            "shape_match": False,
            "max_pct_differ": 0.1,
            "max_rmse_pct": 0.01,
            "note": "Grid snaps outward onto 60m multiples, as in scenario 3. On "
            "one grid, with GDAL held to full resolution, the widened bilinear "
            "kernel matches gdalwarp but for rounding ties: 8 of 93,310 pixels "
            "differ, by 1 DN.",
        },
    },
    {
        "name": "Read: cross-CRS reproject to EPSG:32632, 30m, cubic, no overviews",
        "mode": "read",
        "bbox": BBOX,
        "bbox_crs": 32633,
        "target_crs": 32632,
        "target_resolution": 30.0,
        "reproject_bbox": True,
        "resampling": "cubic",
        "expect": {
            "shape_match": False,
            "max_pct_differ": 50,
            "max_rmse_pct": 0.1,
            "note": "Grid snaps outward onto 30m multiples (reprojected bbox is "
            "off-grid). The two UTM grids are about 4 degrees apart, so this "
            "covers the rotated kernel footprint and the single-pass warp. About "
            "a third of pixels differ, by up to ~60 DN at sharp edges: GDAL's "
            "approximate transformer places source coordinates up to 0.125 px "
            "off by default. Against gdalwarp -et 0, rastera is within 4 DN "
            "(RMSE 0.24). RMSE is the check.",
        },
    },
    {
        "name": "Read: band-interleaved, 10 bands int16, predictor 2, native resolution (synthetic, local)",
        "mode": "read",
        "uri": BAND_PRED2,
        "bbox": BAND_BBOX,
        "bbox_crs": 32632,
        "expect": {"max_pct_differ": 0, "max_rmse_pct": 0},
    },
    {
        "name": "Read: band-interleaved, 10 bands float32, predictor 3, native resolution (synthetic, local)",
        "mode": "read",
        "uri": BAND_PRED3,
        "bbox": BAND_BBOX,
        "bbox_crs": 32632,
        "expect": {"max_pct_differ": 0, "max_rmse_pct": 0},
    },
]


def _write_band_interleaved(path: str, dtype: str, predictor: int) -> None:
    """2048x2048 px of random walks along each row, in 256 px tiles."""
    if Path(path).exists():
        return
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    steps = np.random.default_rng(0).integers(-20, 20, (10, 2048, 2048))
    data = (np.cumsum(steps, axis=2) + 1000).astype(dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2048,
        height=2048,
        count=10,
        dtype=dtype,
        crs="EPSG:32632",
        transform=from_origin(500000, 5020480, 10, 10),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        interleave="band",
        compress="zstd",
        predictor=predictor,
    ) as dst:
        dst.write(data)


if __name__ == "__main__":
    _write_band_interleaved(BAND_PRED2, "int16", 2)
    _write_band_interleaved(BAND_PRED3, "float32", 3)
    run_benchmarks(SCENARIOS)
