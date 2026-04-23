from typing import Any

import pytest

import rastera

live = pytest.mark.live

URI = "s3://e84-earth-search-sentinel-data/sentinel-2-c1-l2a/33/T/TG/2025/7/S2B_T33TTG_20250703T100029_L2A/B03.tif"
BBOX = (255804.0, 4626619.0, 274330.0, 4644625.0)  # UTM subset over Rome


@pytest.mark.skip(
    reason="Downloads full 10980x10980 image (~230MB), too slow for routine use"
)
@live
@pytest.mark.asyncio
async def test_read_full_image():
    src = await rastera.open(URI)

    raster_array = await src.read()

    data: Any = raster_array.data  # type: ignore[reportUnknownMemberType]
    assert data.ndim == 3
    assert data.shape[0] >= 1
    assert data.shape[1] > 0
    assert data.shape[2] > 0
    assert data.mean() != 0


@live
@pytest.mark.asyncio
async def test_read_bbox():
    src = await rastera.open(URI)

    raster_array = await src.read(bbox=BBOX, bbox_crs=32633)

    data: Any = raster_array.data  # type: ignore[reportUnknownMemberType]
    assert data.ndim == 3
    assert data.shape[0] >= 1
    # Should be a subset, not the full 10980x10980
    assert data.shape[1] < 10980
    assert data.shape[2] < 10980
    assert raster_array.width == data.shape[2]
    assert raster_array.height == data.shape[1]
    assert data.mean() != 0


# ── Airbus PNEO VRT → DIMAP → tile TIFF mosaic ─────────────────────────────
#
# End-to-end regression for the issue that motivated DIMAP support: the VRT
# points to DIMAP .XML descriptors, which each fan out to up to 12 TIFF
# tiles across two band-groups. Exercises VRT dispatch, the .xml detection
# hook, lazy tile opens, and the mosaic stitcher all at once.
PNEO_VRT = (
    "s3://terralabs-ingestion-airbus-licensed/airbus/"
    "019d77ad-c11c-718c-a183-960cb78785b0/PNEO/tasking/"
    "fafe3413-8cce-4fda-ba33-c7fb4a9bf7ba/vrt/"
    "REFLECTANCE_VRT_PNEO4_STD_202604101033278_PMS-FS_ORT_PWOI_000534689_1_2_F_1.vrt"
)


@live
@pytest.mark.asyncio
async def test_pneo_vrt_open_and_window_read():
    import numpy as np

    src = await rastera.open(PNEO_VRT, skip_signature=False)
    # PNEO reflectance = RGB + NED = 6 bands.
    assert src.count == 6

    # Small windowed read to avoid downloading the whole product; the
    # exact offsets are arbitrary but chosen to straddle a tile seam.
    from async_geotiff import Window
    arr = await src.read(
        window=Window(col_off=15000, row_off=20000, width=512, height=512),
        band_indices=[1, 2, 3, 4],
    )
    data: Any = arr.data  # type: ignore[reportUnknownMemberType]
    assert data.shape == (4, 512, 512)
    assert data.dtype == np.uint16
    # Reflectance data shouldn't be all-nodata in the middle of the scene.
    assert data.mean() != 0


@live
@pytest.mark.asyncio
async def test_pneo_vrt_reproject_to_wgs84():
    """Stitch-then-reproject: proves the mosaic survives the read-path
    reprojection wrapper unchanged."""
    import numpy as np

    src = await rastera.open(PNEO_VRT, skip_signature=False)
    # ~100m at this latitude; tiny AOI inside the product's footprint.
    bounds = src._geotiff.bounds  # native UTM
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    half = 50.0  # 100m square in native CRS
    native_bbox = (cx - half, cy - half, cx + half, cy + half)

    from rastera.geo import BBox, transform_bbox
    wgs_bbox = transform_bbox(BBox(*native_bbox), src._crs_epsg, 4326)

    arr = await src.read(
        bbox=tuple(wgs_bbox),
        bbox_crs=4326,
        target_crs=4326,
        target_resolution=1e-5,  # ~1m
        band_indices=[1, 2, 3],
    )
    data: Any = arr.data  # type: ignore[reportUnknownMemberType]
    assert data.ndim == 3 and data.shape[0] == 3
    assert data.dtype == np.uint16
