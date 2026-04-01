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

    assert raster_array.data.ndim == 3
    assert raster_array.data.shape[0] >= 1
    assert raster_array.data.shape[1] > 0
    assert raster_array.data.shape[2] > 0
    assert raster_array.data.mean() != 0


@live
@pytest.mark.asyncio
async def test_read_bbox():
    src = await rastera.open(URI)

    raster_array = await src.read(bbox=BBOX, bbox_crs=32633)

    assert raster_array.data.ndim == 3
    assert raster_array.data.shape[0] >= 1
    # Should be a subset, not the full 10980x10980
    assert raster_array.data.shape[1] < 10980
    assert raster_array.data.shape[2] < 10980
    assert raster_array.width == raster_array.data.shape[2]
    assert raster_array.height == raster_array.data.shape[1]
    assert raster_array.data.mean() != 0
