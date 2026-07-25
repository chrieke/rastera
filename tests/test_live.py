from typing import Any, cast

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


# ── VRT → DIMAP → tile TIFF mosaic ──────────────────────────────────────────
#
# End-to-end regression for the issue that motivated DIMAP support: the VRT
# points to DIMAP .XML descriptors, which each fan out to up to 12 TIFF
# tiles across two band-groups. Exercises VRT dispatch, the .xml detection
# hook, lazy tile opens, and the mosaic stitcher all at once.
#
# The subject is read from the local-only catalog outside the repo (see
# tests/vrt_catalog.py) and selected by shape rather than name, so nothing
# environment-specific is committed. Skips without it. See
# tests/test_vrt_live.py for the same pattern with GDAL as a pixel oracle.


@live
@pytest.mark.asyncio
async def test_multitile_vrt_open_and_window_read():
    from async_geotiff import Window

    fx = _large_multitile_vrt()
    src = cast(Any, await rastera.open(fx.s3_uri, skip_signature=False))
    assert src.count == fx.bands

    # Small windowed read to avoid downloading the whole product. Offsets are
    # arbitrary but well inside the scene and past the first tile row/column,
    # so the 512px window straddles seams in both axes.
    arr = await src.read(
        window=Window(
            col_off=fx.width // 3, row_off=fx.height // 3, width=512, height=512
        ),
        band_indices=[1, 2, 3, 4],
    )
    data: Any = arr.data
    assert data.shape == (4, 512, 512)
    assert str(data.dtype) == fx.dtype
    # Data shouldn't be all-nodata in the middle of the scene.
    assert data.mean() != 0


@live
@pytest.mark.asyncio
async def test_multitile_vrt_reproject_to_wgs84():
    """Stitch-then-reproject: proves the mosaic survives the read-path
    reprojection wrapper unchanged."""
    import numpy as np

    src = cast(
        Any, await rastera.open(_large_multitile_vrt().s3_uri, skip_signature=False)
    )
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
    data: Any = arr.data
    assert data.ndim == 3 and data.shape[0] == 3
    assert data.dtype == np.uint16


# ── helpers ─────────────────────────────────────────────────────────────────


def _large_multitile_vrt() -> Any:
    """The largest multi-tile 6-band band-stack VRT in the local catalog.

    Selected by shape, not by name: >20k px on both axes means a descriptor
    that mosaics a tile grid per band-group, which is what makes it a stitcher
    test. ``uint16`` picks the plain band-stack flavour over its uint8
    LUT/processed sibling, which shares the band count and footprint but
    exercises a different read path (covered by tests/test_vrt_live.py).

    Largest, not smallest as elsewhere: the reads below are a fixed 512px
    window either way, so size costs nothing here, and more tiles per
    band-group is a stronger test of the stitcher. It also keeps the window
    off nodata — these are rotated orthorectified footprints, and the smaller
    candidate's one-third point falls in a black corner.

    Skips when the local-only catalog is absent.
    """
    from tests import vrt_catalog

    fixtures = vrt_catalog.load()
    candidates = [
        f
        for f in fixtures.SIX_BAND_VRTS
        if f.dtype == "uint16" and f.width > 20000 and f.height > 20000
    ]
    if not candidates:
        pytest.skip("no large multi-tile 6-band VRT in the local catalog")
    return max(candidates, key=lambda f: f.megapixels)
