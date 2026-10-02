"""rastera on real COGs, read by async-geotiff itself.

Every other offline test stands async-geotiff in with a mock that returns
whatever conftest says, so a change to its Overview objects, window reads,
nodata or transform types would pass them all. These are tiny GDAL-made COGs
and band-interleaved GTiffs with 16x16 blocks, and the references are GDAL's
own reads of them; ``data/make_fixtures.py`` rebuilds both.
"""

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from affine import Affine

import rastera

DATA = Path(__file__).parent / "data"
REF = np.load(DATA / "references.npz")


async def _open(name: str) -> rastera.AsyncGeoTIFF:
    return await rastera.open(DATA / name, cache=False)


def _data(arr: Any) -> np.ndarray[Any, Any]:
    return arr.data[0]


class TestHeader:
    async def test_grid_nodata_and_overviews(self):
        p = (await _open("u16.tif")).profile
        assert (p["width"], p["height"], p["count"]) == (96, 64, 1)
        assert (p["dtype"], p["crs_epsg"], p["nodata"]) == ("uint16", 32632, 0)
        assert p["res"] == (10.0, 20.0)
        assert p["transform"] == Affine(10, 0, 500000, 0, -20, 5001280)
        assert p["overviews"] == [(48, 32), (24, 16), (12, 8)]

    async def test_nan_nodata(self):
        p = (await _open("f32.tif")).profile
        assert p["dtype"] == "float32"
        assert p["nodata"] is not None and math.isnan(p["nodata"])

    async def test_south_up_bounds(self):
        p = (await _open("south_up.tif")).profile
        assert p["bounds"] == rastera.BBox(500000, 5000000, 500320, 5000320)


class TestReads:
    @pytest.mark.parametrize("name", ["u16", "f32", "south_up"])
    async def test_full_read(self, name: str):
        arr = await (await _open(f"{name}.tif")).read()
        np.testing.assert_array_equal(_data(arr), REF[name])

    async def test_bbox_read_is_the_window_it_covers(self):
        ds = await _open("u16.tif")
        arr = await ds.read(bbox=(500100, 5000000, 500300, 5000400), bbox_crs=32632)
        assert arr.transform == Affine(10, 0, 500100, 0, -20, 5000400)
        np.testing.assert_array_equal(_data(arr), REF["u16"][44:, 10:30])

    async def test_window_read(self):
        ds = await _open("u16.tif")
        arr = await ds.read(
            window=rastera.Window(col_off=5, row_off=3, width=17, height=9)
        )
        np.testing.assert_array_equal(_data(arr), REF["u16"][3:12, 5:22])

    async def test_south_up_bbox_read_keeps_the_files_orientation(self):
        ds = await _open("south_up.tif")
        arr = await ds.read(bbox=(500050, 5000100, 500150, 5000200), bbox_crs=32632)
        assert arr.transform == Affine(10, 0, 500050, 0, 10, 5000100)
        np.testing.assert_array_equal(_data(arr), REF["south_up"][10:20, 5:15])

    @pytest.mark.parametrize("name", ["band_pred2", "band_pred3"])
    async def test_band_interleaved_full_read(self, name: str):
        """async-tiff 0.7.2 undoes the predictor across the bands of a tile
        rather than per band; ``rastera.predictor`` redoes it."""
        arr = await (await _open(f"{name}.tif")).read()
        np.testing.assert_array_equal(arr.data, REF[name])  # type: ignore[reportUnknownMemberType]

    async def test_band_interleaved_window_read_across_edge_tiles(self):
        ds = await _open("band_pred2.tif")
        arr = await ds.read(
            window=rastera.Window(col_off=5, row_off=3, width=35, height=21)
        )
        np.testing.assert_array_equal(arr.data, REF["band_pred2"][:, 3:, 5:])  # type: ignore[reportUnknownMemberType]

    async def test_an_overview_level_holds_gdals_pixels(self):
        """Level 1 is 20x40 m: the coarsest that fits a 40 m read on both axes."""
        ds = await _open("u16.tif")
        level = ds._best_overview_for_resolution((40.0, 40.0))
        assert level is not None and (level.width, level.height) == (48, 32)
        arr = await ds._read_native(overview=level)
        assert arr.transform == Affine(20, 0, 500000, 0, -40, 5001280)
        np.testing.assert_array_equal(_data(arr), REF["u16_level1"])

    async def test_resampled_read_matches_gdalwarp(self):
        ds = await _open("f32.tif")
        arr = await ds.read(
            bbox=(400000, 6000000, 400640, 6000480),
            bbox_crs=32633,
            target_resolution=20,
        )
        np.testing.assert_array_equal(_data(arr), REF["f32_20m"])

    async def test_a_reprojecting_read_takes_gdalwarps_resolution(self):
        """It was 0.00009 degrees, 2.6x gdalwarp's pixels for the full read
        and 2.9x for the bbox."""
        ds = await _open("u16.tif")
        # gdalwarp -t_srs EPSG:4326 u16.tif
        full = await ds.read(target_crs=4326)
        assert full.transform.a == pytest.approx(0.0001455295488904, rel=1e-12)
        # gdalwarp -t_srs EPSG:4326 -te <bbox> u16.tif makes 65 columns: it
        # fits its resolution to a whole number of them, where rastera rounds
        # the grid out.
        bbox = (9.001, 45.155, 9.011, 45.164)
        part = await ds.read(bbox=bbox, bbox_crs=4326, target_crs=4326)
        assert round((bbox[2] - bbox[0]) / part.transform.a) == 65

    async def test_merge_matches_gdalbuildvrt(self):
        """gdalbuildvrt takes each pixel from the last file valid there, which
        is mosaic_method="last"."""
        a, b = await rastera.open(
            [DATA / "tile_a.tif", DATA / "tile_b.tif"], cache=False
        )
        arr = await rastera.merge(
            [a, b],
            bbox=(600000, 5100000, 600560, 5100320),
            bbox_crs=32632,
            target_resolution=10,
            mosaic_method="last",
        )
        np.testing.assert_array_equal(_data(arr), REF["mosaic_last"])


class TestLocalPaths:
    """async-tiff percent-encodes the key it passes its own store, and that
    store looked for the encoded name on disk: ``ü`` as ``%C3%BC``."""

    @pytest.mark.parametrize("name", ["ü", "a~b", "a%b", "a#b", "a?b", "a[b]"])
    async def test_special_characters_in_folder_and_file_names(
        self, name: str, tmp_path: Path
    ):
        f = tmp_path / name / f"{name}.tif"
        f.parent.mkdir()
        shutil.copy(DATA / "u16.tif", f)
        arr = await (await rastera.open(f, cache=False)).read()
        np.testing.assert_array_equal(_data(arr), REF["u16"])

    async def test_a_percent_escape_in_a_name_is_read_as_written(self, tmp_path: Path):
        """x%20y.tif was looked for as x%2520y.tif, so that file was read."""
        shutil.copy(DATA / "u16.tif", tmp_path / "x%20y.tif")
        shutil.copy(DATA / "f32.tif", tmp_path / "x%2520y.tif")
        ds = await rastera.open(tmp_path / "x%20y.tif", cache=False)
        assert ds.profile["dtype"] == "uint16"

    async def test_a_list_across_folders_merges(self, tmp_path: Path):
        """The list shares one store at the filesystem root, so the folder
        names are part of every key."""
        a, b = tmp_path / "ü" / "tile_a.tif", tmp_path / "a#b" / "tile_b.tif"
        for f in (a, b):
            f.parent.mkdir()
            shutil.copy(DATA / f.name, f)
        srcs = await rastera.open([a, b], cache=False)
        arr = await rastera.merge(
            srcs,
            bbox=(600000, 5100000, 600560, 5100320),
            bbox_crs=32632,
            target_resolution=10,
            mosaic_method="last",
        )
        np.testing.assert_array_equal(_data(arr), REF["mosaic_last"])
