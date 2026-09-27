"""Rebuild the COG fixtures of ``tests/test_real_files.py``, and GDAL's reads
of them that the tests compare against.

Needs the GDAL command line tools on PATH; the tests themselves do not.

    uv run python tests/data/make_fixtures.py
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent

# GDAL's ENVI data type codes.
_ENVI_TYPES = {np.dtype("u1"): 1, np.dtype("u2"): 12, np.dtype("f4"): 4}


def main() -> None:
    u16 = np.arange(1, 64 * 96 + 1, dtype=np.uint16).reshape(64, 96)
    u16[:8, :8] = 0  # a nodata block
    yy, xx = np.mgrid[0:48, 0:64]
    f32 = (100 + 50 * np.sin(xx / 5) * np.cos(yy / 7)).astype(np.float32)
    f32[20:30, 40:50] = np.nan
    south = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    rows = np.arange(32, dtype=np.uint8)
    tile_a = np.repeat(100 + rows[:, None], 32, axis=1)
    tile_b = np.repeat(200 + rows[None, :], 32, axis=0)
    tile_b[:16, :4] = 0  # nodata where the two tiles overlap

    # name: pixels, EPSG code, GDAL geotransform, nodata.
    fixtures: dict[str, tuple[np.ndarray[Any, Any], int, tuple[int, ...], str]] = {
        # Non-square 10x20 m pixels and three overview levels.
        "u16": (u16, 32632, (500000, 10, 0, 5001280, 0, -20), "0"),
        "f32": (f32, 32633, (400000, 10, 0, 6000480, 0, -10), "nan"),
        # A positive pixel height: rows run north.
        "south_up": (south, 32632, (500000, 10, 0, 5000000, 0, 10), "none"),
        # Overlapping by 8 columns.
        "tile_a": (tile_a, 32632, (600000, 10, 0, 5100320, 0, -10), "0"),
        "tile_b": (tile_b, 32632, (600240, 10, 0, 5100320, 0, -10), "0"),
    }

    refs: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, (pixels, epsg, geotransform, nodata) in fixtures.items():
            _write_cog(work, name, pixels, epsg, geotransform, nodata)
            refs[name] = _gdal(work / name, "gdal_translate", HERE / f"{name}.tif")

        refs["u16_level1"] = _gdal(
            work / "u16_level1", "gdal_translate", HERE / "u16.tif", "-ovr", "0"
        )
        warp = "-r near -tr 20 20 -te 400000 6000000 400640 6000480 -ovr NONE"
        refs["f32_20m"] = _gdal(
            work / "f32_20m", "gdalwarp", HERE / "f32.tif", *warp.split()
        )
        # gdalbuildvrt takes each pixel from the last file that is valid there.
        vrt = work / "ab.vrt"
        tiles = [str(HERE / "tile_a.tif"), str(HERE / "tile_b.tif")]
        subprocess.run(["gdalbuildvrt", "-q", str(vrt), *tiles], check=True)
        refs["mosaic_last"] = _gdal(work / "mosaic", "gdal_translate", vrt)

    np.savez_compressed(HERE / "references.npz", **refs)


def _write_cog(
    work: Path,
    name: str,
    pixels: np.ndarray[Any, Any],
    epsg: int,
    geotransform: tuple[int, ...],
    nodata: str,
) -> None:
    """Write *pixels* as a COG with 16x16 blocks, so a tiny file still has
    several tiles. GDAL warns that COG blocks should be 128 or more, and then
    honours the size anyway."""
    raw = _envi(pixels, work / f"{name}_src.raw")
    options = (
        f"-q -of COG -a_srs EPSG:{epsg} -a_nodata {nodata} "
        f"-co BLOCKSIZE=16 -co COMPRESS=DEFLATE -co PREDICTOR=YES -a_gt"
    ).split() + [str(v) for v in geotransform]
    out = str(HERE / f"{name}.tif")
    command = ["gdal_translate", *options, str(raw), out]
    subprocess.run(command, check=True, stderr=subprocess.DEVNULL)


def _gdal(out: Path, tool: str, src: Path, *options: str) -> np.ndarray[Any, Any]:
    """*src* as GDAL's *tool* reads it, via ENVI so that no TIFF reader is
    involved."""
    raw = out.with_suffix(".raw")
    command = [tool, "-q", "-of", "ENVI", *options, str(src), str(raw)]
    subprocess.run(command, check=True)
    return _load_envi(raw)


def _envi(pixels: np.ndarray[Any, Any], path: Path) -> Path:
    pixels.tofile(path)
    path.with_suffix(".hdr").write_text(
        f"ENVI\nsamples = {pixels.shape[1]}\nlines = {pixels.shape[0]}\n"
        f"bands = 1\nheader offset = 0\nfile type = ENVI Standard\n"
        f"data type = {_ENVI_TYPES[pixels.dtype]}\ninterleave = bsq\n"
        f"byte order = 0\n"
    )
    return path


def _load_envi(path: Path) -> np.ndarray[Any, Any]:
    # GDAL pads some keys: "lines   = 64".
    header = {
        key.strip(): value.strip()
        for key, _, value in (
            line.partition("=")
            for line in path.with_suffix(".hdr").read_text().splitlines()
        )
    }
    dtype = {v: k for k, v in _ENVI_TYPES.items()}[int(header["data type"])]
    shape = (int(header["lines"]), int(header["samples"]))
    return np.fromfile(path, dtype=dtype).reshape(shape)


if __name__ == "__main__":
    main()
