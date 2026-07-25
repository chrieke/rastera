"""Live VRT tests: real production VRTs, with GDAL as the oracle.

These are the tests that caught two silent-wrong-pixel bugs the mocked unit
tests could not see (LUT rounding, and an ignored ``<NoDataValue>`` that let
merge composite nodata over real imagery), so they are worth the credentials
they cost.

Three requirements, each producing a clean skip when unmet:

- ``~/.config/rastera/vrt_fixtures.json`` — a catalog of verified production
  VRTs with the metadata GDAL reported for each. It points at private storage,
  so it lives outside the repository entirely: it cannot be committed, and one
  copy serves every worktree. See ``tests/vrt_catalog.py``. Nothing
  environment-specific appears in this file either — fixtures are selected by
  structural property (dataset family, spatial relationship, size), never by
  name, and the assertions compare against the fixture's own recorded values.
- ``gdal_translate`` / ``gdalbuildvrt`` on PATH — the oracle. Every pixel
  assertion is "rastera returns exactly what GDAL returns for these pixels".
- credentials both rastera and GDAL can see, e.g.
  ``eval "$(aws configure export-credentials --format env)"``.

Bandwidth is bounded on purpose: reads are 128x128 windows well inside each
raster, and the fixtures used are the smallest of each family. Nothing here
reads a whole raster — some fixtures are gigabytes.

Every window is probed for imagery before it is compared (see
``_data_window``). These are rotated orthorectified footprints, so a window
picked by arithmetic alone can be entirely nodata, and "rastera equals GDAL"
over two blocks of zeros asserts nothing. Probing costs at most a handful of
extra 128x128 reads per test, which is why it is affordable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from async_geotiff import Window

import rastera
from tests import vrt_catalog

live = pytest.mark.live

# Module scope: the parametrize lists below are built at import time, so a
# missing catalog has to skip the whole module at collection.
fixtures = vrt_catalog.load()

pytestmark = [
    live,
    pytest.mark.skipif(
        not shutil.which("gdal_translate") or not shutil.which("gdalbuildvrt"),
        reason="GDAL CLI (the oracle) not on PATH",
    ),
]

WIN = 128

# One cheap fixture per structural family, smallest first. Families differ in
# what they exercise: a LUT/processed dataset, a band-stack over a multi-tile
# descriptor, a band-stack over two plain COGs.
_BY_FAMILY: dict[str, Any] = {}
for _fx in sorted(fixtures.ALL_VRT_FIXTURES, key=lambda f: f.megapixels):
    _BY_FAMILY.setdefault(_fx.family, _fx)
CHEAPEST_PER_FAMILY = tuple(_BY_FAMILY.values())

# The LUT path specifically: its output is a rounded interpolation, so it is
# the one family where an off-by-one rounding rule shows up as wrong pixels.
# The cheapest LUT fixture is also its family's representative above, so
# test_reads_match_gdal already compares it against GDAL on the identical
# window — excluding it keeps this from re-running that download verbatim.
LUT_FIXTURES = tuple(
    f
    for f in sorted(fixtures.LUT_PROCESSED_DATASET_VRTS, key=lambda f: f.megapixels)
    if f not in CHEAPEST_PER_FAMILY
)[:2]

# Fixtures whose VRT declares no nodata at all — the majority shape for one
# family, so nodata handling must not assume a value is present.
NODATA_ABSENT = tuple(fixtures.NODATA_ABSENT_VRTS)[:1]


def _ids(fxs: tuple[Any, ...]) -> list[str]:
    # Positional ids: the catalog's family names are environment-specific and
    # this file stays free of them, output included.
    return [f"family{i}" for i in range(len(fxs))]


@pytest.mark.parametrize("fx", CHEAPEST_PER_FAMILY, ids=_ids(CHEAPEST_PER_FAMILY))
@pytest.mark.asyncio
async def test_metadata_matches_gdal(fx: Any):
    """rastera's view of the header must equal GDAL's, recorded in the fixture."""
    ds = cast(Any, await rastera.open(fx.s3_uri, skip_signature=False))
    gt = ds._geotiff

    assert ds.count == fx.bands
    assert str(np.dtype(gt.dtype)) == fx.dtype
    assert (gt.width, gt.height) == (fx.width, fx.height)
    assert ds._crs_epsg == fx.crs_epsg
    assert gt.res[0] == pytest.approx(fx.resolution)  # type: ignore[reportUnknownMemberType]
    # 0.3 m pixels give bounds that are not exactly representable in binary
    # floating point, so allow a hundredth of a pixel.
    assert tuple(gt.bounds) == pytest.approx(  # type: ignore[reportUnknownMemberType]
        fx.bounds, abs=fx.resolution / 100
    )
    # The VRT's own <NoDataValue> is honoured, so this holds even when the
    # underlying source declares nothing.
    assert (None if ds._nodata is None else int(ds._nodata)) == fx.nodata


@pytest.mark.parametrize("fx", CHEAPEST_PER_FAMILY, ids=_ids(CHEAPEST_PER_FAMILY))
@pytest.mark.asyncio
async def test_reads_match_gdal(fx: Any, tmp_path: Path):
    """Window read, bbox read, and band-subset read, all against GDAL."""
    ds = cast(Any, await rastera.open(fx.s3_uri, skip_signature=False))
    win = await _data_window(ds, fx)

    # _data_window has already established this window is not uniform nodata,
    # so none of the comparisons below can pass trivially.
    got = np.asarray((await ds.read(window=win)).data)
    truth = _gdal_srcwin(fx.vsis3_uri, win, tmp_path)
    assert got.shape == truth.shape
    np.testing.assert_array_equal(got, truth)

    # The same pixels addressed in world coordinates must land on the same
    # block: rastera's bbox snapping has to agree with GDAL's -projwin.
    bbox = _window_bbox(ds, win)
    bbox_got = np.asarray((await ds.read(bbox=bbox, bbox_crs=fx.crs_epsg)).data)
    np.testing.assert_array_equal(bbox_got, _gdal_projwin(fx.vsis3_uri, bbox, tmp_path))

    # A single-band read must reproduce that band of the stack — this is where
    # a band-to-source mapping error would surface.
    last = np.asarray((await ds.read(window=win, band_indices=[fx.bands])).data)
    np.testing.assert_array_equal(last[0], truth[fx.bands - 1])


@pytest.mark.parametrize(
    "fx", LUT_FIXTURES, ids=[f"lut{i}" for i in range(len(LUT_FIXTURES))]
)
@pytest.mark.asyncio
async def test_lut_output_matches_gdal_exactly(fx: Any, tmp_path: Path):
    """Regression guard for LUT rounding.

    GDAL's LUT returns a double and its Float64 -> Byte conversion rounds half
    away from zero. numpy's ``rint`` rounds half to even, which sent ~3% of
    pixels one DN low on real display products — invisible to a mocked test,
    since it only shows up on control-point tables whose output steps by 1
    across an even input span.

    Each LUT ships its own control-point table, so this covers tables
    ``test_reads_match_gdal`` never sees; that test already covers the
    cheapest one.
    """
    ds = cast(Any, await rastera.open(fx.s3_uri, skip_signature=False))
    win = await _data_window(ds, fx)

    got = np.asarray((await ds.read(window=win)).data)
    truth = _gdal_srcwin(fx.vsis3_uri, win, tmp_path)
    np.testing.assert_array_equal(got, truth)


@pytest.mark.parametrize("fx", NODATA_ABSENT, ids=["nodata_absent"])
@pytest.mark.asyncio
async def test_nodata_absent_reads_match_gdal(fx: Any, tmp_path: Path):
    """A VRT that declares no nodata must report None, not a guessed 0."""
    ds = cast(Any, await rastera.open(fx.s3_uri, skip_signature=False))
    assert ds._nodata is None

    win = await _data_window(ds, fx)
    got = np.asarray((await ds.read(window=win)).data)
    np.testing.assert_array_equal(got, _gdal_srcwin(fx.vsis3_uri, win, tmp_path))


@pytest.mark.asyncio
async def test_merge_across_seam_matches_gdal_mosaic(tmp_path: Path):
    """Two edge-adjacent VRTs, merged across the shared edge.

    Zero overlap, so a correct merge is unambiguous and any off-by-one at the
    seam shows up as a duplicated or missing pixel row.
    """
    scenario = _scenario("edge_adjacent")
    a, b = scenario.fixtures[:2]

    # Derive the seam rather than assuming a listing order: whichever way the
    # catalog names them, the shared edge is where one's maxy meets the
    # other's miny. Without this a reordered catalog would silently put the
    # window over one scene's top edge and empty space.
    if a.bounds[3] == b.bounds[1]:
        seam_y = a.bounds[3]
    elif b.bounds[3] == a.bounds[1]:
        seam_y = a.bounds[1]
    else:
        pytest.skip("edge_adjacent scenario is not stacked north-south")

    half = (WIN // 2) * a.resolution
    minx = max(a.bounds[0], b.bounds[0]) + 1000 * a.resolution
    bbox = (minx, seam_y - half, minx + WIN * a.resolution, seam_y + half)

    sources = await rastera.open([a.s3_uri, b.s3_uri], skip_signature=False)
    merged = await rastera.merge(
        sources,
        bbox=bbox,
        bbox_crs=a.crs_epsg,
        target_crs=a.crs_epsg,
        target_resolution=a.resolution,
    )
    data: np.ndarray[Any, Any] = merged.data  # type: ignore[reportUnknownMemberType]
    got = np.asarray(data)

    truth = _gdal_projwin(_gdal_mosaic([a, b], tmp_path), bbox, tmp_path)
    np.testing.assert_array_equal(got, truth)
    # No fully-empty row: the seam must not leave an uncovered line.
    assert not (got == 0).all(axis=(0, 2)).any()


@pytest.mark.asyncio
async def test_merge_overlap_respects_declared_nodata(tmp_path: Path):
    """Regression guard for the VRT's ``<NoDataValue>`` being honoured.

    These scenes are rotated orthorectified footprints, so each has nodata
    corners inside its own bounding box, and they partially overlap. When the
    VRT-declared nodata was ignored, one scene's black corner composited *over*
    the other's real imagery: 82% of this window came back zero where GDAL
    returned pixels.
    """
    scenario = _scenario("partial_overlap", n_fixtures=2)
    a, b = scenario.fixtures

    ds_a, ds_b = cast(
        list[Any], await rastera.open([a.s3_uri, b.s3_uri], skip_signature=False)
    )
    assert ds_a._nodata == a.nodata

    bbox = await _nodata_corner_bbox(ds_a, ds_b, a)

    merged = await rastera.merge(
        [ds_a, ds_b],
        bbox=bbox,
        bbox_crs=a.crs_epsg,
        target_crs=a.crs_epsg,
        target_resolution=a.resolution,
        mosaic_method="first",
    )
    data: np.ndarray[Any, Any] = merged.data  # type: ignore[reportUnknownMemberType]
    got = np.asarray(data)

    # gdalbuildvrt draws later sources on top, skipping their nodata, so
    # listing b then a expresses the same "prefer a, fall back to b" as
    # mosaic_method="first" on [a, b].
    truth = _gdal_projwin(_gdal_mosaic([b, a], tmp_path), bbox, tmp_path)
    np.testing.assert_array_equal(got, truth)
    # The window must actually contain pixels only b can supply, else the
    # comparison says nothing about nodata handling.
    assert (got != 0).mean() > 0.1


# ── helpers ─────────────────────────────────────────────────────────────────


def _scenario(relationship: str, *, n_fixtures: int | None = None) -> Any:
    """The cheapest merge scenario with the given spatial relationship."""
    matches = [
        s
        for s in fixtures.MERGE_SCENARIOS
        if s.relationship == relationship
        and (n_fixtures is None or len(s.fixtures) == n_fixtures)
    ]
    if not matches:
        pytest.skip(f"no {relationship} merge scenario in the local catalog")
    return min(matches, key=lambda s: sum(f.megapixels for f in s.fixtures))


# Offsets to try, as (col, row) fractions of the raster. These are
# orthorectified footprints rotated inside their bounding box, so a fixed
# offset lands on the black corner for some fixtures — one third along both
# axes is all-nodata for one of the display products, and the exact centre is
# empty for another. Ordered centre-outwards, since the middle of a rotated
# footprint is the most likely to hold imagery.
_WINDOW_OFFSETS = (
    (0.5, 0.5),
    (0.5, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.5),
    (1 / 3, 1 / 3),
    (2 / 3, 2 / 3),
)


async def _data_window(ds: Any, fx: Any) -> Window:
    """A WIN-sized window inside *fx* that actually contains imagery.

    Probing rather than guessing: a window of uniform nodata makes every
    "rastera equals GDAL" comparison in this file pass trivially, and which
    offsets are empty varies per fixture. The probes are WIN-sized reads of a
    COG, so trying a handful is cheap. Skips if the raster looks empty
    everywhere tried, rather than asserting against nothing.
    """
    for fcol, frow in _WINDOW_OFFSETS:
        win = _clamped_window(fx, int(fx.width * fcol), int(fx.height * frow))
        probe = np.asarray((await ds.read(window=win)).data)
        if len(np.unique(probe)) > 8:
            return win
    pytest.skip("no window with imagery found in this fixture")


def _clamped_window(fx: Any, col: int, row: int) -> Window:
    col = min(col, max(0, fx.width - WIN))
    row = min(row, max(0, fx.height - WIN))
    return Window(
        col_off=col,
        row_off=row,
        width=min(WIN, fx.width - col),
        height=min(WIN, fx.height - row),
    )


def _window_bbox(ds: Any, win: Window) -> tuple[float, float, float, float]:
    """The world-space bbox covering exactly *win*."""
    gt = ds._geotiff
    minx = gt.bounds[0] + win.col_off * gt.res[0]
    maxy = gt.bounds[3] - win.row_off * gt.res[1]
    return (minx, maxy - win.height * gt.res[1], minx + win.width * gt.res[0], maxy)


async def _nodata_corner_bbox(
    ds_a: Any, ds_b: Any, fx_a: Any
) -> tuple[float, float, float, float]:
    """A bbox at a corner of the two datasets' overlap where *a* is largely
    nodata and *b* largely valid — the region where ignoring nodata is visible.
    """
    b_bounds = ds_b._geotiff.bounds
    inter = (
        max(fx_a.bounds[0], b_bounds[0]),
        max(fx_a.bounds[1], b_bounds[1]),
        min(fx_a.bounds[2], b_bounds[2]),
        min(fx_a.bounds[3], b_bounds[3]),
    )
    span = WIN * fx_a.resolution
    best: tuple[float, tuple[float, float, float, float]] | None = None
    for cx, cy in (
        (inter[0], inter[1]),
        (inter[0], inter[3] - span),
        (inter[2] - span, inter[1]),
        (inter[2] - span, inter[3] - span),
    ):
        bbox = (cx, cy, cx + span, cy + span)
        pa = np.asarray((await ds_a.read(bbox=bbox, bbox_crs=fx_a.crs_epsg)).data)
        pb = np.asarray((await ds_b.read(bbox=bbox, bbox_crs=fx_a.crs_epsg)).data)
        score = min(float((pa == 0).mean()), float((pb != 0).mean()))
        if best is None or score > best[0]:
            best = (score, bbox)
    assert best is not None
    if best[0] < 0.1:
        pytest.skip("no overlap corner where one scene is nodata and the other is not")
    return best[1]


_ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
}


def _gdal_mosaic(fxs: list[Any], tmp: Path) -> str:
    """Path to a GDAL mosaic VRT over *fxs*, in listed order."""
    out = tmp / f"mosaic_{len(list(tmp.glob('mosaic_*.vrt')))}.vrt"
    _run(["gdalbuildvrt", "-q", str(out), *(f.vsis3_uri for f in fxs)])
    return str(out)


def _gdal_srcwin(uri: str, win: Window, tmp: Path) -> np.ndarray:
    return _gdal_translate(
        uri,
        [
            "-srcwin",
            str(win.col_off),
            str(win.row_off),
            str(win.width),
            str(win.height),
        ],
        tmp / "srcwin.img",
    )


def _gdal_projwin(
    uri: str, bbox: tuple[float, float, float, float], tmp: Path
) -> np.ndarray:
    minx, miny, maxx, maxy = bbox
    return _gdal_translate(
        uri,
        ["-projwin", str(minx), str(maxy), str(maxx), str(miny)],
        tmp / "projwin.img",
    )


def _gdal_translate(uri: str, args: list[str], out: Path) -> np.ndarray:
    """GDAL's own pixels, as a (bands, height, width) array.

    ENVI output because it is a flat raw cube plus a text header — no TIFF
    reader needed on the far side, and no risk of the oracle sharing a code
    path with what is being tested.
    """
    _run(["gdal_translate", "-q", "-of", "ENVI", *args, uri, str(out)])
    return _read_envi(out)


def _run(cmd: list[str]) -> None:
    # Inherits the environment on purpose — GDAL needs the same AWS credentials
    # rastera is using. NB: do NOT add GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR;
    # with it, gdal_translate fails on these VRTs with rc=1 and empty stderr.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"{cmd[0]} rc={proc.returncode}: {proc.stderr.strip()[:300]}"
        )


def _read_envi(path: Path) -> np.ndarray:
    hdr = {
        k.strip(): v.strip()
        for k, v in (
            line.split("=", 1)
            for line in path.with_suffix(".hdr").read_text().splitlines()
            if "=" in line
        )
    }
    samples, lines, bands = int(hdr["samples"]), int(hdr["lines"]), int(hdr["bands"])
    dtype = np.dtype(_ENVI_DTYPES[int(hdr["data type"])])
    dtype = dtype.newbyteorder(">" if hdr.get("byte order") == "1" else "<")
    raw = np.fromfile(path, dtype=dtype)
    interleave = hdr.get("interleave", "bsq").lower()
    if interleave == "bsq":
        arr = raw.reshape(bands, lines, samples)
    elif interleave == "bil":
        arr = raw.reshape(lines, bands, samples).transpose(1, 0, 2)
    else:  # bip
        arr = raw.reshape(lines, samples, bands).transpose(2, 0, 1)
    return np.ascontiguousarray(arr)
