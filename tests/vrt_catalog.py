"""Loader for the local-only VRT fixture catalog used by the live tests.

The catalog points at private storage, so it lives *outside the repository* —
by default ``~/.config/rastera/vrt_fixtures.json``. Only this loader is
committed, and it contains nothing environment-specific.

Outside the checkout rather than in a gitignored directory, for two reasons.
It cannot be committed at all, not even by ``git add -f`` or a tool that does
not honour ``.gitignore``. And it is shared: each git worktree has its own
working directory, so a copy inside the repo would exist in exactly one
checkout and the live tests would silently skip everywhere else.

Data, not an importable module, for the same first reason — a
``vrt_fixtures.py`` under ``rastera/`` is one packaging glob away from being
shipped in a wheel. JSON also stays readable by hand and by tooling without
importing the test suite.

Call ``load()`` from a live test; it skips cleanly when the catalog is
absent, which is the normal state for anyone who has not built one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

DEFAULT_CATALOG_PATH = Path.home() / ".config" / "rastera" / "vrt_fixtures.json"
ENV_VAR = "RASTERA_VRT_FIXTURES"


def catalog_path() -> Path:
    """Where the catalog is expected, honouring the ``RASTERA_VRT_FIXTURES``
    override (useful for pointing a one-off run at a different catalog)."""
    override = os.environ.get(ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_CATALOG_PATH


def load() -> Any:
    """The catalog, or a clean skip when it is not present locally.

    Returns a namespace with:

    - ``ALL_VRT_FIXTURES`` — every fixture, as attribute-access records
      (``s3_uri``, ``vsis3_uri``, ``family``, ``bands``, ``dtype``,
      ``nodata``, ``crs_epsg``, ``width``, ``height``, ``resolution``,
      ``bounds``, ``megapixels``, …).
    - ``LUT_PROCESSED_DATASET_VRTS`` / ``SIX_BAND_VRTS`` /
      ``NODATA_ABSENT_VRTS`` — named subsets, sharing the same objects as
      ``ALL_VRT_FIXTURES`` so identity and ``in`` comparisons work.
    - ``MERGE_SCENARIOS`` — fixture groups with a known spatial
      ``relationship``.
    """
    path = catalog_path()
    if not path.exists():
        # allow_module_level because test_vrt_live.py calls this at import
        # time to build its parametrize lists; without it a missing catalog is
        # a collection *error* rather than a skip. Harmless from inside a test
        # body, which is how test_live.py calls it — the runner reports the
        # raised Skipped either way.
        pytest.skip(
            f"local-only VRT catalog not present at {path} "
            f"(override with ${ENV_VAR})",
            allow_module_level=True,
        )
    return _parse(json.loads(path.read_text()))


# ── helpers ─────────────────────────────────────────────────────────────────


def _parse(doc: dict[str, Any]) -> Any:
    fixtures = [_fixture(raw) for raw in doc["fixtures"]]
    by_name = {f.name: f for f in fixtures}

    def pick(names: list[str]) -> tuple[Any, ...]:
        # Resolve through by_name rather than rebuilding, so a fixture listed
        # in two groups stays one object — test_vrt_live.py excludes fixtures
        # already covered elsewhere with `not in`.
        return tuple(by_name[n] for n in names)

    groups = doc["groups"]
    return SimpleNamespace(
        ALL_VRT_FIXTURES=tuple(fixtures),
        LUT_PROCESSED_DATASET_VRTS=pick(groups["lut_processed_dataset"]),
        SIX_BAND_VRTS=pick(groups["six_band"]),
        NODATA_ABSENT_VRTS=pick(groups["nodata_absent"]),
        MERGE_SCENARIOS=tuple(_scenario(raw, pick) for raw in doc["merge_scenarios"]),
    )


def _fixture(raw: dict[str, Any]) -> Any:
    fx = SimpleNamespace(**raw)
    fx.bounds = tuple(raw["bounds"])
    fx.colorinterp = tuple(raw["colorinterp"])
    # Derived rather than stored, so the catalog cannot disagree with itself.
    fx.vsis3_uri = raw["s3_uri"].replace("s3://", "/vsis3/", 1)  # GDAL-openable
    fx.megapixels = raw["width"] * raw["height"] / 1e6
    return fx


def _scenario(raw: dict[str, Any], pick: Any) -> Any:
    members = pick(raw["fixture_names"])
    return SimpleNamespace(
        name=raw["name"],
        relationship=raw["relationship"],
        fixtures=members,
        expected_union_bounds=tuple(raw["expected_union_bounds"]),
        expected_union_shape=tuple(raw["expected_union_shape"]),
        notes=raw["notes"],
        crs_epsg=members[0].crs_epsg,
        resolution=members[0].resolution,
    )
