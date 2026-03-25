"""Demonstrates rastera's geoparquet index using public Sentinel-2 COGs.

Usage:
    python examples/index_demo.py build     # Build index from STAC search
    python examples/index_demo.py open      # Open from index vs network
    python examples/index_demo.py query     # Spatial query + merge benchmark
    python examples/index_demo.py all       # Run everything
"""

import asyncio
import json
import os
import sys
import time
import urllib.request

import geopandas as gpd
from shapely.geometry import box

import rastera

S3_OPTS = {"region": "us-west-2"}
INDEX_PATH = "/tmp/sentinel_scandinavia_index.parquet"
STAC_BBOX = [10, 55, 25, 65]  # Scandinavia
STAC_LIMIT = 100
N_QUERIES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def search_stac(bbox: list, limit: int = 100) -> list[str]:
    """Search Element84 Earth Search STAC for Sentinel-2 B04 COG URIs."""
    url = "https://earth-search.aws.element84.com/v1/search"
    body = json.dumps({
        "collections": ["sentinel-2-l2a"],
        "limit": limit,
        "bbox": bbox,
        "datetime": "2024-06-01T00:00:00Z/2024-06-30T23:59:59Z",
        "fields": {"includes": ["assets.red.href"]},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())

    uris = []
    for feat in resp.get("features", []):
        href = feat.get("assets", {}).get("red", {}).get("href", "")
        if href:
            uris.append(href.replace(
                "https://sentinel-cogs.s3.us-west-2.amazonaws.com/",
                "s3://sentinel-cogs/",
            ))
    return uris


def timed(label: str):
    """Context manager that prints elapsed time."""
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self._t0
            print(f"  {label}: {self.elapsed:.2f}s", flush=True)
    return Timer()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_build():
    """Build a geoparquet index from a STAC search."""
    print("Searching STAC...", flush=True)
    uris = search_stac(bbox=STAC_BBOX, limit=STAC_LIMIT)
    print(f"Found {len(uris)} COGs\n", flush=True)

    with timed(f"Built index for {len(uris)} files") as t:
        gdf = await rastera.build_index(uris, concurrency=50, **S3_OPTS)

    gdf.to_parquet(INDEX_PATH, compression="zstd")
    size_kb = os.path.getsize(INDEX_PATH) / 1024
    print(f"  Saved to {INDEX_PATH} ({size_kb:.0f} KB)\n", flush=True)


async def cmd_open():
    """Compare opening files from index vs network."""
    gdf = gpd.read_parquet(INDEX_PATH)
    uris = gdf["uri"].tolist()
    n = len(uris)
    print(f"Index: {n} files\n", flush=True)

    rastera.clear_cache()
    with timed(f"Opened {n} from network") as t_net:
        await rastera.open(uris, **S3_OPTS)

    rastera.clear_cache()
    with timed(f"Opened {n} from index") as t_idx:
        await rastera.open_from_index(gdf, concurrency=50, **S3_OPTS)

    ratio = t_net.elapsed / t_idx.elapsed if t_idx.elapsed > 0 else float("inf")
    print(f"\n  Speedup: {ratio:.1f}x\n", flush=True)


async def cmd_query():
    """Spatial query + merge benchmark, index vs network (cold cache)."""
    gdf = gpd.read_parquet(INDEX_PATH)
    print(f"Index: {len(gdf)} files\n", flush=True)

    # Pick N bboxes spread across the dataset (~10km x 10km each, in EPSG:4326)
    step = len(gdf) // (N_QUERIES + 1)
    bboxes = []
    for i in range(N_QUERIES):
        c = gdf.iloc[step * (i + 1)].geometry.centroid
        bboxes.append((c.x - 0.05, c.y - 0.05, c.x + 0.05, c.y + 0.05))

    print(f"--- {N_QUERIES} queries via index (cold cache) ---", flush=True)
    t_total_idx = 0.0
    for i, qbbox in enumerate(bboxes):
        rastera.clear_cache()
        t0 = time.perf_counter()
        sources = await rastera.open_from_index(gdf, bbox=qbbox, **S3_OPTS)
        data, _ = await rastera.merge(sources, bbox=qbbox, bbox_crs=4326)
        dt = time.perf_counter() - t0
        t_total_idx += dt
        print(f"  [{i+1}] {len(sources)} files  {data.shape}  {dt:.2f}s", flush=True)

    print(f"\n--- {N_QUERIES} queries via network (cold cache) ---", flush=True)
    t_total_net = 0.0
    for i, qbbox in enumerate(bboxes):
        rastera.clear_cache()
        t0 = time.perf_counter()
        matched_uris = gdf[gdf.intersects(box(*qbbox))]["uri"].tolist()
        sources = await rastera.open(matched_uris, **S3_OPTS)
        data, _ = await rastera.merge(sources, bbox=qbbox, bbox_crs=4326)
        dt = time.perf_counter() - t0
        t_total_net += dt
        print(f"  [{i+1}] {len(sources)} files  {data.shape}  {dt:.2f}s", flush=True)

    print(f"\n--- Results ---", flush=True)
    print(f"  Index:   {t_total_idx:.2f}s  ({t_total_idx/N_QUERIES:.2f}s avg)", flush=True)
    print(f"  Network: {t_total_net:.2f}s  ({t_total_net/N_QUERIES:.2f}s avg)", flush=True)
    print(f"  Speedup: {t_total_net / t_total_idx:.1f}x\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMMANDS = {
    "build": cmd_build,
    "open": cmd_open,
    "query": cmd_query,
}


async def run_all():
    for name, fn in COMMANDS.items():
        print(f"{'=' * 60}", flush=True)
        print(f"  {name.upper()}", flush=True)
        print(f"{'=' * 60}\n", flush=True)
        await fn()


def main():
    args = sys.argv[1:]
    if not args or args[0] == "all":
        asyncio.run(run_all())
    elif args[0] in COMMANDS:
        asyncio.run(COMMANDS[args[0]]())
    else:
        print(f"Unknown command: {args[0]}")
        print(f"Available: {', '.join(COMMANDS)} or all")
        sys.exit(1)


if __name__ == "__main__":
    main()
