"""Benchmark rastera's geoparquet index vs network opens.

Runs each operation across many different sites/files to get stable timings.

Usage:
    python examples/index_bench.py
"""

import asyncio
import json
import os
import statistics
import time
import urllib.request

import geopandas as gpd
from shapely.geometry import box

import rastera

S3_OPTS = {"region": "us-west-2"}
INDEX_PATH = "/tmp/sentinel_bench_index.parquet"
N_ROUNDS = 20


def search_stac(bbox: list, limit: int = 100) -> list[str]:
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


def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def stats_row(label: str, idx_times: list[float], net_times: list[float]) -> list[str]:
    ai = statistics.mean(idx_times)
    an = statistics.mean(net_times)
    return [
        label,
        f"{ai:.3f}",
        f"{statistics.stdev(idx_times):.3f}",
        f"{statistics.median(idx_times):.3f}",
        f"{percentile(idx_times, 90):.3f}",
        f"{an:.3f}",
        f"{statistics.stdev(net_times):.3f}",
        f"{statistics.median(net_times):.3f}",
        f"{percentile(net_times, 90):.3f}",
        f"{an / ai:.1f}x" if ai > 0 else "-",
    ]


def print_table(headers: list[str], rows: list[list[str]]):
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(vals):
        return "  ".join(v.rjust(w) if i > 0 else v.ljust(w) for i, (v, w) in enumerate(zip(vals, widths)))

    def sep():
        return "  ".join("-" * w for w in widths)

    print()
    print(line(headers))
    print(sep())
    for row in rows:
        print(line(row))
    print()


async def main():
    # Setup
    print("Searching STAC for Sentinel-2 tiles (Scandinavia, June 2024)...", flush=True)
    uris = search_stac(bbox=[10, 55, 25, 65], limit=100)
    print(f"Found {len(uris)} COGs\n", flush=True)

    if not os.path.exists(INDEX_PATH):
        print("Building index...", flush=True)
        gdf = await rastera.build_index(uris, concurrency=50, **S3_OPTS)
        gdf.to_parquet(INDEX_PATH, compression="zstd")
        print(f"Saved to {INDEX_PATH}\n", flush=True)

    gdf = gpd.read_parquet(INDEX_PATH)
    n_files = len(gdf)

    # Prepare N_ROUNDS different sites (spread across dataset)
    step = max(1, n_files // (N_ROUNDS + 1))
    sites = []
    for i in range(N_ROUNDS):
        row = gdf.iloc[step * (i + 1) % n_files]
        c = row.geometry.centroid
        qbbox = (c.x - 0.05, c.y - 0.05, c.x + 0.05, c.y + 0.05)
        matched = gdf[gdf.intersects(box(*qbbox))]["uri"].tolist()
        sites.append((qbbox, matched))

    print(f"Config: {N_ROUNDS} rounds, {n_files} files in index, cold cache each round", flush=True)
    print(f"Index size: {os.path.getsize(INDEX_PATH) / 1024:.0f} KB (zstd geoparquet)\n", flush=True)

    # --- Benchmark: open single file ---
    print(f"[1/3] Open single file ({N_ROUNDS} different files)...", flush=True)
    t_open1_idx, t_open1_net = [], []
    for i in range(N_ROUNDS):
        uri = uris[i * (n_files // N_ROUNDS) % n_files]
        single_gdf = gdf[gdf["uri"] == uri]

        rastera.clear_cache()
        t0 = time.perf_counter()
        await rastera.open_from_index(single_gdf, **S3_OPTS)
        t_open1_idx.append(time.perf_counter() - t0)

        rastera.clear_cache()
        t0 = time.perf_counter()
        await rastera.open(uri, **S3_OPTS)
        t_open1_net.append(time.perf_counter() - t0)

        print(f"  [{i+1:2d}] index={t_open1_idx[-1]:.3f}s  network={t_open1_net[-1]:.3f}s", flush=True)

    # --- Benchmark: open all files ---
    print(f"\n[2/3] Open all {n_files} files ({N_ROUNDS} rounds)...", flush=True)
    t_open_idx, t_open_net = [], []
    for i in range(N_ROUNDS):
        rastera.clear_cache()
        t0 = time.perf_counter()
        await rastera.open_from_index(gdf, concurrency=50, **S3_OPTS)
        t_open_idx.append(time.perf_counter() - t0)

        rastera.clear_cache()
        t0 = time.perf_counter()
        await rastera.open(uris, **S3_OPTS)
        t_open_net.append(time.perf_counter() - t0)

        print(f"  [{i+1:2d}] index={t_open_idx[-1]:.2f}s  network={t_open_net[-1]:.2f}s", flush=True)

    # --- Benchmark: query + merge ---
    print(f"\n[3/3] Query+merge ~10km window ({N_ROUNDS} different sites)...", flush=True)
    t_query_idx, t_query_net = [], []
    for i, (qbbox, matched_uris) in enumerate(sites):
        rastera.clear_cache()
        t0 = time.perf_counter()
        sources = await rastera.open_from_index(gdf, bbox=qbbox, **S3_OPTS)
        await rastera.merge(sources, bbox=qbbox, bbox_crs=4326)
        t_query_idx.append(time.perf_counter() - t0)

        rastera.clear_cache()
        t0 = time.perf_counter()
        sources = await rastera.open(matched_uris, **S3_OPTS)
        await rastera.merge(sources, bbox=qbbox, bbox_crs=4326)
        t_query_net.append(time.perf_counter() - t0)

        print(f"  [{i+1:2d}] {len(matched_uris)} files  index={t_query_idx[-1]:.2f}s  network={t_query_net[-1]:.2f}s", flush=True)

    # --- Summary table ---
    headers = [
        "Operation",
        "Idx avg", "Idx std", "Idx med", "Idx p90",
        "Net avg", "Net std", "Net med", "Net p90",
        "Speedup",
    ]
    rows = [
        stats_row("Open 1 file", t_open1_idx, t_open1_net),
        stats_row(f"Open {n_files} files", t_open_idx, t_open_net),
        stats_row("Query+merge", t_query_idx, t_query_net),
    ]
    print_table(headers, rows)

    print(f"  {N_ROUNDS} rounds, cold cache, different files/sites each round.")
    print(f"  Data: {n_files} public Sentinel-2 B04 COGs (us-west-2).")
    print(f"  Speedup = mean(network) / mean(index).\n")


if __name__ == "__main__":
    asyncio.run(main())
