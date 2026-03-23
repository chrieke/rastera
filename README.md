# rastera

**Async rasterio for COGs**, build on [async-tiff](https://github.com/developmentseed/async-tiff), no GDAL.

- `read` and multi-file, cross-crs `merge` with `target_crs`, `target_resolution`, `bbox`, `window`
- Parallel everywhere: concurrent file opens with shared connection pool, concurrent tile downloads within each file, concurrent merge reads across files
- Built on [async-tiff](https://github.com/developmentseed/async-tiff) handling async tile fetching, batched range requests, and Rust-native decompression
- In-memory COG header cache (TODO: Between sessions)

**Note:** Only COGs & tiled GeoTIFFs are supported. Stripped (non-tiled) TIFFs will not work.

### Read a single COG

```python
import rastera

uri = "s3://my-bucket/my-cog.tif"
src = await rastera.open(uri)

# Full image
data, profile = await src.read()

# Spatial subset with reprojection
data, profile = await src.read(
    bbox=(minx, miny, maxx, maxy),
    bbox_crs=32633,
    target_crs=32632,
    target_resolution=20
)
```

### Merge to mosaic

```python
uris = ["s3://bucket/tile_a.tif", "s3://bucket/tile_b.tif", ...]
sources = await rastera.open(uris)  # concurrent opens, shared connection pool

data, profile = await rastera.merge(sources, bbox=bbox, bbox_crs=32633, target_resolution=20)
```

## TODO maybe

- Bilinear / cubic / lanczos resampling (currently nearest-neighbor only)
- Persistent COG header cache across sessions (e.g. SQLite/diskcache)
- Clip / read by geometry (polygon mask, not just bbox)
- Basic raster stats (min, max, mean, histogram)

## Concurrency and tile batching

When merging, rastera reads multiple COGs concurrently (`max_concurrency`,
default 6). Each COG read fetches tiles via `async_tiff.fetch_tiles()`, which
fires all HTTP range requests in parallel on the Rust side. For high-resolution
COGs (e.g. 16cm ortofoto), a single bbox read can require 100-200+ tiles.

Without batching, `max_concurrency` COGs each firing 150 tile requests = 900
simultaneous HTTP connections, which overwhelms the HTTP client's connection
pool (reqwest) with "request or response body error". This is a client-side
limit, not S3 throttling.

To prevent this, tile fetches are batched: instead of one `fetch_tiles(150)`
call, rastera issues `fetch_tiles(32)` sequentially per COG. Peak concurrent
requests = `max_concurrency × batch_size` (e.g. 6 × 32 = 192), which is safe.

Concurrent COGs still provide a throughput advantage over sequential reads
(benchmark: 552s at concurrency=3 vs 597s at concurrency=1 for 15 COGs) because
while one COG awaits a tile batch response, other COGs are fetching their
batches in parallel. This advantage is largest with many small images (few tiles
each), where a single COG's fetch finishes quickly and leaves bandwidth idle.
With few large images, one COG's tile batches already saturate bandwidth, so
concurrent COGs mainly help overlap tail latency between COGs.

Tuning:
```python
import rastera

# Adjust tile batch size (default 32) — higher = more concurrent HTTP requests
# per COG, lower = safer but potentially slower for single-COG reads
rastera.set_tile_fetch_batch_size(64)
```

## Development Install
```
# Newest async-tiff state
uv pip install "git+https://github.com/developmentseed/async-tiff.git#subdirectory=python"
uv pip install -e .
uv pip install ipykernel tifffile geopandas matplotlib
```
