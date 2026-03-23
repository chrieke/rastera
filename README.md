# rastera

**Async rasterio for COGs**, build on [async-tiff](https://github.com/developmentseed/async-tiff), no GDAL.

- `read` and `merge` (multi-file, cross-crs) with `target_crs`, `target_resolution`, `bbox`, `window`
- Parallel everywhere: concurrent file opens, tile downloads, and optimized cross-file merges for maximum network throughput
- Built on [async-tiff](https://github.com/developmentseed/async-tiff) handling async tile fetching, batched range requests, and Rust-native decompression

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

## Concurrency and tile batching

When reading a COG, it fires all its tile HTTP range requests in parallel, handled 
concurrently by async-tiff. Big, high-resolution COGs can have multiple hundred tiles.

When mosaicing images via `merge`, we need to read multiple COGs. Reading them sequentially 
(all tiles for COG 1, then all tiles for COG 2, ...) leaves the network idle 
between each COG while tiles are decoded and pasted into the mosaic.
Reading them concurrently (`max_concurrency`, default 4) lets the next COG's
tile fetches overlap with the previous COG's processing, keeping the network busy. 
However, e.g. 4 COGs each firing ~150 tile requests means ~600
simultaneous connections — depending on the client, enough to exhaust the 
HTTP client's connection handling and cause failures.

To keep this in check, tile fetches during merges are batched (`tile_batch_size`,
default 48) per COG. This introduces short pauses between batches within each
COG, but those gaps are filled by other COGs fetching their batches concurrently.
Depending on the client, `tile_batch_size` and `max_concurrency` parameters can be tuned, 
but likely unnecessary. Standalone single-COG reads are unbatched and fire all tile requests at once.


## TODO maybe

- Bilinear / cubic / lanczos resampling (currently nearest-neighbor only)
- Extend current per-session wide persistent COG header cache across sessions (e.g. SQLite/diskcache)
- Basic raster stats (min, max, mean, histogram)