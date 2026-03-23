# rastera

**Async rasterio for COGs**, built on [async-geotiff](https://github.com/developmentseed/async-geotiff), no GDAL.

- `read` and `merge` (multi-file, cross-crs) with `target_crs`, `target_resolution`, `bbox`, `window`
- Parallel everywhere: concurrent file opens, coalesced tile downloads with Rust-native decoding
- Built on [async-geotiff](https://github.com/developmentseed/async-geotiff) handling GeoTIFF parsing, async tile fetching, request coalescing, and Rust-native decompression

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

## Concurrency

When reading a single COG, async-geotiff fires all tile HTTP range requests in
parallel with request coalescing via obspec (contiguous tile ranges are merged
into fewer HTTP requests), and decodes tiles on a Rust thread pool.

When mosaicing via `merge`, COGs are read sequentially — one COG is fully read
and pasted before the next starts. Tile-level parallelism within each COG (via
async-geotiff's coalesced fetches and Rust-native decoding) keeps the network
busy.

Reading multiple COGs concurrently would be faster — the next COG's tile fetches
could overlap with the previous COG's decoding, keeping the network saturated
instead of idle between COGs. However, async-geotiff coalesces contiguous tile
ranges into single HTTP requests, and this coalescing only works within a single
`fetch_tiles()` call. Batching tiles to limit concurrent requests (as needed
when multiple COGs fire requests simultaneously) would break coalescing at batch
boundaries, negating the benefit. For now, sequential reads with full coalescing
per COG is the simpler and more reliable approach.


## TODO maybe

- Bilinear / cubic / lanczos resampling (currently nearest-neighbor only)
- Extend current per-session wide persistent COG header cache across sessions (e.g. SQLite/diskcache)
- Basic raster stats (min, max, mean, histogram)