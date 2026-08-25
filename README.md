# rastera

**Async rasterio for COGs**, built on [async-geotiff](https://github.com/developmentseed/async-geotiff), no GDAL.

- `read` and `merge` (multi-file, cross-crs) with `target_crs`, `target_resolution`, `bbox`, `window`, `resampling`
- Resampling: `nearest` (default), `bilinear`, `cubic` — GDAL-matching kernels with anti-aliasing on downsample and nodata renormalization
- Optional persisted header cache (geoparquet) for ~6x faster opens
- Built on [async-geotiff](https://github.com/developmentseed/async-geotiff) handling GeoTIFF parsing, async tile fetching, request coalescing, and Rust-native decompression
- Limited VRT & DIMAP support — band-stack VRTs and LUTs work, anything more exotic raises `NotImplementedError` instead of returning wrong pixels (see `rastera/vrt.py`)

**Note:** Only COGs & tiled GeoTIFFs are supported. Striped (non-tiled) TIFFs will not work.

### Read a single COG

```python
import rastera

uri = "s3://my-bucket/my-cog.tif"
src = await rastera.open(uri, prefetch=32768, cache=True, meta_overrides=None)

# Full image
raster_array = await src.read()
# raster_array.data, raster_array.transform, raster_array.bounds, raster_array.crs, raster_array.nodata, ...

# Spatial subset with reprojection — bbox_crs must match target_crs
# (merge transforms the bbox for you, read does not)
raster_array = await src.read(
    bbox=(minx, miny, maxx, maxy),
    bbox_crs=32632,
    band_indices=[1, 2, 3],
    target_crs=32632,
    target_resolution=20,
    resampling="nearest",  # also: "bilinear", "cubic"
    snap_to_grid=True,
    use_overviews=False,
)

# Read by pixel window (no reprojection)
raster_array = await src.read(
    window=rastera.Window(col_off=0, row_off=0, width=512, height=512),
    band_indices=[1],
    target_resolution=20,
    use_overviews=False,
)
```

### Dataset metadata

`src.profile` returns the header's metadata as one dict — grid, CRS, dtype, nodata, bounds, overviews — costing no request. Keys and caveats are documented on `rastera.RasterProfile`.

### Merge to mosaic

```python
uris = ["s3://bucket/tile_a.tif", "s3://bucket/tile_b.tif", ...]
sources = await rastera.open(uris)  # concurrent opens, shared connection pool

raster_array = await rastera.merge(
    sources,
    bbox=bbox_shared,
    bbox_crs=utm_crs,
    band_indices=[1],
    nodata=0,  # what uncovered pixels get, and what arr.nodata reports
    target_crs=utm_crs,
    target_resolution=10,
    resampling="nearest",  # also: "bilinear", "cubic"
    mosaic_method="first",
    crs_method="most_common",
    snap_to_grid=True,
    use_overviews=False,
)

# True where an input covered the pixel, False where none did. Gaps hold the
# nodata above — but real pixels can too, so the mask is what tells them apart.
coverage = raster_array.mask

# The exact grid read/merge return for snap_to_grid=True, without any I/O —
# use it to pre-size buffers, key caches, or align a bbox to the grid.
transform, width, height = rastera.snapped_grid_for_bbox(bbox_shared, 10)
```

### COG header cache via geoparquet index

Pre-cache COG headers in a geoparquet file to skip S3 round-trips on open (~6x faster, e.g. 0.2s vs 1.3s for opening 100 COGs).
Requires additional dependencies, install via `pip install rastera[index]`

```python
import rastera

uris = ["s3://bucket/tile_a.tif", "s3://bucket/tile_b.tif", ...]

# Build once, save to disk
gdf = await rastera.build_index(
    uris, prefetch=32768, concurrency=100, region="us-west-2"
)
gdf.to_parquet("index.parquet")

# Open from index (reusable across sessions, ~5-6x faster opens)
sources = await rastera.open_from_index(
    "index.parquet", bbox=(minx, miny, maxx, maxy), region="us-west-2"
)
raster_array = await rastera.merge(
    sources, bbox=bbox, bbox_crs=4326, target_crs=32632, target_resolution=10
)
```

`rastera.open()` also keeps an in-memory LRU cache of parsed headers within the session (default 128 entries, configurable via `set_cache_size()`), so repeated opens of the same URI skip the network fetch even without an index.

By default the read path runs the *outer* fan-out across `merge` contributors, VRT sources, and DIMAP tiles sequentially — async-geotiff already parallelizes block range requests inside each source, so stacking outer concurrency on top tends to multiply the in-flight HTTP request count without adding throughput on a saturated link. Use `rastera.set_concurrency(merge=N, vrt=N, dimap=N)` to opt into outer fan-out per dispatcher; see the `set_concurrency` docstring for the per-knob trade-offs.

Cross-CRS bilinear/cubic downsampling beyond 2x uses a faster two-pass warp; `rastera.set_warp_strategy("single_pass")` opts out when bit-exact reproducibility matters.

### Linting & type checking

```bash
uv run ruff format . && uv run ruff check --fix . && uv run pyright
```
