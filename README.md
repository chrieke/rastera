# rastera

**Async rasterio for COGs**, built on [async-geotiff](https://github.com/developmentseed/async-geotiff), no GDAL.

- `read` and `merge` (multi-file, cross-crs) with `target_crs`, `target_resolution`, `bbox`, `window`
- Optional persisted header cache (geoparquet) for ~6x faster opens
- Built on [async-geotiff](https://github.com/developmentseed/async-geotiff) handling GeoTIFF parsing, async tile fetching, request coalescing, and Rust-native decompression
- Limited VRT & DIMAP support

**Note:** Only COGs & tiled GeoTIFFs are supported. Stripped (non-tiled) TIFFs will not work.

**VRT TODO — band-level features ignored.** The VRT parser currently reads only `<SourceFilename>` + `<SourceBand>` and inherits everything else from the first source. The following are silently dropped:

- `<VRTRasterBand>`: `NoDataValue`, `ColorInterp`, `Description`, `dataType`
- `<SimpleSource>`: `NODATA`, `SrcRect`, `DstRect`
- `<ComplexSource>` (any variant): rejected with `NotImplementedError`

### Read a single COG

```python
import rastera

uri = "s3://my-bucket/my-cog.tif"
src = await rastera.open(uri, prefetch=32768, cache=True, meta_overrides=None)

# Full image
raster_array = await src.read()
# raster_array.data, raster_array.transform, raster_array.bounds, raster_array.crs, raster_array.nodata, ...

# Spatial subset with reprojection
raster_array = await src.read(
    bbox=(minx, miny, maxx, maxy),
    bbox_crs=32633,
    band_indices=[1, 2, 3],
    target_crs=32632,
    target_resolution=20,
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

### Merge to mosaic

```python
uris = ["s3://bucket/tile_a.tif", "s3://bucket/tile_b.tif", ...]
sources = await rastera.open(uris)  # concurrent opens, shared connection pool

raster_array = await rastera.merge(
    sources,
    bbox=bbox_shared,
    bbox_crs=utm_crs,
    band_indices=[1],
    fill_value=0,
    target_crs=utm_crs,
    target_resolution=10,
    mosaic_method="first",
    crs_method="most_common",
    snap_to_grid=True,
    use_overviews=False,
)
```

### COG header cache via geoparquet index

Pre-cache COG headers in a geoparquet file to skip S3 round-trips on open (~6x faster, e.g. 0.2s vs 1.3s for opening 100 COGs).
Requires additional dependencies, install via `pip install rastera[index]`

```python
import rastera

uris = ["s3://bucket/tile_a.tif", "s3://bucket/tile_b.tif", ...]

# Build once, save to disk
gdf = await rastera.build_index(uris, prefetch=32768, concurrency=100, region="us-west-2")
gdf.to_parquet("index.parquet")

# Open from index (reusable across sessions, ~5-6x faster opens)
sources = await rastera.open_from_index("index.parquet", bbox=(minx, miny, maxx, maxy), region="us-west-2")
raster_array = await rastera.merge(sources, bbox=bbox, bbox_crs=4326, target_crs=4326, target_resolution=10)
```

`rastera.open()` also keeps an in-memory LRU cache of parsed headers within the session (default 128 entries, configurable via `set_cache_size()`), so repeated opens of the same URI skip the network fetch even without an index.

### Linting & type checking

```bash
uv run ruff format . && uv run ruff check --fix . && uv run pyright
```