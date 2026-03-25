# rastera

**Async rasterio for COGs**, built on [async-geotiff](https://github.com/developmentseed/async-geotiff), no GDAL.

- `read` and `merge` (multi-file, cross-crs) with `target_crs`, `target_resolution`, `bbox`, `window`
- Optional persisted header cache (geoparquet) for ~6x faster opens
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

### COG header cache via geoparquet index

Pre-cache COG headers in a geoparquet file to skip S3 round-trips on open (~6x faster, e.g. 0.2s vs 1.3s for opening 100 COGs).
Requires additionals dependencies, install via `pip install rastera[index]`

```python
import rastera

uris = ["s3://bucket/tile_a.tif", "s3://bucket/tile_b.tif", ...]

# Build once, save to disk
gdf = await rastera.build_index(uris, region="us-west-2")
gdf.to_parquet("index.parquet")

# Open from index (reusable across sessions, ~5-6x faster opens)
sources = await rastera.open_from_index("index.parquet", bbox=(minx, miny, maxx, maxy), region="us-west-2")
data, profile = await rastera.merge(sources, bbox=bbox, bbox_crs=4326)
```




## TODO maybe

- Bilinear / cubic / lanczos resampling (currently nearest-neighbor only)
- Basic raster stats (min, max, mean, histogram)