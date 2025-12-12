# rastera

"Async rasterio for COGs", build om [async-tiff](https://github.com/developmentseed/async-tiff), no GDAL.

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
    bbox_crs=3006,
    target_crs=32633,
    target_resolution=10
)
```

### Merge to mosaic

```python
uris = ["s3://bucket/tile_a.tif", "s3://bucket/tile_b.tif", ...]
sources = await rastera.open(uris)  # concurrent opens, shared connection pool

data, profile = await rastera.merge(sources, bbox=bbox, bbox_crs=4326, target_resolution=10)
```

## TODO maybe

- Bilinear / cubic / lanczos resampling (currently nearest-neighbor only)
- Persistent COG header cache across sessions (e.g. SQLite/diskcache)
- Clip / read by geometry (polygon mask, not just bbox)
- Basic raster stats (min, max, mean, histogram)

## Development Install
```
# Newest async-tiff state
uv pip install "git+https://github.com/developmentseed/async-tiff.git#subdirectory=python"
uv pip install -e .
uv pip install ipykernel tifffile geopandas matplotlib
```
