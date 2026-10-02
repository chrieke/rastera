"""A dataset's header metadata, collected into one dict."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from affine import Affine
from pyproj import CRS

from .geo import BBox, _grid_bounds

if TYPE_CHECKING:
    from .reader import AsyncGeoTIFF


class RasterProfile(TypedDict):
    """Everything a dataset's header says about it, in one dict.

    Key names follow rasterio's, but this is not a set of creation options:
    passed to ``rasterio.open(path, "w", **profile)``, ``bounds``, ``res``,
    ``crs_epsg`` and ``overviews`` are dropped with only a logged warning.

    - ``crs`` is the file's own CRS object, or the ``meta_overrides`` one.
    - ``nodata`` is the sentinel the pixels use: None where the dtype cannot
      carry the declared value, and a VRT's ``<NoDataValue>`` over its
      source's.
    - ``dtype`` is None only for a sample format async-geotiff cannot map,
      which cannot be read either.

    Storage (compression, tiling, interleave) and per-band metadata (colormap,
    scales, offsets) are absent.
    """

    width: int
    height: int
    count: int
    dtype: str | None
    crs: CRS
    crs_epsg: int | None
    transform: Affine
    bounds: BBox
    res: tuple[float, float]
    nodata: int | float | None
    overviews: list[tuple[int, int]]


def _build_profile(src: AsyncGeoTIFF) -> RasterProfile:
    # Off ``_geotiff``, except what the dataset resolves itself.
    gt = src._geotiff
    res = gt.res
    return {
        "width": gt.width,
        "height": gt.height,
        "count": src.count,
        "dtype": str(gt.dtype) if gt.dtype is not None else None,
        "crs": src._resolved_crs,
        "crs_epsg": src._crs_epsg,
        "transform": gt.transform,
        "bounds": _grid_bounds(gt),
        "res": (res[0], res[1]),
        "nodata": src._nodata,
        "overviews": list(src.overviews),
    }
