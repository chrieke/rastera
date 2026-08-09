"""A dataset's header metadata, collected into one dict."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from affine import Affine
from pyproj import CRS

from .geo import BBox

if TYPE_CHECKING:
    from .reader import AsyncGeoTIFF


class RasterProfile(TypedDict):
    """Everything a dataset's header says about it, in one dict.

    Describes the dataset; it is not a set of writer creation options. Key
    names follow rasterio's where there is an equivalent, but splatting this
    into ``rasterio.open(path, "w", **profile)`` only *appears* to work:
    ``bounds``, ``res``, ``crs_epsg`` and ``overviews`` have no
    creation-option meaning, and GDAL drops unknown creation options with
    nothing but a logged warning. Select the keys a writer needs.

    Two keys are not what the file literally declares, deliberately:

    - ``crs`` is the file's own CRS object, not one rebuilt from ``crs_epsg``,
      and a ``meta_overrides`` CRS replaces both.
    - ``nodata`` is the sentinel this dataset's *pixels* use. It is None where
      the dtype cannot carry the declared value, and a VRT's ``<NoDataValue>``
      replaces its source's.

    ``dtype`` is None only when async-geotiff cannot map the file's sample
    format — but such a file cannot be read either.

    How the pixels are *stored* (compression, tiling, interleave) and per-band
    metadata (colormap, scales, offsets) are absent.
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
    # Read through ``_geotiff`` except where this dataset resolved something
    # different: ``count`` is a property a band-stack VRT overrides, and crs
    # and nodata are resolved at construction.
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
        "bounds": BBox(*gt.bounds),
        "res": (res[0], res[1]),
        "nodata": src._nodata,
        "overviews": list(src.overviews),
    }
