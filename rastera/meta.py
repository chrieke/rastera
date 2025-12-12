from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from affine import Affine
import numpy as np

from .geo import BBox, Window, bounds_from_transform

_GDAL_NODATA_TAG = 42113


@dataclass
class Profile:
    """Core geospatial metadata for a GeoTIFF-like dataset.

    This mirrors the minimal subset of information we need from the TIFF IFD
    to perform pixel/coordinate conversions and windowing.
    """

    width: int
    height: int
    count: int  # number of bands (samples per pixel)
    dtype: np.dtype
    transform: Affine  # maps (col,row) -> (x,y)
    res: tuple[float, float]  # pixel size (xres, yres) in CRS units
    crs_epsg: int | None
    bounds: BBox
    nodata: int | float | None = None
    overviews: list[tuple[int, int]] | None = None

    def __repr__(self) -> str:
        parts = [
            f"width={self.width}",
            f"height={self.height}",
            f"count={self.count}",
            f"dtype={self.dtype!r}",
            f"res={self.res}",
            f"crs_epsg={self.crs_epsg}",
            f"nodata={self.nodata}",
            f"overviews={self.overviews}",
            f"bounds={self.bounds}",
            f"transform={self.transform!r}",
        ]
        return f"Profile({', '.join(parts)})"

    def copy(self) -> Profile:
        return replace(self)

    @classmethod
    def for_bbox(
        cls,
        bbox: BBox,
        res: float,
        crs_epsg: int | None,
        count: int,
        dtype: np.dtype,
        nodata: int | float | None = None,
    ) -> Profile:
        """Build a Profile for a regular grid covering *bbox* at the given resolution."""
        width = max(1, math.ceil(bbox.width / res))
        height = max(1, math.ceil(bbox.height / res))
        transform = Affine(res, 0, bbox.minx, 0, -res, bbox.maxy)
        bounds = BBox(
            bbox.minx,
            bbox.maxy - height * res,
            bbox.minx + width * res,
            bbox.maxy,
        )
        return cls(
            width=width,
            height=height,
            count=count,
            dtype=dtype,
            transform=transform,
            res=(res, res),
            crs_epsg=crs_epsg,
            bounds=bounds,
            nodata=nodata,
        )

    def adjust_to_window(self, window: Window) -> Profile:
        window_transform = self.transform * Affine.translation(
            window.col_min, window.row_min
        )
        width = window.win_width
        height = window.win_height
        return replace(
            self,
            width=width,
            height=height,
            transform=window_transform,
            bounds=bounds_from_transform(window_transform, width, height),
        )

    @classmethod
    def from_ifd(cls, ifd: Any) -> Profile:
        """Construct `Profile` (dimensions, tiling, transform, CRS) from a TIFF IFD."""
        gkd = ifd.geo_key_directory
        crs_epsg = None
        if gkd is not None:
            crs_epsg = getattr(gkd, "projected_type", None) or getattr(gkd, "geographic_type", None)

        transform = _transform_from_ifd(ifd)
        dtype = np.dtype(_dtype_from_ifd(ifd))
        res = (float(transform.a), float(-transform.e))

        return cls(
            width=ifd.image_width,
            height=ifd.image_height,
            count=ifd.samples_per_pixel,
            dtype=dtype,
            transform=transform,
            res=res,
            crs_epsg=crs_epsg,
            bounds=bounds_from_transform(transform, ifd.image_width, ifd.image_height),
            nodata=_nodata_from_ifd(ifd, dtype),
        )


def _transform_from_ifd(ifd: Any) -> Affine:
    """Build an Affine transform from GeoTIFF ModelPixelScale / ModelTiepoint tags."""
    mps = ifd.model_pixel_scale
    mtp = ifd.model_tiepoint

    if not (mps and mtp and len(mps) >= 2 and len(mtp) >= 6):
        msg = "Missing model_pixel_scale/model_tiepoint"
        raise ValueError(msg)

    scale_x, scale_y = mps[0], mps[1]
    i, j, _k, x0, y0, _z = mtp[:6]

    # Standard north-up GeoTIFF -> rasterio-style Affine:
    # x = x0 + (col - i) * scale_x
    # y = y0 - (row - j) * scale_y
    return Affine(
        scale_x,
        0.0,
        x0 - i * scale_x,
        0.0,
        -scale_y,
        y0 + j * scale_y,
    )


def _dtype_from_ifd(ifd: Any) -> np.dtype:
    """Derive a numpy dtype from sample_format and bits_per_sample tags."""
    sf = ifd.sample_format[0]
    bits = ifd.bits_per_sample[0]
    sf_val = int(sf) if not isinstance(sf, int) else sf
    kind = {1: "u", 2: "i", 3: "f"}.get(sf_val)
    if kind is None:
        raise NotImplementedError(f"Unsupported sample_format: {sf}")

    try:
        return np.dtype(f"{kind}{bits // 8}")
    except TypeError:
        raise NotImplementedError(f"Unsupported sample_format/bits: {sf}, {bits}")


def _nodata_from_ifd(ifd: Any, dtype: np.dtype) -> int | float | None:
    """Extract the GDAL_NODATA value from other_tags, coerced to match *dtype*."""
    other = getattr(ifd, "other_tags", None) or {}
    raw = other.get(_GDAL_NODATA_TAG)
    if raw is None:
        return None

    # Unwrap single-element lists (some TIFF writers nest the value).
    while isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]

    # Rational (numerator, denominator) tuple.
    if isinstance(raw, tuple) and len(raw) == 2:
        try:
            raw = float(raw[0]) / float(raw[1])
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    # Coerce to numeric.
    if isinstance(raw, str):
        s = raw.strip().strip("\x00")
        if not s:
            return None
        try:
            v: int | float = float("nan") if s.lower() in {"nan", "+nan", "-nan"} else float(s)
        except ValueError:
            return None
    elif isinstance(raw, (int, float)):
        v = raw
    else:
        return None

    # Cast to match the raster dtype.
    kind = np.dtype(dtype).kind
    if kind in {"i", "u"}:
        return None if (isinstance(v, float) and math.isnan(v)) else int(v)
    if kind == "f":
        return float(v)
    return v
