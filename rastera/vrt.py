"""Internal VRT support.

Two flavours are supported:

- *Band-stack* VRTs: each ``<VRTRasterBand>`` is driven by a single
  ``<SimpleSource>`` or ``<ComplexSource>`` naming a source file and band. All
  sources are assumed to describe the same spatial image, so pixels are read
  on the first source's grid and labelled with its SRS. The VRT's own raster
  size, geotransform, band ``dataType`` and SRS (as an EPSG code, where both
  have one) must agree with the sources. An omitted ``<GeoTransform>`` or
  ``<SRS>`` is taken from the first source, where GDAL would leave the VRT
  ungeoreferenced.

  Anything in the XML that contradicts that is rejected rather than ignored:
  by the ``_reject_*`` guards while parsing, and by
  ``_validate_source_windows`` once the sources are open.

- *Processed* VRTs (``VRTDataset subClass="VRTProcessedDataset"``): a single
  top-level ``<Input>`` plus a ``<ProcessingSteps>`` block. Only the one-step
  ``ReflectanceToDisplay``-style LUT pipeline is supported — one ``lut_N``
  argument per input band, output dtype Byte. This is what Airbus PNEO / SPOT
  / Pleiades ship as their *DISPLAY* VRT alongside the reflectance product.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from async_geotiff import RasterArray, Window
from pyproj import CRS
from pyproj.exceptions import CRSError

from . import config
from .geo import BBox
from .reader import (
    AsyncGeoTIFF,
    MetaOverrides,
    _CrsNodata,
    _GeoTIFFLike,
    _make_output_array,
    _source_store,
)
from .resampling import ResamplingMethod
from .store import (
    _check_source_uri,
    _fetch_descriptor_bytes,
    _join_relative_uri,
    _source_store_kwargs,
)


@dataclass(frozen=True, slots=True)
class _VRTBand:
    """One output band of a band-stack VRT.

    Only the source, its band and the nodata fields change what is read. The
    rest is checked against the opened sources, in ``_validate_source_windows``.
    """

    source_uri: str
    source_band: int  # 1-based
    # <SrcRect>/<DstRect> sizes, None when absent. Their offsets are already
    # checked to be 0.
    src_rect_size: tuple[float, float] | None = None
    dst_rect_size: tuple[float, float] | None = None
    # The VRT root's rasterXSize/rasterYSize, <GeoTransform> and <SRS> code,
    # the same on every band.
    vrt_declared_size: tuple[float, float] | None = None
    vrt_geotransform: Affine | None = None
    vrt_crs_epsg: int | None = None
    # GDAL reads an omitted dataType as Byte.
    data_type: str = "Byte"
    nodata: float | None = None
    # <HideNoDataValue>: GDAL still fills with `nodata` but reports none.
    hide_nodata: bool = False


@dataclass(frozen=True, slots=True)
class _VRTProcessedSpec:
    """A parsed processed VRT. ``luts`` is ``(output_count, _LUT_SIZE)``
    uint8, and ``lut_N`` maps source band N to output band N."""

    input_uri: str
    luts: np.ndarray
    src_nodata: int
    dst_nodata: int
    output_count: int


# URIs of the VRTs currently being opened on this task, outermost first.  A VRT
# source may itself be a VRT (``AsyncGeoTIFF.open`` routes ``.vrt`` back here),
# so without this a self- or mutually-referencing VRT recurses until
# RecursionError, issuing a network GET per level.
_vrt_open_stack: ContextVar[tuple[str, ...]] = ContextVar("_vrt_open_stack", default=())


async def _open_vrt(
    uri: str,
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> AsyncGeoTIFF:
    """Open a VRT, rejecting reference cycles. See :func:`_open_vrt_checked`."""
    stack = _vrt_open_stack.get()
    if uri in stack:
        raise ValueError(f"VRT reference cycle: {' -> '.join((*stack, uri))}")
    token = _vrt_open_stack.set((*stack, uri))
    try:
        return await _open_vrt_checked(
            uri,
            store=store,
            prefetch=prefetch,
            cache=cache,
            meta_overrides=meta_overrides,
            **store_kwargs,
        )
    finally:
        _vrt_open_stack.reset(token)


async def _open_vrt_checked(
    uri: str,
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> AsyncGeoTIFF:
    """Fetch and parse a VRT, and open its sources.

    ``store`` and ``store_kwargs`` go to every source. The XML itself is
    fetched through an obstore store built from ``store_kwargs``: an
    async-tiff store cannot serve that GET.
    """
    xml_bytes = await _fetch_descriptor_bytes(uri, **store_kwargs)
    parsed = _parse_vrt_xml(xml_bytes, uri)
    store_kwargs = _source_store_kwargs(uri, store_kwargs)

    if isinstance(parsed, _VRTProcessedSpec):
        source = await _open_vrt_source(
            parsed.input_uri,
            uri,
            store=store,
            prefetch=prefetch,
            cache=cache,
            meta_overrides=meta_overrides,
            **store_kwargs,
        )
        return _VRTProcessedDataset(uri, parsed, source, meta_overrides=meta_overrides)

    bands = parsed
    unique_uris = list(dict.fromkeys(b.source_uri for b in bands))
    sources_map: dict[str, AsyncGeoTIFF] = {}
    stores: dict[tuple[str, str | None], Any] = {}
    for u in unique_uris:
        sources_map[u] = await _open_vrt_source(
            u,
            uri,
            store=store
            if store is not None
            else _source_store(u, stores, cache, **store_kwargs),
            prefetch=prefetch,
            cache=cache,
            meta_overrides=meta_overrides,
            **store_kwargs,
        )
    _validate_source_windows(
        bands, sources_map, check_crs="crs" not in (meta_overrides or {})
    )
    return _VRTDataset(uri, bands, sources_map, meta_overrides=meta_overrides)


class _VRTDataset(AsyncGeoTIFF):
    """Band-stack VRT dataset presenting as an ``AsyncGeoTIFF``.

    ``_read_native`` is dispatched to the underlying sources, grouping bands
    that share a source into a single call. ``read()`` resamples that stack.
    """

    def __init__(
        self,
        uri: str,
        bands: Sequence[_VRTBand],
        sources_map: dict[str, AsyncGeoTIFF],
        *,
        meta_overrides: MetaOverrides | None = None,
    ):
        first = sources_map[bands[0].source_uri]
        super().__init__(uri, first._geotiff, meta_overrides=meta_overrides)
        # The source's resolved value, not its file's: for a nested VRT those
        # differ, and the file's 0 made merge paste over a hidden nodata.
        self._nodata = first._nodata
        # The first source's pyramid is not the others', so list none.
        self.overviews = []
        self._band_sources: list[tuple[AsyncGeoTIFF, int]] = [
            (sources_map[b.source_uri], b.source_band) for b in bands
        ]
        declared = _declared_nodata(bands)
        if declared is not None:
            self._override_nodata(declared)
        elif _hides_declared_nodata(bands):
            self._nodata = None

    def _override_nodata(self, nodata: float) -> None:
        """Adopt *nodata* and push it onto the sources, so one that fills
        pixels itself, as DIMAP does for a missing tile, fills with it.

        Mutating them is safe: each VRT open wraps its sources afresh, and
        only the inner ``GeoTIFF`` is cached. A processed-VRT source keeps the
        base method on purpose, since its nodata is post-LUT Byte and its
        child's is pre-LUT reflectance.
        """
        super()._override_nodata(nodata)
        for src in {id(s): s for s, _ in self._band_sources}.values():
            src._override_nodata(nodata)

    # With ``overviews`` and ``_nodata`` above, all a stack restates:
    # ``_validate_source_windows`` made band 1's ``_geotiff`` hold for every
    # source.
    @property
    def count(self) -> int:
        return len(self._band_sources)

    def _internal_mask_uri(self) -> str | None:
        # ``_geotiff`` is band 1's source only; a mask on any source counts.
        for src, _ in self._band_sources:
            uri = src._internal_mask_uri()
            if uri is not None:
                return uri
        return None

    async def read(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        bbox_crs: int | CRS | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        target_crs: int | CRS | None = None,
        target_resolution: float | None = None,
        snap_to_grid: bool = True,
        use_overviews: bool = False,
        resampling: ResamplingMethod = "nearest",
    ) -> RasterArray:
        if use_overviews:
            # Each source would pick its own overview level independently,
            # which can yield mismatched output shapes across sources.
            raise NotImplementedError("use_overviews is not supported on VRT datasets")
        # Not forwarded to each source: GDAL resamples the VRT's pixels with
        # the VRT's nodata, which is what the base read does with our stack.
        return await super().read(
            bbox=bbox,
            bbox_crs=bbox_crs,
            window=window,
            band_indices=band_indices,
            target_crs=target_crs,
            target_resolution=target_resolution,
            snap_to_grid=snap_to_grid,
            use_overviews=False,
            resampling=resampling,
        )

    async def _read_native(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        overview: Any | None = None,
        snap_to_grid: bool = True,
    ) -> RasterArray:
        if overview is not None:
            # A caller-supplied overview object is tied to one specific source
            # TIFF and cannot be reused across a VRT's multiple sources.
            raise NotImplementedError(
                "overview reads on VRT datasets are not supported"
            )
        # Internal entry: band_indices are already 0-based (or None for all).
        vrt_indices = (
            list(band_indices)
            if band_indices is not None
            else list(range(len(self._band_sources)))
        )
        return await _dispatch_source_reads(
            self._band_sources,
            vrt_indices,
            read_kwargs=dict(bbox=bbox, window=window, snap_to_grid=snap_to_grid),
            output_nodata=self._nodata,
        )

    def __repr__(self) -> str:
        n_sources = len({id(s) for s, _ in self._band_sources})
        return (
            f"_VRTDataset({self.uri}, bands={len(self._band_sources)}, "
            f"sources={n_sources})"
        )


class _VRTProcessedDataset(AsyncGeoTIFF):
    """Wraps a single input dataset and applies the per-band LUT on read.

    The LUT step is index-to-index — ``lut_N`` is applied to source band N
    and yields output band N. Band selection (``band_indices`` on
    ``_read_native``) is forwarded to the source unchanged; we apply only the
    LUTs matching the selected bands. ``read()`` resamples the LUT's output.
    """

    def __init__(
        self,
        uri: str,
        spec: _VRTProcessedSpec,
        source: AsyncGeoTIFF,
        *,
        meta_overrides: MetaOverrides | None = None,
    ):
        virtual = _processed_virtual_geotiff(
            source._geotiff,
            count=spec.output_count,
            dtype=np.dtype("uint8"),
            nodata=spec.dst_nodata,
        )
        super().__init__(uri, virtual, meta_overrides=meta_overrides)
        # Overview reads through the LUT are not supported, so list none.
        self.overviews = []
        self._spec = spec
        self._source = source

    @property
    def count(self) -> int:
        return self._spec.output_count

    def _internal_mask_uri(self) -> str | None:
        # ``_geotiff`` is synthesized; the source's header is the real one.
        return self._source._internal_mask_uri()

    async def read(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        bbox_crs: int | CRS | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        target_crs: int | CRS | None = None,
        target_resolution: float | None = None,
        snap_to_grid: bool = True,
        use_overviews: bool = False,
        resampling: ResamplingMethod = "nearest",
    ) -> RasterArray:
        if use_overviews:
            # Overview reads through the LUT aren't tested yet; same
            # consistency stance as ``_VRTDataset``.
            raise NotImplementedError(
                "use_overviews is not supported on processed VRT datasets"
            )
        # Not forwarded to the source: GDAL applies the LUT, then resamples its
        # output with ``dst_nodata``, which is what the base read does here.
        return await super().read(
            bbox=bbox,
            bbox_crs=bbox_crs,
            window=window,
            band_indices=band_indices,
            target_crs=target_crs,
            target_resolution=target_resolution,
            snap_to_grid=snap_to_grid,
            use_overviews=False,
            resampling=resampling,
        )

    async def _read_native(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        overview: Any | None = None,
        snap_to_grid: bool = True,
    ) -> RasterArray:
        if overview is not None:
            raise NotImplementedError(
                "overview reads on processed VRT datasets are not supported"
            )
        out_indices_0 = (
            list(band_indices)
            if band_indices is not None
            else list(range(self._spec.output_count))
        )
        src_result = await self._source._read_native(
            bbox=bbox,
            window=window,
            band_indices=out_indices_0,
            snap_to_grid=snap_to_grid,
        )
        return self._apply_luts(src_result, out_indices_0)

    def _apply_luts(
        self, src_result: RasterArray, band_indices_0: Sequence[int]
    ) -> RasterArray:
        in_data: np.ndarray[Any, Any] = src_result.data  # type: ignore[reportUnknownMemberType]
        if in_data.dtype.kind not in ("u", "i"):
            raise NotImplementedError(
                f"Processed VRT source dtype {in_data.dtype} is not integer; "
                f"only integer reflectance sources are supported"
            )
        # Two full passes over the array, so skip them for the dtypes whose
        # whole range already fits the LUT (uint8/uint16 — all that PNEO, SPOT
        # and Pleiades ship).
        if in_data.dtype.kind == "i" or in_data.dtype.itemsize > 2:
            if (
                int(in_data.max(initial=0)) >= _LUT_SIZE
                or int(in_data.min(initial=0)) < 0
            ):
                raise ValueError(
                    f"Processed VRT source has values outside "
                    f"[0, {_LUT_SIZE - 1}]; the LUT only covers that range"
                )
        out = np.empty(
            (len(band_indices_0), in_data.shape[1], in_data.shape[2]),
            dtype=np.uint8,
        )
        for i, b0 in enumerate(band_indices_0):
            out[i] = self._spec.luts[b0][in_data[i]]
        # The CRS is the sub-read's. The nodata is ours, since the LUT writes
        # ``dst_nodata``.
        return _make_output_array(
            out,
            src_result.transform,
            src_result.width,
            src_result.height,
            _CrsNodata(src_result.crs, self._nodata),
        )

    def __repr__(self) -> str:
        return (
            f"_VRTProcessedDataset({self.uri}, bands={self._spec.output_count}, "
            f"source={self._source.uri})"
        )


def _parse_vrt_xml(
    xml_bytes: bytes, vrt_uri: str
) -> list[_VRTBand] | _VRTProcessedSpec:
    """Dispatch on the VRT flavour and return the parsed spec.

    Band-stack VRTs return a list of ``_VRTBand`` (one per ``<VRTRasterBand>``
    in document order). Processed VRTs return a ``_VRTProcessedSpec``.
    """
    root = ET.fromstring(xml_bytes)
    if root.tag != "VRTDataset":
        raise ValueError(f"Not a VRT file (root tag {root.tag!r})")

    _reject_out_of_scope_georeferencing(root)

    subclass = root.attrib.get("subClass")
    if subclass == "VRTProcessedDataset":
        return _parse_processed_vrt(root, vrt_uri)
    if subclass == "VRTWarpedDataset":
        raise NotImplementedError(
            "Warped VRTs (VRTDataset subClass='VRTWarpedDataset') are out of "
            "scope for rastera: honouring one means reimplementing GDAL's "
            "warper. Use GDAL/rasterio for this VRT, or warp it to a COG "
            "first and read that."
        )
    if subclass:
        raise NotImplementedError(
            f"VRTDataset subClass={subclass!r} is not supported; only "
            f"plain band-stack VRTs and 'VRTProcessedDataset' are handled"
        )

    declared_size = _declared_raster_size(root)
    declared_transform = _declared_geotransform(root)
    declared_crs_epsg = _declared_crs_epsg(root)

    bands: list[_VRTBand] = []
    for vrt_band in root.findall("VRTRasterBand"):
        band_no = vrt_band.attrib.get("band", "?")
        _reject_derived_band(vrt_band, band_no)
        # Every kind of source counts, so the guards below also turn away one
        # this list would not name, such as <NoDataFromMaskSource>.
        sources = [child for child in vrt_band if child.tag.endswith("Source")]
        if not sources:
            raise ValueError(f"Malformed VRT band {band_no}: no source element")
        if len(sources) > 1:
            raise NotImplementedError(
                f"VRT band {band_no} has {len(sources)} sources; only "
                f"single-SimpleSource band-stack VRTs are supported"
            )
        src = sources[0]
        if src.tag not in ("SimpleSource", "ComplexSource"):
            raise NotImplementedError(
                f"VRT band {band_no} uses <{src.tag}>; only <SimpleSource> "
                f"and <ComplexSource> are supported"
            )
        source_uri = _source_filename_uri(
            src, vrt_uri, f"Malformed VRT band {band_no}: missing <SourceFilename>"
        )
        source_band = _source_band(src, band_no)
        band_nodata = _band_nodata(vrt_band, band_no)
        src_rect_size, dst_rect_size = _reject_unsupported_source(
            src, band_no, band_nodata
        )
        bands.append(
            _VRTBand(
                source_uri=source_uri,
                source_band=source_band,
                src_rect_size=src_rect_size,
                dst_rect_size=dst_rect_size,
                vrt_declared_size=declared_size,
                vrt_geotransform=declared_transform,
                vrt_crs_epsg=declared_crs_epsg,
                data_type=vrt_band.attrib.get("dataType", "Byte"),
                nodata=band_nodata,
                hide_nodata=_hides_nodata(vrt_band),
            )
        )

    if not bands:
        raise ValueError("VRT has no <VRTRasterBand> elements")
    # Dimension-free, so it belongs here with the other guards rather than in
    # _VRTDataset.__init__ — this rejects an unrepresentable VRT before any
    # source header is fetched. The value itself is read again at construction.
    _declared_nodata(bands)
    return bands


_GDAL_HINT = "Use GDAL/rasterio for this VRT, or translate it to a COG first."


def _reject_out_of_scope_georeferencing(root: ET.Element) -> None:
    """Reject GCP- or RPC-georeferenced VRTs. Ignored, the pixels would be
    placed by the source's geotransform.

    RPCs only when there is no ``<GeoTransform>``: with one, GDAL ignores
    them, and orthorectified products (PNEO, SPOT, Pleiades, Maxar, Planet)
    carry both, which ``gdal_translate -of VRT`` copies through.
    """
    if root.find("GCPList") is not None:
        raise NotImplementedError(
            f"GCP-georeferenced VRTs (<GCPList>) are out of scope for rastera: "
            f"honouring them means reimplementing GDAL's GCP transformer. "
            f"{_GDAL_HINT}"
        )
    if root.find("GeoTransform") is not None:
        return
    for md in root.findall("Metadata"):
        if md.attrib.get("domain") == "RPC":
            raise NotImplementedError(
                f"RPC-georeferenced VRTs (<Metadata domain='RPC'>) are out of "
                f"scope for rastera: honouring them means reimplementing "
                f"GDAL's RPC transformer. {_GDAL_HINT}"
            )


def _reject_derived_band(vrt_band: ET.Element, band_no: str) -> None:
    """Reject pixel-function bands, which would come back as the raw source
    pixels."""
    if (
        vrt_band.attrib.get("subClass") == "VRTDerivedRasterBand"
        or vrt_band.find("PixelFunctionType") is not None
    ):
        raise NotImplementedError(
            f"VRT band {band_no} is a pixel-function band "
            f"(VRTDerivedRasterBand / <PixelFunctionType>), which is out of "
            f"scope for rastera: honouring it means running GDAL's pixel "
            f"functions. {_GDAL_HINT}"
        )


def _declared_raster_size(root: ET.Element) -> tuple[float, float] | None:
    """The VRT's declared ``(rasterXSize, rasterYSize)``, or None if absent."""
    x = root.attrib.get("rasterXSize")
    y = root.attrib.get("rasterYSize")
    if x is None or y is None:
        return None
    try:
        return float(x), float(y)
    except ValueError as e:
        raise ValueError(f"VRT has malformed rasterXSize/rasterYSize: {e}") from e


def _declared_geotransform(root: ET.Element) -> Affine | None:
    """The VRT's declared ``<GeoTransform>``, or None if absent."""
    el = root.find("GeoTransform")
    if el is None or not (el.text or "").strip():
        return None
    try:
        coeffs = [float(v) for v in (el.text or "").split(",")]
    except ValueError as e:
        raise ValueError(f"VRT has a malformed <GeoTransform>: {e}") from e
    if len(coeffs) != 6:
        raise ValueError(
            f"VRT <GeoTransform> has {len(coeffs)} coefficients; expected 6"
        )
    return Affine.from_gdal(*coeffs)


def _declared_crs_epsg(root: ET.Element) -> int | None:
    """The EPSG code of the VRT's ``<SRS>``, or None when it has none, or no
    code can be found for it."""
    text = (root.findtext("SRS") or "").strip()
    if not text:
        return None
    try:
        return CRS.from_string(text).to_epsg()
    except CRSError:
        return None


def _rect(parent: ET.Element, tag: str) -> tuple[float, float, float, float] | None:
    """Parse a ``<SrcRect>``/``<DstRect>`` child into ``(xOff, yOff, xSize, ySize)``.

    ``None`` when absent. Compared exactly downstream: a supported rect is
    whole pixels, and a fractional one never matches, so it is rejected.
    """
    el = parent.find(tag)
    if el is None:
        return None
    try:
        return (
            float(el.attrib["xOff"]),
            float(el.attrib["yOff"]),
            float(el.attrib["xSize"]),
            float(el.attrib["ySize"]),
        )
    except KeyError as e:
        raise ValueError(f"Malformed <{tag}>: missing attribute {e}") from e
    except ValueError as e:
        raise ValueError(f"Malformed <{tag}>: non-numeric attribute ({e})") from e


# The children a source may carry and still copy its pixels through. Every
# other <ComplexSource> child transforms or masks values (<ScaleOffset>, <LUT>,
# <UseMaskBand>, ...) or changes how the source opens (<OpenOptions>). An
# allowlist, so a new GDAL element raises rather than being dropped.
_SIMPLE_SOURCE_CHILDREN = frozenset(
    {"SourceFilename", "SourceBand", "SourceProperties", "SrcRect", "DstRect", "NODATA"}
)


def _reject_unsupported_source(
    src: ET.Element, band_no: str, band_nodata: float | None
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Run the source checks that need no source size. Returns the
    ``<SrcRect>`` and ``<DstRect>`` sizes for ``_validate_source_windows``."""
    for child in src:
        if child.tag not in _SIMPLE_SOURCE_CHILDREN:
            raise NotImplementedError(
                f"VRT band {band_no} has a <{child.tag}> on its <{src.tag}>, "
                f"which asks for a per-pixel transform (rescaling, LUT "
                f"remapping, or masking) that rastera does not apply. "
                f"Reading it would silently return untransformed pixels. "
                f"{_GDAL_HINT}"
            )

    src_rect = _rect(src, "SrcRect")
    dst_rect = _rect(src, "DstRect")

    if src_rect is not None and (src_rect[0], src_rect[1]) != (0.0, 0.0):
        raise NotImplementedError(
            f"VRT band {band_no} has a <SrcRect> offset "
            f"(xOff={src_rect[0]:g}, yOff={src_rect[1]:g}); reading a windowed "
            f"region of a source is not supported. Only full-extent sources "
            f"are handled."
        )
    if dst_rect is not None and (dst_rect[0], dst_rect[1]) != (0.0, 0.0):
        raise NotImplementedError(
            f"VRT band {band_no} has a <DstRect> offset "
            f"(xOff={dst_rect[0]:g}, yOff={dst_rect[1]:g}); this VRT places its "
            f"source at an offset (mosaicking), which is not supported. Use "
            f"rastera.merge() to mosaic separate COGs instead."
        )
    if (
        src_rect is not None
        and dst_rect is not None
        and (src_rect[2], src_rect[3]) != (dst_rect[2], dst_rect[3])
    ):
        raise NotImplementedError(
            f"VRT band {band_no} rescales its source "
            f"({src_rect[2]:g}x{src_rect[3]:g} -> {dst_rect[2]:g}x{dst_rect[3]:g}); "
            f"resampling via <SrcRect>/<DstRect> is not supported. Pass "
            f"target_resolution to read() instead."
        )
    _reject_remapping_nodata(src, band_no, band_nodata)

    return (
        None if src_rect is None else (src_rect[2], src_rect[3]),
        None if dst_rect is None else (dst_rect[2], dst_rect[3]),
    )


def _source_band(src: ET.Element, band_no: str) -> int:
    """The source's ``<SourceBand>``, 1 when absent. Must be positive: 0
    indexes NumPy at -1, the source's last band."""
    el = src.find("SourceBand")
    if el is None or not el.text or not el.text.strip():
        return 1
    try:
        band = int(el.text.strip())
    except ValueError as e:
        raise ValueError(
            f"VRT band {band_no} has a non-integer <SourceBand>: {e}"
        ) from e
    if band < 1:
        raise ValueError(
            f"VRT band {band_no} has <SourceBand>{band}</SourceBand>; source "
            f"bands are 1-based, so this does not name a band."
        )
    return band


def _band_nodata(vrt_band: ET.Element, band_no: str) -> float | None:
    """The band's ``<NoDataValue>``, or None when absent."""
    el = vrt_band.find("NoDataValue")
    if el is None or not el.text or not el.text.strip():
        return None
    try:
        return float(el.text)
    except ValueError as e:
        raise ValueError(
            f"VRT band {band_no} has a malformed <NoDataValue>: {e}"
        ) from e


def _hides_nodata(vrt_band: ET.Element) -> bool:
    """Whether the band carries ``<HideNoDataValue>`` (``gdalbuildvrt
    -hidenodata``). GDAL still fills with the ``<NoDataValue>`` but reports
    none: on GDAL 3.12, ``<NoDataValue>100</NoDataValue>`` with the flag over a
    source ``<NODATA>50</NODATA>`` reads 50 back as 100, and ``gdalinfo`` shows
    no nodata.
    """
    el = vrt_band.find("HideNoDataValue")
    if el is None or not el.text or not el.text.strip():
        return False
    # GDAL writes 1 and reads the flag with CPLTestBool, to which these are the
    # false values, in any case.
    return el.text.strip().lower() not in ("0", "false", "no", "off")


def _reject_remapping_nodata(
    src: ET.Element, band_no: str, band_nodata: float | None
) -> None:
    """Reject a ``<ComplexSource>``'s ``<NODATA>`` that would remap pixels.

    GDAL fills the band with its ``<NoDataValue>`` (0 when unset), then copies
    the source over it, skipping pixels equal to ``<NODATA>``. With one
    full-extent source that changes nothing when the two are equal, which is
    what ``gdalbuildvrt -separate`` writes. When they differ rastera would
    return the unremapped value, so it raises. A ``<SimpleSource>`` never reads
    ``<NODATA>`` (verified on GDAL 3.12), so it passes.

    Known over-rejection: GDAL clamps a ``<NoDataValue>`` outside the band's
    type, so ``gdalbuildvrt -separate -vrtnodata -9999`` over Byte sources with
    nodata 0 reads the raw pixels, while this raises.
    """
    if src.tag != "ComplexSource":
        return
    el = src.find("NODATA")
    if el is None or not el.text or not el.text.strip():
        return
    try:
        source_nodata = float(el.text)
    except ValueError as e:
        raise ValueError(f"VRT band {band_no} has a malformed <NODATA>: {e}") from e

    fill = 0.0 if band_nodata is None else band_nodata
    if source_nodata == fill or (math.isnan(source_nodata) and math.isnan(fill)):
        return

    declared = (
        "the band declares no <NoDataValue>, so GDAL fills masked pixels with 0"
        if band_nodata is None
        else f"the band's <NoDataValue> is {band_nodata:g}"
    )
    raise NotImplementedError(
        f"VRT band {band_no} declares <NODATA>{source_nodata:g}</NODATA> on its "
        f"source but {declared}; GDAL would remap those pixels to {fill:g} and "
        f"rastera does not perform that compositing. Reading it would silently "
        f"return {source_nodata:g} instead. {_GDAL_HINT}"
    )


def _declared_nodata(bands: Sequence[_VRTBand]) -> float | None:
    """The nodata the VRT declares, or None when no band declares one.

    It wins over the sources'. A VRT over a DIMAP declares
    ``<NoDataValue>0</NoDataValue>`` where the DIMAP declares none, and taking
    the DIMAP's None made the black corners of a rotated footprint valid zeros,
    which ``merge`` pasted over a neighbour: 82% of a 128x128 window came back
    zero where GDAL returned imagery. Bands declaring none, or hiding theirs
    (``<HideNoDataValue>``), are skipped. Bands declaring different values
    raise, since rastera carries one.
    """
    declared = {b.nodata for b in bands if b.nodata is not None and not b.hide_nodata}
    # NaN is never equal to itself, so a NaN-nodata VRT would look like a
    # disagreement; collapse those first.
    non_nan = {v for v in declared if not math.isnan(v)}
    if len(declared) > len(non_nan):  # at least one NaN present
        if non_nan:
            raise NotImplementedError(
                f"VRT bands declare both NaN and {sorted(non_nan)} as "
                f"<NoDataValue>; rastera carries one nodata per dataset. "
                f"{_GDAL_HINT}"
            )
        return math.nan
    if len(non_nan) > 1:
        raise NotImplementedError(
            f"VRT bands declare differing <NoDataValue>s {sorted(non_nan)}; "
            f"rastera carries one nodata per dataset, so honouring them all "
            f"is not possible. {_GDAL_HINT}"
        )
    return next(iter(non_nan)) if non_nan else None


def _hides_declared_nodata(bands: Sequence[_VRTBand]) -> bool:
    """Whether every band declaring a ``<NoDataValue>`` also hides it.

    The VRT then reports none, as GDAL does, rather than its source's:
    ``-hidenodata`` makes the fill opaque background, and a reported nodata
    would let ``merge`` paste a neighbour through it. The sources keep theirs.
    """
    declaring = [b for b in bands if b.nodata is not None]
    return bool(declaring) and all(b.hide_nodata for b in declaring)


def _validate_source_windows(
    bands: Sequence[_VRTBand],
    sources_map: dict[str, AsyncGeoTIFF],
    *,
    check_crs: bool = True,
) -> None:
    """Check the opened sources really are the one full image the VRT implies.

    The checks that need each source's real size. GDAL's ``<SourceProperties>``
    is not trusted for it, since it may be absent or stale. *check_crs* is
    False when ``meta_overrides`` names the CRS, which then replaces the VRT's
    as well as the sources'.
    """

    def dims(src: AsyncGeoTIFF) -> tuple[float, float]:
        return float(src._geotiff.width), float(src._geotiff.height)

    reference = sources_map[bands[0].source_uri]
    ref_dims = dims(reference)
    ref_uri = bands[0].source_uri

    for i, band in enumerate(bands, start=1):
        src = sources_map[band.source_uri]
        src_dims = dims(src)

        if src_dims != ref_dims:
            raise NotImplementedError(
                f"VRT band {i} source {band.source_uri!r} is "
                f"{src_dims[0]:g}x{src_dims[1]:g} but band 1 source "
                f"{ref_uri!r} is {ref_dims[0]:g}x{ref_dims[1]:g}; "
                f"band-stack VRTs must reference sources of identical size."
            )
        # Equal size alone does not make two sources stackable: the bands are
        # written into one array typed from band 1 and returned under band 1's
        # transform, so a differing dtype would be silently cast and a differing
        # grid would mislabel the pixels' location.
        if src._geotiff.dtype != reference._geotiff.dtype:
            raise NotImplementedError(
                f"VRT band {i} source {band.source_uri!r} has dtype "
                f"{src._geotiff.dtype} but band 1 source {ref_uri!r} has "
                f"{reference._geotiff.dtype}; band-stack VRTs must reference "
                f"sources of identical dtype."
            )
        if src._crs_epsg != reference._crs_epsg:
            raise NotImplementedError(
                f"VRT band {i} source {band.source_uri!r} is EPSG:"
                f"{src._crs_epsg} but band 1 source {ref_uri!r} is EPSG:"
                f"{reference._crs_epsg}; band-stack VRTs must reference sources "
                f"in one CRS. Use rastera.merge() to combine differing CRSs."
            )
        if not _transforms_match(src._geotiff.transform, reference._geotiff.transform):
            raise NotImplementedError(
                f"VRT band {i} source {band.source_uri!r} has geotransform "
                f"{src._geotiff.transform!r} but band 1 source {ref_uri!r} has "
                f"{reference._geotiff.transform!r}; band-stack VRTs must "
                f"reference sources covering the same extent."
            )
        # GDAL looks the name up case-insensitively.
        declared_dtype = _GDAL_DTYPES.get(band.data_type.casefold())
        if declared_dtype != src._geotiff.dtype:
            # GDAL converts to the declared type, clamping or rounding: a Byte
            # VRT over UInt16 reads 999 as 255. An omitted dataType is Byte.
            raise NotImplementedError(
                f"VRT band {i} declares dataType={band.data_type!r} but its "
                f"source {band.source_uri!r} is {src._geotiff.dtype}; converting "
                f"a source to another type is not supported. {_GDAL_HINT}"
            )
        if band.src_rect_size is not None and band.src_rect_size != src_dims:
            raise NotImplementedError(
                f"VRT band {i} has a <SrcRect> of "
                f"{band.src_rect_size[0]:g}x{band.src_rect_size[1]:g} but its "
                f"source is {src_dims[0]:g}x{src_dims[1]:g}; reading a windowed "
                f"region of a source is not supported. Only full-extent "
                f"sources are handled."
            )
        if band.dst_rect_size is not None and band.dst_rect_size != src_dims:
            # Reachable when <DstRect> is the band's only rect: the parse-time
            # rescaling check needs both rects to spot a mismatch, and an
            # omitted <SrcRect> means GDAL reads the whole source.
            raise NotImplementedError(
                f"VRT band {i} has a <DstRect> of "
                f"{band.dst_rect_size[0]:g}x{band.dst_rect_size[1]:g} but its "
                f"source is {src_dims[0]:g}x{src_dims[1]:g}; this VRT scales its "
                f"source onto a differently sized output canvas, which is not "
                f"supported. Pass target_resolution to read() instead."
            )

    # A VRT that places its source elsewhere — gdal_translate -of VRT -a_ullr or
    # -a_gt — would otherwise be read at the source's location.
    declared_gt = bands[0].vrt_geotransform
    if declared_gt is not None and not _transforms_match(
        declared_gt, reference._geotiff.transform
    ):
        raise NotImplementedError(
            f"VRT declares geotransform {declared_gt.to_gdal()} but its source "
            f"{ref_uri!r} has {reference._geotiff.transform.to_gdal()}; a VRT "
            f"that georeferences its source anew is not supported. {_GDAL_HINT}"
        )

    # Likewise gdal_translate -of VRT -a_srs, read in the source's CRS. Compared
    # by EPSG code, so a WKT spelled differently is not a mismatch.
    declared_epsg = bands[0].vrt_crs_epsg
    source_epsg = reference._crs_epsg
    if (
        check_crs
        and declared_epsg is not None
        and source_epsg is not None
        and declared_epsg != source_epsg
    ):
        raise NotImplementedError(
            f"VRT declares EPSG:{declared_epsg} but its source {ref_uri!r} is "
            f"EPSG:{source_epsg}; a VRT that relabels its source's CRS is not "
            f"supported. Open it with meta_overrides={{'crs': {declared_epsg}}} "
            f"to read it in the VRT's CRS."
        )

    declared = bands[0].vrt_declared_size
    if declared is not None and declared != ref_dims:
        raise NotImplementedError(
            f"VRT declares a {declared[0]:g}x{declared[1]:g} raster but its "
            f"source is {ref_dims[0]:g}x{ref_dims[1]:g}; rastera reads sources "
            f"on their own grid and cannot resample them onto a different "
            f"declared canvas. Pass target_resolution to read(), or use "
            f"GDAL/rasterio for this VRT."
        )


# GDAL band type names, as a VRT's dataType attribute spells them, casefolded
# for the lookup.
_GDAL_DTYPES = {
    "byte": np.dtype("uint8"),
    "int8": np.dtype("int8"),
    "uint16": np.dtype("uint16"),
    "int16": np.dtype("int16"),
    "uint32": np.dtype("uint32"),
    "int32": np.dtype("int32"),
    "uint64": np.dtype("uint64"),
    "int64": np.dtype("int64"),
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "cfloat32": np.dtype("complex64"),
    "cfloat64": np.dtype("complex128"),
}

_VSI_SCHEMES = {"vsis3": "s3", "vsigs": "gs", "vsiaz": "az"}


def _resolve_source_uri(filename: str, relative_to_vrt: bool, vrt_uri: str) -> str:
    """Convert a ``<SourceFilename>`` value into a rastera-friendly URI.

    Handles ``relativeToVRT="1"`` (joined against the VRT's parent directory),
    GDAL's ``/vsis3/bucket/key`` and related ``/vsi…/`` prefixes, and
    ``/vsicurl/https://…`` HTTP passthroughs.
    """
    if filename.startswith("/vsicurl/"):
        return filename[len("/vsicurl/") :]

    if filename.startswith("/vsi"):
        # /vsis3/bucket/key -> s3://bucket/key
        tail = filename.lstrip("/")
        prefix, _, rest = tail.partition("/")
        scheme = _VSI_SCHEMES.get(prefix)
        if scheme is None:
            raise NotImplementedError(f"Unsupported VSI prefix in {filename!r}")
        bucket, _, key = rest.partition("/")
        return f"{scheme}://{bucket}/{key}"

    if relative_to_vrt:
        return _join_relative_uri(vrt_uri, filename)

    return filename


def _source_filename_uri(parent: ET.Element, vrt_uri: str, missing_msg: str) -> str:
    """Resolve *parent*'s ``<SourceFilename>``, raising *missing_msg* if absent."""
    el = parent.find("SourceFilename")
    if el is None or not el.text:
        raise ValueError(missing_msg)
    relative = el.attrib.get("relativeToVRT", "0") == "1"
    source_uri = _resolve_source_uri(el.text, relative, vrt_uri)
    _check_source_uri(source_uri, vrt_uri)
    return source_uri


async def _open_vrt_source(
    source_uri: str, vrt_uri: str, **open_kwargs: Any
) -> AsyncGeoTIFF:
    """Open one VRT source, naming the VRT and the source in async-tiff's
    error, which names neither. ``AsyncTiffException`` is matched by name, as
    async_tiff does not export it."""
    try:
        return await AsyncGeoTIFF.open(source_uri, **open_kwargs)
    except Exception as e:
        cls = type(e)
        if cls.__module__ != "async_tiff" or cls.__name__ != "AsyncTiffException":
            raise
        msg = str(e)
        hint = ""
        if "magic bytes" in msg and ("<" in msg or "xml" in msg.lower()):
            hint = (
                " Source looks like XML, not a TIFF — possibly an "
                "unrecognized GDAL descriptor format (rastera currently "
                "auto-detects DIMAP only)."
            )
        raise ValueError(
            f"VRT {vrt_uri!r} references source {source_uri!r} that could "
            f"not be opened as a TIFF: {msg}.{hint}"
        ) from e


async def _dispatch_source_reads(
    band_sources: Sequence[tuple[AsyncGeoTIFF, int]],
    vrt_indices: Sequence[int],
    *,
    read_kwargs: dict[str, Any],
    output_nodata: int | float | None,
) -> RasterArray:
    """Read each source once for all its bands, and reassemble them in VRT
    order.

    *vrt_indices* are 0-based into ``band_sources``, as are the source bands
    forwarded. *output_nodata* is the VRT's, which the result reports. It has
    no default: None would strip the sources' nodata.
    """
    groups: dict[int, tuple[AsyncGeoTIFF, list[tuple[int, int]]]] = {}
    for out_idx, vrt_idx in enumerate(vrt_indices):
        src, src_band = band_sources[vrt_idx]
        entry = groups.setdefault(id(src), (src, []))
        entry[1].append((out_idx, src_band - 1))

    group_list = list(groups.values())
    # Sequential unless set_concurrency(vrt=N) says otherwise.
    coros: list[Awaitable[RasterArray]] = [
        src._read_native(band_indices=[b for _, b in entries], **read_kwargs)
        for src, entries in group_list
    ]
    results: list[RasterArray] = await config._gather_bounded(
        config._vrt_concurrency, coros
    )

    first = results[0]
    first_data: np.ndarray[Any, Any] = first.data  # type: ignore[reportUnknownMemberType]
    out_data = np.empty(
        (len(vrt_indices), first.height, first.width),
        dtype=first_data.dtype,
    )
    for (_, entries), result in zip(group_list, results):
        res_data: np.ndarray[Any, Any] = result.data  # type: ignore[reportUnknownMemberType]
        if res_data.shape[1:] != first_data.shape[1:]:
            raise ValueError(
                "VRT sub-reads returned mismatched shapes; sources may not "
                "align spatially"
            )
        for i, (out_idx, _) in enumerate(entries):
            out_data[out_idx] = res_data[i]

    # The sub-read's own ``_geotiff``, unless its nodata is not the VRT's. The
    # push in ``_VRTDataset._override_nodata`` does not cover that: a source
    # whose dtype cannot carry the value refuses it, and a hidden one is never
    # pushed.
    geotiff_ref: Any = first._geotiff
    if output_nodata != first.nodata:
        geotiff_ref = _CrsNodata(first.crs, output_nodata)

    return _make_output_array(
        out_data,
        first.transform,
        first.width,
        first.height,
        geotiff_ref,
    )


# ---- Processed-VRT helpers ----


# The uint16 domain, which PNEO, SPOT and Pleiades reflectance uses: 64 KiB a
# band, and one fancy index per band on read.
_LUT_SIZE = 65536


def _parse_processed_vrt(root: ET.Element, vrt_uri: str) -> _VRTProcessedSpec:
    """Parse a ``VRTDataset subClass='VRTProcessedDataset'``."""
    input_el = root.find("Input")
    if input_el is None:
        raise ValueError("VRTProcessedDataset: missing <Input>")
    input_uri = _source_filename_uri(
        input_el, vrt_uri, "VRTProcessedDataset: missing <Input>/<SourceFilename>"
    )

    output_bands = root.findall("VRTRasterBand")
    if not output_bands:
        raise ValueError("VRTProcessedDataset: no <VRTRasterBand> elements")
    output_count = len(output_bands)

    output_dtypes = {b.attrib.get("dataType", "Byte") for b in output_bands}
    if output_dtypes != {"Byte"}:
        raise NotImplementedError(
            f"VRTProcessedDataset output dataType(s) {sorted(output_dtypes)} "
            f"not supported; only 'Byte' is implemented"
        )

    steps_el = root.find("ProcessingSteps")
    if steps_el is None:
        raise ValueError("VRTProcessedDataset: missing <ProcessingSteps>")
    steps = steps_el.findall("Step")
    if len(steps) != 1:
        raise NotImplementedError(
            f"VRTProcessedDataset has {len(steps)} <Step> elements; only "
            f"single-step LUT pipelines are supported"
        )
    step = steps[0]
    algo_el = step.find("Algorithm")
    algo = algo_el.text.strip() if algo_el is not None and algo_el.text else ""
    if algo != "LUT":
        raise NotImplementedError(
            f"VRTProcessedDataset step <Algorithm>{algo}</Algorithm> not "
            f"supported; only 'LUT' is implemented"
        )

    args: dict[str, str] = {}
    for arg in step.findall("Argument"):
        name = arg.attrib.get("name")
        if name and arg.text is not None:
            args[name] = arg.text

    src_nodata = _lut_nodata_arg(args, "src_nodata")
    dst_nodata = _lut_nodata_arg(args, "dst_nodata")
    if not 0 <= dst_nodata <= 255:
        raise ValueError(
            f"VRTProcessedDataset: dst_nodata={dst_nodata} is outside the "
            f"Byte output range [0, 255]"
        )

    luts = np.empty((output_count, _LUT_SIZE), dtype=np.uint8)
    for i in range(output_count):
        key = f"lut_{i + 1}"
        if key not in args:
            raise ValueError(
                f"VRTProcessedDataset: missing <Argument name='{key}'> for "
                f"output band {i + 1}"
            )
        luts[i] = _compile_lut(args[key], src_nodata=src_nodata, dst_nodata=dst_nodata)

    _reject_processed_nodata_mismatch(output_bands, dst_nodata)

    return _VRTProcessedSpec(
        input_uri=input_uri,
        luts=luts,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        output_count=output_count,
    )


def _lut_nodata_arg(args: dict[str, str], name: str) -> int:
    """Read the LUT step's ``src_nodata`` or ``dst_nodata`` as an integer.

    GDAL takes a missing ``src_nodata`` from the input band's nodata and a
    missing ``dst_nodata`` from ``src_nodata``. It compares a fractional
    ``src_nodata`` in double, so no integer pixel matches it. rastera copies
    none of this, so both raise.
    """
    if name not in args:
        raise NotImplementedError(
            f"VRTProcessedDataset: the LUT step has no <Argument name='{name}'>; "
            f"rastera needs both src_nodata and dst_nodata. {_GDAL_HINT}"
        )
    text = args[name].strip()
    try:
        value = float(text)
    except ValueError as e:
        raise ValueError(f"VRTProcessedDataset: bad {name}: {e}") from e
    # is_integer() is False for inf and nan too, so int() cannot overflow.
    if not value.is_integer():
        raise ValueError(f"VRTProcessedDataset: {name}={text!r} is not an integer")
    return int(value)


def _reject_processed_nodata_mismatch(
    output_bands: Sequence[ET.Element], dst_nodata: int
) -> None:
    """Reject a band ``<NoDataValue>`` other than the LUT's ``dst_nodata``.
    GDAL reports the band's while the LUT writes ``dst_nodata``, which rastera
    reports. Real display VRTs keep the two equal."""
    for i, band in enumerate(output_bands, start=1):
        declared = _band_nodata(band, str(i))
        if declared is None or declared == dst_nodata:
            continue
        raise NotImplementedError(
            f"VRTProcessedDataset band {i} declares <NoDataValue>{declared:g} "
            f"but the LUT step writes dst_nodata={dst_nodata} for masked "
            f"pixels; rastera cannot honour both. {_GDAL_HINT}"
        )


def _compile_lut(arg_text: str, *, src_nodata: int, dst_nodata: int) -> np.ndarray:
    """Compile ``"x0:y0,x1:y1,..."`` control points into a dense uint8 LUT.

    Linear between the points, clamped to the end values beyond them, and
    ``lut[src_nodata] = dst_nodata``. Rounds half up, as GDAL's Float64 to
    Byte conversion does: display LUTs step by 1 over even-width intervals, so
    .5 is common, and ``np.rint`` put about 3% of pixels one DN below GDAL.
    """
    pairs = [p.strip() for p in arg_text.strip().split(",") if p.strip()]
    if not pairs:
        raise ValueError("LUT argument is empty")
    xs: list[float] = []
    ys: list[float] = []
    for p in pairs:
        try:
            x_str, y_str = p.split(":")
            xs.append(float(x_str))
            ys.append(float(y_str))
        except ValueError as e:
            raise ValueError(f"LUT control point {p!r} is malformed: {e}") from e
    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    if np.any(np.diff(xs_arr) < 0):
        raise ValueError("LUT control points must be non-decreasing in x")
    grid = np.arange(_LUT_SIZE, dtype=np.float64)
    interp = np.interp(grid, xs_arr, ys_arr)
    # At a repeated x, np.interp returns the last y and GDAL the first.
    first = np.searchsorted(xs_arr, grid, side="left")
    at_x = xs_arr[np.minimum(first, xs_arr.size - 1)] == grid
    interp[at_x] = ys_arr[first[at_x]]
    lut = np.clip(np.floor(interp + 0.5), 0, 255).astype(np.uint8)
    if 0 <= src_nodata < _LUT_SIZE:
        lut[src_nodata] = np.uint8(dst_nodata)
    return lut


def _processed_virtual_geotiff(
    src_geotiff: _GeoTIFFLike,
    *,
    count: int,
    dtype: np.dtype,
    nodata: int | float | None,
) -> _GeoTIFFLike:
    """The source's header metadata, with the post-LUT dtype and count."""
    from .formats.dimap import _VirtualGeoTIFF

    return _VirtualGeoTIFF(
        crs=src_geotiff.crs,
        nodata=nodata,
        dtype=dtype,
        count=count,
        width=src_geotiff.width,
        height=src_geotiff.height,
        res=src_geotiff.res,
        bounds=src_geotiff.bounds,
        transform=src_geotiff.transform,
        overviews=tuple(src_geotiff.overviews),
    )


def _transforms_match(a: Affine, b: Affine) -> bool:
    """Whether two geotransforms describe the same grid.

    Exact float equality would reject sources whose transforms differ only by
    rounding between the GDAL runs that produced them.  The relative tolerance
    covers the scale and origin terms at any CRS unit; the pixel-scaled
    absolute one covers the rotation terms, which are 0.0 in one source and
    sometimes 1e-16 in another.
    """
    abs_tol = max(abs(float(b.a)), abs(float(b.e))) * 1e-6
    return all(
        math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=abs_tol)
        for x, y in zip(a[:6], b[:6])
    )
