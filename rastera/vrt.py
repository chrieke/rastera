"""Internal VRT support.

Two flavours are supported:

- *Band-stack* VRTs: each ``<VRTRasterBand>`` is driven by a single
  ``<SimpleSource>`` that names a source file and a source band. All sources
  are assumed to describe the same spatial image; the VRT's own geotransform,
  SRS, and raster size are ignored in favour of the first source's metadata.
  More complex VRT features (``<ComplexSource>``, multi-source bands,
  mosaicking via ``<SrcRect>`` / ``<DstRect>``) are out of scope and raise
  ``NotImplementedError``.

- *Processed* VRTs (``VRTDataset subClass="VRTProcessedDataset"``): a single
  top-level ``<Input>`` plus a ``<ProcessingSteps>`` block. Only the
  one-step ``ReflectanceToDisplay``-style LUT pipeline is supported — one
  ``lut_N`` argument per input band, output dtype Byte. This is what Airbus
  PNEO / SPOT / Pleiades ship as their *DISPLAY* VRT alongside the
  reflectance product.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from async_geotiff import RasterArray, Window
from pyproj import CRS

from . import config
from .geo import BBox, normalize_band_indices
from .reader import AsyncGeoTIFF, MetaOverrides, _make_output_array
from .resampling import ResamplingMethod
from .store import _fetch_descriptor_bytes, _join_relative_uri


@dataclass(frozen=True, slots=True)
class _VRTBand:
    """One output band of a band-stack VRT."""

    source_uri: str
    source_band: int  # 1-based into the source TIFF


@dataclass(frozen=True, slots=True)
class _VRTProcessedSpec:
    """A parsed processed-VRT: input descriptor + compiled per-band LUT.

    ``luts`` has shape ``(output_count, _LUT_SIZE)`` and dtype ``uint8``.
    ``output_count`` matches the number of ``<VRTRasterBand>`` entries.
    The LUT step is index-to-index: ``lut_N`` applies to source band N
    and produces output band N.
    """

    input_uri: str
    luts: np.ndarray
    src_nodata: int
    dst_nodata: int
    output_count: int


async def _open_vrt(
    uri: str,
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> AsyncGeoTIFF:
    """Fetch + parse a VRT XML and open all referenced source TIFFs.

    ``store`` and ``**store_kwargs`` are forwarded unchanged to every source
    TIFF, which assumes all sources share the VRT's bucket/region/auth.
    Sources in a different bucket than the VRT are not supported via an
    explicit ``store``. ``store`` is also not used for the VRT XML fetch
    itself — that goes through obstore with a store built from
    ``store_kwargs`` — because the async-tiff and obstore store types are
    not interchangeable.

    Returns either a ``_VRTDataset`` (band-stack) or a
    ``_VRTProcessedDataset`` (processed) depending on the VRT flavour.
    """
    xml_bytes = await _fetch_descriptor_bytes(uri, **store_kwargs)
    parsed = _parse_vrt_xml(xml_bytes, uri)

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
    # Sequential opens: header reads only, but kept consistent with the
    # rest of the rastera read path (see _dispatch_source_reads and
    # rastera/formats/dimap.py) which avoids stacking concurrent fan-out.
    sources_map: dict[str, AsyncGeoTIFF] = {}
    for u in unique_uris:
        sources_map[u] = await _open_vrt_source(
            u,
            uri,
            store=store,
            prefetch=prefetch,
            cache=cache,
            meta_overrides=meta_overrides,
            **store_kwargs,
        )
    return _VRTDataset(uri, bands, sources_map, meta_overrides=meta_overrides)


class _VRTDataset(AsyncGeoTIFF):
    """Band-stack VRT dataset presenting as an ``AsyncGeoTIFF``.

    Reads are dispatched to the underlying sources, grouping bands that share
    a source into a single ``read()`` call.
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
        self._band_sources: list[tuple[AsyncGeoTIFF, int]] = [
            (sources_map[b.source_uri], b.source_band) for b in bands
        ]

    @property
    def count(self) -> int:
        return len(self._band_sources)

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
        # Public entry: band_indices are 1-based (or None).
        vrt_indices = normalize_band_indices(band_indices, len(self._band_sources))
        return await _dispatch_source_reads(
            self._band_sources,
            vrt_indices,
            "read",
            # offset 0: src.read is public and takes 1-based band indices;
            # source_band values are already stored 1-based.
            source_band_offset=0,
            read_kwargs=dict(
                bbox=bbox,
                bbox_crs=bbox_crs,
                window=window,
                target_crs=target_crs,
                target_resolution=target_resolution,
                snap_to_grid=snap_to_grid,
                use_overviews=False,
                resampling=resampling,
            ),
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
            "_read_native",
            # offset -1: _read_native is internal and expects 0-based band
            # indices; stored source_band values are 1-based.
            source_band_offset=-1,
            read_kwargs=dict(bbox=bbox, window=window, snap_to_grid=snap_to_grid),
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
    ``read``/``_read_native``) is forwarded to the source unchanged; we
    apply only the LUTs matching the selected bands.
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
        super().__init__(uri, virtual, meta_overrides=meta_overrides)  # type: ignore[arg-type]
        self._spec = spec
        self._source = source

    @property
    def count(self) -> int:
        return self._spec.output_count

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
        # normalize_band_indices returns 0-based output band positions.
        # Forward to ``source.read`` (public API, 1-based) and apply the
        # matching LUT row per output band.
        out_indices_0 = normalize_band_indices(band_indices, self._spec.output_count)
        src_result = await self._source.read(
            bbox=bbox,
            bbox_crs=bbox_crs,
            window=window,
            band_indices=[b + 1 for b in out_indices_0],
            target_crs=target_crs,
            target_resolution=target_resolution,
            snap_to_grid=snap_to_grid,
            use_overviews=False,
            resampling=resampling,
        )
        return self._apply_luts(src_result, out_indices_0)

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
        if int(in_data.max(initial=0)) >= _LUT_SIZE or int(in_data.min(initial=0)) < 0:
            raise ValueError(
                f"Processed VRT source has values outside [0, {_LUT_SIZE - 1}]; "
                f"the LUT only covers that range"
            )
        out = np.empty(
            (len(band_indices_0), in_data.shape[1], in_data.shape[2]),
            dtype=np.uint8,
        )
        for i, b0 in enumerate(band_indices_0):
            out[i] = self._spec.luts[b0][in_data[i]]
        return _make_output_array(
            out,
            src_result.transform,
            src_result.width,
            src_result.height,
            self._geotiff,
        )

    def __repr__(self) -> str:
        return (
            f"_VRTProcessedDataset({self.uri}, bands={self._spec.output_count}, "
            f"source={self._source.uri})"
        )


# ---- XML parsing & URI resolution ----


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

    subclass = root.attrib.get("subClass")
    if subclass == "VRTProcessedDataset":
        return _parse_processed_vrt(root, vrt_uri)
    if subclass:
        raise NotImplementedError(
            f"VRTDataset subClass={subclass!r} is not supported; only "
            f"plain band-stack VRTs and 'VRTProcessedDataset' are handled"
        )

    source_tags = {
        "SimpleSource",
        "ComplexSource",
        "AveragedSource",
        "KernelFilteredSource",
    }
    bands: list[_VRTBand] = []
    for vrt_band in root.findall("VRTRasterBand"):
        band_no = vrt_band.attrib.get("band", "?")
        sources = [child for child in vrt_band if child.tag in source_tags]
        if not sources:
            raise ValueError(f"Malformed VRT band {band_no}: no source element")
        if len(sources) > 1:
            raise NotImplementedError(
                f"VRT band {band_no} has {len(sources)} sources; only "
                f"single-SimpleSource band-stack VRTs are supported"
            )
        src = sources[0]
        if src.tag != "SimpleSource":
            raise NotImplementedError(
                f"VRT band {band_no} uses <{src.tag}>; only <SimpleSource> "
                f"is supported"
            )
        filename_el = src.find("SourceFilename")
        if filename_el is None or not filename_el.text:
            raise ValueError(f"Malformed VRT band {band_no}: missing <SourceFilename>")
        relative = filename_el.attrib.get("relativeToVRT", "0") == "1"
        source_uri = _resolve_source_uri(filename_el.text, relative, vrt_uri)
        source_band_el = src.find("SourceBand")
        source_band = (
            int(source_band_el.text)
            if source_band_el is not None and source_band_el.text
            else 1
        )
        bands.append(_VRTBand(source_uri=source_uri, source_band=source_band))

    if not bands:
        raise ValueError("VRT has no <VRTRasterBand> elements")
    return bands


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


async def _open_vrt_source(
    source_uri: str, vrt_uri: str, **open_kwargs: Any
) -> AsyncGeoTIFF:
    """Open one VRT source, rewrapping async_tiff failures with VRT context.

    ``AsyncGeoTIFF.open`` already routes recognized descriptor formats
    (currently DIMAP) into their own readers. Anything else that isn't
    a TIFF lands here as a bare ``AsyncTiffException`` that names
    neither the source URI nor the VRT — we catch that by class name
    (it is not importable from ``async_tiff``) and re-raise with
    context so the user learns which VRT + which source failed.
    """
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
    method_name: str,
    *,
    source_band_offset: int,
    read_kwargs: dict[str, Any],
) -> RasterArray:
    """Group output bands by source, invoke *method_name* on each source with
    the bundled source-band list, and reassemble into VRT output order.

    *vrt_indices* are 0-based indices into ``band_sources``.
    *source_band_offset* is added to each source's stored (1-based) source band
    before it is forwarded — 0 for public ``read`` (keeps 1-based), -1 for
    internal ``_read_native`` (converts to 0-based).
    """
    # Group output bands by source while preserving output order within each group.
    groups: dict[int, tuple[AsyncGeoTIFF, list[tuple[int, int]]]] = {}
    for out_idx, vrt_idx in enumerate(vrt_indices):
        src, src_band = band_sources[vrt_idx]
        entry = groups.setdefault(id(src), (src, []))
        entry[1].append((out_idx, src_band + source_band_offset))

    group_list = list(groups.values())
    # Sequential by default: each source read already fans out internally
    # (and DIMAP sources fan out further per-tile), so stacking outer
    # concurrency multiplies the HTTP burst without adding throughput on
    # saturated links. Set ``rastera.set_concurrency(vrt=N>1)`` to opt
    # into outer fan-out across distinct sources.
    coros: list[Awaitable[RasterArray]] = [
        getattr(src, method_name)(band_indices=[b for _, b in entries], **read_kwargs)
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

    return _make_output_array(
        out_data,
        first.transform,
        first.width,
        first.height,
        first._geotiff,
    )


# ---- Processed-VRT helpers ----


# A dense uint16-domain LUT is the largest size that fits the integer
# reflectance sources we see (PNEO/SPOT/Pleiades are uint16). 65 536 bytes
# per band × 6 bands ≈ 384 KB — negligible memory, and the LUT lookup
# becomes a single ``np.ndarray.__getitem__`` per band on read.
_LUT_SIZE = 65536


def _parse_processed_vrt(
    root: ET.Element, vrt_uri: str
) -> _VRTProcessedSpec:
    """Parse a ``VRTDataset subClass='VRTProcessedDataset'``."""
    input_el = root.find("Input")
    if input_el is None:
        raise ValueError("VRTProcessedDataset: missing <Input>")
    filename_el = input_el.find("SourceFilename")
    if filename_el is None or not filename_el.text:
        raise ValueError(
            "VRTProcessedDataset: missing <Input>/<SourceFilename>"
        )
    relative = filename_el.attrib.get("relativeToVRT", "0") == "1"
    input_uri = _resolve_source_uri(filename_el.text, relative, vrt_uri)

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

    try:
        src_nodata = int(float(args.get("src_nodata", "0")))
        dst_nodata = int(float(args.get("dst_nodata", "0")))
    except ValueError as e:
        raise ValueError(f"VRTProcessedDataset: bad src/dst nodata: {e}") from e
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

    return _VRTProcessedSpec(
        input_uri=input_uri,
        luts=luts,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        output_count=output_count,
    )


def _compile_lut(
    arg_text: str, *, src_nodata: int, dst_nodata: int
) -> np.ndarray:
    """Compile a ``"x0:y0,x1:y1,..."`` control-point string into a dense LUT.

    Returns a ``uint8`` array of length ``_LUT_SIZE`` whose ``lut[v]`` is
    the display-byte value for source value ``v`` (piecewise-linear
    interpolation between control points; clamped to the first / last
    output beyond the table). ``lut[src_nodata]`` is forced to
    ``dst_nodata`` so source nodata always maps cleanly even if a future
    control-point table omits it.
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
    lut = np.clip(np.rint(interp), 0, 255).astype(np.uint8)
    if 0 <= src_nodata < _LUT_SIZE:
        lut[src_nodata] = np.uint8(dst_nodata)
    return lut


def _processed_virtual_geotiff(
    src_geotiff: Any, *, count: int, dtype: np.dtype, nodata: int | float | None
) -> Any:
    """Wrap the source's ``_geotiff`` metadata with our output dtype/count.

    Reuses ``_DIMAPDataset``'s ``_VirtualGeoTIFF`` (lazy import to avoid the
    formats subpackage at module load) so spatial metadata flows through
    ``AsyncGeoTIFF.__init__`` unchanged while dtype and band count reflect
    the post-LUT output.
    """
    from .formats.dimap import _VirtualGeoTIFF

    return _VirtualGeoTIFF(
        crs=src_geotiff.crs,
        nodata=nodata,
        dtype=dtype,
        count=count,
        width=src_geotiff.width,
        height=src_geotiff.height,
        res=tuple(src_geotiff.res),
        bounds=tuple(src_geotiff.bounds),
        transform=src_geotiff.transform,
        overviews=tuple(getattr(src_geotiff, "overviews", ()) or ()),
    )
