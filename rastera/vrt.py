"""Internal band-stack VRT support.

A *band-stack* VRT composes output bands from one or more source TIFFs, with
each ``<VRTRasterBand>`` driven by a single ``<SimpleSource>`` that names a
source file and a source band. All sources are assumed to describe the same
spatial image; the VRT's own geotransform, SRS, and raster size are ignored in
favour of the first source's metadata. More complex VRT features
(``<ComplexSource>``, multi-source bands, mosaicking via ``<SrcRect>`` /
``<DstRect>``) are out of scope and raise ``NotImplementedError``.
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
from .store import _fetch_descriptor_bytes, _join_relative_uri


@dataclass(frozen=True, slots=True)
class _VRTBand:
    """One output band of a band-stack VRT."""

    source_uri: str
    source_band: int  # 1-based into the source TIFF


async def _open_vrt(
    uri: str,
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> _VRTDataset:
    """Fetch + parse a VRT XML and open all referenced source TIFFs.

    ``store`` and ``**store_kwargs`` are forwarded unchanged to every source
    TIFF, which assumes all sources share the VRT's bucket/region/auth.
    Sources in a different bucket than the VRT are not supported via an
    explicit ``store``. ``store`` is also not used for the VRT XML fetch
    itself — that goes through obstore with a store built from
    ``store_kwargs`` — because the async-tiff and obstore store types are
    not interchangeable.
    """
    xml_bytes = await _fetch_descriptor_bytes(uri, **store_kwargs)
    bands = _parse_vrt_xml(xml_bytes, uri)

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


# ---- XML parsing & URI resolution ----


def _parse_vrt_xml(xml_bytes: bytes, vrt_uri: str) -> list[_VRTBand]:
    """Return one ``_VRTBand`` per ``<VRTRasterBand>`` in document order."""
    root = ET.fromstring(xml_bytes)
    if root.tag != "VRTDataset":
        raise ValueError(f"Not a VRT file (root tag {root.tag!r})")

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
