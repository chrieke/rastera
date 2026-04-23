"""Internal DIMAP (Dimap_Document) support.

DIMAP is the descriptor format used by Airbus Pléiades / PNEO / SPOT
deliveries. One ``DIM_*.XML`` file declares a virtual raster whose pixels
live across many sibling TIFF tiles, grouped by band-group (e.g. RGB and
NED) and laid out on a regular row/column tile grid. GDAL's DIMAP driver
presents all those files as one dataset; this module does the same for
rastera, transparently, so callers can open a DIMAP (or a VRT that wraps
one) through the normal ``AsyncGeoTIFF.open`` entry point.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from async_geotiff import RasterArray, Window
from pyproj import CRS

from ..geo import BBox, ensure_bbox, window_from_bbox
from ..reader import AsyncGeoTIFF, MetaOverrides, _make_output_array
from ..store import _fetch_descriptor_bytes, _join_relative_uri


@dataclass(frozen=True, slots=True)
class _DIMAPBand:
    """One virtual output band.

    ``group_index`` selects a ``_DIMAPBandGroup`` (i.e. the set of tile
    TIFFs the band lives in), and ``source_band`` is the 1-based band
    index to read *within each of those tiles*.
    """

    band_id: str
    band_name: str
    group_index: int
    source_band: int


@dataclass(frozen=True, slots=True)
class _DIMAPBandGroup:
    """A set of tile TIFFs that together cover the full extent for some bands.

    ``tile_paths`` maps (tile_row, tile_col), both 1-based, to the file name
    as written in the DIMAP (relative to the DIMAP XML's parent directory).
    """

    tile_paths: dict[tuple[int, int], str]


@dataclass(frozen=True, slots=True)
class _DIMAPLayout:
    """Fully-parsed DIMAP geometry.

    Immutable. All I/O happens elsewhere; this is the hand-off between
    ``_parse_dimap_xml`` and the mosaic read engine.
    """

    width: int
    height: int
    crs_epsg: int
    transform: Affine
    dtype: np.dtype
    tile_rows: int
    tile_cols: int
    tile_width: int
    tile_height: int
    groups: tuple[_DIMAPBandGroup, ...]
    bands: tuple[_DIMAPBand, ...]


def _parse_dimap_xml(xml_bytes: bytes) -> _DIMAPLayout:
    """Parse a DIMAP ``Dimap_Document`` into a ``_DIMAPLayout``.

    Only the subset used for read dispatch is extracted. The MVP scope is:

    - ``DATA_FILE_ORGANISATION = BAND_COMPOSITE`` (one TIFF per tile per
      band-group; bands packed inside the TIFF).
    - ``DATA_FILE_FORMAT = image/tiff``.
    - ``Regular_Tiling`` with zero overlap.

    Anything else raises ``NotImplementedError`` so callers get a clear
    error instead of silent misreads.
    """
    root = ET.fromstring(xml_bytes)
    if root.tag != "Dimap_Document":
        raise ValueError(f"Not a DIMAP file (root tag {root.tag!r})")

    raster_data = _require(root, "Raster_Data")
    dims = _require(raster_data, "Raster_Dimensions")
    width = int(_require_text(dims, "NCOLS"))
    height = int(_require_text(dims, "NROWS"))

    tile_rows, tile_cols, tile_h, tile_w = _parse_regular_tiling(dims)

    data_access = _require(raster_data, "Data_Access")
    organisation = _require_text(data_access, "DATA_FILE_ORGANISATION")
    if organisation != "BAND_COMPOSITE":
        raise NotImplementedError(
            f"DIMAP DATA_FILE_ORGANISATION={organisation!r}; only "
            f"'BAND_COMPOSITE' is supported"
        )
    fmt = _require_text(data_access, "DATA_FILE_FORMAT")
    if fmt != "image/tiff":
        raise NotImplementedError(
            f"DIMAP DATA_FILE_FORMAT={fmt!r}; only 'image/tiff' is supported"
        )

    groups, bands = _parse_band_groups(data_access)

    transform = _parse_transform(_require(root, "Geoposition"))
    dtype = _parse_dtype(_require(raster_data, "Raster_Encoding"))
    crs_epsg = _parse_crs_epsg(_require(root, "Coordinate_Reference_System"))

    return _DIMAPLayout(
        width=width,
        height=height,
        crs_epsg=crs_epsg,
        transform=transform,
        dtype=dtype,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tile_width=tile_w,
        tile_height=tile_h,
        groups=tuple(groups),
        bands=tuple(bands),
    )


@dataclass(frozen=True, slots=True)
class _VirtualGeoTIFF:
    """Metadata-only stand-in for an ``async_geotiff.GeoTIFF``.

    Exposes just the attribute surface that ``AsyncGeoTIFF.__init__`` and
    its reproject/resample helpers read off ``self._geotiff`` — nothing
    more. This lets a synthesized dataset (``_DIMAPDataset``) pass
    ``super().__init__`` and reuse the full read dispatch without
    holding a real TIFF. Never participates in actual I/O; the dataset
    overrides ``_read_native`` before any read path would try to.
    """

    crs: CRS
    nodata: int | float | None
    dtype: np.dtype
    count: int
    width: int
    height: int
    res: tuple[float, float]
    bounds: tuple[float, float, float, float]
    transform: Affine
    overviews: tuple[Any, ...] = ()


class _DIMAPDataset(AsyncGeoTIFF):
    """Read adapter for a DIMAP descriptor.

    Presents as an ``AsyncGeoTIFF`` with synthesized metadata derived
    from the DIMAP XML; individual tile TIFFs are opened lazily on
    first read. Overview reads are rejected because each tile TIFF
    carries its own overview pyramid and those pyramids cannot safely
    mix across tiles (same constraint as ``_VRTDataset``).
    """

    def __init__(
        self,
        uri: str,
        layout: _DIMAPLayout,
        *,
        store: Any = None,
        prefetch: int = 32768,
        cache: bool = True,
        store_kwargs: dict[str, Any] | None = None,
        meta_overrides: MetaOverrides | None = None,
        first_tile: AsyncGeoTIFF | None = None,
        first_tile_key: tuple[int, int, int] | None = None,
    ):
        # DIMAP XML has no canonical nodata value. When the caller hands
        # us a pre-opened tile (the normal path from ``_maybe_open_dimap``),
        # inherit that tile's TIFF-level nodata so the mosaic pre-fill
        # and per-tile reads agree on the sentinel. Otherwise fall back
        # to 0 — the Airbus convention for integer products.
        nodata: int | float | None = (
            first_tile._nodata if first_tile is not None else 0
        )
        virtual = _virtual_geotiff_for(layout, nodata=nodata)
        super().__init__(uri, virtual, meta_overrides=meta_overrides)  # type: ignore[arg-type]
        self._layout = layout
        self._tile_open_kwargs: dict[str, Any] = {
            "store": store,
            "prefetch": prefetch,
            "cache": cache,
            **(store_kwargs or {}),
        }
        # Single-flight lazy tile opens keyed by (group_idx, tile_row, tile_col).
        self._tile_tasks: dict[
            tuple[int, int, int], asyncio.Future[AsyncGeoTIFF]
        ] = {}
        if first_tile is not None:
            # Prime the cache so the first read doesn't re-fetch this tile.
            assert first_tile_key is not None
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[AsyncGeoTIFF] = loop.create_future()
            fut.set_result(first_tile)
            self._tile_tasks[first_tile_key] = fut

    async def read(self, *args: Any, use_overviews: bool = False, **kwargs: Any):
        if use_overviews:
            raise NotImplementedError(
                "use_overviews is not supported on DIMAP datasets"
            )
        return await super().read(*args, use_overviews=False, **kwargs)

    async def _read_native(
        self,
        bbox: BBox | tuple[float, float, float, float] | None = None,
        window: Window | None = None,
        band_indices: Sequence[int] | None = None,
        overview: Any | None = None,
        snap_to_grid: bool = True,
    ) -> RasterArray:
        """Stitch the requested window from per-tile, per-group TIFF reads.

        Reads all needed tiles concurrently. Output bands are written back
        in the *caller's* requested order, so non-contiguous selections
        like ``[5, 0]`` round-trip correctly instead of getting sorted
        into group order.
        """
        if overview is not None:
            raise NotImplementedError(
                "overview reads on DIMAP datasets are not supported"
            )

        layout = self._layout
        if bbox is None and window is None:
            bbox = BBox(*self._geotiff.bounds)
        if window is None:
            assert bbox is not None
            window = window_from_bbox(self._geotiff, bbox)

        if band_indices is None:
            indices_0 = list(range(len(layout.bands)))
        else:
            indices_0 = list(band_indices)

        # Pre-fill with nodata so pixels past the mosaic edge — or any tile
        # that legitimately returned no data — stay at the sentinel value.
        nodata = self._nodata if self._nodata is not None else 0
        out = np.full(
            (len(indices_0), window.height, window.width),
            nodata,
            dtype=layout.dtype,
        )

        # Group the requested output bands by their source band-group, so
        # each tile is read at most once per call even when several output
        # bands share the same group.
        per_group: dict[int, list[tuple[int, int]]] = {}
        for out_pos, bi in enumerate(indices_0):
            b = layout.bands[bi]
            per_group.setdefault(b.group_index, []).append((out_pos, b.source_band))

        tile_reads = _tile_decomposition(layout, window)

        async def _read_one(
            group_idx: int, tr: _TileRead, src_bands_0: list[int]
        ) -> tuple[_TileRead, np.ndarray]:
            tile_ds = await self._get_tile(group_idx, tr.tile_row, tr.tile_col)
            result = await tile_ds._read_native(
                window=tr.src_window, band_indices=src_bands_0
            )
            return tr, result.data

        jobs: list[
            tuple[int, _TileRead, list[int], list[int]]
        ] = []
        for group_idx, entries in per_group.items():
            out_positions = [e[0] for e in entries]
            src_bands_0 = [e[1] - 1 for e in entries]
            for tr in tile_reads:
                jobs.append((group_idx, tr, out_positions, src_bands_0))

        results = await asyncio.gather(
            *(_read_one(g, tr, sb) for g, tr, _, sb in jobs)
        )

        for (_, _, out_positions, _), (tr, data) in zip(jobs, results):
            # data has shape (len(src_bands_0), tile_h, tile_w); paste each
            # band into its caller-requested output slot.
            for i, pos in enumerate(out_positions):
                out[pos, tr.dst_rows, tr.dst_cols] = data[i]

        out_transform = layout.transform * Affine.translation(
            window.col_off, window.row_off
        )
        if bbox is not None and not snap_to_grid:
            bbox = ensure_bbox(bbox)
            res = layout.transform.a
            out_transform = Affine(res, 0, bbox.minx, 0, -res, bbox.maxy)

        return _make_output_array(
            out, out_transform, window.width, window.height, self._geotiff
        )

    async def _get_tile(
        self, group_idx: int, tile_row: int, tile_col: int
    ) -> AsyncGeoTIFF:
        """Lazy, single-flight tile open. Subsequent callers await the
        same Future rather than re-fetching the TIFF header. A pre-opened
        tile primed by the constructor is returned immediately."""
        key = (group_idx, tile_row, tile_col)
        fut = self._tile_tasks.get(key)
        if fut is None:
            fut = asyncio.ensure_future(
                self._open_tile(group_idx, tile_row, tile_col)
            )
            self._tile_tasks[key] = fut
        return await fut

    async def _open_tile(
        self, group_idx: int, tile_row: int, tile_col: int
    ) -> AsyncGeoTIFF:
        href = self._layout.groups[group_idx].tile_paths[(tile_row, tile_col)]
        tile_uri = _resolve_tile_uri(href, self.uri)
        return await AsyncGeoTIFF.open(tile_uri, **self._tile_open_kwargs)

    def __repr__(self) -> str:
        return (
            f"_DIMAPDataset({self.uri}, "
            f"width={self._layout.width}, height={self._layout.height}, "
            f"bands={len(self._layout.bands)}, "
            f"tiles={self._layout.tile_rows}x{self._layout.tile_cols}x"
            f"{len(self._layout.groups)}groups)"
        )


async def _maybe_open_dimap(
    uri: str,
    *,
    store: Any = None,
    prefetch: int = 32768,
    cache: bool = True,
    meta_overrides: MetaOverrides | None = None,
    **store_kwargs: Any,
) -> _DIMAPDataset | None:
    """Return a ``_DIMAPDataset`` when *uri* is a DIMAP XML, else ``None``.

    Used by ``AsyncGeoTIFF.open``'s ``.xml``-suffix branch. On a miss
    (any other XML flavour) returns ``None`` so the caller can fall
    through to the normal TIFF open path — whose magic-bytes error then
    surfaces the real "this isn't a TIFF" explanation.

    On a hit, eagerly opens the first tile (lowest tile_R/tile_C of
    band-group 0) to learn the real TIFF nodata value and prime the
    tile cache. The DIMAP XML itself carries no nodata value.
    """
    xml_bytes = await _fetch_descriptor_bytes(uri, **store_kwargs)
    # Cheap structural sniff. DIMAPs put the root element in the first
    # kilobyte; scanning the whole document here would waste CPU on
    # large non-DIMAP XMLs that happen to share the extension.
    if b"Dimap_Document" not in xml_bytes[:2048]:
        return None
    layout = _parse_dimap_xml(xml_bytes)
    tile_open_kwargs: dict[str, Any] = {
        "store": store,
        "prefetch": prefetch,
        "cache": cache,
        **store_kwargs,
    }
    first_key, first_tile = await _sniff_first_tile(
        layout, uri, tile_open_kwargs
    )
    return _DIMAPDataset(
        uri,
        layout,
        store=store,
        prefetch=prefetch,
        cache=cache,
        store_kwargs=store_kwargs,
        meta_overrides=meta_overrides,
        first_tile=first_tile,
        first_tile_key=first_key,
    )


async def _sniff_first_tile(
    layout: _DIMAPLayout,
    uri: str,
    tile_open_kwargs: dict[str, Any],
) -> tuple[tuple[int, int, int], AsyncGeoTIFF]:
    """Open the lowest-indexed tile of band-group 0. Factored out so
    tests can patch tile-level I/O at open time without stubbing the
    whole ``AsyncGeoTIFF.open`` classmethod."""
    (r, c) = min(layout.groups[0].tile_paths)
    href = layout.groups[0].tile_paths[(r, c)]
    tile_uri = _resolve_tile_uri(href, uri)
    tile = await AsyncGeoTIFF.open(tile_uri, **tile_open_kwargs)
    return (0, r, c), tile


# ---- helpers ----


def _require(parent: ET.Element, tag: str) -> ET.Element:
    el = parent.find(tag)
    if el is None:
        raise ValueError(f"DIMAP: missing <{tag}> under <{parent.tag}>")
    return el


def _require_text(parent: ET.Element, tag: str) -> str:
    el = _require(parent, tag)
    if el.text is None or not el.text.strip():
        raise ValueError(f"DIMAP: empty <{tag}> under <{parent.tag}>")
    return el.text.strip()


def _parse_regular_tiling(dims: ET.Element) -> tuple[int, int, int, int]:
    tile_set = _require(dims, "Tile_Set")
    regular = tile_set.find("Regular_Tiling")
    if regular is None:
        raise NotImplementedError(
            "DIMAP Tile_Set has no <Regular_Tiling>; irregular tilings are "
            "not supported"
        )
    size = _require(regular, "NTILES_SIZE")
    count = _require(regular, "NTILES_COUNT")
    overlap = regular.find("NTILES_OVERLAP")
    if overlap is not None:
        o_cols = int(overlap.attrib.get("ncols", "0"))
        o_rows = int(overlap.attrib.get("nrows", "0"))
        if o_cols != 0 or o_rows != 0:
            raise NotImplementedError(
                f"DIMAP NTILES_OVERLAP=({o_cols},{o_rows}); only zero-overlap "
                f"tilings are supported"
            )
    tile_w = int(size.attrib["ncols"])
    tile_h = int(size.attrib["nrows"])
    n_cols = int(count.attrib["ntiles_C"])
    n_rows = int(count.attrib["ntiles_R"])
    return n_rows, n_cols, tile_h, tile_w


def _parse_band_groups(
    data_access: ET.Element,
) -> tuple[list[_DIMAPBandGroup], list[_DIMAPBand]]:
    groups: list[_DIMAPBandGroup] = []
    bands: list[_DIMAPBand] = []
    data_file_groups = data_access.findall("Data_Files")
    if not data_file_groups:
        raise ValueError("DIMAP: no <Data_Files> groups under <Data_Access>")
    for group_index, df_group in enumerate(data_file_groups):
        tile_paths: dict[tuple[int, int], str] = {}
        for df in df_group.findall("Data_File"):
            try:
                r = int(df.attrib["tile_R"])
                c = int(df.attrib["tile_C"])
            except KeyError as e:
                raise ValueError(
                    f"DIMAP: <Data_File> missing {e.args[0]} attribute"
                ) from e
            path_el = _require(df, "DATA_FILE_PATH")
            href = path_el.attrib.get("href")
            if not href:
                raise ValueError("DIMAP: <DATA_FILE_PATH> missing href attribute")
            tile_paths[(r, c)] = href
        if not tile_paths:
            raise ValueError(
                f"DIMAP: <Data_Files> group {group_index} has no <Data_File> entries"
            )
        groups.append(_DIMAPBandGroup(tile_paths=tile_paths))

        index_list = df_group.find("Raster_Display/Raster_Index_List")
        if index_list is None:
            raise ValueError(
                f"DIMAP: group {group_index} has no <Raster_Index_List>"
            )
        group_bands = [
            _DIMAPBand(
                band_id=_require_text(ri, "BAND_ID"),
                band_name=(ri.findtext("BAND_NAME") or "").strip(),
                group_index=group_index,
                source_band=int(_require_text(ri, "BAND_INDEX")),
            )
            for ri in index_list.findall("Raster_Index")
        ]
        if not group_bands:
            raise ValueError(
                f"DIMAP: group {group_index} has no <Raster_Index> entries"
            )
        bands.extend(group_bands)
    return groups, bands


def _parse_transform(geoposition: ET.Element) -> Affine:
    insert = geoposition.find("Geoposition_Insert")
    if insert is None:
        raise NotImplementedError(
            "DIMAP Geoposition has no <Geoposition_Insert>; only inserted "
            "geopositioning is supported"
        )
    ulx = float(_require_text(insert, "ULXMAP"))
    uly = float(_require_text(insert, "ULYMAP"))
    xdim = float(_require_text(insert, "XDIM"))
    ydim = float(_require_text(insert, "YDIM"))
    # PIXEL_ORIENTATION is "UL" by DIMAP convention: ULXMAP/ULYMAP are the
    # upper-left corner of the upper-left pixel, and YDIM is pixel height
    # (positive), so the north-up y-step is negative.
    return Affine(xdim, 0.0, ulx, 0.0, -ydim, uly)


def _parse_dtype(encoding: ET.Element) -> np.dtype:
    dt = _require_text(encoding, "DATA_TYPE")
    nbits = int(_require_text(encoding, "NBITS"))
    sign = encoding.findtext("SIGN", default="").strip().upper()
    if dt == "INTEGER":
        # SIGN is technically required by the Airbus DIMAP spec, but some
        # older/irregular deliveries omit it — default to UNSIGNED, which
        # is right for every Airbus imagery product we've seen.
        if not sign:
            sign = "UNSIGNED"
        if sign not in {"UNSIGNED", "SIGNED"}:
            raise ValueError(
                f"DIMAP integer encoding has unrecognized SIGN={sign!r}; "
                f"expected 'UNSIGNED' or 'SIGNED'"
            )
        prefix = "uint" if sign == "UNSIGNED" else "int"
        if nbits not in (8, 16, 32, 64):
            raise NotImplementedError(
                f"DIMAP unsupported integer NBITS={nbits}"
            )
        return np.dtype(f"{prefix}{nbits}")
    if dt == "FLOAT":
        if nbits not in (32, 64):
            raise NotImplementedError(f"DIMAP unsupported float NBITS={nbits}")
        return np.dtype(f"float{nbits}")
    raise NotImplementedError(f"DIMAP DATA_TYPE={dt!r} is not supported")


def _parse_crs_epsg(crs_root: ET.Element) -> int:
    projected = crs_root.find("Projected_CRS")
    geographic = crs_root.find("Geographic_CRS")
    for el, code_tag in (
        (projected, "PROJECTED_CRS_CODE"),
        (geographic, "GEOGRAPHIC_CRS_CODE"),
    ):
        if el is None:
            continue
        code = el.findtext(code_tag)
        if code is None:
            continue
        # e.g. "urn:ogc:def:crs:EPSG::32633" -> 32633
        tail = code.strip().rsplit(":", 1)[-1]
        try:
            return int(tail)
        except ValueError as e:
            raise ValueError(
                f"DIMAP: could not extract EPSG code from {code!r}"
            ) from e
    raise NotImplementedError(
        "DIMAP Coordinate_Reference_System has neither Projected_CRS nor "
        "Geographic_CRS with an EPSG code"
    )


def _virtual_geotiff_for(
    layout: _DIMAPLayout, *, nodata: int | float | None = 0
) -> _VirtualGeoTIFF:
    """Project a parsed layout into the ``_geotiff``-shaped metadata that
    ``AsyncGeoTIFF.__init__`` expects.

    ``nodata`` is supplied by the caller — normally the first tile TIFF's
    own nodata value, since DIMAP XML itself carries no nodata value
    (only a ``Special_Value`` *count*). Falls back to 0 when no tile is
    available (Airbus convention for integer products).
    """
    t = layout.transform
    xres = t.a
    yres = -t.e
    minx = t.c
    maxy = t.f
    maxx = minx + xres * layout.width
    miny = maxy - yres * layout.height
    return _VirtualGeoTIFF(
        crs=CRS.from_epsg(layout.crs_epsg),
        nodata=nodata,
        dtype=layout.dtype,
        count=len(layout.bands),
        width=layout.width,
        height=layout.height,
        res=(xres, yres),
        bounds=(minx, miny, maxx, maxy),
        transform=layout.transform,
    )


@dataclass(frozen=True, slots=True)
class _TileRead:
    """One (tile, src_window, dst_slice) triple for the mosaic stitcher.

    All coordinates are in *pixel* units. ``tile_row``/``tile_col`` are
    1-based so they key directly into ``_DIMAPBandGroup.tile_paths``;
    ``src_window`` is relative to the tile's own origin; ``dst_rows``
    and ``dst_cols`` slice into an output array shaped to the caller's
    requested mosaic window.
    """

    tile_row: int
    tile_col: int
    src_window: Window
    dst_rows: slice
    dst_cols: slice


def _tile_decomposition(
    layout: _DIMAPLayout, window: Window
) -> list[_TileRead]:
    """Decompose a mosaic window into per-tile read instructions.

    Tiles that do not intersect the window are omitted. The window may
    extend past the mosaic's right/bottom edges; those pixels are simply
    not listed here and the caller leaves them at the pre-filled nodata
    value. Edge tiles that are smaller than ``tile_width``/``tile_height``
    (possible when dims aren't exact multiples) are clipped to the
    mosaic extent so src_window never references out-of-bounds pixels.
    """
    tw = layout.tile_width
    th = layout.tile_height
    w_x0 = window.col_off
    w_y0 = window.row_off
    w_x1 = w_x0 + window.width
    w_y1 = w_y0 + window.height

    # Only iterate the tile band that the window touches — cheap index
    # math, avoids scanning all tiles for tiny AOIs on big mosaics.
    c_min = max(1, w_x0 // tw + 1)
    c_max = min(layout.tile_cols, (w_x1 - 1) // tw + 1)
    r_min = max(1, w_y0 // th + 1)
    r_max = min(layout.tile_rows, (w_y1 - 1) // th + 1)

    reads: list[_TileRead] = []
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            tile_x0 = (c - 1) * tw
            tile_y0 = (r - 1) * th
            tile_x1 = min(c * tw, layout.width)
            tile_y1 = min(r * th, layout.height)

            ix0 = max(tile_x0, w_x0)
            iy0 = max(tile_y0, w_y0)
            ix1 = min(tile_x1, w_x1)
            iy1 = min(tile_y1, w_y1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue

            reads.append(
                _TileRead(
                    tile_row=r,
                    tile_col=c,
                    src_window=Window(
                        col_off=ix0 - tile_x0,
                        row_off=iy0 - tile_y0,
                        width=ix1 - ix0,
                        height=iy1 - iy0,
                    ),
                    dst_rows=slice(iy0 - w_y0, iy1 - w_y0),
                    dst_cols=slice(ix0 - w_x0, ix1 - w_x0),
                )
            )
    return reads


def _resolve_tile_uri(href: str, dimap_uri: str) -> str:
    """Resolve a DIMAP ``DATA_FILE_PATH@href`` against the DIMAP URI.

    DIMAP tile paths are always written relative to the XML's parent
    directory (unlike VRT, no absolute or /vsi… variants are observed
    in Airbus deliveries). We still treat absolute paths / URIs as
    pass-through in case a delivery ever uses them.
    """
    if "://" in href or href.startswith("/"):
        return href
    return _join_relative_uri(dimap_uri, href)
