"""Unit tests for DIMAP parser."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray, Window

from rastera.formats.dimap import (
    _DIMAPBand,
    _DIMAPBandGroup,
    _DIMAPDataset,
    _DIMAPLayout,
    _maybe_open_dimap,
    _parse_dimap_xml,
    _resolve_tile_uri,
    _tile_decomposition,
)
from rastera.reader import AsyncGeoTIFF

# Minimal but realistic PNEO-flavoured DIMAP: two band-groups (RGB + NED),
# 2x2 tile grid, UTM33N. Trimmed from a real product descriptor.
PNEO_DIMAP = b"""<Dimap_Document>
  <Coordinate_Reference_System>
    <Projected_CRS>
      <PROJECTED_CRS_NAME>WGS 84 / UTM zone 33N</PROJECTED_CRS_NAME>
      <PROJECTED_CRS_CODE>urn:ogc:def:crs:EPSG::32633</PROJECTED_CRS_CODE>
    </Projected_CRS>
  </Coordinate_Reference_System>
  <Geoposition>
    <Geoposition_Insert>
      <ULXMAP unit="m">369516</ULXMAP>
      <ULYMAP unit="m">6447186</ULYMAP>
      <XDIM unit="m">0.3</XDIM>
      <YDIM unit="m">0.3</YDIM>
    </Geoposition_Insert>
  </Geoposition>
  <Raster_Data>
    <Raster_Dimensions>
      <NROWS>1000</NROWS>
      <NCOLS>800</NCOLS>
      <NBANDS>6</NBANDS>
      <Tile_Set>
        <NTILES>4</NTILES>
        <Regular_Tiling>
          <NTILES_SIZE ncols="400" nrows="500" />
          <NTILES_COUNT ntiles_C="2" ntiles_R="2" />
          <NTILES_OVERLAP ncols="0" nrows="0" />
        </Regular_Tiling>
      </Tile_Set>
    </Raster_Dimensions>
    <Raster_Encoding>
      <DATA_TYPE>INTEGER</DATA_TYPE>
      <NBITS>16</NBITS>
      <SIGN>UNSIGNED</SIGN>
    </Raster_Encoding>
    <Data_Access>
      <DATA_FILE_ORGANISATION>BAND_COMPOSITE</DATA_FILE_ORGANISATION>
      <DATA_FILE_FORMAT>image/tiff</DATA_FILE_FORMAT>
      <DATA_FILE_TILES>true</DATA_FILE_TILES>
      <Data_Files>
        <Data_File tile_R="1" tile_C="1"><DATA_FILE_PATH href="RGB_R1C1.TIF" /></Data_File>
        <Data_File tile_R="1" tile_C="2"><DATA_FILE_PATH href="RGB_R1C2.TIF" /></Data_File>
        <Data_File tile_R="2" tile_C="1"><DATA_FILE_PATH href="RGB_R2C1.TIF" /></Data_File>
        <Data_File tile_R="2" tile_C="2"><DATA_FILE_PATH href="RGB_R2C2.TIF" /></Data_File>
        <Raster_Display>
          <Raster_Index_List>
            <Raster_Index><BAND_ID>R</BAND_ID><BAND_NAME>Red</BAND_NAME><BAND_INDEX>1</BAND_INDEX></Raster_Index>
            <Raster_Index><BAND_ID>G</BAND_ID><BAND_NAME>Green</BAND_NAME><BAND_INDEX>2</BAND_INDEX></Raster_Index>
            <Raster_Index><BAND_ID>B</BAND_ID><BAND_NAME>Blue</BAND_NAME><BAND_INDEX>3</BAND_INDEX></Raster_Index>
          </Raster_Index_List>
        </Raster_Display>
      </Data_Files>
      <Data_Files>
        <Data_File tile_R="1" tile_C="1"><DATA_FILE_PATH href="NED_R1C1.TIF" /></Data_File>
        <Data_File tile_R="1" tile_C="2"><DATA_FILE_PATH href="NED_R1C2.TIF" /></Data_File>
        <Data_File tile_R="2" tile_C="1"><DATA_FILE_PATH href="NED_R2C1.TIF" /></Data_File>
        <Data_File tile_R="2" tile_C="2"><DATA_FILE_PATH href="NED_R2C2.TIF" /></Data_File>
        <Raster_Display>
          <Raster_Index_List>
            <Raster_Index><BAND_ID>NIR</BAND_ID><BAND_NAME>NIR</BAND_NAME><BAND_INDEX>1</BAND_INDEX></Raster_Index>
            <Raster_Index><BAND_ID>RE</BAND_ID><BAND_NAME>Red Edge</BAND_NAME><BAND_INDEX>2</BAND_INDEX></Raster_Index>
            <Raster_Index><BAND_ID>DB</BAND_ID><BAND_NAME>Deep Blue</BAND_NAME><BAND_INDEX>3</BAND_INDEX></Raster_Index>
          </Raster_Index_List>
        </Raster_Display>
      </Data_Files>
    </Data_Access>
  </Raster_Data>
</Dimap_Document>"""


def _modified(xml: bytes, old: bytes, new: bytes) -> bytes:
    """Produce a variant of the fixture by substring replacement. Fails the
    test immediately if *old* isn't found, so stale tests can't silently
    stop exercising what they claim to."""
    assert xml.count(old) >= 1, f"pattern {old!r} not in fixture"
    return xml.replace(old, new, 1)


class TestParseDIMAP:
    def test_parses_grid_and_transform(self):
        layout = _parse_dimap_xml(PNEO_DIMAP)
        assert (layout.width, layout.height) == (800, 1000)
        assert (layout.tile_rows, layout.tile_cols) == (2, 2)
        assert (layout.tile_width, layout.tile_height) == (400, 500)
        # UL corner at ULXMAP/ULYMAP, positive XDIM, negative YDIM for north-up
        assert layout.transform == Affine(0.3, 0.0, 369516.0, 0.0, -0.3, 6447186.0)
        assert layout.crs_epsg == 32633
        assert layout.dtype == np.dtype("uint16")

    def test_parses_band_groups_and_virtual_band_order(self):
        """Virtual bands are concatenated across groups in document order;
        each remembers its group and its 1-based band index *within* a
        tile of that group — this is what the read dispatcher needs."""
        layout = _parse_dimap_xml(PNEO_DIMAP)
        assert [b.band_id for b in layout.bands] == [
            "R",
            "G",
            "B",
            "NIR",
            "RE",
            "DB",
        ]
        assert [(b.group_index, b.source_band) for b in layout.bands] == [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
        ]

    def test_parses_tile_paths_per_group(self):
        layout = _parse_dimap_xml(PNEO_DIMAP)
        assert len(layout.groups) == 2
        assert layout.groups[0].tile_paths == {
            (1, 1): "RGB_R1C1.TIF",
            (1, 2): "RGB_R1C2.TIF",
            (2, 1): "RGB_R2C1.TIF",
            (2, 2): "RGB_R2C2.TIF",
        }
        assert layout.groups[1].tile_paths[(2, 2)] == "NED_R2C2.TIF"

    def test_rejects_non_dimap_root(self):
        with pytest.raises(ValueError, match="Not a DIMAP"):
            _parse_dimap_xml(b"<foo/>")

    def test_rejects_irregular_tiling(self):
        xml = _modified(
            PNEO_DIMAP,
            b"<Regular_Tiling>",
            b"<Irregular_Tiling>",
        ).replace(b"</Regular_Tiling>", b"</Irregular_Tiling>")
        with pytest.raises(NotImplementedError, match="Regular_Tiling"):
            _parse_dimap_xml(xml)

    def test_rejects_overlapping_tiles(self):
        """Overlap would require resolving which tile's pixels 'win' on the
        seam. MVP just refuses rather than guess, to avoid silently
        double-counting reads."""
        xml = _modified(
            PNEO_DIMAP,
            b'<NTILES_OVERLAP ncols="0" nrows="0" />',
            b'<NTILES_OVERLAP ncols="16" nrows="16" />',
        )
        with pytest.raises(NotImplementedError, match="OVERLAP"):
            _parse_dimap_xml(xml)

    def test_rejects_non_band_composite(self):
        xml = _modified(
            PNEO_DIMAP,
            b"<DATA_FILE_ORGANISATION>BAND_COMPOSITE</DATA_FILE_ORGANISATION>",
            b"<DATA_FILE_ORGANISATION>BAND_SEPARATE</DATA_FILE_ORGANISATION>",
        )
        with pytest.raises(NotImplementedError, match="BAND_COMPOSITE"):
            _parse_dimap_xml(xml)

    def test_rejects_non_tiff_format(self):
        xml = _modified(
            PNEO_DIMAP,
            b"<DATA_FILE_FORMAT>image/tiff</DATA_FILE_FORMAT>",
            b"<DATA_FILE_FORMAT>image/jp2</DATA_FILE_FORMAT>",
        )
        with pytest.raises(NotImplementedError, match="image/tiff"):
            _parse_dimap_xml(xml)

    def test_geographic_crs_fallback(self):
        xml = _modified(
            PNEO_DIMAP,
            b"""<Projected_CRS>
      <PROJECTED_CRS_NAME>WGS 84 / UTM zone 33N</PROJECTED_CRS_NAME>
      <PROJECTED_CRS_CODE>urn:ogc:def:crs:EPSG::32633</PROJECTED_CRS_CODE>
    </Projected_CRS>""",
            b"""<Geographic_CRS>
      <GEOGRAPHIC_CRS_CODE>urn:ogc:def:crs:EPSG::4326</GEOGRAPHIC_CRS_CODE>
    </Geographic_CRS>""",
        )
        assert _parse_dimap_xml(xml).crs_epsg == 4326

    def test_signed_integer_dtype(self):
        xml = _modified(
            PNEO_DIMAP,
            b"<SIGN>UNSIGNED</SIGN>",
            b"<SIGN>SIGNED</SIGN>",
        )
        assert _parse_dimap_xml(xml).dtype == np.dtype("int16")

    def test_float_dtype(self):
        xml = _modified(
            PNEO_DIMAP,
            b"""<DATA_TYPE>INTEGER</DATA_TYPE>
      <NBITS>16</NBITS>
      <SIGN>UNSIGNED</SIGN>""",
            b"""<DATA_TYPE>FLOAT</DATA_TYPE>
      <NBITS>32</NBITS>""",
        )
        assert _parse_dimap_xml(xml).dtype == np.dtype("float32")


def _grid_layout(
    *,
    width: int,
    height: int,
    tile_w: int,
    tile_h: int,
    tile_rows: int,
    tile_cols: int,
) -> _DIMAPLayout:
    """Build a minimal layout for tile-decomposition tests.

    Only fields touched by ``_tile_decomposition`` need to be populated;
    the rest get cheap placeholders so the dataclass constructs.
    """
    return _DIMAPLayout(
        width=width,
        height=height,
        crs_epsg=32633,
        transform=Affine(0.3, 0.0, 0.0, 0.0, -0.3, 0.0),
        dtype=np.dtype("uint16"),
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tile_width=tile_w,
        tile_height=tile_h,
        groups=(_DIMAPBandGroup(tile_paths={}),),
        bands=(),
    )


class TestTileDecomposition:
    """``_tile_decomposition`` is pure math; no fixtures, no I/O."""

    def test_window_contained_in_one_tile(self):
        layout = _grid_layout(
            width=800,
            height=1000,
            tile_w=400,
            tile_h=500,
            tile_rows=2,
            tile_cols=2,
        )
        reads = _tile_decomposition(
            layout, Window(col_off=50, row_off=60, width=100, height=80)
        )
        assert len(reads) == 1
        (r,) = reads
        assert (r.tile_row, r.tile_col) == (1, 1)
        assert (r.src_window.col_off, r.src_window.row_off) == (50, 60)
        assert (r.src_window.width, r.src_window.height) == (100, 80)
        assert r.dst_rows == slice(0, 80)
        assert r.dst_cols == slice(0, 100)

    def test_window_spans_all_four_tiles(self):
        layout = _grid_layout(
            width=800,
            height=1000,
            tile_w=400,
            tile_h=500,
            tile_rows=2,
            tile_cols=2,
        )
        # Centered window that straddles both the vertical and horizontal seam.
        reads = _tile_decomposition(
            layout, Window(col_off=350, row_off=450, width=100, height=100)
        )
        assert {(r.tile_row, r.tile_col) for r in reads} == {
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
        }
        # Upper-left piece: 50x50, pasted at (0,0).
        ul = next(r for r in reads if (r.tile_row, r.tile_col) == (1, 1))
        assert ul.src_window.col_off == 350
        assert ul.src_window.row_off == 450
        assert (ul.src_window.width, ul.src_window.height) == (50, 50)
        assert ul.dst_rows == slice(0, 50)
        assert ul.dst_cols == slice(0, 50)
        # Lower-right piece: src offset (0,0) in tile(2,2), pasted at (50,50).
        lr = next(r for r in reads if (r.tile_row, r.tile_col) == (2, 2))
        assert lr.src_window.col_off == 0
        assert lr.src_window.row_off == 0
        assert (lr.src_window.width, lr.src_window.height) == (50, 50)
        assert lr.dst_rows == slice(50, 100)
        assert lr.dst_cols == slice(50, 100)

    def test_dst_slices_tile_the_output_without_gaps(self):
        """Sanity: the per-tile dst slices cover every output pixel exactly
        once. If this fails the mosaic has seams or double-writes."""
        layout = _grid_layout(
            width=1200,
            height=1500,
            tile_w=400,
            tile_h=500,
            tile_rows=3,
            tile_cols=3,
        )
        win = Window(col_off=100, row_off=200, width=900, height=1100)
        reads = _tile_decomposition(layout, win)

        cover = np.zeros((win.height, win.width), dtype=np.int32)
        for r in reads:
            cover[r.dst_rows, r.dst_cols] += 1
        assert cover.min() == 1 and cover.max() == 1

    def test_window_extending_past_mosaic_right_edge(self):
        """Pixels past the mosaic edge produce no reads — the caller fills
        those with nodata. This is how PNEO mosaics with non-multiple dims
        terminate cleanly at the right/bottom."""
        layout = _grid_layout(
            width=750,
            height=1000,
            tile_w=400,
            tile_h=500,
            tile_rows=2,
            tile_cols=2,
        )
        reads = _tile_decomposition(
            layout, Window(col_off=700, row_off=0, width=200, height=500)
        )
        # Only the right-edge tile (1,2), clipped to the mosaic's 750-px width.
        assert [(r.tile_row, r.tile_col) for r in reads] == [(1, 2)]
        (r,) = reads
        assert r.src_window.col_off == 300  # 700 - (2-1)*400
        assert r.src_window.width == 50  # 750 - 700
        assert r.dst_cols == slice(0, 50)  # remaining 150 px stay nodata

    def test_non_multiple_mosaic_dims_edge_tile_clipped(self):
        """Airbus DIMAPs sometimes declare NCOLS/NROWS that aren't exact
        multiples of the tile size; the edge tile is physically smaller.
        Make sure src_window never references past the real tile extent."""
        layout = _grid_layout(
            width=950,
            height=1100,
            tile_w=400,
            tile_h=500,
            tile_rows=3,
            tile_cols=3,
        )
        # Full-mosaic read.
        reads = _tile_decomposition(
            layout, Window(col_off=0, row_off=0, width=950, height=1100)
        )
        edge = next(r for r in reads if (r.tile_row, r.tile_col) == (3, 3))
        # Tile (3,3) covers cols 800..950 (150 px) and rows 1000..1100 (100 px).
        assert (edge.src_window.width, edge.src_window.height) == (150, 100)


def _two_group_layout() -> _DIMAPLayout:
    """A 2x2 tile grid with two 3-band groups (RGB + NED), matching PNEO's
    shape but at a test-friendly size. Six virtual bands across two groups."""
    return _DIMAPLayout(
        width=800,
        height=1000,
        crs_epsg=32633,
        transform=Affine(0.3, 0.0, 369516.0, 0.0, -0.3, 6447186.0),
        dtype=np.dtype("uint16"),
        tile_rows=2,
        tile_cols=2,
        tile_width=400,
        tile_height=500,
        groups=(
            _DIMAPBandGroup(
                tile_paths={
                    (1, 1): "RGB_R1C1.TIF",
                    (1, 2): "RGB_R1C2.TIF",
                    (2, 1): "RGB_R2C1.TIF",
                    (2, 2): "RGB_R2C2.TIF",
                }
            ),
            _DIMAPBandGroup(
                tile_paths={
                    (1, 1): "NED_R1C1.TIF",
                    (1, 2): "NED_R1C2.TIF",
                    (2, 1): "NED_R2C1.TIF",
                    (2, 2): "NED_R2C2.TIF",
                }
            ),
        ),
        bands=(
            _DIMAPBand("R", "Red", 0, 1),
            _DIMAPBand("G", "Green", 0, 2),
            _DIMAPBand("B", "Blue", 0, 3),
            _DIMAPBand("NIR", "NIR", 1, 1),
            _DIMAPBand("RE", "RedEdge", 1, 2),
            _DIMAPBand("DB", "DeepBlue", 1, 3),
        ),
    )


def _mock_tile_ds(fill_fn: Any) -> AsyncGeoTIFF:
    """Build a mock tile ``AsyncGeoTIFF`` whose ``_read_native`` returns an
    array whose values depend on the tile identity and requested bands.

    ``fill_fn(src_bands_0_based, src_window)`` must return the ndarray to
    hand back. This lets each per-tile test assert "this pixel came from
    that tile+that band+that source-window" without wiring up real TIFFs.
    """
    ds = MagicMock(spec=AsyncGeoTIFF)

    async def _read_native(
        *, window: Window, band_indices: list[int], **_: Any
    ) -> RasterArray:
        data = fill_fn(band_indices, window)
        geotiff = MagicMock()
        geotiff.nodata = None
        return RasterArray(
            data=data,
            mask=None,
            width=window.width,
            height=window.height,
            count=data.shape[0],
            transform=Affine(0.3, 0, 0, 0, -0.3, 0),
            _alpha_band_idx=None,
            _geotiff=geotiff,
        )

    ds._read_native = _read_native
    return ds


class TestDIMAPRead:
    """Exercises ``_DIMAPDataset._read_native`` against mocked tile reads.
    The stitcher's hard parts are band-order preservation across groups
    and correct edge-pasting; these tests target both."""

    def _make_ds(self, tile_factory: Any) -> _DIMAPDataset:
        """Build a DIMAP dataset whose ``_get_tile`` returns mock tiles
        produced by ``tile_factory(group_idx, tile_row, tile_col)``."""
        layout = _two_group_layout()
        ds = _DIMAPDataset("/fake/DIM.xml", layout)

        async def _get_tile(g: int, r: int, c: int) -> AsyncGeoTIFF:
            return tile_factory(g, r, c)

        ds._get_tile = _get_tile  # type: ignore[assignment]
        return ds

    @pytest.mark.asyncio
    async def test_single_tile_single_group(self):
        """Window inside tile (1,1); only RGB bands → one tile opened."""
        opened: list[tuple[int, int, int]] = []

        def factory(g: int, r: int, c: int) -> AsyncGeoTIFF:
            opened.append((g, r, c))
            return _mock_tile_ds(
                lambda bands, w: np.full(
                    (len(bands), w.height, w.width),
                    100 + bands[0],
                    dtype=np.uint16,
                )
            )

        ds = self._make_ds(factory)
        arr = await ds._read_native(
            window=Window(col_off=10, row_off=20, width=50, height=40),
            band_indices=[0, 1, 2],
        )
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (3, 40, 50)
        assert opened == [(0, 1, 1)]

    @pytest.mark.asyncio
    async def test_cross_group_band_order_preserved(self):
        """band_indices=[5, 0] selects band 6 (group 1, src_band 3) then
        band 1 (group 0, src_band 1). Output must be [DB, R] — NOT sorted
        by group index. This is the test that catches "bands written in
        group-discovery order" bugs."""

        def factory(g: int, r: int, c: int) -> AsyncGeoTIFF:
            # Tag each value with (group*100 + source_band) so the test
            # can assert exactly which band came from which group.
            return _mock_tile_ds(
                lambda bands, w: np.stack(
                    [
                        np.full((w.height, w.width), g * 100 + (b + 1), dtype=np.uint16)
                        for b in bands
                    ]
                )
            )

        ds = self._make_ds(factory)
        arr = await ds._read_native(
            window=Window(col_off=0, row_off=0, width=20, height=20),
            band_indices=[5, 0],
        )
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (2, 20, 20)
        # out[0] = band 6 = group 1, source_band 3 → tag 103
        np.testing.assert_array_equal(data[0], 103)
        # out[1] = band 1 = group 0, source_band 1 → tag 1
        np.testing.assert_array_equal(data[1], 1)

    @pytest.mark.asyncio
    async def test_cross_tile_stitch(self):
        """Window that straddles the seam between tile (1,1) and (1,2)
        for a single group → two tile reads, correctly pasted."""

        # left half fills with 10, right half with 20; stitched mosaic
        # must have exactly that split at the seam.
        def factory(g: int, r: int, c: int) -> AsyncGeoTIFF:
            value = 10 * c  # tile_col=1 → 10, tile_col=2 → 20
            return _mock_tile_ds(
                lambda bands, w, v=value: np.full(
                    (len(bands), w.height, w.width), v, dtype=np.uint16
                )
            )

        ds = self._make_ds(factory)
        # Window [col 380..420, row 0..10) spans 20 px in tile(1,1)
        # and 20 px in tile(1,2).
        arr = await ds._read_native(
            window=Window(col_off=380, row_off=0, width=40, height=10),
            band_indices=[0],
        )
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (1, 10, 40)
        np.testing.assert_array_equal(data[0, :, :20], 10)
        np.testing.assert_array_equal(data[0, :, 20:], 20)

    @pytest.mark.asyncio
    async def test_full_extent_all_groups_sequential(self):
        """Full-extent read across both groups (4 tiles × 2 groups = 8 tile
        reads) must complete and return correctly stitched output. Tracks
        peak in-flight count across an `await asyncio.sleep(0)` yield to
        assert outer dispatch stays sequential — protects against a future
        regression that reintroduces outer asyncio.gather and re-creates
        the HTTP burst that motivated the sequential design."""
        read_order: list[tuple[int, int, int]] = []
        in_flight = 0
        max_in_flight = 0

        def factory(g: int, r: int, c: int) -> AsyncGeoTIFF:
            ds = MagicMock(spec=AsyncGeoTIFF)

            async def _read_native(
                *, window: Window, band_indices: list[int], **_: Any
            ) -> RasterArray:
                nonlocal in_flight, max_in_flight
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                # Yield to the loop so any concurrently-scheduled tile
                # reads get a chance to enter and bump in_flight.
                await asyncio.sleep(0)
                read_order.append((g, r, c))
                in_flight -= 1
                # Tag each pixel with (group, tile_row, tile_col, src_band)
                # so we can assert correct paste positions afterwards.
                data = np.stack(
                    [
                        np.full(
                            (window.height, window.width),
                            g * 1000 + r * 100 + c * 10 + (b + 1),
                            dtype=np.uint16,
                        )
                        for b in band_indices
                    ]
                )
                geotiff = MagicMock()
                geotiff.nodata = None
                return RasterArray(
                    data=data,
                    mask=None,
                    width=window.width,
                    height=window.height,
                    count=data.shape[0],
                    transform=Affine(0.3, 0, 0, 0, -0.3, 0),
                    _alpha_band_idx=None,
                    _geotiff=geotiff,
                )

            ds._read_native = _read_native
            return ds

        ds = self._make_ds(factory)
        arr = await ds._read_native(band_indices=list(range(6)))
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]

        layout = _two_group_layout()
        assert data.shape == (6, layout.height, layout.width)
        # 8 distinct tile reads (4 tiles × 2 groups), each exactly once.
        assert len(read_order) == 8
        assert set(read_order) == {
            (g, r, c) for g in (0, 1) for r in (1, 2) for c in (1, 2)
        }
        # Outer dispatch must be sequential: only one tile read in flight
        # at a time across the asyncio.sleep(0) yield point.
        assert max_in_flight == 1
        # Spot-check a corner pixel from each (group, tile) lands in the
        # right output slot. RGB band 0 (group 0, src_band 1) at tile (2,2)
        # → tag 0*1000 + 2*100 + 2*10 + 1 = 221.
        assert data[0, layout.height - 1, layout.width - 1] == 221
        # NED band 5 (group 1, src_band 3) at tile (1,1) → 1*1000 + 100 + 10 + 3 = 1113.
        assert data[5, 0, 0] == 1113

    @pytest.mark.asyncio
    async def test_window_past_mosaic_edge_fills_nodata(self):
        """Window extending past the mosaic's right edge → the out-of-extent
        pixels must be filled with nodata (0 by default)."""

        def factory(g: int, r: int, c: int) -> AsyncGeoTIFF:
            return _mock_tile_ds(
                lambda bands, w: np.full(
                    (len(bands), w.height, w.width), 42, dtype=np.uint16
                )
            )

        ds = self._make_ds(factory)
        # mosaic width=800; read to col 850 → last 50 cols are past the edge.
        arr = await ds._read_native(
            window=Window(col_off=750, row_off=0, width=100, height=10),
            band_indices=[0],
        )
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (1, 10, 100)
        np.testing.assert_array_equal(data[0, :, :50], 42)  # inside mosaic
        np.testing.assert_array_equal(data[0, :, 50:], 0)  # nodata fill

    @pytest.mark.asyncio
    async def test_read_rejects_use_overviews(self):
        """DIMAP per-tile overviews cannot safely mix across tiles; the
        public ``read`` must refuse rather than silently produce shape-
        mismatched output."""
        ds = self._make_ds(
            lambda g, r, c: _mock_tile_ds(
                lambda bands, w: np.zeros(
                    (len(bands), w.height, w.width), dtype=np.uint16
                )
            )
        )
        with pytest.raises(NotImplementedError, match="use_overviews"):
            await ds.read(use_overviews=True)

    @pytest.mark.asyncio
    async def test_read_native_rejects_overview(self):
        ds = self._make_ds(
            lambda g, r, c: _mock_tile_ds(
                lambda bands, w: np.zeros(
                    (len(bands), w.height, w.width), dtype=np.uint16
                )
            )
        )
        with pytest.raises(NotImplementedError, match="overview"):
            await ds._read_native(overview=MagicMock())

    @pytest.mark.asyncio
    async def test_tile_opens_are_single_flight(self):
        """Two reads that both touch tile (1,1) must share one tile open
        across the dataset's lifetime — otherwise every read re-fetches
        the tile header."""
        open_count = 0
        layout = _two_group_layout()
        ds = _DIMAPDataset("/fake/DIM.xml", layout)

        async def fake_open(uri: str, **_: Any) -> AsyncGeoTIFF:
            nonlocal open_count
            open_count += 1
            return _mock_tile_ds(
                lambda bands, w: np.zeros(
                    (len(bands), w.height, w.width), dtype=np.uint16
                )
            )

        with patch.object(AsyncGeoTIFF, "open", side_effect=fake_open):
            await ds._read_native(
                window=Window(col_off=0, row_off=0, width=10, height=10),
                band_indices=[0],
            )
            await ds._read_native(
                window=Window(col_off=5, row_off=5, width=10, height=10),
                band_indices=[1],
            )
        assert open_count == 1  # tile (1,1) of group 0 opened exactly once


def _fake_first_tile(nodata: int | float | None = 0) -> Any:
    """Minimal stand-in for the pre-opened first tile supplied to
    ``_DIMAPDataset`` by ``_maybe_open_dimap``. Only ``_nodata`` is read
    on the open path."""
    tile = MagicMock(spec=AsyncGeoTIFF)
    tile._nodata = nodata
    return tile


def _patch_sniff(nodata: int | float | None = 0) -> Any:
    """Patch ``_sniff_first_tile`` to hand back a fake tile + key. Used
    by every test that exercises the open path but doesn't care about
    real tile TIFF I/O."""

    async def _fake(layout: Any, uri: str, kwargs: Any) -> Any:
        return (0, 1, 1), _fake_first_tile(nodata=nodata)

    return patch("rastera.formats.dimap._sniff_first_tile", new=_fake)


class TestDetection:
    """``_maybe_open_dimap`` is the sniff-hook used by ``AsyncGeoTIFF.open``
    to route recognized ``.xml`` descriptors to the DIMAP reader."""

    @pytest.mark.asyncio
    async def test_returns_dataset_for_dimap_xml(self):
        with (
            patch(
                "rastera.formats.dimap._fetch_descriptor_bytes",
                new=AsyncMock(return_value=PNEO_DIMAP),
            ),
            _patch_sniff(),
        ):
            ds = await _maybe_open_dimap("s3://bucket/DIM_PNEO.XML")
        assert isinstance(ds, _DIMAPDataset)
        assert ds.count == 6

    @pytest.mark.asyncio
    async def test_returns_none_for_non_dimap_xml(self):
        """Any other .xml (e.g. STAC item, sidecar, xmp) falls through to
        the normal TIFF open path — which surfaces the real "not a TIFF"
        error instead of a misleading DIMAP parse failure."""
        with patch(
            "rastera.formats.dimap._fetch_descriptor_bytes",
            new=AsyncMock(return_value=b"<?xml version='1.0'?><Foo/>"),
        ):
            assert await _maybe_open_dimap("s3://bucket/random.xml") is None

    @pytest.mark.asyncio
    async def test_async_geotiff_open_routes_dimap_xml(self):
        """``.xml`` URIs go through the DIMAP branch inside
        ``AsyncGeoTIFF.open``; non-DIMAP XMLs fall through to the normal
        TIFF open (and its native error)."""
        with (
            patch(
                "rastera.formats.dimap._fetch_descriptor_bytes",
                new=AsyncMock(return_value=PNEO_DIMAP),
            ),
            _patch_sniff(),
        ):
            ds = await AsyncGeoTIFF.open("s3://bucket/DIM_PNEO.XML")
        assert isinstance(ds, _DIMAPDataset)

    @pytest.mark.asyncio
    async def test_open_inherits_nodata_from_first_tile(self):
        """DIMAP XML carries no canonical nodata value, so the dataset's
        nodata must come from the first tile's TIFF nodata tag — not the
        hardcoded 0 default. Ensures mosaic pre-fill agrees with the
        sentinel used inside tiles (e.g. 65535 for some uint16 products)."""
        with (
            patch(
                "rastera.formats.dimap._fetch_descriptor_bytes",
                new=AsyncMock(return_value=PNEO_DIMAP),
            ),
            _patch_sniff(nodata=65535),
        ):
            ds = await _maybe_open_dimap("s3://bucket/DIM_PNEO.XML")
        assert ds is not None
        assert ds._nodata == 65535
        # Pre-populated tile cache: no re-fetch when the first read hits (1,1).
        assert (0, 1, 1) in ds._tile_tasks

    def test_missing_sign_defaults_to_unsigned(self):
        """Some older DIMAP deliveries omit the <SIGN> element. Treat a
        missing value as UNSIGNED — that's the Airbus default for
        imagery products, and strictness would block otherwise-valid files."""
        xml = _modified(
            PNEO_DIMAP,
            b"<SIGN>UNSIGNED</SIGN>",
            b"",
        )
        assert _parse_dimap_xml(xml).dtype == np.dtype("uint16")


class TestResolveTileURI:
    """Tile hrefs in Airbus DIMAPs are relative to the descriptor's parent
    directory; we also permit absolute URIs / absolute paths as pass-through
    in case a future delivery uses them."""

    def test_relative_href_resolves_against_s3_dimap(self):
        assert (
            _resolve_tile_uri("RGB_R1C1.TIF", "s3://bucket/prod/DIM_PNEO.XML")
            == "s3://bucket/prod/RGB_R1C1.TIF"
        )

    def test_absolute_uri_href_passthrough(self):
        assert (
            _resolve_tile_uri("s3://other/tile.tif", "s3://bucket/DIM_PNEO.XML")
            == "s3://other/tile.tif"
        )

    def test_absolute_path_href_passthrough(self):
        assert (
            _resolve_tile_uri("/mnt/data/tile.tif", "/mnt/data/DIM_PNEO.XML")
            == "/mnt/data/tile.tif"
        )


class TestCRSParsing:
    def test_rejects_missing_epsg_code(self):
        """If neither Projected_CRS nor Geographic_CRS carries an EPSG
        code, we bail rather than guess — downstream reprojection needs
        a real authority code."""
        xml = _modified(
            PNEO_DIMAP,
            b"""<Projected_CRS>
      <PROJECTED_CRS_NAME>WGS 84 / UTM zone 33N</PROJECTED_CRS_NAME>
      <PROJECTED_CRS_CODE>urn:ogc:def:crs:EPSG::32633</PROJECTED_CRS_CODE>
    </Projected_CRS>""",
            b"<Projected_CRS><PROJECTED_CRS_NAME>Unknown</PROJECTED_CRS_NAME></Projected_CRS>",
        )
        with pytest.raises(NotImplementedError, match="EPSG"):
            _parse_dimap_xml(xml)

    def test_rejects_malformed_epsg_code(self):
        xml = _modified(
            PNEO_DIMAP,
            b"urn:ogc:def:crs:EPSG::32633",
            b"urn:ogc:def:crs:EPSG::not-a-number",
        )
        with pytest.raises(ValueError, match="EPSG"):
            _parse_dimap_xml(xml)


class TestReadDefaults:
    """Default-argument behaviour for ``_DIMAPDataset._read_native``."""

    @pytest.mark.asyncio
    async def test_band_indices_none_reads_all_bands(self):
        """``band_indices=None`` must return one output band per virtual
        band, in layout order — not fall through to a zero-band read."""
        layout = _two_group_layout()
        ds = _DIMAPDataset("/fake/DIM.xml", layout)

        async def _get_tile(g: int, r: int, c: int) -> AsyncGeoTIFF:
            return _mock_tile_ds(
                lambda bands, w, g=g: np.stack(
                    [
                        np.full((w.height, w.width), g * 100 + (b + 1), dtype=np.uint16)
                        for b in bands
                    ]
                )
            )

        ds._get_tile = _get_tile  # type: ignore[assignment]
        arr = await ds._read_native(
            window=Window(col_off=0, row_off=0, width=10, height=10),
        )
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (6, 10, 10)
        # Tags confirm per-band (group, source_band) mapping is preserved.
        expected = [1, 2, 3, 101, 102, 103]
        for i, tag in enumerate(expected):
            np.testing.assert_array_equal(data[i], tag)

    @pytest.mark.asyncio
    async def test_non_zero_nodata_fills_past_edge(self):
        """When the first tile carries a non-zero nodata (common for uint16
        Airbus products), pixels past the mosaic edge must be filled with
        *that* sentinel, not a hardcoded 0."""
        first_tile = MagicMock(spec=AsyncGeoTIFF)
        first_tile._nodata = 65535
        ds = _DIMAPDataset(
            "/fake/DIM.xml",
            _two_group_layout(),
            first_tile=first_tile,
            first_tile_key=(0, 1, 1),
        )

        async def _get_tile(g: int, r: int, c: int) -> AsyncGeoTIFF:
            return _mock_tile_ds(
                lambda bands, w: np.full(
                    (len(bands), w.height, w.width), 42, dtype=np.uint16
                )
            )

        ds._get_tile = _get_tile  # type: ignore[assignment]
        arr = await ds._read_native(
            window=Window(col_off=750, row_off=0, width=100, height=10),
            band_indices=[0],
        )
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        np.testing.assert_array_equal(data[0, :, :50], 42)
        np.testing.assert_array_equal(data[0, :, 50:], 65535)


# ── concurrency: dimap ─────────────────────────────────────────────


@pytest.fixture
def _reset_dimap_concurrency():
    yield
    import rastera

    rastera.set_concurrency(merge=1, vrt=1, dimap=1)


class TestDIMAPConcurrencyInvariance:
    """Output across (group, tile) fan-out must match the n=1 baseline."""

    def _make_ds(self) -> _DIMAPDataset:
        layout = _two_group_layout()
        ds = _DIMAPDataset("/fake/DIM.xml", layout)

        async def _get_tile(g: int, r: int, c: int) -> AsyncGeoTIFF:
            # Distinct, deterministic value per (group, row, col, src_band).
            return _mock_tile_ds(
                lambda bands, w, g=g, r=r, c=c: np.stack(
                    [
                        np.full(
                            (w.height, w.width),
                            g * 1000 + r * 100 + c * 10 + (b + 1),
                            dtype=np.uint16,
                        )
                        for b in bands
                    ]
                )
            )

        ds._get_tile = _get_tile  # type: ignore[assignment]
        return ds

    @pytest.mark.parametrize("n", [1, 2, 8])
    async def test_pixel_equal_across_n(self, n, _reset_dimap_concurrency):
        import rastera

        rastera.set_concurrency(dimap=1)
        baseline = await self._make_ds()._read_native(
            window=Window(col_off=0, row_off=0, width=800, height=1000),
            band_indices=[0, 2, 4, 5],
        )

        rastera.set_concurrency(dimap=n)
        result = await self._make_ds()._read_native(
            window=Window(col_off=0, row_off=0, width=800, height=1000),
            band_indices=[0, 2, 4, 5],
        )
        np.testing.assert_array_equal(result.data, baseline.data)
