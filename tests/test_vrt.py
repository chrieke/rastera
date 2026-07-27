"""Unit tests for internal band-stack VRT support."""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray

import rastera
from rastera.reader import AsyncGeoTIFF
from rastera.store import _fetch_descriptor_bytes
from rastera.vrt import (
    _declared_nodata,
    _open_vrt,
    _parse_vrt_xml,
    _resolve_source_uri,
    _transforms_match,
    _VRTBand,
    _VRTDataset,
)
from tests.conftest import make_mock_geotiff

# ── fixtures / helpers ──────────────────────────────────────────────────────

RGBNIR_VRT = b"""<VRTDataset rasterXSize="10000" rasterYSize="10000">
  <SRS>EPSG:3006</SRS>
  <GeoTransform>637500.0, 0.25, 0.0, 6557500.0, 0.0, -0.25</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1">
    <SimpleSource>
      <SourceFilename>/vsis3/bucket/rgb.tif</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="2">
    <SimpleSource>
      <SourceFilename>/vsis3/bucket/rgb.tif</SourceFilename>
      <SourceBand>2</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="3">
    <SimpleSource>
      <SourceFilename>/vsis3/bucket/rgb.tif</SourceFilename>
      <SourceBand>3</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="4">
    <SimpleSource>
      <SourceFilename>/vsis3/bucket/nir.tif</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""


# A VRT's declared raster size must match its sources' real dimensions —
# _validate_source_windows rejects a mismatch, since rastera cannot resample a
# source onto a different declared canvas. Mock sources for RGBNIR_VRT must
# therefore be built at its declared size. TypedDict so that ``**_RGBNIR_DIMS``
# keeps its per-key types instead of widening to the dict's value type, which
# would collide with make_mock_geotiff's non-int parameters.
class _Dims(TypedDict):
    width: int
    height: int


_RGBNIR_DIMS: _Dims = {"width": 10000, "height": 10000}


def _read_result(
    shape: tuple[int, int, int], *, fill: int = 1, dtype: Any = np.uint8
) -> RasterArray:
    data = np.full(shape, fill, dtype=dtype)
    geotiff = MagicMock()
    geotiff.nodata = None
    geotiff.crs = MagicMock()
    geotiff.crs.to_epsg.return_value = 3006
    return RasterArray(
        data=data,
        mask=None,
        width=shape[2],
        height=shape[1],
        count=shape[0],
        transform=Affine(1, 0, 0, 0, -1, shape[1]),
        _alpha_band_idx=None,
        _geotiff=geotiff,
    )


# ── parser ──────────────────────────────────────────────────────────────────


class TestParseVRTXML:
    def test_band_stack_rgbnir(self):
        bands = _parse_vrt_xml(RGBNIR_VRT, "s3://bucket/x.vrt")
        assert isinstance(bands, list)
        assert len(bands) == 4
        assert [b.source_uri for b in bands] == [
            "s3://bucket/rgb.tif",
            "s3://bucket/rgb.tif",
            "s3://bucket/rgb.tif",
            "s3://bucket/nir.tif",
        ]
        assert [b.source_band for b in bands] == [1, 2, 3, 1]

    def test_rejects_non_vrt_root(self):
        with pytest.raises(ValueError, match="Not a VRT"):
            _parse_vrt_xml(b"<foo/>", "s3://b/x.vrt")

    def test_missing_source_filename_raises(self):
        xml = b"""<VRTDataset rasterXSize="1" rasterYSize="1">
          <VRTRasterBand band="1"><SimpleSource><SourceBand>1</SourceBand></SimpleSource></VRTRasterBand>
        </VRTDataset>"""
        with pytest.raises(ValueError, match="SourceFilename"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_missing_source_band_defaults_to_one(self):
        xml = b"""<VRTDataset rasterXSize="1" rasterYSize="1">
          <VRTRasterBand band="1"><SimpleSource>
            <SourceFilename>/vsis3/b/a.tif</SourceFilename>
          </SimpleSource></VRTRasterBand>
        </VRTDataset>"""
        bands = _parse_vrt_xml(xml, "s3://b/x.vrt")
        assert isinstance(bands, list)
        assert bands[0].source_band == 1

    def test_kernel_filtered_source_rejected(self):
        xml = b"""<VRTDataset rasterXSize="1" rasterYSize="1">
          <VRTRasterBand band="1"><KernelFilteredSource>
            <SourceFilename>/vsis3/b/a.tif</SourceFilename><SourceBand>1</SourceBand>
          </KernelFilteredSource></VRTRasterBand>
        </VRTDataset>"""
        with pytest.raises(NotImplementedError, match="KernelFilteredSource"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_multi_source_band_rejected(self):
        xml = b"""<VRTDataset rasterXSize="1" rasterYSize="1">
          <VRTRasterBand band="1">
            <SimpleSource>
              <SourceFilename>/vsis3/b/a.tif</SourceFilename><SourceBand>1</SourceBand>
            </SimpleSource>
            <SimpleSource>
              <SourceFilename>/vsis3/b/b.tif</SourceFilename><SourceBand>1</SourceBand>
            </SimpleSource>
          </VRTRasterBand>
        </VRTDataset>"""
        with pytest.raises(NotImplementedError, match="2 sources"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_no_bands_raises(self):
        with pytest.raises(ValueError, match="no <VRTRasterBand>"):
            _parse_vrt_xml(
                b'<VRTDataset rasterXSize="1" rasterYSize="1"/>', "s3://b/x.vrt"
            )


def _one_band_vrt(
    inner: str = "",
    *,
    root_attrs: str = "",
    band_attrs: str = "",
    band_inner: str = "",
    source_tag: str = "SimpleSource",
    size: int = 100,
) -> bytes:
    """A minimal single-source VRT.

    *inner* is spliced into the source element, *band_inner* into the
    ``<VRTRasterBand>`` ahead of the source (where GDAL puts
    ``<NoDataValue>``).
    """
    return (
        f'<VRTDataset rasterXSize="{size}" rasterYSize="{size}" {root_attrs}>'
        f'<VRTRasterBand band="1" {band_attrs}>{band_inner}<{source_tag}>'
        f"<SourceFilename>/vsis3/b/a.tif</SourceFilename><SourceBand>1</SourceBand>"
        f"{inner}"
        f"</{source_tag}></VRTRasterBand></VRTDataset>"
    ).encode()


def _complex_source_vrt(inner: str = "", *, band_inner: str = "") -> bytes:
    """``_one_band_vrt`` over a ``<ComplexSource>``.

    Anything to do with ``<NODATA>`` belongs here: it is a ComplexSource-only
    element, so testing its semantics on a ``<SimpleSource>`` would pin
    behaviour against XML GDAL reads differently.
    """
    return _one_band_vrt(inner, band_inner=band_inner, source_tag="ComplexSource")


# ── unsupported-feature rejection (silent-wrong-pixel guards) ───────────────


class TestRejectUnsupportedSource:
    """Elements that change which pixels a source contributes must raise
    rather than be ignored — a quiet wrong answer is worse than a missing
    feature. See rastera/vrt.py's guard section."""

    def test_full_extent_identity_rects_still_parse(self):
        """Real gdalbuildvrt output carries explicit full-extent SrcRect and
        DstRect. Rejecting on mere presence would break every such VRT."""
        xml = _one_band_vrt(
            '<SrcRect xOff="0" yOff="0" xSize="100" ySize="100"/>'
            '<DstRect xOff="0" yOff="0" xSize="100" ySize="100"/>'
        )
        bands = _parse_vrt_xml(xml, "s3://b/x.vrt")
        assert bands == [
            _VRTBand(
                source_uri="s3://b/a.tif",
                source_band=1,
                src_rect_size=(100.0, 100.0),
                dst_rect_size=(100.0, 100.0),
                vrt_declared_size=(100.0, 100.0),
            )
        ]

    def test_no_rects_parses_with_no_recorded_sizes(self):
        bands = _parse_vrt_xml(_one_band_vrt(), "s3://b/x.vrt")
        assert isinstance(bands, list)
        assert bands[0].src_rect_size is None
        assert bands[0].dst_rect_size is None
        assert bands[0].vrt_declared_size == (100.0, 100.0)

    def test_dst_rect_offset_rejected(self):
        """A DstRect offset means mosaicking — the source belongs at a
        non-origin position, which rastera would silently paste at 0,0."""
        xml = _one_band_vrt(
            '<SrcRect xOff="0" yOff="0" xSize="50" ySize="50"/>'
            '<DstRect xOff="50" yOff="0" xSize="50" ySize="50"/>'
        )
        with pytest.raises(NotImplementedError, match="<DstRect> offset"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_src_rect_offset_rejected(self):
        xml = _one_band_vrt('<SrcRect xOff="10" yOff="0" xSize="50" ySize="50"/>')
        with pytest.raises(NotImplementedError, match="<SrcRect> offset"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_rescaling_rects_rejected(self):
        xml = _one_band_vrt(
            '<SrcRect xOff="0" yOff="0" xSize="100" ySize="100"/>'
            '<DstRect xOff="0" yOff="0" xSize="50" ySize="50"/>'
        )
        with pytest.raises(NotImplementedError, match="rescales its source"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_malformed_rect_raises_value_error(self):
        xml = _one_band_vrt('<SrcRect xOff="0" yOff="0" xSize="50"/>')
        with pytest.raises(ValueError, match="Malformed <SrcRect>"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")


class TestComplexSourceAndNodata:
    """``gdalbuildvrt -separate`` emits <ComplexSource> with a <NODATA> child
    for every band whose source declares a nodata value, so the canonical
    band-stack VRT is a ComplexSource one. It is accepted exactly when it is
    semantically a <SimpleSource>.

    Everything about <NODATA> here goes through ``_complex_source_vrt``:
    GDAL only parses that element on a complex source, so the same XML under
    <SimpleSource> means something different (see
    ``test_simple_source_nodata_ignored_not_rejected``)."""

    def test_complex_source_accepted(self):
        bands = _parse_vrt_xml(_complex_source_vrt(), "s3://b/x.vrt")
        assert bands == [
            _VRTBand(
                source_uri="s3://b/a.tif",
                source_band=1,
                vrt_declared_size=(100.0, 100.0),
            )
        ]

    def test_gdalbuildvrt_separate_shape_accepted(self):
        """The exact shape real gdalbuildvrt -separate emits: ComplexSource,
        full-extent identity rects, SourceProperties, and matching
        <NODATA>/<NoDataValue>."""
        xml = _complex_source_vrt(
            '<SourceProperties RasterXSize="100" RasterYSize="100" '
            'DataType="UInt16" BlockXSize="100" BlockYSize="13"/>'
            '<SrcRect xOff="0" yOff="0" xSize="100" ySize="100"/>'
            '<DstRect xOff="0" yOff="0" xSize="100" ySize="100"/>'
            "<NODATA>0</NODATA>",
            band_inner="<NoDataValue>0</NoDataValue>",
        )
        bands = _parse_vrt_xml(xml, "s3://b/x.vrt")
        assert isinstance(bands, list)
        assert bands[0].source_uri == "s3://b/a.tif"
        assert bands[0].src_rect_size == (100.0, 100.0)

    def test_nodata_matching_band_nodata_accepted(self):
        """NODATA == NoDataValue means GDAL's masked copy is a no-op, so the
        raw source pixels are bit-correct."""
        xml = _complex_source_vrt(
            "<NODATA>-9999</NODATA>", band_inner="<NoDataValue>-9999</NoDataValue>"
        )
        bands = _parse_vrt_xml(xml, "s3://b/x.vrt")
        assert isinstance(bands, list)
        assert bands[0].source_band == 1

    def test_nodata_zero_without_band_nodata_accepted(self):
        """With no <NoDataValue>, GDAL fills with 0 — so <NODATA>0 is also a
        no-op."""
        assert _parse_vrt_xml(_complex_source_vrt("<NODATA>0</NODATA>"), "s3://b/x.vrt")

    def test_nan_nodata_matching_accepted(self):
        xml = _complex_source_vrt(
            "<NODATA>nan</NODATA>", band_inner="<NoDataValue>nan</NoDataValue>"
        )
        assert _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_remapping_nodata_rejected(self):
        """<NODATA> disagreeing with the band fill really does remap pixels in
        GDAL, so it must still raise."""
        xml = _complex_source_vrt(
            "<NODATA>-9999</NODATA>", band_inner="<NoDataValue>0</NoDataValue>"
        )
        with pytest.raises(NotImplementedError, match="would remap those pixels"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_nonzero_nodata_without_band_nodata_rejected(self):
        """No <NoDataValue> means GDAL fills with 0 while the source masks
        -9999, so those pixels really are remapped. Hand-written shape —
        gdalbuildvrt always writes the two together."""
        xml = _complex_source_vrt("<NODATA>-9999</NODATA>")
        with pytest.raises(NotImplementedError, match="fills masked pixels with 0"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_hidden_band_nodata_still_governs_remapping(self):
        """<HideNoDataValue> suppresses only what GDAL *reports*; it still fills
        masked pixels with the value. Verified on GDAL 3.12 — a source pixel of
        50 under this exact XML reads back as 100. So the remapping guard must
        keep seeing the value even though _declared_nodata skips it."""
        xml = _complex_source_vrt(
            "<NODATA>50</NODATA>",
            band_inner="<NoDataValue>100</NoDataValue>"
            "<HideNoDataValue>1</HideNoDataValue>",
        )
        with pytest.raises(NotImplementedError, match="would remap those pixels"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_simple_source_nodata_ignored_not_rejected(self):
        """<NODATA> is a ComplexSource-only element: VRTSimpleSource never
        parses it, so GDAL copies the source through untouched. Verified on
        GDAL 3.12 — a source pixel of 7 under this exact XML reads back as 7
        through a SimpleSource and 0 through a ComplexSource. rastera returns
        the raw pixels either way, so the SimpleSource form is already
        bit-correct and must not be rejected."""
        xml = _one_band_vrt(
            "<NODATA>-9999</NODATA>", band_inner="<NoDataValue>0</NoDataValue>"
        )
        assert _parse_vrt_xml(xml, "s3://b/x.vrt")

    @pytest.mark.parametrize(
        "child",
        [
            "<ScaleOffset>0</ScaleOffset>",
            "<ScaleRatio>0.0255</ScaleRatio>",
            "<LUT>0:0,10000:255</LUT>",
            "<Exponent>0.5</Exponent>",
            "<UseMaskBand>true</UseMaskBand>",
            "<ColorTableComponent>1</ColorTableComponent>",
            "<OpenOptions><OOI key='OVERVIEW_LEVEL'>0</OOI></OpenOptions>",
        ],
    )
    def test_value_transforming_children_rejected(self, child: str):
        tag = child[1 : child.index(">")]
        with pytest.raises(NotImplementedError, match=f"<{tag}>"):
            _parse_vrt_xml(_complex_source_vrt(child), "s3://b/x.vrt")

    def test_averaged_source_still_rejected(self):
        with pytest.raises(NotImplementedError, match="<AveragedSource>"):
            _parse_vrt_xml(_one_band_vrt(source_tag="AveragedSource"), "s3://b/x.vrt")

    def test_malformed_nodata_raises_value_error(self):
        with pytest.raises(ValueError, match="malformed <NODATA>"):
            _parse_vrt_xml(_complex_source_vrt("<NODATA>abc</NODATA>"), "s3://b/x.vrt")

    def test_malformed_band_nodata_raises_value_error(self):
        xml = _one_band_vrt(band_inner="<NoDataValue>abc</NoDataValue>")
        with pytest.raises(ValueError, match="malformed <NoDataValue>"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")


def _one_source_ds_over(
    pixels: np.ndarray[Any, Any],
    *,
    source_nodata: float | None,
    vrt_nodata: float | None = None,
) -> AsyncGeoTIFF:
    """A dataset over *pixels* at 1 unit/px, for comparing read paths.

    With *vrt_nodata* the result is a 1-band ``_VRTDataset`` declaring it;
    without, a bare ``AsyncGeoTIFF``. Either way the source's ``_read_native``
    returns all of *pixels* — the reads below are full-extent, so the requested
    bbox is the source's own bounds.
    """
    _, height, width = pixels.shape
    gt = make_mock_geotiff(
        width=width, height=height, scale=1.0, count=1, nodata=source_nodata
    )
    src = AsyncGeoTIFF("s3://b/a.tif", gt)

    async def fake_read_native(**_: Any) -> RasterArray:
        return RasterArray(
            data=pixels,
            mask=None,
            width=width,
            height=height,
            count=1,
            transform=gt.transform,
            _alpha_band_idx=None,
            _geotiff=gt,
        )

    src._read_native = fake_read_native  # type: ignore[method-assign]
    if vrt_nodata is None:
        return src
    return _VRTDataset(
        "s3://b/x.vrt",
        [_VRTBand("s3://b/a.tif", 1, nodata=vrt_nodata)],
        {"s3://b/a.tif": src},
    )


class TestDeclaredNodata:
    """The VRT's own ``<NoDataValue>`` is honoured (the one piece of
    ``<VRTRasterBand>`` metadata that is not ignored). Band-stack VRTs over a
    DIMAP descriptor declare it per band while the descriptor declares none;
    inheriting the source's ``None`` made merge composite their black footprint
    corners over a neighbour's real pixels. See ``_declared_nodata``."""

    @staticmethod
    async def _open(xml: bytes, *, source_nodata: float | None = None) -> _VRTDataset:
        gt = make_mock_geotiff(count=1, width=100, height=100, nodata=source_nodata)

        async def fake_open(uri: str, **_: Any) -> AsyncGeoTIFF:
            return AsyncGeoTIFF(uri, gt)

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes", new=AsyncMock(return_value=xml)
            ),
            patch.object(AsyncGeoTIFF, "open", side_effect=fake_open),
        ):
            ds = await _open_vrt("s3://bucket/v.vrt")
        assert isinstance(ds, _VRTDataset)
        return ds

    @pytest.mark.asyncio
    async def test_vrt_nodata_used_when_source_declares_none(self):
        ds = await self._open(_one_band_vrt(band_inner="<NoDataValue>0</NoDataValue>"))
        assert ds._nodata == 0

    @pytest.mark.asyncio
    async def test_vrt_nodata_overrides_source(self):
        """GDAL renders the VRT band, so its NoDataValue wins over the TIFF's."""
        ds = await self._open(
            _one_band_vrt(band_inner="<NoDataValue>65535</NoDataValue>"),
            source_nodata=0,
        )
        assert ds._nodata == 65535

    @pytest.mark.asyncio
    async def test_source_nodata_kept_when_vrt_declares_none(self):
        ds = await self._open(_one_band_vrt(), source_nodata=7)
        assert ds._nodata == 7

    @pytest.mark.asyncio
    async def test_unrepresentable_vrt_nodata_does_not_clear_source(self):
        """NaN nodata on an integer band coerces to None. Letting that through
        as "the VRT says no nodata" would discard the source's real value —
        exactly the loss this whole check exists to prevent."""
        ds = await self._open(
            _one_band_vrt(band_inner="<NoDataValue>nan</NoDataValue>"),
            source_nodata=7,  # mock source dtype is uint16
        )
        assert ds._nodata == 7

    @pytest.mark.asyncio
    async def test_out_of_dtype_nodata_leaves_source_value_alone(self):
        """-9999 on a uint16 band is a sentinel no pixel can hold. Adopting it
        anyway made ``np.array(nodata, dtype=...)`` inside resample raise
        OverflowError on any reprojecting read."""
        ds = await self._open(
            _one_band_vrt(band_inner="<NoDataValue>-9999</NoDataValue>"),
            source_nodata=0,
        )
        assert ds._nodata == 0
        assert ds._band_sources[0][0]._nodata == 0

    @pytest.mark.asyncio
    async def test_declared_nodata_reaches_sources(self):
        """The sources do the resampling on the VRT's behalf, so they need the
        value too — not just the VRT's own metadata. Pixel-level consequence in
        ``test_bilinear_read_honours_declared_nodata``."""
        ds = await self._open(_one_band_vrt(band_inner="<NoDataValue>0</NoDataValue>"))
        assert ds._band_sources[0][0]._nodata == 0

    @pytest.mark.asyncio
    async def test_hidden_nodata_is_not_reported(self):
        """``gdalbuildvrt -hidenodata`` writes <NoDataValue> *and*
        <HideNoDataValue>, and GDAL then reports no nodata — the flag exists so
        the fill value stays opaque background. Reporting it would make merge
        paste a neighbour's pixels through it, the inverse of the bug this
        feature fixes."""
        ds = await self._open(
            _one_band_vrt(
                band_inner="<NoDataValue>0</NoDataValue>"
                "<HideNoDataValue>1</HideNoDataValue>"
            )
        )
        assert ds._nodata is None
        assert ds._band_sources[0][0]._nodata is None

    @pytest.mark.asyncio
    async def test_hidden_nodata_suppresses_the_source_value_too(self):
        """The shape `gdalbuildvrt -separate -hidenodata` actually emits: the
        sources declare a nodata, and gdalbuildvrt copies it into
        `<NoDataValue>` *and* hides it. Merely declining to override would fall
        back to the source's value and keep punching holes, so hiding has to
        suppress. GDAL agrees — it reports no nodata for the band and never
        consults the source's."""
        ds = await self._open(
            _one_band_vrt(
                band_inner="<NoDataValue>0</NoDataValue>"
                "<HideNoDataValue>1</HideNoDataValue>"
            ),
            source_nodata=0,
        )
        assert ds._nodata is None
        # Suppression is about reporting and compositing; the source still
        # resamples around its own value, which GDAL also still fills with.
        assert ds._band_sources[0][0]._nodata == 0

    @pytest.mark.asyncio
    async def test_partly_hidden_nodata_still_reports_the_visible_band(self):
        """A mix is not a suppression claim — the un-hidden band still declares
        a value, and `_declared_nodata` picks it up as usual."""
        xml = RGBNIR_VRT.replace(
            b'<VRTRasterBand dataType="Byte" band="1">',
            b'<VRTRasterBand dataType="Byte" band="1">'
            b"<NoDataValue>0</NoDataValue><HideNoDataValue>1</HideNoDataValue>",
        ).replace(
            b'<VRTRasterBand dataType="Byte" band="4">',
            b'<VRTRasterBand dataType="Byte" band="4"><NoDataValue>0</NoDataValue>',
        )
        bands = _parse_vrt_xml(xml, "s3://b/x.vrt")
        assert isinstance(bands, list)
        assert _declared_nodata(bands) == 0

    @pytest.mark.parametrize("text", ["0", "false"])
    def test_hide_nodata_switched_off_still_reports(self, text: str):
        bands = _parse_vrt_xml(
            _one_band_vrt(
                band_inner=f"<NoDataValue>5</NoDataValue>"
                f"<HideNoDataValue>{text}</HideNoDataValue>"
            ),
            "s3://b/x.vrt",
        )
        assert isinstance(bands, list)
        assert _declared_nodata(bands) == 5

    @pytest.mark.asyncio
    async def test_declared_nodata_reaches_nested_vrt_sources(self):
        """A VRT over a VRT: the push has to recurse to whoever holds real
        pixels, which it does by dispatching through ``_override_nodata``."""
        inner = _vrt_with_one_source(
            "s3://b/inner.vrt",
            "s3://b/a.tif",
            origin_x=0.0,
            width=10,
            height=10,
            scale=1.0,
            fill=1,
        )
        outer = _VRTDataset(
            "s3://b/outer.vrt",
            [_VRTBand("s3://b/inner.vrt", 1, nodata=3.0)],
            {"s3://b/inner.vrt": inner},
        )
        assert outer._nodata == 3
        assert inner._nodata == 3
        assert inner._band_sources[0][0]._nodata == 3

    @pytest.mark.asyncio
    async def test_bilinear_read_honours_declared_nodata(self):
        """The pixel-level consequence of pushing the value to the sources.

        ``_VRTDataset.read`` does not resample — it forwards target_resolution
        to each source, whose bilinear kernel renormalizes around whatever
        nodata *it* carries. A source declaring none averaged the VRT's nodata
        pixels in as real values, so a half-nodata edge came back as a gradient
        instead of a clean step. Measured against GDAL on a real product, 63 of
        400 pixels differed.

        The baseline is the same pixels read through a plain TIFF that declares
        nodata 0 on the file, which was always correct.
        """
        # Left half nodata, right half valid: bilinear across the seam is
        # exactly where an un-renormalized kernel invents intermediates.
        pixels = np.zeros((1, 8, 8), dtype=np.uint16)
        pixels[:, :, 4:] = 100

        vrt = _one_source_ds_over(pixels, source_nodata=None, vrt_nodata=0.0)
        baseline = _one_source_ds_over(pixels, source_nodata=0)

        kwargs: dict[str, Any] = dict(target_resolution=2.0, resampling="bilinear")
        vrt_data: np.ndarray[Any, Any] = (await vrt.read(**kwargs)).data  # type: ignore[reportUnknownMemberType]
        base_data: np.ndarray[Any, Any] = (await baseline.read(**kwargs)).data  # type: ignore[reportUnknownMemberType]
        got, want = np.asarray(vrt_data), np.asarray(base_data)

        np.testing.assert_array_equal(got, want)
        # Independent of the baseline: a renormalized kernel over a two-valued
        # input can only ever emit those two values.
        assert set(np.unique(got).tolist()) <= {0, 100}

    @pytest.mark.asyncio
    async def test_read_result_carries_vrt_nodata(self):
        """Not just the dataset: the returned array must report it too, since
        callers (and merge) key masking off the result."""
        ds = await self._open(_one_band_vrt(band_inner="<NoDataValue>0</NoDataValue>"))
        ds._band_sources[0][0].read = AsyncMock(return_value=_read_result((1, 8, 8)))
        arr = await ds.read()
        assert arr.nodata == 0

    @pytest.mark.asyncio
    async def test_undeclared_bands_do_not_veto(self):
        """A band with no <NoDataValue> is not a claim of "no nodata" — GDAL
        reports the dataset value off band 1 regardless."""
        xml = RGBNIR_VRT.replace(
            b'<VRTRasterBand dataType="Byte" band="1">',
            b'<VRTRasterBand dataType="Byte" band="1"><NoDataValue>0</NoDataValue>',
        )
        gt = make_mock_geotiff(count=3, **_RGBNIR_DIMS)

        async def fake_open(uri: str, **_: Any) -> AsyncGeoTIFF:
            return AsyncGeoTIFF(uri, gt)

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes", new=AsyncMock(return_value=xml)
            ),
            patch.object(AsyncGeoTIFF, "open", side_effect=fake_open),
        ):
            ds = await _open_vrt("s3://bucket/v.vrt")
        assert ds._nodata == 0

    def test_differing_band_nodata_rejected(self):
        """rastera carries one nodata per dataset, so two different declared
        values cannot both be honoured."""
        xml = RGBNIR_VRT.replace(
            b'<VRTRasterBand dataType="Byte" band="1">',
            b'<VRTRasterBand dataType="Byte" band="1"><NoDataValue>0</NoDataValue>',
        ).replace(
            b'<VRTRasterBand dataType="Byte" band="4">',
            b'<VRTRasterBand dataType="Byte" band="4"><NoDataValue>255</NoDataValue>',
        )
        # Rejected at parse time, before any source header is fetched.
        with pytest.raises(NotImplementedError, match="differing <NoDataValue>"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_all_nan_nodata_collapses_to_nan(self):
        """NaN != NaN, so the disagreement check must special-case it."""
        bands = _parse_vrt_xml(
            _one_band_vrt(band_inner="<NoDataValue>nan</NoDataValue>"), "s3://b/x.vrt"
        )
        assert isinstance(bands, list)
        declared = _declared_nodata(bands)
        assert declared is not None and math.isnan(declared)

    def test_multi_band_nan_nodata_collapses_to_nan(self):
        """The single-band case can't exercise the real hazard: each band's
        ``float("nan")`` is a distinct object, and a set keeps distinct NaNs
        (identity check first, and NaN != NaN), so ``{nan, nan, nan, nan}``
        has four members. The collapse must key off ``isnan``, not on the set
        having deduplicated them."""
        # Every band, not just band 1 — that is the whole point here.
        xml = RGBNIR_VRT.replace(
            b"<SimpleSource>", b"<NoDataValue>nan</NoDataValue><SimpleSource>"
        )
        bands = _parse_vrt_xml(xml, "s3://b/x.vrt")
        assert isinstance(bands, list) and len(bands) == 4
        assert len({id(b.nodata) for b in bands}) == 4  # four distinct NaN objects
        declared = _declared_nodata(bands)
        assert declared is not None and math.isnan(declared)

    def test_nan_mixed_with_value_rejected(self):
        xml = RGBNIR_VRT.replace(
            b'<VRTRasterBand dataType="Byte" band="1">',
            b'<VRTRasterBand dataType="Byte" band="1"><NoDataValue>nan</NoDataValue>',
        ).replace(
            b'<VRTRasterBand dataType="Byte" band="4">',
            b'<VRTRasterBand dataType="Byte" band="4"><NoDataValue>0</NoDataValue>',
        )
        with pytest.raises(NotImplementedError, match="both NaN"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")


class TestRejectOutOfScopeFeatures:
    """Permanently out of scope: honouring these means reimplementing GDAL's
    warper / pixel functions / GCP-RPC transformers."""

    def test_warped_vrt_rejected(self):
        xml = _one_band_vrt(root_attrs='subClass="VRTWarpedDataset"')
        with pytest.raises(NotImplementedError, match="Warped VRTs"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_derived_band_subclass_rejected(self):
        xml = _one_band_vrt(band_attrs='subClass="VRTDerivedRasterBand"')
        with pytest.raises(NotImplementedError, match="pixel-function band"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_pixel_function_type_rejected(self):
        """A single-SimpleSource derived band trips no other guard, so without
        this check the pixel function is silently dropped."""
        xml = _one_band_vrt().replace(
            b"<SimpleSource>",
            b"<PixelFunctionType>sum</PixelFunctionType><SimpleSource>",
        )
        with pytest.raises(NotImplementedError, match="pixel-function band"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_gcp_list_rejected(self):
        xml = _one_band_vrt().replace(
            b"><VRTRasterBand", b"><GCPList><GCP/></GCPList><VRTRasterBand"
        )
        with pytest.raises(NotImplementedError, match="GCP-georeferenced"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_rpc_metadata_rejected(self):
        """No <GeoTransform>, so the RPCs are the only georeferencing there is."""
        xml = _one_band_vrt().replace(
            b"><VRTRasterBand",
            b'><Metadata domain="RPC"><MDI key="HEIGHT_OFF">1</MDI></Metadata>'
            b"<VRTRasterBand",
        )
        with pytest.raises(NotImplementedError, match="RPC-georeferenced"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_rpc_metadata_alongside_geotransform_still_parses(self):
        """GDAL prefers the geotransform when both are present, so RPCs here are
        supplementary metadata it ignores. Orthorectified products (PNEO/SPOT/
        Pleiades, Maxar, Planet) ship both and gdal_translate -of VRT copies the
        domain through — rejecting them would be a false positive."""
        xml = _one_band_vrt().replace(
            b"><VRTRasterBand",
            b"><SRS>EPSG:32633</SRS>"
            b"<GeoTransform>0.0, 1.0, 0.0, 100.0, 0.0, -1.0</GeoTransform>"
            b'<Metadata domain="RPC"><MDI key="HEIGHT_OFF">1</MDI></Metadata>'
            b"<VRTRasterBand",
        )
        bands = _parse_vrt_xml(xml, "s3://b/x.vrt")
        assert isinstance(bands, list) and len(bands) == 1

    def test_plain_metadata_domain_still_parses(self):
        """Only the RPC domain is rejected; ordinary <Metadata> is harmless."""
        xml = _one_band_vrt().replace(
            b"><VRTRasterBand",
            b'><Metadata><MDI key="AREA_OR_POINT">Area</MDI></Metadata><VRTRasterBand',
        )
        bands = _parse_vrt_xml(xml, "s3://b/x.vrt")
        assert isinstance(bands, list) and len(bands) == 1


# ── source URI resolution ───────────────────────────────────────────────────


class TestResolveSourceURI:
    def test_vsis3(self):
        assert (
            _resolve_source_uri("/vsis3/bucket/path/to/f.tif", False, "s3://b/x.vrt")
            == "s3://bucket/path/to/f.tif"
        )

    def test_vsigs(self):
        assert (
            _resolve_source_uri("/vsigs/bucket/key.tif", False, "gs://b/x.vrt")
            == "gs://bucket/key.tif"
        )

    def test_vsicurl(self):
        assert (
            _resolve_source_uri(
                "/vsicurl/https://example.com/a.tif", False, "s3://b/x.vrt"
            )
            == "https://example.com/a.tif"
        )

    def test_absolute_s3(self):
        # relativeToVRT="0" with an already-scheme URI: pass through
        assert (
            _resolve_source_uri("s3://other/f.tif", False, "s3://b/x.vrt")
            == "s3://other/f.tif"
        )

    def test_relative_s3(self):
        assert (
            _resolve_source_uri("sub/f.tif", True, "s3://bucket/vrt/dir/x.vrt")
            == "s3://bucket/vrt/dir/sub/f.tif"
        )

    def test_relative_with_parent_traversal(self):
        assert (
            _resolve_source_uri("../other/f.tif", True, "s3://bucket/vrt/dir/x.vrt")
            == "s3://bucket/vrt/other/f.tif"
        )

    def test_relative_local(self, tmp_path: Path):
        vrt = tmp_path / "sub" / "x.vrt"
        vrt.parent.mkdir()
        resolved = _resolve_source_uri("tile.tif", True, str(vrt))
        assert resolved == str(tmp_path / "sub" / "tile.tif")

    def test_unknown_vsi_scheme_raises(self):
        with pytest.raises(NotImplementedError, match="VSI"):
            _resolve_source_uri("/vsihdfs/bucket/f.tif", False, "s3://b/x.vrt")


# ── _open_vrt / _VRTDataset.read ────────────────────────────────────────────


class TestOpenVRT:
    @pytest.mark.asyncio
    async def test_opens_unique_sources_once(self):
        gt_rgb = make_mock_geotiff(count=3, **_RGBNIR_DIMS)
        gt_nir = make_mock_geotiff(count=1, **_RGBNIR_DIMS)
        rgb_src = AsyncGeoTIFF("s3://bucket/rgb.tif", gt_rgb)
        nir_src = AsyncGeoTIFF("s3://bucket/nir.tif", gt_nir)

        async def fake_open(uri: str, **_: Any) -> AsyncGeoTIFF:
            return rgb_src if "rgb" in uri else nir_src

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes",
                new=AsyncMock(return_value=RGBNIR_VRT),
            ),
            patch.object(AsyncGeoTIFF, "open", side_effect=fake_open) as mock_open,
        ):
            ds = await _open_vrt("s3://bucket/v.vrt")

        assert isinstance(ds, _VRTDataset)
        assert mock_open.call_count == 2  # one per unique source
        assert len(ds._band_sources) == 4
        # Bands 1-3 share a source; band 4 is distinct
        assert ds._band_sources[0][0] is ds._band_sources[2][0]
        assert ds._band_sources[0][0] is not ds._band_sources[3][0]

    @pytest.mark.asyncio
    async def test_forwards_meta_overrides_to_sources(self):
        """meta_overrides must reach each source open — otherwise the VRT
        wrapper's CRS override is inconsistent with the sources the reads
        dispatch to."""
        gt_rgb = make_mock_geotiff(count=3, **_RGBNIR_DIMS)
        gt_nir = make_mock_geotiff(count=1, **_RGBNIR_DIMS)

        async def fake_open(uri: str, **kwargs: Any) -> AsyncGeoTIFF:
            gt = gt_rgb if "rgb" in uri else gt_nir
            return AsyncGeoTIFF(uri, gt, meta_overrides=kwargs.get("meta_overrides"))

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes",
                new=AsyncMock(return_value=RGBNIR_VRT),
            ),
            patch.object(AsyncGeoTIFF, "open", side_effect=fake_open) as mock_open,
        ):
            ds = await _open_vrt("s3://bucket/v.vrt", meta_overrides={"crs": 3006})

        for call in mock_open.call_args_list:
            assert call.kwargs["meta_overrides"] == {"crs": 3006}
        # Override took effect on both the wrapper and every source.
        assert isinstance(ds, _VRTDataset)
        assert ds._crs_epsg == 3006
        for src, _ in ds._band_sources:
            assert src._crs_epsg == 3006

    @pytest.mark.asyncio
    async def test_vrt_with_dimap_source_routes_through_detection(self):
        """When a VRT's <SourceFilename> points to a DIMAP .XML, the chain
        VRT → AsyncGeoTIFF.open → .xml branch → _maybe_open_dimap must
        just work — no special casing in _open_vrt_source."""
        from tests.formats.test_dimap import PNEO_DIMAP  # small DIMAP fixture

        vrt_with_xml_source = (
            RGBNIR_VRT.replace(b"/vsis3/bucket/rgb.tif", b"/vsis3/bucket/DIM_PNEO.XML")
            .replace(b"/vsis3/bucket/nir.tif", b"/vsis3/bucket/DIM_PNEO.XML")
            # Declared size must match the PNEO fixture's real dimensions.
            .replace(
                b'rasterXSize="10000" rasterYSize="10000"',
                b'rasterXSize="800" rasterYSize="1000"',
            )
        )

        from tests.formats.test_dimap import _patch_sniff

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes",
                new=AsyncMock(return_value=vrt_with_xml_source),
            ),
            patch(
                "rastera.formats.dimap._fetch_descriptor_bytes",
                new=AsyncMock(return_value=PNEO_DIMAP),
            ),
            _patch_sniff(),
        ):
            ds = await _open_vrt("s3://bucket/v.vrt")

        assert isinstance(ds, _VRTDataset)
        # Both VRT sources resolved to the same DIMAP descriptor → one
        # _DIMAPDataset instance shared across all four VRT bands.
        assert len({id(src) for src, _ in ds._band_sources}) == 1
        from rastera.formats.dimap import _DIMAPDataset

        assert isinstance(ds._band_sources[0][0], _DIMAPDataset)

    @pytest.mark.asyncio
    async def test_declared_nodata_reaches_a_dimap_source(self):
        """The shape the whole feature exists for: a band-stack VRT declaring
        `<NoDataValue>0</NoDataValue>` over a DIMAP descriptor that declares
        none. The `_DIMAPDataset` has to end up carrying it — it is what
        resamples on the VRT's behalf, and what pre-fills mosaic gaps."""
        from tests.formats.test_dimap import PNEO_DIMAP, _patch_sniff

        xml = (
            RGBNIR_VRT.replace(b"/vsis3/bucket/rgb.tif", b"/vsis3/bucket/DIM_PNEO.XML")
            .replace(b"/vsis3/bucket/nir.tif", b"/vsis3/bucket/DIM_PNEO.XML")
            .replace(
                b'rasterXSize="10000" rasterYSize="10000"',
                b'rasterXSize="800" rasterYSize="1000"',
            )
            .replace(b"<SimpleSource>", b"<NoDataValue>0</NoDataValue><SimpleSource>")
        )
        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes", new=AsyncMock(return_value=xml)
            ),
            patch(
                "rastera.formats.dimap._fetch_descriptor_bytes",
                new=AsyncMock(return_value=PNEO_DIMAP),
            ),
            _patch_sniff(nodata=None),  # the descriptor declares nothing
        ):
            ds = await _open_vrt("s3://bucket/v.vrt")

        assert isinstance(ds, _VRTDataset)
        assert ds._nodata == 0
        assert ds._band_sources[0][0]._nodata == 0

    @pytest.mark.asyncio
    async def test_non_tiff_source_raises_informative_error(self):
        """A VRT source that isn't a TIFF (e.g. an Airbus DIMAP .XML) must
        produce an error that names both URIs and hints at the cause —
        not the bare async_tiff ``unexpected magic bytes`` message."""

        class AsyncTiffException(Exception):
            pass

        # Match what async_tiff does at runtime: the exception's __module__
        # claims "async_tiff" even though the class is not importable from
        # there. _open_vrt_source keys off that combo.
        AsyncTiffException.__module__ = "async_tiff"

        async def fake_open(uri: str, **_: Any) -> AsyncGeoTIFF:
            raise AsyncTiffException('General error: unexpected magic bytes b"<?"')

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes",
                new=AsyncMock(return_value=RGBNIR_VRT),
            ),
            patch.object(AsyncGeoTIFF, "open", side_effect=fake_open),
            pytest.raises(ValueError) as exc_info,
        ):
            await _open_vrt("s3://bucket/v.vrt")

        msg = str(exc_info.value)
        assert "s3://bucket/v.vrt" in msg
        assert "rgb.tif" in msg or "nir.tif" in msg
        assert "DIMAP" in msg


class TestValidateSourceWindows:
    """Checks that need a source's *real* size, which is unknowable while
    parsing XML — so they run once the sources are open."""

    @staticmethod
    async def _open(xml: bytes, *, width: int, height: int) -> AsyncGeoTIFF:
        gt = make_mock_geotiff(count=3, width=width, height=height)

        async def fake_open(uri: str, **_: Any) -> AsyncGeoTIFF:
            return AsyncGeoTIFF(uri, gt)

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes", new=AsyncMock(return_value=xml)
            ),
            patch.object(AsyncGeoTIFF, "open", side_effect=fake_open),
        ):
            return await _open_vrt("s3://bucket/v.vrt")

    @pytest.mark.asyncio
    async def test_src_rect_windowing_larger_source_rejected(self):
        """SrcRect sizes match DstRect and offsets are 0, so the parse-time
        guard passes — but the source is physically bigger, so this VRT wants a
        sub-window rastera would silently read past."""
        xml = _one_band_vrt(
            '<SrcRect xOff="0" yOff="0" xSize="500" ySize="500"/>'
            '<DstRect xOff="0" yOff="0" xSize="500" ySize="500"/>',
            size=500,
        )
        with pytest.raises(NotImplementedError, match="<SrcRect> of 500x500"):
            await self._open(xml, width=1000, height=1000)

    @pytest.mark.asyncio
    async def test_lone_dst_rect_smaller_than_source_rejected(self):
        """With no SrcRect to compare against, the parse-time rescaling guard
        can't fire — but GDAL reads the whole source and squeezes it into the
        DstRect, so accepting this would return unresampled full-size pixels."""
        xml = _one_band_vrt('<DstRect xOff="0" yOff="0" xSize="50" ySize="50"/>')
        with pytest.raises(NotImplementedError, match="<DstRect> of 50x50"):
            await self._open(xml, width=100, height=100)

    @pytest.mark.asyncio
    async def test_declared_size_mismatch_rejected(self):
        """No rects at all: the VRT just declares a canvas that differs from
        its source, which GDAL would resample onto and rastera would not."""
        with pytest.raises(NotImplementedError, match="declares a 500x500 raster"):
            await self._open(_one_band_vrt(size=500), width=1000, height=1000)

    @pytest.mark.asyncio
    async def test_consistent_sizes_pass(self):
        ds = await self._open(_one_band_vrt(size=500), width=500, height=500)
        assert isinstance(ds, _VRTDataset)

    @pytest.mark.asyncio
    async def test_mismatched_sources_rejected_at_open(self):
        """Previously only surfaced lazily, as a generic shape error on read."""
        gt_small = make_mock_geotiff(count=3, width=10000, height=10000)
        gt_big = make_mock_geotiff(count=1, width=9999, height=10000)

        async def fake_open(uri: str, **_: Any) -> AsyncGeoTIFF:
            return AsyncGeoTIFF(uri, gt_small if "rgb" in uri else gt_big)

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes",
                new=AsyncMock(return_value=RGBNIR_VRT),
            ),
            patch.object(AsyncGeoTIFF, "open", side_effect=fake_open),
            pytest.raises(NotImplementedError, match="identical size"),
        ):
            await _open_vrt("s3://bucket/v.vrt")

    @staticmethod
    async def _open_rgbnir(nir_kwargs: dict[str, Any]):
        """Open RGBNIR_VRT where nir.tif differs from rgb.tif by *nir_kwargs*."""
        # RGBNIR_VRT's canvas
        base: dict[str, Any] = dict(count=3, width=10000, height=10000)
        nir: dict[str, Any] = {**base, "count": 1, **nir_kwargs}
        gt_rgb = make_mock_geotiff(**base)
        gt_nir = make_mock_geotiff(**nir)

        async def fake_open(uri: str, **_: Any) -> AsyncGeoTIFF:
            return AsyncGeoTIFF(uri, gt_rgb if "rgb" in uri else gt_nir)

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes",
                new=AsyncMock(return_value=RGBNIR_VRT),
            ),
            patch.object(AsyncGeoTIFF, "open", side_effect=fake_open),
        ):
            return await _open_vrt("s3://bucket/v.vrt")

    @pytest.mark.asyncio
    async def test_mismatched_source_dtype_rejected(self):
        """Equal size alone doesn't make sources stackable: bands go into one
        array typed from band 1, so a differing dtype was silently cast."""
        with pytest.raises(NotImplementedError, match="identical dtype"):
            await self._open_rgbnir({"dtype": np.dtype("f4")})

    @pytest.mark.asyncio
    async def test_mismatched_source_crs_rejected(self):
        with pytest.raises(NotImplementedError, match="in one CRS"):
            await self._open_rgbnir({"crs_epsg": 32633})

    @pytest.mark.asyncio
    async def test_mismatched_source_transform_rejected(self):
        """Same size, same CRS, different origin — the bands cover different
        ground but were returned under band 1's transform."""
        with pytest.raises(NotImplementedError, match="same extent"):
            await self._open_rgbnir({"scale": 20.0})

    @pytest.mark.asyncio
    async def test_matching_sources_pass(self):
        ds = await self._open_rgbnir({})
        assert isinstance(ds, _VRTDataset)


class TestTransformsMatch:
    UTM = Affine(10.0, 0.0, 399960.0, 0.0, -10.0, 5900040.0)

    def test_float_noise_tolerated(self):
        """Sources warped in separate GDAL runs can differ by rounding. That is
        the same grid; exact float equality would reject a valid VRT."""
        noisy = Affine(10.0, 0.0, 399960.00000000006, 0.0, -10.0, 5900040.0)
        assert _transforms_match(self.UTM, noisy)

    def test_near_zero_rotation_tolerated(self):
        noisy = Affine(10.0, 1e-16, 399960.0, 1e-16, -10.0, 5900040.0)
        assert _transforms_match(self.UTM, noisy)

    def test_origin_shift_rejected(self):
        shifted = Affine(10.0, 0.0, 399961.0, 0.0, -10.0, 5900040.0)
        assert not _transforms_match(self.UTM, shifted)

    def test_degree_grid_scale_difference_rejected(self):
        """A fixed absolute tolerance (affine's ``almost_equals`` uses 1e-5)
        would call these equal — the whole pixel is 1e-5 degrees."""
        a = Affine(1e-5, 0.0, 9.0, 0.0, -1e-5, 45.0)
        b = Affine(1.9e-5, 0.0, 9.0, 0.0, -1.9e-5, 45.0)
        assert not _transforms_match(a, b)


class TestVRTCycle:
    @pytest.mark.asyncio
    async def test_self_referencing_vrt_rejected(self):
        """A VRT source may itself be a VRT, so without a guard this recursed to
        RecursionError, issuing a network GET per level."""
        xml = (
            b'<VRTDataset rasterXSize="100" rasterYSize="100">'
            b'<VRTRasterBand band="1"><SimpleSource>'
            b"<SourceFilename>/vsis3/bucket/self.vrt</SourceFilename>"
            b"<SourceBand>1</SourceBand>"
            b"</SimpleSource></VRTRasterBand></VRTDataset>"
        )
        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes", new=AsyncMock(return_value=xml)
            ),
            pytest.raises(ValueError, match="reference cycle"),
        ):
            await _open_vrt("s3://bucket/self.vrt")


def _make_rgbnir_ds() -> _VRTDataset:
    """A 4-band VRT: bands 1-3 from rgb.tif, band 4 from nir.tif."""
    rgb_src = AsyncGeoTIFF("s3://bucket/rgb.tif", make_mock_geotiff(count=3))
    nir_src = AsyncGeoTIFF("s3://bucket/nir.tif", make_mock_geotiff(count=1))
    bands = [
        _VRTBand("s3://bucket/rgb.tif", 1),
        _VRTBand("s3://bucket/rgb.tif", 2),
        _VRTBand("s3://bucket/rgb.tif", 3),
        _VRTBand("s3://bucket/nir.tif", 1),
    ]
    return _VRTDataset(
        "s3://bucket/x.vrt",
        bands,
        {"s3://bucket/rgb.tif": rgb_src, "s3://bucket/nir.tif": nir_src},
    )


class TestVRTRead:
    @pytest.mark.asyncio
    async def test_read_all_bands_groups_by_source(self):
        ds = _make_rgbnir_ds()
        rgb_src, nir_src = ds._band_sources[0][0], ds._band_sources[3][0]

        rgb_read = AsyncMock(return_value=_read_result((3, 8, 8), fill=10))
        nir_read = AsyncMock(return_value=_read_result((1, 8, 8), fill=99))
        rgb_src.read = rgb_read
        nir_src.read = nir_read

        arr = await ds.read()

        # One read per unique source, with the full band list bundled.
        assert rgb_read.call_count == 1
        assert nir_read.call_count == 1
        assert rgb_read.call_args.kwargs["band_indices"] == [1, 2, 3]
        assert nir_read.call_args.kwargs["band_indices"] == [1]

        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (4, 8, 8)
        np.testing.assert_array_equal(data[:3], 10)
        np.testing.assert_array_equal(data[3], 99)

    @pytest.mark.asyncio
    async def test_read_reordered_bands(self):
        """band_indices=[4,1] → one NIR read + one RGB read; output order preserved."""
        ds = _make_rgbnir_ds()
        rgb_src, nir_src = ds._band_sources[0][0], ds._band_sources[3][0]

        rgb_data = np.arange(1 * 4 * 4, dtype=np.uint8).reshape(1, 4, 4)
        nir_data = np.full((1, 4, 4), 200, dtype=np.uint8)

        def make_result(data: np.ndarray[Any, Any]) -> RasterArray:
            geotiff = MagicMock()
            geotiff.nodata = None
            return RasterArray(
                data=data,
                mask=None,
                width=4,
                height=4,
                count=data.shape[0],
                transform=Affine(1, 0, 0, 0, -1, 4),
                _alpha_band_idx=None,
                _geotiff=geotiff,
            )

        rgb_src.read = AsyncMock(return_value=make_result(rgb_data))
        nir_src.read = AsyncMock(return_value=make_result(nir_data))

        arr = await ds.read(band_indices=[4, 1])
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (2, 4, 4)
        # out[0] is VRT band 4 → NIR fill=200
        np.testing.assert_array_equal(data[0], nir_data[0])
        # out[1] is VRT band 1 → RGB band 1
        np.testing.assert_array_equal(data[1], rgb_data[0])

    @pytest.mark.asyncio
    async def test_read_single_source(self):
        """Reading only bands from one source issues just one sub-read."""
        ds = _make_rgbnir_ds()
        rgb_src, nir_src = ds._band_sources[0][0], ds._band_sources[3][0]

        rgb_src.read = AsyncMock(return_value=_read_result((2, 4, 4), fill=7))
        nir_read = AsyncMock()
        nir_src.read = nir_read

        arr = await ds.read(band_indices=[1, 3])
        assert nir_read.call_count == 0
        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (2, 4, 4)

    @pytest.mark.asyncio
    async def test_invalid_band_index_raises(self):
        ds = _make_rgbnir_ds()
        with pytest.raises(ValueError, match="out of range"):
            await ds.read(band_indices=[5])

    @pytest.mark.asyncio
    async def test_read_native_dispatches_to_sources(self):
        """_read_native is the primitive merge uses — groups by source like read()."""
        ds = _make_rgbnir_ds()
        rgb_src, nir_src = ds._band_sources[0][0], ds._band_sources[3][0]

        rgb_native = AsyncMock(return_value=_read_result((3, 8, 8), fill=5))
        nir_native = AsyncMock(return_value=_read_result((1, 8, 8), fill=77))
        rgb_src._read_native = rgb_native
        nir_src._read_native = nir_native

        arr = await ds._read_native()

        assert rgb_native.call_count == 1
        assert nir_native.call_count == 1
        # _read_native is the internal primitive: band indices passed to source
        # are 0-based (converted from VRT's stored 1-based source_band).
        assert rgb_native.call_args.kwargs["band_indices"] == [0, 1, 2]
        assert nir_native.call_args.kwargs["band_indices"] == [0]

        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (4, 8, 8)
        np.testing.assert_array_equal(data[:3], 5)
        np.testing.assert_array_equal(data[3], 77)

    @pytest.mark.asyncio
    async def test_read_native_rejects_overview(self):
        ds = _make_rgbnir_ds()
        with pytest.raises(NotImplementedError, match="overview"):
            await ds._read_native(overview=MagicMock())

    @pytest.mark.asyncio
    async def test_read_rejects_use_overviews(self):
        """Public read() refuses use_overviews=True — independent overview
        selection across sources can yield mismatched shapes."""
        ds = _make_rgbnir_ds()
        with pytest.raises(NotImplementedError, match="use_overviews"):
            await ds.read(use_overviews=True)

    def test_count_reflects_vrt_band_count(self):
        """cog.count on a VRT must return the VRT's logical band count, not
        the first source's. merge() relies on this for input validation."""
        ds = _make_rgbnir_ds()
        # First source (rgb.tif) has 3 bands; VRT exposes 4.
        assert ds._geotiff.count == 3
        assert ds.count == 4


# ── merge on VRTs (end-to-end) ──────────────────────────────────────────────


def _vrt_with_one_source(
    uri: str,
    source_uri: str,
    *,
    origin_x: float,
    width: int,
    height: int,
    scale: float,
    crs_epsg: int = 32632,
    fill: int,
    nodata: float | None = None,
    dtype: np.dtype[Any] = np.dtype("u2"),
) -> _VRTDataset:
    """Build a 1-band VRT whose source's `_read_native` returns a constant-fill
    array matching whatever bbox merge requests.

    *nodata* is the VRT band's declared ``<NoDataValue>``; the source itself
    always declares none, which is the shape that motivated honouring it."""
    gt = make_mock_geotiff(
        width=width,
        height=height,
        scale=scale,
        count=1,
        dtype=dtype,
        crs_epsg=crs_epsg,
    )
    # Position the source at origin_x (origin_y = height*scale).
    origin_y = height * scale
    gt.transform = Affine(scale, 0, origin_x, 0, -scale, origin_y)
    gt.bounds = (origin_x, 0, origin_x + width * scale, origin_y)

    src = AsyncGeoTIFF(source_uri, gt)

    async def fake_read_native(
        *, bbox: Any = None, band_indices: Any = None, **_: Any
    ) -> RasterArray:
        # Pretend we read exactly the requested bbox at native resolution.
        assert bbox is not None
        w = max(1, int(round((bbox.maxx - bbox.minx) / scale)))
        h = max(1, int(round((bbox.maxy - bbox.miny) / scale)))
        n_bands = len(band_indices) if band_indices is not None else 1
        arr = np.full((n_bands, h, w), fill, dtype=dtype)
        transform = Affine(scale, 0, bbox.minx, 0, -scale, bbox.maxy)
        return RasterArray(
            data=arr,
            mask=None,
            width=w,
            height=h,
            count=n_bands,
            transform=transform,
            _alpha_band_idx=None,
            _geotiff=gt,
        )

    src._read_native = fake_read_native  # type: ignore[method-assign]

    bands = [_VRTBand(source_uri, 1, nodata=nodata)]
    return _VRTDataset(uri, bands, {source_uri: src})


class TestMergeOnVRT:
    @pytest.mark.asyncio
    async def test_merge_two_vrts_native_fast_path(self):
        """merge() dispatches through each VRT's `_read_native`, which groups
        by source. Two adjacent VRTs should stitch cleanly."""
        from rastera.geo import BBox
        from rastera.merge import merge

        vrt_a = _vrt_with_one_source(
            "s3://b/a.vrt",
            "s3://b/a.tif",
            origin_x=0.0,
            width=10,
            height=10,
            scale=1.0,
            fill=1,
        )
        vrt_b = _vrt_with_one_source(
            "s3://b/b.vrt",
            "s3://b/b.tif",
            origin_x=5.0,
            width=10,
            height=10,
            scale=1.0,
            fill=2,
        )

        result = await merge(
            [vrt_a, vrt_b],
            bbox=BBox(0, 0, 15, 10),
            bbox_crs=32632,
            target_crs=32632,
            target_resolution=1.0,
            mosaic_method="last",
            snap_to_grid=True,
        )
        data: np.ndarray[Any, Any] = result.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (1, 10, 15)
        # vrt_a only (cols 0-4): fill 1
        np.testing.assert_array_equal(data[0, :, :5], 1)
        # vrt_b only (cols 10-14): fill 2
        np.testing.assert_array_equal(data[0, :, 10:], 2)
        # Overlap (cols 5-9) with mosaic_method="last": vrt_b wins
        np.testing.assert_array_equal(data[0, :, 5:10], 2)

    @pytest.mark.asyncio
    async def test_merge_native_fast_path_reports_vrt_nodata(self):
        """The native path used to report the *source's* nodata, so a caller
        masking off `merged.nodata` saw nodata pixels as valid — even though the
        compositing itself had already keyed off the VRT's value."""
        from rastera.geo import BBox
        from rastera.merge import merge

        vrt = _vrt_with_one_source(
            "s3://b/a.vrt",
            "s3://b/a.tif",
            origin_x=0.0,
            width=10,
            height=10,
            scale=1.0,
            fill=1,
            nodata=0.0,
        )
        assert vrt._geotiff.nodata is None  # the source declares none
        result = await merge(
            [vrt],
            bbox=BBox(0, 0, 10, 10),
            bbox_crs=32632,
            target_crs=32632,
            target_resolution=1.0,
            snap_to_grid=True,
        )
        assert result.nodata == 0

    @pytest.mark.asyncio
    async def test_merge_vrt_with_use_overviews_raises(self):
        """use_overviews=True passes an `overview` object to `_read_native`,
        which VRTs can't support across multiple sources."""
        from rastera.geo import BBox
        from rastera.merge import merge

        # Natively 1.0 m/px, request 10.0 m/px to trigger the reprojected
        # path, and give each source a coarse overview so merge selects it.
        vrt_a = _vrt_with_one_source(
            "s3://b/a.vrt",
            "s3://b/a.tif",
            origin_x=0.0,
            width=10,
            height=10,
            scale=1.0,
            fill=1,
        )
        vrt_b = _vrt_with_one_source(
            "s3://b/b.vrt",
            "s3://b/b.tif",
            origin_x=5.0,
            width=10,
            height=10,
            scale=1.0,
            fill=2,
        )
        for vrt in (vrt_a, vrt_b):
            ov = MagicMock()
            ov.width = 1  # overview_res = native_res * (10/1) = 10.0
            ov.height = 1
            # overviews is a read-only property on the real GeoTIFF.
            vrt._band_sources[0][0]._geotiff.overviews = [ov]  # type: ignore[reportAttributeAccessIssue]

        with pytest.raises(NotImplementedError, match="overview"):
            await merge(
                [vrt_a, vrt_b],
                bbox=BBox(0, 0, 15, 10),
                bbox_crs=32632,
                target_crs=32632,
                target_resolution=10.0,
                use_overviews=True,
            )


# ── dispatch from rastera.open ──────────────────────────────────────────────


class TestDispatch:
    @pytest.mark.asyncio
    async def test_open_vrt_returns_vrtdataset(self):
        """`.vrt` URIs route through _open_vrt rather than async_tiff."""
        sentinel = MagicMock(spec=_VRTDataset)
        with patch(
            "rastera.vrt._open_vrt", new=AsyncMock(return_value=sentinel)
        ) as mock_open_vrt:
            result = await rastera.open("s3://bucket/x.vrt")
        mock_open_vrt.assert_awaited_once()
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_open_many_forwards_store_kwargs_to_vrt(self):
        """List-open must forward store_kwargs so _open_vrt can rebuild its
        obstore with the caller's credentials/region, not empty defaults."""
        sentinel = MagicMock(spec=_VRTDataset)
        with (
            patch(
                "rastera.vrt._open_vrt", new=AsyncMock(return_value=sentinel)
            ) as mock_open_vrt,
            patch("rastera.reader._build_store"),
        ):
            await rastera.open(
                ["s3://bucket/a.vrt", "s3://bucket/b.vrt"],
                skip_signature=False,
                region="eu-north-1",
            )
        assert mock_open_vrt.await_count == 2
        for call in mock_open_vrt.await_args_list:
            assert call.kwargs["skip_signature"] is False
            assert call.kwargs["region"] == "eu-north-1"

    @pytest.mark.asyncio
    async def test_non_vrt_does_not_dispatch(self):
        """Non-`.vrt` URIs never reach _open_vrt."""
        gt = make_mock_geotiff()
        with (
            patch("rastera.vrt._open_vrt", new=AsyncMock()) as mock_open_vrt,
            patch("rastera.reader.GeoTIFF") as mock_geotiff_cls,
            patch("rastera.store.from_url"),
        ):
            mock_geotiff_cls.open = AsyncMock(return_value=gt)
            await rastera.open("s3://bucket/plain.tif", cache=False)
        mock_open_vrt.assert_not_called()


# ── _fetch_descriptor_bytes for local paths ─────────────────────────────────


class TestFetchLocal:
    @pytest.mark.asyncio
    async def test_local_file(self, tmp_path: Path):
        vrt = tmp_path / "x.vrt"
        vrt.write_bytes(RGBNIR_VRT)
        data = await _fetch_descriptor_bytes(str(vrt))
        assert data == RGBNIR_VRT


# ── concurrency: vrt ─────────────────────────────────────────────


@pytest.fixture
def _reset_vrt_concurrency() -> Iterator[None]:
    yield
    rastera.set_concurrency(vrt=1)


def _mocked_rgbnir_ds() -> _VRTDataset:
    """``_make_rgbnir_ds`` with both sources' reads stubbed to distinct fills,
    so a group/result mix-up in the reassembly shows up as wrong pixels."""
    ds = _make_rgbnir_ds()
    rgb_src, nir_src = ds._band_sources[0][0], ds._band_sources[3][0]
    rgb_src.read = AsyncMock(return_value=_read_result((3, 8, 8), fill=10))
    nir_src.read = AsyncMock(return_value=_read_result((1, 8, 8), fill=99))
    return ds


class TestVRTConcurrencyInvariance:
    @pytest.mark.parametrize("n", [1, 8])
    async def test_pixel_equal_across_n(
        self, n: int, _reset_vrt_concurrency: None
    ) -> None:
        rastera.set_concurrency(vrt=1)
        baseline = await _mocked_rgbnir_ds().read()

        rastera.set_concurrency(vrt=n)
        result = await _mocked_rgbnir_ds().read()
        np.testing.assert_array_equal(result.data, baseline.data)  # type: ignore[reportUnknownMemberType]
