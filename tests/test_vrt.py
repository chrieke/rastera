"""Unit tests for internal band-stack VRT support."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray

import rastera
from rastera.reader import AsyncGeoTIFF
from rastera.store import _fetch_descriptor_bytes
from rastera.vrt import (
    _open_vrt,
    _parse_vrt_xml,
    _resolve_source_uri,
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
        assert bands[0].source_band == 1

    def test_complex_source_rejected(self):
        xml = b"""<VRTDataset rasterXSize="1" rasterYSize="1">
          <VRTRasterBand band="1"><ComplexSource>
            <SourceFilename>/vsis3/b/a.tif</SourceFilename><SourceBand>1</SourceBand>
          </ComplexSource></VRTRasterBand>
        </VRTDataset>"""
        with pytest.raises(NotImplementedError, match="ComplexSource"):
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
        gt_rgb = make_mock_geotiff(count=3)
        gt_nir = make_mock_geotiff(count=1)
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
        gt_rgb = make_mock_geotiff(count=3)
        gt_nir = make_mock_geotiff(count=1)

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
        assert ds._crs_epsg == 3006
        for src, _ in ds._band_sources:
            assert src._crs_epsg == 3006

    @pytest.mark.asyncio
    async def test_vrt_with_dimap_source_routes_through_detection(self):
        """When a VRT's <SourceFilename> points to a DIMAP .XML, the chain
        VRT → AsyncGeoTIFF.open → .xml branch → _maybe_open_dimap must
        just work — no special casing in _open_vrt_source."""
        from tests.formats.test_dimap import PNEO_DIMAP  # small DIMAP fixture

        vrt_with_xml_source = RGBNIR_VRT.replace(
            b"/vsis3/bucket/rgb.tif", b"/vsis3/bucket/DIM_PNEO.XML"
        ).replace(
            b"/vsis3/bucket/nir.tif", b"/vsis3/bucket/DIM_PNEO.XML"
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


class TestVRTRead:
    def _make_ds(self) -> _VRTDataset:
        gt_rgb = make_mock_geotiff(count=3)
        gt_nir = make_mock_geotiff(count=1)
        rgb_src = AsyncGeoTIFF("s3://bucket/rgb.tif", gt_rgb)
        nir_src = AsyncGeoTIFF("s3://bucket/nir.tif", gt_nir)
        bands = [
            _VRTBand("s3://bucket/rgb.tif", 1),
            _VRTBand("s3://bucket/rgb.tif", 2),
            _VRTBand("s3://bucket/rgb.tif", 3),
            _VRTBand("s3://bucket/nir.tif", 1),
        ]
        return _VRTDataset(
            "s3://bucket/x.vrt",
            bands,
            {
                "s3://bucket/rgb.tif": rgb_src,
                "s3://bucket/nir.tif": nir_src,
            },
        )

    @pytest.mark.asyncio
    async def test_read_all_bands_groups_by_source(self):
        ds = self._make_ds()
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
        ds = self._make_ds()
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
        ds = self._make_ds()
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
        ds = self._make_ds()
        with pytest.raises(ValueError, match="out of range"):
            await ds.read(band_indices=[5])

    @pytest.mark.asyncio
    async def test_read_native_dispatches_to_sources(self):
        """_read_native is the primitive merge uses — groups by source like read()."""
        ds = self._make_ds()
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
        ds = self._make_ds()
        with pytest.raises(NotImplementedError, match="overview"):
            await ds._read_native(overview=MagicMock())

    @pytest.mark.asyncio
    async def test_read_rejects_use_overviews(self):
        """Public read() refuses use_overviews=True — independent overview
        selection across sources can yield mismatched shapes."""
        ds = self._make_ds()
        with pytest.raises(NotImplementedError, match="use_overviews"):
            await ds.read(use_overviews=True)

    def test_count_reflects_vrt_band_count(self):
        """cog.count on a VRT must return the VRT's logical band count, not
        the first source's. merge() relies on this for input validation."""
        ds = self._make_ds()
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
    dtype: np.dtype[Any] = np.dtype("u2"),
) -> _VRTDataset:
    """Build a 1-band VRT whose source's `_read_native` returns a constant-fill
    array matching whatever bbox merge requests."""
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

    bands = [_VRTBand(source_uri, 1)]
    return _VRTDataset(uri, bands, {source_uri: src})


class TestMergeOnVRT:
    @pytest.mark.asyncio
    async def test_merge_two_vrts_native_fast_path(self):
        """merge() dispatches through each VRT's `_read_native`, which groups
        by source. Two adjacent VRTs should stitch cleanly."""
        from rastera.geo import BBox
        from rastera.merge import merge

        vrt_a = _vrt_with_one_source(
            "s3://b/a.vrt", "s3://b/a.tif",
            origin_x=0.0, width=10, height=10, scale=1.0, fill=1,
        )
        vrt_b = _vrt_with_one_source(
            "s3://b/b.vrt", "s3://b/b.tif",
            origin_x=5.0, width=10, height=10, scale=1.0, fill=2,
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
    async def test_merge_vrt_with_use_overviews_raises(self):
        """use_overviews=True passes an `overview` object to `_read_native`,
        which VRTs can't support across multiple sources."""
        from rastera.geo import BBox
        from rastera.merge import merge

        # Natively 1.0 m/px, request 10.0 m/px to trigger the reprojected
        # path, and give each source a coarse overview so merge selects it.
        vrt_a = _vrt_with_one_source(
            "s3://b/a.vrt", "s3://b/a.tif",
            origin_x=0.0, width=10, height=10, scale=1.0, fill=1,
        )
        vrt_b = _vrt_with_one_source(
            "s3://b/b.vrt", "s3://b/b.tif",
            origin_x=5.0, width=10, height=10, scale=1.0, fill=2,
        )
        for vrt in (vrt_a, vrt_b):
            ov = MagicMock()
            ov.width = 1  # overview_res = native_res * (10/1) = 10.0
            ov.height = 1
            vrt._band_sources[0][0]._geotiff.overviews = [ov]

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
            patch("rastera.reader.from_url"),
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
