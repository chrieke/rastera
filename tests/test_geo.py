"""Unit tests for pure geometry, parsing, and utility functions."""

import pytest
from affine import Affine

from rastera.geo import (
    BBox,
    WindowOutOfRangeError,
    compute_paste_slices,
    transform_bbox,
    window_from_bbox,
)
from rastera.reader import _extract_key
from tests.conftest import make_meta

# ── BBox ──────────────────────────────────────────────────────────────────


class TestBBox:
    def test_properties(self):
        b = BBox(0, 0, 10, 5)
        assert b.width == 10
        assert b.height == 5

    def test_iter_and_unpack(self):
        b = BBox(1, 2, 3, 4)
        assert list(b) == [1, 2, 3, 4]
        minx, miny, maxx, maxy = b
        assert (minx, miny, maxx, maxy) == (1, 2, 3, 4)

    def test_intersect_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 15, 15)
        c = a.intersect(b)
        assert c == BBox(5, 5, 10, 10)

    def test_intersect_no_overlap(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(10, 10, 15, 15)
        assert a.intersect(b) is None

    def test_intersect_edge_touch(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(5, 0, 10, 5)
        assert a.intersect(b) is None  # touching edge = no area

    def test_intersect_contained(self):
        outer = BBox(0, 0, 10, 10)
        inner = BBox(2, 2, 8, 8)
        assert outer.intersect(inner) == inner


# ── Window ────────────────────────────────────────────────────────────────


class TestWindow:
    def test_from_bbox_full(self):
        p = make_meta()
        w = window_from_bbox(p, BBox(0, 0, 1000, 1000))  # type: ignore[reportArgumentType]
        assert w.col_off == 0 and w.width == 100
        assert w.row_off == 0 and w.height == 100

    def test_from_bbox_subset(self):
        p = make_meta()
        w = window_from_bbox(p, BBox(100, 200, 500, 800))  # type: ignore[reportArgumentType]
        assert w.width > 0 and w.height > 0
        assert w.col_off >= 10 and w.col_off + w.width <= 50

    def test_from_bbox_no_intersect(self):
        p = make_meta()
        with pytest.raises(WindowOutOfRangeError, match="does not intersect"):
            window_from_bbox(p, BBox(2000, 2000, 3000, 3000))  # type: ignore[reportArgumentType]

    def test_from_bbox_subpixel_overlap_raises(self):
        # make_meta(): 100x100 grid at 10 m/px, x in [0, 1000].
        # bbox spans 0.1 m on x (= 0.01 px) — floor(0.01 + 0.5) = 0,
        # so the rounded window has zero width and window_from_bbox must raise.
        p = make_meta()
        with pytest.raises(WindowOutOfRangeError, match="does not intersect"):
            window_from_bbox(p, BBox(999.9, 0, 1000.0, 1000))  # type: ignore[reportArgumentType]

    def test_from_bbox_clamps(self):
        p = make_meta()
        # bbox extends beyond image
        w = window_from_bbox(p, BBox(-500, -500, 500, 500))  # type: ignore[reportArgumentType]
        assert w.col_off == 0
        assert w.col_off + w.width <= 100 and w.row_off + w.height <= 100
        assert w.width > 0 and w.height > 0


# ── compute_paste_slices ──────────────────────────────────────────────────


class TestComputePasteSlices:
    def test_aligned_paste(self):
        # src profile has origin at (0, 500) in world coords (north-up)
        src = make_meta(width=50, height=50, scale=10.0)
        dst_transform = Affine(10, 0, 0, 0, -10, 1000)
        result = compute_paste_slices(
            src=src,  # type: ignore[reportArgumentType]
            dst_transform=dst_transform,
            dst_width=100,
            dst_height=100,
        )
        assert result is not None
        dst_rows, dst_cols, src_rows, src_cols = result
        assert dst_cols == slice(0, 50)
        assert dst_rows.stop - dst_rows.start == 50  # correct height

    def test_no_overlap(self):
        src = make_meta(width=50, height=50, scale=10.0)
        # destination is far away
        dst_transform = Affine(10, 0, 5000, 0, -10, 10000)
        result = compute_paste_slices(
            src=src,  # type: ignore[reportArgumentType]
            dst_transform=dst_transform,
            dst_width=100,
            dst_height=100,
        )
        assert result is None


# ── _extract_key ─────────────────────────────────────────────────────────


class TestExtractKey:
    def test_s3_scheme(self):
        assert _extract_key("s3://my-bucket/path/to/file.tif") == "path/to/file.tif"

    def test_virtual_hosted_style(self):
        assert (
            _extract_key(
                "https://my-bucket.s3.us-west-2.amazonaws.com/path/to/file.tif"
            )
            == "path/to/file.tif"
        )

    def test_path_style(self):
        assert (
            _extract_key(
                "https://s3.us-west-2.amazonaws.com/my-bucket/path/to/file.tif"
            )
            == "path/to/file.tif"
        )

    def test_local_path(self, tmp_path):
        f = tmp_path / "file.tif"
        f.write_bytes(b"")
        assert _extract_key(str(f)) == "file.tif"

    def test_file_scheme(self, tmp_path):
        f = tmp_path / "file.tif"
        f.write_bytes(b"")
        assert _extract_key(f.as_uri()) == "file.tif"


# ── transform_bbox ───────────────────────────────────────────────────────


class TestTransformBbox:
    def test_same_crs_noop(self):
        bbox = BBox(500000, 5000000, 600000, 5100000)
        result = transform_bbox(bbox, 32632, 32632)
        assert result == bbox

    def test_roundtrip(self):
        bbox = BBox(10.0, 50.0, 11.0, 51.0)  # lon/lat
        projected = transform_bbox(bbox, 4326, 32632)
        assert projected.width > 0 and projected.height > 0
        back = transform_bbox(projected, 32632, 4326)
        # Roundtrip is lossy (envelope of sampled points), but should contain original
        assert back.minx <= bbox.minx and back.maxx >= bbox.maxx
        assert back.miny <= bbox.miny and back.maxy >= bbox.maxy
