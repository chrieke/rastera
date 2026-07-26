"""Unit tests for pure geometry, parsing, and utility functions."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from affine import Affine

from rastera.geo import (
    BBox,
    WindowOutOfRangeError,
    bounds_from_transform,
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

    def test_from_bbox_subpixel_overlap_raises_unsnapped(self):
        # make_meta(): 100x100 grid at 10 m/px, x in [0, 1000].
        # bbox spans 0.1 m on x (= 0.01 px) — floor(0.01 + 0.5) = 0,
        # so the rounded window has zero width and window_from_bbox must raise.
        p = make_meta()
        with pytest.raises(WindowOutOfRangeError, match="does not intersect"):
            window_from_bbox(p, BBox(999.9, 0, 1000.0, 1000), snap_to_grid=False)  # type: ignore[reportArgumentType]

    def test_from_bbox_subpixel_overlap_snaps_to_whole_pixel(self):
        p = make_meta()
        w = window_from_bbox(p, BBox(999.9, 0, 1000.0, 1000))  # type: ignore[reportArgumentType]
        assert w.col_off == 99 and w.width == 1

    def test_from_bbox_clamps(self):
        p = make_meta()
        # bbox extends beyond image
        w = window_from_bbox(p, BBox(-500, -500, 500, 500))  # type: ignore[reportArgumentType]
        assert w.col_off == 0
        assert w.col_off + w.width <= 100 and w.row_off + w.height <= 100
        assert w.width > 0 and w.height > 0

    @pytest.mark.parametrize(
        ("bbox", "where"),
        [
            (BBox(-500, 100, -100, 500), "entirely left"),
            (BBox(1100, 100, 1500, 500), "entirely right"),
            (BBox(100, 1100, 500, 1500), "entirely above"),
            (BBox(100, -500, 500, -100), "entirely below"),
        ],
    )
    def test_from_bbox_outside_image_raises(self, bbox: BBox, where: str):
        """Clamping only the offset left the span positive, so a bbox left of or
        above the image returned a plausible window over the wrong pixels."""
        p = make_meta()
        with pytest.raises(WindowOutOfRangeError, match="does not intersect"):
            window_from_bbox(p, bbox)  # type: ignore[reportArgumentType]

    def test_snapped_keeps_pixel_wholly_inside_bbox(self):
        """Unsnapped sizing floors the near edge but *rounds* the span, so it
        can drop a source pixel that lies entirely within the request."""
        p = make_meta()  # 100x100 at 10 m/px, x in [0, 1000]
        # x 8-20 spans columns 0.8-2.0: column 1 is wholly inside the bbox.
        bbox = BBox(8, 990, 20, 1000)
        assert window_from_bbox(p, bbox, snap_to_grid=False).width == 1  # type: ignore[reportArgumentType]
        assert window_from_bbox(p, bbox).width == 2  # type: ignore[reportArgumentType]

    def test_snapped_keeps_last_column_at_image_edge(self):
        """The merge-seam case: clipping the far edge to the image and then
        rounding the span drops the image's final column."""
        p = make_meta()
        bbox = BBox(8, 0, 1500, 1000)  # 99.2 px, running past the right edge
        assert window_from_bbox(p, bbox, snap_to_grid=False).width == 99  # type: ignore[reportArgumentType]
        w = window_from_bbox(p, bbox)  # type: ignore[reportArgumentType]
        assert w.col_off == 0 and w.width == 100

    @pytest.mark.parametrize(
        ("origin", "res"),
        [(655000.0, 30.0), (234567.0, 30.0), (-180.0, 0.0001), (511.7, 0.5)],
    )
    def test_snapped_grid_aligned_bbox_is_exact(self, origin: float, res: float):
        """Rounding outward must not buy a pixel off ``~transform`` ULP error."""
        p = SimpleNamespace(
            width=20000,
            height=20000,
            transform=Affine(res, 0, origin, 0, -res, origin + 20000 * res),
        )
        top = origin + 20000 * res
        for c0, r0, size in ((11593, 0, 512), (1, 7777, 256), (19487, 19487, 513)):
            bbox = BBox(
                origin + c0 * res,
                top - (r0 + size) * res,
                origin + (c0 + size) * res,
                top - r0 * res,
            )
            w = window_from_bbox(p, bbox)  # type: ignore[reportArgumentType]
            assert (w.col_off, w.row_off, w.width, w.height) == (c0, r0, size, size)

    def test_snapped_stays_within_image(self):
        p = make_meta()
        w = window_from_bbox(p, BBox(-500, -500, 1500, 1500))  # type: ignore[reportArgumentType]
        assert (w.col_off, w.width, w.row_off, w.height) == (0, 100, 0, 100)

    def test_from_bbox_partial_overlap_stays_inside(self):
        # Overlap is x in [0, 500] -> 50 px, not the full 100 px span.
        p = make_meta()
        w = window_from_bbox(p, BBox(-500, 0, 500, 1000))  # type: ignore[reportArgumentType]
        assert w.col_off == 0 and w.width == 50


class TestBoundsFromTransform:
    def test_north_up(self):
        b = bounds_from_transform(Affine(10, 0, 100, 0, -10, 900), 20, 30)
        assert (b.minx, b.miny, b.maxx, b.maxy) == (100, 600, 300, 900)

    def test_rotated_takes_hull_of_four_corners(self):
        """Using only corners (0,0) and (w,h) collapsed a rotated transform to a
        degenerate box."""
        b = bounds_from_transform(Affine.rotation(45) * Affine.scale(1, -1), 10, 10)
        assert b.width > 0 and b.height > 0


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

    def test_local_path(self, tmp_path: Path):
        f = tmp_path / "file.tif"
        f.write_bytes(b"")
        assert _extract_key(str(f)) == "file.tif"

    def test_file_scheme(self, tmp_path: Path):
        f = tmp_path / "file.tif"
        f.write_bytes(b"")
        assert _extract_key(f.as_uri()) == "file.tif"


# ── transform_bbox ───────────────────────────────────────────────────────


class TestTransformBbox:
    def test_same_crs_noop(self):
        bbox = BBox(500000, 5000000, 600000, 5100000)
        result = transform_bbox(bbox, 32632, 32632)
        assert result == bbox

    def test_outside_target_domain_raises(self):
        """A partially non-finite transform used to silently return the hull of
        the finite subset, under-covering the request. A grid rounded outward
        past the pole (ceil in _grid_for_bbox) reaches this.
        """
        with pytest.raises(ValueError, match="inf/nan"):
            transform_bbox(BBox(0.0, 80.0, 10.0, 95.0), 4326, 32632)

    def test_roundtrip(self):
        bbox = BBox(10.0, 50.0, 11.0, 51.0)  # lon/lat
        projected = transform_bbox(bbox, 4326, 32632)
        assert projected.width > 0 and projected.height > 0
        back = transform_bbox(projected, 32632, 4326)
        # Roundtrip is lossy (envelope of sampled points), but should contain original
        assert back.minx <= bbox.minx and back.maxx >= bbox.maxx
        assert back.miny <= bbox.miny and back.maxy >= bbox.maxy
