"""Unit tests for pure geometry, parsing, and utility functions."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine

from rastera.geo import (
    BBox,
    WindowOutOfRangeError,
    bounds_from_transform,
    compute_paste_slices,
    ensure_bbox,
    normalize_band_indices,
    transform_bbox,
    validate_resolution,
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


# ── ensure_bbox ───────────────────────────────────────────────────────────


class TestEnsureBbox:
    """Every consumer takes min()/max() of the corners, so an inverted box used
    to be silently swapped into a different — often enormous — extent."""

    def test_tuple_and_bbox_agree(self):
        assert ensure_bbox((0, 1, 2, 3)) == BBox(0, 1, 2, 3)
        b = BBox(0, 1, 2, 3)
        assert ensure_bbox(b) is b

    def test_antimeridian_bbox_rejected(self):
        """A GeoJSON dateline bbox: min()/max() turned this into the
        complementary 340 degrees of longitude, 17x the requested width."""
        with pytest.raises(ValueError, match="minx < maxx"):
            ensure_bbox((170.0, -10.0, -170.0, 10.0))

    @pytest.mark.parametrize(
        "bbox",
        [
            (10.0, 0.0, 5.0, 10.0),  # minx > maxx
            (0.0, 10.0, 10.0, 5.0),  # miny > maxy
            (5.0, 0.0, 5.0, 10.0),  # zero width
            (0.0, 5.0, 10.0, 5.0),  # zero height
        ],
    )
    def test_degenerate_rejected(self, bbox: tuple[float, float, float, float]):
        with pytest.raises(ValueError, match="minx < maxx"):
            ensure_bbox(bbox)

    def test_already_a_bbox_is_still_validated(self):
        """The isinstance short-circuit used to skip the checks entirely."""
        with pytest.raises(ValueError, match="minx < maxx"):
            ensure_bbox(BBox(10.0, 0.0, 5.0, 10.0))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected(self, bad: float):
        with pytest.raises(ValueError, match="finite"):
            ensure_bbox((0.0, 0.0, bad, 10.0))

    def test_internal_constructions_are_not_validated(self):
        """Only caller-supplied bboxes route through ensure_bbox; BBox itself
        stays permissive because intersect() builds empty boxes on purpose."""
        assert BBox(10, 0, 5, 10).width == -5


# ── normalize_band_indices ────────────────────────────────────────────────


class TestNormalizeBandIndices:
    def test_none_selects_all(self):
        assert normalize_band_indices(None, 3) == [0, 1, 2]

    def test_converts_to_zero_based(self):
        assert normalize_band_indices([1, 3], 3) == [0, 2]

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            normalize_band_indices([], 3)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_below_one_rejected(self, bad: int):
        with pytest.raises(ValueError, match="1-based"):
            normalize_band_indices([bad], 3)

    def test_above_count_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            normalize_band_indices([4], 3)

    def test_float_rejected(self):
        """1.9 passed the range checks and subtracted to 0.899..., which died
        several frames later inside NumPy's indexing."""
        with pytest.raises(ValueError, match="must be integers"):
            normalize_band_indices([1.9, 2.1], 4)  # type: ignore[list-item]

    def test_whole_float_also_rejected(self):
        """2.0 would happen to work; accepting it makes the contract fuzzy."""
        with pytest.raises(ValueError, match="must be integers"):
            normalize_band_indices([2.0], 4)  # type: ignore[list-item]

    def test_numpy_integer_accepted(self):
        """np.integer is not int to a type checker, but it indexes fine and
        arrives this way from any array-derived band list."""
        assert normalize_band_indices([np.int64(2)], 3) == [1]  # type: ignore[list-item]

    def test_bool_rejected(self):
        """bool is an int subclass, so True would silently mean "first band"."""
        with pytest.raises(ValueError, match="must be integers"):
            normalize_band_indices([True], 3)


# ── validate_resolution ───────────────────────────────────────────────────


class TestValidateResolution:
    def test_positive_accepted(self):
        validate_resolution(10.0)
        validate_resolution(1)

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            (0.0, "> 0"),  # ZeroDivisionError deep in _grid_for_bbox
            (-1.0, "> 0"),  # silent 1x1 array with a mirrored transform
            (float("nan"), "> 0"),  # "cannot convert float NaN to integer"
            (float("inf"), "> 0"),  # silent 1x1 array with an inf transform
        ],
    )
    def test_rejected(self, bad: float, match: str):
        with pytest.raises(ValueError, match=match):
            validate_resolution(bad)

    def test_non_number_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_resolution("10")  # type: ignore[arg-type]


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
        with pytest.raises(ValueError, match="area of use"):
            transform_bbox(BBox(0.0, 80.0, 10.0, 95.0), 4326, 32632)

    def test_interior_pole_reaches_90(self):
        """The NSIDC EPSG:3413 sea-ice grid contains the north pole as an
        interior point; hulling densified *edges* tops out near lat 56.
        """
        result = transform_bbox(BBox(-3850000, -5350000, 3750000, 5850000), 3413, 4326)
        assert result.maxy == pytest.approx(90.0)
        assert result.minx == pytest.approx(-180.0)
        assert result.maxx == pytest.approx(180.0)

    def test_antimeridian_crossing_raises(self):
        """UTM zone 60N straddles the dateline; its 4326 envelope wraps."""
        with pytest.raises(ValueError, match="antimeridian"):
            transform_bbox(BBox(166021, 0, 833978, 9329005), 32601, 4326)

    def test_roundtrip(self):
        bbox = BBox(10.0, 50.0, 11.0, 51.0)  # lon/lat
        projected = transform_bbox(bbox, 4326, 32632)
        assert projected.width > 0 and projected.height > 0
        back = transform_bbox(projected, 32632, 4326)
        # Roundtrip is lossy (envelope of sampled points), but should contain original
        assert back.minx <= bbox.minx and back.maxx >= bbox.maxx
        assert back.miny <= bbox.miny and back.maxy >= bbox.maxy
