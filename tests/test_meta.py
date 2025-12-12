"""Unit tests for Profile metadata."""

from rastera.geo import Window
from tests.conftest import make_profile


class TestProfile:
    def test_copy(self):
        p = make_profile()
        p2 = p.copy()
        assert p.width == p2.width
        assert p.transform == p2.transform
        assert p is not p2

    def test_adjust_to_window(self):
        p = make_profile()
        w = Window(10, 50, 20, 80)
        adjusted = p.adjust_to_window(w)
        assert adjusted.width == 40
        assert adjusted.height == 60
        assert adjusted.bounds.width > 0
        assert adjusted.bounds.height > 0
