"""Tests for nnviz.viz.exact_regions.render module."""

from pathlib import Path

import numpy as np
import pytest

from nnviz.viz.exact_regions.render import pattern_to_color, render_regions
from nnviz.viz.exact_regions.enumerate import FinalRegion


class TestPatternToColor:
    """Test pattern to color conversion."""

    def test_returns_rgb_tuple(self):
        pattern = b"\x01\x00\x01\x01"
        color = pattern_to_color(pattern)

        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_color_in_range(self):
        """Color components should be in [0.2, 1.0]."""
        patterns = [
            b"",
            b"\x00",
            b"\xff",
            b"\x01\x02\x03\x04\x05",
            b"\xaa\xbb\xcc\xdd",
        ]

        for pattern in patterns:
            r, g, b = pattern_to_color(pattern)
            assert 0.2 <= r <= 1.0, f"R out of range for {pattern}"
            assert 0.2 <= g <= 1.0, f"G out of range for {pattern}"
            assert 0.2 <= b <= 1.0, f"B out of range for {pattern}"

    def test_deterministic(self):
        """Same pattern should always produce same color."""
        pattern = b"\x01\x02\x03"

        color1 = pattern_to_color(pattern)
        color2 = pattern_to_color(pattern)

        assert color1 == color2

    def test_different_patterns_different_colors(self):
        """Different patterns should (usually) produce different colors."""
        patterns = [b"\x00", b"\x01", b"\x02", b"\x03"]
        colors = [pattern_to_color(p) for p in patterns]

        # At least some should be different
        unique_colors = set(colors)
        assert len(unique_colors) > 1

    def test_empty_pattern(self):
        """Empty pattern should return valid color."""
        color = pattern_to_color(b"")
        assert len(color) == 3
        assert all(0.2 <= c <= 1.0 for c in color)


class TestRenderRegions:
    """Test region rendering."""

    @pytest.fixture
    def simple_regions(self):
        """Create a few simple regions for testing."""
        return [
            FinalRegion(
                vertices=np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]], dtype=np.float64),
                a=np.array([1.0, 0.0]),
                b=0.0,
                pattern=b"\x00",
            ),
            FinalRegion(
                vertices=np.array([[0.5, 0], [1, 0], [1, 1], [0.5, 1]], dtype=np.float64),
                a=np.array([1.0, 0.0]),
                b=0.5,
                pattern=b"\x01",
            ),
        ]

    def test_creates_output_file(self, temp_dir, simple_regions):
        """Should create output image file."""
        output_path = Path(temp_dir) / "regions.png"

        render_regions(
            simple_regions,
            domain=((0.0, 1.0), (0.0, 1.0)),
            output_path=str(output_path),
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_creates_parent_directories(self, temp_dir, simple_regions):
        """Should create parent directories if needed."""
        output_path = Path(temp_dir) / "nested" / "dirs" / "regions.png"

        render_regions(
            simple_regions,
            domain=((0.0, 1.0), (0.0, 1.0)),
            output_path=str(output_path),
        )

        assert output_path.exists()

    def test_respects_domain(self, temp_dir, simple_regions):
        """Should respect the specified domain."""
        output_path = Path(temp_dir) / "regions.png"

        # Use a different domain (regions will be outside, but should still render)
        render_regions(
            simple_regions,
            domain=((-1.0, 2.0), (-1.0, 2.0)),
            output_path=str(output_path),
        )

        assert output_path.exists()

    def test_empty_regions(self, temp_dir):
        """Should handle empty region list."""
        output_path = Path(temp_dir) / "empty.png"

        render_regions(
            [],
            domain=((0.0, 1.0), (0.0, 1.0)),
            output_path=str(output_path),
        )

        assert output_path.exists()

    def test_single_region(self, temp_dir):
        """Should handle single region."""
        output_path = Path(temp_dir) / "single.png"

        regions = [
            FinalRegion(
                vertices=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                a=np.array([1.0, 0.0]),
                b=0.0,
                pattern=b"\x00\x01",
            ),
        ]

        render_regions(
            regions,
            domain=((0.0, 1.0), (0.0, 1.0)),
            output_path=str(output_path),
        )

        assert output_path.exists()

    def test_triangular_region(self, temp_dir):
        """Should handle triangular regions."""
        output_path = Path(temp_dir) / "triangle.png"

        regions = [
            FinalRegion(
                vertices=np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float64),
                a=np.array([1.0, 1.0]),
                b=-0.5,
                pattern=b"\x01\x01",
            ),
        ]

        render_regions(
            regions,
            domain=((0.0, 1.0), (0.0, 1.0)),
            output_path=str(output_path),
        )

        assert output_path.exists()
