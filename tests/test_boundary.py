"""Tests for nnviz.viz.boundary module."""

from pathlib import Path

import numpy as np
import pytest

from nnviz.viz.boundary import (
    extract_boundary_segments,
    render_exact_boundary,
    render_contour_boundary,
)
from nnviz.viz.exact_regions.enumerate import FinalRegion


class TestExtractBoundarySegments:
    """Test boundary segment extraction."""

    def test_region_with_boundary(self):
        """Region where z=0 crosses should produce a segment."""
        # Square region where z(x,y) = x - 0.5
        # z=0 line is x=0.5, which crosses the square
        region = FinalRegion(
            vertices=np.array([
                [0, 0], [1, 0], [1, 1], [0, 1]
            ], dtype=np.float64),
            a=np.array([1.0, 0.0]),
            b=-0.5,
            pattern=b"\x01",
        )

        segments = extract_boundary_segments([region])

        assert len(segments) == 1
        start, end = segments[0]
        # Should be vertical line at x=0.5
        assert abs(start[0] - 0.5) < 1e-6
        assert abs(end[0] - 0.5) < 1e-6

    def test_region_without_boundary(self):
        """Region entirely one side of z=0 should produce no segments."""
        # Square where z(x,y) = x + y + 1 (always > 0)
        region = FinalRegion(
            vertices=np.array([
                [0, 0], [1, 0], [1, 1], [0, 1]
            ], dtype=np.float64),
            a=np.array([1.0, 1.0]),
            b=1.0,
            pattern=b"\x01",
        )

        segments = extract_boundary_segments([region])

        assert len(segments) == 0

    def test_diagonal_boundary(self):
        """Diagonal z=0 line should produce correct segment."""
        # Square where z(x,y) = x + y - 1 (z=0 is diagonal from (0,1) to (1,0))
        region = FinalRegion(
            vertices=np.array([
                [0, 0], [1, 0], [1, 1], [0, 1]
            ], dtype=np.float64),
            a=np.array([1.0, 1.0]),
            b=-1.0,
            pattern=b"\x01",
        )

        segments = extract_boundary_segments([region])

        assert len(segments) == 1
        start, end = segments[0]
        # Should connect (0,1) and (1,0) or vice versa
        points = sorted([tuple(start), tuple(end)])
        expected = sorted([(0.0, 1.0), (1.0, 0.0)])
        assert np.allclose(points[0], expected[0], atol=1e-6)
        assert np.allclose(points[1], expected[1], atol=1e-6)

    def test_multiple_regions(self):
        """Multiple regions with boundaries should produce multiple segments."""
        regions = [
            FinalRegion(
                vertices=np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]], dtype=np.float64),
                a=np.array([1.0, 0.0]),
                b=-0.25,
                pattern=b"\x00",
            ),
            FinalRegion(
                vertices=np.array([[0.5, 0], [1, 0], [1, 1], [0.5, 1]], dtype=np.float64),
                a=np.array([1.0, 0.0]),
                b=-0.75,
                pattern=b"\x01",
            ),
        ]

        segments = extract_boundary_segments(regions)

        # Each region should produce one segment
        assert len(segments) == 2

    def test_empty_regions(self):
        """Empty region list should produce no segments."""
        segments = extract_boundary_segments([])
        assert len(segments) == 0


class TestRenderExactBoundary:
    """Test exact boundary rendering."""

    def test_creates_output_file(self, temp_dir):
        segments = [
            (np.array([0.0, 0.5]), np.array([1.0, 0.5])),
        ]
        output_path = Path(temp_dir) / "boundary.png"

        render_exact_boundary(
            segments,
            domain=((0.0, 1.0), (0.0, 1.0)),
            output_path=str(output_path),
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_empty_segments(self, temp_dir):
        output_path = Path(temp_dir) / "empty_boundary.png"

        render_exact_boundary(
            [],
            domain=((0.0, 1.0), (0.0, 1.0)),
            output_path=str(output_path),
        )

        assert output_path.exists()

    def test_creates_parent_dirs(self, temp_dir):
        segments = [
            (np.array([0.0, 0.5]), np.array([1.0, 0.5])),
        ]
        output_path = Path(temp_dir) / "nested" / "dirs" / "boundary.png"

        render_exact_boundary(
            segments,
            domain=((0.0, 1.0), (0.0, 1.0)),
            output_path=str(output_path),
        )

        assert output_path.exists()


class TestRenderContourBoundary:
    """Test contour boundary rendering (fallback)."""

    def test_creates_output_file(self, temp_dir):
        X = np.linspace(0, 1, 10)
        Y = np.linspace(0, 1, 10)
        X_grid, Y_grid = np.meshgrid(X, Y)
        Z = X_grid - 0.5  # z=0 at x=0.5

        output_path = Path(temp_dir) / "contour.png"

        render_contour_boundary(
            X_grid, Y_grid, Z,
            output_path=str(output_path),
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_no_zero_crossing(self, temp_dir):
        """Should handle case where z never crosses 0."""
        X = np.linspace(0, 1, 10)
        Y = np.linspace(0, 1, 10)
        X_grid, Y_grid = np.meshgrid(X, Y)
        Z = X_grid + Y_grid + 1  # Always positive

        output_path = Path(temp_dir) / "no_contour.png"

        render_contour_boundary(
            X_grid, Y_grid, Z,
            output_path=str(output_path),
        )

        assert output_path.exists()

    def test_creates_parent_dirs(self, temp_dir):
        X = np.linspace(0, 1, 10)
        Y = np.linspace(0, 1, 10)
        X_grid, Y_grid = np.meshgrid(X, Y)
        Z = X_grid - 0.5

        output_path = Path(temp_dir) / "nested" / "dirs" / "contour.png"

        render_contour_boundary(
            X_grid, Y_grid, Z,
            output_path=str(output_path),
        )

        assert output_path.exists()
