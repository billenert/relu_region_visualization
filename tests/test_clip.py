"""Tests for nnviz.viz.exact_regions.clip module."""

import numpy as np
import pytest

from nnviz.viz.exact_regions.clip import (
    clip_polygon_halfspace,
    clip_polygon_halfspace_negative,
    split_polygon_by_halfspace,
)
from nnviz.viz.exact_regions.geometry import polygon_area


EPS = 1e-9
AREA_EPS = 1e-12


class TestClipPolygonHalfspace:
    """Test Sutherland-Hodgman polygon clipping."""

    def test_fully_inside(self, unit_square_vertices):
        """Polygon fully inside halfspace should be unchanged."""
        # Halfspace x >= -1 (entire square is inside)
        u = np.array([1.0, 0.0])
        v = 1.0  # x + 1 >= 0 means x >= -1

        result = clip_polygon_halfspace(unit_square_vertices, u, v, EPS, AREA_EPS)

        assert result is not None
        assert len(result) == 4
        assert abs(polygon_area(result) - 1.0) < 1e-9

    def test_fully_outside(self, unit_square_vertices):
        """Polygon fully outside halfspace should return None."""
        # Halfspace x >= 2 (entire square is outside)
        u = np.array([1.0, 0.0])
        v = -2.0  # x - 2 >= 0 means x >= 2

        result = clip_polygon_halfspace(unit_square_vertices, u, v, EPS, AREA_EPS)

        assert result is None

    def test_horizontal_cut(self, unit_square_vertices):
        """Cut unit square horizontally in half."""
        # Halfspace y >= 0.5
        u = np.array([0.0, 1.0])
        v = -0.5  # y - 0.5 >= 0 means y >= 0.5

        result = clip_polygon_halfspace(unit_square_vertices, u, v, EPS, AREA_EPS)

        assert result is not None
        # Should be a rectangle from y=0.5 to y=1
        assert abs(polygon_area(result) - 0.5) < 1e-9

    def test_vertical_cut(self, unit_square_vertices):
        """Cut unit square vertically in half."""
        # Halfspace x >= 0.5
        u = np.array([1.0, 0.0])
        v = -0.5

        result = clip_polygon_halfspace(unit_square_vertices, u, v, EPS, AREA_EPS)

        assert result is not None
        assert abs(polygon_area(result) - 0.5) < 1e-9

    def test_diagonal_cut(self, unit_square_vertices):
        """Cut unit square diagonally."""
        # Halfspace x + y >= 1 (upper-right triangle)
        u = np.array([1.0, 1.0])
        v = -1.0

        result = clip_polygon_halfspace(unit_square_vertices, u, v, EPS, AREA_EPS)

        assert result is not None
        # Should be a triangle with area 0.5
        assert abs(polygon_area(result) - 0.5) < 1e-9
        assert len(result) == 3

    def test_corner_clip(self, unit_square_vertices):
        """Clip a small corner off the square."""
        # Halfspace x + y >= 0.1 (removes tiny corner at origin)
        u = np.array([1.0, 1.0])
        v = -0.1

        result = clip_polygon_halfspace(unit_square_vertices, u, v, EPS, AREA_EPS)

        assert result is not None
        # Area should be slightly less than 1
        area = polygon_area(result)
        assert 0.99 < area < 1.0
        assert len(result) == 5  # Pentagon after corner clip

    def test_through_vertex(self, unit_square_vertices):
        """Halfspace passes exactly through a vertex."""
        # Halfspace x + y >= 1 passes through (1,0) and (0,1)
        u = np.array([1.0, 1.0])
        v = -1.0

        result = clip_polygon_halfspace(unit_square_vertices, u, v, EPS, AREA_EPS)

        assert result is not None
        # Triangle with vertices at (1,0), (1,1), (0,1)
        assert len(result) == 3


class TestClipPolygonHalfspaceNegative:
    """Test clipping against the negative halfspace."""

    def test_negative_halfspace(self, unit_square_vertices):
        """Clip against x + y <= 1 (complement of diagonal cut)."""
        u = np.array([1.0, 1.0])
        v = -1.0  # x + y - 1 <= 0 means x + y <= 1

        result = clip_polygon_halfspace_negative(
            unit_square_vertices, u, v, EPS, AREA_EPS
        )

        assert result is not None
        # Should be a triangle with area 0.5 (lower-left)
        assert abs(polygon_area(result) - 0.5) < 1e-9

    def test_complementary_clips(self, unit_square_vertices):
        """Positive and negative clips should sum to original area."""
        u = np.array([1.0, 0.5])
        v = -0.75

        pos = clip_polygon_halfspace(unit_square_vertices, u, v, EPS, AREA_EPS)
        neg = clip_polygon_halfspace_negative(unit_square_vertices, u, v, EPS, AREA_EPS)

        # Both should exist and areas should sum to 1
        assert pos is not None
        assert neg is not None
        total_area = polygon_area(pos) + polygon_area(neg)
        assert abs(total_area - 1.0) < 1e-9


class TestSplitPolygonByHalfspace:
    """Test splitting polygon into positive and negative regions."""

    def test_split_in_half(self, unit_square_vertices):
        """Split unit square in half vertically."""
        u = np.array([1.0, 0.0])
        v = -0.5

        pos, neg = split_polygon_by_halfspace(
            unit_square_vertices, u, v, EPS, AREA_EPS
        )

        assert pos is not None
        assert neg is not None
        assert abs(polygon_area(pos) - 0.5) < 1e-9
        assert abs(polygon_area(neg) - 0.5) < 1e-9

    def test_split_asymmetric(self, unit_square_vertices):
        """Split at x >= 0.75."""
        u = np.array([1.0, 0.0])
        v = -0.75

        pos, neg = split_polygon_by_halfspace(
            unit_square_vertices, u, v, EPS, AREA_EPS
        )

        assert pos is not None
        assert neg is not None
        assert abs(polygon_area(pos) - 0.25) < 1e-9
        assert abs(polygon_area(neg) - 0.75) < 1e-9

    def test_one_side_empty(self, unit_square_vertices):
        """Split where one side is empty."""
        # x >= 2 (all outside)
        u = np.array([1.0, 0.0])
        v = -2.0

        pos, neg = split_polygon_by_halfspace(
            unit_square_vertices, u, v, EPS, AREA_EPS
        )

        assert pos is None
        assert neg is not None
        assert abs(polygon_area(neg) - 1.0) < 1e-9


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_degenerate_input(self):
        """Degenerate input (< 3 vertices) should return None."""
        vertices = np.array([[0, 0], [1, 1]], dtype=np.float64)
        u = np.array([1.0, 0.0])
        v = 0.0

        result = clip_polygon_halfspace(vertices, u, v, EPS, AREA_EPS)
        assert result is None

    def test_clip_to_line(self):
        """Clipping that would result in a line should return None."""
        vertices = np.array([
            [0, 0], [1, 0], [1, 0.5], [0, 0.5]
        ], dtype=np.float64)

        # Clip at y >= 0.5 (top edge only)
        u = np.array([0.0, 1.0])
        v = -0.5

        result = clip_polygon_halfspace(vertices, u, v, EPS, AREA_EPS)
        # Result should be a line (degenerate), so None
        assert result is None

    def test_numerical_precision(self, unit_square_vertices):
        """Test with very small epsilon."""
        u = np.array([1.0, 0.0])
        v = -0.5

        result = clip_polygon_halfspace(
            unit_square_vertices, u, v, 1e-15, 1e-18
        )

        assert result is not None
        assert abs(polygon_area(result) - 0.5) < 1e-9

    def test_parallel_edge(self):
        """Edge parallel to the clipping line."""
        vertices = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ], dtype=np.float64)

        # Clip at y >= 0 (bottom edge is on the boundary)
        u = np.array([0.0, 1.0])
        v = 0.0

        result = clip_polygon_halfspace(vertices, u, v, EPS, AREA_EPS)

        assert result is not None
        assert abs(polygon_area(result) - 1.0) < 1e-9
