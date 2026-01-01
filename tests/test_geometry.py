"""Tests for nnviz.viz.exact_regions.geometry module."""

import numpy as np
import pytest

from nnviz.viz.exact_regions.geometry import (
    polygon_area,
    points_equal,
    remove_duplicate_consecutive,
    is_valid_polygon,
    create_domain_rectangle,
)


class TestPolygonArea:
    """Test polygon area calculation."""

    def test_unit_square(self, unit_square_vertices):
        area = polygon_area(unit_square_vertices)
        assert abs(area - 1.0) < 1e-10

    def test_unit_triangle(self, unit_triangle_vertices):
        area = polygon_area(unit_triangle_vertices)
        assert abs(area - 0.5) < 1e-10

    def test_rectangle(self):
        # 2x3 rectangle
        vertices = np.array([
            [0, 0], [2, 0], [2, 3], [0, 3]
        ], dtype=np.float64)
        area = polygon_area(vertices)
        assert abs(area - 6.0) < 1e-10

    def test_clockwise_gives_negative_area(self):
        # Unit square in CW order
        vertices = np.array([
            [0, 0], [0, 1], [1, 1], [1, 0]
        ], dtype=np.float64)
        area = polygon_area(vertices)
        assert abs(area + 1.0) < 1e-10  # Should be -1

    def test_degenerate_polygon(self):
        # Fewer than 3 vertices
        vertices = np.array([[0, 0], [1, 1]], dtype=np.float64)
        area = polygon_area(vertices)
        assert area == 0.0

    def test_empty_polygon(self):
        vertices = np.array([], dtype=np.float64).reshape(0, 2)
        area = polygon_area(vertices)
        assert area == 0.0


class TestPointsEqual:
    """Test point equality within tolerance."""

    def test_same_point(self):
        p = np.array([1.0, 2.0])
        assert points_equal(p, p.copy(), 1e-9)

    def test_close_points(self):
        p1 = np.array([1.0, 2.0])
        p2 = np.array([1.0 + 1e-10, 2.0 - 1e-10])
        assert points_equal(p1, p2, 1e-9)

    def test_far_points(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        assert not points_equal(p1, p2, 1e-9)

    def test_at_boundary(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1e-9, 0.0])
        # Distance is exactly eps, so should return False (distance < eps)
        assert not points_equal(p1, p2, 1e-9)


class TestRemoveDuplicateConsecutive:
    """Test consecutive duplicate removal."""

    def test_no_duplicates(self, unit_square_vertices):
        result = remove_duplicate_consecutive(unit_square_vertices, 1e-9)
        assert len(result) == 4
        np.testing.assert_array_almost_equal(result, unit_square_vertices)

    def test_consecutive_duplicates(self):
        vertices = np.array([
            [0, 0], [0, 0], [1, 0], [1, 1], [1, 1], [0, 1]
        ], dtype=np.float64)
        result = remove_duplicate_consecutive(vertices, 1e-9)
        assert len(result) == 4

    def test_wrap_around_duplicate(self):
        # Last vertex same as first
        vertices = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]
        ], dtype=np.float64)
        result = remove_duplicate_consecutive(vertices, 1e-9)
        assert len(result) == 4
        # Should not include the duplicate at the end

    def test_empty_array(self):
        vertices = np.array([], dtype=np.float64).reshape(0, 2)
        result = remove_duplicate_consecutive(vertices, 1e-9)
        assert len(result) == 0

    def test_near_duplicates(self):
        # Points very close together
        vertices = np.array([
            [0, 0], [0 + 1e-12, 0], [1, 0], [1, 1], [0, 1]
        ], dtype=np.float64)
        result = remove_duplicate_consecutive(vertices, 1e-9)
        assert len(result) == 4


class TestIsValidPolygon:
    """Test polygon validity checking."""

    def test_valid_triangle(self, unit_triangle_vertices):
        assert is_valid_polygon(unit_triangle_vertices, 1e-12)

    def test_valid_square(self, unit_square_vertices):
        assert is_valid_polygon(unit_square_vertices, 1e-12)

    def test_too_few_vertices(self):
        vertices = np.array([[0, 0], [1, 1]], dtype=np.float64)
        assert not is_valid_polygon(vertices, 1e-12)

    def test_zero_area_polygon(self):
        # Collinear points
        vertices = np.array([
            [0, 0], [0.5, 0], [1, 0]
        ], dtype=np.float64)
        assert not is_valid_polygon(vertices, 1e-12)

    def test_tiny_area_polygon(self):
        # Very small triangle
        vertices = np.array([
            [0, 0], [1e-7, 0], [0, 1e-7]
        ], dtype=np.float64)
        # Area = 0.5 * 1e-7 * 1e-7 = 5e-15 < 1e-12
        assert not is_valid_polygon(vertices, 1e-12)

    def test_borderline_area(self):
        # Triangle with area just above threshold
        # Need area > 1e-12, so use slightly larger triangle
        vertices = np.array([
            [0, 0], [1e-5, 0], [0, 1e-5]
        ], dtype=np.float64)
        # Area = 0.5 * 1e-5 * 1e-5 = 5e-11 > 1e-12
        assert is_valid_polygon(vertices, 1e-12)


class TestCreateDomainRectangle:
    """Test domain rectangle creation."""

    def test_unit_domain(self):
        rect = create_domain_rectangle((0.0, 1.0), (0.0, 1.0))
        expected = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ], dtype=np.float64)
        np.testing.assert_array_equal(rect, expected)

    def test_arbitrary_domain(self):
        rect = create_domain_rectangle((-1.0, 2.0), (0.5, 3.5))
        expected = np.array([
            [-1, 0.5], [2, 0.5], [2, 3.5], [-1, 3.5]
        ], dtype=np.float64)
        np.testing.assert_array_equal(rect, expected)

    def test_is_ccw(self):
        rect = create_domain_rectangle((0.0, 1.0), (0.0, 1.0))
        # CCW should have positive area
        area = polygon_area(rect)
        assert area > 0

    def test_is_float64(self):
        rect = create_domain_rectangle((0.0, 1.0), (0.0, 1.0))
        assert rect.dtype == np.float64
