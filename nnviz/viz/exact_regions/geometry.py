"""Geometry utilities for polygon operations."""

import numpy as np


def polygon_area(vertices: np.ndarray) -> float:
    """Compute the signed area of a polygon using the shoelace formula.

    Args:
        vertices: (N, 2) array of vertices in CCW order

    Returns:
        Signed area (positive for CCW, negative for CW)
    """
    if len(vertices) < 3:
        return 0.0

    # Shoelace formula: 0.5 * sum((x_i * y_{i+1} - x_{i+1} * y_i))
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Roll to get next vertices
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    return 0.5 * np.sum(x * y_next - x_next * y)


def points_equal(p1: np.ndarray, p2: np.ndarray, eps: float) -> bool:
    """Check if two points are within eps distance.

    Args:
        p1: First point (2,)
        p2: Second point (2,)
        eps: Distance tolerance

    Returns:
        True if distance < eps
    """
    return np.linalg.norm(p1 - p2) < eps


def remove_duplicate_consecutive(vertices: np.ndarray, eps: float) -> np.ndarray:
    """Remove consecutive duplicate vertices from a polygon.

    Also handles the wrap-around case (last vertex equal to first).

    Args:
        vertices: (N, 2) array of vertices
        eps: Distance tolerance for equality

    Returns:
        Filtered vertices array
    """
    if len(vertices) == 0:
        return vertices

    result = [vertices[0]]

    for i in range(1, len(vertices)):
        if not points_equal(vertices[i], result[-1], eps):
            result.append(vertices[i])

    # Check wrap-around: if last == first, remove last
    if len(result) > 1 and points_equal(result[-1], result[0], eps):
        result.pop()

    return np.array(result, dtype=np.float64)


def is_valid_polygon(vertices: np.ndarray, area_eps: float) -> bool:
    """Check if a polygon is valid (>= 3 vertices and area > area_eps).

    Args:
        vertices: (N, 2) array of vertices
        area_eps: Minimum area threshold

    Returns:
        True if valid polygon
    """
    if len(vertices) < 3:
        return False

    area = abs(polygon_area(vertices))
    return area > area_eps


def create_domain_rectangle(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> np.ndarray:
    """Create a rectangular domain as CCW vertices.

    Args:
        x_bounds: (x_min, x_max)
        y_bounds: (y_min, y_max)

    Returns:
        (4, 2) array of vertices in CCW order
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    return np.array([
        [x_min, y_min],  # Bottom-left
        [x_max, y_min],  # Bottom-right
        [x_max, y_max],  # Top-right
        [x_min, y_max],  # Top-left
    ], dtype=np.float64)
