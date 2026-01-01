"""Sutherland-Hodgman polygon clipping against halfspaces."""

from typing import Optional

import numpy as np

from .geometry import remove_duplicate_consecutive, is_valid_polygon


def clip_polygon_halfspace(
    vertices: np.ndarray,
    u: np.ndarray,
    v: float,
    eps: float,
    area_eps: float,
) -> Optional[np.ndarray]:
    """Clip a polygon against the halfspace f(x) = u^T x + v >= 0.

    Uses the Sutherland-Hodgman algorithm.

    Args:
        vertices: (N, 2) array of polygon vertices in CCW order
        u: (2,) normal vector
        v: Scalar offset
        eps: Tolerance for inside/outside classification
        area_eps: Minimum area for valid polygon

    Returns:
        Clipped polygon vertices, or None if result is degenerate
    """
    if len(vertices) < 3:
        return None

    def f(p: np.ndarray) -> float:
        """Evaluate the halfspace function at point p."""
        return np.dot(u, p) + v

    def inside(p: np.ndarray) -> bool:
        """Check if point is inside or on the halfspace boundary."""
        return f(p) >= -eps

    def compute_intersection(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """Compute intersection of edge a->b with the halfspace boundary.

        Returns None if edge is parallel to the boundary.
        """
        fa = f(a)
        fb = f(b)

        # Check for parallel edge
        if abs(fa - fb) < eps:
            return None

        t = fa / (fa - fb)
        # Clamp t to [0, 1] to avoid numerical issues
        t = max(0.0, min(1.0, t))
        return a + t * (b - a)

    output = []

    for i in range(len(vertices)):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)]

        a_inside = inside(a)
        b_inside = inside(b)

        if a_inside and b_inside:
            # Both inside: output b
            output.append(b.copy())
        elif a_inside and not b_inside:
            # a inside, b outside: output intersection
            intersection = compute_intersection(a, b)
            if intersection is not None:
                output.append(intersection)
        elif not a_inside and b_inside:
            # a outside, b inside: output intersection then b
            intersection = compute_intersection(a, b)
            if intersection is not None:
                output.append(intersection)
            output.append(b.copy())
        # else: both outside, output nothing

    if len(output) < 3:
        return None

    result = np.array(output, dtype=np.float64)
    result = remove_duplicate_consecutive(result, eps)

    if not is_valid_polygon(result, area_eps):
        return None

    return result


def clip_polygon_halfspace_negative(
    vertices: np.ndarray,
    u: np.ndarray,
    v: float,
    eps: float,
    area_eps: float,
) -> Optional[np.ndarray]:
    """Clip a polygon against the halfspace f(x) = u^T x + v <= 0.

    This is equivalent to clipping against -u^T x - v >= 0.

    Args:
        vertices: (N, 2) array of polygon vertices in CCW order
        u: (2,) normal vector
        v: Scalar offset
        eps: Tolerance for inside/outside classification
        area_eps: Minimum area for valid polygon

    Returns:
        Clipped polygon vertices, or None if result is degenerate
    """
    return clip_polygon_halfspace(vertices, -u, -v, eps, area_eps)


def split_polygon_by_halfspace(
    vertices: np.ndarray,
    u: np.ndarray,
    v: float,
    eps: float,
    area_eps: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Split a polygon into positive and negative regions by a halfspace.

    Args:
        vertices: (N, 2) array of polygon vertices in CCW order
        u: (2,) normal vector
        v: Scalar offset
        eps: Tolerance
        area_eps: Minimum area

    Returns:
        (positive_region, negative_region) tuple where each may be None
    """
    positive = clip_polygon_halfspace(vertices, u, v, eps, area_eps)
    negative = clip_polygon_halfspace_negative(vertices, u, v, eps, area_eps)

    return positive, negative
