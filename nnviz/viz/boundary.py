"""Decision boundary visualization."""

from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from .exact_regions.enumerate import FinalRegion


def extract_boundary_segments(
    regions: List[FinalRegion],
    eps: float = 1e-9,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract decision boundary segments from exact regions.

    For each region where z(x) = a^T x + b, find where z=0 intersects
    the polygon boundary.

    Args:
        regions: List of FinalRegion objects
        eps: Tolerance for zero classification

    Returns:
        List of (start, end) tuples representing line segments
    """
    segments = []

    for region in regions:
        vertices = region.vertices
        a = region.a
        b = region.b

        n = len(vertices)
        if n < 3:
            continue

        # Compute z value at each vertex
        z_values = np.array([np.dot(a, v) + b for v in vertices])

        # Check if all vertices are same sign
        if np.all(z_values > eps) or np.all(z_values < -eps):
            continue  # No boundary in this region

        # Find intersection points with polygon edges
        intersection_points = []

        for i in range(n):
            v_i = vertices[i]
            v_next = vertices[(i + 1) % n]
            z_i = z_values[i]
            z_next = z_values[(i + 1) % n]

            # Check if vertex is on the boundary
            if abs(z_i) <= eps:
                intersection_points.append(v_i.copy())

            # Check if edge crosses z=0
            if (z_i > eps and z_next < -eps) or (z_i < -eps and z_next > eps):
                # Compute intersection point
                t = z_i / (z_i - z_next)
                p = v_i + t * (v_next - v_i)
                intersection_points.append(p)

        if len(intersection_points) < 2:
            continue

        # Deduplicate points
        unique_points = []
        for p in intersection_points:
            is_duplicate = False
            for u in unique_points:
                if np.linalg.norm(p - u) < eps:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(p)

        if len(unique_points) < 2:
            continue

        # If exactly 2 points, that's our segment
        if len(unique_points) == 2:
            segments.append((unique_points[0], unique_points[1]))
        else:
            # More than 2 points (collinear case) - find extremes
            # Project onto the line direction and find min/max
            points_array = np.array(unique_points)

            # Use a as direction if non-zero, else arbitrary
            if np.linalg.norm(a) > eps:
                direction = np.array([-a[1], a[0]])  # Perpendicular to normal
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1.0, 0.0])

            projections = points_array @ direction
            min_idx = np.argmin(projections)
            max_idx = np.argmax(projections)

            if min_idx != max_idx:
                segments.append((points_array[min_idx], points_array[max_idx]))

    return segments


def render_exact_boundary(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    domain: Tuple[Tuple[float, float], Tuple[float, float]],
    output_path: str,
    figsize: Tuple[float, float] = (10, 10),
    linewidth: float = 1.5,
    dpi: int = 150,
) -> None:
    """Render exact decision boundary segments.

    Args:
        segments: List of (start, end) line segments
        domain: ((x_min, x_max), (y_min, y_max)) bounds
        output_path: Path to save the image
        figsize: Figure size in inches
        linewidth: Line width for boundary
        dpi: Resolution of output image
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    for start, end in segments:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            "k-",
            linewidth=linewidth,
        )

    (x_min, x_max), (y_min, y_max) = domain
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    # Remove axis labels for cleaner visualization
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_contour_boundary(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z: np.ndarray,
    output_path: str,
    figsize: Tuple[float, float] = (10, 10),
    linewidth: float = 1.5,
    dpi: int = 150,
) -> None:
    """Render decision boundary using contour at z=0 (fallback method).

    Args:
        X_grid: (R, R) meshgrid of x coordinates
        Y_grid: (R, R) meshgrid of y coordinates
        Z: (R, R) array of logit values
        output_path: Path to save the image
        figsize: Figure size in inches
        linewidth: Line width for boundary
        dpi: Resolution of output image
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    try:
        ax.contour(
            X_grid, Y_grid, Z,
            levels=[0],
            colors="black",
            linewidths=linewidth,
        )
    except ValueError:
        # No contour at level 0
        pass

    ax.set_xlim(X_grid.min(), X_grid.max())
    ax.set_ylim(Y_grid.min(), Y_grid.max())
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
