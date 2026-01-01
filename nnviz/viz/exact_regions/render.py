"""Polygon rendering for exact regions visualization."""

import hashlib
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np

from .enumerate import FinalRegion


def pattern_to_color(pattern: bytes) -> Tuple[float, float, float]:
    """Convert an activation pattern to a deterministic RGB color.

    Uses SHA256 hash of the pattern, takes first 3 bytes as RGB,
    then brightens the color.

    Args:
        pattern: Activation pattern bytes

    Returns:
        (R, G, B) tuple with values in [0.2, 1.0]
    """
    h = hashlib.sha256(pattern).digest()

    # Take first 3 bytes as RGB in [0, 1]
    r = h[0] / 255.0
    g = h[1] / 255.0
    b = h[2] / 255.0

    # Brighten: rgb = 0.2 + 0.8 * rgb
    r = 0.2 + 0.8 * r
    g = 0.2 + 0.8 * g
    b = 0.2 + 0.8 * b

    return (r, g, b)


def render_regions(
    regions: List[FinalRegion],
    domain: Tuple[Tuple[float, float], Tuple[float, float]],
    output_path: str,
    figsize: Tuple[float, float] = (10, 10),
    edge_linewidth: float = 0.5,
    dpi: int = 150,
) -> None:
    """Render exact regions to an image file.

    Args:
        regions: List of FinalRegion objects
        domain: ((x_min, x_max), (y_min, y_max)) bounds
        output_path: Path to save the image
        figsize: Figure size in inches
        edge_linewidth: Line width for polygon edges
        dpi: Resolution of output image
    """
    # Ensure parent directories exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    patches = []
    colors = []

    for region in regions:
        poly = MplPolygon(region.vertices, closed=True)
        patches.append(poly)
        colors.append(pattern_to_color(region.pattern))

    collection = PatchCollection(
        patches,
        facecolors=colors,
        edgecolors="black",
        linewidths=edge_linewidth,
    )
    ax.add_collection(collection)

    # Set axis limits to domain
    (x_min, x_max), (y_min, y_max) = domain
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    # Remove axis labels for cleaner visualization
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
