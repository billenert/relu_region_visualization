"""3D surface plot of network logits."""

from pathlib import Path
from typing import Tuple, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_3d_surface(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z: np.ndarray,
    output_path: str,
    show_contour: bool = True,
    figsize: Tuple[float, float] = (12, 10),
    dpi: int = 150,
    cmap: str = "viridis",
    elev: float = 30,
    azim: float = -60,
) -> None:
    """Create a 3D surface plot of the network logits.

    Args:
        X_grid: (R, R) meshgrid of x coordinates
        Y_grid: (R, R) meshgrid of y coordinates
        Z: (R, R) array of logit values
        output_path: Path to save the image
        show_contour: Whether to show z=0 contour on the bottom plane
        figsize: Figure size in inches
        dpi: Resolution of output image
        cmap: Colormap name
        elev: Elevation angle for viewing
        azim: Azimuth angle for viewing
    """
    # Ensure parent directories exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surf = ax.plot_surface(
        X_grid, Y_grid, Z,
        cmap=cmap,
        alpha=0.8,
        linewidth=0,
        antialiased=True,
    )

    # Add contour at z=0 on the bottom plane
    if show_contour:
        z_min = Z.min()
        try:
            ax.contour(
                X_grid, Y_grid, Z,
                levels=[0],
                colors="red",
                linewidths=2,
                offset=z_min,
            )
        except ValueError:
            # No contour at level 0 (all positive or all negative)
            pass

    # Set labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("logit z(x,y)")

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Logit value")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
