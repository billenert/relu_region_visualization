"""Evaluation grid creation and model evaluation for visualization."""

from typing import Tuple

import numpy as np
import torch

from ..model import MLP


def create_evaluation_grid(
    domain: Tuple[Tuple[float, float], Tuple[float, float]],
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a regular evaluation grid over the domain.

    Args:
        domain: ((x_min, x_max), (y_min, y_max)) bounds
        resolution: Number of points along each axis

    Returns:
        (X_grid, Y_grid, points) tuple where:
        - X_grid is (R, R) meshgrid of x coordinates
        - Y_grid is (R, R) meshgrid of y coordinates
        - points is (R*R, 2) array of (x, y) coordinates
    """
    (x_min, x_max), (y_min, y_max) = domain

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)

    X_grid, Y_grid = np.meshgrid(x, y, indexing="xy")

    # Flatten to (R*R, 2) points array
    points = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1).astype(np.float32)

    return X_grid, Y_grid, points


def evaluate_model_on_grid(
    model: MLP,
    points: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Evaluate model on a grid of points.

    Args:
        model: Trained MLP model
        points: (N, 2) array of evaluation points
        device: Device to run evaluation on
        batch_size: Batch size for evaluation

    Returns:
        (N,) array of logit values
    """
    model.eval()
    model.to(device)

    n_points = len(points)
    logits = np.zeros(n_points, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_points = points[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch_points).to(device)
            batch_logits = model(batch_tensor).squeeze(-1)
            logits[i : i + batch_size] = batch_logits.cpu().numpy()

    return logits
