"""Dataset creation from images for nnviz."""

from typing import Tuple

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image and convert to grayscale float array.

    Args:
        path: Path to the image file

    Returns:
        2D numpy array of shape (H, W) with values in [0, 1]
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def image_to_dataset(
    image: np.ndarray,
    threshold: float = 0.5,
    invert: bool = True,
    normalize_coords: str = "unit",
    downsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a grayscale image to a labeled dataset of 2D points.

    The image is treated as a grid where each pixel becomes a labeled point.
    Pixel position (i, j) maps to coordinate (x, y) where:
    - i is the horizontal index (column), j is the vertical index (row)
    - For unit normalization: x = i/(W-1), y = j/(H-1)

    Args:
        image: 2D numpy array of shape (H, W) with values in [0, 1]
        threshold: Darkness threshold for labeling (label=1 if darkness > threshold)
        invert: If True, darkness = 1 - raw_value (so white=0, black=1)
        normalize_coords: "unit" for [0,1]^2, "pixel" for pixel indices
        downsample: Only keep pixels where i % downsample == 0 and j % downsample == 0

    Returns:
        (X, Y) tuple where:
        - X is shape (N, 2) float32 array of coordinates
        - Y is shape (N,) float32 array of labels in {0, 1}
    """
    H, W = image.shape

    # Apply inversion to get "darkness"
    if invert:
        darkness = 1.0 - image
    else:
        darkness = image

    # Create label array: 1 if darkness > threshold, else 0
    # Use strict > as specified
    labels = (darkness > threshold).astype(np.float32)

    # Create coordinate arrays
    # i (column index) -> x, j (row index) -> y
    j_indices, i_indices = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Apply downsampling
    if downsample > 1:
        mask = (i_indices % downsample == 0) & (j_indices % downsample == 0)
        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
        labels = labels[mask]
    else:
        i_indices = i_indices.ravel()
        j_indices = j_indices.ravel()
        labels = labels.ravel()

    # Normalize coordinates
    # Note: Image row j=0 is at the TOP, but we want y=0 at BOTTOM
    # So we flip: y corresponds to (H - 1 - j)
    if normalize_coords == "unit":
        # x = i / (W - 1), y = (H - 1 - j) / (H - 1)
        # Handle edge case of single pixel dimension
        if W > 1:
            x = i_indices.astype(np.float32) / (W - 1)
        else:
            x = np.zeros_like(i_indices, dtype=np.float32)

        if H > 1:
            y = (H - 1 - j_indices).astype(np.float32) / (H - 1)
        else:
            y = np.zeros_like(j_indices, dtype=np.float32)
    elif normalize_coords == "pixel":
        x = i_indices.astype(np.float32)
        y = (H - 1 - j_indices).astype(np.float32)
    else:
        raise ValueError(f"normalize_coords must be 'unit' or 'pixel', got {normalize_coords}")

    X = np.stack([x, y], axis=1)
    Y = labels

    return X, Y


def get_domain_bounds(
    image_shape: Tuple[int, int],
    normalize_coords: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get the domain bounds for visualization based on image shape and normalization.

    Args:
        image_shape: (H, W) tuple of image dimensions
        normalize_coords: "unit" or "pixel"

    Returns:
        ((x_min, x_max), (y_min, y_max)) tuple of domain bounds
    """
    H, W = image_shape

    if normalize_coords == "unit":
        return ((0.0, 1.0), (0.0, 1.0))
    elif normalize_coords == "pixel":
        # x ranges over columns [0, W-1], y ranges over rows [0, H-1]
        return ((0.0, float(W - 1)), (0.0, float(H - 1)))
    else:
        raise ValueError(f"normalize_coords must be 'unit' or 'pixel', got {normalize_coords}")
