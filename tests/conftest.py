"""Pytest fixtures for nnviz tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_2x2_image(temp_dir):
    """Create a simple 2x2 grayscale image.

    Layout (grayscale values 0-255):
        [0,   255]   (black, white)
        [128, 64]    (mid-gray, dark-gray)
    """
    arr = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    path = Path(temp_dir) / "2x2.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


@pytest.fixture
def checkerboard_4x4_image(temp_dir):
    """Create a 4x4 checkerboard pattern.

    Alternating black (0) and white (255).
    """
    arr = np.zeros((4, 4), dtype=np.uint8)
    arr[0::2, 0::2] = 255  # white at even rows, even cols
    arr[1::2, 1::2] = 255  # white at odd rows, odd cols
    path = Path(temp_dir) / "checker4x4.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


@pytest.fixture
def gradient_8x8_image(temp_dir):
    """Create an 8x8 horizontal gradient from black to white."""
    arr = np.zeros((8, 8), dtype=np.uint8)
    for i in range(8):
        arr[:, i] = int(255 * i / 7)
    path = Path(temp_dir) / "gradient8x8.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


@pytest.fixture
def all_black_image(temp_dir):
    """Create a 4x4 all-black image."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    path = Path(temp_dir) / "black4x4.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


@pytest.fixture
def all_white_image(temp_dir):
    """Create a 4x4 all-white image."""
    arr = np.full((4, 4), 255, dtype=np.uint8)
    path = Path(temp_dir) / "white4x4.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


@pytest.fixture
def unit_square_vertices():
    """Return unit square vertices in CCW order."""
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)


@pytest.fixture
def unit_triangle_vertices():
    """Return right triangle vertices in CCW order (area = 0.5)."""
    return np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
