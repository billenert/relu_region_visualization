"""Tests for nnviz.data module."""

import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from nnviz.data import load_image, image_to_dataset, get_domain_bounds


class TestLoadImage:
    """Test image loading."""

    def test_load_grayscale(self, simple_2x2_image):
        img = load_image(simple_2x2_image)
        assert img.shape == (2, 2)
        assert img.dtype == np.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_load_values(self, simple_2x2_image):
        """Test that pixel values are correctly normalized.

        The 2x2 image has layout:
        [[0, 255], [128, 64]] (grayscale 0-255)
        Expected after normalization:
        [[0, 1], [128/255, 64/255]]
        """
        img = load_image(simple_2x2_image)
        np.testing.assert_almost_equal(img[0, 0], 0.0)
        np.testing.assert_almost_equal(img[0, 1], 1.0)
        np.testing.assert_almost_equal(img[1, 0], 128 / 255)
        np.testing.assert_almost_equal(img[1, 1], 64 / 255)

    def test_load_rgb_converts_to_grayscale(self, temp_dir):
        """Test that RGB images are converted to grayscale."""
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red channel
        path = Path(temp_dir) / "rgb.png"
        Image.fromarray(arr, mode="RGB").save(path)

        img = load_image(str(path))
        assert img.shape == (4, 4)  # Should be 2D grayscale


class TestImageToDataset:
    """Test dataset creation from images."""

    def test_output_shapes(self, simple_2x2_image):
        img = load_image(simple_2x2_image)
        X, Y = image_to_dataset(img)
        assert X.shape == (4, 2)  # 2x2 = 4 points, 2 coords each
        assert Y.shape == (4,)
        assert X.dtype == np.float32
        assert Y.dtype == np.float32

    def test_labels_are_binary(self, simple_2x2_image):
        img = load_image(simple_2x2_image)
        X, Y = image_to_dataset(img)
        assert set(np.unique(Y)).issubset({0.0, 1.0})

    def test_invert_true_makes_black_dark(self, all_black_image, all_white_image):
        """With invert=True, black (raw=0) becomes darkness=1, white (raw=1) becomes darkness=0."""
        black_img = load_image(all_black_image)
        white_img = load_image(all_white_image)

        # Black image: raw=0, darkness=1 > 0.5, so label=1
        _, Y_black = image_to_dataset(black_img, threshold=0.5, invert=True)
        assert np.all(Y_black == 1)

        # White image: raw=1, darkness=0 < 0.5, so label=0
        _, Y_white = image_to_dataset(white_img, threshold=0.5, invert=True)
        assert np.all(Y_white == 0)

    def test_invert_false(self, all_black_image, all_white_image):
        """With invert=False, raw values are used directly as darkness."""
        black_img = load_image(all_black_image)
        white_img = load_image(all_white_image)

        # Black image: raw=0, darkness=0 < 0.5, so label=0
        _, Y_black = image_to_dataset(black_img, threshold=0.5, invert=False)
        assert np.all(Y_black == 0)

        # White image: raw=1, darkness=1 > 0.5, so label=1
        _, Y_white = image_to_dataset(white_img, threshold=0.5, invert=False)
        assert np.all(Y_white == 1)

    def test_threshold_strict_greater(self, temp_dir):
        """Test that threshold comparison uses strict > (not >=)."""
        # Create image with exactly 0.5 grayscale value (128/255 ≈ 0.502)
        # Let's use 127 which is 127/255 ≈ 0.498
        arr = np.full((2, 2), 127, dtype=np.uint8)
        path = Path(temp_dir) / "half.png"
        Image.fromarray(arr, mode="L").save(path)

        img = load_image(str(path))
        # With invert=True: darkness = 1 - 127/255 = 128/255 ≈ 0.502
        _, Y = image_to_dataset(img, threshold=0.5, invert=True)
        # 0.502 > 0.5, so should be labeled 1
        assert np.all(Y == 1)

        # At exact threshold (would need darkness == threshold)
        # 1 - threshold = raw -> raw = 0.5 -> 127.5, which isn't exact
        # Let's test with threshold=128/255
        _, Y2 = image_to_dataset(img, threshold=128/255, invert=True)
        # darkness = 128/255, threshold = 128/255, so 128/255 > 128/255 is False
        assert np.all(Y2 == 0)

    def test_unit_coords_range(self, checkerboard_4x4_image):
        """Test that unit coordinates are in [0, 1]."""
        img = load_image(checkerboard_4x4_image)
        X, _ = image_to_dataset(img, normalize_coords="unit")
        assert X[:, 0].min() >= 0.0
        assert X[:, 0].max() <= 1.0
        assert X[:, 1].min() >= 0.0
        assert X[:, 1].max() <= 1.0

    def test_unit_coords_corners(self, simple_2x2_image):
        """Test that corners map to correct unit coordinates."""
        img = load_image(simple_2x2_image)
        X, _ = image_to_dataset(img, normalize_coords="unit")

        # For 2x2 image with y-flip (j=0 at top maps to y=1):
        # Pixel (i=0, j=0) -> (x=0, y=1)  top-left
        # Pixel (i=1, j=0) -> (x=1, y=1)  top-right
        # Pixel (i=0, j=1) -> (x=0, y=0)  bottom-left
        # Pixel (i=1, j=1) -> (x=1, y=0)  bottom-right
        # Order is row-major: j=0 first, then j=1
        expected = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(X, expected)

    def test_pixel_coords(self, checkerboard_4x4_image):
        """Test that pixel coordinates match pixel indices."""
        img = load_image(checkerboard_4x4_image)
        X, _ = image_to_dataset(img, normalize_coords="pixel")

        # For 4x4 image, x should range [0, 3], y should range [0, 3]
        assert X[:, 0].min() == 0.0
        assert X[:, 0].max() == 3.0
        assert X[:, 1].min() == 0.0
        assert X[:, 1].max() == 3.0

    def test_downsample(self, checkerboard_4x4_image):
        """Test downsampling reduces number of points."""
        img = load_image(checkerboard_4x4_image)

        X_full, _ = image_to_dataset(img, downsample=1)
        X_down2, _ = image_to_dataset(img, downsample=2)

        assert X_full.shape[0] == 16  # 4x4
        assert X_down2.shape[0] == 4  # 2x2 (indices 0,2 in each dim)

    def test_downsample_coords(self, checkerboard_4x4_image):
        """Test that downsampled coordinates are correct."""
        img = load_image(checkerboard_4x4_image)
        X, _ = image_to_dataset(img, downsample=2, normalize_coords="pixel")

        # With y-flip: j=0 maps to y=3, j=2 maps to y=1
        # Should have pixels at (i,j): (0,0), (2,0), (0,2), (2,2)
        # -> (x,y): (0,3), (2,3), (0,1), (2,1)
        expected_x = np.array([0, 2, 0, 2], dtype=np.float32)
        expected_y = np.array([3, 3, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(X[:, 0], expected_x)
        np.testing.assert_array_equal(X[:, 1], expected_y)

    def test_single_pixel_image(self, temp_dir):
        """Test handling of 1x1 image."""
        arr = np.array([[128]], dtype=np.uint8)
        path = Path(temp_dir) / "1x1.png"
        Image.fromarray(arr, mode="L").save(path)

        img = load_image(str(path))
        X, Y = image_to_dataset(img, normalize_coords="unit")

        assert X.shape == (1, 2)
        # Single pixel should be at (0, 0) in unit coords
        np.testing.assert_array_equal(X[0], [0, 0])


class TestGetDomainBounds:
    """Test domain bounds calculation."""

    def test_unit_bounds(self):
        bounds = get_domain_bounds((100, 200), "unit")
        assert bounds == ((0.0, 1.0), (0.0, 1.0))

    def test_pixel_bounds(self):
        bounds = get_domain_bounds((100, 200), "pixel")
        # H=100, W=200 -> x in [0, 199], y in [0, 99]
        assert bounds == ((0.0, 199.0), (0.0, 99.0))

    def test_invalid_normalize_coords(self):
        with pytest.raises(ValueError, match="normalize_coords must be"):
            get_domain_bounds((100, 100), "invalid")
