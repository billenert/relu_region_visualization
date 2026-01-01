"""Integration tests for nnviz."""

import json
from pathlib import Path

import pytest

from nnviz.config import NNVizConfig
from nnviz.main import run_pipeline
from nnviz.utils import set_seed


class TestIntegrationSmallNetwork:
    """Integration tests with a small network (should produce exact regions)."""

    def test_full_pipeline(self, simple_2x2_image, temp_dir):
        """Test the complete pipeline end-to-end."""
        set_seed(42)

        config = NNVizConfig(
            image_path=simple_2x2_image,
            width=4,
            depth=1,
            epochs=10,
            batch_size=16,
            seed=42,
            save_dir=temp_dir,
        )

        run_pipeline(config)

        # Check output files exist
        assert (Path(temp_dir) / "config.json").exists()
        assert (Path(temp_dir) / "train_log.csv").exists()
        assert (Path(temp_dir) / "model.pt").exists()
        assert (Path(temp_dir) / "viz_3d.png").exists()
        assert (Path(temp_dir) / "viz_boundary.png").exists()
        # For small network, exact regions should succeed
        assert (Path(temp_dir) / "viz_exact_regions.png").exists()
        assert not (Path(temp_dir) / "exact_failure.json").exists()

    def test_config_saved_correctly(self, simple_2x2_image, temp_dir):
        """Test that config is saved and can be loaded."""
        config = NNVizConfig(
            image_path=simple_2x2_image,
            width=4,
            depth=1,
            epochs=5,
            seed=42,
            save_dir=temp_dir,
        )

        run_pipeline(config)

        # Load and verify config
        config_path = Path(temp_dir) / "config.json"
        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config["width"] == 4
        assert saved_config["depth"] == 1
        assert saved_config["epochs"] == 5
        assert saved_config["seed"] == 42

    def test_train_log_format(self, simple_2x2_image, temp_dir):
        """Test that training log has correct format."""
        config = NNVizConfig(
            image_path=simple_2x2_image,
            width=4,
            depth=1,
            epochs=5,
            seed=42,
            save_dir=temp_dir,
        )

        run_pipeline(config)

        # Read train log
        log_path = Path(temp_dir) / "train_log.csv"
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 6  # header + 5 epochs
        assert "epoch" in lines[0]
        assert "loss" in lines[0]
        assert "accuracy" in lines[0]


class TestIntegrationLargeNetwork:
    """Integration tests with larger network (may cause region explosion)."""

    def test_handles_region_explosion(self, simple_2x2_image, temp_dir):
        """Test that region explosion is handled gracefully."""
        config = NNVizConfig(
            image_path=simple_2x2_image,
            width=16,
            depth=4,
            epochs=5,
            seed=42,
            exact_max_regions=5,  # Set very low to trigger explosion
            save_dir=temp_dir,
        )

        run_pipeline(config)

        # Check output files
        assert (Path(temp_dir) / "config.json").exists()
        assert (Path(temp_dir) / "train_log.csv").exists()
        assert (Path(temp_dir) / "model.pt").exists()
        assert (Path(temp_dir) / "viz_3d.png").exists()
        assert (Path(temp_dir) / "viz_boundary.png").exists()  # Should use contour fallback

        # Exact regions should have failed
        assert (Path(temp_dir) / "exact_failure.json").exists()
        assert not (Path(temp_dir) / "viz_exact_regions.png").exists()

        # Check failure info
        with open(Path(temp_dir) / "exact_failure.json") as f:
            failure_info = json.load(f)
        assert "layer" in failure_info
        assert "neuron" in failure_info
        assert "region_count" in failure_info


class TestIntegrationDifferentInputs:
    """Test with different input configurations."""

    def test_checkerboard(self, checkerboard_4x4_image, temp_dir):
        """Test with checkerboard pattern."""
        config = NNVizConfig(
            image_path=checkerboard_4x4_image,
            width=8,
            depth=2,
            epochs=10,
            seed=42,
            save_dir=temp_dir,
        )

        run_pipeline(config)

        assert (Path(temp_dir) / "viz_3d.png").exists()
        assert (Path(temp_dir) / "viz_boundary.png").exists()

    def test_gradient(self, gradient_8x8_image, temp_dir):
        """Test with gradient image."""
        config = NNVizConfig(
            image_path=gradient_8x8_image,
            width=4,
            depth=1,
            epochs=10,
            seed=42,
            save_dir=temp_dir,
        )

        run_pipeline(config)

        assert (Path(temp_dir) / "viz_3d.png").exists()
        assert (Path(temp_dir) / "viz_boundary.png").exists()

    def test_pixel_coords(self, simple_2x2_image, temp_dir):
        """Test with pixel coordinate normalization."""
        config = NNVizConfig(
            image_path=simple_2x2_image,
            width=4,
            depth=1,
            epochs=5,
            normalize_coords="pixel",
            seed=42,
            save_dir=temp_dir,
        )

        run_pipeline(config)

        assert (Path(temp_dir) / "viz_3d.png").exists()

    def test_no_invert(self, simple_2x2_image, temp_dir):
        """Test with invert=False."""
        config = NNVizConfig(
            image_path=simple_2x2_image,
            width=4,
            depth=1,
            epochs=5,
            invert=False,
            seed=42,
            save_dir=temp_dir,
        )

        run_pipeline(config)

        assert (Path(temp_dir) / "viz_3d.png").exists()

    def test_downsample(self, checkerboard_4x4_image, temp_dir):
        """Test with downsampling."""
        config = NNVizConfig(
            image_path=checkerboard_4x4_image,
            width=4,
            depth=1,
            epochs=5,
            downsample=2,
            seed=42,
            save_dir=temp_dir,
        )

        run_pipeline(config)

        assert (Path(temp_dir) / "viz_3d.png").exists()
