"""Tests for nnviz.config module."""

import json
import tempfile
from pathlib import Path

import pytest

from nnviz.config import NNVizConfig


class TestNNVizConfigDefaults:
    """Test that defaults match the spec."""

    def test_training_defaults(self, simple_2x2_image):
        cfg = NNVizConfig(image_path=simple_2x2_image, width=4, depth=2)
        assert cfg.epochs == 300
        assert cfg.lr == 1e-3
        assert cfg.batch_size == 8192
        assert cfg.weight_decay == 1e-4
        assert cfg.seed == 0
        assert cfg.device == "auto"
        assert cfg.downsample == 1

    def test_labeling_defaults(self, simple_2x2_image):
        cfg = NNVizConfig(image_path=simple_2x2_image, width=4, depth=2)
        assert cfg.threshold == 0.5
        assert cfg.invert is True

    def test_coord_defaults(self, simple_2x2_image):
        cfg = NNVizConfig(image_path=simple_2x2_image, width=4, depth=2)
        assert cfg.normalize_coords == "unit"

    def test_viz_defaults(self, simple_2x2_image):
        cfg = NNVizConfig(image_path=simple_2x2_image, width=4, depth=2)
        assert cfg.viz_res == 512

    def test_exact_region_defaults(self, simple_2x2_image):
        cfg = NNVizConfig(image_path=simple_2x2_image, width=4, depth=2)
        assert cfg.exact_eps == 1e-9
        assert cfg.exact_area_eps == 1e-12
        assert cfg.exact_max_regions == 20000
        assert cfg.exact_order == "neuronwise"


class TestNNVizConfigValidation:
    """Test validation of configuration values."""

    def test_width_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="width must be >= 1"):
            NNVizConfig(image_path=simple_2x2_image, width=0, depth=2)

    def test_depth_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="depth must be >= 1"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=0)

    def test_epochs_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, epochs=0)

    def test_lr_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="lr must be > 0"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, lr=0)

    def test_batch_size_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, batch_size=0)

    def test_weight_decay_must_be_nonnegative(self, simple_2x2_image):
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, weight_decay=-1)

    def test_downsample_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="downsample must be >= 1"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, downsample=0)

    def test_threshold_must_be_in_unit_interval(self, simple_2x2_image):
        with pytest.raises(ValueError, match="threshold must be in"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, threshold=1.5)
        with pytest.raises(ValueError, match="threshold must be in"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, threshold=-0.1)

    def test_normalize_coords_must_be_valid(self, simple_2x2_image):
        with pytest.raises(ValueError, match="normalize_coords must be"):
            NNVizConfig(
                image_path=simple_2x2_image, width=4, depth=2, normalize_coords="invalid"
            )

    def test_viz_res_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="viz_res must be >= 1"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, viz_res=0)

    def test_exact_eps_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="exact_eps must be > 0"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, exact_eps=0)

    def test_exact_area_eps_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="exact_area_eps must be > 0"):
            NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, exact_area_eps=0)

    def test_exact_max_regions_must_be_positive(self, simple_2x2_image):
        with pytest.raises(ValueError, match="exact_max_regions must be >= 1"):
            NNVizConfig(
                image_path=simple_2x2_image, width=4, depth=2, exact_max_regions=0
            )


class TestNNVizConfigSerialization:
    """Test JSON serialization and deserialization."""

    def test_to_dict(self, simple_2x2_image):
        cfg = NNVizConfig(image_path=simple_2x2_image, width=4, depth=2)
        d = cfg.to_dict()
        assert d["image_path"] == simple_2x2_image
        assert d["width"] == 4
        assert d["depth"] == 2
        assert d["epochs"] == 300

    def test_to_json_and_back(self, simple_2x2_image):
        cfg = NNVizConfig(image_path=simple_2x2_image, width=4, depth=2, epochs=100)
        json_str = cfg.to_json()
        cfg2 = NNVizConfig.from_json(json_str)
        assert cfg2.image_path == cfg.image_path
        assert cfg2.width == cfg.width
        assert cfg2.depth == cfg.depth
        assert cfg2.epochs == cfg.epochs

    def test_save_and_load_json_file(self, simple_2x2_image, temp_dir):
        cfg = NNVizConfig(
            image_path=simple_2x2_image, width=8, depth=3, lr=0.01, seed=42
        )
        json_path = Path(temp_dir) / "config.json"
        cfg.save_json(str(json_path))

        cfg2 = NNVizConfig.from_json_file(str(json_path))
        assert cfg2.width == 8
        assert cfg2.depth == 3
        assert cfg2.lr == 0.01
        assert cfg2.seed == 42


class TestNNVizConfigSaveDir:
    """Test save directory generation."""

    def test_default_save_dir_format(self, simple_2x2_image):
        cfg = NNVizConfig(image_path=simple_2x2_image, width=4, depth=2)
        # Should be runs/<timestamp>_<image_stem>
        assert cfg.save_dir.startswith("runs/")
        assert "2x2" in cfg.save_dir

    def test_custom_save_dir(self, simple_2x2_image):
        cfg = NNVizConfig(
            image_path=simple_2x2_image, width=4, depth=2, save_dir="custom/output"
        )
        assert cfg.save_dir == "custom/output"
