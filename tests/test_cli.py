"""Tests for nnviz.cli module."""

import pytest

from nnviz.cli import parse_args
from nnviz.config import NNVizConfig


class TestParseArgs:
    """Test argument parsing."""

    def test_required_args(self, simple_2x2_image):
        args = [simple_2x2_image, "--width", "8", "--depth", "2"]
        config = parse_args(args)

        assert config.image_path == simple_2x2_image
        assert config.width == 8
        assert config.depth == 2

    def test_missing_width(self, simple_2x2_image):
        args = [simple_2x2_image, "--depth", "2"]
        with pytest.raises(SystemExit):
            parse_args(args)

    def test_missing_depth(self, simple_2x2_image):
        args = [simple_2x2_image, "--width", "8"]
        with pytest.raises(SystemExit):
            parse_args(args)

    def test_defaults(self, simple_2x2_image):
        args = [simple_2x2_image, "--width", "8", "--depth", "2"]
        config = parse_args(args)

        assert config.epochs == 300
        assert config.lr == 1e-3
        assert config.batch_size == 8192
        assert config.weight_decay == 1e-4
        assert config.seed == 0
        assert config.device == "auto"
        assert config.downsample == 1
        assert config.threshold == 0.5
        assert config.invert is True
        assert config.normalize_coords == "unit"
        assert config.viz_res == 512
        assert config.exact_eps == 1e-9
        assert config.exact_area_eps == 1e-12
        assert config.exact_max_regions == 20000
        assert config.exact_order == "neuronwise"

    def test_custom_training_params(self, simple_2x2_image):
        args = [
            simple_2x2_image,
            "--width", "16",
            "--depth", "3",
            "--epochs", "100",
            "--lr", "0.01",
            "--batch-size", "256",
            "--weight-decay", "0.001",
            "--seed", "42",
        ]
        config = parse_args(args)

        assert config.width == 16
        assert config.depth == 3
        assert config.epochs == 100
        assert config.lr == 0.01
        assert config.batch_size == 256
        assert config.weight_decay == 0.001
        assert config.seed == 42

    def test_invert_flag(self, simple_2x2_image):
        args = [simple_2x2_image, "--width", "8", "--depth", "2", "--invert", "false"]
        config = parse_args(args)
        assert config.invert is False

        args = [simple_2x2_image, "--width", "8", "--depth", "2", "--invert", "true"]
        config = parse_args(args)
        assert config.invert is True

    def test_normalize_coords(self, simple_2x2_image):
        args = [
            simple_2x2_image, "--width", "8", "--depth", "2",
            "--normalize-coords", "pixel"
        ]
        config = parse_args(args)
        assert config.normalize_coords == "pixel"

    def test_save_dir(self, simple_2x2_image, temp_dir):
        args = [
            simple_2x2_image, "--width", "8", "--depth", "2",
            "--save-dir", temp_dir
        ]
        config = parse_args(args)
        assert config.save_dir == temp_dir

    def test_exact_region_params(self, simple_2x2_image):
        args = [
            simple_2x2_image, "--width", "8", "--depth", "2",
            "--exact-eps", "1e-6",
            "--exact-area-eps", "1e-9",
            "--exact-max-regions", "1000",
        ]
        config = parse_args(args)

        assert config.exact_eps == 1e-6
        assert config.exact_area_eps == 1e-9
        assert config.exact_max_regions == 1000

    def test_returns_nnviz_config(self, simple_2x2_image):
        args = [simple_2x2_image, "--width", "8", "--depth", "2"]
        config = parse_args(args)

        assert isinstance(config, NNVizConfig)
