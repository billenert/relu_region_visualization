"""Tests for nnviz.model module."""

import numpy as np
import pytest
import torch

from nnviz.model import MLP


class TestMLPArchitecture:
    """Test MLP architecture."""

    def test_basic_construction(self):
        model = MLP(width=8, depth=2)
        assert model.width == 8
        assert model.depth == 2

    def test_invalid_width(self):
        with pytest.raises(ValueError, match="width must be >= 1"):
            MLP(width=0, depth=2)

    def test_invalid_depth(self):
        with pytest.raises(ValueError, match="depth must be >= 1"):
            MLP(width=8, depth=0)

    def test_leaky_slope(self):
        model = MLP(width=4, depth=1)
        assert model.LEAKY_SLOPE == 0.02


class TestMLPForward:
    """Test MLP forward pass."""

    def test_output_shape_single(self):
        model = MLP(width=8, depth=2)
        x = torch.randn(1, 2)
        out = model(x)
        assert out.shape == (1, 1)

    def test_output_shape_batch(self):
        model = MLP(width=8, depth=2)
        x = torch.randn(32, 2)
        out = model(x)
        assert out.shape == (32, 1)

    def test_output_is_logits(self):
        """Output should be unbounded logits, not probabilities."""
        model = MLP(width=16, depth=3)
        torch.manual_seed(42)

        # Initialize with larger weights to get extreme values
        for p in model.parameters():
            p.data.uniform_(-2, 2)

        x = torch.randn(100, 2)
        out = model(x)

        # Logits should have some values outside [0, 1]
        assert out.min() < 0 or out.max() > 1

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = MLP(width=8, depth=2)
        x = torch.randn(10, 2, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


class TestMLPParameterCount:
    """Test parameter counting."""

    def test_single_hidden_layer(self):
        """Test: 2 -> W -> 1
        Params: (2*W + W) + (W*1 + 1) = 3W + W + 1 = 4W + 1
        """
        model = MLP(width=8, depth=1)
        # First layer: 2*8 weights + 8 biases = 24
        # Output layer: 8*1 weights + 1 bias = 9
        # Total = 33
        expected = 2 * 8 + 8 + 8 * 1 + 1
        assert model.count_parameters() == expected

    def test_two_hidden_layers(self):
        """Test: 2 -> W -> W -> 1"""
        model = MLP(width=8, depth=2)
        # First layer: 2*8 + 8 = 24
        # Second layer: 8*8 + 8 = 72
        # Output layer: 8*1 + 1 = 9
        # Total = 105
        expected = (2 * 8 + 8) + (8 * 8 + 8) + (8 * 1 + 1)
        assert model.count_parameters() == expected

    def test_three_hidden_layers(self):
        """Test: 2 -> W -> W -> W -> 1"""
        model = MLP(width=4, depth=3)
        # First: 2*4 + 4 = 12
        # Second: 4*4 + 4 = 20
        # Third: 4*4 + 4 = 20
        # Output: 4*1 + 1 = 5
        # Total = 57
        expected = (2 * 4 + 4) + 2 * (4 * 4 + 4) + (4 * 1 + 1)
        assert model.count_parameters() == expected


class TestMLPWeightExtraction:
    """Test weight extraction for exact region enumeration."""

    def test_layer_count(self):
        """Number of weight tuples should equal depth + 1 (hidden + output)."""
        model = MLP(width=8, depth=3)
        weights = model.get_layer_weights()
        assert len(weights) == 4  # 3 hidden + 1 output

    def test_weight_shapes_depth1(self):
        """Test weight shapes for single hidden layer."""
        model = MLP(width=8, depth=1)
        weights = model.get_layer_weights()

        # First hidden: (8, 2) weights, (8,) bias
        assert weights[0][0].shape == (8, 2)
        assert weights[0][1].shape == (8,)

        # Output: (1, 8) weights, (1,) bias
        assert weights[1][0].shape == (1, 8)
        assert weights[1][1].shape == (1,)

    def test_weight_shapes_depth3(self):
        """Test weight shapes for three hidden layers."""
        model = MLP(width=4, depth=3)
        weights = model.get_layer_weights()

        # First hidden: (4, 2)
        assert weights[0][0].shape == (4, 2)
        assert weights[0][1].shape == (4,)

        # Second hidden: (4, 4)
        assert weights[1][0].shape == (4, 4)
        assert weights[1][1].shape == (4,)

        # Third hidden: (4, 4)
        assert weights[2][0].shape == (4, 4)
        assert weights[2][1].shape == (4,)

        # Output: (1, 4)
        assert weights[3][0].shape == (1, 4)
        assert weights[3][1].shape == (1,)

    def test_weights_are_float64(self):
        """Weights should be float64 for numerical precision."""
        model = MLP(width=8, depth=2)
        weights = model.get_layer_weights()

        for W, b in weights:
            assert W.dtype == np.float64
            assert b.dtype == np.float64

    def test_weights_match_parameters(self):
        """Extracted weights should match model parameters."""
        model = MLP(width=8, depth=2)
        weights = model.get_layer_weights()

        # First hidden layer
        expected_W0 = model.hidden[0].weight.detach().cpu().numpy()
        expected_b0 = model.hidden[0].bias.detach().cpu().numpy()
        np.testing.assert_array_almost_equal(weights[0][0], expected_W0)
        np.testing.assert_array_almost_equal(weights[0][1], expected_b0)

        # Second hidden layer
        expected_W1 = model.hidden[2].weight.detach().cpu().numpy()
        expected_b1 = model.hidden[2].bias.detach().cpu().numpy()
        np.testing.assert_array_almost_equal(weights[1][0], expected_W1)
        np.testing.assert_array_almost_equal(weights[1][1], expected_b1)

        # Output layer
        expected_Wout = model.output.weight.detach().cpu().numpy()
        expected_bout = model.output.bias.detach().cpu().numpy()
        np.testing.assert_array_almost_equal(weights[2][0], expected_Wout)
        np.testing.assert_array_almost_equal(weights[2][1], expected_bout)


class TestMLPDeterminism:
    """Test that model is deterministic with same seed."""

    def test_same_seed_same_output(self):
        torch.manual_seed(42)
        model1 = MLP(width=8, depth=2)

        torch.manual_seed(42)
        model2 = MLP(width=8, depth=2)

        x = torch.randn(10, 2)

        out1 = model1(x)
        out2 = model2(x)

        torch.testing.assert_close(out1, out2)

    def test_different_seed_different_output(self):
        torch.manual_seed(42)
        model1 = MLP(width=8, depth=2)

        torch.manual_seed(43)
        model2 = MLP(width=8, depth=2)

        x = torch.randn(10, 2)

        out1 = model1(x)
        out2 = model2(x)

        assert not torch.allclose(out1, out2)
