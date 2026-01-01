"""Tests for nnviz.viz.grid module."""

import numpy as np
import pytest
import torch

from nnviz.model import MLP
from nnviz.utils import set_seed
from nnviz.viz.grid import create_evaluation_grid, evaluate_model_on_grid


class TestCreateEvaluationGrid:
    """Test evaluation grid creation."""

    def test_grid_shapes(self):
        domain = ((0.0, 1.0), (0.0, 1.0))
        X, Y, points = create_evaluation_grid(domain, resolution=10)

        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert points.shape == (100, 2)

    def test_grid_bounds(self):
        domain = ((0.0, 1.0), (0.0, 1.0))
        X, Y, points = create_evaluation_grid(domain, resolution=10)

        assert X.min() == 0.0
        assert X.max() == 1.0
        assert Y.min() == 0.0
        assert Y.max() == 1.0

    def test_non_unit_domain(self):
        domain = ((-1.0, 2.0), (0.5, 1.5))
        X, Y, points = create_evaluation_grid(domain, resolution=5)

        assert X.min() == -1.0
        assert X.max() == 2.0
        assert Y.min() == 0.5
        assert Y.max() == 1.5

    def test_points_dtype(self):
        domain = ((0.0, 1.0), (0.0, 1.0))
        _, _, points = create_evaluation_grid(domain, resolution=10)

        assert points.dtype == np.float32

    def test_points_cover_domain(self):
        domain = ((0.0, 1.0), (0.0, 1.0))
        _, _, points = create_evaluation_grid(domain, resolution=10)

        # Check corners are included
        def has_point(target):
            return np.any(np.all(np.abs(points - target) < 0.01, axis=1))

        assert has_point([0.0, 0.0])
        assert has_point([1.0, 0.0])
        assert has_point([0.0, 1.0])
        assert has_point([1.0, 1.0])


class TestEvaluateModelOnGrid:
    """Test model evaluation on grid."""

    def test_output_shape(self):
        set_seed(42)
        model = MLP(width=8, depth=2)
        device = torch.device("cpu")

        domain = ((0.0, 1.0), (0.0, 1.0))
        _, _, points = create_evaluation_grid(domain, resolution=10)

        logits = evaluate_model_on_grid(model, points, device)

        assert logits.shape == (100,)
        assert logits.dtype == np.float32

    def test_matches_direct_evaluation(self):
        set_seed(42)
        model = MLP(width=8, depth=2)
        device = torch.device("cpu")

        domain = ((0.0, 1.0), (0.0, 1.0))
        _, _, points = create_evaluation_grid(domain, resolution=5)

        # Batch evaluation
        logits = evaluate_model_on_grid(model, points, device)

        # Direct evaluation
        model.eval()
        with torch.no_grad():
            expected = model(torch.from_numpy(points)).squeeze(-1).numpy()

        np.testing.assert_array_almost_equal(logits, expected)

    def test_batching(self):
        set_seed(42)
        model = MLP(width=8, depth=2)
        device = torch.device("cpu")

        domain = ((0.0, 1.0), (0.0, 1.0))
        _, _, points = create_evaluation_grid(domain, resolution=20)  # 400 points

        # Evaluate with small batch size
        logits_small_batch = evaluate_model_on_grid(
            model, points, device, batch_size=32
        )

        # Evaluate with large batch size
        logits_large_batch = evaluate_model_on_grid(
            model, points, device, batch_size=1000
        )

        np.testing.assert_array_almost_equal(logits_small_batch, logits_large_batch)

    def test_deterministic(self):
        set_seed(42)
        model = MLP(width=8, depth=2)
        device = torch.device("cpu")

        domain = ((0.0, 1.0), (0.0, 1.0))
        _, _, points = create_evaluation_grid(domain, resolution=10)

        logits1 = evaluate_model_on_grid(model, points, device)
        logits2 = evaluate_model_on_grid(model, points, device)

        np.testing.assert_array_equal(logits1, logits2)
