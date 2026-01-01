"""Tests for nnviz.train module."""

from pathlib import Path

import numpy as np
import pytest
import torch

from nnviz.config import NNVizConfig
from nnviz.model import MLP
from nnviz.train import (
    create_dataloader,
    train_epoch,
    train_model,
    save_checkpoint,
    load_checkpoint,
)
from nnviz.utils import set_seed


class TestCreateDataloader:
    """Test dataloader creation."""

    def test_dataloader_sizes(self):
        X = np.random.randn(100, 2).astype(np.float32)
        Y = np.random.randint(0, 2, 100).astype(np.float32)

        loader = create_dataloader(X, Y, batch_size=32, shuffle=False)

        # Should have 4 batches (100 / 32 = 3.125 -> 4)
        batches = list(loader)
        assert len(batches) == 4
        assert batches[0][0].shape == (32, 2)
        assert batches[0][1].shape == (32, 1)
        assert batches[-1][0].shape == (4, 2)  # Last batch has 4 samples

    def test_dataloader_tensors(self):
        X = np.random.randn(10, 2).astype(np.float32)
        Y = np.random.randint(0, 2, 10).astype(np.float32)

        loader = create_dataloader(X, Y, batch_size=10, shuffle=False)
        X_batch, Y_batch = next(iter(loader))

        assert X_batch.dtype == torch.float32
        assert Y_batch.dtype == torch.float32
        assert Y_batch.shape == (10, 1)  # Should be unsqueezed


class TestTrainEpoch:
    """Test single epoch training."""

    def test_returns_loss_and_accuracy(self):
        set_seed(42)
        model = MLP(width=8, depth=1)
        device = torch.device("cpu")

        X = np.random.randn(50, 2).astype(np.float32)
        Y = np.random.randint(0, 2, 50).astype(np.float32)
        loader = create_dataloader(X, Y, batch_size=16, shuffle=False)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        loss, accuracy = train_epoch(model, loader, optimizer, criterion, device)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0
        assert 0 <= accuracy <= 1

    def test_model_updates(self):
        set_seed(42)
        model = MLP(width=8, depth=1)
        device = torch.device("cpu")

        X = np.random.randn(50, 2).astype(np.float32)
        Y = np.random.randint(0, 2, 50).astype(np.float32)
        loader = create_dataloader(X, Y, batch_size=16, shuffle=False)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        # Save initial weights
        initial_weight = model.hidden[0].weight.clone()

        train_epoch(model, loader, optimizer, criterion, device)

        # Weights should have changed
        assert not torch.allclose(model.hidden[0].weight, initial_weight)


class TestTrainModel:
    """Test full training loop."""

    def test_loss_decreases(self, simple_2x2_image):
        """Test that loss decreases over training."""
        set_seed(42)
        model = MLP(width=16, depth=2)
        device = torch.device("cpu")

        # Create a simple separable dataset
        # Top half is class 0, bottom half is class 1
        X = np.array([
            [0.25, 0.25], [0.75, 0.25],  # Top (class 0)
            [0.25, 0.75], [0.75, 0.75],  # Bottom (class 1)
        ] * 100, dtype=np.float32)
        Y = np.array([0, 0, 1, 1] * 100, dtype=np.float32)

        config = NNVizConfig(
            image_path=simple_2x2_image,
            width=16,
            depth=2,
            epochs=50,
            lr=0.01,
            batch_size=64,
            seed=42,
        )

        logs = train_model(model, X, Y, config, device)

        assert len(logs) == 50
        # Loss should decrease
        first_loss = logs[0]["loss"]
        last_loss = logs[-1]["loss"]
        assert last_loss < first_loss

    def test_log_format(self, simple_2x2_image):
        set_seed(42)
        model = MLP(width=4, depth=1)
        device = torch.device("cpu")

        X = np.random.randn(20, 2).astype(np.float32)
        Y = np.random.randint(0, 2, 20).astype(np.float32)

        config = NNVizConfig(
            image_path=simple_2x2_image,
            width=4,
            depth=1,
            epochs=5,
            batch_size=10,
        )

        logs = train_model(model, X, Y, config, device)

        assert len(logs) == 5
        for i, log in enumerate(logs):
            assert "epoch" in log
            assert "loss" in log
            assert "accuracy" in log
            assert log["epoch"] == i

    def test_reproducibility(self, simple_2x2_image):
        """Training with same seed should produce same results."""
        device = torch.device("cpu")

        X = np.random.randn(50, 2).astype(np.float32)
        Y = np.random.randint(0, 2, 50).astype(np.float32)

        config = NNVizConfig(
            image_path=simple_2x2_image,
            width=8,
            depth=2,
            epochs=10,
            batch_size=16,
            seed=42,
        )

        # First training run
        set_seed(42)
        model1 = MLP(width=8, depth=2)
        logs1 = train_model(model1, X, Y, config, device)

        # Second training run
        set_seed(42)
        model2 = MLP(width=8, depth=2)
        logs2 = train_model(model2, X, Y, config, device)

        # Logs should be identical
        for l1, l2 in zip(logs1, logs2):
            assert l1["loss"] == l2["loss"]
            assert l1["accuracy"] == l2["accuracy"]


class TestCheckpointing:
    """Test model checkpoint save/load."""

    def test_save_and_load(self, temp_dir):
        set_seed(42)
        model = MLP(width=16, depth=3)

        # Do some training to get non-initial weights
        X = np.random.randn(20, 2).astype(np.float32)
        Y = np.random.randint(0, 2, 20).astype(np.float32)
        loader = create_dataloader(X, Y, batch_size=10, shuffle=False)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

        # Save checkpoint
        path = Path(temp_dir) / "model.pt"
        save_checkpoint(model, str(path))

        # Load checkpoint
        loaded_model = load_checkpoint(str(path))

        # Check architecture matches
        assert loaded_model.width == model.width
        assert loaded_model.depth == model.depth

        # Check weights match
        x = torch.randn(5, 2)
        original_out = model(x)
        loaded_out = loaded_model(x)
        torch.testing.assert_close(original_out, loaded_out)

    def test_load_to_device(self, temp_dir):
        model = MLP(width=8, depth=2)
        path = Path(temp_dir) / "model.pt"
        save_checkpoint(model, str(path))

        loaded_model = load_checkpoint(str(path), device=torch.device("cpu"))

        # All parameters should be on CPU
        for param in loaded_model.parameters():
            assert param.device == torch.device("cpu")
