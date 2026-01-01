"""Tests for nnviz.utils module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from nnviz.utils import (
    set_seed,
    get_device,
    ensure_dir,
    save_json,
    load_json,
    save_csv,
    load_csv,
)


class TestSetSeed:
    """Test seed setting for reproducibility."""

    def test_numpy_reproducibility(self):
        set_seed(42)
        a1 = np.random.rand(10)
        set_seed(42)
        a2 = np.random.rand(10)
        np.testing.assert_array_equal(a1, a2)

    def test_torch_reproducibility(self):
        set_seed(42)
        t1 = torch.rand(10)
        set_seed(42)
        t2 = torch.rand(10)
        torch.testing.assert_close(t1, t2)

    def test_different_seeds_produce_different_results(self):
        set_seed(42)
        a1 = np.random.rand(10)
        set_seed(43)
        a2 = np.random.rand(10)
        assert not np.allclose(a1, a2)


class TestGetDevice:
    """Test device selection."""

    def test_cpu_device(self):
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_auto_returns_valid_device(self):
        device = get_device("auto")
        assert device.type in ("cpu", "cuda", "mps")

    def test_cuda_device_if_available(self):
        if torch.cuda.is_available():
            device = get_device("cuda")
            assert device.type == "cuda"
        else:
            # Should raise or return error if CUDA not available
            try:
                device = get_device("cuda")
                # If it doesn't raise, just check it returns a device
                assert device.type == "cuda"
            except RuntimeError:
                pass  # Expected if CUDA not available

    def test_cuda_with_index(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = get_device("cuda:0")
            assert device.type == "cuda"
            assert device.index == 0


class TestEnsureDir:
    """Test directory creation."""

    def test_creates_directory(self, temp_dir):
        new_dir = Path(temp_dir) / "new" / "nested" / "dir"
        ensure_dir(str(new_dir))
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_idempotent(self, temp_dir):
        new_dir = Path(temp_dir) / "test_dir"
        ensure_dir(str(new_dir))
        ensure_dir(str(new_dir))  # Should not raise
        assert new_dir.exists()


class TestJsonIO:
    """Test JSON save/load functions."""

    def test_save_and_load_json(self, temp_dir):
        data = {"key": "value", "number": 42, "nested": {"a": 1, "b": 2}}
        path = Path(temp_dir) / "test.json"
        save_json(data, str(path))
        loaded = load_json(str(path))
        assert loaded == data

    def test_save_json_creates_parent_dirs(self, temp_dir):
        data = {"test": True}
        path = Path(temp_dir) / "nested" / "dirs" / "test.json"
        save_json(data, str(path))
        assert path.exists()
        loaded = load_json(str(path))
        assert loaded == data


class TestCsvIO:
    """Test CSV save/load functions."""

    def test_save_and_load_csv(self, temp_dir):
        rows = [
            {"epoch": 1, "loss": 0.5, "accuracy": 0.8},
            {"epoch": 2, "loss": 0.3, "accuracy": 0.9},
            {"epoch": 3, "loss": 0.1, "accuracy": 0.95},
        ]
        path = Path(temp_dir) / "test.csv"
        save_csv(rows, str(path))
        loaded = load_csv(str(path))
        # CSV loads everything as strings
        assert len(loaded) == 3
        assert loaded[0]["epoch"] == "1"
        assert loaded[1]["loss"] == "0.3"

    def test_save_csv_with_fieldnames(self, temp_dir):
        rows = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
        ]
        path = Path(temp_dir) / "ordered.csv"
        save_csv(rows, str(path), fieldnames=["c", "b", "a"])
        # Read the file directly to check order
        with open(path) as f:
            header = f.readline().strip()
            assert header == "c,b,a"

    def test_save_empty_csv(self, temp_dir):
        path = Path(temp_dir) / "empty.csv"
        save_csv([], str(path))
        # Empty list should not create file or create empty file
        # The current implementation returns early, so file won't exist
        assert not path.exists()

    def test_save_csv_creates_parent_dirs(self, temp_dir):
        rows = [{"x": 1}]
        path = Path(temp_dir) / "nested" / "dirs" / "test.csv"
        save_csv(rows, str(path))
        assert path.exists()
