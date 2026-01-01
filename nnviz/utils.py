"""Utility functions for nnviz."""

import csv
import json
import random
from pathlib import Path
from typing import Any, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Get torch device from string specification.

    Args:
        device_str: "auto", "cpu", "cuda", or "cuda:N"

    Returns:
        torch.device instance
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: str) -> None:
    """Save dictionary to JSON file."""
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Load dictionary from JSON file."""
    with open(path) as f:
        return json.load(f)


def save_csv(rows: List[dict], path: str, fieldnames: List[str] | None = None) -> None:
    """Save list of dictionaries to CSV file.

    Args:
        rows: List of dictionaries with consistent keys
        path: Output file path
        fieldnames: Optional explicit field order; if None, uses keys from first row
    """
    if not rows:
        return

    ensure_dir(str(Path(path).parent))

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: str) -> List[dict]:
    """Load CSV file as list of dictionaries."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
