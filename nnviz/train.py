"""Training loop for nnviz."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import NNVizConfig
from .model import MLP


def create_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create a DataLoader from numpy arrays.

    Args:
        X: Input features of shape (N, 2)
        Y: Labels of shape (N,)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        generator: Optional random generator for reproducibility

    Returns:
        DataLoader instance
    """
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).float().unsqueeze(1)  # (N,) -> (N, 1)

    dataset = TensorDataset(X_tensor, Y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_epoch(
    model: MLP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: The MLP model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        (mean_loss, accuracy) tuple
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Compute accuracy: prediction is 1 if logit > 0, else 0
        predictions = (logits > 0).float()
        total_correct += (predictions == Y_batch).sum().item()

    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return mean_loss, accuracy


def train_model(
    model: MLP,
    X: np.ndarray,
    Y: np.ndarray,
    config: NNVizConfig,
    device: torch.device,
) -> List[Dict[str, float]]:
    """Train the model for the specified number of epochs.

    Args:
        model: The MLP model
        X: Training features of shape (N, 2)
        Y: Training labels of shape (N,)
        config: Training configuration
        device: Device to train on

    Returns:
        List of dictionaries containing epoch training logs
    """
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    dataloader = create_dataloader(
        X, Y,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    logs = []

    for epoch in range(config.epochs):
        mean_loss, accuracy = train_epoch(
            model, dataloader, optimizer, criterion, device
        )

        log_entry = {
            "epoch": epoch,
            "loss": mean_loss,
            "accuracy": accuracy,
        }
        logs.append(log_entry)

    return logs


def save_checkpoint(model: MLP, path: str) -> None:
    """Save model checkpoint.

    Args:
        model: The MLP model
        path: Path to save checkpoint
    """
    torch.save({
        "width": model.width,
        "depth": model.depth,
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: str, device: torch.device | None = None) -> MLP:
    """Load model from checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded MLP model
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MLP(width=checkpoint["width"], depth=checkpoint["depth"])
    model.load_state_dict(checkpoint["state_dict"])

    if device is not None:
        model.to(device)

    return model
