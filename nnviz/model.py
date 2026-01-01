"""MLP model definition for nnviz."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multilayer perceptron for binary classification in R^2.

    Architecture:
    - Input: 2 dimensions (x, y coordinates)
    - Hidden layers: `depth` layers, each with `width` neurons
    - Activation: LeakyReLU with negative_slope=0.02
    - Output: 1 logit (no sigmoid applied)
    """

    LEAKY_SLOPE = 0.02

    def __init__(self, width: int, depth: int):
        """Initialize the MLP.

        Args:
            width: Number of neurons in each hidden layer
            depth: Number of hidden layers (must be >= 1)
        """
        super().__init__()

        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.width = width
        self.depth = depth

        # Build layers
        layers = []

        # First hidden layer: 2 -> width
        layers.append(nn.Linear(2, width))
        layers.append(nn.LeakyReLU(negative_slope=self.LEAKY_SLOPE))

        # Remaining hidden layers: width -> width
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.LeakyReLU(negative_slope=self.LEAKY_SLOPE))

        self.hidden = nn.Sequential(*layers)

        # Output layer: width -> 1
        self.output = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (batch_size, 2)

        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        h = self.hidden(x)
        return self.output(h)

    def get_layer_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract weights and biases from all layers.

        Returns:
            List of (weight, bias) tuples for each layer including output.
            For hidden layer l: weight is shape (width, in_features), bias is (width,)
            For output layer: weight is shape (1, width), bias is (1,)

        Note:
            Weights are returned as numpy arrays in float64 for numerical precision
            in exact region enumeration.
        """
        result = []

        # Extract hidden layer weights
        for i, layer in enumerate(self.hidden):
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().cpu().numpy().astype(np.float64)
                b = layer.bias.detach().cpu().numpy().astype(np.float64)
                result.append((W, b))

        # Extract output layer weights
        W_out = self.output.weight.detach().cpu().numpy().astype(np.float64)
        b_out = self.output.bias.detach().cpu().numpy().astype(np.float64)
        result.append((W_out, b_out))

        return result

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
