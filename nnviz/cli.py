"""Command-line interface for nnviz."""

import argparse
from typing import List, Optional

from .config import NNVizConfig


def parse_args(argv: Optional[List[str]] = None) -> NNVizConfig:
    """Parse command-line arguments and return configuration.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        NNVizConfig instance with parsed values
    """
    parser = argparse.ArgumentParser(
        prog="nnviz",
        description="Train an MLP on image pixels and visualize decision regions.",
    )

    # Required arguments
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Number of neurons in each hidden layer",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="Number of hidden layers",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size for training (default: 8192)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on: auto, cpu, cuda, cuda:N (default: auto)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample training pixels by this factor (default: 1)",
    )

    # Labeling parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Darkness threshold for labeling (default: 0.5)",
    )
    parser.add_argument(
        "--invert",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Invert pixel values so black=1, white=0 (default: true)",
    )

    # Coordinate normalization
    parser.add_argument(
        "--normalize-coords",
        type=str,
        default="unit",
        choices=["unit", "pixel"],
        help="Coordinate normalization: unit ([0,1]^2) or pixel (default: unit)",
    )

    # Visualization parameters
    parser.add_argument(
        "--viz-res",
        type=int,
        default=512,
        help="Resolution for grid-based visualization (default: 512)",
    )

    # Exact region parameters
    parser.add_argument(
        "--exact-eps",
        type=float,
        default=1e-9,
        help="Tolerance for halfspace membership (default: 1e-9)",
    )
    parser.add_argument(
        "--exact-area-eps",
        type=float,
        default=1e-12,
        help="Minimum polygon area (default: 1e-12)",
    )
    parser.add_argument(
        "--exact-max-regions",
        type=int,
        default=20000,
        help="Maximum number of regions before aborting (default: 20000)",
    )
    parser.add_argument(
        "--exact-order",
        type=str,
        default="neuronwise",
        choices=["neuronwise"],
        help="Region enumeration order (default: neuronwise)",
    )

    # Output directory
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory (default: runs/<timestamp>_<image_stem>)",
    )

    args = parser.parse_args(argv)

    # Convert parsed arguments to config
    config = NNVizConfig(
        image_path=args.image_path,
        width=args.width,
        depth=args.depth,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        downsample=args.downsample,
        threshold=args.threshold,
        invert=args.invert.lower() == "true",
        normalize_coords=args.normalize_coords,
        viz_res=args.viz_res,
        exact_eps=args.exact_eps,
        exact_area_eps=args.exact_area_eps,
        exact_max_regions=args.exact_max_regions,
        exact_order=args.exact_order,
        save_dir=args.save_dir,
    )

    return config
