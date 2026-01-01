"""Main pipeline for nnviz."""

import os
from pathlib import Path

from .config import NNVizConfig
from .data import load_image, image_to_dataset, get_domain_bounds
from .model import MLP
from .train import train_model, save_checkpoint
from .utils import set_seed, get_device, ensure_dir, save_json, save_csv
from .viz.grid import create_evaluation_grid, evaluate_model_on_grid
from .viz.plot3d import plot_3d_surface
from .viz.boundary import (
    extract_boundary_segments,
    render_exact_boundary,
    render_contour_boundary,
)
from .viz.exact_regions import enumerate_regions, render_regions, RegionExplosionError


def run_pipeline(config: NNVizConfig) -> None:
    """Run the full nnviz pipeline.

    Args:
        config: Configuration for the pipeline
    """
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    ensure_dir(config.save_dir)

    print(f"nnviz - Neural Network Region Visualization")
    print(f"=" * 50)
    print(f"Image: {config.image_path}")
    print(f"Network: width={config.width}, depth={config.depth}")
    print(f"Device: {device}")
    print(f"Output: {config.save_dir}")
    print()

    # Save configuration
    config.save_json(os.path.join(config.save_dir, "config.json"))

    # Load image and create dataset
    print("Loading image and creating dataset...")
    image = load_image(config.image_path)
    X, Y = image_to_dataset(
        image,
        threshold=config.threshold,
        invert=config.invert,
        normalize_coords=config.normalize_coords,
        downsample=config.downsample,
    )
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"  Dataset size: {len(X)} samples")
    print(f"  Positive class: {Y.sum():.0f} ({100*Y.mean():.1f}%)")
    print()

    # Create model
    model = MLP(width=config.width, depth=config.depth)
    print(f"Model: {model.count_parameters()} parameters")
    print()

    # Train
    print(f"Training for {config.epochs} epochs...")
    logs = train_model(model, X, Y, config, device)

    # Save training log
    save_csv(
        logs,
        os.path.join(config.save_dir, "train_log.csv"),
        fieldnames=["epoch", "loss", "accuracy"],
    )

    # Save checkpoint
    save_checkpoint(model, os.path.join(config.save_dir, "model.pt"))

    print(f"  Final loss: {logs[-1]['loss']:.4f}")
    print(f"  Final accuracy: {logs[-1]['accuracy']:.4f}")
    print()

    # Get domain bounds
    domain = get_domain_bounds(image.shape, config.normalize_coords)

    # Try exact region enumeration
    print("Enumerating exact regions...")
    exact_regions = None
    try:
        exact_regions = enumerate_regions(
            model,
            domain,
            eps=config.exact_eps,
            area_eps=config.exact_area_eps,
            max_regions=config.exact_max_regions,
        )
        print(f"  Found {len(exact_regions)} regions")
    except RegionExplosionError as e:
        print(f"  Region explosion at layer {e.layer}, neuron {e.neuron}")
        print(f"  Count exceeded {config.exact_max_regions}")
        save_json(
            {
                "error": str(e),
                "layer": e.layer,
                "neuron": e.neuron,
                "region_count": e.region_count,
            },
            os.path.join(config.save_dir, "exact_failure.json"),
        )
    print()

    # Create evaluation grid for fallback and 3D plot
    print("Creating evaluation grid...")
    X_grid, Y_grid, points = create_evaluation_grid(domain, config.viz_res)
    Z_flat = evaluate_model_on_grid(model, points, device)
    Z = Z_flat.reshape(config.viz_res, config.viz_res)
    print(f"  Grid size: {config.viz_res}x{config.viz_res}")
    print()

    # Generate visualizations
    print("Generating visualizations...")

    # 3D plot (always)
    plot_3d_surface(
        X_grid, Y_grid, Z,
        os.path.join(config.save_dir, "viz_3d.png"),
    )
    print("  Saved viz_3d.png")

    # Exact regions visualization (if available)
    if exact_regions is not None:
        render_regions(
            exact_regions,
            domain,
            os.path.join(config.save_dir, "viz_exact_regions.png"),
        )
        print("  Saved viz_exact_regions.png")

        # Exact boundary
        segments = extract_boundary_segments(exact_regions, config.exact_eps)
        render_exact_boundary(
            segments,
            domain,
            os.path.join(config.save_dir, "viz_boundary.png"),
        )
        print("  Saved viz_boundary.png (exact)")
    else:
        # Contour fallback for boundary
        render_contour_boundary(
            X_grid, Y_grid, Z,
            os.path.join(config.save_dir, "viz_boundary.png"),
        )
        print("  Saved viz_boundary.png (contour fallback)")

    print()
    print("Done!")
