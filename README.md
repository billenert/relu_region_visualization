# nnviz

Visualize the exact piecewise-affine decision regions of ReLU neural networks.

This tool trains a small MLP on image pixels and generates visualizations showing how the network partitions 2D space into linear regions.

## Installation

```bash
pip install -e .
```

Or run directly from the repository:

```bash
cd relu_region_visualization
python -m nnviz --help
```

## Usage

```bash
nnviz <image_path> --width <neurons> --depth <layers> [options]
```

**Example:**
```bash
nnviz photo.png --width 8 --depth 2
```

This will:
1. Convert the image to a binary classification dataset (dark pixels = 1, light pixels = 0)
2. Train an MLP with 2 hidden layers of 8 neurons each
3. Output visualizations to `runs/<timestamp>_<image_name>/`

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--width` | required | Neurons per hidden layer |
| `--depth` | required | Number of hidden layers |
| `--epochs` | 300 | Training epochs |
| `--lr` | 0.001 | Learning rate |
| `--threshold` | 0.5 | Darkness threshold for labeling pixels |
| `--save-dir` | auto | Output directory |
| `--exact-max-regions` | 20000 | Max regions before falling back to contour |

See `nnviz --help` for all options.

## Output Files

| File | Description |
|------|-------------|
| `viz_exact_regions.png` | Piecewise-affine regions (each color = one linear region) |
| `viz_boundary.png` | Decision boundary where network output = 0 |
| `viz_3d.png` | 3D surface plot of network logits |
| `train_log.csv` | Training loss and accuracy per epoch |
| `model.pt` | Saved model checkpoint |
| `config.json` | Configuration used for the run |

## How It Works

ReLU networks are piecewise-affine: the input space is partitioned into convex polytopes, and the network is linear within each region. This tool:

1. Enumerates all activation patterns using Sutherland-Hodgman polygon clipping
2. For each region, computes the exact affine function `z(x,y) = ax + by + c`
3. Renders the regions with deterministic colors based on activation patterns

For small networks (width ≤ 8, depth ≤ 3), exact enumeration typically succeeds. Larger networks may exceed the region limit, in which case the boundary visualization falls back to matplotlib contours.

(A small disclaimer: this network uses LeakyReLU, a modified version of ReLU! Oftentimes models using ReLU get stuck in local optima when training, which is something we don't want)

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Pillow
