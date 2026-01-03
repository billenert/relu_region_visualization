# Architecture Overview

This document describes the high-level architecture of `nnviz`, a tool for visualizing the exact piecewise-affine decision regions of ReLU neural networks.

## Core Concept

ReLU networks are piecewise-affine functions: the input space is partitioned into convex polytopes, and within each region the network computes a different linear function `z(x,y) = ax + by + c`. This tool makes that structure visible.

## Data Flow

```
Image → Dataset → Train MLP → Extract Weights → Enumerate Regions → Render
                                    ↓
                              Evaluate Grid → 3D Plot / Contour Fallback
```

1. **Image Loading** (`data.py`): Convert image to grayscale, threshold pixels into binary labels
2. **Training** (`train.py`): Train a LeakyReLU MLP on (x,y) → label
3. **Region Enumeration** (`viz/exact_regions/`): Compute exact polygonal regions
4. **Visualization** (`viz/`): Render regions, decision boundary, and 3D surface

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Dataclass holding all configuration with defaults |
| `data.py` | Image → (X, Y) dataset conversion with coordinate normalization |
| `model.py` | MLP architecture with weight extraction for geometry |
| `train.py` | Training loop with BCEWithLogitsLoss + AdamW |
| `cli.py` | Argument parsing |
| `main.py` | Pipeline orchestration |

### Visualization (`viz/`)

| Module | Purpose |
|--------|---------|
| `grid.py` | Create evaluation grid, batch model inference |
| `plot3d.py` | 3D surface plot of network logits |
| `boundary.py` | Decision boundary rendering (exact or contour fallback) |

### Exact Regions (`viz/exact_regions/`)

| Module | Purpose |
|--------|---------|
| `geometry.py` | Polygon area calculation, point deduplication |
| `clip.py` | Sutherland-Hodgman halfspace clipping |
| `enumerate.py` | Region enumeration through network layers |
| `render.py` | Polygon rendering with deterministic coloring |

## Key Algorithm: Region Enumeration

The core algorithm in `enumerate.py` tracks how the input space is partitioned by each neuron's activation threshold.

### Mathematical Foundation

For a network with layers computing `h_l(x) = LeakyReLU(W_l @ h_{l-1}(x) + b_l)`:

1. Within any region, each layer's output is affine: `h_l(x) = A_l @ x + c_l`
2. Each neuron `j` in layer `l` has a hyperplane `s_{l,j}(x) = 0` where its activation changes
3. The hyperplane equation is `u^T x + v = 0` where `u = (W_l @ A_{l-1})[j]` and `v = (W_l @ c_{l-1} + b_l)[j]`

### Algorithm Steps

1. Start with the domain rectangle as a single region
2. For each layer:
   - Compute the affine pre-activation: `B_l = W_l @ A_{l-1}`, `d_l = W_l @ c_{l-1} + b_l`
   - For each neuron: split all current polygons by its hyperplane
   - Track activation bits (1 = positive slope, 0 = leaky slope)
   - Update affine parameters: `A_l = D_l @ B_l`, `c_l = D_l @ d_l` where `D_l` is diagonal with 1 or 0.02
3. Compute final output affine form: `z(x) = w_out^T @ (A_L @ x + c_L) + b_out`

### Polygon Clipping

Sutherland-Hodgman algorithm clips polygons against halfspaces `f(x) >= 0`:
- For each edge, compute which endpoints are inside/outside
- Output appropriate vertices and intersection points
- Handle numerical edge cases with epsilon tolerances

## Coordinate System

- Image pixel `(i, j)` maps to coordinate `(x, y)` where:
  - `i` (column) → `x`
  - `j` (row) → `y`, with y-axis flipped so image top = plot top
- Unit normalization: coordinates in `[0, 1]^2`
- Pixel normalization: coordinates match pixel indices

## Region Explosion Protection

For large networks, the number of regions grows exponentially. The algorithm:
- Tracks region count during enumeration
- Aborts if count exceeds `--exact-max-regions` (default 20,000)
- Falls back to matplotlib contour for boundary visualization
- Still produces 3D surface plot from grid evaluation

## Deterministic Coloring

Each region is colored based on its activation pattern (which neurons are positive vs. leaky):
- Pattern stored as bytes (one bit per neuron across all layers)
- SHA256 hash computed from pattern bytes
- First 3 bytes → RGB color, brightened for visibility
