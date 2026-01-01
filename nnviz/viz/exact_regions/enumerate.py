"""Region enumeration algorithm for exact piecewise-affine partition."""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from ...model import MLP
from .clip import split_polygon_by_halfspace
from .geometry import create_domain_rectangle


class RegionExplosionError(Exception):
    """Raised when the number of regions exceeds the maximum limit."""

    def __init__(self, message: str, layer: int, neuron: int, region_count: int):
        super().__init__(message)
        self.layer = layer
        self.neuron = neuron
        self.region_count = region_count


@dataclass
class Region:
    """Intermediate region during enumeration.

    Attributes:
        vertices: (N, 2) CCW polygon vertices
        A: (width, 2) affine weight matrix for h_l(x) = A @ x + c
        c: (width,) affine bias vector
        pattern: Activation pattern bits accumulated so far
    """
    vertices: np.ndarray
    A: np.ndarray
    c: np.ndarray
    pattern: bytes


@dataclass
class FinalRegion:
    """Final region with output logit affine form.

    Attributes:
        vertices: (N, 2) CCW polygon vertices
        a: (2,) gradient of output logit z(x) = a^T x + b
        b: Scalar bias of output logit
        pattern: Full activation pattern for coloring
    """
    vertices: np.ndarray
    a: np.ndarray
    b: float
    pattern: bytes


def enumerate_regions(
    model: MLP,
    domain: Tuple[Tuple[float, float], Tuple[float, float]],
    eps: float = 1e-9,
    area_eps: float = 1e-12,
    max_regions: int = 20000,
) -> Union[List[FinalRegion], dict]:
    """Enumerate all piecewise-affine regions of the network.

    Args:
        model: Trained MLP model
        domain: ((x_min, x_max), (y_min, y_max)) bounds
        eps: Tolerance for halfspace membership
        area_eps: Minimum polygon area
        max_regions: Maximum number of regions before aborting

    Returns:
        List of FinalRegion if successful, or dict with failure info if
        region count exceeds max_regions.
    """
    leaky_slope = model.LEAKY_SLOPE
    width = model.width
    depth = model.depth

    # Get layer weights
    layer_weights = model.get_layer_weights()
    # layer_weights[l] = (W_l, b_l) for l in 0..depth-1 (hidden layers)
    # layer_weights[depth] = (W_out, b_out) for output layer

    # Initialize with domain rectangle
    # h_0(x) = x, so A_0 = I_2, c_0 = 0
    initial_vertices = create_domain_rectangle(domain[0], domain[1])
    A_0 = np.eye(2, dtype=np.float64)
    c_0 = np.zeros(2, dtype=np.float64)

    regions: List[Region] = [
        Region(
            vertices=initial_vertices,
            A=A_0,
            c=c_0,
            pattern=b"",
        )
    ]

    # Process each hidden layer
    for layer_idx in range(depth):
        W_l, b_l = layer_weights[layer_idx]
        new_regions: List[Region] = []

        for region_idx, region in enumerate(regions):
            # Compute pre-activation affine form
            # s_l(x) = W_l @ h_{l-1}(x) + b_l
            #        = W_l @ (A_{l-1} @ x + c_{l-1}) + b_l
            #        = (W_l @ A_{l-1}) @ x + (W_l @ c_{l-1} + b_l)
            B_l = W_l @ region.A  # (width, 2)
            d_l = W_l @ region.c + b_l  # (width,)

            # Split by each neuron's hyperplane
            subregions = [(region.vertices, [])]  # (polygon, pattern_bits)

            for neuron_idx in range(width):
                u = B_l[neuron_idx]  # (2,)
                v = d_l[neuron_idx]  # scalar

                next_subregions = []

                for poly, pattern_bits in subregions:
                    pos_poly, neg_poly = split_polygon_by_halfspace(
                        poly, u, v, eps, area_eps
                    )

                    if pos_poly is not None:
                        next_subregions.append((pos_poly, pattern_bits + [1]))
                    if neg_poly is not None:
                        next_subregions.append((neg_poly, pattern_bits + [0]))

                subregions = next_subregions

                # Check region count - simple check without estimating remaining
                current_count = len(new_regions) + len(subregions)
                if current_count > max_regions:
                    raise RegionExplosionError(
                        f"Region count exceeded {max_regions}",
                        layer=layer_idx,
                        neuron=neuron_idx,
                        region_count=current_count,
                    )

            # For each subregion, compute the affine transformation for h_l
            for poly, pattern_bits in subregions:
                # D_l is diagonal with 1 where pattern_bit=1, leaky_slope where 0
                D_l = np.diag([1.0 if bit else leaky_slope for bit in pattern_bits])

                # h_l(x) = D_l @ s_l(x) = D_l @ (B_l @ x + d_l)
                A_l = D_l @ B_l
                c_l = D_l @ d_l

                # Accumulate pattern bytes
                # Convert bits to bytes (pack 8 bits per byte)
                new_pattern = region.pattern + bytes(pattern_bits)

                new_regions.append(Region(
                    vertices=poly,
                    A=A_l,
                    c=c_l,
                    pattern=new_pattern,
                ))

        regions = new_regions

        # Check total count after layer
        if len(regions) > max_regions:
            raise RegionExplosionError(
                f"Region count exceeded {max_regions} after layer {layer_idx}",
                layer=layer_idx,
                neuron=width - 1,
                region_count=len(regions),
            )

    # Compute final output affine form for each region
    W_out, b_out = layer_weights[depth]
    W_out = W_out.ravel()  # (width,) -> (width,)
    b_out = b_out.item() if hasattr(b_out, 'item') else float(b_out[0])

    final_regions: List[FinalRegion] = []

    for region in regions:
        # z(x) = W_out^T @ h_L(x) + b_out
        #      = W_out^T @ (A_L @ x + c_L) + b_out
        #      = (W_out^T @ A_L) @ x + (W_out^T @ c_L + b_out)
        a = W_out @ region.A  # (2,)
        b = np.dot(W_out, region.c) + b_out  # scalar

        final_regions.append(FinalRegion(
            vertices=region.vertices,
            a=a,
            b=b,
            pattern=region.pattern,
        ))

    return final_regions
