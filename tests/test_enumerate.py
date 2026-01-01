"""Tests for nnviz.viz.exact_regions.enumerate module."""

import numpy as np
import pytest
import torch

from nnviz.model import MLP
from nnviz.utils import set_seed
from nnviz.viz.exact_regions.enumerate import (
    enumerate_regions,
    FinalRegion,
    RegionExplosionError,
)
from nnviz.viz.exact_regions.geometry import polygon_area


DOMAIN = ((0.0, 1.0), (0.0, 1.0))


class TestEnumerateRegionsBasic:
    """Basic tests for region enumeration."""

    def test_single_neuron_single_layer(self):
        """With width=1, depth=1, should get at most 2 regions."""
        set_seed(42)
        model = MLP(width=1, depth=1)

        regions = enumerate_regions(model, DOMAIN)

        assert isinstance(regions, list)
        assert 1 <= len(regions) <= 2  # One or both sides of the hyperplane

    def test_returns_final_regions(self):
        """Output should be list of FinalRegion objects."""
        set_seed(42)
        model = MLP(width=2, depth=1)

        regions = enumerate_regions(model, DOMAIN)

        assert isinstance(regions, list)
        for region in regions:
            assert isinstance(region, FinalRegion)
            assert region.vertices.shape[1] == 2
            assert region.a.shape == (2,)
            assert isinstance(region.b, float)
            assert isinstance(region.pattern, bytes)

    def test_regions_cover_domain(self):
        """Total area of regions should equal domain area."""
        set_seed(42)
        model = MLP(width=4, depth=2)

        regions = enumerate_regions(model, DOMAIN)

        total_area = sum(polygon_area(r.vertices) for r in regions)
        expected_area = 1.0  # Unit square
        assert abs(total_area - expected_area) < 1e-6

    def test_regions_have_valid_affine_form(self):
        """Each region should have a valid affine output form."""
        set_seed(42)
        model = MLP(width=2, depth=1)
        model.eval()

        regions = enumerate_regions(model, DOMAIN)

        for region in regions:
            # Pick a point inside the region (centroid)
            centroid = region.vertices.mean(axis=0)

            # Compute expected output using affine form
            expected = np.dot(region.a, centroid) + region.b

            # Compute actual output using model
            with torch.no_grad():
                x = torch.tensor(centroid, dtype=torch.float32).unsqueeze(0)
                actual = model(x).item()

            assert abs(expected - actual) < 1e-5, \
                f"Affine form mismatch: expected {expected}, got {actual}"


class TestEnumerateRegionsCounting:
    """Test region count for various architectures."""

    def test_width1_depth1_max2_regions(self):
        """width=1, depth=1 should produce at most 2 regions."""
        set_seed(0)
        model = MLP(width=1, depth=1)
        regions = enumerate_regions(model, DOMAIN)
        assert len(regions) <= 2

    def test_width2_depth1_max4_regions(self):
        """width=2, depth=1 should produce at most 4 regions."""
        set_seed(0)
        model = MLP(width=2, depth=1)
        regions = enumerate_regions(model, DOMAIN)
        # Each neuron can split each region into 2, so 2^2 = 4 max
        assert len(regions) <= 4

    def test_width2_depth2(self):
        """width=2, depth=2 - more complex case."""
        set_seed(0)
        model = MLP(width=2, depth=2)
        regions = enumerate_regions(model, DOMAIN)
        # Upper bound is much higher, but actual count depends on weights
        assert len(regions) >= 1


class TestRegionExplosion:
    """Test region count limiting."""

    def test_max_regions_exceeded(self):
        """Should raise RegionExplosionError when limit exceeded."""
        set_seed(42)
        model = MLP(width=8, depth=4)

        with pytest.raises(RegionExplosionError) as exc_info:
            enumerate_regions(model, DOMAIN, max_regions=10)

        assert exc_info.value.region_count > 10
        assert exc_info.value.layer >= 0
        assert exc_info.value.neuron >= 0

    def test_error_contains_info(self):
        """RegionExplosionError should contain useful info."""
        set_seed(42)
        model = MLP(width=8, depth=2)

        try:
            enumerate_regions(model, DOMAIN, max_regions=5)
            pytest.fail("Should have raised RegionExplosionError")
        except RegionExplosionError as e:
            assert "exceeded" in str(e).lower()
            assert hasattr(e, "layer")
            assert hasattr(e, "neuron")
            assert hasattr(e, "region_count")


class TestEnumerateRegionsDomain:
    """Test with different domains."""

    def test_non_unit_domain(self):
        """Should work with non-unit domain."""
        set_seed(42)
        model = MLP(width=2, depth=1)
        domain = ((-1.0, 1.0), (-0.5, 0.5))

        regions = enumerate_regions(model, domain)

        # Domain area is 2 * 1 = 2
        total_area = sum(polygon_area(r.vertices) for r in regions)
        assert abs(total_area - 2.0) < 1e-6

    def test_pixel_domain(self):
        """Should work with pixel coordinate domain."""
        set_seed(42)
        model = MLP(width=2, depth=1)
        domain = ((0.0, 255.0), (0.0, 255.0))

        regions = enumerate_regions(model, domain)

        # Domain area is 255 * 255 = 65025
        total_area = sum(polygon_area(r.vertices) for r in regions)
        assert abs(total_area - 65025.0) < 1e-3


class TestEnumerateRegionsPatterns:
    """Test activation pattern handling."""

    def test_patterns_exist(self):
        """Each region should have an activation pattern."""
        set_seed(42)
        model = MLP(width=4, depth=2)

        regions = enumerate_regions(model, DOMAIN)

        for region in regions:
            assert isinstance(region.pattern, bytes)
            assert len(region.pattern) > 0

    def test_pattern_length_consistent(self):
        """All regions should have the same pattern length."""
        set_seed(42)
        model = MLP(width=3, depth=2)

        regions = enumerate_regions(model, DOMAIN)

        # All patterns should have the same length
        lengths = [len(r.pattern) for r in regions]
        assert len(set(lengths)) == 1  # All same length

        # Pattern length should be width * depth (each neuron contributes 1 byte)
        expected_length = 3 * 2  # width * depth
        assert lengths[0] == expected_length


class TestEnumerateRegionsNumericalStability:
    """Test numerical stability."""

    def test_tight_tolerance(self):
        """Should work with tight tolerances."""
        set_seed(42)
        model = MLP(width=2, depth=1)

        regions = enumerate_regions(
            model, DOMAIN, eps=1e-12, area_eps=1e-15
        )

        assert len(regions) >= 1
        total_area = sum(polygon_area(r.vertices) for r in regions)
        assert abs(total_area - 1.0) < 1e-9

    def test_loose_tolerance(self):
        """Should work with loose tolerances."""
        set_seed(42)
        model = MLP(width=2, depth=1)

        regions = enumerate_regions(
            model, DOMAIN, eps=1e-6, area_eps=1e-9
        )

        assert len(regions) >= 1
