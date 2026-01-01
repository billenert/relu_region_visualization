"""Configuration dataclass for nnviz."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path
from datetime import datetime


@dataclass
class NNVizConfig:
    """Configuration for the nnviz neural network visualization tool."""

    # Required parameters
    image_path: str
    width: int
    depth: int

    # Training parameters
    epochs: int = 300
    lr: float = 1e-3
    batch_size: int = 8192
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "auto"
    downsample: int = 1

    # Labeling parameters
    threshold: float = 0.5
    invert: bool = True

    # Coordinate normalization
    normalize_coords: str = "unit"  # "unit" or "pixel"

    # Visualization parameters
    viz_res: int = 512

    # Exact region enumeration parameters
    exact_eps: float = 1e-9
    exact_area_eps: float = 1e-12
    exact_max_regions: int = 20000
    exact_order: str = "neuronwise"

    # Output directory (computed if None)
    save_dir: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and set computed fields."""
        self.validate()
        if self.save_dir is None:
            self.save_dir = self._default_save_dir()

    def validate(self) -> None:
        """Validate configuration values."""
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {self.width}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {self.downsample}")
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.normalize_coords not in ("unit", "pixel"):
            raise ValueError(
                f"normalize_coords must be 'unit' or 'pixel', got {self.normalize_coords}"
            )
        if self.viz_res < 1:
            raise ValueError(f"viz_res must be >= 1, got {self.viz_res}")
        if self.exact_eps <= 0:
            raise ValueError(f"exact_eps must be > 0, got {self.exact_eps}")
        if self.exact_area_eps <= 0:
            raise ValueError(f"exact_area_eps must be > 0, got {self.exact_area_eps}")
        if self.exact_max_regions < 1:
            raise ValueError(
                f"exact_max_regions must be >= 1, got {self.exact_max_regions}"
            )
        if self.exact_order not in ("neuronwise",):
            raise ValueError(f"exact_order must be 'neuronwise', got {self.exact_order}")

    def _default_save_dir(self) -> str:
        """Generate default save directory path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_stem = Path(self.image_path).stem
        return f"runs/{timestamp}_{image_stem}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, d: dict) -> "NNVizConfig":
        """Create config from dictionary."""
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "NNVizConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls, path: str) -> "NNVizConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())
