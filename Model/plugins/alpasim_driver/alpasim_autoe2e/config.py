"""Configuration dataclasses for the AutoE2E AlpaSim driver plugin.

Defines model checkpoints, camera topology settings, and trajectory planning horizon settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Tuple, Dict

_CALIB_DIR = Path(__file__).resolve().parent / "configs" / "calibration"
_DATASET_DIR = Path(__file__).resolve().parent / "configs" / "dataset"

DEFAULT_CAMERA_NAMES: List[str] = [
    "camera_base_front_center",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear",
    "camera_ring_rear_left",
    "camera_ring_rear_right",
]

DEFAULT_IMAGE_MEAN: List[float] = [0.485, 0.456, 0.406]
DEFAULT_IMAGE_STD: List[float] = [0.229, 0.224, 0.225]


def load_dataset_config(
    dataset_config_path: str | Path | None = None,
) -> dict:
    """Load dataset configuration (mean, std, etc.) from JSON."""
    path = (
        Path(dataset_config_path) if dataset_config_path else _DATASET_DIR / "kit_scenes.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_transform(
    image_mean: List[float] | None = None,
    image_std: List[float] | None = None,
):
    """Build image preprocessing transform using dataset-specific normalization."""
    from torchvision import transforms

    if image_mean is None or image_std is None:
        cfg = load_dataset_config()
        image_mean = image_mean or cfg["image_mean"]
        image_std = image_std or cfg["image_std"]

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )


def load_projection_matrices(
    calibration_path: str | Path | None = None,
) -> Dict[str, list[list[float]]]:
    """Load camera projection matrices from a JSON calibration file."""
    path = (
        Path(calibration_path) if calibration_path else _CALIB_DIR / "kit_scenes.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class AutoE2EAlpaSimConfig:
    """Configuration options for ``AutoE2EAlpaSimModel`` driver plugin.

    Registered with AlpaSim under entry point ``alpasim.configs``.
    """

    checkpoint_path: str
    """Path to trained AutoE2E model checkpoint file."""

    allow_mock: bool = False
    """Whether to allow mock fallback mode when running without AlpaSim."""
    allow_untrained_model: bool = False
    """Whether to initialize the model randomly if weights are missing (useful for dry runs)."""

    rewards: Dict[str, float] = field(default_factory=dict)
    """Dictionary mapping reward component names to their scalar weights."""

    image_size: Tuple[int, int] = (256, 256)
    """Target camera resolution ``(H, W)`` expected by perception backbone."""

    planning_horizon_s: float = 6.4
    """Total future trajectory planning horizon in seconds."""

    planning_steps: int = 64
    """Number of output waypoint steps along the planning horizon."""

    camera_names: List[str] = field(default_factory=lambda: list(DEFAULT_CAMERA_NAMES))
    """List of 6 camera names matching KitScenes model input contract."""

    scene_id: str | None = None
    """KITScenes scene ID (e.g., 'c34c778f-...') to load offline map and trajectory masks natively."""

    image_mean: List[float] = field(default_factory=lambda: list(DEFAULT_IMAGE_MEAN))
    """Mean per RGB channel for input image normalization."""

    image_std: List[float] = field(default_factory=lambda: list(DEFAULT_IMAGE_STD))
    """Standard deviation per RGB channel for input image normalization."""
