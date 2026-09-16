from typing import Any, List
import math
from pathlib import Path

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    ModelPrediction,
    PredictionInput,
)

from .config import DEFAULT_CAMERA_NAMES
from .parser import AlpasimStreamParser


def _extract_yaw(quat: Any) -> float:
    """Extract yaw heading angle from quaternion."""
    return math.atan2(
        2.0 * (quat.w * quat.z + quat.x * quat.y),
        1.0 - 2.0 * (quat.y**2 + quat.z**2),
    )


def _unroll_unicycle_controls(
    controls: np.ndarray, v_init: float, dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate (acceleration, curvature) controls into (x, y) waypoints and headings."""
    points = np.zeros_like(controls, dtype=np.float32)
    headings = np.zeros(len(controls), dtype=np.float32)
    x, y, theta = 0.0, 0.0, 0.0
    v = v_init

    for i in range(len(controls)):
        a, k = controls[i, 0], controls[i, 1]
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += v * k * dt
        v += a * dt
        points[i, 0] = x
        points[i, 1] = y
        headings[i] = theta

    return points, headings


class AutoE2EDriver(BaseTrajectoryModel):
    """AutoE2E driver plugin for AlpaSim."""

    def __init__(
        self,
        model_checkpoint: str = "dummy_random.ckpt",
        allow_mock: bool = False,
        allow_untrained_model: bool = False,
        camera_ids: List[str] | None = None,
        scene_id: str | None = None,
    ) -> None:
        super().__init__()
        self.allow_mock = allow_mock
        self.allow_untrained_model = allow_untrained_model
        self.model_checkpoint = model_checkpoint
        self._camera_ids = camera_ids or DEFAULT_CAMERA_NAMES

        self.parser = AlpasimStreamParser(
            camera_names=self._camera_ids,
            scene_id=scene_id,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if model_checkpoint and Path(model_checkpoint).exists():
            checkpoint = torch.load(model_checkpoint, map_location=self.device)
            if hasattr(checkpoint, "forward"):
                self.model = checkpoint
            else:
                from model_components.auto_e2e import AutoE2E

                self.model = AutoE2E(
                    num_views=len(self._camera_ids), is_pretrained=False
                ).to(self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])

            self.model.eval()
        elif self.allow_untrained_model:
            from model_components.auto_e2e import AutoE2E

            self.model = AutoE2E(
                num_views=len(self._camera_ids), is_pretrained=False
            ).to(self.device)
            self.model.eval()
        elif not self.allow_mock:
            raise FileNotFoundError(
                f"Model checkpoint '{model_checkpoint}' not found and allow_mock=False."
            )

    @classmethod
    def from_config(
        cls,
        model_cfg: Any = None,
        device: torch.device = torch.device("cpu"),
        camera_ids: List[str] | None = None,
        context_length: int | None = None,
        output_frequency_hz: int = 10,
    ) -> "AutoE2EDriver":
        checkpoint_path = model_cfg.checkpoint_path if model_cfg is not None else "MOCK"
        driver = cls(
            model_checkpoint=checkpoint_path,
            allow_mock=checkpoint_path == "MOCK" or not checkpoint_path,
            allow_untrained_model=checkpoint_path == "UNTRAINED",
            camera_ids=camera_ids,
        )
        driver.device = device
        return driver

    @property
    def camera_ids(self) -> List[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return 1

    @property
    def output_frequency_hz(self) -> int:
        return 10

    def _encode_command(self, command: Any) -> None:
        """AutoE2E predicts trajectories end-to-end without discrete driving commands."""
        return None

    def predict(self, input_data: PredictionInput) -> ModelPrediction:
        """Process real-time PredictionInput to ModelPrediction.

        Returns:
            ModelPrediction with trajectory_xy [64, 2] and headings [64].
        """
        cameras_dict = {
            cam_name: frames[-1].image
            for cam_name, frames in input_data.camera_images.items()
        }

        speed = input_data.speed
        acceleration = input_data.acceleration

        yaw_rate = 0.0
        curvature = 0.0
        ego_pose = None
        ego_pose_history = input_data.ego_pose_history
        if ego_pose_history and len(ego_pose_history) >= 2:
            prev = ego_pose_history[-2]
            curr = ego_pose_history[-1]
            dt = (curr.timestamp_us - prev.timestamp_us) / 1_000_000.0

            curr_yaw = _extract_yaw(curr.pose.quat)
            ego_pose = (curr.pose.x, curr.pose.y, curr_yaw)

            if dt > 0:
                prev_yaw = _extract_yaw(prev.pose.quat)
                diff = math.atan2(
                    math.sin(curr_yaw - prev_yaw), math.cos(curr_yaw - prev_yaw)
                )
                yaw_rate = diff / dt
                curvature = yaw_rate / max(speed, 0.1)

        observation = {
            "cameras": cameras_dict,
            "speed": speed,
            "acceleration": acceleration,
            "yaw_rate": yaw_rate,
            "curvature": curvature,
            "ego_pose": ego_pose,
        }

        parsed = self.parser.parse_observation(observation)
        tensors = {
            k: v.to(self.device) if hasattr(v, "to") else v for k, v in parsed.items()
        }

        if self.model is not None:
            with torch.no_grad():
                outputs = self.model(**tensors, mode="inference")
                points, headings = _unroll_unicycle_controls(
                    outputs[0].cpu().numpy().reshape(64, 2), speed
                )
        else:
            if not self.allow_mock:
                raise RuntimeError(
                    f"Model checkpoint '{self.model_checkpoint}' failed to load and allow_mock=False. "
                    "Cannot execute live inference without a loaded model."
                )
            x = np.linspace(0.0, max(speed, 1.0) * 6.4, 64, dtype=np.float32)
            points = np.stack([x, np.zeros(64, dtype=np.float32)], axis=1)
            headings = np.zeros(64, dtype=np.float32)

        return ModelPrediction(
            trajectory_xy=points.astype(np.float32),
            headings=headings.astype(np.float32),
        )


__all__ = [
    "AutoE2EDriver",
    "ModelPrediction",
    "PredictionInput",
]
