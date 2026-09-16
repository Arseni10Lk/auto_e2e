import inspect
from typing import Dict, Any
import numpy as np

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree


class GroundTruthDeviationReward:
    """Evaluates trajectory tracking deviation against expert ground-truth demonstration.

    Computes displacement errors (ADE, FDE) between predicted trajectory and ground truth,
    applying a continuous tracking penalty and enforcing a hard threshold for 3DGS
    visual degradation (e.g. max deviation > 3.0m).
    """

    def __init__(
        self,
        ade_weight: float = 1.0,
        fde_weight: float = 0.5,
        max_deviation_threshold: float = 3.0,
        terminal_penalty: float = 10.0,
    ) -> None:
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.max_deviation_threshold = max_deviation_threshold
        self.terminal_penalty = terminal_penalty

    def compute(
        self,
        trajectory_xy: np.ndarray,
        gt_trajectory: np.ndarray,
    ) -> float:
        if len(trajectory_xy) == 0 or len(gt_trajectory) == 0:
            return 0.0

        n = min(len(trajectory_xy), len(gt_trajectory))
        pred_coords = trajectory_xy[:n, :2]
        gt_coords = gt_trajectory[:n, :2]

        diffs = pred_coords - gt_coords
        distances = np.linalg.norm(diffs, axis=-1)

        ade = np.mean(distances)
        fde = distances[-1]
        max_dev = np.max(distances)

        tracking_penalty = -(self.ade_weight * ade + self.fde_weight * fde)
        bound_penalty = (
            -self.terminal_penalty if max_dev > self.max_deviation_threshold else 0.0
        )

        return tracking_penalty + bound_penalty


class SafetyReward:
    """Handcrafted penalty for off-road driving violations (R_safety)."""

    def __init__(self) -> None:
        self._cached_map_version = None
        self._drivable_tree = None

    def compute(
        self,
        ego_pose: tuple[float, float, float],
        trajectory_xy: np.ndarray,
        navigation_map: Any,
    ) -> float:
        # 1. Build or retrieve the spatial index for the drivable area polygons
        if (
            self._cached_map_version != navigation_map.map_version
            or self._drivable_tree is None
        ):
            polygons = []
            for poly_primitive in navigation_map.drivable_polygons:
                pts = poly_primitive.points_enu_m[:, :2]  # Take (X, Y)
                if len(pts) >= 3:
                    polygons.append(Polygon(pts))
            self._drivable_tree = STRtree(polygons) if polygons else None
            self._cached_map_version = navigation_map.map_version

        if self._drivable_tree is None:
            raise ValueError("No drivable area defined")

        # 2. Transform trajectory from ego-centric to map frame (ENU)
        if len(trajectory_xy) == 0:
            return 0.0

        c, s = np.cos(ego_pose[2]), np.sin(ego_pose[2])
        rot_mat = np.array([[c, -s], [s, c]])
        traj_global = (trajectory_xy @ rot_mat.T) + np.array([ego_pose[0], ego_pose[1]])

        off_road_penalty = 0.0

        for x, y in traj_global:
            pt = Point(x, y)

            # --- Off-road check ---
            possible_matches = self._drivable_tree.query(pt)
            if not possible_matches.size:
                off_road_penalty -= 1.0
            else:
                is_on_road = False
                for idx in possible_matches:
                    if self._drivable_tree.geometries[idx].covers(pt):
                        is_on_road = True
                        break
                if not is_on_road:
                    off_road_penalty -= 1.0

        num_steps = len(traj_global)
        if num_steps > 0:
            return off_road_penalty / num_steps
        return 0.0


class RewardRegistry:
    """Manages active reward functions and their scaling weights."""

    def __init__(self, config_weights: Dict[str, float]) -> None:
        """Initialize the registry with specific weights.

        Args:
            config_weights: A dictionary mapping reward names to their weights.
                e.g., {'w_gt_dev': 1.0, 'w_safe': 0.5}
        """
        self.weights = config_weights
        self.rewards: Dict[str, Any] = {}

        if "w_gt_dev" in self.weights:
            self.rewards["w_gt_dev"] = GroundTruthDeviationReward()

        if "w_safe" in self.weights:
            self.rewards["w_safe"] = SafetyReward()

    def compute_total_reward(self, **kwargs: Any) -> tuple[float, Dict[str, float]]:
        """Compute the weighted sum of all registered rewards.

        Filters simulation kwargs to only those explicitly declared by each reward's compute method.

        Returns:
            A tuple containing:
                - The total scalar reward.
                - A dictionary of the unweighted individual components.
        """
        components: Dict[str, float] = {}
        total_reward = 0.0

        for name, reward_func in self.rewards.items():
            sig = inspect.signature(reward_func.compute)
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            val = reward_func.compute(**filtered)
            components[name] = val
            total_reward += self.weights[name] * val

        return total_reward, components


__all__ = [
    "GroundTruthDeviationReward",
    "SafetyReward",
    "RewardRegistry",
]
