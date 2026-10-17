"""Run the AutoE2E driver once on synthetic AlpaSim input, without a simulator.

Use it as a pre-flight check wherever the driver will run (e.g. inside the
AlpaSim driver container): it loads the checkpoint, builds the T8 inputs from
eight frames per camera and 10 s of ego poses, and prints the planned path.

    python Model/plugins/alpasim_driver/examples/smoke_test_alpasim.py \
        --checkpoint /mnt/drivers/autoe2e/checkpoint.pt --device cuda
"""

import argparse
import time

import numpy as np
from alpasim_driver.models.base import PredictionInput
from alpasim_grpc.v0 import common_pb2

from alpasim_autoe2e.plugin import AutoE2EDriver
from alpasim_autoe2e.parser import CONTEXT_FRAMES

CAMERAS = {
    "camera_base_front_center": (1552, 2928),
    "camera_ring_front_left": (568, 876),
    "camera_ring_front_right": (568, 876),
    "camera_ring_rear_left": (568, 876),
    "camera_ring_rear": (568, 876),
    "camera_ring_rear_right": (568, 876),
}
T0_US = 10_000_000


def synthetic_input(speed: float) -> PredictionInput:
    """Grey frames 500 ms apart and a straight drive at ``speed`` m/s."""
    images = {
        camera: [
            (T0_US - (CONTEXT_FRAMES - 1 - k) * 500_000, np.full((*hw, 3), 128, np.uint8))
            for k in range(CONTEXT_FRAMES)
        ]
        for camera, hw in CAMERAS.items()
    }
    poses = []
    for t_us in range(0, T0_US + 1, 100_000):
        pose = common_pb2.PoseAtTime(timestamp_us=t_us)
        pose.pose.vec.x = speed * t_us / 1e6
        pose.pose.quat.w = 1.0
        poses.append(pose)
    return PredictionInput(
        camera_images=images,
        command=0,
        speed=speed,
        acceleration=0.0,
        ego_pose_history=poses,
        inference_seed=0,
        previous_plan=None,
        route=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="UNTRAINED", help="path, MOCK or UNTRAINED")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed", type=float, default=8.0)
    args = parser.parse_args()

    driver = AutoE2EDriver(args.checkpoint, list(CAMERAS), device=args.device)
    prediction_input = synthetic_input(args.speed)
    start = time.perf_counter()
    prediction = driver.predict(prediction_input)
    elapsed = time.perf_counter() - start

    positions = prediction.selected_positions
    print(f"cameras: {driver.camera_ids}")
    print(f"predict: {elapsed:.2f} s, {len(positions)} waypoints, finite={np.isfinite(positions).all()}")
    for seconds in (1, 3, 6.4):
        x, y, _ = positions[int(round(seconds * 10)) - 1]
        print(f"  t+{seconds:>3} s: x={x:7.2f} m  y={y:6.2f} m")


if __name__ == "__main__":
    main()
