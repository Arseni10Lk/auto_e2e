"""Export a nuPlan v1.1 camera rig into the plugin's calibration profile.

nuPlan training images are undistorted to a rectified pinhole (OpenCV
``getOptimalNewCameraMatrix`` with alpha 0) before packing, so the profile
stores the rectified intrinsics and the camera-to-ego extrinsics, read from the
``camera`` table of one log database the same way
``data_parsing/nuplan/packing.py`` reads them. A world rendered for the plugin
must render these rectified pinhole cameras.

Usage (needs OpenCV):
    python Model/plugins/alpasim_driver/tools/export_nuplan_calibration.py \
        ~/nuplan/dataset/nuplan-v1.1/splits/mini/<log>.db \
        Model/plugins/alpasim_driver/alpasim_autoe2e/configs/calibration/nuplan.json
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS  # noqa: E402
from data_parsing.nuplan.packing import (  # noqa: E402
    NUPLAN_CAMERA_CHANNELS,
    _decode_pickle_vector,
    _quaternion_transform,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_db", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    connection = sqlite3.connect(args.log_db)
    connection.row_factory = sqlite3.Row
    rows = {
        str(row["channel"]): row
        for row in connection.execute(
            "SELECT channel, translation, rotation, intrinsic, distortion, width, height "
            "FROM camera"
        )
    }
    vehicle = connection.execute("SELECT vehicle_name FROM log").fetchone()[0]
    connection.close()

    cameras = []
    for channel, slot in zip(NUPLAN_CAMERA_CHANNELS, CANONICAL_SIX_CAMERA_SLOTS):
        row = rows[channel]
        width, height = int(row["width"]), int(row["height"])
        intrinsic = _decode_pickle_vector(
            row["intrinsic"], expected_shape=(3, 3), name=f"{channel} intrinsic"
        )
        distortion = _decode_pickle_vector(
            row["distortion"], expected_shape=(5,), name=f"{channel} distortion"
        )
        rectified, _ = cv2.getOptimalNewCameraMatrix(
            intrinsic, distortion, (width, height), 0.0, (width, height)
        )
        camera_to_ego = _quaternion_transform(
            _decode_pickle_vector(row["translation"], expected_shape=(3,), name="translation"),
            _decode_pickle_vector(row["rotation"], expected_shape=(4,), name="rotation"),
        )
        cameras.append(
            {
                "logical_id": channel,
                "slot": slot,
                "image_size_wh": [width, height],
                "intrinsic": np.asarray(rectified, dtype=float).tolist(),
                "camera_to_reference": camera_to_ego.tolist(),
            }
        )

    profile = {
        "dataset_name": "nuplan/nuplan-v1.1",
        "source": f"camera table of nuPlan v1.1 log {args.log_db.stem} ({vehicle})",
        "reference_frame": (
            "nuPlan ego frame (ego_pose). The AlpaSim rig frame of the rendered "
            "world must match it; cameras are rectified pinhole."
        ),
        # nuPlan shards pack the front camera at 1024 px (front_camera_tile).
        "native_front_camera_tile": True,
        # data_parsing/nuplan/packing.py _state_signals.
        "egomotion": {
            "curvature_speed_floor_mps": 0.5,
            "curvature_zero_below_mps": 0.5,
            "curvature_limit": 0.5,
        },
        "cameras": cameras,
    }
    text = json.dumps(profile, indent=2)
    # One line per matrix row or size pair keeps the matrices readable.
    text = re.sub(
        r"\[\s+([-0-9.e, \n]+?)\s+\]",
        lambda match: "[" + ", ".join(match.group(1).replace(",", " ").split()) + "]",
        text,
    )
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
