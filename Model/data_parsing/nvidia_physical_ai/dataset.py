"""PyTorch Dataset for the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

Usage
-----
    from data_parsing.nvidia_physical_ai import NvidiaAVDataset

    # All valid samples across all clips (for training)
    dataset = NvidiaAVDataset(data_root="/path/to/nvidia_av_camera_subset")

    # Single clip (for smoke tests / forward pass validation)
    dataset = NvidiaAVDataset(
        data_root="/path/to/nvidia_av_camera_subset",
        clip_uuids=["fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"],
    )

    sample = dataset[0]
    # sample["visual_tiles"]       (7, 3, H, W) uint8  7 raw cameras (native res)
    # sample["map_tile"]           (3, H, W) uint8     nav-map (zeros; map branch)
    # sample["visual_history"]     (896,)
    # sample["egomotion_history"]  (256,)
    # sample["trajectory_target"]  (128,)
    # sample["clip_uuid"]          str
    # sample["sample_idx"]         int
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_processing.contract_versions import UID_SCHEMA_VERSION

from .camera import CAMERA_NAMES, load_camera_frame, make_map_tile
from .egomotion import (
    _EGOMOTION_COLUMNS,
    MIN_ROWS,
    _DOWNSAMPLE_STEP,
    _FUTURE_TIMESTEPS,
    _HISTORY_TIMESTEPS,
    load_egomotion,
)

logger = logging.getLogger(__name__)

_DISCOVERY_CAMERA = "camera_front_wide_120fov"

# 64 frames x 14-dim compressed scene memory; matches the trajectory
# planner's visual_history_proj input dim. Emitted as zeros.
_VISUAL_HISTORY_DIM = 896


class ClipSample(TypedDict):
    visual_tiles: torch.Tensor        # (7, 3, H, W) uint8, 7 raw cameras (native)
    map_tile: torch.Tensor            # (3, H, W) uint8, nav-map (zeros; map branch)
    visual_history: torch.Tensor      # (896,)
    egomotion_history: torch.Tensor   # (256,)
    trajectory_target: torch.Tensor   # (128,)
    clip_uuid: str
    sample_idx: int


class NvidiaAVDataset(Dataset):
    """Dataset where each item is one valid (clip_uuid, sample_idx) pair.

    All valid sample indices across all clips are enumerated at construction
    time, so __getitem__ does only I/O.

    Args:
        data_root: Path to the subset directory.
        camera_names: Camera views to load. Defaults to ``CAMERA_NAMES``.
        clip_uuids: Optional explicit list of clip UUIDs. If ``None``, all
            valid clips are discovered automatically. Pass a single-element
            list for smoke tests or forward pass validation.
    """

    def __init__(
        self,
        data_root: Path | str,
        camera_names: list[str] | None = None,
        clip_uuids: list[str] | None = None,
        reasoning_clip_only: bool = False,
        wm_num_frames: int = 4,
        wm_stride: int = 10,
    ) -> None:
        self.data_root = Path(data_root)
        self.camera_names = camera_names or CAMERA_NAMES
        # Reasoning-clip mode: get_front_clip(idx) decodes only the front camera
        # at the reasoning horizons (current + wm_num_frames future 1 Hz steps).
        # wm_stride=10 turns the 10 Hz downsampled egomotion index into 1 Hz.
        self._reasoning_clip_only = reasoning_clip_only
        self._wm_num_frames = wm_num_frames
        self._wm_stride = wm_stride
        self._front_cam = (camera_names or CAMERA_NAMES)[0]

        # __getitem__ returns raw uint8 frames; the shard packer owns the resize
        # and the pre-extracted loader owns the normalize.

        clips = clip_uuids if clip_uuids is not None else self._discover_clip_uuids()
        if not clips:
            raise ValueError(
                f"No valid clips found under: {self.data_root / 'camera' / _DISCOVERY_CAMERA}"
            )
        
        self._egomotion_dfs: dict[str, pd.DataFrame] = {}
        self._camera_timestamps: dict[tuple[str, str], np.ndarray] = {}

        # Build the flat sample index: list of (clip_uuid, sample_idx, egomotion_timestamp_us).
        # Precomputing this means __getitem__ never touches pandas.
        self._samples: list[tuple[str, int, int]] = []
        for clip_uuid in clips:
            samples = self._valid_samples_for_clip(clip_uuid)
            if samples:
                self._samples.extend(samples)
                self._load_camera_timestamps(clip_uuid)

        if not self._samples:
            raise ValueError("No valid samples found across all clips.")

        logger.info(
            "NvidiaAVDataset: %d samples from %d clips", len(self._samples), len(clips)
        )

    def _load_camera_timestamps(self, clip_uuid: str) -> None:
        """Load and cache camera timestamp arrays for all cameras in a clip."""
        for cam_name in self.camera_names:
            timestamps_path = (
                self.data_root / "camera" / cam_name
                / f"{clip_uuid}.{cam_name}.timestamps.parquet"
            )
 
            self._camera_timestamps[(clip_uuid, cam_name)] = (
                pd.read_parquet(timestamps_path)["timestamp"].to_numpy()
            )

    def _discover_clip_uuids(self) -> list[str]:
        """Scan the reference camera directory for clip UUIDs."""
        discovery_dir = self.data_root / "camera" / _DISCOVERY_CAMERA
        if not discovery_dir.exists():
            raise FileNotFoundError(
                f"Reference camera directory not found: {discovery_dir}"
            )
        return sorted(p.name.split(".")[0] for p in discovery_dir.glob("*.mp4"))

    def _validate_clip(self, clip_uuid: str) -> bool:
        """Return True if every camera video and timestamp parquet exists."""
        for cam_name in self.camera_names:
            cam_dir = self.data_root / "camera" / cam_name
            video_path = cam_dir / f"{clip_uuid}.{cam_name}.mp4"
            timestamps_path = cam_dir / f"{clip_uuid}.{cam_name}.timestamps.parquet"

            if not video_path.exists():
                logger.warning(
                    "Clip %s: missing video file for camera %s. Skipping.",
                    clip_uuid, cam_name,
                )
                return False

            if not timestamps_path.exists():
                logger.warning(
                    "Clip %s: missing timestamps parquet for camera %s. Skipping.",
                    clip_uuid, cam_name,
                )
                return False

        return True


    def _validate_egomotion_timestamps(self, clip_uuid: str, df_ds: pd.DataFrame) -> bool:
        """Check that downsampled egomotion timestamps are strictly monotonically increasing."""
        timestamps = df_ds["timestamp"].to_numpy()
        if not np.all(np.diff(timestamps) > 0):
            logger.warning(
                "Clip %s: egomotion timestamps are not strictly monotonically increasing. Skipping.",
                clip_uuid,
            )
            return False
        return True

    def _valid_samples_for_clip(
        self, clip_uuid: str
    ) -> list[tuple[str, int, int]]:
        """Return all valid (clip_uuid, sample_idx, egomotion_timestamp_us) for one clip.

        A sample_idx is valid when there are _HISTORY_TIMESTEPS rows behind it
        and _FUTURE_TIMESTEPS rows ahead of it in the downsampled sequence.
        """

        if not self._validate_clip(clip_uuid):
            return []
        
        parquet_path = (
            self.data_root / "labels" / "egomotion" / f"{clip_uuid}.egomotion.parquet"
        )
        if not parquet_path.exists():
            logger.warning("Egomotion parquet missing for clip %s, skipping.", clip_uuid)
            return []

        try:
            df = pd.read_parquet(parquet_path, columns=_EGOMOTION_COLUMNS)
        except KeyError as e:
            logger.warning(
                "Clip %s: egomotion parquet missing columns: %s. Skipping.", clip_uuid, e
            )
            return []
        except Exception as e:
            logger.warning(
                "Clip %s: failed to read egomotion parquet: %s. Skipping.", clip_uuid, e
            )
            return []
        
        df_ds = df.iloc[::_DOWNSAMPLE_STEP].reset_index(drop=True)

        if len(df_ds) < MIN_ROWS:
            logger.warning(
                "Clip %s has only %d rows after downsampling (need %d), skipping.",
                clip_uuid, len(df_ds), MIN_ROWS,
            )
            return []
        
        if not self._validate_egomotion_timestamps(clip_uuid, df_ds):
            return []
        
        self._egomotion_dfs[clip_uuid] = df_ds

        min_idx = _HISTORY_TIMESTEPS
        max_idx = len(df_ds) - _FUTURE_TIMESTEPS  - 1  # inclusive

        return [
            (clip_uuid, sample_idx, int(df_ds.iloc[sample_idx]["timestamp"]))
            for sample_idx in range(min_idx, max_idx + 1)
        ]

    def projection_spec(self, image_size: int = 256) -> dict | None:
        """Build a native f-theta projection spec from saved calibration.

        Reads calibration/{intrinsics,extrinsics}.pkl (saved at ingest) and
        serializes an FThetaProjection in the native pixel frame. ``image_size``
        is accepted for API symmetry only: f-theta normalizes by the native
        (W, H), which is exact under any resize. Returns None when calibration
        is absent, so the caller uses the explicit pseudo path.
        """
        import pickle

        calib_dir = self.data_root / "calibration"
        intr_path = calib_dir / "intrinsics.pkl"
        extr_path = calib_dir / "extrinsics.pkl"
        if not (intr_path.exists() and extr_path.exists()):
            # Warn so a pseudo-geometry run is never mistaken for f-theta.
            logger.warning(
                "NvidiaAVDataset.projection_spec: no calibration at %s "
                "(intrinsics.pkl / extrinsics.pkl); using pseudo geometry. Run "
                "data_ingest with camera_intrinsics/sensor_extrinsics for real "
                "f-theta projection (#77).", calib_dir,
            )
            return None

        from .calibration import build_ftheta_projection

        with open(intr_path, "rb") as f:
            intrinsics = pickle.load(f)
        with open(extr_path, "rb") as f:
            extrinsics = pickle.load(f)

        proj = build_ftheta_projection(intrinsics, extrinsics, self.camera_names)
        return proj.to_spec()

    def get_front_clip(self, idx: int) -> list[torch.Tensor]:
        """Front-camera clip at the reasoning horizons for sample ``idx``.

        Only valid with ``reasoning_clip_only=True``. Decodes only the front
        camera at the current and future 1 Hz egomotion timestamps; ``idx``
        matches ``__getitem__`` indexing.
        """
        if not self._reasoning_clip_only:
            raise RuntimeError(
                "get_front_clip requires NvidiaAVDataset(reasoning_clip_only=True).")
        from .camera import load_front_clip

        clip_uuid, sample_idx, _ts = self._samples[idx]
        df_ds = self._egomotion_dfs[clip_uuid]
        n_horizons = self._wm_num_frames + 1  # current + N future
        # The valid-sample margins keep sample_idx + N*stride in range.
        rows = [min(sample_idx + h * self._wm_stride, len(df_ds) - 1)
                for h in range(n_horizons)]
        ts_us = [int(df_ds.iloc[r]["timestamp"]) for r in rows]
        cam_ts = self._camera_timestamps.get((clip_uuid, self._front_cam))
        return load_front_clip(
            self.data_root, clip_uuid, ts_us,
            front_cam=self._front_cam, camera_timestamps_us=cam_ts)

    def sample_uid(self, idx: int) -> str:
        """Global, partition-independent sample id.

        Built from (clip_uuid, sample_idx); contains no `.` or `/`, so it is safe
        as a WebDataset ``__key__``.
        """
        clip_uuid, sample_idx, _ts = self._samples[idx]
        return f"nv-{UID_SCHEMA_VERSION}-{clip_uuid}-f{sample_idx:06d}"

    def split_group_uid(self, idx: int) -> str:
        """Train/val split unit: the whole clip, since its frames are correlated."""
        clip_uuid, _sample_idx, _ts = self._samples[idx]
        return f"nv-{clip_uuid}"

    def frame_index(self, idx: int) -> int:
        """Clip-local sample index, used to pick the 1 Hz reasoning-label subset."""
        _clip_uuid, sample_idx, _ts = self._samples[idx]
        return sample_idx

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ClipSample:
        clip_uuid, sample_idx, egomotion_timestamp_us = self._samples[idx]

        camera_timestamps = {
            cam_name: self._camera_timestamps[(clip_uuid, cam_name)]
            for cam_name in self.camera_names
        }

        visual_tiles = load_camera_frame(
            self.data_root,
            clip_uuid,
            egomotion_timestamp_us=egomotion_timestamp_us,
            camera_names=self.camera_names,
            camera_timestamps=camera_timestamps,
        )

        egomotion_history, trajectory_target = load_egomotion(
            self.data_root,
            clip_uuid,
            sample_idx=sample_idx,
            df=self._egomotion_dfs[clip_uuid],
        )

        # NVIDIA has no rendered nav-map; the map branch gets a zero tile shaped
        # like one camera frame. Kept separate from visual_tiles so it never
        # enters the camera BEV projection.
        map_tile = make_map_tile(visual_tiles[0])

        return ClipSample(
            visual_tiles=visual_tiles,
            map_tile=map_tile,
            visual_history=torch.zeros(_VISUAL_HISTORY_DIM),
            egomotion_history=egomotion_history,
            trajectory_target=trajectory_target,
            clip_uuid=clip_uuid,
            sample_idx=sample_idx,
        )
