# Data parsing recipe for [NVIDIA Physical AI Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

Parser for the NVIDIA Physical AI dataset. It is a pre-extraction source: frames are returned raw, the shard packer resizes them and the pre-extracted loader normalizes them.

Download (a subset of) the dataset into `data`.

## Model inputs produced

- `visual_tiles` `(7, 3, H, W)` uint8 — 7 raw camera frames at native resolution
- `map_tile` `(3, H, W)` uint8 — zero tile for the separate map branch (the dataset has no rendered map)
- `visual_history` `(896,)` — 64 frames × 14-dim compressed scene memory (zeros)
- `egomotion_history` `(256,)` — 64 past timesteps × 4 signals at 10 Hz
- `trajectory_target` `(128,)` — 64 future timesteps × 2 signals (supervision target)

### Egomotion history signals `(256,) = 64 × 4`

- `[0]` Speed (m/s) — `sqrt(vx^2 + vy^2)`
- `[1]` Acceleration (m/s^2) — `ax`
- `[2]` Yaw rate (rad/s) — differenced unwrapped ZYX yaw from `EgomotionState.pose`
- `[3]` Curvature (rad/m) — `curvature`

### Trajectory target signals `(128,) = 64 × 2`

- `[0]` Acceleration (m/s^2) — `ax`
- `[1]` Curvature (rad/m) — `curvature`

## Sampling

A sample is defined by a `sample_idx` — an index into the 10 Hz downsampled egomotion sequence. A `sample_idx` is valid when there are at least 64 rows behind it (history window) and 64 rows ahead (target window). For a typical 20s clip (~200 rows at 10 Hz) this gives ~72 valid sample points per clip.

All valid `(clip_uuid, sample_idx)` pairs are enumerated at dataset construction time. `__getitem__` does I/O only — no index arithmetic at call time.

## Usage

```python
from data_parsing.nvidia_physical_ai import NvidiaAVDataset
from torch.utils.data import DataLoader

dataset = NvidiaAVDataset(data_root="/path/to/nvidia_physical_ai_dataset")

# Single clip for smoke tests
dataset = NvidiaAVDataset(
    data_root="/path/to/nvidia_physical_ai_dataset",
    clip_uuids=["fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"],
)

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    visual_tiles      = batch["visual_tiles"]       # (B, 7, 3, H, W) uint8
    egomotion_history = batch["egomotion_history"]  # (B, 256)
    trajectory_target = batch["trajectory_target"]  # (B, 128)
```

## Additional dependencies

```
physical_ai_av   # NVIDIA dataset SDK (pip install git+https://github.com/NVlabs/physical_ai_av.git); requires python >= 3.11
pandas
```