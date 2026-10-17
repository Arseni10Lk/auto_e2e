# Data parsing recipe for [yaak-ai/L2D](https://huggingface.co/datasets/yaak-ai/L2D)

Parser for the L2D LeRobot dataset (v3.0). It is a pre-extraction source: frames are returned raw, the shard packer resizes them and the pre-extracted loader normalizes them.

## Dataset overview

- 100,000 episodes at 10 FPS
- 6 surround camera views and 1 separate BEV map (640×360)
- Vehicle state: float32 shape[8] — speed, heading, heading_error, lat, lon, alt, accel_x, accel_y
- Waypoints: float32 shape[10,2] — GPS lon/lat
- Actions: float32 shape[3] — gas, brake, steering

## Camera calibration approximation

L2D publishes six-camera inter-camera RDF extrinsics but no vehicle pose for
the reference camera, distortion coefficients, or verified rectification
calibration. The parser pins the official extrinsics and assumes the reference
camera is at the ego origin with zero mounting rotation.

The vendor horizontal and vertical FOV pairs cannot both describe one
square-pixel pinhole at the 1920×1080 stream aspect ratio. The production
approximation therefore derives one `fx = fy` source focal length from the
published vertical FOV, preserving the image-row angles used for ground-plane
projection, then applies the independent horizontal and vertical resize scales
for the packed square image. The unused horizontal-FOV discrepancy and focal
ratio are recorded per lens in rig provenance. This remains an audited
approximation, not measured camera calibration.

## Model inputs produced

- `visual_tiles` `(6, 3, H, W)` — 6 camera views
- `map_tile` `(3, H, W)` — BEV map for the separate navigation branch
- `egomotion_history` `(256,)` — 64 past timesteps × 4 signals at 10 Hz
- `visual_history` `(896,)` — zero-initialised; populated during sequential inference
- `trajectory_target` `(128,)` — 64 future timesteps × 2 signals (supervision target)

### Egomotion history signals `(256,) = 64 × 4`

| Index | Signal | Source |
|-------|--------|--------|
| 0 | Speed (m/s) | `observation.state.vehicle[0]` (raw km/h) ÷ 3.6 |
| 1 | Acceleration_x (m/s²) | `observation.state.vehicle[6]` directly |
| 2 | Yaw rate (rad/s) | `diff(radians(heading)) / dt` |
| 3 | Curvature (rad/m) | `yaw_rate / speed` (guarded) |

> Units: the raw L2D `speed` is km/h and `heading` is degrees (per
> `meta/stats.json`). Both are converted to SI in `_derive_signals`.

### Trajectory target signals `(128,) = 64 × 2`

| Index | Signal | Source |
|-------|--------|--------|
| 0 | Acceleration_x (m/s²) | From future frames |
| 1 | Curvature (rad/m) | From future frames |

## Camera views

| Key | Description |
|-----|-------------|
| `observation.images.front_left` | Front left camera |
| `observation.images.left_forward` | Left forward camera |
| `observation.images.right_forward` | Right forward camera |
| `observation.images.left_backward` | Left backward camera |
| `observation.images.rear` | Rear camera |
| `observation.images.right_backward` | Right backward camera |
| `observation.images.map` | BEV map (640×360) |

## Usage

```python
from data_parsing.l2d import L2DDataset
from torch.utils.data import DataLoader

dataset = L2DDataset(repo_id="yaak-ai/L2D")

# Single episode for smoke tests
dataset = L2DDataset(repo_id="yaak-ai/L2D", episodes=[0])

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    visual_tiles      = batch["visual_tiles"]       # (B, 6, 3, H, W)
    map_tile          = batch["map_tile"]           # (B, 3, H, W)
    egomotion_history = batch["egomotion_history"]  # (B, 256)
    trajectory_target = batch["trajectory_target"]  # (B, 128)
```

## Additional dependencies

```
lerobot==0.5.1            # LeRobot dataset SDK (pip install lerobot==0.5.1)
```
