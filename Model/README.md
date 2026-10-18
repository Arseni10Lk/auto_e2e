# AutoE2E Architecture

The v1.0 model is the Reactive branch of `AutoE2E`
(`model_components/auto_e2e.py`). The training configuration is fixed by
`reactive_model_kwargs` in `training/reactive_multitask.py`:

| Component | v1.0 setting |
|-----------|--------------|
| Camera BEV encoder | `bevformer_v2_t8` (BEVFormer V2, current frame plus 7 history frames at 500 ms) |
| Map encoder | `semantic_raster` map plus a separate route encoder (`SplitNavigationEncoder`) |
| Map BEV fusion | `deformable` cross-attention |
| Temporal memory | `no_memory` |
| Trajectory planner | `gru` |
| World Model / reasoning branch | off (`enable_world_model=False`, `enable_reasoning=False`) |
| Auxiliary heads | BEV segmentation (8 classes) and route reconstruction |

## Inputs and Predictions
**AutoE2E consumes as input:**
- Surround-view camera images at 512x512, plus an optional native 1024x1024
  front camera image (`reactive_training_contracts.py`)
- Seven historical camera frames for the T8 temporal BEV, either as raw tiles or
  through a `StatefulCameraFPNCache` at inference time
- A semantic navigation raster: 14 static map channels and 2 route channels
  (selected corridor and destination), on a 450x300 grid at 0.4 m/px
  (`navigation/geometry.py`)
- Egomotion history (`(256,)` = 64 timesteps x `[speed, acceleration, yaw_rate, curvature]`)
- Visual history (`(896,)`), consumed by the planner

**AutoE2E outputs a prediction of:**
- Future driving trajectory, modelled as acceleration and curvature over a 6.4 s
  horizon at 10 Hz (`(B, 64 * 2)`)

**During training, and for model introspection, AutoE2E also predicts:**
- BEV semantic segmentation logits from the image BEV
- Route reconstruction logits from the route contribution

The optional World Model branch (`enable_world_model=True`) predicts future
visual features for a JEPA-style feature reconstruction loss. It is not part of
the v1.0 configuration.

**Forward signature:**
```python
trajectory = model(
    camera_tiles,        # (B, V, 3, H, W)
    map_context,         # (B, C_map, H_map, W_map) — semantic BEV map
    visual_history,      # (B, 896)
    egomotion_history,   # (B, 256)
    route_mask=None,     # (B, 2, H_map, W_map) — route corridor and destination
    projection=None,     # CameraProjectionModel operator — the geometry ABI
    geometry_type=None,  # "pinhole" | "rectified_pinhole" | "ftheta" | "pseudo"
    mode="infer",
)
```

`forward` returns the trajectory `(B, num_timesteps * num_signals)`, or
`(trajectory, aux_outputs)` when auxiliary outputs are produced (`mode="train"`
or `return_auxiliary=True`). `aux_outputs` is a dict keyed by output name, for
example `"bev_segmentation_logits"` and `"route_reconstruction_logits"`.

**Camera geometry is an operator, not a matrix.** There is no `camera_params`
argument; a projection operator carries the calibration so that fisheye
(F-Theta) calibration is expressible and the geometry travels with the data.

```python
from model_components.view_fusion.projection import PinholeProjection

trajectory = model(
    camera_tiles, map_context, visual_history, egomotion_history,
    projection=PinholeProjection(camera_params),  # (B, V, 3, 4) intrinsic @ extrinsic
    geometry_type="pinhole", mode="infer",
)
```

Passing no `projection` falls back to `geometry_type="pseudo"`, a learned
spatial prior rather than real geometry (shape-testing and ablation only).
Passing a matrix-style argument such as `camera_params=` raises `TypeError`
instead of being absorbed by `**kwargs`.

**To learn the driving policy:**
- Imitation learning on the trajectory: predicted controls are integrated to ego
  XY and compared with the expert path. The objective in
  `training/reactive_multitask.py` adds a weighted route reconstruction loss and,
  in the `nuplan_full` stage only, the BEV segmentation loss.
