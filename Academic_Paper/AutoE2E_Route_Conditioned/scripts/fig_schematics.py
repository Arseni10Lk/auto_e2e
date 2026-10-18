# ruff: noqa
"""Schematic figures 2, 3, 6, 7, 8, 14, 15."""
import sys
sys.path.insert(0, "/tmp/autoe2e_paper/figgen")
from svgkit import SVG, MONO

OUT = "/tmp/autoe2e_paper/vivlio/figures/"

# ======================= Fig 2: temporal contract =======================
W,H=900,360
s=SVG(W,H,font_size=10.5)
s.text(18,22,"Temporal contract: camera history, egomotion, control output, and benchmark targets",size=11,weight="bold")
x0,x1=190,870
def tx(t): return x0+(t+6.4)/12.8*(x1-x0)
# now line
s.line(tx(0),35,tx(0),312,color="#d62728",dash="4,3",width=1.2)
s.text(tx(0)+6,48,"now (t = 0)",size=9,color="#d62728")
# Camera row
s.text(18,82,"Camera (T8)",size=10.5,weight="bold")
for t in [-3.5,-3,-2.5,-2,-1.5,-1,-.5,0]:
    cur=t==0; x=tx(t)
    s.add(f'<rect x="{x-7}" y="58" width="14" height="32" rx="2" fill="{"#FBE5E3" if cur else "#E3EEF9"}" stroke="{"#d62728" if cur else "#3F6FAE"}" stroke-width="1.2"/>')
    s.text(x,103,"0" if cur else f"{t:.1f}",size=7.5,anchor="middle",color="#d62728" if cur else "#3F6FAE")
s.text(x0,50,"7 history frames x 6 views @ 512 px; detached",size=8.6,color="#3F6FAE")
s.text(tx(0)+18,72,"6 views @ 512 px",size=8.6,color="#d62728")
s.text(tx(0)+18,84,"+ front @ 1024 px (additive)",size=8.6,color="#d62728")
s.text(tx(0)+18,96,"3.5 s visual span at 2 Hz",size=8.6,color="#333")
# Egomotion row
s.text(18,153,"Egomotion",size=10.5,weight="bold")
s.add(f'<rect x="{tx(-6.4)}" y="126" width="{tx(-4)-tx(-6.4)}" height="34" rx="4" fill="#F5F5F5" stroke="#999"/>')
s.text((tx(-6.4)+tx(-4))/2,147,"zero pad in benchmark",size=8.4,anchor="middle",color="#666")
s.add(f'<rect x="{tx(-4)}" y="126" width="{tx(0)-tx(-4)}" height="34" rx="4" fill="#E3EEF9" stroke="#3F6FAE"/>')
s.text((tx(-4)+tx(0))/2,147,"40 real steps (4.0 s) in benchmark",size=8.6,anchor="middle",weight="bold")
s.text(x0,119,"64 x [speed, acceleration, yaw rate, curvature] @ 10 Hz; training uses all 6.4 s",size=8.5)
# Planner output row
s.text(18,219,"Planner output",size=10.5,weight="bold")
s.add(f'<rect x="{tx(0)}" y="191" width="{tx(6.4)-tx(0)}" height="34" rx="4" fill="#E2F3E4" stroke="#2E8B57"/>')
s.text((tx(0)+tx(6.4))/2,212,"64 x (acceleration, curvature) @ 10 Hz -> rollout -> 64 XY points",size=8.6,anchor="middle",weight="bold")
# Target row
s.text(18,276,"Targets",size=10.5,weight="bold")
s.add(f'<rect x="{tx(0)}" y="248" width="{tx(5)-tx(0)}" height="34" rx="4" fill="#FCF4DB" stroke="#B8860B"/>')
s.text((tx(0)+tx(5))/2,269,"50 valid steps: ADE/FDE @ 1, 2, 3, 5 s; |lateral| / |longitudinal|",size=8.4,anchor="middle",weight="bold")
s.add(f'<rect x="{tx(5)}" y="248" width="{tx(6.4)-tx(5)}" height="34" rx="4" fill="#F5F5F5" stroke="#999"/>')
s.text((tx(5)+tx(6.4))/2,269,"no target",size=8.5,anchor="middle",color="#666")
s.text(tx(5)-4,241,"valid coverage 50/64 = 78.125%",size=8.5,anchor="end",color="#B8860B")
# Axis
ay=312;s.line(x0,ay,x1,ay,color="#333",width=1.1)
for t in range(-6,7):
    s.line(tx(t),ay-4,tx(t),ay+4,color="#333")
    s.text(tx(t),ay+17,"0" if t==0 else f"{t:+d}",size=8.5,anchor="middle")
s.text((x0+x1)/2,ay+36,"time relative to current frame [s]  (10 Hz base rate = 0.1 s per step)",size=9,anchor="middle")
s.text(W-12,22,"benchmark: 40 history / 50 future steps; training: 64 / 64",size=9.5,anchor="end",color="#555")
s.save(OUT + "fig02_temporal_contract_v2.svg")

# ======================= Fig 3: encoder layer detail =======================
W,H=900,510
s=SVG(W,H,font_size=10.5)
s.text(18,22,"BEVFormer V2 encoder layer (x6, frozen) and the current-frame high-resolution front residual",size=11,weight="bold")
# left chain
x,w=30,330
s.box(x,42,w,48,"BEV queries Q  (60,000 x 256)",["300 x 200 learned embeddings + row/column position (128+128)"],style="frozen")
s.box(x,112,w,82,"Temporal self-attention (T1 form)",["deformable; queue = {Q, Q} (current BEV duplicated)","8 heads x 4 points per queue; offsets from [Q, Q+pos]","value_proj / output_proj 256 x 256; 0.23 M"],style="frozen")
s.box(x,216,w,112,"Multi-scale spatial cross-attention",["4 pillar heights z in [-5, 3] m -> calibrated pinhole/fisheye projection","8 heads x 4 FPN levels x 8 points per visible view","visible-view average + level/camera embeddings; 0.33 M"],style="frozen")
s.box(x,350,w,58,"FFN 256 -> 512 -> 256",["ReLU, dropout 0.1; 0.26 M; three post-norm LayerNorms"],style="frozen")
s.box(x,430,w,44,"Layer-6 output B_enc (current BEV)",["also passed to T8 temporal fusion"],style="plain")
for a,b in [(90,112),(194,216),(328,350),(408,430)]: s.arrow(x+w/2,a,x+w/2,b)
for yy in [155,270,378]: s.text(x+w+10,yy,"LN",size=8.5,color="#555")
# FPN input
s.box(390,112,190,82,"Camera FPN features",["6 views x 4 levels x 256 ch","512 px base tiles; history pyramids cached"],style="input")
s.arrow(485,194,360,260,elbow=485,label="values",label_dy=-4)
# right front branch
rx,rw=610,260
s.box(rx,42,rw,62,"Native front tile 1024 x 1024 (t = 0)",["shared ResNet-50 + FPN; 4 levels, one view"],style="input")
s.box(rx,126,rw,112,"Front cross-attention (frozen copy of layer 6)",["reference points from the calibrated front-camera operator","content-only delta: no level/camera embedding affine terms","delta = LN6(Q + attn) - LN6(Q); masked to observed cells"],style="frozen")
s.box(rx,260,rw,62,"Gate tanh(g), g in R^256, zero-initialised",["only trainable camera-side parameter: 256 values"],style="trainable",badge="gate train")
s.box(rx,344,rw,64,"B_0 = B_enc + tanh(g) * delta",["at initialisation: B_0 = B_enc exactly (no-op)"],style="plain",mono_lines=("B_0 = B_enc + tanh(g) * delta",))
s.box(rx,430,rw,44,"T8 fusion input: B_0 + 7 detached history BEVs",[],style="frozen")
for a,b in [(104,126),(238,260),(322,344),(408,430)]: s.arrow(rx+rw/2,a,rx+rw/2,b)
# current BEV feeds front branch query / residual path
s.arrow(360,452,610,376,label="B_enc",label_dy=-4)
# visual styling legend
s.legend(30,500,[("input","input"),("frozen","frozen"),("trainable","trainable")])
s.text(570,500,"The 1024-pixel branch is used only for t=0; every historical front image remains 512 x 512.",size=8.5,anchor="middle",color="#555")
s.save(OUT + "fig03_encoder_layer_v2.svg")

# ======================= Fig 6: navigation encoders + deformable fusion =======================
s = SVG(760, 470, font_size=10.5)
s.text(14, 22, "Split navigation encoding and deformable navigation fusion (all trainable)", size=11, weight="bold")
# E_map stack
mx, mw = 20, 230
s.box(mx, 40, mw, 40, "M: 14 x 450 x 300", ["bilinear resample -> 14 x 300 x 200"], style="input")
s.box(mx, 92, mw, 62, "E_map stem", ["Conv 5x5, 14 -> 96, GroupNorm(16), SiLU", "local path: Conv 1x1, 96 -> 256"], style="trainable")
s.box(mx, 166, mw, 86, "E_map context", ["depthwise 3x3 s2 + Conv 1x1 96 -> 192, GN, SiLU",
      "depthwise 3x3 s2 + Conv 1x1 192 -> 192, GN, SiLU", "Conv 1x1 192 -> 256, GN; bilinear x4 upsample"], style="trainable")
s.box(mx, 264, mw, 40, "E_map(M) = GN+SiLU(local + context)", ["256 x 300 x 200; 0.167 M params"], style="trainable")
for ya, yb in ((80, 92), (154, 166), (252, 264)):
    s.arrow(mx + mw / 2, ya, mx + mw / 2, yb)
# E_route stack
rx, rw = 270, 230
s.box(rx, 40, rw, 40, "R: 2 x 450 x 300 (binary)", ["bilinear resample -> 2 x 300 x 200"], style="input")
s.box(rx, 92, rw, 62, "E_route stem", ["Conv 3x3, 2 -> 48, GN, SiLU", "Conv 3x3, 48 -> 96, GN, SiLU"], style="trainable")
s.box(rx, 166, rw, 86, "E_route context", ["depthwise 3x3 s2 (96) + Conv 1x1 96 -> 96, GN, SiLU",
      "bilinear x2 upsample; sum with stem output", "Conv 1x1 96 -> 256, GN, SiLU"], style="trainable")
s.box(rx, 264, rw, 40, "E_route(R)", ["256 x 300 x 200; 0.078 M params"], style="trainable")
for ya, yb in ((80, 92), (154, 166), (252, 264)):
    s.arrow(rx + rw / 2, ya, rx + rw / 2, yb)
# gate and sum
s.box(rx, 318, rw, 40, "x sigmoid(g)   g in R^256, zero-init -> 0.5", ["route_contribution (-> reconstruction head)"], style="trainable")
s.arrow(rx + rw / 2, 304, rx + rw / 2, 318)
sx, sy = 145, 340
s.add(f'<circle cx="{sx}" cy="{sy}" r="12" fill="white" stroke="#333" stroke-width="1.3"/>')
s.text(sx, sy + 4.5, "+", size=15, anchor="middle", weight="bold")
s.arrow(mx + mw / 2, 304, sx, sy - 12)
s.arrow(rx, 338, sx + 12, 340)
s.text(sx, 374, "B_nav: 256 x 300 x 200", size=10, anchor="middle", weight="bold", family=MONO)
s.text(sx, 388, "validity gates: M *= map_valid, R *= route_valid (sample level)", size=8.5, anchor="middle")
# deformable fusion
fx, fw = 528, 216
s.box(fx, 40, fw, 46, "B_img (query)  256 x 300 x 200", ["frozen camera path output"], style="frozen")
s.box(fx, 100, fw, 174, "Deformable cross-attention",
      ["q = LayerNorm(B_img cell)", "offset_proj: Linear 256 -> 2K (K = 8 points)", "attn_proj: Linear 256 -> 8 heads x K",
       "sample B_nav bilinearly at cell + offsets", "(grid_sample, border padding)", "per-head softmax over K, weighted sum",
       "out_proj: Linear 256 -> 256 (zero-init)", "residual add; then FFN 256->512->256", "(GELU, dropout 0.1, last Linear zero-init)"],
      style="trainable", line_size=8.8)
s.box(fx, 290, fw, 46, "B_fused = B_img + F(B_img, B_nav)", ["0.350 M params; exact identity at init"], style="trainable",
      mono_lines=("B_fused = B_img + F(B_img, B_nav)",))
s.arrow(fx + fw / 2, 86, fx + fw / 2, 100); s.arrow(fx + fw / 2, 274, fx + fw / 2, 290)
s.arrow(sx + 60, 370, fx, 200, elbow=490, label=None)
s.text(492, 300, "B_nav (values)", size=9)
s.text(fx + fw / 2, 372, "chunked over 4096 queries; O(N x K) instead of O(N^2)", size=8.5, anchor="middle")
s.text(fx + fw / 2, 386, "N = 60,000 BEV cells", size=8.5, anchor="middle")
s.box(fx, 400, fw, 56, "Route reconstruction head",
      ["Conv1x1 256->64, GN, SiLU, dw 3x3, SiLU, Conv1x1 64->2", "bilinear to 450 x 300; 0.017 M"], style="aux")
s.arrow(rx + rw, 338, fx, 428, elbow=505, color="#B8860B")
s.legend(20, 462, [("input", "input"), ("frozen", "frozen"), ("trainable", "trainable"), ("aux", "auxiliary")])
s.save(OUT + "fig06_navigation_fusion.svg")

# ======================= Fig 7: GRU planner =======================
s = SVG(760, 330, font_size=10.5)
s.text(14, 22, "Deterministic GRU control planner (0.835 M params) and unicycle rollout", size=11, weight="bold")
s.box(20, 40, 200, 70, "Context c_0 (256-d)", ["P_ego(h_ego): Linear 256 -> 256", "+ P_vis(h_vis): Linear 896 -> 256",
      "h_vis = 0 in evaluated checkpoints"], style="input")
s.box(20, 126, 200, 56, "Values V = value_proj(B_fused)", ["Linear 256 -> 256 per cell", "256 x 300 x 200"], style="trainable")
s.box(20, 198, 200, 54, "Learned ego query e (256)", ["nn.Embedding(1, 256)"], style="trainable")
# unrolled cells
cx0 = 260
for i, lab in enumerate(["t = 1", "t = 2", "...", "t = 64"]):
    cx = cx0 + i * 122
    if lab == "...":
        s.text(cx + 50, 150, "...", size=16, anchor="middle")
        continue
    s.box(cx, 60, 108, 150, f"step {lab}",
          ["q = detach(h) + e", "ref = sigmoid(W_r q)", "off = W_o q (16 x 2)", "loc = ref + 0.1 off",
           "a = sum_k w_k V[loc_k]", "a = W_out a", "h = GRU(a, h)", "u = W_u h (2)"], style="trainable", line_size=8.5)
s.arrow(220, 75, cx0, 75, label="h_0 = c_0", label_dy=-4)
s.arrow(cx0 + 108, 140, cx0 + 122, 140, label="h", label_dy=-4)
s.arrow(cx0 + 122 * 3 - 14, 140, cx0 + 122 * 3, 140, label="h", label_dy=-4)
s.arrow(220, 154, 240, 154); s.text(232, 168, "V shared by all steps (bilinear grid_sample, 16 points, zero padding)", size=8.5)
s.arrow(220, 225, 240, 225); s.text(232, 240, "e shared by all steps", size=8.5)
s.box(260, 232, 474, 40, "u_1..u_64 = (a_t, kappa_t) @ 10 Hz -> semi-implicit unicycle rollout (dt = 0.1 s, float32)",
      ["v_t = max(v_{t-1} + a_t dt, 0);  theta_t = sum v_i kappa_i dt;  x_t = sum v_i cos(theta_i) dt;  y_t = sum v_i sin(theta_i) dt"],
      style="output", line_size=8.8, mono_lines=("v_t = max(v_{t-1} + a_t dt, 0);  theta_t = sum v_i kappa_i dt;  x_t = sum v_i cos(theta_i) dt;  y_t = sum v_i sin(theta_i) dt",))
for i in (0, 1, 3):
    s.arrow(cx0 + i * 122 + 54, 210, cx0 + i * 122 + 54, 232)
s.text(20, 296, "The recurrent state is detached before the deformable lookup, so BEV-lookup gradients do not flow through the recurrence;",
       size=9)
s.text(20, 310, "the same 16-point lookup is applied at every step. Sampling locations are clamped to the BEV extent.", size=9)
s.save(OUT + "fig07_gru_planner.svg")

# ======================= Fig 8: lineage + evaluation protocol =======================
s = SVG(780, 470, font_size=10.5)
s.text(14, 22, "Checkpoint lineage (top) and the two evaluation protocols (bottom)", size=11, weight="bold")
s.box(14, 40, 176, 76, "Official BEVFormer V2 R50 T8",
      ["nuScenes-trained (epoch 24)", "BEV queries/pos. embeddings resized", "to 300 x 200; detector heads dropped"], style="frozen")
s.box(214, 40, 176, 76, "nuPlan trajectory stage",
      ["8 GPU workers, seed 149, LR 1e-4", "camera BEV frozen; traj w=1, route w=1", "BEV seg w=0; 1,024 internal val samples"],
      style="trainable")
s.box(414, 40, 156, 76, "nuPlan Epoch 4 / Epoch 5",
      ["ed00e072471a / ca8b43d7a777", "Epoch 5 = fine-tuning parent", "(frozen Epoch-5 parent profile)"], style="plain")
s.box(594, 40, 172, 76, "KITScenes fine-tuning",
      ["8 workers, seed 149, LR 3e-5", "38,847 samples / 364 scenes", "frozen val 3,820 / 40 scenes; BF16"], style="trainable")
s.arrow(190, 78, 214, 78); s.arrow(390, 78, 414, 78); s.arrow(570, 78, 594, 78)
s.box(594, 132, 172, 52, "KITScenes Epoch 5 / Epoch 7",
      ["120a21639d97 / a1e6b1621018", "Ep. 5 marked best retained checkpoint"], style="plain")
s.arrow(680, 116, 680, 132)
s.text(214, 130, "Optional BEV-only Stage A (unfrozen camera BEV, BEV seg. loss) exists in the code path;", size=8.5, color="#555")
s.text(214, 142, "its use as parent of the reported nuPlan checkpoints is not recorded in the available metadata.", size=8.5, color="#555")
s.text(14, 130, "all four checkpoints:", size=8.5, color="#555"); s.text(14, 142, "eval_gate_pass = false", size=8.5, color="#555")

# bottom: protocols
y0 = 206
s.text(14, y0 - 6, "Protocol A: checkpoint-internal validation (model executed)", size=10.5, weight="bold")
s.box(14, y0, 236, 96, "Frozen internal validation splits",
      ["nuPlan: 1,024 samples", "KITScenes: 3,820 samples / 40 scenes", "Camera + Map + Route (+ counterfactuals)",
       "BF16 autocast"], style="input")
s.box(270, y0, 236, 96, "Metrics @ 6.4 s (64 steps)",
      ["ADE/FDE 6.4 s; route reconstruction IoU", "route corridor compliance/success",
       "route progress proxy; drivable compliance", "comfort (nuPlan thresholds)"], style="aux")
s.arrow(250, y0 + 48, 270, y0 + 48)
s.text(14, y0 + 112, "Not comparable with Protocol B: different samples, horizon, and metric definitions.", size=8.5, color="#B8443F")

y1 = 340
s.text(14, y1 - 6, "Protocol B: external deterministic overlay replay (model NOT re-executed)", size=10.5, weight="bold")
s.box(14, y1, 236, 96, "KITScenes Val v3.5 (primary)",
      ["117 scenes, 11,035 samples (strict identity)", "inputs: Camera + HD Map + Route",
       "Dashboard shows 140 shards / 13,525 samples"], style="input")
s.box(270, y1, 236, 96, "KITScenes Test v1.0 (camera-only track)",
      ["206 scenes, 23,690 samples", "Map and Route unavailable by construction",
       "different scene population from Val"], style="input")
s.box(526, y1, 240, 96, "Replay of published control overlays",
      ["64 x (a, kappa) + v0 per sample -> rollout", "ADE/FDE @ 1, 2, 3, 5 s; |lat|, |lon| error",
       "50 valid steps (78.125 %); 0 nonfinite", "no counterfactuals / gradients / route logits"], style="aux")
s.arrow(250, y1 + 48, 270, y1 + 48, color="#8A8A8A", marker="arrow-gray", dash="3,3")
s.arrow(506, y1 + 48, 526, y1 + 48)
s.text(260, y1 + 112, "Val and Test are different scene populations: Test minus Val is not a Map/Route effect.", size=8.5, color="#B8443F")
s.legend(526, y0 + 130, [("frozen", "pretrained init"), ("trainable", "training stage"), ("input", "data"), ("aux", "metrics")])
s.save(OUT + "fig08_lineage_protocol.svg")

# ======================= Fig 14: route provenance =======================
s = SVG(760, 250, font_size=10.5)
s.text(14, 22, "How the KITScenes HD Map and Route rasters are produced (scene-level, a-posteriori route)", size=11, weight="bold")
s.box(14, 40, 170, 90, "Lanelet2 map (map.osm)",
      ["lanelet polygons -> drivable / intersection", "left/right bounds -> lane boundary", "centerlines -> centerline + direction",
       "crosswalk subtype; stop lines; traffic lights"], style="input", line_size=8.6)
s.box(204, 40, 170, 90, "Recorded ego trace (whole scene)",
      ["10 Hz poses (x, y, yaw)", "used ONLY for route matching", "no future ego points are rasterised"], style="input", line_size=8.6)
s.box(394, 40, 170, 90, "Lanelet2 trace matcher",
      ["candidate radius 8 m, dist. sigma 2 m", "heading sigma 20 deg; transition costs", "same 0 / following 0.25 / adjacent 1 /",
       "disconnected 25; quality thresholds"], style="plain", line_size=8.6)
s.box(584, 40, 162, 90, "Route = driven lane sequence",
      ["lane-level corridor (3.5 m wide)", "destination = last scene pose (r = 2 m)", "confidence & quality metadata",
       "route_valid may be false"], style="output", line_size=8.6)
s.arrow(184, 85, 204, 85); s.arrow(374, 85, 394, 85); s.arrow(564, 85, 584, 85)
s.box(14, 156, 350, 70, "Native C++ rasteriser (450 x 300 @ 0.4 m, ego frame)",
      ["rendered at 500 ms anchor poses (latest non-future pose),", "then SE(2)-warped to the exact sample pose",
       "14 map channels in [0,1] + 2 binary route channels"], style="plain", line_size=8.6)
s.box(394, 156, 352, 70, "Implication for interpretation",
      ["the route is the lane sequence the human actually drove;", "it is an oracle-quality navigation intent, not a live planner output;",
       "route reconstruction IoU measures preservation of this input"], style="note", line_size=8.6)
s.arrow(99, 130, 99, 156); s.arrow(665, 130, 665, 156)
s.save(OUT + "fig14_route_provenance.svg")

# ======================= Fig 15: matched ablation design =======================
s = SVG(760, 300, font_size=10.5)
s.text(14, 22, "Required (not yet run) same-scene matched ablation for a causal Map/Route claim", size=11, weight="bold")
s.box(14, 44, 220, 70, "Fixed inputs for all conditions",
      ["same 11,035 KITScenes Val samples", "same frozen checkpoint (e.g. KITScenes Ep. 5)",
       "identical precomputed camera BEV B_img"], style="input")
for i, (lab, lines, st) in enumerate([
    ("A: Camera only", ["map_valid = 0, route_valid = 0", "B_nav = E_map(0) + 0"], "plain"),
    ("B: Camera + HD Map", ["map_valid = 1, route_valid = 0", "(enable_route_conditioning off)"], "plain"),
    ("C: Camera + Map + Route", ["map_valid = 1, route_valid = 1", "= production contract"], "plain")]):
    x = 262 + i * 166
    s.box(x, 44, 150, 70, lab, lines, style=st, line_size=8.8)
    s.arrow(234, 79, x, 79) if i == 0 else None
s.arrow(412, 79, 428, 79); s.arrow(578, 79, 594, 79)
s.box(14, 136, 730, 60, "Report per condition (paired, scene-level bootstrap CIs, distribution of per-scene deltas)",
      ["ADE/FDE @ 1, 2, 3, 5 s; mean |lat| / |lon| error; route corridor compliance & success; route progress proxy;",
       "drivable compliance & success; comfort rate and each violation; failure rate & nonfinite predictions; latency/memory"],
      style="aux", line_size=8.8)
s.box(14, 214, 730, 62, "Controls and stratifications",
      ["shuffled-route control (another scene's route); destination-marker removal; route-corridor removal; map-only corruption/dropout;",
       "off-route / ambiguous-intersection subsets; valid-map vs invalid-map stratification"], style="note", line_size=8.8)
s.text(14, 292, "No outcome of this experiment is reported in this paper.", size=9.5, color="#B8443F", weight="bold")
s.save(OUT + "fig15_matched_ablation.svg")
print("schematics written")
