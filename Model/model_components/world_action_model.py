"""World Action Model: slow (~1 Hz) world-model branch.

Encodes the recent multi-camera history into a rolling visual-history vector and
predicts future backbone feature maps (JEPA, self-supervised):

1. Backbone: one shared image backbone; the JEPA target is a frozen copy of it
   (``JepaTargetEncoder(mode="frozen")``).
2. Horizons: ``N_past = N_future``, default 4, sampled at 1 Hz.
3. Feature level: the history uses a per-frame 224-dim embedding; the JEPA
   future prediction reconstructs backbone feature maps ``[B, C, 8, 8]``.
4. Visual history: a FIFO of the last ``history_len`` embeddings,
   ``history_len * frame_embed_dim = 4 * 224 = 896`` (``visual_history_dim``),
   fed to the reactive planner.
5. Training: L1 in feature space, equal loss weight; future frames come from
   the 1 Hz stream of the dataloader.

The per-tick ``forward`` returns ``(visual_embedding, future_state_pred)``; the
JEPA loss is computed separately via :meth:`jepa_loss` in the training loop.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .jepa_target_encoder import JepaTargetEncoder, compute_jepa_loss
from .losses.feature_reconstruction_loss import FeatureReconstructionLoss

logger = logging.getLogger(__name__)


class ViewAttentionPool(nn.Module):
    """Default multi-camera fusion via cross-attention: ``[B, V, C] -> [B, C]``.

    A single learned query attends over the per-view tokens with learnable camera-
    position embeddings, weighting cameras by relevance.
    Used by default in ``FrameEncoder`` (``view_aggregator="attention"``); set
    ``view_aggregator="mean"`` to revert to the equal-weight mean.
    """

    def __init__(self, embed_dim: int = 768, num_views: int = 7, num_heads: int = 4,
                 max_views: int = 16):
        super().__init__()
        # Distinct positional embedding per camera up to max_views, so mixed rigs
        # (L2D 6, NVIDIA 7, ...) never share one camera's code for another.
        # Runtime V slices the first V; V > max_views is rejected.
        self.max_views = max(max_views, num_views)
        self.view_embed = nn.Parameter(torch.randn(1, self.max_views, embed_dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, view_tokens: torch.Tensor) -> torch.Tensor:
        # view_tokens: [B, V, C], C = embed_dim
        B, V, C = view_tokens.shape
        if V > self.max_views:
            raise ValueError(
                f"ViewAttentionPool got V={V} cameras but was built for at most "
                f"max_views={self.max_views}; increase max_views."
            )
        v_embed = self.view_embed[:, :V]
        tok = view_tokens + v_embed
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, tok, tok)
        return self.norm(out.squeeze(1))


class FrameEncoder(nn.Module):
    """One multi-camera frame ``[B, 3, H, W]`` -> embedding ``[B, frame_embed_dim]``.

    backbone -> last feature map -> global average pool -> linear projection.

    ``detach_backbone`` (default True) stop-gradients the backbone feature map
    before the projection. The backbone is shared with the trajectory branch;
    without the detach, the JEPA loss pulls it toward future-feature prediction
    and away from what the planner needs. Detaching confines JEPA's gradient to
    this module's projection, aggregator and future predictor; the JEPA target is
    a frozen copy, so the signal stays stable.
    """

    def __init__(self, backbone: nn.Module, feature_channels: int = 768,
                 frame_embed_dim: int = 224, view_aggregator: str = "attention",
                 num_views: int = 7, detach_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.detach_backbone = detach_backbone
        self.proj = nn.Linear(feature_channels, frame_embed_dim)
        if num_views == 1 and view_aggregator == "attention":
            logger.warning(
                "num_views is 1; falling back view_aggregator from 'attention' to 'mean' "
                "to avoid unnecessary attention overhead."
            )
            view_aggregator = "mean"

        self.view_aggregator = view_aggregator
        
        if view_aggregator not in ("mean", "attention"):
            raise ValueError(f"Unknown view_aggregator {view_aggregator!r}. Available: 'mean', 'attention'.")
            
        if view_aggregator == "attention":
            self.view_pool = ViewAttentionPool(embed_dim=feature_channels, num_views=num_views)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` or multi-view ``[B, V, 3, H, W]`` -> ``[B, embed]``.

        For multi-view input each camera is encoded and the per-view embeddings
        are aggregated (via mean or attention) into a single per-frame embedding.
        """
        if frame.dim() == 5:                       # [B, V, 3, H, W]
            B, V = frame.shape[:2]
            feats = self.backbone(frame.reshape(B * V, *frame.shape[2:]))
            m = feats[-1] if isinstance(feats, (list, tuple)) else feats
            if self.detach_backbone:               # keep JEPA grad off the shared backbone
                m = m.detach()
            m = m.mean(dim=(2, 3)).reshape(B, V, -1)             # [B, V, feature_channels]

            if self.view_aggregator == "attention":
                m = self.view_pool(m)                            # [B, feature_channels]
            else:
                m = m.mean(dim=1)                                # [B, feature_channels]

            return self.proj(m)                                  # [B, embed]
        feats = self.backbone(frame)               # [B, 3, H, W]
        m = feats[-1] if isinstance(feats, (list, tuple)) else feats
        if self.detach_backbone:                   # keep JEPA grad off the shared backbone
            m = m.detach()
        return self.proj(m.mean(dim=(2, 3)))                     # [B, embed]


class RollingHistoryBuffer:
    """Inference-time FIFO buffer of the last ``history_len`` frame embeddings.

    Push one embedding per tick; ``visual_history()`` concatenates them
    (oldest -> newest) into ``[B, history_len * frame_embed_dim]``, left-padding
    with zeros until the buffer fills. Mirrors the windowed encoding used in
    training so train/inference share a representation.
    """

    def __init__(self, history_len: int = 4):
        self.history_len = history_len
        self._buf: list[torch.Tensor] = []

    def push(self, embedding: torch.Tensor) -> None:
        self._buf.append(embedding)
        if len(self._buf) > self.history_len:
            self._buf.pop(0)  # first-in, first-out

    def visual_history(self) -> torch.Tensor | None:
        if not self._buf:
            return None
        pad = [torch.zeros_like(self._buf[0])] * (self.history_len - len(self._buf))
        return torch.cat(pad + self._buf, dim=1)


class HistoryAttentionPool(nn.Module):
    """Opt-in learnable temporal aggregator, alternative to the concat FIFO.

    A single learnable query attends over the ``history_len`` frame embeddings plus
    learned temporal positional embeddings (recency-aware), so frames are
    weighted by relevance instead of equally. The pooled per-frame vector is
    projected back to ``history_len * frame_embed_dim`` so it is a drop-in for
    the concat-FIFO interface (same ``visual_history_dim`` output, e.g. 896).
    """

    def __init__(self, frame_embed_dim: int = 224, history_len: int = 4,
                 num_heads: int = 4):
        super().__init__()
        self.history_len = history_len
        self.frame_embed_dim = frame_embed_dim
        self.temporal_pos = nn.Parameter(
            torch.randn(1, history_len, frame_embed_dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, frame_embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            frame_embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(frame_embed_dim)
        self.out = nn.Linear(frame_embed_dim, history_len * frame_embed_dim)

    def forward(self, history_concat: torch.Tensor) -> torch.Tensor:
        """``[B, history_len*frame_embed_dim]`` -> ``[B, history_len*frame_embed_dim]``."""
        B = history_concat.shape[0]
        tokens = history_concat.reshape(B, self.history_len, self.frame_embed_dim)
        tokens = tokens + self.temporal_pos                 # recency-aware
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)            # [B, 1, embed]
        pooled = self.norm(pooled.squeeze(1))               # [B, embed]
        return self.out(pooled)                             # [B, history_len*embed]


class BackboneFeatureMap(nn.Module):
    """One multi-camera frame -> the backbone's last **feature map** ``[B, C, hw, hw]``.

    Unlike :class:`FrameEncoder` (which pools to a vector for the history), this
    keeps the spatial feature map, which is the JEPA target the future predictor
    reconstructs.
    Multi-view frames are mean-pooled over views; the map is adaptive-avg-pooled
    to ``hw x hw`` so the target grid is fixed regardless of the backbone's native
    resolution.
    """

    def __init__(self, backbone: nn.Module, feature_hw: int = 8):
        super().__init__()
        self.backbone = backbone
        self.feature_hw = feature_hw

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() == 5:                                   # [B, V, 3, H, W]
            B, V = frame.shape[:2]
            feats = self.backbone(frame.reshape(B * V, *frame.shape[2:]))
            m = feats[-1] if isinstance(feats, (list, tuple)) else feats
            m = m.reshape(B, V, *m.shape[1:]).mean(dim=1)      # [B, C, h, w]
        else:                                                  # [B, 3, H, W]
            feats = self.backbone(frame)
            m = feats[-1] if isinstance(feats, (list, tuple)) else feats
        if m.shape[-2:] != (self.feature_hw, self.feature_hw):
            m = nn.functional.adaptive_avg_pool2d(m, (self.feature_hw, self.feature_hw))
        return m                                               # [B, C, hw, hw]


class FutureFeatureMapPredictor(nn.Module):
    """Predict ``num_future_steps`` future backbone feature maps ``[B, C, hw, hw]``
    from the Encoded Visual History ``[B, visual_history_dim]``.

    Lightweight decoder: a shared linear seed -> ``[B, mid, hw, hw]`` followed by
    one 1x1 conv head per future step (keeps params modest vs a direct
    ``history_dim -> C*hw*hw`` linear).
    """

    def __init__(self, in_dim: int, channels: int, feature_hw: int,
                 num_future_steps: int, mid: int = 128):
        super().__init__()
        self.channels, self.feature_hw, self.mid = channels, feature_hw, mid
        self.num_future_steps = num_future_steps
        self.seed = nn.Linear(in_dim, mid * feature_hw * feature_hw)
        self.act = nn.GELU()
        self.heads = nn.ModuleList(
            [nn.Conv2d(mid, channels, kernel_size=1) for _ in range(num_future_steps)])

    def forward(self, visual_history: torch.Tensor) -> list:
        B = visual_history.shape[0]
        seed = self.act(self.seed(visual_history))
        seed = seed.reshape(B, self.mid, self.feature_hw, self.feature_hw)
        return [head(seed) for head in self.heads]   # N x [B, C, hw, hw]


class WorldActionModel(nn.Module):
    """Slow world-model branch: history -> visual_history (+ future JEPA in train).

    Args:
        backbone: shared image backbone (e.g. ``Backbone``); a frozen copy is
            used as the JEPA target.
        feature_channels: channels of the backbone's last feature map.
        frame_embed_dim: per-frame embedding size (default 224).
        history_len: number of past frames in the FIFO buffer (default 4).
        num_future_steps: future horizons to predict (default = history_len).
        loss_type: feature-space distance for the JEPA loss (``"l1"`` default).
    """

    def __init__(self, backbone: nn.Module, feature_channels: int = 768,
                 frame_embed_dim: int = 224, history_len: int = 4,
                 num_future_steps: Optional[int] = None, loss_type: str = "l1",
                 history_aggregator: str = "concat", feature_hw: int = 8,
                 view_aggregator: str = "attention", num_views: int = 7):
        super().__init__()
        if history_aggregator not in ("concat", "attention"):
            raise ValueError(
                f"Unknown history_aggregator {history_aggregator!r}. "
                "Available: 'concat' (default), 'attention'.")
        if num_future_steps is None:
            num_future_steps = history_len
        self.history_len = history_len
        self.num_future_steps = num_future_steps
        self.frame_embed_dim = frame_embed_dim
        self.feature_channels = feature_channels
        self.feature_hw = feature_hw
        self.visual_history_dim = history_len * frame_embed_dim  # 4*224 = 896

        # "concat": equal-weight FIFO; "attention": learnable temporal pool.
        self.history_pool = (
            HistoryAttentionPool(frame_embed_dim, history_len)
            if history_aggregator == "attention" else None)

        self.encoder = FrameEncoder(
            backbone, feature_channels, frame_embed_dim,
            view_aggregator=view_aggregator, num_views=num_views
        )
        # JEPA target: frozen, stop-gradient copy of the feature-map extractor.
        self.target = JepaTargetEncoder(
            BackboneFeatureMap(backbone, feature_hw), mode="frozen")
        self.future_predictor = FutureFeatureMapPredictor(
            self.visual_history_dim, feature_channels, feature_hw, num_future_steps)
        self.recon_loss = FeatureReconstructionLoss(
            num_future_steps=num_future_steps, loss_type=loss_type)

    def encode_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        """``[B, history_len, 3, H, W]`` -> visual_history ``[B, 896]`` (FIFO order)."""
        T = history_frames.shape[1]
        embs = [self.encoder(history_frames[:, t]) for t in range(T)]
        return torch.cat(embs, dim=1)

    def aggregate_history(self, history_concat):
        """Turn the FIFO of frame embeddings into the visual_history vector.

        ``concat`` mode returns the concatenation unchanged; ``attention`` mode
        runs the learnable temporal attention-pool. Same ``[B, visual_history_dim]``
        output either way (drop-in). Passes ``None`` through (empty buffer).
        """
        if history_concat is None or self.history_pool is None:
            return history_concat
        return self.history_pool(history_concat)

    def predict_future(self, visual_history: torch.Tensor) -> list:
        """``visual_history [B, visual_history_dim]`` -> list of ``num_future_steps``
        predicted future backbone **feature maps** ``[B, feature_channels, hw, hw]``."""
        return self.future_predictor(visual_history)

    def forward(self, frame: torch.Tensor,
                visual_history: torch.Tensor | None = None):
        """Per-tick (online) call::

            visual_embedding, future_state_pred = WorldActionModel(frame, visual_history)

        Args:
            frame: ``[B, 3, H, W]`` (or ``[B, V, 3, H, W]`` collapsed) current
                1 Hz multi-camera frame.
            visual_history: ``[B, history_len*frame_embed_dim]`` current circular
                buffer (the Encoded Visual History) used to predict the future;
                ``None`` at the very first ticks / pure inference.

        Returns:
            ``(visual_embedding, future_state_pred)`` where
            * ``visual_embedding`` ``[B, frame_embed_dim]`` is pushed to the
              external :class:`RollingHistoryBuffer` (FIFO, size N) which forms
              the Encoded Visual History fed to the reactive planner;
            * ``future_state_pred`` is a list of ``num_future_steps`` future
              backbone **feature maps** ``[B, feature_channels, hw, hw]`` (only
              needed in training; ``None`` if no ``visual_history`` is given).
        """
        visual_embedding = self.encoder(frame)
        future_state_pred = (self.predict_future(visual_history)
                             if visual_history is not None else None)
        return visual_embedding, future_state_pred

    def jepa_loss(self, future_state_pred: list,
                  future_frames: torch.Tensor) -> torch.Tensor:
        """Future Feature Reconstruction Loss (JEPA, L1) vs the FROZEN target.

        Args:
            future_state_pred: list of ``num_future_steps`` feature maps
                ``[B, feature_channels, hw, hw]`` from :meth:`predict_future`.
            future_frames: ``[B, num_future_steps, 3, H, W]`` (or
                ``[B, num_future_steps, V, 3, H, W]``) actual future frames; the
                frozen target maps each to ``[B, feature_channels, hw, hw]``.
        """
        future_obs = [future_frames[:, k] for k in range(self.num_future_steps)]
        return compute_jepa_loss(future_state_pred, future_obs, self.target,
                                 self.recon_loss, weight=1.0)
