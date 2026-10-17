"""Optional encoder of the past visual/ego history.

Compresses a 10 Hz history to a coarser-in-time, richer-in-feature ~1 Hz
sequence and summarises it into a single context vector. This encodes the
past only; future waypoints are decoded by the trajectory planners.
"""

import torch
import torch.nn as nn

from .base import BaseTemporalMemory


class HistoryEncoder(nn.Module):
    """Compress a [B, T, input_dim] past sequence into a [B, hidden_dim] context.

    Pipeline:
      1. Temporal compression: non-overlapping Conv1d with
         ``kernel=stride=subsample_ratio`` pools each ``subsample_ratio``-step
         window (e.g. 10 steps at 10 Hz -> 1 step at 1 Hz) while projecting
         the features to ``hidden_dim``. The history is ordered oldest -> most
         recent; when ``T`` is not a multiple of the ratio, the oldest
         ``T % subsample_ratio`` steps are dropped so the windows stay aligned
         to the present.
      2. Sequence summarisation: a GRU over the ~1 Hz sequence; the final
         hidden state is the history context.

    Args:
        input_dim: feature size of each history step.
        hidden_dim: feature size of the compressed steps and output context.
        subsample_ratio: temporal pooling factor (default 10: 10 Hz -> 1 Hz).
        input_hz: nominal input rate, for documentation/inspection only.

    Example: T=64 at 10 Hz with ``subsample_ratio=10`` -> 6 compressed steps
    (~1 Hz over 6.4 s) -> context ``[B, hidden_dim]``.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 subsample_ratio: int = 10, input_hz: float = 10.0):
        super().__init__()
        if subsample_ratio < 1:
            raise ValueError(
                f"subsample_ratio must be >= 1, got {subsample_ratio}"
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.subsample_ratio = subsample_ratio
        self.input_hz = input_hz
        self.output_hz = input_hz / subsample_ratio

        self.temporal_compress = nn.Conv1d(
            input_dim, hidden_dim,
            kernel_size=subsample_ratio, stride=subsample_ratio,
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def compressed_length(self, T: int) -> int:
        """Number of ~1 Hz steps produced from a T-step input."""
        return T // self.subsample_ratio

    def compress(self, history: torch.Tensor) -> torch.Tensor:
        """Temporal compression only: [B, T, input_dim] -> [B, T', hidden_dim]
        with ``T' = T // subsample_ratio``.

        The oldest ``T % subsample_ratio`` steps are dropped.
        """
        B, T, _ = history.shape
        if T < self.subsample_ratio:
            raise ValueError(
                f"History length {T} is shorter than subsample_ratio "
                f"{self.subsample_ratio}; need at least one full window."
            )
        remainder = T % self.subsample_ratio
        if remainder:
            history = history[:, remainder:, :]
        x = history.transpose(1, 2)            # [B, input_dim, T]
        x = self.temporal_compress(x)          # [B, hidden_dim, T']
        x = self.activation(x)
        x = x.transpose(1, 2)                  # [B, T', hidden_dim]
        return self.norm(x)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Encode the past.

        Args:
            history: ``[B, T, input_dim]`` past sequence ordered oldest ->
                most recent (e.g. T=64 at 10 Hz).

        Returns:
            context: ``[B, hidden_dim]`` summary of the compressed history.
        """
        compressed = self.compress(history)     # [B, T', hidden_dim]
        _, h_n = self.gru(compressed)            # h_n: [1, B, hidden_dim]
        return h_n.squeeze(0)


class OneHzHistoryEncoder(BaseTemporalMemory):
    """Jointly compress concatenated visual and egomotion histories at 1 Hz.

    Flat ``[B, feat]`` inputs are passed through unchanged.
    """
    def __init__(self, visual_dim=896, egomotion_dim=256, subsample_ratio=10, input_hz=10.0):
        super().__init__()
        self.visual_dim = visual_dim
        self.egomotion_dim = egomotion_dim

        joint_dim = visual_dim + egomotion_dim
        self.encoder = HistoryEncoder(
            input_dim=joint_dim,
            hidden_dim=joint_dim,
            subsample_ratio=subsample_ratio,
            input_hz=input_hz
        )

    def forward(self, visual_history, egomotion_history, **kwargs):
        if visual_history.ndim == 2:
            return visual_history, egomotion_history

        # [B, T, visual_dim + egomotion_dim] -> [B, visual_dim + egomotion_dim]
        joint_history = torch.cat([visual_history, egomotion_history], dim=-1)
        joint_context = self.encoder(joint_history)

        v_ctx = joint_context[:, :self.visual_dim]
        e_ctx = joint_context[:, self.visual_dim:]
        return v_ctx, e_ctx
