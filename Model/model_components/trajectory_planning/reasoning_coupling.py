"""Zero-init reasoning→planner coupling.

Injects the reasoning branch's output into a planner conditioning vector behind
a zero-initialised gate, so at initialisation the coupling is a no-op and the
reactive baseline output is unchanged up to numerical tolerance.

Modes:
    * ``none``                    — coupling disabled; the planner is unchanged.
    * ``pooled_latent``           — add ``alpha * reason_proj(reasoning_latent)``.
    * ``horizon_cross_attention`` — a query attends the 5 horizon tokens, then
      ``alpha * reason_proj(attended)`` is added — preserving *when* a hazard
      matters.

``alpha`` is a learned scalar initialised to 0, as in ResidualMapFusion.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

REASONING_MODES = ("none", "pooled_latent", "horizon_cross_attention")


class ReasoningCoupling(nn.Module):
    """Add a zero-init reasoning residual to a planner conditioning vector.

    Args:
        embed_dim: reasoning width (must match ``reasoning_latent`` /
            horizon-token width, 256).
        mode: one of :data:`REASONING_MODES`.
        num_heads: heads for the cross-attention in horizon mode.
        context_dim: width ``D`` of the planner conditioning vector the
            residual is added to. Defaults to ``embed_dim``. Planners whose
            context is wider than the reasoning width (the Bezier planner's
            concatenated pooled-BEV/ego/visual context) pass it explicitly:
            ``reason_proj`` then maps ``embed_dim -> D`` and, in horizon mode,
            a ``query_proj`` maps the ``D``-wide query into the attention width.

    Forward:
        coupling(context[B,D], reasoning_latent=None, horizon_tokens=None,
                 query=None) -> context'[B,D]
        With ``mode="none"`` (or missing reasoning inputs) it returns ``context``
        unchanged. In ``horizon_cross_attention`` mode the attention query is
        ``query`` if given (e.g. the flow-matching action tokens), else the
        context vector itself.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        mode: str = "none",
        num_heads: int = 4,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mode not in REASONING_MODES:
            raise ValueError(f"reasoning_mode must be one of {REASONING_MODES}, got {mode!r}.")
        self.mode = mode
        self.embed_dim = embed_dim
        self.context_dim = embed_dim if context_dim is None else int(context_dim)
        if self.context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {self.context_dim}.")
        self.query_proj: Optional[nn.Linear] = None

        if mode == "none":
            return

        # Only alpha is zero-init; reason_proj keeps normal init. Zeroing both
        # would give every coupling parameter zero gradient (a permanent zero
        # fixed point). With alpha=0 and delta!=0 the output is still a no-op at
        # init, yet alpha receives gradient.
        self.alpha = nn.Parameter(torch.zeros(()))
        self.reason_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, self.context_dim),
        )

        if mode == "horizon_cross_attention":
            if self.context_dim != embed_dim:
                # Only created for a wider context, so planners whose context
                # already matches the reasoning width keep their state_dict.
                self.query_proj = nn.Linear(self.context_dim, embed_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=0.0, batch_first=True
            )

    def forward(
        self,
        context: torch.Tensor,
        reasoning_latent: Optional[torch.Tensor] = None,
        horizon_tokens: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the reasoning-conditioned context (unchanged if mode='none')."""
        if self.mode == "none":
            return context

        if self.mode == "pooled_latent":
            if reasoning_latent is None:
                return context  # no reasoning available this step → no-op
            delta = self.reason_proj(reasoning_latent)          # [B, D]
            return context + self.alpha * delta

        # horizon_cross_attention
        if horizon_tokens is None:
            return context
        q = query if query is not None else context.unsqueeze(1)  # [B, Tq, D]
        attn_q = self.query_proj(q) if self.query_proj is not None else q  # [B, Tq, E]
        attended, _ = self.cross_attn(attn_q, horizon_tokens, horizon_tokens)  # [B, Tq, E]
        delta = self.reason_proj(attended)                        # [B, Tq, D]
        gated = self.alpha * delta
        # Broadcast back onto the caller's context shape: a single-vector context
        # gets the squeezed residual; a per-token query context keeps its tokens.
        if query is None:
            return context + gated.squeeze(1)
        return q + gated
