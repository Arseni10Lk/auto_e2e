"""Gradient-accumulation equivalence for train_il.

The World-Model windows force batch_size=1, so train_il accumulates grads over
N micro-batches with a 1/N loss scale and steps once per window. Accumulating N
size-1 micro-batches (loss/N each, zero_grad only at window start, step at window
end) must produce the same weights as one mean-reduced batch of size N; a wrong
scale or zero_grad/step placement silently changes the effective learning rate.
Pure torch, no GPU / model / shards.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from Platform.pipelines.training_checkpoint import (
    rescale_partial_accumulation_gradients,
)


def _tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def _data(n):
    torch.manual_seed(1)
    return torch.randn(n, 8), torch.randn(n, 4)


def test_accum_matches_full_batch_step():
    """N size-1 micro-batches (loss/N, step once) == one mean-reduced size-N batch.

    SmoothL1 with mean reduction over a size-N batch averages the per-sample
    gradients; summing N per-sample backward()s each scaled by 1/N reproduces that
    exact average, so a single AdamW step must land on identical weights.
    """
    N = 4
    x, y = _data(N)
    loss_fn = nn.SmoothL1Loss()  # mean reduction

    # --- reference: one full mean-reduced batch, one step ---
    ref = _tiny_model()
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-3)
    opt_ref.zero_grad()
    loss_fn(ref(x), y).backward()
    opt_ref.step()

    # --- accumulation: N size-1 micro-batches, loss/N, step at window end ---
    acc = copy.deepcopy(_tiny_model())  # same seed -> identical init as `ref`
    opt_acc = torch.optim.AdamW(acc.parameters(), lr=1e-3)
    accum = N
    micro_idx = 0
    for i in range(N):
        if micro_idx == 0:
            opt_acc.zero_grad()
        xi, yi = x[i : i + 1], y[i : i + 1]
        (loss_fn(acc(xi), yi) / accum).backward()
        micro_idx += 1
        if micro_idx == accum:
            opt_acc.step()
            micro_idx = 0

    for pr, pa in zip(ref.parameters(), acc.parameters()):
        assert torch.allclose(pr, pa, atol=1e-6), "accum step diverged from full-batch step"


def test_partial_trailing_window_is_flushed():
    """A trailing partial window must equal its own mean-reduced full batch."""
    N, accum = 3, 4  # 3 micro-batches, window of 4 -> one partial window
    x, y = _data(N)
    loss_fn = nn.SmoothL1Loss()

    ref = _tiny_model()
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-3)
    opt_ref.zero_grad()
    loss_fn(ref(x), y).backward()
    opt_ref.step()

    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    micro_idx = 0
    for i in range(N):
        if micro_idx == 0:
            opt.zero_grad()
        (loss_fn(model(x[i : i + 1]), y[i : i + 1]) / accum).backward()
        micro_idx += 1
        if micro_idx == accum:
            opt.step()
            micro_idx = 0
    # Flush the trailing partial window (the train_il epoch-end guard).
    if micro_idx > 0:
        rescale_partial_accumulation_gradients(
            model.parameters(),
            accumulation_steps=accum,
            partial_count=micro_idx,
        )
        opt.step()

    for expected, actual in zip(ref.parameters(), model.parameters()):
        assert torch.allclose(expected, actual, atol=1e-6), (
            "partial accumulation did not use the partial-window mean"
        )
