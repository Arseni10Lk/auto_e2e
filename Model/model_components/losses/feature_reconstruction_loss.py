import torch
import torch.nn as nn


class FeatureReconstructionLoss(nn.Module):
    """JEPA-style feature reconstruction loss for the world model.

    Compares the future feature maps predicted by the world model (a
    tuple/list of ``num_future_steps`` tensors, each ``[B, C, H, W]``) against
    target feature maps of identical shapes. The loss is in feature space:
    targets come from a frozen image backbone applied to the future frames at
    +1.6s, +3.2s, +4.8s and +6.4s.

    Supports mean reduction (optionally weighted per future timestep) and a
    ``"none"`` reduction that returns the per-timestep losses for logging.
    """

    # Declares the registered buffer's type so mypy resolves it to Tensor
    # (not Tensor | Module) when it is used in arithmetic below.
    temporal_weights: torch.Tensor

    def __init__(self, num_future_steps: int = 4, temporal_weights=None,
                 reduction: str = "mean", loss_type: str = "l2"):
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        # V-JEPA uses L1 (Bardes et al. 2024, arXiv:2404.08471) where I-JEPA
        # uses L2; both are offered, plus a Huber variant.
        if loss_type not in ("l2", "l1", "smooth_l1"):
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.num_future_steps = num_future_steps
        self.reduction = reduction
        self.loss_type = loss_type

        if temporal_weights is None:
            weights = torch.ones(num_future_steps)
        else:
            weights = torch.as_tensor(temporal_weights, dtype=torch.float32)
            if weights.numel() != num_future_steps:
                raise ValueError(
                    f"temporal_weights must have {num_future_steps} elements, "
                    f"got {weights.numel()}"
                )
        # Normalise so the weighted mean stays on the unweighted scale; a
        # near-zero sum would produce NaN, so such weights are kept as given.
        total = weights.sum()
        if torch.abs(total) > 1e-8:
            weights = weights / total * num_future_steps
        self.register_buffer("temporal_weights", weights)

    def forward(self, predicted_features, target_features) -> torch.Tensor:
        """Compute the loss.

        Args:
            predicted_features: tuple/list of ``num_future_steps`` tensors
                ``[B, C, H, W]`` as returned by ``WorldActionModel.predict_future``.
            target_features: tuple/list of tensors with the same shapes,
                produced by a frozen backbone on the future frames (targets
                should be detached / require no grad).

        Returns:
            Scalar loss (``reduction="mean"``) or per-timestep losses of
            shape ``[num_future_steps]`` (``reduction="none"``).
        """
        if len(predicted_features) != self.num_future_steps:
            raise ValueError(
                f"Expected {self.num_future_steps} predicted feature maps, "
                f"got {len(predicted_features)}"
            )
        if len(target_features) != self.num_future_steps:
            raise ValueError(
                f"Expected {self.num_future_steps} target feature maps, "
                f"got {len(target_features)}"
            )

        per_step_losses: list[torch.Tensor] = []
        for pred, target in zip(predicted_features, target_features):
            if pred.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch: predicted {tuple(pred.shape)} vs "
                    f"target {tuple(target.shape)}"
                )
            if self.loss_type == "l1":
                per_step_losses.append(torch.mean((pred - target).abs()))
            elif self.loss_type == "smooth_l1":
                per_step_losses.append(
                    nn.functional.smooth_l1_loss(pred, target)
                )
            else:  # l2 (default, MSE)
                per_step_losses.append(torch.mean((pred - target) ** 2))

        per_step = torch.stack(per_step_losses)  # [num_future_steps]
        weighted = per_step * self.temporal_weights

        if self.reduction == "none":
            return weighted
        return weighted.mean()
