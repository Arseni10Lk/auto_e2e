from abc import ABC, abstractmethod

import torch.nn as nn


class BasePlanner(nn.Module, ABC):
    """Abstract trajectory planner.

    ``forward()`` always performs inference and returns ``(trajectory)``
    regardless of the underlying decoder. It must not return mode-dependent
    intermediate quantities (e.g. the flow-matching velocity field). A caller
    can rely on the first return being a fully-formed
    ``[B, num_timesteps * num_signals]`` trajectory, so a train loop cannot
    silently regress a velocity field against a trajectory target.

    The flow-matching velocity objective (velocity MSE against x1-x0) is not
    wired into training; train_il regresses the ``forward()`` trajectory.
    """

    @abstractmethod
    def forward(self, bev_features, visual_history, egomotion_history,
                **kwargs):
        """Inference: return ``(trajectory)``."""
        raise NotImplementedError
