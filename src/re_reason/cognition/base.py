"""Abstract base contract for recurrent cognition cores."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseCognitionCore(nn.Module, ABC):
    """Abstract base contract for recurrent cognition cores.

    Defines the interface signature for recursive/recurrent state updates.
    Concrete neural logic and weight-tied recursion belong to Phase 1+.
    """

    @abstractmethod
    def forward(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Computes candidate state update from current state and observation.

        Args:
            state: Current latent state tensor of shape (..., D).
            observation: Encoded observation embedding of shape (..., D_obs).

        Returns:
            Updated latent state tensor of shape (..., D).
        """
        pass
