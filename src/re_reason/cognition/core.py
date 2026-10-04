"""Minimal CognitionCore implementation for Phase 1.

Implements the locked additive residual state update formulation:
    S_t = LayerNorm(S_(t-1) + f_theta(S_(t-1), e_t))
with R=1 (strictly single pass; no recursion).
"""

import torch
import torch.nn as nn

from re_reason.cognition.base import BaseCognitionCore


class CognitionCore(BaseCognitionCore):
    """Phase 1 Cognition Core with additive residual state update.

    Applies a compact feedforward residual transformation over the concatenated
    prior state and observation embedding, stabilized by LayerNorm.

    Recursion depth is locked to R=1. No branching, adaptive halting, or multi-step
    recursion is implemented.
    """

    def __init__(
        self,
        state_dim: int = 256,
        obs_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Feedforward transformation f_theta([S, e])
        self.net = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.layer_norm = nn.LayerNorm(state_dim)

    def forward(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Executes a single-pass additive state update.

        Args:
            state: Prior latent state S_(t-1) of shape (B, state_dim).
            observation: Encoded observation e_t of shape (B, obs_dim).

        Returns:
            Updated latent state S_t of shape (B, state_dim).
        """
        combined = torch.cat([state, observation], dim=-1)
        delta = self.net(combined)
        updated_state = self.layer_norm(state + delta)
        return updated_state
