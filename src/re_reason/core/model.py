"""Phase 1 RE:Reason Model Wrapper.

Strictly follows the locked Phase 1 architecture:
    Input -> Observation Encoder -> Persistent State Update -> Typed Prediction
with R=1, K=1, no recursion, no branching, no adaptive halting, no verifiers,
and no transactional commits.
"""

from typing import Dict, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn

from re_reason.cognition.core import CognitionCore


class ModelOutput(NamedTuple):
    """Output container for Phase 1 sequential inference."""

    logits: torch.Tensor  # Terminal logits at t=T of shape (B, num_registers, num_values)
    final_state: torch.Tensor  # Final latent state S_T of shape (B, state_dim)
    step_logits: Optional[torch.Tensor] = None  # Step-wise logits (B, T, num_registers, num_values)


class ObservationEncoder(nn.Module):
    """Lightweight 2-layer MLP observation encoder for task inputs."""

    def __init__(self, input_dim: int = 21, hidden_dim: int = 64, obs_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, obs_dim),
            nn.LayerNorm(obs_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes observation x_t of shape (..., input_dim) to (..., obs_dim)."""
        return self.net(x)


class TypedDecisionHead(nn.Module):
    """Linear projection from latent state to multi-register categorical logits."""

    def __init__(self, state_dim: int = 256, num_registers: int = 4, num_values: int = 10) -> None:
        super().__init__()
        self.num_registers = num_registers
        self.num_values = num_values
        self.proj = nn.Linear(state_dim, num_registers * num_values)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Projects state tensor to register categorical logits.

        Args:
            state: Latent state of shape (..., state_dim).

        Returns:
            Logits of shape (..., num_registers, num_values).
        """
        raw_logits = self.proj(state)
        new_shape = state.shape[:-1] + (self.num_registers, self.num_values)
        return raw_logits.view(*new_shape)


class ReReasonPhase1Model(nn.Module):
    """RE:Reason Phase 1 Minimal Persistent-State Model.

    Locked formulation:
        e_t = Encoder(x_t)
        S_t = LayerNorm(S_(t-1) + f_theta(S_(t-1), e_t))
        y_hat_t = DecisionHead(S_t)

    Parameters are targeted at approximately <= 250K.
    """

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 256,
        encoder_hidden_dim: int = 64,
        core_hidden_dim: int = 256,
        num_registers: int = 4,
        num_values: int = 10,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_registers = num_registers
        self.num_values = num_values

        self.encoder = ObservationEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            obs_dim=state_dim,
        )
        self.cognition_core = CognitionCore(
            state_dim=state_dim,
            obs_dim=state_dim,
            hidden_dim=core_hidden_dim,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generates zero-initialized initial state S_0 of shape (B, state_dim)."""
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        """Executes sequential state update over horizon T.

        Args:
            inputs: FloatTensor of shape (B, T, input_dim).
            initial_state: Optional initial state S_0 of shape (B, state_dim).
            return_step_diagnostics: If True, computes and returns step-wise logits.

        Returns:
            ModelOutput containing terminal logits at t=T, final state, and optional step logits.
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        state = (
            initial_state
            if initial_state is not None
            else self.initial_state(batch_size, device)
        )

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)

            if return_step_diagnostics:
                step_logits = self.decision_head(state)
                step_logits_list.append(step_logits)

        # Terminal prediction strictly from final state S_T
        terminal_logits = self.decision_head(state)

        step_logits_tensor = (
            torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None
        )

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )
