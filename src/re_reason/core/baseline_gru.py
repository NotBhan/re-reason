"""Parameter-matched GRU Baseline Model for Phase 1.

Uses standard PyTorch nn.GRUCell under identical input representation,
observation encoder, decision head, training budget, and evaluation procedure.
"""

from typing import Optional
import torch
import torch.nn as nn

from re_reason.core.model import ModelOutput, ObservationEncoder, TypedDecisionHead


class GRUPhase1Model(nn.Module):
    """GRU Baseline Model parameter-matched to ReReasonPhase1Model.

    Pipeline:
        e_t = Encoder(x_t)
        h_t = GRUCell(e_t, h_(t-1))
        y_hat_t = DecisionHead(h_t)
    """

    def __init__(
        self,
        input_dim: int = 21,
        encoder_hidden_dim: int = 64,
        encoder_out_dim: int = 256,
        gru_hidden_dim: int = 162,
        num_registers: int = 4,
        num_values: int = 10,
    ) -> None:
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_registers = num_registers
        self.num_values = num_values

        self.encoder = ObservationEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            obs_dim=encoder_out_dim,
        )
        self.gru_cell = nn.GRUCell(
            input_size=encoder_out_dim,
            hidden_size=gru_hidden_dim,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=gru_hidden_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generates zero-initialized hidden state h_0 of shape (B, gru_hidden_dim)."""
        return torch.zeros((batch_size, self.gru_hidden_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        """Executes sequential GRU update over horizon T.

        Args:
            inputs: FloatTensor of shape (B, T, input_dim).
            initial_state: Optional initial hidden state h_0.
            return_step_diagnostics: If True, computes and returns step-wise logits.

        Returns:
            ModelOutput containing terminal logits at t=T, final state, and optional step logits.
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        h = (
            initial_state
            if initial_state is not None
            else self.initial_state(batch_size, device)
        )

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            h = self.gru_cell(e_t, h)

            if return_step_diagnostics:
                step_logits = self.decision_head(h)
                step_logits_list.append(step_logits)

        # Terminal prediction strictly from final state h_T
        terminal_logits = self.decision_head(h)

        step_logits_tensor = (
            torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None
        )

        return ModelOutput(
            logits=terminal_logits,
            final_state=h,
            step_logits=step_logits_tensor,
        )
