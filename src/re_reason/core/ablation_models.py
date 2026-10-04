"""Phase 1.5 Ablation Model Architectures.

Provides modular models for the 4-way Phase 1.5 causal ablation:
A. Current RE:Reason:       S_t = LayerNorm(S_(t-1) + f_theta(S_(t-1), e_t))
B. Residual No LayerNorm:    S_t = S_(t-1) + f_theta(S_(t-1), e_t)
C. Gated Residual:          S_t = S_(t-1) + g_theta(S_(t-1), e_t) ⊙ f_theta(S_(t-1), e_t)
D. GRU Baseline:            h_t = GRUCell(e_t, h_(t-1))

All models share identical observation encoder and typed decision head structures,
and are parameter-matched within <= 1.2% of each other.
"""

from typing import List, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn

from re_reason.cognition.ablation_cores import (
    BoundedAdditiveCore,
    BoundedConvexCore,
    CanonicalMGUCore,
    CanonicalUGRNNCore,
    ConvexUnboundedCore,
    GatedResidualCore,
    IndependentMLPUGRNNCore,
    NonlinearCandidateMGUCore,
    ResidualNoLNCore,
    SharedDeepMLPTrunkCore,
    SharedLinearTrunkCore,
)
from re_reason.cognition.core import CognitionCore
from re_reason.core.baseline_gru import GRUPhase1Model
from re_reason.core.model import (
    ModelOutput,
    ObservationEncoder,
    ReReasonPhase1Model,
    TypedDecisionHead,
)


class IntermediateDiagnosticsOutput(NamedTuple):
    """Container for state telemetry and gradients during forward pass."""

    terminal_logits: torch.Tensor
    final_state: torch.Tensor
    states: List[torch.Tensor]
    step_logits: Optional[torch.Tensor] = None
    candidates: Optional[List[torch.Tensor]] = None
    gates: Optional[List[torch.Tensor]] = None


class ReReasonNoLNModel(nn.Module):
    """Model B: Residual without LayerNorm.

    Formulation:
        e_t = Encoder(x_t)
        S_t = S_(t-1) + f_theta(S_(t-1), e_t)
        y_hat_t = DecisionHead(S_t)
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
        self.cognition_core = ResidualNoLNCore(
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)

            if return_step_diagnostics:
                step_logits = self.decision_head(state)
                step_logits_list.append(step_logits)

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
        )


class ReReasonGatedModel(nn.Module):
    """Model C: Gated Residual.

    Formulation:
        e_t = Encoder(x_t)
        S_t = S_(t-1) + g_theta(S_(t-1), e_t) ⊙ f_theta(S_(t-1), e_t)
        y_hat_t = DecisionHead(S_t)
    """

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 256,
        encoder_hidden_dim: int = 64,
        core_hidden_dim: int = 192,
        num_registers: int = 4,
        num_values: int = 10,
        gate_bias_init: float = 0.0,
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
        self.cognition_core = GatedResidualCore(
            state_dim=state_dim,
            obs_dim=state_dim,
            hidden_dim=core_hidden_dim,
            gate_bias_init=gate_bias_init,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)

            if return_step_diagnostics:
                step_logits = self.decision_head(state)
                step_logits_list.append(step_logits)

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
        )


class ReReasonBoundedConvexModel(nn.Module):
    """Model E: Bounded Convex Residual Update.

    Formulation:
        e_t = Encoder(x_t)
        z_t = sigmoid(g_theta(S_(t-1), e_t))
        candidate_t = tanh(f_theta(S_(t-1), e_t))
        S_t = (1 - z_t) ⊙ S_(t-1) + z_t ⊙ candidate_t
        y_hat_t = DecisionHead(S_t)

    Tests whether coordinate-wise convex interpolation with strictly bounded candidate
    updates resolves representation drift and stabilizes temporal credit assignment without
    requiring a GRU-style reset gate.
    """

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 256,
        encoder_hidden_dim: int = 64,
        core_hidden_dim: int = 192,
        num_registers: int = 4,
        num_values: int = 10,
        gate_bias_init: float = 0.0,
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
        self.cognition_core = BoundedConvexCore(
            state_dim=state_dim,
            obs_dim=state_dim,
            hidden_dim=core_hidden_dim,
            gate_bias_init=gate_bias_init,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)

            if return_step_diagnostics:
                step_logits = self.decision_head(state)
                step_logits_list.append(step_logits)

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            if return_components:
                state, cand, gate = self.cognition_core(state, e_t, return_components=True)
                candidates.append(cand)
                gates.append(gate)
            else:
                state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
            candidates=candidates if return_components else None,
            gates=gates if return_components else None,
        )


def forward_with_states_generic(
    model: nn.Module,
    inputs: torch.Tensor,
) -> IntermediateDiagnosticsOutput:
    """Extracts step-by-step latent states for any Phase 1 / ablation model."""
    if hasattr(model, "forward_with_states"):
        return model.forward_with_states(inputs)

    batch_size, seq_len, _ = inputs.shape
    device = inputs.device

    if isinstance(model, ReReasonPhase1Model):
        state = model.initial_state(batch_size, device)
        states = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = model.encoder(x_t)
            state = model.cognition_core(state, e_t)
            states.append(state)
        terminal_logits = model.decision_head(state)
        return IntermediateDiagnosticsOutput(terminal_logits=terminal_logits, final_state=state, states=states)

    elif isinstance(model, GRUPhase1Model):
        h = model.initial_state(batch_size, device)
        states = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = model.encoder(x_t)
            h = model.gru_cell(e_t, h)
            states.append(h)
        terminal_logits = model.decision_head(h)
        return IntermediateDiagnosticsOutput(terminal_logits=terminal_logits, final_state=h, states=states)

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def create_ablation_model_suite(
    input_dim: int = 21,
    state_dim: int = 256,
    num_registers: int = 4,
    num_values: int = 10,
) -> dict:
    """Instantiates all parameter-matched models for Phase 1.5 ablation."""
    model_a = ReReasonPhase1Model(
        input_dim=input_dim,
        state_dim=state_dim,
        encoder_hidden_dim=64,
        core_hidden_dim=state_dim,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_b = ReReasonNoLNModel(
        input_dim=input_dim,
        state_dim=state_dim,
        encoder_hidden_dim=64,
        core_hidden_dim=state_dim,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_c = ReReasonGatedModel(
        input_dim=input_dim,
        state_dim=state_dim,
        encoder_hidden_dim=64,
        core_hidden_dim=192,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_d = GRUPhase1Model(
        input_dim=input_dim,
        encoder_hidden_dim=64,
        encoder_out_dim=state_dim,
        gru_hidden_dim=162,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_e = ReReasonBoundedConvexModel(
        input_dim=input_dim,
        state_dim=state_dim,
        encoder_hidden_dim=64,
        core_hidden_dim=192,
        num_registers=num_registers,
        num_values=num_values,
    )

    return {
        "model_a_layernorm": model_a,
        "model_b_no_ln": model_b,
        "model_c_gated": model_c,
        "model_d_gru": model_d,
        "model_e_bounded_convex": model_e,
    }


def create_models_a_e_d_suite(
    input_dim: int = 21,
    state_dim: int = 256,
    num_registers: int = 4,
    num_values: int = 10,
) -> dict:
    """Instantiates Models A, E, and D for the Bounded Convex ablation."""
    model_a = ReReasonPhase1Model(
        input_dim=input_dim,
        state_dim=state_dim,
        encoder_hidden_dim=64,
        core_hidden_dim=state_dim,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_e = ReReasonBoundedConvexModel(
        input_dim=input_dim,
        state_dim=state_dim,
        encoder_hidden_dim=64,
        core_hidden_dim=192,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_d = GRUPhase1Model(
        input_dim=input_dim,
        encoder_hidden_dim=64,
        encoder_out_dim=state_dim,
        gru_hidden_dim=162,
        num_registers=num_registers,
        num_values=num_values,
    )
    return {
        "model_a_layernorm": model_a,
        "model_e_bounded_convex": model_e,
        "model_d_gru": model_d,
    }


class ReReasonBoundedAdditiveModel(nn.Module):
    """Variant B: Bounded Additive (S_t = S_(t-1) + tanh(f_theta))."""

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
        self.cognition_core = BoundedAdditiveCore(
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)

            if return_step_diagnostics:
                step_logits = self.decision_head(state)
                step_logits_list.append(step_logits)

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            if return_components:
                state, cand, _ = self.cognition_core(state, e_t, return_components=True)
                candidates.append(cand)
            else:
                state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
            candidates=candidates if return_components else None,
        )


class ReReasonConvexUnboundedModel(nn.Module):
    """Variant C: Convex Unbounded (S_t = (1 - z_t) S_(t-1) + z_t f_theta)."""

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 256,
        encoder_hidden_dim: int = 64,
        core_hidden_dim: int = 192,
        num_registers: int = 4,
        num_values: int = 10,
        gate_bias_init: float = 0.0,
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
        self.cognition_core = ConvexUnboundedCore(
            state_dim=state_dim,
            obs_dim=state_dim,
            hidden_dim=core_hidden_dim,
            gate_bias_init=gate_bias_init,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)

            if return_step_diagnostics:
                step_logits = self.decision_head(state)
                step_logits_list.append(step_logits)

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            if return_components:
                state, cand, gate = self.cognition_core(state, e_t, return_components=True)
                candidates.append(cand)
                gates.append(gate)
            else:
                state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
            candidates=candidates if return_components else None,
            gates=gates if return_components else None,
        )


class CanonicalUGRNNModel(nn.Module):
    """Variant E: Canonical Update Gate RNN (Collins et al., 2017)."""

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 212,
        encoder_hidden_dim: int = 64,
        encoder_out_dim: int = 256,
        num_registers: int = 4,
        num_values: int = 10,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_registers = num_registers
        self.num_values = num_values

        self.encoder = ObservationEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            obs_dim=encoder_out_dim,
        )
        self.cognition_core = CanonicalUGRNNCore(
            state_dim=state_dim,
            obs_dim=encoder_out_dim,
            gate_bias_init=gate_bias_init,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)

            if return_step_diagnostics:
                step_logits = self.decision_head(state)
                step_logits_list.append(step_logits)

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            if return_components:
                state, cand, gate = self.cognition_core(state, e_t, return_components=True)
                candidates.append(cand)
                gates.append(gate)
            else:
                state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
            candidates=candidates if return_components else None,
            gates=gates if return_components else None,
        )


class CanonicalMGUModel(nn.Module):
    """Variant F: Canonical Minimal Gated Unit (Zhou et al., 2016)."""

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 212,
        encoder_hidden_dim: int = 64,
        encoder_out_dim: int = 256,
        num_registers: int = 4,
        num_values: int = 10,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_registers = num_registers
        self.num_values = num_values

        self.encoder = ObservationEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            obs_dim=encoder_out_dim,
        )
        self.cognition_core = CanonicalMGUCore(
            state_dim=state_dim,
            obs_dim=encoder_out_dim,
            gate_bias_init=gate_bias_init,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)

            if return_step_diagnostics:
                step_logits = self.decision_head(state)
                step_logits_list.append(step_logits)

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            if return_components:
                state, cand, gate = self.cognition_core(state, e_t, return_components=True)
                candidates.append(cand)
                gates.append(gate)
            else:
                state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
            candidates=candidates if return_components else None,
            gates=gates if return_components else None,
        )


def create_mechanism_decomposition_suite(
    input_dim: int = 21,
    num_registers: int = 4,
    num_values: int = 10,
) -> dict:
    """Instantiates all 7 parameter-matched models for the mechanism audit."""
    model_a = ReReasonPhase1Model(
        input_dim=input_dim,
        state_dim=256,
        encoder_hidden_dim=64,
        core_hidden_dim=256,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_b = ReReasonBoundedAdditiveModel(
        input_dim=input_dim,
        state_dim=256,
        encoder_hidden_dim=64,
        core_hidden_dim=256,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_c = ReReasonConvexUnboundedModel(
        input_dim=input_dim,
        state_dim=256,
        encoder_hidden_dim=64,
        core_hidden_dim=192,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_d = ReReasonBoundedConvexModel(
        input_dim=input_dim,
        state_dim=256,
        encoder_hidden_dim=64,
        core_hidden_dim=192,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_e = CanonicalUGRNNModel(
        input_dim=input_dim,
        state_dim=212,
        encoder_hidden_dim=64,
        encoder_out_dim=256,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_f = CanonicalMGUModel(
        input_dim=input_dim,
        state_dim=212,
        encoder_hidden_dim=64,
        encoder_out_dim=256,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_g = GRUPhase1Model(
        input_dim=input_dim,
        encoder_hidden_dim=64,
        encoder_out_dim=256,
        gru_hidden_dim=162,
        num_registers=num_registers,
        num_values=num_values,
    )

    return {
        "model_a_layernorm": model_a,
        "model_b_bounded_additive": model_b,
        "model_c_convex_unbounded": model_c,
        "model_d_bounded_convex": model_d,
        "model_e_canonical_ugrnn": model_e,
        "model_f_canonical_mgu": model_f,
        "model_g_canonical_gru": model_g,
    }


class IndependentMLPUGRNNModel(nn.Module):
    """Variant 2: UGRNN with Independent Nonlinear MLPs.

    Candidate and gate each have a separate 2-layer MLP (hidden_dim=128).
    Evaluates whether nonlinear feature mixing alone is sufficient,
    or whether sharing representations between candidate and gate is critical.
    """

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 256,
        encoder_hidden_dim: int = 64,
        core_hidden_dim: int = 128,
        num_registers: int = 4,
        num_values: int = 10,
        gate_bias_init: float = 0.0,
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
        self.cognition_core = IndependentMLPUGRNNCore(
            state_dim=state_dim,
            obs_dim=state_dim,
            hidden_dim=core_hidden_dim,
            gate_bias_init=gate_bias_init,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)
            if return_step_diagnostics:
                step_logits_list.append(self.decision_head(state))

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            if return_components:
                state, cand, gate = self.cognition_core(state, e_t, return_components=True)
                candidates.append(cand)
                gates.append(gate)
            else:
                state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
            candidates=candidates if return_components else None,
            gates=gates if return_components else None,
        )


class SharedDeepMLPTrunkModel(nn.Module):
    """Variant 3: Shared Deep Nonlinear MLP Trunk.

    h = GELU(MLP([S, e])) (2-layer MLP trunk, hidden_dim=164)
    candidate head + gate head
    Tests whether a deeper shared nonlinear trunk improves feature synthesis.
    """

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 256,
        encoder_hidden_dim: int = 64,
        core_hidden_dim: int = 164,
        num_registers: int = 4,
        num_values: int = 10,
        gate_bias_init: float = 0.0,
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
        self.cognition_core = SharedDeepMLPTrunkCore(
            state_dim=state_dim,
            obs_dim=state_dim,
            hidden_dim=core_hidden_dim,
            gate_bias_init=gate_bias_init,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)
            if return_step_diagnostics:
                step_logits_list.append(self.decision_head(state))

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            if return_components:
                state, cand, gate = self.cognition_core(state, e_t, return_components=True)
                candidates.append(cand)
                gates.append(gate)
            else:
                state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
            candidates=candidates if return_components else None,
            gates=gates if return_components else None,
        )


class SharedLinearTrunkModel(nn.Module):
    """Variant 4: Shared Trunk Without GELU (Linear Bottleneck).

    Same architecture as Model D (hidden_dim=192), but removes the GELU activation.
    Isolates whether the non-linear activation in the shared trunk is causal for performance gains.
    """

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 256,
        encoder_hidden_dim: int = 64,
        core_hidden_dim: int = 192,
        num_registers: int = 4,
        num_values: int = 10,
        gate_bias_init: float = 0.0,
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
        self.cognition_core = SharedLinearTrunkCore(
            state_dim=state_dim,
            obs_dim=state_dim,
            hidden_dim=core_hidden_dim,
            gate_bias_init=gate_bias_init,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)
            if return_step_diagnostics:
                step_logits_list.append(self.decision_head(state))

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            if return_components:
                state, cand, gate = self.cognition_core(state, e_t, return_components=True)
                candidates.append(cand)
                gates.append(gate)
            else:
                state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
            candidates=candidates if return_components else None,
            gates=gates if return_components else None,
        )


class NonlinearCandidateMGUModel(nn.Module):
    """Variant 6: MGU with Nonlinear Candidate Processing.

    f_t = sigmoid(W_f e_t + U_f S_(t-1) + b_f)
    h_reset = f_t ⊙ S_(t-1)
    candidate_t = tanh(MLP_cand([h_reset, e_t]))
    S_t = (1 - f_t) ⊙ S_(t-1) + f_t ⊙ candidate_t

    Isolates whether adding nonlinear feature mixing to MGU's candidate state generator
    bridges the remaining gap to Model D.
    """

    def __init__(
        self,
        input_dim: int = 21,
        state_dim: int = 212,
        encoder_hidden_dim: int = 64,
        encoder_out_dim: int = 256,
        candidate_hidden_dim: int = 146,
        num_registers: int = 4,
        num_values: int = 10,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_registers = num_registers
        self.num_values = num_values

        self.encoder = ObservationEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            obs_dim=encoder_out_dim,
        )
        self.cognition_core = NonlinearCandidateMGUCore(
            state_dim=state_dim,
            obs_dim=encoder_out_dim,
            hidden_dim=candidate_hidden_dim,
            gate_bias_init=gate_bias_init,
        )
        self.decision_head = TypedDecisionHead(
            state_dim=state_dim,
            num_registers=num_registers,
            num_values=num_values,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = [] if return_step_diagnostics else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            state = self.cognition_core(state, e_t)
            if return_step_diagnostics:
                step_logits_list.append(self.decision_head(state))

        terminal_logits = self.decision_head(state)
        step_logits_tensor = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_tensor,
        )

    def forward_with_states(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> IntermediateDiagnosticsOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        states: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            if return_components:
                state, cand, gate = self.cognition_core(state, e_t, return_components=True)
                candidates.append(cand)
                gates.append(gate)
            else:
                state = self.cognition_core(state, e_t)
            states.append(state)

        terminal_logits = self.decision_head(state)
        return IntermediateDiagnosticsOutput(
            terminal_logits=terminal_logits,
            final_state=state,
            states=states,
            candidates=candidates if return_components else None,
            gates=gates if return_components else None,
        )


def create_parameterization_isolation_suite(
    input_dim: int = 21,
    num_registers: int = 4,
    num_values: int = 10,
) -> dict:
    """Creates the 7 parameterization-isolation models strictly parameter-matched (~226K parameters).

    Variants:
    1. Canonical UGRNN: Pure affine candidate + pure affine gate.
    2. Independent MLPs: Separate nonlinear candidate MLP and gate MLP (no trunk sharing).
    3. Shared Deep MLP Trunk: h = GELU(MLP([S, e])), candidate head + gate head.
    4. Shared Linear Trunk: Shared trunk without GELU (linear bottleneck).
    5. Canonical MGU: Affine candidate with state pre-gating (f_t ⊙ S_(t-1)).
    6. Nonlinear Candidate MGU: State pre-gating + nonlinear candidate MLP.
    7. Current Model D: Shared 1-layer trunk with GELU (ReReasonBoundedConvexModel).
    """
    model_1 = CanonicalUGRNNModel(
        input_dim=input_dim,
        state_dim=212,
        encoder_hidden_dim=64,
        encoder_out_dim=256,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_2 = IndependentMLPUGRNNModel(
        input_dim=input_dim,
        state_dim=256,
        encoder_hidden_dim=64,
        core_hidden_dim=128,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_3 = SharedDeepMLPTrunkModel(
        input_dim=input_dim,
        state_dim=256,
        encoder_hidden_dim=64,
        core_hidden_dim=164,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_4 = SharedLinearTrunkModel(
        input_dim=input_dim,
        state_dim=256,
        encoder_hidden_dim=64,
        core_hidden_dim=192,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_5 = CanonicalMGUModel(
        input_dim=input_dim,
        state_dim=212,
        encoder_hidden_dim=64,
        encoder_out_dim=256,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_6 = NonlinearCandidateMGUModel(
        input_dim=input_dim,
        state_dim=212,
        encoder_hidden_dim=64,
        encoder_out_dim=256,
        candidate_hidden_dim=146,
        num_registers=num_registers,
        num_values=num_values,
    )
    model_7 = ReReasonBoundedConvexModel(
        input_dim=input_dim,
        state_dim=256,
        encoder_hidden_dim=64,
        core_hidden_dim=192,
        num_registers=num_registers,
        num_values=num_values,
    )

    return {
        "1_canonical_ugrnn": model_1,
        "2_independent_mlp_ugrnn": model_2,
        "3_shared_deep_mlp_trunk": model_3,
        "4_shared_linear_trunk": model_4,
        "5_canonical_mgu": model_5,
        "6_nonlinear_candidate_mgu": model_6,
        "7_current_model_d": model_7,
    }

