"""Model E held-out generalization: four-cell SCLM benchmark candidate models.

Implements the four registered candidates of the 2026-10-18 Phase 1.6 amendment
(`docs/reports/phase_1_6_architectural_reformulation_proposal.md`):

* Model A (SCLM): four explicit 53-dim slots (212-dim latent budget), shared
  decoder/writer, fixed mod-10 circular-convolution ALU — 8,255 trainable parameters.
* Model B (Slot-MLP control): Model A with a learned ALU (20 -> 64 -> 10) — 10,249.
* Model C (Cyclic-MGU control): monolithic 212-dim Model E-architecture state with
  a fixed convolutional ALU feature — 227,062.
* Model D (Monolithic-MLP control): Model C with the learned ALU — 229,056.

All four consume the 22-dimensional Phase 1.6 instruction token and emit post-step
logits for all four registers at every timestep, shape (B, T, 4, 10), used by the
unified objective in `re_reason.training.heldout_objective`.
"""

from typing import Dict, NamedTuple, Optional, Type

import torch
import torch.nn as nn

from re_reason.cognition.ablation_cores import NonlinearCandidateMGUCore
from re_reason.core.model import ObservationEncoder, TypedDecisionHead


NUM_REGISTERS = 4
NUM_VALUES = 10
INSTRUCTION_DIM = 22
SLOT_DIM = 53
STATE_DIM = 212
ENCODER_HIDDEN_DIM = 64
ENCODER_OUT_DIM = 256
CANDIDATE_HIDDEN_DIM = 146
ALU_HIDDEN_DIM = 64


class HeldOutCandidateOutput(NamedTuple):
    """Step-wise post-step register logits for the registered four-cell models."""

    step_logits: torch.Tensor  # (B, T, num_registers, num_values)
    final_state: torch.Tensor  # (B, slot_dim * num_registers) or (B, state_dim)


def circular_convolution(p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
    """Fixed exact mod-10 circular convolution over two operand distributions.

    ``p_sum[..., k] = sum_i p_a[..., i] * p_b[..., (k - i) % num_values]``.

    For one-hot operands this yields the exact one-hot modular sum; it has zero
    trainable parameters. Shapes: ``(..., num_values)`` -> ``(..., num_values)``.
    """
    num_values = p_a.shape[-1]
    idx = torch.arange(num_values, device=p_a.device)
    roll_index = (idx.unsqueeze(1) - idx.unsqueeze(0)) % num_values  # (K, I): (k - i) % 10
    shifted = p_b[..., roll_index]  # (..., K, I)
    return (p_a.unsqueeze(-2) * shifted).sum(dim=-1)


class SharedSlotDecoder(nn.Module):
    """Shared slot decoder Dec: 53 -> 64 -> 10 (GELU; raw logits; 4,106 parameters)."""

    def __init__(
        self,
        slot_dim: int = SLOT_DIM,
        hidden_dim: int = ENCODER_HIDDEN_DIM,
        num_values: int = NUM_VALUES,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_values),
        )

    def forward(self, slot: torch.Tensor) -> torch.Tensor:
        return self.net(slot)


class SharedSlotWriter(nn.Module):
    """Shared slot writer Write: 10 -> 64 -> 53 (GELU; tanh; 4,149 parameters)."""

    def __init__(
        self,
        num_values: int = NUM_VALUES,
        hidden_dim: int = ENCODER_HIDDEN_DIM,
        slot_dim: int = SLOT_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_values, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
            nn.Tanh(),
        )

    def forward(self, distribution: torch.Tensor) -> torch.Tensor:
        return self.net(distribution)


class LearnedALU(nn.Module):
    """Learned MLP ALU: 20 -> 64 -> 10 (GELU; raw logits; 1,994 parameters).

    Used by Model B and Model D; callers apply softmax to form the output
    distribution, matching the registered interface.
    """

    def __init__(
        self,
        num_values: int = NUM_VALUES,
        hidden_dim: int = ALU_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * num_values, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_values),
        )

    def forward(self, operand_pair: torch.Tensor) -> torch.Tensor:
        return self.net(operand_pair)


class _SlotLatentBase(nn.Module):
    """Shared implementation for the discrete-slot models (A and B).

    State: four explicit slots ``z_{t,r} in R^53`` (212-dim latent budget),
    initialized as ``z_{0,r} = Write(one_hot(0))``. Operand retrieval and the
    operation dispatch (SET / SWAP / READ / ADD) are hard-routed by the
    instruction token's one-hot address fields; untouched slots satisfy
    ``||Delta z_r|| = 0`` by construction.
    """

    def __init__(self, alu_mode: str) -> None:
        super().__init__()
        if alu_mode not in ("fixed", "learned"):
            raise ValueError(f"alu_mode must be 'fixed' or 'learned', got {alu_mode!r}")
        self.alu_mode = alu_mode
        self.num_registers = NUM_REGISTERS
        self.num_values = NUM_VALUES
        self.slot_dim = SLOT_DIM
        self.input_dim = INSTRUCTION_DIM

        self.decoder = SharedSlotDecoder()
        self.writer = SharedSlotWriter()
        self.alu: Optional[LearnedALU] = LearnedALU() if alu_mode == "learned" else None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        one_hot_zero = torch.zeros(self.num_values, device=device)
        one_hot_zero[0] = 1.0
        slot_init = self.writer(one_hot_zero)  # (slot_dim,)
        return slot_init.view(1, 1, self.slot_dim).repeat(batch_size, self.num_registers, 1)

    def forward_step(
        self,
        x_t: torch.Tensor,
        state_prev: torch.Tensor,
    ):
        """Single registered transition; returns ``(state_next, step_logits)``."""
        e_op = x_t[:, 0:4]  # (B, 4): SET=0, SWAP=1, READ=2, ADD=3
        e_dest = x_t[:, 4:8]  # (B, 4)
        e_src = x_t[:, 8:12]  # (B, 4)
        e_val = x_t[:, 12:22]  # (B, 10)

        # Pre-update decoded operand distributions (all registers, for selection).
        p_prev = torch.softmax(self.decoder(state_prev), dim=-1)  # (B, 4, 10)
        p_dest = torch.einsum("br,brd->bd", e_dest, p_prev)
        p_src = torch.einsum("br,brd->bd", e_src, p_prev)

        if self.alu is None:
            p_add = circular_convolution(p_dest, p_src)
        else:
            p_add = torch.softmax(self.alu(torch.cat([p_dest, p_src], dim=-1)), dim=-1)

        w_set = e_op[:, 0:1]
        w_swap = e_op[:, 1:2]
        w_read = e_op[:, 2:3]
        w_add = e_op[:, 3:4]

        # Writer payload: e(v) on SET, p_sum on ADD; zero vector otherwise (masked out).
        write_input = w_set * e_val + w_add * p_add  # (B, 10)
        z_written = self.writer(write_input)  # (B, 53)

        u_a_raw = torch.einsum("br,brd->bd", e_dest, state_prev)
        u_b_raw = torch.einsum("br,brd->bd", e_src, state_prev)

        mask_dest = e_dest.unsqueeze(-1)  # (B, 4, 1)
        mask_src = e_src.unsqueeze(-1)

        # SET / ADD: write exclusively to the destination slot.
        state_set_add = (1.0 - mask_dest) * state_prev + mask_dest * z_written.unsqueeze(1)

        # SWAP: atomic exchange from the pre-update snapshot (self-swap is a no-op).
        keep_mask = (1.0 - mask_dest) * (1.0 - mask_src)
        state_swap = (
            keep_mask * state_prev
            + mask_dest * u_b_raw.unsqueeze(1)
            + mask_src * (1.0 - mask_dest) * u_a_raw.unsqueeze(1)
        )

        w_set_add = w_set + w_add  # (B, 1)
        state_next = (
            w_set_add.unsqueeze(-1) * state_set_add
            + w_swap.unsqueeze(-1) * state_swap
            + w_read.unsqueeze(-1) * state_prev
        )

        step_logits = self.decoder(state_next)  # (B, 4, 10)
        return state_next, step_logits

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> HeldOutCandidateOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = []
        for t in range(seq_len):
            state, step_logits = self.forward_step(inputs[:, t, :], state)
            step_logits_list.append(step_logits)

        return HeldOutCandidateOutput(
            step_logits=torch.stack(step_logits_list, dim=1),
            final_state=state,
        )


class SCLMModel(_SlotLatentBase):
    """Model A: Slot-Cyclic Latent Model (8,255 trainable parameters)."""

    def __init__(self) -> None:
        super().__init__(alu_mode="fixed")


class SlotMLPModel(_SlotLatentBase):
    """Model B: Slot-MLP control with the learned ALU (10,249 trainable parameters)."""

    def __init__(self) -> None:
        super().__init__(alu_mode="learned")


class _MonolithicMGUBase(nn.Module):
    """Shared implementation for the monolithic-state models (C and D).

    Reuses the existing Model E architecture components: ``ObservationEncoder``
    (over ``[x_t, f_ALU] in R^32``), ``NonlinearCandidateMGUCore``, and
    ``TypedDecisionHead``. The per-step ALU feature is computed from the
    pre-transition head distributions and injected through the encoder only.
    """

    def __init__(self, alu_mode: str) -> None:
        super().__init__()
        if alu_mode not in ("fixed", "learned"):
            raise ValueError(f"alu_mode must be 'fixed' or 'learned', got {alu_mode!r}")
        self.alu_mode = alu_mode
        self.num_registers = NUM_REGISTERS
        self.num_values = NUM_VALUES
        self.state_dim = STATE_DIM

        self.encoder = ObservationEncoder(
            input_dim=INSTRUCTION_DIM + NUM_VALUES,
            hidden_dim=ENCODER_HIDDEN_DIM,
            obs_dim=ENCODER_OUT_DIM,
        )
        self.core = NonlinearCandidateMGUCore(
            state_dim=STATE_DIM,
            obs_dim=ENCODER_OUT_DIM,
            hidden_dim=CANDIDATE_HIDDEN_DIM,
        )
        self.head = TypedDecisionHead(
            state_dim=STATE_DIM,
            num_registers=NUM_REGISTERS,
            num_values=NUM_VALUES,
        )
        self.alu: Optional[LearnedALU] = LearnedALU() if alu_mode == "learned" else None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward_step(
        self,
        x_t: torch.Tensor,
        state_prev: torch.Tensor,
    ):
        """Single registered transition; returns ``(state_next, step_logits)``."""
        e_dest = x_t[:, 4:8]
        e_src = x_t[:, 8:12]
        w_add = x_t[:, 3:4]

        # Pre-state read from the shared head.
        p_prev = torch.softmax(self.head(state_prev), dim=-1)  # (B, 4, 10)
        p_dest = torch.einsum("br,brd->bd", e_dest, p_prev)
        p_src = torch.einsum("br,brd->bd", e_src, p_prev)

        if self.alu is None:
            f_alu = circular_convolution(p_dest, p_src)
        else:
            f_alu = torch.softmax(self.alu(torch.cat([p_dest, p_src], dim=-1)), dim=-1)
        f_alu = w_add * f_alu  # zero feature on non-ADD transitions

        e_t = self.encoder(torch.cat([x_t, f_alu], dim=-1))  # (B, 256)
        state_next = self.core(state_prev, e_t)  # (B, 212)
        step_logits = self.head(state_next)  # (B, 4, 10)
        return state_next, step_logits

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> HeldOutCandidateOutput:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        state = initial_state if initial_state is not None else self.initial_state(batch_size, device)

        step_logits_list = []
        for t in range(seq_len):
            state, step_logits = self.forward_step(inputs[:, t, :], state)
            step_logits_list.append(step_logits)

        return HeldOutCandidateOutput(
            step_logits=torch.stack(step_logits_list, dim=1),
            final_state=state,
        )


class CyclicMGUControlModel(_MonolithicMGUBase):
    """Model C: monolithic control with the fixed convolutional ALU (227,062 parameters)."""

    def __init__(self) -> None:
        super().__init__(alu_mode="fixed")


class MonolithicMLPControlModel(_MonolithicMGUBase):
    """Model D: monolithic control with the learned ALU (229,056 trainable parameters)."""

    def __init__(self) -> None:
        super().__init__(alu_mode="learned")


CANDIDATE_MODELS: Dict[str, Type[nn.Module]] = {
    "A": SCLMModel,
    "B": SlotMLPModel,
    "C": CyclicMGUControlModel,
    "D": MonolithicMLPControlModel,
}

EXACT_PARAMETER_COUNTS: Dict[str, int] = {
    "A": 8255,
    "B": 10249,
    "C": 227062,
    "D": 229056,
}


def build_candidate(model_id: str) -> nn.Module:
    """Instantiates a registered candidate by identifier ('A', 'B', 'C', 'D')."""
    if model_id not in CANDIDATE_MODELS:
        raise ValueError(
            f"unknown candidate id {model_id!r}; expected one of {sorted(CANDIDATE_MODELS)}"
        )
    return CANDIDATE_MODELS[model_id]()
