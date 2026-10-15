"""Phase 1.6 Model Implementations and Deterministic Reference.

Implements the three candidate architectures and one control for Phase 1.6:
1. Candidate 1 (Slot-MGU): Discrete slot memory with generic MLP ALU (223,414 trainable parameters).
2. Candidate 2 (Cyclic-MGU): Monolithic continuous state with fixed cyclic harmonic ALU (219,018 trainable parameters).
3. Candidate 3 (MSM): Parameter-free deterministic analytical reference (40-dim state, 0 trainable parameters).
4. Control 2 (Vanilla GRU): Parameter-envelope-compliant recurrent baseline with dedicated Stage 1 head (220,874 trainable parameters).

Conforms strictly to the Phase 1.6 architectural specifications in docs/reports/phase_1_6_architectural_reformulation_proposal.md.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Phase16ModelOutput(NamedTuple):
    """Container for Phase 1.6 sequential model predictions."""
    logits: torch.Tensor  # Terminal logits (B, num_classes) or (B, num_registers, num_classes)
    final_state: torch.Tensor  # Final recurrent state
    step_logits: Optional[torch.Tensor] = None  # Step-wise logits across rollout


def get_harmonic_basis(
    num_values: int = 10,
    num_frequencies: int = 5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generates the fixed 5-harmonic cyclic embedding basis Phi(v) in R^10 for v in {0..num_values-1}.

    For each frequency k in {1..num_frequencies}:
        phi_k(v) = [cos(2*pi*k*v / num_values), sin(2*pi*k*v / num_values)]
    Phi(v) is the concatenated vector of shape (num_values, 2 * num_frequencies).
    """
    vals = torch.arange(num_values, dtype=dtype, device=device).unsqueeze(1)  # (V, 1)
    k = torch.arange(1, num_frequencies + 1, dtype=dtype, device=device).unsqueeze(0)  # (1, K)
    theta = 2.0 * np.pi * vals * k / float(num_values)  # (V, K)
    cos_t = torch.cos(theta).unsqueeze(-1)
    sin_t = torch.sin(theta).unsqueeze(-1)
    basis = torch.cat([cos_t, sin_t], dim=-1).view(num_values, 2 * num_frequencies)
    return basis


def project_to_unit_circle(
    x: torch.Tensor,
    num_frequencies: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Projects each 2D harmonic frequency block k in {1..num_frequencies} to the unit circle S^1.

    Args:
        x: Tensor of shape (..., 2 * num_frequencies).
        num_frequencies: Number of 2D harmonic blocks (default: 5).
        eps: Minimum norm clamping floor to prevent division by zero (default: 1e-6).

    Returns:
        Tensor of identical shape with unit-norm 2D blocks.
    """
    orig_shape = x.shape
    x_blocks = x.view(-1, num_frequencies, 2)
    norms = torch.clamp(torch.norm(x_blocks, p=2, dim=-1, keepdim=True), min=eps)
    x_norm = x_blocks / norms
    return x_norm.view(orig_shape)


def cyclic_alu(
    u_a: torch.Tensor,
    u_b: torch.Tensor,
    num_frequencies: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes exact modular addition across K 2D Givens rotation blocks.

    Given u_a^(k) = [cos alpha_k, sin alpha_k] and u_b^(k) = [cos beta_k, sin beta_k]:
        c^(k) = [cos(alpha_k + beta_k), sin(alpha_k + beta_k)]
              = [u_a1*u_b1 - u_a2*u_b2, u_a2*u_b1 + u_a1*u_b2]

    Args:
        u_a: First operand harmonic representation of shape (..., 2 * num_frequencies).
        u_b: Second operand harmonic representation of shape (..., 2 * num_frequencies).
        num_frequencies: Number of 2D harmonic blocks (default: 5).
        eps: Unit-norm clamping floor (default: 1e-6).

    Returns:
        Result harmonic vector of shape (..., 2 * num_frequencies).
    """
    orig_shape = u_a.shape
    a = u_a.view(-1, num_frequencies, 2)
    b = u_b.view(-1, num_frequencies, 2)

    a1, a2 = a[..., 0], a[..., 1]
    b1, b2 = b[..., 0], b[..., 1]

    c1 = a1 * b1 - a2 * b2
    c2 = a2 * b1 + a1 * b2
    c = torch.stack([c1, c2], dim=-1)

    norms = torch.clamp(torch.norm(c, p=2, dim=-1, keepdim=True), min=eps)
    c_norm = c / norms
    return c_norm.view(orig_shape)


# ===========================================================================
# Candidate 1: Slot-MGU (Discrete Slots + Learned MLP ALU)
# ===========================================================================

class SlotMGUPhase16Model(nn.Module):
    """Candidate 1: Explicitly Addressed Slot Memory with Generic MLP ALU.

    Architecture (Section 4.1 of proposal):
    - 4 explicitly addressed register slots (M=4), dimension d_slot = 53 per slot.
    - Total state dimension: 4 * 53 = 212.
    - Unified instruction representation: I_t = [u_a, u_b, x_t] in R^128.
    - Candidate MLP: Linear(128 -> 612) + GELU + Linear(612 -> 53) [111,437 params].
    - Convex-gate MLP: Linear(128 -> 612) + GELU + Linear(612 -> 53) [111,437 params].
    - Dedicated 10-class Head: Linear(53 -> 10) [540 params].
    - Total Trainable Parameters: exactly 223,414.

    Operation Semantics:
    - SET & ADD: Convex update s_{t, r_dest} = (1 - g_t) * s_{t-1, r_dest} + g_t * c_t.
      Non-addressed slots maintain exact mathematical identity: ||Delta s_r|| = 0.0.
    - SWAP: Atomic exchange of slot vectors s_{t, r_dest} = s_{t-1, r_src}, s_{t, r_src} = s_{t-1, r_dest}.
      Unswapped slots maintain exact identity: ||Delta s_r|| = 0.0.
    - READ: Zero state mutation across all slots (S_t = S_{t-1}).
      Head maps retrieved destination slot to class logits.
    """

    def __init__(
        self,
        num_registers: int = 4,
        slot_dim: int = 53,
        input_dim: int = 22,
        hidden_dim: int = 612,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.num_registers = num_registers
        self.slot_dim = slot_dim
        self.input_dim = input_dim
        self.state_dim = num_registers * slot_dim  # 212
        self.num_classes = num_classes

        # Unified input dimension: u_a (53) + u_b (53) + x_t (22) = 128
        self.unified_input_dim = slot_dim + slot_dim + input_dim  # 128

        self.mlp_cand = nn.Sequential(
            nn.Linear(self.unified_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.mlp_gate = nn.Sequential(
            nn.Linear(self.unified_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.head = nn.Linear(slot_dim, num_classes)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count (expected: 223,414)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialized discrete slot state S_0 of shape (B, 4, 53)."""
        return torch.zeros((batch_size, self.num_registers, self.slot_dim), device=device)

    def forward_step(
        self,
        x_t: torch.Tensor,
        state_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes a single-step recurrent transition.

        Args:
            x_t: Tensor of shape (B, 22).
            state_prev: Tensor of shape (B, 4, 53).

        Returns:
            state_next: Tensor of shape (B, 4, 53).
            step_logits: Categorical logits for destination slot of shape (B, 10).
        """
        e_op = x_t[:, 0:4]      # (B, 4): SET=0, SWAP=1, READ=2, ADD=3
        e_dest = x_t[:, 4:8]    # (B, 4): r_dest one-hot
        e_src = x_t[:, 8:12]    # (B, 4): r_src one-hot (or 0_4)

        # 1. Discrete hard multiplexing for operand retrieval
        # u_a = S_{t-1, r_dest} in R^(B, 53)
        u_a = torch.einsum("br,brd->bd", e_dest, state_prev)

        # u_b = S_{t-1, r_src} if op in {ADD, SWAP} else 0_53
        is_add_or_swap = (e_op[:, 1:2] + e_op[:, 3:4])  # (B, 1)
        u_b_raw = torch.einsum("br,brd->bd", e_src, state_prev)
        u_b = is_add_or_swap * u_b_raw

        # 2. Unified concatenated input vector I_t in R^(B, 128)
        I_t = torch.cat([u_a, u_b, x_t], dim=-1)

        # 3. Candidate and convex gate computation
        c_t = self.mlp_cand(I_t)                          # (B, 53)
        g_t = torch.sigmoid(self.mlp_gate(I_t))           # (B, 53)

        # 4. State updates by operation branch
        mask_dest = e_dest.unsqueeze(-1)  # (B, 4, 1)
        mask_src = e_src.unsqueeze(-1)    # (B, 4, 1)

        # SET & ADD branch: convex update exclusively on destination slot
        s_dest_new = (1.0 - g_t) * u_a + g_t * c_t  # (B, 53)
        state_set_add = (1.0 - mask_dest) * state_prev + mask_dest * s_dest_new.unsqueeze(1)

        # SWAP branch: atomic discrete exchange with bitwise preservation
        keep_mask = (1.0 - mask_dest) * (1.0 - mask_src)
        state_swap = (
            keep_mask * state_prev
            + mask_dest * u_b_raw.unsqueeze(1)
            + mask_src * (1.0 - mask_dest) * u_a.unsqueeze(1)
        )

        # Operation gating weights (mutually exclusive one-hot)
        w_set_add = (e_op[:, 0:1] + e_op[:, 3:4]).unsqueeze(-1)  # (B, 1, 1)
        w_swap = e_op[:, 1:2].unsqueeze(-1)                       # (B, 1, 1)
        w_read = e_op[:, 2:3].unsqueeze(-1)                       # (B, 1, 1)

        state_next = w_set_add * state_set_add + w_swap * state_swap + w_read * state_prev

        # 5. Readout of post-transition destination slot
        u_a_post = torch.einsum("br,brd->bd", e_dest, state_next)
        step_logits = self.head(u_a_post)  # (B, 10)

        return state_next, step_logits

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> Phase16ModelOutput:
        """Executes full sequence rollout over horizon T."""
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        state = (
            initial_state
            if initial_state is not None
            else self.initial_state(batch_size, device)
        )

        step_logits_list: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            state, step_logits = self.forward_step(x_t, state)
            if return_step_diagnostics:
                step_logits_list.append(step_logits)

        # Terminal logits from final transition
        terminal_logits = step_logits
        step_logits_t = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return Phase16ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_t,
        )


# ===========================================================================
# Candidate 2: Cyclic-MGU (Monolithic State + Cyclic ALU)
# ===========================================================================

class CyclicMGUPhase16Model(nn.Module):
    """Candidate 2: Monolithic State with Fixed Five-Harmonic Cyclic ALU.

    Architecture (Section 4.2 of proposal):
    - Monolithic continuous state vector S_t in R^212.
    - Fixed 5-harmonic cyclic representation Phi(v) in R^10.
    - Parameter-free cyclic rotation ALU across K=5 2D Givens rotation blocks.
    - Readout query network MLP_read: Linear(216 -> 128) + ReLU + Linear(128 -> 10) [29,066 params].
    - Dual-payload write candidate network MLP_write: Linear(32 -> 212) + ReLU + Linear(212 -> 212) [52,152 params].
    - Recurrent convex gate network MLP_gate: Linear(436 -> 212) + ReLU + Linear(212 -> 212) [137,800 params].
    - Total Trainable Parameters: exactly 219,018.

    Operation Semantics:
    - Pre-update snapshot retrieval extracts u_a, u_b from S_{t-1} prior to mutation.
    - SWAP atomically provides replacement coordinates for both destination and source.
    - READ clamps update gate to zero (S_t = S_{t-1}).
    - State is guaranteed strictly bounded in [-1, 1]^212 via recurrent convex combination.
    """

    def __init__(
        self,
        state_dim: int = 212,
        num_registers: int = 4,
        num_frequencies: int = 5,
        num_classes: int = 10,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_registers = num_registers
        self.num_frequencies = num_frequencies
        self.harmonic_dim = 2 * num_frequencies  # 10
        self.num_classes = num_classes
        self.eps = eps

        # Fixed harmonic basis Phi(v) for v in {0..9} in R^(10, 10)
        self.register_buffer("basis", get_harmonic_basis(num_classes, num_frequencies))

        # 1. Readout query network: [S_{t-1}, e(r)] in R^(212 + 4) = R^216 -> R^10
        self.mlp_read = nn.Sequential(
            nn.Linear(state_dim + num_registers, 128),
            nn.ReLU(),
            nn.Linear(128, self.harmonic_dim),
        )

        # 2. Write candidate network: [c1, r1, c2, r2, e_op] in R^(10 + 4 + 10 + 4 + 4) = R^32 -> R^212
        write_in_dim = self.harmonic_dim + num_registers + self.harmonic_dim + num_registers + 4  # 32
        self.mlp_write = nn.Sequential(
            nn.Linear(write_in_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

        # 3. Recurrent convex gate network: [S_{t-1}, C_t, e_inst] in R^(212 + 212 + 12) = R^436 -> R^212
        gate_in_dim = state_dim + state_dim + 12  # 436
        self.mlp_gate = nn.Sequential(
            nn.Linear(gate_in_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

    def count_parameters(self) -> int:
        """Returns total trainable parameter count (expected: 219,018)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialized monolithic state S_0 of shape (B, 212)."""
        return torch.zeros((batch_size, self.state_dim), device=device)

    def forward_step(
        self,
        x_t: torch.Tensor,
        state_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes a single-step recurrent transition."""
        e_op = x_t[:, 0:4]      # (B, 4)
        e_dest = x_t[:, 4:8]    # (B, 4)
        e_src = x_t[:, 8:12]    # (B, 4)
        e_val = x_t[:, 12:22]   # (B, 10)
        e_inst = x_t[:, 0:12]   # (B, 12)

        # 1. Pre-update snapshot operand retrieval from monolithic state S_{t-1}
        u_a_raw = self.mlp_read(torch.cat([state_prev, e_dest], dim=-1))
        u_b_raw = self.mlp_read(torch.cat([state_prev, e_src], dim=-1))
        u_a = project_to_unit_circle(u_a_raw, self.num_frequencies, self.eps)
        u_b = project_to_unit_circle(u_b_raw, self.num_frequencies, self.eps)

        # 2. Dual write payloads
        val_embed = torch.einsum("bv,vd->bd", e_val, self.basis)
        c1_alu = cyclic_alu(u_a, u_b, self.num_frequencies, self.eps)

        w_set = e_op[:, 0:1]
        w_swap = e_op[:, 1:2]
        w_read = e_op[:, 2:3]
        w_add = e_op[:, 3:4]

        # Primary payload (c1, r1)
        c1 = w_set * val_embed + w_add * c1_alu + w_swap * u_b + w_read * u_a
        r1 = e_dest

        # Secondary payload (c2, r2) — active only during SWAP
        c2 = w_swap * u_a
        r2 = w_swap * e_src

        # 3. Write candidate generation
        I_write = torch.cat([c1, r1, c2, r2, e_op], dim=-1)  # (B, 32)
        C_t = torch.tanh(self.mlp_write(I_write))             # (B, 212)

        # 4. Convex gate generation
        I_gate = torch.cat([state_prev, C_t, e_inst], dim=-1)  # (B, 436)
        g_raw = torch.sigmoid(self.mlp_gate(I_gate))           # (B, 212)
        # Clamped to zero on READ
        g_t = (1.0 - w_read) * g_raw

        # 5. Recurrent convex combination
        state_next = (1.0 - g_t) * state_prev + g_t * C_t      # (B, 212)

        # 6. Post-transition destination query and prototype bank decoding
        u_dest_post = project_to_unit_circle(
            self.mlp_read(torch.cat([state_next, e_dest], dim=-1)),
            self.num_frequencies,
            self.eps,
        )
        step_logits = torch.einsum("bd,vd->bv", u_dest_post, self.basis)  # (B, 10)

        return state_next, step_logits

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> Phase16ModelOutput:
        """Executes full sequence rollout over horizon T."""
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        state = (
            initial_state
            if initial_state is not None
            else self.initial_state(batch_size, device)
        )

        step_logits_list: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            state, step_logits = self.forward_step(x_t, state)
            if return_step_diagnostics:
                step_logits_list.append(step_logits)

        terminal_logits = step_logits
        step_logits_t = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return Phase16ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_t,
        )


# ===========================================================================
# Control 2: Vanilla GRU (Stage 1 Dedicated 10-Class Head)
# ===========================================================================

class VanillaGRUPhase16Model(nn.Module):
    """Control 2: Standard GRU Baseline parameter-envelope-compliant for Phase 1.6.

    Architecture (Section 4.4 & PI Decision 5):
    - Observation Encoder (22 -> 64 -> 256):
        Linear(22 -> 64) [1,472 params] + GELU + Linear(64 -> 256) [16,640 params] + LayerNorm(256) [512 params]
        Subtotal: 18,624 params.
    - Standard PyTorch GRU Cell (hidden_size = 160):
        nn.GRUCell(256, 160) [200,640 params].
    - Dedicated Stage 1 Ten-Class Decision Head:
        Linear(160 -> 10) [1,610 params].
    - Total Trainable Parameters: exactly 220,874.
    """

    def __init__(
        self,
        input_dim: int = 22,
        encoder_hidden_dim: int = 64,
        encoder_out_dim: int = 256,
        gru_hidden_dim: int = 160,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(encoder_hidden_dim, encoder_out_dim),
            nn.LayerNorm(encoder_out_dim),
        )
        self.gru_cell = nn.GRUCell(
            input_size=encoder_out_dim,
            hidden_size=gru_hidden_dim,
        )
        self.head = nn.Linear(gru_hidden_dim, num_classes)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count (expected: 220,874)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialized hidden state h_0 of shape (B, 160)."""
        return torch.zeros((batch_size, self.gru_hidden_dim), device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> Phase16ModelOutput:
        """Executes full sequence rollout over horizon T."""
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        h = (
            initial_state
            if initial_state is not None
            else self.initial_state(batch_size, device)
        )

        step_logits_list: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            e_t = self.encoder(x_t)
            h = self.gru_cell(e_t, h)
            if return_step_diagnostics:
                step_logits = self.head(h)
                step_logits_list.append(step_logits)

        terminal_logits = self.head(h)
        step_logits_t = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return Phase16ModelOutput(
            logits=terminal_logits,
            final_state=h,
            step_logits=step_logits_t,
        )


# ===========================================================================
# Candidate 3: MSM Deterministic Analytical Reference
# ===========================================================================

class MSMDeterministicReference(nn.Module):
    """Candidate 3: Modular Slot Machine (Parameter-Free Deterministic Analytical Reference).

    Architecture (Section 4.3 & PI Decision 1):
    - 4 discrete register slots (M=4), each holding 10 cyclic harmonic value coordinates.
    - Total state dimension: 4 * 10 = 40.
    - Exact parameter-free cyclic arithmetic via 2D Givens rotations.
    - Discrete register routing and unit-norm projection.
    - Output decision via prototype cosine bank argmax.
    - Total Trainable Parameters: exactly 0.
    """

    def __init__(
        self,
        num_registers: int = 4,
        num_frequencies: int = 5,
        num_classes: int = 10,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_registers = num_registers
        self.num_frequencies = num_frequencies
        self.harmonic_dim = 2 * num_frequencies  # 10
        self.state_dim = num_registers * self.harmonic_dim  # 40
        self.num_classes = num_classes
        self.eps = eps

        # Fixed prototype cosine bank Phi(c) in R^(10, 10), non-trainable
        self.register_buffer("basis", get_harmonic_basis(num_classes, num_frequencies))

    def count_parameters(self) -> int:
        """Returns trainable parameters count (strictly 0)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initial state S_0 where all 4 slots are initialized canonically to Phi(0). Shape: (B, 4, 10)."""
        # basis[0] has shape (10,)
        phi_0 = self.basis[0].to(device)
        return phi_0.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_registers, 1)

    def forward_step(
        self,
        x_t: torch.Tensor,
        state_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes a single deterministic step."""
        e_op = x_t[:, 0:4]      # (B, 4)
        e_dest = x_t[:, 4:8]    # (B, 4)
        e_src = x_t[:, 8:12]    # (B, 4)
        e_val = x_t[:, 12:22]   # (B, 10)

        # 1. Extract operand representations
        u_a = torch.einsum("br,brd->bd", e_dest, state_prev)  # (B, 10)
        u_b = torch.einsum("br,brd->bd", e_src, state_prev)   # (B, 10)

        # 2. SET payload
        val_embed = torch.einsum("bv,vd->bd", e_val, self.basis)  # (B, 10)

        # 3. ADD payload
        c_add = cyclic_alu(u_a, u_b, self.num_frequencies, self.eps)

        # Operation indicators
        w_set = e_op[:, 0:1].unsqueeze(-1)    # (B, 1, 1)
        w_swap = e_op[:, 1:2].unsqueeze(-1)   # (B, 1, 1)
        w_read = e_op[:, 2:3].unsqueeze(-1)   # (B, 1, 1)
        w_add = e_op[:, 3:4].unsqueeze(-1)    # (B, 1, 1)

        # Transitions:
        mask_dest = e_dest.unsqueeze(-1)  # (B, 4, 1)
        mask_src = e_src.unsqueeze(-1)    # (B, 4, 1)

        # SET: dest slot becomes val_embed
        state_set = (1.0 - mask_dest) * state_prev + mask_dest * val_embed.unsqueeze(1)

        # ADD: dest slot becomes c_add
        state_add = (1.0 - mask_dest) * state_prev + mask_dest * c_add.unsqueeze(1)

        # SWAP: atomic swap with bitwise preservation
        keep_mask = (1.0 - mask_dest) * (1.0 - mask_src)
        state_swap = (
            keep_mask * state_prev
            + mask_dest * u_b.unsqueeze(1)
            + mask_src * (1.0 - mask_dest) * u_a.unsqueeze(1)
        )

        # Combine
        state_next = (
            w_set * state_set
            + w_add * state_add
            + w_swap * state_swap
            + w_read * state_prev
        )

        # Output logits on destination slot
        u_dest_post = torch.einsum("br,brd->bd", e_dest, state_next)
        step_logits = torch.einsum("bd,vd->bv", u_dest_post, self.basis)

        return state_next, step_logits

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> Phase16ModelOutput:
        """Executes deterministic sequence rollout over horizon T."""
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        state = (
            initial_state
            if initial_state is not None
            else self.initial_state(batch_size, device)
        )

        step_logits_list: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            state, step_logits = self.forward_step(x_t, state)
            if return_step_diagnostics:
                step_logits_list.append(step_logits)

        terminal_logits = step_logits
        step_logits_t = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return Phase16ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_t,
        )


# ===========================================================================
# Candidate 1H: Deterministic Harmonic Register-Machine Execution Kernel
# ===========================================================================

class Candidate1HDeterministicKernel(nn.Module):
    """Candidate 1H Deterministic Execution Kernel.

    Standalone parameter-free register-machine execution kernel using the 5-frequency
    circular harmonic codebook on S^1.

    Architecture and Specifications:
    - 4 discrete register slots (M=4), each holding a 10-dimensional cyclic harmonic vector
      Phi(v) in R^10, formed by K=5 harmonic frequencies on S^1.
    - Initial state: All 4 registers initialized canonically to Phi(0) = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]^T,
      strictly matching the canonical register machine benchmark ground-state 0_4.
    - Exact arithmetic: 2D Givens rotation matrix multiplication modulo 10 (cyclic_alu).
    - Exact discrete routing: Einsum tensor contractions with one-hot address indicators.
    - Exact state isolation: Registers not modified by the active instruction satisfy
      ||Delta s_r|| == 0.000000.
    - Primary Readout: Non-parametric codebook matching via inner-product argmax against
      canonical basis Phi(0)..Phi(9). Zero learned temperature; zero contextual logits.
    - Total Trainable Parameters: exactly 0.

    Classification: Deterministic execution kernel / neuro-symbolic interpreter;
    not a trained neural model.
    """

    def __init__(
        self,
        num_registers: int = 4,
        num_frequencies: int = 5,
        num_classes: int = 10,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_registers = num_registers
        self.num_frequencies = num_frequencies
        self.harmonic_dim = 2 * num_frequencies  # 10
        self.state_dim = num_registers * self.harmonic_dim  # 40
        self.num_classes = num_classes
        self.eps = eps

        # Fixed prototype cosine bank Phi(c) in R^(10, 10), non-trainable
        self.register_buffer("basis", get_harmonic_basis(num_classes, num_frequencies))

    def count_parameters(self) -> int:
        """Returns trainable parameters count (strictly 0)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Initial state S_0 where all 4 slots are initialized canonically to Phi(0). Shape: (B, 4, 10)."""
        if device is None:
            device = self.basis.device
        phi_0 = self.basis[0].to(device)
        return phi_0.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_registers, 1)

    def encode_value(self, val: int) -> torch.Tensor:
        """Returns the canonical 10-dimensional harmonic codeword Phi(val)."""
        if val < 0 or val >= self.num_classes:
            raise ValueError(f"Value {val} outside valid range [0, {self.num_classes - 1}]")
        return self.basis[val].clone()

    def decode_harmonic(self, slot_harm: torch.Tensor) -> torch.Tensor:
        """Decodes harmonic representation(s) to integer class(es) via inner product with basis.

        Args:
            slot_harm: Tensor of shape (..., 10).

        Returns:
            Decoded integer class tensor of shape (...).
        """
        orig_shape = slot_harm.shape[:-1]
        flat_harm = slot_harm.reshape(-1, self.harmonic_dim)
        dots = torch.matmul(flat_harm, self.basis.t())  # (N, 10)
        preds = dots.argmax(dim=-1)
        return preds.reshape(orig_shape)

    def execute_instruction(
        self,
        state_prev: torch.Tensor,
        op_type: int,
        reg_dest: int,
        reg_src: int = -1,
        val: int = -1,
    ) -> Tuple[torch.Tensor, int]:
        """Executes a single discrete instruction on an individual state tensor of shape (4, 10).

        Returns:
            state_next: Updated state of shape (4, 10).
            read_val: Integer decoded value for destination register.
        """
        device = state_prev.device
        x_t = torch.zeros((1, 22), dtype=torch.float32, device=device)
        x_t[0, op_type] = 1.0
        x_t[0, 4 + reg_dest] = 1.0
        if reg_src >= 0:
            x_t[0, 8 + reg_src] = 1.0
        if val >= 0:
            x_t[0, 12 + val] = 1.0

        state_batched = state_prev.unsqueeze(0)
        state_next_batched, step_logits = self.forward_step(x_t, state_batched)
        read_val = int(step_logits.argmax(dim=-1).item())
        return state_next_batched.squeeze(0), read_val

    def forward_step(
        self,
        x_t: torch.Tensor,
        state_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes a single deterministic step across batch.

        Args:
            x_t: One-hot instruction tensor of shape (B, 22).
                 0:4 -> op_type (SET=0, SWAP=1, READ=2, ADD=3)
                 4:8 -> reg_dest
                 8:12 -> reg_src
                 12:22 -> immediate_val
            state_prev: Harmonic register state of shape (B, 4, 10).

        Returns:
            state_next: Updated register state of shape (B, 4, 10).
            step_logits: Codebook projection logits of shape (B, 10) on destination register.
        """
        e_op = x_t[:, 0:4]      # (B, 4)
        e_dest = x_t[:, 4:8]    # (B, 4)
        e_src = x_t[:, 8:12]    # (B, 4)
        e_val = x_t[:, 12:22]   # (B, 10)

        # 1. Extract operand representations
        u_a = torch.einsum("br,brd->bd", e_dest, state_prev)  # (B, 10)
        u_b = torch.einsum("br,brd->bd", e_src, state_prev)   # (B, 10)

        # 2. SET payload: immediate value embedding Phi(val)
        val_embed = torch.einsum("bv,vd->bd", e_val, self.basis)  # (B, 10)

        # 3. ADD payload: exact cyclic Givens rotation
        c_add = cyclic_alu(u_a, u_b, self.num_frequencies, self.eps)  # (B, 10)

        # Operation indicators
        w_set = e_op[:, 0:1].unsqueeze(-1)    # (B, 1, 1)
        w_swap = e_op[:, 1:2].unsqueeze(-1)   # (B, 1, 1)
        w_read = e_op[:, 2:3].unsqueeze(-1)   # (B, 1, 1)
        w_add = e_op[:, 3:4].unsqueeze(-1)    # (B, 1, 1)

        mask_dest = e_dest.unsqueeze(-1)  # (B, 4, 1)
        mask_src = e_src.unsqueeze(-1)    # (B, 4, 1)

        # SET: destination slot becomes val_embed; untouched slots preserved exactly
        state_set = (1.0 - mask_dest) * state_prev + mask_dest * val_embed.unsqueeze(1)

        # ADD: destination slot becomes c_add; untouched slots preserved exactly
        state_add = (1.0 - mask_dest) * state_prev + mask_dest * c_add.unsqueeze(1)

        # SWAP: atomic bidirectional swap; untouched slots preserved exactly
        # Handles self-swap (r_dest == r_src) cleanly as an exact no-op
        keep_mask = (1.0 - mask_dest) * (1.0 - mask_src)
        state_swap = (
            keep_mask * state_prev
            + mask_dest * u_b.unsqueeze(1)
            + mask_src * (1.0 - mask_dest) * u_a.unsqueeze(1)
        )

        # Combine operation branches
        # READ preserves all slots with zero mutation (state_prev)
        state_next = (
            w_set * state_set
            + w_add * state_add
            + w_swap * state_swap
            + w_read * state_prev
        )

        # Primary decision: project destination slot against basis codebook
        u_dest_post = torch.einsum("br,brd->bd", e_dest, state_next)
        step_logits = torch.einsum("bd,vd->bv", u_dest_post, self.basis)

        return state_next, step_logits

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_step_diagnostics: bool = False,
    ) -> Phase16ModelOutput:
        """Executes deterministic sequence rollout over horizon T.

        Args:
            inputs: Tensor of shape (B, T, 22).
            initial_state: Optional initial state of shape (B, 4, 10).
            return_step_diagnostics: Whether to return per-step logits.

        Returns:
            Phase16ModelOutput containing terminal logits, final state, and optional step logits.
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        state = (
            initial_state
            if initial_state is not None
            else self.initial_state(batch_size, device)
        )

        step_logits_list: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            state, step_logits = self.forward_step(x_t, state)
            if return_step_diagnostics:
                step_logits_list.append(step_logits)

        terminal_logits = step_logits
        step_logits_t = torch.stack(step_logits_list, dim=1) if return_step_diagnostics else None

        return Phase16ModelOutput(
            logits=terminal_logits,
            final_state=state,
            step_logits=step_logits_t,
        )
