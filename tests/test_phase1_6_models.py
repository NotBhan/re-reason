"""Independent unit and contract tests for Phase 1.6 model architectures.

Verifies:
1. Exact trainable parameter counts:
   - Candidate 1 (Slot-MGU): exactly 223,414.
   - Candidate 2 (Cyclic-MGU): exactly 219,018.
   - Control 2 (Vanilla GRU): exactly 220,874.
   - Candidate 3 (MSM): exactly 0.
2. Parameter envelope compliance: [203,722, 248,994].
3. State, input, intermediate, and output dimensions.
4. Operation contracts (SET, SWAP, READ, ADD), atomic SWAP, and non-mutating READ.
5. Structural slot preservation in Slot-MGU and MSM (||Delta s_r|| = 0.0 on unaddressed slots).
6. Bounded state dynamics in Cyclic-MGU (S_t in [-1, 1]^212).
7. Exhaustive algebraic verification of fixed Cyclic ALU across all 100 ordered pairs.
8. Synthetic gradient flow to all trainable parameters with ZERO optimizer execution.
9. Deterministic reference accuracy and instance isolation across seeds.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from re_reason.benchmarks.register_machine_benchmark import Phase16RegisterTask, Stage1PairPartition
from re_reason.core.register_machine_models import (
    CyclicMGUPhase16Model,
    MSMDeterministicReference,
    Phase16ModelOutput,
    SlotMGUPhase16Model,
    VanillaGRUPhase16Model,
    cyclic_alu,
    get_harmonic_basis,
    project_to_unit_circle,
)


# ---------------------------------------------------------------------------
# 1. Architecture, Parameter Accounting, and Dimension Contracts
# ---------------------------------------------------------------------------

def test_exact_parameter_counts_and_envelope_compliance() -> None:
    """Verifies that all 4 models match the exact proposal parameter counts and envelope."""
    # Envelope target: 226,358 +/- 10% -> [203,722, 248,994]
    envelope_min = 203722
    envelope_max = 248994

    # 1. Candidate 1: Slot-MGU
    slot_mgu = SlotMGUPhase16Model()
    p_slot = slot_mgu.count_parameters()
    assert p_slot == 223414, f"SlotMGU parameter count mismatch: expected 223414, got {p_slot}"
    assert envelope_min <= p_slot <= envelope_max

    # 2. Candidate 2: Cyclic-MGU
    cyclic_mgu = CyclicMGUPhase16Model()
    p_cyclic = cyclic_mgu.count_parameters()
    assert p_cyclic == 219018, f"CyclicMGU parameter count mismatch: expected 219018, got {p_cyclic}"
    assert envelope_min <= p_cyclic <= envelope_max

    # 3. Control 2: Vanilla GRU (Stage 1 Dedicated 10-Class Head)
    vanilla_gru = VanillaGRUPhase16Model()
    p_gru = vanilla_gru.count_parameters()
    assert p_gru == 220874, f"VanillaGRU parameter count mismatch: expected 220874, got {p_gru}"
    assert envelope_min <= p_gru <= envelope_max

    # 4. Candidate 3: MSM (Parameter-free deterministic reference)
    msm = MSMDeterministicReference()
    p_msm = msm.count_parameters()
    assert p_msm == 0, f"MSM must have exactly 0 trainable parameters, got {p_msm}"


def test_model_dimensions_and_initial_state_contracts() -> None:
    """Verifies state dimensions, shapes, and canonical initial states."""
    b = 4

    # 1. Slot-MGU: (B, 4, 53) -> flattened state dim = 212
    slot_mgu = SlotMGUPhase16Model()
    s0_slot = slot_mgu.initial_state(b, torch.device("cpu"))
    assert s0_slot.shape == (b, 4, 53)
    assert (s0_slot == 0.0).all()

    # 2. Cyclic-MGU: (B, 212)
    cyclic_mgu = CyclicMGUPhase16Model()
    s0_cyclic = cyclic_mgu.initial_state(b, torch.device("cpu"))
    assert s0_cyclic.shape == (b, 212)
    assert (s0_cyclic == 0.0).all()

    # 3. Vanilla GRU: (B, 160)
    vanilla_gru = VanillaGRUPhase16Model()
    s0_gru = vanilla_gru.initial_state(b, torch.device("cpu"))
    assert s0_gru.shape == (b, 160)
    assert (s0_gru == 0.0).all()

    # 4. MSM: (B, 4, 10) -> each slot initialized to Phi(0)
    msm = MSMDeterministicReference()
    s0_msm = msm.initial_state(b, torch.device("cpu"))
    assert s0_msm.shape == (b, 4, 10)
    basis = get_harmonic_basis()
    for r in range(4):
        assert torch.allclose(s0_msm[:, r], basis[0].unsqueeze(0).repeat(b, 1), atol=1e-6)


def test_instance_isolation_no_shared_parameters() -> None:
    """Verifies that separate instances of models do not share weights."""
    m1 = SlotMGUPhase16Model()
    m2 = SlotMGUPhase16Model()

    with torch.no_grad():
        m1.head.weight.fill_(1.0)
        m2.head.weight.fill_(2.0)

    assert not torch.equal(m1.head.weight, m2.head.weight)
    assert m1.head.weight[0, 0].item() == 1.0
    assert m2.head.weight[0, 0].item() == 2.0


# ---------------------------------------------------------------------------
# 2. Operation Contracts and State Transitions
# ---------------------------------------------------------------------------

def test_slot_mgu_operations_and_exact_slot_preservation() -> None:
    """Verifies SlotMGU operation routing and exact mathematical slot preservation (||Delta s_r|| = 0.0)."""
    task = Phase16RegisterTask()
    model = SlotMGUPhase16Model()
    device = torch.device("cpu")

    # Sequence of 4 instructions:
    # 0: SET r0 7
    # 1: SET r1 3
    # 2: SWAP r0 r1
    # 3: READ r0
    # 4: ADD r2 r1
    vecs = [
        task.encode_instruction(task.OP_SET, reg_dest=0, reg_src=-1, immediate_val=7),
        task.encode_instruction(task.OP_SET, reg_dest=1, reg_src=-1, immediate_val=3),
        task.encode_instruction(task.OP_SWAP, reg_dest=0, reg_src=1, immediate_val=-1),
        task.encode_instruction(task.OP_READ, reg_dest=0, reg_src=-1, immediate_val=-1),
        task.encode_instruction(task.OP_ADD, reg_dest=2, reg_src=1, immediate_val=-1),
    ]
    inputs = torch.from_numpy(np.stack(vecs, axis=0)).unsqueeze(0)  # (1, 5, 22)

    # Step 0: SET r0 7
    s0 = model.initial_state(1, device)
    s1, log1 = model.forward_step(inputs[:, 0], s0)
    assert log1.shape == (1, 10)
    # Slots 1, 2, 3 must maintain exact mathematical zero: ||s_r|| = 0.0
    assert s1[0, 1].norm().item() == 0.0
    assert s1[0, 2].norm().item() == 0.0
    assert s1[0, 3].norm().item() == 0.0
    assert s1[0, 0].norm().item() > 0.0  # slot 0 updated

    # Step 1: SET r1 3
    s2, log2 = model.forward_step(inputs[:, 1], s1)
    # Slot 0 must remain EXACTLY identical to s1[0, 0]
    assert torch.equal(s2[0, 0], s1[0, 0])
    assert s2[0, 2].norm().item() == 0.0
    assert s2[0, 3].norm().item() == 0.0

    # Step 2: SWAP r0 r1
    s3, log3 = model.forward_step(inputs[:, 2], s2)
    # r0 should now hold what r1 held; r1 should hold what r0 held
    assert torch.equal(s3[0, 0], s2[0, 1])
    assert torch.equal(s3[0, 1], s2[0, 0])
    # Slots 2 and 3 remain exact zero
    assert s3[0, 2].norm().item() == 0.0
    assert s3[0, 3].norm().item() == 0.0

    # Step 3: READ r0
    s4, log4 = model.forward_step(inputs[:, 3], s3)
    # READ must cause zero state mutation across all slots
    assert torch.equal(s4, s3)

    # Step 4: ADD r2 r1
    s5, log5 = model.forward_step(inputs[:, 4], s4)
    # r2 updated; r0, r1, r3 remain exactly unchanged
    assert torch.equal(s5[0, 0], s4[0, 0])
    assert torch.equal(s5[0, 1], s4[0, 1])
    assert torch.equal(s5[0, 3], s4[0, 3])
    assert s5[0, 3].norm().item() == 0.0


def test_cyclic_mgu_operations_and_bounded_dynamics() -> None:
    """Verifies CyclicMGU transitions, non-mutating READ, and strict state boundedness [-1, 1]^212."""
    task = Phase16RegisterTask()
    model = CyclicMGUPhase16Model()
    device = torch.device("cpu")

    # 1. READ non-mutating contract
    read_vec = task.encode_instruction(task.OP_READ, reg_dest=2, reg_src=-1, immediate_val=-1)
    read_inp = torch.from_numpy(read_vec).unsqueeze(0)  # (1, 22)
    state_rnd = torch.randn(1, 212).clamp(-1.0, 1.0)
    state_post_read, _ = model.forward_step(read_inp, state_rnd)
    assert torch.equal(state_post_read, state_rnd), "READ mutated the state in CyclicMGU"

    # 2. Strict state boundedness over 50 recurrent steps
    rng = np.random.default_rng(2026)
    seq_len = 50
    tokens = []
    for _ in range(seq_len):
        op = int(rng.integers(0, 4))
        dest = int(rng.integers(0, 4))
        src = int(rng.integers(0, 4)) if op in (1, 3) else -1
        val = int(rng.integers(0, 10)) if op == 0 else -1
        tokens.append(task.encode_instruction(op, dest, src, val))

    inputs_50 = torch.from_numpy(np.stack(tokens, axis=0)).unsqueeze(0)  # (1, 50, 22)
    out = model(inputs_50, return_step_diagnostics=True)

    assert out.final_state.shape == (1, 212)
    assert (out.final_state >= -1.0).all() and (out.final_state <= 1.0).all(), (
        f"CyclicMGU state exceeded [-1, 1] bounds: min={out.final_state.min()}, max={out.final_state.max()}"
    )
    assert out.logits.shape == (1, 10)
    assert out.step_logits.shape == (1, 50, 10)
    assert torch.isfinite(out.logits).all()


def test_vanilla_gru_forward_and_diagnostics() -> None:
    """Verifies VanillaGRU output shapes, terminal logits, and step diagnostics."""
    b, t = 4, 10
    inputs = torch.randn(b, t, 22)
    gru = VanillaGRUPhase16Model()

    out = gru(inputs, return_step_diagnostics=True)
    assert out.logits.shape == (b, 10)
    assert out.final_state.shape == (b, 160)
    assert out.step_logits.shape == (b, t, 10)
    assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# 3. Fixed Cyclic Arithmetic ALU Verification
# ---------------------------------------------------------------------------

def test_cyclic_alu_exhaustive_100_pairs() -> None:
    """Exhaustively verifies cyclic rotation ALU on all 100 ordered pairs (a, b) in {0..9}^2."""
    basis = get_harmonic_basis(num_values=10, num_frequencies=5)
    assert basis.shape == (10, 10)

    max_err = 0.0
    for a in range(10):
        for b in range(10):
            u_a = basis[a].unsqueeze(0)
            u_b = basis[b].unsqueeze(0)

            # ALU rotation
            c = cyclic_alu(u_a, u_b, num_frequencies=5)
            expected = basis[(a + b) % 10].unsqueeze(0)

            err = (c - expected).abs().max().item()
            max_err = max(max_err, err)

            # Unit-norm verification on all 5 2D blocks
            c_blocks = c.view(5, 2)
            norms = torch.norm(c_blocks, p=2, dim=-1)
            assert torch.allclose(norms, torch.ones(5), atol=1e-5), f"Block norms not 1.0 for ({a}, {b})"

            # Classification prediction via prototype cosine bank
            logits = c @ basis.t()
            pred = logits.argmax(dim=-1).item()
            assert pred == (a + b) % 10, f"ALU prediction mismatch for {a} + {b}: got {pred}"

    # Documented numerical tolerance check (< 1e-5 in float32)
    assert max_err < 1e-5, f"Maximum ALU rotation error {max_err:.2e} exceeded 1e-5 tolerance"


def test_cyclic_alu_edge_cases_and_properties() -> None:
    """Verifies wrap-around, self-addition, identity with zero, and commutativity."""
    basis = get_harmonic_basis()

    # 1. Identity with zero: a + 0 = a
    for a in range(10):
        res = cyclic_alu(basis[a].unsqueeze(0), basis[0].unsqueeze(0))
        assert torch.allclose(res, basis[a].unsqueeze(0), atol=1e-5)

    # 2. Self-addition: a + a = (2*a) % 10
    for a in range(10):
        res = cyclic_alu(basis[a].unsqueeze(0), basis[a].unsqueeze(0))
        assert torch.allclose(res, basis[(2 * a) % 10].unsqueeze(0), atol=1e-5)

    # 3. Commutativity: a + b = b + a
    for a in range(10):
        for b in range(10):
            res_ab = cyclic_alu(basis[a].unsqueeze(0), basis[b].unsqueeze(0))
            res_ba = cyclic_alu(basis[b].unsqueeze(0), basis[a].unsqueeze(0))
            assert torch.allclose(res_ab, res_ba, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Candidate 3 (MSM) Deterministic Analytical Reference
# ---------------------------------------------------------------------------

def test_msm_deterministic_reference_exact_100_pairs() -> None:
    """Verifies that MSM achieves exact 100/100 accuracy on Stage 1 canonical traces with zero drift."""
    msm = MSMDeterministicReference()
    task = Phase16RegisterTask()

    correct = 0
    for a in range(10):
        for b in range(10):
            inp, step_tgt, final_tgt = task.generate_stage1_trace(a, b)
            inp_t = torch.from_numpy(inp).unsqueeze(0)  # (1, 3, 22)

            out = msm(inp_t, return_step_diagnostics=True)
            pred = out.logits.argmax(dim=-1).item()
            expected = (a + b) % 10

            assert pred == expected, f"MSM failed on pair ({a}, {b}): got {pred}, expected {expected}"
            correct += 1

            # Exact slot preservation check:
            # Step 1: SET r0 a -> r1, r2, r3 must be Phi(0)
            # Step 2: SET r1 b -> r2, r3 must be Phi(0); r0 must be Phi(a)
            # Step 3: ADD r0 r1 -> r2, r3 must be Phi(0); r1 must be Phi(b)
            final_s = out.final_state[0]  # (4, 10)
            basis = msm.basis
            assert torch.allclose(final_s[1], basis[b], atol=1e-5), f"Slot 1 not preserved: {final_s[1]}"
            assert torch.allclose(final_s[2], basis[0], atol=1e-5), f"Slot 2 unaddressed drift: {final_s[2]}"
            assert torch.allclose(final_s[3], basis[0], atol=1e-5), f"Slot 3 unaddressed drift: {final_s[3]}"

    assert correct == 100


def test_msm_swaps_and_reads() -> None:
    """Verifies that MSM handles SWAP and READ with zero mutation and correct discrete routing."""
    msm = MSMDeterministicReference()
    task = Phase16RegisterTask()

    # SET r0 4, SET r1 8, SWAP r0 r1, READ r0
    vecs = [
        task.encode_instruction(task.OP_SET, reg_dest=0, reg_src=-1, immediate_val=4),
        task.encode_instruction(task.OP_SET, reg_dest=1, reg_src=-1, immediate_val=8),
        task.encode_instruction(task.OP_SWAP, reg_dest=0, reg_src=1, immediate_val=-1),
        task.encode_instruction(task.OP_READ, reg_dest=0, reg_src=-1, immediate_val=-1),
    ]
    inp_t = torch.from_numpy(np.stack(vecs, axis=0)).unsqueeze(0)
    out = msm(inp_t, return_step_diagnostics=True)

    # Post-swap & read: r0 should be 8, r1 should be 4
    pred = out.logits.argmax(dim=-1).item()
    assert pred == 8
    final_s = out.final_state[0]
    basis = msm.basis
    assert torch.allclose(final_s[0], basis[8], atol=1e-5)
    assert torch.allclose(final_s[1], basis[4], atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Gradient Pathways and Learning Contracts (Zero Training Runs)
# ---------------------------------------------------------------------------

def test_synthetic_gradient_flow_to_trainable_parameters() -> None:
    """Verifies gradient flow to all trainable parameters without executing optimizer steps."""
    b, t = 4, 3
    task = Phase16RegisterTask()
    partition = Stage1PairPartition()
    train_ds = partition.build_trace_split(partition.train_pairs[:b], task)

    inputs = train_ds.inputs[:b]  # (B, 3, 22)
    targets = train_ds.targets[:b, 0]  # (B,)

    # 1. Slot-MGU Gradient Check
    slot_mgu = SlotMGUPhase16Model()
    out_slot = slot_mgu(inputs)
    loss_slot = F.cross_entropy(out_slot.logits, targets)
    loss_slot.backward()
    for name, p in slot_mgu.named_parameters():
        assert p.requires_grad
        assert p.grad is not None, f"SlotMGU parameter {name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"SlotMGU parameter {name} has non-finite gradient"

    # 2. Cyclic-MGU Gradient Check
    cyclic_mgu = CyclicMGUPhase16Model()
    out_cyclic = cyclic_mgu(inputs)
    loss_cyclic = F.cross_entropy(out_cyclic.logits, targets)
    loss_cyclic.backward()
    for name, p in cyclic_mgu.named_parameters():
        assert p.requires_grad
        assert p.grad is not None, f"CyclicMGU parameter {name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"CyclicMGU parameter {name} has non-finite gradient"

    # 3. Vanilla GRU Gradient Check
    vanilla_gru = VanillaGRUPhase16Model()
    out_gru = vanilla_gru(inputs)
    loss_gru = F.cross_entropy(out_gru.logits, targets)
    loss_gru.backward()
    for name, p in vanilla_gru.named_parameters():
        assert p.requires_grad
        assert p.grad is not None, f"VanillaGRU parameter {name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"VanillaGRU parameter {name} has non-finite gradient"

    # 4. MSM Zero Gradient Check
    msm = MSMDeterministicReference()
    assert len(list(msm.parameters())) == 0
