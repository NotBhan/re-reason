"""Focused Independent Verification Test Suite for Candidate 1H Deterministic Execution Kernel.

Governed by Phase 1.6 Protocol Amendment 1 Authorization (Design A).
Research Objective: Deterministic execution correctness, NOT learned modular arithmetic.

Coverage:
1. Codebook validity: All 10 canonical codewords, exact unit-norm frequency blocks.
2. Exhaustive arithmetic: All 100 operand pairs (a, b) in {0..9}^2, decoded value == (a + b) % 10.
3. SET semantics: Destination receives specified value; untouched registers remain exactly unchanged.
4. SWAP semantics: Both addressed registers exchange values; untouched registers remain exactly unchanged.
5. ADD semantics: Only destination changes to modular sum; untouched registers remain unchanged.
6. READ semantics: Correct decoded output and exact zero state mutation.
7. Initial state: All four registers begin at canonical zero encoding Phi(0).
8. Composed execution: Interleaved traces verified against canonical benchmark interpreter oracle.
9. Numerical stability: Finite states, harmonic amplitude bounds, explicit float32 tolerance eps <= 1e-5.
10. Boundary cases: Self-swap, zero operands, values near modulus boundary (e.g. 9+1, 9+9), repeated additions.
11. Validation datasets: Verification on 16 Stage 1 VAL-SEEN pairs and 1,200 Stage 2 VAL sequences.
    Strictly firewalled: TEST split is NEVER generated, loaded, or inspected.
"""

import math
import numpy as np
import pytest
import torch

from re_reason.benchmarks.register_machine_benchmark import (
    Instruction,
    Phase16RegisterTask,
    Phase16SplitManager,
    Stage1PairPartition,
)
from re_reason.core.register_machine_models import (
    Candidate1HDeterministicKernel,
    get_harmonic_basis,
)


@pytest.fixture
def kernel() -> Candidate1HDeterministicKernel:
    """Fixture providing an instance of the Candidate 1H deterministic kernel."""
    k = Candidate1HDeterministicKernel()
    k.eval()
    return k


# ===========================================================================
# 1. Codebook Validity Test
# ===========================================================================

def test_codebook_validity_and_unit_norm_blocks(kernel: Candidate1HDeterministicKernel) -> None:
    """Asserts that all 10 canonical codewords have exact unit-norm blocks across all 5 frequencies."""
    basis = kernel.basis  # (10, 10)
    assert basis.shape == (10, 10), f"Expected basis shape (10, 10), got {basis.shape}"

    for v in range(10):
        codeword = basis[v]  # (10,)
        blocks = codeword.view(5, 2)

        # Independent calculation of expected trigonometric coordinates
        for k_idx in range(1, 6):
            expected_theta = 2.0 * math.pi * k_idx * v / 10.0
            expected_cos = math.cos(expected_theta)
            expected_sin = math.sin(expected_theta)

            actual_cos = blocks[k_idx - 1, 0].item()
            actual_sin = blocks[k_idx - 1, 1].item()

            assert abs(actual_cos - expected_cos) < 2e-6, f"Cos mismatch at v={v}, k={k_idx}"
            assert abs(actual_sin - expected_sin) < 2e-6, f"Sin mismatch at v={v}, k={k_idx}"

            block_norm = math.sqrt(actual_cos**2 + actual_sin**2)
            assert abs(block_norm - 1.0) < 2e-6, f"Unit norm violation at v={v}, k={k_idx}: norm={block_norm}"


# ===========================================================================
# 2. Exhaustive Arithmetic Test (All 100 Pairs)
# ===========================================================================

def test_exhaustive_modular_addition_100_pairs(kernel: Candidate1HDeterministicKernel) -> None:
    """Evaluates cyclic_alu across all 100 operand pairs (a, b) in {0..9}^2 against analytical ground truth."""
    basis = kernel.basis

    for a in range(10):
        for b in range(10):
            phi_a = basis[a].unsqueeze(0)  # (1, 10)
            phi_b = basis[b].unsqueeze(0)  # (1, 10)

            # Direct execution of cyclic ALU
            phi_c = kernel.forward_step(
                x_t=torch.tensor([[0, 0, 0, 1,  1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
                state_prev=torch.stack([phi_a, phi_b, basis[0].unsqueeze(0), basis[0].unsqueeze(0)], dim=1),
            )[0][0, 0]  # Destination slot 0 after ADD r0, r1

            expected_sum = (a + b) % 10
            expected_codeword = basis[expected_sum]

            # Invariant: Maximum deviation from analytical ground truth must be strictly < 1e-5
            max_dev = torch.max(torch.abs(phi_c - expected_codeword)).item()
            assert max_dev < 1e-5, f"Arithmetic error on pair ({a}, {b}): max deviation {max_dev:.7f}"

            decoded = kernel.decode_harmonic(phi_c).item()
            assert decoded == expected_sum, f"Decoded sum mismatch for ({a}, {b}): expected {expected_sum}, got {decoded}"


# ===========================================================================
# 3. SET Semantics & Untouched Register Isolation
# ===========================================================================

def test_set_semantics_and_exact_isolation(kernel: Candidate1HDeterministicKernel) -> None:
    """Verifies SET writes exact codeword to destination and preserves untouched registers exactly."""
    state = kernel.initial_state(1)  # (1, 4, 10)

    # SET r2 = 7
    x_set = torch.zeros((1, 22), dtype=torch.float32)
    x_set[0, 0] = 1.0       # OP_SET
    x_set[0, 4 + 2] = 1.0   # reg_dest = 2
    x_set[0, 12 + 7] = 1.0  # immediate_val = 7

    state_next, logits = kernel.forward_step(x_set, state)

    # Destination slot 2 must decode to 7
    decoded_dest = kernel.decode_harmonic(state_next[0, 2]).item()
    assert decoded_dest == 7
    assert torch.max(torch.abs(state_next[0, 2] - kernel.basis[7])).item() < 1e-6

    # Untouched registers r0, r1, r3 must have exact zero state mutation
    for r in [0, 1, 3]:
        delta_norm = torch.max(torch.abs(state_next[0, r] - state[0, r])).item()
        assert delta_norm == 0.0, f"Untouched register {r} was corrupted during SET: delta={delta_norm}"


# ===========================================================================
# 4. SWAP Semantics & Exact Bidirectional Exchange
# ===========================================================================

def test_swap_semantics_and_exact_isolation(kernel: Candidate1HDeterministicKernel) -> None:
    """Verifies SWAP exchanges values of addressed registers with zero corruption to unaddressed registers."""
    state = kernel.initial_state(1)
    # Put 3 in r1 and 8 in r3
    state[0, 1] = kernel.basis[3].clone()
    state[0, 3] = kernel.basis[8].clone()

    # SWAP r1, r3
    x_swap = torch.zeros((1, 22), dtype=torch.float32)
    x_swap[0, 1] = 1.0      # OP_SWAP
    x_swap[0, 4 + 1] = 1.0  # reg_dest = 1
    x_swap[0, 8 + 3] = 1.0  # reg_src = 3

    state_next, _ = kernel.forward_step(x_swap, state)

    # Destination r1 must now hold old r3 (8)
    assert kernel.decode_harmonic(state_next[0, 1]).item() == 8
    assert torch.max(torch.abs(state_next[0, 1] - kernel.basis[8])).item() < 1e-6

    # Source r3 must now hold old r1 (3)
    assert kernel.decode_harmonic(state_next[0, 3]).item() == 3
    assert torch.max(torch.abs(state_next[0, 3] - kernel.basis[3])).item() < 1e-6

    # Unaddressed registers r0, r2 must have exact zero mutation
    for r in [0, 2]:
        delta_norm = torch.max(torch.abs(state_next[0, r] - state[0, r])).item()
        assert delta_norm == 0.0, f"Unaddressed register {r} corrupted during SWAP: delta={delta_norm}"


# ===========================================================================
# 5. ADD Semantics & Source Register Preservation
# ===========================================================================

def test_add_semantics_and_source_preservation(kernel: Candidate1HDeterministicKernel) -> None:
    """Verifies ADD mutates only destination register and strictly preserves source register."""
    state = kernel.initial_state(1)
    state[0, 0] = kernel.basis[4].clone()  # r0 = 4
    state[0, 1] = kernel.basis[7].clone()  # r1 = 7

    # ADD r0 += r1  -> (4 + 7) % 10 = 1
    x_add = torch.zeros((1, 22), dtype=torch.float32)
    x_add[0, 3] = 1.0      # OP_ADD
    x_add[0, 4 + 0] = 1.0  # reg_dest = 0
    x_add[0, 8 + 1] = 1.0  # reg_src = 1

    state_next, _ = kernel.forward_step(x_add, state)

    # Destination r0 must hold 1
    assert kernel.decode_harmonic(state_next[0, 0]).item() == 1

    # Source r1 must be strictly unmodified
    assert kernel.decode_harmonic(state_next[0, 1]).item() == 7
    delta_src = torch.max(torch.abs(state_next[0, 1] - state[0, 1])).item()
    assert delta_src == 0.0, f"Source register r1 was mutated during ADD: delta={delta_src}"

    # Untouched registers r2, r3 must be strictly unmodified
    for r in [2, 3]:
        delta_norm = torch.max(torch.abs(state_next[0, r] - state[0, r])).item()
        assert delta_norm == 0.0, f"Untouched register {r} was mutated during ADD: delta={delta_norm}"


# ===========================================================================
# 6. READ Semantics & Zero State Mutation
# ===========================================================================

def test_read_semantics_and_zero_mutation(kernel: Candidate1HDeterministicKernel) -> None:
    """Verifies READ correctly reports the register value with strictly zero state mutation across all registers."""
    state = kernel.initial_state(1)
    state[0, 2] = kernel.basis[6].clone()  # r2 = 6

    # READ r2
    x_read = torch.zeros((1, 22), dtype=torch.float32)
    x_read[0, 2] = 1.0      # OP_READ
    x_read[0, 4 + 2] = 1.0  # reg_dest = 2

    state_next, logits = kernel.forward_step(x_read, state)

    # Output logit argmax must decode to 6
    pred = logits.argmax(dim=-1).item()
    assert pred == 6, f"READ failed: expected 6, got {pred}"

    # Invariant: ||Delta S|| == 0.0 across all registers
    total_delta = torch.max(torch.abs(state_next - state)).item()
    assert total_delta == 0.0, f"READ caused state mutation: max delta = {total_delta}"


# ===========================================================================
# 7. Initial Register State Alignment
# ===========================================================================

def test_initial_state_matches_canonical_benchmark(kernel: Candidate1HDeterministicKernel) -> None:
    """Verifies that initial state has shape (B, 4, 10) with all registers canonically encoded as integer 0."""
    init_state = kernel.initial_state(batch_size=8)
    assert init_state.shape == (8, 4, 10)

    # Independent expected Phi(0) = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    expected_phi_0 = torch.tensor([1.0, 0.0] * 5)

    for b in range(8):
        for r in range(4):
            reg = init_state[b, r]
            diff = torch.max(torch.abs(reg - expected_phi_0)).item()
            assert diff < 1e-6, f"Register {r} in batch {b} not initialized to canonical Phi(0)"

            decoded = kernel.decode_harmonic(reg).item()
            assert decoded == 0, f"Initial state of register {r} decodes to {decoded}, expected 0"


# ===========================================================================
# 8. Composed Multi-Step Trace Execution against Benchmark Oracle
# ===========================================================================

def test_composed_trace_against_benchmark_oracle(kernel: Candidate1HDeterministicKernel) -> None:
    """Executes randomized composed multi-step traces and verifies exact agreement with the benchmark simulator."""
    task = Phase16RegisterTask()
    rng = np.random.default_rng(42)

    for trial in range(25):
        # Generate a random composed trace using the benchmark generator
        inputs_np, step_targets_np, _ = task.generate_single_sequence(rng)
        # Replay ground truth step-by-step
        inputs_torch = torch.from_numpy(inputs_np).unsqueeze(0)  # (1, T, 22)

        out = kernel(inputs_torch, return_step_diagnostics=True)
        assert out.step_logits is not None

        # Compare decoded register states at every step
        T = inputs_np.shape[0]
        state = kernel.initial_state(1)

        for t in range(T):
            x_t = inputs_torch[:, t, :]
            state, step_logits = kernel.forward_step(x_t, state)

            # Check destination register output
            inst = task.decode_instruction(inputs_np[t])
            expected_dest_val = step_targets_np[t, inst.reg_dest]
            actual_dest_val = step_logits.argmax(dim=-1).item()

            assert actual_dest_val == expected_dest_val, (
                f"Step {t} trial {trial} mismatch: op={inst.op_type}, dest={inst.reg_dest}, "
                f"expected={expected_dest_val}, got={actual_dest_val}"
            )

            # Check all 4 registers decoded values
            all_decoded = kernel.decode_harmonic(state[0]).numpy()
            assert np.array_equal(all_decoded, step_targets_np[t]), (
                f"Full register mismatch at step {t} trial {trial}: "
                f"expected {step_targets_np[t]}, got {all_decoded}"
            )


# ===========================================================================
# 9. Numerical Stability and Harmonic Amplitude Bounds
# ===========================================================================

def test_numerical_stability_and_norm_persistence(kernel: Candidate1HDeterministicKernel) -> None:
    """Verifies that across 50 consecutive operations, harmonic coordinates never shrink or explode."""
    state = kernel.initial_state(1)
    rng = np.random.default_rng(999)

    # 50 consecutive operations
    for step in range(50):
        op = rng.choice([0, 1, 3])  # SET, SWAP, ADD
        dest = rng.integers(0, 4)
        src = rng.integers(0, 4)
        val = rng.integers(0, 10)

        x_t = torch.zeros((1, 22), dtype=torch.float32)
        x_t[0, op] = 1.0
        x_t[0, 4 + dest] = 1.0
        x_t[0, 8 + src] = 1.0
        x_t[0, 12 + val] = 1.0

        state, logits = kernel.forward_step(x_t, state)

        # Finiteness check
        assert torch.isfinite(state).all(), f"Non-finite state detected at step {step}"
        assert torch.isfinite(logits).all(), f"Non-finite logits detected at step {step}"

        # Harmonic block norm bounds (must remain in [0.99999, 1.00001])
        blocks = state.view(1, 4, 5, 2)
        norms = torch.norm(blocks, p=2, dim=-1)
        min_norm = norms.min().item()
        max_norm = norms.max().item()

        assert min_norm >= 0.99999, f"Harmonic amplitude contraction at step {step}: min_norm={min_norm}"
        assert max_norm <= 1.00001, f"Harmonic amplitude explosion at step {step}: max_norm={max_norm}"


# ===========================================================================
# 10. Boundary Cases
# ===========================================================================

def test_boundary_cases_and_edge_conditions(kernel: Candidate1HDeterministicKernel) -> None:
    """Tests edge cases: self-swap, zero operands, modular wrap-around, and repeated additions."""
    state = kernel.initial_state(1)

    # Boundary Case A: Self-swap (SWAP r1, r1) must leave all registers unchanged
    state[0, 1] = kernel.basis[5].clone()
    x_self_swap = torch.zeros((1, 22), dtype=torch.float32)
    x_self_swap[0, 1] = 1.0     # SWAP
    x_self_swap[0, 4 + 1] = 1.0 # dest = 1
    x_self_swap[0, 8 + 1] = 1.0 # src = 1

    state_after_swap, _ = kernel.forward_step(x_self_swap, state)
    assert kernel.decode_harmonic(state_after_swap[0, 1]).item() == 5
    assert torch.max(torch.abs(state_after_swap - state)).item() == 0.0

    # Boundary Case B: Modulus wrap-around (9 + 1 = 0, 9 + 9 = 8)
    state[0, 0] = kernel.basis[9].clone()
    state[0, 1] = kernel.basis[1].clone()
    state_next, val = kernel.execute_instruction(state[0], op_type=3, reg_dest=0, reg_src=1)
    assert val == 0, f"Expected 9 + 1 mod 10 = 0, got {val}"

    state[0, 0] = kernel.basis[9].clone()
    state[0, 1] = kernel.basis[9].clone()
    state_next, val = kernel.execute_instruction(state[0], op_type=3, reg_dest=0, reg_src=1)
    assert val == 8, f"Expected 9 + 9 mod 10 = 8, got {val}"

    # Boundary Case C: Repeated additions (10 times +1 wraps back to start)
    curr_state = kernel.initial_state(1)[0]  # all 0
    curr_state[1] = kernel.basis[1].clone()  # r1 = 1
    for step in range(10):
        curr_state, val = kernel.execute_instruction(curr_state, op_type=3, reg_dest=0, reg_src=1)
        expected = (step + 1) % 10
        assert val == expected, f"Repeated addition step {step}: expected {expected}, got {val}"
    assert val == 0, "10 additions of 1 did not cycle back to 0"


# ===========================================================================
# 11. Verification on Existing Stage 1 and Stage 2 Validation Splits
# ===========================================================================

def test_deterministic_execution_on_stage1_val_seen(kernel: Candidate1HDeterministicKernel) -> None:
    """Verifies Candidate 1H achieves 100% exact match on the 16 Stage 1 VAL-SEEN pairs (firewall preserved)."""
    partition = Stage1PairPartition()
    val_ds = partition.build_trace_split(partition.val_pairs)
    assert len(val_ds) == 16, f"Expected 16 validation pairs, got {len(val_ds)}"

    inputs = val_ds.inputs  # (16, 3, 22)
    targets = val_ds.targets[:, 0]  # Terminal value of r0 at step 3

    out = kernel(inputs)
    preds = out.logits.argmax(dim=-1)

    exact_matches = (preds == targets).sum().item()
    accuracy = exact_matches / len(val_ds)

    assert accuracy == 1.0000, f"Stage 1 VAL-SEEN failure: accuracy = {accuracy:.4f} ({exact_matches}/16)"


def test_deterministic_execution_on_stage2_val_sequences(kernel: Candidate1HDeterministicKernel) -> None:
    """Verifies Candidate 1H achieves 100% exact match on the 1,200 Stage 2 validation sequences."""
    mgr = Phase16SplitManager()
    val_ds, _ = mgr.generate_disjoint_split(1200, seed=50042, generation_mode="stage2_binding")
    assert len(val_ds) == 1200, f"Expected 1200 validation sequences, got {len(val_ds)}"

    inputs = val_ds.inputs  # (1200, 10, 22)
    dest_regs = inputs[:, -1, 4:8].argmax(dim=-1)
    targets = val_ds.targets[torch.arange(len(val_ds)), dest_regs]

    out = kernel(inputs)
    preds = out.logits.argmax(dim=-1)

    exact_matches = (preds == targets).sum().item()
    em_rate = exact_matches / len(val_ds)

    assert em_rate == 1.0000, f"Stage 2 VAL failure: EM = {em_rate:.4f} ({exact_matches}/1200)"
