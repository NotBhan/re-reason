"""Focused unit and contract tests for Phase 1.6 Stage 2 Address-Binding Pilot.

Verifies:
1. Stage 2 sequence generation: sequence length T=10, strictly SET, SWAP, READ (zero ADD).
2. Terminal step contract: strictly READ r_dest with zero source operand and zero immediate value.
3. Readout target semantics: target label strictly matches simulated register value at step 10.
4. Cryptographic sequence disjointness between Stage 2 train and validation splits.
5. Model compatibility: Slot-MGU, Cyclic-MGU, Vanilla GRU, and MSM forward passes.
6. MSM deterministic reference exactness: achieves 100% Behavioral Read EM by construction.
7. Behavioral Read Exact Match (EM_read) metric and cross-entropy loss contracts.
"""

from collections import Counter
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from re_reason.benchmarks.register_machine_benchmark import (
    Instruction,
    Phase16RegisterTask,
    Phase16SplitManager,
    SplitDataset,
)
from re_reason.core.register_machine_models import (
    CyclicMGUPhase16Model,
    MSMDeterministicReference,
    SlotMGUPhase16Model,
    VanillaGRUPhase16Model,
)


def test_stage2_sequence_length_and_operation_restrictions() -> None:
    """Verifies that Stage 2 sequences have length T=10 and contain zero ADD operations."""
    task = Phase16RegisterTask()
    rng = np.random.default_rng(12345)

    for _ in range(50):
        inputs, step_targets, final_target = task.generate_stage2_binding_sequence(rng)

        # Check sequence length T=10
        assert inputs.shape == (10, 22), f"Expected shape (10, 22), got {inputs.shape}"
        assert step_targets.shape == (10, 4), f"Expected shape (10, 4), got {step_targets.shape}"
        assert final_target.shape == (4,), f"Expected shape (4,), got {final_target.shape}"

        # Operation types across sequence
        op_types = inputs[:, 0:4].argmax(axis=-1)

        # Zero ADD operations
        assert np.all(op_types != task.OP_ADD), "ADD operation found in Stage 2 sequence!"
        assert np.all(np.isin(op_types, [task.OP_SET, task.OP_SWAP, task.OP_READ]))

        # First 2 steps must be SET
        assert op_types[0] == task.OP_SET, "Step 0 must be SET"
        assert op_types[1] == task.OP_SET, "Step 1 must be SET"

        # Terminal step (step 9) must be READ
        assert op_types[9] == task.OP_READ, "Terminal step must be READ"


def test_stage2_non_leakage_and_operand_structure() -> None:
    """Verifies non-leakage invariants: SWAP/READ have no immediate values; READ has no source."""
    task = Phase16RegisterTask()
    rng = np.random.default_rng(54321)

    for _ in range(30):
        inputs, _, _ = task.generate_stage2_binding_sequence(rng)

        for t in range(10):
            token = inputs[t]
            op = token[0:4].argmax()

            if op == task.OP_SET:
                assert token[8:12].sum() == 0.0, "SET must have no source register"
                assert token[12:22].sum() == 1.0, "SET must have one-hot immediate value"
            elif op == task.OP_SWAP:
                assert token[4:8].sum() == 1.0, "SWAP must have destination register"
                assert token[8:12].sum() == 1.0, "SWAP must have source register"
                assert token[12:22].sum() == 0.0, "SWAP must not leak immediate values"
            elif op == task.OP_READ:
                assert token[4:8].sum() == 1.0, "READ must have destination register"
                assert token[8:12].sum() == 0.0, "READ must not leak source register"
                assert token[12:22].sum() == 0.0, "READ must not leak immediate values"


def test_stage2_readout_target_semantics() -> None:
    """Verifies that the target label strictly matches destination register r_dest at step 10."""
    task = Phase16RegisterTask()
    rng = np.random.default_rng(9999)

    for _ in range(50):
        inputs, step_targets, final_target = task.generate_stage2_binding_sequence(rng)

        # Terminal step reads r_dest
        dest_reg = inputs[-1, 4:8].argmax()

        # Target label must be register value at step 9
        expected_val = step_targets[-1, dest_reg]
        assert final_target[dest_reg] == expected_val, "final_target does not match step_targets[-1]"
        assert 0 <= expected_val <= 9, f"Target value {expected_val} outside [0, 9]"


def test_stage2_split_disjointness_and_determinism() -> None:
    """Verifies sequence disjointness between train and validation splits."""
    mgr = Phase16SplitManager()

    train_ds, train_hashes = mgr.generate_disjoint_split(200, seed=1042, generation_mode="stage2_binding")
    val_ds, val_hashes = mgr.generate_disjoint_split(100, seed=50042, generation_mode="stage2_binding")

    assert len(train_ds) == 200
    assert len(val_ds) == 100

    s_tr = set(train_hashes)
    s_va = set(val_hashes)

    assert len(s_tr) == 200, "Duplicate sequences in train split"
    assert len(s_va) == 100, "Duplicate sequences in val split"
    assert len(s_tr & s_va) == 0, f"Leakage: {len(s_tr & s_va)} sequences overlap between train and val"


def test_stage2_model_forward_compatibility() -> None:
    """Verifies that all three trainable models and MSM run on Stage 2 sequences."""
    mgr = Phase16SplitManager()
    val_ds, _ = mgr.generate_disjoint_split(16, seed=42, generation_mode="stage2_binding")

    inputs = val_ds.inputs  # (16, 10, 22)
    dest_regs = inputs[:, -1, 4:8].argmax(dim=-1)
    targets = val_ds.targets[torch.arange(len(val_ds)), dest_regs]

    models = [
        ("Slot-MGU", SlotMGUPhase16Model()),
        ("Cyclic-MGU", CyclicMGUPhase16Model()),
        ("Vanilla GRU", VanillaGRUPhase16Model()),
        ("MSM Reference", MSMDeterministicReference()),
    ]

    for name, model in models:
        model.eval()
        with torch.no_grad():
            out = model(inputs)

        assert out.logits.shape == (16, 10), f"Expected logits (16, 10), got {out.logits.shape} for {name}"
        assert torch.isfinite(out.logits).all(), f"Non-finite logits for {name}"

        preds = out.logits.argmax(dim=-1)
        assert preds.shape == (16,)
        assert ((preds >= 0) & (preds <= 9)).all()

        loss = F.cross_entropy(out.logits, targets).item()
        assert np.isfinite(loss), f"Non-finite loss for {name}"


def test_msm_exactness_on_stage2_binding() -> None:
    """Verifies that Candidate 3 (MSM Reference) achieves exact EM_read = 1.000 on Stage 2 sequences."""
    mgr = Phase16SplitManager()
    val_ds, _ = mgr.generate_disjoint_split(50, seed=1234, generation_mode="stage2_binding")

    inputs = val_ds.inputs
    dest_regs = inputs[:, -1, 4:8].argmax(dim=-1)
    targets = val_ds.targets[torch.arange(len(val_ds)), dest_regs]

    msm = MSMDeterministicReference()
    msm.eval()
    with torch.no_grad():
        out = msm(inputs)

    preds = out.logits.argmax(dim=-1)
    em_read = (preds == targets).float().mean().item()

    assert em_read == 1.0000, f"MSM failed Stage 2 address binding: EM_read = {em_read:.4f}"

def test_stage2_runner_msm_check_contract() -> None:
    """Verifies that run_stage2_msm_deterministic_check returns a passing record."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from experiments.run_phase1_6_stage2_pilot import run_stage2_msm_deterministic_check

    mgr = Phase16SplitManager()
    val_ds, _ = mgr.generate_disjoint_split(20, seed=999, generation_mode="stage2_binding")
    dest_regs = val_ds.inputs[:, -1, 4:8].argmax(dim=-1)
    targets_val = val_ds.targets[torch.arange(len(val_ds)), dest_regs]

    res = run_stage2_msm_deterministic_check(val_ds, targets_val)
    assert res["verdict"] == "PASS"
    assert res["behavioral_read_em"] == 1.0000
    assert res["parameter_count"] == 0
    assert res["training_updates"] == 0

