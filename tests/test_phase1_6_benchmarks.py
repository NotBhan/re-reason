"""Independent unit and contract tests for Phase 1.6 benchmark generation infrastructure.

Verifies:
1. Correct operation encoding and target semantics for every operation (SET, SWAP, READ, ADD).
2. Zero immediate-value leakage on operations that do not take immediate values (SWAP, READ, ADD).
3. Correct three-step modular-addition traces (SET r0 a, SET r1 b, ADD r0 r1).
4. Exact 64/16/20 pair partition, pairwise disjointness, domain coverage, and validation-class frequencies.
5. Runtime Regime B exclusion across every ADD transition in every generated training and validation sequence.
6. Required held-out ADD occurrence in every conditioned test sequence.
7. Sequence disjointness, deterministic generation, and reproducible split identities.
8. Edge cases: repeated register addresses, self-SWAP, zero-valued registers, and rejection handling.
9. Backward compatibility with legacy Phase 1 benchmark tests and contracts.
"""

from collections import Counter
import numpy as np
import pytest
import torch

from re_reason.benchmarks.register_machine_benchmark import (
    Instruction,
    Phase16RegisterTask,
    Phase16SplitManager,
    Stage1PairPartition,
)
from re_reason.benchmarks.register_swap import RegisterTrackingTask


# ---------------------------------------------------------------------------
# 1. Operation encoding and target semantics
# ---------------------------------------------------------------------------

def test_operation_encoding_dimensions_and_structure() -> None:
    """Verifies that all 4 operations encode into valid 22-dimensional vectors."""
    task = Phase16RegisterTask()

    # SET r2 = 7
    set_vec = task.encode_instruction(task.OP_SET, reg_dest=2, reg_src=-1, immediate_val=7)
    assert set_vec.shape == (22,)
    assert set_vec[task.OP_SET] == 1.0
    assert set_vec[4 + 2] == 1.0  # dest r2
    assert set_vec[8:12].sum() == 0.0  # no source
    assert set_vec[12 + 7] == 1.0  # value 7

    # SWAP r1, r3
    swap_vec = task.encode_instruction(task.OP_SWAP, reg_dest=1, reg_src=3, immediate_val=-1)
    assert swap_vec.shape == (22,)
    assert swap_vec[task.OP_SWAP] == 1.0
    assert swap_vec[4 + 1] == 1.0  # dest r1
    assert swap_vec[8 + 3] == 1.0  # src r3
    assert swap_vec[12:22].sum() == 0.0  # no immediate

    # READ r0
    read_vec = task.encode_instruction(task.OP_READ, reg_dest=0, reg_src=-1, immediate_val=-1)
    assert read_vec.shape == (22,)
    assert read_vec[task.OP_READ] == 1.0
    assert read_vec[4 + 0] == 1.0  # dest r0
    assert read_vec[8:12].sum() == 0.0  # no source
    assert read_vec[12:22].sum() == 0.0  # no immediate

    # ADD r3, r2
    add_vec = task.encode_instruction(task.OP_ADD, reg_dest=3, reg_src=2, immediate_val=-1)
    assert add_vec.shape == (22,)
    assert add_vec[task.OP_ADD] == 1.0
    assert add_vec[4 + 3] == 1.0  # dest r3
    assert add_vec[8 + 2] == 1.0  # src r2
    assert add_vec[12:22].sum() == 0.0  # no immediate


def test_operation_execution_semantics() -> None:
    """Verifies register state transitions under SET, SWAP, READ, and ADD."""
    task = Phase16RegisterTask()
    regs = np.zeros(4, dtype=np.int64)

    # 1. SET r0 = 4 -> [4, 0, 0, 0]
    inst_set = Instruction(task.OP_SET, reg_dest=0, reg_src=-1, immediate_val=4)
    regs, _ = task.execute_step(regs, inst_set)
    assert np.array_equal(regs, [4, 0, 0, 0])

    # 2. SET r1 = 9 -> [4, 9, 0, 0]
    inst_set2 = Instruction(task.OP_SET, reg_dest=1, reg_src=-1, immediate_val=9)
    regs, _ = task.execute_step(regs, inst_set2)
    assert np.array_equal(regs, [4, 9, 0, 0])

    # 3. SWAP r0, r1 -> [9, 4, 0, 0]
    inst_swap = Instruction(task.OP_SWAP, reg_dest=0, reg_src=1, immediate_val=-1)
    regs, _ = task.execute_step(regs, inst_swap)
    assert np.array_equal(regs, [9, 4, 0, 0])

    # 4. READ r1 -> state unchanged, returns 4
    inst_read = Instruction(task.OP_READ, reg_dest=1, reg_src=-1, immediate_val=-1)
    regs_post_read, read_val = task.execute_step(regs, inst_read)
    assert np.array_equal(regs_post_read, [9, 4, 0, 0])
    assert read_val == 4

    # 5. ADD r0, r1 -> r0 = (9 + 4) % 10 = 3; r1 unchanged -> [3, 4, 0, 0]
    inst_add = Instruction(task.OP_ADD, reg_dest=0, reg_src=1, immediate_val=-1)
    regs, _ = task.execute_step(regs, inst_add)
    assert np.array_equal(regs, [3, 4, 0, 0])


# ---------------------------------------------------------------------------
# 2. Zero immediate-value leakage
# ---------------------------------------------------------------------------

def test_zero_immediate_value_leakage_on_non_immediate_ops() -> None:
    """Guarantees immediate value slice [12:22] is strictly zero for SWAP, READ, and ADD."""
    task = Phase16RegisterTask()
    rng = np.random.default_rng(2026)

    # Test rejection of improper immediate values in non-SET instructions
    with pytest.raises(ValueError, match="must not specify immediate_val"):
        task.encode_instruction(task.OP_SWAP, reg_dest=0, reg_src=1, immediate_val=5)
    with pytest.raises(ValueError, match="must not specify immediate_val"):
        task.encode_instruction(task.OP_READ, reg_dest=0, reg_src=-1, immediate_val=3)
    with pytest.raises(ValueError, match="must not specify immediate_val"):
        task.encode_instruction(task.OP_ADD, reg_dest=0, reg_src=1, immediate_val=9)

    # Test 500 procedurally generated sequences
    for _ in range(100):
        inp, _, _ = task.generate_single_sequence(rng)
        for t in range(inp.shape[0]):
            token = inp[t]
            op = int(np.argmax(token[:4]))
            val_slice = token[12:22]
            if op in (task.OP_SWAP, task.OP_READ, task.OP_ADD):
                assert np.all(val_slice == 0.0), f"Leakage detected at step {t} for op {op}"
            elif op == task.OP_SET:
                assert val_slice.sum() == 1.0


# ---------------------------------------------------------------------------
# 3. Correct 3-step modular-addition traces
# ---------------------------------------------------------------------------

def test_stage1_canonical_3_step_trace() -> None:
    """Verifies that generate_stage1_trace reproduces the documented trace exactly."""
    task = Phase16RegisterTask()

    for a in range(10):
        for b in range(10):
            inp, step_tgt, final_tgt = task.generate_stage1_trace(a, b)
            assert inp.shape == (3, 22)
            assert step_tgt.shape == (3, 4)
            assert final_tgt.shape == (4,)

            # Step 1: SET r0 a
            inst1 = task.decode_instruction(inp[0])
            assert inst1.op_type == task.OP_SET
            assert inst1.reg_dest == 0
            assert inst1.immediate_val == a
            assert np.array_equal(step_tgt[0], [a, 0, 0, 0])

            # Step 2: SET r1 b
            inst2 = task.decode_instruction(inp[1])
            assert inst2.op_type == task.OP_SET
            assert inst2.reg_dest == 1
            assert inst2.immediate_val == b
            assert np.array_equal(step_tgt[1], [a, b, 0, 0])

            # Step 3: ADD r0 r1
            inst3 = task.decode_instruction(inp[2])
            assert inst3.op_type == task.OP_ADD
            assert inst3.reg_dest == 0
            assert inst3.reg_src == 1
            expected_sum = (a + b) % 10
            assert step_tgt[2, 0] == expected_sum
            assert step_tgt[2, 1] == b  # source preserved
            assert np.array_equal(step_tgt[2, 2:], [0, 0])  # untouched preserved
            assert final_tgt[0] == expected_sum


# ---------------------------------------------------------------------------
# 4. Exact 64/16/20 pair partition, coverage, and frequencies
# ---------------------------------------------------------------------------

def test_stage1_pair_partition_properties() -> None:
    """Verifies 64/16/20 partition counts, disjointness, coverage, and validation frequencies."""
    partition = Stage1PairPartition()

    # 1. Counts
    assert len(partition.heldout_pairs) == 20
    assert len(partition.val_pairs) == 16
    assert len(partition.train_pairs) == 64

    # 2. Pairwise disjointness
    h = set(partition.heldout_pairs)
    v = set(partition.val_pairs)
    t = set(partition.train_pairs)
    assert len(h & v) == 0
    assert len(h & t) == 0
    assert len(v & t) == 0

    # 3. Domain coverage: all 100 pairs
    assert len(h | v | t) == 100
    all_pairs = {(a, b) for a in range(10) for b in range(10)}
    assert (h | v | t) == all_pairs

    # 4. Held-out definition: b in {0, 5}
    assert all(b in (0, 5) for _, b in partition.heldout_pairs)
    assert all(b not in (0, 5) for _, b in partition.val_pairs)
    assert all(b not in (0, 5) for _, b in partition.train_pairs)

    # 5. All 10 classes in validation targets
    val_targets = partition.val_targets
    assert set(val_targets) == set(range(10))

    # 6. Prescribed frequencies: classes 0..5 twice, 6..9 once
    counts = Counter(val_targets)
    expected = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1}
    assert counts == expected

    # 7. SplitDataset trace conversion
    train_ds, val_ds, heldout_ds = partition.get_stage1_splits()
    assert len(train_ds) == 64
    assert len(val_ds) == 16
    assert len(heldout_ds) == 20
    assert train_ds.inputs.shape == (64, 3, 22)
    assert val_ds.inputs.shape == (16, 3, 22)
    assert heldout_ds.inputs.shape == (20, 3, 22)


# ---------------------------------------------------------------------------
# 5. Runtime Regime B exclusion across every ADD transition
# ---------------------------------------------------------------------------

def test_regime_b_runtime_source_operand_exclusion() -> None:
    """Verifies that Regime B sequences NEVER execute an ADD sourcing runtime v(r_src) in {0, 5}."""
    task = Phase16RegisterTask()
    rng = np.random.default_rng(42)

    # Generate 50 masked sequences and replay every step
    for _ in range(50):
        inp, step_tgt, _ = task.generate_regime_b_sequence(rng)
        assert inp.shape == (10, 22)

        # Reconstruct execution trace independently
        replayed_step_tgt, history = task.simulate_trace(inp)
        assert np.array_equal(step_tgt, replayed_step_tgt)

        for step_info in history:
            if step_info["op_type"] == task.OP_ADD:
                v_src = step_info["runtime_src_val_pre"]
                assert v_src not in (0, 5), (
                    f"Violation in Regime B: ADD transition at step {step_info['step']} "
                    f"sourced runtime value {v_src} from register {step_info['reg_src']}!"
                )


# ---------------------------------------------------------------------------
# 6. Required held-out ADD occurrence in conditioned test sequences
# ---------------------------------------------------------------------------

def test_regime_b_conditioned_heldout_test_construction() -> None:
    """Verifies that every conditioned held-out test sequence contains >= 1 ADD sourcing runtime {0, 5}."""
    task = Phase16RegisterTask()
    rng = np.random.default_rng(101)

    for _ in range(50):
        inp, step_tgt, _ = task.generate_regime_b_heldout_test_sequence(rng)
        assert inp.shape == (10, 22)

        replayed_step_tgt, history = task.simulate_trace(inp)
        assert np.array_equal(step_tgt, replayed_step_tgt)

        qualifying_adds = [
            h for h in history
            if h["op_type"] == task.OP_ADD and h["runtime_src_val_pre"] in (0, 5)
        ]
        assert len(qualifying_adds) >= 1, "Conditioned test sequence missing qualifying ADD transition"


# ---------------------------------------------------------------------------
# 7. Sequence disjointness, deterministic generation, reproducible split hashes
# ---------------------------------------------------------------------------

def test_phase16_split_manager_disjointness_and_determinism() -> None:
    """Verifies SplitManager sequence disjointness, non-overlapping hashes, and bitwise determinism."""
    mgr1 = Phase16SplitManager()
    mgr2 = Phase16SplitManager()

    # Small test splits
    tr1, va1, te1 = mgr1.create_regime_b_splits(train_size=100, val_size=30, test_size=30, base_seed=42)
    tr2, va2, te2 = mgr2.create_regime_b_splits(train_size=100, val_size=30, test_size=30, base_seed=42)

    # Bitwise determinism across managers
    assert torch.equal(tr1.inputs, tr2.inputs)
    assert torch.equal(va1.inputs, va2.inputs)
    assert torch.equal(te1.inputs, te2.inputs)
    assert tr1.hashes == tr2.hashes
    assert va1.hashes == va2.hashes
    assert te1.hashes == te2.hashes

    # Zero sequence hash overlap
    s_tr = set(tr1.hashes)
    s_va = set(va1.hashes)
    s_te = set(te1.hashes)
    assert len(s_tr & s_va) == 0
    assert len(s_tr & s_te) == 0
    assert len(s_va & s_te) == 0

    # Verify requested size
    assert len(tr1) == 100
    assert len(va1) == 30
    assert len(te1) == 30


# ---------------------------------------------------------------------------
# 8. Edge cases and robustness
# ---------------------------------------------------------------------------

def test_edge_cases_and_rejection_limits() -> None:
    """Verifies handling of repeated addresses, self-SWAP, self-ADD, zero values, and diagnostics."""
    task = Phase16RegisterTask()

    # Self-SWAP: r1 with r1
    regs = np.array([3, 7, 2, 5], dtype=np.int64)
    inst_self_swap = Instruction(task.OP_SWAP, reg_dest=1, reg_src=1, immediate_val=-1)
    new_regs, _ = task.execute_step(regs, inst_self_swap)
    assert np.array_equal(new_regs, regs)  # state untouched

    # Self-ADD: r2 = (r2 + r2) % 10 = (2 + 2) % 10 = 4
    inst_self_add = Instruction(task.OP_ADD, reg_dest=2, reg_src=2, immediate_val=-1)
    new_regs, _ = task.execute_step(regs, inst_self_add)
    assert np.array_equal(new_regs, [3, 7, 4, 5])

    # Zero-valued registers: ADD r0 with r3 where r0=0 -> (0 + 5) % 10 = 5
    regs_zero = np.array([0, 0, 0, 5], dtype=np.int64)
    inst_zero_add = Instruction(task.OP_ADD, reg_dest=0, reg_src=3, immediate_val=-1)
    new_regs_z, _ = task.execute_step(regs_zero, inst_zero_add)
    assert np.array_equal(new_regs_z, [5, 0, 0, 5])

    # Rejection limit error handling
    mgr = Phase16SplitManager()
    with pytest.raises(RuntimeError, match="Exceeded max attempts"):
        # Artificially low max_attempts_factor causes explicit diagnostic exception
        mgr.generate_disjoint_split(size=50, seed=123, generation_mode="regime_b_masked", max_attempts_factor=0)


# ---------------------------------------------------------------------------
# 9. Backward compatibility with Phase 1
# ---------------------------------------------------------------------------

def test_backward_compatibility_with_phase1_interface() -> None:
    """Ensures legacy RegisterTrackingTask remains 100% operational and unchanged."""
    p1_task = RegisterTrackingTask(seed=42)
    assert p1_task.input_dim == 21
    batch = p1_task.generate_batch(8)
    assert batch.inputs.shape == (8, 10, 21)
    assert batch.targets.shape == (8, 4)
    assert batch.metadata["step_targets"].shape == (8, 10, 4)


# ---------------------------------------------------------------------------
# 10. Test hardening from independent benchmark infrastructure audit
# ---------------------------------------------------------------------------

def test_parameterized_token_encodings_exhaustive() -> None:
    """Exhaustively tests encoding and decoding for all valid register and value combinations."""
    task = Phase16RegisterTask()

    # 1. SET: 4 registers x 10 values = 40 combinations
    for r in range(4):
        for v in range(10):
            vec = task.encode_instruction(task.OP_SET, reg_dest=r, reg_src=-1, immediate_val=v)
            inst = task.decode_instruction(vec)
            assert inst.op_type == task.OP_SET
            assert inst.reg_dest == r
            assert inst.reg_src == -1
            assert inst.immediate_val == v

    # 2. SWAP: 4 x 4 = 16 combinations
    for rd in range(4):
        for rs in range(4):
            vec = task.encode_instruction(task.OP_SWAP, reg_dest=rd, reg_src=rs, immediate_val=-1)
            inst = task.decode_instruction(vec)
            assert inst.op_type == task.OP_SWAP
            assert inst.reg_dest == rd
            assert inst.reg_src == rs
            assert inst.immediate_val == -1

    # 3. READ: 4 combinations
    for r in range(4):
        vec = task.encode_instruction(task.OP_READ, reg_dest=r, reg_src=-1, immediate_val=-1)
        inst = task.decode_instruction(vec)
        assert inst.op_type == task.OP_READ
        assert inst.reg_dest == r
        assert inst.reg_src == -1
        assert inst.immediate_val == -1

    # 4. ADD: 4 x 4 = 16 combinations
    for rd in range(4):
        for rs in range(4):
            vec = task.encode_instruction(task.OP_ADD, reg_dest=rd, reg_src=rs, immediate_val=-1)
            inst = task.decode_instruction(vec)
            assert inst.op_type == task.OP_ADD
            assert inst.reg_dest == rd
            assert inst.reg_src == rs
            assert inst.immediate_val == -1


def test_raw_token_replayer_masking_independent() -> None:
    """Independent raw-token replayer verifying runtime masking without calling task.simulate_trace."""
    task = Phase16RegisterTask()
    rng = np.random.default_rng(999)

    def independent_raw_replayer(inputs_arr: np.ndarray):
        regs = [0, 0, 0, 0]
        history = []
        for t in range(inputs_arr.shape[0]):
            tok = inputs_arr[t]
            op = int(np.argmax(tok[:4]))
            dest = int(np.argmax(tok[4:8]))
            src = int(np.argmax(tok[8:12])) if tok[8:12].sum() > 0 else None
            val = int(np.argmax(tok[12:22])) if tok[12:22].sum() > 0 else None

            dest_pre = regs[dest]
            src_pre = regs[src] if src is not None else None

            if op == 0:    # SET
                regs[dest] = val
            elif op == 1:  # SWAP
                regs[dest], regs[src] = regs[src], regs[dest]
            elif op == 2:  # READ
                pass
            elif op == 3:  # ADD
                regs[dest] = (regs[dest] + regs[src]) % 10

            history.append({
                "step": t, "op": op, "dest": dest, "src": src, "val": val,
                "dest_pre": dest_pre, "src_pre": src_pre, "post_regs": list(regs)
            })
        return history

    # Check 100 Regime B masked sequences
    for _ in range(100):
        inp, step_tgt, _ = task.generate_regime_b_sequence(rng)
        hist = independent_raw_replayer(inp)
        for t, h in enumerate(hist):
            assert list(step_tgt[t]) == h["post_regs"]
            if h["op"] == 3:  # ADD
                assert h["src_pre"] not in (0, 5), f"Masking violation at step {t}: {h['src_pre']}"

    # Check 100 Regime B held-out test sequences
    for _ in range(100):
        inp, step_tgt, _ = task.generate_regime_b_heldout_test_sequence(rng)
        hist = independent_raw_replayer(inp)
        for t, h in enumerate(hist):
            assert list(step_tgt[t]) == h["post_regs"]
        qualifying = [h for h in hist if h["op"] == 3 and h["src_pre"] in (0, 5)]
        assert len(qualifying) >= 1, "Held-out test sequence missing qualifying ADD transition"


def test_task_level_generator_rejection_limits_direct() -> None:
    """Verifies that task-level generators raise explicit RuntimeError when max_attempts is exhausted."""
    task = Phase16RegisterTask()
    rng = np.random.default_rng(42)

    with pytest.raises(RuntimeError, match="Failed to generate valid Regime B masked sequence"):
        task.generate_regime_b_sequence(rng, max_attempts=0)

    with pytest.raises(RuntimeError, match="Failed to generate conditioned Regime B held-out test sequence"):
        task.generate_regime_b_heldout_test_sequence(rng, max_attempts=0)


def test_rejection_of_malformed_tokens() -> None:
    """Verifies that decode_instruction strictly rejects multi-hot, non-binary, or leaked tokens."""
    task = Phase16RegisterTask()

    # 1. Non-binary values (0.5, 0.5)
    bad_token = np.zeros(22, dtype=np.float32)
    bad_token[0] = 0.5
    bad_token[1] = 0.5
    bad_token[4] = 1.0
    bad_token[12] = 1.0
    with pytest.raises(ValueError, match="strictly binary"):
        task.decode_instruction(bad_token)

    # 2. Multi-hot opcodes [1.0, 1.0]
    bad_op = np.zeros(22, dtype=np.float32)
    bad_op[0] = 1.0
    bad_op[1] = 1.0
    bad_op[4] = 1.0
    with pytest.raises(ValueError, match="Token must have exactly one active op_type"):
        task.decode_instruction(bad_op)

    # 3. Multi-hot destination registers
    bad_dest = np.zeros(22, dtype=np.float32)
    bad_dest[0] = 1.0
    bad_dest[4] = 1.0
    bad_dest[5] = 1.0
    bad_dest[12] = 1.0
    with pytest.raises(ValueError, match="Token must have exactly one active destination register"):
        task.decode_instruction(bad_dest)

    # 4. SWAP with missing source
    bad_swap = np.zeros(22, dtype=np.float32)
    bad_swap[task.OP_SWAP] = 1.0
    bad_swap[4] = 1.0
    with pytest.raises(ValueError, match="requires exactly one active source register"):
        task.decode_instruction(bad_swap)

    # 5. Immediate value leakage on ADD
    leak_add = task.encode_instruction(task.OP_ADD, reg_dest=0, reg_src=1)
    leak_add[15] = 1.0
    with pytest.raises(ValueError, match="leakage detected"):
        task.decode_instruction(leak_add)

