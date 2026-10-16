"""Focused deterministic tests for EXP-CYCLIC-STEPWISE-01.

Covers the registered Stage 1 requirements:

1.  Destination-target extraction matches ``Phase16RegisterTask.simulate_trace()`` for
    every allowed TRAIN-FIT and VAL-SEEN example.
2.  The extracted destination-target sequence is exactly ``(a, b, (a+b) % 10)``.
3.  The fixed cyclic ALU is unchanged and still passes exhaustive 100-pair modular arithmetic.
4.  Parameter invariance: exactly 219,018 trainable parameters.
5.  Default forward-path invariance: diagnostics on/off bit-identical, and bit-identical to
    the frozen pre-change reference captured from the original (HEAD) source.
6.  Diagnostic shapes, addresses, register roles and timestep alignment.
7.  Metric correctness on known canonical reference vectors and known synthetic groupings.
8.  Loss correctness: total == terminal + 1.0 * mean(step-0, step-1), step 2 excluded.
9.  Gradient correctness: finite gradients reaching mlp_read, mlp_write and mlp_gate.
10. No new trainable components from the diagnostic instrumentation.
11. Protected-split firewall: only approved train/validation data paths are used.

The protected TEST split (operand pairs with b in {0, 5}) is never loaded, constructed,
evaluated or queried anywhere in this file.
"""

import hashlib
import inspect
import os
import sys
from typing import Optional

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from re_reason.benchmarks.register_machine_benchmark import Phase16RegisterTask, Stage1PairPartition
from re_reason.core.register_machine_models import (
    CyclicMGUPhase16Model,
    Phase16StepDiagnostics,
    cyclic_alu,
    get_harmonic_basis,
    project_to_unit_circle,
)
from experiments.run_exp_cyclic_stepwise_01 import (
    AUXILIARY_COEFFICIENT,
    COSINE_EPS,
    EXPECTED_PARAMETER_COUNT,
    MIN_VECTOR_NORM,
    _cross_context_consistency,
    assert_canonical_trace,
    assert_protected_domain_absent,
    classify_experiment_outcome,
    compute_losses,
    compute_representation_metrics,
    cosine_similarity,
    evaluate_split,
    extract_destination_targets,
    run_exp_cyclic_stepwise_01,
    select_best_checkpoint,
)

# ---------------------------------------------------------------------------
# Frozen pre-change reference (captured from the unmodified HEAD source of
# src/re_reason/core/register_machine_models.py, seed 20261016, CPU float32,
# first 4 TRAIN-FIT traces). Comparison is bit-exact by construction: the
# diagnostic instrumentation adds no parameters and no RNG consumption.
# ---------------------------------------------------------------------------
REFERENCE_SEED = 20261016
REFERENCE_TRAIN_TRACES = 4
FROZEN_STATE_DICT_SHA256 = "47e958f5990d1cc75f9366f8ac3f79c9561d879dd6f7f49d9aec57f9e57b539a"
FROZEN_INPUTS_SHA256 = "b867c147511b314e9d17b84cfa274d1f9a01154ffdc984b431266531b85dd535"
FROZEN_LOGITS_SHA256 = "6e07a02baaafce08a5336c47745bf00f16a8c0cb9dc8d69f5b4bb38b03bb5107"
FROZEN_FINAL_STATE_SHA256 = "143d489631661d5e6a87796be9fa87cd4cdf0cd80257e930f454f51a7d37e29b"
FROZEN_STEP_LOGITS_SHA256 = "1eafaaaea0ddc6829f7126b728d5d49e3b331abfcd1cd3067e106cbf2a572101"

# cos(Phi(v), Phi(v + d)) for the fixed 5-harmonic basis: -1/5 for odd d, 0 for
# even non-zero d, 1 for d == 0. Used as an exact analytical metric reference.
CANONICAL_ODD_OFFSET_COSINE = -0.2


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().numpy().astype(np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _state_dict_sha256(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        hasher.update(key.encode())
        hasher.update(state[key].detach().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _train_and_val_traces():
    partition = Stage1PairPartition()
    task = Phase16RegisterTask(sequence_length=3)
    train_ds = partition.build_trace_split(partition.train_pairs, task)
    val_ds = partition.build_trace_split(partition.val_pairs, task)
    return partition, train_ds, val_ds


class _FakeMetricModel:
    """Minimal stand-in exposing only what the metric functions consume."""

    def __init__(self, basis: torch.Tensor, num_classes: int = 10) -> None:
        self.basis = basis
        self.num_classes = num_classes


class _FakeMetricOutput:
    """Minimal stand-in for a diagnostics-enabled Phase16ModelOutput."""

    def __init__(self, logits: torch.Tensor, step_logits: torch.Tensor,
                 diagnostics: Optional[Phase16StepDiagnostics]) -> None:
        self.logits = logits
        self.step_logits = step_logits
        self.diagnostics = diagnostics


# ---------------------------------------------------------------------------
# 1-2. Target mapping and canonical trace
# ---------------------------------------------------------------------------

def test_destination_target_mapping_matches_simulate_trace() -> None:
    """Dynamic extraction must match the independent deterministic replay simulator."""
    partition, train_ds, val_ds = _train_and_val_traces()
    task = Phase16RegisterTask(sequence_length=3)

    for split_name, pairs, dataset in (
        ("TRAIN-FIT", partition.train_pairs, train_ds),
        ("VAL-SEEN", partition.val_pairs, val_ds),
    ):
        for index, (a, b) in enumerate(pairs):
            inputs = dataset.inputs[index:index + 1]  # (1, 3, 22)
            step_targets = dataset.step_targets[index:index + 1]

            replayed, history = task.simulate_trace(dataset.inputs[index].numpy())
            assert np.array_equal(replayed, step_targets[0].numpy()), (
                f"[{split_name}] step_targets disagree with simulate_trace for ({a}, {b})"
            )

            extracted = extract_destination_targets(inputs, step_targets)[0]
            expected = [step["runtime_dest_val_post"] for step in history]
            assert extracted.tolist() == expected, (
                f"[{split_name}] destination extraction mismatch for ({a}, {b}): "
                f"{extracted.tolist()} vs {expected}"
            )


def test_canonical_destination_sequence_is_a_b_sum() -> None:
    """The extracted destination-target sequence must be exactly (a, b, (a+b) % 10)."""
    partition, train_ds, val_ds = _train_and_val_traces()

    for split_name, pairs, dataset in (
        ("TRAIN-FIT", partition.train_pairs, train_ds),
        ("VAL-SEEN", partition.val_pairs, val_ds),
    ):
        assert_canonical_trace(dataset.inputs, dataset.step_targets, split_name)
        extracted = extract_destination_targets(dataset.inputs, dataset.step_targets)

        for index, (a, b) in enumerate(pairs):
            assert extracted[index].tolist() == [a, b, (a + b) % 10], (
                f"[{split_name}] canonical sequence mismatch for ({a}, {b})"
            )

        assert bool(torch.equal(
            extract_destination_targets(dataset.inputs, dataset.step_targets)[:, -1],
            dataset.targets[:, 0],
        ))


def test_destination_extraction_rejects_shape_mismatch() -> None:
    """The extractor must fail loudly rather than silently mis-index."""
    inputs = torch.zeros((2, 3, 22))
    with pytest.raises(ValueError, match="shape"):
        extract_destination_targets(inputs, torch.zeros((2, 2, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="incompatible"):
        extract_destination_targets(inputs, torch.zeros((3, 3, 4), dtype=torch.long))


# ---------------------------------------------------------------------------
# 3. ALU invariant
# ---------------------------------------------------------------------------

def test_cyclic_alu_unchanged_exhaustive_100_pairs() -> None:
    """The fixed cyclic ALU must remain exact on all 100 ordered pairs (a, b)."""
    basis = get_harmonic_basis(num_values=10, num_frequencies=5)
    max_error = 0.0
    for a in range(10):
        for b in range(10):
            result = cyclic_alu(basis[a].unsqueeze(0), basis[b].unsqueeze(0), num_frequencies=5)
            expected = basis[(a + b) % 10].unsqueeze(0)

            max_error = max(max_error, (result - expected).abs().max().item())

            block_norms = torch.norm(result.view(5, 2), p=2, dim=-1)
            assert torch.allclose(block_norms, torch.ones(5), atol=1e-5)

            assert (result @ basis.t()).argmax(dim=-1).item() == (a + b) % 10
    assert max_error < 1e-5, f"cyclically-exact rotation drift {max_error:.3e} exceeds tolerance"


def test_cyclic_alu_gradients_flow_through_rotation() -> None:
    """The ALU stays differentiable with respect to both operand reads."""
    basis = get_harmonic_basis()
    u_a = basis[3].clone().requires_grad_(True)
    u_b = basis[4].clone().requires_grad_(True)
    cyclic_alu(u_a.unsqueeze(0), u_b.unsqueeze(0)).sum().backward()
    for name, tensor in (("u_a", u_a), ("u_b", u_b)):
        assert tensor.grad is not None, f"no gradient to {name} through cyclic_alu"
        assert bool(torch.isfinite(tensor.grad).all())


# ---------------------------------------------------------------------------
# 4 and 10. Parameter invariance and no new trainable components
# ---------------------------------------------------------------------------

def test_parameter_count_invariance() -> None:
    model = CyclicMGUPhase16Model()
    assert model.count_parameters() == EXPECTED_PARAMETER_COUNT == 219018
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 219018


def test_diagnostics_add_no_trainable_parameters() -> None:
    """Instrumentation must not introduce parameters, buffers or RNG consumption."""
    torch.manual_seed(11)
    model = CyclicMGUPhase16Model()

    names_before = [name for name, _ in model.named_parameters()]
    buffers_before = [name for name, _ in model.named_buffers()]
    counts_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    partition = Stage1PairPartition()
    inputs = partition.build_trace_split(partition.train_pairs[:2]).inputs

    with torch.no_grad():
        model(inputs, return_step_diagnostics=True)
        model(inputs, return_step_diagnostics=False)

    names_after = [name for name, _ in model.named_parameters()]
    buffers_after = [name for name, _ in model.named_buffers()]

    assert names_before == names_after
    assert buffers_before == buffers_after == ["basis"]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == EXPECTED_PARAMETER_COUNT
    assert counts_before == EXPECTED_PARAMETER_COUNT
    assert model.basis.requires_grad is False

    # The diagnostic container itself must not be a module or a parameter holder.
    diagnostics = Phase16StepDiagnostics(
        u_a=torch.zeros(1, 10), u_b=torch.zeros(1, 10), c1_alu=torch.zeros(1, 10),
        u_dest_post=torch.zeros(1, 3, 10), step_index=2,
    )
    assert not hasattr(diagnostics, "parameters")
    assert sorted(diagnostics._fields) == ["c1_alu", "step_index", "u_a", "u_b", "u_dest_post"]


# ---------------------------------------------------------------------------
# 5. Default forward-path invariance
# ---------------------------------------------------------------------------

def test_diagnostics_do_not_alter_the_forward_computation() -> None:
    """Enabling diagnostics must leave the standard computation bit-identical."""
    partition = Stage1PairPartition()
    inputs = partition.build_trace_split(partition.train_pairs[:8]).inputs

    torch.manual_seed(23)
    model = CyclicMGUPhase16Model()
    model.eval()

    with torch.no_grad():
        plain = model(inputs, return_step_diagnostics=False)
        diagnostics_enabled = model(inputs, return_step_diagnostics=True)

    assert plain.diagnostics is None
    assert plain.step_logits is None
    assert diagnostics_enabled.diagnostics is not None

    # The materialised terminal computation is bit-identical; only the optional per-step
    # capture (and the diagnostic container) differ.
    assert torch.equal(plain.logits, diagnostics_enabled.logits)
    assert torch.equal(plain.final_state, diagnostics_enabled.final_state)
    assert diagnostics_enabled.step_logits is not None

    # Explicitly disabled diagnostics on an explicit state must be a no-op too.
    state = model.initial_state(inputs.shape[0], inputs.device)
    with torch.no_grad():
        next_state, step_logits = model.forward_step(inputs[:, 0, :], state)
        next_state_with, step_logits_with = model.forward_step(inputs[:, 0, :], state, {})
    assert torch.equal(next_state, next_state_with)
    assert torch.equal(step_logits, step_logits_with)


def test_forward_path_matches_frozen_prechange_reference() -> None:
    """Bit-exact comparison against the reference captured from the pre-change source.

    Reference captured with seed 20261016 on the first 4 TRAIN-FIT traces, using the
    unmodified HEAD revision of ``src/re_reason/core/register_machine_models.py``.
    """
    partition = Stage1PairPartition()
    inputs = partition.build_trace_split(partition.train_pairs[:REFERENCE_TRAIN_TRACES]).inputs

    torch.manual_seed(REFERENCE_SEED)
    model = CyclicMGUPhase16Model()
    model.eval()

    assert _state_dict_sha256(model) == FROZEN_STATE_DICT_SHA256, (
        "model initialisation no longer reproduces the pre-change state dictionary — "
        "diagnostic instrumentation must not consume RNG state or add parameters"
    )
    assert _tensor_sha256(inputs) == FROZEN_INPUTS_SHA256, (
        "training inputs changed; the frozen reference is only valid for the registered split"
    )

    with torch.no_grad():
        out = model(inputs, return_step_diagnostics=True)

    failures = []
    for label, tensor, expected in (
        ("logits", out.logits, FROZEN_LOGITS_SHA256),
        ("final_state", out.final_state, FROZEN_FINAL_STATE_SHA256),
        ("step_logits", out.step_logits, FROZEN_STEP_LOGITS_SHA256),
    ):
        actual = _tensor_sha256(tensor)
        if actual != expected:
            failures.append(f"{label}: expected {expected[:16]}… got {actual[:16]}…")
    assert not failures, "default forward path diverged from the pre-change reference: " + "; ".join(failures)


# ---------------------------------------------------------------------------
# 6. Diagnostic shapes, addresses, register roles and timestep alignment
# ---------------------------------------------------------------------------

def test_diagnostic_shapes_roles_and_timestep_alignment() -> None:
    partition = Stage1PairPartition()
    dataset = partition.build_trace_split(partition.val_pairs)
    inputs = dataset.inputs
    batch_size, seq_len = inputs.shape[0], inputs.shape[1]
    assert (batch_size, seq_len) == (16, 3)

    torch.manual_seed(31)
    model = CyclicMGUPhase16Model()
    model.eval()

    with torch.no_grad():
        out = model(inputs, return_step_diagnostics=True)
        diagnostics = out.diagnostics

        # Replay the rollout step by step, keeping the pre-step state of every transition.
        states_before = []
        step_captures = []
        state = model.initial_state(batch_size, inputs.device)
        for t in range(seq_len):
            states_before.append(state)
            capture: dict = {}
            state, _ = model.forward_step(inputs[:, t, :], state, capture)
            step_captures.append(capture)

    assert diagnostics is not None
    assert diagnostics.u_a.shape == (batch_size, 10)
    assert diagnostics.u_b.shape == (batch_size, 10)
    assert diagnostics.c1_alu.shape == (batch_size, 10)
    assert diagnostics.u_dest_post.shape == (batch_size, seq_len, 10)
    assert diagnostics.step_index == seq_len - 1 == 2

    # Address and register-role verification from the instruction encoding itself.
    destination_codes = inputs[:, :, 4:8].argmax(dim=-1)
    source_codes = inputs[:, :, 8:12].argmax(dim=-1)
    assert bool((destination_codes[:, 2] == 0).all()), "step 2 destination is not r0"
    assert bool((source_codes[:, 2] == 1).all()), "step 2 source is not r1"

    # u_a / u_b / c1_alu must be the actual step-2 ALU operands, independently recomputed
    # from the captured pre-step-2 state.
    with torch.no_grad():
        expected_u_a = project_to_unit_circle(
            model.mlp_read(torch.cat([states_before[2], inputs[:, 2, 4:8]], dim=-1)),
            model.num_frequencies, model.eps,
        )
        expected_u_b = project_to_unit_circle(
            model.mlp_read(torch.cat([states_before[2], inputs[:, 2, 8:12]], dim=-1)),
            model.num_frequencies, model.eps,
        )
        expected_c1_alu = cyclic_alu(expected_u_a, expected_u_b, model.num_frequencies, model.eps)

    assert torch.equal(expected_u_a, diagnostics.u_a), "u_a is not the step-2 destination operand read"
    assert torch.equal(expected_u_b, diagnostics.u_b), "u_b is not the step-2 source operand read"
    assert torch.equal(expected_c1_alu, diagnostics.c1_alu), "c1_alu is not the raw step-2 ALU output"

    # c1_alu must be the raw ALU tensor, not the instruction-mixed write payload c1.
    with torch.no_grad():
        value_embed = torch.einsum("bv,vd->bd", inputs[:, 2, 12:22], model.basis)
        assert not torch.equal(value_embed, diagnostics.c1_alu)

    # u_dest_post alignment: per-step stack equals the per-step captures, and the final
    # step decodes to the terminal logits through the fixed prototype bank.
    stacked = torch.stack([capture["u_dest_post"] for capture in step_captures], dim=1)
    assert torch.equal(stacked, diagnostics.u_dest_post)
    assert torch.equal(diagnostics.u_dest_post[:, -1, :], step_captures[-1]["u_dest_post"])
    assert torch.equal(state, out.final_state)

    decoded = torch.einsum("btd,vd->btv", diagnostics.u_dest_post, model.basis)
    assert torch.allclose(decoded, out.step_logits, atol=1e-6)
    assert torch.allclose(decoded[:, -1, :], out.logits, atol=1e-6)
    assert torch.equal(out.logits, out.step_logits[:, -1, :])

    # All captured tensors must be finite and unit-normalised per 2D harmonic block.
    for name, tensor in (
        ("u_a", diagnostics.u_a), ("u_b", diagnostics.u_b),
        ("c1_alu", diagnostics.c1_alu), ("u_dest_post", diagnostics.u_dest_post),
    ):
        assert bool(torch.isfinite(tensor).all())
        block_norms = torch.norm(tensor.view(*tensor.shape[:-1], 5, 2), p=2, dim=-1)
        assert torch.allclose(block_norms, torch.ones_like(block_norms), atol=1e-5), name


def test_diagnostics_disabled_returns_existing_behaviour() -> None:
    """The pre-existing output contract (step_logits None unless requested) is preserved."""
    torch.manual_seed(37)
    model = CyclicMGUPhase16Model()
    inputs = Stage1PairPartition().build_trace_split(Stage1PairPartition().train_pairs[:2]).inputs

    with torch.no_grad():
        default_output = model(inputs, return_step_diagnostics=False)
    assert default_output.step_logits is None
    assert default_output.diagnostics is None
    assert default_output.logits.shape == (2, 10)
    assert default_output.final_state.shape == (2, 212)


# ---------------------------------------------------------------------------
# 7. Metric correctness
# ---------------------------------------------------------------------------

def test_cosine_similarity_known_values_and_zero_norm_rule() -> None:
    e1 = torch.tensor([[1.0, 0.0, 0.0]])
    e2 = torch.tensor([[0.0, 1.0, 0.0]])
    sixty_degrees = torch.tensor([[0.5, float(np.sqrt(3) / 2), 0.0]])
    zero = torch.zeros((1, 3))

    assert cosine_similarity(e1, e1).item() == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(e1, e2).item() == pytest.approx(0.0, abs=1e-6)
    assert cosine_similarity(e1, sixty_degrees).item() == pytest.approx(0.5, abs=1e-6)
    assert cosine_similarity(e1, -e1).item() == pytest.approx(-1.0, abs=1e-6)

    # Zero-norm rule: similarity 0.0 whenever either vector has norm below 1e-7.
    assert cosine_similarity(e1, zero).item() == 0.0
    assert cosine_similarity(zero, e1).item() == 0.0
    assert cosine_similarity(zero, zero).item() == 0.0
    assert cosine_similarity(e1 * 1e-9, e1).item() == 0.0

    assert COSINE_EPS == 1e-8
    assert MIN_VECTOR_NORM == 1e-7


def test_cosine_similarity_canonical_basis_reference() -> None:
    """cos(Phi(v), Phi(v + d)) is -1/5 for odd d and 0 for even non-zero d."""
    basis = get_harmonic_basis()
    for v in range(10):
        assert cosine_similarity(basis[v].unsqueeze(0), basis[v].unsqueeze(0)).item() == pytest.approx(1.0, abs=1e-6)
        for offset in range(1, 10):
            value = cosine_similarity(
                basis[v].unsqueeze(0), basis[(v + offset) % 10].unsqueeze(0)
            ).item()
            if offset % 2 == 1:
                assert value == pytest.approx(CANONICAL_ODD_OFFSET_COSINE, abs=1e-5)
            else:
                assert value == pytest.approx(0.0, abs=1e-5)


def test_cross_context_consistency_grouping_and_aggregation() -> None:
    """Pooling across roles, pairwise means, digit omission and unweighted averaging."""
    def basis_vector(index: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(index), 10).float()

    # digit 3 -> two identical vectors (pairwise cosine 1.0)
    # digit 7 -> two orthogonal vectors (pairwise cosine 0.0)
    # digits 5, 9, 0, 2 -> a single vector each (must be omitted)
    vectors_a = torch.stack([basis_vector(0), basis_vector(1), basis_vector(4), basis_vector(5)])
    values_a = torch.tensor([3, 7, 5, 9])
    vectors_b = torch.stack([basis_vector(0), basis_vector(2), basis_vector(6), basis_vector(8)])
    values_b = torch.tensor([3, 7, 0, 2])

    overall, per_digit, omitted = _cross_context_consistency(vectors_a, values_a, vectors_b, values_b, 10)

    assert per_digit["3"]["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert per_digit["3"]["n_vectors"] == 2
    assert per_digit["3"]["n_distinct_pairs"] == 1
    assert per_digit["7"]["mean_pairwise_cosine"] == pytest.approx(0.0, abs=1e-6)

    # Unweighted mean over the two available digits (1.0 and 0.0), not pooled over pairs.
    assert overall == pytest.approx(0.5, abs=1e-6)
    assert sorted(omitted) == [0, 1, 2, 4, 5, 6, 8, 9]
    assert sorted(int(key) for key in per_digit) == [3, 7]


def test_cross_context_consistency_applies_zero_norm_rule() -> None:
    """A degenerate vector contributes similarity 0.0 and is not silently dropped."""
    e1 = torch.tensor([[1.0, 0.0, 0.0]])
    zero = torch.zeros((1, 3))
    vectors_a = torch.cat([e1, e1, zero], dim=0)
    values_a = torch.tensor([3, 3, 3])
    vectors_b = torch.cat([e1, zero], dim=0)
    values_b = torch.tensor([3, 3])

    overall, per_digit, omitted = _cross_context_consistency(vectors_a, values_a, vectors_b, values_b, 10)

    # Pool of 5 vectors: [e1, e1, 0, e1, 0]. Three e1-rows give C(3,2) = 3 pairs with
    # cosine 1.0; every pair involving a zero-norm vector is assigned 0.0 by the rule.
    assert per_digit["3"]["n_vectors"] == 5
    assert per_digit["3"]["n_distinct_pairs"] == 10
    assert per_digit["3"]["mean_pairwise_cosine"] == pytest.approx(3.0 / 10.0, abs=1e-6)
    assert overall == pytest.approx(3.0 / 10.0, abs=1e-6)
    # Only digit 3 has vectors; every other digit is omitted for having fewer than two.
    assert sorted(omitted) == [0, 1, 2, 4, 5, 6, 7, 8, 9]


def test_representation_metrics_on_canonical_vectors() -> None:
    """Canonical codewords must score exactly 1.0 on every alignment metric."""
    basis = get_harmonic_basis()
    # Duplicated rows so that each operand digit carries at least two step-2 reads,
    # which keeps the cross-context consistency metric defined.
    operand_a = torch.tensor([3, 7, 3, 7])
    operand_b = torch.tensor([4, 2, 4, 2])
    destination = torch.stack([operand_a, operand_b, (operand_a + operand_b) % 10], dim=1)

    u_a = basis[destination[:, 0]]
    u_b = basis[destination[:, 1]]
    c1_alu = basis[destination[:, 2]]
    u_dest_post = torch.stack([basis[destination[:, t]] for t in range(3)], dim=1)
    step_logits = torch.einsum("btd,vd->btv", u_dest_post, basis)

    diagnostics = Phase16StepDiagnostics(
        u_a=u_a, u_b=u_b, c1_alu=c1_alu, u_dest_post=u_dest_post, step_index=2
    )
    output = _FakeMetricOutput(
        logits=step_logits[:, -1, :], step_logits=step_logits, diagnostics=diagnostics
    )

    metrics = compute_representation_metrics(_FakeMetricModel(basis), output, destination)

    assert metrics["c_read_r0"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_read_r1"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_read"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_alu"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_dest"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["s_context"] == pytest.approx(1.0, abs=1e-6)
    assert sorted(int(key) for key in metrics["s_context_per_digit"]) == [2, 3, 4, 7]
    assert metrics["s_context_omitted_digits"] == [0, 1, 5, 6, 8, 9]
    for t in range(3):
        assert metrics[f"c_dest_step{t}"] == pytest.approx(1.0, abs=1e-6)
        assert metrics[f"step{t}_acc"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["acc"] == pytest.approx(1.0, abs=1e-6)


def test_representation_metrics_detect_misaligned_representations() -> None:
    """Alignment metrics must fall below 1.0 when the reads are rotated off canonical.

    Also demonstrates why accuracies alone are insufficient: here the reads are
    *internally consistent* (s_context = 1.0) while being consistently misaligned
    (c_read = -0.2), and the terminal classification is unaffected.
    """
    basis = get_harmonic_basis()
    operand_a = torch.tensor([3, 7, 3, 7])
    operand_b = torch.tensor([4, 2, 4, 2])
    destination = torch.stack([operand_a, operand_b, (operand_a + operand_b) % 10], dim=1)

    # Operand reads deliberately rotated by +1 (wrong value), ALU output left canonical.
    u_a_misaligned = basis[(destination[:, 0] + 1) % 10]
    u_b_misaligned = basis[(destination[:, 1] + 1) % 10]
    c1_alu = basis[destination[:, 2]]
    u_dest_post = torch.stack([basis[destination[:, t]] for t in range(3)], dim=1)
    step_logits = torch.einsum("btd,vd->btv", u_dest_post, basis)

    diagnostics = Phase16StepDiagnostics(
        u_a=u_a_misaligned, u_b=u_b_misaligned, c1_alu=c1_alu,
        u_dest_post=u_dest_post, step_index=2,
    )
    output = _FakeMetricOutput(logits=step_logits[:, -1, :], step_logits=step_logits, diagnostics=diagnostics)

    metrics = compute_representation_metrics(_FakeMetricModel(basis), output, destination)

    assert metrics["c_read_r0"] == pytest.approx(CANONICAL_ODD_OFFSET_COSINE, abs=1e-5)
    assert metrics["c_read_r1"] == pytest.approx(CANONICAL_ODD_OFFSET_COSINE, abs=1e-5)
    assert metrics["c_read"] == pytest.approx(CANONICAL_ODD_OFFSET_COSINE, abs=1e-5)
    # Cross-context consistency remains perfect: consistently wrong is still consistent.
    assert metrics["s_context"] == pytest.approx(1.0, abs=1e-6)
    # Raw ALU-output alignment is unaffected by the misaligned operand reads.
    assert metrics["c_alu"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_dest"] == pytest.approx(1.0, abs=1e-6)
    # Classification accuracy alone would not reveal the misalignment.
    assert metrics["acc"] == pytest.approx(1.0, abs=1e-6)


def test_representation_metrics_reject_missing_diagnostics() -> None:
    basis = get_harmonic_basis()
    output = _FakeMetricOutput(
        logits=torch.zeros(2, 10), step_logits=torch.zeros(2, 3, 10), diagnostics=None
    )
    with pytest.raises(RuntimeError, match="diagnostics-enabled"):
        compute_representation_metrics(_FakeMetricModel(basis), output, torch.zeros(2, 3, dtype=torch.long))


def test_evaluate_split_returns_separated_loss_and_representation_fields() -> None:
    partition = Stage1PairPartition()
    dataset = partition.build_trace_split(partition.val_pairs)

    torch.manual_seed(41)
    model = CyclicMGUPhase16Model()
    metrics = evaluate_split(model, dataset.inputs, dataset.step_targets)

    expected_keys = {
        "loss_total", "loss_terminal", "loss_aux", "acc",
        "c_read", "c_read_r0", "c_read_r1", "s_context", "s_context_per_digit",
        "s_context_omitted_digits", "c_alu", "c_dest",
        "c_dest_step0", "c_dest_step1", "c_dest_step2",
        "step0_acc", "step1_acc", "step2_acc",
    }
    assert expected_keys <= set(metrics)
    for key in ("c_read", "c_read_r0", "c_read_r1", "c_alu", "c_dest", "acc", "s_context"):
        assert -1.0 - 1e-6 <= metrics[key] <= 1.0 + 1e-6, f"{key} outside cosine/accuracy range"
    assert metrics["loss_total"] >= 0.0


# ---------------------------------------------------------------------------
# 8. Loss correctness
# ---------------------------------------------------------------------------

def test_total_loss_equals_terminal_plus_auxiliary_with_unit_coefficient() -> None:
    partition = Stage1PairPartition()
    dataset = partition.build_trace_split(partition.train_pairs)

    torch.manual_seed(43)
    model = CyclicMGUPhase16Model()
    out = model(dataset.inputs, return_step_diagnostics=True)
    y_destination = extract_destination_targets(dataset.inputs, dataset.step_targets)

    terminal_loss, auxiliary_loss, total_loss = compute_losses(
        out, y_destination[:, -1], y_destination[:, 0], y_destination[:, 1]
    )

    assert AUXILIARY_COEFFICIENT == 1.0

    # Independent recomputation straight from the raw logits (no argmax anywhere).
    reference_terminal = torch.nn.functional.cross_entropy(out.logits, y_destination[:, -1])
    reference_step_0 = torch.nn.functional.cross_entropy(out.step_logits[:, 0, :], y_destination[:, 0])
    reference_step_1 = torch.nn.functional.cross_entropy(out.step_logits[:, 1, :], y_destination[:, 1])
    reference_auxiliary = (reference_step_0 + reference_step_1) / 2.0

    assert torch.equal(terminal_loss, reference_terminal)
    assert torch.equal(auxiliary_loss, reference_auxiliary)
    assert torch.equal(total_loss, reference_terminal + 1.0 * reference_auxiliary)
    assert torch.equal(total_loss, terminal_loss + auxiliary_loss)

    # Step 2 must be supervised exactly once: through the terminal loss only.
    reference_step_2 = torch.nn.functional.cross_entropy(out.step_logits[:, 2, :], y_destination[:, 2])
    assert reference_step_2.item() > 1e-6
    double_counted = reference_terminal + (reference_step_0 + reference_step_1 + reference_step_2) / 3.0
    assert not torch.isclose(total_loss, double_counted, atol=1e-9)

    for tensor in (terminal_loss, auxiliary_loss, total_loss):
        assert bool(torch.isfinite(tensor).all())
        assert tensor.requires_grad


def test_losses_drive_gradients_without_argmax_detachment() -> None:
    """Cross-entropy must be applied to the raw logits so the graph stays connected."""
    partition = Stage1PairPartition()
    dataset = partition.build_trace_split(partition.train_pairs[:8])

    torch.manual_seed(47)
    model = CyclicMGUPhase16Model()
    out = model(dataset.inputs, return_step_diagnostics=True)
    y_destination = extract_destination_targets(dataset.inputs, dataset.step_targets)

    _, _, total_loss = compute_losses(
        out, y_destination[:, -1], y_destination[:, 0], y_destination[:, 1]
    )
    assert total_loss.requires_grad
    total_loss.backward()
    assert model.mlp_read[0].weight.grad is not None
    assert model.mlp_write[0].weight.grad is not None
    assert model.mlp_gate[0].weight.grad is not None


# ---------------------------------------------------------------------------
# 9. Gradient correctness
# ---------------------------------------------------------------------------

def _block_squared_grad_norm(model: CyclicMGUPhase16Model, block: str) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(block):
            assert parameter.grad is not None, f"no gradient for {name}"
            total += float((parameter.grad.detach() ** 2).sum().item())
    return total


def test_gradients_are_finite_and_reach_all_trainable_blocks() -> None:
    partition = Stage1PairPartition()
    dataset = partition.build_trace_split(partition.train_pairs)

    torch.manual_seed(53)
    model = CyclicMGUPhase16Model()
    out = model(dataset.inputs, return_step_diagnostics=True)
    y_destination = extract_destination_targets(dataset.inputs, dataset.step_targets)

    _, _, total_loss = compute_losses(
        out, y_destination[:, -1], y_destination[:, 0], y_destination[:, 1]
    )
    total_loss.backward()

    parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter_count += 1
        assert parameter.grad is not None, f"parameter {name} received no gradient"
        assert bool(torch.isfinite(parameter.grad).all()), f"non-finite gradient for {name}"
    # Three two-layer MLPs: (weight + bias) per Linear layer.
    assert parameter_count == 12

    for block in ("mlp_read", "mlp_write", "mlp_gate"):
        assert _block_squared_grad_norm(model, block) > 1e-12, f"gradient collapsed at {block}"


def test_auxiliary_loss_alone_reaches_the_read_network() -> None:
    """The new auxiliary terms must drive mlp_read through the step-0/1 destination reads."""
    partition = Stage1PairPartition()
    dataset = partition.build_trace_split(partition.train_pairs[:8])

    torch.manual_seed(59)
    model = CyclicMGUPhase16Model()
    out = model(dataset.inputs, return_step_diagnostics=True)
    y_destination = extract_destination_targets(dataset.inputs, dataset.step_targets)

    _, auxiliary_loss, _ = compute_losses(
        out, y_destination[:, -1], y_destination[:, 0], y_destination[:, 1]
    )
    auxiliary_loss.backward()
    assert _block_squared_grad_norm(model, "mlp_read") > 1e-12
    assert _block_squared_grad_norm(model, "mlp_gate") > 1e-12


def test_terminal_loss_alone_produces_finite_gradients_everywhere() -> None:
    """Sanity floor: the terminal loss by itself must reach every trainable block."""
    partition = Stage1PairPartition()
    dataset = partition.build_trace_split(partition.train_pairs[:8])

    torch.manual_seed(61)
    model = CyclicMGUPhase16Model()
    out = model(dataset.inputs, return_step_diagnostics=True)
    y_destination = extract_destination_targets(dataset.inputs, dataset.step_targets)

    terminal_loss = torch.nn.functional.cross_entropy(out.logits, y_destination[:, -1])
    terminal_loss.backward()
    for block in ("mlp_read", "mlp_write", "mlp_gate"):
        assert _block_squared_grad_norm(model, block) > 1e-12


# ---------------------------------------------------------------------------
# Runner integration: telemetry schema
# ---------------------------------------------------------------------------

def test_runner_checkpoint_telemetry_schema() -> None:
    """Checkpoints must carry training and validation fields separately."""
    payload = run_exp_cyclic_stepwise_01(
        seeds=[42],
        total_updates=2,
        val_cadence=1,
        batch_size=64,
        device=torch.device("cpu"),
        results_path="",
        run_mode="SMOKE_TEST_ONLY",
    )

    assert payload["experiment_id"] == "EXP-CYCLIC-STEPWISE-01"
    assert payload["loss_definition"]["auxiliary_coefficient"] == 1.0
    assert payload["protected_test_split"]["queries"] == 0
    assert payload["gate_assessment"] is None

    run = payload["runs"][0]
    assert run["seed"] == 42
    assert run["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert run["finite_value_checks_passed"] == 2
    assert sorted(run["gradient_blocks_present"]) == ["mlp_gate", "mlp_read", "mlp_write"]

    expected_train = {
        "train_loss_total", "train_loss_terminal", "train_loss_aux", "train_acc",
        "train_c_read", "train_c_read_r0", "train_c_read_r1", "train_c_alu", "train_c_dest",
        "train_c_dest_step0", "train_c_dest_step1", "train_c_dest_step2",
        "train_step0_acc", "train_step1_acc", "train_step2_acc",
        "train_s_context", "train_s_context_per_digit", "train_s_context_omitted_digits",
    }
    expected_val = {key.replace("train_", "val_", 1) for key in expected_train}

    assert len(run["checkpoints"]) == 2
    for checkpoint in run["checkpoints"]:
        assert set(checkpoint) == {"update"} | expected_train | expected_val
        for key, value in checkpoint.items():
            if key.startswith(("train_", "val_")) and key not in (
                "train_s_context_per_digit", "val_s_context_per_digit",
                "train_s_context_omitted_digits", "val_s_context_omitted_digits",
            ):
                assert np.isfinite(value), f"non-finite telemetry value for {key}"

    # The training objective is the registered one: total == terminal + 1.0 * auxiliary.
    # The logged scalars are float32-derived, so the identity is checked at float32
    # precision here; the exact tensor-level identity is asserted in the loss test above.
    for checkpoint in run["checkpoints"]:
        assert checkpoint["train_loss_total"] == pytest.approx(
            checkpoint["train_loss_terminal"] + checkpoint["train_loss_aux"], rel=1e-6
        )
        assert checkpoint["val_loss_total"] == pytest.approx(
            checkpoint["val_loss_terminal"] + checkpoint["val_loss_aux"], rel=1e-6
        )


def test_runner_rejects_unaligned_batch_and_existing_artifact() -> None:
    with pytest.raises(RuntimeError, match="registered batch size"):
        run_exp_cyclic_stepwise_01(
            seeds=[42], total_updates=1, val_cadence=1, batch_size=8,
            device=torch.device("cpu"), results_path="", run_mode="SMOKE_TEST_ONLY",
        )

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        run_exp_cyclic_stepwise_01(
            seeds=[42], total_updates=1, val_cadence=1, batch_size=64,
            device=torch.device("cpu"), results_path="pyproject.toml", run_mode="SMOKE_TEST_ONLY",
        )


# ---------------------------------------------------------------------------
# 11. Protected-split firewall
# ---------------------------------------------------------------------------

def test_runner_materializes_only_allowed_splits(monkeypatch) -> None:
    """Every split materialized by the runner must come from TRAIN-FIT or VAL-SEEN."""
    import experiments.run_exp_cyclic_stepwise_01 as runner_module

    partition = Stage1PairPartition()
    allowed = set(partition.train_pairs) | set(partition.val_pairs)
    protected = set(partition.heldout_pairs)

    recorded = []
    original = Stage1PairPartition.build_trace_split

    def recording_build_trace_split(self, pairs, task=None):
        recorded.append(tuple(pairs))
        return original(self, pairs, task)

    monkeypatch.setattr(Stage1PairPartition, "build_trace_split", recording_build_trace_split)

    runner_module.run_exp_cyclic_stepwise_01(
        seeds=[42], total_updates=2, val_cadence=2, batch_size=64,
        device=torch.device("cpu"), results_path="", run_mode="SMOKE_TEST_ONLY",
    )

    assert recorded, "runner did not materialize any split"
    for pairs in recorded:
        assert set(pairs) <= allowed, "runner materialized pairs outside the allowed splits"
        assert not (set(pairs) & protected), "runner materialized protected TEST pairs"


def test_runner_source_never_references_the_protected_split() -> None:
    import experiments.run_exp_cyclic_stepwise_01 as runner_module

    source = inspect.getsource(runner_module)
    for forbidden in ("heldout_pairs", "get_stage1_splits", "TEST-HELDOUT", "heldout_ds"):
        assert forbidden not in source, f"runner references forbidden protected-split token {forbidden!r}"


def test_protected_domain_assertion_is_live() -> None:
    """The firewall guard must fire on synthetic inputs carrying a held-out operand b."""
    partition = Stage1PairPartition()
    train_ds = partition.build_trace_split(partition.train_pairs)
    val_ds = partition.build_trace_split(partition.val_pairs)
    assert_protected_domain_absent(train_ds.inputs, train_ds.step_targets, "TRAIN-FIT")
    assert_protected_domain_absent(val_ds.inputs, val_ds.step_targets, "VAL-SEEN")

    # Synthetic (non-trace) tensors that merely place a held-out value at the step-1
    # destination slot. No registered split data is involved.
    synthetic_inputs = torch.zeros((1, 3, 22))
    synthetic_inputs[0, 1, 4 + 1] = 1.0  # step 1 destination = r1
    synthetic_targets = torch.zeros((1, 3, 4), dtype=torch.long)

    for heldout_value in (0, 5):
        synthetic_targets[0, 1, 1] = heldout_value
        with pytest.raises(RuntimeError, match="evaluation firewall violated"):
            assert_protected_domain_absent(synthetic_inputs, synthetic_targets, "SYNTHETIC-PROBE")

    synthetic_targets[0, 1, 1] = 4
    assert_protected_domain_absent(synthetic_inputs, synthetic_targets, "SYNTHETIC-PROBE")


def test_protected_split_definition_is_unchanged() -> None:
    partition = Stage1PairPartition()
    assert len(partition.heldout_pairs) == 20
    assert len(partition.train_pairs) == 64
    assert len(partition.val_pairs) == 16
    assert all(b in (0, 5) for _, b in partition.heldout_pairs)
    assert all(b not in (0, 5) for _, b in partition.val_pairs)
    assert all(b not in (0, 5) for _, b in partition.train_pairs)


# ---------------------------------------------------------------------------
# 12. Deterministic Decision Matrix & Checkpoint Selection Tests
# ---------------------------------------------------------------------------

def test_select_best_checkpoint_policy() -> None:
    """Verifies the preregistered checkpoint selection policy.

    Policy: min val_loss_total, breaking ties by max val_acc, then earliest update.
    """
    # 1. Minimum val_loss_total
    checkpoints = [
        {"update": 50, "val_loss_total": 0.50, "val_acc": 0.80},
        {"update": 100, "val_loss_total": 0.20, "val_acc": 0.90},
        {"update": 150, "val_loss_total": 0.35, "val_acc": 1.00},
    ]
    assert select_best_checkpoint(checkpoints)["update"] == 100

    # 2. Tie broken by highest val_acc
    checkpoints_tie_acc = [
        {"update": 50, "val_loss_total": 0.20, "val_acc": 0.85},
        {"update": 100, "val_loss_total": 0.20, "val_acc": 0.95},
    ]
    assert select_best_checkpoint(checkpoints_tie_acc)["update"] == 100

    # 3. Further tie broken by earliest update step
    checkpoints_tie_step = [
        {"update": 50, "val_loss_total": 0.20, "val_acc": 0.95},
        {"update": 100, "val_loss_total": 0.20, "val_acc": 0.95},
    ]
    assert select_best_checkpoint(checkpoints_tie_step)["update"] == 50

    # 4. Empty list raises ValueError
    with pytest.raises(ValueError, match="empty"):
        select_best_checkpoint([])


def test_decision_matrix_progression_pass_outcomes() -> None:
    """Progression pass (>= 4/5 at 16/16, 0 <= 3/16) across representation states."""
    # 1. Canonical full representation: C_read >= 0.80, S_context >= 0.80, C_alu >= 0.80
    res = classify_experiment_outcome(
        val_accuracies=[1.0, 1.0, 1.0, 1.0, 1.0],
        c_read=0.90, s_context=0.85, c_alu=0.88,
    )
    assert res["behavioral_state"] == "PROGRESSION_PASS"
    assert res["representation_state"] == "CANONICAL_FULL"
    assert res["category"] == "BEHAVIORAL_PASS_CANONICAL_FULL"
    assert res["counts"]["n_pass"] == 5
    assert res["counts"]["n_reject"] == 0

    # 2. Gap 1: Canonical input-only: C_read >= 0.80, S_context >= 0.80, C_alu < 0.80
    res_input_only = classify_experiment_outcome(
        val_accuracies=[1.0, 1.0, 1.0, 1.0, 0.5],  # 4 pass, 1 mid, 0 reject
        c_read=0.85, s_context=0.82, c_alu=0.75,
    )
    assert res_input_only["behavioral_state"] == "PROGRESSION_PASS"
    assert res_input_only["representation_state"] == "CANONICAL_INPUT_ONLY"
    assert res_input_only["category"] == "BEHAVIORAL_PASS_CANONICAL_INPUT_ONLY"

    # 3. Partial representation (e.g. S_context < 0.80, but C_read >= 0.80)
    res_partial = classify_experiment_outcome(
        val_accuracies=[1.0, 1.0, 1.0, 1.0, 1.0],
        c_read=0.85, s_context=0.60, c_alu=0.85,
    )
    assert res_partial["behavioral_state"] == "PROGRESSION_PASS"
    assert res_partial["representation_state"] == "PARTIAL"
    assert res_partial["category"] == "BEHAVIORAL_PASS_PARTIAL_ALIGNMENT"

    # 4. Failed representation (C_read < 0.30 and C_alu < 0.30)
    res_unaligned = classify_experiment_outcome(
        val_accuracies=[1.0, 1.0, 1.0, 1.0, 1.0],
        c_read=0.20, s_context=0.50, c_alu=0.15,
    )
    assert res_unaligned["behavioral_state"] == "PROGRESSION_PASS"
    assert res_unaligned["representation_state"] == "FAILED"
    assert res_unaligned["category"] == "BEHAVIORAL_PASS_UNALIGNED"


def test_decision_matrix_mixed_seed_intermediate_cases() -> None:
    """Gap 4: Mixed-seed distributions unambiguously partitioned as INTERMEDIATE."""
    # Case A: 4 seeds pass, but 1 seed is hard reject (acc < 0.250).
    # Since n_reject != 0, it fails the progression gate and must be INTERMEDIATE.
    res_mixed_a = classify_experiment_outcome(
        val_accuracies=[1.0, 1.0, 1.0, 1.0, 0.125],
        c_read=0.85, s_context=0.82, c_alu=0.85,
    )
    assert res_mixed_a["behavioral_state"] == "INTERMEDIATE"
    assert res_mixed_a["counts"]["n_pass"] == 4
    assert res_mixed_a["counts"]["n_reject"] == 1
    assert res_mixed_a["category"] == "INTERMEDIATE_CANONICAL_FULL"

    # Case B: 3 seeds pass, 2 seeds mid (0.25 <= acc < 0.95).
    res_mixed_b = classify_experiment_outcome(
        val_accuracies=[1.0, 1.0, 1.0, 0.5, 0.625],
        c_read=0.50, s_context=0.50, c_alu=0.50,
    )
    assert res_mixed_b["behavioral_state"] == "INTERMEDIATE"
    assert res_mixed_b["counts"]["n_pass"] == 3
    assert res_mixed_b["counts"]["n_reject"] == 0
    assert res_mixed_b["counts"]["n_mid"] == 2
    assert res_mixed_b["representation_state"] == "PARTIAL"
    assert res_mixed_b["category"] == "INCONCLUSIVE_INTERMEDIATE"

    # Case C: All 5 seeds intermediate (e.g., 0.50), failed representation.
    res_mixed_c = classify_experiment_outcome(
        val_accuracies=[0.5, 0.5, 0.5, 0.5, 0.5],
        c_read=0.20, s_context=0.20, c_alu=0.20,
    )
    assert res_mixed_c["behavioral_state"] == "INTERMEDIATE"
    assert res_mixed_c["counts"]["n_mid"] == 5
    assert res_mixed_c["representation_state"] == "FAILED"
    assert res_mixed_c["category"] == "INCONCLUSIVE_UNALIGNED"


def test_decision_matrix_rejection_boundary_and_alignment_gap3() -> None:
    """Rejection boundary (>= 4/5 seeds < 0.250) and alignment condition consistency."""
    # 1. Full falsification: behavioral rejection + both C_read < 0.30 and C_alu < 0.30
    res_full_falsified = classify_experiment_outcome(
        val_accuracies=[0.125, 0.0, 0.0625, 0.1875, 0.5],  # 4 reject, 1 mid
        c_read=0.15, s_context=0.20, c_alu=0.10,
    )
    assert res_full_falsified["behavioral_state"] == "REJECTION_BOUNDARY"
    assert res_full_falsified["representation_state"] == "FAILED"
    assert res_full_falsified["category"] == "FALSIFIED_FULL"

    # 2. Gap 3: Rejection boundary with partial alignment (e.g. C_read < 0.30 but C_alu >= 0.30)
    res_partial_alu = classify_experiment_outcome(
        val_accuracies=[0.125, 0.0, 0.0, 0.0, 0.0],
        c_read=0.25, s_context=0.40, c_alu=0.45,
    )
    assert res_partial_alu["behavioral_state"] == "REJECTION_BOUNDARY"
    assert res_partial_alu["representation_state"] == "PARTIAL"
    assert res_partial_alu["category"] == "FALSIFIED_PARTIAL_ALIGNMENT"

    # 3. Gap 3: Rejection boundary with partial alignment (C_alu < 0.30 but C_read >= 0.30)
    res_partial_read = classify_experiment_outcome(
        val_accuracies=[0.125, 0.0, 0.0, 0.0, 0.0],
        c_read=0.45, s_context=0.40, c_alu=0.20,
    )
    assert res_partial_read["behavioral_state"] == "REJECTION_BOUNDARY"
    assert res_partial_read["representation_state"] == "PARTIAL"
    assert res_partial_read["category"] == "FALSIFIED_PARTIAL_ALIGNMENT"

    # 4. Decoupled full representation: behavioral rejection despite full canonical alignment
    res_decoupled_full = classify_experiment_outcome(
        val_accuracies=[0.0, 0.0, 0.0, 0.0, 0.0],
        c_read=0.88, s_context=0.85, c_alu=0.82,
    )
    assert res_decoupled_full["behavioral_state"] == "REJECTION_BOUNDARY"
    assert res_decoupled_full["representation_state"] == "CANONICAL_FULL"
    assert res_decoupled_full["category"] == "REJECTION_CANONICAL_FULL_DECOUPLED"

    # 5. Decoupled input-only representation: behavioral rejection despite input alignment
    res_decoupled_input = classify_experiment_outcome(
        val_accuracies=[0.0, 0.0, 0.0, 0.0, 0.0],
        c_read=0.88, s_context=0.85, c_alu=0.72,
    )
    assert res_decoupled_input["behavioral_state"] == "REJECTION_BOUNDARY"
    assert res_decoupled_input["representation_state"] == "CANONICAL_INPUT_ONLY"
    assert res_decoupled_input["category"] == "REJECTION_CANONICAL_INPUT_DECOUPLED"


def test_decision_matrix_per_seed_sequence_aggregation_and_boundaries() -> None:
    """Verifies macro-averaging over 5-seed representation sequences and boundary cases."""
    pass_accs = [1.0, 1.0, 1.0, 1.0, 1.0]

    # Boundary Case 1: 4 seeds high (0.90), 1 seed outlier (0.35)
    # Mean c_read = (4 * 0.90 + 0.35) / 5 = 0.790 < 0.80 -> PARTIAL, NOT CANONICAL_FULL
    res_below_80 = classify_experiment_outcome(
        val_accuracies=pass_accs,
        c_read=[0.90, 0.90, 0.90, 0.90, 0.35],
        s_context=[0.85, 0.85, 0.85, 0.85, 0.85],
        c_alu=[0.85, 0.85, 0.85, 0.85, 0.85],
    )
    assert res_below_80["metrics"]["c_read"] == pytest.approx(0.790, abs=1e-5)
    assert res_below_80["representation_state"] == "PARTIAL"
    assert res_below_80["category"] == "BEHAVIORAL_PASS_PARTIAL_ALIGNMENT"
    assert res_below_80["per_seed_metrics"]["c_read"] == [0.90, 0.90, 0.90, 0.90, 0.35]

    # Boundary Case 2: 4 seeds high (0.90), 1 seed moderate (0.45)
    # Mean c_read = (4 * 0.90 + 0.45) / 5 = 0.810 >= 0.80 -> CANONICAL_FULL
    res_above_80 = classify_experiment_outcome(
        val_accuracies=pass_accs,
        c_read=[0.90, 0.90, 0.90, 0.90, 0.45],
        s_context=[0.85, 0.85, 0.85, 0.85, 0.85],
        c_alu=[0.85, 0.85, 0.85, 0.85, 0.85],
    )
    assert res_above_80["metrics"]["c_read"] == pytest.approx(0.810, abs=1e-5)
    assert res_above_80["representation_state"] == "CANONICAL_FULL"
    assert res_above_80["category"] == "BEHAVIORAL_PASS_CANONICAL_FULL"

    # Boundary Case 3: 4 seeds low (0.20), 1 seed high (0.75)
    # Mean c_read = (4 * 0.20 + 0.75) / 5 = 0.310 >= 0.30 -> PARTIAL, NOT FAILED
    reject_accs = [0.0, 0.0, 0.0, 0.0, 0.0]
    res_above_30 = classify_experiment_outcome(
        val_accuracies=reject_accs,
        c_read=[0.20, 0.20, 0.20, 0.20, 0.75],
        s_context=[0.20, 0.20, 0.20, 0.20, 0.20],
        c_alu=[0.20, 0.20, 0.20, 0.20, 0.20],
    )
    assert res_above_30["metrics"]["c_read"] == pytest.approx(0.310, abs=1e-5)
    assert res_above_30["representation_state"] == "PARTIAL"
    assert res_above_30["category"] == "FALSIFIED_PARTIAL_ALIGNMENT"

    # Boundary Case 4: 4 seeds low (0.20), 1 seed moderate (0.65)
    # Mean c_read = (4 * 0.20 + 0.65) / 5 = 0.290 < 0.30 -> FAILED (since c_alu < 0.30)
    res_below_30 = classify_experiment_outcome(
        val_accuracies=reject_accs,
        c_read=[0.20, 0.20, 0.20, 0.20, 0.65],
        s_context=[0.20, 0.20, 0.20, 0.20, 0.20],
        c_alu=[0.20, 0.20, 0.20, 0.20, 0.20],
    )
    assert res_below_30["metrics"]["c_read"] == pytest.approx(0.290, abs=1e-5)
    assert res_below_30["representation_state"] == "FAILED"
    assert res_below_30["category"] == "FALSIFIED_FULL"


def test_decision_matrix_numerical_and_structural_invalids() -> None:
    """Invalid executions return IMPLEMENTATION_INVALID."""
    valid_accs = [1.0, 1.0, 1.0, 1.0, 1.0]

    # Numerical check flag False
    res = classify_experiment_outcome(valid_accs, 0.9, 0.9, 0.9, numerical_checks_passed=False)
    assert res["category"] == "IMPLEMENTATION_INVALID"

    # Seed count != 5 in accuracies
    res = classify_experiment_outcome([1.0, 1.0, 1.0, 1.0], 0.9, 0.9, 0.9)
    assert res["category"] == "IMPLEMENTATION_INVALID"

    # Seed count != 5 in representation metric sequences
    res = classify_experiment_outcome(valid_accs, [0.9, 0.9, 0.9, 0.9], [0.9]*5, [0.9]*5)
    assert res["category"] == "IMPLEMENTATION_INVALID"

    # Non-finite inside sequence
    res = classify_experiment_outcome(valid_accs, [0.9, float("nan"), 0.9, 0.9, 0.9], [0.9]*5, [0.9]*5)
    assert res["category"] == "IMPLEMENTATION_INVALID"

    # NaN / Inf in accuracies
    res = classify_experiment_outcome([1.0, 1.0, float("nan"), 1.0, 1.0], 0.9, 0.9, 0.9)
    assert res["category"] == "IMPLEMENTATION_INVALID"
    res = classify_experiment_outcome([1.0, 1.0, float("inf"), 1.0, 1.0], 0.9, 0.9, 0.9)
    assert res["category"] == "IMPLEMENTATION_INVALID"

    # Out of range accuracy
    res = classify_experiment_outcome([1.0, 1.0, -0.1, 1.0, 1.0], 0.9, 0.9, 0.9)
    assert res["category"] == "IMPLEMENTATION_INVALID"
    res = classify_experiment_outcome([1.0, 1.0, 1.1, 1.0, 1.0], 0.9, 0.9, 0.9)
    assert res["category"] == "IMPLEMENTATION_INVALID"

    # NaN in scalar representation metrics
    res = classify_experiment_outcome(valid_accs, float("nan"), 0.9, 0.9)
    assert res["category"] == "IMPLEMENTATION_INVALID"
    res = classify_experiment_outcome(valid_accs, 0.9, float("nan"), 0.9)
    assert res["category"] == "IMPLEMENTATION_INVALID"
    res = classify_experiment_outcome(valid_accs, 0.9, 0.9, float("nan"))
    assert res["category"] == "IMPLEMENTATION_INVALID"


def test_decision_matrix_exhaustive_partition_verification() -> None:
    """Verifies all 12 cells in the 3x4 matrix map to distinct specified categories."""
    expected_matrix = {
        ("PROGRESSION_PASS", "CANONICAL_FULL"): "BEHAVIORAL_PASS_CANONICAL_FULL",
        ("PROGRESSION_PASS", "CANONICAL_INPUT_ONLY"): "BEHAVIORAL_PASS_CANONICAL_INPUT_ONLY",
        ("PROGRESSION_PASS", "PARTIAL"): "BEHAVIORAL_PASS_PARTIAL_ALIGNMENT",
        ("PROGRESSION_PASS", "FAILED"): "BEHAVIORAL_PASS_UNALIGNED",
        ("INTERMEDIATE", "CANONICAL_FULL"): "INTERMEDIATE_CANONICAL_FULL",
        ("INTERMEDIATE", "CANONICAL_INPUT_ONLY"): "INTERMEDIATE_CANONICAL_INPUT_ONLY",
        ("INTERMEDIATE", "PARTIAL"): "INCONCLUSIVE_INTERMEDIATE",
        ("INTERMEDIATE", "FAILED"): "INCONCLUSIVE_UNALIGNED",
        ("REJECTION_BOUNDARY", "CANONICAL_FULL"): "REJECTION_CANONICAL_FULL_DECOUPLED",
        ("REJECTION_BOUNDARY", "CANONICAL_INPUT_ONLY"): "REJECTION_CANONICAL_INPUT_DECOUPLED",
        ("REJECTION_BOUNDARY", "PARTIAL"): "FALSIFIED_PARTIAL_ALIGNMENT",
        ("REJECTION_BOUNDARY", "FAILED"): "FALSIFIED_FULL",
    }

    acc_scenarios = {
        "PROGRESSION_PASS": [1.0, 1.0, 1.0, 1.0, 1.0],
        "INTERMEDIATE": [1.0, 1.0, 0.5, 0.5, 0.5],
        "REJECTION_BOUNDARY": [0.0, 0.0, 0.0, 0.0, 0.5],
    }

    rep_scenarios = {
        "CANONICAL_FULL": (0.85, 0.85, 0.85),
        "CANONICAL_INPUT_ONLY": (0.85, 0.85, 0.75),
        "PARTIAL": (0.60, 0.60, 0.60),
        "FAILED": (0.20, 0.50, 0.20),
    }

    assert len(expected_matrix) == 12
    assert len(set(expected_matrix.values())) == 12, "all 12 categories must be distinct"

    for b_state, accs in acc_scenarios.items():
        for r_state, (c_r, s_c, c_a) in rep_scenarios.items():
            res = classify_experiment_outcome(accs, c_r, s_c, c_a)
            assert res["behavioral_state"] == b_state
            assert res["representation_state"] == r_state
            expected_category = expected_matrix[(b_state, r_state)]
            assert res["category"] == expected_category, (
                f"Mismatch for ({b_state}, {r_state}): got {res['category']}, expected {expected_category}"
            )


def test_decision_matrix_cosine_similarity_bounds_and_negative_values() -> None:
    """Verifies that cosine metrics accept finite values in [-1.0, 1.0] and reject out-of-bounds values."""
    pass_accs = [1.0, 1.0, 1.0, 1.0, 1.0]
    reject_accs = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 1. Valid negative alignment producing REP_FAILED under progression pass
    res_neg_pass = classify_experiment_outcome(
        val_accuracies=pass_accs,
        c_read=-0.40,
        s_context=0.20,
        c_alu=-0.50,
    )
    assert res_neg_pass["behavioral_state"] == "PROGRESSION_PASS"
    assert res_neg_pass["representation_state"] == "FAILED"
    assert res_neg_pass["category"] == "BEHAVIORAL_PASS_UNALIGNED"

    # 2. Valid negative alignment producing REP_FAILED under rejection boundary with 5-seed sequence
    res_neg_reject = classify_experiment_outcome(
        val_accuracies=reject_accs,
        c_read=[-0.30, -0.25, -0.40, -0.35, -0.20],
        s_context=[-0.10, -0.05, 0.00, 0.05, 0.10],
        c_alu=[-0.50, -0.45, -0.60, -0.55, -0.40],
    )
    assert res_neg_reject["behavioral_state"] == "REJECTION_BOUNDARY"
    assert res_neg_reject["representation_state"] == "FAILED"
    assert res_neg_reject["category"] == "FALSIFIED_FULL"
    assert res_neg_reject["metrics"]["c_read"] == pytest.approx(-0.30, abs=1e-5)
    assert res_neg_reject["metrics"]["s_context"] == pytest.approx(0.00, abs=1e-5)
    assert res_neg_reject["metrics"]["c_alu"] == pytest.approx(-0.50, abs=1e-5)

    # 3. Exact -1.0 boundary (scalar and sequence)
    res_bound_neg1_scalar = classify_experiment_outcome(pass_accs, -1.0, -1.0, -1.0)
    assert res_bound_neg1_scalar["representation_state"] == "FAILED"
    assert res_bound_neg1_scalar["category"] == "BEHAVIORAL_PASS_UNALIGNED"

    res_bound_neg1_seq = classify_experiment_outcome(reject_accs, [-1.0] * 5, [-1.0] * 5, [-1.0] * 5)
    assert res_bound_neg1_seq["representation_state"] == "FAILED"
    assert res_bound_neg1_seq["category"] == "FALSIFIED_FULL"

    # 4. Exact 1.0 boundary (scalar and sequence)
    res_bound_pos1_scalar = classify_experiment_outcome(pass_accs, 1.0, 1.0, 1.0)
    assert res_bound_pos1_scalar["representation_state"] == "CANONICAL_FULL"
    assert res_bound_pos1_scalar["category"] == "BEHAVIORAL_PASS_CANONICAL_FULL"

    res_bound_pos1_seq = classify_experiment_outcome(pass_accs, [1.0] * 5, [1.0] * 5, [1.0] * 5)
    assert res_bound_pos1_seq["representation_state"] == "CANONICAL_FULL"
    assert res_bound_pos1_seq["category"] == "BEHAVIORAL_PASS_CANONICAL_FULL"

    # 5. Out-of-bounds cosine rejection (scalar)
    for bad_val in [1.0001, 1.05, 2.0, -1.0001, -1.05, -2.0]:
        res_bad_read = classify_experiment_outcome(pass_accs, bad_val, 0.5, 0.5)
        assert res_bad_read["category"] == "IMPLEMENTATION_INVALID"
        assert res_bad_read["representation_state"] == "INVALID"

        res_bad_ctx = classify_experiment_outcome(pass_accs, 0.5, bad_val, 0.5)
        assert res_bad_ctx["category"] == "IMPLEMENTATION_INVALID"

        res_bad_alu = classify_experiment_outcome(pass_accs, 0.5, 0.5, bad_val)
        assert res_bad_alu["category"] == "IMPLEMENTATION_INVALID"

    # 6. Out-of-bounds cosine rejection (sequence element)
    for bad_val in [1.0001, -1.0001]:
        res_seq_read = classify_experiment_outcome(pass_accs, [0.5, 0.5, bad_val, 0.5, 0.5], [0.5] * 5, [0.5] * 5)
        assert res_seq_read["category"] == "IMPLEMENTATION_INVALID"

        res_seq_ctx = classify_experiment_outcome(pass_accs, [0.5] * 5, [0.5, 0.5, bad_val, 0.5, 0.5], [0.5] * 5)
        assert res_seq_ctx["category"] == "IMPLEMENTATION_INVALID"

        res_seq_alu = classify_experiment_outcome(pass_accs, [0.5] * 5, [0.5] * 5, [0.5, 0.5, bad_val, 0.5, 0.5])
        assert res_seq_alu["category"] == "IMPLEMENTATION_INVALID"


def test_decision_matrix_loss_bounds_and_checkpoint_validation() -> None:
    """Verifies that loss fields are validated to be finite and non-negative."""
    pass_accs = [1.0, 1.0, 1.0, 1.0, 1.0]

    # Valid non-negative losses
    res_valid_loss = classify_experiment_outcome(
        val_accuracies=pass_accs,
        c_read=0.85,
        s_context=0.85,
        c_alu=0.85,
        val_losses=[0.1, 0.2, 0.0, 0.3, 0.4],
    )
    assert res_valid_loss["category"] == "BEHAVIORAL_PASS_CANONICAL_FULL"
    assert res_valid_loss["metrics"]["val_loss_total"] == pytest.approx(0.2, abs=1e-5)
    assert res_valid_loss["per_seed_metrics"]["val_loss_total"] == [0.1, 0.2, 0.0, 0.3, 0.4]

    # Negative loss rejected
    res_neg_loss = classify_experiment_outcome(
        val_accuracies=pass_accs,
        c_read=0.85,
        s_context=0.85,
        c_alu=0.85,
        val_losses=[0.1, 0.2, -0.001, 0.3, 0.4],
    )
    assert res_neg_loss["category"] == "IMPLEMENTATION_INVALID"

    # Non-finite loss rejected
    res_nan_loss = classify_experiment_outcome(
        val_accuracies=pass_accs,
        c_read=0.85,
        s_context=0.85,
        c_alu=0.85,
        val_losses=[0.1, 0.2, float("nan"), 0.3, 0.4],
    )
    assert res_nan_loss["category"] == "IMPLEMENTATION_INVALID"

    res_inf_loss = classify_experiment_outcome(
        val_accuracies=pass_accs,
        c_read=0.85,
        s_context=0.85,
        c_alu=0.85,
        val_losses=[0.1, 0.2, float("inf"), 0.3, 0.4],
    )
    assert res_inf_loss["category"] == "IMPLEMENTATION_INVALID"

    # Invalid sequence length for losses
    res_short_loss = classify_experiment_outcome(
        val_accuracies=pass_accs,
        c_read=0.85,
        s_context=0.85,
        c_alu=0.85,
        val_losses=[0.1, 0.2, 0.3, 0.4],
    )
    assert res_short_loss["category"] == "IMPLEMENTATION_INVALID"

    # select_best_checkpoint rejects negative loss
    with pytest.raises(ValueError, match="negative val_loss_total"):
        select_best_checkpoint([{"update": 50, "val_loss_total": -0.1, "val_acc": 0.8}])

    # select_best_checkpoint rejects out-of-bounds accuracy
    with pytest.raises(ValueError, match="out-of-bounds val_acc"):
        select_best_checkpoint([{"update": 50, "val_loss_total": 0.1, "val_acc": 1.2}])


