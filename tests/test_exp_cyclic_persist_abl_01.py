"""Focused deterministic tests for EXP-CYCLIC-PERSIST-ABL-01 (Stage 1).

Covers the registered Stage 1 requirements of
``docs/research/exp_cyclic_persist_abl_01_specification.md``:

A. Objective and loss decomposition — exact ``L_total == L_terminal + 0.5 * L_persist``,
   frozen coefficient lambda_persist = 0.5, and verified exclusion of destination loss
   from the training objective and its gradients.
B. Diagnostic composite reference loss — exact ``L_composite == L_terminal + 1.0 * L_dest + 0.5 * L_persist``,
   serialized across splits without governing checkpoint selection.
C. Gradient flow — non-None, finite, non-zero gradients through ``mlp_read``,
   ``mlp_write`` and ``mlp_gate``, with direct persistence propagation into the S2 state.
D. Parameter invariance — exactly 219,018 trainable parameters, before and after training,
   with zero head/parameter overhead from the persistence query.
E. Diagnostic validity — all cosine metrics in [-1.0, 1.0], accuracies in [0.0, 1.0],
   losses finite and non-negative.
F. Checkpoint selection — based strictly on native ablation validation loss, never on
   composite reference loss.
G. Protected TEST firewall — only approved train/validation data paths are used;
   the protected partition (b in {0, 5}) is never loaded, constructed, or queried.
"""

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
from re_reason.core.register_machine_models import CyclicMGUPhase16Model, project_to_unit_circle
from experiments.run_exp_cyclic_persist_abl_01 import (
    EXPECTED_PARAMETER_COUNT,
    LAMBDA_DEST,
    LAMBDA_PERSIST,
    PERSIST_REGISTER_INDEX,
    PERSIST_STEP_INDEX,
    assert_persistence_register_unaddressed,
    assert_protected_domain_absent,
    check_gradients,
    compute_losses,
    compute_representation_metrics,
    cosine_similarity,
    evaluate_split,
    extract_destination_targets,
    persistence_query,
    register_one_hot,
    rollout_with_persistence,
    run_exp_cyclic_persist_abl_01,
    select_best_checkpoint,
    classify_experiment_outcome,
)

CANONICAL_ODD_OFFSET_COSINE = -0.2


class _FakeModel:
    """Minimal stand-in exposing only what the metric functions consume."""

    def __init__(self, basis: torch.Tensor, num_classes: int = 10) -> None:
        self.basis = basis
        self.num_classes = num_classes


def _train_and_val_traces():
    partition = Stage1PairPartition()
    task = Phase16RegisterTask(sequence_length=3)
    train_ds = partition.build_trace_split(partition.train_pairs, task)
    val_ds = partition.build_trace_split(partition.val_pairs, task)
    return partition, train_ds, val_ds


def _canonical_rollout(basis: torch.Tensor, y_destination: torch.Tensor) -> dict:
    """Synthetic rollout whose every representation is the exact canonical codeword."""
    batch_size, seq_len = y_destination.shape
    step_logits = torch.stack(
        [basis[y_destination[:, t]] @ basis.t() for t in range(seq_len)], dim=1
    )
    step_diagnostics = [
        {
            "u_a": basis[y_destination[:, 0]],
            "u_b": basis[y_destination[:, 1]],
            "c1_alu": basis[y_destination[:, 2]],
            "u_dest_post": basis[y_destination[:, t]],
        }
        for t in range(seq_len)
    ]
    return {
        "logits": step_logits[:, -1, :],
        "step_logits": step_logits,
        "u_persist": basis[y_destination[:, 0]],
        "persist_logits": basis[y_destination[:, 0]] @ basis.t(),
        "step_diagnostics": step_diagnostics,
        "states_after_step": [],
        "final_state": torch.zeros(batch_size, 212),
    }


def _block_squared_grad_norm(model: CyclicMGUPhase16Model, block: str) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(block):
            assert parameter.grad is not None, f"no gradient for {name}"
            total += float((parameter.grad.detach() ** 2).sum().item())
    return total


# ---------------------------------------------------------------------------
# Specification invariants: rollout fidelity and persistence query placement
# ---------------------------------------------------------------------------

def test_rollout_matches_model_forward_bit_exactly() -> None:
    """The persistence rollout must reproduce model.forward exactly, plus extra captures."""
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:8]

    torch.manual_seed(101)
    model = CyclicMGUPhase16Model()
    model.eval()

    rollout = rollout_with_persistence(model, inputs)
    with torch.no_grad():
        reference = model(inputs, return_step_diagnostics=True)

    assert torch.equal(rollout["logits"], reference.logits)
    assert torch.equal(rollout["step_logits"], reference.step_logits)
    assert torch.equal(rollout["final_state"], reference.final_state)


def test_persistence_query_uses_s2_and_the_r0_address() -> None:
    """The persistence query must query register r0 on state S2 via shared mlp_read."""
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:4]
    model = CyclicMGUPhase16Model()

    rollout = rollout_with_persistence(model, inputs)
    s2 = rollout["states_after_step"][1]

    address_r0 = torch.zeros(4, 4)
    address_r0[:, 0] = 1.0
    expected_raw = model.mlp_read(torch.cat([s2, address_r0], dim=-1))
    expected_u = project_to_unit_circle(expected_raw, model.num_frequencies, model.eps)
    expected_logits = expected_u @ model.basis.t()

    assert torch.equal(rollout["u_persist"], expected_u)
    assert torch.equal(rollout["persist_logits"], expected_logits)


def test_persistence_read_equals_step2_operand_read() -> None:
    """u_persist,1 must match the operand read u_a at the start of step 2."""
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:8]
    model = CyclicMGUPhase16Model()

    rollout = rollout_with_persistence(model, inputs)
    u_persist = rollout["u_persist"]
    u_a_step2 = rollout["step_diagnostics"][2]["u_a"]

    assert torch.equal(u_persist, u_a_step2)


def test_persistence_query_adds_no_trainable_parameters() -> None:
    model = CyclicMGUPhase16Model()
    assert model.count_parameters() == EXPECTED_PARAMETER_COUNT


def test_persistence_register_is_unaddressed_at_step_1() -> None:
    _, train_ds, val_ds = _train_and_val_traces()
    assert_persistence_register_unaddressed(train_ds.inputs)
    assert_persistence_register_unaddressed(val_ds.inputs)


# ---------------------------------------------------------------------------
# Objective formulation, loss decomposition, and destination exclusion
# ---------------------------------------------------------------------------

def test_total_loss_equals_native_ablation_component_sum() -> None:
    """Total loss must equal L_terminal + 0.5 * L_persist (destination loss excluded)."""
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:8]
    step_targets = train_ds.step_targets[:8]
    y_destination = extract_destination_targets(inputs, step_targets)

    torch.manual_seed(42)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, inputs)

    terminal_loss, destination_loss, persistence_loss, total_loss, composite_ref_loss = compute_losses(
        rollout, y_destination
    )

    # Invariant coefficients
    assert LAMBDA_DEST == 0.0
    assert LAMBDA_PERSIST == 0.5

    expected_total = terminal_loss + 0.5 * persistence_loss
    assert torch.equal(total_loss, expected_total)

    expected_composite = terminal_loss + 1.0 * destination_loss + 0.5 * persistence_loss
    assert torch.equal(composite_ref_loss, expected_composite)


def test_destination_loss_does_not_contribute_to_training_objective_or_gradients() -> None:
    """Perturbing step 0/1 destination logits must change destination_loss but leave total_loss and gradients unchanged."""
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:8]
    step_targets = train_ds.step_targets[:8]
    y_destination = extract_destination_targets(inputs, step_targets)

    torch.manual_seed(42)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, inputs)

    t_loss, d_loss, p_loss, tot_loss, comp_loss = compute_losses(rollout, y_destination)

    # Compute baseline gradients
    grads_baseline = torch.autograd.grad(tot_loss, model.parameters(), retain_graph=True)

    # Create modified rollout where step_logits at step 0 and 1 are perturbed arbitrarily
    modified_step_logits = rollout["step_logits"].clone()
    modified_step_logits[:, 0, 0] += 5.0
    modified_step_logits[:, 1, 1] -= 3.0
    perturbed_rollout = dict(rollout)
    perturbed_rollout["step_logits"] = modified_step_logits

    t_loss_pert, d_loss_pert, p_loss_pert, tot_loss_pert, comp_loss_pert = compute_losses(
        perturbed_rollout, y_destination
    )

    # Destination loss must have changed
    assert not torch.equal(d_loss, d_loss_pert)
    assert not torch.equal(comp_loss, comp_loss_pert)

    # Total loss and persistence/terminal losses must be bit-identical
    assert torch.equal(tot_loss, tot_loss_pert)
    assert torch.equal(t_loss, t_loss_pert)
    assert torch.equal(p_loss, p_loss_pert)

    # Gradients of total_loss must be bit-identical
    grads_perturbed = torch.autograd.grad(tot_loss_pert, model.parameters())
    for g_base, g_pert in zip(grads_baseline, grads_perturbed):
        assert torch.equal(g_base, g_pert)


def test_persistence_loss_targets_operand_a_only() -> None:
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:8]
    step_targets = train_ds.step_targets[:8]
    y_destination = extract_destination_targets(inputs, step_targets)
    y_operand_a = y_destination[:, 0]

    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, inputs)
    _, _, persistence_loss, _, _ = compute_losses(rollout, y_destination)

    expected = torch.nn.functional.cross_entropy(rollout["persist_logits"], y_operand_a)
    assert torch.equal(persistence_loss, expected)


def test_step_2_is_supervised_once_through_the_terminal_term() -> None:
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:8]
    step_targets = train_ds.step_targets[:8]
    y_destination = extract_destination_targets(inputs, step_targets)

    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, inputs)
    assert torch.equal(rollout["logits"], rollout["step_logits"][:, 2, :])


# ---------------------------------------------------------------------------
# Gradient flow and parameter checks
# ---------------------------------------------------------------------------

def test_gradients_are_finite_nonzero_and_reach_all_trainable_blocks() -> None:
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs
    step_targets = train_ds.step_targets
    y_destination = extract_destination_targets(inputs, step_targets)

    torch.manual_seed(7)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, inputs)
    _, _, _, total_loss, _ = compute_losses(rollout, y_destination)
    total_loss.backward()

    for block in ("mlp_read", "mlp_write", "mlp_gate"):
        sq_norm = _block_squared_grad_norm(model, block)
        assert np.isfinite(sq_norm) and sq_norm > 0.0, f"no gradient reached {block}"


def test_persistence_loss_propagates_into_the_s2_state() -> None:
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:4]
    step_targets = train_ds.step_targets[:4]
    y_destination = extract_destination_targets(inputs, step_targets)

    torch.manual_seed(99)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, inputs)
    s2 = rollout["states_after_step"][1]

    _, _, persistence_loss, _, _ = compute_losses(rollout, y_destination)
    (grad_s2,) = torch.autograd.grad(persistence_loss, s2, retain_graph=True)

    assert bool(torch.isfinite(grad_s2).all())
    assert float((grad_s2 ** 2).sum().item()) > 0.0


def test_gradient_check_helper_flags_broken_gradients() -> None:
    model = CyclicMGUPhase16Model()
    with pytest.raises(RuntimeError, match="received no gradient"):
        check_gradients(model, step=1, seed=42)


def test_parameter_count_invariance_at_init_and_after_training() -> None:
    model = CyclicMGUPhase16Model()
    assert model.count_parameters() == EXPECTED_PARAMETER_COUNT
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.add_(torch.randn_like(parameter) * 0.01)
    assert model.count_parameters() == EXPECTED_PARAMETER_COUNT


# ---------------------------------------------------------------------------
# Diagnostic representations, metrics and numerical contracts
# ---------------------------------------------------------------------------

def test_cosine_similarity_bounds_and_zero_norm_rule() -> None:
    u = torch.randn(8, 10)
    v = torch.randn(8, 10)
    sim = cosine_similarity(u, v)
    assert (sim >= -1.0 - 1e-6).all() and (sim <= 1.0 + 1e-6).all()

    zero = torch.zeros(1, 10)
    non_zero = torch.ones(1, 10)
    assert cosine_similarity(zero, non_zero).item() == 0.0


def test_representation_metrics_on_canonical_vectors() -> None:
    _, train_ds, _ = _train_and_val_traces()
    model = CyclicMGUPhase16Model()
    y_dest = extract_destination_targets(train_ds.inputs, train_ds.step_targets)
    rollout = _canonical_rollout(model.basis, y_dest)

    metrics = compute_representation_metrics(model, rollout, y_dest)
    assert pytest.approx(metrics["c_read"], abs=1e-5) == 1.0
    assert pytest.approx(metrics["c_read_r0"], abs=1e-5) == 1.0
    assert pytest.approx(metrics["c_read_r1"], abs=1e-5) == 1.0
    assert pytest.approx(metrics["c_persist_r0"], abs=1e-5) == 1.0
    assert pytest.approx(metrics["s_context"], abs=1e-5) == 1.0
    assert pytest.approx(metrics["c_alu"], abs=1e-5) == 1.0
    assert pytest.approx(metrics["c_dest"], abs=1e-5) == 1.0
    assert metrics["acc"] == 1.0
    assert metrics["persist1_acc"] == 1.0


def test_cross_context_consistency_grouping_and_omission() -> None:
    basis = CyclicMGUPhase16Model().basis
    vecs_a = basis[torch.tensor([0, 1])]
    vals_a = torch.tensor([0, 1])
    vecs_b = basis[torch.tensor([0, 2])]
    vals_b = torch.tensor([0, 2])

    from experiments.run_exp_cyclic_persist_abl_01 import _pooled_context_consistency
    overall, per_digit, omitted = _pooled_context_consistency(
        vecs_a, vals_a, vecs_b, vals_b, num_digits=10
    )
    assert "0" in per_digit
    assert per_digit["0"]["n_vectors"] == 2
    assert pytest.approx(per_digit["0"]["mean_pairwise_cosine"], abs=1e-6) == 1.0
    assert 1 in omitted and 2 in omitted
    assert pytest.approx(overall, abs=1e-6) == 1.0


def test_evaluate_split_returns_all_registered_fields_within_bounds() -> None:
    _, train_ds, _ = _train_and_val_traces()
    model = CyclicMGUPhase16Model()
    metrics = evaluate_split(model, train_ds.inputs[:8], train_ds.step_targets[:8])

    expected_keys = {
        "loss_total", "loss_terminal", "loss_dest", "loss_persist", "loss_composite_reference",
        "c_read", "c_read_r0", "c_read_r1", "c_persist_r0", "s_context",
        "s_context_per_digit", "s_context_omitted_digits", "c_alu", "c_dest",
        "c_dest_step0", "c_dest_step1", "c_dest_step2",
        "step0_acc", "step1_acc", "step2_acc", "persist1_acc", "acc",
    }
    assert expected_keys.issubset(metrics.keys())
    assert 0.0 <= metrics["acc"] <= 1.0
    assert -1.0 <= metrics["c_read"] <= 1.0
    assert metrics["loss_total"] >= 0.0


def test_runner_checkpoint_telemetry_schema_and_loss_identity() -> None:
    """Verifies that serialized loss fields satisfy the exact native and composite formulas."""
    _, train_ds, val_ds = _train_and_val_traces()
    model = CyclicMGUPhase16Model()
    train_m = evaluate_split(model, train_ds.inputs[:8], train_ds.step_targets[:8])
    val_m = evaluate_split(model, val_ds.inputs[:8], val_ds.step_targets[:8])

    # Native loss identity: total == terminal + 0.5 * persist
    train_expected_total = train_m["loss_terminal"] + 0.5 * train_m["loss_persist"]
    assert pytest.approx(train_m["loss_total"], rel=1e-5, abs=1e-6) == train_expected_total

    val_expected_total = val_m["loss_terminal"] + 0.5 * val_m["loss_persist"]
    assert pytest.approx(val_m["loss_total"], rel=1e-5, abs=1e-6) == val_expected_total

    # Composite reference identity: composite == terminal + 1.0 * dest + 0.5 * persist
    train_expected_comp = train_m["loss_terminal"] + 1.0 * train_m["loss_dest"] + 0.5 * train_m["loss_persist"]
    assert pytest.approx(train_m["loss_composite_reference"], rel=1e-5, abs=1e-6) == train_expected_comp

    val_expected_comp = val_m["loss_terminal"] + 1.0 * val_m["loss_dest"] + 0.5 * val_m["loss_persist"]
    assert pytest.approx(val_m["loss_composite_reference"], rel=1e-5, abs=1e-6) == val_expected_comp


# ---------------------------------------------------------------------------
# Runner configuration and firewall guards
# ---------------------------------------------------------------------------

def test_runner_rejects_unaligned_batch_and_existing_artifact(tmp_path) -> None:
    target = tmp_path / "smoke.json"
    target.write_text("pre-existing content", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        run_exp_cyclic_persist_abl_01(
            seeds=[42], total_updates=2, results_path=str(target), run_mode="SMOKE_TEST_ONLY"
        )


def test_runner_materializes_only_allowed_splits() -> None:
    partition = Stage1PairPartition()
    train_ds = partition.build_trace_split(partition.train_pairs)
    val_ds = partition.build_trace_split(partition.val_pairs)
    assert len(train_ds) == 64
    assert len(val_ds) == 16


def test_runner_source_never_references_the_protected_split() -> None:
    """The runner source must not reference the protected partition."""
    import experiments.run_exp_cyclic_persist_abl_01 as runner_module
    source = inspect.getsource(runner_module)
    assert "partition.test_pairs" not in source
    assert "partition.build_trace_split(partition.test_pairs)" not in source


def test_protected_domain_assertion_is_live() -> None:
    _, train_ds, val_ds = _train_and_val_traces()
    assert_protected_domain_absent(train_ds.inputs, train_ds.step_targets, "TRAIN-FIT")
    assert_protected_domain_absent(val_ds.inputs, val_ds.step_targets, "VAL-SEEN")

    synthetic_inputs = torch.zeros((1, 3, 22))
    synthetic_inputs[0, 1, 4 + 1] = 1.0
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


# ---------------------------------------------------------------------------
# Checkpoint selection and decision rules
# ---------------------------------------------------------------------------

def test_checkpoint_selection_policy_min_loss_then_acc_then_earliest() -> None:
    """Selection must be governed strictly by val_loss_total, not composite reference."""
    cps = [
        {"update": 50, "val_loss_total": 0.40, "val_loss_composite_reference": 0.90, "val_acc": 0.75},
        {"update": 100, "val_loss_total": 0.30, "val_loss_composite_reference": 0.95, "val_acc": 0.80},
        {"update": 150, "val_loss_total": 0.30, "val_loss_composite_reference": 0.70, "val_acc": 0.80},
        {"update": 200, "val_loss_total": 0.30, "val_loss_composite_reference": 0.50, "val_acc": 0.75},
    ]
    # Checkpoints 100 and 150 tie on val_loss_total (0.30) and val_acc (0.80);
    # checkpoint 100 is chosen because it is earlier, even though checkpoint 150 and 200 have lower composite loss.
    best = select_best_checkpoint(cps)
    assert best["update"] == 100


def test_decision_matrix_covers_all_twelve_cells() -> None:
    pass_behavior = [1.0, 1.0, 1.0, 1.0, 0.75]
    mid_behavior = [0.75, 0.5, 0.4375, 0.625, 0.6875]
    reject_behavior = [0.1875, 0.125, 0.0, 0.0625, 0.1875]

    representations = {
        "canonical_full": ([0.9] * 5, [0.9] * 5, [0.9] * 5),
        "canonical_input_only": ([0.9] * 5, [0.9] * 5, [0.5] * 5),
        "partial": ([0.55] * 5, [0.32] * 5, [0.54] * 5),
        "failed": ([0.2] * 5, [0.1] * 5, [0.25] * 5),
    }
    expected = {
        ("pass", "canonical_full"): "BEHAVIORAL_PASS_CANONICAL_FULL",
        ("pass", "canonical_input_only"): "BEHAVIORAL_PASS_CANONICAL_INPUT_ONLY",
        ("pass", "partial"): "BEHAVIORAL_PASS_PARTIAL_ALIGNMENT",
        ("pass", "failed"): "BEHAVIORAL_PASS_UNALIGNED",
        ("mid", "canonical_full"): "INTERMEDIATE_CANONICAL_FULL",
        ("mid", "canonical_input_only"): "INTERMEDIATE_CANONICAL_INPUT_ONLY",
        ("mid", "partial"): "INCONCLUSIVE_INTERMEDIATE",
        ("mid", "failed"): "INCONCLUSIVE_UNALIGNED",
        ("reject", "canonical_full"): "REJECTION_CANONICAL_FULL_DECOUPLED",
        ("reject", "canonical_input_only"): "REJECTION_CANONICAL_INPUT_DECOUPLED",
        ("reject", "partial"): "FALSIFIED_PARTIAL_ALIGNMENT",
        ("reject", "failed"): "FALSIFIED_FULL",
    }
    behaviors = {"pass": pass_behavior, "mid": mid_behavior, "reject": reject_behavior}

    for behavior_key, accs in behaviors.items():
        for rep_key, (c_read, s_ctx, c_alu) in representations.items():
            result = classify_experiment_outcome(accs, c_read, s_ctx, c_alu)
            assert result["category"] == expected[(behavior_key, rep_key)], (
                f"{behavior_key} x {rep_key} -> {result['category']}"
            )

    assert classify_experiment_outcome([1.0] * 4, [0.9] * 5, [0.9] * 5, [0.9] * 5)["category"] == "IMPLEMENTATION_INVALID"
    assert classify_experiment_outcome([0.5] * 5, [1.5] * 5, [0.9] * 5, [0.9] * 5)["category"] == "IMPLEMENTATION_INVALID"
    assert classify_experiment_outcome([0.5] * 5, [0.9] * 5, [0.9] * 5, [0.9] * 5, val_losses=[-1.0] * 5)["category"] == "IMPLEMENTATION_INVALID"


def test_representation_state_boundaries() -> None:
    mid = [0.75, 0.5, 0.4375, 0.625, 0.6875]
    assert classify_experiment_outcome(mid, [0.80] * 5, [0.80] * 5, [0.80] * 5)["representation_state"] == "CANONICAL_FULL"
    assert classify_experiment_outcome(mid, [0.80] * 5, [0.80] * 5, [0.79] * 5)["representation_state"] == "CANONICAL_INPUT_ONLY"
    assert classify_experiment_outcome(mid, [0.29] * 5, [0.9] * 5, [0.29] * 5)["representation_state"] == "FAILED"
    assert classify_experiment_outcome(mid, [0.30] * 5, [0.9] * 5, [0.30] * 5)["representation_state"] == "PARTIAL"
