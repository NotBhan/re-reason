"""Focused deterministic tests for EXP-CYCLIC-PERSIST-01 (Stage 1).

Covers the registered Stage 1 requirements of
``docs/research/exp_cyclic_persist_01_specification.md``:

A. Loss decomposition — exact ``L_total == L_terminal + 1.0 * L_dest + 0.5 * L_persist``,
   frozen coefficients, and component-level correctness.
B. Gradient flow — non-None, finite, non-zero gradients through ``mlp_read``,
   ``mlp_write`` and ``mlp_gate``, including direct propagation into the S2 state that
   carries the unaddressed register r0.
C. Parameter invariance — exactly 219,018 trainable parameters, before and after training,
   with zero head/parameter overhead from the persistence query.
D. Diagnostic validity — all cosine metrics in [-1.0, 1.0], accuracies in [0.0, 1.0],
   losses finite and non-negative.
E. Protected TEST firewall — only approved train/validation data paths are used.

Specification-specific invariants verified here:
* the persistence rollout is bit-identical to ``model.forward``;
* ``u_persist,1`` is queried on S2 with ``e_r0`` (never S1, never the step-1 destination read);
* ``u_persist,1`` equals the step-2 operand read ``u_a`` (specification 5.1 equivalence);
* the persistence register is genuinely unaddressed at step 1;
* the imported checkpoint-selection policy and 3x4 decision matrix match specification 6 and 9.

The protected TEST split (operand pairs with b in {0, 5}) is never loaded, constructed,
evaluated or queried anywhere in this file.
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
from experiments.run_exp_cyclic_persist_01 import (
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
    run_exp_cyclic_persist_01,
    select_best_checkpoint,
    classify_experiment_outcome,
)

# cos(Phi(v), Phi(v + d)) for the fixed 5-harmonic basis: -1/5 for odd d, 0 for even
# non-zero d, 1 for d == 0 (same analytic reference used by the predecessor suite).
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
    assert torch.equal(rollout["logits"], rollout["step_logits"][:, -1, :])

    diagnostics = reference.diagnostics
    assert diagnostics is not None
    assert torch.equal(rollout["step_diagnostics"][2]["u_a"], diagnostics.u_a)
    assert torch.equal(rollout["step_diagnostics"][2]["u_b"], diagnostics.u_b)
    assert torch.equal(rollout["step_diagnostics"][2]["c1_alu"], diagnostics.c1_alu)
    stacked = torch.stack([diag["u_dest_post"] for diag in rollout["step_diagnostics"]], dim=1)
    assert torch.equal(stacked, diagnostics.u_dest_post)


def test_persistence_query_uses_s2_and_the_r0_address() -> None:
    """u_persist must be project(mlp_read([S2, e_r0])) — not S1, not the destination read."""
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:6]
    batch_size = inputs.shape[0]

    torch.manual_seed(103)
    model = CyclicMGUPhase16Model()
    model.eval()

    with torch.no_grad():
        rollout = rollout_with_persistence(model, inputs)
        state_1 = rollout["states_after_step"][0]
        state_2 = rollout["states_after_step"][1]

        address = register_one_hot(batch_size, PERSIST_REGISTER_INDEX, model.num_registers, state_2)
        assert address.shape == (batch_size, 4)
        assert torch.equal(address[:, PERSIST_REGISTER_INDEX], torch.ones(batch_size))
        assert float(address.sum().item()) == float(batch_size)

        def read_from(state):
            return project_to_unit_circle(
                model.mlp_read(torch.cat([state, address], dim=-1)),
                model.num_frequencies,
                model.eps,
            )

        assert torch.equal(read_from(state_2), rollout["u_persist"])
        # S1 is the pre-transition state: its r0 read is a different vector.
        assert not torch.equal(read_from(state_1), rollout["u_persist"])
        # The step-1 destination read addresses r1, not r0.
        assert not torch.equal(rollout["step_diagnostics"][1]["u_dest_post"], rollout["u_persist"])

    assert rollout["u_persist"].shape == (batch_size, 10)
    assert rollout["persist_logits"].shape == (batch_size, 10)
    block_norms = torch.norm(rollout["u_persist"].view(batch_size, 5, 2), p=2, dim=-1)
    assert torch.allclose(block_norms, torch.ones_like(block_norms), atol=1e-5)


def test_persistence_read_equals_step2_operand_read() -> None:
    """Specification 5.1: u_persist,1 is computationally identical to the step-2 u_a."""
    _, train_ds, val_ds = _train_and_val_traces()

    torch.manual_seed(107)
    model = CyclicMGUPhase16Model()
    model.eval()

    for name, inputs in (("TRAIN-FIT", train_ds.inputs), ("VAL-SEEN", val_ds.inputs)):
        with torch.no_grad():
            rollout = rollout_with_persistence(model, inputs)
        assert torch.equal(rollout["u_persist"], rollout["step_diagnostics"][2]["u_a"]), (
            f"[{name}] the persistence read no longer equals the step-2 r0 operand read"
        )
        assert torch.equal(
            rollout["persist_logits"], torch.einsum("bd,vd->bv", rollout["u_persist"], model.basis)
        )


def test_persistence_query_adds_no_trainable_parameters() -> None:
    """The persistence query reuses the shared mlp_read: no new modules, parameters or buffers."""
    _, train_ds, _ = _train_and_val_traces()
    inputs = train_ds.inputs[:4]

    torch.manual_seed(109)
    model = CyclicMGUPhase16Model()
    before_params = [name for name, _ in model.named_parameters()]
    before_buffers = [name for name, _ in model.named_buffers()]

    with torch.no_grad():
        rollout_with_persistence(model, inputs)
        persistence_query(model, model.initial_state(4, torch.device("cpu")), 4)

    assert [name for name, _ in model.named_parameters()] == before_params
    assert [name for name, _ in model.named_buffers()] == before_buffers == ["basis"]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == EXPECTED_PARAMETER_COUNT
    assert len(list(model.parameters())) == 12  # three two-layer MLPs (weight + bias each)


def test_persistence_register_is_unaddressed_at_step_1() -> None:
    """The intervention requires r0 to be unaddressed at the step-1 transition."""
    _, train_ds, val_ds = _train_and_val_traces()
    assert_persistence_register_unaddressed(train_ds.inputs)
    assert_persistence_register_unaddressed(val_ds.inputs)

    destination_codes = train_ds.inputs[:, :, 4:8].argmax(dim=-1)
    assert bool((destination_codes[:, PERSIST_STEP_INDEX] == 1).all()), "step 1 should address r1"
    assert PERSIST_REGISTER_INDEX == 0
    assert PERSIST_STEP_INDEX == 1

    # The guard must fire when r0 is the step-1 destination.
    crafted = train_ds.inputs[:2].clone()
    crafted[:, PERSIST_STEP_INDEX, 4:8] = 0.0
    crafted[:, PERSIST_STEP_INDEX, 4 + PERSIST_REGISTER_INDEX] = 1.0
    with pytest.raises(RuntimeError, match="unaddressed"):
        assert_persistence_register_unaddressed(crafted)


# ---------------------------------------------------------------------------
# A. Loss decomposition
# ---------------------------------------------------------------------------

def test_total_loss_equals_weighted_component_sum() -> None:
    """L_total == L_terminal + 1.0 * L_dest + 0.5 * L_persist, verified exactly."""
    _, train_ds, _ = _train_and_val_traces()

    torch.manual_seed(113)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, train_ds.inputs)
    y = extract_destination_targets(train_ds.inputs, train_ds.step_targets)

    terminal_loss, destination_loss, persistence_loss, total_loss = compute_losses(rollout, y)

    assert LAMBDA_DEST == 1.0
    assert LAMBDA_PERSIST == 0.5

    # Independent recomputation straight from the raw logits (no argmax anywhere).
    reference_terminal = torch.nn.functional.cross_entropy(rollout["logits"], y[:, -1])
    reference_step_0 = torch.nn.functional.cross_entropy(rollout["step_logits"][:, 0, :], y[:, 0])
    reference_step_1 = torch.nn.functional.cross_entropy(rollout["step_logits"][:, 1, :], y[:, 1])
    reference_dest = (reference_step_0 + reference_step_1) / 2.0
    reference_persist = torch.nn.functional.cross_entropy(rollout["persist_logits"], y[:, 0])

    assert torch.equal(terminal_loss, reference_terminal)
    assert torch.equal(destination_loss, reference_dest)
    assert torch.equal(persistence_loss, reference_persist)
    assert torch.equal(total_loss, reference_terminal + 1.0 * reference_dest + 0.5 * reference_persist)

    # The 0.5 coefficient is load-bearing: a unit persistence weight would differ.
    unit_persist_total = reference_terminal + 1.0 * reference_dest + 1.0 * reference_persist
    assert not torch.isclose(total_loss, unit_persist_total, atol=1e-9)

    for tensor in (terminal_loss, destination_loss, persistence_loss, total_loss):
        assert bool(torch.isfinite(tensor).all())
        assert float(tensor.item()) >= 0.0
        assert tensor.requires_grad


def test_persistence_loss_targets_operand_a_only() -> None:
    """L_persist must be CE(persist_logits, a): the r0 value written at step 0."""
    _, train_ds, _ = _train_and_val_traces()

    torch.manual_seed(127)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, train_ds.inputs)
    y = extract_destination_targets(train_ds.inputs, train_ds.step_targets)

    _, _, persistence_loss, _ = compute_losses(rollout, y)
    assert torch.equal(
        persistence_loss,
        torch.nn.functional.cross_entropy(rollout["persist_logits"], y[:, 0]),
    )
    # Not the step-1 destination target b, and not the terminal sum.
    assert not torch.equal(
        persistence_loss,
        torch.nn.functional.cross_entropy(rollout["persist_logits"], y[:, 1]),
    )


def test_step_2_is_supervised_once_through_the_terminal_term() -> None:
    _, train_ds, _ = _train_and_val_traces()

    torch.manual_seed(131)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, train_ds.inputs)
    y = extract_destination_targets(train_ds.inputs, train_ds.step_targets)

    terminal_loss, _, _, _ = compute_losses(rollout, y)
    assert torch.equal(terminal_loss, torch.nn.functional.cross_entropy(rollout["step_logits"][:, 2, :], y[:, 2]))
    assert torch.equal(rollout["logits"], rollout["step_logits"][:, -1, :])


# ---------------------------------------------------------------------------
# B. Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_are_finite_nonzero_and_reach_all_trainable_blocks() -> None:
    _, train_ds, _ = _train_and_val_traces()

    torch.manual_seed(137)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, train_ds.inputs)
    y = extract_destination_targets(train_ds.inputs, train_ds.step_targets)

    _, _, _, total_loss = compute_losses(rollout, y)
    total_loss.backward()

    parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter_count += 1
        assert parameter.grad is not None, f"parameter {name} received no gradient"
        assert bool(torch.isfinite(parameter.grad).all()), f"non-finite gradient for {name}"
    assert parameter_count == 12

    for block in ("mlp_read", "mlp_write", "mlp_gate"):
        assert _block_squared_grad_norm(model, block) > 1e-12, f"gradient collapsed at {block}"


def test_persistence_loss_propagates_into_the_s2_state() -> None:
    """Direct gradient propagation into the S2 state carrying the unaddressed register r0."""
    _, train_ds, _ = _train_and_val_traces()

    torch.manual_seed(139)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, train_ds.inputs)
    y = extract_destination_targets(train_ds.inputs, train_ds.step_targets)

    state_2 = rollout["states_after_step"][PERSIST_STEP_INDEX]
    state_2.retain_grad()

    _, _, persistence_loss, _ = compute_losses(rollout, y)
    assert persistence_loss.requires_grad
    assert rollout["u_persist"].grad_fn is not None, "the persistence read is detached from the graph"

    persistence_loss.backward()

    assert state_2.grad is not None, "L_persist produced no gradient into S2"
    assert bool(torch.isfinite(state_2.grad).all())
    assert float(state_2.grad.detach().abs().sum().item()) > 0.0, "L_persist gradient into S2 is zero"
    assert _block_squared_grad_norm(model, "mlp_read") > 1e-12


def test_gradient_check_helper_flags_broken_gradients() -> None:
    """The per-update gradient audit must actually fail on a detached computation."""
    _, train_ds, _ = _train_and_val_traces()

    torch.manual_seed(149)
    model = CyclicMGUPhase16Model()
    rollout = rollout_with_persistence(model, train_ds.inputs)
    y = extract_destination_targets(train_ds.inputs, train_ds.step_targets)

    _, _, _, total_loss = compute_losses(rollout, y)
    total_loss.backward()
    check_gradients(model, 1, 42)  # must not raise on a healthy graph

    model.zero_grad()
    with pytest.raises(RuntimeError, match="received no gradient"):
        check_gradients(model, 2, 42)


# ---------------------------------------------------------------------------
# C. Parameter invariance
# ---------------------------------------------------------------------------

def test_parameter_count_invariance_at_init_and_after_training() -> None:
    model = CyclicMGUPhase16Model()
    assert model.count_parameters() == EXPECTED_PARAMETER_COUNT == 219018
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 219018
    assert model.mlp_read[0].weight.shape == (128, 216)
    assert model.mlp_write[0].weight.shape == (212, 32)
    assert model.mlp_gate[0].weight.shape == (212, 436)

    payload = run_exp_cyclic_persist_01(
        seeds=[42],
        total_updates=2,
        val_cadence=1,
        device=torch.device("cpu"),
        results_path="",
        run_mode="SMOKE_TEST_ONLY",
    )
    run = payload["runs"][0]
    assert run["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert run["parameter_count_after_training"] == EXPECTED_PARAMETER_COUNT


# ---------------------------------------------------------------------------
# D. Diagnostic validity
# ---------------------------------------------------------------------------

def test_cosine_similarity_bounds_and_zero_norm_rule() -> None:
    e1 = torch.tensor([[1.0, 0.0, 0.0]])
    e2 = torch.tensor([[0.0, 1.0, 0.0]])
    zero = torch.zeros((1, 3))

    assert cosine_similarity(e1, e1).item() == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(e1, e2).item() == pytest.approx(0.0, abs=1e-6)
    assert cosine_similarity(e1, -e1).item() == pytest.approx(-1.0, abs=1e-6)
    assert cosine_similarity(e1, zero).item() == 0.0
    assert cosine_similarity(zero, zero).item() == 0.0


def test_representation_metrics_on_canonical_vectors() -> None:
    """Canonical codewords must score exactly 1.0 on every registered metric."""
    basis = CyclicMGUPhase16Model().basis
    operand_a = torch.tensor([3, 7, 3, 7])
    operand_b = torch.tensor([4, 2, 4, 2])
    y_destination = torch.stack([operand_a, operand_b, (operand_a + operand_b) % 10], dim=1)

    rollout = _canonical_rollout(basis, y_destination)
    metrics = compute_representation_metrics(_FakeModel(basis), rollout, y_destination)

    assert metrics["c_read_r0"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_read_r1"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_read"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_persist_r0"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["s_context"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_alu"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_dest"] == pytest.approx(1.0, abs=1e-6)
    for t in range(3):
        assert metrics[f"c_dest_step{t}"] == pytest.approx(1.0, abs=1e-6)
        assert metrics[f"step{t}_acc"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["persist1_acc"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["acc"] == pytest.approx(1.0, abs=1e-6)
    assert sorted(int(k) for k in metrics["s_context_per_digit"]) == [2, 3, 4, 7]
    assert metrics["s_context_omitted_digits"] == [0, 1, 5, 6, 8, 9]


def test_persistence_metric_detects_a_corrupted_r0_read() -> None:
    """A rotated persistence read must be penalised by c_persist while other metrics hold."""
    basis = CyclicMGUPhase16Model().basis
    operand_a = torch.tensor([3, 7, 3, 7])
    operand_b = torch.tensor([4, 2, 4, 2])
    y_destination = torch.stack([operand_a, operand_b, (operand_a + operand_b) % 10], dim=1)

    rollout = _canonical_rollout(basis, y_destination)
    rollout["u_persist"] = basis[(y_destination[:, 0] + 1) % 10]  # corrupted r0 read
    rollout["persist_logits"] = rollout["u_persist"] @ basis.t()

    metrics = compute_representation_metrics(_FakeModel(basis), rollout, y_destination)

    assert metrics["c_persist_r0"] == pytest.approx(CANONICAL_ODD_OFFSET_COSINE, abs=1e-5)
    # The step-2 read alignment is a separate quantity and is unaffected here.
    assert metrics["c_read_r0"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["c_alu"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["persist1_acc"] == pytest.approx(0.0, abs=1e-9)


def test_cross_context_consistency_grouping_and_omission() -> None:
    from experiments.run_exp_cyclic_persist_01 import _pooled_context_consistency

    def unit(index: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(index), 10).float()

    vectors_a = torch.stack([unit(0), unit(1), unit(4), unit(5)])
    values_a = torch.tensor([3, 7, 5, 9])
    vectors_b = torch.stack([unit(0), unit(2), unit(6), unit(8)])
    values_b = torch.tensor([3, 7, 0, 2])

    overall, per_digit, omitted = _pooled_context_consistency(vectors_a, values_a, vectors_b, values_b, 10)

    assert per_digit["3"]["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert per_digit["3"]["n_vectors"] == 2
    assert per_digit["7"]["mean_pairwise_cosine"] == pytest.approx(0.0, abs=1e-6)
    assert overall == pytest.approx(0.5, abs=1e-6)
    assert sorted(omitted) == [0, 1, 2, 4, 5, 6, 8, 9]


def test_evaluate_split_returns_all_registered_fields_within_bounds() -> None:
    _, train_ds, val_ds = _train_and_val_traces()

    torch.manual_seed(151)
    model = CyclicMGUPhase16Model()

    for name, dataset in (("TRAIN-FIT", train_ds), ("VAL-SEEN", val_ds)):
        metrics = evaluate_split(model, dataset.inputs, dataset.step_targets)

        required = {
            "loss_total", "loss_terminal", "loss_dest", "loss_persist", "acc",
            "c_read", "c_read_r0", "c_read_r1", "c_persist_r0", "s_context",
            "s_context_per_digit", "s_context_omitted_digits", "c_alu", "c_dest",
            "c_dest_step0", "c_dest_step1", "c_dest_step2",
            "step0_acc", "step1_acc", "step2_acc", "persist1_acc",
        }
        assert required <= set(metrics), f"[{name}] missing fields"

        for key in ("loss_total", "loss_terminal", "loss_dest", "loss_persist"):
            assert np.isfinite(metrics[key]) and metrics[key] >= 0.0, f"[{name}] {key}"
        for key in ("acc", "step0_acc", "step1_acc", "step2_acc", "persist1_acc"):
            assert 0.0 <= metrics[key] <= 1.0, f"[{name}] {key}"
        for key in ("c_read", "c_read_r0", "c_read_r1", "c_persist_r0", "c_alu",
                    "c_dest", "c_dest_step0", "c_dest_step1", "c_dest_step2"):
            assert -1.0 <= metrics[key] <= 1.0, f"[{name}] {key}"
        assert metrics["s_context"] is not None and -1.0 <= metrics["s_context"] <= 1.0
        assert metrics["s_context_omitted_digits"] == []
        assert metrics["c_read"] == pytest.approx(0.5 * (metrics["c_read_r0"] + metrics["c_read_r1"]), abs=1e-9)


# ---------------------------------------------------------------------------
# Runner integration, telemetry schema and firewall
# ---------------------------------------------------------------------------

def test_runner_checkpoint_telemetry_schema_and_loss_identity() -> None:
    payload = run_exp_cyclic_persist_01(
        seeds=[42],
        total_updates=2,
        val_cadence=1,
        batch_size=64,
        device=torch.device("cpu"),
        results_path="",
        run_mode="SMOKE_TEST_ONLY",
    )

    assert payload["experiment_id"] == "EXP-CYCLIC-PERSIST-01"
    assert payload["loss_definition"]["lambda_dest"] == 1.0
    assert payload["loss_definition"]["lambda_persist"] == 0.5
    assert payload["protected_test_split"]["queries"] == 0
    assert payload["gate_assessment"] is None

    run = payload["runs"][0]
    assert run["seed"] == 42
    assert run["finite_value_checks_passed"] == 2
    assert sorted(run["gradient_blocks_present"]) == ["mlp_gate", "mlp_read", "mlp_write"]
    assert len(run["checkpoints"]) == 2

    expected_train = {
        "train_loss_total", "train_loss_terminal", "train_loss_dest", "train_loss_persist",
        "train_acc", "train_c_read", "train_c_read_r0", "train_c_read_r1", "train_c_persist_r0",
        "train_c_alu", "train_c_dest", "train_c_dest_step0", "train_c_dest_step1", "train_c_dest_step2",
        "train_step0_acc", "train_step1_acc", "train_step2_acc", "train_persist1_acc",
        "train_s_context", "train_s_context_per_digit", "train_s_context_omitted_digits",
    }
    expected_val = {key.replace("train_", "val_", 1) for key in expected_train}

    for checkpoint in run["checkpoints"]:
        assert set(checkpoint) == {"update"} | expected_train | expected_val
        for prefix in ("train_", "val_"):
            total = checkpoint[f"{prefix}loss_total"]
            terminal = checkpoint[f"{prefix}loss_terminal"]
            dest = checkpoint[f"{prefix}loss_dest"]
            persist = checkpoint[f"{prefix}loss_persist"]
            assert total == pytest.approx(terminal + 1.0 * dest + 0.5 * persist, rel=1e-6)


def test_runner_rejects_unaligned_batch_and_existing_artifact() -> None:
    with pytest.raises(RuntimeError, match="registered batch size"):
        run_exp_cyclic_persist_01(
            seeds=[42], total_updates=1, val_cadence=1, batch_size=8,
            device=torch.device("cpu"), results_path="", run_mode="SMOKE_TEST_ONLY",
        )

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        run_exp_cyclic_persist_01(
            seeds=[42], total_updates=1, val_cadence=1, batch_size=64,
            device=torch.device("cpu"), results_path="pyproject.toml", run_mode="SMOKE_TEST_ONLY",
        )

    with pytest.raises(RuntimeError, match="official results artifact"):
        run_exp_cyclic_persist_01(
            seeds=[42], total_updates=1, val_cadence=1, batch_size=64,
            device=torch.device("cpu"),
            results_path="experiments/exp_cyclic_persist_01_results.json",
            run_mode="SMOKE_TEST_ONLY",
        )


def test_runner_materializes_only_allowed_splits(monkeypatch) -> None:
    import experiments.run_exp_cyclic_persist_01 as runner_module

    partition = Stage1PairPartition()
    allowed = set(partition.train_pairs) | set(partition.val_pairs)
    protected = set(partition.heldout_pairs)

    recorded = []
    original = Stage1PairPartition.build_trace_split

    def recording_build_trace_split(self, pairs, task=None):
        recorded.append(tuple(pairs))
        return original(self, pairs, task)

    monkeypatch.setattr(Stage1PairPartition, "build_trace_split", recording_build_trace_split)

    runner_module.run_exp_cyclic_persist_01(
        seeds=[42], total_updates=2, val_cadence=2, batch_size=64,
        device=torch.device("cpu"), results_path="", run_mode="SMOKE_TEST_ONLY",
    )

    assert recorded, "runner did not materialize any split"
    for pairs in recorded:
        assert set(pairs) <= allowed, "runner materialized pairs outside the allowed splits"
        assert not (set(pairs) & protected), "runner materialized protected TEST pairs"


def test_runner_source_never_references_the_protected_split() -> None:
    import experiments.run_exp_cyclic_persist_01 as runner_module

    source = inspect.getsource(runner_module)
    for forbidden in ("heldout_pairs", "get_stage1_splits", "TEST-HELDOUT", "heldout_ds"):
        assert forbidden not in source, f"runner references forbidden protected-split token {forbidden!r}"


def test_protected_domain_assertion_is_live() -> None:
    partition = Stage1PairPartition()
    train_ds = partition.build_trace_split(partition.train_pairs)
    val_ds = partition.build_trace_split(partition.val_pairs)
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
# Imported gate machinery (specification sections 6 and 9)
# ---------------------------------------------------------------------------

def test_checkpoint_selection_policy_min_loss_then_acc_then_earliest() -> None:
    checkpoints = [
        {"update": 50, "val_loss_total": 1.0, "val_acc": 0.9},
        {"update": 100, "val_loss_total": 0.5, "val_acc": 0.8},
        {"update": 150, "val_loss_total": 0.5, "val_acc": 0.9},
        {"update": 200, "val_loss_total": 0.5, "val_acc": 0.9},
    ]
    assert select_best_checkpoint(checkpoints)["update"] == 150

    with pytest.raises(ValueError, match="negative val_loss_total"):
        select_best_checkpoint([{"update": 50, "val_loss_total": -1.0, "val_acc": 0.5}])
    with pytest.raises(ValueError, match="out-of-bounds val_acc"):
        select_best_checkpoint([{"update": 50, "val_loss_total": 1.0, "val_acc": 1.5}])


def test_decision_matrix_covers_all_twelve_cells() -> None:
    """The imported matrix must reproduce this specification's 3x4 category table."""
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
