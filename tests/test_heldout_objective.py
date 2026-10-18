"""Unit tests for the registered held-out objective, metrics, and Tier 1 gate (pure math).

Covers the 2026-10-18 Phase 1.6 amendment definitions: per-step unique-written-register
masks, the unified loss ``L_term + 1.0 * L_step`` normalization, macro-recall balanced
accuracy, terminal sequence exact match, and Tier 1 gate classification.

No benchmark generator is invoked and no dataset is materialized by this module;
instruction tokens are hand-constructed via ``Phase16RegisterTask.encode_instruction``
or plain tensors, and ``collect_eligible_add_transitions`` is exercised on a
hand-constructed three-instruction trace through the deterministic replay simulator.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from re_reason.benchmarks.register_machine_benchmark import Phase16RegisterTask
from re_reason.training.heldout_objective import (
    classify_tier1_gate,
    collect_eligible_add_transitions,
    heldout_add_balanced_accuracy,
    terminal_sequence_exact_match,
    unified_heldout_loss,
    written_register_masks,
)


def _token_batch(task: Phase16RegisterTask, specs) -> torch.Tensor:
    return torch.from_numpy(np.stack([task.encode_instruction(*spec) for spec in specs], axis=0)).unsqueeze(0)


def test_written_register_masks_contract() -> None:
    task = Phase16RegisterTask()
    tokens = _token_batch(
        task,
        [
            (task.OP_SET, 0, -1, 5),
            (task.OP_SWAP, 1, 2, -1),
            (task.OP_READ, 3, -1, -1),
            (task.OP_ADD, 2, 1, -1),
            (task.OP_SWAP, 1, 1, -1),
        ],
    )
    masks = written_register_masks(tokens)
    assert masks.shape == (1, 5, 4)
    assert masks[0, 0].tolist() == [1, 0, 0, 0]
    assert masks[0, 1].tolist() == [0, 1, 1, 0]
    assert masks[0, 2].tolist() == [0, 0, 0, 0]
    assert masks[0, 3].tolist() == [0, 0, 1, 0]
    assert masks[0, 4].tolist() == [0, 1, 0, 0]


def test_unified_loss_matches_independent_recomputation() -> None:
    task = Phase16RegisterTask()
    specs = [
        (task.OP_SET, 0, -1, 3),
        (task.OP_ADD, 1, 0, -1),
        (task.OP_READ, 1, -1, -1),
    ]
    tokens = _token_batch(task, specs).repeat(2, 1, 1)

    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 10)
    targets = torch.randint(0, 10, (2, 3, 4))

    loss, components = unified_heldout_loss(logits, targets, tokens)

    masks = written_register_masks(tokens)
    per_register_ce = F.cross_entropy(
        logits.reshape(-1, 10), targets.reshape(-1), reduction="none"
    ).reshape(2, 3, 4)
    n_writes = masks.sum(dim=(1, 2))
    expected_step = (per_register_ce * masks).sum(dim=(1, 2)) / n_writes
    expected_terminal = per_register_ce[:, -1, :].mean(dim=1)
    expected_total = (expected_terminal + expected_step).mean()

    assert torch.allclose(loss, expected_total, atol=1e-6)
    assert components["n_writes_mean"] == pytest.approx(2.0)
    assert components["l_step_mean"] == pytest.approx(float(expected_step.mean()))


def test_unified_loss_read_only_sequence_has_zero_step_term() -> None:
    task = Phase16RegisterTask()
    tokens = _token_batch(task, [(task.OP_READ, 0, -1, -1)] * 4)
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 4, 10)
    targets = torch.randint(0, 10, (1, 4, 4))

    loss, components = unified_heldout_loss(logits, targets, tokens)
    per_register_ce = F.cross_entropy(
        logits.reshape(-1, 10), targets.reshape(-1), reduction="none"
    ).reshape(1, 4, 4)
    expected_terminal = per_register_ce[:, -1, :].mean()

    assert components["l_step_mean"] == pytest.approx(0.0)
    assert components["n_writes_mean"] == pytest.approx(0.0)
    assert torch.allclose(loss, expected_terminal, atol=1e-6)


def test_unified_loss_normalizes_step_term_by_n_writes() -> None:
    task = Phase16RegisterTask()
    one_write = _token_batch(task, [(task.OP_SET, 0, -1, 1)])
    two_writes = _token_batch(task, [(task.OP_SWAP, 0, 1, -1)])
    tokens = torch.cat([one_write, two_writes], dim=0)
    logits = torch.zeros(2, 1, 4, 10)
    targets = torch.zeros(2, 1, 4, dtype=torch.long)

    _loss, components = unified_heldout_loss(logits, targets, tokens)

    assert components["n_writes_mean"] == pytest.approx(1.5)
    assert components["l_step_mean"] == pytest.approx(float(np.log(10.0)), rel=1e-5)


def test_balanced_accuracy_macro_recall_hand_computed() -> None:
    result = heldout_add_balanced_accuracy([1, 1, 0, 2, 2], [1, 0, 0, 2, 1])
    assert result["macro_recall"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert result["n_transitions"] == 5
    assert result["per_class_counts"][0] == 2
    assert result["per_class_counts"][1] == 2
    assert result["per_class_counts"][2] == 1
    assert result["per_class_recalls"][0] == pytest.approx(0.5)
    assert result["per_class_recalls"][2] == pytest.approx(1.0)
    assert set(result["per_class_recalls"]) == {0, 1, 2}


def test_balanced_accuracy_rejects_empty_population() -> None:
    with pytest.raises(ValueError, match="no eligible held-out ADD transitions"):
        heldout_add_balanced_accuracy([], [])


def test_terminal_exact_match_definition() -> None:
    predicted = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5], [0, 0, 0, 0]])
    target = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 1]])
    assert terminal_sequence_exact_match(predicted, target) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="empty"):
        terminal_sequence_exact_match(
            torch.zeros(0, 4, dtype=torch.long), torch.zeros(0, 4, dtype=torch.long)
        )


def test_tier1_gate_pass_reject_halt_paths() -> None:
    def metrics(pairs):
        return [{"seed": 42 + index, "acc": acc, "em": em} for index, (acc, em) in enumerate(pairs)]

    assert classify_tier1_gate(metrics([(0.9, 0.8)] * 4 + [(0.5, 0.4)]))["verdict"] == "tier1_pass"
    assert classify_tier1_gate(metrics([(0.1, 0.1)] * 4 + [(0.9, 0.8)]))["verdict"] == "tier1_reject"
    assert classify_tier1_gate(metrics([(0.9, 0.8)] * 3 + [(0.6, 0.5)] * 2))["verdict"] == "tier1_halt"
    boundary = classify_tier1_gate(metrics([(0.9, 0.8)] * 4 + [(0.25, 0.5)]))
    assert boundary["verdict"] == "tier1_pass"
    assert boundary["seed_reject"][-1] is False
    assert classify_tier1_gate(metrics([(0.9, 0.8)] * 4 + [(0.5, 0.150)]))["verdict"] == "tier1_halt"
    assert classify_tier1_gate(metrics([(0.249, 0.9)] * 4 + [(0.9, 0.8)]))["verdict"] == "tier1_reject"
    with pytest.raises(ValueError, match="exactly 5 seeds"):
        classify_tier1_gate(metrics([(0.9, 0.8)] * 4))


def test_eligible_transitions_use_runtime_values() -> None:
    task = Phase16RegisterTask()
    protected_like = _token_batch(
        task,
        [
            (task.OP_SET, 0, -1, 5),
            (task.OP_SET, 1, -1, 7),
            (task.OP_ADD, 1, 0, -1),
        ],
    )
    assert collect_eligible_add_transitions(task, protected_like, (0, 5)) == [[(2, 1)]]
    assert collect_eligible_add_transitions(task, protected_like, (3,)) == [[]]

    proxy_like = _token_batch(
        task,
        [
            (task.OP_SET, 0, -1, 3),
            (task.OP_SET, 1, -1, 4),
            (task.OP_ADD, 1, 0, -1),
        ],
    )
    assert collect_eligible_add_transitions(task, proxy_like, (3,)) == [[(2, 1)]]
    assert collect_eligible_add_transitions(task, proxy_like, (0, 5)) == [[]]
