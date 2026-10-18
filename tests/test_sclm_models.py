"""Unit and contract tests for the four-cell SCLM benchmark candidates (pure tensor tests).

Covers the registered 2026-10-18 Phase 1.6 interfaces: exact parameter counts
(A 8,255; B 10,249; C 227,062; D 229,056), initial-state contracts, operation
dispatch with bit-exact untouched-slot preservation, the fixed convolution math,
independent recomputation of every wiring path, bounded monolithic dynamics,
gradient flow without optimizer execution, and deterministic reruns.

No benchmark generator is invoked and no dataset is materialized by this module;
instruction tokens are hand-constructed via ``Phase16RegisterTask.encode_instruction``
or plain tensors.
"""

import numpy as np
import pytest
import torch

from re_reason.benchmarks.register_machine_benchmark import Phase16RegisterTask
from re_reason.core.sclm_models import (
    CANDIDATE_MODELS,
    EXACT_PARAMETER_COUNTS,
    CyclicMGUControlModel,
    MonolithicMLPControlModel,
    SCLMModel,
    SlotMLPModel,
    build_candidate,
    circular_convolution,
)
from re_reason.training.heldout_objective import unified_heldout_loss


def _tokens(task: Phase16RegisterTask, specs) -> torch.Tensor:
    """Hand-constructs a batched token sequence from explicit instruction specs."""
    return torch.from_numpy(np.stack([task.encode_instruction(*spec) for spec in specs], axis=0)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------

def test_exact_parameter_counts_four_cells() -> None:
    assert EXACT_PARAMETER_COUNTS == {"A": 8255, "B": 10249, "C": 227062, "D": 229056}
    for model_id, expected in EXACT_PARAMETER_COUNTS.items():
        assert build_candidate(model_id).count_parameters() == expected
    assert EXACT_PARAMETER_COUNTS["B"] - EXACT_PARAMETER_COUNTS["A"] == 1994
    assert EXACT_PARAMETER_COUNTS["D"] - EXACT_PARAMETER_COUNTS["C"] == 1994


def test_submodule_parameter_counts_match_registration() -> None:
    model_a = build_candidate("A")
    assert sum(p.numel() for p in model_a.decoder.parameters()) == 4106
    assert sum(p.numel() for p in model_a.writer.parameters()) == 4149
    assert model_a.count_parameters() == 4106 + 4149

    model_b = build_candidate("B")
    assert sum(p.numel() for p in model_b.alu.parameters()) == 1994
    assert model_b.count_parameters() == 4106 + 4149 + 1994

    model_c = build_candidate("C")
    assert sum(p.numel() for p in model_c.head.parameters()) == 8520
    assert sum(p.numel() for p in model_c.encoder.parameters()) == 19264
    assert sum(p.numel() for p in model_c.core.parameters()) == 199278
    assert model_c.count_parameters() == 8520 + 19264 + 199278

    model_d = build_candidate("D")
    assert sum(p.numel() for p in model_d.alu.parameters()) == 1994
    assert model_d.count_parameters() == model_c.count_parameters() + 1994


def test_factory_returns_registered_classes_and_rejects_unknown() -> None:
    assert set(CANDIDATE_MODELS) == {"A", "B", "C", "D"}
    assert isinstance(build_candidate("A"), SCLMModel)
    assert isinstance(build_candidate("B"), SlotMLPModel)
    assert isinstance(build_candidate("C"), CyclicMGUControlModel)
    assert isinstance(build_candidate("D"), MonolithicMLPControlModel)
    with pytest.raises(ValueError, match="unknown candidate"):
        build_candidate("E")


# ---------------------------------------------------------------------------
# Initial state and operation dispatch
# ---------------------------------------------------------------------------

def test_slot_initial_state_is_writer_applied_one_hot_zero() -> None:
    model = SCLMModel()
    state = model.initial_state(3, torch.device("cpu"))
    one_hot_zero = torch.zeros(10)
    one_hot_zero[0] = 1.0
    expected_slot = model.writer(one_hot_zero)
    assert state.shape == (3, 4, 53)
    for register in range(4):
        assert torch.equal(state[:, register], expected_slot.unsqueeze(0).repeat(3, 1))
    assert bool((state >= -1.0).all() and (state <= 1.0).all())


def test_slot_dispatch_preserves_untouched_slots_bitwise() -> None:
    task = Phase16RegisterTask()
    model = SCLMModel()
    inputs = _tokens(
        task,
        [
            (task.OP_SET, 0, -1, 7),
            (task.OP_SET, 1, -1, 3),
            (task.OP_SWAP, 0, 1, -1),
            (task.OP_READ, 0, -1, -1),
            (task.OP_ADD, 2, 1, -1),
        ],
    )

    state_0 = model.initial_state(1, torch.device("cpu"))
    state_1, _ = model.forward_step(inputs[:, 0], state_0)
    for register in (1, 2, 3):
        assert torch.equal(state_1[0, register], state_0[0, register])

    state_2, _ = model.forward_step(inputs[:, 1], state_1)
    for register in (0, 2, 3):
        assert torch.equal(state_2[0, register], state_1[0, register])

    state_3, _ = model.forward_step(inputs[:, 2], state_2)
    assert torch.equal(state_3[0, 0], state_2[0, 1])
    assert torch.equal(state_3[0, 1], state_2[0, 0])
    assert torch.equal(state_3[0, 2], state_2[0, 2])
    assert torch.equal(state_3[0, 3], state_2[0, 3])

    state_4, _ = model.forward_step(inputs[:, 3], state_3)
    assert torch.equal(state_4, state_3)

    state_5, _ = model.forward_step(inputs[:, 4], state_4)
    for register in (0, 1, 3):
        assert torch.equal(state_5[0, register], state_4[0, register])


def test_fixed_convolution_is_exact_on_one_hot_operands() -> None:
    basis = torch.eye(10)
    for a in range(10):
        for b in range(10):
            result = circular_convolution(basis[a], basis[b])
            assert torch.allclose(result, basis[(a + b) % 10], atol=1e-6)
            assert torch.allclose(
                circular_convolution(basis[a], basis[b]),
                circular_convolution(basis[b], basis[a]),
                atol=1e-6,
            )
    for a in range(10):
        assert torch.allclose(circular_convolution(basis[a], basis[0]), basis[a], atol=1e-6)


# ---------------------------------------------------------------------------
# Independent recomputation of the registered wiring paths
# ---------------------------------------------------------------------------

def test_slot_fixed_add_wiring_matches_independent_recomputation() -> None:
    task = Phase16RegisterTask()
    model = SCLMModel()
    torch.manual_seed(0)
    state = torch.randn(2, 4, 53)
    add_token = _tokens(task, [(task.OP_ADD, 1, 2, -1)]).repeat(2, 1, 1)[:, 0, :]

    state_next, logits = model.forward_step(add_token, state)

    p_prev = torch.softmax(model.decoder(state), dim=-1)
    expected_dest = model.writer(circular_convolution(p_prev[:, 1], p_prev[:, 2]))
    assert torch.allclose(state_next[:, 1], expected_dest, atol=1e-6)
    for register in (0, 2, 3):
        assert torch.equal(state_next[:, register], state[:, register])
    assert torch.allclose(logits, model.decoder(state_next), atol=1e-6)


def test_slot_learned_add_wiring_matches_independent_recomputation() -> None:
    task = Phase16RegisterTask()
    model = SlotMLPModel()
    torch.manual_seed(1)
    state = torch.randn(2, 4, 53)
    add_token = _tokens(task, [(task.OP_ADD, 2, 0, -1)]).repeat(2, 1, 1)[:, 0, :]

    state_next, _logits = model.forward_step(add_token, state)

    p_prev = torch.softmax(model.decoder(state), dim=-1)
    operands = torch.cat([p_prev[:, 2], p_prev[:, 0]], dim=-1)
    p_sum = torch.softmax(model.alu(operands), dim=-1)
    expected_dest = model.writer(p_sum)
    assert torch.allclose(state_next[:, 2], expected_dest, atol=1e-6)
    for register in (0, 1, 3):
        assert torch.equal(state_next[:, register], state[:, register])


def test_monolithic_wiring_matches_independent_recomputation() -> None:
    task = Phase16RegisterTask()
    for factory in (CyclicMGUControlModel, MonolithicMLPControlModel):
        model = factory()
        torch.manual_seed(2)
        state = torch.randn(3, 212)
        add_token = _tokens(task, [(task.OP_ADD, 1, 3, -1)]).repeat(3, 1, 1)[:, 0, :]
        set_token = _tokens(task, [(task.OP_SET, 0, -1, 6)]).repeat(3, 1, 1)[:, 0, :]

        state_next, logits = model.forward_step(add_token, state)
        p_prev = torch.softmax(model.head(state), dim=-1)
        if model.alu is None:
            f_alu = circular_convolution(p_prev[:, 1], p_prev[:, 3])
        else:
            f_alu = torch.softmax(
                model.alu(torch.cat([p_prev[:, 1], p_prev[:, 3]], dim=-1)), dim=-1
            )
        f_alu = add_token[:, 3:4] * f_alu
        e_t = model.encoder(torch.cat([add_token, f_alu], dim=-1))
        expected = model.core(state, e_t)
        assert torch.allclose(state_next, expected, atol=1e-6)
        assert torch.allclose(logits, model.head(state_next), atol=1e-6)

        state_set, _ = model.forward_step(set_token, state)
        zero_feature = torch.zeros(3, 10)
        e_set = model.encoder(torch.cat([set_token, zero_feature], dim=-1))
        assert torch.allclose(state_set, model.core(state, e_set), atol=1e-6)


# ---------------------------------------------------------------------------
# Dynamics, shapes, gradients, determinism
# ---------------------------------------------------------------------------

def test_monolithic_state_bounded_over_rollout() -> None:
    task = Phase16RegisterTask()
    rng = np.random.default_rng(0)
    tokens = []
    for _ in range(30):
        op = int(rng.integers(0, 4))
        dest = int(rng.integers(0, 4))
        src = int(rng.integers(0, 4)) if op in (task.OP_SWAP, task.OP_ADD) else -1
        val = int(rng.integers(0, 10)) if op == task.OP_SET else -1
        tokens.append(task.encode_instruction(op, dest, src, val))
    inputs = torch.from_numpy(np.stack(tokens)).unsqueeze(0)

    for model_id in ("C", "D"):
        model = build_candidate(model_id)
        output = model(inputs)
        assert output.final_state.shape == (1, 212)
        assert bool((output.final_state >= -1.0).all() and (output.final_state <= 1.0).all())
        assert torch.isfinite(output.step_logits).all()


def test_output_shapes_all_candidates() -> None:
    inputs = torch.zeros(2, 6, 22)
    for model_id, expected_state_shape in (("A", (2, 4, 53)), ("B", (2, 4, 53)), ("C", (2, 212)), ("D", (2, 212))):
        model = build_candidate(model_id)
        output = model(inputs, initial_state=model.initial_state(2, torch.device("cpu")))
        assert output.step_logits.shape == (2, 6, 4, 10)
        assert output.final_state.shape == expected_state_shape
        assert torch.isfinite(output.step_logits).all()


def test_gradient_flow_all_trainable_parameters_no_optimizer() -> None:
    task = Phase16RegisterTask()
    inputs = _tokens(
        task,
        [
            (task.OP_SET, 0, -1, 5),
            (task.OP_SET, 1, -1, 7),
            (task.OP_ADD, 0, 1, -1),
            (task.OP_SWAP, 0, 1, -1),
            (task.OP_READ, 0, -1, -1),
        ],
    )
    targets = torch.zeros(1, 5, 4, dtype=torch.long)

    for model_id in ("A", "B", "C", "D"):
        model = build_candidate(model_id)
        output = model(inputs)
        loss, _ = unified_heldout_loss(output.step_logits, targets, inputs)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{model_id}:{name} received no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{model_id}:{name} has non-finite gradient"


def test_deterministic_forward_reproducibility() -> None:
    task = Phase16RegisterTask()
    inputs = _tokens(
        task,
        [
            (task.OP_SET, 2, -1, 4),
            (task.OP_ADD, 0, 2, -1),
            (task.OP_SWAP, 0, 1, -1),
        ],
    )
    for model_id in ("A", "B", "C", "D"):
        torch.manual_seed(1234)
        first = build_candidate(model_id)
        torch.manual_seed(1234)
        second = build_candidate(model_id)
        output_first = first(inputs)
        output_second = second(inputs)
        assert torch.equal(output_first.step_logits, output_second.step_logits)
        assert torch.equal(output_first.final_state, output_second.final_state)
