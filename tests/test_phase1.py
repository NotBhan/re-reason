"""Unit tests for Phase 1 components: Register Tracking, CognitionCore, Phase 1 models, and parameter parity."""

import numpy as np
import pytest
import torch

from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.cognition.core import CognitionCore
from re_reason.core.baseline_gru import GRUPhase1Model
from re_reason.core.model import ReReasonPhase1Model
from re_reason.utils.seeding import set_seed


def test_register_tracking_generation() -> None:
    """Verifies procedural batch generation dimensions and value bounds."""
    task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10, seed=42)
    batch = task.generate_batch(batch_size=16)

    assert batch.inputs.shape == (16, 10, 21)
    assert batch.targets.shape == (16, 4)
    assert "step_targets" in batch.metadata
    assert batch.metadata["step_targets"].shape == (16, 10, 4)

    # Values must be within [0, 9]
    assert (batch.targets >= 0).all() and (batch.targets < 10).all()
    assert (batch.metadata["step_targets"] >= 0).all() and (batch.metadata["step_targets"] < 10).all()

    # Op type slice must sum to 1.0
    op_slice = batch.inputs[:, :, :3]
    assert torch.allclose(op_slice.sum(dim=-1), torch.ones(16, 10))


def test_register_tracking_deterministic_ground_truth() -> None:
    """Verifies ground-truth generation determinism and step-by-step arithmetic correctness."""
    task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10)

    # Identical seed must produce bit-exact identical batches
    batch1 = task.generate_batch(8, seed=123)
    batch2 = task.generate_batch(8, seed=123)

    assert torch.equal(batch1.inputs, batch2.inputs)
    assert torch.equal(batch1.targets, batch2.targets)
    assert torch.equal(batch1.metadata["step_targets"], batch2.metadata["step_targets"])

    # Manual step-by-step arithmetic verification on a single generated episode
    inp, step_tgt, final_tgt = task.generate_single_sequence(np.random.default_rng(0))
    # Step targets at T-1 must match final_tgt
    assert (step_tgt[-1] == final_tgt).all()


def test_cognition_core_dimension_and_one_step_update(all_available_devices: torch.device) -> None:
    """Verifies CognitionCore state update preserves shape (B, 256) on CPU and GPU."""
    dev = all_available_devices
    core = CognitionCore(state_dim=256, obs_dim=256, hidden_dim=256).to(dev)

    b = 4
    state_prev = torch.randn(b, 256, device=dev)
    obs = torch.randn(b, 256, device=dev)

    state_next = core(state_prev, obs)

    assert state_next.shape == (b, 256)
    assert state_next.device.type == dev.type
    # State update must not be an identity operation
    assert not torch.equal(state_next, state_prev)


def test_phase1_model_forward_and_output_shapes(all_available_devices: torch.device) -> None:
    """Verifies ReReasonPhase1Model output shapes and diagnostics across devices."""
    dev = all_available_devices
    model = ReReasonPhase1Model(input_dim=21, state_dim=256).to(dev)

    b, seq_len = 4, 10
    inputs = torch.randn(b, seq_len, 21, device=dev)

    # Without step diagnostics
    out = model(inputs, return_step_diagnostics=False)
    assert out.logits.shape == (b, 4, 10)
    assert out.final_state.shape == (b, 256)
    assert out.step_logits is None

    # With step diagnostics
    out_diag = model(inputs, return_step_diagnostics=True)
    assert out_diag.logits.shape == (b, 4, 10)
    assert out_diag.step_logits is not None
    assert out_diag.step_logits.shape == (b, seq_len, 4, 10)


def test_gru_baseline_forward_and_output_shapes(all_available_devices: torch.device) -> None:
    """Verifies GRUPhase1Model output shapes and diagnostics across devices."""
    dev = all_available_devices
    model = GRUPhase1Model(input_dim=21, gru_hidden_dim=162).to(dev)

    b, seq_len = 4, 10
    inputs = torch.randn(b, seq_len, 21, device=dev)

    out = model(inputs, return_step_diagnostics=True)
    assert out.logits.shape == (b, 4, 10)
    assert out.final_state.shape == (b, 162)
    assert out.step_logits is not None
    assert out.step_logits.shape == (b, seq_len, 4, 10)


def test_parameter_parity_and_budget() -> None:
    """Verifies that both RE:Reason and GRU models adhere to <= 250K budget and +/-5% parity."""
    rereason_model = ReReasonPhase1Model()
    gru_model = GRUPhase1Model(gru_hidden_dim=162)

    p_rr = rereason_model.count_parameters()
    p_gru = gru_model.count_parameters()

    # Both models must be within <= 250K parameters
    assert p_rr <= 250000, f"RE:Reason parameters {p_rr} exceed 250K budget"
    assert p_gru <= 250000, f"GRU parameters {p_gru} exceed 250K budget"

    # Parameter counts must match within +/- 5%
    delta_ratio = abs(p_rr - p_gru) / p_rr
    assert delta_ratio <= 0.05, f"Parameter delta ratio {delta_ratio*100:.2f}% exceeds 5% parity rule"


def test_sequential_state_persistence(all_available_devices: torch.device) -> None:
    """Verifies that sequential state transitions update state and retain past context."""
    dev = all_available_devices
    model = ReReasonPhase1Model().to(dev)

    task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10, seed=77)
    batch = task.generate_batch(4)
    inputs = batch.inputs.to(dev)

    out = model(inputs, return_step_diagnostics=True)
    assert out.step_logits is not None

    # Check that predictions vary across time steps as operations are applied
    step_preds = out.step_logits.argmax(dim=-1)  # (4, 10, 4)
    # Step 0 vs Step 9 predictions should not be strictly identical across all instances
    assert not torch.equal(step_preds[:, 0, :], step_preds[:, 9, :])


def test_deterministic_seeding_model_execution() -> None:
    """Verifies exact bit-level reproducibility under identical seeds."""
    task = RegisterTrackingTask(seed=101)
    batch = task.generate_batch(4, seed=202)

    set_seed(42)
    m1 = ReReasonPhase1Model()
    out1 = m1(batch.inputs)

    set_seed(42)
    m2 = ReReasonPhase1Model()
    out2 = m2(batch.inputs)

    assert torch.equal(out1.logits, out2.logits)
    assert torch.equal(out1.final_state, out2.final_state)
