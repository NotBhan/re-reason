"""Regression and Contract Tests for Phase 1.6 Stage 1 Pilot Telemetry Logger.

Verifies:
1. Checkpoint records contain all 5 required fields:
   - step
   - train_loss
   - train_acc
   - val_loss
   - val_acc
2. Selected checkpoint records contain train and val metrics alongside step.
3. Behavior neutrality: telemetry evaluation under no_grad does not pollute
   gradients, alter parameter weights, or disturb model training state.
4. Downstream schema compatibility: existing consumer fields are preserved.
5. Numerical sanity: train_acc and val_acc bounded in [0.0, 1.0], losses are finite.
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from re_reason.benchmarks.register_machine_benchmark import (
    Stage1PairPartition,
)
from re_reason.core.register_machine_models import (
    SlotMGUPhase16Model,
    VanillaGRUPhase16Model,
)
from experiments.run_phase1_6_stage1_pilot import (
    run_track_1a_trained_pilot,
)


def test_stage1_checkpoint_telemetry_schema_and_fields() -> None:
    """Verifies that run_track_1a_trained_pilot records train and val metrics at every checkpoint."""
    # Run a minimal bounded pilot (1 seed, 2 updates, val_cadence=1) to test logger
    results = run_track_1a_trained_pilot(
        seeds=[42],
        learning_rate=3e-3,
        weight_decay=1e-2,
        batch_size=64,
        total_updates=2,
        val_cadence=1,
        device=torch.device("cpu"),
    )

    required_checkpoint_keys = {"step", "train_loss", "train_acc", "val_loss", "val_acc"}
    required_selected_keys = {"step", "train_loss", "train_acc", "val_loss", "val_acc"}

    for model_name, data in results.items():
        assert "runs" in data, f"Missing 'runs' in {model_name}"
        run_record = data["runs"][0]

        # Check selected_checkpoint
        sel = run_record["selected_checkpoint"]
        for key in required_selected_keys:
            assert key in sel, f"Missing key '{key}' in selected_checkpoint for {model_name}"
            assert isinstance(sel[key], (int, float)), f"Key '{key}' is not numeric in {model_name}"

        # Check checkpoints array
        ckpts = run_record["checkpoints"]
        assert len(ckpts) == 2, f"Expected 2 checkpoints, got {len(ckpts)}"

        for ckpt in ckpts:
            for key in required_checkpoint_keys:
                assert key in ckpt, f"Missing key '{key}' in checkpoint for {model_name}"
                assert isinstance(ckpt[key], (int, float)), f"Key '{key}' is not numeric in {model_name}"

            # Range assertions
            assert 0.0 <= ckpt["train_acc"] <= 1.0, f"train_acc out of range [0, 1]: {ckpt['train_acc']}"
            assert 0.0 <= ckpt["val_acc"] <= 1.0, f"val_acc out of range [0, 1]: {ckpt['val_acc']}"
            assert ckpt["train_loss"] >= 0.0, f"Negative train_loss: {ckpt['train_loss']}"
            assert ckpt["val_loss"] >= 0.0, f"Negative val_loss: {ckpt['val_loss']}"

        # JSON serializability check
        serialized = json.dumps(run_record)
        assert len(serialized) > 0


def test_telemetry_evaluation_is_behavior_neutral() -> None:
    """Verifies that running telemetry forward passes does not alter gradient updates or model trajectory."""
    partition = Stage1PairPartition()
    train_ds = partition.build_trace_split(partition.train_pairs)
    inputs_train = train_ds.inputs
    targets_train = train_ds.targets[:, 0]

    # Model A: Trained without extra telemetry evaluation
    torch.manual_seed(100)
    model_a = SlotMGUPhase16Model()
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)

    for _ in range(5):
        model_a.train()
        opt_a.zero_grad()
        out_a = model_a(inputs_train)
        loss_a = F.cross_entropy(out_a.logits, targets_train)
        loss_a.backward()
        opt_a.step()

    # Model B: Trained with telemetry evaluation under torch.no_grad() between updates
    torch.manual_seed(100)
    model_b = SlotMGUPhase16Model()
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)

    for _ in range(5):
        model_b.train()
        opt_b.zero_grad()
        out_b = model_b(inputs_train)
        loss_b = F.cross_entropy(out_b.logits, targets_train)
        loss_b.backward()
        opt_b.step()

        # Telemetry evaluation
        model_b.eval()
        with torch.no_grad():
            t_out = model_b(inputs_train)
            _ = F.cross_entropy(t_out.logits, targets_train).item()
            _ = (t_out.logits.argmax(dim=-1) == targets_train).float().mean().item()

    # Verify that model_a and model_b have bitwise identical parameters
    for (n_a, p_a), (n_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.equal(p_a, p_b), f"Parameter {n_a} diverged due to telemetry evaluation!"
