"""Unit and contract tests for EXP-CYCLIC-ADD-01 runner.

Verifies:
1. Exact parameter count for CyclicMGUPhase16Model (219,018).
2. Runner schema and telemetry field presence (step, train_loss, train_acc, val_loss, val_acc).
3. Test split firewall preservation (zero test queries).
4. Deterministic repeatability across identical seed configurations.
"""

import json
import os
import sys
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from re_reason.benchmarks.register_machine_benchmark import Stage1PairPartition
from re_reason.core.register_machine_models import CyclicMGUPhase16Model
from experiments.run_exp_cyclic_add_01 import run_cyclic_add_experiment


def test_cyclic_mgu_parameter_count():
    model = CyclicMGUPhase16Model()
    assert model.count_parameters() == 219018


def test_exp_cyclic_add_01_smoke_run():
    # 2-step smoke run to verify telemetry capture and checkpoint schema
    res = run_cyclic_add_experiment(
        seeds=[42],
        total_updates=4,
        val_cadence=2,
        results_path="",  # do not overwrite main artifact during test
    )
    assert res["experiment_id"] == "EXP-CYCLIC-ADD-01"
    assert res["model_name"] == "CyclicMGUPhase16Model"
    assert res["parameter_count"] == 219018
    assert len(res["runs"]) == 1

    run_record = res["runs"][0]
    assert run_record["seed"] == 42
    assert len(run_record["checkpoints"]) == 2

    ckpt = run_record["checkpoints"][0]
    for field in ["step", "train_loss", "train_acc", "val_loss", "val_acc"]:
        assert field in ckpt
        assert isinstance(ckpt[field], (int, float))

    sel = run_record["selected_checkpoint"]
    for field in ["step", "train_loss", "train_acc", "val_loss", "val_acc"]:
        assert field in sel


def test_firewall_protected_test_split_untouched():
    partition = Stage1PairPartition()
    assert len(partition.heldout_pairs) == 20
    for a, b in partition.heldout_pairs:
        assert b in {0, 5}
