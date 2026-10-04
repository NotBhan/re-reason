"""Unit tests for the synthetic algorithmic benchmark tasks."""

import numpy as np
import pytest

from re_reason.benchmarks.algorithmic_tasks import (
    ConditionalOverwriteTask,
    MultiRegisterCopyTask,
    PermutationCycleTask,
    RunningAccumulationTask,
)


def test_multi_register_copy_task_deterministic() -> None:
    task = MultiRegisterCopyTask(num_registers=4, sequence_length=10, num_values=10, seed=42)
    batch = task.generate_batch(100, seed=123)
    assert batch.inputs.shape == (100, 10, 21)
    assert batch.targets.shape == (100, 4)
    assert (batch.targets >= 0).all() and (batch.targets < 10).all()


def test_permutation_cycle_task_deterministic() -> None:
    task = PermutationCycleTask(num_registers=4, sequence_length=10, num_values=10, seed=42)
    batch = task.generate_batch(100, seed=123)
    assert batch.inputs.shape == (100, 10, 21)
    assert batch.targets.shape == (100, 4)


def test_running_accumulation_task_deterministic() -> None:
    task = RunningAccumulationTask(num_registers=4, sequence_length=10, num_values=10, seed=42)
    batch = task.generate_batch(100, seed=123)
    assert batch.inputs.shape == (100, 10, 21)
    assert batch.targets.shape == (100, 4)


def test_conditional_overwrite_task_deterministic() -> None:
    task = ConditionalOverwriteTask(num_registers=4, sequence_length=10, num_values=10, seed=42)
    batch = task.generate_batch(100, seed=123)
    assert batch.inputs.shape == (100, 10, 21)
    assert batch.targets.shape == (100, 4)
