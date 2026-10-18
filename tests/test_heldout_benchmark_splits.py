"""Generator and split-integrity tests for the held-out campaign data contract.

EXECUTION STATUS: these tests were authored during the 2026-10-18 implementation
stage but were NOT executed in that session, because the implementation work order
prohibited invoking the benchmark generator (no data generation outside an
authorized stage). They must be executed and pass before any dataset generation or
training under the registered protocol; see the 2026-10-18 RESEARCH_LOG entry.

Covers the registered additive generator/split contract: default {0, 5} backward
compatibility, the held-out campaign {0, 3, 5} runtime ADD-source mask, unrestricted
SET literals, bounded-attempt failure behavior, PROXY-DEV conditioning invariants
(exactly one designated ADD with runtime source 3, zero {0, 5}, seen-only other ADDs,
exact per-class balance, randomized trace details), and cross-split metadata-only
disjointness.
"""

import numpy as np
import pytest
import torch

from re_reason.benchmarks.dataset_split import (
    assert_manifests_disjoint,
    build_hash_manifest,
)
from re_reason.benchmarks.register_machine_benchmark import Phase16RegisterTask, Phase16SplitManager


def test_masked_generator_default_is_bitwise_backward_compatible() -> None:
    task = Phase16RegisterTask()
    default = task.generate_regime_b_sequence(np.random.default_rng(2026))
    explicit = task.generate_regime_b_sequence(
        np.random.default_rng(2026), excluded_source_values=(0, 5)
    )
    assert np.array_equal(default[0], explicit[0])
    assert np.array_equal(default[1], explicit[1])
    assert np.array_equal(default[2], explicit[2])


def test_masked_generator_excludes_runtime_035() -> None:
    task = Phase16RegisterTask()
    rng = np.random.default_rng(7)
    for _ in range(32):
        inputs, step_targets, final_target = task.generate_regime_b_sequence(
            rng, excluded_source_values=(0, 3, 5)
        )
        _, history = task.simulate_trace(inputs)
        for entry in history:
            if entry["op_type"] == task.OP_ADD:
                assert entry["runtime_src_val_pre"] not in (0, 3, 5)
        assert np.array_equal(step_targets[-1], final_target)


def test_masked_generator_set_literals_remain_unrestricted() -> None:
    task = Phase16RegisterTask()
    rng = np.random.default_rng(99)
    literals = set()
    for _ in range(40):
        inputs, _, _ = task.generate_regime_b_sequence(rng, excluded_source_values=(0, 3, 5))
        _, history = task.simulate_trace(inputs)
        for entry in history:
            if entry["op_type"] == task.OP_SET:
                literals.add(int(entry["immediate_val"]))
    assert literals & {0, 3, 5}


def test_masked_generator_attempt_exhaustion_is_fatal(monkeypatch) -> None:
    task = Phase16RegisterTask()
    real_simulate = task.simulate_trace

    def patched_simulate(inputs_arr):
        targets, history = real_simulate(inputs_arr)
        injected = dict(history[-1])
        injected.update({"op_type": task.OP_ADD, "runtime_src_val_pre": 0})
        return targets, list(history) + [injected]

    monkeypatch.setattr(task, "simulate_trace", patched_simulate)
    with pytest.raises(RuntimeError, match="Failed to generate valid Regime B masked sequence"):
        task.generate_regime_b_sequence(
            np.random.default_rng(0), max_attempts=2, excluded_source_values=(0, 3, 5)
        )


def test_masked_generator_rejects_invalid_exclusions() -> None:
    task = Phase16RegisterTask()
    with pytest.raises(ValueError, match="non-empty"):
        task.generate_regime_b_sequence(np.random.default_rng(0), excluded_source_values=())
    with pytest.raises(ValueError, match="must be within"):
        task.generate_regime_b_sequence(np.random.default_rng(0), excluded_source_values=(0, 10))
    with pytest.raises(ValueError, match="at least one eligible"):
        task.generate_regime_b_sequence(
            np.random.default_rng(0), excluded_source_values=tuple(range(10))
        )
    manager = Phase16SplitManager(task)
    with pytest.raises(ValueError, match="only supported for 'regime_b_masked'"):
        manager.generate_disjoint_split(
            2, 0, generation_mode="regime_a", excluded_source_values=(0, 3, 5)
        )


def test_proxy_generator_invariants_and_randomization() -> None:
    task = Phase16RegisterTask()
    rng = np.random.default_rng(4242)
    designated_steps = set()
    address_pairs = set()
    for target_class in range(10):
        for _ in range(2):
            inputs, step_targets, final_target = task.generate_proxy_dev_sequence(rng, target_class)
            _, history = task.simulate_trace(inputs)
            add_steps = [entry for entry in history if entry["op_type"] == task.OP_ADD]
            designated = [
                entry for entry in add_steps if entry["runtime_src_val_pre"] == 3
            ]
            assert len(designated) == 1
            assert all(entry["runtime_src_val_pre"] not in (0, 5) for entry in add_steps)
            assert all(
                entry["runtime_src_val_pre"] in (1, 2, 4, 6, 7, 8, 9)
                for entry in add_steps
                if entry["runtime_src_val_pre"] != 3
            )
            assert (designated[0]["runtime_dest_val_pre"] + 3) % 10 == target_class
            assert np.array_equal(step_targets[-1], final_target)
            designated_steps.add(int(designated[0]["step"]))
            address_pairs.add((int(designated[0]["reg_dest"]), int(designated[0]["reg_src"])))
    assert len(designated_steps) >= 2
    assert len(address_pairs) >= 2


def test_proxy_generator_attempt_exhaustion_is_fatal(monkeypatch) -> None:
    task = Phase16RegisterTask()
    real_simulate = task.simulate_trace

    def patched_simulate(inputs_arr):
        targets, history = real_simulate(inputs_arr)
        rewritten = []
        for entry in history:
            adjusted = dict(entry)
            if adjusted["op_type"] == task.OP_ADD and adjusted["runtime_src_val_pre"] == 3:
                adjusted["runtime_src_val_pre"] = 1
            rewritten.append(adjusted)
        return targets, rewritten

    monkeypatch.setattr(task, "simulate_trace", patched_simulate)
    with pytest.raises(RuntimeError, match="Failed to generate PROXY-DEV"):
        task.generate_proxy_dev_sequence(np.random.default_rng(0), target_class=4, max_attempts=2)


def test_proxy_split_exact_balance_and_uniqueness() -> None:
    task = Phase16RegisterTask()
    manager = Phase16SplitManager(task)
    split = manager.create_proxy_dev_split(per_class=2)
    assert len(split) == 20
    assert len(set(split.hashes)) == 20

    class_counts = {value: 0 for value in range(10)}
    for index in range(len(split)):
        _, history = task.simulate_trace(split.inputs[index].numpy())
        designated = [
            entry
            for entry in history
            if entry["op_type"] == task.OP_ADD and entry["runtime_src_val_pre"] == 3
        ]
        assert len(designated) == 1
        class_counts[(designated[0]["runtime_dest_val_pre"] + 3) % 10] += 1
    assert class_counts == {value: 2 for value in range(10)}


def test_campaign_split_builders_small_and_disjoint() -> None:
    task = Phase16RegisterTask()
    manager = Phase16SplitManager(task)
    train_set, val_set = manager.create_masked_campaign_splits(train_size=8, val_size=4)
    proxy_set = manager.create_proxy_dev_split(per_class=1)
    assert len(train_set) == 8
    assert len(val_set) == 4
    assert len(proxy_set) == 10

    for index in range(len(train_set)):
        _, history = task.simulate_trace(train_set.inputs[index].numpy())
        for entry in history:
            if entry["op_type"] == task.OP_ADD:
                assert entry["runtime_src_val_pre"] not in (0, 3, 5)

    train_manifest = build_hash_manifest(train_set.hashes, "TRAIN-MASKED")
    val_manifest = build_hash_manifest(val_set.hashes, "VAL-SEEN")
    proxy_manifest = build_hash_manifest(proxy_set.hashes, "PROXY-DEV")
    assert_manifests_disjoint(train_manifest, val_manifest)
    assert_manifests_disjoint(train_manifest, proxy_manifest)
    assert_manifests_disjoint(val_manifest, proxy_manifest)
