"""Unit tests for the final held-out test evaluation driver (synthetic episodes only)."""

import os

import pytest

from re_reason.benchmarks.pathfinding_benchmark import PathfindingEpisode
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher

from experiments import run_phase3b_test_evaluation as test_evaluation


def _line_episode(length: int = 2) -> PathfindingEpisode:
    edges = frozenset((index, index + 1) for index in range(length))
    return PathfindingEpisode(num_nodes=12, start=0, goal=length, edges=edges)


def test_checkpoint_manifest_is_frozen() -> None:
    assert set(test_evaluation.TEST_CHECKPOINT_PATHS) == {1, 2, 3, 4, 5}
    expected_filenames = {
        1: "phase3b_bc_seed1_full_20261017T133520Z.pt",
        2: "phase3b_bc_seed2_full_20261017T133530Z.pt",
        3: "phase3b_bc_seed3_full_20261017T133539Z.pt",
        4: "phase3b_bc_seed4_full_20261017T133548Z.pt",
        5: "phase3b_bc_seed5_full_20261017T133557Z.pt",
    }
    for seed, filename in expected_filenames.items():
        path = test_evaluation.TEST_CHECKPOINT_PATHS[seed]
        assert os.path.basename(path) == filename
        assert "smoke" not in path
    assert test_evaluation.TEST_EVALUATION_PARTITION == "test"
    assert test_evaluation.TEST_SIZE == 50
    assert test_evaluation.VALIDATION_ARTIFACT.endswith(
        "phase3b_validation_20261017T135606Z/validation_results.json"
    )


def test_two_rollout_test_evaluation_passes_test_partition(monkeypatch) -> None:
    captured = {}

    def fake_rollout(model, episode, *, model_seed, episode_index, partition):
        captured["partition"] = partition
        captured["model_seed"] = model_seed
        captured["episode_index"] = episode_index
        raise RuntimeError("captured")

    monkeypatch.setattr(test_evaluation, "rollout_two_trajectory", fake_rollout)
    brancher = ActionConditionedBrancher()
    with pytest.raises(RuntimeError, match="captured"):
        test_evaluation.evaluate_two_rollout_test_episode(
            brancher, _line_episode(2), 0, 2, model_seed=1
        )
    assert captured == {"partition": "test", "model_seed": 1, "episode_index": 0}


def test_two_rollout_test_episode_uses_test_rollout_offset() -> None:
    brancher = ActionConditionedBrancher()
    episode = PathfindingEpisode(
        num_nodes=12, start=0, goal=2, edges=frozenset({(0, 1), (1, 0)})
    )

    record = test_evaluation.evaluate_two_rollout_test_episode(
        brancher, episode, 3, 0, model_seed=2
    )

    base = 500000 + 2 * 10000 + 200000 + 3 * 10
    assert record["rollout_seeds"] == {
        "trajectory_1": base + 1,
        "trajectory_2": base + 2,
    }
    assert record["trajectory_1"]["steps"] == 26
    assert record["trajectory_2"]["steps"] == 27
    assert record["success"] is False
    assert record["horizon_exhausted"] is True


def test_single_pass_guard_refuses_second_pass(tmp_path) -> None:
    test_evaluation.assert_no_prior_test_pass(str(tmp_path))
    prior = tmp_path / "phase3b_test_20260101T000000Z"
    prior.mkdir()
    (prior / "test_results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="second test pass"):
        test_evaluation.assert_no_prior_test_pass(str(tmp_path))


def test_main_requires_authorization_flag(tmp_path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        test_evaluation.main(["--output-root", str(tmp_path)])
    assert exit_info.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_validation_comparison_patterns() -> None:
    validation_grr = {
        "re_reason": {1: 0.5, 2: 0.6, 3: 0.5, 4: 0.5, 5: 0.5},
        "un_gated_recurrent": {1: 0.4, 2: 0.5, 3: 0.4, 4: 0.5, 5: 0.4},
        "action_masking_recurrent": {1: 0.5, 2: 0.6, 3: 0.5, 4: 0.5, 5: 0.5},
        "two_rollout_recurrent": {1: 0.7, 2: 0.7, 3: 0.7, 4: 0.7, 5: 0.7},
    }
    test_grr = {
        "re_reason": {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
        "un_gated_recurrent": {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4, 5: 0.4},
        "action_masking_recurrent": {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
        "two_rollout_recurrent": {1: 0.6, 2: 0.6, 3: 0.6, 4: 0.6, 5: 0.6},
    }

    comparison = test_evaluation.compare_partition_patterns(validation_grr, test_grr)

    assert comparison["re_reason_vs_un_gated_positive_in_both"] is True
    assert comparison["re_reason_vs_action_masking_identical_in_both"] is True
    assert comparison["re_reason_vs_two_rollout_negative_in_both"] is True
    assert comparison["partitions_pooled"] is False
    assert comparison["test_mean_difference_vs_un_gated"] == pytest.approx(0.1)

    divergent = dict(test_grr)
    divergent["un_gated_recurrent"] = {seed: 0.6 for seed in (1, 2, 3, 4, 5)}
    comparison = test_evaluation.compare_partition_patterns(validation_grr, divergent)
    assert comparison["re_reason_vs_un_gated_positive_in_both"] is False
