"""Unit tests for the Stage C validation evaluator contracts using synthetic episodes only."""

import math
import statistics

import pytest
import torch

from experiments.run_phase3b_validation import (
    aggregate_episodes,
    classify_margin_decision,
    episode_success,
    evaluate_action_masked_episode,
    evaluate_re_reason_episode,
    evaluate_two_rollout_episode,
    evaluate_un_gated_episode,
    paired_confidence_interval,
)
from re_reason.benchmarks.pathfinding_benchmark import PathfindingEpisode
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher


def _line_episode(length: int = 2) -> PathfindingEpisode:
    edges = frozenset((index, index + 1) for index in range(length))
    return PathfindingEpisode(num_nodes=12, start=0, goal=length, edges=edges)


def _force_policy_bias(brancher: ActionConditionedBrancher, biases: dict) -> None:
    with torch.no_grad():
        brancher.policy_head.weight.zero_()
        brancher.policy_head.bias.zero_()
        for action, value in biases.items():
            brancher.policy_head.bias[action] = value
    brancher.eval()


def test_episode_success_requires_no_invalid_attempts() -> None:
    assert episode_success(True, 0) is True
    assert episode_success(True, 1) is False
    assert episode_success(False, 0) is False
    assert episode_success(False, 2) is False


def test_paired_confidence_interval_known_values() -> None:
    differences = [0.1, 0.2, 0.3, 0.2, 0.1]
    result = paired_confidence_interval(differences)
    mean = statistics.mean(differences)
    standard_deviation = statistics.stdev(differences)
    standard_error = standard_deviation / math.sqrt(5)

    assert result["n"] == 5
    assert result["differences"] == differences
    assert result["mean"] == pytest.approx(mean)
    assert result["sd"] == pytest.approx(standard_deviation)
    assert result["se"] == pytest.approx(standard_error)
    assert result["t_critical"] == 2.776
    assert result["ci_lower"] == pytest.approx(mean - 2.776 * standard_error)
    assert result["ci_upper"] == pytest.approx(mean + 2.776 * standard_error)

    with pytest.raises(ValueError):
        paired_confidence_interval([0.1])
    with pytest.raises(ValueError):
        paired_confidence_interval([0.1] * 4)


def test_margin_decision_categories() -> None:
    assert classify_margin_decision(0.10, 0.20) == {
        "decision": "practical_improvement_supported",
        "subcase": "lower_bound_meets_margin",
    }
    assert classify_margin_decision(-0.20, -0.01) == {
        "decision": "practical_improvement_not_supported",
        "subcase": "negative_effect",
    }
    assert classify_margin_decision(0.01, 0.09)["subcase"] == "positive_below_threshold"
    assert classify_margin_decision(-0.05, 0.05)["subcase"] == "zero_plausible"
    assert classify_margin_decision(0.05, 0.099)["decision"] == "practical_improvement_not_supported"
    assert classify_margin_decision(0.05, 0.10)["decision"] == "inconclusive"
    assert classify_margin_decision(0.09, 0.11)["decision"] == "inconclusive"


def test_action_masked_episode_metrics_and_optimality() -> None:
    brancher = ActionConditionedBrancher()
    _force_policy_bias(brancher, {1: 10.0, 2: 9.0})
    episode = _line_episode(2)

    record = evaluate_action_masked_episode(brancher, episode, episode_index=0, bfs_moves=2)

    assert record["reached_goal"] is True
    assert record["success"] is True
    assert record["steps"] == 2
    assert record["invalid_attempts"] == 0
    assert record["optimality_ratio"] == pytest.approx(1.0)
    assert record["flops_executed"] == 2 * 341_016
    assert record["deadlock"] is False
    assert record["horizon_exhausted"] is False
    assert record["latency_ms"] >= 0.0


def test_un_gated_episode_counts_invalid_attempts_and_fails_grr() -> None:
    brancher = ActionConditionedBrancher()
    _force_policy_bias(brancher, {5: 10.0})
    episode = _line_episode(2)

    record = evaluate_un_gated_episode(brancher, episode, episode_index=0, bfs_moves=2)

    assert record["reached_goal"] is False
    assert record["success"] is False
    assert record["invalid_attempts"] == 24
    assert record["steps"] == 24
    assert record["horizon_exhausted"] is True
    assert record["flops_executed"] == 24 * 341_016


def test_re_reason_episode_commits_legal_transitions() -> None:
    brancher = ActionConditionedBrancher()
    _force_policy_bias(brancher, {1: 10.0, 2: 9.0})
    episode = _line_episode(2)

    record = evaluate_re_reason_episode(brancher, episode, episode_index=0, bfs_moves=2)

    assert record["reached_goal"] is True
    assert record["success"] is True
    assert record["steps"] == 2
    assert record["invalid_attempts"] == 0
    assert record["decision_attempts"] == 2
    assert record["proposal_rounds"] == 2
    assert record["retry_rounds"] == 0
    assert record["aborted_decisions"] == 0
    assert record["flops_executed"] == 2 * (122_880 + 12_312 + 3 * 205_824)


def test_two_rollout_episode_horizons_and_failure_record() -> None:
    brancher = ActionConditionedBrancher()
    episode = PathfindingEpisode(
        num_nodes=12, start=0, goal=2, edges=frozenset({(0, 1), (1, 0)})
    )

    record = evaluate_two_rollout_episode(
        brancher, episode, episode_index=3, bfs_moves=0, model_seed=2
    )

    assert record["reached_goal"] is False
    assert record["success"] is False
    assert record["trajectory_1"]["steps"] == 26
    assert record["trajectory_2"]["steps"] == 27
    assert record["selected_trajectory"] is None
    assert record["deadlock"] is True
    assert record["horizon_exhausted"] is True
    assert record["flops_executed"] == 53 * 341_016

    repeated = evaluate_two_rollout_episode(
        brancher, episode, episode_index=3, bfs_moves=0, model_seed=2
    )
    assert repeated["trajectory_1"] == record["trajectory_1"]
    assert repeated["trajectory_2"] == record["trajectory_2"]


def test_aggregate_episodes_metrics() -> None:
    records = [
        {
            "success": True,
            "reached_goal": True,
            "invalid_attempts": 0,
            "steps": 2,
            "optimality_ratio": 1.0,
            "flops_executed": 100,
            "latency_ms": 10.0,
            "executed_decision_steps": 2,
        },
        {
            "success": False,
            "reached_goal": False,
            "invalid_attempts": 1,
            "steps": 24,
            "optimality_ratio": None,
            "flops_executed": 200,
            "latency_ms": 20.0,
            "executed_decision_steps": 24,
            "deadlock": False,
            "horizon_exhausted": True,
        },
    ]
    aggregate = aggregate_episodes(records)

    assert aggregate["episodes"] == 2
    assert aggregate["successes"] == 1
    assert aggregate["grr"] == 0.5
    assert aggregate["goal_reaching_rate"] == 0.5
    assert aggregate["invalid_attempts_total"] == 1
    assert aggregate["mean_steps_success"] == 2
    assert aggregate["mean_optimality_ratio_success"] == 1.0
    assert aggregate["mean_flops_per_episode"] == 150
    assert aggregate["mean_flops_per_success"] == 100
    assert aggregate["mean_latency_ms_per_episode"] == 15.0
    assert aggregate["ms_per_executed_decision_step"] == pytest.approx(30.0 / 26.0)
    assert aggregate["horizon_exhaustions"] == 1
    assert aggregate["retry_rounds_total"] == "not_applicable"
    assert aggregate["rollbacks_total"] == "not_applicable"
