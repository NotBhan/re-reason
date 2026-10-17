r"""Milestone 3B final held-out evaluation driver (single authorized test pass).

Implements the PI-authorized single evaluation pass on the canonical pathfinding
``test`` partition ($N = 50$ episodes, protocol seed space ``base_seed + 90000``
with ``base_seed = 42``). The existing validation driver is preserved unchanged;
this module imports its partition-agnostic evaluation and aggregation helpers and
adds only the test-specific pieces:

* An explicit ``--heldout-test-authorized`` gate.
* A hard single-pass guard: if any prior test artifact exists under
  ``experiments/phase3b_test_runs/``, execution refuses to run.
* Exact frozen checkpoint paths and full SHA-256 verification against the
  authoritative audited validation artifact, strict architecture loading,
  metadata checks, and tensor-finiteness checks.
* Two-rollout evaluation with the protocol's **test** partition offset
  (``200000``) instead of the validation offset.
* A descriptive validation-versus-test comparison that keeps the partitions
  strictly separate (no pooled estimate).

Frozen conditions: model seeds 1-5, four systems (RE:Reason, un-gated,
action-masking, two-rollout), CPU, zero-initialized initial latent state,
``T_max = 24`` (26/27 for two-rollout), fixed ``tau = 1.0``, protocol rollout
seed formula ``500000 + 10000*s + part_offset + 10*i + traj_id``. No training,
fine-tuning, tuning, seed expansion, or second test pass is implemented.
Protected modular-addition cases ($b \in \{0, 5\}$) are never referenced.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from re_reason.benchmarks.pathfinding_benchmark import (
    PathfindingEpisode,
    PathfindingGeneratorConfig,
    generate_pathfinding_partition,
    solve_shortest_path,
)
from re_reason.core.pathfinding_baselines import TwoRolloutRecurrentBaseline, rollout_two_trajectory
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher
from re_reason.utils.rng_namespaces import ROLLOUT_PARTITION_OFFSETS, rollout_seed
from experiments.run_phase3b_training import check_artifact_targets, write_json_atomic
from experiments.run_phase3b_validation import (
    BASELINE_SYSTEMS,
    DESIGNATED_BASELINE,
    MARGIN,
    SINGLE_STEP_FLOPS,
    SYSTEMS,
    T_CRITICAL,
    TWO_ROLLOUT_HORIZONS,
    _base_record,
    aggregate_episodes,
    classify_margin_decision,
    evaluate_action_masked_episode,
    evaluate_re_reason_episode,
    evaluate_un_gated_episode,
    paired_confidence_interval,
)


TEST_EVALUATION_PARTITION = "test"
TEST_SIZE = 50
GRAPH_BASE_SEED = 42
MODEL_SEEDS = (1, 2, 3, 4, 5)
OUTPUT_ROOT = "experiments/phase3b_test_runs"
VALIDATION_ARTIFACT = (
    "experiments/phase3b_validation_runs/phase3b_validation_20261017T135606Z/"
    "validation_results.json"
)

TEST_CHECKPOINT_PATHS: Dict[int, str] = {
    1: "experiments/phase3b_runs/phase3b_bc_seed1_full_20261017T133520Z.pt",
    2: "experiments/phase3b_runs/phase3b_bc_seed2_full_20261017T133530Z.pt",
    3: "experiments/phase3b_runs/phase3b_bc_seed3_full_20261017T133539Z.pt",
    4: "experiments/phase3b_runs/phase3b_bc_seed4_full_20261017T133548Z.pt",
    5: "experiments/phase3b_runs/phase3b_bc_seed5_full_20261017T133557Z.pt",
}


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def assert_no_prior_test_pass(output_root: str = OUTPUT_ROOT) -> None:
    """Hard single-pass guard: refuse execution when a test artifact already exists."""
    prior = sorted(
        glob.glob(os.path.join(output_root, "phase3b_test_*", "test_results.json"))
    )
    if prior:
        raise RuntimeError(
            f"a held-out test pass artifact already exists ({prior}); a second test pass "
            "is not authorized"
        )


def load_frozen_checkpoints() -> List[Dict[str, Any]]:
    """Verifies the exact frozen checkpoints against the audited hashes and loads them."""
    with open(VALIDATION_ARTIFACT, encoding="utf-8") as handle:
        validation = json.load(handle)
    audited_hashes = {
        int(entry["seed"]): entry["sha256"] for entry in validation["checkpoints"]
    }
    loaded: List[Dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        path = TEST_CHECKPOINT_PATHS[seed]
        if not os.path.exists(path):
            raise RuntimeError(f"required frozen checkpoint missing: {path}")
        digest = _sha256_file(path)
        if digest != audited_hashes.get(seed):
            raise RuntimeError(
                f"checkpoint digest mismatch for seed {seed}: {path} ({digest})"
            )
        payload = torch.load(path, weights_only=False)
        metadata = payload.get("metadata", {})
        if metadata.get("seed") != seed or metadata.get("training_partition") != "train":
            raise RuntimeError(f"checkpoint metadata mismatch for {path}")
        brancher = ActionConditionedBrancher()
        brancher.load_state_dict(payload["state_dict"], strict=True)
        if brancher.count_parameters() != 171660:
            raise RuntimeError(f"unexpected parameter count for {path}")
        for name, tensor in payload["state_dict"].items():
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"non-finite checkpoint tensor {name} for {path}")
        brancher.eval()
        loaded.append(
            {
                "seed": seed,
                "path": path,
                "sha256": digest,
                "parameter_count": brancher.count_parameters(),
                "training_partition": metadata["training_partition"],
                "hyperparameters": metadata.get("hyperparameters"),
                "brancher": brancher,
            }
        )
    return loaded


def evaluate_two_rollout_test_episode(
    brancher: ActionConditionedBrancher,
    episode: PathfindingEpisode,
    episode_index: int,
    bfs_moves: int,
    model_seed: int,
) -> Dict[str, Any]:
    """Evaluates one test episode with the two-rollout baseline (test RNG offset)."""
    model = TwoRolloutRecurrentBaseline(brancher=brancher)
    start_time = time.perf_counter()
    result = rollout_two_trajectory(
        model,
        episode,
        model_seed=model_seed,
        episode_index=episode_index,
        partition=TEST_EVALUATION_PARTITION,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    first = result.trajectory_1
    second = result.trajectory_2
    record = _base_record(
        episode,
        episode_index,
        bfs_moves,
        result.reached_goal,
        invalid_attempts=0,
        steps=result.selected_transitions,
        flops_executed=(first.transitions_used + second.transitions_used) * SINGLE_STEP_FLOPS,
        latency_ms=latency_ms,
        executed_decision_steps=first.transitions_used + second.transitions_used,
    )
    record.update(
        {
            "selected_trajectory": result.selected_trajectory,
            "rollout_seeds": {
                "trajectory_1": rollout_seed(
                    model_seed, episode_index, 1, TEST_EVALUATION_PARTITION
                ),
                "trajectory_2": rollout_seed(
                    model_seed, episode_index, 2, TEST_EVALUATION_PARTITION
                ),
            },
            "trajectory_1": {
                "reached_goal": first.reached_goal,
                "steps": first.transitions_used,
            },
            "trajectory_2": {
                "reached_goal": second.reached_goal,
                "steps": second.transitions_used,
            },
            "deadlock": bool(not result.reached_goal),
            "horizon_exhausted": bool(
                (not first.reached_goal and first.transitions_used >= TWO_ROLLOUT_HORIZONS[0])
                or (not second.reached_goal and second.transitions_used >= TWO_ROLLOUT_HORIZONS[1])
            ),
        }
    )
    return record


def seed_grr_table(systems_payload: Dict[str, Any]) -> Dict[str, Dict[int, float]]:
    """Extracts per-system, per-seed GRR values from a systems payload block."""
    table: Dict[str, Dict[int, float]] = {}
    for system, block in systems_payload.items():
        table[system] = {
            int(entry["seed"]): float(entry["metrics"]["grr"])
            for entry in block["per_seed"]
        }
    return table


def mean_paired_difference(
    seed_grr: Dict[str, Dict[int, float]], comparator: str
) -> float:
    return statistics.mean(
        seed_grr["re_reason"][seed] - seed_grr[comparator][seed] for seed in MODEL_SEEDS
    )


def compare_partition_patterns(
    validation_grr: Dict[str, Dict[int, float]],
    test_grr: Dict[str, Dict[int, float]],
) -> Dict[str, Any]:
    """Descriptively compares relative patterns; never pools partition episodes."""
    validation_primary = mean_paired_difference(validation_grr, DESIGNATED_BASELINE)
    test_primary = mean_paired_difference(test_grr, DESIGNATED_BASELINE)
    validation_action = mean_paired_difference(
        validation_grr, "action_masking_recurrent"
    )
    test_action = mean_paired_difference(test_grr, "action_masking_recurrent")
    validation_two = mean_paired_difference(validation_grr, "two_rollout_recurrent")
    test_two = mean_paired_difference(test_grr, "two_rollout_recurrent")
    return {
        "validation_mean_difference_vs_un_gated": validation_primary,
        "test_mean_difference_vs_un_gated": test_primary,
        "re_reason_vs_un_gated_positive_in_both": bool(
            validation_primary > 0.0 and test_primary > 0.0
        ),
        "validation_mean_difference_vs_action_masking": validation_action,
        "test_mean_difference_vs_action_masking": test_action,
        "re_reason_vs_action_masking_identical_in_both": bool(
            validation_action == 0.0 and test_action == 0.0
        ),
        "validation_mean_difference_vs_two_rollout": validation_two,
        "test_mean_difference_vs_two_rollout": test_two,
        "re_reason_vs_two_rollout_negative_in_both": bool(
            validation_two < 0.0 and test_two < 0.0
        ),
        "partitions_pooled": False,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 3B final held-out test evaluation (requires --heldout-test-authorized; "
            "exactly one pass)"
        )
    )
    parser.add_argument(
        "--heldout-test-authorized",
        action="store_true",
        help="Required explicit PI authorization for the single held-out test pass.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--allow-overwrite-authorized", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.heldout_test_authorized:
        parser.error(
            "held-out test evaluation is not authorized without explicit consent; "
            "re-run with --heldout-test-authorized only under PI authorization."
        )

    assert_no_prior_test_pass(args.output_root)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_directory = os.path.join(args.output_root, f"phase3b_test_{timestamp}")
    results_path = os.path.join(run_directory, "test_results.json")
    overwritten = check_artifact_targets(
        [results_path], args.allow_overwrite_authorized
    )

    loaded_checkpoints = load_frozen_checkpoints()
    checkpoints = []
    for entry in loaded_checkpoints:
        checkpoint_entry = {key: value for key, value in entry.items() if key != "brancher"}
        checkpoints.append(checkpoint_entry)

    episodes = generate_pathfinding_partition(
        PathfindingGeneratorConfig(), TEST_EVALUATION_PARTITION, TEST_SIZE, GRAPH_BASE_SEED
    )
    if len(episodes) != TEST_SIZE:
        raise RuntimeError(f"expected {TEST_SIZE} test episodes, got {len(episodes)}")
    if len({episode.digest() for episode in episodes}) != len(episodes):
        raise RuntimeError("test partition contains duplicate episodes")
    bfs_moves = []
    for episode in episodes:
        solution = solve_shortest_path(episode)
        if not solution.solved:
            raise RuntimeError("test partition contains an unsolvable episode")
        bfs_moves.append(solution.num_moves)
    print(
        f"Test episodes: {len(episodes)} | unique digests: "
        f"{len({episode.digest() for episode in episodes})} | "
        f"BFS moves min={min(bfs_moves)} max={max(bfs_moves)}"
    )

    systems: Dict[str, Any] = {system: {"per_seed": []} for system in SYSTEMS}
    seed_grr: Dict[str, Dict[int, float]] = {system: {} for system in SYSTEMS}

    for loaded in loaded_checkpoints:
        seed = loaded["seed"]
        brancher = loaded["brancher"]
        print(f"Loaded frozen checkpoint seed {seed}: {loaded['path']}")
        for system in SYSTEMS:
            records = []
            for index, (episode, bfs) in enumerate(zip(episodes, bfs_moves)):
                if system == "re_reason":
                    record = evaluate_re_reason_episode(brancher, episode, index, bfs)
                elif system == "un_gated_recurrent":
                    record = evaluate_un_gated_episode(brancher, episode, index, bfs)
                elif system == "action_masking_recurrent":
                    record = evaluate_action_masked_episode(brancher, episode, index, bfs)
                elif system == "two_rollout_recurrent":
                    record = evaluate_two_rollout_test_episode(
                        brancher, episode, index, bfs, model_seed=seed
                    )
                else:  # pragma: no cover - defensive
                    raise RuntimeError(f"unknown system {system}")
                records.append(record)
            aggregate = aggregate_episodes(records)
            systems[system]["per_seed"].append(
                {"seed": seed, "metrics": aggregate, "episodes": records}
            )
            seed_grr[system][seed] = aggregate["grr"]
            print(
                f"  seed {seed} | {system}: GRR={aggregate['grr']:.4f} "
                f"({aggregate['successes']}/{aggregate['episodes']}) | "
                f"goal_rate={aggregate['goal_reaching_rate']:.4f} | "
                f"mean_steps_success={aggregate['mean_steps_success']}"
            )

    for system in SYSTEMS:
        per_seed = [seed_grr[system][seed] for seed in MODEL_SEEDS]
        systems[system]["aggregate"] = {
            "mean_grr": statistics.mean(per_seed),
            "sd_grr": statistics.stdev(per_seed),
            "per_seed_grr": {str(seed): seed_grr[system][seed] for seed in MODEL_SEEDS},
        }

    comparisons = {}
    for system in BASELINE_SYSTEMS:
        differences = [
            seed_grr["re_reason"][seed] - seed_grr[system][seed] for seed in MODEL_SEEDS
        ]
        interval = paired_confidence_interval(differences)
        interval.update(classify_margin_decision(interval["ci_lower"], interval["ci_upper"]))
        comparisons[system] = interval
    primary = comparisons[DESIGNATED_BASELINE]

    with open(VALIDATION_ARTIFACT, encoding="utf-8") as handle:
        validation_payload = json.load(handle)
    validation_grr = seed_grr_table(validation_payload["systems"])
    validation_primary = validation_payload["paired_analysis"]["comparisons"][
        DESIGNATED_BASELINE
    ]
    pattern_comparison = compare_partition_patterns(validation_grr, seed_grr)
    pattern_comparison.update(
        {
            "validation_primary_decision": validation_primary["decision"],
            "validation_primary_subcase": validation_primary["subcase"],
            "validation_margin_met": validation_primary["decision"]
            == "practical_improvement_supported",
            "test_primary_decision": primary["decision"],
            "test_primary_subcase": primary["subcase"],
            "test_margin_met": primary["decision"] == "practical_improvement_supported",
        }
    )

    payload: Dict[str, Any] = {
        "runner": "run_phase3b_test_evaluation",
        "status": "completed",
        "timestamp_utc": timestamp,
        "protocol_reference": "docs/research/pre_3b_experimental_protocol.md",
        "evaluation_scope": "heldout_test_single_pass",
        "single_pass_record": {
            "test_passes_performed": 1,
            "additional_test_passes": 0,
            "retraining_performed": False,
            "tuning_performed": False,
            "seed_expansion_performed": False,
        },
        "frozen_configuration": {
            "partition": TEST_EVALUATION_PARTITION,
            "test_episodes": TEST_SIZE,
            "graph_base_seed": GRAPH_BASE_SEED,
            "model_seeds": list(MODEL_SEEDS),
            "systems": list(SYSTEMS),
            "device": args.device,
            "initial_latent_state": "zero",
            "rollout_temperature": 1.0,
            "two_rollout_horizons": list(TWO_ROLLOUT_HORIZONS),
            "t_max": 24,
            "training_configuration": {
                "epochs": 30,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
                "optimizer": "AdamW",
            },
            "evaluation_rng": {
                "deterministic_systems": [
                    "re_reason",
                    "un_gated_recurrent",
                    "action_masking_recurrent",
                ],
                "two_rollout_seed_formula": (
                    "500000 + model_seed*10000 + part_offset + episode_index*10 + trajectory_id"
                ),
                "two_rollout_partition_offsets": dict(ROLLOUT_PARTITION_OFFSETS),
                "test_partition_offset": ROLLOUT_PARTITION_OFFSETS[TEST_EVALUATION_PARTITION],
            },
            "designated_comparison_baseline": DESIGNATED_BASELINE,
            "margin_delta": MARGIN,
            "t_critical_by_seed_count": {str(k): v for k, v in T_CRITICAL.items()},
            "grr_definition": (
                "fraction of test episodes reaching the goal within the allocated transition "
                "budget without any invalid transition attempt"
            ),
            "statistical_unit": "paired model seed",
            "paired_difference": "GRR_RE:Reason,s - GRR_baseline,s",
        },
        "validation_artifact_reference": VALIDATION_ARTIFACT,
        "checkpoints": checkpoints,
        "test_episodes": [
            {
                "episode_index": index,
                "digest": episode.digest(),
                "start": episode.start,
                "goal": episode.goal,
                "bfs_moves": bfs_moves[index],
            }
            for index, episode in enumerate(episodes)
        ],
        "systems": systems,
        "paired_analysis": {
            "designated_baseline": DESIGNATED_BASELINE,
            "comparisons": comparisons,
            "seed_expansion": {
                "authorized_from_test": False,
                "performed": False,
                "reason": "the held-out test pass is a final estimate and does not authorize seed expansion",
            },
        },
        "validation_comparison": {
            "validation_artifact": VALIDATION_ARTIFACT,
            "validation_mean_grr": {
                system: validation_payload["systems"][system]["aggregate"]["mean_grr"]
                for system in SYSTEMS
            },
            "test_mean_grr": {
                system: systems[system]["aggregate"]["mean_grr"] for system in SYSTEMS
            },
            "validation_primary_comparison": validation_primary,
            "test_primary_comparison": primary,
            "partition_patterns": pattern_comparison,
            "pooled_validation_test_estimate": False,
        },
        "errors": [],
        "warnings": [],
        "overwrite_authorized": bool(args.allow_overwrite_authorized),
        "overwritten_artifacts": overwritten,
    }

    write_json_atomic(results_path, payload)
    print(f"Test evaluation results written: {results_path}")
    print(
        f"Primary comparison ({DESIGNATED_BASELINE}): "
        f"mean={primary['mean']:.4f} sd={primary['sd']:.4f} "
        f"95% CI=[{primary['ci_lower']:.4f}, {primary['ci_upper']:.4f}] "
        f"decision={primary['decision']} ({primary['subcase']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
