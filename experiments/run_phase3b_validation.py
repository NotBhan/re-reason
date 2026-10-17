"""Milestone 3B Stage C validation evaluation driver (validation partition only).

Implements the ratified pre-3B protocol's validation-only evaluation
(``docs/research/pre_3b_experimental_protocol.md``):

* Data: the canonical pathfinding ``validation`` partition ($N = 50$ episodes,
  graph seed space ``base_seed + 50000`` with ``base_seed = 42``). The held-out
  ``test`` partition is never generated, loaded, or referenced.
* Systems: RE:Reason (verifier-gated candidate branching through
  ``TwoPhaseTransitionOrchestrator``), ``UnGatedRecurrentBaseline``,
  ``ActionMaskingRecurrentBaseline``, and ``TwoRolloutRecurrentBaseline``.
  All four share the five Stage B full-training checkpoints (paired model seeds
  ``1..5``); only inference behavior differs.
* Primary metric (protocol §5.1): Goal-Reaching Rate (GRR) = fraction of the 50
  validation episodes in which the system arrives at the goal within its
  allocated transition budget **without any invalid transition attempt**.
  The independent statistical unit is the paired model seed.
* Designated comparison baseline (protocol §2.1): the **Un-Gated Recurrent
  Residual Baseline** (the protocol's "Primary Baseline"). Paired difference per
  seed: ``Delta_s = GRR_RE:Reason,s - GRR_baseline,s``; the two-sided 95%
  Student-t interval uses ``t = 2.776`` at five seeds (``df = 4``) and the
  practical margin ``delta = +0.10`` absolute GRR. The other two comparators are
  reported as additional descriptive comparisons.
* Seed expansion (protocol §6.2): if and only if the designated comparison's
  interval spans ``+0.10``, seeds 6-10 are required. This driver never trains:
  if those checkpoints do not already exist, it reports that additional training
  authorization is required and stops.

Rollout rules: RE:Reason uses the Milestone 3A two-phase orchestrator
(``T_max = 24``, ``R_max = 3``, top-3 legal candidates, verifier-gated commit);
the un-gated and action-masking baselines use single 24-transition trajectories;
the two-rollout baseline uses static horizons 26/27 at fixed ``tau = 1.0`` with
the protocol's isolated partition-scoped rollout seeds and public trajectory
selection. Nominal FLOP accounting follows protocol §7.5
(encoder 122,880; policy 12,312; branch core 205,824; K = 3).

This driver adds no training, checkpoint selection, or held-out path; artifacts
are written once to a unique timestamped directory with overwrite refusal and
atomic replacement.
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
from re_reason.core.pathfinding_baselines import (
    ActionMaskingRecurrentBaseline,
    TwoRolloutRecurrentBaseline,
    UnGatedRecurrentBaseline,
    rollout_action_masked,
    rollout_two_trajectory,
    rollout_un_gated,
)
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher
from re_reason.hypothesis.two_phase_orchestrator import TwoPhaseTransitionOrchestrator
from re_reason.state.container import LatentState
from re_reason.utils.rng_namespaces import ROLLOUT_PARTITION_OFFSETS, rollout_seed
from experiments.run_phase3b_training import check_artifact_targets, write_json_atomic


VALIDATION_SIZE = 50
GRAPH_BASE_SEED = 42
MODEL_SEEDS = (1, 2, 3, 4, 5)
CHECKPOINT_GLOB = "experiments/phase3b_runs/phase3b_bc_seed{seed}_full_*.pt"
OUTPUT_ROOT = "experiments/phase3b_validation_runs"

MARGIN = 0.10
T_CRITICAL = {5: 2.776, 10: 2.262}

T_MAX = 24
TWO_ROLLOUT_HORIZONS = (26, 27)
CANDIDATE_BUDGET = 3
ENCODER_FLOPS = 122_880
POLICY_FLOPS = 12_312
BRANCH_FLOPS = 205_824
SINGLE_STEP_FLOPS = 341_016

DESIGNATED_BASELINE = "un_gated_recurrent"
BASELINE_SYSTEMS = ("un_gated_recurrent", "action_masking_recurrent", "two_rollout_recurrent")
SYSTEMS = ("re_reason",) + BASELINE_SYSTEMS


def episode_success(reached_goal: bool, invalid_attempts: int) -> bool:
    """Protocol GRR success: goal reached with no invalid transition attempt."""
    return bool(reached_goal) and int(invalid_attempts) == 0


def paired_confidence_interval(differences: Sequence[float]) -> Dict[str, Any]:
    """Two-sided 95% Student-t interval for paired seed differences."""
    values = [float(value) for value in differences]
    if len(values) < 2:
        raise ValueError("at least two paired differences are required")
    if len(values) not in T_CRITICAL:
        raise ValueError(f"no ratified t-critical value for {len(values)} seeds")
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    t_critical = T_CRITICAL[len(values)]
    return {
        "n": len(values),
        "differences": values,
        "mean": mean,
        "sd": standard_deviation,
        "se": standard_error,
        "t_critical": t_critical,
        "ci_lower": mean - t_critical * standard_error,
        "ci_upper": mean + t_critical * standard_error,
    }


def classify_margin_decision(
    ci_lower: float, ci_upper: float, margin: float = MARGIN
) -> Dict[str, str]:
    """Applies the protocol's practical-margin decision rule."""
    if ci_lower >= margin:
        return {"decision": "practical_improvement_supported", "subcase": "lower_bound_meets_margin"}
    if ci_upper < margin:
        if ci_upper < 0.0:
            return {"decision": "practical_improvement_not_supported", "subcase": "negative_effect"}
        if ci_lower > 0.0:
            return {"decision": "practical_improvement_not_supported", "subcase": "positive_below_threshold"}
        return {"decision": "practical_improvement_not_supported", "subcase": "zero_plausible"}
    return {"decision": "inconclusive", "subcase": "interval_spans_margin"}


def _base_record(
    episode: PathfindingEpisode,
    episode_index: int,
    bfs_moves: int,
    reached_goal: bool,
    invalid_attempts: int,
    steps: int,
    flops_executed: int,
    latency_ms: float,
    executed_decision_steps: int,
) -> Dict[str, Any]:
    success = episode_success(reached_goal, invalid_attempts)
    optimality = (steps / bfs_moves) if (success and bfs_moves > 0) else None
    return {
        "episode_index": episode_index,
        "episode_digest": episode.digest(),
        "reached_goal": bool(reached_goal),
        "success": success,
        "invalid_attempts": int(invalid_attempts),
        "steps": int(steps),
        "bfs_moves": int(bfs_moves),
        "optimality_ratio": optimality,
        "flops_executed": int(flops_executed),
        "latency_ms": float(latency_ms),
        "executed_decision_steps": int(executed_decision_steps),
    }


def evaluate_re_reason_episode(
    brancher: ActionConditionedBrancher,
    episode: PathfindingEpisode,
    episode_index: int,
    bfs_moves: int,
) -> Dict[str, Any]:
    """Evaluates one episode with verifier-gated candidate branching (RE:Reason)."""
    reference = brancher.node_embeddings.weight
    container = LatentState(
        torch.zeros(1, brancher.state_dim, dtype=reference.dtype, device=reference.device)
    )
    orchestrator = TwoPhaseTransitionOrchestrator(episode, container, brancher)
    decisions = 0
    rounds = 0
    aborted_decisions = 0
    retry_exhaustions = 0

    start_time = time.perf_counter()
    with torch.no_grad():
        while (
            orchestrator.environment_state.transitions < T_MAX
            and not episode.is_goal_reached(orchestrator.environment_state)
        ):
            result = orchestrator.step()
            decisions += 1
            rounds += max(1, result.rounds_attempted)
            if not result.committed:
                aborted_decisions += 1
                if result.rounds_attempted >= TwoPhaseTransitionOrchestrator.MAX_PROPOSAL_ROUNDS:
                    retry_exhaustions += 1
                break
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    environment = orchestrator.environment_state
    reached_goal = episode.is_goal_reached(environment)
    steps = environment.transitions
    flops_executed = decisions * ENCODER_FLOPS + rounds * (
        POLICY_FLOPS + CANDIDATE_BUDGET * BRANCH_FLOPS
    )
    record = _base_record(
        episode,
        episode_index,
        bfs_moves,
        reached_goal,
        invalid_attempts=0,
        steps=steps,
        flops_executed=flops_executed,
        latency_ms=latency_ms,
        executed_decision_steps=decisions,
    )
    record.update(
        {
            "decision_attempts": decisions,
            "proposal_rounds": rounds,
            "retry_rounds": rounds - decisions,
            "aborted_decisions": aborted_decisions,
            "retry_exhaustions": retry_exhaustions,
            "verifier_rejections": max(0, rounds - decisions),
            "rollbacks": 0,
            "deadlock": bool(aborted_decisions > 0 and not reached_goal),
            "horizon_exhausted": bool(steps >= T_MAX and not reached_goal),
        }
    )
    return record


def evaluate_un_gated_episode(
    brancher: ActionConditionedBrancher,
    episode: PathfindingEpisode,
    episode_index: int,
    bfs_moves: int,
) -> Dict[str, Any]:
    """Evaluates one episode with the un-gated recurrent residual baseline."""
    model = UnGatedRecurrentBaseline(brancher=brancher)
    start_time = time.perf_counter()
    result = rollout_un_gated(model, episode, max_transitions=T_MAX)
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    record = _base_record(
        episode,
        episode_index,
        bfs_moves,
        result.reached_goal,
        result.invalid_attempts,
        result.transitions_used,
        result.transitions_used * SINGLE_STEP_FLOPS,
        latency_ms,
        result.transitions_used,
    )
    record.update(
        {
            "deadlock": "not_applicable",
            "horizon_exhausted": bool(result.transitions_used >= T_MAX and not result.reached_goal),
        }
    )
    return record


def evaluate_action_masked_episode(
    brancher: ActionConditionedBrancher,
    episode: PathfindingEpisode,
    episode_index: int,
    bfs_moves: int,
) -> Dict[str, Any]:
    """Evaluates one episode with the action-masking recurrent baseline."""
    model = ActionMaskingRecurrentBaseline(brancher=brancher)
    start_time = time.perf_counter()
    result = rollout_action_masked(model, episode, max_transitions=T_MAX)
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    record = _base_record(
        episode,
        episode_index,
        bfs_moves,
        result.reached_goal,
        invalid_attempts=0,
        steps=result.transitions_used,
        flops_executed=result.transitions_used * SINGLE_STEP_FLOPS,
        latency_ms=latency_ms,
        executed_decision_steps=result.transitions_used,
    )
    stopped_early = bool(not result.reached_goal and result.transitions_used < T_MAX)
    record.update(
        {
            "deadlock": stopped_early,
            "horizon_exhausted": bool(
                not result.reached_goal and result.transitions_used >= T_MAX
            ),
        }
    )
    return record


def evaluate_two_rollout_episode(
    brancher: ActionConditionedBrancher,
    episode: PathfindingEpisode,
    episode_index: int,
    bfs_moves: int,
    model_seed: int,
) -> Dict[str, Any]:
    """Evaluates one episode with the episode-budget-matched two-rollout baseline."""
    model = TwoRolloutRecurrentBaseline(brancher=brancher)
    start_time = time.perf_counter()
    result = rollout_two_trajectory(
        model,
        episode,
        model_seed=model_seed,
        episode_index=episode_index,
        partition="validation",
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    first = result.trajectory_1
    second = result.trajectory_2
    selected = result.selected_result
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
                "trajectory_1": rollout_seed(model_seed, episode_index, 1, "validation"),
                "trajectory_2": rollout_seed(model_seed, episode_index, 2, "validation"),
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
    assert selected is None or selected is result.trajectory_1 or selected is result.trajectory_2
    return record


def aggregate_episodes(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates episode records into per-seed validation metrics."""
    if not records:
        raise ValueError("at least one episode record is required")
    episodes = len(records)
    successes = [record for record in records if record["success"]]
    goal_reaches = [record for record in records if record["reached_goal"]]

    def _mean(values: Sequence[float]) -> Optional[float]:
        return float(statistics.mean(values)) if values else None

    total_latency = sum(record["latency_ms"] for record in records)
    total_decision_steps = sum(record["executed_decision_steps"] for record in records)
    retry_applicable = any("retry_rounds" in record for record in records)
    retry_metrics: Dict[str, Any] = {
        "retry_rounds_total": (
            sum(record.get("retry_rounds", 0) for record in records)
            if retry_applicable
            else "not_applicable"
        ),
        "aborted_decisions_total": (
            sum(record.get("aborted_decisions", 0) for record in records)
            if retry_applicable
            else "not_applicable"
        ),
        "retry_exhaustions_total": (
            sum(record.get("retry_exhaustions", 0) for record in records)
            if retry_applicable
            else "not_applicable"
        ),
        "verifier_rejections_total": (
            sum(record.get("verifier_rejections", 0) for record in records)
            if retry_applicable
            else "not_applicable"
        ),
        "rollbacks_total": (
            sum(record.get("rollbacks", 0) for record in records)
            if retry_applicable
            else "not_applicable"
        ),
    }
    deadlock_applicable = not all(
        record.get("deadlock") == "not_applicable" for record in records
    )
    aggregate = {
        "episodes": episodes,
        "successes": len(successes),
        "grr": len(successes) / episodes,
        "goal_reaches": len(goal_reaches),
        "goal_reaching_rate": len(goal_reaches) / episodes,
        "invalid_attempts_total": sum(record["invalid_attempts"] for record in records),
        "mean_steps_success": _mean([record["steps"] for record in successes]),
        "mean_optimality_ratio_success": _mean(
            [record["optimality_ratio"] for record in successes]
        ),
        "mean_flops_per_episode": _mean([record["flops_executed"] for record in records]),
        "mean_flops_per_success": _mean([record["flops_executed"] for record in successes]),
        "mean_latency_ms_per_episode": _mean([record["latency_ms"] for record in records]),
        "ms_per_executed_decision_step": (
            total_latency / total_decision_steps if total_decision_steps else None
        ),
        "deadlocks": (
            sum(1 for record in records if record.get("deadlock") is True)
            if deadlock_applicable
            else "not_applicable"
        ),
        "horizon_exhaustions": sum(
            1 for record in records if record.get("horizon_exhausted") is True
        ),
    }
    aggregate.update(retry_metrics)
    return aggregate


def _resolve_checkpoint(seed: int) -> str:
    matches = sorted(glob.glob(CHECKPOINT_GLOB.format(seed=seed)))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one full-training checkpoint for seed {seed}, found {matches}"
        )
    return matches[0]


def _checkpoint_digest(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_brancher(seed: int) -> Dict[str, Any]:
    path = _resolve_checkpoint(seed)
    payload = torch.load(path, weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("seed") != seed:
        raise RuntimeError(f"checkpoint metadata seed mismatch for {path}")
    if metadata.get("training_partition") != "train":
        raise RuntimeError(f"checkpoint metadata is not a train-partition artifact: {path}")
    brancher = ActionConditionedBrancher()
    brancher.load_state_dict(payload["state_dict"], strict=True)
    if brancher.count_parameters() != 171660:
        raise RuntimeError(f"unexpected parameter count for {path}")
    brancher.eval()
    return {
        "seed": seed,
        "path": path,
        "sha256": _checkpoint_digest(path),
        "parameter_count": brancher.count_parameters(),
        "training_partition": metadata["training_partition"],
        "hyperparameters": metadata.get("hyperparameters"),
        "brancher": brancher,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Milestone 3B Stage C validation-only evaluation (requires --stage-c-authorized)"
    )
    parser.add_argument(
        "--stage-c-authorized",
        action="store_true",
        help="Required explicit Stage C validation authorization gate.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--allow-overwrite-authorized", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.stage_c_authorized:
        parser.error(
            "Stage C validation is not authorized without explicit consent; "
            "re-run with --stage-c-authorized only under PI authorization."
        )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_directory = os.path.join(args.output_root, f"phase3b_validation_{timestamp}")
    results_path = os.path.join(run_directory, "validation_results.json")
    overwritten = check_artifact_targets(
        [results_path], args.allow_overwrite_authorized
    )

    episodes = generate_pathfinding_partition(
        PathfindingGeneratorConfig(), "validation", VALIDATION_SIZE, GRAPH_BASE_SEED
    )
    if len(episodes) != VALIDATION_SIZE:
        raise RuntimeError(f"expected {VALIDATION_SIZE} validation episodes, got {len(episodes)}")
    if len({episode.digest() for episode in episodes}) != len(episodes):
        raise RuntimeError("validation partition contains duplicate episodes")
    bfs_moves = []
    for episode in episodes:
        solution = solve_shortest_path(episode)
        if not solution.solved:
            raise RuntimeError("validation partition contains an unsolvable episode")
        bfs_moves.append(solution.num_moves)

    print(
        f"Validation episodes: {len(episodes)} | unique digests: "
        f"{len({episode.digest() for episode in episodes})} | "
        f"BFS moves min={min(bfs_moves)} max={max(bfs_moves)}"
    )

    checkpoints = []
    systems: Dict[str, Any] = {system: {"per_seed": []} for system in SYSTEMS}
    seed_grr: Dict[str, Dict[int, float]] = {system: {} for system in SYSTEMS}

    for seed in MODEL_SEEDS:
        loaded = _load_brancher(seed)
        brancher = loaded.pop("brancher")
        checkpoints.append(loaded)
        print(f"Loaded checkpoint seed {seed}: {loaded['path']}")

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
                    record = evaluate_two_rollout_episode(
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
        per_seed_grrs = [seed_grr[system][seed] for seed in MODEL_SEEDS]
        systems[system]["aggregate"] = {
            "mean_grr": statistics.mean(per_seed_grrs),
            "sd_grr": statistics.stdev(per_seed_grrs),
            "per_seed_grr": {
                str(seed): seed_grr[system][seed] for seed in MODEL_SEEDS
            },
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
    expansion_required = primary["decision"] == "inconclusive"
    expansion_availability = {
        str(seed): bool(glob.glob(CHECKPOINT_GLOB.format(seed=seed))) for seed in (6, 7, 8, 9, 10)
    }
    expansion_evaluated = False
    if expansion_required:
        if all(expansion_availability.values()):
            raise RuntimeError(
                "seed expansion checkpoints exist but expansion evaluation is not implemented "
                "in this driver; stop for PI review"
            )
        expansion_reason = (
            "designated comparison interval spans +0.10; seeds 6-10 full-training "
            "checkpoints are unavailable, so additional training authorization is required"
        )
    else:
        expansion_reason = (
            "designated comparison interval does not span +0.10; seed expansion not triggered"
        )

    payload: Dict[str, Any] = {
        "runner": "run_phase3b_validation",
        "status": "completed",
        "timestamp_utc": timestamp,
        "protocol_reference": "docs/research/pre_3b_experimental_protocol.md",
        "evaluation_scope": "validation_only",
        "configuration": {
            "partition": "validation",
            "validation_episodes": VALIDATION_SIZE,
            "graph_base_seed": GRAPH_BASE_SEED,
            "model_seeds": list(MODEL_SEEDS),
            "checkpoint_glob": CHECKPOINT_GLOB,
            "systems": list(SYSTEMS),
            "designated_comparison_baseline": DESIGNATED_BASELINE,
            "designated_comparison_source": (
                "protocol section 2.1: 'Primary Baseline: Un-Gated Recurrent Residual Baseline'"
            ),
            "grr_definition": (
                "fraction of validation episodes reaching the goal within the allocated "
                "transition budget without any invalid transition attempt"
            ),
            "statistical_unit": "paired model seed",
            "paired_difference": "GRR_RE:Reason,s - GRR_baseline,s",
            "margin_delta": MARGIN,
            "t_critical_by_seed_count": {str(k): v for k, v in T_CRITICAL.items()},
            "t_max": T_MAX,
            "two_rollout_horizons": list(TWO_ROLLOUT_HORIZONS),
            "rollout_temperature": 1.0,
            "candidate_budget": CANDIDATE_BUDGET,
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
            },
            "flops_nominal": {
                "encoder": ENCODER_FLOPS,
                "policy": POLICY_FLOPS,
                "branch_core": BRANCH_FLOPS,
                "single_step": SINGLE_STEP_FLOPS,
                "re_reason_three_candidates_per_round": ENCODER_FLOPS
                + POLICY_FLOPS
                + CANDIDATE_BUDGET * BRANCH_FLOPS,
            },
            "initial_latent_state": "zero",
            "device": args.device,
        },
        "validation_episodes": [
            {
                "episode_index": index,
                "digest": episode.digest(),
                "start": episode.start,
                "goal": episode.goal,
                "bfs_moves": bfs_moves[index],
            }
            for index, episode in enumerate(episodes)
        ],
        "checkpoints": checkpoints,
        "systems": systems,
        "paired_analysis": {
            "designated_baseline": DESIGNATED_BASELINE,
            "comparisons": comparisons,
            "seed_expansion": {
                "required": expansion_required,
                "evaluated": expansion_evaluated,
                "seeds_6_10_checkpoint_availability": expansion_availability,
                "reason": expansion_reason,
            },
        },
        "overwrite_authorized": bool(args.allow_overwrite_authorized),
        "overwritten_artifacts": overwritten,
        "warnings": [],
    }

    write_json_atomic(results_path, payload)
    print(f"Validation results written: {results_path}")
    print(
        f"Primary comparison ({DESIGNATED_BASELINE}): "
        f"mean={primary['mean']:.4f} sd={primary['sd']:.4f} "
        f"95% CI=[{primary['ci_lower']:.4f}, {primary['ci_upper']:.4f}] "
        f"decision={primary['decision']} ({primary['subcase']})"
    )
    if expansion_required:
        print(f"SEED EXPANSION REQUIRED — {expansion_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
