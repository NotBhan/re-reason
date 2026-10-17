"""Benchmarks package contracts and benchmark tasks."""

from re_reason.benchmarks.base import BaseTask, TaskBatch
from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.benchmarks.register_machine_benchmark import (
    Phase16RegisterTask,
    Phase16SplitManager,
    Stage1PairPartition,
)
from re_reason.benchmarks.pathfinding_benchmark import (
    PathValidation,
    PathfindingAction,
    PathfindingEpisode,
    PathfindingGeneratorConfig,
    PathfindingObservation,
    PathfindingSolution,
    PathfindingSplitSet,
    PathfindingState,
    generate_pathfinding_episode,
    generate_pathfinding_partition,
    generate_pathfinding_splits,
    goal_reachable_from,
    solve_shortest_path,
    validate_path,
)

__all__ = [
    "BaseTask",
    "TaskBatch",
    "RegisterTrackingTask",
    "Phase16RegisterTask",
    "Stage1PairPartition",
    "Phase16SplitManager",
    "PathValidation",
    "PathfindingAction",
    "PathfindingEpisode",
    "PathfindingGeneratorConfig",
    "PathfindingObservation",
    "PathfindingSolution",
    "PathfindingSplitSet",
    "PathfindingState",
    "generate_pathfinding_episode",
    "generate_pathfinding_partition",
    "generate_pathfinding_splits",
    "goal_reachable_from",
    "solve_shortest_path",
    "validate_path",
]
