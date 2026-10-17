"""Utility modules for seeding, RNG namespaces, and diagnostics."""

from re_reason.utils.memory import (
    check_monotonic_growth,
    get_memory_diagnostics,
    inspect_tensor_retention,
)
from re_reason.utils.seeding import set_seed
from re_reason.utils.rng_namespaces import (
    ROLLOUT_PARTITION_OFFSETS,
    ROLLOUT_SEED_BASE,
    ROLLOUT_TRAJECTORY_IDS,
    TRAINING_SEED_OFFSET,
    TrainingRng,
    configure_training_rng,
    make_rollout_generator,
    rollout_seed,
    shuffled_batch_indices,
)

__all__ = [
    "set_seed",
    "inspect_tensor_retention",
    "get_memory_diagnostics",
    "check_monotonic_growth",
    "TRAINING_SEED_OFFSET",
    "ROLLOUT_SEED_BASE",
    "ROLLOUT_PARTITION_OFFSETS",
    "ROLLOUT_TRAJECTORY_IDS",
    "TrainingRng",
    "configure_training_rng",
    "rollout_seed",
    "make_rollout_generator",
    "shuffled_batch_indices",
]
