"""Independent, reproducible RNG namespaces for Milestone 3B training and rollout sampling.

Implements the pre-3B protocol's namespace separation without cross-subsystem
coupling:

* **Graph generation (``RNG_GRAPH``)** uses per-call ``np.random.default_rng(seed)``
  streams inside the pathfinding benchmark; nothing in this module advances them.
* **Training (``RNG_TRAIN``)** is paired across comparators by model seed ``s``:
  ``torch.manual_seed(s)`` for weight initialization,
  ``np.random.default_rng(100000 + s)`` for NumPy-side draws, and
  ``random.seed(100000 + s)`` for the Python standard library. Batch ordering uses
  an isolated ``torch.Generator`` seeded with ``s``.
* **Stochastic rollout sampling (``RNG_EVAL_SAMPLE``)** uses isolated
  ``torch.Generator`` instances seeded by the partition-scoped formula
  ``500000 + s * 10000 + part_offset + i * 10 + traj_id`` with
  ``part_offset = 0`` for ``validation`` and ``200000`` for ``test``. These
  generators never advance or perturb the training stream.

All helpers validate their inputs explicitly and are deterministic for identical
arguments.
"""

from dataclasses import dataclass
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch


TRAINING_SEED_OFFSET = 100000
ROLLOUT_SEED_BASE = 500000
ROLLOUT_PARTITION_OFFSETS = {"validation": 0, "test": 200000}
ROLLOUT_TRAJECTORY_IDS = (1, 2)


@dataclass(frozen=True, eq=False)
class TrainingRng:
    """Seeds and isolated generators for one paired training seed.

    Attributes:
        model_seed: Paired model seed ``s``.
        python_seed: Seed applied to the Python standard library
            (``100000 + s``).
        numpy_rng: NumPy generator seeded with ``100000 + s``.
        batch_generator: Isolated ``torch.Generator`` seeded with ``s`` for
            deterministic batch ordering.
    """

    model_seed: int
    python_seed: int
    numpy_rng: np.random.Generator
    batch_generator: torch.Generator


def _require_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def configure_training_rng(model_seed: int) -> TrainingRng:
    """Seeds the paired-training namespaces and returns isolated generators.

    Applies the protocol's training seeds: ``torch.manual_seed(model_seed)`` for
    weight initialization, ``random.seed(100000 + model_seed)`` for Python, and a
    NumPy generator ``np.random.default_rng(100000 + model_seed)``. Batch ordering
    uses a separate ``torch.Generator`` seeded with ``model_seed`` so the global
    PyTorch stream is never reused for shuffling.
    """
    _require_int("model_seed", model_seed, 1)
    torch.manual_seed(model_seed)
    python_seed = TRAINING_SEED_OFFSET + model_seed
    random.seed(python_seed)
    return TrainingRng(
        model_seed=model_seed,
        python_seed=python_seed,
        numpy_rng=np.random.default_rng(python_seed),
        batch_generator=torch.Generator().manual_seed(model_seed),
    )


def rollout_seed(
    model_seed: int,
    episode_index: int,
    trajectory_id: int,
    partition: str,
) -> int:
    """Computes the protocol's partition-scoped rollout-sampling seed.

    ``seed_rollout(s, i, traj, part) = 500000 + s * 10000 + part_offset + i * 10 + traj``
    with ``part_offset = 0`` for ``validation`` and ``200000`` for ``test``.
    """
    _require_int("model_seed", model_seed, 1)
    _require_int("episode_index", episode_index, 0)
    if trajectory_id not in ROLLOUT_TRAJECTORY_IDS:
        raise ValueError(
            f"trajectory_id must be one of {ROLLOUT_TRAJECTORY_IDS}, got {trajectory_id!r}"
        )
    if partition not in ROLLOUT_PARTITION_OFFSETS:
        raise ValueError(
            f"partition must be one of {sorted(ROLLOUT_PARTITION_OFFSETS)}, got {partition!r}"
        )
    return (
        ROLLOUT_SEED_BASE
        + model_seed * 10000
        + ROLLOUT_PARTITION_OFFSETS[partition]
        + episode_index * 10
        + trajectory_id
    )


def make_rollout_generator(seed: int) -> torch.Generator:
    """Creates an isolated CPU ``torch.Generator`` seeded with ``seed``."""
    _require_int("seed", seed, 0)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def shuffled_batch_indices(
    num_items: int,
    batch_size: int,
    generator: torch.Generator,
    drop_last: bool = True,
) -> List[Tuple[int, ...]]:
    """Returns deterministic shuffled batch index tuples for one epoch.

    The permutation is drawn from the supplied isolated generator, so identical
    generator seeds reproduce identical batch orders. With ``drop_last=True`` a
    trailing partial batch is omitted; otherwise it is retained.
    """
    _require_int("num_items", num_items, 0)
    _require_int("batch_size", batch_size, 1)
    if not isinstance(generator, torch.Generator):
        raise TypeError(f"generator must be a torch.Generator, got {type(generator).__name__}")

    order = torch.randperm(num_items, generator=generator).tolist()
    batches: List[Tuple[int, ...]] = []
    for start in range(0, num_items, batch_size):
        batch = tuple(order[start : start + batch_size])
        if len(batch) < batch_size and drop_last:
            break
        batches.append(batch)
    return batches
