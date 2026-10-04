"""Strictly separated and sequence-hashed dataset splits for train, validation, and test.

Ensures zero data leakage across splits via cryptographic SHA-256 sequence hashing.
"""

import hashlib
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch

from re_reason.benchmarks.base import TaskBatch
from re_reason.benchmarks.register_swap import RegisterTrackingTask


def hash_sequence(inputs_arr: np.ndarray) -> str:
    """Computes a canonical SHA-256 hash of a procedural operation sequence."""
    hasher = hashlib.sha256()
    hasher.update(inputs_arr.tobytes())
    return hasher.hexdigest()


class SplitDataset:
    """Fixed-size procedural dataset split containing pre-generated, verified unique episodes."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, step_targets: torch.Tensor, hashes: List[str]) -> None:
        self.inputs = inputs
        self.targets = targets
        self.step_targets = step_targets
        self.hashes = hashes

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def get_batch(self, batch_size: int, idx: int = 0) -> TaskBatch:
        """Retrieves a slice or cyclic batch from the split."""
        n = len(self)
        start = (idx * batch_size) % n
        indices = [(start + i) % n for i in range(batch_size)]

        sub_inputs = self.inputs[indices]
        sub_targets = self.targets[indices]
        sub_step_targets = self.step_targets[indices]

        return TaskBatch(
            inputs=sub_inputs,
            targets=sub_targets,
            metadata={"step_targets": sub_step_targets},
        )


class DisjointSplitManager:
    """Manages strictly partitioned Train, Validation, and Test datasets with collision verification."""

    def __init__(
        self,
        task: RegisterTrackingTask,
        train_size: int = 8000,
        val_size: int = 1500,
        test_size: int = 1500,
        base_seed: int = 42,
    ) -> None:
        self.task = task
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # Generate each split with non-overlapping seed spaces
        self.train_set, train_hashes = self._generate_unique_split(train_size, seed=base_seed + 1000)
        self.val_set, val_hashes = self._generate_unique_split(val_size, seed=base_seed + 50000)
        self.test_set, test_hashes = self._generate_unique_split(test_size, seed=base_seed + 90000)

        # Cryptographic leak verification
        train_hash_set = set(train_hashes)
        val_hash_set = set(val_hashes)
        test_hash_set = set(test_hashes)

        train_val_overlap = train_hash_set.intersection(val_hash_set)
        train_test_overlap = train_hash_set.intersection(test_hash_set)
        val_test_overlap = val_hash_set.intersection(test_hash_set)

        if train_val_overlap:
            raise RuntimeError(f"Data leakage detected! {len(train_val_overlap)} samples overlap between train and val.")
        if train_test_overlap:
            raise RuntimeError(f"Data leakage detected! {len(train_test_overlap)} samples overlap between train and test.")
        if val_test_overlap:
            raise RuntimeError(f"Data leakage detected! {len(val_test_overlap)} samples overlap between val and test.")

        self.train_hashes = train_hashes
        self.val_hashes = val_hashes
        self.test_hashes = test_hashes

    def _generate_unique_split(self, size: int, seed: int) -> Tuple[SplitDataset, List[str]]:
        rng = np.random.default_rng(seed)
        seen_hashes: Set[str] = set()
        hashes_list: List[str] = []

        inputs_list: List[np.ndarray] = []
        step_targets_list: List[np.ndarray] = []
        final_targets_list: List[np.ndarray] = []

        while len(inputs_list) < size:
            inp, step_tgt, final_tgt = self.task.generate_single_sequence(rng)
            h = hash_sequence(inp)
            if h not in seen_hashes:
                seen_hashes.add(h)
                hashes_list.append(h)
                inputs_list.append(inp)
                step_targets_list.append(step_tgt)
                final_targets_list.append(final_tgt)

        inputs_t = torch.from_numpy(np.stack(inputs_list, axis=0)).float()
        step_targets_t = torch.from_numpy(np.stack(step_targets_list, axis=0)).long()
        final_targets_t = torch.from_numpy(np.stack(final_targets_list, axis=0)).long()

        return SplitDataset(inputs_t, final_targets_t, step_targets_t, hashes_list), hashes_list
