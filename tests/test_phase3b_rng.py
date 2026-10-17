"""Unit tests for the Milestone 3B RNG namespace separation and reproducibility helpers."""

import random

import numpy as np
import pytest
import torch

from re_reason.utils.rng_namespaces import (
    ROLLOUT_PARTITION_OFFSETS,
    ROLLOUT_SEED_BASE,
    TRAINING_SEED_OFFSET,
    configure_training_rng,
    make_rollout_generator,
    rollout_seed,
    shuffled_batch_indices,
)


def test_training_rng_is_reproducible_and_seed_paired() -> None:
    first = configure_training_rng(7)
    torch_first = torch.randn(6)
    numpy_first = first.numpy_rng.random(6)
    python_first = random.random()

    second = configure_training_rng(7)
    torch_second = torch.randn(6)
    numpy_second = second.numpy_rng.random(6)
    python_second = random.random()

    assert first.model_seed == 7
    assert first.python_seed == TRAINING_SEED_OFFSET + 7
    assert torch.equal(torch_first, torch_second)
    np.testing.assert_array_equal(numpy_first, numpy_second)
    assert python_first == python_second

    configure_training_rng(8)
    assert not torch.equal(torch_first, torch.randn(6))

    with pytest.raises(ValueError):
        configure_training_rng(0)


def test_batch_generators_are_reproducible_and_seed_dependent() -> None:
    first = shuffled_batch_indices(20, 5, configure_training_rng(7).batch_generator)
    second = shuffled_batch_indices(20, 5, configure_training_rng(7).batch_generator)
    third = shuffled_batch_indices(20, 5, configure_training_rng(8).batch_generator)

    assert first == second
    assert first != third


def test_rollout_seed_formula_and_partition_offsets() -> None:
    assert ROLLOUT_SEED_BASE == 500000
    assert ROLLOUT_PARTITION_OFFSETS == {"validation": 0, "test": 200000}

    assert rollout_seed(3, 7, 1, "validation") == 500000 + 30000 + 70 + 1
    assert rollout_seed(3, 7, 2, "validation") == 500000 + 30000 + 70 + 2
    assert rollout_seed(3, 7, 1, "test") == 500000 + 30000 + 200000 + 70 + 1
    assert rollout_seed(4, 0, 1, "validation") - rollout_seed(3, 0, 1, "validation") == 10000
    assert rollout_seed(3, 8, 1, "validation") - rollout_seed(3, 7, 1, "validation") == 10
    assert rollout_seed(3, 7, 2, "validation") - rollout_seed(3, 7, 1, "validation") == 1

    with pytest.raises(ValueError):
        rollout_seed(0, 0, 1, "validation")
    with pytest.raises(ValueError):
        rollout_seed(1, -1, 1, "validation")
    with pytest.raises(ValueError):
        rollout_seed(1, 0, 3, "validation")
    with pytest.raises(ValueError):
        rollout_seed(1, 0, 1, "train")


def test_rollout_generators_are_isolated_and_reproducible() -> None:
    first = make_rollout_generator(123)
    second = make_rollout_generator(123)
    generated = [torch.rand(1, generator=first) for _ in range(5)]
    repeated = [torch.rand(1, generator=second) for _ in range(5)]
    assert all(torch.equal(left, right) for left, right in zip(generated, repeated))

    torch.manual_seed(999)
    isolated = make_rollout_generator(123)
    assert torch.equal(torch.rand(1, generator=isolated), generated[0])

    different = make_rollout_generator(124)
    assert not torch.equal(torch.rand(1, generator=different), generated[0])

    with pytest.raises(ValueError):
        make_rollout_generator(-1)


def test_batch_indices_cover_all_items_without_duplicates() -> None:
    generator = torch.Generator().manual_seed(5)
    batches = shuffled_batch_indices(7, 3, generator)
    assert [len(batch) for batch in batches] == [3, 3]
    flattened = [index for batch in batches for index in batch]
    assert len(flattened) == 6
    assert len(set(flattened)) == 6

    generator = torch.Generator().manual_seed(5)
    with_trailing = shuffled_batch_indices(7, 3, generator, drop_last=False)
    assert [len(batch) for batch in with_trailing] == [3, 3, 1]
    all_indices = [index for batch in with_trailing for index in batch]
    assert sorted(all_indices) == list(range(7))
    assert len(set(all_indices)) == 7

    with pytest.raises(ValueError):
        shuffled_batch_indices(7, 0, torch.Generator())
    with pytest.raises(TypeError):
        shuffled_batch_indices(7, 3, object())
