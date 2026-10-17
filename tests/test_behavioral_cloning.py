"""Unit tests for behavioral-cloning demonstrations, transition-weighted loss, masking, and gradient contracts."""

import pytest
import torch
import torch.nn.functional as F

from re_reason.benchmarks.pathfinding_benchmark import PathfindingEpisode, PathfindingState
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher
from re_reason.training.behavioral_cloning import (
    BehavioralCloningLoss,
    Demonstration,
    build_demonstration,
    build_demonstrations,
    iter_training_batches,
    load_behavioral_cloning_demonstrations,
    teacher_forced_batch_loss,
)
from re_reason.utils.rng_namespaces import configure_training_rng


def _line_episode(length: int) -> PathfindingEpisode:
    edges = frozenset((index, index + 1) for index in range(length))
    return PathfindingEpisode(num_nodes=12, start=0, goal=length, edges=edges)


def _unsolved_episode() -> PathfindingEpisode:
    return PathfindingEpisode(
        num_nodes=12, start=0, goal=5, edges=frozenset({(0, 1)})
    )


def _reference_loss(
    brancher: ActionConditionedBrancher, demonstrations
):
    reference = brancher.node_embeddings.weight
    total = None
    count = 0
    for demonstration in demonstrations:
        state = torch.zeros(
            (1, brancher.state_dim), dtype=reference.dtype, device=reference.device
        )
        for observation, action in zip(demonstration.observations, demonstration.actions):
            encoded = brancher.encode_observation(observation).unsqueeze(0)
            logits = brancher.policy_logits(state, encoded)
            step_loss = F.cross_entropy(
                logits, torch.tensor([action], dtype=torch.long, device=logits.device)
            )
            total = step_loss if total is None else total + step_loss
            count += 1
            state = brancher.advance_states(
                state,
                encoded,
                torch.tensor([action], dtype=torch.long, device=logits.device),
            )
    return total / count, count


# ---------------------------------------------------------------------------
# Demonstration construction
# ---------------------------------------------------------------------------

def test_build_demonstration_matches_the_bfs_path() -> None:
    episode = _line_episode(3)
    demonstration = build_demonstration(episode)

    assert demonstration is not None
    assert demonstration.length == 3
    assert demonstration.actions == (1, 2, 3)
    assert tuple(observation.current for observation in demonstration.observations) == (0, 1, 2)
    for observation in demonstration.observations:
        assert observation.goal == episode.goal
    for index, observation in enumerate(demonstration.observations):
        expected = episode.observe(PathfindingState(current=index, transitions=index))
        assert observation == expected


def test_unsolved_and_zero_transition_episodes_are_skipped_defensively() -> None:
    solved = _line_episode(2)
    unsolved = _unsolved_episode()
    zero_length = PathfindingEpisode(
        num_nodes=12, start=3, goal=3, edges=frozenset()
    )

    assert build_demonstration(unsolved) is None
    assert build_demonstration(zero_length) is None
    demonstrations = build_demonstrations([solved, unsolved, zero_length])
    assert len(demonstrations) == 1
    assert demonstrations[0].actions == (1, 2)


# ---------------------------------------------------------------------------
# Transition-weighted loss, padding, and gradients
# ---------------------------------------------------------------------------

def test_transition_weighted_loss_matches_independent_reference() -> None:
    brancher = ActionConditionedBrancher()
    demonstrations = [
        build_demonstration(_line_episode(2)),
        build_demonstration(_line_episode(4)),
    ]
    assert all(demonstration is not None for demonstration in demonstrations)

    report = teacher_forced_batch_loss(brancher, demonstrations)
    reference_loss, reference_count = _reference_loss(brancher, demonstrations)

    assert isinstance(report, BehavioralCloningLoss)
    assert report.batch_size == 2
    assert report.max_length == 4
    assert report.total_transitions == 6
    assert reference_count == 6
    assert torch.allclose(report.loss, reference_loss, rtol=1e-4, atol=1e-6)


def test_padding_positions_contribute_zero_gradient() -> None:
    brancher = ActionConditionedBrancher()
    demonstrations = [
        build_demonstration(_line_episode(2)),
        build_demonstration(_line_episode(4)),
    ]

    report = teacher_forced_batch_loss(brancher, demonstrations)
    reference_loss, _ = _reference_loss(brancher, demonstrations)

    parameters = [parameter for _, parameter in brancher.named_parameters()]
    batch_gradients = torch.autograd.grad(
        report.loss, parameters, allow_unused=True
    )
    reference_gradients = torch.autograd.grad(
        reference_loss, parameters, allow_unused=True
    )
    for batch_gradient, reference_gradient in zip(batch_gradients, reference_gradients):
        assert batch_gradient is not None and reference_gradient is not None
        assert torch.allclose(batch_gradient, reference_gradient, rtol=1e-4, atol=1e-6)


def test_downstream_policy_losses_reach_the_branch_core() -> None:
    brancher = ActionConditionedBrancher()
    demonstration = build_demonstration(_line_episode(3))
    assert demonstration is not None

    report = teacher_forced_batch_loss(brancher, [demonstration])
    report.loss.backward()

    for name, parameter in brancher.named_parameters():
        assert parameter.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"

    branch_squared_norm = sum(
        float((parameter.grad ** 2).sum().item())
        for name, parameter in brancher.named_parameters()
        if name.startswith(("branch_input", "branch_output", "branch_norm"))
    )
    assert branch_squared_norm > 0.0


def test_terminal_step_has_no_downstream_policy_gradient() -> None:
    brancher = ActionConditionedBrancher()
    demonstration = build_demonstration(_line_episode(1))
    assert demonstration is not None and demonstration.length == 1

    report = teacher_forced_batch_loss(brancher, [demonstration])
    report.loss.backward()

    for name, parameter in brancher.named_parameters():
        if name.startswith(("branch_input", "branch_output", "branch_norm")):
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, (
                f"terminal-step branch-core parameter {name} unexpectedly received gradient"
            )
        else:
            assert parameter.grad is not None, f"missing gradient for {name}"
            assert torch.count_nonzero(parameter.grad) > 0, f"zero gradient for {name}"


def test_loss_rejects_invalid_inputs() -> None:
    brancher = ActionConditionedBrancher()
    demonstration = build_demonstration(_line_episode(2))
    assert demonstration is not None

    with pytest.raises(ValueError):
        teacher_forced_batch_loss(brancher, [])
    with pytest.raises(TypeError):
        teacher_forced_batch_loss(object(), [demonstration])
    with pytest.raises(TypeError):
        teacher_forced_batch_loss(brancher, "demonstrations")
    with pytest.raises(TypeError):
        teacher_forced_batch_loss(brancher, [object()])
    with pytest.raises(ValueError):
        Demonstration(observations=(), actions=())
    with pytest.raises(ValueError):
        Demonstration(observations=demonstration.observations, actions=(1,))


# ---------------------------------------------------------------------------
# Training batches and partition-restricted loading
# ---------------------------------------------------------------------------

def test_training_batches_are_reproducible_and_complete() -> None:
    episodes = [_line_episode(length) for length in range(1, 8)]
    demonstrations = build_demonstrations(episodes)
    assert len(demonstrations) == 7
    index_of = {id(demonstration): index for index, demonstration in enumerate(demonstrations)}

    first_rng = configure_training_rng(11)
    first_order = [
        tuple(index_of[id(demonstration)] for demonstration in batch)
        for batch in iter_training_batches(demonstrations, 3, first_rng.batch_generator)
    ]
    second_rng = configure_training_rng(11)
    second_order = [
        tuple(index_of[id(demonstration)] for demonstration in batch)
        for batch in iter_training_batches(demonstrations, 3, second_rng.batch_generator)
    ]

    assert first_order == second_order
    flattened = [index for batch in first_order for index in batch]
    assert len(flattened) == 6
    assert len(set(flattened)) == 6

    different_rng = configure_training_rng(12)
    different_order = [
        tuple(index_of[id(demonstration)] for demonstration in batch)
        for batch in iter_training_batches(demonstrations, 3, different_rng.batch_generator)
    ]
    assert different_order != first_order

    trailing_rng = configure_training_rng(11)
    with_trailing = list(
        iter_training_batches(
            demonstrations, 3, trailing_rng.batch_generator, drop_last=False
        )
    )
    assert len(with_trailing) == 3
    assert len(with_trailing[-1]) == 1
    all_indices = [
        index_of[id(demonstration)]
        for batch in with_trailing
        for demonstration in batch
    ]
    assert sorted(all_indices) == list(range(7))


def test_loader_requests_only_the_train_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = []

    def fake_generate_partition(config, partition, size, base_seed=42):
        recorded.append((partition, size, base_seed))
        return (_line_episode(2),)

    monkeypatch.setattr(
        "re_reason.training.behavioral_cloning.generate_pathfinding_partition",
        fake_generate_partition,
    )

    demonstrations = load_behavioral_cloning_demonstrations(size=3, base_seed=77)

    assert recorded == [("train", 3, 77)]
    assert len(demonstrations) == 1


def test_loader_uses_the_real_train_partition() -> None:
    demonstrations = load_behavioral_cloning_demonstrations(size=3)
    assert len(demonstrations) > 0
    for demonstration in demonstrations:
        assert demonstration.length >= 1
        assert len(demonstration.observations) == len(demonstration.actions)
