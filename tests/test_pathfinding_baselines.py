"""Unit tests for the Milestone 3B comparator baselines: architecture, masking, rollouts, and selection."""

import pytest
import torch

from re_reason.benchmarks.pathfinding_benchmark import (
    PathfindingEpisode,
    PathfindingState,
)
from re_reason.core.pathfinding_baselines import (
    ROLLOUT_TEMPERATURE,
    SINGLE_TRAJECTORY_TRANSITIONS,
    TWO_ROLLOUT_HORIZONS,
    ActionMaskingRecurrentBaseline,
    PathfindingRecurrentBaseline,
    TrajectoryResult,
    TwoRolloutRecurrentBaseline,
    TwoTrajectoryResult,
    UnGatedRecurrentBaseline,
    rollout_action_masked,
    rollout_two_trajectory,
    rollout_un_gated,
    select_trajectory,
)
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher


def _straight_episode() -> PathfindingEpisode:
    return PathfindingEpisode(
        num_nodes=12,
        start=0,
        goal=2,
        edges=frozenset({(0, 1), (1, 2)}),
    )


def _branching_episode() -> PathfindingEpisode:
    return PathfindingEpisode(
        num_nodes=12,
        start=0,
        goal=2,
        edges=frozenset({(0, 1), (1, 2), (0, 3)}),
    )


def _stuck_episode() -> PathfindingEpisode:
    return PathfindingEpisode(
        num_nodes=12,
        start=0,
        goal=5,
        edges=frozenset({(0, 1)}),
    )


def _no_move_episode() -> PathfindingEpisode:
    return PathfindingEpisode(
        num_nodes=12,
        start=0,
        goal=5,
        edges=frozenset(),
    )


def _unreachable_cycle_episode() -> PathfindingEpisode:
    return PathfindingEpisode(
        num_nodes=12,
        start=0,
        goal=2,
        edges=frozenset({(0, 1), (1, 0)}),
    )


def _force_policy_bias(model: PathfindingRecurrentBaseline, biases: dict) -> None:
    with torch.no_grad():
        model.brancher.policy_head.weight.zero_()
        model.brancher.policy_head.bias.zero_()
        for action, value in biases.items():
            model.brancher.policy_head.bias[action] = value


def _trajectory(reached: bool, transitions: int) -> TrajectoryResult:
    return TrajectoryResult(
        reached_goal=reached,
        actions=tuple(0 for _ in range(transitions)),
        final_environment_state=PathfindingState(current=0),
    )


# ---------------------------------------------------------------------------
# Shared architecture and tensor contracts
# ---------------------------------------------------------------------------

def test_baselines_share_the_exact_parameter_budget() -> None:
    for baseline_class in (
        UnGatedRecurrentBaseline,
        ActionMaskingRecurrentBaseline,
        TwoRolloutRecurrentBaseline,
    ):
        model = baseline_class()
        assert isinstance(model, PathfindingRecurrentBaseline)
        assert model.count_parameters() == 171660
        assert isinstance(model.brancher, ActionConditionedBrancher)
        assert model.state_dim == 256
        assert model.num_nodes == 12
        assert sum(p.numel() for p in model.brancher.node_embeddings.parameters()) == 384
        assert sum(p.numel() for p in model.brancher.observation_encoder.parameters()) == 61824
        assert sum(p.numel() for p in model.brancher.policy_head.parameters()) == 6156
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 171660

    shared = ActionConditionedBrancher()
    wrapper = UnGatedRecurrentBaseline(brancher=shared)
    assert wrapper.brancher is shared
    with pytest.raises(TypeError):
        UnGatedRecurrentBaseline(brancher=object())


def test_baseline_forward_shapes_and_state_dimensions() -> None:
    model = UnGatedRecurrentBaseline()
    episode = _straight_episode()
    observation = episode.observe(episode.initial_state())

    encoded = model.encode_observation(observation)
    assert encoded.shape == (256,)

    state = model.initial_state(1)
    assert state.shape == (1, 256)
    assert state.dtype == model.brancher.node_embeddings.weight.dtype
    assert state.device == model.brancher.node_embeddings.weight.device
    assert model.initial_state(3).shape == (3, 256)

    logits = model.policy_logits(state, encoded.unsqueeze(0))
    assert logits.shape == (1, 12)

    advanced = model.advance_state(state, encoded.unsqueeze(0), 1)
    assert advanced.shape == (1, 256)


def test_advance_states_matches_the_shared_candidate_core() -> None:
    brancher = ActionConditionedBrancher()
    states = torch.randn(2, brancher.state_dim)
    encoded = torch.randn(2, brancher.state_dim)
    action_ids = torch.tensor([1, 3], dtype=torch.long)

    advanced = brancher.advance_states(states, encoded, action_ids)
    candidates = brancher.evaluate_candidates(states, encoded, (1, 3))

    assert advanced.shape == (2, 256)
    assert torch.allclose(advanced[0], candidates[0, 0], rtol=1e-5, atol=1e-6)
    assert torch.allclose(advanced[1], candidates[1, 1], rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError):
        brancher.advance_states(states, encoded, action_ids.to(torch.float32))
    with pytest.raises(ValueError):
        brancher.advance_states(states, encoded, torch.tensor([1]))
    with pytest.raises(ValueError):
        brancher.advance_states(states, encoded, torch.tensor([1, 12]))


# ---------------------------------------------------------------------------
# Action selection and masking
# ---------------------------------------------------------------------------

def test_un_gated_selection_ignores_legality() -> None:
    model = UnGatedRecurrentBaseline()
    episode = _branching_episode()
    observation = episode.observe(episode.initial_state())
    _force_policy_bias(model, {5: 100.0, 3: 5.0, 1: 1.0})

    encoded = model.encode_observation(observation).unsqueeze(0)
    logits = model.policy_logits(model.initial_state(1), encoded)

    assert 5 not in observation.allowed_moves
    assert model.select_action(logits) == 5


def test_action_masking_restricts_selection_to_legal_moves() -> None:
    model = ActionMaskingRecurrentBaseline()
    episode = _branching_episode()
    observation = episode.observe(episode.initial_state())
    _force_policy_bias(model, {5: 1000.0, 3: 10.0, 1: 1.0})

    encoded = model.encode_observation(observation).unsqueeze(0)
    logits = model.policy_logits(model.initial_state(1), encoded)

    assert model.select_action(logits, observation) == 3

    no_move_episode = _no_move_episode()
    blocked = no_move_episode.observe(no_move_episode.initial_state())
    blocked_logits = torch.zeros(1, 12)
    assert blocked.allowed_moves == ()
    assert model.select_action(blocked_logits, blocked) is None
    with pytest.raises(TypeError):
        model.select_action(blocked_logits, "observation")


def test_two_rollout_sampling_is_legal_and_reproducible() -> None:
    model = TwoRolloutRecurrentBaseline()
    episode = _branching_episode()
    observation = episode.observe(episode.initial_state())
    _force_policy_bias(model, {5: 1000.0, 1: 2.0, 3: 0.0})
    encoded = model.encode_observation(observation).unsqueeze(0)
    logits = model.policy_logits(model.initial_state(1), encoded)

    assert ROLLOUT_TEMPERATURE == 1.0
    assert TWO_ROLLOUT_HORIZONS == (26, 27)
    assert SINGLE_TRAJECTORY_TRANSITIONS == 24

    generator_a = torch.Generator().manual_seed(1234)
    generator_b = torch.Generator().manual_seed(1234)
    generator_c = torch.Generator().manual_seed(1235)
    sequence_a = [model.sample_action(logits, observation, generator_a) for _ in range(20)]
    sequence_b = [model.sample_action(logits, observation, generator_b) for _ in range(20)]
    sequence_c = [model.sample_action(logits, observation, generator_c) for _ in range(20)]

    assert all(1 <= action <= 3 and action != 2 for action in sequence_a)
    assert sequence_a == sequence_b
    assert sequence_a != sequence_c
    single = model.sample_action(logits, observation, torch.Generator().manual_seed(1234))
    assert single == sequence_a[0]

    no_move_episode = _no_move_episode()
    static = no_move_episode.observe(no_move_episode.initial_state())
    assert model.sample_action(torch.zeros(1, 12), static, generator_a) is None


# ---------------------------------------------------------------------------
# Trajectory selection and result contracts
# ---------------------------------------------------------------------------

def test_select_trajectory_protocol_rule() -> None:
    assert select_trajectory(_trajectory(True, 3), _trajectory(True, 5)) == 1
    assert select_trajectory(_trajectory(True, 5), _trajectory(True, 3)) == 2
    assert select_trajectory(_trajectory(True, 4), _trajectory(True, 4)) == 1
    assert select_trajectory(_trajectory(True, 2), _trajectory(False, 9)) == 1
    assert select_trajectory(_trajectory(False, 9), _trajectory(True, 2)) == 2
    assert select_trajectory(_trajectory(False, 2), _trajectory(False, 3)) is None


def test_two_trajectory_result_consistency_and_properties() -> None:
    first = _trajectory(True, 3)
    second = _trajectory(True, 5)
    selected = TwoTrajectoryResult(first, second, selected_trajectory=1)
    assert selected.reached_goal
    assert selected.selected_result is first
    assert selected.selected_transitions == 3

    tie = TwoTrajectoryResult(_trajectory(True, 4), _trajectory(True, 4), selected_trajectory=1)
    assert tie.selected_transitions == 4

    failure = TwoTrajectoryResult(_trajectory(False, 1), _trajectory(False, 2), selected_trajectory=None)
    assert not failure.reached_goal
    assert failure.selected_result is None
    assert failure.selected_transitions == 0

    with pytest.raises(ValueError):
        TwoTrajectoryResult(first, second, selected_trajectory=2)
    with pytest.raises(ValueError):
        TwoTrajectoryResult(_trajectory(False, 1), second, selected_trajectory=1)


# ---------------------------------------------------------------------------
# Episode rollouts
# ---------------------------------------------------------------------------

def test_un_gated_rollout_commits_latent_state_unconditionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = UnGatedRecurrentBaseline()
    episode = _straight_episode()
    _force_policy_bias(model, {5: 100.0})
    advance_calls = []
    original_advance = model.advance_state

    def counting_advance(state, encoded, action_id):
        advance_calls.append(action_id)
        return original_advance(state, encoded, action_id)

    monkeypatch.setattr(model, "advance_state", counting_advance)

    result = rollout_un_gated(model, episode, max_transitions=4)

    assert result.reached_goal is False
    assert result.actions == (5, 5, 5, 5)
    assert result.transitions_used == 4
    assert result.invalid_attempts == 4
    assert result.final_environment_state.current == episode.start
    assert result.final_environment_state.transitions == 0
    assert advance_calls == [5, 5, 5, 5]


def test_action_masked_rollout_advances_along_legal_path() -> None:
    model = ActionMaskingRecurrentBaseline()
    episode = _straight_episode()
    _force_policy_bias(model, {5: 1000.0, 1: 10.0, 2: 9.0})

    result = rollout_action_masked(model, episode, max_transitions=10)

    assert result.reached_goal
    assert result.actions == (1, 2)
    assert result.transitions_used == 2
    assert result.invalid_attempts == 0
    assert result.final_environment_state.current == episode.goal
    assert result.final_environment_state.transitions == 2


def test_action_masked_rollout_stops_at_dead_end() -> None:
    model = ActionMaskingRecurrentBaseline()
    episode = _stuck_episode()

    result = rollout_action_masked(model, episode, max_transitions=10)

    assert result.reached_goal is False
    assert result.actions == (1,)
    assert result.final_environment_state.current == 1


def test_two_rollout_static_horizons_budgets_and_reproducibility() -> None:
    model = TwoRolloutRecurrentBaseline()
    cycling = _unreachable_cycle_episode()

    first = rollout_two_trajectory(model, cycling, model_seed=3, episode_index=7, partition="validation")
    second = rollout_two_trajectory(model, cycling, model_seed=3, episode_index=7, partition="validation")

    assert first == second
    assert first.selected_trajectory is None
    assert first.reached_goal is False
    assert first.trajectory_1.transitions_used == TWO_ROLLOUT_HORIZONS[0]
    assert first.trajectory_2.transitions_used == TWO_ROLLOUT_HORIZONS[1]
    assert first.trajectory_1.actions != first.trajectory_2.actions


def test_two_rollout_prefers_shorter_or_first_on_tie() -> None:
    model = TwoRolloutRecurrentBaseline()
    episode = _straight_episode()
    _force_policy_bias(model, {1: 10.0, 2: 9.0})

    result = rollout_two_trajectory(
        model, episode, model_seed=1, episode_index=0, partition="validation"
    )

    assert result.reached_goal
    assert result.trajectory_1.reached_goal and result.trajectory_2.reached_goal
    assert result.trajectory_1.transitions_used == 2
    assert result.trajectory_2.transitions_used == 2
    assert result.selected_trajectory == 1
    assert result.selected_transitions == 2


def test_two_rollout_does_not_touch_global_torch_rng() -> None:
    model = TwoRolloutRecurrentBaseline()
    episode = _straight_episode()

    torch.manual_seed(4321)
    expected_after = torch.rand(6)

    torch.manual_seed(4321)
    rollout_two_trajectory(model, episode, model_seed=2, episode_index=1, partition="validation")
    actual_after = torch.rand(6)

    assert torch.equal(expected_after, actual_after)

    torch.manual_seed(999)
    first = rollout_two_trajectory(model, episode, model_seed=2, episode_index=1, partition="validation")
    torch.manual_seed(1)
    second = rollout_two_trajectory(model, episode, model_seed=2, episode_index=1, partition="validation")
    assert first == second


def test_rollout_argument_validation() -> None:
    straight_model = UnGatedRecurrentBaseline()
    two_rollout_model = TwoRolloutRecurrentBaseline()
    episode = _straight_episode()

    with pytest.raises(TypeError):
        rollout_un_gated(two_rollout_model, episode)
    with pytest.raises(TypeError):
        rollout_action_masked(straight_model, episode)
    with pytest.raises(TypeError):
        rollout_un_gated(straight_model, "episode")
    with pytest.raises(ValueError):
        rollout_un_gated(straight_model, episode, max_transitions=0)
    with pytest.raises(TypeError):
        rollout_two_trajectory(straight_model, episode, model_seed=1, episode_index=0, partition="validation")
    with pytest.raises(ValueError):
        rollout_two_trajectory(two_rollout_model, episode, model_seed=0, episode_index=0, partition="validation")
    with pytest.raises(ValueError):
        rollout_two_trajectory(two_rollout_model, episode, model_seed=1, episode_index=0, partition="train")
