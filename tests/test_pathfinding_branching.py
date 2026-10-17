"""Unit tests for the Milestone 3A pathfinding branching architecture and its state/transaction integration."""

import pytest
import torch

from re_reason.benchmarks.pathfinding_benchmark import (
    PathfindingEpisode,
    PathfindingObservation,
)
from re_reason.hypothesis.pathfinding_branching import (
    ActionConditionedBrancher,
    BranchProposal,
    CandidateProposal,
    ObservationEncoder,
)
from re_reason.state.container import LatentState
from re_reason.state.speculative import SpeculativeCandidateSet
from re_reason.verification.transaction_gate import (
    CandidateRecord,
    TransactionCommitGate,
    VerificationOutcome,
)


def _episode(**overrides) -> PathfindingEpisode:
    parameters = dict(
        num_nodes=12,
        start=0,
        goal=11,
        edges=frozenset({(0, 1), (1, 11), (0, 2), (0, 3), (0, 4), (2, 5), (3, 6)}),
    )
    parameters.update(overrides)
    return PathfindingEpisode(**parameters)


def _observation(episode: PathfindingEpisode) -> PathfindingObservation:
    return episode.observe(episode.initial_state())


def _proposal(brancher: ActionConditionedBrancher, episode: PathfindingEpisode) -> BranchProposal:
    container = LatentState(torch.zeros(1, brancher.state_dim))
    with torch.no_grad():
        return brancher.propose_candidates(container.committed_state, _observation(episode))


def _param_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


# ---------------------------------------------------------------------------
# Observation encoder contract
# ---------------------------------------------------------------------------

def test_observation_encoder_dimensions_and_parameter_count() -> None:
    encoder = ObservationEncoder()
    assert encoder.count_parameters() == 61824

    single = encoder(torch.zeros(220))
    batch = encoder(torch.zeros(4, 220))
    sequence = encoder(torch.zeros(2, 5, 220))

    assert single.shape == (256,)
    assert batch.shape == (4, 256)
    assert sequence.shape == (2, 5, 256)


def test_observation_encoder_rejects_malformed_inputs() -> None:
    encoder = ObservationEncoder()
    with pytest.raises(TypeError):
        encoder("observation")
    with pytest.raises(ValueError):
        encoder(torch.zeros(219))
    with pytest.raises(ValueError):
        encoder(torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Frozen architecture and parameter accounting
# ---------------------------------------------------------------------------

def test_total_parameter_count_matches_frozen_budget() -> None:
    brancher = ActionConditionedBrancher()

    assert brancher.count_parameters() == 171660
    assert _param_count(brancher.node_embeddings) == 384
    assert _param_count(brancher.observation_encoder) == 61824
    assert _param_count(brancher.policy_head) == 6156
    branch_core_parameters = sum(
        parameter.numel()
        for name, parameter in brancher.named_parameters()
        if name.startswith(("branch_input", "branch_output", "branch_norm"))
    )
    assert branch_core_parameters == 103296
    assert len(list(brancher.parameters())) == 15


def test_branch_core_is_shared_across_candidates() -> None:
    brancher = ActionConditionedBrancher()
    container_state = torch.zeros(1, brancher.state_dim)
    observation = _observation(_episode())
    recorded_shapes = []

    def hook(module, inputs, output) -> None:
        recorded_shapes.append(tuple(inputs[0].shape))

    handle = brancher.branch_input.register_forward_hook(hook)
    try:
        with torch.no_grad():
            proposal = brancher.propose_candidates(container_state, observation)
    finally:
        handle.remove()

    assert recorded_shapes == [(1, 3, 544)]
    assert proposal.stacked_states().shape == (1, 3, 256)


def test_evaluate_candidates_supports_batches() -> None:
    brancher = ActionConditionedBrancher()
    states = torch.zeros(2, brancher.state_dim)
    encoded = torch.zeros(2, brancher.state_dim)

    with torch.no_grad():
        candidates = brancher.evaluate_candidates(states, encoded, (1, 3))

    assert candidates.shape == (2, 2, 256)
    pointers = {candidates[0, 0].data_ptr(), candidates[0, 1].data_ptr()}
    assert len(pointers) == 2
    second_candidate_before = candidates[0, 1].clone()
    with torch.no_grad():
        candidates[0, 0].add_(1.0)
    assert torch.equal(candidates[0, 1], second_candidate_before)
    assert not torch.equal(candidates[0, 0], candidates[0, 1])


# ---------------------------------------------------------------------------
# Observation features and information boundary
# ---------------------------------------------------------------------------

def test_observation_features_layout_uses_only_the_public_surface() -> None:
    brancher = ActionConditionedBrancher()
    episode = _episode()
    observation = _observation(episode)

    features = brancher.observation_features(observation)

    assert features.shape == (220,)
    assert torch.equal(features[:32], brancher.node_embeddings.weight[observation.current])
    assert torch.equal(features[32:64], brancher.node_embeddings.weight[observation.goal])
    expected_mask = torch.zeros(12)
    expected_mask[list(observation.allowed_moves)] = 1.0
    assert torch.equal(features[64:76], expected_mask)
    expected_adjacency = torch.zeros(12, 12)
    for node, neighbors in enumerate(observation.effective_adjacency):
        for neighbor in neighbors:
            expected_adjacency[node, neighbor] = 1.0
    assert torch.equal(features[76:].reshape(12, 12), expected_adjacency)


def test_observation_features_do_not_leak_hidden_ground_truth() -> None:
    brancher = ActionConditionedBrancher()
    public_edges = frozenset({(0, 1), (1, 11), (0, 2)})
    episode_a = PathfindingEpisode(num_nodes=12, start=0, goal=11, edges=public_edges)
    episode_b = PathfindingEpisode(
        num_nodes=12,
        start=0,
        goal=11,
        edges=public_edges | {(0, 5)},
        blocked_edges=frozenset({(0, 5)}),
    )
    assert episode_a.digest() != episode_b.digest()

    observation_a = _observation(episode_a)
    observation_b = _observation(episode_b)
    assert observation_a.effective_adjacency == observation_b.effective_adjacency
    assert torch.equal(
        brancher.observation_features(observation_a),
        brancher.observation_features(observation_b),
    )


# ---------------------------------------------------------------------------
# Deterministic candidate proposal
# ---------------------------------------------------------------------------

def test_candidate_shape_distinct_actions_and_score_order() -> None:
    brancher = ActionConditionedBrancher()
    episode = _episode()
    proposal = _proposal(brancher, episode)

    assert proposal.stacked_states().shape == (1, 3, 256)
    assert len(proposal.candidates) == 3
    assert len(set(proposal.candidate_action_ids)) == 3
    assert set(proposal.candidate_action_ids) <= set(episode.allowed_moves(episode.start))
    assert list(proposal.candidate_scores) == sorted(proposal.candidate_scores, reverse=True)
    for candidate in proposal.candidates:
        assert candidate.state.shape == (1, 256)
        assert candidate.action_id in episode.allowed_moves(episode.start)


def test_fewer_than_three_eligible_actions_are_returned() -> None:
    brancher = ActionConditionedBrancher()

    two_moves = _episode(edges=frozenset({(0, 1), (1, 11), (0, 2)}))
    one_move = _episode(edges=frozenset({(0, 1), (1, 11)}))
    no_moves = _episode(edges=frozenset({(1, 11)}))

    assert len(_proposal(brancher, two_moves).candidate_action_ids) == 2
    assert len(_proposal(brancher, one_move).candidate_action_ids) == 1
    empty = _proposal(brancher, no_moves)
    assert empty.candidate_action_ids == ()
    assert empty.candidates == ()


def test_masked_actions_are_excluded() -> None:
    brancher = ActionConditionedBrancher()
    episode = _episode()
    container = LatentState(torch.zeros(1, brancher.state_dim))
    observation = _observation(episode)
    legal = episode.allowed_moves(episode.start)

    with torch.no_grad():
        masked = brancher.propose_candidates(
            container.committed_state, observation, masked_actions=legal[:2]
        )
        all_masked = brancher.propose_candidates(
            container.committed_state, observation, masked_actions=legal
        )

    assert set(masked.candidate_action_ids) == set(legal[2:])
    assert all_masked.candidate_action_ids == ()


def test_policy_cannot_propose_illegal_actions() -> None:
    brancher = ActionConditionedBrancher()
    episode = _episode()
    container = LatentState(torch.zeros(1, brancher.state_dim))
    observation = _observation(episode)
    legal = episode.allowed_moves(episode.start)
    illegal = next(node for node in range(12) if node not in legal)

    with torch.no_grad():
        brancher.policy_head.weight.zero_()
        brancher.policy_head.bias.zero_()
        brancher.policy_head.bias[illegal] = 100.0
        proposal = brancher.propose_candidates(container.committed_state, observation)

    assert illegal not in proposal.candidate_action_ids
    assert set(proposal.candidate_action_ids) <= set(legal)

    with torch.no_grad():
        brancher.policy_head.bias[illegal] = 0.0
        brancher.policy_head.bias[legal[0]] = 100.0
        preferred = brancher.propose_candidates(container.committed_state, observation)
    assert preferred.candidate_action_ids[0] == legal[0]


def test_candidate_generation_is_deterministic() -> None:
    brancher = ActionConditionedBrancher()
    episode = _episode()
    container = LatentState(torch.zeros(1, brancher.state_dim))
    observation = _observation(episode)

    with torch.no_grad():
        first = brancher.propose_candidates(container.committed_state, observation)
        second = brancher.propose_candidates(container.committed_state, observation)

    assert first.candidate_action_ids == second.candidate_action_ids
    assert first.candidate_scores == second.candidate_scores
    for first_candidate, second_candidate in zip(first.candidates, second.candidates):
        assert torch.equal(first_candidate.state, second_candidate.state)


def test_dtype_consistency_with_double_precision_model() -> None:
    brancher = ActionConditionedBrancher().double()
    container = LatentState(torch.zeros(1, brancher.state_dim, dtype=torch.float64))
    observation = _observation(_episode())

    with torch.no_grad():
        features = brancher.observation_features(observation)
        proposal = brancher.propose_candidates(container.committed_state, observation)

    assert features.dtype == torch.float64
    assert all(candidate.state.dtype == torch.float64 for candidate in proposal.candidates)
    assert all(
        candidate.state.device == container.committed_state.device
        for candidate in proposal.candidates
    )


def test_malformed_inputs_fail_clearly() -> None:
    brancher = ActionConditionedBrancher()
    episode = _episode()
    container = LatentState(torch.zeros(1, brancher.state_dim))
    observation = _observation(episode)

    small_episode = PathfindingEpisode(
        num_nodes=5, start=0, goal=4, edges=frozenset({(0, 1), (1, 4)})
    )
    small_observation = small_episode.observe(small_episode.initial_state())

    with pytest.raises(TypeError):
        brancher.observation_features("observation")
    with pytest.raises(ValueError):
        brancher.observation_features(small_observation)
    with pytest.raises(ValueError):
        brancher.propose_candidates(torch.zeros(2, brancher.state_dim), observation)
    with pytest.raises(ValueError):
        brancher.propose_candidates(torch.zeros(1, brancher.state_dim, dtype=torch.float64), observation)
    with pytest.raises(TypeError):
        brancher.propose_candidates("state", observation)
    with pytest.raises(ValueError):
        brancher.propose_candidates(container.committed_state, observation, num_candidates=0)
    with pytest.raises(ValueError):
        brancher.propose_candidates(container.committed_state, observation, masked_actions=(99,))
    with pytest.raises(ValueError):
        brancher.propose_candidates(container.committed_state, observation, masked_actions=(1, 1))


def test_branch_proposal_rejects_inconsistent_records() -> None:
    state = torch.zeros(1, 256)
    with pytest.raises(ValueError):
        CandidateProposal(action_id=-1, score=0.1, state=state)
    with pytest.raises(ValueError):
        CandidateProposal(action_id=1, score=float("nan"), state=state)
    with pytest.raises(ValueError):
        CandidateProposal(action_id=1, score=0.1, state=torch.zeros(256))
    with pytest.raises(ValueError):
        BranchProposal(
            candidates=(CandidateProposal(1, 0.1, state), CandidateProposal(1, 0.2, state))
        )
    with pytest.raises(ValueError):
        BranchProposal(
            candidates=(
                CandidateProposal(1, 0.1, state),
                CandidateProposal(2, 0.2, torch.zeros(1, 128)),
            )
        )
    with pytest.raises(ValueError):
        BranchProposal(candidates=()).stacked_states()


# ---------------------------------------------------------------------------
# Speculative isolation and gate integration
# ---------------------------------------------------------------------------

def test_proposal_leaves_committed_state_untouched() -> None:
    brancher = ActionConditionedBrancher()
    container = LatentState(torch.zeros(1, brancher.state_dim))
    observation = _observation(_episode())
    before = container.committed_state.clone()

    with torch.no_grad():
        brancher.propose_candidates(container.committed_state, observation)

    assert torch.equal(container.committed_state, before)
    assert not container.has_speculative


def test_candidate_tensor_isolation_through_speculative_set() -> None:
    brancher = ActionConditionedBrancher()
    container = LatentState(torch.zeros(1, brancher.state_dim))
    candidate_set = SpeculativeCandidateSet(container, k=2)
    first_values = torch.full((1, brancher.state_dim), 1.0)
    second_values = torch.full((1, brancher.state_dim), 2.0)

    candidate_set.update(0, first_values)
    candidate_set.update(1, second_values)
    first_view = candidate_set.state(0)
    second_view = candidate_set.state(1)
    assert first_view.data_ptr() != second_view.data_ptr()
    assert container.committed_state.data_ptr() != first_view.data_ptr()

    first_view.add_(5.0)
    assert torch.equal(second_view, second_values)
    assert torch.equal(container.committed_state, torch.zeros(1, brancher.state_dim))
    candidate_set.discard()


def test_gate_commits_only_the_highest_ranked_valid_candidate() -> None:
    brancher = ActionConditionedBrancher()
    container = LatentState(torch.zeros(1, brancher.state_dim))
    observation = _observation(_episode())
    with torch.no_grad():
        proposal = brancher.propose_candidates(container.committed_state, observation)

    candidate_set = SpeculativeCandidateSet(container, k=len(proposal.candidates))
    for index, candidate in enumerate(proposal.candidates):
        candidate_set.update(index, candidate.state)

    records = [
        CandidateRecord(
            candidate_id=proposal.candidates[0].action_id,
            state=candidate_set.state(0),
            verification=VerificationOutcome(valid=False),
            score=10.0,
        ),
        CandidateRecord(
            candidate_id=proposal.candidates[1].action_id,
            state=candidate_set.state(1),
            verification=VerificationOutcome(valid=True),
            score=proposal.candidates[1].score,
        ),
        CandidateRecord(
            candidate_id=proposal.candidates[2].action_id,
            state=candidate_set.state(2),
            verification=VerificationOutcome(valid=True),
            score=proposal.candidates[2].score,
        ),
    ]

    gate = TransactionCommitGate(container)
    decision = gate.decide(records)
    assert decision.selected_id == proposal.candidates[1].action_id

    outcome = gate.apply(decision)
    assert outcome.committed
    assert torch.equal(container.committed_state, proposal.candidates[1].state)
    assert not container.has_speculative


def test_gate_decision_is_read_only_and_abort_releases_safely() -> None:
    brancher = ActionConditionedBrancher()
    container = LatentState(torch.zeros(1, brancher.state_dim))
    observation = _observation(_episode())
    with torch.no_grad():
        proposal = brancher.propose_candidates(container.committed_state, observation)
    before = container.committed_state.clone()

    candidate_set = SpeculativeCandidateSet(container, k=len(proposal.candidates))
    for index, candidate in enumerate(proposal.candidates):
        candidate_set.update(index, candidate.state)
    records = [
        CandidateRecord(
            candidate_id=candidate.action_id,
            state=candidate_set.state(index),
            verification=VerificationOutcome(valid=False),
            score=candidate.score,
        )
        for index, candidate in enumerate(proposal.candidates)
    ]

    gate = TransactionCommitGate(container)
    decision = gate.decide(records)

    assert decision.status == TransactionCommitGate.ABORT
    assert torch.equal(container.committed_state, before)
    assert candidate_set.is_active

    candidate_set.discard()
    assert torch.equal(container.committed_state, before)
    assert not container.has_speculative
