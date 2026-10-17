"""Unit tests for PathfindingVerifier: deterministic single-transition checks, diagnostics, and purity."""

import pytest

from re_reason.benchmarks.pathfinding_benchmark import (
    TRANSITION_BLOCKED,
    TRANSITION_NO_SUCH_EDGE,
    TRANSITION_OK,
    TRANSITION_STATE_MISMATCH,
    TRANSITION_UNKNOWN_NODE,
    PathfindingAction,
    PathfindingEpisode,
    PathfindingGeneratorConfig,
    PathfindingState,
    generate_pathfinding_episode,
)
from re_reason.verification.pathfinding_verifier import PathfindingVerifier
from re_reason.verification.transaction_gate import VerificationOutcome


def _config() -> PathfindingGeneratorConfig:
    return PathfindingGeneratorConfig()


def _episode() -> PathfindingEpisode:
    return PathfindingEpisode(
        num_nodes=4,
        start=0,
        goal=3,
        edges=frozenset({(0, 1), (1, 3), (0, 2), (2, 1)}),
        blocked_edges=frozenset({(0, 1)}),
    )


def test_verifier_accepts_valid_transition_with_structured_diagnostics() -> None:
    verifier = PathfindingVerifier(_episode())
    outcome = verifier.verify_transition(PathfindingState(current=0), PathfindingAction(current=0, target=2))

    assert isinstance(outcome, VerificationOutcome)
    assert outcome.valid is True
    assert outcome.details == {"current": 0, "proposed": 2, "reason": TRANSITION_OK}


def test_verifier_rejection_reasons_are_distinguished() -> None:
    verifier = PathfindingVerifier(_episode())
    cases = [
        (PathfindingState(current=0), PathfindingAction(current=0, target=3), TRANSITION_NO_SUCH_EDGE),
        (PathfindingState(current=0), PathfindingAction(current=0, target=1), TRANSITION_BLOCKED),
        (PathfindingState(current=0), PathfindingAction(current=2, target=1), TRANSITION_STATE_MISMATCH),
        (PathfindingState(current=0), PathfindingAction(current=0, target=9), TRANSITION_UNKNOWN_NODE),
        (PathfindingState(current=9), PathfindingAction(current=9, target=0), TRANSITION_UNKNOWN_NODE),
    ]
    for state, action, expected_reason in cases:
        outcome = verifier.verify_transition(state, action)
        assert outcome.valid is False
        assert outcome.details["reason"] == expected_reason
        assert outcome.details["current"] == state.current
        assert outcome.details["proposed"] == action.target


def test_verifier_rejects_unknown_nodes_and_missing_edges_independently() -> None:
    episode = _episode()
    verifier = PathfindingVerifier(episode)

    unknown_target = verifier.verify_transition(PathfindingState(current=0), PathfindingAction(current=0, target=9))
    missing_edge = verifier.verify_transition(PathfindingState(current=0), PathfindingAction(current=0, target=3))

    assert unknown_target.valid is False and unknown_target.details["reason"] == TRANSITION_UNKNOWN_NODE
    assert missing_edge.valid is False and missing_edge.details["reason"] == TRANSITION_NO_SUCH_EDGE
    assert (0, 3) not in episode.edges


def test_verifier_is_deterministic() -> None:
    verifier = PathfindingVerifier(_episode())
    state = PathfindingState(current=0)
    action = PathfindingAction(current=0, target=1)

    first = verifier.verify_transition(state, action)
    second = verifier.verify_transition(state, action)

    assert first == second
    assert first.details == second.details


def test_verifier_output_surface_is_bounded() -> None:
    verifier = PathfindingVerifier(generate_pathfinding_episode(_config(), seed=2))
    state = PathfindingState(current=verifier.episode.start)

    for target in range(verifier.episode.num_nodes):
        outcome = verifier.verify_transition(state, PathfindingAction(current=state.current, target=target))
        assert sorted(outcome.details.keys()) == ["current", "proposed", "reason"]
        assert isinstance(outcome.details["current"], int)
        assert isinstance(outcome.details["proposed"], int)
        assert isinstance(outcome.details["reason"], str)


def test_verifier_is_side_effect_free() -> None:
    episode = generate_pathfinding_episode(_config(), seed=6)
    verifier = PathfindingVerifier(episode)
    digest_before = episode.digest()
    initial = episode.initial_state()

    for u in range(episode.num_nodes):
        for v in range(episode.num_nodes):
            verifier.verify_transition(PathfindingState(current=u), PathfindingAction(current=u, target=v))

    assert episode.digest() == digest_before
    assert initial == episode.initial_state()
    assert episode.blocked_edges <= episode.edges


def test_verifier_rejects_argument_type_errors() -> None:
    with pytest.raises(TypeError):
        PathfindingVerifier(object())

    verifier = PathfindingVerifier(_episode())
    with pytest.raises(TypeError):
        verifier.verify_transition("state", PathfindingAction(current=0, target=2))
    with pytest.raises(TypeError):
        verifier.verify_transition(PathfindingState(current=0), "action")


def test_cross_api_agreement_on_allowed_transitions() -> None:
    episode = generate_pathfinding_episode(_config(), seed=8)
    verifier = PathfindingVerifier(episode)
    observation = episode.observe(episode.initial_state())

    assert observation.allowed_moves == episode.allowed_moves(episode.start)

    for u in range(episode.num_nodes):
        state = PathfindingState(current=u)
        moves = episode.allowed_moves(u)
        for v in range(episode.num_nodes):
            expected = v in moves
            action = PathfindingAction(current=u, target=v)
            outcome = verifier.verify_transition(state, action)
            assert outcome.valid == expected
            assert episode.is_allowed_move(u, v) == expected
            if expected:
                advanced = episode.apply_transition(state, action)
                assert advanced.current == v
                assert advanced.transitions == state.transitions + 1
            else:
                with pytest.raises(ValueError):
                    episode.apply_transition(state, action)
