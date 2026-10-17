"""Unit and failure-injection tests for the Milestone 3A two-phase transition orchestrator."""

import gc

import pytest
import torch

from re_reason.benchmarks.pathfinding_benchmark import (
    PathfindingEpisode,
    PathfindingState,
)
from re_reason.hypothesis.pathfinding_branching import (
    ActionConditionedBrancher,
    BranchProposal,
    CandidateProposal,
)
from re_reason.hypothesis.two_phase_orchestrator import (
    STATUS_ABORTED,
    STATUS_COMMITTED,
    TransitionResult,
    TwoPhaseTransitionOrchestrator,
)
from re_reason.state.container import LatentState
from re_reason.state.speculative import SpeculativeCandidateSet
from re_reason.verification.transaction_gate import (
    TransactionDecision,
    TransactionGateError,
    VerificationOutcome,
)


def _episode() -> PathfindingEpisode:
    return PathfindingEpisode(
        num_nodes=12,
        start=0,
        goal=11,
        edges=frozenset({(0, 1), (1, 11), (0, 2), (0, 3)}),
    )


def _orchestrator(
    environment_state: PathfindingState | None = None,
) -> TwoPhaseTransitionOrchestrator:
    return TwoPhaseTransitionOrchestrator(
        _episode(),
        LatentState(torch.zeros(1, 256)),
        ActionConditionedBrancher(),
        environment_state,
    )


def _precomputed_proposal(orchestrator: TwoPhaseTransitionOrchestrator):
    container = orchestrator.container
    observation = orchestrator.episode.observe(orchestrator.environment_state)
    with torch.no_grad():
        return orchestrator.brancher.propose_candidates(container.committed_state, observation)


def _invalid_outcome(state, action) -> VerificationOutcome:
    return VerificationOutcome(valid=False, details={"reason": "injected_rejection"})


# ---------------------------------------------------------------------------
# Frozen design bounds
# ---------------------------------------------------------------------------

def test_design_bounds_are_frozen() -> None:
    assert TwoPhaseTransitionOrchestrator.MAX_PROPOSAL_ROUNDS == 3
    assert TwoPhaseTransitionOrchestrator.MAX_EPISODE_TRANSITIONS == 24
    assert TwoPhaseTransitionOrchestrator.MAX_BACKTRACKING_SNAPSHOTS == 4
    assert TwoPhaseTransitionOrchestrator.CANDIDATE_BUDGET == 3


# ---------------------------------------------------------------------------
# Successful transition path
# ---------------------------------------------------------------------------

def test_successful_transition_aligns_latent_and_environment_state() -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    proposal = _precomputed_proposal(orchestrator)

    result = orchestrator.step()

    assert result == TransitionResult(
        committed=True,
        status=STATUS_COMMITTED,
        selected_action_id=proposal.candidate_action_ids[0],
        rounds_attempted=1,
        reason=result.reason,
    )
    assert orchestrator.environment_state.current == proposal.candidate_action_ids[0]
    assert orchestrator.environment_state.transitions == 1
    assert torch.equal(container.committed_state, proposal.candidates[0].state)
    assert not container.has_speculative
    assert orchestrator.gate.container is container


def test_step_enforces_episode_transition_budget() -> None:
    exhausted = _orchestrator(
        environment_state=PathfindingState(current=0, transitions=24)
    )
    with pytest.raises(RuntimeError, match="transition budget"):
        exhausted.step()
    assert exhausted.environment_state == PathfindingState(current=0, transitions=24)

    boundary = _orchestrator(
        environment_state=PathfindingState(current=0, transitions=23)
    )
    assert boundary.step().committed
    assert boundary.environment_state.transitions == 24
    with pytest.raises(RuntimeError, match="transition budget"):
        boundary.step()


def test_commit_uses_highest_ranked_valid_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _orchestrator()
    proposal = _precomputed_proposal(orchestrator)
    assert len(proposal.candidate_action_ids) == 3
    rejected_action = proposal.candidate_action_ids[0]
    expected_action = proposal.candidate_action_ids[1]

    def selective_verification(state, action) -> VerificationOutcome:
        if action.target == rejected_action:
            return VerificationOutcome(valid=False, details={"reason": "injected_rejection"})
        return VerificationOutcome(valid=True)

    monkeypatch.setattr(orchestrator.verifier, "verify_transition", selective_verification)

    result = orchestrator.step()

    assert result.committed
    assert result.selected_action_id == expected_action
    assert orchestrator.environment_state.current == expected_action
    assert torch.equal(
        orchestrator.container.committed_state, proposal.candidates[1].state
    )


def test_dead_end_transition_is_committed_not_rejected() -> None:
    orchestrator = _orchestrator()
    with torch.no_grad():
        orchestrator.brancher.policy_head.weight.zero_()
        orchestrator.brancher.policy_head.bias.zero_()
        orchestrator.brancher.policy_head.bias[2] = 10.0

    proposal = _precomputed_proposal(orchestrator)
    assert proposal.candidate_action_ids[0] == 2

    result = orchestrator.step()

    assert result.committed
    assert orchestrator.environment_state.current == 2
    assert orchestrator.episode.is_stuck(orchestrator.environment_state)
    assert not orchestrator.episode.is_goal_reached(orchestrator.environment_state)


# ---------------------------------------------------------------------------
# Phase-1 failure boundaries (no committed-state mutation)
# ---------------------------------------------------------------------------

def test_verifier_exception_preserves_state(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state

    def boom(state, action):
        raise RuntimeError("injected verifier failure")

    monkeypatch.setattr(orchestrator.verifier, "verify_transition", boom)

    with pytest.raises(RuntimeError, match="injected verifier failure"):
        orchestrator.step()

    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before
    assert not container.has_speculative
    assert orchestrator.gate.container is container


def test_gate_decision_exception_preserves_state(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state

    def boom(records):
        raise RuntimeError("injected gate decision failure")

    monkeypatch.setattr(orchestrator.gate, "decide", boom)

    with pytest.raises(RuntimeError, match="injected gate decision failure"):
        orchestrator.step()

    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before
    assert not container.has_speculative


def test_gate_invalid_decision_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state

    def invalid_decision(records) -> TransactionDecision:
        return TransactionDecision(
            status="BROKEN",
            selected=None,
            selected_id=None,
            ranked_candidate_ids=(),
            reason="injected invalid decision",
        )

    monkeypatch.setattr(orchestrator.gate, "decide", invalid_decision)

    with pytest.raises(TransactionGateError, match="unexpected transaction gate decision status"):
        orchestrator.step()

    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before
    assert not container.has_speculative


def test_preview_exception_preserves_state(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state

    def boom(action_id):
        raise RuntimeError("injected preview failure")

    monkeypatch.setattr(orchestrator, "_preview_environment_transition", boom)

    with pytest.raises(RuntimeError, match="injected preview failure"):
        orchestrator.step()

    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before
    assert not container.has_speculative


# ---------------------------------------------------------------------------
# Phase-2 failure boundaries (recovery via the external checkpoint)
# ---------------------------------------------------------------------------

def test_commit_exception_before_mutation_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state

    def boom(decision):
        raise RuntimeError("injected commit failure before mutation")

    monkeypatch.setattr(orchestrator.gate, "apply", boom)

    with pytest.raises(RuntimeError, match="injected commit failure before mutation"):
        orchestrator.step()

    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before
    assert not container.has_speculative


def test_commit_exception_after_mutation_restores_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state
    original_apply = orchestrator.gate.apply

    def apply_then_fail(decision):
        original_apply(decision)
        raise RuntimeError("injected post-mutation commit failure")

    monkeypatch.setattr(orchestrator.gate, "apply", apply_then_fail)

    with pytest.raises(RuntimeError, match="injected post-mutation commit failure"):
        orchestrator.step()

    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before
    assert container.committed_state.grad_fn is None
    assert not container.has_speculative
    assert orchestrator.gate.container is container


def test_environment_advance_exception_restores_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state

    def boom(next_state):
        raise RuntimeError("injected environment advance failure")

    monkeypatch.setattr(orchestrator, "_advance_environment", boom)

    with pytest.raises(RuntimeError, match="injected environment advance failure"):
        orchestrator.step()

    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before
    assert not container.has_speculative
    assert orchestrator.gate.container is container


def test_recovery_preserves_binding_and_allows_next_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()

    def boom(next_state):
        raise RuntimeError("injected environment advance failure")

    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, "_advance_environment", boom)
        with pytest.raises(RuntimeError, match="injected environment advance failure"):
            orchestrator.step()

    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.gate.container is container

    result = orchestrator.step()
    assert result.committed
    assert orchestrator.environment_state.transitions == 1


def test_candidate_cleanup_failure_is_reported_and_buffers_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state
    monkeypatch.setattr(orchestrator.verifier, "verify_transition", _invalid_outcome)

    def boom(self):
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(SpeculativeCandidateSet, "discard", boom)

    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        orchestrator.step()

    assert not container.has_speculative
    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before


# ---------------------------------------------------------------------------
# Bounded retry policy
# ---------------------------------------------------------------------------

def test_retry_masks_rejected_actions_and_aborts_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state
    monkeypatch.setattr(orchestrator.verifier, "verify_transition", _invalid_outcome)

    result = orchestrator.step()

    assert result.committed is False
    assert result.status == STATUS_ABORTED
    assert result.selected_action_id is None
    assert result.rounds_attempted <= TwoPhaseTransitionOrchestrator.MAX_PROPOSAL_ROUNDS
    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before
    assert not container.has_speculative


def test_stubborn_proposals_exhaust_bounded_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    calls = []

    def proposal_stub(committed_state, observation, num_candidates=3, masked_actions=()):
        calls.append(tuple(masked_actions))
        return BranchProposal(
            candidates=(
                CandidateProposal(
                    action_id=7,
                    score=0.5,
                    state=torch.zeros(1, orchestrator.brancher.state_dim),
                ),
            )
        )

    monkeypatch.setattr(orchestrator.brancher, "propose_candidates", proposal_stub)
    monkeypatch.setattr(orchestrator.verifier, "verify_transition", _invalid_outcome)

    result = orchestrator.step()

    assert result.committed is False
    assert result.rounds_attempted == TwoPhaseTransitionOrchestrator.MAX_PROPOSAL_ROUNDS
    assert "retry budget" in result.reason
    assert len(calls) == TwoPhaseTransitionOrchestrator.MAX_PROPOSAL_ROUNDS
    assert torch.equal(container.committed_state, checkpoint)
    assert not container.has_speculative


def test_illegal_proposal_is_rejected_and_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    masked_actions_seen = []

    def proposal_stub(committed_state, observation, num_candidates=3, masked_actions=()):
        masked_actions_seen.append(tuple(masked_actions))
        if len(masked_actions_seen) == 1:
            return BranchProposal(
                candidates=(
                    CandidateProposal(
                        action_id=7,
                        score=0.5,
                        state=torch.zeros(1, orchestrator.brancher.state_dim),
                    ),
                )
            )
        return BranchProposal(candidates=())

    monkeypatch.setattr(orchestrator.brancher, "propose_candidates", proposal_stub)

    result = orchestrator.step()

    assert result.committed is False
    assert result.status == STATUS_ABORTED
    assert masked_actions_seen[1] == (7,)
    assert torch.equal(container.committed_state, checkpoint)
    assert not container.has_speculative


# ---------------------------------------------------------------------------
# Repeated-failure hygiene and construction validation
# ---------------------------------------------------------------------------

def test_repeated_aborts_leave_no_stale_speculative_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    container = orchestrator.container
    checkpoint = container.committed_state.clone()
    environment_before = orchestrator.environment_state
    monkeypatch.setattr(orchestrator.verifier, "verify_transition", _invalid_outcome)

    for _ in range(200):
        result = orchestrator.step()
        assert result.committed is False
        assert not container.has_speculative

    assert torch.equal(container.committed_state, checkpoint)
    assert orchestrator.environment_state == environment_before
    gc.collect()
    assert len(gc.garbage) == 0


def test_orchestrator_validates_construction() -> None:
    episode = _episode()
    brancher = ActionConditionedBrancher()
    container = LatentState(torch.zeros(1, 256))

    with pytest.raises(TypeError):
        TwoPhaseTransitionOrchestrator(object(), container, brancher)
    with pytest.raises(TypeError):
        TwoPhaseTransitionOrchestrator(episode, object(), brancher)
    with pytest.raises(TypeError):
        TwoPhaseTransitionOrchestrator(episode, container, object())

    small_brancher = ActionConditionedBrancher(num_nodes=5, observation_input_dim=94)
    with pytest.raises(ValueError, match="nodes"):
        TwoPhaseTransitionOrchestrator(episode, container, small_brancher)
    with pytest.raises(ValueError, match="committed state shape"):
        TwoPhaseTransitionOrchestrator(episode, LatentState(torch.zeros(1, 128)), brancher)
    with pytest.raises(ValueError):
        TwoPhaseTransitionOrchestrator(
            episode, container, brancher, PathfindingState(current=99)
        )
