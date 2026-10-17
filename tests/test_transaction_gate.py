"""Unit tests for TransactionCommitGate: deterministic selection, fail-closed validation, atomic commit/rollback."""

import gc
import weakref

import pytest
import torch

from re_reason.state.container import LatentState
from re_reason.state.speculative import SpeculativeCandidateSet, StateLifecycleError
from re_reason.utils.memory import inspect_tensor_retention
from re_reason.verification.transaction_gate import (
    CandidateRecord,
    DuplicateCandidateError,
    InvalidCandidateStateError,
    InvalidRankingScoreError,
    TransactionCommitGate,
    TransactionDecision,
    TransactionGateError,
    UnverifiedCandidateError,
    VerificationOutcome,
)


def _container(batch_size: int = 4, dim: int = 16) -> LatentState:
    return LatentState.zeros(batch_size, dim)


def _state(container: LatentState, fill: float) -> torch.Tensor:
    return torch.full(container.shape, fill)


def _record(
    candidate_id,
    state: torch.Tensor,
    valid: bool,
    score=None,
    details=None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        state=state,
        verification=VerificationOutcome(valid=valid, details=details or {}),
        score=score,
    )


def test_single_valid_candidate_commits() -> None:
    """Case A: exactly one valid candidate is committed regardless of the other records."""
    container = _container()
    before = container.committed_state.clone()
    gate = TransactionCommitGate(container)
    winner = _record("b", _state(container, 3.0), valid=True, score=1)

    outcome = gate.decide_and_commit(
        [
            _record("a", _state(container, 1.0), valid=False),
            winner,
            _record("c", _state(container, 2.0), valid=False),
        ]
    )

    assert outcome.status == TransactionCommitGate.COMMIT
    assert outcome.committed is True
    assert outcome.selected_id == "b"
    assert torch.equal(container.committed_state, winner.state)
    assert not torch.equal(container.committed_state, before)
    assert not container.has_speculative


def test_highest_ranked_valid_candidate_wins() -> None:
    """Case B: among several valid candidates the highest scoring one wins."""
    container = _container()
    gate = TransactionCommitGate(container)
    records = [
        _record("lo", _state(container, 1.0), valid=True, score=0.1),
        _record("hi", _state(container, 9.0), valid=True, score=0.9),
        _record("mid", _state(container, 5.0), valid=True, score=0.5),
    ]

    decision = gate.decide(records)
    assert decision.ranked_candidate_ids == ("hi", "mid", "lo")

    outcome = gate.apply(decision)
    assert outcome.selected_id == "hi"
    assert torch.equal(container.committed_state, _state(container, 9.0))


def test_invalid_candidate_with_highest_score_never_wins() -> None:
    """An invalid candidate must not win merely because its ranking score is higher."""
    container = _container()
    gate = TransactionCommitGate(container)
    records = [
        _record("invalid-top", _state(container, 9.0), valid=False, score=9.0),
        _record("valid-low", _state(container, 2.0), valid=True, score=0.001),
    ]

    outcome = gate.decide_and_commit(records)

    assert outcome.selected_id == "valid-low"
    assert torch.equal(container.committed_state, _state(container, 2.0))


def test_score_ties_break_deterministically_by_earliest_position() -> None:
    """Equal scores resolve to the earliest candidate; reordering swaps the deterministic winner."""
    container = _container()
    gate = TransactionCommitGate(container)
    first = _record("first", _state(container, 1.0), valid=True, score=0.5)
    second = _record("second", _state(container, 2.0), valid=True, score=0.5)

    decision_one = gate.decide([first, second])
    decision_repeat = gate.decide([first, second])
    assert decision_one.selected_id == "first"
    assert decision_one.selected_id == decision_repeat.selected_id
    assert decision_one.ranked_candidate_ids == decision_repeat.ranked_candidate_ids
    assert decision_one.reason == decision_repeat.reason

    decision_reversed = gate.decide([second, first])
    assert decision_reversed.selected_id == "second"
    assert decision_reversed.ranked_candidate_ids == ("second", "first")


def test_all_rejected_aborts_and_preserves_master_bit_exactly() -> None:
    """Case C: an all-rejected collection aborts explicitly and the master stays bit-exact."""
    container = _container()
    before = container.committed_state.clone()
    gate = TransactionCommitGate(container)
    records = [
        _record("a", _state(container, 1.0), valid=False),
        _record("b", _state(container, 2.0), valid=False, score=5.0),
    ]

    decision = gate.decide(records)
    assert decision.status == TransactionCommitGate.ABORT
    assert decision.selected is None
    assert decision.selected_id is None
    assert decision.ranked_candidate_ids == ()
    assert torch.equal(container.committed_state, before)

    outcome = gate.decide_and_commit(records)
    assert outcome.status == TransactionCommitGate.ABORT
    assert outcome.committed is False
    assert outcome.selected_id is None
    assert torch.equal(container.committed_state, before)
    assert not container.has_speculative


def test_missing_verification_outcome_never_commits() -> None:
    """Case D: a candidate without a typed verification outcome cannot be committed."""
    container = _container()
    before = container.committed_state.clone()
    gate = TransactionCommitGate(container)

    missing = CandidateRecord(
        candidate_id="unverified",
        state=_state(container, 1.0),
        verification=None,
        score=1.0,
    )
    with pytest.raises(UnverifiedCandidateError):
        gate.decide_and_commit([missing])

    wrong_type = CandidateRecord(
        candidate_id="wrong-type",
        state=_state(container, 1.0),
        verification="valid",
        score=1.0,
    )
    with pytest.raises(UnverifiedCandidateError):
        gate.decide([wrong_type])

    assert torch.equal(container.committed_state, before)


def test_ranking_score_policy_is_fail_closed() -> None:
    """Missing, non-finite, or wrong-typed ranking scores raise before any mutation."""
    container = _container()
    before = container.committed_state.clone()
    gate = TransactionCommitGate(container)

    malformed_scores = [
        (True, None),
        (True, float("nan")),
        (True, float("inf")),
        (True, float("-inf")),
        (True, "0.5"),
        (True, True),
        (False, float("nan")),
    ]
    for valid, score in malformed_scores:
        record = _record("candidate", _state(container, 1.0), valid=valid, score=score)
        with pytest.raises(InvalidRankingScoreError):
            gate.decide_and_commit([record])
        assert torch.equal(container.committed_state, before)

    tolerated = _record("invalid-no-score", _state(container, 1.0), valid=False)
    outcome = gate.decide_and_commit([tolerated])
    assert outcome.status == TransactionCommitGate.ABORT
    assert torch.equal(container.committed_state, before)


def test_duplicate_candidate_identities_are_rejected() -> None:
    """Duplicate identifiers (including numerically equal ids) are rejected explicitly."""
    container = _container()
    before = container.committed_state.clone()
    gate = TransactionCommitGate(container)

    duplicate_same = [
        _record("x", _state(container, 1.0), valid=True, score=0.1),
        _record("x", _state(container, 2.0), valid=True, score=0.2),
    ]
    with pytest.raises(DuplicateCandidateError):
        gate.decide(duplicate_same)

    duplicate_equal = [
        _record(1, _state(container, 1.0), valid=True, score=0.1),
        _record(1.0, _state(container, 2.0), valid=True, score=0.2),
    ]
    with pytest.raises(DuplicateCandidateError):
        gate.decide(duplicate_equal)

    assert torch.equal(container.committed_state, before)


def test_malformed_candidate_collections_are_rejected() -> None:
    """Empty, non-sequence, non-record, and unhashable-id collections are rejected."""
    container = _container()
    before = container.committed_state.clone()
    gate = TransactionCommitGate(container)

    with pytest.raises(TransactionGateError, match="at least one"):
        gate.decide([])
    with pytest.raises(TransactionGateError):
        gate.decide((record for record in [_record("a", _state(container, 1.0), valid=True, score=0.1)]))
    with pytest.raises(TransactionGateError, match="not a CandidateRecord"):
        gate.decide([object()])
    unhashable = CandidateRecord(
        candidate_id=[1, 2],
        state=_state(container, 1.0),
        verification=VerificationOutcome(valid=True),
        score=0.1,
    )
    with pytest.raises(TransactionGateError, match="unhashable"):
        gate.decide([unhashable])

    assert torch.equal(container.committed_state, before)


def test_invalid_candidate_tensor_states_are_rejected() -> None:
    """Mis-shaped, mis-typed, mis-placed, non-tensor, and non-finite states are rejected."""
    container = _container()
    before = container.committed_state.clone()
    gate = TransactionCommitGate(container)

    invalid_states = [
        _state(container, 1.0)[:, :8],
        torch.zeros(4, 16, dtype=torch.float64),
        torch.zeros(4, 16, device="meta"),
        _state(container, float("nan")),
        "not a tensor",
    ]
    for state in invalid_states:
        with pytest.raises(InvalidCandidateStateError):
            gate.decide([_record("bad", state, valid=True, score=0.1)])

    mixed = [
        _record("good", _state(container, 1.0), valid=True, score=0.1),
        _record("bad-shape", _state(container, 1.0)[:2, :], valid=True, score=0.9),
    ]
    with pytest.raises(InvalidCandidateStateError):
        gate.decide(mixed)

    assert torch.equal(container.committed_state, before)


def test_apply_revalidates_winner_and_cannot_partially_commit() -> None:
    """A winner corrupted between decide and apply raises with the master never mutated."""
    container = _container()
    before = container.committed_state.clone()
    gate = TransactionCommitGate(container)
    winner = _record("winner", _state(container, 7.0), valid=True, score=0.5)

    decision = gate.decide([winner])
    assert decision.status == TransactionCommitGate.COMMIT

    winner.state.fill_(float("nan"))
    with pytest.raises(InvalidCandidateStateError):
        gate.apply(decision)
    assert torch.equal(container.committed_state, before)


def test_apply_rejects_unknown_status_and_non_decisions() -> None:
    """Malformed decisions are rejected explicitly by apply."""
    container = _container()
    gate = TransactionCommitGate(container)

    bogus = TransactionDecision(
        status="MAYBE",
        selected=None,
        selected_id=None,
        ranked_candidate_ids=(),
        reason="bogus",
    )
    with pytest.raises(TransactionGateError, match="unknown decision status"):
        gate.apply(bogus)

    empty_commit = TransactionDecision(
        status=TransactionCommitGate.COMMIT,
        selected=None,
        selected_id=None,
        ranked_candidate_ids=(),
        reason="malformed",
    )
    with pytest.raises(TransactionGateError, match="no selected candidate"):
        gate.apply(empty_commit)

    with pytest.raises(TransactionGateError, match="TransactionDecision"):
        gate.apply("not a decision")


def test_decide_is_read_only_and_repeatable() -> None:
    """decide leaves the container, candidates, and speculative buffers untouched."""
    container = _container()
    gate = TransactionCommitGate(container)
    candidates = SpeculativeCandidateSet(container, k=2)
    candidates.update(0, _state(container, 1.5))
    candidates.update(1, _state(container, 2.5))
    master_before = container.committed_state.clone()

    records = [
        _record("ok", candidates.state(0), valid=True, score=0.3),
        _record("no", candidates.state(1), valid=False),
    ]
    first = gate.decide(records)
    second = gate.decide(records)

    assert first.status == second.status
    assert first.selected_id == second.selected_id
    assert first.ranked_candidate_ids == second.ranked_candidate_ids
    assert first.reason == second.reason
    assert torch.equal(container.committed_state, master_before)
    assert container.has_speculative
    assert torch.equal(candidates.state(1), _state(container, 2.5))


def test_commit_through_gate_promotes_winner_and_disposes_candidates() -> None:
    """Gate commit through a spawned candidate set promotes the winner and orphans the set."""
    container = _container()
    gate = TransactionCommitGate(container)
    candidates = SpeculativeCandidateSet(container, k=3)
    candidates.update(1, _state(container, 9.0))

    records = [
        _record("a", candidates.state(0), valid=False),
        _record("b", candidates.state(1), valid=True, score=0.2),
        _record("c", candidates.state(2), valid=False),
    ]
    outcome = gate.decide_and_commit(records)

    assert outcome.committed is True
    assert outcome.selected_id == "b"
    assert torch.equal(container.committed_state, _state(container, 9.0))
    assert not container.has_speculative
    with pytest.raises(StateLifecycleError, match="invalidated"):
        candidates.state(0)


def test_abort_through_gate_rolls_back_and_preserves_master_exactly() -> None:
    """Gate abort through a spawned candidate set preserves the master bit-exactly."""
    container = _container()
    gate = TransactionCommitGate(container)
    before = container.committed_state.clone()
    candidates = SpeculativeCandidateSet(container, k=2)
    candidates.update(0, _state(container, 1.0))
    candidates.update(1, _state(container, 2.0))

    records = [
        _record("a", candidates.state(0), valid=False),
        _record("b", candidates.state(1), valid=False),
    ]
    outcome = gate.decide_and_commit(records)

    assert outcome.status == TransactionCommitGate.ABORT
    assert torch.equal(container.committed_state, before)
    assert not container.has_speculative
    with pytest.raises(StateLifecycleError, match="invalidated"):
        candidates.state(1)


def test_verification_outcome_requires_strict_boolean() -> None:
    """Verification outcomes reject non-boolean validity flags."""
    with pytest.raises(TransactionGateError):
        VerificationOutcome(valid=1)
    with pytest.raises(TransactionGateError):
        VerificationOutcome(valid="yes")

    outcome = VerificationOutcome(valid=False, details={"reason": "blocked"})
    assert outcome.valid is False
    assert outcome.details["reason"] == "blocked"


def test_gate_requires_latent_state_container() -> None:
    """The gate refuses containers that are not LatentState instances."""
    with pytest.raises(TransactionGateError):
        TransactionCommitGate(object())


def test_metadata_and_ranked_ids_are_preserved_for_auditing() -> None:
    """Decision records retain the winning record's metadata and the full ranked eligible order."""
    container = _container()
    gate = TransactionCommitGate(container)
    winner = _record("w", _state(container, 1.0), valid=True, score=0.7, details={"op": "probe"})
    decision = gate.decide(
        [
            _record("x", _state(container, 2.0), valid=True, score=0.2),
            winner,
            _record("y", _state(container, 3.0), valid=False, score=5.0),
        ]
    )

    assert decision.selected is winner
    assert decision.selected.metadata == {}
    assert decision.selected.verification.details == {"op": "probe"}
    assert decision.ranked_candidate_ids == ("w", "x")


def test_repeated_gate_cycles_do_not_retain_candidates_or_graphs() -> None:
    """1,000 commit/abort gate cycles: no candidate tensor outlives its cycle.

    CPU-only loop using exact reference-death diagnostics (tensor weakrefs plus gc
    garbage inspection) rather than allocator reservation statistics.
    """
    container = _container()
    gate = TransactionCommitGate(container)
    references: list[weakref.ReferenceType] = []

    def cycle(index: int) -> weakref.ReferenceType:
        candidate_state = _state(container, float(index % 7) + 1.0)
        reference = weakref.ref(candidate_state)
        valid = index % 2 == 0
        outcome = gate.decide_and_commit(
            [_record("candidate", candidate_state, valid=valid, score=0.5 if valid else None)]
        )
        assert outcome.committed is valid
        return reference

    for index in range(1000):
        references.append(cycle(index))

    gc.collect()
    alive = [reference for reference in references if reference() is not None]
    assert alive == [], f"{len(alive)} candidate tensors outlived their cycles"
    assert len(gc.garbage) == 0
    assert inspect_tensor_retention(container.committed_state)["has_grad_fn"] is False
