"""Transactional commit/rollback arbitration over verified speculative candidates.

Milestone 1 (Transaction Commit Gate): the gate consumes externally supplied
candidate records — each carrying an explicit candidate identity, a candidate
working-state tensor, an externally supplied verification outcome, an optional
finite ranking score, and audit metadata — and deterministically selects at most
one eligible (verified-valid) candidate for commit. Invalid candidates can never
win merely because their ranking scores are higher. If no candidate is eligible,
the gate returns an explicit abort and the committed master state is preserved
bit-exactly.

Decision policy (deterministic):

1. Eligibility is exactly ``verification.valid is True``. A candidate whose
   verification outcome is missing or false is never eligible.
2. Among eligible candidates, the winner is the one with the highest finite
   ranking score. Ranking scores are ordering keys only; they are not claimed to
   be calibrated probabilities.
3. Ties are broken by the earliest position in the supplied candidate sequence.
4. No eligible candidate yields an explicit ``ABORT``; the gate never silently
   selects an invalid candidate and never retries.

Input contract (fail-closed, validated before any state mutation):

* ``candidates``: a non-empty sequence of :class:`CandidateRecord`; candidate
  identifiers must be hashable and unique within the collection.
* ``verification``: a :class:`VerificationOutcome` per record; a missing or
  non-outcome value raises :class:`UnverifiedCandidateError`.
* ``state``: a ``torch.Tensor`` whose shape, dtype, and device match the bound
  container's committed master state, with finite values.
* ``score``: when provided, a finite real number (booleans excluded); every
  eligible candidate must provide one. Non-finite, missing, or wrong-typed scores
  raise :class:`InvalidRankingScoreError`.

Verification boundary: the gate consumes verification outcomes; it does not
perform verification itself, does not learn a verifier, and does not evaluate
semantic correctness of latent states. Mapping from a
:class:`re_reason.verification.base.BaseVerifier` batch result
(``valid_mask, diagnostics``) to per-candidate :class:`VerificationOutcome`
records is the orchestrator's responsibility; any ranking score sourced from
verifier diagnostics remains an uninterpreted ordering key.

Transaction semantics: :meth:`TransactionCommitGate.decide` is read-only.
The transaction is finalized only by :meth:`TransactionCommitGate.apply` (or the
convenience :meth:`TransactionCommitGate.decide_and_commit`): a commit replaces
the committed master state and releases speculative buffers through the
container contract; an abort rolls the container back to its pre-step snapshot
and releases speculative buffers. The gate is bound to exactly one
:class:`~re_reason.state.container.LatentState` container and assumes
single-threaded orchestration: the container must not be mutated between
``decide`` and ``apply``. Validation, selection, and application form one
logical transaction; a malformed input or failed validation raises before any
state mutation, so the committed master state is never left partially updated.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Mapping, Optional, Tuple

import torch

from re_reason.state.container import LatentState


class TransactionGateError(ValueError):
    """Base class for malformed transaction inputs and invalid gate usage."""


class DuplicateCandidateError(TransactionGateError):
    """Raised when a candidate collection contains duplicate identifiers."""


class UnverifiedCandidateError(TransactionGateError):
    """Raised when a candidate has no typed verification outcome (fail-closed)."""


class InvalidCandidateStateError(TransactionGateError):
    """Raised when a candidate state tensor violates the container's state invariants."""


class InvalidRankingScoreError(TransactionGateError):
    """Raised when a candidate ranking score violates the documented score policy."""


@dataclass(frozen=True)
class VerificationOutcome:
    """Typed externally supplied verification result for one candidate.

    ``valid`` is the authoritative eligibility signal; it must be a strict
    boolean. ``details`` optionally carries verifier diagnostics for auditing and
    must not be treated as a calibrated correctness probability.
    """

    valid: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TransactionGateError(
                f"verification outcome 'valid' must be a bool, got {type(self.valid).__name__}"
            )


@dataclass(frozen=True, eq=False)
class CandidateRecord:
    """One speculative candidate presented to the transaction gate.

    Attributes:
        candidate_id: Hashable identity, unique within a candidate collection.
        state: Candidate working-state tensor conforming to the bound container's
            committed master shape, dtype, and device.
        verification: Externally supplied verification outcome.
        score: Optional finite ranking score used only for deterministic ordering
            among eligible candidates. Not a probability.
        metadata: Optional audit metadata, carried through untouched.
    """

    candidate_id: Hashable
    state: torch.Tensor
    verification: VerificationOutcome
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class TransactionDecision:
    """Read-only outcome of candidate evaluation.

    ``status`` is ``COMMIT`` or ``ABORT``. ``selected`` is the winning record for
    a commit, otherwise ``None``. ``ranked_candidate_ids`` lists all eligible
    candidates in selection order (highest score first, ties by earliest
    position); it is empty for an abort.
    """

    status: str
    selected: Optional[CandidateRecord]
    selected_id: Optional[Hashable]
    ranked_candidate_ids: Tuple[Hashable, ...]
    reason: str


@dataclass(frozen=True)
class TransactionOutcome:
    """Result of applying a decision to the bound container.

    ``committed`` is True only when a verified-valid candidate replaced the
    committed master state; an abort leaves the master state untouched.
    """

    status: str
    committed: bool
    selected_id: Optional[Hashable]
    reason: str


class TransactionCommitGate:
    """Deterministic, fail-closed commit/rollback gate for verified candidates."""

    COMMIT = "COMMIT"
    ABORT = "ABORT"

    def __init__(self, container: LatentState) -> None:
        """Binds the gate to the container whose master state it commits or rolls back.

        Raises:
            TransactionGateError: If ``container`` is not a ``LatentState``.
        """
        if not isinstance(container, LatentState):
            raise TransactionGateError(
                f"container must be a LatentState instance, got {type(container).__name__}"
            )
        self._container = container

    @property
    def container(self) -> LatentState:
        """The ``LatentState`` container this gate arbitrates."""
        return self._container

    def decide(self, candidates: Sequence[CandidateRecord]) -> TransactionDecision:
        """Evaluates candidates and returns a read-only decision.

        Performs the full input validation pass and deterministic selection; it
        does not mutate the container, the candidates, or any speculative buffer.
        """
        records = self._validate_collection(candidates)
        eligible = [(position, record) for position, record in enumerate(records) if record.verification.valid]

        if not eligible:
            return TransactionDecision(
                status=self.ABORT,
                selected=None,
                selected_id=None,
                ranked_candidate_ids=(),
                reason=(
                    f"all {len(records)} candidate(s) rejected by verification; "
                    "committed master state preserved"
                ),
            )

        ordered = sorted(eligible, key=lambda item: (-float(item[1].score), item[0]))
        winner = ordered[0][1]
        return TransactionDecision(
            status=self.COMMIT,
            selected=winner,
            selected_id=winner.candidate_id,
            ranked_candidate_ids=tuple(record.candidate_id for _, record in ordered),
            reason=(
                f"candidate {winner.candidate_id!r} selected at rank 1 of "
                f"{len(ordered)} eligible candidate(s) among {len(records)}"
            ),
        )

    def apply(self, decision: TransactionDecision) -> TransactionOutcome:
        """Finalizes a decision against the bound container.

        A ``COMMIT`` decision re-validates the winning candidate state and then
        commits it through the container contract (detach, clone into fresh
        storage, release speculative buffers). An ``ABORT`` decision rolls the
        container back to its pre-step snapshot and releases speculative buffers.
        Validation failures raise before any mutation, so the committed master
        state cannot be left partially updated.
        """
        if not isinstance(decision, TransactionDecision):
            raise TransactionGateError(
                f"decision must be a TransactionDecision, got {type(decision).__name__}"
            )

        if decision.status == self.COMMIT:
            winner = decision.selected
            if winner is None:
                raise TransactionGateError("COMMIT decision carries no selected candidate")
            self._validate_candidate_state(winner.state, candidate_id=winner.candidate_id)
            self._container.commit(winner.state)
            return TransactionOutcome(
                status=self.COMMIT,
                committed=True,
                selected_id=winner.candidate_id,
                reason=decision.reason,
            )

        if decision.status == self.ABORT:
            self._container.rollback()
            return TransactionOutcome(
                status=self.ABORT,
                committed=False,
                selected_id=None,
                reason=decision.reason,
            )

        raise TransactionGateError(f"unknown decision status {decision.status!r}")

    def decide_and_commit(self, candidates: Sequence[CandidateRecord]) -> TransactionOutcome:
        """Evaluates candidates and finalizes the transaction in one call."""
        return self.apply(self.decide(candidates))

    def _validate_collection(self, candidates: Sequence[CandidateRecord]) -> list[CandidateRecord]:
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TransactionGateError(
                "candidates must be a non-empty sequence of CandidateRecord"
            )
        if len(candidates) == 0:
            raise TransactionGateError("candidate collection must contain at least one record")

        seen_ids: set[Hashable] = set()
        records: list[CandidateRecord] = []
        for position, record in enumerate(candidates):
            if not isinstance(record, CandidateRecord):
                raise TransactionGateError(
                    f"candidate at position {position} is not a CandidateRecord "
                    f"(got {type(record).__name__})"
                )

            try:
                hash(record.candidate_id)
            except TypeError:
                raise TransactionGateError(
                    f"candidate id at position {position} is unhashable: {record.candidate_id!r}"
                ) from None
            if record.candidate_id in seen_ids:
                raise DuplicateCandidateError(
                    f"duplicate candidate id at position {position}: {record.candidate_id!r}"
                )
            seen_ids.add(record.candidate_id)

            if not isinstance(record.verification, VerificationOutcome):
                raise UnverifiedCandidateError(
                    f"candidate {record.candidate_id!r} has no VerificationOutcome "
                    f"(got {type(record.verification).__name__}); refusing to commit unverified state"
                )

            self._validate_score(record)

            self._validate_candidate_state(record.state, candidate_id=record.candidate_id)
            records.append(record)
        return records

    def _validate_score(self, record: CandidateRecord) -> None:
        score = record.score
        if score is None:
            if record.verification.valid:
                raise InvalidRankingScoreError(
                    f"valid candidate {record.candidate_id!r} requires a finite ranking score"
                )
            return
        if isinstance(score, bool) or not isinstance(score, Real):
            raise InvalidRankingScoreError(
                f"candidate {record.candidate_id!r} ranking score must be a finite real number, "
                f"got {type(score).__name__}"
            )
        if not math.isfinite(float(score)):
            raise InvalidRankingScoreError(
                f"candidate {record.candidate_id!r} ranking score is not finite: {score!r}"
            )

    def _validate_candidate_state(self, state: torch.Tensor, candidate_id: Optional[Hashable] = None) -> None:
        label = "candidate state" if candidate_id is None else f"candidate {candidate_id!r} state"
        if not isinstance(state, torch.Tensor):
            raise InvalidCandidateStateError(
                f"{label} must be a torch.Tensor, got {type(state).__name__}"
            )
        master_state = self._container.committed_state
        if state.shape != master_state.shape:
            raise InvalidCandidateStateError(
                f"{label} shape {tuple(state.shape)} does not match master shape "
                f"{tuple(master_state.shape)}"
            )
        if state.dtype != master_state.dtype:
            raise InvalidCandidateStateError(
                f"{label} dtype {state.dtype} does not match master dtype {master_state.dtype}"
            )
        if state.device != master_state.device:
            raise InvalidCandidateStateError(
                f"{label} device {state.device} does not match master device {master_state.device}"
            )
        if not bool(torch.isfinite(state).all()):
            raise InvalidCandidateStateError(f"{label} contains non-finite values")
