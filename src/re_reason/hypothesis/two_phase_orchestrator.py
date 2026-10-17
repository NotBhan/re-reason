"""Two-phase speculative transition orchestration for Milestone 3A.

Coordinates one pathfinding decision point end to end:

Phase 1 (prepare — committed latent and environment state remain untouched):

1. Build the public observation for the current environment state.
2. Encode it and generate up to ``K = 3`` deterministic candidate actions and
   successor latent tensors with the shared :class:`ActionConditionedBrancher`.
3. Populate a :class:`SpeculativeCandidateSet` (isolated working buffers spawned
   from the committed latent state) with the candidate states.
4. Verify every candidate transition with the independent
   :class:`PathfindingVerifier`.
5. Build :class:`CandidateRecord` objects and request the read-only decision from
   :class:`TransactionCommitGate`.
6. Preview the selected transition with the benchmark's pure transition
   semantics and validate the prospective environment state.

Phase 2 (commit — the only mutating phase):

1. Capture an external detached checkpoint of the pre-transition committed state.
2. Apply the selected candidate through the gate's commit path.
3. Assign the precomputed immutable next environment state.
4. Finalize candidate lifecycle state and release any remaining speculative
   buffers.

Failure handling
----------------
Any exception raised during Phase 2 restores the environment reference to its
pre-transition value and restores the committed latent state bit-exactly from the
external checkpoint via :meth:`LatentState.restore_checkpoint`, then re-raises
the original exception. Phase-1 exceptions release speculative buffers and
propagate without any committed-state mutation. Clean aborts (no eligible action,
or exhaustion of the bounded proposal-round budget :math:`R_{max} = 3`) return
``TransitionResult(committed=False, ...)`` and never mutate either state.

Known limitation: if recovery itself fails (for example, if
``restore_checkpoint`` raises after a post-mutation commit failure), full
atomicity cannot be guaranteed and the original exception remains the active
exception context. Multi-step backtracking junction snapshots
(:math:`B_{max} = 4`) are declared for the future backtracking milestone but are
not implemented here.

Proposal policy
---------------
The neural brancher proposes only actions that are legal at the current node
(derived from the public observation) and not already attempted at this decision
point. Verification remains an independent authoritative check on every proposal:
a candidate rejected by verification is never committed, and its action is masked
for subsequent proposal rounds. Legal transitions into dead ends are committed
normally — dead ends are not illegal actions and are never masked for that reason.

Compute scope: this orchestrator is inference-only and runs the neural brancher
under ``torch.no_grad()``. No training, optimizer, loss, or checkpoint-selection
logic is implemented in this milestone.
"""

from dataclasses import dataclass
from typing import Optional, Set

import torch

from re_reason.benchmarks.pathfinding_benchmark import (
    PathfindingAction,
    PathfindingEpisode,
    PathfindingState,
)
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher
from re_reason.state.container import LatentState
from re_reason.state.speculative import SpeculativeCandidateSet
from re_reason.verification.pathfinding_verifier import PathfindingVerifier
from re_reason.verification.transaction_gate import (
    CandidateRecord,
    TransactionCommitGate,
    TransactionDecision,
    TransactionGateError,
)


STATUS_COMMITTED = "committed"
STATUS_ABORTED = "aborted"

MAX_PROPOSAL_ROUNDS = 3
MAX_EPISODE_TRANSITIONS = 24
MAX_BACKTRACKING_SNAPSHOTS = 4
CANDIDATE_BUDGET = 3


@dataclass(frozen=True)
class TransitionResult:
    """Outcome of one orchestrated decision point.

    A committed transition reports the selected action id; an aborted decision
    reports no selected action and never changed the latent or environment state.
    """

    committed: bool
    status: str
    selected_action_id: Optional[int]
    rounds_attempted: int
    reason: str

    def __post_init__(self) -> None:
        if self.committed and self.selected_action_id is None:
            raise ValueError("a committed transition must report its selected action id")
        if not self.committed and self.selected_action_id is not None:
            raise ValueError("an aborted transition must not report a selected action id")


@dataclass(frozen=True)
class _PreparedTransition:
    candidate_set: SpeculativeCandidateSet
    decision: TransactionDecision
    next_state: PathfindingState
    rounds_attempted: int


class TwoPhaseTransitionOrchestrator:
    """Coordinates branching, verification, gated commit, and environment advance.

    The orchestrator is bound to one immutable episode, one ``LatentState``
    container (shape ``(1, state_dim)``), and one brancher; it creates its own
    verifier (bound to the episode) and transaction gate (bound to the container).
    Both the container object and the gate binding are preserved across recovery.
    """

    MAX_PROPOSAL_ROUNDS = MAX_PROPOSAL_ROUNDS
    MAX_EPISODE_TRANSITIONS = MAX_EPISODE_TRANSITIONS
    MAX_BACKTRACKING_SNAPSHOTS = MAX_BACKTRACKING_SNAPSHOTS
    CANDIDATE_BUDGET = CANDIDATE_BUDGET

    def __init__(
        self,
        episode: PathfindingEpisode,
        container: LatentState,
        brancher: ActionConditionedBrancher,
        environment_state: Optional[PathfindingState] = None,
    ) -> None:
        if not isinstance(episode, PathfindingEpisode):
            raise TypeError(f"episode must be a PathfindingEpisode, got {type(episode).__name__}")
        if not isinstance(container, LatentState):
            raise TypeError(f"container must be a LatentState, got {type(container).__name__}")
        if not isinstance(brancher, ActionConditionedBrancher):
            raise TypeError(
                f"brancher must be an ActionConditionedBrancher, got {type(brancher).__name__}"
            )
        if episode.num_nodes != brancher.num_nodes:
            raise ValueError(
                f"episode has {episode.num_nodes} nodes but the brancher expects "
                f"{brancher.num_nodes}"
            )
        expected_state_shape = (1, brancher.state_dim)
        if container.committed_state.shape != expected_state_shape:
            raise ValueError(
                f"container committed state shape {tuple(container.committed_state.shape)} does "
                f"not match the required {expected_state_shape}"
            )

        state = episode.initial_state() if environment_state is None else environment_state
        if not isinstance(state, PathfindingState):
            raise TypeError(
                f"environment_state must be a PathfindingState, got {type(state).__name__}"
            )
        if not 0 <= state.current < episode.num_nodes:
            raise ValueError(
                f"environment_state.current {state.current} is out of range for "
                f"{episode.num_nodes} nodes"
            )

        self._episode = episode
        self._container = container
        self._brancher = brancher
        self._environment_state = state
        self._verifier = PathfindingVerifier(episode)
        self._gate = TransactionCommitGate(container)

    @property
    def episode(self) -> PathfindingEpisode:
        """The immutable ground-truth episode."""
        return self._episode

    @property
    def container(self) -> LatentState:
        """The committed latent-state container."""
        return self._container

    @property
    def brancher(self) -> ActionConditionedBrancher:
        """The neural candidate generator."""
        return self._brancher

    @property
    def verifier(self) -> PathfindingVerifier:
        """The authoritative deterministic transition verifier."""
        return self._verifier

    @property
    def gate(self) -> TransactionCommitGate:
        """The transaction commit gate bound to the container."""
        return self._gate

    @property
    def environment_state(self) -> PathfindingState:
        """The current immutable environment state."""
        return self._environment_state

    def step(self) -> TransitionResult:
        """Executes one two-phase transition attempt.

        Returns a committed or cleanly aborted :class:`TransitionResult`. Raises
        the original exception (after recovery) for any failure in the commit
        phase, and propagates Phase-1 failures after releasing speculative state.
        """
        self._require_transition_budget()
        prepared = self._prepare()
        if isinstance(prepared, TransitionResult):
            return prepared

        checkpoint = self._container.committed_state.detach().clone()
        previous_environment_state = self._environment_state
        try:
            self._gate.apply(prepared.decision)
            self._advance_environment(prepared.next_state)
            self._finalize_candidates(prepared.candidate_set)
        except BaseException:
            self._environment_state = previous_environment_state
            self._container.restore_checkpoint(checkpoint)
            raise

        return TransitionResult(
            committed=True,
            status=STATUS_COMMITTED,
            selected_action_id=prepared.decision.selected_id,
            rounds_attempted=prepared.rounds_attempted,
            reason=prepared.decision.reason,
        )

    def _require_transition_budget(self) -> None:
        if self._environment_state.transitions >= self.MAX_EPISODE_TRANSITIONS:
            raise RuntimeError(
                f"episode transition budget exhausted: {self._environment_state.transitions} "
                f"of {self.MAX_EPISODE_TRANSITIONS} transitions already completed"
            )

    def _prepare(self) -> "TransitionResult | _PreparedTransition":
        observation = self._episode.observe(self._environment_state)
        attempted: Set[int] = set()
        rounds_attempted = 0

        for round_index in range(self.MAX_PROPOSAL_ROUNDS):
            rounds_attempted = round_index + 1
            with torch.no_grad():
                proposal = self._brancher.propose_candidates(
                    self._container.committed_state,
                    observation,
                    num_candidates=self.CANDIDATE_BUDGET,
                    masked_actions=tuple(sorted(attempted)),
                )
            if not proposal.candidates:
                return TransitionResult(
                    committed=False,
                    status=STATUS_ABORTED,
                    selected_action_id=None,
                    rounds_attempted=round_index,
                    reason="no eligible candidate action remains at this decision point",
                )

            candidate_set = SpeculativeCandidateSet(
                self._container, k=len(proposal.candidates)
            )
            try:
                for index, candidate in enumerate(proposal.candidates):
                    candidate_set.update(index, candidate.state)

                records = []
                for index, candidate in enumerate(proposal.candidates):
                    outcome = self._verifier.verify_transition(
                        self._environment_state,
                        PathfindingAction(
                            current=self._environment_state.current, target=candidate.action_id
                        ),
                    )
                    records.append(
                        CandidateRecord(
                            candidate_id=candidate.action_id,
                            state=candidate_set.state(index),
                            verification=outcome,
                            score=candidate.score,
                            metadata={"round_index": round_index, "policy_rank": index},
                        )
                    )

                decision = self._gate.decide(records)
                if decision.status == TransactionCommitGate.COMMIT:
                    if decision.selected_id is None:
                        raise TransactionGateError("COMMIT decision carries no selected candidate")
                    next_state = self._preview_environment_transition(decision.selected_id)
                    return _PreparedTransition(
                        candidate_set=candidate_set,
                        decision=decision,
                        next_state=next_state,
                        rounds_attempted=rounds_attempted,
                    )
                if decision.status != TransactionCommitGate.ABORT:
                    raise TransactionGateError(
                        f"unexpected transaction gate decision status {decision.status!r}"
                    )
            except BaseException:
                self._release_candidates(candidate_set)
                raise

            self._release_candidates(candidate_set)
            attempted.update(candidate.action_id for candidate in proposal.candidates)

        return TransitionResult(
            committed=False,
            status=STATUS_ABORTED,
            selected_action_id=None,
            rounds_attempted=rounds_attempted,
            reason=(
                f"proposal retry budget exhausted after {self.MAX_PROPOSAL_ROUNDS} bounded "
                "round(s) at this decision point"
            ),
        )

    def _preview_environment_transition(self, action_id: int) -> PathfindingState:
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise TransactionGateError(
                f"selected candidate id must be an integer, got {action_id!r}"
            )
        action = PathfindingAction(current=self._environment_state.current, target=action_id)
        next_state = self._episode.apply_transition(self._environment_state, action)
        if (
            next_state.current != action_id
            or next_state.transitions != self._environment_state.transitions + 1
        ):
            raise RuntimeError("environment preview is inconsistent with the transition contract")
        return next_state

    def _advance_environment(self, next_state: PathfindingState) -> None:
        if not isinstance(next_state, PathfindingState):
            raise TypeError(f"next_state must be a PathfindingState, got {type(next_state).__name__}")
        self._environment_state = next_state

    @staticmethod
    def _release_candidates(candidate_set: Optional[SpeculativeCandidateSet]) -> None:
        if candidate_set is None or not candidate_set.is_active:
            return
        container = candidate_set.container
        if not container.has_speculative:
            return
        try:
            candidate_set.discard()
        except BaseException:
            container.clear_speculative()
            raise

    @staticmethod
    def _finalize_candidates(candidate_set: Optional[SpeculativeCandidateSet]) -> None:
        if candidate_set is not None and candidate_set.is_active and candidate_set.container.has_speculative:
            candidate_set.discard()
