"""Deterministic single-transition verifier for constrained pathfinding episodes.

Verification level (explicit)
-----------------------------
This verifier checks **single transitions**: whether a proposed movement from a
given task state is an allowed transition under the episode's authoritative
rules. It deliberately does not validate complete proposed paths; that separate
concern belongs to
``re_reason.benchmarks.pathfinding_benchmark.validate_path``. The three levels
are kept distinct: transition verification (here), state-transition consistency
(action/state agreement, checked here), and complete-path validation (oracle
module).

Result contract
---------------
The verifier consumes the existing Milestone 1 result contract
(:class:`~re_reason.verification.transaction_gate.VerificationOutcome`), so its
outcomes can feed ``CandidateRecord`` transaction records without adapters. It
does not implement :class:`~re_reason.verification.base.BaseVerifier`: that
contract is a batched tensor interface (``verify(candidate, observation) ->
(valid_mask, diagnostics)``) aimed at learned pipelines, while deterministic
transition checking operates on discrete graph structures; forcing a tensor
encoding would require artificial adapters and is deferred to the
model-integration milestone.

Guarantees
----------
The verifier is deterministic and side-effect free: it never mutates the episode,
task states, or a committed master state. Diagnostics expose exactly the checked
transition and a reason code, and distinguish the main rejection causes
(``unknown_node``, ``state_mismatch``, ``no_such_edge``, ``blocked_transition``).
No learned scores, heuristics, or confidence claims are involved.
"""

from re_reason.benchmarks.pathfinding_benchmark import (
    TRANSITION_OK,
    TRANSITION_STATE_MISMATCH,
    PathfindingAction,
    PathfindingEpisode,
    PathfindingState,
)
from re_reason.verification.transaction_gate import VerificationOutcome


class PathfindingVerifier:
    """Deterministic transition verifier bound to one authoritative episode.

    The verifier answers whether a proposed movement from a given task state is
    an allowed transition, using the episode's own authoritative classification
    (``transition_reason``) as the single source of truth. State/action agreement
    is checked first: an action that starts at a different node than the provided
    state is rejected as ``state_mismatch`` regardless of its target.
    """

    def __init__(self, episode: PathfindingEpisode) -> None:
        if not isinstance(episode, PathfindingEpisode):
            raise TypeError(f"episode must be a PathfindingEpisode, got {type(episode).__name__}")
        self._episode = episode

    @property
    def episode(self) -> PathfindingEpisode:
        """The authoritative episode this verifier checks transitions against."""
        return self._episode

    def verify_transition(
        self, state: PathfindingState, action: PathfindingAction
    ) -> VerificationOutcome:
        """Verifies one proposed transition and returns a structured outcome.

        Accepted transitions yield ``VerificationOutcome(valid=True, ...)`` with
        reason ``ok``. Rejections yield ``valid=False`` with a stable reason code
        and the checked ``current``/``proposed`` node pair, without revealing any
        state beyond the checked transition.
        """
        if not isinstance(state, PathfindingState):
            raise TypeError(f"state must be a PathfindingState, got {type(state).__name__}")
        if not isinstance(action, PathfindingAction):
            raise TypeError(f"action must be a PathfindingAction, got {type(action).__name__}")

        if action.current != state.current:
            reason = TRANSITION_STATE_MISMATCH
        else:
            reason = self._episode.transition_reason(state.current, action.target)

        return VerificationOutcome(
            valid=reason == TRANSITION_OK,
            details={
                "current": state.current,
                "proposed": action.target,
                "reason": reason,
            },
        )
