"""Speculative candidate-set lifecycle over the committed latent state container.

Milestone 1 (Speculative State Isolation): the committed master state remains
authoritative for transition ``t`` while ``K >= 1`` independently stored candidate
working states are created from a single master-state snapshot, mutated in
isolation from the master and from each other, and finalized only through an
explicit commit, rollback, or discard operation.

Isolation builds directly on the existing ``LatentState`` contract without
modifying it: ``spawn_branches`` clones the committed master into a ``(B, K, D)``
buffer with one independent storage region per candidate (``torch.clone()`` of a
repeated tensor), ``commit`` detaches and clones the selected candidate into
fresh master storage, and ``rollback``/``clear_speculative`` release speculative
references. This module adds explicit candidate-level lifecycle and integrity
semantics on top of those primitives; the container implementation and its
public interface are unchanged.

Scope: flat ``(B, D)`` master states, matching the Phase 1 locked state
representation (``S_t in R^{B x D}``). Non-flat masters are rejected explicitly.

Gradient semantics: candidate access returns views into the detached speculative
buffer, and candidate updates use the same in-place copy semantics as the
container's established branch-mutation pattern. No additional graph detachment
is introduced; ``LatentState.commit`` continues to detach the committed state by
the existing container contract.
"""

from typing import Optional

import torch

from re_reason.state.container import LatentState


class StateLifecycleError(RuntimeError):
    """Raised on invalid speculative candidate-set lifecycle transitions."""


class StateIntegrityError(ValueError):
    """Raised when a candidate tensor or constructor argument violates state invariants."""


class SpeculativeCandidateSet:
    """``K`` isolated working candidates derived from one committed master snapshot.

    Lifecycle states:

    * ``ACTIVE``: candidates exist and may be accessed, updated, committed,
      rolled back, or discarded.
    * ``COMMITTED``: one candidate was promoted to the committed master through
      ``commit_candidate``; the set is finalized.
    * ``ROLLED_BACK``: the container was rolled back to its pre-step snapshot;
      the set is finalized.
    * ``DISCARDED``: speculative buffers were released without touching the
      committed master; the set is finalized.

    Every terminal transition is explicit and irreversible for the set instance:
    any further candidate operation raises :class:`StateLifecycleError`. If the
    underlying container is finalized externally (for example, by the transaction
    commit gate committing or rolling back), the set is orphaned and subsequent
    operations raise :class:`StateLifecycleError` as well.

    Candidate isolation follows the container contract: each candidate occupies
    its own storage region inside the ``(B, K, D)`` speculative buffer, the
    committed master state is never aliased by candidate views, and committing
    one candidate clones its values into fresh master storage before the
    speculative buffer is released.
    """

    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    DISCARDED = "discarded"

    def __init__(self, container: LatentState, k: int) -> None:
        """Creates ``k`` isolated candidates from the container's committed master state.

        Args:
            container: The ``LatentState`` owning the committed master state. The
                container must not already hold active speculative buffers.
            k: Number of independent speculative candidates (``k >= 1``).

        Raises:
            StateIntegrityError: If ``container`` is not a ``LatentState``, ``k``
                is not an integer of at least one, or the master state is not a
                flat ``(B, D)`` tensor.
            StateLifecycleError: If the container already holds active
                speculative buffers from another producer.
        """
        if not isinstance(container, LatentState):
            raise StateIntegrityError(
                f"container must be a LatentState instance, got {type(container).__name__}"
            )
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise StateIntegrityError(f"branch count k must be an integer >= 1, got {k!r}")

        master_state = container.committed_state
        if master_state.ndim != 2:
            raise StateIntegrityError(
                "Speculative candidate sets support flat (B, D) master states "
                f"(Phase 1 state contract); got shape {tuple(master_state.shape)}"
            )
        if container.has_speculative:
            raise StateLifecycleError(
                "container already holds active speculative buffers; "
                "discard or finalize them before spawning a new candidate set"
            )

        self._container = container
        self._branches: torch.Tensor = container.spawn_branches(k)
        self._k = k
        self._lifecycle = self.ACTIVE

    @property
    def container(self) -> LatentState:
        """The ``LatentState`` container this candidate set operates on."""
        return self._container

    @property
    def k(self) -> int:
        """Number of speculative candidates in this set."""
        return self._k

    @property
    def lifecycle(self) -> str:
        """Current lifecycle state of the set."""
        return self._lifecycle

    @property
    def is_active(self) -> bool:
        """True while the set is active and its candidates may be used."""
        return self._lifecycle == self.ACTIVE

    @property
    def branch_states(self) -> torch.Tensor:
        """The ``(B, K, D)`` speculative working buffer of all candidates."""
        self._require_active()
        return self._branches

    def state(self, index: int) -> torch.Tensor:
        """Returns the ``(B, D)`` working state of candidate ``index``.

        The returned tensor is a view into the speculative buffer: in-place
        mutation affects only this candidate's storage region, never the
        committed master state or sibling candidates.
        """
        self._require_active()
        index = self._validated_index(index)
        return self._branches[:, index, :]

    def update(self, index: int, state: torch.Tensor) -> None:
        """Copies ``state`` into candidate ``index`` in place after integrity validation.

        The update uses the container's established in-place branch-assignment
        semantics. It does not detach, cast, or move the supplied tensor: shape,
        dtype, and device must already match the committed master state, and all
        values must be finite.
        """
        self._require_active()
        index = self._validated_index(index)
        self._validate_candidate_tensor(state)
        self._branches[:, index, :].copy_(state)

    def commit_candidate(self, index: int) -> None:
        """Promotes candidate ``index`` to the committed master state.

        Delegates to the container contract (detach, clone into fresh storage,
        release speculative buffers) and finalizes this set as ``COMMITTED``.
        """
        self._require_active()
        index = self._validated_index(index)
        self._container.commit(index)
        self._lifecycle = self.COMMITTED

    def rollback(self) -> None:
        """Restores the container's pre-step snapshot bit-exactly and finalizes the set.

        Delegates to ``LatentState.rollback``: the committed master state is
        restored from the pre-step snapshot and speculative buffers are released.
        """
        self._require_active()
        self._container.rollback()
        self._lifecycle = self.ROLLED_BACK

    def discard(self) -> None:
        """Releases speculative candidates without touching the committed master state.

        Delegates to ``LatentState.clear_speculative``; the committed master
        state is left exactly as it was.
        """
        self._require_active()
        self._container.clear_speculative()
        self._lifecycle = self.DISCARDED

    def _require_active(self) -> None:
        if self._lifecycle != self.ACTIVE:
            raise StateLifecycleError(
                f"candidate set is {self._lifecycle}; only an active set supports this operation"
            )
        if self._container.working_branches is not self._branches:
            raise StateLifecycleError(
                "candidate set was invalidated by an external container finalization"
            )

    def _validated_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise StateIntegrityError(f"candidate index must be an integer, got {index!r}")
        if index < 0 or index >= self._k:
            raise StateIntegrityError(f"candidate index {index} out of range for k={self._k}")
        return index

    def _validate_candidate_tensor(self, state: torch.Tensor) -> None:
        if not isinstance(state, torch.Tensor):
            raise StateIntegrityError(
                f"candidate state must be a torch.Tensor, got {type(state).__name__}"
            )
        master_state = self._container.committed_state
        if state.shape != master_state.shape:
            raise StateIntegrityError(
                f"candidate state shape {tuple(state.shape)} does not match master shape "
                f"{tuple(master_state.shape)}"
            )
        if state.dtype != master_state.dtype:
            raise StateIntegrityError(
                f"candidate state dtype {state.dtype} does not match master dtype {master_state.dtype}"
            )
        if state.device != master_state.device:
            raise StateIntegrityError(
                f"candidate state device {state.device} does not match master device {master_state.device}"
            )
        if not bool(torch.isfinite(state).all()):
            raise StateIntegrityError("candidate state contains non-finite values")
