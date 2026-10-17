"""Unit tests for SpeculativeCandidateSet: isolated candidates, explicit commit, rollback, and lifecycle."""

import gc
import weakref

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from re_reason.state.container import LatentState
from re_reason.state.speculative import (
    SpeculativeCandidateSet,
    StateIntegrityError,
    StateLifecycleError,
)
from re_reason.utils.memory import inspect_tensor_retention


def _master(batch_size: int, dim: int, device: torch.device) -> LatentState:
    values = torch.arange(batch_size * dim, dtype=torch.float32, device=device).reshape(batch_size, dim)
    return LatentState(values / 7.0)


def _filled(batch_size: int, dim: int, device: torch.device, fill: float) -> torch.Tensor:
    return torch.full((batch_size, dim), fill, device=device)


def _assert_finalized(candidates: SpeculativeCandidateSet) -> None:
    master_shape = candidates.container.committed_state.shape
    operations = [
        lambda: candidates.state(0),
        lambda: candidates.update(0, torch.zeros(master_shape)),
        lambda: candidates.commit_candidate(0),
        lambda: candidates.rollback(),
        lambda: candidates.discard(),
        lambda: candidates.branch_states,
    ]
    for operation in operations:
        with pytest.raises(StateLifecycleError):
            operation()


def test_spawns_k_isolated_candidates_from_common_master(all_available_devices: torch.device) -> None:
    """Candidates start as equivalent copies of the master with distinct storage regions."""
    device = all_available_devices
    state = _master(4, 16, device)
    master_before = state.committed_state.clone()

    candidates = SpeculativeCandidateSet(state, k=3)

    assert candidates.k == 3
    assert candidates.lifecycle == SpeculativeCandidateSet.ACTIVE
    assert candidates.is_active
    assert candidates.branch_states.shape == (4, 3, 16)
    assert state.has_speculative
    assert state.snapshot_state is not None

    pointers = {state.committed_state.data_ptr()}
    for index in range(3):
        assert torch.equal(candidates.state(index), master_before)
        pointers.add(candidates.state(index).data_ptr())
    assert len(pointers) == 4, "master and every candidate must occupy distinct storage"


def test_candidate_mutation_isolated_from_master_and_siblings(all_available_devices: torch.device) -> None:
    """Mutating one candidate never modifies the committed master or sibling candidates."""
    device = all_available_devices
    state = _master(4, 16, device)
    master_before = state.committed_state.clone()
    candidates = SpeculativeCandidateSet(state, k=3)

    candidates.state(0).add_(100.0)
    assert torch.equal(state.committed_state, master_before)
    assert torch.equal(candidates.state(1), master_before)
    assert torch.equal(candidates.state(2), master_before)

    candidates.state(1).mul_(-2.0)
    assert torch.equal(state.committed_state, master_before)
    assert torch.equal(candidates.state(2), master_before)
    assert not torch.equal(candidates.state(0), candidates.state(1))


def test_update_writes_only_target_candidate(all_available_devices: torch.device) -> None:
    """Updating one candidate leaves the master and the other candidates unchanged."""
    device = all_available_devices
    state = _master(4, 16, device)
    master_before = state.committed_state.clone()
    candidates = SpeculativeCandidateSet(state, k=2)

    target = _filled(4, 16, device, 5.0)
    candidates.update(1, target)

    assert torch.equal(candidates.state(1), target)
    assert torch.equal(candidates.state(0), master_before)
    assert torch.equal(state.committed_state, master_before)


def test_commit_promotes_selected_candidate_with_fresh_storage(all_available_devices: torch.device) -> None:
    """Commit promotes only the selected candidate and isolates the master from the old buffer."""
    device = all_available_devices
    state = _master(4, 16, device)
    candidates = SpeculativeCandidateSet(state, k=3)
    winning = _filled(4, 16, device, 42.0)
    candidates.update(2, winning)
    retained_view = candidates.state(2)
    retained_view_pointer = retained_view.data_ptr()

    candidates.commit_candidate(2)

    assert candidates.lifecycle == SpeculativeCandidateSet.COMMITTED
    assert torch.equal(state.committed_state, winning)
    assert state.committed_state.data_ptr() != retained_view_pointer
    assert not state.has_speculative

    retained_view.add_(1000.0)
    assert torch.equal(state.committed_state, winning)
    _assert_finalized(candidates)


def test_commit_with_autograd_candidate_commits_detached(all_available_devices: torch.device) -> None:
    """Committing a candidate produced by autograd yields a detached committed master."""
    device = all_available_devices
    state = LatentState.zeros(2, 8, device=device)
    candidates = SpeculativeCandidateSet(state, k=2)

    weight = torch.nn.Parameter(torch.ones(8, device=device))
    graph_state = candidates.state(0) + weight
    assert graph_state.requires_grad
    candidates.update(0, graph_state)
    candidates.commit_candidate(0)

    retention = inspect_tensor_retention(state.committed_state)
    assert not retention["has_grad_fn"]
    assert not retention["requires_grad"]


def test_discard_preserves_master_exactly_and_finalizes(all_available_devices: torch.device) -> None:
    """Discarding candidates leaves the committed master bit-exact and finalizes the set."""
    device = all_available_devices
    state = _master(4, 16, device)
    master_before = state.committed_state.clone()
    candidates = SpeculativeCandidateSet(state, k=2)
    candidates.state(0).add_(7.0)
    candidates.state(1).sub_(3.0)

    candidates.discard()

    assert candidates.lifecycle == SpeculativeCandidateSet.DISCARDED
    assert torch.equal(state.committed_state, master_before)
    assert not state.has_speculative
    _assert_finalized(candidates)


def test_rollback_restores_master_bit_exact_and_finalizes(all_available_devices: torch.device) -> None:
    """Rollback restores the pre-step master state bit-exactly and finalizes the set."""
    device = all_available_devices
    state = _master(4, 16, device)
    master_before = state.committed_state.clone()
    candidates = SpeculativeCandidateSet(state, k=4)
    candidates.state(0).mul_(2.5)
    candidates.state(1).add_(999.0)
    candidates.state(2).fill_(float("nan"))
    candidates.state(3).fill_(float("inf"))

    candidates.rollback()

    assert candidates.lifecycle == SpeculativeCandidateSet.ROLLED_BACK
    assert torch.equal(state.committed_state, master_before)
    assert not state.has_speculative
    _assert_finalized(candidates)


def test_second_active_set_and_external_finalization_are_rejected(all_available_devices: torch.device) -> None:
    """A container supports one active candidate set; external finalization orphans the set."""
    device = all_available_devices
    state = _master(4, 16, device)
    candidates = SpeculativeCandidateSet(state, k=2)

    with pytest.raises(StateLifecycleError, match="already holds active speculative buffers"):
        SpeculativeCandidateSet(state, k=1)

    state.rollback()

    assert candidates.is_active
    with pytest.raises(StateLifecycleError, match="invalidated"):
        candidates.state(0)
    with pytest.raises(StateLifecycleError, match="invalidated"):
        candidates.discard()


def test_constructor_validation_rejects_invalid_configurations(all_available_devices: torch.device) -> None:
    """Invalid containers, branch counts, and non-flat masters raise explicit errors."""
    state = LatentState.zeros(2, 8)

    for invalid_k in (0, -1, 2.5, True, None):
        with pytest.raises(StateIntegrityError):
            SpeculativeCandidateSet(state, k=invalid_k)

    with pytest.raises(StateIntegrityError):
        SpeculativeCandidateSet(object(), k=2)

    non_flat = LatentState(torch.zeros(2, 4, 8))
    with pytest.raises(StateIntegrityError, match="flat"):
        SpeculativeCandidateSet(non_flat, k=2)

    assert state.has_speculative is False


def test_update_integrity_violations_raise_before_mutation(all_available_devices: torch.device) -> None:
    """Non-tensor, mis-shaped, mis-typed, mis-placed, and non-finite updates are rejected."""
    device = all_available_devices
    state = _master(4, 16, device)
    master_before = state.committed_state.clone()
    candidates = SpeculativeCandidateSet(state, k=2)

    with pytest.raises(StateIntegrityError):
        candidates.update(0, "not a tensor")
    with pytest.raises(StateIntegrityError):
        candidates.update(0, torch.zeros(4, 15, device=device))
    with pytest.raises(StateIntegrityError):
        candidates.update(0, torch.zeros(5, 16, device=device))
    with pytest.raises(StateIntegrityError):
        candidates.update(0, torch.zeros(4, 16, dtype=torch.float64, device=device))
    with pytest.raises(StateIntegrityError):
        candidates.update(0, torch.zeros(4, 16, device="meta"))
    with pytest.raises(StateIntegrityError):
        candidates.update(0, torch.full((4, 16), float("nan"), device=device))
    with pytest.raises(StateIntegrityError):
        candidates.update(0, torch.full((4, 16), float("inf"), device=device))

    assert torch.equal(state.committed_state, master_before)
    assert torch.equal(candidates.state(0), master_before)
    assert torch.equal(candidates.state(1), master_before)


def test_candidate_index_validation(all_available_devices: torch.device) -> None:
    """Candidate indices must be integers inside the spawned range."""
    device = all_available_devices
    state = _master(4, 16, device)
    candidates = SpeculativeCandidateSet(state, k=2)

    with pytest.raises(StateIntegrityError):
        candidates.state(2)
    with pytest.raises(StateIntegrityError):
        candidates.state(-1)
    with pytest.raises(StateIntegrityError):
        candidates.state(1.0)
    with pytest.raises(StateIntegrityError):
        candidates.state(True)

    valid = _filled(4, 16, device, 1.0)
    candidates.update(0, valid)
    assert torch.equal(candidates.state(0), valid)


@settings(max_examples=25, deadline=None)
@given(
    b=st.integers(min_value=1, max_value=8),
    d=st.integers(min_value=4, max_value=32),
    k=st.integers(min_value=1, max_value=6),
)
def test_property_candidate_isolation_and_rollback(b: int, d: int, k: int) -> None:
    """Property test: mutation isolation and bit-exact rollback across arbitrary shapes and k."""
    master = torch.randn(b, d)
    state = LatentState(master)
    candidates = SpeculativeCandidateSet(state, k=k)

    candidates.state(0).add_(10.0)
    assert torch.equal(state.committed_state, master)
    if k > 1:
        assert torch.equal(candidates.state(1), master)

    candidates.rollback()
    assert torch.equal(state.committed_state, master)
    assert not state.has_speculative


def test_repeated_cycles_do_not_retain_references() -> None:
    """1,000 spawn/finalize cycles: no speculative buffer outlives its cycle and no graphs accumulate.

    CPU-only loop using exact reference-death diagnostics (tensor weakrefs plus gc
    garbage inspection) rather than allocator reservation statistics.
    """
    state = LatentState.zeros(8, 256)
    buffer_references: list[weakref.ReferenceType] = []

    def cycle(index: int) -> weakref.ReferenceType:
        candidates = SpeculativeCandidateSet(state, k=3)
        reference = weakref.ref(candidates.branch_states)
        candidates.state(index % 3).add_(0.5)
        if index % 3 == 0:
            candidates.discard()
        elif index % 3 == 1:
            candidates.rollback()
        else:
            candidates.commit_candidate(index % 3)
        return reference

    for index in range(1000):
        buffer_references.append(cycle(index))

    gc.collect()
    alive = [reference for reference in buffer_references if reference() is not None]
    assert alive == [], f"{len(alive)} speculative buffers outlived their cycles"
    assert len(gc.garbage) == 0
    assert inspect_tensor_retention(state.committed_state)["has_grad_fn"] is False
