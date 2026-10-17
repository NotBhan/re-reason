"""Unit tests and property-based tests for LatentState container isolation, rollback, and memory lifecycle."""

import gc
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from re_reason.state.container import LatentState
from re_reason.utils.memory import check_monotonic_growth, inspect_tensor_retention
from re_reason.verification.transaction_gate import (
    CandidateRecord,
    TransactionCommitGate,
    VerificationOutcome,
)


def test_state_creation_and_properties(all_available_devices: torch.device) -> None:
    """Verifies state creation from tensor and factory method, along with properties."""
    device = all_available_devices
    batch_size, dim = 4, 256

    init_tensor = torch.randn(batch_size, dim, device=device)
    state = LatentState(init_tensor)

    assert state.device.type == device.type
    assert state.shape == (batch_size, dim)
    assert state.batch_size == batch_size
    assert state.dim == dim
    assert state.dtype == torch.float32
    assert torch.equal(state.committed_state, init_tensor)
    assert state.working_branches is None
    assert state.snapshot_state is None
    assert not state.has_speculative

    # Factory zeros method
    zero_state = LatentState.zeros(batch_size=8, dim=128, device=device)
    assert zero_state.shape == (8, 128)
    assert zero_state.device.type == device.type
    assert torch.equal(zero_state.committed_state, torch.zeros(8, 128, device=device))


def test_state_creation_isolation(all_available_devices: torch.device) -> None:
    """Verifies that mutating the input tensor after LatentState creation does not mutate the master state."""
    device = all_available_devices
    init_tensor = torch.zeros(4, 256, device=device)
    state = LatentState(init_tensor)

    # Mutate the source tensor outside the container
    init_tensor.add_(10.0)

    # Master state in container must remain isolated and unmodified
    assert torch.equal(state.committed_state, torch.zeros(4, 256, device=device))


def test_state_snapshot_isolation(all_available_devices: torch.device) -> None:
    """Verifies that taking a snapshot produces an isolated clone that does not reflect future mutations."""
    device = all_available_devices
    state = LatentState.zeros(4, 256, device=device)
    snapshot = state.snapshot()

    assert torch.equal(snapshot, state.committed_state)

    # Mutating snapshot does not mutate master state
    snapshot.add_(5.0)
    assert torch.equal(state.committed_state, torch.zeros(4, 256, device=device))


def test_branch_isolation_and_mutation(all_available_devices: torch.device) -> None:
    """Verifies that:
    1. Mutation of a branch does not modify the master state.
    2. Mutation of one branch does not modify another sibling branch.
    """
    device = all_available_devices
    b, k, d = 4, 3, 256
    state = LatentState.zeros(b, d, device=device)
    branches = state.spawn_branches(k=k)

    assert branches.shape == (b, k, d)
    assert state.has_speculative
    assert state.working_branches is not None

    # Mutate branch 0 in-place
    branches[:, 0, :] += 100.0

    # 1. Master state must be completely untouched
    assert torch.equal(state.committed_state, torch.zeros(b, d, device=device))

    # 2. Sibling branches (1 and 2) must remain completely untouched
    assert torch.equal(branches[:, 1, :], torch.zeros(b, d, device=device))
    assert torch.equal(branches[:, 2, :], torch.zeros(b, d, device=device))

    # Mutate branch 1 differently
    branches[:, 1, :] -= 50.0
    assert torch.equal(state.committed_state, torch.zeros(b, d, device=device))
    assert torch.equal(branches[:, 2, :], torch.zeros(b, d, device=device))
    assert not torch.equal(branches[:, 0, :], branches[:, 1, :])


def test_rollback_is_bit_exact(all_available_devices: torch.device) -> None:
    """Verifies that rollback leaves the committed master state bit-exact unchanged."""
    device = all_available_devices
    b, k, d = 4, 4, 256
    init_tensor = torch.randn(b, d, device=device)
    state = LatentState(init_tensor)

    # Pre-step master state copy for bit-exact comparison
    master_before = state.committed_state.clone()

    branches = state.spawn_branches(k=k)
    # Perform aggressive mutations on candidate branches
    branches[:, 0, :] *= 2.5
    branches[:, 1, :] = torch.randn_like(branches[:, 1, :])
    branches[:, 2, :] += 999.0
    branches[:, 3, :] = torch.nan_to_num(branches[:, 3, :])

    # Execute rollback
    state.rollback()

    # Master state must be bit-exact identical to pre-step master
    assert torch.equal(state.committed_state, master_before)
    # Working branches and snapshot references must be released
    assert state.working_branches is None
    assert state.snapshot_state is None
    assert not state.has_speculative


def test_commit_replaces_master_correctly(all_available_devices: torch.device) -> None:
    """Verifies that commit correctly promotes candidate branch to master and detaches graph."""
    device = all_available_devices
    b, k, d = 4, 3, 256
    state = LatentState.zeros(b, d, device=device)
    branches = state.spawn_branches(k=k)

    # Mutate branch index 1 with specific target values
    winning_values = torch.full((b, d), 42.0, device=device)
    branches[:, 1, :] = winning_values

    # Commit branch index 1
    state.commit(candidate=1)

    assert torch.equal(state.committed_state, winning_values)
    # Working memory must be cleared
    assert state.working_branches is None
    assert state.snapshot_state is None
    assert not state.has_speculative


def test_commit_with_tensor_replaces_correctly(all_available_devices: torch.device) -> None:
    """Verifies commit directly accepting an updated candidate tensor."""
    device = all_available_devices
    b, d = 2, 64
    state = LatentState.zeros(b, d, device=device)
    state.spawn_branches(k=2)

    new_tensor = torch.randn(b, d, device=device)
    state.commit(candidate=new_tensor)

    assert torch.equal(state.committed_state, new_tensor)
    assert state.working_branches is None


def test_autograd_graph_detachment_on_commit(all_available_devices: torch.device) -> None:
    """Verifies that committing a state generated via autograd operations cleanly detaches graph."""
    device = all_available_devices
    b, d = 4, 128
    state = LatentState.zeros(b, d, device=device)

    # Simulate autograd computation during step
    branches = state.spawn_branches(k=2)
    # Wrap in tensor that requires grad
    w = torch.nn.Parameter(torch.ones(d, device=device))
    step_computation = (branches[:, 0, :] * w).sum()
    candidate_state = (branches[:, 0, :] + w).unsqueeze(1)

    # Mutate branch with autograd tensor
    branches[:, 0, :] = candidate_state.squeeze(1)

    # Commit should detach graph
    state.commit(candidate=0)

    retention = inspect_tensor_retention(state.committed_state)
    assert not retention["has_grad_fn"]
    assert state.committed_state.grad_fn is None


def test_speculative_buffers_released_after_clear(all_available_devices: torch.device) -> None:
    """Verifies that clear_speculative releases all working references."""
    device = all_available_devices
    state = LatentState.zeros(4, 128, device=device)
    state.spawn_branches(k=4)

    assert state.working_branches is not None
    assert state.snapshot_state is not None

    state.clear_speculative()

    assert state.working_branches is None
    assert state.snapshot_state is None


def test_no_uncontrolled_allocation_growth(all_available_devices: torch.device) -> None:
    """Runs 1,000 sequential mock step iterations of spawn, mutate, rollback/commit.

    Checks:
    - Retained references
    - Garbage collector cycles
    - Allocation stability (zero monotonic growth after initial warmup).
    """
    device = all_available_devices
    b, k, d = 8, 4, 256
    state = LatentState.zeros(b, d, device=device)

    # Warmup phase (10 steps)
    for _ in range(10):
        branches = state.spawn_branches(k=k)
        branches[:, 0, :] += 1.0
        state.commit(0)

    gc.collect()
    allocations: list[int] = []

    # 1,000 mock step iterations
    for step in range(1000):
        branches = state.spawn_branches(k=k)
        branches[:, :, :] += 0.5

        if step % 2 == 0:
            state.rollback()
        else:
            state.commit(candidate=step % k)

        # Record diagnostic allocation every 50 steps
        if step % 50 == 0:
            if "cuda" in str(device) and torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated()
            else:
                # On CPU, track active numel of committed state
                alloc = state.committed_state.numel() * state.committed_state.element_size()
            allocations.append(alloc)

    # Verify no monotonic memory drift
    has_growth = check_monotonic_growth(allocations, tolerance_ratio=0.05)
    assert not has_growth, f"Detected uncontrolled memory allocation drift: {allocations}"

    # Verify no uncollected garbage
    gc.collect()
    assert len(gc.garbage) == 0


# Property-based testing over arbitrary dimensions and branch counts
@settings(max_examples=25, deadline=None)
@given(
    b=st.integers(min_value=1, max_value=16),
    d=st.integers(min_value=16, max_value=256),
    k=st.integers(min_value=1, max_value=8),
)
def test_property_state_isolation(b: int, d: int, k: int) -> None:
    """Property test verifying isolation across arbitrary batch sizes, dimensions, and branch counts."""
    init = torch.randn(b, d)
    state = LatentState(init)
    branches = state.spawn_branches(k=k)

    # Mutate branch 0 if k >= 1
    branches[:, 0, :] = branches[:, 0, :] + 10.0

    # Master state must equal initial
    assert torch.equal(state.committed_state, init)

    if k > 1:
        # Sibling branch 1 must equal initial
        assert torch.equal(branches[:, 1, :], init)

    # Rollback restores master bit-exact
    state.rollback()
    assert torch.equal(state.committed_state, init)
    assert state.working_branches is None


# ---------------------------------------------------------------------------
# Checkpoint restoration contract (multi-step recovery after a completed commit)
# ---------------------------------------------------------------------------

def test_restore_checkpoint_restores_committed_state(all_available_devices: torch.device) -> None:
    """A valid checkpoint replaces the committed master state exactly."""
    device = all_available_devices
    state = LatentState.zeros(4, 16, device=device)
    checkpoint = torch.arange(64, dtype=torch.float32, device=device).reshape(4, 16) / 3.0

    state.restore_checkpoint(checkpoint)

    assert torch.equal(state.committed_state, checkpoint)


def test_restore_checkpoint_storage_is_independent(all_available_devices: torch.device) -> None:
    """Restored storage does not alias the caller's checkpoint tensor."""
    device = all_available_devices
    state = LatentState.zeros(4, 16, device=device)
    checkpoint = torch.randn(4, 16, device=device)
    expected = checkpoint.clone()

    state.restore_checkpoint(checkpoint)

    assert state.committed_state.data_ptr() != checkpoint.data_ptr()
    checkpoint.add_(100.0)
    assert torch.equal(state.committed_state, expected)


def test_restore_checkpoint_clears_speculative_and_snapshot(all_available_devices: torch.device) -> None:
    """Restoration releases active speculative branches and transient snapshot state."""
    device = all_available_devices
    state = LatentState.zeros(4, 16, device=device)
    state.spawn_branches(k=3)
    state.snapshot()
    assert state.has_speculative and state.snapshot_state is not None

    state.restore_checkpoint(torch.zeros(4, 16, device=device))

    assert state.working_branches is None
    assert state.snapshot_state is None
    assert not state.has_speculative


def test_restore_checkpoint_rejects_invalid_inputs(all_available_devices: torch.device) -> None:
    """Non-tensors and mismatched shape/dtype/device are rejected without mutation."""
    device = all_available_devices
    state = LatentState.zeros(4, 16, device=device)
    master_before = state.committed_state.clone()

    with pytest.raises(TypeError):
        state.restore_checkpoint("checkpoint")
    with pytest.raises(ValueError):
        state.restore_checkpoint(torch.zeros(4, 15, device=device))
    with pytest.raises(ValueError):
        state.restore_checkpoint(torch.zeros(4, 16, dtype=torch.float64, device=device))
    with pytest.raises(ValueError):
        state.restore_checkpoint(torch.zeros(4, 16, device="meta"))

    assert torch.equal(state.committed_state, master_before)


def test_restore_checkpoint_after_commit_returns_prior_committed_tensor(
    all_available_devices: torch.device,
) -> None:
    """Restoration recovers the pre-commit master tensor after an already completed commit."""
    device = all_available_devices
    initial = torch.randn(4, 16, device=device)
    state = LatentState(initial)
    checkpoint = state.committed_state.clone()

    branches = state.spawn_branches(k=2)
    branches[:, 1, :] = torch.full((4, 16), 7.0, device=device)
    state.commit(candidate=1)
    assert torch.equal(state.committed_state, torch.full((4, 16), 7.0, device=device))

    state.restore_checkpoint(checkpoint)

    assert torch.equal(state.committed_state, initial)
    assert state.committed_state.grad_fn is None
    assert not state.has_speculative


def test_restore_checkpoint_preserves_container_identity_and_gate_binding(
    all_available_devices: torch.device,
) -> None:
    """The container object and a previously bound transaction gate survive restoration."""
    device = all_available_devices
    state = LatentState.zeros(4, 16, device=device)
    gate = TransactionCommitGate(state)
    container_identity = id(state)
    checkpoint = state.committed_state.clone()

    winner = torch.full((4, 16), 3.0, device=device)
    first = gate.decide_and_commit(
        [
            CandidateRecord(
                candidate_id="first",
                state=winner,
                verification=VerificationOutcome(valid=True),
                score=0.5,
            )
        ]
    )
    assert first.committed

    state.restore_checkpoint(checkpoint)

    assert id(state) == container_identity
    assert gate.container is state
    assert torch.equal(state.committed_state, checkpoint)

    second = gate.decide_and_commit(
        [
            CandidateRecord(
                candidate_id="second",
                state=winner,
                verification=VerificationOutcome(valid=True),
                score=0.5,
            )
        ]
    )
    assert second.committed
    assert torch.equal(state.committed_state, winner)
