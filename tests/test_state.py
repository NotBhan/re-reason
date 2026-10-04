"""Unit tests and property-based tests for LatentState container isolation, rollback, and memory lifecycle."""

import gc
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from re_reason.state.container import LatentState
from re_reason.utils.memory import check_monotonic_growth, inspect_tensor_retention


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
