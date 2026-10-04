"""Latent state container contracts and implementation using isolated tensor cloning."""

from abc import ABC, abstractmethod
from typing import Optional, Union
import torch


class BaseStateContainer(ABC):
    """Abstract base contract for latent state containers.

    Manages persistent committed master state and speculative candidate branches
    using isolated tensor cloning.
    """

    @property
    @abstractmethod
    def committed_state(self) -> torch.Tensor:
        """Returns the persistent master state tensor."""
        pass

    @property
    @abstractmethod
    def working_branches(self) -> Optional[torch.Tensor]:
        """Returns the active speculative working branches tensor, or None if inactive."""
        pass

    @abstractmethod
    def snapshot(self) -> torch.Tensor:
        """Takes an isolated snapshot of the committed master state."""
        pass

    @abstractmethod
    def spawn_branches(self, k: int) -> torch.Tensor:
        """Spawns k speculative candidate working branches from the committed master state.

        Uses isolated tensor cloning to ensure complete memory isolation between
        branches and the persistent master state.
        """
        pass

    @abstractmethod
    def commit(self, candidate: Union[int, torch.Tensor]) -> None:
        """Atomically commits a candidate state to become the new persistent master state.

        Detaches computation graphs and releases speculative working buffers.
        """
        pass

    @abstractmethod
    def rollback(self) -> None:
        """Rolls back any speculative computation, leaving committed master state unchanged."""
        pass

    @abstractmethod
    def clear_speculative(self) -> None:
        """Releases speculative tensor buffers to allow memory reclamation by the caching allocator."""
        pass


class LatentState(BaseStateContainer):
    """Minimal latent state container with isolated tensor cloning and rollback semantics.

    Important implementation properties:
    - State isolation is implemented via isolated tensor cloning (`torch.clone()`).
      PyTorch does not natively provide hardware-level copy-on-write memory paging.
    - Speculative candidates operate in `_working_branches`.
    - Rollback guarantees bit-exact preservation of `_committed_state`.
    - Commits call `.detach()` to prevent infinite autograd graph chaining across steps.
    """

    def __init__(self, initial_state: torch.Tensor) -> None:
        """Initializes the latent state with an initial committed tensor.

        Args:
            initial_state: Tensor representing initial state, typically of shape (B, D).
        """
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError(f"initial_state must be a torch.Tensor, got {type(initial_state)}")

        # Isolate committed master state from external references
        self._committed_state: torch.Tensor = initial_state.detach().clone()
        self._snapshot_state: Optional[torch.Tensor] = None
        self._working_branches: Optional[torch.Tensor] = None

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        dim: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "LatentState":
        """Factory method creating a zero-initialized LatentState.

        Args:
            batch_size: Batch dimension B.
            dim: Latent dimension D.
            device: Target device (CPU or ROCm/CUDA).
            dtype: Tensor data type. Defaults to torch.float32.
        """
        state_tensor = torch.zeros((batch_size, dim), device=device, dtype=dtype)
        return cls(state_tensor)

    @property
    def committed_state(self) -> torch.Tensor:
        """Returns the committed persistent master state tensor."""
        return self._committed_state

    @property
    def working_branches(self) -> Optional[torch.Tensor]:
        """Returns the speculative working branches tensor, or None."""
        return self._working_branches

    @property
    def snapshot_state(self) -> Optional[torch.Tensor]:
        """Returns the active snapshot tensor, or None."""
        return self._snapshot_state

    @property
    def has_speculative(self) -> bool:
        """Returns True if speculative working branches are currently allocated."""
        return self._working_branches is not None

    @property
    def device(self) -> torch.device:
        """Device of the committed master state."""
        return self._committed_state.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the committed master state."""
        return self._committed_state.dtype

    @property
    def shape(self) -> torch.Size:
        """Shape of the committed master state."""
        return self._committed_state.shape

    @property
    def batch_size(self) -> int:
        """Batch size dimension of committed master state."""
        return self._committed_state.shape[0]

    @property
    def dim(self) -> int:
        """Latent dimension D of committed master state."""
        return self._committed_state.shape[-1]

    def snapshot(self) -> torch.Tensor:
        """Takes an isolated snapshot of the committed master state.

        Uses isolated tensor cloning (`torch.clone()`) to create an independent buffer.
        """
        self._snapshot_state = self._committed_state.detach().clone()
        return self._snapshot_state.clone()

    def spawn_branches(self, k: int) -> torch.Tensor:
        """Spawns k speculative candidate working branches from the master state.

        Uses isolated tensor cloning to guarantee that mutations to candidate branches
        do not propagate to the committed master state or across sibling branches.

        Args:
            k: Number of candidate branches to instantiate (k >= 1).

        Returns:
            Working branches tensor of shape (B, K, D).
        """
        if k < 1:
            raise ValueError(f"Branch count k must be >= 1, got {k}")

        # Automatically capture a snapshot before speculative branching if none exists
        if self._snapshot_state is None:
            self._snapshot_state = self._committed_state.detach().clone()

        # Isolated tensor cloning: creates independent memory buffer for each candidate branch
        # Shape: (B, D) -> (B, 1, D) -> repeat to (B, K, D) -> clone()
        if self._committed_state.ndim == 2:
            self._working_branches = (
                self._committed_state.unsqueeze(1).repeat(1, k, 1).clone()
            )
        else:
            expand_shape = list(self._committed_state.shape)
            expand_shape.insert(len(expand_shape) - 1, k)
            self._working_branches = (
                self._committed_state.unsqueeze(-2).expand(*expand_shape).clone()
            )

        return self._working_branches

    def commit(self, candidate: Union[int, torch.Tensor]) -> None:
        """Atomically commits a candidate state to become the new persistent master state.

        Calls `.detach()` to prevent infinite autograd graph chaining across steps,
        and releases all speculative working memory buffers.

        Args:
            candidate: Either an integer branch index k_idx or a candidate tensor.
        """
        if isinstance(candidate, int):
            if self._working_branches is None:
                raise RuntimeError("Cannot commit by branch index when no working branches exist.")
            k = self._working_branches.shape[1]
            if candidate < 0 or candidate >= k:
                raise IndexError(f"Candidate branch index {candidate} out of bounds for k={k}")

            new_master = self._working_branches[:, candidate, :].detach().clone()
        elif isinstance(candidate, torch.Tensor):
            if candidate.shape != self._committed_state.shape:
                raise ValueError(
                    f"Candidate tensor shape {candidate.shape} does not match master shape {self._committed_state.shape}"
                )
            new_master = candidate.detach().clone()
        else:
            raise TypeError(f"candidate must be an int or torch.Tensor, got {type(candidate)}")

        self._committed_state = new_master
        self.clear_speculative()

    def rollback(self) -> None:
        """Rolls back any speculative computation, leaving committed master state unchanged.

        Reverts to the exact bit-level pre-step snapshot (or retains unchanged master state)
        and clears speculative working buffers.
        """
        if self._snapshot_state is not None:
            self._committed_state = self._snapshot_state.detach().clone()
        self.clear_speculative()

    def clear_speculative(self) -> None:
        """Releases all speculative working references to prevent memory leaks and graph retention."""
        self._working_branches = None
        self._snapshot_state = None
