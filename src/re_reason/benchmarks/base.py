"""Abstract base contract for procedural and synthetic benchmark tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional
import torch


class TaskBatch(NamedTuple):
    """Container for a single batch of task inputs, targets, and optional metadata."""

    inputs: torch.Tensor
    targets: torch.Tensor
    metadata: Dict[str, Any] = {}


class BaseTask(ABC):
    """Abstract base contract for synthetic and procedural tasks.

    In Phase 0, this defines the interface signature.
    Concrete benchmark task implementations (e.g. Register Tracking) belong to Phase 1+.
    """

    @abstractmethod
    def generate_batch(self, batch_size: int, seed: Optional[int] = None) -> TaskBatch:
        """Generates a batch of synthetic task instances.

        Args:
            batch_size: Number of parallel task sequences to generate.
            seed: Optional integer seed for procedural reproducibility.

        Returns:
            TaskBatch containing inputs, ground-truth targets, and metadata.
        """
        pass

    @abstractmethod
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluates model predictions against ground truth targets.

        Args:
            predictions: Model output tensor.
            targets: Ground-truth target tensor.

        Returns:
            Dictionary of scalar evaluation metrics (e.g., accuracy, error rate).
        """
        pass
