"""Abstract base contract for candidate transition and output verification."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import torch


class BaseVerifier(ABC):
    """Abstract base contract for candidate transition and output verifiers.

    Defines the verification signature. Deterministic constraint checks and learned
    critics belong strictly to Phase 5 and Phase 6.
    """

    @abstractmethod
    def verify(
        self, candidate: torch.Tensor, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Verifies candidate states or predictions against task constraints.

        Args:
            candidate: Candidate state or output tensor.
            observation: Observation embedding or input tensor.

        Returns:
            Tuple of:
                valid_mask: Boolean/binary tensor indicating valid (True) or invalid (False).
                diagnostics: Dictionary containing verifier scores or diagnostics.
        """
        pass
