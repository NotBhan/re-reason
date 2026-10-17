"""Unit tests verifying abstract contracts and deterministic seeding behavior."""

from typing import Any, Dict, Optional, Tuple
import pytest
import torch
import torch.nn as nn

from re_reason.benchmarks.base import BaseTask, TaskBatch
from re_reason.cognition.base import BaseCognitionCore
from re_reason.core import (
    BaseCognitionCore as ExportedCognitionCore,
    BaseStateContainer,
    BaseTask as ExportedTask,
    BaseVerifier,
    LatentState,
)
from re_reason.utils.seeding import set_seed
from re_reason.verification.base import BaseVerifier as DirectVerifier


def test_base_task_contract_enforcement() -> None:
    """Verifies that BaseTask cannot be instantiated without implementing abstract methods."""
    with pytest.raises(TypeError):
        # Incomplete implementation missing generate_batch and evaluate
        class IncompleteTask(BaseTask):
            pass

        IncompleteTask()

    # Valid implementation satisfies contract
    class MockTask(BaseTask):
        def generate_batch(self, batch_size: int, seed: Optional[int] = None) -> TaskBatch:
            return TaskBatch(
                inputs=torch.zeros(batch_size, 10),
                targets=torch.ones(batch_size, 4),
            )

        def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
            return {"accuracy": 1.0}

    task = MockTask()
    batch = task.generate_batch(batch_size=2)
    assert batch.inputs.shape == (2, 10)
    assert batch.targets.shape == (2, 4)
    eval_res = task.evaluate(batch.inputs, batch.targets)
    assert eval_res["accuracy"] == 1.0


def test_base_cognition_core_contract_enforcement() -> None:
    """Verifies that BaseCognitionCore behaves as an nn.Module and enforces forward method."""
    with pytest.raises(TypeError):
        class IncompleteCore(BaseCognitionCore):
            pass

        IncompleteCore()

    class MockCore(BaseCognitionCore):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(32, 32)

        def forward(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
            return self.linear(state + observation)

    core = MockCore()
    assert isinstance(core, nn.Module)
    s = torch.randn(2, 32)
    obs = torch.randn(2, 32)
    out = core(s, obs)
    assert out.shape == (2, 32)


def test_base_verifier_contract_enforcement() -> None:
    """Verifies that BaseVerifier enforces verify method."""
    with pytest.raises(TypeError):
        class IncompleteVerifier(BaseVerifier):
            pass

        IncompleteVerifier()

    class MockVerifier(BaseVerifier):
        def verify(
            self, candidate: torch.Tensor, observation: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            valid_mask = torch.ones(candidate.shape[0], dtype=torch.bool)
            return valid_mask, {"score": 1.0}

    verifier = MockVerifier()
    c = torch.randn(4, 64)
    o = torch.randn(4, 64)
    mask, diags = verifier.verify(c, o)
    assert mask.shape == (4,)
    assert diags["score"] == 1.0


def test_deterministic_seeding_reproducibility() -> None:
    """Verifies that set_seed ensures reproducible random numbers across generators."""
    set_seed(12345)
    t1 = torch.randn(10, 10)

    set_seed(12345)
    t2 = torch.randn(10, 10)

    assert torch.equal(t1, t2)

    # Different seed produces different tensors
    set_seed(54321)
    t3 = torch.randn(10, 10)
    assert not torch.equal(t1, t3)


def test_core_package_exports() -> None:
    """Verifies that all Phase 0 contracts are cleanly accessible from the root/core package."""
    assert ExportedTask is BaseTask
    assert ExportedCognitionCore is BaseCognitionCore
    assert DirectVerifier is BaseVerifier
    assert issubclass(LatentState, BaseStateContainer)


def test_state_container_contract_includes_checkpoint_restoration() -> None:
    """The state-container interface exposes checkpoint restoration for recovery."""
    assert "restore_checkpoint" in BaseStateContainer.__abstractmethods__
    assert callable(getattr(LatentState, "restore_checkpoint"))
