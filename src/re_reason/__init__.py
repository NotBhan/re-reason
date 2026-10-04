"""RE:Reason — Latent Recurrent Inference Architecture Research Framework.

Phase 0 Contracts & Scaffolding.
"""

from re_reason.benchmarks.base import BaseTask, TaskBatch

from re_reason.cognition.base import BaseCognitionCore
from re_reason.state.container import BaseStateContainer, LatentState
from re_reason.utils.seeding import set_seed
from re_reason.verification.base import BaseVerifier

__version__ = "0.1.0"

__all__ = [
    "BaseTask",
    "TaskBatch",
    "BaseStateContainer",
    "LatentState",
    "BaseCognitionCore",
    "BaseVerifier",
    "set_seed",
]
