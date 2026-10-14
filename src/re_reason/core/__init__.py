"""Core package exports and models for RE:Reason."""

from re_reason.benchmarks.base import BaseTask, TaskBatch
from re_reason.cognition.base import BaseCognitionCore
from re_reason.cognition.core import CognitionCore
from re_reason.core.baseline_gru import GRUPhase1Model
from re_reason.core.model import (
    ModelOutput,
    ObservationEncoder,
    ReReasonPhase1Model,
    TypedDecisionHead,
)
from re_reason.state.container import BaseStateContainer, LatentState
from re_reason.verification.base import BaseVerifier

from re_reason.core.register_machine_models import (
    CyclicMGUPhase16Model,
    MSMDeterministicReference,
    Phase16ModelOutput,
    SlotMGUPhase16Model,
    VanillaGRUPhase16Model,
    cyclic_alu,
    get_harmonic_basis,
    project_to_unit_circle,
)

__all__ = [
    "BaseTask",
    "TaskBatch",
    "BaseStateContainer",
    "LatentState",
    "BaseCognitionCore",
    "CognitionCore",
    "BaseVerifier",
    "ObservationEncoder",
    "TypedDecisionHead",
    "ModelOutput",
    "ReReasonPhase1Model",
    "GRUPhase1Model",
    "SlotMGUPhase16Model",
    "CyclicMGUPhase16Model",
    "VanillaGRUPhase16Model",
    "MSMDeterministicReference",
    "Phase16ModelOutput",
    "cyclic_alu",
    "get_harmonic_basis",
    "project_to_unit_circle",
]
