"""Training package contracts and utilities."""

from re_reason.training.trainer import (
    EvalMetrics,
    compute_loss,
    evaluate_model,
    train_phase1_model,
)
from re_reason.training.behavioral_cloning import (
    BehavioralCloningLoss,
    Demonstration,
    build_demonstration,
    build_demonstrations,
    iter_training_batches,
    load_behavioral_cloning_demonstrations,
    teacher_forced_batch_loss,
)

__all__ = [
    "EvalMetrics",
    "compute_loss",
    "evaluate_model",
    "train_phase1_model",
    "BehavioralCloningLoss",
    "Demonstration",
    "build_demonstration",
    "build_demonstrations",
    "iter_training_batches",
    "load_behavioral_cloning_demonstrations",
    "teacher_forced_batch_loss",
]
