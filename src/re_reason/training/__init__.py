"""Training package contracts and utilities."""

from re_reason.training.trainer import (
    EvalMetrics,
    compute_loss,
    evaluate_model,
    train_phase1_model,
)

__all__ = [
    "EvalMetrics",
    "compute_loss",
    "evaluate_model",
    "train_phase1_model",
]
