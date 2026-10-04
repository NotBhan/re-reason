"""Phase 1 Training and Evaluation Engine.

Provides unified training, evaluation, and diagnostic tracking for both
ReReasonPhase1Model and GRUPhase1Model under strict parameter and compute parity.
"""

import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.utils.seeding import set_seed


class EvalMetrics(NamedTuple):
    """Container for evaluation results."""

    final_exact_match: float  # Exact match across all registers at t=10
    final_per_reg_acc: float  # Per-register accuracy at t=10
    step_exact_matches: List[float]  # Exact match accuracy at each step t=1..10 (diagnostic)
    mean_loss: float
    inference_latency_ms: float
    peak_memory_mb: float


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes terminal multi-class cross entropy across all registers.

    Args:
        logits: Tensor of shape (B, num_registers, num_values).
        targets: LongTensor of shape (B, num_registers).

    Returns:
        Scalar loss tensor.
    """
    b, num_regs, num_vals = logits.shape
    return F.cross_entropy(logits.view(-1, num_vals), targets.view(-1))


def evaluate_model(
    model: nn.Module,
    task: RegisterTrackingTask,
    num_eval_samples: int = 1000,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    eval_seed: int = 9999,
) -> EvalMetrics:
    """Evaluates model strictly on unseen test batches with fixed seed."""
    dev = device if device is not None else next(model.parameters()).device
    model.eval()

    total_exact_match = 0
    total_per_reg_acc = 0.0
    total_loss = 0.0
    total_samples = 0

    num_steps = task.sequence_length
    step_exact_accum = [0.0] * num_steps

    num_batches = (num_eval_samples + batch_size - 1) // batch_size

    latencies: List[float] = []

    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        for b_idx in range(num_batches):
            current_b = min(batch_size, num_eval_samples - total_samples)
            if current_b <= 0:
                break

            batch = task.generate_batch(current_b, seed=eval_seed + b_idx * 17)
            inputs = batch.inputs.to(dev)
            targets = batch.targets.to(dev)
            step_targets = batch.metadata["step_targets"].to(dev)

            t0 = time.perf_counter()
            out = model(inputs, return_step_diagnostics=True)
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 / current_b)

            loss = compute_loss(out.logits, targets)
            total_loss += loss.item() * current_b

            pred_terminal = out.logits.argmax(dim=-1)
            exact = (pred_terminal == targets).all(dim=-1).sum().item()
            per_reg = (pred_terminal == targets).float().sum().item() / task.num_registers

            total_exact_match += exact
            total_per_reg_acc += per_reg
            total_samples += current_b

            # Step-wise diagnostic accuracy
            if out.step_logits is not None:
                step_preds = out.step_logits.argmax(dim=-1)  # (B, T, 4)
                for t in range(num_steps):
                    step_exact = (
                        (step_preds[:, t, :] == step_targets[:, t, :])
                        .all(dim=-1)
                        .sum()
                        .item()
                    )
                    step_exact_accum[t] += step_exact

    mean_exact = total_exact_match / total_samples
    mean_per_reg = total_per_reg_acc / total_samples
    mean_loss = total_loss / total_samples
    mean_step_exact = [s / total_samples for s in step_exact_accum]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    peak_mem_mb = 0.0
    if dev.type == "cuda" and torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    return EvalMetrics(
        final_exact_match=mean_exact,
        final_per_reg_acc=mean_per_reg,
        step_exact_matches=mean_step_exact,
        mean_loss=mean_loss,
        inference_latency_ms=avg_latency_ms,
        peak_memory_mb=peak_mem_mb,
    )


def train_phase1_model(
    model: nn.Module,
    task: RegisterTrackingTask,
    total_steps: int = 1200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    gradient_clip_norm: float = 1.0,
    device: Optional[torch.device] = None,
    seed: int = 42,
    log_interval: int = 200,
) -> Tuple[EvalMetrics, float]:
    """Trains a Phase 1 model using AdamW with terminal supervision.

    Args:
        model: ReReasonPhase1Model or GRUPhase1Model.
        task: RegisterTrackingTask generator.
        total_steps: Number of optimizer steps.
        batch_size: Batch size B.
        learning_rate: Optimizer learning rate.
        weight_decay: AdamW weight decay.
        gradient_clip_norm: Maximum gradient norm.
        device: Execution device.
        seed: Seed for reproducibility.
        log_interval: Logging frequency.

    Returns:
        Tuple of (final_eval_metrics, wall_clock_seconds).
    """
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    set_seed(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    start_time = time.perf_counter()
    model.train()

    for step in range(1, total_steps + 1):
        batch = task.generate_batch(batch_size, seed=seed * 10000 + step)
        inputs = batch.inputs.to(dev)
        targets = batch.targets.to(dev)

        optimizer.zero_grad()
        out = model(inputs, return_step_diagnostics=False)
        loss = compute_loss(out.logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        if step % log_interval == 0 or step == total_steps:
            with torch.no_grad():
                preds = out.logits.argmax(dim=-1)
                exact_acc = (preds == targets).all(dim=-1).float().mean().item()
            # print diagnostic progress if needed

    wall_clock = time.perf_counter() - start_time

    # Final evaluation on holdout set
    eval_metrics = evaluate_model(
        model=model,
        task=task,
        num_eval_samples=1000,
        batch_size=batch_size,
        device=dev,
        eval_seed=seed + 50000,
    )

    return eval_metrics, wall_clock
