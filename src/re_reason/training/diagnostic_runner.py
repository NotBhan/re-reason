"""Diagnostic training, evaluation, and logging framework for RE:Reason Phase 1.

Tracks detailed learning curves, evaluates strictly on validation for model selection,
and tests final models on independent, held-out test splits.
"""

import copy
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from re_reason.benchmarks.dataset_split import DisjointSplitManager, SplitDataset
from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.utils.seeding import set_seed


class TrajectoryPoint(NamedTuple):
    step: int
    train_loss: float
    val_loss: float
    train_exact: float
    val_exact: float
    train_per_reg: float
    val_per_reg: float


class DiagnosticSummary(NamedTuple):
    model_name: str
    seed: int
    params: int
    total_steps: int
    wall_clock_s: float
    best_val_exact: float
    best_val_step: int
    final_train_exact: float
    final_train_loss: float
    final_val_exact: float
    final_val_loss: float
    final_test_exact: float
    final_test_loss: float
    final_test_per_reg: float
    step_wise_test_exact: List[float]
    inference_latency_ms: float
    peak_memory_mb: float
    trajectory: List[TrajectoryPoint]


def evaluate_split(
    model: nn.Module,
    dataset: SplitDataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    num_samples: Optional[int] = None,
) -> Tuple[float, float, float, List[float], float]:
    """Evaluates model on a specific SplitDataset.

    Returns:
        (mean_loss, exact_match, per_reg_acc, step_exact_matches, latency_ms_per_sample)
    """
    dev = device if device is not None else next(model.parameters()).device
    model.eval()

    total_samples = min(len(dataset), num_samples if num_samples is not None else len(dataset))
    num_batches = (total_samples + batch_size - 1) // batch_size

    total_loss = 0.0
    total_exact = 0
    total_per_reg = 0.0
    seq_len = dataset.inputs.shape[1]
    step_exact_accum = [0.0] * seq_len

    latencies: List[float] = []

    with torch.no_grad():
        for b_idx in range(num_batches):
            b_size = min(batch_size, total_samples - b_idx * batch_size)
            if b_size <= 0:
                break

            batch = dataset.get_batch(b_size, idx=b_idx)
            inputs = batch.inputs.to(dev)
            targets = batch.targets.to(dev)
            step_targets = batch.metadata["step_targets"].to(dev)

            t0 = time.perf_counter()
            out = model(inputs, return_step_diagnostics=True)
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 / b_size)

            loss = F.cross_entropy(out.logits.view(-1, 10), targets.view(-1))
            total_loss += loss.item() * b_size

            preds = out.logits.argmax(dim=-1)
            total_exact += (preds == targets).all(dim=-1).sum().item()
            total_per_reg += (preds == targets).float().sum().item() / targets.shape[1]

            if out.step_logits is not None:
                step_preds = out.step_logits.argmax(dim=-1)
                for t in range(seq_len):
                    step_exact_accum[t] += (
                        (step_preds[:, t, :] == step_targets[:, t, :])
                        .all(dim=-1)
                        .sum()
                        .item()
                    )

    mean_loss = total_loss / total_samples
    exact_match = total_exact / total_samples
    per_reg_acc = total_per_reg / total_samples
    step_exacts = [s / total_samples for s in step_exact_accum]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return mean_loss, exact_match, per_reg_acc, step_exacts, avg_latency


def run_diagnostic_training(
    model: nn.Module,
    splits: DisjointSplitManager,
    total_steps: int = 1200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    gradient_clip_norm: float = 1.0,
    device: Optional[torch.device] = None,
    seed: int = 42,
    eval_interval: int = 200,
    supervise_stepwise: bool = False,
) -> DiagnosticSummary:
    """Executes a diagnostic training run with trajectory recording and test evaluation."""
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    set_seed(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    trajectory: List[TrajectoryPoint] = []
    best_val_exact = -1.0
    best_val_step = -1
    best_state_dict = None

    start_time = time.perf_counter()
    train_set = splits.train_set
    val_set = splits.val_set
    test_set = splits.test_set

    for step in range(1, total_steps + 1):
        model.train()
        batch = train_set.get_batch(batch_size, idx=step)
        inputs = batch.inputs.to(dev)
        targets = batch.targets.to(dev)

        optimizer.zero_grad()

        if supervise_stepwise:
            out = model(inputs, return_step_diagnostics=True)
            step_targets = batch.metadata["step_targets"].to(dev)
            # Loss across all time steps
            b, t, n_reg, n_val = out.step_logits.shape
            step_loss = F.cross_entropy(
                out.step_logits.view(-1, n_val),
                step_targets.view(-1),
            )
            term_loss = F.cross_entropy(out.logits.view(-1, n_val), targets.view(-1))
            loss = 0.5 * (step_loss + term_loss)
        else:
            # Official Phase 1: Terminal supervision strictly at t=10
            out = model(inputs, return_step_diagnostics=False)
            loss = F.cross_entropy(out.logits.view(-1, 10), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        # Validation checkpointing and trajectory logging
        if step % eval_interval == 0 or step == total_steps:
            model.eval()
            with torch.no_grad():
                # Train batch metrics
                train_preds = out.logits.argmax(dim=-1)
                train_exact = (train_preds == targets).all(dim=-1).float().mean().item()
                train_per_reg = (train_preds == targets).float().mean().item()

                # Validation split metrics (1,000 samples)
                val_loss, val_exact, val_per_reg, _, _ = evaluate_split(
                    model, val_set, batch_size=batch_size, device=dev, num_samples=1000
                )

                trajectory.append(
                    TrajectoryPoint(
                        step=step,
                        train_loss=loss.item(),
                        val_loss=val_loss,
                        train_exact=train_exact,
                        val_exact=val_exact,
                        train_per_reg=train_per_reg,
                        val_per_reg=val_per_reg,
                    )
                )

                if val_exact > best_val_exact:
                    best_val_exact = val_exact
                    best_val_step = step
                    best_state_dict = copy.deepcopy(model.state_dict())

    wall_clock = time.perf_counter() - start_time

    # Final evaluation using the best validation model on the strictly held-out TEST set
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_exact, test_per_reg, step_exacts, latency_ms = evaluate_split(
        model, test_set, batch_size=batch_size, device=dev
    )

    final_point = trajectory[-1] if trajectory else TrajectoryPoint(0, 0, 0, 0, 0, 0, 0)
    peak_mem_mb = 0.0
    if dev.type == "cuda" and torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return DiagnosticSummary(
        model_name=getattr(model, "__class__", type(model)).__name__,
        seed=seed,
        params=params,
        total_steps=total_steps,
        wall_clock_s=wall_clock,
        best_val_exact=best_val_exact,
        best_val_step=best_val_step,
        final_train_exact=final_point.train_exact,
        final_train_loss=final_point.train_loss,
        final_val_exact=final_point.val_exact,
        final_val_loss=final_point.val_loss,
        final_test_exact=test_exact,
        final_test_loss=test_loss,
        final_test_per_reg=test_per_reg,
        step_wise_test_exact=step_exacts,
        inference_latency_ms=latency_ms,
        peak_memory_mb=peak_mem_mb,
        trajectory=trajectory,
    )
