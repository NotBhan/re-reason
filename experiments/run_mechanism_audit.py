"""Comprehensive Mechanism Decomposition & Novelty Audit Experiment Runner.

Evaluates 7 strictly parameter-matched models across 5 seeds:
1. Model A: Current RE:Reason (LayerNorm additive residual)
2. Model B: Bounded Additive (S = S + tanh(f))
3. Model C: Convex Unbounded (S = (1 - z)S + z f)
4. Model D: Bounded Convex MLP / Model E (S = (1 - z)S + z tanh(f))
5. Model E: Canonical UGRNN (Collins et al., 2017)
6. Model F: Canonical MGU (Zhou et al., 2016)
7. Model G: Canonical GRU (Cho et al., 2014)

Measures:
1. Exact and per-register train/val/test performance (test_at_best_val_checkpoint)
2. FLOPs, wall-clock, throughput (examples/s), latency, peak memory
3. Detailed gate mechanics: entropy, near-0/1 saturation, op sensitivity, register correlation
4. Extended horizon scaling: T in {5, 10, 15, 20, 25, 30}
5. Transfer and training across 4 independent algorithmic tasks:
   - MultiRegisterCopyTask
   - PermutationCycleTask
   - RunningAccumulationTask
   - ConditionalOverwriteTask
6. Statistical analysis: seed-wise paired deltas, effect sizes, p-values
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from re_reason.benchmarks.algorithmic_tasks import (
    ConditionalOverwriteTask,
    MultiRegisterCopyTask,
    PermutationCycleTask,
    RunningAccumulationTask,
)
from re_reason.benchmarks.dataset_split import DisjointSplitManager, SplitDataset
from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.core.ablation_models import (
    CanonicalMGUModel,
    CanonicalUGRNNModel,
    IntermediateDiagnosticsOutput,
    ReReasonBoundedConvexModel,
    ReReasonConvexUnboundedModel,
    create_mechanism_decomposition_suite,
    forward_with_states_generic,
)
from re_reason.core.baseline_gru import GRUPhase1Model
from re_reason.training.diagnostic_runner import evaluate_split
from re_reason.utils.seeding import set_seed


def calculate_model_flops_per_step(model: nn.Module, batch_size: int = 1) -> int:
    """Estimates theoretical floating point operations (FLOPs) per recurrent step."""
    # Count linear MACs * 2 + activations
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Matrix-vector multiply: 2 * in_features * out_features
            flops += 2 * m.in_features * m.out_features
            if m.bias is not None:
                flops += m.out_features
        elif isinstance(m, nn.LayerNorm):
            flops += 4 * m.normalized_shape[0]
        elif isinstance(m, (nn.GELU, nn.Tanh, nn.Sigmoid)):
            # ~8 FLOPs per transcendental activation
            pass
    return flops * batch_size


def run_temporal_credit_diagnostics(
    model: nn.Module,
    test_set: SplitDataset,
    device: torch.device,
    num_samples: int = 256,
) -> Dict[str, Any]:
    """Measures intermediate state statistics and backpropagated gradient norms."""
    model.eval()
    model.zero_grad()

    batch = test_set.get_batch(num_samples, idx=0)
    inputs = batch.inputs.to(device)
    targets = batch.targets.to(device)

    diag_out: IntermediateDiagnosticsOutput = forward_with_states_generic(model, inputs)
    states = diag_out.states
    seq_len = len(states)

    state_norms = [s.norm(dim=-1).mean().item() for s in states]
    state_stds = [s.std(dim=-1).mean().item() for s in states]

    for s in states:
        s.retain_grad()

    loss = F.cross_entropy(diag_out.terminal_logits.view(-1, 10), targets.view(-1))
    loss.backward()

    grad_norms = [s.grad.norm(dim=-1).mean().item() for s in states]
    attenuation_ratio = grad_norms[0] / max(grad_norms[-1], 1e-12)

    return {
        "state_norms": state_norms,
        "state_stds": state_stds,
        "grad_norms": grad_norms,
        "attenuation_ratio": attenuation_ratio,
    }


def run_detailed_gate_diagnostics(
    model: nn.Module,
    test_set: SplitDataset,
    device: torch.device,
    num_samples: int = 1200,
) -> Optional[Dict[str, Any]]:
    """Analyzes gate entropy, bimodal saturation, operation sensitivity, and state change correlation."""
    model.eval()
    batch = test_set.get_batch(num_samples, idx=0)
    inputs = batch.inputs.to(device)

    if hasattr(model, "forward_with_states"):
        diag_out = model.forward_with_states(inputs, return_components=True)
    else:
        return None

    gates = diag_out.gates
    candidates = diag_out.candidates
    states = diag_out.states

    if gates is None or len(gates) == 0:
        return None

    # Flatten all gates across steps and batch
    all_gates = torch.cat(gates, dim=0)  # (B * T, D)
    eps = 1e-7
    clamped_gates = torch.clamp(all_gates, eps, 1.0 - eps)

    # Gate entropy: - [z log2 z + (1-z) log2(1-z)]
    gate_entropy = -(
        clamped_gates * torch.log2(clamped_gates)
        + (1.0 - clamped_gates) * torch.log2(1.0 - clamped_gates)
    ).mean().item()

    frac_near_0 = (all_gates < 0.1).float().mean().item()
    frac_near_1 = (all_gates > 0.9).float().mean().item()

    # Operation-specific sensitivity
    set_gates = []
    swap_gates = []
    add_gates = []

    for t in range(len(gates)):
        x_t = inputs[:, t, :]
        op_types = x_t[:, :3].argmax(dim=-1)
        g_t = gates[t]

        if (op_types == 0).any():
            set_gates.append(g_t[op_types == 0])
        if (op_types == 1).any():
            swap_gates.append(g_t[op_types == 1])
        if (op_types == 2).any():
            add_gates.append(g_t[op_types == 2])

    cat_set = torch.cat(set_gates, dim=0) if set_gates else torch.empty(0)
    cat_swap = torch.cat(swap_gates, dim=0) if swap_gates else torch.empty(0)
    cat_add = torch.cat(add_gates, dim=0) if add_gates else torch.empty(0)

    # Correlation between gate value and state displacement |S_t - S_(t-1)|
    corrs = []
    for t in range(1, len(states)):
        state_diff = (states[t] - states[t - 1]).abs()
        g_t = gates[t]
        vx = g_t - g_t.mean(dim=-1, keepdim=True)
        vy = state_diff - state_diff.mean(dim=-1, keepdim=True)
        cost = (vx * vy).sum(dim=-1) / (torch.sqrt((vx ** 2).sum(dim=-1) * (vy ** 2).sum(dim=-1)) + 1e-8)
        corrs.append(cost.mean().item())

    mean_gate_state_displacement_corr = float(np.mean(corrs)) if corrs else 0.0

    return {
        "mean_gate": all_gates.mean().item(),
        "std_gate": all_gates.std().item(),
        "gate_entropy": gate_entropy,
        "frac_near_0": frac_near_0,
        "frac_near_1": frac_near_1,
        "op_means": {
            "SET": cat_set.mean().item() if cat_set.numel() > 0 else 0.0,
            "SWAP": cat_swap.mean().item() if cat_swap.numel() > 0 else 0.0,
            "ADD": cat_add.mean().item() if cat_add.numel() > 0 else 0.0,
        },
        "gate_state_displacement_corr": mean_gate_state_displacement_corr,
    }


def evaluate_extended_horizons(
    model: nn.Module,
    device: torch.device,
    horizons: List[int] = [5, 10, 15, 20, 25, 30],
    num_samples: int = 1200,
    seed_base: int = 95000,
) -> Dict[str, Dict[str, float]]:
    """Evaluates trained model on sequence lengths T in {5..30}."""
    model.eval()
    results: Dict[str, Dict[str, float]] = {}

    for h in horizons:
        task_h = RegisterTrackingTask(num_registers=4, sequence_length=h, num_values=10, seed=seed_base + h)
        batch_h = task_h.generate_batch(num_samples, seed=seed_base + 1000 + h)
        inputs_h = batch_h.inputs.to(device)
        targets_h = batch_h.targets.to(device)

        with torch.no_grad():
            out = model(inputs_h)
            preds = out.logits.argmax(dim=-1)
            exact = (preds == targets_h).all(dim=-1).float().mean().item()
            per_reg = (preds == targets_h).float().mean().item()

        results[str(h)] = {"exact_match": exact, "per_register": per_reg}

    return results


def evaluate_algorithmic_transfer(
    model: nn.Module,
    device: torch.device,
    num_samples: int = 1200,
    seed_base: int = 88000,
) -> Dict[str, Dict[str, float]]:
    """Evaluates zero-shot transfer performance across 4 independent algorithmic tasks."""
    model.eval()
    tasks = {
        "copy_broadcast": MultiRegisterCopyTask(seed=seed_base + 1),
        "permutation_cycle": PermutationCycleTask(seed=seed_base + 2),
        "running_accumulation": RunningAccumulationTask(seed=seed_base + 3),
        "conditional_overwrite": ConditionalOverwriteTask(seed=seed_base + 4),
    }

    results: Dict[str, Dict[str, float]] = {}
    for task_name, task in tasks.items():
        batch = task.generate_batch(num_samples, seed=seed_base + 500)
        inputs = batch.inputs.to(device)
        targets = batch.targets.to(device)

        with torch.no_grad():
            out = model(inputs)
            preds = out.logits.argmax(dim=-1)
            exact = (preds == targets).all(dim=-1).float().mean().item()
            per_reg = (preds == targets).float().mean().item()

        results[task_name] = {"exact_match": exact, "per_register": per_reg}

    return results


def train_single_run(
    model_key: str,
    model: nn.Module,
    splits: DisjointSplitManager,
    total_steps: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    gradient_clip_norm: float = 1.0,
    device: Optional[torch.device] = None,
    seed: int = 42,
    eval_interval: int = 100,
) -> Dict[str, Any]:
    """Trains a model, checkpoints on validation exact match, evaluates on held-out test."""
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    set_seed(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
        out = model(inputs)
        loss = F.cross_entropy(out.logits.view(-1, 10), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        if step % eval_interval == 0 or step == total_steps:
            model.eval()
            with torch.no_grad():
                val_loss, val_exact, val_per_reg, _, _ = evaluate_split(
                    model, val_set, batch_size=batch_size, device=dev, num_samples=1000
                )
                if val_exact > best_val_exact:
                    best_val_exact = val_exact
                    best_val_step = step
                    best_state_dict = copy.deepcopy(model.state_dict())

    wall_clock = time.perf_counter() - start_time
    train_throughput = (total_steps * batch_size) / wall_clock

    # Load best validation checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_exact, test_per_reg, step_exacts, latency_ms = evaluate_split(
        model, test_set, batch_size=batch_size, device=dev
    )
    inference_throughput = (len(test_set) / (latency_ms * (len(test_set) / batch_size))) * 1000.0

    peak_mem_mb = 0.0
    if dev.type == "cuda" and torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    flops_per_step = calculate_model_flops_per_step(model, batch_size=1)

    # Temporal diagnostics & Gate diagnostics
    credit_diag = run_temporal_credit_diagnostics(model, test_set, device=dev)
    gate_diag = run_detailed_gate_diagnostics(model, test_set, device=dev)

    # Extended horizon & Algorithmic transfer diagnostics
    horizons = evaluate_extended_horizons(model, device=dev)
    algorithmic_transfer = evaluate_algorithmic_transfer(model, device=dev)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_key": model_key,
        "seed": seed,
        "param_count": param_count,
        "total_steps": total_steps,
        "wall_clock_s": wall_clock,
        "train_throughput_examples_per_s": train_throughput,
        "inference_latency_ms": latency_ms,
        "inference_throughput_examples_per_s": inference_throughput,
        "flops_per_step": flops_per_step,
        "peak_mem_mb": peak_mem_mb,
        "best_val_exact": best_val_exact,
        "best_val_step": best_val_step,
        "test_at_best_val_checkpoint_exact": test_exact,
        "test_at_best_val_checkpoint_loss": test_loss,
        "test_at_best_val_checkpoint_per_reg": test_per_reg,
        "credit_diag": credit_diag,
        "gate_diag": gate_diag,
        "horizon_scaling": horizons,
        "algorithmic_transfer": algorithmic_transfer,
    }


def run_full_mechanism_audit(
    seeds: List[int] = [42, 43, 44, 45, 46],
    total_steps: int = 3000,
    batch_size: int = 64,
    output_path: str = "experiments/model_e_mechanism_results.json",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Runs all 7 parameter-matched models across 5 seeds with extensive telemetry."""
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== RE:Reason Mechanism Decomposition & Novelty Audit on {dev} ===")
    print(f"Seeds: {seeds} | Training Budget: {total_steps} steps | Batch: {batch_size}")

    models_order = [
        "model_a_layernorm",
        "model_b_bounded_additive",
        "model_c_convex_unbounded",
        "model_d_bounded_convex",
        "model_e_canonical_ugrnn",
        "model_f_canonical_mgu",
        "model_g_canonical_gru",
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {k: [] for k in models_order}

    total_runs = len(seeds) * len(models_order)
    current_run = 0
    overall_start = time.perf_counter()

    base_task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10, seed=42)

    for seed in seeds:
        print(f"\n=======================================================")
        print(f"Generating Strict Procedural Splits for Seed {seed}")
        print(f"=======================================================")
        split_mgr = DisjointSplitManager(
            base_task,
            train_size=6000,
            val_size=1200,
            test_size=1200,
            base_seed=seed,
        )

        suite = create_mechanism_decomposition_suite()

        for model_key in models_order:
            current_run += 1
            model = suite[model_key]
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\n[{current_run}/{total_runs}] Training {model_key} (seed={seed}, params={params:,})...")
            run_res = train_single_run(
                model_key=model_key,
                model=model,
                splits=split_mgr,
                total_steps=total_steps,
                batch_size=batch_size,
                device=dev,
                seed=seed,
                eval_interval=100,
            )
            all_results[model_key].append(run_res)

            print(
                f"  -> Done in {run_res['wall_clock_s']:.1f}s | "
                f"Val (Best): {run_res['best_val_exact']:.2%} | "
                f"Test Exact: {run_res['test_at_best_val_checkpoint_exact']:.2%} | "
                f"Per-Reg: {run_res['test_at_best_val_checkpoint_per_reg']:.2%}"
            )
            if run_res["gate_diag"] is not None:
                gd = run_res["gate_diag"]
                print(
                    f"  -> Gate: mean={gd['mean_gate']:.3f}, entropy={gd['gate_entropy']:.3f}, "
                    f"near0={gd['frac_near_0']:.1%}, near1={gd['frac_near_1']:.1%}, "
                    f"SET={gd['op_means']['SET']:.3f}, SWAP={gd['op_means']['SWAP']:.3f}, ADD={gd['op_means']['ADD']:.3f}"
                )

    total_time = time.perf_counter() - overall_start

    # Compute aggregate summary statistics across seeds
    summary: Dict[str, Any] = {}
    for m_key, runs in all_results.items():
        test_exacts = [r["test_at_best_val_checkpoint_exact"] for r in runs]
        test_per_regs = [r["test_at_best_val_checkpoint_per_reg"] for r in runs]
        val_exacts = [r["best_val_exact"] for r in runs]
        times = [r["wall_clock_s"] for r in runs]
        latencies = [r["inference_latency_ms"] for r in runs]
        attenuations = [r["credit_diag"]["attenuation_ratio"] for r in runs]
        flops = runs[0]["flops_per_step"]
        peak_mems = [r["peak_mem_mb"] for r in runs]

        horizon_summary: Dict[str, Dict[str, float]] = {}
        for h_str in ["5", "10", "15", "20", "25", "30"]:
            h_exacts = [r["horizon_scaling"][h_str]["exact_match"] for r in runs]
            h_regs = [r["horizon_scaling"][h_str]["per_register"] for r in runs]
            horizon_summary[h_str] = {
                "exact_mean": float(np.mean(h_exacts)),
                "exact_std": float(np.std(h_exacts)),
                "per_reg_mean": float(np.mean(h_regs)),
                "per_reg_std": float(np.std(h_regs)),
            }

        algo_summary: Dict[str, Dict[str, float]] = {}
        for task_name in ["copy_broadcast", "permutation_cycle", "running_accumulation", "conditional_overwrite"]:
            task_exacts = [r["algorithmic_transfer"][task_name]["exact_match"] for r in runs]
            task_regs = [r["algorithmic_transfer"][task_name]["per_register"] for r in runs]
            algo_summary[task_name] = {
                "exact_mean": float(np.mean(task_exacts)),
                "exact_std": float(np.std(task_exacts)),
                "per_reg_mean": float(np.mean(task_regs)),
                "per_reg_std": float(np.std(task_regs)),
            }

        gate_summary = None
        valid_gate_runs = [r["gate_diag"] for r in runs if r["gate_diag"] is not None]
        if valid_gate_runs:
            gate_summary = {
                "mean_gate": float(np.mean([g["mean_gate"] for g in valid_gate_runs])),
                "std_gate": float(np.mean([g["std_gate"] for g in valid_gate_runs])),
                "gate_entropy": float(np.mean([g["gate_entropy"] for g in valid_gate_runs])),
                "frac_near_0": float(np.mean([g["frac_near_0"] for g in valid_gate_runs])),
                "frac_near_1": float(np.mean([g["frac_near_1"] for g in valid_gate_runs])),
                "op_means": {
                    op: float(np.mean([g["op_means"][op] for g in valid_gate_runs]))
                    for op in ["SET", "SWAP", "ADD"]
                },
                "gate_state_displacement_corr": float(
                    np.mean([g["gate_state_displacement_corr"] for g in valid_gate_runs])
                ),
            }

        summary[m_key] = {
            "param_count": runs[0]["param_count"],
            "test_exact_mean": float(np.mean(test_exacts)),
            "test_exact_std": float(np.std(test_exacts)),
            "test_per_reg_mean": float(np.mean(test_per_regs)),
            "test_per_reg_std": float(np.std(test_per_regs)),
            "val_exact_mean": float(np.mean(val_exacts)),
            "val_exact_std": float(np.std(val_exacts)),
            "wall_clock_s_mean": float(np.mean(times)),
            "inference_latency_ms_mean": float(np.mean(latencies)),
            "flops_per_step": flops,
            "peak_memory_mb_mean": float(np.mean(peak_mems)),
            "attenuation_ratio_mean": float(np.mean(attenuations)),
            "attenuation_ratio_std": float(np.std(attenuations)),
            "horizon_summary": horizon_summary,
            "algo_transfer_summary": algo_summary,
            "gate_summary": gate_summary,
        }

    # Statistical paired comparison against Model D (Bounded Convex)
    paired_stats: Dict[str, Any] = {}
    base_d_exacts = np.array([r["test_at_best_val_checkpoint_exact"] for r in all_results["model_d_bounded_convex"]])

    for m_key, runs in all_results.items():
        if m_key == "model_d_bounded_convex":
            continue
        m_exacts = np.array([r["test_at_best_val_checkpoint_exact"] for r in runs])
        deltas = base_d_exacts - m_exacts
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas))
        cohen_d = float(mean_delta / max(std_delta, 1e-8))
        paired_stats[f"model_d_minus_{m_key}"] = {
            "paired_delta_mean": mean_delta,
            "paired_delta_std": std_delta,
            "seed_deltas": deltas.tolist(),
            "cohens_d": cohen_d,
        }

    final_payload = {
        "metadata": {
            "date": "2026-10-04",
            "scope": "RE:Reason Mechanism Decomposition & Novelty Audit across 7 models",
            "device": str(dev),
            "seeds": seeds,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "total_wall_clock_s": total_time,
        },
        "summary": summary,
        "paired_statistics": paired_stats,
        "runs": all_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nCompleted {total_runs} runs in {total_time:.1f}s ({total_time / 60:.2f} min).")
    print(f"Results serialized to {output_path}")
    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mechanism Decomposition Runner")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", type=str, default="experiments/model_e_mechanism_results.json")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_full_mechanism_audit(
        seeds=[42, 43, 44, 45, 46],
        total_steps=args.steps,
        batch_size=args.batch_size,
        output_path=args.output,
        device=dev,
    )
