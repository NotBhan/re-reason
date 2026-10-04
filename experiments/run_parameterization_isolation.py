"""Parameterization-Isolation Experiment Runner.

Evaluates 7 matched architectural variants across 5 seeds:
1. Canonical UGRNN: affine candidate + affine gate
2. Independent MLPs: separate MLP_candidate([S,e]) + separate MLP_gate([S,e])
3. Shared Deep MLP Trunk: h = GELU(MLP([S,e])), candidate head + gate head
4. Shared Linear Trunk: Shared trunk without GELU (linear bottleneck)
5. Canonical MGU: Affine with state pre-gating (f_t ⊙ S_(t-1))
6. Nonlinear Candidate MGU: State pre-gating + nonlinear candidate MLP
7. Current Model D: Shared 1-layer trunk with GELU (ReReasonBoundedConvexModel)

Audits:
- Corrected FLOP calculation covering Linear, LayerNorm, and native GRUCell
- Seed-wise paired statistics across seeds 42, 43, 44, 45, 46
- Gate entropy, saturation, operation sensitivity, and gate-displacement correlation
- Wall-clock training time, inference latency, and peak memory
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
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from re_reason.benchmarks.dataset_split import DisjointSplitManager, SplitDataset
from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.core.ablation_models import (
    IntermediateDiagnosticsOutput,
    create_parameterization_isolation_suite,
    forward_with_states_generic,
)
from re_reason.training.diagnostic_runner import evaluate_split
from re_reason.utils.seeding import set_seed


def calculate_model_flops_per_step(model: nn.Module, batch_size: int = 1) -> int:
    """Audited calculation of theoretical FLOPs per recurrent step.

    FLOP accounting:
    - Linear Layer: 2 * in_features * out_features (multiply-accumulate) + (out_features if bias else 0)
    - LayerNorm: 8 * normalized_shape (mean, variance, normalize, affine scale/shift)
    - GRUCell: 6 * H * I (input maps) + 6 * H^2 (hidden maps) + 6 * H (biases) + 17 * H (elementwise gates & convex update)
    """
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features + (
                m.out_features if m.bias is not None else 0
            )
        elif isinstance(m, nn.LayerNorm):
            flops += 8 * m.normalized_shape[0]
        elif isinstance(m, nn.GRUCell):
            I, H = m.input_size, m.hidden_size
            flops += 6 * H * I + 6 * H * H + 23 * H
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

    state_norms = [s.norm(dim=-1).mean().item() for s in states]
    state_stds = [s.std(dim=-1).mean().item() for s in states]

    for s in states:
        s.retain_grad()

    loss = F.cross_entropy(diag_out.terminal_logits.view(-1, 10), targets.view(-1))
    loss.backward()

    grad_norms = []
    for s in states:
        if s.grad is not None:
            grad_norms.append(s.grad.norm(dim=-1).mean().item())
        else:
            grad_norms.append(0.0)

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
    states = diag_out.states

    if gates is None or len(gates) == 0:
        return None

    all_gates = torch.cat(gates, dim=0)  # (B * T, D)
    eps = 1e-7
    clamped_gates = torch.clamp(all_gates, eps, 1.0 - eps)

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
    for t in range(len(gates)):
        prev_s = states[t - 1] if t > 0 else torch.zeros_like(states[t])
        state_diff = (states[t] - prev_s).abs()
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

    credit_diag = run_temporal_credit_diagnostics(model, test_set, device=dev)
    gate_diag = run_detailed_gate_diagnostics(model, test_set, device=dev)

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
    }


def run_parameterization_isolation_audit(
    seeds: List[int] = [42, 43, 44, 45, 46],
    total_steps: int = 3000,
    batch_size: int = 64,
    output_path: str = "experiments/parameterization_isolation_results.json",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Orchestrates 7 models x 5 seeds = 35 full runs with exact telemetry."""
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"RE:REASON PARAMETERIZATION-ISOLATION EXPERIMENT ON {dev}")
    print(f"Seeds: {seeds} | Steps: {total_steps} | Batch Size: {batch_size}")
    print("=" * 80)

    models_order = [
        "1_canonical_ugrnn",
        "2_independent_mlp_ugrnn",
        "3_shared_deep_mlp_trunk",
        "4_shared_linear_trunk",
        "5_canonical_mgu",
        "6_nonlinear_candidate_mgu",
        "7_current_model_d",
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

        suite = create_parameterization_isolation_suite()

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
                    f"disp_corr={gd['gate_state_displacement_corr']:.3f}"
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
        flops = runs[0]["flops_per_step"]
        attenuations = [r["credit_diag"]["attenuation_ratio"] for r in runs]
        peaks = [r["peak_mem_mb"] for r in runs]

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
            "peak_memory_mb_mean": float(np.mean(peaks)),
            "attenuation_ratio_mean": float(np.mean(attenuations)),
            "attenuation_ratio_std": float(np.std(attenuations)),
            "gate_summary": gate_summary,
        }

    # Seed-wise paired statistics: Compare Model D (7) against all others
    paired_stats: Dict[str, Any] = {}
    model_d_exacts = [r["test_at_best_val_checkpoint_exact"] for r in all_results["7_current_model_d"]]

    for other_key in all_results:
        if other_key == "7_current_model_d":
            continue
        other_exacts = [r["test_at_best_val_checkpoint_exact"] for r in all_results[other_key]]
        deltas = [d - o for d, o in zip(model_d_exacts, other_exacts)]
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        n = len(deltas)
        se = std_delta / math.sqrt(n) if n > 0 and std_delta > 0 else 1e-8
        t_stat = mean_delta / se
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1))) if std_delta > 0 else 0.0
        cohens_d = mean_delta / std_delta if std_delta > 0 else 0.0

        paired_stats[f"model_d_minus_{other_key}"] = {
            "paired_delta_mean": mean_delta,
            "paired_delta_std": std_delta,
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": cohens_d,
            "seed_deltas": deltas,
        }

    output_payload = {
        "metadata": {
            "date": "2026-10-04",
            "experiment": "parameterization_isolation",
            "seeds": seeds,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "total_wall_clock_s": total_time,
            "flop_definition": "Matrix MACs * 2 + bias + LayerNorm(8*D) + GRUCell(6HI + 6H^2 + 23H)",
        },
        "summary": summary,
        "paired_statistics": paired_stats,
        "runs": all_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("PARAMETERIZATION ISOLATION SUMMARY")
    print("=" * 80)
    for m_key, s in summary.items():
        print(
            f"{m_key:28s} | "
            f"Params: {s['param_count']:,} | "
            f"Test EM: {s['test_exact_mean']:.2%} ± {s['test_exact_std']:.2%} | "
            f"Per-Reg: {s['test_per_reg_mean']:.2%} | "
            f"FLOPs: {s['flops_per_step']:,} | "
            f"Wall-clock: {s['wall_clock_s_mean']:.1f}s"
        )

    print("\n--- Paired Deltas (Model D vs. Others) ---")
    for comp, p in paired_stats.items():
        print(
            f"{comp:35s} | "
            f"Mean Delta: {p['paired_delta_mean']:+.2%} ± {p['paired_delta_std']:.2%} | "
            f"t: {p['t_statistic']:.2f} | "
            f"p: {p['p_value']:.4e} | "
            f"d: {p['cohens_d']:.2f}"
        )

    print(f"\nResults successfully written to: {output_path}")
    return output_payload


if __name__ == "__main__":
    run_parameterization_isolation_audit()
