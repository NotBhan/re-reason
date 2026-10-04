"""Phase 1.5 Bounded Convex Residual Update Ablation Experiment Runner.

Evaluates Models A, E, and D across 5 seeds:
- Model A: Current RE:Reason (LayerNorm additive residual)
- Model E: Bounded Convex Residual Update (tanh candidate + coordinate gate)
- Model D: GRU baseline

Measures:
1. Exact and per-register train/val/test performance (test_at_best_val_checkpoint)
2. State norms ||S_t|| and candidate norms ||S_tilde_t|| across recurrent steps
3. Gate statistics across recurrent steps (mean, std, min, max)
4. Operation-specific gate distributions (SET, SWAP, ADD)
5. Gradient norms ||dL/dS_t|| and attenuation ratios
6. Step-to-step Jacobian transition ratios
7. Horizon generalization across T in {5, 10, 15, 20}
8. Multi-seed aggregates across seeds [42, 43, 44, 45, 46]
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from re_reason.benchmarks.dataset_split import DisjointSplitManager, SplitDataset
from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.core.ablation_models import (
    IntermediateDiagnosticsOutput,
    ReReasonBoundedConvexModel,
    create_models_a_e_d_suite,
    forward_with_states_generic,
)
from re_reason.training.diagnostic_runner import evaluate_split
from re_reason.utils.seeding import set_seed


class BoundedAblationRunResult(NamedTuple):
    model_key: str
    seed: int
    param_count: int
    total_steps: int
    wall_clock_s: float
    best_val_exact: float
    best_val_step: int
    final_train_exact: float
    final_train_loss: float
    final_train_per_reg: float
    final_val_exact: float
    final_val_loss: float
    final_val_per_reg: float
    test_at_best_val_checkpoint_exact: float
    test_at_best_val_checkpoint_loss: float
    test_at_best_val_checkpoint_per_reg: float
    step_wise_test_exact: List[float]
    inference_latency_ms: float
    peak_memory_mb: float
    state_norms_by_step: List[float]
    state_stds_by_step: List[float]
    candidate_norms_by_step: Optional[List[float]]
    gate_stats_by_step: Optional[List[Dict[str, float]]]
    op_gate_stats: Optional[Dict[str, Dict[str, float]]]
    register_chunk_correlation: Optional[Dict[str, float]]
    grad_norms_by_step: List[float]
    attenuation_ratio: float
    step_jacobian_ratios: List[float]
    horizon_results: Dict[str, Dict[str, float]]
    trajectory: List[Dict[str, Any]]


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

    step_jacobian_ratios = []
    for t in range(seq_len - 1, 0, -1):
        ratio = grad_norms[t - 1] / max(grad_norms[t], 1e-12)
        step_jacobian_ratios.append(ratio)
    step_jacobian_ratios.reverse()

    return {
        "state_norms": state_norms,
        "state_stds": state_stds,
        "grad_norms": grad_norms,
        "attenuation_ratio": attenuation_ratio,
        "step_jacobian_ratios": step_jacobian_ratios,
    }


def run_model_e_gate_diagnostics(
    model: nn.Module,
    test_set: SplitDataset,
    device: torch.device,
    num_samples: int = 1200,
) -> Dict[str, Any]:
    """Measures candidate norms, gate statistics, and op-specific gate distributions for Model E."""
    if not isinstance(model, ReReasonBoundedConvexModel):
        return {
            "candidate_norms": None,
            "gate_stats": None,
            "op_gate_stats": None,
            "register_chunk_correlation": None,
        }

    model.eval()
    batch = test_set.get_batch(num_samples, idx=0)
    inputs = batch.inputs.to(device)  # (B, 10, 21)

    diag_out = model.forward_with_states(inputs, return_components=True)
    candidates = diag_out.candidates  # 10 tensors of (B, 256)
    gates = diag_out.gates            # 10 tensors of (B, 256)

    cand_norms_by_step = [c.norm(dim=-1).mean().item() for c in candidates]

    gate_stats_by_step = []
    for t, g in enumerate(gates):
        gate_stats_by_step.append({
            "step": t + 1,
            "mean": g.mean().item(),
            "std": g.std().item(),
            "min": g.min().item(),
            "max": g.max().item(),
        })

    # Operation-specific gate statistics
    set_gates = []
    swap_gates = []
    add_gates = []

    for t in range(len(gates)):
        x_t = inputs[:, t, :]
        op_types = x_t[:, :3].argmax(dim=-1)
        g_t = gates[t]

        set_mask = (op_types == 0)
        swap_mask = (op_types == 1)
        add_mask = (op_types == 2)

        if set_mask.any():
            set_gates.append(g_t[set_mask])
        if swap_mask.any():
            swap_gates.append(g_t[swap_mask])
        if add_mask.any():
            add_gates.append(g_t[add_mask])

    cat_set = torch.cat(set_gates, dim=0) if set_gates else torch.empty(0)
    cat_swap = torch.cat(swap_gates, dim=0) if swap_gates else torch.empty(0)
    cat_add = torch.cat(add_gates, dim=0) if add_gates else torch.empty(0)

    op_gate_stats = {
        "SET": {
            "mean": cat_set.mean().item() if cat_set.numel() > 0 else 0.0,
            "std": cat_set.std().item() if cat_set.numel() > 0 else 0.0,
            "min": cat_set.min().item() if cat_set.numel() > 0 else 0.0,
            "max": cat_set.max().item() if cat_set.numel() > 0 else 0.0,
            "count": int(cat_set.shape[0]) if cat_set.numel() > 0 else 0,
        },
        "SWAP": {
            "mean": cat_swap.mean().item() if cat_swap.numel() > 0 else 0.0,
            "std": cat_swap.std().item() if cat_swap.numel() > 0 else 0.0,
            "min": cat_swap.min().item() if cat_swap.numel() > 0 else 0.0,
            "max": cat_swap.max().item() if cat_swap.numel() > 0 else 0.0,
            "count": int(cat_swap.shape[0]) if cat_swap.numel() > 0 else 0,
        },
        "ADD": {
            "mean": cat_add.mean().item() if cat_add.numel() > 0 else 0.0,
            "std": cat_add.std().item() if cat_add.numel() > 0 else 0.0,
            "min": cat_add.min().item() if cat_add.numel() > 0 else 0.0,
            "max": cat_add.max().item() if cat_add.numel() > 0 else 0.0,
            "count": int(cat_add.shape[0]) if cat_add.numel() > 0 else 0,
        },
    }

    # Test coordinate-to-symbolic-register chunk alignment hypothesis (4 x 64 dims)
    chunk_diffs = []
    for t in range(len(gates)):
        x_t = inputs[:, t, :]
        reg_a = x_t[:, 3:7].argmax(dim=-1)
        g_t = gates[t]

        for b in range(inputs.shape[0]):
            ra = reg_a[b].item()
            target_slice = g_t[b, ra * 64 : (ra + 1) * 64]
            other_indices = [i for i in range(256) if not (ra * 64 <= i < (ra + 1) * 64)]
            other_slice = g_t[b, other_indices]
            chunk_diffs.append((target_slice.mean() - other_slice.mean()).item())

    register_chunk_correlation = {
        "mean_gate_diff_target_vs_other_chunks": float(np.mean(chunk_diffs)),
        "std_gate_diff_target_vs_other_chunks": float(np.std(chunk_diffs)),
    }

    return {
        "candidate_norms": cand_norms_by_step,
        "gate_stats": gate_stats_by_step,
        "op_gate_stats": op_gate_stats,
        "register_chunk_correlation": register_chunk_correlation,
    }


def evaluate_horizon_generalization(
    model: nn.Module,
    device: torch.device,
    horizons: List[int] = [5, 10, 15, 20],
    num_samples: int = 1200,
    seed_base: int = 90000,
) -> Dict[str, Dict[str, float]]:
    """Evaluates trained model on sequence lengths T without retraining."""
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


def train_and_evaluate_bounded_model(
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
) -> BoundedAblationRunResult:
    """Executes full diagnostic training, checkpointing on Val, and evaluation on Test."""
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    set_seed(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trajectory: List[Dict[str, Any]] = []
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
                train_preds = out.logits.argmax(dim=-1)
                train_exact = (train_preds == targets).all(dim=-1).float().mean().item()
                train_per_reg = (train_preds == targets).float().mean().item()

                val_loss, val_exact, val_per_reg, _, _ = evaluate_split(
                    model, val_set, batch_size=batch_size, device=dev, num_samples=1000
                )

                trajectory.append({
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "train_exact": train_exact,
                    "val_exact": val_exact,
                    "train_per_reg": train_per_reg,
                    "val_per_reg": val_per_reg,
                })

                if val_exact > best_val_exact:
                    best_val_exact = val_exact
                    best_val_step = step
                    best_state_dict = copy.deepcopy(model.state_dict())

    wall_clock = time.perf_counter() - start_time

    # Load BEST validation checkpoint for held-out test evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_exact, test_per_reg, step_exacts, latency_ms = evaluate_split(
        model, test_set, batch_size=batch_size, device=dev
    )

    # Run temporal credit diagnostics on held-out test data
    credit_diag = run_temporal_credit_diagnostics(model, test_set, device=dev)

    # Run Model E specific gate diagnostics
    gate_diag = run_model_e_gate_diagnostics(model, test_set, device=dev)

    # Run horizon generalization diagnostics
    horizon_results = evaluate_horizon_generalization(model, device=dev)

    peak_mem_mb = 0.0
    if dev.type == "cuda" and torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    final_pt = trajectory[-1] if trajectory else {}

    return BoundedAblationRunResult(
        model_key=model_key,
        seed=seed,
        param_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_steps=total_steps,
        wall_clock_s=wall_clock,
        best_val_exact=best_val_exact,
        best_val_step=best_val_step,
        final_train_exact=final_pt.get("train_exact", 0.0),
        final_train_loss=final_pt.get("train_loss", 0.0),
        final_train_per_reg=final_pt.get("train_per_reg", 0.0),
        final_val_exact=final_pt.get("val_exact", 0.0),
        final_val_loss=final_pt.get("val_loss", 0.0),
        final_val_per_reg=final_pt.get("val_per_reg", 0.0),
        test_at_best_val_checkpoint_exact=test_exact,
        test_at_best_val_checkpoint_loss=test_loss,
        test_at_best_val_checkpoint_per_reg=test_per_reg,
        step_wise_test_exact=step_exacts,
        inference_latency_ms=latency_ms,
        peak_memory_mb=peak_mem_mb,
        state_norms_by_step=credit_diag["state_norms"],
        state_stds_by_step=credit_diag["state_stds"],
        candidate_norms_by_step=gate_diag["candidate_norms"],
        gate_stats_by_step=gate_diag["gate_stats"],
        op_gate_stats=gate_diag["op_gate_stats"],
        register_chunk_correlation=gate_diag["register_chunk_correlation"],
        grad_norms_by_step=credit_diag["grad_norms"],
        attenuation_ratio=credit_diag["attenuation_ratio"],
        step_jacobian_ratios=credit_diag["step_jacobian_ratios"],
        horizon_results=horizon_results,
        trajectory=trajectory,
    )


def run_full_bounded_ablation(
    seeds: List[int] = [42, 43, 44, 45, 46],
    total_steps: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    output_path: str = "experiments/ablation_1_5_bounded_results.json",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Runs Models A, E, and D across all seeds with strict controls."""
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== RE:Reason Phase 1.5 Bounded Convex Ablation on {dev} ===")
    print(f"Seeds: {seeds}, Total steps per model: {total_steps}, Batch size: {batch_size}")

    all_results: Dict[str, List[Dict[str, Any]]] = {
        "model_a_layernorm": [],
        "model_e_bounded_convex": [],
        "model_d_gru": [],
    }

    models_order = ["model_a_layernorm", "model_e_bounded_convex", "model_d_gru"]

    total_runs = len(seeds) * len(models_order)
    current_run = 0

    overall_start = time.perf_counter()

    base_task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10, seed=42)

    for seed in seeds:
        print(f"\n=======================================================")
        print(f"Generating Disjoint Splits for Seed {seed}")
        print(f"=======================================================")
        split_mgr = DisjointSplitManager(
            base_task,
            train_size=6000,
            val_size=1200,
            test_size=1200,
            base_seed=seed,
        )

        # Instantiate fresh models for this seed
        suite = create_models_a_e_d_suite()

        for model_key in models_order:
            current_run += 1
            model = suite[model_key]
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\n[{current_run}/{total_runs}] Training {model_key} (seed={seed}, params={params:,})...")
            run_result = train_and_evaluate_bounded_model(
                model_key=model_key,
                model=model,
                splits=split_mgr,
                total_steps=total_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=dev,
                seed=seed,
                eval_interval=100,
            )

            res_dict = run_result._asdict()
            all_results[model_key].append(res_dict)

            print(
                f"  -> Done in {run_result.wall_clock_s:.1f}s | "
                f"Best Val: {run_result.best_val_exact:.2%} (step {run_result.best_val_step}) | "
                f"Test Exact: {run_result.test_at_best_val_checkpoint_exact:.2%} | "
                f"Test Per-Reg: {run_result.test_at_best_val_checkpoint_per_reg:.2%}"
            )
            if run_result.candidate_norms_by_step is not None:
                print(f"  -> Model E cand norms: {run_result.candidate_norms_by_step[:3]} ... {run_result.candidate_norms_by_step[-1]:.2f}")
                if run_result.op_gate_stats is not None:
                    print(
                        f"  -> Model E Op Gates: SET mean={run_result.op_gate_stats['SET']['mean']:.4f}, "
                        f"SWAP mean={run_result.op_gate_stats['SWAP']['mean']:.4f}, "
                        f"ADD mean={run_result.op_gate_stats['ADD']['mean']:.4f}"
                    )

    # Compute summary statistics across seeds
    summary: Dict[str, Any] = {}
    for model_key, run_list in all_results.items():
        test_exacts = [r["test_at_best_val_checkpoint_exact"] for r in run_list]
        test_per_regs = [r["test_at_best_val_checkpoint_per_reg"] for r in run_list]
        val_exacts = [r["best_val_exact"] for r in run_list]
        train_exacts = [r["final_train_exact"] for r in run_list]
        wall_times = [r["wall_clock_s"] for r in run_list]
        latencies = [r["inference_latency_ms"] for r in run_list]
        attenuations = [r["attenuation_ratio"] for r in run_list]
        state_norms = np.array([r["state_norms_by_step"] for r in run_list]).mean(axis=0).tolist()
        grad_norms = np.array([r["grad_norms_by_step"] for r in run_list]).mean(axis=0).tolist()

        horizon_summary: Dict[str, Dict[str, float]] = {}
        for h_str in ["5", "10", "15", "20"]:
            h_exacts = [r["horizon_results"][h_str]["exact_match"] for r in run_list]
            h_regs = [r["horizon_results"][h_str]["per_register"] for r in run_list]
            horizon_summary[h_str] = {
                "exact_mean": float(np.mean(h_exacts)),
                "exact_std": float(np.std(h_exacts)),
                "per_reg_mean": float(np.mean(h_regs)),
                "per_reg_std": float(np.std(h_regs)),
            }

        cand_norms = None
        op_gate_summary = None
        gate_stats_summary = None
        reg_chunk_summary = None

        if model_key == "model_e_bounded_convex":
            cand_norms = np.array([r["candidate_norms_by_step"] for r in run_list]).mean(axis=0).tolist()
            op_gate_summary = {
                op: {
                    "mean": float(np.mean([r["op_gate_stats"][op]["mean"] for r in run_list])),
                    "std": float(np.mean([r["op_gate_stats"][op]["std"] for r in run_list])),
                    "min": float(np.min([r["op_gate_stats"][op]["min"] for r in run_list])),
                    "max": float(np.max([r["op_gate_stats"][op]["max"] for r in run_list])),
                }
                for op in ["SET", "SWAP", "ADD"]
            }
            gate_stats_summary = []
            for t in range(10):
                gate_stats_summary.append({
                    "step": t + 1,
                    "mean": float(np.mean([r["gate_stats_by_step"][t]["mean"] for r in run_list])),
                    "std": float(np.mean([r["gate_stats_by_step"][t]["std"] for r in run_list])),
                    "min": float(np.min([r["gate_stats_by_step"][t]["min"] for r in run_list])),
                    "max": float(np.max([r["gate_stats_by_step"][t]["max"] for r in run_list])),
                })
            reg_chunk_summary = {
                "mean_gate_diff_target_vs_other_chunks": float(
                    np.mean([r["register_chunk_correlation"]["mean_gate_diff_target_vs_other_chunks"] for r in run_list])
                ),
                "std_gate_diff_target_vs_other_chunks": float(
                    np.mean([r["register_chunk_correlation"]["std_gate_diff_target_vs_other_chunks"] for r in run_list])
                ),
            }

        summary[model_key] = {
            "param_count": run_list[0]["param_count"],
            "test_exact_mean": float(np.mean(test_exacts)),
            "test_exact_std": float(np.std(test_exacts)),
            "test_per_reg_mean": float(np.mean(test_per_regs)),
            "test_per_reg_std": float(np.std(test_per_regs)),
            "val_exact_mean": float(np.mean(val_exacts)),
            "val_exact_std": float(np.std(val_exacts)),
            "train_exact_mean": float(np.mean(train_exacts)),
            "train_exact_std": float(np.std(train_exacts)),
            "wall_clock_s_mean": float(np.mean(wall_times)),
            "inference_latency_ms_mean": float(np.mean(latencies)),
            "attenuation_ratio_mean": float(np.mean(attenuations)),
            "attenuation_ratio_std": float(np.std(attenuations)),
            "state_norms_by_step_mean": state_norms,
            "candidate_norms_by_step_mean": cand_norms,
            "gate_stats_by_step_mean": gate_stats_summary,
            "op_gate_summary": op_gate_summary,
            "register_chunk_correlation": reg_chunk_summary,
            "grad_norms_by_step_mean": grad_norms,
            "horizon_summary": horizon_summary,
        }

    total_time = time.perf_counter() - overall_start
    print(f"\nAll {total_runs} runs completed in {total_time:.1f}s ({total_time / 60:.2f} min).")

    final_payload = {
        "metadata": {
            "date": "2026-10-04",
            "scope": "Phase 1.5 Bounded Convex Residual Update Ablation",
            "device": str(dev),
            "seeds": seeds,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "total_wall_clock_s": total_time,
        },
        "summary": summary,
        "runs": all_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nResults serialized to {output_path}")
    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1.5 Bounded Convex Ablation Runner")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps per run")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", type=str, default="experiments/ablation_1_5_bounded_results.json")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_full_bounded_ablation(
        seeds=[42, 43, 44, 45, 46],
        total_steps=args.steps,
        batch_size=args.batch_size,
        output_path=args.output,
        device=dev,
    )
