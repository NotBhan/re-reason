"""Phase 1.5 Causal Architectural Ablation Experiment Runner.

Compares four parameter-matched models:
A. Current RE:Reason:       S_t = LayerNorm(S_(t-1) + f_theta(S_(t-1), e_t))
B. Residual No LayerNorm:    S_t = S_(t-1) + f_theta(S_(t-1), e_t)
C. Gated Residual:          S_t = S_(t-1) + g_theta(S_(t-1), e_t) ⊙ f_theta(S_(t-1), e_t)
D. Existing GRU Baseline:    h_t = GRUCell(e_t, h_(t-1))

Evaluates:
- Training, validation, and held-out test performance (test_at_best_val_checkpoint)
- Temporal credit-assignment diagnostics:
  * Gradient norms ||dL/dS_t|| across recurrent timesteps
  * Gradient attenuation ratios ||dL/dS_1|| / ||dL/dS_T||
  * State norm statistics ||S_t||, mean, std across timesteps
  * Step-to-step Jacobian transition ratios
- Sequence horizon generalization (T = 5, 10, 15, 20)
- Multi-seed statistics across 5 seeds (42, 43, 44, 45, 46)
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
    create_ablation_model_suite,
    forward_with_states_generic,
)
from re_reason.training.diagnostic_runner import evaluate_split
from re_reason.utils.seeding import set_seed


class AblationRunResult(NamedTuple):
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


def evaluate_horizon_generalization(
    model: nn.Module,
    device: torch.device,
    horizons: List[int] = [5, 10, 15, 20],
    num_samples: int = 1200,
    seed_base: int = 80000,
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


def train_and_evaluate_ablation_model(
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
    eval_interval: int = 200,
) -> AblationRunResult:
    """Executes full diagnostic training, model selection on Val, and evaluation on Test."""
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

    # Run horizon generalization diagnostics
    horizon_results = evaluate_horizon_generalization(model, device=dev)

    peak_mem_mb = 0.0
    if dev.type == "cuda" and torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    final_pt = trajectory[-1] if trajectory else {}

    return AblationRunResult(
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
        grad_norms_by_step=credit_diag["grad_norms"],
        attenuation_ratio=credit_diag["attenuation_ratio"],
        step_jacobian_ratios=credit_diag["step_jacobian_ratios"],
        horizon_results=horizon_results,
        trajectory=trajectory,
    )


def run_phase1_5_ablation(
    device_str: str = "auto",
    seeds: List[int] = [42, 43, 44, 45, 46],
    total_steps: int = 3000,
    output_json: str = "experiments/ablation_1_5_results.json",
) -> None:
    dev = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")
    print("=" * 80)
    print("RE:REASON PHASE 1.5 CAUSAL ARCHITECTURAL ABLATION")
    print(f"Device: {dev} | Seeds: {seeds} | Training Budget: {total_steps} steps")
    print("=" * 80 + "\n")

    base_task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10, seed=42)

    raw_results: Dict[str, List[Dict[str, Any]]] = {
        "model_a_layernorm": [],
        "model_b_no_ln": [],
        "model_c_gated": [],
        "model_d_gru": [],
    }

    for s_idx, seed in enumerate(seeds):
        print(f"\n>>> Running Evaluation on Seed {seed} ({s_idx + 1}/{len(seeds)})...")
        # Strict independent cryptographic splits per seed
        splits = DisjointSplitManager(base_task, train_size=6000, val_size=1200, test_size=1200, base_seed=seed)
        suite = create_ablation_model_suite()

        for model_key in ["model_a_layernorm", "model_b_no_ln", "model_c_gated", "model_d_gru"]:
            model = suite[model_key]
            p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            res = train_and_evaluate_ablation_model(
                model_key=model_key,
                model=model,
                splits=splits,
                total_steps=total_steps,
                device=dev,
                seed=seed,
            )

            res_dict = res._asdict()
            raw_results[model_key].append(res_dict)

            print(
                f"    [{model_key:18s}] P: {p_count:,} | "
                f"Val (Best): {res.best_val_exact*100:5.2f}% (step {res.best_val_step}) | "
                f"Test@BestVal: {res.test_at_best_val_checkpoint_exact*100:5.2f}% (Reg {res.test_at_best_val_checkpoint_per_reg*100:5.2f}%) | "
                f"Grad Atten (S1/ST): {res.attenuation_ratio:6.3f} | "
                f"Time: {res.wall_clock_s:4.1f}s"
            )

    # Compute aggregate summary statistics across seeds
    summary_stats: Dict[str, Any] = {}
    print("\n" + "=" * 80)
    print("PHASE 1.5 ABLATION AGGREGATE SUMMARY ACROSS 5 SEEDS")
    print("=" * 80)

    for m_key in ["model_a_layernorm", "model_b_no_ln", "model_c_gated", "model_d_gru"]:
        runs = raw_results[m_key]
        n_seeds = len(runs)

        test_exacts = [r["test_at_best_val_checkpoint_exact"] for r in runs]
        test_per_regs = [r["test_at_best_val_checkpoint_per_reg"] for r in runs]
        val_exacts = [r["best_val_exact"] for r in runs]
        train_exacts = [r["final_train_exact"] for r in runs]
        times = [r["wall_clock_s"] for r in runs]
        atten_ratios = [r["attenuation_ratio"] for r in runs]

        mean_grad_norms = np.mean([r["grad_norms_by_step"] for r in runs], axis=0).tolist()
        mean_state_norms = np.mean([r["state_norms_by_step"] for r in runs], axis=0).tolist()

        h_exacts: Dict[str, List[float]] = {"5": [], "10": [], "15": [], "20": []}
        h_per_regs: Dict[str, List[float]] = {"5": [], "10": [], "15": [], "20": []}
        for r in runs:
            for h_str in ["5", "10", "15", "20"]:
                h_exacts[h_str].append(r["horizon_results"][h_str]["exact_match"])
                h_per_regs[h_str].append(r["horizon_results"][h_str]["per_register"])

        summary_stats[m_key] = {
            "param_count": runs[0]["param_count"],
            "test_at_best_val_exact_mean": float(np.mean(test_exacts)),
            "test_at_best_val_exact_std": float(np.std(test_exacts)),
            "test_at_best_val_per_reg_mean": float(np.mean(test_per_regs)),
            "test_at_best_val_per_reg_std": float(np.std(test_per_regs)),
            "val_exact_mean": float(np.mean(val_exacts)),
            "val_exact_std": float(np.std(val_exacts)),
            "train_exact_mean": float(np.mean(train_exacts)),
            "train_exact_std": float(np.std(train_exacts)),
            "time_s_mean": float(np.mean(times)),
            "attenuation_ratio_mean": float(np.mean(atten_ratios)),
            "attenuation_ratio_std": float(np.std(atten_ratios)),
            "mean_grad_norms_by_step": mean_grad_norms,
            "mean_state_norms_by_step": mean_state_norms,
            "horizon_generalization": {
                h_str: {
                    "exact_mean": float(np.mean(h_exacts[h_str])),
                    "exact_std": float(np.std(h_exacts[h_str])),
                    "per_reg_mean": float(np.mean(h_per_regs[h_str])),
                    "per_reg_std": float(np.std(h_per_regs[h_str])),
                }
                for h_str in ["5", "10", "15", "20"]
            },
        }

        print(f"\nModel: {m_key.upper()} (Params: {runs[0]['param_count']:,})")
        print(f"  Test Exact (at best val): {np.mean(test_exacts)*100:5.2f}% ± {np.std(test_exacts)*100:4.2f}%")
        print(f"  Test Per-Reg:             {np.mean(test_per_regs)*100:5.2f}% ± {np.std(test_per_regs)*100:4.2f}%")
        print(f"  Best Val Exact:           {np.mean(val_exacts)*100:5.2f}% ± {np.std(val_exacts)*100:4.2f}%")
        print(f"  Final Train Exact:        {np.mean(train_exacts)*100:5.2f}% ± {np.std(train_exacts)*100:4.2f}%")
        h_summary = [f"{summary_stats[m_key]['horizon_generalization'][h]['exact_mean']*100:.1f}%" for h in ['5', '10', '15', '20']]
        print(f"  Horizon Exact [T=5,10,15,20]: {h_summary}")

    # Serialize complete payload to JSON
    output_payload = {
        "metadata": {
            "timestamp": time.time(),
            "device": str(dev),
            "seeds": seeds,
            "total_steps": total_steps,
        },
        "summary": summary_stats,
        "runs": raw_results,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nComplete machine-readable telemetry written to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1.5 Causal Ablation Suite")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", type=str, default="experiments/ablation_1_5_results.json")
    args = parser.parse_args()

    run_phase1_5_ablation(
        device_str=args.device,
        seeds=[42, 43, 44, 45, 46],
        total_steps=args.steps,
        output_json=args.output,
    )
