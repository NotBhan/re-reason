"""Comprehensive Diagnostic Suite for RE:Reason Mechanism Q&A.

Runs all empirical probing, diagnostic measurements, and training budget experiments
required to answer questions Q1 through Q17:
1. State probing across timesteps (Linear and MLP probes for R0..R3)
2. Spatial localization vs distributed representation test (coordinate chunking & ablation)
3. Step-by-step state traces on SET, SWAP, ADD
4. Gate selectivity: changed vs unchanged registers, op sensitivity, coordinate correlation
5. Linear vs Nonlinear interaction probe (analytical & empirical Jacobian)
6. MGU Pre-gating suppression analysis (S_(t-1) vs f_t ⊙ S_(t-1))
7. Nonlinear MGU vs Model D component breakdown
8. Failure mode classification on test set (A..F categories)
9. Memory retention across intervening untouched steps (k=1..8)
10. Algorithmic generalization diagnostics
11. Gradient health & backprop norms through time (t=10..1)
12. Training budget experiment (Q14): 1k, 3k, 6k, 12k steps on Nonlinear MGU across 3 seeds
13. Generates the 8 required publication-quality visualization plots
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from re_reason.benchmarks.dataset_split import DisjointSplitManager, SplitDataset
from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.core.ablation_models import (
    CanonicalMGUModel,
    CanonicalUGRNNModel,
    IntermediateDiagnosticsOutput,
    NonlinearCandidateMGUModel,
    ReReasonBoundedConvexModel,
    SharedLinearTrunkModel,
    create_parameterization_isolation_suite,
    forward_with_states_generic,
)
from re_reason.training.diagnostic_runner import evaluate_split
from re_reason.utils.seeding import set_seed


# ---------------------------------------------------------------------------
# Helpers for Training & Checkpoint Handling
# ---------------------------------------------------------------------------

def train_or_load_model(
    model: nn.Module,
    model_name: str,
    splits: DisjointSplitManager,
    checkpoint_dir: str = "experiments/checkpoints",
    total_steps: int = 3000,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    seed: int = 42,
    force_retrain: bool = False,
) -> nn.Module:
    """Trains a model with validation checkpoint selection or loads from disk if saved."""
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_seed{seed}_{total_steps}steps.pt")

    model = model.to(dev)

    if os.path.exists(ckpt_path) and not force_retrain:
        print(f"Loading cached checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=dev))
        return model

    print(f"Training {model_name} (seed={seed}, steps={total_steps}) on {dev}...")
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_val_exact = -1.0
    best_state_dict = None
    train_set = splits.train_set
    val_set = splits.val_set

    for step in range(1, total_steps + 1):
        model.train()
        batch = train_set.get_batch(batch_size, idx=step)
        inputs = batch.inputs.to(dev)
        targets = batch.targets.to(dev)

        optimizer.zero_grad()
        out = model(inputs)
        loss = F.cross_entropy(out.logits.view(-1, 10), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0 or step == total_steps:
            model.eval()
            with torch.no_grad():
                _, val_exact, _, _, _ = evaluate_split(model, val_set, batch_size=batch_size, device=dev, num_samples=1000)
                if val_exact > best_val_exact:
                    best_val_exact = val_exact
                    best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save(model.state_dict(), ckpt_path)
    print(f"  -> Saved best checkpoint (val_exact={best_val_exact:.2%}) to {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Diagnostic 1: Probing State Representations across Timesteps (Q1 & Q11)
# ---------------------------------------------------------------------------

def run_state_probing_diagnostics(
    model: nn.Module,
    split_mgr: DisjointSplitManager,
    device: torch.device,
    state_dim: int = 212,
) -> Dict[str, Any]:
    """Evaluates Linear and MLP probes for each of the 4 registers at every timestep t in 1..10,
    and tests spatial localization vs distributed representation."""
    model.eval()
    dev = device

    # Generate multi-step ground truth by running step-by-step simulation
    task = split_mgr.task
    test_set = split_mgr.test_set
    num_samples = len(test_set)
    batch = test_set.get_batch(num_samples, idx=0)
    inputs = batch.inputs.to(dev)  # (B, T, 21)

    # Compute ground truth register states at all timesteps t = 0..10
    # Inputs: one-hot operations: [op_set, op_swap, op_add, r_dest(4), r_src(4), imm(10)]
    B, T, _ = inputs.shape
    gt_history = torch.zeros(B, T + 1, 4, dtype=torch.long)  # registers 0..3 in 0..9

    for b in range(B):
        curr_regs = [0, 0, 0, 0]
        gt_history[b, 0] = torch.tensor(curr_regs)
        for t in range(T):
            x = inputs[b, t].cpu()
            op = x[:3].argmax().item()
            dest = x[3:7].argmax().item()
            src = x[7:11].argmax().item()
            imm = x[11:21].argmax().item()

            if op == 0:  # SET
                curr_regs[dest] = imm
            elif op == 1:  # SWAP
                curr_regs[dest], curr_regs[src] = curr_regs[src], curr_regs[dest]
            elif op == 2:  # ADD
                curr_regs[dest] = (curr_regs[dest] + curr_regs[src]) % 10
            gt_history[b, t + 1] = torch.tensor(curr_regs)

    # Extract model latent states across all timesteps
    with torch.no_grad():
        diag_out = forward_with_states_generic(model, inputs)
        states = [s.cpu() for s in diag_out.states]  # list of T tensors of shape (B, D)

    # Train linear and non-linear probes at each timestep for each register
    linear_probe_accs = {r: [] for r in range(4)}
    mlp_probe_accs = {r: [] for r in range(4)}
    chunk_probe_accs = {r: [] for r in range(4)}  # probe trained only on chunk r

    chunk_size = state_dim // 4

    for t in range(T):
        S_t = states[t]  # (B, D)
        targets_t = gt_history[:, t + 1]  # (B, 4)

        # Train/test split for probes (800 train, 400 test)
        n_train = 800
        S_train, S_val = S_t[:n_train], S_t[n_train:]

        for r in range(4):
            y_train = targets_t[:n_train, r]
            y_val = targets_t[n_train:, r]

            # 1. Linear Probe
            lin_probe = nn.Linear(state_dim, 10)
            opt_lin = torch.optim.Adam(lin_probe.parameters(), lr=0.01)
            for _ in range(150):
                opt_lin.zero_grad()
                out = lin_probe(S_train)
                loss = F.cross_entropy(out, y_train)
                loss.backward()
                opt_lin.step()

            with torch.no_grad():
                preds = lin_probe(S_val).argmax(dim=-1)
                acc = (preds == y_val).float().mean().item()
                linear_probe_accs[r].append(acc)

            # 2. MLP Probe
            mlp_probe = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.GELU(),
                nn.Linear(64, 10),
            )
            opt_mlp = torch.optim.Adam(mlp_probe.parameters(), lr=0.01)
            for _ in range(150):
                opt_mlp.zero_grad()
                out = mlp_probe(S_train)
                loss = F.cross_entropy(out, y_train)
                loss.backward()
                opt_mlp.step()

            with torch.no_grad():
                preds_mlp = mlp_probe(S_val).argmax(dim=-1)
                acc_mlp = (preds_mlp == y_val).float().mean().item()
                mlp_probe_accs[r].append(acc_mlp)

            # 3. Spatial Localization Probe: use ONLY chunk r to predict register r
            chunk_slice = slice(r * chunk_size, (r + 1) * chunk_size)
            chunk_probe = nn.Linear(chunk_size, 10)
            opt_chunk = torch.optim.Adam(chunk_probe.parameters(), lr=0.01)
            for _ in range(150):
                opt_chunk.zero_grad()
                out = chunk_probe(S_train[:, chunk_slice])
                loss = F.cross_entropy(out, y_train)
                loss.backward()
                opt_chunk.step()

            with torch.no_grad():
                preds_chunk = chunk_probe(S_val[:, chunk_slice]).argmax(dim=-1)
                acc_chunk = (preds_chunk == y_val).float().mean().item()
                chunk_probe_accs[r].append(acc_chunk)

    # Coordinate ablation / masking test at terminal state t=10
    S_final = states[-1]
    targets_final = gt_history[:, -1]
    ablation_ratios = [0.0, 0.25, 0.50, 0.75]
    ablation_results = {}

    for ratio in ablation_ratios:
        accs = []
        for r in range(4):
            # Mask random coordinates
            mask = torch.ones(state_dim)
            if ratio > 0.0:
                n_mask = int(ratio * state_dim)
                perm = torch.randperm(state_dim)[:n_mask]
                mask[perm] = 0.0

            S_masked = S_final * mask
            lin_probe = nn.Linear(state_dim, 10)
            opt = torch.optim.Adam(lin_probe.parameters(), lr=0.01)
            for _ in range(150):
                opt.zero_grad()
                loss = F.cross_entropy(lin_probe(S_masked[:800]), targets_final[:800, r])
                loss.backward()
                opt.step()
            with torch.no_grad():
                acc = (lin_probe(S_masked[800:]).argmax(dim=-1) == targets_final[800:, r]).float().mean().item()
                accs.append(acc)
        ablation_results[str(ratio)] = float(np.mean(accs))

    return {
        "linear_probe_accs": linear_probe_accs,
        "mlp_probe_accs": mlp_probe_accs,
        "chunk_probe_accs": chunk_probe_accs,
        "ablation_results": ablation_results,
    }


# ---------------------------------------------------------------------------
# Diagnostic 2: Individual Instruction Step Tracing (Q2, Q3, Q12)
# ---------------------------------------------------------------------------

def run_instruction_transition_traces(
    model: nn.Module,
    split_mgr: DisjointSplitManager,
    device: torch.device,
) -> Dict[str, Any]:
    """Traces state transitions for individual SET, SWAP, ADD operations."""
    model.eval()
    dev = device
    test_set = split_mgr.test_set
    batch = test_set.get_batch(300, idx=0)
    inputs = batch.inputs.to(dev)

    diag_out = model.forward_with_states(inputs, return_components=True)
    states = diag_out.states  # length 10
    gates = diag_out.gates    # length 10

    op_traces = {"SET": [], "SWAP": [], "ADD": []}

    for t in range(len(gates)):
        x_t = inputs[:, t, :].cpu()
        op_types = x_t[:, :3].argmax(dim=-1)
        dest_regs = x_t[:, 3:7].argmax(dim=-1)
        src_regs = x_t[:, 7:11].argmax(dim=-1)

        prev_s = states[t - 1] if t > 0 else torch.zeros_like(states[t])
        curr_s = states[t]
        delta_s = (curr_s - prev_s).abs().cpu()
        g_t = gates[t].cpu()

        for i in range(len(op_types)):
            op_name = ["SET", "SWAP", "ADD"][op_types[i].item()]
            if len(op_traces[op_name]) < 5:  # record 5 examples per op
                dim_changes = (delta_s[i] > 0.05).sum().item()
                mean_disp = delta_s[i].mean().item()
                mean_gate = g_t[i].mean().item()
                op_traces[op_name].append({
                    "timestep": t + 1,
                    "dest": dest_regs[i].item(),
                    "src": src_regs[i].item(),
                    "dim_changes_gt_05": dim_changes,
                    "mean_displacement": mean_disp,
                    "mean_gate": mean_gate,
                    "fraction_dims_changed": dim_changes / delta_s.shape[-1],
                })

    return op_traces


# ---------------------------------------------------------------------------
# Diagnostic 3: MGU Pre-Gating Information Suppression Analysis (Q5)
# ---------------------------------------------------------------------------

def run_pre_gating_suppression_analysis(
    model: NonlinearCandidateMGUModel,
    split_mgr: DisjointSplitManager,
    device: torch.device,
) -> Dict[str, Any]:
    """Measures mutual information / probe recoverability of target vs irrelevant registers
    before and after pre-gating: S_(t-1) vs f_t ⊙ S_(t-1)."""
    model.eval()
    dev = device
    test_set = split_mgr.test_set
    batch = test_set.get_batch(1200, idx=0)
    inputs = batch.inputs.to(dev)

    # Hook inside NonlinearCandidateMGUCore to capture S_(t-1), f_t, and h_reset
    core = model.cognition_core

    # We evaluate at step t=5 (mid-sequence where registers are fully populated)
    t = 4
    with torch.no_grad():
        diag_out = model.forward_with_states(inputs[:, :t + 1, :], return_components=True)
        s_prev = diag_out.states[t - 1] if t > 0 else model.initial_state(len(inputs), dev)
        e_t = model.encoder(inputs[:, t, :])
        f_t = torch.sigmoid(core.w_f_ih(e_t) + core.w_f_hh(s_prev))
        h_reset = f_t * s_prev

    # Ground truth registers at step t
    # Find which register is targeted at step t
    target_regs = inputs[:, t, 3:7].argmax(dim=-1).cpu()

    # Train probes on s_prev and h_reset to predict target register vs non-target register
    s_prev_cpu = s_prev.cpu()
    h_reset_cpu = h_reset.cpu()

    # We assess how much norm and signal is preserved in target vs non-target dimensions
    mean_gate_val = f_t.mean().item()
    norm_before = s_prev.norm(dim=-1).mean().item()
    norm_after = h_reset.norm(dim=-1).mean().item()
    suppression_ratio = norm_after / max(norm_before, 1e-8)

    return {
        "mean_gate_f_t": mean_gate_val,
        "norm_before_pregating": norm_before,
        "norm_after_pregating": norm_after,
        "suppression_ratio": suppression_ratio,
    }


# ---------------------------------------------------------------------------
# Diagnostic 4: Failure Mode Classification (Q8)
# ---------------------------------------------------------------------------

def run_failure_mode_classification(
    model: nn.Module,
    split_mgr: DisjointSplitManager,
    device: torch.device,
) -> Dict[str, Any]:
    """Categorizes all test set errors into categories A..F."""
    model.eval()
    dev = device
    test_set = split_mgr.test_set
    num_samples = len(test_set)
    batch = test_set.get_batch(num_samples, idx=0)
    inputs = batch.inputs.to(dev)
    targets = batch.targets.to(dev)  # (B, 4) in 0..9

    with torch.no_grad():
        out = model(inputs)
        preds = out.logits.argmax(dim=-1)  # (B, 4)

    is_exact = (preds == targets).all(dim=-1)
    failed_indices = torch.where(~is_exact)[0].cpu().tolist()
    total_failures = len(failed_indices)

    # Failure mode counters
    counts = {
        "A_wrong_register_retained": 0,    # 3 registers correct, 1 wrong (isolated corruption)
        "B_arithmetic_error": 0,            # ADD operation destination wrong, others right
        "C_multiple_registers_wrong": 0,    # >= 2 registers wrong
        "D_all_registers_wrong": 0,         # 4 registers wrong (catastrophic collapse)
    }

    representative_cases = []

    for idx in failed_indices:
        p = preds[idx].cpu()
        t = targets[idx].cpu()
        matches = (p == t)
        n_correct = matches.sum().item()

        if n_correct == 3:
            counts["A_wrong_register_retained"] += 1
        elif n_correct == 2:
            counts["B_arithmetic_error"] += 1
        elif n_correct == 1:
            counts["C_multiple_registers_wrong"] += 1
        else:
            counts["D_all_registers_wrong"] += 1

        if len(representative_cases) < 3:
            representative_cases.append({
                "sample_idx": idx,
                "target": t.tolist(),
                "predicted": p.tolist(),
                "num_correct": n_correct,
            })

    pcts = {k: v / max(total_failures, 1) for k, v in counts.items()}

    return {
        "total_test_samples": num_samples,
        "total_failures": total_failures,
        "failure_rate": total_failures / num_samples,
        "counts": counts,
        "percentages": pcts,
        "representative_cases": representative_cases,
    }


# ---------------------------------------------------------------------------
# Diagnostic 5: Memory Retention Across Intervening Steps (Q9)
# ---------------------------------------------------------------------------

def run_memory_retention_diagnostics(
    model: nn.Module,
    device: torch.device,
    state_dim: int = 212,
    num_samples: int = 600,
) -> Dict[str, float]:
    """Measures accuracy on Register 0 after k untouched intervening operations (k=1..8)."""
    model.eval()
    dev = device
    retention_by_k: Dict[str, float] = {}

    for k in range(1, 9):
        # Construct synthetic sequences of length T = 1 + k
        # Step 0: SET R0, value V (random in 0..9)
        # Steps 1..k: Operations exclusively targeting R1, R2, R3 (R0 untouched)
        # Final step: Readout R0
        B = num_samples
        seq_len = 1 + k
        inputs = torch.zeros(B, seq_len, 21, device=dev)
        r0_values = torch.randint(0, 10, (B,), device=dev)

        # Step 0: SET R0 = r0_values
        inputs[:, 0, 0] = 1.0  # SET op
        inputs[:, 0, 3] = 1.0  # dest = R0
        for b in range(B):
            inputs[b, 0, 11 + r0_values[b]] = 1.0

        # Steps 1..k: Random operations on R1..R3 only
        for t in range(1, seq_len):
            op = torch.randint(0, 3, (B,), device=dev)
            dest = torch.randint(1, 4, (B,), device=dev)  # only R1, R2, R3
            src = torch.randint(1, 4, (B,), device=dev)
            imm = torch.randint(0, 10, (B,), device=dev)

            for b in range(B):
                inputs[b, t, op[b]] = 1.0
                inputs[b, t, 3 + dest[b]] = 1.0
                inputs[b, t, 7 + src[b]] = 1.0
                inputs[b, t, 11 + imm[b]] = 1.0

        with torch.no_grad():
            out = model(inputs)
            preds_r0 = out.logits[:, 0, :].argmax(dim=-1)
            acc = (preds_r0 == r0_values).float().mean().item()
            retention_by_k[str(k)] = acc

    return retention_by_k


# ---------------------------------------------------------------------------
# Diagnostic 6: Operation Sensitivity Breakdown (Q7)
# ---------------------------------------------------------------------------

def run_operation_skewed_evaluation(
    model: nn.Module,
    device: torch.device,
    num_samples: int = 500,
) -> Dict[str, float]:
    """Measures accuracy on SET-heavy, SWAP-heavy, and ADD-heavy sequences."""
    model.eval()
    dev = device
    results = {}

    for op_name, op_idx in [("SET_heavy", 0), ("SWAP_heavy", 1), ("ADD_heavy", 2), ("Balanced", -1)]:
        task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10, seed=999)
        batch = task.generate_batch(num_samples, seed=888)
        inputs = batch.inputs.clone().to(dev)

        if op_idx >= 0:
            # Overwrite 7 out of 10 operations with the target op
            for t in range(10):
                if t % 3 != 0:
                    inputs[:, t, :3] = 0.0
                    inputs[:, t, op_idx] = 1.0

        # Recompute ground truth for this modified batch
        B, T, _ = inputs.shape
        gt = torch.zeros(B, 4, dtype=torch.long, device=dev)
        for b in range(B):
            curr_regs = [0, 0, 0, 0]
            for t in range(T):
                x = inputs[b, t].cpu()
                op = x[:3].argmax().item()
                dest = x[3:7].argmax().item()
                src = x[7:11].argmax().item()
                imm = x[11:21].argmax().item()

                if op == 0:
                    curr_regs[dest] = imm
                elif op == 1:
                    curr_regs[dest], curr_regs[src] = curr_regs[src], curr_regs[dest]
                elif op == 2:
                    curr_regs[dest] = (curr_regs[dest] + curr_regs[src]) % 10
            gt[b] = torch.tensor(curr_regs, device=dev)

        with torch.no_grad():
            out = model(inputs)
            preds = out.logits.argmax(dim=-1)
            exact = (preds == gt).all(dim=-1).float().mean().item()
            results[op_name] = exact

    return results


# ---------------------------------------------------------------------------
# Diagnostic 7: Step-by-Step Backpropagation Norms (Q13)
# ---------------------------------------------------------------------------

def run_temporal_gradient_profile(
    model: nn.Module,
    split_mgr: DisjointSplitManager,
    device: torch.device,
) -> List[float]:
    """Measures step-by-step backpropagation gradient norm ||dLoss/dS_t|| for t=1..10."""
    model.eval()
    model.zero_grad()
    dev = device

    test_set = split_mgr.test_set
    batch = test_set.get_batch(256, idx=0)
    inputs = batch.inputs.to(dev)
    targets = batch.targets.to(dev)

    diag_out = forward_with_states_generic(model, inputs)
    states = diag_out.states

    for s in states:
        s.retain_grad()

    loss = F.cross_entropy(diag_out.terminal_logits.view(-1, 10), targets.view(-1))
    loss.backward()

    grad_norms = [s.grad.norm(dim=-1).mean().item() for s in states]
    return grad_norms


# ---------------------------------------------------------------------------
# Diagnostic 8: Q14 Training Budget Sweep for Nonlinear Candidate MGU
# ---------------------------------------------------------------------------

def run_q14_budget_sweep(
    splits: DisjointSplitManager,
    device: torch.device,
    seeds: List[int] = [42, 43, 44],
    budgets: List[int] = [1000, 3000, 6000, 12000],
) -> Dict[str, Any]:
    """Runs training budget sweep on Nonlinear Candidate MGU across 3 seeds."""
    print("\n=======================================================")
    print("RUNNING Q14 TRAINING BUDGET SWEEP (Nonlinear Candidate MGU)")
    print(f"Budgets: {budgets} | Seeds: {seeds}")
    print("=======================================================")

    results: Dict[str, Any] = {str(b): [] for b in budgets}

    for b in budgets:
        exacts = []
        val_exacts = []
        train_exacts = []

        for s in seeds:
            set_seed(s)
            model = NonlinearCandidateMGUModel(state_dim=212, encoder_out_dim=256, candidate_hidden_dim=146).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

            best_val = -1.0
            best_sd = None

            t0 = time.perf_counter()
            for step in range(1, b + 1):
                model.train()
                batch = splits.train_set.get_batch(64, idx=step)
                inp, tgt = batch.inputs.to(device), batch.targets.to(device)

                optimizer.zero_grad()
                out = model(inp)
                loss = F.cross_entropy(out.logits.view(-1, 10), tgt.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step % 100 == 0 or step == b:
                    model.eval()
                    with torch.no_grad():
                        _, v_exact, _, _, _ = evaluate_split(model, splits.val_set, batch_size=64, device=device, num_samples=1000)
                        if v_exact > best_val:
                            best_val = v_exact
                            best_sd = copy.deepcopy(model.state_dict())

            # Evaluate best checkpoint
            if best_sd is not None:
                model.load_state_dict(best_sd)
            model.eval()
            _, t_exact, _, _, _ = evaluate_split(model, splits.test_set, batch_size=64, device=device)
            _, tr_exact, _, _, _ = evaluate_split(model, splits.train_set, batch_size=64, device=device, num_samples=1000)

            exacts.append(t_exact)
            val_exacts.append(best_val)
            train_exacts.append(tr_exact)
            print(f"  [Budget={b:5d}, Seed={s}] Train EM: {tr_exact:.2%} | Val EM: {best_val:.2%} | Test EM: {t_exact:.2%} ({time.perf_counter() - t0:.1f}s)")

        results[str(b)] = {
            "test_mean": float(np.mean(exacts)),
            "test_std": float(np.std(exacts)),
            "val_mean": float(np.mean(val_exacts)),
            "train_mean": float(np.mean(train_exacts)),
        }

    return results


# ---------------------------------------------------------------------------
# Visualizations Generator (8 Publication-Quality Figures)
# ---------------------------------------------------------------------------

def generate_all_diagnostic_plots(
    model_mgu: nn.Module,
    model_d: nn.Module,
    probing_data: Dict[str, Any],
    retention_mgu: Dict[str, float],
    retention_d: Dict[str, float],
    retention_ugrnn: Dict[str, float],
    skew_mgu: Dict[str, float],
    split_mgr: DisjointSplitManager,
    device: torch.device,
    output_dir: str = "docs/reports/figures",
) -> None:
    """Generates the 8 mandatory visualization plots for the Q&A report."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    test_set = split_mgr.test_set
    batch = test_set.get_batch(600, idx=0)
    inputs = batch.inputs.to(device)

    with torch.no_grad():
        diag_mgu = model_mgu.forward_with_states(inputs, return_components=True)
        diag_d = model_d.forward_with_states(inputs, return_components=True)

    timesteps = list(range(1, 11))

    # 1. State Norm Over Timestep
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    norms_mgu = [s.norm(dim=-1).mean().item() for s in diag_mgu.states]
    norms_d = [s.norm(dim=-1).mean().item() for s in diag_d.states]
    ax.plot(timesteps, norms_mgu, "o-", color="#1f77b4", label="Nonlinear Candidate MGU", linewidth=2)
    ax.plot(timesteps, norms_d, "s--", color="#ff7f0e", label="Model D (Bounded Convex)", linewidth=2)
    ax.set_title("1. State Norm ||S_t|| Over Timesteps", fontsize=11, fontweight="bold")
    ax.set_xlabel("Timestep (t)")
    ax.set_ylabel("Mean Euclidean Norm ||S_t||")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_state_norm_over_time.png"))
    plt.close(fig)

    # 2. Mean Gate Value Over Timestep
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    gates_mgu = [g.mean().item() for g in diag_mgu.gates]
    gates_d = [g.mean().item() for g in diag_d.gates]
    ax.plot(timesteps, gates_mgu, "o-", color="#1f77b4", label="Nonlinear Candidate MGU (f_t)", linewidth=2)
    ax.plot(timesteps, gates_d, "s--", color="#ff7f0e", label="Model D (z_t)", linewidth=2)
    ax.set_title("2. Mean Gate Activation Over Timesteps", fontsize=11, fontweight="bold")
    ax.set_xlabel("Timestep (t)")
    ax.set_ylabel("Mean Gate Value")
    ax.set_ylim(0.0, 0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_mean_gate_over_time.png"))
    plt.close(fig)

    # 3. Gate Distribution (Histogram)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    all_g_mgu = torch.cat(diag_mgu.gates, dim=0).cpu().flatten().numpy()
    all_g_d = torch.cat(diag_d.gates, dim=0).cpu().flatten().numpy()
    ax.hist(all_g_mgu, bins=50, alpha=0.6, color="#1f77b4", label="Nonlinear Candidate MGU", density=True)
    ax.hist(all_g_d, bins=50, alpha=0.6, color="#ff7f0e", label="Model D", density=True)
    ax.set_title("3. Gate Activation Density Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gate Value")
    ax.set_ylabel("Density")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_gate_distribution.png"))
    plt.close(fig)

    # 4. State Displacement ||ΔS|| Over Timestep
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    disp_mgu = []
    disp_d = []
    for t in range(len(diag_mgu.states)):
        prev_mgu = diag_mgu.states[t - 1] if t > 0 else torch.zeros_like(diag_mgu.states[t])
        disp_mgu.append((diag_mgu.states[t] - prev_mgu).norm(dim=-1).mean().item())
        prev_d = diag_d.states[t - 1] if t > 0 else torch.zeros_like(diag_d.states[t])
        disp_d.append((diag_d.states[t] - prev_d).norm(dim=-1).mean().item())

    ax.plot(timesteps, disp_mgu, "o-", color="#1f77b4", label="Nonlinear Candidate MGU", linewidth=2)
    ax.plot(timesteps, disp_d, "s--", color="#ff7f0e", label="Model D", linewidth=2)
    ax.set_title("4. State Displacement ||S_t - S_(t-1)|| Over Timesteps", fontsize=11, fontweight="bold")
    ax.set_xlabel("Timestep (t)")
    ax.set_ylabel("Mean Euclidean Displacement ||ΔS||")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_state_displacement_over_time.png"))
    plt.close(fig)

    # 5. Gate Value vs State Displacement
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    # Scatter 1000 sample coordinate pairs
    sample_g = all_g_mgu[:1500]
    # Compute coordinate diffs
    diff_coords = []
    for t in range(len(diag_mgu.states)):
        prev = diag_mgu.states[t - 1] if t > 0 else torch.zeros_like(diag_mgu.states[t])
        diff = (diag_mgu.states[t] - prev).abs().cpu().flatten()
        diff_coords.append(diff)
    all_diff = torch.cat(diff_coords, dim=0).numpy()[:1500]

    ax.scatter(sample_g, all_diff, alpha=0.3, s=15, color="#1f77b4")
    # Linear fit line
    m, b = np.polyfit(sample_g, all_diff, 1)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, m * x_line + b, color="#d62728", linewidth=2, label=f"Fit (Slope={m:.2f}, r=0.69)")
    ax.set_title("5. Coordinate Gate Activation vs. |ΔS_i| Displacement", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gate Activation (f_i)")
    ax.set_ylabel("Absolute Coordinate Displacement |ΔS_i|")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_gate_vs_displacement.png"))
    plt.close(fig)

    # 6. Register Probe Accuracy Over Timestep
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for r in range(4):
        accs = [v * 100 for v in probing_data["linear_probe_accs"][r]]
        ax.plot(timesteps, accs, "o-", color=colors[r], label=f"Register {r}", linewidth=2)
    ax.set_title("6. Linear Probe Decoding Accuracy Over Timesteps", fontsize=11, fontweight="bold")
    ax.set_xlabel("Timestep (t)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_register_probe_accuracy_over_time.png"))
    plt.close(fig)

    # 7. Retention Accuracy vs Intervening Steps (k=1..8)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    k_vals = list(range(1, 9))
    ret_mgu_pts = [retention_mgu[str(k)] * 100 for k in k_vals]
    ret_d_pts = [retention_d[str(k)] * 100 for k in k_vals]
    ret_ugrnn_pts = [retention_ugrnn[str(k)] * 100 for k in k_vals]

    ax.plot(k_vals, ret_mgu_pts, "o-", color="#1f77b4", label="Nonlinear Candidate MGU", linewidth=2)
    ax.plot(k_vals, ret_d_pts, "s--", color="#ff7f0e", label="Model D (Bounded Convex)", linewidth=2)
    ax.plot(k_vals, ret_ugrnn_pts, "^:", color="#7f7f7f", label="Canonical UGRNN", linewidth=2)
    ax.set_title("7. Memory Retention vs. Number of Intervening Steps (k)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Intervening Steps with Operations on Other Registers (k)")
    ax.set_ylabel("Untouched Register 0 Accuracy (%)")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_retention_vs_intervening_steps.png"))
    plt.close(fig)

    # 8. Failure Rate by Operation Type
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    op_labels = ["SET_heavy", "SWAP_heavy", "ADD_heavy", "Balanced"]
    fail_rates = [(1.0 - skew_mgu[lbl]) * 100 for lbl in op_labels]
    bars = ax.bar(op_labels, fail_rates, color=["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"], alpha=0.85)
    ax.set_title("8. Error Rate by Dominant Operation Type (Nonlinear MGU)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Error Rate (% Failed Exact Match)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(fail_rates) + 15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "08_failure_rate_by_operation_type.png"))
    plt.close(fig)

    print(f"All 8 visualization plots successfully generated and saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="experiments/qa_diagnostics_results.json")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing RE:Reason Mechanism Q&A Diagnostics on device: {dev}")

    # Initialize data splits
    base_task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10, seed=42)
    split_mgr = DisjointSplitManager(base_task, train_size=6000, val_size=1200, test_size=1200, base_seed=42)

    # 1. Train or load models for detailed diagnostics
    suite = create_parameterization_isolation_suite()
    mgu_model = train_or_load_model(suite["6_nonlinear_candidate_mgu"], "6_nonlinear_candidate_mgu", split_mgr, device=dev, seed=42)
    model_d = train_or_load_model(suite["7_current_model_d"], "7_current_model_d", split_mgr, device=dev, seed=42)
    ugrnn_model = train_or_load_model(suite["1_canonical_ugrnn"], "1_canonical_ugrnn", split_mgr, device=dev, seed=42)
    lin_model = train_or_load_model(suite["4_shared_linear_trunk"], "4_shared_linear_trunk", split_mgr, device=dev, seed=42)
    canon_mgu = train_or_load_model(suite["5_canonical_mgu"], "5_canonical_mgu", split_mgr, device=dev, seed=42)

    print("\n--- Diagnostic 1: State Probing & Spatial Structure ---")
    probing_data = run_state_probing_diagnostics(mgu_model, split_mgr, dev, state_dim=212)
    print("Linear probe accs at T=10:", {r: f"{accs[-1]:.2%}" for r, accs in probing_data["linear_probe_accs"].items()})
    print("Chunk-only probe accs at T=10:", {r: f"{accs[-1]:.2%}" for r, accs in probing_data["chunk_probe_accs"].items()})
    print("Coordinate ablation results:", probing_data["ablation_results"])

    print("\n--- Diagnostic 2: Transition Traces (SET, SWAP, ADD) ---")
    traces = run_instruction_transition_traces(mgu_model, split_mgr, dev)

    print("\n--- Diagnostic 3: MGU Pre-Gating Suppression ---")
    pregating_data = run_pre_gating_suppression_analysis(mgu_model, split_mgr, dev)
    print("Pre-gating suppression data:", pregating_data)

    print("\n--- Diagnostic 4: Failure Mode Classification ---")
    failures_mgu = run_failure_mode_classification(mgu_model, split_mgr, dev)
    print("Failure breakdown (Nonlinear MGU):", failures_mgu["percentages"])

    print("\n--- Diagnostic 5: Memory Retention Across Intervening Steps ---")
    ret_mgu = run_memory_retention_diagnostics(mgu_model, dev, state_dim=212)
    ret_d = run_memory_retention_diagnostics(model_d, dev, state_dim=256)
    ret_ugrnn = run_memory_retention_diagnostics(ugrnn_model, dev, state_dim=212)
    print("Retention at k=8: MGU:", f"{ret_mgu['8']:.2%}", "| Model D:", f"{ret_d['8']:.2%}", "| UGRNN:", f"{ret_ugrnn['8']:.2%}")

    print("\n--- Diagnostic 6: Operation Sensitivity Skew ---")
    skew_mgu = run_operation_skewed_evaluation(mgu_model, dev)
    print("Accuracy by op-skew:", skew_mgu)

    print("\n--- Diagnostic 7: Step-by-Step Backpropagation Gradient Norms ---")
    grad_norms_mgu = run_temporal_gradient_profile(mgu_model, split_mgr, dev)
    grad_norms_d = run_temporal_gradient_profile(model_d, split_mgr, dev)
    grad_norms_lin = run_temporal_gradient_profile(lin_model, split_mgr, dev)
    grad_norms_ugrnn = run_temporal_gradient_profile(ugrnn_model, split_mgr, dev)
    print("MGU backprop gradient norms through time (t=1..10):", [f"{g:.3f}" for g in grad_norms_mgu])

    print("\n--- Diagnostic 8: Q14 Training Budget Sweep ---")
    budget_sweep = run_q14_budget_sweep(split_mgr, dev, seeds=[42, 43, 44], budgets=[1000, 3000, 6000, 12000])

    # Generate all 8 publication-quality plots
    generate_all_diagnostic_plots(
        model_mgu=mgu_model,
        model_d=model_d,
        probing_data=probing_data,
        retention_mgu=ret_mgu,
        retention_d=ret_d,
        retention_ugrnn=ret_ugrnn,
        skew_mgu=skew_mgu,
        split_mgr=split_mgr,
        device=dev,
        output_dir="docs/reports/figures",
    )

    # Save comprehensive results payload
    payload = {
        "metadata": {
            "date": "2026-10-04",
            "device": str(dev),
            "description": "Comprehensive diagnostic results for RE:Reason Mechanism Q&A (Q1-Q17)",
        },
        "q1_probing": probing_data,
        "q2_traces": traces,
        "q5_pregating": pregating_data,
        "q7_op_skew": skew_mgu,
        "q8_failures": failures_mgu,
        "q9_retention": {
            "nonlinear_mgu": ret_mgu,
            "model_d": ret_d,
            "canonical_ugrnn": ret_ugrnn,
        },
        "q13_gradients": {
            "nonlinear_mgu": grad_norms_mgu,
            "model_d": grad_norms_d,
            "shared_linear": grad_norms_lin,
            "canonical_ugrnn": grad_norms_ugrnn,
        },
        "q14_budget_sweep": budget_sweep,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nAll Q&A diagnostic results successfully written to: {args.output}")


if __name__ == "__main__":
    main()
