#!/usr/bin/env python3
"""RE:Reason — Phase-1 Generator Repair Feasibility.

Question
--------
Is the ~54-58% terminal-EM ceiling of the Nonlinear Candidate MGU caused by
terminal-only supervision rather than by the recurrent transition itself?

Protocol (identical for every variant except the loss):
  * Model: NonlinearCandidateMGUModel (226,358 params) — architecture, state dim,
    candidate MLP, gate, encoder, readout, parameter budget all frozen.
  * Data: DisjointSplitManager(RegisterTrackingTask(seed=42), train 6000 / val 1200
    / test 1200, base_seed=seed) — the parameterization-isolation protocol.
  * Optimization: AdamW lr 1e-3, wd 0.01, batch 64, grad-clip 1.0, 3000 steps,
    validation every 100 steps (1000 samples), best-val checkpoint, held-out test.
  * Seeds: 42, 43, 44, 45, 46.

Supervision variants (only the loss changes):
  A_terminal    L = CE(y_T)
  B_stepwise    L = (1/T) * sum_t CE(y_t)
  C1/C2/C3      L = lam * CE(y_T) + (1-lam) * (1/T) * sum_t CE(y_t), lam in {.25,.5,.75}
  D_late        L = (1/|S|) * sum_{t>=6} CE(y_t)   (1-indexed t = 6..10)
  E_candidate   L = CE(y_T) + beta * mean_{t,written regs} CE(head(C_t)), beta = 1.
                The auxiliary term reuses the EXISTING decision head on the EXISTING
                candidate vectors C_t = tanh(cand_mlp([f_t * S_(t-1), e_t])); the
                committed-state update stays S_t = (1-f_t) S_(t-1) + f_t C_t.
  F_combined    L = CE(y_T) + lambda_step * (1/T) sum_t CE(head(S_t))
                      + beta * mean_{t,written regs} CE(head(C_t)),  lambda_step = beta = 1.
                All three terms are per-prediction mean cross-entropies (identical
                averaging convention); per-component means are logged during training.

Evaluation (all variants):
  * canonical T=10 test split: terminal EM, per-register accuracy
  * horizon sweep T in {1,2,3,5,8,10,15,20} (zero-shot; never retrained per length)
  * per-step committed/candidate decoding (full-state + written-register), gates,
    first-error chain, controlled ADD diagnostics, destination decodability probe
  * compute accounting: steps, forward passes, head evals, FLOPs, wall clock
  * 5-seed paired statistics vs Model A (paired t, Wilcoxon, Cohen's dz)

Usage:
  python run_generator_repair_feasibility.py                 # full matrix coordinator
  python run_generator_repair_feasibility.py --run-one A_terminal --seed 42
  python run_generator_repair_feasibility.py --aggregate
"""

import argparse
import copy
import glob
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from re_reason.benchmarks.dataset_split import DisjointSplitManager, SplitDataset, hash_sequence
from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.core.ablation_models import NonlinearCandidateMGUModel
from re_reason.training.diagnostic_runner import evaluate_split
from re_reason.utils.seeding import set_seed

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44, 45, 46]
TOTAL_STEPS = 3000
BATCH_SIZE = 64
LR = 1e-3
WD = 0.01
CLIP = 1.0
EVAL_INTERVAL = 100
VAL_SAMPLES = 1000
TRAIN_SIZE = 6000
VAL_SIZE = 1200
TEST_SIZE = 1200

HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20]
HORIZON_SIZES = {1: 40, 2: 1000}  # L=1 has only 40 possible sequences

CHECKPOINT = "experiments/checkpoints/6_nonlinear_candidate_mgu_seed42_3000steps.pt"
OUT_JSON = "experiments/generator_repair_results.json"
FIG_DIR = "docs/reports/figures"

STATE_DIM = 212
NUM_REGISTERS = 4
NUM_VALUES = 10

VARIANTS: Dict[str, Dict[str, Any]] = {
    "A_terminal": {"mode": "terminal", "label": "Terminal"},
    "B_stepwise": {"mode": "stepwise", "label": "Full stepwise"},
    "C1_hybrid_lam0.25": {"mode": "hybrid", "lam": 0.25, "label": "Hybrid lambda=0.25"},
    "C2_hybrid_lam0.50": {"mode": "hybrid", "lam": 0.50, "label": "Hybrid lambda=0.50"},
    "C3_hybrid_lam0.75": {"mode": "hybrid", "lam": 0.75, "label": "Hybrid lambda=0.75"},
    "D_late_t6": {"mode": "late", "late_start_0idx": 5, "label": "Late-step t>=6"},
    "E_candidate": {"mode": "candidate", "beta": 1.0, "label": "Candidate/local"},
    # Combined-objective confirmation (terminal + full stepwise + candidate-local).
    "F_combined": {"mode": "combined", "lambda_step": 1.0, "beta": 1.0,
                   "label": "Combined (term + stepwise + cand)"},
}


# ---------------------------------------------------------------------------
# Data / model construction
# ---------------------------------------------------------------------------

def build_task(seq_len: int = 10) -> RegisterTrackingTask:
    return RegisterTrackingTask(num_registers=NUM_REGISTERS, sequence_length=seq_len,
                                num_values=NUM_VALUES, seed=42)


def canonical_splits(seed: int) -> DisjointSplitManager:
    return DisjointSplitManager(build_task(10), train_size=TRAIN_SIZE, val_size=VAL_SIZE,
                                test_size=TEST_SIZE, base_seed=seed)


def horizon_test_set(seed: int, L: int) -> SplitDataset:
    """Fresh disjoint test split for horizon L (never seen in training).

    Small horizons have tiny reachable sequence spaces (L=1: 40 sequences), so the
    strict DisjointSplitManager leak check cannot be used there; we sample unique
    sequences directly from a dedicated stream instead.
    """
    if L == 10:
        return canonical_splits(seed).test_set
    size = HORIZON_SIZES.get(L, 1000)
    stream = 700000 + seed * 100 + L
    task = build_task(L)
    rng = np.random.default_rng(stream)
    max_unique = 40 if L == 1 else None
    seen, inputs, steps, finals = set(), [], [], []
    while len(inputs) < size:
        inp, st, ft = task.generate_single_sequence(rng)
        h = hash_sequence(inp)
        if h in seen:
            if max_unique is not None and len(seen) >= max_unique:
                break
            continue
        seen.add(h)
        inputs.append(inp); steps.append(st); finals.append(ft)
        if max_unique is not None and len(seen) >= max_unique:
            break
    return SplitDataset(
        torch.from_numpy(np.stack(inputs)).float(),
        torch.from_numpy(np.stack(finals)).long(),
        torch.from_numpy(np.stack(steps)).long(),
        [],
    )


def make_model() -> NonlinearCandidateMGUModel:
    return NonlinearCandidateMGUModel(
        input_dim=21, state_dim=STATE_DIM, encoder_hidden_dim=64, encoder_out_dim=256,
        candidate_hidden_dim=146, num_registers=NUM_REGISTERS, num_values=NUM_VALUES,
    )


def write_mask_from_inputs(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derives (write_mask (B,T,4) bool, op (B,T), reg_a (B,T), reg_b (B,T)) from one-hots."""
    op = inputs[..., :3].argmax(-1)
    reg_a = inputs[..., 3:7].argmax(-1)
    reg_b = inputs[..., 7:11].argmax(-1)
    mask = F.one_hot(reg_a, NUM_REGISTERS).bool()
    swap = (op == RegisterTrackingTask.OP_SWAP)
    swap_mask = F.one_hot(reg_b, NUM_REGISTERS).bool() & swap[..., None]
    return mask | swap_mask, op, reg_a, reg_b


# ---------------------------------------------------------------------------
# Supervision losses (the only thing that varies across variants)
# ---------------------------------------------------------------------------

def compute_supervision_loss(cfg: Dict[str, Any], model: NonlinearCandidateMGUModel,
                             inputs: torch.Tensor, targets: torch.Tensor,
                             step_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Returns (total loss, component info).

    All components are per-prediction mean cross-entropies (identical averaging
    convention: mean CE over the supervised scalar targets), so equal weights are
    numerically meaningful. `info` carries the per-component loss values used for
    scale inspection during training.
    """
    mode = cfg["mode"]
    B, T, _ = inputs.shape
    info: Dict[str, float] = {}

    if mode == "terminal":
        out = model(inputs)
        loss = F.cross_entropy(out.logits.reshape(-1, NUM_VALUES), targets.reshape(-1))
        info["head_evals_per_step"] = 1
        info["terminal_loss"] = float(loss.item())
        info["total_loss"] = float(loss.item())
        return loss, info

    if mode in ("stepwise", "hybrid", "late"):
        out = model(inputs, return_step_diagnostics=True)
        step_logits = out.step_logits  # (B, T, R, V)
        term = F.cross_entropy(out.logits.reshape(-1, NUM_VALUES), targets.reshape(-1))
        per_step = [
            F.cross_entropy(step_logits[:, t].reshape(-1, NUM_VALUES), step_targets[:, t].reshape(-1))
            for t in range(T)
        ]
        stepwise = sum(per_step) / T
        if mode == "stepwise":
            loss = stepwise
        elif mode == "hybrid":
            lam = float(cfg["lam"])
            loss = lam * term + (1.0 - lam) * stepwise
        else:  # late
            s = int(cfg["late_start_0idx"])
            loss = sum(per_step[s:]) / (T - s)
        info["head_evals_per_step"] = T + 1
        info["terminal_loss"] = float(term.item())
        info["stepwise_loss"] = float(stepwise.item())
        info["total_loss"] = float(loss.item())
        return loss, info

    if mode == "candidate":
        diag = model.forward_with_states(inputs, return_components=True)
        B, T = inputs.shape[0], inputs.shape[1]
        term = F.cross_entropy(diag.terminal_logits.reshape(-1, NUM_VALUES), targets.reshape(-1))
        cand = torch.stack(diag.candidates, dim=1)  # (B, T, D)
        cand_logits = model.decision_head(cand.reshape(-1, STATE_DIM)).reshape(B, T, NUM_REGISTERS, NUM_VALUES)
        mask, _, _, _ = write_mask_from_inputs(inputs)
        ce = F.cross_entropy(cand_logits.reshape(-1, NUM_VALUES), step_targets.reshape(-1),
                             reduction="none").reshape(B, T, NUM_REGISTERS)
        cand_loss = (ce * mask).sum() / mask.sum().clamp_min(1)
        loss = term + float(cfg["beta"]) * cand_loss
        info["head_evals_per_step"] = T + 1  # T candidate heads + 1 terminal head
        info["terminal_loss"] = float(term.item())
        info["candidate_loss"] = float(cand_loss.item())
        info["total_loss"] = float(loss.item())
        return loss, info

    if mode == "combined":
        # F = terminal + lambda_step * stepwise(committed, all steps) + beta * candidate(written regs)
        diag = model.forward_with_states(inputs, return_components=True)
        B, T = inputs.shape[0], inputs.shape[1]
        term = F.cross_entropy(diag.terminal_logits.reshape(-1, NUM_VALUES), targets.reshape(-1))

        states = torch.stack(diag.states, dim=1)  # (B, T, D)
        step_logits = model.decision_head(states.reshape(-1, STATE_DIM)).reshape(B, T, NUM_REGISTERS, NUM_VALUES)
        per_step = [
            F.cross_entropy(step_logits[:, t].reshape(-1, NUM_VALUES), step_targets[:, t].reshape(-1))
            for t in range(T)
        ]
        stepwise = sum(per_step) / T

        cand = torch.stack(diag.candidates, dim=1)
        cand_logits = model.decision_head(cand.reshape(-1, STATE_DIM)).reshape(B, T, NUM_REGISTERS, NUM_VALUES)
        mask, _, _, _ = write_mask_from_inputs(inputs)
        ce = F.cross_entropy(cand_logits.reshape(-1, NUM_VALUES), step_targets.reshape(-1),
                             reduction="none").reshape(B, T, NUM_REGISTERS)
        cand_loss = (ce * mask).sum() / mask.sum().clamp_min(1)

        loss = term + float(cfg["lambda_step"]) * stepwise + float(cfg["beta"]) * cand_loss
        info["head_evals_per_step"] = 2 * T + 1  # T committed heads + T candidate heads + 1 terminal
        info["terminal_loss"] = float(term.item())
        info["stepwise_loss"] = float(stepwise.item())
        info["candidate_loss"] = float(cand_loss.item())
        info["total_loss"] = float(loss.item())
        return loss, info

    raise ValueError(f"Unknown mode {mode}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_run(variant: str, seed: int, device: torch.device) -> Dict[str, Any]:
    cfg = VARIANTS[variant]
    set_seed(seed)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    splits = canonical_splits(seed)
    train_set, val_set = splits.train_set, splits.val_set

    best_val_exact, best_val_step, best_state = -1.0, -1, None
    loss_sums: Dict[str, float] = {}
    loss_steps = 0
    t0 = time.perf_counter()
    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        batch = train_set.get_batch(BATCH_SIZE, idx=step)
        inputs = batch.inputs.to(device)
        targets = batch.targets.to(device)
        step_targets = batch.metadata["step_targets"].to(device)

        optimizer.zero_grad()
        loss, info = compute_supervision_loss(cfg, model, inputs, targets, step_targets)
        for k, v in info.items():
            if k == "head_evals_per_step":
                continue
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v)
        loss_steps += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS:
            model.eval()
            with torch.no_grad():
                _, val_exact, _, _, _ = evaluate_split(model, val_set, batch_size=BATCH_SIZE,
                                                       device=device, num_samples=VAL_SAMPLES)
                if val_exact > best_val_exact:
                    best_val_exact = float(val_exact)
                    best_val_step = step
                    best_state = copy.deepcopy(model.state_dict())
    wall_clock = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "best_val_exact": best_val_exact, "best_val_step": best_val_step,
            "wall_clock_s": wall_clock, "param_count": sum(p.numel() for p in model.parameters()),
            "loss_components": {k: v / max(loss_steps, 1) for k, v in loss_sums.items()}}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def trajectory_analysis(model: NonlinearCandidateMGUModel, inputs: torch.Tensor,
                        targets: torch.Tensor, step_targets: torch.Tensor,
                        batch_size: int = 256) -> Dict[str, Any]:
    """Committed/candidate per-step decoding, gates, first-error chain, ADD stats."""
    B, T, _ = inputs.shape
    dev = inputs.device
    mask, op, reg_a, reg_b = write_mask_from_inputs(inputs)

    committed_pred = torch.zeros(B, T, NUM_REGISTERS, dtype=torch.long, device=dev)
    cand_pred = torch.zeros(B, T, NUM_REGISTERS, dtype=torch.long, device=dev)
    gate_vals = torch.zeros(B, T, STATE_DIM, device=dev)
    term_pred = torch.zeros(B, NUM_REGISTERS, dtype=torch.long, device=dev)
    state_norms = torch.zeros(B, T, device=dev)

    for i in range(0, B, batch_size):
        xb = inputs[i:i + batch_size]
        diag = model.forward_with_states(xb, return_components=True)
        states = torch.stack(diag.states, dim=1)      # (b,T,D)
        cands = torch.stack(diag.candidates, dim=1)   # (b,T,D)
        gates = torch.stack(diag.gates, dim=1)
        b = xb.shape[0]
        committed_pred[i:i + b] = model.decision_head(states.reshape(-1, STATE_DIM)).reshape(b, T, NUM_REGISTERS, NUM_VALUES).argmax(-1)
        cand_pred[i:i + b] = model.decision_head(cands.reshape(-1, STATE_DIM)).reshape(b, T, NUM_REGISTERS, NUM_VALUES).argmax(-1)
        gate_vals[i:i + b] = gates
        term_pred[i:i + b] = diag.terminal_logits.argmax(-1)
        state_norms[i:i + b] = states.norm(dim=-1)

    true = step_targets
    committed_ok = committed_pred == true           # (B,T,R)
    cand_ok = cand_pred == true

    # per-step aggregates
    committed_full_step = committed_ok.all(-1).float().mean(0)            # (T,)
    committed_written_step = (committed_ok | ~mask).all(-1).float().mean(0)
    cand_full_step = cand_ok.all(-1).float().mean(0)
    cand_written_step = (cand_ok | ~mask).all(-1).float().mean(0)
    committed_reg_step = committed_ok.float().mean((0, 2))                # per t, per-register acc
    cand_reg_step = cand_ok.float().mean((0, 2))

    # first error chain (strict = any register; real = written registers only)
    first_strict = np.full(B, -1, dtype=np.int64)
    first_real = np.full(B, -1, dtype=np.int64)
    cs = (~committed_ok.all(-1)).cpu().numpy()
    cr = (~((committed_ok | ~mask).all(-1))).cpu().numpy()
    for b in range(B):
        w = np.nonzero(cs[b])[0]
        if w.size:
            first_strict[b] = int(w[0]) + 1
        w = np.nonzero(cr[b])[0]
        if w.size:
            first_real[b] = int(w[0]) + 1

    # origin analysis at first real error
    # origin 0 = candidate wrong (generator), 1 = candidate right but committed wrong (commit/readout)
    origins = {"candidate_error": 0, "commit_or_readout_error": 0}
    err_op_counts = {k: 0 for k in ("SET", "SWAP", "ADD_no_wrap", "ADD_wrap")}
    err_reg_counts = {f"R{r}": 0 for r in range(NUM_REGISTERS)}
    prev_true = torch.zeros_like(true)
    prev_true[:, 1:] = true[:, :-1]
    for b in range(B):
        t = first_real[b]
        if t < 0:
            continue
        t0 = t - 1
        wrong_regs = [r for r in range(NUM_REGISTERS)
                      if mask[b, t0, r].item() and committed_pred[b, t0, r].item() != true[b, t0, r].item()]
        any_cand_wrong = any(cand_pred[b, t0, r].item() != true[b, t0, r].item() for r in wrong_regs)
        origins["candidate_error" if any_cand_wrong else "commit_or_readout_error"] += 1
        o = int(op[b, t0].item())
        if o == RegisterTrackingTask.OP_ADD:
            a_, b_ = int(reg_a[b, t0].item()), int(reg_b[b, t0].item())
            wrap = bool((prev_true[b, t0, a_] + prev_true[b, t0, b_]).item() >= NUM_VALUES)
            err_op_counts["ADD_wrap" if wrap else "ADD_no_wrap"] += 1
        elif o == RegisterTrackingTask.OP_SWAP:
            err_op_counts["SWAP"] += 1
        else:
            err_op_counts["SET"] += 1
        for r in wrong_regs:
            err_reg_counts[f"R{r}"] += 1

    # ADD transition diagnostics (committed + candidate written-dest accuracy)
    add_stats: Dict[str, Dict[str, float]] = {}
    for t in range(1, T):
        sel = (op[:, t] == RegisterTrackingTask.OP_ADD)
        if not sel.any():
            continue
        repeated = sel & (op[:, t - 1] == RegisterTrackingTask.OP_ADD)
        isolated = sel & ~repeated
        a_idx = reg_a[:, t]
        dest_ok_committed = committed_pred[:, t].gather(1, a_idx[:, None]).squeeze(1) == true[:, t].gather(1, a_idx[:, None]).squeeze(1)
        dest_ok_cand = cand_pred[:, t].gather(1, a_idx[:, None]).squeeze(1) == true[:, t].gather(1, a_idx[:, None]).squeeze(1)
        wrap = (prev_true[:, t].gather(1, a_idx[:, None]).squeeze(1)
                + prev_true[:, t].gather(1, reg_b[:, t, None]).squeeze(1)) >= NUM_VALUES
        for key, s in (("all", sel), ("isolated", isolated), ("repeated", repeated),
                       ("no_wrap", sel & ~wrap), ("wrap", sel & wrap)):
            if s.sum() > 0:
                add_stats.setdefault(key, {"n": 0, "committed_dest_ok": 0.0, "candidate_dest_ok": 0.0})
                add_stats[key]["n"] += int(s.sum())
                add_stats[key]["committed_dest_ok"] += float(dest_ok_committed[s].sum())
                add_stats[key]["candidate_dest_ok"] += float(dest_ok_cand[s].sum())
    for k in add_stats:
        n = max(add_stats[k]["n"], 1)
        add_stats[k]["committed_dest_ok"] /= n
        add_stats[k]["candidate_dest_ok"] /= n

    # gates
    eps = 1e-7
    g = gate_vals.clamp(eps, 1 - eps)
    gate_entropy = float((-(g * g.log2() + (1 - g) * (1 - g).log2())).mean().item())
    gate_mean = float(gate_vals.mean().item())
    frac_near0 = float((gate_vals < 0.1).float().mean().item())
    frac_near1 = float((gate_vals > 0.9).float().mean().item())
    op_gate_means = {}
    for name, oid in (("SET", 0), ("SWAP", 1), ("ADD", 2)):
        sel = op == oid
        op_gate_means[name] = float(gate_vals[sel].mean().item()) if sel.any() else float("nan")

    term_ok = (term_pred == targets).all(-1)
    per_reg = float((term_pred == targets).float().mean().item())
    exact = float(term_ok.float().mean().item())

    return {
        "terminal_em": exact,
        "per_register_accuracy": per_reg,
        "committed_full_step": committed_full_step.tolist(),
        "committed_written_step": committed_written_step.tolist(),
        "candidate_full_step": cand_full_step.tolist(),
        "candidate_written_step": cand_written_step.tolist(),
        "committed_reg_step": committed_reg_step.tolist(),
        "candidate_reg_step": cand_reg_step.tolist(),
        "committed_full_mean": float(committed_full_step.mean().item()),
        "committed_written_mean": float(committed_written_step.mean().item()),
        "candidate_full_mean": float(cand_full_step.mean().item()),
        "candidate_written_mean": float(cand_written_step.mean().item()),
        "first_strict_hist": {str(t): int((first_strict == t).sum()) for t in list(range(1, T + 1)) + [-1]},
        "first_real_hist": {str(t): int((first_real == t).sum()) for t in list(range(1, T + 1)) + [-1]},
        "first_strict_error_rate": float((first_strict > 0).mean()),
        "first_real_error_rate": float((first_real > 0).mean()),
        "mean_state_drift": float(1.0 - committed_reg_step.mean().item()),
        "origin_counts": origins,
        "first_error_op_counts": err_op_counts,
        "first_error_reg_counts": err_reg_counts,
        "add_transition_stats": add_stats,
        "gate": {"mean": gate_mean, "entropy": gate_entropy, "frac_near0": frac_near0,
                 "frac_near1": frac_near1, "op_means": op_gate_means},
        "_committed_pred": committed_pred.cpu().numpy(),
        "_cand_pred": cand_pred.cpu().numpy(),
        "_gate_vals": gate_vals.cpu().numpy(),
    }


@torch.no_grad()
def gate_displacement_corr(model: NonlinearCandidateMGUModel, inputs: torch.Tensor,
                           batch_size: int = 256) -> float:
    """Mean cosine correlation between per-step gate vector and |S_t - S_(t-1)| state displacement."""
    B, T, _ = inputs.shape
    corrs = []
    for i in range(0, B, batch_size):
        xb = inputs[i:i + batch_size]
        diag = model.forward_with_states(xb, return_components=True)
        states = torch.stack(diag.states, dim=1)
        gates = torch.stack(diag.gates, dim=1)
        prev = torch.cat([torch.zeros_like(states[:, :1]), states[:, :-1]], dim=1)
        disp = (states - prev).abs()
        for t in range(T):
            vx = gates[:, t] - gates[:, t].mean(dim=-1, keepdim=True)
            vy = disp[:, t] - disp[:, t].mean(dim=-1, keepdim=True)
            c = (vx * vy).sum(-1) / (torch.sqrt((vx ** 2).sum(-1) * (vy ** 2).sum(-1)) + 1e-8)
            corrs.append(c.mean().item())
    return float(np.mean(corrs))


def destination_decodability(model: NonlinearCandidateMGUModel, train_set: SplitDataset,
                             test_gates: torch.Tensor, test_op: torch.Tensor,
                             test_reg_a: torch.Tensor, device: torch.device,
                             num_train_seqs: int = 1000) -> float:
    """Ridge one-hot probe from gate vector f_t to destination register reg_a (4-way)."""
    batch = train_set.get_batch(num_train_seqs, idx=0)
    xin = batch.inputs.to(device)
    with torch.no_grad():
        diag = model.forward_with_states(xin, return_components=True)
        gtr = torch.stack(diag.gates, dim=1)                     # (N,T,D)
    _, _, ra_tr, _ = write_mask_from_inputs(xin)
    X = gtr.reshape(-1, STATE_DIM)
    Y = F.one_hot(ra_tr.reshape(-1) % NUM_REGISTERS, NUM_REGISTERS).float()
    XtX = X.t() @ X + 1e-3 * torch.eye(STATE_DIM, device=device)
    W = torch.linalg.solve(XtX, X.t() @ Y)
    Xte = test_gates.reshape(-1, STATE_DIM)
    pred = (Xte @ W).argmax(-1)
    acc = (pred == (test_reg_a.reshape(-1) % NUM_REGISTERS)).float().mean().item()
    return float(acc)


@torch.no_grad()
def horizon_em(model: NonlinearCandidateMGUModel, dataset: SplitDataset,
               batch_size: int = 256) -> Dict[str, float]:
    inputs = dataset.inputs
    targets = dataset.targets
    N = inputs.shape[0]
    correct = 0
    per_reg = 0.0
    for i in range(0, N, batch_size):
        xb = inputs[i:i + batch_size].to(next(model.parameters()).device)
        tb = targets[i:i + batch_size].to(next(model.parameters()).device)
        out = model(xb)
        pred = out.logits.argmax(-1)
        correct += int((pred == tb).all(-1).sum().item())
        per_reg += float((pred == tb).float().sum().item())
    return {"em": correct / N, "per_register": per_reg / (N * NUM_REGISTERS)}


# ---------------------- controlled ADD diagnostics ------------------------

def _build_controlled_add_sequences(n_samples: int = 400, seed: int = 42) -> Dict[str, np.ndarray]:
    """Same builders as the Round-2 audit Q10 (10-step sequences, fixed procedural rng)."""
    task = build_task(10)
    rng = np.random.default_rng(seed)

    def single_add():
        seq = [(task.OP_SET, r, -1, int(rng.integers(0, 10))) for r in range(4)]
        seq += [(task.OP_SET, int(rng.integers(0, 4)), -1, int(rng.integers(0, 10))) for _ in range(5)]
        seq.append((task.OP_ADD, int(rng.integers(0, 4)), int(rng.integers(0, 4)), -1))
        return seq

    def add_then_set():
        seq = [(task.OP_SET, r, -1, int(rng.integers(0, 10))) for r in range(4)]
        seq += [(task.OP_SET, int(rng.integers(0, 4)), -1, int(rng.integers(0, 10))) for _ in range(4)]
        seq.append((task.OP_ADD, int(rng.integers(0, 4)), int(rng.integers(0, 4)), -1))
        seq.append((task.OP_SET, int(rng.integers(0, 4)), -1, int(rng.integers(0, 10))))
        return seq

    def four_adds():
        seq = [(task.OP_SET, r, -1, int(rng.integers(0, 10))) for r in range(4)]
        seq += [(task.OP_SET, int(rng.integers(0, 4)), -1, int(rng.integers(0, 10))) for _ in range(2)]
        seq += [(task.OP_ADD, int(rng.integers(0, 4)), int(rng.integers(0, 4)), -1) for _ in range(4)]
        return seq

    def wrap():
        r0 = int(rng.integers(5, 10)); r1 = int(rng.integers(5, 10))
        seq = [(task.OP_SET, 0, -1, r0), (task.OP_SET, 1, -1, r1), (task.OP_SET, 2, -1, 3), (task.OP_SET, 3, -1, 4)]
        seq += [(task.OP_SET, 2, -1, int(rng.integers(0, 10))) for _ in range(5)]
        seq.append((task.OP_ADD, 0, 1, -1))
        return seq

    def no_wrap():
        r0 = int(rng.integers(0, 5)); r1 = int(rng.integers(0, 5))
        seq = [(task.OP_SET, 0, -1, r0), (task.OP_SET, 1, -1, r1), (task.OP_SET, 2, -1, 3), (task.OP_SET, 3, -1, 4)]
        seq += [(task.OP_SET, 2, -1, int(rng.integers(0, 10))) for _ in range(5)]
        seq.append((task.OP_ADD, 0, 1, -1))
        return seq

    out = {}
    for name, fn in (("isolated_add", single_add), ("add_then_set", add_then_set),
                     ("repeated_add", four_adds), ("add_no_wrap", no_wrap), ("add_wrap", wrap)):
        arr = np.zeros((n_samples, 10, task.input_dim), dtype=np.float32)
        tgt_step = np.zeros((n_samples, 10, NUM_REGISTERS), dtype=np.int64)
        tgt_final = np.zeros((n_samples, NUM_REGISTERS), dtype=np.int64)
        add_step = np.zeros(n_samples, dtype=np.int64)
        for i in range(n_samples):
            seq = fn()
            regs = [0] * NUM_REGISTERS
            last_add = -1
            for t, (o, d, s, v) in enumerate(seq):
                arr[i, t] = task._encode_operation(o, d, s, v)
                if o == task.OP_SET:
                    regs[d] = v
                elif o == task.OP_SWAP:
                    regs[d], regs[s] = regs[s], regs[d]
                elif o == task.OP_ADD:
                    regs[d] = (regs[d] + regs[s]) % NUM_VALUES
                    last_add = t
                tgt_step[i, t] = regs
            tgt_final[i] = regs
            add_step[i] = last_add
        out[name] = {"inputs": arr, "step_targets": tgt_step, "final_targets": tgt_final,
                     "add_step": add_step}
    return out


@torch.no_grad()
def controlled_add_analysis(model: NonlinearCandidateMGUModel, device: torch.device) -> Dict[str, Any]:
    seqs = _build_controlled_add_sequences()
    res = {}
    for name, d in seqs.items():
        inputs = torch.from_numpy(d["inputs"]).to(device)
        step_tgt = torch.from_numpy(d["step_targets"]).to(device)
        final_tgt = torch.from_numpy(d["final_targets"]).to(device)
        add_step = d["add_step"]
        diag = model.forward_with_states(inputs, return_components=True)
        states = torch.stack(diag.states, dim=1)
        cands = torch.stack(diag.candidates, dim=1)
        B = inputs.shape[0]
        cp = model.decision_head(states.reshape(-1, STATE_DIM)).reshape(B, 10, NUM_REGISTERS, NUM_VALUES).argmax(-1)
        kp = model.decision_head(cands.reshape(-1, STATE_DIM)).reshape(B, 10, NUM_REGISTERS, NUM_VALUES).argmax(-1)
        # destination register and accuracy at the last ADD step
        reg_a = inputs[..., 3:7].argmax(-1)
        rows = torch.arange(B, device=device)
        t = torch.from_numpy(add_step).to(device).clamp_min(0)
        dest = reg_a[rows, t]
        committed_ok = (cp[rows, t, dest] == step_tgt[rows, t, dest]).float().mean().item()
        cand_ok = (kp[rows, t, dest] == step_tgt[rows, t, dest]).float().mean().item()
        res[name] = {
            "terminal_em": float((model(inputs).logits.argmax(-1) == final_tgt).all(-1).float().mean().item()),
            "committed_dest_acc_at_add": float(committed_ok),
            "candidate_dest_acc_at_add": float(cand_ok),
        }
    return res


# ---------------------------------------------------------------------------
# Per-run orchestration
# ---------------------------------------------------------------------------

def evaluate_run(model: NonlinearCandidateMGUModel, seed: int, device: torch.device,
                 include_heavy: bool = True) -> Dict[str, Any]:
    splits = canonical_splits(seed)
    test_set = splits.test_set
    batch = test_set.get_batch(TEST_SIZE, idx=0)
    inputs = batch.inputs.to(device)
    targets = batch.targets.to(device)
    step_targets = batch.metadata["step_targets"].to(device)

    traj = trajectory_analysis(model, inputs, targets, step_targets)
    # keep heavy prediction arrays out of the run record (saved separately if needed)
    committed_pred = traj.pop("_committed_pred")
    cand_pred = traj.pop("_cand_pred")
    gate_vals = traj.pop("_gate_vals")

    traj["gate_displacement_corr"] = gate_displacement_corr(model, inputs)
    _, op, reg_a, _ = write_mask_from_inputs(inputs)
    t_gate = torch.from_numpy(gate_vals).to(device)
    traj["destination_decodability"] = destination_decodability(
        model, splits.train_set, t_gate, op, reg_a, device)

    # per-op terminal EM by the FINAL operation of each sequence
    op_last = op[:, -1]
    prev_true = torch.zeros_like(step_targets)
    prev_true[:, 1:] = step_targets[:, :-1]
    reg_b_last = inputs[:, -1, 7:11].argmax(-1)
    wrap_last = (prev_true[:, -1].gather(1, reg_a[:, -1, None]).squeeze(1)
                 + prev_true[:, -1].gather(1, reg_b_last[:, None]).squeeze(1)) >= NUM_VALUES
    term_ok = (model(inputs).logits.argmax(-1) == targets).all(-1)
    by_op = {}
    for name, cond in (
        ("SET", op_last == 0),
        ("SWAP", op_last == 1),
        ("ADD_no_wrap", (op_last == 2) & ~wrap_last),
        ("ADD_wrap", (op_last == 2) & wrap_last),
    ):
        by_op[name] = float(term_ok[cond].float().mean().item()) if cond.any() else float("nan")

    # horizon sweep (zero-shot)
    horizons = {}
    for L in HORIZONS:
        if L == 10:
            hres = {"em": traj["terminal_em"], "per_register": traj["per_register_accuracy"]}
        else:
            ds = horizon_test_set(seed, L)
            hres = horizon_em(model, ds)
        horizons[str(L)] = hres

    add_controlled = controlled_add_analysis(model, device) if include_heavy else {}

    return {
        "terminal_em": traj["terminal_em"],
        "per_register_accuracy": traj["per_register_accuracy"],
        "trajectory": traj,
        "by_final_op": by_op,
        "horizons": horizons,
        "controlled_add": add_controlled,
    }


def run_single(variant: str, seed: int, device: torch.device) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out = train_run(variant, seed, device)
    model = out.pop("model")
    evals = evaluate_run(model, seed, device)
    record = {"variant": variant, "seed": seed, "cfg": VARIANTS[variant],
              "train": {k: v for k, v in out.items()}, **evals}
    record["total_wall_clock_s"] = time.perf_counter() - t0
    return record, model


# ---------------------------------------------------------------------------
# Aggregation / statistics
# ---------------------------------------------------------------------------

def paired_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired comparison a (variant) vs b (baseline), per-seed arrays."""
    d = a - b
    n = len(d)
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1)) if n > 1 else 0.0
    out = {"mean_diff": mean_d, "sd_diff": sd_d}
    try:
        from scipy import stats as st
        t_stat, p_t = st.ttest_rel(a, b)
        out["paired_t_p"] = float(p_t)
        out["paired_t_stat"] = float(t_stat)
        if np.allclose(d, 0.0):
            out["wilcoxon_p"] = float("nan")
        else:
            try:
                _, p_w = st.wilcoxon(a, b)
                out["wilcoxon_p"] = float(p_w)
            except ValueError:
                out["wilcoxon_p"] = float("nan")
    except Exception as exc:  # pragma: no cover
        out["stats_error"] = str(exc)
    out["cohens_dz"] = mean_d / sd_d if sd_d > 0 else float("nan")
    return out


def aggregate(partials_dir: str, output_path: str, fig_prefix: str = "generator_repair") -> Dict[str, Any]:
    records = []
    for path in sorted(glob.glob(os.path.join(partials_dir, "*.json"))):
        with open(path) as f:
            records.append(json.load(f))
    if not records:
        raise SystemExit(f"No partial results found in {partials_dir}")

    variants = [v for v in VARIANTS if any(r["variant"] == v for r in records)]
    per_variant: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}
    for r in records:
        per_variant[r["variant"]].append(r)
    for v in variants:
        per_variant[v].sort(key=lambda r: r["seed"])

    def arr(v, fn):
        return np.array([fn(r) for r in per_variant[v]], dtype=float)

    key_metrics = {
        "terminal_em": lambda r: r["terminal_em"],
        "per_register_accuracy": lambda r: r["per_register_accuracy"],
        "horizon_T20": lambda r: r["horizons"]["20"]["em"],
        "horizon_mean": lambda r: float(np.mean([r["horizons"][str(L)]["em"] for L in HORIZONS])),
        "candidate_full_mean": lambda r: r["trajectory"]["candidate_full_mean"],
        "candidate_written_mean": lambda r: r["trajectory"]["candidate_written_mean"],
        "committed_full_mean": lambda r: r["trajectory"]["committed_full_mean"],
        "committed_written_mean": lambda r: r["trajectory"]["committed_written_mean"],
        "first_strict_error_rate": lambda r: r["trajectory"]["first_strict_error_rate"],
        "first_real_error_rate": lambda r: r["trajectory"]["first_real_error_rate"],
        "mean_state_drift": lambda r: r["trajectory"]["mean_state_drift"],
        "gate_mean": lambda r: r["trajectory"]["gate"]["mean"],
        "gate_entropy": lambda r: r["trajectory"]["gate"]["entropy"],
        "destination_decodability": lambda r: r["trajectory"]["destination_decodability"],
        "gate_displacement_corr": lambda r: r["trajectory"]["gate_displacement_corr"],
    }

    summary: Dict[str, Any] = {}
    for v in variants:
        summary[v] = {"n_seeds": len(per_variant[v]), "label": VARIANTS[v]["label"]}
        for name, fn in key_metrics.items():
            vals = arr(v, fn)
            summary[v][name] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                                "per_seed": [float(x) for x in vals]}
        # horizons mean/sd
        summary[v]["horizons"] = {
            str(L): {"mean": float(arr(v, lambda r, L=L: r["horizons"][str(L)]["em"]).mean()),
                     "sd": float(arr(v, lambda r, L=L: r["horizons"][str(L)]["em"]).std(ddof=1)) if len(per_variant[v]) > 1 else 0.0}
            for L in HORIZONS}
        summary[v]["by_final_op"] = {
            op: {"mean": float(arr(v, lambda r, op=op: r["by_final_op"][op]).mean()),
                 "sd": float(arr(v, lambda r, op=op: r["by_final_op"][op]).std(ddof=1)) if len(per_variant[v]) > 1 else 0.0}
            for op in ("SET", "SWAP", "ADD_no_wrap", "ADD_wrap")}
        summary[v]["controlled_add"] = {
            k: {"terminal_em": float(arr(v, lambda r, k=k: r["controlled_add"][k]["terminal_em"]).mean()),
                "committed_dest_acc_at_add": float(arr(v, lambda r, k=k: r["controlled_add"][k]["committed_dest_acc_at_add"]).mean()),
                "candidate_dest_acc_at_add": float(arr(v, lambda r, k=k: r["controlled_add"][k]["candidate_dest_acc_at_add"]).mean())}
            for k in ("isolated_add", "add_then_set", "repeated_add", "add_no_wrap", "add_wrap")}

    stats_vs_A = {}
    for v in variants:
        if v == "A_terminal":
            continue
        stats_vs_A[v] = {}
        for name, fn in key_metrics.items():
            stats_vs_A[v][name] = paired_stats(arr(v, fn), arr("A_terminal", fn))

    # compute accounting
    per_step_flops = 452598  # audited: one recurrent step incl. one head eval (PI runner)
    head_flops = 2 * STATE_DIM * (NUM_REGISTERS * NUM_VALUES) + NUM_REGISTERS * NUM_VALUES
    recurrence_flops = per_step_flops - head_flops
    compute = {}
    for v in variants:
        heads = {"A_terminal": 1, "B_stepwise": 11, "C1_hybrid_lam0.25": 11, "C2_hybrid_lam0.50": 11,
                 "C3_hybrid_lam0.75": 11, "D_late_t6": 11, "E_candidate": 11, "F_combined": 21}[v]
        train_flops = 3.0 * TOTAL_STEPS * BATCH_SIZE * (10 * recurrence_flops + heads * head_flops)
        compute[v] = {
            "runs": len(per_variant[v]),
            "optimization_steps_total": TOTAL_STEPS * len(per_variant[v]),
            "train_sample_passes": TOTAL_STEPS * BATCH_SIZE * len(per_variant[v]),
            "recurrence_forward_passes": TOTAL_STEPS * len(per_variant[v]),
            "head_evals_per_train_step": heads,
            "approx_train_flops_per_run": train_flops,
            "approx_train_flops_total": train_flops * len(per_variant[v]),
            "inference_flops_per_episode": 10 * per_step_flops,
            "wall_clock_s_total": float(sum(r["total_wall_clock_s"] for r in per_variant[v])),
        }

    payload = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "scope": "Phase-1 generator repair feasibility: supervision objective only",
            "checkpoint_reference": CHECKPOINT,
            "protocol": {"steps": TOTAL_STEPS, "batch": BATCH_SIZE, "lr": LR, "wd": WD,
                         "clip": CLIP, "seeds": SEEDS, "train": TRAIN_SIZE, "val": VAL_SIZE,
                         "test": TEST_SIZE, "eval_interval": EVAL_INTERVAL},
            "variants": {k: VARIANTS[k] for k in variants},
            "horizons": HORIZONS,
        },
        "summary": summary,
        "stats_vs_A": stats_vs_A,
        "compute": compute,
        "runs": [
            {k: v for k, v in r.items() if k != "trajectory_arrays"} for r in records
        ],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    make_figures(payload, fig_prefix=fig_prefix)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_figures(payload: Dict[str, Any], fig_prefix: str = "generator_repair") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(FIG_DIR, exist_ok=True)
        runs = payload["runs"]
        variants = [v for v in VARIANTS if any(r["variant"] == v for r in runs)]
        by_var = {v: [r for r in runs if r["variant"] == v] for v in variants}

        # Best hybrid by terminal EM for the comparison curve.
        hybrids = [v for v in variants if v.startswith("C")]
        best_hybrid = max(hybrids, key=lambda v: np.mean([r["terminal_em"] for r in by_var[v]])) if hybrids else None

        # Fig 1: zero-shot horizon curves
        plt.figure(figsize=(7.2, 4.4))
        show = [v for v in ["A_terminal", "B_stepwise", best_hybrid, "D_late_t6", "E_candidate", "F_combined"] if v]
        for v in show:
            rs = by_var[v]
            ms = [float(np.mean([r["horizons"][str(L)]["em"] for r in rs])) for L in HORIZONS]
            sds = [float(np.std([r["horizons"][str(L)]["em"] for r in rs], ddof=1)) if len(rs) > 1 else 0.0 for L in HORIZONS]
            plt.errorbar(HORIZONS, ms, yerr=sds, marker="o", capsize=3, label=VARIANTS[v]["label"])
        plt.xlabel("sequence length T (zero-shot; trained at T=10)")
        plt.ylabel("exact match")
        plt.xticks(HORIZONS)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.title("Horizon generalization by supervision")
        plt.savefig(os.path.join(FIG_DIR, f"{fig_prefix}_horizons.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Fig 2: per-step committed per-register accuracy (T=10 canonical test)
        plt.figure(figsize=(7.2, 4.4))
        for v in [x for x in ["A_terminal", "B_stepwise", best_hybrid, "D_late_t6", "E_candidate", "F_combined"] if x]:
            rs = by_var[v]
            curves = np.array([r["trajectory"]["committed_reg_step"] for r in rs])
            plt.plot(range(1, 11), curves.mean(0), marker="o", label=VARIANTS[v]["label"])
        plt.xlabel("timestep t")
        plt.ylabel("committed per-register accuracy")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.title("Per-step committed decoding (T=10 test)")
        plt.savefig(os.path.join(FIG_DIR, f"{fig_prefix}_stepwise.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Fig 3: T=10 and T=20 EM bars
        plt.figure(figsize=(8.6, 4.0))
        width = 0.38
        xs = np.arange(len(variants))
        t10 = [np.mean([r["terminal_em"] for r in by_var[v]]) for v in variants]
        t20 = [np.mean([r["horizons"]["20"]["em"] for r in by_var[v]]) for v in variants]
        plt.bar(xs - width / 2, t10, width, label="T=10 (trained horizon)")
        plt.bar(xs + width / 2, t20, width, label="T=20 (zero-shot)")
        plt.xticks(xs, [VARIANTS[v]["label"] for v in variants], rotation=20, ha="right", fontsize=8)
        plt.ylabel("exact match")
        plt.legend(fontsize=8)
        plt.title("Terminal EM by supervision variant")
        plt.grid(alpha=0.3, axis="y")
        plt.savefig(os.path.join(FIG_DIR, f"{fig_prefix}_em.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover
        print(f"Figure generation skipped: {exc}")


def fmt(x: float, p: int = 3) -> str:
    return "N/A" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}f}"


def print_tables(payload: Dict[str, Any]) -> None:
    summ = payload["summary"]
    order = [v for v in VARIANTS if v in summ]
    print("\nPRIMARY TABLE — zero-shot horizon EM (mean over seeds)")
    print("| Model | Supervision | T1 | T2 | T5 | T10 | T20 | Overall |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for v in order:
        s = summ[v]
        t = {L: s["horizons"][str(L)]["mean"] for L in HORIZONS}
        overall = float(np.mean([t[L] for L in HORIZONS]))
        print(f"| {v} | {s['label']} | {fmt(t[1])} | {fmt(t[2])} | {fmt(t[5])} | {fmt(t[10])} | {fmt(t[20])} | {fmt(overall)} |")

    print("\nSECOND TABLE — terminal EM at T=10 by final operation (mean over seeds)")
    print("| Model | SET | SWAP | ADD no-wrap | ADD wrap | Mean |")
    print("|---|---:|---:|---:|---:|---:|")
    for v in order:
        s = summ[v]
        o = s["by_final_op"]
        mean = s["terminal_em"]["mean"]
        print(f"| {s['label']} | {fmt(o['SET']['mean'])} | {fmt(o['SWAP']['mean'])} | "
              f"{fmt(o['ADD_no_wrap']['mean'])} | {fmt(o['ADD_wrap']['mean'])} | {fmt(mean)} |")

    print("\nTHIRD TABLE — candidate vs committed intermediate decoding (mean over seeds)")
    print("| Model | Candidate EM | Committed EM | First-error rate | Mean state drift |")
    print("|---|---:|---:|---:|---:|")
    for v in order:
        s = summ[v]
        print(f"| {s['label']} | {fmt(s['candidate_full_mean']['mean'])} | {fmt(s['committed_full_mean']['mean'])} | "
              f"{fmt(s['first_strict_error_rate']['mean'])} | {fmt(s['mean_state_drift']['mean'])} |")

    print("\nSTATISTICS vs Model A (paired by seed): terminal EM @T=10")
    print("| Model | mean +/- sd | delta vs A | paired-t p | Wilcoxon p | Cohen dz |")
    print("|---|---:|---:|---:|---:|---:|")
    for v in order:
        s = summ[v]
        if v == "A_terminal":
            print(f"| {s['label']} | {fmt(s['terminal_em']['mean'])} +/- {fmt(s['terminal_em']['sd'])} | - | - | - | - |")
            continue
        st = payload["stats_vs_A"][v]["terminal_em"]
        print(f"| {s['label']} | {fmt(s['terminal_em']['mean'])} +/- {fmt(s['terminal_em']['sd'])} | "
              f"{st['mean_diff']:+.4f} | {fmt(st.get('paired_t_p'))} | {fmt(st.get('wilcoxon_p'))} | {fmt(st.get('cohens_dz'))} |")

    print("\nSUPPORTING METRICS (mean over seeds)")
    print("| Model | cand-write | comm-write | first-real-err | gate mean | gate ent | dest-decodable | gate|cand T20 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in order:
        s = summ[v]
        t20 = s["horizons"]["20"]["mean"]
        print(f"| {s['label']} | {fmt(s['candidate_written_mean']['mean'])} | {fmt(s['committed_written_mean']['mean'])} | "
              f"{fmt(s['first_real_error_rate']['mean'])} | {fmt(s['gate_mean']['mean'])} | {fmt(s['gate_entropy']['mean'])} | "
              f"{fmt(s['destination_decodability']['mean'])} | {fmt(s['gate_displacement_corr']['mean'])} | {fmt(t20)} |")


def run_worker(variant: str, seed: int, device_str: str, partials_dir: str) -> None:
    device = torch.device(device_str)
    record, model = run_single(variant, seed, device)
    os.makedirs(partials_dir, exist_ok=True)
    out_path = os.path.join(partials_dir, f"{variant}__seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    # also persist the trained weights for potential re-analysis (scratch)
    model_dir = os.path.join(partials_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f"{variant}__seed{seed}.pt"))
    print(f"[done] {variant} seed={seed} EM={record['terminal_em']:.4f} "
          f"val={record['train']['best_val_exact']:.4f} wall={record['total_wall_clock_s']:.0f}s")


def run_matrix(args) -> None:
    os.makedirs(args.partials_dir, exist_ok=True)
    os.makedirs(os.path.join(args.partials_dir, "logs"), exist_ok=True)
    variants = args.variants or list(VARIANTS)
    seeds = args.seeds or SEEDS
    jobs = [(v, s) for v in variants for s in seeds]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(args.partials_dir, f"{j[0]}__seed{j[1]}.json"))]
    print(f"Coordinator: {len(jobs)} jobs, {args.workers} workers, device={args.device}")
    running: List[Tuple[subprocess.Popen, Tuple[str, int]]] = []
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    while jobs or running:
        while jobs and len(running) < args.workers:
            v, s = jobs.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "--run-one", v, "--seed", str(s),
                   "--partials-dir", args.partials_dir, "--device", args.device]
            log = open(os.path.join(args.partials_dir, "logs", f"{v}__seed{s}.log"), "w")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, (v, s)))
            print(f"  launched {v} seed={s} (pid {proc.pid})")
        time.sleep(2)
        for proc, (v, s) in running[:]:
            if proc.poll() is not None:
                running.remove((proc, (v, s)))
                status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"  finished {v} seed={s}: {status}")
    print("All workers finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator-repair feasibility experiment")
    parser.add_argument("--run-one", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--partials-dir", type=str, default="experiments/generator_repair_partials")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--output", type=str, default=OUT_JSON)
    parser.add_argument("--fig-prefix", type=str, default="generator_repair")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    if args.run_one:
        assert args.seed is not None
        run_worker(args.run_one, args.seed, args.device, args.partials_dir)
        return
    if args.aggregate:
        payload = aggregate(args.partials_dir, args.output, fig_prefix=args.fig_prefix)
        print_tables(payload)
        return
    run_matrix(args)


if __name__ == "__main__":
    main()
