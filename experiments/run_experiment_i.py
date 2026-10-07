#!/usr/bin/env python3
"""Experiment I — coordinate-wise decision-margin commit analysis (measurement only).

Question
--------
When the ADD candidate state is already correct but the committed state is wrong, does the
element-wise gate fail to deliver enough movement on coordinates relevant to the decoder's
actual decision boundary, or does the frozen readout require unusually precise latent
alignment even when useful decision-direction movement has been delivered?

Method (no training, no model change, no rollouts, frozen SHA-256-pinned checkpoints)
-------
For every ADD transition of the strict test split (1,200 seq x 10 steps x 5 seeds) and the
five existing controlled ADD builders, this runner re-runs inference on the frozen E and F
checkpoints and records, per transition:

  * the vectors S_(t-1), C_t, S_t and the 212-dim gate f_t (Experiment G stored only scalar
    summaries; Experiment H stored only scalars/digits, so the vectors are captured again);
  * DECISION-SPACE quantities. The frozen readout `decision_head` is a single `nn.Linear`
    (212 -> 4*10), so for a fixed destination register d and value v the logit is exactly
        z_{d,v}(S) = w_{d,v} . S + b_{d,v}.
    We therefore use the pairwise decision margin against the STRONGEST COMPETING CLASS
    chosen at S_(t-1):
        a*      = argmax_{a != y} z_a(S_(t-1))                 (fixed local reference)
        m(S)    = z_y(S) - z_{a*}(S)                           (exact affine in S)
        m_prev / m_cand / m_commit  =  m(S_(t-1)) / m(C_t) / m(S_t)
        dm_cand = m_cand - m_prev , dm_commit = m_commit - m_prev
    The true-logit movement (dw_y . (S - S_prev)) is retained as a SECONDARY diagnostic.
    Sensitivity to recomputing a* at S_t is also recorded (a_star_commit, m_commit_altstar).
  * rho_margin = dm_commit / dm_cand, defined ONLY when dm_cand > EPS_ABS (documented rule);
    otherwise NaN. margin_shortfall = m_cand - m_commit.
  * alpha_star_margin: the minimal alpha >= 0 such that the margin against the FIXED a*
    is non-negative along the local diagnostic line S(alpha) = S_prev + alpha (C - S_prev).
    alpha_star_full: the minimal alpha such that the full argmax equals y (all competitors).
    Both are EXACT because the head is linear (no nonlinearity before the final logits);
    alpha*_margin is an empirical local decision-boundary statistic, NOT a model threshold.
  * COORDINATE DECOMPOSITION (exact for a linear head). With
        q_cand_i = (w_y - w_a*)_i (C_i - S_prev_i)     (candidate-available contribution)
        q_i      = (w_y - w_a*)_i f_i (C_i - S_prev_i) (gate-delivered contribution)
    the identity  sum_i q_i = dm_commit  holds exactly. Stored per transition:
    gate/contribution-weighted gate statistics and the positive/negative decomposition.
  * For the PRIMARY analysis set only (candidate-correct general-stratum test ADDs) the full
    per-coordinate arrays are stored: f_i, move_i = f_i (C_i - S_prev_i), q_cand_i,
    |C_i - S_prev_i|.

This is MEASUREMENT ONLY: frozen checkpoints, no optimizer, no weight modification, all
inference under torch.no_grad(). Checkpoint SHA-256 values are verified against the
Experiment G metadata before and after the run.

Outputs
-------
    experiments/experiment_i_coordinate.npz        scalar table + primary coordinate arrays
    experiments/experiment_i_coordinate_meta.json  columns, thresholds, checkpoint hashes,
                                                   Experiment G/H alignment checks

Reproduce
---------
    python experiments/run_experiment_i.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_i_coordinate.npz
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402  (collector conventions, checkpoint loading)
import run_generator_repair_feasibility as gr  # noqa: E402

# Documented stability threshold for the ratio / boundary statistics. A denominator
# |dm_cand| at or below this value is treated as numerically unstable and rho_margin is
# left undefined (NaN); alpha_star statistics handle the same case explicitly.
EPS_ABS = 1e-3

# Scalar columns recorded for every ADD transition of every source.
COLS = [
    "seed", "source", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b",
    "wrap", "isolated",
    "cand_ok", "comm_ok", "prev_ok", "comm_eq_old",
    "cand_pred", "comm_pred", "prev_pred",
    "y_true", "a_star_prev", "a_star_commit",
    "m_prev", "m_cand", "m_commit", "m_commit_altstar",
    "dm_cand", "dm_commit", "margin_shortfall", "rho_margin", "rho_defined",
    "sign_m_prev_pos", "sign_m_cand_pos", "sign_m_commit_pos",
    "crossed_at_commit", "lost_at_commit",
    "alpha_star_margin", "alpha_star_full", "alpha_full_solution",
    "align_ratio",
    "f_mean", "f_min", "f_max", "f_std",
    "d_cs", "d_ss", "d_sc",
    "q_cand_sum", "q_cand_pos_sum", "q_cand_neg_sum", "q_cand_abs_sum",
    "q_deliv_sum", "q_deliv_pos_sum", "q_deliv_neg_sum",
    "gw_all", "gw_pos", "gw_neg", "gw_mag",
    "delivered_frac_pos", "suppressed_frac_pos",
    "delivered_frac_neg", "suppressed_frac_neg",
    "replay_maxdiff",
]

COORD_KEYS = ["f", "move", "qcand", "abscand"]


def _logit_matrix(W: torch.Tensor, b: torch.Tensor, S: torch.Tensor,
                  dest: torch.Tensor) -> torch.Tensor:
    """z_{dest,v}(S) for every v, given W (R,V,D) and b (R,V). Returns (n, V)."""
    Wd = W[dest]                     # (n, V, D)
    bd = b[dest]                     # (n, V)
    return torch.einsum("nd,nvd->nv", S, Wd) + bd


@torch.no_grad()
def collect_dataset(model: torch.nn.Module, inputs: torch.Tensor, step_targets: torch.Tensor,
                    seed: int, source_id: int) -> Dict[str, Any]:
    """Per-ADD-transition decision-margin records for one (model, seed, dataset)."""
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    step_targets = step_targets.to(device)
    B, T, _ = inputs.shape
    mask, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, dim=1)      # (B, T, D)
    cands = torch.stack(diag.candidates, dim=1)
    gates = torch.stack(diag.gates, dim=1)
    prev_state = torch.cat([torch.zeros(B, 1, gr.STATE_DIM, device=device), states[:, :-1]], dim=1)
    prev_true = torch.zeros_like(step_targets)
    prev_true[:, 1:] = step_targets[:, :-1]

    head = model.decision_head
    R, V, D = gr.NUM_REGISTERS, gr.NUM_VALUES, gr.STATE_DIM
    W = head.proj.weight.detach().view(R, V, D)
    bvec = head.proj.bias.detach().view(R, V)

    idx = (op == gr.RegisterTrackingTask.OP_ADD).nonzero()
    b_i, t_i = idx[:, 0], idx[:, 1]
    n = int(len(idx))
    S_prev = prev_state[b_i, t_i]
    C = cands[b_i, t_i]
    S_t = states[b_i, t_i]
    fvec = gates[b_i, t_i]
    dest = reg_a[b_i, t_i]
    src = reg_b[b_i, t_i]
    oa = prev_true[b_i, t_i, dest]
    ob = prev_true[b_i, t_i, src]
    y_true = step_targets[b_i, t_i, dest]          # true destination new value
    wrap = (oa + ob) >= gr.NUM_VALUES

    ar = torch.arange(n, device=device)

    z_prev = _logit_matrix(W, bvec, S_prev, dest)   # (n, V)
    z_cand = _logit_matrix(W, bvec, C, dest)
    z_comm = _logit_matrix(W, bvec, S_t, dest)

    yv = y_true
    z_prev_y = z_prev[ar, yv]
    z_cand_y = z_cand[ar, yv]
    z_comm_y = z_comm[ar, yv]

    # Pairwise margins z_y - z_a for every competitor a (a == y gives 0).
    Mprev = z_prev_y[:, None] - z_prev           # (n, V)
    Mcand = z_cand_y[:, None] - z_cand
    Mcomm = z_comm_y[:, None] - z_comm

    # Strongest competitor at S_prev (primary reference) == argmin_a!=y Mprev[a].
    Mprev_masked = Mprev.clone()
    Mprev_masked[ar, yv] = float("inf")
    a_star_prev = Mprev_masked.argmin(dim=1)
    m_prev = Mprev.gather(1, a_star_prev[:, None]).squeeze(1)
    m_cand = Mcand.gather(1, a_star_prev[:, None]).squeeze(1)
    m_commit = Mcomm.gather(1, a_star_prev[:, None]).squeeze(1)

    # Sensitivity: strongest competitor recomputed at the committed state.
    Mcomm_masked = Mcomm.clone()
    Mcomm_masked[ar, yv] = float("inf")
    a_star_commit = Mcomm_masked.argmin(dim=1)
    m_commit_altstar = Mcomm.gather(1, a_star_commit[:, None]).squeeze(1)

    dm_cand_all = Mcand - Mprev                   # (n, V)
    dm_cand = dm_cand_all.gather(1, a_star_prev[:, None]).squeeze(1)
    dm_commit = m_commit - m_prev
    margin_shortfall = m_cand - m_commit

    nan_t = torch.full_like(dm_cand, float("nan"))
    rho_defined = dm_cand > EPS_ABS
    rho_margin = torch.where(rho_defined, dm_commit / dm_cand.clamp_min(EPS_ABS), nan_t)

    sign_prev_pos = (m_prev > 0).float()
    sign_cand_pos = (m_cand > 0).float()
    sign_comm_pos = (m_commit > 0).float()
    crossed_at_commit = ((m_prev <= 0) & (m_commit > 0)).float()
    lost_at_commit = ((m_prev > 0) & (m_commit <= 0)).float()

    # alpha*_margin: minimal alpha >= 0 with m_prev + alpha * dm_cand >= 0 (fixed a*).
    alpha_margin = torch.where(
        m_prev >= 0,
        torch.zeros_like(m_prev),
        torch.where(dm_cand > EPS_ABS, -m_prev / dm_cand.clamp_min(EPS_ABS), nan_t),
    )
    # alpha*_full: minimal alpha with the full argmax equal to y (all competitors cleared).
    neg_needed = Mprev < 0
    dma = dm_cand_all
    need = torch.where(neg_needed, (-Mprev) / dma.clamp_min(EPS_ABS), torch.zeros_like(Mprev))
    infeasible = neg_needed & (dma <= EPS_ABS)
    any_infeasible = infeasible.any(dim=1)
    alpha_full = need.max(dim=1).values
    alpha_full = torch.where(m_prev >= 0, torch.zeros_like(alpha_full), alpha_full)
    alpha_full = torch.where(any_infeasible, nan_t, alpha_full)

    # Coordinate decomposition (exact for the linear head).
    w_y = W[dest, yv]                              # (n, D)
    w_star = W[dest, a_star_prev]                  # (n, D)
    dvec = C - S_prev
    qcand = (w_y - w_star) * dvec                  # (n, D)
    qdeliv = qcand * fvec
    move = fvec * dvec

    qcand_pos = qcand.clamp(min=0.0)
    qcand_neg_mag = (-qcand).clamp(min=0.0)
    qdeliv_pos = qdeliv.clamp(min=0.0)
    qdeliv_neg_mag = (-qdeliv).clamp(min=0.0)
    abscand = dvec.abs()

    q_cand_sum = qcand.sum(dim=1)
    q_cand_pos_sum = qcand_pos.sum(dim=1)
    q_cand_neg_sum = qcand_neg_mag.sum(dim=1)
    q_cand_abs_sum = qcand.abs().sum(dim=1)
    q_deliv_sum = qdeliv.sum(dim=1)
    q_deliv_pos_sum = qdeliv_pos.sum(dim=1)
    q_deliv_neg_sum = qdeliv_neg_mag.sum(dim=1)

    epst = EPS_ABS
    gw_all = (qcand.abs() * fvec).sum(dim=1) / q_cand_abs_sum.clamp_min(epst)
    gw_pos = (qcand_pos * fvec).sum(dim=1) / q_cand_pos_sum.clamp_min(epst)
    gw_neg = (qcand_neg_mag * fvec).sum(dim=1) / q_cand_neg_sum.clamp_min(epst)
    gw_mag = (abscand * fvec).sum(dim=1) / abscand.sum(dim=1).clamp_min(epst)
    delivered_frac_pos = q_deliv_pos_sum / q_cand_pos_sum.clamp_min(epst)
    suppressed_frac_pos = 1.0 - delivered_frac_pos
    delivered_frac_neg = q_deliv_neg_sum / q_cand_neg_sum.clamp_min(epst)
    suppressed_frac_neg = 1.0 - delivered_frac_neg
    # Undefined weighted statistics where there is no contribution on that side.
    gw_pos = torch.where(q_cand_pos_sum > epst, gw_pos, nan_t)
    gw_neg = torch.where(q_cand_neg_sum > epst, gw_neg, nan_t)
    delivered_frac_pos = torch.where(q_cand_pos_sum > epst, delivered_frac_pos, nan_t)
    suppressed_frac_pos = torch.where(q_cand_pos_sum > epst, suppressed_frac_pos, nan_t)
    delivered_frac_neg = torch.where(q_cand_neg_sum > epst, delivered_frac_neg, nan_t)
    suppressed_frac_neg = torch.where(q_cand_neg_sum > epst, suppressed_frac_neg, nan_t)

    cand_pred = z_cand.argmax(dim=1)
    comm_pred = z_comm.argmax(dim=1)
    prev_pred = z_prev.argmax(dim=1)

    d_cs = (C - S_prev).norm(dim=1)
    d_ss = (S_t - S_prev).norm(dim=1)
    d_sc = (S_t - C).norm(dim=1)
    align_ratio = dm_cand / d_cs.clamp_min(1e-12)

    replay = (1.0 - fvec) * S_prev + fvec * C
    replay_maxdiff = (replay - S_t).abs().max(dim=1).values

    def to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    cols = {
        "seed": np.full(n, seed, np.float64), "source": np.full(n, source_id, np.float64),
        "seq_id": to_np(b_i).astype(np.float64), "t": to_np(t_i).astype(np.float64),
        "dest": to_np(dest).astype(np.float64), "src_reg": to_np(src).astype(np.float64),
        "operand_a": to_np(oa).astype(np.float64), "operand_b": to_np(ob).astype(np.float64),
        "wrap": to_np(wrap.float()), "isolated": to_np(
            ((t_i == 0) | (op[b_i, (t_i - 1).clamp_min(0)] != gr.RegisterTrackingTask.OP_ADD)).float()),
        "cand_ok": to_np((cand_pred == yv).float()), "comm_ok": to_np((comm_pred == yv).float()),
        "prev_ok": to_np((prev_pred == oa).float()), "comm_eq_old": to_np((comm_pred == oa).float()),
        "cand_pred": to_np(cand_pred).astype(np.float64), "comm_pred": to_np(comm_pred).astype(np.float64),
        "prev_pred": to_np(prev_pred).astype(np.float64),
        "y_true": to_np(yv).astype(np.float64),
        "a_star_prev": to_np(a_star_prev).astype(np.float64),
        "a_star_commit": to_np(a_star_commit).astype(np.float64),
        "m_prev": to_np(m_prev), "m_cand": to_np(m_cand), "m_commit": to_np(m_commit),
        "m_commit_altstar": to_np(m_commit_altstar),
        "dm_cand": to_np(dm_cand), "dm_commit": to_np(dm_commit),
        "margin_shortfall": to_np(margin_shortfall), "rho_margin": to_np(rho_margin),
        "rho_defined": to_np(rho_defined.float()),
        "sign_m_prev_pos": to_np(sign_prev_pos), "sign_m_cand_pos": to_np(sign_cand_pos),
        "sign_m_commit_pos": to_np(sign_comm_pos),
        "crossed_at_commit": to_np(crossed_at_commit), "lost_at_commit": to_np(lost_at_commit),
        "alpha_star_margin": to_np(alpha_margin), "alpha_star_full": to_np(alpha_full),
        "alpha_full_solution": to_np((~any_infeasible).float()),
        "align_ratio": to_np(align_ratio),
        "f_mean": to_np(fvec.mean(dim=1)), "f_min": to_np(fvec.min(dim=1).values),
        "f_max": to_np(fvec.max(dim=1).values), "f_std": to_np(fvec.std(dim=1)),
        "d_cs": to_np(d_cs), "d_ss": to_np(d_ss), "d_sc": to_np(d_sc),
        "q_cand_sum": to_np(q_cand_sum), "q_cand_pos_sum": to_np(q_cand_pos_sum),
        "q_cand_neg_sum": to_np(q_cand_neg_sum), "q_cand_abs_sum": to_np(q_cand_abs_sum),
        "q_deliv_sum": to_np(q_deliv_sum), "q_deliv_pos_sum": to_np(q_deliv_pos_sum),
        "q_deliv_neg_sum": to_np(q_deliv_neg_sum),
        "gw_all": to_np(gw_all), "gw_pos": to_np(gw_pos), "gw_neg": to_np(gw_neg),
        "gw_mag": to_np(gw_mag),
        "delivered_frac_pos": to_np(delivered_frac_pos),
        "suppressed_frac_pos": to_np(suppressed_frac_pos),
        "delivered_frac_neg": to_np(delivered_frac_neg),
        "suppressed_frac_neg": to_np(suppressed_frac_neg),
        "replay_maxdiff": to_np(replay_maxdiff),
    }
    scalars = np.stack([cols[c] for c in COLS], axis=1).astype(np.float64)

    out: Dict[str, Any] = {"scalars": scalars, "primary_local": None, "coords": None}
    if source_id == eg.SOURCE_IDS["test"]:
        is_primary = (dest != src) & (ob > 0) & (cand_pred == yv)
        pl = torch.nonzero(is_primary, as_tuple=False).squeeze(1)
        if pl.numel() > 0:
            out["primary_local"] = to_np(pl).astype(np.int64)
            out["coords"] = {
                "f": to_np(fvec[pl]).astype(np.float32),
                "move": to_np(move[pl]).astype(np.float32),
                "qcand": to_np(qcand[pl]).astype(np.float32),
                "abscand": to_np(abscand[pl]).astype(np.float32),
            }
    return out


def _concat(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalars = np.concatenate([p["scalars"] for p in parts], axis=0)
    rows: List[np.ndarray] = []
    coords: Dict[str, List[np.ndarray]] = {k: [] for k in COORD_KEYS}
    offset = 0
    for p in parts:
        n = p["scalars"].shape[0]
        if p["primary_local"] is not None and len(p["primary_local"]):
            rows.append(p["primary_local"] + offset)
            for k in COORD_KEYS:
                coords[k].append(p["coords"][k])
        offset += n
    primary_row = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    coord_out = {k: (np.concatenate(v, axis=0) if v else np.zeros((0, gr.STATE_DIM), np.float32))
                 for k, v in coords.items()}
    return {"scalars": scalars, "primary_row": primary_row, "coords": coord_out}


def verify_against_experiment_g(h: Dict[str, Any], g_arr: np.ndarray,
                                g_cols: List[str], model_key: str) -> Dict[str, Any]:
    """Row-level verification that I's re-collected transitions match Experiment G's dataset."""
    gi = {c: i for i, c in enumerate(g_cols)}
    n = h["scalars"].shape[0]
    ci = {c: i for i, c in enumerate(COLS)}
    checks: Dict[str, Any] = {"model": model_key, "n_rows_i": int(n), "n_rows_g": int(g_arr.shape[0])}
    if n != g_arr.shape[0]:
        raise RuntimeError(f"{model_key}: row count mismatch I={n} G={g_arr.shape[0]}")
    exact_cols = ["seed", "source", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b",
                  "wrap", "isolated", "cand_ok", "comm_ok", "prev_ok", "comm_eq_old",
                  "cand_pred", "comm_pred", "prev_pred"]
    mismatches = {}
    for col in exact_cols:
        a = np.round(h["scalars"][:, ci[col]]).astype(np.int64)
        b = np.round(g_arr[:, gi[col]]).astype(np.int64)
        nb = int((a != b).sum())
        if nb:
            mismatches[col] = nb
    checks["exact_mismatches"] = mismatches
    checks["passed"] = len(mismatches) == 0
    if not checks["passed"]:
        raise RuntimeError(f"{model_key}: I/G verification failed: {mismatches}")

    rec = {
        "replay_maxdiff_max": float(np.nanmax(h["scalars"][:, ci["replay_maxdiff"]])),
        "replay_identity_ok": bool(np.nanmax(h["scalars"][:, ci["replay_maxdiff"]]) < 1e-4),
        "q_deliv_equals_dm_commit_maxdiff": float(np.nanmax(np.abs(
            h["scalars"][:, ci["q_deliv_sum"]] - h["scalars"][:, ci["dm_commit"]]))),
    }
    checks["algebra"] = rec
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment I collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--g-transitions", default="experiments/experiment_g_transitions.npz")
    ap.add_argument("--output", default="experiments/experiment_i_coordinate.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_i_coordinate_meta.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    gz = np.load(args.g_transitions, allow_pickle=False)
    g_meta = json.loads(str(gz["meta_json"][0]))
    builders = gr._build_controlled_add_sequences()
    meta: Dict[str, Any] = {
        "cols": COLS, "coord_keys": COORD_KEYS, "eps_abs": EPS_ABS,
        "source_ids": eg.SOURCE_IDS, "op_add": int(gr.RegisterTrackingTask.OP_ADD),
        "decision_head": "TypedDecisionHead = single nn.Linear(212, 4*10); logits affine in S",
        "models": {}, "g_verification": {},
        "notes": (
            "Measurement only: frozen checkpoints, no training, no optimizer, no weight "
            "modification. Primary competitor a* is chosen at S_(t-1) (fixed local reference); "
            "m_commit_altstar uses a* recomputed at S_t as a sensitivity check. rho_margin "
            f"defined only when dm_cand > {EPS_ABS}. alpha* values are exact for the linear "
            "readout and are empirical local decision-boundary statistics, not model thresholds."),
    }
    per_model: Dict[str, Any] = {}
    for model_key in eg.MODELS:
        g_arr = gz[model_key]
        parts: List[Dict[str, Any]] = []
        meta["models"][model_key] = {"checkpoints": {}}
        for seed in gr.SEEDS:
            loaded = eg.load_checkpoint(args.checkpoints_dir, model_key, seed)
            sha_before = loaded["sha256"]
            g_sha = g_meta["models"][model_key]["checkpoints"][str(seed)]["sha256"]
            if sha_before != g_sha:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint hash changed vs Experiment G")
            model = loaded["model"]
            splits = gr.canonical_splits(seed)
            ds = splits.test_set
            batch = ds.get_batch(len(ds), idx=0)
            parts.append(collect_dataset(model, batch.inputs, batch.metadata["step_targets"],
                                         seed, eg.SOURCE_IDS["test"]))
            for name, d in builders.items():
                parts.append(collect_dataset(model, torch.from_numpy(d["inputs"]),
                                             torch.from_numpy(d["step_targets"]),
                                             seed, eg.SOURCE_IDS[name]))
            sha_after = eg.sha256_file(loaded["path"])
            if sha_after != sha_before:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint file changed during run")
            meta["models"][model_key]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {model_key} seed={seed}: cumulative rows={sum(len(p['scalars']) for p in parts)}")
        h = _concat(parts)
        meta["g_verification"][model_key] = verify_against_experiment_g(h, g_arr, g_meta["columns"], model_key)
        per_model[model_key] = h
        print(f"  {model_key}: {h['scalars'].shape[0]} transitions, "
              f"{h['primary_row'].shape[0]} primary coordinate rows; "
              f"G verification passed={meta['g_verification'][model_key]['passed']}")

    meta["n_rows"] = {k: int(v["scalars"].shape[0]) for k, v in per_model.items()}
    meta["n_primary_rows"] = {k: int(v["primary_row"].shape[0]) for k, v in per_model.items()}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload = {"meta_json": np.array([json.dumps(meta)])}
    for model_key, h in per_model.items():
        payload[f"{model_key}__scalars"] = h["scalars"].astype(np.float32)
        payload[f"{model_key}__primary_row"] = h["primary_row"].astype(np.int32)
        for k in COORD_KEYS:
            payload[f"{model_key}__primary_{k}"] = h["coords"][k]
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
