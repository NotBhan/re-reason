#!/usr/bin/env python3
"""Experiment N — Parity-Origin Localization (measurement only).

Experiment M localized the off-path "third basin" to a systematic **same-parity** signature:
the wrong destination value preserves the target's parity in 73.7% (E) / 64.0% (F) vs a 50%
permutation null. N asks WHERE that parity structure lives:

  N-A readout value-geometry : the frozen 40-way head itself has parity-organized geometry;
  N-B state-displacement     : the committed state/label alignment produces the effect;
  N-C joint;  N-D inconclusive.

Central design principle (avoid circularity): separate structure in W itself from
state-conditioned effects. The primary parity axis is derived from W ALONE (parity centroids of
the normalized value weight vectors), never from the wrong-value labels. Pairing-break controls
(readout-only label permutation and state/decode permutation) probe whether the effect depends on
the specific state<->label alignment.

Measurement only: frozen checkpoints, no training, no optimizer, no weight/readout/gate/commit
modification, no intervention, no recursive counterfactual rollout, no learned classifier as
primary evidence. All inference under torch.no_grad(); checkpoint SHA-256 verified before/after
and against the Experiment G/L/M metadata.

This collector reuses Experiment M/L for W, b, the primary population and the scalars, and
re-collects ONLY the minimum extra forward pass needed to capture all-primary state vectors (M
stored raw vectors for the failure groups only). Exact alignment against M is required:
max |Delta| = 0 on the shared identifying/scalar columns and on the group-0/2 raw vectors.

Outputs
-------
    experiments/experiment_n_parity_origin.npz        per-seed W,b + primary scalars + failure vectors
    experiments/experiment_n_parity_origin_meta.json  columns, checkpoint hashes, M/L alignment

Reproduce
---------
    python experiments/run_experiment_n.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_n_parity_origin.npz
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402
import run_experiment_l as rl  # noqa: E402  (readout matrix, EPS)
import run_experiment_m as rm  # noqa: E402  (population conventions, COL reuse)
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

MODULUS = gr.NUM_VALUES
EPS_DOT = 1e-6
PRIMARY_MODELS = ["E_candidate", "F_combined"]

# Minimal per-primary-row scalar table N needs (aligned to M's primary_row order).
N_COLS = [
    "seed", "dest", "src_reg", "operand_a", "operand_b", "y_true", "wrap",
    "cand_pred", "comm_pred",
    "dec_commit", "dec_prev", "dec_cand",
    "par_proj_prev", "par_proj_cand", "par_proj_commit",
    "par_delta_commit", "par_delta_cand", "norm_delta_commit", "norm_delta_cand",
    "par_ratio_commit", "par_ratio_cand",
]
FAIL_VEC_KEYS = ["S_prev", "C", "S_commit"]


def parity_axes(W: np.ndarray) -> np.ndarray:
    """Per-seed, per-register parity direction p = normalize(mu_even - mu_odd) from W alone.

    W: (seeds, R, V, D) -> P: (seeds, R, D). Value parity is index parity (v % 2).
    """
    S, R, V, D = W.shape
    P = np.zeros((S, R, D), np.float64)
    for s in range(S):
        for d in range(R):
            X = W[s, d].astype(np.float64)
            Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
            even = np.arange(V) % 2 == 0
            diff = Xn[even].mean(0) - Xn[~even].mean(0)
            n = np.linalg.norm(diff)
            P[s, d] = diff / n if n > 1e-9 else 0.0
    return P


@torch.no_grad()
def collect_primary_states(model: torch.nn.Module, inputs: torch.Tensor,
                           step_targets: torch.Tensor, seed: int) -> Dict[str, Any]:
    """All-primary state vectors + parity-axis projections for one (model, seed, test split)."""
    device = next(model.parameters()).device
    inputs = inputs.to(device); step_targets = step_targets.to(device)
    B, T, _ = inputs.shape
    _, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, dim=1)
    cands = torch.stack(diag.candidates, dim=1)
    prev_state = torch.cat([torch.zeros(B, 1, gr.STATE_DIM, device=device), states[:, :-1]], dim=1)
    prev_true = torch.zeros_like(step_targets); prev_true[:, 1:] = step_targets[:, :-1]

    W, b = rl._matrix(model.decision_head)          # (R,V,D), (R,V)
    idx = (op == RegisterTrackingTask.OP_ADD).nonzero()
    b_i, t_i = idx[:, 0], idx[:, 1]
    n = int(len(idx))
    S_prev = prev_state[b_i, t_i]; C = cands[b_i, t_i]; S_t = states[b_i, t_i]
    dest = reg_a[b_i, t_i]; src = reg_b[b_i, t_i]
    oa = prev_true[b_i, t_i, dest]; ob = prev_true[b_i, t_i, src]
    y = step_targets[b_i, t_i, dest]
    wrap = (oa + ob) >= MODULUS
    dev = S_prev.device

    def decode(S: torch.Tensor) -> torch.Tensor:
        Wd = W[dest]; bd = b[dest]
        return (torch.einsum("nd,nvd->nv", S, Wd) + bd).argmax(-1)

    dec_commit, dec_prev, dec_cand = decode(S_t), decode(S_prev), decode(C)

    # parity axis for this seed (per register)
    Wnp = W.detach().cpu().numpy()[None]                       # (1,R,V,D)
    P = parity_axes(Wnp)[0]                                    # (R,D)
    p_ax = torch.from_numpy(P).to(dev)[dest]                   # (n,D) per-transition axis
    par_proj_prev = (S_prev * p_ax).sum(1)
    par_proj_cand = (C * p_ax).sum(1)
    par_proj_commit = (S_t * p_ax).sum(1)
    dcom = S_t - S_prev; dcan = C - S_prev
    par_delta_commit = (dcom * p_ax).sum(1)
    par_delta_cand = (dcan * p_ax).sum(1)
    norm_delta_commit = dcom.norm(dim=1)
    norm_delta_cand = dcan.norm(dim=1)
    par_ratio_commit = par_delta_commit.abs() / norm_delta_commit.clamp_min(EPS_DOT)
    par_ratio_cand = par_delta_cand.abs() / norm_delta_cand.clamp_min(EPS_DOT)

    # group masks (same definition as M/L)
    gen = (dest != src) & (ob > 0)
    cand_ok = dec_cand == y
    comm_ok = dec_commit == y
    z_prev = torch.einsum("nd,nvd->nv", S_prev, W[dest]) + b[dest]
    z_cand = torch.einsum("nd,nvd->nv", C, W[dest]) + b[dest]
    z_comm = torch.einsum("nd,nvd->nv", S_t, W[dest]) + b[dest]
    ar = torch.arange(n, device=dev)
    m_prev_all = z_prev[ar, y][:, None] - z_prev
    m_cand_all = z_cand[ar, y][:, None] - z_cand
    m_comm_all = z_comm[ar, y][:, None] - z_comm
    Mm = m_prev_all.clone(); Mm[ar, y] = float("inf")
    a_prev = Mm.argmin(1)
    m_prev = m_prev_all.gather(1, a_prev[:, None]).squeeze(1)
    m_cand = m_cand_all.gather(1, a_prev[:, None]).squeeze(1)
    m_commit = m_comm_all.gather(1, a_prev[:, None]).squeeze(1)
    dm_cand_all = m_cand_all - m_prev_all
    dm_cand = dm_cand_all.gather(1, a_prev[:, None]).squeeze(1)
    dm_commit = m_commit - m_prev
    neg = m_prev_all < 0
    need = torch.where(neg, (-m_prev_all) / dm_cand_all.clamp_min(rl.EPS_ABS), torch.zeros_like(m_prev_all))
    infeasible = neg & (dm_cand_all <= rl.EPS_ABS)
    alpha_full = need.max(1).values
    alpha_full = torch.where(m_prev >= 0, torch.zeros_like(alpha_full), alpha_full)
    alpha_full = torch.where(infeasible.any(1), torch.full_like(alpha_full, float("nan")), alpha_full)
    rho_defined = dm_cand > rl.EPS_ABS
    rho = torch.where(rho_defined, dm_commit / dm_cand.clamp_min(rl.EPS_ABS),
                      torch.full_like(dm_cand, float("nan")))
    finite = torch.isfinite(alpha_full) & torch.isfinite(rho)
    offpath = gen & cand_ok & (~comm_ok) & finite & (rho >= alpha_full)
    succ = gen & cand_ok & comm_ok
    nonoff = gen & cand_ok & (~comm_ok) & ~(finite & (rho >= alpha_full))
    group = torch.full((n,), -1, dtype=torch.long, device=dev)
    group[offpath] = 0; group[succ] = 1; group[nonoff] = 2

    to = lambda x: x.detach().cpu().numpy()
    cols = {
        "seed": np.full(n, seed, np.float64), "dest": to(dest).astype(np.float64),
        "src_reg": to(src).astype(np.float64), "operand_a": to(oa).astype(np.float64),
        "operand_b": to(ob).astype(np.float64), "y_true": to(y).astype(np.float64),
        "wrap": to(wrap.float()), "cand_pred": to(dec_cand).astype(np.float64),
        "comm_pred": to(dec_commit).astype(np.float64),
        "dec_commit": to(dec_commit).astype(np.float64), "dec_prev": to(dec_prev).astype(np.float64),
        "dec_cand": to(dec_cand).astype(np.float64),
        "par_proj_prev": to(par_proj_prev), "par_proj_cand": to(par_proj_cand),
        "par_proj_commit": to(par_proj_commit), "par_delta_commit": to(par_delta_commit),
        "par_delta_cand": to(par_delta_cand), "norm_delta_commit": to(norm_delta_commit),
        "norm_delta_cand": to(norm_delta_cand), "par_ratio_commit": to(par_ratio_commit),
        "par_ratio_cand": to(par_ratio_cand),
    }
    scalars = np.stack([cols[c] for c in N_COLS], axis=1).astype(np.float64)
    prow = torch.nonzero(group >= 0, as_tuple=False).squeeze(1)
    fail_mask = (group == 0) | (group == 2)
    return {
        "scalars": scalars, "group": to(group).astype(np.int64),
        "primary_row": to(prow).astype(np.int64),
        "primary_group": to(group[prow]).astype(np.int64),
        "W": to(W.float()), "b": to(b.float()),
        "fail_row": to(torch.nonzero(fail_mask, as_tuple=False).squeeze(1)).astype(np.int64),
        "fail_S_prev": to(S_prev[fail_mask].float()), "fail_C": to(C[fail_mask].float()),
        "fail_S_commit": to(S_t[fail_mask].float()),
    }


def concat(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    off = 0
    scal, grp, prow, pgrp, frow, fS, fC, fSc, Ws, bs = [], [], [], [], [], [], [], [], [], []
    for p in parts:
        n = p["scalars"].shape[0]
        scal.append(p["scalars"])
        grp.append(p["group"])
        prow.append(p["primary_row"] + off); pgrp.append(p["primary_group"])
        frow.append(p["fail_row"] + off)
        fS.append(p["fail_S_prev"]); fC.append(p["fail_C"]); fSc.append(p["fail_S_commit"])
        Ws.append(p["W"]); bs.append(p["b"])
        off += n
    return {
        "scalars": np.concatenate(scal), "group": np.concatenate(grp),
        "primary_row": np.concatenate(prow), "primary_group": np.concatenate(pgrp),
        "fail_row": np.concatenate(frow), "fail_S_prev": np.concatenate(fS, 0),
        "fail_C": np.concatenate(fC, 0), "fail_S_commit": np.concatenate(fSc, 0),
        "W": np.stack(Ws).astype(np.float32), "b": np.stack(bs).astype(np.float32),
    }


def verify_against_m(n: Dict[str, Any], mz, model_key: str) -> Dict[str, Any]:
    """Exact alignment: primary rows/groups identical, group-0/2 raw vectors identical, counts match."""
    m_row = mz[f"{model_key}__primary_row"].astype(np.int64)
    m_group = mz[f"{model_key}__primary_group"].astype(np.int64)
    checks: Dict[str, Any] = {"model": model_key}
    if not np.array_equal(n["primary_row"], m_row):
        raise RuntimeError(f"{model_key}: N/M primary_row mismatch")
    if not np.array_equal(n["primary_group"], m_group):
        raise RuntimeError(f"{model_key}: N/M primary_group mismatch")
    m_fail_row = mz[f"{model_key}__fail_row"].astype(np.int64)
    if not np.array_equal(n["fail_row"], m_fail_row):
        raise RuntimeError(f"{model_key}: N/M fail_row mismatch")
    maxd = {}
    for k in FAIL_VEC_KEYS:
        a = n[f"fail_{k}"]; b = mz[f"{model_key}__fail_{k}"].astype(np.float64)
        maxd[k] = float(np.nanmax(np.abs(a.astype(np.float64) - b)))
    checks["fail_vector_max_abs_diff"] = maxd
    checks["n_primary"] = int(n["primary_row"].shape[0])
    checks["n_offpath"] = int((n["primary_group"] == 0).sum())
    checks["passed"] = all(v == 0.0 for v in maxd.values())
    if not checks["passed"]:
        raise RuntimeError(f"{model_key}: N/M fail-vector alignment failed: {maxd}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment N collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--m-geometry", default="experiments/experiment_m_geometry.npz")
    ap.add_argument("--output", default="experiments/experiment_n_parity_origin.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_n_parity_origin_meta.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    mz = np.load(args.m_geometry, allow_pickle=False)
    m_meta = json.loads(str(mz["meta_json"][0]))
    meta: Dict[str, Any] = {
        "n_cols": N_COLS, "seeds": gr.SEEDS, "models": {}, "m_alignment": {},
        "offpath_definition": m_meta["offpath_definition"],
        "parity_axis": "per-seed, per-register p = normalize(mean normalized W[even v] - mean normalized W[odd v]); derived from W alone",
        "note": ("Measurement only: frozen checkpoints, no training, no optimizer, no weight/"
                 "readout/gate/commit modification, no intervention. Pairing-break permutation "
                 "controls are analysis-only counterfactuals on frozen objects."),
    }
    per_model: Dict[str, Any] = {}
    for model_key in PRIMARY_MODELS:
        parts: List[Dict[str, Any]] = []
        meta["models"][model_key] = {"checkpoints": {}}
        for seed in gr.SEEDS:
            loaded = eg.load_checkpoint(args.checkpoints_dir, model_key, seed)
            sha_before = loaded["sha256"]
            m_sha = m_meta["models"][model_key]["checkpoints"][str(seed)]["sha256"]
            if sha_before != m_sha:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint hash changed vs Experiment M")
            batch = gr.canonical_splits(seed).test_set.get_batch(gr.TEST_SIZE, idx=0)
            parts.append(collect_primary_states(loaded["model"], batch.inputs,
                                                batch.metadata["step_targets"], seed))
            if eg.sha256_file(loaded["path"]) != sha_before:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint changed during run")
            meta["models"][model_key]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {model_key} seed={seed}: cumulative primary={sum(len(p['primary_row']) for p in parts)}")
        n = concat(parts)
        meta["m_alignment"][model_key] = verify_against_m(n, mz, model_key)
        per_model[model_key] = n
        print(f"  {model_key}: primary={n['primary_row'].shape[0]} off-path={(n['primary_group']==0).sum()} "
              f"M alignment={meta['m_alignment'][model_key]['passed']}")

    meta["n_primary"] = {k: int(v["primary_row"].shape[0]) for k, v in per_model.items()}
    meta["n_offpath"] = {k: int((v["primary_group"] == 0).sum()) for k, v in per_model.items()}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload: Dict[str, np.ndarray] = {"meta_json": np.array([json.dumps(meta)])}
    for model_key, n in per_model.items():
        payload[f"{model_key}__scalars"] = n["scalars"].astype(np.float32)
        payload[f"{model_key}__primary_row"] = n["primary_row"]
        payload[f"{model_key}__primary_group"] = n["primary_group"]
        payload[f"{model_key}__W"] = n["W"]; payload[f"{model_key}__b"] = n["b"]
        payload[f"{model_key}__fail_row"] = n["fail_row"]
        for k in FAIL_VEC_KEYS:
            payload[f"{model_key}__fail_{k}"] = n[f"fail_{k}"]
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
