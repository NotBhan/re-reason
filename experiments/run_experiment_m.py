#!/usr/bin/env python3
"""Experiment M — Representation Geometry of the Third Basin (measurement only).

Purpose
-------
Experiment L localized the strictly-defined off-path residual (candidate correct, committed
wrong, rho_margin >= alpha*_full) to latent-state *placement*: the committed state clears the
competitor that was binding at S_prev (100%), yet the committed argmax becomes a different
competitor and the decoded destination value is almost always a third value; the fixed readout
bias is negligible.

M asks WHAT the third basin is, geometrically:
  * is the wrong destination value systematically related to modular addition / wrap-around,
    or an adjacent modular offset; or
  * is the wrong-region identity and the off-axis displacement unsystematic (a collapse)?

This collector captures, for every ADD transition of the canonical strict test split, the exact
frozen-readout quantities M needs: the flat 40-way committed winner (register, value), the
arithmetic mapping columns, the candidate/commit off-axis decomposition, and the true-vs-winner
margins. It stores the raw S_prev/C/S_commit vectors for the small failure groups (off-path and
non-off-path) and the unit off-axis direction for every primary transition (so the directional
consistency and permutation null can be computed without storing three vectors per row).

Measurement only: frozen checkpoints, no training, no optimizer, no weight/readout/gate/commit
modification, no intervention, no recursive counterfactual trajectories. All inference under
torch.no_grad(). Checkpoint SHA-256 verified before/after and against Experiment G metadata.

Primary population (EXACTLY Experiment L's, not redefined after seeing results)
-------------------------------------------------------------------------------
strict test split | ADD | general (dest != src AND operand_b > 0) | candidate correct |
baseline commit wrong | rho_margin >= alpha_star_full.  E primary; F secondary control.

Outputs
-------
    experiments/experiment_m_geometry.npz        scalars + primary unit off-axis + failure vectors
    experiments/experiment_m_geometry_meta.json  columns, checkpoint hashes, L/G alignment

Reproduce
---------
    python experiments/run_experiment_m.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_m_geometry.npz
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
import run_experiment_i as ri  # noqa: E402
import run_experiment_l as rl  # noqa: E402  (readout matrix + conventions)
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

EPS_ABS = ri.EPS_ABS
EPS_DOT = 1e-6
MODULUS = gr.NUM_VALUES  # 10

COLS = [
    "seed", "source", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b",
    "wrap", "isolated",
    "cand_ok", "comm_ok", "prev_ok", "comm_eq_old",
    "cand_pred", "comm_pred", "prev_pred",
    "y_true", "a_star_prev", "a_star_commit", "competitor_changed",
    "m_prev", "m_cand", "m_commit", "m_commit_alt",
    "dm_cand", "dm_commit", "rho_margin", "rho_defined", "alpha_star_full",
    "f_mean", "d_cs", "d_ss", "d_sc", "lambda_eff", "offpath_dist",
    "bias_av", "state_av_commit", "bias_ac", "state_ac_commit",
    "norm_dw_av", "norm_dw_ac", "hyper_av_commit", "hyper_ac_commit",
    # flat 40-way committed winner (register, value) at S_prev / C / S_commit
    "flat_reg_prev", "flat_val_prev", "flat_reg_cand", "flat_val_cand",
    "flat_reg_commit", "flat_val_commit",
    # true-vs-flat-winner margins at S_prev / C / S_commit and normalized distance
    "mv_prev", "mv_cand", "mv_commit", "hyper_flat_commit", "norm_dw_flat",
    # off-axis decomposition of Delta_commit vs Delta_candidate
    "off_par_norm", "off_orth_norm", "off_cos", "off_ratio",
    # arithmetic mapping of the wrong destination value
    "wrong_offset", "dmod_wrong_true", "dmod_wrong_old", "dmod_wrong_operand",
]


def _mod_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """d_mod(a,b) = min((a-b) mod 10, (b-a) mod 10)."""
    d = np.abs(a - b) % MODULUS
    return np.minimum(d, MODULUS - d)


@torch.no_grad()
def collect_dataset(model: torch.nn.Module, inputs: torch.Tensor, step_targets: torch.Tensor,
                    seed: int, source_id: int) -> Dict[str, Any]:
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    step_targets = step_targets.to(device)
    B, T, _ = inputs.shape
    _, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, dim=1)
    cands = torch.stack(diag.candidates, dim=1)
    gates = torch.stack(diag.gates, dim=1)
    prev_state = torch.cat([torch.zeros(B, 1, gr.STATE_DIM, device=device), states[:, :-1]], dim=1)
    prev_true = torch.zeros_like(step_targets)
    prev_true[:, 1:] = step_targets[:, :-1]

    head = model.decision_head
    R, V, D = gr.NUM_REGISTERS, gr.NUM_VALUES, gr.STATE_DIM
    W, bvec = rl._matrix(head)                       # (R,V,D), (R,V)
    Wflat = head.proj.weight.detach()                # (R*V, D)
    bflat = head.proj.bias.detach()                  # (R*V,)

    idx = (op == RegisterTrackingTask.OP_ADD).nonzero()
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
    y = step_targets[b_i, t_i, dest]
    wrap = (oa + ob) >= MODULUS
    ar = torch.arange(n, device=device)

    def zmat(S: torch.Tensor) -> torch.Tensor:
        Wd = W[dest]
        bd = bvec[dest]
        return torch.einsum("nd,nvd->nv", S, Wd) + bd

    def flat_logits(S: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,md->nm", S, Wflat) + bflat      # (n, R*V)

    z_prev, z_cand, z_comm = zmat(S_prev), zmat(C), zmat(S_t)
    fl_prev, fl_cand, fl_comm = flat_logits(S_prev), flat_logits(C), flat_logits(S_t)
    flat_prev = fl_prev.argmax(1); flat_cand = fl_cand.argmax(1); flat_comm = fl_comm.argmax(1)

    # strongest competitor (destination register) at S_prev / commit
    Mp = z_prev[ar, y][:, None] - z_prev
    Mm = Mp.clone(); Mm[ar, y] = float("inf")
    a_prev = Mm.argmin(dim=1)
    Mc = z_comm[ar, y][:, None] - z_comm
    Mcm = Mc.clone(); Mcm[ar, y] = float("inf")
    a_commit = Mcm.argmin(dim=1)
    m_prev = Mp.gather(1, a_prev[:, None]).squeeze(1)
    m_cand = (z_cand[ar, y][:, None] - z_cand).gather(1, a_prev[:, None]).squeeze(1)
    m_commit = Mc.gather(1, a_prev[:, None]).squeeze(1)
    m_commit_alt = Mc.gather(1, a_commit[:, None]).squeeze(1)
    dm_cand_all = (z_cand[ar, y][:, None] - z_cand) - Mp
    dm_cand = dm_cand_all.gather(1, a_prev[:, None]).squeeze(1)
    dm_commit = m_commit - m_prev
    nan_t = torch.full_like(dm_cand, float("nan"))
    rho_defined = dm_cand > EPS_ABS
    rho = torch.where(rho_defined, dm_commit / dm_cand.clamp_min(EPS_ABS), nan_t)
    neg = Mp < 0
    need = torch.where(neg, (-Mp) / dm_cand_all.clamp_min(EPS_ABS), torch.zeros_like(Mp))
    infeasible = neg & (dm_cand_all <= EPS_ABS)
    alpha_full = need.max(dim=1).values
    alpha_full = torch.where(m_prev >= 0, torch.zeros_like(alpha_full), alpha_full)
    alpha_full = torch.where(infeasible.any(dim=1), nan_t, alpha_full)

    w_y = W[dest, y]
    w_av = W[dest, a_prev]; w_ac = W[dest, a_commit]
    bias_av = bvec[dest, y] - bvec[dest, a_prev]
    bias_ac = bvec[dest, y] - bvec[dest, a_commit]
    state_av_commit = torch.einsum("nd,nd->n", w_y - w_av, S_t)
    state_ac_commit = torch.einsum("nd,nd->n", w_y - w_ac, S_t)
    norm_dw_av = (w_y - w_av).norm(dim=1).clamp_min(EPS_DOT)
    norm_dw_ac = (w_y - w_ac).norm(dim=1).clamp_min(EPS_DOT)
    hyper_av_commit = m_commit / norm_dw_av
    hyper_ac_commit = m_commit_alt / norm_dw_ac

    # true-vs-flat-winner margins (flat 40-way committed class)
    def true_logit(S: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,nd->n", S, w_y) + bvec[dest, y]
    def flat_winner_logit(S: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        wf = Wflat[flat]
        return torch.einsum("nd,nd->n", S, wf) + bflat[flat]
    mv_prev = true_logit(S_prev) - flat_winner_logit(S_prev, flat_prev)
    mv_cand = true_logit(C) - flat_winner_logit(C, flat_cand)
    mv_commit = true_logit(S_t) - flat_winner_logit(S_t, flat_comm)
    w_flat_commit = Wflat[flat_comm]
    dw_flat = w_y - w_flat_commit
    norm_dw_flat = dw_flat.norm(dim=1).clamp_min(EPS_DOT)
    hyper_flat_commit = mv_commit / norm_dw_flat

    # distances + off-axis decomposition
    d_cs = (C - S_prev).norm(dim=1)
    d_ss = (S_t - S_prev).norm(dim=1)
    d_sc = (S_t - C).norm(dim=1)
    dcand = C - S_prev
    dcommit = S_t - S_prev
    denom = (dcand * dcand).sum(dim=1).clamp_min(1e-12)
    lam = ((dcommit * dcand).sum(dim=1) / denom)
    par = lam[:, None] * dcand
    orth = dcommit - par
    off_par_norm = par.norm(dim=1)
    off_orth_norm = orth.norm(dim=1)
    off_cos = (dcommit * dcand).sum(dim=1) / (dcommit.norm(dim=1).clamp_min(EPS_DOT) * d_cs.clamp_min(EPS_DOT))
    off_ratio = off_orth_norm / d_cs.clamp_min(EPS_DOT)
    lambda_eff = lam.clamp(-1.0, 2.0)
    offpath_dist = orth.norm(dim=1)                  # == ||orth||
    unit_off = orth / off_orth_norm[:, None].clamp_min(EPS_DOT)

    probs_c = z_cand.softmax(-1); probs_s = z_comm.softmax(-1)
    cand_pt = probs_c[ar, y]; comm_pt = probs_s[ar, y]

    to = lambda x: x.detach().cpu().numpy()
    cand_pred = z_cand.argmax(dim=1); comm_pred = z_comm.argmax(dim=1); prev_pred = z_prev.argmax(dim=1)

    # arithmetic mapping (numpy, int)
    oa_np = oa.detach().cpu().numpy().astype(np.int64)
    ob_np = ob.detach().cpu().numpy().astype(np.int64)
    y_np = y.detach().cpu().numpy().astype(np.int64)
    wrong_np = comm_pred.detach().cpu().numpy().astype(np.int64)
    wrong_offset = ((wrong_np - y_np + MODULUS // 2) % MODULUS) - MODULUS // 2   # [-5,4]
    dmod_wrong_true = _mod_distance(wrong_np, y_np)
    dmod_wrong_old = _mod_distance(wrong_np, oa_np)
    dmod_wrong_operand = _mod_distance(wrong_np, ob_np % MODULUS)

    cols: Dict[str, np.ndarray] = {
        "seed": np.full(n, seed, np.float64), "source": np.full(n, source_id, np.float64),
        "seq_id": to(b_i).astype(np.float64), "t": to(t_i).astype(np.float64),
        "dest": to(dest).astype(np.float64), "src_reg": to(src).astype(np.float64),
        "operand_a": oa_np.astype(np.float64), "operand_b": ob_np.astype(np.float64),
        "wrap": to(wrap.float()), "isolated": to(
            ((t_i == 0) | (op[b_i, (t_i - 1).clamp_min(0)] != RegisterTrackingTask.OP_ADD)).float()),
        "cand_ok": to((cand_pred == y).float()), "comm_ok": to((comm_pred == y).float()),
        "prev_ok": to((prev_pred == oa).float()), "comm_eq_old": to((comm_pred == oa).float()),
        "cand_pred": to(cand_pred).astype(np.float64), "comm_pred": wrong_np.astype(np.float64),
        "prev_pred": to(prev_pred).astype(np.float64),
        "y_true": y_np.astype(np.float64), "a_star_prev": to(a_prev).astype(np.float64),
        "a_star_commit": to(a_commit).astype(np.float64),
        "competitor_changed": to((a_commit != a_prev).float()),
        "m_prev": to(m_prev), "m_cand": to(m_cand), "m_commit": to(m_commit), "m_commit_alt": to(m_commit_alt),
        "dm_cand": to(dm_cand), "dm_commit": to(dm_commit),
        "rho_margin": to(rho), "rho_defined": to(rho_defined.float()), "alpha_star_full": to(alpha_full),
        "f_mean": to(fvec.mean(dim=1)),
        "d_cs": to(d_cs), "d_ss": to(d_ss), "d_sc": to(d_sc),
        "lambda_eff": to(lambda_eff), "offpath_dist": to(offpath_dist),
        "bias_av": to(bias_av), "state_av_commit": to(state_av_commit),
        "bias_ac": to(bias_ac), "state_ac_commit": to(state_ac_commit),
        "norm_dw_av": to(norm_dw_av), "norm_dw_ac": to(norm_dw_ac),
        "hyper_av_commit": to(hyper_av_commit), "hyper_ac_commit": to(hyper_ac_commit),
        "flat_reg_prev": (flat_prev // V).cpu().numpy().astype(np.float64),
        "flat_val_prev": (flat_prev % V).cpu().numpy().astype(np.float64),
        "flat_reg_cand": (flat_cand // V).cpu().numpy().astype(np.float64),
        "flat_val_cand": (flat_cand % V).cpu().numpy().astype(np.float64),
        "flat_reg_commit": (flat_comm // V).cpu().numpy().astype(np.float64),
        "flat_val_commit": (flat_comm % V).cpu().numpy().astype(np.float64),
        "mv_prev": to(mv_prev), "mv_cand": to(mv_cand), "mv_commit": to(mv_commit),
        "hyper_flat_commit": to(hyper_flat_commit), "norm_dw_flat": to(norm_dw_flat),
        "off_par_norm": to(off_par_norm), "off_orth_norm": to(off_orth_norm),
        "off_cos": to(off_cos), "off_ratio": to(off_ratio),
        "wrong_offset": wrong_offset.astype(np.float64),
        "dmod_wrong_true": dmod_wrong_true.astype(np.float64),
        "dmod_wrong_old": dmod_wrong_old.astype(np.float64),
        "dmod_wrong_operand": dmod_wrong_operand.astype(np.float64),
        "cand_pt": to(cand_pt), "comm_pt": to(comm_pt),
    }
    scalars = np.stack([cols[c] for c in COLS], axis=1).astype(np.float64)

    out: Dict[str, Any] = {"scalars": scalars, "W": to(W.float()), "b": to(bvec.float()),
                           "primary": None}
    if source_id == eg.SOURCE_IDS["test"]:
        gen_general = (dest != src) & (ob > 0)
        cand_ok = cand_pred == y
        comm_ok = comm_pred == y
        finite_rho = torch.isfinite(alpha_full) & torch.isfinite(rho)
        offpath = gen_general & cand_ok & (~comm_ok) & finite_rho & (rho >= alpha_full)
        succ = gen_general & cand_ok & comm_ok
        nonoff = gen_general & cand_ok & (~comm_ok) & ~(finite_rho & (rho >= alpha_full))
        group = torch.full((n,), -1, dtype=torch.long, device=device)
        group[offpath] = 0
        group[succ] = 1
        group[nonoff] = 2
        prow = torch.nonzero(group >= 0, as_tuple=False).squeeze(1)
        out["primary"] = {
            "row": to(prow).astype(np.int64), "group": to(group[prow]).astype(np.int64),
            "unit": to(unit_off[prow].float()),
            "fail_rows": to(torch.nonzero((group == 0) | (group == 2), as_tuple=False).squeeze(1)).astype(np.int64),
            "fail_S_prev": to(S_prev[(group == 0) | (group == 2)].float()),
            "fail_C": to(C[(group == 0) | (group == 2)].float()),
            "fail_S_commit": to(S_t[(group == 0) | (group == 2)].float()),
        }
    return out


def _concat(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalars = np.concatenate([p["scalars"] for p in parts], axis=0)
    W = np.stack([p["W"] for p in parts]); b = np.stack([p["b"] for p in parts])
    rows, groups, units, fr = [], [], [], []
    fS, fC, fSc = [], [], []
    offset = 0
    for p in parts:
        n = p["scalars"].shape[0]
        if p["primary"] is not None:
            rows.append(p["primary"]["row"] + offset)
            groups.append(p["primary"]["group"]); units.append(p["primary"]["unit"])
            fr.append(p["primary"]["fail_rows"] + offset)
            fS.append(p["primary"]["fail_S_prev"]); fC.append(p["primary"]["fail_C"])
            fSc.append(p["primary"]["fail_S_commit"])
        offset += n
    out = {
        "scalars": scalars, "W": W.astype(np.float32), "b": b.astype(np.float32),
        "primary_row": np.concatenate(rows) if rows else np.zeros(0, np.int64),
        "primary_group": np.concatenate(groups) if groups else np.zeros(0, np.int64),
        "primary_unit": np.concatenate(units, axis=0) if units else np.zeros((0, gr.STATE_DIM), np.float32),
        "fail_row": np.concatenate(fr) if fr else np.zeros(0, np.int64),
        "fail_S_prev": np.concatenate(fS, axis=0) if fS else np.zeros((0, gr.STATE_DIM), np.float32),
        "fail_C": np.concatenate(fC, axis=0) if fC else np.zeros((0, gr.STATE_DIM), np.float32),
        "fail_S_commit": np.concatenate(fSc, axis=0) if fSc else np.zeros((0, gr.STATE_DIM), np.float32),
    }
    return out


def verify_against_l(m: Dict[str, Any], l_scal: np.ndarray, l_cols: List[str], l_off_seed: np.ndarray,
                     model_key: str) -> Dict[str, Any]:
    """Exact alignment of M's re-collected rows against Experiment L."""
    li = {c: i for i, c in enumerate(l_cols)}
    mi = {c: i for i, c in enumerate(COLS)}
    if m["scalars"].shape[0] != l_scal.shape[0]:
        raise RuntimeError(f"{model_key}: row count differs M={m['scalars'].shape[0]} L={l_scal.shape[0]}")
    shared = ["seed", "source", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b",
              "wrap", "isolated", "cand_ok", "comm_ok", "prev_ok", "comm_eq_old", "cand_pred",
              "comm_pred", "prev_pred", "y_true", "a_star_prev", "a_star_commit",
              "competitor_changed"]
    maxd = {}
    for col in shared:
        if col in li and col in mi:
            a = m["scalars"][:, mi[col]]; b_ = l_scal[:, li[col]]
            maxd[col] = float(np.nanmax(np.abs(a - b_)))
    # off-path count must match L
    n_m_off = int((m["primary_group"] == 0).sum())
    checks = {"model": model_key, "n_rows_m": int(m["scalars"].shape[0]),
              "n_rows_l": int(l_scal.shape[0]), "max_abs_diff": maxd,
              "n_offpath_m": n_m_off, "passed": all(v == 0.0 for v in maxd.values())}
    if not checks["passed"]:
        raise RuntimeError(f"{model_key}: M/L alignment failed: {maxd}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment M collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--g-transitions", default="experiments/experiment_g_transitions.npz")
    ap.add_argument("--l-readout", default="experiments/experiment_l_readout.npz")
    ap.add_argument("--output", default="experiments/experiment_m_geometry.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_m_geometry_meta.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    gz = np.load(args.g_transitions, allow_pickle=False)
    lz = np.load(args.l_readout, allow_pickle=False)
    l_meta = json.loads(str(lz["meta_json"][0]))
    meta: Dict[str, Any] = {
        "cols": COLS, "seeds": gr.SEEDS, "models": {}, "l_alignment": {},
        "readout": l_meta["readout"],
        "offpath_definition": l_meta["offpath_definition"],
        "mod_distance": "d_mod(a,b) = min((a-b) mod 10, (b-a) mod 10)",
        "note": ("Measurement only: frozen checkpoints, no training, no optimizer, no weight/"
                 "readout/gate/commit modification, no intervention, no recursive counterfactuals."),
    }
    per_model: Dict[str, Any] = {}
    for model_key in eg.MODELS:
        parts: List[Dict[str, Any]] = []
        meta["models"][model_key] = {"checkpoints": {}}
        for seed in gr.SEEDS:
            loaded = eg.load_checkpoint(args.checkpoints_dir, model_key, seed)
            sha_before = loaded["sha256"]
            l_sha = l_meta["models"][model_key]["checkpoints"][str(seed)]["sha256"]
            if sha_before != l_sha:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint hash changed vs Experiment L")
            model = loaded["model"]
            batch = gr.canonical_splits(seed).test_set.get_batch(gr.TEST_SIZE, idx=0)
            parts.append(collect_dataset(model, batch.inputs, batch.metadata["step_targets"],
                                         seed, eg.SOURCE_IDS["test"]))
            if eg.sha256_file(loaded["path"]) != sha_before:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint changed during run")
            meta["models"][model_key]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {model_key} seed={seed}: cumulative rows={sum(len(p['scalars']) for p in parts)}")
        m = _concat(parts)
        l_scal = lz[f"{model_key}__scalars"].astype(np.float64)
        meta["l_alignment"][model_key] = verify_against_l(m, l_scal, l_meta["cols"],
                                                          lz[f"{model_key}__offpath_seed"], model_key)
        meta["l_alignment"][model_key]["n_offpath_l"] = int(l_meta["n_offpath"][model_key])
        per_model[model_key] = m
        print(f"  {model_key}: {m['scalars'].shape[0]} rows, primary={m['primary_row'].shape[0]} "
              f"(off-path={(m['primary_group']==0).sum()}), L alignment="
              f"{meta['l_alignment'][model_key]['passed']}")

    meta["n_rows"] = {k: int(v["scalars"].shape[0]) for k, v in per_model.items()}
    meta["n_primary"] = {k: int(v["primary_row"].shape[0]) for k, v in per_model.items()}
    meta["n_offpath"] = {k: int((v["primary_group"] == 0).sum()) for k, v in per_model.items()}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload: Dict[str, np.ndarray] = {"meta_json": np.array([json.dumps(meta)])}
    for model_key, m in per_model.items():
        payload[f"{model_key}__scalars"] = m["scalars"].astype(np.float32)
        payload[f"{model_key}__W"] = m["W"]; payload[f"{model_key}__b"] = m["b"]
        payload[f"{model_key}__primary_row"] = m["primary_row"]
        payload[f"{model_key}__primary_group"] = m["primary_group"]
        payload[f"{model_key}__primary_unit"] = m["primary_unit"]
        payload[f"{model_key}__fail_row"] = m["fail_row"]
        payload[f"{model_key}__fail_S_prev"] = m["fail_S_prev"]
        payload[f"{model_key}__fail_C"] = m["fail_C"]
        payload[f"{model_key}__fail_S_commit"] = m["fail_S_commit"]
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
