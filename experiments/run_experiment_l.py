#!/usr/bin/env python3
"""Experiment L — Readout-Side Localization of the Off-Path Component (measurement only).

Question
--------
When the ADD candidate is correct and the committed state contains substantial useful
decision movement, why can the frozen linear readout still decode the wrong value? Two
candidate explanations:

  A. Readout-side bias/calibration: the committed latent state is adequately placed, but the
     frozen linear readout assigns the wrong logits because of fixed offsets / class geometry.
  B. Latent state-placement geometry: the committed state itself lies in the wrong region of
     latent space (toward/inside a competing basin) despite sufficient aggregate movement.

L measures only. No training, no model/readout/gate/commit modification, no intervention, no
recursive counterfactual trajectories. Every counterfactual is a one-step analysis-only
computation on frozen states.

Frozen readout (actual implementation, not assumed from docs)
------------------------------------------------------------
`model.decision_head` is `TypedDecisionHead` whose `proj` is a single `nn.Linear(212, 40)`;
its weight reshaped to (R=4, V=10, D=212) is `W` and its bias reshaped to (R,V) is `b`, so for
destination register d and value v the logit is EXACTLY  z_{d,v}(S) = W[d,v] . S + b[d,v].

Primary population
------------------
strict test split | ADD transition | general stratum (dest != src AND source operand > 0) |
candidate correct | baseline commit wrong | rho_margin >= alpha*_full
(the exact Experiment I/J/K off-path definition; NOT redefined after seeing results). E is the
primary scientific model, F the secondary control; they are never pooled.

Conventions reused verbatim from Experiments H/I:
  a* = argmax_{a != y} z_a(S_prev)          (strongest competitor at S_prev, fixed reference)
  m(S) = z_y(S) - z_{a*}(S),  rho_margin = dm_commit / dm_cand  (defined when dm_cand > EPS_ABS)
  alpha*_full = minimal alpha with full argmax of head(S_prev + alpha (C - S_prev)) == y
  lambda_eff = <S_t - S_prev, C - S_prev> / ||C - S_prev||^2,  offpath_dist = ||S_t - proj||

Outputs
-------
    experiments/experiment_l_readout.npz        per-ADD-transition scalars + off-path vectors
    experiments/experiment_l_readout_meta.json  columns, checkpoint hashes, G/I/H checks

Reproduce
---------
    python experiments/run_experiment_l.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_l_readout.npz
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
import run_experiment_i as ri  # noqa: E402  (EPS_ABS, decision-margin conventions)
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

EPS_ABS = ri.EPS_ABS          # 1e-3, the documented ratio-instability threshold
EPS_DOT = 1e-6                # dot-product denominator guard

COLS = [
    "seed", "source", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b",
    "wrap", "isolated",
    "cand_ok", "comm_ok", "prev_ok", "comm_eq_old",
    "cand_pred", "comm_pred", "prev_pred",
    "y_true", "a_star_prev", "a_star_commit", "competitor_changed",
    "m_prev", "m_cand", "m_commit", "m_commit_alt",
    "dm_cand", "dm_commit", "margin_shortfall", "rho_margin", "rho_defined",
    "alpha_star_margin", "alpha_star_full",
    "f_mean",
    "d_cs", "d_ss", "d_sc", "lambda_eff", "offpath_dist",
    # pair (y, a_prev): bias and state-projection margins at S_prev / C / S_commit
    "bias_av", "state_av_prev", "state_av_cand", "state_av_commit",
    # pair (y, a_commit): bias and state-projection margins at S_prev / C / S_commit
    "bias_ac", "state_ac_prev", "state_ac_cand", "state_ac_commit",
    # bias-removed argmax decodes
    "nobias_prev_pred", "nobias_cand_pred", "nobias_comm_pred", "nobias_comm_ok",
    # pairwise decision direction movement
    "dot_av_commit", "dot_av_cand", "dot_ac_commit", "dot_ac_cand",
    "norm_dw_av", "norm_dw_ac",
    # signed normalized hyperplane distances (margin / ||dw||)
    "hyper_av_prev", "hyper_av_cand", "hyper_av_commit",
    "hyper_ac_prev", "hyper_ac_cand", "hyper_ac_commit",
    "cand_pt", "comm_pt",
]

OFF_VEC_KEYS = ["S_prev", "C", "S_commit"]


def _matrix(head: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """(W (R,V,D), b (R,V)) from the actual single nn.Linear readout."""
    R, V = gr.NUM_REGISTERS, gr.NUM_VALUES
    W = head.proj.weight.detach().view(R, V, gr.STATE_DIM)
    b = head.proj.bias.detach().view(R, V)
    return W, b


@torch.no_grad()
def collect_dataset(model: torch.nn.Module, inputs: torch.Tensor, step_targets: torch.Tensor,
                    seed: int, source_id: int) -> Dict[str, Any]:
    """Per-ADD-transition readout-decomposition records for one (model, seed, dataset)."""
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    step_targets = step_targets.to(device)
    B, T, _ = inputs.shape
    _, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, dim=1)      # (B, T, D)
    cands = torch.stack(diag.candidates, dim=1)
    gates = torch.stack(diag.gates, dim=1)
    prev_state = torch.cat([torch.zeros(B, 1, gr.STATE_DIM, device=device), states[:, :-1]], dim=1)
    prev_true = torch.zeros_like(step_targets)
    prev_true[:, 1:] = step_targets[:, :-1]

    head = model.decision_head
    R, V, D = gr.NUM_REGISTERS, gr.NUM_VALUES, gr.STATE_DIM
    W, bvec = _matrix(head)

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
    wrap = (oa + ob) >= gr.NUM_VALUES
    ar = torch.arange(n, device=device)

    # logits z_{d,v}(S) for the destination register
    def zmat(S: torch.Tensor) -> torch.Tensor:
        Wd = W[dest]                                    # (n, V, D)
        bd = bvec[dest]                                 # (n, V)
        return torch.einsum("nd,nvd->nv", S, Wd) + bd

    z_prev, z_cand, z_comm = zmat(S_prev), zmat(C), zmat(S_t)

    # strongest competitor at S_prev (fixed reference) and at the commit
    Mp = z_prev[ar, y][:, None] - z_prev
    Mm = Mp.clone(); Mm[ar, y] = float("inf")
    a_prev = Mm.argmin(dim=1)
    Mc = z_comm[ar, y][:, None] - z_comm
    Mcm = Mc.clone(); Mcm[ar, y] = float("inf")
    a_commit = Mcm.argmin(dim=1)
    competitor_changed = (a_commit != a_prev).float()

    m_prev = Mp.gather(1, a_prev[:, None]).squeeze(1)
    m_cand = (z_cand[ar, y][:, None] - z_cand).gather(1, a_prev[:, None]).squeeze(1)
    m_commit = Mc.gather(1, a_prev[:, None]).squeeze(1)
    m_commit_alt = Mc.gather(1, a_commit[:, None]).squeeze(1)
    dm_cand_all = (z_cand[ar, y][:, None] - z_cand) - Mp
    dm_cand = dm_cand_all.gather(1, a_prev[:, None]).squeeze(1)
    dm_commit = m_commit - m_prev
    margin_shortfall = m_cand - m_commit
    nan_t = torch.full_like(dm_cand, float("nan"))
    rho_defined = dm_cand > EPS_ABS
    rho = torch.where(rho_defined, dm_commit / dm_cand.clamp_min(EPS_ABS), nan_t)

    # alpha* (exact for the linear head)
    alpha_margin = torch.where(
        m_prev >= 0, torch.zeros_like(m_prev),
        torch.where(dm_cand > EPS_ABS, -m_prev / dm_cand.clamp_min(EPS_ABS), nan_t))
    neg = Mp < 0
    need = torch.where(neg, (-Mp) / dm_cand_all.clamp_min(EPS_ABS), torch.zeros_like(Mp))
    infeasible = neg & (dm_cand_all <= EPS_ABS)
    alpha_full = need.max(dim=1).values
    alpha_full = torch.where(m_prev >= 0, torch.zeros_like(alpha_full), alpha_full)
    alpha_full = torch.where(infeasible.any(dim=1), nan_t, alpha_full)

    # --- bias vs state decomposition (exact; readout is linear) ---
    def pair(dw: torch.Tensor, db: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,nd->n", dw, S)          # state margin (w_y-w_a).S

    w_y = W[dest, y]
    w_av = W[dest, a_prev]; w_ac = W[dest, a_commit]
    b_av = bvec[dest, a_prev]; b_ac = bvec[dest, a_commit]
    dw_av = w_y - w_av; dw_ac = w_y - w_ac
    bias_av = bvec[dest, y] - b_av
    bias_ac = bvec[dest, y] - b_ac
    state_av_prev, state_av_cand, state_av_commit = pair(dw_av, bias_av, S_prev), pair(dw_av, bias_av, C), pair(dw_av, bias_av, S_t)
    state_ac_prev, state_ac_cand, state_ac_commit = pair(dw_ac, bias_ac, S_prev), pair(dw_ac, bias_ac, C), pair(dw_ac, bias_ac, S_t)

    # bias-removed decodes: argmax_v (W[d,v].S)
    def nobias_pred(S: torch.Tensor) -> torch.Tensor:
        Wd = W[dest]
        return torch.einsum("nd,nvd->nv", S, Wd).argmax(-1)
    nb_prev, nb_cand, nb_comm = nobias_pred(S_prev), nobias_pred(C), nobias_pred(S_t)

    # pairwise decision-direction movement
    dot_av_commit = torch.einsum("nd,nd->n", dw_av, S_t - S_prev)
    dot_av_cand = torch.einsum("nd,nd->n", dw_av, C - S_prev)
    dot_ac_commit = torch.einsum("nd,nd->n", dw_ac, S_t - S_prev)
    dot_ac_cand = torch.einsum("nd,nd->n", dw_ac, C - S_prev)
    norm_dw_av = dw_av.norm(dim=1).clamp_min(EPS_DOT)
    norm_dw_ac = dw_ac.norm(dim=1).clamp_min(EPS_DOT)

    # signed normalized hyperplane distances (margin / ||dw||)
    hyper_av_prev = m_prev / norm_dw_av
    hyper_av_cand = m_cand / norm_dw_av
    hyper_av_commit = m_commit / norm_dw_av
    ac_margins = (z_comm[ar, y][:, None] - z_comm).gather(1, a_commit[:, None]).squeeze(1)
    ac_cand = (z_cand[ar, y][:, None] - z_cand).gather(1, a_commit[:, None]).squeeze(1)
    ac_prev = (z_prev[ar, y][:, None] - z_prev).gather(1, a_commit[:, None]).squeeze(1)
    hyper_ac_prev = ac_prev / norm_dw_ac
    hyper_ac_cand = ac_cand / norm_dw_ac
    hyper_ac_commit = ac_margins / norm_dw_ac

    # distances + H's projection conventions
    d_cs = (C - S_prev).norm(dim=1)
    d_ss = (S_t - S_prev).norm(dim=1)
    d_sc = (S_t - C).norm(dim=1)
    diff = C - S_prev
    denom = (diff * diff).sum(dim=1).clamp_min(1e-12)
    lambda_eff = (((S_t - S_prev) * diff).sum(dim=1) / denom).clamp(-1.0, 2.0)
    offpath_dist = (S_t - (S_prev + lambda_eff[:, None] * diff)).norm(dim=1)

    probs_c = z_cand.softmax(-1)
    probs_s = z_comm.softmax(-1)
    cand_pt = probs_c[ar, y]
    comm_pt = probs_s[ar, y]

    to = lambda x: x.detach().cpu().numpy()
    def f32(x: torch.Tensor) -> np.ndarray:
        return to(x.float())

    src_pred = z_cand.argmax(dim=1)
    comm_pred = z_comm.argmax(dim=1)
    prev_pred = z_prev.argmax(dim=1)

    cols: Dict[str, np.ndarray] = {
        "seed": np.full(n, seed, np.float64), "source": np.full(n, source_id, np.float64),
        "seq_id": to(b_i).astype(np.float64), "t": to(t_i).astype(np.float64),
        "dest": to(dest).astype(np.float64), "src_reg": to(src).astype(np.float64),
        "operand_a": to(oa).astype(np.float64), "operand_b": to(ob).astype(np.float64),
        "wrap": to(wrap.float()), "isolated": to(
            ((t_i == 0) | (op[b_i, (t_i - 1).clamp_min(0)] != RegisterTrackingTask.OP_ADD)).float()),
        "cand_ok": to((src_pred == y).float()), "comm_ok": to((comm_pred == y).float()),
        "prev_ok": to((prev_pred == oa).float()), "comm_eq_old": to((comm_pred == oa).float()),
        "cand_pred": to(src_pred).astype(np.float64), "comm_pred": to(comm_pred).astype(np.float64),
        "prev_pred": to(prev_pred).astype(np.float64),
        "y_true": to(y).astype(np.float64), "a_star_prev": to(a_prev).astype(np.float64),
        "a_star_commit": to(a_commit).astype(np.float64), "competitor_changed": to(competitor_changed),
        "m_prev": to(m_prev), "m_cand": to(m_cand), "m_commit": to(m_commit), "m_commit_alt": to(m_commit_alt),
        "dm_cand": to(dm_cand), "dm_commit": to(dm_commit), "margin_shortfall": to(margin_shortfall),
        "rho_margin": to(rho), "rho_defined": to(rho_defined.float()),
        "alpha_star_margin": to(alpha_margin), "alpha_star_full": to(alpha_full),
        "f_mean": to(fvec.mean(dim=1)),
        "d_cs": to(d_cs), "d_ss": to(d_ss), "d_sc": to(d_sc),
        "lambda_eff": to(lambda_eff), "offpath_dist": to(offpath_dist),
        "bias_av": to(bias_av), "state_av_prev": to(state_av_prev), "state_av_cand": to(state_av_cand),
        "state_av_commit": to(state_av_commit),
        "bias_ac": to(bias_ac), "state_ac_prev": to(state_ac_prev), "state_ac_cand": to(state_ac_cand),
        "state_ac_commit": to(state_ac_commit),
        "nobias_prev_pred": to(nb_prev).astype(np.float64), "nobias_cand_pred": to(nb_cand).astype(np.float64),
        "nobias_comm_pred": to(nb_comm).astype(np.float64), "nobias_comm_ok": to((nb_comm == y).float()),
        "dot_av_commit": to(dot_av_commit), "dot_av_cand": to(dot_av_cand),
        "dot_ac_commit": to(dot_ac_commit), "dot_ac_cand": to(dot_ac_cand),
        "norm_dw_av": to(norm_dw_av), "norm_dw_ac": to(norm_dw_ac),
        "hyper_av_prev": to(hyper_av_prev), "hyper_av_cand": to(hyper_av_cand),
        "hyper_av_commit": to(hyper_av_commit),
        "hyper_ac_prev": to(hyper_ac_prev), "hyper_ac_cand": to(hyper_ac_cand),
        "hyper_ac_commit": to(hyper_ac_commit),
        "cand_pt": to(cand_pt), "comm_pt": to(comm_pt),
    }
    scalars = np.stack([cols[c] for c in COLS], axis=1).astype(np.float64)

    # off-path vector subset (test split only) for projection/perturbation reuse
    out: Dict[str, Any] = {"scalars": scalars, "offpath_local": None, "offpath": None, "W": to(W.float()), "b": to(bvec.float())}
    if source_id == eg.SOURCE_IDS["test"]:
        gen_general = (dest != src) & (ob > 0)
        cand_ok = src_pred == y
        comm_ok = comm_pred == y
        offpath = gen_general & cand_ok & (~comm_ok) & torch.isfinite(alpha_full) & (rho >= alpha_full)
        ol = torch.nonzero(offpath, as_tuple=False).squeeze(1)
        if ol.numel() > 0:
            out["offpath_local"] = to(ol).astype(np.int64)
            out["offpath"] = {
                "S_prev": to(S_prev[ol].float()), "C": to(C[ol].float()), "S_commit": to(S_t[ol].float()),
                "seed": np.full(int(ol.numel()), seed, np.int64),
            }
    return out


def _concat(parts: List[Dict[str, Any]], model_key: str) -> Dict[str, Any]:
    scalars = np.concatenate([p["scalars"] for p in parts], axis=0)
    off_vec: Dict[str, List[np.ndarray]] = {k: [] for k in OFF_VEC_KEYS}
    off_seed: List[np.ndarray] = []
    Ws, bs = [], []
    for p in parts:
        if p["offpath"] is not None:
            for k in OFF_VEC_KEYS:
                off_vec[k].append(p["offpath"][k])
            off_seed.append(p["offpath"]["seed"])
        Ws.append(p["W"]); bs.append(p["b"])
    out = {
        "scalars": scalars,
        "offpath": {k: (np.concatenate(v, axis=0) if v else np.zeros((0, gr.STATE_DIM), np.float32))
                    for k, v in off_vec.items()},
        "offpath_seed": np.concatenate(off_seed) if off_seed else np.zeros(0, np.int64),
        "W": np.stack(Ws).astype(np.float32), "b": np.stack(bs).astype(np.float32),
    }
    return out


def verify_against_g(l: Dict[str, Any], g_arr: np.ndarray, g_cols: List[str], model_key: str) -> Dict[str, Any]:
    """Row-level verification against Experiment G, restricted to the test-split rows."""
    gi = {c: i for i, c in enumerate(g_cols)}
    l_scal = l["scalars"]
    ci = {c: i for i, c in enumerate(COLS)}
    # G rows with source == 0 (canonical test split), in order
    src_g = np.round(g_arr[:, gi["source"]]).astype(np.int64)
    g_test = g_arr[src_g == 0]
    checks: Dict[str, Any] = {"model": model_key, "n_rows_l": int(l_scal.shape[0]), "n_rows_g_test": int(g_test.shape[0])}
    if l_scal.shape[0] != g_test.shape[0]:
        raise RuntimeError(f"{model_key}: L/G-test row count mismatch L={l_scal.shape[0]} G_test={g_test.shape[0]}")
    exact_cols = ["seed", "source", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b",
                  "wrap", "isolated", "cand_ok", "comm_ok", "prev_ok", "comm_eq_old",
                  "cand_pred", "comm_pred", "prev_pred"]
    mism: Dict[str, int] = {}
    for col in exact_cols:
        a = np.round(l_scal[:, ci[col]]).astype(np.int64)
        b = np.round(g_test[:, gi[col]]).astype(np.int64)
        nb = int((a != b).sum())
        if nb:
            mism[col] = nb
    checks["exact_mismatches"] = mism
    checks["max_f_mean_diff"] = float(np.abs(l_scal[:, ci["f_mean"]] - g_test[:, gi["f_mean"]]).max())
    checks["passed"] = (len(mism) == 0) and checks["max_f_mean_diff"] < 1e-4
    if not checks["passed"]:
        raise RuntimeError(f"{model_key}: L/G verification failed: {checks}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment L collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--g-transitions", default="experiments/experiment_g_transitions.npz")
    ap.add_argument("--output", default="experiments/experiment_l_readout.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_l_readout_meta.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    gz = np.load(args.g_transitions, allow_pickle=False)
    g_meta = json.loads(str(gz["meta_json"][0]))
    meta: Dict[str, Any] = {
        "cols": COLS, "off_vec_keys": OFF_VEC_KEYS, "eps_abs": EPS_ABS, "eps_dot": EPS_DOT,
        "seeds": gr.SEEDS, "models": {}, "g_verification": {},
        "readout": "TypedDecisionHead = single nn.Linear(212, 4*10); z_{d,v}(S)=W[d,v].S+b[d,v] exactly",
        "offpath_definition": ("strict test split, ADD, general (dest!=src AND operand_b>0), candidate "
                               "correct, baseline commit wrong, rho_margin >= alpha_star_full"),
        "note": ("Measurement only: frozen checkpoints, no training, no optimizer, no weight "
                 "modification, no intervention. Counterfactuals (bias removal, perturbations) are "
                 "one-step analysis-only computations on frozen states, never used in a rollout."),
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
            sha_after = eg.sha256_file(loaded["path"])
            if sha_after != sha_before:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint file changed during run")
            meta["models"][model_key]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {model_key} seed={seed}: cumulative rows={sum(len(p['scalars']) for p in parts)}")
        l = _concat(parts, model_key)
        meta["g_verification"][model_key] = verify_against_g(l, g_arr, g_meta["columns"], model_key)
        per_model[model_key] = l
        print(f"  {model_key}: {l['scalars'].shape[0]} test ADD transitions, "
              f"{l['offpath']['S_prev'].shape[0]} off-path vectors; G verification="
              f"{meta['g_verification'][model_key]['passed']}")

    meta["n_rows"] = {k: int(v["scalars"].shape[0]) for k, v in per_model.items()}
    meta["n_offpath"] = {k: int(v["offpath"]["S_prev"].shape[0]) for k, v in per_model.items()}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload: Dict[str, np.ndarray] = {"meta_json": np.array([json.dumps(meta)])}
    for model_key, l in per_model.items():
        payload[f"{model_key}__scalars"] = l["scalars"].astype(np.float32)
        payload[f"{model_key}__W"] = l["W"]
        payload[f"{model_key}__b"] = l["b"]
        payload[f"{model_key}__offpath_seed"] = l["offpath_seed"]
        for k in OFF_VEC_KEYS:
            payload[f"{model_key}__offpath_{k}"] = l["offpath"][k]
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
