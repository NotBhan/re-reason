#!/usr/bin/env python3
"""Experiment K — Coordinate-Preserving Gate-Balance Ablation (interventional).

Purpose
-------
Experiment J established that globally *increasing* the element-wise commit transfer
(`f' = clip(gamma f, 0, 1)`) raises decision-margin delivery and rescues a minority of
candidate->commit failures, but never improves T=10/T=20 sequence EM and degrades it for
gamma >= 1.25. Experiment I additionally showed failures *under-move* the decision-helping
coordinates while relatively *over*-delivering harmful coordinates. K tests one narrower
hypothesis:

    Does changing the SHAPE / BALANCE of the existing element-wise gate -- while preserving
    coordinate identity and WITHOUT simply increasing aggregate transfer everywhere -- improve
    inference-time performance?

This is a controlled intervention, NOT a production fix and NOT architecture adoption.

Primary intervention (pre-registered, exactly one family)
---------------------------------------------------------
    f'_i = sigmoid(kappa * logit(clip(f_i, eps, 1-eps))),    eps = 1e-6
    kappa = 2.0  (sharpening)   vs   kappa = 1.0  (identity baseline)

The transform is coordinate-preserving and monotone about 0.5:
  * f < 0.5  => f' < 0.5;   f = 0.5 => f' = 0.5 exactly;   f > 0.5 => f' > 0.5;
  * coordinate identity/order is preserved (element-wise, no permutation);
  * already-near-0 / near-1 gates become MORE saturated;
  * mid-range gates are pushed toward their existing side of 0.5;
  * this is NOT equivalent to the global multiplication f -> gamma f;
  * it does not intentionally increase every coordinate's transfer uniformly.
kappa = 1.0 is the identity member of the family and is returned unchanged to guarantee
exact baseline parity.

The intervention depends ONLY on f_i (no label, no logits, no decision margin, no register /
operation identity, no candidate correctness, no analysis quantity from Experiments I/J).

Recursive rollout
-----------------
The FULL 212-dim gate is transformed, then the actual recursive commit is performed:
    S'_t = (1 - f'_t) . S'_(t-1) + f'_t . C_t
and candidate generation / gate computation at t+1 use the modified state S'_t exactly as
the frozen core does (reusing Experiment J's `rollout`, which tracks the modified trajectory).

PRE-REGISTRATION: the two kappa values, the family, the controls, and the A/B/C/D decision
rule are fixed before any result is inspected. No kappa value is added or removed after
seeing results; no kappa is tuned or searched. No new random/scramble control is introduced
(Experiment J already established that coordinate scrambling is catastrophic; K's question is
gate SHAPE while preserving coordinate identity).

Measurement only: frozen checkpoints, no optimizer, no training, torch.no_grad() throughout.
Checkpoint SHA-256 values are verified before and after the run.

Outputs
-------
    experiments/experiment_k_records.json    per (model, seed, kappa) aggregates + gate shape
Reproduce
---------
    python experiments/run_experiment_k.py --out experiments/experiment_k_records.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402
import run_experiment_i as ri  # noqa: E402  (EPS_ABS, _logit_matrix conventions)
import run_experiment_j as rj  # noqa: E402  (rollout / parity / rescue helpers)
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

# ---------------------------------------------------------------------------
# PRE-REGISTRATION (do not edit after running)
# ---------------------------------------------------------------------------

EPS = 1e-6
KAPPA_GRID = [1.0, 2.0]                  # fixed; kappa=1.0 is the identity baseline
BASELINE_KAPPA = 1.0
STATE_DIM = gr.STATE_DIM                 # 212 (exported for tests/tools)
PRIMARY_FAMILY = "sharpen"               # f' = sigmoid(kappa * logit(clip(f, eps, 1-eps)))
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20]
MODEL_PLAN = {
    "E_candidate": [PRIMARY_FAMILY],     # primary scientific model
    "F_combined": [PRIMARY_FAMILY],      # secondary control (simple; identical code path)
}
HIST_BINS = 50

PREREGISTRATION = {
    "kind": "pre-registered two-condition intervention",
    "kappa_grid": KAPPA_GRID, "baseline_kappa": BASELINE_KAPPA,
    "primary_family": PRIMARY_FAMILY, "eps": EPS,
    "transform": "f'_i = sigmoid(kappa * logit(clip(f_i, eps, 1-eps)))",
    "horizons": HORIZONS, "model_plan": MODEL_PLAN,
    "decision_rule": (
        "A: beneficial, sign-consistent (>=4/5 seeds) and conventionally large (|Cohen's dz| >= 0.8, "
        "p < 0.05) paired SEQUENCE-LEVEL effect at T=10 or T=20, together with net candidate->commit "
        "rescue (rescue>regression) and no large operation-level regression. "
        "B: the local commit mechanism moves (margin / coordinate delivery) or net rescue is positive, "
        "but there is no beneficial sequence-level effect. "
        "C: no meaningful benefit (neither margin/coordinate benefit nor sequence benefit). "
        "D: harmful sequence-level effect with neither margin nor rescue benefit. "
        "Local-only improvement with T10/T20 unchanged classifies as B, not A."),
    "note": ("Intervention on the full 212-dim gate vector only, as a coordinate-preserving "
             "monotone remap about 0.5; recursive rollout; no readout/label/decision-margin "
             "information enters the intervention. kappa values fixed before results; no "
             "random/scramble control is introduced (Experiment J already covered scattering)."),
}


# ---------------------------------------------------------------------------
# Commit-rule family (operates on the full gate vector; no analysis quantities)
# ---------------------------------------------------------------------------

def tighten(f: torch.Tensor, kappa: float, eps: float = EPS) -> torch.Tensor:
    """Coordinate-preserving monotone sharpening of a gate vector about 0.5.

    f'_i = sigmoid(kappa * logit(clip(f_i, eps, 1-eps))).  kappa = 1.0 returns f unchanged.
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if kappa == 1.0:
        return f
    fc = f.clamp(eps, 1.0 - eps)
    return torch.sigmoid(kappa * torch.logit(fc))


def make_commit_fn(family: str, kappa: float) -> Callable:
    """Returns f -> f' for the pre-registered commit-rule family. All outputs in [0,1]."""
    if family != PRIMARY_FAMILY:
        raise ValueError(f"unknown family {family!r}")
    if kappa <= 0:
        raise ValueError("sharpen requires kappa > 0")
    return lambda f: tighten(f, kappa)


def validate_kappas(kappas: List[float]) -> List[float]:
    """Only kappa values inside the pre-registered grid are permitted (no post-hoc search)."""
    for k in kappas:
        if k not in KAPPA_GRID:
            raise ValueError(f"kappa {k} is not in the pre-registered grid {KAPPA_GRID}")
    return list(kappas)


# ---------------------------------------------------------------------------
# Gate-shape statistics
# ---------------------------------------------------------------------------

def _gate_dist(g: torch.Tensor) -> Dict[str, float]:
    """Distribution summary of a tensor of gate values (any shape)."""
    flat = g.reshape(-1)
    gc = flat.clamp(1e-7, 1.0 - 1e-7)
    ent = float((-(gc * gc.log2() + (1 - gc) * (1 - gc).log2())).mean())
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean()), "median": float(flat.median()), "std": float(flat.std()),
        "entropy": ent,
        "frac_lt_025": float((flat < 0.25).float().mean()),
        "frac_025_05": float(((flat >= 0.25) & (flat < 0.5)).float().mean()),
        "frac_05_075": float(((flat >= 0.5) & (flat < 0.75)).float().mean()),
        "frac_ge_075": float((flat >= 0.75).float().mean()),
    }


def _per_transition_gate_shape(f: torch.Tensor, fm: torch.Tensor) -> Dict[str, np.ndarray]:
    """Per-row (transition) gate shape statistics before (f) and after (fm). Shape (n, D)."""
    def row(g: torch.Tensor) -> Dict[str, np.ndarray]:
        gc = g.clamp(1e-7, 1.0 - 1e-7)
        return {
            "mean": g.mean(dim=1), "median": g.median(dim=1).values,
            "entropy": (-(gc * gc.log2() + (1 - gc) * (1 - gc).log2())).mean(dim=1),
            "frac_lt_025": (g < 0.25).float().mean(dim=1),
            "frac_025_05": ((g >= 0.25) & (g < 0.5)).float().mean(dim=1),
            "frac_05_075": ((g >= 0.5) & (g < 0.75)).float().mean(dim=1),
            "frac_ge_075": (g >= 0.75).float().mean(dim=1),
        }
    before = row(f)
    after = row(fm)
    d = fm - f
    out: Dict[str, np.ndarray] = {}
    for k in before:
        out[f"before_{k}"] = before[k].detach().cpu().numpy()
        out[f"after_{k}"] = after[k].detach().cpu().numpy()
        out[f"delta_{k}"] = (after[k] - before[k]).detach().cpu().numpy()
    out["abs_change"] = d.abs().mean(dim=1).detach().cpu().numpy()
    out["signed_change"] = d.mean(dim=1).detach().cpu().numpy()
    out["transfer_change"] = (fm.mean(dim=1) - f.mean(dim=1)).detach().cpu().numpy()
    return out


def _hist(x: np.ndarray, lo: float, hi: float, bins: int = HIST_BINS) -> Dict[str, Any]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts],
            "n": int(x.size)}


def _agg(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "median": None, "sd": None, "p05": None, "p95": None, "n": 0}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)), "n": int(x.size)}


# ---------------------------------------------------------------------------
# Transition-level + decision-margin arrays on the canonical T=10 test split
# (extends Experiment J's `test_transition_data`: same conventions + gw_all/gw_mag,
#  competitor-change tracking and gate-shape statistics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_transition_data(model: torch.nn.Module, dataset, commit_fn: Callable) -> Dict[str, Any]:
    inputs = dataset.inputs
    step_targets = dataset.step_targets
    targets = dataset.targets
    mask, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)
    roll = rj.rollout(model, inputs, commit_fn)
    states, cands, gates, gates_mod = roll["states"], roll["cands"], roll["gates"], roll["gates_mod"]
    B, T, D = states.shape
    dev = states.device
    head = model.decision_head

    cp = head(states.reshape(-1, D)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)
    kp = head(cands.reshape(-1, D)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)
    cok = cp == step_targets
    kok = kp == step_targets

    comm_full_mean = float(cok.all(-1).float().mean())
    comm_written_mean = float((cok | ~mask).all(-1).float().mean())
    cand_full_mean = float(kok.all(-1).float().mean())
    cand_written_mean = float((kok | ~mask).all(-1).float().mean())

    cs = (~cok.all(-1)).cpu().numpy()
    cr = (~((cok | ~mask).all(-1))).cpu().numpy()
    first_strict = np.full(B, -1, np.int64); first_real = np.full(B, -1, np.int64)
    for b in range(B):
        w = np.nonzero(cs[b])[0]
        if w.size:
            first_strict[b] = int(w[0]) + 1
        w = np.nonzero(cr[b])[0]
        if w.size:
            first_real[b] = int(w[0]) + 1

    g = gates.clamp(1e-7, 1 - 1e-7)
    gm = gates_mod.clamp(1e-7, 1 - 1e-7)
    gate_entropy = float((-(g * g.log2() + (1 - g) * (1 - g).log2())).mean())
    gate_entropy_mod = float((-(gm * gm.log2() + (1 - gm) * (1 - gm).log2())).mean())
    seq_em = (roll["term_logits"].argmax(-1) == targets).all(-1)

    # per-ADD-transition decision margins (intervention trajectory), Experiment I/J conventions
    prev_state = torch.cat([torch.zeros(B, 1, D, device=dev), states[:, :-1]], dim=1)
    prev_true = torch.zeros_like(step_targets); prev_true[:, 1:] = step_targets[:, :-1]
    W = head.proj.weight.detach().view(gr.NUM_REGISTERS, gr.NUM_VALUES, D)
    bvec = head.proj.bias.detach().view(gr.NUM_REGISTERS, gr.NUM_VALUES)
    idx = (op == RegisterTrackingTask.OP_ADD).nonzero()
    bi, ti = idx[:, 0], idx[:, 1]
    n = int(len(idx))
    S_prev = prev_state[bi, ti]; C = cands[bi, ti]; S_t = states[bi, ti]
    fvec = gates[bi, ti]; fmod = gates_mod[bi, ti]
    dest = reg_a[bi, ti]; src = reg_b[bi, ti]
    oa = prev_true[bi, ti, dest]; ob = prev_true[bi, ti, src]
    yv = step_targets[bi, ti, dest]
    ar = torch.arange(n, device=dev)
    z_prev = ri._logit_matrix(W, bvec, S_prev, dest)
    z_cand = ri._logit_matrix(W, bvec, C, dest)
    z_comm = ri._logit_matrix(W, bvec, S_t, dest)
    Mprev = z_prev[ar, yv][:, None] - z_prev
    Mcand = z_cand[ar, yv][:, None] - z_cand
    Mcomm = z_comm[ar, yv][:, None] - z_comm
    Mmask = Mprev.clone(); Mmask[ar, yv] = float("inf")
    a_star = Mmask.argmin(dim=1)
    Mmask_c = Mcomm.clone(); Mmask_c[ar, yv] = float("inf")
    a_star_commit = Mmask_c.argmin(dim=1)
    m_prev = Mprev.gather(1, a_star[:, None]).squeeze(1)
    m_cand = Mcand.gather(1, a_star[:, None]).squeeze(1)
    m_commit = Mcomm.gather(1, a_star[:, None]).squeeze(1)
    dm_cand_all = Mcand - Mprev
    dm_cand = dm_cand_all.gather(1, a_star[:, None]).squeeze(1)
    dm_commit = m_commit - m_prev
    margin_shortfall = m_cand - m_commit
    nan_t = torch.full_like(dm_cand, float("nan"))
    rho = torch.where(dm_cand > ri.EPS_ABS, dm_commit / dm_cand.clamp_min(ri.EPS_ABS), nan_t)
    # alpha*_margin (fixed a*) and alpha*_full (all competitors), exact for the linear head
    alpha_margin = torch.where(
        m_prev >= 0, torch.zeros_like(m_prev),
        torch.where(dm_cand > ri.EPS_ABS, -m_prev / dm_cand.clamp_min(ri.EPS_ABS), nan_t))
    neg_needed = Mprev < 0
    need = torch.where(neg_needed, (-Mprev) / dm_cand_all.clamp_min(ri.EPS_ABS), torch.zeros_like(Mprev))
    infeasible = neg_needed & (dm_cand_all <= ri.EPS_ABS)
    alpha_full = need.max(dim=1).values
    alpha_full = torch.where(m_prev >= 0, torch.zeros_like(alpha_full), alpha_full)
    alpha_full = torch.where(infeasible.any(dim=1), nan_t, alpha_full)
    # coordinate decomposition (exact for the linear readout) using the MODIFIED gate
    w_y = W[dest, yv]; w_star = W[dest, a_star]
    dvec = C - S_prev
    qcand = (w_y - w_star) * dvec
    qcand_pos = qcand.clamp(min=0.0); qcand_neg = (-qcand).clamp(min=0.0)
    qpos = qcand_pos.sum(dim=1); qneg = qcand_neg.sum(dim=1)
    gw_pos = torch.where(qpos > ri.EPS_ABS, (qcand_pos * fmod).sum(dim=1) / qpos.clamp_min(ri.EPS_ABS), nan_t)
    gw_neg = torch.where(qneg > ri.EPS_ABS, (qcand_neg * fmod).sum(dim=1) / qneg.clamp_min(ri.EPS_ABS), nan_t)
    gw_all = (qcand.abs() * fmod).sum(dim=1) / qcand.abs().sum(dim=1).clamp_min(ri.EPS_ABS)
    gw_mag = (dvec.abs() * fmod).sum(dim=1) / dvec.abs().sum(dim=1).clamp_min(ri.EPS_ABS)

    to = lambda x: x.detach().cpu().numpy()
    cand_pred = kp[bi, ti, dest]; comm_pred = cp[bi, ti, dest]

    primary_local = ((dest != src) & (ob > 0) & (z_cand.argmax(dim=1) == yv))
    pl = torch.nonzero(primary_local, as_tuple=False).squeeze(1)
    if pl.numel() > 0:
        coord_signed_mean_per_dim = (fmod[pl] - fvec[pl]).mean(dim=0).detach().cpu().numpy()
        coord_delta_hist = _hist(to((fmod[pl] - fvec[pl]).reshape(-1)), -0.5, 0.5)
        coord_abs_hist = _hist(to((fmod[pl] - fvec[pl]).abs().reshape(-1)), 0.0, 0.5)
    else:
        coord_signed_mean_per_dim = np.zeros(D, np.float32)
        coord_delta_hist = _hist(np.zeros(0), -0.5, 0.5)
        coord_abs_hist = _hist(np.zeros(0), 0.0, 0.5)

    pts = _per_transition_gate_shape(fvec, fmod)

    return {
        "cand_ok": to((cand_pred == yv).float()), "comm_ok": to((comm_pred == yv).float()),
        "gen_general": to(((dest != src) & (ob > 0)).float()),
        "wrap": to(((oa + ob) >= gr.NUM_VALUES).float()),
        "m_prev": to(m_prev), "m_cand": to(m_cand), "m_commit": to(m_commit),
        "dm_cand": to(dm_cand), "dm_commit": to(dm_commit), "rho": to(rho),
        "margin_shortfall": to(margin_shortfall),
        "alpha_full": to(alpha_full), "alpha_margin": to(alpha_margin),
        "gw_pos": to(gw_pos), "gw_neg": to(gw_neg), "gw_all": to(gw_all), "gw_mag": to(gw_mag),
        "f_mean": to(fvec.mean(dim=1)), "f_scaled_mean": to(fmod.mean(dim=1)),
        "a_star_prev": to(a_star), "a_star_commit": to(a_star_commit),
        "competitor_changed": to((a_star_commit != a_star).float()),
        "n_add": n,
        "scalar": {
            "comm_full_mean": comm_full_mean, "comm_written_mean": comm_written_mean,
            "cand_full_mean": cand_full_mean, "cand_written_mean": cand_written_mean,
            "first_strict_error_rate": float((first_strict > 0).mean()),
            "first_real_error_rate": float((first_real > 0).mean()),
            "gate_mean": float(gates.mean()), "gate_scaled_mean": float(gates_mod.mean()),
            "gate_entropy": gate_entropy, "gate_scaled_entropy": gate_entropy_mod,
        },
        "gate_shape": {
            "all_gates": {"before": _gate_dist(gates), "after": _gate_dist(gates_mod)},
            "coord_signed_mean_per_dim": [float(x) for x in coord_signed_mean_per_dim],
            "coord_delta_hist": coord_delta_hist, "coord_abs_hist": coord_abs_hist,
            "transition_stats": {k: _agg(v) for k, v in pts.items()},
            "gate_value_hist": {"before": _hist(to(gates.reshape(-1)), 0.0, 1.0),
                                "after": _hist(to(gates_mod.reshape(-1)), 0.0, 1.0)},
            "entropy_hist": {"before": _hist(pts["before_entropy"], 0.0, 1.0),
                             "after": _hist(pts["after_entropy"], 0.0, 1.0)},
            "meangate_hist": {"before": _hist(pts["before_mean"], 0.0, 1.0),
                              "after": _hist(pts["after_mean"], 0.0, 1.0)},
        },
        "seq_em": to(seq_em.float()),
        "by_final_op": rj._by_final_op(cp, op, reg_a, reg_b, step_targets, targets, mask),
    }


# ---------------------------------------------------------------------------
# Per-config evaluation + primary/strata/rescue accounting
# ---------------------------------------------------------------------------

def _primary_stats(td: Dict[str, Any], base_primary: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": int(base_primary.sum())}
    if base_primary.sum() == 0:
        return out
    for key in ("m_prev", "m_cand", "m_commit", "dm_cand", "dm_commit", "rho", "margin_shortfall",
                "alpha_full", "alpha_margin", "gw_pos", "gw_neg", "gw_all", "gw_mag",
                "competitor_changed", "f_mean", "f_scaled_mean"):
        v = td[key][base_primary]
        v = v[np.isfinite(v)]
        out[key + "_median"] = float(np.median(v)) if v.size else None
        out[key + "_mean"] = float(v.mean()) if v.size else None
    out["comm_acc"] = float(td["comm_ok"][base_primary].mean())
    out["cand_acc"] = float(td["cand_ok"][base_primary].mean())
    return out


def _rescue_accounting(td: Dict[str, Any], base_td: Dict[str, Any], base_primary: np.ndarray,
                       strata: Dict[str, np.ndarray], sufficient: np.ndarray) -> Dict[str, Any]:
    """Rescue/regression among baseline candidate-correct general ADDs (transition level)."""
    base_comm = base_td["comm_ok"][base_primary] > 0.5
    new_comm = td["comm_ok"][base_primary] > 0.5
    n_succ = int(base_comm.sum()); n_fail = int((~base_comm).sum())
    rescued = int((~base_comm & new_comm).sum())
    regressed = int((base_comm & ~new_comm).sum())
    rec: Dict[str, Any] = {
        "n_base_succ": n_succ, "n_base_fail": n_fail,
        "rescued": rescued, "regressed": regressed,
        "rescue_rate": rescued / n_fail if n_fail else None,
        "regression_rate": regressed / n_succ if n_succ else None,
        "net_recovery": rescued - regressed,
        "net_recovery_rate": (rescued - regressed) / max(n_succ + n_fail, 1),
    }
    by = {}
    for name, mask in strata.items():
        if name == "edges":
            continue
        m = mask[base_primary]
        b = base_comm[m]; nn = new_comm[m]
        by[name] = {
            "n": int(m.sum()), "n_base_fail": int((~b).sum()),
            "rescued": int((~b & nn).sum()), "regressed": int((b & ~nn).sum()),
            "rescue_rate": float((~b & nn).sum() / max((~b).sum(), 1)),
            "regression_rate": float((b & ~nn).sum() / max(b.sum(), 1)),
            "comm_acc": float(nn.mean()) if m.any() else None,
        }
    rec["by_stratum"] = by
    # off-path subset: baseline failures already delivering >= scalar-equivalent transfer
    suff_sub = sufficient[base_primary] if sufficient.shape[0] == base_primary.shape[0] else np.zeros(base_comm.shape, bool)
    if suff_sub.any():
        b = base_comm[suff_sub]; nn = new_comm[suff_sub]
        comp = td["competitor_changed"][base_primary][suff_sub]
        dm_base = base_td["dm_commit"][base_primary][suff_sub]
        dm_new = td["dm_commit"][base_primary][suff_sub]
        fail_sub = ~b
        rec["offpath"] = {
            "n_sufficient": int(suff_sub.sum()),
            "n_base_fail_sufficient": int(fail_sub.sum()),
            "rescued_sufficient": int((fail_sub & nn).sum()),
            "rescue_rate_sufficient": float((fail_sub & nn).sum() / max(fail_sub.sum(), 1)),
            "competitor_changed_rate_sufficient_fail": float(comp[fail_sub].mean()) if fail_sub.any() else None,
            "dm_commit_mean_sufficient_fail": float(dm_new[fail_sub].mean()) if fail_sub.any() else None,
            "dm_commit_mean_sufficient_fail_baseline": float(dm_base[fail_sub].mean()) if fail_sub.any() else None,
        }
    return rec


def _seq_rescue(td: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    return rj._seq_rescue(td, base)


def evaluate_config(model: torch.nn.Module, datasets: Dict[int, Any], builders: Dict[str, Any],
                    commit_fn: Callable) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rec: Dict[str, Any] = {"horizons": {}, "controlled": rj.controlled_metrics(model, builders, commit_fn)}
    for L in HORIZONS:
        rec["horizons"][str(L)] = rj.horizon_metrics(model, datasets[L], commit_fn)
    td = test_transition_data(model, datasets[10], commit_fn)
    rec["test"] = td["scalar"]
    rec["by_final_op"] = td["by_final_op"]
    rec["gate_shape"] = td["gate_shape"]
    rec["rollout_time_s"] = time.perf_counter() - t0
    rec["_td"] = td
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_datasets(seed: int) -> Dict[int, Any]:
    return {L: gr.horizon_test_set(seed, L) for L in HORIZONS}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment K runner (interventional, frozen)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--out", default="experiments/experiment_k_records.json")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--kappas", nargs="*", type=float, default=None, help="debug only; must be a subset of KAPPA_GRID")
    args = ap.parse_args()

    seeds = args.seeds or gr.SEEDS
    models = args.models or list(MODEL_PLAN)
    try:
        kappas = validate_kappas(args.kappas or list(KAPPA_GRID))
    except ValueError as exc:
        raise SystemExit(str(exc))

    t0 = time.perf_counter()
    builders = gr._build_controlled_add_sequences()
    records: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"preregistration": PREREGISTRATION, "seeds": seeds,
                            "checkpoints": {}, "families_per_model": MODEL_PLAN,
                            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    ds_cache: Dict[int, Dict[int, Any]] = {}

    for model_key in models:
        meta["checkpoints"][model_key] = {}
        for seed in seeds:
            loaded = eg.load_checkpoint(args.checkpoints_dir, model_key, seed)
            sha_before = loaded["sha256"]
            g_sha = rj._g_sha(model_key, seed)
            if g_sha is not None and sha_before != g_sha:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint hash changed vs Experiment G")
            model = loaded["model"]
            if seed not in ds_cache:
                ds_cache[seed] = build_datasets(seed)
            datasets = ds_cache[seed]

            # baseline (kappa = 1.0) trajectory & fixed baseline sets
            baseline = evaluate_config(model, datasets, builders, make_commit_fn(PRIMARY_FAMILY, BASELINE_KAPPA))
            btd = baseline.pop("_td")
            repro = rj._verify_baseline_reproduction(model, datasets[10])
            base_comm_ok = btd["comm_ok"]
            base_primary = (btd["gen_general"] > 0.5) & (btd["cand_ok"] > 0.5)
            strata = rj._cand_margin_strata(btd["m_cand"], base_primary)
            suff = base_primary & np.isfinite(btd["alpha_full"]) & np.isfinite(btd["rho"]) & (btd["rho"] >= btd["alpha_full"])
            base_seq = btd["seq_em"]
            base_for_rescue = {"comm_ok": base_comm_ok, "seq_em": base_seq}

            for kappa in kappas:
                if kappa == BASELINE_KAPPA:
                    cfg = dict(baseline)
                    td = btd
                else:
                    cfg = evaluate_config(model, datasets, builders, make_commit_fn(PRIMARY_FAMILY, kappa))
                    td = cfg.pop("_td")
                cfg.update({
                    "model": model_key, "seed": seed, "family": PRIMARY_FAMILY, "kappa": kappa,
                    "cand_margin_strata_edges": [float(x) for x in strata["edges"]],
                    "primary": _primary_stats(td, base_primary),
                    "rescue": _rescue_accounting(td, btd, base_primary, strata, suff),
                    "seq_rescue": _seq_rescue(td, base_for_rescue),
                    "baseline_reproduction": repro if kappa == BASELINE_KAPPA else None,
                    "param_count": int(sum(p.numel() for p in model.parameters())),
                    "trainable_param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                })
                cfg.pop("_td", None)
                records.append(cfg)
            sha_after = rj.sha256_file(loaded["path"])
            if sha_after != sha_before:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint changed during run")
            meta["checkpoints"][model_key][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {model_key} seed={seed}: {len(records)} records so far "
                  f"(base T10 EM={baseline['horizons']['10']['em']:.4f}, repro={repro['passed']})")

    meta["n_records"] = len(records)
    meta["wall_clock_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=2)
    print(f"Wrote {args.out}: {len(records)} records in {meta['wall_clock_s']:.1f}s")


if __name__ == "__main__":
    main()
