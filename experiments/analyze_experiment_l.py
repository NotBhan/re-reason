#!/usr/bin/env python3
"""Experiment L — readout-side localization of the off-path component (analysis, measurement only).

Reads experiments/experiment_l_readout.npz (collector output), performs the ten required
analyses on the frozen linear readout, assigns each primary off-path failure to one of four
deterministic measurement categories, applies the pre-registered localization decision, and
produces the required figures.

Everything here is measurement only: no training, no model/readout/gate/commit modification, no
intervention, no recursive counterfactual trajectories. Bias-removal and perturbation
counterfactuals are one-step analysis-only computations on frozen states.

Statistical discipline: the five frozen E checkpoints are the independent units; F is a
secondary control, never pooled with E. Transition-level tables are descriptive; seed-level
summaries (mean +/- SD over 5 seeds) and paired E-vs-F comparisons carry the inference.

Pre-registered category rules (fixed before the formal analysis):
  bias_sig   = the bias-removed readout argmax at S_commit equals the true value
  moved      = <w_y - w_{a_prev}, S_commit - S_prev> > 0     (intended-direction movement)
  cleared    = m(S_commit; a_prev) >= 0                      (intended competitor cleared)
  changed    = the committed strongest competitor != the S_prev strongest competitor
  READOUT (B)      = bias_sig
  PLACEMENT (C)    = not bias_sig and cleared and changed
  SHORTFALL (A)    = not bias_sig and not PLACEMENT
  MIXED (D)        = non-finite inputs
Pre-registered localization decision (three-way, converging evidence):
  r_bias  = share of off-path failures with bias_sig
  r_place = share with PLACEMENT
  A readout/calibration if r_bias >= 0.30 and r_place < 0.60
  B latent placement    if r_place >= 0.60 and r_bias < 0.30
  C both                if r_bias >= 0.30 and r_place >= 0.60
  D inconclusive        otherwise
The 0.30 / 0.60 bars are declared conventions (stated before the aggregate was inspected), not
data-tuned cutoffs; the raw shares are reported so the verdict can be judged independently.

Reproduce:
    python experiments/analyze_experiment_l.py \
        --readout experiments/experiment_l_readout.npz \
        --output experiments/experiment_l_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_l
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRIMARY_MODEL = "E_candidate"
CONTROL_MODEL = "F_combined"
SEEDS = [42, 43, 44, 45, 46]
READOUT_BAR = 0.30
PLACEMENT_BAR = 0.60
PERTURB_EPS = [0.01, 0.05, 0.10]
PERTURB_DIRS = 32


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

class LTable:
    def __init__(self, scalars: np.ndarray, cols: List[str]) -> None:
        self.scalars = scalars.astype(np.float64)
        self.index = {c: i for i, c in enumerate(cols)}

    def __getitem__(self, name: str) -> np.ndarray:
        return self.scalars[:, self.index[name]]

    def __len__(self) -> int:
        return self.scalars.shape[0]


def load_readout(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    cols = meta["cols"]
    out: Dict[str, Any] = {"meta": meta, "cols": cols}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        out[m] = {
            "table": LTable(z[f"{m}__scalars"], cols),
            "W": z[f"{m}__W"], "b": z[f"{m}__b"],
            "offpath_seed": z[f"{m}__offpath_seed"],
            "offpath": {k: z[f"{m}__offpath_{k}"] for k in meta["off_vec_keys"]},
        }
    return out


def seed_masks(t: LTable) -> Dict[int, np.ndarray]:
    """Boolean row masks per seed (rows are concatenated in SEEDS order)."""
    seed_col = t["seed"]
    return {s: (np.round(seed_col).astype(int) == s) for s in SEEDS}


def primary_mask(t: LTable) -> np.ndarray:
    gen = (t["dest"] != t["src_reg"]) & (t["operand_b"] > 0)
    return (t["cand_ok"] > 0.5) & gen


def offpath_mask(t: LTable) -> np.ndarray:
    prim = primary_mask(t)
    return (prim & (t["comm_ok"] < 0.5)
            & np.isfinite(t["rho_margin"]) & (t["rho_margin"] >= t["alpha_star_full"]))


# ---------------------------------------------------------------------------
# per-transition signatures + categories
# ---------------------------------------------------------------------------

def signatures(t: LTable, mask: np.ndarray) -> Dict[str, np.ndarray]:
    bias_sig = t["nobias_comm_ok"][mask] > 0.5
    moved = t["dot_av_commit"][mask] > 0.0
    cleared = t["m_commit"][mask] >= 0.0
    changed = t["a_star_commit"][mask] != t["a_star_prev"][mask]
    dk_ac = t["dot_ac_commit"][mask]
    dc_ac = t["dot_ac_cand"][mask]
    full_deliv = (dc_ac > 1e-6) & (dk_ac >= dc_ac)
    finite = np.isfinite(dk_ac) & np.isfinite(dc_ac) & np.isfinite(t["m_commit"][mask])
    return {"bias_sig": bias_sig, "moved": moved, "cleared": cleared, "changed": changed,
            "full_deliv": full_deliv, "finite": finite}


def categorize(sig: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Deterministic per-transition categories: readout_calibration / latent_misplacement /
    boundary_shortfall / mixed_unresolved."""
    n = len(sig["bias_sig"])
    place = (~sig["bias_sig"]) & (sig["cleared"] & sig["changed"])
    short = (~sig["bias_sig"]) & (~place)
    mixed = ~sig["finite"]
    readout = sig["bias_sig"] & (~place)
    cat = np.full(n, "mixed_unresolved", dtype=object)
    cat[short] = "boundary_shortfall"
    cat[place] = "latent_misplacement"
    cat[readout] = "readout_calibration"
    return {"readout_calibration": cat == "readout_calibration",
            "latent_misplacement": cat == "latent_misplacement",
            "boundary_shortfall": cat == "boundary_shortfall",
            "mixed_unresolved": cat == "mixed_unresolved",
            "place_sig": place, "short_sig": short, "readout_sig": readout}


def decide(sig: Dict[str, np.ndarray], cat: Dict[str, np.ndarray]) -> Dict[str, Any]:
    r_bias = float(sig["bias_sig"].mean()) if len(sig["bias_sig"]) else float("nan")
    r_place = float(cat["place_sig"].mean()) if len(cat["place_sig"]) else float("nan")
    readout_ev = (not np.isnan(r_bias)) and r_bias >= READOUT_BAR
    place_ev = (not np.isnan(r_place)) and r_place >= PLACEMENT_BAR
    if readout_ev and place_ev:
        verdict = "C_both"
    elif place_ev:
        verdict = "B_latent_placement"
    elif readout_ev:
        verdict = "A_readout_calibration"
    else:
        verdict = "D_inconclusive"
    return {
        "verdict": verdict, "r_bias": r_bias, "r_place": r_place,
        "readout_evidence": readout_ev, "placement_evidence": place_ev,
        "category_fractions": {k: float(v.mean()) if len(v) else None for k, v in cat.items()
                               if k in ("readout_calibration", "latent_misplacement",
                                        "boundary_shortfall", "mixed_unresolved")},
        "rule_note": (f"Pre-registered: readout evidence if r_bias >= {READOUT_BAR}; placement "
                      f"evidence if r_place >= {PLACEMENT_BAR}; both -> C, placement only -> B, "
                      "readout only -> A, neither -> D."),
    }


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def agg_over_seeds(t: LTable, per_seed_fn) -> Dict[str, Any]:
    """Apply a per-seed function (returns dict of scalars) and summarize mean +/- SD."""
    rows: Dict[str, List[float]] = {}
    for s in SEEDS:
        m = seed_masks(t)[s]
        vals = per_seed_fn(t, m)
        for k, v in vals.items():
            rows.setdefault(k, []).append(float(v) if v is not None else np.nan)
    out: Dict[str, Any] = {}
    for k, v in rows.items():
        arr = np.asarray(v, float)
        out[k] = {"mean": float(np.nanmean(arr)), "sd": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
                  "per_seed": [None if np.isnan(x) else float(x) for x in arr]}
    return out


def paired(a: Dict[str, Any], b: Dict[str, Any], key: str) -> Dict[str, Any]:
    x = np.array([np.nan if v is None else v for v in a[key]["per_seed"]], float)
    y = np.array([np.nan if v is None else v for v in b[key]["per_seed"]], float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return {}
    d = x[ok] - y[ok]
    sd = d.std(ddof=1)
    out = {"E_mean": float(x[ok].mean()), "F_mean": float(y[ok].mean()),
           "mean_diff": float(d.mean()), "sign_positive": int((d > 0).sum()),
           "sign_negative": int((d < 0).sum()),
           "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"), "n": int(ok.sum())}
    try:
        out["paired_t_p"] = float(st.ttest_rel(x[ok], y[ok]).pvalue)
    except Exception:
        out["paired_t_p"] = None
    return out


# ---------------------------------------------------------------------------
# the ten analyses
# ---------------------------------------------------------------------------

def analyze_model(model_key: str, d: Dict[str, Any]) -> Dict[str, Any]:
    t: LTable = d["table"]
    off = offpath_mask(t)
    prim = primary_mask(t)
    succ = prim & (t["comm_ok"] > 0.5)
    fail = prim & (t["comm_ok"] < 0.5)
    nonoff = fail & (~off)
    sig = signatures(t, off)
    cat = categorize(sig)
    decision = decide(sig, cat)

    res: Dict[str, Any] = {"n_primary": int(prim.sum()), "n_offpath": int(off.sum()),
                           "n_primary_succ": int(succ.sum()), "n_nonoff_fail": int(nonoff.sum()),
                           "decision": decision}

    def med(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else float("nan")

    def mean(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else float("nan")

    # Analysis 1 — logit / margin decomposition (off-path failures)
    res["analysis_1_margins"] = {
        "m_prev": med(t["m_prev"][off]), "m_cand": med(t["m_cand"][off]), "m_commit": med(t["m_commit"][off]),
        "m_commit_vs_committed_competitor": med(t["m_commit_alt"][off]),
        "dm_cand": med(t["dm_cand"][off]), "dm_commit": med(t["dm_commit"][off]),
        "rho_margin": med(t["rho_margin"][off]),
        "alpha_star_full": med(t["alpha_star_full"][off]),
    }
    # Analysis 2 — bias vs state contribution
    bias_share = np.abs(t["bias_ac"][off]) / (np.abs(t["bias_ac"][off]) + np.abs(t["state_ac_commit"][off]) + 1e-9)
    bias_share_av = np.abs(t["bias_av"][off]) / (np.abs(t["bias_av"][off]) + np.abs(t["state_av_commit"][off]) + 1e-9)
    res["analysis_2_bias_vs_state"] = {
        "bias_av": med(t["bias_av"][off]), "state_av_commit": med(t["state_av_commit"][off]),
        "bias_ac": med(t["bias_ac"][off]), "state_ac_commit": med(t["state_ac_commit"][off]),
        "bias_share_ac_median": med(bias_share), "bias_share_ac_p95": float(np.percentile(bias_share, 95)),
        "bias_share_av_median": med(bias_share_av),
        "state_ac_commit_negative_rate": mean((t["state_ac_commit"][off] < 0).astype(float)),
    }
    # Analysis 3 — readout-calibration counterfactuals
    nb_flip = (t["nobias_comm_ok"][off] > 0.5)
    res["analysis_3_counterfactuals"] = {
        "actual_offpath_committed_acc": 0.0,
        "nobias_flip_to_correct_rate": mean(nb_flip.astype(float)),
        "nobias_offpath_committed_acc": mean((t["nobias_comm_pred"][off] == t["y_true"][off]).astype(float)),
        "nobias_decision_change_rate_offpath": mean((t["nobias_comm_pred"][off] != t["comm_pred"][off]).astype(float)),
        "nobias_decision_change_rate_primary": mean((t["nobias_comm_pred"][prim] != t["comm_pred"][prim]).astype(float)),
    }
    # Analysis 4 — pairwise decision geometry
    res["analysis_4_geometry"] = {
        "dot_av_commit": med(t["dot_av_commit"][off]), "dot_av_cand": med(t["dot_av_cand"][off]),
        "dot_ac_commit": med(t["dot_ac_commit"][off]), "dot_ac_cand": med(t["dot_ac_cand"][off]),
        "moved_intended_rate": mean((t["dot_av_commit"][off] > 0).astype(float)),
        "moved_vs_committed_competitor_rate": mean((t["dot_ac_commit"][off] > 0).astype(float)),
        "competitor_changed_offpath": mean((t["a_star_commit"][off] != t["a_star_prev"][off]).astype(float)),
        "competitor_changed_success": mean((t["a_star_commit"][succ] != t["a_star_prev"][succ]).astype(float)),
    }
    # Analysis 5 — state-projection error
    res["analysis_5_projection"] = {
        "offpath_dist_off": mean(t["offpath_dist"][off]), "offpath_dist_succ": mean(t["offpath_dist"][succ]),
        "offpath_ratio_off": med(t["offpath_dist"][off] / np.clip(t["d_cs"][off], 1e-9, None)),
        "offpath_ratio_succ": med(t["offpath_dist"][succ] / np.clip(t["d_cs"][succ], 1e-9, None)),
        "lambda_eff_off": med(t["lambda_eff"][off]), "lambda_eff_succ": med(t["lambda_eff"][succ]),
        "d_cs_off": med(t["d_cs"][off]), "d_cs_succ": med(t["d_cs"][succ]),
    }
    # Analysis 6 — decision-hyperplane distance
    res["analysis_6_hyperplane"] = {
        "hyper_av_prev": med(t["hyper_av_prev"][off]), "hyper_av_cand": med(t["hyper_av_cand"][off]),
        "hyper_av_commit": med(t["hyper_av_commit"][off]),
        "hyper_ac_commit": med(t["hyper_ac_commit"][off]),
        "cleared_intended_rate": mean((t["m_commit"][off] >= 0).astype(float)),
        "cleared_intended_rate_nonoff_fail": mean((t["m_commit"][nonoff] >= 0).astype(float)),
        "cleared_intended_rate_succ": mean((t["m_commit"][succ] >= 0).astype(float)),
    }
    # Analysis 8 — candidate vs commit geometry (offpath failure vs success)
    res["analysis_8_candidate_vs_commit"] = {
        "offpath_m_cand": med(t["m_cand"][off]), "offpath_m_commit": med(t["m_commit"][off]),
        "offpath_margin_loss": med(t["margin_shortfall"][off]),
        "succ_m_cand": med(t["m_cand"][succ]), "succ_m_commit": med(t["m_commit"][succ]),
        "succ_margin_loss": med(t["margin_shortfall"][succ]),
        "nonoff_m_cand": med(t["m_cand"][nonoff]), "nonoff_m_commit": med(t["m_commit"][nonoff]),
    }
    # Analysis 9 — class / value effects (descriptive)
    yv = t["y_true"][off].astype(int)
    res["analysis_9_classes"] = {
        "wrap_rate": mean(t["wrap"][off]),
        "wrong_decode_third_rate": mean((t["comm_pred"][off] != t["operand_a"][off]).astype(float)),
        "dest_hist": np.bincount(t["dest"][off].astype(int), minlength=4).tolist(),
        "y_hist": np.bincount(yv, minlength=10).tolist(),
        "committed_pred_hist": np.bincount(t["comm_pred"][off].astype(int), minlength=10).tolist(),
        "operand_b_hist": np.bincount(t["operand_b"][off].astype(int), minlength=10).tolist(),
    }

    # Analysis 7 — local perturbation stability (analysis only, deterministic)
    res["analysis_7_perturbation"] = perturbation_stability(d)

    # seed-level summaries of the key signatures (independent-unit inference)
    def per_seed(t2: LTable, m: np.ndarray) -> Dict[str, float]:
        off = offpath_mask(t2) & m
        if off.sum() == 0:
            return {"offpath_rate": None, "r_bias": None, "r_place": None, "r_short": None,
                    "competitor_changed": None, "cleared_intended": None, "third_basin": None,
                    "bias_share_ac": None, "offpath_ratio": None, "hyper_ac_commit": None}
        s = signatures(t2, off)
        c = categorize(s)
        return {
            "offpath_rate": float(off.sum() / max(m.sum(), 1)),
            "r_bias": float(s["bias_sig"].mean()),
            "r_place": float(c["place_sig"].mean()),
            "r_short": float(c["short_sig"].mean()),
            "competitor_changed": float(s["changed"].mean()),
            "cleared_intended": float(s["cleared"].mean()),
            "third_basin": float((t2["comm_pred"][off] != t2["operand_a"][off]).mean()),
            "bias_share_ac": float(np.median(np.abs(t2["bias_ac"][off]) / (np.abs(t2["bias_ac"][off]) + np.abs(t2["state_ac_commit"][off]) + 1e-9))),
            "offpath_ratio": float(np.median(t2["offpath_dist"][off] / np.clip(t2["d_cs"][off], 1e-9, None))),
            "hyper_ac_commit": float(np.median(t2["hyper_ac_commit"][off])),
        }
    res["seed_level"] = agg_over_seeds(t, per_seed)
    return res


def perturbation_stability(d: Dict[str, Any]) -> Dict[str, Any]:
    """Analysis 7: isotropic deterministic perturbations around S_commit (off-path only)."""
    t: LTable = d["table"]
    off = offpath_mask(t)
    idx = np.nonzero(off)[0]
    S_commit = d["offpath"]["S_commit"]
    C = d["offpath"]["C"]
    S_prev = d["offpath"]["S_prev"]
    seeds = d["offpath_seed"]
    W, b = d["W"], d["b"]
    if len(idx) == 0:
        return {"n": 0, "per_eps": {}}
    dest = t["dest"][idx].astype(int)
    y = t["y_true"][idx].astype(int)
    comm = t["comm_pred"][idx].astype(int)
    scale = np.linalg.norm(C - S_prev, axis=1)
    seed_order = {s: i for i, s in enumerate(SEEDS)}
    per_eps: Dict[str, Any] = {}
    for eps in PERTURB_EPS:
        stab, corr = [], []
        for j in range(len(idx)):
            Wd = W[seed_order[int(seeds[j])], dest[j]]        # (V, D)
            bd = b[seed_order[int(seeds[j])], dest[j]]        # (V,)
            rng = np.random.default_rng(100003 * int(seeds[j]) + int(t["seq_id"][idx[j]]) * 37 + int(t["t"][idx[j]]))
            dirs = rng.standard_normal((PERTURB_DIRS, S_commit.shape[1]))
            dirs /= np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)
            pert = S_commit[j][None, :] + eps * float(scale[j]) * dirs
            logits = pert @ Wd.T + bd
            dec = logits.argmax(1)
            stab.append(float((dec == comm[j]).mean()))
            corr.append(float((dec == y[j]).mean()))
        per_eps[f"{eps}"] = {
            "stability_mean": float(np.mean(stab)) if stab else None,
            "stability_median": float(np.median(stab)) if stab else None,
            "perturb_correct_rate": float(np.mean(corr)) if corr else None,
        }
    # fraction of off-path failures where the unperturbed state is stable under ALL eps/dirs
    return {"n": int(len(idx)), "per_eps": per_eps}


# ---------------------------------------------------------------------------
# cross-experiment alignment (G / H / I)
# ---------------------------------------------------------------------------

def align_against_experiments(d: Dict[str, Any], i_path: str, h_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    shared_i = ["seed", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b",
                "cand_ok", "comm_ok", "a_star_prev", "m_prev", "m_cand", "m_commit",
                "dm_cand", "dm_commit", "rho_margin", "alpha_star_full", "alpha_star_margin",
                "f_mean", "d_cs", "d_ss", "d_sc"]
    shared_h = ["seed", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b", "f_mean",
                "lambda_eff", "offpath_dist"]
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        t: LTable = d[m]["table"]
        src = np.round(t["source"]).astype(int)
        test_rows = src == 0
        res: Dict[str, Any] = {}
        if os.path.exists(i_path):
            zi = np.load(i_path, allow_pickle=False)
            mi = json.loads(str(zi["meta_json"][0]))
            ic = {c: k for k, c in enumerate(mi["cols"])}
            si = zi[f"{m}__scalars"].astype(np.float64)
            src_i = np.round(si[:, ic["source"]]).astype(int)
            si_test = si[src_i == 0]
            md = {}
            for col in shared_i:
                if col in ic and col in t.index:
                    a = t[col][test_rows]; b_ = si_test[:, ic[col]]
                    if a.shape[0] == b_.shape[0]:
                        md[col] = float(np.nanmax(np.abs(a - b_)))
            res["vs_I_max_abs_diff"] = md
            res["vs_I_ok"] = all(v < 1e-3 for k, v in md.items() if k != "f_mean") and md.get("f_mean", 0) < 1e-4
        if os.path.exists(h_path):
            zh = np.load(h_path, allow_pickle=False)
            mh = json.loads(str(zh["meta_json"][0]))
            hc = {c: k for k, c in enumerate(mh["scalar_cols"])}
            sh = zh[f"{m}__scalars"].astype(np.float64)
            src_h = np.round(sh[:, hc["source"]]).astype(int)
            sh_test = sh[src_h == 0]
            md = {}
            for col in shared_h:
                if col in hc and col in t.index:
                    a = t[col][test_rows]; b_ = sh_test[:, hc[col]]
                    if a.shape[0] == b_.shape[0]:
                        md[col] = float(np.nanmax(np.abs(a - b_)))
            res["vs_H_max_abs_diff"] = md
            res["vs_H_ok"] = all(v < 1e-3 for k, v in md.items() if k != "f_mean") and md.get("f_mean", 0) < 1e-4
        out[m] = res
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d: Dict[str, Any], results: Dict[str, Any], out_dir: str, prefix="experiment_l") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    tE: LTable = d[PRIMARY_MODEL]["table"]
    tF: LTable = d[CONTROL_MODEL]["table"]
    offE = offpath_mask(tE); offF = offpath_mask(tF)
    primE = primary_mask(tE); succE = primE & (tE["comm_ok"] > 0.5)

    def cdf(x, lo, hi, bins=40):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return np.histogram(x, bins=bins, range=(lo, hi))[0]

    # 1. state vs readout (bias) decomposition for off-path failures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, t, off, name in ((axes[0], tE, offE, "E"), (axes[1], tF, offF, "F")):
        ax.scatter(t["state_ac_commit"][off], t["bias_ac"][off], s=6, alpha=0.4)
        ax.axhline(0, color="k", lw=0.7); ax.axvline(0, color="k", lw=0.7)
        ax.set_xlabel("state margin (w_y-w_a)·S_commit"); ax.set_ylabel("bias margin b_y-b_a")
        ax.set_title(f"{name}: off-path state vs bias (pair vs committed competitor)", fontsize=9)
        ax.grid(alpha=0.3)
    save(fig, "decomposition")

    # 2. pairwise margin progression S_prev -> C -> S_commit
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    stages = ["m_prev", "m_cand", "m_commit"]
    lab = ["S_prev", "C (candidate)", "S_commit"]
    for t, off, name, c in ((tE, offE, "E", "tab:blue"), (tF, offF, "F", "tab:orange")):
        med = [np.median(t[s][off]) for s in stages]
        ax.plot(lab, med, marker="o", label=f"{name} off-path (vs a_prev)", color=c)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("median margin m = z_y - z_a_prev"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Pairwise margin progression on off-path failures")
    save(fig, "margin_progression")

    # 3. bias contribution vs state contribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    share = np.abs(tE["bias_ac"][offE]) / (np.abs(tE["bias_ac"][offE]) + np.abs(tE["state_ac_commit"][offE]) + 1e-9)
    axes[0].hist(share, bins=40, range=(0, 1), color="tab:purple", alpha=0.8)
    axes[0].axvline(np.median(share), color="k", ls="--", label=f"median {np.median(share):.3f}")
    axes[0].set_xlabel("|bias| / (|bias|+|state|) for (y, committed competitor)")
    axes[0].set_ylabel("off-path count"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title("bias share of margin (E)", fontsize=9)
    axes[1].hist(tE["state_ac_commit"][offE], bins=40, color="tab:blue", alpha=0.7, label="state margin")
    axes[1].hist(tE["bias_ac"][offE], bins=40, color="tab:red", alpha=0.7, label="bias margin")
    axes[1].axvline(0, color="k", lw=0.8); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_title("state vs bias margin (E)", fontsize=9)
    fig.suptitle("Bias versus state contribution on off-path failures")
    save(fig, "bias_vs_state")

    # 4. off-path distance success vs failure
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    bins = np.linspace(0, 40, 41)
    ax.hist(tE["offpath_dist"][succE], bins=bins, density=True, alpha=0.6, label="success", color="tab:green")
    ax.hist(tE["offpath_dist"][offE], bins=bins, density=True, alpha=0.6, label="off-path failure", color="tab:red")
    ax.set_xlabel("off-path distance ||S_commit - proj_{S_prev->C}||"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("Off-path distance: success vs off-path failure (E)")
    save(fig, "offpath_dist")

    # 5. decision-hyperplane distance
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for col, name, c in (("hyper_av_prev", "S_prev", "tab:gray"), ("hyper_av_cand", "C", "tab:blue"),
                         ("hyper_av_commit", "S_commit", "tab:red")):
        v = tE[col][offE]; v = v[np.isfinite(v)]
        ax.hist(v, bins=40, range=(-20, 20), alpha=0.5, label=f"{name} (vs a_prev)", color=c, density=True)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_xlabel("signed normalized hyperplane distance  margin/||w_y-w_a_prev||")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("Decision-hyperplane distance on off-path failures (E)")
    save(fig, "hyperplane_dist")

    # 6. competitor switching analysis
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    cats = ["off-path failure", "success", "non-off-path failure"]
    vals = [np.mean((tE["a_star_commit"][offE] != tE["a_star_prev"][offE])),
            np.mean((tE["a_star_commit"][succE] != tE["a_star_prev"][succE])),
            np.mean((tE["a_star_commit"][primE & (tE["comm_ok"] < 0.5) & (~offE)] != tE["a_star_prev"][primE & (tE["comm_ok"] < 0.5) & (~offE)]))]
    ax.bar(cats, vals, color=["tab:red", "tab:green", "tab:gray"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("competitor-changed rate"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Committed competitor change rate (E)")
    save(fig, "competitor_switch")

    # 7. candidate vs commit margin distributions
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    bins = np.linspace(-10, 25, 50)
    ax.hist(tE["m_cand"][offE], bins=bins, alpha=0.6, label="off-path candidate margin", color="tab:blue", density=True)
    ax.hist(tE["m_commit"][offE], bins=bins, alpha=0.6, label="off-path commit margin (vs a_prev)", color="tab:red", density=True)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_xlabel("margin"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Candidate vs commit margin on off-path failures (E)")
    save(fig, "margin_dist")

    # 8. latent projection / orthogonal residual
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.scatter(tE["lambda_eff"][succE], tE["offpath_dist"][succE], s=5, alpha=0.3, label="success", color="tab:green")
    ax.scatter(tE["lambda_eff"][offE], tE["offpath_dist"][offE], s=8, alpha=0.6, label="off-path failure", color="tab:red")
    ax.set_xlabel("λ_eff (projection onto S_prev→C)"); ax.set_ylabel("orthogonal residual ||r_perp||")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("Latent projection / orthogonal residual (E)")
    save(fig, "projection_residual")

    # 9. failure categories A/B/C/D
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    catE = results[PRIMARY_MODEL]["decision"]["category_fractions"]
    catF = results[CONTROL_MODEL]["decision"]["category_fractions"]
    keys = ["boundary_shortfall", "readout_calibration", "latent_misplacement", "mixed_unresolved"]
    xs = np.arange(len(keys)); w = 0.38
    ax.bar(xs - w / 2, [catE[k] for k in keys], w, label="E", color="tab:blue")
    ax.bar(xs + w / 2, [catF[k] for k in keys], w, label="F", color="tab:orange")
    ax.set_xticks(xs); ax.set_xticklabels(["A shortfall", "B readout", "C placement", "D mixed"], fontsize=8)
    ax.set_ylabel("fraction of off-path failures"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Off-path failure categories")
    save(fig, "categories")

    # 10. value/operand breakdown
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].bar(range(10), np.bincount(tE["comm_pred"][offE].astype(int), minlength=10), color="tab:red")
    axes[0].set_xlabel("committed (wrong) decoded value"); axes[0].set_ylabel("count")
    axes[0].set_title("committed (wrong) value dist (E off-path)", fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].bar(range(10), np.bincount(tE["operand_b"][offE].astype(int), minlength=10), color="tab:blue")
    axes[1].set_xlabel("source operand value"); axes[1].set_ylabel("count")
    axes[1].set_title("source operand dist (E off-path)", fontsize=9); axes[1].grid(alpha=0.3)
    fig.suptitle("Value/operand breakdown of off-path failures")
    save(fig, "value_breakdown")

    # 11. E vs F comparison
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    keys2 = ["r_bias", "r_place", "competitor_changed", "cleared_intended", "third_basin", "offpath_ratio"]
    lab2 = ["r_bias", "r_place", "competitor changed", "cleared intended", "third basin", "offpath ratio"]
    e = results[PRIMARY_MODEL]["seed_level"]; f = results[CONTROL_MODEL]["seed_level"]
    xs = np.arange(len(keys2)); w = 0.38
    ax.bar(xs - w / 2, [e[k]["mean"] for k in keys2], w, yerr=[e[k]["sd"] for k in keys2], capsize=3, label="E", color="tab:blue")
    ax.bar(xs + w / 2, [f[k]["mean"] for k in keys2], w, yerr=[f[k]["sd"] for k in keys2], capsize=3, label="F", color="tab:orange")
    ax.set_xticks(xs); ax.set_xticklabels(lab2, fontsize=7, rotation=15); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E vs F off-path signatures (mean ± SD over 5 seeds)")
    save(fig, "E_vs_F")

    # 12. perturbation stability
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    pe = results[PRIMARY_MODEL]["analysis_7_perturbation"]["per_eps"]
    eps = list(pe.keys())
    ax.errorbar(range(len(eps)), [pe[k]["stability_mean"] for k in eps],
                yerr=None, marker="o", label="decision stability (same decode)", color="tab:blue")
    ax.plot(range(len(eps)), [pe[k]["perturb_correct_rate"] for k in eps], marker="s",
            label="perturbed decode correct", color="tab:green")
    ax.set_xticks(range(len(eps))); ax.set_xticklabels([f"ε={k}" for k in eps])
    ax.set_ylabel("fraction"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Local perturbation stability around S_commit (E off-path)")
    save(fig, "perturbation")

    return paths


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build(d: Dict[str, Any], i_path: str, h_path: str) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": SEEDS, "primary_model": PRIMARY_MODEL, "control_model": CONTROL_MODEL,
            "checkpoints": d["meta"]["models"],
            "readout": d["meta"]["readout"],
            "offpath_definition": d["meta"]["offpath_definition"],
            "thresholds": {"readout_bar": READOUT_BAR, "placement_bar": PLACEMENT_BAR,
                           "perturb_eps": PERTURB_EPS, "perturb_dirs": PERTURB_DIRS},
            "statistical_discipline": ("Five frozen E checkpoints are independent units; F is a "
                                       "secondary control (not pooled). Transition-level tables are "
                                       "descriptive; bias-removal/perturbation counterfactuals are "
                                       "analysis-only one-step computations on frozen states."),
        },
    }
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        res[m] = analyze_model(m, d[m])
    res["E_vs_F_paired"] = {
        k: paired(res[PRIMARY_MODEL]["seed_level"], res[CONTROL_MODEL]["seed_level"], k)
        for k in ("r_bias", "r_place", "competitor_changed", "cleared_intended", "third_basin",
                  "bias_share_ac", "offpath_ratio", "hyper_ac_commit")
    }
    res["alignment"] = align_against_experiments(d, i_path, h_path)
    res["primary_decision"] = res[PRIMARY_MODEL]["decision"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment L analysis (measurement only)")
    ap.add_argument("--readout", default="experiments/experiment_l_readout.npz")
    ap.add_argument("--i-coordinate", default="experiments/experiment_i_coordinate.npz")
    ap.add_argument("--h-interp", default="experiments/experiment_h_interp.npz")
    ap.add_argument("--output", default="experiments/experiment_l_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_l")
    args = ap.parse_args()

    d = load_readout(args.readout)
    res = build(d, args.i_coordinate, args.h_interp)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    print("E verdict:", res[PRIMARY_MODEL]["decision"]["verdict"],
          f"(r_bias={res[PRIMARY_MODEL]['decision']['r_bias']:.3f}, r_place={res[PRIMARY_MODEL]['decision']['r_place']:.3f})")
    print("F verdict:", res[CONTROL_MODEL]["decision"]["verdict"])
    paths = make_figures(d, res, args.fig_dir, prefix=args.fig_prefix)
    print("Figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
