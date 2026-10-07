#!/usr/bin/env python3
"""Experiment M — Representation Geometry of the Third Basin (analysis, measurement only).

Reads experiments/experiment_m_geometry.npz (M collector) and answers the core questions:
  A. What is the third (wrong) value arithmetically?
  B. Is it a wrap/modular basin?
  C. Is the wrong register systematic?
  D. Is the off-axis displacement directionally consistent?
  E. Does the off-axis direction track arithmetic structure?
  F. How deep/broad is the third basin?
  + candidate-vs-commit geometry, controls (A off-path / B success / C non-off-path failure), E vs F.

Measurement only: no training, no weight/readout/gate/commit modification, no intervention, no
recursive counterfactuals, no learned classifier as primary evidence, no causal claim.

PRE-REGISTERED OUTCOME CLASSES (declared before inspecting aggregate results)
-----------------------------------------------------------------------------
modular_identity  = same-parity rate among off-path failures >= 0.60
                    AND permutation p <= 0.01   (wrong value preserves the target's parity)
wrap_specific     = TV(P(v_wrong|wrap), P(v_wrong|no-wrap)) >= 0.30
directional_systematic = off-axis mean-resultant R >= 0.15 AND permutation p <= 0.01
directional_weak  = permutation p <= 0.01 (reported as a sub-finding, below the magnitude bar)

  M-A systematic modular geometry      : modular_identity and directional_systematic
  M-B systematic non-modular placement : directional_systematic and not modular_identity
  M-C unsystematic representation collapse : not modular_identity and not directional_weak
  M-D mixed / inconclusive             : otherwise
The 0.60 / 0.01 / 0.30 / 0.15 bars are declared conventions, stated before the aggregate was
inspected, not data-tuned cutoffs; the raw quantities are always reported so the reader can
judge independently.

Reproduce:
    python experiments/analyze_experiment_m.py \
        --geometry experiments/experiment_m_geometry.npz \
        --output experiments/experiment_m_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_m
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
MODULUS = 10

# Pre-registered thresholds (conventions; see module docstring).
MODULAR_PARITY_BAR = 0.60
PERM_P_BAR = 0.01
WRAP_TV_BAR = 0.30
R_BAR = 0.15
N_PERM = 2000
PERM_SEED = 20261007

GROUP_NAMES = {0: "offpath_failure", 1: "candidate_correct_success", 2: "nonoffpath_failure"}


class MTable:
    def __init__(self, scalars: np.ndarray, cols: List[str]) -> None:
        self.scalars = scalars.astype(np.float64)
        self.index = {c: i for i, c in enumerate(cols)}

    def __getitem__(self, name: str) -> np.ndarray:
        return self.scalars[:, self.index[name]]

    def __len__(self) -> int:
        return self.scalars.shape[0]


def load_geometry(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    cols = meta["cols"]
    out: Dict[str, Any] = {"meta": meta, "cols": cols}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        out[m] = {
            "table": MTable(z[f"{m}__scalars"], cols),
            "primary_row": z[f"{m}__primary_row"].astype(np.int64),
            "primary_group": z[f"{m}__primary_group"].astype(np.int64),
            "primary_unit": z[f"{m}__primary_unit"].astype(np.float64),
            "fail_row": z[f"{m}__fail_row"].astype(np.int64),
            "fail_S_prev": z[f"{m}__fail_S_prev"].astype(np.float64),
            "fail_C": z[f"{m}__fail_C"].astype(np.float64),
            "fail_S_commit": z[f"{m}__fail_S_commit"].astype(np.float64),
        }
    return out


def group_rows(d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {GROUP_NAMES[int(g)]: d["primary_row"][d["primary_group"] == g] for g in (0, 1, 2)}


# ---------------------------------------------------------------------------
# directional consistency
# ---------------------------------------------------------------------------

def unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def directional_resultant(u: np.ndarray) -> float:
    """R = ||mean of unit directions|| (u already unit or normalized here)."""
    if len(u) == 0:
        return float("nan")
    return float(np.linalg.norm(unit_rows(u).mean(0)))


def permutation_null_R(u: np.ndarray, n_perm: int = N_PERM, seed: int = PERM_SEED) -> np.ndarray:
    """Sign-flip permutation null for R (valid under the isotropic-direction null)."""
    U = unit_rows(u)
    rng = np.random.default_rng(seed)
    n = len(U)
    out = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=n)
        out[i] = np.linalg.norm((U * s[:, None]).mean(0))
    return out


def mean_pairwise_cosine(u: np.ndarray, max_n: int = 400, seed: int = PERM_SEED) -> Dict[str, float]:
    U = unit_rows(u)
    if len(U) < 2:
        return {"mean": float("nan"), "median": float("nan"), "n": int(len(U))}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(U)) if len(U) <= max_n else rng.choice(len(U), size=max_n, replace=False)
    S = U[idx]
    C = S @ S.T
    iu = np.triu_indices(len(idx), 1)
    return {"mean": float(C[iu].mean()), "median": float(np.median(C[iu])), "n": int(len(idx))}


def permutation_null_parity(vw: np.ndarray, y: np.ndarray, n_perm: int = N_PERM,
                            seed: int = PERM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    out = np.empty(n_perm)
    for i in range(n_perm):
        out[i] = float((rng.permutation(vw) % 2 == y % 2).mean())
    return out


# ---------------------------------------------------------------------------
# core analyses
# ---------------------------------------------------------------------------

def analyze_model(model_key: str, d: Dict[str, Any]) -> Dict[str, Any]:
    t: MTable = d["table"]
    grp = group_rows(d)
    off = grp["offpath_failure"]; succ = grp["candidate_correct_success"]
    nonoff = grp["nonoffpath_failure"]
    c = lambda n: t[n]
    res: Dict[str, Any] = {"n_offpath": int(len(off)), "n_success": int(len(succ)),
                           "n_nonoff_failure": int(len(nonoff))}

    # --- Core A: third-value arithmetic mapping ---
    vw = c("comm_pred")[off].astype(int); y = c("y_true")[off].astype(int)
    oa = c("operand_a")[off].astype(int); ob = c("operand_b")[off].astype(int)
    offs = ((vw - y + MODULUS // 2) % MODULUS) - MODULUS // 2
    same_par = (offs % 2 == 0)
    null_par = permutation_null_parity(vw, y)
    p_par = float((null_par >= same_par.mean()).mean())
    ref_y = np.bincount(c("y_true")[grp["offpath_failure"]].astype(int), minlength=MODULUS).astype(float)
    obs_v = np.bincount(vw, minlength=MODULUS).astype(float)
    res["core_A_wrong_value"] = {
        "wrong_offset_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(offs, return_counts=True))},
        "abs_offset_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(np.abs(offs), return_counts=True))},
        "same_parity_rate": float(same_par.mean()),
        "same_parity_null_mean": float(null_par.mean()),
        "same_parity_null_p95": float(np.percentile(null_par, 95)),
        "same_parity_perm_p": p_par,
        "wrong_eq_old_rate": float((c("dmod_wrong_old")[off] == 0).mean()),
        "wrong_eq_operand_rate": float((c("dmod_wrong_operand")[off] == 0).mean()),
        "dmod_wrong_true_mean": float(c("dmod_wrong_true")[off].mean()),
        "dmod_wrong_old_mean": float(c("dmod_wrong_old")[off].mean()),
        "dmod_wrong_operand_mean": float(c("dmod_wrong_operand")[off].mean()),
        "tv_vs_reference_y": float(0.5 * np.abs(obs_v / obs_v.sum() - ref_y / ref_y.sum()).sum()),
        "tv_vs_uniform": float(0.5 * np.abs(obs_v / obs_v.sum() - 1.0 / MODULUS).sum()),
        "reference_y_distribution": ref_y.astype(int).tolist(),
        "observed_wrong_distribution": obs_v.astype(int).tolist(),
    }

    # --- Core B: wrap / no-wrap ---
    w = c("wrap")[off] > 0.5
    pw = np.bincount(vw[w], minlength=MODULUS) / max(w.sum(), 1)
    pn = np.bincount(vw[~w], minlength=MODULUS) / max((~w).sum(), 1)
    res["core_B_wrap"] = {
        "wrap_fraction": float(w.mean()),
        "P_vwrong_given_wrap": pw.tolist(), "P_vwrong_given_nowrap": pn.tolist(),
        "tv_wrap_vs_nowrap": float(0.5 * np.abs(pw - pn).sum()),
        "same_parity_wrap": float(same_par[w].mean()) if w.any() else None,
        "same_parity_nowrap": float(same_par[~w].mean()) if (~w).any() else None,
        "dmod_wrong_true_wrap": float(c("dmod_wrong_true")[off][w].mean()) if w.any() else None,
        "dmod_wrong_true_nowrap": float(c("dmod_wrong_true")[off][~w].mean()) if (~w).any() else None,
    }

    # --- Core C: wrong register (flat 40-way winner) ---
    fr = c("flat_reg_commit")[off].astype(int); dd = c("dest")[off].astype(int); sr = c("src_reg")[off].astype(int)
    frs = c("flat_reg_commit")[succ].astype(int); dds = c("dest")[succ].astype(int); srs = c("src_reg")[succ].astype(int)
    res["core_C_register"] = {
        "offpath_dest_src_neither": [float((fr == dd).mean()), float((fr == sr).mean()),
                                     float(((fr != dd) & (fr != sr)).mean())],
        "success_dest_src_neither": [float((frs == dds).mean()), float((frs == srs).mean()),
                                     float(((frs != dds) & (frs != srs)).mean())],
        "offpath_src_gt_dest": float((fr == sr).mean() - (fr == dd).mean()),
    }

    # --- Core D: off-axis decomposition + directional consistency ---
    res["core_D_offaxis"] = {}
    for name, sel in (("offpath", off), ("success", succ), ("nonoff", nonoff)):
        res["core_D_offaxis"][name] = {
            "off_par_norm_median": float(np.median(c("off_par_norm")[sel])),
            "off_orth_norm_median": float(np.median(c("off_orth_norm")[sel])),
            "off_cos_median": float(np.median(c("off_cos")[sel])),
            "off_ratio_median": float(np.median(c("off_ratio")[sel])),
        }
    u_off = d["primary_unit"][d["primary_group"] == 0]
    R = directional_resultant(u_off)
    nullR = permutation_null_R(u_off)
    pR = float((nullR >= R).mean())
    mp = mean_pairwise_cosine(u_off)
    res["core_D_offaxis"]["directional_consistency_offpath"] = {
        "R": R, "null_R_mean": float(nullR.mean()), "null_R_p95": float(np.percentile(nullR, 95)),
        "R_over_null": float(R / nullR.mean()) if nullR.mean() > 0 else float("nan"),
        "perm_p": pR, "mean_pairwise_cosine": mp["mean"], "median_pairwise_cosine": mp["median"],
        "n_units": int(len(u_off)), "n_pair_subsample": mp["n"],
    }
    u_all = d["primary_unit"]
    res["core_D_offaxis"]["directional_consistency_all_primary"] = {"R": directional_resultant(u_all)}
    # wrap/no-wrap subgroup consistency
    wu = c("wrap")[off] > 0.5
    res["core_D_offaxis"]["R_wrap"] = directional_resultant(u_off[wu])
    res["core_D_offaxis"]["R_nowrap"] = directional_resultant(u_off[~wu])

    # --- Core E: does off-axis direction track arithmetic? ---
    def cos_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        nb = np.linalg.norm(b)
        if nb < 1e-12:
            return np.full(len(a), np.nan)
        return (unit_rows(a) @ b) / nb
    mw = unit_rows(u_off)[wu].mean(0); mn = unit_rows(u_off)[~wu].mean(0)
    obn = c("operand_b")[off].astype(int)
    within = {int(k): directional_resultant(u_off[obn == k]) for k in np.unique(obn) if (obn == k).sum() >= 15}
    res["core_E_arithmetic_direction"] = {
        "cos_to_wrap_dir_wrap": float(np.nanmean(cos_to(u_off[wu], mw))),
        "cos_to_wrap_dir_nowrap": float(np.nanmean(cos_to(u_off[~wu], mw))),
        "cos_to_nowrap_dir_wrap": float(np.nanmean(cos_to(u_off[wu], mn))),
        "cos_to_nowrap_dir_nowrap": float(np.nanmean(cos_to(u_off[~wu], mn))),
        "R_within_operand_b": {str(k): v for k, v in within.items()},
        "offaxis_offset_correlation": float(np.corrcoef(offs, c("off_orth_norm")[off])[0, 1]),
    }

    # --- Core F: third-basin geometry / depth ---
    res["core_F_third_basin"] = {
        "mv_prev_off": float(np.median(c("mv_prev")[off])),
        "mv_cand_off": float(np.median(c("mv_cand")[off])),
        "mv_commit_off": float(np.median(c("mv_commit")[off])),
        "hyper_flat_commit_off": float(np.median(c("hyper_flat_commit")[off])),
        "hyper_flat_commit_succ": float(np.median(c("hyper_flat_commit")[succ])),
        "hyper_ac_commit_off": float(np.median(c("hyper_ac_commit")[off])),
        "m_commit_off": float(np.median(c("m_commit")[off])),
        "m_commit_succ": float(np.median(c("m_commit")[succ])),
    }

    # --- Core 8: candidate-vs-commit geometry ---
    res["core8_cand_vs_commit"] = {}
    for name, sel in (("offpath", off), ("success", succ), ("nonoff", nonoff)):
        res["core8_cand_vs_commit"][name] = {
            "m_cand": float(np.median(c("m_cand")[sel])), "m_commit": float(np.median(c("m_commit")[sel])),
            "margin_loss": float(np.median(c("m_cand")[sel] - c("m_commit")[sel])),
            "d_cs": float(np.median(c("d_cs")[sel])), "offpath_dist": float(np.median(c("offpath_dist")[sel])),
            "off_ratio": float(np.median(c("off_ratio")[sel])),
        }

    # --- per-seed summaries ---
    def per_seed(t2: MTable, s: int) -> Dict[str, float]:
        g2 = group_rows(d)
        o = g2["offpath_failure"]
        o = o[np.round(t2["seed"][o]).astype(int) == s]
        u = d["primary_unit"][(d["primary_group"] == 0) & (np.round(t2["seed"][d["primary_row"]]).astype(int) == s)]
        vws = c("comm_pred")[o].astype(int); ys = c("y_true")[o].astype(int)
        if len(o) == 0:
            return {"offpath_count": 0, "same_parity": None, "R": None, "mean_cos": None}
        return {
            "offpath_count": int(len(o)),
            "same_parity": float(((vws - ys) % 2 == 0).mean()),
            "R": directional_resultant(u),
            "mean_cos": float(mean_pairwise_cosine(u)["mean"]) if len(u) > 1 else None,
        }
    seed_rows = [per_seed(t, s) for s in SEEDS]
    res["per_seed"] = {k: [r[k] for r in seed_rows] for k in seed_rows[0]}

    # --- seed-level directional consistency (the seeds are the independent units) ---
    seed_dir = []
    for s in SEEDS:
        o = off[np.round(t["seed"][off]).astype(int) == s]
        u = d["primary_unit"][(d["primary_group"] == 0) &
                              (np.round(t["seed"][d["primary_row"]]).astype(int) == s)]
        if len(o) == 0:
            continue
        R_s = directional_resultant(u)
        null_s = permutation_null_R(u, n_perm=500, seed=PERM_SEED + s)
        seed_dir.append({"seed": s, "n": int(len(o)), "R": R_s,
                         "null_mean": float(null_s.mean()), "null_p95": float(np.percentile(null_s, 95)),
                         "p": float((null_s >= R_s).mean())})
    res["seed_directional"] = seed_dir

    # --- pre-registered classification ---
    res["classification"] = classify(res)
    return res


def classify(res: Dict[str, Any]) -> Dict[str, Any]:
    ca = res["core_A_wrong_value"]
    cb = res["core_B_wrap"]
    cd = res["core_D_offaxis"]["directional_consistency_offpath"]
    sd = res["seed_directional"]
    same_par = ca["same_parity_rate"]; p_par = ca["same_parity_perm_p"]
    tv = cb["tv_wrap_vs_nowrap"]
    R_seed_mean = float(np.mean([x["R"] for x in sd])) if sd else float("nan")
    dir_all_sig = bool(sd) and all(x["p"] <= PERM_P_BAR for x in sd)
    modular_identity = bool(same_par >= MODULAR_PARITY_BAR and p_par <= PERM_P_BAR)
    wrap_specific = bool(tv >= WRAP_TV_BAR)
    directional_systematic = bool(dir_all_sig and (R_seed_mean >= R_BAR))
    directional_weak = bool(sd) and any(x["p"] <= PERM_P_BAR for x in sd)
    if modular_identity and directional_systematic:
        verdict = "M-A_systematic_modular_geometry"
    elif directional_systematic and not modular_identity:
        verdict = "M-B_systematic_nonmodular_placement"
    elif (not modular_identity) and (not directional_weak):
        verdict = "M-C_unsystematic_collapse"
    else:
        verdict = "M-D_mixed_inconclusive"
    return {
        "verdict": verdict,
        "modular_identity": modular_identity, "wrap_specific": wrap_specific,
        "directional_systematic": directional_systematic, "directional_weak": directional_weak,
        "same_parity_rate": same_par, "same_parity_perm_p": p_par, "wrap_tv": tv,
        "R_seed_mean": R_seed_mean, "R_all_seeds_significant": dir_all_sig,
        "R_pooled_descriptive": cd["R"], "R_pooled_perm_p": cd["perm_p"],
        "seed_R": [x["R"] for x in sd], "seed_R_p": [x["p"] for x in sd],
        "thresholds": {"modular_parity_bar": MODULAR_PARITY_BAR, "perm_p_bar": PERM_P_BAR,
                       "wrap_tv_bar": WRAP_TV_BAR, "R_bar": R_BAR},
        "rule_note": ("M-A: modular_identity and directional_systematic (seed-level R bar). M-B: "
                      "directional_systematic and not modular_identity. M-C: not modular_identity "
                      "and not directional_weak. M-D otherwise. Directional systematics uses the "
                      "seed-level mean resultant (independent units = the 5 checkpoints); the pooled "
                      "R is reported descriptively. Bars are declared conventions, not tuned cutoffs."),
    }


# ---------------------------------------------------------------------------
# statistics over seeds
# ---------------------------------------------------------------------------

def seed_summary(res: Dict[str, Any], key: str) -> Dict[str, Any]:
    vals = [v for v in res["per_seed"][key] if v is not None]
    if not vals:
        return {"mean": None, "sd": None, "per_seed": res["per_seed"][key]}
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "per_seed": res["per_seed"][key]}


def paired_seed(resE: Dict[str, Any], resF: Dict[str, Any], key: str) -> Dict[str, Any]:
    a = np.array([np.nan if v is None else v for v in resE["per_seed"][key]], float)
    b = np.array([np.nan if v is None else v for v in resF["per_seed"][key]], float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return {}
    d = a[ok] - b[ok]; sd = d.std(ddof=1)
    out = {"E_mean": float(a[ok].mean()), "F_mean": float(b[ok].mean()), "mean_diff": float(d.mean()),
           "sign_positive": int((d > 0).sum()), "sign_negative": int((d < 0).sum()),
           "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"), "n": int(ok.sum())}
    try:
        out["paired_t_p"] = float(st.ttest_rel(a[ok], b[ok]).pvalue)
    except Exception:
        out["paired_t_p"] = None
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d: Dict[str, Any], res: Dict[str, Any], out_dir: str, prefix="experiment_m") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    tE: MTable = d[PRIMARY_MODEL]["table"]; tF: MTable = d[CONTROL_MODEL]["table"]
    gE = group_rows(d[PRIMARY_MODEL]); gF = group_rows(d[CONTROL_MODEL])
    offE = gE["offpath_failure"]; succE = gE["candidate_correct_success"]; nonoffE = gE["nonoffpath_failure"]
    c = lambda n: tE[n]

    # 1. wrong-value distribution vs reference vs uniform
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(MODULUS); w = 0.28
    ca = res[PRIMARY_MODEL]["core_A_wrong_value"]
    obs = np.array(ca["observed_wrong_distribution"], float); obs /= obs.sum()
    ref = np.array(ca["reference_y_distribution"], float); ref /= ref.sum()
    ax.bar(x - w, obs, w, label="observed wrong value", color="tab:red")
    ax.bar(x, ref, w, label="reference (y distribution)", color="tab:blue")
    ax.bar(x + w, np.full(MODULUS, 1 / MODULUS), w, label="uniform", color="tab:gray")
    ax.set_xticks(x); ax.set_xlabel("value"); ax.set_ylabel("fraction")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E off-path: wrong destination value vs reference/uniform")
    save(fig, "wrong_value_hist")

    # 2. wrong offset distribution, wrap vs no-wrap (E)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vw = c("comm_pred")[offE].astype(int); y = c("y_true")[offE].astype(int)
    offs = ((vw - y + MODULUS // 2) % MODULUS) - MODULUS // 2
    wmask = c("wrap")[offE] > 0.5
    bins = np.arange(-5.5, 5.5, 1)
    ax.hist(offs[wmask], bins=bins, alpha=0.6, label="wrap", color="tab:orange")
    ax.hist(offs[~wmask], bins=bins, alpha=0.6, label="no-wrap", color="tab:blue")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("signed modular offset (v_wrong - y) in [-5,4]"); ax.set_ylabel("count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("E off-path: wrong-value offset by wrap status")
    save(fig, "wrong_offset")

    # 3. modular distances
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, ttl in ((axes[0], "dmod_wrong_true", "to target y"), (axes[1], "dmod_wrong_old", "to old value"),
                         (axes[2], "dmod_wrong_operand", "to operand")):
        v = c(col)[offE]
        ax.hist(v, bins=np.arange(-0.5, 5.5, 1), color="tab:purple")
        ax.set_title(f"m_d_mod(v_wrong, {ttl})", fontsize=9); ax.grid(alpha=0.3)
        ax.set_xlabel("modular distance")
    fig.suptitle("E off-path: modular distances of the wrong value")
    save(fig, "modular_distance")

    # 4. wrap split wrong-value distributions
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pw = np.bincount(vw[wmask], minlength=MODULUS) / max(wmask.sum(), 1)
    pn = np.bincount(vw[~wmask], minlength=MODULUS) / max((~wmask).sum(), 1)
    ax.plot(x, pw, marker="o", label="wrap", color="tab:orange")
    ax.plot(x, pn, marker="s", label="no-wrap", color="tab:blue")
    ax.set_xlabel("wrong value"); ax.set_ylabel("fraction"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"E off-path: P(v_wrong | wrap) vs P(v_wrong | no-wrap)  (TV={res[PRIMARY_MODEL]['core_B_wrap']['tv_wrap_vs_nowrap']:.3f})")
    save(fig, "wrap_split")

    # 5. wrong register
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    co = res[PRIMARY_MODEL]["core_C_register"]["offpath_dest_src_neither"]
    cs = res[PRIMARY_MODEL]["core_C_register"]["success_dest_src_neither"]
    xs = np.arange(3); w = 0.38
    ax.bar(xs - w / 2, co, w, label="off-path failure", color="tab:red")
    ax.bar(xs + w / 2, cs, w, label="success", color="tab:green")
    ax.set_xticks(xs); ax.set_xticklabels(["flat winner == dest", "== source", "neither"])
    ax.set_ylabel("fraction"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: flat 40-way committed winner register")
    save(fig, "wrong_register")

    # 6. off-axis decomposition
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, sel, col in (("off-path", offE, "tab:red"), ("success", succE, "tab:green"), ("non-off", nonoffE, "tab:gray")):
        axes[0].hist(c("off_ratio")[sel], bins=40, range=(0, 1.5), density=True, alpha=0.5, label=name, color=col)
        axes[1].hist(c("off_cos")[sel], bins=40, range=(0, 1), density=True, alpha=0.5, label=name, color=col)
    axes[0].set_xlabel("||orthogonal|| / ||Delta_candidate||"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("cos(Delta_commit, Delta_candidate)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("E: off-axis decomposition by group")
    save(fig, "offaxis_decomposition")

    # 7. directional consistency + permutation null (seed-level = independent units)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    sd = res[PRIMARY_MODEL]["seed_directional"]
    xs = np.arange(len(sd))
    ax.bar(xs, [x["R"] for x in sd], 0.5, color="tab:red", label="per-seed R")
    ax.plot(xs, [x["null_mean"] for x in sd], "k_", ms=15, label="within-seed null mean")
    ax.axhline(R_BAR, color="tab:blue", ls="--", lw=1, label=f"R bar {R_BAR}")
    ax.set_xticks(xs); ax.set_xticklabels([f"seed {x['seed']} (n={x['n']})" for x in sd], fontsize=8)
    ax.set_ylabel("R = ||mean unit off-axis||"); ax.set_ylim(0, max(0.4, max(x["R"] for x in sd) * 1.2))
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E off-path: seed-level off-axis directional consistency vs null")
    save(fig, "directional_consistency")

    # 8. off-axis direction vs arithmetic
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ce = res[PRIMARY_MODEL]["core_E_arithmetic_direction"]
    axes[0].bar(["cos\n(wrap grp,\nwrap dir)", "cos\n(no-wrap grp,\nno-wrap dir)", "cos\n(no-wrap grp,\nwrap dir)", "cos\n(wrap grp,\nno-wrap dir)"],
                [ce["cos_to_wrap_dir_wrap"], ce["cos_to_wrap_dir_nowrap"],
                 ce["cos_to_nowrap_dir_nowrap"], ce["cos_to_nowrap_dir_wrap"]],
                color=["tab:orange", "tab:blue", "tab:blue", "tab:orange"])
    axes[0].set_ylabel("mean cosine"); axes[0].grid(alpha=0.3, axis="y"); axes[0].set_title("group alignment", fontsize=9)
    keys = list(ce["R_within_operand_b"].keys())
    axes[1].bar([str(k) for k in keys], [ce["R_within_operand_b"][k] for k in keys], color="tab:green")
    axes[1].axhline(R_BAR, color="k", ls="--", lw=0.8, label=f"R bar {R_BAR}")
    axes[1].set_xlabel("operand_b value"); axes[1].set_ylabel("R within group"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_title("within-operand directional consistency", fontsize=9)
    fig.suptitle("E off-path: does off-axis direction track arithmetic structure?")
    save(fig, "offaxis_arithmetic")

    # 9. third-basin depth
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.hist(c("hyper_flat_commit")[offE], bins=40, range=(-15, 20), density=True, alpha=0.6,
            label="off-path failure", color="tab:red")
    ax.hist(c("hyper_flat_commit")[succE], bins=40, range=(-15, 20), density=True, alpha=0.6,
            label="success", color="tab:green")
    ax.axvline(0, color="k", lw=0.9)
    ax.set_xlabel("normalized true-vs-flat-winner distance"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E: third-basin depth (true vs flat 40-way winner)")
    save(fig, "third_basin_depth")

    # 10. candidate vs commit margin progression
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    stages = ["m_prev", "m_cand", "m_commit"]; lab = ["S_prev", "C", "S_commit"]
    for sel, name, col in ((offE, "off-path", "tab:red"), (succE, "success", "tab:green"), (nonoffE, "non-off", "tab:gray")):
        ax.plot(lab, [np.median(c(s)[sel]) for s in stages], marker="o", label=name, color=col)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("median margin (vs a_prev)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("E: pairwise margin progression by group")
    save(fig, "candidate_vs_commit")

    # 11. control comparison
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    metrics = ["same_parity", "off_ratio", "off_cos"]
    groups = [("off-path", offE), ("success", succE), ("non-off", nonoffE)]
    xs = np.arange(len(groups)); w = 0.25
    vals = {}
    for j, met in enumerate(metrics):
        arr = []
        for _, sel in groups:
            if met == "same_parity":
                vw2 = c("comm_pred")[sel].astype(int); y2 = c("y_true")[sel].astype(int)
                arr.append(float(((vw2 - y2) % 2 == 0).mean()))
            else:
                arr.append(float(np.median(c(met)[sel])))
        vals[met] = arr
        ax.bar(xs + (j - 1) * w, arr, w, label=met)
    ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in groups]); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: control comparison (A off-path / B success / C non-off-path failure)")
    save(fig, "control_comparison")

    # 12. E vs F
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    keys = ["same_parity_rate", "tv_wrap_vs_nowrap"]
    e = res[PRIMARY_MODEL]["core_A_wrong_value"]["same_parity_rate"]
    f = res[CONTROL_MODEL]["core_A_wrong_value"]["same_parity_rate"]
    eR = res[PRIMARY_MODEL]["classification"]["R_seed_mean"]
    fR = res[CONTROL_MODEL]["classification"]["R_seed_mean"]
    eTV = res[PRIMARY_MODEL]["core_B_wrap"]["tv_wrap_vs_nowrap"]
    fTV = res[CONTROL_MODEL]["core_B_wrap"]["tv_wrap_vs_nowrap"]
    ax.bar([0, 1, 2], [e, eR, eTV], 0.38, label="E", color="tab:blue")
    ax.bar([0.4, 1.4, 2.4], [f, fR, fTV], 0.38, label="F", color="tab:orange")
    ax.set_xticks([0.2, 1.2, 2.2]); ax.set_xticklabels(["same-parity", "R (direction)", "wrap TV"])
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("E vs F: third-basin geometry")
    save(fig, "E_vs_F")

    return paths


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build(d: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": SEEDS, "primary_model": PRIMARY_MODEL, "control_model": CONTROL_MODEL,
            "checkpoints": d["meta"]["models"], "l_alignment": d["meta"]["l_alignment"],
            "offpath_definition": d["meta"]["offpath_definition"],
            "mod_distance": d["meta"]["mod_distance"],
            "thresholds": {"modular_parity_bar": MODULAR_PARITY_BAR, "perm_p_bar": PERM_P_BAR,
                           "wrap_tv_bar": WRAP_TV_BAR, "R_bar": R_BAR, "n_perm": N_PERM},
            "statistical_discipline": ("Five frozen E checkpoints are independent units; F is a "
                                       "secondary control (not pooled). Transition-level tables are "
                                       "descriptive; directional/parity nulls are permutation tests."),
        },
    }
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        res[m] = analyze_model(m, d[m])
    res["E_vs_F_paired"] = {
        "same_parity": paired_seed(res[PRIMARY_MODEL], res[CONTROL_MODEL], "same_parity"),
        "R": paired_seed(res[PRIMARY_MODEL], res[CONTROL_MODEL], "R"),
        "mean_cos": paired_seed(res[PRIMARY_MODEL], res[CONTROL_MODEL], "mean_cos"),
    }
    res["seed_level"] = {m: {k: seed_summary(res[m], k) for k in ("offpath_count", "same_parity", "R", "mean_cos")}
                         for m in (PRIMARY_MODEL, CONTROL_MODEL)}
    res["primary_decision"] = res[PRIMARY_MODEL]["classification"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment M analysis (measurement only)")
    ap.add_argument("--geometry", default="experiments/experiment_m_geometry.npz")
    ap.add_argument("--output", default="experiments/experiment_m_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_m")
    args = ap.parse_args()

    d = load_geometry(args.geometry)
    res = build(d)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    print("E verdict:", res[PRIMARY_MODEL]["classification"]["verdict"])
    print("F verdict:", res[CONTROL_MODEL]["classification"]["verdict"])
    paths = make_figures(d, res, args.fig_dir, prefix=args.fig_prefix)
    print("Figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
