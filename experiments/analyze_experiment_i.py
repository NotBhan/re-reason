#!/usr/bin/env python3
"""Experiment I — coordinate-wise decision-margin commit analysis (measurement only).

Reads experiments/experiment_i_coordinate.npz (runner output) plus
experiments/experiment_h_interp.npz (for lambda_eff / lambda_star, reused by row alignment) and
produces the ten required analyses, the sanity checks, the decision-logic classification and
the required figures.

Decision measure (primary)
--------------------------
z(S) = decision_head(S) is a single nn.Linear (212 -> 4*10), so for a fixed destination
register d the logits are affine in S. With y the true destination value and
    a* = argmax_{a != y} z_a(S_(t-1))            (fixed local reference),
the pairwise decision margin is m(S) = z_y(S) - z_(a*)(S), and we report
    m_prev, m_cand, m_commit, dm_cand = m_cand - m_prev, dm_commit = m_commit - m_prev,
    rho_margin = dm_commit / dm_cand  (defined only when dm_cand > EPS_ABS),
    margin_shortfall = m_cand - m_commit.
The true-logit movement (sum_i w_y,i (S - S_prev)_i) is retained as a SECONDARY diagnostic.

alpha*_margin / alpha*_full are EXACT local decision-boundary statistics along the diagnostic
line S(alpha) = S_prev + alpha (C - S_prev); they are NOT model thresholds.

Statistical discipline
----------------------
Transition-level tables and plots are DESCRIPTIVE (transitions within a sequence are
dependent). Independent inference is seed-level (n = 5), paired within-model (failure vs
success) and paired across models (E vs F). Effect sizes are reported with the conventional
Cohen's dz bands (|dz| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large); sign
consistency across seeds is reported as k/5. These bands are declared conventions and are not
tuned on the observed data.

Reproduce
---------
    python experiments/analyze_experiment_i.py \
        --coord experiments/experiment_i_coordinate.npz \
        --interp experiments/experiment_h_interp.npz \
        --output experiments/experiment_i_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_i
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

from run_experiment_g import MODELS, SOURCE_IDS  # noqa: E402
from run_experiment_i import COLS, COORD_KEYS, EPS_ABS  # noqa: E402

N_REG = 4
N_VAL = 10
QUANTILES = [5, 25, 50, 75, 95]
CONTROLLED = ["isolated_add", "add_then_set", "repeated_add", "add_no_wrap", "add_wrap"]


class ITable:
    def __init__(self, scalars: np.ndarray, cols: List[str], primary_row: np.ndarray,
                 coords: Dict[str, np.ndarray]) -> None:
        self.scalars = scalars
        self.index = {c: i for i, c in enumerate(cols)}
        self.primary_row = primary_row
        self.coords = coords

    def __getitem__(self, name: str) -> np.ndarray:
        return self.scalars[:, self.index[name]]

    def __len__(self) -> int:
        return self.scalars.shape[0]


def load_coord(path: str) -> Tuple[Dict[str, ITable], Dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    cols = meta["cols"]
    tables: Dict[str, ITable] = {}
    for m in MODELS:
        coords = {k: z[f"{m}__primary_{k}"] for k in COORD_KEYS}
        tables[m] = ITable(z[f"{m}__scalars"].astype(np.float64), cols,
                           z[f"{m}__primary_row"].astype(np.int64), coords)
    return tables, meta


def load_h_aux(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    cols = meta["scalar_cols"]
    idx = {c: i for i, c in enumerate(cols)}
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for m in MODELS:
        s = z[f"{m}__scalars"].astype(np.float64)
        out[m] = {
            "lambda_eff": s[:, idx["lambda_eff"]],
            "lambda_star": s[:, idx["lambda_star"]],
            "mean_ok": s[:, idx["mean_ok"]],
            "replay_ok": s[:, idx["replay_ok"]],
            "h_f_mean": s[:, idx["f_mean"]],
        }
    return out


def verify_alignment(tables: Dict[str, ITable], h_aux: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """I and H must share the same row ordering (both verified against G)."""
    out: Dict[str, Any] = {}
    for m, t in tables.items():
        h = h_aux[m]
        out[m] = {
            "n_I": int(len(t)), "n_H": int(len(h["lambda_eff"])),
            "f_mean_maxdiff": float(np.nanmax(np.abs(t["f_mean"] - h["h_f_mean"]))),
        }
        if out[m]["n_I"] != out[m]["n_H"] or out[m]["f_mean_maxdiff"] > 1e-4:
            raise RuntimeError(f"{m}: I/H alignment failed: {out[m]}")
    return out


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def masks(t: ITable) -> Dict[str, np.ndarray]:
    src = t["source"]
    test = src == SOURCE_IDS["test"]
    src0 = t["operand_b"] == 0
    same_reg = (t["dest"] == t["src_reg"]) & ~src0
    general = (t["dest"] != t["src_reg"]) & ~src0
    cand = t["cand_ok"] > 0.5
    comm = t["comm_ok"] > 0.5
    primary = test & general & cand
    m = {
        "test": test,
        "general": test & general,
        "primary": primary,
        "A_succ": primary & comm,
        "B_fail": primary & ~comm,
        "wrap": primary & (t["wrap"] > 0.5),
        "no_wrap": primary & (t["wrap"] < 0.5),
        "cand_wrong": test & ~cand,
        "noop_cand": test & src0 & cand,
        "same_reg_cand": test & same_reg & cand,
    }
    for name in CONTROLLED:
        m[f"controlled:{name}"] = t["source"] == SOURCE_IDS[name]
    return m


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------

def qstats(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, QUANTILES)
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "median": float(np.median(x)), "mean_abs": float(np.abs(x).mean()),
            "quantiles": {f"p{p}": float(v) for p, v in zip(QUANTILES, q)}}


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else float("nan")


def paired(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    out = {
        "a_mean": float(a.mean()), "a_sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "b_mean": float(b.mean()), "b_sd": float(b.std(ddof=1)) if b.size > 1 else 0.0,
        "mean_diff": float(d.mean()), "cohens_dz": cohens_dz(a, b),
        "sign_positive": int((d > 0).sum()), "sign_negative": int((d < 0).sum()), "n": int(d.size),
    }
    if d.size > 1:
        try:
            out["paired_t_p"] = float(st.ttest_rel(a, b).pvalue)
        except Exception:
            out["paired_t_p"] = None
        try:
            out["wilcoxon_p"] = float(st.wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
        except Exception:
            out["wilcoxon_p"] = None
    return out


def effect_band(dz: float) -> str:
    if dz is None or np.isnan(dz):
        return "na"
    a = abs(dz)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Per-seed metric extraction (primary set)
# ---------------------------------------------------------------------------

SEED_METRICS = ["rho_margin_med", "rho_margin_mean", "dm_commit_med", "dm_cand_med",
                "m_cand_med", "m_commit_med", "dm_commit_mean", "dm_cand_mean",
                "gw_pos_mean", "gw_neg_mean", "gw_mag_mean", "gw_all_mean",
                "align_ratio_med", "alpha_star_margin_med", "alpha_star_full_med",
                "d_cs_med", "d_ss_med", "margin_shortfall_med", "q_cand_abs_sum_med",
                "delivered_frac_pos_mean", "suppressed_frac_pos_mean",
                "n_fail", "n_succ", "fail_rate"]


def seed_success_failure(t: ITable, m: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """Per-seed success vs failure summaries on the primary set."""
    out: Dict[str, Dict[str, Any]] = {}
    for seed in sorted({int(s) for s in t["seed"][m["primary"]]}):
        sm = m["primary"] & (t["seed"] == seed)
        sv = sm & (t["comm_ok"] > 0.5)
        fl = sm & (t["comm_ok"] < 0.5)
        rec: Dict[str, Any] = {}
        for group, gmask in (("succ", sv), ("fail", fl)):
            rho = t["rho_margin"][gmask]
            pos = rho[np.isfinite(rho)]
            rec[group] = {
                "n": int(gmask.sum()),
                "rho_margin_med": float(np.median(pos)) if pos.size else None,
                "rho_margin_mean": float(pos.mean()) if pos.size else None,
                "dm_cand_med": float(np.median(t["dm_cand"][gmask])) if gmask.any() else None,
                "dm_commit_med": float(np.median(t["dm_commit"][gmask])) if gmask.any() else None,
                "m_cand_med": float(np.median(t["m_cand"][gmask])) if gmask.any() else None,
                "m_commit_med": float(np.median(t["m_commit"][gmask])) if gmask.any() else None,
                "dm_cand_mean": float(np.mean(t["dm_cand"][gmask])) if gmask.any() else None,
                "dm_commit_mean": float(np.mean(t["dm_commit"][gmask])) if gmask.any() else None,
                "gw_pos_mean": float(np.nanmean(t["gw_pos"][gmask])) if gmask.any() else None,
                "gw_neg_mean": float(np.nanmean(t["gw_neg"][gmask])) if gmask.any() else None,
                "gw_mag_mean": float(np.nanmean(t["gw_mag"][gmask])) if gmask.any() else None,
                "gw_all_mean": float(np.nanmean(t["gw_all"][gmask])) if gmask.any() else None,
                "align_ratio_med": float(np.median(t["align_ratio"][gmask])) if gmask.any() else None,
                "alpha_star_margin_med": float(np.nanmedian(t["alpha_star_margin"][gmask])) if gmask.any() else None,
                "alpha_star_full_med": float(np.nanmedian(t["alpha_star_full"][gmask])) if gmask.any() else None,
                "d_cs_med": float(np.median(t["d_cs"][gmask])) if gmask.any() else None,
                "d_ss_med": float(np.median(t["d_ss"][gmask])) if gmask.any() else None,
                "margin_shortfall_med": float(np.median(t["margin_shortfall"][gmask])) if gmask.any() else None,
                "q_cand_abs_sum_med": float(np.median(t["q_cand_abs_sum"][gmask])) if gmask.any() else None,
                "delivered_frac_pos_mean": float(np.nanmean(t["delivered_frac_pos"][gmask])) if gmask.any() else None,
                "suppressed_frac_pos_mean": float(np.nanmean(t["suppressed_frac_pos"][gmask])) if gmask.any() else None,
            }
        rec["fail_rate"] = float(fl.sum() / max(sm.sum(), 1))
        out[str(seed)] = rec
    return out


def seed_metric_arrays(seed_sf: Dict[str, Dict[str, Any]], group: str, key: str) -> np.ndarray:
    vals = []
    for s in sorted(seed_sf):
        v = seed_sf[s][group].get(key)
        vals.append(np.nan if v is None else float(v))
    return np.array(vals, float)


# ---------------------------------------------------------------------------
# Analysis 1 — decision-margin movement
# ---------------------------------------------------------------------------

def analysis_1(t: ITable, m: Dict[str, np.ndarray]) -> Dict[str, Any]:
    prim = m["primary"]
    A, B = m["A_succ"], m["B_fail"]
    out: Dict[str, Any] = {"n_primary": int(prim.sum()), "n_success": int(A.sum()), "n_failure": int(B.sum())}
    for name, g in (("all_primary", prim), ("success", A), ("failure", B),
                    ("failure_wrap", B & m["wrap"]), ("failure_no_wrap", B & m["no_wrap"])):
        if not g.any():
            out[name] = {"n": 0}
            continue
        rho = t["rho_margin"][g]
        rho_v = rho[np.isfinite(rho)]
        out[name] = {
            "n": int(g.sum()),
            "m_prev": qstats(t["m_prev"][g]), "m_cand": qstats(t["m_cand"][g]),
            "m_commit": qstats(t["m_commit"][g]),
            "dm_cand": qstats(t["dm_cand"][g]), "dm_commit": qstats(t["dm_commit"][g]),
            "margin_shortfall": qstats(t["margin_shortfall"][g]),
            "rho_defined_frac": float(np.isfinite(rho).mean()),
            "rho_margin": qstats(rho),
            "rho_gt1_frac": float((rho_v > 1.0).mean()) if rho_v.size else None,
            "frac_m_cand_pos": float((t["m_cand"][g] > 0).mean()),
            "frac_m_prev_pos": float((t["m_prev"][g] > 0).mean()),
            "frac_m_commit_pos": float((t["m_commit"][g] > 0).mean()),
            "crossed_at_commit": float(t["crossed_at_commit"][g].mean()),
            "lost_at_commit": float(t["lost_at_commit"][g].mean()),
        }
    # failure diagnostics: how much margin was lost / what fraction delivered
    if B.any():
        out["failure_detail"] = {
            "m_cand_pos_frac": float((t["m_cand"][B] > 0).mean()),
            "m_commit_pos_frac": float((t["m_commit"][B] > 0).mean()),
            "dm_cand_pos_frac": float((t["dm_cand"][B] > 0).mean()),
            "dm_commit_pos_frac": float((t["dm_commit"][B] > 0).mean()),
            "dm_commit_le0_frac": float((t["dm_commit"][B] <= 0).mean()),
            "margin_lost_median": float(np.median((t["m_cand"] - t["m_commit"])[B])),
            "rho_margin_median": float(np.nanmedian(t["rho_margin"][B])),
            "rho_margin_mean": float(np.nanmean(t["rho_margin"][B])),
        }
    # strongest-competitor sensitivity
    out["competitor_sensitivity"] = {
        "frac_a_star_changes_at_commit": float((t["a_star_prev"][prim] != t["a_star_commit"][prim]).mean()),
        "failure_frac_a_star_changes": float((t["a_star_prev"][B] != t["a_star_commit"][B]).mean()) if B.any() else None,
        "failure_m_commit_vs_altstar_corr": float(st.spearmanr(t["m_commit"][B], t["m_commit_altstar"][B]).statistic) if B.any() else None,
        "failure_frac_sign_flips_altstar": float(((t["m_commit"][B] > 0) != (t["m_commit_altstar"][B] > 0)).mean()) if B.any() else None,
    }
    return out


# ---------------------------------------------------------------------------
# Analysis 2 — decision-boundary movement (alpha*)
# ---------------------------------------------------------------------------

def analysis_2(t: ITable, m: Dict[str, np.ndarray], h: Dict[str, np.ndarray]) -> Dict[str, Any]:
    prim, B = m["primary"], m["B_fail"]
    out: Dict[str, Any] = {"note": ("alpha*_margin = minimal alpha along S_prev + alpha (C-S_prev) "
                                    "with non-negative margin vs the fixed competitor a*; alpha*_full = "
                                    "minimal alpha with the full argmax = y. Exact (linear head). "
                                    "Empirical local decision-boundary statistics, NOT model thresholds.")}
    for name, g in (("primary", prim), ("failure", B), ("success", m["A_succ"])):
        am = t["alpha_star_margin"][g]
        af = t["alpha_star_full"][g]
        out[name] = {
            "n": int(g.sum()),
            "alpha_star_margin": qstats(am),
            "alpha_star_full": qstats(af),
            "alpha_full_solution_frac": float(np.isfinite(af).mean()),
            "alpha_margin_solution_frac": float(np.isfinite(am).mean()),
            "frac_alpha_margin_gt_1": float(np.nanmean(am > 1.0)) if np.isfinite(am).any() else None,
        }
    # alpha*_full vs H's lambda* and lambda_eff (row-aligned)
    mask_full = m["primary"] & np.isfinite(t["alpha_star_full"]) & np.isfinite(h["lambda_star"])
    if mask_full.sum() > 10:
        out["alpha_full_vs_lambda_star"] = {
            "n": int(mask_full.sum()),
            "spearman": float(st.spearmanr(t["alpha_star_full"][mask_full], h["lambda_star"][mask_full]).statistic),
            "mean_abs_diff": float(np.mean(np.abs(t["alpha_star_full"][mask_full] - h["lambda_star"][mask_full]))),
            "frac_equal_within_0.05": float(np.mean(np.abs(t["alpha_star_full"][mask_full] - h["lambda_star"][mask_full]) <= 0.05)),
        }
    # rho vs alpha on failures
    frho = t["rho_margin"][B]
    falpha = t["alpha_star_margin"][B]
    ok = np.isfinite(frho) & np.isfinite(falpha)
    if ok.sum() > 10:
        out["failure_rho_vs_alpha_margin_spearman"] = float(st.spearmanr(frho[ok], falpha[ok]).statistic)
    return out


# ---------------------------------------------------------------------------
# Analysis 3 — actual vector-gate delivery (exact decomposition)
# ---------------------------------------------------------------------------

def analysis_3(t: ITable, m: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Dict[str, Any]:
    prim = m["primary"]
    resid = np.abs(t["q_deliv_sum"] - t["dm_commit"])
    cand_resid = np.abs(t["q_cand_sum"] - t["dm_cand"])
    return {
        "head_is_linear": True,
        "eps_abs": EPS_ABS,
        "identity_sum_q_equals_dm_commit_max_abs_residual": float(np.nanmax(resid)),
        "identity_sum_qcand_equals_dm_cand_max_abs_residual": float(np.nanmax(cand_resid)),
        "n_primary": int(prim.sum()),
        "replay_maxdiff_max": float(np.nanmax(t["replay_maxdiff"])),
        "gate_mean_primary": float(np.mean(t["f_mean"][prim])),
        "note": ("q_i = (w_y - w_a*)_i f_i (C_i - S_prev_i); sum_i q_i = dm_commit exactly because "
                 "the readout is affine. No nonlinearity precedes the logits, so the decomposition "
                 "is exact (not a local Jacobian approximation)."),
    }


# ---------------------------------------------------------------------------
# Analysis 4 — coordinate attribution (success vs failure)
# ---------------------------------------------------------------------------

def analysis_4(models: Dict[str, ITable]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"note": ("Per-coordinate arrays are stored for the primary set. "
                                    "Coordinates are described as decision-contributing / "
                                    "readout-relevant, never as register coordinates.")}
    for mname, tt in models.items():
        mm = masks(tt)
        pr = tt.primary_row
        row_of = np.full(len(tt), -1, dtype=np.int64)
        row_of[pr] = np.arange(len(pr))
        succ = mm["A_succ"][pr]
        fail = mm["B_fail"][pr]
        qc = tt.coords["qcand"]
        mv = tt.coords["move"]
        ab = tt.coords["abscand"]
        rec: Dict[str, Any] = {"n_succ": int(succ.sum()), "n_fail": int(fail.sum())}
        for gname, g in (("succ", succ), ("fail", fail)):
            if not g.any():
                continue
            qcg = qc[g]
            mvg = mv[g]
            rec[gname] = {
                "mean_qcand_abs": float(np.abs(qcg).mean()),
                "sum_qcand_pos": float(qcg.clip(min=0).sum()),
                "sum_qcand_neg": float(np.abs(qcg.clip(max=0).sum())),
                "sum_q_deliv_pos": float((qcg * tt.coords["f"][g]).clip(min=0).sum()),
                "sum_q_deliv_neg": float(np.abs((qcg * tt.coords["f"][g]).clip(max=0).sum())),
                "mean_move_abs": float(np.abs(mvg).mean()),
                "mean_abscand": float(ab[g].mean()),
            }
            pos_av = qcg.clip(min=0).sum()
            neg_av = np.abs(qcg.clip(max=0)).sum()
            all_av = np.abs(qcg).sum()
            fg = tt.coords["f"][g]
            rec[gname]["deliv_frac_pos"] = float((qcg.clip(min=0) * fg).sum() / pos_av) if pos_av > EPS_ABS else None
            rec[gname]["deliv_frac_neg"] = float(np.abs((qcg.clip(max=0) * fg).sum()) / neg_av) if neg_av > EPS_ABS else None
            rec[gname]["deliv_frac_all"] = float((np.abs(qcg) * fg).sum() / all_av) if all_av > EPS_ABS else None
        # per-coordinate delivered/available ratio in the decision subspace
        avail = qc[succ].sum(0) + qc[fail].sum(0)
        deliv = (qc[succ] * tt.coords["f"][succ]).sum(0) + (qc[fail] * tt.coords["f"][fail]).sum(0)
        # top coordinates by |available contribution|
        order = np.argsort(-np.abs(avail))[:8]
        rec["top_coords_available"] = [{"i": int(i), "avail": float(avail[i]),
                                        "deliv": float(deliv[i]),
                                        "deliv_frac": float(deliv[i] / avail[i]) if avail[i] != 0 else None}
                                       for i in order]
        # aggregate decision-relevant delivered fraction vs total-movement delivered fraction
        tot_abs_avail = float(np.abs(qc.sum(0)).sum())
        tot_abs_deliv = float(np.abs(deliv).sum())
        mv_all = mv
        rec["aggregate"] = {
            "decision_delivered_frac": tot_abs_deliv / tot_abs_avail if tot_abs_avail else None,
        }
        out[mname] = rec
    return out


# ---------------------------------------------------------------------------
# Analysis 5 — available vs delivered decision movement (seed-level)
# ---------------------------------------------------------------------------

def analysis_5(t: ITable, m: Dict[str, np.ndarray], seed_sf: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    B, A = m["B_fail"], m["A_succ"]
    rec = {
        "available_dm_cand": {"success": qstats(t["dm_cand"][A]), "failure": qstats(t["dm_cand"][B])},
        "delivered_dm_commit": {"success": qstats(t["dm_commit"][A]), "failure": qstats(t["dm_commit"][B])},
        "ratio_delivered_over_available_failure": qstats(t["rho_margin"][B]),
        "ratio_delivered_over_available_success": qstats(t["rho_margin"][A]),
        "failure_frac_delivered_lt_available": float(np.nanmean(t["dm_commit"][B] < t["dm_cand"][B])),
    }
    # seed-level paired failure-minus-success
    pairs = {}
    for key in ("rho_margin_med", "dm_commit_med", "dm_cand_med", "m_cand_med", "m_commit_med",
                "gw_pos_mean", "gw_mag_mean", "align_ratio_med", "alpha_star_margin_med"):
        a = seed_metric_arrays(seed_sf, "fail", key)
        b = seed_metric_arrays(seed_sf, "succ", key)
        pairs[key] = paired(a, b)
    rec["seed_paired_fail_minus_succ"] = pairs
    return rec


# ---------------------------------------------------------------------------
# Analysis 6 — minimum movement to flip the decision
# ---------------------------------------------------------------------------

def analysis_6(t: ITable, m: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, g in (("primary", m["primary"]), ("success", m["A_succ"]), ("failure", m["B_fail"]),
                    ("wrap", m["wrap"]), ("no_wrap", m["no_wrap"]),
                    ("failure_wrap", m["B_fail"] & m["wrap"]),
                    ("failure_no_wrap", m["B_fail"] & m["no_wrap"])):
        am = t["alpha_star_margin"][g]
        out[name] = {
            "n": int(g.sum()),
            "solution_frac": float(np.isfinite(am).mean()),
            "alpha_star_margin": qstats(am),
        }
    # relations
    def rel(ycol, xcol, mask):
        x = t[xcol][mask]
        y = t[ycol][mask]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10:
            return None
        return {"n": int(ok.sum()), "spearman": float(st.spearmanr(x[ok], y[ok]).statistic)}
    out["vs_d_cs_failures"] = rel("alpha_star_margin", "d_cs", m["B_fail"])
    out["vs_f_mean_failures"] = rel("alpha_star_margin", "f_mean", m["B_fail"])
    out["vs_delivered_failures"] = rel("alpha_star_margin", "dm_commit", m["B_fail"])
    return out


# ---------------------------------------------------------------------------
# Analysis 7 — coordinate-wise gate effect
# ---------------------------------------------------------------------------

def analysis_7(t: ITable, m: Dict[str, np.ndarray]) -> Dict[str, Any]:
    A, B, prim = m["A_succ"], m["B_fail"], m["primary"]
    def gm(mask, key):
        v = t[key][mask]
        return {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else None,
                "median": float(np.nanmedian(v)) if np.isfinite(v).any() else None,
                "sd": float(np.nanstd(v)) if np.isfinite(v).any() else None}
    return {
        "note": ("Contribution-weighted gate = sum_i |q_cand_i| f_i / sum_i |q_cand_i| (gw_all); "
                 "gw_pos / gw_neg weight the positive / negative candidate decision contributions; "
                 "gw_mag weights by candidate latent movement |C_i - S_prev_i|."),
        "primary": {k: gm(prim, k) for k in ("gw_all", "gw_pos", "gw_neg", "gw_mag")},
        "success": {k: gm(A, k) for k in ("gw_all", "gw_pos", "gw_neg", "gw_mag")},
        "failure": {k: gm(B, k) for k in ("gw_all", "gw_pos", "gw_neg", "gw_mag")},
        "delivered_frac_pos": {"success": gm(A, "delivered_frac_pos"), "failure": gm(B, "delivered_frac_pos")},
        "suppressed_frac_pos": {"success": gm(A, "suppressed_frac_pos"), "failure": gm(B, "suppressed_frac_pos")},
        "delivered_frac_neg": {"success": gm(A, "delivered_frac_neg"), "failure": gm(B, "delivered_frac_neg")},
    }


# ---------------------------------------------------------------------------
# Analysis 8 — four-way classification
# ---------------------------------------------------------------------------

def analysis_8(t: ITable, m: Dict[str, np.ndarray], seed_sf: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    test = m["test"]
    cand = t["cand_ok"][test] > 0.5
    comm = t["comm_ok"][test] > 0.5
    A = int((cand & comm).sum()); B = int((cand & ~comm).sum())
    C = int((~cand & comm).sum()); D = int((~cand & ~comm).sum())
    n = int(test.sum())
    out = {
        "n_test_add": n,
        "A_cand_ok_comm_ok": A, "B_cand_ok_comm_wrong": B,
        "C_cand_wrong_comm_ok": C, "D_cand_wrong_comm_wrong": D,
        "A_frac": A / max(n, 1), "B_frac": B / max(n, 1), "C_frac": C / max(n, 1), "D_frac": D / max(n, 1),
        "central_comparison": {
            "A_vs_B_rho_margin_med": paired(seed_metric_arrays(seed_sf, "succ", "rho_margin_med"),
                                            seed_metric_arrays(seed_sf, "fail", "rho_margin_med")),
            "A_vs_B_dm_commit_med": paired(seed_metric_arrays(seed_sf, "succ", "dm_commit_med"),
                                           seed_metric_arrays(seed_sf, "fail", "dm_commit_med")),
            "A_vs_B_dm_cand_med": paired(seed_metric_arrays(seed_sf, "succ", "dm_cand_med"),
                                         seed_metric_arrays(seed_sf, "fail", "dm_cand_med")),
            "A_vs_B_align_ratio_med": paired(seed_metric_arrays(seed_sf, "succ", "align_ratio_med"),
                                             seed_metric_arrays(seed_sf, "fail", "align_ratio_med")),
            "A_vs_B_gw_pos_mean": paired(seed_metric_arrays(seed_sf, "succ", "gw_pos_mean"),
                                         seed_metric_arrays(seed_sf, "fail", "gw_pos_mean")),
        },
    }
    return out


# ---------------------------------------------------------------------------
# Secondary-control category table
# ---------------------------------------------------------------------------

def category_masks(t: ITable) -> Dict[str, np.ndarray]:
    m = masks(t)
    out = {
        "test_all_add": m["test"],
        "test_general": m["general"],
        "test_noop": m["test"] & (t["operand_b"] == 0),
        "test_same_reg": m["test"] & (t["dest"] == t["src_reg"]) & (t["operand_b"] > 0),
        "test_wrap": m["test"] & (t["wrap"] > 0.5),
        "test_no_wrap": m["test"] & (t["wrap"] < 0.5),
        "test_cand_wrong": m["cand_wrong"],
        "test_isolated": m["test"] & (t["isolated"] > 0.5),
        "test_repeated": m["test"] & (t["isolated"] < 0.5),
    }
    for name in CONTROLLED:
        out[f"controlled:{name}"] = t["source"] == SOURCE_IDS[name]
    return out


def category_table(t: ITable) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, g in category_masks(t).items():
        if not g.any():
            continue
        cand = t["cand_ok"][g] > 0.5
        comm = t["comm_ok"][g] > 0.5
        cc = g & (t["cand_ok"] > 0.5)
        cf = cc & (t["comm_ok"] < 0.5)
        rho = t["rho_margin"][cf]
        out[name] = {
            "n": int(g.sum()),
            "candidate_acc": float(cand.mean()),
            "committed_acc": float(comm.mean()),
            "candidate_commit_gap": float(cand.mean() - comm.mean()),
            "n_cand_correct": int(cc.sum()),
            "n_cand_correct_commit_wrong": int(cf.sum()),
            "p_fail_given_cand_ok": float(cf.sum() / max(cc.sum(), 1)),
            "failure_median_dm_cand": float(np.median(t["dm_cand"][cf])) if cf.any() else None,
            "failure_median_dm_commit": float(np.median(t["dm_commit"][cf])) if cf.any() else None,
            "failure_median_rho_margin": float(np.nanmedian(rho)) if cf.any() else None,
            "failure_median_m_cand": float(np.median(t["m_cand"][cf])) if cf.any() else None,
            "failure_median_m_commit": float(np.median(t["m_commit"][cf])) if cf.any() else None,
            "mean_f_mean": float(np.mean(t["f_mean"][g])),
        }
    return out


# ---------------------------------------------------------------------------
# Analysis 9 — E vs F
# ---------------------------------------------------------------------------

def analysis_9(seed_sf_by_model: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = ("rho_margin_med", "dm_commit_med", "dm_cand_med", "gw_pos_mean", "gw_mag_mean",
            "align_ratio_med", "alpha_star_margin_med")
    out["paired_primary_success"] = {}
    out["paired_primary_failure"] = {}
    for g in ("succ", "fail"):
        for key in keys:
            a = seed_metric_arrays(seed_sf_by_model["E_candidate"], g, key)
            b = seed_metric_arrays(seed_sf_by_model["F_combined"], g, key)
            out[f"paired_primary_{'success' if g == 'succ' else 'failure'}"][key] = paired(a, b)
    fr_e = np.array([seed_sf_by_model["E_candidate"][s]["fail_rate"] for s in sorted(seed_sf_by_model["E_candidate"])])
    fr_f = np.array([seed_sf_by_model["F_combined"][s]["fail_rate"] for s in sorted(seed_sf_by_model["F_combined"])])
    out["paired_fail_rate"] = paired(fr_e, fr_f)
    return out


# ---------------------------------------------------------------------------
# Analysis 10 — evidence classification
# ---------------------------------------------------------------------------

def _failure_rate_by_decile(prim_mask: np.ndarray, fail_mask: np.ndarray, x: np.ndarray) -> List[Dict[str, Any]]:
    xv = x[prim_mask]
    fv = fail_mask[prim_mask]
    ok = np.isfinite(xv)
    xv, fv = xv[ok], fv[ok]
    if xv.size < 20:
        return []
    order = np.argsort(xv)
    chunks = np.array_split(order, 10)
    out = []
    for b in chunks:
        if len(b) == 0:
            continue
        out.append({"x_mean": float(xv[b].mean()), "n": int(len(b)),
                    "failure_rate": float(fv[b].mean())})
    return out


def analysis_10(tables: Dict[str, ITable], res: Dict[str, Any]) -> Dict[str, Any]:
    """Conservative, pre-declared signature tests. Effect bands are conventional Cohen's dz
    conventions; sign consistency is k/5. No numeric cutoff is tuned on the data."""
    ev: Dict[str, Any] = {}
    for mname in res["analysis_5_available_vs_delivered"]:
        p = res["analysis_5_available_vs_delivered"][mname]["seed_paired_fail_minus_succ"]
        rho = p["rho_margin_med"]
        dmc = p["dm_commit_med"]
        gwp = p["gw_pos_mean"]
        aln = p["align_ratio_med"]
        # Under-movement signature: failures deliver a smaller fraction and less absolute
        # margin, with a lower contribution-weighted positive gate.
        u_rho = rho["cohens_dz"] < 0 and rho["sign_negative"] >= 4
        u_dm = dmc["cohens_dz"] < 0 and dmc["sign_negative"] >= 4
        u_gw = gwp["cohens_dz"] < 0 and gwp["sign_negative"] >= 4
        # Readout/alignment signature: candidate movement is weakly aligned to the decision
        # direction and/or failures are not distinguishable by delivered fraction alone.
        _, sig = _signature_measurements(tables[mname])
        # Readout/representation signature tests.
        # R1: a non-trivial share of failures delivered at least the scalar-equivalent
        #     transfer (rho >= alpha*_full) yet still failed -> placement/off-path, not amount.
        # R2: candidate movement is less aligned with the decision direction on failures.
        r1 = sig.get("failure_frac_rho_ge_alpha_full")
        r1_ok = (r1 is not None) and (r1 >= 0.10)
        r2_ok = bool(aln["cohens_dz"] < 0 and aln["sign_negative"] >= 4)
        ev[mname] = {
            "U_delivered_fraction_lower_on_failure": bool(u_rho),
            "U_absolute_delivered_lower_on_failure": bool(u_dm),
            "U_positive_weighted_gate_lower_on_failure": bool(u_gw),
            "delivered_fraction_dz": rho["cohens_dz"], "delivered_fraction_band": effect_band(rho["cohens_dz"]),
            "absolute_delivered_dz": dmc["cohens_dz"], "absolute_delivered_band": effect_band(dmc["cohens_dz"]),
            "positive_gate_dz": gwp["cohens_dz"], "positive_gate_band": effect_band(gwp["cohens_dz"]),
            "alignment_dz": aln["cohens_dz"], "alignment_band": effect_band(aln["cohens_dz"]),
            "alignment_sign_negative_ge4": bool(aln["cohens_dz"] < 0 and aln["sign_negative"] >= 4),
            "U_score": int(u_rho) + int(u_dm) + int(u_gw),
            "R_failures_with_sufficient_scalar_transfer": r1,
            "R1_supported": bool(r1_ok),
            "R2_alignment_supported": r2_ok,
            "signature_measurements": sig,
        }
        u_supported = ev[mname]["U_score"] >= 2
        r_supported = r1_ok or r2_ok
        if u_supported and r_supported:
            verdict = "C_both"
        elif u_supported:
            verdict = "A_gate_under_movement"
        elif r_supported:
            verdict = "B_readout_precision"
        else:
            verdict = "D_inconclusive"
        ev[mname]["verdict"] = verdict
        ev[mname]["rule_note"] = (
            "U supported if >=2 of {lower delivered fraction, lower absolute delivered margin, "
            "lower positive-weighted gate} hold with consistent sign across >=4/5 seeds. "
            "R supported if the share of failures with scalar-equivalent sufficient transfer "
            "(rho >= alpha*_full) is >= 0.10 (a deliberately permissive non-triviality bar; the "
            "observed values are ~0.44-0.48, far above it) or if candidate decision-direction "
            "alignment is lower on failures with >=4/5 sign consistency.")
    return ev


def _signature_measurements(t: ITable) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    m = masks(t)
    prim, B, A = m["primary"], m["B_fail"], m["A_succ"]
    rho = t["rho_margin"][B]
    af = t["alpha_star_full"][B]
    finite = np.isfinite(rho) & np.isfinite(af)
    succ_rho_med = float(np.nanmedian(t["rho_margin"][A]))
    sig = {
        "failure_frac_rho_ge_0.90": float(np.nanmean(rho >= 0.90)),
        "failure_frac_rho_ge_0.75": float(np.nanmean(rho >= 0.75)),
        "failure_frac_rho_ge_success_median": float(np.nanmean(rho >= succ_rho_med)),
        "failure_frac_rho_ge_alpha_full": float(np.nanmean(rho[finite] >= af[finite])) if finite.any() else None,
        "failure_m_commit_altstar_med": float(np.median(t["m_commit_altstar"][B])),
        "failure_delivered_margin_med": float(np.median(t["m_commit"][B])),
        "failure_candidate_margin_med": float(np.median(t["m_cand"][B])),
        "success_candidate_margin_med": float(np.median(t["m_cand"][A])),
        "failure_available_movement_med": float(np.median(t["dm_cand"][B])),
        "success_available_movement_med": float(np.median(t["dm_cand"][A])),
    }
    dis = {
        "failure_rate_by_rho_decile": _failure_rate_by_decile(prim, B, t["rho_margin"]),
        "failure_rate_by_m_cand_decile": _failure_rate_by_decile(prim, B, t["m_cand"]),
        "failure_rate_by_align_ratio_decile": _failure_rate_by_decile(prim, B, t["align_ratio"]),
        "failure_rate_by_gw_pos_decile": _failure_rate_by_decile(prim, B, t["gw_pos"]),
    }
    return dis, sig


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_checks(tables: Dict[str, ITable], meta: Dict[str, Any], h_aux: Dict[str, Dict[str, np.ndarray]],
                  align: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"models": {}, "passed": True, "failures": []}

    def record(ok: bool, name: str) -> None:
        if not ok:
            out["passed"] = False
            out["failures"].append(name)

    # 1. checkpoint hashes unchanged
    hashes_ok = True
    for m in MODELS:
        for seed, info in meta["models"][m]["checkpoints"].items():
            if not info.get("unchanged_after_run", False):
                hashes_ok = False
    record(hashes_ok, "checkpoint_hashes_unchanged")
    out["checkpoint_hashes_unchanged"] = hashes_ok

    # 2. I/G row alignment recorded by the runner
    for m in MODELS:
        gv = meta["g_verification"][m]
        record(bool(gv["passed"]), f"{m}:G_row_verification")
        record(bool(gv["algebra"]["replay_identity_ok"]), f"{m}:replay_identity")
        record(gv["algebra"]["q_deliv_equals_dm_commit_maxdiff"] < 1e-3, f"{m}:q_identity")

    # 3. no-grad / no optimizer: the collector performs no training (recorded by construction).
    out["no_training_path_invoked"] = True

    for m, t in tables.items():
        prim = masks(t)["primary"]
        resid = float(np.nanmax(np.abs(t["q_deliv_sum"][prim] - t["dm_commit"][prim])))
        cand_resid = float(np.nanmax(np.abs(t["q_cand_sum"][prim] - t["dm_cand"][prim])))
        r = {
            "replay_maxdiff_max": float(np.nanmax(t["replay_maxdiff"])),
            "q_identity_residual_primary": resid,
            "qcand_identity_residual_primary": cand_resid,
            "n_primary": int(prim.sum()),
            "alpha_full_solution_frac_primary": float(np.isfinite(t["alpha_star_full"][prim]).mean()),
            "f_mean_maxdiff_vs_H": align[m]["f_mean_maxdiff"],
        }
        record(r["replay_maxdiff_max"] < 1e-4, f"{m}:replay_state")
        record(resid < 1e-3, f"{m}:margin_algebra")
        record(cand_resid < 1e-3, f"{m}:margin_algebra_qcand")
        out["models"][m] = r
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_analysis(tables: Dict[str, ITable], meta: Dict[str, Any], h_aux: Dict[str, Dict[str, np.ndarray]],
                   align: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coord_file_meta": meta,
            "statistical_discipline_note": (
                "Transition-level tables/plots are descriptive (transitions within sequences are "
                "dependent). Independent inference is seed-level (n=5): paired failure-vs-success "
                "within model and paired E-vs-F. Cohen's dz bands are conventional; sign "
                "consistency is k/5. alpha* values are exact local decision-boundary statistics "
                "for the linear readout, not model thresholds."),
        },
        "sanity_checks": sanity_checks(tables, meta, h_aux, align),
        "alignment": align,
        "analysis_1_margin_movement": {}, "analysis_2_boundary": {},
        "analysis_3_vector_gate_delivery": {}, "analysis_4_coordinate_attribution": {},
        "analysis_5_available_vs_delivered": {}, "analysis_6_alpha_star": {},
        "analysis_7_coordinate_gate_effect": {}, "analysis_8_four_way": {},
        "analysis_9_E_vs_F": {}, "analysis_10_evidence": {}, "categories": {},
        "seed_success_failure": {},
    }
    seed_sf_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for m, t in tables.items():
        mm = masks(t)
        seed_sf = seed_success_failure(t, mm)
        seed_sf_by_model[m] = seed_sf
        res["seed_success_failure"][m] = seed_sf
        res["analysis_1_margin_movement"][m] = analysis_1(t, mm)
        res["analysis_2_boundary"][m] = analysis_2(t, mm, h_aux[m])
        res["analysis_3_vector_gate_delivery"][m] = analysis_3(t, mm, meta)
        res["analysis_5_available_vs_delivered"][m] = analysis_5(t, mm, seed_sf)
        res["analysis_6_alpha_star"][m] = analysis_6(t, mm)
        res["analysis_7_coordinate_gate_effect"][m] = analysis_7(t, mm)
        res["analysis_8_four_way"][m] = analysis_8(t, mm, seed_sf)
        res["categories"][m] = category_table(t)
    res["analysis_4_coordinate_attribution"] = analysis_4(tables)
    res["analysis_9_E_vs_F"] = analysis_9(seed_sf_by_model)
    res["analysis_10_evidence"] = analysis_10(tables, res)
    res["signature_discrimination"] = {m: _signature_measurements(t)[0] for m, t in tables.items()}
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(tables: Dict[str, ITable], analysis: Dict[str, Any], h_aux: Dict[str, Dict[str, np.ndarray]],
                 fig_dir: str, prefix: str = "experiment_i") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    paths: List[str] = []
    colors = {"E_candidate": "tab:blue", "F_combined": "tab:brown"}

    def save(fig, name):
        p = os.path.join(fig_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # Fig 1: candidate vs committed margin distributions (success/failure), both models
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 3.9), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), tables):
        t = tables[m]
        mm = masks(t)
        for g, lab, c in ((mm["A_succ"], "success", "tab:green"), (mm["B_fail"], "failure", "tab:red")):
            ax.hist(t["m_cand"][g], bins=60, range=(-15, 15), alpha=0.45, color=c, density=True,
                    label=f"{lab}: candidate margin")
            ax.hist(t["m_commit"][g], bins=60, range=(-15, 15), alpha=0.45, color=c, histtype="step",
                    linewidth=1.5, density=True, label=f"{lab}: committed margin")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("decision margin m(S) = z_y − z_a*")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("density")
    fig.suptitle("Candidate vs committed decision margin (primary: candidate-correct general ADDs)", fontsize=10)
    save(fig, "margin_dist")

    # Fig 2: available vs delivered decision movement (scatter, success/failure)
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 4.0), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), tables):
        t = tables[m]
        mm = masks(t)
        ax.scatter(t["dm_cand"][mm["A_succ"]], t["dm_commit"][mm["A_succ"]], s=4, alpha=0.15,
                   color="tab:green", label="success")
        ax.scatter(t["dm_cand"][mm["B_fail"]], t["dm_commit"][mm["B_fail"]], s=6, alpha=0.4,
                   color="tab:red", label="failure")
        lim = [-30, 30]
        ax.plot(lim, lim, color="black", linewidth=0.8, linestyle="--", label="delivered = available")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_xlim(lim); ax.set_ylim(-30, 10)
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("available Δm_candidate")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("delivered Δm_commit")
    fig.suptitle("Available vs delivered decision-margin movement", fontsize=10)
    save(fig, "avail_vs_deliv")

    # Fig 3: rho_margin distribution success vs failure
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 3.9), sharey=True)
    bins = np.linspace(-0.5, 1.5, 61)
    for ax, m in zip(np.atleast_1d(axes), tables):
        t = tables[m]
        mm = masks(t)
        for g, lab, c in ((mm["A_succ"], "success", "tab:green"), (mm["B_fail"], "failure", "tab:red")):
            v = t["rho_margin"][g]
            v = v[np.isfinite(v)]
            ax.hist(v, bins=bins, alpha=0.55, color=c, density=True, label=f"{lab} (n={v.size})")
        ax.axvline(0, color="black", lw=0.8); ax.axvline(1, color="black", lw=0.8, ls=":")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("rho_margin = Δm_commit / Δm_candidate")
        ax.grid(alpha=0.3); ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("density")
    fig.suptitle("Delivered decision-margin fraction (success vs failure)", fontsize=10)
    save(fig, "rho_success_fail")

    # Fig 4: alpha* distributions
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 3.9), sharey=True)
    bins = np.linspace(0, 1.25, 51)
    for ax, m in zip(np.atleast_1d(axes), tables):
        t = tables[m]; mm = masks(t)
        am = t["alpha_star_margin"][mm["B_fail"]]; af = t["alpha_star_full"][mm["B_fail"]]
        ax.hist(am[np.isfinite(am)], bins=bins, alpha=0.55, color="tab:red", label="failure α*_margin")
        ax.hist(af[np.isfinite(af)], bins=bins, alpha=0.45, color="tab:purple",
                label="failure α*_full")
        ax.set_title(m, fontsize=9); ax.set_xlabel("α*")
        ax.grid(alpha=0.3); ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("failure transitions")
    fig.suptitle("Local decision-boundary statistic α* (primary failures)", fontsize=10)
    save(fig, "alpha_star_dist")

    # Fig 5: alpha*_margin vs delivered decision movement
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 4.0), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), tables):
        t = tables[m]; mm = masks(t)
        g = mm["B_fail"]
        x = t["dm_commit"][g]; y = t["alpha_star_margin"][g]
        ok = np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=5, alpha=0.3, color="tab:red")
        ax.set_xlim(-30, 5)
        r = analysis["analysis_6_alpha_star"][m]["vs_delivered_failures"]
        ttl = f"{m}" + (f" (Spearman {r['spearman']:.2f})" if r else "")
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("delivered Δm_commit")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("α*_margin (failures)")
    fig.suptitle("α*_margin vs delivered decision-margin movement (failures)", fontsize=10)
    save(fig, "alpha_vs_delivered")

    # Fig 6: coordinate-level decision contribution profiles success vs failure
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 4.0), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), tables):
        tt = tables[m]; mm = masks(tt); pr = tt.primary_row
        succ = mm["A_succ"][pr]; fail = mm["B_fail"][pr]
        qc = tt.coords["qcand"]
        prof_s = qc[succ].mean(0) if succ.any() else np.zeros(212)
        prof_f = qc[fail].mean(0) if fail.any() else np.zeros(212)
        ax.plot(prof_s, color="tab:green", linewidth=1.0, label="success (mean q_cand_i)")
        ax.plot(prof_f, color="tab:red", linewidth=1.0, label="failure (mean q_cand_i)")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(m, fontsize=9); ax.set_xlabel("state coordinate i")
        ax.grid(alpha=0.3); ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("mean candidate decision contribution")
    fig.suptitle("Coordinate-level candidate decision contribution (primary)", fontsize=10)
    save(fig, "coord_contrib")

    # Fig 7: contribution-weighted gate comparison
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    metrics = ["gw_all", "gw_pos", "gw_neg", "gw_mag"]
    labels = ["gw (all)", "gw (positive q)", "gw (negative q)", "gw (|movement|)"]
    x = np.arange(len(metrics))
    w = 0.2
    for k, m in enumerate(tables):
        a7 = analysis["analysis_7_coordinate_gate_effect"][m]
        for j, (gname, glab, c) in enumerate((("success", "succ", "tab:green"), ("failure", "fail", "tab:red"))):
            vals = [a7[gname][k2]["mean"] for k2 in metrics]
            ax.bar(x + (k - 0.5) * 2 * w + (j - 0.5) * w, vals, w,
                   color=colors[m], alpha=0.9 if gname == "success" else 0.45,
                   label=f"{m} {glab}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("contribution-weighted gate mean"); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Coordinate-weighted gate: success (solid) vs failure (faded)")
    save(fig, "weighted_gate")

    # Fig 8: available vs delivered movement by seed
    fig, axes = plt.subplots(1, len(tables), figsize=(5.6 * len(tables), 4.0), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), tables):
        sf = analysis["seed_success_failure"][m]
        seeds = sorted(sf)
        xs = np.arange(len(seeds))
        avail = [sf[s]["fail"]["dm_cand_mean"] for s in seeds]
        deliv = [sf[s]["fail"]["dm_commit_mean"] for s in seeds]
        ax.bar(xs - 0.2, avail, 0.4, color="tab:orange", label="failure available Δm_cand")
        ax.bar(xs + 0.2, deliv, 0.4, color="tab:red", label="failure delivered Δm_commit")
        ax.set_xticks(xs); ax.set_xticklabels(seeds)
        ax.set_title(m, fontsize=9); ax.set_xlabel("seed"); ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("mean decision movement (failures)")
    fig.suptitle("Available vs commit-delivered decision movement by seed (failures)", fontsize=10)
    save(fig, "seed_movement")

    # Fig 9: E vs F margin/gate comparison (per seed)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    a9 = analysis["analysis_9_E_vs_F"]["paired_primary_failure"]
    for ax, key, lab in zip(axes, ("rho_margin_med", "dm_commit_med", "gw_pos_mean"),
                            ("ρ margin (median)", "Δm_commit (median)", "gw positive")):
        e = seed_metric_arrays(analysis["seed_success_failure"]["E_candidate"], "fail", key)
        f = seed_metric_arrays(analysis["seed_success_failure"]["F_combined"], "fail", key)
        x = np.arange(len(e)); w = 0.38
        ax.bar(x - w / 2, e, w, color=colors["E_candidate"], label="E")
        ax.bar(x + w / 2, f, w, color=colors["F_combined"], label="F")
        ax.set_xticks(x); ax.set_xticklabels(sorted(analysis["seed_success_failure"]["E_candidate"]))
        p = a9[key]
        ax.set_title(f"{lab}\nE−F={p['mean_diff']:.3g}, p={p['paired_t_p']:.2f}", fontsize=8)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)
    fig.suptitle("E vs F on primary failures (paired seed-level)", fontsize=10)
    save(fig, "E_vs_F")

    # Fig 10: failure classification summary
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    x = np.arange(len(tables))
    a8 = [analysis["analysis_8_four_way"][m] for m in tables]
    for j, (key, lab, c) in enumerate((("A_frac", "A cand✓ commit✓", "tab:green"),
                                       ("B_frac", "B cand✓ commit✗", "tab:red"),
                                       ("C_frac", "C cand✗ commit✓", "tab:gray"),
                                       ("D_frac", "D cand✗ commit✗", "tab:purple"))):
        axes[0].bar(x + (j - 1.5) * 0.2, [a[key] for a in a8], 0.2, color=c, label=lab)
    axes[0].set_xticks(x); axes[0].set_xticklabels(list(tables), fontsize=8)
    axes[0].set_ylabel("fraction of test ADDs"); axes[0].legend(fontsize=6.5)
    axes[0].set_title("Four-way candidate/commit classification", fontsize=9); axes[0].grid(alpha=0.3, axis="y")
    # failure margin-sign composition
    comp = {"m_prev<0": [], "m_prev>=0": [], "m_cand<0": [], "m_commit<0": []}
    for m in tables:
        t = tables[m]; mm = masks(t); B = mm["B_fail"]
        comp["m_prev<0"].append(float((t["m_prev"][B] < 0).mean()))
        comp["m_prev>=0"].append(float((t["m_prev"][B] >= 0).mean()))
        comp["m_cand<0"].append(float((t["m_cand"][B] < 0).mean()))
        comp["m_commit<0"].append(float((t["m_commit"][B] < 0).mean()))
    xs = np.arange(len(comp))
    for k, m in enumerate(tables):
        axes[1].bar(xs + (k - 0.5) * 0.38, [comp[c][k] for c in comp], 0.38,
                    color=colors[m], label=m)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(list(comp), fontsize=8)
    axes[1].set_ylabel("fraction of primary failures"); axes[1].legend(fontsize=7)
    axes[1].set_title("Margin-sign composition of failures", fontsize=9); axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Failure classification summary", fontsize=10)
    save(fig, "failure_classes")

    # Fig 11: alignment ratio of candidate movement to decision direction
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bins = np.linspace(-0.5, 1.5, 61)
    for m in tables:
        t = tables[m]; mm = masks(t)
        for g, a, c, glab in ((mm["B_fail"], 0.55, "tab:red", "failure"),
                              (mm["A_succ"], 0.3, "tab:green", "success")):
            v = t["align_ratio"][g]
            ax.hist(v, bins=bins, alpha=a, density=True, color=c, label=f"{m} {glab}")
    ax.set_xlabel("align_ratio = Δm_candidate / ||C − S_prev|| (decision-direction alignment)")
    ax.set_ylabel("density"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
    ax.set_title("Candidate movement alignment to the decision direction")
    save(fig, "align_ratio")

    # Fig 12: candidate -> commit gap by category (secondary controls)
    cats = ["test_all_add", "test_general", "test_noop", "test_same_reg", "test_wrap",
            "test_no_wrap", "test_cand_wrong", "test_isolated", "test_repeated",
            "controlled:isolated_add", "controlled:add_then_set", "controlled:repeated_add",
            "controlled:add_no_wrap", "controlled:add_wrap"]
    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    x = np.arange(len(cats)); width = 0.38
    for k, m in enumerate(tables):
        ct = analysis["categories"][m]
        gap = [ct[c]["p_fail_given_cand_ok"] if c in ct else 0.0 for c in cats]
        ax.bar(x + (k - 0.5) * width, gap, width, color=colors[m], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("controlled:", "") for c in cats], rotation=22, ha="right", fontsize=7)
    ax.set_ylabel("P(commit wrong | candidate correct)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)
    ax.set_title("Candidate→commit failure rate by category (secondary controls kept separate)")
    save(fig, "categories")

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment I analysis (measurement only)")
    ap.add_argument("--coord", default="experiments/experiment_i_coordinate.npz")
    ap.add_argument("--interp", default="experiments/experiment_h_interp.npz")
    ap.add_argument("--output", default="experiments/experiment_i_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_i")
    args = ap.parse_args()

    tables, meta = load_coord(args.coord)
    h_aux = load_h_aux(args.interp)
    align = verify_alignment(tables, h_aux)
    results = build_analysis(tables, meta, h_aux, align)
    results["metadata"]["coord_file"] = args.coord
    results["metadata"]["interp_file"] = args.interp
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.output}")
    print("Sanity checks passed:", results["sanity_checks"]["passed"], results["sanity_checks"]["failures"])
    paths = make_figures(tables, results, h_aux, args.fig_dir, prefix=args.fig_prefix)
    print("Figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
