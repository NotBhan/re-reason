#!/usr/bin/env python3
"""Experiment H — analysis of the λ-interpolation decode sweep (measurement only).

Reads experiments/experiment_h_interp.npz (runner output) and produces:

  * Sanity checks: λ=0 → previous-state decode; λ=1 → candidate decode; vector-gate replay
    reproduces the recorded committed state and decode; checkpoint hashes unchanged.
  * Accuracy vs λ curves (pooled, per seed, wrap/no-wrap, conditional on committed outcome,
    conditional on old-value / third-value decode).
  * λ* statistics (first correct λ along the scalar path) and their relations to
    ||C−S_prev||, the actual mean gate, and actual commit correctness.
  * Failure categorization for candidate-correct/committed-wrong transitions, using the
    ACTUAL vector-gate replay as reference and the scalar path for decodability; comparisons
    against λ_eff (projection of the actual committed state onto the S_prev→C segment).
  * Decode-trajectory classes and near-miss analysis.
  * Category-level gap quantification (candidate / committed / replay / mean-gate / λ=1).
  * Explicit hypothesis evaluations H1–H5 and seed-level paired E vs F inference.

Statistical discipline: transition-level tables are descriptive (transitions within a
sequence are dependent); independent inference is seed-level (n=5, paired E vs F).

Reproduce:
    python experiments/analyze_experiment_h.py \
        --interp experiments/experiment_h_interp.npz \
        --output experiments/experiment_h_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_h
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

from run_experiment_g import MODELS, SOURCE_IDS, SOURCE_NAMES  # noqa: E402

N_REG = 4
N_VAL = 10


class HTable:
    def __init__(self, scalars: np.ndarray, cols: List[str], digits: np.ndarray,
                 correct: np.ndarray, lambdas: np.ndarray) -> None:
        self.scalars = scalars
        self.index = {c: i for i, c in enumerate(cols)}
        self.digits = digits
        self.correct = correct.astype(bool)
        self.lambdas = lambdas
        # λ* computed directly from the correctness matrix (the stored column is a
        # redundant record and is cross-checked in sanity_checks).
        has = self.correct.any(axis=1)
        first = np.argmax(self.correct, axis=1)
        self.lambda_star_calc = np.where(has, self.lambdas[first], np.nan)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.scalars[:, self.index[name]]

    def __len__(self) -> int:
        return len(self.scalars)


def load_interp(path: str) -> Tuple[Dict[str, HTable], Dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    lambdas = np.asarray(meta["lambdas"], dtype=np.float64)
    tables = {}
    for m in MODELS:
        key = f"{m}__scalars"
        if key in z:
            tables[m] = HTable(z[key], meta["scalar_cols"], z[f"{m}__digits"], z[f"{m}__correct"], lambdas)
    return tables, meta


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def masks(t: HTable) -> Dict[str, np.ndarray]:
    src = t["source"]
    test = src == SOURCE_IDS["test"]
    src0 = t["operand_b"] == 0
    same_reg = (t["dest"] == t["src_reg"]) & ~src0
    general = (t["dest"] != t["src_reg"]) & ~src0
    cand = t["cand_ok"] > 0.5
    return {
        "test": test,
        "test_cand": test & cand,
        "general": test & general,
        "general_cand": test & general & cand,
        "general_cand_fail": test & general & cand & (t["comm_ok"] < 0.5),
        "noop_cand": test & src0 & cand,
        "same_reg_cand": test & same_reg & cand,
        "cand_wrong": test & ~cand,
        "test_wrap": test & (t["wrap"] > 0.5),
        "test_no_wrap": test & (t["wrap"] < 0.5),
        "general_cand_fail_old": test & general & cand & (t["comm_ok"] < 0.5) & (t["comm_eq_old"] > 0.5),
        "general_cand_fail_third": test & general & cand & (t["comm_ok"] < 0.5) & (t["comm_eq_old"] < 0.5),
    }


def acc_curve(t: HTable, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.full(len(t.lambdas), np.nan)
    return t.correct[mask].mean(axis=0)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_checks(tables: Dict[str, HTable], meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"models": {}, "hashes_unchanged": {}, "passed": True}

    def record(ok: bool, name: str) -> None:
        if not ok:
            out["passed"] = False
        out.setdefault("failures", [])
        if not ok:
            out["failures"].append(name)

    for m, t in tables.items():
        lam0 = t.digits[:, 0]
        lam1 = t.digits[:, -1]
        rec = {
            "lambda0_eq_prev_decode": float((lam0 == np.round(t["prev_pred"])).mean()),
            "lambda1_eq_cand_decode": float((lam1 == np.round(t["cand_pred"])).mean()),
            "replay_pred_eq_committed_decode": float((np.round(t["replay_pred"]) == np.round(t["comm_pred"])).mean()),
            "replay_vs_recorded_maxdiff": float(t["replay_maxdiff"].max()),
            "replay_vs_recorded_mean_absdiff": float(t["replay_maxdiff"].mean()),
            "n": int(len(t)),
        }
        record(rec["lambda0_eq_prev_decode"] == 1.0, f"{m}:lambda0")
        record(rec["lambda1_eq_cand_decode"] == 1.0, f"{m}:lambda1")
        record(rec["replay_pred_eq_committed_decode"] == 1.0, f"{m}:replay_decode")
        record(rec["replay_vs_recorded_maxdiff"] < 1e-4, f"{m}:replay_state")
        # λ=1 correctness equals candidate correctness by construction
        rec["lambda1_correct_eq_cand_ok"] = float((t.correct[:, -1] == (t["cand_ok"] > 0.5)).mean())
        record(rec["lambda1_correct_eq_cand_ok"] == 1.0, f"{m}:lambda1_cand")
        # stored λ* column must equal the value recomputed from the correctness matrix
        stored = t["lambda_star"].astype(np.float64)
        calc = t.lambda_star_calc
        both_nan = np.isnan(stored) & np.isnan(calc)
        diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(stored) - np.nan_to_num(calc)))
        rec["lambda_star_stored_vs_computed_maxdiff"] = float(diff.max()) if len(diff) else 0.0
        record(rec["lambda_star_stored_vs_computed_maxdiff"] < 1e-6, f"{m}:lambda_star_consistency")
        out["models"][m] = rec
    for m in MODELS:
        for seed, info in meta["models"][m]["checkpoints"].items():
            out["hashes_unchanged"][f"{m}__seed{seed}"] = bool(info.get("unchanged_after_run"))
    out["passed"] = out["passed"] and all(out["hashes_unchanged"].values())
    return out


# ---------------------------------------------------------------------------
# λ* statistics
# ---------------------------------------------------------------------------

QUANTILES = [5, 25, 50, 75, 95]


def lambda_star_stats(t: HTable, mask: np.ndarray) -> Dict[str, Any]:
    ls = t.lambda_star_calc[mask]
    valid = ~np.isnan(ls)
    if len(ls) == 0:
        return {"n": 0}
    q = np.percentile(ls[valid], QUANTILES) if valid.any() else np.full(5, np.nan)
    return {
        "n": int(len(ls)),
        "n_valid": int(valid.sum()),
        "frac_valid": float(valid.mean()),
        "frac_no_solution": float(1.0 - valid.mean()),
        "mean": float(np.nanmean(ls)) if valid.any() else None,
        "quantiles": {f"p{p}": float(x) for p, x in zip(QUANTILES, q)},
        "frac_le_0.5": float((ls[valid] <= 0.5 + 1e-9).mean()) if valid.any() else None,
        "frac_eq_1.0": float((ls[valid] >= 1.0 - 1e-9).mean()) if valid.any() else None,
    }


def lambda_star_vs_feature(t: HTable, mask: np.ndarray, feature: str, n_bins: int = 10) -> Dict[str, Any]:
    ls = t.lambda_star_calc[mask]
    x = t[feature][mask]
    valid = ~np.isnan(ls)
    if valid.sum() < 10:
        return {"n": int(valid.sum())}
    xv, lv = x[valid], ls[valid]
    rho = float(st.spearmanr(xv, lv).statistic)
    order = np.argsort(xv)
    bins = np.array_split(order, n_bins)
    table = []
    for b in bins:
        table.append({"x_mean": float(xv[b].mean()),
                      "lambda_star_mean": float(lv[b].mean()),
                      "lambda_star_median": float(np.median(lv[b])),
                      "n": int(len(b))})
    return {"n": int(valid.sum()), "spearman": rho, "bins": table}


def failure_vs_actual(t: HTable, mask: np.ndarray) -> Dict[str, Any]:
    """For candidate-correct/committed-wrong transitions: is λ* <= λ_eff / <= f̄ ?"""
    ls = t.lambda_star_calc[mask]
    le = t["lambda_eff"][mask]
    fm = t["f_mean"][mask]
    valid = ~np.isnan(ls)
    n = int(len(ls))
    return {
        "n": n,
        "frac_lambda_star_le_lambda_eff": float((ls[valid] <= le[valid] + 1e-9).mean()) if valid.any() else None,
        "frac_lambda_star_le_f_mean": float((ls[valid] <= fm[valid] + 1e-9).mean()) if valid.any() else None,
        "lambda_eff_mean": float(le.mean()), "lambda_eff_p25": float(np.percentile(le, 25)),
        "lambda_eff_p75": float(np.percentile(le, 75)), "lambda_eff_frac_in_01": float(((le >= 0) & (le <= 1)).mean()),
        "offpath_dist_mean": float(t["offpath_dist"][mask].mean()),
        "offpath_dist_median": float(np.median(t["offpath_dist"][mask])),
        "mean_gate_correct_rate": float((t["mean_ok"][mask] > 0.5).mean()),
        "lambda_star_p50": float(np.median(ls[valid])) if valid.any() else None,
    }


# ---------------------------------------------------------------------------
# Trajectory classes and near-miss analysis
# ---------------------------------------------------------------------------

def trajectory_classes(t: HTable, mask: np.ndarray) -> Dict[str, Any]:
    digits = t.digits[mask]
    correct = t.correct[mask]
    old = np.round(t["operand_a"][mask]).astype(int)
    counts = {"immediately_correct": 0, "wrong_then_correct_monotone": 0,
              "correct_then_reverts": 0, "never_correct": 0}
    sub = {"starts_old": 0, "starts_other": 0}
    examples: Dict[str, List[int]] = {}
    rows = np.nonzero(mask)[0]
    for i in range(len(digits)):
        cor = correct[i]
        if not cor.any():
            counts["never_correct"] += 1
            examples.setdefault("never_correct", []).append(int(rows[i]))
            continue
        first = int(np.argmax(cor))
        if first == 0:
            counts["immediately_correct"] += 1
            examples.setdefault("immediately_correct", []).append(int(rows[i]))
            continue
        if cor[first:].all():
            counts["wrong_then_correct_monotone"] += 1
            if digits[i, 0] == old[i]:
                sub["starts_old"] += 1
            else:
                sub["starts_other"] += 1
            examples.setdefault("wrong_then_correct_monotone", []).append(int(rows[i]))
        else:
            counts["correct_then_reverts"] += 1
            examples.setdefault("correct_then_reverts", []).append(int(rows[i]))
    return {"counts": counts, "wrong_then_correct_sub": sub,
            "n": int(mask.sum()),
            "examples": {k: v[:5] for k, v in examples.items()}}


def near_miss_analysis(t: HTable, mask: np.ndarray) -> Dict[str, Any]:
    """Classify wrong λ-slots relative to the continuous interpolation path in Z_10."""
    lam = t.lambdas[None, :]
    old = np.round(t["operand_a"][mask]).astype(float)
    ob = np.round(t["operand_b"][mask]).astype(float)
    new = (old + ob) % N_VAL
    digits = t.digits[mask].astype(float)
    correct = t.correct[mask]
    p = (old[:, None] + lam * ob[:, None]) % N_VAL
    cd_path = np.minimum(np.abs(digits - p) % N_VAL, N_VAL - np.abs(digits - p) % N_VAL)
    wrong = ~correct
    is_old = digits == old[:, None]
    is_new = digits == new[:, None]
    near_path = (cd_path <= 1.0) & ~is_new
    classes = {
        "correct": int(correct.sum()),
        "old": int((wrong & is_old).sum()),
        "near_path": int((wrong & ~is_old & near_path).sum()),
        "other": int((wrong & ~is_old & ~near_path).sum()),
    }
    total_wrong = sum(v for k, v in classes.items() if k != "correct")
    # last wrong slot before the first correct λ (or last slot if none)
    last_wrong_digit = np.full(len(digits), np.nan)
    last_wrong_lam = np.full(len(digits), np.nan)
    for i in range(len(digits)):
        cor = correct[i]
        if cor.all():
            continue
        if cor.any():
            first = int(np.argmax(cor))
            j = first - 1
        else:
            j = len(cor) - 1
        last_wrong_digit[i] = digits[i, j]
        last_wrong_lam[i] = t.lambdas[j]
    lw = last_wrong_digit[~np.isnan(last_wrong_digit)]
    lwn = (np.round(lw) - new[~np.isnan(last_wrong_digit)]) % N_VAL
    cd_new = np.minimum(lwn % N_VAL, N_VAL - lwn % N_VAL)
    out = {
        "n": int(mask.sum()),
        "slot_classes": classes,
        "wrong_slot_shares": {k: (v / total_wrong if total_wrong else None) for k, v in classes.items() if k != "correct"},
        "last_wrong_before_correct": {
            "n": int(len(lw)),
            "frac_adjacent_to_new": float((cd_new == 1).mean()) if len(lw) else None,
            "frac_equal_old": float((lw == old[~np.isnan(last_wrong_digit)]).mean()) if len(lw) else None,
            "frac_one_step_from_old": float((cd_new == 2).mean()) if len(lw) else None,
            "mean_last_wrong_lambda": float(np.mean(last_wrong_lam[~np.isnan(last_wrong_lam)]))
            if np.any(~np.isnan(last_wrong_lam)) else None,
        },
    }
    return out


# ---------------------------------------------------------------------------
# Replay vs scalar counterfactual
# ---------------------------------------------------------------------------

def replay_vs_scalar(t: HTable, mask: np.ndarray) -> Dict[str, Any]:
    def rate(col: str) -> float:
        return float((t[col][mask] > 0.5).mean())
    out = {
        "n": int(mask.sum()),
        "candidate_acc": rate("cand_ok"),
        "recorded_commit_acc": rate("comm_ok"),
        "vector_gate_replay_acc": rate("replay_ok"),
        "mean_gate_counterfactual_acc": rate("mean_ok"),
        "lambda1_acc": float(t.correct[mask][:, -1].mean()),
        "lambda0_acc": float(t.correct[mask][:, 0].mean()),
        "replay_decode_agreement_with_recorded": float(
            (np.round(t["replay_pred"][mask]) == np.round(t["comm_pred"][mask])).mean()),
        "mean_gate_decode_agreement_with_replay": float(
            (np.round(t["mean_pred"][mask]) == np.round(t["replay_pred"][mask])).mean()),
        "max_replay_state_diff": float(t["replay_maxdiff"][mask].max()),
    }
    return out


# ---------------------------------------------------------------------------
# Category-level gap quantification
# ---------------------------------------------------------------------------

def category_masks(t: HTable) -> Dict[str, np.ndarray]:
    m = masks(t)
    out = {
        "test_all": m["test"],
        "test_general": m["general"],
        "test_isolated": m["test"] & (t["isolated"] > 0.5),
        "test_repeated": m["test"] & (t["isolated"] < 0.5),
        "test_wrap": m["test_wrap"],
        "test_no_wrap": m["test_no_wrap"],
    }
    for name, sid in SOURCE_IDS.items():
        if sid == 0:
            continue
        out[f"controlled:{name}"] = t["source"] == sid
    return out


def category_table(t: HTable) -> Dict[str, Any]:
    out = {}
    for name, mask in category_masks(t).items():
        if not mask.any():
            continue
        entry = replay_vs_scalar(t, mask)
        entry["any_lambda_acc"] = float(t.correct[mask].any(axis=1).mean())
        entry["actual_candidate_commit_gap"] = entry["candidate_acc"] - entry["recorded_commit_acc"]
        entry["scalar_oracle_gap_vs_commit"] = entry["any_lambda_acc"] - entry["recorded_commit_acc"]
        entry["scalar_mean_gate_gap_vs_replay"] = entry["vector_gate_replay_acc"] - entry["mean_gate_counterfactual_acc"]
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Failure categorization (candidate-correct / committed-wrong)
# ---------------------------------------------------------------------------

def failure_categorization(t: HTable, mask: np.ndarray) -> Dict[str, Any]:
    n = int(mask.sum())
    replay_ok = t["replay_ok"][mask] > 0.5
    ls = t.lambda_star_calc[mask]
    le = t["lambda_eff"][mask]
    fm = t["f_mean"][mask]
    has_star = ~np.isnan(ls)
    a = int(replay_ok.sum())
    b = int(((~replay_ok) & has_star).sum())
    c = int(((~replay_ok) & ~has_star).sum())
    b_mask = (~replay_ok) & has_star
    out = {
        "n": n,
        "A_actual_commit_correct": a, "A_pct": 100.0 * a / max(n, 1),
        "B_actual_wrong_scalar_correct_exists": b, "B_pct": 100.0 * b / max(n, 1),
        "C_no_scalar_correct": c, "C_pct": 100.0 * c / max(n, 1),
        "B_below_f_mean": int((b_mask & (ls <= fm + 1e-9)).sum()),
        "B_above_f_mean": int((b_mask & (ls > fm + 1e-9)).sum()),
        "B_below_lambda_eff": int((b_mask & (ls <= le + 1e-9)).sum()),
        "B_above_lambda_eff": int((b_mask & (ls > le + 1e-9)).sum()),
        "B_mean_gate_correct": int((b_mask & (t["mean_ok"][mask] > 0.5)).sum()),
        "B_lambda1_correct": int((b_mask & t.correct[mask][:, -1]).sum()),
    }
    return out


# ---------------------------------------------------------------------------
# Seed-level inference
# ---------------------------------------------------------------------------

def seed_level_stats(t: HTable, m: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for seed in sorted(set(int(s) for s in t["seed"][m["test"]])):
        seed_mask = t["seed"] == seed
        pm = m["general_cand"] & seed_mask
        pf = m["general_cand_fail"] & seed_mask
        ls = t.lambda_star_calc[pf]
        valid = ~np.isnan(ls)
        out[str(seed)] = {
            "n_primary": int(pm.sum()),
            "n_failures": int(pf.sum()),
            "committed_acc_primary": float((t["comm_ok"][pm] > 0.5).mean()),
            "replay_acc_primary": float((t["replay_ok"][pm] > 0.5).mean()),
            "mean_gate_acc_primary": float((t["mean_ok"][pm] > 0.5).mean()),
            "lambda_star_mean_failures": float(ls[valid].mean()) if valid.any() else None,
            "lambda_star_median_failures": float(np.median(ls[valid])) if valid.any() else None,
            "frac_failures_star_le_lambda_eff": float((ls[valid] <= t["lambda_eff"][pf][valid] + 1e-9).mean())
            if valid.any() else None,
            "frac_failures_star_le_fmean": float((ls[valid] <= t["f_mean"][pf][valid] + 1e-9).mean())
            if valid.any() else None,
            "acc_at_lambda_0": float(t.correct[pm][:, 0].mean()) if pm.any() else None,
            "acc_at_lambda_0.5": float(t.correct[pm][:, 10].mean()) if pm.any() else None,
            "acc_at_lambda_1": float(t.correct[pm][:, -1].mean()) if pm.any() else None,
        }
    return out


def paired_seed_stats(se: Dict[str, Dict[str, Any]], sf: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    keys = [k for k in next(iter(se.values())) if k != "n_primary" and k != "n_failures"]
    for k in keys:
        a = np.array([se[s][k] for s in sorted(se) if se[s][k] is not None], dtype=float)
        b = np.array([sf[s][k] for s in sorted(sf) if sf[s][k] is not None], dtype=float)
        if len(a) != len(b) or len(a) < 2 or np.isnan(a).any() or np.isnan(b).any():
            continue
        d = a - b
        t_stat, p_t = st.ttest_rel(a, b)
        try:
            _, p_w = st.wilcoxon(a, b)
        except ValueError:
            p_w = float("nan")
        out[k] = {"E_mean": float(a.mean()), "E_sd": float(a.std(ddof=1)),
                  "F_mean": float(b.mean()), "F_sd": float(b.std(ddof=1)),
                  "E_minus_F": float(d.mean()), "paired_t_p": float(p_t),
                  "wilcoxon_p": float(p_w)}
    return out


# ---------------------------------------------------------------------------
# Hypothesis evaluations
# ---------------------------------------------------------------------------

def hypothesis_evaluations(tables: Dict[str, HTable], analysis: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m, t in tables.items():
        mm = masks(t)
        curves = {
            "primary": acc_curve(t, mm["general_cand"]),
            "all_test": acc_curve(t, mm["test"]),
            "failures": acc_curve(t, mm["general_cand_fail"]),
        }
        prim = curves["primary"]
        drops = np.diff(prim)
        interior = int(np.argmax(curves["all_test"]))
        cat = analysis["failure_categorization"][m]
        rv = analysis["replay_vs_scalar"][m]["test_general"]
        traj = analysis["trajectories"][m]["general_cand_fail"]
        out[m] = {
            "H1_primary_curve_final": float(prim[-1]),
            "H1_max_consecutive_drop": float(drops.min()),
            "H1_monotone_violations": int((drops < -1e-9).sum()),
            "H2_all_curve_argmax_lambda": float(t.lambdas[interior]),
            "H2_all_curve_max": float(np.nanmax(curves["all_test"])),
            "H2_all_curve_final": float(curves["all_test"][-1]),
            "H2_revert_transition_rate": float(traj["counts"]["correct_then_reverts"] / max(traj["n"], 1)),
            "H3_failures_with_lambda_star": int(cat["B_actual_wrong_scalar_correct_exists"]),
            "H3_frac_star_le_lambda_eff": analysis["failure_vs_actual"][m]["frac_lambda_star_le_lambda_eff"],
            "H3_frac_star_le_fmean": analysis["failure_vs_actual"][m]["frac_lambda_star_le_f_mean"],
            "H4_no_solution_primary_frac": analysis["lambda_star"][m]["general_cand"]["frac_no_solution"],
            "H4_cand_wrong_no_solution_frac": analysis["lambda_star"][m]["cand_wrong"]["frac_no_solution"],
            "H5_replay_vs_mean_gate_acc_diff": rv["vector_gate_replay_acc"] - rv["mean_gate_counterfactual_acc"],
            "H5_decode_disagreement_rate": 1.0 - rv["mean_gate_decode_agreement_with_replay"],
            "H5_offpath_dist_mean": analysis["failure_vs_actual"][m]["offpath_dist_mean"],
        }
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_analysis(tables: Dict[str, HTable], meta: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runner_meta": meta,
            "statistical_discipline_note": (
                "Transition-level tables are descriptive (transitions within a sequence are "
                "dependent); independent inference is seed-level (n=5, paired E vs F). λ* is an "
                "empirical decodable-point statistic along the scalar interpolation path, not a "
                "causal threshold. S(f̄_t) is a counterfactual scalarization; the actual commit "
                "uses the 212-dim element-wise gate."),
        },
        "sanity_checks": sanity_checks(tables, meta),
        "curves": {}, "lambda_star": {}, "failure_categorization": {}, "failure_vs_actual": {},
        "trajectories": {}, "near_miss": {}, "replay_vs_scalar": {}, "categories": {},
        "seed_level": {}, "hypotheses": {},
    }
    for m, t in tables.items():
        mm = masks(t)
        res["curves"][m] = {
            "lambdas": [float(x) for x in t.lambdas],
            "primary": [float(x) for x in acc_curve(t, mm["general_cand"])],
            "primary_wrap": [float(x) for x in acc_curve(t, mm["general_cand"] & mm["test_wrap"])],
            "primary_no_wrap": [float(x) for x in acc_curve(t, mm["general_cand"] & mm["test_no_wrap"])],
            "all_test": [float(x) for x in acc_curve(t, mm["test"])],
            "all_test_wrap": [float(x) for x in acc_curve(t, mm["test_wrap"])],
            "all_test_no_wrap": [float(x) for x in acc_curve(t, mm["test_no_wrap"])],
            "failures": [float(x) for x in acc_curve(t, mm["general_cand_fail"])],
            "failures_old": [float(x) for x in acc_curve(t, mm["general_cand_fail_old"])],
            "failures_third": [float(x) for x in acc_curve(t, mm["general_cand_fail_third"])],
            "noop_cand": [float(x) for x in acc_curve(t, mm["noop_cand"])],
            "same_reg_cand": [float(x) for x in acc_curve(t, mm["same_reg_cand"])],
        }
        res["lambda_star"][m] = {
            "general_cand": lambda_star_stats(t, mm["general_cand"]),
            "general_cand_fail": lambda_star_stats(t, mm["general_cand_fail"]),
            "general_cand_fail_wrap": lambda_star_stats(t, mm["general_cand_fail"] & mm["test_wrap"]),
            "general_cand_fail_no_wrap": lambda_star_stats(t, mm["general_cand_fail"] & mm["test_no_wrap"]),
            "noop_cand": lambda_star_stats(t, mm["noop_cand"]),
            "same_reg_cand": lambda_star_stats(t, mm["same_reg_cand"]),
            "cand_wrong": lambda_star_stats(t, mm["cand_wrong"]),
            "vs_dcs_failures": lambda_star_vs_feature(t, mm["general_cand_fail"], "d_cs"),
            "vs_fmean_failures": lambda_star_vs_feature(t, mm["general_cand_fail"], "f_mean"),
            "vs_dcs_primary": lambda_star_vs_feature(t, mm["general_cand"], "d_cs"),
        }
        res["failure_categorization"][m] = failure_categorization(t, mm["general_cand_fail"])
        res["failure_vs_actual"][m] = failure_vs_actual(t, mm["general_cand_fail"])
        res["trajectories"][m] = {
            "general_cand_fail": trajectory_classes(t, mm["general_cand_fail"]),
            "general_cand": trajectory_classes(t, mm["general_cand"]),
        }
        res["near_miss"][m] = near_miss_analysis(t, mm["general_cand_fail"])
        res["replay_vs_scalar"][m] = {
            "test_all": replay_vs_scalar(t, mm["test"]),
            "test_general": replay_vs_scalar(t, mm["general"]),
            "general_cand": replay_vs_scalar(t, mm["general_cand"]),
            "general_cand_fail": replay_vs_scalar(t, mm["general_cand_fail"]),
        }
        res["categories"][m] = category_table(t)
        res["seed_level"][m] = seed_level_stats(t, mm)
    res["seed_paired_E_vs_F"] = paired_seed_stats(res["seed_level"]["E_candidate"],
                                                  res["seed_level"]["F_combined"])
    res["hypotheses"] = hypothesis_evaluations(tables, res)
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(tables: Dict[str, HTable], analysis: Dict[str, Any], fig_dir: str,
                 prefix: str = "experiment_h") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    paths: List[str] = []
    lam = None
    for t in tables.values():
        lam = t.lambdas
        break
    colors = {"E_candidate": "tab:blue", "F_combined": "tab:brown"}

    def save(fig, name):
        p = os.path.join(fig_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # Fig 1: accuracy vs λ, pooled E/F (primary + all-test)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for m in tables:
        c = analysis["curves"][m]
        ax.plot(lam, c["primary"], marker="o", ms=3, color=colors[m], label=f"{m}: primary (cand-correct general)")
        ax.plot(lam, c["all_test"], linestyle="--", marker=".", ms=2, color=colors[m], alpha=0.7,
                label=f"{m}: all test ADDs")
        ax.axhline(c["primary"][-1], color=colors[m], alpha=0.25, linewidth=0.8)
    ax.set_xlabel("λ (scalar interpolation S(λ) = (1−λ)S_prev + λC)")
    ax.set_ylabel("destination decode accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5)
    ax.set_title("Decode accuracy vs λ (dashed horizontal = candidate accuracy)")
    save(fig, "acc_vs_lambda")

    # Fig 2: wrap vs no-wrap (primary), two panels
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    for ax, m in zip(axes, tables):
        c = analysis["curves"][m]
        ax.plot(lam, c["primary_no_wrap"], marker="o", ms=3, label="no wrap", color="tab:green")
        ax.plot(lam, c["primary_wrap"], marker="s", ms=3, label="wrap", color="tab:red")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("λ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("destination decode accuracy")
    fig.suptitle("Accuracy vs λ by wrap status (candidate-correct general ADDs)", fontsize=10)
    save(fig, "acc_vs_lambda_wrap")

    # Fig 3: failures only (old-value vs third-value decode)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    for ax, m in zip(axes, tables):
        c = analysis["curves"][m]
        ax.plot(lam, c["failures"], marker="o", ms=3, label="all failures", color="tab:purple")
        ax.plot(lam, c["failures_old"], linestyle="--", label="failure decoded to old value", color="tab:gray")
        ax.plot(lam, c["failures_third"], linestyle=":", label="failure decoded to third value", color="tab:red")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("λ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel("destination decode accuracy")
    fig.suptitle("Accuracy vs λ for candidate-correct / committed-wrong ADD transitions", fontsize=10)
    save(fig, "acc_vs_lambda_failures")

    # Fig 4: λ* distribution
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bins = np.arange(-0.025, 1.05, 0.05)
    for m in tables:
        ls = None
        t = tables[m]
        mm = masks(t)
        ls = t.lambda_star_calc[mm["general_cand_fail"]]
        ls = ls[~np.isnan(ls)]
        ax.hist(ls, bins=bins, alpha=0.55, color=colors[m], label=f"{m} (n={len(ls)})")
    ax.set_xlabel("λ* (first λ with correct destination decode)")
    ax.set_ylabel("failure transitions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("λ* distribution (candidate-correct general ADD failures; no-solution = 0)")
    save(fig, "lambda_star_dist")

    # Fig 5: λ* vs ||C-S_prev||
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    for ax, m in zip(axes, tables):
        t = tables[m]
        mm = masks(t)
        x = t["d_cs"][mm["general_cand_fail"]]
        y = t.lambda_star_calc[mm["general_cand_fail"]]
        ax.scatter(x, y, s=4, alpha=0.1, color=colors[m])
        vb = analysis["lambda_star"][m]["vs_dcs_failures"]["bins"]
        ax.plot([b["x_mean"] for b in vb], [b["lambda_star_mean"] for b in vb],
                color="black", marker="o", linewidth=2, label="decile mean")
        ax.set_title(f"{m} (Spearman {analysis['lambda_star'][m]['vs_dcs_failures']['spearman']:.2f})", fontsize=9)
        ax.set_xlabel("||C − S_prev||")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("λ*")
    fig.suptitle("λ* vs candidate displacement (failures only)", fontsize=10)
    save(fig, "lambda_star_vs_dcs")

    # Fig 6: λ* vs actual mean gate
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    for ax, m in zip(axes, tables):
        t = tables[m]
        mm = masks(t)
        x = t["f_mean"][mm["general_cand_fail"]]
        y = t.lambda_star_calc[mm["general_cand_fail"]]
        ax.scatter(x, y, s=4, alpha=0.1, color=colors[m])
        vb = analysis["lambda_star"][m]["vs_fmean_failures"]["bins"]
        ax.plot([b["x_mean"] for b in vb], [b["lambda_star_mean"] for b in vb],
                color="black", marker="o", linewidth=2, label="decile mean")
        ax.set_title(f"{m} (Spearman {analysis['lambda_star'][m]['vs_fmean_failures']['spearman']:.2f})", fontsize=9)
        ax.set_xlabel("actual full-vector mean gate f̄")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("λ*")
    fig.suptitle("λ* vs actual mean gate (failures only; scalar projection only)", fontsize=10)
    save(fig, "lambda_star_vs_gate")

    # Fig 7: vector-gate replay vs scalar counterfactual
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    metrics = ["candidate_acc", "recorded_commit_acc", "vector_gate_replay_acc",
               "mean_gate_counterfactual_acc", "lambda1_acc", "lambda0_acc"]
    labels = ["candidate\nhead(C)", "recorded commit\nhead(S_t)", "vector replay\nhead((1−f)⊙S+C)",
              "mean-gate scalar\nhead((1−f̄)S+f̄C)", "λ=1", "λ=0"]
    x = np.arange(len(metrics))
    width = 0.38
    for k, m in enumerate(tables):
        vals = [analysis["replay_vs_scalar"][m]["general_cand"][k2] for k2 in metrics]
        ax.bar(x + (k - 0.5) * width, vals, width, color=colors[m], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("destination accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    ax.set_title("Candidate / actual commit / replay / scalar counterfactual (candidate-correct general ADDs)")
    save(fig, "replay_vs_scalar")

    # Fig 8: trajectory classes + examples
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    classes = ["wrong_then_correct_monotone", "correct_then_reverts", "never_correct", "immediately_correct"]
    x = np.arange(len(classes))
    for k, m in enumerate(tables):
        tr = analysis["trajectories"][m]["general_cand_fail"]
        n = max(tr["n"], 1)
        vals = [tr["counts"][c] / n for c in classes]
        axes[0].bar(x + (k - 0.5) * 0.38, vals, 0.38, color=colors[m], label=m)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["wrong→correct\n(monotone)", "correct→reverts", "never correct", "immediate"], fontsize=7)
    axes[0].set_ylabel("fraction of failures")
    axes[0].set_title("Trajectory classes (failures)", fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)
    # examples from E: pick up to 5 wrong_then_correct examples and 2 reverts if any
    tE = tables["E_candidate"]
    mmE = masks(tE)
    trE = trajectory_classes(tE, mmE["general_cand_fail"])
    ex_rows = []
    for cls in ("wrong_then_correct_monotone", "correct_then_reverts", "never_correct"):
        for r in trE["examples"].get(cls, [])[:4]:
            ex_rows.append((cls, r))
    ex_rows = ex_rows[:6]
    for cls, r in ex_rows:
        traj = tE.digits[r].astype(float)
        axes[1].plot(lam, traj, marker="o", ms=2.5, linewidth=1,
                     label=f"row {r} [{cls[:14]}]")
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("decoded destination digit")
    axes[1].set_title("Example failure trajectories (E_candidate)", fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=5.5)
    fig.suptitle("Decode trajectories along the scalar interpolation path", fontsize=10)
    save(fig, "trajectories")

    # Fig 9: category gap quantification
    cats = ["test_all", "test_general", "test_isolated", "test_repeated", "test_wrap", "test_no_wrap",
            "controlled:isolated_add", "controlled:add_no_wrap", "controlled:add_wrap", "controlled:repeated_add"]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True)
    metrics2 = ["candidate_acc", "recorded_commit_acc", "vector_gate_replay_acc",
                "mean_gate_counterfactual_acc", "lambda1_acc"]
    labels2 = ["candidate", "recorded commit", "vector replay", "mean-gate scalar", "λ=1"]
    x = np.arange(len(cats))
    width = 0.15
    for ax, m in zip(axes, tables):
        ct = analysis["categories"][m]
        for j, (k2, lab) in enumerate(zip(metrics2, labels2)):
            vals = [ct[c][k2] for c in cats]
            ax.bar(x + (j - 2) * width, vals, width, label=lab)
        ax.set_ylabel("accuracy")
        ax.set_title(m, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7, ncol=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace("controlled:", "") for c in cats], rotation=18, ha="right", fontsize=7)
    fig.suptitle("Candidate vs commit vs replay vs scalar counterfactuals by ADD category (transition-level)", fontsize=10)
    save(fig, "category_gap")

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment H analysis (measurement only)")
    ap.add_argument("--interp", default="experiments/experiment_h_interp.npz")
    ap.add_argument("--output", default="experiments/experiment_h_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_h")
    args = ap.parse_args()

    tables, meta = load_interp(args.interp)
    results = build_analysis(tables, meta)
    results["metadata"]["interp_file"] = args.interp
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.output}")
    print("Sanity checks passed:", results["sanity_checks"]["passed"])
    fig_paths = make_figures(tables, results, args.fig_dir, prefix=args.fig_prefix)
    print("Figures:")
    for p in fig_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
