#!/usr/bin/env python3
"""Experiment G — analysis: localize the ADD candidate->commit gap (measurement only).

Reads experiments/experiment_g_transitions.npz (produced by run_experiment_g.py) and performs:

  Analysis 1  Gate-dilution decomposition. ADD transitions are pooled over seeds and binned by
              the full-vector mean gate f_t (fixed 0.1 grid; sparse bins are merged by a
              deterministic rule). Per bin: counts, candidate/committed/previous-state
              accuracy, the candidate->commit gap, displacement geometry, wrap fraction, and
              the conditionals P(commit correct | candidate correct, bin) and
              P(previous state correct | candidate correct AND committed wrong, bin).

  Analysis 2  Oracle-commit upper bound (analysis only, no rollout, no model change):
              decode head(C_t) instead of head(S_t) at ADD transitions, per ADD category
              (isolated / ADD->SET / 4x repeated / no-wrap / wrap and test-split variants).

  Analysis 3  Predictability of (candidate correct AND committed wrong) from f_t and
              ||C_t - S_{t-1}||: Spearman/point-biserial correlations, a transparent L2
              logistic regression, grouped 5-fold cross-validation (groups = sequences, so
              transitions within a sequence never straddle folds), pooled ROC-AUC / PR-AUC,
              a grouped-bootstrap CI for the AUC difference of {both} vs {single}, and a
              leave-seeds-out transfer check.

Statistical discipline: all transition-level tables are DESCRIPTIVE (transitions within a
sequence are dependent). Independent inference is performed at the seed level: per-seed
metrics are computed, then compared across the 5 seeds (mean +/- SD, paired t-test and
Wilcoxon for E vs F on identical seeds/datasets).

Gate statistic: f_t is a 212-dimensional vector with no principled register/destination
coordinate; the scalar used is the full-vector mean (the existing implementation's
`mean_gate` convention), with min/max recorded. No semantic mapping is invented.

Reproduce:
    python experiments/analyze_experiment_g.py \
        --transitions experiments/experiment_g_transitions.npz \
        --output experiments/experiment_g_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_g
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

from run_experiment_g import COLS, MODELS, SOURCE_IDS, SOURCE_NAMES  # noqa: E402
import run_experiment_g as rg_mod  # noqa: E402
from run_verifier_feasibility import auroc, average_precision  # noqa: E402

GATE_STATISTIC_NOTE = (
    "f_t is a 212-dim per-coordinate interpolation gate. The existing implementation reports "
    "only full-vector gate summaries (mean/std/entropy); there is no principled destination- or "
    "register-specific gate coordinate. The per-transition scalar used here is the full-vector "
    "mean gate (consistent with the existing mean_gate convention), reported alongside min/max. "
    "No semantic mapping between gate coordinates and registers is assumed."
)

STAT_DISCIPLINE_NOTE = (
    "Transition-level tables are descriptive; transitions within a sequence are dependent. "
    "Inference is seed-level (n=5): per-seed metrics, mean +/- SD, paired t-test and Wilcoxon "
    "for E vs F on identical seeds/datasets."
)

MIN_BIN_COUNT = 200
LOW_GATE = 0.35
HIGH_GATE = 0.50


class Table:
    """Column-oriented view of one model's transition records."""

    def __init__(self, arr: np.ndarray, cols: List[str]) -> None:
        self.arr = arr
        self.index = {c: i for i, c in enumerate(cols)}

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arr[:, self.index[name]]

    def __len__(self) -> int:
        return self.arr.shape[0]


def load_transitions(path: str) -> Tuple[Dict[str, Table], Dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    tables = {m: Table(z[m], COLS) for m in MODELS if m in z}
    return tables, meta


# ---------------------------------------------------------------------------
# Binning / Analysis 1
# ---------------------------------------------------------------------------

def merge_bins(edges: np.ndarray, counts: List[int], min_count: int = MIN_BIN_COUNT,
               min_bins: int = 3) -> Tuple[np.ndarray, List[int]]:
    """Deterministically merge sparse adjacent bins.

    Rule: while a bin has fewer than `min_count` transitions, merge the globally sparsest
    bin into its less-populated neighbour (ties -> left neighbor). Stop when all bins have
    >= min_count or `min_bins` bins remain. The resulting edges are reported.
    """
    counts = list(int(c) for c in counts)
    edges = list(float(e) for e in edges)
    while len(counts) > min_bins:
        sparse = [i for i, c in enumerate(counts) if c < min_count]
        if not sparse:
            break
        i = min(sparse, key=lambda j: counts[j])
        if i == 0:
            j = 1
        elif i == len(counts) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        lo, hi = (j, i) if j < i else (i, j)
        counts[lo] = counts[lo] + counts[hi]
        counts.pop(hi)
        edges.pop(hi)
    return np.asarray(edges), counts


def choose_bin_edges(tbl: Table, source: int = 0, min_count: int = MIN_BIN_COUNT) -> np.ndarray:
    """Fixed 0.1 grid on the full-vector mean gate, then deterministic sparse-bin merging."""
    f = tbl["f_mean"][tbl["source"] == source]
    counts, edges = np.histogram(f, bins=np.linspace(0.0, 1.0, 11))
    edges, _ = merge_bins(edges, list(counts), min_count=min_count)
    return edges


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def gate_decomposition(tbl: Table, edges: np.ndarray, source: int = 0) -> Dict[str, Any]:
    """Descriptive per-bin statistics for ADD transitions of one model."""
    sel = tbl["source"] == source
    f_all = tbl["f_mean"][sel]
    idx = np.clip(np.digitize(f_all, edges[1:-1]), 0, len(edges) - 2)
    bins: List[Dict[str, Any]] = []
    for k in range(len(edges) - 1):
        m = idx == k
        n = int(m.sum())
        entry: Dict[str, Any] = {"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": n}
        if n == 0:
            bin_entry = {**entry, "cand_acc": None, "comm_acc": None, "prev_acc": None,
                         "gap": None, "cond_commit_given_cand": None,
                         "cond_prev_given_fail": None, "mean_d_cs": None, "mean_d_ss": None,
                         "mean_d_sc": None, "wrap_frac": None, "f_mean_in_bin": None}
            bins.append(bin_entry)
            continue
        cand = tbl["cand_ok"][sel][m].astype(bool)
        comm = tbl["comm_ok"][sel][m].astype(bool)
        prev = tbl["prev_ok"][sel][m].astype(bool)
        fail = cand & ~comm
        cand_ok_sel = cand
        entry.update({
            "cand_acc": float(cand.mean()),
            "comm_acc": float(comm.mean()),
            "prev_acc": float(prev.mean()),
            "gap": float(cand.mean() - comm.mean()),
            "cond_commit_given_cand": (float(comm[cand_ok_sel].mean()) if cand_ok_sel.any() else None),
            "cond_prev_given_fail": (float(prev[fail].mean()) if fail.any() else None),
            "mean_d_cs": _safe_mean(tbl["d_cs"][sel][m]),
            "mean_d_ss": _safe_mean(tbl["d_ss"][sel][m]),
            "mean_d_sc": _safe_mean(tbl["d_sc"][sel][m]),
            "wrap_frac": float(tbl["wrap"][sel][m].mean()),
            "f_mean_in_bin": float(f_all[m].mean()),
        })
        bins.append(entry)
    return {"bin_edges": [float(e) for e in edges], "bins": bins}


def conditional_gate_analysis(tbl: Table, source: int = 0) -> Dict[str, Any]:
    """Candidate-correct subset: does low gate predict commit failure? (descriptive)"""
    sel = (tbl["source"] == source)
    cand = tbl["cand_ok"][sel].astype(bool)
    comm = tbl["comm_ok"][sel].astype(bool)
    f = tbl["f_mean"][sel]
    fail = cand & ~comm
    ok = cand & comm
    n_cand = int(cand.sum())
    out = {
        "n_transitions": int(sel.sum()),
        "n_cand_correct": n_cand,
        "comm_ok_given_cand_ok": float(comm[cand].mean()) if cand.any() else None,
        "n_cand_ok_commit_wrong": int(fail.sum()),
        "mean_f_commit_ok": _safe_mean(f[ok]),
        "mean_f_commit_wrong": _safe_mean(f[fail]),
        "f_auc_for_failure": (auroc(fail[cand], f[cand]) if fail[cand].any() else None),
        "share_failures_below_low_gate": (float((f[fail] < LOW_GATE).mean()) if fail.any() else None),
        "share_cand_ok_below_low_gate": (float((f[cand] < LOW_GATE).mean()) if cand.any() else None),
        "share_failures_at_or_above_high_gate": (float((f[fail] >= HIGH_GATE).mean()) if fail.any() else None),
        "share_cand_ok_at_or_above_high_gate": (float((f[cand] >= HIGH_GATE).mean()) if cand.any() else None),
        "prev_ok_given_failure": (float(tbl["prev_ok"][sel][fail].mean()) if fail.any() else None),
        "prev_ok_given_success": (float(tbl["prev_ok"][sel][ok].mean()) if ok.any() else None),
        "comm_eq_old_given_failure": (float(tbl["comm_eq_old"][sel][fail].mean()) if fail.any() else None),
        "mean_d_sc_fail": (float(np.mean(tbl["d_sc"][sel][fail])) if fail.any() else None),
        "mean_d_sc_ok": (float(np.mean(tbl["d_sc"][sel][ok])) if ok.any() else None),
        "mean_d_ss_fail": (float(np.mean(tbl["d_ss"][sel][fail])) if fail.any() else None),
        "mean_d_ss_ok": (float(np.mean(tbl["d_ss"][sel][ok])) if ok.any() else None),
        "low_gate": LOW_GATE,
        "high_gate": HIGH_GATE,
    }
    return out


def confound_stratification(tbl: Table, edges: np.ndarray, source: int = 0) -> Dict[str, Any]:
    """Stratify ADD transitions by arithmetic neighborhood (descriptive).

    Documented confounds of the raw gate view:
      no_op_src0            : source operand value 0 -> the correct ADD result equals the old value
                              (no state movement is required; committed decode of the old value is
                              automatically correct).
      same_register_src_gt0 : destination == source register, source value > 0 (self-referential
                              update; the operation reads and writes the same register).
      general               : destination != source register and source value > 0.
    """
    sel = tbl["source"] == source
    f = tbl["f_mean"][sel]
    cand = tbl["cand_ok"][sel].astype(bool)
    comm = tbl["comm_ok"][sel].astype(bool)
    ob = tbl["operand_b"][sel]
    oa = tbl["operand_a"][sel]
    same = tbl["dest"][sel] == tbl["src_reg"][sel]
    strata = {
        "no_op_src0": ob == 0,
        "same_register_src_gt0": same & (ob > 0),
        "general": (~same) & (ob > 0),
    }
    fail = cand & ~comm
    total_fail = int(fail.sum())

    def _summ(m: np.ndarray) -> Dict[str, Any]:
        if not m.any():
            return {"n": 0}
        fm = m & fail
        cm = m & cand
        both_classes = bool(fm[cm].any()) and bool((~fm[cm]).any()) if cm.any() else False
        return {
            "n": int(m.sum()),
            "cand_acc": float(cand[m].mean()),
            "comm_acc": float(comm[m].mean()),
            "gap_rate": float((cand[m] & ~comm[m]).mean()),
            "gap_cond_on_cand": (float(fm.sum() / cand[m].sum()) if cand[m].any() else None),
            "n_failures": int(fm.sum()),
            "share_of_all_failures": (float(fm.sum() / total_fail) if total_fail else None),
            "wrap_frac": float(tbl["wrap"][sel][m].mean()),
            "share_pred_equals_old": (float(tbl["comm_eq_old"][sel][fm].mean()) if fm.any() else None),
            "f_auc_for_failure": (auroc(fm[cm], f[cm]) if both_classes else None),
            "dcs_auc_for_failure": (auroc(fm[cm], tbl["d_cs"][sel][cm]) if both_classes else None),
        }

    out: Dict[str, Any] = {"strata": {k: _summ(m) for k, m in strata.items()}, "by_bin": {}}
    idx = np.clip(np.digitize(f, edges[1:-1]), 0, len(edges) - 2)
    for k in range(len(edges) - 1):
        bm = idx == k
        entry: Dict[str, Any] = {"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": int(bm.sum())}
        for name, sm in strata.items():
            entry[name] = _summ(bm & sm)
        out["by_bin"][f"{edges[k]:.2f}-{edges[k + 1]:.2f}"] = entry

    new_val = (oa + ob) % rg_mod.gr.NUM_VALUES
    if total_fail:
        pred = tbl["comm_pred"][sel][fail]
        old_flag = tbl["comm_eq_old"][sel][fail].astype(bool)
        out["failure_decode_composition"] = {
            "n_failures": total_fail,
            "share_pred_equals_old": float(old_flag.mean()),
            "share_pred_equals_new": float((pred == new_val[fail]).mean()),
            "share_pred_other": float((~old_flag & (pred != new_val[fail])).mean()),
        }
    else:
        out["failure_decode_composition"] = {"n_failures": 0}
    return out


# ---------------------------------------------------------------------------
# Analysis 2 — oracle commit
# ---------------------------------------------------------------------------

def oracle_table(tbl: Table) -> Dict[str, Dict[str, Any]]:
    """Actual committed vs oracle candidate destination decoding by ADD category."""
    out: Dict[str, Dict[str, Any]] = {}
    sel0 = tbl["source"] == 0
    test_cats = {
        "test_all": sel0,
        "test_isolated": sel0 & (tbl["isolated"] == 1),
        "test_repeated": sel0 & (tbl["isolated"] == 0),
        "test_no_wrap": sel0 & (tbl["wrap"] == 0),
        "test_wrap": sel0 & (tbl["wrap"] == 1),
    }
    for name, m in test_cats.items():
        out[name] = _oracle_entry(tbl, m)
    for name, sid in SOURCE_IDS.items():
        if sid == 0:
            continue
        out[f"controlled:{name}"] = _oracle_entry(tbl, tbl["source"] == sid)
    return out


def _oracle_entry(tbl: Table, m: np.ndarray) -> Dict[str, Any]:
    cand = tbl["cand_ok"][m].astype(bool)
    comm = tbl["comm_ok"][m].astype(bool)
    lost = cand & ~comm
    return {
        "n": int(m.sum()),
        "committed_dest_acc": (float(comm.mean()) if m.any() else None),
        "candidate_dest_acc": (float(cand.mean()) if m.any() else None),
        "oracle_minus_committed_gap": (float(cand.mean() - comm.mean()) if m.any() else None),
        "n_lost_at_commitment": int(lost.sum()),
        "frac_cand_correct_lost_at_commitment": (float(lost.sum() / cand.sum()) if cand.any() else None),
    }


# ---------------------------------------------------------------------------
# Seed-level inference
# ---------------------------------------------------------------------------

def seed_metrics(tbl: Table, source: int = 0) -> Dict[str, Dict[str, Any]]:
    """Per-seed metrics on ADD transitions (independent unit = seed)."""
    out: Dict[str, Dict[str, Any]] = {}
    sel = tbl["source"] == source
    for seed in sorted(set(int(s) for s in tbl["seed"][sel])):
        m = sel & (tbl["seed"] == seed)
        cand = tbl["cand_ok"][m].astype(bool)
        comm = tbl["comm_ok"][m].astype(bool)
        f = tbl["f_mean"][m]
        fail = cand & ~comm
        out[str(seed)] = {
            "n_add": int(m.sum()),
            "cand_acc": float(cand.mean()),
            "comm_acc": float(comm.mean()),
            "gap_rate": float(cand.mean() - comm.mean()),
            "cond_commit_ok_given_cand_ok": (float(comm[cand].mean()) if cand.any() else None),
            "cond_fail_rate_given_cand_ok": (float(fail.sum() / cand.sum()) if cand.any() else None),
            "mean_f": float(f.mean()),
            "mean_f_fail": (float(f[fail].mean()) if fail.any() else None),
            "mean_f_success": (float(f[cand & comm].mean()) if (cand & comm).any() else None),
            "fail_share_low_gate": (float((f[fail] < LOW_GATE).mean()) if fail.any() else None),
            "fail_share_high_gate": (float((f[fail] >= HIGH_GATE).mean()) if fail.any() else None),
            "prev_ok_given_fail": (float(tbl["prev_ok"][m][fail].mean()) if fail.any() else None),
            "comm_eq_old_given_fail": (float(tbl["comm_eq_old"][m][fail].mean()) if fail.any() else None),
        }
    return out


def summarize_seed_metrics(per_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = [k for k in next(iter(per_seed.values())) if k != "n_add"]
    out: Dict[str, Any] = {"n_seeds": len(per_seed)}
    for k in keys:
        vals = np.array([per_seed[s][k] for s in sorted(per_seed) if per_seed[s][k] is not None], dtype=float)
        out[k] = {"mean": float(vals.mean()) if len(vals) else None,
                  "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                  "per_seed": {s: per_seed[s][k] for s in sorted(per_seed)}}
    return out


def paired_model_stats(summary_E: Dict[str, Any], summary_F: Dict[str, Any]) -> Dict[str, Any]:
    """Paired E vs F across seeds for every seed-level metric."""
    out: Dict[str, Any] = {}
    for k, vE in summary_E.items():
        if not isinstance(vE, dict) or "per_seed" not in vE or k not in summary_F:
            continue
        seeds = sorted(vE["per_seed"])
        a = np.array([vE["per_seed"][s] for s in seeds], dtype=float)
        b = np.array([summary_F[k]["per_seed"][s] for s in seeds], dtype=float)
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        d = a - b
        t, p_t = st.ttest_rel(a, b)
        try:
            _, p_w = st.wilcoxon(a, b)
        except ValueError:
            p_w = float("nan")
        out[k] = {"E_mean": float(a.mean()), "F_mean": float(b.mean()),
                  "E_minus_F": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                  "paired_t_p": float(p_t), "wilcoxon_p": float(p_w),
                  "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None,
                  "per_seed_diff": {s: float(x) for s, x in zip(seeds, d)}}
    return out


def paired_transition_analysis(tbl_e: Table, tbl_f: Table, source: int = 0) -> Dict[str, Any]:
    """Paired E vs F comparison on identical transitions (same seeds, same test streams).

    Restricted to transitions where BOTH models' candidates are correct; the committed
    outcome is then compared 2x2 (both ok / E-only fail / F-only fail / both fail).
    Raw cross-tabulations are descriptive; per-seed rates carry the independent-unit view.
    """
    key_cols = ["seed", "source", "seq_id", "t", "dest", "src_reg"]
    key_e = np.stack([tbl_e[c] for c in key_cols], axis=1)
    key_f = np.stack([tbl_f[c] for c in key_cols], axis=1)
    if key_e.shape != key_f.shape or not np.array_equal(key_e, key_f):
        raise ValueError("E and F transition tables are not aligned on (seed, source, seq_id, t, dest, src_reg)")
    sel = (tbl_e["source"] == source)
    cand_e = tbl_e["cand_ok"][sel].astype(bool)
    cand_f = tbl_f["cand_ok"][sel].astype(bool)
    comm_e = tbl_e["comm_ok"][sel].astype(bool)
    comm_f = tbl_f["comm_ok"][sel].astype(bool)
    seeds = tbl_e["seed"][sel].astype(int)
    both_cand = cand_e & cand_f
    e_fail = both_cand & ~comm_e
    f_fail = both_cand & ~comm_f
    table = {
        "n_both_cand_correct": int(both_cand.sum()),
        "both_commit_ok": int((both_cand & comm_e & comm_f).sum()),
        "E_only_fail": int((e_fail & ~f_fail).sum()),
        "F_only_fail": int((f_fail & ~e_fail).sum()),
        "both_fail": int((e_fail & f_fail).sum()),
    }
    per_seed = {}
    for s in sorted(set(seeds)):
        m = both_cand & (seeds == s)
        n = max(int(m.sum()), 1)
        per_seed[str(s)] = {"n": int(m.sum()),
                            "E_fail_rate": float((e_fail & m).sum() / n),
                            "F_fail_rate": float((f_fail & m).sum() / n)}
    a = np.array([per_seed[str(s)]["E_fail_rate"] for s in sorted(set(seeds))])
    b = np.array([per_seed[str(s)]["F_fail_rate"] for s in sorted(set(seeds))])
    d = a - b
    t, p_t = st.ttest_rel(a, b)
    try:
        _, p_w = st.wilcoxon(a, b)
    except ValueError:
        p_w = float("nan")
    return {"contingency": table, "per_seed": per_seed,
            "E_fail_rate_mean": float(a.mean()), "F_fail_rate_mean": float(b.mean()),
            "E_minus_F": float(d.mean()), "paired_t_p": float(p_t), "wilcoxon_p": float(p_w)}


# ---------------------------------------------------------------------------
# Analysis 3 — predictability
# ---------------------------------------------------------------------------

def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 60) -> np.ndarray:
    """Newton-Raphson L2 logistic regression (transparent, low-capacity). X standardized."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) * l2
    reg[0, 0] = 0.0
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-z))
        W = np.clip(p * (1.0 - p), 1e-9, None)
        grad = Xb.T @ (p - y) + reg @ w
        H = Xb.T @ (Xb * W[:, None]) + reg
        w = w - np.linalg.solve(H, grad)
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = np.hstack([np.ones((len(X), 1)), X]) @ w
    return 1.0 / (1.0 + np.exp(-z))


def grouped_folds(groups: np.ndarray, n_folds: int = 5) -> np.ndarray:
    """Deterministic grouped folds: unique groups assigned round-robin in sorted order."""
    uniq = np.unique(groups)
    mapping = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([mapping[g] for g in groups], dtype=int)


def _standardize(Xtr: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def grouped_cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       n_folds: int = 5, l2: float = 1.0) -> Dict[str, Any]:
    """Out-of-fold predictions with sequence-grouped folds; pooled AUC/AP + per-fold AUCs."""
    folds = grouped_folds(groups, n_folds=n_folds)
    oof = np.full(len(y), np.nan)
    fold_aucs: List[float] = []
    fold_aps: List[float] = []
    for k in range(n_folds):
        te = folds == k
        tr = ~te
        if y[tr].sum() == 0 or y[te].sum() == 0 or (~y[te].astype(bool)).sum() == 0:
            continue
        mu, sd = _standardize(X[tr], X)
        w = fit_logistic((X[tr] - mu) / sd, y[tr].astype(float), l2=l2)
        oof[te] = predict_logistic(w, (X[te] - mu) / sd)
        fold_aucs.append(auroc(y[te].astype(bool), oof[te]))
        fold_aps.append(average_precision(y[te].astype(bool), oof[te]))
    mask = ~np.isnan(oof)
    return {"oof": oof, "mask": mask,
            "pooled_auc": auroc(y[mask].astype(bool), oof[mask]),
            "pooled_ap": average_precision(y[mask].astype(bool), oof[mask]),
            "fold_aucs": fold_aucs, "fold_aps": fold_aps,
            "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
            "fold_auc_sd": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else None}


def bootstrap_auc_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray,
                       n_boot: int = 200, seed: int = 0) -> Dict[str, Any]:
    """Grouped bootstrap CI for AUC(p_a) - AUC(p_b) (resampling sequences).

    Vectorized: rows are pre-sorted by group; each bootstrap draw is applied as a
    multiplicity over groups, then rows are repeated accordingly.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    uniq, starts = np.unique(g_sorted, return_index=True)
    sizes = np.diff(np.append(starts, len(g_sorted)))
    base_rows = np.arange(len(g_sorted))
    diffs: List[float] = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        k = np.bincount(draw, minlength=len(uniq))
        mult = np.repeat(k, sizes)
        sel = np.repeat(base_rows, mult)
        idx = order[sel]
        yy = y[idx].astype(bool)
        if yy.sum() == 0 or (~yy).sum() == 0:
            continue
        diffs.append(auroc(yy, p_a[idx]) - auroc(yy, p_b[idx]))
    diffs_arr = np.asarray(diffs, dtype=float)
    return {"mean_diff": float(diffs_arr.mean()) if len(diffs_arr) else None,
            "ci95": [float(np.percentile(diffs_arr, 2.5)), float(np.percentile(diffs_arr, 97.5))]
            if len(diffs_arr) else None,
            "n_boot_valid": int(len(diffs_arr))}


def predictability(tbl: Table, source: int = 0) -> Dict[str, Any]:
    """Targets: (A) cand_ok & ~comm_ok over all ADDs; (B) ~comm_ok | cand_ok (conditional)."""
    sel = tbl["source"] == source
    f = tbl["f_mean"][sel]
    d_cs = tbl["d_cs"][sel]
    cand = tbl["cand_ok"][sel].astype(bool)
    comm = tbl["comm_ok"][sel].astype(bool)
    seeds = tbl["seed"][sel].astype(int)
    seqs = tbl["seq_id"][sel].astype(int)
    groups = seeds * 100000 + seqs

    out: Dict[str, Any] = {}
    targets = {
        "all_adds__cand_ok_and_comm_wrong": (cand & ~comm),
        "cand_correct__comm_wrong": (cand & ~comm)[cand],
    }
    sel_masks = {
        "all_adds__cand_ok_and_comm_wrong": np.ones(len(f), dtype=bool),
        "cand_correct__comm_wrong": cand,
    }
    for target_name, y in targets.items():
        m = sel_masks[target_name]
        yv = y.astype(float)
        fv, dv, gv = f[m], d_cs[m], groups[m]
        Xf = fv[:, None]
        Xd = dv[:, None]
        Xb = np.column_stack([fv, dv])
        entry: Dict[str, Any] = {
            "n": int(m.sum()),
            "base_rate": float(yv.mean()),
            "spearman_f": float(st.spearmanr(fv, yv).statistic),
            "spearman_d_cs": float(st.spearmanr(dv, yv).statistic),
            "spearman_f_vs_d_cs": float(st.spearmanr(fv, dv).statistic),
            "pointbiserial_f": float(st.pointbiserialr(yv, fv).statistic),
            "pointbiserial_d_cs": float(st.pointbiserialr(yv, dv).statistic),
            "cv": {}, "leave_seeds_out": {},
        }
        for name, X in (("f_only", Xf), ("d_cs_only", Xd), ("both", Xb)):
            cv = grouped_cv_predict(X, yv, gv, n_folds=5)
            entry["cv"][name] = {k: v for k, v in cv.items() if k not in ("oof", "mask")}
            entry["cv"][name]["_oof"] = cv["oof"].tolist()
        # leave-seeds-out transfer: train 42-44, test 45-46
        tr = np.isin(seeds[m], [42, 43, 44])
        te = ~tr
        if yv[tr].sum() > 0 and yv[te].sum() > 0:
            mu, sd = _standardize(Xb[tr], Xb)
            w = fit_logistic((Xb[tr] - mu) / sd, yv[tr], l2=1.0)
            p_te = predict_logistic(w, (Xb[te] - mu) / sd)
            entry["leave_seeds_out"] = {
                "auc_both": auroc(yv[te].astype(bool), p_te),
                "ap_both": average_precision(yv[te].astype(bool), p_te),
                "n_train": int(tr.sum()), "n_test": int(te.sum())}
        # bootstrap CI for AUC(both) - AUC(single) on pooled OOF predictions
        oof_both = entry["cv"]["both"]["_oof"]
        oof_f = entry["cv"]["f_only"]["_oof"]
        oof_dcs = entry["cv"]["d_cs_only"]["_oof"]
        if not np.isnan(np.array(oof_both)).all() and not np.isnan(np.array(oof_f)).all():
            entry["auc_diff_both_minus_f"] = bootstrap_auc_diff(
                yv, np.array(oof_both), np.array(oof_f), gv, n_boot=200)
        if not np.isnan(np.array(oof_both)).all() and not np.isnan(np.array(oof_dcs)).all():
            entry["auc_diff_both_minus_dcs"] = bootstrap_auc_diff(
                yv, np.array(oof_both), np.array(oof_dcs), gv, n_boot=200)
        entry["cv"]["both"].pop("_oof", None)
        entry["cv"]["f_only"].pop("_oof", None)
        entry["cv"]["d_cs_only"].pop("_oof", None)
        out[target_name] = entry
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(tables: Dict[str, Table], analysis: Dict[str, Any],
                 fig_dir: str, prefix: str = "experiment_g") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    paths: List[str] = []

    # Fig 1: gate vs committed accuracy (candidate accuracy as reference)
    plt.figure(figsize=(7.0, 4.4))
    for model in MODELS:
        dec = analysis["analysis_1_gate_decomposition"][model]
        edges = dec["bin_edges"]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        comm = [b["comm_acc"] for b in dec["bins"]]
        cand = [b["cand_acc"] for b in dec["bins"]]
        ns = [b["n"] for b in dec["bins"]]
        style = {"E_candidate": ("tab:blue", "o"), "F_combined": ("tab:brown", "s")}[model]
        plt.plot(centers, comm, marker=style[1], color=style[0], label=f"{model} committed")
        plt.plot(centers, cand, marker=style[1], linestyle="--", color=style[0], alpha=0.6,
                 label=f"{model} candidate")
        for xc, yc, nc in zip(centers, comm, ns):
            if yc is not None:
                plt.annotate(str(nc), (xc, yc), textcoords="offset points", xytext=(0, -12),
                             fontsize=6, ha="center", color=style[0])
    plt.xlabel("full-vector mean gate f (ADD transitions, binned)")
    plt.ylabel("destination accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.title("ADD destination decoding vs gate (test split, pooled seeds; n per bin)")
    p = os.path.join(fig_dir, f"{prefix}_gate_vs_commit_acc.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # Fig 2: gate histograms by outcome (candidate-correct subset)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, model in zip(axes, MODELS):
        tbl = tables[model]
        sel = tbl["source"] == 0
        cand = tbl["cand_ok"][sel].astype(bool)
        comm = tbl["comm_ok"][sel].astype(bool)
        f = tbl["f_mean"][sel]
        bins = np.linspace(0, 1, 21)
        ax.hist(f[cand & comm], bins=bins, density=True, alpha=0.55,
                label="cand ok / commit ok", color="tab:green")
        ax.hist(f[cand & ~comm], bins=bins, density=True, alpha=0.55,
                label="cand ok / commit wrong", color="tab:red")
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("full-vector mean gate f")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    axes[1].legend(fontsize=7)
    fig.suptitle("Gate distribution by outcome (ADD, test split, pooled seeds)", fontsize=10)
    p = os.path.join(fig_dir, f"{prefix}_gate_histograms.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # Fig 3: candidate accuracy vs committed accuracy across gate bins
    plt.figure(figsize=(4.8, 4.6))
    for model in MODELS:
        dec = analysis["analysis_1_gate_decomposition"][model]
        xs = [b["cand_acc"] for b in dec["bins"] if b["cand_acc"] is not None]
        ys = [b["comm_acc"] for b in dec["bins"] if b["cand_acc"] is not None]
        ns = [b["n"] for b in dec["bins"] if b["cand_acc"] is not None]
        color = {"E_candidate": "tab:blue", "F_combined": "tab:brown"}[model]
        marker = {"E_candidate": "o", "F_combined": "s"}[model]
        plt.scatter(xs, ys, s=[max(20, n / 60) for n in ns], color=color, marker=marker,
                    alpha=0.8, label=f"{model} (size ~ n)")
    lims = [0.4, 1.0]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("candidate destination accuracy (per gate bin)")
    plt.ylabel("committed destination accuracy (per gate bin)")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.title("Candidate vs committed accuracy across gate bins")
    p = os.path.join(fig_dir, f"{prefix}_cand_vs_comm.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # Fig 4: ||C - S_prev|| vs gate by outcome (candidate-correct subset)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, model in zip(axes, MODELS):
        tbl = tables[model]
        sel = tbl["source"] == 0
        cand = tbl["cand_ok"][sel].astype(bool)
        comm = tbl["comm_ok"][sel].astype(bool)
        f = tbl["f_mean"][sel]
        d = tbl["d_cs"][sel]
        for mask, color, label in ((cand & comm, "tab:green", "cand ok / commit ok"),
                                   (cand & ~comm, "tab:red", "cand ok / commit wrong")):
            xs, ys = f[mask], d[mask]
            if len(xs) > 4000:
                sub = rng.choice(len(xs), 4000, replace=False)
                xs, ys = xs[sub], ys[sub]
            ax.scatter(xs, ys, s=3, alpha=0.12, color=color)
        # bin means
        edges = analysis["analysis_1_gate_decomposition"][model]["bin_edges"]
        idx = np.clip(np.digitize(f, edges[1:-1]), 0, len(edges) - 2)
        centers, mean_ok, mean_fail = [], [], []
        for k in range(len(edges) - 1):
            m = idx == k
            if not m.sum():
                continue
            centers.append((edges[k] + edges[k + 1]) / 2)
            m_ok = m & cand & comm
            m_fail = m & cand & ~comm
            mean_ok.append(np.mean(d[m_ok]) if m_ok.any() else np.nan)
            mean_fail.append(np.mean(d[m_fail]) if m_fail.any() else np.nan)
        ax.plot(centers, mean_ok, color="tab:green", marker="o", linewidth=2, label="mean |C-S_prev| (ok)")
        ax.plot(centers, mean_fail, color="tab:red", marker="o", linewidth=2, label="mean |C-S_prev| (fail)")
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("full-vector mean gate f")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("||C_t - S_(t-1)||")
    fig.suptitle("Candidate displacement vs gate, by commit outcome (candidate-correct ADDs)", fontsize=10)
    p = os.path.join(fig_dir, f"{prefix}_dcs_vs_gate.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # Fig 5: oracle candidate vs actual committed accuracy by ADD category
    cats = ["controlled:isolated_add", "controlled:add_no_wrap", "controlled:add_wrap",
            "controlled:repeated_add"]
    labels = ["isolated ADD", "ADD no-wrap", "ADD wrap", "4x repeated ADD"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    x = np.arange(len(cats))
    width = 0.38
    for ax, model in zip(axes, MODELS):
        ot = analysis["analysis_2_oracle_commit"][model]
        committed = [ot[c]["committed_dest_acc"] for c in cats]
        candidate = [ot[c]["candidate_dest_acc"] for c in cats]
        ax.bar(x - width / 2, committed, width, label="committed state", color="tab:gray")
        ax.bar(x + width / 2, candidate, width, label="candidate (oracle decode)", color="tab:orange")
        for xi, (cm, ca) in enumerate(zip(committed, candidate)):
            ax.annotate(f"+{ca - cm:.02f}", (xi, max(cm, ca) + 0.02), ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
        ax.set_title(model, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("destination accuracy")
    fig.suptitle("Oracle candidate vs committed destination decoding (controlled ADD builders)", fontsize=10)
    p = os.path.join(fig_dir, f"{prefix}_oracle_vs_committed.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # Fig 6: ROC curves (grouped CV out-of-fold predictions)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2))
    target_names = ["all_adds__cand_ok_and_comm_wrong", "cand_correct__comm_wrong"]
    target_labels = ["cand ok & commit wrong (all ADDs)", "commit wrong | cand ok"]
    for col, model in enumerate(MODELS):
        for row, (tname, tlabel) in enumerate(zip(target_names, target_labels)):
            ax = axes[row, col]
            entry = analysis["analysis_3_predictability"][model][tname]
            # Recompute the out-of-fold scores for plotting (same deterministic CV).
            tbl = tables[model]
            sel = tbl["source"] == 0
            f = tbl["f_mean"][sel]
            d = tbl["d_cs"][sel]
            cand = tbl["cand_ok"][sel].astype(bool)
            comm = tbl["comm_ok"][sel].astype(bool)
            groups = tbl["seed"][sel].astype(int) * 100000 + tbl["seq_id"][sel].astype(int)
            y_full = cand & ~comm
            m = np.ones(len(f), bool) if tname.startswith("all") else cand
            Xf, Xd = f[m][:, None], d[m][:, None]
            Xb = np.column_stack([f[m], d[m]])
            for name, X, color in (("f only", Xf, "tab:blue"), ("||C-S_prev|| only", Xd, "tab:green"),
                                   ("both", Xb, "tab:red")):
                cv = grouped_cv_predict(X, y_full[m].astype(float), groups[m], n_folds=5)
                mask = cv["mask"]
                _plot_roc(ax, y_full[m][mask].astype(bool), cv["oof"][mask],
                          label=f"{name} (AUC {cv['pooled_auc']:.3f})", color=color)
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_title(f"{model} — {tlabel}", fontsize=8)
            ax.legend(fontsize=6, loc="lower right")
            ax.grid(alpha=0.3)
    fig.suptitle("Predictability of the candidate->commit gap (grouped 5-fold CV, sequence groups)",
                 fontsize=10)
    p = os.path.join(fig_dir, f"{prefix}_prediction.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # Fig 7: gap rate (candidate-correct only) by gate bin and arithmetic stratum
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    strata_style = (("general", "tab:red", "o"), ("same_register_src_gt0", "tab:purple", "s"),
                    ("no_op_src0", "tab:gray", "^"))
    for ax, model in zip(axes, MODELS):
        cs = analysis["analysis_1_gate_decomposition"][model]["confound_stratification"]
        for name, color, marker in strata_style:
            xs, ys, ns = [], [], []
            for _, entry in sorted(cs["by_bin"].items()):
                s = entry[name]
                if not s.get("n") or s.get("gap_cond_on_cand") is None:
                    continue
                xs.append((entry["lo"] + entry["hi"]) / 2)
                ys.append(s["gap_cond_on_cand"])
                ns.append(s["n"])
            ax.plot(xs, ys, marker=marker, color=color, label=name)
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6),
                            fontsize=6, ha="center")
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("full-vector mean gate f")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("P(commit wrong | candidate correct)")
    fig.suptitle("Gap rate by gate and arithmetic stratum (candidate-correct ADDs, test split)",
                 fontsize=10)
    p = os.path.join(fig_dir, f"{prefix}_confound_strata.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def _plot_roc(ax, y: np.ndarray, scores: np.ndarray, label: str, color: str) -> None:
    order = np.argsort(-scores)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    ax.plot(np.concatenate([[0], fpr]), np.concatenate([[0], tpr]), color=color, linewidth=1.2,
            label=label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def summarize_for_decision(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Key numbers used to classify the evidence (no causal claim)."""
    out: Dict[str, Any] = {}
    for model in MODELS:
        cond = analysis["analysis_1_gate_decomposition"][model]["conditional_cand_correct"]
        cs = analysis["analysis_1_gate_decomposition"][model]["confound_stratification"]
        seed_sum = analysis["seed_level_inference"]["per_model"][model]["summary"]
        pred_all = analysis["analysis_3_predictability"][model]["all_adds__cand_ok_and_comm_wrong"]
        out[model] = {
            "n_cand_ok": cond["n_cand_correct"],
            "n_gap": cond["n_cand_ok_commit_wrong"],
            "gap_share_of_cand_ok": (cond["n_cand_ok_commit_wrong"] / max(cond["n_cand_correct"], 1)),
            "mean_f_success": cond["mean_f_commit_ok"],
            "mean_f_failure": cond["mean_f_commit_wrong"],
            "f_auc_for_failure_among_cand_ok": cond["f_auc_for_failure"],
            "fail_share_below_0.35": cond["share_failures_below_low_gate"],
            "cand_ok_share_below_0.35": cond["share_cand_ok_below_low_gate"],
            "fail_share_at_or_above_0.50": cond["share_failures_at_or_above_high_gate"],
            "prev_ok_given_failure": cond["prev_ok_given_failure"],
            "comm_eq_old_given_failure": cond["comm_eq_old_given_failure"],
            "mean_d_sc_fail_vs_ok": [cond["mean_d_sc_fail"], cond["mean_d_sc_ok"]],
            "seed_level_cond_fail_rate": seed_sum["cond_fail_rate_given_cand_ok"]["mean"],
            "seed_level_cond_fail_rate_sd": seed_sum["cond_fail_rate_given_cand_ok"]["sd"],
            "cv_auc_both": pred_all["cv"]["both"]["pooled_auc"],
            "cv_auc_f_only": pred_all["cv"]["f_only"]["pooled_auc"],
            "cv_auc_dcs_only": pred_all["cv"]["d_cs_only"]["pooled_auc"],
            "bootstrap_auc_diff_both_minus_f": pred_all.get("auc_diff_both_minus_f"),
            "bootstrap_auc_diff_both_minus_dcs": pred_all.get("auc_diff_both_minus_dcs"),
            "gap_cond_general": cs["strata"]["general"].get("gap_cond_on_cand"),
            "gap_cond_same_reg": cs["strata"]["same_register_src_gt0"].get("gap_cond_on_cand"),
            "gap_cond_no_op": cs["strata"]["no_op_src0"].get("gap_cond_on_cand"),
            "general_f_auc_failure": cs["strata"]["general"].get("f_auc_for_failure"),
            "general_dcs_auc_failure": cs["strata"]["general"].get("dcs_auc_for_failure"),
            "failures_no_op": cs["strata"]["no_op_src0"].get("n_failures"),
            "share_failures_general": cs["strata"]["general"].get("share_of_all_failures"),
            "share_failures_same_reg": cs["strata"]["same_register_src_gt0"].get("share_of_all_failures"),
            "failure_share_pred_equals_old": cs["failure_decode_composition"].get("share_pred_equals_old"),
        }
    return out


def build_analysis(tables: Dict[str, Table], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the full results payload (JSON-serializable)."""
    results: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "collector_meta": meta,
            "gate_statistic_note": GATE_STATISTIC_NOTE,
            "statistical_discipline_note": STAT_DISCIPLINE_NOTE,
            "bin_rule": (f"fixed 0.1 grid on full-vector mean gate; sparse bins (< {MIN_BIN_COUNT}) "
                         f"merged deterministically into the less-populated neighbour until all "
                         f"bins >= {MIN_BIN_COUNT} or 3 bins remain; final edges reported"),
            "low_gate": LOW_GATE, "high_gate": HIGH_GATE,
        },
        "analysis_1_gate_decomposition": {},
        "analysis_2_oracle_commit": {},
        "analysis_3_predictability": {},
        "seed_level_inference": {"per_model": {}, "paired_E_vs_F": None},
    }

    for model in MODELS:
        if model not in tables:
            continue
        tbl = tables[model]
        edges = choose_bin_edges(tbl)
        decomposition = gate_decomposition(tbl, edges)
        decomposition["conditional_cand_correct"] = conditional_gate_analysis(tbl)
        decomposition["confound_stratification"] = confound_stratification(tbl, edges)
        results["analysis_1_gate_decomposition"][model] = decomposition
        results["analysis_2_oracle_commit"][model] = oracle_table(tbl)
        results["analysis_3_predictability"][model] = predictability(tbl)
        per_seed = seed_metrics(tbl)
        results["seed_level_inference"]["per_model"][model] = {
            "per_seed": per_seed, "summary": summarize_seed_metrics(per_seed)}

    if "E_candidate" in results["seed_level_inference"]["per_model"] and \
            "F_combined" in results["seed_level_inference"]["per_model"]:
        results["seed_level_inference"]["paired_E_vs_F"] = paired_model_stats(
            results["seed_level_inference"]["per_model"]["E_candidate"]["summary"],
            results["seed_level_inference"]["per_model"]["F_combined"]["summary"])
    if "E_candidate" in tables and "F_combined" in tables:
        results["paired_E_vs_F_transition_level"] = paired_transition_analysis(
            tables["E_candidate"], tables["F_combined"])

    results["summary_for_decision"] = summarize_for_decision(results)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment G analysis (measurement only)")
    ap.add_argument("--transitions", default="experiments/experiment_g_transitions.npz")
    ap.add_argument("--output", default="experiments/experiment_g_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_g")
    args = ap.parse_args()

    tables, meta = load_transitions(args.transitions)
    results = build_analysis(tables, meta)
    results["metadata"]["transitions_file"] = args.transitions

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.output}")

    fig_paths = make_figures(tables, results, args.fig_dir, prefix=args.fig_prefix)
    print("Figures:")
    for p in fig_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
