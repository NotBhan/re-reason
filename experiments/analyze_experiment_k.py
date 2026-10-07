#!/usr/bin/env python3
"""Experiment K — analysis of the coordinate-preserving gate-balance ablation.

Reads experiments/experiment_k_records.json (runner output), performs seed-level paired
statistics (kappa=2 vs the kappa=1 identity baseline), gate-shape / balance accounting,
decision-margin and coordinate-delivery comparison, rescue/regression accounting,
candidate-margin stratification, the off-path subset, operation-level safety checks, the
pre-registered A/B/C/D decision, the comparison against Experiment J's historical gamma
reference, and the required figures.

Statistical discipline: the five frozen E checkpoints (seeds 42-46) are the independent
experimental units. Transition-level rescue/regression counts are descriptive; primary
inference is seed-level paired vs kappa=1 (mean +/- SD, Cohen's dz, paired t, sign k/5).
Cohen's dz bands are declared conventions, not data-tuned cutoffs.

Reproduce:
    python experiments/analyze_experiment_k.py \
        --records experiments/experiment_k_records.json \
        --output experiments/experiment_k_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_k
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRIMARY_MODEL = "E_candidate"
CONTROL_MODEL = "F_combined"
BASELINE_KAPPA = 1.0
SHARPEN_KAPPA = 2.0
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20]
J_RESULTS_DEFAULT = "experiments/experiment_j_results.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def paired(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    out = {"a_mean": float(a.mean()), "a_sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
           "b_mean": float(b.mean()), "b_sd": float(b.std(ddof=1)) if b.size > 1 else 0.0,
           "mean_diff": float(d.mean()), "n": int(d.size),
           "sign_positive": int((d > 0).sum()), "sign_negative": int((d < 0).sum())}
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    out["cohens_dz"] = float(d.mean() / sd) if sd > 0 else float("nan")
    if d.size > 1:
        try:
            out["paired_t_p"] = float(st.ttest_rel(a, b).pvalue)
        except Exception:
            out["paired_t_p"] = None
    return out


def effect_band(dz: Optional[float]) -> str:
    if dz is None or (isinstance(dz, float) and np.isnan(dz)):
        return "na"
    a = abs(dz)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def sig_effect(p: Optional[Dict[str, Any]], beneficial: bool) -> bool:
    """Pre-registered 'sequence-level effect': sign-consistent >=4/5, |dz|>=0.8, p<0.05."""
    if p is None or p.get("cohens_dz") is None or np.isnan(p.get("cohens_dz", float("nan"))):
        return False
    same = p["sign_positive"] >= 4 if beneficial else p["sign_negative"] >= 4
    pv = p.get("paired_t_p")
    return bool(same and abs(p["cohens_dz"]) >= 0.8 and pv is not None and pv < 0.05)


def _g(r: Dict[str, Any], *keys: Any) -> Optional[float]:
    cur: Any = r
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, (list, tuple)) and isinstance(k, int):
            cur = cur[k] if 0 <= k < len(cur) else None
        else:
            return None
    return None if cur is None else float(cur)


def _diff(r: Dict[str, Any], *a_keys: Any, b_keys: Any) -> Optional[float]:
    x = _g(r, *a_keys); y = _g(r, *b_keys)
    return None if (x is None or y is None) else x - y


# name -> callable(record) -> float|None
SEED_METRICS: Dict[str, Callable[[Dict[str, Any]], Optional[float]]] = {
    "t1_em": lambda r: _g(r, "horizons", "1", "em"),
    "t2_em": lambda r: _g(r, "horizons", "2", "em"),
    "t3_em": lambda r: _g(r, "horizons", "3", "em"),
    "t5_em": lambda r: _g(r, "horizons", "5", "em"),
    "t8_em": lambda r: _g(r, "horizons", "8", "em"),
    "t10_em": lambda r: _g(r, "horizons", "10", "em"),
    "t15_em": lambda r: _g(r, "horizons", "15", "em"),
    "t20_em": lambda r: _g(r, "horizons", "20", "em"),
    "comm_written": lambda r: _g(r, "test", "comm_written_mean"),
    "cand_written": lambda r: _g(r, "test", "cand_written_mean"),
    "comm_full": lambda r: _g(r, "test", "comm_full_mean"),
    "cand_full": lambda r: _g(r, "test", "cand_full_mean"),
    "first_real_err": lambda r: _g(r, "test", "first_real_error_rate"),
    "first_strict_err": lambda r: _g(r, "test", "first_strict_error_rate"),
    "gate_mean": lambda r: _g(r, "test", "gate_mean"),
    "gate_scaled_mean": lambda r: _g(r, "test", "gate_scaled_mean"),
    "gate_entropy": lambda r: _g(r, "test", "gate_entropy"),
    "gate_scaled_entropy": lambda r: _g(r, "test", "gate_scaled_entropy"),
    # primary set (baseline candidate-correct general ADD)
    "primary_comm_acc": lambda r: _g(r, "primary", "comm_acc"),
    "primary_cand_acc": lambda r: _g(r, "primary", "cand_acc"),
    "primary_rho": lambda r: _g(r, "primary", "rho_median"),
    "primary_m_prev": lambda r: _g(r, "primary", "m_prev_median"),
    "primary_m_cand": lambda r: _g(r, "primary", "m_cand_median"),
    "primary_m_commit": lambda r: _g(r, "primary", "m_commit_median"),
    "primary_dm_commit": lambda r: _g(r, "primary", "dm_commit_mean"),
    "primary_shortfall": lambda r: _g(r, "primary", "margin_shortfall_median"),
    "primary_gw_pos": lambda r: _g(r, "primary", "gw_pos_mean"),
    "primary_gw_neg": lambda r: _g(r, "primary", "gw_neg_mean"),
    "primary_gw_all": lambda r: _g(r, "primary", "gw_all_mean"),
    "primary_gw_mag": lambda r: _g(r, "primary", "gw_mag_mean"),
    "primary_comp_changed": lambda r: _g(r, "primary", "competitor_changed_mean"),
    # rescue / regression (descriptive transition counts, seed-level means)
    "rescued": lambda r: _g(r, "rescue", "rescued"),
    "regressed": lambda r: _g(r, "rescue", "regressed"),
    "net_recovery": lambda r: _g(r, "rescue", "net_recovery"),
    "rescue_rate": lambda r: _g(r, "rescue", "rescue_rate"),
    "regression_rate": lambda r: _g(r, "rescue", "regression_rate"),
    "seq_rescued": lambda r: _g(r, "seq_rescue", "rescued_seq"),
    "seq_broken": lambda r: _g(r, "seq_rescue", "broken_seq"),
    # operation-level
    "SET": lambda r: _g(r, "by_final_op", "SET"),
    "SWAP": lambda r: _g(r, "by_final_op", "SWAP"),
    "ADD_no_wrap": lambda r: _g(r, "by_final_op", "ADD_no_wrap"),
    "ADD_wrap": lambda r: _g(r, "by_final_op", "ADD_wrap"),
    "ctl_repeated": lambda r: _g(r, "controlled", "repeated_add", "terminal_em"),
    "ctl_repeated_dest": lambda r: _g(r, "controlled", "repeated_add", "committed_dest_acc_at_add"),
    "ctl_wrap": lambda r: _g(r, "controlled", "add_wrap", "terminal_em"),
    "ctl_wrap_dest": lambda r: _g(r, "controlled", "add_wrap", "committed_dest_acc_at_add"),
    "ctl_isolated": lambda r: _g(r, "controlled", "isolated_add", "terminal_em"),
    # gate shape
    "gs_before_mean": lambda r: _g(r, "gate_shape", "all_gates", "before", "mean"),
    "gs_after_mean": lambda r: _g(r, "gate_shape", "all_gates", "after", "mean"),
    "gs_delta_mean": lambda r: _diff(r, "gate_shape", "all_gates", "after", "mean",
                                     b_keys=("gate_shape", "all_gates", "before", "mean")),
    "gs_delta_median": lambda r: _diff(r, "gate_shape", "all_gates", "after", "median",
                                       b_keys=("gate_shape", "all_gates", "before", "median")),
    "gs_before_entropy": lambda r: _g(r, "gate_shape", "all_gates", "before", "entropy"),
    "gs_after_entropy": lambda r: _g(r, "gate_shape", "all_gates", "after", "entropy"),
    "gs_delta_entropy": lambda r: _diff(r, "gate_shape", "all_gates", "after", "entropy",
                                        b_keys=("gate_shape", "all_gates", "before", "entropy")),
    "gs_abs_change": lambda r: _g(r, "gate_shape", "transition_stats", "abs_change", "mean"),
    "gs_signed_change": lambda r: _g(r, "gate_shape", "transition_stats", "signed_change", "mean"),
    "gs_transfer_change": lambda r: _g(r, "gate_shape", "transition_stats", "transfer_change", "mean"),
    "gs_frac_lt025_before": lambda r: _g(r, "gate_shape", "all_gates", "before", "frac_lt_025"),
    "gs_frac_lt025_after": lambda r: _g(r, "gate_shape", "all_gates", "after", "frac_lt_025"),
    "gs_frac_ge075_before": lambda r: _g(r, "gate_shape", "all_gates", "before", "frac_ge_075"),
    "gs_frac_ge075_after": lambda r: _g(r, "gate_shape", "all_gates", "after", "frac_ge_075"),
}


# ---------------------------------------------------------------------------
# indexing / aggregation
# ---------------------------------------------------------------------------

def load(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data["records"], data["meta"]


def index(records: List[Dict[str, Any]]) -> Dict[Tuple[str, float], List[Dict[str, Any]]]:
    idx: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in records:
        idx.setdefault((r["model"], float(r["kappa"])), []).append(r)
    for k in idx:
        idx[k].sort(key=lambda r: r["seed"])
    return idx


def summarize(idx: Dict[Tuple[str, float], List[Dict[str, Any]]], model: str) -> Dict[str, Any]:
    keys = [k for k in idx if k[0] == model]
    kappas = sorted({k[1] for k in keys})
    out: Dict[str, Any] = {"kappas": kappas, "per_kappa": {}, "paired_vs_baseline": {}}
    for kp in kappas:
        recs = idx[(model, kp)]
        entry: Dict[str, Any] = {"n_seeds": len(recs), "per_seed": {}, "mean": {}, "sd": {},
                                 "band": {}}
        for name, fn in SEED_METRICS.items():
            vals = np.array([fn(r) if fn(r) is not None else np.nan for r in recs], float)
            entry["per_seed"][name] = [None if np.isnan(v) else float(v) for v in vals]
            entry["mean"][name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
            entry["sd"][name] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else 0.0
        if recs and "by_stratum" in recs[0]["rescue"]:
            for sname in ("low", "mid", "high"):
                entry.setdefault("strata_comm_acc", {})[sname] = float(np.mean(
                    [r["rescue"]["by_stratum"][sname]["comm_acc"] for r in recs]))
                entry.setdefault("strata_rescue_rate", {})[sname] = float(np.mean(
                    [r["rescue"]["by_stratum"][sname]["rescue_rate"] for r in recs]))
                entry.setdefault("strata_regression_rate", {})[sname] = float(np.mean(
                    [r["rescue"]["by_stratum"][sname]["regression_rate"] for r in recs]))
        out["per_kappa"][str(kp)] = entry
    if str(BASELINE_KAPPA) in out["per_kappa"]:
        base = out["per_kappa"][str(BASELINE_KAPPA)]
        for kp in kappas:
            if kp == BASELINE_KAPPA:
                continue
            cur = out["per_kappa"][str(kp)]
            pv: Dict[str, Any] = {}
            for name in SEED_METRICS:
                a = np.array([np.nan if v is None else v for v in cur["per_seed"][name]], float)
                b = np.array([np.nan if v is None else v for v in base["per_seed"][name]], float)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 2:
                    p = paired(a[ok], b[ok])
                    p["band"] = effect_band(p["cohens_dz"])
                    pv[name] = p
            out["paired_vs_baseline"][str(kp)] = pv
    return out


# ---------------------------------------------------------------------------
# decision logic (pre-registered)
# ---------------------------------------------------------------------------

def decide(summary: Dict[str, Any]) -> Dict[str, Any]:
    pv = summary["paired_vs_baseline"].get(str(SHARPEN_KAPPA), {})
    mean = summary["per_kappa"][str(SHARPEN_KAPPA)]["mean"]

    beneficial_seq, harmful_seq = [], []
    for ep in ("t10_em", "t20_em", "t5_em", "t15_em"):
        if ep in pv:
            if sig_effect(pv[ep], beneficial=True):
                beneficial_seq.append({"endpoint": ep, "mean_diff": pv[ep]["mean_diff"],
                                       "dz": pv[ep]["cohens_dz"], "p": pv[ep].get("paired_t_p")})
            if sig_effect(pv[ep], beneficial=False):
                harmful_seq.append({"endpoint": ep, "mean_diff": pv[ep]["mean_diff"],
                                    "dz": pv[ep]["cohens_dz"], "p": pv[ep].get("paired_t_p")})

    net_rescue_positive = bool(mean.get("net_recovery") is not None and mean["net_recovery"] > 0)
    margin_improves = []
    for mk in ("primary_rho", "primary_gw_pos", "primary_gw_all", "primary_comm_acc", "primary_dm_commit"):
        if mk in pv and pv[mk]["sign_positive"] >= 4 and pv[mk]["mean_diff"] > 0:
            margin_improves.append({"metric": mk, "mean_diff": pv[mk]["mean_diff"],
                                    "dz": pv[mk]["cohens_dz"]})
    neg_over_delivery = []
    for mk in ("primary_gw_neg",):
        if mk in pv and pv[mk]["sign_positive"] >= 4 and pv[mk]["mean_diff"] > 0:
            neg_over_delivery.append({"metric": mk, "mean_diff": pv[mk]["mean_diff"]})

    op_regression = []
    for op in ("SET", "SWAP", "ADD_no_wrap", "ADD_wrap"):
        if op in pv and sig_effect(pv[op], beneficial=False):
            op_regression.append({"operation": op, "mean_diff": pv[op]["mean_diff"],
                                  "dz": pv[op]["cohens_dz"]})

    local_benefit = bool(margin_improves) or net_rescue_positive

    if beneficial_seq and net_rescue_positive and not op_regression:
        verdict = "A_sequence_beneficial_mechanistically_supportive"
    elif harmful_seq:
        verdict = "D_harmful"
    elif local_benefit:
        verdict = "B_locally_beneficial_sequence_neutral"
    else:
        verdict = "C_no_meaningful_benefit"

    return {
        "verdict": verdict,
        "beneficial_sequence_effects": beneficial_seq,
        "harmful_sequence_effects": harmful_seq,
        "net_transition_rescue_positive": net_rescue_positive,
        "net_recovery_mean": mean.get("net_recovery"),
        "margin_or_delivery_improvements": margin_improves,
        "negative_coordinate_over_delivery": neg_over_delivery,
        "operation_level_regressions": op_regression,
        "sub_findings": {
            "no_beneficial_sequence_level_effect": not beneficial_seq,
            "sequence_level_harmful": bool(harmful_seq),
            "local_mechanism_moves": bool(margin_improves),
            "net_transition_rescue_positive": net_rescue_positive,
            "operation_level_regression": bool(op_regression),
        },
        "rule_note": ("Pre-registered: A requires a sign-consistent (>=4/5) conventionally large "
                      "(|dz|>=0.8, p<0.05) BENEFICIAL sequence-level effect at T10/T20 with net "
                      "candidate->commit rescue and no large operation regression. B: the local "
                      "commit mechanism moves (margin / coordinate delivery) or net rescue is "
                      "positive but there is no beneficial sequence-level effect. D: harmful "
                      "sequence-level effect. C: neither. Local-only improvement with T10/T20 "
                      "unchanged classifies as B, not A."),
    }


# ---------------------------------------------------------------------------
# Experiment J historical reference (read only; never modified)
# ---------------------------------------------------------------------------

def load_j_reference(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        j = json.load(f)
    s = j.get("summaries", {}).get("E_candidate::scale", {})
    out: Dict[str, Any] = {}
    for gkey in ("1.0", "1.1", "1.25", "2.0"):
        if gkey in s.get("per_gamma", {}):
            m = s["per_gamma"][gkey]["mean"]
            out[gkey] = {"t10_em": m.get("t10_em"), "t20_em": m.get("t20_em"),
                         "primary_rho": m.get("primary_rho"), "primary_gw_pos": m.get("primary_gw_pos"),
                         "primary_comm_acc": m.get("primary_comm_acc"),
                         "rescued": m.get("rescued"), "regressed": m.get("regressed"),
                         "SET": m.get("SET"), "SWAP": m.get("SWAP"),
                         "ADD_wrap": m.get("ADD_wrap")}
    return out


# ---------------------------------------------------------------------------
# compute parity (analytic + measured latency)
# ---------------------------------------------------------------------------

def compute_parity(meta: Dict[str, Any], idx: Dict[Tuple[str, float], List[Dict[str, Any]]]) -> Dict[str, Any]:
    state, obs, hid, R, V = 212, 256, 146, 4, 10
    macs = obs * state + state * state + (state + obs) * hid + hid * state + state * (R * V)
    # sharpen: clamp (>=2 cmp) + log + log1p/sub + sub + mul + sigmoid(exp,add,div) ~= 10 elementwise ops
    extra = state * 10
    base_lat = [r.get("rollout_time_s") for r in idx.get((PRIMARY_MODEL, BASELINE_KAPPA), [])]
    sh_lat = [r.get("rollout_time_s") for r in idx.get((PRIMARY_MODEL, SHARPEN_KAPPA), [])]
    base_lat = [x for x in base_lat if x]; sh_lat = [x for x in sh_lat if x]
    return {
        "params_total": 226358, "additional_trainable_params": 0,
        "extra_ops_per_step_approx": extra,
        "macs_per_step_approx": macs,
        "extra_elementwise_ops_fraction": extra / (macs + extra),
        "inference_flops_per_episode_approx": 10 * macs,
        "note": ("Sharpening adds only element-wise ops (clamp/log/log1p/sub/mul/exp/div) per state "
                 "coordinate per step; zero trainable parameters; no optimizer; inference only."),
        "rollout_time_s_baseline_mean": float(np.mean(base_lat)) if base_lat else None,
        "rollout_time_s_sharpen_mean": float(np.mean(sh_lat)) if sh_lat else None,
        "wall_clock_s_total": meta.get("wall_clock_s"),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(idx, summaries, decision, j_ref, out_dir, prefix="experiment_k") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # summaries keyed by (model, kappa)
    EB = summaries[(PRIMARY_MODEL, BASELINE_KAPPA)]
    ES = summaries[(PRIMARY_MODEL, SHARPEN_KAPPA)]

    def mv(entry, metric):
        vals = entry["mean"].get(metric)
        return np.nan if vals is None else vals

    def sv(entry, metric):
        s = entry["sd"].get(metric)
        return 0.0 if s is None else s

    # 1. T10/T20 EM baseline vs sharpening
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, metric, lab in ((axes[0], "t10_em", "T=10 terminal EM"), (axes[1], "t20_em", "T=20 terminal EM")):
        ax.bar([0, 1], [mv(EB, metric), mv(ES, metric)], yerr=[sv(EB, metric), sv(ES, metric)],
               capsize=4, color=["tab:blue", "tab:orange"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["kappa=1 (identity)", "kappa=2 (sharpen)"])
        ax.set_ylabel(lab); ax.set_title(lab); ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate([mv(EB, metric), mv(ES, metric)]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("E: primary sequence-level endpoints (mean ± SD over 5 seeds)")
    save(fig, "t10_t20_em")

    # 2. full horizon curves
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for entry, lab, c in ((EB, "kappa=1 (identity)", "tab:blue"), (ES, "kappa=2 (sharpen)", "tab:orange")):
        ms = [entry["mean"].get(f"t{L}_em") for L in HORIZONS]
        sds = [entry["sd"].get(f"t{L}_em") for L in HORIZONS]
        ax.errorbar(HORIZONS, ms, yerr=sds, marker="o", capsize=3, label=lab, color=c)
    ax.set_xlabel("sequence length T (zero-shot)"); ax.set_ylabel("terminal exact match")
    ax.set_xticks(HORIZONS); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("E: horizon EM, identity vs sharpening (mean ± SD)")
    save(fig, "horizons_em")

    # 3. rescue vs regression
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xs = np.arange(2)
    axes[0].bar(xs - 0.2, [mv(EB, "rescued"), mv(ES, "rescued")], 0.4, label="rescued", color="tab:green")
    axes[0].bar(xs + 0.2, [mv(EB, "regressed"), mv(ES, "regressed")], 0.4, label="regressed", color="tab:red")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(["kappa=1", "kappa=2"]); axes[0].legend(fontsize=8)
    axes[0].set_ylabel("candidate→commit transitions (mean/seed)"); axes[0].set_title("transition rescue/regression")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(xs - 0.2, [mv(EB, "seq_rescued"), mv(ES, "seq_rescued")], 0.4, label="rescued seqs", color="tab:green")
    axes[1].bar(xs + 0.2, [mv(EB, "seq_broken"), mv(ES, "seq_broken")], 0.4, label="broken seqs", color="tab:red")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(["kappa=1", "kappa=2"]); axes[1].legend(fontsize=8)
    axes[1].set_ylabel("sequences (mean/seed)"); axes[1].set_title("sequence rescue/breakage")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("E: what sharpening fixes vs what it breaks")
    save(fig, "rescue_regression")

    # 4. gate distribution before/after (all gate values; kappa=2 record)
    rec2 = idx[(PRIMARY_MODEL, SHARPEN_KAPPA)][0]
    gh = rec2["gate_shape"]["gate_value_hist"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    edges = np.asarray(gh["before"]["edges"]); ctr = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(ctr, np.asarray(gh["before"]["counts"]) / max(gh["before"]["n"], 1), label="before (f)", color="tab:blue")
    ax.plot(ctr, np.asarray(gh["after"]["counts"]) / max(gh["after"]["n"], 1), label="after (f')", color="tab:orange")
    ax.axvline(0.5, color="black", ls=":", lw=0.9, label="0.5 (fixed point)")
    ax.set_xlabel("gate value"); ax.set_ylabel("fraction of coordinates"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("E: gate coordinate distribution before/after sharpening (kappa=2)")
    save(fig, "gate_distribution")

    # 5. gate entropy / distribution shape
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    eh = rec2["gate_shape"]["entropy_hist"]
    ee = np.asarray(eh["before"]["edges"]); ec = 0.5 * (ee[1:] + ee[:-1])
    axes[0].plot(ec, np.asarray(eh["before"]["counts"]) / max(eh["before"]["n"], 1), label="before", color="tab:blue")
    axes[0].plot(ec, np.asarray(eh["after"]["counts"]) / max(eh["after"]["n"], 1), label="after", color="tab:orange")
    axes[0].set_xlabel("per-transition gate entropy (bits)"); axes[0].set_ylabel("fraction"); axes[0].legend(fontsize=8)
    axes[0].set_title("per-transition gate entropy", fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].bar([0, 1], [mv(EB, "gate_entropy"), mv(ES, "gate_entropy")], 0.5, color=["tab:blue", "tab:orange"])
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["kappa=1", "kappa=2"])
    axes[1].set_ylabel("mean gate entropy (bits)"); axes[1].set_title("aggregate gate entropy", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("E: gate entropy / distribution shape")
    save(fig, "gate_entropy_shape")

    # 6. decision-margin delivery
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    metrics = [("primary_rho", "ρ margin (median)"), ("primary_m_cand", "candidate margin (median)"),
               ("primary_m_commit", "commit margin (median)"), ("primary_shortfall", "candidate→commit gap")]
    xs = np.arange(len(metrics)); w = 0.38
    ax.bar(xs - w / 2, [mv(EB, m) for m, _ in metrics], w, label="kappa=1", color="tab:blue")
    ax.bar(xs + w / 2, [mv(ES, m) for m, _ in metrics], w, label="kappa=2", color="tab:orange")
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in metrics], fontsize=8, rotation=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: decision-margin delivery (baseline candidate-correct general ADD)")
    save(fig, "margin_delivery")

    # 7. gw_pos vs gw_neg
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = np.arange(2); w = 0.38
    ax.bar(xs - w / 2, [mv(EB, "primary_gw_pos"), mv(ES, "primary_gw_pos")], w, label="gw_pos (helpful)", color="tab:green")
    ax.bar(xs + w / 2, [mv(EB, "primary_gw_neg"), mv(ES, "primary_gw_neg")], w, label="gw_neg (harmful)", color="tab:red")
    ax.set_xticks(xs); ax.set_xticklabels(["kappa=1", "kappa=2"]); ax.legend(fontsize=8)
    ax.set_ylabel("contribution-weighted gate (mean)"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: positive vs negative coordinate delivery")
    save(fig, "gw_pos_vs_neg")

    # 8. candidate→commit gap
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar([0, 1], [mv(EB, "primary_comm_acc"), mv(ES, "primary_comm_acc")], 0.5,
                color=["tab:blue", "tab:orange"])
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["kappa=1", "kappa=2"])
    axes[0].set_ylabel("committed acc (baseline cand-correct set)"); axes[0].set_title("committed accuracy", fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar([0, 1], [mv(EB, "primary_shortfall"), mv(ES, "primary_shortfall")], 0.5,
                color=["tab:blue", "tab:orange"])
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["kappa=1", "kappa=2"])
    axes[1].set_ylabel("margin shortfall (median)"); axes[1].set_title("candidate→commit margin gap", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("E: candidate→commit gap")
    save(fig, "commit_gap")

    # 9. candidate-margin-stratified effects
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, key, ttl in ((axes[0], "strata_comm_acc", "committed acc by candidate-margin tercile"),
                         (axes[1], "strata_rescue_rate", "rescue rate by candidate-margin tercile")):
        xs = np.arange(3); w = 0.38
        ax.bar(xs - w / 2, [EB.get(key, {}).get(s) for s in ("low", "mid", "high")], w, label="kappa=1", color="tab:blue")
        ax.bar(xs + w / 2, [ES.get(key, {}).get(s) for s in ("low", "mid", "high")], w, label="kappa=2", color="tab:orange")
        ax.set_xticks(xs); ax.set_xticklabels(["low", "mid", "high"]); ax.legend(fontsize=8)
        ax.set_title(ttl, fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("E: effect stratified by BASELINE candidate-margin tercile")
    save(fig, "margin_strata")

    # 10. ADD operation/category comparison
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    metrics = [("ADD_no_wrap", "ADD no-wrap"), ("ADD_wrap", "ADD wrap"),
               ("ctl_repeated", "controlled repeated ADD"), ("ctl_wrap", "controlled wrap"),
               ("ctl_isolated", "controlled isolated ADD")]
    xs = np.arange(len(metrics)); w = 0.38
    ax.bar(xs - w / 2, [mv(EB, m) for m, _ in metrics], w, label="kappa=1", color="tab:blue")
    ax.bar(xs + w / 2, [mv(ES, m) for m, _ in metrics], w, label="kappa=2", color="tab:orange")
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in metrics], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("EM"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: ADD operation/category comparison")
    save(fig, "add_ops")

    # 11. SET/SWAP regression comparison
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    metrics = [("SET", "SET"), ("SWAP", "SWAP"), ("ADD_no_wrap", "ADD no-wrap"), ("ADD_wrap", "ADD wrap")]
    xs = np.arange(len(metrics)); w = 0.38
    ax.bar(xs - w / 2, [mv(EB, m) for m, _ in metrics], w, label="kappa=1", color="tab:blue")
    ax.bar(xs + w / 2, [mv(ES, m) for m, _ in metrics], w, label="kappa=2", color="tab:orange")
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in metrics])
    ax.set_ylabel("final-op EM"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: operation-level safety (regression check)")
    save(fig, "set_swap_regression")

    # 12. off-path rescue
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    op_s = [r["rescue"].get("offpath", {}) for r in idx[(PRIMARY_MODEL, SHARPEN_KAPPA)]]
    n_suff = np.mean([d.get("n_base_fail_sufficient", 0) for d in op_s]) if op_s else 0
    rescue_rate = np.mean([d.get("rescue_rate_sufficient", 0.0) for d in op_s]) if op_s else 0.0
    comp_ch = np.mean([d.get("competitor_changed_rate_sufficient_fail", 0.0) for d in op_s]) if op_s else 0.0
    ax.bar(["off-path failures\n(ρ≥α*_full, n/seed)", "rescued fraction", "competitor changed\nat commit"],
           [n_suff, rescue_rate, comp_ch], color=["tab:gray", "tab:green", "tab:purple"])
    for i, v in enumerate([n_suff, rescue_rate, comp_ch]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("E: off-path subset under sharpening (kappa=2)")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "offpath_rescue")

    # 13. coordinate-wise gate-change distribution
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    sd = rec2["gate_shape"]["coord_delta_hist"]; ad = rec2["gate_shape"]["coord_abs_hist"]
    e = np.asarray(sd["edges"]); c = 0.5 * (e[1:] + e[:-1])
    axes[0].bar(c, np.asarray(sd["counts"]) / max(sd["n"], 1), width=(e[1] - e[0]), color="tab:orange")
    axes[0].axvline(0, color="black", lw=0.8); axes[0].set_xlabel("signed change f'−f")
    axes[0].set_ylabel("fraction of coordinates"); axes[0].set_title("signed change (primary set)", fontsize=9)
    axes[0].grid(alpha=0.3)
    e = np.asarray(ad["edges"]); c = 0.5 * (e[1:] + e[:-1])
    axes[1].bar(c, np.asarray(ad["counts"]) / max(ad["n"], 1), width=(e[1] - e[0]), color="tab:red")
    axes[1].set_xlabel("|f'−f|"); axes[1].set_title("absolute change (primary set)", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle("E: coordinate-wise gate change under sharpening (kappa=2, κ preserves coordinate identity)")
    save(fig, "coord_gate_change")

    # 14. comparison against Experiment J historical reference
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    labels = ["E baseline\n(J γ=1.0)", "K κ=2.0", "J γ=1.10", "J γ=1.25"]
    t10 = [mv(EB, "t10_em"), mv(ES, "t10_em"),
           j_ref.get("1.1", {}).get("t10_em"), j_ref.get("1.25", {}).get("t10_em")]
    t20 = [mv(EB, "t20_em"), mv(ES, "t20_em"),
           j_ref.get("1.1", {}).get("t20_em"), j_ref.get("1.25", {}).get("t20_em")]
    rho = [mv(EB, "primary_rho"), mv(ES, "primary_rho"),
           j_ref.get("1.1", {}).get("primary_rho"), j_ref.get("1.25", {}).get("primary_rho")]
    gwp = [mv(EB, "primary_gw_pos"), mv(ES, "primary_gw_pos"),
           j_ref.get("1.1", {}).get("primary_gw_pos"), j_ref.get("1.25", {}).get("primary_gw_pos")]
    for ax, vals, ttl in ((axes[0], t10, "T=10 EM"), (axes[1], rho, "ρ margin (median)"),
                          (axes[2], gwp, "gw_pos")):
        vv = [np.nan if v is None else v for v in vals]
        ax.bar(range(len(labels)), vv, color=["tab:blue", "tab:orange", "tab:gray", "tab:brown"][:len(labels)])
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(ttl, fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("E: K κ=2.0 vs Experiment J historical γ reference (J values are historical, unchanged)")
    save(fig, "vs_j_reference")

    return paths


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build(records, meta, j_ref):
    idx = index(records)
    models = sorted({m for (m, _k) in idx})
    summaries: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for m in models:
        s = summarize(idx, m)
        for kp in s["kappas"]:
            summaries[(m, kp)] = {"kappas": s["kappas"], "per_kappa": s["per_kappa"],
                                  "paired_vs_baseline": s["paired_vs_baseline"],
                                  "mean": s["per_kappa"][str(kp)]["mean"],
                                  "sd": s["per_kappa"][str(kp)]["sd"],
                                  "per_seed": s["per_kappa"][str(kp)]["per_seed"],
                                  **{k: v for k, v in s["per_kappa"][str(kp)].items() if k.startswith("strata")}}
    res: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "preregistration": meta["preregistration"],
            "n_records": meta["n_records"], "seeds": meta["seeds"],
            "checkpoints": meta["checkpoints"],
            "statistical_discipline": ("Seeds (5 frozen checkpoints) are independent units; transition-level "
                                       "rescue/regression is descriptive. Paired seed-level inference vs kappa=1."),
        },
        "summaries": {f"{m}::kappa{kp}": summaries[(m, kp)] for (m, kp) in summaries},
        "j_historical_reference": j_ref,
        "compute_parity": compute_parity(meta, idx),
    }
    if (PRIMARY_MODEL, BASELINE_KAPPA) in summaries:
        res["decision_E_primary"] = decide(summaries[(PRIMARY_MODEL, BASELINE_KAPPA)])
    if (CONTROL_MODEL, BASELINE_KAPPA) in summaries:
        res["decision_F_control"] = decide(summaries[(CONTROL_MODEL, BASELINE_KAPPA)])
    return res, summaries, idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment K analysis")
    ap.add_argument("--records", default="experiments/experiment_k_records.json")
    ap.add_argument("--output", default="experiments/experiment_k_results.json")
    ap.add_argument("--j-results", default=J_RESULTS_DEFAULT)
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_k")
    args = ap.parse_args()

    records, meta = load(args.records)
    j_ref = load_j_reference(args.j_results)
    res, summaries, idx = build(records, meta, j_ref)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    print("E primary verdict:", res["decision_E_primary"]["verdict"])
    if "decision_F_control" in res:
        print("F control verdict:", res["decision_F_control"]["verdict"])
    paths = make_figures(idx, {(m, kp): summaries[(m, kp)] for (m, kp) in summaries},
                         res["decision_E_primary"], j_ref, args.fig_dir, prefix=args.fig_prefix)
    print("Figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
