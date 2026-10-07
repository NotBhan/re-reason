#!/usr/bin/env python3
"""Experiment J — analysis of the controlled inference-time commit-rule intervention.

Reads experiments/experiment_j_records.json (runner output), performs seed-level paired
statistics vs the gamma=1 baseline, rescue/regression accounting, candidate-margin
stratification, operation-level regression checks, the scrambled control comparison, the
pre-registered A/B/C/D decision, and produces the required figures.

Statistical discipline: the five frozen E checkpoints (seeds 42-46) are the independent
experimental units. Transition-level rescue/regression counts are descriptive; primary
inference is seed-level paired vs gamma=1 (mean +/- SD, Cohen's dz, paired t, sign k/5).
Cohen's dz bands (|dz|<0.2 negligible, <0.5 small, <0.8 medium, >=0.8 large) are declared
conventions, not data-tuned cutoffs.

Reproduce:
    python experiments/analyze_experiment_j.py \
        --records experiments/experiment_j_records.json \
        --output experiments/experiment_j_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_j
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
FAMILIES = ["scale", "scramble", "one_minus"]
BASELINE_GAMMA = 1.0
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20]
CONTROLLED_NAMES = ["isolated_add", "add_then_set", "repeated_add", "add_no_wrap", "add_wrap"]


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
        try:
            out["wilcoxon_p"] = float(st.wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
        except Exception:
            out["wilcoxon_p"] = None
    return out


def effect_band(dz: Optional[float]) -> str:
    if dz is None or (isinstance(dz, float) and np.isnan(dz)):
        return "na"
    a = abs(dz)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def sig_effect(p: Dict[str, Any], beneficial: bool) -> bool:
    """Pre-registered 'sequence-level effect': sign-consistent >=4/5, |dz|>=0.8, p<0.05, sign."""
    if p is None or p.get("cohens_dz") is None or np.isnan(p.get("cohens_dz", float("nan"))):
        return False
    same = p["sign_positive"] >= 4 if beneficial else p["sign_negative"] >= 4
    pv = p.get("paired_t_p")
    return bool(same and abs(p["cohens_dz"]) >= 0.8 and pv is not None and pv < 0.05)


# ---------------------------------------------------------------------------
# indexing / aggregation
# ---------------------------------------------------------------------------

def load(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data["records"], data["meta"]


def index(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, float], List[Dict[str, Any]]]:
    idx: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for r in records:
        idx.setdefault((r["model"], r["family"], float(r["gamma"])), []).append(r)
    for k in idx:
        idx[k].sort(key=lambda r: r["seed"])
    return idx


def metric_of(r: Dict[str, Any], path: str) -> Optional[float]:
    cur: Any = r
    for p in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
    return None if cur is None else float(cur)


SEED_METRICS = {
    "t10_em": "horizons.10.em", "t20_em": "horizons.20.em", "t5_em": "horizons.5.em",
    "t15_em": "horizons.15.em", "t2_em": "horizons.2.em", "t3_em": "horizons.3.em",
    "t8_em": "horizons.8.em", "t1_em": "horizons.1.em",
    "comm_written": "test.comm_written_mean", "cand_written": "test.cand_written_mean",
    "comm_full": "test.comm_full_mean", "first_real_err": "test.first_real_error_rate",
    "gate_mean": "test.gate_mean", "gate_scaled_mean": "test.gate_scaled_mean",
    "primary_comm_acc": "primary.comm_acc", "primary_rho": "primary.rho_median",
    "primary_gw_pos": "primary.gw_pos_mean", "primary_gw_neg": "primary.gw_neg_mean",
    "primary_m_cand": "primary.m_cand_median",
    "rescued": "rescue.rescued", "regressed": "rescue.regressed",
    "net_recovery": "rescue.net_recovery", "rescue_rate": "rescue.rescue_rate",
    "regression_rate": "rescue.regression_rate",
    "seq_rescued": "seq_rescue.rescued_seq", "seq_broken": "seq_rescue.broken_seq",
    "SET": "by_final_op.SET", "SWAP": "by_final_op.SWAP",
    "ADD_no_wrap": "by_final_op.ADD_no_wrap", "ADD_wrap": "by_final_op.ADD_wrap",
    "ctl_repeated": "controlled.repeated_add.terminal_em",
    "ctl_repeated_dest": "controlled.repeated_add.committed_dest_acc_at_add",
    "ctl_wrap": "controlled.add_wrap.terminal_em",
    "ctl_wrap_dest": "controlled.add_wrap.committed_dest_acc_at_add",
}


def summarize(idx: Dict[Tuple[str, str, float], List[Dict[str, Any]]], model: str, family: str
              ) -> Dict[str, Any]:
    keys = [k for k in idx if k[0] == model and k[1] == family]
    gammas = sorted({k[2] for k in keys})
    out: Dict[str, Any] = {"gammas": gammas, "per_gamma": {}, "paired_vs_baseline": {}}
    for g in gammas:
        recs = idx[(model, family, g)]
        entry: Dict[str, Any] = {"n_seeds": len(recs), "per_seed": {}, "mean": {}, "sd": {}}
        for name, path in SEED_METRICS.items():
            vals = np.array([metric_of(r, path) if metric_of(r, path) is not None else np.nan for r in recs], float)
            entry["per_seed"][name] = [None if np.isnan(v) else float(v) for v in vals]
            entry["mean"][name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
            entry["sd"][name] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else 0.0
        # strata comm acc means
        if recs and "by_stratum" in recs[0]["rescue"]:
            for sname in ("low", "mid", "high"):
                vals = [r["rescue"]["by_stratum"][sname]["comm_acc"] for r in recs]
                entry.setdefault("strata_comm_acc", {})[sname] = float(np.mean(vals))
                entry.setdefault("strata_rescue_rate", {})[sname] = float(np.mean(
                    [r["rescue"]["by_stratum"][sname]["rescue_rate"] for r in recs]))
                entry.setdefault("strata_regression_rate", {})[sname] = float(np.mean(
                    [r["rescue"]["by_stratum"][sname]["regression_rate"] for r in recs]))
        out["per_gamma"][str(g)] = entry
    if str(BASELINE_GAMMA) in out["per_gamma"]:
        base = out["per_gamma"][str(BASELINE_GAMMA)]
        for g in gammas:
            if g == BASELINE_GAMMA:
                continue
            cur = out["per_gamma"][str(g)]
            pv: Dict[str, Any] = {}
            for name in SEED_METRICS:
                a = np.array([np.nan if v is None else v for v in cur["per_seed"][name]], float)
                b = np.array([np.nan if v is None else v for v in base["per_seed"][name]], float)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 2:
                    pv[name] = paired(a[ok], b[ok])
            out["paired_vs_baseline"][str(g)] = pv
    return out


# ---------------------------------------------------------------------------
# decision logic (pre-registered)
# ---------------------------------------------------------------------------

def decide(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-registered A/B/C/D rule, applied to the hypothesis direction (gamma > 1) for
    sequence-level effects, with gamma < 1 reported separately as the transfer-reduction
    control (reducing transfer is expected to be harmful and is not the hypothesis direction)."""
    pv = summary["paired_vs_baseline"]
    amp = [g for g in pv if float(g) > 1.0]
    red = [g for g in pv if float(g) < 1.0]

    def seq_effects(gs):
        ben, har = [], []
        for g in gs:
            for ep in ("t10_em", "t20_em"):
                if ep in pv[g]:
                    if sig_effect(pv[g][ep], beneficial=True):
                        ben.append({"gamma": float(g), "endpoint": ep, "mean_diff": pv[g][ep]["mean_diff"],
                                    "dz": pv[g][ep]["cohens_dz"], "p": pv[g][ep].get("paired_t_p")})
                    if sig_effect(pv[g][ep], beneficial=False):
                        har.append({"gamma": float(g), "endpoint": ep, "mean_diff": pv[g][ep]["mean_diff"],
                                    "dz": pv[g][ep]["cohens_dz"], "p": pv[g][ep].get("paired_t_p")})
        return ben, har

    beneficial_seq, harmful_seq = seq_effects(amp)
    _, harmful_red = seq_effects(red)

    net_rescue = []
    for g in sorted(amp, key=float):
        m = summary["per_gamma"][g]["mean"]
        if m.get("net_recovery") is not None and m["net_recovery"] > 0:
            net_rescue.append({"gamma": float(g), "mean_net_recovery": m["net_recovery"],
                               "mean_rescued": m["rescued"], "mean_regressed": m["regressed"]})
    op_regression = []
    for g in sorted(amp, key=float):
        for op in ("SET", "SWAP", "ADD_no_wrap"):
            if op in pv[g] and sig_effect(pv[g][op], beneficial=False):
                op_regression.append({"gamma": float(g), "operation": op,
                                      "mean_diff": pv[g][op]["mean_diff"], "dz": pv[g][op]["cohens_dz"]})
    margin_improves = []
    for g in sorted(amp, key=float):
        for mk in ("primary_rho", "primary_gw_pos"):
            if mk in pv[g] and pv[g][mk]["sign_positive"] >= 4 and pv[g][mk]["mean_diff"] > 0:
                margin_improves.append({"gamma": float(g), "metric": mk, "mean_diff": pv[g][mk]["mean_diff"]})

    has_benef = bool(beneficial_seq)
    has_harm = bool(harmful_seq)
    has_rescue = bool(net_rescue)
    has_margin = bool(margin_improves)
    has_opreg = bool(op_regression)

    if has_benef and has_rescue and not has_opreg:
        verdict = "A_strong_causal_support"
    elif has_margin or has_rescue:
        # the intervention causally moves the local commit mechanism, but no beneficial
        # sequence-level effect exists -> partial support (the commit is a real local
        # bottleneck, not the dominant long-horizon one; amplification beyond the net-rescue
        # optimum is harmful).
        verdict = "B_partial_causal_support"
    elif has_harm:
        verdict = "D_harmful"
    else:
        verdict = "C_little_or_no_causal_support"

    return {
        "verdict": verdict,
        "beneficial_sequence_effects_gamma_gt_1": beneficial_seq,
        "harmful_sequence_effects_gamma_gt_1": harmful_seq,
        "harmful_sequence_effects_gamma_lt_1_transfer_reduction_control": harmful_red,
        "net_transition_rescue_gamma_gt_1": net_rescue,
        "margin_improvements_gamma_gt_1": margin_improves,
        "operation_level_regressions_gamma_gt_1": op_regression,
        "sub_findings": {
            "amplified_regime_harmful": bool(has_harm),
            "no_beneficial_sequence_level_effect": not has_benef,
            "local_mechanism_moves_as_hypothesized": has_margin,
            "net_transition_rescue_positive": has_rescue,
        },
        "rule_note": ("Pre-registered: A requires a gamma>1 beneficial sign-consistent (>=4/5) "
                      "large (|dz|>=0.8, p<0.05) sequence-level effect at T10/T20 with net rescue "
                      "and no large operation regression. B: the local commit mechanism moves "
                      "causally (margin delivery / net rescue) but with no beneficial sequence-level "
                      "effect (or mixed). D: harmful sequence-level effect with neither margin nor "
                      "rescue benefit. C: neither. gamma<1 is the transfer-reduction control."),
    }


# ---------------------------------------------------------------------------
# compute parity (analytic)
# ---------------------------------------------------------------------------

def compute_parity(meta: Dict[str, Any]) -> Dict[str, Any]:
    state, obs, hid, R, V = 212, 256, 146, 4, 10
    macs = obs * state + state * state + (state + obs) * hid + hid * state + state * (R * V)
    extra = state * 2  # multiply by gamma + clamp per coordinate
    params = 226358
    return {
        "params_total": params, "additional_trainable_params": 0,
        "extra_ops_per_step": extra,
        "macs_per_step_approx": macs,
        "extra_elementwise_ops_fraction": extra / (macs + extra),
        "note": ("The intervention adds one scalar multiply and one clamp per state coordinate per "
                 "step; zero trainable parameters; no optimizer; inference only. Cost is negligible "
                 "relative to the core."),
        "wall_clock_s_total": meta.get("wall_clock_s"),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(idx, summaries, decision, out_dir, prefix="experiment_j") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    E = summaries[(PRIMARY_MODEL, "scale")]

    def series(summary, metric):
        gs = [g for g in summary["gammas"] if g != BASELINE_GAMMA]
        gs = sorted(summary["gammas"])
        return gs, [summary["per_gamma"][str(g)]["mean"].get(metric) for g in gs], \
            [summary["per_gamma"][str(g)]["sd"].get(metric) for g in gs]

    # 1. terminal EM vs gamma for T=5/10/15/20
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for metric, lab in (("t5_em", "T=5"), ("t10_em", "T=10"), ("t15_em", "T=15"), ("t20_em", "T=20")):
        gs, m, s = series(E, metric)
        ax.errorbar(gs, m, yerr=s, marker="o", capsize=3, label=lab)
    ax.axvline(1.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("gamma (commit gate scale)"); ax.set_ylabel("terminal exact match")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("E: terminal EM vs gamma (mean ± SD over 5 seeds; scale family)")
    save(fig, "em_vs_gamma")

    # 2. per-seed T10 and T20 paired changes
    seeds = sorted({r["seed"] for r in idx[(PRIMARY_MODEL, "scale", BASELINE_GAMMA)]})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, metric, lab in ((axes[0], "t10_em", "T=10"), (axes[1], "t20_em", "T=20")):
        gs = sorted(E["gammas"])
        for si, seed in enumerate(seeds):
            base = E["per_gamma"][str(BASELINE_GAMMA)]["per_seed"][metric][si]
            ys = []
            for g in gs:
                cur = E["per_gamma"][str(g)]["per_seed"][metric][si]
                ys.append(0.0 if g == BASELINE_GAMMA else ((cur - base) if (cur is not None and base is not None) else np.nan))
            ax.plot(gs, ys, marker="o", ms=3, alpha=0.8, label=f"seed {seed}")
        ax.axhline(0, color="black", lw=0.7); ax.axvline(1.0, color="black", ls=":", lw=0.8)
        ax.set_xlabel("gamma"); ax.set_ylabel(f"Δ {lab} EM vs γ=1 (per seed)")
        ax.grid(alpha=0.3); ax.set_title(f"E {lab}: per-seed paired change")
        ax.legend(fontsize=6)
    fig.suptitle("Per-seed paired sequence-level changes vs gamma=1")
    save(fig, "paired_seed_changes")

    # 3. rescue vs regression rate vs gamma
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    gs = sorted(E["gammas"])
    rr = [E["per_gamma"][str(g)]["mean"]["rescue_rate"] for g in gs]
    gr = [E["per_gamma"][str(g)]["mean"]["regression_rate"] for g in gs]
    ax.plot(gs, rr, marker="o", label="rescue rate (baseline failures fixed)")
    ax.plot(gs, gr, marker="s", label="regression rate (baseline successes broken)")
    ax.axvline(1.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("gamma"); ax.set_ylabel("rate")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("E: candidate→commit rescue vs regression vs gamma")
    save(fig, "rescue_regression")

    # 4. candidate->commit gap vs gamma
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    gap = [E["per_gamma"][str(g)]["mean"]["primary_comm_acc"] for g in gs]
    ax.plot(gs, gap, marker="o")
    ax.axvline(1.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("gamma"); ax.set_ylabel("committed accuracy on baseline cand-correct set")
    ax.grid(alpha=0.3)
    ax.set_title("E: candidate→commit accuracy vs gamma (primary set)")
    save(fig, "commit_gap")

    # 5. rho_margin vs gamma
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for metric, lab in (("primary_rho", "ρ margin (median)"), ("primary_gw_pos", "gw positive"),
                        ("primary_gw_neg", "gw negative"), ("primary_m_cand", "m_cand (median)")):
        m = [E["per_gamma"][str(g)]["mean"].get(metric) for g in gs]
        ax.plot(gs, m, marker="o", label=lab)
    ax.axvline(1.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("gamma"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("E: decision-margin delivery vs gamma")
    save(fig, "margin_vs_gamma")

    # 6. positive/negative weighted gate vs gamma (baseline cand-correct: successes vs failures)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, metric, lab in ((axes[0], "primary_gw_pos", "gw positive"), (axes[1], "primary_gw_neg", "gw negative")):
        m = [E["per_gamma"][str(g)]["mean"].get(metric) for g in gs]
        ax.plot(gs, m, marker="o", label=lab)
        ax.axvline(1.0, color="black", ls=":", lw=0.8)
        ax.set_xlabel("gamma"); ax.grid(alpha=0.3); ax.set_title(lab, fontsize=9); ax.legend(fontsize=7)
    fig.suptitle("E: contribution-weighted gate vs gamma (primary set)")
    save(fig, "weighted_gate_vs_gamma")

    # 7. candidate-margin stratified intervention effect
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for sname in ("low", "mid", "high"):
        axes[0].plot(gs, [E["per_gamma"][str(g)].get("strata_comm_acc", {}).get(sname) for g in gs],
                     marker="o", label=f"{sname} margin")
        axes[1].plot(gs, [E["per_gamma"][str(g)].get("strata_rescue_rate", {}).get(sname) for g in gs],
                     marker="o", label=f"{sname} margin")
    for ax, ttl in ((axes[0], "committed acc by stratum"), (axes[1], "rescue rate by stratum")):
        ax.axvline(1.0, color="black", ls=":", lw=0.8); ax.set_xlabel("gamma")
        ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title(ttl, fontsize=9)
    fig.suptitle("E: effect stratified by BASELINE candidate-margin tercile")
    save(fig, "margin_strata")

    # 8. ADD no-wrap vs wrap vs repeated ADD vs gamma
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for metric, lab in (("ADD_no_wrap", "ADD no-wrap (final-op EM)"), ("ADD_wrap", "ADD wrap (final-op EM)"),
                        ("ctl_repeated", "controlled repeated ADD (terminal EM)")):
        m = [E["per_gamma"][str(g)]["mean"].get(metric) for g in gs]
        ax.plot(gs, m, marker="o", label=lab)
    ax.axvline(1.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("gamma"); ax.set_ylabel("EM"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("E: ADD strata vs gamma")
    save(fig, "add_strata")

    # 9. operation-level regression analysis
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for metric, lab in (("SET", "SET"), ("SWAP", "SWAP"), ("ADD_no_wrap", "ADD no-wrap"), ("ADD_wrap", "ADD wrap")):
        m = [E["per_gamma"][str(g)]["mean"].get(metric) for g in gs]
        ax.plot(gs, m, marker="o", label=lab)
    ax.axvline(1.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("gamma"); ax.set_ylabel("final-op EM"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("E: operation-level regression check")
    save(fig, "op_regression")

    # 10. baseline-failure rescue aggregate (transition + sequence) + off-path subset
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(gs, [E["per_gamma"][str(g)]["mean"].get("rescued") for g in gs], marker="o", label="rescued transitions")
    ax.plot(gs, [E["per_gamma"][str(g)]["mean"].get("regressed") for g in gs], marker="s", label="regressed transitions")
    ax.plot(gs, [E["per_gamma"][str(g)]["mean"].get("seq_rescued") for g in gs], marker="^", label="rescued sequences")
    ax.plot(gs, [E["per_gamma"][str(g)]["mean"].get("seq_broken") for g in gs], marker="v", label="broken sequences")
    ax.axvline(1.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("gamma"); ax.set_ylabel("count (mean over seeds)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("E: baseline-failure rescue vs baseline-success damage")
    save(fig, "rescue_examples")

    # 11. coordinate-preserving vs scrambled control
    if (PRIMARY_MODEL, "scramble") in summaries:
        Sc = summaries[(PRIMARY_MODEL, "scramble")]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
        for ax, metric, lab in ((axes[0], "t10_em", "T=10 EM"), (axes[1], "primary_gw_pos", "gw positive")):
            for summ, name, c in ((E, "scale (coord-preserving)", "tab:blue"),
                                  (Sc, "scramble (mean-matched)", "tab:red")):
                gsx = sorted(summ["gammas"])
                ax.plot(gsx, [summ["per_gamma"][str(g)]["mean"].get(metric) for g in gsx],
                        marker="o", color=c, label=name)
            ax.axvline(1.0, color="black", ls=":", lw=0.8); ax.set_xlabel("gamma")
            ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title(lab, fontsize=9)
        fig.suptitle("E: coordinate-preserving scaling vs mean-matched scrambled gate")
        save(fig, "coord_vs_scramble")

    # 12. compute / latency parity
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    cp = compute_parity({"wall_clock_s": None})
    ax.bar(["core MACs/step", "intervention ops/step"], [cp["macs_per_step_approx"], cp["extra_ops_per_step"]])
    ax.set_ylabel("count")
    ax.set_title("Compute parity (analytic): intervention is negligible (%.3f%% of ops)"
                 % (100 * cp["extra_elementwise_ops_fraction"]))
    save(fig, "compute_parity")

    return paths


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build(records, meta):
    idx = index(records)
    summaries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (model, family, _g) in idx:
        summaries[(model, family)] = summarize(idx, model, family)
    res: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "preregistration": meta["preregistration"],
            "n_records": meta["n_records"], "seeds": meta["seeds"],
            "checkpoints": meta["checkpoints"],
            "statistical_discipline": ("Seeds (5 frozen checkpoints) are independent units; "
                                       "transition-level rescue/regression is descriptive. Paired "
                                       "seed-level inference vs gamma=1."),
        },
        "summaries": {f"{k[0]}::{k[1]}": v for k, v in summaries.items()},
        "compute_parity": compute_parity(meta),
    }
    res["decision_E_primary"] = decide(summaries[(PRIMARY_MODEL, "scale")])
    if (CONTROL_MODEL, "scale") in summaries:
        res["decision_F_control"] = decide(summaries[(CONTROL_MODEL, "scale")])
    return res, summaries, idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment J analysis")
    ap.add_argument("--records", default="experiments/experiment_j_records.json")
    ap.add_argument("--output", default="experiments/experiment_j_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_j")
    args = ap.parse_args()

    records, meta = load(args.records)
    res, summaries, idx = build(records, meta)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    print("E primary verdict:", res["decision_E_primary"]["verdict"])
    if "decision_F_control" in res:
        print("F control verdict:", res["decision_F_control"]["verdict"])
    paths = make_figures(idx, summaries, res["decision_E_primary"], args.fig_dir, prefix=args.fig_prefix)
    print("Figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
