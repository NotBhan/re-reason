#!/usr/bin/env python3
"""Experiment O — analysis of the controlled representation-side parity intervention.

Reads experiments/experiment_o_representation_intervention.npz and evaluates whether removing
the W-derived parity component of the candidate (or of the committed state) causally changes the
frozen off-path outcome, relative to a norm-matched orthogonal control, and whether the result
is sequence-level.

PRE-REGISTRATION (declared before inspecting aggregate results)
--------------------------------------------------------------
Primary population: the exact M/L/N off-path set (strict test | ADD | general | candidate correct |
baseline commit wrong | rho >= alpha*_full). Controls: B successes, C non-off-path failures, D all.

Family: kappa in {0, 0.5, 1.0}; candidate-side primary kappa=1; commit-side = stage control.
Matched orthogonal control: same displacement magnitude, direction orthogonal to p_dest.

Per seed s (5 frozen E checkpoints = independent units):
  net(s, config) = (# off-path baseline-wrong -> correct) - (# baseline-correct -> wrong)
  success_regression_rate(s, config) = fraction of baseline successes broken

Specificity (declared bars):
  cand_specific = mean_s net_cand1 > 0 AND >=4/5 net_cand1 > 0
                  AND mean_s (net_cand1 - net_orth1) > 0 AND >=4/5 seeds positive
                  AND mean_s success_reg_cand1 <= mean_s success_reg_orth1 + 0.02
  commit_specific = analogous with commit1 vs commit_orth1
Classification:
  O-A if cand_specific and not commit_specific
  O-B if commit_specific and not cand_specific
  O-C if both
  O-D otherwise
Sequence-level endpoints (T=5/10/15/20 EM) are reported separately; a local rescue without a
sequence improvement is labelled "causally effective locally, sequence-level insufficient".

Reproduce:
    python experiments/analyze_experiment_o.py \
        --npz experiments/experiment_o_representation_intervention.npz \
        --output experiments/experiment_o_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_o
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
SEQ_HORIZONS = ["5", "10", "15", "20"]

# candidate (family, kappa) pairs
CANDS = {"baseline": ("none", 0.0), "cand0.5": ("cand", 0.5), "cand1": ("cand", 1.0),
         "orth0.5": ("orth", 0.5), "orth1": ("orth", 1.0)}
COMMITS = {"commit0.5": ("commit", 0.5), "commit1": ("commit", 1.0),
           "commit_orth0.5": ("commit_orth", 0.5), "commit_orth1": ("commit_orth", 1.0)}
BARS = {"specific_margin": 0.0, "success_reg_margin": 0.02}


def label(fam: str, k: float) -> str:
    return f"{fam}_{k}"


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def group_masks(z, m: str) -> Dict[str, np.ndarray]:
    grp = z[f"{m}__add_group"].astype(int)
    return {"offpath": grp == 0, "success": grp == 1, "nonoff": grp == 2, "all_add": grp >= 0}


def per_seed(z, m: str, mask: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    """Rescue/regression/net vs baseline for each config, per seed, on `mask`."""
    seed_col = z[f"{m}__add_seed"].astype(int)
    base = z[f"{m}__baseline_comm_ok"].astype(bool)
    out: Dict[str, Any] = {}
    for s in SEEDS:
        sm = mask & (seed_col == s)
        b = base[sm]
        row: Dict[str, Any] = {"n": int(sm.sum()), "n_base_fail": int((~b).sum()),
                               "n_base_succ": int(b.sum()), "configs": {}}
        for lab in labels:
            c = z[f"{m}__{lab}__comm_ok"].astype(bool)[sm]
            resc = int((~b & c).sum()); regr = int((b & ~c).sum())
            row["configs"][lab] = {
                "rescued": resc, "regressed": regr, "net": resc - regr,
                "comm_acc": float(c.mean()) if len(c) else None,
                "success_regression_rate": float(regr / max(int(b.sum()), 1)),
            }
        out[str(s)] = row
    return out


def agg_seed(per: Dict[str, Any], lab: str, key: str) -> Dict[str, Any]:
    vals = [per[str(s)]["configs"][lab][key] for s in SEEDS if str(s) in per]
    a = np.array([v for v in vals if v is not None], float)
    return {"mean": float(a.mean()) if a.size else None,
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "per_seed": [float(x) for x in a], "n": int(a.size)}


def paired(a: List[float], b: List[float]) -> Dict[str, Any]:
    x = np.array(a, float); y = np.array(b, float)
    d = x - y; sd = d.std(ddof=1) if d.size > 1 else 0.0
    out = {"a_mean": float(x.mean()), "b_mean": float(y.mean()), "mean_diff": float(d.mean()),
           "sign_positive": int((d > 0).sum()), "sign_negative": int((d < 0).sum()),
           "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"), "n": int(d.size)}
    try:
        out["paired_t_p"] = float(st.ttest_rel(x, y).pvalue)
    except Exception:
        out["paired_t_p"] = None
    return out


def analyze_model(z, meta: Dict[str, Any], m: str) -> Dict[str, Any]:
    masks = group_masks(z, m)
    all_labels = [label(f, k) for f, k in
                  [("none", 0.0), ("cand", 0.5), ("cand", 1.0), ("orth", 0.5), ("orth", 1.0),
                   ("commit", 0.5), ("commit", 1.0), ("commit_orth", 0.5), ("commit_orth", 1.0)]]
    res: Dict[str, Any] = {"groups": {g: int(masks[g].sum()) for g in masks}}
    # per group: baseline vs candidate-config committed accuracy (pooled, descriptive)
    res["group_accuracy"] = {}
    for g, mask in masks.items():
        base = z[f"{m}__baseline_comm_ok"].astype(bool)[mask]
        entry = {"n": int(mask.sum()), "baseline": float(base.mean())}
        for lab in ["cand_0.5", "cand_1.0", "orth_1.0", "commit_1.0", "commit_orth_1.0"]:
            c = z[f"{m}__{lab}__comm_ok"].astype(bool)[mask]
            entry[lab] = float(c.mean())
        res["group_accuracy"][g] = entry

    # primary: rescue/regression on off-path, per seed
    for gname, mask in (("offpath", masks["offpath"]), ("success", masks["success"]),
                        ("nonoff", masks["nonoff"])):
        per = per_seed(z, m, mask, all_labels)
        blk: Dict[str, Any] = {"per_seed": per}
        for lab in all_labels:
            blk[label("net", 0) if False else lab] = {
                "net": agg_seed(per, lab, "net"), "comm_acc": agg_seed(per, lab, "comm_acc"),
                "success_regression_rate": agg_seed(per, lab, "success_regression_rate"),
                "rescued": agg_seed(per, lab, "rescued"), "regressed": agg_seed(per, lab, "regressed"),
            }
        res[f"{gname}_rescue"] = blk

    # sequence endpoints per config/seed
    hr = meta["horizon_results"][m]
    seq: Dict[str, Any] = {}
    for lab in all_labels:
        seq[lab] = {}
        for L in SEQ_HORIZONS:
            vals = [hr[lab][str(s)][L]["em"] for s in SEEDS]
            seq[lab][L] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                           "per_seed": [float(v) for v in vals]}
    res["sequence_em"] = seq
    # canonical-test seq EM (from npz)
    res["canonical_seq_em"] = {}
    for lab in all_labels:
        arr = z[f"{m}__{lab}__seq_em"].astype(float)
        res["canonical_seq_em"][lab] = {"mean": float(arr.mean())}

    # paired sequence endpoints vs baseline (seed-level)
    res["sequence_paired_vs_baseline"] = {}
    for lab in all_labels:
        if lab == label(*CANDS["baseline"]):
            continue
        res["sequence_paired_vs_baseline"][lab] = {
            L: paired(seq[lab][L]["per_seed"], seq[label("none", 0.0)][L]["per_seed"])
            for L in SEQ_HORIZONS}

    # specificity: candidate-side and commit-side net vs matched orthogonal control
    off = res["offpath_rescue"]
    res["specificity"] = {}
    for side, netlab, ctrlab in (("candidate", "cand_1.0", "orth_1.0"),
                                 ("commit", "commit_1.0", "commit_orth_1.0")):
        net = off[netlab]["net"]["per_seed"]; ctr = off[ctrlab]["net"]["per_seed"]
        diff = paired(net, ctr)
        reg_c = res["success_rescue"][netlab]["success_regression_rate"]["per_seed"]
        reg_o = res["success_rescue"][ctrlab]["success_regression_rate"]["per_seed"]
        res["specificity"][side] = {
            "net": off[netlab]["net"], "control_net": off[ctrlab]["net"],
            "net_minus_control": diff,
            "mean_net_positive_seeds": int(sum(1 for v in net if v > 0)),
            "mean_diff_positive_seeds": int(sum(1 for v in np.array(net) - np.array(ctr) if v > 0)),
            "success_reg_self": float(np.mean(reg_c)), "success_reg_control": float(np.mean(reg_o)),
        }
    # geometry (per config, mean over seeds)
    res["geometry"] = {}
    for lab in all_labels:
        entries = [meta["geometry"][m][lab][i] for i in range(len(meta["geometry"][m][lab]))]
        n_add = sum(e["n_add"] for e in entries)
        res["geometry"][lab] = {
            "cand_removal_mean": float(np.mean([e["cand_removal"] for e in entries])),
            "commit_removal_mean": float(np.mean([e["commit_removal"] for e in entries])),
            "cos_delta_p_mean": float(np.mean([e["cos_delta_p"] for e in entries])),
            "abs_proj_mean": float(np.mean([e["abs_proj"] for e in entries])),
            "n_add_total": int(n_add),
        }
    # by-op / per-reg / hamming (canonical test, per config, mean over seeds)
    res["by_op"] = {}
    for lab in all_labels:
        sm = meta["seq_metrics"][m][lab]
        res["by_op"][lab] = {
            op: float(np.mean([sm[str(s)]["by_op"][op] for s in SEEDS if sm[str(s)]["by_op"][op] is not None]))
            for op in ("SET", "SWAP", "ADD_no_wrap", "ADD_wrap")}
        res.setdefault("per_reg_acc", {})[lab] = float(np.mean([sm[str(s)]["per_reg"] for s in SEEDS]))
        hh = np.sum([sm[str(s)]["hamming_hist"] for s in SEEDS], axis=0)
        res.setdefault("hamming", {})[lab] = [int(x) for x in hh]
    res["classification"] = classify(res)
    return res


def classify(res: Dict[str, Any]) -> Dict[str, Any]:
    spec = res["specificity"]
    def specific(side_key):
        s = spec[side_key]
        net = s["net"]["per_seed"]; ctr = s["control_net"]["per_seed"]
        diff = np.array(net) - np.array(ctr)
        ok_net = bool((np.mean(net) > 0) and (int(sum(v > 0 for v in net)) >= 4))
        ok_diff = bool((diff.mean() > BARS["specific_margin"]) and (int(sum(v > 0 for v in diff)) >= 4))
        ok_collateral = bool(s["success_reg_self"] <= s["success_reg_control"] + BARS["success_reg_margin"])
        return {"net_positive": ok_net, "exceeds_control": ok_diff, "collateral_ok": ok_collateral,
                "specific": bool(ok_net and ok_diff and ok_collateral),
                "mean_net": float(np.mean(net)), "mean_diff": float(diff.mean()),
                "success_reg_self": s["success_reg_self"], "success_reg_control": s["success_reg_control"]}
    cand = specific("candidate"); commit = specific("commit")
    if cand["specific"] and not commit["specific"]:
        verdict = "O-A_upstream_candidate"
    elif commit["specific"] and not cand["specific"]:
        verdict = "O-B_downstream_commit"
    elif cand["specific"] and commit["specific"]:
        verdict = "O-C_both"
    else:
        verdict = "O-D_no_specific_parity_causality"
    return {"verdict": verdict, "candidate_side": cand, "commit_side": commit,
            "bars": BARS,
            "rule_note": ("candidate/commit-specific requires net>0 in >=4/5 seeds AND "
                          "(net - matched-orthogonal net)>0 in >=4/5 seeds AND success-regression "
                          "not worse than the matched control (+0.02). O-A/B/C/D by which side is "
                          "specific. Bars are declared conventions, not tuned cutoffs.")}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d: Dict[str, Any], res: Dict[str, Any], out_dir: str, prefix="experiment_o") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    rE = res[PRIMARY_MODEL]
    labs = ["cand_0.5", "cand_1.0", "orth_0.5", "orth_1.0", "commit_0.5", "commit_1.0",
            "commit_orth_0.5", "commit_orth_1.0"]

    # 1. rescue vs regression on off-path
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(labs)); w = 0.38
    resc = [np.mean(rE["offpath_rescue"][l]["rescued"]["per_seed"]) for l in labs]
    regr = [np.mean(rE["offpath_rescue"][l]["regressed"]["per_seed"]) for l in labs]
    ax.bar(x - w / 2, resc, w, label="rescued", color="tab:green")
    ax.bar(x + w / 2, regr, w, label="regressed", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("off-path transitions (mean/seed)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: off-path rescue vs regression by intervention")
    save(fig, "rescue_regression")

    # 2. candidate vs orthogonal (net per seed)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    net_c = rE["specificity"]["candidate"]["net"]["per_seed"]
    net_o = rE["specificity"]["candidate"]["control_net"]["per_seed"]
    xs = np.arange(5)
    ax.bar(xs - 0.2, net_c, 0.4, label="cand κ=1 (parity removal)", color="tab:blue")
    ax.bar(xs + 0.2, net_o, 0.4, label="orth κ=1 (matched control)", color="tab:orange")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(SEEDS)
    ax.set_ylabel("net off-path rescue"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: candidate-side parity removal vs matched orthogonal control")
    save(fig, "cand_vs_orth")

    # 3. commit vs orthogonal
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    netc = rE["specificity"]["commit"]["net"]["per_seed"]
    neto = rE["specificity"]["commit"]["control_net"]["per_seed"]
    ax.bar(xs - 0.2, netc, 0.4, label="commit κ=1", color="tab:purple")
    ax.bar(xs + 0.2, neto, 0.4, label="commit-orth κ=1", color="tab:gray")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(SEEDS)
    ax.set_ylabel("net off-path rescue"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: commit-side parity removal vs matched orthogonal control")
    save(fig, "commit_vs_orth")

    # 4. sequence EM curves
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for lab, c in (("none_0.0", "tab:blue"), ("cand_1.0", "tab:red"), ("orth_1.0", "tab:orange"),
                   ("commit_1.0", "tab:purple")):
        e = rE["sequence_em"][lab]
        ax.errorbar(SEQ_HORIZONS, [e[L]["mean"] for L in SEQ_HORIZONS],
                    yerr=[e[L]["sd"] for L in SEQ_HORIZONS], marker="o", capsize=3, label=lab, color=c)
    ax.set_xlabel("horizon T"); ax.set_ylabel("terminal exact match"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E: sequence-level EM under interventions")
    save(fig, "sequence_em")

    # 5. T10/T20 bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, L in ((axes[0], "10"), (axes[1], "20")):
        vals = [rE["sequence_em"][l][L]["mean"] for l in ["none_0.0", "cand_1.0", "orth_1.0", "commit_1.0"]]
        sds = [rE["sequence_em"][l][L]["sd"] for l in ["none_0.0", "cand_1.0", "orth_1.0", "commit_1.0"]]
        ax.bar(range(4), vals, yerr=sds, capsize=4, color=["tab:blue", "tab:red", "tab:orange", "tab:purple"])
        ax.set_xticks(range(4)); ax.set_xticklabels(["baseline", "cand κ=1", "orth κ=1", "commit κ=1"], fontsize=7)
        ax.set_title(f"T={L} exact match", fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("E: long-horizon endpoints")
    save(fig, "t10_t20")

    # 6. group accuracy
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    groups = ["offpath", "success", "nonoff", "all_add"]
    ks = ["baseline", "cand_0.5", "cand_1.0", "orth_1.0", "commit_1.0"]
    x = np.arange(len(groups)); w = 0.15
    for j, k in enumerate(ks):
        ax.bar(x + (j - 2) * w, [rE["group_accuracy"][g][k] for g in groups], w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("committed accuracy (pooled)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y"); ax.set_title("E: committed accuracy by group")
    save(fig, "group_accuracy")

    # 7. collateral: success regression rate
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    regs = [np.mean(rE["success_rescue"][l]["success_regression_rate"]["per_seed"]) for l in labs]
    ax.bar(range(len(labs)), regs, color="tab:red")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("baseline-success broken rate"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: collateral damage on baseline successes")
    save(fig, "collateral")

    # 8. geometry
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for lab, c in (("cand_1.0", "tab:red"), ("orth_1.0", "tab:orange"), ("commit_1.0", "tab:purple"),
                   ("commit_orth_1.0", "tab:gray")):
        g = rE["geometry"][lab]
        ax.bar(lab, g["cand_removal_mean"] + g["commit_removal_mean"], color=c)
    ax.set_ylabel("mean intervention displacement (sum over seeds)")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("E: intervention displacement (matched magnitudes)")
    save(fig, "geometry")

    # 9. by-op
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ops = ["SET", "SWAP", "ADD_no_wrap", "ADD_wrap"]
    x = np.arange(len(ops)); w = 0.25
    for j, k in enumerate(["none_0.0", "cand_1.0", "orth_1.0"]):
        ax.bar(x + (j - 1) * w, [rE["by_op"][k][op] for op in ops], w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(ops); ax.set_ylabel("final-op EM"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y"); ax.set_title("E: operation-level behavior")
    save(fig, "by_op")

    # 10. hamming
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for k, c in (("none_0.0", "tab:blue"), ("cand_1.0", "tab:red"), ("orth_1.0", "tab:orange")):
        h = np.array(rE["hamming"][k], float); h = h / h.sum()
        ax.plot(range(len(h)), h, marker="o", label=k, color=c)
    ax.set_xlabel("Hamming distance (wrong registers)"); ax.set_ylabel("fraction of sequences")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("E: terminal Hamming distribution")
    save(fig, "hamming")

    # 11. kappa monotonicity
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for side, a, b in (("cand", "cand_0.5", "cand_1.0"), ("commit", "commit_0.5", "commit_1.0")):
        ax.plot([0, 0.5, 1.0], [0,
                np.mean(rE["offpath_rescue"][a]["net"]["per_seed"]),
                np.mean(rE["offpath_rescue"][b]["net"]["per_seed"])], marker="o", label=f"{side} net rescue")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("kappa"); ax.set_ylabel("mean net off-path rescue")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("E: kappa gradient of the effect")
    save(fig, "kappa_monotone")

    # 12. E vs F
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    rF = res[CONTROL_MODEL]
    labels = ["cand_1.0", "orth_1.0", "commit_1.0", "commit_orth_1.0"]
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, [np.mean(rE["offpath_rescue"][l]["net"]["per_seed"]) for l in labels], w, label="E", color="tab:blue")
    ax.bar(x + w / 2, [np.mean(rF["offpath_rescue"][l]["net"]["per_seed"]) for l in labels], w, label="F", color="tab:orange")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("mean net off-path rescue"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E vs F: net off-path rescue")
    save(fig, "E_vs_F")

    return paths


def build(data: Dict[str, Any]) -> Dict[str, Any]:
    z = data["z"]; meta = data["meta"]
    res: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": SEEDS, "primary_model": PRIMARY_MODEL, "control_model": CONTROL_MODEL,
            "checkpoints": meta["models"], "reproduction": meta["reproduction"],
            "configs": meta["configs"], "bars": BARS,
            "statistical_discipline": ("Five frozen E checkpoints are independent units; F is a "
                                       "secondary control (not pooled). Transition-level counts are "
                                       "descriptive; inference is seed-level paired."),
        },
    }
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        res[m] = analyze_model(z, meta, m)
    # E vs F paired on net rescue
    res["E_vs_F"] = {}
    for lab in ["cand_1.0", "orth_1.0", "commit_1.0", "commit_orth_1.0"]:
        res["E_vs_F"][lab] = paired(res[PRIMARY_MODEL]["offpath_rescue"][lab]["net"]["per_seed"],
                                    res[CONTROL_MODEL]["offpath_rescue"][lab]["net"]["per_seed"])
    res["primary_decision"] = res[PRIMARY_MODEL]["classification"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment O analysis")
    ap.add_argument("--npz", default="experiments/experiment_o_representation_intervention.npz")
    ap.add_argument("--output", default="experiments/experiment_o_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_o")
    args = ap.parse_args()

    data = load(args.npz)
    res = build(data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        c = res[m]["classification"]
        print(f"{m}: {c['verdict']} | cand net={c['candidate_side']['mean_net']:+.2f} "
              f"diff={c['candidate_side']['mean_diff']:+.2f} | commit net={c['commit_side']['mean_net']:+.2f} "
              f"diff={c['commit_side']['mean_diff']:+.2f}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print("Figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
