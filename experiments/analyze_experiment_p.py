#!/usr/bin/env python3
"""Experiment P — analysis of the sequence-level error budget (measurement only).

Reads experiments/experiment_p_error_budget.npz and characterizes, per seed, the frozen model's
sequence-level error budget: the earliest observed task-level failure site (A candidate / B
candidate->commit residual / C terminal-readout residual / D mixed), its horizon trajectory,
survival/hazard, per-class profiles, untouched-register retention, recovery, and the T10->T20
transition. OBSERVATIONAL attribution only — no causal claim, no counterfactual rollout.

PRE-REGISTRATION (declared before inspecting aggregate results)
--------------------------------------------------------------
Classes are exclusive per failed sequence (A > B > C by earliest observed site; C subdivided
C1 untouched-register / C2 written-register / C3 near-boundary by normalized true-vs-winner
distance < 0.5). Dominance = largest mean per-seed fraction AND largest in >=4/5 E seeds.
If the top two categories swap ordering across seeds or are within 0.05, the budget is
"mixed / not cleanly dominated". Inference: five E checkpoints; F secondary.

Figures: budget bars (T10/T20), horizon budget, survival, hazard, first-error timestep,
Class-A operation profile, Class-B margin profile, Class-C readout, untouched-register,
recovery, E-vs-F.

Reproduce:
    python experiments/analyze_experiment_p.py \
        --npz experiments/experiment_p_error_budget.npz \
        --output experiments/experiment_p_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_p
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRIMARY_MODEL = "E_candidate"
CONTROL_MODEL = "F_combined"
SEEDS = [42, 43, 44, 45, 46]
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20]
CLASSES = ["A", "B", "C1", "C2", "C3", "D"]
NEAR_MIXED = 0.05


def class_codes(meta):
    return {k: int(v) for k, v in meta["class_codes"].items()}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def per_seed_budget(z, meta, m: str, L: int) -> Dict[str, Any]:
    cc = class_codes(meta)
    fc = z[f"{m}__T{L}__first_class"].astype(int)
    tok = z[f"{m}__T{L}__terminal_ok"].astype(int)
    seed = z[f"{m}__T{L}__seed"].astype(int)
    out: Dict[str, Any] = {}
    for s in SEEDS:
        mask = seed == s
        if mask.sum() == 0:
            continue
        f = fc[mask]; t = tok[mask]
        fail = t == 0; nfail = int(fail.sum())
        counts = {c: int(((f == cc[c]) & fail).sum()) for c in CLASSES}
        fracs = {c: (counts[c] / nfail if nfail else 0.0) for c in CLASSES}
        out[str(s)] = {"n": int(mask.sum()), "n_fail": nfail, "em": float(t.mean()),
                       "counts": counts, "fracs": fracs}
    return out


def summarize_budget(per: Dict[str, Any]) -> Dict[str, Any]:
    fracs = {c: np.array([per[s]["fracs"][c] for s in per], float) for c in CLASSES}
    ems = np.array([per[s]["em"] for s in per], float)
    fails = np.array([per[s]["n_fail"] for s in per], float)
    out = {"em_mean": float(ems.mean()), "em_sd": float(ems.std(ddof=1)) if len(ems) > 1 else 0.0,
           "em_per_seed": ems.tolist(), "n_fail_mean": float(fails.mean()),
           "per_class": {c: {"mean": float(fracs[c].mean()),
                             "sd": float(fracs[c].std(ddof=1)) if len(fracs[c]) > 1 else 0.0,
                             "per_seed": fracs[c].tolist()} for c in CLASSES}}
    return out


def dominator(summary: Dict[str, Any]) -> Dict[str, Any]:
    means = {c: summary["per_class"][c]["mean"] for c in CLASSES}
    order = sorted(CLASSES, key=lambda c: -means[c])
    top = order[0]
    # per-seed largest
    per = {c: np.array(summary["per_class"][c]["per_seed"]) for c in CLASSES}
    n = len(per[top])
    top_seeds = int(np.sum([int(np.argmax([per[c][i] for c in CLASSES])) == CLASSES.index(top) for i in range(n)]))
    second = order[1]
    gap = means[top] - means[second]
    clean = (top_seeds >= 4) and (gap >= NEAR_MIXED)
    return {"top": top, "second": second, "top_mean": means[top], "second_mean": means[second],
            "gap": gap, "top_largest_in_seeds": top_seeds,
            "dominant": bool(clean), "verdict": top if clean else "mixed"}


def survival(z, m: str, L: int, key: str) -> np.ndarray:
    ok = z[f"{m}__T{L}__{key}"].astype(bool)     # (N,T)
    N, T = ok.shape
    surv = np.array([float(ok[:, :t + 1].all(1).mean()) for t in range(T)])
    return surv


def hazard(surv: np.ndarray) -> np.ndarray:
    h = np.zeros_like(surv)
    prev = 1.0
    for t in range(len(surv)):
        h[t] = 1.0 - surv[t] / prev if prev > 0 else 0.0
        prev = surv[t]
    return h


def profile(z, m: str, L: int, cls: str, meta, fields: List[str]) -> Dict[str, Any]:
    cc = class_codes(meta)
    fc = z[f"{m}__T{L}__first_class"].astype(int)
    sel = fc == cc[cls]
    out: Dict[str, Any] = {"n": int(sel.sum())}
    if sel.sum() == 0:
        return out
    for f in fields:
        v = z[f"{m}__T{L}__{f}"].astype(np.float64)
        if v.ndim == 2:
            # reduce over the first-error timestep where applicable
            ft = z[f"{m}__T{L}__first_t"].astype(int)
            idx = np.clip(ft, 0, v.shape[1] - 1)
            vals = v[np.arange(len(v)), idx][sel]
        else:
            vals = v[sel]
        vals = vals[np.isfinite(vals)]
        out[f] = {"mean": float(vals.mean()) if vals.size else None,
                  "median": float(np.median(vals)) if vals.size else None}
    return out


def first_error_ops(z, m: str, L: int, cls: str, meta) -> Dict[str, int]:
    cc = class_codes(meta)
    fc = z[f"{m}__T{L}__first_class"].astype(int)
    ft = z[f"{m}__T{L}__first_t"].astype(int)
    op = z[f"{m}__T{L}__op"].astype(int)          # (N,T)
    V = 0
    out = {"SET": 0, "SWAP": 0, "ADD_no_wrap": 0, "ADD_wrap": 0}
    wrap = z[f"{m}__T{L}__wrap"].astype(bool)
    for i in np.where(fc == cc[cls])[0]:
        t = ft[i]
        if t < 0:
            continue
        o = op[i, t]
        if o == 0:
            out["SET"] += 1
        elif o == 1:
            out["SWAP"] += 1
        elif o == 2:
            out["ADD_wrap" if wrap[i, t] else "ADD_no_wrap"] += 1
    return out


def paired(a: List[float], b: List[float]) -> Dict[str, Any]:
    x = np.array(a, float); y = np.array(b, float); d = x - y
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    out = {"E_mean": float(x.mean()), "F_mean": float(y.mean()), "mean_diff": float(d.mean()),
           "sign_positive": int((d > 0).sum()), "sign_negative": int((d < 0).sum()),
           "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"), "n": int(d.size)}
    try:
        out["paired_t_p"] = float(st.ttest_rel(x, y).pvalue)
    except Exception:
        out["paired_t_p"] = None
    return out


def analyze_model(z, meta, m: str) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    b10 = per_seed_budget(z, meta, m, 10); b20 = per_seed_budget(z, meta, m, 20)
    res["budget_T10"] = {"per_seed": b10, "summary": summarize_budget(b10)}
    res["budget_T20"] = {"per_seed": b20, "summary": summarize_budget(b20)}
    res["budget_T10"]["dominator"] = dominator(res["budget_T10"]["summary"])
    res["budget_T20"]["dominator"] = dominator(res["budget_T20"]["summary"])
    # horizon trajectory
    hz = {}
    for L in HORIZONS:
        fc = z[f"{m}__h{L}__first_class"].astype(int)
        tok = z[f"{m}__h{L}__terminal_ok"].astype(int)
        cc = class_codes(meta)
        fail = tok == 0; nf = int(fail.sum())
        hz[str(L)] = {"em": float(tok.mean()), "first_error_rate": float(fail.mean()),
                      "class_fracs": {c: (float(((fc == cc[c]) & fail).sum()) / nf if nf else 0.0) for c in CLASSES},
                      "n": int(len(fc))}
    res["horizon_budget"] = hz
    # survival + hazard (T10)
    for key, name in (("cand_written_ok", "candidate"), ("comm_written_ok", "commit"),
                      ("full_comm_ok", "full_comm")):
        s = survival(z, m, 10, key); hh = hazard(s)
        res.setdefault("survival", {})[name] = {"surv": s.tolist(), "hazard": hh.tolist()}
    # profiles
    res["classA_profile"] = {f: profile(z, m, 10, "A", meta, [f]) for f in ["cand_margin", "offpath_dist", "gate_mean"]}
    res["classA_ops"] = first_error_ops(z, m, 10, "A", meta)
    res["classB_profile"] = {f: profile(z, m, 10, "B", meta, [f]) for f in
                             ["cand_margin", "comm_margin", "rho", "alpha_full", "offpath_dist", "gate_mean", "competitor_changed"]}
    res["classB_ops"] = first_error_ops(z, m, 10, "B", meta)
    res["classC_n"] = {"C1": int((z[f"{m}__T10__first_class"] == class_codes(meta)["C1"]).sum()),
                       "C2": int((z[f"{m}__T10__first_class"] == class_codes(meta)["C2"]).sum()),
                       "C3": int((z[f"{m}__T10__first_class"] == class_codes(meta)["C3"]).sum())}
    # untouched-register contribution
    we = z[f"{m}__T10__written_ever"].astype(bool); wr = z[f"{m}__T10__wrong_regs"].astype(bool)
    fc = z[f"{m}__T10__first_class"].astype(int); fail = fc != 0
    untouched_wrong = (wr & (~we)).any(1)
    res["untouched_register"] = {
        "any_untouched_terminal_error_rate_all": float(untouched_wrong.mean()),
        "any_untouched_terminal_error_rate_failures": float(untouched_wrong[fail].mean()) if fail.any() else 0.0,
        "n_failures_with_untouched": int(untouched_wrong[fail].sum()),
    }
    # recovery
    rec = {"cand": z[f"{m}__T10__rec_cand"], "comm": z[f"{m}__T10__rec_comm"], "full": z[f"{m}__T10__rec_full"]}
    res["recovery"] = {}
    for cls in ["A", "B", "C3"]:
        sel = fc == class_codes(meta)[cls]
        res["recovery"][cls] = {k: (float(rec[k][sel].mean()) if sel.any() else None) for k in rec}
    # propagation: subsequent wrong timesteps / terminal hamming
    res["propagation"] = {}
    for cls in ["A", "B"]:
        sel = fc == class_codes(meta)[cls]
        n_sub = z[f"{m}__T10__n_full_err"].astype(float)
        res["propagation"][cls] = {"mean_subsequent_full_err_steps": float(n_sub[sel].mean()) if sel.any() else None}
    # T10->T20 cross-tab (same sequences: T20 dataset prefix at step 10)
    f20 = z[f"{m}__T20__full_comm_ok"].astype(bool)      # (N,20)
    t10 = f20[:, 9]; t20 = f20[:, -1]
    res["T10_T20_crosstab"] = {
        "n": int(len(t10)),
        "T10succ_T20succ": int((t10 & t20).sum()),
        "T10succ_T20fail": int((t10 & ~t20).sum()),
        "T10fail_T20fail": int((~t10 & ~t20).sum()),
        "T10fail_T20succ": int((~t10 & t20).sum()),
        "P_T20fail_given_T10fail": float((~t20)[~t10].mean()) if (~t10).any() else None,
        "P_T20fail_given_T10succ": float((~t20)[t10].mean()) if t10.any() else None,
    }
    # first-error timestep distribution
    ft = z[f"{m}__T10__first_t"].astype(int)
    res["first_error_t_hist"] = np.bincount(ft[(fc == class_codes(meta)["A"]) | (fc == class_codes(meta)["B"])],
                                            minlength=11).tolist()
    return res


def make_figures(d, res, out_dir, prefix="experiment_p") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    rE = res[PRIMARY_MODEL]; rF = res[CONTROL_MODEL]
    # 1/2 budget T10/T20
    for L, tag in ((10, "budget_t10"), (20, "budget_t20")):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        x = np.arange(len(CLASSES)); w = 0.38
        ax.bar(x - w / 2, [rE[f"budget_T{L}"]["summary"]["per_class"][c]["mean"] for c in CLASSES], w,
               yerr=[rE[f"budget_T{L}"]["summary"]["per_class"][c]["sd"] for c in CLASSES], capsize=3,
               label="E", color="tab:blue")
        ax.bar(x + w / 2, [rF[f"budget_T{L}"]["summary"]["per_class"][c]["mean"] for c in CLASSES], w,
               yerr=[rF[f"budget_T{L}"]["summary"]["per_class"][c]["sd"] for c in CLASSES], capsize=3,
               label="F", color="tab:orange")
        ax.set_xticks(x); ax.set_xticklabels(["A cand", "B commit", "C1", "C2", "C3", "D"])
        ax.set_ylabel("fraction of failed sequences"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"First-error budget at T={L} (E primary, F control)")
        save(fig, tag)
    # 3 horizon budget
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for c, col in (("A", "tab:red"), ("B", "tab:blue"), ("C3", "tab:green")):
        ax.plot([str(L) for L in HORIZONS], [rE["horizon_budget"][str(L)]["class_fracs"][c] for L in HORIZONS],
                marker="o", label=f"E {c}", color=col)
        ax.plot([str(L) for L in HORIZONS], [rF["horizon_budget"][str(L)]["class_fracs"][c] for L in HORIZONS],
                marker="x", ls="--", label=f"F {c}", color=col, alpha=0.6)
    ax.set_xlabel("horizon T"); ax.set_ylabel("fraction of failures"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Error-budget composition vs horizon")
    save(fig, "budget_horizons")
    # 4 survival
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for key, lab in (("candidate", "candidate written"), ("commit", "committed written"), ("full_comm", "full-state")):
        ax.plot(range(1, 11), rE["survival"][key]["surv"], marker="o", label=lab)
    ax.set_xlabel("timestep t"); ax.set_ylabel("P(error-free through t)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E: survival of correctness through the trajectory (T=10)")
    save(fig, "survival")
    # 5 hazard
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for key, lab in (("candidate", "first candidate error"), ("commit", "first commit residual"), ("full_comm", "first full-state error")):
        ax.plot(range(1, 11), rE["survival"][key]["hazard"], marker="o", label=lab)
    ax.set_xlabel("timestep t"); ax.set_ylabel("empirical first-error incidence"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E: first-error incidence (hazard) by timestep")
    save(fig, "hazard")
    # 6 first-error timestep
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(range(1, len(rE["first_error_t_hist"])), rE["first_error_t_hist"][1:], color="tab:red")
    ax.set_xlabel("first-error timestep"); ax.set_ylabel("count"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: first-error timestep distribution (A/B, T=10)")
    save(fig, "first_error_t")
    # 7 Class A ops
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ops = ["SET", "SWAP", "ADD_no_wrap", "ADD_wrap"]
    ax.bar(range(4), [rE["classA_ops"][o] for o in ops], color="tab:red", label="Class A (E)")
    ax.set_xticks(range(4)); ax.set_xticklabels(ops); ax.set_ylabel("count")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("E: candidate-origin (Class A) first errors by operation")
    save(fig, "classA_ops")
    # 8 Class B margin profile
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labs = ["cand_margin", "comm_margin", "rho", "offpath_dist"]
    vals = [rE["classB_profile"][f][f]["mean"] for f in labs]
    ax.bar(range(4), [v if v is not None else 0.0 for v in vals], color="tab:blue")
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: Class B (commit-residual) profile at the first error")
    save(fig, "classB_profile")
    # 9 Class C readout (C3 near-boundary)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    c3n = rE["classC_n"]
    ax.bar(["C1", "C2", "C3"], [c3n["C1"], c3n["C2"], c3n["C3"]], color=["tab:blue", "tab:green", "tab:orange"])
    ax.set_ylabel("count (E, T=10)"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: Class C terminal/readout residual subtypes")
    save(fig, "classC_readout")
    # 10 untouched register
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(["all seqs", "failures"], [rE["untouched_register"]["any_untouched_terminal_error_rate_all"],
                                      rE["untouched_register"]["any_untouched_terminal_error_rate_failures"]],
           color="tab:purple")
    ax.set_ylabel("fraction with untouched-register terminal error"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: untouched-register terminal corruption")
    save(fig, "untouched")
    # 11 recovery
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(3); w = 0.25
    for j, k in enumerate(["cand", "comm", "full"]):
        ax.bar(x + (j - 1) * w, [rE["recovery"][c][k] if rE["recovery"][c][k] is not None else 0 for c in ["A", "B", "C3"]],
               w, label=f"recovery: {k}")
    ax.set_xticks(x); ax.set_xticklabels(["after A", "after B", "after C3"]); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y"); ax.set_title("E: recovery after first observed error")
    save(fig, "recovery")
    # 12 E vs F budget comparison (T10)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([0, 1, 2], [rE["budget_T10"]["summary"]["per_class"][c]["mean"] for c in ["A", "B", "C3"]], 0.38, label="E", color="tab:blue")
    ax.bar([0.4, 1.4, 2.4], [rF["budget_T10"]["summary"]["per_class"][c]["mean"] for c in ["A", "B", "C3"]], 0.38, label="F", color="tab:orange")
    ax.set_xticks([0.2, 1.2, 2.2]); ax.set_xticklabels(["A", "B", "C3"]); ax.set_ylabel("fraction of failures (T=10)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("E vs F: T=10 first-error budget")
    save(fig, "E_vs_F")
    return paths


def build(data: Dict[str, Any]) -> Dict[str, Any]:
    z = data["z"]; meta = data["meta"]
    res: Dict[str, Any] = {"metadata": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS,
                                        "checkpoints": meta["models"], "class_codes": meta["class_codes"],
                                        "c3_bar": meta["c3_bar"],
                                        "statistical_discipline": ("Five frozen E checkpoints are independent "
                                                                   "units; F is a secondary control (not pooled). "
                                                                   "Transition/sequence counts are descriptive; "
                                                                   "inference is seed-level.")}}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        res[m] = analyze_model(z, meta, m)
    res["E_vs_F_paired"] = {c: paired(res[PRIMARY_MODEL]["budget_T10"]["summary"]["per_class"][c]["per_seed"],
                                      res[CONTROL_MODEL]["budget_T10"]["summary"]["per_class"][c]["per_seed"])
                            for c in ["A", "B", "C3"]}
    res["primary_decision"] = {
        "T10_dominator_E": res[PRIMARY_MODEL]["budget_T10"]["dominator"],
        "T10_dominator_F": res[CONTROL_MODEL]["budget_T10"]["dominator"],
        "T20_dominator_E": res[PRIMARY_MODEL]["budget_T20"]["dominator"],
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment P analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_p_error_budget.npz")
    ap.add_argument("--output", default="experiments/experiment_p_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_p")
    args = ap.parse_args()

    data = load(args.npz)
    res = build(data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        for L in (10, 20):
            s = res[m][f"budget_T{L}"]["summary"]
            print(f"{m} T={L}: EM={s['em_mean']:.4f} fail={s['n_fail_mean']:.0f} " +
                  " ".join(f"{c}={s['per_class'][c]['mean']:.3f}" for c in CLASSES))
        print(f"   T10 dominator: {res[m]['budget_T10']['dominator']['verdict']}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
