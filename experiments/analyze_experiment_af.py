#!/usr/bin/env python3
"""Experiment AF — analysis: held-out operand task-difficulty stratification (measurement only).

AE showed the AdamW readers do not transfer to the 20 held-out operand pairs (b in {0,5}). AF asks a
task-side question, using U's existing representation-independent exact-interface machinery on the
unchanged V dataset:

    Do the weak held-out-operand results coincide with identifiable differences in task-side label
    geometry or raw-interface target recoverability?

AF trains no reader, probes no RE:Reason latent state, changes no task and modifies nothing. It uses
U's estimator exactly (`analyze_experiment_u.history_features` / `exact_bayes`); no alternative Bayes
estimator is introduced and no replacement model is fit. Raw-interface **identifiability** is kept
separate from **learnability by a bounded reader** and from AE's train-derived baselines.

Predefined strata (from the operand values and the arithmetic rule, before inspecting outcomes):
    b = 0 vs b = 5; wrap (a+b>=10) vs no-wrap (a+b<10); a<5 vs a>=5; a==b vs a!=b;
    target class y_true. Held-out pairs only (b in {0,5}); the b not in {0,5} rows are a seen-operand
    reference group. Confounding note: within the held-out set wrap is structurally tied to b=5 and
    a>=5 (b=0 never wraps) — these are descriptive strata, not independent causal factors.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_af.py
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_u as au  # noqa: E402
import analyze_experiment_v as av  # noqa: E402

PRIMARY_MODEL = "E_candidate"
CONTROL_MODEL = "F_combined"
MODELS = [PRIMARY_MODEL, CONTROL_MODEL]
SEEDS = [42, 43, 44, 45, 46]
REPS = au.REPS
REP_LABELS = [r for r, _ in REPS]
DECISIVE = ["L4", "Lfull"]
POPS = ["all_add", "classA"]
NCLASS = 10
HELD_OUT_B = (0, 5)

# predefined held-out strata (predicates over the operand values)
HELD_OUT_GROUPS: List[Tuple[str, Any]] = [
    ("b0", lambda a, b: b == 0),
    ("b5", lambda a, b: b == 5),
    ("wrap", lambda a, b: (a + b) >= 10),
    ("nowrap", lambda a, b: (a + b) < 10),
    ("a_lt5", lambda a, b: a < 5),
    ("a_ge5", lambda a, b: a >= 5),
    ("eq", lambda a, b: a == b),
    ("neq", lambda a, b: a != b),
]
HELD_OUT_GROUP_NAMES = [g for g, _ in HELD_OUT_GROUPS]
ALL_GROUPS = ["overall", "heldout", "seen"] + HELD_OUT_GROUP_NAMES
PAIRS_HELDOUT = [(a, b) for a in range(10) for b in HELD_OUT_B]

MATERIAL_GAP = 0.15        # U/V material-gain convention, used descriptively (not an acceptance bar)
CONSISTENT_SEEDS = 4       # >=4/5 seeds
MIN_SUPPORT = 20           # per-seed subgroup rows considered adequate support


def _ms(vals):
    return as_.mean_sd(vals)


def _entropy(counts: List[int]) -> float:
    c = np.asarray(counts, float)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c[c > 0] / tot
    return float(-(p * np.log2(p)).sum())


def _flatten_block(b: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Flatten a per-(pop,rep,group) exact-interface block into primitive fields (for mean/sd)."""
    if b is None:
        return {}
    out: Dict[str, Optional[float]] = {"n": b["n"], "unique_fraction": b["unique_fraction"],
                                       "mean_multiplicity": b["mean_multiplicity"]}
    for lab in ("y_true", "operand_a", "operand_b"):
        eb = b[lab]
        for f in ("bayes_acc", "n_unique", "collision_rate", "ambiguous_mass",
                  "max_multiplicity", "mean_target_entropy"):
            out[f"{lab}__{f}"] = eb[f]
    return out


def _bayes_block(hist_rep: np.ndarray, labels: Dict[str, np.ndarray],
                 mask: np.ndarray) -> Optional[Dict[str, Any]]:
    n = int(mask.sum())
    if n == 0:
        return None
    keys = au._keys(hist_rep[mask])
    out: Dict[str, Any] = {"n": n}
    for lab, nc in (("y_true", NCLASS), ("operand_a", NCLASS), ("operand_b", NCLASS)):
        out[lab] = au.exact_bayes(keys, labels[lab][mask].astype(int), nc)
    out["unique_fraction"] = out["y_true"]["n_unique"] / n
    out["mean_multiplicity"] = n / max(out["y_true"]["n_unique"], 1)
    return out


def analyze_pooled(z, m: str) -> Dict[str, Any]:
    """Pooled estimate: concatenate all seeds' test rows, then run U's exact-interface estimator."""
    oa, ob, y, ok, seq, t = [], [], [], [], [], []
    for s in SEEDS:
        sel = z[f"{m}__test__seed"] == s
        oa.append(z[f"{m}__test__operand_a"][sel].astype(int))
        ob.append(z[f"{m}__test__operand_b"][sel].astype(int))
        y.append(z[f"{m}__test__y_true"][sel].astype(int))
        ok.append(z[f"{m}__test__cand_ok"][sel].astype(bool))
        seq.append(z[f"{m}__test__seq_obs"][sel])
        t.append(z[f"{m}__test__t"][sel])
    oa = np.concatenate(oa); ob = np.concatenate(ob); y = np.concatenate(y)
    ok = np.concatenate(ok); seq = np.concatenate(seq); t = np.concatenate(t)
    labels = {"y_true": y, "operand_a": oa, "operand_b": ob}
    held = np.isin(ob, list(HELD_OUT_B))
    base_by_pop = {"all_add": np.ones(len(y), bool), "classA": ~ok}
    hist = {rep: au.history_features(seq, t, L) for rep, L in REPS}
    out: Dict[str, Any] = {}
    for pop, base in base_by_pop.items():
        out[pop] = {}
        for rep, _ in REPS:
            h = hist[rep]
            node = {"overall": _bayes_block(h, labels, base),
                    "heldout": _bayes_block(h, labels, base & held),
                    "seen": _bayes_block(h, labels, base & ~held)}
            for gname, pred in HELD_OUT_GROUPS:
                gm = np.array([pred(int(a), int(b)) for a, b in zip(oa, ob)], bool) & held
                node[gname] = _bayes_block(h, labels, base & gm)
            out[pop][rep] = {g: _flatten_block(node[g]) for g in ALL_GROUPS}
    return out


def analyze_seed(z, m: str, seed: int) -> Dict[str, Any]:
    sel = z[f"{m}__test__seed"] == seed
    oa = z[f"{m}__test__operand_a"][sel].astype(int)
    ob = z[f"{m}__test__operand_b"][sel].astype(int)
    y = z[f"{m}__test__y_true"][sel].astype(int)
    ok = z[f"{m}__test__cand_ok"][sel].astype(bool)
    seq = z[f"{m}__test__seq_obs"][sel]
    t = z[f"{m}__test__t"][sel]
    labels = {"y_true": y, "operand_a": oa, "operand_b": ob}
    held = np.isin(ob, list(HELD_OUT_B))
    n_seed_test = int(sel.sum())
    base_by_pop = {"all_add": np.ones(n_seed_test, bool), "classA": ~ok}
    hist = {rep: au.history_features(seq, t, L) for rep, L in REPS}

    ident: Dict[str, Any] = {}
    geom: Dict[str, Any] = {}
    per_pair: Dict[str, Any] = {}
    for pop, base in base_by_pop.items():
        blocks: Dict[str, Dict[str, Any]] = {}
        for rep, _ in REPS:
            h = hist[rep]
            blocks.setdefault(rep, {})["overall"] = _bayes_block(h, labels, base)
            blocks[rep]["heldout"] = _bayes_block(h, labels, base & held)
            blocks[rep]["seen"] = _bayes_block(h, labels, base & ~held)
            for gname, pred in HELD_OUT_GROUPS:
                gm = np.array([pred(int(a), int(b)) for a, b in zip(oa, ob)], bool) & held
                blocks[rep][gname] = _bayes_block(h, labels, base & gm)
            # per-pair y_true identifiability (support reported explicitly)
            for a, b in PAIRS_HELDOUT:
                pm = base & (oa == a) & (ob == b)
                blk = _bayes_block(h, labels, pm)
                per_pair.setdefault(rep, {})[f"{a}_{b}"] = blk
        ident[pop] = {rep: {g: _flatten_block(blocks[rep][g]) for g in ALL_GROUPS} for rep, _ in REPS}
        # geometry: counts + entropy for heldout and seen (all reps share the same labels)
        gm_all = held
        counts_ho = [int((y[base & gm_all] == c).sum()) for c in range(NCLASS)]
        counts_seen = [int((y[base & ~held] == c).sum()) for c in range(NCLASS)]
        gnode: Dict[str, Any] = {"overall": {"n": int(base.sum()),
                                             "counts": [int((y[base] == c).sum()) for c in range(NCLASS)],
                                             "entropy": _entropy([int((y[base] == c).sum())
                                                                  for c in range(NCLASS)])},
                                 "heldout": {"n": int((base & held).sum()), "counts": counts_ho,
                                             "entropy": _entropy(counts_ho)},
                                 "seen": {"n": int((base & ~held).sum()), "counts": counts_seen,
                                          "entropy": _entropy(counts_seen)}}
        for gname, pred in HELD_OUT_GROUPS:
            gm = np.array([pred(int(a), int(b)) for a, b in zip(oa, ob)], bool) & held & base
            cnt = [int((y[gm] == c).sum()) for c in range(NCLASS)]
            gnode[gname] = {"n": int(gm.sum()), "counts": cnt, "entropy": _entropy(cnt)}
        geom[pop] = gnode
    return {"seed": seed, "ident": ident, "geom": geom, "per_pair": per_pair}


def aggregate(per: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"identifiability": {}, "geometry": {}, "per_pair": {}}
    for pop in POPS:
        for rep in REP_LABELS:
            for g in ALL_GROUPS:
                node: Dict[str, Any] = {}
                seeds_blocks = [s["ident"][pop][rep][g] for s in per]
                keys = sorted({k for b in seeds_blocks for k in b})
                for k in keys:
                    node[k] = _ms([b.get(k) for b in seeds_blocks])
                out["identifiability"].setdefault(pop, {}).setdefault(rep, {})[g] = node
        for g in ALL_GROUPS:
            seeds_g = [s["geom"][pop][g] for s in per]
            out["geometry"].setdefault(pop, {})[g] = {
                "n": _ms([sg["n"] for sg in seeds_g]),
                "counts": [int(sum(sg["counts"][i] for sg in seeds_g)) for i in range(NCLASS)],
                "entropy": _ms([sg["entropy"] for sg in seeds_g])}
        for rep in REP_LABELS:
            for a, b in PAIRS_HELDOUT:
                seeds_blocks = [s["per_pair"][rep][f"{a}_{b}"] for s in per]
                node = {}
                keys = sorted({k for blk in seeds_blocks if blk for k in _flatten_block(blk)})
                for k in keys:
                    node[k] = _ms([(_flatten_block(blk).get(k) if blk else None) for blk in seeds_blocks])
                out["per_pair"].setdefault(pop, {}).setdefault(rep, {})[f"{a}_{b}"] = node
    return out


# ---------------------------------------------------------------------------
# decision (descriptive)
# ---------------------------------------------------------------------------

def classify(res: Dict[str, Any]) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    supported_keys: List[str] = []
    adequate_gaps: List[float] = []
    for pop in POPS:
        for rep in DECISIVE:
            ov = res["identifiability"][pop][rep]["overall"]["y_true__bayes_acc"]
            for g in HELD_OUT_GROUP_NAMES:
                node = res["identifiability"][pop][rep][g]
                b = node["y_true__bayes_acc"]
                n = node["n"]["mean"]
                uf = node["unique_fraction"]["mean"]
                gaps = [o - gv for o, gv in zip(ov["per_seed"], b["per_seed"])
                        if o is not None and gv is not None]
                mean_gap = float(np.mean(gaps)) if gaps else None
                n_lower = sum(1 for d in gaps if d >= MATERIAL_GAP)
                adequate = bool(n is not None and n >= MIN_SUPPORT)
                key = f"{pop}|{rep}|{g}"
                cells[key] = {"overall_bayes": ov["mean"], "group_bayes": b["mean"],
                              "mean_gap": mean_gap, "n": n, "unique_fraction": uf,
                              "seeds_lower_ge_bar": n_lower, "adequate_support": adequate}
                if adequate and mean_gap is not None:
                    adequate_gaps.append(mean_gap)
                    if mean_gap >= MATERIAL_GAP and n_lower >= CONSISTENT_SEEDS:
                        supported_keys.append(key)
    if supported_keys:
        verdict = "task_side_subgroup_limitation_supported"
    elif adequate_gaps and max(adequate_gaps) < MATERIAL_GAP:
        verdict = "task_side_limitation_not_indicated"
    else:
        verdict = "inconclusive_mixed"
    return {"verdict": verdict, "supported_subgroups": supported_keys, "cells": cells,
            "thresholds": {"material_gap": MATERIAL_GAP, "consistent_seeds": CONSISTENT_SEEDS,
                           "min_support": MIN_SUPPORT},
            "rule_note": ("Descriptive, not an acceptance threshold: a held-out subgroup is treated as "
                          "materially lower if its exact-interface y_true Bayes ceiling is >=0.15 below "
                          "the same-population overall ceiling in >=4/5 seeds with per-seed support "
                          ">=20 (task_side_subgroup_limitation_supported); if every adequately-supported "
                          "held-out subgroup stays within 0.15 of the overall ceiling the observed "
                          "reader failure is not explained by an evident loss of raw-interface target "
                          "identifiability (task_side_limitation_not_indicated); otherwise inconclusive. "
                          "This cannot identify which reader-side limitation explains AF/AE.")}


def summarize(res: Dict[str, Any]) -> Dict[str, Any]:
    """Compact per-population/rep/group y_true ceiling table for quick reading."""
    out: Dict[str, Any] = {}
    for pop in POPS:
        out[pop] = {}
        for rep in DECISIVE:
            out[pop][rep] = {g: {"bayes": res["identifiability"][pop][rep][g]["y_true__bayes_acc"]["mean"],
                                 "n": res["identifiability"][pop][rep][g]["n"]["mean"],
                                 "n_unique": res["identifiability"][pop][rep][g]["y_true__n_unique"]["mean"],
                                 "unique_fraction": res["identifiability"][pop][rep][g]["unique_fraction"]["mean"],
                                 "collision": res["identifiability"][pop][rep][g]["y_true__collision_rate"]["mean"],
                                 "operand_a_bayes": res["identifiability"][pop][rep][g]["operand_a__bayes_acc"]["mean"]}
                               for g in ALL_GROUPS}
    return out


def build(data: Dict[str, Any]) -> Dict[str, Any]:
    z = data["z"]
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "models": MODELS,
        "primary_model": PRIMARY_MODEL, "control_model": CONTROL_MODEL,
        "target": "y_true", "n_classes": NCLASS, "reps": REP_LABELS, "decisive": DECISIVE,
        "pops": POPS, "held_out_b": list(HELD_OUT_B), "held_out_pairs": len(PAIRS_HELDOUT),
        "groups": ALL_GROUPS, "held_out_groups": HELD_OUT_GROUP_NAMES,
        "estimator": "U exact-interface (history_features + exact_bayes)",
        "source_identity": data["meta"].get("source"), "checkpoints": data["meta"].get("models"),
        "confounding_note": ("Within the held-out set wrap is structurally tied to b=5 and a>=5 "
                             "(b=0 never wraps); the strata are descriptive, not independent causal "
                             "factors."),
        "statistical_discipline": (
            "Five frozen E checkpoints are the independent units; F is a control. Strata are defined "
            "from operand values and the arithmetic rule before inspecting outcomes. The exact-interface "
            "ceiling is a property of the raw observation history (U's estimator, unchanged) and is kept "
            "separate from any train-derived baseline and from AE's reader accuracies. No reader is fit; "
            "no rule/length/subgroup is selected from outcomes. Sparse/unique-history estimates are "
            "flagged and never interpreted as practical learnability.")}}
    for m in MODELS:
        per = [analyze_seed(z, m, s) for s in SEEDS]
        agg = aggregate(per)
        res[m] = {"identifiability": agg["identifiability"], "geometry": agg["geometry"],
                  "per_pair": agg["per_pair"], "per_seed": per,
                  "pooled": analyze_pooled(z, m)}
    res["classification"] = classify(res[PRIMARY_MODEL])
    res["control_classification"] = classify(res[CONTROL_MODEL])
    res["summary"] = summarize(res[PRIMARY_MODEL])
    res["pooled_summary"] = {
        pop: {rep: {g: {"bayes": res[PRIMARY_MODEL]["pooled"][pop][rep][g]["y_true__bayes_acc"],
                        "n": res[PRIMARY_MODEL]["pooled"][pop][rep][g]["n"],
                        "n_unique": res[PRIMARY_MODEL]["pooled"][pop][rep][g]["y_true__n_unique"],
                        "collision": res[PRIMARY_MODEL]["pooled"][pop][rep][g]["y_true__collision_rate"],
                        "operand_a_bayes": res[PRIMARY_MODEL]["pooled"][pop][rep][g]["operand_a__bayes_acc"]}
                    for g in ALL_GROUPS} for rep in DECISIVE} for pop in POPS}
    return res


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(res, out_dir, prefix="experiment_af") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    groups = ["heldout", "seen", "b0", "b5", "wrap", "nowrap", "a_lt5", "a_ge5", "eq", "neq"]

    def bval(pop, rep, g):
        return res[PRIMARY_MODEL]["identifiability"][pop][rep][g]["y_true__bayes_acc"]["mean"]

    # 1 y_true Bayes ceiling by group (L4/Lfull, all-ADD & Class-A)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    xi = np.arange(len(groups))
    w = 0.2
    for i, (pop, rep, c) in enumerate([("all_add", "L4", "tab:blue"), ("all_add", "Lfull", "tab:cyan"),
                                       ("classA", "L4", "tab:orange"), ("classA", "Lfull", "tab:red")]):
        ax.bar(xi + (i - 1.5) * w, [bval(pop, rep, g) for g in groups], w, label=f"{pop} {rep}", color=c)
    ax.axhline(1.0, ls=":", c="k", lw=0.8)
    ax.set_xticks(xi); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("exact-interface y_true Bayes ceiling"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AF: raw-interface target identifiability by held-out stratum (E)")
    save(fig, "bayes_by_group")

    # 2 operand_a Bayes by group
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, (pop, rep, c) in enumerate([("all_add", "L4", "tab:blue"), ("classA", "L4", "tab:orange")]):
        ax.bar(xi + (i - 0.5) * 0.4, [
            res[PRIMARY_MODEL]["identifiability"][pop][rep][g]["operand_a__bayes_acc"]["mean"] for g in groups],
            0.4, label=f"{pop} {rep}", color=c)
    ax.set_xticks(xi); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("exact-interface operand_a Bayes ceiling"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AF: operand_a identifiability by stratum (E)")
    save(fig, "operand_a_by_group")

    # 3 support and unique fraction
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ns = [res[PRIMARY_MODEL]["identifiability"]["classA"]["L4"][g]["n"]["mean"] for g in groups]
    uf = [res[PRIMARY_MODEL]["identifiability"]["classA"]["L4"][g]["unique_fraction"]["mean"] for g in groups]
    ax.bar(xi - 0.2, ns, 0.4, label="n (Class-A rows/seed)", color="tab:gray")
    ax2 = ax.twinx()
    ax2.plot(xi, uf, "o-", color="tab:purple", label="unique-history fraction")
    ax2.set_ylim(0, 1.05)
    ax.set_xticks(xi); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("n (Class-A, per seed)"); ax2.set_ylabel("unique-history fraction")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AF: subgroup support and history uniqueness (E, Class-A, L4)")
    save(fig, "support_by_group")

    # 4 per-pair y_true Bayes vs support (Class-A, L4)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for rep, c, mk in (("L4", "tab:orange", "o"), ("Lfull", "tab:red", "s")):
        pp = res[PRIMARY_MODEL]["per_pair"]["classA"][rep]
        xs = [pp[f"{a}_{b}"]["n"]["mean"] for a, b in PAIRS_HELDOUT]
        ys = [pp[f"{a}_{b}"]["y_true__bayes_acc"]["mean"] for a, b in PAIRS_HELDOUT]
        ok = [x is not None and y is not None for x, y in zip(xs, ys)]
        ax.scatter([x for x, o in zip(xs, ok) if o], [y for y, o in zip(ys, ok) if o], color=c, marker=mk,
                   s=45, alpha=0.8, label=rep)
    ax.axhline(1.0, ls=":", c="k", lw=0.8)
    ax.set_xlabel("Class-A rows per held-out pair (per seed)"); ax.set_ylabel("y_true Bayes ceiling")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("AF: per-pair raw-interface identifiability vs support (E, Class-A)")
    save(fig, "per_pair_support")

    # 5 target-class distribution held-out vs seen (Class-A)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    xi2 = np.arange(NCLASS)
    ho = res[PRIMARY_MODEL]["geometry"]["classA"]["heldout"]["counts"]
    seen = res[PRIMARY_MODEL]["geometry"]["classA"]["seen"]["counts"]
    ho = np.array(ho, float); seen = np.array(seen, float)
    ax.bar(xi2 - 0.2, ho / ho.sum(), 0.4, label="held-out operands", color="tab:purple")
    ax.bar(xi2 + 0.2, seen / seen.sum(), 0.4, label="seen operands", color="tab:gray")
    ax.axhline(0.1, ls=":", c="gray", lw=0.8, label="uniform 0.1")
    ax.set_xticks(xi2); ax.set_xticklabels([str(c) for c in range(10)])
    ax.set_xlabel("target class y_true"); ax.set_ylabel("fraction (Class-A)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AF: Class-A target-class distribution, held-out vs seen (pooled seeds)")
    save(fig, "target_distribution")

    # 6 entropy by group (Class-A held-out)
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ents = [res[PRIMARY_MODEL]["geometry"]["classA"][g]["entropy"]["mean"] for g in groups]
    ax.bar(xi, ents, color="tab:green")
    ax.axhline(np.log2(10), ls=":", c="k", lw=0.8, label="uniform 10-class entropy")
    ax.set_xticks(xi); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("target entropy (bits)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AF: Class-A held-out target entropy by stratum (E)")
    save(fig, "entropy_by_group")

    # 7 ceiling gap vs overall (E, all groups, L4/Lfull, Class-A and all-ADD)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    labs = [f"{pop}\n{rep}" for pop in POPS for rep in DECISIVE]
    gaps = []
    for pop in POPS:
        for rep in DECISIVE:
            cl = res["classification"]["cells"]
            gaps.append(max((cl[f"{pop}|{rep}|{g}"]["mean_gap"] or 0.0) for g in HELD_OUT_GROUP_NAMES))
    ax.bar(np.arange(len(labs)), gaps, color=["tab:blue", "tab:cyan", "tab:orange", "tab:red"])
    ax.axhline(MATERIAL_GAP, ls="--", c="k", lw=0.9, label="0.15 descriptive gap")
    ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("max subgroup ceiling gap vs overall")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AF: worst held-out subgroup identifiability gap (E)")
    save(fig, "ceiling_gap")

    # 8 decision summary
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    d = res["classification"]
    cats = ["overall\nClass-A L4", "worst held-out\nsubgroup (L4)", "overall\nall-ADD L4",
            "worst held-out\nsubgroup (all-ADD)"]
    ovA = res[PRIMARY_MODEL]["identifiability"]["classA"]["L4"]["overall"]["y_true__bayes_acc"]["mean"]
    ovB = res[PRIMARY_MODEL]["identifiability"]["all_add"]["L4"]["overall"]["y_true__bayes_acc"]["mean"]
    wA = max(res[PRIMARY_MODEL]["identifiability"]["classA"]["L4"][g]["y_true__bayes_acc"]["mean"]
             for g in HELD_OUT_GROUP_NAMES)
    wB = max(res[PRIMARY_MODEL]["identifiability"]["all_add"]["L4"][g]["y_true__bayes_acc"]["mean"]
             for g in HELD_OUT_GROUP_NAMES)
    ax.bar(np.arange(4), [ovA, wA, ovB, wB], color=["tab:orange", "tab:purple", "tab:blue", "tab:cyan"])
    ax.axhline(1.0, ls=":", c="k", lw=0.8)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(cats, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("exact-interface y_true Bayes ceiling"); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"AF: decision summary ({d['verdict']})")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AF analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--meta", default="experiments/experiment_af_task_difficulty_meta.json")
    ap.add_argument("--output", default="experiments/experiment_af_results.json")
    ap.add_argument("--raw", default="experiments/experiment_af_task_difficulty_raw.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_af")
    ap.add_argument("--use-raw", action="store_true")
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=False)
    ad_meta = json.loads(str(z["meta_json"][0]))
    af_meta = json.load(open(args.meta)) if os.path.exists(args.meta) else {}
    data = {"z": z, "meta": ad_meta, "af_meta": af_meta}
    t0 = time.perf_counter()
    res = build(data)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    # retained raw subgroup statistics (per-seed geometry + identifiability)
    raw = {m: {"per_seed_geometry": [s["geom"] for s in res[m]["per_seed"]],
               "per_seed_identifiability": [s["ident"] for s in res[m]["per_seed"]]} for m in MODELS}
    with open(args.raw, "w") as f:
        json.dump(raw, f)
    print(f"Wrote {args.output} and {args.raw}")
    d = res["classification"]
    print(f"AF verdict: {d['verdict']}")
    for pop in POPS:
        for rep in DECISIVE:
            ov = res[PRIMARY_MODEL]["identifiability"][pop][rep]["overall"]["y_true__bayes_acc"]["mean"]
            print(f"  {pop} {rep}: overall y_true Bayes={ov:.4f} | " + " ".join(
                f"{g}={res[PRIMARY_MODEL]['identifiability'][pop][rep][g]['y_true__bayes_acc']['mean']:.3f}"
                f"(n={res[PRIMARY_MODEL]['identifiability'][pop][rep][g]['n']['mean']:.0f})"
                for g in HELD_OUT_GROUP_NAMES))
    paths = make_figures(res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
