#!/usr/bin/env python3
"""Experiment AG — analysis: bounded-reader training-pool coverage and generalization stress.

AE showed the AdamW readers fit the training pool far above its baseline without consistent transfer to
the held-out operand pairs; AF removed task-identifiability as the explanation. AG asks a reader-side
question conditional on the readers/data/protocol tested:

    Is the fit–transfer gap sensitive to the sampling of the ordinary-ADD training pool, in particular
    operand-pair coverage and target-label composition, with the held-out pairs (b in {0,5}) excluded
    from every condition?

AG is an analysis-reader training experiment, NOT a RE:Reason training or representation experiment.
Readers, optimizer, protocol, validation/test sets, target and seeds are fixed; only the training-pool
construction changes, with matched model initialization across conditions.

Prespecified training-pool conditions (eligible pool = V trainord rows with b not in {0,5}):
    ae_native                : the exact eligible pool (AE's heldout-variant anchor), no resampling;
    natural_bootstrap        : sample-with-replacement from the eligible pool (empirical distribution);
    pair_balanced_bootstrap  : draw eligible operand pairs uniformly, then an example from that pair;
    target_balanced_bootstrap: draw the 10 target classes uniformly, then an example from that class.
Conditions 2–4 keep n = the seed's ae_native pool size; three deterministic replicates per seed; the
sampling seed is separate from the model-initialization seed. Held-out pairs are never trained on.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_ag.py --jobs 16
    PYTHONPATH=src python experiments/analyze_experiment_ag.py --jobs 16 --use-raw
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_experiment_ab as ab  # noqa: E402
import analyze_experiment_ad as ad  # noqa: E402
import analyze_experiment_ae as ae  # noqa: E402
import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_u as au  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import analyze_experiment_x as ax  # noqa: E402

PRIMARY_MODEL = "E_candidate"
SEEDS = av.SEEDS
READERS = ["GRU", "TCN"]
REPS = au.REPS
REP_LABELS = [r for r, _ in REPS]
DECISIVE = ["L4", "Lfull"]
NCLASS = 10
CHANCE = 1.0 / NCLASS
HELD_OUT_B = (0, 5)
PAIRS_HELDOUT = [(a, b) for a in range(10) for b in HELD_OUT_B]
NON_HELD_PAIRS = [(a, b) for a in range(10) for b in range(10) if b not in HELD_OUT_B]

CONDITIONS = ["ae_native", "natural_bootstrap", "pair_balanced_bootstrap", "target_balanced_bootstrap"]
BOOTSTRAP_CONDITIONS = CONDITIONS[1:]
REPLICATES = 3
UPDATES = ad.UPDATES
EVAL_EVERY = ad.EVAL_EVERY
COND = ad.COND_BY_NAME["adamw"]

SAMPLE_BASE = as_.MLP_SEED + 8_000_000       # distinct from av.reader_seed's model-init seeds
MATERIAL_CHANGE = 0.15                        # descriptive fit-change bar (U/V convention)
CONSISTENT_SEEDS = 4
PAIR_MAJORITY = 11                            # > half of the 20 held-out pairs


def _ms(vals):
    return as_.mean_sd(vals)


def _entropy(counts) -> float:
    c = np.asarray(counts, float)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c[c > 0] / tot
    return float(-(p * np.log2(p)).sum())


def _sample_seed(seed: int, cond: str, replicate: int) -> int:
    return SAMPLE_BASE + seed * 1000 + CONDITIONS.index(cond) * 100 + replicate


def _sample_positions(oa: np.ndarray, ob: np.ndarray, y: np.ndarray, cond: str,
                      rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    if cond == "ae_native":
        return np.arange(n)
    if cond == "natural_bootstrap":
        return rng.integers(0, n, size=n)
    if cond == "pair_balanced_bootstrap":
        by: Dict[Tuple[int, int], List[int]] = {}
        for i, (a, b) in enumerate(zip(oa, ob)):
            by.setdefault((int(a), int(b)), []).append(i)
        keys = sorted(by)
        sel = np.empty(n, int)
        for k in range(n):
            lst = by[keys[rng.integers(len(keys))]]
            sel[k] = lst[rng.integers(len(lst))]
        return sel
    if cond == "target_balanced_bootstrap":
        by: Dict[int, List[int]] = {}
        for i, c in enumerate(y):
            by.setdefault(int(c), []).append(i)
        keys = sorted(by)
        sel = np.empty(n, int)
        for k in range(n):
            lst = by[keys[rng.integers(len(keys))]]
            sel[k] = lst[rng.integers(len(lst))]
        return sel
    raise ValueError(cond)


def _pool_diag(oa: np.ndarray, ob: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    pair_counts = [int(((oa == a) & (ob == b)).sum()) for a, b in NON_HELD_PAIRS]
    class_counts = [int((y == c).sum()) for c in range(NCLASS)]
    n = len(y)
    return {"n": n, "n_unique": int(np.unique(np.stack([oa, ob, y], 1), axis=0).shape[0]),
            "pairs_present": int(sum(1 for c in pair_counts if c > 0)),
            "pair_min": int(min(pair_counts)), "pair_max": int(max(pair_counts)),
            "class_counts": class_counts, "class_entropy": _entropy(class_counts),
            "class_min_share": float(min(class_counts) / n) if n else None,
            "class_max_share": float(max(class_counts) / n) if n else None}


def _task(spec: Tuple[int, str, str, str, int]) -> Dict[str, Any]:
    seed, reader, rep, cond, replicate = spec
    ad._G["feat"] = {}
    z = ad._G["z"]
    arch = ad._G["arch"]
    L = dict(REPS)[rep]

    tr = np.nonzero(z[f"{PRIMARY_MODEL}__trainord__seed"] == seed)[0]
    held_tr = au._pairs(z, PRIMARY_MODEL, "trainord", tr)
    keep = ~held_tr
    elig = tr[keep]
    Xelig = ad._feat("trainord", seed, L)[keep]
    y_e = z[f"{PRIMARY_MODEL}__trainord__y_true"][elig].astype(int)
    oa_e = z[f"{PRIMARY_MODEL}__trainord__operand_a"][elig].astype(int)
    ob_e = z[f"{PRIMARY_MODEL}__trainord__operand_b"][elig].astype(int)

    rng = np.random.default_rng(_sample_seed(seed, cond, replicate))
    pos = _sample_positions(oa_e, ob_e, y_e, cond, rng)
    Xtr, ytr = Xelig[pos], y_e[pos]
    diag = _pool_diag(oa_e[pos], ob_e[pos], ytr)

    va = np.nonzero(z[f"{PRIMARY_MODEL}__valord__seed"] == seed)[0]
    Xva = ad._feat("valord", seed, L)
    yva = z[f"{PRIMARY_MODEL}__valord__y_true"][va].astype(int)
    te_sel = z[f"{PRIMARY_MODEL}__test__seed"] == seed
    te_idx = np.nonzero(te_sel)[0]
    held_te = au._pairs(z, PRIMARY_MODEL, "test", te_idx)
    masks = au._pop_masks(z, PRIMARY_MODEL, seed)
    ClsA = masks["classA"][te_sel]; allA = masks["all_add"][te_sel]
    matched = masks["matched_correct"][te_sel]; correct = masks["correct_all"][te_sel]
    yte = z[f"{PRIMARY_MODEL}__test__y_true"][te_sel].astype(int)
    oa_te = z[f"{PRIMARY_MODEL}__test__operand_a"][te_sel].astype(int)
    ob_te = z[f"{PRIMARY_MODEL}__test__operand_b"][te_sel].astype(int)

    # matched model initialization across conditions (optimizer condition is not part of the seed)
    fit_seed = av.reader_seed(seed, reader, "y_true", rep, "heldout")
    init_h = ad._init_hash(reader, NCLASS, fit_seed, COND["init"], arch)
    fit = ab.train_ab(reader, Xtr, ytr, Xva, yva, NCLASS, fit_seed, UPDATES, COND, arch,
                      eval_every=EVAL_EVERY)
    state = fit["best_val"][1]
    Xte = ad._feat("test", seed, L)
    pred_fit = ax.predict_state(reader, state, Xtr, None, NCLASS, av.INPUT_DIM, arch)
    pred_base = ax.predict_state(reader, state, Xelig, None, NCLASS, av.INPUT_DIM, arch)
    pred_va = ax.predict_state(reader, state, Xva, None, NCLASS, av.INPUT_DIM, arch)
    pred_te = ax.predict_state(reader, state, Xte, None, NCLASS, av.INPUT_DIM, arch)

    pops = {
        "train_fit": ae._summarize(pred_fit, ytr, ytr),
        "train_base": ae._summarize(pred_base, y_e, y_e),
        "val": ae._summarize(pred_va, yva, ytr),
        "test_classA": ae._summarize(pred_te[ClsA], yte[ClsA], ytr),
        "test_all_add": ae._summarize(pred_te[allA], yte[allA], ytr),
        "heldout_classA": ae._summarize(pred_te[ClsA & held_te], yte[ClsA & held_te], ytr),
        "seen_classA": ae._summarize(pred_te[ClsA & ~held_te], yte[ClsA & ~held_te], ytr),
        "heldout_all": ae._summarize(pred_te[allA & held_te], yte[allA & held_te], ytr),
        "seen_all": ae._summarize(pred_te[allA & ~held_te], yte[allA & ~held_te], ytr),
    }
    per_pair = {}
    for a, b in PAIRS_HELDOUT:
        m = ClsA & (oa_te == a) & (ob_te == b)
        per_pair[f"{a}_{b}"] = ({"n": int(m.sum()),
                                 "acc": float((pred_te[m] == yte[m]).mean())} if m.any() else None)
    return {"seed": seed, "reader": reader, "rep": rep, "condition": cond, "replicate": replicate,
            "init_hash": init_h, "params": av.reader_param_count(reader, NCLASS, arch),
            "best_val_acc": float(fit["best_val"][0]), "best_val_update": int(fit["best_val"][2]),
            "finite": bool(fit["finite"]), "pool": diag, "pops": pops, "per_pair": per_pair}


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _agg_pops(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    blocks = [b for b in blocks if b is not None]
    if not blocks:
        return {}
    out: Dict[str, Any] = {}
    for k in blocks[0]:
        vals = [b[k] for b in blocks]
        v0 = vals[0]
        if isinstance(v0, list):
            out[k] = [_ms([v[i] for v in vals]) for i in range(len(v0))]
        else:
            out[k] = _ms(vals)
    return out


def aggregate(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for reader in READERS:
        for rep in REP_LABELS:
            for cond in CONDITIONS:
                recs = [p for p in parts if p["reader"] == reader and p["rep"] == rep
                        and p["condition"] == cond]
                if not recs:
                    continue
                nodes: Dict[str, Any] = {}
                for key in ("train_fit", "train_base", "val", "test_classA", "test_all_add",
                            "heldout_classA", "seen_classA", "heldout_all", "seen_all"):
                    nodes[key] = _agg_pops([r["pops"][key] for r in recs])
                pool = {k: _ms([r["pool"][k] for r in recs])
                        for k in ("n", "n_unique", "pairs_present", "pair_min", "pair_max",
                                  "class_entropy", "class_min_share", "class_max_share")}
                pool["class_counts"] = [int(np.mean([r["pool"]["class_counts"][i] for r in recs]))
                                        for i in range(NCLASS)]
                per_pair = {f"{a}_{b}": _ms([(r["per_pair"].get(f"{a}_{b}") or {}).get("acc")
                                             for r in recs]) for a, b in PAIRS_HELDOUT}
                # per-seed means over replicates (paired comparisons)
                per_seed: Dict[str, Any] = {}
                for s in SEEDS:
                    srecs = [r for r in recs if r["seed"] == s]
                    per_seed[str(s)] = {
                        "heldout_classA_raw": float(np.mean([r["pops"]["heldout_classA"]["raw_acc"]
                                                             for r in srecs])),
                        "heldout_classA_bal": float(np.mean([r["pops"]["heldout_classA"]["balanced_acc"]
                                                             for r in srecs])),
                        "heldout_all_raw": float(np.mean([r["pops"]["heldout_all"]["raw_acc"]
                                                          for r in srecs])),
                        "train_base": float(np.mean([r["pops"]["train_base"]["raw_acc"] for r in srecs])),
                        "train_fit": float(np.mean([r["pops"]["train_fit"]["raw_acc"] for r in srecs])),
                        "val": float(np.mean([r["pops"]["val"]["raw_acc"] for r in srecs]))}
                out.setdefault(reader, {}).setdefault(rep, {})[cond] = {
                    "n_draws": len(recs), "pops": nodes, "pool": pool, "per_pair": per_pair,
                    "per_seed": per_seed,
                    "init_hashes": sorted({r["init_hash"] for r in recs})}
    return out


# ---------------------------------------------------------------------------
# decision (descriptive)
# ---------------------------------------------------------------------------

def classify(res: Dict[str, Any]) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    transfer_signals: List[str] = []
    fit_changes: List[str] = []
    for reader in READERS:
        for rep in DECISIVE:
            native = res["readers"][reader][rep]["ae_native"]
            native_bal = [native["per_seed"][str(s)]["heldout_classA_bal"] for s in SEEDS]
            native_fit = [native["per_seed"][str(s)]["train_base"] for s in SEEDS]
            native_pair = {k: native["per_pair"][k]["mean"] for k in native["per_pair"]}
            for cond in BOOTSTRAP_CONDITIONS:
                node = res["readers"][reader][rep][cond]
                bal = [node["per_seed"][str(s)]["heldout_classA_bal"] for s in SEEDS]
                fit = [node["per_seed"][str(s)]["train_base"] for s in SEEDS]
                d_bal = [b - n for b, n in zip(bal, native_bal)]
                d_fit = [f - n for f, n in zip(fit, native_fit)]
                pair_improved = sum(1 for k in native_pair
                                    if node["per_pair"][k]["mean"] is not None
                                    and native_pair[k] is not None
                                    and node["per_pair"][k]["mean"] > native_pair[k])
                mean_dbal = float(np.mean(d_bal))
                mean_dfit = float(np.mean(d_fit))
                pos_bal = sum(1 for d in d_bal if d > 0)
                # consistent fit change in EITHER direction
                if mean_dfit >= 0:
                    n_fit_change = sum(1 for d in d_fit if d >= MATERIAL_CHANGE)
                else:
                    n_fit_change = sum(1 for d in d_fit if d <= -MATERIAL_CHANGE)
                tsignal = bool(mean_dbal > 0 and pos_bal >= CONSISTENT_SEEDS and pair_improved >= PAIR_MAJORITY)
                fchange = bool(abs(mean_dfit) >= MATERIAL_CHANGE and n_fit_change >= CONSISTENT_SEEDS)
                key = f"{reader}|{rep}|{cond}"
                cells[key] = {"heldout_classA_bal_native": float(np.mean(native_bal)),
                              "heldout_classA_bal_cond": float(np.mean(bal)),
                              "mean_delta_bal": mean_dbal, "seeds_bal_positive": pos_bal,
                              "train_base_native": float(np.mean(native_fit)),
                              "train_base_cond": float(np.mean(fit)),
                              "mean_delta_fit": mean_dfit, "seeds_fit_change": n_fit_change,
                              "pairs_improved": pair_improved, "transfer_signal": tsignal,
                              "fit_change": fchange}
                if tsignal:
                    transfer_signals.append(key)
                if fchange:
                    fit_changes.append(key)
    if transfer_signals:
        verdict = "sampling_sensitive_fit_transfer_gap"
    elif fit_changes:
        verdict = "persistent_fit_without_transfer"
    else:
        verdict = "inconclusive_mixed"
    return {"verdict": verdict, "transfer_signals": transfer_signals, "fit_changes": fit_changes,
            "cells": cells,
            "thresholds": {"material_change": MATERIAL_CHANGE, "consistent_seeds": CONSISTENT_SEEDS,
                           "pair_majority": PAIR_MAJORITY},
            "rule_note": ("Descriptive (categories, not acceptance thresholds). transfer_signal if a "
                          "bootstrap condition improves held-out Class-A balanced accuracy over "
                          "ae_native in >=4/5 seeds AND improves >10/20 held-out operand pairs; "
                          "fit_change if training accuracy changes (either direction) by >=0.15 in "
                          ">=4/5 seeds. sampling_sensitive_fit_transfer_gap if any transfer signal; "
                          "else persistent_fit_without_transfer if training fit changes consistently "
                          "but held-out balanced accuracy stays weak/inconsistent; else "
                          "inconclusive_mixed. Replicates are nested draws, not independent "
                          "checkpoints. V-A/V-B are not altered.")}


def accessibility(res: Dict[str, Any]) -> Dict[str, Any]:
    """V's V-A criterion (unchanged): any condition/reader/history with Class-A >=0.50 and held-out >=0.30."""
    best = None
    for reader in READERS:
        for rep in DECISIVE:
            for cond in CONDITIONS:
                node = res["readers"][reader][rep][cond]["pops"]
                a = node["test_classA"]["raw_acc"]["mean"]
                h = node["heldout_classA"]["raw_acc"]["mean"]
                if a is None or h is None:
                    continue
                if best is None or a > best["classA"]:
                    best = {"reader": reader, "rep": rep, "cond": cond, "classA": a, "heldout": h}
    vA = bool(best and best["classA"] >= 0.50 and best["heldout"] >= 0.30)
    return {"V_A_reached": vA, "best": best,
            "note": "V's V-A accessibility criterion is unchanged; AG does not re-define it."}


def _collect(data: Dict[str, Any], jobs: int, readers: List[str],
             reps: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    npz_path = data["path"]
    arch = av.architecture()
    reps = reps or REP_LABELS
    specs = []
    for s in SEEDS:
        for R in readers:
            for rep in reps:
                specs.append((s, R, rep, "ae_native", 0))
                for cond in BOOTSTRAP_CONDITIONS:
                    for r in range(REPLICATES):
                        specs.append((s, R, rep, cond, r))
    t0 = time.perf_counter()
    if jobs <= 1:
        ad._init_worker(npz_path, arch)
        return [_task(sp) for sp in specs]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs, initializer=ad._init_worker, initargs=(npz_path, arch)) as pool:
        parts, done = [], 0
        for part in pool.imap_unordered(_task, specs, chunksize=1):
            parts.append(part)
            done += 1
            if done % 20 == 0 or done == len(specs):
                print(f"  [{done}/{len(specs)}] fits done", flush=True)
    print(f"collection wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    parts.sort(key=lambda p: (p["reader"], p["seed"], p["rep"], p["condition"], p["replicate"]))
    return parts


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(res, out_dir, prefix="experiment_ag") -> List[str]:
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

    def nz(v):
        return np.nan if v is None else v

    R = res["readers"]
    cmap = {"ae_native": "tab:gray", "natural_bootstrap": "tab:blue",
            "pair_balanced_bootstrap": "tab:orange", "target_balanced_bootstrap": "tab:green"}

    def val(reader, rep, cond, pop, field):
        return R[reader][rep][cond]["pops"][pop][field]["mean"]

    # 1 held-out Class-A balanced accuracy by condition (L4/Lfull, readers)
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    xi = np.arange(len(CONDITIONS))
    for reader in READERS:
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            ax.plot(xi, [nz(val(reader, rep, c, "heldout_classA", "balanced_acc")) for c in CONDITIONS],
                    marker="o", ls=ls, color=("tab:blue" if reader == "GRU" else "tab:orange"),
                    alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep}")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(xi); ax.set_xticklabels([c.replace("_", "\n") for c in CONDITIONS], fontsize=7)
    ax.set_ylabel("held-out Class-A balanced accuracy"); ax.set_ylim(0, 0.4)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AG: held-out balanced accuracy by training-pool condition")
    save(fig, "heldout_bal_by_condition")

    # 2 training fit (base) vs held-out balanced accuracy
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for reader in READERS:
        for cond in CONDITIONS:
            for rep in DECISIVE:
                x = val(reader, rep, cond, "train_base", "raw_acc")
                y = val(reader, rep, cond, "heldout_classA", "balanced_acc")
                ax.scatter(nz(x), nz(y), color=cmap[cond], marker=("o" if rep == "L4" else "s"),
                           s=45, alpha=0.8)
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 0.4)
    ax.set_xlabel("training accuracy (eligible base pool)"); ax.set_ylabel("held-out Class-A balanced")
    handles = [plt.Line2D([], [], color=cmap[c], marker="o", ls="") for c in CONDITIONS]
    ax.legend(handles, CONDITIONS, fontsize=7)
    ax.grid(alpha=0.3); ax.set_title("AG: fit vs held-out transfer by condition")
    save(fig, "fit_vs_transfer")

    # 3 per-seed paired deltas of held-out balanced accuracy vs ae_native
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    labels, data = [], []
    for reader in READERS:
        for rep in DECISIVE:
            for cond in BOOTSTRAP_CONDITIONS:
                labels.append(f"{reader}\n{rep}\n{cond.split('_')[0]}")
                native = [R[reader][rep]["ae_native"]["per_seed"][str(s)]["heldout_classA_bal"] for s in SEEDS]
                cur = [R[reader][rep][cond]["per_seed"][str(s)]["heldout_classA_bal"] for s in SEEDS]
                data.append([c - n for c, n in zip(cur, native)])
    xi = np.arange(len(labels))
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i, float) + np.linspace(-0.15, 0.15, len(d)), d, s=18, color="tab:purple")
    ax.axhline(0, c="gray", lw=0.8)
    ax.set_xticks(xi); ax.set_xticklabels(labels, fontsize=5.5, rotation=0)
    ax.set_ylabel("held-out Class-A balanced Δ vs ae_native")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("AG: per-seed paired deltas (bootstrap conditions vs native)")
    save(fig, "paired_deltas")

    # 4 realized target-class composition of the generated pools
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    xi2 = np.arange(NCLASS)
    w = 0.2
    for i, cond in enumerate(CONDITIONS):
        counts = R["TCN"]["L4"][cond]["pool"]["class_counts"]
        tot = sum(counts) or 1
        ax.bar(xi2 + (i - 1.5) * w, [c / tot for c in counts], w, label=cond, color=cmap[cond])
    ax.axhline(0.1, ls=":", c="k", lw=0.8, label="uniform 0.1")
    ax.set_xticks(xi2); ax.set_xticklabels([str(c) for c in range(10)])
    ax.set_xlabel("target class"); ax.set_ylabel("realized share")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AG: realized target-class composition by condition (TCN, L4)")
    save(fig, "pool_target_composition")

    # 5 realized pair coverage
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for i, cond in enumerate(CONDITIONS):
        pmin = R["TCN"]["L4"][cond]["pool"]["pair_min"]["mean"]
        pmax = R["TCN"]["L4"][cond]["pool"]["pair_max"]["mean"]
        pres = R["TCN"]["L4"][cond]["pool"]["pairs_present"]["mean"]
        ax.bar(i - 0.2, pmin, 0.4, color=cmap[cond])
        ax.bar(i + 0.2, pmax, 0.4, color=cmap[cond], alpha=0.5)
        ax.text(i, pmax + 1, f"{pres:.0f}/80 pairs", ha="center", fontsize=7)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([c.replace("_", "\n") for c in CONDITIONS], fontsize=7)
    ax.set_ylabel("per-pair count (min solid / max faded)")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("AG: realized operand-pair coverage (TCN, L4)")
    save(fig, "pool_pair_coverage")

    # 6 per-pair held-out accuracy heatmap (TCN L4, native vs pair-balanced)
    for cond in ("ae_native", "pair_balanced_bootstrap", "target_balanced_bootstrap"):
        M = np.array([[nz(R["TCN"]["L4"][cond]["per_pair"][f"{a}_{b}"]["mean"]) for b in HELD_OUT_B]
                      for a in range(10)], float)
        fig, ax = plt.subplots(figsize=(5.0, 5.4))
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"b={b}" for b in HELD_OUT_B])
        ax.set_yticks(range(10)); ax.set_yticklabels([f"a={a}" for a in range(10)])
        for i in range(10):
            for j in range(len(HELD_OUT_B)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
        fig.colorbar(im, ax=ax); ax.set_title(f"AG: held-out Class-A by pair ({cond}, TCN/L4)")
        save(fig, f"per_pair_{cond}")

    # 7 balanced vs raw held-out accuracy (label-composition guard)
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    labs, raw, bal = [], [], []
    for reader in READERS:
        for rep in DECISIVE:
            for cond in CONDITIONS:
                labs.append(f"{reader}/{rep}/{cond.split('_')[0]}")
                raw.append(nz(val(reader, rep, cond, "heldout_classA", "raw_acc")))
                bal.append(nz(val(reader, rep, cond, "heldout_classA", "balanced_acc")))
    xi = np.arange(len(labs))
    ax.bar(xi - 0.2, raw, 0.4, label="raw", color="tab:blue")
    ax.bar(xi + 0.2, bal, 0.4, label="balanced", color="tab:green")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(xi); ax.set_xticklabels(labs, fontsize=5, rotation=90)
    ax.set_ylabel("held-out Class-A accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AG: held-out raw vs balanced accuracy (all conditions)")
    save(fig, "heldout_raw_vs_balanced")

    # 8 decision summary
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    d = res["primary_decision"]
    cats = ["train fit\n(native)", "train fit\n(worst cond)", "held-out bal\n(native)",
            "held-out bal\n(best cond)"]
    fits = [val("TCN", "L4", c, "train_base", "raw_acc") for c in CONDITIONS]
    bals = [val("TCN", "L4", c, "heldout_classA", "balanced_acc") for c in CONDITIONS]
    vals = [nz(val("TCN", "L4", "ae_native", "train_base", "raw_acc")), nz(max(fits)),
            nz(val("TCN", "L4", "ae_native", "heldout_classA", "balanced_acc")), nz(max(bals))]
    ax.bar(np.arange(4), vals, color=["tab:orange", "gold", "tab:purple", "tab:brown"])
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(cats, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"AG: decision summary ({d['verdict']}) [TCN/L4]")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AG analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--meta", default="experiments/experiment_ag_training_pool_stress_meta.json")
    ap.add_argument("--output", default="experiments/experiment_ag_results.json")
    ap.add_argument("--raw", default="experiments/experiment_ag_grid_raw.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_ag")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--readers", nargs="*", default=None)
    ap.add_argument("--reps", nargs="*", default=None)
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--use-raw", action="store_true")
    args = ap.parse_args()
    data = ad.load(args.npz)
    data["path"] = args.npz
    data["ag_meta"] = json.load(open(args.meta)) if os.path.exists(args.meta) else {}
    problems = ad._validate(data["meta"])
    if problems:
        raise SystemExit(f"AG source invalid: {problems}")
    if args.use_raw and os.path.exists(args.raw):
        with open(args.raw) as f:
            parts = json.load(f)["parts"]
    else:
        parts = _collect(data, args.jobs, args.readers or READERS, args.reps)
        os.makedirs(os.path.dirname(args.raw), exist_ok=True)
        with open(args.raw, "w") as f:
            json.dump({"parts": parts}, f)
        if args.collect_only:
            print(f"Wrote raw {args.raw} ({len(parts)} fits)")
            return
    t0 = time.perf_counter()
    res = _assemble(data, parts)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"AG verdict: {d['verdict']} | V-A reached: {res['accessibility']['V_A_reached']}")
    for k, v in d["cells"].items():
        print(f"  {k}: held-out bal {v['heldout_classA_bal_native']:.3f}->{v['heldout_classA_bal_cond']:.3f} "
              f"(Δ{v['mean_delta_bal']:+.4f}, seeds+{v['seeds_bal_positive']}/5, pairs+{v['pairs_improved']}/20) "
              f"fit {v['train_base_native']:.3f}->{v['train_base_cond']:.3f} (Δ{v['mean_delta_fit']:+.3f})")
    paths = make_figures(res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


def _assemble(data: Dict[str, Any], parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "readers": READERS,
        "target": "y_true", "n_classes": NCLASS, "reps": REP_LABELS, "decisive": DECISIVE,
        "conditions": CONDITIONS, "bootstrap_conditions": BOOTSTRAP_CONDITIONS,
        "replicates": REPLICATES, "held_out_b": list(HELD_OUT_B), "held_out_pairs": len(PAIRS_HELDOUT),
        "optimizer": {"name": "adamw", **{k: COND[k] for k in ("opt", "init")}},
        "lr": ad.LR, "weight_decay": ad.WD, "updates": UPDATES, "eval_every": EVAL_EVERY,
        "sample_base": SAMPLE_BASE,
        "source_identity": data["meta"].get("source"), "checkpoints": data["meta"].get("models"),
        "eligibility_support": data.get("ag_meta", {}).get("eligibility_support"),
        "statistical_discipline": (
            "Five frozen E checkpoints are the independent units; the three resampling replicates per "
            "seed are nested draws, not independent checkpoints. Readers are analysis-only, fitted on "
            "held-out-excluded ordinary-ADD pools, tuned on V's canonical VAL, evaluated once on the "
            "untouched canonical TEST at the best-on-VAL state. Model initialization is matched across "
            "conditions (same av.reader_seed); sampling seeds are separate. Selection is VAL-only. "
            "Majority baselines are training-derived; balanced accuracy and per-class recall are "
            "reported so raw-accuracy changes from label composition are not mistaken for rule recovery. "
            "V-A/V-B are unchanged.")}}
    res["architecture"] = av.architecture()
    res["readers"] = aggregate(parts)
    res["per_draw"] = parts
    res["primary_decision"] = classify(res)
    res["accessibility"] = accessibility(res)
    return res


if __name__ == "__main__":
    main()
