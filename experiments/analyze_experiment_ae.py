#!/usr/bin/env python3
"""Experiment AE — analysis: baseline-aware fit-transfer diagnostic (AD's AdamW condition).

AD (AD-B) showed the AB-qualified `adamw` recipe removes the analysis readers' *fitting* obstacle at
V's full scale (best-on-VAL training accuracy 0.44–0.86) with no transfer to the untouched canonical
TEST. AE asks a single further measurement question:

    Do the readers' high-training-accuracy AdamW solutions show evidence of transferable structure on
    held-out operand pairs, or do they predominantly fit the observed training pool?

AE is measurement-only. It does NOT test RE:Reason's frozen representation; the readers are separate
analysis-only models on a computed label. AE does NOT introduce a new target, alternate loss,
representation probe, architecture, optimizer or training sweep.

Source / reproduction
---------------------
AE inspects AD's retained artifacts and finds that AD retained the best-on-VAL scalar metrics and the
train/val trajectories but **no state dictionaries**, so the selected reader states cannot be reused.
AE therefore **deterministically reproduces the exact AdamW fits** with AD's trainer
(`ab.train_ab`), V's source arrays, V's deterministic reader-seed convention (`av.reader_seed`) and the
AB-qualified `adamw` condition (lr 3e-3, wd 1e-2, 200 full-batch updates, best-on-VAL selection).
Only `adamw` is reproduced; no optimizer condition, initialization, target, loss or tuning is added.
(Recorded by `run_experiment_ae.py` in `experiments/experiment_ae_fit_transfer_meta.json`.)

Measurements
------------
For each seed / reader / history / variant at the best-on-VAL state:
  * training (fit pool), validation and canonical-test populations, including Class-A and all-ADD;
  * for the `heldout` variant, test rows with b in {0,5} (held-out operand pairs) and other rows
    (seen), reported separately; the same breakdown is reported for `main` as a subgroup analysis
    (those operands were present in its training pool, so it is not an unseen-operand test);
  * raw accuracy, balanced accuracy + per-class recall, a majority-class baseline derived exclusively
    from the corresponding training labels, and the test-label frequencies;
  * per-pair accuracy for the 20 excluded operand pairs and the target-class confusion, retained.

Interpretation is descriptive and baseline-aware (fit_without_transfer / partial_transfer /
inconclusive_mixed); V's V-A/V-B accessibility criterion is retained unchanged.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_ae.py --jobs 16
    PYTHONPATH=src python experiments/analyze_experiment_ae.py --jobs 16 --use-raw
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

import analyze_experiment_ab as ab  # noqa: E402  (AB's exact trainer/optimizer/init)
import analyze_experiment_ad as ad  # noqa: E402  (AD helpers: features, conditions, seeds)
import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_u as au  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import analyze_experiment_x as ax  # noqa: E402

PRIMARY_MODEL = av.PRIMARY_MODEL
CONTROL_MODEL = av.CONTROL_MODEL
SEEDS = av.SEEDS
READERS = ["GRU", "TCN"]
REPS = au.REPS
REP_LABELS = au.REP_LABELS
DECISIVE = ["L4", "Lfull"]
COND_NAME = "adamw"
COND = ad.COND_BY_NAME[COND_NAME]
UPDATES = ad.UPDATES
EVAL_EVERY = ad.EVAL_EVERY
NCLASS = 10
CHANCE = 1.0 / NCLASS
HELD_OUT_BS = (0, 5)
PAIRS = [(a, b) for a in range(10) for b in (0, 5)]
MATERIAL_GAIN = 0.15
RECOVER_BAR = 0.50
HELDOUT_MATERIAL = 0.30
HELDOUT_WEAK = 0.25

POP_KEYS = ["train", "val", "test_all_add", "test_classA", "test_matched", "test_correct",
            "test_seen_all", "test_heldout_all", "test_seen_classA", "test_heldout_classA"]


def _ms(vals):
    return as_.mean_sd(vals)


def _summarize(pred: np.ndarray, y: np.ndarray, train_labels: np.ndarray) -> Optional[Dict[str, Any]]:
    """Raw accuracy, balanced accuracy + per-class recall, majority baseline, label frequencies."""
    n = int(len(y))
    if n == 0:
        return None
    maj = int(np.bincount(train_labels, minlength=NCLASS)[:NCLASS].argmax())
    recalls = [float((pred[y == c] == c).mean()) if (y == c).any() else None for c in range(NCLASS)]
    present = [r for r in recalls if r is not None]
    return {
        "n": n,
        "raw_acc": float((pred == y).mean()),
        "balanced_acc": float(np.mean(present)) if present else None,
        "per_class_recall": recalls,
        "majority_class": maj,
        "majority_baseline_acc": float((y == maj).mean()),
        "label_counts": [int((y == c).sum()) for c in range(NCLASS)],
        "chance": CHANCE,
    }


def _task(spec: Tuple[int, str, str, str]) -> Dict[str, Any]:
    seed, reader, rep, variant = spec
    ad._G["feat"] = {}
    z = ad._G["z"]
    arch = ad._G["arch"]
    L = dict(REPS)[rep]

    tr = np.nonzero(z[f"{PRIMARY_MODEL}__trainord__seed"] == seed)[0]
    va = np.nonzero(z[f"{PRIMARY_MODEL}__valord__seed"] == seed)[0]
    te_sel = z[f"{PRIMARY_MODEL}__test__seed"] == seed
    te_idx = np.nonzero(te_sel)[0]
    held_te = au._pairs(z, PRIMARY_MODEL, "test", te_idx)
    masks = au._pop_masks(z, PRIMARY_MODEL, seed)
    ClsA = masks["classA"][te_sel]
    allA = masks["all_add"][te_sel]
    matched = masks["matched_correct"][te_sel]
    correct = masks["correct_all"][te_sel]
    yte = z[f"{PRIMARY_MODEL}__test__y_true"][te_sel].astype(int)
    oa_te = z[f"{PRIMARY_MODEL}__test__operand_a"][te_sel].astype(int)
    ob_te = z[f"{PRIMARY_MODEL}__test__operand_b"][te_sel].astype(int)
    yva = z[f"{PRIMARY_MODEL}__valord__y_true"][va].astype(int)
    Xva = ad._feat("valord", seed, L)
    Xte = ad._feat("test", seed, L)

    if variant == "main":
        Xtr = ad._feat("trainord", seed, L)
        ytr = z[f"{PRIMARY_MODEL}__trainord__y_true"][tr].astype(int)
        n_train = int(len(tr))
    else:
        held_tr = au._pairs(z, PRIMARY_MODEL, "trainord", tr)
        keep = ~held_tr
        Xtr = ad._feat("trainord", seed, L)[keep]
        ytr = z[f"{PRIMARY_MODEL}__trainord__y_true"][tr[keep]].astype(int)
        n_train = int(keep.sum())

    fit_seed = av.reader_seed(seed, reader, "y_true", rep, variant)
    fit = ab.train_ab(reader, Xtr, ytr, Xva, yva, NCLASS, fit_seed, UPDATES, COND, arch,
                      eval_every=EVAL_EVERY)
    state = fit["best_val"][1]
    pred_tr = ax.predict_state(reader, state, Xtr, None, NCLASS, av.INPUT_DIM, arch)
    pred_va = ax.predict_state(reader, state, Xva, None, NCLASS, av.INPUT_DIM, arch)
    pred_te = ax.predict_state(reader, state, Xte, None, NCLASS, av.INPUT_DIM, arch)

    pops: Dict[str, Any] = {
        "train": _summarize(pred_tr, ytr, ytr),
        "val": _summarize(pred_va, yva, ytr),
        "test_all_add": _summarize(pred_te[allA], yte[allA], ytr),
        "test_classA": _summarize(pred_te[ClsA], yte[ClsA], ytr),
        "test_matched": _summarize(pred_te[matched], yte[matched], ytr),
        "test_correct": _summarize(pred_te[correct], yte[correct], ytr),
        "test_seen_all": _summarize(pred_te[allA & ~held_te], yte[allA & ~held_te], ytr),
        "test_heldout_all": _summarize(pred_te[allA & held_te], yte[allA & held_te], ytr),
        "test_seen_classA": _summarize(pred_te[ClsA & ~held_te], yte[ClsA & ~held_te], ytr),
        "test_heldout_classA": _summarize(pred_te[ClsA & held_te], yte[ClsA & held_te], ytr),
    }

    # per-pair Class-A accuracy (all 100 operand pairs; the 20 with b in {0,5} are the held-out ones)
    per_pair: Dict[str, Any] = {}
    for a in range(10):
        for b in range(10):
            m = ClsA & (oa_te == a) & (ob_te == b)
            if m.any():
                per_pair[f"{a}_{b}"] = {"n": int(m.sum()),
                                        "acc": float((pred_te[m] == yte[m]).mean())}
    per_pair_heldout = {f"{a}_{b}": per_pair.get(f"{a}_{b}") for a, b in PAIRS}

    # target-class confusion on the Class-A test population
    conf = np.zeros((NCLASS, NCLASS), int)
    for t, p in zip(yte[ClsA], pred_te[ClsA]):
        conf[int(t), int(p)] += 1

    return {
        "seed": seed, "reader": reader, "rep": rep, "variant": variant,
        "init_hash": ad._init_hash(reader, NCLASS, fit_seed, COND["init"], arch),
        "params": av.reader_param_count(reader, NCLASS, arch), "n_train": n_train,
        "best_val_acc": float(fit["best_val"][0]), "best_val_update": int(fit["best_val"][2]),
        "finite": bool(fit["finite"]),
        "pops": pops,
        "per_pair_heldout": per_pair_heldout,
        "confusion_classA": conf.tolist(),
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _agg_pop(pops: List[Dict[str, Any]]) -> Dict[str, Any]:
    pops = [p for p in pops if p is not None]
    if not pops:
        return {}
    out: Dict[str, Any] = {}
    for k in pops[0]:
        vals = [p[k] for p in pops]
        v0 = vals[0]
        if isinstance(v0, list):
            out[k] = [_ms([v[i] for v in vals]) for i in range(len(v0))]
        else:
            out[k] = _ms(vals)
    return out


def aggregate(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {
        (p["reader"], p["seed"], p["rep"], p["variant"]): p for p in parts}
    out: Dict[str, Any] = {}
    for reader in READERS:
        for rep in REP_LABELS:
            for variant in ("main", "heldout"):
                cells = [by[(reader, s, rep, variant)] for s in SEEDS]
                node = {"params": cells[0]["params"], "n_train": _ms([c["n_train"] for c in cells]),
                        "best_val_acc": _ms([c["best_val_acc"] for c in cells]),
                        "best_val_update": _ms([c["best_val_update"] for c in cells]),
                        "all_finite": bool(all(c["finite"] for c in cells)),
                        "init_hashes": [c["init_hash"] for c in cells],
                        "pops": {k: _agg_pop([c["pops"][k] for c in cells]) for k in POP_KEYS}}
                if variant == "heldout":
                    node["per_pair_heldout"] = {
                        f"{a}_{b}": _ms([(c["per_pair_heldout"].get(f"{a}_{b}") or {}).get("acc")
                                         for c in cells]) for a, b in PAIRS}
                conf = np.zeros((NCLASS, NCLASS), int)
                for c in cells:
                    conf += np.array(c["confusion_classA"], int)
                node["confusion_classA"] = conf.tolist()
                out.setdefault(reader, {}).setdefault(rep, {})[variant] = node
    return out


def _m(pops: Dict[str, Any], key: str, field: str) -> Optional[float]:
    node = pops.get(key)
    if not node:
        return None
    return node.get(field, {}).get("mean")


# ---------------------------------------------------------------------------
# interpretation (descriptive, baseline-aware)
# ---------------------------------------------------------------------------

def classify(res: Dict[str, Any], parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {
        (p["reader"], p["seed"], p["rep"], p["variant"]): p for p in parts}
    cells: Dict[str, Any] = {}
    for reader in READERS:
        for rep in DECISIVE:
            main = res["readers"][reader][rep]["main"]["pops"]
            ho = res["readers"][reader][rep]["heldout"]["pops"]
            train_acc = _m(main, "train", "raw_acc")
            train_base = _m(main, "train", "majority_baseline_acc")
            test_clsA = _m(main, "test_classA", "raw_acc")
            ho_acc = _m(ho, "test_heldout_classA", "raw_acc")
            ho_base = _m(ho, "test_heldout_classA", "majority_baseline_acc")
            seen_acc = _m(ho, "test_seen_classA", "raw_acc")
            fit_gain = (train_acc - train_base) if (train_acc is not None and train_base is not None) else None
            transfer_gain = (ho_acc - ho_base) if (ho_acc is not None and ho_base is not None) else None
            # per-seed transfer gain
            per_seed = []
            for s in SEEDS:
                c = by[(reader, s, rep, "heldout")]
                a = c["pops"]["test_heldout_classA"]["raw_acc"] if c["pops"]["test_heldout_classA"] else None
                b = c["pops"]["test_heldout_classA"]["majority_baseline_acc"] if c["pops"]["test_heldout_classA"] else None
                per_seed.append(None if (a is None or b is None) else float(a) - float(b))
            pos = sum(1 for v in per_seed if v is not None and v > 0)
            # reproducibility across the 20 held-out operand pairs (Class-A), vs chance
            pair_means = [res["readers"][reader][rep]["heldout"]["per_pair_heldout"][f"{a}_{b}"]["mean"]
                          for a, b in PAIRS]
            pair_positive = sum(1 for v in pair_means if v is not None and v > CHANCE)
            # reproducibility across target classes (held-out Class-A recall) vs chance
            recalls = ho["test_heldout_classA"]["per_class_recall"]
            class_positive = sum(1 for r in recalls if r["mean"] is not None and r["mean"] > CHANCE)
            fits = bool((fit_gain is not None and fit_gain >= MATERIAL_GAIN) or
                        (train_acc is not None and train_acc >= 0.5))
            if fits and (transfer_gain is None or transfer_gain <= 0.0 or pos < 4):
                cat = "fit_without_transfer"
            elif (transfer_gain is not None and transfer_gain > 0 and pos >= 4 and pair_positive > 10):
                cat = "partial_transfer"
            else:
                cat = "inconclusive_mixed"
            cells[f"{reader}|{rep}"] = {
                "train_acc": train_acc, "train_baseline": train_base, "fit_gain_over_baseline": fit_gain,
                "test_classA_acc": test_clsA,
                "train_to_test_gap": (train_acc - test_clsA) if (train_acc is not None and test_clsA is not None) else None,
                "heldout_classA_acc": ho_acc, "heldout_classA_baseline": ho_base,
                "transfer_gain_over_baseline": transfer_gain,
                "seen_classA_acc": seen_acc,
                "seen_vs_heldout_gap": (seen_acc - ho_acc) if (seen_acc is not None and ho_acc is not None) else None,
                "per_seed_transfer_gain": per_seed, "positive_transfer_seeds": pos,
                "heldout_pair_means": pair_means, "heldout_pairs_above_chance": pair_positive,
                "heldout_classes_above_chance": class_positive,
                "fits": fits, "category": cat}
    fitted = [v for v in cells.values() if v["fits"]]
    if any(v["category"] == "partial_transfer" for v in fitted):
        overall = "partial_transfer"
    elif fitted and all(v["category"] == "fit_without_transfer" for v in fitted):
        overall = "fit_without_transfer"
    elif fitted:
        overall = "inconclusive_mixed"
    else:
        overall = "inconclusive_mixed"
    return {"verdict": overall, "cells": cells,
            "rule_note": ("Descriptive and baseline-aware (categories, not new acceptance thresholds): "
                          "fit_without_transfer if the reader fits above its training-derived baseline "
                          "while the held-out-operand accuracy is at/below its variant-matched "
                          "majority-class baseline or the per-seed gain is positive in fewer than 4/5 "
                          "seeds; partial_transfer if the held-out accuracy beats its baseline in >=4/5 "
                          "seeds AND a majority (>10/20) of the held-out operand pairs exceed chance; else "
                          "inconclusive_mixed. V's V-A/V-B accessibility criterion is retained unchanged "
                          "and reported separately.")}


def accessibility(res: Dict[str, Any], u_base: Dict[str, Any]) -> Dict[str, Any]:
    """V's V-A/V-B criterion unchanged, over the reproduced adamw cells (main Class-A + heldout Class-A)."""
    per_reader: Dict[str, Any] = {}
    flat = []
    for reader in READERS:
        entries = []
        for rep in DECISIVE:
            a = _m(res["readers"][reader][rep]["main"]["pops"], "test_classA", "raw_acc")
            h = _m(res["readers"][reader][rep]["heldout"]["pops"], "test_heldout_classA", "raw_acc")
            entries.append((rep, a, h))
            flat.append((reader, rep, a, h))
        best = max((e for e in entries if e[1] is not None), key=lambda e: e[1])
        rep, a, h = best
        bayes = u_base["models"][PRIMARY_MODEL]["bayes"]["classA"][rep]
        per_reader[reader] = {"best_rep": rep, "classA": a, "heldout_classA": h, "bayes": bayes,
                              "bayes_minus_reader": (bayes - a) if (bayes is not None and a is not None) else None,
                              "per_horizon": {e[0]: {"classA": e[1], "heldout_classA": e[2]} for e in entries}}
    flat = [f for f in flat if f[2] is not None]
    reader, rep, a, h = max(flat, key=lambda f: f[2])
    bayes = u_base["models"][PRIMARY_MODEL]["bayes"]["classA"][rep]
    vA = bool(a >= RECOVER_BAR and h is not None and h >= HELDOUT_MATERIAL)
    vB = bool(a <= HELDOUT_WEAK and h is not None and h <= HELDOUT_WEAK
              and bayes is not None and (bayes - a) >= 0.30)
    verdict = ("V-A_sequence_accessible" if vA else
               "V-B_sequence_limited" if vB else "V-partial/inconclusive")
    return {"verdict": verdict, "best_reader": reader, "best_rep": rep, "classA": a,
            "heldout_classA": h, "bayes": bayes, "conditions": {"V-A": vA, "V-B": vB},
            "per_reader": per_reader}


def ad_bridge(res: Dict[str, Any], ad_raw_path: str) -> Dict[str, Any]:
    """Reproduced adamw metrics vs AD's retained adamw metrics (consistency bridge)."""
    with open(ad_raw_path) as f:
        parts = json.load(f)["parts"]
    by = {(p["seed"], p["reader"], p["rep"], p["variant"]): p for p in parts}
    mx_train = mx_ca = mx_ho = 0.0
    worst = None
    for reader in READERS:
        for rep in REP_LABELS:
            for s in SEEDS:
                ref = by[(s, reader, rep, "main")]["fits"]["adamw"]
                got_train = res["readers"][reader][rep]["main"]["pops"]["train"]["raw_acc"]["per_seed"][SEEDS.index(s)]
                d = abs(got_train - ref["best_val"]["train_acc"])
                if d > mx_train:
                    mx_train, worst = d, f"{reader}|{rep}|{s}|train"
                ref_ca = ref["best_val"]["acc"][f"{PRIMARY_MODEL}:classA"]
                got_ca = res["readers"][reader][rep]["main"]["pops"]["test_classA"]["raw_acc"]["per_seed"][SEEDS.index(s)]
                mx_ca = max(mx_ca, abs(got_ca - ref_ca))
                refho = by[(s, reader, rep, "heldout")]["fits"]["adamw"]["best_val"]["heldout"][f"{PRIMARY_MODEL}:classA"]
                gotho = res["readers"][reader][rep]["heldout"]["pops"]["test_heldout_classA"]["raw_acc"]["per_seed"][SEEDS.index(s)]
                mx_ho = max(mx_ho, abs(gotho - refho))
    return {"max_abs_diff_train_acc": mx_train, "max_abs_diff_classA": mx_ca,
            "max_abs_diff_heldout_classA": mx_ho, "worst": worst,
            "note": ("AE reproduces AD's exact adamw fits (same trainer/source/seeds/environment); the "
                     "bridge should be ~0. Reported, never gating.")}


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------

def _collect(data: Dict[str, Any], jobs: int, readers: List[str], reps: Optional[List[str]] = None
             ) -> List[Dict[str, Any]]:
    npz_path = data["path"]
    arch = av.architecture()
    reps = reps or REP_LABELS
    specs = [(s, R, rep, v) for s in SEEDS for R in readers for rep in reps
             for v in ("main", "heldout")]
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
                print(f"  [{done}/{len(specs)}] adamw fits reproduced", flush=True)
    print(f"collection wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    parts.sort(key=lambda p: (p["reader"], p["seed"], p["rep"], p["variant"]))
    return parts


def assemble(data: Dict[str, Any], parts: List[Dict[str, Any]], u_base: Dict[str, Any],
             ad_meta: Dict[str, Any]) -> Dict[str, Any]:
    arch = av.architecture()
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "readers": READERS,
        "target": "y_true", "n_classes": NCLASS, "reps": REP_LABELS, "decisive": DECISIVE,
        "condition": {"name": COND_NAME, **{k: COND[k] for k in ("opt", "init")}},
        "reused_retained_states": False,
        "reproduction": ("AD retained metrics/trajectories but no state dictionaries; AE reproduces the "
                         "exact AD adamw fits with AD's trainer and V's source arrays."),
        "lr": ad.LR, "weight_decay": ad.WD, "updates": UPDATES, "eval_every": EVAL_EVERY,
        "held_out_b": list(HELD_OUT_BS), "populations": POP_KEYS, "chance": CHANCE,
        "source_identity": data["meta"].get("source"), "class_counts": data["meta"].get("class_counts"),
        "target_derivation": data["meta"].get("target_derivation"),
        "disjointness": data["meta"].get("disjointness"),
        "checkpoints": data["meta"].get("models"), "n": data["meta"].get("n"),
        "thresholds": {"material_gain": MATERIAL_GAIN, "recover_bar": RECOVER_BAR,
                       "heldout_material": HELDOUT_MATERIAL, "heldout_weak": HELDOUT_WEAK},
        "statistical_discipline": (
            "Five frozen E checkpoints are the independent units. The two readers are analysis-only, fit "
            "on V's ordinary-ADD pool (cap 1,500/seed; held-out variant excludes b in {0,5}), tuned on "
            "V's canonical VAL (cap 800/seed), evaluated once on V's untouched canonical TEST at the "
            "best-on-VAL state. Selection is VAL-only; TEST never selects. Majority-class baselines are "
            "derived from the corresponding training labels only. Classical accuracy thresholds are "
            "unchanged; interpretation is descriptive and baseline-aware.")}}
    res["architecture"] = arch
    res["readers"] = aggregate(parts)
    res["per_seed"] = parts
    res["primary_decision"] = classify(res, parts)
    res["accessibility"] = accessibility(res, u_base)
    res["ad_bridge"] = ad_bridge(res, data.get("ad_raw", "experiments/experiment_ad_grid_raw.json"))
    return res


def build(data: Dict[str, Any], u_base: Dict[str, Any], ad_meta: Dict[str, Any], jobs: int,
          readers: Optional[List[str]] = None, reps: Optional[List[str]] = None) -> Dict[str, Any]:
    return assemble(data, _collect(data, jobs, readers or READERS, reps), u_base, ad_meta)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(res, out_dir, prefix="experiment_ae") -> List[str]:
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
    d = res["primary_decision"]
    cmap = {"GRU": "tab:blue", "TCN": "tab:orange"}

    # 1 fit vs transfer scatter (train accuracy vs Class-A test accuracy)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for reader in READERS:
        for rep, mk in (("L4", "o"), ("Lfull", "s")):
            x = _m(R[reader][rep]["main"]["pops"], "train", "raw_acc")
            y = _m(R[reader][rep]["main"]["pops"], "test_classA", "raw_acc")
            ax.scatter(nz(x), nz(y), color=cmap[reader], marker=mk, s=80, zorder=3)
            ax.annotate(f"{reader[0]}/{rep}", (nz(x), nz(y)), fontsize=6, xytext=(4, 4),
                        textcoords="offset points")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 V-A bar")
    ax.axhline(HELDOUT_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("training accuracy (fit pool)"); ax.set_ylabel("Class-A TEST accuracy (main)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("AE: fit vs transfer (adamw)")
    save(fig, "fit_vs_transfer")

    # 2 train-to-test and seen-to-heldout gaps
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labs, gaps = [], []
    for reader in READERS:
        for rep in DECISIVE:
            labs.append(f"{reader}\n{rep}")
            gaps.append(nz(d["cells"][f"{reader}|{rep}"]["train_to_test_gap"]))
    ax.bar(np.arange(len(labs)), gaps, color=["tab:blue" if l.startswith("GRU") else "tab:orange"
                                              for l in labs])
    ax.axhline(0, c="gray", lw=0.8)
    ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("train accuracy − Class-A TEST accuracy")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("AE: train-to-test gap (adamw)")
    save(fig, "train_to_test_gap")

    # 3 seen vs held-out Class-A (heldout variant) with majority baselines
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labs, seen, hold, hbase = [], [], [], []
    for reader in READERS:
        for rep in DECISIVE:
            labs.append(f"{reader}\n{rep}")
            seen.append(nz(_m(R[reader][rep]["heldout"]["pops"], "test_seen_classA", "raw_acc")))
            hold.append(nz(_m(R[reader][rep]["heldout"]["pops"], "test_heldout_classA", "raw_acc")))
            hbase.append(nz(_m(R[reader][rep]["heldout"]["pops"], "test_heldout_classA",
                               "majority_baseline_acc")))
    xi = np.arange(len(labs))
    ax.bar(xi - 0.27, seen, 0.27, label="seen operands", color="tab:gray")
    ax.bar(xi, hold, 0.27, label="held-out operands", color="tab:purple")
    ax.plot(xi + 0.27, hbase, "k^--", ms=6, label="held-out majority baseline")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance 0.10")
    ax.set_xticks(xi); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("Class-A accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AE: held-out vs seen operands (heldout variant, adamw)")
    save(fig, "heldout_vs_seen")

    # 4 per-seed held-out accuracy vs baseline
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    xi = np.arange(len(SEEDS))
    for reader in READERS:
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            ys = R[reader][rep]["heldout"]["pops"]["test_heldout_classA"]["raw_acc"]["per_seed"]
            ax.plot(xi, [nz(v) for v in ys], marker="o", ls=ls, color=cmap[reader],
                    alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep}")
        base = nz(_m(R[reader]["L4"]["heldout"]["pops"], "test_heldout_classA", "majority_baseline_acc"))
        ax.axhline(base, ls=":", color=cmap[reader], alpha=0.7)
    ax.set_xticks(xi); ax.set_xticklabels([str(s) for s in SEEDS]); ax.set_xlabel("E seed")
    ax.set_ylabel("held-out Class-A accuracy"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AE: per-seed held-out Class-A accuracy (dotted = majority baseline)")
    save(fig, "per_seed_heldout")

    # 5 per-pair held-out accuracy heatmap (Class-A), per reader/rep
    for reader in READERS:
        for rep in DECISIVE:
            M = np.array([[nz(R[reader][rep]["heldout"]["per_pair_heldout"][f"{a}_{b}"]["mean"])
                           for b in HELD_OUT_BS] for a in range(10)], float)
            fig, ax = plt.subplots(figsize=(5.2, 5.6))
            im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(0.4, float(np.nanmax(M)) if np.isfinite(M).any() else 0.4))
            ax.set_xticks([0, 1]); ax.set_xticklabels([f"b={b}" for b in HELD_OUT_BS])
            ax.set_yticks(range(10)); ax.set_yticklabels([f"a={a}" for a in range(10)])
            for i in range(10):
                for j in range(len(HELD_OUT_BS)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"AE: held-out Class-A accuracy by pair ({reader}/{rep})")
            save(fig, f"per_pair_{reader}_{rep}")

    # 6 confusion (Class-A, TCN/Lfull, heldout variant)
    conf = np.array(R["TCN"]["Lfull"]["heldout"]["confusion_classA"], float)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax)
    ax.set_title("AE: Class-A confusion (TCN/Lfull, heldout variant, pooled seeds)")
    save(fig, "confusion_classA")

    # 7 balanced vs raw accuracy (Class-A and held-out Class-A)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labs, raw, bal = [], [], []
    for reader in READERS:
        for rep in DECISIVE:
            labs.append(f"{reader}\n{rep}")
            raw.append(nz(_m(R[reader][rep]["main"]["pops"], "test_classA", "raw_acc")))
            bal.append(nz(_m(R[reader][rep]["main"]["pops"], "test_classA", "balanced_acc")))
    xi = np.arange(len(labs))
    ax.bar(xi - 0.2, raw, 0.4, label="raw", color="tab:blue")
    ax.bar(xi + 0.2, bal, 0.4, label="balanced", color="tab:green")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance 0.10")
    ax.set_xticks(xi); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("Class-A accuracy (main)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AE: raw vs balanced Class-A accuracy (adamw)")
    save(fig, "balanced_vs_raw")

    # 8 per-class recall on held-out Class-A
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    w = 0.2
    for i, (reader, rep) in enumerate([("GRU", "L4"), ("GRU", "Lfull"), ("TCN", "L4"), ("TCN", "Lfull")]):
        recalls = R[reader][rep]["heldout"]["pops"]["test_heldout_classA"]["per_class_recall"]
        ax.bar(np.arange(10) + (i - 1.5) * w, [nz(r["mean"]) for r in recalls], w,
               label=f"{reader} {rep}", alpha=0.85)
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(range(10)); ax.set_xticklabels([str(c) for c in range(10)])
    ax.set_xlabel("target class (y_true)"); ax.set_ylabel("recall (held-out Class-A)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AE: per-class recall on held-out operands (adamw)")
    save(fig, "per_class_recall_heldout")

    # 9 decision summary
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    cats = ["fit\n(train acc)", "fit baseline\n(train)", "Class-A\n(main)", "held-out\nClass-A",
            "held-out\nbaseline", "seen\nClass-A"]
    vals = []
    worst = None
    for reader in READERS:
        for rep in DECISIVE:
            c = d["cells"][f"{reader}|{rep}"]
            if reader == "TCN" and rep == "Lfull":
                worst = c
    c = worst or list(d["cells"].values())[0]
    vals = [nz(c["train_acc"]), nz(c["train_baseline"]), nz(c["test_classA_acc"]),
            nz(c["heldout_classA_acc"]), nz(c["heldout_classA_baseline"]), nz(c["seen_classA_acc"])]
    ax.bar(np.arange(len(cats)), vals,
           color=["tab:orange", "tab:gray", "tab:blue", "tab:purple", "tab:gray", "tab:olive"])
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 V-A")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance 0.10")
    ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels(cats, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"AE: decision summary ({d['verdict']}, shown TCN/Lfull)")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AE analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--ad-meta", default="experiments/experiment_ad_qualified_access_meta.json")
    ap.add_argument("--ad-raw", default="experiments/experiment_ad_grid_raw.json")
    ap.add_argument("--u-results", default="experiments/experiment_u_results.json")
    ap.add_argument("--output", default="experiments/experiment_ae_results.json")
    ap.add_argument("--raw", default="experiments/experiment_ae_grid_raw.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_ae")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--readers", nargs="*", default=None)
    ap.add_argument("--reps", nargs="*", default=None)
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--use-raw", action="store_true")
    args = ap.parse_args()
    data = ad.load(args.npz)
    data["path"] = args.npz
    data["ad_raw"] = args.ad_raw
    ad_meta = json.load(open(args.ad_meta))
    problems = ad._validate(data["meta"])
    if problems:
        print(json.dumps({"validation_problems": problems}, indent=2))
        raise SystemExit("AE source invalid")
    u_base = av.load_u_baseline(args.u_results)
    if args.use_raw and os.path.exists(args.raw):
        with open(args.raw) as f:
            parts = json.load(f)["parts"]
    else:
        parts = _collect(data, args.jobs, args.readers or READERS, args.reps)
        os.makedirs(os.path.dirname(args.raw), exist_ok=True)
        with open(args.raw, "w") as f:
            json.dump({"parts": parts}, f)
        if args.collect_only:
            print(f"Wrote raw {args.raw} ({len(parts)} per-seed adamw records)")
            return
    t0 = time.perf_counter()
    res = assemble(data, parts, u_base, ad_meta)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"fit-transfer verdict: {d['verdict']}")
    for k, v in d["cells"].items():
        print(f"  {k}: train={v['train_acc']:.3f} (base {v['train_baseline']:.3f}) "
              f"classA={v['test_classA_acc']:.3f} held-out={v['heldout_classA_acc']} "
              f"(base {v['heldout_classA_baseline']}) gain={v['transfer_gain_over_baseline']} "
              f"pos_seeds={v['positive_transfer_seeds']} pairs>chance={v['heldout_pairs_above_chance']} "
              f"-> {v['category']}")
    print(f"accessibility: {res['accessibility']['verdict']}")
    print(f"AD bridge: {res['ad_bridge']['max_abs_diff_train_acc']:.2e} "
          f"{res['ad_bridge']['max_abs_diff_classA']:.2e} {res['ad_bridge']['max_abs_diff_heldout_classA']:.2e}")
    paths = make_figures(res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
