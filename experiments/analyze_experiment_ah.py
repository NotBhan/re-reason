#!/usr/bin/env python3
"""Experiment AH — analysis: bounded-reader capacity / inductive-bias sensitivity at V's scale.

AG showed that the training-pool construction changes the readers' fit but not the held-out
operand-pair transfer; AF removed the task-side explanation. AH asks the reader-side question at
V's scale:

    Holding V's arrays, held-out protocol (`b in {0,5}` excluded from every fit), optimizer recipe
    (AB-qualified AdamW, lr 3e-3, wd 1e-2, 200 full-batch updates, best-on-VAL), reader families,
    decisive histories (L4, Lfull) and seeds fixed, is the fit–transfer gap sensitive to reader
    **capacity**?

This is an analysis-reader training experiment, NOT a RE:Reason training or representation
experiment. The pre-registered width ladder was fixed in `run_experiment_ah.py` before any fit:

    small (42/26), medium_small (60/37), baseline (85/52), medium_large (120/74), large (170/104)

(`gru_hidden` / `tcn_channels`; every level within 4.0x the baseline level's exact parameter
count). The `baseline` level reproduces V's deterministic architecture selection, and its fit-seed
offset is 0 — AE's exact heldout-variant init seed. Grid: 2 readers x 5 widths x 2 histories x
5 seeds = 100 fits, each on the seed's native eligible pool (`trainord` rows with `b not in {0,5}`,
no resampling).

Hypotheses (evaluated descriptively; categories, not new acceptance thresholds)
-------------------------------------------------------------------------------
    H1 capacity-associated generalization : some width's held-out Class-A *balanced* accuracy
        improves by >= +0.15 over the baseline level in >= 4/5 seeds AND reaches >= 0.30.
    H2 fit without transfer : training fit is present at some level (mean train acc >= 0.50 — X's
        "training accuracy still low" boundary reused as the fit bar — or a >= +0.15 gain over
        the baseline level in >= 4/5 seeds) while every fitting level's held-out balanced
        accuracy stays <= 0.25 (chance 0.10).
    H3 capacity-insensitive fitting failure : no level in the ladder fits the training pool.
    H4 mixed / inconclusive : everything else (borderline / seed-dependent; continuous quantities
        are reported and no verdict is forced).
V's V-A/V-B accessibility criterion is evaluated separately and unchanged.

Measurements (per fit, at the best-on-VAL state; `ae._summarize` for every population)
---------------------------------------------------------------------------------------
    train_fit, val, test_classA, test_all_add, heldout_classA (b in {0,5}), seen_classA,
    heldout_all, seen_all — raw + balanced accuracy, per-class recall, a majority-class baseline
    derived exclusively from the fitted pool's labels, and the population label frequencies;
    per-pair accuracy for the 20 held-out pairs. Aggregates are mean ± SD across the five E seeds;
    paired per-seed deltas are computed against the `baseline` width.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_ah.py --jobs 16
    PYTHONPATH=src python experiments/analyze_experiment_ah.py --jobs 16 --use-raw
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

import analyze_experiment_ab as ab  # noqa: E402  (AB's exact optimizer/init/trainer pieces)
import analyze_experiment_ad as ad  # noqa: E402  (AD helpers: source loading, features, conditions)
import analyze_experiment_ae as ae  # noqa: E402  (AE's baseline-aware population summarizer)
import analyze_experiment_s as as_  # noqa: E402  (mean_sd / class_weights)
import analyze_experiment_u as au  # noqa: E402  (history construction / populations / pairs)
import analyze_experiment_v as av  # noqa: E402  (V's exact readers / seeds / U baseline loader)

PRIMARY_MODEL = av.PRIMARY_MODEL
SEEDS = av.SEEDS
READERS = ["GRU", "TCN"]
HISTORIES = ["L4", "Lfull"]
REPS = au.REPS
TARGET = "y_true"
NCLASS = 10
CHANCE = 1.0 / NCLASS
HELD_OUT_B = (0, 5)
PAIRS_HELDOUT = [(a, b) for a in range(10) for b in HELD_OUT_B]

# Pre-registered capacity ladder, fixed in `run_experiment_ah.py` before any fit:
# (level_key, level_name, gru_hidden, tcn_channels).
LEVELS: List[Tuple[str, str, int, int]] = [
    ("small", "Small", 42, 26),
    ("medium_small", "Medium-small", 60, 37),
    ("baseline", "Baseline", 85, 52),
    ("medium_large", "Medium-large", 120, 74),
    ("large", "Large", 170, 104),
]
LEVEL_KEYS = [l[0] for l in LEVELS]
LEVEL_IDX = {l[0]: i for i, l in enumerate(LEVELS)}
LEVEL_BY_KEY = {l[0]: l for l in LEVELS}
BASELINE_KEY = "baseline"
BASELINE_IDX = LEVEL_IDX[BASELINE_KEY]
KERNEL = 3
DILATIONS = [1, 2, 4, 8]

COND_NAME = "adamw"
COND = ad.COND_BY_NAME[COND_NAME]
UPDATES = ad.UPDATES
EVAL_EVERY = ad.EVAL_EVERY

# Declared conventions (reused; no new acceptance thresholds)
MATERIAL_GAIN = 0.15          # U/V paired material-gain convention
TRAIN_FIT_BAR = 0.50          # X's "training accuracy still low" boundary, used here as the fit bar
HELDOUT_WEAK = 0.25           # V's weak-accuracy bar
HELDOUT_MATERIAL = 0.30       # V's held-out-material bar
RECOVER_BAR = 0.50            # V's V-A Class-A bar
BAYES_GAP_BAR = 0.30          # V's V-B Bayes-gap bar
CONSISTENT_SEEDS = 4          # >= 4/5 seeds

POP_KEYS = ["train_fit", "val", "test_classA", "test_all_add",
            "heldout_classA", "seen_classA", "heldout_all", "seen_all"]


def _ms(vals):
    return as_.mean_sd(vals)


def _fmt(v, spec: str = ".3f") -> str:
    return "n/a" if v is None else format(v, spec)


# ---------------------------------------------------------------------------
# fit seeds / level-specific readers (AB's exact protocol at each ladder width)
# ---------------------------------------------------------------------------

def _fit_seed(seed: int, reader: str, rep: str, level_key: str) -> int:
    """Pre-registered model-init/optimizer seed for one ladder cell.

    V's exact heldout-variant reader seed (the optimizer condition is not part of it) plus a
    per-level offset of `(level_idx - 2) * 100000`: the `baseline` level (level_idx == 2) uses
    offset 0 and therefore matches AE's exact init seed. Offsets reuse the per-seed stride, so
    some (seed, level) cells share an init seed across the grid; within any single level the five
    seeds remain distinct, and every cell is deterministic.
    """
    return (av.reader_seed(seed, reader, TARGET, rep, "heldout")
            + (LEVEL_IDX[level_key] - BASELINE_IDX) * 100000)


def _build_level_model(reader: str, level: Tuple[str, str, int, int]):
    """Level-specific reader: V's exact classes, width from the ladder, nothing else changed."""
    _, _, gru_hidden, tcn_channels = level
    if reader == "GRU":
        return av.GRUReader(gru_hidden, nclass=NCLASS)
    if reader == "TCN":
        return av.TCNReader(tcn_channels, nclass=NCLASS, kernel=KERNEL, dilations=DILATIONS)
    raise ValueError(reader)


def _param_count(reader: str, level: Tuple[str, str, int, int]) -> int:
    """Exact trainable-parameter count of the level's reader, taken from the implementation."""
    return int(sum(p.numel() for p in _build_level_model(reader, level).parameters()
                   if p.requires_grad))


def _init_hash(reader: str, level: Tuple[str, str, int, int], seed: int) -> str:
    """Deterministic initial-parameter fingerprint for one (reader, level, seed) triple."""
    import torch
    torch.manual_seed(seed)
    model = _build_level_model(reader, level)
    ab.apply_init(model, COND["init"])
    return ad._state_hash(model)


def _train_ladder(reader: str, level: Tuple[str, str, int, int], Xtr, ytr, Xva, yva,
                  seed: int) -> Dict[str, Any]:
    """AB's exact `adamw` fit protocol with a level-specific width.

    Identical ops/order to `ab.train_ab` (same seed -> same init; full-batch AdamW lr 3e-3 /
    wd 1e-2; class-balanced CE; evaluate every 10 updates; best-on-VAL selection with a cloned
    state dict), so the `baseline` level reproduces AE's stored heldout-variant fits.
    """
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    Xn = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    Xv = torch.tensor(np.asarray(Xva, dtype=np.float32))
    model = _build_level_model(reader, level)
    ab.apply_init(model, COND["init"])
    opt = ab.make_optimizer(model, COND)
    yn = torch.tensor(ytr, dtype=torch.long)
    yv = torch.tensor(yva, dtype=torch.long)
    wc = torch.tensor(as_.class_weights(ytr, NCLASS), dtype=torch.float32)
    best_val: Tuple[float, Any, int] = (-1.0, None, -1)
    events: List[Dict[str, float]] = []
    for ep in range(UPDATES):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xn), yn, weight=wc)
        loss.backward()
        opt.step()
        if (ep + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vlog = model(Xv)
                vloss = float(F.cross_entropy(vlog, yv, weight=wc))
                vacc = float((vlog.argmax(1) == yv).float().mean())
                trlog = model(Xn)
                trloss = float(F.cross_entropy(trlog, yn, weight=wc))
                tracc = float((trlog.argmax(1) == yn).float().mean())
            events.append({"update": ep + 1, "train_loss": trloss, "train_acc": tracc,
                           "val_loss": vloss, "val_acc": vacc})
            st = {k: v.clone() for k, v in model.state_dict().items()}
            if vacc > best_val[0]:
                best_val = (vacc, st, ep + 1)
    if best_val[1] is None:  # unreachable under 200 updates / eval every 10; kept for safety
        best_val = (events[-1]["val_acc"], {k: v.clone() for k, v in model.state_dict().items()},
                    UPDATES)
    finite = bool(all(np.isfinite([e["train_loss"], e["train_acc"], e["val_loss"], e["val_acc"]]).all()
                      for e in events))
    return {"best_val": best_val, "events": events, "finite": finite,
            "train_acc_final": events[-1]["train_acc"], "val_acc_final": events[-1]["val_acc"]}


def _predict(reader: str, level: Tuple[str, str, int, int], state, X) -> np.ndarray:
    import torch
    model = _build_level_model(reader, level)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(np.asarray(X, dtype=np.float32))).argmax(1).numpy()


# ---------------------------------------------------------------------------
# worker: one (seed, reader, history, level) task
# ---------------------------------------------------------------------------

def _task(spec: Tuple[int, str, str, str]) -> Dict[str, Any]:
    seed, reader, rep, level_key = spec
    ad._G["feat"] = {}
    z = ad._G["z"]
    level = LEVEL_BY_KEY[level_key]
    L = dict(REPS)[rep]

    tr = np.nonzero(z[f"{PRIMARY_MODEL}__trainord__seed"] == seed)[0]
    held_tr = au._pairs(z, PRIMARY_MODEL, "trainord", tr)
    keep = ~held_tr                      # native eligible pool: b not in {0,5}, no resampling
    elig = tr[keep]
    Xtr = ad._feat("trainord", seed, L)[keep]
    ytr = z[f"{PRIMARY_MODEL}__trainord__{TARGET}"][elig].astype(int)

    va = np.nonzero(z[f"{PRIMARY_MODEL}__valord__seed"] == seed)[0]
    Xva = ad._feat("valord", seed, L)
    yva = z[f"{PRIMARY_MODEL}__valord__{TARGET}"][va].astype(int)

    te_sel = z[f"{PRIMARY_MODEL}__test__seed"] == seed
    te_idx = np.nonzero(te_sel)[0]
    held_te = au._pairs(z, PRIMARY_MODEL, "test", te_idx)
    masks = au._pop_masks(z, PRIMARY_MODEL, seed)
    ClsA = masks["classA"][te_sel]
    allA = masks["all_add"][te_sel]
    yte = z[f"{PRIMARY_MODEL}__test__{TARGET}"][te_sel].astype(int)
    oa_te = z[f"{PRIMARY_MODEL}__test__operand_a"][te_sel].astype(int)
    ob_te = z[f"{PRIMARY_MODEL}__test__operand_b"][te_sel].astype(int)

    fit_seed = _fit_seed(seed, reader, rep, level_key)
    fit = _train_ladder(reader, level, Xtr, ytr, Xva, yva, fit_seed)
    state = fit["best_val"][1]
    Xte = ad._feat("test", seed, L)
    pred_fit = _predict(reader, level, state, Xtr)
    pred_va = _predict(reader, level, state, Xva)
    pred_te = _predict(reader, level, state, Xte)

    pops: Dict[str, Any] = {
        "train_fit": ae._summarize(pred_fit, ytr, ytr),
        "val": ae._summarize(pred_va, yva, ytr),
        "test_classA": ae._summarize(pred_te[ClsA], yte[ClsA], ytr),
        "test_all_add": ae._summarize(pred_te[allA], yte[allA], ytr),
        "heldout_classA": ae._summarize(pred_te[ClsA & held_te], yte[ClsA & held_te], ytr),
        "seen_classA": ae._summarize(pred_te[ClsA & ~held_te], yte[ClsA & ~held_te], ytr),
        "heldout_all": ae._summarize(pred_te[allA & held_te], yte[allA & held_te], ytr),
        "seen_all": ae._summarize(pred_te[allA & ~held_te], yte[allA & ~held_te], ytr),
    }
    per_pair: Dict[str, Any] = {}
    for a, b in PAIRS_HELDOUT:
        m = ClsA & (oa_te == a) & (ob_te == b)
        per_pair[f"{a}_{b}"] = ({"n": int(m.sum()),
                                 "acc": float((pred_te[m] == yte[m]).mean())} if m.any() else None)

    return {"seed": seed, "reader": reader, "rep": rep, "level_key": level_key,
            "level_name": level[1], "gru_hidden": level[2], "tcn_channels": level[3],
            "init_hash": _init_hash(reader, level, fit_seed),
            "params": _param_count(reader, level), "n_train": int(keep.sum()),
            "best_val_acc": float(fit["best_val"][0]), "best_val_update": int(fit["best_val"][2]),
            "finite": bool(fit["finite"]), "pops": pops, "per_pair": per_pair}


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _agg_pop(blocks: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    blocks = [b for b in blocks if b is not None]
    if not blocks:
        return {}
    out: Dict[str, Any] = {}
    for k in blocks[0]:
        vals = [b[k] for b in blocks]
        if isinstance(vals[0], list):
            out[k] = [_ms([v[i] for v in vals]) for i in range(len(vals[0]))]
        else:
            out[k] = _ms(vals)
    return out


def aggregate(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean ± SD across the five seeds for every (reader, history, level) population block."""
    by = {(p["reader"], p["seed"], p["rep"], p["level_key"]): p for p in parts}
    out: Dict[str, Any] = {}
    for reader in READERS:
        for rep in HISTORIES:
            for level_key in LEVEL_KEYS:
                cells = [by.get((reader, s, rep, level_key)) for s in SEEDS]
                cells = [c for c in cells if c is not None]
                if not cells:
                    continue
                out.setdefault(reader, {}).setdefault(rep, {})[level_key] = {
                    "level_name": cells[0]["level_name"],
                    "gru_hidden": cells[0]["gru_hidden"], "tcn_channels": cells[0]["tcn_channels"],
                    "params": cells[0]["params"], "n_train": _ms([c["n_train"] for c in cells]),
                    "best_val_acc": _ms([c["best_val_acc"] for c in cells]),
                    "best_val_update": _ms([c["best_val_update"] for c in cells]),
                    "all_finite": bool(all(c["finite"] for c in cells)),
                    "init_hashes": [c["init_hash"] for c in cells],
                    "pops": {k: _agg_pop([c["pops"].get(k) for c in cells]) for k in POP_KEYS},
                    "per_pair": {f"{a}_{b}": _ms([(c["per_pair"].get(f"{a}_{b}") or {}).get("acc")
                                                  for c in cells]) for a, b in PAIRS_HELDOUT},
                }
    return out


def paired_deltas(res: Dict[str, Any]) -> Dict[str, Any]:
    """Per-seed paired deltas (level - baseline) for held-out Class-A balanced and train fit."""
    out: Dict[str, Any] = {}
    for reader in READERS:
        for rep in HISTORIES:
            group = res["readers"].get(reader, {}).get(rep, {})
            base = group.get(BASELINE_KEY)
            levels: Dict[str, Any] = {}
            if base is not None:
                b_bal = base["pops"]["heldout_classA"]["balanced_acc"]["per_seed"]
                b_fit = base["pops"]["train_fit"]["raw_acc"]["per_seed"]
                for level_key in LEVEL_KEYS:
                    node = group.get(level_key)
                    if node is None:
                        continue
                    l_bal = node["pops"]["heldout_classA"]["balanced_acc"]["per_seed"]
                    l_fit = node["pops"]["train_fit"]["raw_acc"]["per_seed"]
                    d_bal = [None if (a is None or b is None) else float(a) - float(b)
                             for a, b in zip(l_bal, b_bal)]
                    d_fit = [None if (a is None or b is None) else float(a) - float(b)
                             for a, b in zip(l_fit, b_fit)]
                    levels[level_key] = {
                        "delta_heldout_classA_bal": _ms(d_bal),
                        "delta_train_fit_raw": _ms(d_fit),
                        "seeds_bal_material_gain": int(sum(1 for d in d_bal
                                                           if d is not None and d >= MATERIAL_GAIN)),
                        "seeds_fit_material_gain": int(sum(1 for d in d_fit
                                                           if d is not None and d >= MATERIAL_GAIN)),
                    }
            out.setdefault(reader, {})[rep] = levels
    return out


# ---------------------------------------------------------------------------
# decision (descriptive)
# ---------------------------------------------------------------------------

def classify(res: Dict[str, Any]) -> Dict[str, Any]:
    """H1–H4 evaluation over all (reader, history, level) cells vs the baseline width."""
    R = res["readers"]
    P = res.get("paired_deltas", {})
    cells: Dict[str, Any] = {}
    baseline_fit: Dict[str, Any] = {}
    for reader in READERS:
        for rep in HISTORIES:
            group = R.get(reader, {}).get(rep, {})
            base = group.get(BASELINE_KEY)
            if base is None:
                continue
            base_bal = base["pops"]["heldout_classA"]["balanced_acc"]["mean"]
            base_train = base["pops"]["train_fit"]["raw_acc"]["mean"]
            baseline_fit[f"{reader}|{rep}"] = bool(base_train is not None
                                                   and float(base_train) >= TRAIN_FIT_BAR)
            for level_key in LEVEL_KEYS:
                if level_key == BASELINE_KEY:
                    continue
                node = group.get(level_key)
                pd = P.get(reader, {}).get(rep, {}).get(level_key)
                if node is None or pd is None:
                    continue
                bal = node["pops"]["heldout_classA"]["balanced_acc"]["mean"]
                train = node["pops"]["train_fit"]["raw_acc"]["mean"]
                if bal is None or train is None:
                    continue
                cells[f"{reader}|{rep}|{level_key}"] = {
                    "level_name": node["level_name"], "params": node["params"],
                    "heldout_classA_bal": float(bal),
                    "heldout_classA_bal_baseline": (None if base_bal is None else float(base_bal)),
                    "mean_delta_bal": pd["delta_heldout_classA_bal"]["mean"],
                    "seeds_bal_material_gain": pd["seeds_bal_material_gain"],
                    "train_fit": float(train),
                    "mean_delta_fit": pd["delta_train_fit_raw"]["mean"],
                    "seeds_fit_material_gain": pd["seeds_fit_material_gain"],
                    "transfer_signal": bool(pd["seeds_bal_material_gain"] >= CONSISTENT_SEEDS
                                            and bal >= HELDOUT_MATERIAL),
                    "fit_present": bool(train >= TRAIN_FIT_BAR
                                        or pd["seeds_fit_material_gain"] >= CONSISTENT_SEEDS),
                    "heldout_weak": bool(bal <= HELDOUT_WEAK),
                }
    h1 = [k for k, v in cells.items() if v["transfer_signal"]]
    fit_records = [{"key": k, "heldout_weak": v["heldout_weak"]}
                   for k, v in cells.items() if v["fit_present"]]
    for reader in READERS:                       # the baseline level's own fit enters H2/H3 too
        for rep in HISTORIES:
            group = R.get(reader, {}).get(rep, {})
            base = group.get(BASELINE_KEY)
            if base is None:
                continue
            base_train = base["pops"]["train_fit"]["raw_acc"]["mean"]
            base_bal = base["pops"]["heldout_classA"]["balanced_acc"]["mean"]
            if base_train is not None and base_train >= TRAIN_FIT_BAR:
                fit_records.append({"key": f"{reader}|{rep}|{BASELINE_KEY}",
                                    "heldout_weak": bool(base_bal is not None
                                                         and base_bal <= HELDOUT_WEAK)})
    if h1:
        verdict = "H1_capacity_associated_generalization"
        framing = "AH-A (held-out transfer emerges at some width under this ladder)"
    elif fit_records and all(r["heldout_weak"] for r in fit_records):
        verdict = "H2_fit_without_transfer"
        framing = "AH-B (the fit-transfer gap is capacity-insensitive over this ladder)"
    elif not fit_records:
        verdict = "H3_capacity_insensitive_fitting_failure"
        framing = "AH-B (the fitting failure is capacity-insensitive over this ladder)"
    else:
        verdict = "H4_mixed_inconclusive"
        framing = "partial/inconclusive (borderline cells reported; no verdict forced)"
    return {"verdict": verdict, "framing": framing, "h1_cells": h1,
            "fit_cells": [r["key"] for r in fit_records],
            "baseline_fit": baseline_fit, "cells": cells,
            "thresholds": {"material_gain": MATERIAL_GAIN, "train_fit_bar": TRAIN_FIT_BAR,
                           "heldout_weak": HELDOUT_WEAK, "heldout_material": HELDOUT_MATERIAL,
                           "consistent_seeds": CONSISTENT_SEEDS},
            "rule_note": ("Descriptive categories, no new acceptance thresholds. H1 if some width's "
                          "held-out Class-A balanced accuracy gains >= +0.15 over the baseline "
                          "level in >= 4/5 seeds AND reaches >= 0.30; H2 if training fit is present "
                          "at some level (mean train acc >= 0.50, X's 'training accuracy still low' "
                          "boundary, or a >= +0.15 gain over baseline in >= 4/5 seeds) while every "
                          "fitting level's held-out balanced accuracy stays <= 0.25 (chance 0.10); "
                          "H3 if no level in the ladder fits the training pool; else H4 "
                          "mixed/inconclusive. H1 takes priority (a transfer signal at any width "
                          "falsifies capacity-insensitivity); H2/H3 map to the AH-B "
                          "capacity-insensitive gap. V's V-A/V-B criterion is separate and "
                          "unchanged.")}


def accessibility(res: Dict[str, Any], u_base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """V's unchanged V-A/V-B sequence-accessibility criterion over the ladder's best cell."""
    best: Optional[Dict[str, Any]] = None
    per_cell: List[Dict[str, Any]] = []
    for reader in READERS:
        for rep in HISTORIES:
            for level_key in LEVEL_KEYS:
                node = res["readers"].get(reader, {}).get(rep, {}).get(level_key)
                if node is None:
                    continue
                a = node["pops"]["test_classA"]["raw_acc"]["mean"]
                h = node["pops"]["heldout_classA"]["raw_acc"]["mean"]
                if a is None:
                    continue
                rec = {"reader": reader, "rep": rep, "level_key": level_key,
                       "classA": float(a), "heldout_classA": (None if h is None else float(h))}
                per_cell.append(rec)
                if best is None or rec["classA"] > best["classA"]:
                    best = rec
    bayes = None
    if best is not None and u_base is not None:
        try:
            bayes = u_base["models"][PRIMARY_MODEL]["bayes"]["classA"][best["rep"]]
        except (KeyError, TypeError):
            bayes = None
    gap = (float(bayes) - best["classA"]) if (bayes is not None and best is not None) else None
    vA = bool(best is not None and best["classA"] >= RECOVER_BAR
              and best["heldout_classA"] is not None and best["heldout_classA"] >= HELDOUT_MATERIAL)
    vB = bool(best is not None and best["classA"] <= HELDOUT_WEAK
              and best["heldout_classA"] is not None and best["heldout_classA"] <= HELDOUT_WEAK
              and (gap is None or gap >= BAYES_GAP_BAR))
    verdict = ("V-A_sequence_accessible" if vA else
               "V-B_sequence_limited" if vB else "V-partial/inconclusive")
    return {"verdict": verdict, "best": best, "bayes": (None if bayes is None else float(bayes)),
            "bayes_minus_reader": gap, "bayes_gap_evaluated": gap is not None,
            "conditions": {"V-A": vA, "V-B": vB}, "per_cell": per_cell,
            "note": ("V's V-A/V-B criterion is unchanged; the V-B Bayes-gap condition uses U's "
                     "Class-A Bayes ceiling when the U baseline is available (AH does not redefine "
                     "the criterion).")}


def ae_bridge(res: Dict[str, Any], ae_raw_path: str) -> Dict[str, Any]:
    """Baseline-level consistency bridge against AE's stored heldout-variant fits (never gating)."""
    if not os.path.exists(ae_raw_path):
        return {"available": False, "path": ae_raw_path,
                "note": "AE raw artifact not present; bridge skipped (never gating)."}
    with open(ae_raw_path) as f:
        parts = json.load(f)["parts"]
    by = {(p["seed"], p["reader"], p["rep"]): p for p in parts if p["variant"] == "heldout"}
    max_diff = {"train_fit": 0.0, "classA": 0.0, "heldout_classA": 0.0, "best_val_acc": 0.0}
    worst: Optional[str] = None
    for reader in READERS:
        for rep in HISTORIES:
            node = res["readers"].get(reader, {}).get(rep, {}).get(BASELINE_KEY)
            if node is None:
                continue
            for i, s in enumerate(SEEDS):
                ref = by.get((s, reader, rep))
                if ref is None:
                    continue
                pairs = (
                    ("train_fit", node["pops"]["train_fit"]["raw_acc"]["per_seed"][i],
                     ref["pops"]["train"]["raw_acc"]),
                    ("classA", node["pops"]["test_classA"]["raw_acc"]["per_seed"][i],
                     ref["pops"]["test_classA"]["raw_acc"]),
                    ("heldout_classA", node["pops"]["heldout_classA"]["raw_acc"]["per_seed"][i],
                     ref["pops"]["test_heldout_classA"]["raw_acc"]),
                    ("best_val_acc", node["best_val_acc"]["per_seed"][i], ref["best_val_acc"]),
                )
                for key, got, exp in pairs:
                    if got is None or exp is None:
                        continue
                    diff = abs(float(got) - float(exp))
                    if diff > max_diff[key]:
                        max_diff[key] = diff
                        worst = f"{reader}|{rep}|seed{s}|{key}"
    return {"available": True, "path": ae_raw_path,
            "max_abs_diff_train_fit": max_diff["train_fit"],
            "max_abs_diff_classA": max_diff["classA"],
            "max_abs_diff_heldout_classA": max_diff["heldout_classA"],
            "max_abs_diff_best_val_acc": max_diff["best_val_acc"], "worst": worst,
            "note": ("AH's baseline level (level_idx 2 -> seed offset 0) re-runs AE's exact "
                     "heldout-variant fit seed on the same eligible pool under the same recipe; "
                     "same-environment runs should agree to ~0. Reported, never gating.")}


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------

def _collect(data: Dict[str, Any], jobs: int, readers: List[str], histories: List[str],
             levels: List[str]) -> List[Dict[str, Any]]:
    npz_path = data["path"]
    arch = av.architecture()
    specs = [(s, R, rep, lk) for s in SEEDS for R in readers for rep in histories
             for lk in levels]
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
                print(f"  [{done}/{len(specs)}] capacity-ladder fits done", flush=True)
    print(f"collection wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    parts.sort(key=lambda p: (p["reader"], p["seed"], p["rep"], LEVEL_IDX[p["level_key"]]))
    return parts


def _check_meta(ah_meta: Dict[str, Any]) -> List[str]:
    """Verify the analysis grid against the AH pre-registration meta when it is available."""
    problems: List[str] = []
    if not ah_meta:
        return problems
    ladder = ah_meta.get("capacity_ladder", {}).get("levels")
    if ladder is not None:
        got = [(r.get("level_key"), r.get("level_name"), int(r.get("gru_hidden", -1)),
                int(r.get("tcn_channels", -1))) for r in ladder]
        if got != [tuple(l) for l in LEVELS]:
            problems.append(f"capacity-ladder meta does not match the pre-registered LEVELS: {got}")
    grid = ah_meta.get("grid", {})
    if grid and int(grid.get("n_fits", -1)) != len(READERS) * len(LEVELS) * len(HISTORIES) * len(SEEDS):
        problems.append(f"pre-registered grid mismatch: {grid}")
    proto = ah_meta.get("protocol_matches_v", {})
    if proto and (proto.get("readers") != READERS or proto.get("histories") != HISTORIES
                  or proto.get("seeds") != SEEDS):
        problems.append(f"protocol drift vs AH pre-registration: {proto}")
    return problems


def assemble(data: Dict[str, Any], parts: List[Dict[str, Any]],
             u_base: Optional[Dict[str, Any]], ah_meta: Dict[str, Any]) -> Dict[str, Any]:
    levels = [{"level_key": key, "level_name": name, "gru_hidden": gh, "tcn_channels": tc,
               "gru_params": _param_count("GRU", (key, name, gh, tc)),
               "tcn_params": _param_count("TCN", (key, name, gh, tc))}
              for key, name, gh, tc in LEVELS]
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": SEEDS, "readers": READERS, "histories": HISTORIES,
        "target": TARGET, "n_classes": NCLASS, "chance": CHANCE,
        "levels": [l["level_key"] for l in levels], "baseline_level": BASELINE_KEY,
        "total_fits": len(READERS) * len(LEVELS) * len(HISTORIES) * len(SEEDS),
        "train_pool": ("native eligible pool: trainord rows with operand_b not in {0,5}, "
                       "no resampling"),
        "held_out_b": list(HELD_OUT_B), "held_out_pairs": len(PAIRS_HELDOUT),
        "fit_seed_convention": ("av.reader_seed(seed, reader, 'y_true', rep, 'heldout') + "
                                "(level_idx - 2) * 100000; baseline offset 0 = AE's heldout-variant "
                                "seed; offsets reuse the per-seed stride so some (seed, level) cells "
                                "share an init seed while each level's five seeds stay distinct"),
        "optimizer": {"name": COND_NAME, **{k: COND[k] for k in ("opt", "init")}},
        "lr": ad.LR, "weight_decay": ad.WD, "updates": UPDATES, "eval_every": EVAL_EVERY,
        "source_identity": data["meta"].get("source"),
        "class_counts": data["meta"].get("class_counts"),
        "target_derivation": data["meta"].get("target_derivation"),
        "disjointness": data["meta"].get("disjointness"),
        "history_causality": data["meta"].get("history_causality"),
        "checkpoints": data["meta"].get("models"), "n": data["meta"].get("n"),
        "capacity_ladder_meta": (ah_meta.get("capacity_ladder") if ah_meta else None),
        "eligibility_support": (ah_meta.get("eligible_pool") if ah_meta else None),
        "thresholds": {"material_gain": MATERIAL_GAIN, "train_fit_bar": TRAIN_FIT_BAR,
                       "heldout_weak": HELDOUT_WEAK, "heldout_material": HELDOUT_MATERIAL,
                       "recover_bar": RECOVER_BAR, "bayes_gap_bar": BAYES_GAP_BAR,
                       "consistent_seeds": CONSISTENT_SEEDS},
        "statistical_discipline": (
            "Five frozen E checkpoints are the independent units. The two readers are analysis-only, "
            "fitted on each seed's native eligible pool (trainord rows with b not in {0,5}; no "
            "resampling) with the AB-qualified AdamW recipe, tuned on V's canonical VAL, evaluated "
            "once on the untouched canonical TEST at the best-on-VAL state. Selection is VAL-only; "
            "TEST never selects. Majority-class baselines are derived from the corresponding "
            "training labels only. The pre-registered ladder, histories and seeds are fixed; every "
            "width is reported, including non-fitting ones. V-A/V-B are unchanged.")}}
    res["levels"] = levels
    res["architecture"] = av.architecture()
    res["readers"] = aggregate(parts)
    res["per_seed"] = parts
    res["paired_deltas"] = paired_deltas(res)
    res["primary_decision"] = classify(res)
    res["accessibility"] = accessibility(res, u_base)
    res["ae_bridge"] = ae_bridge(res, data.get("ae_raw", "experiments/experiment_ae_grid_raw.json"))
    return res


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(res, out_dir, prefix="experiment_ah") -> List[str]:
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
    x = np.arange(len(LEVELS))
    xnames = [l[1] for l in LEVELS]

    def cell(reader, rep, level_key):
        return R.get(reader, {}).get(rep, {}).get(level_key)

    def stat(reader, rep, level_key, pop, field, sub="mean"):
        node = cell(reader, rep, level_key)
        if node is None:
            return None
        return node["pops"][pop][field][sub]

    # 1 training fit vs width level
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for reader in READERS:
        for rep, ls, alpha in (("L4", "-", 1.0), ("Lfull", "--", 0.6)):
            means = [nz(stat(reader, rep, lk, "train_fit", "raw_acc", "mean")) for lk in LEVEL_KEYS]
            sds = [nz(stat(reader, rep, lk, "train_fit", "raw_acc", "sd")) for lk in LEVEL_KEYS]
            ax.errorbar(x, means, yerr=sds, marker="o", ls=ls, color=cmap[reader], alpha=alpha,
                        capsize=3, label=f"{reader} {rep}")
    ax.axhline(TRAIN_FIT_BAR, ls=":", c="k", lw=0.8, label=f"{TRAIN_FIT_BAR:.2f} fit bar")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label=f"chance {CHANCE:.2f}")
    ax.axvline(BASELINE_IDX, ls="-", c="gray", lw=0.6, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(xnames, fontsize=8)
    ax.set_xlabel("reader width level")
    ax.set_ylabel("training accuracy (eligible pool, best-on-VAL)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AH: training fit vs reader width (AdamW, 200 updates)")
    save(fig, "width_vs_fit")

    # 2 held-out Class-A balanced accuracy vs width level
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for reader in READERS:
        for rep, ls, alpha in (("L4", "-", 1.0), ("Lfull", "--", 0.6)):
            means = [nz(stat(reader, rep, lk, "heldout_classA", "balanced_acc", "mean"))
                     for lk in LEVEL_KEYS]
            sds = [nz(stat(reader, rep, lk, "heldout_classA", "balanced_acc", "sd"))
                   for lk in LEVEL_KEYS]
            ax.errorbar(x, means, yerr=sds, marker="o", ls=ls, color=cmap[reader], alpha=alpha,
                        capsize=3, label=f"{reader} {rep}")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label=f"chance {CHANCE:.2f}")
    ax.axhline(HELDOUT_WEAK, ls=":", c="k", lw=0.8, label=f"{HELDOUT_WEAK:.2f} weak")
    ax.axhline(HELDOUT_MATERIAL, ls="--", c="k", lw=0.8, label=f"{HELDOUT_MATERIAL:.2f} material")
    ax.axvline(BASELINE_IDX, ls="-", c="gray", lw=0.6, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(xnames, fontsize=8)
    ax.set_xlabel("reader width level")
    ax.set_ylabel("held-out Class-A balanced accuracy (best-on-VAL)")
    ax.set_ylim(0, 0.5); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AH: held-out operand generalization vs reader width")
    save(fig, "width_vs_heldout_bal")

    # 3 paired per-seed deltas vs the baseline width
    P = res.get("paired_deltas", {})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharex=True)
    panels = ((axes[0], "delta_heldout_classA_bal", "Δ held-out Class-A balanced"),
              (axes[1], "delta_train_fit_raw", "Δ training accuracy (eligible pool)"))
    for ax_, key, ylab in panels:
        for reader in READERS:
            for rep, mk, alpha in (("L4", "o", 1.0), ("Lfull", "s", 0.6)):
                for i, lk in enumerate(LEVEL_KEYS):
                    node = P.get(reader, {}).get(rep, {}).get(lk)
                    if node is None:
                        continue
                    vals = node[key]["per_seed"]
                    xs = np.full(len(vals), i, float) + np.linspace(-0.18, 0.18, len(vals))
                    ax_.scatter(xs, [nz(v) for v in vals], color=cmap[reader], marker=mk,
                                s=18, alpha=alpha)
        ax_.axhline(0.0, c="gray", lw=0.8)
        ax_.axhline(MATERIAL_GAIN, ls="--", c="k", lw=0.8, label=f"+{MATERIAL_GAIN:.2f}")
        ax_.axvline(BASELINE_IDX, ls="-", c="gray", lw=0.6, alpha=0.5)
        ax_.set_xticks(x); ax_.set_xticklabels(xnames, fontsize=7, rotation=15)
        ax_.set_ylabel(ylab); ax_.grid(alpha=0.3, axis="y")
        ax_.set_xlabel("reader width level"); ax_.legend(fontsize=7)
    handles = [plt.Line2D([], [], color=cmap[r], marker="o", ls="") for r in READERS] + [
        plt.Line2D([], [], color="k", marker="o", ls=""), plt.Line2D([], [], color="k", marker="s", ls="")]
    axes[1].legend(handles, READERS + ["L4", "Lfull"], fontsize=7)
    fig.suptitle("AH: per-seed paired deltas vs the baseline width (AdamW)", y=1.02)
    save(fig, "paired_deltas")

    # 4 per-seed held-out accuracy across the ladder
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    for ri, reader in enumerate(READERS):
        for ci, rep in enumerate(HISTORIES):
            ax_ = axes[ri][ci]
            per_seed_all = [stat(reader, rep, lk, "heldout_classA", "raw_acc", "per_seed")
                            for lk in LEVEL_KEYS]
            for j, s in enumerate(SEEDS):
                ys = [nz(p[j]) if p else np.nan for p in per_seed_all]
                ax_.plot(x, ys, marker=".", ms=3, lw=0.8, alpha=0.6, label=f"seed {s}")
            means = [nz(stat(reader, rep, lk, "heldout_classA", "raw_acc", "mean"))
                     for lk in LEVEL_KEYS]
            ax_.plot(x, means, marker="o", color="k", lw=1.6, label="mean")
            ax_.axhline(CHANCE, ls=":", c="gray", lw=0.8)
            ax_.axvline(BASELINE_IDX, ls="-", c="gray", lw=0.6, alpha=0.5)
            ax_.set_ylim(0, 0.5); ax_.grid(alpha=0.3)
            ax_.set_title(f"{reader} {rep}", fontsize=9)
            if ri == 1:
                ax_.set_xticks(x); ax_.set_xticklabels(xnames, fontsize=7, rotation=10)
            if ci == 0:
                ax_.set_ylabel("held-out Class-A raw acc")
    axes[0][1].legend(fontsize=6, ncol=2)
    fig.suptitle("AH: per-seed held-out Class-A accuracy across the width ladder")
    save(fig, "per_seed_heldout")

    # 5 exact parameter count vs held-out accuracy
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for reader in READERS:
        pkey = "gru_params" if reader == "GRU" else "tcn_params"
        for rep, mk, alpha in (("L4", "o", 1.0), ("Lfull", "s", 0.55)):
            xs, ys, es = [], [], []
            for li, lk in enumerate(LEVEL_KEYS):
                if cell(reader, rep, lk) is None:
                    continue
                xs.append(res["levels"][li][pkey])
                ys.append(nz(stat(reader, rep, lk, "heldout_classA", "balanced_acc", "mean")))
                es.append(nz(stat(reader, rep, lk, "heldout_classA", "balanced_acc", "sd")))
            ax.errorbar(xs, ys, yerr=es, marker=mk, ls="-", color=cmap[reader], alpha=alpha,
                        capsize=3, label=f"{reader} {rep}")
    for li, lev in enumerate(res["levels"]):
        t = stat("GRU", "L4", lev["level_key"], "heldout_classA", "balanced_acc", "mean")
        if t is not None:
            ax.annotate(lev["level_key"], (lev["gru_params"], t), fontsize=6,
                        xytext=(2, 6), textcoords="offset points")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label=f"chance {CHANCE:.2f}")
    ax.axhline(HELDOUT_WEAK, ls=":", c="k", lw=0.8, label=f"{HELDOUT_WEAK:.2f} weak")
    ax.axhline(HELDOUT_MATERIAL, ls="--", c="k", lw=0.8, label=f"{HELDOUT_MATERIAL:.2f} material")
    ax.set_xlabel("exact trainable parameter count (reader-specific)")
    ax.set_ylabel("held-out Class-A balanced accuracy")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AH: capacity (exact parameter count) vs held-out generalization")
    save(fig, "param_vs_acc")

    # 6 decision summary (best non-baseline width vs the baseline width)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    best = None
    for reader in READERS:
        for rep in HISTORIES:
            for lk in LEVEL_KEYS:
                if lk == BASELINE_KEY:
                    continue
                v = stat(reader, rep, lk, "heldout_classA", "balanced_acc", "mean")
                if v is None:
                    continue
                if best is None or v > best[1]:
                    best = ((reader, rep, lk), v)
    if best is None:
        reader, rep, cmp_key = READERS[0], HISTORIES[0], BASELINE_KEY
    else:
        (reader, rep, cmp_key), _ = best
    cats = [f"train fit\n{cmp_key}", f"train fit\n{BASELINE_KEY}",
            f"held-out bal\n{cmp_key}", f"held-out bal\n{BASELINE_KEY}",
            f"Class-A raw\n{cmp_key}"]
    vals = [stat(reader, rep, cmp_key, "train_fit", "raw_acc", "mean"),
            stat(reader, rep, BASELINE_KEY, "train_fit", "raw_acc", "mean"),
            stat(reader, rep, cmp_key, "heldout_classA", "balanced_acc", "mean"),
            stat(reader, rep, BASELINE_KEY, "heldout_classA", "balanced_acc", "mean"),
            stat(reader, rep, cmp_key, "test_classA", "raw_acc", "mean")]
    ax.bar(np.arange(len(cats)), [nz(v) for v in vals],
           color=["tab:orange", "tab:gray", "tab:purple", "tab:gray", "tab:blue"])
    for yval, lab in ((TRAIN_FIT_BAR, "0.50 fit bar"), (HELDOUT_MATERIAL, "0.30 material"),
                      (HELDOUT_WEAK, "0.25 weak"), (CHANCE, "0.10 chance")):
        ax.axhline(yval, ls=":", c="k", lw=0.7)
    ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels(cats, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy"); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"AH: decision summary ({d['verdict']}) [{reader}/{rep}, best width vs baseline]")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AH analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--meta", default="experiments/experiment_ah_capacity_ladder_meta.json")
    ap.add_argument("--u-results", default="experiments/experiment_u_results.json")
    ap.add_argument("--ae-raw", default="experiments/experiment_ae_grid_raw.json")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--use-raw", action="store_true")
    ap.add_argument("--output", default="experiments/experiment_ah_results.json")
    ap.add_argument("--raw-output", default="experiments/experiment_ah_grid_raw.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_ah")
    ap.add_argument("--readers", nargs="*", default=None)
    ap.add_argument("--histories", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    data = ad.load(args.npz)
    data["path"] = args.npz
    data["ae_raw"] = args.ae_raw
    problems = ad._validate(data["meta"])
    if problems:
        raise SystemExit(f"AH source invalid: {problems}")
    ah_meta = json.load(open(args.meta)) if os.path.exists(args.meta) else {}
    meta_problems = _check_meta(ah_meta)
    if meta_problems:
        raise SystemExit(f"AH pre-registration mismatch: {meta_problems}")
    u_base = av.load_u_baseline(args.u_results) if os.path.exists(args.u_results) else None

    readers = args.readers or READERS
    histories = args.histories or HISTORIES
    levels = args.levels or LEVEL_KEYS
    bad = ([r for r in readers if r not in READERS]
           + [h for h in histories if h not in HISTORIES]
           + [k for k in levels if k not in LEVEL_KEYS])
    if bad:
        raise SystemExit(f"unknown grid subset: {bad}")

    if args.use_raw and os.path.exists(args.raw_output):
        with open(args.raw_output) as f:
            parts = json.load(f)["parts"]
    else:
        parts = _collect(data, args.jobs, readers, histories, levels)
        os.makedirs(os.path.dirname(args.raw_output), exist_ok=True)
        with open(args.raw_output, "w") as f:
            json.dump({"parts": parts}, f)
        if args.collect_only:
            print(f"Wrote raw {args.raw_output} ({len(parts)} fits)")
            return
    t0 = time.perf_counter()
    res = assemble(data, parts, u_base, ah_meta)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"AH verdict: {d['verdict']} ({d['framing']})")
    for k, v in d["cells"].items():
        tag = "transfer" if v["transfer_signal"] else "fit" if v["fit_present"] else "no-fit"
        print(f"  {k}: held-out bal {_fmt(v['heldout_classA_bal'])} "
              f"(base {_fmt(v['heldout_classA_bal_baseline'])}, "
              f"Δ{_fmt(v['mean_delta_bal'], '+.3f')}, seeds+{v['seeds_bal_material_gain']}/5) "
              f"train {_fmt(v['train_fit'])} (Δ{_fmt(v['mean_delta_fit'], '+.3f')}) -> {tag}")
    print(f"accessibility: {res['accessibility']['verdict']}")
    bridge = res["ae_bridge"]
    if bridge.get("available"):
        worst = f" (worst {bridge['worst']})" if bridge.get("worst") else ""
        print(f"AE baseline bridge: max|d| train={bridge['max_abs_diff_train_fit']:.2e} "
              f"classA={bridge['max_abs_diff_classA']:.2e} "
              f"heldout={bridge['max_abs_diff_heldout_classA']:.2e} "
              f"best_val={bridge['max_abs_diff_best_val_acc']:.2e}{worst}")
    else:
        print("AE baseline bridge: skipped (AE raw not present)")
    paths = make_figures(res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
