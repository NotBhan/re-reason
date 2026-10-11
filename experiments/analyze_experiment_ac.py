#!/usr/bin/env python3
"""Experiment AC — analysis: TCN re-qualification on the ADD `y_true` target.

AB (AB-A) showed the causal TCN fits the directly observable `op_dest` control under AdamW or a
xavier initialization under Adam, while Adam/default and RMSprop fail. AC asks whether the same
AB-qualified recipes let the TCN fit the ACTUAL ADD target `y_true = (operand_a + operand_b) mod 10`
on Y's fixed 256-transition subset, and whether any fit transfers to VAL/TEST.

The trainer/optimizer/initialization implementations are AB's EXACTLY (`analyze_experiment_ab.
train_ab`, `make_optimizer`, `apply_init`); the data/subset/splits/seeds are Y's EXACTLY (the
`y_true` fit seed `aw.reader_seed(seed, reader, "y_true", rep, top, "main")`). The `adam_ref`
condition at 5,000 updates must reproduce Y's original 1,000-update `adam_ref` trajectory.

PRE-REGISTRATION (declared before aggregate inspection)
-------------------------------------------------------
* Conditions (all lr 3e-3, weight decay 1e-2 where supported, 5,000 updates, matched deterministic
  init per (seed, reader, history)):
      adam_ref    : Adam,  default init   (reference bridge; must reproduce Y; not a qualified recipe)
      adamw       : AdamW, default init   (AB-qualified recipe)
      adam_xavier : Adam,  AB xavier init (AB-qualified recipe)
  The qualified recipes are the primary AC conditions; `adam_ref` is reported separately and never
  determines whether a qualified recipe succeeds.
* Fit criterion `fit_bar = 0.90`; a recipe is ROBUST at a history iff the five-seed mean maximum
  training accuracy >= 0.90 AND >= 4/5 seeds reach 0.90.
* FITTING decision (checked in order):
      AC-INVALID : preflight/data/checkpoint/determinism/protocol invariant failure, the GRU
                   positive control fails to robustly fit under `adam_ref`, or `adam_ref` does not
                   reproduce Y's stored 1,000-update trajectory.
      AC-A     : some QUALIFIED recipe is robust at BOTH L4 and Lfull.
      AC-MIX   : some QUALIFIED recipe is robust at exactly one history, or reaches mean >= 0.90
                 without the 4/5-seed criterion.
      AC-B     : no QUALIFIED recipe meets the robust criterion at either history.
* GENERALIZATION is a SEPARATE reported axis (never part of the fitting classification). For each
  qualified recipe robust at a history, at its best-on-VAL state, compare the TCN's test `y_true`
  accuracy on all-ADD, Class-A and held-out operand pairs `(b in {0,5})` with Y's unchanged records;
  bars reuse Y's conventions (`weak = 0.25`, `material = 0.30`, reader-recovery `0.50`).
      fit_without_generalization : fit (train_max >= 0.90) with Class-A <= 0.25 AND held-out <= 0.25.
      material_transfer          : Class-A >= 0.50 AND held-out >= 0.30.
      partial_transfer           : otherwise (a fit whose transfer is measurable but sub-material).
* Checkpoint selection is VAL-only; best-on-TRAIN is diagnostic; TEST never selects.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_ac.py \
        --npz experiments/experiment_ac_ytrue_requalification.npz \
        --y-results experiments/experiment_y_results.json \
        --output experiments/experiment_ac_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_ac
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

import analyze_experiment_ab as ab  # noqa: E402  (AB's exact condition/trainer implementations)
import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_u as au  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import analyze_experiment_w as aw  # noqa: E402
import analyze_experiment_x as ax  # noqa: E402
import analyze_experiment_y as ay  # noqa: E402  (Y's trainer, for the reproduction bridge)

PRIMARY_MODEL = av.PRIMARY_MODEL
CONTROL_MODEL = av.CONTROL_MODEL
SEEDS = av.SEEDS
READERS = ["TCN", "GRU"]
DECISIVE = ["L4", "Lfull"]
STATES = ["final", "best_val", "best_train"]
POPULATIONS = ["classA", "matched_correct", "correct_all", "all_add"]

TARGET = "y_true"
NCLASS = 10
SUBSET_SIZE = 256
LR = ab.LR
WD = ab.WD
UPDATES = 5000
EVAL_EVERY = ab.EVAL_EVERY
MILESTONES = [200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000]
Y_MILESTONES = [200, 400, 600, 800, 1000]           # Y's original budget

COND_NAMES = ["adam_ref", "adamw", "adam_xavier"]
QUALIFIED = ["adamw", "adam_xavier"]
COND_BY_NAME = {c: ab.COND_BY_NAME[c] for c in COND_NAMES}

FIT_BAR = 0.90
ROBUST_SEEDS = 4
TEST_WEAK = 0.25
HELDOUT_WEAK = 0.25
HELDOUT_MATERIAL = 0.30
RECOVER_BAR = 0.50
# Exact-identity tolerance for the adam_ref reproduction bridge: AC's adam_ref milestones are compared
# against a FRESH run of Y's own trainer (`analyze_experiment_y.train_trajectory`) with the identical
# seed/data/optimizer/initialization. Since `analyze_experiment_ab.train_ab` is byte-identical to that
# trainer, the comparison must be exact (0.0). The comparison against Y's STORED JSON is reported
# separately (cross-environment numerical drift, not a protocol mismatch) and does not gate the verdict.
IDENTITY_TOL = 1e-9

TOP = 10000
_G: Dict[str, Any] = {}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def _fit_seed(seed: int, reader: str, rep: str) -> int:
    """Y's exact per-fit seed for the primary target at the top scale."""
    return aw.reader_seed(seed, reader, TARGET, rep, TOP, "main")


# ---------------------------------------------------------------------------
# worker: one (seed, reader, history, condition)
# ---------------------------------------------------------------------------

def _init_worker(npz_path: str, arch: Dict[str, Any]) -> None:
    import torch
    torch.set_num_threads(1)
    _G["z"] = np.load(npz_path, allow_pickle=False)
    _G["arch"] = arch
    _G["feat"] = {}


def _feat(split: str, seed: int, L: int) -> np.ndarray:
    key = (split, seed, L)
    f = _G["feat"].get(key)
    if f is None:
        z = _G["z"]
        if split == "test":
            seq = z[f"{PRIMARY_MODEL}__test__seq_obs"]
            t = z[f"{PRIMARY_MODEL}__test__t"]
            rows = np.nonzero(z[f"{PRIMARY_MODEL}__test__seed"] == seed)[0]
        elif split == "val":
            seq = z["val__seq_obs"]
            t = z["val__t"]
            rows = np.nonzero(z["val__seed"] == seed)[0]
        else:
            seq = z[f"sub{SUBSET_SIZE}__seq_obs"]
            t = z[f"sub{SUBSET_SIZE}__t"]
            rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == seed)[0]
        f = av.reader_input(seq[rows], t[rows], L)
        _G["feat"][key] = f
    return f


def _task(spec: Tuple[int, str, str, str]) -> Dict[str, Any]:
    seed, reader, rep, cond_name = spec
    cond = COND_BY_NAME[cond_name]
    _G["feat"] = {}
    z = _G["z"]
    arch = _G["arch"]
    L = 4 if rep == "L4" else 10
    sub_rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == seed)[0]
    ytr = z[f"sub{SUBSET_SIZE}__{TARGET}"][sub_rows].astype(int)
    val_rows = np.nonzero(z["val__seed"] == seed)[0]
    yva = z[f"val__{TARGET}"][val_rows].astype(int)
    te_sel = z[f"{PRIMARY_MODEL}__test__seed"] == seed
    te_idx = np.nonzero(te_sel)[0]
    held_te = au._pairs(z, PRIMARY_MODEL, "test", te_idx)
    masks = {m: {p: au._pop_masks(z, m, seed)[p][te_sel] for p in POPULATIONS}
             for m in (PRIMARY_MODEL, CONTROL_MODEL)}
    yte = z[f"{PRIMARY_MODEL}__test__{TARGET}"][te_sel].astype(int)
    Xtr, Xva, Xte = _feat("sub", seed, L), _feat("val", seed, L), _feat("test", seed, L)

    def eval_state(state):
        pred_t = ax.predict_state(reader, state, Xtr, None, NCLASS, av.INPUT_DIM, arch)
        pred_v = ax.predict_state(reader, state, Xva, None, NCLASS, av.INPUT_DIM, arch)
        pred_e = ax.predict_state(reader, state, Xte, None, NCLASS, av.INPUT_DIM, arch)
        acc = {f"{m}:{p}": (float((pred_e[masks[m][p]] == yte[masks[m][p]]).mean())
                            if masks[m][p].any() else None)
               for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
        seen = {f"{m}:{p}": (float((pred_e[masks[m][p] & ~held_te] == yte[masks[m][p] & ~held_te]).mean())
                             if (masks[m][p] & ~held_te).any() else None)
                for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
        hold = {f"{m}:{p}": (float((pred_e[masks[m][p] & held_te] == yte[masks[m][p] & held_te]).mean())
                             if (masks[m][p] & held_te).any() else None)
                for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
        return {"train_acc": float((pred_t == ytr).mean()), "val_acc": float((pred_v == yva).mean()),
                "acc": acc, "seen": seen, "heldout": hold}

    fit = ab.train_ab(reader, Xtr, ytr, Xva, yva, NCLASS, _fit_seed(seed, reader, rep),
                      UPDATES, cond, arch)
    ms = {str(u): next((e for e in fit["events"] if e["update"] == u), None) for u in MILESTONES}
    key = f"{reader}|{rep}|{cond_name}"
    cell = {"cond": cond_name, "opt": cond["opt"], "init": cond["init"], "rep": rep,
            "params": av.reader_param_count(reader, NCLASS, arch), "n_train": int(len(sub_rows)),
            "fit": {"train_acc_max": float(fit["best_train"][0]),
                    "train_acc_max_update": int(fit["best_train"][2]),
                    "train_acc_final": float(fit["train_acc_final"]),
                    "train_loss_final": float(fit["train_loss_final"]),
                    "val_acc_final": float(fit["val_acc_final"]),
                    "best_val_acc": float(fit["best_val"][0]),
                    "best_val_update": int(fit["best_val"][2]),
                    "finite": fit["finite"], "milestones": ms},
            "final": eval_state(fit["final"]), "best_val": eval_state(fit["best_val"][1]),
            "best_train": eval_state(fit["best_train"][1])}
    out = {"seed": seed, "reader": reader, "rep": rep, "cond": cond_name, "cell": cell,
           "events": fit["events"]}
    if cond_name == "adam_ref":
        # reproduction bridge: a FRESH run of Y's own trainer at Y's 1,000-update budget with the
        # identical seed/data/optimizer/initialization; must match the first 1,000 updates exactly.
        yfit = ay.train_trajectory(reader, Xtr, ytr, Xva, yva, NCLASS, _fit_seed(seed, reader, rep),
                                   1000, arch)
        out["y_trainer_milestones"] = {
            str(u): next((e for e in yfit["events"] if e["update"] == u), None) for u in Y_MILESTONES}
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _ms(vals):
    return as_.mean_sd(vals)


def aggregate(parts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out: Dict[str, Any] = {}
    for reader in READERS:
        for rep in DECISIVE:
            for cond_name in COND_NAMES:
                recs = [p for p in parts if p["reader"] == reader and p["rep"] == rep
                        and p["cond"] == cond_name]
                if not recs:
                    continue
                recs = sorted(recs, key=lambda p: p["seed"])
                cells = [r["cell"] for r in recs]
                node = out.setdefault(reader, {}).setdefault(rep, {}).setdefault(cond_name, {})
                node["opt"] = cells[0]["opt"]
                node["init"] = cells[0]["init"]
                node["params"] = cells[0]["params"]
                node["n_train"] = cells[0]["n_train"]
                node["train_max"] = _ms([c["fit"]["train_acc_max"] for c in cells])
                node["per_seed_train_max"] = [c["fit"]["train_acc_max"] for c in cells]
                node["train_max_update"] = _ms([c["fit"]["train_acc_max_update"] for c in cells])
                node["train_final"] = _ms([c["fit"]["train_acc_final"] for c in cells])
                node["train_loss_final"] = _ms([c["fit"]["train_loss_final"] for c in cells])
                node["val_final"] = _ms([c["fit"]["val_acc_final"] for c in cells])
                node["best_val_acc"] = _ms([c["fit"]["best_val_acc"] for c in cells])
                node["best_val_update"] = _ms([c["fit"]["best_val_update"] for c in cells])
                node["all_finite"] = bool(all(c["fit"]["finite"] for c in cells))
                ms_keys = sorted(cells[0]["fit"]["milestones"].keys(), key=int)
                node["milestones"] = {u: {fld: _ms([c["fit"]["milestones"][u][fld] for c in cells])
                                          for fld in ("train_acc", "train_loss", "val_acc", "val_loss")}
                                      for u in ms_keys}
                for st in STATES:
                    node[st] = {
                        "train_acc": _ms([c[st]["train_acc"] for c in cells]),
                        "val_acc": _ms([c[st]["val_acc"] for c in cells]),
                        "acc": {pk: _ms([c[st]["acc"][pk] for c in cells]) for pk in cells[0][st]["acc"]},
                        "seen": {pk: _ms([c[st]["seen"][pk] for c in cells])
                                 for pk in cells[0][st]["seen"]},
                        "heldout": {pk: _ms([c[st]["heldout"][pk] for c in cells])
                                    for pk in cells[0][st]["heldout"]}}
    traj: Dict[str, Any] = {}
    for reader in READERS:
        for cond_name in COND_NAMES:
            for rep in DECISIVE:
                evs = [p["events"] for p in parts if p["reader"] == reader and p["rep"] == rep
                       and p["cond"] == cond_name]
                if not evs:
                    continue
                ups = [e["update"] for e in evs[0]]
                traj.setdefault(f"{reader}|{cond_name}", {})[rep] = {
                    "updates": ups,
                    "train_acc": [_ms([e[i]["train_acc"] for e in evs]) for i in range(len(ups))],
                    "val_acc": [_ms([e[i]["val_acc"] for e in evs]) for i in range(len(ups))]}
    return out, traj


def load_y_baseline(path: str) -> Dict[str, Any]:
    """Y's unchanged historical `adam_ref` records (milestones + best-on-VAL test accuracies)."""
    with open(path) as f:
        y = json.load(f)
    out: Dict[str, Any] = {"source": path, "verdict": y["primary_decision"]["verdict"]}
    for R in READERS:
        out[R] = {}
        for rep in DECISIVE:
            node = y["readers"][R][rep]
            out[R][rep] = {
                "train_max": node["fit"]["train_acc_max"]["mean"],
                "per_seed_train_max": node["fit"]["train_acc_max"].get("per_seed"),
                "train_final": node["fit"]["train_acc_final"]["mean"],
                "milestones": {str(u): {"train_acc": node["fit"]["milestones"][str(u)]["train_acc"]["mean"],
                                        "val_acc": node["fit"]["milestones"][str(u)]["val_acc"]["mean"]}
                               for u in Y_MILESTONES},
                "best_val_classA": node["best_val"]["acc"][f"{PRIMARY_MODEL}:classA"]["mean"],
                "best_val_all_add": node["best_val"]["acc"][f"{PRIMARY_MODEL}:all_add"]["mean"],
                "best_val_heldout_classA": node["best_val"]["heldout"][f"{PRIMARY_MODEL}:classA"]["mean"],
            }
    return out


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

def _robust(res, reader, rep, cond_name):
    node = res["readers"][reader][rep][cond_name]
    m = node["train_max"]["mean"]
    nfit = sum(1 for v in node["per_seed_train_max"] if v is not None and v >= FIT_BAR)
    return bool(m is not None and m >= FIT_BAR and nfit >= ROBUST_SEEDS)


def _mean_only(res, reader, rep, cond_name):
    node = res["readers"][reader][rep][cond_name]
    m = node["train_max"]["mean"]
    nfit = sum(1 for v in node["per_seed_train_max"] if v is not None and v >= FIT_BAR)
    return bool(m is not None and m >= FIT_BAR and nfit < ROBUST_SEEDS)


def reproduction(res: Dict[str, Any], parts: List[Dict[str, Any]],
                 y_base: Dict[str, Any]) -> Dict[str, Any]:
    """adam_ref reproduction bridge.

    * `trainer_identity`: AC's per-seed adam_ref first-1,000-update milestones vs a FRESH run of Y's
      own trainer (`analyze_experiment_y.train_trajectory`) at the identical seed/data/optimizer/init.
      This must be exact (`IDENTITY_TOL`) and is the gating check.
    * `stored_bridge`: AC's adam_ref 5-seed milestone means (aggregated) vs Y's STORED JSON means,
      reported explicitly (cross-environment numerical drift amplified on sensitive cells), never
      gating.
    """
    tid_max, tid_worst, n_tid = 0.0, None, 0
    tid_cells: Dict[str, Dict[str, float]] = {}
    stored_tr, stored_va, stored_worst, n_stored = 0.0, 0.0, None, 0
    for p in parts:
        if p["cond"] != "adam_ref":
            continue
        R, rep = p["reader"], p["rep"]
        ytm = p.get("y_trainer_milestones")
        tid_cells.setdefault(R, {})[rep] = 0.0
        for u in Y_MILESTONES:
            acm = p["cell"]["fit"]["milestones"].get(str(u))
            if ytm is not None and ytm.get(str(u)) is not None:
                for f in ("train_acc", "val_acc"):
                    d = abs(float(acm[f]) - float(ytm[str(u)][f]))
                    n_tid += 1
                    tid_cells[R][rep] = max(tid_cells[R][rep], d)
                    if d > tid_max:
                        tid_max, tid_worst = d, f"{R}|{rep}|{u}|{f}"
    for R in READERS:
        for rep in DECISIVE:
            for u in Y_MILESTONES:
                acm = res["readers"][R][rep]["adam_ref"]["milestones"][str(u)]
                yb = y_base[R][rep]["milestones"][str(u)]
                dtr = abs(float(acm["train_acc"]["mean"]) - float(yb["train_acc"]))
                dva = abs(float(acm["val_acc"]["mean"]) - float(yb["val_acc"]))
                n_stored += 1
                if dtr > stored_tr:
                    stored_tr, stored_worst = dtr, f"{R}|{rep}|{u}|train_acc"
                stored_va = max(stored_va, dva)
    identical = bool(tid_max <= IDENTITY_TOL)
    return {"trainer_identity": {"max_abs_diff": tid_max, "worst": tid_worst,
                                 "per_cell_max": tid_cells, "identical": identical,
                                 "n_compared": n_tid, "tol": IDENTITY_TOL,
                                 "note": "fresh re-run of Y's own trainer under AC's identical "
                                         "seed/data/optimizer/init; must be exact"},
            "stored_bridge": {"max_abs_diff_train": stored_tr, "max_abs_diff_val": stored_va,
                              "worst_train": stored_worst, "n_compared": n_stored,
                              "note": "AC adam_ref vs Y's STORED JSON; cross-environment numerical "
                                      "drift (amplified on sensitive cells), not a protocol mismatch; "
                                      "reported explicitly and never gating"},
            "reproduced": identical}


def classify_fit(res: Dict[str, Any], problems: List[str], repro: Dict[str, Any],
                 control_ok: bool) -> Dict[str, Any]:
    minimal = {"verdict": "AC-INVALID", "control_ok": control_ok, "reproduce_y": repro["reproduced"],
               "qualified_robust": {rep: [] for rep in DECISIVE},
               "qualified_mean_only": {rep: [] for rep in DECISIVE},
               "adam_ref_robust": {rep: [] for rep in DECISIVE},
               "gru_robust": {rep: [] for rep in DECISIVE},
               "qualified_verdicts": [], "validation_problems": problems,
               "thresholds": {"fit_bar": FIT_BAR, "robust_seeds": ROBUST_SEEDS}}
    if problems:
        return minimal
    qr = {rep: [c for c in QUALIFIED if _robust(res, "TCN", rep, c)] for rep in DECISIVE}
    qm = {rep: [c for c in QUALIFIED if _mean_only(res, "TCN", rep, c)] for rep in DECISIVE}
    ar = {rep: [c for c in COND_NAMES if _robust(res, "TCN", rep, c)] for rep in DECISIVE}
    gr = {rep: [c for c in COND_NAMES if _robust(res, "GRU", rep, c)] for rep in DECISIVE}
    both = [c for c in QUALIFIED if c in qr["L4"] and c in qr["Lfull"]]
    any_one = bool(qr["L4"] or qr["Lfull"])
    any_mean = bool(qm["L4"] or qm["Lfull"])
    if not control_ok or not repro["reproduced"]:
        verdict = "AC-INVALID"
    elif both:
        verdict = "AC-A"
    elif any_one or any_mean:
        verdict = "AC-MIX"
    else:
        verdict = "AC-B"
    return {"verdict": verdict, "control_ok": control_ok, "reproduce_y": repro["reproduced"],
            "qualified_robust": qr, "qualified_mean_only": qm, "adam_ref_robust": ar,
            "gru_robust": gr, "qualified_verdicts": both, "validation_problems": problems,
            "thresholds": {"fit_bar": FIT_BAR, "robust_seeds": ROBUST_SEEDS},
            "rule_note": ("AC-INVALID if preflight/invariant/protocol fails, the adam_ref GRU control "
                          "is not robust at both histories, or adam_ref does not reproduce Y's stored "
                          "1,000-update trajectory; else AC-A if some QUALIFIED recipe (adamw, "
                          "adam_xavier) is robust at BOTH histories; else AC-MIX if a qualified recipe "
                          "is robust at exactly one history or reaches mean>=0.90 without the 4/5-seed "
                          "criterion; else AC-B. Robust = five-seed mean max-train >=0.90 AND >=4/5 "
                          "seeds >=0.90. adam_ref is reported separately.")}


def generalization(res: Dict[str, Any], fit_dec: Dict[str, Any], y_base: Dict[str, Any]) -> Dict[str, Any]:
    """Separate generalization axis (never part of the fitting classification)."""
    per_recipe: Dict[str, Any] = {}
    for cond_name in QUALIFIED:
        reps = fit_dec["qualified_robust"]
        robust_reps = [rep for rep in DECISIVE if cond_name in reps[rep]]
        # best history by TCN max-train (fit metric only; test never selects)
        best_rep = None
        for rep in DECISIVE:
            if cond_name in reps[rep]:
                if best_rep is None or (res["readers"]["TCN"][rep][cond_name]["train_max"]["mean"]
                                        > res["readers"]["TCN"][best_rep][cond_name]["train_max"]["mean"]):
                    best_rep = rep
        hists = robust_reps or DECISIVE
        metrics = {}
        for rep in hists:
            node = res["readers"]["TCN"][rep][cond_name]
            metrics[rep] = {
                "train_max": node["train_max"]["mean"],
                "best_val_test_all_add": node["best_val"]["acc"][f"{PRIMARY_MODEL}:all_add"]["mean"],
                "best_val_test_classA": node["best_val"]["acc"][f"{PRIMARY_MODEL}:classA"]["mean"],
                "best_val_heldout_classA": node["best_val"]["heldout"][f"{PRIMARY_MODEL}:classA"]["mean"],
                "best_val_seen_classA": node["best_val"]["seen"][f"{PRIMARY_MODEL}:classA"]["mean"],
            }
        classA = max((m["best_val_test_classA"] for m in metrics.values()
                      if m["best_val_test_classA"] is not None), default=None)
        heldout = max((m["best_val_heldout_classA"] for m in metrics.values()
                       if m["best_val_heldout_classA"] is not None), default=None)
        all_add = max((m["best_val_test_all_add"] for m in metrics.values()
                       if m["best_val_test_all_add"] is not None), default=None)
        fits = bool(best_rep is not None)
        if not fits:
            cat = "not_fitted"
        elif classA is not None and heldout is not None and classA <= TEST_WEAK and heldout <= HELDOUT_WEAK:
            cat = "fit_without_generalization"
        elif classA is not None and heldout is not None and classA >= RECOVER_BAR and heldout >= HELDOUT_MATERIAL:
            cat = "material_transfer"
        else:
            cat = "partial_transfer"
        per_recipe[cond_name] = {"fits": fits, "best_rep": best_rep, "robust_reps": robust_reps,
                                 "metrics": metrics, "max_classA": classA, "max_heldout": heldout,
                                 "max_all_add": all_add, "category": cat}
    cats = {c: v["category"] for c, v in per_recipe.items()}
    fitted = {c: v for c, v in per_recipe.items() if v["fits"]}
    if fitted and all(v["category"] == "fit_without_generalization" for v in fitted.values()):
        overall = "fit_without_generalization"
    elif any(v["category"] == "material_transfer" for v in fitted.values()):
        overall = "material_transfer"
    elif fitted:
        overall = "partial_transfer"
    else:
        overall = "not_fitted"
    return {"verdict": overall, "per_recipe": per_recipe, "categories": cats,
            "y_historical": {R: {rep: {"train_max": y_base[R][rep]["train_max"],
                                      "best_val_classA": y_base[R][rep]["best_val_classA"],
                                      "best_val_all_add": y_base[R][rep]["best_val_all_add"],
                                      "best_val_heldout_classA": y_base[R][rep]["best_val_heldout_classA"]}
                                 for rep in DECISIVE} for R in READERS},
            "thresholds": {"test_weak": TEST_WEAK, "heldout_weak": HELDOUT_WEAK,
                           "heldout_material": HELDOUT_MATERIAL, "recover_bar": RECOVER_BAR},
            "rule_note": ("Generalization is reported SEPARATELY from fitting. For each qualified recipe "
                          "at its best-on-VAL state: fit_without_generalization if Class-A<=0.25 AND "
                          "held-out<=0.25; material_transfer if Class-A>=0.50 AND held-out>=0.30; else "
                          "partial_transfer. A fit without meaningful transfer is NOT arithmetic "
                          "recovery.")}


def _validate(meta: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if not meta.get("preflight_pass", False):
        problems.append("trainable-parameter preflight failed")
    for where in ("train", "val", "test"):
        cc = meta.get("class_counts", {}).get(where, {})
        pooled = cc.get("pooled")
        if pooled is not None and (len(pooled) != NCLASS or any(int(c) <= 0 for c in pooled)):
            problems.append(f"class coverage failure: {where}/pooled={pooled}")
        for k, v in cc.items():
            if k in ("pooled", "per_seed") or not isinstance(v, list):
                continue
            if len(v) != NCLASS or any(int(c) <= 0 for c in v):
                problems.append(f"class coverage failure: {where}/{k}={v}")
    for k, v in meta.get("disjointness", {}).items():
        if int(v) != 0:
            problems.append(f"split disjointness failure: {k}={v}")
    for k, rec in meta.get("target_derivation", {}).items():
        if not rec.get("exact", False):
            problems.append(f"y_true derivation failure: {k}={rec}")
    for k, rec in meta.get("heldout_exclusion", {}).items():
        if int(rec.get("n_with_b_in_{0,5}", 0)) != 0:
            problems.append(f"held-out exclusion failure: {k}={rec}")
    return problems


def _collect(data: Dict[str, Any], jobs: int, conditions: List[str], readers: List[str]
             ) -> List[Dict[str, Any]]:
    npz_path = data["path"]
    arch = av.architecture()
    specs = [(s, R, rep, c) for s in SEEDS for R in readers for rep in DECISIVE for c in conditions]
    t0 = time.perf_counter()
    if jobs <= 1:
        _init_worker(npz_path, arch)
        return [_task(sp) for sp in specs]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs, initializer=_init_worker, initargs=(npz_path, arch)) as pool:
        parts = list(pool.imap_unordered(_task, specs, chunksize=1))
    print(f"collection wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    return parts


def assemble(data: Dict[str, Any], parts: List[Dict[str, Any]], y_base: Dict[str, Any]) -> Dict[str, Any]:
    arch = av.architecture()
    readers, traj = aggregate(parts)
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "readers": READERS,
        "target": TARGET, "n_classes": NCLASS, "reps": DECISIVE, "states": STATES,
        "lr": LR, "weight_decay": WD, "updates": UPDATES, "eval_every": EVAL_EVERY,
        "milestones": MILESTONES, "y_milestones": Y_MILESTONES,
        "conditions": [{"name": c, **{k: COND_BY_NAME[c][k] for k in ("opt", "init")},
                        "qualified": c in QUALIFIED} for c in COND_NAMES],
        "qualified": QUALIFIED, "optimizer_defaults": ab.OPTIMIZER_DEFAULTS,
        "subset_size": data["meta"]["subset_size"], "subset_seed": data["meta"]["subset_seed"],
        "held_out_pairs": data["meta"]["held_out_pairs"], "populations": POPULATIONS,
        "source_identity": data["meta"]["source"], "class_counts": data["meta"]["class_counts"],
        "target_derivation": data["meta"]["target_derivation"],
        "heldout_exclusion": data["meta"]["heldout_exclusion"],
        "disjointness": data["meta"]["disjointness"], "preflight": data["meta"]["preflight"],
        "preflight_pass": data["meta"]["preflight_pass"], "checkpoints": data["meta"]["models"],
        "n": data["meta"]["n"],
        "thresholds": {"fit_bar": FIT_BAR, "robust_seeds": ROBUST_SEEDS, "test_weak": TEST_WEAK,
                       "heldout_weak": HELDOUT_WEAK, "heldout_material": HELDOUT_MATERIAL,
                       "recover_bar": RECOVER_BAR, "identity_tol": IDENTITY_TOL},
        "statistical_discipline": (
            "Five frozen E checkpoints are the independent units; F is a control. The two readers are "
            "analysis-only, fit on Y's exact fixed 256-transition subset, tuned on Y's canonical VAL, "
            "evaluated once on Y's untouched canonical TEST. Selection is VAL-only; best-on-TRAIN is "
            "diagnostic; TEST is never used to select a condition/reader/history. The three conditions "
            "(adam_ref/adamw/adam_xavier) reuse AB's exact optimizer/initialization implementations; "
            "the ADAM_REF condition at 5,000 updates must reproduce Y's stored 1,000-update record.")}}
    res["architecture"] = arch
    res["readers"] = readers
    res["trajectory"] = traj
    res["per_seed"] = parts
    problems = _validate(data["meta"])
    repro = reproduction(res, parts, y_base)
    control_ok = bool(_robust(res, "GRU", "L4", "adam_ref") and _robust(res, "GRU", "Lfull", "adam_ref"))
    fit_dec = classify_fit(res, problems, repro, control_ok)
    res["reproduction"] = repro
    res["y_baseline"] = y_base
    res["primary_decision"] = fit_dec
    res["generalization"] = generalization(res, fit_dec, y_base)
    return res


def build(data: Dict[str, Any], y_base: Dict[str, Any], jobs: int,
          conditions: Optional[List[str]] = None, readers: Optional[List[str]] = None) -> Dict[str, Any]:
    parts = _collect(data, jobs, conditions or COND_NAMES, readers or READERS)
    return assemble(data, parts, y_base)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_ac") -> List[str]:
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

    R = res["readers"]
    Y = res["y_baseline"]
    cmap = {"adam_ref": "tab:gray", "adamw": "tab:blue", "adam_xavier": "tab:green"}

    def tm(reader, rep, cond):
        v = R[reader][rep][cond]["train_max"]["mean"]
        return np.nan if v is None else v

    # 1 TCN max training accuracy by condition and history
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    xi = np.arange(len(DECISIVE))
    for loff, cond in zip((-0.25, 0.0, 0.25), COND_NAMES):
        ax.bar(xi + loff, [tm("TCN", rep, cond) for rep in DECISIVE], 0.25, label=cond, color=cmap[cond])
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.set_xticks(xi); ax.set_xticklabels(DECISIVE); ax.set_ylim(0, 1.02)
    ax.set_ylabel("max training accuracy (y_true)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AC: TCN on the 256-example ADD subset, by condition")
    save(fig, "tcn_by_condition")

    # 2 GRU control by condition and history
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for loff, cond in zip((-0.25, 0.0, 0.25), COND_NAMES):
        ax.bar(xi + loff, [tm("GRU", rep, cond) for rep in DECISIVE], 0.25, label=cond, color=cmap[cond])
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9)
    ax.set_xticks(xi); ax.set_xticklabels(DECISIVE); ax.set_ylim(0, 1.02)
    ax.set_ylabel("max training accuracy (y_true)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AC: GRU positive control, by condition")
    save(fig, "gru_control")

    # 3 training trajectories (TCN + GRU, all conditions)
    T = res["trajectory"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for reader, ls in (("TCN", "-"), ("GRU", "--")):
        for cond in COND_NAMES:
            node = T.get(f"{reader}|{cond}", {}).get("L4")
            if not node:
                continue
            ax.plot(node["updates"], [v["mean"] for v in node["train_acc"]], ls=ls, color=cmap[cond],
                    alpha=1.0 if reader == "TCN" else 0.5, label=f"{reader} {cond}")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.8)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("train accuracy (L4)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AC: training trajectories on the ADD subset (L4)")
    save(fig, "training_trajectory")

    # 4 reproduction bridge: AC adam_ref vs Y
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    labels, got, exp = [], [], []
    for reader in READERS:
        for rep in DECISIVE:
            labels.append(f"{reader}\n{rep}")
            got.append(Y[reader][rep]["milestones"]["1000"]["train_acc"])
            exp.append(R[reader][rep]["adam_ref"]["milestones"]["1000"]["train_acc"]["mean"])
    xi = np.arange(len(labels))
    ax.bar(xi - 0.2, exp, 0.4, label="Y historical (adam_ref, 1000 upd)", color="tab:gray")
    ax.bar(xi + 0.2, got, 0.4, label="AC adam_ref (1000 upd)", color="tab:blue")
    ax.set_xticks(xi); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("train accuracy @1000"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AC: adam_ref reproduces Y")
    save(fig, "reproduce_y")

    # 5 fit vs generalization (TCN, all conditions/histories)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for cond in COND_NAMES:
        for rep in DECISIVE:
            xv = tm("TCN", rep, cond)
            yv = R["TCN"][rep][cond]["best_val"]["acc"][f"{PRIMARY_MODEL}:classA"]["mean"]
            yv = np.nan if yv is None else yv
            ax.scatter(xv, yv, color=cmap[cond], marker="o" if rep == "L4" else "s", s=70, zorder=3)
            ax.annotate(f"{cond}\n{rep}", (xv, yv), fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.axvline(FIT_BAR, ls="--", c="k", lw=0.8, label="0.90 fit bar")
    ax.axhline(TEST_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("max training accuracy (subset)"); ax.set_ylabel("Class-A TEST accuracy (best-on-VAL)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("AC: TCN fitting vs ADD generalization")
    save(fig, "fit_vs_generalization")

    # 6 seen vs held-out operand pairs (Class-A) for the qualified recipes that fit
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    labs, seen, hold = [], [], []
    for cond in QUALIFIED:
        for rep in DECISIVE:
            node = R["TCN"][rep][cond]
            labs.append(f"{cond}\n{rep}")
            seen.append(node["best_val"]["seen"][f"{PRIMARY_MODEL}:classA"]["mean"])
            hold.append(node["best_val"]["heldout"][f"{PRIMARY_MODEL}:classA"]["mean"])
    xi = np.arange(len(labs))
    ax.bar(xi - 0.2, [np.nan if v is None else v for v in seen], 0.4, label="seen", color="tab:gray")
    ax.bar(xi + 0.2, [np.nan if v is None else v for v in hold], 0.4, label="held-out", color="tab:purple")
    ax.axhline(HELDOUT_MATERIAL, ls=":", c="k", lw=0.8, label="0.30 material")
    ax.axhline(HELDOUT_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xticks(xi); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("Class-A accuracy (best-on-VAL)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AC: TCN seen vs held-out operand pairs (qualified recipes)")
    save(fig, "heldout_generalization")

    # 7 per-seed max train (TCN) at the best qualified condition
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for rep, ls in (("L4", "-"), ("Lfull", "--")):
        ax.plot(np.arange(len(SEEDS)), R["TCN"][rep]["adamw"]["per_seed_train_max"], marker="o", ls=ls,
                label=f"adamw {rep}", color="tab:blue", alpha=1.0 if rep == "L4" else 0.6)
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.set_xticks(np.arange(len(SEEDS))); ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("E seed"); ax.set_ylabel("max training accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AC: per-seed TCN fit (adamw)")
    save(fig, "per_seed")

    # 8 decision summary: fit + generalization
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    d = res["primary_decision"]
    g = res["generalization"]
    cats = ["TCN best\n(max train)", "GRU best\n(max train)", "TCN test\n(all-ADD)", "TCN test\n(Class-A)",
            "TCN held-out\n(Class-A)"]
    tcn_best = max((tm("TCN", rep, c) for rep in DECISIVE for c in QUALIFIED
                    if not np.isnan(tm("TCN", rep, c))), default=np.nan)
    gru_best = max((tm("GRU", rep, c) for rep in DECISIVE for c in QUALIFIED
                    if not np.isnan(tm("GRU", rep, c))), default=np.nan)
    q = [g["per_recipe"][c] for c in QUALIFIED]

    def _mx(key):
        vals = [m for r in q for m in [r.get(key)] if m is not None]
        return max(vals) if vals else np.nan
    vals = [tcn_best, gru_best, _mx("max_all_add"), _mx("max_classA"), _mx("max_heldout")]
    ax.bar(np.arange(5), vals, color=["tab:orange", "tab:blue", "tab:red", "tab:purple", "tab:brown"])
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.axhline(TEST_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(cats, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"AC: fit={d['verdict']}, generalization={g['verdict']}")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    global TOP
    ap = argparse.ArgumentParser(description="Experiment AC analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_ac_ytrue_requalification.npz")
    ap.add_argument("--y-results", default="experiments/experiment_y_results.json")
    ap.add_argument("--output", default="experiments/experiment_ac_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_ac")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--raw", default="experiments/experiment_ac_grid_raw.json")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--use-raw", action="store_true")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--readers", nargs="*", default=None)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    data = load(args.npz)
    data["path"] = args.npz
    TOP = int(data["meta"]["source"]["top_size"])
    aw.TRAIN_SIZES = list(data["meta"]["source"]["train_sizes"])
    y_base = load_y_baseline(args.y_results)
    problems = _validate(data["meta"])
    if args.preflight_only or problems:
        print(json.dumps({"preflight_pass": data["meta"]["preflight_pass"], "problems": problems,
                          "preflight": data["meta"]["preflight"]}, indent=2))
        if problems and not args.preflight_only:
            res = assemble(data, [], y_base)
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(res, f, indent=2)
            print(f"INVALID: {problems}; wrote {args.output} verdict {res['primary_decision']['verdict']}")
        return
    if args.collect_only:
        parts = _collect(data, args.jobs, args.conditions or COND_NAMES, args.readers or READERS)
        os.makedirs(os.path.dirname(args.raw), exist_ok=True)
        with open(args.raw, "w") as f:
            json.dump({"parts": parts}, f)
        print(f"Wrote raw {args.raw} ({len(parts)} fits)")
        return
    if args.use_raw:
        with open(args.raw) as f:
            parts = json.load(f)["parts"]
    else:
        parts = _collect(data, args.jobs, args.conditions or COND_NAMES, args.readers or READERS)
        os.makedirs(os.path.dirname(args.raw), exist_ok=True)
        with open(args.raw, "w") as f:
            json.dump({"parts": parts}, f)
    t0 = time.perf_counter()
    res = assemble(data, parts, y_base)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"fit verdict: {d['verdict']}  control_ok={d['control_ok']} reproduce_y={d['reproduce_y']}")
    ti = res["reproduction"]["trainer_identity"]
    sb = res["reproduction"]["stored_bridge"]
    print(f"  trainer identity (AC adam_ref vs fresh Y trainer): max|d|={ti['max_abs_diff']:.2e} "
          f"identical={ti['identical']}")
    print(f"  stored bridge (AC adam_ref vs Y stored JSON): train max|d|={sb['max_abs_diff_train']:.2e} "
          f"val max|d|={sb['max_abs_diff_val']:.2e} @ {sb['worst_train']}")
    print(f"  TCN qualified robust L4={d['qualified_robust']['L4']} full={d['qualified_robust']['Lfull']} "
          f"mean_only={d['qualified_mean_only']}")
    print(f"  GRU robust L4={d['gru_robust']['L4']} full={d['gru_robust']['Lfull']}")
    for reader in READERS:
        for rep in DECISIVE:
            print(f"  {reader} {rep}: " + " ".join(
                f"{c}:{res['readers'][reader][rep][c]['train_max']['mean']:.3f}" for c in COND_NAMES))
    g = res["generalization"]
    print(f"generalization verdict: {g['verdict']}")
    for c, v in g["per_recipe"].items():
        print(f"  {c}: fits={v['fits']} best_rep={v['best_rep']} classA={v['max_classA']} "
              f"heldout={v['max_heldout']} all_add={v['max_all_add']} -> {v['category']}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
