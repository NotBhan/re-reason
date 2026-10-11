#!/usr/bin/env python3
"""Experiment AD — analysis: qualified-recipe re-evaluation of the ADD-target access test.

Experiments V/W/X evaluated the predeclared parameter-bounded causal readers (GRU hidden 85; causal
TCN channels 52) with **Adam** and PyTorch's default initialization. AB/AC established that the causal
TCN's fitting behaviour is optimizer-recipe sensitive: the AB-qualified `adamw` recipe (decoupled
weight decay, default init) lets the TCN fit the directly observable `op_dest` control and the
small-subset ADD `y_true` target (though the small-subset fit does not generalize). AD asks:

    Does changing Adam to the AB-qualified AdamW recipe change the TCN's access to the ADD target
    under V's original full training-data and evaluation protocol?

Three questions are kept strictly separate:
  (1) does the optimizer change materially affect Class-A target access?      -> AD-A / AD-B
  (2) does it improve generalization to held-out operand pairs?               -> paired delta axis
  (3) does it cross V's existing sequence-accessibility criterion?            -> V-A / V-B axis
The fourth question (RE:Reason frozen-representation claims) is not answerable by this reader-only
experiment and is out of scope.

Readers/trainer/optimizer/initialization are V's / AB's EXACT implementations (`av.build_reader`,
`ab.train_ab`, `ab.make_optimizer`, `ab.apply_init`); the data/splits/seeds are V's exactly. For each
(seed, reader, history, variant) the two conditions share the same deterministic initialization
(V's `av.reader_seed`, optimizer condition never enters the seed). Only the optimizer family differs.

PRE-REGISTRATION (declared before aggregate inspection)
------------------------------------------------------
* Conditions (lr 3e-3, weight decay 1e-2, 200 full-batch updates, evaluated every 10):
      adam_ref : Adam,  default init  (contemporaneous reference bridge; must reproduce V)
      adamw    : AdamW, default init  (AB-qualified recipe)
  Optimizer defaults are AB's (`betas (0.9,0.999)`, `eps 1e-8`); no scheduler, no extra optimizer,
  no alternative initialization, no hyperparameter search.
* Variants: `main` (ordinary-ADD training pool, V's inclusion rule) and `heldout` (V's held-out
  operand protocol, excluding the 20 pairs (a,b) with b in {0,5}); both across all six histories.
  Grid = 2 readers x 2 conditions x 6 histories x 2 variants x 5 seeds = 240 fits.
* Decision bars are V's existing conventions:
      MATERIAL_GAIN   = +0.15 absolute accuracy (paired, adamw - adam_ref).
      AD-A : at either L4 or Lfull, adamw yields a mean paired improvement >= +0.15 in Class-A y_true
             accuracy OR held-out Class-A accuracy, over the contemporaneous adam_ref.
      AD-B : neither metric improves by +0.15 at L4 or Lfull.
  A borderline / seed-dependent result is reported explicitly; the threshold is never changed.
* Sequence-accessibility is a SEPARATE axis using V's unchanged bar: V-A iff Class-A test accuracy
  >= 0.50 AND held-out Class-A accuracy >= 0.30 at L4 or Lfull; else V-B / partial / inconclusive by
  V's original rules. An AD-A verdict can coexist with failure to reach V-A.
* Checkpoint selection is VAL-only; TEST never selects. Bayes ceilings are U's, unmodified.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_ad.py \
        --npz experiments/experiment_ad_qualified_access.npz \
        --v-results experiments/experiment_v_results.json \
        --u-results experiments/experiment_u_results.json \
        --output experiments/experiment_ad_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_ad
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_experiment_ab as ab  # noqa: E402  (AB's exact optimizer/init/trainer)
import analyze_experiment_s as as_  # noqa: E402  (mean_sd / paired)
import analyze_experiment_u as au  # noqa: E402  (history / populations / pairs)
import analyze_experiment_v as av  # noqa: E402  (V's exact readers / seeds / U baseline loader)
import analyze_experiment_x as ax  # noqa: E402  (build_model / predict_state)

PRIMARY_MODEL = av.PRIMARY_MODEL
CONTROL_MODEL = av.CONTROL_MODEL
SEEDS = av.SEEDS
READERS = ["GRU", "TCN"]
REPS = au.REPS
REP_LABELS = au.REP_LABELS
DECISIVE = ["L4", "Lfull"]
POPULATIONS = ["classA", "matched_correct", "correct_all", "all_add"]
TARGET = "y_true"
NCLASS = 10

LR = av.READER_LR
WD = av.READER_WD
UPDATES = av.READER_EPOCHS              # 200 full-batch epochs (V's protocol)
EVAL_EVERY = 10
TRAIN_ORD_CAP = 1500
VAL_ORD_CAP = 800

CONDITIONS = [
    {"name": "adam_ref", "opt": "adam", "init": "default"},
    {"name": "adamw", "opt": "adamw", "init": "default"},
]
COND_NAMES = [c["name"] for c in CONDITIONS]
COND_BY_NAME = {c["name"]: c for c in CONDITIONS}
QUALIFIED = ["adamw"]

# V's existing decision conventions (unchanged)
RECOVER_BAR = 0.50
HELDOUT_MATERIAL = 0.30
HELDOUT_WEAK = 0.25
MATERIAL_GAIN = 0.15
BAYES_GAP_BAR = 0.30

_G: Dict[str, Any] = {}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def _ms(vals):
    return as_.mean_sd(vals)


def _state_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _init_hash(reader: str, nclass: int, seed: int, init: str, arch: Dict[str, Any]) -> str:
    """Deterministic initial-parameter fingerprint for a (reader, seed, init) triple."""
    import torch
    torch.manual_seed(seed)
    model = ax.build_model(reader, nclass, av.INPUT_DIM, arch)
    ab.apply_init(model, init)
    return _state_hash(model)


# ---------------------------------------------------------------------------
# workers: one (seed, reader, history, variant) task -> both optimizer conditions
# ---------------------------------------------------------------------------

def _init_worker(npz_path: str, arch: Dict[str, Any]) -> None:
    import torch
    torch.set_num_threads(1)
    _G["z"] = np.load(npz_path, allow_pickle=False)
    _G["arch"] = arch
    _G["feat"] = {}


def _feat(split: str, seed: int, L: int) -> np.ndarray:
    """V's exact per-timestep reader input (raw 21-dim observation + 1 validity bit = 22 features)."""
    key = (split, seed, L)
    f = _G["feat"].get(key)
    if f is None:
        z = _G["z"]
        if split == "test":
            seq = z[f"{PRIMARY_MODEL}__test__seq_obs"]
            t = z[f"{PRIMARY_MODEL}__test__t"]
            rows = np.nonzero(z[f"{PRIMARY_MODEL}__test__seed"] == seed)[0]
        else:
            seq = z[f"{PRIMARY_MODEL}__{split}__seq_obs"]
            t = z[f"{PRIMARY_MODEL}__{split}__t"]
            rows = np.nonzero(z[f"{PRIMARY_MODEL}__{split}__seed"] == seed)[0]
        f = av.reader_input(seq[rows], t[rows], L)
        _G["feat"][key] = f
    return f


def _task(spec: Tuple[int, str, str, str]) -> Dict[str, Any]:
    seed, reader, rep, variant = spec
    _G["feat"] = {}
    z = _G["z"]
    arch = _G["arch"]
    L = dict(REPS)[rep]

    tr = np.nonzero(z[f"{PRIMARY_MODEL}__trainord__seed"] == seed)[0]
    va = np.nonzero(z[f"{PRIMARY_MODEL}__valord__seed"] == seed)[0]
    te_sel = z[f"{PRIMARY_MODEL}__test__seed"] == seed
    te_idx = np.nonzero(te_sel)[0]
    held_te = au._pairs(z, PRIMARY_MODEL, "test", te_idx)
    masks = {m: {p: au._pop_masks(z, m, seed)[p][te_sel] for p in POPULATIONS}
             for m in (PRIMARY_MODEL, CONTROL_MODEL)}
    yte = z[f"{PRIMARY_MODEL}__test__{TARGET}"][te_sel].astype(int)
    yva = z[f"{PRIMARY_MODEL}__valord__{TARGET}"][va].astype(int)
    Xva = _feat("valord", seed, L)
    Xte = _feat("test", seed, L)

    if variant == "main":
        Xtr = _feat("trainord", seed, L)
        ytr = z[f"{PRIMARY_MODEL}__trainord__{TARGET}"][tr].astype(int)
        n_train = int(len(tr))
    else:
        held_tr = au._pairs(z, PRIMARY_MODEL, "trainord", tr)
        keep = ~held_tr
        Xtr = _feat("trainord", seed, L)[keep]
        ytr = z[f"{PRIMARY_MODEL}__trainord__{TARGET}"][tr[keep]].astype(int)
        n_train = int(keep.sum())

    def eval_state(state):
        pred_te = ax.predict_state(reader, state, Xte, None, NCLASS, av.INPUT_DIM, arch)
        pred_va = ax.predict_state(reader, state, Xva, None, NCLASS, av.INPUT_DIM, arch)
        node: Dict[str, Any] = {
            "val_acc": float((pred_va == yva).mean()),
            "acc": {f"{m}:{p}": (float((pred_te[masks[m][p]] == yte[masks[m][p]]).mean())
                                 if masks[m][p].any() else None)
                    for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}}
        if variant == "main":
            pred_tr = ax.predict_state(reader, state, Xtr, None, NCLASS, av.INPUT_DIM, arch)
            node["train_acc"] = float((pred_tr == ytr).mean())
        else:
            node["seen"] = {f"{m}:{p}": (float((pred_te[masks[m][p] & ~held_te] == yte[masks[m][p] & ~held_te]).mean())
                                          if (masks[m][p] & ~held_te).any() else None)
                            for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
            node["heldout"] = {f"{m}:{p}": (float((pred_te[masks[m][p] & held_te] == yte[masks[m][p] & held_te]).mean())
                                             if (masks[m][p] & held_te).any() else None)
                               for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
        return node

    out: Dict[str, Any] = {"seed": seed, "reader": reader, "rep": rep, "variant": variant,
                           "fits": {}, "events": {}}
    for cond_name in COND_NAMES:
        cond = COND_BY_NAME[cond_name]
        # V's exact deterministic reader-seed convention; the optimizer condition is NOT part of it,
        # so adam_ref and adamw share identical initial parameters.
        fit_seed = av.reader_seed(seed, reader, TARGET, rep, variant)
        init_h = _init_hash(reader, NCLASS, fit_seed, cond["init"], arch)
        fit = ab.train_ab(reader, Xtr, ytr, Xva, yva, NCLASS, fit_seed, UPDATES, cond, arch,
                          eval_every=EVAL_EVERY)
        out["fits"][cond_name] = {
            "cond": cond_name, "opt": cond["opt"], "init": cond["init"], "rep": rep, "variant": variant,
            "params": av.reader_param_count(reader, NCLASS, arch), "n_train": n_train,
            "init_hash": init_h,
            "best_val_acc": float(fit["best_val"][0]), "best_val_update": int(fit["best_val"][2]),
            "train_acc_final": float(fit["train_acc_final"]), "val_acc_final": float(fit["val_acc_final"]),
            "finite": bool(fit["finite"]),
            "best_val": eval_state(fit["best_val"][1]),
            "final": eval_state(fit["final"])}
        out["events"][cond_name] = fit["events"]
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _agg_cells(cells: List[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    node = {"params": cells[0]["params"], "n_train": _ms([c["n_train"] for c in cells]),
            "init_hash": cells[0]["init_hash"], "init_hashes": [c["init_hash"] for c in cells],
            "best_val_acc": _ms([c["best_val_acc"] for c in cells]),
            "best_val_update": _ms([c["best_val_update"] for c in cells]),
            "train_acc_final": _ms([c["train_acc_final"] for c in cells]),
            "val_acc_final": _ms([c["val_acc_final"] for c in cells]),
            "all_finite": bool(all(c["finite"] for c in cells))}
    bv = [c["best_val"] for c in cells]
    node["val_acc"] = _ms([b["val_acc"] for b in bv])
    node["acc"] = {pk: _ms([b["acc"][pk] for b in bv]) for pk in bv[0]["acc"]}
    if variant == "main":
        node["train_acc"] = _ms([b["train_acc"] for b in bv])
    else:
        node["seen"] = {pk: _ms([b["seen"][pk] for b in bv]) for pk in bv[0]["seen"]}
        node["heldout"] = {pk: _ms([b["heldout"][pk] for b in bv]) for pk in bv[0]["heldout"]}
    return node


def aggregate(parts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    by: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {
        (p["reader"], p["seed"], p["rep"], p["variant"]): p for p in parts}
    out: Dict[str, Any] = {}
    for reader in READERS:
        for rep in REP_LABELS:
            for cond_name in COND_NAMES:
                node: Dict[str, Any] = {}
                for variant in ("main", "heldout"):
                    cells = [by[(reader, s, rep, variant)]["fits"][cond_name] for s in SEEDS]
                    node[variant] = _agg_cells(cells, variant)
                out.setdefault(reader, {}).setdefault(rep, {})[cond_name] = node
    # trajectories: mean over seeds per (reader, cond_name, rep, variant)
    traj: Dict[str, Any] = {}
    for reader in READERS:
        for cond_name in COND_NAMES:
            for rep in REP_LABELS:
                for variant in ("main", "heldout"):
                    evs = [by[(reader, s, rep, variant)]["events"][cond_name] for s in SEEDS]
                    ups = [e["update"] for e in evs[0]]
                    traj.setdefault(f"{reader}|{cond_name}|{variant}", {})[rep] = {
                        "updates": ups,
                        "train_acc": [_ms([e[i]["train_acc"] for e in evs]) for i in range(len(ups))],
                        "val_acc": [_ms([e[i]["val_acc"] for e in evs]) for i in range(len(ups))]}
    return out, traj


def _cell(by, reader, seed, rep, variant, cond, *path):
    node = by[(reader, seed, rep, variant)]["fits"][cond]["best_val"]
    for p in path:
        node = node[p]
    return node


def paired_metrics(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-seed paired `adamw - adam_ref` deltas for Class-A and held-out Class-A accuracy."""
    by = {(p["reader"], p["seed"], p["rep"], p["variant"]): p for p in parts}
    out: Dict[str, Any] = {}
    for reader in READERS:
        out[reader] = {}
        for rep in REP_LABELS:
            aw_ca = [_cell(by, reader, s, rep, "main", "adamw", "acc", f"{PRIMARY_MODEL}:classA") for s in SEEDS]
            rf_ca = [_cell(by, reader, s, rep, "main", "adam_ref", "acc", f"{PRIMARY_MODEL}:classA") for s in SEEDS]
            aw_ho = [_cell(by, reader, s, rep, "heldout", "adamw", "heldout", f"{PRIMARY_MODEL}:classA") for s in SEEDS]
            rf_ho = [_cell(by, reader, s, rep, "heldout", "adam_ref", "heldout", f"{PRIMARY_MODEL}:classA") for s in SEEDS]
            ca = as_.paired(aw_ca, rf_ca)
            ho = as_.paired(aw_ho, rf_ho)
            ca["per_seed_delta"] = [None if (a is None or b is None) else float(a) - float(b)
                                    for a, b in zip(aw_ca, rf_ca)]
            ca["per_seed_adamw"] = aw_ca
            ca["per_seed_adam_ref"] = rf_ca
            ho["per_seed_delta"] = [None if (a is None or b is None) else float(a) - float(b)
                                    for a, b in zip(aw_ho, rf_ho)]
            ho["per_seed_adamw"] = aw_ho
            ho["per_seed_adam_ref"] = rf_ho
            out[reader][rep] = {"classA": ca, "heldout_classA": ho}
    return out


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

def classify_optimizer(paired: Dict[str, Any]) -> Dict[str, Any]:
    """AD-A / AD-B: material paired optimizer effect at L4 or Lfull."""
    rows: Dict[str, Any] = {}
    for reader in READERS:
        for rep in DECISIVE:
            for metric in ("classA", "heldout_classA"):
                pd = paired[reader][rep][metric]
                rows[f"{reader}|{rep}|{metric}"] = {
                    "mean_paired_delta": pd["mean_diff"], "adamw_mean": pd["a_mean"],
                    "adam_ref_mean": pd["b_mean"], "per_seed_delta": pd["per_seed_delta"],
                    "sign_positive": pd["sign_positive"], "sign_negative": pd["sign_negative"],
                    "cohens_dz": pd.get("cohens_dz"), "paired_t_p": pd.get("paired_t_p")}
    crossed = [k for k, v in rows.items()
               if v["mean_paired_delta"] is not None and v["mean_paired_delta"] >= MATERIAL_GAIN]
    tcn_crossed = [k for k in crossed if k.startswith("TCN")]
    verdict = "AD-A" if crossed else "AD-B"
    return {"verdict": verdict, "threshold": MATERIAL_GAIN, "rows": rows,
            "crossed": crossed, "tcn_crossed": tcn_crossed,
            "any_reader_crossed": bool(crossed),
            "consistent_across_seeds": {k: (rows[k]["sign_positive"] >= 4) for k in crossed},
            "rule_note": ("AD-A if at L4 or Lfull the paired adamw-adam_ref mean improvement is "
                          ">= +0.15 (V's material-gain convention) in Class-A y_true accuracy OR "
                          "held-out Class-A accuracy for either reader; else AD-B. A borderline / "
                          "seed-dependent result is reported explicitly and the bar is not moved.")}


def sequence_access(res: Dict[str, Any], u_base: Dict[str, Any], cond_name: str) -> Dict[str, Any]:
    """V's unchanged sequence-accessibility assessment (separate from the optimizer-effect axis)."""
    per_reader: Dict[str, Any] = {}
    for reader in READERS:
        entries = []
        for rep in DECISIVE:
            a = res["readers"][reader][rep][cond_name]["main"]["acc"][f"{PRIMARY_MODEL}:classA"]["mean"]
            h = res["readers"][reader][rep][cond_name]["heldout"]["heldout"][f"{PRIMARY_MODEL}:classA"]["mean"]
            entries.append((rep, a, h))
        best = max((e for e in entries if e[1] is not None), key=lambda e: e[1])
        rep, a, h = best
        bayes = u_base["models"][PRIMARY_MODEL]["bayes"]["classA"][rep]
        gap = (bayes - a) if (bayes is not None and a is not None) else None
        u_best = max(u_base["models"][PRIMARY_MODEL]["probes"]["classA"][r] for r in DECISIVE
                     if u_base["models"][PRIMARY_MODEL]["probes"]["classA"][r] is not None)
        gain = (a - u_best) if a is not None else None
        vA = bool(a is not None and h is not None and a >= RECOVER_BAR and h >= HELDOUT_MATERIAL)
        vB = bool(a is not None and h is not None and a <= HELDOUT_WEAK and h <= HELDOUT_WEAK
                  and gap is not None and gap >= BAYES_GAP_BAR)
        vC = bool((not vA) and (not vB) and gain is not None and gain >= MATERIAL_GAIN)
        per_reader[reader] = {"best_rep": rep, "classA": a, "heldout_classA": h, "bayes": bayes,
                              "bayes_minus_reader": gap, "u_mlp128_best": u_best, "gain_over_u": gain,
                              "V_A": vA, "V_B": vB, "V_C": vC,
                              "per_horizon": {e[0]: {"classA": e[1], "heldout_classA": e[2]} for e in entries}}
    # primary (E checkpoint, both readers): the V decision selects the best Class-A at a decisive horizon
    flat = [(rr, rep, per_reader[rr]["per_horizon"][rep]["classA"],
             per_reader[rr]["per_horizon"][rep]["heldout_classA"])
            for rr in READERS for rep in DECISIVE]
    flat = [f for f in flat if f[2] is not None]
    if not flat:
        return {"verdict": "V-partial/inconclusive", "best_reader": None, "best_rep": None,
                "classA": None, "heldout_classA": None, "bayes": None, "bayes_minus_reader": None,
                "u_mlp128_best": None, "gain_over_u": None,
                "conditions": {"V-A": False, "V-B": False, "V-C": False}, "per_reader": per_reader}
    rr, rep, a, h = max(flat, key=lambda f: f[2])
    bayes = u_base["models"][PRIMARY_MODEL]["bayes"]["classA"][rep]
    gap = (bayes - a) if (bayes is not None and a is not None) else None
    u_best = max(u_base["models"][PRIMARY_MODEL]["probes"]["classA"][r] for r in DECISIVE
                 if u_base["models"][PRIMARY_MODEL]["probes"]["classA"][r] is not None)
    gain = a - u_best
    vA = bool(a >= RECOVER_BAR and h is not None and h >= HELDOUT_MATERIAL)
    vB = bool(a <= HELDOUT_WEAK and h is not None and h <= HELDOUT_WEAK and gap is not None
              and gap >= BAYES_GAP_BAR)
    vC = bool((not vA) and (not vB) and gain >= MATERIAL_GAIN)
    if vA:
        verdict = "V-A_sequence_accessible"
    elif vB:
        verdict = "V-B_sequence_limited"
    elif vC:
        verdict = "V-C_partial_sequence_benefit"
    else:
        verdict = "V-partial/inconclusive"
    return {"verdict": verdict, "best_reader": rr, "best_rep": rep, "classA": a,
            "heldout_classA": h, "bayes": bayes, "bayes_minus_reader": gap,
            "u_mlp128_best": u_best, "gain_over_u": gain,
            "conditions": {"V-A": vA, "V-B": vB, "V-C": vC}, "per_reader": per_reader}


def load_v_historical(path: str) -> Dict[str, Any]:
    with open(path) as f:
        v = json.load(f)
    out: Dict[str, Any] = {"source": path, "verdict": v["primary_decision"]["verdict"], "readers": {}}
    for reader in READERS:
        out["readers"][reader] = {}
        for rep in REP_LABELS:
            node = v["readers"][reader]["y_true"][rep]
            out["readers"][reader][rep] = {
                "main_classA": node["main"]["acc"][f"{PRIMARY_MODEL}:classA"]["mean"],
                "main_classA_sd": node["main"]["acc"][f"{PRIMARY_MODEL}:classA"]["sd"],
                "heldout_classA": node["heldout"]["heldout"][f"{PRIMARY_MODEL}:classA"]["mean"],
                "heldout_classA_sd": node["heldout"]["heldout"][f"{PRIMARY_MODEL}:classA"]["sd"],
                "all_add": node["main"]["acc"][f"{PRIMARY_MODEL}:all_add"]["mean"],
                "train_acc": node["main"]["train_acc"]["mean"]}
    return out


def _drift(res: Dict[str, Any], v_hist: Dict[str, Any]) -> Dict[str, Any]:
    """Contemporaneous Adam/default bridge vs V's stored historical result (reported, never gating)."""
    cells: Dict[str, Any] = {}
    mx_ca = mx_ho = 0.0
    for reader in READERS:
        for rep in REP_LABELS:
            ad_ca = res["readers"][reader][rep]["adam_ref"]["main"]["acc"][f"{PRIMARY_MODEL}:classA"]["mean"]
            ad_ho = res["readers"][reader][rep]["adam_ref"]["heldout"]["heldout"][f"{PRIMARY_MODEL}:classA"]["mean"]
            h_ca = v_hist["readers"][reader][rep]["main_classA"]
            h_ho = v_hist["readers"][reader][rep]["heldout_classA"]
            d_ca = (ad_ca - h_ca) if (ad_ca is not None and h_ca is not None) else None
            d_ho = (ad_ho - h_ho) if (ad_ho is not None and h_ho is not None) else None
            cells[f"{reader}|{rep}"] = {"adam_ref_classA": ad_ca, "v_historical_classA": h_ca,
                                        "drift_classA": d_ca,
                                        "adam_ref_heldout": ad_ho, "v_historical_heldout": h_ho,
                                        "drift_heldout": d_ho}
            if d_ca is not None:
                mx_ca = max(mx_ca, abs(d_ca))
            if d_ho is not None:
                mx_ho = max(mx_ho, abs(d_ho))
    return {"max_abs_drift_classA": mx_ca, "max_abs_drift_heldout": mx_ho, "per_cell": cells,
            "note": ("AD's contemporaneous Adam/default bridge vs V's stored JSON; cross-environment "
                     "numerical drift is reported explicitly and never gates any verdict.")}


def _validate(meta: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    for ds in ("trainord", "valord", "test"):
        for mk in (PRIMARY_MODEL, CONTROL_MODEL):
            cc = meta.get("class_counts", {}).get(ds, {}).get(mk)
            if cc is not None and (len(cc) != NCLASS or any(int(c) <= 0 for c in cc)):
                problems.append(f"class coverage failure: {mk}/{ds}={cc}")
    for mk in (PRIMARY_MODEL, CONTROL_MODEL):
        for k, v in meta.get("disjointness", {}).get(mk, {}).items():
            if k.startswith(("trainord_vs", "valord_vs")) and int(v) != 0:
                problems.append(f"split disjointness failure: {mk}/{k}={v}")
    for k, rec in meta.get("target_derivation", {}).items():
        if not rec.get("exact", False):
            problems.append(f"y_true derivation failure: {k}={rec}")
    if not meta.get("history_causality"):
        problems.append("no history-causality verification recorded")
    return problems


def _collect(data: Dict[str, Any], jobs: int, readers: List[str], variants: List[str],
             reps: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    npz_path = data["path"]
    arch = av.architecture()
    reps = reps or REP_LABELS
    specs = [(s, R, rep, v) for s in SEEDS for R in readers for rep in reps for v in variants]
    t0 = time.perf_counter()
    if jobs <= 1:
        _init_worker(npz_path, arch)
        return [_task(sp) for sp in specs]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs, initializer=_init_worker, initargs=(npz_path, arch)) as pool:
        parts = []
        done = 0
        for part in pool.imap_unordered(_task, specs, chunksize=1):
            parts.append(part)
            done += 1
            if done % 20 == 0 or done == len(specs):
                print(f"  [{done}/{len(specs)}] fits-collected (each carries 2 optimizer conditions)",
                      flush=True)
    print(f"collection wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    parts.sort(key=lambda p: (p["reader"], p["seed"], p["rep"], p["variant"]))
    return parts


def assemble(data: Dict[str, Any], parts: List[Dict[str, Any]], u_base: Dict[str, Any],
             v_hist: Dict[str, Any]) -> Dict[str, Any]:
    arch = av.architecture()
    readers, traj = aggregate(parts)
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "readers": READERS,
        "target": TARGET, "n_classes": NCLASS, "reps": REP_LABELS, "decisive": DECISIVE,
        "populations": POPULATIONS, "variants": ["main", "heldout"],
        "conditions": [{"name": c["name"], "opt": c["opt"], "init": c["init"],
                        "qualified": c["name"] in QUALIFIED} for c in CONDITIONS],
        "qualified": QUALIFIED, "optimizer_defaults": ab.OPTIMIZER_DEFAULTS,
        "lr": LR, "weight_decay": WD, "updates": UPDATES, "eval_every": EVAL_EVERY,
        "reader_epochs": UPDATES, "train_ord_cap": TRAIN_ORD_CAP, "val_ord_cap": VAL_ORD_CAP,
        "held_out_pairs": sorted(au.HELD_OUT_PAIRS),
        "source_identity": data["meta"].get("source"), "class_counts": data["meta"].get("class_counts"),
        "target_derivation": data["meta"].get("target_derivation"),
        "heldout_pairs": data["meta"].get("heldout_pairs"),
        "disjointness": data["meta"].get("disjointness"),
        "history_causality": data["meta"].get("history_causality"),
        "checkpoints": data["meta"].get("models"), "n": data["meta"].get("n"),
        "thresholds": {"material_gain": MATERIAL_GAIN, "recover_bar": RECOVER_BAR,
                       "heldout_material": HELDOUT_MATERIAL, "heldout_weak": HELDOUT_WEAK,
                       "bayes_gap_bar": BAYES_GAP_BAR},
        "statistical_discipline": (
            "Five frozen E checkpoints are the independent units; F is a control. The two readers are "
            "analysis-only, fit on V's ordinary-ADD training pool (cap 1,500/seed), tuned on V's "
            "canonical VAL (cap 800/seed), evaluated once on V's untouched canonical TEST. For each "
            "(seed, reader, history, variant) the two optimizer conditions share V's exact deterministic "
            "initialization (the optimizer condition is not part of the seed); only the optimizer family "
            "differs. Selection is VAL-only; TEST never selects. Bayes ceilings are U's, unmodified.")}}
    res["architecture"] = arch
    res["readers"] = readers
    res["trajectory"] = traj
    res["per_seed"] = parts
    problems = _validate(data["meta"])
    res["validation_problems"] = problems
    res["paired"] = paired_metrics(parts) if not problems else {}
    res["primary_decision"] = classify_optimizer(res["paired"]) if not problems else {
        "verdict": "AD-INVALID", "rows": {}, "crossed": [], "validation_problems": problems}
    res["sequence_access"] = {c: sequence_access(res, u_base, c) for c in COND_NAMES} if not problems else {}
    res["bayes_check"] = {
        "classA_L4": u_base["models"][PRIMARY_MODEL]["bayes"]["classA"]["L4"],
        "classA_Lfull": u_base["models"][PRIMARY_MODEL]["bayes"]["classA"]["Lfull"]}
    res["v_historical"] = v_hist
    res["v_drift"] = _drift(res, v_hist) if not problems else {}
    # E/F control agreement (all-ADD populations should be identical)
    ef = 0.0
    for reader in READERS:
        for rep in REP_LABELS:
            for cond_name in COND_NAMES:
                for variant in ("main",):
                    a = res["readers"][reader][rep][cond_name][variant]["acc"]
                    ea = a[f"{PRIMARY_MODEL}:all_add"]["mean"]
                    fa = a[f"{CONTROL_MODEL}:all_add"]["mean"]
                    if ea is not None and fa is not None:
                        ef = max(ef, abs(ea - fa))
    res["E_F_max_abs_diff_alladd"] = ef
    return res


def build(data: Dict[str, Any], u_base: Dict[str, Any], v_hist: Dict[str, Any], jobs: int,
          readers: Optional[List[str]] = None, variants: Optional[List[str]] = None,
          reps: Optional[List[str]] = None) -> Dict[str, Any]:
    parts = _collect(data, jobs, readers or READERS, variants or ["main", "heldout"], reps)
    return assemble(data, parts, u_base, v_hist)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(res, u_base, out_dir, prefix="experiment_ad") -> List[str]:
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
    x = np.arange(len(REP_LABELS))
    cmap = {"adam_ref": "tab:gray", "adamw": "tab:blue"}

    def val(reader, rep, cond, variant, *path):
        node = R[reader][rep][cond][variant]
        for p in path:
            node = node[p]
        return nz(node["mean"])

    # 1 Class-A accuracy vs history (main)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for reader, col in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        for cond, ls in (("adam_ref", ":"), ("adamw", "-")):
            ax.plot(x, [val(reader, rep, cond, "main", "acc", f"{PRIMARY_MODEL}:classA") for rep in REP_LABELS],
                    marker="o", ls=ls, color=col, alpha=1.0 if cond == "adamw" else 0.55,
                    label=f"{reader} {cond}")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.set_xticks(x); ax.set_xticklabels(REP_LABELS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("Class-A TEST accuracy (y_true)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AD: Class-A ADD-target access vs history (E)")
    save(fig, "classA_vs_history")

    # 2 held-out operand accuracy vs history
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for reader, col in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        for cond, ls in (("adam_ref", ":"), ("adamw", "-")):
            ax.plot(x, [val(reader, rep, cond, "heldout", "heldout", f"{PRIMARY_MODEL}:classA") for rep in REP_LABELS],
                    marker="s", ls=ls, color=col, alpha=1.0 if cond == "adamw" else 0.55,
                    label=f"{reader} {cond}")
    ax.axhline(HELDOUT_MATERIAL, ls=":", c="k", lw=0.8, label="0.30 material")
    ax.axhline(HELDOUT_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xticks(x); ax.set_xticklabels(REP_LABELS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("held-out Class-A accuracy")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AD: held-out operand generalization vs history")
    save(fig, "heldout_vs_history")

    # 3 paired optimizer deltas (Class-A) per history
    P = res["paired"]
    for metric, fname, ylab in (("classA", "optimizer_delta_classA", "Class-A accuracy"),
                                ("heldout_classA", "optimizer_delta_heldout", "held-out Class-A accuracy")):
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        w = 0.35
        for i, (reader, col) in enumerate((("GRU", "tab:blue"), ("TCN", "tab:orange"))):
            means = [nz(P[reader][rep][metric]["mean_diff"]) for rep in REP_LABELS]
            sds = []
            for rep in REP_LABELS:
                d = np.array([v for v in P[reader][rep][metric]["per_seed_delta"] if v is not None], float)
                sds.append(float(d.std(ddof=1)) if d.size > 1 else 0.0)
            ax.bar(x + (i - 0.5) * w, means, w, yerr=sds, capsize=3, color=col, label=reader)
        ax.axhline(MATERIAL_GAIN, ls="--", c="k", lw=0.9, label="+0.15 material")
        ax.axhline(0, c="gray", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(REP_LABELS)
        ax.set_xlabel("causal history window"); ax.set_ylabel(f"paired adamw - adam_ref ({ylab})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"AD: optimizer effect on {ylab}")
        save(fig, fname)

    # 4 per-seed deltas at decisive horizons (TCN, Class-A and held-out)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    groups = [("classA", "L4"), ("classA", "Lfull"), ("heldout_classA", "L4"), ("heldout_classA", "Lfull")]
    labels = [f"{m}\n{rep}" for m, rep in groups]
    for gi, (m, rep) in enumerate(groups):
        deltas = P["TCN"][rep][m]["per_seed_delta"]
        xj = np.full(len(SEEDS), gi, float) + np.linspace(-0.18, 0.18, len(SEEDS))
        ax.scatter(xj, [nz(v) for v in deltas],
                   color="tab:purple" if m == "classA" else "tab:brown", zorder=3,
                   label=("Class-A" if m == "classA" else "held-out") if gi < 2 else None)
    ax.axhline(MATERIAL_GAIN, ls="--", c="k", lw=0.9, label="+0.15")
    ax.axhline(0, c="gray", lw=0.8)
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("paired per-seed delta (adamw - adam_ref)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AD: TCN per-seed optimizer deltas (five E seeds)")
    save(fig, "per_seed_delta")

    # 5 train/val trajectories (main, TCN, L4)
    T = res["trajectory"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for reader, col in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        for cond, ls in (("adam_ref", ":"), ("adamw", "-")):
            node = T.get(f"{reader}|{cond}|main", {}).get("L4")
            if not node:
                continue
            ax.plot(node["updates"], [v["mean"] for v in node["train_acc"]], ls=ls, color=col,
                    alpha=1.0 if cond == "adamw" else 0.55, label=f"{reader} {cond} train")
    ax.set_xlabel("full-batch optimizer updates"); ax.set_ylabel("training accuracy (main, L4)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AD: training trajectories — optimizer conditions (E, L4)")
    save(fig, "training_trajectory")

    # 6 fit vs generalization (Class-A) scatter
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for reader, col in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        for cond, mk in (("adam_ref", "x"), ("adamw", "o")):
            for rep in DECISIVE:
                xv = val(reader, rep, cond, "main", "train_acc")
                yv = val(reader, rep, cond, "main", "acc", f"{PRIMARY_MODEL}:classA")
                ax.scatter(xv, yv, color=col, marker=mk, s=70, zorder=3)
                ax.annotate(f"{reader[0]}{cond[-3:]}/{rep}", (xv, yv), fontsize=6,
                            xytext=(4, 4), textcoords="offset points")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.axhline(HELDOUT_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("training accuracy (train pool)"); ax.set_ylabel("Class-A TEST accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("AD: fitting vs Class-A generalization")
    save(fig, "fit_vs_generalization")

    # 7 Bayes ceiling vs reader (Class-A) — reuses U's existing ceiling, unmodified
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bayes_curve = [u_base["models"][PRIMARY_MODEL]["bayes"]["classA"][rep] for rep in REP_LABELS]
    ax.plot(x, bayes_curve, marker="^", ls="-", color="tab:green", label="Class-A Bayes (U)")
    for reader, col in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(x, [val(reader, rep, "adamw", "main", "acc", f"{PRIMARY_MODEL}:classA") for rep in REP_LABELS],
                marker="o", color=col, label=f"{reader} adamw")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.set_xticks(x); ax.set_xticklabels(REP_LABELS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("accuracy")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("AD: Bayes ceiling vs qualified reader (E, Class-A)")
    save(fig, "bayes_vs_reader")

    # 8 sequence-accessibility summary (adamw and adam_ref)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labs, ca_v, ho_v = [], [], []
    for cond in COND_NAMES:
        for rep in DECISIVE:
            for reader in READERS:
                labs.append(f"{reader}\n{rep}\n{cond}")
                ca_v.append(val(reader, rep, cond, "main", "acc", f"{PRIMARY_MODEL}:classA"))
                ho_v.append(val(reader, rep, cond, "heldout", "heldout", f"{PRIMARY_MODEL}:classA"))
    xi = np.arange(len(labs))
    ax.bar(xi - 0.2, ca_v, 0.4, label="Class-A", color="tab:blue")
    ax.bar(xi + 0.2, ho_v, 0.4, label="held-out", color="tab:purple")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.9, label="0.50 V-A bar")
    ax.axhline(HELDOUT_MATERIAL, ls=":", c="k", lw=0.9, label="0.30 held-out")
    ax.set_xticks(xi); ax.set_xticklabels(labs, fontsize=6, rotation=35, ha="right")
    ax.set_ylabel("TEST accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AD: sequence-accessibility assessment (V's criteria)")
    save(fig, "sequence_access_summary")

    # 9 V historical vs AD adam_ref bridge
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labs, hv, av_ = [], [], []
    V = res["v_historical"]
    for reader in READERS:
        for rep in REP_LABELS:
            labs.append(f"{reader}\n{rep}")
            hv.append(nz(V["readers"][reader][rep]["main_classA"]))
            av_.append(val(reader, rep, "adam_ref", "main", "acc", f"{PRIMARY_MODEL}:classA"))
    xi = np.arange(len(labs))
    ax.bar(xi - 0.2, hv, 0.4, label="V historical (Adam/default)", color="tab:gray")
    ax.bar(xi + 0.2, av_, 0.4, label="AD adam_ref (contemporaneous)", color="tab:blue")
    ax.set_xticks(xi); ax.set_xticklabels(labs, fontsize=6, rotation=35, ha="right")
    ax.set_ylabel("Class-A accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"AD: Adam/default bridge vs V (max |Δ|={res['v_drift']['max_abs_drift_classA']:.4f})")
    save(fig, "v_bridge")

    # 10 decision summary
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    d = res["primary_decision"]
    dn = res["sequence_access"]["adamw"]
    cats = ["TCN ΔClass-A\n(L4)", "TCN ΔClass-A\n(full)", "TCN Δheldout\n(L4)", "TCN Δheldout\n(full)",
            "adamw best\nClass-A", "adamw best\nheld-out"]
    rows = res["paired"]["TCN"]
    vals = [nz(rows["L4"]["classA"]["mean_diff"]), nz(rows["Lfull"]["classA"]["mean_diff"]),
            nz(rows["L4"]["heldout_classA"]["mean_diff"]), nz(rows["Lfull"]["heldout_classA"]["mean_diff"]),
            nz(dn["classA"]), nz(dn["heldout_classA"])]
    ax.bar(np.arange(len(cats)), vals,
           color=["tab:blue", "tab:blue", "tab:purple", "tab:purple", "tab:orange", "tab:brown"])
    ax.axhline(MATERIAL_GAIN, ls="--", c="k", lw=0.9, label="+0.15 material")
    ax.axhline(RECOVER_BAR, ls=":", c="gray", lw=0.9, label="0.50 V-A")
    ax.axhline(0, c="gray", lw=0.8)
    ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("delta / accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"AD: optimizer={d['verdict']}, sequence-access={dn['verdict']}")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AD analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--v-results", default="experiments/experiment_v_results.json")
    ap.add_argument("--u-results", default="experiments/experiment_u_results.json")
    ap.add_argument("--output", default="experiments/experiment_ad_results.json")
    ap.add_argument("--raw", default="experiments/experiment_ad_grid_raw.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_ad")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--readers", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--reps", nargs="*", default=None)
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--use-raw", action="store_true")
    args = ap.parse_args()
    data = load(args.npz)
    data["path"] = args.npz
    u_base = av.load_u_baseline(args.u_results)
    v_hist = load_v_historical(args.v_results)
    problems = _validate(data["meta"])
    if problems:
        print(json.dumps({"validation_problems": problems}, indent=2))
        res = assemble(data, [], u_base, v_hist)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"INVALID: {problems}; wrote {args.output}")
        return
    if args.use_raw and os.path.exists(args.raw):
        with open(args.raw) as f:
            parts = json.load(f)["parts"]
    elif args.collect_only:
        parts = _collect(data, args.jobs, args.readers or READERS,
                         args.variants or ["main", "heldout"], args.reps)
        os.makedirs(os.path.dirname(args.raw), exist_ok=True)
        with open(args.raw, "w") as f:
            json.dump({"parts": parts}, f)
        print(f"Wrote raw {args.raw} ({len(parts)} per-seed records x 2 conditions)")
        return
    else:
        parts = _collect(data, args.jobs, args.readers or READERS,
                         args.variants or ["main", "heldout"], args.reps)
        os.makedirs(os.path.dirname(args.raw), exist_ok=True)
        with open(args.raw, "w") as f:
            json.dump({"parts": parts}, f)
    t0 = time.perf_counter()
    res = assemble(data, parts, u_base, v_hist)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"optimizer verdict: {d['verdict']}  crossed={d['crossed']}")
    for k, v in d["rows"].items():
        print(f"  {k}: adamw={v['adamw_mean']} adam_ref={v['adam_ref_mean']} "
              f"delta={v['mean_paired_delta']} per_seed={v['per_seed_delta']}")
    for c in COND_NAMES:
        sa = res["sequence_access"][c]
        print(f"sequence-access [{c}]: {sa['verdict']} best={sa['best_reader']}/{sa['best_rep']} "
              f"classA={sa['classA']} heldout={sa['heldout_classA']} bayes-gap={sa['bayes_minus_reader']}")
    print(f"V drift: classA max|d|={res['v_drift']['max_abs_drift_classA']:.4f} "
          f"heldout max|d|={res['v_drift']['max_abs_drift_heldout']:.4f}")
    print(f"E/F all-ADD max|diff|={res['E_F_max_abs_diff_alladd']}")
    paths = make_figures(res, u_base, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
