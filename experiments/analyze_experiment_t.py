#!/usr/bin/env python3
"""Experiment T — analysis: upstream-representation recoverability on the Class-A hard tail.

S probed the frozen candidate network's INTERNAL activations and found no stage materially beats
the final candidate C_t (S-C). T asks the upstream question: which representation BEFORE the
candidate network first fails to carry generalizable information about the true written ADD value,
and which of the candidate input's sources (e_t vs h_reset) is responsible?

Predeclared activation stages (from the ACTUAL implementation; see run_experiment_t.py):
    raw_obs(21) enc_h1_pre(64) enc_h1_post(64) enc_out_pre(256) e_t(256)
    S_prev(212) gate_pre(212) f_gate(212) h_reset(212) cand_input(468) C_t(212)
Encoder path   : raw_obs -> enc_h1_pre -> enc_h1_post -> enc_out_pre -> e_t
State path     : S_prev -> gate_pre -> f_gate -> h_reset
Bridge         : cand_input = [h_reset, e_t] -> C_t

PRE-REGISTRATION (declared before inspecting any TEST aggregate)
---------------------------------------------------------------
* Activation set: all 11 stages are reported; none is dropped.
* Probe family (R/S's, fixed and not extended after results): linear ridge; MLP-16/32/64/128
  (tanh, Adam 3e-3, weight-decay 1e-2, 200 epochs, early stop on val, fixed seed, balanced class
  weights) — imported verbatim from analyze_experiment_s so the family is literally identical.
* Capacity x distribution: for NON-key stages {linear, MLP-32} x {D0, D0m, D1, D1x}; for the
  predeclared KEY stages {linear, MLP-16, MLP-32} x {D0, D0m, D1, D1x} plus {MLP-64, MLP-128} x
  {D1x} (bounded full sweep on the key stages).
* Distributions (R/S's): D0 ordinary, D0m size-matched ordinary, D1 designated hard-tail, D1x
  augmented i.i.d. hard pool. Evaluation is always the SAME untouched canonical-TEST Class-A tail.
* Held-out operand protocol: 20 pairs {(a,b): b in {0,5}} (all 10 targets represented); the probe
  is fit excluding them and evaluated on hard-test rows with held-out vs seen pairs (D0 and D1x).
* Localization verdict (checked in order; bars are declared conventions):
    T-MIX  subtype spread |wrap-nowrap| >= 0.15 at >= 2 KEY stages, or E/F verdict categories
           disagree.
    T-C    (source-fusion) e_t >= 0.50 AND h_reset >= 0.50 AND cand_input <= min(e_t,h_reset) - 0.10,
           with the same ordering preserved on held-out operands.
    T-B    (state-limited) encoder-path best >= state-path best + 0.10, encoder-path best >= 0.50,
           and the ordering preserved on held-out operands.
    T-A    (encoder-limited) encoder-path best <= 0.25 (weak) AND state-path best >= encoder best
           + 0.10 AND state best held-out >= encoder best held-out + 0.05 AND >= 4/5 seeds.
    T-D    (weak throughout) both pathways weak (<= 0.25) or no material difference, and no stage
           reaches the 0.50 recoverability bar, and held-out stays <= 0.25 at every stage.
    else   T-MIX.
No test-based stage/capacity selection: every predeclared stage and capacity is reported.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_t.py \
        --npz experiments/experiment_t_upstream_representation.npz \
        --output experiments/experiment_t_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_t
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

import analyze_experiment_s as as_  # noqa: E402  (probe family, verbatim)
import run_experiment_t as rt  # noqa: E402

# Reuse the EXACT R/S probe primitives (ridge_fit, mlp_fit, class_weights, param_count, mean_sd,
# paired, matched_control, test_masks, macro_acc, MLP_SEED). analyze_experiment_s also fixes the
# torch thread count for bit-reproducibility.

PRIMARY_MODEL = "E_candidate"
CONTROL_MODEL = "F_combined"
SEEDS = [42, 43, 44, 45, 46]
N_MOD = 10
N_RAW = 19

STAGES = rt.STAGES
STAGE_DIMS = rt.STAGE_DIMS
ENCODER_STAGES = rt.ENCODER_STAGES
STATE_STAGES = rt.STATE_STAGES
KEY_STAGES = rt.KEY_STAGES
BRIDGE_STAGES = rt.BRIDGE_STAGES
NONKEY_STAGES = [s for s in STAGES if s not in KEY_STAGES]

CAPS = [("linear", None), ("mlp16", 16), ("mlp32", 32), ("mlp64", 64), ("mlp128", 128)]
CAP_HIDDEN = dict(CAPS)
REGIMES = ["D0", "D0m", "D1", "D1x"]
KEY_CAPS_MAIN = ["linear", "mlp16", "mlp32"]
NONKEY_CAPS = ["linear", "mlp32"]
KEY_EXTRA_CAPS = ["mlp64", "mlp128"]
KEY_EXTRA_REGIMES = ["D1x"]
ARITH_CAPS = ["linear", "mlp32"]
ARITH_REGIMES = ["D0", "D1x"]
HELDOUT_CAPS = ["linear", "mlp32"]
HELDOUT_REGIMES = ["D0", "D1x"]
HELD_OUT_PAIRS = {(a, b) for a in range(10) for b in range(10) if b in (0, 5)}
SECONDARY_TARGETS = ["op_dest", "operand_a", "operand_b"]  # operation is constant (all ADD) here
SECONDARY_NCLASS = {"op_dest": 4, "operand_a": 10, "operand_b": 10}

CHANCE = 0.10
# decision bars (declared conventions)
RECOVER_BAR = 0.50
MAT_MARGIN = 0.10
MIX_BAR = 0.15
HELDOUT_WEAK = 0.25
HELDOUT_BAR = 0.30
HO_PERSIST = 0.05
MEMO_GAP = 0.20

_TRAIN_DS = {"D0": "trainord", "D0m": "trainord", "D1": "trainhard", "D1x": "hardextra"}
_VAL_DS = {"D0": "valord", "D0m": "valord", "D1": "valhard", "D1x": "valhard"}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def probes_for(stage: str) -> List[Tuple[str, str, Optional[int]]]:
    """Predeclared (regime, cap_name, hidden) probe list for a stage — a PURE function of the
    stage name (never of any test metric)."""
    out: List[Tuple[str, str, Optional[int]]] = []
    caps = KEY_CAPS_MAIN if stage in KEY_STAGES else NONKEY_CAPS
    for reg in REGIMES:
        for name in caps:
            out.append((reg, name, CAP_HIDDEN[name]))
    if stage in KEY_STAGES:
        for reg in KEY_EXTRA_REGIMES:
            for name in KEY_EXTRA_CAPS:
                out.append((reg, name, CAP_HIDDEN[name]))
    return out


# ---------------------------------------------------------------------------
# probes (analysis-only; primitives identical to Experiment R/S)
# ---------------------------------------------------------------------------

def _fit(z, m, seed, regime, stage, kind, hidden, lab="y_true", nclass=N_MOD,
         exclude_heldout=False) -> Dict[str, Any]:
    def rows(ds):
        return np.nonzero(z[f"{m}__{ds}__seed"] == seed)[0]
    tds = _TRAIN_DS[regime]
    vds = _VAL_DS[regime]
    tr = rows(tds)
    va = rows(vds)
    if regime == "D0m":
        n_hard = int((z[f"{m}__trainhard__seed"] == seed).sum())
        tr = tr[:n_hard]
    if exclude_heldout:
        held = _operand_pairs(z, m, tds, tr)
        tr = tr[~held]
    Xtr = z[f"{m}__{tds}__{stage}"][tr].astype(np.float64)
    ytr = z[f"{m}__{tds}__{lab}"][tr].astype(int)
    Xva = z[f"{m}__{vds}__{stage}"][va].astype(np.float64)
    yva = z[f"{m}__{vds}__{lab}"][va].astype(int)
    wc = as_.class_weights(ytr, nclass)
    if kind == "linear":
        pred, vacc, lam = as_.ridge_fit(Xtr, ytr, wc, Xva, yva, nclass)
    else:
        pred, vacc = as_.mlp_fit(Xtr, ytr, Xva, yva, hidden, wc, as_.MLP_SEED + seed * 100 + hidden, nclass)
        lam = None
    return {"predict": pred, "train_acc": float((pred(Xtr).argmax(1) == ytr).mean()),
            "val_acc": vacc, "n_train": len(tr), "n_val": len(va), "lambda": lam}


def _operand_pairs(z, m, ds, idx):
    oa = z[f"{m}__{ds}__operand_a"][idx]
    ob = z[f"{m}__{ds}__operand_b"][idx]
    return np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oa, ob)])


def evaluate(z, m, fitted, stage, hard_mask, correct_mask, matched_mask) -> Dict[str, Any]:
    pred = fitted["predict"]

    def acc(mask):
        if not mask.any():
            return None
        X = z[f"{m}__test__{stage}"][mask].astype(np.float64)
        y = z[f"{m}__test__y_true"][mask].astype(int)
        return float((pred(X).argmax(1) == y).mean())

    def macro(mask):
        if not mask.any():
            return None
        X = z[f"{m}__test__{stage}"][mask].astype(np.float64)
        y = z[f"{m}__test__y_true"][mask].astype(int)
        return as_.macro_acc(pred(X).argmax(1), y)

    wrap = z[f"{m}__test__wrap"].astype(bool)
    rep = z[f"{m}__test__repeated"].astype(bool)
    oa = z[f"{m}__test__operand_a"].astype(int)
    ob = z[f"{m}__test__operand_b"].astype(int)
    held = np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oa, ob)])
    hardtest = acc(hard_mask)
    return {
        "hardtest": hardtest, "hardtest_macro": macro(hard_mask),
        "recoverability": None if hardtest is None else (hardtest - CHANCE) / (1 - CHANCE),
        "hard_wrap": acc(hard_mask & wrap), "hard_nowrap": acc(hard_mask & ~wrap),
        "hard_repeated": acc(hard_mask & rep), "hard_isolated": acc(hard_mask & ~rep),
        "hard_heldout_pairs": acc(hard_mask & held), "hard_seen_pairs": acc(hard_mask & ~held),
        "correct_all": acc(correct_mask), "correct_matched": acc(matched_mask),
    }


def heldout_fit(z, m, seed, stage, name, regime) -> Dict[str, Any]:
    kind = "linear" if name == "linear" else "mlp"
    hidden = CAP_HIDDEN[name]
    f = _fit(z, m, seed, regime, stage, kind, hidden, exclude_heldout=True)
    sel = (z[f"{m}__test__seed"] == seed)
    hard = sel & (z[f"{m}__test__cand_ok"] == 0)
    X = z[f"{m}__test__{stage}"][hard].astype(np.float64)
    y = z[f"{m}__test__y_true"][hard].astype(int)
    h2 = _operand_pairs(z, m, "test", np.nonzero(hard)[0])
    p = f["predict"](X).argmax(1)
    return {"val_acc": f["val_acc"], "n_fit": f["n_train"],
            "acc_all": float((p == y).mean()),
            "acc_heldout": float((p[h2] == y[h2]).mean()) if h2.any() else None,
            "acc_seen": float((p[~h2] == y[~h2]).mean()) if (~h2).any() else None,
            "n_heldout": int(h2.sum())}


def geometry(z, m, seed) -> Dict[str, Any]:
    hard, correct = as_.test_masks(z, m, seed)
    out: Dict[str, Any] = {}
    ca = z[f"{m}__test__y_true"][hard].astype(int)
    for st in STAGES:
        X = z[f"{m}__test__{st}"][hard].astype(np.float64)
        Xc = z[f"{m}__test__{st}"][correct].astype(np.float64)
        mu_all = X.mean(0)
        within, between = 0.0, 0.0
        for c in range(N_MOD):
            mc = X[ca == c]
            if len(mc) == 0:
                continue
            muc = mc.mean(0)
            within += float(((mc - muc) ** 2).sum())
            between += float(len(mc) * ((muc - mu_all) ** 2).sum())
        within /= max(len(X) * X.shape[1], 1)
        between /= max(len(X) * X.shape[1], 1)
        cenA = X.mean(0)
        cenC = Xc.mean(0)
        nA = np.linalg.norm(cenA)
        nC = np.linalg.norm(cenC)
        cos = float(np.dot(cenA, cenC) / (nA * nC + 1e-12))
        out[st] = {
            "dim": int(X.shape[1]),
            "norm_hard": float(np.linalg.norm(X, axis=1).mean()),
            "norm_correct": float(np.linalg.norm(Xc, axis=1).mean()),
            "within_var": within, "between_var": between,
            "fisher": between / (within + 1e-12),
            "centroid_dist": float(np.linalg.norm(cenA - cenC) / (0.5 * (nA + nC) + 1e-12)),
            "centroid_cos": cos,
        }
    return out


def _target_acc(z, m, fitted, stage, mask, lab) -> Optional[float]:
    if not mask.any():
        return None
    X = z[f"{m}__test__{stage}"][mask].astype(np.float64)
    y = z[f"{m}__test__{lab}"][mask].astype(int)
    return float((fitted["predict"](X).argmax(1) == y).mean())


def analyze_seed(z, m, seed) -> Dict[str, Any]:
    hard_mask, correct_mask = as_.test_masks(z, m, seed)
    matched_idx = as_.matched_control(z, m, seed)
    matched_mask = np.zeros(len(hard_mask), bool)
    matched_mask[matched_idx] = True
    res: Dict[str, Any] = {"seed": seed, "n_hard": int(hard_mask.sum()),
                           "n_correct": int(correct_mask.sum()), "n_matched": int(matched_mask.sum()),
                           "caps": {}, "arithmetic": {}, "heldout_operand": {}, "secondary": {},
                           "geometry": {}}
    for regime in REGIMES:
        res["caps"][regime] = {}
        for stage in STAGES:
            res["caps"][regime][stage] = {}
            for reg, name, hidden in probes_for(stage):
                if reg != regime:
                    continue
                kind = "linear" if name == "linear" else "mlp"
                f = _fit(z, m, seed, regime, stage, kind, hidden)
                ev = evaluate(z, m, f, stage, hard_mask, correct_mask, matched_mask)
                ev.update({"train_acc": f["train_acc"], "val_acc": f["val_acc"], "n_train": f["n_train"],
                           "params": as_.param_count(kind, hidden, STAGE_DIMS[stage])})
                res["caps"][regime][stage][name] = ev
    # raw-sum vs modular (all stages; D0 and D1x; linear + mlp32)
    for regime in ARITH_REGIMES:
        res["arithmetic"][regime] = {}
        for stage in STAGES:
            res["arithmetic"][regime][stage] = {}
            for name in ARITH_CAPS:
                hidden = CAP_HIDDEN[name]
                kind = "linear" if name == "linear" else "mlp"
                fr = _fit(z, m, seed, regime, stage, kind, hidden, lab="raw_sum", nclass=N_RAW)
                res["arithmetic"][regime][stage][name] = _target_acc(z, m, fr, stage, hard_mask, "raw_sum")
    # held-out-operand generalization (all stages; D0 and D1x; linear + mlp32)
    for regime in HELDOUT_REGIMES:
        res["heldout_operand"][regime] = {}
        for stage in STAGES:
            res["heldout_operand"][regime][stage] = {}
            for name in HELDOUT_CAPS:
                res["heldout_operand"][regime][stage][name] = heldout_fit(z, m, seed, stage, name, regime)
    # explanatory secondary targets at D1x/mlp32 (operation is constant = ADD on this population)
    res["secondary"]["D1x_mlp32"] = {}
    for stage in KEY_STAGES:
        res["secondary"]["D1x_mlp32"][stage] = {}
        for lab in SECONDARY_TARGETS:
            f = _fit(z, m, seed, "D1x", stage, "mlp", 32, lab=lab, nclass=SECONDARY_NCLASS[lab])
            res["secondary"]["D1x_mlp32"][stage][lab] = _target_acc(z, m, f, stage, hard_mask, lab)
        res["secondary"]["D1x_mlp32"][stage]["operation"] = 1.0  # op is ADD for every row (constant)
    res["geometry"] = geometry(z, m, seed)
    return res


def mean_sd(vals: List[Optional[float]]) -> Dict[str, Any]:
    return as_.mean_sd(vals)


def paired(a: List[Optional[float]], b: List[Optional[float]]) -> Dict[str, Any]:
    return as_.paired(a, b)


def aggregate(per: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_seed": per, "caps": {}, "arithmetic": {}, "heldout_operand": {},
                           "secondary": {}, "geometry": {}}
    fields = ["hardtest", "hardtest_macro", "recoverability", "hard_wrap", "hard_nowrap", "hard_repeated",
              "hard_isolated", "hard_heldout_pairs", "hard_seen_pairs", "correct_all", "correct_matched",
              "train_acc", "val_acc", "n_train"]
    for stage in STAGES:
        for reg, name, _ in probes_for(stage):
            out["caps"].setdefault(reg, {}).setdefault(stage, {})
            out["caps"][reg][stage][name] = {
                f: mean_sd([p["caps"][reg][stage][name][f] for p in per]) for f in fields}
            out["caps"][reg][stage][name]["params"] = per[0]["caps"][reg][stage][name]["params"]
    for regime in ARITH_REGIMES:
        out["arithmetic"][regime] = {st: {name: mean_sd([p["arithmetic"][regime][st][name] for p in per])
                                          for name in ARITH_CAPS} for st in STAGES}
    for regime in HELDOUT_REGIMES:
        out["heldout_operand"][regime] = {
            st: {name: {f: mean_sd([p["heldout_operand"][regime][st][name][f] for p in per])
                        for f in ("val_acc", "acc_all", "acc_heldout", "acc_seen")}
                 for name in HELDOUT_CAPS} for st in STAGES}
    out["secondary"]["D1x_mlp32"] = {
        st: {lab: mean_sd([p["secondary"]["D1x_mlp32"][st][lab] for p in per])
             for lab in SECONDARY_TARGETS + ["operation"]} for st in KEY_STAGES}
    for st in STAGES:
        out["geometry"][st] = {f: mean_sd([p["geometry"][st][f] for p in per])
                               for f in ("norm_hard", "norm_correct", "within_var", "between_var",
                                         "fisher", "centroid_dist", "centroid_cos")}
    # chain scalars + adjacent changes along the full upstream trajectory (D1x/mlp32 and D0/mlp32)
    out["chain"] = {}
    for regime in ("D0", "D1x"):
        out["chain"][regime] = {}
        for st in STAGES:
            try:
                out["chain"][regime][st] = out["caps"][regime][st]["mlp32"]["hardtest"]["mean"]
            except KeyError:
                out["chain"][regime][st] = None
        drops = {}
        for i in range(len(STAGES) - 1):
            a, b = STAGES[i], STAGES[i + 1]
            va, vb = out["chain"][regime][a], out["chain"][regime][b]
            drops[f"{a}->{b}"] = (vb - va) if (va is not None and vb is not None) else None
        out["chain"][regime + "_drops"] = drops
    out["stage_vs_Ct"] = {
        st: paired([p["caps"]["D1x"][st]["mlp32"]["hardtest"] for p in per],
                   [p["caps"]["D1x"]["C_t"]["mlp32"]["hardtest"] for p in per]) for st in STAGES}
    return out


def _best_heldout(res: Dict[str, Any], stages: List[str], cap: str = "mlp32") -> Tuple[Optional[float], Optional[str]]:
    best, bst = None, None
    for s in stages:
        v = res["heldout_operand"]["D1x"][s][cap]["acc_heldout"]["mean"]
        if v is not None and (best is None or v > best):
            best, bst = v, s
    return best, bst


def _best_hardtest(res: Dict[str, Any], stages: List[str], cap: str = "mlp32") -> Tuple[Optional[float], Optional[str]]:
    best, bst = None, None
    for s in stages:
        try:
            v = res["caps"]["D1x"][s][cap]["hardtest"]["mean"]
        except KeyError:
            v = None
        if v is not None and (best is None or v > best):
            best, bst = v, s
    return best, bst


def _verdict_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    enc_best, enc_stage = _best_hardtest(res, ENCODER_STAGES)
    state_best, state_stage = _best_hardtest(res, STATE_STAGES)
    enc_ho, enc_ho_stage = _best_heldout(res, ENCODER_STAGES)
    state_ho, state_ho_stage = _best_heldout(res, STATE_STAGES)
    max_stage = max((v for v in res["chain"]["D1x"].values() if v is not None), default=None)
    max_ho = max((res["heldout_operand"]["D1x"][s]["mlp32"]["acc_heldout"]["mean"] for s in STAGES
                  if res["heldout_operand"]["D1x"][s]["mlp32"]["acc_heldout"]["mean"] is not None),
                 default=None)
    def hv(st):
        return res["caps"]["D1x"][st]["mlp32"]["hardtest"]["mean"]
    spread = {}
    for st in KEY_STAGES:
        w = res["caps"]["D0"][st]["mlp32"]["hard_wrap"]["mean"] if "mlp32" in res["caps"]["D0"][st] else None
        nw = res["caps"]["D0"][st]["mlp32"]["hard_nowrap"]["mean"] if "mlp32" in res["caps"]["D0"][st] else None
        if w is not None and nw is not None:
            spread[st] = abs(w - nw)
    return {"enc_best": enc_best, "enc_best_stage": enc_stage, "state_best": state_best,
            "state_best_stage": state_stage, "enc_best_heldout": enc_ho, "enc_best_heldout_stage": enc_ho_stage,
            "state_best_heldout": state_ho, "state_best_heldout_stage": state_ho_stage,
            "e_t": hv("e_t"), "h_reset": hv("h_reset"), "cand_input": hv("cand_input"), "C_t": hv("C_t"),
            "e_t_heldout": res["heldout_operand"]["D1x"]["e_t"]["mlp32"]["acc_heldout"]["mean"],
            "h_reset_heldout": res["heldout_operand"]["D1x"]["h_reset"]["mlp32"]["acc_heldout"]["mean"],
            "cand_input_heldout": res["heldout_operand"]["D1x"]["cand_input"]["mlp32"]["acc_heldout"]["mean"],
            "C_t_heldout": res["heldout_operand"]["D1x"]["C_t"]["mlp32"]["acc_heldout"]["mean"],
            "chain_d1x": res["chain"]["D1x"], "chain_d0": res["chain"]["D0"],
            "max_stage_d1x": max_stage, "max_heldout": max_ho, "subtype_spread_D0": spread}


def decide(res: Dict[str, Any]) -> Dict[str, Any]:
    met = {m: _verdict_metrics(res[m]) for m in (PRIMARY_MODEL, CONTROL_MODEL)}
    mE, mF = met[PRIMARY_MODEL], met[CONTROL_MODEL]
    per = res[PRIMARY_MODEL]["per_seed"]
    enc_seed = [max(p["caps"]["D1x"][st]["mlp32"]["hardtest"] for st in ENCODER_STAGES) for p in per]
    state_seed = [max(p["caps"]["D1x"][st]["mlp32"]["hardtest"] for st in STATE_STAGES) for p in per]
    diff = [a - b for a, b in zip(state_seed, enc_seed)]
    repro = int(sum(1 for d in diff if d > 0))
    state_minus_enc = (mE["state_best"] - mE["enc_best"]) if (mE["state_best"] is not None and mE["enc_best"] is not None) else None
    enc_minus_state = (mE["enc_best"] - mE["state_best"]) if (mE["state_best"] is not None and mE["enc_best"] is not None) else None
    ho_diff = (mE["state_best_heldout"] - mE["enc_best_heldout"]) if (mE["state_best_heldout"] is not None and mE["enc_best_heldout"] is not None) else None

    # T-MIX: subtype spread at >=2 KEY stages
    spread_over = sum(1 for v in mE["subtype_spread_D0"].values() if v is not None and v >= MIX_BAR)
    # T-C: both sources materially recoverable individually, fused weaker
    sC = bool(mE["e_t"] is not None and mE["h_reset"] is not None and mE["cand_input"] is not None
              and mE["e_t"] >= RECOVER_BAR and mE["h_reset"] >= RECOVER_BAR
              and mE["cand_input"] <= min(mE["e_t"], mE["h_reset"]) - MAT_MARGIN)
    # T-B: encoder materially stronger and materially recoverable, held-out supports
    sB = bool(enc_minus_state is not None and enc_minus_state >= MAT_MARGIN
              and mE["enc_best"] is not None and mE["enc_best"] >= RECOVER_BAR
              and ho_diff is not None and -ho_diff >= HO_PERSIST)
    # T-A: encoder weak, state materially stronger, held-out supports, reproducible
    sA = bool(mE["enc_best"] is not None and mE["enc_best"] <= HELDOUT_WEAK
              and state_minus_enc is not None and state_minus_enc >= MAT_MARGIN
              and ho_diff is not None and ho_diff >= HO_PERSIST and repro >= 4)
    # T-D: both weak or no material difference; nothing reaches the bar; held-out weak everywhere
    both_weak = (mE["enc_best"] is not None and mE["state_best"] is not None
                 and mE["enc_best"] <= HELDOUT_WEAK and mE["state_best"] <= HELDOUT_WEAK)
    no_material = (state_minus_enc is not None and abs(state_minus_enc) < MAT_MARGIN)
    nothing_recovers = (mE["max_stage_d1x"] is None or mE["max_stage_d1x"] < RECOVER_BAR)
    ho_weak = (mE["max_heldout"] is None or mE["max_heldout"] <= HELDOUT_WEAK)
    sD = bool((both_weak or no_material) and nothing_recovers and ho_weak)

    if spread_over >= 2:
        verdict = "T-MIX"
    elif sC:
        verdict = "T-C_source_fusion_limited"
    elif sB:
        verdict = "T-B_state_limited"
    elif sA:
        verdict = "T-A_encoder_limited"
    elif sD:
        verdict = "T-D_weak_throughout"
    else:
        verdict = "T-MIX"
    return {"verdict": verdict, "state_minus_encoder": state_minus_enc,
            "encoder_minus_state": enc_minus_state, "heldout_state_minus_encoder": ho_diff,
            "reproducible_seeds_state_gt_encoder": repro,
            "subtype_stages_over_mix_bar": spread_over,
            "conditions": {"T-A": sA, "T-B": sB, "T-C": sC, "T-D": sD},
            "E_metrics": mE, "F_metrics": mF,
            "thresholds": {"recover_bar": RECOVER_BAR, "mat_margin": MAT_MARGIN, "mix_bar": MIX_BAR,
                           "heldout_weak": HELDOUT_WEAK, "heldout_bar": HELDOUT_BAR,
                           "ho_persist": HO_PERSIST, "memo_gap": MEMO_GAP},
            "rule_note": ("T-MIX first (subtype spread >= 0.15 at >= 2 KEY stages or E/F categories "
                          "disagree); T-C if e_t AND h_reset >= 0.50 with cand_input >= 0.10 weaker; "
                          "T-B if encoder best >= state best + 0.10 and encoder best >= 0.50 with "
                          "held-out support; T-A if encoder best <= 0.25 and state best >= encoder "
                          "best + 0.10 with held-out support (>= 4/5 seeds); T-D if both pathways "
                          "weak/no material difference, nothing reaches 0.50, and held-out <= 0.25 "
                          "everywhere. Bars are declared conventions; continuous effects reported.")}


def build(data: Dict[str, Any]) -> Dict[str, Any]:
    z = data["z"]
    meta = data["meta"]
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS,
        "checkpoints": meta["models"], "splits": meta["n"],
        "encoder": meta.get("encoder"), "candidate_network": meta.get("candidate_network"),
        "stages": STAGES, "stage_dims": STAGE_DIMS, "encoder_stages": ENCODER_STAGES,
        "state_stages": STATE_STAGES, "key_stages": KEY_STAGES, "bridge_stages": BRIDGE_STAGES,
        "caps": [(n, h) for n, h in CAPS], "regimes": REGIMES,
        "key_extra_caps": KEY_EXTRA_CAPS, "key_extra_regimes": KEY_EXTRA_REGIMES,
        "heldout_pairs": sorted(HELD_OUT_PAIRS),
        "thresholds": {"recover_bar": RECOVER_BAR, "mat_margin": MAT_MARGIN, "mix_bar": MIX_BAR,
                       "heldout_weak": HELDOUT_WEAK, "heldout_bar": HELDOUT_BAR,
                       "ho_persist": HO_PERSIST, "memo_gap": MEMO_GAP},
        "statistical_discipline": ("Five frozen E checkpoints are the independent units; F is a "
                                   "control. Probes are analysis-only, fit on a regime's train, tuned "
                                   "on its val, evaluated once on the untouched test hard tail. Every "
                                   "predeclared stage and capacity is reported; no test-based "
                                   "activation or capacity selection.")}}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        per = [analyze_seed(z, m, s) for s in SEEDS]
        res[m] = aggregate(per)
    dec = decide(res)
    # E/F category agreement: flag disagreement as T-MIX
    mF = dec["F_metrics"]
    def _cat(m):
        enc_best = m["enc_best"]
        state_best = m["state_best"]
        if enc_best is None or state_best is None:
            return None
        if enc_best <= HELDOUT_WEAK and state_best <= HELDOUT_WEAK:
            return "weak"
        if state_best <= HELDOUT_WEAK and enc_best >= RECOVER_BAR:
            return "encoder"
        return "mixed"
    agree = (_cat(dec["E_metrics"]) == _cat(mF))
    if not agree:
        dec["verdict"] = "T-MIX"
        dec["E_F_disagreement"] = True
    else:
        dec["E_F_disagreement"] = False
    res["E_vs_F"] = {st: paired([p["caps"]["D1x"][st]["mlp32"]["hardtest"]
                                 for p in res[PRIMARY_MODEL]["per_seed"]],
                                [p["caps"]["D1x"][st]["mlp32"]["hardtest"]
                                 for p in res[CONTROL_MODEL]["per_seed"]]) for st in STAGES}
    res["primary_decision"] = dec
    return res


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_t") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    rE = res[PRIMARY_MODEL]
    sx = np.arange(len(STAGES))

    def s1x(st):
        return rE["caps"]["D1x"][st]["mlp32"]["hardtest"]["mean"]

    def s0(st):
        return rE["caps"]["D0"][st]["mlp32"]["hardtest"]["mean"]

    def slin1x(st):
        return rE["caps"]["D1x"][st]["linear"]["hardtest"]["mean"]

    # 1 upstream recoverability by stage (D0/lin, D1x/lin, D1x/mlp32)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(sx - 0.27, [s0(st) for st in STAGES], 0.27, label="D0/lin", color="tab:blue")
    ax.bar(sx, [slin1x(st) for st in STAGES], 0.27, label="D1x/lin", color="tab:green")
    ax.bar(sx + 0.27, [s1x(st) for st in STAGES], 0.27, label="D1x/mlp32", color="tab:orange")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.9, label="0.50 bar")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    for i, st in enumerate(STAGES):
        if st in ENCODER_STAGES:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:blue", alpha=0.05)
        elif st in STATE_STAGES:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:red", alpha=0.05)
    ax.set_title("E: upstream recoverability by stage (blue=encoder, red=state)")
    save(fig, "upstream_recoverability_by_stage")

    # 2 encoder-stage capacity curves (D1x)
    names = [n for n, _ in CAPS]
    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for st in ENCODER_STAGES:
        ys = []
        for n in names:
            try:
                ys.append(rE["caps"]["D1x"][st][n]["hardtest"]["mean"])
            except KeyError:
                ys.append(np.nan)
        ax.plot(xs, ys, marker="o", label=st)
    ax.set_xticks(xs); ax.set_xticklabels(names); ax.set_ylabel("hard-test accuracy (D1x)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("E: encoder-stage capacity curves (D1x)")
    save(fig, "encoder_capacity_curves")

    # 3 state-path capacity curves (D1x)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for st in STATE_STAGES + ["cand_input", "C_t"]:
        ys = []
        for n in names:
            try:
                ys.append(rE["caps"]["D1x"][st][n]["hardtest"]["mean"])
            except KeyError:
                ys.append(np.nan)
        ax.plot(xs, ys, marker="o", label=st)
    ax.set_xticks(xs); ax.set_xticklabels(names); ax.set_ylabel("hard-test accuracy (D1x)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("E: state-path + bridge capacity curves (D1x)")
    save(fig, "state_capacity_curves")

    # 4 source decomposition e_t vs h_reset vs cand_input vs C_t
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    srcs = ["e_t", "S_prev", "h_reset", "cand_input", "C_t"]
    x4 = np.arange(len(srcs))
    ax.bar(x4, [s1x(s) for s in srcs], color=["tab:blue", "tab:red", "tab:purple", "tab:brown", "tab:orange"])
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.9, label="0.50 bar")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(x4); ax.set_xticklabels(srcs, rotation=25, ha="right")
    ax.set_ylabel("hard-test accuracy (D1x/mlp32)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: source decomposition")
    save(fig, "source_decomposition")

    # 5 ordinary vs hard by stage
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(sx - 0.2, [s0(st) for st in STAGES], 0.4, label="D0 ordinary", color="tab:blue")
    ax.bar(sx + 0.2, [s1x(st) for st in STAGES], 0.4, label="D1x hard", color="tab:orange")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: ordinary- vs hard-trained by upstream stage")
    save(fig, "ordinary_vs_hard_by_stage")

    # 6 held-out operand by stage
    fig, ax = plt.subplots(figsize=(11, 4.6))
    seen = [rE["heldout_operand"]["D1x"][st]["mlp32"]["acc_seen"]["mean"] for st in STAGES]
    ho = [rE["heldout_operand"]["D1x"][st]["mlp32"]["acc_heldout"]["mean"] for st in STAGES]
    ax.bar(sx - 0.2, seen, 0.4, label="seen pairs", color="tab:gray")
    ax.bar(sx + 0.2, ho, 0.4, label="held-out pairs", color="tab:purple")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("hard-test accuracy (D1x/mlp32)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: held-out vs seen operand pairs by stage")
    save(fig, "heldout_operand_by_stage")

    # 7 seen - held-out gap by stage
    fig, ax = plt.subplots(figsize=(11, 4.3))
    gaps = [(a - b) if (a is not None and b is not None) else np.nan for a, b in zip(seen, ho)]
    ax.bar(sx, gaps, color="tab:red")
    ax.axhline(MEMO_GAP, ls="--", c="k", lw=0.9, label=f"{MEMO_GAP} bar")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("seen - held-out accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: seen-vs-held-out gap by stage")
    save(fig, "seen_vs_heldout_gap")

    # 8 gate_pre vs gate_post
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for i, reg in enumerate(("D0", "D1x")):
        pre = rE["caps"][reg]["gate_pre"]["mlp32"]["hardtest"]["mean"]
        post = rE["caps"][reg]["f_gate"]["mlp32"]["hardtest"]["mean"]
        ax.bar(i - 0.2, pre, 0.4, label="gate_pre" if i == 0 else None, color="tab:blue")
        ax.bar(i + 0.2, post, 0.4, label="f_gate (post)" if i == 0 else None, color="tab:orange")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["D0/mlp32", "D1x/mlp32"])
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8)
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: gate preactivation vs postactivation")
    save(fig, "gate_pre_vs_gate_post")

    # 9 h_reset vs S_prev
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for i, reg in enumerate(("D0", "D1x")):
        sp = rE["caps"][reg]["S_prev"]["mlp32"]["hardtest"]["mean"]
        hr = rE["caps"][reg]["h_reset"]["mlp32"]["hardtest"]["mean"]
        ax.bar(i - 0.2, sp, 0.4, label="S_prev" if i == 0 else None, color="tab:red")
        ax.bar(i + 0.2, hr, 0.4, label="h_reset" if i == 0 else None, color="tab:purple")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["D0/mlp32", "D1x/mlp32"])
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8)
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: S_prev vs h_reset")
    save(fig, "h_reset_vs_S_prev")

    # 10 raw-sum vs modular by stage
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(sx - 0.2, [s1x(st) for st in STAGES], 0.4, label="modular (D1x/mlp32)", color="tab:blue")
    ax.bar(sx + 0.2, [rE["arithmetic"]["D1x"][st]["mlp32"]["mean"] for st in STAGES], 0.4,
           label="raw sum (D1x/mlp32)", color="tab:orange")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: modular vs raw-sum target by upstream stage")
    save(fig, "rawsum_vs_modular_by_stage")

    # 11 wrap vs no-wrap
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(sx - 0.2, [rE["caps"]["D0"][st]["mlp32"]["hard_wrap"]["mean"] for st in STAGES], 0.4,
           label="wrap (D0)", color="tab:red")
    ax.bar(sx + 0.2, [rE["caps"]["D0"][st]["mlp32"]["hard_nowrap"]["mean"] for st in STAGES], 0.4,
           label="no-wrap (D0)", color="tab:cyan")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E (D0): wrap vs no-wrap by upstream stage")
    save(fig, "wrap_nowrap")

    # 12 repeated vs isolated
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(sx - 0.2, [rE["caps"]["D0"][st]["mlp32"]["hard_repeated"]["mean"] for st in STAGES], 0.4,
           label="repeated (D0)", color="tab:green")
    ax.bar(sx + 0.2, [rE["caps"]["D0"][st]["mlp32"]["hard_isolated"]["mean"] for st in STAGES], 0.4,
           label="isolated (D0)", color="tab:olive")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E (D0): repeated vs isolated ADD by upstream stage")
    save(fig, "repeated_isolated")

    # 13 candidate-correct vs Class-A controls (all-stage + matched)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(sx - 0.2, [rE["caps"]["D1x"][st]["mlp32"]["correct_all"]["mean"] for st in STAGES], 0.4,
           label="all candidate-correct", color="tab:green")
    ax.bar(sx + 0.2, [rE["caps"]["D1x"][st]["mlp32"]["correct_matched"]["mean"] for st in STAGES], 0.4,
           label="matched candidate-correct", color="tab:olive")
    ax.plot(sx, [s1x(st) for st in STAGES], "k^--", ms=5, label="Class-A (D1x/mlp32)")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: candidate-correct vs Class-A controls by stage")
    save(fig, "candidate_correct_vs_classA")

    # 14 full upstream information trajectory
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(sx, [rE["chain"]["D1x"][st] for st in STAGES], marker="o", label="D1x/mlp32", color="tab:orange")
    ax.plot(sx, [rE["chain"]["D0"][st] for st in STAGES], marker="s", label="D0/mlp32", color="tab:blue")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.9, label="0.50 bar")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E: full upstream information trajectory")
    save(fig, "information_trajectory")

    # 15 per-seed source recoverability (D1x/mlp32)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for si, s in enumerate(SEEDS):
        ax.plot(sx, [rE["per_seed"][si]["caps"]["D1x"][st]["mlp32"]["hardtest"] for st in STAGES],
                marker="o", label=str(s), alpha=0.8)
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=40, ha="right")
    ax.set_ylabel("hard-test accuracy (D1x/mlp32)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("E: per-seed upstream recoverability")
    save(fig, "per_seed_source")

    # 16 source-level localization summary
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = ["encoder best", "state best", "cand_input", "C_t"]

    def _ho(st):
        return rE["heldout_operand"]["D1x"][st]["mlp32"]["acc_heldout"]["mean"]

    enc_ho_stage = max(ENCODER_STAGES, key=_ho)
    state_ho_stage = max(STATE_STAGES, key=_ho)
    vals = [max(s1x(st) for st in ENCODER_STAGES), max(s1x(st) for st in STATE_STAGES),
            s1x("cand_input"), s1x("C_t")]
    hos = [_ho(enc_ho_stage), _ho(state_ho_stage), _ho("cand_input"), _ho("C_t")]
    x16 = np.arange(4)
    ax.bar(x16 - 0.2, vals, 0.4, label="D1x/mlp32", color="tab:orange")
    ax.bar(x16 + 0.2, hos, 0.4, label="D1x held-out", color="tab:purple")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.9, label="0.50 bar")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(x16); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: source-level localisation summary")
    save(fig, "localization_summary")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment T analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_t_upstream_representation.npz")
    ap.add_argument("--output", default="experiments/experiment_t_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_t")
    args = ap.parse_args()
    data = load(args.npz)
    res = build(data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"E: {d['verdict']} | state-enc={d['state_minus_encoder']} ho_diff="
          f"{d['heldout_state_minus_encoder']} repro={d['reproducible_seeds_state_gt_encoder']}")
    m = d["E_metrics"]
    print(f"  enc_best={m['enc_best']:.3f} ({m['enc_best_stage']})  "
          f"state_best={m['state_best']:.3f} ({m['state_best_stage']})  "
          f"cand_input={m['cand_input']:.3f}  C_t={m['C_t']:.3f}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
