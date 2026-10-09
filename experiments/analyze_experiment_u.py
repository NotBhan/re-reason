#!/usr/bin/env python3
"""Experiment U — analysis: raw-observation / task-encoding localisation of the Class-A information.

T found the frozen observation encoder is structurally operand-blind (it depends only on x_t) and
that no upstream frozen representation is robustly recoverable (T-D). U asks the representation-
INDEPENDENT question:

    Does the raw causal observation history x_0..x_t uniquely determine the current operands/target,
    and how much history is required before the information becomes (a) exactly identifiable and
    (b) accessible to a bounded learned probe?

Two evidence families are reported separately (spec §3):
  * EXACT interface evidence — identical-history collisions, unique-history counts, Bayes-optimal
    accuracy, conditional entropy, collision/ambiguity mass, max multiplicity.
  * LEARNED probe evidence — the R/S/T probe family (linear ridge; MLP-16/32/64/128) on the raw
    observation history, evaluated on the untouched canonical TEST populations.

Predeclared causal history grid (spec §4): L in {1, 2, 4, 8, 16} and the full causal prefix (T=10).
A window always ends at the current timestep t; early timesteps use the causal prefix left-padded
with an explicit binary validity mask (declared before analysis). No future observation is used.

PRE-REGISTRATION (declared before inspecting any TEST aggregate)
---------------------------------------------------------------
* Representations: raw (=L1), H2, H4, H8, H16, full_prefix (flattened one-hot window + mask; the
  PRIMARY probe input is RAW observation vectors only — no state, register values, reconstructed
  operands, labels, candidate logits, gate values, or hidden features).
* Probe family (R/S/T's, fixed, not extended after results; imported verbatim from
  analyze_experiment_s): linear ridge; MLP-16/32/64/128 (tanh, Adam 3e-3, weight-decay 1e-2,
  200 epochs, best-on-val state, fixed seed, balanced class weights).
* Fit on ordinary TRAIN ADD transitions (cap 1500/seed), tune on ordinary VAL (cap 800/seed),
  evaluate ONCE on the untouched canonical TEST; λ / capacity selection uses VAL only.
* Populations: Class-A hard (frozen candidate wrong), matched candidate-correct, all
  candidate-correct, all ADD. Targets: y_true (primary), operand_a, operand_b, raw_sum,
  op_dest; operation is constant (ADD-only population) and reported as structurally non-informative.
* Held-out operand protocol: 20 pairs {(a,b): b in {0,5}} excluded from fitting.
* Decision classes (checked in order; bars are declared conventions):
    U-MIX  E/F categories disagree, or wrap/no-wrap target-Bayes spread >= 0.15 at >= 2 lengths.
    U-B    Bayes_target(full prefix) < 0.99 (substantial collisions remain).
    U-D    Bayes_target(full) >= 0.99 AND best full-prefix learned probe held-out <= 0.25 AND
           (Bayes - probe) gap >= 0.30.
    U-A    Bayes_target(full) >= 0.99 AND best full-prefix learned probe >= 0.50 AND its held-out
           accuracy >= 0.30.
    U-C    Bayes_target(L=1) <= 0.50 AND the Bayes curve is non-decreasing (tol 0.02) and reaches
           >= 0.99 at a finite length L* <= full.
    else U-MIX.
No test-based history/capacity selection: every predeclared history length and capacity is reported.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_u.py \
        --npz experiments/experiment_u_task_interface.npz \
        --output experiments/experiment_u_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_u
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_experiment_s as as_  # noqa: E402  (probe family, verbatim)
import run_experiment_u as ru  # noqa: E402

PRIMARY_MODEL = "E_candidate"
CONTROL_MODEL = "F_combined"
SEEDS = [42, 43, 44, 45, 46]
T_MAX = ru.T_MAX
D_OBS = ru.OBS_DIM

# (representation label, window length). "Lfull" = whole causal prefix over T=10.
REPS: List[Tuple[str, int]] = [("L1", 1), ("L2", 2), ("L4", 4), ("L8", 8), ("L16", 16),
                               ("Lfull", ru.FULL_PREFIX_LENGTH)]
REP_LABELS = [r for r, _ in REPS]
CAPS = [("linear", None), ("mlp16", 16), ("mlp32", 32), ("mlp64", 64), ("mlp128", 128)]
CAP_HIDDEN = dict(CAPS)
ARITH_CAPS = ["linear", "mlp32"]
HELD_OUT_CAPS = ["linear", "mlp32"]
HELD_OUT_PAIRS = {(a, b) for a in range(10) for b in range(10) if b in (0, 5)}
TARGETS = {"y_true": 10, "operand_a": 10, "operand_b": 10, "raw_sum": 19, "op_dest": 4}
POPS = ["classA", "matched_correct", "correct_all", "all_add"]
CHANCE = 0.10
N_MOD = 10
N_RAW = 19
# decision bars (declared conventions)
RECOVER_BAR = 0.50
HELDOUT_WEAK = 0.25
HELDOUT_MATERIAL = 0.30
MAT_MARGIN = 0.10
MEMO_GAP = 0.20
MIX_BAR = 0.15
BAYES_FULL = 0.99


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


# ---------------------------------------------------------------------------
# causal history construction (no future leakage)
# ---------------------------------------------------------------------------

def build_history(seq_obs: np.ndarray, t: np.ndarray, L: int,
                  T: int = T_MAX, D: int = D_OBS) -> Tuple[np.ndarray, np.ndarray]:
    """Left-padded causal window of the last L observations ending at t, plus a validity mask.

    H_L(t) = [x_{t-L+1}, ..., x_t]; early timesteps use the available causal prefix (never future
    padding). Raises if a construction would use an observation index > t.
    """
    N = seq_obs.shape[0]
    full = seq_obs.reshape(N, T, D)
    X = np.zeros((N, L * D), np.float32)
    M = np.zeros((N, L), np.float32)
    for i in range(N):
        ti = int(t[i])
        j0 = ti - L + 1
        if j0 < 0:
            j0 = 0
        if j0 > ti:
            raise ValueError(f"history start {j0} > current timestep {ti}")
        win = full[i, j0:ti + 1]           # indices j0..ti, all <= t
        if win.shape[0] > L:
            raise ValueError("window longer than L")
        nn = win.shape[0]
        X[i, (L - nn) * D:] = win.reshape(-1)
        M[i, L - nn:] = 1.0
    return X, M


def history_features(seq_obs: np.ndarray, t: np.ndarray, L: int) -> np.ndarray:
    X, M = build_history(seq_obs, t, L)
    return np.concatenate([X, M], axis=1)


# ---------------------------------------------------------------------------
# exact collision / Bayes analysis (representation-independent)
# ---------------------------------------------------------------------------

def exact_bayes(keys: List[bytes], y: np.ndarray, nclass: int) -> Dict[str, Any]:
    """Bayes-optimal accuracy from exact history lookup + collision diagnostics."""
    groups: Dict[bytes, np.ndarray] = defaultdict(lambda: np.zeros(nclass, int))
    for k, yy in zip(keys, y):
        groups[k][int(yy)] += 1
    total = len(y)
    correct = 0
    coll = 0
    ambig = 0
    maxmult = 0
    ent = 0.0
    for c in groups.values():
        s = int(c.sum())
        correct += int(c.max())
        if s > 1:
            coll += s
        if int((c > 0).sum()) > 1:
            ambig += s
        maxmult = max(maxmult, s)
        p = c / s
        ent += float(-np.sum(p[p > 0] * np.log2(p[p > 0]))) * (s / total)
    return {"bayes_acc": correct / total if total else None, "n_unique": len(groups),
            "collision_rate": coll / total if total else None, "ambiguous_mass": ambig / total if total else None,
            "max_multiplicity": maxmult, "mean_target_entropy": ent, "n": total}


def _keys(feats: np.ndarray) -> List[bytes]:
    return [feats[i].tobytes() for i in range(feats.shape[0])]


# ---------------------------------------------------------------------------
# probes (analysis-only; primitives identical to R/S/T)
# ---------------------------------------------------------------------------

def _fit_probe(hist: Dict[Tuple[str, str], np.ndarray], z, m, seed, rep, kind, hidden,
               lab="y_true", nclass=N_MOD, exclude_heldout=False) -> Dict[str, Any]:
    tr = np.nonzero(z[f"{m}__trainord__seed"] == seed)[0]
    va = np.nonzero(z[f"{m}__valord__seed"] == seed)[0]
    if exclude_heldout:
        held = _pairs(z, m, "trainord", tr)
        tr = tr[~held]
    Xtr = hist[("trainord", rep)][tr].astype(np.float64)
    ytr = z[f"{m}__trainord__{lab}"][tr].astype(int)
    Xva = hist[("valord", rep)][va].astype(np.float64)
    yva = z[f"{m}__valord__{lab}"][va].astype(int)
    wc = as_.class_weights(ytr, nclass)
    if kind == "linear":
        pred, vacc, lam = as_.ridge_fit(Xtr, ytr, wc, Xva, yva, nclass)
    else:
        pred, vacc = as_.mlp_fit(Xtr, ytr, Xva, yva, hidden, wc, as_.MLP_SEED + seed * 100 + hidden, nclass)
        lam = None
    return {"predict": pred, "train_acc": float((pred(Xtr).argmax(1) == ytr).mean()),
            "val_acc": vacc, "n_train": len(tr), "lambda": lam}


def _pairs(z, m, ds, idx):
    oa = z[f"{m}__{ds}__operand_a"][idx]
    ob = z[f"{m}__{ds}__operand_b"][idx]
    return np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oa, ob)])


def _acc(hist, z, m, rep, mask, pred, lab="y_true") -> Optional[float]:
    if not mask.any():
        return None
    X = hist[("test", rep)][mask].astype(np.float64)
    y = z[f"{m}__test__{lab}"][mask].astype(int)
    return float((pred(X).argmax(1) == y).mean())


def _pop_masks(z, m, seed) -> Dict[str, np.ndarray]:
    sel = z[f"{m}__test__seed"] == seed
    ok = z[f"{m}__test__cand_ok"].astype(bool)
    mm = np.zeros(len(sel), bool)
    mm[as_.matched_control(z, m, seed)] = True
    return {"classA": sel & ~ok, "correct_all": sel & ok, "all_add": sel, "matched_correct": mm}


def prep_histories(z, m) -> Dict[Tuple[str, str], np.ndarray]:
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for split in ("trainord", "valord", "test"):
        seq = z[f"{m}__{split}__seq_obs"]
        t = z[f"{m}__{split}__t"]
        for rep, L in REPS:
            out[(split, rep)] = history_features(seq, t, L)
    return out


def reconstruct_from_prefix(seq_obs: np.ndarray, t: np.ndarray, T: int = T_MAX,
                            D: int = D_OBS) -> Dict[str, np.ndarray]:
    """Exact reference control: deterministically replay registers from x_0..x_{t-1} and read the
    operands/target at t. Used ONLY as a validity control (never as probe input)."""
    N = seq_obs.shape[0]
    full = seq_obs.reshape(N, T, D)
    oa = np.zeros(N, int)
    ob = np.zeros(N, int)
    y = np.zeros(N, int)
    dest = np.zeros(N, int)
    src = np.zeros(N, int)
    for i in range(N):
        ti = int(t[i])
        regs = [0, 0, 0, 0]
        for j in range(ti):
            x = full[i, j]
            op = int(np.argmax(x[:3]))
            ra = int(np.argmax(x[3:7]))
            rb = int(np.argmax(x[7:11]))
            imm = int(np.argmax(x[11:21]))
            if op == 0:
                regs[ra] = imm
            elif op == 1:
                regs[ra], regs[rb] = regs[rb], regs[ra]
            elif op == 2:
                regs[ra] = (regs[ra] + regs[rb]) % 10
        x = full[i, ti]
        ra = int(np.argmax(x[3:7]))
        rb = int(np.argmax(x[7:11]))
        dest[i] = ra
        src[i] = rb
        oa[i] = regs[ra]
        ob[i] = regs[rb]
        y[i] = (regs[ra] + regs[rb]) % 10
    return {"operand_a": oa, "operand_b": ob, "y_true": y, "dest": dest, "src": src}


def analyze_seed(z, m, seed, hist) -> Dict[str, Any]:
    pops = _pop_masks(z, m, seed)
    res: Dict[str, Any] = {"seed": seed,
                           "n": {p: int(pops[p].sum()) for p in POPS},
                           "bayes": {}, "probes": {}, "heldout": {}, "rawsum": {}, "operands": {}}
    # ---- exact Bayes per (population, target, representation)
    for p in POPS:
        mask = pops[p]
        res["bayes"][p] = {}
        for rep, _ in REPS:
            keys = _keys(hist[("test", rep)][mask])
            res["bayes"][p][rep] = {tgt: exact_bayes(keys, z[f"{m}__test__{tgt}"][mask].astype(int), nc)
                                    for tgt, nc in TARGETS.items()}
    # ---- learned probes on the primary target (y_true): fit once per (rep, cap), score all pops
    res["probes"] = {p: {rep: {} for rep, _ in REPS} for p in POPS}
    for rep, _ in REPS:
        for name, hidden in CAPS:
            kind = "linear" if name == "linear" else "mlp"
            f = _fit_probe(hist, z, m, seed, rep, kind, hidden)
            for p in POPS:
                res["probes"][p][rep][name] = {"acc": _acc(hist, z, m, rep, pops[p], f["predict"]),
                                               "train_acc": f["train_acc"], "val_acc": f["val_acc"],
                                               "n_train": f["n_train"]}
    # ---- held-out operand control (classA and all_add), linear + mlp32: fit once, score both pops
    for p in ("classA", "all_add"):
        res["heldout"][p] = {rep: {} for rep, _ in REPS}
    oper_a = z[f"{m}__test__operand_a"]
    oper_b = z[f"{m}__test__operand_b"]
    is_held = np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oper_a, oper_b)])
    for rep, _ in REPS:
        for name in HELD_OUT_CAPS:
            hidden = CAP_HIDDEN[name]
            kind = "linear" if name == "linear" else "mlp"
            f = _fit_probe(hist, z, m, seed, rep, kind, hidden, exclude_heldout=True)
            for p in ("classA", "all_add"):
                mask = pops[p]
                res["heldout"][p][rep][name] = {
                    "seen": _acc(hist, z, m, rep, mask & ~is_held, f["predict"]),
                    "heldout": _acc(hist, z, m, rep, mask & is_held, f["predict"]),
                    "all": _acc(hist, z, m, rep, mask, f["predict"])}
    # ---- raw-sum vs modular (all_add), linear + mlp32
    for rep, _ in REPS:
        res["rawsum"][rep] = {}
        for name in ARITH_CAPS:
            hidden = CAP_HIDDEN[name]
            kind = "linear" if name == "linear" else "mlp"
            f = _fit_probe(hist, z, m, seed, rep, kind, hidden, lab="raw_sum", nclass=N_RAW)
            res["rawsum"][rep][name] = _acc(hist, z, m, rep, pops["all_add"], f["predict"], lab="raw_sum")
    # ---- operand probes (all_add), mlp32
    for rep, _ in REPS:
        res["operands"][rep] = {}
        for lab in ("operand_a", "operand_b"):
            f = _fit_probe(hist, z, m, seed, rep, "mlp", 32, lab=lab, nclass=N_MOD)
            res["operands"][rep][lab] = _acc(hist, z, m, rep, pops["all_add"], f["predict"], lab=lab)
    return res


def mean_sd(vals):
    return as_.mean_sd(vals)


def paired(a, b):
    return as_.paired(a, b)


def aggregate(per: List[Dict[str, Any]], pooled: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_seed": per, "pooled": pooled, "bayes": {}, "probes": {}, "heldout": {},
                           "rawsum": {}, "operands": {}}
    for p in POPS:
        out["bayes"][p] = {}
        for rep, _ in REPS:
            out["bayes"][p][rep] = {tgt: {f: mean_sd([s["bayes"][p][rep][tgt][f] for s in per])
                                          for f in ("bayes_acc", "n_unique", "collision_rate",
                                                    "ambiguous_mass", "max_multiplicity",
                                                    "mean_target_entropy")} for tgt in TARGETS}
    for p in POPS:
        out["probes"][p] = {}
        for rep, _ in REPS:
            out["probes"][p][rep] = {name: {f: mean_sd([s["probes"][p][rep][name][f] for s in per])
                                            for f in ("acc", "train_acc", "val_acc")}
                                     for name, _ in CAPS}
    for p in ("classA", "all_add"):
        out["heldout"][p] = {}
        for rep, _ in REPS:
            out["heldout"][p][rep] = {name: {f: mean_sd([s["heldout"][p][rep][name][f] for s in per])
                                             for f in ("seen", "heldout", "all")}
                                      for name in HELD_OUT_CAPS}
    for rep, _ in REPS:
        out["rawsum"][rep] = {name: mean_sd([s["rawsum"][rep][name] for s in per]) for name in ARITH_CAPS}
        out["operands"][rep] = {lab: mean_sd([s["operands"][rep][lab] for s in per])
                                for lab in ("operand_a", "operand_b")}
    return out


def pooled_bayes(z, m, hist) -> Dict[str, Any]:
    sel = z[f"{m}__test__seed"] >= 0
    ok = z[f"{m}__test__cand_ok"].astype(bool)
    masks = {"classA": sel & ~ok, "correct_all": sel & ok, "all_add": sel}
    out: Dict[str, Any] = {}
    for p, mask in masks.items():
        out[p] = {}
        for rep, _ in REPS:
            keys = _keys(hist[("test", rep)][mask])
            out[p][rep] = {tgt: exact_bayes(keys, z[f"{m}__test__{tgt}"][mask].astype(int), nc)
                           for tgt, nc in TARGETS.items()}
    return out


def pooled_subtype(z, m, hist) -> Dict[str, Any]:
    """Wrap / no-wrap / repeated / isolated target-Bayes curves on the all-ADD test population."""
    sel = z[f"{m}__test__seed"] >= 0
    wrap = z[f"{m}__test__wrap"].astype(bool)
    rep = z[f"{m}__test__repeated"].astype(bool)
    masks = {"wrap": sel & wrap, "nowrap": sel & ~wrap, "repeated": sel & rep, "isolated": sel & ~rep}
    out: Dict[str, Any] = {}
    for name, mask in masks.items():
        out[name] = {}
        for rep_lbl, _ in REPS:
            keys = _keys(hist[("test", rep_lbl)][mask])
            out[name][rep_lbl] = {tgt: exact_bayes(keys, z[f"{m}__test__{tgt}"][mask].astype(int), nc)
                                  for tgt, nc in TARGETS.items()}
    return out


def _verdict_metrics(res) -> Dict[str, Any]:
    def bcurve(tgt):
        return {rep: res["pooled"]["all_add"][rep][tgt]["bayes_acc"] for rep, _ in REPS}
    bt, ba, bb = bcurve("y_true"), bcurve("operand_a"), bcurve("operand_b")
    best_probe = max(res["probes"]["all_add"][rep][cap]["acc"]["mean"]
                     for rep, _ in REPS for cap, _ in CAPS
                     if res["probes"]["all_add"][rep][cap]["acc"]["mean"] is not None)
    best_full_probe = max(res["probes"]["all_add"]["Lfull"][cap]["acc"]["mean"]
                          for cap, _ in CAPS
                          if res["probes"]["all_add"]["Lfull"][cap]["acc"]["mean"] is not None)
    best_full_ho = max(res["heldout"]["all_add"]["Lfull"][cap]["heldout"]["mean"]
                       for cap in HELD_OUT_CAPS
                       if res["heldout"]["all_add"]["Lfull"][cap]["heldout"]["mean"] is not None)
    L_star = next((rep for rep, _ in REPS if bt[rep] is not None and bt[rep] >= BAYES_FULL), None)
    L1_star = next((rep for rep, _ in REPS if ba[rep] is not None and ba[rep] >= 0.99), None)
    L2_star = next((rep for rep, _ in REPS if bb[rep] is not None and bb[rep] >= 0.99), None)
    spread = {}
    for rep, _ in REPS:
        w = res["subtype"]["wrap"][rep]["y_true"]["bayes_acc"]
        nw = res["subtype"]["nowrap"][rep]["y_true"]["bayes_acc"]
        spread[rep] = abs(w - nw) if (w is not None and nw is not None) else None
    return {"bayes_target_curve": bt, "bayes_operand_a_curve": ba, "bayes_operand_b_curve": bb,
            "bayes_target_L1": bt["L1"], "bayes_target_full": bt["Lfull"],
            "bayes_Lstar_target": L_star, "bayes_Lstar_operand_a": L1_star,
            "bayes_Lstar_operand_b": L2_star,
            "best_probe_all_add": best_probe, "best_full_probe": best_full_probe,
            "best_full_probe_heldout": best_full_ho,
            "full_gap": (bt["Lfull"] - best_full_probe) if (bt["Lfull"] is not None and best_full_probe is not None) else None,
            "subtype_spread_D0": spread}


def decide(res: Dict[str, Any]) -> Dict[str, Any]:
    mE = _verdict_metrics(res[PRIMARY_MODEL])
    mF = _verdict_metrics(res[CONTROL_MODEL])
    curve = [mE["bayes_target_curve"][rep] for rep, _ in REPS]
    nondec = all((curve[i + 1] is None or curve[i] is None or curve[i + 1] >= curve[i] - 0.02)
                 for i in range(len(curve) - 1))
    full_bayes = mE["bayes_target_full"]
    probe = mE["best_full_probe"]
    ho = mE["best_full_probe_heldout"]
    gap = mE["full_gap"]
    sB = bool(full_bayes is not None and full_bayes < BAYES_FULL)
    sD = bool(full_bayes is not None and full_bayes >= BAYES_FULL and ho is not None
              and ho <= HELDOUT_WEAK and gap is not None and gap >= 0.30)
    sA = bool(full_bayes is not None and full_bayes >= BAYES_FULL and probe is not None
              and probe >= RECOVER_BAR and ho is not None and ho >= HELDOUT_MATERIAL)
    sC = bool(mE["bayes_target_L1"] is not None and mE["bayes_target_L1"] <= RECOVER_BAR
              and nondec and mE["bayes_Lstar_target"] is not None)
    spread_over = sum(1 for v in mE["subtype_spread_D0"].values() if v is not None and v >= MIX_BAR)
    if spread_over >= 2:
        verdict = "U-MIX"
    elif sA:
        verdict = "U-B_interface_incomplete"
    elif sD:
        verdict = "U-D_interface_identifiable_probe_limited"
    elif sC:
        verdict = "U-C_history_dependent"
    else:
        verdict = "U-MIX"
    return {"verdict": verdict, "E_metrics": mE, "F_metrics": mF,
            "conditions": {"U-A": sA, "U-B": sB, "U-C": sC, "U-D": sD},
            "nondecreasing_bayes": nondec, "subtype_stages_over_mix_bar": spread_over,
            "thresholds": {"recover_bar": RECOVER_BAR, "heldout_weak": HELDOUT_WEAK,
                           "heldout_material": HELDOUT_MATERIAL, "mat_margin": MAT_MARGIN,
                           "memo_gap": MEMO_GAP, "mix_bar": MIX_BAR, "bayes_full": BAYES_FULL},
            "rule_note": ("U-MIX first (E/F categories disagree or wrap/no-wrap spread >=0.15 at "
                          ">=2 lengths); U-B if Bayes_target(full) < 0.99; U-D if Bayes_target(full) "
                          ">=0.99 and best full-prefix probe held-out <=0.25 with gap >=0.30; U-A if "
                          "Bayes_target(full) >=0.99 and best full-prefix probe >=0.50 with held-out "
                          ">=0.30; U-C if Bayes_target(L=1) <=0.50 and the Bayes curve is "
                          "non-decreasing and reaches >=0.99 at finite L*. Bars are declared "
                          "conventions; continuous values reported.")}


def build(data: Dict[str, Any]) -> Dict[str, Any]:
    z = data["z"]
    meta = data["meta"]
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS,
        "checkpoints": meta["models"], "splits": meta["n"],
        "obs_schema": meta.get("obs_schema"), "history_lengths": ru.HISTORY_LENGTHS,
        "full_prefix_length": ru.FULL_PREFIX_LENGTH, "reps": REP_LABELS,
        "caps": [(n, h) for n, h in CAPS], "populations": POPS, "targets": list(TARGETS),
        "heldout_pairs": sorted(HELD_OUT_PAIRS),
        "thresholds": {"recover_bar": RECOVER_BAR, "heldout_weak": HELDOUT_WEAK,
                       "heldout_material": HELDOUT_MATERIAL, "mat_margin": MAT_MARGIN,
                       "memo_gap": MEMO_GAP, "mix_bar": MIX_BAR, "bayes_full": BAYES_FULL},
        "statistical_discipline": ("Five frozen E checkpoints are the independent units; F is a "
                                   "control. Raw-observation/history probes are analysis-only, fit on "
                                   "ordinary TRAIN ADD transitions, tuned on VAL, evaluated once on "
                                   "the untouched canonical TEST. Every predeclared history length "
                                   "and capacity is reported; no test-based selection. Exact Bayes "
                                   "collision statistics are computed independently of any probe.")}}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        hist = prep_histories(z, m)
        per = [analyze_seed(z, m, s, hist) for s in SEEDS]
        res[m] = aggregate(per, pooled_bayes(z, m, hist))
        res[m]["subtype"] = pooled_subtype(z, m, hist)
    dec = decide(res)
    # E/F category agreement -> U-MIX on disagreement
    def _cat(m):
        full = m["bayes_target_full"]
        if full is None:
            return None
        if full < BAYES_FULL:
            return "incomplete"
        if m["best_full_probe"] is not None and m["best_full_probe"] >= RECOVER_BAR \
                and m["best_full_probe_heldout"] is not None and m["best_full_probe_heldout"] >= HELDOUT_MATERIAL:
            return "complete"
        return "probe_limited"
    agree = (_cat(dec["E_metrics"]) == _cat(dec["F_metrics"]))
    if not agree:
        dec["verdict"] = "U-MIX"
        dec["E_F_disagreement"] = True
    else:
        dec["E_F_disagreement"] = False
    res["E_vs_F"] = {rep: paired([s["probes"]["all_add"][rep]["mlp32"]["acc"] for s in res[PRIMARY_MODEL]["per_seed"]],
                                 [s["probes"]["all_add"][rep]["mlp32"]["acc"] for s in res[CONTROL_MODEL]["per_seed"]])
                     for rep, _ in REPS}
    res["primary_decision"] = dec
    return res


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_u") -> List[str]:
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
    labels = REP_LABELS
    x = np.arange(len(labels))
    Ls = [L for _, L in REPS]

    def bt(pop="all_add", tgt="y_true"):
        return [rE["pooled"][pop][rep][tgt]["bayes_acc"] for rep in labels]

    def probe(pop, cap):
        return [rE["probes"][pop][rep][cap]["acc"]["mean"] for rep in labels]

    def ho(pop, cap):
        return [rE["heldout"][pop][rep][cap]["heldout"]["mean"] for rep in labels]

    def seen(pop, cap):
        return [rE["heldout"][pop][rep][cap]["seen"]["mean"] for rep in labels]

    # 1 Bayes ceiling vs history length (target, operands, raw sum)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for tgt, c in (("y_true", "tab:blue"), ("operand_a", "tab:orange"), ("operand_b", "tab:green"),
                   ("raw_sum", "tab:red")):
        ax.plot(x, bt("all_add", tgt), marker="o", label=tgt, color=c)
    ax.plot(x, [rE["pooled"]["classA"][rep]["y_true"]["bayes_acc"] for rep in labels], marker="s",
            ls="--", label="target (Class-A)", color="tab:purple")
    ax.axhline(0.99, ls=":", c="k", lw=0.8, label="0.99")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("Bayes ceiling")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("U: exact Bayes ceiling vs history length")
    save(fig, "bayes_ceiling_vs_history")

    # 2 learned probe accuracy vs history
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for cap, _ in CAPS:
        ax.plot(x, probe("all_add", cap), marker="o", label=f"{cap} (all-ADD)")
    ax.plot(x, probe("classA", "mlp32"), marker="s", ls="--", color="k", label="mlp32 (Class-A)")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("TEST accuracy")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("U: learned probe accuracy vs history length")
    save(fig, "probe_accuracy_vs_history")

    # 3 Bayes vs probe gap
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    b = bt("all_add", "y_true"); p = probe("all_add", "mlp32")
    ax.plot(x, b, marker="o", label="Bayes ceiling", color="tab:blue")
    ax.plot(x, p, marker="s", label="best MLP-32 probe", color="tab:orange")
    ax.fill_between(x, p, b, color="tab:red", alpha=0.15, label="gap")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("U: Bayes ceiling vs learned probe (gap)")
    save(fig, "bayes_vs_probe_gap")

    # 4-6 operand/target Bayes vs history
    for name, tgt in (("operand_a_bayes_vs_history", "operand_a"),
                      ("operand_b_bayes_vs_history", "operand_b"),
                      ("target_bayes_vs_history", "y_true")):
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        ax.plot(x, bt("all_add", tgt), marker="o", label="all-ADD", color="tab:blue")
        ax.plot(x, [rE["pooled"]["classA"][rep][tgt]["bayes_acc"] for rep in labels], marker="s",
                ls="--", label="Class-A", color="tab:purple")
        ax.plot(x, [rE["pooled"]["correct_all"][rep][tgt]["bayes_acc"] for rep in labels], marker="^",
                ls=":", label="candidate-correct", color="tab:green")
        ax.axhline(0.99, ls=":", c="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
        ax.set_xlabel("causal history window"); ax.set_ylabel("Bayes ceiling")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title(f"U: {tgt} Bayes ceiling vs history")
        save(fig, name)

    # 7 modular vs raw-sum vs history
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(x, bt("all_add", "y_true"), marker="o", label="modular target (Bayes)", color="tab:blue")
    ax.plot(x, bt("all_add", "raw_sum"), marker="s", label="raw sum (Bayes)", color="tab:orange")
    ax.plot(x, [rE["rawsum"][rep]["mlp32"]["mean"] for rep in labels], marker="^", ls="--",
            label="raw sum (MLP-32 probe)", color="tab:red")
    ax.plot(x, [rE["probes"]["all_add"][rep]["mlp32"]["acc"]["mean"] for rep in labels], marker="v",
            ls="--", label="modular (MLP-32 probe)", color="tab:cyan")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("accuracy")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("U: modular vs raw-sum vs history")
    save(fig, "modular_vs_rawsum_vs_history")

    # 8 current-observation collision distribution (L=1)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    st = rE["pooled"]["all_add"]["L1"]
    vals = [st[t]["bayes_acc"] for t in ("y_true", "operand_a", "operand_b", "raw_sum", "op_dest")]
    ax.bar(np.arange(5), vals, color="tab:blue")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="target chance 0.10")
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(["target", "op_a", "op_b", "raw_sum", "dest"])
    ax.set_ylabel("L=1 Bayes ceiling"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("U: current-observation (L=1) exact Bayes ceiling")
    save(fig, "current_observation_collision")

    # 9 history collision metrics by L
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(x, [rE["pooled"]["all_add"][rep]["y_true"]["collision_rate"] for rep in labels],
            marker="o", label="collision rate")
    ax.plot(x, [rE["pooled"]["all_add"][rep]["y_true"]["ambiguous_mass"] for rep in labels],
            marker="s", label="ambiguous mass")
    ax.plot(x, [1.0 - rE["pooled"]["all_add"][rep]["y_true"]["bayes_acc"] for rep in labels],
            marker="^", label="1 - Bayes")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("fraction")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("U: history collision metrics vs length (all-ADD)")
    save(fig, "history_collision_by_L")

    # 10 held-out operand accuracy vs history
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for cap in HELD_OUT_CAPS:
        ax.plot(x, seen("all_add", cap), marker="o", label=f"seen ({cap})")
        ax.plot(x, ho("all_add", cap), marker="s", ls="--", label=f"held-out ({cap})")
    ax.axhline(HELDOUT_WEAK, ls=":", c="k", lw=0.8, label="0.25 weak")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("accuracy (all-ADD)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("U: held-out vs seen operand pairs vs history")
    save(fig, "heldout_operand_vs_history")

    # 11 seen-vs-heldout gap vs history
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for pop in ("all_add", "classA"):
        gaps = []
        for rep in labels:
            s = rE["heldout"][pop][rep]["mlp32"]["seen"]["mean"]
            h = rE["heldout"][pop][rep]["mlp32"]["heldout"]["mean"]
            gaps.append((s - h) if (s is not None and h is not None) else np.nan)
        ax.plot(x, gaps, marker="o", label=pop)
    ax.axhline(MEMO_GAP, ls="--", c="k", lw=0.9, label=f"{MEMO_GAP} bar")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel("causal history window")
    ax.set_ylabel("seen - held-out (mlp32)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("U: seen-vs-held-out gap vs history")
    save(fig, "seen_vs_heldout_gap_vs_history")

    # 12 Class-A vs candidate-correct vs all-ADD (Bayes target)
    def bc(pop):
        if pop in rE["pooled"]:
            return [rE["pooled"][pop][rep]["y_true"]["bayes_acc"] for rep in labels]
        return [rE["bayes"][pop][rep]["y_true"]["bayes_acc"]["mean"] for rep in labels]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for pop, c, m_ in (("all_add", "tab:blue", "o"), ("classA", "tab:purple", "s"),
                       ("correct_all", "tab:green", "^"), ("matched_correct", "tab:olive", "d")):
        ax.plot(x, bc(pop), marker=m_, label=pop, color=c)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("Bayes ceiling (target)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("U: Bayes ceiling by population")
    save(fig, "classA_vs_correct_vs_alladd")

    # 13 wrap/no-wrap vs history (target Bayes, all-ADD) — computed from stored per-row subtype keys
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for name, key in (("wrap", "wrap"), ("no-wrap", "nowrap")):
        ax.plot(x, [rE["subtype"][key][rep]["y_true"]["bayes_acc"] for rep in labels], marker="o", label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("target Bayes ceiling")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("U: wrap vs no-wrap target Bayes vs history")
    save(fig, "wrap_nowrap_vs_history")

    # 14 repeated/isolated vs history
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for name, key in (("repeated", "repeated"), ("isolated", "isolated")):
        ax.plot(x, [rE["subtype"][key][rep]["y_true"]["bayes_acc"] for rep in labels], marker="o", label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("target Bayes ceiling")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("U: repeated vs isolated target Bayes vs history")
    save(fig, "repeated_isolated_vs_history")

    # 15 per-seed probe accuracy
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for si, s in enumerate(SEEDS):
        ax.plot(x, [rE["per_seed"][si]["probes"]["all_add"][rep]["mlp32"]["acc"] for rep in labels],
                marker="o", label=str(s), alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel("causal history window")
    ax.set_ylabel("MLP-32 accuracy (all-ADD)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("U: per-seed learned probe accuracy vs history")
    save(fig, "per_seed_probe")

    # 16 full-prefix final comparison
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    cats = ["Bayes\n(target)", "Bayes\n(op_a)", "Bayes\n(op_b)", "probe\nmlp32", "probe\nheld-out"]
    vals = [rE["pooled"]["all_add"]["Lfull"]["y_true"]["bayes_acc"],
            rE["pooled"]["all_add"]["Lfull"]["operand_a"]["bayes_acc"],
            rE["pooled"]["all_add"]["Lfull"]["operand_b"]["bayes_acc"],
            rE["probes"]["all_add"]["Lfull"]["mlp32"]["acc"]["mean"],
            rE["heldout"]["all_add"]["Lfull"]["mlp32"]["heldout"]["mean"]]
    ax.bar(np.arange(5), vals, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])
    ax.axhline(0.99, ls=":", c="k", lw=0.8, label="0.99")
    ax.axhline(0.50, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(cats)
    ax.set_ylabel("accuracy (all-ADD TEST)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("U: full causal prefix — exact vs learned")
    save(fig, "full_prefix_comparison")

    # 17 earliest identifiable-history summary
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    series = [("target", bt("all_add", "y_true")), ("operand_a", bt("all_add", "operand_a")),
              ("operand_b", bt("all_add", "operand_b")), ("raw_sum", bt("all_add", "raw_sum"))]
    for name, vals_ in series:
        ax.plot(x, vals_, marker="o", label=name)
    ax.axhline(0.99, ls=":", c="k", lw=0.8, label="identifiable 0.99")
    ax.axhline(0.50, ls="--", c="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("Bayes ceiling (all-ADD)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("U: interface-identifiable history horizon")
    save(fig, "earliest_identifiable_summary")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment U analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_u_task_interface.npz")
    ap.add_argument("--output", default="experiments/experiment_u_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_u")
    args = ap.parse_args()
    data = load(args.npz)
    res = build(data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"E: {d['verdict']}")
    print(f"  Bayes target L1={d['E_metrics']['bayes_target_L1']} full={d['E_metrics']['bayes_target_full']}"
          f" L*={d['E_metrics']['bayes_Lstar_target']}  best_full_probe={d['E_metrics']['best_full_probe']}"
          f" heldout={d['E_metrics']['best_full_probe_heldout']} gap={d['E_metrics']['full_gap']}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
