#!/usr/bin/env python3
"""Experiment X — analysis: fixed-data optimization-budget sensitivity (200 vs 1,000 updates).

W (W-B) fit three bounded, analysis-only readers (U's flat MLP-128, V's causal GRU, V's causal TCN)
on 1,500 / 4,000 / 10,000 unique sequence-disjoint ordinary-ADD transitions with a FIXED 200
full-batch updates; Class-A target accuracy stayed weak AND training accuracy stayed low
(underfitting). X manipulates exactly ONE factor — the number of full-batch optimizer updates,
`optimizer_updates ∈ {200, 1000}` — holding the data (W's top-scale 10,000-transition pool), the
reader architectures/widths, Adam/3e-3/1e-2, the class-balanced CE loss, the full-batch regime, the
history representations, the targets, the VAL/TEST splits, the held-out operand protocol and the
deterministic fit seeds EXACTLY fixed.

PRE-REGISTRATION (declared before aggregate inspection)
-------------------------------------------------------
* Factor: optimizer_updates in {200, 1000}. One trajectory is run per (seed, reader, target, rep,
  variant) for 1000 full-batch updates, evaluating VAL every 10 updates; the 200-budget result is
  the best-on-VAL checkpoint over the first 200 updates (W's exact selection protocol) and the
  1000-budget result is the best-on-VAL checkpoint over all 1000. Selection uses VAL only.
* Readers (reused verbatim from V/W): MLP-128 (flattened history, tanh); causal GRU hidden 85;
  causal TCN channels 52, kernel 3, dilations [1,2,4,8]. No width changes.
* Data (fixed): W's pool10000 per E seed; W's canonical ordinary VAL (800/seed); W's untouched
  canonical TEST. Held-out variant excludes the 20 pairs {(a,b): b in {0,5}} from fitting.
* History grid = U/V/W {L1,L2,L4,L8,L16,Lfull}; y_true at all lengths; operands + op_dest at the
  decisive horizons {L4, Lfull}; held-out at {L4, Lfull}.
* Decision (checked in order; bars are declared conventions):
    X-A optimization-limited : some reader at L4/full has test >= 0.50 AND held-out >= 0.30 (1000).
    X-B persistent bounded   : all readers' best 1000-budget test <= 0.25 AND held-out <= 0.25 AND
                               training accuracy still low (< 0.50).
    else                     : inconclusive / partial improvement (continuous results reported).
* The 200-budget condition must reproduce W's stored 200-update result exactly (same seed, data,
  procedure) — an explicit bridge is computed and tested.
No LR/scheduler/batch/curriculum/restart tuning; no TEST-based selection of checkpoint, reader or
history.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_x.py \
        --npz experiments/experiment_x_optimization_budget.npz \
        --w-results experiments/experiment_w_results.json \
        --u-results experiments/experiment_u_results.json \
        --output experiments/experiment_x_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_x
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

import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_u as au  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import analyze_experiment_w as aw  # noqa: E402

PRIMARY_MODEL = av.PRIMARY_MODEL
CONTROL_MODEL = av.CONTROL_MODEL
SEEDS = av.SEEDS
READERS = ["MLP128", "GRU", "TCN"]
REPS = av.REPS
REP_LABELS = av.REP_LABELS
DECISIVE = av.DECISIVE
TARGETS = {"y_true": 10, "operand_a": 10, "operand_b": 10, "op_dest": 4}
MAIN_SCOPE = {"y_true": REP_LABELS, "operand_a": DECISIVE, "operand_b": DECISIVE, "op_dest": DECISIVE}
HELDOUT_SCOPE = {"y_true": DECISIVE, "operand_a": DECISIVE, "operand_b": DECISIVE}
POPULATIONS = ["classA", "matched_correct", "correct_all", "all_add"]
HELD_OUT_PAIRS = au.HELD_OUT_PAIRS
BUDGETS = [200, 1000]
EVAL_EVERY = 10

RECOVER_BAR = 0.50
HELDOUT_WEAK = 0.25
HELDOUT_MATERIAL = 0.30
MATERIAL_GAIN = 0.15
BAYES_GAP_BAR = 0.30
TRAIN_LOW_BAR = 0.50

# The fixed data scale = W's top-scale pool. Filled from the X meta in main()/build().
TOP = 10000

_G: Dict[str, Any] = {}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def _fit_seed(seed: int, reader: str, target: str, rep: str, variant: str) -> int:
    """W's exact per-fit seed at the top scale (so the 200-budget trajectory is paired with W)."""
    return aw.reader_seed(seed, reader, target, rep, TOP, variant)


# ---------------------------------------------------------------------------
# reader models (reused verbatim from U/V) + trajectory trainer
# ---------------------------------------------------------------------------

def build_model(reader: str, nclass: int, indim: int, arch: Dict[str, Any]):
    import torch
    if reader == "MLP128":
        return torch.nn.Sequential(torch.nn.Linear(indim, 128), torch.nn.Tanh(),
                                   torch.nn.Linear(128, nclass))
    return av.build_reader(reader, nclass, arch)


def predict_state(reader: str, state, X, norm, nclass: int, indim: int, arch) -> np.ndarray:
    import torch
    model = build_model(reader, nclass, indim, arch)
    model.load_state_dict(state)
    model.eval()
    if reader == "MLP128":
        m, s = norm
        Xn = torch.tensor((np.asarray(X, dtype=np.float64) - m) / s, dtype=torch.float32)
    else:
        Xn = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return model(Xn).argmax(1).numpy()


def train_trajectory(reader: str, Xtr, ytr, Xva, yva, nclass: int, seed: int, updates: int,
                     arch: Dict[str, Any], eval_every: int = EVAL_EVERY) -> Dict[str, Any]:
    """Run `updates` full-batch Adam updates, evaluating VAL every `eval_every`; best-on-VAL only.

    Identical ops/order to U/V/W's readers (`av.train_reader` / `as_.mlp_fit`) so that the first 200
    updates reproduce W's 200-update fit bit-for-bit (same seed, same data, same optimizer).
    """
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    if reader == "MLP128":
        Xtr = np.asarray(Xtr, dtype=np.float64)
        Xva = np.asarray(Xva, dtype=np.float64)
        m = Xtr.mean(0)
        s = Xtr.std(0) + 1e-6
        Xn = torch.tensor((Xtr - m) / s, dtype=torch.float32)
        Xv = torch.tensor((Xva - m) / s, dtype=torch.float32)
        indim = int(Xtr.shape[1])
        lr, wd = as_.MLP_LR, as_.MLP_WD
        norm: Optional[Tuple[np.ndarray, np.ndarray]] = (m, s)
    else:
        Xn = torch.tensor(Xtr, dtype=torch.float32)
        Xv = torch.tensor(Xva, dtype=torch.float32)
        indim = av.INPUT_DIM
        lr, wd = av.READER_LR, av.READER_WD
        norm = None
    model = build_model(reader, nclass, indim, arch)
    yn = torch.tensor(ytr, dtype=torch.long)
    yv = torch.tensor(yva, dtype=torch.long)
    wc = torch.tensor(as_.class_weights(ytr, nclass), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_all: Tuple[float, Any, int] = (-1.0, None, -1)
    best_200: Tuple[float, Any, int] = (-1.0, None, -1)
    events: List[Dict[str, float]] = []
    for ep in range(updates):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xn), yn, weight=wc)
        loss.backward()
        opt.step()
        if (ep + 1) % eval_every == 0:
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
            if vacc > best_all[0]:
                best_all = (vacc, st, ep + 1)
            if ep + 1 <= 200 and vacc > best_200[0]:
                best_200 = (vacc, st, ep + 1)
    if best_all[1] is None:                       # guard: updates < eval_every (never for X budgets)
        best_all = (0.0, {k: v.clone() for k, v in model.state_dict().items()}, updates)
    if best_200[1] is None:
        best_200 = best_all
    return {"events": events, "best_all": best_all, "best_200": best_200, "norm": norm,
            "train_loss_final": events[-1]["train_loss"], "train_acc_final": events[-1]["train_acc"]}


# ---------------------------------------------------------------------------
# worker: one (seed, reader, target, variant) family
# ---------------------------------------------------------------------------

def _init_worker(npz_path: str, arch: Dict[str, Any]) -> None:
    import torch
    torch.set_num_threads(1)
    _G["z"] = np.load(npz_path, allow_pickle=False)
    _G["arch"] = arch
    _G["feat"] = {}


def _feat(split: str, seed: int, L: int, reader: str) -> np.ndarray:
    key = (split, seed, L, reader)
    f = _G["feat"].get(key)
    if f is None:
        z = _G["z"]
        if split == "test":
            seq = z[f"{PRIMARY_MODEL}__test__seq_obs"]
            t = z[f"{PRIMARY_MODEL}__test__t"]
            rows = np.nonzero(z[f"{PRIMARY_MODEL}__test__seed"] == seed)[0]
        else:
            seq = z[f"{split}__seq_obs"]
            t = z[f"{split}__t"]
            rows = np.nonzero(z[f"{split}__seed"] == seed)[0]
        f = (au.history_features(seq[rows], t[rows], L) if reader == "MLP128"
             else av.reader_input(seq[rows], t[rows], L))
        _G["feat"][key] = f
    return f


def _param_count(reader: str, nclass: int, indim: int, arch: Dict[str, Any]) -> int:
    return (as_.param_count("mlp", 128, indim) if reader == "MLP128"
            else av.reader_param_count(reader, nclass, arch))


def _task(spec: Tuple[int, str, str, str]) -> Dict[str, Any]:
    seed, reader, tgt, variant = spec
    _G["feat"] = {}
    z = _G["z"]
    arch = _G["arch"]
    nclass = TARGETS[tgt]
    te_sel = z[f"{PRIMARY_MODEL}__test__seed"] == seed
    te_idx = np.nonzero(te_sel)[0]
    held_te = au._pairs(z, PRIMARY_MODEL, "test", te_idx)
    masks = {m: {p: au._pop_masks(z, m, seed)[p][te_sel] for p in POPULATIONS}
             for m in (PRIMARY_MODEL, CONTROL_MODEL)}
    yte = {t: z[f"{PRIMARY_MODEL}__test__{t}"][te_sel].astype(int) for t in TARGETS}
    val_rows = np.nonzero(z["val__seed"] == seed)[0]
    reps = MAIN_SCOPE[tgt] if variant == "main" else HELDOUT_SCOPE.get(tgt, [])
    out: Dict[str, Any] = {"seed": seed, "reader": reader, "fits": {}, "traj": {}}

    def eval_pops(pred_te):
        return {f"{m}:{p}": (float((pred_te[masks[m][p]] == yte[tgt][masks[m][p]]).mean())
                             if masks[m][p].any() else None)
                for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}

    for rep in reps:
        L = dict(REPS)[rep]
        Xte = _feat("test", seed, L, reader)
        Xva = _feat("val", seed, L, reader)
        yva = z[f"val__{tgt}"][val_rows].astype(int)
        rows = np.nonzero(z[f"pool{TOP}__seed"] == seed)[0]
        Xall = _feat(f"pool{TOP}", seed, L, reader)
        if variant == "main":
            Xtr, ytr = Xall, z[f"pool{TOP}__{tgt}"][rows].astype(int)
        else:
            oa = z[f"pool{TOP}__operand_a"][rows]
            ob = z[f"pool{TOP}__operand_b"][rows]
            held = np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oa, ob)])
            Xtr = Xall[~held]
            ytr = z[f"pool{TOP}__{tgt}"][rows[~held]].astype(int)
        fit = train_trajectory(reader, Xtr, ytr, Xva, yva, nclass,
                               _fit_seed(seed, reader, tgt, rep, variant), max(BUDGETS), arch)
        indim = int(Xtr.shape[1]) if reader == "MLP128" else av.INPUT_DIM
        params = _param_count(reader, nclass, indim, arch)
        for budget, tag in ((200, "best_200"), (1000, "best_all")):
            vacc, st, idx = fit[tag]
            pred_te = predict_state(reader, st, Xte, fit["norm"], nclass, indim, arch)
            node = {"params": params, "updates": budget, "best_idx": int(idx),
                    "val_acc": float(vacc), "n_train": int(len(rows))}
            if variant == "main":
                node["train_acc"] = fit["train_acc_final"]
                node["train_loss"] = fit["train_loss_final"]
                node["acc"] = eval_pops(pred_te)
            else:
                node["n_train"] = int(len(rows) - held.sum())
                node["acc"] = eval_pops(pred_te)
                node["seen"] = {f"{m}:{p}": (float((pred_te[masks[m][p] & ~held_te] == yte[tgt][masks[m][p] & ~held_te]).mean())
                                             if (masks[m][p] & ~held_te).any() else None)
                                for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
                node["heldout"] = {f"{m}:{p}": (float((pred_te[masks[m][p] & held_te] == yte[tgt][masks[m][p] & held_te]).mean())
                                                if (masks[m][p] & held_te).any() else None)
                                   for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
            out["fits"][f"{reader}|{tgt}|{rep}|{budget}|{variant}"] = node
        if variant == "main" and tgt == "y_true":
            out["traj"][rep] = fit["events"]
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _ms(vals):
    return as_.mean_sd(vals)


def aggregate(per: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for reader in READERS:
        recs_all = [s for s in per if s["reader"] == reader]
        if not recs_all:
            continue
        keys = sorted(recs_all[0]["fits"].keys())
        for k in keys:
            _, tgt, rep, budget, variant = k.split("|")
            recs = [s["fits"][k] for s in recs_all]
            node = (out.setdefault(reader, {}).setdefault(tgt, {}).setdefault(rep, {})
                    .setdefault(variant, {}).setdefault(budget, {}))
            node["params"] = recs[0]["params"]
            node["updates"] = recs[0]["updates"]
            node["n_train"] = _ms([r["n_train"] for r in recs])
            node["best_idx"] = _ms([r["best_idx"] for r in recs])
            node["val_acc"] = _ms([r["val_acc"] for r in recs])
            if "train_acc" in recs[0]:
                node["train_acc"] = _ms([r["train_acc"] for r in recs])
                node["train_loss"] = _ms([r["train_loss"] for r in recs])
            node["acc"] = {pk: _ms([r["acc"][pk] for r in recs]) for pk in recs[0]["acc"]}
            if variant == "heldout":
                node["seen"] = {pk: _ms([r["seen"][pk] for r in recs]) for pk in recs[0]["seen"]}
                node["heldout"] = {pk: _ms([r["heldout"][pk] for r in recs]) for pk in recs[0]["heldout"]}
    # trajectory (y_true main): mean/sd over seeds at each recorded update
    traj: Dict[str, Any] = {}
    for reader in READERS:
        recs = [s for s in per if s["reader"] == reader]
        if not recs:
            continue
        for rep in REP_LABELS:
            evs = [s.get("traj", {}).get(rep) for s in recs]
            if not evs or any(e is None or not e for e in evs):
                continue
            ups = [e["update"] for e in evs[0]]
            traj.setdefault(reader, {})[rep] = {
                "updates": ups,
                "val_acc": [_ms([s["traj"][rep][i]["val_acc"] for s in recs]) for i in range(len(ups))],
                "train_acc": [_ms([s["traj"][rep][i]["train_acc"] for s in recs]) for i in range(len(ups))],
                "val_loss": [_ms([s["traj"][rep][i]["val_loss"] for s in recs]) for i in range(len(ups))],
                "train_loss": [_ms([s["traj"][rep][i]["train_loss"] for s in recs]) for i in range(len(ups))]}
    out_traj = traj
    return out, out_traj


def _main_acc(res, reader, rep, budget, pop="E_candidate:classA"):
    return res["readers"][reader]["y_true"][rep]["main"][str(budget)]["acc"][pop]["mean"]


def _ho_acc(res, reader, rep, budget, pop="E_candidate:classA"):
    return res["readers"][reader]["y_true"][rep]["heldout"][str(budget)]["heldout"][pop]["mean"]


def _train_acc(res, reader, rep, budget):
    return res["readers"][reader]["y_true"][rep]["main"][str(budget)]["train_acc"]["mean"]


def load_w_baseline(path: str, top: int) -> Dict[str, Any]:
    """W's stored 200-update Class-A / all-ADD numbers at the fixed top scale (external baseline)."""
    with open(path) as f:
        w = json.load(f)
    out = {"verdict": w["primary_decision"]["verdict"], "train_sizes": w["metadata"]["train_sizes"],
           "readers": {}}
    R = w["readers"]
    for r in READERS:
        out["readers"][r] = {}
        for rep in REP_LABELS:
            node = R[r]["y_true"][rep][str(top)]
            out["readers"][r][rep] = {
                "main_classA": node["main"]["acc"]["E_candidate:classA"]["mean"],
                "main_all_add": node["main"]["acc"]["E_candidate:all_add"]["mean"],
                "train_acc": node["main"]["train_acc"]["mean"],
            }
    return out


def classify(res: Dict[str, Any]) -> Dict[str, Any]:
    per_reader: Dict[str, Any] = {}
    for R in READERS:
        best, brep = None, None
        for rep in DECISIVE:
            a = _main_acc(res, R, rep, 1000)
            if a is not None and (best is None or a > best):
                best, brep = a, rep
        ho = _ho_acc(res, R, brep, 1000) if brep else None
        tr = _train_acc(res, R, brep, 1000) if brep else None
        b200 = max((_main_acc(res, R, rep, 200) for rep in DECISIVE
                    if _main_acc(res, R, rep, 200) is not None), default=None)
        per_reader[R] = {"best_1000": best, "best_rep": brep, "heldout_1000": ho,
                         "train_1000": tr, "best_200": b200,
                         "delta_1000_200": (best - b200) if (best is not None and b200 is not None) else None}
    best_all = max((v["best_1000"] for v in per_reader.values() if v["best_1000"] is not None), default=None)
    best_ho = max((v["heldout_1000"] for v in per_reader.values() if v["heldout_1000"] is not None), default=None)
    max_train = max((v["train_1000"] for v in per_reader.values() if v["train_1000"] is not None), default=None)
    xA = any(v["best_1000"] is not None and v["heldout_1000"] is not None
             and v["best_1000"] >= RECOVER_BAR and v["heldout_1000"] >= HELDOUT_MATERIAL
             for v in per_reader.values())
    xB = bool(best_all is not None and best_all <= HELDOUT_WEAK
              and best_ho is not None and best_ho <= HELDOUT_WEAK
              and max_train is not None and max_train < TRAIN_LOW_BAR)
    if xA:
        verdict = "X-A_optimization_limited"
    elif xB:
        verdict = "X-B_persistent_bounded"
    else:
        verdict = "X-inconclusive_partial"
    return {"verdict": verdict, "per_reader": per_reader, "best_1000_all": best_all,
            "best_heldout_1000_all": best_ho, "max_train_1000": max_train,
            "conditions": {"X-A": xA, "X-B": xB},
            "thresholds": {"recover_bar": RECOVER_BAR, "heldout_weak": HELDOUT_WEAK,
                           "heldout_material": HELDOUT_MATERIAL, "material_gain": MATERIAL_GAIN,
                           "bayes_gap_bar": BAYES_GAP_BAR, "train_low_bar": TRAIN_LOW_BAR},
            "rule_note": ("X-A if some reader at L4/full has 1000-budget test>=0.50 AND held-out>=0.30; "
                          "X-B if every reader's best 1000-budget test<=0.25 AND held-out<=0.25 AND "
                          "training accuracy <0.50 (still low); else inconclusive/partial. Bars are "
                          "declared conventions; continuous values reported.")}


def _collect(data: Dict[str, Any], jobs: int, readers: List[str]) -> List[Dict[str, Any]]:
    npz_path = data["path"]
    arch = av.architecture()
    specs = [(s, R, t, v) for s in SEEDS for R in readers for t in TARGETS
             for v in (["main", "heldout"] if t in HELDOUT_SCOPE else ["main"])]
    t0 = time.perf_counter()
    merged: Dict[Tuple[int, str], Dict[str, Any]] = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs, initializer=_init_worker, initargs=(npz_path, arch)) as pool:
        done = 0
        for part in pool.imap_unordered(_task, specs, chunksize=1):
            key = (part["seed"], part["reader"])
            node = merged.setdefault(key, {"seed": part["seed"], "reader": part["reader"],
                                           "fits": {}, "traj": {}})
            node["fits"].update(part["fits"])
            node["traj"].update(part["traj"])
            done += 1
            if done % 10 == 0 or done == len(specs):
                print(f"  [{done}/{len(specs)}] tasks done", flush=True)
    print(f"collection wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    for node in merged.values():
        node["fits"] = dict(sorted(node["fits"].items()))
    return [merged[(s, R)] for s in SEEDS for R in readers]


def assemble(data: Dict[str, Any], per: List[Dict[str, Any]], u_base: Dict[str, Any],
             w_base: Dict[str, Any]) -> Dict[str, Any]:
    arch = av.architecture()
    readers, traj = aggregate(per)
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "readers": READERS,
        "budgets": BUDGETS, "eval_every": EVAL_EVERY, "top_size": TOP,
        "train_sizes_source": data["meta"]["source"]["train_sizes"],
        "reps": REP_LABELS, "main_scope": MAIN_SCOPE, "heldout_scope": HELDOUT_SCOPE,
        "populations": POPULATIONS, "heldout_pairs": sorted(HELD_OUT_PAIRS),
        "reader_lr": av.READER_LR, "reader_wd": av.READER_WD, "mlp_lr": as_.MLP_LR,
        "mlp_wd": as_.MLP_WD,
        "data_identity": {"source_npz": data["meta"]["source"]["npz"],
                          "source_sha256": data["meta"]["source"]["sha256"],
                          "per_seed": data["meta"]["per_seed"],
                          "disjointness": data["meta"]["disjointness"]},
        "checkpoints": data["meta"]["models"], "n": data["meta"]["n"],
        "thresholds": {"recover_bar": RECOVER_BAR, "heldout_weak": HELDOUT_WEAK,
                       "heldout_material": HELDOUT_MATERIAL, "material_gain": MATERIAL_GAIN,
                       "bayes_gap_bar": BAYES_GAP_BAR, "train_low_bar": TRAIN_LOW_BAR},
        "statistical_discipline": ("Five frozen E checkpoints are the independent units; F is a "
                                   "control. Readers are analysis-only, fit on W's fixed top-scale "
                                   "pool with the SAME seed as W, tuned on the fixed canonical VAL, "
                                   "evaluated once on the untouched canonical TEST. The ONLY factor "
                                   "is optimizer_updates in {200,1000}; selection is VAL-only; no "
                                   "TEST-based selection.")}}
    res["architecture"] = arch
    res["readers"] = readers
    res["trajectory"] = traj
    res["per_seed"] = per
    res["u_bayes"] = u_base
    res["w_baseline"] = w_base
    # explicit 200-budget bridge against W's stored result
    bridge: Dict[str, Any] = {}
    max_diff = 0.0
    for R in READERS:
        for rep in REP_LABELS:
            wv = w_base["readers"][R][rep]
            xv = _main_acc(res, R, rep, 200)
            d = abs((xv if xv is not None else np.nan) - wv["main_classA"])
            bridge[f"{R}|{rep}"] = {"x_200": xv, "w_200": wv["main_classA"], "abs_diff": d}
            if np.isfinite(d):
                max_diff = max(max_diff, float(d))
    res["w_bridge"] = {"max_abs_diff": max_diff, "per_condition": bridge}
    dec = classify(res)
    dec["w_delta_1000_minus_w200"] = {
        R: (dec["per_reader"][R]["best_1000"] - w_base["readers"][R][dec["per_reader"][R]["best_rep"]]["main_classA"])
        for R in READERS if dec["per_reader"][R]["best_rep"] is not None}
    res["primary_decision"] = dec
    return res


def build(data: Dict[str, Any], u_base: Dict[str, Any], w_base: Dict[str, Any], jobs: int,
          readers: Optional[List[str]] = None) -> Dict[str, Any]:
    return assemble(data, _collect(data, jobs, readers or READERS), u_base, w_base)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_x") -> List[str]:
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
    U = res["u_bayes"]
    xs = np.arange(len(BUDGETS))
    colors = {"MLP128": "tab:green", "GRU": "tab:blue", "TCN": "tab:orange"}

    def ma(reader, rep, budget, pop=f"{PRIMARY_MODEL}:classA"):
        return nz(R[reader]["y_true"][rep]["main"][str(budget)]["acc"][pop]["mean"])

    def ho(reader, rep, budget, pop=f"{PRIMARY_MODEL}:classA"):
        return nz(R[reader]["y_true"][rep]["heldout"][str(budget)]["heldout"][pop]["mean"])

    def ta(reader, rep, budget):
        return nz(R[reader]["y_true"][rep]["main"][str(budget)]["train_acc"]["mean"])

    # 1 accuracy vs updates (Class-A, L4 & full)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader in READERS:
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            ax.plot(xs, [ma(reader, rep, b) for b in BUDGETS], marker="o", ls=ls,
                    color=colors[reader], alpha=1.0 if rep == "L4" else 0.6,
                    label=f"{reader} {rep}")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("Class-A TEST accuracy (y_true)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("X: Class-A accuracy vs optimizer updates")
    save(fig, "accuracy_vs_updates")

    # 2 held-out vs updates
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader in READERS:
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            ax.plot(xs, [ho(reader, rep, b) for b in BUDGETS], marker="s", ls=ls,
                    color=colors[reader], alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep}")
    ax.axhline(HELDOUT_MATERIAL, ls=":", c="k", lw=0.8, label="0.30 material")
    ax.axhline(HELDOUT_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("held-out operand accuracy (Class-A)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("X: held-out operand accuracy vs optimizer updates")
    save(fig, "heldout_vs_updates")

    # 3 Bayes gap vs updates
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader in READERS:
        gaps = [U[PRIMARY_MODEL]["L4"] - ma(reader, "L4", b) for b in BUDGETS]
        ax.plot(xs, gaps, marker="o", color=colors[reader], label=f"{reader} L4")
    ax.axhline(BAYES_GAP_BAR, ls="--", c="k", lw=0.8, label="0.30 bar")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS]); ax.set_ylim(0, 1.05)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("Bayes(L4) - reader (Class-A)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("X: Bayes-vs-reader gap vs optimizer updates")
    save(fig, "bayes_gap_vs_updates")

    # 4 train / val / test vs updates (L4)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader in READERS:
        ax.plot(xs, [ta(reader, "L4", b) for b in BUDGETS], marker="o", color=colors[reader],
                label=f"{reader} train")
        ax.plot(xs, [nz(R[reader]["y_true"]["L4"]["main"][str(b)]["val_acc"]["mean"]) for b in BUDGETS],
                marker="^", ls=":", color=colors[reader], alpha=0.7, label=f"{reader} val")
        ax.plot(xs, [ma(reader, "L4", b) for b in BUDGETS], marker="s", ls="--", color=colors[reader],
                alpha=0.7, label=f"{reader} test (Class-A)")
    ax.axhline(TRAIN_LOW_BAR, ls="-.", c="gray", lw=0.8, label="train-low bar 0.50")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("accuracy (L4)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("X: train / val / test accuracy vs optimizer updates (L4)")
    save(fig, "train_val_test_vs_updates")

    # 5 reader comparison at 1000 updates
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    groups = [f"{r}\n{rep}" for r in READERS for rep in DECISIVE]
    vals = [ma(r, rep, 1000) for r in READERS for rep in DECISIVE]
    ax.bar(np.arange(len(groups)), vals, color=[colors[r] for r in READERS for _ in DECISIVE])
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.axhline(HELDOUT_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("Class-A accuracy (1000 updates)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("X: readers at 1000 updates")
    save(fig, "reader_comparison")

    # 6 L4 comparison
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader in READERS:
        ax.plot(xs, [ma(reader, "L4", b) for b in BUDGETS], marker="o", color=colors[reader], label=reader)
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("Class-A L4 accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("X: L4 Class-A accuracy vs optimizer updates")
    save(fig, "l4_comparison")

    # 7 full-prefix comparison
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader in READERS:
        ax.plot(xs, [ma(reader, "Lfull", b) for b in BUDGETS], marker="o", color=colors[reader], label=reader)
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("Class-A full-prefix accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("X: full-prefix Class-A accuracy vs optimizer updates")
    save(fig, "fullprefix_comparison")

    # 8 seen vs held-out at 1000
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    labs, seen, hold = [], [], []
    for reader in READERS:
        for rep in DECISIVE:
            labs.append(f"{reader}\n{rep}")
            seen.append(nz(R[reader]["y_true"][rep]["heldout"]["1000"]["seen"][f"{PRIMARY_MODEL}:classA"]["mean"]))
            hold.append(ho(reader, rep, 1000))
    xi = np.arange(len(labs))
    ax.bar(xi - 0.2, seen, 0.4, label="seen", color="tab:gray")
    ax.bar(xi + 0.2, hold, 0.4, label="held-out", color="tab:purple")
    ax.set_xticks(xi); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("accuracy (1000 updates)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("X: seen vs held-out operand pairs (Class-A, 1000 updates)")
    save(fig, "seen_vs_heldout")

    # 9 per-seed (Class-A L4, 1000)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    xi = np.arange(len(SEEDS))
    for reader in READERS:
        ax.plot(xi, [s["fits"][f"{reader}|y_true|L4|1000|main"]["acc"][f"{PRIMARY_MODEL}:classA"]
                     for s in res["per_seed"] if s["reader"] == reader], marker="o",
                color=colors[reader], label=reader)
    ax.set_xticks(xi); ax.set_xticklabels([str(s) for s in SEEDS]); ax.set_xlabel("E seed")
    ax.set_ylabel("Class-A L4 accuracy (1000 updates)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("X: per-seed accuracy (1000 updates)")
    save(fig, "per_seed")

    # 10 Class-A vs all-ADD
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader in READERS:
        ax.plot(xs, [ma(reader, "L4", b) for b in BUDGETS], marker="o", color=colors[reader],
                label=f"{reader} Class-A")
        ax.plot(xs, [ma(reader, "L4", b, f"{PRIMARY_MODEL}:all_add") for b in BUDGETS], marker="^",
                ls="--", color=colors[reader], alpha=0.6, label=f"{reader} all-ADD")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("TEST accuracy (L4, y_true)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("X: Class-A vs all-ADD vs optimizer updates (L4)")
    save(fig, "classA_vs_alladd")

    # 11 update trajectory (val accuracy vs update, y_true L4 & full)
    T = res.get("trajectory", {})
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for reader in READERS:
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            node = T.get(reader, {}).get(rep)
            if not node:
                continue
            y = [v["mean"] for v in node["val_acc"]]
            ax.plot(node["updates"], y, ls=ls, color=colors[reader], alpha=1.0 if rep == "L4" else 0.6,
                    label=f"{reader} {rep} (val)")
    ax.axvline(200, ls=":", c="k", lw=0.9, label="200 budget")
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("validation accuracy (y_true)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("X: validation accuracy trajectory vs optimizer updates")
    save(fig, "update_trajectory")

    # 12 decision summary
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    d = res["primary_decision"]
    cats = ["best 1000\n(Class-A)", "best 1000\nheld-out", "Bayes(L4)\n-reader", "max train\nacc (1000)"]
    gaps = min((U[PRIMARY_MODEL]["L4"] - v) for v in
               [R[r]["y_true"]["L4"]["main"]["1000"]["acc"][f"{PRIMARY_MODEL}:classA"]["mean"] for r in READERS])
    vals = [nz(d["best_1000_all"]), nz(d["best_heldout_1000_all"]), gaps, nz(d["max_train_1000"])]
    ax.bar(np.arange(4), vals, color=["tab:blue", "tab:purple", "tab:red", "tab:gray"])
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.axhline(TRAIN_LOW_BAR, ls="-.", c="gray", lw=0.8, label="train-low bar")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("value"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"X: decision summary ({d['verdict']})")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    global TOP
    ap = argparse.ArgumentParser(description="Experiment X analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_x_optimization_budget.npz")
    ap.add_argument("--w-results", default="experiments/experiment_w_results.json")
    ap.add_argument("--u-results", default="experiments/experiment_u_results.json")
    ap.add_argument("--output", default="experiments/experiment_x_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_x")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--readers", nargs="*", default=None,
                    help="subset of readers to fit (MLP128 GRU TCN); default all")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()
    data = load(args.npz)
    data["path"] = args.npz
    TOP = int(data["meta"]["source"]["top_size"])
    aw.TRAIN_SIZES = list(data["meta"]["source"]["train_sizes"])   # so aw.reader_seed matches W
    u_base = aw.load_u_baseline(args.u_results)
    w_base = load_w_baseline(args.w_results, TOP)
    t0 = time.perf_counter()
    if args.collect_only:
        per = _collect(data, args.jobs, args.readers or READERS)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"per_seed": per}, f, indent=2)
        print(f"Wrote partial {args.output} in {time.perf_counter() - t0:.1f}s", flush=True)
        return
    res = build(data, u_base, w_base, args.jobs, readers=args.readers)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"verdict: {d['verdict']}  best1000={d['best_1000_all']} heldout1000={d['best_heldout_1000_all']} "
          f"max_train1000={d['max_train_1000']}")
    for R in READERS:
        pr = d["per_reader"][R]
        print(f"  {R}: best1000={pr['best_1000']} ({pr['best_rep']}) ho1000={pr['heldout_1000']} "
              f"best200={pr['best_200']} delta={pr['delta_1000_200']}")
    print(f"  W-bridge max|X200 - W200| = {res['w_bridge']['max_abs_diff']}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
