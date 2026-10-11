#!/usr/bin/env python3
"""Experiment AA — analysis: TCN optimization-recipe diagnostic on the `op_dest` control.

Z (Z-B) showed the causal TCN does not fit the directly observable `op_dest` target under the fixed
protocol while the GRU control fits it to 1.000. AA runs a **predeclared crossed optimizer grid**
and asks whether some recipe lets the existing TCN implementation fit `op_dest`.

PRE-REGISTRATION (declared before aggregate inspection)
-------------------------------------------------------
* Grid: learning rate in {3e-4, 1e-3, 3e-3, 1e-2, 3e-2} x update budget in {1,000, 5,000} x
  readers {TCN, GRU} x histories {L4, Lfull} x seeds 42-46 (full cross). Adam with the existing
  weight decay 1e-2, full-batch class-balanced CE; only LR and budget change. Initialization is
  matched per (seed, reader, history) across all LR/budget cells (`aw.reader_seed(seed, reader,
  "op_dest", rep, top, "main")`), so LR/budget comparisons share one deterministic init.
* Data: Z's exact `op_dest` dataset (Y's fixed 256-example subset, canonical VAL/TEST; byte-verified).
* Fitting threshold `fit_bar = 0.90`. A recipe is ROBUST at a history iff the five-seed MEAN maximum
  training accuracy >= 0.90 AND at least 4/5 seeds reach 0.90.
* Decision (checked in order; runs only if the implementation audit and all invariants pass):
    AA-INVALID : audit/invariant/GRU-control failure (the GRU control must robustly fit at BOTH
                 histories under at least one recipe).
    AA-A       : some recipe is robust at BOTH L4 and Lfull.
    AA-MIX     : some recipe is robust at exactly one history, OR a recipe reaches mean >= 0.90 at
                 some history without the 4/5-seed robust criterion (seed-dependent).
    AA-B       : no recipe reaches the robust criterion at either history.
* Checkpoint selection is VAL-only; the best-on-TRAIN state is diagnostic. TEST is never used to
  select LR, budget, reader or history. Every grid cell is reported, including divergence/negative.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_aa.py \
        --npz experiments/experiment_aa_optimizer_grid.npz \
        --output experiments/experiment_aa_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_aa
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
import analyze_experiment_x as ax  # noqa: E402

PRIMARY_MODEL = av.PRIMARY_MODEL
CONTROL_MODEL = av.CONTROL_MODEL
SEEDS = av.SEEDS
READERS = ["TCN", "GRU"]
DECISIVE = ["L4", "Lfull"]
STATES = ["final", "best_val", "best_train"]
POPULATIONS = ["classA", "matched_correct", "correct_all", "all_add"]

LRS = [("3e-4", 3e-4), ("1e-3", 1e-3), ("3e-3", 3e-3), ("1e-2", 1e-2), ("3e-2", 3e-2)]
LR_LABELS = [lbl for lbl, _ in LRS]
BUDGETS = [1000, 5000]
WD = av.READER_WD
EVAL_EVERY = 10
MILESTONES = [200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000]

TARGET = "op_dest"
NCLASS = 4
SUBSET_SIZE = 256
FIT_BAR = 0.90
ROBUST_SEEDS = 4

TOP = 10000

_G: Dict[str, Any] = {}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def _fit_seed(seed: int, reader: str, rep: str) -> int:
    """Matched deterministic init seed for every LR/budget cell of a (seed, reader, history)."""
    return aw.reader_seed(seed, reader, TARGET, rep, TOP, "main")


def _param_count(reader: str) -> int:
    return av.reader_param_count(reader, NCLASS, av.architecture())


# ---------------------------------------------------------------------------
# grid trainer (ay.train_trajectory protocol, with an explicit learning rate)
# ---------------------------------------------------------------------------

def train_grid(reader: str, Xtr, ytr, Xva, yva, nclass: int, seed: int, updates: int, lr: float,
               wd: float, arch: Dict[str, Any], eval_every: int = EVAL_EVERY) -> Dict[str, Any]:
    """Exactly `ay.train_trajectory`'s protocol with a selectable learning rate; GRU/TCN only."""
    import torch
    import torch.nn.functional as F
    assert reader in ("GRU", "TCN"), reader
    torch.manual_seed(seed)
    Xn = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    Xv = torch.tensor(np.asarray(Xva, dtype=np.float32))
    indim = av.INPUT_DIM
    model = ax.build_model(reader, nclass, indim, arch)
    yn = torch.tensor(ytr, dtype=torch.long)
    yv = torch.tensor(yva, dtype=torch.long)
    wc = torch.tensor(as_.class_weights(ytr, nclass), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val: Tuple[float, Any, int] = (-1.0, None, -1)
    best_train: Tuple[float, Any, int] = (-1.0, None, -1)
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
            if vacc > best_val[0]:
                best_val = (vacc, st, ep + 1)
            if tracc > best_train[0]:
                best_train = (tracc, st, ep + 1)
    final = {k: v.clone() for k, v in model.state_dict().items()}
    if best_val[1] is None:
        best_val = (events[-1]["val_acc"], {k: v.clone() for k, v in final.items()}, updates)
    if best_train[1] is None:
        best_train = (events[-1]["train_acc"], {k: v.clone() for k, v in final.items()}, updates)
    finite = bool(all(np.isfinite([e["train_loss"], e["train_acc"], e["val_loss"], e["val_acc"]]).all()
                      for e in events))
    return {"events": events, "best_val": best_val, "best_train": best_train, "final": final,
            "norm": None, "train_acc_final": events[-1]["train_acc"],
            "train_loss_final": events[-1]["train_loss"], "val_acc_final": events[-1]["val_acc"],
            "val_loss_final": events[-1]["val_loss"], "finite": finite}


# ---------------------------------------------------------------------------
# worker: one (seed, reader)
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


def _task(spec: Tuple[int, str, str]) -> Dict[str, Any]:
    seed, reader, rep = spec
    _G["feat"] = {}
    z = _G["z"]
    arch = _G["arch"]
    sub_rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == seed)[0]
    ytr = z[f"sub{SUBSET_SIZE}__{TARGET}"][sub_rows].astype(int)
    val_rows = np.nonzero(z["val__seed"] == seed)[0]
    yva = z[f"val__{TARGET}"][val_rows].astype(int)
    te_sel = z[f"{PRIMARY_MODEL}__test__seed"] == seed
    te_idx = np.nonzero(te_sel)[0]
    masks = {m: {p: au._pop_masks(z, m, seed)[p][te_sel] for p in POPULATIONS}
             for m in (PRIMARY_MODEL, CONTROL_MODEL)}
    yte = z[f"{PRIMARY_MODEL}__test__{TARGET}"][te_sel].astype(int)
    out: Dict[str, Any] = {"seed": seed, "reader": reader, "cells": {}, "traj": {}}

    def eval_state(state):
        pred_t = ax.predict_state(reader, state, Xtr, None, NCLASS, av.INPUT_DIM, arch)
        pred_v = ax.predict_state(reader, state, Xva, None, NCLASS, av.INPUT_DIM, arch)
        pred_e = ax.predict_state(reader, state, Xte, None, NCLASS, av.INPUT_DIM, arch)
        acc = {f"{m}:{p}": (float((pred_e[masks[m][p]] == yte[masks[m][p]]).mean())
                            if masks[m][p].any() else None)
               for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
        return {"train_acc": float((pred_t == ytr).mean()), "val_acc": float((pred_v == yva).mean()),
                "acc": acc}

    for _rep, L in (("L4", 4), ("Lfull", 10)):
        if _rep != rep:
            continue
        Xtr = _feat("sub", seed, L)
        Xva = _feat("val", seed, L)
        Xte = _feat("test", seed, L)
        for lr_label, lr in LRS:
            for b in BUDGETS:
                fit = train_grid(reader, Xtr, ytr, Xva, yva, NCLASS, _fit_seed(seed, reader, rep),
                                 b, lr, WD, arch)
                ms = {str(u): next((e for e in fit["events"] if e["update"] == u), None)
                      for u in MILESTONES if u <= b}
                key = f"{reader}|{rep}|{lr_label}|{b}"
                out["cells"][key] = {
                    "lr": lr, "updates": b, "rep": rep, "params": _param_count(reader),
                    "n_train": int(len(sub_rows)),
                    "fit": {"train_acc_max": float(fit["best_train"][0]),
                            "train_acc_max_update": int(fit["best_train"][2]),
                            "train_acc_final": float(fit["train_acc_final"]),
                            "train_loss_final": float(fit["train_loss_final"]),
                            "val_acc_final": float(fit["val_acc_final"]),
                            "val_loss_final": float(fit["val_loss_final"]),
                            "best_val_acc": float(fit["best_val"][0]),
                            "best_val_update": int(fit["best_val"][2]),
                            "finite": fit["finite"], "milestones": ms},
                    "final": eval_state(fit["final"]),
                    "best_val": eval_state(fit["best_val"][1]),
                    "best_train": eval_state(fit["best_train"][1]),
                }
                if lr_label == "3e-3" and b == 1000:
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
        recs = [s for s in per if s["reader"] == reader]
        if not recs:
            continue
        for key in sorted(recs[0]["cells"].keys()):
            _, rep, lr_label, b = key.split("|")
            b = int(b)
            cells = [s["cells"][key] for s in recs]
            node = out.setdefault(reader, {}).setdefault(rep, {}).setdefault(lr_label, {})[b] = {}
            node["lr"] = cells[0]["lr"]
            node["updates"] = cells[0]["updates"]
            node["params"] = cells[0]["params"]
            node["n_train"] = cells[0]["n_train"]
            node["train_max"] = _ms([c["fit"]["train_acc_max"] for c in cells])
            node["per_seed_train_max"] = [c["fit"]["train_acc_max"] for c in cells]
            node["train_max_update"] = _ms([c["fit"]["train_acc_max_update"] for c in cells])
            node["train_final"] = _ms([c["fit"]["train_acc_final"] for c in cells])
            node["train_loss_final"] = _ms([c["fit"]["train_loss_final"] for c in cells])
            node["val_final"] = _ms([c["fit"]["val_acc_final"] for c in cells])
            node["best_val_acc"] = _ms([c["fit"]["best_val_acc"] for c in cells])
            node["all_finite"] = bool(all(c["fit"]["finite"] for c in cells))
            ms_keys = sorted(cells[0]["fit"]["milestones"].keys(), key=int)
            node["milestones"] = {u: {fld: _ms([c["fit"]["milestones"][u][fld] for c in cells])
                                      for fld in ("train_acc", "train_loss", "val_acc", "val_loss")}
                                  for u in ms_keys}
            node["best_val"] = {"train_acc": _ms([c["best_val"]["train_acc"] for c in cells]),
                                "val_acc": _ms([c["best_val"]["val_acc"] for c in cells]),
                                "acc": {pk: _ms([c["best_val"]["acc"][pk] for c in cells])
                                        for pk in cells[0]["best_val"]["acc"]}}
            node["final"] = {"train_acc": _ms([c["final"]["train_acc"] for c in cells]),
                             "val_acc": _ms([c["final"]["val_acc"] for c in cells]),
                             "acc": {pk: _ms([c["final"]["acc"][pk] for c in cells])
                                     for pk in cells[0]["final"]["acc"]}}
    return out


def _traj(per: List[Dict[str, Any]]) -> Dict[str, Any]:
    traj: Dict[str, Any] = {}
    for reader in READERS:
        recs = [s for s in per if s["reader"] == reader]
        if not recs:
            continue
        for rep in DECISIVE:
            evs = [s.get("traj", {}).get(rep) for s in recs]
            if not evs or any(e is None or not e for e in evs):
                continue
            ups = [e["update"] for e in evs[0]]
            traj.setdefault(reader, {})[rep] = {
                "updates": ups,
                "train_acc": [_ms([s["traj"][rep][i]["train_acc"] for s in recs]) for i in range(len(ups))],
                "val_acc": [_ms([s["traj"][rep][i]["val_acc"] for s in recs]) for i in range(len(ups))],
                "train_loss": [_ms([s["traj"][rep][i]["train_loss"] for s in recs]) for i in range(len(ups))]}
    return traj


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

def _robust_cells(res, reader, rep):
    """Recipes robust at a history: five-seed mean >= fit_bar AND >=4/5 seeds >= fit_bar."""
    out = []
    for lr_label in LR_LABELS:
        for b in BUDGETS:
            node = res["readers"][reader][rep][lr_label][b]
            m = node["train_max"]["mean"]
            nfit = sum(1 for v in node["per_seed_train_max"] if v is not None and v >= FIT_BAR)
            if m is not None and m >= FIT_BAR and nfit >= ROBUST_SEEDS:
                out.append((lr_label, b))
    return out


def _mean_only_cells(res, reader, rep):
    out = []
    for lr_label in LR_LABELS:
        for b in BUDGETS:
            node = res["readers"][reader][rep][lr_label][b]
            m = node["train_max"]["mean"]
            nfit = sum(1 for v in node["per_seed_train_max"] if v is not None and v >= FIT_BAR)
            if m is not None and m >= FIT_BAR and nfit < ROBUST_SEEDS:
                out.append((lr_label, b))
    return out


def classify(res: Dict[str, Any], problems: List[str]) -> Dict[str, Any]:
    if problems:
        return {"verdict": "AA-INVALID", "control_ok": False,
                "tcn_robust": {rep: [] for rep in DECISIVE},
                "tcn_mean_only": {rep: [] for rep in DECISIVE},
                "gru_robust": {rep: [] for rep in DECISIVE}, "qualified_recipes": [],
                "validation_problems": problems,
                "thresholds": {"fit_bar": FIT_BAR, "robust_seeds": ROBUST_SEEDS, "n_seeds": len(SEEDS)},
                "rule_note": ("AA-INVALID: the implementation audit, a data/split invariant, or the "
                              "GRU positive control failed; no TCN-capacity conclusion is drawn.")}
    tcn = {rep: _robust_cells(res, "TCN", rep) for rep in DECISIVE}
    tcn_near = {rep: _mean_only_cells(res, "TCN", rep) for rep in DECISIVE}
    gru = {rep: _robust_cells(res, "GRU", rep) for rep in DECISIVE}
    control_ok = bool(set(gru["L4"]) & set(gru["Lfull"]))
    both = sorted(set(tcn["L4"]) & set(tcn["Lfull"]))
    if problems or not control_ok:
        verdict = "AA-INVALID"
    elif both:
        verdict = "AA-A"
    elif tcn["L4"] or tcn["Lfull"] or tcn_near["L4"] or tcn_near["Lfull"]:
        verdict = "AA-MIX"
    else:
        verdict = "AA-B"
    return {"verdict": verdict, "control_ok": control_ok,
            "tcn_robust": {rep: [list(c) for c in tcn[rep]] for rep in DECISIVE},
            "tcn_mean_only": {rep: [list(c) for c in tcn_near[rep]] for rep in DECISIVE},
            "gru_robust": {rep: [list(c) for c in gru[rep]] for rep in DECISIVE},
            "qualified_recipes": [list(c) for c in both],
            "validation_problems": problems,
            "thresholds": {"fit_bar": FIT_BAR, "robust_seeds": ROBUST_SEEDS, "n_seeds": len(SEEDS)},
            "rule_note": ("AA-INVALID if audit/invariant/control fails (control = GRU robust at both "
                          "histories for some recipe); else AA-A if some recipe is robust at BOTH "
                          "histories; else AA-MIX if some recipe is robust at exactly one history or "
                          "reaches mean>=0.90 without the 4/5-seed criterion; else AA-B. Robust = "
                          "five-seed mean max-train acc >=0.90 AND >=4/5 seeds >=0.90.")}


def _validate(meta: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if not meta.get("audit_pass", False):
        problems.append("TCN implementation audit failed")
    for where in ("train", "val", "test"):
        for k, v in meta.get("class_counts", {}).get(where, {}).items():
            if len(v) != NCLASS or any(int(c) <= 0 for c in v):
                problems.append(f"class coverage failure: {where}/{k}={v}")
    for k, v in meta.get("disjointness", {}).items():
        if int(v) != 0:
            problems.append(f"split disjointness failure: {k}={v}")
    for k, rec in meta.get("recoverability", {}).items():
        if abs(float(rec.get("rate", 0.0)) - 1.0) > 1e-12:
            problems.append(f"op_dest recoverability failure: {k}={rec}")
    return problems


def _collect(data: Dict[str, Any], jobs: int, readers: List[str]) -> List[Dict[str, Any]]:
    npz_path = data["path"]
    arch = av.architecture()
    specs = [(s, R, rep) for s in SEEDS for R in readers for rep in DECISIVE]
    t0 = time.perf_counter()
    if jobs <= 1:
        _init_worker(npz_path, arch)
        parts = [_task(sp) for sp in specs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=jobs, initializer=_init_worker, initargs=(npz_path, arch)) as pool:
            parts = list(pool.imap_unordered(_task, specs, chunksize=1))
    print(f"collection wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    merged: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for part in parts:
        node = merged.setdefault((part["seed"], part["reader"]),
                                 {"seed": part["seed"], "reader": part["reader"],
                                  "cells": {}, "traj": {}})
        node["cells"].update(part["cells"])
        node["traj"].update(part["traj"])
    for node in merged.values():
        node["cells"] = dict(sorted(node["cells"].items()))
    return [merged[(s, R)] for s in SEEDS for R in readers]


def assemble(data: Dict[str, Any], per: List[Dict[str, Any]]) -> Dict[str, Any]:
    arch = av.architecture()
    readers = aggregate(per)
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "readers": READERS,
        "target": TARGET, "n_classes": NCLASS, "reps": DECISIVE, "states": STATES,
        "lrs": [list(x) for x in LRS], "budgets": BUDGETS, "weight_decay": WD,
        "eval_every": EVAL_EVERY, "milestones": MILESTONES,
        "subset_size": data["meta"]["subset_size"], "subset_seed": data["meta"]["subset_seed"],
        "populations": POPULATIONS, "source_identity": data["meta"]["source"],
        "class_counts": data["meta"]["class_counts"], "recoverability": data["meta"]["recoverability"],
        "disjointness": data["meta"]["disjointness"], "audit": data["meta"]["audit"],
        "audit_pass": data["meta"]["audit_pass"], "checkpoints": data["meta"]["models"],
        "n": data["meta"]["n"], "thresholds": {"fit_bar": FIT_BAR, "robust_seeds": ROBUST_SEEDS},
        "matched_init": ("every LR/budget cell of a (seed, reader, history) uses the same "
                         "deterministic init seed `aw.reader_seed(seed, reader, 'op_dest', rep, top, "
                         "'main')`; only LR and the update budget change between cells"),
        "statistical_discipline": (
            "Five frozen E checkpoints are the independent units; F is a control. The two readers are "
            "analysis-only, fit on Z/Y's exact fixed 256-transition subset, tuned on Y's canonical VAL, "
            "evaluated once on Y's untouched canonical TEST. Selection is VAL-only; the best-on-TRAIN "
            "state is diagnostic; TEST is never used to select LR/budget/reader/history.")}}
    res["architecture"] = arch
    res["readers"] = readers
    res["trajectory"] = _traj(per)
    res["per_seed"] = per
    problems = _validate(data["meta"])
    res["primary_decision"] = classify(res, problems)
    return res


def build(data: Dict[str, Any], jobs: int, readers: Optional[List[str]] = None) -> Dict[str, Any]:
    return assemble(data, _collect(data, jobs, readers or READERS))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_aa") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    def _cell(node, b):
        return node[str(b)] if str(b) in node else node[b]

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    R = res["readers"]
    lrx = np.arange(len(LR_LABELS))

    def tm(reader, rep, lr_label, b):
        v = _cell(R[reader][rep][lr_label], b)["train_max"]["mean"]
        return np.nan if v is None else v

    # 1 LR curves per reader/history/budget
    for reader, fname in (("TCN", "lr_curve_tcn"), ("GRU", "lr_curve_gru")):
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            for b, c in ((1000, "tab:blue"), (5000, "tab:orange")):
                ax.plot(lrx, [tm(reader, rep, l, b) for l in LR_LABELS], marker="o", ls=ls, color=c,
                        alpha=1.0 if rep == "L4" else 0.6, label=f"{rep} {b}")
        ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
        ax.set_xticks(lrx); ax.set_xticklabels(LR_LABELS); ax.set_ylim(0, 1.02)
        ax.set_xlabel("learning rate"); ax.set_ylabel("max training accuracy (op_dest, 256-subset)")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        ax.set_title(f"AA: {reader} — max training accuracy vs learning rate")
        save(fig, fname)

    # 2 budget effect (TCN)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for rep, ls in (("L4", "-"), ("Lfull", "--")):
        ax.plot(lrx, [tm("TCN", rep, l, 1000) for l in LR_LABELS], marker="o", ls=ls, color="tab:blue",
                alpha=1.0 if rep == "L4" else 0.6, label=f"{rep} 1000")
        ax.plot(lrx, [tm("TCN", rep, l, 5000) for l in LR_LABELS], marker="s", ls=ls, color="tab:orange",
                alpha=1.0 if rep == "L4" else 0.6, label=f"{rep} 5000")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9)
    ax.set_xticks(lrx); ax.set_xticklabels(LR_LABELS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("learning rate"); ax.set_ylabel("max training accuracy")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AA: TCN update-budget effect vs learning rate")
    save(fig, "budget_effect")

    # 3 heatmaps (TCN): LR x budget, per history
    for rep in DECISIVE:
        M = np.array([[tm("TCN", rep, l, b) for b in BUDGETS] for l in LR_LABELS])
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(BUDGETS))); ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_yticks(range(len(LR_LABELS))); ax.set_yticklabels(LR_LABELS)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < 0.6 else "k", fontsize=9)
        fig.colorbar(im, ax=ax, label="max train acc")
        ax.set_xlabel("update budget"); ax.set_ylabel("learning rate")
        ax.set_title(f"AA: TCN fit (max train acc), {rep}")
        save(fig, "fit_heatmap" if rep == "L4" else f"fit_heatmap_{rep}")

    # 4 per-seed at the best TCN cell
    def best_cell(reader):
        best = (-1.0, None)
        for rep in DECISIVE:
            for l in LR_LABELS:
                for b in BUDGETS:
                    v = tm(reader, rep, l, b)
                    if v is not None and not np.isnan(v) and v > best[0]:
                        best = (v, (rep, l, b))
        return best
    bv, bcell = best_cell("TCN")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    if bcell:
        rep, l, b = bcell
        ax.bar(np.arange(len(SEEDS)),
               _cell(R["TCN"][rep][l], b)["per_seed_train_max"], color="tab:orange")
        ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
        ax.set_title(f"AA: TCN per-seed max train acc (best cell: {rep}, lr={l}, budget={b}, mean={bv:.3f})")
    ax.set_xticks(np.arange(len(SEEDS))); ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("E seed"); ax.set_ylabel("max training accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    save(fig, "per_seed")

    # 5 train/val curves at lr=3e-3 for both readers (baseline recipe)
    T = res.get("trajectory", {})
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for reader, c in (("TCN", "tab:orange"), ("GRU", "tab:blue")):
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            node = T.get(reader, {}).get(rep)
            if not node:
                continue
            ax.plot(node["updates"], [v["mean"] for v in node["train_acc"]], ls=ls, color=c,
                    alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep} train")
            ax.plot(node["updates"], [v["mean"] for v in node["val_acc"]], ls=":", color=c,
                    alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep} val")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.8)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.3)
    ax.set_title("AA: training/validation trajectories at lr=3e-3 (1000 updates)")
    save(fig, "train_val_curves")

    # 6 reader comparison across grid (best over cells)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, reader in enumerate(READERS):
        vals = [tm(reader, rep, l, b) for rep in DECISIVE for l in LR_LABELS for b in BUDGETS]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        ax.bar(i, max(vals) if vals else np.nan, color={"TCN": "tab:orange", "GRU": "tab:blue"}[reader],
               width=0.5)
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.set_xticks(range(len(READERS))); ax.set_xticklabels(READERS); ax.set_ylim(0, 1.02)
    ax.set_ylabel("max train acc (best grid cell)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AA: reader comparison (best over grid)")
    save(fig, "reader_comparison")

    # 7 GRU control across grid
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for rep, ls in (("L4", "-"), ("Lfull", "--")):
        for b, c in ((1000, "tab:blue"), (5000, "tab:orange")):
            ax.plot(lrx, [tm("GRU", rep, l, b) for l in LR_LABELS], marker="o", ls=ls, color=c,
                    alpha=1.0 if rep == "L4" else 0.6, label=f"{rep} {b}")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9)
    ax.set_xticks(lrx); ax.set_xticklabels(LR_LABELS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("learning rate"); ax.set_ylabel("max training accuracy")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AA: GRU positive control across the grid")
    save(fig, "gru_control")

    # 8 decision summary
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    d = res["primary_decision"]
    tcn_best = max((tm("TCN", rep, l, b) for rep in DECISIVE for l in LR_LABELS for b in BUDGETS
                    if tm("TCN", rep, l, b) is not None), default=np.nan)
    gru_best = max((tm("GRU", rep, l, b) for rep in DECISIVE for l in LR_LABELS for b in BUDGETS
                    if tm("GRU", rep, l, b) is not None), default=np.nan)
    cats = ["TCN best\n(max train)", "GRU best\n(max train)", "TCN test\n(all-ADD, best-VAL)"]
    tcn_test = np.nan
    for rep in DECISIVE:
        for l in LR_LABELS:
            for b in BUDGETS:
                v = _cell(R["TCN"][rep][l], b)["best_val"]["acc"][f"{PRIMARY_MODEL}:all_add"]["mean"]
                if v is not None and (np.isnan(tcn_test) or v > tcn_test):
                    tcn_test = v
    ax.bar(np.arange(3), [tcn_best, gru_best, tcn_test], color=["tab:orange", "tab:blue", "tab:red"])
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(cats, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"AA: decision summary ({d['verdict']})")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    global TOP
    ap = argparse.ArgumentParser(description="Experiment AA analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_aa_optimizer_grid.npz")
    ap.add_argument("--output", default="experiments/experiment_aa_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_aa")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--readers", nargs="*", default=None, help="subset of readers (TCN GRU)")
    ap.add_argument("--raw", default="experiments/experiment_aa_grid_raw.json",
                    help="raw per-seed grid fits (resumable)")
    ap.add_argument("--collect-only", action="store_true", help="fit the grid and write --raw only")
    ap.add_argument("--use-raw", action="store_true", help="assemble from --raw (skip fitting)")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    data = load(args.npz)
    data["path"] = args.npz
    TOP = int(data["meta"]["source"]["top_size"])
    aw.TRAIN_SIZES = list(data["meta"]["source"]["train_sizes"])
    problems = _validate(data["meta"])
    if args.audit_only or problems:
        print(json.dumps({"audit_pass": data["meta"]["audit_pass"], "problems": problems,
                          "audit": data["meta"]["audit"]}, indent=2))
        if problems and not args.audit_only:
            res = assemble(data, [])
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(res, f, indent=2)
            print(f"INVALID: {problems}; wrote {args.output} with verdict "
                  f"{res['primary_decision']['verdict']}")
        return
    if args.collect_only:
        per = _collect(data, args.jobs, args.readers or READERS)
        os.makedirs(os.path.dirname(args.raw), exist_ok=True)
        with open(args.raw, "w") as f:
            json.dump({"per_seed": per}, f)
        print(f"Wrote raw {args.raw}")
        return
    if args.use_raw:
        with open(args.raw) as f:
            per = json.load(f)["per_seed"]
    else:
        per = _collect(data, args.jobs, args.readers or READERS)
        os.makedirs(os.path.dirname(args.raw), exist_ok=True)
        with open(args.raw, "w") as f:
            json.dump({"per_seed": per}, f)
    t0 = time.perf_counter()
    res = assemble(data, per)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"verdict: {d['verdict']}  control_ok={d['control_ok']}")
    print(f"  TCN robust L4={d['tcn_robust']['L4']} full={d['tcn_robust']['Lfull']} "
          f"mean_only={d['tcn_mean_only']}")
    print(f"  GRU robust L4={d['gru_robust']['L4']} full={d['gru_robust']['Lfull']}")
    for reader in READERS:
        for rep in DECISIVE:
            row = " ".join(f"{l}:{res['readers'][reader][rep][l][b]['train_max']['mean']:.3f}"
                           for l in LR_LABELS for b in BUDGETS)
            print(f"  {reader} {rep}: {row}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
