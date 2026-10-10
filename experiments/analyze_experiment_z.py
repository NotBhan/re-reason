#!/usr/bin/env python3
"""Experiment Z — analysis: TCN instrument qualification on the directly observable `op_dest` target.

Y (Y-MIX) showed the GRU fits a fixed 256-example `y_true` subset but does not generalize, while the
TCN does not fit even the small subset. Z asks whether the TCN can fit a **directly observable**
four-way target — `op_dest` = the destination register index = `argmax(x_t[3:7])` of the current
observation — under the **same** input data, subset, readers, optimizer protocol and history
conditions as Y.

Reader set (exactly two, reused verbatim from U/V/W/X/Y):
  * causal GRU, hidden 85  (positive control);
  * causal TCN, channels 52, kernel 3, dilations [1,2,4,8], 4 layers (the subject).
History conditions: `L4` and `Lfull`. The target has 4 classes, so only the output head changes
(`Linear(width, 4)` instead of `Linear(width, 10)`); the temporal reader body, causal padding,
input features and validity-mask semantics are unchanged.

PRE-REGISTRATION (declared before aggregate inspection)
-------------------------------------------------------
* Only `op_dest` is trained; the label is used as a target only (never as an input feature). The
  collector verifies `op_dest` is exactly `argmax(x_t[3:7])` (100% recoverable) in train/val/test.
* Fitting set = Y's EXACT fixed 256-transition class-balanced subset/seed (copied verbatim);
  VAL/TEST = Y's canonical VAL/TEST (copied verbatim, byte-verified). No regeneration/resampling.
* Training = Y's exact protocol: Adam 3e-3, weight-decay 1e-2, full-batch, class-balanced CE,
  exactly 1,000 updates, VAL evaluated every 10 updates, deterministic per-fit seed
  (`aw.reader_seed(seed, reader, "op_dest", rep, TOP, "main")`). No scheduler/curriculum/restart/
  tuning/early stopping.
* Three states reported per (reader, history): final (after exactly 1,000 updates), best-on-VAL
  (validation-only selection), best-on-TRAIN (diagnostic; never a deployable selection).
* Decision (checked in order; `fit_bar = 0.90` on the maximum observed TRAINING accuracy):
    Z-INVALID : the GRU control fails (< 0.90 at some history), or a class/split/recoverability
                invariant fails.
    Z-A       : the GRU reaches the threshold and the TCN reaches >= 0.90 at BOTH L4 and Lfull.
    Z-B       : the GRU reaches the threshold but the TCN is below 0.90 at BOTH histories.
    Z-MIX     : the TCN meets the threshold at one history but not the other.
* Z is an instrument qualification, not a RE:Reason capability experiment. All thresholds and
  classifications are declared before aggregate results are inspected.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_z.py \
        --npz experiments/experiment_z_op_dest.npz \
        --output experiments/experiment_z_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_z
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
import analyze_experiment_y as ay  # noqa: E402  (train_trajectory / predict_state / param helpers)

PRIMARY_MODEL = av.PRIMARY_MODEL
CONTROL_MODEL = av.CONTROL_MODEL
SEEDS = av.SEEDS
READERS = ["GRU", "TCN"]
REPS = av.REPS
DECISIVE = ["L4", "Lfull"]
STATES = ["final", "best_val", "best_train"]
POPULATIONS = ["classA", "matched_correct", "correct_all", "all_add"]
HELD_OUT_PAIRS = au.HELD_OUT_PAIRS

TARGET = "op_dest"
NCLASS = 4
SUBSET_SIZE = 256
UPDATES = 1000
EVAL_EVERY = 10
MILESTONES = [200, 400, 600, 800, 1000]

FIT_BAR = 0.90

TOP = 10000

_G: Dict[str, Any] = {}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def _fit_seed(seed: int, reader: str, rep: str) -> int:
    """Deterministic reader-seed convention for the op_dest target at the top scale."""
    return aw.reader_seed(seed, reader, TARGET, rep, TOP, "main")


def _param_count(reader: str, indim: int, arch: Dict[str, Any], nclass: int = NCLASS) -> int:
    return av.reader_param_count(reader, nclass, arch)


# ---------------------------------------------------------------------------
# worker: one (seed, reader)
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
        elif split == "val":
            seq = z["val__seq_obs"]
            t = z["val__t"]
            rows = np.nonzero(z["val__seed"] == seed)[0]
        else:
            seq = z[f"sub{SUBSET_SIZE}__seq_obs"]
            t = z[f"sub{SUBSET_SIZE}__t"]
            rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == seed)[0]
        f = (au.history_features(seq[rows], t[rows], L) if reader == "MLP128"
             else av.reader_input(seq[rows], t[rows], L))
        _G["feat"][key] = f
    return f


def _task(spec: Tuple[int, str]) -> Dict[str, Any]:
    seed, reader = spec
    _G["feat"] = {}
    z = _G["z"]
    arch = _G["arch"]
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
    out: Dict[str, Any] = {"seed": seed, "reader": reader, "fits": {}, "traj": {}}

    def eval_state(state, indim):
        pred_t = ay.predict_state(reader, state, Xtr, fit["norm"], NCLASS, indim, arch)
        pred_v = ay.predict_state(reader, state, Xva, fit["norm"], NCLASS, indim, arch)
        pred_e = ay.predict_state(reader, state, Xte, fit["norm"], NCLASS, indim, arch)
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

    for rep in DECISIVE:
        L = dict(REPS)[rep]
        Xtr = _feat("sub", seed, L, reader)
        Xva = _feat("val", seed, L, reader)
        Xte = _feat("test", seed, L, reader)
        fit = ay.train_trajectory(reader, Xtr, ytr, Xva, yva, NCLASS,
                                  _fit_seed(seed, reader, rep), UPDATES, arch)
        indim = int(Xtr.shape[1]) if reader == "MLP128" else av.INPUT_DIM
        milestones = {str(u): next((e for e in fit["events"] if e["update"] == u), None)
                      for u in MILESTONES}
        out["fits"][f"{reader}|{rep}"] = {
            "params": _param_count(reader, indim, arch), "n_train": int(len(sub_rows)),
            "updates": UPDATES,
            "fit": {"train_acc_max": float(fit["best_train"][0]),
                    "train_acc_max_update": int(fit["best_train"][2]),
                    "train_acc_final": float(fit["train_acc_final"]),
                    "train_loss_final": float(fit["train_loss_final"]),
                    "val_acc_final": float(fit["val_acc_final"]),
                    "val_loss_final": float(fit["val_loss_final"]),
                    "best_val_acc": float(fit["best_val"][0]),
                    "best_val_update": int(fit["best_val"][2]),
                    "milestones": milestones},
            "final": eval_state(fit["final"], indim),
            "best_val": eval_state(fit["best_val"][1], indim),
            "best_train": eval_state(fit["best_train"][1], indim),
        }
        out["traj"][rep] = fit["events"]
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _ms(vals):
    return as_.mean_sd(vals)


def aggregate(per: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out: Dict[str, Any] = {}
    for reader in READERS:
        recs = [s for s in per if s["reader"] == reader]
        if not recs:
            continue
        for k in sorted(recs[0]["fits"].keys()):
            _, rep = k.split("|")
            fits = [s["fits"][k] for s in recs]
            node = out.setdefault(reader, {}).setdefault(rep, {})
            node["params"] = fits[0]["params"]
            node["n_train"] = fits[0]["n_train"]
            node["updates"] = fits[0]["updates"]
            node["fit"] = {
                "train_acc_max": _ms([f["fit"]["train_acc_max"] for f in fits]),
                "train_acc_max_update": _ms([f["fit"]["train_acc_max_update"] for f in fits]),
                "train_acc_final": _ms([f["fit"]["train_acc_final"] for f in fits]),
                "train_loss_final": _ms([f["fit"]["train_loss_final"] for f in fits]),
                "val_acc_final": _ms([f["fit"]["val_acc_final"] for f in fits]),
                "val_loss_final": _ms([f["fit"]["val_loss_final"] for f in fits]),
                "best_val_acc": _ms([f["fit"]["best_val_acc"] for f in fits]),
                "best_val_update": _ms([f["fit"]["best_val_update"] for f in fits]),
                "milestones": {str(u): {fld: _ms([f["fit"]["milestones"][str(u)][fld] for f in fits])
                                        for fld in ("train_acc", "train_loss", "val_acc", "val_loss")}
                               for u in MILESTONES},
            }
            for st in STATES:
                node[st] = {
                    "train_acc": _ms([f[st]["train_acc"] for f in fits]),
                    "val_acc": _ms([f[st]["val_acc"] for f in fits]),
                    "acc": {pk: _ms([f[st]["acc"][pk] for f in fits]) for pk in fits[0][st]["acc"]},
                    "seen": {pk: _ms([f[st]["seen"][pk] for f in fits]) for pk in fits[0][st]["seen"]},
                    "heldout": {pk: _ms([f[st]["heldout"][pk] for f in fits])
                                for pk in fits[0][st]["heldout"]},
                }
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
                "train_loss": [_ms([s["traj"][rep][i]["train_loss"] for s in recs]) for i in range(len(ups))],
                "val_acc": [_ms([s["traj"][rep][i]["val_acc"] for s in recs]) for i in range(len(ups))],
                "val_loss": [_ms([s["traj"][rep][i]["val_loss"] for s in recs]) for i in range(len(ups))]}
    return out, traj


# ---------------------------------------------------------------------------
# validation + decision
# ---------------------------------------------------------------------------

def _validate(meta: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    cc = meta.get("class_counts", {})
    for where in ("train", "val", "test"):
        for k, v in cc.get(where, {}).items():
            if len(v) != NCLASS or any(int(c) <= 0 for c in v):
                problems.append(f"class coverage failure: {where}/{k}={v}")
    for k, v in meta.get("disjointness", {}).items():
        if int(v) != 0:
            problems.append(f"split disjointness failure: {k}={v}")
    for k, rec in meta.get("recoverability", {}).items():
        if abs(float(rec.get("rate", 0.0)) - 1.0) > 1e-12:
            problems.append(f"op_dest recoverability failure: {k}={rec}")
    return problems


def _fit_m(res, reader, rep, field):
    return res["readers"][reader][rep]["fit"][field]["mean"]


def classify(res: Dict[str, Any], problems: List[str]) -> Dict[str, Any]:
    per_pair: Dict[str, Any] = {}
    for R in READERS:
        for rep in DECISIVE:
            tr_max = _fit_m(res, R, rep, "train_acc_max")
            tr_final = _fit_m(res, R, rep, "train_acc_final")
            ho = _state_m(res, R, rep, "best_val", "heldout", f"{PRIMARY_MODEL}:classA")
            tst = _state_m(res, R, rep, "best_val", "acc", f"{PRIMARY_MODEL}:all_add")
            per_pair[f"{R}|{rep}"] = {"train_acc_max": tr_max, "train_acc_final": tr_final,
                                      "fits": bool(tr_max is not None and tr_max >= FIT_BAR),
                                      "heldout_classA": ho, "test_alladd": tst}
    control_ok = all(per_pair[f"GRU|{rep}"]["fits"] for rep in DECISIVE)
    tcn_fits = {rep: per_pair[f"TCN|{rep}"]["fits"] for rep in DECISIVE}
    if problems:
        verdict = "Z-INVALID"
    elif not control_ok:
        verdict = "Z-INVALID"
    elif all(tcn_fits.values()):
        verdict = "Z-A"
    elif not any(tcn_fits.values()):
        verdict = "Z-B"
    else:
        verdict = "Z-MIX"
    return {"verdict": verdict, "per_pair": per_pair, "control_ok": control_ok,
            "tcn_fits": tcn_fits, "validation_problems": problems,
            "thresholds": {"fit_bar": FIT_BAR},
            "rule_note": ("Z-INVALID if a class/split/recoverability invariant fails or the GRU "
                          "control is <0.90 at some history; else Z-A if the TCN reaches >=0.90 train "
                          "at BOTH L4 and Lfull; else Z-B if the TCN is <0.90 at BOTH; else Z-MIX "
                          "(one history only). fit_bar=0.90 on the maximum observed training accuracy. "
                          "Bars are declared conventions; continuous values reported.")}


def _collect(data: Dict[str, Any], jobs: int, readers: List[str]) -> List[Dict[str, Any]]:
    npz_path = data["path"]
    arch = av.architecture()
    specs = [(s, R) for s in SEEDS for R in readers]
    t0 = time.perf_counter()
    if jobs <= 1:
        _init_worker(npz_path, arch)
        return sorted([_task(sp) for sp in specs], key=lambda r: (r["seed"], r["reader"]))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs, initializer=_init_worker, initargs=(npz_path, arch)) as pool:
        out = list(pool.imap_unordered(_task, specs, chunksize=1))
    print(f"collection wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    return sorted(out, key=lambda r: (r["seed"], r["reader"]))


def _state_m(res, reader, rep, state, field, pop):
    return res["readers"][reader][rep][state][field][pop]["mean"]


def assemble(data: Dict[str, Any], per: List[Dict[str, Any]], y_meta: Dict[str, Any]) -> Dict[str, Any]:
    arch = av.architecture()
    readers, traj = aggregate(per)
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "readers": READERS,
        "target": TARGET, "n_classes": NCLASS, "reps": DECISIVE, "states": STATES,
        "updates": UPDATES, "eval_every": EVAL_EVERY, "milestones": MILESTONES,
        "subset_size": data["meta"]["subset_size"], "subset_seed": data["meta"]["subset_seed"],
        "subset_per_seed": data["meta"]["n"]["subset_per_seed"], "populations": POPULATIONS,
        "heldout_pairs": sorted(HELD_OUT_PAIRS),
        "reader_lr": av.READER_LR, "reader_wd": av.READER_WD,
        "source_identity": data["meta"]["source"], "class_counts": data["meta"]["class_counts"],
        "recoverability": data["meta"]["recoverability"], "disjointness": data["meta"]["disjointness"],
        "y_data_identity": y_meta.get("source"), "checkpoints": data["meta"]["models"],
        "n": data["meta"]["n"], "thresholds": {"fit_bar": FIT_BAR},
        "head_note": ("output head is Linear(width, 4) for op_dest (4 classes) instead of "
                      "Linear(width, 10) for y_true; the temporal reader body, causal padding, input "
                      "features and validity-mask semantics are unchanged"),
        "statistical_discipline": (
            "Five frozen E checkpoints are the independent units; F is a control. The two readers are "
            "analysis-only, fit on Y's EXACT fixed 256-transition subset with Y's exact protocol, "
            "tuned on Y's canonical VAL, evaluated once on Y's untouched canonical TEST. Checkpoint "
            "selection is VAL-only; the best-on-TRAIN state is a diagnostic and is never used to "
            "select a deployable model; TEST is never used to select, tune or adjust. op_dest is "
            "used as a label only.")}}
    res["architecture"] = arch
    res["readers"] = readers
    res["trajectory"] = traj
    res["per_seed"] = per
    res["y_meta"] = y_meta
    problems = _validate(data["meta"])
    res["primary_decision"] = classify(res, problems)
    return res


def build(data: Dict[str, Any], y_meta: Dict[str, Any], jobs: int,
          readers: Optional[List[str]] = None) -> Dict[str, Any]:
    return assemble(data, _collect(data, jobs, readers or READERS), y_meta)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_z") -> List[str]:
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
    colors = {"GRU": "tab:blue", "TCN": "tab:orange"}

    def fit_m(reader, rep, field):
        return nz(R[reader][rep]["fit"][field]["mean"])

    def st_m(reader, rep, state, field, pop=f"{PRIMARY_MODEL}:all_add"):
        return nz(R[reader][rep][state][field][pop]["mean"])

    groups = [f"{r}\n{rep}" for r in READERS for rep in DECISIVE]
    gx = np.arange(len(groups))

    # 1 training fit: max vs final training accuracy
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    best = [fit_m(r, rep, "train_acc_max") for r in READERS for rep in DECISIVE]
    final = [fit_m(r, rep, "train_acc_final") for r in READERS for rep in DECISIVE]
    ax.bar(gx - 0.2, best, 0.4, label="max training accuracy", color="tab:blue")
    ax.bar(gx + 0.2, final, 0.4, label="final training accuracy", color="tab:cyan")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.set_xticks(gx); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_xlabel("reader / history")
    ax.set_ylabel("accuracy on the 256-example subset")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Z: can the readers fit the directly observable op_dest target?")
    save(fig, "training_fit")

    # 2 trajectory
    T = res.get("trajectory", {})
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for reader in READERS:
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            node = T.get(reader, {}).get(rep)
            if not node:
                continue
            ax.plot(node["updates"], [v["mean"] for v in node["train_acc"]], ls=ls,
                    color=colors[reader], alpha=1.0 if rep == "L4" else 0.6,
                    label=f"{reader} {rep} train")
            ax.plot(node["updates"], [v["mean"] for v in node["val_acc"]], ls=":", color=colors[reader],
                    alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep} val")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.8, label="0.90 fit bar")
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.3)
    ax.set_title("Z: op_dest training / validation trajectory (256-example subset)")
    save(fig, "training_trajectory")

    # 3 reader comparison
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x2 = np.arange(len(READERS))
    ax.bar(x2 - 0.2, [max(fit_m(r, rep, "train_acc_max") for rep in DECISIVE) for r in READERS],
           0.4, label="max train (best history)", color="tab:blue")
    ax.bar(x2 + 0.2, [min(fit_m(r, rep, "train_acc_max") for rep in DECISIVE) for r in READERS],
           0.4, label="min train (worst history)", color="tab:purple")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.8, label="0.90 fit bar")
    ax.set_xticks(x2); ax.set_xticklabels(READERS); ax.set_ylim(0, 1.02)
    ax.set_ylabel("op_dest training accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Z: reader comparison on op_dest")
    save(fig, "reader_comparison")

    # 4 L4 vs full
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xi = np.arange(len(READERS))
    ax.bar(xi - 0.3, [fit_m(r, "L4", "train_acc_max") for r in READERS], 0.2, label="L4 train(max)",
           color="tab:blue")
    ax.bar(xi - 0.1, [st_m(r, "L4", "best_val", "acc") for r in READERS], 0.2, label="L4 test(all-ADD)",
           color="tab:cyan")
    ax.bar(xi + 0.1, [fit_m(r, "Lfull", "train_acc_max") for r in READERS], 0.2, label="full train(max)",
           color="tab:orange")
    ax.bar(xi + 0.3, [st_m(r, "Lfull", "best_val", "acc") for r in READERS], 0.2,
           label="full test(all-ADD)", color="tab:pink")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.8)
    ax.set_xticks(xi); ax.set_xticklabels(READERS); ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy"); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Z: L4 vs full-prefix (train fit and test accuracy)")
    save(fig, "l4_vs_full")

    # 5 per-seed
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xs = np.arange(len(SEEDS))
    for reader in READERS:
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            vals = [s["fits"][f"{reader}|{rep}"]["fit"]["train_acc_max"]
                    for s in res["per_seed"] if s["reader"] == reader]
            ax.plot(xs, vals, marker="o", ls=ls, color=colors[reader],
                    alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep}")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.8, label="0.90 fit bar")
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in SEEDS]); ax.set_xlabel("E seed")
    ax.set_ylabel("max op_dest training accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("Z: per-seed op_dest fitting")
    save(fig, "per_seed")

    # 6 test accuracy by state
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    states = ["best_val", "final", "best_train"]
    w = 0.25
    xi = np.arange(len(READERS) * len(DECISIVE))
    for si, st in enumerate(states):
        vals = [st_m(r, rep, st, "acc") for r in READERS for rep in DECISIVE]
        ax.bar(xi + (si - 1) * w, vals, w, label=st)
    ax.set_xticks(xi); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("op_dest TEST accuracy (all-ADD)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Z: op_dest test accuracy by reported state")
    save(fig, "test_accuracy")

    # 7 direct-observation control summary (train max vs test per reader, best history)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for r in READERS:
        btr = max(fit_m(r, rep, "train_acc_max") for rep in DECISIVE)
        bte = max(st_m(r, rep, "best_val", "acc") for rep in DECISIVE)
        ax.scatter(btr, bte, color=colors[r], s=80, zorder=3, label=f"{r}")
        ax.annotate(r, (btr, bte), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=0.8, label="train = test")
    ax.axvline(FIT_BAR, ls="--", c="k", lw=0.8)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("max op_dest training accuracy (256-example subset)")
    ax.set_ylabel("op_dest TEST accuracy (best-on-VAL, all-ADD)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Z: direct-observation control — fit vs test")
    save(fig, "op_dest_control")

    # 8 decision summary
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    d = res["primary_decision"]
    cats = ["GRU max\ntrain", "TCN max\ntrain (L4)", "TCN max\ntrain (full)", "TCN test\n(all-ADD, best-VAL)"]
    vals = [nz(d["per_pair"]["GRU|L4"]["train_acc_max"]),
            nz(d["per_pair"]["TCN|L4"]["train_acc_max"]),
            nz(d["per_pair"]["TCN|Lfull"]["train_acc_max"]),
            max(nz(d["per_pair"][f"TCN|{rep}"]["test_alladd"]) for rep in DECISIVE)]
    ax.bar(np.arange(4), vals, color=["tab:blue", "tab:orange", "tab:orange", "tab:red"])
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Z: decision summary ({d['verdict']})")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    global TOP
    ap = argparse.ArgumentParser(description="Experiment Z analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_z_op_dest.npz")
    ap.add_argument("--y-npz", default="experiments/experiment_y_fitting_control.npz")
    ap.add_argument("--output", default="experiments/experiment_z_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_z")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--readers", nargs="*", default=None, help="subset of readers (GRU TCN); default both")
    args = ap.parse_args()
    data = load(args.npz)
    data["path"] = args.npz
    TOP = int(data["meta"]["source"]["top_size"])
    aw.TRAIN_SIZES = list(data["meta"]["source"]["train_sizes"])
    yz = np.load(args.y_npz, allow_pickle=False)
    y_meta = json.loads(str(yz["meta_json"][0]))
    t0 = time.perf_counter()
    res = build(data, y_meta, args.jobs, readers=args.readers)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"verdict: {d['verdict']}  control_ok={d['control_ok']}  tcn_fits={d['tcn_fits']}")
    if d["validation_problems"]:
        print(f"  validation problems: {d['validation_problems']}")
    for k, v in d["per_pair"].items():
        print(f"  {k}: train_max={v['train_acc_max']:.3f} final={v['train_acc_final']:.3f} "
              f"fits={v['fits']} test_alladd={v['test_alladd']}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
