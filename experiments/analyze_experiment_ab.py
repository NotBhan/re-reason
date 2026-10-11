#!/usr/bin/env python3
"""Experiment AB — analysis: TCN optimizer-family and initialization diagnostic on `op_dest`.

Z/AA show the causal TCN cannot fit the directly observable `op_dest` control under the tested Adam
learning-rate/update-budget grid, while the GRU control fits it perfectly. AB runs a **predeclared**
optimizer-family comparison (AdamW, SGD+momentum, SGD+Nesterov, RMSprop) and a **single** alternative
initialization (xavier-uniform conv/head weights + zero biases, under Adam) at fixed lr 3e-3 and
5,000 updates, with a deterministic Adam/default reference cell that must reproduce AA's existing
Adam result.

PRE-REGISTRATION (declared before aggregate inspection)
-------------------------------------------------------
* Conditions (each fully specifies optimizer family + initialization), all at lr 3e-3, budget 5,000,
  weight decay 1e-2 where supported:
      adam_ref     : Adam,  default init        (deterministic reference; must reproduce AA)
      adamw        : AdamW, default init
      sgd_mom      : SGD(momentum=0.9, nesterov=False), default init
      sgd_nesterov : SGD(momentum=0.9, nesterov=True),  default init
      rmsprop      : RMSprop(defaults alpha 0.99, eps 1e-8, momentum 0), default init
      adam_xavier  : Adam, xavier-uniform conv/head weights + zero biases
  The initialization condition is compared only under Adam (optimizer and initialization questions
  kept separately interpretable). Optimizer-specific settings are recorded; effective regularization
  is NOT claimed identical across families. No further optimizer types are added after results.
* Fit criterion `fit_bar = 0.90`; a condition is ROBUST at a history iff the five-seed mean maximum
  training accuracy >= 0.90 AND >= 4/5 seeds reach 0.90.
* Decision (checked in order):
    AB-INVALID : preflight/invariant failure, OR the adam_ref GRU control fails to robustly fit at a
                 history, OR adam_ref does not reproduce AA's Adam baseline.
    AB-A       : some condition is robust for the TCN at BOTH L4 and Lfull.
    AB-MIX     : some condition is robust at exactly one history, or reaches mean >= 0.90 without the
                 4/5-seed criterion.
    AB-B       : no condition meets the robust criterion at either history.
* Checkpoint selection is VAL-only; best-on-TRAIN is diagnostic. TEST never selects.
* AA's GRU result is reused unchanged as historical reference evidence.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_ab.py \
        --npz experiments/experiment_ab_optimizer_init.npz \
        --output experiments/experiment_ab_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_ab
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

TARGET = "op_dest"
NCLASS = 4
SUBSET_SIZE = 256
LR = 3e-3
WD = 1e-2
UPDATES = 5000
EVAL_EVERY = 10
MILESTONES = [200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000]
FIT_BAR = 0.90
ROBUST_SEEDS = 4

CONDITIONS = [
    {"name": "adam_ref", "opt": "adam", "init": "default"},
    {"name": "adamw", "opt": "adamw", "init": "default"},
    {"name": "sgd_mom", "opt": "sgd", "init": "default", "momentum": 0.9, "nesterov": False},
    {"name": "sgd_nesterov", "opt": "sgd", "init": "default", "momentum": 0.9, "nesterov": True},
    {"name": "rmsprop", "opt": "rmsprop", "init": "default"},
    {"name": "adam_xavier", "opt": "adam", "init": "xavier"},
]
COND_NAMES = [c["name"] for c in CONDITIONS]
COND_BY_NAME = {c["name"]: c for c in CONDITIONS}

OPTIMIZER_DEFAULTS = {
    "adam": {"betas": [0.9, 0.999], "eps": 1e-8},
    "adamw": {"betas": [0.9, 0.999], "eps": 1e-8},
    "sgd": {"momentum": 0.9, "dampening": 0.0, "nesterov": "per-condition"},
    "rmsprop": {"alpha": 0.99, "eps": 1e-8, "momentum": 0.0, "centered": False},
}

TOP = 10000
_G: Dict[str, Any] = {}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def _fit_seed(seed: int, reader: str, rep: str) -> int:
    return aw.reader_seed(seed, reader, TARGET, rep, TOP, "main")


def apply_init(model, scheme: str) -> None:
    """Deterministic initialization scheme. 'default' = leave PyTorch's own init in place."""
    import torch.nn as nn
    if scheme == "default":
        return
    if scheme != "xavier":
        raise ValueError(scheme)
    for mod in model.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.GRU):
            # analogous treatment for the GRU control: xavier on the recurrent weight matrices,
            # zero on biases (the declared scheme is TCN-specific; this keeps the control coherent)
            for name, p in mod.named_parameters():
                if name.endswith("weight_ih_l0") or name.endswith("weight_hh_l0"):
                    nn.init.xavier_uniform_(p)
                elif name.endswith("bias_ih_l0") or name.endswith("bias_hh_l0"):
                    nn.init.zeros_(p)


def make_optimizer(model, cond: Dict[str, Any]):
    import torch
    opt = cond["opt"]
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    if opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    if opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WD,
                               momentum=cond.get("momentum", 0.9),
                               nesterov=cond.get("nesterov", False))
    if opt == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=WD)
    raise ValueError(opt)


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

def train_ab(reader: str, Xtr, ytr, Xva, yva, nclass: int, seed: int, updates: int,
             cond: Dict[str, Any], arch: Dict[str, Any], eval_every: int = EVAL_EVERY) -> Dict[str, Any]:
    """Y/AA protocol with a predeclared optimizer family + initialization; GRU/TCN only."""
    import torch
    import torch.nn.functional as F
    assert reader in ("GRU", "TCN"), reader
    torch.manual_seed(seed)
    Xn = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    Xv = torch.tensor(np.asarray(Xva, dtype=np.float32))
    model = ax.build_model(reader, nclass, av.INPUT_DIM, arch)
    apply_init(model, cond["init"])
    opt = make_optimizer(model, cond)
    yn = torch.tensor(ytr, dtype=torch.long)
    yv = torch.tensor(yva, dtype=torch.long)
    wc = torch.tensor(as_.class_weights(ytr, nclass), dtype=torch.float32)
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
        return {"train_acc": float((pred_t == ytr).mean()), "val_acc": float((pred_v == yva).mean()),
                "acc": acc}

    fit = train_ab(reader, Xtr, ytr, Xva, yva, NCLASS, _fit_seed(seed, reader, rep), UPDATES, cond, arch)
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
    return {"seed": seed, "reader": reader, "rep": rep, "cond": cond_name, "cell": cell,
            "events": fit["events"]}


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
                node["all_finite"] = bool(all(c["fit"]["finite"] for c in cells))
                ms_keys = sorted(cells[0]["fit"]["milestones"].keys(), key=int)
                node["milestones"] = {u: {fld: _ms([c["fit"]["milestones"][u][fld] for c in cells])
                                          for fld in ("train_acc", "train_loss", "val_acc")}
                                      for u in ms_keys}
                node["best_val"] = {"train_acc": _ms([c["best_val"]["train_acc"] for c in cells]),
                                    "val_acc": _ms([c["best_val"]["val_acc"] for c in cells]),
                                    "acc": {pk: _ms([c["best_val"]["acc"][pk] for c in cells])
                                            for pk in cells[0]["best_val"]["acc"]}}
                node["final"] = {"train_acc": _ms([c["final"]["train_acc"] for c in cells]),
                                 "val_acc": _ms([c["final"]["val_acc"] for c in cells]),
                                 "acc": {pk: _ms([c["final"]["acc"][pk] for c in cells])
                                         for pk in cells[0]["final"]["acc"]}}
    # trajectories for the reference + best conditions
    traj: Dict[str, Any] = {}
    for reader in READERS:
        for cond_name in COND_NAMES:
            for rep in DECISIVE:
                evs = [p["events"] for p in parts if p["reader"] == reader and p["rep"] == rep
                       and p["cond"] == cond_name]
                if not evs:
                    continue
                evs = sorted(evs, key=lambda e: 0)
                ups = [e["update"] for e in evs[0]]
                traj.setdefault(f"{reader}|{cond_name}", {})[rep] = {
                    "updates": ups,
                    "train_acc": [_ms([e[i]["train_acc"] for e in evs]) for i in range(len(ups))],
                    "val_acc": [_ms([e[i]["val_acc"] for e in evs]) for i in range(len(ups))]}
    return out, traj


def load_aa_baseline(path: str) -> Dict[str, Any]:
    with open(path) as f:
        aa = json.load(f)

    def cell(reader, rep):
        node = aa["readers"][reader][rep]["3e-3"]
        return node["5000"] if "5000" in node else node[5000]

    out = {"source": path}
    for R in READERS:
        out[R] = {}
        for rep in DECISIVE:
            c = cell(R, rep)
            out[R][rep] = {"train_max": c["train_max"]["mean"],
                           "per_seed_train_max": c.get("per_seed_train_max"),
                           "train_final": c["train_final"]["mean"],
                           "test_alladd": c["best_val"]["acc"][f"{PRIMARY_MODEL}:all_add"]["mean"]}
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


def classify(res: Dict[str, Any], problems: List[str], aa_base: Dict[str, Any]) -> Dict[str, Any]:
    minimal = {"verdict": "AB-INVALID", "control_ok": False, "reproduce_aa": None,
               "tcn_robust": {rep: [] for rep in DECISIVE},
               "tcn_mean_only": {rep: [] for rep in DECISIVE},
               "gru_robust": {rep: [] for rep in DECISIVE},
               "qualified_conditions": [], "validation_problems": problems,
               "thresholds": {"fit_bar": FIT_BAR, "robust_seeds": ROBUST_SEEDS}}
    if problems:
        return minimal
    # control = adam_ref GRU robust at both histories; also verify adam_ref reproduces AA exactly
    control_ok = _robust(res, "GRU", "L4", "adam_ref") and _robust(res, "GRU", "Lfull", "adam_ref")
    repro = True
    for R in READERS:
        for rep in DECISIVE:
            got = res["readers"][R][rep]["adam_ref"]["train_max"]["mean"]
            exp = aa_base[R][rep]["train_max"]
            if got is None or exp is None or abs(got - exp) > 1e-9:
                repro = False
    tcn = {rep: [c for c in COND_NAMES if _robust(res, "TCN", rep, c)] for rep in DECISIVE}
    tcn_near = {rep: [c for c in COND_NAMES if _mean_only(res, "TCN", rep, c)] for rep in DECISIVE}
    gru = {rep: [c for c in COND_NAMES if _robust(res, "GRU", rep, c)] for rep in DECISIVE}
    both = [c for c in COND_NAMES if c in tcn["L4"] and c in tcn["Lfull"]]
    if not control_ok or not repro:
        verdict = "AB-INVALID"
    elif both:
        verdict = "AB-A"
    elif tcn["L4"] or tcn["Lfull"] or tcn_near["L4"] or tcn_near["Lfull"]:
        verdict = "AB-MIX"
    else:
        verdict = "AB-B"
    return {"verdict": verdict, "control_ok": control_ok, "reproduce_aa": repro,
            "tcn_robust": tcn, "tcn_mean_only": tcn_near, "gru_robust": gru,
            "qualified_conditions": both, "validation_problems": problems,
            "thresholds": {"fit_bar": FIT_BAR, "robust_seeds": ROBUST_SEEDS},
            "rule_note": ("AB-INVALID if preflight/invariant fails, or the adam_ref GRU control is "
                          "not robust at both histories, or adam_ref does not reproduce AA's Adam "
                          "baseline; else AB-A if some condition is robust for the TCN at BOTH "
                          "histories; else AB-MIX if robust at one history or mean>=0.90 without the "
                          "4/5-seed criterion; else AB-B. Robust = five-seed mean max-train >=0.90 AND "
                          ">=4/5 seeds >=0.90.")}


def _validate(meta: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if not meta.get("preflight_pass", False):
        problems.append("trainable-parameter preflight failed")
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


def assemble(data: Dict[str, Any], parts: List[Dict[str, Any]], aa_base: Dict[str, Any]) -> Dict[str, Any]:
    arch = av.architecture()
    readers, traj = aggregate(parts)
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "readers": READERS,
        "target": TARGET, "n_classes": NCLASS, "reps": DECISIVE, "states": STATES,
        "lr": LR, "weight_decay": WD, "updates": UPDATES, "eval_every": EVAL_EVERY,
        "milestones": MILESTONES, "conditions": CONDITIONS, "optimizer_defaults": OPTIMIZER_DEFAULTS,
        "subset_size": data["meta"]["subset_size"], "subset_seed": data["meta"]["subset_seed"],
        "populations": POPULATIONS, "source_identity": data["meta"]["source"],
        "class_counts": data["meta"]["class_counts"], "recoverability": data["meta"]["recoverability"],
        "disjointness": data["meta"]["disjointness"], "preflight": data["meta"]["preflight"],
        "preflight_pass": data["meta"]["preflight_pass"], "checkpoints": data["meta"]["models"],
        "n": data["meta"]["n"], "thresholds": {"fit_bar": FIT_BAR, "robust_seeds": ROBUST_SEEDS},
        "matched_init": ("each grid cell resets to the same deterministic init seed and applies the "
                         "declared initialization; optimizer-family and initialization questions are "
                         "kept separately interpretable (the alternative init is compared only under Adam)"),
        "statistical_discipline": (
            "Five frozen E checkpoints are the independent units; F is a control. The two readers are "
            "analysis-only, fit on Z/Y's exact fixed 256-transition subset, tuned on Y's canonical VAL, "
            "evaluated once on Y's untouched canonical TEST. Selection is VAL-only; the best-on-TRAIN "
            "state is diagnostic; TEST is never used to select a condition/reader/history.")}}
    res["architecture"] = arch
    res["readers"] = readers
    res["trajectory"] = traj
    res["per_seed"] = parts
    res["aa_baseline"] = aa_base
    problems = _validate(data["meta"])
    res["primary_decision"] = classify(res, problems, aa_base)
    return res


def build(data: Dict[str, Any], aa_base: Dict[str, Any], jobs: int,
          conditions: Optional[List[str]] = None, readers: Optional[List[str]] = None) -> Dict[str, Any]:
    parts = _collect(data, jobs, conditions or COND_NAMES, readers or READERS)
    return assemble(data, parts, aa_base)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_ab") -> List[str]:
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
    A = res["aa_baseline"]
    cx = np.arange(len(COND_NAMES))

    def tm(reader, rep, cond):
        v = R[reader][rep][cond]["train_max"]["mean"]
        return np.nan if v is None else v

    # 1 optimizer comparison (TCN)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    opt_conds = [c for c in COND_NAMES if c != "adam_xavier"]
    ox = np.arange(len(opt_conds))
    ax.bar(ox - 0.2, [tm("TCN", "L4", c) for c in opt_conds], 0.4, label="L4", color="tab:blue")
    ax.bar(ox + 0.2, [tm("TCN", "Lfull", c) for c in opt_conds], 0.4, label="Lfull", color="tab:orange")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.set_xticks(ox); ax.set_xticklabels(opt_conds, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("max training accuracy (op_dest)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AB: TCN by optimizer family (lr 3e-3, 5000 updates)")
    save(fig, "optimizer_comparison")

    # 2 initialization comparison (Adam)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    vals_def = [tm("TCN", rep, "adam_ref") for rep in DECISIVE]
    vals_xav = [tm("TCN", rep, "adam_xavier") for rep in DECISIVE]
    xi = np.arange(len(DECISIVE))
    ax.bar(xi - 0.2, vals_def, 0.4, label="default init", color="tab:blue")
    ax.bar(xi + 0.2, vals_xav, 0.4, label="xavier init", color="tab:green")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.set_xticks(xi); ax.set_xticklabels(DECISIVE); ax.set_ylim(0, 1.02)
    ax.set_ylabel("TCN max training accuracy (Adam)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AB: TCN initialization comparison (Adam, lr 3e-3, 5000 updates)")
    save(fig, "init_comparison")

    # 3 GRU control by condition
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(ox - 0.2, [tm("GRU", "L4", c) for c in opt_conds], 0.4, label="L4", color="tab:blue")
    ax.bar(ox + 0.2, [tm("GRU", "Lfull", c) for c in opt_conds], 0.4, label="Lfull", color="tab:orange")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9)
    ax.set_xticks(ox); ax.set_xticklabels(opt_conds, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("GRU max training accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AB: GRU control by optimizer family")
    save(fig, "gru_control")

    # 4 per-seed at the best TCN condition
    best = (-1.0, None)
    for rep in DECISIVE:
        for c in COND_NAMES:
            v = tm("TCN", rep, c)
            if v is not None and not np.isnan(v) and v > best[0]:
                best = (v, (rep, c))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    if best[1]:
        rep, c = best[1]
        ax.bar(np.arange(len(SEEDS)), R["TCN"][rep][c]["per_seed_train_max"], color="tab:orange")
        ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
        ax.set_title(f"AB: TCN per-seed max train acc (best: {rep}/{c}, mean={best[0]:.3f})")
    ax.set_xticks(np.arange(len(SEEDS))); ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("E seed"); ax.set_ylabel("max training accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    save(fig, "per_seed")

    # 5 train/val curves at adam_ref
    T = res.get("trajectory", {})
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for reader, c in (("TCN", "tab:orange"), ("GRU", "tab:blue")):
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            node = T.get(f"{reader}|adam_ref", {}).get(rep)
            if not node:
                continue
            ax.plot(node["updates"], [v["mean"] for v in node["train_acc"]], ls=ls, color=c,
                    alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep} train")
            ax.plot(node["updates"], [v["mean"] for v in node["val_acc"]], ls=":", color=c,
                    alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep} val")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.8)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.3)
    ax.set_title("AB: training/validation trajectories (Adam, default init)")
    save(fig, "train_val_curves")

    # 6 TCN trajectories across conditions (L4)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for c in COND_NAMES:
        node = T.get(f"TCN|{c}", {}).get("L4")
        if not node:
            continue
        ax.plot(node["updates"], [v["mean"] for v in node["train_acc"]], label=c)
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9)
    ax.set_xlabel("optimizer updates"); ax.set_ylabel("TCN train accuracy (L4)"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("AB: TCN training trajectory by condition (L4)")
    save(fig, "tcn_trajectories")

    # 7 reproduce-AA bridge
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    labels, got, exp = [], [], []
    for Rr in READERS:
        for rep in DECISIVE:
            labels.append(f"{Rr}\n{rep}")
            got.append(tm(Rr, rep, "adam_ref"))
            exp.append(A[Rr][rep]["train_max"])
    xi = np.arange(len(labels))
    ax.bar(xi - 0.2, got, 0.4, label="AB adam_ref", color="tab:blue")
    ax.bar(xi + 0.2, exp, 0.4, label="AA Adam (3e-3, 5000)", color="tab:gray")
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9)
    ax.set_xticks(xi); ax.set_xticklabels(labels, fontsize=7); ax.set_ylim(0, 1.02)
    ax.set_ylabel("max training accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AB: adam_ref reproduces the AA Adam baseline")
    save(fig, "reproduce_aa")

    # 8 decision summary
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    d = res["primary_decision"]
    tcn_best = max((tm("TCN", rep, c) for rep in DECISIVE for c in COND_NAMES
                    if not np.isnan(tm("TCN", rep, c))), default=np.nan)
    gru_best = max((tm("GRU", rep, c) for rep in DECISIVE for c in COND_NAMES
                    if not np.isnan(tm("GRU", rep, c))), default=np.nan)
    tcn_test = np.nan
    for rep in DECISIVE:
        for c in COND_NAMES:
            v = R["TCN"][rep][c]["best_val"]["acc"][f"{PRIMARY_MODEL}:all_add"]["mean"]
            if v is not None and (np.isnan(tcn_test) or v > tcn_test):
                tcn_test = v
    ax.bar(np.arange(3), [tcn_best, gru_best, tcn_test], color=["tab:orange", "tab:blue", "tab:red"])
    ax.axhline(FIT_BAR, ls="--", c="k", lw=0.9, label="0.90 fit bar")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["TCN best\n(max train)", "GRU best\n(max train)",
                                                     "TCN test\n(all-ADD)"], fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"AB: decision summary ({d['verdict']})")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    global TOP
    ap = argparse.ArgumentParser(description="Experiment AB analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_ab_optimizer_init.npz")
    ap.add_argument("--aa-results", default="experiments/experiment_aa_results.json")
    ap.add_argument("--output", default="experiments/experiment_ab_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_ab")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--raw", default="experiments/experiment_ab_grid_raw.json")
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
    problems = _validate(data["meta"])
    if args.preflight_only or problems:
        print(json.dumps({"preflight_pass": data["meta"]["preflight_pass"], "problems": problems,
                          "preflight": data["meta"]["preflight"]}, indent=2))
        if problems and not args.preflight_only:
            res = assemble(data, [], load_aa_baseline(args.aa_results))
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(res, f, indent=2)
            print(f"INVALID: {problems}; wrote {args.output} verdict {res['primary_decision']['verdict']}")
        return
    aa_base = load_aa_baseline(args.aa_results)
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
    res = assemble(data, parts, aa_base)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"verdict: {d['verdict']}  control_ok={d['control_ok']} reproduce_aa={d['reproduce_aa']}")
    print(f"  TCN robust L4={d['tcn_robust']['L4']} full={d['tcn_robust']['Lfull']} mean_only={d['tcn_mean_only']}")
    print(f"  GRU robust L4={d['gru_robust']['L4']} full={d['gru_robust']['Lfull']}")
    for reader in READERS:
        for rep in DECISIVE:
            print(f"  {reader} {rep}: " + " ".join(
                f"{c}:{res['readers'][reader][rep][c]['train_max']['mean']:.3f}" for c in COND_NAMES))
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
