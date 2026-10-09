#!/usr/bin/env python3
"""Experiment W — analysis: bounded-reader learnability vs training-data scale.

V (V-B) showed that U's flat MLP-128 and V's causal GRU/TCN do not access the raw-interface
information that is exactly identifiable from a 4-step history. W manipulates ONE factor — the
number of unique ordinary-ADD training transitions (N_train in {1500, 8000, 32000}) — holding the
reader architectures, widths, optimizer, LR, weight decay, epoch count (200, full-batch), history
representation, held-out protocol, VAL and TEST exactly fixed (reused verbatim from U/V).

Predeclared reader set (REUSED, not reconstructed): U flat MLP-128 (29,578 params), V causal GRU
(hidden 85), V causal TCN (channels 52, kernel 3, dilations [1,2,4,8]).

PRE-REGISTRATION (declared before aggregate inspection)
-------------------------------------------------------
* Primary target y_true; secondary operand_a, operand_b; op_dest as a compact sanity control.
  History grid = U/V {L1,L2,L4,L8,L16,Lfull}; y_true main at all lengths; operands/op_dest at the
  decisive horizons {L4, Lfull}; held-out at {L4, Lfull}.
* Fit on the fresh ordinary-ADD pool of size N_train (held-out pairs {(a,b): b in {0,5}} removed for
  the held-out variant), tune on the fixed canonical VAL (800/seed), evaluate once on the untouched
  canonical TEST. Optimizer updates = 200 for every scale (epochs fixed; do not compensate).
* Populations: Class-A (primary), matched candidate-correct, all candidate-correct, all ADD.
  E seeds 42..46 are the independent units; F is a control.
* Data-scale gain: gain_full = accuracy(N_train_max) - accuracy(N_train_min) over the predeclared
  ladder (TRAIN_SIZES), pooled over the 5 E seeds.
* Decision:
    W-A data-limited : some reader at L4/full has test >= 0.50 AND held-out >= 0.30 AND gain_full >= 0.15.
    W-B data-saturated/bounded : best top-scale Class-A test <= 0.25 AND held-out <= 0.25 AND
        Bayes-reader >= 0.30 AND every reader's gain_full < 0.15.
    W-C data-benefit-without-recovery : some reader gain_full >= 0.15 (or an intermediate best
        top-scale test > 0.25) without W-A.
    W-MIX : E/F material disagreement, or genuinely contradictory reader conclusions.
No scale-dependent tuning; no TEST-based selection.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_w.py \
        --npz experiments/experiment_w_data_scale.npz \
        --u-results experiments/experiment_u_results.json \
        --output experiments/experiment_w_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_w
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
import run_experiment_w as rw  # noqa: E402

PRIMARY_MODEL = av.PRIMARY_MODEL
CONTROL_MODEL = av.CONTROL_MODEL
SEEDS = av.SEEDS
TRAIN_SIZES = rw.TRAIN_SIZES
READERS = ["MLP128", "GRU", "TCN"]
REPS = av.REPS
REP_LABELS = av.REP_LABELS
DECISIVE = ["L4", "Lfull"]
TARGETS = {"y_true": 10, "operand_a": 10, "operand_b": 10, "op_dest": 4}
MAIN_SCOPE = {"y_true": REP_LABELS, "operand_a": DECISIVE, "operand_b": DECISIVE, "op_dest": DECISIVE}
HELDOUT_SCOPE = {"y_true": DECISIVE, "operand_a": DECISIVE, "operand_b": DECISIVE}
POPULATIONS = ["classA", "matched_correct", "correct_all", "all_add"]
HELD_OUT_PAIRS = au.HELD_OUT_PAIRS
MODEL_FAMILY = {"MLP128": "MLP128", "GRU": "GRU", "TCN": "TCN"}

RECOVER_BAR = 0.50
HELDOUT_WEAK = 0.25
HELDOUT_MATERIAL = 0.30
MATERIAL_GAIN = 0.15
BAYES_GAP_BAR = 0.30

# Compute device for the causal GRU reader. The GRU module and training loop are V's exactly; only
# the compute device is selectable (default CPU, matching U/V). The causal TCN and the flat MLP-128
# stay on CPU (V's / U's exact code), so the three readers differ only in device where configured.
GRU_DEVICE = os.environ.get("W_GRU_DEVICE", "cpu")

_G: Dict[str, Any] = {}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def reader_seed(seed: int, reader: str, target: str, rep: str, size: int, variant: str) -> int:
    """Deterministic fit seed (no TEST-derived component)."""
    ri = READERS.index(reader)
    ti = list(TARGETS).index(target)
    li = REP_LABELS.index(rep)
    si = TRAIN_SIZES.index(size)
    vi = 0 if variant == "main" else 1
    return as_.MLP_SEED + seed * 1_000_000 + ri * 100_000 + ti * 10_000 + li * 100 + si * 2 + vi


# ---------------------------------------------------------------------------
# worker: one (seed, reader) task, single-threaded, deterministic
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


def _train_reader_dev(reader: str, Xtr, ytr, Xva, yva, nclass: int, seed: int, device: str):
    """Exact copy of `analyze_experiment_v.train_reader` with an explicit compute device.

    Same optimizer, LR, weight decay, 200 full-batch epochs, class-balanced cross-entropy and
    best-on-VAL checkpoint; only the tensors/module are placed on `device`. Uses `av.build_reader`
    (V's reader definitions) so the architecture is unchanged.
    """
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    model = av.build_reader(reader, nclass, _G["arch"]).to(device)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.long, device=device)
    wc = torch.tensor(as_.class_weights(ytr, nclass), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=av.READER_LR, weight_decay=av.READER_WD)
    best = (-1.0, None)
    for ep in range(av.READER_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr_t), ytr_t, weight=wc)
        loss.backward()
        opt.step()
        if ep % 10 == 9:
            model.eval()
            with torch.no_grad():
                v = float((model(Xva_t).argmax(1) == yva_t).float().mean())
            if v > best[0]:
                best = (v, {k: val.clone() for k, val in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    return model, best[0]


def _predict_dev(model, X, device: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()


def _fit(reader: str, Xtr, ytr, Xva, yva, nclass: int, seed: int):
    """Fit the reader with the EXACT U/V protocol; return (predict_labels, val_acc, train_acc)."""
    if reader == "MLP128":
        Xtr = Xtr.astype(np.float64); Xva = Xva.astype(np.float64)
        wc = as_.class_weights(ytr, nclass)
        logits, vacc = as_.mlp_fit(Xtr, ytr, Xva, yva, 128, wc, seed, nclass)

        def pred(X):
            return logits(np.asarray(X, dtype=np.float64)).argmax(1)
    elif reader == "GRU" and GRU_DEVICE != "cpu":
        model, vacc = _train_reader_dev(reader, Xtr, ytr, Xva, yva, nclass, seed, GRU_DEVICE)

        def pred(X):
            return _predict_dev(model, X, GRU_DEVICE)
    else:
        model, vacc = av.train_reader(reader, Xtr, ytr, Xva, yva, nclass, seed, _G["arch"])

        def pred(X):
            return av.reader_predict(model, X)
    tracc = float((pred(Xtr) == ytr).mean())
    return pred, float(vacc), tracc


def _param_count(reader: str, nclass: int, L: int) -> int:
    return (as_.param_count("mlp", 128, 22 * L) if reader == "MLP128"
            else av.reader_param_count(reader, nclass, _G["arch"]))


def _task(spec: Tuple[int, str, str, str]) -> Dict[str, Any]:
    """Fit one (seed, reader, target, variant) family across its reps and sizes.

    Parallelisation is split at this granularity (rather than one task per reader) purely so the
    CPU work is load-balanced; the fitted model, seed, data, epochs and evaluation are identical
    to fitting the whole (seed, reader) block sequentially.
    """
    seed, reader, tgt, variant = spec
    _G["feat"] = {}                       # bound per-task feature cache memory
    z = _G["z"]
    nclass = TARGETS[tgt]
    te_sel = z[f"{PRIMARY_MODEL}__test__seed"] == seed
    te_idx = np.nonzero(te_sel)[0]
    held_te = au._pairs(z, PRIMARY_MODEL, "test", te_idx)
    masks = {m: {p: au._pop_masks(z, m, seed)[p][te_sel] for p in POPULATIONS}
             for m in (PRIMARY_MODEL, CONTROL_MODEL)}
    yte = {t: z[f"{PRIMARY_MODEL}__test__{t}"][te_sel].astype(int) for t in TARGETS}
    val_rows = np.nonzero(z["val__seed"] == seed)[0]
    reps = MAIN_SCOPE[tgt] if variant == "main" else HELDOUT_SCOPE.get(tgt, [])
    out: Dict[str, Any] = {"seed": seed, "reader": reader, "fits": {}}

    def eval_pops(pred_te):
        return {f"{m}:{p}": (float((pred_te[masks[m][p]] == yte[tgt][masks[m][p]]).mean())
                             if masks[m][p].any() else None)
                for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}

    for rep in reps:
        L = dict(REPS)[rep]
        Xte = _feat("test", seed, L, reader)
        Xva = _feat("val", seed, L, reader)
        yva = z[f"val__{tgt}"][val_rows].astype(int)
        for size in TRAIN_SIZES:
            rows = np.nonzero(z[f"pool{size}__seed"] == seed)[0]
            Xall = _feat(f"pool{size}", seed, L, reader)
            if variant == "main":
                Xtr, ytr = Xall, z[f"pool{size}__{tgt}"][rows].astype(int)
            else:
                oa = z[f"pool{size}__operand_a"][rows]
                ob = z[f"pool{size}__operand_b"][rows]
                held = np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oa, ob)])
                Xtr = Xall[~held]
                ytr = z[f"pool{size}__{tgt}"][rows[~held]].astype(int)
            pred, vacc, tracc = _fit(reader, Xtr, ytr, Xva, yva, nclass,
                                     reader_seed(seed, reader, tgt, rep, size, variant))
            pred_te = pred(Xte)
            params = _param_count(reader, nclass, L)
            if variant == "main":
                out["fits"][f"{reader}|{tgt}|{rep}|{size}|main"] = {
                    "params": params, "val_acc": vacc, "train_acc": tracc,
                    "n_train": int(len(rows)), "updates": av.READER_EPOCHS,
                    "acc": eval_pops(pred_te)}
            else:
                seen = {f"{m}:{p}": (float((pred_te[masks[m][p] & ~held_te] == yte[tgt][masks[m][p] & ~held_te]).mean())
                                     if (masks[m][p] & ~held_te).any() else None)
                        for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
                hold = {f"{m}:{p}": (float((pred_te[masks[m][p] & held_te] == yte[tgt][masks[m][p] & held_te]).mean())
                                     if (masks[m][p] & held_te).any() else None)
                        for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
                out["fits"][f"{reader}|{tgt}|{rep}|{size}|heldout"] = {
                    "params": params, "val_acc": vacc, "train_acc": tracc,
                    "n_train": int(len(rows) - held.sum()),
                    "n_excluded_heldout": int(held.sum()), "updates": av.READER_EPOCHS,
                    "seen": seen, "heldout": hold}
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _ms(vals):
    return as_.mean_sd(vals)


def aggregate(per: List[Dict[str, Any]], arch: Dict[str, Any]) -> Dict[str, Any]:
    # `per` holds one node per (seed, reader); group by reader so each key is averaged over seeds.
    out: Dict[str, Any] = {}
    for reader in READERS:
        recs_all = [s for s in per if s["reader"] == reader]
        if not recs_all:
            continue
        keys = sorted(recs_all[0]["fits"].keys())
        for k in keys:
            _, tgt, rep, size, variant = k.split("|")
            recs = [s["fits"][k] for s in recs_all]
            node = out.setdefault(reader, {}).setdefault(tgt, {}).setdefault(rep, {}).setdefault(size, {})
            if variant == "main":
                node["main"] = {
                    "params": recs[0]["params"], "updates": recs[0]["updates"],
                    "n_train": _ms([r["n_train"] for r in recs]),
                    "val_acc": _ms([r["val_acc"] for r in recs]),
                    "train_acc": _ms([r["train_acc"] for r in recs]),
                    "acc": {pk: _ms([r["acc"][pk] for r in recs]) for pk in recs[0]["acc"]}}
            else:
                node["heldout"] = {
                    "params": recs[0]["params"], "updates": recs[0]["updates"],
                    "n_train": _ms([r["n_train"] for r in recs]),
                    "n_excluded_heldout": _ms([r["n_excluded_heldout"] for r in recs]),
                    "val_acc": _ms([r["val_acc"] for r in recs]),
                    "train_acc": _ms([r["train_acc"] for r in recs]),
                    "seen": {pk: _ms([r["seen"][pk] for r in recs]) for pk in recs[0]["seen"]},
                    "heldout": {pk: _ms([r["heldout"][pk] for r in recs]) for pk in recs[0]["heldout"]}}
    return out


def _main_acc(res, reader, rep, size, pop):
    return res["readers"][reader]["y_true"][rep][str(size)]["main"]["acc"][pop]["mean"]


def _ho_acc(res, reader, rep, size, pop):
    return res["readers"][reader]["y_true"][rep][str(size)]["heldout"]["heldout"][pop]["mean"]


def load_u_baseline(path: str) -> Dict[str, Any]:
    with open(path) as f:
        u = json.load(f)
    return {p: {rep: u[p]["pooled"]["classA"][rep]["y_true"]["bayes_acc"] for rep, _ in REPS}
            for p in (PRIMARY_MODEL, CONTROL_MODEL)}


def classify(model: str, res: Dict[str, Any], u_base: Dict[str, Any]) -> Dict[str, Any]:
    pop = f"{model}:classA"
    per_reader: Dict[str, Any] = {}
    for R in READERS:
        best_top, best_rep = None, None
        for rep in DECISIVE:
            a = _main_acc(res, R, rep, TRAIN_SIZES[-1], pop)
            if a is not None and (best_top is None or a > best_top):
                best_top, best_rep = a, rep
        ho_top = _ho_acc(res, R, best_rep, TRAIN_SIZES[-1], pop) if best_rep else None
        gain = None
        gains = {}
        for rep in DECISIVE:
            a0 = _main_acc(res, R, rep, TRAIN_SIZES[0], pop)
            a1 = _main_acc(res, R, rep, TRAIN_SIZES[-1], pop)
            gains[rep] = (a1 - a0) if (a0 is not None and a1 is not None) else None
        gain = max((g for g in gains.values() if g is not None), default=None)
        bayes = u_base[model][best_rep] if best_rep else None
        per_reader[R] = {"best_top": best_top, "best_rep": best_rep, "heldout_top": ho_top,
                         "gain_full": gain, "gains_by_rep": gains, "bayes": bayes,
                         "bayes_minus_reader": (bayes - best_top) if (bayes is not None and best_top is not None) else None}
    # W-A
    wA = any(pr["best_top"] is not None and pr["heldout_top"] is not None and pr["gain_full"] is not None
             and pr["best_top"] >= RECOVER_BAR and pr["heldout_top"] >= HELDOUT_MATERIAL
             and pr["gain_full"] >= MATERIAL_GAIN for pr in per_reader.values())
    best_top_all = max((pr["best_top"] for pr in per_reader.values() if pr["best_top"] is not None), default=None)
    best_ho = max((pr["heldout_top"] for pr in per_reader.values() if pr["heldout_top"] is not None), default=None)
    best_gap = min((pr["bayes_minus_reader"] for pr in per_reader.values()
                    if pr["bayes_minus_reader"] is not None), default=None)
    max_gain = max((pr["gain_full"] for pr in per_reader.values() if pr["gain_full"] is not None), default=None)
    wB = bool(best_top_all is not None and best_top_all <= HELDOUT_WEAK and best_ho is not None
              and best_ho <= HELDOUT_WEAK and best_gap is not None and best_gap >= BAYES_GAP_BAR
              and (max_gain is None or max_gain < MATERIAL_GAIN))
    wC = bool((not wA) and (not wB) and (max_gain is not None and max_gain >= MATERIAL_GAIN))
    # contradictory reader conclusions: one reader recovers, another is fully weak
    recovered = [R for R, pr in per_reader.items() if pr["best_top"] is not None and pr["best_top"] >= RECOVER_BAR]
    weak = [R for R, pr in per_reader.items() if pr["best_top"] is not None and pr["best_top"] <= HELDOUT_WEAK]
    reader_contradiction = bool(recovered and weak)
    if wA:
        verdict = "W-A_data_limited"
    elif reader_contradiction:
        verdict = "W-MIX"                            # a reader recovers while another is fully weak
    elif wB:
        verdict = "W-B_data_saturated_bounded"
    elif wC:
        verdict = "W-C_data_benefit_no_recovery"
    elif best_top_all is not None and best_top_all > HELDOUT_WEAK:
        verdict = "W-C_data_benefit_no_recovery"      # intermediate, sub-material benefit
    else:
        verdict = "W-B_data_saturated_bounded"
    return {"verdict": verdict, "per_reader": per_reader, "best_top_all": best_top_all,
            "best_heldout_all": best_ho, "best_bayes_gap": best_gap, "max_gain": max_gain,
            "conditions": {"W-A": wA, "W-B": wB, "W-C": wC, "reader_contradiction": reader_contradiction},
            "thresholds": {"recover_bar": RECOVER_BAR, "heldout_weak": HELDOUT_WEAK,
                           "heldout_material": HELDOUT_MATERIAL, "material_gain": MATERIAL_GAIN,
                           "bayes_gap_bar": BAYES_GAP_BAR},
            "rule_note": ("W-A if some reader at L4/full has top-scale test>=0.50 AND held-out>=0.30 "
                          "AND gain_full>=0.15; W-B if best top-scale test<=0.25 AND held-out<=0.25 AND "
                          "Bayes-reader>=0.30 AND every gain_full<0.15; W-C if a material gain (>=0.15) "
                          "or an intermediate best top-scale test (>0.25) without W-A; W-MIX on E/F or "
                          "reader contradiction.")}


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
            node = merged.setdefault(key, {"seed": part["seed"], "reader": part["reader"], "fits": {}})
            node["fits"].update(part["fits"])
            done += 1
            if done % 10 == 0 or done == len(specs):
                print(f"  [{done}/{len(specs)}] tasks done", flush=True)
    print(f"collection wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    for node in merged.values():
        node["fits"] = dict(sorted(node["fits"].items()))   # deterministic key order (imap_unordered)
    return [merged[(s, R)] for s in SEEDS for R in readers]


def assemble(data: Dict[str, Any], u_base: Dict[str, Any], per: List[Dict[str, Any]]) -> Dict[str, Any]:
    arch = av.architecture()
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS, "readers": READERS,
        "train_sizes": TRAIN_SIZES, "reps": REP_LABELS,
        "main_scope": {t: MAIN_SCOPE[t] for t in TARGETS}, "heldout_scope": HELDOUT_SCOPE,
        "populations": POPULATIONS, "heldout_pairs": sorted(HELD_OUT_PAIRS),
        "reader_epochs": av.READER_EPOCHS, "reader_lr": av.READER_LR, "reader_wd": av.READER_WD,
        "gru_device": GRU_DEVICE,
        "train_ord_cap": rw.rr.TRAIN_ORD_CAP, "val_ord_cap": rw.rr.VAL_ORD_CAP,
        "pool_stream_base": rw.POOL_STREAM_BASE, "pool_offsets": rw.POOL_STREAM_OFFSETS,
        "checkpoints": data["meta"]["models"], "disjointness": data["meta"]["disjointness"],
        "n": data["meta"]["n"],
        "thresholds": {"recover_bar": RECOVER_BAR, "heldout_weak": HELDOUT_WEAK,
                       "heldout_material": HELDOUT_MATERIAL, "material_gain": MATERIAL_GAIN,
                       "bayes_gap_bar": BAYES_GAP_BAR},
        "statistical_discipline": ("Five frozen E checkpoints are the independent units; F is a "
                                   "control. Readers are analysis-only, fit on the fresh training "
                                   "pool, tuned on the fixed canonical VAL, evaluated once on the "
                                   "untouched canonical TEST. The ONLY experimental factor is "
                                   "N_train; optimizer updates are 200 at every scale. No "
                                   "TEST-based selection.")}}
    res["architecture"] = arch
    res["readers"] = aggregate(per, arch)
    res["per_seed"] = per
    res["u_bayes"] = u_base
    dec = classify(PRIMARY_MODEL, res, u_base)
    decF = classify(CONTROL_MODEL, res, u_base)
    dec["F_verdict"] = decF["verdict"]
    dec["F_metrics"] = decF["per_reader"]
    # E/F material disagreement on the primary reader-classification
    ef_disagree = (dec["verdict"] != decF["verdict"])
    if ef_disagree:
        dec["verdict"] = "W-MIX"
    dec["E_F_disagreement"] = bool(ef_disagree)
    res["primary_decision"] = dec
    return res


def build(data: Dict[str, Any], u_base: Dict[str, Any], jobs: int,
          readers: Optional[List[str]] = None) -> Dict[str, Any]:
    return assemble(data, u_base, _collect(data, jobs, readers or READERS))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_w") -> List[str]:
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

    def nz(v):
        return np.nan if v is None else v

    R = res["readers"]
    U = res["u_bayes"]
    sizes = res["metadata"]["train_sizes"]        # ladder actually collected/analyzed
    xs = np.arange(len(sizes))

    def ma(reader, rep, size, pop):
        return nz(R[reader]["y_true"][rep][str(size)]["main"]["acc"][pop]["mean"])

    def ho(reader, rep, size, pop):
        return nz(R[reader]["y_true"][rep][str(size)]["heldout"]["heldout"][pop]["mean"])

    # 1 accuracy vs data scale (Class-A, L4 & full)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader, c in (("MLP128", "tab:green"), ("GRU", "tab:blue"), ("TCN", "tab:orange")):
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            ax.plot(xs, [ma(reader, rep, s, f"{PRIMARY_MODEL}:classA") for s in sizes], marker="o",
                    ls=ls, color=c, alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep}")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}" for s in sizes]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("N_train"); ax.set_ylabel("Class-A TEST accuracy (y_true)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("W: Class-A accuracy vs training-data scale")
    save(fig, "accuracy_vs_data_scale")

    # 2 held-out vs data scale (L4/full, Class-A)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader, c in (("MLP128", "tab:green"), ("GRU", "tab:blue"), ("TCN", "tab:orange")):
        for rep, ls in (("L4", "-"), ("Lfull", "--")):
            ax.plot(xs, [ho(reader, rep, s, f"{PRIMARY_MODEL}:classA") for s in sizes], marker="s",
                    ls=ls, color=c, alpha=1.0 if rep == "L4" else 0.6, label=f"{reader} {rep}")
    ax.axhline(HELDOUT_MATERIAL, ls=":", c="k", lw=0.8, label="0.30 material")
    ax.axhline(HELDOUT_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}" for s in sizes]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("N_train"); ax.set_ylabel("held-out operand accuracy (Class-A, y_true)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("W: held-out operand accuracy vs training-data scale")
    save(fig, "heldout_vs_data_scale")

    # 3 Bayes gap vs data scale
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader, c in (("MLP128", "tab:green"), ("GRU", "tab:blue"), ("TCN", "tab:orange")):
        gaps = [U[PRIMARY_MODEL]["L4"] - ma(reader, "L4", s, f"{PRIMARY_MODEL}:classA") for s in sizes]
        ax.plot(xs, gaps, marker="o", color=c, label=f"{reader} L4")
    ax.axhline(BAYES_GAP_BAR, ls="--", c="k", lw=0.8, label="0.30 bar")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}" for s in sizes]); ax.set_ylim(0, 1.05)
    ax.set_xlabel("N_train"); ax.set_ylabel("Bayes(L4) - reader (Class-A)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("W: Bayes-vs-reader gap vs training-data scale")
    save(fig, "bayes_gap_vs_data_scale")

    # 4 Class-A vs all-ADD
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader, c in (("MLP128", "tab:green"), ("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(xs, [ma(reader, "L4", s, f"{PRIMARY_MODEL}:classA") for s in sizes], marker="o", color=c,
                label=f"{reader} Class-A")
        ax.plot(xs, [ma(reader, "L4", s, f"{PRIMARY_MODEL}:all_add") for s in sizes], marker="^", ls="--",
                color=c, alpha=0.6, label=f"{reader} all-ADD")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}" for s in sizes]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("N_train"); ax.set_ylabel("TEST accuracy (L4, y_true)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("W: Class-A vs all-ADD vs data scale (L4)")
    save(fig, "classA_vs_alladd")

    # 5 reader comparison at the top scale (Class-A, L4/full)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    groups = [f"{r}\n{rep}" for r in READERS for rep in DECISIVE]
    vals = [ma(r, rep, sizes[-1], f"{PRIMARY_MODEL}:classA") for r in READERS for rep in DECISIVE]
    ax.bar(np.arange(len(groups)), vals, color=["tab:green", "tab:green", "tab:blue", "tab:blue",
                                                "tab:orange", "tab:orange"])
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.axhline(HELDOUT_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel(f"Class-A accuracy (N={sizes[-1]})"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"W: readers at {sizes[-1]}")
    save(fig, "reader_comparison")

    # 6 L4 comparison
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader, c in (("MLP128", "tab:green"), ("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(xs, [ma(reader, "L4", s, f"{PRIMARY_MODEL}:classA") for s in sizes], marker="o", color=c,
                label=reader)
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in sizes]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("N_train"); ax.set_ylabel("Class-A L4 accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("W: L4 Class-A accuracy vs data scale")
    save(fig, "l4_comparison")

    # 7 full-prefix comparison
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader, c in (("MLP128", "tab:green"), ("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(xs, [ma(reader, "Lfull", s, f"{PRIMARY_MODEL}:classA") for s in sizes], marker="o", color=c,
                label=reader)
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in sizes]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("N_train"); ax.set_ylabel("Class-A full-prefix accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("W: full-prefix Class-A accuracy vs data scale")
    save(fig, "fullprefix_comparison")

    # 8 seen vs held-out at the top scale
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    labs, seen, hold = [], [], []
    for reader in READERS:
        for rep in DECISIVE:
            labs.append(f"{reader}\n{rep}")
            seen.append(nz(R[reader]["y_true"][rep][str(sizes[-1])]["heldout"]["seen"][f"{PRIMARY_MODEL}:classA"]["mean"]))
            hold.append(ho(reader, rep, sizes[-1], f"{PRIMARY_MODEL}:classA"))
    xi = np.arange(len(labs))
    ax.bar(xi - 0.2, seen, 0.4, label="seen", color="tab:gray")
    ax.bar(xi + 0.2, hold, 0.4, label="held-out", color="tab:purple")
    ax.set_xticks(xi); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel(f"accuracy (N={sizes[-1]})"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"W: seen vs held-out operand pairs (Class-A, {sizes[-1]})")
    save(fig, "seen_vs_heldout")

    # 9 train vs test (Class-A L4)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reader, c in (("MLP128", "tab:green"), ("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(xs, [nz(R[reader]["y_true"]["L4"][str(s)]["main"]["train_acc"]["mean"]) for s in sizes],
                marker="o", color=c, label=f"{reader} train")
        ax.plot(xs, [ma(reader, "L4", s, f"{PRIMARY_MODEL}:classA") for s in sizes], marker="s", ls="--",
                color=c, alpha=0.6, label=f"{reader} test (Class-A)")
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in sizes]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("N_train"); ax.set_ylabel("accuracy"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("W: train vs test accuracy vs data scale (L4)")
    save(fig, "train_vs_test")

    # 10 per-seed (Class-A L4, top scale)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    xi = np.arange(len(SEEDS))
    for reader, c in (("MLP128", "tab:green"), ("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(xi, [s["fits"][f"{reader}|y_true|L4|{sizes[-1]}|main"]["acc"][f"{PRIMARY_MODEL}:classA"]
                     for s in res["per_seed"] if s["reader"] == reader], marker="o", color=c, label=reader)
    ax.set_xticks(xi); ax.set_xticklabels([str(s) for s in SEEDS]); ax.set_xlabel("E seed")
    ax.set_ylabel(f"Class-A L4 accuracy ({sizes[-1]})"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"W: per-seed accuracy ({sizes[-1]})")
    save(fig, "per_seed")

    # 11 parameter invariance
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for reader, c in (("MLP128", "tab:green"), ("GRU", "tab:blue"), ("TCN", "tab:orange")):
        vals = [R[reader]["y_true"]["L4"][str(s)]["main"]["params"] for s in sizes]
        ax.plot(xs, vals, marker="o", color=c, label=f"{reader} ({vals[0]})")
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("N_train"); ax.set_ylabel("trainable parameters"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("W: reader parameter count is invariant across scales")
    save(fig, "parameter_invariance")

    # 12 decision summary
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    d = res["primary_decision"]
    cats = [f"best {sizes[-1]}\n(Class-A)", f"best {sizes[-1]}\nheld-out", "Bayes(L4)\n-reader",
            f"max gain\n({sizes[-1]}-{sizes[0]})"]
    vals = [nz(d["best_top_all"]), nz(d["best_heldout_all"]), nz(d["best_bayes_gap"]), nz(d["max_gain"])]
    ax.bar(np.arange(4), vals, color=["tab:blue", "tab:purple", "tab:red", "tab:orange"])
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.axhline(MATERIAL_GAIN, ls=":", c="k", lw=0.8, label="0.15 material gain")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("value"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"W: decision summary ({d['verdict']})")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment W analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_w_data_scale.npz")
    ap.add_argument("--u-results", default="experiments/experiment_u_results.json")
    ap.add_argument("--output", default="experiments/experiment_w_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_w")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--readers", nargs="*", default=None,
                    help="subset of readers to fit (MLP128 GRU TCN); default all")
    ap.add_argument("--collect-only", action="store_true",
                    help="fit the selected readers and write a partial {'per_seed': [...]} JSON")
    args = ap.parse_args()
    data = load(args.npz)
    data["path"] = args.npz
    u_base = load_u_baseline(args.u_results)
    t0 = time.perf_counter()
    if args.collect_only:
        per = _collect(data, args.jobs, args.readers or READERS)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"per_seed": per}, f, indent=2)
        print(f"Wrote partial {args.output} (readers={args.readers or READERS}) "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)
        return
    res = build(data, u_base, args.jobs, readers=args.readers)
    print(f"analysis wall_seconds={time.perf_counter() - t0:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"E: {d['verdict']}  best_top={d['best_top_all']} heldout={d['best_heldout_all']} "
          f"gap={d['best_bayes_gap']} maxgain={d['max_gain']}")
    for R in READERS:
        pr = d["per_reader"][R]
        print(f"  {R}: best_top={pr['best_top']} ({pr['best_rep']}) ho={pr['heldout_top']} "
              f"gain={pr['gain_full']} gains={pr['gains_by_rep']}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
