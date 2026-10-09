#!/usr/bin/env python3
"""Experiment V — analysis: bounded sequence-aware access to the raw task-interface information.

U (U-D) showed the raw causal observation history uniquely determines the ADD operands/target
(L=4 target Bayes ~0.998; full-prefix 1.000) while the bounded FLAT probe family stays weak (best
full-prefix 0.218 all-ADD, ~0.09 Class-A; held-out <= 0.25). V tests one question: does a
parameter-bounded causal SEQUENCE-AWARE reader recover that exactly-identifiable information?

Reader family (exactly TWO, predeclared):
  A. single-layer causal GRU (unidirectional)          -> last-step hidden state -> linear head
  B. causal 1-D convolutional reader (TCN-style)       -> causal dilated convs -> last-position head
Per-timestep input = raw 21-dim observation x_t + 1 validity bit = 22 features.
The reader at step t consumes only [x_0..x_t]; no future observation.

Parameter budget: every reader must satisfy trainable_params <= the exact U MLP-128 parameter count
at the full-prefix representation (input dim 22*10 = 220). Widths are chosen deterministically as the
LARGEST integer within budget, BEFORE any aggregate TEST result is inspected.

PRE-REGISTRATION (declared before aggregate inspection)
-------------------------------------------------------
* Reader families: {GRU, TCN}; history grid = U's {L1,L2,L4,L8,L16,Lfull}; targets = y_true
  (primary), operand_a, operand_b, raw_sum, op_dest. Variants: main (all ordinary-train rows) and
  held-out-excluded (20 pairs {(a,b): b in {0,5}} removed).
* Training (identical to U/R/S/T): Adam 3e-3, weight-decay 1e-2, 200 epochs, class-balanced
  cross-entropy, best-on-VAL checkpoint, fixed deterministic seed. No tuning of LR/WD/epochs/width/
  kernel/dilation/layers.
* Splits: ordinary TRAIN ADD (cap 1500/seed) -> fit, ordinary VAL (cap 800/seed) -> tune, canonical
  TEST -> evaluate once. Populations: Class-A hard (primary), matched candidate-correct, all
  candidate-correct, all ADD. E seeds 42..46 are the independent units; F is a control.
* Decision (bars are declared conventions):
    V-A sequence-accessible   : some reader test >= 0.50 AND held-out >= 0.30 at L4 or full.
    V-B sequence-limited      : best reader test <= 0.25 AND held-out <= 0.25 AND Bayes-reader >= 0.30.
    V-C partial-sequence-benefit: improves over U MLP-128 by >= 0.15 but not V-A.
    V-MIX                     : E/F material disagreement, or a genuine GRU-vs-TCN disagreement.
No test-based history/width/architecture selection.

Reproduce:
    PYTHONPATH=src python experiments/analyze_experiment_v.py \
        --npz experiments/experiment_v_sequence_access.npz \
        --u-results experiments/experiment_u_results.json \
        --output experiments/experiment_v_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_v
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_experiment_s as as_  # noqa: E402  (probe primitives: param_count, class_weights, mean_sd, paired)
import analyze_experiment_u as au  # noqa: E402  (history construction / exact Bayes / populations)
import run_experiment_u as ru  # noqa: E402

torch.set_num_threads(int(os.environ.get("V_TORCH_THREADS", "8")))

PRIMARY_MODEL = "E_candidate"
CONTROL_MODEL = "F_combined"
SEEDS = [42, 43, 44, 45, 46]
FAMILIES = ["GRU", "TCN"]
REPS = au.REPS                       # [("L1",1),("L2",2),("L4",4),("L8",8),("L16",16),("Lfull",10)]
REP_LABELS = au.REP_LABELS
DECISIVE = ["L4", "Lfull"]

INPUT_DIM = ru.OBS_DIM + 1           # 21 raw observation + 1 validity bit = 22
N_MOD = 10
N_RAW = 19
TARGETS = {"y_true": N_MOD, "operand_a": N_MOD, "operand_b": N_MOD, "raw_sum": N_RAW, "op_dest": 4}
KERNEL = 3
DILATIONS = (1, 2, 4, 8)
N_LAYERS = len(DILATIONS)
# reader-training protocol (identical to U/R/S/T probe discipline)
READER_LR = 3e-3
READER_WD = 1e-2
READER_EPOCHS = 200
SEED_BASE = as_.MLP_SEED
POPULATIONS = ["classA", "matched_correct", "correct_all", "all_add"]

# History scope per target (predeclared; raw_sum/op_dest are secondary and run at the decisive
# horizons only; held-out control runs for y_true (all L) and the operands at the decisive horizons).
ALL_L = REP_LABELS
MAIN_L = {"y_true": ALL_L, "operand_a": ALL_L, "operand_b": ALL_L, "raw_sum": DECISIVE, "op_dest": DECISIVE}
HELDOUT_L = {"y_true": ALL_L, "operand_a": DECISIVE, "operand_b": DECISIVE}
HELD_OUT_PAIRS = au.HELD_OUT_PAIRS

# decision bars (declared conventions)
RECOVER_BAR = 0.50
HELDOUT_WEAK = 0.25
HELDOUT_MATERIAL = 0.30
MATERIAL_GAIN = 0.15
BAYES_GAP_BAR = 0.30


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


# ---------------------------------------------------------------------------
# parameter budget + deterministic architecture selection
# ---------------------------------------------------------------------------

def u_mlp128_param_count() -> int:
    """Exact U MLP-128 parameter count at the full-prefix representation (input dim 22*10 = 220)."""
    return as_.param_count("mlp", 128, INPUT_DIM * ru.FULL_PREFIX_LENGTH)


MAX_NCLASS = max(TARGETS.values())    # 19 (raw_sum) -> widths chosen so EVERY target's reader fits


def gru_param_count(hidden: int, indim: int = INPUT_DIM, nclass: int = N_MOD) -> int:
    """nn.GRU(input, hidden, num_layers=1) + Linear(hidden, nclass)."""
    return 3 * hidden * indim + 3 * hidden * hidden + 6 * hidden + hidden * nclass + nclass


def tcn_param_count(channels: int, indim: int = INPUT_DIM, nclass: int = N_MOD,
                    kernel: int = KERNEL, nlayers: int = N_LAYERS) -> int:
    """N causal Conv1d layers (first: indim->channels, rest: channels->channels) + Linear head."""
    p = indim * channels * kernel + channels
    p += (nlayers - 1) * (channels * channels * kernel + channels)
    p += channels * nclass + nclass
    return p


def _largest_within(param_fn, budget: int, cap: int = 100000) -> int:
    """Deterministic largest integer width whose parameter count is <= budget (never exceeds it)."""
    v = 1
    while v + 1 <= cap and param_fn(v + 1) <= budget:
        v += 1
    if param_fn(v) > budget:
        raise ValueError("no width within budget")
    return v


def architecture(budget: Optional[int] = None) -> Dict[str, Any]:
    """Deterministic reader widths within the U MLP-128 budget.

    Widths are chosen against the LARGEST target class count (raw_sum, 19) so that EVERY reader
    variant (10-class primary and 19-class raw-sum) satisfies the budget with one shared width.
    """
    if budget is None:
        budget = u_mlp128_param_count()
    nclass = MAX_NCLASS
    hidden = _largest_within(lambda h: gru_param_count(h, nclass=nclass), budget)
    channels = _largest_within(lambda c: tcn_param_count(c, nclass=nclass), budget)
    return {
        "budget_u_mlp128": budget,
        "max_nclass": nclass,
        "GRU": {"hidden": hidden, "params": gru_param_count(hidden, nclass=nclass),
                "params_primary": gru_param_count(hidden, nclass=N_MOD),
                "ratio": gru_param_count(hidden, nclass=nclass) / budget},
        "TCN": {"channels": channels, "params": tcn_param_count(channels, nclass=nclass),
                "params_primary": tcn_param_count(channels, nclass=N_MOD),
                "ratio": tcn_param_count(channels, nclass=nclass) / budget,
                "kernel": KERNEL, "dilations": list(DILATIONS), "n_layers": N_LAYERS},
    }


# ---------------------------------------------------------------------------
# causal readers
# ---------------------------------------------------------------------------

class GRUReader(nn.Module):
    """Single-layer unidirectional (causal) GRU; prediction from the last-step hidden state."""

    def __init__(self, hidden: int, nclass: int = N_MOD, indim: int = INPUT_DIM) -> None:
        super().__init__()
        self.gru = nn.GRU(indim, hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden, nclass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, L, indim)
        out, _ = self.gru(x)
        return self.head(out[:, -1])                         # last step = current time t


class TCNReader(nn.Module):
    """Causal dilated 1-D convolutional reader (TCN-style); prediction from the last position."""

    def __init__(self, channels: int, nclass: int = N_MOD, indim: int = INPUT_DIM,
                 kernel: int = KERNEL, dilations=DILATIONS) -> None:
        super().__init__()
        self.kernel = kernel
        self.dilations = list(dilations)
        self.convs = nn.ModuleList()
        in_ch = indim
        for d in self.dilations:
            self.convs.append(nn.Conv1d(in_ch, channels, kernel, padding=d * (kernel - 1), dilation=d))
            in_ch = channels
        self.act = nn.ReLU()
        self.head = nn.Linear(channels, nclass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, L, indim)
        z = x.transpose(1, 2)                                # (B, indim, L)
        for conv in self.convs:
            z = conv(z)
            d = conv.dilation[0]
            k = conv.kernel_size[0]
            trim = d * (k - 1)
            if trim > 0:
                z = z[:, :, :-trim]                          # drop the future-padded right side
            z = self.act(z)
        return self.head(z[:, :, -1])                        # last position = current time t


def build_reader(family: str, nclass: int, arch: Optional[Dict[str, Any]] = None) -> nn.Module:
    arch = arch or architecture()
    if family == "GRU":
        assert gru_param_count(arch["GRU"]["hidden"], nclass=nclass) <= arch["budget_u_mlp128"]
        return GRUReader(arch["GRU"]["hidden"], nclass=nclass)
    if family == "TCN":
        assert tcn_param_count(arch["TCN"]["channels"], nclass=nclass) <= arch["budget_u_mlp128"]
        return TCNReader(arch["TCN"]["channels"], nclass=nclass)
    raise ValueError(family)


def reader_param_count(family: str, nclass: int = N_MOD, arch: Optional[Dict[str, Any]] = None) -> int:
    return sum(p.numel() for p in build_reader(family, nclass, arch).parameters() if p.requires_grad)


def reader_seed(seed: int, family: str, target: str, rep: str, variant: str) -> int:
    fi = FAMILIES.index(family)
    ti = list(TARGETS).index(target)
    ri = REP_LABELS.index(rep)
    vi = 0 if variant == "main" else 1
    return SEED_BASE + seed * 100000 + fi * 10000 + ti * 1000 + ri * 10 + vi


# ---------------------------------------------------------------------------
# reader input (raw observations + validity bit); causal, no future leakage
# ---------------------------------------------------------------------------

def reader_input(seq_obs: np.ndarray, t: np.ndarray, L: int) -> np.ndarray:
    """(N, L, 22) per-timestep [raw x (21) | validity mask bit] via U's exact causal window."""
    X, M = au.build_history(seq_obs, t, L)                  # X (N, L*21), M (N, L)
    N = X.shape[0]
    xr = X.reshape(N, L, au.D_OBS)
    return np.concatenate([xr, M[:, :, None]], axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# training / prediction
# ---------------------------------------------------------------------------

def train_reader(family: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
                 nclass: int, seed: int, arch: Optional[Dict[str, Any]] = None):
    torch.manual_seed(seed)
    model = build_reader(family, nclass, arch)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.long)
    wc = torch.tensor(as_.class_weights(ytr, nclass), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=READER_LR, weight_decay=READER_WD)
    best = (-1.0, None)
    for ep in range(READER_EPOCHS):
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


@torch.no_grad()
def reader_predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


# ---------------------------------------------------------------------------
# per-seed reader runs (readers are model-independent; population masks are per model)
# ---------------------------------------------------------------------------

def _pairs(z, ds, idx):
    return au._pairs(z, PRIMARY_MODEL, ds, idx)


def run_readers(z, seed: int, arch: Dict[str, Any]) -> Dict[str, Any]:
    tr = np.nonzero(z[f"{PRIMARY_MODEL}__trainord__seed"] == seed)[0]
    va = np.nonzero(z[f"{PRIMARY_MODEL}__valord__seed"] == seed)[0]
    te = z[f"{PRIMARY_MODEL}__test__seed"] == seed

    Xtr = {rep: reader_input(z[f"{PRIMARY_MODEL}__trainord__seq_obs"][tr],
                             z[f"{PRIMARY_MODEL}__trainord__t"][tr], L) for rep, L in REPS}
    Xva = {rep: reader_input(z[f"{PRIMARY_MODEL}__valord__seq_obs"][va],
                             z[f"{PRIMARY_MODEL}__valord__t"][va], L) for rep, L in REPS}
    Xte = {rep: reader_input(z[f"{PRIMARY_MODEL}__test__seq_obs"][te],
                             z[f"{PRIMARY_MODEL}__test__t"][te], L) for rep, L in REPS}

    held_tr = _pairs(z, "trainord", tr)
    held_te = _pairs(z, "test", np.nonzero(te)[0])
    yte = {t: z[f"{PRIMARY_MODEL}__test__{t}"][te].astype(int) for t in TARGETS}

    masks = {m: {p: au._pop_masks(z, m, seed)[p][te] for p in POPULATIONS}
             for m in (PRIMARY_MODEL, CONTROL_MODEL)}

    def acc_on(pred, m, p, tgt):
        mask = masks[m][p]
        if not mask.any():
            return None
        return float((pred[mask] == yte[tgt][mask]).mean())

    out: Dict[str, Any] = {"seed": seed, "fits": {}}
    for fam in FAMILIES:
        for tgt, nclass in TARGETS.items():
            for rep in MAIN_L[tgt]:
                ytr_t = z[f"{PRIMARY_MODEL}__trainord__{tgt}"][tr].astype(int)
                yva_t = z[f"{PRIMARY_MODEL}__valord__{tgt}"][va].astype(int)
                model, vacc = train_reader(fam, Xtr[rep], ytr_t, Xva[rep], yva_t, nclass,
                                           reader_seed(seed, fam, tgt, rep, "main"), arch)
                pred = reader_predict(model, Xte[rep])       # aligned to the seed's test rows
                tracc = float((reader_predict(model, Xtr[rep]) == ytr_t).mean())
                acc = {f"{m}:{p}": acc_on(pred, m, p, tgt)
                       for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
                out["fits"][f"{fam}|{tgt}|{rep}|main"] = {
                    "family": fam, "target": tgt, "rep": rep, "variant": "main",
                    "params": reader_param_count(fam, nclass, arch), "train_acc": tracc, "val_acc": vacc,
                    "acc": acc}
        for tgt in HELDOUT_L:
            nclass = TARGETS[tgt]
            for rep in HELDOUT_L[tgt]:
                tr_m = tr[~held_tr]
                Xtr_m = reader_input(z[f"{PRIMARY_MODEL}__trainord__seq_obs"][tr_m],
                                     z[f"{PRIMARY_MODEL}__trainord__t"][tr_m], dict(REPS)[rep])
                ytr_t = z[f"{PRIMARY_MODEL}__trainord__{tgt}"][tr_m].astype(int)
                yva_t = z[f"{PRIMARY_MODEL}__valord__{tgt}"][va].astype(int)
                model, vacc = train_reader(fam, Xtr_m, ytr_t, Xva[rep], yva_t, nclass,
                                           reader_seed(seed, fam, tgt, rep, "heldout"), arch)
                pred = reader_predict(model, Xte[rep])
                seen = {f"{m}:{p}": None for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
                hold = {f"{m}:{p}": None for m in (PRIMARY_MODEL, CONTROL_MODEL) for p in POPULATIONS}
                for m in (PRIMARY_MODEL, CONTROL_MODEL):
                    for p in POPULATIONS:
                        base = masks[m][p]
                        seen[m + ":" + p] = float((pred[base & ~held_te] == yte[tgt][base & ~held_te]).mean()) if (base & ~held_te).any() else None
                        hold[m + ":" + p] = float((pred[base & held_te] == yte[tgt][base & held_te]).mean()) if (base & held_te).any() else None
                out["fits"][f"{fam}|{tgt}|{rep}|heldout"] = {
                    "family": fam, "target": tgt, "rep": rep, "variant": "heldout",
                    "params": reader_param_count(fam, nclass, arch), "val_acc": vacc, "n_train": int(len(tr_m)),
                    "seen": seen, "heldout": hold}
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _ms(vals):
    return as_.mean_sd(vals)


def aggregate_readers(per: List[Dict[str, Any]], arch: Dict[str, Any]) -> Dict[str, Any]:
    keys = sorted(per[0]["fits"].keys())
    out: Dict[str, Any] = {}
    for k in keys:
        fam, tgt, rep, variant = k.split("|")
        node = out.setdefault(fam, {}).setdefault(tgt, {}).setdefault(rep, {})
        recs = [s["fits"][k] for s in per]
        if variant == "main":
            node["main"] = {
                "params": recs[0]["params"],
                "train_acc": _ms([r["train_acc"] for r in recs]),
                "val_acc": _ms([r["val_acc"] for r in recs]),
                "acc": {pk: _ms([r["acc"][pk] for r in recs]) for pk in recs[0]["acc"]}}
        else:
            node["heldout"] = {
                "params": recs[0]["params"], "val_acc": _ms([r["val_acc"] for r in recs]),
                "n_train": recs[0]["n_train"],
                "seen": {pk: _ms([r["seen"][pk] for r in recs]) for pk in recs[0]["seen"]},
                "heldout": {pk: _ms([r["heldout"][pk] for r in recs]) for pk in recs[0]["heldout"]}}
    return out


def load_u_baseline(path: str) -> Dict[str, Any]:
    with open(path) as f:
        u = json.load(f)
    base: Dict[str, Any] = {"models": {}}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        base["models"][m] = {
            "probes": {p: {rep: u[m]["probes"][p][rep]["mlp128"]["acc"]["mean"] for rep, _ in REPS}
                       for p in POPULATIONS},
            "heldout": {p: {rep: u[m]["heldout"][p][rep]["mlp32"]["heldout"]["mean"] for rep, _ in REPS}
                        for p in ("classA", "all_add")},
            "bayes": {p: {rep: u[m]["pooled"][p][rep]["y_true"]["bayes_acc"] for rep, _ in REPS}
                      for p in ("classA", "correct_all", "all_add")}}
    return base


def _reader_val(res, fam, tgt, rep, variant, field, pop):
    node = res["readers"][fam][tgt][rep].get(variant)
    if node is None:
        return None
    return node[field][pop]["mean"]


def _bayes(u_base, m, pop, rep):
    return u_base["models"][m]["bayes"][pop][rep]


def _cat(m: str, res, u_base) -> Dict[str, Any]:
    """Classification metrics for one model key (E or F) from the aggregated reader results."""
    # best reader at decisive horizons (y_true, main) on the primary population (Class-A)
    entries = [(fam, rep) for fam in FAMILIES for rep in DECISIVE
               if _reader_val(res, fam, "y_true", rep, "main", "acc", f"{m}:classA") is not None]
    sc = [(_reader_val(res, fam, "y_true", rep, "main", "acc", f"{m}:classA"), fam, rep) for fam, rep in entries]
    sc.sort(key=lambda x: x[0])
    best, bfam, brep = sc[-1]
    bho = _reader_val(res, bfam, "y_true", brep, "heldout", "heldout", f"{m}:classA")
    if bho is None:
        bho = _reader_val(res, bfam, "y_true", brep, "heldout", "heldout", f"{m}:all_add")
    bayes = _bayes(u_base, m, "classA", brep)
    gap = (bayes - best) if (bayes is not None and best is not None) else None
    u_base_best = max(u_base["models"][m]["probes"]["classA"][rep] for rep in DECISIVE
                      if u_base["models"][m]["probes"]["classA"][rep] is not None)
    gain = (best - u_base_best) if best is not None else None
    sA = bool(best is not None and bho is not None and best >= RECOVER_BAR and bho >= HELDOUT_MATERIAL)
    sB = bool(best is not None and bho is not None and best <= HELDOUT_WEAK and bho <= HELDOUT_WEAK
              and gap is not None and gap >= BAYES_GAP_BAR)
    sC = bool((not sA) and (not sB) and gain is not None and gain >= MATERIAL_GAIN)
    # GRU vs TCN preregistered disagreement: one family reaches V-A while the other is weak
    fam_best = {f: max((_reader_val(res, f, "y_true", rep, "main", "acc", f"{m}:classA") for rep in DECISIVE
                        if _reader_val(res, f, "y_true", rep, "main", "acc", f"{m}:classA") is not None),
                       default=None) for f in FAMILIES}
    fam_ho = {f: max((_reader_val(res, f, "y_true", rep, "heldout", "heldout", f"{m}:classA") for rep in DECISIVE
                      if _reader_val(res, f, "y_true", rep, "heldout", "heldout", f"{m}:classA") is not None),
                     default=None) for f in FAMILIES}
    disagree = False
    if all(fam_best[f] is not None and fam_ho[f] is not None for f in FAMILIES):
        a = [fam_best[f] >= RECOVER_BAR and fam_ho[f] >= HELDOUT_MATERIAL for f in FAMILIES]
        w = [fam_best[f] <= HELDOUT_WEAK and fam_ho[f] <= HELDOUT_WEAK for f in FAMILIES]
        disagree = (any(a) and any(w))
    if sA:
        cat = "V-A_sequence_accessible"
    elif sB:
        cat = "V-B_sequence_limited"
    elif sC:
        cat = "V-C_partial_sequence_benefit"
    elif disagree:
        cat = "V-MIX"
    else:
        cat = "V-C_partial_sequence_benefit"     # conservative residual: benefit present, not V-A
    return {"category": cat, "best": best, "best_family": bfam, "best_rep": brep, "best_heldout": bho,
            "bayes": bayes, "bayes_minus_reader": gap, "u_mlp128_best": u_base_best, "gain_over_u": gain,
            "conditions": {"V-A": sA, "V-B": sB, "V-C": sC, "family_disagreement": disagree},
            "per_family_best": fam_best, "per_family_heldout": fam_ho}


def decide(res: Dict[str, Any], u_base: Dict[str, Any]) -> Dict[str, Any]:
    mE = _cat(PRIMARY_MODEL, res, u_base)
    mF = _cat(CONTROL_MODEL, res, u_base)
    # E/F material disagreement -> V-MIX
    ef_disagree = False
    if mE["best"] is not None and mF["best"] is not None:
        ef_disagree = abs(mE["best"] - mF["best"]) >= MATERIAL_GAIN and mE["category"] != mF["category"]
    verdict = mE["category"]
    if ef_disagree or mE["conditions"]["family_disagreement"]:
        verdict = "V-MIX"
    return {"verdict": verdict, "E_metrics": mE, "F_metrics": mF,
            "E_F_disagreement": bool(ef_disagree),
            "thresholds": {"recover_bar": RECOVER_BAR, "heldout_weak": HELDOUT_WEAK,
                           "heldout_material": HELDOUT_MATERIAL, "material_gain": MATERIAL_GAIN,
                           "bayes_gap_bar": BAYES_GAP_BAR},
            "rule_note": ("V-A if some reader test>=0.50 AND held-out>=0.30 at L4/full; V-B if best "
                          "reader test<=0.25 AND held-out<=0.25 AND Bayes-reader>=0.30; V-C if the "
                          "best reader improves over the U MLP-128 baseline by >=0.15 without V-A; "
                          "V-MIX on material E/F disagreement or a genuine GRU-vs-TCN disagreement.")}


def build(data: Dict[str, Any], u_base: Dict[str, Any]) -> Dict[str, Any]:
    z = data["z"]
    meta = data["meta"]
    arch = architecture()
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS,
        "checkpoints": meta["models"], "splits": meta["n"],
        "obs_schema": meta.get("obs_schema"), "reps": REP_LABELS, "families": FAMILIES,
        "targets": list(TARGETS), "populations": POPULATIONS, "decisive": DECISIVE,
        "main_scope": {t: MAIN_L[t] for t in TARGETS}, "heldout_scope": HELDOUT_L,
        "input_dim": INPUT_DIM, "reader_lr": READER_LR, "reader_wd": READER_WD,
        "reader_epochs": READER_EPOCHS, "seed_base": SEED_BASE,
        "heldout_pairs": sorted(HELD_OUT_PAIRS),
        "thresholds": {"recover_bar": RECOVER_BAR, "heldout_weak": HELDOUT_WEAK,
                       "heldout_material": HELDOUT_MATERIAL, "material_gain": MATERIAL_GAIN,
                       "bayes_gap_bar": BAYES_GAP_BAR},
        "statistical_discipline": ("Five frozen E checkpoints are the independent units; F is a "
                                   "control. V readers are analysis-only, fit on ordinary TRAIN ADD, "
                                   "tuned on VAL, evaluated once on the untouched canonical TEST. "
                                   "Every predeclared family/history/target is reported; no "
                                   "test-based selection. Bayes ceilings are U's, unmodified.")}}
    res["architecture"] = arch
    per = [run_readers(z, s, arch) for s in SEEDS]
    res["readers"] = aggregate_readers(per, arch)
    res["per_seed"] = per
    res["u_baseline"] = u_base
    # verify U Bayes anchors are unchanged
    res["bayes_check"] = {
        "L4_all_add_y_true": u_base["models"][PRIMARY_MODEL]["bayes"]["all_add"]["L4"],
        "Lfull_all_add_y_true": u_base["models"][PRIMARY_MODEL]["bayes"]["all_add"]["Lfull"]}
    dec = decide(res, u_base)
    res["E_vs_F"] = {f"{fam}|{rep}": as_.paired(
        [s["fits"][f"{fam}|y_true|{rep}|main"]["acc"][f"{PRIMARY_MODEL}:classA"] for s in per],
        [s["fits"][f"{fam}|y_true|{rep}|main"]["acc"][f"{CONTROL_MODEL}:classA"] for s in per])
        for fam in FAMILIES for rep in DECISIVE}
    res["primary_decision"] = dec
    return res


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_v") -> List[str]:
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
    U = res["u_baseline"]["models"][PRIMARY_MODEL]
    x = np.arange(len(REP_LABELS))

    def rd(fam, tgt, rep, variant, field, pop):
        node = R.get(fam, {}).get(tgt, {}).get(rep, {}).get(variant)
        if node is None:
            return None
        return node[field][pop]["mean"]

    def curve(fam, tgt, variant, field, pop):
        return [nz(rd(fam, tgt, rep, variant, field, pop)) for rep in REP_LABELS]

    # 1 reader vs history (Class-A, y_true)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for fam, c in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(x, curve(fam, "y_true", "main", "acc", f"{PRIMARY_MODEL}:classA"), marker="o",
                label=f"{fam} (Class-A)", color=c)
        ax.plot(x, curve(fam, "y_true", "main", "acc", f"{PRIMARY_MODEL}:all_add"), marker="s",
                ls="--", label=f"{fam} (all-ADD)", color=c, alpha=0.6)
    ax.plot(x, [U["probes"]["classA"][rep] for rep in REP_LABELS], marker="x", ls=":", color="k",
            label="U MLP-128 (Class-A)")
    ax.plot(x, [U["bayes"]["classA"][rep] for rep in REP_LABELS], marker="^", ls="-", color="tab:green",
            label="Class-A Bayes")
    ax.axhline(RECOVER_BAR, ls="--", c="gray", lw=0.8, label="0.50 bar")
    ax.set_xticks(x); ax.set_xticklabels(REP_LABELS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("TEST accuracy")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("V: sequence reader vs history length (E)")
    save(fig, "reader_vs_history")

    # 2 reader comparison (L4 & full, Class-A)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    groups = ["GRU\nL4", "TCN\nL4", "GRU\nfull", "TCN\nfull", "U MLP128\nfull"]
    vals = [nz(rd("GRU", "y_true", "L4", "main", "acc", f"{PRIMARY_MODEL}:classA")),
            nz(rd("TCN", "y_true", "L4", "main", "acc", f"{PRIMARY_MODEL}:classA")),
            nz(rd("GRU", "y_true", "Lfull", "main", "acc", f"{PRIMARY_MODEL}:classA")),
            nz(rd("TCN", "y_true", "Lfull", "main", "acc", f"{PRIMARY_MODEL}:classA")),
            nz(U["probes"]["classA"]["Lfull"])]
    ax.bar(np.arange(5), vals, color=["tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:gray"])
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(groups)
    ax.set_ylabel("TEST accuracy (Class-A)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("V: reader comparison at decisive horizons")
    save(fig, "reader_comparison")

    # 3 Bayes vs reader gap
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(x, [U["bayes"]["classA"][rep] for rep in REP_LABELS], marker="^", label="Class-A Bayes", color="tab:green")
    for fam, c in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(x, curve(fam, "y_true", "main", "acc", f"{PRIMARY_MODEL}:classA"), marker="o",
                label=f"{fam} (Class-A)", color=c)
    ax.set_xticks(x); ax.set_xticklabels(REP_LABELS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("V: Bayes ceiling vs sequence reader (E, Class-A)")
    save(fig, "bayes_vs_reader")

    # 4 held-out operand vs history (y_true, Class-A)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for fam, c in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(x, curve(fam, "y_true", "heldout", "seen", f"{PRIMARY_MODEL}:classA"), marker="o",
                label=f"{fam} seen", color=c)
        ax.plot(x, curve(fam, "y_true", "heldout", "heldout", f"{PRIMARY_MODEL}:classA"), marker="s",
                ls="--", label=f"{fam} held-out", color=c)
    ax.axhline(HELDOUT_MATERIAL, ls=":", c="k", lw=0.8, label="0.30 material")
    ax.axhline(HELDOUT_WEAK, ls=":", c="gray", lw=0.8, label="0.25 weak")
    ax.set_xticks(x); ax.set_xticklabels(REP_LABELS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("accuracy (Class-A, y_true)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("V: seen vs held-out operand pairs (y_true)")
    save(fig, "heldout_vs_history")

    # 5 seen vs held-out at decisive horizons (all populations)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    labs, seen, hold = [], [], []
    for fam in FAMILIES:
        for rep in DECISIVE:
            for pop, pname in ((f"{PRIMARY_MODEL}:classA", "Class-A"), (f"{PRIMARY_MODEL}:all_add", "all-ADD")):
                labs.append(f"{fam}\n{rep}\n{pname}")
                seen.append(nz(rd(fam, "y_true", rep, "heldout", "seen", pop)))
                hold.append(nz(rd(fam, "y_true", rep, "heldout", "heldout", pop)))
    xi = np.arange(len(labs))
    ax.bar(xi - 0.2, seen, 0.4, label="seen", color="tab:gray")
    ax.bar(xi + 0.2, hold, 0.4, label="held-out", color="tab:purple")
    ax.set_xticks(xi); ax.set_xticklabels(labs, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("V: seen vs held-out operand pairs (y_true)")
    save(fig, "seen_vs_heldout")

    # 6 L4 comparison across populations
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    pops = [f"{PRIMARY_MODEL}:classA", f"{PRIMARY_MODEL}:matched_correct",
            f"{PRIMARY_MODEL}:correct_all", f"{PRIMARY_MODEL}:all_add"]
    xi = np.arange(len(pops))
    ax.bar(xi - 0.2, [nz(rd("GRU", "y_true", "L4", "main", "acc", p)) for p in pops], 0.4, label="GRU", color="tab:blue")
    ax.bar(xi + 0.2, [nz(rd("TCN", "y_true", "L4", "main", "acc", p)) for p in pops], 0.4, label="TCN", color="tab:orange")
    ax.plot(xi, [nz(U["probes"][p.split(":")[1]]["L4"]) for p in pops], "k^--", ms=5, label="U MLP-128")
    ax.set_xticks(xi); ax.set_xticklabels(["Class-A", "matched", "correct", "all-ADD"])
    ax.set_ylabel("TEST accuracy (L=4)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("V: L=4 sequence readers across populations")
    save(fig, "l4_comparison")

    # 7 full-prefix comparison across populations
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(xi - 0.2, [nz(rd("GRU", "y_true", "Lfull", "main", "acc", p)) for p in pops], 0.4, label="GRU", color="tab:blue")
    ax.bar(xi + 0.2, [nz(rd("TCN", "y_true", "Lfull", "main", "acc", p)) for p in pops], 0.4, label="TCN", color="tab:orange")
    ax.plot(xi, [nz(U["probes"][p.split(":")[1]]["Lfull"]) for p in pops], "k^--", ms=5, label="U MLP-128")
    ax.set_xticks(xi); ax.set_xticklabels(["Class-A", "matched", "correct", "all-ADD"])
    ax.set_ylabel("TEST accuracy (full prefix)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("V: full-prefix sequence readers across populations")
    save(fig, "fullprefix_comparison")

    # 8 per-seed (Class-A, y_true, Lfull)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    xi = np.arange(len(SEEDS))
    for fam, c in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(xi, [s["fits"][f"{fam}|y_true|Lfull|main"]["acc"][f"{PRIMARY_MODEL}:classA"]
                     for s in res["per_seed"]], marker="o", label=fam, color=c)
    ax.set_xticks(xi); ax.set_xticklabels([str(s) for s in SEEDS]); ax.set_xlabel("E seed")
    ax.set_ylabel("Class-A accuracy (full prefix)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("V: per-seed sequence-reader accuracy")
    save(fig, "per_seed")

    # 9 Class-A vs candidate-correct
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for fam, c in (("GRU", "tab:blue"), ("TCN", "tab:orange")):
        ax.plot(x, curve(fam, "y_true", "main", "acc", f"{PRIMARY_MODEL}:classA"), marker="o",
                label=f"{fam} Class-A", color=c)
        ax.plot(x, curve(fam, "y_true", "main", "acc", f"{PRIMARY_MODEL}:correct_all"), marker="s",
                ls="--", label=f"{fam} candidate-correct", color=c, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(REP_LABELS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("causal history window"); ax.set_ylabel("TEST accuracy")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("V: Class-A vs candidate-correct")
    save(fig, "classA_vs_correct")

    # 10 target vs operands (Class-A, full prefix)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    tgt_labels = ["y_true", "operand_a", "operand_b", "raw_sum", "op_dest"]
    xi = np.arange(len(tgt_labels))
    gv, tv = [], []
    for t in tgt_labels:
        gv.append(nz(rd("GRU", t, "Lfull", "main", "acc", f"{PRIMARY_MODEL}:classA")))
        tv.append(nz(rd("TCN", t, "Lfull", "main", "acc", f"{PRIMARY_MODEL}:classA")))
    ax.bar(xi - 0.2, gv, 0.4, label="GRU", color="tab:blue")
    ax.bar(xi + 0.2, tv, 0.4, label="TCN", color="tab:orange")
    ax.set_xticks(xi); ax.set_xticklabels(tgt_labels, rotation=15)
    ax.set_ylabel("TEST accuracy (Class-A, full prefix)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("V: target vs operands (full prefix)")
    save(fig, "target_vs_operands")

    # 11 parameter budget
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    a = res["architecture"]
    ax.bar(np.arange(3), [a["budget_u_mlp128"], a["GRU"]["params"], a["TCN"]["params"]],
           color=["tab:gray", "tab:blue", "tab:orange"])
    ax.axhline(a["budget_u_mlp128"], ls="--", c="k", lw=0.9, label="U MLP-128 budget")
    for i, v in enumerate([a["budget_u_mlp128"], a["GRU"]["params"], a["TCN"]["params"]]):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["U MLP-128", f"GRU h={a['GRU']['hidden']}",
                                                     f"TCN c={a['TCN']['channels']}"])
    ax.set_ylabel("trainable parameters"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("V: parameter parity within the U MLP-128 budget")
    save(fig, "parameter_budget")

    # 12 decision summary
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    d = res["primary_decision"]
    cats = ["best reader\n(Class-A)", "best reader\nheld-out", "U MLP-128\n(Class-A)", "Class-A\nBayes",
            "Bayes - reader"]
    vals = [nz(d["E_metrics"]["best"]), nz(d["E_metrics"]["best_heldout"]), nz(d["E_metrics"]["u_mlp128_best"]),
            nz(d["E_metrics"]["bayes"]), nz(d["E_metrics"]["bayes_minus_reader"])]
    ax.bar(np.arange(5), vals, color=["tab:blue", "tab:purple", "tab:gray", "tab:green", "tab:red"])
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.8, label="0.50 bar")
    ax.axhline(HELDOUT_MATERIAL, ls=":", c="k", lw=0.8, label="0.30 material")
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"V: decision summary ({d['verdict']})")
    save(fig, "decision_summary")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment V analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_v_sequence_access.npz")
    ap.add_argument("--u-results", default="experiments/experiment_u_results.json")
    ap.add_argument("--output", default="experiments/experiment_v_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_v")
    args = ap.parse_args()
    data = load(args.npz)
    u_base = load_u_baseline(args.u_results)
    res = build(data, u_base)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"E: {d['verdict']}")
    m = d["E_metrics"]
    print(f"  best={m['best']} ({m['best_family']}/{m['best_rep']}) heldout={m['best_heldout']} "
          f"U_mlp128={m['u_mlp128_best']} bayes={m['bayes']} gap={m['bayes_minus_reader']} gain={m['gain_over_u']}")
    print(f"  arch: GRU h={res['architecture']['GRU']['hidden']} ({res['architecture']['GRU']['params']}), "
          f"TCN c={res['architecture']['TCN']['channels']} ({res['architecture']['TCN']['params']}), "
          f"budget={res['architecture']['budget_u_mlp128']}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
