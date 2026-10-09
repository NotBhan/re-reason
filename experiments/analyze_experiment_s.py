#!/usr/bin/env python3
"""Experiment S — analysis: candidate-INTERNAL activation recoverability on the Class-A hard tail.

Localises Q/R's weak recoverability of the true written ADD value: is it already absent from the
frozen candidate network's internal activations (a deeper computation limit), or present internally
but missing from the final candidate vector C_t (an output-projection / output-geometry limit)?

Predeclared activation stages (from the ACTUAL implementation; see run_experiment_s.py):
    S_prev(212) f_gate(212) h_reset(212) cand_input(468) h1_pre(146) h1_post(146) out_pre(212) C_t(212)
Candidate-network chain: cand_input -> h1_pre -> h1_post -> out_pre -> C_t.

PRE-REGISTRATION (declared before inspecting any TEST aggregate)
---------------------------------------------------------------
* Activation set: all 8 stages are reported; the primary localization chain is the candidate chain.
* Probe family (R's, fixed and not extended after results): linear ridge; MLP-16/32/64/128 (tanh, Adam
  3e-3, weight-decay 1e-2, 200 epochs, early stop on val, fixed seed, balanced class weights).
* Capacity design: ALL stages x {linear, mlp16, mlp32}; predeclared subset
  {cand_input, h1_post, out_pre, C_t} x {mlp64, mlp128}.
* Distributions (R's): D0 ordinary, D0m size-matched ordinary, D1 designated hard-tail, D1x augmented
  i.i.d. hard pool. Evaluation is always the SAME untouched canonical-TEST Class-A hard tail.
* Held-out operand protocol: 20 pairs (b in {0,5}); the probe is fit excluding them and evaluated on
  hard-test rows with held-out vs seen pairs (D0 and D1x).
* Localization verdict (checked in order; bars are declared conventions):
    S-MIX  subtype spread |wrap-nowrap| >= 0.15, or E/F verdict categories disagree.
    S-A    (output/downstream) an internal chain stage A (not C_t) shows hard-test recoverability
           >= C_t + 0.10 under D1x/mlp32, reproduced >=4/5 seeds, held-out-operand accuracy
           >= 0.30 and >= C_t held-out + 0.10, and seen-heldout gap < 0.20.
    S-B    (progressive loss) the D1x/mlp32 chain scalar is ~non-increasing with a max adjacent drop
           >= 0.10, earliest chain stage >= 0.30, largest drop reproducible >=4/5 seeds.
    S-C    (computation-limited) max internal gain over C_t < 0.10 and max internal held-out <= 0.25.
    else   S-MIX.
No test-based activation selection: every predeclared stage is reported.

Reproduce:
    python experiments/analyze_experiment_s.py \
        --npz experiments/experiment_s_candidate_internal.npz \
        --output experiments/experiment_s_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_s
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

import run_experiment_s as rs  # noqa: E402

# Fixed thread count: makes the probe fits fast AND bit-reproducible across runs.
import torch as _torch  # noqa: E402
_TORCH_THREADS = int(os.environ.get("S_TORCH_THREADS", "8"))
_torch.set_num_threads(_TORCH_THREADS)

PRIMARY_MODEL = "E_candidate"
CONTROL_MODEL = "F_combined"
SEEDS = [42, 43, 44, 45, 46]
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]
N_MOD = 10
N_RAW = 19
CAPS = [("linear", None), ("mlp16", 16), ("mlp32", 32), ("mlp64", 64), ("mlp128", 128)]
REGIMES = ["D0", "D0m", "D1", "D1x"]
REF_CAP = "mlp32"
STAGES = rs.STAGES
STAGE_DIMS = rs.STAGE_DIMS
CHAIN = rs.PRIMARY_CHAIN                       # ["cand_input","h1_pre","h1_post","out_pre","C_t"]
ALL_STAGE_CAPS = ["linear", "mlp16", "mlp32"]  # every stage x these
FULL_STAGES = ["cand_input", "h1_post", "out_pre", "C_t"]  # predeclared subset for mlp64/128
FULL_CAPS = ["mlp64", "mlp128"]
ARITH_CAPS = ["linear", "mlp32"]
HELD_OUT_CAPS = ["linear", "mlp32"]
HELD_OUT_PAIRS = {(a, b) for a in range(10) for b in range(10) if b in (0, 5)}
MLP_LR = 3e-3
MLP_WD = 1e-2
MLP_EPOCHS = 200
MLP_SEED = 20261007
CHANCE = 0.10
# decision bars (declared conventions)
RECOVER_BAR = 0.50
MAT_MARGIN = 0.10
MIX_BAR = 0.15
HELDOUT_BAR = 0.30
MEMO_GAP = 0.20
MEMO_HELDOUT = 0.25

_TRAIN_DS = {"D0": "trainord", "D0m": "trainord", "D1": "trainhard", "D1x": "hardextra"}
_VAL_DS = {"D0": "valord", "D0m": "valord", "D1": "valhard", "D1x": "valhard"}


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


def cap_list_for(stage: str) -> List[Tuple[str, Optional[int]]]:
    out = [(n, h) for n, h in CAPS if n in ALL_STAGE_CAPS]
    if stage in FULL_STAGES:
        out += [(n, h) for n, h in CAPS if n in FULL_CAPS]
    return out


# ---------------------------------------------------------------------------
# probes (analysis-only; identical conventions to Experiment R)
# ---------------------------------------------------------------------------

def class_weights(y: np.ndarray, nclass: int) -> np.ndarray:
    w = np.ones(nclass, float)
    for c in range(nclass):
        n = int((y == c).sum())
        w[c] = (len(y) / (nclass * n)) if n > 0 else 0.0
    return w


def ridge_fit(Xtr, ytr, wc, Xva, yva, nclass) -> Tuple[Any, float, float]:
    wrow = wc[ytr]
    Xb = np.c_[Xtr, np.ones(len(Xtr))]
    Y = np.eye(nclass)[ytr] * wrow[:, None]
    Xw = Xb * wrow[:, None]
    G = Xw.T @ Xb
    XtY = Xw.T @ Y
    Xvb = np.c_[Xva, np.ones(len(Xva))]
    best = (None, -1.0, None)
    for lam in LAMBDAS:
        W = np.linalg.solve(G + lam * np.eye(Xb.shape[1]), XtY)
        acc = float(((Xvb @ W).argmax(1) == yva).mean())
        if acc > best[1]:
            best = (W, acc, lam)
    W = best[0]
    return (lambda X: np.c_[X, np.ones(len(X))] @ W), best[1], best[2]


def mlp_fit(Xtr, ytr, Xva, yva, hidden, wc, seed, nclass=N_MOD) -> Tuple[Any, float]:
    import torch
    torch.manual_seed(seed)
    m = Xtr.mean(0)
    s = Xtr.std(0) + 1e-6
    Xn = torch.tensor((Xtr - m) / s, dtype=torch.float32)
    yn = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor((Xva - m) / s, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.long)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, nclass))
    wt = torch.tensor(wc, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    best = (-1.0, None)
    for ep in range(MLP_EPOCHS):
        net.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(Xn), yn, weight=wt)
        loss.backward()
        opt.step()
        if ep % 10 == 9:
            net.eval()
            with torch.no_grad():
                v = float((net(Xv).argmax(1) == yv).float().mean())
            if v > best[0]:
                best = (v, {k: val.clone() for k, val in net.state_dict().items()})
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()

    def predict(X):
        with torch.no_grad():
            return net(torch.tensor((X - m) / s, dtype=torch.float32)).numpy()
    return predict, best[0]


def param_count(kind: str, hidden: Optional[int], indim: int) -> int:
    if kind == "linear":
        return indim * N_MOD + N_MOD
    return indim * hidden + hidden + hidden * N_MOD + N_MOD


def macro_acc(pred, y, nclass=N_MOD):
    recs = []
    for c in range(nclass):
        m = y == c
        if m.any():
            recs.append(float((pred[m] == c).mean()))
    return float(np.mean(recs)) if recs else None


def _operand_pairs(z, m, ds, idx):
    oa = z[f"{m}__{ds}__operand_a"][idx]
    ob = z[f"{m}__{ds}__operand_b"][idx]
    return np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oa, ob)])


def fit_regime(z, m, seed, regime, stage, kind, hidden, target="mod",
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
    lab = "y_true" if target == "mod" else "raw_sum"
    nclass = N_MOD if target == "mod" else N_RAW
    Xtr = z[f"{m}__{tds}__{stage}"][tr].astype(np.float64)
    ytr = z[f"{m}__{tds}__{lab}"][tr].astype(int)
    Xva = z[f"{m}__{vds}__{stage}"][va].astype(np.float64)
    yva = z[f"{m}__{vds}__{lab}"][va].astype(int)
    wc = class_weights(ytr, nclass)
    if kind == "linear":
        pred, vacc, lam = ridge_fit(Xtr, ytr, wc, Xva, yva, nclass)
    else:
        pred, vacc = mlp_fit(Xtr, ytr, Xva, yva, hidden, wc, MLP_SEED + seed * 100 + hidden, nclass)
        lam = None
    return {"predict": pred, "train_acc": float((pred(Xtr).argmax(1) == ytr).mean()),
            "val_acc": vacc, "n_train": len(tr), "n_val": len(va), "lambda": lam}


def test_masks(z, m, seed):
    sel = (z[f"{m}__test__seed"] == seed)
    ok = z[f"{m}__test__cand_ok"].astype(bool)
    return sel & (~ok), sel & ok


def matched_control(z, m, seed) -> np.ndarray:
    hard, correct = test_masks(z, m, seed)
    feats = ["wrap", "repeated", "t", "m_cand", "norm_C", "norm_dC"]

    def F(mask):
        cols = []
        for f in feats:
            v = z[f"{m}__test__{f}"][mask].astype(np.float64)
            if f == "m_cand":
                v = np.abs(v)
            cols.append(v)
        return np.stack(cols, 1)
    A = F(hard)
    B = F(correct)
    mu = np.concatenate([A, B], 0).mean(0)
    sd = np.concatenate([A, B], 0).std(0) + 1e-9
    A = (A - mu) / sd
    B = (B - mu) / sd
    used = np.zeros(len(B), bool)
    pick = np.full(len(A), -1, int)
    for i in np.argsort(A.sum(1)):
        d = np.linalg.norm(B - A[i], axis=1)
        d[used] = np.inf
        j = int(d.argmin())
        if np.isfinite(d[j]):
            used[j] = True
            pick[i] = j
    correct_idx = np.nonzero(correct)[0]
    return correct_idx[pick[pick >= 0]]


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
        return macro_acc(pred(X).argmax(1), y)

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


def geometry(z, m, seed) -> Dict[str, Any]:
    """Class geometry of the Class-A test tail at each stage (norms, class separation, centroid dist)."""
    hard, correct = test_masks(z, m, seed)
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


def analyze_seed(z, m, seed) -> Dict[str, Any]:
    hard_mask, correct_mask = test_masks(z, m, seed)
    matched_idx = matched_control(z, m, seed)
    matched_mask = np.zeros(len(hard_mask), bool)
    matched_mask[matched_idx] = True
    res: Dict[str, Any] = {"seed": seed, "n_hard": int(hard_mask.sum()),
                           "n_correct": int(correct_mask.sum()), "n_matched": int(matched_mask.sum()),
                           "caps": {}, "arithmetic": {}, "heldout_operand": {}, "geometry": {},
                           "confusion": {}}
    for regime in REGIMES:
        res["caps"][regime] = {}
        for stage in STAGES:
            res["caps"][regime][stage] = {}
            for name, hidden in cap_list_for(stage):
                kind = "linear" if name == "linear" else "mlp"
                f = fit_regime(z, m, seed, regime, stage, kind, hidden)
                ev = evaluate(z, m, f, stage, hard_mask, correct_mask, matched_mask)
                ev.update({"train_acc": f["train_acc"], "val_acc": f["val_acc"], "n_train": f["n_train"],
                           "params": param_count(kind, hidden, STAGE_DIMS[stage])})
                res["caps"][regime][stage][name] = ev
    # raw-sum vs modular (all stages; D0 and D1x; linear + mlp32)
    for regime in ("D0", "D1x"):
        res["arithmetic"][regime] = {}
        for stage in STAGES:
            res["arithmetic"][regime][stage] = {}
            for name in ARITH_CAPS:
                hidden = dict(CAPS)[name]
                kind = "linear" if name == "linear" else "mlp"
                fr = fit_regime(z, m, seed, regime, stage, kind, hidden, target="raw")
                X = z[f"{m}__test__{stage}"][hard_mask].astype(np.float64)
                yr = z[f"{m}__test__raw_sum"][hard_mask].astype(int)
                res["arithmetic"][regime][stage][name] = float((fr["predict"](X).argmax(1) == yr).mean())
    # held-out-operand generalization (selected stages; D0 and D1x; linear + mlp32)
    for regime in ("D0", "D1x"):
        res["heldout_operand"][regime] = {}
        for stage in FULL_STAGES:
            res["heldout_operand"][regime][stage] = {}
            for name in HELD_OUT_CAPS:
                res["heldout_operand"][regime][stage][name] = heldout_fit(z, m, seed, stage, name, regime, hard_mask)
    res["geometry"] = geometry(z, m, seed)
    # confusion for the primary scalar (D1x/mlp32) on C_t and on the best-scalar internal stage
    best_internal = max(("cand_input", "h1_pre", "h1_post", "out_pre"),
                        key=lambda s: res["caps"]["D1x"][s][REF_CAP]["hardtest"])
    for stage in ("C_t", best_internal):
        f = fit_regime(z, m, seed, "D1x", stage, "mlp", dict(CAPS)[REF_CAP])
        X = z[f"{m}__test__{stage}"][hard_mask].astype(np.float64)
        y = z[f"{m}__test__y_true"][hard_mask].astype(int)
        p = f["predict"](X).argmax(1)
        Cm = np.zeros((N_MOD, N_MOD), int)
        for t, q in zip(p, y):
            Cm[q, t] += 1
        res["confusion"][stage] = Cm.tolist()
    res["best_internal_stage"] = best_internal
    return res


def heldout_fit(z, m, seed, stage, name, regime, hard_mask) -> Dict[str, Any]:
    kind = "linear" if name == "linear" else "mlp"
    hidden = dict(CAPS)[name]
    f = fit_regime(z, m, seed, regime, stage, kind, hidden, exclude_heldout=True)
    X = z[f"{m}__test__{stage}"][hard_mask].astype(np.float64)
    y = z[f"{m}__test__y_true"][hard_mask].astype(int)
    h2 = _operand_pairs(z, m, "test", np.nonzero(hard_mask)[0])
    p = f["predict"](X).argmax(1)
    return {"val_acc": f["val_acc"], "n_fit": f["n_train"],
            "acc_all": float((p == y).mean()),
            "acc_heldout": float((p[h2] == y[h2]).mean()) if h2.any() else None,
            "acc_seen": float((p[~h2] == y[~h2]).mean()) if (~h2).any() else None,
            "n_heldout": int(h2.sum())}


def mean_sd(vals: List[Optional[float]]) -> Dict[str, Any]:
    a = np.array([v for v in vals if v is not None], float)
    return {"mean": float(a.mean()) if a.size else None,
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "per_seed": vals}


def paired(a: List[Optional[float]], b: List[Optional[float]]) -> Dict[str, Any]:
    x = np.array([v for v in a if v is not None], float)
    y = np.array([v for v in b if v is not None], float)
    d = x - y
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    out = {"a_mean": float(x.mean()), "b_mean": float(y.mean()), "mean_diff": float(d.mean()),
           "sign_positive": int((d > 0).sum()), "sign_negative": int((d < 0).sum()),
           "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan")}
    try:
        out["paired_t_p"] = float(st.ttest_rel(x, y).pvalue)
    except Exception:
        out["paired_t_p"] = None
    return out


def aggregate(per: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_seed": per, "caps": {}, "arithmetic": {}, "heldout_operand": {},
                           "geometry": {}}
    fields = ["hardtest", "hardtest_macro", "recoverability", "hard_wrap", "hard_nowrap", "hard_repeated",
              "hard_isolated", "hard_heldout_pairs", "hard_seen_pairs", "correct_all", "correct_matched",
              "train_acc", "val_acc", "n_train"]
    for regime in REGIMES:
        out["caps"][regime] = {}
        for stage in STAGES:
            out["caps"][regime][stage] = {}
            for name, _ in cap_list_for(stage):
                out["caps"][regime][stage][name] = {
                    f: mean_sd([p["caps"][regime][stage][name][f] for p in per]) for f in fields}
                out["caps"][regime][stage][name]["params"] = per[0]["caps"][regime][stage][name]["params"]
    for regime in ("D0", "D1x"):
        out["arithmetic"][regime] = {st: {name: mean_sd([p["arithmetic"][regime][st][name] for p in per])
                                          for name in ARITH_CAPS} for st in STAGES}
        out["heldout_operand"][regime] = {
            st: {name: {f: mean_sd([p["heldout_operand"][regime][st][name][f] for p in per])
                        for f in ("val_acc", "acc_all", "acc_heldout", "acc_seen")}
                 for name in HELD_OUT_CAPS} for st in FULL_STAGES}
    for st in STAGES:
        out["geometry"][st] = {f: mean_sd([p["geometry"][st][f] for p in per])
                               for f in ("norm_hard", "norm_correct", "within_var", "between_var",
                                         "fisher", "centroid_dist", "centroid_cos")}
    # chain scalars and adjacent drops (D1x/mlp32 and D0/mlp32)
    out["chain"] = {}
    for regime in ("D0", "D1x"):
        out["chain"][regime] = {st: out["caps"][regime][st][REF_CAP]["hardtest"]["mean"] for st in CHAIN}
        drops = {}
        for i in range(len(CHAIN) - 1):
            a, b = CHAIN[i], CHAIN[i + 1]
            drops[f"{a}->{b}"] = (out["chain"][regime][b] - out["chain"][regime][a]
                                  if out["chain"][regime][b] is not None and out["chain"][regime][a] is not None
                                  else None)
        out["chain"][regime + "_drops"] = drops
    out["internal_vs_Ct"] = {
        st: paired([p["caps"]["D1x"][st][REF_CAP]["hardtest"] for p in per],
                   [p["caps"]["D1x"]["C_t"][REF_CAP]["hardtest"] for p in per])
        for st in STAGES}
    out["internal_vs_Ct_D0"] = {
        st: paired([p["caps"]["D0"][st][REF_CAP]["hardtest"] for p in per],
                   [p["caps"]["D0"]["C_t"][REF_CAP]["hardtest"] for p in per])
        for st in STAGES}
    out["confusion"] = {}
    for stage in set(p["best_internal_stage"] for p in per) | {"C_t"}:
        Cm = np.zeros((N_MOD, N_MOD), int)
        for p in per:
            if stage in p["confusion"]:
                Cm += np.array(p["confusion"][stage], int)
        out["confusion"][stage] = Cm.tolist()
    return out


def _verdict_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    """Predeclared localization metrics from an aggregated model result (mean over seeds)."""
    caps = res["caps"]
    def s(regime, stage, cap=REF_CAP):
        return caps[regime][stage][cap]["hardtest"]["mean"]
    internal = ["cand_input", "h1_pre", "h1_post", "out_pre"]
    s1x = {st: s("D1x", st) for st in STAGES}
    s0 = {st: s("D0", st) for st in STAGES}
    # held-out generalization (D1x)
    ho = {st: res["heldout_operand"]["D1x"][st][REF_CAP] for st in FULL_STAGES}
    ho_acc = {st: ho[st]["acc_heldout"]["mean"] for st in FULL_STAGES}
    seen_acc = {st: ho[st]["acc_seen"]["mean"] for st in FULL_STAGES}
    gap = {st: (seen_acc[st] - ho_acc[st]) if (seen_acc[st] is not None and ho_acc[st] is not None) else None
           for st in FULL_STAGES}
    gain = {st: (s1x[st] - s1x["C_t"]) if (s1x[st] is not None and s1x["C_t"] is not None) else None
            for st in STAGES}
    # subtype spread at REF_CAP (D0): wrap vs no-wrap
    spread = {}
    for st in STAGES:
        w = caps["D0"][st][REF_CAP]["hard_wrap"]["mean"]
        nw = caps["D0"][st][REF_CAP]["hard_nowrap"]["mean"]
        if w is not None and nw is not None:
            spread[st] = abs(w - nw)
    drops = res["chain"]["D1x_drops"]
    max_drop = max([v for v in drops.values() if v is not None], default=None)
    max_drop_key = None
    if max_drop is not None:
        max_drop_key = max(((k, v) for k, v in drops.items() if v is not None), key=lambda kv: kv[1])[0]
    return {"s1x": s1x, "s0": s0, "gain_over_Ct": gain, "heldout_acc": ho_acc, "seen_acc": seen_acc,
            "seen_heldout_gap": gap, "subtype_spread_D0": spread, "max_adjacent_drop": max_drop,
            "max_adjacent_drop_key": max_drop_key, "chain_d1x": res["chain"]["D1x"]}


def decide(res: Dict[str, Any]) -> Dict[str, Any]:
    met = {}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        met[m] = _verdict_metrics(res[m])
    mE, mF = met[PRIMARY_MODEL], met[CONTROL_MODEL]
    internal = ["cand_input", "h1_pre", "h1_post", "out_pre"]
    # memorization check is folded into S-C; explicit flags reported
    best_internal_gain = max((v for k, v in mE["gain_over_Ct"].items() if k in internal and v is not None),
                             default=None)
    best_internal_stage = None
    if best_internal_gain is not None:
        best_internal_stage = max((k for k in internal if mE["gain_over_Ct"][k] is not None),
                                  key=lambda k: mE["gain_over_Ct"][k])
    max_internal_ho = max((v for v in (mE["heldout_acc"].get(k) for k in FULL_STAGES) if v is not None),
                          default=None)
    # reproducible internal gain over C_t (>=4/5 seeds) at the best internal stage
    repro_positive = None
    if best_internal_stage is not None:
        pb = res[PRIMARY_MODEL]["internal_vs_Ct"][best_internal_stage]
        repro_positive = max(pb["sign_positive"], pb["sign_negative"])
    # subtype spread: how many stages exceed the mix bar
    spread_over = sum(1 for v in mE["subtype_spread_D0"].values() if v is not None and v >= MIX_BAR)
    # S-A condition on the best internal stage
    sA = False
    if best_internal_stage is not None and best_internal_gain is not None:
        st = best_internal_stage
        gain_ok = best_internal_gain >= MAT_MARGIN
        repro_ok = (repro_positive is not None and repro_positive >= 4)
        ho_st = mE["heldout_acc"].get(st)
        ho_ct = mE["heldout_acc"].get("C_t")
        gap_st = mE["seen_heldout_gap"].get(st)
        ho_ok = (ho_st is not None and ho_st >= HELDOUT_BAR and ho_ct is not None and (ho_st - ho_ct) >= MAT_MARGIN)
        gap_ok = (gap_st is None or gap_st < MEMO_GAP)
        sA = bool(gain_ok and repro_ok and ho_ok and gap_ok)
    # S-B: chain non-increasing (tolerance) with material max drop and early stage signal
    chain = mE["chain_d1x"]
    vals = [chain.get(k) for k in CHAIN]
    noninc = all((vals[i + 1] is None or vals[i] is None or vals[i + 1] <= vals[i] + 0.05)
                 for i in range(len(vals) - 1))
    max_drop = mE["max_adjacent_drop"]
    sB = bool(noninc and max_drop is not None and max_drop >= MAT_MARGIN
              and vals[0] is not None and vals[0] >= HELDOUT_BAR)
    # S-C: no material internal gain and no held-out generalization
    sC = bool((best_internal_gain is not None and best_internal_gain < MAT_MARGIN)
              and (max_internal_ho is None or max_internal_ho <= MEMO_HELDOUT))
    # E/F category agreement handled outside (needs both decisions) -> provisional here
    if spread_over >= 2:
        verdict = "S-MIX"
    elif sA:
        verdict = "S-A_downstream_output_localized"
    elif sB:
        verdict = "S-B_progressive_loss"
    elif sC:
        verdict = "S-C_computation_limited"
    else:
        verdict = "S-MIX"
    return {"verdict": verdict, "best_internal_stage": best_internal_stage,
            "best_internal_gain_over_Ct": best_internal_gain, "reproducible_seeds": repro_positive,
            "max_internal_heldout_acc": max_internal_ho, "subtype_stages_over_mix_bar": spread_over,
            "sA_conditions": {"gain": sA}, "sB_conditions": {"nonincreasing": noninc}, "sC_conditions": {"ok": sC},
            "E_metrics": mE, "F_metrics": mF,
            "thresholds": {"recover_bar": RECOVER_BAR, "mat_margin": MAT_MARGIN, "mix_bar": MIX_BAR,
                           "heldout_bar": HELDOUT_BAR, "memo_gap": MEMO_GAP, "memo_heldout": MEMO_HELDOUT},
            "rule_note": ("S-MIX if subtype spread >= 0.15 at >=2 stages or E/F disagree; S-A if an internal "
                          "chain stage beats C_t by >=0.10 (>=4/5 seeds) with held-out accuracy >=0.30 and "
                          "seen-heldout gap <0.20; S-B if the D1x chain scalar is non-increasing with max drop "
                          ">=0.10 and earliest stage >=0.30; S-C if no internal gain <0.10 and held-out <=0.25. "
                          "Bars are declared conventions; continuous effect sizes are reported alongside.")}


def analyze_model(z, meta, m) -> Dict[str, Any]:
    per = [analyze_seed(z, m, s) for s in SEEDS]
    agg = aggregate(per)
    return agg


def build(data: Dict[str, Any]) -> Dict[str, Any]:
    z = data["z"]
    meta = data["meta"]
    res: Dict[str, Any] = {"metadata": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS,
        "checkpoints": meta["models"], "splits": meta["n"],
        "candidate_network": meta["candidate_network"], "stages": STAGES, "stage_dims": STAGE_DIMS,
        "primary_chain": CHAIN, "caps": [(n, h) for n, h in CAPS],
        "all_stage_caps": ALL_STAGE_CAPS, "full_stages": FULL_STAGES, "full_caps": FULL_CAPS,
        "regimes": REGIMES, "thresholds": {"recover_bar": RECOVER_BAR, "mat_margin": MAT_MARGIN,
                                           "mix_bar": MIX_BAR, "heldout_bar": HELDOUT_BAR,
                                           "memo_gap": MEMO_GAP, "memo_heldout": MEMO_HELDOUT},
        "statistical_discipline": ("Five frozen E checkpoints are the independent units; F is a control. "
                                   "Probes are analysis-only, fit on a regime's train, tuned on its val, "
                                   "evaluated once on the untouched test hard tail. Every predeclared "
                                   "stage is reported; no test-based activation selection.")}}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        res[m] = analyze_model(z, meta, m)
    dec = decide(res)
    # E/F agreement: recompute verdicts per model and flag disagreements as S-MIX
    vE = dec["verdict"]
    chainF = res[CONTROL_MODEL]["chain"]["D1x"]
    metF = dec["F_metrics"]
    internal = ["cand_input", "h1_pre", "h1_post", "out_pre"]
    gainF = max((v for k, v in metF["gain_over_Ct"].items() if k in internal and v is not None), default=None)
    agree = ((vE == "S-C_computation_limited") == ((gainF is not None and gainF < MAT_MARGIN)))
    if not agree:
        dec["verdict"] = "S-MIX"
        dec["E_F_disagreement"] = True
    else:
        dec["E_F_disagreement"] = False
    res["E_vs_F"] = {st: paired([p["caps"]["D1x"][st][REF_CAP]["hardtest"]
                                 for p in res[PRIMARY_MODEL]["per_seed"]],
                                [p["caps"]["D1x"][st][REF_CAP]["hardtest"]
                                 for p in res[CONTROL_MODEL]["per_seed"]]) for st in STAGES}
    res["primary_decision"] = dec
    return res


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_s") -> List[str]:
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
    names = [n for n, _ in CAPS]
    x = np.arange(len(names))
    sx = np.arange(len(STAGES))

    def s1x(st):
        return rE["caps"]["D1x"][st][REF_CAP]["hardtest"]["mean"]

    def s0(st):
        return rE["caps"]["D0"][st][REF_CAP]["hardtest"]["mean"]
    # 1 recoverability by stage
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(sx - 0.2, [s1x(s) for s in STAGES], 0.4, label="D1x/mlp32", color="tab:orange")
    ax.bar(sx + 0.2, [s0(s) for s in STAGES], 0.4, label="D0/mlp32", color="tab:blue")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.9, label="0.50 bar")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=35, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: recoverability by candidate activation stage")
    save(fig, "recoverability_by_stage")
    # 2 capacity curves by stage (D1x)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for st in STAGES:
        xs_, ys_ = [], []
        for i, (name, _) in enumerate(CAPS):
            if name in rE["caps"]["D1x"][st]:
                xs_.append(i); ys_.append(rE["caps"]["D1x"][st][name]["hardtest"]["mean"])
        ax.plot(xs_, ys_, marker="o", label=st)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("hard-test accuracy (D1x)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("E: capacity curves by stage (D1x)")
    save(fig, "capacity_curves_by_stage")
    # 3 ordinary vs hard by stage
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(sx - 0.2, [s0(s) for s in STAGES], 0.4, label="D0 ordinary", color="tab:blue")
    ax.bar(sx + 0.2, [s1x(s) for s in STAGES], 0.4, label="D1x hard", color="tab:orange")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=35, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: ordinary- vs hard-trained by stage")
    save(fig, "ordinary_vs_hard_by_stage")
    # 4 capacity x distribution heatmap (C_t and best internal)
    best_internal = res["primary_decision"]["best_internal_stage"] or "h1_post"
    for st in ("C_t", best_internal):
        M = np.array([[rE["caps"][reg][st][n]["hardtest"]["mean"] for n in names if n in rE["caps"][reg][st]]
                      for reg in REGIMES])
        fig, ax = plt.subplots(figsize=(7, 4.6))
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(0.6, float(M.max())))
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
        ax.set_yticks(range(len(REGIMES))); ax.set_yticklabels(REGIMES)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
        fig.colorbar(im, ax=ax); ax.set_title(f"E: capacity x distribution ({st})")
        save(fig, "capacity_distribution_heatmap" if st == "C_t" else f"capacity_distribution_heatmap_{st}")
    # 5 held-out operand accuracy by stage
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    xs2 = np.arange(len(FULL_STAGES))
    ax.bar(xs2 - 0.2, [rE["heldout_operand"]["D1x"][st]["mlp32"]["acc_seen"]["mean"] for st in FULL_STAGES],
           0.4, label="seen pairs", color="tab:gray")
    ax.bar(xs2 + 0.2, [rE["heldout_operand"]["D1x"][st]["mlp32"]["acc_heldout"]["mean"] for st in FULL_STAGES],
           0.4, label="held-out pairs", color="tab:purple")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(xs2); ax.set_xticklabels(FULL_STAGES, rotation=30, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E (D1x): held-out vs seen operand pairs by stage")
    save(fig, "heldout_operand_by_stage")
    # 6 seen-heldout gap by stage
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    gaps = [(rE["heldout_operand"]["D1x"][st]["mlp32"]["acc_seen"]["mean"]
             - rE["heldout_operand"]["D1x"][st]["mlp32"]["acc_heldout"]["mean"]) for st in FULL_STAGES]
    ax.bar(xs2, gaps, color="tab:red")
    ax.axhline(MEMO_GAP, ls="--", c="k", lw=0.9, label=f"{MEMO_GAP} bar")
    ax.set_xticks(xs2); ax.set_xticklabels(FULL_STAGES, rotation=30, ha="right")
    ax.set_ylabel("seen - held-out accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E (D1x): seen-vs-held-out gap by stage")
    save(fig, "seen_vs_heldout_gap")
    # 7 raw-sum vs modular by stage
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(sx - 0.2, [rE["caps"]["D1x"][st]["mlp32"]["hardtest"]["mean"] for st in STAGES], 0.4,
           label="modular (D1x/mlp32)", color="tab:blue")
    ax.bar(sx + 0.2, [rE["arithmetic"]["D1x"][st]["mlp32"]["mean"] for st in STAGES], 0.4,
           label="raw sum (D1x/mlp32)", color="tab:orange")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=35, ha="right")
    ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: modular vs raw-sum target by stage")
    save(fig, "rawsum_vs_modular_by_stage")
    # 8 wrap vs no-wrap by stage
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(sx - 0.2, [rE["caps"]["D0"][st]["mlp32"]["hard_wrap"]["mean"] for st in STAGES], 0.4,
           label="wrap (D0)", color="tab:red")
    ax.bar(sx + 0.2, [rE["caps"]["D0"][st]["mlp32"]["hard_nowrap"]["mean"] for st in STAGES], 0.4,
           label="no-wrap (D0)", color="tab:cyan")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=35, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E (D0): wrap vs no-wrap by stage")
    save(fig, "wrap_nowrap_by_stage")
    # 9 repeated vs isolated by stage
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(sx - 0.2, [rE["caps"]["D0"][st]["mlp32"]["hard_repeated"]["mean"] for st in STAGES], 0.4,
           label="repeated (D0)", color="tab:green")
    ax.bar(sx + 0.2, [rE["caps"]["D0"][st]["mlp32"]["hard_isolated"]["mean"] for st in STAGES], 0.4,
           label="isolated (D0)", color="tab:olive")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=35, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E (D0): repeated vs isolated ADD by stage")
    save(fig, "repeated_isolated_by_stage")
    # 10 preactivation vs postactivation
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    pairs = [("h1_pre", "h1_post"), ("out_pre", "C_t")]
    labels = ["hidden (pre vs post)", "output (pre vs C_t)"]
    ax.bar(np.arange(2) - 0.2, [rE["caps"]["D1x"][a][REF_CAP]["hardtest"]["mean"] for a, b in pairs],
           0.4, label="preactivation", color="tab:blue")
    ax.bar(np.arange(2) + 0.2, [rE["caps"]["D1x"][b][REF_CAP]["hardtest"]["mean"] for a, b in pairs],
           0.4, label="postactivation / C_t", color="tab:orange")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(labels)
    ax.set_ylabel("hard-test accuracy (D1x)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: preactivation vs postactivation")
    save(fig, "preactivation_vs_postactivation")
    # 11 information trajectory
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    cx = np.arange(len(CHAIN))
    ax.plot(cx, [rE["chain"]["D1x"][s] for s in CHAIN], marker="o", label="D1x/mlp32", color="tab:orange")
    ax.plot(cx, [rE["chain"]["D0"][s] for s in CHAIN], marker="s", label="D0/mlp32", color="tab:blue")
    ax.axhline(CHANCE, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(cx); ax.set_xticklabels(CHAIN, rotation=30, ha="right")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E: candidate-input -> C_t information trajectory")
    save(fig, "information_trajectory")
    # 12 candidate-correct vs Class-A centroid geometry
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(sx - 0.2, [rE["geometry"][st]["norm_hard"]["mean"] for st in STAGES], 0.4,
           label="Class-A norm", color="tab:red")
    ax.bar(sx + 0.2, [rE["geometry"][st]["norm_correct"]["mean"] for st in STAGES], 0.4,
           label="candidate-correct norm", color="tab:green")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=35, ha="right")
    ax.set_ylabel("mean activation norm"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: candidate-correct vs Class-A activation norms")
    save(fig, "candidate_correct_vs_classA_geometry")
    # 13 class geometry (fisher) + centroid distance
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(sx, [rE["geometry"][st]["fisher"]["mean"] for st in STAGES], color="tab:purple")
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=35, ha="right")
    ax.set_ylabel("between/within class variance (Fisher)")
    ax.set_title("E: Class-A value geometry by stage")
    save(fig, "activation_geometry")
    # 14 per-seed stage recoverability (D1x/mlp32)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for si, s in enumerate(SEEDS):
        ax.plot(sx, [rE["per_seed"][si]["caps"]["D1x"][st][REF_CAP]["hardtest"] for st in STAGES],
                marker="o", label=str(s), alpha=0.8)
    ax.set_xticks(sx); ax.set_xticklabels(STAGES, rotation=35, ha="right")
    ax.set_ylabel("hard-test accuracy (D1x/mlp32)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("E: per-seed stage recoverability")
    save(fig, "per_seed_stage_recoverability")
    # 15 adjacent-stage information drop
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    keys = list(rE["chain"]["D1x_drops"].keys())
    vals = [rE["chain"]["D1x_drops"][k] for k in keys]
    ax.bar(np.arange(len(keys)), vals, color=["tab:red" if (v is not None and v < 0) else "tab:green" for v in vals])
    ax.axhline(0, c="gray", lw=0.8)
    ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_ylabel("delta hard-test accuracy (next - prev), D1x")
    ax.set_title("E: adjacent-stage information drop")
    save(fig, "adjacent_information_drop")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment S analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_s_candidate_internal.npz")
    ap.add_argument("--output", default="experiments/experiment_s_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_s")
    args = ap.parse_args()
    data = load(args.npz)
    res = build(data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    d = res["primary_decision"]
    print(f"E: {d['verdict']} | best_internal={d['best_internal_stage']} "
          f"gain={d['best_internal_gain_over_Ct']} repro={d['reproducible_seeds']} "
          f"max_internal_heldout={d['max_internal_heldout_acc']}")
    print(f"  chain D1x: " + ", ".join(f"{k}={v:.3f}" for k, v in d["E_metrics"]["chain_d1x"].items()
                                       if v is not None))
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
