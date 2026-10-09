#!/usr/bin/env python3
"""Experiment R — analysis: capacity x distribution recoverability on the Class-A hard tail.

2-D design (analysis-only probes; frozen model untouched):
  Factor A capacity  : P0 frozen head, P1 linear ridge, P2/P3/P4/P5 MLP (16/32/64/128)
  Factor B distribution: D0 ordinary-train (1500), D0m size-matched ordinary (==|hard-train|),
                         D1 designated-hard-train (~144), D1x augmented i.i.d. hard pool
  Evaluation: the SAME untouched canonical-TEST hard tail (Class-A = candidate head wrong).

PRE-REGISTRATION (declared before aggregate inspection)
------------------------------------------------------
best_hardtest = max over bounded probes of mean hard-test accuracy (E seeds 42-46).
delta_distribution = mean over seeds of (hard-trained - size-matched-ordinary) accuracy on hard-test.
R-A distribution artifact : max delta_distribution >= +0.15, reproducible (>=4/5 seeds > 0), persists on held-out operands.
R-B capacity-limited      : best bounded probe hard-test >= 0.50 AND material capacity gain AND persists on held-out.
R-D memorization          : train accuracy high but large train->hard-test gap; capacity helps train not held-out.
R-C robust weak           : little capacity gain, little distribution gain, best < 0.50.
R-MIX                     : subtype or E/F divergence, or capacity/distribution conflict.
Bars are declared conventions; continuous effect sizes are reported alongside.

Reproduce:
    python experiments/analyze_experiment_r.py \
        --npz experiments/experiment_r_candidate_representation.npz \
        --output experiments/experiment_r_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_r
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

PRIMARY_MODEL = "E_candidate"
CONTROL_MODEL = "F_combined"
SEEDS = [42, 43, 44, 45, 46]
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]
N_MOD = 10
N_RAW = 19
CAPS = [("linear", None), ("mlp16", 16), ("mlp32", 32), ("mlp64", 64), ("mlp128", 128)]
REGIMES = ["D0", "D0m", "D1", "D1x"]
REF_CAP = "mlp32"                 # pre-declared "best-small-nonlinear" reference
HELD_OUT_PAIRS = {(a, b) for a in range(10) for b in range(10) if b in (0, 5)}  # 20 pairs, all targets
MLP_LR = 3e-3
MLP_WD = 1e-2
MLP_EPOCHS = 200
MLP_SEED = 20261007
# decision bars (declared conventions)
DIST_BAR = 0.15
RECOVER_BAR = 0.50
CAP_GAIN_BAR = 0.10


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


# ---------------------------------------------------------------------------
# probes (analysis-only)
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
    m = Xtr.mean(0); s = Xtr.std(0) + 1e-6
    Xn = torch.tensor((Xtr - m) / s, dtype=torch.float32); yn = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor((Xva - m) / s, dtype=torch.float32); yv = torch.tensor(yva, dtype=torch.long)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, nclass))
    wt = torch.tensor(wc, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    best = (-1.0, None)
    for ep in range(MLP_EPOCHS):
        net.train(); opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(Xn), yn, weight=wt)
        loss.backward(); opt.step()
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


def param_count(kind, hidden):
    if kind == "linear":
        return 212 * N_MOD + N_MOD
    return 212 * hidden + hidden + hidden * N_MOD + N_MOD


def macro_acc(pred, y, nclass=N_MOD):
    recs = []
    for c in range(nclass):
        m = y == c
        if m.any():
            recs.append(float((pred[m] == c).mean()))
    return float(np.mean(recs)) if recs else None


_TRAIN_DS = {"D0": "trainord", "D0m": "trainord", "D1": "trainhard", "D1x": "hardextra"}
_VAL_DS = {"D0": "valord", "D0m": "valord", "D1": "valhard", "D1x": "valhard"}


def fit_regime(z, m, seed, regime, kind, hidden, target="mod") -> Dict[str, Any]:
    def rows(ds):
        return np.nonzero(z[f"{m}__{ds}__seed"] == seed)[0]
    tds = _TRAIN_DS[regime]; vds = _VAL_DS[regime]
    tr = rows(tds); va = rows(vds)
    if regime == "D0m":
        n_hard = int((z[f"{m}__trainhard__seed"] == seed).sum())
        tr = tr[:n_hard]
    lab = "y_true" if target == "mod" else "raw_sum"
    nclass = N_MOD if target == "mod" else N_RAW
    Xtr = z[f"{m}__{tds}__C"][tr].astype(np.float64); ytr = z[f"{m}__{tds}__{lab}"][tr].astype(int)
    Xva = z[f"{m}__{vds}__C"][va].astype(np.float64); yva = z[f"{m}__{vds}__{lab}"][va].astype(int)
    wc = class_weights(ytr, nclass)
    if kind == "linear":
        pred, vacc, lam = ridge_fit(Xtr, ytr, wc, Xva, yva, nclass)
    else:
        pred, vacc = mlp_fit(Xtr, ytr, Xva, yva, hidden, wc, MLP_SEED + seed * 100 + hidden, nclass)
        lam = None
    return {"predict": pred, "train_acc": float((pred(Xtr).argmax(1) == ytr).mean()),
            "val_acc": vacc, "n_train": len(tr), "n_val": len(va), "lambda": lam}


def test_hard_rows(z, m, seed):
    sel = (z[f"{m}__test__seed"] == seed)
    ok = z[f"{m}__test__cand_ok"].astype(bool)
    return sel & (~ok), sel & ok


def matched_control(z, m, seed) -> np.ndarray:
    """Deterministic nearest-neighbour matching of candidate-correct test rows to Class-A rows on
    frozen-only metadata (wrap, repeated, t, |margin|, ||C||, ||dC||)."""
    hard, correct = test_hard_rows(z, m, seed)
    feats = ["wrap", "repeated", "t", "m_cand", "norm_C", "norm_dC"]
    def F(mask):
        cols = []
        for f in feats:
            v = z[f"{m}__test__{f}"][mask].astype(np.float64)
            if f == "m_cand":
                v = np.abs(v)
            cols.append(v)
        return np.stack(cols, 1)
    A = F(hard); B = F(correct)
    mu = np.concatenate([A, B], 0).mean(0); sd = np.concatenate([A, B], 0).std(0) + 1e-9
    A = (A - mu) / sd; B = (B - mu) / sd
    used = np.zeros(len(B), bool)
    pick = np.full(len(A), -1, int)
    order = np.argsort(A.sum(1))          # deterministic processing order
    for i in order:
        d = np.linalg.norm(B - A[i], axis=1)
        d[used] = np.inf
        j = int(d.argmin())
        if np.isfinite(d[j]):
            used[j] = True; pick[i] = j
    correct_idx = np.nonzero(correct)[0]
    return correct_idx[pick[pick >= 0]]


def evaluate(z, m, fitted, hard_mask, correct_mask, matched_mask) -> Dict[str, Any]:
    pred = fitted["predict"]
    def acc(mask):
        X = z[f"{m}__test__C"][mask].astype(np.float64)
        y = z[f"{m}__test__y_true"][mask].astype(int)
        return float((pred(X).argmax(1) == y).mean()) if mask.any() else None
    def macro(mask):
        X = z[f"{m}__test__C"][mask].astype(np.float64)
        y = z[f"{m}__test__y_true"][mask].astype(int)
        return macro_acc(pred(X).argmax(1), y) if mask.any() else None
    wrap = z[f"{m}__test__wrap"].astype(bool); rep = z[f"{m}__test__repeated"].astype(bool)
    oa = z[f"{m}__test__operand_a"].astype(int); ob = z[f"{m}__test__operand_b"].astype(int)
    held = np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oa, ob)])
    return {
        "hardtest": acc(hard_mask), "hardtest_macro": macro(hard_mask),
        "hard_wrap": acc(hard_mask & wrap), "hard_nowrap": acc(hard_mask & ~wrap),
        "hard_repeated": acc(hard_mask & rep), "hard_isolated": acc(hard_mask & ~rep),
        "hard_heldout_pairs": acc(hard_mask & held), "hard_seen_pairs": acc(hard_mask & ~held),
        "correct_all": acc(correct_mask), "correct_matched": acc(matched_mask),
    }


def analyze_seed(z, m, seed) -> Dict[str, Any]:
    hard_mask, correct_mask = test_hard_rows(z, m, seed)
    matched_idx = matched_control(z, m, seed)
    matched_mask = np.zeros(len(hard_mask), bool); matched_mask[matched_idx] = True
    res: Dict[str, Any] = {"seed": seed, "n_hard": int(hard_mask.sum()), "n_correct": int(correct_mask.sum()),
                           "n_matched": int(matched_mask.sum()), "caps": {}, "confusion": {}}
    for regime in REGIMES:
        res["caps"][regime] = {}
        for name, hidden in CAPS:
            f = fit_regime(z, m, seed, regime, "linear" if name == "linear" else "mlp", hidden)
            ev = evaluate(z, m, f, hard_mask, correct_mask, matched_mask)
            ev.update({"train_acc": f["train_acc"], "val_acc": f["val_acc"], "n_train": f["n_train"],
                       "params": param_count("linear" if name == "linear" else "mlp", hidden)})
            res["caps"][regime][name] = ev
            if name == REF_CAP and regime in ("D0", "D1x"):
                X = z[f"{m}__test__C"][hard_mask].astype(np.float64)
                y = z[f"{m}__test__y_true"][hard_mask].astype(int)
                p = f["predict"](X).argmax(1)
                Cm = np.zeros((N_MOD, N_MOD), int)
                for t, q in zip(p, y):
                    Cm[q, t] += 1
                res["confusion"][regime] = Cm.tolist()
    # raw-sum vs modular (linear + REF_CAP) under D0 and D1x
    res["arithmetic"] = {}
    for regime in ("D0", "D1x"):
        res["arithmetic"][regime] = {}
        for name, hidden in (("linear", None), (REF_CAP, dict(CAPS)[REF_CAP])):
            fr = fit_regime(z, m, seed, regime, "linear" if name == "linear" else "mlp", hidden, target="raw")
            X = z[f"{m}__test__C"][hard_mask].astype(np.float64)
            yr = z[f"{m}__test__raw_sum"][hard_mask].astype(int)
            res["arithmetic"][regime][name] = float((fr["predict"](X).argmax(1) == yr).mean())
    # held-out-operand probe (linear + REF_CAP) fit excluding held-out pairs, for D0 and D1x
    res["heldout_operand"] = {}
    for regime in ("D0", "D1x"):
        res["heldout_operand"][regime] = {}
        for name, hidden in (("linear", None), (REF_CAP, dict(CAPS)[REF_CAP])):
            res["heldout_operand"][regime][name] = heldout_fit(z, m, seed, name, hidden, regime, hard_mask)
    return res


def heldout_fit(z, m, seed, name, hidden, regime, hard_mask) -> Dict[str, Any]:
    """Fit on the regime's train data EXCLUDING held-out operand pairs; evaluate hard-test
    held-out vs seen operand pairs (generalization to unseen arithmetic combinations)."""
    ds = _TRAIN_DS[regime]; vds = _VAL_DS[regime]
    idx = np.nonzero(z[f"{m}__{ds}__seed"] == seed)[0]
    oa = z[f"{m}__{ds}__operand_a"][idx]; ob = z[f"{m}__{ds}__operand_b"][idx]
    held = np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oa, ob)])
    fit = idx[~held]
    Xtr = z[f"{m}__{ds}__C"][fit].astype(np.float64); ytr = z[f"{m}__{ds}__y_true"][fit].astype(int)
    va = np.nonzero(z[f"{m}__{vds}__seed"] == seed)[0]
    Xva = z[f"{m}__{vds}__C"][va].astype(np.float64); yva = z[f"{m}__{vds}__y_true"][va].astype(int)
    wc = class_weights(ytr, N_MOD)
    if name == "linear":
        pred, vacc, _ = ridge_fit(Xtr, ytr, wc, Xva, yva, N_MOD)
    else:
        pred, vacc = mlp_fit(Xtr, ytr, Xva, yva, hidden, wc, MLP_SEED + seed * 100 + hidden)
    X = z[f"{m}__test__C"][hard_mask].astype(np.float64); y = z[f"{m}__test__y_true"][hard_mask].astype(int)
    oa2 = z[f"{m}__test__operand_a"][hard_mask]; ob2 = z[f"{m}__test__operand_b"][hard_mask]
    h2 = np.array([(int(a), int(b)) in HELD_OUT_PAIRS for a, b in zip(oa2, ob2)])
    p = pred(X).argmax(1)
    return {"val_acc": vacc,
            "acc_all": float((p == y).mean()),
            "acc_heldout": float((p[h2] == y[h2]).mean()) if h2.any() else None,
            "acc_seen": float((p[~h2] == y[~h2]).mean()) if (~h2).any() else None,
            "n_heldout": int(h2.sum()), "n_fit": len(fit)}


def mean_sd(vals: List[Optional[float]]) -> Dict[str, Any]:
    a = np.array([v for v in vals if v is not None], float)
    return {"mean": float(a.mean()) if a.size else None,
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "per_seed": vals}


def paired(a: List[Optional[float]], b: List[Optional[float]]) -> Dict[str, Any]:
    x = np.array([v for v in a if v is not None], float); y = np.array([v for v in b if v is not None], float)
    d = x - y; sd = d.std(ddof=1) if d.size > 1 else 0.0
    out = {"a_mean": float(x.mean()), "b_mean": float(y.mean()), "mean_diff": float(d.mean()),
           "sign_positive": int((d > 0).sum()), "sign_negative": int((d < 0).sum()),
           "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan")}
    try:
        out["paired_t_p"] = float(st.ttest_rel(x, y).pvalue)
    except Exception:
        out["paired_t_p"] = None
    return out


def aggregate(per: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_seed": per, "caps": {}}
    for regime in REGIMES:
        out["caps"][regime] = {}
        for name, _ in CAPS:
            fields = ["hardtest", "hardtest_macro", "hard_wrap", "hard_nowrap", "hard_repeated",
                      "hard_isolated", "hard_heldout_pairs", "hard_seen_pairs", "correct_all",
                      "correct_matched", "train_acc", "val_acc", "n_train"]
            out["caps"][regime][name] = {f: mean_sd([p["caps"][regime][name][f] for p in per]) for f in fields}
            out["caps"][regime][name]["params"] = per[0]["caps"][regime][name]["params"]
    # distribution shift per capacity (D1 vs size-matched D0m)
    out["delta_distribution"] = {}
    for name, _ in CAPS:
        out["delta_distribution"][name] = paired([p["caps"]["D1"][name]["hardtest"] for p in per],
                                                 [p["caps"]["D0m"][name]["hardtest"] for p in per])
        out["delta_distribution"][name + "_vs_D0"] = paired([p["caps"]["D1x"][name]["hardtest"] for p in per],
                                                            [p["caps"]["D0"][name]["hardtest"] for p in per])
    # capacity effect per regime
    out["capacity_effect"] = {regime: paired([p["caps"][regime][REF_CAP]["hardtest"] for p in per],
                                             [p["caps"][regime]["linear"]["hardtest"] for p in per])
                              for regime in REGIMES}
    out["arithmetic"] = {regime: {name: mean_sd([p["arithmetic"][regime][name] for p in per])
                                  for name in ("linear", REF_CAP)} for regime in ("D0", "D1x")}
    out["heldout_operand"] = {regime: {name: {f: mean_sd([p["heldout_operand"][regime][name][f] for p in per])
                                                for f in ("val_acc", "acc_all", "acc_heldout", "acc_seen")}
                                        for name in ("linear", REF_CAP)} for regime in ("D0", "D1x")}
    out["confusion"] = {}
    for regime in ("D0", "D1x"):
        Cm = np.zeros((N_MOD, N_MOD), int)
        for p in per:
            if p.get("confusion", {}).get(regime):
                Cm += np.array(p["confusion"][regime], int)
        out["confusion"][regime] = Cm.tolist()
    return out


def decide(res: Dict[str, Any]) -> Dict[str, Any]:
    caps = res["caps"]
    best = {"acc": -1, "probe": None, "regime": None}
    for regime in REGIMES:
        for name, _ in CAPS:
            acc = caps[regime][name]["hardtest"]["mean"]
            if acc is not None and acc > best["acc"]:
                best = {"acc": acc, "probe": name, "regime": regime}
    # distribution effect: size-matched (D1 - D0m) and augmented (D1x - D0)
    dd_primary = paired([p["caps"]["D1"][REF_CAP]["hardtest"] for p in res["per_seed"]],
                        [p["caps"]["D0m"][REF_CAP]["hardtest"] for p in res["per_seed"]])
    dd_aug = paired([p["caps"]["D1x"][REF_CAP]["hardtest"] for p in res["per_seed"]],
                    [p["caps"]["D0"][REF_CAP]["hardtest"] for p in res["per_seed"]])
    dd_repro = max(dd_primary["sign_positive"], dd_primary["sign_negative"])
    cap_eff = res["capacity_effect"]
    cap_gain_best = max((v["mean_diff"] for v in cap_eff.values() if v["mean_diff"] is not None), default=0.0)
    train_max = max(caps[r][n]["train_acc"]["mean"] for r in REGIMES for n, _ in CAPS
                    if caps[r][n]["train_acc"]["mean"] is not None)
    ho = res["heldout_operand"]["D1x"][REF_CAP]
    ho_gap = None
    ho_heldout = ho["acc_heldout"]["mean"]
    if ho["acc_seen"]["mean"] is not None and ho_heldout is not None:
        ho_gap = ho["acc_seen"]["mean"] - ho_heldout
    # decision order (memorization is checked first: it invalidates apparent gains)
    if ho_gap is not None and ho_gap >= 0.20 and ho_heldout is not None and ho_heldout <= 0.25:
        verdict = "R-D_memorization_artifact"
    elif best["acc"] >= RECOVER_BAR and cap_gain_best >= CAP_GAIN_BAR and (ho_gap is not None and ho_gap < 0.15):
        verdict = "R-B_capacity_limited_recoverability"
    elif dd_primary["mean_diff"] is not None and dd_primary["mean_diff"] >= DIST_BAR and dd_repro >= 4 \
            and (ho_gap is not None and ho_gap < 0.15):
        verdict = "R-A_distribution_artifact"
    elif best["acc"] < RECOVER_BAR and cap_gain_best < CAP_GAIN_BAR and \
            (dd_primary["mean_diff"] is None or dd_primary["mean_diff"] < DIST_BAR):
        verdict = "R-C_robust_weak_recoverability"
    else:
        verdict = "R-MIX"
    return {"verdict": verdict, "best_probe": best, "delta_distribution_size_matched": dd_primary,
            "delta_distribution_augmented": dd_aug, "delta_distribution_reproducible_seeds": dd_repro,
            "capacity_gain_best": cap_gain_best, "train_acc_max": train_max, "heldout_operand_gap": ho_gap,
            "thresholds": {"dist_bar": DIST_BAR, "recover_bar": RECOVER_BAR, "cap_gain_bar": CAP_GAIN_BAR,
                           "heldout_gap_bar": 0.30},
            "rule_note": ("R-D (checked first): held-out-operand gap >= 0.30 -> apparent gains are "
                          "memorization, not recoverability. R-B: best bounded probe >= 0.50 with material "
                          "capacity gain and held-out operands preserved. R-A: size-matched distribution "
                          "shift >= 0.15, >=4/5 seeds, held-out operands preserved. R-C: no capacity gain, "
                          "no size-matched distribution shift, best < 0.50. Otherwise R-MIX. Bars are declared conventions.")}


def analyze_model(z, meta, m) -> Dict[str, Any]:
    per = [analyze_seed(z, m, s) for s in SEEDS]
    agg = aggregate(per)
    agg["decision"] = decide(agg)
    return agg


def make_figures(d, res, out_dir, prefix="experiment_r") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)
    rE = res[PRIMARY_MODEL]; rF = res[CONTROL_MODEL]
    names = [n for n, _ in CAPS]
    x = np.arange(len(names))
    # 1 capacity vs hard-test accuracy (all regimes, E)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for reg, col in zip(REGIMES, ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]):
        ax.plot(x, [rE["caps"][reg][n]["hardtest"]["mean"] for n in names], marker="o", label=reg, color=col)
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.9, label="0.50 bar")
    ax.axhline(0.10, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_xlabel("probe capacity"); ax.set_ylabel("hard-test accuracy")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("E: capacity vs hard-test accuracy")
    save(fig, "capacity_vs_hardtest")
    # 2 ordinary vs hard trained
    fig, ax = plt.subplots(figsize=(8, 4.3))
    w = 0.4
    ax.bar(x - w / 2, [rE["caps"]["D0"][n]["hardtest"]["mean"] for n in names], w, label="D0 ordinary", color="tab:blue")
    ax.bar(x + w / 2, [rE["caps"]["D1x"][n]["hardtest"]["mean"] for n in names], w, label="D1x augmented-hard", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("hard-test accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("E: ordinary- vs hard-trained")
    save(fig, "ordinary_vs_hard")
    # 3 capacity x distribution heatmap
    fig, ax = plt.subplots(figsize=(7, 4.6))
    M = np.array([[rE["caps"][reg][n]["hardtest"]["mean"] for n in names] for reg in REGIMES])
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(0.6, M.max()))
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_yticks(range(len(REGIMES))); ax.set_yticklabels(REGIMES)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax); ax.set_title("E: capacity x distribution (hard-test)")
    save(fig, "capacity_distribution_heatmap")
    # 4 train/val/hard-test curves (E, D1x)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for key, col in (("train_acc", "tab:green"), ("val_acc", "tab:olive"), ("hardtest", "tab:red")):
        ax.plot(x, [rE["caps"]["D1x"][n][key]["mean"] for n in names], marker="o", label=key, color=col)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E (D1x): train / val / hard-test curves")
    save(fig, "train_val_hardtest")
    # 5 hard vs matched candidate-correct
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.bar(x - 0.2, [rE["caps"]["D1x"][n]["hardtest"]["mean"] for n in names], 0.4, label="Class-A hard", color="tab:red")
    ax.bar(x + 0.2, [rE["caps"]["D1x"][n]["correct_matched"]["mean"] for n in names], 0.4, label="matched candidate-correct", color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E (D1x): Class-A vs matched candidate-correct")
    save(fig, "hard_vs_matched")
    # 6 held-out operand-pair generalization (D1x)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ho = rE["heldout_operand"]["D1x"]
    ax.bar([0, 1], [ho["linear"]["acc_heldout"]["mean"], ho[REF_CAP]["acc_heldout"]["mean"]], 0.4, color="tab:purple")
    ax.bar([0.4, 1.4], [ho["linear"]["acc_seen"]["mean"], ho[REF_CAP]["acc_seen"]["mean"]], 0.4, color="tab:gray")
    ax.set_xticks([0.2, 1.2]); ax.set_xticklabels(["linear", REF_CAP])
    ax.axhline(0.10, ls=":", c="gray", lw=0.8, label="chance")
    ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: held-out (purple) vs seen (gray) operand pairs")
    save(fig, "heldout_operand")
    # 7 wrap/no-wrap capacity curves
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(x, [rE["caps"]["D1x"][n]["hard_wrap"]["mean"] for n in names], marker="o", label="wrap")
    ax.plot(x, [rE["caps"]["D1x"][n]["hard_nowrap"]["mean"] for n in names], marker="s", label="no-wrap")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E (D1x): wrap vs no-wrap")
    save(fig, "wrap_nowrap")
    # 8 repeated vs isolated
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(x, [rE["caps"]["D1x"][n]["hard_repeated"]["mean"] for n in names], marker="o", label="repeated")
    ax.plot(x, [rE["caps"]["D1x"][n]["hard_isolated"]["mean"] for n in names], marker="s", label="isolated")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("hard-test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E (D1x): repeated vs isolated ADD")
    save(fig, "repeated_isolated")
    # 9 raw-sum vs modular
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.bar([0, 1], [rE["caps"]["D1x"]["linear"]["hardtest"]["mean"], rE["arithmetic"]["D1x"]["linear"]["mean"]], 0.4, color="tab:blue")
    ax.bar([0.4, 1.4], [rE["caps"]["D1x"][REF_CAP]["hardtest"]["mean"], rE["arithmetic"]["D1x"][REF_CAP]["mean"]], 0.4, color="tab:orange")
    ax.set_xticks([0.2, 1.2]); ax.set_xticklabels(["linear", REF_CAP])
    ax.set_ylabel("accuracy"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E (D1x): modular (blue/orange left) vs raw-sum (right)")
    save(fig, "rawsum_vs_modular")
    # 10 per-seed recoverability (D1x, REF_CAP)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.bar(np.arange(5), [p["caps"]["D1x"][REF_CAP]["hardtest"] for p in rE["per_seed"]], color="tab:orange")
    ax.bar(np.arange(5), [p["caps"]["D0"][REF_CAP]["hardtest"] for p in rE["per_seed"]], 0.35, color="tab:blue")
    ax.axhline(RECOVER_BAR, ls="--", c="k", lw=0.9)
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(SEEDS); ax.set_ylabel("hard-test accuracy")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("E: per-seed hard-test (orange D1x, blue D0)")
    save(fig, "per_seed")
    # 11 params vs held-out accuracy
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    for reg, col in zip(REGIMES, ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]):
        pp = [rE["caps"][reg][n]["params"] for n in names]
        ax.plot(pp, [rE["caps"][reg][n]["hardtest"]["mean"] for n in names], marker="o", label=reg, color=col)
    ax.set_xscale("log"); ax.set_xlabel("probe parameter count"); ax.set_ylabel("hard-test accuracy")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("E: parameter count vs held-out accuracy")
    save(fig, "params_vs_heldout")
    # 12 distribution-shift effect
    fig, ax = plt.subplots(figsize=(8, 4.3))
    dd = rE["delta_distribution"]
    dvals = [dd[n]["mean_diff"] for n in names]
    errs = [dd[n]["cohens_dz"] for n in names]
    ax.bar(x, dvals, color=["tab:red" if d > 0 else "tab:blue" for d in dvals])
    ax.axhline(DIST_BAR, ls="--", c="k", lw=0.9, label=f"+{DIST_BAR} bar")
    ax.axhline(0, c="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("D1 - D0m hard-test accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("E: distribution-shift effect (size-matched)")
    save(fig, "distribution_shift")
    # 13 confusion for strongest legitimate probe (D1x / REF_CAP on hard-test)
    Cm = np.array(rE["confusion"].get("D1x", np.zeros((N_MOD, N_MOD), int)), float)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(Cm, cmap="Blues")
    ax.set_xlabel("predicted (D1x/%s)" % REF_CAP); ax.set_ylabel("true value")
    ax.set_xticks(range(N_MOD)); ax.set_yticks(range(N_MOD))
    fig.colorbar(im, ax=ax)
    bp = rE["decision"]["best_probe"]
    ax.set_title(f"E: hard-test confusion (D1x/{REF_CAP}; best {bp['regime']}/{bp['probe']} {bp['acc']:.2f})")
    save(fig, "confusion_best")
    # 14 candidate-correct sanity control
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for reg, col in zip(REGIMES, ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]):
        ax.plot(x, [rE["caps"][reg][n]["correct_all"]["mean"] for n in names], marker="o", label=reg, color=col)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("candidate-correct accuracy")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("E: candidate-correct sanity control")
    save(fig, "candidate_correct_control")
    return paths


def build(data: Dict[str, Any]) -> Dict[str, Any]:
    z = data["z"]; meta = data["meta"]
    res: Dict[str, Any] = {"metadata": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS,
                                        "checkpoints": meta["models"], "splits": meta["n"],
                                        "thresholds": {"dist_bar": DIST_BAR, "recover_bar": RECOVER_BAR,
                                                       "cap_gain_bar": CAP_GAIN_BAR},
                                        "caps": [(n, h) for n, h in CAPS], "regimes": REGIMES,
                                        "statistical_discipline": ("Five frozen E checkpoints are the independent "
                                                                   "units; F is a control. Probes are analysis-only, "
                                                                   "fit on the regime's train, tuned on its val, "
                                                                   "evaluated once on the untouched test hard tail.")}}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        res[m] = analyze_model(z, meta, m)
    res["E_vs_F"] = {n: paired([p["caps"]["D1x"][n]["hardtest"] for p in res[PRIMARY_MODEL]["per_seed"]],
                               [p["caps"]["D1x"][n]["hardtest"] for p in res[CONTROL_MODEL]["per_seed"]])
                     for n, _ in CAPS}
    res["primary_decision"] = res[PRIMARY_MODEL]["decision"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment R analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_r_candidate_representation.npz")
    ap.add_argument("--output", default="experiments/experiment_r_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_r")
    args = ap.parse_args()
    data = load(args.npz)
    res = build(data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        d = res[m]["decision"]
        print(f"{m}: {d['verdict']} | best={d['best_probe']['regime']}/{d['best_probe']['probe']} "
              f"acc={d['best_probe']['acc']:.3f} | dd_sizematched={d['delta_distribution_size_matched']['mean_diff']:.3f} "
              f"dd_aug={d['delta_distribution_augmented']['mean_diff']:.3f} "
              f"cap_gain={d['capacity_gain_best']:.3f} ho_gap={d['heldout_operand_gap']:.3f}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
