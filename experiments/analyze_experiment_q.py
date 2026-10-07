#!/usr/bin/env python3
"""Experiment Q — analysis: candidate-generation origin / recoverability (measurement only).

Reads experiments/experiment_q_candidate_origin.npz and measures how much of the true written value
is recoverable from the frozen candidate vector C_t on the transitions where the frozen candidate
head is wrong, using analysis-only probes fit on the disjoint TRAIN split, tuned on VAL, evaluated
once on TEST. Never enters model inference; no model parameter changes.

PRE-REGISTRATION (declared before aggregate inspection)
------------------------------------------------------
populations (test ADD): Class-A = candidate head wrong; control = candidate head correct.
recoverability(pop, probe) = (probe_acc - frozen_acc) / (1 - frozen_acc).
Q-A linearly recoverable : mean linear recoverability on Class-A >= 0.50 AND wrap & repeated same direction
Q-B nonlinear-only       : linear recoverability < 0.50 AND (nonlinear - linear) >= 0.20
Q-C weak recoverability  : linear < 0.50 AND (nonlinear - linear) < 0.20 AND nonlinear < 0.50
Q-MIX                    : wrap / no-wrap / repeated disagree materially (max-min >= 0.15)
Inference: five E checkpoints are the independent units; F is a control.

Reproduce:
    python experiments/analyze_experiment_q.py \
        --npz experiments/experiment_q_candidate_origin.npz \
        --output experiments/experiment_q_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_q
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
RECOVER_BAR = 0.50
NONLINEAR_BAR = 0.20
MIX_BAR = 0.15
MLP_HIDDEN = 32
MLP_WD = 1e-2
MLP_LR = 3e-3
MLP_EPOCHS = 300
MLP_SEED = 20261007


def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    return {"z": z, "meta": meta}


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def ridge_probe(Xtr, ytr, Xva, yva, nclass) -> Tuple[Any, float, float]:
    """Closed-form ridge multinomial probe with lambda chosen on validation. Deterministic."""
    Xb = np.c_[Xtr, np.ones(len(Xtr))]
    Y = np.eye(nclass)[ytr]
    G = Xb.T @ Xb
    XtY = Xb.T @ Y
    Xvb = np.c_[Xva, np.ones(len(Xva))]
    best = (None, -1.0, None)
    for lam in LAMBDAS:
        W = np.linalg.solve(G + lam * np.eye(Xb.shape[1]), XtY)
        acc = float(((Xvb @ W).argmax(1) == yva).mean())
        if acc > best[1]:
            best = (W, acc, lam)
    W = best[0]
    def predict(X):
        return np.c_[X, np.ones(len(X))] @ W
    return predict, best[1], best[2]


class _MLP:
    def __init__(self, hidden, seed):
        import torch
        torch.manual_seed(seed)
        self.t = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(212, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, N_MOD))
    def fit(self, Xtr, ytr, Xva, yva):
        t = self.t
        m = Xtr.mean(0); s = Xtr.std(0) + 1e-6
        self.mu, self.sd = m, s
        Xn = t.tensor((Xtr - m) / s, dtype=t.float32); yn = t.tensor(ytr, dtype=t.long)
        Xv = t.tensor((Xva - m) / s, dtype=t.float32); yv = t.tensor(yva, dtype=t.long)
        opt = t.optim.Adam(self.net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
        best = (-1.0, None)
        for ep in range(MLP_EPOCHS):
            self.net.train(); opt.zero_grad()
            loss = t.nn.functional.cross_entropy(self.net(Xn), yn); loss.backward(); opt.step()
            if ep % 10 == 9:
                self.net.eval()
                with t.no_grad():
                    v = float((self.net(Xv).argmax(1) == yv).float().mean())
                if v > best[0]:
                    best = (v, {k: val.clone() for k, val in self.net.state_dict().items()})
        if best[1] is not None:
            self.net.load_state_dict(best[1])
        self.net.eval()
        return best[0]
    def predict(self, X):
        t = self.t
        with t.no_grad():
            return self.net(t.tensor((X - self.mu) / self.sd, dtype=t.float32)).numpy()


def acc_of(logits, y):
    return float((logits.argmax(1) == y).mean()) if len(y) else None


# ---------------------------------------------------------------------------
# per-seed analysis
# ---------------------------------------------------------------------------

def analyze_seed(z, m: str, seed: int) -> Dict[str, Any]:
    s = str(seed)
    def G(split, key):
        return z[f"{m}__{split}__{key}"]
    seed_col = G("train", "seed")
    tr = seed_col == seed
    tr = np.nonzero(tr)[0]
    va = np.nonzero(G("val", "seed") == seed)[0]
    te = np.nonzero(G("test", "seed") == seed)[0]
    if len(tr) == 0 or len(te) == 0:
        return {}
    def sub(split, arr, idx):
        return G(split, arr)[idx]
    Xtr = sub("train", "C", tr).astype(np.float64); ytr = sub("train", "y_true", tr).astype(int)
    rawtr = sub("train", "raw_sum", tr).astype(int)
    Xva = sub("val", "C", va).astype(np.float64); yva = sub("val", "y_true", va).astype(int)
    Xte = sub("test", "C", te).astype(np.float64); yte = sub("test", "y_true", te).astype(int)
    rawte = sub("test", "raw_sum", te).astype(int)
    ok = sub("test", "cand_ok", te).astype(bool)
    wrap = sub("test", "wrap", te).astype(bool)
    rep = sub("test", "repeated", te).astype(bool)
    # sanity: seed column present
    if len(tr) == 0:
        return {}

    lin, lin_val, lin_lam = ridge_probe(Xtr, ytr, Xva, yva, N_MOD)
    val_raw = sub("val", "raw_sum", va).astype(int)
    lin_raw, raw_val, raw_lam = ridge_probe(Xtr, rawtr, Xva, val_raw, N_RAW)
    mlp = _MLP(MLP_HIDDEN, MLP_SEED + seed)
    mlp_val = mlp.fit(Xtr, ytr, Xva, yva)

    def npred(pred, mask):
        return pred(Xte[mask])
    def mnpred(mask):
        return mlp.predict(Xte[mask])
    def rawnpred(pred, mask):
        return pred(Xte[mask])

    def block(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        frozen = float((sub("test", "cand_ok", te)[mask] > 0.5).mean())
        lin_a = acc_of(npred(lin, mask), yte[mask])
        raw_a = acc_of(rawnpred(lin_raw, mask), rawte[mask])
        mlp_a = acc_of(mnpred(mask), yte[mask])
        finite = frozen < 1 - 1e-6
        rec_lin = (lin_a - frozen) / (1 - frozen) if finite else None
        rec_mlp = (mlp_a - frozen) / (1 - frozen) if finite else None
        return {"n": n, "frozen_acc": frozen, "linear_acc": lin_a, "nonlinear_acc": mlp_a,
                "rawsum_acc": raw_a, "recover_linear": rec_lin, "recover_nonlinear": rec_mlp,
                "gap_mlp_minus_lin": mlp_a - lin_a}
    cw = ~ok
    out = {
        "seed": seed, "lin_val_acc": lin_val, "lin_lambda": lin_lam, "mlp_val_acc": mlp_val,
        "classA": block(cw), "control_candok": block(ok),
        "classA_wrap": block(cw & wrap), "classA_nowrap": block(cw & ~wrap),
        "classA_repeated": block(cw & rep), "classA_isolated": block(cw & ~rep),
        "control_wrap": block(ok & wrap), "control_nowrap": block(ok & ~wrap),
        "lin_acc_all_test": acc_of(npred(lin, np.ones(len(yte), bool)), yte),
        "mlp_acc_all_test": acc_of(mnpred(np.ones(len(yte), bool)), yte),
        "raw_acc_all_test": acc_of(rawnpred(lin_raw, np.ones(len(rawte), bool)), rawte),
        "frozen_acc_all_test": float(ok.mean()),
        "confusion_classA": None,
    }
    # confusion matrix on Class-A (linear probe)
    if cw.any():
        p = npred(lin, cw).argmax(1)
        Cm = np.zeros((N_MOD, N_MOD), int)
        for t, q in zip(p, yte[cw]):
            Cm[q, t] += 1
        out["confusion_classA"] = Cm.tolist()
    return out


def mean_sd(vals: List[Optional[float]]) -> Dict[str, Any]:
    a = np.array([v for v in vals if v is not None], float)
    return {"mean": float(a.mean()) if a.size else None,
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "per_seed": vals, "n": int(a.size)}


def paired(a: List[Optional[float]], b: List[Optional[float]]) -> Dict[str, Any]:
    x = np.array([v for v in a if v is not None], float)
    y = np.array([v for v in b if v is not None], float)
    d = x - y; sd = d.std(ddof=1) if d.size > 1 else 0.0
    out = {"E_mean": float(x.mean()), "F_mean": float(y.mean()), "mean_diff": float(d.mean()),
           "sign_positive": int((d > 0).sum()), "sign_negative": int((d < 0).sum()),
           "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"), "n": int(d.size)}
    try:
        out["paired_t_p"] = float(st.ttest_rel(x, y).pvalue)
    except Exception:
        out["paired_t_p"] = None
    return out


def analyze_model(z, meta, m: str) -> Dict[str, Any]:
    per = [analyze_seed(z, m, s) for s in SEEDS]
    per = [p for p in per if p]
    res: Dict[str, Any] = {"per_seed": per}
    keys = ["classA", "control_candok", "classA_wrap", "classA_nowrap", "classA_repeated",
            "classA_isolated", "control_wrap", "control_nowrap"]
    res["summary"] = {}
    for k in keys:
        res["summary"][k] = {f: mean_sd([p[k].get(f) for p in per]) for f in
                             ("n", "frozen_acc", "linear_acc", "nonlinear_acc", "rawsum_acc",
                              "recover_linear", "recover_nonlinear", "gap_mlp_minus_lin")}
    for k in ("lin_val_acc", "mlp_val_acc", "lin_acc_all_test", "mlp_acc_all_test", "raw_acc_all_test", "frozen_acc_all_test"):
        res[k] = mean_sd([p.get(k) for p in per])
    # confusion summed over seeds
    Cm = np.zeros((N_MOD, N_MOD), int)
    for p in per:
        if p.get("confusion_classA"):
            Cm += np.array(p["confusion_classA"], int)
    res["confusion_classA_total"] = Cm.tolist()
    res["decision"] = decide(res)
    return res


def decide(res: Dict[str, Any]) -> Dict[str, Any]:
    summ = res["summary"]
    rec_lin = summ["classA"]["recover_linear"]["mean"]
    rec_mlp = summ["classA"]["recover_nonlinear"]["mean"]
    lin = summ["classA"]["linear_acc"]["mean"]
    mlp = summ["classA"]["nonlinear_acc"]["mean"]
    wrap_lin = summ["classA_wrap"]["linear_acc"]["mean"]
    nowrap_lin = summ["classA_nowrap"]["linear_acc"]["mean"]
    rep_lin = summ["classA_repeated"]["linear_acc"]["mean"]
    spread = max([v for v in [wrap_lin, nowrap_lin, rep_lin] if v is not None]) - \
             min([v for v in [wrap_lin, nowrap_lin, rep_lin] if v is not None])
    if rec_lin is not None and rec_lin >= RECOVER_BAR and (wrap_lin or 0) >= 0.30 and (rep_lin or 0) >= 0.30:
        verdict = "Q-A_linearly_recoverable"
    elif rec_lin is not None and rec_lin < RECOVER_BAR and (mlp - lin) >= NONLINEAR_BAR:
        verdict = "Q-B_nonlinear_only"
    elif spread >= MIX_BAR and (rec_lin is None or rec_lin < RECOVER_BAR):
        verdict = "Q-MIX"
    else:
        verdict = "Q-C_weak_recoverability"
    return {"verdict": verdict, "recover_linear": rec_lin, "recover_nonlinear": rec_mlp,
            "linear_acc": lin, "nonlinear_acc": mlp, "wrap_lin": wrap_lin, "nowrap_lin": nowrap_lin,
            "repeated_lin": rep_lin, "nonlinear_minus_linear": (mlp - lin) if (mlp is not None and lin is not None) else None,
            "spread_wrap_nowrap_rep": spread,
            "thresholds": {"recover_bar": RECOVER_BAR, "nonlinear_bar": NONLINEAR_BAR, "mix_bar": MIX_BAR},
            "rule_note": ("Q-A: linear recoverability >= 0.50 with wrap/repeated same direction. "
                          "Q-B: linear < 0.50 and nonlinear-linear >= 0.20. Q-C: both weak. "
                          "Q-MIX: wrap/no-wrap/repeated disagree (spread >= 0.15). Bars are declared conventions.")}


# ---------------------------------------------------------------------------
# historical objective cross-check (read only)
# ---------------------------------------------------------------------------

def historical(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out: Dict[str, Any] = {}
    summ = d.get("summary", {})
    for v in ("A_terminal", "E_candidate", "F_combined"):
        if v in summ:
            s = summ[v]
            out[v] = {
                "candidate_written_mean": s.get("candidate_written_mean", {}).get("mean"),
                "candidate_full_mean": s.get("candidate_full_mean", {}).get("mean"),
                "terminal_em": s.get("terminal_em", {}).get("mean"),
                "T20": s.get("horizons", {}).get("20", {}).get("mean"),
                "add_transition_stats": s.get("add_transition_stats") or
                {k: v2.get("candidate_dest_ok") for k, v2 in (d.get("summary", {}).get(v, {}).get("add_transition_stats", {}) or {}).items()},
            }
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d, res, out_dir, prefix="experiment_q") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    rE = res[PRIMARY_MODEL]; rF = res[CONTROL_MODEL]
    sE, sF = rE["summary"], rF["summary"]

    # 1 frozen head vs linear probe by population
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = ["Class-A\n(cand wrong)", "control\n(cand correct)", "Class-A wrap", "Class-A no-wrap",
              "Class-A repeated", "Class-A isolated"]
    keys = ["classA", "control_candok", "classA_wrap", "classA_nowrap", "classA_repeated", "classA_isolated"]
    x = np.arange(len(keys)); w = 0.27
    ax.bar(x - w, [sE[k]["frozen_acc"]["mean"] for k in keys], w, label="frozen head", color="tab:gray")
    ax.bar(x, [sE[k]["linear_acc"]["mean"] for k in keys], w, label="linear probe", color="tab:blue")
    ax.bar(x + w, [sE[k]["nonlinear_acc"]["mean"] for k in keys], w, label="nonlinear probe", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("E: frozen head vs analysis-only probes")
    save(fig, "head_vs_linear")

    # 2 wrap vs no-wrap
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(2); w = 0.3
    ax.bar(x - w, [sE["classA_wrap"]["linear_acc"]["mean"], sE["classA_nowrap"]["linear_acc"]["mean"]], w, label="E linear", color="tab:blue")
    ax.bar(x, [sE["classA_wrap"]["nonlinear_acc"]["mean"], sE["classA_nowrap"]["nonlinear_acc"]["mean"]], w, label="E nonlinear", color="tab:orange")
    ax.bar(x + w, [sF["classA_wrap"]["linear_acc"]["mean"], sF["classA_nowrap"]["linear_acc"]["mean"]], w, label="F linear", color="tab:cyan")
    ax.set_xticks(x); ax.set_xticklabels(["wrap", "no-wrap"]); ax.set_ylabel("Class-A recoverability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("E/F: wrap vs no-wrap Class-A recoverability")
    save(fig, "wrap_nowrap")

    # 3 repeated vs isolated
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(2); w = 0.3
    ax.bar(x - w, [sE["classA_repeated"]["linear_acc"]["mean"], sE["classA_isolated"]["linear_acc"]["mean"]], w, label="E linear", color="tab:blue")
    ax.bar(x, [sE["classA_repeated"]["nonlinear_acc"]["mean"], sE["classA_isolated"]["nonlinear_acc"]["mean"]], w, label="E nonlinear", color="tab:orange")
    ax.bar(x + w, [sF["classA_repeated"]["linear_acc"]["mean"], sF["classA_isolated"]["linear_acc"]["mean"]], w, label="F linear", color="tab:cyan")
    ax.set_xticks(x); ax.set_xticklabels(["repeated ADD", "isolated ADD"]); ax.set_ylabel("Class-A recoverability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("E/F: repeated vs isolated Class-A recoverability")
    save(fig, "repeated")

    # 4 raw-sum vs modular
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(2)
    ax.bar(x - 0.2, [sE["classA"]["linear_acc"]["mean"], sE["classA"]["rawsum_acc"]["mean"]], 0.4,
           color=["tab:blue", "tab:purple"])
    ax.set_xticks(x); ax.set_xticklabels(["modular target (10-way)", "raw sum (19-way)"])
    ax.set_ylabel("Class-A probe accuracy"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: raw-sum vs modular-target linear probe on Class-A")
    save(fig, "rawsum_vs_modular")

    # 5 candidate margin distributions
    z = d["z"]; te = z["E_candidate__test__seed"] == 42
    ok = z["E_candidate__test__cand_ok"][te].astype(bool)
    mc = z["E_candidate__test__m_cand"][te]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.hist(mc[ok], bins=40, range=(-5, 20), density=True, alpha=0.6, label="cand correct", color="tab:green")
    ax.hist(mc[~ok], bins=40, range=(-5, 20), density=True, alpha=0.6, label="cand wrong (Class-A)", color="tab:red")
    ax.set_xlabel("candidate margin (vs strongest competitor at S_prev)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E (seed 42): candidate margin distribution")
    save(fig, "cand_margin")

    # 6 norms
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    nc = z["E_candidate__test__norm_C"][te]
    ax.hist(nc[ok], bins=40, density=True, alpha=0.6, label="cand correct", color="tab:green")
    ax.hist(nc[~ok], bins=40, density=True, alpha=0.6, label="cand wrong", color="tab:red")
    ax.set_xlabel("||C_t||"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("E (seed 42): candidate norm")
    save(fig, "norms")

    # 7 candidate value geometry (per true-value probe accuracy on Class-A, E)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    cm = np.array(rE["confusion_classA_total"], float)
    per_val = [cm[v, v] / max(cm[v].sum(), 1) for v in range(10)]
    ax.bar(range(10), per_val, color="tab:blue")
    ax.set_xlabel("true written value"); ax.set_ylabel("linear-probe recall (Class-A)")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("E: Class-A linear-probe recall by value")
    save(fig, "value_geometry")

    # 8 confusion matrix
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted (linear probe)"); ax.set_ylabel("true value")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax)
    ax.set_title("E: Class-A linear-probe confusion (all seeds)")
    save(fig, "confusion")

    # 9 nonlinear vs linear
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(2); w = 0.38
    ax.bar(x - w / 2, [sE["classA"]["linear_acc"]["mean"], sE["classA"]["nonlinear_acc"]["mean"]], w, label="E", color="tab:blue")
    ax.bar(x + w / 2, [sF["classA"]["linear_acc"]["mean"], sF["classA"]["nonlinear_acc"]["mean"]], w, label="F", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(["linear probe", "nonlinear probe"]); ax.set_ylabel("Class-A accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("E/F: nonlinear vs linear probe on Class-A")
    save(fig, "nonlinear_vs_linear")

    # 10 per-seed recoverability
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(5); w = 0.35
    ax.bar(x - w / 2, [p["classA"]["recover_linear"] for p in rE["per_seed"]], w, label="E linear recover", color="tab:blue")
    ax.bar(x + w / 2, [p["classA"]["recover_nonlinear"] for p in rE["per_seed"]], w, label="E nonlinear recover", color="tab:orange")
    ax.axhline(RECOVER_BAR, color="k", ls="--", lw=0.9, label=f"Q-A bar {RECOVER_BAR}")
    ax.set_xticks(x); ax.set_xticklabels(SEEDS); ax.set_ylabel("Class-A recoverability fraction")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("E: per-seed Class-A recoverability")
    save(fig, "per_seed")

    # 11 probe gap by operation
    fig, ax = plt.subplots(figsize=(8, 4.2))
    keys2 = ["classA", "classA_wrap", "classA_nowrap", "classA_repeated", "classA_isolated"]
    labs = ["all A", "wrap", "no-wrap", "repeated", "isolated"]
    x = np.arange(len(keys2))
    ax.bar(x - 0.2, [sE[k]["linear_acc"]["mean"] for k in keys2], 0.4, label="E linear", color="tab:blue")
    ax.bar(x + 0.2, [sF[k]["linear_acc"]["mean"] for k in keys2], 0.4, label="F linear", color="tab:orange")
    ax.axhline(0.10, color="k", ls=":", lw=0.8, label="chance (10-way)")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("Class-A linear recoverability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("Recoverability gap by operation")
    save(fig, "gap_by_op")

    # 12 historical objective comparison
    hist = res.get("historical", {})
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    if hist:
        labs = [k for k in ("A_terminal", "E_candidate") if k in hist]
        vals = [hist[k]["candidate_written_mean"] for k in labs]
        ax.bar(range(len(labs)), vals, color=["tab:gray", "tab:green"][:len(labs)])
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
        ax.set_ylabel("historical candidate written-register accuracy")
        ax.set_title("Historical objective cross-check (generator-repair, read-only)")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "historical")
    return paths


def build(data: Dict[str, Any]) -> Dict[str, Any]:
    z = data["z"]; meta = data["meta"]
    res: Dict[str, Any] = {"metadata": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": SEEDS,
                                        "checkpoints": meta["models"], "thresholds":
                                        {"recover_bar": RECOVER_BAR, "nonlinear_bar": NONLINEAR_BAR, "mix_bar": MIX_BAR},
                                        "statistical_discipline": ("Five frozen E checkpoints are independent "
                                                                   "units; F is a control. Probes are analysis-only, "
                                                                   "fit on TRAIN, tuned on VAL, tested once on TEST; "
                                                                   "they never enter model inference.")}}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        res[m] = analyze_model(z, meta, m)
    res["E_vs_F"] = {k: paired([p["classA"].get(k) for p in res[PRIMARY_MODEL]["per_seed"]],
                               [p["classA"].get(k) for p in res[CONTROL_MODEL]["per_seed"]])
                     for k in ("linear_acc", "nonlinear_acc", "recover_linear", "rawsum_acc")}
    res["historical"] = historical("experiments/generator_repair_results.json")
    res["primary_decision"] = res[PRIMARY_MODEL]["decision"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment Q analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_q_candidate_origin.npz")
    ap.add_argument("--output", default="experiments/experiment_q_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_q")
    args = ap.parse_args()

    data = load(args.npz)
    res = build(data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        s = res[m]["summary"]; d = res[m]["decision"]
        print(f"{m}: {d['verdict']} | Class-A frozen={s['classA']['frozen_acc']['mean']:.3f} "
              f"linear={s['classA']['linear_acc']['mean']:.3f} nonlinear={s['classA']['nonlinear_acc']['mean']:.3f} "
              f"recover_lin={d['recover_linear']:.3f} recover_mlp={d['recover_nonlinear']:.3f}")
    paths = make_figures(data, res, args.fig_dir, prefix=args.fig_prefix)
    print(f"Figures: {len(paths)}")


if __name__ == "__main__":
    main()
