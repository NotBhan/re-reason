#!/usr/bin/env python3
"""Experiment N — Parity-Origin Localization (analysis, measurement only).

Reads experiments/experiment_n_parity_origin.npz and localizes the same-parity signature M found
to (a) the frozen readout's weight geometry or (b) the state<->label alignment.

PRE-REGISTRATION (declared before inspecting aggregate results)
--------------------------------------------------------------
W_effect(seed)  = mean_{register, pairs} [cosine(normalized W_u, normalized W_v) | same parity]
                  - mean[... | different parity]                          (W-only geometry)
W permutation null: within each register, randomly permute the 10 value labels (destroying the
                  value<->parity relationship) and recompute; N_PERM permutations per seed.
W_strong        = mean W_effect >= 0.05 AND at least 4/5 seeds have permutation p <= 0.05.
obs             = same-parity rate of the committed decode (P(dec(S_commit) % 2 == y % 2)).
null_pair       = pairing-break null: permute y relative to the fixed decode within seed x dest
                  stratum (readout-only) / equivalently permute the decode relative to y
                  (state-permutation); preserving all state vectors and the readout.
readout_fraction   = (null_pair - 0.5) / (obs - 0.5)      # share of the excess over chance
alignment_fraction = 1 - readout_fraction                  # share from the state<->label pairing

Decision:
  N-A readout-dominant  : W_strong AND readout_fraction >= 0.50
  N-C joint             : W_strong AND readout_fraction >= 0.30 AND alignment_fraction >= 0.30
  N-B state-dominant    : alignment_fraction >= 0.60
  N-D inconclusive      : otherwise
The 0.05 / 4-of-5 / 0.50 / 0.30 / 0.60 bars are declared conventions, not data-tuned cutoffs; all
raw quantities are reported so the verdict can be judged independently. The readout value-label
permutation is reported as a necessary-condition control (it is degenerate for parity: mapping the
decoded label back is a no-op, and not mapping back trivially randomizes parity to ~0.5).

Reproduce:
    python experiments/analyze_experiment_n.py \
        --npz experiments/experiment_n_parity_origin.npz \
        --output experiments/experiment_n_results.json \
        --fig-dir docs/reports/figures --fig-prefix experiment_n
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
MODULUS = 10

W_EFFECT_BAR = 0.05
W_SEED_P_BAR = 0.05
W_SEED_COUNT = 4
ALIGNMENT_FRACTION_BAR = 0.60
READOUT_DOMINANT_BAR = 0.50
JOINT_BAR = 0.30
N_PERM = 2000
PERM_SEED = 20261007

GROUP_NAMES = {0: "offpath_failure", 1: "candidate_correct_success", 2: "nonoffpath_failure"}


class NTable:
    def __init__(self, scalars: np.ndarray, cols: List[str]) -> None:
        self.scalars = scalars.astype(np.float64)
        self.index = {c: i for i, c in enumerate(cols)}

    def __getitem__(self, name: str) -> np.ndarray:
        return self.scalars[:, self.index[name]]

    def __len__(self) -> int:
        return self.scalars.shape[0]


def load_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"][0]))
    cols = meta["n_cols"]
    out: Dict[str, Any] = {"meta": meta, "cols": cols}
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        out[m] = {
            "table": NTable(z[f"{m}__scalars"], cols),
            "primary_row": z[f"{m}__primary_row"].astype(np.int64),
            "primary_group": z[f"{m}__primary_group"].astype(np.int64),
            "W": z[f"{m}__W"].astype(np.float64), "b": z[f"{m}__b"].astype(np.float64),
            "fail_row": z[f"{m}__fail_row"].astype(np.int64),
            "fail_S_prev": z[f"{m}__fail_S_prev"].astype(np.float64),
            "fail_C": z[f"{m}__fail_C"].astype(np.float64),
            "fail_S_commit": z[f"{m}__fail_S_commit"].astype(np.float64),
        }
    return out


# ---------------------------------------------------------------------------
# W-only geometry
# ---------------------------------------------------------------------------

def w_same_diff_cosine(Wsd: np.ndarray) -> Tuple[float, float]:
    """Wsd: (R,V,D) for one seed. Same/diff-parity mean pairwise cosine (within register, off-diag)."""
    R, V, D = Wsd.shape
    iu = np.triu_indices(V, 1)
    p1, p2 = iu
    same_pairs = (p1 % 2) == (p2 % 2)
    s_same, s_diff = [], []
    for d in range(R):
        X = Wsd[d]
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        C = Xn @ Xn.T
        s_same.append(C[iu][same_pairs].mean()); s_diff.append(C[iu][~same_pairs].mean())
    return float(np.mean(s_same)), float(np.mean(s_diff))


def w_effect_with_parity(Wsd: np.ndarray, parity: np.ndarray) -> float:
    R, V, D = Wsd.shape
    iu = np.triu_indices(V, 1); p1, p2 = iu
    same_pairs = parity[p1] == parity[p2]
    vals = []
    for d in range(R):
        X = Wsd[d]
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        C = Xn @ Xn.T
        vals.append(C[iu][same_pairs].mean() - C[iu][~same_pairs].mean())
    return float(np.mean(vals))


def w_geometry(model: str, d: Dict[str, Any]) -> Dict[str, Any]:
    W = d["W"]
    parity = np.arange(MODULUS) % 2
    rng = np.random.default_rng(PERM_SEED)
    per_seed = []
    for s in range(len(SEEDS)):
        obs = w_effect_with_parity(W[s], parity)
        null = np.array([w_effect_with_parity(W[s], rng.permutation(parity)) for _ in range(N_PERM)])
        cs, cd = w_same_diff_cosine(W[s])
        # difference-vector geometry: ||W_v - W_u|| same vs diff parity
        dnorm = np.zeros((MODULUS, MODULUS))
        for u in range(MODULUS):
            for v in range(MODULUS):
                dnorm[u, v] = np.mean([np.linalg.norm(W[s, r][u] - W[s, r][v]) for r in range(W.shape[1])])
        iu = np.triu_indices(MODULUS, 1); same = parity[iu[0]] == parity[iu[1]]
        # parity axis separation
        seps = []
        for r in range(W.shape[1]):
            X = W[s, r]; Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
            ev = np.arange(MODULUS) % 2 == 0
            seps.append(float(np.linalg.norm(Xn[ev].mean(0) - Xn[~ev].mean(0))))
        per_seed.append({
            "seed": SEEDS[s], "effect": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p": float((null >= obs).mean()),
            "cos_same": cs, "cos_diff": cd, "cos_effect": cs - cd,
            "diffvec_same": float(dnorm[iu][same].mean()), "diffvec_diff": float(dnorm[iu][~same].mean()),
            "axis_separation_mean": float(np.mean(seps)), "axis_separation_min": float(np.min(seps)),
        })
    effects = np.array([p["effect"] for p in per_seed])
    out = {
        "per_seed": per_seed,
        "mean_effect": float(effects.mean()), "sd_effect": float(effects.std(ddof=1)),
        "mean_cos_same": float(np.mean([p["cos_same"] for p in per_seed])),
        "mean_cos_diff": float(np.mean([p["cos_diff"] for p in per_seed])),
        "n_seeds_sig": int(sum(p["p"] <= W_SEED_P_BAR for p in per_seed)),
        "mean_axis_separation": float(np.mean([p["axis_separation_mean"] for p in per_seed])),
    }
    # cross-register control: parity geometry for same-register vs cross-register value pairs
    iu_cross = [(r1, u, r2, v) for r1 in range(W.shape[1]) for r2 in range(W.shape[1])
                for u in range(MODULUS) for v in range(MODULUS)]
    sr_same, sr_diff, cr_same, cr_diff = [], [], [], []
    for s in range(len(SEEDS)):
        Xn = W[s] / np.clip(np.linalg.norm(W[s], axis=2, keepdims=True), 1e-12, None)  # (R,V,D)
        for r1, u, r2, v in iu_cross:
            if r1 == r2 and u == v:
                continue
            c = float(Xn[r1, u] @ Xn[r2, v])
            same = (u % 2) == (v % 2)
            if r1 == r2:
                (sr_same if same else sr_diff).append(c)
            else:
                (cr_same if same else cr_diff).append(c)
    out["cross_register"] = {
        "same_parity_same_reg": float(np.mean(sr_same)), "diff_parity_same_reg": float(np.mean(sr_diff)),
        "same_parity_cross_reg": float(np.mean(cr_same)), "diff_parity_cross_reg": float(np.mean(cr_diff)),
        "within_register_effect": float(np.mean(sr_same) - np.mean(sr_diff)),
        "cross_register_effect": float(np.mean(cr_same) - np.mean(cr_diff)),
    }
    # bias analysis
    b = d["b"]
    bias_same, bias_diff = [], []
    for s in range(len(SEEDS)):
        for r in range(b.shape[1]):
            bv = b[s, r]
            for u in range(MODULUS):
                for v in range(u + 1, MODULUS):
                    (bias_same if (u % 2) == (v % 2) else bias_diff).append(abs(bv[u] - bv[v]))
    out["bias"] = {"mean_abs_bias_same_parity": float(np.mean(bias_same)),
                   "mean_abs_bias_diff_parity": float(np.mean(bias_diff))}
    return out


# ---------------------------------------------------------------------------
# state projections + pairing-break controls
# ---------------------------------------------------------------------------

def pair_break_controls(model: str, d: Dict[str, Any]) -> Dict[str, Any]:
    t: NTable = d["table"]
    off = d["primary_row"][d["primary_group"] == 0]
    y = t["y_true"][off].astype(int); dec = t["dec_commit"][off].astype(int)
    dest = t["dest"][off].astype(int); seed = t["seed"][off].astype(int)
    rng = np.random.default_rng(PERM_SEED + 1)
    obs = float((dec % 2 == y % 2).mean())
    # pairing-break null (readout-only = state-permutation): permute labels within seed x dest
    yp_same, dp_same = [], []
    for _ in range(N_PERM):
        yp = y.copy()
        for key in set(zip(seed.tolist(), dest.tolist())):
            idx = np.where((seed == key[0]) & (dest == key[1]))[0]
            yp[idx] = rng.permutation(y[idx])
        yp_same.append(float((dec % 2 == yp % 2).mean()))
        dp = dec.copy()
        for key in set(zip(seed.tolist(), dest.tolist())):
            idx = np.where((seed == key[0]) & (dest == key[1]))[0]
            dp[idx] = rng.permutation(dec[idx])
        dp_same.append(float((dp % 2 == y % 2).mean()))
    yp_same = np.array(yp_same); dp_same = np.array(dp_same)
    # readout value-label permutation (degenerate control), on off-path raw vectors
    fo = d["primary_group"] == 0
    S_com = d["fail_S_commit"][np.isin(d["fail_row"], off)]
    # rebuild order: fail rows ascending; off-path subset
    fail_row = d["fail_row"]; fail_group = None
    W = d["W"]; b = d["b"]; seed_order = {s: i for i, s in enumerate(SEEDS)}
    off_set = set(off.tolist())
    sel = np.array([(fr in off_set) for fr in fail_row])
    S_off = d["fail_S_commit"][sel]
    row_off = fail_row[sel]
    order = np.argsort(row_off)
    S_off = S_off[order]
    # decode mapped-back vs unmapped under value-label permutation
    si = seed[0:0]  # placeholder
    mapped_same, unmapped_same = [], []
    for _ in range(200):
        d2 = np.zeros(len(S_off)); d3 = np.zeros(len(S_off))
        for j in range(len(S_off)):
            ss = seed_order[int(seed[j])]; dd = dest[j]
            perm = rng.permutation(MODULUS)
            Wp = W[ss, dd][perm]; bp = b[ss, dd][perm]
            vp = int(np.argmax(Wp @ S_off[j] + bp))
            d2[j] = perm[vp]          # map back to original value identity
            d3[j] = vp                # permuted index (parity randomized)
        mapped_same.append(float((d2.astype(int) % 2 == y % 2).mean()))
        unmapped_same.append(float((d3.astype(int) % 2 == y % 2).mean()))
    return {
        "obs_same_parity": obs,
        "readout_only_null_mean": float(yp_same.mean()), "readout_only_null_sd": float(yp_same.std()),
        "state_perm_null_mean": float(dp_same.mean()), "state_perm_null_sd": float(dp_same.std()),
        "gap_readout_only": obs - float(yp_same.mean()),
        "gap_state_perm": obs - float(dp_same.mean()),
        "readout_perm_mapped_back_mean": float(np.mean(mapped_same)),
        "readout_perm_unmapped_mean": float(np.mean(unmapped_same)),
        "chance": 0.5,
    }


def per_seed_pair_break(model: str, d: Dict[str, Any]) -> List[Dict[str, Any]]:
    t: NTable = d["table"]
    off = d["primary_row"][d["primary_group"] == 0]
    y = t["y_true"][off].astype(int); dec = t["dec_commit"][off].astype(int)
    dest = t["dest"][off].astype(int); seed = t["seed"][off].astype(int)
    rng = np.random.default_rng(PERM_SEED + 2)
    out = []
    for s in SEEDS:
        m = seed == s
        if m.sum() == 0:
            continue
        ym, dm, dem = y[m], dest[m], dec[m]
        obs = float((dem % 2 == ym % 2).mean())
        nulls = []
        for _ in range(500):
            yp = ym.copy()
            for dv in np.unique(dm):
                idx = np.where(dm == dv)[0]; yp[idx] = rng.permutation(ym[idx])
            nulls.append(float((dem % 2 == yp % 2).mean()))
        nulls = np.array(nulls)
        out.append({"seed": s, "n": int(m.sum()), "obs": obs, "null_mean": float(nulls.mean()),
                    "gap": obs - float(nulls.mean()), "p_null_ge_obs": float((nulls >= obs).mean())})
    return out


# ---------------------------------------------------------------------------
# value-pair decision geometry
# ---------------------------------------------------------------------------

def value_pair_geometry(model: str, d: Dict[str, Any]) -> Dict[str, Any]:
    t: NTable = d["table"]
    off = d["primary_row"][d["primary_group"] == 0]
    W = d["W"]; b = d["b"]; seed_order = {s: i for i, s in enumerate(SEEDS)}
    dest = t["dest"][off].astype(int); y = t["y_true"][off].astype(int)
    wrong = t["comm_pred"][off].astype(int); seed = t["seed"][off].astype(int)
    # off-path raw vectors aligned to `off` (ascending fail rows intersected with off)
    fail_row = d["fail_row"]; off_set = set(off.tolist())
    sel = np.array([(fr in off_set) for fr in fail_row])
    row_off = fail_row[sel]; order = np.argsort(row_off)
    S_prev = d["fail_S_prev"][sel][order]; C = d["fail_C"][sel][order]; Sc = d["fail_S_commit"][sel][order]
    same = (wrong % 2) == (y % 2)
    res: Dict[str, Any] = {"n_same_parity_wrong": int(same.sum()), "n_diff_parity_wrong": int((~same).sum())}
    for label, mask in (("same_parity", same), ("diff_parity", ~same), ("all", np.ones_like(same))):
        if mask.sum() == 0:
            continue
        nrm, cos, bdiff = [], [], []
        hp = {"prev": [], "cand": [], "commit": []}
        for j in np.where(mask)[0]:
            si = seed_order[int(seed[j])]; dd = dest[j]
            wy = W[si, dd, y[j]]; ww = W[si, dd, wrong[j]]
            dw = wy - ww
            nrm.append(float(np.linalg.norm(dw)))
            cos.append(float(wy @ ww / (np.linalg.norm(wy) * np.linalg.norm(ww) + 1e-12)))
            bdiff.append(float(b[si, dd, y[j]] - b[si, dd, wrong[j]]))
            nd = float(np.linalg.norm(dw))
            for st, V in (("prev", S_prev[j]), ("cand", C[j]), ("commit", Sc[j])):
                m = float(dw @ V + (b[si, dd, y[j]] - b[si, dd, wrong[j]]))
                hp[st].append(m / (nd + 1e-12))
        res[label] = {
            "n": int(mask.sum()), "dw_norm_mean": float(np.mean(nrm)),
            "wy_wrong_cosine_mean": float(np.mean(cos)), "bias_diff_mean": float(np.mean(bdiff)),
            "hyper_dist_prev_median": float(np.median(hp["prev"])),
            "hyper_dist_cand_median": float(np.median(hp["cand"])),
            "hyper_dist_commit_median": float(np.median(hp["commit"])),
        }
    return res


# ---------------------------------------------------------------------------
# arithmetic-conditioned + group summaries
# ---------------------------------------------------------------------------

def group_summary(model: str, d: Dict[str, Any]) -> Dict[str, Any]:
    t: NTable = d["table"]
    row = d["primary_row"]; grp = d["primary_group"]
    res: Dict[str, Any] = {}
    for g in (0, 1, 2):
        sel = row[grp == g]
        dec = t["dec_commit"][sel].astype(int); y = t["y_true"][sel].astype(int)
        res[GROUP_NAMES[g]] = {
            "n": int(len(sel)),
            "same_parity_rate": float((dec % 2 == y % 2).mean()) if len(sel) else None,
            "par_proj_prev_median": float(np.median(t["par_proj_prev"][sel])) if len(sel) else None,
            "par_proj_cand_median": float(np.median(t["par_proj_cand"][sel])) if len(sel) else None,
            "par_proj_commit_median": float(np.median(t["par_proj_commit"][sel])) if len(sel) else None,
            "par_delta_commit_median": float(np.median(t["par_delta_commit"][sel])) if len(sel) else None,
            "par_ratio_commit_median": float(np.median(t["par_ratio_commit"][sel])) if len(sel) else None,
            "par_ratio_cand_median": float(np.median(t["par_ratio_cand"][sel])) if len(sel) else None,
        }
    # arithmetic-conditioned (off-path)
    off = row[grp == 0]
    pd = t["par_delta_commit"][off]
    y = t["y_true"][off].astype(int); oa = t["operand_a"][off].astype(int)
    ob = t["operand_b"][off].astype(int); wrap = t["wrap"][off] > 0.5
    res["arithmetic_conditioned"] = {
        "par_delta_vs_target_parity_even": float(pd[y % 2 == 0].mean()),
        "par_delta_vs_target_parity_odd": float(pd[y % 2 == 1].mean()),
        "par_delta_vs_operand_parity_even": float(pd[(ob % MODULUS) % 2 == 0].mean()),
        "par_delta_vs_operand_parity_odd": float(pd[(ob % MODULUS) % 2 == 1].mean()),
        "par_delta_wrap": float(pd[wrap].mean()) if wrap.any() else None,
        "par_delta_nowrap": float(pd[~wrap].mean()) if (~wrap).any() else None,
        "target_parity_corr": float(np.corrcoef(pd, y % 2)[0, 1]),
    }
    return res


def paired_seed(a: List[Dict[str, Any]], b: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    av = {x["seed"]: x[key] for x in a}; bv = {x["seed"]: x[key] for x in b}
    seeds = sorted(set(av) & set(bv))
    x = np.array([av[s] for s in seeds], float); y = np.array([bv[s] for s in seeds], float)
    dd = x - y; sd = dd.std(ddof=1)
    out = {"E_mean": float(x.mean()), "F_mean": float(y.mean()), "mean_diff": float(dd.mean()),
           "sign_positive": int((dd > 0).sum()), "sign_negative": int((dd < 0).sum()),
           "cohens_dz": float(dd.mean() / sd) if sd > 0 else float("nan"), "n": len(seeds)}
    try:
        out["paired_t_p"] = float(st.ttest_rel(x, y).pvalue)
    except Exception:
        out["paired_t_p"] = None
    return out


def classify(nres: Dict[str, Any]) -> Dict[str, Any]:
    wg = nres["W_geometry"]
    pc = nres["pair_break"]
    obs = pc["obs_same_parity"]
    null_pair = pc["readout_only_null_mean"]
    w_strong = bool(wg["mean_effect"] >= W_EFFECT_BAR and wg["n_seeds_sig"] >= W_SEED_COUNT)
    denom = max(obs - 0.5, 1e-9)
    readout_fraction = float((null_pair - 0.5) / denom)
    alignment_fraction = float(1.0 - readout_fraction)
    if w_strong and readout_fraction >= READOUT_DOMINANT_BAR:
        verdict = "N-A_readout_dominant"
    elif w_strong and readout_fraction >= JOINT_BAR and alignment_fraction >= JOINT_BAR:
        verdict = "N-C_joint"
    elif alignment_fraction >= ALIGNMENT_FRACTION_BAR:
        verdict = "N-B_state_dominant"
    else:
        verdict = "N-D_inconclusive"
    return {
        "verdict": verdict, "w_strong": w_strong,
        "mean_W_effect": wg["mean_effect"], "n_seeds_W_sig": wg["n_seeds_sig"],
        "obs_same_parity": obs, "null_pair_mean": null_pair,
        "readout_fraction": readout_fraction, "alignment_fraction": alignment_fraction,
        "thresholds": {"w_effect_bar": W_EFFECT_BAR, "w_seed_p_bar": W_SEED_P_BAR,
                       "w_seed_count": W_SEED_COUNT, "alignment_fraction_bar": ALIGNMENT_FRACTION_BAR,
                       "readout_dominant_bar": READOUT_DOMINANT_BAR, "joint_bar": JOINT_BAR},
        "rule_note": ("N-A: W_strong and readout_fraction>=0.50. N-C: W_strong and "
                      "readout_fraction>=0.30 and alignment_fraction>=0.30. N-B: "
                      "alignment_fraction>=0.60. N-D otherwise. Bars are declared conventions, "
                      "not data-tuned cutoffs."),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(d: Dict[str, Any], res: Dict[str, Any], out_dir: str, prefix="experiment_n") -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    def save(fig, name):
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    t: NTable = d[PRIMARY_MODEL]["table"]
    row = d[PRIMARY_MODEL]["primary_row"]; grp = d[PRIMARY_MODEL]["primary_group"]

    # 1. W geometry: same/diff cosine effect per seed
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    wg = res[PRIMARY_MODEL]["W_geometry"]; ps = wg["per_seed"]
    axes[0].bar(range(5), [p["cos_same"] for p in ps], 0.4, label="same parity", color="tab:blue")
    axes[0].bar([x + 0.4 for x in range(5)], [p["cos_diff"] for p in ps], 0.4, label="diff parity", color="tab:red")
    axes[0].set_xticks([x + 0.2 for x in range(5)]); axes[0].set_xticklabels([p["seed"] for p in ps])
    axes[0].set_ylabel("mean pairwise cosine"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_title("W value-vector cosine by parity", fontsize=9)
    axes[1].bar(range(5), [p["effect"] for p in ps], color="tab:green")
    axes[1].axhline(W_EFFECT_BAR, color="k", ls="--", lw=0.8, label=f"bar {W_EFFECT_BAR}")
    axes[1].set_xticks(range(5)); axes[1].set_xticklabels([p["seed"] for p in ps])
    axes[1].set_ylabel("same - diff effect"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_title("W parity effect per seed", fontsize=9)
    fig.suptitle("E: readout weight geometry (W alone)")
    save(fig, "w_geometry")

    # 2. W permutation null
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    rng = np.random.default_rng(PERM_SEED)
    W = d[PRIMARY_MODEL]["W"]; parity = np.arange(MODULUS) % 2
    null_all = np.array([w_effect_with_parity(W[0], rng.permutation(parity)) for _ in range(500)])
    ax.hist(null_all, bins=40, color="tab:gray", alpha=0.7, label="permutation null (seed-1)")
    ax.axvline(wg["mean_effect"], color="tab:red", lw=2, label=f"mean observed {wg['mean_effect']:.3f}")
    ax.set_xlabel("W same-diff parity cosine effect"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("E: W parity effect vs value-label permutation null")
    save(fig, "w_null")

    # 3. parity axis projections
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for s in range(5):
        X = W[s] / np.clip(np.linalg.norm(W[s], axis=2, keepdims=True), 1e-12, None)
        projs = []
        for r in range(W.shape[1]):
            ev = np.arange(MODULUS) % 2 == 0
            mu = X[r][ev].mean(0) - X[r][~ev].mean(0); p = mu / np.clip(np.linalg.norm(mu), 1e-12, None)
            projs.extend(list(X[r] @ p))
        ax.hist(projs, bins=40, alpha=0.4, label=f"seed {SEEDS[s]}")
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("value weight projection onto parity axis")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("E: value-weight projections onto W-derived parity axis")
    save(fig, "parity_axis")

    # 4. cross-register control
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    cr = wg["cross_register"]
    ax.bar(range(4), [cr["same_parity_same_reg"], cr["diff_parity_same_reg"],
                      cr["same_parity_cross_reg"], cr["diff_parity_cross_reg"]],
           color=["tab:blue", "tab:red", "tab:blue", "tab:red"], alpha=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["same-par/same-reg", "diff-par/same-reg", "same-par/cross-reg", "diff-par/cross-reg"], fontsize=7)
    ax.set_ylabel("mean cosine"); ax.grid(alpha=0.3, axis="y"); ax.set_title("E: cross-register parity control")
    save(fig, "cross_register")

    # 5. state projections by group
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for g, name, col in ((0, "off-path", "tab:red"), (1, "success", "tab:green"), (2, "non-off", "tab:gray")):
        sel = row[grp == g]
        ax.hist(t["par_proj_commit"][sel], bins=40, range=(-20, 20), density=True, alpha=0.5, label=name, color=col)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("parity-axis projection of S_commit")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("E: state projection onto W-derived parity axis")
    save(fig, "state_proj")

    # 6. parity displacement by group
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for g, name, col in ((0, "off-path", "tab:red"), (1, "success", "tab:green"), (2, "non-off", "tab:gray")):
        sel = row[grp == g]
        axes[0].hist(t["par_delta_commit"][sel], bins=40, range=(-20, 20), density=True, alpha=0.5, label=name, color=col)
        axes[1].hist(t["par_ratio_commit"][sel], bins=40, range=(0, 0.6), density=True, alpha=0.5, label=name, color=col)
    axes[0].set_xlabel("signed parity displacement"); axes[1].set_xlabel("|parity displacement| / ||Delta||")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("E: committed displacement along the parity axis by group")
    save(fig, "par_delta")

    # 7. readout-only / state-perm null
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    pc = res[PRIMARY_MODEL]["pair_break"]
    bars = ["chance", "pairing-break null", "observed"]
    vals = [0.5, pc["readout_only_null_mean"], pc["obs_same_parity"]]
    ax.bar(bars, vals, color=["tab:gray", "tab:orange", "tab:red"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("same-parity rate"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E off-path: observed same-parity vs pairing-break null")
    save(fig, "readout_null")

    # 8. controls summary
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    labels = ["observed", "readout-only\nnull (label perm)", "state-perm\nnull", "readout-perm\n(mapped back)", "readout-perm\n(unmapped)"]
    vals = [pc["obs_same_parity"], pc["readout_only_null_mean"], pc["state_perm_null_mean"],
            pc["readout_perm_mapped_back_mean"], pc["readout_perm_unmapped_mean"]]
    ax.bar(range(len(labels)), vals, color=["tab:red", "tab:orange", "tab:orange", "tab:purple", "tab:gray"])
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("same-parity rate")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("E: localization controls")
    save(fig, "controls")

    # 9. value-pair geometry
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    vp = res[PRIMARY_MODEL]["value_pair"]
    labels = ["same-parity wrong", "diff-parity wrong"]
    keys = ["same_parity", "diff_parity"]
    xs = np.arange(3); w = 0.38
    ax.bar(xs - w / 2, [vp[keys[0]]["hyper_dist_prev_median"], vp[keys[0]]["hyper_dist_cand_median"],
                        vp[keys[0]]["hyper_dist_commit_median"]], w, label=labels[0], color="tab:blue")
    ax.bar(xs + w / 2, [vp[keys[1]]["hyper_dist_prev_median"], vp[keys[1]]["hyper_dist_cand_median"],
                        vp[keys[1]]["hyper_dist_commit_median"]], w, label=labels[1], color="tab:red")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(["S_prev", "C", "S_commit"])
    ax.set_ylabel("normalized hyperplane distance (y vs wrong)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E: value-pair decision geometry")
    save(fig, "value_pair_geometry")

    # 10. arithmetic-conditioned
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ac = res[PRIMARY_MODEL]["arithmetic_conditioned"]
    labels = ["target even", "target odd", "operand even", "operand odd", "wrap", "no-wrap"]
    vals = [ac["par_delta_vs_target_parity_even"], ac["par_delta_vs_target_parity_odd"],
            ac["par_delta_vs_operand_parity_even"], ac["par_delta_vs_operand_parity_odd"],
            ac["par_delta_wrap"], ac["par_delta_nowrap"]]
    ax.bar(range(len(labels)), vals, color="tab:cyan")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=20); ax.set_ylabel("mean parity displacement")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("E off-path: parity displacement vs arithmetic metadata")
    save(fig, "arithmetic")

    # 11. control groups
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    gs = res[PRIMARY_MODEL]["groups"]
    names = ["offpath_failure", "candidate_correct_success", "nonoffpath_failure"]
    ax.bar(range(3), [gs[n]["same_parity_rate"] for n in names], color=["tab:red", "tab:green", "tab:gray"])
    for i, n in enumerate(names):
        v = gs[n]["same_parity_rate"]; ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_xticks(range(3))
    ax.set_xticklabels(["off-path", "success", "non-off"], fontsize=8); ax.set_ylabel("same-parity rate")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("E: same-parity rate by control group")
    save(fig, "control_groups")

    # 12. E vs F
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    e = res[PRIMARY_MODEL]; f = res[CONTROL_MODEL]
    labels = ["obs same-parity", "pairing-break null", "alignment frac (x0.1)", "W effect"]
    vals_e = [e["pair_break"]["obs_same_parity"], e["pair_break"]["readout_only_null_mean"],
              e["classification"]["alignment_fraction"] * 0.1, e["W_geometry"]["mean_effect"]]
    vals_f = [f["pair_break"]["obs_same_parity"], f["pair_break"]["readout_only_null_mean"],
              f["classification"]["alignment_fraction"] * 0.1, f["W_geometry"]["mean_effect"]]
    xs = np.arange(len(labels)); w = 0.38
    ax.bar(xs - w / 2, vals_e, w, label="E", color="tab:blue")
    ax.bar(xs + w / 2, vals_f, w, label="F", color="tab:orange")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("E vs F: parity-origin localization")
    save(fig, "E_vs_F")

    return paths


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build(d: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": SEEDS, "primary_model": PRIMARY_MODEL, "control_model": CONTROL_MODEL,
            "checkpoints": d["meta"]["models"], "m_alignment": d["meta"]["m_alignment"],
            "offpath_definition": d["meta"]["offpath_definition"], "parity_axis": d["meta"]["parity_axis"],
            "thresholds": {"w_effect_bar": W_EFFECT_BAR, "w_seed_p_bar": W_SEED_P_BAR,
                           "w_seed_count": W_SEED_COUNT, "alignment_fraction_bar": ALIGNMENT_FRACTION_BAR,
                           "readout_dominant_bar": READOUT_DOMINANT_BAR, "joint_bar": JOINT_BAR,
                           "n_perm": N_PERM},
            "statistical_discipline": ("Five frozen E checkpoints are independent units; F is a "
                                       "secondary control (not pooled). Pairing-break permutation "
                                       "controls are analysis-only counterfactuals on frozen objects, "
                                       "not interventions."),
        },
    }
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        wg = w_geometry(m, d[m])
        pc = pair_break_controls(m, d[m])
        gs = group_summary(m, d[m])
        vp = value_pair_geometry(m, d[m])
        res[m] = {"W_geometry": wg, "pair_break": pc, "groups": gs, "value_pair": vp,
                  "arithmetic_conditioned": gs["arithmetic_conditioned"],
                  "per_seed_pair_break": per_seed_pair_break(m, d[m])}
        res[m]["classification"] = classify(res[m])
    res["E_vs_F_paired"] = {
        "per_seed_gap": paired_seed(res[PRIMARY_MODEL]["per_seed_pair_break"],
                                    res[CONTROL_MODEL]["per_seed_pair_break"], "gap"),
        "per_seed_obs": paired_seed(res[PRIMARY_MODEL]["per_seed_pair_break"],
                                    res[CONTROL_MODEL]["per_seed_pair_break"], "obs"),
    }
    res["primary_decision"] = res[PRIMARY_MODEL]["classification"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment N analysis (measurement only)")
    ap.add_argument("--npz", default="experiments/experiment_n_parity_origin.npz")
    ap.add_argument("--output", default="experiments/experiment_n_results.json")
    ap.add_argument("--fig-dir", default="docs/reports/figures")
    ap.add_argument("--fig-prefix", default="experiment_n")
    args = ap.parse_args()

    d = load_npz(args.npz)
    res = build(d)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")
    for m in (PRIMARY_MODEL, CONTROL_MODEL):
        c = res[m]["classification"]
        print(f"{m}: {c['verdict']}  W_effect={c['mean_W_effect']:.4f} sig={c['n_seeds_W_sig']}/5 "
              f"obs={c['obs_same_parity']:.3f} null={c['null_pair_mean']:.3f} "
              f"readout_frac={c['readout_fraction']:.3f} align_frac={c['alignment_fraction']:.3f}")
    paths = make_figures(d, res, args.fig_dir, prefix=args.fig_prefix)
    print("Figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
