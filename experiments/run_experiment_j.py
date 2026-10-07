#!/usr/bin/env python3
"""Experiment J — Controlled Inference-Time Commit-Rule Intervention (interventional).

Purpose
-------
Test causally whether the ADD candidate->commit residual localized by Experiments G/H/I is
recoverable by changing ONLY the inference-time element-wise commit rule, on the frozen
checkpoints, with no training and no weight modification.

The actual commit is
        S_t = (1 - f_t) . S_(t-1) + f_t . C_t ,     f_t in R^212.
We intervene on the FULL 212-dimensional gate vector (never on the scalar mean gate) with a
pre-registered monotone coordinate-wise scaling, and run a COMPLETE recursive rollout: the
modified committed state S'_t is used at step t+1 for both gate and candidate computation
(the frozen core computes `f_t = sigmoid(w_f_ih e_t + w_f_hh S_(t-1))`,
`C_t = tanh(cand_mlp([f_t . S_(t-1), e_t]))`), so candidate generation tracks the modified
trajectory exactly as the unmodified architecture would.

PRE-REGISTRATION (fixed before any result is inspected)
-------------------------------------------------------
Primary family (coordinate-preserving monotone scaling):
        f'_i = clip(gamma * f_i, 0, 1),     gamma in GAMMA_GRID = {0.75,0.90,1.00,1.10,1.25,1.50,2.00}
gamma = 1.00 is the unmodified frozen baseline (identity) and must reproduce the baseline exactly.
Pre-registered neutral control (one): coordinate-scrambled gate, deterministically permuted per
(base_seed, seed) and mean-matched per row to the coordinate-preserving scaled gate at the same
gamma. Pre-registered optional secondary monotone family (one): f'_i = 1 - (1-f_i)^gamma.
No gamma value is added or removed after seeing results. No readout/decision-margin/label
information enters the intervention itself; those quantities are used only in analysis.

Decision rule (pre-registered; see analyze_experiment_j.py): classification A/B/C/D is based on
whether a beneficial SIGN-CONSISTENT (>=4/5 seeds) and CONVENTIONALLY LARGE (|Cohen's dz| >= 0.8)
paired sequence-level effect exists at T=10 or T=20, together with net candidate->commit rescue
and the absence of a large operation-level regression; transition-only effects without a
sequence-level effect classify as B (partial). No pp threshold is tuned on the data.

Measurement only: frozen checkpoints, no optimizer, no training, torch.no_grad() throughout.
Checkpoint SHA-256 values are verified before and after the run.

Outputs
-------
    experiments/experiment_j_records.json    per (model, seed, family, gamma) aggregates + arrays-free stats
Reproduce
---------
    python experiments/run_experiment_j.py --out experiments/experiment_j_records.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402
import run_experiment_i as ri  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

# ---------------------------------------------------------------------------
# PRE-REGISTRATION (do not edit after running)
# ---------------------------------------------------------------------------

GAMMA_GRID = [0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]   # fixed; gamma=1.00 is baseline
BASELINE_GAMMA = 1.00
PRIMARY_FAMILY = "scale"                 # f' = clip(gamma f, 0, 1)
CONTROL_FAMILY = "scramble"              # mean-matched coordinate permutation of f
SECONDARY_FAMILY = "one_minus"           # f' = 1 - (1 - f)^gamma  (pre-registered optional)
SCRAMBLE_SEED = 20261006                 # deterministic permutation seed
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20]
MODEL_PLAN = {
    "E_candidate": [PRIMARY_FAMILY, CONTROL_FAMILY, SECONDARY_FAMILY],
    "F_combined": [PRIMARY_FAMILY],
}
MEAN_MATCH_ITERS = 60
MEAN_MATCH_HI = 50.0

PREREGISTRATION = {
    "gamma_grid": GAMMA_GRID, "baseline_gamma": BASELINE_GAMMA,
    "primary_family": PRIMARY_FAMILY, "control_family": CONTROL_FAMILY,
    "secondary_family": SECONDARY_FAMILY, "scramble_seed": SCRAMBLE_SEED,
    "horizons": HORIZONS, "model_plan": MODEL_PLAN,
    "decision_rule": ("Beneficial sequence-level effect at T=10 or T=20 requires sign "
                      "consistency >=4/5 seeds and |Cohen's dz| >= 0.8 (conventional large "
                      "band); net candidate->commit rescue (rescue>regression); no large "
                      "operation-level regression. Transition-only effect => partial (B)."),
    "note": ("Intervention on the full 212-dim gate vector only; recursive rollout; no readout/"
             "label information enters the intervention. gamma values fixed before results."),
}


# ---------------------------------------------------------------------------
# Commit-rule families (operate on the full gate vector; no analysis quantities)
# ---------------------------------------------------------------------------

def _mean_matched_scale(f_perm: torch.Tensor, target_mean: torch.Tensor) -> torch.Tensor:
    """Per-row scalar gamma_s with mean(clip(gamma_s * f_perm,0,1)) ~= target_mean (bisection)."""
    lo = torch.zeros_like(target_mean)
    hi = torch.full_like(target_mean, MEAN_MATCH_HI)
    for _ in range(MEAN_MATCH_ITERS):
        mid = 0.5 * (lo + hi)
        m = (mid[:, None] * f_perm).clamp(0.0, 1.0).mean(dim=1)
        too_small = m < target_mean
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)
    return 0.5 * (lo + hi)


def make_commit_fn(family: str, gamma: float, perm: Optional[torch.Tensor] = None) -> Callable:
    """Returns f -> f' for the pre-registered commit-rule families. All outputs in [0,1]."""
    if family == "scale":
        if gamma <= 0:
            raise ValueError("scale requires gamma > 0")
        return lambda f: (gamma * f).clamp(0.0, 1.0)
    if family == "one_minus":
        if gamma <= 0:
            raise ValueError("one_minus requires gamma > 0")
        return lambda f: 1.0 - (1.0 - f).clamp(0.0, 1.0).pow(gamma)
    if family == "scramble":
        if perm is None:
            raise ValueError("scramble requires a permutation")
        def fn(f: torch.Tensor) -> torch.Tensor:
            fp = f[:, perm]
            target = (gamma * f).clamp(0.0, 1.0).mean(dim=1)
            gs = _mean_matched_scale(fp, target)
            return (gs[:, None] * fp).clamp(0.0, 1.0)
        return fn
    raise ValueError(f"unknown family {family!r}")


def make_perm(state_dim: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(SCRAMBLE_SEED + seed)
    return torch.from_numpy(rng.permutation(state_dim)).long()


# ---------------------------------------------------------------------------
# Rollout (frozen core; only the commit combination is modified)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout(model: torch.nn.Module, inputs: torch.Tensor, commit_fn: Callable,
            chunk: int = 4096) -> Dict[str, torch.Tensor]:
    """Complete recursive rollout under a modified commit rule.

    f_t = sigmoid(w_f_ih e_t + w_f_hh S'_(t-1)); C_t = tanh(cand_mlp([f_t . S'_(t-1), e_t]));
    S'_t = (1 - commit_fn(f_t)) . S'_(t-1) + commit_fn(f_t) . C_t.
    Returns stacked states/candidates/gates/gates_mod and terminal logits.
    """
    device = next(model.parameters()).device
    core = model.cognition_core
    enc = model.encoder
    head = model.decision_head
    B, T, _ = inputs.shape
    states: List[torch.Tensor] = []
    cands: List[torch.Tensor] = []
    gates: List[torch.Tensor] = []
    gates_mod: List[torch.Tensor] = []
    term_logits = None
    for i in range(0, B, chunk):
        xb = inputs[i:i + chunk].to(device)
        S = model.initial_state(xb.shape[0], device)
        s_list, c_list, g_list, gm_list = [], [], [], []
        for t in range(T):
            e = enc(xb[:, t, :])
            f = torch.sigmoid(core.w_f_ih(e) + core.w_f_hh(S))
            h_reset = f * S
            C = torch.tanh(core.cand_mlp(torch.cat([h_reset, e], dim=-1)))
            fm = commit_fn(f)
            S = (1.0 - fm) * S + fm * C
            s_list.append(S); c_list.append(C); g_list.append(f); gm_list.append(fm)
        states.append(torch.stack(s_list, dim=1))
        cands.append(torch.stack(c_list, dim=1))
        gates.append(torch.stack(g_list, dim=1))
        gates_mod.append(torch.stack(gm_list, dim=1))
        lg = head(S)  # (b, R, V)
        term_logits = lg if term_logits is None else torch.cat([term_logits, lg], dim=0)
    return {
        "states": torch.cat(states, dim=0), "cands": torch.cat(cands, dim=0),
        "gates": torch.cat(gates, dim=0), "gates_mod": torch.cat(gates_mod, dim=0),
        "term_logits": term_logits,
    }


@torch.no_grad()
def horizon_metrics(model: torch.nn.Module, dataset, commit_fn: Callable) -> Dict[str, float]:
    inputs = dataset.inputs
    targets = dataset.targets
    roll = rollout(model, inputs, commit_fn)
    pred = roll["term_logits"].argmax(-1)
    return {"em": float((pred == targets).all(dim=-1).float().mean()),
            "per_reg": float((pred == targets).float().mean())}


# ---------------------------------------------------------------------------
# Transition-level + decision-margin arrays on the canonical T=10 test split
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_transition_data(model: torch.nn.Module, dataset, commit_fn: Callable) -> Dict[str, Any]:
    """Per-step committed/candidate decoding + per-ADD-transition decision margins."""
    inputs = dataset.inputs
    step_targets = dataset.step_targets
    targets = dataset.targets
    mask, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)
    roll = rollout(model, inputs, commit_fn)
    states, cands, gates, gates_mod = roll["states"], roll["cands"], roll["gates"], roll["gates_mod"]
    B, T, D = states.shape
    dev = states.device
    head = model.decision_head

    cp = head(states.reshape(-1, D)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)
    kp = head(cands.reshape(-1, D)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)
    cok = cp == step_targets
    kok = kp == step_targets

    comm_full_mean = float(cok.all(-1).float().mean())
    comm_written_mean = float((cok | ~mask).all(-1).float().mean())
    cand_full_mean = float(kok.all(-1).float().mean())
    cand_written_mean = float((kok | ~mask).all(-1).float().mean())

    cs = (~cok.all(-1)).cpu().numpy()
    cr = (~((cok | ~mask).all(-1))).cpu().numpy()
    first_strict = np.full(B, -1, np.int64); first_real = np.full(B, -1, np.int64)
    for b in range(B):
        w = np.nonzero(cs[b])[0]
        if w.size:
            first_strict[b] = int(w[0]) + 1
        w = np.nonzero(cr[b])[0]
        if w.size:
            first_real[b] = int(w[0]) + 1

    g = gates.clamp(1e-7, 1 - 1e-7)
    gate_entropy = float((-(g * g.log2() + (1 - g) * (1 - g).log2())).mean())
    seq_em = (roll["term_logits"].argmax(-1) == targets).all(-1)

    # per-ADD-transition decision margins (intervention trajectory)
    prev_state = torch.cat([torch.zeros(B, 1, D, device=dev), states[:, :-1]], dim=1)
    prev_true = torch.zeros_like(step_targets); prev_true[:, 1:] = step_targets[:, :-1]
    W = head.proj.weight.detach().view(gr.NUM_REGISTERS, gr.NUM_VALUES, D)
    bvec = head.proj.bias.detach().view(gr.NUM_REGISTERS, gr.NUM_VALUES)
    idx = (op == RegisterTrackingTask.OP_ADD).nonzero()
    bi, ti = idx[:, 0], idx[:, 1]
    n = int(len(idx))
    S_prev = prev_state[bi, ti]; C = cands[bi, ti]; S_t = states[bi, ti]
    fvec = gates[bi, ti]; fmod = gates_mod[bi, ti]
    dest = reg_a[bi, ti]; src = reg_b[bi, ti]
    oa = prev_true[bi, ti, dest]; ob = prev_true[bi, ti, src]
    yv = step_targets[bi, ti, dest]
    ar = torch.arange(n, device=dev)
    z_prev = ri._logit_matrix(W, bvec, S_prev, dest)
    z_cand = ri._logit_matrix(W, bvec, C, dest)
    z_comm = ri._logit_matrix(W, bvec, S_t, dest)
    Mprev = z_prev[ar, yv][:, None] - z_prev
    Mcand = z_cand[ar, yv][:, None] - z_cand
    Mcomm = z_comm[ar, yv][:, None] - z_comm
    Mmask = Mprev.clone(); Mmask[ar, yv] = float("inf")
    a_star = Mmask.argmin(dim=1)
    m_prev = Mprev.gather(1, a_star[:, None]).squeeze(1)
    m_cand = Mcand.gather(1, a_star[:, None]).squeeze(1)
    m_commit = Mcomm.gather(1, a_star[:, None]).squeeze(1)
    dm_cand_all = Mcand - Mprev
    dm_cand = dm_cand_all.gather(1, a_star[:, None]).squeeze(1)
    dm_commit = m_commit - m_prev
    nan_t = torch.full_like(dm_cand, float("nan"))
    rho = torch.where(dm_cand > ri.EPS_ABS, dm_commit / dm_cand.clamp_min(ri.EPS_ABS), nan_t)
    # alpha*_full (exact, linear head)
    neg_needed = Mprev < 0
    need = torch.where(neg_needed, (-Mprev) / dm_cand_all.clamp_min(ri.EPS_ABS), torch.zeros_like(Mprev))
    infeasible = neg_needed & (dm_cand_all <= ri.EPS_ABS)
    alpha_full = need.max(dim=1).values
    alpha_full = torch.where(m_prev >= 0, torch.zeros_like(alpha_full), alpha_full)
    alpha_full = torch.where(infeasible.any(dim=1), nan_t, alpha_full)
    # coordinate decomposition
    w_y = W[dest, yv]; w_star = W[dest, a_star]
    dvec = C - S_prev
    qcand = (w_y - w_star) * dvec
    qcand_pos = qcand.clamp(min=0.0); qcand_neg = (-qcand).clamp(min=0.0)
    qpos = qcand_pos.sum(dim=1); qneg = qcand_neg.sum(dim=1)
    gw_pos = torch.where(qpos > ri.EPS_ABS, (qcand_pos * fmod).sum(dim=1) / qpos.clamp_min(ri.EPS_ABS), nan_t)
    gw_neg = torch.where(qneg > ri.EPS_ABS, (qcand_neg * fmod).sum(dim=1) / qneg.clamp_min(ri.EPS_ABS), nan_t)

    to = lambda x: x.detach().cpu().numpy()
    cand_pred = kp[bi, ti, dest]; comm_pred = cp[bi, ti, dest]
    return {
        "cand_ok": to((cand_pred == yv).float()), "comm_ok": to((comm_pred == yv).float()),
        "gen_general": to(((dest != src) & (ob > 0)).float()),
        "wrap": to(((oa + ob) >= gr.NUM_VALUES).float()),
        "m_prev": to(m_prev), "m_cand": to(m_cand), "m_commit": to(m_commit),
        "dm_cand": to(dm_cand), "dm_commit": to(dm_commit), "rho": to(rho),
        "alpha_full": to(alpha_full), "gw_pos": to(gw_pos), "gw_neg": to(gw_neg),
        "f_mean": to(fvec.mean(dim=1)), "f_scaled_mean": to(fmod.mean(dim=1)),
        "n_add": n,
        "scalar": {
            "comm_full_mean": comm_full_mean, "comm_written_mean": comm_written_mean,
            "cand_full_mean": cand_full_mean, "cand_written_mean": cand_written_mean,
            "first_strict_error_rate": float((first_strict > 0).mean()),
            "first_real_error_rate": float((first_real > 0).mean()),
            "gate_mean": float(gates.mean()), "gate_scaled_mean": float(gates_mod.mean()),
            "gate_entropy": gate_entropy,
        },
        "seq_em": to(seq_em.float()),
        "by_final_op": _by_final_op(cp, op, reg_a, reg_b, step_targets, targets, mask),
    }


def _by_final_op(cp, op, reg_a, reg_b, step_targets, targets, mask) -> Dict[str, float]:
    last = op[:, -1]
    prev_true = torch.zeros_like(step_targets); prev_true[:, 1:] = step_targets[:, :-1]
    reg_b_last = reg_b[:, -1]
    wrap_last = (prev_true[:, -1].gather(1, reg_a[:, -1, None]).squeeze(1)
                 + prev_true[:, -1].gather(1, reg_b_last[:, None]).squeeze(1)) >= gr.NUM_VALUES
    term_ok = (cp[:, -1] == step_targets[:, -1]).all(-1)  # committed decode at T as terminal proxy
    return {
        "SET": float(term_ok[last == 0].float().mean()) if (last == 0).any() else None,
        "SWAP": float(term_ok[last == 1].float().mean()) if (last == 1).any() else None,
        "ADD_no_wrap": float(term_ok[(last == 2) & ~wrap_last].float().mean()) if ((last == 2) & ~wrap_last).any() else None,
        "ADD_wrap": float(term_ok[(last == 2) & wrap_last].float().mean()) if ((last == 2) & wrap_last).any() else None,
    }


@torch.no_grad()
def controlled_metrics(model: torch.nn.Module, builders: Dict[str, Any], commit_fn: Callable) -> Dict[str, Any]:
    out = {}
    for name, d in builders.items():
        inputs = torch.from_numpy(d["inputs"]); step_tgt = torch.from_numpy(d["step_targets"])
        final_tgt = torch.from_numpy(d["final_targets"])
        roll = rollout(model, inputs, commit_fn)
        states = roll["states"]; cands = roll["cands"]
        B, T, D = states.shape
        head = model.decision_head
        cp = head(states.reshape(-1, D)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)
        kp = head(cands.reshape(-1, D)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)
        reg_a = inputs[..., 3:7].argmax(-1)
        rows = torch.arange(B)
        tadd = torch.from_numpy(np.asarray(d["add_step"])).clamp_min(0)
        dest = reg_a[rows, tadd]
        term_pred = roll["term_logits"].argmax(-1)
        out[name] = {
            "terminal_em": float((term_pred == final_tgt).all(-1).float().mean()),
            "committed_dest_acc_at_add": float((cp[rows, tadd, dest] == step_tgt[rows, tadd, dest]).float().mean()),
            "candidate_dest_acc_at_add": float((kp[rows, tadd, dest] == step_tgt[rows, tadd, dest]).float().mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Per-config evaluation + rescue accounting vs baseline
# ---------------------------------------------------------------------------

def _primary_stats(td: Dict[str, Any], base_primary: np.ndarray) -> Dict[str, Any]:
    """Margin summaries over the fixed baseline candidate-correct general ADD set."""
    out = {"n": int(base_primary.sum())}
    if base_primary.sum() == 0:
        return out
    for key in ("m_cand", "m_commit", "dm_cand", "dm_commit", "rho", "gw_pos", "gw_neg", "alpha_full"):
        v = td[key][base_primary]
        v = v[np.isfinite(v)]
        out[key + "_median"] = float(np.median(v)) if v.size else None
        out[key + "_mean"] = float(v.mean()) if v.size else None
    out["comm_acc"] = float(td["comm_ok"][base_primary].mean())
    out["cand_acc"] = float(td["cand_ok"][base_primary].mean())
    return out


def _rescue_accounting(td: Dict[str, Any], base: Dict[str, Any], base_primary: np.ndarray,
                       strata: Dict[str, np.ndarray], sufficient: np.ndarray) -> Dict[str, Any]:
    """Rescue/regression among baseline candidate-correct general ADDs (transition level)."""
    base_comm = base["comm_ok"][base_primary] > 0.5
    new_comm = td["comm_ok"][base_primary] > 0.5
    n_succ = int(base_comm.sum()); n_fail = int((~base_comm).sum())
    rescued = int((~base_comm & new_comm).sum())
    regressed = int((base_comm & ~new_comm).sum())
    rec = {
        "n_base_succ": n_succ, "n_base_fail": n_fail,
        "rescued": rescued, "regressed": regressed,
        "rescue_rate": rescued / n_fail if n_fail else None,
        "regression_rate": regressed / n_succ if n_succ else None,
        "net_recovery": rescued - regressed,
        "net_recovery_rate": (rescued - regressed) / max(n_succ + n_fail, 1),
    }
    # by candidate-margin stratum (predeclared quantile bins on the BASELINE distribution)
    by = {}
    for name, mask in strata.items():
        if name == "edges":
            continue
        m = mask[base_primary]
        b = base_comm[m]; nn = new_comm[m]
        by[name] = {
            "n": int(m.sum()), "n_base_fail": int((~b).sum()),
            "rescued": int((~b & nn).sum()), "regressed": int((b & ~nn).sum()),
            "rescue_rate": float((~b & nn).sum() / max((~b).sum(), 1)),
            "regression_rate": float((b & ~nn).sum() / max(b.sum(), 1)),
            "comm_acc": float(nn.mean()) if m.any() else None,
        }
    rec["by_stratum"] = by
    # off-path subset: baseline failures already delivering >= scalar-equivalent transfer
    suff_sub = sufficient[base_primary] if sufficient.shape[0] == base_primary.shape[0] else np.zeros(base_comm.shape, bool)
    if suff_sub.any():
        b = base_comm[suff_sub]; nn = new_comm[suff_sub]
        rec["offpath"] = {
            "n_sufficient": int(suff_sub.sum()),
            "n_base_fail_sufficient": int((~b).sum()),
            "rescued_sufficient": int((~b & nn).sum()),
            "rescue_rate_sufficient": float((~b & nn).sum() / max((~b).sum(), 1)),
        }
    return rec


def _seq_rescue(td: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    b = base["seq_em"] > 0.5; n = td["seq_em"] > 0.5
    return {
        "n_seq": int(len(b)), "n_base_fail_seq": int((~b).sum()), "n_base_succ_seq": int(b.sum()),
        "rescued_seq": int((~b & n).sum()), "broken_seq": int((b & ~n).sum()),
        "rescue_rate_seq": float((~b & n).sum() / max((~b).sum(), 1)),
        "break_rate_seq": float((b & ~n).sum() / max(b.sum(), 1)),
    }


def evaluate_config(model: torch.nn.Module, seed: int, datasets: Dict[int, Any], builders: Dict[str, Any],
                    commit_fn: Callable, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"horizons": {}, "controlled": controlled_metrics(model, builders, commit_fn)}
    for L in HORIZONS:
        rec["horizons"][str(L)] = horizon_metrics(model, datasets[L], commit_fn)
    td = test_transition_data(model, datasets[10], commit_fn)
    rec["test"] = td["scalar"]
    rec["by_final_op"] = td["by_final_op"]
    rec["_td"] = td
    rec["_seq_em"] = td["seq_em"]
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_datasets(seed: int) -> Dict[int, Any]:
    return {L: gr.horizon_test_set(seed, L) for L in HORIZONS}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment J runner (interventional, frozen)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--out", default="experiments/experiment_j_records.json")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--families", nargs="*", default=None, help="restrict families (debug)")
    args = ap.parse_args()

    seeds = args.seeds or gr.SEEDS
    models = args.models or list(MODEL_PLAN)
    t0 = time.perf_counter()
    builders = gr._build_controlled_add_sequences()
    records: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"preregistration": PREREGISTRATION, "seeds": seeds,
                            "checkpoints": {}, "families_per_model": MODEL_PLAN,
                            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    ds_cache: Dict[int, Dict[int, Any]] = {}

    for model_key in models:
        meta["checkpoints"][model_key] = {}
        families = [f for f in MODEL_PLAN[model_key] if (args.families is None or f in args.families)]
        for seed in seeds:
            loaded = eg.load_checkpoint(args.checkpoints_dir, model_key, seed)
            sha_before = loaded["sha256"]
            g_sha = _g_sha(model_key, seed)
            if g_sha is not None and sha_before != g_sha:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint hash changed vs Experiment G")
            model = loaded["model"]
            if seed not in ds_cache:
                ds_cache[seed] = build_datasets(seed)
            datasets = ds_cache[seed]
            perm = make_perm(gr.STATE_DIM, seed)

            baseline = evaluate_config(model, seed, datasets, builders,
                                       make_commit_fn("scale", BASELINE_GAMMA))
            btd = baseline.pop("_td"); base_seq = baseline.pop("_seq_em")
            baseline["seq_em"] = base_seq
            # baseline reproduction check vs the frozen forward
            repro = _verify_baseline_reproduction(model, datasets[10])
            base = {"comm_ok": btd["comm_ok"], "seq_em": base_seq}
            base_primary = (btd["gen_general"] > 0.5) & (btd["cand_ok"] > 0.5)
            strata = _cand_margin_strata(btd["m_cand"], base_primary)
            # off-path sufficient-transfer subset (baseline)
            suff = base_primary & np.isfinite(btd["alpha_full"]) & np.isfinite(btd["rho"]) & (btd["rho"] >= btd["alpha_full"])

            for family in families:
                for gamma in GAMMA_GRID:
                    cf = make_commit_fn(family, gamma, perm)
                    cfg = evaluate_config(model, seed, datasets, builders, cf)
                    td = cfg.pop("_td"); seq = cfg.pop("_seq_em")
                    cfg.update({"model": model_key, "seed": seed, "family": family, "gamma": gamma,
                                "cand_margin_strata_edges": [float(x) for x in strata["edges"]],
                                "primary": _primary_stats(td, base_primary),
                                "rescue": _rescue_accounting(td, base, base_primary, strata, suff),
                                "seq_rescue": _seq_rescue(td, base),
                                "baseline_reproduction": repro if family == PRIMARY_FAMILY and gamma == BASELINE_GAMMA else None})
                    records.append(cfg)
            sha_after = sha256_file(loaded["path"])
            if sha_after != sha_before:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint changed during run")
            meta["checkpoints"][model_key][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {model_key} seed={seed}: {len(records)} records so far "
                  f"(base T10 EM={baseline['horizons']['10']['em']:.4f}, repro={repro['passed']})")

    meta["n_records"] = len(records)
    meta["wall_clock_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=2)
    print(f"Wrote {args.out}: {len(records)} records in {meta['wall_clock_s']:.1f}s")


def _cand_margin_strata(m_cand: np.ndarray, base_primary: np.ndarray) -> Dict[str, np.ndarray]:
    """Predeclared tercile bins of candidate margin on the BASELINE candidate-correct set."""
    v = m_cand[base_primary]
    q1, q2 = np.percentile(v, [33.333, 66.667])
    idx = np.nonzero(base_primary)[0]
    lo = idx[v <= q1]; hi = idx[v > q2]; mid = idx[(v > q1) & (v <= q2)]
    return {"low": _mask_of(len(m_cand), lo), "mid": _mask_of(len(m_cand), mid),
            "high": _mask_of(len(m_cand), hi), "edges": np.array([q1, q2])}


def _mask_of(n: int, rows: np.ndarray) -> np.ndarray:
    m = np.zeros(n, dtype=bool); m[rows] = True; return m


def _g_sha(model_key: str, seed: int) -> Optional[str]:
    path = "experiments/experiment_g_transitions_meta.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        meta = json.load(f)
    return meta["models"][model_key]["checkpoints"][str(seed)]["sha256"]


@torch.no_grad()
def _verify_baseline_reproduction(model: torch.nn.Module, dataset) -> Dict[str, Any]:
    """gamma=1 rollout must reproduce the frozen forward: numerically to float32 round-off and
    with bit-identical decoded outputs and terminal predictions."""
    inputs = dataset.inputs
    frozen = model.forward_with_states(inputs, return_components=True)
    roll = rollout(model, inputs, make_commit_fn("scale", 1.0))
    fs = torch.stack(frozen.states, dim=1); fc = torch.stack(frozen.candidates, dim=1)
    fg = torch.stack(frozen.gates, dim=1)
    d_state = float((fs - roll["states"]).abs().max())
    d_cand = float((fc - roll["cands"]).abs().max())
    d_gate = float((fg - roll["gates"]).abs().max())
    head = model.decision_head
    B, T, D = fs.shape
    dec_frozen = head(fs.reshape(-1, D)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)
    dec_roll = head(roll["states"].reshape(-1, D)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)
    decode_equal = bool((dec_frozen == dec_roll).all())
    term_frozen = frozen.terminal_logits.argmax(-1)
    term_roll = roll["term_logits"].argmax(-1)
    term_equal = bool((term_frozen == term_roll).all())
    tol = 1e-5
    return {"passed": bool(max(d_state, d_cand, d_gate) < tol and decode_equal and term_equal),
            "max_state_diff": d_state, "max_candidate_diff": d_cand, "max_gate_diff": d_gate,
            "decode_equal": decode_equal, "terminal_equal": term_equal, "tol": tol}


if __name__ == "__main__":
    main()
