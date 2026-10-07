#!/usr/bin/env python3
"""Experiment O — Controlled Representation-Side Test of Parity Alignment (interventional).

Experiment N localized the off-path same-parity signature to the state<->label alignment (N-B),
with the frozen readout geometry a small non-dominant substrate. O makes ONE tightly controlled
causal test of the state-side mechanism: it removes the W-derived parity component of the
candidate representation BEFORE the frozen gate/commit and measures whether the off-path
third-basin error decreases.

Intervention (primary, candidate-side)
--------------------------------------
    p_dest = normalize(mean normalised W[dest, even v] - mean normalised W[dest, odd v])
             (derived from the frozen readout ONLY; no target/failure labels)
    deltaC = C - S_prev
    delta_p = <deltaC, p_dest> p_dest
    C' = C - kappa delta_p                      (kappa in {0, 0.5, 1.0})
    S_commit' = (1 - f) S_prev + f C'           (ORIGINAL frozen gate f and commit rule)

The modified candidate flows through the original frozen gate/commit mechanism (the gate f does
not depend on C). Applied at ADD transitions only, deterministically, state-locally.

Controls
--------
* kappa = 0 must be mathematically identical to the frozen computation (baseline).
* Norm-matched orthogonal control: C_control = C - kappa |<deltaC,p>| q with q a deterministic
  unit vector orthogonal to p_dest (no arithmetic labels).
* Commit-side stage control: S_commit' = S_commit - kappa <(S_commit - S_prev), p_dest> p_dest,
  applied after the original commit (plus its own norm-matched orthogonal control).

Measurement only in the sense that NO weights/readout/gate/commit rule/benchmark are modified:
the intervention lives entirely in inference-time state computation. No training, no optimizer,
no gradients, no learned policy, no verifier, no branching, no Phase 2. Checkpoint SHA-256
verified before/after.

Outputs
-------
    experiments/experiment_o_representation_intervention.npz        per-config transition/seq arrays
    experiments/experiment_o_representation_intervention_meta.json  config, hashes, reproduction
Reproduce
---------
    python experiments/run_experiment_o.py
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402
import run_experiment_l as rl  # noqa: E402
import run_experiment_n as rn  # noqa: E402  (parity_axes, read-only reuse)
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
CONFIGS: List[Tuple[str, float]] = [
    ("none", 0.0),
    ("cand", 0.5), ("cand", 1.0),
    ("orth", 0.5), ("orth", 1.0),
    ("commit", 0.5), ("commit", 1.0),
    ("commit_orth", 0.5), ("commit_orth", 1.0),
]
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20]
ADD = RegisterTrackingTask.OP_ADD


def config_label(family: str, kappa: float) -> str:
    return f"{family}_{kappa}"


def _fallback_dir(p: torch.Tensor) -> torch.Tensor:
    """Deterministic unit vector orthogonal to each row of p."""
    j = p.abs().argmin(dim=1)
    pj = p.gather(1, j[:, None])
    e = torch.zeros_like(p); e.scatter_(1, j[:, None], 1.0)
    q = e - pj * p
    return q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def rollout_o(model: torch.nn.Module, inputs: torch.Tensor, P: np.ndarray,
              family: str, kappa: float, chunk: int = 4096) -> Dict[str, Any]:
    """Complete recursive rollout with the O intervention. Returns stacks + geometry aggregates."""
    device = next(model.parameters()).device
    core = model.cognition_core; enc = model.encoder; head = model.decision_head
    B, T, _ = inputs.shape
    _, op, reg_a, _ = gr.write_mask_from_inputs(inputs)
    P_t = torch.from_numpy(P).to(device).float()
    states: List[torch.Tensor] = []; cands: List[torch.Tensor] = []; cands_mod: List[torch.Tensor] = []
    gates: List[torch.Tensor] = []
    geom = {"cand_removal": 0.0, "commit_removal": 0.0, "cos_delta_p": 0.0, "abs_proj": 0.0, "n_add": 0}
    term = None
    for i in range(0, B, chunk):
        xb = inputs[i:i + chunk].to(device)
        opc = op[i:i + chunk].to(device); rac = reg_a[i:i + chunk].to(device)
        S = model.initial_state(xb.shape[0], device)
        for t in range(T):
            e = enc(xb[:, t, :])
            f = torch.sigmoid(core.w_f_ih(e) + core.w_f_hh(S))
            C = torch.tanh(core.cand_mlp(torch.cat([f * S, e], dim=-1)))
            p = P_t[rac[:, t]]                              # (b, D) parity axis of dest register
            apply = (opc[:, t] == ADD)
            Cmod = C
            if family in ("cand", "orth"):
                dC = C - S
                proj = (dC * p).sum(-1, keepdim=True)
                dp = proj * p
                if family == "cand":
                    delta = kappa * dp
                else:
                    resid = dC - dp
                    rn_ = resid.norm(dim=1, keepdim=True)
                    q = resid / rn_.clamp_min(1e-6)
                    q = torch.where(rn_ > 1e-6, q, _fallback_dir(p))
                    delta = kappa * proj.abs() * q
                Cmod = C - delta
                Cmod = torch.where(apply[:, None], Cmod, C)
                if kappa != 0.0:
                    m = apply
                    geom["cand_removal"] += float(((Cmod - C).norm(dim=1) * m).sum())
                    geom["abs_proj"] += float((proj.abs().squeeze(1) * m).sum())
                    w = ((Cmod - C) * p).sum(1)
                    c = w / torch.where((Cmod - C).norm(dim=1) == 0,
                                        torch.ones_like(w), (Cmod - C).norm(dim=1))
                    geom["cos_delta_p"] += float((c * m).sum())
                    geom["n_add"] += int(m.sum())
            S_prev = S
            S = (1.0 - f) * S + f * Cmod
            if family in ("commit", "commit_orth"):
                dS = S - S_prev
                proj = (dS * p).sum(-1, keepdim=True)
                dp = proj * p
                if family == "commit":
                    delta = kappa * dp
                else:
                    resid = dS - dp
                    rn_ = resid.norm(dim=1, keepdim=True)
                    q = resid / rn_.clamp_min(1e-6)
                    q = torch.where(rn_ > 1e-6, q, _fallback_dir(p))
                    delta = kappa * proj.abs() * q
                Sp = S - delta
                S = torch.where(apply[:, None], Sp, S)
                if kappa != 0.0:
                    geom["commit_removal"] += float((delta.norm(dim=1) * apply).sum())
            states.append(S); cands.append(C); cands_mod.append(Cmod); gates.append(f)
        lg = head(S)
        term = lg if term is None else torch.cat([term, lg], 0)
    return {
        "states": torch.stack(states, 1), "cands": torch.stack(cands, 1),
        "cands_mod": torch.stack(cands_mod, 1), "gates": torch.stack(gates, 1),
        "term_logits": term, "geom": geom,
    }


def _decode(head, states, R, V):
    B, T, D = states.shape
    return head(states.reshape(-1, D)).reshape(B, T, R, V).argmax(-1)


@torch.no_grad()
def transition_and_seq(model, ds, P, family, kappa) -> Dict[str, Any]:
    """Canonical T=10 rollout: per-ADD-transition committed/candidate decode + sequence metrics."""
    roll = rollout_o(model, ds.inputs, P, family, kappa)
    states, cands_mod = roll["states"], roll["cands_mod"]
    R, V = gr.NUM_REGISTERS, gr.NUM_VALUES
    cp = _decode(model.decision_head, states, R, V)
    kp = _decode(model.decision_head, cands_mod, R, V)
    step_tgt = ds.step_targets
    _, op, reg_a, _ = gr.write_mask_from_inputs(ds.inputs)
    idx = (op == ADD).nonzero()
    b, t = idx[:, 0], idx[:, 1]
    dest = reg_a[b, t]
    y = step_tgt[b, t, dest]
    comm_ok = (cp[b, t, dest] == y)
    cand_ok = (kp[b, t, dest] == y)
    term = roll["term_logits"].argmax(-1)
    tgt = ds.targets
    seq_em = (term == tgt).all(-1)
    hamming = (term != tgt).sum(-1)
    per_reg = (term == tgt).float().mean()
    # by final operation (terminal decode proxy, matching rj._by_final_op convention)
    last = op[:, -1]
    prev_true = torch.zeros_like(step_tgt); prev_true[:, 1:] = step_tgt[:, :-1]
    reg_b_last = gr.write_mask_from_inputs(ds.inputs)[3][:, -1]
    wrap_last = (prev_true[:, -1].gather(1, reg_a[:, -1, None]).squeeze(1)
                 + prev_true[:, -1].gather(1, reg_b_last[:, None]).squeeze(1)) >= V
    term_ok = (cp[:, -1] == step_tgt[:, -1]).all(-1)
    def rate(mask):
        return float(term_ok[mask].float().mean()) if bool(mask.any()) else None
    by_op = {"SET": rate(last == 0), "SWAP": rate(last == 1),
             "ADD_no_wrap": rate((last == 2) & ~wrap_last), "ADD_wrap": rate((last == 2) & wrap_last)}
    return {"comm_ok": comm_ok.detach().cpu().numpy(), "cand_ok": cand_ok.detach().cpu().numpy(),
            "seq_em": seq_em.detach().cpu().numpy(), "geom": roll["geom"],
            "per_reg": float(per_reg), "hamming_hist": hamming.detach().cpu().numpy(),
            "by_op": by_op}


@torch.no_grad()
def horizon_em(model, ds, P, family, kappa) -> Dict[str, float]:
    roll = rollout_o(model, ds.inputs, P, family, kappa)
    pred = roll["term_logits"].argmax(-1)
    tgt = ds.targets
    return {"em": float((pred == tgt).all(-1).float().mean()),
            "per_reg": float((pred == tgt).float().mean())}


def baseline_groups(model, ds, P) -> Dict[str, np.ndarray]:
    """Group labels (0 off-path / 1 success / 2 non-off-path) computed from the frozen baseline."""
    roll = rollout_o(model, ds.inputs, P, "none", 0.0)
    states, cands = roll["states"], roll["cands"]
    R, V, D = gr.NUM_REGISTERS, gr.NUM_VALUES, gr.STATE_DIM
    head = model.decision_head
    cp = _decode(head, states, R, V); kp = _decode(head, cands, R, V)
    step_tgt = ds.step_targets
    _, op, reg_a, reg_b = gr.write_mask_from_inputs(ds.inputs)
    prev_true = torch.zeros_like(step_tgt); prev_true[:, 1:] = step_tgt[:, :-1]
    idx = (op == ADD).nonzero(); b, t = idx[:, 0], idx[:, 1]
    dest = reg_a[b, t]; src = reg_b[b, t]; y = step_tgt[b, t, dest]
    oa = prev_true[b, t, dest]; ob = prev_true[b, t, src]
    W, bvec = rl._matrix(head)
    ar = torch.arange(len(idx), device=states.device)
    def z(S): return torch.einsum("nd,nvd->nv", S, W[dest]) + bvec[dest]
    prev_state = torch.cat([torch.zeros_like(states[:, :1]), states[:, :-1]], 1)
    zp = z(prev_state[b, t]); zc = z(cands[b, t]); zk = z(states[b, t])
    Mp = zp[ar, y][:, None] - zp; Mc = zc[ar, y][:, None] - zc; Mk = zk[ar, y][:, None] - zk
    Mm = Mp.clone(); Mm[ar, y] = float("inf"); a_prev = Mm.argmin(1)
    m_prev = Mp.gather(1, a_prev[:, None]).squeeze(1)
    m_cand = Mc.gather(1, a_prev[:, None]).squeeze(1)
    m_commit = Mk.gather(1, a_prev[:, None]).squeeze(1)
    dm_all = Mc - Mp
    dm_cand = dm_all.gather(1, a_prev[:, None]).squeeze(1)
    neg = Mp < 0
    need = torch.where(neg, (-Mp) / dm_all.clamp_min(rl.EPS_ABS), torch.zeros_like(Mp))
    infeasible = neg & (dm_all <= rl.EPS_ABS)
    alpha_full = need.max(1).values
    alpha_full = torch.where(m_prev >= 0, torch.zeros_like(alpha_full), alpha_full)
    alpha_full = torch.where(infeasible.any(1), torch.full_like(alpha_full, float("nan")), alpha_full)
    rho = torch.where(dm_cand > rl.EPS_ABS, (m_commit - m_prev) / dm_cand.clamp_min(rl.EPS_ABS),
                      torch.full_like(dm_cand, float("nan")))
    gen = (dest != src) & (ob > 0)
    cand_ok = kp[b, t, dest] == y
    comm_ok = cp[b, t, dest] == y
    finite = torch.isfinite(alpha_full) & torch.isfinite(rho)
    offpath = gen & cand_ok & (~comm_ok) & finite & (rho >= alpha_full)
    succ = gen & cand_ok & comm_ok
    nonoff = gen & cand_ok & (~comm_ok) & ~(finite & (rho >= alpha_full))
    group = np.full(len(idx), -1, np.int64)
    to = lambda x: x.detach().cpu().numpy()
    group[to(offpath)] = 0; group[to(succ)] = 1; group[to(nonoff)] = 2
    return {"group": group, "comm_ok": to(comm_ok), "cand_ok": to(cand_ok)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment O runner (interventional, frozen)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_o_representation_intervention.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_o_representation_intervention_meta.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    model_keys = args.models or MODEL_KEYS
    seeds = args.seeds or gr.SEEDS
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "configs": [{"family": f, "kappa": k, "label": config_label(f, k)} for f, k in CONFIGS],
        "horizons": HORIZONS, "models": {}, "reproduction": {},
        "primary_endpoint": "candidate-side kappa=1 off-path rescue; sequence T=10/T=20 EM",
        "intervention": ("p_dest from W alone; candidate-side C' = C - kappa <C-S_prev,p> p through the "
                         "original frozen gate/commit; applied at ADD transitions only."),
        "note": ("Interventional but weight-frozen: no training, no optimizer, no weight/readout/gate/"
                 "commit-rule modification; intervention lives entirely in inference-time state "
                 "computation. No target/failure labels enter the intervention direction."),
    }
    payload: Dict[str, np.ndarray] = {}
    for mk in model_keys:
        meta["models"][mk] = {"checkpoints": {}}
        meta["reproduction"][mk] = {}
        per_seed_groups = []; per_seed_seedcol = []; per_seed_base_comm = []; per_seed_base_cand = []
        per_seed_term = {config_label(f, k): [] for f, k in CONFIGS}
        per_seed_comm = {config_label(f, k): [] for f, k in CONFIGS}
        per_seed_cand = {config_label(f, k): [] for f, k in CONFIGS}
        geom_by_cfg = {config_label(f, k): [] for f, k in CONFIGS}
        horizon_by_cfg = {config_label(f, k): {} for f, k in CONFIGS}
        for seed in seeds:
            loaded = eg.load_checkpoint(args.checkpoints_dir, mk, seed)
            sha_before = loaded["sha256"]
            model = loaded["model"]
            W = model.decision_head.proj.weight.detach().cpu().numpy().reshape(
                gr.NUM_REGISTERS, gr.NUM_VALUES, gr.STATE_DIM)
            P = rn.parity_axes(W[None])[0]                    # (R, D) from W alone
            ds = gr.canonical_splits(seed).test_set
            # baseline groups + reproduction check vs the frozen forward
            groups = baseline_groups(model, ds, P)
            per_seed_groups.append(groups["group"]); per_seed_base_comm.append(groups["comm_ok"])
            per_seed_base_cand.append(groups["cand_ok"])
            per_seed_seedcol.append(np.full(len(groups["group"]), seed, np.int64))
            frozen = model.forward_with_states(ds.inputs, return_components=True)
            base_roll = rollout_o(model, ds.inputs, P, "none", 0.0)
            fs = torch.stack(frozen.states, 1).detach()
            repro = {
                "max_state_diff": float((fs - base_roll["states"]).abs().max()),
                "decode_equal": bool((frozen.terminal_logits.argmax(-1) == base_roll["term_logits"].argmax(-1)).all()),
            }
            repro["passed"] = repro["max_state_diff"] < 1e-5 and repro["decode_equal"]
            meta["reproduction"][mk][str(seed)] = repro

            for fam, k in CONFIGS:
                lab = config_label(fam, k)
                td = transition_and_seq(model, ds, P, fam, k)
                per_seed_term[lab].append(td["seq_em"])
                per_seed_comm[lab].append(td["comm_ok"].astype(np.uint8))
                per_seed_cand[lab].append(td["cand_ok"].astype(np.uint8))
                meta.setdefault("seq_metrics", {}).setdefault(mk, {}).setdefault(lab, {})[str(seed)] = {
                    "per_reg": td["per_reg"], "hamming_hist": td["hamming_hist"].tolist(),
                    "by_op": td["by_op"]}
                geom_by_cfg[lab].append(td["geom"])
                hres = {}
                for L in HORIZONS:
                    hds = gr.horizon_test_set(seed, L)
                    hres[str(L)] = horizon_em(model, hds, P, fam, k)
                horizon_by_cfg[lab][str(seed)] = hres
            if eg.sha256_file(loaded["path"]) != sha_before:
                raise RuntimeError(f"{mk} seed={seed}: checkpoint changed during run")
            meta["models"][mk]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {mk} seed={seed}: configs done (repro={repro['passed']})")
        payload[f"{mk}__add_group"] = np.concatenate(per_seed_groups)
        payload[f"{mk}__add_seed"] = np.concatenate(per_seed_seedcol)
        payload[f"{mk}__baseline_comm_ok"] = np.concatenate(per_seed_base_comm).astype(np.uint8)
        payload[f"{mk}__baseline_cand_ok"] = np.concatenate(per_seed_base_cand).astype(np.uint8)
        for lab in per_seed_term:
            payload[f"{mk}__{lab}__seq_em"] = np.concatenate(per_seed_term[lab]).astype(np.uint8)
            payload[f"{mk}__{lab}__comm_ok"] = np.concatenate(per_seed_comm[lab]).astype(np.uint8)
            payload[f"{mk}__{lab}__cand_ok"] = np.concatenate(per_seed_cand[lab]).astype(np.uint8)
        meta["reproduction"][mk + "__all_passed"] = all(
            meta["reproduction"][mk][str(s)]["passed"] for s in seeds)
        meta.setdefault("geometry", {})[mk] = geom_by_cfg
        meta.setdefault("horizon_results", {})[mk] = horizon_by_cfg

    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
