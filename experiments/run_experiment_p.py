#!/usr/bin/env python3
"""Experiment P — Sequence-Level Error-Budget Attribution (measurement only).

Where does the frozen E model's T=10/T=20 exact-match error budget live? For every sequence and
timestep P captures the frozen trajectory (S_prev, C, S_commit) and the exact frozen-readout
decodes, then assigns each failed sequence its EARLIEST OBSERVED task-level failure site:

  A  candidate-generation error : the frozen candidate already decodes the written register wrong
  B  candidate->commit residual : candidate correct, committed written-register decode wrong
  C  terminal / readout residual: no active-write candidate/commit error occurred, yet the
                                  terminal full-state decode is wrong
       C1 untouched-register terminal error
       C2 written-register correct at write, terminal decode wrong
       C3 readout-adjacent / near-boundary terminal signature (small normalized margin)
  D  mixed / unresolved

This is OBSERVATIONAL attribution. "candidate-origin sequence" means "the earliest observed
task-level error occurs when the frozen candidate is already decoded incorrectly"; it does NOT
mean candidate generation is the proven cause. No counterfactual rollout, no intervention, no
branching, no repair: all analysis uses the actual frozen trajectory. Checkpoint SHA-256 verified
before/after.

Outputs
-------
    experiments/experiment_p_error_budget.npz        per-step trajectory arrays + per-sequence classes
    experiments/experiment_p_error_budget_meta.json  config, hashes, horizons, alignment
Reproduce
---------
    python experiments/run_experiment_p.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_p_error_budget.npz
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
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20]
FULL_ARRAY_HORIZONS = [10, 20]     # store per-step arrays only for these
C3_BAR = 0.5                        # normalized true-vs-winner distance bar for "near boundary"

CLASS_CODE = {"success": 0, "A": 1, "B": 2, "C1": 3, "C2": 4, "C3": 5, "D": 6}
CLASS_NAME = {v: k for k, v in CLASS_CODE.items()}
ADD = RegisterTrackingTask.OP_ADD


def classify_first_error(cand_written_ok: np.ndarray, comm_written_ok: np.ndarray, terminal_ok: bool,
                         wrong_regs: np.ndarray, written_ever: np.ndarray,
                         term_norm_margin: np.ndarray, c3_bar: float = C3_BAR) -> Tuple[str, int]:
    """Earliest-observed task-level failure class for one sequence (exclusive A>B>C; C1/C2/C3).

    A : candidate written-register decode wrong at the earliest such t
    B : candidate correct but committed written-register wrong at the earliest such t
    C1: no A/B; terminal full-state wrong; a wrong register was never written
    C2: no A/B; terminal wrong; wrong registers were all written at some point
    C3: no A/B; terminal wrong; near-boundary (min normalized true-vs-winner margin < c3_bar)
    D : failure not cleanly attributable
    """
    tA = int(np.argmax(~cand_written_ok)) if not cand_written_ok.all() else None
    tb = np.where(cand_written_ok & (~comm_written_ok))[0]
    tB = int(tb[0]) if tb.size else None
    if tA is not None and (tB is None or tA <= tB):
        return "A", tA
    if tB is not None:
        return "B", tB
    if terminal_ok:
        return "success", -1
    wrong = wrong_regs
    if not wrong.any():
        return "D", -1
    mn = float(np.min(term_norm_margin[wrong]))       # negative for a wrong register
    if mn > -c3_bar:                                  # near the decision boundary
        return "C3", -1
    if bool(np.any(wrong & (~written_ever))):
        return "C1", -1
    return "C2", -1


def horizon_dataset(seed: int, L: int):
    if L == 10:
        return gr.canonical_splits(seed).test_set
    return gr.horizon_test_set(seed, L)


@torch.no_grad()
def collect_horizon(model: torch.nn.Module, ds, seed: int, L: int,
                    store_full: bool) -> Dict[str, np.ndarray]:
    inputs = ds.inputs
    step_tgt = ds.step_targets
    N, T, _ = inputs.shape
    dev = next(model.parameters()).device
    head = model.decision_head
    R, V, D = gr.NUM_REGISTERS, gr.NUM_VALUES, gr.STATE_DIM
    W, bvec = rl._matrix(head)

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, 1)                 # (N,T,D)
    cands = torch.stack(diag.candidates, 1)
    gates = torch.stack(diag.gates, 1)
    mask, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)
    prev_state = torch.cat([torch.zeros(N, 1, D, device=states.device), states[:, :-1]], 1)
    prev_true = torch.zeros_like(step_tgt); prev_true[:, 1:] = step_tgt[:, :-1]

    cp = head(states.reshape(-1, D)).reshape(N, T, R, V).argmax(-1)   # committed decode
    kp = head(cands.reshape(-1, D)).reshape(N, T, R, V).argmax(-1)    # candidate decode

    ar_r = torch.arange(N, device=states.device)[:, None]
    dest = reg_a
    true_written = step_tgt.gather(2, dest[:, :, None]).squeeze(2)     # (N,T) true written value
    cand_pred = kp.gather(2, dest[:, :, None]).squeeze(2)
    comm_pred = cp.gather(2, dest[:, :, None]).squeeze(2)
    cand_written_ok = cand_pred == true_written
    comm_written_ok = comm_pred == true_written
    full_cand_ok = (kp == step_tgt).all(-1)
    full_comm_ok = (cp == step_tgt).all(-1)
    terminal_ok = full_comm_ok[:, -1]
    written_ever = mask.any(1)                                        # (N,4) ever written

    # per-transition decision margins vs strongest competitor at S_prev (dest register)
    def z(Smat):
        return torch.einsum("ntd,ntvd->ntv", Smat, W[dest]) + bvec[dest]
    zp = z(prev_state); zc = z(cands); zk = z(states)
    yv = true_written[:, :, None]
    Mp = zp.gather(2, yv) - zp
    Mc = zc.gather(2, yv) - zc
    Mk = zk.gather(2, yv) - zk
    iy = torch.arange(N, device=states.device)[:, None].expand(N, T)
    it = torch.arange(T, device=states.device)[None, :].expand(N, T)
    Mm = Mp.clone(); Mm[iy, it, true_written] = float("inf")
    a_prev = Mm.argmin(-1)
    ap = a_prev[:, :, None]
    m_prev = Mp.gather(2, ap).squeeze(2); m_cand = Mc.gather(2, ap).squeeze(2)
    m_commit = Mk.gather(2, ap).squeeze(2)
    dm_cand_all = Mc - Mp
    dm_cand = dm_cand_all.gather(2, ap).squeeze(2)
    neg = Mp < 0
    need = torch.where(neg, (-Mp) / dm_cand_all.clamp_min(rl.EPS_ABS), torch.zeros_like(Mp))
    infeasible = neg & (dm_cand_all <= rl.EPS_ABS)
    alpha_full = need.max(-1).values
    alpha_full = torch.where(m_prev >= 0, torch.zeros_like(alpha_full), alpha_full)
    alpha_full = torch.where(infeasible.any(-1), torch.full_like(alpha_full, float("nan")), alpha_full)
    rho = torch.where(dm_cand > rl.EPS_ABS, (m_commit - m_prev) / dm_cand.clamp_min(rl.EPS_ABS),
                      torch.full_like(dm_cand, float("nan")))
    diff = cands - prev_state
    denom = (diff * diff).sum(-1).clamp_min(1e-12)
    lam = (((states - prev_state) * diff).sum(-1) / denom).clamp(-1.0, 2.0)
    offpath_dist = (states - (prev_state + lam[:, :, None] * diff)).norm(dim=-1)
    gate_mean = gates.mean(-1)

    # terminal per-register true-vs-winner normalized margin (for C3)
    final_true = step_tgt[:, -1]                                     # (N,4)
    Wf = W                                                           # (R,V,D)
    zT = torch.einsum("nd,rvd->nrv", states[:, -1], Wf) + bvec[None]  # (N,R,V)
    win = zT.argmax(-1)                                              # (N,R)
    margin = zT.gather(2, final_true[:, :, None]).squeeze(2) - zT.gather(2, win[:, :, None]).squeeze(2)
    ddw = (Wf[torch.arange(R, device=states.device)[None].expand(N, R), final_true]
           - Wf[torch.arange(R, device=states.device)[None].expand(N, R), win])
    norm = ddw.norm(dim=-1).clamp_min(1e-6)
    term_norm_margin = margin / norm                                 # (N,4)

    to = lambda x: x.detach().cpu().numpy()

    # classification
    cw = to(cand_written_ok); km = to(comm_written_ok); tok = to(terminal_ok)
    we = to(written_ever); wrong_regs = to(cp[:, -1] != step_tgt[:, -1])   # (N,4)
    tnm = to(term_norm_margin)
    first_class = np.zeros(N, np.int64); first_t = np.full(N, -1, np.int64)
    n_cand_err = np.zeros(N, np.int64); n_comm_err = np.zeros(N, np.int64); n_full_err = np.zeros(N, np.int64)
    rec_cand = np.zeros(N, np.int64); rec_comm = np.zeros(N, np.int64); rec_full = np.zeros(N, np.int64)
    fcw = to(full_comm_ok); fkd = to(full_cand_ok)
    for i in range(N):
        n_cand_err[i] = int((~cw[i]).sum()); n_comm_err[i] = int((~km[i]).sum())
        n_full_err[i] = int((~fcw[i]).sum())
        cls, tt = classify_first_error(cw[i], km[i], bool(tok[i]), wrong_regs[i], we[i], tnm[i])
        first_class[i] = CLASS_CODE[cls]; first_t[i] = tt
        if tt >= 0 and tt + 1 < T:
            rec_cand[i] = int(cw[i, tt + 1:].any())
            rec_comm[i] = int(km[i, tt + 1:].any())
            rec_full[i] = int(fcw[i, tt + 1:].any())

    out: Dict[str, np.ndarray] = {
        "seq": np.arange(N), "seed": np.full(N, seed), "horizon": np.full(N, L),
        "terminal_ok": tok.astype(np.uint8), "first_class": first_class, "first_t": first_t,
        "n_cand_err": n_cand_err, "n_comm_err": n_comm_err, "n_full_err": n_full_err,
        "rec_cand": rec_cand, "rec_comm": rec_comm, "rec_full": rec_full,
        "written_ever": we.astype(np.uint8),
        "cand_written_ok": cw.astype(np.uint8), "comm_written_ok": km.astype(np.uint8),
        "full_comm_ok": fcw.astype(np.uint8), "full_cand_ok": fkd.astype(np.uint8),
    }
    if store_full:
        out.update({
            "op": to(op).astype(np.int64), "dest": to(dest).astype(np.int64),
            "src_reg": to(reg_b).astype(np.int64),
            "true_written": to(true_written).astype(np.int64),
            "true_state": to(step_tgt).astype(np.int64),
            "operand_a": to(prev_true.gather(2, dest[:, :, None]).squeeze(2)),
            "operand_b": to(prev_true.gather(2, reg_b[:, :, None]).squeeze(2)),
            "wrap": to(((prev_true.gather(2, dest[:, :, None]).squeeze(2)
                         + prev_true.gather(2, reg_b[:, :, None]).squeeze(2)) >= V).float()),
            "cand_margin": to(m_cand), "comm_margin": to(m_commit), "m_prev": to(m_prev),
            "rho": to(rho), "alpha_full": to(alpha_full), "offpath_dist": to(offpath_dist),
            "gate_mean": to(gate_mean), "competitor_changed": to((Mk.argmin(-1) != a_prev).float()),
            "term_norm_margin": to(term_norm_margin), "wrong_regs": wrong_regs.astype(np.uint8),
        })
    return out


def concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = set().union(*[set(p.keys()) for p in parts])
    out = {}
    for k in keys:
        if all(k in p for p in parts):
            out[k] = np.concatenate([p[k] for p in parts], axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment P collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_p_error_budget.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_p_error_budget_meta.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    model_keys = args.models or MODEL_KEYS
    seeds = args.seeds or gr.SEEDS
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "horizons": HORIZONS, "full_array_horizons": FULL_ARRAY_HORIZONS, "c3_bar": C3_BAR,
        "class_codes": CLASS_CODE, "models": {}, "split": "canonical strict test (T=10) + horizon sets",
        "classification": ("earliest observed task-level failure: A candidate written error; "
                           "B candidate-correct/commit-wrong; C no active-write error but terminal "
                           "full-state wrong (C1 untouched, C2 written, C3 near-boundary); D mixed."),
        "note": ("Measurement only: frozen checkpoints, no training, no intervention, no "
                 "counterfactual rollout; all analysis on the actual frozen trajectory."),
    }
    payload: Dict[str, np.ndarray] = {"meta_json": np.array(["placeholder"])}
    meta["n_T10"] = {}; meta["n_T20"] = {}
    for mk in model_keys:
        meta["models"][mk] = {"checkpoints": {}}
        full_parts = {L: [] for L in FULL_ARRAY_HORIZONS}
        horizon_parts = {L: [] for L in HORIZONS}
        for seed in seeds:
            loaded = eg.load_checkpoint(args.checkpoints_dir, mk, seed)
            sha_before = loaded["sha256"]
            model = loaded["model"]
            for L in HORIZONS:
                ds = horizon_dataset(seed, L)
                rec = collect_horizon(model, ds, seed, L, store_full=(L in FULL_ARRAY_HORIZONS))
                horizon_parts[L].append(rec)
                if L in FULL_ARRAY_HORIZONS:
                    full_parts[L].append(rec)
            if eg.sha256_file(loaded["path"]) != sha_before:
                raise RuntimeError(f"{mk} seed={seed}: checkpoint changed during run")
            meta["models"][mk]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {mk} seed={seed}: horizons done")
        for L, parts in full_parts.items():
            c = concat(parts)
            for k, v in c.items():
                payload[f"{mk}__T{L}__{k}"] = v
        for L, parts in horizon_parts.items():
            c = concat(parts)
            payload[f"{mk}__h{L}__first_class"] = c["first_class"]
            payload[f"{mk}__h{L}__terminal_ok"] = c["terminal_ok"]
            payload[f"{mk}__h{L}__seed"] = c["seed"]
        meta["n_T10"][mk] = int(payload[f"{mk}__T10__seq"].shape[0])
        meta["n_T20"][mk] = int(payload[f"{mk}__T20__seq"].shape[0])
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
