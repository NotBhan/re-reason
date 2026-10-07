#!/usr/bin/env python3
"""Experiment Q — Candidate-Generation Origin / Recoverability (measurement only).

P showed the sequence-level first-error budget is dominated by candidate-generation errors
(Class A). Q asks whether that dominant failure is a candidate-REPRESENTATION/arithmetic limit or
more consistent with exposure/decoding/training alignment, by measuring how much of the true
written value is *linearly recoverable from the frozen candidate vector C_t* on the transitions
where the frozen candidate head is wrong.

Three distinct quantities (never conflated):
  1. frozen-head candidate error        : head(C_t) is wrong
  2. linear recoverability              : an independent analysis-only linear probe C_t -> value succeeds
  3. nonlinear information recoverability: a small fixed-capacity nonlinear probe recovers it

Measurement only: frozen checkpoints, no training of the model, no optimizer on model weights, no
architecture/candidate/core/gate/commit/readout modification, no branching/verifier/oracle, no
counterfactual rollout. The ONLY fitted objects are analysis-only probes trained on the disjoint
TRAIN split, tuned on the VAL split, and evaluated once on the untouched TEST split; they never
enter model inference or any rollout. All model execution is under torch.no_grad().

Outputs
-------
    experiments/experiment_q_candidate_origin.npz        frozen C_t (train/val/test) + metadata + W,b
    experiments/experiment_q_candidate_origin_meta.json  config, hashes, split sizes
Reproduce
---------
    python experiments/run_experiment_q.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_q_candidate_origin.npz
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402
import run_experiment_l as rl  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
TRAIN_CAP = 1500        # deterministic subsample of train ADD transitions (probe fitting)
VAL_CAP = 800           # deterministic subsample of val ADD transitions (probe tuning)
ADD = RegisterTrackingTask.OP_ADD


@torch.no_grad()
def capture_split(model: torch.nn.Module, ds, seed: int, cap: Optional[int]) -> Dict[str, np.ndarray]:
    """Capture per-ADD-transition candidate vector C_t + metadata for one split (test: cap=None)."""
    inputs = ds.inputs; step_targets = ds.step_targets
    B, T, _ = inputs.shape
    head = model.decision_head
    R, V, D = gr.NUM_REGISTERS, gr.NUM_VALUES, gr.STATE_DIM
    W, bvec = rl._matrix(head)

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, 1)
    cands = torch.stack(diag.candidates, 1)
    _, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)
    prev_state = torch.cat([torch.zeros(B, 1, D, device=states.device), states[:, :-1]], 1)
    prev_true = torch.zeros_like(step_targets); prev_true[:, 1:] = step_targets[:, :-1]

    idx = (op == ADD).nonzero()
    b, t = idx[:, 0], idx[:, 1]
    if cap is not None:
        idx = idx[:cap]; b, t = idx[:, 0], idx[:, 1]
    n = int(len(idx))
    C = cands[b, t]; Sp = prev_state[b, t]; Sc = states[b, t]
    dest = reg_a[b, t]; src = reg_b[b, t]
    oa = prev_true[b, t, dest]; ob = prev_true[b, t, src]
    y = step_targets[b, t, dest]
    wrap = (oa + ob) >= V
    repeated = (t > 0) & (op[b, (t - 1).clamp_min(0)] == ADD)

    ar = torch.arange(n, device=states.device)
    z = torch.einsum("nd,nvd->nv", C, W[dest]) + bvec[dest]         # frozen candidate logits at dest
    pred = z.argmax(-1)
    cand_ok = pred == y

    zp = torch.einsum("nd,nvd->nv", Sp, W[dest]) + bvec[dest]
    Mp = zp[ar, y][:, None] - zp
    Mm = Mp.clone(); Mm[ar, y] = float("inf")
    a_star = Mm.argmin(1)
    m_cand = (z[ar, y][:, None] - z).gather(1, a_star[:, None]).squeeze(1)
    dvec = C - Sp
    denom = (dvec * dvec).sum(-1).clamp_min(1e-12)
    lam = ((Sc - Sp) * dvec).sum(-1) / denom
    offpath = (Sc - (Sp + lam[:, None] * dvec)).norm(dim=1)

    to = lambda x: x.detach().cpu().numpy()
    return {
        "C": to(C.float()),
        "y_true": to(y).astype(np.int64), "raw_sum": to((oa + ob)).astype(np.int64),
        "cand_pred": to(pred).astype(np.int64), "cand_ok": to(cand_ok).astype(np.uint8),
        "m_cand": to(m_cand), "offpath": to(offpath), "norm_C": to(C.norm(dim=1)),
        "norm_dC": to(dvec.norm(dim=1)),
        "op_dest": to(dest).astype(np.int64), "src": to(src).astype(np.int64),
        "operand_a": to(oa).astype(np.int64), "operand_b": to(ob).astype(np.int64),
        "wrap": to(wrap.float()).astype(np.int64), "repeated": to(repeated.float()).astype(np.int64),
        "t": to(t).astype(np.int64), "seq": to(b).astype(np.int64),
        "seed": np.full(n, seed, np.int64),
    }


def concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = set(parts[0].keys())
    return {k: np.concatenate([p[k] for p in parts], 0) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment Q collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_q_candidate_origin.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_q_candidate_origin_meta.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    model_keys = args.models or MODEL_KEYS
    seeds = args.seeds or gr.SEEDS
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "train_cap": TRAIN_CAP, "val_cap": VAL_CAP, "models": {},
        "splits": {"train": "DisjointSplitManager train (6000 seq)", "val": "DisjointSplitManager val (1200 seq)",
                   "test": "canonical strict test split (1200 seq, stream 90042+seed)"},
        "readout": "frozen TypedDecisionHead (single nn.Linear); W,b stored per seed, unchanged",
        "note": ("Measurement only: frozen checkpoints, no model training, no intervention, no "
                 "counterfactual rollout. Analysis-only probes are fit on TRAIN, tuned on VAL, "
                 "evaluated once on TEST; they never enter model inference."),
    }
    payload: Dict[str, np.ndarray] = {"meta_json": np.array(["placeholder"])}
    for mk in model_keys:
        meta["models"][mk] = {"checkpoints": {}}
        for split, cap in (("train", TRAIN_CAP), ("val", VAL_CAP), ("test", None)):
            parts = []; Ws = []; bs = []
            for seed in seeds:
                loaded = eg.load_checkpoint(args.checkpoints_dir, mk, seed)
                sha_before = loaded["sha256"]
                model = loaded["model"]
                splits = gr.canonical_splits(seed)
                ds = {"train": splits.train_set, "val": splits.val_set, "test": splits.test_set}[split]
                parts.append(capture_split(model, ds, seed, cap))
                W, bvec = rl._matrix(model.decision_head)
                Ws.append(W.detach().cpu().numpy().astype(np.float32))
                bs.append(bvec.detach().cpu().numpy().astype(np.float32))
                if eg.sha256_file(loaded["path"]) != sha_before:
                    raise RuntimeError(f"{mk} seed={seed}: checkpoint changed during run")
                meta["models"][mk]["checkpoints"][str(seed)] = {
                    "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            c = concat(parts)
            for k, v in c.items():
                payload[f"{mk}__{split}__{k}"] = v
            if split == "test":
                payload[f"{mk}__W"] = np.stack(Ws); payload[f"{mk}__b"] = np.stack(bs)
            print(f"  {mk} {split}: {c['C'].shape[0]} ADD transitions")

    meta["n_test"] = {mk: int(payload[f"{mk}__test__C"].shape[0]) for mk in model_keys}
    meta["n_train"] = {mk: int(payload[f"{mk}__train__C"].shape[0]) for mk in model_keys}
    meta["n_val"] = {mk: int(payload[f"{mk}__val__C"].shape[0]) for mk in model_keys}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
