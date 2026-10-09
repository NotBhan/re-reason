#!/usr/bin/env python3
"""Experiment U — Raw-Observation / Task-Encoding Localisation (collector).

T concluded T-D: no tested upstream representation was robustly recoverable, the frozen observation
encoder is structurally operand-blind, and the state pathway exposed operand information only
non-generally. U asks the final upstream question:

    Does the RAW TASK INTERFACE itself contain enough causal information to determine the current
    operands / target, and how much observation history is required before that information
    becomes probe-recoverable?

U is measurement-only over the frozen checkpoints: the checkpoints are used ONLY to define the
established Class-A test populations and to reproduce baseline metrics. The raw-observation /
history probes are analysis-only objects in `analyze_experiment_u.py` and never enter model
inference. There is NO model training, no checkpoint modification, no optimizer, no architecture /
encoder / recurrent-core / state / gate / candidate / commit / readout modification, no
intervention, no state replacement, no feature editing, no branching/verifier/oracle, no
counterfactual rollout, no Phase 2. All model execution under `torch.no_grad()`; checkpoint SHA-256
verified before/after.

For every ADD transition this collector stores:
    seq_obs    — the raw benchmark observation sequence x_0..x_{T-1} (T=10, 21-dim each), flattened
    t          — the current timestep
    + established metadata (y_true, raw_sum, operand_a, operand_b, op_dest, src, wrap, repeated,
      cand_ok, cand_pred, seq, seed)

No latent state, reconstructed register values, task labels beyond the recorded targets, candidate
logits, gate values, or hidden model features are stored as probe features.

Outputs
-------
    experiments/experiment_u_task_interface.npz        raw-observation rows + metadata
    experiments/experiment_u_task_interface_meta.json  config, hashes, split sizes
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_u.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_u_task_interface.npz
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402
import run_experiment_l as rl  # noqa: E402
import run_experiment_r as rr  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
ADD = RegisterTrackingTask.OP_ADD
T_MAX = 10                       # canonical benchmark sequence length
OBS_DIM = 21                     # benchmark raw observation dimension
SEQ_FLAT_DIM = T_MAX * OBS_DIM   # 210

# Predeclared causal history grid (spec §4). "full" = the whole causal prefix x_0..x_t, represented
# over the benchmark maximum length T=10.
HISTORY_LENGTHS: List[int] = [1, 2, 4, 8, 16]
FULL_PREFIX_LENGTH = T_MAX       # 10

# Regression control: the two split arrays captured per model.
SPLITS = ["trainord", "valord", "test"]

META_COLS = ["y_true", "raw_sum", "cand_pred", "cand_ok", "m_cand", "offpath", "norm_C", "norm_dC",
             "op_dest", "src", "operand_a", "operand_b", "wrap", "repeated", "t", "seq", "seed"]


@torch.no_grad()
def capture_split(model: torch.nn.Module, ds, seed: int, cap: Optional[int]) -> Dict[str, np.ndarray]:
    """Capture per-ADD-transition raw observation sequences + metadata for one split.

    The frozen model is used only to define the Class-A population (cand_ok) and reproduce the
    frozen baseline; no model feature is stored as a probe input.
    """
    inputs = ds.inputs
    step_targets = ds.step_targets
    B, T, D = inputs.shape
    W, bvec = rl._matrix(model.decision_head)

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, 1)         # (B,T,D_state) committed S_t
    cands = torch.stack(diag.candidates, 1)      # (B,T,D_state) final C_t
    _, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)
    prev_state = torch.cat([torch.zeros(B, 1, states.shape[-1], device=states.device), states[:, :-1]], 1)
    prev_true = torch.zeros_like(step_targets)
    prev_true[:, 1:] = step_targets[:, :-1]

    idx = (op == ADD).nonzero()
    if idx.numel() == 0:
        out: Dict[str, np.ndarray] = {"seq_obs": np.zeros((0, SEQ_FLAT_DIM), np.float32)}
        for k in META_COLS:
            out[k] = np.zeros((0,), np.int64)
        return out
    b, t = idx[:, 0], idx[:, 1]
    if cap is not None:
        idx = idx[:cap]
        b, t = idx[:, 0], idx[:, 1]
    n = int(len(idx))

    dest = reg_a[b, t]
    src = reg_b[b, t]
    oa = prev_true[b, t, dest]
    ob = prev_true[b, t, src]
    y = step_targets[b, t, dest]
    C = cands[b, t]

    ar = torch.arange(n, device=inputs.device)
    z = torch.einsum("nd,nvd->nv", C, W[dest]) + bvec[dest]
    pred = z.argmax(-1)
    cand_ok = pred == y
    wrap = (oa + ob) >= gr.NUM_VALUES
    repeated = (t > 0) & (op[b, (t - 1).clamp_min(0)] == ADD)

    Sp = prev_state[b, t]
    Sc = states[b, t]
    zp = torch.einsum("nd,nvd->nv", Sp, W[dest]) + bvec[dest]
    Mp = zp[ar, y][:, None] - zp
    Mm = Mp.clone()
    Mm[ar, y] = float("inf")
    a_star = Mm.argmin(1)
    m_cand = (z[ar, y][:, None] - z).gather(1, a_star[:, None]).squeeze(1)
    dvec = C - Sp
    denom = (dvec * dvec).sum(-1).clamp_min(1e-12)
    lam = ((Sc - Sp) * dvec).sum(-1) / denom
    offpath = (Sc - (Sp + lam[:, None] * dvec)).norm(dim=1)

    seq_obs = inputs.reshape(B, T * D)[b]                                  # (n, 210) raw x_0..x_{T-1}
    to = lambda x: x.detach().cpu().numpy()
    return {
        "seq_obs": seq_obs.float().numpy().astype(np.float32),
        "y_true": to(y).astype(np.int64), "raw_sum": to(oa + ob).astype(np.int64),
        "cand_pred": to(pred).astype(np.int64), "cand_ok": to(cand_ok).astype(np.uint8),
        "m_cand": to(m_cand).astype(np.float32), "offpath": to(offpath).astype(np.float32),
        "norm_C": to(C.norm(dim=1)).astype(np.float32), "norm_dC": to(dvec.norm(dim=1)).astype(np.float32),
        "op_dest": to(dest).astype(np.int64), "src": to(src).astype(np.int64),
        "operand_a": to(oa).astype(np.int64), "operand_b": to(ob).astype(np.int64),
        "wrap": to(wrap.float()).astype(np.int64), "repeated": to(repeated.float()).astype(np.int64),
        "t": to(t).astype(np.int64), "seq": to(b).astype(np.int64),
        "seed": np.full(n, seed, np.int64),
    }


def concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], 0) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment U collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_u_task_interface.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_u_task_interface_meta.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    model_keys = args.models or MODEL_KEYS
    seeds = args.seeds or gr.SEEDS
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "train_ord_cap": rr.TRAIN_ORD_CAP, "val_ord_cap": rr.VAL_ORD_CAP,
        "obs_schema": ("21-dim one-hot: op[0:3] (SET=0, SWAP=1, ADD=2), reg_a[3:7], reg_b[7:11], "
                       "immediate[11:21]"),
        "history_lengths": HISTORY_LENGTHS, "full_prefix_length": FULL_PREFIX_LENGTH,
        "obs_dim": OBS_DIM, "seq_flat_dim": SEQ_FLAT_DIM,
        "models": {},
        "splits": {"trainord": "DisjointSplitManager train ADD subsample (cap 1500/seed)",
                   "valord": "DisjointSplitManager val ADD subsample (cap 800/seed)",
                   "test": "canonical strict test split (1200 seq, stream 90042+seed), all ADD"},
        "hard_tail": ("ADD transition with frozen candidate written-register decode != true written "
                      "value (Q's Class-A definition, unchanged)"),
        "note": ("Measurement only: frozen checkpoints define the Class-A population and reproduce "
                 "the baseline; the raw-observation/history probes are analysis-only and never enter "
                 "model inference. No model feature is stored as a probe input."),
    }
    payload: Dict[str, np.ndarray] = {"meta_json": np.array(["placeholder"])}
    for mk in model_keys:
        meta["models"][mk] = {"checkpoints": {}}
        acc: Dict[str, List[Dict[str, np.ndarray]]] = {k: [] for k in SPLITS}
        for seed in seeds:
            loaded = eg.load_checkpoint(args.checkpoints_dir, mk, seed)
            sha_before = loaded["sha256"]
            model = loaded["model"]
            splits = gr.canonical_splits(seed)
            full_tr = capture_split(model, splits.train_set, seed, cap=None)
            full_va = capture_split(model, splits.val_set, seed, cap=None)
            full_te = capture_split(model, splits.test_set, seed, cap=None)
            n = len(full_tr["y_true"])
            ord_tr = np.arange(min(rr.TRAIN_ORD_CAP, n))
            m = len(full_va["y_true"])
            ord_va = np.arange(min(rr.VAL_ORD_CAP, m))
            acc["trainord"].append({k: v[ord_tr] for k, v in full_tr.items()})
            acc["valord"].append({k: v[ord_va] for k, v in full_va.items()})
            acc["test"].append(full_te)
            if eg.sha256_file(loaded["path"]) != sha_before:
                raise RuntimeError(f"{mk} seed={seed}: checkpoint changed during run")
            meta["models"][mk]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {mk} seed={seed}: train_ord={len(ord_tr)} val_ord={len(ord_va)} "
                  f"test_add={len(full_te['y_true'])} classA={int((full_te['cand_ok'] == 0).sum())}")
        for k, parts in acc.items():
            c = concat(parts)
            for a, v in c.items():
                payload[f"{mk}__{k}__{a}"] = v

    meta["n"] = {mk: {k: int(payload[f"{mk}__{k}__y_true"].shape[0]) for k in SPLITS}
                 for mk in model_keys}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
