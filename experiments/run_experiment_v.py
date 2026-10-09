#!/usr/bin/env python3
"""Experiment V — Bounded Sequence-Aware Access Test (collector).

U (U-D) established that the raw causal observation history uniquely determines the ADD
operands/target (target Bayes 0.290 -> 0.998 at L=4 -> 1.000 at the full prefix) yet the bounded
FLAT probe family (linear / MLP-16..128 on the flattened one-hot window) stays weak (best
full-prefix 0.218 all-ADD, ~0.09 Class-A; held-out <= 0.25). V asks exactly one remaining question:

    Does a parameter-bounded causal SEQUENCE-AWARE reader recover the information that is exactly
    identifiable from the raw L=4 history, or does the U-D gap survive when temporal structure is
    represented explicitly?

V is measurement-only. The frozen E/F checkpoints are used ONLY to define the Class-A population and
reproduce baseline metrics; the V readers are separate analysis-only models and never enter frozen
model inference. There is NO model training of RE:Reason, no optimizer on RE:Reason parameters, no
checkpoint/architecture/state/candidate/commit/readout modification, no state replacement, no feature
editing, no oracle/reconstructed state as a reader feature, no branching/verifier, no counterfactual
rollout, no Phase 2. All model execution under `torch.no_grad()`; checkpoint SHA-256 verified
before/after.

Raw-observation collection is REUSED VERBATIM from Experiment U (`run_experiment_u.capture_split`),
so the collected arrays are bit-identical to U's (verified by hashing the shared columns). Per ADD
transition V stores the raw observation sequence `x_0..x_{T-1}` (T=10, 21-dim each, flattened) plus
the same metadata columns U stores. No latent state, reconstructed register values, task labels
beyond the recorded targets, candidate logits, gate values, or hidden model features are stored.

Outputs
-------
    experiments/experiment_v_sequence_access.npz        raw-observation rows + metadata
    experiments/experiment_v_sequence_access_meta.json  config, hashes, split sizes
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_v.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_v_sequence_access.npz
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402
import run_experiment_r as rr  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402

MODEL_KEYS = ru.MODEL_KEYS
SPLITS = ru.SPLITS
META_COLS = ru.META_COLS
T_MAX = ru.T_MAX
OBS_DIM = ru.OBS_DIM
SEQ_FLAT_DIM = ru.SEQ_FLAT_DIM
FULL_PREFIX_LENGTH = ru.FULL_PREFIX_LENGTH


def concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], 0) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment V collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_v_sequence_access.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_v_sequence_access_meta.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    model_keys = args.models or MODEL_KEYS
    seeds = args.seeds or gr.SEEDS
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "train_ord_cap": rr.TRAIN_ORD_CAP, "val_ord_cap": rr.VAL_ORD_CAP,
        "obs_schema": ("21-dim one-hot: op[0:3] (SET=0, SWAP=1, ADD=2), reg_a[3:7], reg_b[7:11], "
                       "immediate[11:21]; V readers receive x_t (21) + 1 validity bit = 22 features"),
        "history_lengths": ru.HISTORY_LENGTHS, "full_prefix_length": FULL_PREFIX_LENGTH,
        "obs_dim": OBS_DIM, "seq_flat_dim": SEQ_FLAT_DIM,
        "collection": ("raw-observation collection reused verbatim from Experiment U "
                       "(run_experiment_u.capture_split); arrays are bit-identical to U's"),
        "models": {},
        "splits": {"trainord": "DisjointSplitManager train ADD subsample (cap 1500/seed)",
                   "valord": "DisjointSplitManager val ADD subsample (cap 800/seed)",
                   "test": "canonical strict test split (1200 seq, stream 90042+seed), all ADD"},
        "hard_tail": ("ADD transition with frozen candidate written-register decode != true written "
                      "value (Q's Class-A definition, unchanged)"),
        "note": ("Measurement only: frozen checkpoints define the Class-A population and reproduce "
                 "the baseline; the V sequence readers are analysis-only and never enter model "
                 "inference. No model feature is stored as a reader input."),
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
            full_tr = ru.capture_split(model, splits.train_set, seed, cap=None)
            full_va = ru.capture_split(model, splits.val_set, seed, cap=None)
            full_te = ru.capture_split(model, splits.test_set, seed, cap=None)
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
