#!/usr/bin/env python3
"""Experiment W — Bounded-Reader Learnability vs Training-Data Scale (collector).

V (V-B, sequence-limited) showed that a parameter-bounded causal GRU and causal TCN — like U's flat
MLP-128 — do not access the raw-interface information that is exactly identifiable from a 4-step
causal history. V left one narrow question open: is the persistent reader failure primarily a
training-data / statistical-learning limitation, an optimization limitation at the fixed protocol, or
a bounded-capacity / algorithmic limitation? W measures **training-data scale only**.

Reader architectures, widths, optimizer, epochs, history representation, held-out protocol, splits
and populations are all REUSED UNCHANGED from U/V. The ONLY experimental factor is TRAIN size
(the active ladder is read from `TRAIN_SIZES`, i.e. the `W_TRAIN_SIZES` env var; the default is the
predeclared full ladder N_train in {1500, 8000, 32000}).

This collector produces, per frozen E seed:
  * three **independent, pairwise sequence-disjoint** fresh i.i.d. ordinary-ADD training pools of
    exactly 1500 / 8000 / 32000 transitions (distinct RNG streams; each pool dedups sequences
    internally; all pools are sequence-disjoint from the canonical VAL/TEST and from each other);
  * the canonical VAL ordinary-ADD population (the U/V `valord` protocol, fixed across scales);
  * the canonical untouched TEST (per model E and F), used to define the Class-A population and the
    candidate-correct controls.

Collector is measurement-only: the frozen E/F checkpoints are used ONLY to define the Class-A test
population and reproduce baseline metrics; the W readers are analysis-only and never enter model
inference. No model training, no weight/state modification, no intervention, no Phase 2. All model
execution under `torch.no_grad()`; checkpoint SHA-256 verified before/after.

Outputs
-------
    experiments/experiment_w_data_scale.npz        fresh train pools + VAL + canonical TEST rows
    experiments/experiment_w_data_scale_meta.json  config, hashes, disjointness, sizes
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_w.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_w_data_scale.npz
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
import run_experiment_r as rr  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.dataset_split import hash_sequence  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
ADD = RegisterTrackingTask.OP_ADD
T_MAX = ru.T_MAX
OBS_DIM = ru.OBS_DIM

# Predeclared training-data scales (the ONLY experimental factor). Defaults are the full declared
# ladder {1500, 8000, 32000}; a reduced ladder can be selected via W_TRAIN_SIZES for a fast, still
# methodologically complete evaluation on modest hardware (the analysis reads this list).
_DEFAULT_SIZES = [1500, 8000, 32000]
TRAIN_SIZES: List[int] = [int(x) for x in
                          os.environ.get("W_TRAIN_SIZES", ",".join(map(str, _DEFAULT_SIZES))).split(",")]
# Independent fresh RNG stream bases/offsets (disjoint from canonical train/val/test and from each
# other). One offset slot per size keeps the pools on separate streams -> pairwise sequence-disjoint.
POOL_STREAM_BASE = 2_000_000
POOL_STREAM_OFFSETS = {s: i * 1_000_000 for i, s in enumerate(TRAIN_SIZES)}

POOL_COLS = ["seq_obs", "t", "y_true", "raw_sum", "operand_a", "operand_b", "op_dest",
             "wrap", "repeated", "seq", "seed"]
TEST_COLS = POOL_COLS + ["cand_ok", "m_cand", "norm_C", "norm_dC"]


def make_pool(seed: int, size: int, stream: int) -> Dict[str, np.ndarray]:
    """Fresh i.i.d. ordinary-ADD pool of exactly `size` transitions from an independent stream."""
    task = gr.build_task(T_MAX)
    rng = np.random.default_rng(stream + seed)
    seen = set()
    out: Dict[str, List[Any]] = {k: [] for k in POOL_COLS}
    seq_id = 0
    while len(out["y_true"]) < size:
        inp, st, ft = task.generate_single_sequence(rng)
        h = hash_sequence(inp)
        if h in seen:
            continue                      # dedup within the pool
        seen.add(h)
        op = inp[:, :3].argmax(1)
        ra = inp[:, 3:7].argmax(1)
        rb = inp[:, 7:11].argmax(1)
        prev = np.zeros_like(st)
        prev[1:] = st[:-1]
        flat = inp.reshape(-1).astype(np.float32)
        for tt in range(T_MAX):
            if int(op[tt]) != ADD:
                continue
            a = int(prev[tt, ra[tt]])
            b = int(prev[tt, rb[tt]])
            out["seq_obs"].append(flat)
            out["t"].append(tt)
            out["y_true"].append((a + b) % gr.NUM_VALUES)
            out["raw_sum"].append(a + b)
            out["operand_a"].append(a)
            out["operand_b"].append(b)
            out["op_dest"].append(int(ra[tt]))
            out["wrap"].append(int((a + b) >= gr.NUM_VALUES))
            out["repeated"].append(int(tt > 0 and int(op[tt - 1]) == ADD))
            out["seq"].append(seq_id)
            out["seed"].append(seed)
            if len(out["y_true"]) >= size:
                break
        seq_id += 1
    return {
        "seq_obs": np.stack(out["seq_obs"]).astype(np.float32),
        "t": np.asarray(out["t"], np.int64), "y_true": np.asarray(out["y_true"], np.int64),
        "raw_sum": np.asarray(out["raw_sum"], np.int64),
        "operand_a": np.asarray(out["operand_a"], np.int64),
        "operand_b": np.asarray(out["operand_b"], np.int64),
        "op_dest": np.asarray(out["op_dest"], np.int64),
        "wrap": np.asarray(out["wrap"], np.int64), "repeated": np.asarray(out["repeated"], np.int64),
        "seq": np.asarray(out["seq"], np.int64), "seed": np.asarray(out["seed"], np.int64),
    }


def _seq_hashes(arr: np.ndarray) -> set:
    return {hash_sequence(arr[i].reshape(T_MAX, OBS_DIM)) for i in range(arr.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment W collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_w_data_scale.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_w_data_scale_meta.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    model_keys = args.models or MODEL_KEYS
    seeds = args.seeds or gr.SEEDS
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "train_sizes": TRAIN_SIZES, "history_lengths": ru.HISTORY_LENGTHS,
        "full_prefix_length": ru.FULL_PREFIX_LENGTH,
        "pool_stream_base": POOL_STREAM_BASE, "pool_stream_offsets": POOL_STREAM_OFFSETS,
        "train_ord_cap": rr.TRAIN_ORD_CAP, "val_ord_cap": rr.VAL_ORD_CAP,
        "obs_schema": ("21-dim one-hot: op[0:3] (SET=0, SWAP=1, ADD=2), reg_a[3:7], reg_b[7:11], "
                       "immediate[11:21]"),
        "pool_design": ("three INDEPENDENT, pairwise sequence-disjoint fresh i.i.d. ordinary-ADD "
                        f"pools per seed (sizes {'/'.join(map(str, TRAIN_SIZES))}) from distinct "
                        "RNG streams; NOT nested and NOT recycled/duplicated"),
        "val": "canonical DisjointSplitManager VAL split, ordinary ADD subsample (cap 800/seed); fixed across scales",
        "test": "canonical strict test split (1200 seq, stream 90042+seed), all ADD; untouched (identical across scales)",
        "models": {},
        "disjointness": {},
    }
    payload: Dict[str, np.ndarray] = {"meta_json": np.array(["placeholder"])}
    for mk in model_keys:
        meta["models"][mk] = {"checkpoints": {}}
    acc = {S: {c: [] for c in POOL_COLS} for S in TRAIN_SIZES}
    acc_val = {c: [] for c in POOL_COLS}
    acc_test = {mk: {c: [] for c in TEST_COLS} for mk in model_keys}
    for seed in seeds:
        # fresh independent pools
        pools = {S: make_pool(seed, S, POOL_STREAM_BASE + POOL_STREAM_OFFSETS[S]) for S in TRAIN_SIZES}
        for S in TRAIN_SIZES:
            for col in POOL_COLS:
                acc[S][col].append(pools[S][col])
        # disjointness checks (sequence-hash overlap): pools vs each other and vs canonical VAL/TEST
        hs = {S: _seq_hashes(pools[S]["seq_obs"]) for S in TRAIN_SIZES}
        canon = gr.canonical_splits(seed)
        val_h = set(canon.val_set.hashes)
        test_h = set(canon.test_set.hashes)
        pool_h = set().union(*hs.values())
        pair_overlaps = {}
        for i in range(len(TRAIN_SIZES)):
            for j in range(i + 1, len(TRAIN_SIZES)):
                a, b = TRAIN_SIZES[i], TRAIN_SIZES[j]
                pair_overlaps[f"pool{a}_vs_{b}"] = len(hs[a] & hs[b])
        meta["disjointness"][str(seed)] = {
            **pair_overlaps,
            "pools_vs_val": len(pool_h & val_h),
            "pools_vs_test": len(pool_h & test_h),
            "pool_unique_sequences": {str(S): len(hs[S]) for S in TRAIN_SIZES}}
        # canonical VAL (once; model-independent) and canonical TEST (per model)
        loaded = eg.load_checkpoint(args.checkpoints_dir, model_keys[0], seed)
        sha_before = loaded["sha256"]
        splits = gr.canonical_splits(seed)
        full_va = ru.capture_split(loaded["model"], splits.val_set, seed, cap=None)
        m = len(full_va["y_true"])
        ord_va = np.arange(min(rr.VAL_ORD_CAP, m))
        for col in POOL_COLS:
            acc_val[col].append(full_va[col][ord_va])
        if eg.sha256_file(loaded["path"]) != sha_before:
            raise RuntimeError(f"{model_keys[0]} seed={seed}: checkpoint changed during run")
        for mk in model_keys:
            loaded = eg.load_checkpoint(args.checkpoints_dir, mk, seed)
            sha_before = loaded["sha256"]
            full_te = ru.capture_split(loaded["model"], splits.test_set, seed, cap=None)
            for col in TEST_COLS:
                acc_test[mk][col].append(full_te[col])
            if eg.sha256_file(loaded["path"]) != sha_before:
                raise RuntimeError(f"{mk} seed={seed}: checkpoint changed during run")
            meta["models"][mk]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {mk} seed={seed}: test_add={len(full_te['y_true'])} "
                  f"classA={int((full_te['cand_ok'] == 0).sum())}")
        print(f"  seed={seed}: pools " + "/".join(str(len(pools[S]['y_true'])) for S in TRAIN_SIZES)
              + f" val={len(ord_va)}")

    for S in TRAIN_SIZES:
        for col in POOL_COLS:
            payload[f"pool{S}__{col}"] = np.concatenate(acc[S][col], 0)
    for col in POOL_COLS:
        payload[f"val__{col}"] = np.concatenate(acc_val[col], 0)
    for mk in model_keys:
        for col in TEST_COLS:
            payload[f"{mk}__test__{col}"] = np.concatenate(acc_test[mk][col], 0)

    meta["n"] = {"val": int(payload["val__y_true"].shape[0])}
    for S in TRAIN_SIZES:
        meta["n"][f"pool{S}"] = int(payload[f"pool{S}__y_true"].shape[0])
    for mk in model_keys:
        meta["n"][f"{mk}__test"] = int(payload[f"{mk}__test__y_true"].shape[0])
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
