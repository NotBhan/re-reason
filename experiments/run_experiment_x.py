#!/usr/bin/env python3
"""Experiment X — Fixed-Data Optimization-Budget Sensitivity (collector).

X asks a single question that W left open: W fit three bounded, analysis-only readers (U's flat
MLP-128, V's causal GRU, V's causal TCN) on 1,500 / 4,000 / 10,000 unique sequence-disjoint
ordinary-ADD transitions with a FIXED 200 full-batch updates, saw only weak Class-A target
accuracy, and also saw LOW training accuracy (underfitting). Is that weakness a fixed-protocol
**optimization** limit, or a **bounded-reader** limit, when the data, model and every other
training condition are held fixed and only the number of full-batch optimizer updates changes
(200 vs 1,000)?

X is measurement-only and manipulates exactly ONE factor: `optimizer_updates ∈ {200, 1000}`.
Everything else — the training pool, reader architectures/widths, optimizer, LR, weight decay,
loss, full-batch regime, history representations, targets/encodings, VAL/TEST splits, held-out
operand protocol, deterministic seeds — is FIXED.

Data (fixed): X reuses **W's existing fresh 10,000-transition ordinary-ADD pool per E seed**, W's
canonical ordinary VAL, and W's untouched canonical TEST (E and F). Nothing is regenerated. This
collector copies those arrays verbatim out of W's stored artifact and verifies byte identity with
SHA-256, so X's fixed-data snapshot cannot silently diverge from W.

Outputs
-------
    experiments/experiment_x_optimization_budget.npz        fixed pool10000 + VAL + E/F TEST
    experiments/experiment_x_optimization_budget_meta.json  identity hashes, counts, disjointness
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_x.py \
        --w-npz experiments/experiment_w_data_scale.npz \
        --output experiments/experiment_x_optimization_budget.npz
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_experiment_u as au  # noqa: E402  (held-out operand pairs; single source of truth)
import run_experiment_u as ru  # noqa: E402
import run_experiment_w as rw  # noqa: E402
from re_reason.benchmarks.dataset_split import hash_sequence  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
HELD_OUT_PAIRS = au.HELD_OUT_PAIRS
POOL_COLS = rw.POOL_COLS
TEST_COLS = rw.TEST_COLS
T_MAX = ru.T_MAX
OBS_DIM = ru.OBS_DIM
# The fixed data scale is W's TOP-scale pool. Read from W's artifact rather than hardcoded.
TOP_SIZE_ENV = os.environ.get("X_TOP_SIZE", "")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _seq_hashes(arr: np.ndarray) -> set:
    return {hash_sequence(arr[i].reshape(T_MAX, OBS_DIM)) for i in range(arr.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment X fixed-data collector (measurement only)")
    ap.add_argument("--w-npz", default="experiments/experiment_w_data_scale.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_x_optimization_budget.npz")
    ap.add_argument("--meta-output",
                    default="experiments/experiment_x_optimization_budget_meta.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    if not os.path.exists(args.w_npz):
        raise SystemExit(f"W artifact not found: {args.w_npz}. Run run_experiment_w.py first.")
    wz = np.load(args.w_npz, allow_pickle=False)
    w_meta = json.loads(str(wz["meta_json"][0]))
    ladder = list(w_meta["train_sizes"])
    top = int(ladder[-1]) if not TOP_SIZE_ENV else int(TOP_SIZE_ENV)
    if top not in ladder:
        raise SystemExit(f"top size {top} not in W ladder {ladder}")
    if set(w_meta["n"]) < {f"pool{top}", "val"}:
        raise SystemExit(f"W artifact missing pool{top}/val in meta")

    meta: Dict[str, Any] = {
        "factor": "optimizer_updates",
        "budgets": [200, 1000],
        "source": {
            "npz": args.w_npz,
            "sha256": sha256_file(args.w_npz),
            "train_sizes": ladder,
            "top_size": top,
            "pool_stream_base": w_meta.get("pool_stream_base"),
            "pool_stream_offsets": w_meta.get("pool_stream_offsets"),
            "date": w_meta.get("date"),
        },
        "obs_schema": w_meta.get("obs_schema"),
        "pool_design": (f"reuses W's fresh top-scale pool{top} (pairwise sequence-disjoint i.i.d. "
                        "ordinary-ADD); NOT regenerated"),
        "val": "W's canonical ordinary VAL (cap 800/seed); identical across X conditions",
        "test": "W's untouched canonical strict TEST (all ADD); identical across X conditions",
        "models": {},
        "identity": {},
        "disjointness": {},
    }

    # ---- copy the fixed arrays verbatim ----
    payload: Dict[str, np.ndarray] = {}
    for c in POOL_COLS:
        payload[f"pool{top}__{c}"] = wz[f"pool{top}__{c}"]
        payload[f"val__{c}"] = wz[f"val__{c}"]
    for mk in MODEL_KEYS:
        for c in TEST_COLS:
            payload[f"{mk}__test__{c}"] = wz[f"{mk}__test__{c}"]

    # ---- per-array identity (byte-level) against W ----
    for k, v in payload.items():
        assert np.array_equal(v, wz[k]), f"copy mismatch for {k}"
        meta["identity"][k] = {"sha256": sha256_array(v),
                               "shape": list(v.shape), "dtype": str(v.dtype)}

    # ---- counts / uniqueness / duplicate / held-out / disjointness (per seed) ----
    seeds = sorted(int(s) for s in w_meta["disjointness"].keys())
    meta["seeds"] = seeds
    meta["n"] = {"pool": int(payload[f"pool{top}__y_true"].shape[0]),
                 "val": int(payload["val__y_true"].shape[0])}
    for mk in MODEL_KEYS:
        meta["n"][f"{mk}__test"] = int(payload[f"{mk}__test__y_true"].shape[0])
    meta["per_seed"] = {}
    for s in seeds:
        rows = np.nonzero(payload[f"pool{top}__seed"] == s)[0]
        seqids = payload[f"pool{top}__seq"][rows]
        fp = {(int(payload[f"pool{top}__seq"][r]), int(payload[f"pool{top}__t"][r])) for r in rows}
        oa = payload[f"pool{top}__operand_a"][rows]
        ob = payload[f"pool{top}__operand_b"][rows]
        held = int(sum(1 for a, b in zip(oa, ob) if (int(a), int(b)) in HELD_OUT_PAIRS))
        meta["per_seed"][str(s)] = {
            "n_train": int(len(rows)),
            "n_unique_sequences": int(len(set(int(x) for x in seqids))),
            "n_unique_seq_t": int(len(fp)),
            "n_duplicate_rows": int(len(rows) - len(fp)),
            "n_heldout_pair_rows": held,
            "classA": int(((payload[f"{MODEL_KEYS[0]}__test__seed"] == s)
                           & (payload[f"{MODEL_KEYS[0]}__test__cand_ok"] == 0)).sum()),
        }
    # disjointness: pool vs VAL / TEST (exact sequence-hash overlap), pooled over seeds
    pool_h = _seq_hashes(payload[f"pool{top}__seq_obs"])
    val_h = _seq_hashes(payload["val__seq_obs"])
    meta["disjointness"] = {"pool_vs_val": len(pool_h & val_h)}
    for mk in MODEL_KEYS:
        test_h = _seq_hashes(payload[f"{mk}__test__seq_obs"])
        meta["disjointness"][f"pool_vs_{mk}_test"] = len(pool_h & test_h)
        meta["disjointness"][f"val_vs_{mk}_test"] = len(val_h & test_h)

    # ---- checkpoint hashes (frozen E/F; unchanged by this measurement-only collector) ----
    for mk in MODEL_KEYS:
        meta["models"][mk] = {"checkpoints": {}}
        for s in seeds:
            p = os.path.join(args.checkpoints_dir, f"{mk}_seed{s}_3000steps.pt")
            meta["models"][mk]["checkpoints"][str(s)] = {"path": p, "sha256": sha256_file(p)}

    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Copied {len(payload) - 1} fixed arrays from W ({args.w_npz})")
    print(f"pool={meta['n']['pool']} (top={top}) val={meta['n']['val']} "
          f"test(E)={meta['n']['E_candidate__test']} test(F)={meta['n']['F_combined__test']}")
    print(f"disjointness={meta['disjointness']}")
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
