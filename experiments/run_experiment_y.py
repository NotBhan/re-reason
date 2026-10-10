#!/usr/bin/env python3
"""Experiment Y — Bounded-Reader Fitting Control (collector).

X (X-B) found that the three bounded, analysis-only readers (U's flat MLP-128, V's causal GRU, V's
causal TCN) do NOT fit even the fixed 10,000-transition training pool (training accuracy 0.07-0.15,
near/below the 0.10 chance) under the fixed 200/1,000-update protocol. That single fact can separate
"the ADD task is the obstacle" from "the reader training pipeline is the obstacle", and it is what Y
measures.

Y is measurement-only and manipulates exactly ONE factor for the fitting diagnostic: the fitting set
is a **deterministic, fixed, class-balanced small subset of 256 unique ordinary-ADD transitions**,
drawn from W's existing 10,000-transition top-scale pool per E seed. Everything else — the reader
architectures/widths, optimizer, LR, weight decay, loss, full-batch regime, history
representations, targets, VAL/TEST splits, held-out operand protocol and deterministic seeds — is
FIXED and reused from U/V/W/X.

Y does not modify RE:Reason, does not train it, does not introduce Phase 2, and does not touch any
completed report Q-X. The readers are separate analysis-only models.

Subset design (predeclared):
  * 256 unique ADD transitions per seed, stratified by the true written-value target `y_true` over
    its 10 classes as evenly as possible (base 25/class + 1 to the first 6 classes = 6x26 + 4x25).
  * Operand pairs with `b in {0,5}` (the held-out protocol) are EXCLUDED from the fitting subset.
  * No example duplication; selection uses a fixed deterministic seed before any fitting and never
    depends on reader performance.
  * If a seed cannot supply 256 eligible, unique examples with the requested per-class coverage,
    the collector raises an explicit validation error (it never silently shrinks the subset).

Outputs
-------
    experiments/experiment_y_fitting_control.npz        sub256 + W's VAL + E/F TEST (verbatim)
    experiments/experiment_y_fitting_control_meta.json  identity hashes, counts, provenance
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_y.py \
        --w-npz experiments/experiment_w_data_scale.npz \
        --output experiments/experiment_y_fitting_control.npz
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

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

SUBSET_SIZE = 256
NUMS_CLASS = 10
# Fixed deterministic subset-selection seed (declared before any fitting; no reader dependence).
Y_SUBSET_SEED = 20261010


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def subset_class_targets(size: int = SUBSET_SIZE, nclass: int = NUMS_CLASS) -> List[int]:
    """Even per-class target counts summing to `size` (base + one to the first `size % nclass`)."""
    base, rem = divmod(size, nclass)
    return [base + (1 if c < rem else 0) for c in range(nclass)]


def select_subset_rows(y_true: np.ndarray, operand_a: np.ndarray, operand_b: np.ndarray, seed: int,
                       size: int = SUBSET_SIZE, subset_seed: int = Y_SUBSET_SEED,
                       held_out_pairs=HELD_OUT_PAIRS) -> Tuple[np.ndarray, List[int]]:
    """Deterministic, fixed, class-balanced subset of pool row indices (into the passed arrays).

    Excludes held-out operand pairs `(a,b)` with `b in {0,5}`; never duplicates a row; returns the
    chosen rows sorted ascending and the actual per-class counts. Raises on insufficient coverage.
    """
    eligible = np.array([(int(a), int(b)) not in held_out_pairs
                         for a, b in zip(operand_a, operand_b)])
    idx = np.nonzero(eligible)[0]
    targets = subset_class_targets(size)
    rng = np.random.default_rng(subset_seed + seed)
    chosen: List[int] = []
    for c in range(len(targets)):
        cand = idx[y_true[idx] == c]
        if len(cand) < targets[c]:
            raise ValueError(
                f"seed {seed}: class {c} has only {len(cand)} eligible unique examples, "
                f"need {targets[c]} for a {size}-row balanced subset; refusing to shrink the subset")
        perm = rng.permutation(len(cand))
        chosen.extend(int(cand[j]) for j in perm[:targets[c]])
    chosen_arr = np.array(sorted(chosen), np.int64)
    if chosen_arr.shape[0] != size:
        raise ValueError(f"seed {seed}: selected {chosen_arr.shape[0]} rows, expected {size}")
    counts = [int((y_true[chosen_arr] == c).sum()) for c in range(len(targets))]
    if sorted(counts) != sorted(targets):
        raise ValueError(f"seed {seed}: per-class counts {counts} != targets {targets}")
    return chosen_arr, counts


def _seq_hashes(arr: np.ndarray) -> set:
    return {hash_sequence(arr[i].reshape(T_MAX, OBS_DIM)) for i in range(arr.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment Y fitting-control collector (measurement only)")
    ap.add_argument("--w-npz", default="experiments/experiment_w_data_scale.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_y_fitting_control.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_y_fitting_control_meta.json")
    ap.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    ap.add_argument("--subset-seed", type=int, default=Y_SUBSET_SEED)
    args = ap.parse_args()
    t0 = time.perf_counter()

    if not os.path.exists(args.w_npz):
        raise SystemExit(f"W artifact not found: {args.w_npz}. Run run_experiment_w.py first.")
    wz = np.load(args.w_npz, allow_pickle=False)
    w_meta = json.loads(str(wz["meta_json"][0]))
    ladder = list(w_meta["train_sizes"])
    top = int(ladder[-1])
    if set(w_meta["n"]) < {f"pool{top}", "val"}:
        raise SystemExit(f"W artifact missing pool{top}/val in meta")

    meta: Dict[str, Any] = {
        "factor": "fitting_subset",
        "subset_size": int(args.subset_size),
        "subset_seed": int(args.subset_seed),
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
        "subset_design": (f"deterministic fixed class-balanced small fitting subset of "
                          f"{args.subset_size} UNIQUE ordinary-ADD transitions per seed, stratified "
                          f"by y_true over its {NUMS_CLASS} classes as evenly as possible "
                          f"({subset_class_targets(args.subset_size)}), drawn from W's top-scale "
                          f"pool{top}; held-out operand pairs (b in {{0,5}}) excluded; no "
                          "duplication; selection seed fixed before any fitting"),
        "val": "W's canonical ordinary VAL (cap 800/seed); identical to W, byte-verified",
        "test": "W's untouched canonical strict TEST (all ADD); identical to W, byte-verified",
        "models": {},
        "identity": {},
        "per_seed": {},
        "disjointness": {},
    }

    # ---- build the fixed subset per seed + copy W's fixed VAL/TEST verbatim ----
    seeds = sorted(int(s) for s in w_meta["disjointness"].keys())
    sub_acc: Dict[str, List[np.ndarray]] = {c: [] for c in POOL_COLS + ["pool_row"]}
    val_payload: Dict[str, np.ndarray] = {}
    test_payload: Dict[str, np.ndarray] = {}
    for s in seeds:
        rows = np.nonzero(wz[f"pool{top}__seed"] == s)[0]
        yt = wz[f"pool{top}__y_true"][rows]
        oa = wz[f"pool{top}__operand_a"][rows]
        ob = wz[f"pool{top}__operand_b"][rows]
        chosen, counts = select_subset_rows(yt, oa, ob, s, size=args.subset_size,
                                            subset_seed=args.subset_seed)
        pool_rows = rows[chosen]
        for c in POOL_COLS:
            sub_acc[c].append(wz[f"pool{top}__{c}"][pool_rows])
        sub_acc["pool_row"].append(pool_rows)
        seq_c = wz[f"pool{top}__seq"][pool_rows]
        t_c = wz[f"pool{top}__t"][pool_rows]
        provenance = sorted((int(a), int(b)) for a, b in zip(seq_c, t_c))
        digest = hashlib.sha256(json.dumps(provenance).encode()).hexdigest()
        n_held = sum(1 for a, b in zip(oa, ob) if (int(a), int(b)) in HELD_OUT_PAIRS)
        meta["per_seed"][str(s)] = {
            "n_pool": int(len(rows)),
            "n_eligible": int(len(rows) - n_held),
            "n_heldout_pair_rows_pool": int(n_held),
            "n_selected": int(len(chosen)),
            "class_counts_target": subset_class_targets(args.subset_size),
            "class_counts_actual": counts,
            "selection_seed": int(args.subset_seed),
            "n_unique_seq_t": int(len(set(provenance))),
            "n_unique_sequences": int(len(set(int(x) for x in seq_c))),
            "n_duplicate_rows": int(len(provenance) - len(set(provenance))),
            "provenance_sha256": digest,
        }

    payload: Dict[str, np.ndarray] = {}
    for c in POOL_COLS + ["pool_row"]:
        payload[f"sub{args.subset_size}__{c}"] = np.concatenate(sub_acc[c], 0)
    for c in POOL_COLS:
        payload[f"val__{c}"] = wz[f"val__{c}"]
    for mk in MODEL_KEYS:
        for c in TEST_COLS:
            payload[f"{mk}__test__{c}"] = wz[f"{mk}__test__{c}"]

    # ---- byte identity vs W for the copied arrays ----
    for c in POOL_COLS:
        assert np.array_equal(payload[f"val__{c}"], wz[f"val__{c}"]), f"val copy mismatch {c}"
    for mk in MODEL_KEYS:
        for c in TEST_COLS:
            assert np.array_equal(payload[f"{mk}__test__{c}"], wz[f"{mk}__test__{c}"]), \
                f"test copy mismatch {mk} {c}"
    for k, v in payload.items():
        meta["identity"][k] = {"sha256": sha256_array(v), "shape": list(v.shape),
                               "dtype": str(v.dtype)}

    # ---- counts / disjointness ----
    meta["n"] = {"subset": int(payload[f"sub{args.subset_size}__y_true"].shape[0]),
                 "subset_per_seed": int(args.subset_size),
                 "val": int(payload["val__y_true"].shape[0])}
    for mk in MODEL_KEYS:
        meta["n"][f"{mk}__test"] = int(payload[f"{mk}__test__y_true"].shape[0])
    sub_h = _seq_hashes(payload[f"sub{args.subset_size}__seq_obs"])
    val_h = _seq_hashes(payload["val__seq_obs"])
    meta["disjointness"] = {"subset_vs_val": len(sub_h & val_h)}
    for mk in MODEL_KEYS:
        test_h = _seq_hashes(payload[f"{mk}__test__seq_obs"])
        meta["disjointness"][f"subset_vs_{mk}_test"] = len(sub_h & test_h)
        meta["disjointness"][f"val_vs_{mk}_test"] = len(val_h & test_h)
    meta["n_subset_unique_sequences"] = int(len(sub_h))

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
    print(f"Built subset{args.subset_size} + copied W VAL/TEST from {args.w_npz}")
    print(f"subset={meta['n']['subset']} val={meta['n']['val']} "
          f"test(E)={meta['n']['E_candidate__test']} test(F)={meta['n']['F_combined__test']}")
    for s in seeds:
        r = meta["per_seed"][str(s)]
        print(f"  seed {s}: selected={r['n_selected']} class_counts={r['class_counts_actual']} "
              f"unique_seq={r['n_unique_sequences']} dup={r['n_duplicate_rows']}")
    print(f"disjointness={meta['disjointness']}")
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
