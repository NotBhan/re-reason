#!/usr/bin/env python3
"""Experiment Z — TCN Instrument Qualification with a Directly Observable Target (collector).

Y (Y-MIX) found that, on a fixed, class-balanced 256-example subset of the ADD target `y_true`, the
**GRU fits but does not transfer** while the **TCN does not fit even the small subset** (training
accuracy flat at chance). Z qualifies the TCN on a **directly observable** control:

    Can the TCN fit the four-way `op_dest` target — the destination register index, which is
    literally the argmax of the current 21-dim observation's `reg_a` block `x_t[3:7]` — using the
    SAME input data, subset, readers, optimizer protocol and history conditions as Y?

`op_dest` is directly recoverable from the current observation x_t (verified 100% in this collector),
so it is a pure direct-observation control: if the TCN cannot fit it, the obstacle is the TCN
training/optimization instrument; if it can, the TCN's `y_true` non-fitting is target-specific under
these conditions (which does NOT qualify the TCN for ADD-target representation claims).

Z is measurement-only and reuses Y's exact fixed subset and provenance. Nothing is regenerated,
resampled, rebalanced or enlarged; no RE:Reason component is modified; no Phase 2. The readers are
separate analysis-only models. `op_dest` is used as a LABEL only (never as an input feature); the
reader input remains the raw 21-dim observation + validity bit.

Outputs
-------
    experiments/experiment_z_op_dest.npz        Y's fixed sub256 + VAL + E/F TEST (verbatim) + op_dest
    experiments/experiment_z_op_dest_meta.json  identity hashes, op_dest class counts, verification
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_z.py \
        --y-npz experiments/experiment_y_fitting_control.npz \
        --output experiments/experiment_z_op_dest.npz
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

import run_experiment_u as ru  # noqa: E402
import run_experiment_y as ry  # noqa: E402  (subset key naming / POOL_COLS / TEST_COLS)
from re_reason.benchmarks.dataset_split import hash_sequence  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
SUBSET_PREFIX = "sub256"
N_OP_DEST = 4
T_MAX = ru.T_MAX
OBS_DIM = ru.OBS_DIM


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _seq_hashes(arr: np.ndarray) -> set:
    return {hash_sequence(arr[i].reshape(T_MAX, OBS_DIM)) for i in range(arr.shape[0])}


def op_dest_counts(dest: np.ndarray) -> List[int]:
    return np.bincount(dest.astype(np.int64), minlength=N_OP_DEST).tolist()


def assert_all_classes(counts: List[int], where: str) -> None:
    if any(int(c) <= 0 for c in counts):
        raise ValueError(f"op_dest class coverage failure in {where}: counts={counts}; "
                         "refusing to proceed (a required class is absent)")


def verify_direct_recoverability(seq_obs: np.ndarray, t: np.ndarray, dest: np.ndarray,
                                 where: str) -> Dict[str, Any]:
    """`op_dest` must be exactly the argmax of the current observation's reg_a block x_t[3:7]."""
    N = seq_obs.shape[0]
    full = seq_obs.reshape(N, T_MAX, OBS_DIM)
    ok = 0
    for i in range(N):
        x = full[i, int(t[i])]
        ok += int(int(np.argmax(x[3:7])) == int(dest[i]))
    if ok != N:
        raise ValueError(f"op_dest is not directly recoverable from the current observation in "
                         f"{where}: {ok}/{N}")
    return {"n": int(N), "recoverable": int(ok), "rate": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment Z op_dest instrument-qualification collector")
    ap.add_argument("--y-npz", default="experiments/experiment_y_fitting_control.npz")
    ap.add_argument("--w-npz", default="experiments/experiment_w_data_scale.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_z_op_dest.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_z_op_dest_meta.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    if not os.path.exists(args.y_npz):
        raise SystemExit(f"Y artifact not found: {args.y_npz}. Run run_experiment_y.py first.")
    yz = np.load(args.y_npz, allow_pickle=False)
    y_meta = json.loads(str(yz["meta_json"][0]))
    top = int(y_meta["source"]["top_size"])

    # Y must itself be pinned to a verified W source.
    w_sha = sha256_file(args.w_npz)
    if w_sha != y_meta["source"]["sha256"]:
        raise SystemExit(f"W source SHA mismatch: Y recorded {y_meta['source']['sha256']} but "
                         f"{args.w_npz} is {w_sha}")

    meta: Dict[str, Any] = {
        "experiment": "Z — TCN instrument qualification on a directly observable target (op_dest)",
        "target": "op_dest",
        "n_classes": N_OP_DEST,
        "subset_size": int(y_meta["subset_size"]),
        "subset_seed": int(y_meta["subset_seed"]),
        "readers": ["GRU", "TCN"],
        "histories": ["L4", "Lfull"],
        "source": {
            "y_npz": args.y_npz,
            "y_sha256": sha256_file(args.y_npz),
            "y_subset_seed": y_meta["subset_seed"],
            "w_npz": args.w_npz,
            "w_sha256": w_sha,
            "train_sizes": y_meta["source"]["train_sizes"],
            "top_size": top,
        },
        "obs_schema": y_meta.get("obs_schema"),
        "subset_design": ("reuses Experiment Y's EXACT fixed class-balanced subset "
                          f"({y_meta['subset_size']} unique ordinary-ADD transitions per seed, "
                          "stratified by y_true; NOT regenerated, resampled, rebalanced or enlarged)"),
        "op_dest_schema": ("destination register index = argmax(x_t[3:7]) of the current 21-dim "
                           "observation; directly observable at the current timestep; used as a LABEL "
                           "only, never as an input feature"),
        "models": {},
        "identity": {},
        "class_counts": {},
        "recoverability": {},
        "disjointness": {},
    }

    seeds = sorted(int(s) for s in y_meta["per_seed"].keys())

    # ---- copy Y's fixed subset + VAL + TEST verbatim (byte identity) ----
    payload: Dict[str, np.ndarray] = {}
    sub_cols = ry.POOL_COLS + ["pool_row"]
    for c in sub_cols:
        payload[f"{SUBSET_PREFIX}__{c}"] = yz[f"{SUBSET_PREFIX}__{c}"]
    for c in ry.POOL_COLS:
        payload[f"val__{c}"] = yz[f"val__{c}"]
    for mk in MODEL_KEYS:
        for c in ry.TEST_COLS:
            payload[f"{mk}__test__{c}"] = yz[f"{mk}__test__{c}"]

    for c in sub_cols:
        assert np.array_equal(payload[f"{SUBSET_PREFIX}__{c}"], yz[f"{SUBSET_PREFIX}__{c}"]), c
    for c in ry.POOL_COLS:
        assert np.array_equal(payload[f"val__{c}"], yz[f"val__{c}"]), c
    for mk in MODEL_KEYS:
        for c in ry.TEST_COLS:
            assert np.array_equal(payload[f"{mk}__test__{c}"], yz[f"{mk}__test__{c}"]), (mk, c)
    for k, v in payload.items():
        meta["identity"][k] = {"sha256": sha256_array(v), "shape": list(v.shape), "dtype": str(v.dtype)}

    # ---- subset provenance must match Y's recorded digest ----
    prov_mismatches = []
    for s in seeds:
        rows = np.nonzero(payload[f"{SUBSET_PREFIX}__seed"] == s)[0]
        prov = sorted((int(a), int(b)) for a, b in
                      zip(payload[f"{SUBSET_PREFIX}__seq"][rows], payload[f"{SUBSET_PREFIX}__t"][rows]))
        digest = hashlib.sha256(json.dumps(prov).encode()).hexdigest()
        if digest != y_meta["per_seed"][str(s)]["provenance_sha256"]:
            prov_mismatches.append(str(s))
    if prov_mismatches:
        raise SystemExit(f"subset provenance mismatch vs Y for seeds {prov_mismatches}")

    # ---- op_dest class counts + coverage (train per seed, val per seed, test) ----
    meta["class_counts"] = {"train": {}, "val": {}, "test": {}}
    pooled_train = np.zeros(N_OP_DEST, int)
    for s in seeds:
        rows = np.nonzero(payload[f"{SUBSET_PREFIX}__seed"] == s)[0]
        counts = op_dest_counts(payload[f"{SUBSET_PREFIX}__op_dest"][rows])
        assert_all_classes(counts, f"train seed {s}")
        meta["class_counts"]["train"][str(s)] = counts
        pooled_train += np.array(counts)
    meta["class_counts"]["train"]["pooled"] = pooled_train.tolist()
    for s in seeds:
        rows = np.nonzero(payload["val__seed"] == s)[0]
        counts = op_dest_counts(payload["val__op_dest"][rows])
        assert_all_classes(counts, f"val seed {s}")
        meta["class_counts"]["val"][str(s)] = counts
    meta["class_counts"]["val"]["pooled"] = op_dest_counts(payload["val__op_dest"])
    for mk in MODEL_KEYS:
        counts = op_dest_counts(payload[f"{mk}__test__op_dest"])
        assert_all_classes(counts, f"{mk} test")
        meta["class_counts"]["test"][mk] = counts

    # ---- op_dest direct recoverability (current observation) ----
    meta["recoverability"]["train"] = verify_direct_recoverability(
        payload[f"{SUBSET_PREFIX}__seq_obs"], payload[f"{SUBSET_PREFIX}__t"],
        payload[f"{SUBSET_PREFIX}__op_dest"], "train subset")
    meta["recoverability"]["val"] = verify_direct_recoverability(
        payload["val__seq_obs"], payload["val__t"], payload["val__op_dest"], "val")
    for mk in MODEL_KEYS:
        meta["recoverability"][f"{mk}_test"] = verify_direct_recoverability(
            payload[f"{mk}__test__seq_obs"], payload[f"{mk}__test__t"],
            payload[f"{mk}__test__op_dest"], f"{mk} test")

    # ---- leakage / disjointness ----
    sub_h = _seq_hashes(payload[f"{SUBSET_PREFIX}__seq_obs"])
    val_h = _seq_hashes(payload["val__seq_obs"])
    meta["disjointness"] = {"subset_vs_val": len(sub_h & val_h)}
    for mk in MODEL_KEYS:
        test_h = _seq_hashes(payload[f"{mk}__test__seq_obs"])
        meta["disjointness"][f"subset_vs_{mk}_test"] = len(sub_h & test_h)
        meta["disjointness"][f"val_vs_{mk}_test"] = len(val_h & test_h)
    if any(v != 0 for v in meta["disjointness"].values()):
        raise SystemExit(f"split disjointness failure: {meta['disjointness']}")

    # ---- checkpoint hashes (frozen E/F; unchanged) ----
    for mk in MODEL_KEYS:
        meta["models"][mk] = {"checkpoints": {}}
        for s in seeds:
            p = os.path.join(args.checkpoints_dir, f"{mk}_seed{s}_3000steps.pt")
            meta["models"][mk]["checkpoints"][str(s)] = {"path": p, "sha256": sha256_file(p)}

    meta["n"] = {"subset": int(payload[f"{SUBSET_PREFIX}__y_true"].shape[0]),
                 "subset_per_seed": int(y_meta["n"]["subset_per_seed"]),
                 "val": int(payload["val__y_true"].shape[0])}
    for mk in MODEL_KEYS:
        meta["n"][f"{mk}__test"] = int(payload[f"{mk}__test__y_true"].shape[0])
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Copied Y's fixed subset{meta['n']['subset_per_seed']}/seed + VAL + E/F TEST from {args.y_npz}")
    print(f"op_dest train pooled counts={meta['class_counts']['train']['pooled']} "
          f"val pooled={meta['class_counts']['val']['pooled']}")
    for mk in MODEL_KEYS:
        print(f"  {mk} test op_dest counts={meta['class_counts']['test'][mk]}")
    print(f"op_dest recoverability: train={meta['recoverability']['train']['rate']} "
          f"val={meta['recoverability']['val']['rate']}")
    print(f"disjointness={meta['disjointness']}")
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
