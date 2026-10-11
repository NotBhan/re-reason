#!/usr/bin/env python3
"""Experiment AE — source inspection + mandatory verification (measurement only).

AE characterises the gap between **training fit** and **transfer** for AD's AdamW-qualified analysis
readers under V's original data protocol. It does NOT test RE:Reason's frozen representation.

This runner does NOT train. It inspects AD's retained artifacts, determines whether the selected
reader states and required evaluation data can be recovered, and verifies the source before the AE
analysis reproduces the AdamW fits.

Finding recorded by this collector (see `experiments/experiment_ae_fit_transfer_meta.json`):
    * AD retained, per (seed, reader, history, variant, condition):
        - the best-on-VAL **scalar accuracy metrics** per population (Class-A, matched, correct, all-ADD,
          seen/held-out) and the train/val trajectories;
        - but **no state dictionaries / weights** (`has_state_dict=False`).
    * Therefore AE **cannot** reuse selected reader states. It must **deterministically reproduce the
      exact AdamW fits** using AD's trainer (`ab.train_ab`), V's source arrays, V's deterministic
      reader-seed convention and the AB-qualified `adamw` condition (lr 3e-3, wd 1e-2, 200 full-batch
      updates, best-on-VAL selection). AE adds no optimizer condition, no initialization change and no
      tuning, and reproduces ONLY the `adamw` condition.

Verified before (and re-verified after) the analysis:
    * AD artifact SHA-256 and its source identity (V artifact SHA-256 recorded in AD meta);
    * exact array identity of the AD artifact against V's source artifact;
    * `y_true = (operand_a + operand_b) mod 10` derivation; 10-class coverage; zero trainord/valord/test
      sequence overlap and no test leakage; V's causal-history verification present;
    * frozen E/F checkpoint SHA-256.

Outputs
-------
    experiments/experiment_ae_fit_transfer_meta.json    source identity, inspector findings, checkpoints
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_ae.py \
        --ad-npz experiments/experiment_ad_qualified_access.npz \
        --ad-raw experiments/experiment_ad_grid_raw.json
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

import analyze_experiment_u as au  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
SEEDS = [42, 43, 44, 45, 46]
CONDITIONS = ["adam_ref", "adamw"]
AE_CONDITION = "adamw"
SPLITS = ["trainord", "valord", "test"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_ad_raw(path: str) -> Dict[str, Any]:
    """Determine what AD retained — in particular whether selected reader weights are available."""
    with open(path) as f:
        parts = json.load(f)["parts"]
    fit_keys = sorted(parts[0]["fits"]["adamw"].keys())
    has_weights = any(("state" in k.lower() or "weight" in k.lower()) for k in fit_keys)
    return {"n_parts": len(parts), "fit_keys": fit_keys,
            "has_state_dict": bool(has_weights),
            "retained_metrics": ["best_val_acc", "best_val_update", "train_acc_final", "val_acc_final",
                                 "best_val.acc", "best_val.train_acc", "best_val.val_acc", "events"],
            "retained_seen_heldout": any("seen" in p["fits"]["adamw"].get("best_val", {})
                                         or "heldout" in p["fits"]["adamw"].get("best_val", {})
                                         for p in parts),
            "conditions": sorted({p["fits"][c]["cond"] for p in parts for c in CONDITIONS}),
            "readers": sorted({p["reader"] for p in parts}),
            "variants": sorted({p["variant"] for p in parts})}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AE source inspection + verification")
    ap.add_argument("--ad-npz", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--ad-meta", default="experiments/experiment_ad_qualified_access_meta.json")
    ap.add_argument("--ad-raw", default="experiments/experiment_ad_grid_raw.json")
    ap.add_argument("--v-npz", default="experiments/experiment_v_sequence_access.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--meta-output", default="experiments/experiment_ae_fit_transfer_meta.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    for p in (args.ad_npz, args.ad_meta, args.ad_raw, args.v_npz):
        if not os.path.exists(p):
            raise SystemExit(f"required artifact not found: {p}")

    ad_meta = json.load(open(args.ad_meta))
    ad_sha = sha256_file(args.ad_npz)
    v_sha = sha256_file(args.v_npz)

    # AD must have recorded V as its source; verify the recorded hash still matches the file on disk.
    recorded_v = ad_meta.get("source", {}).get("v_sha256")
    if recorded_v != v_sha:
        raise SystemExit(f"V source SHA mismatch: AD recorded {recorded_v}, disk {v_sha}")

    # exact array identity AD == V
    ad = np.load(args.ad_npz, allow_pickle=False)
    vv = np.load(args.v_npz, allow_pickle=False)
    mismatches = []
    for k in vv.files:
        if k == "meta_json":
            continue
        if k not in ad.files or not np.array_equal(ad[k], vv[k]):
            mismatches.append(k)
    if mismatches:
        raise SystemExit(f"AD/V array identity failure: {mismatches}")

    # invariants recorded by AD
    problems: List[str] = []
    for k, rec in ad_meta.get("target_derivation", {}).items():
        if not rec.get("exact", False):
            problems.append(f"target derivation: {k}")
    for ds in SPLITS:
        for mk in MODEL_KEYS:
            cc = ad_meta.get("class_counts", {}).get(ds, {}).get(mk)
            if cc is None or len(cc) != 10 or any(int(c) <= 0 for c in cc):
                problems.append(f"class coverage: {mk}/{ds}={cc}")
    for mk in MODEL_KEYS:
        for k, val in ad_meta.get("disjointness", {}).get(mk, {}).items():
            if k.startswith(("trainord_vs", "valord_vs")) and int(val) != 0:
                problems.append(f"disjointness: {mk}/{k}={val}")
    if not ad_meta.get("history_causality"):
        problems.append("history causality missing")
    if problems:
        raise SystemExit(f"AD invariant failure: {problems}")

    # frozen checkpoint hashes (before; re-verified after)
    ck_before: Dict[str, Any] = {}
    for mk in MODEL_KEYS:
        for s in SEEDS:
            rec = ad_meta["models"][mk]["checkpoints"][str(s)]
            ck_before[f"{mk}|{s}"] = sha256_file(rec["path"])
            if ck_before[f"{mk}|{s}"] != rec["sha256"]:
                raise SystemExit(f"checkpoint drift: {mk} seed={s}")

    inspection = inspect_ad_raw(args.ad_raw)
    if inspection["has_state_dict"]:
        raise SystemExit("AD retained state dictionaries; the AE reproduction plan expects them absent")

    meta: Dict[str, Any] = {
        "experiment": "AE — baseline-aware fit-transfer diagnostic (AD adamw condition)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"ad_npz": args.ad_npz, "ad_npz_sha256": ad_sha,
                   "ad_meta": args.ad_meta, "ad_meta_sha256": sha256_file(args.ad_meta),
                   "ad_raw": args.ad_raw, "ad_raw_sha256": sha256_file(args.ad_raw),
                   "v_npz": args.v_npz, "v_npz_sha256": v_sha,
                   "v_sha256_recorded_in_ad_meta": recorded_v,
                   "ad_v_array_identity": True},
        "condition": AE_CONDITION,
        "reused_retained_states": False,
        "reproduction_plan": (
            "AD retained the best-on-VAL accuracy metrics and train/val trajectories but NO state "
            "dictionaries, so the selected reader states cannot be reused. AE deterministically "
            "reproduces the exact AdamW fits with AD's trainer (ab.train_ab), V's source arrays, V's "
            "deterministic reader-seed convention (av.reader_seed) and the AB-qualified adamw condition "
            "(lr 3e-3, wd 1e-2, 200 full-batch updates, best-on-VAL selection). No optimizer condition, "
            "initialization change, target, loss or tuning is added; only `adamw` is reproduced."),
        "inspection": inspection,
        "invariants_ok": True,
        "checkpoints": {mk: {s: ad_meta["models"][mk]["checkpoints"][str(s)] for s in map(str, SEEDS)}
                        for mk in MODEL_KEYS},
    }

    # re-verify checkpoints after inspection
    for mk in MODEL_KEYS:
        for s in SEEDS:
            rec = ad_meta["models"][mk]["checkpoints"][str(s)]
            if sha256_file(rec["path"]) != rec["sha256"]:
                raise SystemExit(f"checkpoint changed during AE collection: {mk} seed={s}")
            rec["unchanged_after_run"] = True
    meta["post_run"] = {"all_checkpoints_unchanged": True}

    os.makedirs(os.path.dirname(args.meta_output), exist_ok=True)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"AD source verified: {args.ad_npz}")
    print(f"  V source SHA matches AD meta: {recorded_v == v_sha}")
    print(f"  AD == V array identity: {not mismatches}")
    print(f"  AD retained state_dict: {inspection['has_state_dict']} -> AE reproduces adamw fits")
    print(f"  conditions retained in AD raw: {inspection['conditions']}")
    print(f"  checkpoints verified unchanged: True")
    print(f"Wrote {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
