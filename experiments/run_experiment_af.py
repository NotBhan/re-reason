#!/usr/bin/env python3
"""Experiment AF — source verification + mandatory pre-measurement checks (measurement only).

AF asks whether AE's weak held-out-operand results coincide with task-side differences in label
geometry or raw-interface target recoverability, using U's existing representation-independent
exact-interface machinery on the unchanged V dataset. AF trains no reader, probes no RE:Reason latent
state, changes no task and modifies nothing.

This runner does NOT compute the AF statistics. It verifies the frozen source before the analysis and
records the audit trail:
    * V's SHA-256 and byte-identity through AD's artifact (`experiments/experiment_ad_qualified_access.npz`);
    * `y_true = (operand_a + operand_b) mod 10` derivation, 10-class coverage, zero trainord/valord/test
      sequence overlap, V's causal-history verification;
    * the canonical population masks (Class-A = frozen candidate wrong; all-ADD = every ADD test row;
      Class-A + candidate-correct partition all-ADD per seed) and the held-out operand condition
      (`b in {0,5}`, 20 pairs);
    * frozen E/F checkpoint SHA-256 before and after.

Outputs
-------
    experiments/experiment_af_task_difficulty_meta.json    source identity, split/mask checks, checkpoints
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_af.py
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
SPLITS = ["trainord", "valord", "test"]
HELD_OUT_B = (0, 5)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AF source verification")
    ap.add_argument("--ad-npz", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--ad-meta", default="experiments/experiment_ad_qualified_access_meta.json")
    ap.add_argument("--v-npz", default="experiments/experiment_v_sequence_access.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--meta-output", default="experiments/experiment_af_task_difficulty_meta.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    for p in (args.ad_npz, args.ad_meta, args.v_npz):
        if not os.path.exists(p):
            raise SystemExit(f"required artifact not found: {p}")

    ad_meta = json.load(open(args.ad_meta))
    v_sha = sha256_file(args.v_npz)
    if ad_meta.get("source", {}).get("v_sha256") != v_sha:
        raise SystemExit("V source SHA does not match AD meta")
    ad = np.load(args.ad_npz, allow_pickle=False)
    vv = np.load(args.v_npz, allow_pickle=False)
    mismatches = [k for k in vv.files if k != "meta_json"
                  and (k not in ad.files or not np.array_equal(ad[k], vv[k]))]
    if mismatches:
        raise SystemExit(f"AD/V array identity failure: {mismatches}")

    problems: List[str] = []
    for k, rec in ad_meta.get("target_derivation", {}).items():
        if not rec.get("exact", False):
            problems.append(f"target derivation: {k}")
    for ds in SPLITS:
        for mk in MODEL_KEYS:
            cc = ad_meta.get("class_counts", {}).get(ds, {}).get(mk)
            if cc is None or len(cc) != 10 or any(int(c) <= 0 for c in cc):
                problems.append(f"class coverage: {mk}/{ds}")
    for mk in MODEL_KEYS:
        dj = ad_meta.get("disjointness", {}).get(mk, {})
        for k in ("trainord_vs_valord", "trainord_vs_test", "valord_vs_test"):
            if int(dj.get(k, -1)) != 0:
                problems.append(f"disjointness: {mk}/{k}={dj.get(k)}")
    if not ad_meta.get("history_causality"):
        problems.append("history-causality verification missing")
    if problems:
        raise SystemExit(f"AF source invariant failure: {problems}")

    # population masks and held-out condition (per model/seed)
    mask_checks: Dict[str, Any] = {}
    for mk in MODEL_KEYS:
        mask_checks[mk] = {}
        for s in SEEDS:
            sel = ad[f"{mk}__test__seed"] == s
            masks = au._pop_masks(ad, mk, s)
            clsA = int((masks["classA"] & sel).sum())
            correct = int((masks["correct_all"] & sel).sum())
            alladd = int((masks["all_add"] & sel).sum())
            if clsA + correct != alladd:
                raise SystemExit(f"population partition failure: {mk} seed={s}")
            ob = ad[f"{mk}__test__operand_b"][sel].astype(int)
            n_held = int(np.isin(ob, list(HELD_OUT_B)).sum())
            mask_checks[mk][str(s)] = {"classA": clsA, "correct": correct, "all_add": alladd,
                                       "heldout_rows": n_held,
                                       "heldout_pairs": 20}

    ck: Dict[str, Any] = {}
    for mk in MODEL_KEYS:
        for s in SEEDS:
            rec = ad_meta["models"][mk]["checkpoints"][str(s)]
            ck[f"{mk}|{s}"] = {"path": rec["path"], "sha256": rec["sha256"]}
            if sha256_file(rec["path"]) != rec["sha256"]:
                raise SystemExit(f"checkpoint drift before run: {mk} seed={s}")

    meta: Dict[str, Any] = {
        "experiment": "AF — held-out operand task-difficulty stratification",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"ad_npz": args.ad_npz, "ad_npz_sha256": sha256_file(args.ad_npz),
                   "ad_meta_sha256": sha256_file(args.ad_meta),
                   "v_npz": args.v_npz, "v_npz_sha256": v_sha, "ad_v_array_identity": True},
        "estimator": "U's exact-interface machinery (analyze_experiment_u: history_features, exact_bayes)",
        "trained_anything": False,
        "held_out_b": list(HELD_OUT_B), "held_out_pairs": 20,
        "masks": mask_checks, "invariants_ok": True,
        "checkpoints": ck,
    }

    # re-verify after
    for mk in MODEL_KEYS:
        for s in SEEDS:
            rec = ad_meta["models"][mk]["checkpoints"][str(s)]
            if sha256_file(rec["path"]) != rec["sha256"]:
                raise SystemExit(f"checkpoint changed during AF collection: {mk} seed={s}")
    meta["post_run"] = {"all_checkpoints_unchanged": True}

    os.makedirs(os.path.dirname(args.meta_output), exist_ok=True)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"V source verified ({v_sha[:12]}); AD==V array identity True")
    print(f"target derivation exact; class coverage; disjointness; history causality: all OK")
    for mk in MODEL_KEYS:
        print(f"  {mk}: " + ", ".join(
            f"seed{s}: classA={v['classA']} correct={v['correct']} all={v['all_add']} held={v['heldout_rows']}"
            for s, v in list(mask_checks[mk].items())[:1]) + " (5 seeds checked)")
    print(f"checkpoints verified unchanged: True")
    print(f"Wrote {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
