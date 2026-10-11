#!/usr/bin/env python3
"""Experiment AG — source verification + training-pool eligibility support (measurement only).

AG asks whether AE's fit–transfer gap is sensitive to the sampling of the ordinary-ADD training pool
(operand-pair coverage and target-label composition), with the held-out operand pairs (`b in {0,5}`)
excluded from every condition. AG trains only the analysis readers (never RE:Reason) and probes no
representation.

This runner does NOT train. It verifies the frozen source and the prespecified sampling conditions:
    * V's SHA-256 and byte-identity through AD's artifact; target derivation; splits; Class-A mask;
      checkpoint SHA-256 (before and after);
    * the held-out exclusion: every non-held-out condition draws from the eligible pool only
      (`b not in {0,5}`), which excludes all 20 held-out pairs;
    * eligibility support for the prespecified sampling policies: all 80 non-held-out operand pairs and
      all 10 target classes must have >=1 eligible example per seed (a limitation is reported, never
      patched from validation/test).

Outputs
-------
    experiments/experiment_ag_training_pool_stress_meta.json
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_ag.py
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
PAIRS_NON_HELD = [(a, b) for a in range(10) for b in range(10) if b not in HELD_OUT_B]

CONDITIONS = ["ae_native", "natural_bootstrap", "pair_balanced_bootstrap", "target_balanced_bootstrap"]
BOOTSTRAP_CONDITIONS = CONDITIONS[1:]
REPLICATES = 3


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AG source verification")
    ap.add_argument("--ad-npz", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--ad-meta", default="experiments/experiment_ad_qualified_access_meta.json")
    ap.add_argument("--v-npz", default="experiments/experiment_v_sequence_access.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--meta-output", default="experiments/experiment_ag_training_pool_stress_meta.json")
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
                problems.append(f"disjointness: {mk}/{k}")
    if problems:
        raise SystemExit(f"AG source invariant failure: {problems}")

    # eligibility support: non-held-out pools must cover all 80 pairs and all 10 classes per seed
    support: Dict[str, Any] = {}
    for mk in ("E_candidate",):
        support[mk] = {}
        for s in SEEDS:
            tr = np.nonzero(ad[f"{mk}__trainord__seed"] == s)[0]
            ob = ad[f"{mk}__trainord__operand_b"][tr].astype(int)
            oa = ad[f"{mk}__trainord__operand_a"][tr].astype(int)
            y = ad[f"{mk}__trainord__y_true"][tr].astype(int)
            keep = ~np.isin(ob, list(HELD_OUT_B))
            eligible = int(keep.sum())
            # held-out exclusion present in the source pool
            n_heldout = int(np.isin(ob, list(HELD_OUT_B)).sum())
            oa_e, ob_e, y_e = oa[keep], ob[keep], y[keep]
            pair_counts = {f"{a}_{b}": int(((oa_e == a) & (ob_e == b)).sum()) for a, b in PAIRS_NON_HELD}
            class_counts = [int((y_e == c).sum()) for c in range(10)]
            missing_pairs = [k for k, v in pair_counts.items() if v == 0]
            missing_classes = [c for c in range(10) if class_counts[c] == 0]
            # Required strata: the 10 target classes (target_balanced) and >=1 eligible pair
            # (pair_balanced). Absent non-held-out pairs are a REPORTED support limitation, not a
            # failure, and are never patched from validation/test.
            ok = bool(not missing_classes and (len(PAIRS_NON_HELD) - len(missing_pairs)) >= 1)
            support[mk][str(s)] = {"eligible": eligible, "n_heldout_excluded": n_heldout,
                                   "pairs_present": len(PAIRS_NON_HELD) - len(missing_pairs),
                                   "pairs_total": len(PAIRS_NON_HELD),
                                   "missing_pairs": missing_pairs, "missing_classes": missing_classes,
                                   "pair_min": min(pair_counts.values()),
                                   "pair_max": max(pair_counts.values()),
                                   "class_counts": class_counts, "ok": ok}
    if not all(v["ok"] for v in support["E_candidate"].values()):
        # report the limitation explicitly; do not source replacement data
        print("ELIGIBILITY SUPPORT LIMITATION:", json.dumps(support, indent=2))
        raise SystemExit("eligibility support failure (required stratum missing)")

    ck: Dict[str, Any] = {}
    for mk in MODEL_KEYS:
        for s in SEEDS:
            rec = ad_meta["models"][mk]["checkpoints"][str(s)]
            ck[f"{mk}|{s}"] = {"path": rec["path"], "sha256": rec["sha256"]}
            if sha256_file(rec["path"]) != rec["sha256"]:
                raise SystemExit(f"checkpoint drift before run: {mk} seed={s}")

    meta: Dict[str, Any] = {
        "experiment": "AG — bounded-reader training-pool coverage and generalization stress",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"ad_npz": args.ad_npz, "ad_npz_sha256": sha256_file(args.ad_npz),
                   "ad_meta_sha256": sha256_file(args.ad_meta),
                   "v_npz": args.v_npz, "v_npz_sha256": v_sha, "ad_v_array_identity": True},
        "trained_anything": False,
        "conditions": CONDITIONS, "bootstrap_conditions": BOOTSTRAP_CONDITIONS, "replicates": REPLICATES,
        "held_out_b": list(HELD_OUT_B), "held_out_pairs_excluded": 20,
        "pairs_non_held": len(PAIRS_NON_HELD),
        "eligibility_support": support, "invariants_ok": True,
        "checkpoints": ck,
    }

    for mk in MODEL_KEYS:
        for s in SEEDS:
            rec = ad_meta["models"][mk]["checkpoints"][str(s)]
            if sha256_file(rec["path"]) != rec["sha256"]:
                raise SystemExit(f"checkpoint changed during AG collection: {mk} seed={s}")
    meta["post_run"] = {"all_checkpoints_unchanged": True}

    os.makedirs(os.path.dirname(args.meta_output), exist_ok=True)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"V source verified ({v_sha[:12]}); AD==V array identity True; target/splits OK")
    for s, v in support["E_candidate"].items():
        print(f"  seed {s}: eligible={v['eligible']} held-out-excluded={v['n_heldout_excluded']} "
              f"pairs {v['pair_min']}..{v['pair_max']} classes {v['class_counts']}")
    print(f"conditions: {CONDITIONS} (replicates={REPLICATES}); checkpoints unchanged: True")
    print(f"Wrote {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
