#!/usr/bin/env python3
"""Experiment AH — source verification + pre-registered reader capacity ladder (measurement only).

AG showed that the training-pool construction changes the reader fit but not the held-out-operand
transfer; AF removed the task-side explanation. AH asks the reader-side question at V's scale:
holding V's arrays, held-out protocol (`b in {0,5}` excluded from every fit), optimizer recipe,
reader families, histories and seeds fixed, is the fit-transfer gap sensitive to reader **capacity**?
A pre-declared 5-level width ladder is fit at V's decisive histories (L4, Lfull); AH-A if held-out
transfer emerges at some width (capacity-sensitive), AH-B if the gap is capacity-insensitive.

This runner does NOT train. It verifies the frozen source and records the pre-registered ladder and
fit grid before any fit exists:
    * V's SHA-256 and byte-identity through AD's artifact
      (`experiments/experiment_ad_qualified_access.npz`);
    * the SHA-256 / shape / dtype of every AD source array against AD's recorded identity;
    * AD's invariants (target derivation, class coverage, split disjointness, history causality);
    * the capacity ladder with EXACT parameter counts from V's own formulas
      (`analyze_experiment_v.gru_param_count` / `tcn_param_count`), every level within 4.0x the
      baseline level's parameter count;
    * the native eligible training pool (`trainord` rows with `b not in {0,5}`) per seed;
    * frozen E/F checkpoint SHA-256 before and after.

The ladder is fixed in code before any fit is run: no width, history, target or seed is selected
after seeing a result (V's no-test-selection rule is retained).

Outputs
-------
    experiments/experiment_ah_capacity_ladder_meta.json
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_ah.py
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

import analyze_experiment_v as av  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
SEEDS: List[int] = [42, 43, 44, 45, 46]
SPLITS = ["trainord", "valord", "test"]
HELD_OUT_B: Tuple[int, ...] = (0, 5)
CHECKPOINT_TEMPLATE = "{model}_seed{seed}_3000steps.pt"

NCLASS: int = av.N_MOD                 # 10 — primary target (y_true)
MAX_NCLASS: int = av.MAX_NCLASS        # 19 — raw_sum; V shares one width across all targets
CAP_RATIO: float = 4.0

# Pre-registered capacity ladder: (level_key, level_name, gru_hidden, tcn_channels).
# 'baseline' reproduces V's deterministic width selection at the U MLP-128 budget.
LEVELS: List[Tuple[str, str, int, int]] = [
    ("small", "Small", 42, 26),
    ("medium_small", "Medium-small", 60, 37),
    ("baseline", "Baseline", 85, 52),
    ("medium_large", "Medium-large", 120, 74),
    ("large", "Large", 170, 104),
]

READERS: List[str] = ["GRU", "TCN"]
HISTORIES: List[str] = ["L4", "Lfull"]
GRID_DEFINITION = "2 readers x 5 widths x 2 histories (L4, Lfull) x 5 seeds = 100 fits"
PRE_REGISTERED_FITS = 100
ELIGIBLE_SIZE_RANGE: Tuple[int, int] = (741, 781)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def verify_protocol_identity() -> None:
    """AH holds V's reader families, decisive histories and independent seeds fixed."""
    if READERS != list(av.FAMILIES) or HISTORIES != list(av.DECISIVE) or SEEDS != list(av.SEEDS):
        raise SystemExit("AH protocol drift against V's pre-registered reader protocol")


def verify_array_hashes(ad: Any, ad_meta: Dict[str, Any]) -> int:
    """Recompute the SHA-256 / shape / dtype of every AD source array against AD's identity record."""
    ident = ad_meta.get("identity", {})
    keys = [k for k in ad.files if k != "meta_json"]
    for k in keys:
        rec = ident.get(k)
        if rec is None:
            raise SystemExit(f"array identity record missing: {k}")
        if (sha256_array(ad[k]) != rec.get("sha256") or list(ad[k].shape) != rec.get("shape")
                or str(ad[k].dtype) != rec.get("dtype")):
            raise SystemExit(f"array identity drift: {k}")
    return len(keys)


def verify_ad_invariants(ad_meta: Dict[str, Any]) -> None:
    """Re-assert the AD-recorded source invariants AH relies on (fail loudly on any violation)."""
    problems: List[str] = []
    for k, rec in ad_meta.get("target_derivation", {}).items():
        if not rec.get("exact", False):
            problems.append(f"target derivation: {k}")
    for ds in SPLITS:
        for mk in MODEL_KEYS:
            cc = ad_meta.get("class_counts", {}).get(ds, {}).get(mk)
            if cc is None or len(cc) != NCLASS or any(int(c) <= 0 for c in cc):
                problems.append(f"class coverage: {mk}/{ds}")
    for mk in MODEL_KEYS:
        dj = ad_meta.get("disjointness", {}).get(mk, {})
        for k in ("trainord_vs_valord", "trainord_vs_test", "valord_vs_test"):
            if int(dj.get(k, -1)) != 0:
                problems.append(f"disjointness: {mk}/{k}={dj.get(k)}")
    if not ad_meta.get("history_causality"):
        problems.append("history-causality verification missing")
    if problems:
        raise SystemExit(f"AH source invariant failure: {problems}")


def param_counts(gru_hidden: int, tcn_channels: int) -> Dict[str, int]:
    """Exact parameter counts for one ladder level using V's own formulas (indim=22, k=3, 4 layers).

    Primary counts are at the 10-class target (y_true); the 19-class (raw_sum) counts are recorded
    as well because V shares one width across every target.
    """
    return {
        "gru_params": av.gru_param_count(gru_hidden, indim=av.INPUT_DIM, nclass=NCLASS),
        "tcn_params": av.tcn_param_count(tcn_channels, indim=av.INPUT_DIM, nclass=NCLASS,
                                         kernel=av.KERNEL, nlayers=av.N_LAYERS),
        "gru_params_max_nclass": av.gru_param_count(gru_hidden, indim=av.INPUT_DIM,
                                                    nclass=MAX_NCLASS),
        "tcn_params_max_nclass": av.tcn_param_count(tcn_channels, indim=av.INPUT_DIM,
                                                    nclass=MAX_NCLASS, kernel=av.KERNEL,
                                                    nlayers=av.N_LAYERS),
    }


def verify_capacity_ladder() -> Dict[str, Any]:
    """Record the ladder's exact parameter counts and ratios; enforce the 4.0x-baseline cap."""
    counts = {key: param_counts(gh, tc) for key, _, gh, tc in LEVELS}
    base = counts["baseline"]
    levels: List[Dict[str, Any]] = []
    for key, name, gh, tc in LEVELS:
        p = counts[key]
        rec: Dict[str, Any] = {"level_key": key, "level_name": name,
                               "gru_hidden": gh, "tcn_channels": tc, **p}
        rec["gru_ratio_vs_baseline"] = p["gru_params"] / base["gru_params"]
        rec["tcn_ratio_vs_baseline"] = p["tcn_params"] / base["tcn_params"]
        rec["gru_ratio_vs_baseline_max_nclass"] = (p["gru_params_max_nclass"]
                                                   / base["gru_params_max_nclass"])
        rec["tcn_ratio_vs_baseline_max_nclass"] = (p["tcn_params_max_nclass"]
                                                   / base["tcn_params_max_nclass"])
        rec["within_cap"] = max(rec["gru_ratio_vs_baseline"],
                                rec["tcn_ratio_vs_baseline"]) <= CAP_RATIO
        rec["within_cap_max_nclass"] = max(rec["gru_ratio_vs_baseline_max_nclass"],
                                           rec["tcn_ratio_vs_baseline_max_nclass"]) <= CAP_RATIO
        if not (rec["within_cap"] and rec["within_cap_max_nclass"]):
            raise SystemExit(f"capacity cap violation (> {CAP_RATIO}x baseline): {key}")
        levels.append(rec)

    # a genuine capacity ladder: strictly increasing widths in both reader families
    for seq, label in (([r["gru_hidden"] for r in levels], "GRU hidden"),
                       ([r["tcn_channels"] for r in levels], "TCN channels")):
        if any(b <= a for a, b in zip(seq, seq[1:])):
            raise SystemExit(f"ladder not strictly increasing: {label}")

    # the baseline level must reproduce V's deterministic width selection (U MLP-128 budget)
    arch = av.architecture()
    baseline = next(r for r in levels if r["level_key"] == "baseline")
    if (baseline["gru_hidden"] != int(arch["GRU"]["hidden"])
            or baseline["tcn_channels"] != int(arch["TCN"]["channels"])
            or baseline["gru_params"] != int(arch["GRU"]["params_primary"])
            or baseline["tcn_params"] != int(arch["TCN"]["params_primary"])):
        raise SystemExit("baseline level does not reproduce V's deterministic architecture selection")

    return {
        "cap_ratio": CAP_RATIO,
        "n_levels": len(levels),
        "indim": av.INPUT_DIM, "kernel": av.KERNEL, "n_causal_layers": av.N_LAYERS,
        "nclass_primary": NCLASS, "max_nclass": MAX_NCLASS,
        "u_mlp128_budget": int(arch["budget_u_mlp128"]),
        "levels": levels,
        "all_levels_within_cap": True,
        "baseline_reproduces_v_architecture": True,
        "max_ratio_vs_baseline": max(max(r["gru_ratio_vs_baseline"], r["tcn_ratio_vs_baseline"])
                                     for r in levels),
    }


def _eligible_triples(ad: Any, mk: str, seed: int) -> np.ndarray:
    """Sorted (operand_a, operand_b, y_true) triples of the native eligible pool (b not in {0,5})."""
    tr = np.nonzero(ad[f"{mk}__trainord__seed"] == seed)[0]
    ob = ad[f"{mk}__trainord__operand_b"][tr].astype(np.int64)
    t = tr[~np.isin(ob, list(HELD_OUT_B))]
    rows = np.stack([ad[f"{mk}__trainord__operand_a"][t], ad[f"{mk}__trainord__operand_b"][t],
                     ad[f"{mk}__trainord__y_true"][t]], axis=1).astype(np.int64)
    return rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]


def verify_eligible_pools(ad: Any) -> Dict[str, Any]:
    """Native eligible training pool per seed and model: trainord rows with b not in {0,5}."""
    pools: Dict[str, Any] = {"definition": "trainord rows with operand_b not in {0, 5}",
                             "held_out_b": list(HELD_OUT_B), "held_out_pairs_excluded": 20,
                             "expected_size_range": list(ELIGIBLE_SIZE_RANGE), "models": {}}
    for mk in MODEL_KEYS:
        pools["models"][mk] = {}
        for s in SEEDS:
            tr = np.nonzero(ad[f"{mk}__trainord__seed"] == s)[0]
            ob = ad[f"{mk}__trainord__operand_b"][tr].astype(int)
            eligible = int((~np.isin(ob, list(HELD_OUT_B))).sum())
            rec = {"trainord_rows": int(len(tr)), "eligible": eligible,
                   "heldout_excluded": int(len(tr)) - eligible}
            if rec["heldout_excluded"] + rec["eligible"] != rec["trainord_rows"]:
                raise SystemExit(f"eligible/held-out partition failure: {mk} seed={s}")
            rec["in_range"] = bool(ELIGIBLE_SIZE_RANGE[0] <= eligible <= ELIGIBLE_SIZE_RANGE[1])
            if not rec["in_range"]:
                raise SystemExit(f"eligible pool size outside {ELIGIBLE_SIZE_RANGE}: "
                                 f"{mk} seed={s} n={eligible}")
            pools["models"][mk][str(s)] = rec

    # explicit consistency check: E and F draw the same eligible (a, b, y) rows per seed
    ident = {str(s): bool(np.array_equal(_eligible_triples(ad, "E_candidate", s),
                                         _eligible_triples(ad, "F_combined", s)))
             for s in SEEDS}
    if not all(ident.values()):
        raise SystemExit(f"E/F eligible-pool identity failure: {ident}")
    pools["e_f_eligible_pool_identical"] = ident
    return pools


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AH source verification + capacity ladder")
    ap.add_argument("--ad-npz", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--ad-meta", default="experiments/experiment_ad_qualified_access_meta.json")
    ap.add_argument("--v-npz", default="experiments/experiment_v_sequence_access.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--meta-output", default="experiments/experiment_ah_capacity_ladder_meta.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    for p in (args.ad_npz, args.ad_meta, args.v_npz):
        if not os.path.exists(p):
            raise SystemExit(f"required artifact not found: {p}")

    verify_protocol_identity()

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

    n_arrays = verify_array_hashes(ad, ad_meta)
    verify_ad_invariants(ad_meta)

    ladder = verify_capacity_ladder()
    pools = verify_eligible_pools(ad)

    if len(READERS) * len(LEVELS) * len(HISTORIES) * len(SEEDS) != PRE_REGISTERED_FITS:
        raise SystemExit("grid does not match the pre-registered 100 fits")

    ck: Dict[str, Any] = {}
    for mk in MODEL_KEYS:
        for s in SEEDS:
            rec = ad_meta["models"][mk]["checkpoints"][str(s)]
            path = os.path.join(args.checkpoints_dir, CHECKPOINT_TEMPLATE.format(model=mk, seed=s))
            sha = sha256_file(path)
            if sha != rec["sha256"]:
                raise SystemExit(f"checkpoint drift before run: {mk} seed={s}")
            ck[f"{mk}|{s}"] = {"path": path, "sha256": sha,
                               "ad_recorded_path": rec["path"], "ad_recorded_sha256": rec["sha256"]}

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    meta: Dict[str, Any] = {
        "experiment": "AH — bounded-reader capacity / inductive-bias sensitivity at V's scale",
        "date": ts,
        "verification_timestamp": ts,
        "source": {"ad_npz": args.ad_npz, "ad_npz_sha256": sha256_file(args.ad_npz),
                   "ad_meta": args.ad_meta, "ad_meta_sha256": sha256_file(args.ad_meta),
                   "v_npz": args.v_npz, "v_npz_sha256": v_sha, "ad_v_array_identity": True,
                   "array_sha256_verified": n_arrays},
        "trained_anything": False,
        "protocol_matches_v": {"readers": READERS, "histories": HISTORIES, "seeds": SEEDS},
        "held_out_b": list(HELD_OUT_B),
        "grid": {"definition": GRID_DEFINITION, "n_readers": len(READERS),
                 "n_widths": len(LEVELS), "n_histories": len(HISTORIES), "n_seeds": len(SEEDS),
                 "n_fits": PRE_REGISTERED_FITS},
        "capacity_ladder": ladder,
        "eligible_pool": pools,
        "invariants_ok": True,
        "checkpoints": ck,
    }

    for mk in MODEL_KEYS:
        for s in SEEDS:
            path = os.path.join(args.checkpoints_dir, CHECKPOINT_TEMPLATE.format(model=mk, seed=s))
            if sha256_file(path) != ad_meta["models"][mk]["checkpoints"][str(s)]["sha256"]:
                raise SystemExit(f"checkpoint changed during AH verification: {mk} seed={s}")
    if sha256_file(args.v_npz) != v_sha:
        raise SystemExit("V source changed during AH verification")
    meta["post_run"] = {"all_checkpoints_unchanged": True, "v_npz_sha256": sha256_file(args.v_npz),
                        "ad_npz_sha256": sha256_file(args.ad_npz)}

    os.makedirs(os.path.dirname(args.meta_output), exist_ok=True)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"V source verified ({v_sha[:12]}); AD==V array identity True; "
          f"array SHA-256 re-verified: {n_arrays}")
    print("target derivation / class coverage / disjointness / history causality: all OK")
    for r in ladder["levels"]:
        print(f"  {r['level_key']:12s} GRU({r['gru_hidden']:3d})={r['gru_params']:6d} "
              f"({r['gru_ratio_vs_baseline']:.3f}x)  TCN({r['tcn_channels']:3d})={r['tcn_params']:6d} "
              f"({r['tcn_ratio_vs_baseline']:.3f}x)  within {CAP_RATIO:g}x: {r['within_cap']}")
    for mk in MODEL_KEYS:
        sizes = ", ".join(f"{s}:{v['eligible']}" for s, v in pools["models"][mk].items())
        print(f"  eligible {mk}: {sizes}")
    print(f"grid: {GRID_DEFINITION}; checkpoints verified unchanged: True")
    print(f"Wrote {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
