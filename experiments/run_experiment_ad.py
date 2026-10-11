#!/usr/bin/env python3
"""Experiment AD — collector + mandatory source verification (measurement only).

AB/AC established that the causal TCN's fitting behaviour is optimizer-recipe sensitive (AdamW can
fit the `op_dest` control and the small-subset `y_true` target, though the small-subset ADD fit does
not generalize). V/W/X produced their ADD-target-access conclusions with the **unqualified**
Adam/default TCN. AD re-evaluates V's original access test under the AB-qualified `adamw` recipe.

This collector does NOT train. It reuses V's exact dataset and stored analysis artifacts verbatim
(`experiments/experiment_v_sequence_access.npz`): the ordinary-ADD training pool (cap 1,500/seed),
ordinary validation pool (cap 800/seed), the untouched canonical TEST population, the frozen-E
Class-A (`cand_ok`) mask (F control), the sequence/timestep provenance and the per-seed assignment.
It copies V's arrays byte-for-byte into the AD artifact and verifies, before and after, the source
SHA-256, exact array identity, the `y_true = (operand_a + operand_b) mod 10` derivation,
10-class coverage, sequence disjointness (trainord/valord/test), absence of test leakage, V's exact
causal history construction (no future observation, zero left-padding, validity mask) and the frozen
E/F checkpoint SHA-256. It fails explicitly on any invariant violation.

Measurement-only; frozen E/F checkpoints unchanged; no RE:Reason modification; no Phase 2.

Outputs
-------
    experiments/experiment_ad_qualified_access.npz        V's arrays verbatim + meta
    experiments/experiment_ad_qualified_access_meta.json  identities, invariants, checkpoints
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_ad.py \
        --v-npz experiments/experiment_v_sequence_access.npz \
        --output experiments/experiment_ad_qualified_access.npz
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
import run_experiment_g as eg  # noqa: E402
from re_reason.benchmarks.dataset_split import hash_sequence  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
SPLITS = ["trainord", "valord", "test"]
T_MAX = au.T_MAX
OBS_DIM = au.D_OBS
NCLASS = 10
SEEDS = [42, 43, 44, 45, 46]
CHECKPOINT_TEMPLATE = "{model}_seed{seed}_3000steps.pt"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def class_counts(y: np.ndarray, nclass: int = NCLASS) -> List[int]:
    return [int((y == c).sum()) for c in range(nclass)]


def assert_all_classes(counts: List[int], where: str) -> None:
    if len(counts) != NCLASS or any(c <= 0 for c in counts):
        raise SystemExit(f"class coverage failure: {where} = {counts}")


def verify_target_derivation(oa: np.ndarray, ob: np.ndarray, y: np.ndarray, where: str) -> Dict[str, Any]:
    """y_true == (operand_a + operand_b) mod 10, exactly."""
    expected = (oa.astype(np.int64) + ob.astype(np.int64)) % NCLASS
    ok = bool(np.array_equal(expected, y.astype(np.int64)))
    if not ok:
        n_bad = int((expected != y.astype(np.int64)).sum())
        raise SystemExit(f"y_true derivation failure: {where} ({n_bad} mismatches)")
    return {"n": int(len(y)), "exact": ok, "rule": "(operand_a + operand_b) mod 10"}


def _seq_hashes(seq_obs: np.ndarray) -> set:
    return {hash_sequence(seq_obs[i].reshape(T_MAX, OBS_DIM)) for i in range(seq_obs.shape[0])}


def verify_no_future(seq_obs: np.ndarray, t: np.ndarray, L: int, rng: np.random.Generator,
                     n_sample: int = 64) -> Dict[str, Any]:
    """V's exact causal builder must use only observations x_0..x_t (zero left-padding + mask)."""
    n = seq_obs.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    X, M = au.build_history(seq_obs[idx], t[idx], L)
    full = seq_obs[idx].reshape(len(idx), T_MAX, OBS_DIM)
    max_used_index = -1
    for i in range(len(idx)):
        ti = int(t[idx][i])
        nn = min(ti + 1, L)
        valid = M[i] > 0.5
        if int(valid.sum()) != nn:
            raise SystemExit(f"history validity failure L={L} row={i}: {valid.sum()} != {nn}")
        if bool(valid[: L - nn].any()):
            raise SystemExit(f"history padding failure L={L} row={i}")
        # the valid slots must equal the causal prefix x_{ti-nn+1}..x_{ti}
        win = full[i, ti - nn + 1:ti + 1].reshape(-1)
        if not np.allclose(X[i, (L - nn) * OBS_DIM:], win):
            raise SystemExit(f"history value failure L={L} row={i}")
        # padded slots must be exactly zero
        if bool((X[i, : (L - nn) * OBS_DIM] != 0).any()):
            raise SystemExit(f"history non-zero padding failure L={L} row={i}")
        max_used_index = max(max_used_index, ti)
    return {"n_sampled": int(len(idx)), "max_used_observation_index": int(max_used_index),
            "rule": "H_L(t) = [x_{t-L+1}..x_t]; padded slots zero, no observation index > t"}


def _contiguous_windows(seq_obs: np.ndarray, t: np.ndarray, L: int) -> np.ndarray:
    """Independent causal window built without U's loop (for a cross-check)."""
    full = seq_obs.reshape(seq_obs.shape[0], T_MAX, OBS_DIM)
    out = np.zeros((seq_obs.shape[0], L, OBS_DIM), np.float32)
    for i in range(seq_obs.shape[0]):
        ti = int(t[i])
        j0 = max(0, ti - L + 1)
        out[i, L - (ti - j0 + 1):] = full[i, j0:ti + 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AD collector + source verification")
    ap.add_argument("--v-npz", default="experiments/experiment_v_sequence_access.npz")
    ap.add_argument("--v-meta", default="experiments/experiment_v_sequence_access_meta.json")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_ad_qualified_access.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_ad_qualified_access_meta.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    if not os.path.exists(args.v_npz):
        raise SystemExit(f"V artifact not found: {args.v_npz}. Run run_experiment_v.py first.")
    v_sha = sha256_file(args.v_npz)
    vv = np.load(args.v_npz, allow_pickle=False)
    v_meta = json.loads(str(vv["meta_json"][0]))
    v_meta_sha = sha256_file(args.v_meta) if os.path.exists(args.v_meta) else None

    meta: Dict[str, Any] = {
        "experiment": "AD — qualified-recipe re-evaluation of the ADD-target access test",
        "target": "y_true", "n_classes": NCLASS,
        "readers": ["GRU", "TCN"], "histories": [r for r, _ in au.REPS],
        "conditions": ["adam_ref", "adamw"],
        "source": {"v_npz": args.v_npz, "v_sha256": v_sha,
                   "v_meta": args.v_meta, "v_meta_sha256": v_meta_sha,
                   "v_train_ord_cap": v_meta.get("train_ord_cap"),
                   "v_val_ord_cap": v_meta.get("val_ord_cap"),
                   "v_n": v_meta.get("n"), "v_date": v_meta.get("date")},
        "identity": {}, "class_counts": {}, "target_derivation": {}, "heldout_pairs": {},
        "disjointness": {}, "history_causality": {}, "models": {},
    }

    payload: Dict[str, np.ndarray] = {}
    for k in vv.files:
        if k == "meta_json":
            continue
        payload[k] = vv[k]
        assert np.array_equal(payload[k], vv[k]), f"copy mismatch {k}"
        meta["identity"][k] = {"sha256": sha256_array(payload[k]), "shape": list(payload[k].shape),
                               "dtype": str(payload[k].dtype)}

    # ---- class coverage / target derivation ----
    meta["class_counts"] = {"trainord": {}, "valord": {}, "test": {}}
    meta["target_derivation"] = {}
    for mk in MODEL_KEYS:
        for ds in SPLITS:
            y = payload[f"{mk}__{ds}__y_true"]
            c = class_counts(y)
            assert_all_classes(c, f"{mk}/{ds}")
            meta["class_counts"][ds][mk] = c
            meta["target_derivation"][f"{mk}/{ds}"] = verify_target_derivation(
                payload[f"{mk}__{ds}__operand_a"], payload[f"{mk}__{ds}__operand_b"], y, f"{mk}/{ds}")

    # held-out-pair coverage of the ordinary-ADD training pool (V's protocol: b in {0,5}); the
    # `main` variant fits on all of it, the `heldout` variant excludes these rows at fit time.
    for mk in MODEL_KEYS:
        ob = payload[f"{mk}__trainord__operand_b"].astype(np.int64)
        mt = payload[f"{mk}__test__operand_b"].astype(np.int64)
        meta["heldout_pairs"][mk] = {
            "trainord_rows_with_b_in_{0,5}": int(np.isin(ob, [0, 5]).sum()),
            "trainord_total": int(len(ob)),
            "test_rows_with_b_in_{0,5}": int(np.isin(mt, [0, 5]).sum()),
            "test_total": int(len(mt)),
            "excluded_pairs_b": [0, 5]}

    # ---- sequence disjointness / no test leakage ----
    for mk in MODEL_KEYS:
        h_tr = _seq_hashes(payload[f"{mk}__trainord__seq_obs"])
        h_va = _seq_hashes(payload[f"{mk}__valord__seq_obs"])
        h_te = _seq_hashes(payload[f"{mk}__test__seq_obs"])
        meta["disjointness"][mk] = {
            "trainord_vs_valord": len(h_tr & h_va),
            "trainord_vs_test": len(h_tr & h_te),
            "valord_vs_test": len(h_va & h_te),
            "n_unique_trainord": len(h_tr), "n_unique_valord": len(h_va), "n_unique_test": len(h_te)}
    leak_keys = ("trainord_vs_valord", "trainord_vs_test", "valord_vs_test")
    if any(meta["disjointness"][mk][k] != 0 for mk in MODEL_KEYS for k in leak_keys):
        raise SystemExit(f"split sequence disjointness failure: {meta['disjointness']}")

    # ---- causal history construction (no future observation) + builder cross-check ----
    rng = np.random.default_rng(20261011)
    for mk in MODEL_KEYS:
        for ds in SPLITS:
            seq = payload[f"{mk}__{ds}__seq_obs"]
            t = payload[f"{mk}__{ds}__t"]
            for rep, L in au.REPS:
                key = f"{mk}/{ds}/{rep}"
                rec = verify_no_future(seq, t, L, rng)
                # cross-check the independent contiguous-window builder matches U's exactly
                X, M = au.build_history(seq, t, L)
                xr = X.reshape(len(seq), L, OBS_DIM)
                cw = _contiguous_windows(seq, t, L)
                if not np.allclose(xr, cw):
                    raise SystemExit(f"history cross-check failure {key}")
                rec["builder_cross_check"] = True
                meta["history_causality"][key] = rec

    # ---- frozen checkpoint hashes (before) ----
    for mk in MODEL_KEYS:
        meta["models"][mk] = {"checkpoints": {}}
        for s in SEEDS:
            p = os.path.join(args.checkpoints_dir, CHECKPOINT_TEMPLATE.format(model=mk, seed=s))
            meta["models"][mk]["checkpoints"][str(s)] = {"path": p, "sha256": sha256_file(p)}

    meta["n"] = {mk: {ds: int(payload[f"{mk}__{ds}__y_true"].shape[0]) for ds in SPLITS}
                 for mk in MODEL_KEYS}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---- write, then recompute and verify hashes/identity AFTER the run ----
    payload["meta_json"] = np.array([json.dumps(meta)])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **payload)

    for mk in MODEL_KEYS:
        for s in SEEDS:
            rec = meta["models"][mk]["checkpoints"][str(s)]
            if sha256_file(rec["path"]) != rec["sha256"]:
                raise SystemExit(f"checkpoint changed after run: {mk} seed={s}")
            rec["unchanged_after_run"] = True
    if sha256_file(args.v_npz) != v_sha:
        raise SystemExit("V source changed during collector run")
    zz = np.load(args.output, allow_pickle=False)
    for k in vv.files:
        if k == "meta_json":
            continue
        if not np.array_equal(zz[k], vv[k]):
            raise SystemExit(f"written array identity failure: {k}")
    meta["post_run"] = {"v_sha256": sha256_file(args.v_npz),
                        "all_checkpoints_unchanged": True,
                        "written_arrays_identical": True}
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Copied {len(meta['identity'])} arrays verbatim from {args.v_npz}")
    print(f"target derivation exact: "
          f"{all(r['exact'] for r in meta['target_derivation'].values())}")
    print(f"disjointness: { {mk: {k: v for k, v in meta['disjointness'][mk].items() if k.startswith(('trainord_vs', 'valord_vs'))} for mk in MODEL_KEYS} }")
    print(f"history causality verified for {len(meta['history_causality'])} (model/split/rep) cells")
    print(f"checkpoints verified unchanged: "
          f"{ {mk: all(c['unchanged_after_run'] for c in meta['models'][mk]['checkpoints'].values()) for mk in MODEL_KEYS} }")
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
