#!/usr/bin/env python3
"""Experiment R — Candidate-Representation Information Test on the Class-A Tail.

Q found only weak recoverability of the true written value from the frozen candidate vector C_t on
the Class-A (candidate-wrong) test population and could not tell whether that is a genuine
information limit or an artifact of probe capacity / training on the mostly-easy distribution.
R resolves the ambiguity with a 2-D design — capacity x training-distribution — evaluated on the
SAME hard held-out test population.

Factor A (capacity): P0 frozen head, P1 linear ridge, P2/P3/P4/P5 small MLPs (16/32/64/128).
Factor B (distribution): D0 ordinary-train (all ADD, Q's protocol), D1 hard-tail-train (only
candidate-wrong ADD).

Measurement only: frozen checkpoints, no model training, no optimizer on model weights, no
architecture/candidate/core/gate/commit/readout modification, no intervention, no branching/
verifier/oracle, no counterfactual rollout. Analysis-only probes are fit on TRAIN (D0) or the
hard-train subset, tuned on VAL / hard-val, and evaluated once on the untouched canonical TEST
hard tail; they never enter model inference. All model execution under torch.no_grad().
Checkpoint SHA-256 verified before/after.

Outputs
-------
    experiments/experiment_r_candidate_representation.npz   frozen C_t (ordinary + hard splits) + W,b
    experiments/experiment_r_candidate_representation_meta.json
Reproduce
---------
    python experiments/run_experiment_r.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_r_candidate_representation.npz
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402
import run_experiment_q as rq  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset, hash_sequence  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
TRAIN_ORD_CAP = 1500     # ordinary-distribution train subsample (Q protocol)
VAL_ORD_CAP = 800        # ordinary-distribution val subsample
EXTRA_STREAM = 810000    # i.i.d. augmented hard-tail stream (disjoint from train/val/test)
EXTRA_SEQS = 5000


def extra_hard_stream(seed: int) -> SplitDataset:
    """Extra i.i.d. sequences from the same task distribution, via a stream disjoint from
    train/val/test (the frozen model memorized its designated train split, leaving only ~144
    candidate errors there; this pool supplies a data-adequate hard-tail training set)."""
    task = gr.build_task(10)
    rng = np.random.default_rng(EXTRA_STREAM + seed)
    seen, inputs, steps, finals = set(), [], [], []
    while len(inputs) < EXTRA_SEQS:
        inp, st, ft = task.generate_single_sequence(rng)
        h = hash_sequence(inp)
        if h in seen:
            continue
        seen.add(h); inputs.append(inp); steps.append(st); finals.append(ft)
    return SplitDataset(torch.from_numpy(np.stack(inputs)).float(),
                        torch.from_numpy(np.stack(finals)).long(),
                        torch.from_numpy(np.stack(steps)).long(), [])


def split_arrays(full: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[idx] for k, v in full.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment R collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_r_candidate_representation.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_r_candidate_representation_meta.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    model_keys = args.models or MODEL_KEYS
    seeds = args.seeds or gr.SEEDS
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "train_ord_cap": TRAIN_ORD_CAP, "val_ord_cap": VAL_ORD_CAP, "models": {},
        "splits": {"train": "DisjointSplitManager train (6000 seq)", "val": "DisjointSplitManager val (1200 seq)",
                   "test": "canonical strict test split (1200 seq, stream 90042+seed)"},
        "hard_tail": ("ADD transition with frozen candidate written-register decode != true written "
                      "value (Q's Class-A definition, unchanged)"),
        "note": ("Measurement only: frozen checkpoints, no model training/intervention/counterfactual "
                 "rollout. Analysis-only probes are fit on ordinary-train or hard-train, tuned on the "
                 "matching val split, evaluated once on the untouched canonical-test hard tail."),
    }
    payload: Dict[str, np.ndarray] = {"meta_json": np.array(["placeholder"])}
    for mk in model_keys:
        meta["models"][mk] = {"checkpoints": {}}
        acc: Dict[str, List[Dict[str, np.ndarray]]] = {k: [] for k in
                                                       ("trainord", "trainhard", "valord", "valhard", "test", "hardextra")}
        Ws, bs = [], []
        for seed in seeds:
            loaded = eg.load_checkpoint(args.checkpoints_dir, mk, seed)
            sha_before = loaded["sha256"]
            model = loaded["model"]
            splits = gr.canonical_splits(seed)
            full_tr = rq.capture_split(model, splits.train_set, seed, cap=None)
            full_va = rq.capture_split(model, splits.val_set, seed, cap=None)
            full_te = rq.capture_split(model, splits.test_set, seed, cap=None)
            full_ex = rq.capture_split(model, extra_hard_stream(seed), seed, cap=None)
            n = len(full_tr["C"])
            ord_tr = np.arange(min(TRAIN_ORD_CAP, n))
            hard_tr = np.nonzero(full_tr["cand_ok"] == 0)[0]
            m = len(full_va["C"])
            ord_va = np.arange(min(VAL_ORD_CAP, m))
            hard_va = np.nonzero(full_va["cand_ok"] == 0)[0]
            hard_ex = np.nonzero(full_ex["cand_ok"] == 0)[0]
            acc["trainord"].append(split_arrays(full_tr, ord_tr))
            acc["trainhard"].append(split_arrays(full_tr, hard_tr))
            acc["valord"].append(split_arrays(full_va, ord_va))
            acc["valhard"].append(split_arrays(full_va, hard_va))
            acc["test"].append(full_te)
            acc["hardextra"].append(split_arrays(full_ex, hard_ex))
            W, bvec = rq.rl._matrix(model.decision_head)
            Ws.append(W.detach().cpu().numpy().astype(np.float32))
            bs.append(bvec.detach().cpu().numpy().astype(np.float32))
            if eg.sha256_file(loaded["path"]) != sha_before:
                raise RuntimeError(f"{mk} seed={seed}: checkpoint changed during run")
            meta["models"][mk]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {mk} seed={seed}: train_ord={len(ord_tr)} train_hard={len(hard_tr)} "
                  f"val_ord={len(ord_va)} val_hard={len(hard_va)} test_add={len(full_te['C'])} "
                  f"hard_extra={len(hard_ex)}")
        for k, parts in acc.items():
            c = rq.concat(parts)
            for a, v in c.items():
                payload[f"{mk}__{k}__{a}"] = v
        payload[f"{mk}__W"] = np.stack(Ws); payload[f"{mk}__b"] = np.stack(bs)

    meta["n"] = {mk: {k: int(payload[f"{mk}__{k}__C"].shape[0]) for k in
                      ("trainord", "trainhard", "valord", "valhard", "test", "hardextra")} for mk in model_keys}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
