#!/usr/bin/env python3
"""Experiment G — collector: per-ADD-transition records for the frozen E and F checkpoints.

Measurement only. No training, no model changes, no rollouts. For every ADD transition in
(a) the canonical strict test split (1,200 sequences x 10 steps per seed) and
(b) the existing controlled ADD builders (isolated / ADD->SET / 4x repeated / no-wrap / wrap),
this runner records geometry, gate statistics and candidate/committed decoding.

Gate statistic convention (important):
    f_t is a 212-dimensional per-coordinate interpolation vector. The existing implementation
    (`run_detailed_gate_diagnostics` in experiments/run_parameterization_isolation.py) reports
    only full-vector summary statistics; there is NO principled destination- or
    register-specific gate coordinate, and we do not invent one. The per-transition scalar
    used throughout is therefore the full-vector mean gate (consistent with the existing
    `mean_gate` convention), reported alongside the full-vector min and max.

Outputs:
    experiments/experiment_g_transitions.npz          raw column-oriented transition tables (E, F)
    experiments/experiment_g_transitions_meta.json    metadata sidecar (columns, sources, hashes)

Reproduce:
    python experiments/run_experiment_g.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_g_transitions.npz
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

MODELS = ["E_candidate", "F_combined"]
CHECKPOINT_TEMPLATE = "{model}_seed{seed}_3000steps.pt"
OP_ADD = RegisterTrackingTask.OP_ADD

# Source ids: 0 = canonical test split; 1..5 = controlled builders.
SOURCE_IDS = {"test": 0, "isolated_add": 1, "add_then_set": 2, "repeated_add": 3,
              "add_no_wrap": 4, "add_wrap": 5}
SOURCE_NAMES = {v: k for k, v in SOURCE_IDS.items()}

COLS = [
    "seed", "source", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b",
    "wrap", "isolated", "cand_ok", "comm_ok", "prev_ok", "comm_eq_old",
    "cand_pred", "comm_pred", "prev_pred",
    "f_mean", "f_min", "f_max",
    "d_cs", "d_ss", "d_sc", "cos_sc",
    "cand_ce", "comm_ce", "cand_pt", "comm_pt", "cand_pmax", "comm_pmax",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(checkpoints_dir: str, model_key: str, seed: int) -> Dict[str, Any]:
    name = CHECKPOINT_TEMPLATE.format(model=model_key, seed=seed)
    path = os.path.join(checkpoints_dir, name)
    model = gr.make_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return {"model": model, "path": path, "sha256": sha256_file(path)}


@torch.no_grad()
def _decode_logits(head: torch.nn.Module, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(argmax (...,R), softmax probs (...,R,V)) from a stack of states (...,D)."""
    lead = states.shape[:-1]
    logits = head(states.reshape(-1, gr.STATE_DIM)).reshape(*lead, gr.NUM_REGISTERS, gr.NUM_VALUES)
    return logits.argmax(-1), logits.softmax(-1)


@torch.no_grad()
def collect_from_model(model: torch.nn.Module, seed: int, source_id: int,
                       inputs: torch.Tensor, step_targets: torch.Tensor) -> List[List[float]]:
    """Per-ADD-transition rows for one (model, seed, dataset)."""
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    step_targets = step_targets.to(device)
    B, T, _ = inputs.shape
    mask, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, dim=1)      # (B, T, D)
    cands = torch.stack(diag.candidates, dim=1)   # (B, T, D)
    gates = torch.stack(diag.gates, dim=1)        # (B, T, D)

    cp, probs_c = _decode_logits(model.decision_head, states)
    kp, probs_k = _decode_logits(model.decision_head, cands)
    prev_state = torch.cat([torch.zeros(B, 1, gr.STATE_DIM, device=device), states[:, :-1]], dim=1)
    pp, _ = _decode_logits(model.decision_head, prev_state)

    # Vectorized geometry over all transitions: distances and cosine.
    d_cs_all = (cands - prev_state).norm(dim=-1)
    d_ss_all = (states - prev_state).norm(dim=-1)
    d_sc_all = (states - cands).norm(dim=-1)
    cos_sc_all = F.cosine_similarity(states, cands, dim=-1, eps=1e-12)

    prev_true = torch.zeros_like(step_targets)
    prev_true[:, 1:] = step_targets[:, :-1]

    eps = 1e-12
    rows: List[List[float]] = []
    for b in range(B):
        for t in range(T):
            if int(op[b, t]) != OP_ADD:
                continue
            dest = int(reg_a[b, t])
            src = int(reg_b[b, t])
            oa = int(prev_true[b, t, dest])
            ob = int(prev_true[b, t, src])
            true_val = int(step_targets[b, t, dest])

            cand_ok = int(kp[b, t, dest]) == true_val
            comm_ok = int(cp[b, t, dest]) == true_val
            prev_ok = int(pp[b, t, dest]) == oa
            comm_eq_old = int(cp[b, t, dest]) == oa

            f = gates[b, t]
            d_cs = float(d_cs_all[b, t].item())
            d_ss = float(d_ss_all[b, t].item())
            d_sc = float(d_sc_all[b, t].item())
            cos_sc = float(cos_sc_all[b, t].item())

            cand_pt = float(probs_k[b, t, dest, true_val].item())
            comm_pt = float(probs_c[b, t, dest, true_val].item())
            cand_ce = -float(np.log(max(cand_pt, eps)))
            comm_ce = -float(np.log(max(comm_pt, eps)))

            isolated = (t == 0) or (int(op[b, t - 1]) != OP_ADD)
            wrap = (oa + ob) >= gr.NUM_VALUES

            rows.append([
                seed, source_id, b, t, dest, src, oa, ob,
                int(wrap), int(isolated), int(cand_ok), int(comm_ok), int(prev_ok), int(comm_eq_old),
                int(kp[b, t, dest]), int(cp[b, t, dest]), int(pp[b, t, dest]),
                float(f.mean().item()), float(f.min().item()), float(f.max().item()),
                d_cs, d_ss, d_sc, cos_sc,
                cand_ce, comm_ce, cand_pt, comm_pt,
                float(probs_k[b, t, dest].max().item()), float(probs_c[b, t, dest].max().item()),
            ])
    return rows


def collect_all(checkpoints_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    builders = gr._build_controlled_add_sequences()
    per_model: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {"models": {}, "seeds": gr.SEEDS, "columns": COLS,
                            "source_ids": SOURCE_IDS, "op_add": OP_ADD}
    for model_key in MODELS:
        all_rows: List[List[float]] = []
        meta["models"][model_key] = {"checkpoints": {}}
        for seed in gr.SEEDS:
            loaded = load_checkpoint(checkpoints_dir, model_key, seed)
            model = loaded["model"]
            meta["models"][model_key]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": loaded["sha256"]}
            splits = gr.canonical_splits(seed)
            ds = splits.test_set
            batch = ds.get_batch(len(ds), idx=0)
            all_rows += collect_from_model(model, seed, SOURCE_IDS["test"],
                                           batch.inputs, batch.metadata["step_targets"])
            for name, d in builders.items():
                all_rows += collect_from_model(model, seed, SOURCE_IDS[name],
                                               torch.from_numpy(d["inputs"]),
                                               torch.from_numpy(d["step_targets"]))
            print(f"  {model_key} seed={seed}: cumulative rows={len(all_rows)}")
        per_model[model_key] = np.asarray(all_rows, dtype=np.float64)
    meta["n_rows"] = {k: int(v.shape[0]) for k, v in per_model.items()}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return per_model, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment G collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_g_transitions.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_g_transitions_meta.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    per_model, meta = collect_all(args.checkpoints_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, meta_json=np.array([json.dumps(meta)]), **per_model)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} ({meta['n_rows']}) and {args.meta_output} "
          f"in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
