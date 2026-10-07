#!/usr/bin/env python3
"""Experiment H — λ-interpolation decode sweep (measurement only).

Question: when an ADD candidate is already correct but the actual committed state is
wrong, does the old→candidate state path contain a point the existing readout can decode?

Method (no training, no model changes, no rollouts):
  * Frozen checkpoints E_candidate and F_combined, seeds 42–46 (SHA-256 pinned).
  * One forward pass with components per checkpoint over the strict test split and the
    existing controlled ADD builders (same data and ordering as Experiment G).
  * For every ADD transition the following vectors are captured: S_(t-1), C_t, S_t, the
    212-dim gate f_t. (Experiment G recorded only scalar summaries, so the vectors must be
    captured again — the G dataset is reused for row alignment and verification.)
  * SCALAR λ SWEEP: digits of head((1-λ)S_(t-1) + λ C_t) at the destination register for
    λ in {0.00, 0.05, ..., 1.00} (21 points), correctness vs the true post-ADD value.
  * ACTUAL VECTOR-GATE REPLAY: S_actual = (1-f_t) ⊙ S_(t-1) + f_t ⊙ C_t; verified against
    the recorded S_t; decoded and compared with the recorded committed decode.
  * MEAN-GATE COUNTERFACTUAL: S_mean = (1-f̄_t) S_(t-1) + f̄_t C_t with f̄_t the full-vector
    mean gate (explicitly a counterfactual scalarization, not the actual commit).
  * Additional scalars: λ_eff (projection of S_t onto the S_(t-1)→C_t segment), off-path
    distance, distances, probabilities.

IMPORTANT: the actual commit uses the 212-dim gate element-wise, so S(f̄_t) is NOT the real
commit. All comparisons between scalar λ and the actual vector-gate state are descriptive.

Outputs:
    experiments/experiment_h_interp.npz        per-transition scalars + digit/correct matrices
    experiments/experiment_h_interp_meta.json  columns, λ grid, checkpoint hashes, G checks

Reproduce:
    python experiments/run_experiment_h.py \
        --checkpoints-dir experiments/checkpoints \
        --g-transitions experiments/experiment_g_transitions.npz \
        --output experiments/experiment_h_interp.npz
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiment_g as eg  # noqa: E402  (collector conventions, checkpoint loading)
import run_generator_repair_feasibility as gr  # noqa: E402

LAMBDAS = np.round(np.arange(0.0, 1.0001, 0.05), 2).astype(np.float32)  # 21 points
N_LAMBDA = len(LAMBDAS)

SCALAR_COLS = [
    "seed", "source", "seq_id", "t", "dest", "src_reg", "operand_a", "operand_b",
    "wrap", "isolated",
    "cand_ok", "comm_ok", "prev_ok", "comm_eq_old",
    "cand_pred", "comm_pred", "prev_pred", "replay_pred", "mean_pred",
    "replay_ok", "mean_ok", "cand_ok_direct",
    "f_mean", "f_min", "f_max",
    "d_cs", "d_ss", "d_sc",
    "replay_maxdiff", "lambda_eff", "offpath_dist",
    "cand_pt", "comm_pt",
    "lambda_star", "n_correct",
]


def _dec_dest(head: torch.nn.Module, x: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
    """Destination-register argmax decode for a batch of states x (n, D)."""
    logits = head(x)  # (n, R, V)
    rows = torch.arange(x.shape[0], device=x.device)
    return logits[rows, dest].argmax(-1)


@torch.no_grad()
def sweep_dataset(model: torch.nn.Module, inputs: torch.Tensor, step_targets: torch.Tensor,
                  seed: int, source_id: int, chunk: int = 2048) -> Dict[str, np.ndarray]:
    """Per-ADD-transition scalar record + λ-sweep digit/correctness matrices."""
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    step_targets = step_targets.to(device)
    B, T, _ = inputs.shape
    mask, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, dim=1)       # (B, T, D)
    cands = torch.stack(diag.candidates, dim=1)
    gates = torch.stack(diag.gates, dim=1)
    prev_state = torch.cat([torch.zeros(B, 1, gr.STATE_DIM, device=device), states[:, :-1]], dim=1)
    prev_true = torch.zeros_like(step_targets)
    prev_true[:, 1:] = step_targets[:, :-1]

    idx = (op == gr.RegisterTrackingTask.OP_ADD).nonzero()
    b, t = idx[:, 0], idx[:, 1]
    n = int(len(idx))
    S_prev, C, S_t, fvec = prev_state[b, t], cands[b, t], states[b, t], gates[b, t]
    dest, src = reg_a[b, t], reg_b[b, t]
    oa, ob = prev_true[b, t, dest], prev_true[b, t, src]
    true_val = step_targets[b, t, dest]

    head = model.decision_head
    cand_pred = _dec_dest(head, C, dest)
    comm_pred = _dec_dest(head, S_t, dest)
    prev_pred = _dec_dest(head, S_prev, dest)

    replay = (1.0 - fvec) * S_prev + fvec * C
    replay_maxdiff = (replay - S_t).abs().max(dim=-1).values
    replay_pred = _dec_dest(head, replay, dest)

    f_mean = fvec.mean(dim=-1)
    mean_state = (1.0 - f_mean)[:, None] * S_prev + f_mean[:, None] * C
    mean_pred = _dec_dest(head, mean_state, dest)

    d_cs = (C - S_prev).norm(dim=-1)
    d_ss = (S_t - S_prev).norm(dim=-1)
    d_sc = (S_t - C).norm(dim=-1)

    # Projection of the actual committed state onto the S_prev -> C segment.
    diff = C - S_prev
    denom = (diff * diff).sum(dim=-1).clamp_min(1e-12)
    lambda_eff = (((S_t - S_prev) * diff).sum(dim=-1) / denom).clamp(-1.0, 2.0)
    path_pt = S_prev + lambda_eff[:, None] * diff
    offpath_dist = (S_t - path_pt).norm(dim=-1)

    probs_c = head(C).softmax(-1)
    probs_s = head(S_t).softmax(-1)
    rows_ar = torch.arange(n, device=device)
    cand_pt = probs_c[rows_ar, dest, true_val]
    comm_pt = probs_s[rows_ar, dest, true_val]

    # λ sweep (chunked to bound memory).
    digits = np.zeros((n, N_LAMBDA), dtype=np.int8)
    correct = np.zeros((n, N_LAMBDA), dtype=np.uint8)
    lambda_star = np.full(n, np.nan, dtype=np.float64)
    n_correct = np.zeros(n, dtype=np.int32)
    lam_t = torch.from_numpy(LAMBDAS)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        m = sl.stop - sl.start
        lam = lam_t[None, :, None]
        s_l = (1.0 - lam) * S_prev[sl][:, None, :] + lam * C[sl][:, None, :]  # (m,21,D)
        lg = head(s_l.reshape(-1, gr.STATE_DIM)).reshape(m, N_LAMBDA, gr.NUM_REGISTERS, gr.NUM_VALUES)
        dg = lg.argmax(-1)  # (m,21,R)
        dd = dg[torch.arange(m, device=device)[:, None],
                torch.arange(N_LAMBDA, device=device)[None, :],
                dest[sl][:, None]]
        cor = (dd == true_val[sl][:, None]).cpu().numpy().astype(np.uint8)
        digits[sl] = dd.cpu().numpy().astype(np.int8)
        correct[sl] = cor
        has = cor.any(axis=1)
        first = np.argmax(cor, axis=1)
        rows_full = np.arange(sl.start, sl.stop)[has]
        lambda_star[rows_full] = LAMBDAS[first[has]]
        n_correct[sl] = cor.sum(axis=1)

    to_np = lambda x: x.detach().cpu().numpy()
    out = {
        "seed": np.full(n, seed, dtype=np.float32), "source": np.full(n, source_id, np.float32),
        "seq_id": to_np(b).astype(np.float32), "t": to_np(t).astype(np.float32),
        "dest": to_np(dest).astype(np.float32), "src_reg": to_np(src).astype(np.float32),
        "operand_a": to_np(oa).astype(np.float32), "operand_b": to_np(ob).astype(np.float32),
        "wrap": to_np(((oa + ob) >= gr.NUM_VALUES).float()),
        "isolated": to_np((((t == 0) | (op[b, t - 1] != gr.RegisterTrackingTask.OP_ADD)).float())),
        "cand_ok": to_np((cand_pred == true_val).float()),
        "comm_ok": to_np((comm_pred == true_val).float()),
        "prev_ok": to_np((prev_pred == oa).float()),
        "comm_eq_old": to_np((comm_pred == oa).float()),
        "cand_pred": to_np(cand_pred).astype(np.float32),
        "comm_pred": to_np(comm_pred).astype(np.float32),
        "prev_pred": to_np(prev_pred).astype(np.float32),
        "replay_pred": to_np(replay_pred).astype(np.float32),
        "mean_pred": to_np(mean_pred).astype(np.float32),
        "replay_ok": to_np((replay_pred == true_val).float()),
        "mean_ok": to_np((mean_pred == true_val).float()),
        "cand_ok_direct": to_np((cand_pred == true_val).float()),
        "f_mean": to_np(f_mean), "f_min": to_np(fvec.min(dim=-1).values), "f_max": to_np(fvec.max(dim=-1).values),
        "d_cs": to_np(d_cs), "d_ss": to_np(d_ss), "d_sc": to_np(d_sc),
        "replay_maxdiff": to_np(replay_maxdiff), "lambda_eff": to_np(lambda_eff),
        "offpath_dist": to_np(offpath_dist),
        "cand_pt": to_np(cand_pt), "comm_pt": to_np(comm_pt),
        "lambda_star": lambda_star.astype(np.float32), "n_correct": n_correct.astype(np.float32),
        "digits": digits, "correct": correct,
    }
    return out


def _concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in parts[0]:
        out[key] = np.concatenate([p[key] for p in parts], axis=0)
    return out


def verify_against_experiment_g(h: Dict[str, np.ndarray], g_arr: np.ndarray,
                                g_cols: List[str], model_key: str) -> Dict[str, Any]:
    """Row-level verification that H's re-collected transitions match Experiment G's dataset."""
    gi = {c: i for i, c in enumerate(g_cols)}
    checks: Dict[str, Any] = {"model": model_key, "n_rows_h": int(len(h["seed"])), "n_rows_g": int(len(g_arr))}
    if len(h["seed"]) != len(g_arr):
        raise RuntimeError(f"{model_key}: row count mismatch H={len(h['seed'])} G={len(g_arr)}")
    exact_cols = [("seed", "seed"), ("source", "source"), ("seq_id", "seq_id"), ("t", "t"),
                  ("dest", "dest"), ("src_reg", "src_reg"), ("operand_a", "operand_a"),
                  ("operand_b", "operand_b"), ("wrap", "wrap"), ("isolated", "isolated"),
                  ("cand_ok", "cand_ok"), ("comm_ok", "comm_ok"), ("prev_ok", "prev_ok"),
                  ("comm_eq_old", "comm_eq_old"), ("cand_pred", "cand_pred"),
                  ("comm_pred", "comm_pred"), ("prev_pred", "prev_pred")]
    mismatches = {}
    for h_col, g_col in exact_cols:
        a = np.round(h[h_col]).astype(np.int64)
        b = np.round(g_arr[:, gi[g_col]]).astype(np.int64)
        n_bad = int((a != b).sum())
        if n_bad:
            mismatches[h_col] = n_bad
    float_cols = [("f_mean", "f_mean"), ("d_cs", "d_cs"), ("d_ss", "d_ss"), ("d_sc", "d_sc")]
    max_diffs = {}
    for h_col, g_col in float_cols:
        d = float(np.abs(h[h_col] - g_arr[:, gi[g_col]]).max())
        max_diffs[h_col] = d
        if d > 1e-4:
            mismatches[f"{h_col}_float"] = d
    checks["exact_mismatches"] = mismatches
    checks["max_float_diffs"] = max_diffs
    checks["passed"] = len(mismatches) == 0
    if not checks["passed"]:
        raise RuntimeError(f"{model_key}: H/G verification failed: {mismatches}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment H runner (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--g-transitions", default="experiments/experiment_g_transitions.npz")
    ap.add_argument("--output", default="experiments/experiment_h_interp.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_h_interp_meta.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    gz = np.load(args.g_transitions, allow_pickle=False)
    g_meta = json.loads(str(gz["meta_json"][0]))
    builders = gr._build_controlled_add_sequences()
    meta: Dict[str, Any] = {
        "lambdas": [float(x) for x in LAMBDAS], "scalar_cols": SCALAR_COLS,
        "source_ids": eg.SOURCE_IDS, "op_add": int(gr.RegisterTrackingTask.OP_ADD),
        "models": {}, "g_verification": {},
        "note": ("Measurement only: frozen checkpoints, no training, no optimizer, no weight "
                 "modification. S(f̄_t) is a counterfactual scalarization; the actual commit uses "
                 "the 212-dim element-wise gate."),
    }
    per_model: Dict[str, Dict[str, np.ndarray]] = {}
    for model_key in eg.MODELS:
        g_arr = gz[model_key]
        parts: List[Dict[str, np.ndarray]] = []
        meta["models"][model_key] = {"checkpoints": {}}
        for seed in gr.SEEDS:
            loaded = eg.load_checkpoint(args.checkpoints_dir, model_key, seed)
            sha_before = loaded["sha256"]
            g_sha = g_meta["models"][model_key]["checkpoints"][str(seed)]["sha256"]
            if sha_before != g_sha:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint hash changed vs Experiment G")
            model = loaded["model"]
            splits = gr.canonical_splits(seed)
            ds = splits.test_set
            batch = ds.get_batch(len(ds), idx=0)
            parts.append(sweep_dataset(model, batch.inputs, batch.metadata["step_targets"],
                                       seed, eg.SOURCE_IDS["test"]))
            for name, d in builders.items():
                parts.append(sweep_dataset(model, torch.from_numpy(d["inputs"]),
                                           torch.from_numpy(d["step_targets"]),
                                           seed, eg.SOURCE_IDS[name]))
            sha_after = eg.sha256_file(loaded["path"])
            if sha_after != sha_before:
                raise RuntimeError(f"{model_key} seed={seed}: checkpoint file changed during run")
            meta["models"][model_key]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {model_key} seed={seed}: rows={sum(len(p['seed']) for p in parts)}")
        h = _concat(parts)
        meta["g_verification"][model_key] = verify_against_experiment_g(h, g_arr, g_meta["columns"], model_key)
        meta["g_verification"][model_key]["passed"] = (
            meta["g_verification"][model_key]["passed"]
            and meta["g_verification"][model_key]["max_float_diffs"]["f_mean"] < 1e-5)
        per_model[model_key] = h
        print(f"  {model_key}: {len(h['seed'])} transitions collected; G verification passed="
              f"{meta['g_verification'][model_key]['passed']}")

    meta["n_rows"] = {k: int(v["seed"].shape[0]) for k, v in per_model.items()}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload = {"meta_json": np.array([json.dumps(meta)])}
    for model_key, h in per_model.items():
        payload[f"{model_key}__scalars"] = np.stack([h[c] for c in SCALAR_COLS], axis=1).astype(np.float32)
        payload[f"{model_key}__digits"] = h["digits"]
        payload[f"{model_key}__correct"] = h["correct"]
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
