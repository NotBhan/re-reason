#!/usr/bin/env python3
"""Experiment S — Controlled Candidate-Internal Representation Investigation (collector).

Q found only weak recoverability of the true written value from the frozen candidate vector C_t on
the Class-A (candidate-wrong) test population; R showed that survives controlled capacity and
size-matched distribution changes and that the only regime that raises apparent accuracy is
operand-combination memorization (R-D). Neither could say WHERE in the candidate computation the
information is lost — output projection/geometry vs a deeper candidate-computation limit.

S localises that loss by measuring recoverability from the ACTUAL internal activations of the frozen
candidate network (forward-only, torch.no_grad(), no weight change, no intervention).

Actual candidate network (NonlinearCandidateMGUCore, Model E / F; 226,358 params):
    f_t        = sigmoid(W_f_ih e_t + W_f_hh S_(t-1) + b_f)     # gate                (212)
    h_reset    = f_t * S_(t-1)                                  # pre-gated state     (212)
    cand_input = [h_reset, e_t]                                 # candidate input     (468)
    h1_pre     = cand_mlp[0](cand_input)                        # hidden-1 preact     (146)
    h1_post    = GELU(h1_pre)                                   # hidden-1 postact    (146)
    out_pre    = cand_mlp[2](h1_post)                           # candidate logits    (212)
    C_t        = tanh(out_pre)                                  # final candidate     (212)

There is only ONE hidden layer, not two. The predeclared stage set is captured exactly from the real
implementation via non-invasive forward hooks (the frozen computation is unchanged).

Measurement only: frozen checkpoints, no model training, no optimizer on model weights, no
architecture/candidate/core/gate/commit/readout modification, no inference-time intervention, no
branching/verifier/oracle, no counterfactual rollout. Checkpoint SHA-256 verified before/after.

Outputs
-------
    experiments/experiment_s_candidate_internal.npz        frozen per-stage activations + metadata
    experiments/experiment_s_candidate_internal_meta.json  config, stage map, hashes, split sizes
Reproduce
---------
    python experiments/run_experiment_s.py \
        --checkpoints-dir experiments/checkpoints \
        --output experiments/experiment_s_candidate_internal.npz
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
import run_experiment_l as rl  # noqa: E402
import run_experiment_r as rr  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
ADD = RegisterTrackingTask.OP_ADD
STATE_DIM = gr.STATE_DIM           # 212
CAND_HIDDEN = 146
CAND_INPUT_DIM = STATE_DIM + 256   # 468

# Predeclared activation stages, identified from the ACTUAL implementation (see module docstring).
# Order is the candidate-computation order; `S_prev` is the upstream pre-candidate state.
STAGES: List[str] = ["S_prev", "f_gate", "h_reset", "cand_input", "h1_pre", "h1_post", "out_pre", "C_t"]
STAGE_DIMS: Dict[str, int] = {
    "S_prev": STATE_DIM, "f_gate": STATE_DIM, "h_reset": STATE_DIM, "cand_input": CAND_INPUT_DIM,
    "h1_pre": CAND_HIDDEN, "h1_post": CAND_HIDDEN, "out_pre": STATE_DIM, "C_t": STATE_DIM,
}
# The candidate-network chain (input -> hidden -> output -> C_t).
PRIMARY_CHAIN: List[str] = ["cand_input", "h1_pre", "h1_post", "out_pre", "C_t"]

META_COLS = ["y_true", "raw_sum", "cand_pred", "cand_ok", "m_cand", "offpath", "norm_C", "norm_dC",
             "op_dest", "src", "operand_a", "operand_b", "wrap", "repeated", "t", "seq", "seed"]


def _stage_hook(stage: str, buf: Dict[str, List[torch.Tensor]], pre: bool, post: bool):
    def hook(module, inp, out):
        if pre:
            buf[stage + "__IN"].append(inp[0].detach())
        if post:
            buf[stage].append(out.detach())
    return hook


@torch.no_grad()
def capture_split(model: torch.nn.Module, ds, seed: int, cap: Optional[int],
                  chunk: int = 512) -> Dict[str, np.ndarray]:
    """Capture per-ADD-transition internal activations + metadata for one split.

    Activations are collected with read-only forward hooks inside the existing
    `forward_with_states` (return_components=True); the frozen computation is unchanged.
    """
    inputs = ds.inputs
    step_targets = ds.step_targets
    B, T, _ = inputs.shape
    head = model.decision_head
    R, V, D = gr.NUM_REGISTERS, gr.NUM_VALUES, STATE_DIM
    W, bvec = rl._matrix(head)
    core = model.cognition_core

    stage_buf: Dict[str, List[np.ndarray]] = {k: [] for k in STAGES}
    meta_buf: Dict[str, List[np.ndarray]] = {k: [] for k in META_COLS}
    collected = 0

    for i0 in range(0, B, chunk):
        if cap is not None and collected >= cap:
            break
        xb = inputs[i0:i0 + chunk]
        stb = step_targets[i0:i0 + chunk]
        bsz = xb.shape[0]

        buf: Dict[str, List[torch.Tensor]] = {"cand_input": [], "h1_pre": [], "h1_post": [],
                                              "out_pre": [], "cand_input__IN": []}
        h0 = core.cand_mlp[0].register_forward_hook(
            _stage_hook("cand_input", buf, pre=True, post=False))
        h1 = core.cand_mlp[0].register_forward_hook(
            _stage_hook("h1_pre", buf, pre=False, post=True))
        h2 = core.cand_mlp[1].register_forward_hook(
            _stage_hook("h1_post", buf, pre=False, post=True))
        h3 = core.cand_mlp[2].register_forward_hook(
            _stage_hook("out_pre", buf, pre=False, post=True))
        try:
            diag = model.forward_with_states(xb, return_components=True)
        finally:
            for h in (h0, h1, h2, h3):
                h.remove()

        states = torch.stack(diag.states, 1)        # (b,T,D) committed S_t
        cands = torch.stack(diag.candidates, 1)     # (b,T,D) final C_t
        gates = torch.stack(diag.gates, 1)          # (b,T,D) f_t
        prev_state = torch.cat([torch.zeros(bsz, 1, D, device=states.device), states[:, :-1]], 1)
        h_reset = gates * prev_state                # exact same op as the core

        act: Dict[str, torch.Tensor] = {
            "S_prev": prev_state,
            "f_gate": gates,
            "h_reset": h_reset,
            "cand_input": torch.stack(buf["cand_input__IN"], 1),
            "h1_pre": torch.stack(buf["h1_pre"], 1),
            "h1_post": torch.stack(buf["h1_post"], 1),
            "out_pre": torch.stack(buf["out_pre"], 1),
            "C_t": cands,
        }

        _, op, reg_a, reg_b = gr.write_mask_from_inputs(xb)
        prev_true = torch.zeros_like(stb)
        prev_true[:, 1:] = stb[:, :-1]

        idx = (op == ADD).nonzero()
        if idx.numel() == 0:
            continue
        b = idx[:, 0]
        t = idx[:, 1]
        if cap is not None:
            room = cap - collected
            idx = idx[:room]
            b, t = idx[:, 0], idx[:, 1]
        n = int(len(idx))
        if n == 0:
            continue

        dest = reg_a[b, t]
        src = reg_b[b, t]
        oa = prev_true[b, t, dest]
        ob = prev_true[b, t, src]
        y = stb[b, t, dest]
        C = cands[b, t]
        Sp = prev_state[b, t]
        Sc = states[b, t]
        wrap = (oa + ob) >= V
        repeated = (t > 0) & (op[b, (t - 1).clamp_min(0)] == ADD)

        ar = torch.arange(n, device=states.device)
        z = torch.einsum("nd,nvd->nv", C, W[dest]) + bvec[dest]
        pred = z.argmax(-1)
        cand_ok = pred == y

        zp = torch.einsum("nd,nvd->nv", Sp, W[dest]) + bvec[dest]
        Mp = zp[ar, y][:, None] - zp
        Mm = Mp.clone()
        Mm[ar, y] = float("inf")
        a_star = Mm.argmin(1)
        m_cand = (z[ar, y][:, None] - z).gather(1, a_star[:, None]).squeeze(1)
        dvec = C - Sp
        denom = (dvec * dvec).sum(-1).clamp_min(1e-12)
        lam = ((Sc - Sp) * dvec).sum(-1) / denom
        offpath = (Sc - (Sp + lam[:, None] * dvec)).norm(dim=1)

        # chunk-local ADD rows, ordered (b-major, t-minor) as in Q/R's `idx[:cap]`
        for k in STAGES:
            stage_buf[k].append(act[k][b, t].detach().cpu().numpy().astype(np.float32))
        meta_buf["y_true"].append(y.detach().cpu().numpy().astype(np.int64))
        meta_buf["raw_sum"].append((oa + ob).detach().cpu().numpy().astype(np.int64))
        meta_buf["cand_pred"].append(pred.detach().cpu().numpy().astype(np.int64))
        meta_buf["cand_ok"].append(cand_ok.detach().cpu().numpy().astype(np.uint8))
        meta_buf["m_cand"].append(m_cand.detach().cpu().numpy().astype(np.float32))
        meta_buf["offpath"].append(offpath.detach().cpu().numpy().astype(np.float32))
        meta_buf["norm_C"].append(C.norm(dim=1).detach().cpu().numpy().astype(np.float32))
        meta_buf["norm_dC"].append(dvec.norm(dim=1).detach().cpu().numpy().astype(np.float32))
        meta_buf["op_dest"].append(dest.detach().cpu().numpy().astype(np.int64))
        meta_buf["src"].append(src.detach().cpu().numpy().astype(np.int64))
        meta_buf["operand_a"].append(oa.detach().cpu().numpy().astype(np.int64))
        meta_buf["operand_b"].append(ob.detach().cpu().numpy().astype(np.int64))
        meta_buf["wrap"].append(wrap.detach().cpu().numpy().astype(np.int64))
        meta_buf["repeated"].append(repeated.detach().cpu().numpy().astype(np.int64))
        meta_buf["t"].append(t.detach().cpu().numpy().astype(np.int64))
        meta_buf["seq"].append((b + i0).detach().cpu().numpy().astype(np.int64))
        meta_buf["seed"].append(np.full(n, seed, np.int64))
        collected += n

    out: Dict[str, np.ndarray] = {}
    for k in STAGES:
        out[k] = np.concatenate(stage_buf[k], 0) if stage_buf[k] else np.zeros((0, STAGE_DIMS[k]), np.float32)
        assert out[k].shape[1] == STAGE_DIMS[k], (k, out[k].shape)
    for k in META_COLS:
        out[k] = np.concatenate(meta_buf[k], 0) if meta_buf[k] else np.zeros((0,), np.int64)
    return out


def concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], 0) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment S collector (measurement only)")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_s_candidate_internal.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_s_candidate_internal_meta.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    model_keys = args.models or MODEL_KEYS
    seeds = args.seeds or gr.SEEDS
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "train_ord_cap": rr.TRAIN_ORD_CAP, "val_ord_cap": rr.VAL_ORD_CAP,
        "extra_stream": rr.EXTRA_STREAM, "extra_seqs": rr.EXTRA_SEQS,
        "arch": "NonlinearCandidateMGUModel (Model E/F), state_dim=212, candidate_hidden_dim=146",
        "candidate_network": "cand_mlp = Sequential(Linear(468,146), GELU, Linear(146,212)); C_t=tanh(out_pre)",
        "stages": STAGES, "stage_dims": STAGE_DIMS, "primary_chain": PRIMARY_CHAIN,
        "models": {},
        "splits": {"train": "DisjointSplitManager train (6000 seq)",
                   "val": "DisjointSplitManager val (1200 seq)",
                   "test": "canonical strict test split (1200 seq, stream 90042+seed)",
                   "hardextra": "augmented i.i.d. stream (base 810000+seed, 5000 seq), sequence-disjoint"},
        "hard_tail": ("ADD transition with frozen candidate written-register decode != true written "
                      "value (Q's Class-A definition, unchanged)"),
        "note": ("Measurement only: frozen checkpoints, no model training/intervention/counterfactual "
                 "rollout. Internal activations are captured read-only via forward hooks inside the "
                 "existing forward_with_states; the frozen computation is unchanged. Analysis-only "
                 "probes (in analyze_experiment_s.py) are fit on a regime's train, tuned on its val, "
                 "evaluated once on the untouched canonical-test hard tail."),
    }
    payload: Dict[str, np.ndarray] = {"meta_json": np.array(["placeholder"])}
    for mk in model_keys:
        meta["models"][mk] = {"checkpoints": {}}
        acc: Dict[str, List[Dict[str, np.ndarray]]] = {k: [] for k in
                                                       ("trainord", "trainhard", "valord", "valhard",
                                                        "test", "hardextra")}
        Ws, bs = [], []
        for seed in seeds:
            loaded = eg.load_checkpoint(args.checkpoints_dir, mk, seed)
            sha_before = loaded["sha256"]
            model = loaded["model"]
            splits = gr.canonical_splits(seed)
            full_tr = capture_split(model, splits.train_set, seed, cap=None)
            full_va = capture_split(model, splits.val_set, seed, cap=None)
            full_te = capture_split(model, splits.test_set, seed, cap=None)
            full_ex = capture_split(model, rr.extra_hard_stream(seed), seed, cap=None)
            n = len(full_tr["y_true"])
            ord_tr = np.arange(min(rr.TRAIN_ORD_CAP, n))
            hard_tr = np.nonzero(full_tr["cand_ok"] == 0)[0]
            m = len(full_va["y_true"])
            ord_va = np.arange(min(rr.VAL_ORD_CAP, m))
            hard_va = np.nonzero(full_va["cand_ok"] == 0)[0]
            hard_ex = np.nonzero(full_ex["cand_ok"] == 0)[0]
            acc["trainord"].append({k: v[ord_tr] for k, v in full_tr.items()})
            acc["trainhard"].append({k: v[hard_tr] for k, v in full_tr.items()})
            acc["valord"].append({k: v[ord_va] for k, v in full_va.items()})
            acc["valhard"].append({k: v[hard_va] for k, v in full_va.items()})
            acc["test"].append(full_te)
            acc["hardextra"].append({k: v[hard_ex] for k, v in full_ex.items()})
            W, bvec = rl._matrix(model.decision_head)
            Ws.append(W.detach().cpu().numpy().astype(np.float32))
            bs.append(bvec.detach().cpu().numpy().astype(np.float32))
            if eg.sha256_file(loaded["path"]) != sha_before:
                raise RuntimeError(f"{mk} seed={seed}: checkpoint changed during run")
            meta["models"][mk]["checkpoints"][str(seed)] = {
                "path": loaded["path"], "sha256": sha_before, "unchanged_after_run": True}
            print(f"  {mk} seed={seed}: train_ord={len(ord_tr)} train_hard={len(hard_tr)} "
                  f"val_ord={len(ord_va)} val_hard={len(hard_va)} test_add={len(full_te['y_true'])} "
                  f"hard_extra={len(hard_ex)}")
        for k, parts in acc.items():
            c = concat(parts)
            for a, v in c.items():
                payload[f"{mk}__{k}__{a}"] = v
        payload[f"{mk}__W"] = np.stack(Ws)
        payload[f"{mk}__b"] = np.stack(bs)

    meta["n"] = {mk: {k: int(payload[f"{mk}__{k}__y_true"].shape[0]) for k in
                      ("trainord", "trainhard", "valord", "valhard", "test", "hardextra")}
                 for mk in model_keys}
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
