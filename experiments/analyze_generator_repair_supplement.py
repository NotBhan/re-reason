#!/usr/bin/env python3
"""Supplementary per-transition, per-operation analysis for the generator-repair experiment.

Loads the trained checkpoints saved by run_generator_repair_feasibility.py and computes
transition-level committed/candidate written-register decoding accuracy by operation type
(SET / SWAP / ADD no-wrap / ADD wrap), which the main runner does not record.

Outputs experiments/generator_repair_supplement.json.
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

NUM_REGISTERS = gr.NUM_REGISTERS
NUM_VALUES = gr.NUM_VALUES
STATE_DIM = gr.STATE_DIM


@torch.no_grad()
def per_op_analysis(model, seed: int, device: torch.device) -> Dict[str, Any]:
    splits = gr.canonical_splits(seed)
    batch = splits.test_set.get_batch(len(splits.test_set), idx=0)
    inputs = batch.inputs.to(device)
    step_tgt = batch.metadata["step_targets"].to(device)
    final_tgt = batch.targets.to(device)
    mask, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)
    B, T, _ = inputs.shape

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, dim=1)
    cands = torch.stack(diag.candidates, dim=1)
    cp = model.decision_head(states.reshape(-1, STATE_DIM)).reshape(B, T, NUM_REGISTERS, NUM_VALUES).argmax(-1)
    kp = model.decision_head(cands.reshape(-1, STATE_DIM)).reshape(B, T, NUM_REGISTERS, NUM_VALUES).argmax(-1)
    term_pred = diag.terminal_logits.argmax(-1)

    ok_c = (cp == step_tgt)
    ok_k = (kp == step_tgt)
    committed_all_written = (ok_c | ~mask).all(-1)
    cand_all_written = (ok_k | ~mask).all(-1)

    prev = torch.zeros_like(step_tgt)
    prev[:, 1:] = step_tgt[:, :-1]
    wrap = torch.zeros(B, T, dtype=torch.bool, device=device)
    is_add = op == RegisterTrackingTask.OP_ADD
    if is_add.any():
        wrap[is_add] = (
            prev[is_add].gather(1, reg_a[is_add][:, None]).squeeze(1)
            + prev[is_add].gather(1, reg_b[is_add][:, None]).squeeze(1)
        ) >= NUM_VALUES

    cats = {
        "SET": op == RegisterTrackingTask.OP_SET,
        "SWAP": op == RegisterTrackingTask.OP_SWAP,
        "ADD_no_wrap": is_add & ~wrap,
        "ADD_wrap": is_add & wrap,
    }
    out: Dict[str, Any] = {"terminal_em": float((term_pred == final_tgt).all(-1).float().mean().item()),
                           "per_op": {}}
    for name, sel in cats.items():
        n = int(sel.sum().item())
        if n == 0:
            continue
        out["per_op"][name] = {
            "n_transitions": n,
            "committed_written_all_acc": float(committed_all_written[sel].float().mean().item()),
            "candidate_written_all_acc": float(cand_all_written[sel].float().mean().item()),
            "committed_written_reg_acc": float((ok_c & mask)[sel].float().sum().item() / max(int(mask[sel].sum().item()), 1)),
            "candidate_written_reg_acc": float((ok_k & mask)[sel].float().sum().item() / max(int(mask[sel].sum().item()), 1)),
        }
    out["committed_written_all_overall"] = float(committed_all_written.float().mean().item())
    out["candidate_written_all_overall"] = float(cand_all_written.float().mean().item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partials-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="experiments/generator_repair_supplement.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    records = []
    for path in sorted(glob.glob(os.path.join(args.partials_dir, "models", "*.pt"))):
        base = os.path.basename(path)[:-3]
        variant, seed_s = base.split("__seed")
        seed = int(seed_s)
        model = gr.make_model().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        res = per_op_analysis(model, seed, device)
        res.update({"variant": variant, "seed": seed})
        records.append(res)
        print(f"{variant} seed={seed}: " +
              " ".join(f"{k}={v['committed_written_all_acc']:.3f}" for k, v in res["per_op"].items()))

    variants = sorted({r["variant"] for r in records}, key=lambda v: list(gr.VARIANTS).index(v))
    summary: Dict[str, Any] = {}
    for v in variants:
        rs = [r for r in records if r["variant"] == v]
        summary[v] = {"n_seeds": len(rs)}
        for op_name in ("SET", "SWAP", "ADD_no_wrap", "ADD_wrap"):
            vals = [r["per_op"][op_name]["committed_written_all_acc"] for r in rs if op_name in r["per_op"]]
            cvals = [r["per_op"][op_name]["candidate_written_all_acc"] for r in rs if op_name in r["per_op"]]
            summary[v][op_name] = {
                "committed_mean": float(np.mean(vals)), "committed_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "candidate_mean": float(np.mean(cvals)), "candidate_sd": float(np.std(cvals, ddof=1)) if len(cvals) > 1 else 0.0,
            }
        summary[v]["overall_committed_written_mean"] = float(np.mean([r["committed_written_all_overall"] for r in rs]))
        summary[v]["overall_candidate_written_mean"] = float(np.mean([r["candidate_written_all_overall"] for r in rs]))

    payload = {"summary": summary, "runs": records}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten {args.output}")
    print("\nSUPPLEMENT — transition-level written-register accuracy by operation (mean over seeds)")
    print("| Variant | SET | SWAP | ADD no-wrap | ADD wrap | Mean |")
    print("|---|---:|---:|---:|---:|---:|")
    for v in variants:
        s = summary[v]
        row = [s[k]["committed_mean"] for k in ("SET", "SWAP", "ADD_no_wrap", "ADD_wrap")]
        print(f"| {gr.VARIANTS[v]['label']} | " + " | ".join(f"{x:.3f}" for x in row) +
              f" | {s['overall_committed_written_mean']:.3f} |")


if __name__ == "__main__":
    main()
