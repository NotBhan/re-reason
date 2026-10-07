#!/usr/bin/env python3
"""Combined-objective analysis for the Phase-1 confirmation experiment.

Computes, for A (terminal), D (late-step), E (candidate/local) and F (combined):
  1. The 4-way first-real-error classification requested by the task:
       A_candidate_generation  : candidate wrong on a written SWAP register (routing/copy failure)
       C_arithmetic            : candidate wrong on a written ADD register (modular arithmetic failure)
       B_commit_readout        : candidate correct, committed state/readout wrong
       D_other                 : anything else (expected ~0)
  2. The T=1 diagnostic: written-register vs untouched-register (ghost) readout errors.
  3. Static loss components (terminal / stepwise / candidate CE) of the trained
     checkpoints on the training split, as objective-scale context.

Outputs experiments/combined_objective_analysis.json.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402

VARIANTS = ["A_terminal", "D_late_t6", "E_candidate", "F_combined"]
DEV = torch.device("cpu")


@torch.no_grad()
def failure_chain(model, seed: int) -> Dict[str, Any]:
    splits = gr.canonical_splits(seed)
    batch = splits.test_set.get_batch(len(splits.test_set), idx=0)
    inputs = batch.inputs
    step_tgt = batch.metadata["step_targets"]
    mask, op, reg_a, reg_b = gr.write_mask_from_inputs(inputs)
    B, T, _ = inputs.shape

    diag = model.forward_with_states(inputs, return_components=True)
    states = torch.stack(diag.states, dim=1)
    cands = torch.stack(diag.candidates, dim=1)
    cp = model.decision_head(states.reshape(-1, gr.STATE_DIM)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)
    kp = model.decision_head(cands.reshape(-1, gr.STATE_DIM)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES).argmax(-1)

    ok_c = (cp == step_tgt)
    ok_k = (kp == step_tgt)
    real_err = ~((ok_c | ~mask).all(-1))  # (B, T) committed written-register error

    counts = {"A_candidate_generation": 0, "C_arithmetic": 0, "B_commit_readout": 0, "D_other": 0}
    per_op_origin: Dict[str, int] = {}
    for b in range(B):
        idx = np.nonzero(real_err[b].numpy())[0]
        if len(idx) == 0:
            continue
        t0 = int(idx[0])
        wrong = [r for r in range(gr.NUM_REGISTERS)
                 if bool(mask[b, t0, r]) and int(cp[b, t0, r]) != int(step_tgt[b, t0, r])]
        cand_wrong = any(int(kp[b, t0, r]) != int(step_tgt[b, t0, r]) for r in wrong)
        o = int(op[b, t0])
        if not cand_wrong:
            counts["B_commit_readout"] += 1
        elif o == RegisterTrackingTask.OP_ADD:
            counts["C_arithmetic"] += 1
        elif o == RegisterTrackingTask.OP_SWAP:
            counts["A_candidate_generation"] += 1
        else:
            counts["D_other"] += 1
        key = f"op{int(o)}_cand{int(cand_wrong)}_n{len(wrong)}"
        per_op_origin[key] = per_op_origin.get(key, 0) + 1

    n_err = sum(counts.values())
    return {"n_sequences": int(B), "n_first_real_errors": n_err,
            "counts": counts,
            "percent_of_errors": {k: (100.0 * v / n_err if n_err else float("nan")) for k, v in counts.items()},
            "percent_of_sequences": {k: (100.0 * v / B) for k, v in counts.items()},
            "per_op_origin": per_op_origin}


@torch.no_grad()
def t1_diagnostic(model, seed: int) -> Dict[str, Any]:
    ds = gr.horizon_test_set(seed, 1)
    x = ds.inputs[:, 0]
    pred = model(ds.inputs).logits.argmax(-1)
    true = ds.targets
    wrong = pred != true
    written = x[:, 3:7].argmax(-1)
    rows = torch.arange(len(x))
    written_wrong = wrong[rows, written]
    ghost_eps = wrong.any(-1) & ~written_wrong
    return {"em": float(1.0 - wrong.any(-1).float().mean().item()),
            "failing_episodes": int(wrong.any(-1).sum()),
            "n_sequences": int(len(x)),
            "written_register_wrong": int(written_wrong.sum()),
            "ghost_only_failures": int(ghost_eps.sum()),
            "errors_by_register": {f"R{r}": int(wrong[:, r].sum()) for r in range(gr.NUM_REGISTERS)}}


@torch.no_grad()
def static_loss_components(model, seed: int) -> Dict[str, float]:
    """Mean terminal / stepwise(committed) / candidate CE of the trained checkpoint on its training split."""
    splits = gr.canonical_splits(seed)
    train_set = splits.train_set
    totals = {"terminal": 0.0, "stepwise": 0.0, "candidate": 0.0}
    nb = 0
    for b_idx in range(0, len(train_set), 256):
        batch = train_set.get_batch(256, idx=b_idx // 256)
        inputs, targets = batch.inputs, batch.targets
        step_tgt = batch.metadata["step_targets"]
        diag = model.forward_with_states(inputs, return_components=True)
        B, T, _ = inputs.shape
        totals["terminal"] += float(F.cross_entropy(diag.terminal_logits.reshape(-1, gr.NUM_VALUES),
                                                    targets.reshape(-1)).item())
        states = torch.stack(diag.states, dim=1)
        sl = model.decision_head(states.reshape(-1, gr.STATE_DIM)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES)
        totals["stepwise"] += float(np.mean([
            float(F.cross_entropy(sl[:, t].reshape(-1, gr.NUM_VALUES), step_tgt[:, t].reshape(-1)).item())
            for t in range(T)]))
        cands = torch.stack(diag.candidates, dim=1)
        cl = model.decision_head(cands.reshape(-1, gr.STATE_DIM)).reshape(B, T, gr.NUM_REGISTERS, gr.NUM_VALUES)
        mask, _, _, _ = gr.write_mask_from_inputs(inputs)
        ce = F.cross_entropy(cl.reshape(-1, gr.NUM_VALUES), step_tgt.reshape(-1), reduction="none").reshape(B, T, gr.NUM_REGISTERS)
        totals["candidate"] += float(((ce * mask).sum() / mask.sum().clamp_min(1)).item())
        nb += 1
    return {k: v / nb for k, v in totals.items()}


def load_model(partials_dir: str, variant: str, seed: int):
    model = gr.make_model().to(DEV)
    model.load_state_dict(torch.load(os.path.join(partials_dir, "models", f"{variant}__seed{seed}.pt"),
                                     map_location="cpu"))
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partials-dir", required=True)
    ap.add_argument("--output", default="experiments/combined_objective_analysis.json")
    args = ap.parse_args()

    out: Dict[str, Any] = {"failure_chain": {}, "t1": {}, "static_loss_components": {}}
    for v in VARIANTS:
        fc_runs, t1_runs, loss_runs = [], [], []
        for seed in gr.SEEDS:
            model = load_model(args.partials_dir, v, seed)
            fc_runs.append(failure_chain(model, seed))
            t1_runs.append(t1_diagnostic(model, seed))
            loss_runs.append(static_loss_components(model, seed))
        def mean_of(key):
            return float(np.mean([r[key] for r in fc_runs]))
        fc_summary = {
            "n_first_real_errors_mean": mean_of("n_first_real_errors"),
            "counts_mean": {k: float(np.mean([r["counts"][k] for r in fc_runs])) for k in fc_runs[0]["counts"]},
            "percent_of_errors_mean": {k: float(np.mean([r["percent_of_errors"][k] for r in fc_runs])) for k in fc_runs[0]["percent_of_errors"]},
            "per_op_origin_mean": {},
        }
        all_keys = set()
        for r in fc_runs:
            all_keys.update(r["per_op_origin"].keys())
        for k in sorted(all_keys):
            fc_summary["per_op_origin_mean"][k] = float(np.mean([r["per_op_origin"].get(k, 0) for r in fc_runs]))
        out["failure_chain"][v] = {"runs": fc_runs, "summary": fc_summary}
        out["t1"][v] = {
            "runs": t1_runs,
            "summary": {"em_mean": float(np.mean([r["em"] for r in t1_runs])),
                        "written_wrong_total": int(sum(r["written_register_wrong"] for r in t1_runs)),
                        "ghost_failures_total": int(sum(r["ghost_only_failures"] for r in t1_runs))},
        }
        out["static_loss_components"][v] = {
            k: {"mean": float(np.mean([r[k] for r in loss_runs])),
                "sd": float(np.std([r[k] for r in loss_runs], ddof=1))}
            for k in ("terminal", "stepwise", "candidate")}
        print(f"{v}: first-real-errors/week={fc_summary['n_first_real_errors_mean']:.0f} "
              f"A={fc_summary['counts_mean']['A_candidate_generation']:.0f} "
              f"B={fc_summary['counts_mean']['B_commit_readout']:.0f} "
              f"C={fc_summary['counts_mean']['C_arithmetic']:.0f} "
              f"D={fc_summary['counts_mean']['D_other']:.0f} | "
              f"T1 EM={out['t1'][v]['summary']['em_mean']:.3f} "
              f"(written wrong {out['t1'][v]['summary']['written_wrong_total']}, "
              f"ghost {out['t1'][v]['summary']['ghost_failures_total']}) | "
              f"losses {out['static_loss_components'][v]['terminal']['mean']:.3f}/"
              f"{out['static_loss_components'][v]['stepwise']['mean']:.3f}/"
              f"{out['static_loss_components'][v]['candidate']['mean']:.3f}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten {args.output}")


if __name__ == "__main__":
    main()
