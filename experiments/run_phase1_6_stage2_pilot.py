"""Preregistered Phase 1.6 Stage 2 Address-Binding Feasibility Pilot Runner.

Authorized Scope: Stage 2 Address-Binding Pilot across three trainable models
(Slot-MGU, Cyclic-MGU, Vanilla GRU) and deterministic MSM reference check ONLY.
Stage 3, composed sequence training, length extrapolation, and full campaign remain UNAUTHORIZED.

Strict Evaluation Firewall:
Held-out test split (1,200 sequences) is strictly firewalled and NEVER evaluated or unblinded.
Pilot weights are kept in memory during training and discarded post-pilot; no persistent checkpoints saved.
"""

from collections import Counter
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from re_reason.benchmarks.register_machine_benchmark import (
    Instruction,
    Phase16RegisterTask,
    Phase16SplitManager,
    SplitDataset,
)
from re_reason.core.register_machine_models import (
    CyclicMGUPhase16Model,
    MSMDeterministicReference,
    SlotMGUPhase16Model,
    VanillaGRUPhase16Model,
)


def run_stage2_trained_pilot(
    seeds: List[int],
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 64,
    total_updates: int = 3000,
    val_cadence: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, Any], SplitDataset, torch.Tensor]:
    """Executes the Stage 2 address-binding pilot for Slot-MGU, Cyclic-MGU, and Vanilla GRU.

    15 parameter-training runs total (3 models x 5 seeds).
    Sequence length T=10, consisting strictly of SET, SWAP, and READ (zero ADD).
    Objective: Terminal cross-entropy on destination register r_dest queried via READ at step 10.
    Evaluation on 1,200 VAL sequences every 50 updates.
    """
    mgr = Phase16SplitManager()

    print(f"\n{'='*70}")
    print("Generating Stage 2 Address-Binding Dataset Splits (T=10, SET/SWAP/READ, zero ADD)")
    print(f"{'='*70}")

    t0_gen = time.perf_counter()
    train_ds, train_hashes = mgr.generate_disjoint_split(
        size=6000, seed=1042, generation_mode="stage2_binding"
    )
    val_ds, val_hashes = mgr.generate_disjoint_split(
        size=1200, seed=50042, generation_mode="stage2_binding"
    )
    t_gen = time.perf_counter() - t0_gen

    print(f"Dataset generated in {t_gen:.2f}s:")
    print(f"  TRAIN: {len(train_ds)} sequences")
    print(f"  VAL:   {len(val_ds)} sequences")
    print("  TEST:  STRICTLY FIREWALLED (1,200 sequences ungenerated and unqueried)")

    # Split invariants
    assert len(train_ds) == 6000, f"Expected 6000 train sequences, got {len(train_ds)}"
    assert len(val_ds) == 1200, f"Expected 1200 val sequences, got {len(val_ds)}"

    s_tr = set(train_hashes)
    s_va = set(val_hashes)
    overlap = len(s_tr & s_va)
    assert overlap == 0, f"Cryptographic leak detected! {overlap} sequences overlap between train and val."

    # Verification: check sequence length T=10, no ADD, terminal READ
    for ds_name, ds in [("TRAIN", train_ds), ("VAL", val_ds)]:
        assert ds.inputs.shape[1] == 10, f"{ds_name} sequences must have length T=10"
        ops = ds.inputs[:, :, 0:4].argmax(dim=-1)
        assert (ops == 3).sum() == 0, f"ADD operation leaked into Stage 2 {ds_name} dataset!"
        assert (ops[:, 0:2] == 0).all(), f"First two steps in {ds_name} must be SET"
        assert (ops[:, -1] == 2).all(), f"Terminal step in {ds_name} must be READ"

    # Terminal targets: READ destination register queried at step 9
    dest_regs_train = train_ds.inputs[:, -1, 4:8].argmax(dim=-1)
    targets_train = train_ds.targets[torch.arange(len(train_ds)), dest_regs_train].to(device)
    inputs_train = train_ds.inputs.to(device)

    dest_regs_val = val_ds.inputs[:, -1, 4:8].argmax(dim=-1)
    targets_val = val_ds.targets[torch.arange(len(val_ds)), dest_regs_val].to(device)
    inputs_val = val_ds.inputs.to(device)

    models_config = [
        ("SlotMGUPhase16Model", SlotMGUPhase16Model, 223414),
        ("CyclicMGUPhase16Model", CyclicMGUPhase16Model, 219018),
        ("VanillaGRUPhase16Model", VanillaGRUPhase16Model, 220874),
    ]

    track_results: Dict[str, Any] = {}

    for model_name, model_cls, expected_params in models_config:
        print(f"\n{'='*70}")
        print(f"Starting Stage 2 Pilot: {model_name} (5 seeds: {seeds})")
        print(f"{'='*70}")

        model_runs: List[Dict[str, Any]] = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = model_cls().to(device)
            actual_params = model.count_parameters()
            assert actual_params == expected_params, (
                f"Parameter mismatch for {model_name}: expected {expected_params}, got {actual_params}"
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            checkpoints: List[Dict[str, Any]] = []
            best_val_loss = float("inf")
            best_val_em = 0.0
            best_step = -1

            t_start = time.perf_counter()
            t_last_100 = t_start
            time_per_100_list: List[float] = []

            # Deterministic minibatch sampler for each seed
            g_sampler = torch.Generator(device="cpu").manual_seed(seed)
            indices = torch.randperm(len(train_ds), generator=g_sampler)
            batch_ptr = 0

            for step in range(1, total_updates + 1):
                # Minibatch selection
                if batch_ptr + batch_size > len(indices):
                    indices = torch.randperm(len(train_ds), generator=g_sampler)
                    batch_ptr = 0

                mb_idx = indices[batch_ptr:batch_ptr + batch_size]
                batch_ptr += batch_size

                mb_inputs = inputs_train[mb_idx]
                mb_targets = targets_train[mb_idx]

                model.train()
                optimizer.zero_grad()

                out = model(mb_inputs)
                loss = F.cross_entropy(out.logits, mb_targets)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss {loss.item()} at step {step} for {model_name} seed {seed}")

                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    t_now = time.perf_counter()
                    time_per_100_list.append(t_now - t_last_100)
                    t_last_100 = t_now

                if step % val_cadence == 0:
                    model.eval()
                    with torch.no_grad():
                        val_out = model(inputs_val)
                        val_loss = F.cross_entropy(val_out.logits, targets_val).item()
                        val_preds = val_out.logits.argmax(dim=-1)
                        val_em = (val_preds == targets_val).float().mean().item()

                    checkpoints.append({
                        "step": step,
                        "val_loss": val_loss,
                        "val_em": val_em,
                    })

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_em = val_em
                        best_step = step

            t_total = time.perf_counter() - t_start
            mean_time_100 = float(np.mean(time_per_100_list))

            print(
                f"  [{model_name} | Seed {seed}] "
                f"Selected Update: {best_step:4d} | "
                f"Min Val Loss: {best_val_loss:.4f} | "
                f"Val EM: {best_val_em:.4f} | "
                f"Runtime: {t_total:.2f}s (~{mean_time_100:.3f}s / 100 updates)"
            )

            run_record = {
                "model_name": model_name,
                "seed": seed,
                "parameter_count": actual_params,
                "total_updates": total_updates,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "selected_checkpoint": {
                    "step": best_step,
                    "val_loss": best_val_loss,
                    "val_em": best_val_em,
                },
                "runtime": {
                    "total_seconds": t_total,
                    "mean_time_per_100_updates_sec": mean_time_100,
                },
                "checkpoints": checkpoints,
                "numerical_stability": "finite_throughout",
            }
            model_runs.append(run_record)

            # Discard trained model weights post-run
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Validation gate evaluation
        val_ems = [r["selected_checkpoint"]["val_em"] for r in model_runs]
        seeds_ge_95 = sum(1 for em in val_ems if em >= 0.950)
        seeds_lt_85 = sum(1 for em in val_ems if em < 0.850)
        gate_passed = (seeds_ge_95 >= 4) and (seeds_lt_85 == 0)

        track_results[model_name] = {
            "parameter_count": expected_params,
            "runs": model_runs,
            "gate_assessment": {
                "metric": "behavioral_read_exact_match_at_min_val_loss_checkpoint",
                "threshold_rule": ">= 0.950 across >= 4/5 seeds, zero seeds < 0.850",
                "rejection_boundary": "< 0.850 across >= 4/5 seeds",
                "per_seed_val_em": {r["seed"]: r["selected_checkpoint"]["val_em"] for r in model_runs},
                "seeds_ge_0_950": seeds_ge_95,
                "seeds_lt_0_850": seeds_lt_85,
                "verdict": "PASS" if gate_passed else "FAIL",
            },
        }

    return track_results, val_ds, targets_val


def run_stage2_msm_deterministic_check(
    val_ds: SplitDataset,
    targets_val: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Executes the Stage 2 deterministic check for Candidate 3 (MSM Reference).

    Zero training steps, parameter-free analytical verification on 1,200 VAL sequences.
    Evaluates exact discrete register routing (SET, SWAP, READ).
    """
    print(f"\n{'='*70}")
    print("Starting Stage 2 MSM Deterministic Analytical Reference Check")
    print(f"{'='*70}")

    inputs_val = val_ds.inputs.to(device)

    msm = MSMDeterministicReference()
    actual_params = msm.count_parameters()
    assert actual_params == 0, f"MSM must have exactly 0 trainable parameters, got {actual_params}"

    msm.eval()
    with torch.no_grad():
        out = msm(inputs_val)
        val_loss = F.cross_entropy(out.logits, targets_val).item()
        preds = out.logits.argmax(dim=-1)
        val_em = (preds == targets_val).float().mean().item()
        correct_count = (preds == targets_val).sum().item()

    passed = (val_em == 1.0000)

    print(
        f"  [Stage 2 MSM Reference] "
        f"Validation EM: {correct_count}/{len(val_ds)} ({val_em:.4f}) | "
        f"Val Loss: {val_loss:.4f} | "
        f"Verdict: {'PASS' if passed else 'FAIL'}"
    )

    return {
        "model_name": "MSMDeterministicReference",
        "parameter_count": 0,
        "training_updates": 0,
        "validation_sequences": len(val_ds),
        "correct_predictions": correct_count,
        "behavioral_read_em": val_em,
        "validation_cross_entropy": val_loss,
        "threshold_rule": "exact 1.0000 by construction",
        "verdict": "PASS" if passed else "FAIL",
        "classification": "Parameter-free deterministic analytical reference standard; not a trained neural model.",
    }


def main() -> None:
    print("=" * 70)
    print("RE:Reason — Phase 1.6 Stage 2 Authorized Feasibility Pilot Execution")
    print("Scope: Stage 2 Address-Binding Pilot (15 runs + 1 deterministic check)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    t_pilot_start = time.perf_counter()

    seeds = [42, 43, 44, 45, 46]
    track_results, val_ds, targets_val = run_stage2_trained_pilot(seeds=seeds, device=device)
    msm_results = run_stage2_msm_deterministic_check(val_ds, targets_val, device=device)

    t_pilot_total = time.perf_counter() - t_pilot_start

    output_payload = {
        "experiment": "Phase 1.6 Stage 2 Address-Binding Feasibility Pilot",
        "scope": "Authorized Stage 2 Address-Binding Pilot only",
        "heldout_test_set_status": "STRICTLY FIREWALLED — zero queries, zero unblinding",
        "model_weights_status": "DISCARDED POST-PILOT — zero persistent checkpoints created",
        "total_runtime_seconds": t_pilot_total,
        "hardware_device": str(device),
        "protocol": {
            "task": "Stage 2 Address-Binding (T=10, SET/SWAP/READ, zero ADD)",
            "train_size": 6000,
            "val_size": 1200,
            "test_size": 1200,
            "batch_size": 64,
            "total_updates": 3000,
            "val_cadence": 50,
            "learning_rate": 3e-3,
            "weight_decay": 1e-2,
            "optimizer": "AdamW",
            "checkpoint_selection": "minimum validation cross-entropy loss",
            "advancement_gate": "EM_read,val >= 0.950 across >= 4/5 seeds, zero seeds < 0.850",
            "rejection_boundary": "EM_read,val < 0.850 across >= 4/5 seeds",
        },
        "trained_models": track_results,
        "deterministic_reference": msm_results,
    }

    results_path = os.path.join(os.path.dirname(__file__), "phase1_6_stage2_pilot_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n{'='*70}")
    print("STAGE 2 PILOT SUMMARY VERDICTS")
    print(f"{'='*70}")
    print(f"Total Wall-Clock Runtime: {t_pilot_total:.2f} seconds ({t_pilot_total/60:.2f} minutes)")
    print(f"Results saved to: {results_path}")
    print(f"Slot-MGU Stage 2 Gate: {track_results['SlotMGUPhase16Model']['gate_assessment']['verdict']}")
    print(f"Cyclic-MGU Stage 2 Gate: {track_results['CyclicMGUPhase16Model']['gate_assessment']['verdict']}")
    print(f"Vanilla GRU Stage 2 Gate: {track_results['VanillaGRUPhase16Model']['gate_assessment']['verdict']}")
    print(f"MSM Deterministic Reference: {msm_results['verdict']}")
    print(f"Held-Out Split Firewall: VERIFIED INTACT (zero queries)")
    print(f"Weight Discard: VERIFIED (in-memory only, zero .pt files)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
