"""Preregistered Experiment Runner: EXP-CYCLIC-ADD-01.

Primary objective: Determine whether the existing trainable Cyclic-MGU architecture
can correctly integrate its fixed cyclic ALU into the register-machine execution pipeline
and generalize modular addition to the registered validation pairs.

Model: CyclicMGUPhase16Model (219,018 trainable parameters).
Seeds: 42, 43, 44, 45, 46.
Task: Phase 1.6 Stage 1 3-step modular addition trace on 64 training pairs and 16 validation pairs.
Evaluation Firewall: Protected TEST split (20 pairs with b in {0, 5}) is strictly firewalled.
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
    Phase16RegisterTask,
    Stage1PairPartition,
)
from re_reason.core.register_machine_models import (
    CyclicMGUPhase16Model,
    get_harmonic_basis,
)


def run_cyclic_add_experiment(
    seeds: List[int] = [42, 43, 44, 45, 46],
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 64,
    total_updates: int = 3000,
    val_cadence: int = 50,
    device: torch.device = torch.device("cpu"),
    results_path: str = "experiments/exp_cyclic_add_01_results.json",
) -> Dict[str, Any]:
    """Executes EXP-CYCLIC-ADD-01 across all 5 registered seeds."""
    partition = Stage1PairPartition()

    # Strict firewall check: only build train and val splits
    train_ds = partition.build_trace_split(partition.train_pairs)
    val_ds = partition.build_trace_split(partition.val_pairs)

    assert len(train_ds) == 64, f"Expected 64 training pairs, got {len(train_ds)}"
    assert len(val_ds) == 16, f"Expected 16 validation pairs, got {len(val_ds)}"

    inputs_train = train_ds.inputs.to(device)  # (64, 3, 22)
    targets_train = train_ds.targets[:, 0].to(device)  # (64,) destination r0 modular sum
    inputs_val = val_ds.inputs.to(device)  # (16, 3, 22)
    targets_val = val_ds.targets[:, 0].to(device)  # (16,) destination r0 modular sum

    expected_params = 219018
    model_name = "CyclicMGUPhase16Model"

    print(f"\n{'='*75}")
    print(f"EXP-CYCLIC-ADD-01: {model_name} (5 seeds: {seeds})")
    print(f"Updates: {total_updates} | LR: {learning_rate} | WD: {weight_decay} | Batch: {batch_size}")
    print(f"{'='*75}")

    model_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = CyclicMGUPhase16Model().to(device)
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
        best_val_acc = 0.0
        best_train_loss = float("inf")
        best_train_acc = 0.0
        best_step = -1

        t_start = time.perf_counter()
        t_last_100 = t_start
        time_per_100_list: List[float] = []

        for step in range(1, total_updates + 1):
            model.train()
            optimizer.zero_grad()

            out = model(inputs_train)
            loss = F.cross_entropy(out.logits, targets_train)

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
                    train_out = model(inputs_train)
                    train_loss = F.cross_entropy(train_out.logits, targets_train).item()
                    train_preds = train_out.logits.argmax(dim=-1)
                    train_acc = (train_preds == targets_train).float().mean().item()

                    val_out = model(inputs_val)
                    val_loss = F.cross_entropy(val_out.logits, targets_val).item()
                    val_preds = val_out.logits.argmax(dim=-1)
                    val_acc = (val_preds == targets_val).float().mean().item()

                checkpoints.append({
                    "step": step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_train_loss = train_loss
                    best_train_acc = train_acc
                    best_step = step

        t_total = time.perf_counter() - t_start
        mean_time_100 = float(np.mean(time_per_100_list)) if time_per_100_list else 0.0

        print(
            f"  [{model_name} | Seed {seed}] "
            f"Selected Update: {best_step:4d} | "
            f"Train Loss: {best_train_loss:.4f} | "
            f"Train Acc: {best_train_acc:.4f} | "
            f"Min Val Loss: {best_val_loss:.4f} | "
            f"Val Acc: {best_val_acc:.4f} | "
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
                "train_loss": best_train_loss,
                "train_acc": best_train_acc,
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
            },
            "runtime": {
                "total_seconds": t_total,
                "mean_time_per_100_updates_sec": mean_time_100,
            },
            "checkpoints": checkpoints,
            "numerical_stability": "finite_throughout",
        }
        model_runs.append(run_record)

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_accs = [r["selected_checkpoint"]["val_acc"] for r in model_runs]
    seeds_ge_95 = sum(1 for acc in val_accs if acc >= 0.950)
    seeds_lt_25 = sum(1 for acc in val_accs if acc < 0.250)

    # Progression: >= 0.950 across >= 4/5 seeds, zero seeds < 0.250
    # Rejection: < 0.250 across >= 4/5 seeds
    if seeds_ge_95 >= 4 and seeds_lt_25 == 0:
        verdict = "PASS_PROGRESSION"
    elif seeds_lt_25 >= 4:
        verdict = "FAIL_REJECTION"
    else:
        verdict = "INCONCLUSIVE"

    results: Dict[str, Any] = {
        "experiment_id": "EXP-CYCLIC-ADD-01",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objective": (
            "Determine whether the existing trainable Cyclic-MGU architecture can correctly "
            "integrate its fixed cyclic ALU into the register-machine execution pipeline "
            "and generalize modular addition to the registered validation pairs."
        ),
        "hypothesis": (
            "The fixed cyclic arithmetic inductive bias may allow Cyclic-MGU to generalize "
            "modular addition more reliably than the tested Slot-MGU and Vanilla GRU configurations, "
            "provided that surrounding trainable components integrate the arithmetic result into execution."
        ),
        "scientific_distinction": (
            "The cyclic ALU is fixed and analytical. A successful result demonstrates integration "
            "of a deterministic arithmetic component into a trainable architecture, not learned arithmetic from data."
        ),
        "model_name": model_name,
        "parameter_count": expected_params,
        "protocol": {
            "seeds": seeds,
            "total_updates": total_updates,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "val_cadence": val_cadence,
            "checkpoint_selection": "minimum validation loss (val_loss)",
            "train_pairs_count": 64,
            "val_pairs_count": 16,
            "test_pairs_firewall": "protected (0 queries, never evaluated)",
        },
        "runs": model_runs,
        "gate_assessment": {
            "metric": "validation_accuracy_at_min_val_loss_checkpoint",
            "progression_criterion": ">= 0.950 across >= 4/5 seeds, zero seeds < 0.250",
            "rejection_boundary": "< 0.250 across >= 4/5 seeds",
            "per_seed_val_acc": {r["seed"]: r["selected_checkpoint"]["val_acc"] for r in model_runs},
            "per_seed_train_acc": {r["seed"]: r["selected_checkpoint"]["train_acc"] for r in model_runs},
            "seeds_ge_0_950": seeds_ge_95,
            "seeds_lt_0_250": seeds_lt_25,
            "verdict": verdict,
        },
    }

    if results_path:
        os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results artifact to: {results_path}")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_cyclic_add_experiment(device=device)
