"""Preregistered Experiment Runner: EXP-CYCLIC-PERSIST-ABL-01 (Persistence-Only Ablation).

Scientific objective and intervention
-------------------------------------
Evaluates whether auxiliary supervision of *unaddressed-register persistence alone*
across the step-1 transition (without intermediate destination write supervision)
is sufficient to preserve register r0 and support arithmetic generalization in
``CyclicMGUPhase16Model`` (219,018 trainable parameters, unchanged).

In the primary experiment (EXP-CYCLIC-PERSIST-01), joint supervision was applied:
    L_total = L_terminal + 1.0 * L_dest + 0.5 * L_persist

In this ablation, intermediate destination supervision is completely removed from
the training objective:
    L_total = L_terminal + 0.5 * L_persist
    L_terminal = CE(logits_terminal, (a + b) % 10)
    L_persist  = CE(logits_persist_1, a)

Destination losses at step 0 (r0 <- a) and step 1 (r1 <- b) are computed strictly
for diagnostic telemetry and composite reference evaluation; they do NOT contribute
to L_total and receive zero gradient backpropagation.

Stage gating
------------
* ``--smoke`` (Stage 1, authorized): seed 42, 100 updates, smoke-labelled artifact
  serializing to ``experiments/exp_cyclic_persist_abl_01_SMOKE_ONLY_seed42.json``.
* ``--stage2-authorized`` (Stage 2, NOT authorized by the Stage 1 task): official
  five-seed experiment serializing ``experiments/exp_cyclic_persist_abl_01_results.json``.

Shared gate machinery
---------------------
The checkpoint-selection policy (minimum native validation loss, ties broken by
highest validation accuracy, then earliest update) and the 3x4 decision matrix are
imported from ``experiments/run_exp_cyclic_stepwise_01.py``, ensuring identical
gate mechanics across the experimental suite.

Evaluation firewall
-------------------
The protected TEST split (20 operand pairs with b in {0, 5}) is never loaded,
constructed, evaluated, queried, or logged. Only the registered TRAIN-FIT (64
pairs) and VAL-SEEN (16 pairs) splits are materialized.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from re_reason.benchmarks.register_machine_benchmark import Stage1PairPartition
from re_reason.core.register_machine_models import CyclicMGUPhase16Model, project_to_unit_circle
from experiments.run_exp_cyclic_stepwise_01 import (
    COSINE_EPS,
    MIN_VECTOR_NORM,
    assert_canonical_trace,
    assert_protected_domain_absent,
    classify_experiment_outcome,
    cosine_similarity,
    extract_destination_targets,
    select_best_checkpoint,
)

EXPERIMENT_ID = "EXP-CYCLIC-PERSIST-ABL-01"
LAMBDA_DEST = 0.0  #: Invariant for this ablation: destination supervision is excluded from training loss
LAMBDA_PERSIST = 0.5  #: Approved Option 1 coefficient, preserving nominal scalar weight invariance
EXPECTED_PARAMETER_COUNT = 219018

#: r0 — written at step 0, unaddressed at step 1, and the step-2 ADD destination.
PERSIST_REGISTER_INDEX = 0
#: The S1 -> S2 transition (instruction SET r1 b) at which r0 must persist.
PERSIST_STEP_INDEX = 1

REGISTERED_TRAIN_PAIRS = 64
REGISTERED_VAL_PAIRS = 16
PROTECTED_TEST_PAIRS = 20
PROTECTED_TEST_B_VALUES = (0, 5)

SMOKE_ARTIFACT_PATH = "experiments/exp_cyclic_persist_abl_01_SMOKE_ONLY_seed42.json"
OFFICIAL_ARTIFACT_PATH = "experiments/exp_cyclic_persist_abl_01_results.json"
SMOKE_ARTIFACT_STATUS = (
    "SMOKE TEST TELEMETRY ONLY — ONE SEED, 100 UPDATES — NOT OFFICIAL RESULTS — "
    "NOT EVIDENCE FOR OR AGAINST THE RESEARCH HYPOTHESIS — NOT FOR GATE ASSESSMENT"
)


# ---------------------------------------------------------------------------
# Persistence query and rollout
# ---------------------------------------------------------------------------

def register_one_hot(batch_size: int, register_index: int, num_registers: int, reference: torch.Tensor) -> torch.Tensor:
    """Builds a (B, num_registers) one-hot address vector matching ``reference``'s device/dtype."""
    one_hot = torch.zeros((batch_size, num_registers), device=reference.device, dtype=reference.dtype)
    one_hot[:, register_index] = 1.0
    return one_hot


def persistence_query(
    model: CyclicMGUPhase16Model,
    state_after_step: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Queries the unaddressed register r0 on the post-transition state S2.

    Uses the existing shared ``mlp_read`` network and the fixed prototype basis; no new
    parameters, heads, or projections are introduced.

    Returns:
        u_persist: projected read vector of shape (B, 10).
        persist_logits: prototype-bank logits of shape (B, 10).
    """
    address = register_one_hot(batch_size, PERSIST_REGISTER_INDEX, model.num_registers, state_after_step)
    u_raw = model.mlp_read(torch.cat([state_after_step, address], dim=-1))
    u_persist = project_to_unit_circle(u_raw, model.num_frequencies, model.eps)
    persist_logits = torch.einsum("bd,vd->bv", u_persist, model.basis)
    return u_persist, persist_logits


def rollout_with_persistence(
    model: CyclicMGUPhase16Model,
    inputs: torch.Tensor,
) -> Dict[str, Any]:
    """Runs the canonical trace rollout while capturing the step-1 persistence query.

    The rollout composes ``model.forward_step`` exactly as ``model.forward`` does, so its
    outputs are bit-identical to ``model(inputs, return_step_diagnostics=True)`` (asserted
    in the focused test suite) while additionally exposing the intermediate states needed
    to query the unaddressed register on S2.
    """
    batch_size, seq_len, _ = inputs.shape
    state = model.initial_state(batch_size, inputs.device)

    step_logits_list: List[torch.Tensor] = []
    step_diagnostics_list: List[Dict[str, torch.Tensor]] = []
    states_after_step: List[torch.Tensor] = []

    for t in range(seq_len):
        step_diagnostics: Dict[str, torch.Tensor] = {}
        state, step_logits = model.forward_step(inputs[:, t, :], state, step_diagnostics)
        step_logits_list.append(step_logits)
        step_diagnostics_list.append(step_diagnostics)
        states_after_step.append(state)

    step_logits = torch.stack(step_logits_list, dim=1)  # (B, T, 10)

    u_persist, persist_logits = persistence_query(
        model, states_after_step[PERSIST_STEP_INDEX], batch_size
    )

    return {
        "logits": step_logits[:, -1, :],
        "step_logits": step_logits,
        "final_state": state,
        "states_after_step": states_after_step,
        "step_diagnostics": step_diagnostics_list,
        "u_persist": u_persist,
        "persist_logits": persist_logits,
    }


# ---------------------------------------------------------------------------
# Objective formulation
# ---------------------------------------------------------------------------

def compute_losses(
    rollout: Dict[str, Any],
    y_destination: torch.Tensor,
    lambda_persist: float = LAMBDA_PERSIST,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ablation objective: terminal CE + weighted persistence term.

    Destination targets follow the verified ``(a, b, (a + b) % 10)`` sequence; the
    persistence target is operand ``a`` (the r0 value that must survive the step-1
    transition). Step 0 and step 1 destination cross-entropy losses are computed
    strictly for diagnostic telemetry and composite reference evaluation; neither
    contributes to total_loss or its backward gradients.

    Returns:
        terminal_loss: CE(logits_terminal, (a + b) % 10)
        destination_loss: 0.5 * [CE(logits_dest_0, a) + CE(logits_dest_1, b)] (diagnostic only)
        persistence_loss: CE(logits_persist_1, a)
        total_loss: terminal_loss + lambda_persist * persistence_loss (native training objective)
        composite_reference_loss: terminal_loss + 1.0 * destination_loss + 0.5 * persistence_loss
    """
    y_operand_a = y_destination[:, 0]
    y_operand_b = y_destination[:, 1]
    y_terminal = y_destination[:, -1]

    terminal_loss = F.cross_entropy(rollout["logits"], y_terminal)
    destination_loss = 0.5 * (
        F.cross_entropy(rollout["step_logits"][:, 0, :], y_operand_a)
        + F.cross_entropy(rollout["step_logits"][:, 1, :], y_operand_b)
    )
    persistence_loss = F.cross_entropy(rollout["persist_logits"], y_operand_a)

    total_loss = terminal_loss + lambda_persist * persistence_loss
    composite_reference_loss = terminal_loss + 1.0 * destination_loss + 0.5 * persistence_loss
    return terminal_loss, destination_loss, persistence_loss, total_loss, composite_reference_loss


# ---------------------------------------------------------------------------
# Representation metrics
# ---------------------------------------------------------------------------

def _pooled_context_consistency(
    vectors_a: torch.Tensor,
    values_a: torch.Tensor,
    vectors_b: torch.Tensor,
    values_b: torch.Tensor,
    num_digits: int,
) -> Tuple[Optional[float], Dict[str, Dict[str, float]], List[int]]:
    """Mean pairwise cosine similarity of same-integer step-2 operand reads across roles/contexts."""
    per_digit: Dict[str, Dict[str, float]] = {}
    omitted: List[int] = []
    digit_means: List[float] = []

    for v in range(num_digits):
        parts: List[torch.Tensor] = []
        selection_a = values_a == v
        if bool(selection_a.any()):
            parts.append(vectors_a[selection_a])
        selection_b = values_b == v
        if bool(selection_b.any()):
            parts.append(vectors_b[selection_b])
        if not parts:
            omitted.append(v)
            continue

        pooled = torch.cat(parts, dim=0)
        count = pooled.shape[0]
        if count < 2:
            omitted.append(v)
            continue

        pairwise = cosine_similarity(pooled.unsqueeze(1), pooled.unsqueeze(0))
        upper = torch.triu_indices(count, count, offset=1)
        mean_similarity = pairwise[upper[0], upper[1]].mean().item()

        per_digit[str(v)] = {
            "n_vectors": count,
            "n_distinct_pairs": count * (count - 1) // 2,
            "mean_pairwise_cosine": mean_similarity,
        }
        digit_means.append(mean_similarity)

    overall = float(np.mean(digit_means)) if digit_means else None
    return overall, per_digit, omitted


def compute_representation_metrics(
    model: CyclicMGUPhase16Model,
    rollout: Dict[str, Any],
    y_destination: torch.Tensor,
) -> Dict[str, Any]:
    """Computes every registered representation and classification diagnostic."""
    basis = model.basis  # (10, 10) canonical 5-harmonic Fourier codewords
    step_diagnostics = rollout["step_diagnostics"]
    seq_len = rollout["step_logits"].shape[1]
    if seq_len != 3:
        raise RuntimeError(f"the registered metrics assume the 3-step canonical trace, got T={seq_len}")

    u_a_step2 = step_diagnostics[2]["u_a"]
    u_b_step2 = step_diagnostics[2]["u_b"]
    c1_alu_step2 = step_diagnostics[2]["c1_alu"]
    u_dest_post = torch.stack([diag["u_dest_post"] for diag in step_diagnostics], dim=1)

    y_operand_a = y_destination[:, 0]
    y_operand_b = y_destination[:, 1]
    y_sum = y_destination[:, -1]

    # 7.1 Register-specific read alignment at step 2
    c_read_r0 = cosine_similarity(u_a_step2, basis[y_operand_a]).mean().item()
    c_read_r1 = cosine_similarity(u_b_step2, basis[y_operand_b]).mean().item()
    c_read = 0.5 * (c_read_r0 + c_read_r1)

    # 7.2 Step-1 persistence alignment of the unaddressed register r0
    c_persist = cosine_similarity(rollout["u_persist"], basis[y_operand_a]).mean().item()

    # 7.3 Cross-context read consistency at step 2
    s_context, s_context_per_digit, omitted_digits = _pooled_context_consistency(
        u_a_step2, y_operand_a, u_b_step2, y_operand_b, model.num_classes
    )

    # 7.4 Raw ALU-output phase alignment at step 2
    c_alu = cosine_similarity(c1_alu_step2, basis[y_sum]).mean().item()

    # 7.5 Post-update destination alignment at steps 0, 1, 2
    c_dest_steps = [
        cosine_similarity(u_dest_post[:, t, :], basis[y_destination[:, t]]).mean().item()
        for t in range(seq_len)
    ]
    c_dest = float(np.mean(c_dest_steps))

    # 7.6 Per-step, persistence and terminal accuracies
    step_0_accuracy = (rollout["step_logits"][:, 0, :].argmax(dim=-1) == y_operand_a).float().mean().item()
    step_1_accuracy = (rollout["step_logits"][:, 1, :].argmax(dim=-1) == y_operand_b).float().mean().item()
    step_2_accuracy = (rollout["logits"].argmax(dim=-1) == y_sum).float().mean().item()
    persistence_accuracy = (rollout["persist_logits"].argmax(dim=-1) == y_operand_a).float().mean().item()

    return {
        "c_read": c_read,
        "c_read_r0": c_read_r0,
        "c_read_r1": c_read_r1,
        "c_persist_r0": c_persist,
        "s_context": s_context,
        "s_context_per_digit": s_context_per_digit,
        "s_context_omitted_digits": omitted_digits,
        "c_alu": c_alu,
        "c_dest": c_dest,
        "c_dest_step0": c_dest_steps[0],
        "c_dest_step1": c_dest_steps[1],
        "c_dest_step2": c_dest_steps[2],
        "step0_acc": step_0_accuracy,
        "step1_acc": step_1_accuracy,
        "step2_acc": step_2_accuracy,
        "persist1_acc": persistence_accuracy,
        "acc": step_2_accuracy,
    }


def evaluate_split(
    model: CyclicMGUPhase16Model,
    inputs: torch.Tensor,
    step_targets: torch.Tensor,
    lambda_persist: float = LAMBDA_PERSIST,
) -> Dict[str, Any]:
    """Evaluates every registered metric on one allowed split under the ablation contract."""
    model.eval()
    with torch.no_grad():
        rollout = rollout_with_persistence(model, inputs)

    y_destination = extract_destination_targets(inputs, step_targets)
    terminal_loss, destination_loss, persistence_loss, total_loss, composite_ref_loss = compute_losses(
        rollout, y_destination, lambda_persist
    )

    metrics: Dict[str, Any] = {
        "loss_total": total_loss.item(),
        "loss_terminal": terminal_loss.item(),
        "loss_dest": destination_loss.item(),
        "loss_persist": persistence_loss.item(),
        "loss_composite_reference": composite_ref_loss.item(),
    }
    metrics.update(compute_representation_metrics(model, rollout, y_destination))

    cosine_keys = (
        "c_read", "c_read_r0", "c_read_r1", "c_persist_r0", "s_context",
        "c_alu", "c_dest", "c_dest_step0", "c_dest_step1", "c_dest_step2",
    )
    for key, value in metrics.items():
        if isinstance(value, float):
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite metric {key} = {value}")
            if "loss" in key and value < 0.0:
                raise RuntimeError(f"negative loss metric {key} = {value}")
            if "acc" in key and (value < 0.0 or value > 1.0):
                raise RuntimeError(f"out-of-bounds accuracy metric {key} = {value}")
            if key in cosine_keys and (value < -1.0 or value > 1.0):
                raise RuntimeError(f"out-of-bounds cosine metric {key} = {value}")
    return metrics


# ---------------------------------------------------------------------------
# Numerical-stability and gradient-path checks
# ---------------------------------------------------------------------------

def require_finite(tensor: torch.Tensor, label: str, step: int, seed: int) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"non-finite {label} at update {step} (seed {seed}): {tensor}")
    if "loss" in label.lower() and bool((tensor < 0.0).any()):
        raise RuntimeError(f"negative {label} at update {step} (seed {seed}): {tensor}")


def check_gradients(model: CyclicMGUPhase16Model, step: int, seed: int) -> Dict[str, float]:
    """Verifies finite, non-zero gradients reaching the intended trainable computation."""
    block_squared_norm = {"mlp_read": 0.0, "mlp_write": 0.0, "mlp_gate": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"parameter {name} received no gradient at update {step} (seed {seed})")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"non-finite gradient for {name} at update {step} (seed {seed})")
        for block in block_squared_norm:
            if name.startswith(block):
                block_squared_norm[block] += float((parameter.grad.detach() ** 2).sum().item())

    for block, squared_norm in block_squared_norm.items():
        if not np.isfinite(squared_norm) or squared_norm <= 0.0:
            raise RuntimeError(
                f"gradient path to {block} is broken or vanishing at update {step} "
                f"(seed {seed}): squared gradient norm = {squared_norm}"
            )
        if squared_norm < 1e-12:
            raise RuntimeError(f"gradient signal to {block} is numerically negligible at update {step} (seed {seed})")
    return block_squared_norm


def assert_persistence_register_unaddressed(inputs: torch.Tensor) -> None:
    """Verifies the persistence register r0 is genuinely unaddressed at step 1."""
    destination_codes = inputs[:, :, 4:8].argmax(dim=-1)
    if bool((destination_codes[:, PERSIST_STEP_INDEX] == PERSIST_REGISTER_INDEX).any()):
        raise RuntimeError(
            f"persistence register r{PERSIST_REGISTER_INDEX} is the destination at step "
            f"{PERSIST_STEP_INDEX}; the intervention requires an unaddressed register"
        )


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

def run_exp_cyclic_persist_abl_01(
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-2,
    betas: Tuple[float, float] = (0.9, 0.999),
    batch_size: int = 64,
    total_updates: int = 3000,
    val_cadence: int = 50,
    lambda_persist: float = LAMBDA_PERSIST,
    device: torch.device = torch.device("cpu"),
    results_path: str = OFFICIAL_ARTIFACT_PATH,
    run_mode: str = "OFFICIAL_FIVE_SEED",
) -> Dict[str, Any]:
    """Executes EXP-CYCLIC-PERSIST-ABL-01 for the supplied seeds."""
    if results_path:
        if run_mode == "SMOKE_TEST_ONLY" and os.path.abspath(results_path) == os.path.abspath(OFFICIAL_ARTIFACT_PATH):
            raise RuntimeError(
                "smoke-test telemetry must never be written to the official results artifact "
                f"({OFFICIAL_ARTIFACT_PATH}); that artifact belongs to the unauthorized Stage 2 experiment"
            )
        if run_mode == "SMOKE_TEST_ONLY" and os.path.exists(results_path):
            raise RuntimeError(f"refusing to overwrite existing artifact: {results_path}")

    partition = Stage1PairPartition()

    # Strict firewall: only the registered TRAIN-FIT and VAL-SEEN splits are materialized.
    train_ds = partition.build_trace_split(partition.train_pairs)
    val_ds = partition.build_trace_split(partition.val_pairs)

    if len(train_ds) != REGISTERED_TRAIN_PAIRS:
        raise RuntimeError(f"expected {REGISTERED_TRAIN_PAIRS} training pairs, got {len(train_ds)}")
    if len(val_ds) != REGISTERED_VAL_PAIRS:
        raise RuntimeError(f"expected {REGISTERED_VAL_PAIRS} validation pairs, got {len(val_ds)}")

    inputs_train = train_ds.inputs.to(device)
    step_targets_train = train_ds.step_targets.to(device)
    inputs_val = val_ds.inputs.to(device)
    step_targets_val = val_ds.step_targets.to(device)

    assert_canonical_trace(inputs_train, step_targets_train, "TRAIN-FIT")
    assert_canonical_trace(inputs_val, step_targets_val, "VAL-SEEN")
    assert_protected_domain_absent(inputs_train, step_targets_train, "TRAIN-FIT")
    assert_protected_domain_absent(inputs_val, step_targets_val, "VAL-SEEN")
    assert_persistence_register_unaddressed(inputs_train)
    assert_persistence_register_unaddressed(inputs_val)

    y_destination_train = extract_destination_targets(inputs_train, step_targets_train)
    y_destination_val = extract_destination_targets(inputs_val, step_targets_val)
    if not bool(torch.equal(y_destination_train[:, -1], train_ds.targets[:, 0].to(device))):
        raise RuntimeError("dynamic terminal target disagrees with the registered final target (TRAIN-FIT)")
    if not bool(torch.equal(y_destination_val[:, -1], val_ds.targets[:, 0].to(device))):
        raise RuntimeError("dynamic terminal target disagrees with the registered final target (VAL-SEEN)")
    if batch_size != inputs_train.shape[0]:
        raise RuntimeError(
            f"registered batch size {batch_size} must equal the {inputs_train.shape[0]}-pair TRAIN-FIT split"
        )

    print("=" * 78)
    print(f"{EXPERIMENT_ID} — run mode: {run_mode}")
    print("=" * 78)
    print(f"Objective (Ablation): L_terminal + {lambda_persist} * L_persist (L_dest excluded from training)")
    print(f"Persistence query: r{PERSIST_REGISTER_INDEX} on S{PERSIST_STEP_INDEX + 1} via shared mlp_read")
    print(f"Updates: {total_updates} | Batch: {batch_size} | Eval cadence: {val_cadence}")
    print(f"Optimizer: AdamW lr={learning_rate} wd={weight_decay} betas={betas}")
    print(f"Device: {device}")

    run_records: List[Dict[str, Any]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = CyclicMGUPhase16Model().to(device)
        actual_parameters = model.count_parameters()
        if actual_parameters != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                f"parameter mismatch: expected {EXPECTED_PARAMETER_COUNT}, got {actual_parameters}"
            )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )

        checkpoints: List[Dict[str, Any]] = []
        gradient_squared_norms: List[Dict[str, float]] = []
        t_start = time.perf_counter()

        for update in range(1, total_updates + 1):
            model.train()
            optimizer.zero_grad()

            rollout = rollout_with_persistence(model, inputs_train)
            terminal_loss, destination_loss, persistence_loss, total_loss, composite_ref_loss = compute_losses(
                rollout, y_destination_train, lambda_persist
            )

            require_finite(terminal_loss, "terminal loss", update, seed)
            require_finite(destination_loss, "destination loss", update, seed)
            require_finite(persistence_loss, "persistence loss", update, seed)
            require_finite(total_loss, "total loss", update, seed)
            require_finite(composite_ref_loss, "composite reference loss", update, seed)

            total_loss.backward()
            gradient_squared_norms.append(check_gradients(model, update, seed))
            optimizer.step()

            if update % val_cadence == 0:
                train_metrics = evaluate_split(model, inputs_train, step_targets_train, lambda_persist)
                val_metrics = evaluate_split(model, inputs_val, step_targets_val, lambda_persist)

                checkpoint: Dict[str, Any] = {"update": update}
                for key, value in train_metrics.items():
                    checkpoint[f"train_{key}"] = value
                for key, value in val_metrics.items():
                    checkpoint[f"val_{key}"] = value
                checkpoints.append(checkpoint)

                print(
                    f"  [seed {seed}] update {update:5d} | "
                    f"train: total={train_metrics['loss_total']:.4f} "
                    f"term={train_metrics['loss_terminal']:.4f} "
                    f"persist={train_metrics['loss_persist']:.4f} "
                    f"(dest_diag={train_metrics['loss_dest']:.4f} comp_ref={train_metrics['loss_composite_reference']:.4f}) "
                    f"acc={train_metrics['acc']:.4f} | "
                    f"val: total={val_metrics['loss_total']:.4f} "
                    f"term={val_metrics['loss_terminal']:.4f} "
                    f"persist={val_metrics['loss_persist']:.4f} "
                    f"(dest_diag={val_metrics['loss_dest']:.4f} comp_ref={val_metrics['loss_composite_reference']:.4f}) "
                    f"acc={val_metrics['acc']:.4f}"
                )

        runtime = time.perf_counter() - t_start
        selected = select_best_checkpoint(checkpoints)
        final = checkpoints[-1]
        final_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if final_parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                f"parameter count changed during training: {final_parameter_count}"
            )

        run_records.append({
            "seed": seed,
            "parameter_count": actual_parameters,
            "parameter_count_after_training": final_parameter_count,
            "total_updates": total_updates,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "betas": list(betas),
            "lambda_dest": LAMBDA_DEST,
            "lambda_persist": lambda_persist,
            "runtime_seconds": runtime,
            "finite_value_checks_passed": total_updates,
            "gradient_blocks_present": sorted(gradient_squared_norms[-1].keys()) if gradient_squared_norms else [],
            "selected_checkpoint": selected,
            "final_checkpoint": final,
            "checkpoints": checkpoints,
            "numerical_stability": "finite_throughout",
        })

        print(
            f"  [seed {seed}] done in {runtime:.2f}s | "
            f"selected val acc={selected['val_acc']:.4f} (update {selected['update']}) | "
            f"final val acc={final['val_acc']:.4f}"
        )

        del model, optimizer

    payload: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "run_mode": run_mode,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objective": (
            "Determine whether auxiliary supervision of unaddressed-register persistence alone "
            "(L_terminal + 0.5 * L_persist) is sufficient to induce canonical coordinate preservation "
            "and modular addition generalization in CyclicMGUPhase16Model, or whether intermediate "
            "destination supervision is causally necessary."
        ),
        "hypothesis": (
            "HYPOTHESIS (not a proven cause): persistence supervision alone provides sufficient inductive "
            "constraint to preserve r0 coordinates during the S1 -> S2 transition without requiring direct "
            "intermediate destination write losses."
        ),
        "scientific_distinction": (
            "The cyclic ALU is fixed and analytical. Any improvement or degradation in this ablation "
            "concerns the supervision of the learned read, write and gate networks around it; it is not "
            "evidence of learned arithmetic from data."
        ),
        "loss_definition": {
            "total_loss": "terminal_loss + 0.5 * persistence_loss",
            "terminal_loss": "CE(logits_terminal, (a + b) % 10)",
            "destination_loss": "0.5 * [CE(logits_dest_0, a) + CE(logits_dest_1, b)] (diagnostic telemetry only, excluded from training objective)",
            "persistence_loss": "CE(logits_persist_1, a)",
            "composite_reference_loss": "terminal_loss + 1.0 * destination_loss + 0.5 * persistence_loss (diagnostic reference matching primary EXP-CYCLIC-PERSIST-01)",
            "lambda_dest": LAMBDA_DEST,
            "lambda_persist": lambda_persist,
            "step_2_auxiliary_supervision": "excluded (logits_terminal == step_logits[:, -1, :])",
            "effective_per_assertion_weight": {"persist_1": 0.5, "dest_0": 0.0, "dest_1": 0.0},
        },
        "persistence_query_definition": {
            "state": f"S{PERSIST_STEP_INDEX + 1} (post step-{PERSIST_STEP_INDEX} transition, instruction 'SET r1 b')",
            "register": f"r{PERSIST_REGISTER_INDEX} (unaddressed at step {PERSIST_STEP_INDEX}, written at step 0)",
            "address": "e_r0 = [1, 0, 0, 0]",
            "readout": "project(mlp_read([S2, e_r0])) using the existing shared mlp_read network",
            "decode": "prototype bank u_persist @ Phi^T (fixed 5-harmonic basis)",
            "target": "a (the r0 value written at step 0)",
            "equivalence": "u_persist_1 is computationally identical to the step-2 operand read u_a",
        },
        "target_definition": {
            "canonical_trace": ["SET r0 a", "SET r1 b", "ADD r0 r1"],
            "destination_target_sequence": "(a, b, (a + b) % 10)",
            "persistence_target": "a",
            "extraction": "step_targets[b, t, argmax(inputs[b, t, 4:8])]",
            "verification": "matched Phase16RegisterTask.simulate_trace() on every allowed example",
        },
        "metric_definitions": {
            "c_read": "mean of the per-role mean cosine(u_a, Phi(a)) and cosine(u_b, Phi(b)) at step 2",
            "c_read_r0": "mean cosine(u_a, Phi(a)) at step 2",
            "c_read_r1": "mean cosine(u_b, Phi(b)) at step 2",
            "c_persist_r0": "mean cosine(u_persist_1, Phi(a)) at step 1",
            "s_context": "unweighted mean over digits of the mean pairwise cosine similarity of "
                         "same-integer step-2 operand reads pooled across roles and contexts",
            "c_alu": "mean cosine of the raw cyclic_alu(u_a, u_b) tensor at step 2 with Phi((a+b)%10)",
            "c_dest": "mean of the step-0/1/2 mean cosines of u_dest_post with Phi(y_destination_t)",
            "loss_composite_reference": "terminal_loss + 1.0 * destination_loss + 0.5 * persistence_loss",
            "cosine_guard": f"denominator epsilon {COSINE_EPS}; similarity 0.0 if either norm < {MIN_VECTOR_NORM}",
        },
        "protocol": {
            "seeds": list(seeds),
            "total_updates": total_updates,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "betas": list(betas),
            "evaluation_cadence": val_cadence,
            "checkpoint_selection": "minimum validation total loss under native ablation objective, ties by highest validation accuracy, then earliest update",
            "train_pairs_count": REGISTERED_TRAIN_PAIRS,
            "val_pairs_count": REGISTERED_VAL_PAIRS,
            "model_parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "protected_test_split": {
            "definition": f"{PROTECTED_TEST_PAIRS} operand pairs with b in {list(PROTECTED_TEST_B_VALUES)}",
            "status": "STRICTLY FIREWALLED — never loaded, constructed, evaluated, queried or logged",
            "queries": 0,
        },
        "runs": run_records,
    }

    if run_mode == "SMOKE_TEST_ONLY":
        payload["artifact_status"] = SMOKE_ARTIFACT_STATUS
        payload["gate_assessment"] = None
        payload["gate_assessment_note"] = (
            "No gate assessment is produced. A single-seed 100-update smoke run is an implementation and "
            "instrumentation check only; the registered gate requires the separate, unauthorized five-seed "
            "(seeds 42-46, 3000-update) Stage 2 experiment."
        )
    elif run_mode == "OFFICIAL_FIVE_SEED":
        val_accs = [r["selected_checkpoint"]["val_acc"] for r in run_records]
        c_reads = [r["selected_checkpoint"]["val_c_read"] for r in run_records]
        s_ctxs = [r["selected_checkpoint"]["val_s_context"] for r in run_records]
        c_alus = [r["selected_checkpoint"]["val_c_alu"] for r in run_records]
        val_losses = [r["selected_checkpoint"]["val_loss_total"] for r in run_records]
        payload["gate_assessment"] = classify_experiment_outcome(
            val_accs, c_reads, s_ctxs, c_alus, val_losses=val_losses
        )

    if results_path:
        directory = os.path.dirname(os.path.abspath(results_path))
        os.makedirs(directory, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote telemetry artifact: {results_path}")

    return payload


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA/ROCm device is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_ID} runner (staged authorization)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Stage 1 smoke test only: seed 42, 100 updates, smoke-labelled telemetry artifact.",
    )
    parser.add_argument(
        "--stage2-authorized",
        action="store_true",
        help="Run the official five-seed experiment. Requires separate explicit PI Stage 2 authorization.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.smoke and args.stage2_authorized:
        parser.error("--smoke and --stage2-authorized are mutually exclusive")

    device = _resolve_device(args.device)

    if args.smoke:
        run_exp_cyclic_persist_abl_01(
            seeds=[42],
            total_updates=100,
            val_cadence=50,
            device=device,
            results_path=SMOKE_ARTIFACT_PATH,
            run_mode="SMOKE_TEST_ONLY",
        )
        print(
            "\nStage 1 smoke test complete. This is NOT scientific evidence about the research "
            "hypothesis; the five-seed Stage 2 experiment remains unauthorized."
        )
        return

    if args.stage2_authorized:
        run_exp_cyclic_persist_abl_01(
            seeds=[42, 43, 44, 45, 46],
            total_updates=3000,
            val_cadence=50,
            device=device,
            results_path=OFFICIAL_ARTIFACT_PATH,
            run_mode="OFFICIAL_FIVE_SEED",
        )
        return

    parser.error(
        "Stage 1 authorizes only --smoke. The official five-seed experiment requires explicit "
        "Stage 2 authorization (--stage2-authorized)."
    )


if __name__ == "__main__":
    main()
