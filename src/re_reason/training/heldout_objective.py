"""Model E held-out generalization: unified objective, held-out metrics, and Tier 1 gate.

Implements the registered protocol math of the 2026-10-18 Phase 1.6 amendment
(`docs/reports/phase_1_6_architectural_reformulation_proposal.md`):

* Unified objective (all four candidates):
  ``L_total = (1/B) sum_i (L_term^(i) + 1.0 * L_step^(i))`` where ``L_term`` is the
  mean terminal cross-entropy across the four registers at step ``T`` and
  ``L_step`` is the per-step unique-written-register cross-entropy normalized by
  the sequence's total written-register count (0.0 when no register is written).
* Held-out ADD balanced accuracy: macro recall across true output classes present
  among eligible held-out ADD transitions (runtime source value in the registered
  eligible set).
* Terminal sequence exact match: predicted final four-register state equal to the
  target state at ``t = T``.
* Tier 1 gate classification for the PROXY-DEV evaluation of Model A.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


NUM_REGISTERS = 4
NUM_VALUES = 10
LAMBDA_STEP = 1.0

TIER1_ACC_MIN = 0.800
TIER1_EM_MIN = 0.700
TIER1_REJECTION_ACC = 0.250  # rejection zone when acc < 0.250
TIER1_REJECTION_EM = 0.150  # rejection zone when em <= 0.150
TIER1_SEEDS_REQUIRED = 5


def written_register_masks(tokens: torch.Tensor) -> torch.Tensor:
    """Per-step unique-written-register masks derived from instruction tokens.

    ``SET`` / ``ADD`` -> ``{r_dest}``; ``SWAP`` -> ``{r_dest, r_src}`` (set
    semantics; a self-swap counts once); ``READ`` -> ``{}``.

    Args:
        tokens: One-hot instruction tokens of shape ``(B, T, 22)``.

    Returns:
        Float mask of shape ``(B, T, 4)`` with values in ``{0, 1}``.
    """
    op = tokens[..., 0:4]
    dest = tokens[..., 4:8]
    src = tokens[..., 8:12]

    w_set = op[..., 0]
    w_swap = op[..., 1]
    w_add = op[..., 3]

    write_mask = (w_set + w_add).unsqueeze(-1) * dest + w_swap.unsqueeze(-1) * (dest + src).clamp(
        max=1.0
    )
    return write_mask


def unified_heldout_loss(
    step_logits: torch.Tensor,
    step_targets: torch.Tensor,
    tokens: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Registered unified objective for the four-cell held-out campaign.

    Args:
        step_logits: Post-step register logits of shape ``(B, T, 4, 10)``.
        step_targets: Post-step register targets of shape ``(B, T, 4)``.
        tokens: Instruction tokens of shape ``(B, T, 22)``.

    Returns:
        Tuple of (scalar loss, component telemetry dict).
    """
    batch_size, seq_len, num_regs, num_vals = step_logits.shape
    if step_targets.shape != (batch_size, seq_len, num_regs):
        raise ValueError(
            f"step_targets shape {tuple(step_targets.shape)} does not match logits "
            f"{(batch_size, seq_len, num_regs)}"
        )
    if tokens.shape[:2] != (batch_size, seq_len):
        raise ValueError(f"tokens shape {tuple(tokens.shape)} incompatible with logits")

    masks = written_register_masks(tokens)  # (B, T, 4)
    per_register_ce = F.cross_entropy(
        step_logits.reshape(-1, num_vals),
        step_targets.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len, num_regs)

    n_writes = masks.sum(dim=(1, 2))  # (B,)
    step_sums = (per_register_ce * masks).sum(dim=(1, 2))  # (B,)
    l_step = torch.where(
        n_writes > 0.0,
        step_sums / n_writes.clamp_min(1.0),
        torch.zeros_like(step_sums),
    )
    l_terminal = per_register_ce[:, -1, :].mean(dim=1)  # (B,)

    per_sequence = l_terminal + LAMBDA_STEP * l_step
    total = per_sequence.mean()

    components = {
        "l_terminal_mean": float(l_terminal.mean().detach()),
        "l_step_mean": float(l_step.mean().detach()),
        "n_writes_mean": float(n_writes.mean().detach()),
    }
    return total, components


def collect_eligible_add_transitions(
    task: Any,
    inputs: torch.Tensor,
    eligible_source_values: Tuple[int, ...],
) -> List[List[Tuple[int, int]]]:
    """Replays sequences and returns eligible ADD transitions as ``(step, reg_dest)``.

    A transition is eligible when its runtime source-register value ``v(r_src)`` is
    in ``eligible_source_values`` (e.g. ``(3,)`` for PROXY-DEV, ``(0, 5)`` for the
    protected split). Eligibility is evaluated at the exact runtime moment of the
    transition via the deterministic replay simulator.
    """
    inputs_np = inputs.detach().cpu().numpy()
    eligible: List[List[Tuple[int, int]]] = []
    for i in range(inputs_np.shape[0]):
        _, history = task.simulate_trace(inputs_np[i])
        eligible.append(
            [
                (int(h["step"]), int(h["reg_dest"]))
                for h in history
                if h["op_type"] == task.OP_ADD and h["runtime_src_val_pre"] in eligible_source_values
            ]
        )
    return eligible


def heldout_add_balanced_accuracy(
    predicted_classes: Sequence[int],
    true_classes: Sequence[int],
) -> Dict[str, Any]:
    """Macro recall across true output classes present among eligible transitions.

    Returns per-class instance counts, per-class recalls, and the total eligible
    transition count ``N`` alongside the macro-averaged recall.
    """
    if len(predicted_classes) != len(true_classes):
        raise ValueError("predicted_classes and true_classes must have equal length")
    if len(true_classes) == 0:
        raise ValueError("no eligible held-out ADD transitions: balanced accuracy undefined")

    counts: Dict[int, int] = {c: 0 for c in range(NUM_VALUES)}
    correct: Dict[int, int] = {c: 0 for c in range(NUM_VALUES)}
    for pred, target in zip(predicted_classes, true_classes):
        counts[target] += 1
        if pred == target:
            correct[target] += 1

    present = [c for c in range(NUM_VALUES) if counts[c] > 0]
    recalls = {c: (correct[c] / counts[c]) for c in present}
    macro = sum(recalls.values()) / len(present)

    return {
        "macro_recall": float(macro),
        "n_transitions": len(true_classes),
        "per_class_counts": counts,
        "per_class_recalls": {c: float(recalls[c]) for c in present},
    }


def terminal_sequence_exact_match(
    predicted_terminal: torch.Tensor,
    target_terminal: torch.Tensor,
) -> float:
    """Fraction of sequences whose predicted final four-register state matches exactly."""
    if predicted_terminal.shape != target_terminal.shape:
        raise ValueError(
            f"predicted and target terminal states differ in shape: "
            f"{tuple(predicted_terminal.shape)} vs {tuple(target_terminal.shape)}"
        )
    if predicted_terminal.shape[0] == 0:
        raise ValueError("empty terminal-state population: exact match undefined")
    matches = (predicted_terminal == target_terminal).all(dim=-1)
    return float(matches.float().mean())


def classify_tier1_gate(seed_metrics: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    """Classifies the Tier 1 PROXY-DEV gate for the registered five seeds.

    Per-seed pass: ``acc >= 0.800`` AND ``em >= 0.700``.
    Per-seed rejection: ``acc < 0.250`` OR ``em <= 0.150``.
    Overall pass: >= 4/5 seeds pass with zero seeds in the rejection zone.
    Overall rejection: >= 4/5 seeds in the rejection zone. Otherwise: halt.
    """
    if len(seed_metrics) != TIER1_SEEDS_REQUIRED:
        raise ValueError(
            f"Tier 1 gate is registered over exactly {TIER1_SEEDS_REQUIRED} seeds, "
            f"got {len(seed_metrics)}"
        )

    seed_pass: List[bool] = []
    seed_reject: List[bool] = []
    for entry in seed_metrics:
        acc = float(entry["acc"])
        em = float(entry["em"])
        seed_pass.append(acc >= TIER1_ACC_MIN and em >= TIER1_EM_MIN)
        seed_reject.append(acc < TIER1_REJECTION_ACC or em <= TIER1_REJECTION_EM)

    n_pass = sum(seed_pass)
    n_reject = sum(seed_reject)

    if n_pass >= 4 and n_reject == 0:
        verdict = "tier1_pass"
    elif n_reject >= 4:
        verdict = "tier1_reject"
    else:
        verdict = "tier1_halt"

    return {
        "verdict": verdict,
        "n_pass": n_pass,
        "n_reject": n_reject,
        "seed_pass": seed_pass,
        "seed_reject": seed_reject,
        "thresholds": {
            "acc_min": TIER1_ACC_MIN,
            "em_min": TIER1_EM_MIN,
            "rejection_acc": TIER1_REJECTION_ACC,
            "rejection_em": TIER1_REJECTION_EM,
        },
    }
