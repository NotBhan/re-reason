#!/usr/bin/env python3
"""RE:Reason Final Phase-1 Experiment — Oracle Branching Feasibility.

Core question
-------------
Can multiple speculative candidate states, sampled by injecting stochasticity
ONLY into the candidate generator (frozen S_(t-1), e_t and gate f_t), recover
errors that the frozen deterministic K=1 model cannot?

Rules enforced by construction
------------------------------
* The recurrent core, benchmark and trained checkpoint are untouched.
* The gate is never perturbed; candidate sampling never sees ground truth.
* The true register state is used ONLY to (a) score candidates (oracle
  verifier) and (b) define errors.

Frozen model (checkpoint, no retraining)
----------------------------------------
Nonlinear Candidate MGU:
    f_t = sigmoid(W_f e_t + U_f S_(t-1) + b_f)
    h   = GELU(W_1 [f_t * S_(t-1), e_t] + b_1)          <- noise injected HERE only
    C   = tanh(W_2 h + b_2)
    S_t = (1 - f_t) * S_(t-1) + f_t * C

Candidate construction per transition
--------------------------------------
C_0 = deterministic candidate (zero noise, exactly reproduces the frozen model).
C_1..C_{K-1} = K-1 stochastic draws (draw prefix is nested across K values).
Branch state B_k = (1 - f_t) * S_(t-1) + f_t * C_k (the state a branch would carry).

Experiments
-----------
EXP 1  candidate diversity / pairwise L2 distance / pairwise cosine similarity
EXP 2  per-transition oracle recovery of the modified register(s)
EXP 3  per-transition oracle recovery of the full 4-register state
EXP 4  failure-only A/B/C/D taxonomy (on deterministic full-state failures)
EXP 5  SET / SWAP / ADD (wrap vs no-wrap) breakdown
EXP 6  first-error-timestep breakdown
EXP 7  random-S_candidate-perturbation control (magnitude matched)
EXP 8  oracle EM upper bound (perfect per-step verifier, K in {1,2,4,8})

Nothing here is "Phase 2 branching": this is an offline feasibility probe of the
candidate generator's support. The oracle EM number is an upper bound under a
perfect selector, not an achievable result.
"""

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from re_reason.benchmarks.dataset_split import DisjointSplitManager
from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.core.ablation_models import NonlinearCandidateMGUModel


CHECKPOINT = "experiments/checkpoints/6_nonlinear_candidate_mgu_seed42_3000steps.pt"
DEFAULT_OUTPUT = "experiments/branching_feasibility_results.json"

OP_SET, OP_SWAP, OP_ADD = 0, 1, 2
OP_CAT_NAMES = {0: "SET", 1: "SWAP", 2: "ADD_no_wrap", 3: "ADD_wrap"}
K_VALUES = (1, 2, 4, 8)
K_MAX = 8
N_DRAWS = K_MAX - 1

NOISE_SEED_LOCAL = 20261005
NOISE_SEED_EM = 20262005

PRIMARY_MECHANISM = "gauss_hidden_c1.0"

# Mechanism definitions. Primary is fixed a priori (activation-scale Gaussian noise
# on the candidate MLP hidden activation). The rest exist for sensitivity reporting
# and for the EXP 7 control; none of them are tuned against ground truth.
MECHANISMS: List[Tuple[int, str, Dict[str, Any]]] = [
    (0, "gauss_hidden_c1.0", {"kind": "gauss_hidden", "scale": 1.0}),
    (1, "gauss_hidden_c0.25", {"kind": "gauss_hidden", "scale": 0.25}),
    (2, "gauss_hidden_c0.5", {"kind": "gauss_hidden", "scale": 0.5}),
    (3, "gauss_hidden_c2.0", {"kind": "gauss_hidden", "scale": 2.0}),
    (4, "dropout_hidden_p0.1", {"kind": "dropout", "p": 0.1}),
    (5, "dropout_hidden_p0.2", {"kind": "dropout", "p": 0.2}),
    (6, "rand_S_cand_matched", {"kind": "rand_cand", "scale": 1.0, "matched": True}),
    (7, "rand_S_cand_c1.0", {"kind": "rand_cand", "scale": 1.0, "matched": False}),
]


# ---------------------------------------------------------------------------
# Model loading and deterministic pass
# ---------------------------------------------------------------------------

def load_frozen_model(device: torch.device) -> NonlinearCandidateMGUModel:
    model = NonlinearCandidateMGUModel(
        input_dim=21,
        state_dim=212,
        encoder_out_dim=256,
        candidate_hidden_dim=146,
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def deterministic_pass(
    model: NonlinearCandidateMGUModel, inputs: torch.Tensor
) -> Dict[str, List[torch.Tensor]]:
    """Records the frozen deterministic trajectory components for every transition."""
    batch_size, seq_len, _ = inputs.shape
    device = inputs.device
    core = model.cognition_core

    state = torch.zeros(batch_size, model.state_dim, device=device)
    gates: List[torch.Tensor] = []
    embeddings: List[torch.Tensor] = []
    hiddens: List[torch.Tensor] = []
    candidates: List[torch.Tensor] = []
    states: List[torch.Tensor] = []

    for t in range(seq_len):
        e_t = model.encoder(inputs[:, t, :])
        f_t = torch.sigmoid(core.w_f_ih(e_t) + core.w_f_hh(state))
        h_t = core.cand_mlp[1](core.cand_mlp[0](torch.cat([f_t * state, e_t], dim=-1)))
        c_t = torch.tanh(core.cand_mlp[2](h_t))
        next_state = (1.0 - f_t) * state + f_t * c_t

        gates.append(f_t)
        embeddings.append(e_t)
        hiddens.append(h_t)
        candidates.append(c_t)
        states.append(next_state)
        state = next_state

    return {
        "gates": gates,
        "embeddings": embeddings,
        "hiddens": hiddens,
        "candidates": candidates,
        "states": states,
    }


def build_ground_truth_meta(
    inputs: torch.Tensor,
    step_targets: torch.Tensor,
    final_targets: torch.Tensor,
    det_preds: np.ndarray,
) -> Dict[str, Any]:
    """Per-transition ground-truth metadata used only for scoring."""
    batch_size, seq_len, _ = inputs.shape
    inp = inputs.cpu().numpy()
    true_states = step_targets.cpu().numpy().astype(np.int64)
    final_true = final_targets.cpu().numpy().astype(np.int64)

    op = inp[:, :, :3].argmax(-1)
    reg_a = inp[:, :, 3:7].argmax(-1)
    reg_b = inp[:, :, 7:11].argmax(-1)
    prev = np.zeros_like(true_states)
    prev[:, 1:, :] = true_states[:, :-1, :]

    write_mask = np.zeros((batch_size, seq_len, 4), dtype=bool)
    op_cat = np.zeros((batch_size, seq_len), dtype=np.int64)
    add_same_register = np.zeros((batch_size, seq_len), dtype=bool)

    for b in range(batch_size):
        for t in range(seq_len):
            o = int(op[b, t])
            a = int(reg_a[b, t])
            if o == OP_SWAP:
                write_mask[b, t, a] = True
                write_mask[b, t, int(reg_b[b, t])] = True
                op_cat[b, t] = 1
            elif o == OP_ADD:
                write_mask[b, t, a] = True
                wrap = prev[b, t, a] + prev[b, t, int(reg_b[b, t])] >= 10
                op_cat[b, t] = 3 if wrap else 2
                add_same_register[b, t] = int(reg_b[b, t]) == a
            else:
                write_mask[b, t, a] = True
                op_cat[b, t] = 0

    det_full_ok = (det_preds == true_states).all(-1)
    det_tgt_ok = ((det_preds == true_states) | (~write_mask)).all(-1)
    terminal_fail = (det_preds[:, -1, :] != final_true).any(-1)

    first_error_t = np.full(batch_size, -1, dtype=np.int64)
    first_target_error_t = np.full(batch_size, -1, dtype=np.int64)
    for b in range(batch_size):
        bad = np.nonzero(~det_full_ok[b])[0]
        if bad.size > 0:
            first_error_t[b] = int(bad[0]) + 1  # 1-indexed timestep
        bad_t = np.nonzero(~det_tgt_ok[b])[0]
        if bad_t.size > 0:
            first_target_error_t[b] = int(bad_t[0]) + 1

    no_step_error_but_terminal_fail = int(np.sum(terminal_fail & (first_error_t == -1)))
    ghost_only_transitions = (~det_full_ok) & det_tgt_ok

    return {
        "true_states": true_states,
        "final_true": final_true,
        "write_mask": write_mask,
        "op_cat": op_cat,
        "add_same_register": add_same_register,
        "det_full_ok": det_full_ok,
        "det_tgt_ok": det_tgt_ok,
        "terminal_fail": terminal_fail,
        "first_error_t": first_error_t,
        "first_target_error_t": first_target_error_t,
        "no_step_error_but_terminal_fail": no_step_error_but_terminal_fail,
        "failure_anatomy": {
            "per_transition_full_state_failures": int((~det_full_ok).sum()),
            "per_transition_target_register_failures": int((~det_tgt_ok).sum()),
            "per_transition_ghost_only_failures": int(ghost_only_transitions.sum()),
            "terminal_failures": int(terminal_fail.sum()),
            "terminal_failures_with_real_target_error": int(
                (terminal_fail & (first_target_error_t > 0)).sum()
            ),
            "terminal_failures_ghost_only": int(
                (terminal_fail & (first_target_error_t == -1)).sum()
            ),
        },
    }


# ---------------------------------------------------------------------------
# Candidate sampling (stochasticity ONLY in candidate generation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_candidates(
    core: Any,
    s_prev: torch.Tensor,
    f_t: torch.Tensor,
    h_t: torch.Tensor,
    mech: Dict[str, Any],
    stats: Dict[str, torch.Tensor],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Returns (B, K_MAX, state_dim) raw candidates C; index 0 is deterministic."""
    batch_size = h_t.shape[0]
    w2 = core.cand_mlp[2].weight
    b2 = core.cand_mlp[2].bias
    c0 = torch.tanh(F.linear(h_t, w2, b2))

    kind = mech["kind"]
    if kind == "gauss_hidden":
        eps = torch.randn(batch_size, N_DRAWS, h_t.shape[-1], generator=generator).to(device)
        sigma = stats["h_std"].to(device)
        h_pert = h_t[:, None, :] + mech["scale"] * sigma[None, None, :] * eps
        c_extra = torch.tanh(F.linear(h_pert, w2, b2))
    elif kind == "dropout":
        keep = (torch.rand(batch_size, N_DRAWS, h_t.shape[-1], generator=generator) >= mech["p"]).to(device)
        h_pert = h_t[:, None, :] * keep / (1.0 - mech["p"])
        c_extra = torch.tanh(F.linear(h_pert, w2, b2))
    elif kind == "rand_cand":
        eps = torch.randn(batch_size, N_DRAWS, core.state_dim, generator=generator).to(device)
        sigma = stats["c_std"].to(device)
        c_extra = torch.clamp(
            c0[:, None, :] + mech["scale"] * sigma[None, None, :] * eps, -1.0, 1.0
        )
    else:
        raise ValueError(f"Unknown mechanism kind: {kind}")

    return torch.cat([c0[:, None, :], c_extra], dim=1)


# ---------------------------------------------------------------------------
# EXP 1-7: local transition-level pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_local_pass(
    model: NonlinearCandidateMGUModel,
    det: Dict[str, List[torch.Tensor]],
    mech_id: int,
    mech: Dict[str, Any],
    stats: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Samples candidates at every frozen transition and records decodes + geometry."""
    core = model.cognition_core
    head = model.decision_head
    batch_size = det["gates"][0].shape[0]
    seq_len = len(det["gates"])

    branch_preds = np.zeros((batch_size, seq_len, K_MAX, 4), dtype=np.int8)
    cand_preds = np.zeros((batch_size, seq_len, K_MAX, 4), dtype=np.int8)
    pair_l2_cand = {k: np.zeros((batch_size, seq_len), dtype=np.float32) for k in (2, 4, 8)}
    pair_cos_cand = {k: np.zeros((batch_size, seq_len), dtype=np.float32) for k in (2, 4, 8)}
    pair_l2_branch = {k: np.zeros((batch_size, seq_len), dtype=np.float32) for k in (2, 4, 8)}
    pair_cos_branch = {k: np.zeros((batch_size, seq_len), dtype=np.float32) for k in (2, 4, 8)}
    displacement = np.zeros((batch_size, seq_len, N_DRAWS), dtype=np.float32)

    for t in range(seq_len):
        s_prev = (
            torch.zeros(batch_size, model.state_dim, device=device)
            if t == 0
            else det["states"][t - 1]
        )
        f_t = det["gates"][t]
        h_t = det["hiddens"][t]

        generator = torch.Generator(device="cpu").manual_seed(
            NOISE_SEED_LOCAL + 1000 * mech_id + t
        )
        candidates = sample_candidates(core, s_prev, f_t, h_t, mech, stats, generator, device)
        branch_states = (1.0 - f_t)[:, None, :] * s_prev[:, None, :] + f_t[:, None, :] * candidates

        dec_c = head(candidates.reshape(batch_size * K_MAX, -1)).view(batch_size, K_MAX, 4, 10).argmax(-1)
        dec_b = head(branch_states.reshape(batch_size * K_MAX, -1)).view(batch_size, K_MAX, 4, 10).argmax(-1)
        cand_preds[:, t] = dec_c.cpu().numpy().astype(np.int8)
        branch_preds[:, t] = dec_b.cpu().numpy().astype(np.int8)

        displacement[:, t, :] = (
            (candidates[:, 1:, :] - candidates[:, :1, :]).norm(dim=-1).cpu().numpy()
        )

        for k in (2, 4, 8):
            iu = torch.triu_indices(k, k, offset=1)
            for tag, tensor in (("cand", candidates), ("branch", branch_states)):
                sub = tensor[:, :k, :]
                dmat = torch.cdist(sub, sub)
                dn = F.normalize(sub, dim=-1)
                cmat = torch.bmm(dn, dn.transpose(1, 2))
                if tag == "cand":
                    pair_l2_cand[k][:, t] = dmat[:, iu[0], iu[1]].mean(dim=-1).cpu().numpy()
                    pair_cos_cand[k][:, t] = cmat[:, iu[0], iu[1]].mean(dim=-1).cpu().numpy()
                else:
                    pair_l2_branch[k][:, t] = dmat[:, iu[0], iu[1]].mean(dim=-1).cpu().numpy()
                    pair_cos_branch[k][:, t] = cmat[:, iu[0], iu[1]].mean(dim=-1).cpu().numpy()

    return {
        "branch_preds": branch_preds,
        "cand_preds": cand_preds,
        "pair_l2_cand": pair_l2_cand,
        "pair_cos_cand": pair_cos_cand,
        "pair_l2_branch": pair_l2_branch,
        "pair_cos_branch": pair_cos_branch,
        "displacement": displacement,
    }


def aggregate_local(res: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Computes EXP 1-7 aggregates for one mechanism across all K values."""
    true = meta["true_states"]
    write_mask = meta["write_mask"]
    op_cat = meta["op_cat"]
    det_fail = ~meta["det_full_ok"]
    det_tgt_fail = ~meta["det_tgt_ok"]
    terminal_fail = meta["terminal_fail"]
    first_error_t = meta["first_error_t"]
    first_target_error_t = meta["first_target_error_t"]
    seq_len = true.shape[1]

    out: Dict[str, Any] = {"per_K": {}}

    for k in K_VALUES:
        bp = res["branch_preds"][:, :, :k, :].astype(np.int64)
        cp = res["cand_preds"][:, :, :k, :].astype(np.int64)
        full_ok = (bp == true[:, :, None, :]).all(-1)
        tgt_ok = ((bp == true[:, :, None, :]) | (~write_mask[:, :, None, :])).all(-1)
        cand_full_ok = (cp == true[:, :, None, :]).all(-1)
        cand_tgt_ok = ((cp == true[:, :, None, :]) | (~write_mask[:, :, None, :])).all(-1)

        any_full = full_ok.any(-1)
        any_tgt = tgt_ok.any(-1)
        n_full = full_ok.sum(-1)

        n_fail = int(det_fail.sum())
        n_tgt_fail = int(det_tgt_fail.sum())

        # EXP 4: A/B/C/D taxonomy on deterministic failures (strict full-state).
        nf = n_full[det_fail]
        any_tgt_on_fail = any_tgt[det_fail]
        categories = {
            "A_no_candidate_correct": int(((nf == 0) & (~any_tgt_on_fail)).sum()),
            "B_exactly_one_full_correct": int((nf == 1).sum()),
            "C_multiple_full_correct": int((nf >= 2).sum()),
            "D_target_only_correct": int(((nf == 0) & any_tgt_on_fail).sum()),
        }
        nf_t = n_full[det_tgt_fail]
        any_full_on_tgt_fail = any_full[det_tgt_fail]
        any_tgt_on_tgt_fail = any_tgt[det_tgt_fail]
        categories_target_failures = {
            "A_no_candidate_correct": int(((nf_t == 0) & (~any_tgt_on_tgt_fail)).sum()),
            "B_exactly_one_full_correct": int((nf_t == 1).sum()),
            "C_multiple_full_correct": int((nf_t >= 2).sum()),
            "D_target_only_correct": int(((nf_t == 0) & any_tgt_on_tgt_fail).sum()),
        }

        # EXP 5: operation breakdown.
        by_op: Dict[str, Any] = {}
        by_op_target_failures: Dict[str, Any] = {}
        for cat in range(4):
            m = det_fail & (op_cat == cat)
            by_op[OP_CAT_NAMES[cat]] = {
                "failures": int(m.sum()),
                "recoverable_full_state": int((any_full & m).sum()),
                "recoverable_target_register": int((any_tgt & m).sum()),
            }
            mt = det_tgt_fail & (op_cat == cat)
            by_op_target_failures[OP_CAT_NAMES[cat]] = {
                "failures": int(mt.sum()),
                "recoverable_full_state": int((any_full & mt).sum()),
                "recoverable_target_register": int((any_tgt & mt).sum()),
            }

        # EXP 6: first-error timestep breakdown (terminal failures only).
        by_first_error: Dict[str, Any] = {}
        for t in range(1, seq_len + 1):
            sel = np.where(terminal_fail & (first_error_t == t))[0]
            n = int(sel.size)
            rec_full = int(any_full[sel, t - 1].sum()) if n else 0
            rec_tgt = int(any_tgt[sel, t - 1].sum()) if n else 0
            by_first_error[str(t)] = {
                "failed_trajectories": n,
                "first_error_transition_recoverable_full_state": rec_full,
                "first_error_transition_recoverable_target_register": rec_tgt,
            }

        # EXP 6b: first *target-register* error timestep (real computational slips only).
        by_first_target_error: Dict[str, Any] = {}
        for t in range(1, seq_len + 1):
            sel = np.where(terminal_fail & (first_target_error_t == t))[0]
            n = int(sel.size)
            rec_full = int(any_full[sel, t - 1].sum()) if n else 0
            rec_tgt = int(any_tgt[sel, t - 1].sum()) if n else 0
            by_first_target_error[str(t)] = {
                "failed_trajectories": n,
                "first_error_transition_recoverable_full_state": rec_full,
                "first_error_transition_recoverable_target_register": rec_tgt,
            }

        # EXP 1: diversity aggregates.
        def masked_mean(arr: np.ndarray) -> float:
            return float(arr[det_fail].mean()) if det_fail.any() else 0.0

        if k == 1:
            diversity = {
                "mean_pairwise_l2_candidates": 0.0,
                "mean_pairwise_cos_candidates": 1.0,
                "mean_pairwise_l2_candidates_on_failures": 0.0,
                "mean_pairwise_cos_candidates_on_failures": 1.0,
                "mean_pairwise_l2_branch_states": 0.0,
                "mean_pairwise_cos_branch_states": 1.0,
            }
        else:
            diversity = {
                "mean_pairwise_l2_candidates": float(res["pair_l2_cand"][k].mean()),
                "mean_pairwise_cos_candidates": float(res["pair_cos_cand"][k].mean()),
                "mean_pairwise_l2_candidates_on_failures": masked_mean(res["pair_l2_cand"][k]),
                "mean_pairwise_cos_candidates_on_failures": masked_mean(res["pair_cos_cand"][k]),
                "mean_pairwise_l2_branch_states": float(res["pair_l2_branch"][k].mean()),
                "mean_pairwise_cos_branch_states": float(res["pair_cos_branch"][k].mean()),
            }

        per_register_branch_acc = (bp == true[:, :, None, :]).mean(axis=(0, 1)).tolist()
        per_register_cand_acc = (cp == true[:, :, None, :]).mean(axis=(0, 1)).tolist()

        out["per_K"][str(k)] = {
            "any_target_register_rate_all_transitions": float(any_tgt.mean()),
            "any_full_state_rate_all_transitions": float(any_full.mean()),
            "any_target_register_rate_on_det_full_state_failures": (
                float(any_tgt[det_fail].mean()) if n_fail else 0.0
            ),
            "any_full_state_rate_on_det_full_state_failures": (
                float(any_full[det_fail].mean()) if n_fail else 0.0
            ),
            "any_target_register_rate_on_det_target_failures": (
                float(any_tgt[det_tgt_fail].mean()) if n_tgt_fail else 0.0
            ),
            "any_full_state_rate_on_det_target_failures": (
                float(any_full[det_tgt_fail].mean()) if n_tgt_fail else 0.0
            ),
            "raw_candidate_any_target_register_rate_all_transitions": float(cand_tgt_ok.any(-1).mean()),
            "raw_candidate_any_full_state_rate_all_transitions": float(cand_full_ok.any(-1).mean()),
            "raw_candidate_any_target_register_rate_on_det_failures": (
                float(cand_tgt_ok.any(-1)[det_fail].mean()) if n_fail else 0.0
            ),
            "raw_candidate_any_full_state_rate_on_det_failures": (
                float(cand_full_ok.any(-1)[det_fail].mean()) if n_fail else 0.0
            ),
            "raw_stochastic_any_target_register_rate_on_det_target_failures": (
                float(cand_tgt_ok[:, :, 1:].any(-1)[det_tgt_fail].mean()) if n_tgt_fail else 0.0
            ),
            "raw_stochastic_any_full_state_rate_on_det_target_failures": (
                float(cand_full_ok[:, :, 1:].any(-1)[det_tgt_fail].mean()) if n_tgt_fail else 0.0
            ),
            "deterministic_full_state_failures": n_fail,
            "deterministic_target_register_failures": n_tgt_fail,
            "categories_on_det_full_state_failures": categories,
            "categories_on_det_target_failures": categories_target_failures,
            "by_operation": by_op,
            "by_operation_target_failures": by_op_target_failures,
            "by_first_error_timestep": by_first_error,
            "by_first_target_error_timestep": by_first_target_error,
            "diversity": diversity,
            "branch_per_register_accuracy": per_register_branch_acc,
            "raw_candidate_per_register_accuracy": per_register_cand_acc,
        }

    out["mean_displacement_norm_primary_draws"] = float(res["displacement"].mean())

    # Per-candidate-index diagnostics (is recovery a structured capability or a lottery?).
    bp8 = res["branch_preds"].astype(np.int64)
    cp8 = res["cand_preds"].astype(np.int64)
    branch_tgt8 = ((bp8 == true[:, :, None, :]) | (~write_mask[:, :, None, :])).all(-1)
    branch_full8 = (bp8 == true[:, :, None, :]).all(-1)
    raw_tgt8 = ((cp8 == true[:, :, None, :]) | (~write_mask[:, :, None, :])).all(-1)

    def index_rates(arr: np.ndarray, mask: np.ndarray) -> List[float]:
        denom = int(mask.sum())
        if denom == 0:
            return [0.0] * K_MAX
        return [float(arr[:, :, j][mask].mean()) for j in range(K_MAX)]

    out["per_candidate_index"] = {
        "branch_target_ok_on_det_target_failures": index_rates(branch_tgt8, det_tgt_fail),
        "branch_full_ok_on_det_target_failures": index_rates(branch_full8, det_tgt_fail),
        "raw_candidate_target_ok_on_det_target_failures": index_rates(raw_tgt8, det_tgt_fail),
        "branch_target_ok_all_transitions": index_rates(branch_tgt8, np.ones_like(det_tgt_fail, dtype=bool)),
    }
    return out


# ---------------------------------------------------------------------------
# EXP 8: oracle EM upper bound (perfect per-step verifier)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_oracle_em(
    model: NonlinearCandidateMGUModel,
    inputs: torch.Tensor,
    true_states: np.ndarray,
    write_mask: np.ndarray,
    final_true: np.ndarray,
    mech_id: int,
    mech: Dict[str, Any],
    stats: Dict[str, torch.Tensor],
    k: int,
    commit_mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Simulates a straight-line trajectory with a perfect per-step candidate verifier.

    At every transition the K candidates are scored against the TRUE next register
    state (oracle). The first fully-correct candidate is committed (deterministic
    fallback otherwise). Generation never uses ground truth. Selection does, which
    is exactly what makes this an upper bound under a perfect selector.
    """
    core = model.cognition_core
    head = model.decision_head
    batch_size, seq_len, _ = inputs.shape
    true_t = torch.from_numpy(true_states).to(device)
    write_t = torch.from_numpy(write_mask).to(device)

    state = torch.zeros(batch_size, model.state_dim, device=device)
    n_oracle_fixes = np.zeros(batch_size, dtype=np.int64)
    oracle_fix_steps: List[List[int]] = [[] for _ in range(batch_size)]

    for t in range(seq_len):
        e_t = model.encoder(inputs[:, t, :])
        f_t = torch.sigmoid(core.w_f_ih(e_t) + core.w_f_hh(state))
        h_t = core.cand_mlp[1](core.cand_mlp[0](torch.cat([f_t * state, e_t], dim=-1)))

        generator = torch.Generator(device="cpu").manual_seed(NOISE_SEED_EM + 1000 * mech_id + t)
        candidates = sample_candidates(core, state, f_t, h_t, mech, stats, generator, device)
        branch_states = (1.0 - f_t)[:, None, :] * state[:, None, :] + f_t[:, None, :] * candidates

        if k == 1:
            idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            dec = head(branch_states[:, :k].reshape(batch_size * k, -1)).view(batch_size, k, 4, 10).argmax(-1)
            eq = dec == true_t[:, t][:, None, :]
            if commit_mode == "target":
                eq = eq | (~write_t[:, t][:, None, :])
            ok = eq.all(-1)
            idx = ok.int().argmax(dim=-1)  # first correct candidate, else 0 (deterministic)

        chosen = branch_states.gather(
            1, idx[:, None, None].expand(-1, 1, model.state_dim)
        ).squeeze(1)
        fixed = (idx != 0).cpu().numpy()
        n_oracle_fixes += fixed
        for b in np.where(fixed)[0]:
            oracle_fix_steps[int(b)].append(t + 1)
        state = chosen

    final_pred = head(state).argmax(-1).cpu().numpy()
    recovered = (final_pred == final_true).all(-1)
    return {
        "recovered": recovered,
        "n_oracle_fixes": n_oracle_fixes,
        "oracle_fix_steps": oracle_fix_steps,
    }


@torch.no_grad()
def run_oracle_oneshot_repair(
    model: NonlinearCandidateMGUModel,
    inputs: torch.Tensor,
    true_states: np.ndarray,
    write_mask: np.ndarray,
    final_true: np.ndarray,
    mech_id: int,
    mech: Dict[str, Any],
    stats: Dict[str, torch.Tensor],
    k: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Deterministic trajectory with at most ONE applied oracle repair.

    While no repair has been applied, at every failing transition K candidates are
    sampled once and, if one is fully correct, it is committed (oracle); from then on
    the trajectory runs deterministically (no further sampling). Failed attempts do not
    alter the deterministic trajectory. This separates "a repaired state is a genuinely
    good computational state" from "the oracle can re-roll the candidate lottery at
    every step".
    """
    core = model.cognition_core
    head = model.decision_head
    batch_size, seq_len, _ = inputs.shape
    true_t = torch.from_numpy(true_states).to(device)

    state = torch.zeros(batch_size, model.state_dim, device=device)
    fixed = np.zeros(batch_size, dtype=bool)

    for t in range(seq_len):
        e_t = model.encoder(inputs[:, t, :])
        f_t = torch.sigmoid(core.w_f_ih(e_t) + core.w_f_hh(state))
        h_t = core.cand_mlp[1](core.cand_mlp[0](torch.cat([f_t * state, e_t], dim=-1)))
        c0 = torch.tanh(core.cand_mlp[2](h_t))
        det_branch = (1.0 - f_t) * state + f_t * c0

        dec0 = head(det_branch).argmax(-1)
        ok0 = (dec0 == true_t[:, t]).all(-1).cpu().numpy()
        attempt = (~ok0) & (~fixed) & (k > 1)

        if attempt.any():
            generator = torch.Generator(device="cpu").manual_seed(
                NOISE_SEED_EM + 5000 + 1000 * mech_id + t
            )
            candidates = sample_candidates(core, state, f_t, h_t, mech, stats, generator, device)
            branch_states = (1.0 - f_t)[:, None, :] * state[:, None, :] + f_t[:, None, :] * candidates
            dec = head(branch_states[:, :k].reshape(batch_size * k, -1)).view(batch_size, k, 4, 10).argmax(-1)
            ok = (dec == true_t[:, t][:, None, :]).all(-1)
            idx = ok.int().argmax(dim=-1)
            chosen_fix = branch_states.gather(
                1, idx[:, None, None].expand(-1, 1, model.state_dim)
            ).squeeze(1)
            use_fix = torch.from_numpy(attempt & ok.any(-1).cpu().numpy()).to(device)
            state = torch.where(use_fix[:, None], chosen_fix, det_branch)
            fixed = fixed | use_fix.cpu().numpy()
        else:
            state = det_branch

    final_pred = head(state).argmax(-1).cpu().numpy()
    recovered = (final_pred == final_true).all(-1)
    return {"recovered": recovered, "fixed": fixed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle Branching Feasibility Experiment")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    device = torch.device(args.device)
    started = time.time()

    print("=" * 88)
    print("RE:REASON — FINAL PHASE-1 EXPERIMENT: ORACLE BRANCHING FEASIBILITY")
    print("=" * 88)
    print(f"Device: {device} | Checkpoint: {CHECKPOINT} (frozen, no retraining)")

    model = load_frozen_model(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Frozen parameters: {param_count:,}")

    # Exact test split protocol used by the Round-2 audit / parameterization isolation.
    task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10, seed=42)
    split_mgr = DisjointSplitManager(task=task, base_seed=42)
    test_set = split_mgr.test_set
    batch = test_set.get_batch(len(test_set), idx=0)
    inputs = batch.inputs.to(device)
    final_targets = batch.targets.to(device)
    step_targets = batch.metadata["step_targets"].to(device)
    n_seq = inputs.shape[0]
    n_trans = n_seq * inputs.shape[1]
    print(f"Test split: {n_seq} sequences x {inputs.shape[1]} transitions = {n_trans}")

    # ---- Deterministic pass -------------------------------------------------
    det = deterministic_pass(model, inputs)
    with torch.no_grad():
        det_step_logits = torch.stack(
            [model.decision_head(s) for s in det["states"]], dim=1
        )  # (B, T, 4, 10)
        ref_logits = model(inputs).logits
        max_logit_diff = float((det_step_logits[:, -1] - ref_logits).abs().max().item())
        det_preds = det_step_logits.argmax(-1).cpu().numpy().astype(np.int64)  # (B, T, 4)
    assert max_logit_diff < 1e-5, f"Deterministic recomputation mismatch: {max_logit_diff}"

    meta = build_ground_truth_meta(inputs, step_targets, final_targets, det_preds)
    det_terminal_fail = meta["terminal_fail"]
    det_em = float(1.0 - det_terminal_fail.mean())
    det_step_full_ok = float(meta["det_full_ok"].mean())
    det_step_tgt_ok = float(meta["det_tgt_ok"].mean())
    print(f"Deterministic K=1 test EM: {det_em:.4f} "
          f"({int((~det_terminal_fail).sum())}/{n_seq})")
    print(f"Deterministic per-transition: full-state ok={det_step_full_ok:.4f}, "
          f"target-register ok={det_step_tgt_ok:.4f}")

    # Pooled statistics for truth-free noise scaling.
    h_stack = torch.cat(det["hiddens"], dim=0)
    c_stack = torch.cat(det["candidates"], dim=0)
    stats = {"h_std": h_stack.std(dim=0), "c_std": c_stack.std(dim=0)}

    # ---- EXP 1-7: local passes ----------------------------------------------
    local_results: Dict[str, Any] = {}
    mechanism_scale_summary: Dict[str, float] = {}
    reference_disp = None

    ordered = sorted(MECHANISMS, key=lambda item: item[0])
    for mech_id, name, mech in ordered:
        if name == "rand_S_cand_matched" and reference_disp is None:
            raise RuntimeError("Matched control requires the primary mechanism first")
        mech = dict(mech)
        if mech.get("matched"):
            per_dim_energy = float(torch.sqrt((stats["c_std"] ** 2).sum()).item())
            mech["scale"] = reference_disp / max(per_dim_energy, 1e-12)
        print(f"\n[local] mechanism={name}: scale={mech.get('scale', mech.get('p'))}")
        res = run_local_pass(model, det, mech_id, mech, stats, device)
        agg = aggregate_local(res, meta)
        agg["mechanism_scale"] = float(mech.get("scale", mech.get("p", 0.0)))
        local_results[name] = agg
        if name == PRIMARY_MECHANISM:
            reference_disp = agg["mean_displacement_norm_primary_draws"]
        if mech.get("matched"):
            matched_scale = mech["scale"]
            mechanism_scale_summary["rand_scale_matched"] = matched_scale
            local_results[name]["matched_target_displacement_norm"] = reference_disp

    # ---- EXP 8: oracle EM upper bound ---------------------------------------
    print("\n[oracle EM] running perfect-verifier trajectory simulations...")
    mech_by_name = {name: (mid, dict(cfg)) for mid, name, cfg in MECHANISMS}
    for mid, name, cfg in MECHANISMS:
        if cfg.get("matched"):
            per_dim_energy = float(torch.sqrt((stats["c_std"] ** 2).sum()).item())
            cfg = dict(cfg)
            cfg["scale"] = reference_disp / max(per_dim_energy, 1e-12)
            mech_by_name[name] = (mid, cfg)

    primary_k1_em = None
    em_fulls: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def em_pass(name: str, k: int, commit_mode: str = "full") -> Dict[str, Any]:
        mid, cfg = mech_by_name[name]
        return run_oracle_em(
            model, inputs, meta["true_states"], meta["write_mask"], meta["final_true"],
            mid, cfg, stats, k, commit_mode, device,
        )

    for name in [PRIMARY_MECHANISM, "rand_S_cand_matched"]:
        em_fulls[name] = {}
        for k in K_VALUES:
            out = em_pass(name, k, "full")
            recovered = out["recovered"]
            k1_fail = det_terminal_fail
            recovered_failures = int((recovered & k1_fail).sum())
            total_failures = int(k1_fail.sum())
            em_fulls[name][k] = {
                "oracle_em": float(recovered.mean()),
                "deterministic_failures": total_failures,
                "oracle_recoverable_failures": recovered_failures,
                "oracle_recoverable_failure_rate": (
                    recovered_failures / total_failures if total_failures else 0.0
                ),
                "lost_successes": int(((~k1_fail) & (~recovered)).sum()),
                "mean_oracle_fixes_per_sequence": float(out["n_oracle_fixes"].mean()),
                "sequences_with_at_least_one_fix": int((out["n_oracle_fixes"] > 0).sum()),
            }
            if k == 1 and name == PRIMARY_MECHANISM:
                primary_k1_em = float(recovered.mean())
                assert abs(primary_k1_em - det_em) < 1e-12, "Oracle K=1 must equal deterministic EM"

    # Sensitivity: all other mechanisms at K=8 (full commit).
    em_other: Dict[str, Any] = {}
    for mid, name, cfg in MECHANISMS:
        if name in (PRIMARY_MECHANISM, "rand_S_cand_matched"):
            continue
        out = em_pass(name, 8, "full")
        recovered = out["recovered"]
        recovered_failures = int((recovered & det_terminal_fail).sum())
        em_other[name] = {
            "oracle_em_K8": float(recovered.mean()),
            "oracle_recoverable_failures": recovered_failures,
            "oracle_recoverable_failure_rate": (
                recovered_failures / max(int(det_terminal_fail.sum()), 1)
            ),
        }

    # Target-register commit variant (weaker verifier) for the primary mechanism.
    em_target_commit: Dict[int, Any] = {}
    for k in K_VALUES:
        out = em_pass(PRIMARY_MECHANISM, k, "target")
        recovered = out["recovered"]
        em_target_commit[k] = {
            "oracle_em": float(recovered.mean()),
            "oracle_recoverable_failures": int((recovered & det_terminal_fail).sum()),
        }

    # One-shot repair: a single oracle fix at the first failing transition, then
    # fully deterministic continuation (no per-step re-lottery).
    oneshot: Dict[str, Any] = {}
    for k in K_VALUES:
        out = run_oracle_oneshot_repair(
            model, inputs, meta["true_states"], meta["write_mask"], meta["final_true"],
            0, mech_by_name[PRIMARY_MECHANISM][1], stats, k, device,
        )
        recovered = out["recovered"]
        oneshot[str(k)] = {
            "oracle_em_oneshot": float(recovered.mean()),
            "sequences_fixed": int(out["fixed"].sum()),
            "recovered_failures": int((recovered & det_terminal_fail).sum()),
            "lost_successes": int(((~det_terminal_fail) & (~recovered)).sum()),
        }
    assert abs(oneshot["1"]["oracle_em_oneshot"] - det_em) < 1e-12, "One-shot K=1 must equal deterministic EM"

    # EM-based end-to-end recovery per first-error timestep (primary K=8).
    primary_k8 = em_pass(PRIMARY_MECHANISM, 8, "full")["recovered"]
    em_recovery_by_first_error: Dict[str, Any] = {}
    for t in range(1, 11):
        sel = np.where(det_terminal_fail & (meta["first_error_t"] == t))[0]
        n = int(sel.size)
        em_recovery_by_first_error[str(t)] = {
            "failed_trajectories": n,
            "em_recovered_under_oracle_K8": int(primary_k8[sel].sum()) if n else 0,
        }

    # ---- ADD wrap-around confusion (primary mechanism, deterministic failures) --
    res_primary = local_results[PRIMARY_MECHANISM]["per_K"]["8"]
    wrap_fail_mask = (~meta["det_tgt_ok"]) & (meta["op_cat"] == 3)
    confusions: Dict[str, int] = {}
    for b, t in zip(*np.where(wrap_fail_mask)):
        wr = int(np.argmax(meta["write_mask"][b, t]))
        pair = f"{int(meta['true_states'][b, t, wr])}->{int(det_preds[b, t, wr])}"
        confusions[pair] = confusions.get(pair, 0) + 1
    top_confusions = dict(sorted(confusions.items(), key=lambda kv: -kv[1])[:10])

    # ---- Assemble payload ----------------------------------------------------
    payload = {
        "metadata": {
            "date": "2026-10-05",
            "scope": "Oracle branching feasibility — candidate-generation stochasticity only",
            "checkpoint": CHECKPOINT,
            "checkpoint_frozen": True,
            "test_split": {
                "manager": "DisjointSplitManager(RegisterTrackingTask(seed=42), base_seed=42)",
                "sequences": n_seq,
                "transitions": n_trans,
            },
            "primary_mechanism": PRIMARY_MECHANISM,
            "candidate_construction": (
                "C_0 deterministic; C_1..C_7 stochastic draws, nested across K; "
                "branch state B_k = (1-f)*S_(t-1) + f*C_k; decoding via the model's own head"
            ),
            "truth_usage": "evaluation/scoring only (oracle verifier); never generation",
            "deterministic": {
                "terminal_em": det_em,
                "per_transition_full_state_ok": det_step_full_ok,
                "per_transition_target_register_ok": det_step_tgt_ok,
                "terminal_failures": int(det_terminal_fail.sum()),
                "no_step_error_but_terminal_fail": meta["no_step_error_but_terminal_fail"],
                "logit_recompute_max_abs_diff": max_logit_diff,
            },
            "failure_anatomy": meta["failure_anatomy"],
            "noise_stats": {
                "h_std_mean": float(stats["h_std"].mean()),
                "c_std_mean": float(stats["c_std"].mean()),
                "primary_mean_displacement_norm": reference_disp,
                "matched_control_scale": mechanism_scale_summary.get("rand_scale_matched"),
            },
            "runtime_seconds": None,
            "device": str(device),
        },
        "local": local_results,
        "oracle_em": {
            "full_commit": {str(k): v for k, v in em_fulls[PRIMARY_MECHANISM].items()},
            "full_commit_control_matched": {str(k): v for k, v in em_fulls["rand_S_cand_matched"].items()},
            "full_commit_other_mechanisms_K8": em_other,
            "target_commit_primary": {str(k): v for k, v in em_target_commit.items()},
            "oneshot_repair_primary": oneshot,
            "em_recovery_by_first_error_timestep_K8_primary": em_recovery_by_first_error,
        },
        "add_wrap_confusion_top10_deterministic": top_confusions,
    }
    payload["metadata"]["runtime_seconds"] = time.time() - started

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {args.output}")

    # ---- Console tables ------------------------------------------------------
    prim = local_results[PRIMARY_MECHANISM]["per_K"]
    anatomy = meta["failure_anatomy"]
    print("\nFAILURE ANATOMY (deterministic model)")
    print(f"  per-transition full-state failures: {anatomy['per_transition_full_state_failures']} "
          f"(target-register failures {anatomy['per_transition_target_register_failures']}, "
          f"ghost-only readout mismatches {anatomy['per_transition_ghost_only_failures']})")
    print(f"  terminal failures: {anatomy['terminal_failures']} "
          f"(with >=1 real target error {anatomy['terminal_failures_with_real_target_error']}, "
          f"ghost-only {anatomy['terminal_failures_ghost_only']})")

    print("\nTABLE 1 — Primary mechanism (Gaussian hidden-activation noise, c=1.0)")
    print("| K | Candidate diversity | Any correct target register | Any correct full state | Oracle recoverable failures | Oracle EM upper bound |")
    print("|---|---:|---:|---:|---:|---:|")
    for k in K_VALUES:
        e = prim[str(k)]
        div = e["diversity"]
        div_str = (
            "0.000 / cos=1.000" if k == 1
            else f"L2={div['mean_pairwise_l2_candidates']:.3f} / cos={div['mean_pairwise_cos_candidates']:.3f}"
        )
        em = em_fulls[PRIMARY_MECHANISM][k]
        recov = f"{em['oracle_recoverable_failures']}/{em['deterministic_failures']} ({em['oracle_recoverable_failure_rate']:.1%})"
        print(
            f"| {k} | {div_str} | {e['any_target_register_rate_all_transitions']:.4f} | "
            f"{e['any_full_state_rate_all_transitions']:.4f} | {recov} | {em['oracle_em']:.4f} |"
        )

    print("\nTABLE 1b — CONDITIONED ON REAL (target-register) FAILURES "
          f"({anatomy['per_transition_target_register_failures']} transitions)")
    print("| K | any candidate target ok | any candidate full state ok | oracle EM (trajectory) |")
    print("|---|---:|---:|---:|")
    for k in K_VALUES:
        e = prim[str(k)]
        em = em_fulls[PRIMARY_MECHANISM][k]
        print(f"| {k} | {e['any_target_register_rate_on_det_target_failures']:.4f} | "
              f"{e['any_full_state_rate_on_det_target_failures']:.4f} | {em['oracle_em']:.4f} |")

    print("\nTABLE 2 — Failure-type breakdown (K=8, deterministic REAL failures: target register wrong)")
    print("| Failure Type | Count | % of failures | Branch recoverable | Non-recoverable |")
    print("|---|---:|---:|---:|---:|")
    by_op_t = prim["8"]["by_operation_target_failures"]
    total_fail = sum(v["failures"] for v in by_op_t.values())
    for cat in ["SET", "ADD_no_wrap", "ADD_wrap", "SWAP"]:
        v = by_op_t[cat]
        rec = v["recoverable_full_state"]
        print(
            f"| {cat} | {v['failures']} | {v['failures'] / max(total_fail, 1):.1%} | "
            f"{rec} ({rec / max(v['failures'], 1):.1%}) | {v['failures'] - rec} "
            f"({(v['failures'] - rec) / max(v['failures'], 1):.1%}) |"
        )
    print(f"| (Other) | 0 | 0.0% | 0 | 0 |")
    print(f"| **Total** | **{total_fail}** | **100%** | "
          f"**{sum(v['recoverable_full_state'] for v in by_op_t.values())}** | "
          f"**{total_fail - sum(v['recoverable_full_state'] for v in by_op_t.values())}** |")
    print("  (partial fixes, target register corrected but another register wrong, "
          "target-register-recoverable column:)")
    for cat in ["SET", "ADD_no_wrap", "ADD_wrap", "SWAP"]:
        v = by_op_t[cat]
        print(f"    {cat:<12} {v['recoverable_target_register']}/{v['failures']}")

    print("\nTABLE 3 — First-error timestep (K=8, terminally failed trajectories)")
    print("| First error timestep | Failures | Branch recoverable | Recovery % |")
    print("|---|---:|---:|---:|")
    fe = prim["8"]["by_first_target_error_timestep"]
    for t in range(1, 11):
        v = fe[str(t)]
        n = v["failed_trajectories"]
        rec = v["first_error_transition_recoverable_full_state"]
        print(f"| {t} | {n} | {rec} | {(rec / n if n else 0.0):.1%} |")
    print("  (first-error defined by a real target-register error; ghost readout mismatches excluded)")
    fe_strict = prim["8"]["by_first_error_timestep"]
    strict_rec = sum(v["first_error_transition_recoverable_full_state"] for v in fe_strict.values())
    strict_n = sum(v["failed_trajectories"] for v in fe_strict.values())
    print(f"  strict any-register first error (includes ghosts): {strict_n} trajectories, "
          f"{strict_rec} recoverable at that step ({strict_rec / max(strict_n, 1):.1%})")

    print("\nTABLE 4 — One-shot repair (single oracle fix at first failure, then deterministic)")
    print("| K | fixed sequences | recovered failures | lost successes | oracle EM (one-shot) |")
    print("|---|---:|---:|---:|---:|")
    for k in K_VALUES:
        v = oneshot[str(k)]
        print(f"| {k} | {v['sequences_fixed']} | {v['recovered_failures']}/{int(det_terminal_fail.sum())} | "
              f"{v['lost_successes']} | {v['oracle_em_oneshot']:.4f} |")
    print(f"  (full per-step oracle EM was {em_fulls[PRIMARY_MECHANISM][8]['oracle_em']:.4f} at K=8)")

    print("\nLOTTERY DIAGNOSTIC — per-candidate-index written-register accuracy on REAL failures")
    pci = local_results[PRIMARY_MECHANISM]["per_candidate_index"]
    print("  branch state :", [round(x, 4) for x in pci["branch_target_ok_on_det_target_failures"]])
    print("  raw candidate:", [round(x, 4) for x in pci["raw_candidate_target_ok_on_det_target_failures"]])
    print("  (index 0 = deterministic; indices 1-7 = stochastic draws; chance = 0.10 per register value)")

    print("\nCONTROL — matched-norm random S_candidate perturbation (EXP 7)")
    ctrl = local_results["rand_S_cand_matched"]["per_K"]
    print("| K | any target reg | any full state | oracle EM | recoverable failures |")
    print("|---|---:|---:|---:|---:|")
    for k in K_VALUES:
        e = ctrl[str(k)]
        em = em_fulls["rand_S_cand_matched"][k]
        print(f"| {k} | {e['any_target_register_rate_all_transitions']:.4f} | "
              f"{e['any_full_state_rate_all_transitions']:.4f} | {em['oracle_em']:.4f} | "
              f"{em['oracle_recoverable_failures']}/{em['deterministic_failures']} |")

    print("\nSENSITIVITY — oracle EM at K=8, other mechanisms")
    for name, v in em_other.items():
        print(f"  {name:<24} oracle_EM={v['oracle_em_K8']:.4f} "
              f"recovered_failures={v['oracle_recoverable_failures']} "
              f"({v['oracle_recoverable_failure_rate']:.1%})")
    print(f"  {'target-commit (K=8)':<24} oracle_EM={em_target_commit[8]['oracle_em']:.4f} "
          f"recovered_failures={em_target_commit[8]['oracle_recoverable_failures']}")

    print("\nADD wrap-around (a+b>=10): deterministic target-register failures, top confusions (true->pred):")
    print(f"  {top_confusions if top_confusions else 'none'}")
    print(f"  of {int(wrap_fail_mask.sum())} deterministic target-register failures, "
          f"K=8 candidates recover the target register on "
          f"{by_op_t['ADD_wrap']['recoverable_target_register']} transitions and the full state on "
          f"{by_op_t['ADD_wrap']['recoverable_full_state']} transitions")

    print(f"\nTotal runtime: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
