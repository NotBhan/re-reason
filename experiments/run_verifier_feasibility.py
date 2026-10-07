#!/usr/bin/env python3
"""RE:Reason Phase-2 Verifier Feasibility Experiment.

Question: can a learned selector/verifier recognize a good latent transition
before committing it, using ONLY inference-available information?

Frozen components (no weight updates): recurrent core, observation encoder,
candidate network, gate network, terminal readout. Candidate sets are generated
with the same stochastic mechanism as the branching-feasibility experiment
(Gaussian activation noise on the candidate MLP hidden activation; candidate
index 0 is the deterministic candidate).

Labels (VALID / INVALID = candidate matches the required next register state
under the benchmark transition semantics) are used ONLY to construct
train/validation targets and to score metrics. Features NEVER contain ground
truth, future observations, or future targets.

K = 4 primary; K = 8 secondary (ranking only). Strict disjoint procedural
splits: verifier train (seed stream 2042), verifier validation (stream 51042),
test = the frozen model's untouched test split (stream 90042, 1500 episodes).

Deliverables: this runner, experiments/verifier_feasibility_results.json,
figures under docs/reports/figures/.
"""

import argparse
import json
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_branching_feasibility as bf  # noqa: E402  (frozen-model utilities)

from re_reason.benchmarks.dataset_split import DisjointSplitManager  # noqa: E402
from re_reason.benchmarks.register_swap import RegisterTrackingTask  # noqa: E402


CHECKPOINT = "experiments/checkpoints/6_nonlinear_candidate_mgu_seed42_3000steps.pt"
OUT_JSON = "experiments/verifier_feasibility_results.json"
FIG_DIR = "docs/reports/figures"

K_PRIMARY = 4
K_SECONDARY = 8
TAU_DEFAULT = 0.5

FEATURE_ORDER = [
    "cand", "prev", "gate", "instr", "baseline", "delta_prev", "delta_base",
    "dists", "dec_c", "dec_0", "cand_norm",
]

ABLATIONS: Dict[str, List[str]] = {
    "A_candidate_only": ["cand"],
    "B_candidate_plus_prev": ["cand", "prev"],
    "C_candidate_plus_gate": ["cand", "gate"],
    "D_candidate_plus_instr": ["cand", "instr"],
    "E_candidate_plus_decode": ["cand", "dec_c"],
    "F_all": ["cand", "prev", "gate", "instr", "baseline", "delta_prev",
              "delta_base", "dists", "dec_c", "dec_0"],
    "diag_norms_only": ["dists"],
    "diag_direction_only": ["cand_norm"],
}

GENERATION_SEEDS = {"train": 31001, "val": 33001, "test": 35001}
GEN_MECH = {"kind": "gauss_hidden", "scale": 1.0}


# ---------------------------------------------------------------------------
# Frozen model / splits
# ---------------------------------------------------------------------------

def load_frozen_model(device: torch.device):
    model = bf.load_frozen_model(device)
    return model


def make_splits():
    """Verifier train/val streams are disjoint from every stream the checkpoint saw."""
    task = RegisterTrackingTask(num_registers=4, sequence_length=10, num_values=10, seed=42)
    train_mgr = DisjointSplitManager(task, train_size=1200, val_size=500, test_size=10, base_seed=1042)
    test_mgr = DisjointSplitManager(task, train_size=10, val_size=10, test_size=1500, base_seed=42)
    return train_mgr.train_set, train_mgr.val_set, test_mgr.test_set


def prepare_split(model, split, device):
    batch = split.get_batch(len(split), idx=0)
    inputs = batch.inputs.to(device)
    final_targets = batch.targets.to(device)
    step_targets = batch.metadata["step_targets"].to(device)
    det = bf.deterministic_pass(model, inputs)
    with torch.no_grad():
        det_preds = torch.stack(
            [model.decision_head(s) for s in det["states"]], dim=1
        ).argmax(-1).cpu().numpy().astype(np.int64)
    meta = bf.build_ground_truth_meta(inputs, step_targets, final_targets, det_preds)
    meta["final_targets"] = final_targets
    return {"inputs": inputs, "det": det, "meta": meta, "det_preds": det_preds}


def compute_noise_stats(model, prepared_train, K: int):
    """Truth-free noise scaling, estimated from the verifier-train split only."""
    det = prepared_train["det"]
    h_stack = torch.cat(det["hiddens"], dim=0)
    c_stack = torch.cat(det["candidates"], dim=0)
    stats = {"h_std": h_stack.std(dim=0), "c_std": c_stack.std(dim=0)}

    disp_total, n_pairs = 0.0, 0
    B, T, _ = prepared_train["inputs"].shape
    for t in range(T):
        s_prev = torch.zeros(B, model.state_dim, device=prepared_train["inputs"].device) if t == 0 else det["states"][t - 1]
        gen = torch.Generator(device="cpu").manual_seed(GENERATION_SEEDS["train"] + t)
        C = bf.sample_candidates(
            model.cognition_core, s_prev, det["gates"][t], det["hiddens"][t], GEN_MECH, stats, gen,
            prepared_train["inputs"].device,
        )[:, :K]
        disp_total += float((C[:, 1:, :] - C[:, :1, :]).norm(dim=-1).sum().item())
        n_pairs += B * (K - 1)
    reference_disp = disp_total / max(n_pairs, 1)
    per_dim_energy = float(torch.sqrt((stats["c_std"] ** 2).sum()).item())
    noise_mech = {"kind": "rand_cand", "scale": reference_disp / max(per_dim_energy, 1e-12)}
    return stats, GEN_MECH, noise_mech, reference_disp


# ---------------------------------------------------------------------------
# Candidate generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_single_draw(core, h_t, sigma_h, generator, device):
    """One noisy candidate per transition (1x candidate compute; used by the ensemble control)."""
    eps = torch.randn(h_t.shape, generator=generator).to(device)
    h_pert = h_t + sigma_h * eps
    return torch.tanh(F.linear(h_pert, core.cand_mlp[2].weight, core.cand_mlp[2].bias))


# ---------------------------------------------------------------------------
# Inference-only features
# ---------------------------------------------------------------------------

@torch.no_grad()
def step_feature_groups(model, x_t, e_t, s_prev, f_t, C):
    """Feature groups for K candidates of one transition. No ground truth anywhere."""
    B, K, D = C.shape
    head = model.decision_head
    C0 = C[:, :1, :]

    dec = head(C.reshape(B * K, D)).view(B, K, 4, 10)
    probs = dec.softmax(-1)
    dec0 = head(C0.reshape(B, D)).view(B, 1, 4, 10)
    probs0 = dec0.softmax(-1)

    eps = 1e-6
    c_norm = C.norm(dim=-1)
    s_norm = s_prev.norm(dim=-1)[:, None]
    b0_norm = C0.norm(dim=-1)

    groups: Dict[str, torch.Tensor] = {}
    groups["cand"] = C
    groups["prev"] = s_prev[:, None, :].expand(B, K, D)
    groups["gate"] = f_t[:, None, :].expand(B, K, D)
    groups["instr"] = torch.cat([e_t, x_t], dim=-1)[:, None, :].expand(B, K, e_t.shape[-1] + x_t.shape[-1])
    groups["baseline"] = C0.expand(B, K, D)
    groups["delta_prev"] = C - s_prev[:, None, :]
    groups["delta_base"] = C - C0

    d_prev = groups["delta_prev"].norm(dim=-1)
    d_base = groups["delta_base"].norm(dim=-1)
    cos_prev = (C * s_prev[:, None, :]).sum(-1) / (c_norm * s_norm + eps)
    cos_base = (C * C0).sum(-1) / (c_norm * b0_norm + eps)
    groups["dists"] = torch.stack(
        [c_norm, d_prev, d_base, cos_prev, cos_base, C.std(dim=-1)], dim=-1
    )

    maxp = probs.max(-1).values
    ent = -(probs.clamp_min(1e-9).log() * probs).sum(-1)
    top2 = probs.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    groups["dec_c"] = torch.cat([dec.flatten(2), probs.flatten(2), maxp, ent, margin], dim=-1)

    p0 = probs0.squeeze(1)
    maxp0 = p0.max(-1).values
    top20 = p0.topk(2, dim=-1).values
    dec0_feat = torch.cat([p0.flatten(1), maxp0, top20[..., 0] - top20[..., 1]], dim=-1)
    groups["dec_0"] = dec0_feat[:, None, :].expand(B, K, dec0_feat.shape[-1])

    groups["cand_norm"] = C / (c_norm[..., None] + eps)
    return groups


def feature_dim_map(groups: Dict[str, torch.Tensor]) -> Dict[str, int]:
    return {name: int(groups[name].shape[-1]) for name in FEATURE_ORDER}


def column_index(group_slices: Dict[str, Tuple[int, int]], names: List[str]) -> np.ndarray:
    return np.concatenate([np.arange(*group_slices[n]) for n in names]).astype(np.int64)


def branch_states(s_prev, f_t, C):
    """The state a branch would commit: B_k = (1 - f) * S_prev + f * C_k."""
    return (1.0 - f_t)[:, None, :] * s_prev[:, None, :] + f_t[:, None, :] * C


def decode_labels(model, states, true_t, write_mask):
    """VALID/INVALID labels from decoding the *committed* branch states.

    Full-state validity is the primary label; target-register validity is secondary.
    """
    B, K, D = states.shape
    dec = model.decision_head(states.reshape(B * K, D)).view(B, K, 4, 10).argmax(-1)
    if not isinstance(true_t, torch.Tensor):
        true_t = torch.from_numpy(np.asarray(true_t)).to(dec.device)
    if not isinstance(write_mask, torch.Tensor):
        write_mask = torch.from_numpy(np.asarray(write_mask)).to(dec.device)
    full = (dec == true_t[:, None, :]).all(-1)
    tgt = ((dec == true_t[:, None, :]) | (~write_mask)[:, None, :]).all(-1)
    return full, tgt


# ---------------------------------------------------------------------------
# Extraction / chunked scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_split(model, prepared, K, stats, mech, seed_base, device):
    """Builds the full feature matrix + labels for a split (used for train/val)."""
    inputs, det, meta = prepared["inputs"], prepared["det"], prepared["meta"]
    B, T, _ = inputs.shape
    rows = []
    labels_full = np.zeros((T, B, K), dtype=bool)
    labels_tgt = np.zeros((T, B, K), dtype=bool)
    for t in range(T):
        s_prev = torch.zeros(B, model.state_dim, device=device) if t == 0 else det["states"][t - 1]
        gen = torch.Generator(device="cpu").manual_seed(seed_base + t)
        C = bf.sample_candidates(
            model.cognition_core, s_prev, det["gates"][t], det["hiddens"][t], mech, stats, gen, device
        )[:, :K]
        groups = step_feature_groups(model, inputs[:, t], det["embeddings"][t], s_prev, det["gates"][t], C)
        rows.append(torch.cat([groups[n] for n in FEATURE_ORDER], -1).reshape(B * K, -1).float().cpu().numpy())
        B_states = branch_states(s_prev, det["gates"][t], C)
        lf, lt = decode_labels(model, B_states, meta["true_states"][:, t, :], meta["write_mask"][:, t, :])
        labels_full[t] = lf.cpu().numpy()
        labels_tgt[t] = lt.cpu().numpy()
    X = np.concatenate(rows, axis=0)
    return X, labels_full.reshape(-1), labels_tgt.reshape(-1), labels_full, labels_tgt


@torch.no_grad()
def score_split(bundles: Dict[str, "Bundle"], model, prepared, K, stats, mech, seed_base, device):
    """Scores every bundle on a split without materializing the full feature matrix."""
    inputs, det, meta = prepared["inputs"], prepared["det"], prepared["meta"]
    B, T, _ = inputs.shape
    scores = {name: np.zeros((T, B, K), dtype=np.float32) for name in bundles}
    labels_full = np.zeros((T, B, K), dtype=bool)
    labels_tgt = np.zeros((T, B, K), dtype=bool)
    for t in range(T):
        s_prev = torch.zeros(B, model.state_dim, device=device) if t == 0 else det["states"][t - 1]
        gen = torch.Generator(device="cpu").manual_seed(seed_base + t)
        C = bf.sample_candidates(
            model.cognition_core, s_prev, det["gates"][t], det["hiddens"][t], mech, stats, gen, device
        )[:, :K]
        groups = step_feature_groups(model, inputs[:, t], det["embeddings"][t], s_prev, det["gates"][t], C)
        X = torch.cat([groups[n] for n in FEATURE_ORDER], -1).reshape(B * K, -1)
        for name, b in bundles.items():
            sel = X[:, b.col_idx_t]
            sel = (sel - b.mu) / b.sd
            scores[name][t] = b.module(sel).sigmoid().squeeze(-1).view(B, K).cpu().numpy()
        B_states = branch_states(s_prev, det["gates"][t], C)
        lf, lt = decode_labels(model, B_states, meta["true_states"][:, t, :], meta["write_mask"][:, t, :])
        labels_full[t] = lf.cpu().numpy()
        labels_tgt[t] = lt.cpu().numpy()
    return scores, labels_full, labels_tgt


# ---------------------------------------------------------------------------
# Selector models
# ---------------------------------------------------------------------------

class Bundle:
    def __init__(self, name, kind, groups, module, mu, sd, col_idx):
        self.name = name
        self.kind = kind
        self.groups = groups
        self.module = module
        self.mu = mu
        self.sd = sd
        self.col_idx = col_idx
        self.col_idx_t = torch.from_numpy(col_idx)

    def score_groups(self, groups: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Score candidates from live feature groups -> (B, K) probabilities."""
        B, K, _ = groups["cand"].shape
        X = torch.cat([groups[n] for n in FEATURE_ORDER], -1).reshape(B * K, -1)
        sel = X[:, self.col_idx_t]
        sel = (sel - self.mu) / self.sd
        return self.module(sel).sigmoid().squeeze(-1).view(B, K)


def standardize_stats(X: np.ndarray, col_idx: np.ndarray):
    mu = torch.from_numpy(X[:, col_idx].mean(axis=0)).float()
    sd = torch.from_numpy(X[:, col_idx].std(axis=0)).float()
    sd = torch.clamp(sd, min=1e-6)
    return mu, sd


def build_module(kind: str, dim: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(dim, 1)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
        )
    raise ValueError(kind)


def train_selector(kind, Xtr, ytr, Xval, yval, col_idx, name, seed=1234,
                   epochs=30, batch=4096, patience=6, device="cpu"):
    torch.manual_seed(seed)
    mu, sd = standardize_stats(Xtr, col_idx)
    Xtr_t = torch.from_numpy(Xtr[:, col_idx]).float()
    Xval_t = torch.from_numpy(Xval[:, col_idx]).float()
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    yval_t = torch.from_numpy(yval.astype(np.float32))

    def prep(X):
        return (X.to(device) - mu.to(device)) / sd.to(device)

    lr = 5e-3 if kind == "linear" else 1e-3
    module = build_module(kind, len(col_idx)).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state, best_auroc, best_epoch = None, -1.0, -1
    history = []
    for epoch in range(1, epochs + 1):
        module.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(Xtr_t), batch):
            idx = perm[i:i + batch]
            xb = prep(Xtr_t[idx])
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(module(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        module.eval()
        with torch.no_grad():
            val_scores = module(prep(Xval_t)).squeeze(-1).cpu().numpy()
        val_auroc = auroc(yval, val_scores)
        history.append({"epoch": epoch, "train_loss": total / len(Xtr_t), "val_auroc": val_auroc})
        if val_auroc > best_auroc + 1e-6:
            best_auroc = val_auroc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    return Bundle(name, kind, None, module, mu, sd, col_idx), history, best_epoch


# ---------------------------------------------------------------------------
# Metrics (numpy, no sklearn)
# ---------------------------------------------------------------------------

def _average_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return ap


def far_frr(labels: np.ndarray, scores: np.ndarray, tau: float = TAU_DEFAULT):
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores) >= tau
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    far = fp / max(fp + tn, 1)
    frr = fn / max(fn + tp, 1)
    return far, frr, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ece_bins(labels: np.ndarray, scores: np.ndarray, n_bins: int = 15):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (scores > lo) & (scores <= hi) if i > 0 else (scores >= lo) & (scores <= hi)
        n = int(m.sum())
        if n == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "conf": None, "acc": None})
            continue
        conf = float(scores[m].mean())
        acc = float(labels[m].mean())
        ece += (n / len(scores)) * abs(acc - conf)
        bins.append({"lo": lo, "hi": hi, "count": n, "conf": conf, "acc": acc})
    brier = float(np.mean((scores - labels.astype(np.float64)) ** 2))
    return float(ece), brier, bins


def ranking_metrics(scores_TBK: np.ndarray, labels_TBK: np.ndarray) -> Dict[str, float]:
    T, B, K = scores_TBK.shape
    top1 = np.zeros((T, B), dtype=bool)
    top2 = np.zeros((T, B), dtype=bool)
    rr = np.zeros((T, B), dtype=np.float64)
    has = np.zeros((T, B), dtype=bool)
    for t in range(T):
        s, y = scores_TBK[t], labels_TBK[t]
        order = np.argsort(-s, axis=1, kind="mergesort")
        sy = np.take_along_axis(y, order, axis=1)
        top1[t] = sy[:, 0]
        top2[t] = sy[:, :2].any(axis=1)
        has[t] = y.any(axis=1)
        first = np.argmax(sy, axis=1)
        rr[t] = np.where(sy.any(axis=1), 1.0 / (first + 1.0), 0.0)
    return {
        "top1_selection_accuracy": float(top1.mean()),
        "top2_recovery": float(top2.mean()),
        "mrr": float(rr[has].mean()) if has.any() else float("nan"),
        "top1_given_exists": float(top1[has].mean()) if has.any() else float("nan"),
        "exists_rate": float(has.mean()),
        "random_top1_expected": float(labels_TBK.mean()),
        "oracle_top1": float(has.mean()),
    }


def draw_relevant_metrics(scores_TBK, labels_TBK):
    """Decision-relevant selection metrics: can the verifier identify a correct
    draw when the deterministic candidate is wrong? (This is the subset that matters.)"""
    det_invalid = ~labels_TBK[:, :, 0]
    draw_scores = scores_TBK[:, :, 1:].reshape(-1)
    draw_labels = labels_TBK[:, :, 1:].reshape(-1)
    out = {"auroc_draws_only": auroc(draw_labels, draw_scores)}
    m3 = np.broadcast_to(det_invalid[:, :, None], labels_TBK[:, :, 1:].shape)
    out["auroc_decision_relevant_draws"] = auroc(draw_labels[m3.reshape(-1)], draw_scores[m3.reshape(-1)])
    top1_draw = np.zeros(labels_TBK.shape[:2], dtype=bool)
    for t in range(labels_TBK.shape[0]):
        order = np.argsort(-scores_TBK[t, :, 1:], axis=1, kind="mergesort")
        sy = np.take_along_axis(labels_TBK[t, :, 1:], order, axis=1)
        top1_draw[t] = sy[:, 0]
    out["top1_draws_given_det_invalid"] = (
        float(top1_draw[det_invalid].mean()) if det_invalid.any() else float("nan")
    )
    out["det_invalid_rate"] = float(det_invalid.mean())
    return out


def bundle_test_metrics(scores_TBK, labels_full_TBK, labels_tgt_TBK, meta_TB, tau=TAU_DEFAULT):
    sf = scores_TBK.reshape(-1)
    yf = labels_full_TBK.reshape(-1)
    yt = labels_tgt_TBK.reshape(-1)
    out = {
        "ranking_full_state": ranking_metrics(scores_TBK, labels_full_TBK),
        "ranking_target_register": ranking_metrics(scores_TBK, labels_tgt_TBK),
        "auroc_full": auroc(yf, sf),
        "auprc_full": average_precision(yf, sf),
        "auroc_target": auroc(yt, sf),
        "valid_rate_full": float(yf.mean()),
        "valid_rate_target": float(yt.mean()),
    }
    far, frr, conf = far_frr(yf, sf, tau)
    ece, brier, bins = ece_bins(yf, sf)
    out.update({"tau": tau, "far": far, "frr": frr, "confusion": conf,
                "ece": ece, "brier": brier, "reliability_bins": bins})
    # group breakdowns
    op_cat, first_err, term_fail = meta_TB["op_cat"], meta_TB["first_error_t"], meta_TB["terminal_fail"]
    top1 = np.zeros(labels_full_TBK.shape[:2], dtype=bool)
    for t in range(labels_full_TBK.shape[0]):
        order = np.argsort(-scores_TBK[t], axis=1, kind="mergesort")
        sy = np.take_along_axis(labels_full_TBK[t], order, axis=1)
        top1[t] = sy[:, 0]
    top1_draw = np.zeros(labels_full_TBK.shape[:2], dtype=bool)
    for t in range(labels_full_TBK.shape[0]):
        order = np.argsort(-scores_TBK[t, :, 1:], axis=1, kind="mergesort")
        sy = np.take_along_axis(labels_full_TBK[t, :, 1:], order, axis=1)
        top1_draw[t] = sy[:, 0]
    out["auroc_by_op"] = {}
    out["top1_by_op"] = {}
    out["det_valid_rate_by_op"] = {}
    out["top1_draws_by_op"] = {}
    for c, name in bf.OP_CAT_NAMES.items():
        m = (op_cat == c)
        if m.sum() == 0:
            continue
        m3 = np.broadcast_to(m[:, :, None], labels_full_TBK.shape)
        out["auroc_by_op"][name] = auroc(yf[m3.reshape(-1)], sf[m3.reshape(-1)])
        out["top1_by_op"][name] = float(top1[m].mean()) if m.sum() else float("nan")
        out["det_valid_rate_by_op"][name] = float(labels_full_TBK[:, :, 0][m].mean())
        out["top1_draws_by_op"][name] = float(top1_draw[m].mean()) if m.sum() else float("nan")
    det_invalid_all = ~labels_full_TBK[:, :, 0]
    out["top1_all_candidates_given_det_invalid"] = (
        float(top1[det_invalid_all].mean()) if det_invalid_all.any() else float("nan")
    )

    # decision-relevant subset: the deterministic candidate is wrong and draws are the only option
    out.update(draw_relevant_metrics(scores_TBK, labels_full_TBK))
    # by episode first-error timestep
    out["auroc_by_first_error_t"] = {}
    for t_err in list(range(1, 11)) + [-1]:
        m = (first_err == t_err)
        if m.sum() == 0:
            continue
        m3 = np.broadcast_to(m[:, :, None], labels_full_TBK.shape)
        out["auroc_by_first_error_t"][str(t_err)] = auroc(yf[m3.reshape(-1)], sf[m3.reshape(-1)])
    for flag, name in ((True, "terminal_fail_episodes"), (False, "terminal_success_episodes")):
        m = (term_fail == flag)
        if m.sum() == 0:
            continue
        m3 = np.broadcast_to(m[:, :, None], labels_full_TBK.shape)
        out.setdefault("auroc_by_episode_outcome", {})[name] = auroc(yf[m3.reshape(-1)], sf[m3.reshape(-1)])
    # recovery by oracle rank of the first valid candidate
    T, B, K = scores_TBK.shape
    rank_hist = np.zeros(K + 2, dtype=np.int64)
    rank_recovered = np.zeros(K + 2, dtype=np.int64)
    for t in range(T):
        s, y = scores_TBK[t], labels_full_TBK[t]
        order = np.argsort(-s, axis=1, kind="mergesort")
        sy = np.take_along_axis(y, order, axis=1)
        oracle_pos = np.argmax(sy, axis=1)
        has = sy.any(axis=1)
        chosen_correct = sy[:, 0]
        for b in range(B):
            if not has[b]:
                rank_hist[K + 1] += 1
                continue
            r = int(oracle_pos[b]) + 1
            rank_hist[r] += 1
            rank_recovered[r] += int(chosen_correct[b])
    out["selection_by_oracle_rank"] = {
        "counts": {str(i): int(rank_hist[i]) for i in range(1, K + 1)},
        "recovered": {str(i): int(rank_recovered[i]) for i in range(1, K + 1)},
    }
    return out


def taxonomy(scores_TBK, labels_TBK, tau=TAU_DEFAULT, confident=0.9):
    T, B, K = scores_TBK.shape
    counts = {"A_found_and_chosen": 0, "B_found_but_wrong_choice": 0,
              "C_no_correct_candidate": 0, "D_rejected_all": 0,
              "E_confident_wrong_choice": 0}
    for t in range(T):
        s, y = scores_TBK[t], labels_TBK[t]
        for b in range(B):
            if not y[b].any():
                counts["C_no_correct_candidate"] += 1
                continue
            mx = float(s[b].max())
            if mx < tau:
                counts["D_rejected_all"] += 1
                continue
            idx = int(np.argmax(s[b]))
            if y[b, idx]:
                counts["A_found_and_chosen"] += 1
            else:
                counts["B_found_but_wrong_choice"] += 1
                if float(s[b, idx]) >= confident:
                    counts["E_confident_wrong_choice"] += 1
    total = T * B
    return {"counts": counts, "total_transitions": total,
            "percentages": {k: 100.0 * v / total for k, v in counts.items()}}


# ---------------------------------------------------------------------------
# Trajectory rollouts
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_trajectory(method, model, prepared, stats, K, tau=TAU_DEFAULT,
                   bundle: Optional[Bundle] = None, seed: int = 0, device: torch.device = torch.device("cpu")):
    inputs, det, meta = prepared["inputs"], prepared["det"], prepared["meta"]
    B, T, _ = inputs.shape
    core = model.cognition_core
    head = model.decision_head
    true_states = meta["true_states"]
    state = torch.zeros(B, model.state_dim, device=device)
    step5_valid = np.zeros(B, dtype=bool)
    non_det_commits = np.zeros(B, dtype=np.int64)
    commit_hist = np.zeros(max(K, 2), dtype=np.int64)

    for t in range(T):
        x_t = inputs[:, t]
        e_t = model.encoder(x_t)
        f_t = torch.sigmoid(core.w_f_ih(e_t) + core.w_f_hh(state))
        h_t = core.cand_mlp[1](core.cand_mlp[0](torch.cat([f_t * state, e_t], dim=-1)))

        if method == "k1":
            c0 = torch.tanh(core.cand_mlp[2](h_t))
            chosen = c0
            idx = torch.zeros(B, dtype=torch.long)
        elif method == "single_draw":
            gen = torch.Generator(device="cpu").manual_seed(seed + t)
            chosen = sample_single_draw(core, h_t, stats["h_std"], gen, device)
            idx = torch.ones(B, dtype=torch.long)
        else:
            gen = torch.Generator(device="cpu").manual_seed(seed + 100 * t)
            C = bf.sample_candidates(core, state, f_t, h_t, GEN_MECH, stats, gen, device)[:, :K]
            if method == "k4_random":
                rng = np.random.default_rng(seed * 100003 + t)
                idx_np = rng.integers(0, K, size=B)
            elif method == "k4_oracle":
                Bk = branch_states(state, f_t, C)
                full, _ = decode_labels(model, Bk, true_states[:, t, :], meta["write_mask"][:, t, :])
                idx_np = np.argmax(full.cpu().numpy(), axis=1)
                none_valid = ~full.cpu().numpy().any(axis=1)
                idx_np[none_valid] = 0
            elif method == "k4_learned":
                groups = step_feature_groups(model, x_t, e_t, state, f_t, C)
                scores = bundle.score_groups(groups).cpu().numpy()
                idx_np = np.argmax(scores, axis=1)
                idx_np[scores.max(axis=1) < tau] = 0
            else:
                raise ValueError(method)
            idx = torch.from_numpy(idx_np).long().to(device)
            chosen = C[torch.arange(B), idx]

        if t == 4:
            dec5 = head(chosen).argmax(-1).cpu().numpy()
            step5_valid = (dec5 == true_states[:, t, :]).all(-1)
        if method != "k1":
            bc = np.bincount(idx.cpu().numpy(), minlength=len(commit_hist))
            commit_hist += bc[:len(commit_hist)]
            non_det_commits += (idx.cpu().numpy() != 0).astype(np.int64)
        state = (1.0 - f_t) * state + f_t * chosen

    final_pred = head(state).argmax(-1).cpu().numpy()
    correct = (final_pred == meta["final_true"]).all(-1)
    per_reg = float((final_pred == meta["final_true"]).mean())
    continuation = float(correct[step5_valid].mean()) if step5_valid.any() else float("nan")
    return {
        "preds": final_pred,
        "correct": correct,
        "exact_match": float(correct.mean()),
        "per_register": per_reg,
        "continuation_accuracy": continuation,
        "step5_valid_count": int(step5_valid.sum()),
        "non_deterministic_commits_mean": float(non_det_commits.mean()),
        "commit_histogram": commit_hist.tolist() if method != "k1" else [float(B * T)],
    }


@torch.no_grad()
def run_ensemble_control(model, prepared, stats, seed_base=78001, device=torch.device("cpu")):
    """4 independent single-candidate stochastic runs + per-register majority vote."""
    runs = [run_trajectory("single_draw", model, prepared, stats, 1, seed=seed_base + i, device=device)
            for i in range(4)]
    preds = np.stack([r["preds"] for r in runs], axis=0)  # (4,B,4)
    B = preds.shape[1]
    voted = np.zeros((B, 4), dtype=np.int64)
    det_preds = prepared["det_preds"][:, -1, :]
    for b in range(B):
        for r in range(4):
            vals, counts = np.unique(preds[:, b, r], return_counts=True)
            best = vals[counts == counts.max()]
            if det_preds[b, r] in best:
                voted[b, r] = det_preds[b, r]
            else:
                voted[b, r] = int(best.min())
    final_true = prepared["meta"]["final_true"]
    correct = (voted == final_true).all(-1)
    return {
        "exact_match": float(correct.mean()),
        "per_register": float((voted == final_true).mean()),
        "individual_exact_matches": [float(r["exact_match"]) for r in runs],
        "single_run_mean_exact_match": float(np.mean([r["exact_match"] for r in runs])),
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# FLOPs accounting
# ---------------------------------------------------------------------------

def model_mac_table(model):
    return {
        "encoder": 21 * 64 + 64 * 256,
        "gate": 212 * 256 + 212 * 212,
        "w1": (212 + 256) * 146,
        "w2": 146 * 212,
        "head": 212 * 40,
    }


def flops_for_method(method, T, K, macs, n_features, selector_kind=None):
    shared = macs["encoder"] + macs["gate"] + macs["w1"]
    if method == "k1":
        total = T * (shared + macs["w2"]) + macs["head"]
    elif method in ("k4_random", "k4_oracle"):
        total = T * (shared + K * macs["w2"]) + macs["head"]
    elif method == "k4_learned":
        sel = n_features if selector_kind == "linear" else n_features * 256 + 256 * 128 + 128
        total = T * (shared + K * (macs["w2"] + macs["head"]) + K * sel) + macs["head"]
    elif method == "ens4":
        total = 4 * (T * (shared + macs["w2"]) + macs["head"])
    else:
        raise ValueError(method)
    return 2.0 * total  # FLOPs = 2 x MACs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    parser = argparse.ArgumentParser(description="Phase-2 Verifier Feasibility")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=OUT_JSON)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    device = torch.device(args.device)
    started = time.time()

    print("=" * 90)
    print("RE:REASON — PHASE-2 VERIFIER FEASIBILITY (frozen core; K=4 primary)")
    print("=" * 90)

    model = load_frozen_model(device)
    train_split, val_split, test_split = make_splits()
    prepared = {
        "train": prepare_split(model, train_split, device),
        "val": prepare_split(model, val_split, device),
        "test": prepare_split(model, test_split, device),
    }
    det_test_em = float((prepared["test"]["det_preds"][:, -1, :] == prepared["test"]["meta"]["final_true"]).all(-1).mean())
    print(f"Split sizes: train={len(train_split)}, val={len(val_split)}, test={len(test_split)}")
    print(f"Deterministic K=1 test EM (sanity): {det_test_em:.4f}")
    assert abs(det_test_em - 0.5366666666666666) < 1e-9, "test split mismatch vs prior experiments"

    stats, primary_mech, noise_mech, reference_disp = compute_noise_stats(model, prepared["train"], K_PRIMARY)
    print(f"Noise stats: h_std_mean={float(stats['h_std'].mean()):.3f}, "
          f"reference_disp={reference_disp:.3f}, matched_noise_scale={noise_mech['scale']:.3f}")

    # ---- Features -----------------------------------------------------------
    print("\n[features] extracting train/val candidate sets (K=4)...")
    Xtr, ytr_full, ytr_tgt, ytr_full_TBK, _ = extract_split(
        model, prepared["train"], K_PRIMARY, stats, primary_mech, GENERATION_SEEDS["train"], device)
    Xval, yval_full, yval_tgt, yval_full_TBK, _ = extract_split(
        model, prepared["val"], K_PRIMARY, stats, primary_mech, GENERATION_SEEDS["val"], device)
    det_full_ok_train = prepared["train"]["meta"]["det_full_ok"]  # (B,T)
    assert (ytr_full_TBK[:, :, 0].T == det_full_ok_train).all(), \
        "candidate 0 labels must equal the deterministic committed-state labels"
    group_slices: Dict[str, Tuple[int, int]] = {}
    off = 0
    for name in FEATURE_ORDER:
        dim = {"cand": 212, "prev": 212, "gate": 212, "instr": 277, "baseline": 212,
               "delta_prev": 212, "delta_base": 212, "dists": 6, "dec_c": 92, "dec_0": 48,
               "cand_norm": 212}[name]
        group_slices[name] = (off, off + dim)
        off += dim
    print(f"  Xtr={Xtr.shape}, Xval={Xval.shape}, full feature dim={off}")
    print(f"  VALID base rates: train full={ytr_full.mean():.3f}, val full={yval_full.mean():.3f}, "
          f"train target={ytr_tgt.mean():.3f}")

    # ---- Train selectors (ablations x kinds) --------------------------------
    print("\n[train] selectors (linear + MLP) x 8 feature sets...")
    bundles: Dict[str, Bundle] = {}
    train_info: Dict[str, Any] = {}
    for ab_name, names in ABLATIONS.items():
        col_idx = column_index(group_slices, names)
        for kind in ("linear", "mlp"):
            key = f"{ab_name}__{kind}"
            bundle, history, best_epoch = train_selector(
                kind, Xtr, ytr_full, Xval, yval_full, col_idx, key,
                epochs=args.epochs, device=args.device)
            bundles[key] = bundle
            train_info[key] = {
                "feature_groups": names, "n_features": int(len(col_idx)),
                "val_auroc_best": history[best_epoch - 1]["val_auroc"] if best_epoch > 0 else None,
                "best_epoch": best_epoch, "epochs_ran": len(history),
            }
            print(f"  {key:<32} nf={len(col_idx):<5} val_auroc={history[best_epoch - 1]['val_auroc']:.4f}")

    del Xtr, Xval  # free training matrices

    # ---- Test scoring (K=4 primary) -----------------------------------------
    print("\n[score] test split, K=4, all selectors...")
    test_bundles = {k: v for k, v in bundles.items() if k.split("__")[0] in ABLATIONS}
    scores_test, labels_test_full, labels_test_tgt = score_split(
        test_bundles, model, prepared["test"], K_PRIMARY, stats, primary_mech,
        GENERATION_SEEDS["test"], device)
    assert (labels_test_full[:, :, 0].T == prepared["test"]["meta"]["det_full_ok"]).all(), \
        "test candidate 0 labels must equal deterministic committed-state validity"
    print(f"  test VALID base rates: full={labels_test_full.mean():.3f}, "
          f"target={labels_test_tgt.mean():.3f}, "
          f"det(full)={labels_test_full[:, :, 0].mean():.3f}, "
          f"draws(full)={labels_test_full[:, :, 1:].mean():.3f}")
    meta_TB = {
        "op_cat": prepared["test"]["meta"]["op_cat"].T,
        "first_error_t": np.broadcast_to(
            prepared["test"]["meta"]["first_error_t"][:, None], (len(test_split), 10)).T,
        "terminal_fail": np.broadcast_to(
            prepared["test"]["meta"]["terminal_fail"][:, None], (len(test_split), 10)).T,
    }

    test_metrics: Dict[str, Any] = {}
    for key, b in test_bundles.items():
        test_metrics[key] = bundle_test_metrics(
            scores_test[key], labels_test_full, labels_test_tgt, meta_TB, tau=TAU_DEFAULT)

    # ---- K=8 secondary ------------------------------------------------------
    print("\n[score] test split, K=8 (secondary ranking)...")
    k8_bundles = {k: v for k, v in bundles.items() if k.endswith("F_all__mlp") or k.endswith("F_all__linear")}
    scores_test_k8, labels_test_full_k8, _ = score_split(
        k8_bundles, model, prepared["test"], K_SECONDARY, stats, primary_mech,
        GENERATION_SEEDS["test"], device)
    k8_metrics = {k: {"ranking_full": ranking_metrics(scores_test_k8[k], labels_test_full_k8)}
                  for k in k8_bundles}

    # ---- Noise-family control (Task 6) --------------------------------------
    print("\n[noise control] matched random-noise candidates...")
    nXtr, nXval = None, None
    nXtr, nytr, _, _, _ = extract_split(
        model, prepared["train"], K_PRIMARY, stats, noise_mech, GENERATION_SEEDS["train"] + 500, device)
    nXval, nyval, _, _, _ = extract_split(
        model, prepared["val"], K_PRIMARY, stats, noise_mech, GENERATION_SEEDS["val"] + 500, device)
    noise_bundles: Dict[str, Bundle] = {}
    for kind in ("linear", "mlp"):
        col_idx = bundles[f"F_all__{kind}"].col_idx
        b, history, best_epoch = train_selector(
            kind, nXtr, nytr, nXval, nyval, col_idx, f"NoiseTrained_F_all__{kind}",
            epochs=args.epochs, device=args.device)
        noise_bundles[f"NoiseTrained_F_all__{kind}"] = b
    del nXtr, nXval

    all_noise_bundles = dict(noise_bundles)
    all_noise_bundles["PrimaryTrained_F_all__linear"] = bundles["F_all__linear"]
    all_noise_bundles["PrimaryTrained_F_all__mlp"] = bundles["F_all__mlp"]
    scores_noise, labels_noise_full, _ = score_split(
        all_noise_bundles, model, prepared["test"], K_PRIMARY, stats, noise_mech,
        GENERATION_SEEDS["test"] + 500, device)
    noise_metrics = {k: {
        "ranking": ranking_metrics(scores_noise[k], labels_noise_full),
        "auroc": auroc(labels_noise_full.reshape(-1), scores_noise[k].reshape(-1)),
        "valid_rate": float(labels_noise_full.mean()),
        **draw_relevant_metrics(scores_noise[k], labels_noise_full),
    } for k in all_noise_bundles}

    # ---- Trajectories -------------------------------------------------------
    print("\n[trajectories] running policies on the test split...")
    traj: Dict[str, Any] = {}
    flops_table: Dict[str, float] = {}
    macs = model_mac_table(model)
    preds_store: Dict[str, np.ndarray] = {}

    def timed(name, fn):
        before = rss_mb()
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        out["latency_ms_per_episode"] = 1000.0 * dt / len(test_split)
        out["wall_seconds"] = dt
        out["peak_rss_mb"] = rss_mb()
        out["rss_before_mb"] = before
        if "preds" in out:
            preds_store[name] = out.pop("preds")
        out.pop("correct", None)
        return out

    traj["k1"] = timed("k1", lambda: run_trajectory("k1", model, prepared["test"], stats, K_PRIMARY, device=device))
    assert abs(traj["k1"]["exact_match"] - det_test_em) < 1e-9, "K=1 rollout must reproduce the deterministic model"
    traj["k4_random"] = timed("k4_random", lambda: run_trajectory(
        "k4_random", model, prepared["test"], stats, K_PRIMARY, seed=77001, device=device))
    traj["k4_oracle"] = timed("k4_oracle", lambda: run_trajectory(
        "k4_oracle", model, prepared["test"], stats, K_PRIMARY, seed=77002, device=device))
    traj["k4_linear"] = timed("k4_linear", lambda: run_trajectory(
        "k4_learned", model, prepared["test"], stats, K_PRIMARY, tau=TAU_DEFAULT,
        bundle=bundles["F_all__linear"], seed=77003, device=device))
    traj["k4_mlp"] = timed("k4_mlp", lambda: run_trajectory(
        "k4_learned", model, prepared["test"], stats, K_PRIMARY, tau=TAU_DEFAULT,
        bundle=bundles["F_all__mlp"], seed=77004, device=device))

    # validation-tuned threshold for the MLP selector (val split only)
    tau_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tau_sweep = {}
    for tau in tau_grid:
        r = run_trajectory("k4_learned", model, prepared["val"], stats, K_PRIMARY, tau=tau,
                           bundle=bundles["F_all__mlp"], seed=88001, device=device)
        tau_sweep[tau] = {"val_em": r["exact_match"], "val_draw_commits_mean": r["non_deterministic_commits_mean"]}
    tau_star = max(tau_grid, key=lambda x: (tau_sweep[x]["val_em"], -abs(x - 0.5)))
    print(f"  tau sweep (val EM): { {k: round(v['val_em'], 4) for k, v in tau_sweep.items()} } -> tau*={tau_star}")
    traj["k4_mlp_tau_val"] = timed("k4_mlp_tau_val", lambda: run_trajectory(
        "k4_learned", model, prepared["test"], stats, K_PRIMARY, tau=tau_star,
        bundle=bundles["F_all__mlp"], seed=77004, device=device))

    ens = timed("ens4", lambda: run_ensemble_control(model, prepared["test"], stats, device=device))
    traj["ens4_stochastic"] = {k: v for k, v in ens.items() if k != "correct"}

    for key, method, kind in (("k1", "k1", None), ("k4_random", "k4_random", None),
                              ("k4_oracle", "k4_oracle", None),
                              ("k4_linear", "k4_learned", "linear"), ("k4_mlp", "k4_learned", "mlp"),
                              ("k4_mlp_tau_val", "k4_learned", "mlp"), ("ens4_stochastic", "ens4", None)):
        nf = len(bundles["F_all__mlp"].col_idx) if kind else 0
        flops_table[key] = flops_for_method(method, 10, K_PRIMARY, macs, nf, kind)
    flops_table["k1_with_4x_recurrence"] = None  # not applicable: frozen architecture has no R>1 path

    # ---- Task 8 taxonomy (MLP primary) --------------------------------------
    taxonomy_mlp = taxonomy(scores_test["F_all__mlp"], labels_test_full, tau=TAU_DEFAULT)
    taxonomy_linear = taxonomy(scores_test["F_all__linear"], labels_test_full, tau=TAU_DEFAULT)
    taxonomy_mlp_tau = taxonomy(scores_test["F_all__mlp"], labels_test_full, tau=tau_star)

    # ---- Assemble -----------------------------------------------------------
    payload = {
        "metadata": {
            "date": "2026-10-05",
            "scope": "Phase-2 verifier feasibility; frozen core; K=4 primary, K=8 ranking secondary",
            "checkpoint": CHECKPOINT,
            "test_split": {"sequences": len(test_split), "transitions": len(test_split) * 10},
            "verifier_train_sequences": len(train_split),
            "verifier_val_sequences": len(val_split),
            "noise_mech": {k: (float(v) if isinstance(v, float) else v) for k, v in noise_mech.items()},
            "reference_disp": reference_disp,
            "det_test_em_sanity": det_test_em,
            "runtime_s": None,
        },
        "label_base_rates": {
            "train_full": float(ytr_full.mean()), "train_target": float(ytr_tgt.mean()),
            "val_full": float(yval_full.mean()),
            "test_full": float(labels_test_full.mean()), "test_target": float(labels_test_tgt.mean()),
        },
        "selector_training": train_info,
        "test_metrics": test_metrics,
        "k8_secondary": k8_metrics,
        "noise_control": noise_metrics,
        "trajectories": traj,
        "tau_sweep": {str(k): v for k, v in tau_sweep.items()},
        "tau_star": tau_star,
        "taxonomy": {"mlp_tau0.5": taxonomy_mlp, "linear_tau0.5": taxonomy_linear,
                     f"mlp_tau{tau_star}": taxonomy_mlp_tau},
        "flops_per_episode": flops_table,
        "flops_ratio_vs_k1": {k: (v / flops_table["k1"] if v else None) for k, v in flops_table.items()},
    }
    payload["metadata"]["runtime_s"] = time.time() - started
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    np.savez_compressed("experiments/verifier_feasibility_preds.npz", **preds_store)
    print(f"\nResults written to {args.output}")

    # ---- Console: required primary table -----------------------------------
    def fmt(x, p=3):
        return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"

    mlp_m = test_metrics["F_all__mlp"]
    lin_m = test_metrics["F_all__linear"]
    print("\nPRIMARY TABLE")
    header = ("| Method | EM | Per-Reg | Top-1 Candidate Selection | AUROC | FAR | FRR | ECE | FLOPs/Episode |"
              "\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(header)
    print(f"| K=1 | {traj['k1']['exact_match']:.4f} | {traj['k1']['per_register']:.4f} | N/A | N/A | N/A | N/A | N/A | {flops_table['k1']/1e6:.2f}M |")
    print(f"| K=4 Random | {traj['k4_random']['exact_match']:.4f} | {traj['k4_random']['per_register']:.4f} | "
          f"{labels_test_full.mean():.4f} (expected) | N/A | N/A | N/A | N/A | {flops_table['k4_random']/1e6:.2f}M |")
    print(f"| K=4 Oracle | {traj['k4_oracle']['exact_match']:.4f} | {traj['k4_oracle']['per_register']:.4f} | "
          f"100% (solvable rate {mlp_m['ranking_full_state']['exists_rate']:.4f}) | 1.0 | 0 | 0 | 0 | {flops_table['k4_oracle']/1e6:.2f}M |")
    print(f"| K=4 Linear Selector | {traj['k4_linear']['exact_match']:.4f} | {traj['k4_linear']['per_register']:.4f} | "
          f"{lin_m['ranking_full_state']['top1_selection_accuracy']:.4f} | {lin_m['auroc_full']:.4f} | "
          f"{lin_m['far']:.4f} | {lin_m['frr']:.4f} | {lin_m['ece']:.4f} | {flops_table['k4_linear']/1e6:.2f}M |")
    print(f"| K=4 MLP Selector | {traj['k4_mlp']['exact_match']:.4f} | {traj['k4_mlp']['per_register']:.4f} | "
          f"{mlp_m['ranking_full_state']['top1_selection_accuracy']:.4f} | {mlp_m['auroc_full']:.4f} | "
          f"{mlp_m['far']:.4f} | {mlp_m['frr']:.4f} | {mlp_m['ece']:.4f} | {flops_table['k4_mlp']/1e6:.2f}M |")

    print("\nSUPPLEMENTARY")
    print(f"  MLP ranking: top2={mlp_m['ranking_full_state']['top2_recovery']:.4f}, "
          f"mrr={mlp_m['ranking_full_state']['mrr']:.4f}, "
          f"top1_given_exists={mlp_m['ranking_full_state']['top1_given_exists']:.4f}, "
          f"oracle_top1={mlp_m['ranking_full_state']['oracle_top1']:.4f}, "
          f"random_top1={mlp_m['ranking_full_state']['random_top1_expected']:.4f}")
    print(f"  Linear ranking: top1={lin_m['ranking_full_state']['top1_selection_accuracy']:.4f}, "
          f"auroc={lin_m['auroc_full']:.4f}, top1_given_exists={lin_m['ranking_full_state']['top1_given_exists']:.4f}")
    print(f"  Val->test AUROC: mlp {train_info['F_all__mlp']['val_auroc_best']:.4f} -> {mlp_m['auroc_full']:.4f}; "
          f"linear {train_info['F_all__linear']['val_auroc_best']:.4f} -> {lin_m['auroc_full']:.4f}")
    print(f"  Trajectories: ens4 EM={traj['ens4_stochastic']['exact_match']:.4f} "
          f"(single run mean {traj['ens4_stochastic']['single_run_mean_exact_match']:.4f}); "
          f"mlp tau*={tau_star}: EM={traj['k4_mlp_tau_val']['exact_match']:.4f}")
    print(f"  MLP draw commits/step (test): {traj['k4_mlp']['non_deterministic_commits_mean']:.3f}, "
          f"commit hist={traj['k4_mlp']['commit_histogram']}")
    det_top1_baseline = float(labels_test_full[:, :, 0].mean())
    print(f"  Always-deterministic top-1 baseline: {det_top1_baseline:.4f}")
    print(f"  Decision-relevant subset (det-invalid {mlp_m['det_invalid_rate']:.3f} of transitions): "
          f"mlp AUROC(draws)={mlp_m['auroc_decision_relevant_draws']:.4f}, "
          f"top-1 draw={mlp_m['top1_draws_given_det_invalid']:.4f}; "
          f"linear AUROC={lin_m['auroc_decision_relevant_draws']:.4f}, "
          f"top-1 draw={lin_m['top1_draws_given_det_invalid']:.4f}")
    print(f"  Ablation AUROC (mlp): " + ", ".join(
        f"{a}={test_metrics[a + '__mlp']['auroc_full']:.4f}" for a in ABLATIONS))
    print(f"  Noise control: primary-trained on noise AUROC="
          f"{noise_metrics['PrimaryTrained_F_all__mlp']['auroc']:.4f}, "
          f"noise-trained within-family AUROC={noise_metrics['NoiseTrained_F_all__mlp']['auroc']:.4f}; "
          f"decision-relevant AUROC primary-trained={noise_metrics['PrimaryTrained_F_all__mlp']['auroc_decision_relevant_draws']:.4f}, "
          f"noise-trained={noise_metrics['NoiseTrained_F_all__mlp']['auroc_decision_relevant_draws']:.4f}")
    print(f"  Taxonomy (MLP, tau=0.5): {json.dumps(taxonomy_mlp['counts'])}")
    print(f"  K=8 secondary MLP: {json.dumps(k8_metrics['F_all__mlp']['ranking_full'])}")

    # ---- Figures ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(FIG_DIR, exist_ok=True)
        bins = mlp_m["reliability_bins"]
        xs = [b["conf"] for b in bins if b["count"] > 0]
        ys = [b["acc"] for b in bins if b["count"] > 0]
        ns = [b["count"] for b in bins if b["count"] > 0]
        plt.figure(figsize=(4.6, 4.6))
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.scatter(xs, ys, s=[max(8, n / 300.0) for n in ns], c="tab:blue")
        plt.xlabel("predicted confidence")
        plt.ylabel("empirical VALID rate")
        plt.title("MLP selector reliability (test, K=4)")
        plt.savefig(os.path.join(FIG_DIR, "verifier_reliability.png"), dpi=150, bbox_inches="tight")
        plt.close()

        names = ["K=1", "K=4 Rand", "K=4 Oracle", "K=4 Linear", "K=4 MLP", "4x K=1 Ens"]
        vals = [traj["k1"]["exact_match"], traj["k4_random"]["exact_match"],
                traj["k4_oracle"]["exact_match"], traj["k4_linear"]["exact_match"],
                traj["k4_mlp"]["exact_match"], traj["ens4_stochastic"]["exact_match"]]
        plt.figure(figsize=(6.4, 3.6))
        bars = plt.bar(names, vals, color=["gray", "tab:orange", "tab:green", "tab:blue", "tab:purple", "tab:brown"])
        plt.axhline(traj["k1"]["exact_match"], color="gray", linestyle="--", linewidth=1)
        plt.ylabel("test exact match")
        plt.title("Terminal EM (K=4 methods; oracle is not achievable)")
        for bar, v in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        plt.ylim(0, max(vals) * 1.2)
        plt.savefig(os.path.join(FIG_DIR, "verifier_methods_em.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figures written to {FIG_DIR}/verifier_reliability.png and verifier_methods_em.png")
    except Exception as exc:  # pragma: no cover
        print(f"Figure generation skipped: {exc}")

    print(f"\nTotal runtime: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
