"""RE:Reason Mechanistic Audit Round 2 — Empirical Investigation Runner.

Executes all 17 controlled quantitative tests required to answer Q1 through Q17:
- Q1: Single complete computation with exact numerical values and statistics.
- Q2: Information content in S (R0..R3, R0+R1, R0-R1, R0 XOR R1, (R0+R1)%10).
- Q3: Register trace through time (SET R0=7, then 8 irrelevant operations).
- Q4: Gate conditional statistics (op, dest, src, displacement, timestep).
- Q5: Pre-gating information survival (S vs f * S probing).
- Q6: GELU vs Linear Jacobian dependence d(h)/d(S) on e.
- Q7: Candidate pre-gating ablations (f*S, raw S, random mask, shuffled gate, zero state).
- Q8: Step-by-step performance decomposition (1, 2, 3, 5, 8, 10 steps; SET/SWAP/ADD-only).
- Q9: Follow 20 failures backward (first error timestep, op, gate, candidate error).
- Q10: Controlled ADD diagnostics (isolated, SET+ADD, ADD+SET, repeated ADD, wrap vs no-wrap).
- Q11: Redundancy vs entanglement (single coordinate, masking, SVD spectrum, effective rank).
- Q12: Memory address equivariance (SET R0=5 vs R1=5 vs R2=5 vs R3=5).
- Q13: SWAP mechanics & candidate decoding.
- Q14: Internal arithmetic representation geometry (centroids of 0..9, pairwise distances).
- Q15: Readout capacity (Linear vs 2-layer MLP vs 3-layer MLP on frozen S_10).
- Q16: Candidate vs committed decodability on failures.
- Q17: Simulated branching oracle (K=1, 2, 4 upper bound recovery).
"""

import copy
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from re_reason.benchmarks.dataset_split import DisjointSplitManager, SplitDataset
from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.core.ablation_models import NonlinearCandidateMGUModel
from re_reason.utils.seeding import set_seed


# ---------------------------------------------------------------------------
# Setup & Helper Functions
# ---------------------------------------------------------------------------

def load_trained_mgu(checkpoint_path: str, device: torch.device) -> NonlinearCandidateMGUModel:
    model = NonlinearCandidateMGUModel(input_dim=21, state_dim=212, encoder_out_dim=256, candidate_hidden_dim=146)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def get_task_instance(seed: int = 42) -> RegisterTrackingTask:
    return RegisterTrackingTask(num_registers=4, sequence_length=10, seed=seed)


def summarize_tensor(t: torch.Tensor, name: str) -> Dict[str, Any]:
    flat = t.detach().cpu().flatten()
    return {
        "name": name,
        "shape": list(t.shape),
        "norm": float(flat.norm().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "slice_first_8": [round(float(v), 4) for v in flat[:8].tolist()],
    }


# ---------------------------------------------------------------------------
# Q1: Single Complete Computation
# ---------------------------------------------------------------------------

def run_q1_single_computation(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    batch = split_mgr.test_set.get_batch(1, idx=0)
    inp = batch.inputs.to(device)  # (1, 10, 21)
    target = batch.targets[0].tolist()

    with torch.no_grad():
        diag = model.forward_with_states(inp, return_components=True)

    # Let's inspect step t=2 (index 1) where S_(t-1) is non-zero
    x_2 = inp[0, 1]  # (21,)
    e_2 = model.encoder(x_2.unsqueeze(0))[0]  # (256,)
    S_prev = diag.states[0][0]  # S_1, shape (212,)
    
    # Gate calculation
    f_2 = torch.sigmoid(model.cognition_core.w_f_ih(e_2) + model.cognition_core.w_f_hh(S_prev))
    
    # Pre-gated state
    S_filt = f_2 * S_prev
    
    # Candidate hidden layer
    cand_mlp = model.cognition_core.cand_mlp
    cand_combined = torch.cat([S_filt, e_2], dim=-1)
    h_2 = cand_mlp[1](cand_mlp[0](cand_combined))  # GELU output, (146,)
    S_cand = torch.tanh(cand_mlp[2](h_2))          # (212,)
    
    # Committed state
    S_2 = (1.0 - f_2) * S_prev + f_2 * S_cand      # (212,)
    
    # Readout from S_2
    logits_2 = model.decision_head(S_2.unsqueeze(0))[0]  # (4, 10)
    preds_2 = logits_2.argmax(dim=-1).tolist()
    
    # Readout from final state S_10
    final_preds = model.decision_head(diag.states[-1][0].unsqueeze(0))[0].argmax(dim=-1).tolist()

    return {
        "step_2_transition": {
            "x_2_active_indices": (x_2 > 0).nonzero().squeeze().tolist(),
            "e_2": summarize_tensor(e_2, "e_2 (observation embedding)"),
            "S_prev": summarize_tensor(S_prev, "S_1 (previous state)"),
            "f_2": summarize_tensor(f_2, "f_2 (gate vector)"),
            "S_filt_2": summarize_tensor(S_filt, "f_2 * S_1 (pre-gated state)"),
            "h_2": summarize_tensor(h_2, "h_2 (GELU candidate hidden)"),
            "S_cand_2": summarize_tensor(S_cand, "S_cand_2 (candidate state)"),
            "S_2": summarize_tensor(S_2, "S_2 (committed state)"),
            "displacement_norm": float((S_2 - S_prev).norm().item()),
            "cosine_similarity_prev": float(F.cosine_similarity(S_2.unsqueeze(0), S_prev.unsqueeze(0)).item()),
            "step_2_readout_preds": preds_2,
            "final_target": target,
            "final_preds": final_preds,
        }
    }


# ---------------------------------------------------------------------------
# Q2: What Information is in S? (Arithmetic Probing)
# ---------------------------------------------------------------------------

def run_q2_arithmetic_probing(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    def extract_states_and_targets(split: SplitDataset, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = split.get_batch(n_samples, idx=0)
        with torch.no_grad():
            out = model.forward_with_states(batch.inputs.to(device))
            return out.states[-1].cpu(), batch.targets.cpu()

    train_S, train_R = extract_states_and_targets(split_mgr.train_set, 2000)
    test_S, test_R = extract_states_and_targets(split_mgr.test_set, 1200)

    def fit_linear_probe(y_tr: torch.Tensor, y_te: torch.Tensor, num_classes: int) -> float:
        probe = nn.Linear(train_S.shape[-1], num_classes)
        opt = torch.optim.AdamW(probe.parameters(), lr=0.01, weight_decay=1e-3)
        for _ in range(250):
            opt.zero_grad()
            loss = F.cross_entropy(probe(train_S), y_tr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = probe(test_S).argmax(dim=-1)
            return float((preds == y_te).float().mean().item())

    def fit_mlp_probe(y_tr: torch.Tensor, y_te: torch.Tensor, num_classes: int) -> float:
        probe = nn.Sequential(
            nn.Linear(train_S.shape[-1], 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )
        opt = torch.optim.AdamW(probe.parameters(), lr=0.01, weight_decay=1e-3)
        for _ in range(250):
            opt.zero_grad()
            loss = F.cross_entropy(probe(train_S), y_tr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = probe(test_S).argmax(dim=-1)
            return float((preds == y_te).float().mean().item())

    results = {}
    for i in range(4):
        y_tr = train_R[:, i]
        y_te = test_R[:, i]
        results[f"R{i}"] = {
            "linear_acc": fit_linear_probe(y_tr, y_te, 10),
            "mlp_acc": fit_mlp_probe(y_tr, y_te, 10),
            "chance": 0.10,
        }

    combos = {
        "R0_plus_R1": (train_R[:, 0] + train_R[:, 1], test_R[:, 0] + test_R[:, 1], 19),
        "R0_minus_R1": (train_R[:, 0] - train_R[:, 1] + 9, test_R[:, 0] - test_R[:, 1] + 9, 19),
        "R0_xor_R1": (train_R[:, 0] ^ train_R[:, 1], test_R[:, 0] ^ test_R[:, 1], 16),
        "R0_plus_R1_mod10": ((train_R[:, 0] + train_R[:, 1]) % 10, (test_R[:, 0] + test_R[:, 1]) % 10, 10),
    }

    for name, (y_tr, y_te, n_cls) in combos.items():
        results[name] = {
            "linear_acc": fit_linear_probe(y_tr, y_te, n_cls),
            "mlp_acc": fit_mlp_probe(y_tr, y_te, n_cls),
            "chance": 1.0 / n_cls,
        }

    return results


# ---------------------------------------------------------------------------
# Q3: Trace a Register Through Time (SET R0=7, 8 irrelevant operations)
# ---------------------------------------------------------------------------

def run_q3_trace_untouched_register(model: NonlinearCandidateMGUModel, device: torch.device) -> Dict[str, Any]:
    task = get_task_instance(seed=42)
    n_seq = 300
    rng = np.random.default_rng(42)
    
    samples = []
    for _ in range(n_seq):
        seq = np.zeros((10, task.input_dim), dtype=np.float32)
        # Step 0: SET R0 = 7
        seq[0] = task._encode_operation(task.OP_SET, 0, -1, 7)
        # Steps 1..8: random ops on R1, R2, R3 (dest in [1, 2, 3], src in [1, 2, 3])
        for t in range(1, 9):
            op = int(rng.integers(0, 3))
            d = int(rng.integers(1, 4))
            s = int(rng.integers(1, 4))
            v = int(rng.integers(0, 10))
            seq[t] = task._encode_operation(op, d, s, v)
        # Step 9: another op on R1, R2, R3
        seq[9] = task._encode_operation(task.OP_ADD, 1, 2, -1)
        samples.append(torch.from_numpy(seq))

    batch_inp = torch.stack(samples).to(device)  # (300, 10, 21)
    with torch.no_grad():
        diag = model.forward_with_states(batch_inp)
        states = diag.states  # 10 tensors of (300, 212)

    S_1 = states[0]  # Reference state right after SET R0 = 7
    timesteps_data = []

    for t_idx in range(1, 10):
        S_t = states[t_idx]
        euc_dist = (S_t - S_1).norm(dim=-1).mean().item()
        cos_sim = F.cosine_similarity(S_t, S_1, dim=-1).mean().item()
        norm_t = S_t.norm(dim=-1).mean().item()
        
        # Readout R0 accuracy
        logits_t = model.decision_head(S_t)  # (300, 4, 10)
        preds_r0 = logits_t[:, 0].argmax(dim=-1)
        acc_r0 = (preds_r0 == 7).float().mean().item()

        timesteps_data.append({
            "timestep": t_idx + 1,
            "r0_readout_acc": acc_r0,
            "euclidean_dist_from_S1": euc_dist,
            "cosine_sim_with_S1": cos_sim,
            "mean_norm": norm_t,
        })

    return {"steps": timesteps_data}


# ---------------------------------------------------------------------------
# Q4: What Does the Gate Know? (Conditional Statistics)
# ---------------------------------------------------------------------------

def run_q4_gate_statistics(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    test_set = split_mgr.test_set
    batch = test_set.get_batch(1000, idx=0)
    inp = batch.inputs.to(device)

    with torch.no_grad():
        diag = model.forward_with_states(inp, return_components=True)

    op_gates = {"SET": [], "SWAP": [], "ADD": []}
    dest_gates = {0: [], 1: [], 2: [], 3: []}
    t_gates = {t: [] for t in range(1, 11)}

    all_gates = []
    all_dests = []
    all_ops = []

    for t in range(10):
        g = diag.gates[t]        # (1000, 212)
        x_t = inp[:, t]          # (1000, 21)
        op_idx = x_t[:, :3].argmax(dim=-1).cpu().numpy()
        dest_idx = x_t[:, 3:7].argmax(dim=-1).cpu().numpy()
        g_means = g.mean(dim=-1).cpu().numpy()
        g_np = g.cpu().numpy()

        for b in range(1000):
            op_str = ["SET", "SWAP", "ADD"][op_idx[b]]
            op_gates[op_str].append(g_means[b])
            dest_gates[dest_idx[b]].append(g_means[b])
            t_gates[t + 1].append(g_means[b])

            if t < 2:  # collect sample for classification
                all_gates.append(g_np[b])
                all_dests.append(dest_idx[b])
                all_ops.append(op_idx[b])

    # Linear probe on gate vector to predict dest (4 classes) and op (3 classes)
    gate_t = torch.tensor(np.array(all_gates), dtype=torch.float32)
    dest_t = torch.tensor(np.array(all_dests), dtype=torch.long)
    op_t = torch.tensor(np.array(all_ops), dtype=torch.long)

    probe_dest = nn.Linear(212, 4)
    opt = torch.optim.Adam(probe_dest.parameters(), lr=0.01)
    for _ in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(probe_dest(gate_t[:1500]), dest_t[:1500])
        loss.backward()
        opt.step()
    dest_acc = (probe_dest(gate_t[1500:]).argmax(dim=-1) == dest_t[1500:]).float().mean().item()

    probe_op = nn.Linear(212, 3)
    opt_op = torch.optim.Adam(probe_op.parameters(), lr=0.01)
    for _ in range(200):
        opt_op.zero_grad()
        loss = F.cross_entropy(probe_op(gate_t[:1500]), op_t[:1500])
        loss.backward()
        opt_op.step()
    op_acc = (probe_op(gate_t[1500:]).argmax(dim=-1) == op_t[1500:]).float().mean().item()

    return {
        "mean_gate_by_op": {k: float(np.mean(v)) for k, v in op_gates.items()},
        "std_gate_by_op": {k: float(np.std(v)) for k, v in op_gates.items()},
        "mean_gate_by_dest": {str(k): float(np.mean(v)) for k, v in dest_gates.items()},
        "mean_gate_by_timestep": {str(k): float(np.mean(v)) for k, v in t_gates.items()},
        "gate_predicts_dest_accuracy": dest_acc,
        "gate_predicts_dest_chance": 0.25,
        "gate_predicts_op_accuracy": op_acc,
        "gate_predicts_op_chance": 0.333,
    }


# ---------------------------------------------------------------------------
# Q5: Pre-Gating Information Survival (S vs f * S)
# ---------------------------------------------------------------------------

def run_q5_pregating_survival(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    test_set = split_mgr.test_set
    batch = test_set.get_batch(1000, idx=0)
    inp = batch.inputs.to(device)

    with torch.no_grad():
        diag = model.forward_with_states(inp, return_components=True)

    S_prev = diag.states[3]   # S_4, (1000, 212)
    gate = diag.gates[4]      # f_5, (1000, 212)
    S_filt = gate * S_prev    # f_5 * S_4, (1000, 212)

    norm_before = S_prev.norm(dim=-1).mean().item()
    norm_after = S_filt.norm(dim=-1).mean().item()
    cos_sim = F.cosine_similarity(S_prev, S_filt, dim=-1).mean().item()

    # Train linear probe on S_prev vs S_filt to decode Register 0
    # using targets from model's own readout of S_prev
    with torch.no_grad():
        targets_r0 = model.decision_head(S_prev)[:, 0].argmax(dim=-1)

    def probe_acc(feats: torch.Tensor) -> float:
        probe = nn.Linear(feats.shape[-1], 10).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=0.01)
        for _ in range(150):
            opt.zero_grad()
            loss = F.cross_entropy(probe(feats[:700]), targets_r0[:700])
            loss.backward()
            opt.step()
        preds = probe(feats[700:]).argmax(dim=-1)
        return float((preds == targets_r0[700:]).float().mean().item())

    acc_before = probe_acc(S_prev)
    acc_after = probe_acc(S_filt)

    return {
        "norm_before": norm_before,
        "norm_after": norm_after,
        "suppression_ratio": norm_after / norm_before,
        "cosine_similarity_before_after": cos_sim,
        "probe_r0_decodability_before": acc_before,
        "probe_r0_decodability_after": acc_after,
    }


# ---------------------------------------------------------------------------
# Q6: GELU vs Linear Jacobian Dependence on e
# ---------------------------------------------------------------------------

def run_q6_jacobian_analysis(device: torch.device) -> Dict[str, Any]:
    torch.manual_seed(42)
    H_s, H_e, H_h = 212, 256, 146

    cand_linear1 = nn.Linear(H_s + H_e, H_h).to(device)
    W_S = cand_linear1.weight[:, :H_s]  # (146, 212)
    W_e = cand_linear1.weight[:, H_s:]  # (146, 256)
    b = cand_linear1.bias

    S = torch.randn(1, H_s, device=device)

    def gelu_prime(x: torch.Tensor) -> torch.Tensor:
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        pdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * x * x)
        return cdf + x * pdf

    obs_list = [torch.randn(1, H_e, device=device) for _ in range(50)]
    jacobians = []
    for e in obs_list:
        z = cand_linear1(torch.cat([S, e], dim=-1))
        diag_d = gelu_prime(z).squeeze(0)
        J = torch.diag(diag_d) @ W_S
        jacobians.append(J)

    diffs = []
    norms = [j.norm().item() for j in jacobians]
    for i in range(len(jacobians)):
        for j in range(i + 1, min(i + 10, len(jacobians))):
            rel_diff = (jacobians[i] - jacobians[j]).norm() / jacobians[i].norm()
            diffs.append(rel_diff.item())

    return {
        "linear_jacobian_dependence_on_e": 0.0,
        "gelu_mean_jacobian_norm": float(np.mean(norms)),
        "gelu_mean_relative_jacobian_difference_across_e": float(np.mean(diffs)),
        "gelu_max_relative_jacobian_difference_across_e": float(np.max(diffs)),
    }


# ---------------------------------------------------------------------------
# Q7: Candidate Pre-Gating Ablations
# ---------------------------------------------------------------------------

def run_q7_pregating_ablations(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    test_set = split_mgr.test_set
    batch = test_set.get_batch(500, idx=0)
    inp = batch.inputs.to(device)
    targets = batch.targets.to(device)

    def evaluate_ablation(mode: str) -> float:
        with torch.no_grad():
            B = inp.shape[0]
            S = torch.zeros(B, model.state_dim, device=device)
            cand_mlp = model.cognition_core.cand_mlp
            
            for t in range(10):
                x_t = inp[:, t]
                e_t = model.encoder(x_t)
                f_t = torch.sigmoid(model.cognition_core.w_f_ih(e_t) + model.cognition_core.w_f_hh(S))

                if mode == "standard":
                    S_filt = f_t * S
                elif mode == "raw_s":
                    S_filt = S
                elif mode == "random_mask":
                    mask = (torch.rand_like(S) < 0.33).float()
                    S_filt = mask * S
                elif mode == "shuffled_gate":
                    perm = torch.randperm(model.state_dim)
                    S_filt = f_t[:, perm] * S
                elif mode == "zero_state":
                    S_filt = torch.zeros_like(S)
                else:
                    raise ValueError(mode)

                cand_inp = torch.cat([S_filt, e_t], dim=-1)
                h = cand_mlp[1](cand_mlp[0](cand_inp))
                S_cand = torch.tanh(cand_mlp[2](h))
                S = (1.0 - f_t) * S + f_t * S_cand

            preds = model.decision_head(S).argmax(dim=-1)
            exact = (preds == targets).all(dim=-1).float().mean().item()
        return exact

    return {
        "standard_pregating_exact": evaluate_ablation("standard"),
        "raw_s_exact": evaluate_ablation("raw_s"),
        "random_mask_exact": evaluate_ablation("random_mask"),
        "shuffled_gate_exact": evaluate_ablation("shuffled_gate"),
        "zero_state_exact": evaluate_ablation("zero_state"),
    }


# ---------------------------------------------------------------------------
# Q8: Step-by-Step Performance Decomposition
# ---------------------------------------------------------------------------

def run_q8_step_decomposition(model: NonlinearCandidateMGUModel, device: torch.device) -> Dict[str, Any]:
    task = get_task_instance(seed=42)
    lengths = [1, 2, 3, 5, 8, 10]
    len_results = {}
    rng = np.random.default_rng(42)

    for L in lengths:
        sub_task = RegisterTrackingTask(num_registers=4, sequence_length=L, seed=100 + L)
        samples = []
        for i in range(500):
            inp_arr, step_tgts, final_tgt = sub_task.generate_single_sequence(rng)
            samples.append((torch.from_numpy(inp_arr), torch.from_numpy(final_tgt)))

        inps = torch.stack([s[0] for s in samples]).to(device)
        targets = torch.stack([s[1] for s in samples]).to(device)

        with torch.no_grad():
            B = inps.shape[0]
            S = torch.zeros(B, model.state_dim, device=device)
            cand_mlp = model.cognition_core.cand_mlp

            for t in range(L):
                x_t = inps[:, t]
                e_t = model.encoder(x_t)
                f_t = torch.sigmoid(model.cognition_core.w_f_ih(e_t) + model.cognition_core.w_f_hh(S))
                S_filt = f_t * S
                cand_inp = torch.cat([S_filt, e_t], dim=-1)
                h = cand_mlp[1](cand_mlp[0](cand_inp))
                S_cand = torch.tanh(cand_mlp[2](h))
                S = (1.0 - f_t) * S + f_t * S_cand

            preds = model.decision_head(S).argmax(dim=-1)
            exact = (preds == targets).all(dim=-1).float().mean().item()
            per_reg = (preds == targets).float().mean().item()
            len_results[str(L)] = {"exact_match": exact, "per_register": per_reg}

    def eval_homogeneous(op_type: int) -> float:
        samples = []
        for _ in range(500):
            seq = np.zeros((10, task.input_dim), dtype=np.float32)
            regs = [0, 0, 0, 0]
            for t in range(10):
                d = int(rng.integers(0, 4))
                s = int(rng.integers(0, 4))
                v = int(rng.integers(0, 10))
                seq[t] = task._encode_operation(op_type, d, s, v)
                if op_type == task.OP_SET:
                    regs[d] = v
                elif op_type == task.OP_SWAP:
                    regs[d], regs[s] = regs[s], regs[d]
                elif op_type == task.OP_ADD:
                    regs[d] = (regs[d] + regs[s]) % 10
            samples.append((torch.from_numpy(seq), torch.tensor(regs)))

        inps = torch.stack([s[0] for s in samples]).to(device)
        targets = torch.stack([s[1] for s in samples]).to(device)

        with torch.no_grad():
            out = model(inps)
            preds = out.logits.argmax(dim=-1)
            exact = (preds == targets).all(dim=-1).float().mean().item()
        return exact

    homo_results = {
        "SET_only": eval_homogeneous(task.OP_SET),
        "SWAP_only": eval_homogeneous(task.OP_SWAP),
        "ADD_only": eval_homogeneous(task.OP_ADD),
    }

    return {"by_sequence_length": len_results, "homogeneous_sequences": homo_results}


# ---------------------------------------------------------------------------
# Q9: Follow 20 Failures Backward
# ---------------------------------------------------------------------------

def run_q9_failure_chains(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> List[Dict[str, Any]]:
    test_set = split_mgr.test_set
    batch = test_set.get_batch(600, idx=0)
    inp = batch.inputs.to(device)
    targets = batch.targets.to(device)

    with torch.no_grad():
        diag = model.forward_with_states(inp, return_components=True)
        final_preds = model.decision_head(diag.states[-1]).argmax(dim=-1)

    failed_indices = (final_preds != targets).any(dim=-1).nonzero().squeeze().tolist()
    failure_table = []

    for idx in failed_indices[:20]:
        sample_inp = inp[idx].cpu().numpy()
        target = targets[idx].tolist()
        pred = final_preds[idx].tolist()

        seq = []
        for t in range(10):
            x = sample_inp[t]
            op_idx = int(x[:3].argmax())
            dest_idx = int(x[3:7].argmax())
            src_idx = int(x[7:11].argmax())
            val_idx = int(x[11:21].argmax())
            op = ["SET", "SWAP", "ADD"][op_idx]
            seq.append((op, dest_idx, src_idx, val_idx))

        true_states = []
        cur_r = [0, 0, 0, 0]
        for op, d, s, v in seq:
            if op == "SET":
                cur_r[d] = v
            elif op == "SWAP":
                cur_r[d], cur_r[s] = cur_r[s], cur_r[d]
            elif op == "ADD":
                cur_r[d] = (cur_r[d] + cur_r[s]) % 10
            true_states.append(list(cur_r))

        first_error_step = None
        first_error_op = None
        first_error_dest = None
        first_error_true = None
        first_error_pred = None
        gate_val = None

        for t in range(10):
            st = diag.states[t][idx]
            pred_t = model.decision_head(st.unsqueeze(0))[0].argmax(dim=-1).tolist()
            if pred_t != true_states[t] and first_error_step is None:
                first_error_step = t + 1
                first_error_op = seq[t][0]
                first_error_dest = seq[t][1]
                first_error_true = true_states[t]
                first_error_pred = pred_t
                gate_val = float(diag.gates[t][idx].mean().item())

        failure_table.append({
            "sample_index": idx,
            "final_target": target,
            "final_pred": pred,
            "num_registers_wrong": sum(p != tgt for p, tgt in zip(pred, target)),
            "first_error_step": first_error_step if first_error_step is not None else 10,
            "operation_at_error": first_error_op if first_error_op is not None else seq[-1][0],
            "target_reg_at_error": first_error_dest if first_error_dest is not None else seq[-1][1],
            "true_state_at_error": first_error_true,
            "pred_state_at_error": first_error_pred,
            "mean_gate_at_error": gate_val,
        })

    return failure_table


# ---------------------------------------------------------------------------
# Q10: Controlled ADD Diagnostics
# ---------------------------------------------------------------------------

def run_q10_controlled_add(model: NonlinearCandidateMGUModel, device: torch.device) -> Dict[str, Any]:
    task = get_task_instance(seed=42)
    rng = np.random.default_rng(42)

    def eval_custom_sequences(seq_builders: List[Any]) -> float:
        samples = []
        for builder in seq_builders:
            seq_ops = builder()
            seq = np.zeros((10, task.input_dim), dtype=np.float32)
            regs = [0, 0, 0, 0]
            for t, (op_type, d, s, v) in enumerate(seq_ops):
                seq[t] = task._encode_operation(op_type, d, s, v)
                if op_type == task.OP_SET:
                    regs[d] = v
                elif op_type == task.OP_SWAP:
                    regs[d], regs[s] = regs[s], regs[d]
                elif op_type == task.OP_ADD:
                    regs[d] = (regs[d] + regs[s]) % 10
            samples.append((torch.from_numpy(seq), torch.tensor(regs)))

        inps = torch.stack([s[0] for s in samples]).to(device)
        targets = torch.stack([s[1] for s in samples]).to(device)

        with torch.no_grad():
            out = model(inps)
            preds = out.logits.argmax(dim=-1)
            exact = (preds == targets).all(dim=-1).float().mean().item()
        return exact

    n_samples = 400

    def make_single_add():
        seq = [(task.OP_SET, r, -1, int(rng.integers(0, 10))) for r in range(4)]
        seq += [(task.OP_SET, int(rng.integers(0, 4)), -1, int(rng.integers(0, 10))) for _ in range(5)]
        seq.append((task.OP_ADD, int(rng.integers(0, 4)), int(rng.integers(0, 4)), -1))
        return seq

    def make_add_then_set():
        seq = [(task.OP_SET, r, -1, int(rng.integers(0, 10))) for r in range(4)]
        seq += [(task.OP_SET, int(rng.integers(0, 4)), -1, int(rng.integers(0, 10))) for _ in range(4)]
        seq.append((task.OP_ADD, int(rng.integers(0, 4)), int(rng.integers(0, 4)), -1))
        seq.append((task.OP_SET, int(rng.integers(0, 4)), -1, int(rng.integers(0, 10))))
        return seq

    def make_four_adds():
        seq = [(task.OP_SET, r, -1, int(rng.integers(0, 10))) for r in range(4)]
        seq += [(task.OP_SET, int(rng.integers(0, 4)), -1, int(rng.integers(0, 10))) for _ in range(2)]
        seq += [(task.OP_ADD, int(rng.integers(0, 4)), int(rng.integers(0, 4)), -1) for _ in range(4)]
        return seq

    def make_wrap():
        r0 = int(rng.integers(5, 10))
        r1 = int(rng.integers(5, 10))  # sum >= 10
        seq = [(task.OP_SET, 0, -1, r0), (task.OP_SET, 1, -1, r1), (task.OP_SET, 2, -1, 3), (task.OP_SET, 3, -1, 4)]
        seq += [(task.OP_SET, 2, -1, int(rng.integers(0, 10))) for _ in range(5)]
        seq.append((task.OP_ADD, 0, 1, -1))
        return seq

    def make_no_wrap():
        r0 = int(rng.integers(0, 5))
        r1 = int(rng.integers(0, 5))  # sum < 10
        seq = [(task.OP_SET, 0, -1, r0), (task.OP_SET, 1, -1, r1), (task.OP_SET, 2, -1, 3), (task.OP_SET, 3, -1, 4)]
        seq += [(task.OP_SET, 2, -1, int(rng.integers(0, 10))) for _ in range(5)]
        seq.append((task.OP_ADD, 0, 1, -1))
        return seq

    return {
        "single_add_at_end": eval_custom_sequences([make_single_add for _ in range(n_samples)]),
        "add_then_set": eval_custom_sequences([make_add_then_set for _ in range(n_samples)]),
        "four_consecutive_adds": eval_custom_sequences([make_four_adds for _ in range(n_samples)]),
        "add_with_modular_wrap": eval_custom_sequences([make_wrap for _ in range(n_samples)]),
        "add_without_modular_wrap": eval_custom_sequences([make_no_wrap for _ in range(n_samples)]),
    }


# ---------------------------------------------------------------------------
# Q11: Redundancy vs Entanglement (Ablations & SVD)
# ---------------------------------------------------------------------------

def run_q11_redundancy_and_svd(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    test_set = split_mgr.test_set
    batch = test_set.get_batch(1000, idx=0)
    inp = batch.inputs.to(device)
    targets = batch.targets.to(device)

    with torch.no_grad():
        out = model(inp)
        baseline_exact = (out.logits.argmax(dim=-1) == targets).all(dim=-1).float().mean().item()
        diag = model.forward_with_states(inp)
        S_final = diag.states[-1]  # (1000, 212)

    coord_drops = []
    for c in range(min(50, model.state_dim)):
        S_abl = S_final.clone()
        S_abl[:, c] = 0.0
        preds = model.decision_head(S_abl).argmax(dim=-1)
        ex = (preds == targets).all(dim=-1).float().mean().item()
        coord_drops.append(baseline_exact - ex)

    mask_results = {}
    for frac in [0.05, 0.10, 0.25, 0.50]:
        drops = []
        for _ in range(10):
            mask = (torch.rand_like(S_final) >= frac).float()
            preds = model.decision_head(S_final * mask).argmax(dim=-1)
            ex = (preds == targets).all(dim=-1).float().mean().item()
            drops.append(ex)
        mask_results[str(frac)] = float(np.mean(drops))

    U, s, V = torch.svd(S_final - S_final.mean(dim=0, keepdim=True))
    s_np = s.cpu().numpy()
    variance_explained = (s_np ** 2) / np.sum(s_np ** 2)
    cum_var = np.cumsum(variance_explained)

    pr = float((np.sum(s_np ** 2) ** 2) / np.sum(s_np ** 4))
    n_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    n_99 = int(np.searchsorted(cum_var, 0.99) + 1)

    return {
        "baseline_exact": baseline_exact,
        "mean_single_coord_drop": float(np.mean(coord_drops)),
        "max_single_coord_drop": float(np.max(coord_drops)),
        "random_masking_exact": mask_results,
        "participation_ratio": pr,
        "components_for_90pct_variance": n_90,
        "components_for_99pct_variance": n_99,
        "top_5_singular_values": [round(float(v), 2) for v in s_np[:5].tolist()],
    }


# ---------------------------------------------------------------------------
# Q12: Is There a Real Memory Address?
# ---------------------------------------------------------------------------

def run_q12_memory_address_equivariance(model: NonlinearCandidateMGUModel, device: torch.device) -> Dict[str, Any]:
    task = get_task_instance(seed=42)
    rng = np.random.default_rng(777)
    
    # 5 common prefix steps
    base_seq = np.zeros((10, task.input_dim), dtype=np.float32)
    for t in range(5):
        base_seq[t] = task._encode_operation(task.OP_SET, t % 4, -1, int(rng.integers(0, 10)))

    variants = []
    for r in range(4):
        seq = base_seq.copy()
        seq[5] = task._encode_operation(task.OP_SET, r, -1, 5)
        variants.append(torch.from_numpy(seq))

    batch = torch.stack(variants).to(device)  # (4, 10, 21)
    with torch.no_grad():
        diag = model.forward_with_states(batch, return_components=True)

    S_prev = diag.states[4]   # S_5 (identical for all 4)
    S_cur = diag.states[5]    # S_6
    gates = diag.gates[5]     # f_6
    disp = S_cur - S_prev

    cos_matrix = torch.mm(F.normalize(disp, dim=-1), F.normalize(disp, dim=-1).t()).cpu().numpy()
    gate_cos = torch.mm(F.normalize(gates, dim=-1), F.normalize(gates, dim=-1).t()).cpu().numpy()

    return {
        "displacement_pairwise_cosine_similarity": [[round(float(c), 3) for c in row] for row in cos_matrix],
        "gate_pairwise_cosine_similarity": [[round(float(c), 3) for c in row] for row in gate_cos],
        "mean_off_diagonal_displacement_cos": float((cos_matrix.sum() - 4.0) / 12.0),
        "mean_off_diagonal_gate_cos": float((gate_cos.sum() - 4.0) / 12.0),
    }


# ---------------------------------------------------------------------------
# Q13: SWAP Mechanics & Candidate Decoding
# ---------------------------------------------------------------------------

def run_q13_swap_candidate_decoding(model: NonlinearCandidateMGUModel, device: torch.device) -> Dict[str, Any]:
    task = get_task_instance(seed=42)
    rng = np.random.default_rng(42)
    samples = []

    for _ in range(300):
        seq = np.zeros((10, task.input_dim), dtype=np.float32)
        regs = [0, 0, 0, 0]
        for t in range(4):
            v = int(rng.integers(0, 10))
            seq[t] = task._encode_operation(task.OP_SET, t, -1, v)
            regs[t] = v
        for t in range(4, 9):
            d = int(rng.integers(0, 4))
            v = int(rng.integers(0, 10))
            seq[t] = task._encode_operation(task.OP_SET, d, -1, v)
            regs[d] = v
        # Step 10: SWAP R1, R2
        seq[9] = task._encode_operation(task.OP_SWAP, 1, 2, -1)
        regs[1], regs[2] = regs[2], regs[1]
        samples.append((torch.from_numpy(seq), torch.tensor(regs)))

    inps = torch.stack([s[0] for s in samples]).to(device)
    targets = torch.stack([s[1] for s in samples]).to(device)

    with torch.no_grad():
        diag = model.forward_with_states(inps, return_components=True)
        S_final = diag.states[-1]
        
        # Candidate at step 10
        S_prev = diag.states[-2]
        x_10 = inps[:, -1]
        e_10 = model.encoder(x_10)
        f_10 = diag.gates[-1]
        S_filt = f_10 * S_prev
        cand_inp = torch.cat([S_filt, e_10], dim=-1)
        cand_mlp = model.cognition_core.cand_mlp
        h = cand_mlp[1](cand_mlp[0](cand_inp))
        S_cand = torch.tanh(cand_mlp[2](h))

        preds_committed = model.decision_head(S_final).argmax(dim=-1)
        preds_cand = model.decision_head(S_cand).argmax(dim=-1)

        exact_committed = (preds_committed == targets).all(dim=-1).float().mean().item()
        exact_cand = (preds_cand == targets).all(dim=-1).float().mean().item()

        reg_acc_committed = (preds_committed == targets).float().mean(dim=0).tolist()
        reg_acc_cand = (preds_cand == targets).float().mean(dim=0).tolist()

    return {
        "committed_exact_match": exact_committed,
        "candidate_exact_match": exact_cand,
        "committed_per_register_acc": [round(a, 3) for a in reg_acc_committed],
        "candidate_per_register_acc": [round(a, 3) for a in reg_acc_cand],
    }


# ---------------------------------------------------------------------------
# Q14: Internal Arithmetic Representation Geometry
# ---------------------------------------------------------------------------

def run_q14_arithmetic_geometry(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    test_set = split_mgr.test_set
    batch = test_set.get_batch(1200, idx=0)
    inp = batch.inputs.to(device)
    targets = batch.targets.to(device)

    with torch.no_grad():
        diag = model.forward_with_states(inp)
        S_final = diag.states[-1]

    r0_targets = targets[:, 0].cpu().numpy()
    centroids = []
    for v in range(10):
        idx = (r0_targets == v).nonzero()[0]
        if len(idx) > 0:
            c = S_final[idx].mean(dim=0)
            centroids.append(c)
        else:
            centroids.append(torch.zeros(model.state_dim, device=device))

    centroids_t = torch.stack(centroids)
    dist_matrix = torch.cdist(centroids_t, centroids_t).cpu().numpy()

    adjacent_dists = [float(dist_matrix[v, (v + 1) % 10]) for v in range(10)]
    opposite_dists = [float(dist_matrix[v, (v + 5) % 10]) for v in range(10)]

    return {
        "mean_adjacent_distance": float(np.mean(adjacent_dists)),
        "std_adjacent_distance": float(np.std(adjacent_dists)),
        "mean_opposite_distance": float(np.mean(opposite_dists)),
        "std_opposite_distance": float(np.std(opposite_dists)),
        "adjacent_dists": [round(d, 3) for d in adjacent_dists],
        "is_circular_ring_like": bool(np.mean(opposite_dists) > np.mean(adjacent_dists)),
        "distance_matrix_sample": [[round(float(dist_matrix[i, j]), 2) for j in range(5)] for i in range(5)],
    }


# ---------------------------------------------------------------------------
# Q15: Representation vs Readout Capacity
# ---------------------------------------------------------------------------

def run_q15_readout_capacity(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    def extract_s10(split: SplitDataset, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        b = split.get_batch(n, idx=0)
        with torch.no_grad():
            out = model.forward_with_states(b.inputs.to(device))
            return out.states[-1].detach(), b.targets.to(device)

    train_S, train_Y = extract_s10(split_mgr.train_set, 2500)
    test_S, test_Y = extract_s10(split_mgr.test_set, 1200)

    with torch.no_grad():
        preds_curr = model.decision_head(test_S).argmax(dim=-1)
        curr_exact = (preds_curr == test_Y).all(dim=-1).float().mean().item()

    # Retrained Linear Readout
    head_lin = nn.Linear(model.state_dim, 40).to(device)
    opt_lin = torch.optim.AdamW(head_lin.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(300):
        opt_lin.zero_grad()
        loss = F.cross_entropy(head_lin(train_S).view(-1, 10), train_Y.view(-1))
        loss.backward()
        opt_lin.step()

    with torch.no_grad():
        preds_lin = head_lin(test_S).view(-1, 4, 10).argmax(dim=-1)
        lin_exact = (preds_lin == test_Y).all(dim=-1).float().mean().item()

    # 2-layer MLP Readout
    head_mlp2 = nn.Sequential(
        nn.Linear(model.state_dim, 256),
        nn.GELU(),
        nn.Linear(256, 40),
    ).to(device)
    opt_mlp2 = torch.optim.AdamW(head_mlp2.parameters(), lr=0.005, weight_decay=1e-3)
    for _ in range(400):
        opt_mlp2.zero_grad()
        loss = F.cross_entropy(head_mlp2(train_S).view(-1, 10), train_Y.view(-1))
        loss.backward()
        opt_mlp2.step()

    with torch.no_grad():
        preds_mlp2 = head_mlp2(test_S).view(-1, 4, 10).argmax(dim=-1)
        mlp2_exact = (preds_mlp2 == test_Y).all(dim=-1).float().mean().item()

    # 3-layer MLP Readout
    head_mlp3 = nn.Sequential(
        nn.Linear(model.state_dim, 512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Linear(256, 40),
    ).to(device)
    opt_mlp3 = torch.optim.AdamW(head_mlp3.parameters(), lr=0.003, weight_decay=1e-3)
    for _ in range(400):
        opt_mlp3.zero_grad()
        loss = F.cross_entropy(head_mlp3(train_S).view(-1, 10), train_Y.view(-1))
        loss.backward()
        opt_mlp3.step()

    with torch.no_grad():
        preds_mlp3 = head_mlp3(test_S).view(-1, 4, 10).argmax(dim=-1)
        mlp3_exact = (preds_mlp3 == test_Y).all(dim=-1).float().mean().item()

    return {
        "current_trained_readout_exact": curr_exact,
        "retrained_linear_readout_exact": lin_exact,
        "mlp_2layer_readout_exact": mlp2_exact,
        "mlp_3layer_readout_exact": mlp3_exact,
        "delta_mlp3_vs_current": mlp3_exact - curr_exact,
    }


# ---------------------------------------------------------------------------
# Q16: Candidate vs Committed Decodability on Failures
# ---------------------------------------------------------------------------

def run_q16_candidate_on_failures(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    test_set = split_mgr.test_set
    batch = test_set.get_batch(1000, idx=0)
    inp = batch.inputs.to(device)
    targets = batch.targets.to(device)

    with torch.no_grad():
        diag = model.forward_with_states(inp, return_components=True)
        S_final = diag.states[-1]
        preds_final = model.decision_head(S_final).argmax(dim=-1)

        S_prev = diag.states[-2]
        x_10 = inp[:, -1]
        e_10 = model.encoder(x_10)
        f_10 = diag.gates[-1]
        S_filt = f_10 * S_prev
        cand_inp = torch.cat([S_filt, e_10], dim=-1)
        cand_mlp = model.cognition_core.cand_mlp
        h = cand_mlp[1](cand_mlp[0](cand_inp))
        S_cand = torch.tanh(cand_mlp[2](h))
        preds_cand = model.decision_head(S_cand).argmax(dim=-1)

    failed_mask = (preds_final != targets).any(dim=-1)
    n_failed = failed_mask.sum().item()

    cand_correct_on_failed = ((preds_cand == targets).all(dim=-1) & failed_mask).sum().item()

    dest_10 = x_10[:, 3:7].argmax(dim=-1)
    dest_correct_committed = (preds_final.gather(1, dest_10.unsqueeze(1)) == targets.gather(1, dest_10.unsqueeze(1))).squeeze()
    dest_correct_cand = (preds_cand.gather(1, dest_10.unsqueeze(1)) == targets.gather(1, dest_10.unsqueeze(1))).squeeze()

    dest_cand_recovered = (dest_correct_cand & (~dest_correct_committed) & failed_mask).sum().item()

    return {
        "total_evaluated": 1000,
        "total_failures": n_failed,
        "candidate_fully_correct_on_failure": cand_correct_on_failed,
        "fraction_candidate_fully_correct": cand_correct_on_failed / max(1, n_failed),
        "target_reg_correct_in_candidate_when_wrong_in_committed": dest_cand_recovered,
        "fraction_target_recovered_by_candidate": dest_cand_recovered / max(1, n_failed),
    }


# ---------------------------------------------------------------------------
# Q17: Simulated Branching Oracle (K=1, 2, 4)
# ---------------------------------------------------------------------------

def run_q17_simulated_branching_oracle(model: NonlinearCandidateMGUModel, split_mgr: DisjointSplitManager, device: torch.device) -> Dict[str, Any]:
    test_set = split_mgr.test_set
    batch = test_set.get_batch(500, idx=0)
    inp = batch.inputs.to(device)
    targets = batch.targets.to(device)

    def simulate_oracle_branching(K: int) -> float:
        with torch.no_grad():
            B = inp.shape[0]
            branches = [torch.zeros(B, model.state_dim, device=device) for _ in range(K)]
            cand_mlp = model.cognition_core.cand_mlp

            for t in range(10):
                x_t = inp[:, t]
                e_t = model.encoder(x_t)
                
                new_branches = []
                for b_idx in range(len(branches)):
                    S_b = branches[b_idx]
                    f_t = torch.sigmoid(model.cognition_core.w_f_ih(e_t) + model.cognition_core.w_f_hh(S_b))

                    if K == 1:
                        S_filt = f_t * S_b
                        cand_inp = torch.cat([S_filt, e_t], dim=-1)
                        h = cand_mlp[1](cand_mlp[0](cand_inp))
                        S_cand = torch.tanh(cand_mlp[2](h))
                        S_next = (1.0 - f_t) * S_b + f_t * S_cand
                        new_branches.append(S_next)
                    elif K == 2:
                        # Variant 1: standard
                        S_filt1 = f_t * S_b
                        h1 = cand_mlp[1](cand_mlp[0](torch.cat([S_filt1, e_t], dim=-1)))
                        S_cand1 = torch.tanh(cand_mlp[2](h1))
                        S_next1 = (1.0 - f_t) * S_b + f_t * S_cand1
                        new_branches.append(S_next1)

                        # Variant 2: sharper gate commit
                        f_t2 = f_t.pow(0.8)
                        S_filt2 = f_t2 * S_b
                        h2 = cand_mlp[1](cand_mlp[0](torch.cat([S_filt2, e_t], dim=-1)))
                        S_cand2 = torch.tanh(cand_mlp[2](h2))
                        S_next2 = (1.0 - f_t2) * S_b + f_t2 * S_cand2
                        new_branches.append(S_next2)
                    elif K == 4:
                        for scale in [0.7, 0.9, 1.1, 1.3]:
                            f_s = torch.clamp(f_t * scale, 0.01, 0.99)
                            S_filt_s = f_s * S_b
                            h_s = cand_mlp[1](cand_mlp[0](torch.cat([S_filt_s, e_t], dim=-1)))
                            S_cand_s = torch.tanh(cand_mlp[2](h_s))
                            S_next_s = (1.0 - f_s) * S_b + f_s * S_cand_s
                            new_branches.append(S_next_s)

                branches = new_branches[:K]

            branch_preds = [model.decision_head(br).argmax(dim=-1) for br in branches]
            any_exact = torch.zeros(B, dtype=torch.bool, device=device)
            for p in branch_preds:
                any_exact = any_exact | (p == targets).all(dim=-1)

            return float(any_exact.float().mean().item())

    res_k1 = simulate_oracle_branching(1)
    res_k2 = simulate_oracle_branching(2)
    res_k4 = simulate_oracle_branching(4)

    return {
        "K1_baseline_exact": res_k1,
        "K2_oracle_exact": res_k2,
        "K4_oracle_exact": res_k4,
        "potential_gain_K2": res_k2 - res_k1,
        "potential_gain_K4": res_k4 - res_k1,
    }


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing RE:Reason Round 2 Mechanistic Audit on: {device}")

    ckpt_path = "experiments/checkpoints/6_nonlinear_candidate_mgu_seed42_3000steps.pt"
    mgu_model = load_trained_mgu(ckpt_path, device)
    task = get_task_instance(seed=42)
    split_mgr = DisjointSplitManager(task=task, base_seed=42)

    results = {}

    print("\n--- Running Q1: Single Concrete Computation Trace ---")
    results["q1_single_computation"] = run_q1_single_computation(mgu_model, split_mgr, device)
    print("  Completed Q1 trace.")

    print("\n--- Running Q2: Information Content in S (Arithmetic Probes) ---")
    results["q2_arithmetic_probes"] = run_q2_arithmetic_probing(mgu_model, split_mgr, device)
    print("  Results Q2:", results["q2_arithmetic_probes"])

    print("\n--- Running Q3: Trace Untouched Register R0 Across Time ---")
    results["q3_untouched_register_trace"] = run_q3_trace_untouched_register(mgu_model, device)
    print("  Completed Q3 trace.")

    print("\n--- Running Q4: What Does the Gate Know? (Conditional Stats) ---")
    results["q4_gate_statistics"] = run_q4_gate_statistics(mgu_model, split_mgr, device)
    print("  Results Q4:", results["q4_gate_statistics"])

    print("\n--- Running Q5: Pre-Gating Information Survival (S vs f*S) ---")
    results["q5_pregating_survival"] = run_q5_pregating_survival(mgu_model, split_mgr, device)
    print("  Results Q5:", results["q5_pregating_survival"])

    print("\n--- Running Q6: GELU vs Linear Jacobian Dependence on e ---")
    results["q6_jacobian_analysis"] = run_q6_jacobian_analysis(device)
    print("  Results Q6:", results["q6_jacobian_analysis"])

    print("\n--- Running Q7: Candidate Pre-Gating Ablations ---")
    results["q7_pregating_ablations"] = run_q7_pregating_ablations(mgu_model, split_mgr, device)
    print("  Results Q7:", results["q7_pregating_ablations"])

    print("\n--- Running Q8: Step-by-Step Performance Decomposition ---")
    results["q8_step_decomposition"] = run_q8_step_decomposition(mgu_model, device)
    print("  Results Q8:", results["q8_step_decomposition"])

    print("\n--- Running Q9: Follow 20 Failures Backward ---")
    results["q9_failure_chains"] = run_q9_failure_chains(mgu_model, split_mgr, device)
    print(f"  Followed {len(results['q9_failure_chains'])} failures.")

    print("\n--- Running Q10: Controlled ADD Diagnostics ---")
    results["q10_controlled_add"] = run_q10_controlled_add(mgu_model, device)
    print("  Results Q10:", results["q10_controlled_add"])

    print("\n--- Running Q11: Redundancy vs Entanglement (SVD & Ablations) ---")
    results["q11_redundancy_and_svd"] = run_q11_redundancy_and_svd(mgu_model, split_mgr, device)
    print("  Results Q11:", results["q11_redundancy_and_svd"])

    print("\n--- Running Q12: Memory Address Equivariance ---")
    results["q12_memory_address_equivariance"] = run_q12_memory_address_equivariance(mgu_model, device)
    print("  Results Q12:", results["q12_memory_address_equivariance"])

    print("\n--- Running Q13: SWAP Mechanics & Candidate Decoding ---")
    results["q13_swap_candidate_decoding"] = run_q13_swap_candidate_decoding(mgu_model, device)
    print("  Results Q13:", results["q13_swap_candidate_decoding"])

    print("\n--- Running Q14: Internal Arithmetic Representation Geometry ---")
    results["q14_arithmetic_geometry"] = run_q14_arithmetic_geometry(mgu_model, split_mgr, device)
    print("  Results Q14:", results["q14_arithmetic_geometry"])

    print("\n--- Running Q15: Representation vs Readout Capacity ---")
    results["q15_readout_capacity"] = run_q15_readout_capacity(mgu_model, split_mgr, device)
    print("  Results Q15:", results["q15_readout_capacity"])

    print("\n--- Running Q16: Candidate vs Committed on Failures ---")
    results["q16_candidate_on_failures"] = run_q16_candidate_on_failures(mgu_model, split_mgr, device)
    print("  Results Q16:", results["q16_candidate_on_failures"])

    print("\n--- Running Q17: Simulated Branching Oracle ---")
    results["q17_simulated_branching_oracle"] = run_q17_simulated_branching_oracle(mgu_model, split_mgr, device)
    print("  Results Q17:", results["q17_simulated_branching_oracle"])

    out_file = "experiments/round2_audit_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll Round 2 Mechanistic Audit results written to {out_file} successfully!")


if __name__ == "__main__":
    main()
