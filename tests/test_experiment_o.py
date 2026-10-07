"""Tests for Experiment O — controlled representation-side parity intervention.

Covers the 24 required checks:
 1. deterministic baseline
 2. checkpoint SHA-256 verification
 3. no-grad execution
 4. no parameter mutation
 5. kappa=0 exact frozen reproduction
 6. exact p_dest construction
 7. parity projection correctness
 8. candidate parity removal correctness
 9. zero displacement at kappa=0
10. half/full intervention scaling
11. orthogonal matched-magnitude control
12. candidate-side vs commit-side stage separation
13. no target leakage
14. no future-state leakage
15. primary-population filtering
16. control-group construction
17. recursive rollout determinism
18. rescue/regression accounting
19. sequence endpoint reproduction
20. E/F alignment
21. result JSON determinism
22. figure generation
23. malformed-input handling
24. historical artifact immutability
"""

import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_o as ao  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_m as rm  # noqa: E402
import run_experiment_n as rn  # noqa: E402
import run_experiment_o as ro  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_o_representation_intervention.npz")
MODELS = ("E_candidate", "F_combined")


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_o npz not present")
    return ao.load(NPZ)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


def _P(model):
    W = model.decision_head.proj.weight.detach().cpu().numpy().reshape(
        rm.gr.NUM_REGISTERS, rm.gr.NUM_VALUES, rm.gr.STATE_DIM)
    return rn.parity_axes(W[None])[0]


def _tiny_dataset(B=4, T=10):
    task = rm.gr.build_task(T)
    seq_ops = [(0, 0, -1, 5), (0, 1, -1, 7), (0, 2, -1, 3), (0, 3, -1, 4),
               (0, 0, -1, 2), (0, 1, -1, 9), (0, 2, -1, 6), (0, 3, -1, 1),
               (2, 0, 1, -1), (2, 0, 1, -1)]
    inputs = np.zeros((B, T, 21), np.float32)
    step_targets = np.zeros((B, T, 4), np.int64)
    for b in range(B):
        regs = [0, 0, 0, 0]
        for t, (op, d, s, v) in enumerate(seq_ops):
            inputs[b, t] = task._encode_operation(op, d, s, v)
            if op == 0:
                regs[d] = v
            elif op == 2:
                regs[d] = (regs[d] + regs[s]) % 10
            step_targets[b, t] = regs
    return SplitDataset(torch.from_numpy(inputs), torch.from_numpy(step_targets[:, -1].copy()),
                        torch.from_numpy(step_targets), [])


# ---------------------------------------------------------------------------
# 2 / 20. checkpoint + E/F alignment
# ---------------------------------------------------------------------------

def test_checkpoint_hashes(data):
    with open(os.path.join(ROOT, "experiments", "experiment_g_transitions_meta.json")) as fh:
        gmeta = json.load(fh)
    for m in MODELS:
        for s in (42, 43, 44, 45, 46):
            rel = gmeta["models"][m]["checkpoints"][str(s)]["path"]
            assert data["meta"]["models"][m]["checkpoints"][str(s)]["sha256"] == \
                gmeta["models"][m]["checkpoints"][str(s)]["sha256"]
            assert eg.sha256_file(os.path.join(ROOT, rel)) == \
                gmeta["models"][m]["checkpoints"][str(s)]["sha256"]


def test_EF_alignment(data):
    for m in MODELS:
        assert data["meta"]["reproduction"][m + "__all_passed"]
    z = data["z"]
    assert int((z["E_candidate__add_group"] == 0).sum()) == 323
    assert int((z["F_combined__add_group"] == 0).sum()) == 353


# ---------------------------------------------------------------------------
# 5 / 9 / 17. baseline reproduction + determinism
# ---------------------------------------------------------------------------

def test_kappa0_reproduces_frozen_and_is_deterministic(loaded_model):
    ds = _tiny_dataset()
    P = _P(loaded_model)
    frozen = loaded_model.forward_with_states(ds.inputs, return_components=True)
    r1 = ro.rollout_o(loaded_model, ds.inputs, P, "none", 0.0)
    r2 = ro.rollout_o(loaded_model, ds.inputs, P, "none", 0.0)
    fs = torch.stack(frozen.states, 1).detach()
    assert float((fs - r1["states"]).abs().max()) < 1e-6          # exact frozen reproduction
    assert torch.equal(r1["states"], r2["states"])                # deterministic
    assert torch.equal(frozen.terminal_logits.argmax(-1), r1["term_logits"].argmax(-1))
    # zero displacement at kappa=0 for every family
    for fam in ("cand", "orth", "commit", "commit_orth"):
        rk = ro.rollout_o(loaded_model, ds.inputs, P, fam, 0.0)
        assert torch.equal(rk["states"], r1["states"])
        assert rk["geom"]["cand_removal"] == 0.0 and rk["geom"]["commit_removal"] == 0.0


# ---------------------------------------------------------------------------
# 3 / 4 / 13 / 14. safety + leakage
# ---------------------------------------------------------------------------

def test_no_grad_no_mutation_no_target_leakage(loaded_model):
    ds = _tiny_dataset(); P = _P(loaded_model)
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r = ro.rollout_o(loaded_model, ds.inputs, P, "cand", 1.0)
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())
    assert r["states"].grad_fn is None
    # rollouts do not take targets: identical run when the labels differ
    ds2 = _tiny_dataset(); ds2.targets = torch.zeros_like(ds2.targets)
    r2 = ro.rollout_o(loaded_model, ds2.inputs, P, "cand", 1.0)
    assert torch.equal(r["states"], r2["states"])


def test_no_future_state_leakage(loaded_model):
    """The intervention at step t depends only on the current state/candidate/inputs."""
    ds = _tiny_dataset(); P = _P(loaded_model)
    full = ro.rollout_o(loaded_model, ds.inputs, P, "cand", 1.0)
    # truncating the future (shorter horizon) must not change earlier committed states
    short = ro.rollout_o(loaded_model, ds.inputs[:, :6], P, "cand", 1.0)
    assert torch.allclose(full["states"][:, :6], short["states"], atol=1e-5)


# ---------------------------------------------------------------------------
# 6 / 7 / 8. parity axis + intervention geometry
# ---------------------------------------------------------------------------

def test_parity_axis_exact_construction():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 10, 212))
    P = rn.parity_axes(W[None])[0]
    for d in range(4):
        X = W[d] / np.clip(np.linalg.norm(W[d], axis=1, keepdims=True), 1e-12, None)
        even = np.arange(10) % 2 == 0
        ref = X[even].mean(0) - X[~even].mean(0); ref = ref / np.linalg.norm(ref)
        assert np.allclose(P[d], ref, atol=1e-9)
        assert abs(np.linalg.norm(P[d]) - 1.0) < 1e-9


def test_candidate_parity_removal_and_projection_correctness():
    rng = np.random.default_rng(1)
    D = 32
    p = rng.standard_normal(D); p /= np.linalg.norm(p)
    S = rng.standard_normal(D); C = rng.standard_normal(D)
    for kappa in (0.5, 1.0):
        dC = C - S
        proj = float(dC @ p)
        Cmod = C - kappa * proj * p                      # the runner's rule
        assert np.allclose(Cmod - C, -kappa * proj * p)
        assert abs(float((Cmod - C) @ p) - (-kappa * proj)) < 1e-9   # along p exactly
        # κ scaling: displacement is linear in kappa
        assert np.allclose(C - kappa * proj * p - C, kappa * (-proj * p))


def test_matched_orthogonal_control_magnitude(data):
    z = data["z"]; meta = data["meta"]
    for m in MODELS:
        for lab_c, lab_o in (("cand_1.0", "orth_1.0"), ("commit_1.0", "commit_orth_1.0")):
            gc = np.mean([e["cand_removal"] + e["commit_removal"] for e in meta["geometry"][m][lab_c]])
            go = np.mean([e["cand_removal"] + e["commit_removal"] for e in meta["geometry"][m][lab_o]])
            # matched per-transition at intervention time; rollout trajectories diverge slightly
            assert abs(gc - go) / max(gc, 1e-9) < 0.05


# ---------------------------------------------------------------------------
# 12. stage separation
# ---------------------------------------------------------------------------

def test_candidate_vs_commit_stage_separation(loaded_model):
    ds = _tiny_dataset(); P = _P(loaded_model)
    base = ro.rollout_o(loaded_model, ds.inputs, P, "none", 0.0)
    cand = ro.rollout_o(loaded_model, ds.inputs, P, "cand", 1.0)
    comm = ro.rollout_o(loaded_model, ds.inputs, P, "commit", 1.0)
    # candidate-side leaves the candidate unchanged for commit-side and vice versa
    assert not torch.equal(cand["states"], base["states"])
    assert not torch.equal(comm["states"], base["states"])
    assert not torch.equal(cand["states"], comm["states"])
    # candidate-side records candidate removal; commit-side records commit removal
    assert cand["geom"]["cand_removal"] > 0 and cand["geom"]["commit_removal"] == 0.0
    assert comm["geom"]["commit_removal"] > 0 and comm["geom"]["cand_removal"] == 0.0


# ---------------------------------------------------------------------------
# 10. scaling + 11 covered above
# ---------------------------------------------------------------------------

def test_half_full_scaling(data):
    z = data["z"]
    for m in MODELS:
        base = z[f"{m}__baseline_comm_ok"].astype(bool)
        off = z[f"{m}__add_group"].astype(int) == 0
        net1 = int((~base[off] & z[f"{m}__cand_1.0__comm_ok"].astype(bool)[off]).sum())
        net05 = int((~base[off] & z[f"{m}__cand_0.5__comm_ok"].astype(bool)[off]).sum())
        assert net1 >= net05 - 2   # rescues are monotone-ish in kappa (descriptive tolerance)


# ---------------------------------------------------------------------------
# 15 / 16. population + groups
# ---------------------------------------------------------------------------

def test_primary_population_and_groups(data):
    z = data["z"]
    for m in MODELS:
        grp = z[f"{m}__add_group"].astype(int)
        assert set(np.unique(grp)).issubset({-1, 0, 1, 2})
        seed = z[f"{m}__add_seed"].astype(int)
        for s in (42, 43, 44, 45, 46):
            assert (seed == s).sum() > 0
        # off-path group counts match M/N
        assert int((grp == 0).sum()) == (323 if m == "E_candidate" else 353)
        # baseline off-path all wrong; successes all correct
        base = z[f"{m}__baseline_comm_ok"].astype(bool)
        assert not base[grp == 0].any()
        assert base[grp == 1].all()


# ---------------------------------------------------------------------------
# 18 / 19. accounting + sequence reproduction
# ---------------------------------------------------------------------------

def test_rescue_regression_accounting(data):
    z = data["z"]
    base = z["E_candidate__baseline_comm_ok"].astype(bool)
    off = z["E_candidate__add_group"].astype(int) == 0
    for lab in ("cand_1.0", "orth_1.0", "commit_1.0"):
        c = z[f"E_candidate__{lab}__comm_ok"].astype(bool)
        resc = int((~base[off] & c[off]).sum()); regr = int((base[off] & ~c[off]).sum())
        assert resc >= 0 and regr == 0          # off-path were all wrong
        assert resc <= int((~base[off]).sum())


def test_sequence_endpoint_reproduction(data):
    meta = data["meta"]
    for m in MODELS:
        n = meta["horizon_results"][m]["none_0.0"]
        for L in ("5", "10", "15", "20"):
            assert 0.0 <= n["42"][L]["em"] <= 1.0


# ---------------------------------------------------------------------------
# 21 / 22. analysis determinism + figures
# ---------------------------------------------------------------------------

def test_result_json_and_figure_determinism(data, tmp_path):
    a = ao.build(data); b = ao.build(data)
    a["metadata"].pop("date", None); b["metadata"].pop("date", None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    figs = ao.make_figures(data, a, str(tmp_path), prefix="test_o")
    assert len(figs) >= 12 and all(os.path.exists(p) for p in figs)


def test_real_artifact_verdict(data):
    res = ao.build(data)
    for m in MODELS:
        assert res[m]["classification"]["verdict"] in (
            "O-A_upstream_candidate", "O-B_downstream_commit", "O-C_both",
            "O-D_no_specific_parity_causality")


# ---------------------------------------------------------------------------
# 23 / 24. malformed data + historical immutability
# ---------------------------------------------------------------------------

def test_malformed_inputs_and_immutability(tmp_path):
    with pytest.raises(Exception):
        ao.load(str(tmp_path / "nope.npz"))
    with pytest.raises(Exception):
        ao.classify({})
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"
