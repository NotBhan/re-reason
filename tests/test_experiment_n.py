"""Tests for Experiment N — parity-origin localization (measurement only).

Covers the 20 required checks:
 1. deterministic collection
 2. checkpoint SHA-256 verification
 3. no-grad execution
 4. no parameter mutation
 5. exact M/L primary-population filtering
 6. W shape and register/value indexing
 7. weight normalization
 8. same/different-parity pair construction
 9. permutation-null determinism
10. parity-axis construction
11. state projection correctness
12. displacement projection correctness
13. state-permutation control
14. readout-permutation control
15. control-group construction
16. E/F row alignment
17. result JSON determinism
18. figure generation
19. malformed-data handling
20. historical-artifact immutability
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

import analyze_experiment_n as an  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_m as rm  # noqa: E402
import run_experiment_n as rn  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_n_parity_origin.npz")
MODELS = ("E_candidate", "F_combined")


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_n_parity_origin.npz not present")
    return an.load_npz(NPZ)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


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
# 1 / 3 / 4. collection
# ---------------------------------------------------------------------------

def test_collection_deterministic_no_grad_no_mutation(loaded_model):
    ds = _tiny_dataset()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r1 = rn.collect_primary_states(loaded_model, ds.inputs, ds.step_targets, 42)
    r2 = rn.collect_primary_states(loaded_model, ds.inputs, ds.step_targets, 42)
    assert np.array_equal(r1["scalars"], r2["scalars"])
    assert np.allclose(r1["W"], r2["W"])
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


# ---------------------------------------------------------------------------
# 2. checkpoint hashes
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


# ---------------------------------------------------------------------------
# 5. primary-population filtering
# ---------------------------------------------------------------------------

def test_primary_population_filtering(data):
    for m in MODELS:
        t = data[m]["table"]; grp = data[m]["primary_group"]; row = data[m]["primary_row"]
        off = row[grp == 0]; succ = row[grp == 1]; nonoff = row[grp == 2]
        assert len(off) + len(succ) + len(nonoff) == len(row)
        for sel in (off, succ, nonoff):
            assert np.all(t["cand_pred"][sel].astype(int) == t["y_true"][sel].astype(int))
            assert np.all(t["dest"][sel] != t["src_reg"][sel])
            assert np.all(t["operand_b"][sel] > 0)
        assert np.all(t["comm_pred"][off].astype(int) != t["y_true"][off].astype(int))
        assert np.all(t["comm_pred"][succ].astype(int) == t["y_true"][succ].astype(int))
        assert int(len(off)) == data["meta"]["n_offpath"][m]
    assert data["meta"]["n_offpath"]["E_candidate"] == 323
    assert data["meta"]["n_offpath"]["F_combined"] == 353


# ---------------------------------------------------------------------------
# 6 / 7 / 10. W geometry
# ---------------------------------------------------------------------------

def test_w_shape_and_indexing(data):
    for m in MODELS:
        W = data[m]["W"]; b = data[m]["b"]
        assert W.shape == (5, 4, 10, 212)
        assert b.shape == (5, 4, 10)
        assert np.all(np.isfinite(W))


def test_parity_axes_normalized_and_nondegenerate(data):
    P = rn.parity_axes(data["E_candidate"]["W"])
    assert P.shape == (5, 4, 212)
    norms = np.linalg.norm(P, axis=2)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # non-degenerate: even and odd centroids differ substantially
    W = data["E_candidate"]["W"]
    for s in range(5):
        X = W[s] / np.clip(np.linalg.norm(W[s], axis=2, keepdims=True), 1e-12, None)
        for d in range(4):
            ev = np.arange(10) % 2 == 0
            sep = np.linalg.norm(X[d][ev].mean(0) - X[d][~ev].mean(0))
            assert sep > 0.3


def test_same_diff_pair_construction():
    iu = np.triu_indices(10, 1)
    parity = np.arange(10) % 2
    same = parity[iu[0]] == parity[iu[1]]
    assert int(same.sum()) == 20     # C(5,2)*2 within-parity pairs
    assert int((~same).sum()) == 25  # 5*5 cross-parity pairs
    assert same.sum() + (~same).sum() == 45


# ---------------------------------------------------------------------------
# 9. permutation-null determinism
# ---------------------------------------------------------------------------

def test_permutation_null_determinism():
    rng1 = np.random.default_rng(1); rng2 = np.random.default_rng(1)
    p1 = rng1.permutation(np.arange(10) % 2); p2 = rng2.permutation(np.arange(10) % 2)
    assert np.array_equal(p1, p2)
    W = np.random.default_rng(0).standard_normal((4, 10, 212))
    assert an.w_effect_with_parity(W, p1) == an.w_effect_with_parity(W, p2)


# ---------------------------------------------------------------------------
# 11 / 12. projection correctness
# ---------------------------------------------------------------------------

def test_state_and_displacement_projection_correctness(data):
    m = "E_candidate"
    t = data[m]["table"]; grp = data[m]["primary_group"]; row = data[m]["primary_row"]
    off = row[grp == 0]
    fail_row = data[m]["fail_row"]; off_set = set(off.tolist())
    sel = np.array([fr in off_set for fr in fail_row])
    order = np.argsort(fail_row[sel])
    S_prev = data[m]["fail_S_prev"][sel][order]
    Sc = data[m]["fail_S_commit"][sel][order]
    W = data[m]["W"]; seed_order = {s: i for i, s in enumerate(an.SEEDS)}
    dest = t["dest"][off].astype(int); seed = t["seed"][off].astype(int)
    P = rn.parity_axes(W)
    proj, dproj = [], []
    for j in range(len(off)):
        p = P[seed_order[int(seed[j])], dest[j]]
        proj.append(float(Sc[j] @ p))
        dproj.append(float((Sc[j] - S_prev[j]) @ p))
    assert np.allclose(proj, t["par_proj_commit"][off], atol=1e-3)
    assert np.allclose(dproj, t["par_delta_commit"][off], atol=1e-3)


# ---------------------------------------------------------------------------
# 13 / 14. controls
# ---------------------------------------------------------------------------

def test_state_and_readout_permutation_controls(data):
    for m in MODELS:
        pc = an.pair_break_controls(m, data[m])
        obs = pc["obs_same_parity"]
        # pairing-break nulls collapse toward chance
        assert abs(pc["readout_only_null_mean"] - 0.5) < 0.10
        assert abs(pc["state_perm_null_mean"] - 0.5) < 0.10
        assert pc["gap_readout_only"] > 0.05
        # readout value-label permutation: mapping the decoded label back is a no-op (== obs)
        assert abs(pc["readout_perm_mapped_back_mean"] - obs) < 1e-6
        # ... and without mapping back it randomizes parity toward chance
        assert abs(pc["readout_perm_unmapped_mean"] - 0.5) < 0.05


# ---------------------------------------------------------------------------
# 15. control groups
# ---------------------------------------------------------------------------

def test_control_group_construction(data):
    for m in MODELS:
        grp = data[m]["primary_group"]
        assert set(np.unique(grp)).issubset({0, 1, 2})
        assert int((grp == 0).sum()) < int((grp == 1).sum())  # off-path rarer than successes


# ---------------------------------------------------------------------------
# 16. E/F alignment
# ---------------------------------------------------------------------------

def test_EF_alignment(data):
    for m in MODELS:
        assert data["meta"]["m_alignment"][m]["passed"]
        assert data["meta"]["m_alignment"][m]["n_offpath"] == data["meta"]["n_offpath"][m]
        assert data["meta"]["m_alignment"][m]["fail_vector_max_abs_diff"]["S_commit"] == 0.0


# ---------------------------------------------------------------------------
# 17. result JSON determinism  /  18. figures
# ---------------------------------------------------------------------------

def test_results_json_determinism(data):
    a = an.build(data); b = an.build(data)
    a["metadata"].pop("date", None); b["metadata"].pop("date", None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_figures_generation(data, tmp_path):
    res = an.build(data)
    figs = an.make_figures(data, res, str(tmp_path), prefix="test_n")
    assert len(figs) >= 12
    assert all(os.path.exists(p) for p in figs)


def test_real_artifact_verdict(data):
    res = an.build(data)
    for m in MODELS:
        c = res[m]["classification"]
        assert c["verdict"] in ("N-A_readout_dominant", "N-B_state_dominant",
                                "N-C_joint", "N-D_inconclusive")
        assert abs(c["readout_fraction"] + c["alignment_fraction"] - 1.0) < 1e-6
    assert res[an.PRIMARY_MODEL]["classification"]["verdict"] == "N-B_state_dominant"


# ---------------------------------------------------------------------------
# 19. malformed / missing data
# ---------------------------------------------------------------------------

def test_malformed_and_missing_data(tmp_path):
    with pytest.raises(Exception):
        an.load_npz(str(tmp_path / "nope.npz"))
    with pytest.raises((KeyError, ValueError, IndexError)):
        an.NTable(np.zeros((2, 2)), ["a", "b"])["missing"]
    with pytest.raises(Exception):
        an.classify({})
    # degenerate parity axis -> zero vector, no exception
    P = rn.parity_axes(np.zeros((1, 4, 10, 212)))
    assert np.allclose(P, 0.0)


# ---------------------------------------------------------------------------
# 20. historical-artifact immutability
# ---------------------------------------------------------------------------

def test_historical_artifacts_immutable():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p), rel
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"
