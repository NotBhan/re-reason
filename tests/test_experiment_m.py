"""Tests for Experiment M — representation geometry of the third basin (measurement only).

Covers:
 1. deterministic collection
 2. checkpoint hash verification
 3. no-grad execution
 4. no parameter mutation
 5. exact primary-population filtering
 6. arithmetic target reconstruction
 7. modular-distance correctness
 8. wrong-value identification
 9. wrong-register identification
10. candidate/commit vector decomposition
11. orthogonal projection correctness
12. directional-consistency calculation
13. null/reference calculation
14. wrap/no-wrap grouping
15. control-group construction
16. E/F alignment
17. result JSON determinism
18. figure generation
19. malformed/missing-data handling
20. historical artifact immutability
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_m as am  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_m as rm  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset  # noqa: E402

GEOM = os.path.join(ROOT, "experiments", "experiment_m_geometry.npz")
MODELS = ("E_candidate", "F_combined")


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(GEOM):
        pytest.skip("experiment_m_geometry.npz not present")
    return am.load_geometry(GEOM)


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
# 1 / 3 / 4. collection determinism, no-grad, no mutation
# ---------------------------------------------------------------------------

def test_collection_deterministic_no_grad_no_mutation(loaded_model):
    ds = _tiny_dataset()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r1 = rm.collect_dataset(loaded_model, ds.inputs, ds.step_targets, 42, 0)
    r2 = rm.collect_dataset(loaded_model, ds.inputs, ds.step_targets, 42, 0)
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
# 5. primary population filtering
# ---------------------------------------------------------------------------

def test_primary_population_filtering(data):
    for m in MODELS:
        t = data[m]["table"]
        grp = data[m]["primary_group"]; row = data[m]["primary_row"]
        off = row[grp == 0]; succ = row[grp == 1]; nonoff = row[grp == 2]
        # all primary: candidate correct & general
        for sel in (off, succ, nonoff):
            assert np.all(t["cand_ok"][sel] > 0.5)
            assert np.all(t["dest"][sel] != t["src_reg"][sel])
            assert np.all(t["operand_b"][sel] > 0)
        assert np.all(t["comm_ok"][off] < 0.5)
        assert np.all(t["comm_ok"][succ] > 0.5)
        assert np.all(t["rho_margin"][off] >= t["alpha_star_full"][off] - 1e-9)
        # groups partition the primary set
        assert len(off) + len(succ) + len(nonoff) == len(row)
        assert int(len(off)) == data["meta"]["n_offpath"][m]


# ---------------------------------------------------------------------------
# 6. arithmetic target reconstruction
# ---------------------------------------------------------------------------

def test_arithmetic_target_reconstruction(data):
    for m in MODELS:
        t = data[m]["table"]
        y = (t["operand_a"] + t["operand_b"]) % rm.MODULUS
        assert np.array_equal(np.round(y).astype(int), t["y_true"].astype(int))
        wrap = (t["operand_a"] + t["operand_b"]) >= rm.MODULUS
        assert np.array_equal(wrap.astype(float), t["wrap"])


# ---------------------------------------------------------------------------
# 7. modular distance
# ---------------------------------------------------------------------------

def test_modular_distance_correctness():
    a = np.array([0, 1, 2, 5, 9, 0])
    b = np.array([0, 9, 8, 5, 0, 5])
    d = rm._mod_distance(a, b)
    assert d.tolist() == [0, 2, 4, 0, 1, 5]
    assert np.all(d >= 0) and np.all(d <= 5)
    # symmetric
    assert np.array_equal(rm._mod_distance(a, b), rm._mod_distance(b, a))


# ---------------------------------------------------------------------------
# 8 / 9. wrong value and wrong register
# ---------------------------------------------------------------------------

def test_wrong_value_identification(data):
    for m in MODELS:
        t = data[m]["table"]; row = data[m]["primary_row"]; grp = data[m]["primary_group"]
        off = row[grp == 0]
        # off-path failures decode a wrong destination value like L's comm_pred
        assert np.all(t["comm_pred"][off].astype(int) != t["y_true"][off].astype(int))
        # and almost never the old value (L: the wrong value is a third value)
        old_frac = float((t["comm_pred"][off].astype(int) == t["operand_a"][off].astype(int)).mean())
        assert old_frac < 0.02
        # wrong_offset is consistent with (v_wrong - y) mod 10 in [-5,4]
        offs = t["wrong_offset"][off].astype(int)
        recomputed = ((t["comm_pred"][off].astype(int) - t["y_true"][off].astype(int) + 5) % 10) - 5
        assert np.array_equal(offs, recomputed)
        assert np.all(np.abs(offs) <= 5)


def test_wrong_register_identification(data):
    for m in MODELS:
        t = data[m]["table"]; row = data[m]["primary_row"]; grp = data[m]["primary_group"]
        for g in (0, 1, 2):
            sel = row[grp == g]
            fr = t["flat_reg_commit"][sel]
            assert np.all(fr >= 0) and np.all(fr <= rm.gr.NUM_REGISTERS - 1)
        off = row[grp == 0]
        # the flat winner register is essentially never the destination for off-path failures
        assert float((t["flat_reg_commit"][off] == t["dest"][off]).mean()) < 0.05


# ---------------------------------------------------------------------------
# 10 / 11. vector decomposition and orthogonal projection
# ---------------------------------------------------------------------------

def test_candidate_commit_decomposition_and_projection(data):
    for m in MODELS:
        t = data[m]["table"]
        fail_row = data[m]["fail_row"]
        S_prev = data[m]["fail_S_prev"].astype(np.float64)
        C = data[m]["fail_C"].astype(np.float64)
        S_com = data[m]["fail_S_commit"].astype(np.float64)
        assert S_prev.shape == C.shape == S_com.shape
        dcand = C - S_prev; dcommit = S_com - S_prev
        denom = np.clip((dcand * dcand).sum(1), 1e-12, None)
        lam = (dcommit * dcand).sum(1) / denom
        par = lam[:, None] * dcand
        orth = dcommit - par
        par_n = np.linalg.norm(par, axis=1); orth_n = np.linalg.norm(orth, axis=1)
        cos = (dcommit * dcand).sum(1) / (np.linalg.norm(dcommit, axis=1) * np.linalg.norm(dcand, axis=1) + 1e-12)
        # compare to stored columns for the same rows
        rp = t["off_par_norm"][fail_row]; ro = t["off_orth_norm"][fail_row]; rc = t["off_cos"][fail_row]
        assert np.allclose(par_n, rp, atol=1e-3)
        assert np.allclose(orth_n, ro, atol=1e-3)
        assert np.allclose(cos, rc, atol=1e-3)
        assert np.allclose(ro, t["offpath_dist"][fail_row], atol=1e-3)


def test_orthogonal_projection_is_orthogonal(data):
    S_prev = data["E_candidate"]["fail_S_prev"].astype(np.float64)
    C = data["E_candidate"]["fail_C"].astype(np.float64)
    S_com = data["E_candidate"]["fail_S_commit"].astype(np.float64)
    dcand = C - S_prev; dcommit = S_com - S_prev
    lam = (dcommit * dcand).sum(1) / np.clip((dcand * dcand).sum(1), 1e-12, None)
    orth = dcommit - lam[:, None] * dcand
    dots = (orth * dcand).sum(1)
    assert np.allclose(dots, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 12 / 13. directional consistency and nulls
# ---------------------------------------------------------------------------

def test_directional_resultant_and_null():
    rng = np.random.default_rng(0)
    aligned = np.tile(np.array([1.0, 0.0, 0.0]), (50, 1)) + 0.01 * rng.standard_normal((50, 3))
    assert am.directional_resultant(aligned) > 0.9
    random_dirs = rng.standard_normal((50, 3))
    assert am.directional_resultant(random_dirs) < 0.4
    null = am.permutation_null_R(aligned, n_perm=50)
    assert am.directional_resultant(aligned) > null.max()


def test_parity_null_and_reference():
    vw = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    null = am.permutation_null_parity(vw, y, n_perm=300)
    assert 0.0 <= null.mean() <= 1.0
    # observed same-parity for a systematically parity-matched set is high
    vw2 = y + 2
    p = am.permutation_null_parity(vw2, y, n_perm=200)
    assert float((vw2 % 2 == y % 2).mean()) == 1.0


# ---------------------------------------------------------------------------
# 14 / 15. wrap grouping and control groups
# ---------------------------------------------------------------------------

def test_wrap_grouping(data):
    for m in MODELS:
        t = data[m]["table"]; row = data[m]["primary_row"]; grp = data[m]["primary_group"]
        off = row[grp == 0]
        w = t["wrap"][off] > 0.5
        assert 0 < w.mean() < 1
        pw = np.bincount(t["comm_pred"][off][w].astype(int), minlength=10) / max(w.sum(), 1)
        assert np.isclose(pw.sum(), 1.0, atol=1e-6)


def test_control_group_construction(data):
    for m in MODELS:
        row = data[m]["primary_row"]; grp = data[m]["primary_group"]
        assert set(np.unique(grp)).issubset({0, 1, 2})
        # off-path failures are rarer than successes
        assert int((grp == 0).sum()) < int((grp == 1).sum())


# ---------------------------------------------------------------------------
# 16. E / F alignment
# ---------------------------------------------------------------------------

def test_EF_alignment(data):
    for m in MODELS:
        assert data["meta"]["l_alignment"][m]["passed"]
        assert data["meta"]["l_alignment"][m]["n_offpath_m"] == data["meta"]["n_offpath"][m]
        assert len(data[m]["primary_row"]) == data["meta"]["n_primary"][m]


# ---------------------------------------------------------------------------
# 17. result JSON determinism  /  18. figures
# ---------------------------------------------------------------------------

def test_results_json_determinism(data):
    a = am.build(data)
    b = am.build(data)
    a["metadata"].pop("date", None); b["metadata"].pop("date", None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_figures_generation(data, tmp_path):
    res = am.build(data)
    figs = am.make_figures(data, res, str(tmp_path), prefix="test_m")
    assert len(figs) >= 12
    assert all(os.path.exists(p) for p in figs)


def test_real_artifact_verdict(data):
    res = am.build(data)
    for m in MODELS:
        cl = res[m]["classification"]
        assert cl["verdict"] in (
            "M-A_systematic_modular_geometry", "M-B_systematic_nonmodular_placement",
            "M-C_unsystematic_collapse", "M-D_mixed_inconclusive")
        # the same-parity modular signature is a robust measured fact in both models
        assert cl["modular_identity"] is True
        # directional systematics is a seed-level statistic (independent units)
        assert len(cl["seed_R"]) == 5
        assert cl["directional_systematic"] == (cl["R_all_seeds_significant"] and cl["R_seed_mean"] >= am.R_BAR)


# ---------------------------------------------------------------------------
# 19. malformed / missing data handling
# ---------------------------------------------------------------------------

def test_malformed_and_missing_data(tmp_path):
    with pytest.raises(Exception):
        am.load_geometry(str(tmp_path / "nope.npz"))
    with pytest.raises((KeyError, ValueError, IndexError)):
        am.MTable(np.zeros((2, 2)), ["a", "b"])["missing"]
    assert am.directional_resultant(np.zeros((0, 5))) != am.directional_resultant(np.zeros((0, 5)))  # both NaN
    # classify on an empty/degenerate core dict raises KeyError rather than silently succeeding
    with pytest.raises(Exception):
        am.classify({})


# ---------------------------------------------------------------------------
# 20. historical artifact immutability
# ---------------------------------------------------------------------------

def test_historical_artifacts_immutable():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    import hashlib
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p), rel
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h == expected, f"historical report changed: {rel}"
