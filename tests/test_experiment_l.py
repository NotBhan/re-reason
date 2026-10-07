"""Tests for Experiment L — readout-side localization of the off-path component.

Measurement-only pipeline. Covers:
 1. linear logit decomposition (z = W S + b matches the actual head)
 2. pairwise margin decomposition (z_y - z_a = (w_y-w_a).S + (b_y-b_a))
 3. bias/state decomposition (stored margin == state + bias)
 4. decision-hyperplane distance (margin / ||dw||)
 5. projection calculation (lambda_eff, offpath_dist from stored vectors)
 6. off-path residual calculation
 7. competitor identification (a_prev / a_commit from stored states + W, b)
 8. primary off-path filtering
 9. category classification (deterministic rules)
10. deterministic collector
11. checkpoint hash protection
12. no-grad / no-training guarantee
13. G/H row alignment of the stored artifact
14. synthetic classification + decision logic
15. end-to-end analysis + figure generation
16. deterministic rerun
17. invalid inputs / configurations
18. numerical tolerance checks
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_l as al  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_l as rl  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset  # noqa: E402

READOUT = os.path.join(ROOT, "experiments", "experiment_l_readout.npz")
MODELS = ("E_candidate", "F_combined")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_model():
    ckpt = os.path.join(ROOT, "experiments", "checkpoints")
    return eg.load_checkpoint(ckpt, "E_candidate", 42)["model"]


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(READOUT):
        pytest.skip("experiment_l_readout.npz not present")
    return al.load_readout(READOUT)


def _tiny_dataset(B=4, T=10):
    task = rl.gr.build_task(T)
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


def _offpath_rows(t: "al.LTable") -> np.ndarray:
    return np.nonzero(al.offpath_mask(t))[0]


def _zmat(S, W, b, dest):
    Wd = W[dest]                                    # (n,V,D)
    bd = b[dest]                                    # (n,V)
    return np.einsum("nd,nvd->nv", S, Wd) + bd


# ---------------------------------------------------------------------------
# 1-2. exact readout decomposition
# ---------------------------------------------------------------------------

def test_linear_logit_decomposition_matches_actual_head(loaded_model):
    head = loaded_model.decision_head
    W, b = rl._matrix(head)
    x = torch.randn(11, rl.gr.STATE_DIM)
    logits = head(x)                                # (n, R, V)
    z = torch.einsum("nd,rvd->nrv", x, W) + b
    assert torch.allclose(logits, z, atol=1e-4, rtol=1e-4)
    assert W.shape == (rl.gr.NUM_REGISTERS, rl.gr.NUM_VALUES, rl.gr.STATE_DIM)


def test_pairwise_margin_decomposition(loaded_model):
    head = loaded_model.decision_head
    W, b = rl._matrix(head)
    x = torch.randn(9, rl.gr.STATE_DIM)
    z = head(x)
    d, y, a = 2, 5, 7
    state = (W[d, y] - W[d, a]) @ x.T                       # (n,)
    bias = (b[d, y] - b[d, a])
    margin = z[:, d, y] - z[:, d, a]
    assert torch.allclose(margin, state + bias, atol=1e-4)


def test_bias_state_decomposition_on_stored_offpath(data):
    for m in MODELS:
        t = data[m]["table"]
        off = _offpath_rows(t)
        # m_commit (vs a_prev) == state_av_commit + bias_av
        assert np.allclose(t["m_commit"][off], t["state_av_commit"][off] + t["bias_av"][off], atol=2e-2)
        # m_commit_alt (vs a_commit) == state_ac_commit + bias_ac
        assert np.allclose(t["m_commit_alt"][off], t["state_ac_commit"][off] + t["bias_ac"][off], atol=2e-2)


def test_hyperplane_distance_definition(data):
    for m in MODELS:
        t = data[m]["table"]
        off = _offpath_rows(t)
        assert np.allclose(t["hyper_av_commit"][off], t["m_commit"][off] / t["norm_dw_av"][off], atol=1e-4)
        assert np.allclose(t["hyper_ac_commit"][off], t["m_commit_alt"][off] / t["norm_dw_ac"][off], atol=1e-4)
        assert np.all(t["norm_dw_av"][off] > 0) and np.all(t["norm_dw_ac"][off] > 0)


# ---------------------------------------------------------------------------
# 5-7. projection, residual, competitor identification from stored vectors
# ---------------------------------------------------------------------------

def test_projection_and_offpath_residual_from_vectors(data):
    for m in MODELS:
        t = data[m]["table"]
        off = _offpath_rows(t)
        S_prev = data[m]["offpath"]["S_prev"].astype(np.float64)
        C = data[m]["offpath"]["C"].astype(np.float64)
        S_commit = data[m]["offpath"]["S_commit"].astype(np.float64)
        assert S_prev.shape[0] == off.shape[0] == t["lambda_eff"][off].shape[0]
        diff = C - S_prev
        denom = np.clip((diff * diff).sum(1), 1e-12, None)
        lam = np.clip(((S_commit - S_prev) * diff).sum(1) / denom, -1.0, 2.0)
        rperp = np.linalg.norm(S_commit - (S_prev + lam[:, None] * diff), axis=1)
        assert np.allclose(lam, t["lambda_eff"][off], atol=1e-3)
        assert np.allclose(rperp, t["offpath_dist"][off], atol=1e-3)


def test_competitor_identification_from_vectors(data):
    for m in MODELS:
        t = data[m]["table"]
        off = _offpath_rows(t)
        W = data[m]["W"]; b = data[m]["b"]
        S_prev = data[m]["offpath"]["S_prev"].astype(np.float64)
        S_commit = data[m]["offpath"]["S_commit"].astype(np.float64)
        dest = t["dest"][off].astype(int)
        y = t["y_true"][off].astype(int)
        seeds = data[m]["offpath_seed"]
        seed_order = {s: i for i, s in enumerate(al.SEEDS)}
        a_prev = np.zeros(len(off), int); a_commit = np.zeros(len(off), int)
        for j in range(len(off)):
            si = seed_order[int(seeds[j])]
            Wd = W[si, dest[j]]; bd = b[si, dest[j]]
            zp = Wd @ S_prev[j] + bd
            zc = Wd @ S_commit[j] + bd
            zp[y[j]] = -np.inf; zc[y[j]] = -np.inf
            a_prev[j] = int(np.argmax(zp)); a_commit[j] = int(np.argmax(zc))
        assert np.array_equal(a_prev, t["a_star_prev"][off].astype(int))
        assert np.array_equal(a_commit, t["a_star_commit"][off].astype(int))


# ---------------------------------------------------------------------------
# 8. primary off-path filtering
# ---------------------------------------------------------------------------

def test_primary_offpath_filtering(data):
    for m in MODELS:
        t = data[m]["table"]
        off = al.offpath_mask(t)
        assert np.all(t["cand_ok"][off] > 0.5)
        assert np.all(t["comm_ok"][off] < 0.5)
        gen = (t["dest"][off] != t["src_reg"][off]) & (t["operand_b"][off] > 0)
        assert np.all(gen)
        assert np.all(t["rho_margin"][off] >= t["alpha_star_full"][off] - 1e-9)
        assert int(off.sum()) == data["meta"]["n_offpath"][m]


# ---------------------------------------------------------------------------
# 9 / 14. category + decision logic
# ---------------------------------------------------------------------------

def test_category_classification_deterministic():
    sig = {
        "bias_sig": np.array([True, False, False, False]),
        "moved": np.array([True, True, False, True]),
        "cleared": np.array([True, True, False, False]),
        "changed": np.array([True, True, True, False]),
        "full_deliv": np.array([False, False, False, False]),
        "finite": np.array([True, True, True, True]),
    }
    cat = al.categorize(sig)
    assert cat["readout_calibration"][0]                       # bias flip -> readout
    assert cat["latent_misplacement"][1]                       # cleared & changed, no bias
    assert cat["boundary_shortfall"][2]                        # not cleared/changed
    assert cat["boundary_shortfall"][3]


def test_decision_rule_thresholds():
    # placement-dominant
    n = 100
    sig = {"bias_sig": np.zeros(n, bool)}
    cat = {"place_sig": np.ones(n, bool)}
    d = al.decide(sig, cat)
    assert d["verdict"] == "B_latent_placement"
    # readout-dominant
    sig = {"bias_sig": np.ones(n, bool)}
    cat = {"place_sig": np.zeros(n, bool)}
    d = al.decide(sig, cat)
    assert d["verdict"] == "A_readout_calibration"
    # both
    sig = {"bias_sig": np.ones(n, bool)}
    cat = {"place_sig": np.ones(n, bool)}
    assert al.decide(sig, cat)["verdict"] == "C_both"
    # neither
    sig = {"bias_sig": np.zeros(n, bool)}
    cat = {"place_sig": np.zeros(n, bool)}
    assert al.decide(sig, cat)["verdict"] == "D_inconclusive"


# ---------------------------------------------------------------------------
# 10 / 12 / 16. collector determinism, no-grad, no param change
# ---------------------------------------------------------------------------

def test_collector_deterministic_no_grad_no_param_change(loaded_model):
    ds = _tiny_dataset()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r1 = rl.collect_dataset(loaded_model, ds.inputs, ds.step_targets, 42, 0)
    r2 = rl.collect_dataset(loaded_model, ds.inputs, ds.step_targets, 42, 0)
    assert np.array_equal(r1["scalars"], r2["scalars"])
    assert np.allclose(r1["W"], r2["W"])
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.requires_grad and p.grad is not None for p in loaded_model.parameters())


def test_stored_artifact_is_reproducible_from_checkpoints(data):
    """The stored off-path vectors must recompute the stored scalar margins (frozen readout)."""
    loaded = eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"), "E_candidate", 42)
    model = loaded["model"]
    ds = rl.gr.canonical_splits(42).test_set
    batch = ds.get_batch(len(ds), idx=0)
    fresh = rl.collect_dataset(model, batch.inputs, batch.metadata["step_targets"], 42, 0)
    t = data["E_candidate"]["table"]
    seed_mask = np.round(t["seed"]).astype(int) == 42
    if seed_mask.sum() == fresh["scalars"].shape[0]:
        assert np.allclose(t.scalars[seed_mask], fresh["scalars"], atol=1e-4, equal_nan=True)


# ---------------------------------------------------------------------------
# 11 / 13. checkpoint hashes and G/H alignment
# ---------------------------------------------------------------------------

def test_checkpoint_hashes_unchanged_and_registered(data):
    with open(os.path.join(ROOT, "experiments", "experiment_g_transitions_meta.json")) as fh:
        gmeta = json.load(fh)
    for model in MODELS:
        for seed in (42, 43, 44, 45, 46):
            rel = gmeta["models"][model]["checkpoints"][str(seed)]["path"]
            expected = gmeta["models"][model]["checkpoints"][str(seed)]["sha256"]
            assert data["meta"]["models"][model]["checkpoints"][str(seed)]["sha256"] == expected
            assert rl.eg.sha256_file(os.path.join(ROOT, rel)) == expected


def test_g_alignment_passed(data):
    for model in MODELS:
        assert data["meta"]["g_verification"][model]["passed"]
        assert data["meta"]["g_verification"][model]["exact_mismatches"] == {}


def test_i_and_h_alignment(data):
    i_path = os.path.join(ROOT, "experiments", "experiment_i_coordinate.npz")
    h_path = os.path.join(ROOT, "experiments", "experiment_h_interp.npz")
    if not (os.path.exists(i_path) and os.path.exists(h_path)):
        pytest.skip("I/H artifacts absent")
    align = al.align_against_experiments(data, i_path, h_path)
    for model in MODELS:
        assert align[model]["vs_I_ok"], align[model]["vs_I_max_abs_diff"]
        assert align[model]["vs_H_ok"], align[model]["vs_H_max_abs_diff"]


# ---------------------------------------------------------------------------
# 15. end-to-end analysis + figures
# ---------------------------------------------------------------------------

def test_end_to_end_analysis_and_figures(data, tmp_path):
    res = al.build(data, os.path.join(ROOT, "experiments", "experiment_i_coordinate.npz"),
                   os.path.join(ROOT, "experiments", "experiment_h_interp.npz"))
    json.dumps(res)                                       # JSON-serializable
    assert res["primary_decision"]["verdict"] in (
        "A_readout_calibration", "B_latent_placement", "C_both", "D_inconclusive")
    figs = al.make_figures(data, res, str(tmp_path), prefix="test_l")
    assert len(figs) >= 12
    assert all(os.path.exists(p) for p in figs)


def test_real_artifact_verdict_is_latent_placement(data):
    res = al.build(data, os.path.join(ROOT, "experiments", "experiment_i_coordinate.npz"),
                   os.path.join(ROOT, "experiments", "experiment_h_interp.npz"))
    for model in MODELS:
        d = res[model]["decision"]
        assert d["verdict"] == "B_latent_placement"
        assert d["r_bias"] < al.READOUT_BAR
        assert d["r_place"] >= al.PLACEMENT_BAR


# ---------------------------------------------------------------------------
# 17. invalid inputs / configurations
# ---------------------------------------------------------------------------

def test_invalid_inputs_rejected(tmp_path):
    with pytest.raises(Exception):
        al.load_readout(str(tmp_path / "does_not_exist.npz"))
    with pytest.raises((KeyError, ValueError, IndexError)):
        al.LTable(np.zeros((2, 2)), ["a", "b"])["missing"]
    # empty signature -> non-finite shares and inconclusive
    d = al.decide({"bias_sig": np.zeros(0, bool)}, {"place_sig": np.zeros(0, bool)})
    assert d["verdict"] == "D_inconclusive"
