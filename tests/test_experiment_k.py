"""Tests for Experiment K — coordinate-preserving gate-balance ablation.

Covers the required checks:
 1. kappa=1 is exact identity
 2. sharpening remains in [0,1]
 3. f<0.5 remains <0.5
 4. f>0.5 remains >0.5
 5. f=0.5 remains 0.5 (exact)
 6. monotonicity of the remap
 7. checkpoint hashes unchanged
 8. no parameters modified
 9. no gradients
10. deterministic rollout
11. kappa=1 baseline reproduction of the frozen forward
12. rescue/regression accounting is internally consistent
13. metric accounting matches the frozen linear readout / gate-shape conventions
14. synthetic end-to-end analysis
15. figure generation
16. invalid configuration rejected
plus coordinate-identity preservation (element-wise commutes with permutation).
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_k as ak  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_j as rj  # noqa: E402
import run_experiment_k as rk  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Transform properties
# ---------------------------------------------------------------------------

def test_kappa_one_is_exact_identity():
    f = torch.rand(64, rk.STATE_DIM)
    out = rk.tighten(f, 1.0)
    assert torch.equal(out, f)
    assert torch.allclose(rk.make_commit_fn("sharpen", 1.0)(f), f, atol=0, rtol=0)


def test_sharpening_within_unit_interval():
    f = torch.cat([torch.rand(5000), torch.tensor([0.0, 1.0, 1e-9, 1 - 1e-9, 0.5])])
    for k in rk.KAPPA_GRID:
        out = rk.tighten(f, k)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert torch.isfinite(out).all()


def test_sign_of_deviation_from_half_preserved():
    x = torch.linspace(1e-4, 1 - 1e-4, 20001)
    y = rk.tighten(x, 2.0)
    assert bool((y[x < 0.5] < 0.5).all())
    assert bool((y[x > 0.5] > 0.5).all())
    # saturation: already-extreme gates become strictly more extreme
    lo = torch.tensor([0.01, 0.001, 0.9, 0.99])
    yl = rk.tighten(lo, 2.0)
    assert bool((yl[:2] < lo[:2]).all()) and bool((yl[2:] > lo[2:]).all())


def test_half_is_fixed_point():
    for k in rk.KAPPA_GRID:
        out = rk.tighten(torch.tensor([0.5]), k)
        assert float(out[0]) == 0.5


def test_monotonicity():
    x = torch.linspace(1e-5, 1 - 1e-5, 50001)
    for k in rk.KAPPA_GRID:
        y = rk.tighten(x, k)
        assert bool((y[1:] >= y[:-1] - 1e-7).all())


def test_coordinate_identity_preserved_under_permutation():
    f = torch.rand(32, rk.STATE_DIM)
    perm = torch.randperm(rk.STATE_DIM)
    for k in rk.KAPPA_GRID:
        assert torch.allclose(rk.tighten(f, k)[:, perm], rk.tighten(f[:, perm], k), atol=0, rtol=0)


def test_transform_is_not_global_multiplication():
    # f -> gamma f would scale the mean; sharpening does not, and moves the mean toward
    # whichever side of 0.5 dominates. Verify the two are not proportional on a sample.
    torch.manual_seed(0)
    f = torch.sigmoid(torch.randn(5000, rk.STATE_DIM))
    a = rk.tighten(f, 2.0)
    b = (1.5 * f).clamp(0, 1)
    assert not torch.allclose(a, b, atol=1e-3)


# ---------------------------------------------------------------------------
# Invalid configuration
# ---------------------------------------------------------------------------

def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        rk.tighten(torch.rand(4, 3), 0.0)
    with pytest.raises(ValueError):
        rk.tighten(torch.rand(4, 3), -1.0)
    with pytest.raises(ValueError):
        rk.make_commit_fn("scale", 2.0)          # K has exactly one family
    with pytest.raises(ValueError):
        rk.make_commit_fn("sharpen", 0.0)
    with pytest.raises(ValueError):
        rk.validate_kappas([1.0, 3.0])           # kappa outside the pre-registered grid
    assert rk.validate_kappas([1.0, 2.0]) == [1.0, 2.0]
    assert rk.KAPPA_GRID == [1.0, 2.0]           # pre-registered two-condition grid


# ---------------------------------------------------------------------------
# Checkpoint integrity
# ---------------------------------------------------------------------------

def test_checkpoint_hashes_unchanged():
    with open(os.path.join(ROOT, "experiments", "experiment_g_transitions_meta.json")) as fh:
        meta = json.load(fh)
    for model in ("E_candidate", "F_combined"):
        for seed in (42, 43, 44, 45, 46):
            rel = meta["models"][model]["checkpoints"][str(seed)]["path"]
            expected = meta["models"][model]["checkpoints"][str(seed)]["sha256"]
            actual = rj.sha256_file(os.path.join(ROOT, rel))
            assert actual == expected


# ---------------------------------------------------------------------------
# Tiny sequences / real checkpoint fixture
# ---------------------------------------------------------------------------

def _tiny_dataset(B=4, T=10):
    task = rk.gr.build_task(T)
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


@pytest.fixture(scope="module")
def loaded_model():
    ckpt = os.path.join(ROOT, "experiments", "checkpoints")
    return eg.load_checkpoint(ckpt, "E_candidate", 42)["model"]


def test_baseline_kappa_one_reproduces_frozen_forward(loaded_model):
    ds = _tiny_dataset()
    frozen = loaded_model.forward_with_states(ds.inputs, return_components=True)
    roll = rj.rollout(loaded_model, ds.inputs, rk.make_commit_fn("sharpen", 1.0))
    fs = torch.stack(frozen.states, dim=1); fc = torch.stack(frozen.candidates, dim=1)
    fg = torch.stack(frozen.gates, dim=1)
    assert torch.equal(fs, roll["states"])          # exact identity at kappa=1
    assert torch.equal(fc, roll["cands"])
    assert torch.equal(fg, roll["gates"])
    assert torch.equal(fg, roll["gates_mod"])       # gates_mod == gates at kappa=1
    assert torch.equal(frozen.terminal_logits.argmax(-1), roll["term_logits"].argmax(-1))
    check = rj._verify_baseline_reproduction(loaded_model, ds)
    assert check["passed"]


def test_deterministic_no_grad_no_param_change(loaded_model):
    ds = _tiny_dataset()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    cf = rk.make_commit_fn("sharpen", 2.0)
    r1 = rj.rollout(loaded_model, ds.inputs, cf)
    r2 = rj.rollout(loaded_model, ds.inputs, cf)
    assert torch.equal(r1["states"], r2["states"])
    assert torch.equal(r1["gates_mod"], r2["gates_mod"])
    for v in r1.values():
        assert not v.requires_grad and v.grad_fn is None
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])


# ---------------------------------------------------------------------------
# Metric accounting
# ---------------------------------------------------------------------------

def test_metric_accounting_matches_linear_readout_and_gate_conventions(loaded_model):
    ds = _tiny_dataset()
    x = rk.test_transition_data(loaded_model, ds, rk.make_commit_fn("sharpen", 1.0))
    # decision-margin algebra
    assert np.allclose(x["dm_commit"], x["m_commit"] - x["m_prev"], atol=1e-6)
    assert np.allclose(x["dm_cand"], x["m_cand"] - x["m_prev"], atol=1e-6)
    assert np.allclose(x["margin_shortfall"], x["m_cand"] - x["m_commit"], atol=1e-6)
    cc = (x["cand_ok"] > 0.5) & (x["gen_general"] > 0.5)
    if cc.any():
        assert np.all(x["dm_cand"][cc] >= -1e-6)
        rho = x["rho"][cc]; fin = np.isfinite(rho)
        assert np.allclose(rho[fin], x["dm_commit"][cc][fin] / x["dm_cand"][cc][fin], atol=1e-6)
    # kappa=1 identity: no gate change anywhere
    ts = x["gate_shape"]["transition_stats"]
    assert ts["abs_change"]["mean"] == 0.0 and ts["signed_change"]["mean"] == 0.0
    assert x["gate_shape"]["all_gates"]["before"]["mean"] == x["gate_shape"]["all_gates"]["after"]["mean"]
    # bounded weighted gates
    for key in ("gw_pos", "gw_neg", "gw_all", "gw_mag"):
        v = x[key][np.isfinite(x[key])]
        assert np.all(v >= -1e-6) and np.all(v <= 1 + 1e-6)


def test_sharpen_changes_gate_shape_but_preserves_coordinate_identity(loaded_model):
    ds = _tiny_dataset()
    x = rk.test_transition_data(loaded_model, ds, rk.make_commit_fn("sharpen", 2.0))
    ts = x["gate_shape"]["transition_stats"]
    assert ts["abs_change"]["mean"] > 0.0
    # mean transfer does not simply increase everywhere: signed change is near zero on average
    assert abs(ts["signed_change"]["mean"]) < 0.1


# ---------------------------------------------------------------------------
# Rescue / regression accounting
# ---------------------------------------------------------------------------

def test_rescue_accounting_consistency():
    base_comm = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], bool)
    new_comm = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1], bool)
    n = 10
    base_primary = np.ones(n, bool)
    base_td = {"comm_ok": base_comm.astype(float), "dm_commit": np.zeros(n)}
    td = {"comm_ok": new_comm.astype(float), "dm_commit": np.zeros(n),
          "competitor_changed": np.zeros(n)}
    strata = {"low": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool),
              "mid": np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0], bool),
              "high": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], bool), "edges": np.array([1.0, 2.0])}
    suff = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    rec = rk._rescue_accounting(td, base_td, base_primary, strata, suff)
    assert rec["n_base_fail"] == 4 and rec["n_base_succ"] == 6
    assert rec["rescued"] == 2 and rec["regressed"] == 2 and rec["net_recovery"] == 0
    assert sum(rec["by_stratum"][s]["n"] for s in ("low", "mid", "high")) == n
    assert rec["offpath"]["n_sufficient"] == 4


def test_seq_rescue_consistency():
    base = {"seq_em": np.array([1, 0, 0, 1, 0], float)}
    td = {"seq_em": np.array([1, 1, 0, 0, 0], float)}
    out = rk._seq_rescue(td, base)
    assert out["n_base_fail_seq"] == 3 and out["n_base_succ_seq"] == 2
    assert out["rescued_seq"] == 1 and out["broken_seq"] == 1


# ---------------------------------------------------------------------------
# Paired seed alignment
# ---------------------------------------------------------------------------

def test_paired_and_index_alignment():
    recs = [{"model": "m", "kappa": 2.0, "seed": s, "horizons": {"10": {"em": 0.1 * s}}} for s in (46, 42, 44, 43, 45)]
    idx = ak.index(recs)
    assert [r["seed"] for r in idx[("m", 2.0)]] == [42, 43, 44, 45, 46]
    out = ak.paired(np.array([0.5, 0.52, 0.51, 0.49, 0.50]), np.array([0.4, 0.41, 0.42, 0.39, 0.40]))
    assert out["sign_positive"] == 5 and out["cohens_dz"] > 3 and out["paired_t_p"] < 0.01


# ---------------------------------------------------------------------------
# Synthetic end-to-end analysis + figures
# ---------------------------------------------------------------------------

def _gate_shape_synth(kappa):
    before_hist = {"edges": list(np.linspace(0, 1, 51)), "counts": [100] * 50, "n": 5000}
    after_hist = {"edges": list(np.linspace(0, 1, 51)), "counts": [100] * 50, "n": 5000}
    delta_hist = {"edges": list(np.linspace(-0.5, 0.5, 51)), "counts": [100] * 50, "n": 5000}
    abs_hist = {"edges": list(np.linspace(0, 0.5, 51)), "counts": [100] * 50, "n": 5000}
    stats = {k: {"mean": 0.0, "median": 0.0, "sd": 0.0, "p05": 0.0, "p95": 0.0, "n": 100}
             for k in ("before_mean", "after_mean", "before_median", "after_median",
                       "before_entropy", "after_entropy", "delta_mean", "delta_median",
                       "delta_entropy", "abs_change", "signed_change", "transfer_change")}
    if kappa == 2.0:
        stats["abs_change"]["mean"] = 0.03
        stats["signed_change"]["mean"] = -0.002
        stats["transfer_change"]["mean"] = -0.004
        stats["delta_entropy"]["mean"] = -0.05
    return {
        "all_gates": {"before": {"mean": 0.36, "median": 0.35, "std": 0.2, "entropy": 0.9,
                                 "frac_lt_025": 0.3, "frac_025_05": 0.3, "frac_05_075": 0.3,
                                 "frac_ge_075": 0.1, "n": 1000},
                      "after": {"mean": 0.355 if kappa == 2.0 else 0.36, "median": 0.345 if kappa == 2.0 else 0.35,
                                "std": 0.2, "entropy": 0.85 if kappa == 2.0 else 0.9,
                                "frac_lt_025": 0.32 if kappa == 2.0 else 0.3, "frac_025_05": 0.29,
                                "frac_05_075": 0.30, "frac_ge_075": 0.10, "n": 1000}},
        "coord_signed_mean_per_dim": [0.001] * 212,
        "coord_delta_hist": delta_hist, "coord_abs_hist": abs_hist,
        "transition_stats": stats, "gate_value_hist": {"before": before_hist, "after": after_hist},
        "entropy_hist": {"before": before_hist, "after": after_hist},
        "meangate_hist": {"before": before_hist, "after": after_hist},
    }


def _synthetic_records():
    seeds = [42, 43, 44, 45, 46]
    records = []
    for model in (ak.PRIMARY_MODEL, ak.CONTROL_MODEL):
        for kappa in (1.0, 2.0):
            for seed in seeds:
                jitter = 0.001 * (seed - 42)
                # sequence EM: locally improved margins, sequence neutral/slightly down at kappa=2
                t10 = 0.755 - (0.005 if kappa == 2.0 else 0.0) + jitter
                t20 = 0.407 - (0.002 if kappa == 2.0 else 0.0) + jitter
                net = 0 if kappa == 1.0 else -8
                r = {"model": model, "seed": seed, "family": rk.PRIMARY_FAMILY, "kappa": kappa,
                     "cand_margin_strata_edges": [1.0, 5.0], "baseline_reproduction": None,
                     "param_count": 226358, "trainable_param_count": 226358,
                     "rollout_time_s": 0.5 + 0.05 * (kappa == 2.0),
                     "horizons": {str(L): {"em": t10 - 0.01 * abs(L - 10), "per_reg": t10}
                                  for L in rk.HORIZONS},
                     "test": {"comm_written_mean": 0.9, "cand_written_mean": 0.95,
                              "comm_full_mean": 0.85, "cand_full_mean": 0.9,
                              "first_real_error_rate": 0.3, "first_strict_error_rate": 0.4,
                              "gate_mean": 0.36, "gate_scaled_mean": 0.355 if kappa == 2 else 0.36,
                              "gate_entropy": 0.9, "gate_scaled_entropy": 0.85 if kappa == 2 else 0.9},
                     "by_final_op": {"SET": 0.836, "SWAP": 0.778,
                                     "ADD_no_wrap": 0.709, "ADD_wrap": 0.493},
                     "controlled": {k: {"terminal_em": 0.5, "committed_dest_acc_at_add": 0.5,
                                        "candidate_dest_acc_at_add": 0.6}
                                    for k in ("isolated_add", "add_then_set", "repeated_add",
                                              "add_no_wrap", "add_wrap")},
                     "gate_shape": _gate_shape_synth(kappa),
                     "primary": {"n": 1000, "comm_acc": 0.855 + (0.005 if kappa == 2 else 0.0),
                                 "cand_acc": 1.0, "rho_median": 0.884 + (0.016 if kappa == 2 else 0.0),
                                 "m_prev_median": -10.0, "m_cand_median": 8.5,
                                 "m_commit_median": 6.6, "dm_commit_mean": 15.0,
                                 "margin_shortfall_median": 1.84, "gw_pos_mean": 0.69 + (0.01 if kappa == 2 else 0.0),
                                 "gw_neg_mean": 0.28 - (0.004 if kappa == 2 else 0.0),
                                 "gw_all_mean": 0.6, "gw_mag_mean": 0.34,
                                 "competitor_changed_mean": 0.66},
                     "rescue": {"n_base_succ": 800, "n_base_fail": 120,
                                "rescued": max(0, net + 3) if kappa == 2 else 0,
                                "regressed": max(0, 3 - min(net, 3)) if kappa == 2 else 0,
                                "rescue_rate": 0.05, "regression_rate": 0.01,
                                "net_recovery": net, "net_recovery_rate": net / 1000,
                                "by_stratum": {s: {"n": 333, "n_base_fail": 40, "rescued": 2, "regressed": 1,
                                                   "rescue_rate": 0.05, "regression_rate": 0.01,
                                                   "comm_acc": 0.85} for s in ("low", "mid", "high")},
                                "offpath": {"n_sufficient": 50, "n_base_fail_sufficient": 40,
                                            "rescued_sufficient": 2, "rescue_rate_sufficient": 0.05,
                                            "competitor_changed_rate_sufficient_fail": 0.7,
                                            "dm_commit_mean_sufficient_fail": 0.0,
                                            "dm_commit_mean_sufficient_fail_baseline": 0.0}},
                     "seq_rescue": {"n_seq": 1200, "n_base_fail_seq": 300, "n_base_succ_seq": 900,
                                    "rescued_seq": max(0, net), "broken_seq": max(0, -net),
                                    "rescue_rate_seq": 0.02, "break_rate_seq": 0.01}}
                records.append(r)
    meta = {"preregistration": rk.PREREGISTRATION, "n_records": len(records), "seeds": seeds,
            "checkpoints": {}, "wall_clock_s": 1.0}
    return records, meta


def test_synthetic_end_to_end_and_figures(tmp_path):
    records, meta = _synthetic_records()
    res, summaries, idx = ak.build(records, meta, {"1.0": {"t10_em": 0.755, "t20_em": 0.407},
                                                   "1.1": {"t10_em": 0.7525, "t20_em": 0.4064},
                                                   "1.25": {"t10_em": 0.7463, "t20_em": 0.3924}})
    json.dumps(res)  # JSON-serializable
    assert res["decision_E_primary"]["verdict"] in (
        "A_sequence_beneficial_mechanistically_supportive", "B_locally_beneficial_sequence_neutral",
        "C_no_meaningful_benefit", "D_harmful")
    assert (ak.PRIMARY_MODEL, ak.BASELINE_KAPPA) in summaries
    figs = ak.make_figures(idx, summaries, res["decision_E_primary"], res["j_historical_reference"],
                           str(tmp_path), prefix="test_k")
    assert len(figs) >= 14
    assert all(os.path.exists(p) for p in figs)


def test_verdict_b_when_margins_move_but_sequence_neutral():
    records, meta = _synthetic_records()
    res, summaries, idx = ak.build(records, meta, {})
    d = res["decision_E_primary"]
    # synthetic: margins/rho improve for kappa=2, sequence EM flat-to-slightly-down, net rescue negative
    assert d["verdict"] == "B_locally_beneficial_sequence_neutral"
    assert d["sub_findings"]["local_mechanism_moves"]
    assert d["sub_findings"]["no_beneficial_sequence_level_effect"]
