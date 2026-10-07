"""Tests for the Experiment J controlled inference-time commit-rule intervention.

Covers the 15 required checks:
 1. gamma=1 is identity
 2. clipping is correct
 3. all gate values remain in [0,1]
 4. checkpoint hashes are unchanged
 5. baseline gamma=1 reproduces the frozen forward (numeric + decoded)
 6. same inputs/checkpoint/gamma are deterministic
 7. no gradients are created
 8. no parameters change
 9. rescue/regression accounting is internally consistent
10. margin accounting matches the frozen linear readout
11. paired seed alignment is correct
12. control permutation is deterministic and mean-matched
13. synthetic end-to-end intervention analysis
14. figure generation
15. invalid gamma/configuration is rejected
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_j as aj  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_j as rj  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Commit-rule families
# ---------------------------------------------------------------------------

def test_gamma_one_is_identity():
    f = torch.rand(64, rj.gr.STATE_DIM)
    fn = rj.make_commit_fn("scale", 1.0)
    assert torch.allclose(fn(f), f, atol=0, rtol=0)
    fn2 = rj.make_commit_fn("one_minus", 1.0)
    assert torch.allclose(fn2(f), f, atol=1e-6)


def test_clipping_is_correct():
    f = torch.tensor([[0.1, 0.5, 0.9, 0.0, 1.0]])
    out = rj.make_commit_fn("scale", 2.0)(f)
    assert torch.allclose(out, torch.tensor([[0.2, 1.0, 1.0, 0.0, 1.0]]))


def test_all_families_produce_unit_interval():
    f = torch.rand(32, rj.gr.STATE_DIM)
    perm = rj.make_perm(rj.gr.STATE_DIM, 42)
    for fam in ("scale", "one_minus"):
        for g in rj.GAMMA_GRID:
            out = rj.make_commit_fn(fam, g)(f)
            assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    for g in rj.GAMMA_GRID:
        out = rj.make_commit_fn("scramble", g, perm)(f)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        rj.make_commit_fn("scale", 0.0)
    with pytest.raises(ValueError):
        rj.make_commit_fn("scale", -1.0)
    with pytest.raises(ValueError):
        rj.make_commit_fn("one_minus", 0.0)
    with pytest.raises(ValueError):
        rj.make_commit_fn("scramble", 1.0, None)  # requires a permutation
    with pytest.raises(ValueError):
        rj.make_commit_fn("nope", 1.0)


# ---------------------------------------------------------------------------
# Control permutation
# ---------------------------------------------------------------------------

def test_permutation_deterministic():
    p1 = rj.make_perm(rj.gr.STATE_DIM, 42)
    p2 = rj.make_perm(rj.gr.STATE_DIM, 42)
    assert torch.equal(p1, p2)
    assert sorted(p1.tolist()) == list(range(rj.gr.STATE_DIM))
    assert not torch.equal(p1, rj.make_perm(rj.gr.STATE_DIM, 43))


def test_scramble_is_mean_matched():
    torch.manual_seed(0)
    f = torch.rand(8, rj.gr.STATE_DIM)
    perm = rj.make_perm(rj.gr.STATE_DIM, 42)
    for g in rj.GAMMA_GRID:
        target = (g * f).clamp(0, 1).mean(dim=1)
        scr = rj.make_commit_fn("scramble", g, perm)(f)
        assert torch.allclose(scr.mean(dim=1), target, atol=5e-3)


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
# Tiny sequences / real checkpoint fixtures
# ---------------------------------------------------------------------------

def _tiny_dataset(B=4, T=10):
    task = rj.gr.build_task(T)
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


def test_baseline_gamma_one_reproduces_frozen_forward(loaded_model):
    ds = _tiny_dataset()
    frozen = loaded_model.forward_with_states(ds.inputs, return_components=True)
    roll = rj.rollout(loaded_model, ds.inputs, rj.make_commit_fn("scale", 1.0))
    fs = torch.stack(frozen.states, dim=1); fc = torch.stack(frozen.candidates, dim=1)
    fg = torch.stack(frozen.gates, dim=1)
    assert torch.allclose(fs, roll["states"], atol=1e-6)
    assert torch.allclose(fc, roll["cands"], atol=1e-6)
    assert torch.allclose(fg, roll["gates"], atol=1e-6)
    assert torch.equal(frozen.terminal_logits.argmax(-1), roll["term_logits"].argmax(-1))


def test_deterministic_and_no_grad_and_no_param_change(loaded_model):
    ds = _tiny_dataset()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    cf = rj.make_commit_fn("scale", 1.25)
    r1 = rj.rollout(loaded_model, ds.inputs, cf)
    r2 = rj.rollout(loaded_model, ds.inputs, cf)
    assert torch.equal(r1["states"], r2["states"])
    # no gradients / no grad graph on any returned tensor
    for k, v in r1.items():
        assert not v.requires_grad and v.grad_fn is None
    # parameters unchanged
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])


# ---------------------------------------------------------------------------
# Margin accounting on the frozen linear readout
# ---------------------------------------------------------------------------

def test_margin_accounting_matches_linear_readout(loaded_model):
    ds = _tiny_dataset()
    td = rj.test_transition_data(loaded_model, ds, rj.make_commit_fn("scale", 1.0))
    # dm_commit == m_commit - m_prev, and rho consistent
    assert np.allclose(td["dm_commit"], td["m_commit"] - td["m_prev"], atol=1e-6)
    # a* is a true competitor (a_star != y implicit); dm_cand >= 0 for candidate-correct
    cc = (td["cand_ok"] > 0.5) & (td["gen_general"] > 0.5)
    if cc.any():
        assert np.all(td["dm_cand"][cc] >= -1e-6)
        rho = td["rho"][cc]
        finite = np.isfinite(rho)
        assert np.allclose(rho[finite], td["dm_commit"][cc][finite] / td["dm_cand"][cc][finite], atol=1e-6)


# ---------------------------------------------------------------------------
# Rescue / regression accounting
# ---------------------------------------------------------------------------

def test_rescue_accounting_consistency():
    # 10 baseline candidate-correct transitions: 4 failures, 6 successes
    base_comm = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], bool)
    new_comm = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1], bool)
    # rescues: failures becoming correct = 2 ; regressions: successes becoming wrong = 2
    n = 10
    base_primary = np.ones(n, bool)
    base = {"comm_ok": base_comm.astype(float), "seq_em": np.zeros(4)}
    td = {"comm_ok": new_comm.astype(float)}
    strata = {"low": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool),
              "mid": np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0], bool),
              "high": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], bool), "edges": np.array([1.0, 2.0])}
    suff = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    rec = rj._rescue_accounting(td, base, base_primary, strata, suff)
    assert rec["n_base_fail"] == 4 and rec["n_base_succ"] == 6
    assert rec["rescued"] + rec["regressed"] == 4
    assert rec["net_recovery"] == rec["rescued"] - rec["regressed"]
    # stratum counts partition the primary set
    assert sum(rec["by_stratum"][s]["n"] for s in ("low", "mid", "high")) == n
    assert rec["offpath"]["n_sufficient"] == 4


def test_seq_rescue_consistency():
    base = {"seq_em": np.array([1, 0, 0, 1, 0], float)}
    td = {"seq_em": np.array([1, 1, 0, 0, 0], float)}
    out = rj._seq_rescue(td, base)
    assert out["n_base_fail_seq"] == 3 and out["n_base_succ_seq"] == 2
    assert out["rescued_seq"] == 1 and out["broken_seq"] == 1


# ---------------------------------------------------------------------------
# Paired seed alignment
# ---------------------------------------------------------------------------

def test_paired_and_index_alignment():
    recs = [{"model": "m", "family": "scale", "gamma": 1.0, "seed": s, "x": 0.1 * s} for s in (46, 42, 44, 43, 45)]
    idx = aj.index(recs)
    assert [r["seed"] for r in idx[("m", "scale", 1.0)]] == [42, 43, 44, 45, 46]
    out = aj.paired(np.array([0.5, 0.52, 0.51, 0.49, 0.50]), np.array([0.4, 0.41, 0.42, 0.39, 0.40]))
    assert out["sign_positive"] == 5 and out["cohens_dz"] > 3 and out["paired_t_p"] < 0.01


# ---------------------------------------------------------------------------
# Synthetic end-to-end analysis + figures
# ---------------------------------------------------------------------------

def _synthetic_records():
    seeds = [42, 43, 44, 45, 46]
    records = []
    for model in (aj.PRIMARY_MODEL, aj.CONTROL_MODEL):
        for family in ("scale", "scramble"):
            for gamma in rj.GAMMA_GRID:
                for seed in seeds:
                    jitter = 0.001 * (seed - 42)
                    # sequence EM: flat for g>1, worse for g<1
                    t10 = 0.75 - (0.01 * (1 - gamma)) * (gamma < 1) - 0.02 * max(0.0, gamma - 1.25) * 8 + jitter
                    if family == "scramble":
                        t10 = 0.01 + jitter
                    t20 = t10 - 0.35
                    rho = 0.88 + 0.05 * (gamma - 1) + jitter
                    net = 0 if gamma == 1 else (5 if gamma == 1.1 else -3)
                    r = {"model": model, "seed": seed, "family": family, "gamma": gamma,
                         "cand_margin_strata_edges": [1.0, 5.0],
                         "baseline_reproduction": None,
                         "horizons": {str(L): {"em": t10 - 0.01 * abs(L - 10), "per_reg": t10}
                                      for L in rj.HORIZONS},
                         "test": {"comm_written_mean": 0.9, "cand_written_mean": 0.95,
                                  "comm_full_mean": 0.85, "first_real_error_rate": 0.3,
                                  "gate_mean": 0.35, "gate_scaled_mean": 0.35 * gamma},
                         "by_final_op": {"SET": 0.84 - 0.01 * max(0, gamma - 1) * 5,
                                         "SWAP": 0.78, "ADD_no_wrap": 0.71, "ADD_wrap": 0.49},
                         "controlled": {"repeated_add": {"terminal_em": 0.15, "committed_dest_acc_at_add": 0.5,
                                                         "candidate_dest_acc_at_add": 0.6},
                                        "add_wrap": {"terminal_em": 0.35, "committed_dest_acc_at_add": 0.4,
                                                     "candidate_dest_acc_at_add": 0.48},
                                        "isolated_add": {"terminal_em": 0.66, "committed_dest_acc_at_add": 0.66,
                                                         "candidate_dest_acc_at_add": 0.8},
                                        "add_then_set": {"terminal_em": 0.66, "committed_dest_acc_at_add": 0.66,
                                                         "candidate_dest_acc_at_add": 0.8},
                                        "add_no_wrap": {"terminal_em": 0.6, "committed_dest_acc_at_add": 0.6,
                                                        "candidate_dest_acc_at_add": 0.68}},
                         "primary": {"n": 1000, "comm_acc": 0.855 + 0.005 * (gamma - 1),
                                     "cand_acc": 1.0, "rho_median": rho, "gw_pos_mean": 0.69 + 0.02 * (gamma - 1),
                                     "gw_neg_mean": 0.28, "m_cand_median": 4.0},
                         "rescue": {"n_base_succ": 800, "n_base_fail": 120 + int(0.0),
                                    "rescued": max(0, net + 3), "regressed": max(0, 3 - min(net, 3)),
                                    "rescue_rate": 0.05, "regression_rate": 0.01,
                                    "net_recovery": net, "net_recovery_rate": net / 1000,
                                    "by_stratum": {s: {"n": 333, "n_base_fail": 40, "rescued": 2, "regressed": 1,
                                                       "rescue_rate": 0.05, "regression_rate": 0.01,
                                                       "comm_acc": 0.85} for s in ("low", "mid", "high")},
                                    "offpath": {"n_sufficient": 50, "n_base_fail_sufficient": 40,
                                                "rescued_sufficient": 2, "rescue_rate_sufficient": 0.05}},
                         "seq_rescue": {"n_seq": 1200, "n_base_fail_seq": 300, "n_base_succ_seq": 900,
                                        "rescued_seq": max(0, net), "broken_seq": max(0, -net),
                                        "rescue_rate_seq": 0.02, "break_rate_seq": 0.01}}
                    records.append(r)
    meta = {"preregistration": rj.PREREGISTRATION, "n_records": len(records), "seeds": seeds,
            "checkpoints": {}, "wall_clock_s": 1.0}
    return records, meta


def test_synthetic_end_to_end_and_figures(tmp_path):
    records, meta = _synthetic_records()
    res, summaries, idx = aj.build(records, meta)
    json.dumps(res)  # JSON-serializable
    assert res["decision_E_primary"]["verdict"] in ("A_strong_causal_support", "B_partial_causal_support",
                                                    "C_little_or_no_causal_support", "D_harmful")
    assert (aj.PRIMARY_MODEL, "scale") in summaries
    figs = aj.make_figures(idx, summaries, res["decision_E_primary"], str(tmp_path), prefix="test_j")
    assert len(figs) >= 12
    assert all(os.path.exists(p) for p in figs)


def test_verdict_partial_when_margins_move_but_em_flat():
    records, meta = _synthetic_records()
    res, summaries, idx = aj.build(records, meta)
    d = aj.decide(summaries[(aj.PRIMARY_MODEL, "scale")])
    # synthetic data: margins improve for g>1, EM does not -> partial (B)
    assert d["verdict"] == "B_partial_causal_support"
    assert d["sub_findings"]["no_beneficial_sequence_level_effect"]
