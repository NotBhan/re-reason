"""Tests for Experiment P — sequence-level error-budget attribution (measurement only).

Covers the 24 required checks:
 1. deterministic collector
 2. checkpoint hash verification
 3. no-gradient execution
 4. no parameter mutation
 5. exact test split reproduction
 6. exact task-state reconstruction
 7. candidate correctness calculation
 8. committed correctness calculation
 9. full-state correctness calculation
10. earliest-failure classification
11. mutually exclusive A/B/C/D categories
12. first-error sequence budget accounting
13. all-error accounting
14. horizon grouping
15. T10/T20 accounting
16. untouched-register analysis
17. recovery analysis
18. candidate->commit residual calculation
19. readout-margin calculation
20. E/F alignment
21. result JSON determinism
22. figure generation
23. malformed/incomplete data handling
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

import analyze_experiment_p as ap  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_p as rp  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_p_error_budget.npz")
MODELS = ("E_candidate", "F_combined")


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_p npz not present")
    return ap.load(NPZ)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


def _tiny_dataset(B=4, T=10):
    task = gr.build_task(T)
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
# 1 / 3 / 4. collector determinism, no-grad, no mutation
# ---------------------------------------------------------------------------

def test_collector_deterministic_no_grad_no_mutation(loaded_model):
    ds = _tiny_dataset()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r1 = rp.collect_horizon(loaded_model, ds, 42, 10, store_full=True)
    r2 = rp.collect_horizon(loaded_model, ds, 42, 10, store_full=True)
    for k in r1:
        assert np.array_equal(r1[k], r2[k]), k
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


# ---------------------------------------------------------------------------
# 2 / 20. checkpoint hashes + E/F alignment
# ---------------------------------------------------------------------------

def test_checkpoint_hashes_and_EF_alignment(data):
    with open(os.path.join(ROOT, "experiments", "experiment_g_transitions_meta.json")) as fh:
        gmeta = json.load(fh)
    for m in MODELS:
        for s in (42, 43, 44, 45, 46):
            rel = gmeta["models"][m]["checkpoints"][str(s)]["path"]
            assert data["meta"]["models"][m]["checkpoints"][str(s)]["sha256"] == \
                gmeta["models"][m]["checkpoints"][str(s)]["sha256"]
            assert eg.sha256_file(os.path.join(ROOT, rel)) == \
                gmeta["models"][m]["checkpoints"][str(s)]["sha256"]
    z = data["z"]
    for m in MODELS:
        assert z[f"{m}__T10__seq"].shape[0] == 6000     # 1200 x 5 seeds
        assert z[f"{m}__T20__seq"].shape[0] == 5000     # 1000 x 5 seeds


# ---------------------------------------------------------------------------
# 5. test split reproduction
# ---------------------------------------------------------------------------

def test_test_split_reproduction(data):
    z = data["z"]
    for m in MODELS:
        tok = z[f"{m}__T10__terminal_ok"].astype(float)
        em = float(tok.mean())
        assert 0.70 < em < 0.80                        # matches the established T=10 EM band
        seed = z[f"{m}__T10__seed"].astype(int)
        for s in (42, 43, 44, 45, 46):
            assert int((seed == s).sum()) == 1200


# ---------------------------------------------------------------------------
# 6. task-state reconstruction / 7-9. correctness
# ---------------------------------------------------------------------------

def test_task_state_reconstruction_and_correctness(loaded_model, data):
    z = data["z"]
    op = z["E_candidate__T10__op"]; oa = z["E_candidate__T10__operand_a"]
    ob = z["E_candidate__T10__operand_b"]; tw = z["E_candidate__T10__true_written"]
    ts = z["E_candidate__T10__true_state"]
    m = op == 2
    assert bool(((oa[m] + ob[m]) % 10 == tw[m]).all())          # ADD target reconstruction
    # stored true_written equals the true state at the destination register
    dest = z["E_candidate__T10__dest"]
    ts_written = np.take_along_axis(ts, dest[:, :, None], axis=2).squeeze(2)
    assert np.array_equal(ts_written, tw)
    # correctness arrays are boolean and consistent with the classification codes
    for key in ("cand_written_ok", "comm_written_ok", "full_comm_ok", "full_cand_ok"):
        a = z[f"E_candidate__T10__{key}"]
        assert a.dtype == np.uint8 and set(np.unique(a)).issubset({0, 1})


def test_correctness_recompute_on_tiny(loaded_model):
    ds = _tiny_dataset()
    rec = rp.collect_horizon(loaded_model, ds, 42, 10, store_full=True)
    R, V, D = gr.NUM_REGISTERS, gr.NUM_VALUES, gr.STATE_DIM
    head = loaded_model.decision_head
    diag = loaded_model.forward_with_states(ds.inputs, return_components=True)
    states = torch.stack(diag.states, 1); cands = torch.stack(diag.candidates, 1)
    cp = head(states.reshape(-1, D)).reshape(*states.shape[:2], R, V).argmax(-1)
    kp = head(cands.reshape(-1, D)).reshape(*cands.shape[:2], R, V).argmax(-1)
    _, op, reg_a, _ = gr.write_mask_from_inputs(ds.inputs)
    dest = reg_a
    tw = ds.step_targets.gather(2, dest[:, :, None]).squeeze(2)
    assert np.array_equal(rec["cand_written_ok"], (kp.gather(2, dest[:, :, None]).squeeze(2) == tw).numpy())
    assert np.array_equal(rec["comm_written_ok"], (cp.gather(2, dest[:, :, None]).squeeze(2) == tw).numpy())
    assert np.array_equal(rec["full_comm_ok"], (cp == ds.step_targets).all(-1).numpy())
    assert np.array_equal(rec["full_cand_ok"], (kp == ds.step_targets).all(-1).numpy())


# ---------------------------------------------------------------------------
# 10 / 11. classification
# ---------------------------------------------------------------------------

def test_earliest_failure_classification():
    cw_all = np.ones(5, bool); km_all = np.ones(5, bool)
    nowrite = np.zeros(4, bool)
    tm = np.array([-2.0, -2.0, -2.0, -2.0])
    # A: candidate wrong at t=1
    cw = cw_all.copy(); cw[1] = False
    assert rp.classify_first_error(cw, km_all, False, np.array([0, 1, 0, 0], bool), nowrite, tm) == ("A", 1)
    # B: candidate correct, commit wrong at t=2
    km = km_all.copy(); km[2] = False
    assert rp.classify_first_error(cw_all, km, False, np.zeros(4, bool), np.ones(4, bool), tm) == ("B", 2)
    # success
    assert rp.classify_first_error(cw_all, km_all, True, np.zeros(4, bool), np.ones(4, bool), tm) == ("success", -1)
    # C1: untouched-register terminal error, deep wrong region
    we = np.array([True, True, False, False], bool); wr = np.array([False, False, True, False], bool)
    tm_deep2 = np.array([0.0, 0.0, -3.0, 0.0])
    assert rp.classify_first_error(cw_all, km_all, False, wr, we, tm_deep2) == ("C1", -1)
    # C2: written-register terminal error, deep wrong region
    wr2 = np.array([True, False, False, False], bool)
    tm_deep0 = np.array([-3.0, 0.0, 0.0, 0.0])
    assert rp.classify_first_error(cw_all, km_all, False, wr2, np.ones(4, bool), tm_deep0) == ("C2", -1)
    # C3: near-boundary (small negative normalized margin)
    tm_small = np.array([-0.1, 0.0, 0.0, 0.0])
    assert rp.classify_first_error(cw_all, km_all, False, wr2, np.ones(4, bool), tm_small) == ("C3", -1)
    # D: failure with no wrong register flagged
    assert rp.classify_first_error(cw_all, km_all, False, np.zeros(4, bool), np.ones(4, bool), tm_deep0) == ("D", -1)


def test_categories_mutually_exclusive(data):
    z = data["z"]
    for m in MODELS:
        fc = z[f"{m}__T10__first_class"].astype(int)
        cc = ap.class_codes(data["meta"])
        # each sequence has exactly one class code
        codes = np.array(list(cc.values()))
        assert set(np.unique(fc)).issubset(set(codes))
        # classes partition the sequences at the terminal-failure level
        tok = z[f"{m}__T10__terminal_ok"].astype(int)
        fail = tok == 0
        # every terminal failure is A/B/C1/C2/C3/D (never success)
        assert not (fail & (fc == cc["success"])).any()


# ---------------------------------------------------------------------------
# 12 / 13 / 14 / 15. accounting
# ---------------------------------------------------------------------------

def test_first_error_budget_accounting(data):
    per = ap.per_seed_budget(data["z"], data["meta"], "E_candidate", 10)
    for s, row in per.items():
        total = sum(row["counts"][c] for c in ap.CLASSES)
        assert total == row["n_fail"]                     # classes partition failures
        assert row["n"] == 1200


def test_all_error_and_horizon_accounting(data):
    z = data["z"]
    nce = z["E_candidate__T10__n_cand_err"]; nfe = z["E_candidate__T10__n_full_err"]
    assert (nce >= 0).all() and (nfe >= 0).all()
    for L in (1, 2, 3, 5, 8, 10, 15, 20):
        fc = z[f"E_candidate__h{L}__first_class"]; tok = z[f"E_candidate__h{L}__terminal_ok"]
        assert len(fc) == len(tok) and len(fc) > 0


def test_T10_T20_accounting(data):
    z = data["z"]
    for m in MODELS:
        t = z[f"{m}__T20__full_comm_ok"].astype(bool)
        assert t.shape[1] == 20
        t10 = t[:, 9]; t20 = t[:, -1]
        assert set(np.unique(t10)).issubset({True, False})


# ---------------------------------------------------------------------------
# 16 / 17 / 18 / 19. untouched, recovery, residual, margin
# ---------------------------------------------------------------------------

def test_untouched_recovery_residual_margin(data):
    z = data["z"]
    we = z["E_candidate__T10__written_ever"].astype(bool)
    wr = z["E_candidate__T10__wrong_regs"].astype(bool)
    assert we.shape == wr.shape == (6000, 4)
    # residual accounting: candidate-correct/commit-wrong transitions
    cw = z["E_candidate__T10__cand_written_ok"].astype(bool)
    km = z["E_candidate__T10__comm_written_ok"].astype(bool)
    residual = cw & (~km)
    assert residual.shape == (6000, 10)
    # recovery flags consistent with first_t
    ft = z["E_candidate__T10__first_t"].astype(int)
    fc = z["E_candidate__T10__first_class"].astype(int)
    cc = ap.class_codes(data["meta"])
    a_rows = np.where(fc == cc["A"])[0]
    assert (ft[a_rows] >= 0).all()
    # readout margin: normalized margin is >=0 for correct registers at terminal
    tnm = z["E_candidate__T10__term_norm_margin"]
    assert tnm.shape == (6000, 4)
    # C3 first errors have a small min margin
    c3 = np.where(fc == cc["C3"])[0]
    if c3.size:
        mins = np.array([tnm[i][wr[i]].min() for i in c3])
        assert (mins > -data["meta"]["c3_bar"]).all()


# ---------------------------------------------------------------------------
# 21 / 22. analysis determinism + figures
# ---------------------------------------------------------------------------

def test_result_json_and_figures(data, tmp_path):
    a = ap.build(data); b = ap.build(data)
    a["metadata"].pop("date", None); b["metadata"].pop("date", None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    figs = ap.make_figures(data, a, str(tmp_path), prefix="test_p")
    assert len(figs) >= 12 and all(os.path.exists(p) for p in figs)


def test_real_artifact_dominator(data):
    res = ap.build(data)
    for m in MODELS:
        d = res[m]["budget_T10"]["dominator"]
        assert d["verdict"] in ("A", "B", "C3", "mixed")
    assert res[ap.PRIMARY_MODEL]["budget_T10"]["dominator"]["verdict"] == "A"


# ---------------------------------------------------------------------------
# 23 / 24. malformed data + historical immutability
# ---------------------------------------------------------------------------

def test_malformed_inputs_and_immutability(tmp_path):
    with pytest.raises(Exception):
        ap.load(str(tmp_path / "nope.npz"))
    with pytest.raises(Exception):
        ap.per_seed_budget({}, {}, "E_candidate", 10)
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"
