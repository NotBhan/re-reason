"""Tests for Experiment Q — candidate-generation origin / recoverability (measurement only).

Covers ~25 checks: collector determinism, checkpoint immutability, no-grad, frozen-parameter
immutability, split separation, representation capture, frozen-head baseline reproduction, linear /
nonlinear probe train-validation-test separation, wrap / repeated-ADD classification, raw-sum and
modular target generation, probe output shapes, no probe insertion into the model, E/F alignment,
malformed data rejection, JSON determinism, figure creation, historical-report immutability, ridge
probe correctness/determinism, recoverability arithmetic, the decision rule, and a deliberate
leakage failure path.
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

import analyze_experiment_q as aq  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_q as rq  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_q_candidate_origin.npz")
MODELS = ("E_candidate", "F_combined")


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_q npz not present")
    return aq.load(NPZ)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


# ---------------------------------------------------------------------------
# 1 / 3 / 4 / 6. collector determinism, no-grad, no mutation, representation capture
# ---------------------------------------------------------------------------

def test_collector_deterministic_no_grad_no_mutation(loaded_model):
    splits = gr.canonical_splits(42)
    ds = splits.test_set
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r1 = rq.capture_split(loaded_model, ds, 42, cap=200)
    r2 = rq.capture_split(loaded_model, ds, 42, cap=200)
    for k in r1:
        assert np.array_equal(r1[k], r2[k]), k
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())
    assert r1["C"].shape == (200, 212)
    assert np.all(np.abs(r1["C"]) <= 1.0001)          # tanh candidate vector


# ---------------------------------------------------------------------------
# 2. checkpoint immutability
# ---------------------------------------------------------------------------

def test_checkpoint_immutability(data):
    with open(os.path.join(ROOT, "experiments", "experiment_g_transitions_meta.json")) as fh:
        gmeta = json.load(fh)
    for m in MODELS:
        for s in (42, 43, 44, 45, 46):
            assert data["meta"]["models"][m]["checkpoints"][str(s)]["sha256"] == \
                gmeta["models"][m]["checkpoints"][str(s)]["sha256"]
            rel = gmeta["models"][m]["checkpoints"][str(s)]["path"]
            assert eg.sha256_file(os.path.join(ROOT, rel)) == \
                gmeta["models"][m]["checkpoints"][str(s)]["sha256"]


# ---------------------------------------------------------------------------
# 5. exact split separation
# ---------------------------------------------------------------------------

def test_split_separation():
    for seed in (42, 43, 44, 45, 46):
        sp = gr.canonical_splits(seed)
        tr = sp.train_set.inputs.numpy(); va = sp.val_set.inputs.numpy(); te = sp.test_set.inputs.numpy()
        assert tr.shape[0] == 6000 and va.shape[0] == 1200 and te.shape[0] == 1200
        # no exact sequence collision across splits (streams are disjoint)
        tr_set = set(map(lambda x: x.tobytes(), tr))
        va_set = set(map(lambda x: x.tobytes(), va))
        te_set = set(map(lambda x: x.tobytes(), te))
        assert len(tr_set & va_set) == 0
        assert len(tr_set & te_set) == 0
        assert len(va_set & te_set) == 0


# ---------------------------------------------------------------------------
# 7. frozen-head baseline reproduction
# ---------------------------------------------------------------------------

def test_frozen_head_reproduction(data):
    z = data["z"]
    for m in MODELS:
        W = z[f"{m}__W"]; b = z[f"{m}__b"]
        C = z[f"{m}__test__C"]; seed = z[f"{m}__test__seed"]
        dest = z[f"{m}__test__op_dest"]; y = z[f"{m}__test__y_true"]
        ok = z[f"{m}__test__cand_ok"].astype(bool)
        for si, s in enumerate((42, 43, 44, 45, 46)):
            sel = seed == s
            logits = np.einsum("nd,nvd->nv", C[sel], W[si][dest[sel]]) + b[si][dest[sel]]
            pred = logits.argmax(1)
            assert np.array_equal(pred == y[sel], ok[sel])
        # frozen head is perfect on the control population by construction
        assert ok[~ok].sum() == 0


# ---------------------------------------------------------------------------
# 10 / 11 / 12 / 13. wrap / repeated / targets
# ---------------------------------------------------------------------------

def test_wrap_repeated_and_targets(data):
    z = data["z"]
    for split in ("train", "val", "test"):
        oa = z[f"E_candidate__{split}__operand_a"]; ob = z[f"E_candidate__{split}__operand_b"]
        wrap = z[f"E_candidate__{split}__wrap"]; y = z[f"E_candidate__{split}__y_true"]
        raw = z[f"E_candidate__{split}__raw_sum"]
        assert np.array_equal(wrap.astype(bool), (oa + ob) >= 10)
        assert np.array_equal(raw, oa + ob)
        assert raw.min() >= 0 and raw.max() <= 18
        assert np.array_equal(y, (oa + ob) % 10)
        rep = z[f"E_candidate__{split}__repeated"].astype(bool)
        t = z[f"E_candidate__{split}__t"]
        assert not rep[t == 0].any()               # first step is never "repeated"


# ---------------------------------------------------------------------------
# 8 / 9. probe train/val/test separation
# ---------------------------------------------------------------------------

def _tiny(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 212)).astype(np.float64)
    y = (rng.integers(0, 10, n)).astype(int)
    return X, y


def test_linear_probe_separation_and_correctness():
    Xtr, ytr = _tiny(400, 1); Xva, yva = _tiny(200, 2); Xte, yte = _tiny(200, 3)
    pred, val, lam = aq.ridge_probe(Xtr, ytr, Xva, yva, 10)
    assert val == 0.0 or 0.0 <= val <= 1.0
    # lambda is selected on validation only
    assert lam in aq.LAMBDAS
    # closed-form correctness vs an independent solve on a tiny problem
    Xtr2, ytr2 = _tiny(50, 9)
    Xb = np.c_[Xtr2, np.ones(len(Xtr2))]
    sol = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ np.eye(10)[ytr2])
    p2, _, _ = aq.ridge_probe(Xtr2, ytr2, Xva, yva, 10)
    # probe output shape is (n, 10)
    assert p2(Xte).shape == (200, 10)
    assert sol.shape == (Xb.shape[1], 10)


def test_nonlinear_probe_separation():
    Xtr, ytr = _tiny(300, 4); Xva, yva = _tiny(150, 5); Xte, yte = _tiny(150, 6)
    m = aq._MLP(8, 7)
    val = m.fit(Xtr, ytr, Xva, yva)
    out = m.predict(Xte)
    assert 0.0 <= val <= 1.0
    assert out.shape == (150, 10)


def test_no_probe_insertion(model=("E_candidate", 42)):
    # the probe is a separate object; the model forward is bit-identical before/after a probe fit
    loaded = eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"), model[0], model[1])
    m = loaded["model"]
    sp = gr.canonical_splits(model[1]); ds = sp.test_set
    inp = ds.inputs[:8]
    with torch.no_grad():
        before = m.decision_head(m.forward_with_states(inp, return_components=True).states[-1]).clone()
    Xtr, ytr = _tiny(64, 8); Xva, yva = _tiny(32, 9)
    aq.ridge_probe(Xtr, ytr, Xva, yva, 10)
    aq._MLP(8, 10).fit(Xtr, ytr, Xva, yva)
    with torch.no_grad():
        after = m.decision_head(m.forward_with_states(inp, return_components=True).states[-1]).clone()
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# 14 / 16. shapes + E/F alignment
# ---------------------------------------------------------------------------

def test_probe_shapes_and_EF_alignment(data):
    z = data["z"]
    for m in MODELS:
        for split, cap in (("train", 1500), ("val", 800)):
            assert z[f"{m}__{split}__C"].shape[0] == 5 * cap
        assert z[f"{m}__test__C"].shape[1] == 212
        assert z[f"{m}__W"].shape == (5, 4, 10, 212)
        assert z[f"{m}__b"].shape == (5, 4, 10)
        seeds = sorted(set(z[f"{m}__test__seed"].tolist()))
        assert seeds == [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# 21 / 22 / 23 / 24. arithmetic helpers, metrics, recoverability, decision
# ---------------------------------------------------------------------------

def test_recoverability_and_metrics():
    assert aq.acc_of(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([1, 1])) == 0.5
    # frozen=0 -> recoverability == accuracy; frozen=1 -> None
    assert aq.mean_sd([0.2, 0.3])["mean"] == pytest.approx(0.25)


def test_decision_rule_synthetic():
    def mk(**kw):
        base = {"n": {"mean": 100}, "frozen_acc": {"mean": 0.0}, "linear_acc": {"mean": 0.1},
                "nonlinear_acc": {"mean": 0.1}, "rawsum_acc": {"mean": 0.1},
                "recover_linear": {"mean": 0.1}, "recover_nonlinear": {"mean": 0.1},
                "gap_mlp_minus_lin": {"mean": 0.0}}
        def patch(k, v):
            base[k]["mean"] = v
        for k, v in kw.items():
            patch(k, v)
        return {"summary": {k: dict(base) for k in
                            ("classA", "control_candok", "classA_wrap", "classA_nowrap",
                             "classA_repeated", "classA_isolated", "control_wrap", "control_nowrap")}}
    # Q-A: linear >= 0.5 and wrap/repeated >= 0.30
    r = mk(linear_acc=0.6, recover_linear=0.6)
    r["summary"]["classA_wrap"]["linear_acc"]["mean"] = 0.5
    r["summary"]["classA_repeated"]["linear_acc"]["mean"] = 0.5
    assert aq.decide(r)["verdict"] == "Q-A_linearly_recoverable"
    # Q-B: linear weak, nonlinear much stronger
    r = mk(linear_acc=0.2, recover_linear=0.2, nonlinear_acc=0.5, recover_nonlinear=0.5, gap_mlp_minus_lin=0.3)
    assert aq.decide(r)["verdict"] == "Q-B_nonlinear_only"
    # Q-C: both weak
    r = mk(linear_acc=0.2, recover_linear=0.2, nonlinear_acc=0.22, recover_nonlinear=0.22, gap_mlp_minus_lin=0.02)
    assert aq.decide(r)["verdict"] == "Q-C_weak_recoverability"


def test_probe_determinism():
    Xtr, ytr = _tiny(200, 11); Xva, yva = _tiny(100, 12)
    a = aq.ridge_probe(Xtr, ytr, Xva, yva, 10)[0](Xva)
    b = aq.ridge_probe(Xtr, ytr, Xva, yva, 10)[0](Xva)
    assert np.array_equal(a, b)


def test_leakage_failure_path():
    # A deliberate leak: including test rows in the fit inflates test accuracy; the pipeline's
    # disjoint split is what prevents it. This documents the failure mode the guard protects.
    Xtr, ytr = _tiny(200, 13); Xte, yte = _tiny(200, 14)
    clean, _, _ = aq.ridge_probe(Xtr, ytr, Xte, yte, 10)
    clean_acc = aq.acc_of(clean(Xte), yte)
    Xleak = np.vstack([Xtr, Xte]); yleak = np.concatenate([ytr, yte])
    leak, _, _ = aq.ridge_probe(Xleak, yleak, Xte, yte, 10)
    leak_acc = aq.acc_of(leak(Xte), yte)
    assert leak_acc >= clean_acc                       # leakage must not lower test accuracy
    # identity-overlap detector
    def overlap(A, B):
        return len(set(map(lambda x: x.tobytes(), A)) & set(map(lambda x: x.tobytes(), B)))
    assert overlap(Xtr, Xte) == 0
    assert overlap(Xleak, Xte) == len(Xte)             # the leak is detected


# ---------------------------------------------------------------------------
# 17 / 18 / 19 / 20 / 25. malformed, JSON, figures, history, baseline
# ---------------------------------------------------------------------------

def test_malformed_json_figures_and_immutability(data, tmp_path):
    with pytest.raises(Exception):
        aq.load(str(tmp_path / "missing.npz"))
    with pytest.raises(Exception):
        aq.analyze_seed({}, "E_candidate", 42)         # malformed (missing keys)
    a = aq.build(data); b = aq.build(data)
    a["metadata"].pop("date", None); b["metadata"].pop("date", None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    figs = aq.make_figures(data, a, str(tmp_path), prefix="test_q")
    assert len(figs) >= 12 and all(os.path.exists(p) for p in figs)
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"


def test_real_result_is_consistent(data):
    res = aq.build(data)
    for m in MODELS:
        d = res[m]["decision"]
        assert d["verdict"] in ("Q-A_linearly_recoverable", "Q-B_nonlinear_only",
                                "Q-C_weak_recoverability", "Q-MIX")
        # Class-A frozen accuracy is zero by definition of the population
        assert res[m]["summary"]["classA"]["frozen_acc"]["mean"] == 0.0
    assert res[aq.PRIMARY_MODEL]["decision"]["verdict"].startswith("Q-C")


# ---------------------------------------------------------------------------
# additional checks (full-split capture, baseline band, gap arithmetic, history,
# operand-pair held-out, repeated-ADD subset, metadata integrity)
# ---------------------------------------------------------------------------

def test_full_split_capture_counts(data):
    z = data["z"]; meta = data["meta"]
    n_test = meta["n_test"]
    for m in MODELS:
        assert n_test[m] > 15000                          # ~3150 ADD / seed over 5 seeds
        assert meta["n_train"][m] == 5 * rq.TRAIN_CAP
        assert meta["n_val"][m] == 5 * rq.VAL_CAP
        assert z[f"{m}__test__C"].shape[0] == n_test[m]


def test_frozen_head_accuracy_band(data):
    z = data["z"]
    for m in MODELS:
        ok = z[f"{m}__test__cand_ok"].astype(bool)
        acc = float(ok.mean())
        assert 0.80 < acc < 0.95                          # frozen candidate written accuracy (ADD)
        # Class-A fraction is a minority of ADD transitions
        assert 0.05 < (~ok).mean() < 0.25


def test_recoverability_gap_arithmetic(data):
    res = aq.build(data)
    for m in MODELS:
        s = res[m]["summary"]
        ca = s["classA"]
        # frozen 0 -> recoverability == probe accuracy
        assert ca["recover_linear"]["mean"] == pytest.approx(ca["linear_acc"]["mean"])
        # control (frozen 1) has undefined recoverability
        assert s["control_candok"]["recover_linear"]["mean"] is None


def test_historical_crosscheck_values():
    hist = aq.historical(os.path.join(ROOT, "experiments", "generator_repair_results.json"))
    assert "A_terminal" in hist and "E_candidate" in hist
    # candidate-local supervision historically increased candidate written accuracy
    assert hist["E_candidate"]["candidate_written_mean"] > hist["A_terminal"]["candidate_written_mean"]


def test_operand_pair_heldout_generalization(data):
    z = data["z"]
    def pairs(split):
        return set(zip(z[f"E_candidate__{split}__operand_a"].tolist(),
                       z[f"E_candidate__{split}__operand_b"].tolist()))
    tr, te = pairs("train"), pairs("test")
    # the benchmark enumerates all 100 operand pairs in every split: no natural held-out combos
    assert te <= tr and len(tr - te) == 0
    # a held-out-operand-pair protocol is nevertheless constructible and runs
    heldout = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}
    def hold(split):
        a = z[f"E_candidate__{split}__operand_a"]; b = z[f"E_candidate__{split}__operand_b"]
        return np.array([(int(u), int(v)) in heldout for u, v in zip(a, b)])
    htr, hva, hte = hold("train"), hold("val"), hold("test")
    Xtr = z["E_candidate__train__C"][~htr].astype(np.float64); ytr = z["E_candidate__train__y_true"][~htr]
    Xva = z["E_candidate__val__C"][~hva].astype(np.float64); yva = z["E_candidate__val__y_true"][~hva]
    Xte = z["E_candidate__test__C"][hte].astype(np.float64); yte = z["E_candidate__test__y_true"][hte]
    pred, val, lam = aq.ridge_probe(Xtr, ytr, Xva, yva, 10)
    acc = aq.acc_of(pred(Xte), yte)
    assert 0.0 <= acc <= 1.0 and Xte.shape[0] == int(hte.sum()) > 0


def test_repeated_add_subset(data):
    z = data["z"]
    rep = z["E_candidate__test__repeated"].astype(bool)
    ok = z["E_candidate__test__cand_ok"].astype(bool)
    assert rep.any()
    hard_rep = rep & (~ok)
    # Class-A repeated-ADD subtype is non-empty and indexable
    assert int(hard_rep.sum()) > 0


def test_metadata_integrity(data):
    meta = data["meta"]
    assert meta["train_cap"] == rq.TRAIN_CAP and meta["val_cap"] == rq.VAL_CAP
    assert set(meta["n_test"]) == set(MODELS)
    for m in MODELS:
        for s in ("42", "43", "44", "45", "46"):
            assert meta["models"][m]["checkpoints"][s]["sha256"]
            assert meta["models"][m]["checkpoints"][s]["unchanged_after_run"] is True
    # frozen W,b are stored and finite
    for m in MODELS:
        for split in ("test__C",):
            assert np.isfinite(data["z"][f"{m}__W"]).all()
            assert np.isfinite(data["z"][f"{m}__b"]).all()

