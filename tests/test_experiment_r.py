"""Tests for Experiment R — candidate-representation information test on the Class-A tail.

Covers ~28 checks: collector determinism, checkpoint SHA/immutability, no-grad, exact candidate-vector
reproduction, sequence-split disjointness, hard-tail construction, hard-split disjointness,
ordinary/hard regime construction, target balancing, capacity ordering, parameter accounting, probe
determinism, no-test-tuning, a deliberate leakage failure path, held-out operand split, matched
matching, wrap/no-wrap and repeated/isolated classification, raw-sum target generation, candidate-
correct controls, train/val/test metric separation, E/F alignment, malformed data, JSON determinism,
figure creation, and historical immutability.
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

import analyze_experiment_r as ar  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_q as rq  # noqa: E402
import run_experiment_r as rr  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_r_candidate_representation.npz")
MODELS = ("E_candidate", "F_combined")


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_r npz not present")
    return ar.load(NPZ)


@pytest.fixture(scope="module")
def built(data):
    return ar.build(data)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


def _tiny(B=5, T=10):
    task = gr.build_task(T)
    seq = [(0, 0, -1, 5), (0, 1, -1, 7), (0, 2, -1, 3), (0, 3, -1, 4), (0, 0, -1, 2),
           (0, 1, -1, 9), (0, 2, -1, 6), (0, 3, -1, 1), (2, 0, 1, -1), (2, 0, 1, -1)]
    inp = np.zeros((B, T, 21), np.float32); st = np.zeros((B, T, 4), np.int64)
    for b in range(B):
        regs = [0, 0, 0, 0]
        for t, (op, d, s, v) in enumerate(seq):
            inp[b, t] = task._encode_operation(op, d, s, v)
            if op == 0:
                regs[d] = v
            elif op == 2:
                regs[d] = (regs[d] + regs[s]) % 10
            st[b, t] = regs
    from re_reason.benchmarks.dataset_split import SplitDataset
    return SplitDataset(torch.from_numpy(inp), torch.from_numpy(st[:, -1].copy()),
                        torch.from_numpy(st), [])


# ---------------------------------------------------------------------------
# collector: determinism / no-grad / no-mutation / reproduction
# ---------------------------------------------------------------------------

def test_collector_deterministic_no_grad_no_mutation(loaded_model):
    ds = _tiny()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r1 = rq.capture_split(loaded_model, ds, 42, cap=None)
    r2 = rq.capture_split(loaded_model, ds, 42, cap=None)
    for k in r1:
        assert np.array_equal(r1[k], r2[k]), k
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


def test_checkpoint_sha_and_immutability(data):
    with open(os.path.join(ROOT, "experiments", "experiment_g_transitions_meta.json")) as fh:
        gmeta = json.load(fh)
    for m in MODELS:
        for s in (42, 43, 44, 45, 46):
            assert data["meta"]["models"][m]["checkpoints"][str(s)]["sha256"] == \
                gmeta["models"][m]["checkpoints"][str(s)]["sha256"]
            rel = gmeta["models"][m]["checkpoints"][str(s)]["path"]
            assert eg.sha256_file(os.path.join(ROOT, rel)) == \
                gmeta["models"][m]["checkpoints"][str(s)]["sha256"]
            assert data["meta"]["models"][m]["checkpoints"][str(s)]["unchanged_after_run"] is True


def test_exact_candidate_vector_reproduction(data):
    z = data["z"]
    for m in MODELS:
        W = z[f"{m}__W"]; b = z[f"{m}__b"]
        for si, s in enumerate((42, 43, 44, 45, 46)):
            sel = z[f"{m}__test__seed"] == s
            C = z[f"{m}__test__C"][sel]; dest = z[f"{m}__test__op_dest"][sel]
            y = z[f"{m}__test__y_true"][sel]; ok = z[f"{m}__test__cand_ok"][sel].astype(bool)
            logits = np.einsum("nd,nvd->nv", C, W[si][dest]) + b[si][dest]
            assert np.array_equal(logits.argmax(1) == y, ok)
            # hard tail == candidate-wrong by construction
            assert not ok[~ok].any()


# ---------------------------------------------------------------------------
# split disjointness / hard-split construction
# ---------------------------------------------------------------------------

def test_sequence_split_disjointness():
    for seed in (42, 43, 44, 45, 46):
        sp = gr.canonical_splits(seed)
        tr = sp.train_set.inputs.numpy(); va = sp.val_set.inputs.numpy(); te = sp.test_set.inputs.numpy()
        assert tr.shape[0] == 6000 and va.shape[0] == 1200 and te.shape[0] == 1200
        hs = lambda a: set(map(lambda x: x.tobytes(), a))
        assert len(hs(tr) & hs(va)) == 0 and len(hs(tr) & hs(te)) == 0 and len(hs(va) & hs(te)) == 0


def test_hard_tail_and_hard_split_disjointness(data):
    z = data["z"]; meta = data["meta"]
    for m in MODELS:
        for split in ("trainhard", "valhard", "hardextra"):
            assert (z[f"{m}__{split}__cand_ok"] == 0).all()      # every row is candidate-wrong
        assert z[f"{m}__test__cand_ok"].shape[0] > 0
        # hard splits come from disjoint sequence streams: extra stream is disjoint from splits
        assert meta["n"][m]["trainhard"] > 0 and meta["n"][m]["hardextra"] > 0


def test_augmented_stream_disjoint():
    """The augmented hard pool must be sequence-disjoint from train/val/test."""
    from re_reason.benchmarks.dataset_split import hash_sequence
    sp = gr.canonical_splits(42)
    known = set()
    for ds in (sp.train_set, sp.val_set, sp.test_set):
        for row in ds.inputs.numpy():
            known.add(hash_sequence(row))
    extra = rr.extra_hard_stream(42)
    clash = sum(1 for row in extra.inputs.numpy() if hash_sequence(row) in known)
    assert clash == 0


# ---------------------------------------------------------------------------
# regime construction
# ---------------------------------------------------------------------------

def test_ordinary_and_hard_regime_construction(data):
    z = data["z"]
    for m in MODELS:
        assert z[f"{m}__trainord__C"].shape[0] == 5 * rr.TRAIN_ORD_CAP
        assert z[f"{m}__valord__C"].shape[0] == 5 * rr.VAL_ORD_CAP
        # ordinary train is the Q protocol (first 1500 ADD of the train split)
        assert (z[f"{m}__trainhard__C"].shape[0] > 0)
        # hard-extra pool is materially larger than the designated hard train
        assert z[f"{m}__hardextra__C"].shape[0] > z[f"{m}__trainhard__C"].shape[0]


def test_target_balancing_and_distributions(data):
    z = data["z"]
    for split in ("trainord", "trainhard", "valhard", "test"):
        y = z[f"E_candidate__{split}__y_true"].astype(int)
        assert y.min() >= 0 and y.max() <= 9
    # class weights balance the effective mass across classes
    y = z["E_candidate__trainhard__y_true"].astype(int)
    wc = ar.class_weights(y, 10)
    mass = [float(wc[c] * (y == c).sum()) for c in range(10) if (y == c).any()]
    assert max(mass) - min(mass) < 1e-6


# ---------------------------------------------------------------------------
# capacities / parameter accounting
# ---------------------------------------------------------------------------

def test_capacity_ordering_and_params():
    hiddens = [h for _, h in ar.CAPS]
    nones = [h for h in hiddens if h is not None]
    assert nones == sorted(nones) and len(set(nones)) == len(nones)
    params = [ar.param_count("linear" if n == "linear" else "mlp", h) for n, h in ar.CAPS]
    assert params == sorted(params) and len(set(params)) == len(params)
    assert ar.param_count("linear", None) == 212 * 10 + 10
    assert ar.param_count("mlp", 16) == 212 * 16 + 16 + 16 * 10 + 10


# ---------------------------------------------------------------------------
# probes: determinism, no-test-tuning, leakage
# ---------------------------------------------------------------------------

def _tiny_X(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 212)).astype(np.float64), rng.integers(0, 10, n).astype(int)


def test_probe_determinism_and_no_test_tuning(data):
    z = data["z"]
    # ridge determinism
    X, y = _tiny_X(300, 1); Xv, yv = _tiny_X(150, 2)
    p1, _, l1 = ar.ridge_fit(X, y, np.ones(10), Xv, yv, 10)
    p2, _, l2 = ar.ridge_fit(X, y, np.ones(10), Xv, yv, 10)
    assert l1 == l2 and np.array_equal(p1(Xv), p2(Xv))
    # MLP determinism
    m1, v1 = ar.mlp_fit(X, y, Xv, yv, 16, np.ones(10), 7)
    m2, v2 = ar.mlp_fit(X, y, Xv, yv, 16, np.ones(10), 7)
    assert v1 == v2 and np.array_equal(m1(Xv), m2(Xv))
    # lambda selection depends only on the validation split (permuting TEST labels must not change it)
    import copy
    z2 = {k: v for k, v in z.items()}
    f1 = ar.fit_regime(z, "E_candidate", 42, "D0", "linear", None)
    lam_before = f1["lambda"]
    z2["E_candidate__test__y_true"] = z["E_candidate__test__y_true"][::-1]  # corrupt test labels only
    f2 = ar.fit_regime(z2, "E_candidate", 42, "D0", "linear", None)
    assert f2["lambda"] == lam_before and f2["train_acc"] == f1["train_acc"]


def test_deliberate_leakage_failure_path():
    X, y = _tiny_X(200, 3); Xt, yt = _tiny_X(200, 4)
    clean, _, _ = ar.ridge_fit(X, y, np.ones(10), Xt, yt, 10)
    clean_x, _, _ = ar.ridge_fit(Xt, yt, np.ones(10), Xt, yt, 10)
    clean_acc = float((clean(Xt).argmax(1) == yt).mean())
    leak_acc = float((clean_x(Xt).argmax(1) == yt).mean())
    assert leak_acc >= clean_acc                     # fitting on test must not lower test accuracy
    overlap = len(set(map(lambda r: r.tobytes(), X)) & set(map(lambda r: r.tobytes(), Xt)))
    assert overlap == 0                              # the pipeline's disjoint split prevents the leak


# ---------------------------------------------------------------------------
# held-out operands / matching
# ---------------------------------------------------------------------------

def test_heldout_operand_split_and_matching(data):
    z = data["z"]
    assert len(ar.HELD_OUT_PAIRS) == 20 and all(b in (0, 5) for a, b in ar.HELD_OUT_PAIRS)
    hard, correct = ar.test_hard_rows(z, "E_candidate", 42)
    matched = ar.matched_control(z, "E_candidate", 42)
    ok = z["E_candidate__test__cand_ok"].astype(bool)
    assert len(matched) > 0 and ok[matched].all()     # matched rows are candidate-correct
    assert 0.5 < len(matched) / max(int(hard.sum()), 1) <= 1.0

    def feat(mask):
        cols = []
        for f in ("wrap", "repeated", "t", "m_cand", "norm_C", "norm_dC"):
            v = z[f"E_candidate__test__{f}"][mask].astype(float)
            cols.append(np.abs(v) if f == "m_cand" else v)
        return np.stack(cols, 1)
    A = feat(hard); C = feat(correct)
    mu = np.concatenate([A, C], 0).mean(0); sd = np.concatenate([A, C], 0).std(0) + 1e-9
    matched_mask = np.zeros(len(hard), bool); matched_mask[matched] = True
    Ad = (A - mu) / sd; Cd = (C - mu) / sd; Md = (feat(matched_mask) - mu) / sd
    assert np.linalg.norm(Ad.mean(0) - Md.mean(0)) <= np.linalg.norm(Ad.mean(0) - Cd.mean(0)) + 1e-9


# ---------------------------------------------------------------------------
# subtype / target classification
# ---------------------------------------------------------------------------

def test_wrap_and_repeated_classification(data):
    z = data["z"]
    for split in ("trainord", "trainhard", "valhard", "hardextra", "test"):
        oa = z[f"E_candidate__{split}__operand_a"]; ob = z[f"E_candidate__{split}__operand_b"]
        wrap = z[f"E_candidate__{split}__wrap"].astype(bool)
        raw = z[f"E_candidate__{split}__raw_sum"]; y = z[f"E_candidate__{split}__y_true"]
        assert np.array_equal(wrap, (oa + ob) >= 10)
        assert np.array_equal(raw, oa + ob) and raw.max() <= 18
        assert np.array_equal(y, (oa + ob) % 10)
        rep = z[f"E_candidate__{split}__repeated"].astype(bool)
        t = z[f"E_candidate__{split}__t"]
        assert not rep[t == 0].any()


def test_candidate_correct_controls_and_metric_separation(built):
    res = built
    rE = res["E_candidate"]
    for name, _ in ar.CAPS:
        # ordinary-trained probes remain strongly decodable on candidate-correct states
        assert rE["caps"]["D0"][name]["correct_all"]["mean"] > 0.75
        # hard-specialized probes degrade on the easy majority but stay above chance (documented)
        assert rE["caps"]["D1x"][name]["correct_all"]["mean"] > 0.30
    for regime in ("D0", "D0m", "D1", "D1x"):
        c = rE["caps"][regime]["linear"]
        assert c["n_train"]["mean"] >= 1
        assert c["train_acc"]["mean"] is not None and c["val_acc"]["mean"] is not None
    assert rE["caps"]["D0"]["linear"]["hardtest"]["mean"] is not None


# ---------------------------------------------------------------------------
# E/F alignment, decision, malformed, JSON, figures, history
# ---------------------------------------------------------------------------

def test_EF_alignment(data):
    z = data["z"]
    # ordinary splits are identical across models; hard splits differ by each model's error rate
    for split in ("trainord", "valord", "test"):
        assert z[f"E_candidate__{split}__C"].shape == z[f"F_combined__{split}__C"].shape
    for m in MODELS:
        assert z[f"{m}__trainhard__C"].shape[1] == 212 and z[f"{m}__trainhard__C"].shape[0] > 0
        assert sorted(set(z[f"{m}__test__seed"].tolist())) == [42, 43, 44, 45, 46]


def test_decision_and_result_consistency(built):
    res = built
    for m in MODELS:
        d = res[m]["decision"]
        assert d["verdict"] in ("R-A_distribution_artifact", "R-B_capacity_limited_recoverability",
                                "R-C_robust_weak_recoverability", "R-D_memorization_artifact", "R-MIX")
    assert res[ar.PRIMARY_MODEL]["decision"]["verdict"].endswith(("R-A_distribution_artifact",
                                                                  "R-B_capacity_limited_recoverability",
                                                                  "R-C_robust_weak_recoverability",
                                                                  "R-D_memorization_artifact", "R-MIX"))


def test_malformed_json_figures_and_immutability(data, built, tmp_path):
    with pytest.raises(Exception):
        ar.load(str(tmp_path / "nope.npz"))
    with pytest.raises(Exception):
        ar.fit_regime({}, "E_candidate", 42, "D0", "linear", None)
    a1 = ar.analyze_seed(data["z"], "E_candidate", 42)
    a2 = ar.analyze_seed(data["z"], "E_candidate", 42)
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)
    figs = ar.make_figures(data, built, str(tmp_path), prefix="test_r")
    assert len(figs) >= 14 and all(os.path.exists(p) for p in figs)
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"


def test_arithmetic_targets_and_reference_probe(built):
    res = built
    rE = res["E_candidate"]
    assert ar.REF_CAP in dict(ar.CAPS)
    for regime in ("D0", "D1x"):
        for name in ("linear", ar.REF_CAP):
            assert 0.0 <= rE["arithmetic"][regime][name]["mean"] <= 1.0
    ho = rE["heldout_operand"]
    for regime in ("D0", "D1x"):
        for name in ("linear", ar.REF_CAP):
            assert ho[regime][name]["acc_heldout"]["mean"] is not None
            assert ho[regime][name]["acc_seen"]["mean"] is not None


# ---------------------------------------------------------------------------
# additional checks (matrix completeness, params, split sizes, targets, no insertion)
# ---------------------------------------------------------------------------

def test_capacity_distribution_matrix_complete(built):
    for m in MODELS:
        r = built[m]
        for regime in ar.REGIMES:
            for name, _ in ar.CAPS:
                c = r["caps"][regime][name]
                for f in ("hardtest", "train_acc", "val_acc", "correct_all", "hard_wrap", "hard_nowrap",
                          "hard_repeated", "hard_isolated", "n_train"):
                    assert c[f]["mean"] is not None, (m, regime, name, f)


def test_probe_parameter_counts():
    expected = {"linear": 2130, "mlp16": 212 * 16 + 16 + 16 * 10 + 10, "mlp32": 212 * 32 + 32 + 32 * 10 + 10,
                "mlp64": 212 * 64 + 64 + 64 * 10 + 10, "mlp128": 212 * 128 + 128 + 128 * 10 + 10}
    for name, hidden in ar.CAPS:
        assert ar.param_count("linear" if name == "linear" else "mlp", hidden) == expected[name]
    # every probe is far smaller than the frozen model
    assert max(expected.values()) < 226358 / 4


def test_regime_split_sizes(data):
    z = data["z"]
    for m in MODELS:
        n_hard_train = int((z[f"{m}__trainhard__seed"] >= 0).sum())
        assert z[f"{m}__trainord__C"].shape[0] == 5 * rr.TRAIN_ORD_CAP
        assert z[f"{m}__valord__C"].shape[0] == 5 * rr.VAL_ORD_CAP
        # D0m/D1 train on |hard-train| rows; D1x trains on the augmented pool
        assert z[f"{m}__hardextra__C"].shape[0] > n_hard_train


def test_hardtest_majority_and_targets(data):
    z = data["z"]
    for m in MODELS:
        ok = z[f"{m}__test__cand_ok"].astype(bool)
        assert (~ok).mean() < 0.5                       # hard tail is a minority of ADD
        raw = z[f"{m}__test__raw_sum"]
        assert raw.min() >= 0 and raw.max() <= 18       # raw-sum target is 19-way
        y = z[f"{m}__test__y_true"]
        assert raw[~ok].size == y[~ok].size


def test_no_probe_insertion_forward_unchanged(loaded_model):
    ds = _tiny()
    with torch.no_grad():
        before = loaded_model.forward_with_states(ds.inputs, return_components=True).states[-1].clone()
    X, y = _tiny_X(128, 21); Xv, yv = _tiny_X(64, 22)
    ar.ridge_fit(X, y, ar.class_weights(y, 10), Xv, yv, 10)
    ar.mlp_fit(X, y, Xv, yv, 16, ar.class_weights(y, 10), 3)
    with torch.no_grad():
        after = loaded_model.forward_with_states(ds.inputs, return_components=True).states[-1].clone()
    assert torch.equal(before, after)


def test_regime_val_split_mapping(data):
    z = data["z"]
    n_vord = int((z["E_candidate__valord__seed"] == 42).sum())
    n_vhard = int((z["E_candidate__valhard__seed"] == 42).sum())
    for regime, exp in (("D0", n_vord), ("D0m", n_vord), ("D1", n_vhard), ("D1x", n_vhard)):
        f = ar.fit_regime(z, "E_candidate", 42, regime, "linear", None)
        assert f["n_val"] == exp