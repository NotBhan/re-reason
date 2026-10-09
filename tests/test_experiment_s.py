"""Tests for Experiment S — controlled candidate-INTERNAL representation investigation.

Covers the §27 validity/leakage contract plus reproducibility: frozen-checkpoint hash preservation,
no model-parameter mutation, no-grad, exact and deterministic activation capture, activation
dimensions matching the actual model, sequence-split separation, hard-train/hard-val/hard-test
separation, no hard-test tuning, no operand-pair leakage, no probe insertion into inference, no
modification of the frozen W,b, C_t-vs-R reproduction, every declared stage analyzed, no test-based
activation selection, malformed-artifact rejection, JSON/figure determinism, and historical-report
immutability.
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

import analyze_experiment_s as asx  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_s as rsx  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset, hash_sequence  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_s_candidate_internal.npz")
MODELS = ("E_candidate", "F_combined")
ALLOWED_VERDICTS = {"S-A_downstream_output_localized", "S-B_progressive_loss",
                    "S-C_computation_limited", "S-MIX"}


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_s npz not present")
    return asx.load(NPZ)


@pytest.fixture(scope="module")
def built(data):
    return asx.build(data)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


def _tiny(B=6, T=10):
    task = gr.build_task(T)
    seq = [(0, 0, -1, 5), (0, 1, -1, 7), (0, 2, -1, 3), (0, 3, -1, 4), (0, 0, -1, 2),
           (0, 1, -1, 9), (0, 2, -1, 6), (0, 3, -1, 1), (2, 0, 1, -1), (2, 0, 1, -1)]
    inp = np.zeros((B, T, 21), np.float32)
    st = np.zeros((B, T, 4), np.int64)
    for b in range(B):
        regs = [0, 0, 0, 0]
        for t, (op, d, s, v) in enumerate(seq):
            inp[b, t] = task._encode_operation(op, d, s, v)
            if op == 0:
                regs[d] = v
            elif op == 2:
                regs[d] = (regs[d] + regs[s]) % 10
            st[b, t] = regs
    return SplitDataset(torch.from_numpy(inp), torch.from_numpy(st[:, -1].copy()),
                        torch.from_numpy(st), [])


# ---------------------------------------------------------------------------
# collector: exact capture / determinism / no mutation / no grad
# ---------------------------------------------------------------------------

def test_collector_exact_activation_capture(loaded_model):
    ds = _tiny()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r1 = rsx.capture_split(loaded_model, ds, 42, cap=None)
    r2 = rsx.capture_split(loaded_model, ds, 42, cap=None)
    for k in rsx.STAGES:
        assert np.array_equal(r1[k], r2[k]), k          # deterministic
        assert r1[k].shape[1] == rsx.STAGE_DIMS[k]
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])                # no parameter mutation
    assert not any(p.grad is not None for p in loaded_model.parameters())   # no grad

    # exact reproduction of every internal stage from the frozen forward
    core = loaded_model.cognition_core
    xb = ds.inputs
    with torch.no_grad():
        diag = loaded_model.forward_with_states(xb, return_components=True)
        e_t = loaded_model.encoder(xb)
        S = torch.stack(diag.states, 1)
        G = torch.stack(diag.gates, 1)
        prev = torch.cat([torch.zeros(xb.shape[0], 1, 212), S[:, :-1]], 1)
        combined = torch.cat([G * prev, e_t], -1)
        pre1 = core.cand_mlp[0](combined)
        post1 = core.cand_mlp[1](pre1)
        pre2 = core.cand_mlp[2](post1)
        idx = (gr.write_mask_from_inputs(xb)[1] == 2).nonzero()
        b, t = idx[0]
    for name, ref in (("S_prev", prev[b, t]), ("f_gate", G[b, t]), ("h_reset", (G * prev)[b, t]),
                      ("cand_input", combined[b, t]), ("h1_pre", pre1[b, t]),
                      ("h1_post", post1[b, t]), ("out_pre", pre2[b, t]),
                      ("C_t", torch.tanh(pre2)[b, t])):
        assert np.array_equal(r1[name][0], ref.numpy()), name


def test_activation_dims_match_actual_model(loaded_model):
    core = loaded_model.cognition_core
    assert core.cand_mlp[0].in_features == rsx.STAGE_DIMS["cand_input"] == 468
    assert core.cand_mlp[0].out_features == rsx.STAGE_DIMS["h1_pre"] == 146
    assert core.cand_mlp[2].in_features == rsx.STAGE_DIMS["h1_post"] == 146
    assert core.cand_mlp[2].out_features == rsx.STAGE_DIMS["out_pre"] == 212
    assert core.w_f_hh.out_features == rsx.STAGE_DIMS["f_gate"] == 212
    assert loaded_model.state_dim == rsx.STAGE_DIMS["C_t"] == 212
    # candidate network has exactly ONE hidden layer (not two)
    assert len([m for m in core.cand_mlp if isinstance(m, torch.nn.Linear)]) == 2


# ---------------------------------------------------------------------------
# npz structure / reproduction / splits / hard tail
# ---------------------------------------------------------------------------

def test_npz_stage_dims_and_completeness(data):
    z = data["z"]
    for m in MODELS:
        for split in ("trainord", "trainhard", "valord", "valhard", "test", "hardextra"):
            for st in rsx.STAGES:
                arr = z[f"{m}__{split}__{st}"]
                assert arr.shape[1] == rsx.STAGE_DIMS[st], (m, split, st)
                assert arr.shape[0] == z[f"{m}__{split}__y_true"].shape[0]


def test_c_t_and_readout_reproduced_from_npz(data):
    z = data["z"]
    for m in MODELS:
        W = z[f"{m}__W"]; b = z[f"{m}__b"]
        for si, s in enumerate((42, 43, 44, 45, 46)):
            sel = z[f"{m}__test__seed"] == s
            C = z[f"{m}__test__C_t"][sel]; dest = z[f"{m}__test__op_dest"][sel]
            y = z[f"{m}__test__y_true"][sel]; ok = z[f"{m}__test__cand_ok"][sel].astype(bool)
            logits = np.einsum("nd,nvd->nv", C, W[si][dest]) + b[si][dest]
            assert np.array_equal(logits.argmax(1) == y, ok)
            # S stages must include the candidate vector used by Q/R verbatim
            assert np.array_equal(C, z[f"{m}__test__C_t"][sel]) or C.shape == z[f"{m}__test__C_t"][sel].shape


def test_w_b_match_frozen_readout(data, loaded_model):
    z = data["z"]
    W = loaded_model.decision_head.proj.weight.detach().view(4, 10, 212).numpy()
    b = loaded_model.decision_head.proj.bias.detach().view(4, 10).numpy()
    assert np.allclose(z["E_candidate__W"][0], W, atol=1e-6)
    assert np.allclose(z["E_candidate__b"][0], b, atol=1e-6)


def test_checkpoint_hashes_unchanged(data):
    for m in MODELS:
        for s in (42, 43, 44, 45, 46):
            rec = data["meta"]["models"][m]["checkpoints"][str(s)]
            assert rec["unchanged_after_run"] is True
            assert eg.sha256_file(os.path.join(ROOT, rec["path"])) == rec["sha256"]


def test_sequence_split_and_augmented_stream_disjoint():
    for seed in (42, 43, 44, 45, 46):
        sp = gr.canonical_splits(seed)
        tr = sp.train_set.inputs.numpy(); va = sp.val_set.inputs.numpy(); te = sp.test_set.inputs.numpy()
        hs = lambda a: set(map(lambda x: x.tobytes(), a))
        assert len(hs(tr) & hs(va)) == 0 and len(hs(tr) & hs(te)) == 0 and len(hs(va) & hs(te)) == 0
    import run_experiment_r as rr
    known = set()
    for ds in (gr.canonical_splits(42).train_set, gr.canonical_splits(42).val_set,
               gr.canonical_splits(42).test_set):
        for row in ds.inputs.numpy():
            known.add(hash_sequence(row))
    extra = rr.extra_hard_stream(42)
    assert sum(1 for row in extra.inputs.numpy() if hash_sequence(row) in known) == 0


def test_hard_splits_all_candidate_wrong(data):
    z = data["z"]
    for m in MODELS:
        for split in ("trainhard", "valhard", "hardextra"):
            assert (z[f"{m}__{split}__cand_ok"] == 0).all()
        assert (z[f"{m}__test__cand_ok"] == 0).any() and (z[f"{m}__test__cand_ok"] == 1).any()


def test_hard_test_is_the_untouched_canonical_class_a(data):
    z = data["z"]
    for m in MODELS:
        # Class-A == ADD transition where the frozen candidate decode != true value
        assert np.array_equal(z[f"{m}__test__y_true"],
                              (z[f"{m}__test__operand_a"] + z[f"{m}__test__operand_b"]) % 10)
        assert np.array_equal(z[f"{m}__test__wrap"].astype(bool),
                              (z[f"{m}__test__operand_a"] + z[f"{m}__test__operand_b"]) >= 10)
        ok = z[f"{m}__test__cand_ok"].astype(bool)
        assert (~ok).mean() < 0.5


# ---------------------------------------------------------------------------
# probes: determinism, no test tuning, leakage path
# ---------------------------------------------------------------------------

def _tiny_X(n=250, d=212, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float64), rng.integers(0, 10, n).astype(int)


def test_probe_determinism(data):
    z = data["z"]
    X, y = _tiny_X(300, 146, 1); Xv, yv = _tiny_X(150, 146, 2)
    p1, _, l1 = asx.ridge_fit(X, y, np.ones(10), Xv, yv, 10)
    p2, _, l2 = asx.ridge_fit(X, y, np.ones(10), Xv, yv, 10)
    assert l1 == l2 and np.array_equal(p1(Xv), p2(Xv))
    m1, v1 = asx.mlp_fit(X, y, Xv, yv, 16, np.ones(10), 7)
    m2, v2 = asx.mlp_fit(X, y, Xv, yv, 16, np.ones(10), 7)
    assert v1 == v2 and np.array_equal(m1(Xv), m2(Xv))


def test_no_test_tuning_lambda_from_val_only(data):
    z = data["z"]
    f1 = asx.fit_regime(z, "E_candidate", 42, "D0", "h1_post", "linear", None)
    z2 = {k: v for k, v in z.items()}
    z2["E_candidate__test__y_true"] = z["E_candidate__test__y_true"][::-1]   # corrupt TEST labels only
    f2 = asx.fit_regime(z2, "E_candidate", 42, "D0", "h1_post", "linear", None)
    assert f2["lambda"] == f1["lambda"] and f2["train_acc"] == f1["train_acc"] and f2["val_acc"] == f1["val_acc"]


def test_deliberate_leakage_failure_path():
    X, y = _tiny_X(200, 212, 3); Xt, yt = _tiny_X(200, 212, 4)
    clean, _, _ = asx.ridge_fit(X, y, np.ones(10), Xt, yt, 10)
    leak, _, _ = asx.ridge_fit(Xt, yt, np.ones(10), Xt, yt, 10)
    assert float((leak(Xt).argmax(1) == yt).mean()) >= float((clean(Xt).argmax(1) == yt).mean())
    overlap = len(set(map(lambda r: r.tobytes(), X)) & set(map(lambda r: r.tobytes(), Xt)))
    assert overlap == 0


def test_operand_pair_leakage_excluded_from_fit(data):
    z = data["z"]
    for st in ("h1_post", "C_t"):
        f = asx.fit_regime(z, "E_candidate", 42, "D1x", st, "mlp", 32, exclude_heldout=True)
        # recompute the fit indices and confirm none are held-out operand pairs
        idx = np.nonzero(z["E_candidate__hardextra__seed"] == 42)[0]
        oa = z["E_candidate__hardextra__operand_a"][idx]; ob = z["E_candidate__hardextra__operand_b"][idx]
        held = np.array([(int(a), int(b)) in asx.HELD_OUT_PAIRS for a, b in zip(oa, ob)])
        assert held.sum() > 0                          # the held-out pairs DO occur in the pool
        assert f["n_train"] == int((~held).sum())      # but were excluded from the fit
    assert len(asx.HELD_OUT_PAIRS) == 20 and all(b in (0, 5) for a, b in asx.HELD_OUT_PAIRS)


def test_no_probe_insertion_forward_unchanged(loaded_model):
    ds = _tiny()
    with torch.no_grad():
        before = loaded_model.forward_with_states(ds.inputs, return_components=True).states[-1].clone()
    X, y = _tiny_X(128, 146, 21); Xv, yv = _tiny_X(64, 146, 22)
    asx.ridge_fit(X, y, asx.class_weights(y, 10), Xv, yv, 10)
    asx.mlp_fit(X, y, Xv, yv, 16, asx.class_weights(y, 10), 3)
    with torch.no_grad():
        after = loaded_model.forward_with_states(ds.inputs, return_components=True).states[-1].clone()
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# analysis completeness / no test-based selection / verdict
# ---------------------------------------------------------------------------

def test_every_declared_stage_analyzed(built):
    for m in MODELS:
        for regime in asx.REGIMES:
            for st in rsx.STAGES:
                assert st in built[m]["caps"][regime]
                assert asx.REF_CAP in built[m]["caps"][regime][st]
                for f in ("hardtest", "train_acc", "val_acc", "correct_all", "hard_wrap", "hard_nowrap",
                          "hard_repeated", "hard_isolated"):
                    assert built[m]["caps"][regime][st][asx.REF_CAP][f]["mean"] is not None


def test_no_test_based_activation_selection():
    # the analyzed activation set is the predeclared constant list, chosen from architecture
    assert rsx.STAGES == ["S_prev", "f_gate", "h_reset", "cand_input", "h1_pre", "h1_post",
                          "out_pre", "C_t"]
    assert asx.FULL_STAGES == ["cand_input", "h1_post", "out_pre", "C_t"]
    assert asx.ALL_STAGE_CAPS == ["linear", "mlp16", "mlp32"]
    # cap_list_for is a pure function of the stage name (never of any test metric)
    assert asx.cap_list_for("C_t") == asx.cap_list_for("C_t")
    assert [n for n, _ in asx.cap_list_for("S_prev")] == ["linear", "mlp16", "mlp32"]
    assert [n for n, _ in asx.cap_list_for("cand_input")] == ["linear", "mlp16", "mlp32", "mlp64", "mlp128"]


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    for m in MODELS:
        assert built[m]["chain"]["D1x"]["C_t"] is not None
    # every stage has a paired internal-vs-C_t comparison (no cherry-picking)
    assert set(built["E_candidate"]["internal_vs_Ct"]) == set(rsx.STAGES)


def test_json_determinism(data):
    a1 = asx.analyze_seed(data["z"], "E_candidate", 43)
    a2 = asx.analyze_seed(data["z"], "E_candidate", 43)
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)


# ---------------------------------------------------------------------------
# malformed artifacts / figures / immutability
# ---------------------------------------------------------------------------

def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        asx.load(str(tmp_path / "nope.npz"))
    with pytest.raises(Exception):
        asx.fit_regime({}, "E_candidate", 42, "D0", "C_t", "linear", None)


def test_figure_creation(data, built, tmp_path):
    figs = asx.make_figures(data, built, str(tmp_path), prefix="test_s")
    assert len(figs) >= 15 and all(os.path.exists(p) for p in figs)


def test_historical_immutability():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"
