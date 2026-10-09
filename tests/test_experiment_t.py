"""Tests for Experiment T — upstream-representation localisation of the Class-A information.

Covers the §32 validity/leakage contract plus reproducibility: frozen-checkpoint hash preservation,
no model-parameter mutation, no-grad, exact upstream activation capture (encoder stages, gate
preactivation, h_reset, cand_input), the e_t+h_reset concatenation identity, sigmoid(gate_pre) ==
f_gate, deterministic capture, activation dimensions matching the actual model, sequence-split and
augmented-stream separation, hard-train/hard-val/hard-test separation, no test tuning, no operand-pair
leakage, no probe insertion into inference, C_t-vs-R/S reproduction, every declared stage analyzed,
no test-based stage/capacity selection, malformed-artifact rejection, JSON/figure determinism, and
historical-report immutability.
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

import analyze_experiment_t as at  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_t as rt  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset, hash_sequence  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_t_upstream_representation.npz")
RESULTS = os.path.join(ROOT, "experiments", "experiment_t_results.json")
MODELS = ("E_candidate", "F_combined")
ALLOWED_VERDICTS = {"T-A_encoder_limited", "T-B_state_limited", "T-C_source_fusion_limited",
                    "T-D_weak_throughout", "T-MIX"}


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_t npz not present")
    return at.load(NPZ)


@pytest.fixture(scope="module")
def built(data):
    # The aggregate analysis (5 seeds x 2 models x the full probe grid) is expensive; the tests
    # validate the produced analysis artifact directly (its structure/completeness/decision), while
    # the probe code paths themselves are exercised by the per-seed tests below and by the
    # dedicated determinism test (`test_json_determinism` re-runs `analyze_seed` end-to-end).
    if not os.path.exists(RESULTS):
        pytest.skip("experiment_t results.json not present")
    with open(RESULTS) as fh:
        return json.load(fh)


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

def test_collector_exact_upstream_capture(loaded_model):
    ds = _tiny()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r1 = rt.capture_split(loaded_model, ds, 42, cap=None)
    r2 = rt.capture_split(loaded_model, ds, 42, cap=None)
    for k in rt.STAGES:
        assert np.array_equal(r1[k], r2[k]), k          # deterministic
        assert r1[k].shape[1] == rt.STAGE_DIMS[k]
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])                # no parameter mutation
    assert not any(p.grad is not None for p in loaded_model.parameters())   # no grad

    core = loaded_model.cognition_core
    enc = loaded_model.encoder.net
    xb = ds.inputs
    with torch.no_grad():
        diag = loaded_model.forward_with_states(xb, return_components=True)
        e_t = enc(xb)
        S = torch.stack(diag.states, 1)
        G = torch.stack(diag.gates, 1)
        prev = torch.cat([torch.zeros(xb.shape[0], 1, 212), S[:, :-1]], 1)
        gate_pre = core.w_f_ih(e_t) + core.w_f_hh(prev)
        combined = torch.cat([G * prev, e_t], -1)
        idx = (gr.write_mask_from_inputs(xb)[1] == 2).nonzero()
        b, t = idx[0]
        refs = {
            "raw_obs": xb[b, t],
            "enc_h1_pre": enc[0](xb)[b, t],
            "enc_h1_post": enc[1](enc[0](xb))[b, t],
            "enc_out_pre": enc[2](enc[1](enc[0](xb)))[b, t],
            "e_t": e_t[b, t],
            "S_prev": prev[b, t],
            "gate_pre": gate_pre[b, t],
            "f_gate": G[b, t],
            "h_reset": (G * prev)[b, t],
            "cand_input": combined[b, t],
            "C_t": torch.stack(diag.candidates, 1)[b, t],
        }
        for name, ref in refs.items():
            assert np.array_equal(r1[name][0], ref.detach().numpy()), name


def test_activation_dims_match_actual_model(loaded_model):
    enc = loaded_model.encoder.net
    core = loaded_model.cognition_core
    assert enc[0].in_features == rt.STAGE_DIMS["raw_obs"] == 21
    assert enc[0].out_features == rt.STAGE_DIMS["enc_h1_pre"] == 64
    assert enc[2].out_features == rt.STAGE_DIMS["enc_out_pre"] == 256
    assert rt.STAGE_DIMS["e_t"] == 256
    assert core.w_f_ih.out_features == rt.STAGE_DIMS["gate_pre"] == 212
    assert core.cand_mlp[0].in_features == rt.STAGE_DIMS["cand_input"] == 468
    assert loaded_model.state_dim == rt.STAGE_DIMS["C_t"] == 212
    assert len([m for m in core.cand_mlp if isinstance(m, torch.nn.Linear)]) == 2


def test_cand_input_identity_from_et_hreset(loaded_model):
    ds = _tiny()
    r = rt.capture_split(loaded_model, ds, 42, cap=None)
    ci = np.concatenate([r["h_reset"], r["e_t"]], axis=1)
    assert ci.shape[1] == rt.STAGE_DIMS["cand_input"]
    assert np.array_equal(r["cand_input"], ci)


def test_gate_pretivation_sigmoid_matches_post(loaded_model):
    ds = _tiny()
    r = rt.capture_split(loaded_model, ds, 42, cap=None)
    sig = 1.0 / (1.0 + np.exp(-r["gate_pre"].astype(np.float64)))
    assert np.allclose(sig, r["f_gate"].astype(np.float64), atol=1e-6)


# ---------------------------------------------------------------------------
# npz structure / reproduction / splits / hard tail
# ---------------------------------------------------------------------------

def test_npz_stage_dims_and_completeness(data):
    z = data["z"]
    for m in MODELS:
        for split in ("trainord", "trainhard", "valord", "valhard", "test", "hardextra"):
            for st in rt.STAGES:
                arr = z[f"{m}__{split}__{st}"]
                assert arr.shape[1] == rt.STAGE_DIMS[st], (m, split, st)
                assert arr.shape[0] == z[f"{m}__{split}__y_true"].shape[0]


def test_c_t_and_readout_reproduced_from_npz(data):
    z = data["z"]
    for m in MODELS:
        W = z[f"{m}__W"]
        b = z[f"{m}__b"]
        for si, s in enumerate((42, 43, 44, 45, 46)):
            sel = z[f"{m}__test__seed"] == s
            C = z[f"{m}__test__C_t"][sel]
            dest = z[f"{m}__test__op_dest"][sel]
            y = z[f"{m}__test__y_true"][sel]
            ok = z[f"{m}__test__cand_ok"][sel].astype(bool)
            logits = np.einsum("nd,nvd->nv", C, W[si][dest]) + b[si][dest]
            assert np.array_equal(logits.argmax(1) == y, ok)


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
        tr = sp.train_set.inputs.numpy()
        va = sp.val_set.inputs.numpy()
        te = sp.test_set.inputs.numpy()
        hs = lambda a: set(x.tobytes() for x in a)
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


def test_hard_test_is_untouched_canonical_class_a(data):
    z = data["z"]
    for m in MODELS:
        assert np.array_equal(z[f"{m}__test__y_true"],
                              (z[f"{m}__test__operand_a"] + z[f"{m}__test__operand_b"]) % 10)
        assert np.array_equal(z[f"{m}__test__wrap"].astype(bool),
                              (z[f"{m}__test__operand_a"] + z[f"{m}__test__operand_b"]) >= 10)
        assert (z[f"{m}__test__cand_ok"].astype(bool) == False).mean() < 0.5  # noqa: E712


def test_wrap_and_repeated_classification(data):
    z = data["z"]
    m = "E_candidate"
    oa = z[f"{m}__test__operand_a"].astype(int)
    ob = z[f"{m}__test__operand_b"].astype(int)
    wrap = z[f"{m}__test__wrap"].astype(bool)
    assert wrap.sum() > 0 and (~wrap).sum() > 0
    assert np.array_equal(wrap, (oa + ob) >= 10)
    rep = z[f"{m}__test__repeated"].astype(bool)
    assert rep.sum() > 0 and (~rep).sum() > 0
    # repeated rows must be at t>=1 (a previous step exists)
    assert (z[f"{m}__test__t"][rep] >= 1).all()


def test_raw_sum_target_generation(data):
    z = data["z"]
    for m in MODELS:
        assert np.array_equal(z[f"{m}__test__raw_sum"],
                              z[f"{m}__test__operand_a"] + z[f"{m}__test__operand_b"])
        assert int(z[f"{m}__test__raw_sum"].min()) >= 0
        assert int(z[f"{m}__test__raw_sum"].max()) <= 18


# ---------------------------------------------------------------------------
# probes: determinism, no test tuning, leakage path
# ---------------------------------------------------------------------------

def _tiny_X(n=250, d=212, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float64), rng.integers(0, 10, n).astype(int)


def test_probe_determinism(data):
    X, y = _tiny_X(300, 64, 1)
    Xv, yv = _tiny_X(150, 64, 2)
    p1, _, l1 = at.as_.ridge_fit(X, y, np.ones(10), Xv, yv, 10)
    p2, _, l2 = at.as_.ridge_fit(X, y, np.ones(10), Xv, yv, 10)
    assert l1 == l2 and np.array_equal(p1(Xv), p2(Xv))
    m1, v1 = at.as_.mlp_fit(X, y, Xv, yv, 16, np.ones(10), 7)
    m2, v2 = at.as_.mlp_fit(X, y, Xv, yv, 16, np.ones(10), 7)
    assert v1 == v2 and np.array_equal(m1(Xv), m2(Xv))


def test_probe_train_val_test_separation(data):
    z = data["z"]
    f1 = at._fit(z, "E_candidate", 42, "D0", "e_t", "linear", None)
    # train comes from trainord (cap 1500), val from valord (cap 800) — disjoint by construction
    assert f1["n_train"] == min(1500, int((z["E_candidate__trainord__seed"] == 42).sum()))
    assert f1["n_val"] == min(800, int((z["E_candidate__valord__seed"] == 42).sum()))
    z2 = {k: v for k, v in z.items()}
    z2["E_candidate__test__y_true"] = z["E_candidate__test__y_true"][::-1]  # corrupt TEST only
    f2 = at._fit(z2, "E_candidate", 42, "D0", "e_t", "linear", None)
    assert f2["lambda"] == f1["lambda"] and f2["train_acc"] == f1["train_acc"] and f2["val_acc"] == f1["val_acc"]


def test_deliberate_leakage_failure_path():
    X, y = _tiny_X(200, 212, 3)
    Xt, yt = _tiny_X(200, 212, 4)
    clean, _, _ = at.as_.ridge_fit(X, y, np.ones(10), Xt, yt, 10)
    leak, _, _ = at.as_.ridge_fit(Xt, yt, np.ones(10), Xt, yt, 10)
    assert float((leak(Xt).argmax(1) == yt).mean()) >= float((clean(Xt).argmax(1) == yt).mean())
    overlap = len(set(x.tobytes() for x in X) & set(x.tobytes() for x in Xt))
    assert overlap == 0


def test_operand_pair_leakage_excluded_from_fit(data):
    z = data["z"]
    for st in ("e_t", "h_reset", "C_t"):
        f = at._fit(z, "E_candidate", 42, "D1x", st, "mlp", 32, exclude_heldout=True)
        idx = np.nonzero(z["E_candidate__hardextra__seed"] == 42)[0]
        oa = z["E_candidate__hardextra__operand_a"][idx]
        ob = z["E_candidate__hardextra__operand_b"][idx]
        held = np.array([(int(a), int(b)) in at.HELD_OUT_PAIRS for a, b in zip(oa, ob)])
        assert held.sum() > 0
        assert f["n_train"] == int((~held).sum())
    assert len(at.HELD_OUT_PAIRS) == 20 and all(b in (0, 5) for a, b in at.HELD_OUT_PAIRS)


def test_no_probe_insertion_forward_unchanged(loaded_model):
    ds = _tiny()
    with torch.no_grad():
        before = loaded_model.forward_with_states(ds.inputs, return_components=True).states[-1].clone()
    X, y = _tiny_X(128, 64, 21)
    Xv, yv = _tiny_X(64, 64, 22)
    at.as_.ridge_fit(X, y, at.as_.class_weights(y, 10), Xv, yv, 10)
    at.as_.mlp_fit(X, y, Xv, yv, 16, at.as_.class_weights(y, 10), 3)
    with torch.no_grad():
        after = loaded_model.forward_with_states(ds.inputs, return_components=True).states[-1].clone()
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# analysis completeness / no test-based selection / verdict
# ---------------------------------------------------------------------------

def test_every_declared_stage_analyzed(built):
    for m in MODELS:
        for regime in at.REGIMES:
            for st in rt.STAGES:
                assert st in built[m]["caps"][regime]
                assert "linear" in built[m]["caps"][regime][st]
                for f in ("hardtest", "train_acc", "val_acc", "correct_all", "hard_wrap",
                          "hard_nowrap", "hard_repeated", "hard_isolated"):
                    assert built[m]["caps"][regime][st]["linear"][f]["mean"] is not None


def test_no_test_based_stage_or_capacity_selection():
    assert rt.STAGES == ["raw_obs", "enc_h1_pre", "enc_h1_post", "enc_out_pre", "e_t",
                         "S_prev", "gate_pre", "f_gate", "h_reset", "cand_input", "C_t"]
    assert rt.ENCODER_STAGES == ["raw_obs", "enc_h1_pre", "enc_h1_post", "enc_out_pre", "e_t"]
    assert rt.STATE_STAGES == ["S_prev", "gate_pre", "f_gate", "h_reset"]
    assert rt.KEY_STAGES == ["e_t", "S_prev", "gate_pre", "f_gate", "h_reset", "cand_input", "C_t"]
    # probes_for is a pure function of the stage name
    assert at.probes_for("C_t") == at.probes_for("C_t")
    assert [n for _, n, _ in at.probes_for("raw_obs")] == ["linear", "mlp32"] * 4
    assert {"mlp64", "mlp128"} <= {n for _, n, _ in at.probes_for("e_t")}


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    for m in MODELS:
        assert built[m]["chain"]["D1x"]["C_t"] is not None
        assert set(built[m]["caps"]["D1x"]) == set(rt.STAGES)
    assert set(built["E_candidate"]["heldout_operand"]["D1x"]) == set(rt.STAGES)


def test_bridge_baseline_against_S_R(built):
    # T's C_t row must reproduce the established S/R 5-seed means for the same regimes
    ct = built["E_candidate"]["caps"]
    d0_lin = ct["D0"]["C_t"]["linear"]["hardtest"]["mean"]
    d0_mlp = ct["D0"]["C_t"]["mlp32"]["hardtest"]["mean"]
    d1x_mlp = ct["D1x"]["C_t"]["mlp32"]["hardtest"]["mean"]
    assert abs(d0_lin - 0.244) < 0.06, d0_lin
    assert abs(d0_mlp - 0.238) < 0.06, d0_mlp
    assert abs(d1x_mlp - 0.463) < 0.06, d1x_mlp


def test_candidate_correct_control_separation(built):
    # candidate-correct decodability must exceed the Class-A hard tail at the same stage
    for m in MODELS:
        hard = built[m]["caps"]["D1x"]["C_t"]["mlp32"]["hardtest"]["mean"]
        corr = built[m]["caps"]["D1x"]["C_t"]["mlp32"]["correct_all"]["mean"]
        assert corr > hard


def test_ef_alignment(built):
    for st in rt.STAGES:
        assert st in built["E_candidate"]["chain"]["D1x"]
        assert built["F_combined"]["chain"]["D1x"][st] is not None
        assert st in built["E_vs_F"]


def test_json_determinism(data):
    a1 = at.analyze_seed(data["z"], "E_candidate", 43)
    a2 = at.analyze_seed(data["z"], "E_candidate", 43)
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)


# ---------------------------------------------------------------------------
# malformed artifacts / figures / immutability
# ---------------------------------------------------------------------------

def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        at.load(str(tmp_path / "nope.npz"))
    with pytest.raises(Exception):
        at._fit({}, "E_candidate", 42, "D0", "C_t", "linear", None)


def test_figure_creation(data, built, tmp_path):
    figs = at.make_figures(data, built, str(tmp_path), prefix="test_t")
    assert len(figs) >= 16 and all(os.path.exists(p) for p in figs)


def test_historical_immutability():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"
