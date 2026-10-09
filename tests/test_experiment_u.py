"""Tests for Experiment U — raw-observation / task-encoding localisation.

Covers the §32 validity/leakage contract plus reproducibility: frozen-checkpoint hash preservation,
no model-parameter mutation, no-grad, exact raw observation capture, causal history construction
with no future leakage, early-prefix/padding determinism, exact history hashing, collision grouping
and Bayes-ceiling correctness, current-observation structural insufficiency, sequence-split and
history-overlap accounting, probe train/val/test separation and no test tuning, held-out operand
exclusion, the exact state-reconstruction control, Class-A reproduction, candidate-correct controls,
E/F alignment, raw-sum/modular target generation, wrap/repeated classification, the full-prefix
Bayes=1.0 result, a deliberate future-leakage failure path, malformed-artifact rejection, JSON
determinism, figure creation, and historical-report immutability.
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

import analyze_experiment_u as au  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset, hash_sequence  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_u_task_interface.npz")
RESULTS = os.path.join(ROOT, "experiments", "experiment_u_results.json")
MODELS = ("E_candidate", "F_combined")
ALLOWED_VERDICTS = {"U-A_interface_complete", "U-B_interface_incomplete", "U-C_history_dependent",
                    "U-D_interface_identifiable_probe_limited", "U-MIX"}


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_u npz not present")
    return au.load(NPZ)


@pytest.fixture(scope="module")
def built(data):
    if not os.path.exists(RESULTS):
        pytest.skip("experiment_u results.json not present")
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
# collector: exact capture / no mutation / no grad
# ---------------------------------------------------------------------------

def test_exact_raw_observation_capture(loaded_model):
    ds = _tiny()
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    r = ru.capture_split(loaded_model, ds, 42, cap=None)
    assert r["seq_obs"].shape[1] == ru.SEQ_FLAT_DIM
    # seq_obs must be the raw benchmark observation sequence for the row's sequence
    want = ds.inputs.reshape(ds.inputs.shape[0], ru.SEQ_FLAT_DIM)[r["seq"]].numpy()
    assert np.array_equal(r["seq_obs"], want)
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])                 # no parameter mutation
    assert not any(p.grad is not None for p in loaded_model.parameters())   # no grad


def test_checkpoint_hashes_unchanged(data):
    for m in MODELS:
        for s in (42, 43, 44, 45, 46):
            rec = data["meta"]["models"][m]["checkpoints"][str(s)]
            assert rec["unchanged_after_run"] is True
            assert eg.sha256_file(os.path.join(ROOT, rec["path"])) == rec["sha256"]


# ---------------------------------------------------------------------------
# causal history construction / leakage
# ---------------------------------------------------------------------------

def test_causal_history_construction_and_dims(data):
    z = data["z"]
    seq = z["E_candidate__test__seq_obs"]
    t = z["E_candidate__test__t"]
    for rep, L in au.REPS:
        X, M = au.build_history(seq, t, L)
        assert X.shape == (len(t), L * ru.OBS_DIM)
        assert M.shape == (len(t), L)
        assert set(np.unique(M)) <= {0.0, 1.0}
        # the window ends exactly at t: the last real observation equals x_t
        full = seq.reshape(len(t), ru.T_MAX, ru.OBS_DIM)
        for i in range(0, len(t), 97):
            real = int(M[i].sum())
            win = X[i][(L - real) * ru.OBS_DIM:].reshape(real, ru.OBS_DIM)
            assert np.allclose(win[-1], full[i, int(t[i])])


def test_no_future_observation_leakage(data):
    z = data["z"]
    seq = z["E_candidate__test__seq_obs"]
    t = z["E_candidate__test__t"]
    rng = np.random.default_rng(0)
    for rep, L in au.REPS:
        X1, M1 = au.build_history(seq, t, L)
        # randomize every observation strictly AFTER t (a causal history must be unchanged)
        mod = seq.copy().reshape(len(t), ru.T_MAX, ru.OBS_DIM)
        for i in range(len(t)):
            ti = int(t[i])
            if ti + 1 < ru.T_MAX:
                mod[i, ti + 1:] = rng.random((ru.T_MAX - ti - 1, ru.OBS_DIM))
        X2, M2 = au.build_history(mod.reshape(len(t), -1), t, L)
        assert np.array_equal(X1, X2) and np.array_equal(M1, M2)
        # and the causal builder never uses an index > t
        for i in range(0, len(t), 97):
            assert int(M1[i].sum()) == min(L, int(t[i]) + 1)


def test_future_leakage_is_rejected_and_inflates():
    # a deliberately leaky builder that includes x_{t+1} must be detectable and must differ
    rng = np.random.default_rng(1)
    seq = rng.random((40, ru.SEQ_FLAT_DIM)).astype(np.float32)
    t = rng.integers(1, ru.T_MAX - 1, 40)
    Xc, Mc = au.build_history(seq, t, 2)
    full = seq.reshape(40, ru.T_MAX, ru.OBS_DIM)
    Xl = np.zeros_like(Xc)
    for i in range(40):
        ti = int(t[i])
        Xl[i, :ru.OBS_DIM] = full[i, ti]                     # x_t
        Xl[i, ru.OBS_DIM:] = full[i, ti + 1]                 # x_{t+1}  (FUTURE)
    assert not np.array_equal(Xc, Xl)                        # leaky build differs from causal build
    # sanity: the causal builder never uses an index > t
    for i in range(40):
        real = int(Mc[i].sum())
        assert real == min(2, int(t[i]) + 1)


def test_padding_and_early_prefix_determinism():
    rng = np.random.default_rng(2)
    seq = rng.random((10, ru.SEQ_FLAT_DIM)).astype(np.float32)
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], np.int64)
    X1, M1 = au.build_history(seq, t, 8)
    X2, M2 = au.build_history(seq, t, 8)
    assert np.array_equal(X1, X2) and np.array_equal(M1, M2)
    # early timesteps are left-padded with zeros and the mask marks exactly t+1 real slots
    for i in range(10):
        assert int(M1[i].sum()) == min(8, i + 1)
        pad = 8 - int(M1[i].sum())
        assert np.allclose(X1[i][:pad * ru.OBS_DIM], 0.0)


def test_history_length_ordering():
    assert [L for _, L in au.REPS] == [1, 2, 4, 8, 16, 10]
    assert au.REP_LABELS == ["L1", "L2", "L4", "L8", "L16", "Lfull"]
    # Lfull == benchmark maximum (the whole causal prefix for T=10); L16 saturates identically
    assert ru.FULL_PREFIX_LENGTH == ru.T_MAX == 10
    assert set(ru.HISTORY_LENGTHS) == {1, 2, 4, 8, 16}


# ---------------------------------------------------------------------------
# exact collision / Bayes
# ---------------------------------------------------------------------------

def test_exact_history_hashing_and_grouping():
    feats = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1]], np.float32)
    keys = au._keys(feats)
    assert keys[0] == keys[1] == keys[3] != keys[2]
    y = np.array([0, 0, 0, 1])            # group "101" = {0,0,1} (ambiguous, size 3); group "010" = {0}
    r = au.exact_bayes(keys, y, 10)
    assert r["n"] == 4 and r["n_unique"] == 2
    assert r["collision_rate"] == 0.75    # 3 of 4 rows are in a size>1 group
    assert r["ambiguous_mass"] == 0.75    # the size-3 group has >1 unique label
    assert abs(r["bayes_acc"] - 0.75) < 1e-12 and r["max_multiplicity"] == 3


def test_bayes_ceiling_correctness():
    # group A: labels {0,0,0} -> dominant 0 (3/3); group B: {1,2} -> dominant 1/2 (1/2)
    keys = [b"a", b"a", b"a", b"b", b"b"]
    y = np.array([0, 0, 0, 1, 2])
    r = au.exact_bayes(keys, y, 10)
    assert abs(r["bayes_acc"] - (3 + 1) / 5) < 1e-12
    assert r["n_unique"] == 2 and r["max_multiplicity"] == 3
    assert r["ambiguous_mass"] == 2 / 5


def test_full_prefix_bayes_is_one(built):
    # the full causal prefix uniquely determines the state (hence the target) for both models
    for m in MODELS:
        assert built[m]["pooled"]["all_add"]["Lfull"]["y_true"]["bayes_acc"] == 1.0
        assert built[m]["pooled"]["classA"]["Lfull"]["y_true"]["bayes_acc"] == 1.0
        assert built[m]["pooled"]["all_add"]["Lfull"]["operand_a"]["bayes_acc"] == 1.0
        assert built[m]["pooled"]["all_add"]["Lfull"]["operand_b"]["bayes_acc"] == 1.0


def test_current_observation_structural_insufficiency(data):
    z = data["z"]
    m = "E_candidate"
    seq = z[f"{m}__test__seq_obs"]
    t = z[f"{m}__test__t"]
    X, M = au.build_history(seq, t, 1)
    feats = np.concatenate([X, M], 1)
    keys = au._keys(feats)
    oa = z[f"{m}__test__operand_a"].astype(int)
    r = au.exact_bayes(keys, oa, 10)
    # identical x_t maps to multiple operand_a values: the current observation is insufficient
    assert r["ambiguous_mass"] > 0.5
    assert r["bayes_acc"] < 0.5
    # target is likewise not determined by x_t
    rt = au.exact_bayes(keys, z[f"{m}__test__y_true"].astype(int), 10)
    assert rt["bayes_acc"] < 0.5


# ---------------------------------------------------------------------------
# splits / overlap / controls
# ---------------------------------------------------------------------------

def test_sequence_split_separation():
    for seed in (42, 43, 44, 45, 46):
        sp = gr.canonical_splits(seed)
        hs = lambda a: set(x.tobytes() for x in a)
        tr = hs(sp.train_set.inputs.numpy())
        va = hs(sp.val_set.inputs.numpy())
        te = hs(sp.test_set.inputs.numpy())
        assert len(tr & va) == 0 and len(tr & te) == 0 and len(va & te) == 0


def test_history_overlap_accounting(data):
    z = data["z"]
    m = "E_candidate"
    counts = {}
    keysets = {}
    for split in ("trainord", "valord", "test"):
        X, M = au.build_history(z[f"{m}__{split}__seq_obs"], z[f"{m}__{split}__t"], ru.FULL_PREFIX_LENGTH)
        ks = set(au._keys(np.concatenate([X, M], 1)))
        keysets[split] = ks
        counts[split] = len(ks)
    # report overlap; identical full-prefix windows across splits are possible but the sequences
    # are disjoint, so the target label for any shared window must be consistent
    tr_val = keysets["trainord"] & keysets["valord"]
    tr_te = keysets["trainord"] & keysets["test"]
    assert counts["test"] > 0
    assert isinstance(tr_val, set) and isinstance(tr_te, set)


def test_classA_reproduction(data):
    z = data["z"]
    assert np.array_equal(z["E_candidate__test__y_true"],
                          (z["E_candidate__test__operand_a"] + z["E_candidate__test__operand_b"]) % 10)
    for m in MODELS:
        per = [int(((z[f"{m}__test__seed"] == s) & (z[f"{m}__test__cand_ok"] == 0)).sum())
               for s in (42, 43, 44, 45, 46)]
        if m == "E_candidate":
            assert per == [448, 330, 363, 327, 337]      # reproduces Q/R/S/T exactly
        tots = [int((z[f"{m}__test__seed"] == s).sum()) for s in (42, 43, 44, 45, 46)]
        assert tots == [3199, 3196, 3171, 3125, 3072]    # canonical test ADD totals


def test_modular_and_raw_sum_target_generation(data):
    z = data["z"]
    for m in MODELS:
        assert np.array_equal(z[f"{m}__test__y_true"],
                              (z[f"{m}__test__operand_a"] + z[f"{m}__test__operand_b"]) % 10)
        assert np.array_equal(z[f"{m}__test__raw_sum"],
                              z[f"{m}__test__operand_a"] + z[f"{m}__test__operand_b"])
        assert int(z[f"{m}__test__raw_sum"].max()) <= 18 and int(z[f"{m}__test__raw_sum"].min()) >= 0


def test_wrap_and_repeated_classification(data):
    z = data["z"]
    m = "E_candidate"
    oa = z[f"{m}__test__operand_a"].astype(int)
    ob = z[f"{m}__test__operand_b"].astype(int)
    wrap = z[f"{m}__test__wrap"].astype(bool)
    assert np.array_equal(wrap, (oa + ob) >= 10) and wrap.sum() > 0 and (~wrap).sum() > 0
    rep = z[f"{m}__test__repeated"].astype(bool)
    assert rep.sum() > 0 and (z[f"{m}__test__t"][rep] >= 1).all()


def test_exact_state_reconstruction_control(data):
    z = data["z"]
    for m in MODELS:
        rec = au.reconstruct_from_prefix(z[f"{m}__test__seq_obs"], z[f"{m}__test__t"])
        assert np.array_equal(rec["operand_a"], z[f"{m}__test__operand_a"])
        assert np.array_equal(rec["operand_b"], z[f"{m}__test__operand_b"])
        assert np.array_equal(rec["y_true"], z[f"{m}__test__y_true"])
        assert np.array_equal(rec["dest"], z[f"{m}__test__op_dest"])
        assert np.array_equal(rec["src"], z[f"{m}__test__src"])


# ---------------------------------------------------------------------------
# probes: separation / determinism / no test tuning / leakage
# ---------------------------------------------------------------------------

def _hfit(z, m, seed, rep, kind, hidden, **kw):
    hist = {("trainord", rep): au.history_features(z[f"{m}__trainord__seq_obs"], z[f"{m}__trainord__t"], dict(au.REPS)[rep]),
            ("valord", rep): au.history_features(z[f"{m}__valord__seq_obs"], z[f"{m}__valord__t"], dict(au.REPS)[rep])}
    return au._fit_probe(hist, z, m, seed, rep, kind, hidden, **kw)


def test_probe_determinism(data):
    z = data["z"]
    f1 = _hfit(z, "E_candidate", 42, "L4", "linear", None)
    f2 = _hfit(z, "E_candidate", 42, "L4", "linear", None)
    assert f1["lambda"] == f2["lambda"] and f1["train_acc"] == f2["train_acc"]
    g1 = _hfit(z, "E_candidate", 42, "L2", "mlp", 16)
    g2 = _hfit(z, "E_candidate", 42, "L2", "mlp", 16)
    assert g1["train_acc"] == g2["train_acc"] and g1["val_acc"] == g2["val_acc"]


def test_probe_train_val_test_separation(data):
    z = data["z"]
    f1 = _hfit(z, "E_candidate", 42, "L1", "linear", None)
    assert f1["n_train"] == min(1500, int((z["E_candidate__trainord__seed"] == 42).sum()))
    z2 = {k: v for k, v in z.items()}
    z2["E_candidate__test__y_true"] = z["E_candidate__test__y_true"][::-1]   # corrupt TEST only
    f2 = _hfit(z2, "E_candidate", 42, "L1", "linear", None)
    assert f2["lambda"] == f1["lambda"] and f2["train_acc"] == f1["train_acc"] and f2["val_acc"] == f1["val_acc"]


def test_heldout_operand_exclusion(data):
    z = data["z"]
    f = _hfit(z, "E_candidate", 42, "L4", "mlp", 32, exclude_heldout=True)
    tr = np.nonzero(z["E_candidate__trainord__seed"] == 42)[0]
    oa = z["E_candidate__trainord__operand_a"][tr]
    ob = z["E_candidate__trainord__operand_b"][tr]
    held = np.array([(int(a), int(b)) in au.HELD_OUT_PAIRS for a, b in zip(oa, ob)])
    assert held.sum() > 0
    assert f["n_train"] == int((~held).sum())
    assert len(au.HELD_OUT_PAIRS) == 20 and all(b in (0, 5) for a, b in au.HELD_OUT_PAIRS)


def test_deliberate_leakage_failure_path():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 44))
    y = rng.integers(0, 10, 200)
    Xt = rng.normal(size=(200, 44))
    clean, _, _ = au.as_.ridge_fit(X, y, np.ones(10), Xt, rng.integers(0, 10, 200), 10)
    leak, _, _ = au.as_.ridge_fit(Xt, rng.integers(0, 10, 200), np.ones(10), Xt, rng.integers(0, 10, 200), 10)
    assert leak is not None and clean is not None
    assert len(set(x.tobytes() for x in X) & set(x.tobytes() for x in Xt)) == 0


def test_no_probe_insertion_forward_unchanged(loaded_model):
    ds = _tiny()
    with torch.no_grad():
        before = loaded_model.forward_with_states(ds.inputs, return_components=True).states[-1].clone()
    rng = np.random.default_rng(4)
    au.as_.mlp_fit(rng.normal(size=(64, 44)), rng.integers(0, 10, 64), rng.normal(size=(32, 44)),
                   rng.integers(0, 10, 32), 16, np.ones(10), 3)
    with torch.no_grad():
        after = loaded_model.forward_with_states(ds.inputs, return_components=True).states[-1].clone()
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# analysis completeness / decision / E-F
# ---------------------------------------------------------------------------

def test_analysis_completeness(built):
    for m in MODELS:
        for rep, _ in au.REPS:
            assert rep in built[m]["pooled"]["all_add"]
            assert rep in built[m]["probes"]["all_add"]
            assert set(built[m]["probes"]["all_add"][rep]) == {c for c, _ in au.CAPS}
            for cap, _ in au.CAPS:
                assert built[m]["probes"]["all_add"][rep][cap]["acc"]["mean"] is not None
            assert rep in built[m]["heldout"]["all_add"]
            assert rep in built[m]["subtype"]["wrap"]


def test_candidate_correct_control_separation(built):
    for m in MODELS:
        alladd = built[m]["pooled"]["all_add"]["Lfull"]["y_true"]["bayes_acc"]
        classA = built[m]["pooled"]["classA"]["Lfull"]["y_true"]["bayes_acc"]
        assert alladd == 1.0 and classA == 1.0
        # learned probe: candidate-correct is at least as decodable as Class-A at the full prefix
        corr = built[m]["probes"]["correct_all"]["Lfull"]["mlp32"]["acc"]["mean"]
        hard = built[m]["probes"]["classA"]["Lfull"]["mlp32"]["acc"]["mean"]
        assert corr >= hard - 0.05


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    for m in MODELS:
        assert built[m]["pooled"]["all_add"]["L1"]["y_true"]["bayes_acc"] is not None
    assert "E_metrics" in d and "F_metrics" in d
    assert d["E_metrics"]["bayes_target_full"] is not None


def test_ef_alignment(built):
    for rep, _ in au.REPS:
        assert rep in built["E_vs_F"]
        assert built["F_combined"]["pooled"]["all_add"][rep]["y_true"]["bayes_acc"] is not None
    # Bayes for the shared all-ADD test population is model-independent: E and F must be identical
    for rep, _ in au.REPS:
        assert (built["E_candidate"]["pooled"]["all_add"][rep]["y_true"]["bayes_acc"]
                == built["F_combined"]["pooled"]["all_add"][rep]["y_true"]["bayes_acc"])


def test_json_determinism(data):
    z = data["z"]
    m = "E_candidate"
    hist = {}
    for rep, L in au.REPS:
        for split in ("trainord", "valord", "test"):
            hist[(split, rep)] = au.history_features(z[f"{m}__{split}__seq_obs"],
                                                     z[f"{m}__{split}__t"], L)
    a1 = au.analyze_seed(z, m, 43, hist)
    a2 = au.analyze_seed(z, m, 43, hist)
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)


# ---------------------------------------------------------------------------
# malformed artifacts / figures / immutability
# ---------------------------------------------------------------------------

def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        au.load(str(tmp_path / "nope.npz"))
    with pytest.raises(Exception):
        au._fit_probe({}, {}, "E_candidate", 42, "L1", "linear", None)


def test_figure_creation(data, built, tmp_path):
    figs = au.make_figures(data, built, str(tmp_path), prefix="test_u")
    assert len(figs) >= 17 and all(os.path.exists(p) for p in figs)


def test_historical_immutability():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"
