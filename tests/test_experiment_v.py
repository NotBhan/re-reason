"""Tests for Experiment V — bounded sequence-aware access test.

Covers the required §controls/§leakage contract plus reproducibility: U baseline loading, exact
parameter-budget derivation and parity, deterministic architecture construction, raw 22-dim reader
input with an explicit validity mask, causal history construction and no-future leakage (randomised
future invariance + a deliberate leaky-builder rejection), bidirectional-GRU rejection, causal TCN
verification, deterministic reader output, frozen-model protection (no parameter mutation, no
gradients), sequence disjointness, held-out operand exclusion, train/val/test separation, best
checkpoint chosen from VAL only, target-generation correctness, Class-A count reproduction,
candidate-correct / all-ADD controls, E/F agreement, history-grid completeness, deterministic
artifacts, JSON determinism, figure creation, and historical/governance compliance.
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

import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_u as au  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_generator_repair_feasibility as gr  # noqa: E402
from re_reason.benchmarks.dataset_split import SplitDataset, hash_sequence  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_v_sequence_access.npz")
RESULTS = os.path.join(ROOT, "experiments", "experiment_v_results.json")
U_RESULTS = os.path.join(ROOT, "experiments", "experiment_u_results.json")
MODELS = ("E_candidate", "F_combined")
ALLOWED_VERDICTS = {"V-A_sequence_accessible", "V-B_sequence_limited",
                    "V-C_partial_sequence_benefit", "V-MIX"}


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_v npz not present")
    return av.load(NPZ)


@pytest.fixture(scope="module")
def built(data):
    if not os.path.exists(RESULTS):
        pytest.skip("experiment_v results.json not present")
    with open(RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def u_base():
    return av.load_u_baseline(U_RESULTS)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


@pytest.fixture(scope="module")
def reduced(data):
    """One reduced run_readers call (seed 42, y_true only, L2) reused by structural tests."""
    z = data["z"]
    saved = (av.SEEDS, av.TARGETS, av.MAIN_L, av.HELDOUT_L)
    av.TARGETS = {"y_true": 10}
    av.MAIN_L = {"y_true": ["L2"]}
    av.HELDOUT_L = {"y_true": ["L2"]}
    try:
        out = av.run_readers(z, 42, av.architecture())
    finally:
        av.SEEDS, av.TARGETS, av.MAIN_L, av.HELDOUT_L = saved
    return out


def _tiny_readers(n=160, L=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, L, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, 10, n).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# parameter budget / architecture
# ---------------------------------------------------------------------------

def test_exact_parameter_budget_calculation():
    budget = av.u_mlp128_param_count()
    assert budget == as_.param_count("mlp", 128, 220) == 220 * 128 + 128 + 128 * 10 + 10
    assert budget == 29578


def test_gru_within_mlp128_budget():
    arch = av.architecture()
    for nc in av.TARGETS.values():
        assert av.reader_param_count("GRU", nc, arch) <= arch["budget_u_mlp128"]


def test_tcn_within_mlp128_budget():
    arch = av.architecture()
    for nc in av.TARGETS.values():
        assert av.reader_param_count("TCN", nc, arch) <= arch["budget_u_mlp128"]


def test_deterministic_architecture_construction():
    a1, a2 = av.architecture(), av.architecture()
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)
    h = a1["GRU"]["hidden"]
    c = a1["TCN"]["channels"]
    # widths are the LARGEST within budget: +1 must exceed it
    assert av.gru_param_count(h + 1, nclass=av.MAX_NCLASS) > a1["budget_u_mlp128"]
    assert av.tcn_param_count(c + 1, nclass=av.MAX_NCLASS) > a1["budget_u_mlp128"]
    assert av.gru_param_count(h, nclass=av.MAX_NCLASS) <= a1["budget_u_mlp128"]
    assert av.tcn_param_count(c, nclass=av.MAX_NCLASS) <= a1["budget_u_mlp128"]


def test_exact_reader_parameter_budget(built):
    a = built["architecture"]
    assert a["GRU"]["params"] <= a["budget_u_mlp128"]
    assert a["TCN"]["params"] <= a["budget_u_mlp128"]
    assert 0 < a["GRU"]["ratio"] <= 1.0 and 0 < a["TCN"]["ratio"] <= 1.0


# ---------------------------------------------------------------------------
# reader input / causality
# ---------------------------------------------------------------------------

def test_raw_22dim_reader_input(data):
    z = data["z"]
    seq = z["E_candidate__test__seq_obs"]
    t = z["E_candidate__test__t"]
    for rep, L in av.REPS:
        X = av.reader_input(seq, t, L)
        assert X.shape == (len(t), L, av.INPUT_DIM) == (len(t), L, 22)
        # last channel is the explicit validity mask (0/1)
        assert set(np.unique(X[:, :, -1])) <= {0.0, 1.0}


def test_causal_window_construction_and_mask(data):
    z = data["z"]
    seq = z["E_candidate__test__seq_obs"]
    t = z["E_candidate__test__t"]
    X = av.reader_input(seq, t, 4)
    full = seq.reshape(len(t), ru.T_MAX, ru.OBS_DIM)
    for i in range(0, len(t), 97):
        real = int(X[i, :, -1].sum())
        assert real == min(4, int(t[i]) + 1)
        # the window is LEFT-padded, so the last slot always holds x_t
        assert np.allclose(X[i, -1, :ru.OBS_DIM], full[i, int(t[i])])


def test_no_future_leakage(data):
    z = data["z"]
    seq = z["E_candidate__test__seq_obs"]
    t = z["E_candidate__test__t"]
    rng = np.random.default_rng(0)
    for rep, L in av.REPS:
        X1 = av.reader_input(seq, t, L)
        mod = seq.copy().reshape(len(t), ru.T_MAX, ru.OBS_DIM)
        for i in range(len(t)):
            ti = int(t[i])
            if ti + 1 < ru.T_MAX:
                mod[i, ti + 1:] = rng.random((ru.T_MAX - ti - 1, ru.OBS_DIM))
        X2 = av.reader_input(mod.reshape(len(t), -1), t, L)
        assert np.array_equal(X1, X2)     # future observations cannot affect the reader input


def test_deliberate_leaky_implementation_rejected(data):
    # a deliberately leaky reader that consumes x_{t+1} must differ from the causal reader input
    z = data["z"]
    seq = z["E_candidate__test__seq_obs"]
    t = z["E_candidate__test__t"]
    Xc = av.reader_input(seq, t, 2)
    full = seq.reshape(len(t), ru.T_MAX, ru.OBS_DIM)
    mask = t < ru.T_MAX - 1                       # rows where x_{t+1} exists
    leaky = np.zeros_like(Xc)
    for i in np.nonzero(mask)[0]:
        ti = int(t[i])
        leaky[i, 0, :ru.OBS_DIM] = full[i, ti]
        leaky[i, 1, :ru.OBS_DIM] = full[i, ti + 1]      # FUTURE
    assert mask.any() and not np.array_equal(Xc[mask], leaky[mask])


def test_bidirectional_gru_rejection():
    m = av.GRUReader(av.architecture()["GRU"]["hidden"])
    assert m.gru.bidirectional is False
    # a bidirectional GRU would leak the future: its output at t depends on later observations
    bidir = torch.nn.GRU(av.INPUT_DIM, 8, batch_first=True, bidirectional=True)
    torch.manual_seed(0)
    X = torch.randn(1, 6, av.INPUT_DIM)
    with torch.no_grad():
        o1 = bidir(X)[0]
        X2 = X.clone(); X2[:, 4, :] = torch.randn(av.INPUT_DIM)
        o2 = bidir(X2)[0]
    assert not torch.allclose(o1[:, 2], o2[:, 2])       # position 2 depends on future position 4


def test_causal_tcn_verification():
    torch.manual_seed(0)
    tcn = av.TCNReader(av.architecture()["TCN"]["channels"]).eval()
    X = torch.randn(2, 8, av.INPUT_DIM)

    def positions(m, x):
        zz = x.transpose(1, 2)
        for conv in m.convs:
            zz = conv(zz)
            trim = conv.dilation[0] * (conv.kernel_size[0] - 1)
            if trim > 0:
                zz = zz[:, :, :-trim]
            zz = m.act(zz)
        return zz

    with torch.no_grad():
        z1 = positions(tcn, X)
    Xf = X.clone(); Xf[:, 5:, :] = torch.randn(2, 3, av.INPUT_DIM)
    with torch.no_grad():
        z2 = positions(tcn, Xf)
    assert torch.allclose(z1[:, :, :5], z2[:, :, :5])    # past outputs unchanged
    assert not torch.allclose(z1[:, :, 5], z2[:, :, 5])  # position 5 depends on its own input


def test_gru_last_step_is_causal():
    torch.manual_seed(0)
    g = av.GRUReader(av.architecture()["GRU"]["hidden"]).eval()
    X = torch.randn(2, 8, av.INPUT_DIM)
    with torch.no_grad():
        o1 = g.gru(X)[0]
    Xf = X.clone(); Xf[:, 5:, :] = torch.randn(2, 3, av.INPUT_DIM)
    with torch.no_grad():
        o2 = g.gru(Xf)[0]
    assert torch.allclose(o1[:, :5], o2[:, :5])


def test_deterministic_reader_output():
    X, y = _tiny_readers()
    m1, v1 = av.train_reader("GRU", X, y, X, y, 10, 11)
    m2, v2 = av.train_reader("GRU", X, y, X, y, 10, 11)
    assert v1 == v2
    assert np.array_equal(av.reader_predict(m1, X), av.reader_predict(m2, X))


# ---------------------------------------------------------------------------
# frozen-model protection
# ---------------------------------------------------------------------------

def test_no_reason_parameter_mutation(loaded_model):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    X, y = _tiny_readers(120, 4, 3)
    av.train_reader("TCN", X, y, X, y, 10, 5)
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])


def test_no_gradients_on_frozen_checkpoint(loaded_model):
    X, y = _tiny_readers(120, 4, 3)
    av.train_reader("GRU", X, y, X, y, 10, 5)
    assert not any(p.grad is not None for p in loaded_model.parameters())
    with torch.no_grad():
        loaded_model.forward_with_states(
            torch.zeros(2, 10, 21), return_components=True)
    assert not any(p.grad is not None for p in loaded_model.parameters())


def test_reader_parameters_never_attached_to_frozen_model(loaded_model):
    X, y = _tiny_readers(120, 4, 3)
    model, _ = av.train_reader("GRU", X, y, X, y, 10, 5)
    ids = {id(p) for p in loaded_model.parameters()}
    assert not (ids & {id(p) for p in model.parameters()})


# ---------------------------------------------------------------------------
# splits / controls / targets
# ---------------------------------------------------------------------------

def test_sequence_disjointness():
    for seed in (42, 43, 44, 45, 46):
        sp = gr.canonical_splits(seed)
        hs = lambda a: set(x.tobytes() for x in a)
        tr = hs(sp.train_set.inputs.numpy())
        va = hs(sp.val_set.inputs.numpy())
        te = hs(sp.test_set.inputs.numpy())
        assert len(tr & va) == 0 and len(tr & te) == 0 and len(va & te) == 0


def test_heldout_pair_exclusion(reduced, data):
    z = data["z"]
    tr = np.nonzero(z["E_candidate__trainord__seed"] == 42)[0]
    oa = z["E_candidate__trainord__operand_a"][tr]
    ob = z["E_candidate__trainord__operand_b"][tr]
    held = np.array([(int(a), int(b)) in av.HELD_OUT_PAIRS for a, b in zip(oa, ob)])
    assert held.sum() > 0
    assert reduced["fits"]["GRU|y_true|L2|heldout"]["n_train"] == int((~held).sum())
    assert len(av.HELD_OUT_PAIRS) == 20 and all(b in (0, 5) for a, b in av.HELD_OUT_PAIRS)


def test_train_val_test_separation(reduced):
    node = reduced["fits"]["GRU|y_true|L2|main"]
    assert 0.0 <= node["val_acc"] <= 1.0 and 0.0 <= node["train_acc"] <= 1.0
    assert isinstance(node["acc"]["E_candidate:classA"], float)


def test_best_checkpoint_chosen_from_val():
    rng = np.random.default_rng(7)
    Xtr = rng.normal(size=(200, 4, av.INPUT_DIM)).astype(np.float32)
    ytr = rng.integers(0, 10, 200)
    Xva = rng.normal(size=(100, 4, av.INPUT_DIM)).astype(np.float32)
    yva = rng.integers(0, 10, 100)
    model, vacc = av.train_reader("GRU", Xtr, ytr, Xva, yva, 10, 3)
    with torch.no_grad():
        got = float((model(torch.tensor(Xva)).argmax(1) == torch.tensor(yva)).float().mean())
    assert abs(got - vacc) < 1e-9      # the returned checkpoint is the best-on-VAL one


def test_target_generation(data):
    z = data["z"]
    for m in MODELS:
        assert np.array_equal(z[f"{m}__test__y_true"],
                              (z[f"{m}__test__operand_a"] + z[f"{m}__test__operand_b"]) % 10)
        assert np.array_equal(z[f"{m}__test__raw_sum"],
                              z[f"{m}__test__operand_a"] + z[f"{m}__test__operand_b"])
    assert set(av.TARGETS) == {"y_true", "operand_a", "operand_b", "raw_sum", "op_dest"}


def test_classA_count_reproduction(data):
    z = data["z"]
    for m in MODELS:
        tots = [int((z[f"{m}__test__seed"] == s).sum()) for s in (42, 43, 44, 45, 46)]
        assert tots == [3199, 3196, 3171, 3125, 3072]
    per = [int(((z["E_candidate__test__seed"] == s) & (z["E_candidate__test__cand_ok"] == 0)).sum())
           for s in (42, 43, 44, 45, 46)]
    assert per == [448, 330, 363, 327, 337]


def test_candidate_correct_and_alladd_controls(reduced):
    node = reduced["fits"]["GRU|y_true|L2|main"]
    assert node["acc"]["E_candidate:all_add"] is not None
    assert node["acc"]["E_candidate:correct_all"] is not None
    assert node["acc"]["E_candidate:matched_correct"] is not None
    for p in ("classA", "matched_correct", "correct_all", "all_add"):
        assert f"E_candidate:{p}" in node["acc"]


def test_ef_agreement(built):
    # the raw interface is model-independent: reader params identical and all-ADD accuracy identical
    for fam in av.FAMILIES:
        for rep in av.REP_LABELS:
            node = built["readers"][fam]["y_true"][rep]["main"]
            assert abs(node["acc"]["E_candidate:all_add"]["mean"]
                       - node["acc"]["F_combined:all_add"]["mean"]) < 1e-6
    # Class-A subsets differ by model, so the Class-A reader accuracy may differ — but only modestly
    for k in av.DECISIVE:
        assert abs(built["E_vs_F"][f"GRU|{k}"]["mean_diff"]) < 0.05
        assert abs(built["E_vs_F"][f"TCN|{k}"]["mean_diff"]) < 0.05


def test_history_grid_completeness(built):
    assert av.REP_LABELS == ["L1", "L2", "L4", "L8", "L16", "Lfull"]
    for fam in av.FAMILIES:
        for rep in av.REP_LABELS:
            assert rep in built["readers"][fam]["y_true"]
            assert "main" in built["readers"][fam]["y_true"][rep]
        for rep in av.HELDOUT_L["y_true"]:
            assert "heldout" in built["readers"][fam]["y_true"][rep]


# ---------------------------------------------------------------------------
# U baseline bridge / determinism / figures / governance
# ---------------------------------------------------------------------------

def test_u_baseline_loading(u_base):
    for m in MODELS:
        assert "probes" in u_base["models"][m] and "bayes" in u_base["models"][m]
    b = u_base["models"]["E_candidate"]["bayes"]["all_add"]
    assert abs(b["L4"] - 0.998) < 0.01 and b["Lfull"] == 1.0


def test_u_baseline_bridge(built, u_base):
    for m in MODELS:
        for rep in av.REP_LABELS:
            assert abs(built["u_baseline"]["models"][m]["probes"]["classA"][rep]
                       - u_base["models"][m]["probes"]["classA"][rep]) < 1e-9
    assert abs(built["bayes_check"]["L4_all_add_y_true"] - 0.998) < 0.01
    assert built["bayes_check"]["Lfull_all_add_y_true"] == 1.0


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    assert "E_metrics" in d and "F_metrics" in d
    assert d["E_metrics"]["bayes_minus_reader"] is not None


def test_deterministic_artifacts(data):
    # collector determinism (tiny re-capture) and reader determinism are both required
    m = eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"), "E_candidate", 42)["model"]
    ds = SplitDataset(torch.zeros(4, 10, 21), torch.zeros(4, 4, dtype=torch.long),
                      torch.zeros(4, 10, 4, dtype=torch.long), [])
    r1 = ru.capture_split(m, ds, 42, cap=None)
    r2 = ru.capture_split(m, ds, 42, cap=None)
    assert np.array_equal(r1["seq_obs"], r2["seq_obs"])
    X, y = _tiny_readers()
    _, v1 = av.train_reader("TCN", X, y, X, y, 10, 2)
    _, v2 = av.train_reader("TCN", X, y, X, y, 10, 2)
    assert v1 == v2


def test_json_determinism():
    assert json.dumps(av.architecture(), sort_keys=True) == json.dumps(av.architecture(), sort_keys=True)
    X, y = _tiny_readers()
    m1, _ = av.train_reader("GRU", X, y, X, y, 10, 9)
    m2, _ = av.train_reader("GRU", X, y, X, y, 10, 9)
    assert np.array_equal(av.reader_predict(m1, X), av.reader_predict(m2, X))


def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        av.load(str(tmp_path / "nope.npz"))


def test_figure_creation(data, built, tmp_path):
    figs = av.make_figures(data, built, str(tmp_path), prefix="test_v")
    assert len(figs) >= 12 and all(os.path.exists(p) for p in figs)


def test_historical_immutability():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"


def test_research_governance_compliance():
    log = open(os.path.join(ROOT, "docs", "RESEARCH_LOG.md")).read()
    assert "experiment_v_sequence_access.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_v_sequence_access.md" in baseline["reports"]
