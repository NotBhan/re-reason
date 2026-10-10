"""Tests for Experiment X — fixed-data optimization-budget sensitivity (200 vs 1,000 updates).

Resource-light suite: the live reader fits use tiny synthetic tensors and a short update budget; the
structural tests read the collected artifacts. Covers: V/W reader reuse and architecture identity,
fixed training-pool / VAL / TEST identity against W, parameter-count invariance, exact update
counts, deterministic paired seeds, the exact 200-update bridge against W's procedure, VAL-only
checkpoint selection, held-out operand exclusion, sequence disjointness, no duplicate data, causal
history construction, future-observation leakage rejection, causal reader guards, target-generation
correctness, Class-A construction and counts, E/F control, reader/model separation, no gradients /
no mutation on the frozen model, deterministic results and artifacts, figure generation, historical
immutability, and documentation-governance compliance.
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
import analyze_experiment_w as aw  # noqa: E402
import analyze_experiment_x as ax  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_u as ru  # noqa: E402

# Single-threaded torch (set AFTER the imports, since analyze_experiment_v resets it): matches the
# analyzer workers and keeps the tiny live fits fast and bit-reproducible.
torch.set_num_threads(int(os.environ.get("X_TEST_THREADS", "1")))

NPZ = os.path.join(ROOT, "experiments", "experiment_x_optimization_budget.npz")
META = os.path.join(ROOT, "experiments", "experiment_x_optimization_budget_meta.json")
RESULTS = os.path.join(ROOT, "experiments", "experiment_x_results.json")
W_NPZ = os.path.join(ROOT, "experiments", "experiment_w_data_scale.npz")
W_RESULTS = os.path.join(ROOT, "experiments", "experiment_w_results.json")
U_RESULTS = os.path.join(ROOT, "experiments", "experiment_u_results.json")
MODELS = ("E_candidate", "F_combined")
READERS = ["MLP128", "GRU", "TCN"]
BUDGETS = [200, 1000]
ALLOWED_VERDICTS = {"X-A_optimization_limited", "X-B_persistent_bounded", "X-inconclusive_partial"}


def _x_meta():
    if os.path.exists(NPZ):
        return json.loads(str(np.load(NPZ, allow_pickle=False)["meta_json"][0]))
    return None


_XMETA = _x_meta()
TOP = int(_XMETA["source"]["top_size"]) if _XMETA else 10000
# Align the analyzer's ladder/TOP globals with the collected artifact so seed derivation matches W.
ax.TOP = TOP
if _XMETA:
    aw.TRAIN_SIZES = list(_XMETA["source"]["train_sizes"])


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_x npz not present")
    return ax.load(NPZ)


@pytest.fixture(scope="module")
def built(data):
    if not os.path.exists(RESULTS):
        pytest.skip("experiment_x results.json not present")
    with open(RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def u_base():
    return aw.load_u_baseline(U_RESULTS)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


def _tiny(reader, n=64, L=4, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n, av.INPUT_DIM * L) if reader == "MLP128" else (n, L, av.INPUT_DIM)
    X = rng.normal(size=shape).astype(np.float32)
    y = rng.integers(0, 10, n)
    return X, y


# ---------------------------------------------------------------------------
# architecture reuse / parameter counts
# ---------------------------------------------------------------------------

def test_reader_architecture_reuse():
    arch = av.architecture()
    assert arch["GRU"]["hidden"] == 85 and arch["TCN"]["channels"] == 52
    assert arch["TCN"]["kernel"] == 3 and arch["TCN"]["dilations"] == [1, 2, 4, 8]
    assert arch["TCN"]["n_layers"] == 4
    # X builds readers through V's definitions (no reconstruction)
    for reader in ("GRU", "TCN"):
        m = ax.build_model(reader, 10, av.INPUT_DIM, arch)
        assert type(m) is type(av.build_reader(reader, 10, arch))


def test_parameter_counts():
    arch = av.architecture()
    assert as_.param_count("mlp", 128, 220) == 29578
    assert av.reader_param_count("GRU", 10, arch) == 28655
    assert av.reader_param_count("TCN", 10, arch) == 28506
    for R in ("GRU", "TCN"):
        assert av.reader_param_count(R, 19, arch) <= arch["budget_u_mlp128"]


def test_parameter_count_invariance(built):
    # the parameter count of each reader is identical at both update budgets (only updates change)
    for R in READERS:
        for rep in ("L4", "Lfull"):
            vals = [built["readers"][R]["y_true"][rep]["main"][str(b)]["params"] for b in BUDGETS]
            assert len(set(vals)) == 1, (R, rep, vals)


def test_budget_metadata(built):
    md = built["metadata"]
    assert md["budgets"] == BUDGETS
    assert md["reader_lr"] == av.READER_LR == 3e-3
    assert md["reader_wd"] == av.READER_WD == 1e-2
    assert md["statistical_discipline"]
    assert built["w_bridge"]["max_abs_diff"] < 1e-9          # X's 200 budget reproduces W exactly


# ---------------------------------------------------------------------------
# fixed data / split identity against W
# ---------------------------------------------------------------------------

def test_fixed_training_pool_identity(data):
    assert os.path.exists(W_NPZ)
    xz, wz = data["z"], np.load(W_NPZ, allow_pickle=False)
    for c in ax_art_cols():
        k = f"pool{TOP}__{c}"
        assert np.array_equal(xz[k], wz[k]), k
    # recorded per-array hashes reproduce
    for k, rec in data["meta"]["identity"].items():
        assert hashlib.sha256(np.ascontiguousarray(xz[k]).tobytes()).hexdigest() == rec["sha256"]


def ax_art_cols():
    return ["seq_obs", "t", "y_true", "raw_sum", "operand_a", "operand_b", "op_dest",
            "wrap", "repeated", "seq", "seed"]


def test_validation_identity_against_w(data):
    xz, wz = data["z"], np.load(W_NPZ, allow_pickle=False)
    for c in ax_art_cols():
        assert np.array_equal(xz[f"val__{c}"], wz[f"val__{c}"])
    assert data["meta"]["n"]["val"] == 5 * 800


def test_test_identity_against_w(data):
    xz, wz = data["z"], np.load(W_NPZ, allow_pickle=False)
    for m in MODELS:
        for c in ax_art_cols() + ["cand_ok", "m_cand", "norm_C", "norm_dC"]:
            assert np.array_equal(xz[f"{m}__test__{c}"], wz[f"{m}__test__{c}"])
        assert xz[f"{m}__test__y_true"].shape[0] == 15763


def test_source_npz_hash_recorded(data):
    assert hashlib.sha256(open(W_NPZ, "rb").read()).hexdigest() == data["meta"]["source"]["sha256"]


def test_sequence_disjointness(data):
    d = data["meta"]["disjointness"]
    assert all(v == 0 for v in d.values()), d


def test_no_duplicate_data(data):
    for _, rec in data["meta"]["per_seed"].items():
        assert rec["n_train"] == TOP
        assert rec["n_duplicate_rows"] == 0
        assert rec["n_unique_seq_t"] == rec["n_train"]


def test_heldout_operand_exclusion(data):
    z = data["z"]
    rows = np.nonzero(z[f"pool{TOP}__seed"] == 42)[0]
    oa = z[f"pool{TOP}__operand_a"][rows]
    ob = z[f"pool{TOP}__operand_b"][rows]
    held = np.array([(int(a), int(b)) in au.HELD_OUT_PAIRS for a, b in zip(oa, ob)])
    assert held.sum() > 0
    assert data["meta"]["per_seed"]["42"]["n_heldout_pair_rows"] == int(held.sum())
    assert len(au.HELD_OUT_PAIRS) == 20 and all(b in (0, 5) for _, b in au.HELD_OUT_PAIRS)


def test_target_generation(data):
    z = data["z"]
    assert np.array_equal(z[f"pool{TOP}__y_true"],
                          (z[f"pool{TOP}__operand_a"] + z[f"pool{TOP}__operand_b"]) % 10)
    assert np.array_equal(z[f"pool{TOP}__raw_sum"],
                          z[f"pool{TOP}__operand_a"] + z[f"pool{TOP}__operand_b"])


def test_classA_construction_and_counts(data):
    z = data["z"]
    per = [int(((z["E_candidate__test__seed"] == s) & (z["E_candidate__test__cand_ok"] == 0)).sum())
           for s in aw.SEEDS]
    assert per == [448, 330, 363, 327, 337]
    assert data["meta"]["per_seed"]["42"]["classA"] == 448


# ---------------------------------------------------------------------------
# causal history / leakage guards
# ---------------------------------------------------------------------------

def test_causal_history_construction(data):
    z = data["z"]
    rows = np.nonzero(z[f"pool{TOP}__seed"] == 42)[0][:400]
    seq = z[f"pool{TOP}__seq_obs"][rows]
    t = z[f"pool{TOP}__t"][rows]
    X, M = au.build_history(seq, t, 4)
    full = seq.reshape(len(rows), ru.T_MAX, ru.OBS_DIM)
    for i in range(len(rows)):
        assert int(M[i].sum()) == min(4, int(t[i]) + 1)
        assert np.allclose(X[i][-ru.OBS_DIM:], full[i, int(t[i])])


def test_no_future_leakage(data):
    z = data["z"]
    rows = np.nonzero(z[f"pool{TOP}__seed"] == 43)[0][:400]
    seq = z[f"pool{TOP}__seq_obs"][rows]
    t = z[f"pool{TOP}__t"][rows]
    rng = np.random.default_rng(0)
    X1, M1 = au.build_history(seq, t, 8)
    mod = seq.copy().reshape(len(rows), ru.T_MAX, ru.OBS_DIM)
    for i in range(len(rows)):
        ti = int(t[i])
        if ti + 1 < ru.T_MAX:
            mod[i, ti + 1:] = rng.random((ru.T_MAX - ti - 1, ru.OBS_DIM))
    X2, M2 = au.build_history(mod.reshape(len(rows), -1), t, 8)
    assert np.array_equal(X1, X2) and np.array_equal(M1, M2)


def test_causal_reader_guards():
    assert av.GRUReader(av.architecture()["GRU"]["hidden"]).gru.bidirectional is False
    torch.manual_seed(0)
    X = torch.randn(1, 6, av.INPUT_DIM)
    bidir = torch.nn.GRU(av.INPUT_DIM, 8, batch_first=True, bidirectional=True)
    with torch.no_grad():
        o1 = bidir(X)[0]
        X2 = X.clone()
        X2[:, 4, :] = torch.randn(av.INPUT_DIM)
        o2 = bidir(X2)[0]
    assert not torch.allclose(o1[:, 2], o2[:, 2])


# ---------------------------------------------------------------------------
# optimizer budget / seed / bridge / VAL-only selection
# ---------------------------------------------------------------------------

def test_deterministic_paired_seed():
    assert ax.TOP == TOP
    for R in READERS:
        assert ax._fit_seed(42, R, "y_true", "L4", "main") == \
            aw.reader_seed(42, R, "y_true", "L4", TOP, "main")
    combos = {ax._fit_seed(s, R, t, rep, v)
              for s in aw.SEEDS for R in READERS for t in ("y_true", "operand_a")
              for rep in ("L4", "Lfull") for v in ("main", "heldout")}
    assert len(combos) == 5 * 3 * 2 * 2 * 2


def test_exact_update_budgets(built):
    for R in READERS:
        for rep in ("L4", "Lfull"):
            for b in BUDGETS:
                node = built["readers"][R]["y_true"][rep]["main"][str(b)]
                assert node["updates"] == b
                assert 0 < node["best_idx"]["mean"] <= b


def test_exactly_1000_updates_run():
    # the trainer runs exactly `updates` steps and never repeats/reweights examples
    X, y = _tiny("MLP128", 64, 4, 5)
    fit = ax.train_trajectory("MLP128", X, y, X, y, 10, 3, 40, av.architecture())
    assert fit["events"][-1]["update"] == 40
    assert len(fit["events"]) == 4          # eval every 10 -> updates 10,20,30,40


@pytest.mark.parametrize("reader", ["MLP128", "GRU", "TCN"])
def test_update_200_bridge_against_w_procedure(reader):
    # X's 200-update (best-on-VAL) trajectory reproduces U/V/W's reader training exactly: same
    # best-on-VAL accuracy AND identical predictions on the fitting tensor.
    X, y = _tiny(reader, 64, 4, 1)
    arch = av.architecture()
    fit = ax.train_trajectory(reader, X, y, X, y, 10, 7, 200, arch)
    indim = X.shape[1] if reader == "MLP128" else av.INPUT_DIM
    pred_x = ax.predict_state(reader, fit["best_200"][1], X, fit["norm"], 10, indim, arch)
    if reader == "MLP128":
        wc = as_.class_weights(y, 10)
        predict, v = as_.mlp_fit(X.astype(np.float64), y, X.astype(np.float64), y, 128, wc, 7, 10)
        pred_w = predict(X).argmax(1)
    else:
        m, v = av.train_reader(reader, X, y, X, y, 10, 7, arch)
        pred_w = av.reader_predict(m, X)
    assert fit["best_200"][0] == v
    assert np.array_equal(pred_x, pred_w)


def test_validation_only_checkpoint_selection():
    Xtr, ytr = _tiny("GRU", 96, 4, 2)
    Xva, yva = _tiny("GRU", 48, 4, 3)
    arch = av.architecture()
    fit = ax.train_trajectory("GRU", Xtr, ytr, Xva, yva, 10, 5, 200, arch)
    assert abs(fit["best_all"][0] - max(e["val_acc"] for e in fit["events"])) < 1e-12
    pred = ax.predict_state("GRU", fit["best_all"][1], Xva, fit["norm"], 10, av.INPUT_DIM, arch)
    assert abs(float((pred == yva).mean()) - fit["best_all"][0]) < 1e-6


def test_deterministic_trajectory():
    X, y = _tiny("GRU", 64, 4, 0)
    a = ax.train_trajectory("GRU", X, y, X, y, 10, 11, 60, av.architecture())
    b = ax.train_trajectory("GRU", X, y, X, y, 10, 11, 60, av.architecture())
    assert [e["val_acc"] for e in a["events"]] == [e["val_acc"] for e in b["events"]]
    assert a["best_all"][0] == b["best_all"][0]


# ---------------------------------------------------------------------------
# completeness of the produced analysis
# ---------------------------------------------------------------------------

def test_all_readers_present(built):
    assert set(built["readers"]) == set(READERS)


def test_all_budgets_present(built):
    for R in READERS:
        for rep in aw.REP_LABELS:
            for b in BUDGETS:
                assert str(b) in built["readers"][R]["y_true"][rep]["main"]


def test_heldout_outputs_present(built):
    for R in READERS:
        for rep in ("L4", "Lfull"):
            node = built["readers"][R]["y_true"][rep]["heldout"]["1000"]
            assert node["seen"]["E_candidate:classA"]["mean"] is not None
            assert node["heldout"]["E_candidate:classA"]["mean"] is not None


def test_ef_control(built):
    for R in READERS:
        for rep in aw.REP_LABELS:
            node = built["readers"][R]["y_true"][rep]["main"]["1000"]["acc"]
            assert abs(node["E_candidate:all_add"]["mean"] - node["F_combined:all_add"]["mean"]) < 1e-6


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    assert d["best_1000_all"] is not None and d["max_train_1000"] is not None
    assert d["thresholds"]["recover_bar"] == 0.50 and d["thresholds"]["train_low_bar"] == 0.50
    for R in READERS:
        assert R in d["per_reader"]


def test_u_baseline_loading(u_base):
    for m in MODELS:
        assert "L4" in u_base[m] and "Lfull" in u_base[m]
    assert abs(u_base["E_candidate"]["L4"] - 0.999) < 0.01


def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        ax.load(str(tmp_path / "nope.npz"))


# ---------------------------------------------------------------------------
# frozen-model protection
# ---------------------------------------------------------------------------

def test_no_reason_parameter_mutation_and_no_grad(loaded_model):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    X, y = _tiny("TCN", 64, 4, 0)
    ax.train_trajectory("TCN", X, y, X, y, 10, 5, 20, av.architecture())
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


def test_reader_model_separation(loaded_model):
    X, y = _tiny("GRU", 64, 4, 0)
    model = ax.build_model("GRU", 10, av.INPUT_DIM, av.architecture())
    ids = {id(p) for p in loaded_model.parameters()}
    assert not (ids & {id(p) for p in model.parameters()})


# ---------------------------------------------------------------------------
# figures / immutability / governance
# ---------------------------------------------------------------------------

def test_figure_generation(data, built, tmp_path):
    figs = ax.make_figures(data, built, str(tmp_path), prefix="test_x")
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
    assert "experiment_x_optimization_budget.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_x_optimization_budget.md" in baseline["reports"]
