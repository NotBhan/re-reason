"""Tests for Experiment W — bounded-reader learnability vs training-data scale.

Resource-light suite. The live reader fits use a handful of epochs on tiny synthetic tensors
instead of the full 200-epoch protocol (which is verified from the stored JSON), and the
training-data ladder is read from the collected artifact rather than hardcoded — so the suite runs
on modest hardware while still covering the whole contract.

Covers: V architecture reuse, parameter-count invariance, ladder consistency, declared-stream
reproduction of the collected pools, sequence disjointness, VAL/TEST invariance, no duplicate
training rows, held-out operand exclusion, causal history correctness, no-future leakage, causal
reader guards, exact target generation, Class-A count reproduction, reader/optimizer invariance,
exactly 200 optimizer updates, best-checkpoint-from-VAL, deterministic fit seeds, frozen-model
protection, reader/model separation, history/scale/reader completeness, deterministic artifacts,
malformed-artifact rejection, figure creation, and historical/governance compliance.
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
import run_experiment_g as eg  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_experiment_w as rw  # noqa: E402

NPZ = os.path.join(ROOT, "experiments", "experiment_w_data_scale.npz")
RESULTS = os.path.join(ROOT, "experiments", "experiment_w_results.json")
U_RESULTS = os.path.join(ROOT, "experiments", "experiment_u_results.json")
MODELS = ("E_candidate", "F_combined")
READERS = ["MLP128", "GRU", "TCN"]
ALLOWED_VERDICTS = {"W-A_data_limited", "W-B_data_saturated_bounded",
                    "W-C_data_benefit_no_recovery", "W-MIX"}


def _collected_sizes():
    """Training-data ladder actually stored in the collected artifact (falls back to the runner)."""
    if os.path.exists(NPZ):
        meta = json.loads(str(np.load(NPZ, allow_pickle=False)["meta_json"][0]))
        return list(meta["train_sizes"])
    return list(rw.TRAIN_SIZES)


SIZES = _collected_sizes()
TOP = SIZES[-1]
# Align the analyzer's ladder global with the collected artifact so seed derivation is consistent.
aw.TRAIN_SIZES = list(SIZES)


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_w npz not present")
    return aw.load(NPZ)


@pytest.fixture(scope="module")
def built(data):
    if not os.path.exists(RESULTS):
        pytest.skip("experiment_w results.json not present")
    with open(RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def u_base():
    return aw.load_u_baseline(U_RESULTS)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


@pytest.fixture
def fast_fit(monkeypatch):
    """Live fits use a few epochs (12 >= the ep%10==9 VAL checkpoint) instead of the full 200."""
    monkeypatch.setattr(av, "READER_EPOCHS", 12)
    monkeypatch.setattr(as_, "MLP_EPOCHS", 12)
    yield


def _tiny(reader, n=96, L=4, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n, av.INPUT_DIM * L) if reader == "MLP128" else (n, L, av.INPUT_DIM)
    X = rng.normal(size=shape).astype(np.float32)
    y = rng.integers(0, 10, n)
    return X, y


# ---------------------------------------------------------------------------
# architecture reuse / parameter invariance
# ---------------------------------------------------------------------------

def test_reader_architecture_reuse():
    # W must reuse V's reader classes / widths literally
    arch = av.architecture()
    assert arch["GRU"]["hidden"] == 85 and arch["TCN"]["channels"] == 52
    assert arch["TCN"]["kernel"] == 3 and arch["TCN"]["dilations"] == [1, 2, 4, 8]
    assert hasattr(av, "GRUReader") and hasattr(av, "TCNReader")
    assert hasattr(av, "train_reader") and hasattr(av, "reader_input")


def test_parameter_counts():
    arch = av.architecture()
    assert as_.param_count("mlp", 128, 220) == 29578
    assert av.reader_param_count("GRU", 10, arch) == 28655
    assert av.reader_param_count("TCN", 10, arch) == 28506
    for R in ("GRU", "TCN"):
        assert av.reader_param_count(R, 19, arch) <= arch["budget_u_mlp128"]


def test_reader_architecture_invariance(built):
    # the parameter count of each reader is identical across every collected training scale
    for R in READERS:
        for rep in ("L4", "Lfull"):
            vals = [built["readers"][R]["y_true"][rep][str(S)]["main"]["params"] for S in SIZES]
            assert len(set(vals)) == 1, (R, rep, vals)


def test_optimizer_configuration_invariance(built):
    md = built["metadata"]
    assert md["reader_epochs"] == av.READER_EPOCHS == 200
    assert md["reader_lr"] == av.READER_LR == 3e-3
    assert md["reader_wd"] == av.READER_WD == 1e-2


def test_exactly_200_updates_every_scale(built):
    for R in READERS:
        for S in SIZES:
            node = built["readers"][R]["y_true"]["L4"][str(S)]
            assert node["main"]["updates"] == 200
            assert node["heldout"]["updates"] == 200


def test_deterministic_fit_seed():
    a = aw.reader_seed(42, "GRU", "y_true", "L4", SIZES[0], "main")
    assert a == aw.reader_seed(42, "GRU", "y_true", "L4", SIZES[0], "main")
    combos = {aw.reader_seed(s, R, t, rep, S, v)
              for s in aw.SEEDS for R in READERS for t in ("y_true", "operand_a")
              for rep in ("L4", "Lfull") for S in SIZES for v in ("main", "heldout")}
    assert len(combos) == 5 * 3 * 2 * 2 * len(SIZES) * 2      # all distinct


# ---------------------------------------------------------------------------
# training-data generation / accounting
# ---------------------------------------------------------------------------

def test_training_size_generation_exact_counts(data):
    meta = data["meta"]
    base, off = meta["pool_stream_base"], meta["pool_stream_offsets"]
    for S in SIZES:
        pool = rw.make_pool(42, S, base + off[str(S)])
        assert pool["y_true"].shape[0] == S
        assert pool["seq_obs"].shape == (S, ru.SEQ_FLAT_DIM)


def test_declared_stream_reproduces_collected_pool(data):
    # the collector's declared RNG stream must regenerate the exact stored pool (seed 42)
    meta, z = data["meta"], data["z"]
    base, off = meta["pool_stream_base"], meta["pool_stream_offsets"]
    for S in SIZES:
        p = rw.make_pool(42, S, base + off[str(S)])
        rows = np.nonzero(z[f"pool{S}__seed"] == 42)[0]
        assert np.array_equal(p["y_true"], z[f"pool{S}__y_true"][rows])
        assert np.array_equal(p["operand_a"], z[f"pool{S}__operand_a"][rows])
        assert np.allclose(p["seq_obs"], z[f"pool{S}__seq_obs"][rows])


def test_all_scales_present_in_npz(data):
    meta, z = data["meta"], data["z"]
    for S in SIZES:
        assert meta["n"][f"pool{S}"] == 5 * S
        assert z[f"pool{S}__y_true"].shape[0] == 5 * S
        assert (z[f"pool{S}__seed"] == 42).sum() == S


def test_train_pool_sequence_disjointness(data):
    for _, d in data["meta"]["disjointness"].items():
        for i in range(len(SIZES)):
            for j in range(i + 1, len(SIZES)):
                assert d[f"pool{SIZES[i]}_vs_{SIZES[j]}"] == 0
        assert d["pools_vs_val"] == 0
        assert d["pools_vs_test"] == 0


def test_no_duplicate_training_rows(data):
    z, S = data["z"], TOP
    rows = np.nonzero(z[f"pool{S}__seed"] == 42)[0]
    fp = {(int(z[f"pool{S}__seq"][r]), int(z[f"pool{S}__t"][r])) for r in rows}
    assert len(fp) == len(rows)


def test_val_invariance(data):
    assert data["meta"]["n"]["val"] == 5 * 800
    z = data["z"]
    for seed in aw.SEEDS:
        assert (z["val__seed"] == seed).sum() == 800


def test_test_invariance(data):
    # the canonical TEST is stored once and identical across scales (only one copy exists)
    z = data["z"]
    for m in MODELS:
        assert z[f"{m}__test__y_true"].shape[0] == 15763
    tots = [int((z["E_candidate__test__seed"] == s).sum()) for s in aw.SEEDS]
    assert tots == [3199, 3196, 3171, 3125, 3072]


def test_heldout_operand_exclusion(data):
    z, S = data["z"], TOP
    rows = np.nonzero(z[f"pool{S}__seed"] == 42)[0]
    oa = z[f"pool{S}__operand_a"][rows]
    ob = z[f"pool{S}__operand_b"][rows]
    held = np.array([(int(a), int(b)) in au.HELD_OUT_PAIRS for a, b in zip(oa, ob)])
    assert held.sum() > 0
    assert held.sum() < len(rows)
    assert len(au.HELD_OUT_PAIRS) == 20


def test_target_generation(data):
    z = data["z"]
    for S in SIZES:
        assert np.array_equal(z[f"pool{S}__y_true"],
                              (z[f"pool{S}__operand_a"] + z[f"pool{S}__operand_b"]) % 10)
        assert np.array_equal(z[f"pool{S}__raw_sum"],
                              z[f"pool{S}__operand_a"] + z[f"pool{S}__operand_b"])


def test_classA_count_reproduction(data):
    z = data["z"]
    per = [int(((z["E_candidate__test__seed"] == s)
                & (z["E_candidate__test__cand_ok"] == 0)).sum()) for s in aw.SEEDS]
    assert per == [448, 330, 363, 327, 337]


# ---------------------------------------------------------------------------
# causal history / leakage guards
# ---------------------------------------------------------------------------

def test_causal_history_correctness(data):
    z, S = data["z"], TOP
    rows = np.nonzero(z[f"pool{S}__seed"] == 42)[0][:500]
    seq = z[f"pool{S}__seq_obs"][rows]
    t = z[f"pool{S}__t"][rows]
    X, M = au.build_history(seq, t, 4)
    full = seq.reshape(len(rows), ru.T_MAX, ru.OBS_DIM)
    for i in range(len(rows)):
        assert int(M[i].sum()) == min(4, int(t[i]) + 1)
        assert np.allclose(X[i][-ru.OBS_DIM:], full[i, int(t[i])])


def test_no_future_leakage(data):
    z, S = data["z"], TOP
    rows = np.nonzero(z[f"pool{S}__seed"] == 43)[0][:400]
    seq = z[f"pool{S}__seq_obs"][rows]
    t = z[f"pool{S}__t"][rows]
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
    # the reused GRU reader is unidirectional; a bidirectional GRU would depend on the future
    assert av.GRUReader(av.architecture()["GRU"]["hidden"]).gru.bidirectional is False
    torch.manual_seed(0)
    X = torch.randn(1, 6, av.INPUT_DIM)
    bidir = torch.nn.GRU(av.INPUT_DIM, 8, batch_first=True, bidirectional=True)
    with torch.no_grad():
        o1 = bidir(X)[0]
        X2 = X.clone()
        X2[:, 4, :] = torch.randn(av.INPUT_DIM)
        o2 = bidir(X2)[0]
    assert not torch.allclose(o1[:, 2], o2[:, 2])       # position 2 depends on future position 4


# ---------------------------------------------------------------------------
# completeness of the produced analysis
# ---------------------------------------------------------------------------

def test_all_readers_present(built):
    assert set(built["readers"]) == set(READERS)


def test_ladder_matches_collection(built):
    assert built["metadata"]["train_sizes"] == SIZES


def test_all_scales_present(built):
    for R in READERS:
        for S in SIZES:
            assert str(S) in built["readers"][R]["y_true"]["L4"]
            assert "main" in built["readers"][R]["y_true"]["L4"][str(S)]


def test_all_history_lengths_present(built):
    for R in READERS:
        for rep in av.REP_LABELS:
            assert rep in built["readers"][R]["y_true"]
            assert "main" in built["readers"][R]["y_true"][rep][str(SIZES[0])]


def test_heldout_outputs_present(built):
    for R in READERS:
        for rep in ("L4", "Lfull"):
            node = built["readers"][R]["y_true"][rep][str(TOP)]["heldout"]
            assert node["seen"]["E_candidate:classA"]["mean"] is not None
            assert node["heldout"]["E_candidate:classA"]["mean"] is not None


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    assert d["best_top_all"] is not None and d["max_gain"] is not None
    assert d["thresholds"] == {"recover_bar": 0.50, "heldout_weak": 0.25,
                               "heldout_material": 0.30, "material_gain": 0.15,
                               "bayes_gap_bar": 0.30}
    for R in READERS:
        assert R in d["per_reader"]
        assert "best_top" in d["per_reader"][R] and "gain_full" in d["per_reader"][R]


# ---------------------------------------------------------------------------
# frozen-model protection / determinism (resource-light fits)
# ---------------------------------------------------------------------------

def test_no_reason_parameter_mutation_and_no_grad(loaded_model, fast_fit):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    aw._G["arch"] = av.architecture()
    X, y = _tiny("GRU")
    aw._fit("GRU", X, y, X, y, 10, 5)
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


def test_reader_model_separation(loaded_model, fast_fit):
    aw._G["arch"] = av.architecture()
    X, y = _tiny("TCN")
    aw._fit("TCN", X, y, X, y, 10, 5)
    ids = {id(p) for p in loaded_model.parameters()}
    reader = av.build_reader("TCN", 10, aw._G["arch"])
    assert not (ids & {id(p) for p in reader.parameters()})


def test_best_checkpoint_from_val(fast_fit):
    aw._G["arch"] = av.architecture()
    Xtr, ytr = _tiny("GRU", 128, 4, 3)
    Xva, yva = _tiny("GRU", 64, 4, 4)
    pred, vacc, tracc = aw._fit("GRU", Xtr, ytr, Xva, yva, 10, 7)
    assert abs(float((pred(Xva) == yva).mean()) - vacc) < 1e-6


def test_deterministic_artifacts(fast_fit):
    p1 = rw.make_pool(42, SIZES[0], 424242)
    p2 = rw.make_pool(42, SIZES[0], 424242)
    assert np.array_equal(p1["seq_obs"], p2["seq_obs"])
    assert np.array_equal(p1["t"], p2["t"])
    aw._G["arch"] = av.architecture()
    X, y = _tiny("GRU")
    _, v1, _ = aw._fit("GRU", X, y, X, y, 10, 11)
    _, v2, _ = aw._fit("GRU", X, y, X, y, 10, 11)
    assert v1 == v2


def test_u_baseline_loading(u_base):
    for m in MODELS:
        assert "L4" in u_base[m] and "Lfull" in u_base[m]
    assert abs(u_base["E_candidate"]["L4"] - 0.999) < 0.01


def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        aw.load(str(tmp_path / "nope.npz"))


# ---------------------------------------------------------------------------
# figures / immutability / governance
# ---------------------------------------------------------------------------

def test_figure_creation(data, built, tmp_path):
    figs = aw.make_figures(data, built, str(tmp_path), prefix="test_w")
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
    assert "experiment_w_data_scale.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_w_data_scale.md" in baseline["reports"]
