"""Tests for Experiment Z — TCN instrument qualification on the directly observable `op_dest` target.

Resource-light suite: the live reader fits use tiny synthetic tensors and a short update budget; the
structural tests read the collected artifacts. Covers: reuse of Y's exact fixed subset/provenance and
byte-identical VAL/TEST, the four-way `op_dest` class coverage in every population, direct
recoverability of `op_dest` from the current observation, split/leakage invariants, reader architecture
and four-class output-head parameter counts, exactly 1,000 optimizer updates, milestone recording,
maximum-training-accuracy calculation, final-checkpoint integrity, validation-only checkpoint
selection, no TEST-based selection, reader-seed convention, frozen RE:Reason checkpoint SHA
verification, no gradients / no parameter mutation, reader/model separation, E/F control,
deterministic trajectories, malformed-artifact rejection, figure generation, historical immutability,
and documentation-governance compliance.
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
import analyze_experiment_y as ay  # noqa: E402
import analyze_experiment_z as az  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_experiment_y as ry  # noqa: E402
import run_experiment_z as rz  # noqa: E402

torch.set_num_threads(int(os.environ.get("Z_TEST_THREADS", "1")))

NPZ = os.path.join(ROOT, "experiments", "experiment_z_op_dest.npz")
META = os.path.join(ROOT, "experiments", "experiment_z_op_dest_meta.json")
RESULTS = os.path.join(ROOT, "experiments", "experiment_z_results.json")
Y_NPZ = os.path.join(ROOT, "experiments", "experiment_y_fitting_control.npz")
W_NPZ = os.path.join(ROOT, "experiments", "experiment_w_data_scale.npz")
MODELS = ("E_candidate", "F_combined")
READERS = ["GRU", "TCN"]
REPS = ("L4", "Lfull")
STATES = ("final", "best_val", "best_train")
SUBSET_SIZE = 256
MILESTONES = ("200", "400", "600", "800", "1000")
ALLOWED_VERDICTS = {"Z-A", "Z-B", "Z-MIX", "Z-INVALID"}
POOLED_TRAIN_OPDEST = [306, 315, 300, 359]


def _z_meta():
    if os.path.exists(NPZ):
        return json.loads(str(np.load(NPZ, allow_pickle=False)["meta_json"][0]))
    return None


def _y_meta():
    with open(os.path.join(ROOT, "experiments", "experiment_y_fitting_control_meta.json")) as fh:
        return json.load(fh)


_ZMETA = _z_meta()
_YMETA = _y_meta() if os.path.exists(os.path.join(ROOT, "experiments",
                                                  "experiment_y_fitting_control_meta.json")) else {}
TOP = int(_ZMETA["source"]["top_size"]) if _ZMETA else 10000
az.TOP = TOP
if _ZMETA:
    aw.TRAIN_SIZES = list(_ZMETA["source"]["train_sizes"])


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_z npz not present")
    return az.load(NPZ)


@pytest.fixture(scope="module")
def built(data):
    if not os.path.exists(RESULTS):
        pytest.skip("experiment_z results.json not present")
    with open(RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


def _tiny(reader, n=64, L=4, seed=0, nclass=4):
    rng = np.random.default_rng(seed)
    shape = (n, av.INPUT_DIM * L) if reader == "MLP128" else (n, L, av.INPUT_DIM)
    X = rng.normal(size=shape).astype(np.float32)
    y = rng.integers(0, nclass, n)
    return X, y


# ---------------------------------------------------------------------------
# architecture reuse / four-class head
# ---------------------------------------------------------------------------

def test_reader_architecture_reuse():
    arch = av.architecture()
    assert arch["GRU"]["hidden"] == 85 and arch["TCN"]["channels"] == 52
    assert arch["TCN"]["kernel"] == 3 and arch["TCN"]["dilations"] == [1, 2, 4, 8]
    for reader in READERS:
        m = av.build_reader(reader, az.NCLASS, arch)
        assert type(m) is type(av.build_reader(reader, 10, arch))


def test_four_class_head_parameter_counts():
    arch = av.architecture()
    assert az.NCLASS == 4
    for reader, width in (("GRU", 85), ("TCN", 52)):
        p4 = av.reader_param_count(reader, 4, arch)
        p10 = av.reader_param_count(reader, 10, arch)
        assert p4 < p10
        assert p10 - p4 == (width + 1) * (10 - 4)      # only the head changes
    assert av.reader_param_count("GRU", 4, arch) == 28139
    assert av.reader_param_count("TCN", 4, arch) == 28188


def test_parameter_count_in_results(built):
    arch = av.architecture()
    for R in READERS:
        for rep in REPS:
            assert built["readers"][R][rep]["params"] == av.reader_param_count(R, 4, arch)


# ---------------------------------------------------------------------------
# fixed subset + provenance + op_dest target
# ---------------------------------------------------------------------------

def test_subset_size_exact(data):
    z = data["z"]
    for s in aw.SEEDS:
        assert int((z[f"sub{SUBSET_SIZE}__seed"] == s).sum()) == SUBSET_SIZE
    assert data["meta"]["n"]["subset_per_seed"] == SUBSET_SIZE


def test_subset_identity_vs_y(data):
    yz = np.load(Y_NPZ, allow_pickle=False)
    z = data["z"]
    for c in ry.POOL_COLS + ["pool_row"]:
        assert np.array_equal(z[f"sub{SUBSET_SIZE}__{c}"], yz[f"sub{SUBSET_SIZE}__{c}"]), c


def test_val_test_identity_vs_y(data):
    yz = np.load(Y_NPZ, allow_pickle=False)
    z = data["z"]
    for c in ry.POOL_COLS:
        assert np.array_equal(z[f"val__{c}"], yz[f"val__{c}"])
    for m in MODELS:
        for c in ry.TEST_COLS:
            assert np.array_equal(z[f"{m}__test__{c}"], yz[f"{m}__test__{c}"])


def test_source_hashes_recorded(data):
    y_sha = hashlib.sha256(open(Y_NPZ, "rb").read()).hexdigest()
    w_sha = hashlib.sha256(open(W_NPZ, "rb").read()).hexdigest()
    assert data["meta"]["source"]["y_sha256"] == y_sha
    assert data["meta"]["source"]["w_sha256"] == w_sha
    assert data["meta"]["source"]["y_subset_seed"] == _y_meta()["subset_seed"]


def test_subset_provenance_matches_y(data):
    z = data["z"]
    ymeta = _y_meta()
    for s in aw.SEEDS:
        rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == s)[0]
        prov = sorted((int(a), int(b)) for a, b in
                      zip(z[f"sub{SUBSET_SIZE}__seq"][rows], z[f"sub{SUBSET_SIZE}__t"][rows]))
        digest = hashlib.sha256(json.dumps(prov).encode()).hexdigest()
        assert digest == ymeta["per_seed"][str(s)]["provenance_sha256"]


def test_op_dest_class_coverage(data):
    meta = data["meta"]
    cc = meta["class_counts"]
    for where in ("train", "val"):
        for k, v in cc[where].items():
            assert len(v) == 4 and all(int(c) > 0 for c in v), (where, k, v)
    for m in MODELS:
        v = cc["test"][m]
        assert len(v) == 4 and all(int(c) > 0 for c in v), (m, v)
    assert cc["train"]["pooled"] == POOLED_TRAIN_OPDEST
    assert sum(cc["train"]["pooled"]) == SUBSET_SIZE * len(aw.SEEDS)


def test_op_dest_direct_recoverability(data):
    z = data["z"]
    # op_dest == argmax(current observation reg_a block x_t[3:7]) in every population
    assert rz.verify_direct_recoverability(
        z[f"sub{SUBSET_SIZE}__seq_obs"], z[f"sub{SUBSET_SIZE}__t"],
        z[f"sub{SUBSET_SIZE}__op_dest"], "train")["rate"] == 1.0
    assert rz.verify_direct_recoverability(
        z["val__seq_obs"], z["val__t"], z["val__op_dest"], "val")["rate"] == 1.0
    for m in MODELS:
        assert rz.verify_direct_recoverability(
            z[f"{m}__test__seq_obs"], z[f"{m}__test__t"], z[f"{m}__test__op_dest"],
            f"{m} test")["rate"] == 1.0
    for k, rec in data["meta"]["recoverability"].items():
        assert float(rec["rate"]) == 1.0


def test_split_disjointness(data):
    d = data["meta"]["disjointness"]
    assert all(int(v) == 0 for v in d.values()), d
    assert "subset_vs_val" in d


# ---------------------------------------------------------------------------
# causal history / leakage
# ---------------------------------------------------------------------------

def test_causal_history_construction(data):
    z = data["z"]
    rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == 42)[0][:200]
    seq = z[f"sub{SUBSET_SIZE}__seq_obs"][rows]
    t = z[f"sub{SUBSET_SIZE}__t"][rows]
    X, M = au.build_history(seq, t, 4)
    full = seq.reshape(len(rows), ru.T_MAX, ru.OBS_DIM)
    for i in range(len(rows)):
        assert int(M[i].sum()) == min(4, int(t[i]) + 1)
        assert np.allclose(X[i][-ru.OBS_DIM:], full[i, int(t[i])])


def test_no_future_leakage(data):
    z = data["z"]
    rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == 43)[0][:200]
    seq = z[f"sub{SUBSET_SIZE}__seq_obs"][rows]
    t = z[f"sub{SUBSET_SIZE}__t"][rows]
    rng = np.random.default_rng(0)
    X1, M1 = au.build_history(seq, t, 8)
    mod = seq.copy().reshape(len(rows), ru.T_MAX, ru.OBS_DIM)
    for i in range(len(rows)):
        ti = int(t[i])
        if ti + 1 < ru.T_MAX:
            mod[i, ti + 1:] = rng.random((ru.T_MAX - ti - 1, ru.OBS_DIM))
    X2, M2 = au.build_history(mod.reshape(len(rows), -1), t, 8)
    assert np.array_equal(X1, X2) and np.array_equal(M1, M2)


def test_reader_input_representation(data):
    z = data["z"]
    rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == 42)[0][:64]
    Xi = av.reader_input(z[f"sub{SUBSET_SIZE}__seq_obs"][rows], z[f"sub{SUBSET_SIZE}__t"][rows], 4)
    assert Xi.shape == (64, 4, av.INPUT_DIM) and av.INPUT_DIM == 22


# ---------------------------------------------------------------------------
# protocol / budget / milestones / selection
# ---------------------------------------------------------------------------

def test_trainer_reuses_y_protocol():
    # az trains through ay.train_trajectory (Y's exact trainer); the object is identical
    assert az.ay.train_trajectory is ay.train_trajectory


def test_exact_update_count_and_events():
    X, y = _tiny("TCN", 64, 4, 5, nclass=4)
    fit = az.ay.train_trajectory("TCN", X, y, X, y, 4, 3, 40, av.architecture())
    assert fit["events"][-1]["update"] == 40 and len(fit["events"]) == 4


def test_results_updates_1000(built):
    assert built["metadata"]["updates"] == 1000 and built["metadata"]["target"] == "op_dest"
    for R in READERS:
        for rep in REPS:
            assert built["readers"][R][rep]["updates"] == 1000
            assert built["readers"][R][rep]["n_train"] == SUBSET_SIZE


def test_milestones_recorded(built):
    for R in READERS:
        for rep in REPS:
            ms = built["readers"][R][rep]["fit"]["milestones"]
            assert set(ms.keys()) == set(MILESTONES)
            for u in MILESTONES:
                for f in ("train_acc", "train_loss", "val_acc", "val_loss"):
                    assert ms[u][f]["mean"] is not None


def test_max_train_acc_calculation():
    X, y = _tiny("GRU", 64, 4, 2, nclass=4)
    fit = az.ay.train_trajectory("GRU", X, y, X, y, 4, 11, 60, av.architecture())
    idx = int(np.argmax([e["train_acc"] for e in fit["events"]]))
    assert abs(fit["best_train"][0] - fit["events"][idx]["train_acc"]) < 1e-12
    assert fit["best_train"][2] == fit["events"][idx]["update"]


def test_final_state_integrity():
    X, y = _tiny("TCN", 64, 4, 3, nclass=4)
    arch = av.architecture()
    fit = az.ay.train_trajectory("TCN", X, y, X, y, 4, 17, 50, arch)
    pred = az.ay.predict_state("TCN", fit["final"], X, fit["norm"], 4, av.INPUT_DIM, arch)
    assert abs(float((pred == y).mean()) - fit["events"][-1]["train_acc"]) < 1e-9


def test_validation_only_checkpoint_selection(built):
    for s in built["per_seed"]:
        for rep in REPS:
            vmax = max(e["val_acc"] for e in s["traj"][rep])
            assert abs(s["fits"][f"{s['reader']}|{rep}"]["fit"]["best_val_acc"] - vmax) < 1e-12


def test_no_test_based_selection(built):
    disc = built["metadata"]["statistical_discipline"]
    assert "VAL-only" in disc and "never used" in disc
    assert built["primary_decision"]["rule_note"].count("fit_bar") >= 1


def test_fit_seed_convention():
    for seed in aw.SEEDS:
        for R in READERS:
            for rep in REPS:
                assert az._fit_seed(seed, R, rep) == aw.reader_seed(seed, R, "op_dest", rep, TOP, "main")
                assert az._fit_seed(seed, R, rep) != aw.reader_seed(seed, R, "y_true", rep, TOP, "main")


def test_deterministic_trajectory():
    X, y = _tiny("GRU", 64, 4, 0, nclass=4)
    a = az.ay.train_trajectory("GRU", X, y, X, y, 4, 11, 60, av.architecture())
    b = az.ay.train_trajectory("GRU", X, y, X, y, 4, 11, 60, av.architecture())
    assert [e["val_acc"] for e in a["events"]] == [e["val_acc"] for e in b["events"]]
    assert a["best_train"][0] == b["best_train"][0]


# ---------------------------------------------------------------------------
# completeness / controls / decision
# ---------------------------------------------------------------------------

def test_all_readers_reported(built):
    assert set(built["readers"]) == set(READERS)
    for R in READERS:
        assert set(built["readers"][R]) == set(REPS)


def test_fitting_and_states_reported(built):
    for R in READERS:
        for rep in REPS:
            node = built["readers"][R][rep]
            assert node["fit"]["train_acc_max"]["mean"] is not None
            assert node["fit"]["train_acc_final"]["mean"] is not None
            for st in STATES:
                assert node[st]["train_acc"]["mean"] is not None
                assert node[st]["acc"]["E_candidate:all_add"]["mean"] is not None


def test_ef_control(built):
    # readers are model-independent; op_dest test arrays are identical for E and F
    for R in READERS:
        for rep in REPS:
            for st in STATES:
                node = built["readers"][R][rep][st]["acc"]
                assert abs(node["E_candidate:all_add"]["mean"]
                           - node["F_combined:all_add"]["mean"]) < 1e-9


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    assert d["thresholds"]["fit_bar"] == 0.90
    assert set(d["per_pair"].keys()) == {f"{R}|{rep}" for R in READERS for rep in REPS}
    for k, v in d["per_pair"].items():
        assert v["fits"] == (v["train_acc_max"] >= 0.90)
    if d["verdict"] in ("Z-A", "Z-B", "Z-MIX"):
        assert d["control_ok"] is True


def test_decision_rule_matches_values(built):
    d = built["primary_decision"]
    tcn = d["tcn_fits"]
    if d["verdict"] == "Z-A":
        assert all(tcn.values()) and d["control_ok"]
    elif d["verdict"] == "Z-B":
        assert not any(tcn.values()) and d["control_ok"]
    elif d["verdict"] == "Z-MIX":
        assert any(tcn.values()) and not all(tcn.values()) and d["control_ok"]


def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        az.load(str(tmp_path / "nope.npz"))


# ---------------------------------------------------------------------------
# frozen-model protection
# ---------------------------------------------------------------------------

def test_no_reason_parameter_mutation_and_no_grad(loaded_model):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    X, y = _tiny("TCN", 64, 4, 0, nclass=4)
    az.ay.train_trajectory("TCN", X, y, X, y, 4, 5, 20, av.architecture())
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


def test_reader_model_separation(loaded_model):
    model = av.build_reader("GRU", 4, av.architecture())
    ids = {id(p) for p in loaded_model.parameters()}
    assert not (ids & {id(p) for p in model.parameters()})


def test_checkpoint_hashes_unchanged(data):
    for mk in MODELS:
        for s in aw.SEEDS:
            rec = data["meta"]["models"][mk]["checkpoints"][str(s)]
            assert eg.sha256_file(os.path.join(ROOT, rec["path"])) == rec["sha256"]


# ---------------------------------------------------------------------------
# figures / immutability / governance
# ---------------------------------------------------------------------------

def test_figure_generation(data, built, tmp_path):
    figs = az.make_figures(data, built, str(tmp_path), prefix="test_z")
    assert len(figs) >= 8 and all(os.path.exists(p) for p in figs)
    names = {os.path.basename(p) for p in figs}
    for req in ("training_fit", "training_trajectory", "reader_comparison", "l4_vs_full",
                "per_seed", "test_accuracy", "op_dest_control", "decision_summary"):
        assert f"test_z_{req}.png" in names


def test_historical_immutability():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"


def test_research_governance_compliance():
    log = open(os.path.join(ROOT, "docs", "RESEARCH_LOG.md")).read()
    assert "experiment_z_op_dest.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_z_op_dest.md" in baseline["reports"]
