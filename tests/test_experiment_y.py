"""Tests for Experiment Y — bounded-reader fitting control at fixed capacity and data.

Resource-light suite: the live reader fits use tiny synthetic tensors and a short update budget; the
structural tests read the collected artifacts. Covers: V/X reader reuse and architecture identity,
exact 256-row fixed subset construction, deterministic selection, class balance and target
correctness, held-out operand-pair exclusion, no duplicate (sequence_id, timestep), sequence
disjointness from canonical VAL/TEST, exact L4/full causal history construction,
future-observation leakage rejection, fixed model/optimizer configuration, exactly 1,000 optimizer
updates, milestone recording, maximum-training-accuracy calculation, final-checkpoint integrity,
validation-only checkpoint selection, no TEST-based selection, frozen RE:Reason checkpoint SHA
verification, no gradients / no parameter mutation, reader/model separation, deterministic outputs,
malformed-artifact rejection, figure generation, historical-report immutability, and
documentation-governance compliance.
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
import analyze_experiment_y as ay  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_experiment_y as ry  # noqa: E402

torch.set_num_threads(int(os.environ.get("Y_TEST_THREADS", "1")))

NPZ = os.path.join(ROOT, "experiments", "experiment_y_fitting_control.npz")
META = os.path.join(ROOT, "experiments", "experiment_y_fitting_control_meta.json")
RESULTS = os.path.join(ROOT, "experiments", "experiment_y_results.json")
W_NPZ = os.path.join(ROOT, "experiments", "experiment_w_data_scale.npz")
MODELS = ("E_candidate", "F_combined")
READERS = ["MLP128", "GRU", "TCN"]
SEQUENCE_READERS = ("GRU", "TCN")
REPS = ("L4", "Lfull")
STATES = ("final", "best_val", "best_train")
SUBSET_SIZE = 256
MILESTONES = ("200", "400", "600", "800", "1000")
ALLOWED_VERDICTS = {"Y-A", "Y-B", "Y-MIX", "Y-inconclusive"}


def _y_meta():
    if os.path.exists(NPZ):
        return json.loads(str(np.load(NPZ, allow_pickle=False)["meta_json"][0]))
    return None


_YMETA = _y_meta()
TOP = int(_YMETA["source"]["top_size"]) if _YMETA else 10000
ay.TOP = TOP
if _YMETA:
    aw.TRAIN_SIZES = list(_YMETA["source"]["train_sizes"])


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(NPZ):
        pytest.skip("experiment_y npz not present")
    return ay.load(NPZ)


@pytest.fixture(scope="module")
def built(data):
    if not os.path.exists(RESULTS):
        pytest.skip("experiment_y results.json not present")
    with open(RESULTS) as fh:
        return json.load(fh)


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


def _indim(reader, L):
    return av.INPUT_DIM * L if reader == "MLP128" else av.INPUT_DIM


# ---------------------------------------------------------------------------
# architecture reuse / parameter counts
# ---------------------------------------------------------------------------

def test_reader_architecture_reuse():
    arch = av.architecture()
    assert arch["GRU"]["hidden"] == 85 and arch["TCN"]["channels"] == 52
    assert arch["TCN"]["kernel"] == 3 and arch["TCN"]["dilations"] == [1, 2, 4, 8]
    assert arch["TCN"]["n_layers"] == 4
    for reader in ("GRU", "TCN"):
        m = ay.ax.build_model(reader, 10, av.INPUT_DIM, arch)
        assert type(m) is type(av.build_reader(reader, 10, arch))


def test_parameter_counts():
    arch = av.architecture()
    assert as_.param_count("mlp", 128, av.INPUT_DIM * 4) == 12682      # L4 flat reader
    assert as_.param_count("mlp", 128, av.INPUT_DIM * 10) == 29578    # full prefix
    assert ay._param_count("MLP128", av.INPUT_DIM * 10, arch) == 29578
    assert ay._param_count("GRU", av.INPUT_DIM, arch) == av.reader_param_count("GRU", 10, arch)
    assert ay._param_count("TCN", av.INPUT_DIM, arch) == av.reader_param_count("TCN", 10, arch)
    assert av.reader_param_count("GRU", 10, arch) <= arch["budget_u_mlp128"]
    assert av.reader_param_count("TCN", 10, arch) <= arch["budget_u_mlp128"]


def test_parameter_count_invariance(built):
    # the sequence readers' width is L-independent; the flat MLP-128 input scales with the window
    for R in ("GRU", "TCN"):
        vals = {built["readers"][R][rep]["params"] for rep in REPS}
        assert len(vals) == 1, (R, vals)
    assert built["readers"]["MLP128"]["L4"]["params"] == 12682
    assert built["readers"]["MLP128"]["Lfull"]["params"] == 29578


# ---------------------------------------------------------------------------
# fixed small-subset construction
# ---------------------------------------------------------------------------

def test_subset_size_exact(data):
    z = data["z"]
    for s in aw.SEEDS:
        assert int((z[f"sub{SUBSET_SIZE}__seed"] == s).sum()) == SUBSET_SIZE
    assert z[f"sub{SUBSET_SIZE}__y_true"].shape[0] == SUBSET_SIZE * len(aw.SEEDS)
    assert data["meta"]["n"]["subset_per_seed"] == SUBSET_SIZE


def test_subset_class_balance(data):
    z = data["z"]
    for s in aw.SEEDS:
        rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == s)[0]
        counts = [int((z[f"sub{SUBSET_SIZE}__y_true"][rows] == c).sum()) for c in range(10)]
        assert counts == [26] * 6 + [25] * 4, (s, counts)
        assert data["meta"]["per_seed"][str(s)]["class_counts_actual"] == counts
        assert sum(counts) == SUBSET_SIZE


def test_subset_seed_recorded(data):
    assert data["meta"]["subset_seed"] == 20261010
    for s in aw.SEEDS:
        assert data["meta"]["per_seed"][str(s)]["selection_seed"] == 20261010


def test_subset_selection_deterministic(data):
    wz = np.load(W_NPZ, allow_pickle=False)
    z = data["z"]
    for s in aw.SEEDS:
        rows = np.nonzero(wz[f"pool{TOP}__seed"] == s)[0]
        chosen, counts = ry.select_subset_rows(
            wz[f"pool{TOP}__y_true"][rows], wz[f"pool{TOP}__operand_a"][rows],
            wz[f"pool{TOP}__operand_b"][rows], s, size=SUBSET_SIZE,
            subset_seed=data["meta"]["subset_seed"])
        seq = wz[f"pool{TOP}__seq"][rows][chosen]
        t = wz[f"pool{TOP}__t"][rows][chosen]
        prov = sorted((int(a), int(b)) for a, b in zip(seq, t))
        digest = hashlib.sha256(json.dumps(prov).encode()).hexdigest()
        assert digest == data["meta"]["per_seed"][str(s)]["provenance_sha256"]
        assert counts == data["meta"]["per_seed"][str(s)]["class_counts_actual"]
        # the stored subset arrays equal the recomputation
        sub_rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == s)[0]
        assert np.array_equal(np.sort(z[f"sub{SUBSET_SIZE}__seq"][sub_rows]), np.sort(seq))
        assert np.array_equal(np.sort(z[f"sub{SUBSET_SIZE}__t"][sub_rows]), np.sort(t))


def test_subset_target_correctness(data):
    z = data["z"]
    a = z[f"sub{SUBSET_SIZE}__operand_a"]
    b = z[f"sub{SUBSET_SIZE}__operand_b"]
    assert np.array_equal(z[f"sub{SUBSET_SIZE}__y_true"], (a + b) % 10)
    assert np.array_equal(z[f"sub{SUBSET_SIZE}__raw_sum"], a + b)


def test_subset_heldout_excluded(data):
    z = data["z"]
    a = z[f"sub{SUBSET_SIZE}__operand_a"]
    b = z[f"sub{SUBSET_SIZE}__operand_b"]
    held = sum(1 for x, y in zip(a, b) if (int(x), int(y)) in au.HELD_OUT_PAIRS)
    assert held == 0
    assert len(au.HELD_OUT_PAIRS) == 20 and all(y in (0, 5) for _, y in au.HELD_OUT_PAIRS)


def test_subset_no_duplicate_provenance(data):
    z = data["z"]
    for s in aw.SEEDS:
        rows = np.nonzero(z[f"sub{SUBSET_SIZE}__seed"] == s)[0]
        prov = {(int(x), int(t)) for x, t in
                zip(z[f"sub{SUBSET_SIZE}__seq"][rows], z[f"sub{SUBSET_SIZE}__t"][rows])}
        assert len(prov) == SUBSET_SIZE
        assert data["meta"]["per_seed"][str(s)]["n_duplicate_rows"] == 0
        assert data["meta"]["per_seed"][str(s)]["n_unique_seq_t"] == SUBSET_SIZE


def test_subset_sequence_disjoint(data):
    d = data["meta"]["disjointness"]
    assert all(v == 0 for v in d.values()), d
    assert "subset_vs_val" in d and "subset_vs_E_candidate_test" in d


# ---------------------------------------------------------------------------
# fixed VAL / TEST identity against W
# ---------------------------------------------------------------------------

def test_fixed_val_identity(data):
    wz = np.load(W_NPZ, allow_pickle=False)
    z = data["z"]
    for c in ry.POOL_COLS:
        assert np.array_equal(z[f"val__{c}"], wz[f"val__{c}"])
    assert data["meta"]["n"]["val"] == 5 * 800


def test_fixed_test_identity(data):
    wz = np.load(W_NPZ, allow_pickle=False)
    z = data["z"]
    for m in MODELS:
        for c in ry.TEST_COLS:
            assert np.array_equal(z[f"{m}__test__{c}"], wz[f"{m}__test__{c}"])
        assert z[f"{m}__test__y_true"].shape[0] == 15763


def test_source_hash_recorded(data):
    expected = hashlib.sha256(open(W_NPZ, "rb").read()).hexdigest()
    assert expected == data["meta"]["source"]["sha256"]
    for k, rec in data["meta"]["identity"].items():
        assert hashlib.sha256(np.ascontiguousarray(data["z"][k]).tobytes()).hexdigest() == rec["sha256"]


# ---------------------------------------------------------------------------
# causal history / leakage guards
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
    X, M = au.build_history(z[f"sub{SUBSET_SIZE}__seq_obs"][rows], z[f"sub{SUBSET_SIZE}__t"][rows], 4)
    assert np.array_equal(Xi[:, :, -1], M.astype(np.float32))


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
# trainer protocol / update budget / milestone / selection
# ---------------------------------------------------------------------------

def test_exact_update_count_and_events():
    X, y = _tiny("GRU", 64, 4, 5)
    fit = ay.train_trajectory("GRU", X, y, X, y, 10, 3, 40, av.architecture())
    assert fit["events"][-1]["update"] == 40
    assert len(fit["events"]) == 4                 # eval every 10 -> 10,20,30,40


def test_results_updates_1000(built):
    assert built["metadata"]["updates"] == 1000
    assert built["metadata"]["eval_every"] == 10
    for R in READERS:
        for rep in REPS:
            node = built["readers"][R][rep]
            assert node["updates"] == 1000 and node["n_train"] == SUBSET_SIZE


def test_milestones_recorded(built):
    for R in READERS:
        for rep in REPS:
            ms = built["readers"][R][rep]["fit"]["milestones"]
            assert set(ms.keys()) == set(MILESTONES)
            for u in MILESTONES:
                for f in ("train_acc", "train_loss", "val_acc", "val_loss"):
                    assert ms[u][f]["mean"] is not None


def test_max_train_acc_calculation():
    X, y = _tiny("MLP128", 64, 4, 2)
    fit = ay.train_trajectory("MLP128", X, y, X, y, 10, 11, 60, av.architecture())
    idx = int(np.argmax([e["train_acc"] for e in fit["events"]]))
    assert abs(fit["best_train"][0] - fit["events"][idx]["train_acc"]) < 1e-12
    assert fit["best_train"][2] == fit["events"][idx]["update"]


def test_final_state_integrity():
    X, y = _tiny("TCN", 64, 4, 3)
    arch = av.architecture()
    fit = ay.train_trajectory("TCN", X, y, X, y, 10, 17, 50, arch)
    pred = ay.predict_state("TCN", fit["final"], X, fit["norm"], 10, av.INPUT_DIM, arch)
    assert abs(float((pred == y).mean()) - fit["events"][-1]["train_acc"]) < 1e-9


def test_validation_only_checkpoint_selection():
    Xtr, ytr = _tiny("GRU", 96, 4, 2)
    Xva, yva = _tiny("GRU", 48, 4, 3)
    fit = ay.train_trajectory("GRU", Xtr, ytr, Xva, yva, 10, 5, 200, av.architecture())
    assert abs(fit["best_val"][0] - max(e["val_acc"] for e in fit["events"])) < 1e-12


def test_no_test_based_selection(built):
    disc = built["metadata"]["statistical_discipline"]
    assert "VAL-only" in disc and "never used" in disc
    # best-on-VAL is the validation maximum for every seed/reader/rep (no TEST enters selection)
    for s in built["per_seed"]:
        for rep in REPS:
            traj = s["traj"][rep]
            vmax = max(e["val_acc"] for e in traj)
            assert abs(s["fits"][f"{s['reader']}|{rep}"]["fit"]["best_val_acc"] - vmax) < 1e-12


@pytest.mark.parametrize("reader", [["MLP128", "GRU", "TCN"]])
def test_trainer_matches_x_protocol(reader):
    for R in reader:
        X, y = _tiny(R, 64, 4, 1)
        arch = av.architecture()
        a = ay.train_trajectory(R, X, y, X, y, 10, 7, 200, arch)
        b = ax.train_trajectory(R, X, y, X, y, 10, 7, 200, arch)
        assert a["best_val"][0] == b["best_all"][0]
        indim = _indim(R, 4)
        pa = ay.predict_state(R, a["best_val"][1], X, a["norm"], 10, indim, arch)
        pb = ax.predict_state(R, b["best_all"][1], X, b["norm"], 10, indim, arch)
        assert np.array_equal(pa, pb)


def test_deterministic_trajectory():
    X, y = _tiny("GRU", 64, 4, 0)
    a = ay.train_trajectory("GRU", X, y, X, y, 10, 11, 60, av.architecture())
    b = ay.train_trajectory("GRU", X, y, X, y, 10, 11, 60, av.architecture())
    assert [e["val_acc"] for e in a["events"]] == [e["val_acc"] for e in b["events"]]
    assert a["best_train"][0] == b["best_train"][0]


@pytest.mark.parametrize("reader", ["MLP128", "GRU", "TCN"])
def test_fit_seed_is_x_convention(reader):
    for seed in aw.SEEDS:
        for rep in REPS:
            assert ay._fit_seed(seed, reader, rep) == aw.reader_seed(seed, reader, "y_true", rep, TOP, "main")


# ---------------------------------------------------------------------------
# completeness / decision / controls
# ---------------------------------------------------------------------------

def test_all_readers_and_reps_present(built):
    assert set(built["readers"]) == set(READERS)
    for R in READERS:
        assert set(built["readers"][R]) == set(REPS)


def test_fitting_and_generalization_reported(built):
    for R in READERS:
        for rep in REPS:
            node = built["readers"][R][rep]
            assert node["fit"]["train_acc_max"]["mean"] is not None
            assert node["fit"]["train_acc_final"]["mean"] is not None
            assert node["fit"]["train_loss_final"]["mean"] is not None
            for st in STATES:
                assert node[st]["train_acc"]["mean"] is not None
                assert node[st]["val_acc"]["mean"] is not None


def test_heldout_outputs_present(built):
    for R in READERS:
        for rep in REPS:
            for st in STATES:
                node = built["readers"][R][rep][st]
                for m in MODELS:
                    assert node["heldout"][f"{m}:classA"]["mean"] is not None
                    assert node["seen"][f"{m}:classA"]["mean"] is not None


def test_ef_control(built):
    # the readers are model-independent: the all-ADD population is identical for E and F
    for R in READERS:
        for rep in REPS:
            for st in STATES:
                node = built["readers"][R][rep][st]["acc"]
                for p in ("all_add",):
                    assert abs(node[f"E_candidate:{p}"]["mean"]
                               - node[f"F_combined:{p}"]["mean"]) < 1e-9


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    assert d["thresholds"]["fit_bar"] == 0.90 and d["thresholds"]["heldout_weak"] == 0.25
    assert set(d["per_pair"].keys()) == {f"{R}|{rep}" for R in SEQUENCE_READERS for rep in REPS}
    for k, v in d["per_pair"].items():
        assert set(v["heldout_classA"].keys()) == set(STATES)
        assert v["fits"] == (v["train_acc_max"] >= 0.90)
    if d["verdict"] == "Y-A":
        assert any(v["fits"] and v["weak_heldout"] for v in d["per_pair"].values())
    if d["verdict"] == "Y-B":
        assert not any(v["fits"] for v in d["per_pair"].values())


def test_baselines_reused_unchanged(built):
    assert set(built["baselines"]) == {"w_200", "x_1000"}
    for src in ("w_200", "x_1000"):
        for R in READERS:
            for rep in REPS:
                node = built["baselines"][src][R][rep]
                assert node["classA"] is not None and node["train_acc"] is not None
    assert built["baselines"]["w_200"]["GRU"]["L4"]["classA"] == pytest.approx(0.116, abs=0.01)
    assert built["baselines"]["x_1000"]["TCN"]["Lfull"]["classA"] == pytest.approx(0.111, abs=0.01)


def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        ay.load(str(tmp_path / "nope.npz"))


# ---------------------------------------------------------------------------
# frozen-model protection
# ---------------------------------------------------------------------------

def test_no_reason_parameter_mutation_and_no_grad(loaded_model):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    X, y = _tiny("TCN", 64, 4, 0)
    ay.train_trajectory("TCN", X, y, X, y, 10, 5, 20, av.architecture())
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


def test_reader_model_separation(loaded_model):
    X, y = _tiny("GRU", 64, 4, 0)
    model = ay.ax.build_model("GRU", 10, av.INPUT_DIM, av.architecture())
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
    figs = ay.make_figures(data, built, str(tmp_path), prefix="test_y")
    assert len(figs) >= 8 and all(os.path.exists(p) for p in figs)
    names = {os.path.basename(p) for p in figs}
    for req in ("training_fit", "training_trajectory", "train_vs_test", "heldout_generalization",
                "reader_comparison", "l4_vs_full", "per_seed", "decision_summary"):
        assert f"test_y_{req}.png" in names


def test_historical_immutability():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"


def test_research_governance_compliance():
    log = open(os.path.join(ROOT, "docs", "RESEARCH_LOG.md")).read()
    assert "experiment_y_fitting_control.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_y_fitting_control.md" in baseline["reports"]
