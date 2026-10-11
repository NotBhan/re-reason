"""Tests for Experiment AB — TCN optimizer and initialization diagnostic on the `op_dest` control.

Resource-light suite: live fits use tiny synthetic tensors and a short budget; structural tests read
the collected artifacts. Covers: the mandatory trainable-parameter preflight (parameter groups,
finiteness, per-stage gradient/update evidence, dead-stage detection with a frozen negative control),
reuse of Z's `op_dest` dataset (byte identity), data/split/class/recoverability invariants, checkpoint
hashes, optimizer-family construction and recorded defaults, deterministic xavier initialization
(bounds and zero biases), the Adam/default trainer's protocol identity with Y, exact update counts,
determinism, the predeclared robust-fit decision rule (AB-A/AB-MIX/AB-B/AB-INVALID) encoded and tested,
condition completeness, the adam_ref↔AA reproduction bridge, GRU control, E/F control, VAL-only
selection, frozen-model protection, malformed-artifact rejection, figure generation, historical
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

import analyze_experiment_ab as ab  # noqa: E402
import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import analyze_experiment_w as aw  # noqa: E402
import analyze_experiment_x as ax  # noqa: E402
import analyze_experiment_y as ay  # noqa: E402
import analyze_experiment_z as az  # noqa: E402
import run_experiment_ab as ra  # noqa: E402
import run_experiment_g as eg  # noqa: E402

torch.set_num_threads(int(os.environ.get("AB_TEST_THREADS", "1")))

AB_NPZ = os.path.join(ROOT, "experiments", "experiment_ab_optimizer_init.npz")
AB_META = os.path.join(ROOT, "experiments", "experiment_ab_optimizer_init_meta.json")
AB_RESULTS = os.path.join(ROOT, "experiments", "experiment_ab_results.json")
Z_NPZ = os.path.join(ROOT, "experiments", "experiment_z_op_dest.npz")
Y_NPZ = os.path.join(ROOT, "experiments", "experiment_y_fitting_control.npz")
AA_RESULTS = os.path.join(ROOT, "experiments", "experiment_aa_results.json")
MODELS = ("E_candidate", "F_combined")
READERS = ["TCN", "GRU"]
REPS = ("L4", "Lfull")
SEEDS = [42, 43, 44, 45, 46]
STATES = ("final", "best_val", "best_train")
NCLASS = 4
ALLOWED_VERDICTS = {"AB-A", "AB-B", "AB-MIX", "AB-INVALID"}


def _ab_meta():
    if os.path.exists(AB_NPZ):
        return json.loads(str(np.load(AB_NPZ, allow_pickle=False)["meta_json"][0]))
    return None


_ABMETA = _ab_meta()
TOP = int(_ABMETA["source"]["top_size"]) if _ABMETA else 10000
ab.TOP = TOP
if _ABMETA:
    aw.TRAIN_SIZES = list(_ABMETA["source"]["train_sizes"])


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(AB_NPZ):
        pytest.skip("experiment_ab npz not present")
    return ab.load(AB_NPZ)


@pytest.fixture(scope="module")
def built(data):
    if not os.path.exists(AB_RESULTS):
        pytest.skip("experiment_ab results.json not present")
    with open(AB_RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


def _diag(seed=42, L=4, n=32):
    z = np.load(Z_NPZ, allow_pickle=False)
    rows = np.nonzero(z["sub256__seed"] == seed)[0][:n]
    X = av.reader_input(z["sub256__seq_obs"][rows], z["sub256__t"][rows], L)
    y = z["sub256__op_dest"][rows].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# mandatory trainable-parameter preflight
# ---------------------------------------------------------------------------

def test_preflight_passes(data):
    assert data["meta"]["preflight_pass"] is True
    for rep in REPS:
        assert data["meta"]["preflight"][rep]["passed"] is True


def test_preflight_param_groups(data):
    arch = av.architecture()
    for rep in REPS:
        p = data["meta"]["preflight"][rep]
        assert p["n_trainable_params"] == av.reader_param_count("TCN", NCLASS, arch) == 28188
        assert p["all_in_optimizer"] is True and p["all_requires_grad"] is True


def test_preflight_loss_and_grad_finite(data):
    for rep in REPS:
        p = data["meta"]["preflight"][rep]
        assert p["loss_finite"] is True and p["grad_finite"] is True
        assert p["aggregate_grad_norm"] > 0
        assert p["dead_stages"] == []


def test_preflight_stage_updates(data):
    for rep in REPS:
        p = data["meta"]["preflight"][rep]
        assert p["any_parameter_updated"] is True
        assert p["logit_change"] > 0
        for st, v in p["per_stage_grad"].items():
            assert v["grad_norm"] > 0, (rep, st)
        for st, v in p["per_stage_update"].items():
            assert v["effectively_unchanged"] is False, (rep, st)


def test_grad_stats_flags_dead_stage():
    arch = av.architecture()
    # frozen model -> all stages dead (validates the detection mechanism); the input carries
    # requires_grad so the backward pass is well-defined even though no parameter needs a gradient
    frozen = av.build_reader("TCN", NCLASS, arch)
    for prm in frozen.parameters():
        prm.requires_grad_(False)
    X = torch.randn(8, 4, av.INPUT_DIM, requires_grad=True)
    y = torch.zeros(8, dtype=torch.long)
    torch.nn.functional.cross_entropy(frozen(X), y).backward()
    st = ra._grad_stats(ra.iter_stages(frozen))
    assert all(v["dead"] for v in st.values())
    # trainable model -> not all dead
    live = av.build_reader("TCN", NCLASS, arch)
    torch.nn.functional.cross_entropy(live(torch.randn(8, 4, av.INPUT_DIM)), y).backward()
    st2 = ra._grad_stats(ra.iter_stages(live))
    assert any(not v["dead"] for v in st2.values())


def test_preflight_direct_call():
    X, y = _diag(L=4)
    p = ra.preflight_tcn(av.architecture(), X, y, X, y, name="t", steps=5)
    assert p["passed"] is True
    assert set(p["per_stage_grad"]) == {"conv0", "conv1", "conv2", "conv3", "head"}


# ---------------------------------------------------------------------------
# data identity / invariants
# ---------------------------------------------------------------------------

def test_reuse_z_dataset_byte_identity(data):
    zz = np.load(Z_NPZ, allow_pickle=False)
    for k in data["z"].files:
        if k == "meta_json":
            continue
        assert np.array_equal(data["z"][k], zz[k]), k


def test_source_hashes_recorded(data):
    m = data["meta"]["source"]
    assert m["z_sha256"] == hashlib.sha256(open(Z_NPZ, "rb").read()).hexdigest()
    assert m["y_sha256"] == hashlib.sha256(open(Y_NPZ, "rb").read()).hexdigest()
    assert data["meta"]["subset_seed"] == 20261010


def test_class_coverage(data):
    cc = data["meta"]["class_counts"]
    assert cc["train"]["pooled"] == [306, 315, 300, 359]
    for mk in MODELS:
        assert len(cc["test"][mk]) == 4 and all(c > 0 for c in cc["test"][mk])


def test_recoverability_and_disjointness(data):
    for k, rec in data["meta"]["recoverability"].items():
        assert rec["rate"] == 1.0, k
    assert all(v == 0 for v in data["meta"]["disjointness"].values())


def test_checkpoint_hashes_unchanged(data):
    for mk in MODELS:
        for s in SEEDS:
            rec = data["meta"]["models"][mk]["checkpoints"][str(s)]
            assert eg.sha256_file(os.path.join(ROOT, rec["path"])) == rec["sha256"]


# ---------------------------------------------------------------------------
# trainer / optimizer / init
# ---------------------------------------------------------------------------

def test_train_ab_matches_y_protocol_at_adam():
    X, y = _diag(L=4)
    arch = av.architecture()
    am = ab.COND_BY_NAME["adam_ref"]
    seed = aw.reader_seed(42, "TCN", "op_dest", "L4", TOP, "main")
    a = ab.train_ab("TCN", X, y, X, y, NCLASS, seed, 200, am, arch)
    b = ay.train_trajectory("TCN", X, y, X, y, NCLASS, seed, 200, arch)
    assert a["best_val"][0] == b["best_val"][0]
    pa = ax.predict_state("TCN", a["best_val"][1], X, None, NCLASS, av.INPUT_DIM, arch)
    pb = ay.predict_state("TCN", b["best_val"][1], X, None, NCLASS, av.INPUT_DIM, arch)
    assert np.array_equal(pa, pb)


def test_apply_init_xavier_deterministic():
    arch = av.architecture()
    torch.manual_seed(7)
    m1 = av.build_reader("TCN", NCLASS, arch); ab.apply_init(m1, "xavier")
    torch.manual_seed(7)
    m2 = av.build_reader("TCN", NCLASS, arch); ab.apply_init(m2, "xavier")
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(v1, v2)
    # default init differs, and xavier zeroes biases
    torch.manual_seed(7)
    m0 = av.build_reader("TCN", NCLASS, arch)
    assert not torch.equal(m1.convs[0].weight, m0.convs[0].weight)
    assert float(m1.head.bias.abs().sum()) == 0.0
    assert float(m1.convs[1].bias.abs().sum()) == 0.0


def test_apply_init_xavier_bounds():
    arch = av.architecture()
    torch.manual_seed(0)
    m = av.build_reader("TCN", NCLASS, arch); ab.apply_init(m, "xavier")
    for prm_name, prm in [("head", m.head.weight)] + [(f"conv{i}", c.weight) for i, c in
                                                      enumerate(m.convs)]:
        assert float(prm.abs().max()) <= 1.0  # xavier bound sqrt(6/(fan_in+fan_out)); small here


def test_make_optimizer_families():
    arch = av.architecture()
    m = av.build_reader("TCN", NCLASS, arch)
    a = ab.make_optimizer(m, ab.COND_BY_NAME["adam_ref"])
    assert isinstance(a, torch.optim.Adam) and abs(a.param_groups[0]["lr"] - ab.LR) < 1e-12
    w = ab.make_optimizer(m, ab.COND_BY_NAME["adamw"])
    assert isinstance(w, torch.optim.AdamW) and abs(w.param_groups[0]["weight_decay"] - ab.WD) < 1e-12
    s = ab.make_optimizer(m, ab.COND_BY_NAME["sgd_mom"])
    assert isinstance(s, torch.optim.SGD) and s.param_groups[0]["momentum"] == 0.9
    assert s.param_groups[0]["nesterov"] is False
    sn = ab.make_optimizer(m, ab.COND_BY_NAME["sgd_nesterov"])
    assert sn.param_groups[0]["nesterov"] is True
    r = ab.make_optimizer(m, ab.COND_BY_NAME["rmsprop"])
    assert isinstance(r, torch.optim.RMSprop) and r.param_groups[0]["alpha"] == 0.99


def test_train_ab_exact_updates_and_determinism():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NCLASS, 16)
    am = ab.COND_BY_NAME["adam_ref"]
    a = ab.train_ab("TCN", X, y, X, y, NCLASS, 3, 40, am, av.architecture())
    assert a["events"][-1]["update"] == 40 and len(a["events"]) == 4
    b = ab.train_ab("TCN", X, y, X, y, NCLASS, 3, 40, am, av.architecture())
    assert [e["val_acc"] for e in a["events"]] == [e["val_acc"] for e in b["events"]]


def test_fit_seed_convention():
    for seed in SEEDS:
        for R in READERS:
            for rep in REPS:
                assert ab._fit_seed(seed, R, rep) == aw.reader_seed(seed, R, "op_dest", rep, TOP, "main")


# ---------------------------------------------------------------------------
# decision-rule encoding
# ---------------------------------------------------------------------------

def _mk_res(cell_fn):
    readers = {}
    for R in READERS:
        readers[R] = {}
        for rep in REPS:
            readers[R][rep] = {}
            for c in ab.COND_NAMES:
                mean, nfit = cell_fn(R, rep, c)
                readers[R][rep][c] = {"train_max": {"mean": mean},
                                      "per_seed_train_max": [0.99] * nfit + [0.4] * (5 - nfit),
                                      "best_val": {"acc": {f"{ab.PRIMARY_MODEL}:all_add": {"mean": mean}}},
                                      "final": {"acc": {f"{ab.PRIMARY_MODEL}:all_add": {"mean": mean}}}}
    return {"readers": readers}


def _mk_aa(cell_fn):
    return {R: {rep: {"train_max": cell_fn(R, rep, "adam_ref")[0]} for rep in REPS} for R in READERS}


def test_rule_ab_b():
    def fn(R, rep, c):
        return (1.0, 5) if R == "GRU" else (0.30, 0)
    d = ab.classify(_mk_res(fn), [], _mk_aa(fn))
    assert d["verdict"] == "AB-B" and d["control_ok"] and d["reproduce_aa"]


def test_rule_ab_a():
    def fn(R, rep, c):
        if R == "GRU":
            return (1.0, 5)
        if c == "adam_xavier":
            return (0.96, 5)
        return (0.30, 0)
    d = ab.classify(_mk_res(fn), [], _mk_aa(fn))
    assert d["verdict"] == "AB-A" and "adam_xavier" in d["qualified_conditions"]


def test_rule_ab_mix_one_history():
    def fn(R, rep, c):
        if R == "GRU":
            return (1.0, 5)
        if rep == "L4" and c == "rmsprop":
            return (0.95, 5)
        return (0.30, 0)
    d = ab.classify(_mk_res(fn), [], _mk_aa(fn))
    assert d["verdict"] == "AB-MIX"


def test_rule_ab_mix_seed_dependent():
    def fn(R, rep, c):
        if R == "GRU":
            return (1.0, 5)
        if rep == "L4" and c == "sgd_mom":
            return (0.92, 3)
        return (0.30, 0)
    d = ab.classify(_mk_res(fn), [], _mk_aa(fn))
    assert d["verdict"] == "AB-MIX"


def test_rule_ab_invalid_control():
    d = ab.classify(_mk_res(lambda R, rep, c: (0.30, 0)), [], _mk_aa(lambda R, rep, c: (0.30, 0)))
    assert d["verdict"] == "AB-INVALID" and d["control_ok"] is False


def test_rule_ab_invalid_reproduction():
    def fn(R, rep, c):
        return (1.0, 5) if R == "GRU" else (0.30, 0)
    aa = _mk_aa(fn)
    aa["TCN"]["L4"]["train_max"] = 0.99          # adam_ref will NOT match AA
    d = ab.classify(_mk_res(fn), [], aa)
    assert d["verdict"] == "AB-INVALID" and d["reproduce_aa"] is False


def test_rule_ab_invalid_problems():
    d = ab.classify(_mk_res(lambda R, rep, c: (1.0, 5)), ["preflight failed"],
                    _mk_aa(lambda R, rep, c: (1.0, 5)))
    assert d["verdict"] == "AB-INVALID"


# ---------------------------------------------------------------------------
# results completeness / controls
# ---------------------------------------------------------------------------

def test_all_conditions_reported(built):
    R = built["readers"]
    assert set(R) == set(READERS)
    for reader in READERS:
        for rep in REPS:
            for c in ab.COND_NAMES:
                assert len(R[reader][rep][c]["per_seed_train_max"]) == 5


def test_reproduce_aa_in_results(built):
    d = built["primary_decision"]
    assert d["reproduce_aa"] is True
    for R in READERS:
        for rep in REPS:
            got = built["readers"][R][rep]["adam_ref"]["train_max"]["mean"]
            exp = built["aa_baseline"][R][rep]["train_max"]
            assert abs(got - exp) < 1e-9


def test_gru_control_ok(built):
    assert built["primary_decision"]["control_ok"] is True


def test_ef_control(built):
    R = built["readers"]
    for reader in READERS:
        for rep in REPS:
            for c in ab.COND_NAMES:
                for st in ("best_val", "final"):
                    a = R[reader][rep][c][st]["acc"]
                    assert abs(a["E_candidate:all_add"]["mean"]
                               - a["F_combined:all_add"]["mean"]) < 1e-9


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    assert d["thresholds"]["fit_bar"] == 0.90 and d["thresholds"]["robust_seeds"] == 4
    if d["verdict"] == "AB-A":
        assert d["qualified_conditions"]


def test_no_test_based_selection(built):
    disc = built["metadata"]["statistical_discipline"]
    assert "VAL-only" in disc and "never used" in disc


def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        ab.load(str(tmp_path / "nope.npz"))


# ---------------------------------------------------------------------------
# frozen-model protection
# ---------------------------------------------------------------------------

def test_no_reason_parameter_mutation_and_no_grad(loaded_model):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NCLASS, 16)
    ab.train_ab("TCN", X, y, X, y, NCLASS, 5, 20, ab.COND_BY_NAME["adamw"], av.architecture())
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


def test_reader_model_separation(loaded_model):
    model = av.build_reader("TCN", NCLASS, av.architecture())
    ids = {id(p) for p in loaded_model.parameters()}
    assert not (ids & {id(p) for p in model.parameters()})


# ---------------------------------------------------------------------------
# figures / immutability / governance
# ---------------------------------------------------------------------------

def test_figure_generation(data, built, tmp_path):
    figs = ab.make_figures(data, built, str(tmp_path), prefix="test_ab")
    assert len(figs) >= 8 and all(os.path.exists(p) for p in figs)


def test_historical_immutability():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"


def test_research_governance_compliance():
    log = open(os.path.join(ROOT, "docs", "RESEARCH_LOG.md")).read()
    assert "experiment_ab_optimizer_init.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_ab_optimizer_init.md" in baseline["reports"]
