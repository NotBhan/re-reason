"""Tests for Experiment AC — TCN re-qualification on the ADD `y_true` target.

Resource-light suite: live fits use tiny synthetic tensors and a short budget; structural tests read
the collected artifacts. Covers: the mandatory trainable-parameter preflight (parameter groups,
finiteness, per-stage gradient/update evidence, dead-stage detection with a frozen negative control),
reuse of Y's `y_true` dataset (byte identity), the `y_true = (a+b) mod 10` derivation, held-out-pair
exclusion, data/split/class invariants, checkpoint hashes, the three predeclared conditions (Adam/
default, AdamW/default, Adam/xavier) and their reuse of AB's exact optimizer/initialization
implementations, the byte-identical-to-Y trainer, determinism, the fitting decision rule
(AC-A/AC-MIX/AC-B/AC-INVALID) and the separate generalization rule, the adam_ref↔Y reproduction
bridge, the GRU control, E/F control, VAL-only selection, frozen-model protection, malformed-artifact
rejection, figure generation, historical immutability, and documentation-governance compliance.
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
import analyze_experiment_ac as ac  # noqa: E402
import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import analyze_experiment_w as aw  # noqa: E402
import analyze_experiment_y as ay  # noqa: E402
import run_experiment_ac as rac  # noqa: E402
import run_experiment_g as eg  # noqa: E402

torch.set_num_threads(int(os.environ.get("AC_TEST_THREADS", "1")))

AC_NPZ = os.path.join(ROOT, "experiments", "experiment_ac_ytrue_requalification.npz")
AC_META = os.path.join(ROOT, "experiments", "experiment_ac_ytrue_meta.json")
AC_RESULTS = os.path.join(ROOT, "experiments", "experiment_ac_results.json")
Z_NPZ = os.path.join(ROOT, "experiments", "experiment_z_op_dest.npz")
Y_NPZ = os.path.join(ROOT, "experiments", "experiment_y_fitting_control.npz")
Y_RESULTS = os.path.join(ROOT, "experiments", "experiment_y_results.json")
MODELS = ("E_candidate", "F_combined")
READERS = ["TCN", "GRU"]
REPS = ("L4", "Lfull")
SEEDS = [42, 43, 44, 45, 46]
STATES = ("final", "best_val", "best_train")
NCLASS = 10
ALLOWED_FIT = {"AC-A", "AC-B", "AC-MIX", "AC-INVALID"}
ALLOWED_GEN = {"fit_without_generalization", "material_transfer", "partial_transfer", "not_fitted"}


def _ac_meta():
    if os.path.exists(AC_NPZ):
        return json.loads(str(np.load(AC_NPZ, allow_pickle=False)["meta_json"][0]))
    return None


_ACMETA = _ac_meta()
TOP = int(_ACMETA["source"]["top_size"]) if _ACMETA else 10000
ac.TOP = TOP
if _ACMETA:
    aw.TRAIN_SIZES = list(_ACMETA["source"]["train_sizes"])


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(AC_NPZ):
        pytest.skip("experiment_ac npz not present")
    return ac.load(AC_NPZ)


@pytest.fixture(scope="module")
def built(data):
    if not os.path.exists(AC_RESULTS):
        pytest.skip("experiment_ac results.json not present")
    with open(AC_RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


def _diag(seed=42, L=4, n=32):
    z = np.load(AC_NPZ, allow_pickle=False)
    rows = np.nonzero(z["sub256__seed"] == seed)[0][:n]
    X = av.reader_input(z["sub256__seq_obs"][rows], z["sub256__t"][rows], L)
    y = z["sub256__y_true"][rows].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# mandatory trainable-parameter preflight
# ---------------------------------------------------------------------------

def test_preflight_passes(data):
    assert data["meta"]["preflight_pass"] is True
    for k in ("TCN_L4", "TCN_Lfull", "GRU_L4", "GRU_Lfull"):
        assert data["meta"]["preflight"][k]["passed"] is True


def test_preflight_param_groups(data):
    arch = av.architecture()
    assert data["meta"]["preflight"]["TCN_L4"]["n_trainable_params"] == \
        av.reader_param_count("TCN", NCLASS, arch) == 28506
    assert data["meta"]["preflight"]["GRU_L4"]["n_trainable_params"] == \
        av.reader_param_count("GRU", NCLASS, arch) == 28655
    for k, p in data["meta"]["preflight"].items():
        assert p["all_in_optimizer"] is True and p["all_requires_grad"] is True


def test_preflight_loss_and_grad_finite(data):
    for k, p in data["meta"]["preflight"].items():
        assert p["loss_finite"] is True and p["grad_finite"] is True
        assert p["aggregate_grad_norm"] > 0
        assert p["dead_stages"] == []


def test_preflight_stage_updates(data):
    for k, p in data["meta"]["preflight"].items():
        assert p["any_parameter_updated"] is True and p["logit_change"] > 0
        for st, v in p["per_stage_grad"].items():
            assert v["grad_norm"] > 0, (k, st)
        for st, v in p["per_stage_update"].items():
            assert v["effectively_unchanged"] is False, (k, st)


def test_grad_stats_flags_dead_stage():
    arch = av.architecture()
    frozen = av.build_reader("TCN", NCLASS, arch)
    for prm in frozen.parameters():
        prm.requires_grad_(False)
    X = torch.randn(8, 4, av.INPUT_DIM, requires_grad=True)
    y = torch.zeros(8, dtype=torch.long)
    torch.nn.functional.cross_entropy(frozen(X), y).backward()
    st = rac._grad_stats(rac.iter_stages(frozen, "TCN"))
    assert all(v["dead"] for v in st.values())
    live = av.build_reader("TCN", NCLASS, arch)
    torch.nn.functional.cross_entropy(live(torch.randn(8, 4, av.INPUT_DIM)), y).backward()
    st2 = rac._grad_stats(rac.iter_stages(live, "TCN"))
    assert any(not v["dead"] for v in st2.values())


def test_preflight_direct_call():
    X, y = _diag(L=4)
    p = rac.preflight_reader("TCN", av.architecture(), X, y, name="t", steps=5)
    assert p["passed"] is True
    assert set(p["per_stage_grad"]) == {"conv0", "conv1", "conv2", "conv3", "head"}
    g = rac.preflight_reader("GRU", av.architecture(), X, y, name="g", steps=5)
    assert g["passed"] is True and set(g["per_stage_grad"]) == {"gru", "head"}


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
    assert data["meta"]["subset_size"] == 256


def test_target_derivation_and_class_coverage(data):
    td = data["meta"]["target_derivation"]
    for k, rec in td.items():
        assert rec["exact"] is True and rec["rule"] == "(operand_a + operand_b) mod 10", k
    cc = data["meta"]["class_counts"]
    assert cc["train"]["pooled"] == [130, 130, 130, 130, 130, 130, 125, 125, 125, 125]
    for mk in MODELS:
        assert len(cc["test"][mk]) == NCLASS and all(c > 0 for c in cc["test"][mk])


def test_heldout_exclusion_and_disjointness(data):
    assert data["meta"]["heldout_exclusion"]["train"]["n_with_b_in_{0,5}"] == 0
    assert all(v == 0 for v in data["meta"]["disjointness"].values())
    z = data["z"]
    assert not np.isin(z["sub256__operand_b"].astype(int), [0, 5]).any()


def test_checkpoint_hashes_unchanged(data):
    for mk in MODELS:
        for s in SEEDS:
            rec = data["meta"]["models"][mk]["checkpoints"][str(s)]
            assert eg.sha256_file(os.path.join(ROOT, rec["path"])) == rec["sha256"]


# ---------------------------------------------------------------------------
# conditions / trainer
# ---------------------------------------------------------------------------

def test_condition_definition():
    assert ac.COND_NAMES == ["adam_ref", "adamw", "adam_xavier"]
    assert ac.QUALIFIED == ["adamw", "adam_xavier"]
    assert "adam_ref" not in ac.QUALIFIED
    assert ac.COND_BY_NAME["adam_ref"]["opt"] == "adam" and ac.COND_BY_NAME["adam_ref"]["init"] == "default"
    assert ac.COND_BY_NAME["adamw"]["opt"] == "adamw" and ac.COND_BY_NAME["adamw"]["init"] == "default"
    assert ac.COND_BY_NAME["adam_xavier"]["opt"] == "adam" and ac.COND_BY_NAME["adam_xavier"]["init"] == "xavier"
    # reuse of AB's exact implementations
    assert ac.ab.train_ab is ab.train_ab
    assert ac.ab.apply_init is ab.apply_init and ac.ab.make_optimizer is ab.make_optimizer


def test_trainer_bit_identical_to_y():
    """AC's adam_ref trainer reproduces Y's trainer exactly (the strong protocol-identity check)."""
    X, y = _diag(L=4)
    seed = aw.reader_seed(42, "TCN", "y_true", "L4", TOP, "main")
    a = ab.train_ab("TCN", X, y, X, y, NCLASS, seed, 100, ab.COND_BY_NAME["adam_ref"], av.architecture())
    b = ay.train_trajectory("TCN", X, y, X, y, NCLASS, seed, 100, av.architecture())
    assert [e["train_acc"] for e in a["events"]] == [e["train_acc"] for e in b["events"]]
    assert [e["val_acc"] for e in a["events"]] == [e["val_acc"] for e in b["events"]]
    assert a["best_val"][0] == b["best_val"][0]
    # GRU too
    a2 = ab.train_ab("GRU", X, y, X, y, NCLASS, seed, 100, ab.COND_BY_NAME["adam_ref"], av.architecture())
    b2 = ay.train_trajectory("GRU", X, y, X, y, NCLASS, seed, 100, av.architecture())
    assert [e["train_acc"] for e in a2["events"]] == [e["train_acc"] for e in b2["events"]]


def test_fit_seed_convention():
    for seed in SEEDS:
        for R in READERS:
            for rep in REPS:
                assert ac._fit_seed(seed, R, rep) == aw.reader_seed(seed, R, "y_true", rep, TOP, "main")


def test_train_ab_exact_updates_and_determinism():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NCLASS, 16)
    am = ac.COND_BY_NAME["adamw"]
    a = ab.train_ab("TCN", X, y, X, y, NCLASS, 3, 40, am, av.architecture())
    assert a["events"][-1]["update"] == 40 and len(a["events"]) == 4
    b = ab.train_ab("TCN", X, y, X, y, NCLASS, 3, 40, am, av.architecture())
    assert [e["val_acc"] for e in a["events"]] == [e["val_acc"] for e in b["events"]]


def test_apply_init_xavier_deterministic_and_reused():
    arch = av.architecture()
    torch.manual_seed(7)
    m1 = av.build_reader("TCN", NCLASS, arch); ab.apply_init(m1, "xavier")
    torch.manual_seed(7)
    m2 = av.build_reader("TCN", NCLASS, arch); ab.apply_init(m2, "xavier")
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(v1, v2)
    assert float(m1.head.bias.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# decision-rule encoding
# ---------------------------------------------------------------------------

def _mk_res(cell_fn):
    """cell_fn(reader, rep, cond) -> (train_mean, n_fit_ge_bar, classA, heldout, alladd)."""
    readers = {}
    for R in READERS:
        readers[R] = {}
        for rep in REPS:
            readers[R][rep] = {}
            for c in ac.COND_NAMES:
                mean, nfit, ca, ho, aa = cell_fn(R, rep, c)
                readers[R][rep][c] = {
                    "train_max": {"mean": mean},
                    "per_seed_train_max": [0.99] * nfit + [0.4] * (5 - nfit),
                    "best_val": {
                        "acc": {f"{ac.PRIMARY_MODEL}:classA": {"mean": ca},
                                f"{ac.PRIMARY_MODEL}:all_add": {"mean": aa}},
                        "heldout": {f"{ac.PRIMARY_MODEL}:classA": {"mean": ho}},
                        "seen": {f"{ac.PRIMARY_MODEL}:classA": {"mean": ca}},
                    },
                }
    return {"readers": readers}


def _repro(ok=True):
    return {"reproduced": ok, "max_abs_diff": 0.0 if ok else 0.5, "worst": None}


def test_rule_ac_a():
    def fn(R, rep, c):
        if R == "GRU":
            return (1.0, 5, 0.1, 0.1, 0.1)
        if c in ac.QUALIFIED:
            return (0.97, 5, 0.10, 0.10, 0.12)   # fits but does not transfer
        return (0.10, 0, 0.10, 0.10, 0.12)
    d = ac.classify_fit(_mk_res(fn), [], _repro(True), True)
    assert d["verdict"] == "AC-A" and d["qualified_verdicts"] == ["adamw", "adam_xavier"]


def test_rule_ac_mix_one_history():
    def fn(R, rep, c):
        if R == "GRU":
            return (1.0, 5, 0.1, 0.1, 0.1)
        if rep == "L4" and c == "adamw":
            return (0.95, 5, 0.1, 0.1, 0.1)
        return (0.10, 0, 0.1, 0.1, 0.1)
    d = ac.classify_fit(_mk_res(fn), [], _repro(True), True)
    assert d["verdict"] == "AC-MIX"


def test_rule_ac_mix_seed_dependent():
    def fn(R, rep, c):
        if R == "GRU":
            return (1.0, 5, 0.1, 0.1, 0.1)
        if c == "adam_xavier":
            return (0.92, 3, 0.1, 0.1, 0.1)     # mean>=0.90 but only 3/5 seeds
        return (0.10, 0, 0.1, 0.1, 0.1)
    d = ac.classify_fit(_mk_res(fn), [], _repro(True), True)
    assert d["verdict"] == "AC-MIX"


def test_rule_ac_b():
    def fn(R, rep, c):
        return (1.0, 5, 0.1, 0.1, 0.1) if R == "GRU" else (0.10, 0, 0.1, 0.1, 0.1)
    d = ac.classify_fit(_mk_res(fn), [], _repro(True), True)
    assert d["verdict"] == "AC-B"


def test_rule_ac_invalid_control():
    d = ac.classify_fit(_mk_res(lambda R, rep, c: (0.10, 0, 0.1, 0.1, 0.1)), [], _repro(True), False)
    assert d["verdict"] == "AC-INVALID"


def test_rule_ac_invalid_reproduction():
    def fn(R, rep, c):
        return (1.0, 5, 0.1, 0.1, 0.1) if R == "GRU" else (0.97, 5, 0.1, 0.1, 0.1)
    d = ac.classify_fit(_mk_res(fn), [], _repro(False), True)
    assert d["verdict"] == "AC-INVALID" and d["reproduce_y"] is False


def test_rule_ac_invalid_problems():
    d = ac.classify_fit(_mk_res(lambda R, rep, c: (1.0, 5, 0.1, 0.1, 0.1)), ["preflight failed"],
                        _repro(True), True)
    assert d["verdict"] == "AC-INVALID"


def test_generalization_rule():
    def fn(R, rep, c):
        if c in ac.QUALIFIED:
            return (0.97, 5, 0.10, 0.10, 0.12)
        return (0.10, 0, 0.10, 0.10, 0.12)
    res = _mk_res(fn)
    fit_dec = ac.classify_fit(res, [], _repro(True), True)
    y_base = {R: {rep: {"train_max": 0.1, "best_val_classA": 0.1, "best_val_all_add": 0.1,
                        "best_val_heldout_classA": 0.1} for rep in REPS} for R in READERS}
    g = ac.generalization(res, fit_dec, y_base)
    assert g["verdict"] == "fit_without_generalization"
    assert g["categories"]["adamw"] == "fit_without_generalization"


def test_generalization_material_transfer():
    def fn(R, rep, c):
        if c in ac.QUALIFIED:
            return (0.97, 5, 0.60, 0.35, 0.55)   # fits and transfers materially
        return (0.10, 0, 0.10, 0.10, 0.12)
    res = _mk_res(fn)
    fit_dec = ac.classify_fit(res, [], _repro(True), True)
    y_base = {R: {rep: {"train_max": 0.1, "best_val_classA": 0.1, "best_val_all_add": 0.1,
                        "best_val_heldout_classA": 0.1} for rep in REPS} for R in READERS}
    g = ac.generalization(res, fit_dec, y_base)
    assert g["categories"]["adamw"] == "material_transfer"


# ---------------------------------------------------------------------------
# results completeness / controls
# ---------------------------------------------------------------------------

def test_all_conditions_reported(built):
    R = built["readers"]
    assert set(R) == set(READERS)
    for reader in READERS:
        for rep in REPS:
            for c in ac.COND_NAMES:
                assert len(R[reader][rep][c]["per_seed_train_max"]) == 5
                for st in STATES:
                    assert st in R[reader][rep][c]


def test_reproduce_y_in_results(built):
    d = built["primary_decision"]
    assert d["reproduce_y"] is True
    ti = built["reproduction"]["trainer_identity"]
    assert ti["identical"] is True
    assert ti["max_abs_diff"] <= ac.IDENTITY_TOL and ti["n_compared"] > 0
    # the stored-JSON bridge is reported explicitly (drift is informational, never gating)
    sb = built["reproduction"]["stored_bridge"]
    assert sb["n_compared"] > 0 and sb["max_abs_diff_train"] is not None


def test_gru_control_ok(built):
    assert built["primary_decision"]["control_ok"] is True


def test_ef_control(built):
    R = built["readers"]
    for reader in READERS:
        for rep in REPS:
            for c in ac.COND_NAMES:
                for st in ("best_val", "final"):
                    a = R[reader][rep][c][st]["acc"]
                    assert abs(a["E_candidate:all_add"]["mean"]
                               - a["F_combined:all_add"]["mean"]) < 1e-9


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_FIT
    assert d["thresholds"]["fit_bar"] == 0.90 and d["thresholds"]["robust_seeds"] == 4
    assert built["generalization"]["verdict"] in ALLOWED_GEN


def test_no_test_based_selection(built):
    disc = built["metadata"]["statistical_discipline"]
    assert "VAL-only" in disc and "never used" in disc


def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        ac.load(str(tmp_path / "nope.npz"))


# ---------------------------------------------------------------------------
# frozen-model protection
# ---------------------------------------------------------------------------

def test_no_reason_parameter_mutation_and_no_grad(loaded_model):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NCLASS, 16)
    ab.train_ab("TCN", X, y, X, y, NCLASS, 5, 20, ac.COND_BY_NAME["adam_xavier"], av.architecture())
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
    figs = ac.make_figures(data, built, str(tmp_path), prefix="test_ac")
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
    assert "experiment_ac_ytrue_requalification.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_ac_ytrue_requalification.md" in baseline["reports"]
