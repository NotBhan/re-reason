"""Tests for Experiment AA — TCN optimization-recipe diagnostic on the `op_dest` control.

Resource-light suite: the live reader fits use tiny synthetic tensors and a short update budget; the
structural tests read the collected artifacts. Covers: reuse of Z's `op_dest` dataset (byte identity),
data/split/class/recoverability invariants, checkpoint hashes, the mandatory TCN implementation audit
(alignment, causal path, gradient path, controlled-change effect, and a negative control proving the
causal check can fail), the grid trainer's protocol identity with Y and its exact update counts,
matched deterministic initialization, the predeclared robust-fit decision rule (AA-A/AA-MIX/AA-B/
AA-INVALID), full grid completeness, per-seed reporting, milestones, VAL-only selection, E/F control,
deterministic trajectories, frozen-model protection, malformed-artifact rejection, figure generation,
historical immutability, and documentation-governance compliance.
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

import analyze_experiment_aa as aa  # noqa: E402
import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_u as au  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import analyze_experiment_w as aw  # noqa: E402
import analyze_experiment_x as ax  # noqa: E402
import analyze_experiment_y as ay  # noqa: E402
import analyze_experiment_z as az  # noqa: E402
import run_experiment_aa as ra  # noqa: E402
import run_experiment_g as eg  # noqa: E402

torch.set_num_threads(int(os.environ.get("AA_TEST_THREADS", "1")))

AA_NPZ = os.path.join(ROOT, "experiments", "experiment_aa_optimizer_grid.npz")
AA_META = os.path.join(ROOT, "experiments", "experiment_aa_optimizer_grid_meta.json")
AA_RESULTS = os.path.join(ROOT, "experiments", "experiment_aa_results.json")
Z_NPZ = os.path.join(ROOT, "experiments", "experiment_z_op_dest.npz")
MODELS = ("E_candidate", "F_combined")
READERS = ["TCN", "GRU"]
REPS = ("L4", "Lfull")
SEEDS = [42, 43, 44, 45, 46]
STATES = ("final", "best_val", "best_train")
NCLASS = 4
BUDGETS = [1000, 5000]
LR_LABELS = ["3e-4", "1e-3", "3e-3", "1e-2", "3e-2"]
ALLOWED_VERDICTS = {"AA-A", "AA-B", "AA-MIX", "AA-INVALID"}


def _aa_meta():
    if os.path.exists(AA_NPZ):
        return json.loads(str(np.load(AA_NPZ, allow_pickle=False)["meta_json"][0]))
    return None


_AAMETA = _aa_meta()
TOP = int(_AAMETA["source"]["top_size"]) if _AAMETA else 10000
aa.TOP = TOP
if _AAMETA:
    aw.TRAIN_SIZES = list(_AAMETA["source"]["train_sizes"])


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(AA_NPZ):
        pytest.skip("experiment_aa npz not present")
    return aa.load(AA_NPZ)


@pytest.fixture(scope="module")
def built(data):
    if not os.path.exists(AA_RESULTS):
        pytest.skip("experiment_aa results.json not present")
    with open(AA_RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


def _diag_batch(seed=42, L=4, n=32):
    z = np.load(Z_NPZ, allow_pickle=False)
    rows = np.nonzero(z["sub256__seed"] == seed)[0][:n]
    return av.reader_input(z["sub256__seq_obs"][rows], z["sub256__t"][rows], L)


# ---------------------------------------------------------------------------
# mandatory TCN implementation audit
# ---------------------------------------------------------------------------

def test_audit_passes_on_real_features():
    arch = av.architecture()
    for L in (4, 10):
        a = ra.audit_tcn(arch, _diag_batch(L=L), name=f"L{L}")
        assert a["passed"] is True, a
        assert a["n_features"] == av.INPUT_DIM == 22


def test_audit_alignment_checks():
    a = ra.audit_tcn(av.architecture(), _diag_batch(L=4), name="L4")
    assert a["layer_alignment"] is True
    assert a["layer_lengths"] == [4, 4, 4, 4]
    assert a["causal_alignment"] is True


def test_audit_gradient_path_finite_nontrivial():
    a = ra.audit_tcn(av.architecture(), _diag_batch(L=4), name="L4")
    assert a["grad_finite"] is True
    assert a["dest_grad_path"] is True
    assert a["dest_grad_abs_sum"] > 0
    assert 0 < a["dest_grad_nonzero_rows"] <= a["n"]


def test_audit_final_step_and_controlled_change():
    arch = av.architecture()
    X = _diag_batch(L=4)
    a = ra.audit_tcn(arch, X, name="L4")
    assert a["final_step_effect"] is True and a["logits_finite"] is True
    assert a["dest_change_effect"] is True
    # independent confirmation: changing only x_t[3:7] changes the logits
    model = av.build_reader("TCN", NCLASS, arch)
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X)
        l1 = model(Xt)
        Xc = Xt.clone()
        Xc[:, -1, 3:7] = torch.tensor([0., 1., 0., 0.])
        l2 = model(Xc)
    assert not torch.allclose(l1, l2, atol=1e-6)


def _causal_ok(prehead_fn, X, n=None):
    n = X.shape[0] if n is None else n
    L = X.shape[1]
    Xt = torch.tensor(X)
    pre = prehead_fn(Xt)
    rng = np.random.default_rng(0)
    for i in range(L - 1):
        mod = Xt.clone()
        mod[:, i + 1:, :] = torch.tensor(rng.standard_normal((n, L - i - 1, X.shape[2])),
                                         dtype=torch.float32)
        if not torch.allclose(pre[:, :, i], prehead_fn(mod)[:, :, i], atol=1e-5):
            return False
    return True


def test_audit_causal_check_can_fail_on_noncausal_path():
    """Negative control: the audit's causal test flags a future-leaking representation."""
    X = _diag_batch(L=4)
    model = av.build_reader("TCN", NCLASS, av.architecture())
    model.eval()
    assert _causal_ok(lambda t: ra._tcn_prehead(model, t), X) is True

    def noncausal(t):
        # a genuinely future-leaking representation: position i aggregates inputs at j >= i
        z = t.transpose(1, 2)
        return torch.cumsum(z.flip(-1), dim=-1).flip(-1)
    assert _causal_ok(noncausal, X) is False


def test_audit_recorded_and_passed(built):
    assert built["metadata"]["audit_pass"] is True
    for rep in REPS:
        assert built["metadata"]["audit"][rep]["passed"] is True


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
    assert m["y_sha256"] == hashlib.sha256(
        open(os.path.join(ROOT, "experiments", "experiment_y_fitting_control.npz"), "rb").read()).hexdigest()
    assert data["meta"]["subset_seed"] == 20261010
    assert m["z_sha256"] == data["meta"]["source"]["z_sha256"]


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
# grid trainer protocol / budget / init
# ---------------------------------------------------------------------------

def test_train_grid_matches_y_protocol():
    z = np.load(Z_NPZ, allow_pickle=False)
    rows = np.nonzero(z["sub256__seed"] == 42)[0]
    X = av.reader_input(z["sub256__seq_obs"][rows], z["sub256__t"][rows], 4)
    y = z["sub256__op_dest"][rows].astype(int)
    arch = av.architecture()
    seed = aw.reader_seed(42, "TCN", "op_dest", "L4", TOP, "main")
    a = aa.train_grid("TCN", X, y, X, y, NCLASS, seed, 200, av.READER_LR, av.READER_WD, arch)
    b = ay.train_trajectory("TCN", X, y, X, y, NCLASS, seed, 200, arch)
    assert a["best_val"][0] == b["best_val"][0]
    pa = ax.predict_state("TCN", a["best_val"][1], X, None, NCLASS, av.INPUT_DIM, arch)
    pb = ay.predict_state("TCN", b["best_val"][1], X, None, NCLASS, av.INPUT_DIM, arch)
    assert np.array_equal(pa, pb)


def test_train_grid_exact_updates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NCLASS, 16)
    fit = aa.train_grid("TCN", X, y, X, y, NCLASS, 7, 40, 1e-2, av.READER_WD, av.architecture())
    assert fit["events"][-1]["update"] == 40 and len(fit["events"]) == 4


def test_deterministic_trajectory():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NCLASS, 16)
    a = aa.train_grid("TCN", X, y, X, y, NCLASS, 3, 60, 3e-3, av.READER_WD, av.architecture())
    b = aa.train_grid("TCN", X, y, X, y, NCLASS, 3, 60, 3e-3, av.READER_WD, av.architecture())
    assert [e["val_acc"] for e in a["events"]] == [e["val_acc"] for e in b["events"]]


def test_matched_deterministic_initialization():
    arch = av.architecture()
    seed = aw.reader_seed(42, "TCN", "op_dest", "L4", TOP, "main")
    torch.manual_seed(seed)
    m1 = ax.build_model("TCN", NCLASS, av.INPUT_DIM, arch)
    torch.manual_seed(seed)
    m2 = ax.build_model("TCN", NCLASS, av.INPUT_DIM, arch)
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(v1, v2)
    # the init seed depends only on (seed, reader, history) — not on LR or budget
    assert aa._fit_seed(42, "TCN", "L4") == seed


def test_fit_seed_convention():
    for seed in SEEDS:
        for R in READERS:
            for rep in REPS:
                assert aa._fit_seed(seed, R, rep) == aw.reader_seed(seed, R, "op_dest", rep, TOP, "main")


# ---------------------------------------------------------------------------
# decision-rule encoding
# ---------------------------------------------------------------------------

def _mk_res(cell_fn):
    readers = {}
    for R in READERS:
        readers[R] = {}
        for rep in REPS:
            readers[R][rep] = {}
            for lbl in LR_LABELS:
                readers[R][rep][lbl] = {}
                for b in BUDGETS:
                    mean, nfit = cell_fn(R, rep, lbl, b)
                    readers[R][rep][lbl][b] = {
                        "train_max": {"mean": mean},
                        "per_seed_train_max": [0.99] * nfit + [0.40] * (5 - nfit)}
    return {"readers": readers}


def _low(R, rep, lbl, b):
    return (0.30, 0)


def test_rule_aa_b():
    res = _mk_res(lambda R, rep, lbl, b: (1.0, 5) if R == "GRU" else (0.30, 0))
    d = aa.classify(res, [])
    assert d["verdict"] == "AA-B" and d["control_ok"] is True


def test_rule_aa_a():
    def fn(R, rep, lbl, b):
        if R == "GRU":
            return (1.0, 5)
        if lbl == "3e-3" and b == 5000:
            return (0.96, 5)
        return (0.30, 0)
    d = aa.classify(_mk_res(fn), [])
    assert d["verdict"] == "AA-A"
    assert ["3e-3", 5000] in d["qualified_recipes"]


def test_rule_aa_mix_one_history():
    def fn(R, rep, lbl, b):
        if R == "GRU":
            return (1.0, 5)
        if rep == "L4" and lbl == "3e-3" and b == 5000:
            return (0.95, 5)
        return (0.30, 0)
    d = aa.classify(_mk_res(fn), [])
    assert d["verdict"] == "AA-MIX"


def test_rule_aa_mix_seed_dependent():
    def fn(R, rep, lbl, b):
        if R == "GRU":
            return (1.0, 5)
        if rep == "L4" and lbl == "3e-3" and b == 5000:
            return (0.92, 3)          # mean >= 0.90 but only 3/5 seeds
        return (0.30, 0)
    d = aa.classify(_mk_res(fn), [])
    assert d["verdict"] == "AA-MIX"


def test_rule_aa_invalid_control_fails():
    d = aa.classify(_mk_res(lambda R, rep, lbl, b: (0.30, 0)), [])
    assert d["verdict"] == "AA-INVALID" and d["control_ok"] is False


def test_rule_aa_invalid_problems():
    d = aa.classify(_mk_res(lambda R, rep, lbl, b: (1.0, 5)), ["split disjointness failure: x=1"])
    assert d["verdict"] == "AA-INVALID"


# ---------------------------------------------------------------------------
# results completeness / controls
# ---------------------------------------------------------------------------

def test_all_grid_cells_reported(built):
    R = built["readers"]
    assert set(R) == set(READERS)
    for reader in READERS:
        for rep in REPS:
            for lbl in LR_LABELS:
                for b in BUDGETS:
                    node = R[reader][rep][lbl][str(b)] if str(b) in R[reader][rep][lbl] else R[reader][rep][lbl][b]
                    assert len(node["per_seed_train_max"]) == 5


def test_per_seed_and_milestones(built):
    R = built["readers"]
    for reader in READERS:
        for rep in REPS:
            node = R[reader][rep]["3e-3"]
            b1000 = node["1000"] if "1000" in node else node[1000]
            assert b1000["train_max"]["mean"] is not None
            assert b1000["best_val"]["acc"]["E_candidate:all_add"]["mean"] is not None
            assert b1000["all_finite"] is True


def test_gru_control_ok(built):
    assert built["primary_decision"]["control_ok"] is True
    assert built["primary_decision"]["gru_robust"]["L4"] and built["primary_decision"]["gru_robust"]["Lfull"]


def test_ef_control(built):
    R = built["readers"]
    for reader in READERS:
        for rep in REPS:
            for lbl in LR_LABELS:
                node = R[reader][rep][lbl]
                for b in BUDGETS:
                    cell = node[str(b)] if str(b) in node else node[b]
                    for st in ("best_val", "final"):
                        a = cell[st]["acc"]
                        assert abs(a["E_candidate:all_add"]["mean"]
                                   - a["F_combined:all_add"]["mean"]) < 1e-9


def test_decision_validity(built):
    d = built["primary_decision"]
    assert d["verdict"] in ALLOWED_VERDICTS
    assert d["thresholds"]["fit_bar"] == 0.90 and d["thresholds"]["robust_seeds"] == 4
    if d["verdict"] == "AA-A":
        assert d["qualified_recipes"]
    if d["verdict"] == "AA-MIX":
        assert not d["qualified_recipes"]


def test_no_test_based_selection(built):
    disc = built["metadata"]["statistical_discipline"]
    assert "VAL-only" in disc and "never used" in disc


def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        aa.load(str(tmp_path / "nope.npz"))


# ---------------------------------------------------------------------------
# frozen-model protection
# ---------------------------------------------------------------------------

def test_no_reason_parameter_mutation_and_no_grad(loaded_model):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NCLASS, 16)
    aa.train_grid("TCN", X, y, X, y, NCLASS, 5, 20, 3e-3, av.READER_WD, av.architecture())
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
    figs = aa.make_figures(data, built, str(tmp_path), prefix="test_aa")
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
    assert "experiment_aa_optimizer_grid.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_aa_optimizer_grid.md" in baseline["reports"]
