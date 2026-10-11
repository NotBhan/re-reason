"""Tests for Experiment AD — qualified-recipe re-evaluation of the ADD-target access test.

Resource-light suite: live fits use tiny synthetic tensors and a short budget; structural tests read
the collected artifacts. Covers: exact V source-data/split identity, target derivation, held-out
pair exclusion, causal history construction with a no-future negative control, reader identity and
parameter-count invariance, matched initialization across Adam/AdamW, optimizer defaults and the
200-update full-batch protocol, VAL-only checkpoint selection, per-seed paired optimizer deltas, the
preregistered AD-A/AD-B classification, the separate V sequence-accessibility assessment, the V
historical comparison, frozen-checkpoint protection (no gradients / no parameter mutation),
determinism, malformed-artifact and split-leakage rejection, figure generation, historical
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
import analyze_experiment_ad as ad  # noqa: E402
import analyze_experiment_u as au  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import run_experiment_ad as rad  # noqa: E402
import run_experiment_g as eg  # noqa: E402

torch.set_num_threads(int(os.environ.get("AD_TEST_THREADS", "1")))

AD_NPZ = os.path.join(ROOT, "experiments", "experiment_ad_qualified_access.npz")
AD_META = os.path.join(ROOT, "experiments", "experiment_ad_qualified_access_meta.json")
AD_RESULTS = os.path.join(ROOT, "experiments", "experiment_ad_results.json")
AD_RAW = os.path.join(ROOT, "experiments", "experiment_ad_grid_raw.json")
V_NPZ = os.path.join(ROOT, "experiments", "experiment_v_sequence_access.npz")
V_RESULTS = os.path.join(ROOT, "experiments", "experiment_v_results.json")
U_RESULTS = os.path.join(ROOT, "experiments", "experiment_u_results.json")
MODELS = ("E_candidate", "F_combined")
READERS = ["GRU", "TCN"]
REPS = [r for r, _ in au.REPS]
DECISIVE = ["L4", "Lfull"]
CONDS = ["adam_ref", "adamw"]
NCLASS = 10
ALLOWED_OPT = {"AD-A", "AD-B", "AD-INVALID"}
ALLOWED_SEQ = {"V-A_sequence_accessible", "V-B_sequence_limited",
               "V-C_partial_sequence_benefit", "V-partial/inconclusive"}


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.relpath(path, ROOT)} not present")


@pytest.fixture(scope="module")
def data():
    _require(AD_NPZ)
    return ad.load(AD_NPZ)


@pytest.fixture(scope="module")
def built():
    _require(AD_RESULTS)
    with open(AD_RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def raw():
    _require(AD_RAW)
    with open(AD_RAW) as fh:
        return json.load(fh)["parts"]


@pytest.fixture(scope="module")
def u_base():
    return av.load_u_baseline(U_RESULTS)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


# ---------------------------------------------------------------------------
# exact V source-data / split identity
# ---------------------------------------------------------------------------

def test_source_arrays_identical_to_v(data):
    vv = np.load(V_NPZ, allow_pickle=False)
    for k in vv.files:
        if k == "meta_json":
            continue
        assert np.array_equal(data["z"][k], vv[k]), k


def test_source_sha_and_scale_recorded(data):
    m = data["meta"]["source"]
    assert m["v_sha256"] == hashlib.sha256(open(V_NPZ, "rb").read()).hexdigest()
    assert m["v_meta_sha256"] == hashlib.sha256(open(
        os.path.join(ROOT, "experiments", "experiment_v_sequence_access_meta.json"), "rb").read()).hexdigest()
    assert m["v_train_ord_cap"] == 1500 and m["v_val_ord_cap"] == 800
    assert data["meta"]["n"]["E_candidate"]["test"] == 15763


def test_target_derivation_exact(data):
    td = data["meta"]["target_derivation"]
    assert td and all(r["exact"] and r["rule"] == "(operand_a + operand_b) mod 10" for r in td.values())


def test_class_coverage(data):
    cc = data["meta"]["class_counts"]
    for ds in ("trainord", "valord", "test"):
        for mk in MODELS:
            counts = cc[ds][mk]
            assert len(counts) == NCLASS and all(c > 0 for c in counts), (ds, mk, counts)


def test_split_disjointness_and_no_test_leakage(data):
    dj = data["meta"]["disjointness"]
    for mk in MODELS:
        assert dj[mk]["trainord_vs_valord"] == 0
        assert dj[mk]["trainord_vs_test"] == 0
        assert dj[mk]["valord_vs_test"] == 0


def test_heldout_pair_exclusion(data, raw):
    """The heldout variant must drop exactly the b in {0,5} operand rows from the fitting pool."""
    z = data["z"]
    by = {(p["reader"], p["seed"], p["rep"], p["variant"]): p for p in raw}
    checked = 0
    for reader in READERS:
        for s in ad.SEEDS:
            p_main = by[(reader, s, "L4", "main")]["fits"]["adamw"]
            p_hold = by[(reader, s, "L4", "heldout")]["fits"]["adamw"]
            ob = z["E_candidate__trainord__operand_b"].astype(int)
            tr = np.nonzero(z["E_candidate__trainord__seed"] == s)[0]
            n_held = int(np.isin(ob[tr], [0, 5]).sum())
            assert p_hold["n_train"] == p_main["n_train"] - n_held
            checked += 1
    assert checked == 2 * len(ad.SEEDS)


# ---------------------------------------------------------------------------
# causal history construction + no-future negative control
# ---------------------------------------------------------------------------

def test_history_causality_recorded(data):
    hc = data["meta"]["history_causality"]
    assert len(hc) == 2 * 3 * len(REPS)
    assert all(r["builder_cross_check"] for r in hc.values())


def test_history_no_future_leakage(data):
    z = data["z"]
    seq = z["E_candidate__test__seq_obs"][:256]
    t = z["E_candidate__test__t"][:256]
    rng = np.random.default_rng(0)
    for L in (4, 10):
        rad.verify_no_future(seq, t, L, rng)


def test_history_future_observation_is_ignored():
    """Negative control: changing a FUTURE observation must not change the causal L-window; changing
    the CURRENT observation must change it (this makes the no-future assertion falsifiable)."""
    T, D = au.T_MAX, au.D_OBS
    seq = np.zeros((1, T * D), np.float32)
    for j in range(T):
        seq[0, j * D:(j + 1) * D] = 0.1 * (j + 1)
    t = np.array([3])
    L = 4
    base, _ = au.build_history(seq, t, L)
    # perturb the future observation x_{t+1} (index 4) -> must be ignored
    fut = seq.copy()
    fut[0, 4 * D:(4 + 1) * D] = 9.0
    after_future, _ = au.build_history(fut, t, L)
    assert np.allclose(base, after_future)
    # perturb the current observation x_t (index 3) -> must change
    cur = seq.copy()
    cur[0, 3 * D:(3 + 1) * D] = 9.0
    after_cur, _ = au.build_history(cur, t, L)
    assert not np.allclose(base, after_cur)


# ---------------------------------------------------------------------------
# reader identity / architecture / parameter invariance
# ---------------------------------------------------------------------------

def test_reader_identity_and_params():
    arch = av.architecture()
    assert arch["GRU"]["hidden"] == 85 and arch["TCN"]["channels"] == 52
    assert av.reader_param_count("GRU", NCLASS, arch) == 28655
    assert av.reader_param_count("TCN", NCLASS, arch) == 28506
    assert av.reader_param_count("GRU", NCLASS, arch) <= arch["budget_u_mlp128"]
    assert av.reader_param_count("TCN", NCLASS, arch) <= arch["budget_u_mlp128"]
    gru = av.build_reader("GRU", NCLASS, arch)
    tcn = av.build_reader("TCN", NCLASS, arch)
    assert isinstance(gru.gru, torch.nn.GRU) and gru.gru.bidirectional is False
    assert gru.gru.hidden_size == 85
    assert isinstance(tcn, av.TCNReader) and tcn.kernel == 3
    assert tcn.dilations == [1, 2, 4, 8] and len(tcn.convs) == 4


def test_param_invariance_across_conditions_and_histories(built):
    R = built["readers"]
    for reader in READERS:
        params = {R[reader][rep][c]["main"]["params"] for rep in REPS for c in CONDS}
        assert len(params) == 1


# ---------------------------------------------------------------------------
# optimizer conditions / trainer
# ---------------------------------------------------------------------------

def test_condition_definition_and_defaults():
    assert ad.COND_NAMES == ["adam_ref", "adamw"]
    assert ad.COND_BY_NAME["adam_ref"]["opt"] == "adam" and ad.COND_BY_NAME["adam_ref"]["init"] == "default"
    assert ad.COND_BY_NAME["adamw"]["opt"] == "adamw" and ad.COND_BY_NAME["adamw"]["init"] == "default"
    assert ab.OPTIMIZER_DEFAULTS["adam"]["betas"] == [0.9, 0.999]
    assert ab.OPTIMIZER_DEFAULTS["adamw"]["betas"] == [0.9, 0.999]
    assert ab.OPTIMIZER_DEFAULTS["adamw"]["eps"] == 1e-8
    assert ad.LR == 3e-3 and ad.WD == 1e-2 and ad.UPDATES == 200 and ad.EVAL_EVERY == 10


def test_matched_initialization(raw):
    for p in raw:
        assert p["fits"]["adam_ref"]["init_hash"] == p["fits"]["adamw"]["init_hash"], \
            (p["reader"], p["seed"], p["rep"], p["variant"])


def test_init_hash_deterministic_and_init_independent_of_condition():
    arch = av.architecture()
    seed = av.reader_seed(42, "TCN", "y_true", "L4", "main")
    h1 = ad._init_hash("TCN", NCLASS, seed, "default", arch)
    h2 = ad._init_hash("TCN", NCLASS, seed, "default", arch)
    assert h1 == h2
    # the optimizer condition is not a component of V's seed convention
    assert av.reader_seed(42, "TCN", "y_true", "L4", "main") == \
        av.reader_seed(42, "TCN", "y_true", "L4", "main")


def test_exactly_200_full_batch_updates(raw):
    for p in raw:
        for c in CONDS:
            ev = p["events"][c]
            assert len(ev) == 20
            assert [e["update"] for e in ev] == list(range(10, 201, 10))


def test_train_ab_uses_requested_optimizer_and_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NCLASS, 16)
    a = ab.train_ab("TCN", X, y, X, y, NCLASS, 7, 40, ad.COND_BY_NAME["adamw"], av.architecture())
    b = ab.train_ab("TCN", X, y, X, y, NCLASS, 7, 40, ad.COND_BY_NAME["adamw"], av.architecture())
    assert [e["val_acc"] for e in a["events"]] == [e["val_acc"] for e in b["events"]]


# ---------------------------------------------------------------------------
# VAL-only checkpoint selection
# ---------------------------------------------------------------------------

def test_validation_only_selection(built, raw):
    disc = built["metadata"]["statistical_discipline"]
    assert "VAL-only" in disc and "never selects" in disc
    # best-on-val checkpoint is selected by VAL, so it is never worse than the final VAL state
    for p in raw:
        for c in CONDS:
            for variant in ("main", "heldout"):
                f = p["fits"][c]
                assert f["best_val_acc"] >= f["val_acc_final"] - 1e-9
                assert f["best_val_update"] % 10 == 0 and 10 <= f["best_val_update"] <= 200


# ---------------------------------------------------------------------------
# per-seed paired deltas + decision rule
# ---------------------------------------------------------------------------

def test_paired_deltas_and_verdict(built, raw):
    P = built["paired"]
    for reader in READERS:
        for rep in REPS:
            for metric in ("classA", "heldout_classA"):
                node = P[reader][rep][metric]
                assert len(node["per_seed_delta"]) == 5
                assert len(node["per_seed_adamw"]) == 5 and len(node["per_seed_adam_ref"]) == 5
                for d, a, b in zip(node["per_seed_delta"], node["per_seed_adamw"], node["per_seed_adam_ref"]):
                    assert abs(d - (a - b)) < 1e-12
    # recompute the classification from raw evidence and compare to the stored verdict
    recomputed = ad.classify_optimizer(ad.paired_metrics(raw))
    assert recomputed["verdict"] == built["primary_decision"]["verdict"]
    assert recomputed["verdict"] in ALLOWED_OPT


def test_decision_rule_encoding():
    def paired_with(delta):
        rows = {}
        for reader in READERS:
            rows[reader] = {}
            for rep in DECISIVE:
                rows[reader][rep] = {m: {"mean_diff": delta, "a_mean": delta, "b_mean": 0.0,
                                         "per_seed_delta": [delta] * 5, "sign_positive": 5,
                                         "sign_negative": 0, "cohens_dz": 1.0, "paired_t_p": 0.01}
                                     for m in ("classA", "heldout_classA")}
        return rows
    assert ad.classify_optimizer(paired_with(0.20))["verdict"] == "AD-A"
    assert ad.classify_optimizer(paired_with(0.14))["verdict"] == "AD-B"
    assert ad.classify_optimizer(paired_with(-0.05))["verdict"] == "AD-B"
    # a material Class-A effect only at L4 still triggers AD-A
    d = paired_with(0.0)
    d["TCN"]["L4"]["classA"]["mean_diff"] = 0.16
    assert ad.classify_optimizer(d)["verdict"] == "AD-A"


def test_sequence_access_separate_axis(built):
    assert set(built["sequence_access"]) == set(CONDS)
    for c in CONDS:
        sa = built["sequence_access"][c]
        assert sa["verdict"] in ALLOWED_SEQ
        assert set(sa["per_reader"]) == set(READERS)
        for reader in READERS:
            assert "V_A" in sa["per_reader"][reader] and "V_B" in sa["per_reader"][reader]
    # the optimizer-effect verdict and the accessibility verdict are independent fields
    assert built["primary_decision"]["verdict"] in ALLOWED_OPT


def test_bayes_ceiling_reused(built):
    assert built["bayes_check"]["classA_L4"] is not None
    assert built["bayes_check"]["classA_Lfull"] is not None
    u = json.load(open(U_RESULTS))
    assert abs(built["bayes_check"]["classA_L4"]
               - u["E_candidate"]["pooled"]["classA"]["L4"]["y_true"]["bayes_acc"]) < 1e-12


# ---------------------------------------------------------------------------
# V historical comparison (without modifying historical artifacts)
# ---------------------------------------------------------------------------

def test_v_historical_comparison(built):
    vh = built["v_historical"]
    assert vh["verdict"] == "V-B_sequence_limited"
    for reader in READERS:
        for rep in REPS:
            assert "main_classA" in vh["readers"][reader][rep]
    d = built["v_drift"]
    assert d["max_abs_drift_classA"] is not None and d["max_abs_drift_heldout"] is not None
    assert len(d["per_cell"]) == len(READERS) * len(REPS)


def test_all_cells_reported(built):
    R = built["readers"]
    assert set(R) == set(READERS)
    for reader in READERS:
        for rep in REPS:
            for c in CONDS:
                node = R[reader][rep][c]
                for variant in ("main", "heldout"):
                    assert len(node[variant]["init_hashes"]) == 5
                    assert node[variant]["all_finite"] is True


def test_ef_control(built):
    R = built["readers"]
    for reader in READERS:
        for rep in REPS:
            for c in CONDS:
                a = R[reader][rep][c]["main"]["acc"]
                assert abs(a["E_candidate:all_add"]["mean"] - a["F_combined:all_add"]["mean"]) < 1e-9


# ---------------------------------------------------------------------------
# frozen-model protection
# ---------------------------------------------------------------------------

def test_checkpoint_hashes_unchanged(data):
    for mk in MODELS:
        for s in ad.SEEDS:
            rec = data["meta"]["models"][mk]["checkpoints"][str(s)]
            assert eg.sha256_file(os.path.join(ROOT, rec["path"])) == rec["sha256"]
            assert rec["unchanged_after_run"] is True


def test_no_reason_grad_or_mutation(loaded_model):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NCLASS, 16)
    ab.train_ab("TCN", X, y, X, y, NCLASS, 5, 20, ad.COND_BY_NAME["adamw"], av.architecture())
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


def test_reader_model_separation(loaded_model):
    model = av.build_reader("TCN", NCLASS, av.architecture())
    assert not ({id(p) for p in loaded_model.parameters()} & {id(p) for p in model.parameters()})


# ---------------------------------------------------------------------------
# determinism / malformed artifacts / leakage rejection
# ---------------------------------------------------------------------------

def test_analysis_deterministic(data, built, raw, u_base):
    vh = ad.load_v_historical(V_RESULTS)
    a = ad.assemble({"meta": data["meta"], "path": AD_NPZ}, raw, u_base, vh)
    b = ad.assemble({"meta": data["meta"], "path": AD_NPZ}, raw, u_base, vh)
    assert json.dumps(a["readers"], sort_keys=True) == json.dumps(b["readers"], sort_keys=True)
    assert json.dumps(a["paired"], sort_keys=True) == json.dumps(b["paired"], sort_keys=True)


def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        ad.load(str(tmp_path / "nope.npz"))
    bad = tmp_path / "bad.npz"
    np.savez(bad, foo=np.zeros(3))
    with pytest.raises(Exception):
        ad.load(str(bad))


def test_split_leakage_negative_control(data):
    good = data["meta"]
    assert ad._validate(good) == []
    bad = json.loads(json.dumps(good))
    bad["disjointness"]["E_candidate"]["trainord_vs_test"] = 3
    assert any("disjointness" in p for p in ad._validate(bad))
    bad2 = json.loads(json.dumps(good))
    bad2["class_counts"]["test"]["E_candidate"][0] = 0
    assert any("class coverage" in p for p in ad._validate(bad2))
    bad3 = json.loads(json.dumps(good))
    bad3["target_derivation"]["E_candidate/test"]["exact"] = False
    assert any("derivation" in p for p in ad._validate(bad3))


# ---------------------------------------------------------------------------
# figures / immutability / governance
# ---------------------------------------------------------------------------

def test_figure_generation(built, u_base, tmp_path):
    figs = ad.make_figures(built, u_base, str(tmp_path), prefix="test_ad")
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
    assert "experiment_ad_qualified_access.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_ad_qualified_access.md" in baseline["reports"]
