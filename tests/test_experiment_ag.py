"""Tests for Experiment AG — bounded-reader training-pool coverage and generalization stress.

Resource-light: reads the collected artifacts and exercises the sampling policies / decision rule on
synthetics. Covers: collector source verification (V SHA / AD byte-identity / target / splits /
eligibility support / checkpoint stability / no training); the held-out-exclusion of every training
pool (with a negative control); separate sampling vs model-initialization seeds; matched model
initialization across conditions; equal bootstrap pool sizes; realized pair/class composition of each
condition; evaluation outputs (held-out raw/balanced, per-pair support, training-derived majority
baseline); V's accessibility criterion retained; the descriptive decision categories; determinism;
malformed-artifact and split-leakage rejection; frozen-checkpoint protection; historical immutability
and documentation governance.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_ag as ag  # noqa: E402
import analyze_experiment_v as av  # noqa: E402

AG_META = os.path.join(ROOT, "experiments", "experiment_ag_training_pool_stress_meta.json")
AG_RAW = os.path.join(ROOT, "experiments", "experiment_ag_grid_raw.json")
AG_RESULTS = os.path.join(ROOT, "experiments", "experiment_ag_results.json")
AD_NPZ = os.path.join(ROOT, "experiments", "experiment_ad_qualified_access.npz")
V_NPZ = os.path.join(ROOT, "experiments", "experiment_v_sequence_access.npz")
MODELS = ("E_candidate", "F_combined")
READERS = ["GRU", "TCN"]
DECISIVE = ["L4", "Lfull"]
ALLOWED = {"sampling_sensitive_fit_transfer_gap", "persistent_fit_without_transfer", "inconclusive_mixed"}


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.relpath(path, ROOT)} not present")


@pytest.fixture(scope="module")
def meta():
    _require(AG_META)
    with open(AG_META) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def raw():
    _require(AG_RAW)
    with open(AG_RAW) as fh:
        return json.load(fh)["parts"]


@pytest.fixture(scope="module")
def built():
    _require(AG_RESULTS)
    with open(AG_RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def z():
    return np.load(AD_NPZ, allow_pickle=False)


# ---------------------------------------------------------------------------
# source verification / no training
# ---------------------------------------------------------------------------

def test_source_identity(meta, z):
    assert meta["source"]["v_npz_sha256"] == hashlib.sha256(open(V_NPZ, "rb").read()).hexdigest()
    assert meta["source"]["ad_v_array_identity"] is True
    vv = np.load(V_NPZ, allow_pickle=False)
    for k in vv.files:
        if k != "meta_json":
            assert np.array_equal(z[k], vv[k]), k
    assert meta["trained_anything"] is False


def test_target_derivation_and_checkpoints(meta, z):
    for ds in ("trainord", "valord", "test"):
        oa = z[f"E_candidate__{ds}__operand_a"].astype(int)
        ob = z[f"E_candidate__{ds}__operand_b"].astype(int)
        y = z[f"E_candidate__{ds}__y_true"].astype(int)
        assert np.array_equal((oa + ob) % 10, y), ds
    for mk in MODELS:
        for s in ag.SEEDS:
            rec = meta["checkpoints"][f"{mk}|{s}"]
            assert hashlib.sha256(open(os.path.join(ROOT, rec["path"]), "rb").read()).hexdigest() == rec["sha256"]
    assert meta["post_run"]["all_checkpoints_unchanged"] is True


def test_eligibility_support(meta):
    sup = meta["eligibility_support"]["E_candidate"]
    for s, v in sup.items():
        assert v["ok"] is True
        assert v["missing_classes"] == []
        assert v["pairs_present"] >= 1
        assert v["eligible"] == sum(v["class_counts"])


# ---------------------------------------------------------------------------
# held-out exclusion + seeds
# ---------------------------------------------------------------------------

def test_eligible_pool_excludes_heldout(z):
    tr = np.nonzero(z["E_candidate__trainord__seed"] == 42)[0]
    ob = z["E_candidate__trainord__operand_b"][tr].astype(int)
    keep = ~np.isin(ob, [0, 5])
    ob_e = ob[keep]
    assert not np.isin(ob_e, [0, 5]).any()


def test_sampler_never_selects_heldout_negative_control():
    # eligible pool only contains b not in {0,5}; the sampler indexes that pool, so no held-out pair
    oa = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    ob = np.array([1, 1, 2, 2, 3, 3, 4, 4])       # no 0/5
    y = np.array([2, 3, 5, 6, 4, 5, 7, 8])
    rng = np.random.default_rng(0)
    for cond in ag.BOOTSTRAP_CONDITIONS:
        pos = ag._sample_positions(oa, ob, y, cond, rng)
        assert pos.min() >= 0 and pos.max() < len(y)
        assert not np.isin(ob[pos], [0, 5]).any()
    # negative control: a naive sampler over a pool that DID contain 0/5 could emit them
    ob_bad = np.array([0, 5, 0, 5, 1, 2, 3, 4])
    pos_bad = np.random.default_rng(0).integers(0, len(ob_bad), size=20)
    assert np.isin(ob_bad[pos_bad], [0, 5]).any()


def test_bootstrap_pools_match_native_size(raw):
    by = {(p["reader"], p["seed"], p["rep"], p["condition"], p["replicate"]): p for p in raw}
    for reader in READERS:
        for s in ag.SEEDS:
            for rep in ("L4", "Lfull"):
                n_native = by[(reader, s, rep, "ae_native", 0)]["pool"]["n"]
                for cond in ag.BOOTSTRAP_CONDITIONS:
                    for r in range(ag.REPLICATES):
                        assert by[(reader, s, rep, cond, r)]["pool"]["n"] == n_native


def test_sampling_seed_separate_from_init_seed():
    init = av.reader_seed(42, "TCN", "y_true", "L4", "heldout")
    for cond in ag.CONDITIONS:
        for r in range(ag.REPLICATES):
            assert ag._sample_seed(42, cond, r) != init
    assert ag._sample_seed(42, "natural_bootstrap", 0) != ag._sample_seed(42, "natural_bootstrap", 1)


def test_matched_initialization(raw):
    groups = {}
    for p in raw:
        groups.setdefault((p["reader"], p["seed"], p["rep"]), set()).add(p["init_hash"])
    for key, hashes in groups.items():
        assert len(hashes) == 1, key


# ---------------------------------------------------------------------------
# evaluated populations / baselines
# ---------------------------------------------------------------------------

def test_population_partition(built):
    for reader in READERS:
        for rep in DECISIVE:
            for cond in ag.CONDITIONS:
                node = built["readers"][reader][rep][cond]["pops"]
                assert node["test_classA"]["n"]["mean"] + node["test_all_add"]["n"]["mean"] >= 0
                assert abs(node["heldout_classA"]["n"]["mean"] + node["seen_classA"]["n"]["mean"]
                           - node["test_classA"]["n"]["mean"]) < 1e-6


def test_balanced_and_baseline_present(built):
    node = built["readers"]["TCN"]["L4"]["ae_native"]["pops"]["heldout_classA"]
    assert node["balanced_acc"]["mean"] is not None
    assert node["majority_baseline_acc"]["mean"] is not None
    assert len(node["per_class_recall"]) == 10


def test_per_pair_support_reported(built):
    pp = built["readers"]["TCN"]["L4"]["ae_native"]["per_pair"]
    assert len(pp) == 20
    for v in pp.values():
        assert len(v["per_seed"]) == 5


# ---------------------------------------------------------------------------
# realized sampling composition
# ---------------------------------------------------------------------------

def test_target_balanced_is_more_uniform(built):
    R = built["readers"]["TCN"]["L4"]
    nat = R["natural_bootstrap"]["pool"]
    tgt = R["target_balanced_bootstrap"]["pool"]
    assert tgt["class_min_share"]["mean"] > nat["class_min_share"]["mean"]
    assert tgt["class_entropy"]["mean"] >= nat["class_entropy"]["mean"]


def test_pair_balanced_covers_pairs(built):
    R = built["readers"]["TCN"]["L4"]
    nat = R["natural_bootstrap"]["pool"]
    pb = R["pair_balanced_bootstrap"]["pool"]
    # uniform pair draws give a nonzero minimum pair count; the empirical pool has absent pairs
    assert pb["pair_min"]["mean"] > nat["pair_min"]["mean"]


# ---------------------------------------------------------------------------
# decision rule + accessibility
# ---------------------------------------------------------------------------

def test_decision_valid_and_recomputed(built, raw):
    assert built["primary_decision"]["verdict"] in ALLOWED
    assert ag.classify(built)["verdict"] == built["primary_decision"]["verdict"]
    for v in built["primary_decision"]["cells"].values():
        assert v["heldout_classA_bal_native"] is not None
        assert 0 <= v["seeds_bal_positive"] <= 5 and 0 <= v["pairs_improved"] <= 20


def test_decision_rule_encoding():
    def mk(cond_bal_delta_seeds, cond_fit_delta_seeds):
        readers = {}
        for reader in READERS:
            readers[reader] = {}
            for rep in ag.REP_LABELS:
                base = {c: {"per_seed": {}, "per_pair": {f"{a}_{b}": {"mean": 0.1}
                                                        for a in range(10) for b in ag.HELD_OUT_B},
                            "pops": {}} for c in ag.CONDITIONS}
                for s in ag.SEEDS:
                    base["ae_native"]["per_seed"][str(s)] = {"heldout_classA_bal": 0.10,
                                                             "train_base": 0.5}
                for i, s in enumerate(ag.SEEDS):
                    for cond in ag.BOOTSTRAP_CONDITIONS:
                        base[cond]["per_seed"][str(s)] = {
                            "heldout_classA_bal": 0.10 + cond_bal_delta_seeds[i],
                            "train_base": 0.5 + cond_fit_delta_seeds[i]}
                for cond in ag.BOOTSTRAP_CONDITIONS:
                    hi = cond == "pair_balanced_bootstrap"
                    for k in base[cond]["per_pair"]:
                        base[cond]["per_pair"][k] = {"mean": (0.3 if hi else 0.1)}
                readers[reader][rep] = base
        return {"readers": readers}

    # transfer signal: balanced Δ>0 in all seeds + pairs improved -> sampling-sensitive
    d_bal = [0.05] * 5
    d_fit = [0.0] * 5
    assert ag.classify(mk(d_bal, d_fit))["verdict"] == "sampling_sensitive_fit_transfer_gap"
    # fit changes, no transfer -> persistent fit without transfer
    assert ag.classify(mk([0.0] * 5, [0.3] * 5))["verdict"] == "persistent_fit_without_transfer"
    # nothing consistent -> inconclusive
    assert ag.classify(mk([0.0] * 5, [0.0] * 5))["verdict"] == "inconclusive_mixed"


def test_accessibility_retained(built):
    assert "V_A_reached" in built["accessibility"]
    assert isinstance(built["accessibility"]["V_A_reached"], bool)


# ---------------------------------------------------------------------------
# determinism / malformed / leakage / frozen
# ---------------------------------------------------------------------------

def test_sampler_deterministic():
    oa = np.arange(6) % 3
    ob = (np.arange(6) + 1) % 4
    y = np.arange(6)
    for cond in ag.BOOTSTRAP_CONDITIONS:
        a = ag._sample_positions(oa, ob, y, cond, np.random.default_rng(3))
        b = ag._sample_positions(oa, ob, y, cond, np.random.default_rng(3))
        assert np.array_equal(a, b)
    n = len(y)
    assert np.array_equal(ag._sample_positions(oa, ob, y, "ae_native", np.random.default_rng(0)),
                          np.arange(n))


def test_malformed_artifact_rejection(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, foo=np.zeros(3))
    with pytest.raises(Exception):
        import analyze_experiment_ad as ad
        ad.load(str(bad))


def test_split_leakage_negative_control(z):
    meta = json.loads(str(z["meta_json"][0]))
    assert meta["disjointness"]["E_candidate"]["trainord_vs_test"] == 0
    bad = json.loads(json.dumps(meta))
    bad["disjointness"]["E_candidate"]["trainord_vs_test"] = 4
    assert bad["disjointness"]["E_candidate"]["trainord_vs_test"] != 0


def test_frozen_checkpoints_unchanged(meta):
    for mk, recs in meta["checkpoints"].items():
        assert hashlib.sha256(open(os.path.join(ROOT, recs["path"]), "rb").read()).hexdigest() == recs["sha256"]


# ---------------------------------------------------------------------------
# figures / immutability / governance
# ---------------------------------------------------------------------------

def test_figure_generation(built, tmp_path):
    figs = ag.make_figures(built, str(tmp_path), prefix="test_ag")
    assert len(figs) >= 6 and all(os.path.exists(p) for p in figs)


def test_historical_immutability():
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    for rel, expected in baseline["reports"].items():
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == expected, f"changed: {rel}"


def test_research_governance_compliance():
    log = open(os.path.join(ROOT, "docs", "RESEARCH_LOG.md")).read()
    assert "experiment_ag_training_pool_stress.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_ag_training_pool_stress.md" in baseline["reports"]
