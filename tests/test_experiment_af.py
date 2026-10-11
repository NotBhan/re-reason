"""Tests for Experiment AF — held-out operand task-difficulty stratification (measurement only).

Resource-light: reads the collected artifacts and exercises the descriptive decision rule on
synthetics. Covers: the collector source verification (V SHA / AD byte-identity / target derivation /
split disjointness / population-mask partition / checkpoint stability / no training); reuse of U's
exact-interface estimator; held-out stratum definitions being confined to b in {0,5}; geometry count
partition identities; per-seed + pooled exact-interface estimates; reproduction of U's published
overall ceilings; held-out subgroup ceilings vs the overall ceiling; E primary + F control; the
descriptive decision categories with negative controls; determinism; malformed-artifact rejection;
frozen-checkpoint protection; historical immutability and documentation governance.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_af as af  # noqa: E402
import analyze_experiment_u as au  # noqa: E402

AF_META = os.path.join(ROOT, "experiments", "experiment_af_task_difficulty_meta.json")
AF_RAW = os.path.join(ROOT, "experiments", "experiment_af_task_difficulty_raw.json")
AF_RESULTS = os.path.join(ROOT, "experiments", "experiment_af_results.json")
AD_NPZ = os.path.join(ROOT, "experiments", "experiment_ad_qualified_access.npz")
V_NPZ = os.path.join(ROOT, "experiments", "experiment_v_sequence_access.npz")
U_RESULTS = os.path.join(ROOT, "experiments", "experiment_u_results.json")
MODELS = ("E_candidate", "F_combined")
POPS = ["all_add", "classA"]
DECISIVE = ["L4", "Lfull"]
REPS = [r for r, _ in af.REPS]
ALLOWED = {"task_side_subgroup_limitation_supported", "task_side_limitation_not_indicated",
           "inconclusive_mixed"}


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.relpath(path, ROOT)} not present")


@pytest.fixture(scope="module")
def meta():
    _require(AF_META)
    with open(AF_META) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def built():
    _require(AF_RESULTS)
    with open(AF_RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def raw():
    _require(AF_RAW)
    with open(AF_RAW) as fh:
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


def test_no_training_and_estimator_reuse(meta, built):
    assert meta["trained_anything"] is False
    assert "analyze_experiment_u" in meta["estimator"]
    # the analysis uses U's estimator directly
    assert af.au.exact_bayes is au.exact_bayes
    assert af.au.history_features is au.history_features
    # no reader accuracies are present anywhere in the AF results
    assert "readers" not in built


def test_target_derivation_and_checkpoints(meta, z):
    for ds in ("trainord", "valord", "test"):
        for mk in MODELS:
            oa = z[f"{mk}__{ds}__operand_a"].astype(int)
            ob = z[f"{mk}__{ds}__operand_b"].astype(int)
            y = z[f"{mk}__{ds}__y_true"].astype(int)
            assert np.array_equal((oa + ob) % 10, y), (mk, ds)
    for mk in MODELS:
        for s in af.SEEDS:
            rec = meta["checkpoints"][f"{mk}|{s}"]
            assert hashlib.sha256(open(os.path.join(ROOT, rec["path"]), "rb").read()).hexdigest() == rec["sha256"]
    assert meta["post_run"]["all_checkpoints_unchanged"] is True


def test_population_partition_and_heldout(meta, z):
    for mk in MODELS:
        for s in map(str, af.SEEDS):
            rec = meta["masks"][mk][s]
            assert rec["classA"] + rec["correct"] == rec["all_add"]
            assert rec["heldout_pairs"] == 20
    # held-out rows are exactly the b in {0,5} rows
    sel = z["E_candidate__test__seed"] == 42
    ob = z["E_candidate__test__operand_b"][sel].astype(int)
    assert meta["masks"]["E_candidate"]["42"]["heldout_rows"] == int(np.isin(ob, [0, 5]).sum())


# ---------------------------------------------------------------------------
# stratum definitions
# ---------------------------------------------------------------------------

def test_group_definitions_confined_to_heldout(z):
    sel = z["E_candidate__test__seed"] == 42
    oa = z["E_candidate__test__operand_a"][sel].astype(int)
    ob = z["E_candidate__test__operand_b"][sel].astype(int)
    held = np.isin(ob, [0, 5])
    for name, pred in af.HELD_OUT_GROUPS:
        gm = np.array([pred(int(a), int(b)) for a, b in zip(oa, ob)], bool) & held
        # every selected row is held out; b0 never has b=5 etc.
        assert held[gm].all()
        if name == "b0":
            assert (ob[gm] == 0).all()
        if name == "b5":
            assert (ob[gm] == 5).all()
        if name == "wrap":
            assert ((oa[gm] + ob[gm]) >= 10).all()


def test_group_definitions_negative_control():
    """A predicate true only on seen operands must yield an empty held-out group (mask is confined)."""
    pred = lambda a, b: b not in (0, 5)          # only seen operands
    oa = np.array([1, 2, 1, 5, 0])
    ob = np.array([3, 7, 0, 5, 0])               # b in {0,5} at indices 2,3,4
    held = np.isin(ob, [0, 5])
    gm = np.array([pred(int(a), int(b)) for a, b in zip(oa, ob)], bool) & held
    assert gm.sum() == 0


# ---------------------------------------------------------------------------
# geometry count partition identities
# ---------------------------------------------------------------------------

def test_geometry_count_partitions(built):
    for pop in POPS:
        g = built["E_candidate"]["geometry"][pop]
        for key in ("overall", "heldout", "seen", "b0", "b5", "wrap", "nowrap", "a_lt5", "a_ge5", "eq", "neq"):
            node = g[key]
            assert len(node["counts"]) == 10
            # counts are pooled over the 5 seeds; n is reported per seed
            assert sum(node["counts"]) == round(node["n"]["mean"] * len(af.SEEDS)), (pop, key)
        for a, b in (("b0", "b5"), ("wrap", "nowrap"), ("a_lt5", "a_ge5"), ("eq", "neq")):
            for i in range(10):
                assert g[a]["counts"][i] + g[b]["counts"][i] == g["heldout"]["counts"][i], (pop, a, b, i)
        for i in range(10):
            assert g["heldout"]["counts"][i] + g["seen"]["counts"][i] == g["overall"]["counts"][i], (pop, i)


def test_heldout_label_geometry_skew(built):
    g = built["E_candidate"]["geometry"]["classA"]
    # equal-operand held-out pairs (0,0) and (5,5) both map to target 0 -> class 0 mass, entropy 0
    assert g["eq"]["entropy"]["mean"] == 0.0
    assert sum(g["eq"]["counts"][1:]) == 0
    assert g["neq"]["counts"][0] == 0


# ---------------------------------------------------------------------------
# exact-interface estimates
# ---------------------------------------------------------------------------

def test_reproduces_u_baseline(built):
    u = json.load(open(U_RESULTS))
    ps = built["pooled_summary"]
    assert abs(ps["classA"]["L4"]["overall"]["bayes"]
               - u["E_candidate"]["pooled"]["classA"]["L4"]["y_true"]["bayes_acc"]) < 1e-3
    assert abs(ps["all_add"]["L4"]["overall"]["bayes"]
               - u["E_candidate"]["pooled"]["all_add"]["L4"]["y_true"]["bayes_acc"]) < 1e-3


def test_heldout_strata_are_identifiable(built):
    for pop in POPS:
        for rep in DECISIVE:
            overall = built["E_candidate"]["identifiability"][pop][rep]["overall"]["y_true__bayes_acc"]["mean"]
            for g in af.HELD_OUT_GROUP_NAMES:
                node = built["E_candidate"]["identifiability"][pop][rep][g]
                assert node["y_true__bayes_acc"]["mean"] >= overall - 0.01, (pop, rep, g)
                assert len(node["n"]["per_seed"]) == 5
                assert len(node["y_true__bayes_acc"]["per_seed"]) == 5


def test_per_seed_and_pooled_and_control(built):
    assert set(built["E_candidate"]["pooled"]) == set(POPS)
    assert "pooled" in built["F_combined"]
    for m in MODELS:
        for pop in POPS:
            for rep in REPS:
                node = built[m]["identifiability"][pop][rep]["overall"]
                assert len(node["y_true__bayes_acc"]["per_seed"]) == 5
    assert set(built["E_candidate"]["per_pair"]) == set(POPS)


def test_per_pair_support_reported(built):
    pp = built["E_candidate"]["per_pair"]["classA"]["L4"]
    assert len(pp) == 20
    for key, node in pp.items():
        assert "n" in node and "y_true__bayes_acc" in node
        assert len(node["n"]["per_seed"]) == 5


# ---------------------------------------------------------------------------
# decision rule + negative controls
# ---------------------------------------------------------------------------

def _mk_res(overall_seeds, group_seeds, n):
    def ms(vals):
        return {"mean": float(np.mean(vals)), "sd": 0.0, "per_seed": list(vals)}
    ident = {pop: {rep: {} for rep in af.REP_LABELS} for pop in POPS}
    for pop in POPS:
        for rep in af.REP_LABELS:
            for g in af.ALL_GROUPS:
                if g == "overall":
                    ident[pop][rep][g] = {"y_true__bayes_acc": ms(overall_seeds), "n": ms([1000] * 5),
                                          "unique_fraction": ms([0.9] * 5)}
                else:
                    ident[pop][rep][g] = {"y_true__bayes_acc": ms(group_seeds), "n": ms([n] * 5),
                                          "unique_fraction": ms([0.9] * 5)}
    return {"identifiability": ident}


def test_decision_supported():
    r = _mk_res([1.0] * 5, [0.80] * 5, n=50)
    assert af.classify(r)["verdict"] == "task_side_subgroup_limitation_supported"


def test_decision_not_indicated():
    r = _mk_res([1.0] * 5, [0.995] * 5, n=50)
    d = af.classify(r)
    assert d["verdict"] == "task_side_limitation_not_indicated" and d["supported_subgroups"] == []


def test_decision_inconclusive_inconsistent():
    r = _mk_res([1.0] * 5, [0.90, 0.90, 0.90, 0.75, 0.75], n=50)   # mean gap 0.16, only 2 seeds
    assert af.classify(r)["verdict"] == "inconclusive_mixed"


def test_decision_inconclusive_low_support():
    r = _mk_res([1.0] * 5, [0.80] * 5, n=5)                          # inadequate support
    assert af.classify(r)["verdict"] == "inconclusive_mixed"


def test_stored_verdict_valid(built):
    assert built["classification"]["verdict"] in ALLOWED
    assert built["control_classification"]["verdict"] in ALLOWED
    assert built["classification"]["supported_subgroups"] == []


# ---------------------------------------------------------------------------
# determinism / malformed / leakage negative control / frozen protection
# ---------------------------------------------------------------------------

def test_deterministic_analysis(z):
    # per-seed analysis and aggregation are deterministic (full build would take ~43s)
    a = af.analyze_seed(z, "E_candidate", 42)
    b = af.analyze_seed(z, "E_candidate", 42)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    ag1 = af.aggregate([a] * len(af.SEEDS))
    ag2 = af.aggregate([b] * len(af.SEEDS))
    assert json.dumps(ag1, sort_keys=True) == json.dumps(ag2, sort_keys=True)
    assert af.classify(ag1)["verdict"] in ALLOWED


def test_malformed_artifact_rejection(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, foo=np.zeros(3))
    with pytest.raises(Exception):
        z = np.load(str(bad), allow_pickle=False)
        json.loads(str(z["meta_json"][0]))


def test_source_invariant_negative_control(z):
    """A validator of the kind run_experiment_af uses must reject a corrupted meta and accept the real one."""
    meta = json.loads(str(z["meta_json"][0]))

    def validate(m):
        problems = []
        for mk in MODELS:
            dj = m.get("disjointness", {}).get(mk, {})
            for k in ("trainord_vs_valord", "trainord_vs_test", "valord_vs_test"):
                if int(dj.get(k, -1)) != 0:
                    problems.append(f"disjointness: {mk}/{k}={dj.get(k)}")
        return problems

    assert validate(meta) == []                      # real meta passes
    bad = json.loads(json.dumps(meta))
    bad["disjointness"]["E_candidate"]["valord_vs_test"] = 7
    assert validate(bad)                             # corrupted meta is rejected


def test_frozen_checkpoints_unchanged(meta):
    for mk, recs in meta["checkpoints"].items():
        path = os.path.join(ROOT, recs["path"])
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == recs["sha256"]


# ---------------------------------------------------------------------------
# figures / immutability / governance
# ---------------------------------------------------------------------------

def test_figure_generation(built, tmp_path):
    figs = af.make_figures(built, str(tmp_path), prefix="test_af")
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
    assert "experiment_af_task_difficulty.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_af_task_difficulty.md" in baseline["reports"]
