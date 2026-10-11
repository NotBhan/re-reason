"""Tests for Experiment AE — baseline-aware fit-transfer diagnostic (AD's AdamW condition).

Resource-light: live fits use tiny synthetic tensors; structural tests read the collected artifacts.
Covers: AD/V source identity recorded by the AE collector; the retained-state inspection (no AD state
dictionaries -> fits reproduced); frozen-checkpoint stability; the reproduction bridge against AD's
retained adamw metrics; target/label coverage and held-out exclusion; population alignment; the
majority-class baseline being derived from training labels only and consistent with the label counts;
balanced-accuracy consistency with per-class recall; per-seed train-to-test and seen-to-heldout gaps;
the 20 held-out operand pairs and per-seed variation; the descriptive fit-transfer decision rule and
its independence from V's accessibility axis; V-A/V-B retained; determinism; malformed-artifact and
split-leakage rejection with negative controls; frozen-model protection; historical immutability and
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

import analyze_experiment_ad as ad  # noqa: E402
import analyze_experiment_ae as ae  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import run_experiment_ae as rae  # noqa: E402
import run_experiment_g as eg  # noqa: E402

torch.set_num_threads(int(os.environ.get("AE_TEST_THREADS", "1")))

AE_META = os.path.join(ROOT, "experiments", "experiment_ae_fit_transfer_meta.json")
AE_RAW = os.path.join(ROOT, "experiments", "experiment_ae_grid_raw.json")
AE_RESULTS = os.path.join(ROOT, "experiments", "experiment_ae_results.json")
AD_NPZ = os.path.join(ROOT, "experiments", "experiment_ad_qualified_access.npz")
AD_RAW = os.path.join(ROOT, "experiments", "experiment_ad_grid_raw.json")
V_NPZ = os.path.join(ROOT, "experiments", "experiment_v_sequence_access.npz")
U_RESULTS = os.path.join(ROOT, "experiments", "experiment_u_results.json")
MODELS = ("E_candidate", "F_combined")
READERS = ["GRU", "TCN"]
REPS = [r for r, _ in ae.REPS]
DECISIVE = ["L4", "Lfull"]
POP_KEYS = ["train", "val", "test_all_add", "test_classA", "test_matched", "test_correct",
            "test_seen_all", "test_heldout_all", "test_seen_classA", "test_heldout_classA"]
ALLOWED = {"fit_without_transfer", "partial_transfer", "inconclusive_mixed"}


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.relpath(path, ROOT)} not present")


@pytest.fixture(scope="module")
def meta():
    _require(AE_META)
    with open(AE_META) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def raw():
    _require(AE_RAW)
    with open(AE_RAW) as fh:
        return json.load(fh)["parts"]


@pytest.fixture(scope="module")
def built():
    _require(AE_RESULTS)
    with open(AE_RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def u_base():
    return av.load_u_baseline(U_RESULTS)


@pytest.fixture(scope="module")
def loaded_model():
    return eg.load_checkpoint(os.path.join(ROOT, "experiments", "checkpoints"),
                              "E_candidate", 42)["model"]


# ---------------------------------------------------------------------------
# source inspection / reproduction plan
# ---------------------------------------------------------------------------

def test_collector_records_reproduction(meta):
    assert meta["reused_retained_states"] is False
    assert meta["inspection"]["has_state_dict"] is False
    assert meta["source"]["ad_v_array_identity"] is True
    assert "reproduce" in meta["reproduction_plan"].lower()


def test_source_hashes(meta):
    assert meta["source"]["ad_npz_sha256"] == hashlib.sha256(open(AD_NPZ, "rb").read()).hexdigest()
    assert meta["source"]["v_npz_sha256"] == hashlib.sha256(open(V_NPZ, "rb").read()).hexdigest()
    assert meta["source"]["v_sha256_recorded_in_ad_meta"] == meta["source"]["v_npz_sha256"]


def test_checkpoints_unchanged(meta):
    for mk in MODELS:
        for s in ae.SEEDS:
            rec = meta["checkpoints"][mk][str(s)]
            assert eg.sha256_file(os.path.join(ROOT, rec["path"])) == rec["sha256"]
            assert rec.get("unchanged_after_run") is True


def test_only_adamw_condition(built):
    assert built["metadata"]["condition"]["name"] == "adamw"
    assert built["metadata"]["condition"]["opt"] == "adamw"
    assert set(built["readers"]) == set(READERS)


# ---------------------------------------------------------------------------
# reproduction bridge against AD
# ---------------------------------------------------------------------------

def test_ad_bridge_zero(built):
    b = built["ad_bridge"]
    assert b["max_abs_diff_train_acc"] == 0.0
    assert b["max_abs_diff_classA"] == 0.0
    assert b["max_abs_diff_heldout_classA"] == 0.0


# ---------------------------------------------------------------------------
# target / labels / populations
# ---------------------------------------------------------------------------

def test_target_and_class_coverage(built):
    td = built["metadata"]["target_derivation"]
    assert td and all(r["exact"] and r["rule"] == "(operand_a + operand_b) mod 10" for r in td.values())


def test_population_alignment(raw):
    for p in raw:
        po = p["pops"]
        assert po["test_classA"]["n"] + po["test_correct"]["n"] == po["test_all_add"]["n"]
        assert po["test_seen_all"]["n"] + po["test_heldout_all"]["n"] == po["test_all_add"]["n"]
        assert po["test_seen_classA"]["n"] + po["test_heldout_classA"]["n"] == po["test_classA"]["n"]
        assert po["train"]["n"] == p["n_train"]


def test_heldout_exclusion_and_label_counts(raw):
    for p in raw:
        if p["variant"] != "heldout":
            continue
        # the fit pool of the heldout variant must exclude the b in {0,5} operand rows
        assert p["n_train"] < 1500
        for key in POP_KEYS:
            node = p["pops"][key]
            if node is None:
                continue
            assert sum(node["label_counts"]) == node["n"]
            assert len(node["label_counts"]) == 10


# ---------------------------------------------------------------------------
# baselines / balanced accuracy
# ---------------------------------------------------------------------------

def test_majority_baseline_from_training_labels(raw):
    for p in raw:
        train = p["pops"]["train"]
        maj = train["majority_class"]
        assert maj == int(np.argmax(train["label_counts"]))
        for key in POP_KEYS:
            node = p["pops"][key]
            if node is None:
                continue
            assert node["majority_class"] == maj, (p["reader"], key)
            expected = node["label_counts"][maj] / node["n"]
            assert abs(node["majority_baseline_acc"] - expected) < 1e-9, (p["reader"], key)


def test_balanced_accuracy_matches_recall(raw):
    for p in raw:
        for key in POP_KEYS:
            node = p["pops"][key]
            if node is None:
                continue
            present = [r for r in node["per_class_recall"] if r is not None]
            if not present:
                continue
            assert abs(node["balanced_acc"] - float(np.mean(present))) < 1e-9


def test_majority_baseline_negative_control():
    """The majority rule must follow the training labels, not the evaluation labels."""
    tr = np.array([0, 0, 0, 0, 1, 1, 2])          # majority = 0
    y_eval = np.array([1, 1, 1, 2])               # evaluation majority = 1
    p = np.zeros_like(y_eval)                     # predicting 0
    s = ae._summarize(p, y_eval, tr)
    assert s["majority_class"] == 0
    assert abs(s["majority_baseline_acc"] - 0.0) < 1e-12


# ---------------------------------------------------------------------------
# per-seed gaps / per-pair / decision
# ---------------------------------------------------------------------------

def test_gaps_and_per_seed(built):
    for reader in READERS:
        for rep in DECISIVE:
            c = built["primary_decision"]["cells"][f"{reader}|{rep}"]
            assert len(c["per_seed_transfer_gain"]) == 5
            assert c["heldout_pairs_above_chance"] <= 20
            assert c["seen_vs_heldout_gap"] is not None
    R = built["readers"]
    for reader in READERS:
        node = R[reader]["Lfull"]["heldout"]["per_pair_heldout"]
        assert len(node) == 20
        for v in node.values():
            assert len(v["per_seed"]) == 5


def test_decision_rule_and_consistency(built, raw, u_base):
    assert built["primary_decision"]["verdict"] in ALLOWED
    recomputed = ae.classify(built, raw)
    assert recomputed["verdict"] == built["primary_decision"]["verdict"]
    # sequence accessibility is a separate axis and is retained
    assert built["accessibility"]["verdict"] in {"V-A_sequence_accessible", "V-B_sequence_limited",
                                                 "V-partial/inconclusive"}


def test_decision_rule_encoding():
    """Synthetic check of the descriptive rule without running fits."""

    def res_for(train, clsA, ho, hbase, pair_hi):
        readers = {}
        for reader in READERS:
            readers[reader] = {}
            for rep in DECISIVE:
                pp = {f"{a}_{b}": {"mean": (0.2 if pair_hi else 0.05)} for a in range(10) for b in (0, 5)}
                readers[reader][rep] = {
                    "main": {"pops": {"train": {"raw_acc": {"mean": train},
                                                "majority_baseline_acc": {"mean": 0.1}},
                                      "test_classA": {"raw_acc": {"mean": clsA}}}},
                    "heldout": {"pops": {"test_heldout_classA": {
                        "raw_acc": {"mean": ho}, "majority_baseline_acc": {"mean": hbase},
                        "per_class_recall": [{"mean": 0.1}] * 10}},
                        "per_pair_heldout": pp}}
        return {"readers": readers}

    def parts_for(delta):
        out = []
        for reader in READERS:
            for s in ae.SEEDS:
                for rep in DECISIVE:
                    for variant in ("main", "heldout"):
                        out.append({
                            "reader": reader, "seed": s, "rep": rep, "variant": variant,
                            "pops": {"test_heldout_classA": {"raw_acc": 0.1 + delta,
                                                             "majority_baseline_acc": 0.1}}})
        return out

    # fit without transfer: high train, held-out at/below baseline
    assert ae.classify(res_for(0.8, 0.1, 0.1, 0.1, False),
                       parts_for(-0.05))["verdict"] == "fit_without_transfer"
    # partial transfer: held-out above baseline in all seeds + pairs above chance
    assert ae.classify(res_for(0.8, 0.5, 0.35, 0.1, True),
                       parts_for(0.25))["verdict"] == "partial_transfer"


def test_deterministic_analysis(built, raw, u_base, meta):
    data = {"meta": json.loads(str(np.load(AD_NPZ, allow_pickle=False)["meta_json"][0])),
            "path": AD_NPZ, "ad_raw": AD_RAW}
    a = ae.assemble(data, raw, u_base, meta)
    b = ae.assemble(data, raw, u_base, meta)
    assert json.dumps(a["readers"], sort_keys=True) == json.dumps(b["readers"], sort_keys=True)
    assert json.dumps(a["primary_decision"], sort_keys=True) == \
        json.dumps(b["primary_decision"], sort_keys=True)


# ---------------------------------------------------------------------------
# malformed artifacts / leakage negative controls
# ---------------------------------------------------------------------------

def test_malformed_artifact_rejection(tmp_path):
    with pytest.raises(Exception):
        ad.load(str(tmp_path / "nope.npz"))
    bad = tmp_path / "bad.npz"
    np.savez(bad, foo=np.zeros(3))
    with pytest.raises(Exception):
        ad.load(str(bad))


def test_split_leakage_rejection_negative_control():
    meta = json.loads(str(np.load(AD_NPZ, allow_pickle=False)["meta_json"][0]))
    assert ad._validate(meta) == []
    meta2 = json.loads(json.dumps(meta))
    meta2["disjointness"]["E_candidate"]["trainord_vs_test"] = 5
    assert any("disjointness" in p for p in ad._validate(meta2))


# ---------------------------------------------------------------------------
# frozen-model protection
# ---------------------------------------------------------------------------

def test_no_reason_grad_or_mutation(loaded_model):
    before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4, av.INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, 10, 16)
    import analyze_experiment_ab as ab
    ab.train_ab("TCN", X, y, X, y, 10, 5, 20, ae.COND, av.architecture())
    for k, v in loaded_model.state_dict().items():
        assert torch.equal(v, before[k])
    assert not any(p.grad is not None for p in loaded_model.parameters())


# ---------------------------------------------------------------------------
# figures / immutability / governance
# ---------------------------------------------------------------------------

def test_figure_generation(built, tmp_path):
    figs = ae.make_figures(built, str(tmp_path), prefix="test_ae")
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
    assert "experiment_ae_fit_transfer.md" in log
    with open(os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")) as fh:
        baseline = json.load(fh)
    assert "docs/reports/experiment_ae_fit_transfer.md" in baseline["reports"]
