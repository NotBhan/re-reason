"""Tests for Experiment AH — bounded-reader capacity / inductive-bias sensitivity at V's scale.

Resource-light: reads the collected AH artifacts (pre-registration meta, 100-fit raw grid, and
aggregated results) and exercises the pre-registered ladder cap and the descriptive H1-H4 rule on
synthetics. Covers: V/AD source identity (V SHA-256, AD==V byte-identity, no training) and frozen
E/F checkpoint stability; the pre-registered 5-level width ladder (exact parameter counts from V's
formulas and the implemented readers, every level within 4.0x the baseline, baseline reproducing
V's deterministic width selection) with an over-cap negative control; raw-grid completeness over
all (reader, level, history, seed) cells (finite fits, eligible-pool sizes, population partitions);
the AE heldout-variant baseline bridge (max |delta| == 0.0, matching init hash and parameter
count); the aggregated population/level structure and per-seed mean consistency; paired per-seed
deltas against the baseline width; the AH-B capacity-insensitive fit-without-transfer verdict with
V's retained V-B accessibility axis; the 6 decision figures; and synthetic exercises of all four
H1-H4 branches.
"""

import hashlib
import itertools
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_ah as ah  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import run_experiment_ah as rah  # noqa: E402

AH_META = os.path.join(ROOT, "experiments", "experiment_ah_capacity_ladder_meta.json")
AH_RAW = os.path.join(ROOT, "experiments", "experiment_ah_grid_raw.json")
AH_RESULTS = os.path.join(ROOT, "experiments", "experiment_ah_results.json")
AD_NPZ = os.path.join(ROOT, "experiments", "experiment_ad_qualified_access.npz")
AD_META = os.path.join(ROOT, "experiments", "experiment_ad_qualified_access_meta.json")
V_NPZ = os.path.join(ROOT, "experiments", "experiment_v_sequence_access.npz")
AE_RAW = os.path.join(ROOT, "experiments", "experiment_ae_grid_raw.json")
U_RESULTS = os.path.join(ROOT, "experiments", "experiment_u_results.json")
FIG_DIR = os.path.join(ROOT, "docs", "reports", "figures")

MODELS = ("E_candidate", "F_combined")
READERS = ah.READERS
HISTORIES = ah.HISTORIES
LEVELS = ah.LEVEL_KEYS
POP_KEYS = ah.POP_KEYS
POP_FIELDS = {"n", "raw_acc", "balanced_acc", "per_class_recall", "majority_class",
              "majority_baseline_acc", "label_counts", "chance"}
FIGURES = ["experiment_ah_width_vs_fit.png", "experiment_ah_width_vs_heldout_bal.png",
           "experiment_ah_paired_deltas.png", "experiment_ah_per_seed_heldout.png",
           "experiment_ah_param_vs_acc.png", "experiment_ah_decision_summary.png"]


def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _require(path: str) -> None:
    if not os.path.exists(path):
        pytest.skip(f"{os.path.relpath(path, ROOT)} not present")


@pytest.fixture(scope="module")
def meta():
    _require(AH_META)
    with open(AH_META) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def raw():
    _require(AH_RAW)
    with open(AH_RAW) as fh:
        return json.load(fh)["parts"]


@pytest.fixture(scope="module")
def built():
    _require(AH_RESULTS)
    with open(AH_RESULTS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def v_arch():
    return av.architecture()


# ---------------------------------------------------------------------------
# source identity / frozen checkpoints
# ---------------------------------------------------------------------------

def test_source_identity(meta):
    assert meta["trained_anything"] is False
    assert meta["invariants_ok"] is True
    assert meta["held_out_b"] == list(ah.HELD_OUT_B)
    assert meta["protocol_matches_v"] == {"readers": READERS, "histories": HISTORIES,
                                          "seeds": ah.SEEDS}
    assert meta["source"]["v_npz_sha256"] == _sha256(V_NPZ)
    assert meta["source"]["ad_npz_sha256"] == _sha256(AD_NPZ)
    assert meta["source"]["ad_meta_sha256"] == _sha256(AD_META)
    assert meta["post_run"]["v_npz_sha256"] == meta["source"]["v_npz_sha256"]
    assert meta["post_run"]["ad_npz_sha256"] == meta["source"]["ad_npz_sha256"]
    # AD's artifact records the same V source hash, and the arrays are byte-identical
    with open(AD_META) as fh:
        ad_meta = json.load(fh)
    assert ad_meta["source"]["v_sha256"] == meta["source"]["v_npz_sha256"]
    assert meta["source"]["ad_v_array_identity"] is True
    ad = np.load(AD_NPZ, allow_pickle=False)
    vv = np.load(V_NPZ, allow_pickle=False)
    keys = [k for k in vv.files if k != "meta_json"]
    assert keys and all(k in ad.files for k in keys)
    for k in keys:
        assert np.array_equal(ad[k], vv[k]), k
    assert meta["source"]["array_sha256_verified"] == len(keys) == len(ad_meta["identity"])


def test_checkpoints_unchanged(meta):
    assert len(meta["checkpoints"]) == len(MODELS) * len(ah.SEEDS)
    for mk in MODELS:
        for s in ah.SEEDS:
            rec = meta["checkpoints"][f"{mk}|{s}"]
            path = os.path.join(ROOT, rec["path"])
            assert os.path.exists(path)
            assert _sha256(path) == rec["sha256"]
            assert rec["ad_recorded_path"] == rec["path"]
            assert rec["ad_recorded_sha256"] == rec["sha256"]
    assert meta["post_run"]["all_checkpoints_unchanged"] is True


# ---------------------------------------------------------------------------
# pre-registered capacity ladder
# ---------------------------------------------------------------------------

def test_ladder_specification(meta, built, v_arch):
    ladder = meta["capacity_ladder"]
    assert ladder["n_levels"] == 5 == len(ladder["levels"])
    assert [r["level_key"] for r in ladder["levels"]] == LEVELS
    assert ([(r["level_key"], r["level_name"], r["gru_hidden"], r["tcn_channels"])
             for r in ladder["levels"]]
            == [tuple(l) for l in rah.LEVELS] == [tuple(l) for l in ah.LEVELS])
    assert ladder["cap_ratio"] == rah.CAP_RATIO == 4.0
    assert ladder["u_mlp128_budget"] == int(v_arch["budget_u_mlp128"])
    base = ladder["levels"][LEVELS.index(ah.BASELINE_KEY)]
    for rec in ladder["levels"]:
        # exact parameter counts from V's formulas and from the implemented readers
        assert rec["gru_params"] == av.gru_param_count(rec["gru_hidden"])
        assert rec["tcn_params"] == av.tcn_param_count(rec["tcn_channels"])
        assert rec["gru_params_max_nclass"] == av.gru_param_count(rec["gru_hidden"],
                                                                  nclass=av.MAX_NCLASS)
        assert rec["tcn_params_max_nclass"] == av.tcn_param_count(rec["tcn_channels"],
                                                                  nclass=av.MAX_NCLASS)
        level = ah.LEVEL_BY_KEY[rec["level_key"]]
        assert ah._param_count("GRU", level) == rec["gru_params"]
        assert ah._param_count("TCN", level) == rec["tcn_params"]
        # every level sits within the declared 4.0x-baseline cap under both class counts
        assert rec["gru_ratio_vs_baseline"] == pytest.approx(
            rec["gru_params"] / base["gru_params"], rel=1e-12)
        assert rec["tcn_ratio_vs_baseline"] == pytest.approx(
            rec["tcn_params"] / base["tcn_params"], rel=1e-12)
        assert rec["gru_ratio_vs_baseline_max_nclass"] == pytest.approx(
            rec["gru_params_max_nclass"] / base["gru_params_max_nclass"], rel=1e-12)
        assert rec["tcn_ratio_vs_baseline_max_nclass"] == pytest.approx(
            rec["tcn_params_max_nclass"] / base["tcn_params_max_nclass"], rel=1e-12)
        for key in ("gru_ratio_vs_baseline", "tcn_ratio_vs_baseline",
                    "gru_ratio_vs_baseline_max_nclass", "tcn_ratio_vs_baseline_max_nclass"):
            assert rec[key] <= ladder["cap_ratio"] + 1e-12, (rec["level_key"], key)
        assert rec["within_cap"] is True and rec["within_cap_max_nclass"] is True
    assert ladder["all_levels_within_cap"] is True
    assert ladder["max_ratio_vs_baseline"] == max(
        max(r["gru_ratio_vs_baseline"], r["tcn_ratio_vs_baseline"]) for r in ladder["levels"])
    assert ladder["max_ratio_vs_baseline"] <= ladder["cap_ratio"]
    # the baseline level reproduces V's deterministic width selection
    assert ladder["baseline_reproduces_v_architecture"] is True
    assert (base["gru_hidden"], base["tcn_channels"]) == (
        int(v_arch["GRU"]["hidden"]), int(v_arch["TCN"]["channels"]))
    assert base["gru_params"] == int(v_arch["GRU"]["params_primary"])
    assert base["tcn_params"] == int(v_arch["TCN"]["params_primary"])
    # the runner's own recomputation agrees with the recorded ladder
    assert rah.verify_capacity_ladder() == ladder
    # the aggregated results carry the same ladder
    assert [(l["level_key"], l["gru_params"], l["tcn_params"]) for l in built["levels"]] == [
        (r["level_key"], r["gru_params"], r["tcn_params"]) for r in ladder["levels"]]


# ---------------------------------------------------------------------------
# raw grid
# ---------------------------------------------------------------------------

def test_raw_grid_completeness(raw, meta):
    assert len(raw) == 100 == meta["grid"]["n_fits"]
    assert meta["grid"]["n_widths"] == len(LEVELS) and meta["grid"]["n_seeds"] == len(ah.SEEDS)
    assert meta["grid"]["definition"].startswith("2 readers x 5 widths x 2 histories")
    combos = {(p["reader"], p["level_key"], p["rep"], p["seed"]) for p in raw}
    assert combos == set(itertools.product(READERS, LEVELS, HISTORIES, ah.SEEDS))
    counts = {r["level_key"]: r for r in meta["capacity_ladder"]["levels"]}
    eligible = meta["eligible_pool"]["models"]
    assert meta["eligible_pool"]["expected_size_range"] == [741, 781]
    assert meta["eligible_pool"]["e_f_eligible_pool_identical"] == {str(s): True
                                                                    for s in ah.SEEDS}
    for s in ah.SEEDS:
        for mk in MODELS:
            rec = eligible[mk][str(s)]
            assert rec["in_range"] is True
            assert rec["heldout_excluded"] + rec["eligible"] == rec["trainord_rows"]
    for p in raw:
        assert p["finite"] is True
        assert 0 < p["best_val_update"] <= ah.UPDATES
        assert 741 <= p["n_train"] <= 781
        assert p["n_train"] == eligible["E_candidate"][str(p["seed"])]["eligible"]
        assert p["n_train"] == eligible["F_combined"][str(p["seed"])]["eligible"]
        level = ah.LEVEL_BY_KEY[p["level_key"]]
        assert (p["gru_hidden"], p["tcn_channels"]) == (level[2], level[3])
        assert p["level_name"] == level[1]
        rec = counts[p["level_key"]]
        assert p["params"] == (rec["gru_params"] if p["reader"] == "GRU" else rec["tcn_params"])
        pop = p["pops"]
        assert set(pop) == set(POP_KEYS)
        assert pop["train_fit"]["n"] == p["n_train"]
        assert pop["heldout_classA"]["n"] + pop["seen_classA"]["n"] == pop["test_classA"]["n"]
        assert pop["heldout_all"]["n"] + pop["seen_all"]["n"] == pop["test_all_add"]["n"]
        assert len(p["per_pair"]) == len(ah.PAIRS_HELDOUT)


# ---------------------------------------------------------------------------
# AE baseline bridge
# ---------------------------------------------------------------------------

def test_ae_baseline_bridge(built, raw):
    bridge = built["ae_bridge"]
    assert bridge["available"] is True
    assert bridge["max_abs_diff_train_fit"] == 0.0
    assert bridge["max_abs_diff_classA"] == 0.0
    assert bridge["max_abs_diff_heldout_classA"] == 0.0
    assert bridge["max_abs_diff_best_val_acc"] == 0.0
    assert bridge["worst"] is None
    _require(AE_RAW)
    recomputed = ah.ae_bridge(built, AE_RAW)
    assert recomputed["available"] is True
    for key in ("max_abs_diff_train_fit", "max_abs_diff_classA",
                "max_abs_diff_heldout_classA", "max_abs_diff_best_val_acc"):
        assert recomputed[key] == 0.0, key
    # direct per-fit bridge: the baseline level re-runs AE's heldout-variant fits exactly
    with open(AE_RAW) as fh:
        ae = json.load(fh)["parts"]
    ref = {(p["seed"], p["reader"], p["rep"]): p for p in ae
           if p["variant"] == "heldout" and p["rep"] in HISTORIES}
    base = {(p["seed"], p["reader"], p["rep"]): p for p in raw
            if p["level_key"] == ah.BASELINE_KEY}
    assert set(ref) == set(base)
    worst = 0.0
    for key, p in base.items():
        r = ref[key]
        assert p["init_hash"] == r["init_hash"], key
        assert p["params"] == r["params"], key
        worst = max(worst,
                    abs(p["pops"]["train_fit"]["raw_acc"] - r["pops"]["train"]["raw_acc"]),
                    abs(p["pops"]["test_classA"]["raw_acc"] - r["pops"]["test_classA"]["raw_acc"]),
                    abs(p["pops"]["heldout_classA"]["raw_acc"]
                        - r["pops"]["test_heldout_classA"]["raw_acc"]),
                    abs(p["best_val_acc"] - r["best_val_acc"]))
    assert worst == 0.0


# ---------------------------------------------------------------------------
# aggregated results / paired deltas
# ---------------------------------------------------------------------------

def test_aggregated_results(built, raw):
    assert set(built["readers"]) == set(READERS)
    assert built["metadata"]["total_fits"] == 100
    assert built["metadata"]["baseline_level"] == ah.BASELINE_KEY
    assert built["metadata"]["held_out_b"] == list(ah.HELD_OUT_B)
    assert built["metadata"]["held_out_pairs"] == len(ah.PAIRS_HELDOUT)
    assert built["metadata"]["thresholds"] == {
        "material_gain": ah.MATERIAL_GAIN, "train_fit_bar": ah.TRAIN_FIT_BAR,
        "heldout_weak": ah.HELDOUT_WEAK, "heldout_material": ah.HELDOUT_MATERIAL,
        "recover_bar": ah.RECOVER_BAR, "bayes_gap_bar": ah.BAYES_GAP_BAR,
        "consistent_seeds": ah.CONSISTENT_SEEDS}
    for reader in READERS:
        for rep in HISTORIES:
            group = built["readers"][reader][rep]
            assert set(group) == set(LEVELS)
            for lk, node in group.items():
                assert node["level_name"] == ah.LEVEL_BY_KEY[lk][1]
                assert node["params"] == ah._param_count(reader, ah.LEVEL_BY_KEY[lk])
                assert len(node["init_hashes"]) == len(ah.SEEDS)
                assert node["all_finite"] is True
                assert 741 <= node["n_train"]["mean"] <= 781
                for key in POP_KEYS:
                    pop = node["pops"][key]
                    assert pop and set(pop) == POP_FIELDS, (reader, rep, lk, key)
                    assert len(pop["raw_acc"]["per_seed"]) == len(ah.SEEDS)
                    assert pop["raw_acc"]["mean"] == pytest.approx(
                        float(np.mean(pop["raw_acc"]["per_seed"])), abs=1e-12)
                    assert pop["balanced_acc"]["mean"] == pytest.approx(
                        float(np.mean(pop["balanced_acc"]["per_seed"])), abs=1e-12)
                assert len(node["per_pair"]) == len(ah.PAIRS_HELDOUT)
    # the aggregation is reproducible from the raw grid
    assert ah.aggregate(raw) == built["readers"]


def test_paired_deltas(built, raw):
    P = built["paired_deltas"]
    assert set(P) == set(READERS)
    assert ah.paired_deltas(built) == P
    by = {}
    for p in raw:
        by.setdefault(p["level_key"], {})[(p["reader"], p["rep"], p["seed"])] = p
    base = by[ah.BASELINE_KEY]
    for reader in READERS:
        for rep in HISTORIES:
            assert set(P[reader][rep]) == set(LEVELS)
            for lk in LEVELS:
                node = P[reader][rep][lk]
                d_bal = node["delta_heldout_classA_bal"]["per_seed"]
                d_fit = node["delta_train_fit_raw"]["per_seed"]
                assert len(d_bal) == len(d_fit) == len(ah.SEEDS)
                if lk == ah.BASELINE_KEY:
                    assert d_bal == [0.0] * len(ah.SEEDS)
                    assert d_fit == [0.0] * len(ah.SEEDS)
                # per-seed paired deltas are exactly level - baseline
                for i, s in enumerate(ah.SEEDS):
                    lvl = by[lk][(reader, rep, s)]
                    bse = base[(reader, rep, s)]
                    assert d_bal[i] == pytest.approx(
                        lvl["pops"]["heldout_classA"]["balanced_acc"]
                        - bse["pops"]["heldout_classA"]["balanced_acc"], abs=1e-12)
                    assert d_fit[i] == pytest.approx(
                        lvl["pops"]["train_fit"]["raw_acc"]
                        - bse["pops"]["train_fit"]["raw_acc"], abs=1e-12)
                assert node["seeds_bal_material_gain"] == sum(
                    1 for d in d_bal if d >= ah.MATERIAL_GAIN)
                assert node["seeds_fit_material_gain"] == sum(
                    1 for d in d_fit if d >= ah.MATERIAL_GAIN)
                assert node["delta_heldout_classA_bal"]["mean"] == pytest.approx(
                    float(np.mean(d_bal)), abs=1e-12)
                assert node["delta_train_fit_raw"]["mean"] == pytest.approx(
                    float(np.mean(d_fit)), abs=1e-12)


# ---------------------------------------------------------------------------
# decision + accessibility
# ---------------------------------------------------------------------------

def test_classification_decision(built):
    d = built["primary_decision"]
    assert d["verdict"] == "H2_fit_without_transfer"
    assert d["framing"].startswith("AH-B")
    assert d["h1_cells"] == []
    assert d["fit_cells"] and all(v["heldout_weak"] for k, v in d["cells"].items()
                                  if k in d["fit_cells"])
    assert set(d["cells"]) == {f"{r}|{rep}|{lk}" for r in READERS for rep in HISTORIES
                               for lk in LEVELS if lk != ah.BASELINE_KEY}
    for key, v in d["cells"].items():
        assert v["transfer_signal"] == (v["seeds_bal_material_gain"] >= ah.CONSISTENT_SEEDS
                                        and v["heldout_classA_bal"] >= ah.HELDOUT_MATERIAL), key
        assert v["fit_present"] == (v["train_fit"] >= ah.TRAIN_FIT_BAR
                                    or v["seeds_fit_material_gain"] >= ah.CONSISTENT_SEEDS), key
        assert v["heldout_weak"] == (v["heldout_classA_bal"] <= ah.HELDOUT_WEAK), key
    # the baseline level's own fit flag agrees with the aggregated training accuracy
    for reader in READERS:
        for rep in HISTORIES:
            train = built["readers"][reader][rep][ah.BASELINE_KEY][
                "pops"]["train_fit"]["raw_acc"]["mean"]
            assert d["baseline_fit"][f"{reader}|{rep}"] == (train >= ah.TRAIN_FIT_BAR)
    assert d["thresholds"] == {"material_gain": ah.MATERIAL_GAIN, "train_fit_bar": ah.TRAIN_FIT_BAR,
                               "heldout_weak": ah.HELDOUT_WEAK,
                               "heldout_material": ah.HELDOUT_MATERIAL,
                               "consistent_seeds": ah.CONSISTENT_SEEDS}
    # deterministic re-evaluation of the descriptive rule
    assert ah.classify(built) == d
    # V's accessibility axis is retained and separate from the capacity verdict
    acc = built["accessibility"]
    assert acc["verdict"] == "V-B_sequence_limited"
    assert acc["conditions"] == {"V-A": False, "V-B": True}
    assert len(acc["per_cell"]) == len(READERS) * len(HISTORIES) * len(LEVELS)
    assert acc["best"] in acc["per_cell"]
    if os.path.exists(U_RESULTS):
        recomputed = ah.accessibility(built, av.load_u_baseline(U_RESULTS))
        assert recomputed["verdict"] == acc["verdict"]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def test_figures_exist():
    for name in FIGURES:
        path = os.path.join(FIG_DIR, name)
        assert os.path.exists(path), path
        assert os.path.getsize(path) > 0, path


# ---------------------------------------------------------------------------
# synthetic decision-rule coverage / cap negative control
# ---------------------------------------------------------------------------

def _synth(bal, train, bal_gain, fit_gain):
    """Synthetic classify() input: per-level held-out balanced/train accuracies and seed counts."""
    n = len(ah.SEEDS)
    readers = {}
    for reader in READERS:
        readers[reader] = {}
        for rep in HISTORIES:
            readers[reader][rep] = {
                lk: {"level_name": ah.LEVEL_BY_KEY[lk][1], "params": 0,
                     "pops": {"heldout_classA": {"balanced_acc": {"mean": bal[lk],
                                                                  "per_seed": [bal[lk]] * n}},
                              "train_fit": {"raw_acc": {"mean": train[lk],
                                                        "per_seed": [train[lk]] * n}}}}
                for lk in LEVELS}
    deltas = {reader: {rep: {
        lk: {"delta_heldout_classA_bal": {"mean": 0.0, "per_seed": [0.0] * n},
             "delta_train_fit_raw": {"mean": 0.0, "per_seed": [0.0] * n},
             "seeds_bal_material_gain": bal_gain[lk],
             "seeds_fit_material_gain": fit_gain[lk]}
        for lk in LEVELS} for rep in HISTORIES} for reader in READERS}
    return {"readers": readers, "paired_deltas": deltas}


def test_classification_synthetic():
    weak_bal = dict.fromkeys(LEVELS, 0.10)
    no_gain = dict.fromkeys(LEVELS, 0)

    # H1: a width gains >= +0.15 over baseline in >= 4/5 seeds and reaches >= 0.30 balanced
    h1 = ah.classify(_synth({**weak_bal, "large": 0.35}, dict.fromkeys(LEVELS, 0.80),
                            {**no_gain, "large": 4}, no_gain))
    assert h1["verdict"] == "H1_capacity_associated_generalization"
    assert h1["framing"].startswith("AH-A") and h1["h1_cells"]

    # H2: training fit is present while every fitting level stays at/below the weak bar
    h2 = ah.classify(_synth(weak_bal, dict.fromkeys(LEVELS, 0.80), no_gain, no_gain))
    assert h2["verdict"] == "H2_fit_without_transfer"
    assert h2["framing"].startswith("AH-B")

    # H3: no level in the ladder fits the training pool
    h3 = ah.classify(_synth(weak_bal, dict.fromkeys(LEVELS, 0.30), no_gain, no_gain))
    assert h3["verdict"] == "H3_capacity_insensitive_fitting_failure"
    assert h3["fit_cells"] == []

    # H4: a fitting level above the weak bar without a transfer signal is left inconclusive
    h4 = ah.classify(_synth({**weak_bal, "large": 0.26},
                            {**dict.fromkeys(LEVELS, 0.30), "large": 0.80}, no_gain, no_gain))
    assert h4["verdict"] == "H4_mixed_inconclusive"
    assert h4["fit_cells"] and not all(v["heldout_weak"] for k, v in h4["cells"].items()
                                       if k in h4["fit_cells"])


def test_over_cap_negative_control(monkeypatch):
    # the real ladder passes its own cap check
    ladder = rah.verify_capacity_ladder()
    assert ladder["max_ratio_vs_baseline"] <= rah.CAP_RATIO
    assert all(r["within_cap"] and r["within_cap_max_nclass"] for r in ladder["levels"])
    levels = {l[0]: l for l in rah.LEVELS}
    base = rah.param_counts(levels["baseline"][2], levels["baseline"][3])
    assert rah.param_counts(levels["large"][2], levels["large"][3])["gru_params"] \
        / base["gru_params"] <= rah.CAP_RATIO
    # a wider level crosses the 4.0x cap and must be rejected, not silently accepted
    over_width = next(h for h in range(levels["large"][2] + 1, 1000)
                      if rah.param_counts(h, levels["large"][3])["gru_params"]
                      / base["gru_params"] > rah.CAP_RATIO)
    over = list(rah.LEVELS[:-1]) + [("large", "Large", over_width, levels["large"][3])]
    monkeypatch.setattr(rah, "LEVELS", over)
    with pytest.raises(SystemExit) as exc:
        rah.verify_capacity_ladder()
    assert "capacity cap violation" in str(exc.value)
