"""Tests for the Experiment G analysis pipeline (ADD candidate->commit gap localization).

Covers: deterministic bin merging, the gate-dilution decomposition, the conditional
candidate-correct analysis, the oracle-commit table, the logistic predictor + grouped
cross-validation + bootstrap, seed-level paired statistics, the collector on a real repo
checkpoint, and an end-to-end synthetic run of the full analysis + figure pipeline.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_g as ag  # noqa: E402
import run_experiment_g as rg  # noqa: E402
from run_experiment_g import COLS, SOURCE_IDS  # noqa: E402


def make_row(**kw):
    row = {c: 0.0 for c in COLS}
    row.update(kw)
    return [row[c] for c in COLS]


def table(rows):
    return ag.Table(np.asarray(rows, dtype=float), COLS)


# ---------------------------------------------------------------------------
# Binning / Analysis 1
# ---------------------------------------------------------------------------

def test_merge_bins_deterministic_and_count_preserving():
    edges = np.linspace(0.0, 1.0, 11)
    counts = [500, 400, 10, 5, 3, 2, 1, 1, 1, 1]
    e1, c1 = ag.merge_bins(edges, counts, min_count=200, min_bins=3)
    e2, c2 = ag.merge_bins(edges, counts, min_count=200, min_bins=3)
    assert len(c1) == len(e1) - 1
    assert sum(c1) == sum(counts)
    assert all(c >= 200 for c in c1) or len(c1) == 3
    assert np.all(np.diff(e1) > 0)
    assert np.allclose(e1, e2) and c1 == c2


def test_gate_decomposition_known_values():
    rows = []
    for i in range(4):  # low-gate bin: 1 success, 3 failures among candidate-correct
        rows.append(make_row(source=0, f_mean=0.2, cand_ok=1, comm_ok=1 if i == 0 else 0,
                             prev_ok=1, d_cs=1.0, d_ss=0.5, d_sc=0.5, wrap=0))
    for i in range(4):  # high-gate bin: all succeed, all wrap
        rows.append(make_row(source=0, f_mean=0.8, cand_ok=1, comm_ok=1, prev_ok=1,
                             d_cs=2.0, d_ss=1.0, d_sc=0.5, wrap=1))
    dec = ag.gate_decomposition(table(rows), np.array([0.0, 0.5, 1.0]))
    b0, b1 = dec["bins"]
    assert b0["n"] == 4 and b1["n"] == 4
    assert b0["cand_acc"] == 1.0 and b0["comm_acc"] == pytest.approx(0.25)
    assert b0["cond_commit_given_cand"] == pytest.approx(0.25)
    assert b0["gap"] == pytest.approx(0.75)
    assert b0["prev_acc"] == 1.0 and b0["mean_d_cs"] == pytest.approx(1.0)
    assert b1["comm_acc"] == 1.0 and b1["wrap_frac"] == 1.0
    assert b1["cond_prev_given_fail"] is None  # no failures in the high bin


def test_choose_bin_edges_merges_sparse_high_gates():
    rows = [make_row(source=0, f_mean=0.05 + 0.001 * i) for i in range(500)]
    rows += [make_row(source=0, f_mean=0.95 + 0.0001 * i) for i in range(5)]
    edges = ag.choose_bin_edges(table(rows))
    assert edges[0] == 0.0 and edges[-1] == 1.0
    assert len(edges) <= 11


def test_conditional_gate_analysis_low_gate_failures():
    rows = []
    for _ in range(5):
        rows.append(make_row(source=0, f_mean=0.1, cand_ok=1, comm_ok=0, prev_ok=1, comm_eq_old=1))
    for _ in range(5):
        rows.append(make_row(source=0, f_mean=0.9, cand_ok=1, comm_ok=1, prev_ok=1, comm_eq_old=0))
    out = ag.conditional_gate_analysis(table(rows))
    assert out["n_cand_correct"] == 10
    assert out["n_cand_ok_commit_wrong"] == 5
    assert out["share_failures_below_low_gate"] == 1.0
    assert out["share_failures_at_or_above_high_gate"] == 0.0
    assert out["mean_f_commit_wrong"] < out["mean_f_commit_ok"]
    assert out["prev_ok_given_failure"] == 1.0
    assert out["comm_eq_old_given_failure"] == 1.0
    # f separates failures (low f) perfectly => AUC 0.0 (lower score = failure)
    assert out["f_auc_for_failure"] == pytest.approx(0.0)


def test_confound_stratification_known_subgroups():
    rows = []
    # general: dest != src, src value > 0: 3 ok + 1 commit failure (decodes to the old value 4)
    for i in range(4):
        fail = i == 3
        rows.append(make_row(source=0, dest=0, src_reg=1, operand_a=4, operand_b=3,
                             cand_ok=1, comm_ok=0 if fail else 1,
                             comm_pred=4 if fail else 7, comm_eq_old=1 if fail else 0))
    # no-op: src value 0 (correct result equals the old value): all ok
    for _ in range(4):
        rows.append(make_row(source=0, dest=2, src_reg=2, operand_a=5, operand_b=0,
                             cand_ok=1, comm_ok=1))
    # same-register, src value > 0: 1 ok + 1 commit failure (decodes to the old value 6)
    for i in range(2):
        fail = i == 1
        rows.append(make_row(source=0, dest=3, src_reg=3, operand_a=6, operand_b=4,
                             cand_ok=1, comm_ok=0 if fail else 1,
                             comm_pred=6 if fail else 0, comm_eq_old=1 if fail else 0))
    cs = ag.confound_stratification(table(rows), np.array([0.0, 1.0]))
    assert cs["strata"]["general"]["n"] == 4
    assert cs["strata"]["general"]["gap_cond_on_cand"] == pytest.approx(0.25)
    assert cs["strata"]["general"]["n_failures"] == 1
    assert cs["strata"]["no_op_src0"]["n"] == 4
    assert cs["strata"]["no_op_src0"]["n_failures"] == 0
    assert cs["strata"]["same_register_src_gt0"]["n"] == 2
    assert cs["strata"]["same_register_src_gt0"]["n_failures"] == 1
    assert cs["failure_decode_composition"]["n_failures"] == 2
    assert cs["failure_decode_composition"]["share_pred_equals_old"] == pytest.approx(1.0)
    assert cs["failure_decode_composition"]["share_pred_equals_new"] == pytest.approx(0.0)


def test_paired_transition_analysis_alignment_and_counts():
    rows_e, rows_f = [], []
    patterns = [(1, 1), (1, 0), (0, 1), (0, 0)]  # (comm_e, comm_f)
    for i, (ce, cf) in enumerate(patterns):
        base = dict(seed=42, source=0, seq_id=i, t=9, dest=0, src_reg=1, cand_ok=1)
        rows_e.append(make_row(comm_ok=ce, **base))
        rows_f.append(make_row(comm_ok=cf, **base))
    out = ag.paired_transition_analysis(table(rows_e), table(rows_f))
    assert out["contingency"] == {"n_both_cand_correct": 4, "both_commit_ok": 1,
                                  "E_only_fail": 1, "F_only_fail": 1, "both_fail": 1}
    assert out["per_seed"]["42"]["E_fail_rate"] == pytest.approx(0.5)
    assert out["per_seed"]["42"]["F_fail_rate"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ag.paired_transition_analysis(table(rows_e), table(rows_f[:3]))


# ---------------------------------------------------------------------------
# Analysis 2
# ---------------------------------------------------------------------------

def test_oracle_table_gap_and_lost_counts():
    rows = []
    for i in range(4):
        rows.append(make_row(source=SOURCE_IDS["isolated_add"], cand_ok=1,
                             comm_ok=1 if i < 3 else 0))
    for _ in range(3):
        rows.append(make_row(source=SOURCE_IDS["add_no_wrap"], cand_ok=1, comm_ok=1))
    t = ag.oracle_table(table(rows))
    iso = t["controlled:isolated_add"]
    assert iso["n"] == 4
    assert iso["committed_dest_acc"] == pytest.approx(0.75)
    assert iso["candidate_dest_acc"] == 1.0
    assert iso["oracle_minus_committed_gap"] == pytest.approx(0.25)
    assert iso["n_lost_at_commitment"] == 1
    assert t["controlled:add_no_wrap"]["oracle_minus_committed_gap"] == 0.0
    assert t["controlled:add_wrap"]["n"] == 0


# ---------------------------------------------------------------------------
# Analysis 3
# ---------------------------------------------------------------------------

def test_logistic_grouped_cv_separable_and_deterministic():
    rng = np.random.default_rng(3)
    y = np.tile([0.0, 1.0], 6)  # 12 samples, mixed classes per group
    X = (y * 2.0 + rng.normal(0, 0.3, size=12))[:, None]
    groups = np.repeat(np.arange(6), 2)
    cv = ag.grouped_cv_predict(X, y, groups, n_folds=5)
    assert cv["pooled_auc"] > 0.9
    w = ag.fit_logistic(X, y)
    p = ag.predict_logistic(w, X)
    assert np.all((p > 0.0) & (p < 1.0))
    assert p[0] < 0.5 and p[1] > 0.5
    cv2 = ag.grouped_cv_predict(X, y, groups, n_folds=5)
    assert np.allclose(cv["oof"], cv2["oof"], equal_nan=True)


def test_grouped_cv_noise_stays_near_chance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 1))
    y = (rng.random(80) < 0.5).astype(float)
    groups = np.repeat(np.arange(16), 5)
    cv = ag.grouped_cv_predict(X, y, groups, n_folds=5)
    assert 0.2 < cv["pooled_auc"] < 0.8


def test_bootstrap_auc_diff_identical_and_ordered_scores():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(20), 5)
    y = (rng.random(len(groups)) < 0.4).astype(float)
    p = rng.random(len(groups))
    same = ag.bootstrap_auc_diff(y, p, p, groups, n_boot=30, seed=0)
    assert same["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert same["ci95"][0] <= 0.0 <= same["ci95"][1]
    informative = y * 0.8 + rng.normal(0, 0.2, size=len(groups))
    weak = y * 0.1 + rng.normal(0, 0.2, size=len(groups))
    diff = ag.bootstrap_auc_diff(y, informative, weak, groups, n_boot=30, seed=0)
    assert diff["mean_diff"] > 0.05


# ---------------------------------------------------------------------------
# Seed-level inference
# ---------------------------------------------------------------------------

def test_paired_model_stats_sign_and_p_value():
    seeds = [42, 43, 44, 45, 46]
    per_seed_E = {str(s): {"x": v} for s, v in zip(seeds, [0.30, 0.29, 0.31, 0.30, 0.30])}
    per_seed_F = {str(s): {"x": v} for s, v in zip(seeds, [0.20, 0.20, 0.21, 0.20, 0.20])}
    sumE = ag.summarize_seed_metrics(per_seed_E)
    sumF = ag.summarize_seed_metrics(per_seed_F)
    out = ag.paired_model_stats(sumE, sumF)
    assert out["x"]["E_minus_F"] == pytest.approx(0.098, abs=1e-12)
    assert out["x"]["paired_t_p"] < 0.01
    assert out["x"]["E_mean"] == pytest.approx(0.3)
    assert out["x"]["cohens_dz"] > 3.0


def test_seed_metrics_per_seed_gap_and_summary():
    rows = []
    for seed, (cand_ok, comm_ok) in [(42, (1, 0)), (43, (0, 0))]:
        for _ in range(10):
            rows.append(make_row(source=0, seed=seed, cand_ok=cand_ok, comm_ok=comm_ok, f_mean=0.5))
    m = ag.seed_metrics(table(rows))
    assert m["42"]["gap_rate"] == pytest.approx(1.0)
    assert m["43"]["gap_rate"] == pytest.approx(0.0)
    s = ag.summarize_seed_metrics(m)
    assert s["gap_rate"]["mean"] == pytest.approx(0.5)
    assert s["gap_rate"]["sd"] == pytest.approx(np.sqrt(0.5))


# ---------------------------------------------------------------------------
# Collector smoke test (real repo checkpoint, tiny hand-built sequences)
# ---------------------------------------------------------------------------

def _tiny_sequences():
    task = rg.gr.build_task(10)
    seq_ops = [(0, 0, -1, 5), (0, 1, -1, 7), (0, 2, -1, 3), (0, 3, -1, 4),
               (0, 0, -1, 2), (0, 1, -1, 9), (0, 2, -1, 6), (0, 3, -1, 1),
               (2, 0, 1, -1), (2, 0, 1, -1)]
    B = 3
    inputs = np.zeros((B, 10, 21), dtype=np.float32)
    step_targets = np.zeros((B, 10, 4), dtype=np.int64)
    for b in range(B):
        regs = [0, 0, 0, 0]
        for t, (op, d, s, v) in enumerate(seq_ops):
            inputs[b, t] = task._encode_operation(op, d, s, v)
            if op == task.OP_SET:
                regs[d] = v
            elif op == task.OP_SWAP:
                regs[d], regs[s] = regs[s], regs[d]
            elif op == task.OP_ADD:
                regs[d] = (regs[d] + regs[s]) % 10
            step_targets[b, t] = regs
    return torch.from_numpy(inputs), torch.from_numpy(step_targets)


def test_collector_smoke_from_repo_checkpoint():
    ckpt_dir = os.path.join(ROOT, "experiments", "checkpoints")
    loaded = rg.load_checkpoint(ckpt_dir, "6_nonlinear_candidate_mgu", 42)
    inputs, step_targets = _tiny_sequences()
    rows = rg.collect_from_model(loaded["model"], 42, SOURCE_IDS["test"], inputs, step_targets)
    assert len(rows) == 6  # 2 ADD steps x 3 sequences
    arr = np.asarray(rows, dtype=float)
    assert arr.shape[1] == len(COLS)
    ci = {c: i for i, c in enumerate(COLS)}
    assert set(np.unique(arr[:, ci["cand_ok"]])) <= {0.0, 1.0}
    assert set(np.unique(arr[:, ci["comm_ok"]])) <= {0.0, 1.0}
    assert np.all(arr[:, ci["f_mean"]] >= 0.0) and np.all(arr[:, ci["f_mean"]] <= 1.0)
    assert np.all(arr[:, ci["d_cs"]] >= 0.0) and np.all(arr[:, ci["d_ss"]] >= 0.0)
    # First sequence ADD at t=8: dest=0, src=1, operands 2 and 9 -> wrap (11 >= 10)
    assert arr[0, ci["t"]] == 8.0 and arr[0, ci["dest"]] == 0.0 and arr[0, ci["src_reg"]] == 1.0
    assert arr[0, ci["operand_a"]] == 2.0 and arr[0, ci["operand_b"]] == 9.0
    assert arr[0, ci["wrap"]] == 1.0 and arr[0, ci["isolated"]] == 1.0
    # Second ADD at t=9: operands 1 and 9 -> wrap (10 >= 10)
    assert arr[1, ci["t"]] == 9.0 and arr[1, ci["wrap"]] == 1.0


# ---------------------------------------------------------------------------
# End-to-end synthetic pipeline (analysis + figures + JSON)
# ---------------------------------------------------------------------------

def _synthetic_table(rng, n_test=420):
    rows = []
    for i in range(n_test):
        f = float(rng.uniform(0.05, 0.9))
        cand = float(rng.random() < 0.8)
        comm = float(cand and (rng.random() < 0.35 + 0.6 * f))
        rows.append(make_row(
            seed=42 + (i % 2), source=0, seq_id=i // 10, t=2 + (i % 8),
            dest=float(i % 4), src_reg=float((i + 1) % 4), operand_a=float(i % 10),
            operand_b=float((i + 3) % 10), wrap=float(i % 3 == 0), isolated=float(i % 2 == 0),
            cand_ok=cand, comm_ok=comm, prev_ok=float(rng.random() < 0.9),
            comm_eq_old=float(not comm and rng.random() < 0.5),
            f_mean=f, f_min=f - 0.1, f_max=f + 0.1,
            d_cs=float(rng.uniform(0, 10)), d_ss=float(rng.uniform(0, 6)),
            d_sc=float(rng.uniform(0, 6)), cos_sc=float(rng.uniform(-1, 1)),
            cand_ce=float(rng.uniform(0, 3)), comm_ce=float(rng.uniform(0, 3)),
            cand_pt=float(rng.uniform(0, 1)), comm_pt=float(rng.uniform(0, 1)),
            cand_pmax=float(rng.uniform(0.3, 1)), comm_pmax=float(rng.uniform(0.3, 1))))
    for sid in (1, 2, 3, 4, 5):
        for j in range(12):
            cand = float(rng.random() < 0.8)
            comm = float(cand and rng.random() < 0.6)
            rows.append(make_row(seed=42 + (j % 2), source=sid, seq_id=j, t=8 + (j % 2),
                                 cand_ok=cand, comm_ok=comm, f_mean=float(rng.uniform(0.05, 0.9)),
                                 d_cs=float(rng.uniform(0, 10)), d_ss=float(rng.uniform(0, 6)),
                                 d_sc=float(rng.uniform(0, 6))))
    return table(rows)


def test_build_analysis_end_to_end_synthetic(tmp_path):
    rng = np.random.default_rng(7)
    tables = {"E_candidate": _synthetic_table(rng), "F_combined": _synthetic_table(rng)}
    meta = {"n_rows": {k: len(v) for k, v in tables.items()}, "columns": COLS,
            "source_ids": SOURCE_IDS}
    results = ag.build_analysis(tables, meta)
    json.dumps(results)  # must be JSON-serializable
    for model in ("E_candidate", "F_combined"):
        assert model in results["analysis_1_gate_decomposition"]
        assert model in results["analysis_2_oracle_commit"]
        assert model in results["analysis_3_predictability"]
        assert model in results["summary_for_decision"]
    assert results["seed_level_inference"]["paired_E_vs_F"] is not None
    figs = ag.make_figures(tables, results, str(tmp_path), prefix="test_g")
    assert len(figs) == 7
    assert all(os.path.exists(p) for p in figs)
