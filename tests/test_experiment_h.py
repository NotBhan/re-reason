"""Tests for the Experiment H λ-interpolation pipeline.

Covers: λ* statistics, trajectory classification, near-miss analysis, replay-vs-scalar
comparison, failure categorization, seed-level paired stats, the analysis assembly +
all nine figures on synthetic data, an npz round trip, and a runner smoke test on a real
repo checkpoint (λ=0/λ=1/replay consistency).
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_h as ah  # noqa: E402
import run_experiment_h as rh  # noqa: E402
from run_experiment_g import SOURCE_IDS  # noqa: E402

LAMBDAS = rh.LAMBDAS


def make_row(**kw):
    row = {c: 0.0 for c in rh.SCALAR_COLS}
    row.update(kw)
    return row


def make_table(rows, digits, correct):
    scalars = np.array([[float(r[c]) for c in rh.SCALAR_COLS] for r in rows], dtype=np.float64)
    return ah.HTable(scalars, rh.SCALAR_COLS, np.asarray(digits, dtype=np.int8),
                     np.asarray(correct, dtype=np.uint8), LAMBDAS)


def digits_from_lambda_star(lam_star, old=4, new=8, n_lam=len(LAMBDAS)):
    """Simple digits: old before λ*, correct from λ* on."""
    d = np.full(n_lam, old, dtype=np.int8)
    c = np.zeros(n_lam, dtype=np.uint8)
    if lam_star is not None:
        first = int(round(lam_star / 0.05))
        d[first:] = new
        c[first:] = 1
    return d, c


# ---------------------------------------------------------------------------
# λ* statistics
# ---------------------------------------------------------------------------

def test_lambda_star_stats_basic():
    rows, digits, correct = [], [], []
    for lam in (0.0, 0.5, 1.0):
        d, c = digits_from_lambda_star(lam)
        rows.append(make_row(source=0, dest=0, src_reg=1, operand_a=4, operand_b=4))
        digits.append(d)
        correct.append(c)
    d, c = digits_from_lambda_star(None)
    rows.append(make_row(source=0, dest=0, src_reg=1, operand_a=4, operand_b=4))
    digits.append(d)
    correct.append(c)
    t = make_table(rows, digits, correct)
    stats = ah.lambda_star_stats(t, np.ones(4, bool))
    assert stats["n"] == 4 and stats["n_valid"] == 3
    assert stats["frac_no_solution"] == pytest.approx(0.25)
    assert stats["quantiles"]["p50"] == pytest.approx(0.5)
    assert stats["frac_eq_1.0"] == pytest.approx(1 / 3)
    assert stats["frac_le_0.5"] == pytest.approx(2 / 3)


def test_lambda_star_vs_feature_bins():
    rng = np.random.default_rng(0)
    rows, digits, correct = [], [], []
    xs = np.linspace(1, 10, 50)
    for i, x in enumerate(xs):
        lam = float(np.clip(x / 12.0, 0, 1))
        d, c = digits_from_lambda_star(lam)
        rows.append(make_row(source=0, dest=0, src_reg=1, operand_a=4, operand_b=4, d_cs=x))
        digits.append(d)
        correct.append(c)
    t = make_table(rows, digits, correct)
    out = ah.lambda_star_vs_feature(t, np.ones(len(rows), bool), "d_cs", n_bins=5)
    assert out["n"] == 50
    assert out["spearman"] > 0.9  # λ* increases with d_cs by construction
    assert len(out["bins"]) == 5
    assert out["bins"][-1]["lambda_star_mean"] > out["bins"][0]["lambda_star_mean"]


# ---------------------------------------------------------------------------
# Trajectories / near-miss
# ---------------------------------------------------------------------------

def test_trajectory_classes():
    rows = []
    digits = []
    correct = []
    # (a) immediately correct
    d = np.full(len(LAMBDAS), 8, dtype=np.int8); c = np.ones(len(LAMBDAS), np.uint8)
    rows.append(make_row(source=0, operand_a=4, operand_b=4)); digits.append(d); correct.append(c)
    # (b) starts old, monotone to correct at 0.3
    d, c = digits_from_lambda_star(0.3, old=4, new=8)
    d[:6] = 4
    rows.append(make_row(source=0, operand_a=4, operand_b=4)); digits.append(d); correct.append(c)
    # (c) correct at 0.2 then reverts at 0.6
    d, c = digits_from_lambda_star(0.2, old=4, new=8)
    d[12:] = 6; c[12:] = 0
    rows.append(make_row(source=0, operand_a=4, operand_b=4)); digits.append(d); correct.append(c)
    # (d) never correct
    d, c = digits_from_lambda_star(None, old=4, new=8)
    rows.append(make_row(source=0, operand_a=4, operand_b=4)); digits.append(d); correct.append(c)
    t = make_table(rows, digits, correct)
    tr = ah.trajectory_classes(t, np.ones(4, bool))
    assert tr["counts"]["immediately_correct"] == 1
    assert tr["counts"]["wrong_then_correct_monotone"] == 1
    assert tr["counts"]["correct_then_reverts"] == 1
    assert tr["counts"]["never_correct"] == 1
    assert tr["wrong_then_correct_sub"]["starts_old"] == 1


def test_near_miss_analysis_path_vs_unrelated():
    # Path row: old=4, ob=4 -> p(λ)=4+4λ; wrong decodes follow the path, correct at 0.75.
    n = len(LAMBDAS)
    d = np.array([4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], dtype=np.int8)
    c = (d == 8).astype(np.uint8)
    # Unrelated row: decode jumps to an unrelated digit until λ=0.5
    d2 = np.full(n, 4, dtype=np.int8); d2[4:] = 0  # 0 is far from the path (4+4λ)
    c2 = (d2 == 8).astype(np.uint8)
    rows = [make_row(source=0, operand_a=4, operand_b=4, comm_ok=0)] * 2
    t = make_table(rows, [d, d2], [c, c2])
    out = ah.near_miss_analysis(t, np.ones(2, bool))
    classes = out["slot_classes"]
    assert classes["correct"] == int(c.sum() + c2.sum())
    # Path row contributes near-path wrong slots; unrelated row contributes "other"
    assert classes["near_path"] >= 5
    assert classes["other"] >= 5
    assert out["last_wrong_before_correct"]["frac_adjacent_to_new"] >= 0.5


# ---------------------------------------------------------------------------
# Replay / counterfactual / categorization
# ---------------------------------------------------------------------------

def test_replay_vs_scalar_rates():
    rows = [
        make_row(source=0, cand_ok=1, comm_ok=0, replay_ok=0, mean_ok=0,
                 replay_pred=4, comm_pred=4, mean_pred=5, f_mean=0.3, replay_maxdiff=1e-6),
        make_row(source=0, cand_ok=1, comm_ok=1, replay_ok=1, mean_ok=1,
                 replay_pred=8, comm_pred=8, mean_pred=8, f_mean=0.4, replay_maxdiff=1e-6),
    ]
    digits = [digits_from_lambda_star(1.0)[0], digits_from_lambda_star(0.5)[0]]
    correct = [digits_from_lambda_star(1.0)[1], digits_from_lambda_star(0.5)[1]]
    t = make_table(rows, digits, correct)
    out = ah.replay_vs_scalar(t, np.ones(2, bool))
    assert out["candidate_acc"] == pytest.approx(1.0)
    assert out["recorded_commit_acc"] == pytest.approx(0.5)
    assert out["vector_gate_replay_acc"] == pytest.approx(0.5)
    assert out["mean_gate_counterfactual_acc"] == pytest.approx(0.5)
    assert out["lambda1_acc"] == pytest.approx(1.0)
    assert out["replay_decode_agreement_with_recorded"] == pytest.approx(1.0)


def test_failure_categorization_counts_and_splits():
    rows, digits, correct = [], [], []
    # A: replay correct (inconsistent with "committed-wrong" by design; verified as empty in real data)
    rows.append(make_row(source=0, cand_ok=1, comm_ok=0, replay_ok=1, f_mean=0.3, lambda_eff=0.5))
    digits.append(digits_from_lambda_star(0.5)[0]); correct.append(digits_from_lambda_star(0.5)[1])
    # B below f_mean and below lambda_eff
    rows.append(make_row(source=0, cand_ok=1, comm_ok=0, replay_ok=0, f_mean=0.4, lambda_eff=0.6))
    digits.append(digits_from_lambda_star(0.3)[0]); correct.append(digits_from_lambda_star(0.3)[1])
    # B above f_mean and above lambda_eff
    rows.append(make_row(source=0, cand_ok=1, comm_ok=0, replay_ok=0, f_mean=0.3, lambda_eff=0.2))
    digits.append(digits_from_lambda_star(0.8)[0]); correct.append(digits_from_lambda_star(0.8)[1])
    # C: no solution
    rows.append(make_row(source=0, cand_ok=1, comm_ok=0, replay_ok=0, f_mean=0.3, lambda_eff=0.5))
    digits.append(digits_from_lambda_star(None)[0]); correct.append(digits_from_lambda_star(None)[1])
    t = make_table(rows, digits, correct)
    out = ah.failure_categorization(t, np.ones(4, bool))
    assert out["A_actual_commit_correct"] == 1
    assert out["B_actual_wrong_scalar_correct_exists"] == 2
    assert out["C_no_scalar_correct"] == 1
    assert out["B_below_f_mean"] == 1 and out["B_above_f_mean"] == 1
    assert out["B_below_lambda_eff"] == 1 and out["B_above_lambda_eff"] == 1


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _synthetic_meta(n_models=1):
    return {"lambdas": [float(x) for x in LAMBDAS],
            "models": {"E_candidate": {"checkpoints": {"42": {"sha256": "x", "unchanged_after_run": True}}}}}


def test_sanity_checks_pass_and_fail():
    n = 3
    rows = [make_row(source=0, cand_ok=1, prev_pred=4, cand_pred=8, comm_pred=4, replay_pred=4,
                     replay_maxdiff=1e-6, lambda_star=0.5) for _ in range(n)]
    d, c = digits_from_lambda_star(0.5)
    t_ok = make_table(rows, [d] * n, [c] * n)
    meta = {"lambdas": [float(x) for x in LAMBDAS],
            "models": {m: {"checkpoints": {"42": {"sha256": "x", "unchanged_after_run": True}}}
                       for m in ("E_candidate", "F_combined")}}
    out = ah.sanity_checks({"E_candidate": t_ok, "F_combined": t_ok}, meta)
    assert out["passed"]
    rows_bad = [make_row(source=0, prev_pred=0, cand_pred=8, comm_pred=4, replay_pred=4,
                         replay_maxdiff=1e-2, lambda_star=0.9) for _ in range(n)]
    t_bad = make_table(rows_bad, [d] * n, [c] * n)
    out_bad = ah.sanity_checks({"E_candidate": t_bad, "F_combined": t_ok}, meta)
    assert not out_bad["passed"]


# ---------------------------------------------------------------------------
# Synthetic end-to-end assembly + figures
# ---------------------------------------------------------------------------

def _synthetic_htable(seed_offset: int = 0) -> ah.HTable:
    rng = np.random.default_rng(11 + seed_offset)
    rows, digits, correct = [], [], []
    n_test = 260
    for i in range(n_test):
        seed = 42 + (i % 2)
        dest, src_reg = i % 4, (i + 1) % 4
        ob = (i % 9) + 1
        oa = (i * 3) % 10
        new = (oa + ob) % 10
        cand = bool(rng.random() < 0.85)
        lam_star = float(rng.uniform(0.05, 1.0)) if cand else None
        d, c = digits_from_lambda_star(lam_star, old=oa, new=new)
        if not cand:
            c[:] = 0
            d[:] = oa
        elif i % 17 == 0:  # non-monotone: correct early, revert, correct again at the end
            d[12:17] = (oa + 1) % 10
            c[12:17] = 0
        cand_pred = int(d[20])
        comm = bool(cand and rng.random() < 0.65)
        comm_pred = new if comm else (oa if rng.random() < 0.5 else (oa + 2) % 10)
        rows.append(make_row(
            seed=seed, source=float(SOURCE_IDS["test"]), seq_id=i // 10, t=2 + (i % 8),
            dest=dest, src_reg=src_reg, operand_a=oa, operand_b=ob,
            wrap=float(oa + ob >= 10), isolated=float(i % 2 == 0),
            cand_ok=float(cand_pred == new), comm_ok=float(comm_pred == new),
            prev_ok=float(rng.random() < 0.9), comm_eq_old=float(comm_pred == oa),
            cand_pred=cand_pred, comm_pred=comm_pred, prev_pred=oa, replay_pred=comm_pred,
            mean_pred=int(d[13]), replay_ok=float(comm_pred == new), mean_ok=float(d[13] == new),
            cand_ok_direct=float(cand_pred == new),
            f_mean=float(rng.uniform(0.2, 0.5)), d_cs=float(rng.uniform(2, 18)),
            d_ss=float(rng.uniform(0, 10)), d_sc=float(rng.uniform(0, 10)),
            replay_maxdiff=1e-7, lambda_eff=float(rng.uniform(0, 1)),
            offpath_dist=float(rng.uniform(0, 6)),
            cand_pt=0.9, comm_pt=0.8,
            lambda_star=(lam_star if lam_star is not None else np.nan),
            n_correct=float(c.sum())))
        digits.append(d)
        correct.append(c)
    for sid in (1, 2, 3, 4, 5):
        for j in range(6):
            oa = (j * 2) % 10
            ob = (j % 5) + 1
            new = (oa + ob) % 10
            d, c = digits_from_lambda_star(0.4, old=oa, new=new)
            rows.append(make_row(seed=42, source=float(sid), seq_id=j, t=8, dest=0, src_reg=1,
                                 operand_a=oa, operand_b=ob, cand_ok=1, comm_ok=float(c[20]),
                                 cand_pred=int(d[20]), comm_pred=int(d[20]), prev_pred=oa,
                                 replay_pred=int(d[20]), mean_pred=int(d[10]),
                                 replay_ok=float(c[20]), mean_ok=float(c[10]),
                                 lambda_eff=0.5, f_mean=0.3, d_cs=8.0, lambda_star=0.4,
                                 n_correct=float(c.sum())))
            digits.append(d)
            correct.append(c)
    return make_table(rows, digits, correct)


def test_build_analysis_end_to_end_and_figures(tmp_path):
    tables = {"E_candidate": _synthetic_htable(0), "F_combined": _synthetic_htable(1)}
    meta = {"lambdas": [float(x) for x in LAMBDAS],
            "models": {m: {"checkpoints": {"42": {"sha256": "x", "unchanged_after_run": True},
                                           "43": {"sha256": "y", "unchanged_after_run": True}}}
                       for m in tables}}
    results = ah.build_analysis(tables, meta)
    json.dumps(results)  # JSON-serializable
    for key in ("sanity_checks", "curves", "lambda_star", "failure_categorization",
                "trajectories", "near_miss", "replay_vs_scalar", "categories", "hypotheses"):
        assert key in results
    figs = ah.make_figures(tables, results, str(tmp_path), prefix="test_h")
    assert len(figs) == 9
    assert all(os.path.exists(p) for p in figs)


# ---------------------------------------------------------------------------
# npz round trip
# ---------------------------------------------------------------------------

def test_load_interp_roundtrip(tmp_path):
    t = _synthetic_htable(0)
    scalars = t.scalars.astype(np.float32)
    path = str(tmp_path / "interp.npz")
    meta = {"lambdas": [float(x) for x in LAMBDAS], "scalar_cols": rh.SCALAR_COLS}
    np.savez_compressed(path, meta_json=np.array([json.dumps(meta)]),
                        E_candidate__scalars=scalars, E_candidate__digits=t.digits,
                        E_candidate__correct=t.correct.astype(np.uint8),
                        F_combined__scalars=scalars, F_combined__digits=t.digits,
                        F_combined__correct=t.correct.astype(np.uint8))
    tables, loaded_meta = ah.load_interp(path)
    assert set(tables) == {"E_candidate", "F_combined"}
    assert np.allclose(tables["E_candidate"].scalars, scalars, equal_nan=True)
    assert np.array_equal(tables["E_candidate"].digits, t.digits)
    assert np.array_equal(tables["E_candidate"].correct, t.correct)
    assert loaded_meta["scalar_cols"] == rh.SCALAR_COLS


# ---------------------------------------------------------------------------
# Runner smoke test (real repo checkpoint)
# ---------------------------------------------------------------------------

def _tiny_sequences():
    import run_generator_repair_feasibility as gr
    task = gr.build_task(10)
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


def test_runner_smoke_lambda0_lambda1_replay_consistency():
    import run_experiment_g as eg
    ckpt_dir = os.path.join(ROOT, "experiments", "checkpoints")
    loaded = eg.load_checkpoint(ckpt_dir, "6_nonlinear_candidate_mgu", 42)
    inputs, step_targets = _tiny_sequences()
    out = rh.sweep_dataset(loaded["model"], inputs, step_targets, seed=42, source_id=0)
    n = len(out["seed"])
    assert n == 6  # 2 ADD steps x 3 sequences
    assert out["digits"].shape == (n, len(LAMBDAS))
    # λ=0 → previous-state decode; λ=1 → candidate decode
    assert np.array_equal(out["digits"][:, 0], np.round(out["prev_pred"]).astype(np.int8))
    assert np.array_equal(out["digits"][:, -1], np.round(out["cand_pred"]).astype(np.int8))
    # vector-gate replay reproduces the recorded committed state and decode
    assert out["replay_maxdiff"].max() < 1e-4
    assert np.array_equal(np.round(out["replay_pred"]), np.round(out["comm_pred"]))
    # λ=1 correctness equals candidate correctness
    assert np.array_equal(out["correct"][:, -1].astype(bool), out["cand_ok"] > 0.5)
    # λ* exists for candidate-correct transitions
    cc = out["cand_ok"] > 0.5
    assert np.all(~np.isnan(out["lambda_star"][cc]))
