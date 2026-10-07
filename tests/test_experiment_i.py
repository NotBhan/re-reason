"""Tests for the Experiment I coordinate-wise decision-margin commit pipeline.

Covers:
  * pure helpers (quantile stats, paired effects, decile failure rates, mask definitions);
  * the exact decision-margin algebra and the alpha* closed form against a brute-force scan;
  * the collector on a real repo checkpoint (replay identity, sum(q)=dm, primary coordinate
    alignment, determinism);
  * the runner's Experiment-G row-verification;
  * sanity checks (passing and failing);
  * the evidence-classification rule;
  * an end-to-end synthetic run of the analysis + all figures.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_experiment_i as ai  # noqa: E402
import run_experiment_g as eg  # noqa: E402
import run_experiment_i as ri  # noqa: E402
from run_experiment_g import SOURCE_IDS  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_qstats_and_effect_band():
    q = ai.qstats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert q["n"] == 4 and q["mean"] == pytest.approx(2.5)
    assert q["quantiles"]["p50"] == pytest.approx(2.5)
    assert ai.effect_band(0.1) == "negligible"
    assert ai.effect_band(-0.3) == "small"
    assert ai.effect_band(0.6) == "medium"
    assert ai.effect_band(1.5) == "large"
    assert ai.effect_band(float("nan")) == "na"


def test_paired_stats_sign_and_pvalue():
    a = np.array([0.5, 0.52, 0.51, 0.49, 0.50])
    b = np.array([0.4, 0.41, 0.42, 0.39, 0.40])
    out = ai.paired(a, b)
    assert out["mean_diff"] == pytest.approx(0.1, abs=1e-6)
    assert out["cohens_dz"] > 3
    assert out["sign_positive"] == 5 and out["sign_negative"] == 0
    assert out["paired_t_p"] < 0.01


def test_failure_rate_by_decile_monotone():
    x = np.linspace(0, 1, 200)
    fail = np.zeros(200, dtype=bool)
    fail[:100] = True  # low-x half fails
    prim = np.ones(200, dtype=bool)
    tbl = ai._failure_rate_by_decile(prim, fail, x)
    assert len(tbl) == 10
    assert tbl[0]["failure_rate"] > tbl[-1]["failure_rate"]
    assert sum(t["n"] for t in tbl) == 200


def test_masks_definitions():
    t = _synthetic_itable(0)
    m = ai.masks(t)
    # primary = test & general & candidate-correct
    assert m["primary"].sum() > 0
    assert not np.any(m["primary"] & (t["source"] != SOURCE_IDS["test"]))
    assert not np.any(m["primary"] & (t["operand_b"] == 0))
    assert not np.any(m["primary"] & (t["dest"] == t["src_reg"]))
    assert not np.any(m["primary"] & (t["cand_ok"] < 0.5))
    # A and B partition the primary set
    assert np.array_equal(m["A_succ"] | m["B_fail"], m["primary"])
    assert not np.any(m["A_succ"] & m["B_fail"])


# ---------------------------------------------------------------------------
# Synthetic table builders
# ---------------------------------------------------------------------------

def _row(**kw):
    row = {c: 0.0 for c in ri.COLS}
    row.update(kw)
    return row


def _synthetic_itable(seed_offset: int) -> ai.ITable:
    rng = np.random.default_rng(1234 + seed_offset)
    n_prim = 240
    n_other_test = 60
    rows = []
    coords = {k: [] for k in ri.COORD_KEYS}
    for i in range(n_prim):
        dest, src = i % 4, (i + 1) % 4
        ob = (i % 9) + 1
        oa = (i * 3) % 10
        y = (oa + ob) % 10
        m_prev = -float(rng.uniform(5, 16))
        m_cand = float(rng.uniform(0.5, 12))
        dm_cand = m_cand - m_prev
        rho = float(rng.uniform(0.4, 1.05))
        dm_commit = rho * dm_cand
        m_commit = m_prev + dm_commit
        comm_ok = 1.0 if (m_commit > 0.3 and rng.random() > 0.15) else 0.0
        alpha_full = -m_prev / dm_cand
        qc = rng.normal(0, 1.0, size=212).astype(np.float32)
        fvec = rng.uniform(0, 1, size=212).astype(np.float32)
        # make the scalar sums consistent with the coordinate arrays
        rows.append(_row(
            seed=42 + (i % 2), source=SOURCE_IDS["test"], seq_id=i // 10, t=2 + (i % 8),
            dest=dest, src_reg=src, operand_a=oa, operand_b=ob,
            wrap=float(oa + ob >= 10), isolated=float(i % 2 == 0),
            cand_ok=1.0, comm_ok=comm_ok, prev_ok=1.0, comm_eq_old=float(comm_ok == 0),
            cand_pred=y, comm_pred=y if comm_ok else (oa), prev_pred=oa,
            y_true=y, a_star_prev=(y + 1) % 10, a_star_commit=(y + 2) % 10,
            m_prev=m_prev, m_cand=m_cand, m_commit=m_commit, m_commit_altstar=-0.5,
            dm_cand=dm_cand, dm_commit=dm_commit, margin_shortfall=m_cand - m_commit,
            rho_margin=dm_commit / dm_cand, rho_defined=1.0,
            sign_m_prev_pos=float(m_prev > 0), sign_m_cand_pos=float(m_cand > 0),
            sign_m_commit_pos=float(m_commit > 0),
            crossed_at_commit=float(m_prev <= 0 < m_commit), lost_at_commit=0.0,
            alpha_star_margin=alpha_full, alpha_star_full=alpha_full, alpha_full_solution=1.0,
            align_ratio=dm_cand / float(rng.uniform(8, 20)),
            f_mean=float(fvec.mean()), f_min=float(fvec.min()), f_max=float(fvec.max()),
            f_std=float(fvec.std()), d_cs=float(rng.uniform(8, 20)),
            d_ss=float(rng.uniform(4, 14)), d_sc=float(rng.uniform(4, 14)),
            q_cand_sum=dm_cand, q_cand_pos_sum=float(qc.clip(min=0).sum()),
            q_cand_neg_sum=float(np.abs(qc.clip(max=0)).sum()), q_cand_abs_sum=float(np.abs(qc).sum()),
            q_deliv_sum=dm_commit, q_deliv_pos_sum=float((qc * fvec).clip(min=0).sum()),
            q_deliv_neg_sum=float(np.abs((qc * fvec).clip(max=0)).sum()),
            gw_all=float((np.abs(qc) * fvec).sum() / max(np.abs(qc).sum(), 1e-9)),
            gw_pos=float((qc.clip(min=0) * fvec).sum() / max(qc.clip(min=0).sum(), 1e-9)),
            gw_neg=float((np.abs(qc.clip(max=0)) * fvec).sum() / max(np.abs(qc.clip(max=0)).sum(), 1e-9)),
            gw_mag=float((np.abs(rng.normal(0, 1, 212)) * fvec).sum() / 200.0),
            delivered_frac_pos=float(rng.uniform(0.4, 1.0)), suppressed_frac_pos=float(rng.uniform(0, 0.6)),
            delivered_frac_neg=float(rng.uniform(0.2, 0.9)), suppressed_frac_neg=float(rng.uniform(0.1, 0.8)),
            replay_maxdiff=0.0))
        for k in ri.COORD_KEYS:
            coords[k].append({"f": fvec, "move": fvec * rng.normal(0, 1, 212),
                              "qcand": qc, "abscand": np.abs(rng.normal(0, 1, 212))}[k])
    for j in range(n_other_test):
        rows.append(_row(seed=42 + (j % 2), source=SOURCE_IDS["test"], seq_id=100 + j, t=9,
                         dest=0, src_reg=0, operand_a=3, operand_b=0, cand_ok=1.0, comm_ok=1.0,
                         f_mean=0.3, rho_margin=float("nan")))
    for sid in (1, 2, 3, 4, 5):
        for j in range(12):
            rows.append(_row(seed=42 + (j % 2), source=sid, seq_id=j, t=8,
                             dest=0, src_reg=1, operand_a=4, operand_b=3, cand_ok=1.0,
                             comm_ok=float(j % 2 == 0), f_mean=0.3, rho_margin=0.7,
                             dm_cand=10.0, dm_commit=7.0, m_cand=5.0, m_prev=-5.0, m_commit=2.0,
                             q_cand_sum=10.0, q_deliv_sum=7.0))
    scalars = np.asarray([[r[c] for c in ri.COLS] for r in rows], dtype=np.float64)
    primary_row = np.arange(n_prim, dtype=np.int64)
    coord_arrays = {k: np.stack(coords[k], axis=0).astype(np.float32) for k in ri.COORD_KEYS}
    # realign coordinate arrays to primary rows (they were appended in the same order)
    return ai.ITable(scalars, ri.COLS, primary_row, coord_arrays)


def _synthetic_tables():
    return {"E_candidate": _synthetic_itable(0), "F_combined": _synthetic_itable(1)}


def _synthetic_meta(tables):
    return {
        "cols": ri.COLS, "coord_keys": ri.COORD_KEYS, "eps_abs": ri.EPS_ABS,
        "source_ids": SOURCE_IDS, "op_add": 2,
        "models": {m: {"checkpoints": {"42": {"path": "x", "sha256": "y", "unchanged_after_run": True},
                                       "43": {"path": "x", "sha256": "y", "unchanged_after_run": True}}}
                   for m in tables},
        "g_verification": {m: {"passed": True, "algebra": {"replay_identity_ok": True,
                                                           "q_deliv_equals_dm_commit_maxdiff": 0.0}}
                           for m in tables},
    }


def _synthetic_h_aux(tables):
    out = {}
    for m, t in tables.items():
        n = len(t)
        out[m] = {
            "lambda_eff": np.full(n, 0.3), "lambda_star": np.full(n, 0.8),
            "mean_ok": np.zeros(n), "replay_ok": np.zeros(n), "h_f_mean": t["f_mean"].copy(),
        }
    return out


# ---------------------------------------------------------------------------
# End-to-end synthetic analysis
# ---------------------------------------------------------------------------

def test_build_analysis_end_to_end_and_figures(tmp_path):
    tables = _synthetic_tables()
    meta = _synthetic_meta(tables)
    h_aux = _synthetic_h_aux(tables)
    align = {m: {"f_mean_maxdiff": 0.0} for m in tables}
    res = ai.build_analysis(tables, meta, h_aux, align)
    json.dumps(res)  # must be JSON-serializable
    for key in ("sanity_checks", "analysis_1_margin_movement", "analysis_2_boundary",
                "analysis_3_vector_gate_delivery", "analysis_4_coordinate_attribution",
                "analysis_5_available_vs_delivered", "analysis_6_alpha_star",
                "analysis_7_coordinate_gate_effect", "analysis_8_four_way",
                "analysis_9_E_vs_F", "analysis_10_evidence", "categories"):
        assert key in res
    for m in tables:
        assert res["analysis_10_evidence"][m]["verdict"] in (
            "A_gate_under_movement", "B_readout_precision", "C_both", "D_inconclusive")
        assert res["analysis_1_margin_movement"][m]["n_primary"] > 0
    # Determinism of the analysis assembly (excluding the wall-clock timestamp).
    res2 = ai.build_analysis(tables, meta, h_aux, align)
    strip = lambda r: {k: ({**v} if k != "metadata" else {kk: vv for kk, vv in v.items() if kk != "date"})
                       for k, v in r.items()}
    assert json.dumps(strip(res), sort_keys=True) == json.dumps(strip(res2), sort_keys=True)

    figs = ai.make_figures(tables, res, h_aux, str(tmp_path), prefix="test_i")
    assert len(figs) == 12
    assert all(os.path.exists(p) for p in figs)


def test_analysis_3_identity_exact_on_synthetic():
    tables = _synthetic_tables()
    meta = _synthetic_meta(tables)
    h_aux = _synthetic_h_aux(tables)
    res = ai.build_analysis(tables, meta, h_aux, {m: {"f_mean_maxdiff": 0.0} for m in tables})
    for m in tables:
        a3 = res["analysis_3_vector_gate_delivery"][m]
        assert a3["identity_sum_q_equals_dm_commit_max_abs_residual"] < 1e-9
        assert a3["head_is_linear"] is True


def test_sanity_checks_pass_and_fail():
    tables = _synthetic_tables()
    meta = _synthetic_meta(tables)
    h_aux = _synthetic_h_aux(tables)
    align = {m: {"f_mean_maxdiff": 0.0} for m in tables}
    assert ai.sanity_checks(tables, meta, h_aux, align)["passed"]
    meta_bad = json.loads(json.dumps(meta))
    meta_bad["g_verification"]["E_candidate"]["passed"] = False
    out = ai.sanity_checks(tables, meta_bad, h_aux, align)
    assert not out["passed"] and "E_candidate:G_row_verification" in out["failures"]


# ---------------------------------------------------------------------------
# Evidence rule
# ---------------------------------------------------------------------------

def test_evidence_rule_directions():
    base = {"analysis_5_available_vs_delivered": {
        "E_candidate": {"seed_paired_fail_minus_succ": {
            "rho_margin_med": {"cohens_dz": -2.0, "sign_negative": 5},
            "dm_commit_med": {"cohens_dz": -2.0, "sign_negative": 5},
            "gw_pos_mean": {"cohens_dz": -2.0, "sign_negative": 5},
            "align_ratio_med": {"cohens_dz": 0.0, "sign_negative": 0}}}}}
    tables = {"E_candidate": _synthetic_itable(3)}
    # force the R1 share to zero -> under-movement only
    t = tables["E_candidate"]
    mm = ai.masks(t)
    B = mm["B_fail"]
    t.scalars[:, t.index["rho_margin"]][B] = 0.05
    t.scalars[:, t.index["alpha_star_full"]][B] = 0.9
    res = ai.analysis_10(tables, base)
    assert res["E_candidate"]["U_score"] == 3
    assert res["E_candidate"]["verdict"] in ("A_gate_under_movement", "C_both")


# ---------------------------------------------------------------------------
# Collector smoke test on a real repo checkpoint
# ---------------------------------------------------------------------------

def _tiny_sequences():
    task = eg.gr.build_task(10)
    seq_ops = [(0, 0, -1, 5), (0, 1, -1, 7), (0, 2, -1, 3), (0, 3, -1, 4),
               (0, 0, -1, 2), (0, 1, -1, 9), (0, 2, -1, 6), (0, 3, -1, 1),
               (2, 0, 1, -1), (2, 0, 1, -1)]
    B = 4
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


def test_collector_identity_algebra_and_determinism():
    ckpt_dir = os.path.join(ROOT, "experiments", "checkpoints")
    loaded = eg.load_checkpoint(ckpt_dir, "E_candidate", 42)
    model = loaded["model"]
    inputs, step_targets = _tiny_sequences()
    out = ri.collect_dataset(model, inputs, step_targets, 42, SOURCE_IDS["test"])
    ci = {c: i for i, c in enumerate(ri.COLS)}
    sc = out["scalars"]
    assert sc.shape[1] == len(ri.COLS)
    # replay identity: (1-f) S_prev + f C == recorded S_t
    assert np.nanmax(sc[:, ci["replay_maxdiff"]]) < 1e-4
    # exact coordinate decomposition: sum_i q_i == dm_commit and sum_i q_cand_i == dm_cand
    assert np.nanmax(np.abs(sc[:, ci["q_deliv_sum"]] - sc[:, ci["dm_commit"]])) < 1e-4
    assert np.nanmax(np.abs(sc[:, ci["q_cand_sum"]] - sc[:, ci["dm_cand"]])) < 1e-4
    # candidate-correct primary rows carry coordinate arrays
    assert out["primary_local"] is not None
    n_prim = len(out["primary_local"])
    assert out["coords"]["qcand"].shape == (n_prim, 212)
    # determinism
    out2 = ri.collect_dataset(model, inputs, step_targets, 42, SOURCE_IDS["test"])
    assert np.array_equal(sc, out2["scalars"], equal_nan=True)


def test_alpha_star_margin_matches_brute_force_scan():
    ckpt_dir = os.path.join(ROOT, "experiments", "checkpoints")
    loaded = eg.load_checkpoint(ckpt_dir, "E_candidate", 42)
    model = loaded["model"]
    inputs, step_targets = _tiny_sequences()
    out = ri.collect_dataset(model, inputs, step_targets, 42, SOURCE_IDS["test"])
    ci = {c: i for i, c in enumerate(ri.COLS)}
    sc = out["scalars"]

    with torch.no_grad():
        diag = model.forward_with_states(inputs, return_components=True)
        states = torch.stack(diag.states, dim=1)
        cands = torch.stack(diag.candidates, dim=1)
        gates = torch.stack(diag.gates, dim=1)
    B, T, _ = states.shape
    prev = torch.cat([torch.zeros(B, 1, ri.gr.STATE_DIM), states[:, :-1]], dim=1)
    prev_true = torch.zeros_like(step_targets)
    prev_true[:, 1:] = step_targets[:, :-1]
    mask, op, reg_a, reg_b = ri.gr.write_mask_from_inputs(inputs)

    rows = []
    for b in range(B):
        for t in range(T):
            if int(op[b, t]) != ri.gr.RegisterTrackingTask.OP_ADD:
                continue
            d = int(reg_a[b, t])
            y = int(step_targets[b, t, d])
            Sp, C = prev[b, t], cands[b, t]
            with torch.no_grad():
                z_prev = model.decision_head(Sp).reshape(4, 10)[d]
                a_star = max((a for a in range(10) if a != y), key=lambda a: float(z_prev[a]))
                alphas = np.linspace(0, 3, 6001)
                crossing = np.nan
                m_prev = float(z_prev[y] - z_prev[a_star])
                for alpha in alphas:
                    s = Sp + float(alpha) * (C - Sp)
                    z = model.decision_head(s).reshape(4, 10)[d]
                    if float(z[y] - z[a_star]) >= 0:
                        crossing = float(alpha)
                        break
                # analytic: alpha = max(0, -m_prev / dm_cand)
                dm_cand = float((model.decision_head(C).reshape(4, 10)[d][y]
                                 - model.decision_head(C).reshape(4, 10)[d][a_star]) - m_prev)
                analytic = 0.0 if m_prev >= 0 else (-m_prev / dm_cand if dm_cand > ri.EPS_ABS else np.nan)
            rows.append((analytic, crossing, m_prev))
    # compare analytic vs brute force for rows with a finite analytic value
    diffs = [abs(a - c) for a, c, _ in rows if np.isfinite(a) and np.isfinite(c)]
    assert diffs and max(diffs) < 0.01
    # stored alpha column matches the analytic recomputation
    stored = sc[:, ci["alpha_star_margin"]]
    for (analytic, _, _), s in zip(rows, stored):
        if np.isfinite(analytic):
            assert abs(analytic - s) < 1e-4


def test_verify_against_experiment_g_roundtrip():
    ckpt_dir = os.path.join(ROOT, "experiments", "checkpoints")
    loaded = eg.load_checkpoint(ckpt_dir, "E_candidate", 42)
    inputs, step_targets = _tiny_sequences()
    out = ri.collect_dataset(loaded["model"], inputs, step_targets, 42, SOURCE_IDS["test"])
    h = {"scalars": out["scalars"]}
    ci = {c: i for i, c in enumerate(ri.COLS)}
    gc = eg.COLS
    g_arr = np.zeros((len(out["scalars"]), len(gc)))
    gcols = {c: i for i, c in enumerate(gc)}
    for col in eg.COLS:
        if col in ci:
            g_arr[:, gcols[col]] = out["scalars"][:, ci[col]]
    checks = ri.verify_against_experiment_g(h, g_arr, gc, "E_candidate")
    assert checks["passed"]
    g_bad = g_arr.copy()
    g_bad[0, gcols["dest"]] = (g_bad[0, gcols["dest"]] + 1) % 4
    with pytest.raises(RuntimeError):
        ri.verify_against_experiment_g(h, g_bad, gc, "E_candidate")
