# PLAN — Current To-Be-Done Work

**Last updated:** 2026-10-16
**Rule:** this file contains only the current to-be-done plan; it is rewritten after every agent task completion and is not a history or a research record (see `AGENTS.md`).

## Current plan

### 1. EXP-CYCLIC-PERSIST-01 — Stage 2 Official Experiment Authorization Gate

* **Context:** Stage 1 (runner implementation in `experiments/run_exp_cyclic_persist_01.py`, 26 focused unit tests in `tests/test_exp_cyclic_persist_01.py`, and the 100-update seed-42 smoke test in `experiments/exp_cyclic_persist_01_SMOKE_ONLY_seed42.json`) is complete, independently audited, and verified. Loss decomposition equality, gradient flow across `mlp_read`/`mlp_write`/`mlp_gate`, parameter invariance (219,018), and protected split firewall (`queries = 0`) are fully verified.
* **Pending task:** PI authorization decision for Stage 2 (official five-seed run across seeds 42–46 for 3,000 updates each via `--stage2-authorized`, serializing `experiments/exp_cyclic_persist_01_results.json`).
* **Status:** Stage 1 complete. Stage 2 remains strictly unauthorized pending separate explicit PI authorization.
* **Owner:** PI (authorization decision) / DeepSeek (execution upon authorization).

### 2. Pre-existing Unrelated Test Drift — Experiment L

* **Context:** `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails on this worktree. Confirmed pre-existing and unrelated to Phase 1.6 / cyclic models (Experiment L artifact dated 2026-10-07; fails even when unmodified HEAD model code is force-loaded).
* **Pending task:** Audit Experiment L artifact reproduction path and determine whether an amendment under `docs/DOCUMENTATION_POLICY.md` §4 is required.
* **Status:** Pending PI guidance.
* **Owner:** Antigravity / PI.

## Owner map

* **Antigravity:** Research governance, audit of Stage 1 artifacts, independent review of Stage 2 outcomes upon completion; investigation of Experiment L test drift.
* **DeepSeek:** Stage 2 execution (on hold pending explicit PI authorization).

## Canonical references

* `docs/research/exp_cyclic_persist_01_specification.md` (authoritative protocol specification)
* `experiments/run_exp_cyclic_persist_01.py` (Stage 1 runner)
* `tests/test_exp_cyclic_persist_01.py` (Stage 1 unit tests)
* `experiments/exp_cyclic_persist_01_SMOKE_ONLY_seed42.json` (Stage 1 smoke artifact)
* `docs/reports/exp_cyclic_stepwise_01_report.md` (predecessor STEPWISE experiment report)
* `experiments/exp_cyclic_stepwise_01_results.json` (predecessor STEPWISE official artifact)
* `docs/reports/exp_cyclic_add_01_report.md` (predecessor ADD baseline experiment report)
* `docs/reports/HISTORICAL_BASELINE.json` (Layer A baseline registry)
* `docs/RESEARCH_LOG.md`; `docs/RESEARCH_STATUS.md`; `docs/DOCUMENTATION_POLICY.md`; `AGENTS.md`
