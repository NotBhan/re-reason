# PLAN — Current To-Be-Done Work

**Last updated:** 2026-10-16
**Rule:** this file contains only the current to-be-done plan; it is rewritten after every agent task completion and is not a history or a research record (see `AGENTS.md`).

## Current plan

### 1. EXP-CYCLIC-PERSIST-01 — Stage 1 Implementation Authorization Gate

* **Context:** Authoritative experiment specification and protocol preregistration finalized in `docs/research/exp_cyclic_persist_01_specification.md`. The PI has approved Option 1 ($\lambda_{\text{dest}} = 1.0, \lambda_{\text{persist}} = 0.5$), providing equal effective weighting ($0.5$) across all three intermediate register assertions while preserving the individual destination gradient scale from `EXP-CYCLIC-STEPWISE-01`. Architecture remains invariant (219,018 trainable parameters, 0 new heads, shared readout query).
* **Pending task:** PI authorization for Stage 1 (DeepSeek implementation of `experiments/run_exp_cyclic_persist_01.py`, focused unit tests in `tests/test_exp_cyclic_persist_01.py`, and the 100-update seed-42 smoke test).
* **Status:** Protocol finalized and frozen. Stage 1 implementation, runner code changes, and smoke testing remain strictly unauthorized pending explicit PI approval.
* **Owner:** PI (authorization gate) / DeepSeek (implementation upon authorization).

### 2. Pre-existing Unrelated Test Drift — Experiment L

* **Context:** `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails on this worktree. Confirmed pre-existing and unrelated to Phase 1.6 / cyclic models (Experiment L artifact dated 2026-10-07; fails even when unmodified HEAD model code is force-loaded).
* **Pending task:** Audit Experiment L artifact reproduction path and determine whether an amendment under `docs/DOCUMENTATION_POLICY.md` §4 is required.
* **Status:** Pending PI guidance.
* **Owner:** Antigravity / PI.

## Owner map

* **Antigravity:** Research specification and proposal for `EXP-CYCLIC-PERSIST-01`; investigation of Experiment L test drift.
* **DeepSeek:** Implementation and execution (on hold; Stage 1 implementation pending PI authorization).

## Canonical references

* `docs/research/exp_cyclic_persist_01_specification.md` (authoritative protocol specification)
* `docs/reports/exp_cyclic_stepwise_01_report.md` (predecessor STEPWISE experiment report)
* `experiments/exp_cyclic_stepwise_01_results.json` (predecessor STEPWISE official artifact)
* `docs/reports/exp_cyclic_add_01_report.md` (predecessor ADD baseline experiment report)
* `docs/reports/HISTORICAL_BASELINE.json` (Layer A baseline registry)
* `docs/RESEARCH_LOG.md`; `docs/RESEARCH_STATUS.md`; `docs/DOCUMENTATION_POLICY.md`; `AGENTS.md`
