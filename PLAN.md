# PLAN — Current To-Be-Done Work

**Last updated:** 2026-10-16
**Rule:** this file contains only the current to-be-done plan; it is rewritten after every agent task completion and is not a history or a research record (see `AGENTS.md`).

## Current plan

### 1. EXP-CYCLIC-PERSIST-ABL-01 — Stage 2 Authorization Review

* **Context:** Authoritative experiment protocol approved and frozen by PI on 2026-10-16 in `docs/research/exp_cyclic_persist_abl_01_specification.md` with Option 1 ($\lambda_{\text{persist}} = 0.5$; $\mathcal{L}_{\text{total}}^{\text{abl}} = \mathcal{L}_{\text{terminal}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$). Stage 1 execution (runner, unit tests, 100-update seed-42 smoke run) completed and verified.
* **Current State:**
  1. Runner implemented: `experiments/run_exp_cyclic_persist_abl_01.py`.
  2. Focused unit tests implemented: `tests/test_exp_cyclic_persist_abl_01.py` (26/26 passing).
  3. Smoke run completed and verified: `experiments/exp_cyclic_persist_abl_01_SMOKE_ONLY_seed42.json`.
  4. Trainable parameters invariant at 219,018; loss decomposition and finite gradient flow verified; protected TEST partition intact (`queries = 0`).
* **Gating:** Stage 2 official five-seed training remains strictly unauthorized pending separate PI review and explicit authorization.
* **Owner:** PI (Stage 2 decision) / Antigravity (Stage 2 execution upon authorization).

### 2. Candidate 2 Post-Progression Next Steps (Phase 1.6)

* **Context:** `EXP-CYCLIC-PERSIST-01` achieved `BEHAVIORAL_PASS_CANONICAL_FULL` on `VAL-SEEN` ($acc_{\text{val}} = 0.9875$; asymmetry resolved at $\bar{C}_{\text{read}}(r0) = 0.8879$ and $\bar{C}_{\text{read}}(r1) = 0.9303$).
* **Pending PI Decisions (all require explicit authorization before action):**
  1. *Phase 1.6 Stage 3 Composed Sequence Benchmark Formulation:* Whether to formulate the composed sequence evaluation protocol for Candidate 2 (`CyclicMGUPhase16Model`).
  2. *Protected TEST-HELDOUT Evaluation:* Whether and when to evaluate the held-out operand partition ($b \in \{0, 5\}$) under single-evaluation firewall governance.
* **Status:** Awaiting PI direction.
* **Owner:** PI.

### 3. Pre-existing Unrelated Test Drift — Experiment L

* **Context:** `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails on this worktree. Confirmed pre-existing and unrelated to Phase 1.6 / cyclic models (Experiment L artifact dated 2026-10-07; fails even when unmodified HEAD model code is force-loaded). Unchanged by recent experiments.
* **Pending task:** Audit Experiment L artifact reproduction path and determine whether an amendment under `docs/DOCUMENTATION_POLICY.md` §4 is required.
* **Status:** Pending PI guidance.
* **Owner:** Antigravity / PI.

## Owner map

* **PI:** Direction and authorization on EXP-CYCLIC-PERSIST-ABL-01 proposal review, Stage 1 execution clearance, Stage 3 composed sequence evaluation, or protected TEST evaluation; Experiment L guidance.
* **Antigravity:** Protocol specification, focused unit test implementation (when authorized), smoke execution (when authorized), and governance audits.

## Canonical references

* `docs/research/exp_cyclic_persist_abl_01_specification.md` (proposed ablation specification)
* `docs/reports/exp_cyclic_persist_01_report.md` (authoritative primary experiment report)
* `experiments/exp_cyclic_persist_01_results.json` (official Stage 2 primary artifact)
* `docs/research/exp_cyclic_persist_01_specification.md` (primary experiment specification)
* `docs/reports/exp_cyclic_stepwise_01_report.md`; `experiments/exp_cyclic_stepwise_01_results.json` (predecessor STEPWISE)
* `docs/reports/HISTORICAL_BASELINE.json`; `docs/RESEARCH_LOG.md`; `docs/RESEARCH_STATUS.md`; `docs/DOCUMENTATION_POLICY.md`; `AGENTS.md`
