# PLAN — Current To-Be-Done Work

**Last updated:** 2026-10-17
**Rule:** this file contains only the current to-be-done plan; it is rewritten after every agent task completion and is not a history or a research record (see `AGENTS.md`).

## Current plan

### 1. PI Decisions on Architectural Pivot & Milestone Roadmap

* **Context:** Comprehensive architecture-to-implementation gap audit completed and registered on 2026-10-17 (`docs/reports/architecture_gap_audit_and_milestone_proposal.md`). Audit established that 5 of 7 core state-machine mechanisms are absent from active code. Primitive modular addition 2x2 supervision grid is empirically closed. Phase 1.6 Stage 3 composed sequence benchmark recommended for de-prioritization.
* **Pending PI Decisions (all require explicit authorization before action):**
  1. *Roadmap Approval:* Authorize Milestone 1 (Speculative State Isolation & Transaction Gate Engine implementation and unit testing).
  2. *Task Domain Authorization:* Approve Constrained Graph Pathfinding as the target benchmark for end-to-end state machine validation (Milestone 2).
  3. *Stage 3 De-prioritization:* Confirm de-prioritizing Phase 1.6 Stage 3 composed sequence evaluation.
  4. *Firewall Confirmation:* Reaffirm single-evaluation firewall protecting held-out test split ($b \in \{0, 5\}$).
  5. *Runner Hardening & Exp L Drift:* Authorize adding artifact overwrite refusal checks to official runners and auditing Experiment L reproduction drift.
* **Status:** Awaiting PI direction.
* **Owner:** PI.

### 2. Milestone 1 — Speculative State Isolation & Transaction Gate Engine

* **Context:** Software engineering milestone implementing `TransactionCommitGate`, multi-candidate `StateContainer` buffering, and isolated rollback unit tests.
* **Pending task:** Implement and test Milestone 1 components upon explicit PI authorization.
* **Status:** Blocked pending PI decision on Item 1.
* **Owner:** Antigravity.

### 3. Pre-existing Unrelated Test Drift — Experiment L

* **Context:** `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails on this worktree. Confirmed pre-existing and unrelated to Phase 1.6 / cyclic models (Experiment L artifact dated 2026-10-07; fails even when unmodified HEAD model code is force-loaded). Unchanged by recent experiments.
* **Pending task:** Audit Experiment L artifact reproduction path and determine whether an amendment under `docs/DOCUMENTATION_POLICY.md` §4 is required.
* **Status:** Pending PI guidance.
* **Owner:** Antigravity / PI.

## Owner map

* **PI:** Direction and authorization on architecture roadmap, task selection, runner hardening, and test drift.
* **Antigravity:** Component implementation, test suites, and documentation governance upon explicit PI authorization.

## Canonical references

* `docs/reports/architecture_gap_audit_and_milestone_proposal.md` (authoritative architecture gap audit & roadmap)
* `docs/reports/exp_cyclic_persist_abl_01_report.md` (authoritative ablation report)
* `docs/reports/exp_cyclic_persist_01_report.md` (authoritative primary report)
* `docs/02_core_architecture.md` (target conceptual pipeline)
* `docs/03_development_roadmap.md` (development roadmap)
* `docs/RESEARCH_STATUS.md` (current validated research state)
* `docs/DOCUMENTATION_POLICY.md`; `AGENTS.md`
