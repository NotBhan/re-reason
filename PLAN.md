# PLAN — Current To-Be-Done Work

**Last updated:** 2026-10-17
**Rule:** this file contains only the current to-be-done plan; it is rewritten after every agent task completion and is not a history or a research record (see `AGENTS.md`).

## Current plan

### 1. PI Decision & Authorization: Milestone 1 Acceptance & Milestone 2 Authorization

* **Context:** Milestone 1 implementation and independent Antigravity audit are complete (Verdict: PASS, 2026-10-17). `SpeculativeCandidateSet` (`src/re_reason/state/speculative.py`) and `TransactionCommitGate` (`src/re_reason/verification/transaction_gate.py`) verified against contracts, isolation invariants, and non-training tests (48 passed, 21 skipped on CPU).
* **Pending PI decision:** Accept Milestone 1 audit handoff and authorize next roadmap step (Milestone 2: Constrained Pathfinding Benchmark Suite).
* **Gating:** Milestone 2 implementation, model integration, neural branching, and any model training or experiment runners remain strictly unauthorized pending explicit PI instruction.
* **Owner:** PI.

### 2. Remaining PI Decisions on Architectural Pivot & Milestone Roadmap

* **Context:** `docs/reports/architecture_gap_audit_and_milestone_proposal.md` (2026-10-17). The Milestone 1 roadmap decision has been executed; the following remain open.
* **Still pending (each requires explicit PI authorization before action):**
  1. *Task Domain Authorization:* Approve Constrained Graph Pathfinding as the target benchmark for end-to-end state-machine validation (Milestone 2).
  2. *Stage 3 De-prioritization:* Confirm de-prioritizing Phase 1.6 Stage 3 composed sequence evaluation.
  3. *Firewall Confirmation:* Reaffirm the single-evaluation firewall protecting the held-out test split ($b \in \{0, 5\}$).
  4. *Runner Hardening & Exp L Drift:* Authorize artifact overwrite refusal checks in official runners and audit Experiment L reproduction drift (not covered by the Milestone 1 work order).
* **Status:** Awaiting PI direction.
* **Owner:** PI.

### 3. Pre-existing Unrelated Test Drift — Experiment L

* **Context:** `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails on this worktree. Confirmed pre-existing and unrelated to Phase 1.6 / cyclic models (Experiment L artifact dated 2026-10-07; fails even when unmodified HEAD model code is force-loaded). Unchanged by recent work.
* **Pending task:** Audit Experiment L artifact reproduction path and determine whether an amendment under `docs/DOCUMENTATION_POLICY.md` §4 is required.
* **Status:** Pending PI guidance.
* **Owner:** Antigravity / PI.

## Owner map

* **PI:** Next-milestone authorization, task-domain decision, Stage 3 de-prioritization, firewall confirmation, runner hardening and Experiment L guidance.
* **Antigravity:** Independent audit of the Milestone 1 implementation; subsequent implementation and governance work upon explicit PI authorization.

## Canonical references

* `docs/reports/architecture_gap_audit_and_milestone_proposal.md` (authoritative architecture gap audit & roadmap)
* `src/re_reason/state/speculative.py`; `src/re_reason/verification/transaction_gate.py` (Milestone 1 infrastructure)
* `tests/test_speculative_state.py`; `tests/test_transaction_gate.py` (Milestone 1 focused suites)
* `src/re_reason/state/container.py`; `src/re_reason/verification/base.py` (existing contracts built upon)
* `docs/reports/exp_cyclic_persist_abl_01_report.md` (authoritative ablation report)
* `docs/reports/exp_cyclic_persist_01_report.md` (authoritative primary report)
* `docs/02_core_architecture.md` (target conceptual pipeline)
* `docs/03_development_roadmap.md` (development roadmap)
* `docs/RESEARCH_STATUS.md` (current validated research state)
* `docs/DOCUMENTATION_POLICY.md`; `AGENTS.md`
