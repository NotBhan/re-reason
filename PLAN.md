# PLAN — Current To-Be-Done Work

**Last updated:** 2026-10-17
**Rule:** this file contains only the current to-be-done plan; it is rewritten after every agent task completion and is not a history or a research record (see `AGENTS.md`).

## Current plan

### 1. Milestone 3A — Neural Hypothesis Branching & Transaction-Safe Integration: audit complete (Verdict: PASS); pending PI acceptance

* **Context:** Authorized and executed 2026-10-17 under the PI work order "Milestone 3A: Neural Hypothesis Branching, Speculative State Isolation, and Transaction-Safe Integration". Non-training engineering only: no model training, no experiment execution, no baseline selection, no held-out evaluation.
* **Deliverables:** `src/re_reason/hypothesis/pathfinding_branching.py` (`ObservationEncoder`, `ActionConditionedBrancher`, typed `CandidateProposal`/`BranchProposal`; exactly 171,660 trainable parameters); `src/re_reason/hypothesis/two_phase_orchestrator.py` (`TwoPhaseTransitionOrchestrator`, `TransitionResult`; bounded `R_max = 3` proposal rounds, `T_max = 24` episode transitions, `B_max = 4` reserved for deferred backtracking, failure recovery via checkpoint restoration); additive `LatentState.restore_checkpoint` on the shared `BaseStateContainer` contract (`src/re_reason/state/container.py`); additive exports in `src/re_reason/hypothesis/__init__.py`; focused suites `tests/test_pathfinding_branching.py` and `tests/test_two_phase_orchestrator.py`, plus additive restoration tests in `tests/test_state.py` and a contract check in `tests/test_contracts.py`.
* **Audit Verdict (2026-10-17):** **PASS**. Independent code audit confirmed: exact 171,660 trainable parameter count; shared embedding and branch-core weights; strict public observation surface; independent verifier rejection; compensating checkpoint rollback across all failure boundaries; bounded retry and episode horizons; and full passing test suite (131 passed, 27 skipped on CPU host; 0 failures).
* **Pending tasks:** (1) PI formal acceptance of Milestone 3A; (2) PI decisions on pre-3B parameters (primary baseline selection, margin ratification, seed contingency).
* **Gating:** Milestone 3B, neural training, experiment runners, primary-baseline selection, statistical-margin ratification, seed-expansion rules, and held-out pathfinding test evaluation remain strictly unauthorized pending separate explicit PI authorization.
* **Owner:** PI (formal acceptance and next-action decision).

### 2. Remaining PI Decisions on Architectural Pivot & Milestone Roadmap

* **Resolved by the Milestone 3A work order:** approval of the additive `restore_checkpoint(tensor)` container API.
* **Still pending (each requires explicit PI authorization before action):**
  1. *Pre-3B Primary Baseline Selection:* designate the primary comparator from the distinguishable alternatives (Un-Gated GRU, Action-Masking GRU, Episode-Budget Matched GRU). The algorithmically equivalent sequential "Beam-3" formulation remains rejected and must not be reintroduced as a scientific control.
  2. *Pre-3B Margin Ratification:* ratify or adjust the provisional +0.10 absolute goal-reaching-rate improvement margin.
  3. *Pre-3B Inconclusive-Outcome Rule:* confirm whether inconclusive outcomes trigger expansion from five to ten seed pairs or remain inconclusive.
  4. *Stage 3 De-prioritization:* Confirm de-prioritizing Phase 1.6 Stage 3 composed sequence evaluation.
  5. *Firewall Confirmation:* Reaffirm the single-evaluation firewalls protecting the modular-addition held-out split ($b \in \{0, 5\}$) and the pathfinding held-out `test` partition.
  6. *Runner Hardening & Exp L Drift:* Authorize artifact overwrite refusal checks in official runners and audit Experiment L reproduction drift.
* **Status:** Awaiting PI direction.
* **Owner:** PI.

### 3. Pre-existing Unrelated Test Drift — Experiment L

* **Context:** `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails on this worktree. Confirmed pre-existing and unrelated to Phase 1.6 / cyclic models (Experiment L artifact dated 2026-10-07; fails even when unmodified HEAD model code is force-loaded). Unchanged by recent work.
* **Pending task:** Audit Experiment L artifact reproduction path and determine whether an amendment under `docs/DOCUMENTATION_POLICY.md` §4 is required.
* **Status:** Pending PI guidance.
* **Owner:** Antigravity / PI.

## Owner map

* **PI:** Next-milestone authorization, pre-3B baseline/margin/contingency decisions, Stage 3 de-prioritization, firewall confirmation, runner hardening and Experiment L guidance.
* **Antigravity:** Independent audit of the Milestone 3A implementation; subsequent implementation and governance upon explicit PI authorization.

## Canonical references

* `docs/reports/architecture_gap_audit_and_milestone_proposal.md` (authoritative architecture gap audit & roadmap)
* `src/re_reason/hypothesis/pathfinding_branching.py`; `src/re_reason/hypothesis/two_phase_orchestrator.py` (Milestone 3A architecture & orchestration)
* `tests/test_pathfinding_branching.py`; `tests/test_two_phase_orchestrator.py` (Milestone 3A focused suites)
* `src/re_reason/state/container.py` (checkpoint restoration contract); `src/re_reason/state/speculative.py`; `src/re_reason/verification/transaction_gate.py` (Milestone 1 infrastructure)
* `src/re_reason/benchmarks/pathfinding_benchmark.py`; `src/re_reason/verification/pathfinding_verifier.py` (Milestone 2 benchmark suite)
* `tests/test_pathfinding_benchmark.py`; `tests/test_pathfinding_verifier.py` (Milestone 2 focused suites)
* `docs/reports/exp_cyclic_persist_abl_01_report.md` (authoritative ablation report)
* `docs/reports/exp_cyclic_persist_01_report.md` (authoritative primary report)
* `docs/02_core_architecture.md` (target conceptual pipeline)
* `docs/03_development_roadmap.md` (development roadmap)
* `docs/RESEARCH_STATUS.md` (current validated research state)
* `docs/DOCUMENTATION_POLICY.md`; `AGENTS.md`
