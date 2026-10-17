# PLAN — Current To-Be-Done Work

**Last updated:** 2026-10-17
**Rule:** this file contains strictly the current to-be-done plan; it is rewritten after every agent task completion and is not a history or a research record (see `AGENTS.md`).

## Current plan

### 1. Milestone 3B Stage A — Baseline Models & Behavioral-Cloning Infrastructure: implementation complete; independent audit pending

* **Context:** Authorized and executed 2026-10-17 under the PI work order "Milestone 3B Stage A: Baseline Implementation and Unit Tests". Engineering only: no training, optimizer steps, pilot runs, partition inference, or benchmark comparisons.
* **Deliverables:**
  * `src/re_reason/core/pathfinding_baselines.py` — `UnGatedRecurrentBaseline`, `ActionMaskingRecurrentBaseline`, and `TwoRolloutRecurrentBaseline` (each exactly 171,660 parameters, sharing the Milestone 3A architecture), plus deterministic rollouts, the protocol trajectory-selection rule, and static horizons (24 / 26+27; fixed τ = 1.0).
  * `src/re_reason/training/behavioral_cloning.py` — BFS demonstration construction (unsolved episodes skipped defensively), teacher-forced unrolling, transition-weighted masked batch loss, deterministic epoch batching, and a loader hard-restricted to the `train` partition.
  * `src/re_reason/utils/rng_namespaces.py` — paired training seeds and the partition-scoped isolated rollout-seed formula (validation offset 0; test offset 200000).
  * `experiments/run_phase3b_training.py` — inert, authorization-gated training runner with default overwrite refusal, explicit `--allow-overwrite-authorized`, unique timestamped run paths, and atomic artifact writes; no evaluation path or held-out access.
  * Additive integration points: `ActionConditionedBrancher.advance_states` (per-row teacher-forced core advance) and `generate_pathfinding_partition` (single-partition generation so training never materializes held-out partitions).
  * Focused suites: `tests/test_pathfinding_baselines.py`, `tests/test_behavioral_cloning.py`, `tests/test_phase3b_rng.py`, `tests/test_phase3b_runner.py`.
* **Verification (2026-10-17):** Stage A focused suites green on CPU; Milestone 1, 2, and 3A suites and documentation governance green (170 passed, 27 skipped across the authorized battery; all skips are pre-existing CPU-host CUDA parametrizations).
* **Pending tasks:** (1) independent Antigravity audit of the Stage A implementation, parameter accounting, BC masking/gradient contracts, and runner safeguards; (2) separate explicit PI authorization for Stage B (training execution).
* **Gating:** Training, optimizer steps, pilot runs, validation inference, held-out `test` evaluation, benchmark comparisons, Stage C, and all research claims remain unauthorized pending audit and separate PI authorization.
* **Owner:** Antigravity (audit) / PI (Stage B authorization).

### 2. Remaining Project-Level Decisions on Architecture Pivot & Governance

* **Resolved:** Stage A implementation authorization; three-comparator roster (un-gated, action-masking, two-rollout); additive `restore_checkpoint` API; BC loss/masking contract; fixed temperature; RNG namespaces; +0.10 margin and validation-only 5-to-10 seed expansion rule (all ratified in `docs/research/pre_3b_experimental_protocol.md`).
* **Still pending (each requires explicit PI direction before action):**
  1. *Stage 3 De-prioritization:* Confirm de-prioritizing Phase 1.6 Stage 3 composed sequence evaluation.
  2. *Firewall Confirmation:* Reaffirm the single-evaluation firewalls protecting the modular-addition held-out split ($b \in \{0, 5\}$) and the pathfinding held-out `test` partition.
  3. *Runner Hardening:* Authorize artifact overwrite refusal checks in the remaining official experiment runners (the new Milestone 3B training runner already implements them).
  4. *Experiment L Reproduction-Drift Audit:* Authorize the narrow audit scope for the pre-existing `tests/test_experiment_l.py` drift (memo recommendation: defer until after Milestone 3B).
* **Status:** Awaiting PI direction.
* **Owner:** PI.

### 3. Pre-existing Unrelated Test Drift — Experiment L

* **Context:** `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails on this worktree. Confirmed pre-existing and unrelated to Phase 1.6, Milestones 1–3A, or Milestone 3B Stage A (Experiment L artifact dated 2026-10-07; checkpoint hashes remain intact). Unchanged by recent work.
* **Pending task:** Execute the memo's narrow audit (environment diff, channel-specific error distribution, split determinism, governance pathway) and determine whether an amendment under `docs/DOCUMENTATION_POLICY.md` §4 is required.
* **Status:** Pending PI guidance and authorization.
* **Owner:** Antigravity / PI.

## Owner map

* **PI:** Stage B authorization, Stage 3 de-prioritization, firewall re-affirmation, remaining runner hardening, and Experiment L guidance.
* **Antigravity:** Independent audit of Milestone 3B Stage A; implementation and experimentation strictly deferred pending explicit PI authorization.

## Canonical references

* `docs/research/pre_3b_experimental_protocol.md` (authoritative pre-3B experimental protocol)
* `src/re_reason/core/pathfinding_baselines.py`; `src/re_reason/training/behavioral_cloning.py`; `src/re_reason/utils/rng_namespaces.py` (Milestone 3B Stage A)
* `experiments/run_phase3b_training.py` (authorization-gated training runner infrastructure)
* `tests/test_pathfinding_baselines.py`; `tests/test_behavioral_cloning.py`; `tests/test_phase3b_rng.py`; `tests/test_phase3b_runner.py` (Stage A focused suites)
* `src/re_reason/hypothesis/pathfinding_branching.py`; `src/re_reason/hypothesis/two_phase_orchestrator.py` (Milestone 3A)
* `src/re_reason/state/container.py`; `src/re_reason/state/speculative.py`; `src/re_reason/verification/transaction_gate.py` (Milestone 1)
* `src/re_reason/benchmarks/pathfinding_benchmark.py`; `src/re_reason/verification/pathfinding_verifier.py` (Milestone 2)
* `docs/reports/architecture_gap_audit_and_milestone_proposal.md` (authoritative architecture gap audit & roadmap)
* `docs/DOCUMENTATION_POLICY.md`; `AGENTS.md`
