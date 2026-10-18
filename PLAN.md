# PLAN — Current To-Be-Done Work

**Last updated:** 2026-10-19
**Rule:** this file contains strictly the current to-be-done plan; it is rewritten after every agent task completion and is not a history or a research record (see `AGENTS.md`).

## Current plan

### 1. Milestone 3B — Final held-out test evaluation complete (single authorized pass); PI disposition pending

* **Context:** Authorized and executed 2026-10-17 under the PI Final Held-Out Test Evaluation work order (Option A). The five frozen full-training checkpoints (seeds 1–5) were evaluated on the canonical 50-episode pathfinding `test` partition across RE:Reason and the three predeclared comparators. Exactly one test pass was performed; no training, fine-tuning, tuning, seed expansion, test rerun, or checkpoint modification occurred.
* **Validation result (authoritative artifact `experiments/phase3b_validation_runs/phase3b_validation_20261017T135606Z/validation_results.json`):** RE:Reason mean GRR 0.528; Un-Gated 0.468; paired Δ **+0.060**, 95% CI [0.0424, 0.0776] → `practical_improvement_not_supported` (`positive_below_threshold`); expansion criterion not met.
* **Held-out test result (authoritative artifact `experiments/phase3b_test_runs/phase3b_test_20261017T142017Z/test_results.json`):** RE:Reason mean GRR 0.524; Un-Gated 0.468; paired Δ **+0.056**, 95% CI [0.0192, 0.0928] → `practical_improvement_not_supported` (`positive_below_threshold`). Action-Masking identical to RE:Reason on both partitions (Δ = 0.0000); Two-Rollout higher on both (validation 0.668, test 0.628; paired Δ −0.1400 / −0.1040, `negative_effect` in both). Relative patterns replicated across partitions; the +0.10 practical margin was not met on either partition.
* **Pending tasks:** (1) independent Antigravity audit of the held-out evaluation driver, artifact, recomputation evidence, and single-pass compliance; (2) PI decision on the disposition of the completed Milestone 3B results (registration of the outcome and any follow-up direction, e.g. report/amendments per `docs/DOCUMENTATION_POLICY.md`).
* **Gating:** No further training, tuning, seed expansion, test reruns, or follow-up experiments are authorized. The held-out `test` partition has now been consumed by its single authorized pass; no further test access is authorized. No scientific success is claimed.
* **Owner:** Antigravity (audit) / PI (disposition decision).

### 2. Remaining Project-Level Decisions on Architecture Pivot & Governance

* **Resolved:** Stage A/B/C execution authorizations; three-comparator roster; additive `restore_checkpoint` API; BC loss/masking contract; fixed temperature; RNG namespaces; +0.10 margin and validation-only 5-to-10 seed expansion rule; validation decision (expansion not triggered); single frozen held-out test pass (completed, margin not met).
* **Still pending (each requires explicit PI direction before action):**
  1. *Result disposition:* Decide how to register the completed Milestone 3B validation and held-out outcomes (report registration, amendments, or status updates per `docs/DOCUMENTATION_POLICY.md`).
  2. *Stage 3 De-prioritization:* Confirm de-prioritizing Phase 1.6 Stage 3 composed sequence evaluation.
  3. *Firewall Confirmation:* Reaffirm that the modular-addition held-out split ($b \in \{0, 5\}$) remains untouchable and that the pathfinding test partition is permanently frozen after its single pass.
  4. *Runner Hardening:* Authorize artifact overwrite refusal checks in the remaining official experiment runners (the Milestone 3B runners already implement them).
  5. *Experiment L Reproduction-Drift Audit:* Authorize the narrow audit scope for the pre-existing `tests/test_experiment_l.py` drift.
* **Status:** Awaiting PI direction.
* **Owner:** PI.

### 3. Pre-existing Unrelated Test Drift — Experiment L

* **Context:** `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails on this worktree. Confirmed pre-existing and unrelated to Phase 1.6, Milestones 1–3A, or Milestone 3B (Experiment L artifact dated 2026-10-07; checkpoint hashes intact). Unchanged by recent work.
* **Pending task:** Execute the narrow audit and determine whether an amendment under `docs/DOCUMENTATION_POLICY.md` §4 is required.
* **Status:** Pending PI guidance and authorization.
* **Owner:** Antigravity / PI.

### 4. Model E Held-Out Generalization — SCLM implementation committed; execution gates pending

* **Context:** The accepted design proposal is `docs/research/model_e_heldout_generalization_design.md`; its append-only protocol amendment and implementation are recorded in `docs/reports/phase_1_6_architectural_reformulation_proposal.md` and `docs/RESEARCH_LOG.md`. SCLM is a four-slot, 53-dim-per-slot latent model with fixed exact cyclic arithmetic and matched memory/arithmetic comparisons. Model E remains the preserved research baseline; SCLM is not yet trained or qualified for the goal system.
* **Pending tasks:** (1) Establish and approve the protected-set hash manifest before training; (2) run the registered split tests and authorized dataset generation; (3) obtain explicit authorization for and complete the profiling stop-gate, then the registered training campaign; (4) evaluate the proxy gate and, only if it passes, obtain authorization for the one-time protected test pass. Follow the phase amendment and firewall at every stage.
* **Status:** Protocol amendment and guarded implementation committed (`3db8183`; smoke-check scope correction `3908a46`). Implementation-stage tests are recorded in the research log; no official training, profiling, dataset campaign, checkpoint inspection, or held-out evaluation has occurred. The local runtime must be measured below 12 hours before the full campaign. The protected `b∈{0,5}` split remains sealed for one final pass only.
* **Owner:** PI (stage authorizations and protected-set manifest); engineering execution only after each required gate.

## Owner map

* **PI:** Disposition of Milestone 3B results, Stage 3 de-prioritization, firewall re-affirmation, remaining runner hardening, and Experiment L guidance.
* **Antigravity:** Independent audit of the held-out test evaluation; its evidence-only SCLM planning review is complete. Further research action is deferred pending the listed gates.

## Canonical references

* `docs/research/pre_3b_experimental_protocol.md` (authoritative pre-3B experimental protocol)
* `experiments/run_phase3b_test_evaluation.py`; `experiments/phase3b_test_runs/` (final held-out test driver and artifacts)
* `experiments/run_phase3b_validation.py`; `experiments/phase3b_validation_runs/` (Stage C validation driver and artifacts)
* `experiments/run_phase3b_training.py`; `experiments/phase3b_runs/` (Stage B training runner and checkpoints)
* `src/re_reason/core/pathfinding_baselines.py`; `src/re_reason/training/behavioral_cloning.py`; `src/re_reason/utils/rng_namespaces.py` (Milestone 3B implementation)
* `tests/test_pathfinding_baselines.py`; `tests/test_behavioral_cloning.py`; `tests/test_phase3b_rng.py`; `tests/test_phase3b_runner.py`; `tests/test_phase3b_validation.py`; `tests/test_phase3b_test_evaluation.py` (Milestone 3B suites)
* `src/re_reason/hypothesis/pathfinding_branching.py`; `src/re_reason/hypothesis/two_phase_orchestrator.py` (Milestone 3A)
* `src/re_reason/state/container.py`; `src/re_reason/state/speculative.py`; `src/re_reason/verification/transaction_gate.py` (Milestone 1)
* `src/re_reason/benchmarks/pathfinding_benchmark.py`; `src/re_reason/verification/pathfinding_verifier.py` (Milestone 2)
* `docs/reports/architecture_gap_audit_and_milestone_proposal.md` (authoritative architecture gap audit & roadmap)
* `docs/DOCUMENTATION_POLICY.md`; `AGENTS.md`
