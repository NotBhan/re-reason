# PLAN — Current To-Be-Done Work

**Last updated:** 2026-10-16
**Rule:** this file contains only the current to-be-done plan; it is rewritten after every agent task completion and is not a history or a research record (see `AGENTS.md`).

## Current plan

### EXP-CYCLIC-STEPWISE-01 — Stage 2 (blocked, awaiting explicit PI authorization)

* **Task:** Official five-seed experiment (seeds 42–46, 3,000 updates, evaluation cadence 50) for the single registered intervention — auxiliary supervision of intermediate destination values in `CyclicMGUPhase16Model`, auxiliary coefficient fixed at 1.0, model parameters invariant at 219,018.
* **Status:** Blocked awaiting explicit PI authorization. The final preregistration audit and representation metric bounds verification are complete: (1) cosine representation metrics ($C_{\text{read}}, S_{\text{context}}, C_{\text{ALU}}$) accept all finite values in $[-1.0, 1.0]$ including negative values (where negative $C_{\text{read}}$ and $C_{\text{ALU}}$ produce `REP_FAILED`), reject values outside $[-1.0, 1.0]$ as `IMPLEMENTATION_INVALID`, and are clamped in `cosine_similarity`; (2) validation accuracies are verified in $[0.0, 1.0]$ and losses are verified finite and non-negative in runner checks, checkpoint selection, and classifier; (3) all 12 behavioral/representation outcomes are distinctly classified with strict epistemic discipline; and (4) all 52 focused deterministic unit and documentation-governance tests pass.
* **Gating condition:** PI review of the Stage 1 report and finalized preregistration contract, followed by separate, explicit Stage 2 authorization. No training runs or official results artifact creation may proceed before authorization.
* **Stage 2 deliverable:** the official results artifact at `experiments/exp_cyclic_stepwise_01_results.json`, which must not be created or populated before authorization.
* **Operational restrictions:** single intervention only; protected TEST split (operand pairs with b in {0, 5}) firewalled; no Git commits, pushes, resets, or history changes.

### Open decision — pre-existing unrelated test failure

* `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails on the current worktree. Confirmed pre-existing and unrelated: it also fails when the unmodified HEAD revision of `src/re_reason/core/register_machine_models.py` is force-loaded, and it references no file touched by the Stage 1 task.
* **Decision needed:** treat as historical artifact drift (Experiment L artifact dated 2026-10-07) requiring a dated correction/amendment under `docs/DOCUMENTATION_POLICY.md` §4, or investigate the reproduction path. Owner: Antigravity / PI.

### Owner map

* **DeepSeek:** Stage 1 implementation, focused tests, smoke test (complete); five-seed Stage 2 execution only upon separate explicit authorization.
* **Antigravity:** specification ownership, post-run verification, permanent research documentation (`docs/RESEARCH_LOG.md`, `docs/RESEARCH_STATUS.md`, `docs/reports/`).

### Canonical references

* `docs/research/exp_cyclic_stepwise_01_specification.md` (authoritative experiment specification with closed 3x4 decision matrix and aggregation rule)
* `AGENTS.md`; `docs/DOCUMENTATION_POLICY.md`
* `experiments/run_exp_cyclic_stepwise_01.py`; `tests/test_exp_cyclic_stepwise_01.py`
* `experiments/exp_cyclic_stepwise_01_SMOKE_ONLY_seed42.json` (Stage 1 smoke telemetry only — explicitly not official results)
* `experiments/run_exp_cyclic_add_01.py`; `experiments/exp_cyclic_add_01_results.json` (historical predecessor EXP-CYCLIC-ADD-01)
* `docs/reports/phase_1_6_architectural_reformulation_proposal.md`
