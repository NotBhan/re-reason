# AGENTS.md — Repository Agent & Developer Instructions

RE:Reason is an experimental research project (candidate recurrent inference architecture), not a finalized architecture. Start with `PROJECT_PLAN.md` and the specification suite in `docs/`; the current to-be-done plan is maintained in `PLAN.md` (repository root). The project operates under the empirical cycle BUILD → MEASURE → ABLATE → INTERPRET → KEEP OR KILL → EXTEND.

## RESEARCH DOCUMENTATION POLICY

**Completed research documentation is historical and append-only.**
**Never silently rewrite or delete historical experiment results, conclusions, failed hypotheses, or numerical measurements.**
**New evidence, corrections, and reinterpretations must be appended as dated amendments or new experiment entries.**
**Maintain the current validated state separately from the historical research record.**

The full, canonical policy is `docs/DOCUMENTATION_POLICY.md`; read it before modifying any research documentation.

## Documentation layers (do not mix them)

* **Evidence (immutable after completion):** `docs/reports/*.md` — completed experiment reports. Additions only as dated appends (`## Amendment — YYYY-MM-DD`, etc.). Never rewrite results, numbers, decisions, or interpretations.
* **Chronology (append-only):** `docs/RESEARCH_LOG.md` — new entries go at the end; existing entries are never edited or reordered.
* **Current state (mutable synthesis):** `docs/RESEARCH_STATUS.md` — may be updated when evidence arrives, must cite reports, and must preserve the fact that earlier conclusions were different.
* **Immutability anchor:** `docs/reports/HISTORICAL_BASELINE.json` — SHA-256 of every completed report, enforced by `tests/test_documentation_governance.py`. If a report is intentionally amended (appended section only), update its hash **and** record the amendment in the ledger and the baseline's `history` array.
* **Specification suite:** `docs/01`–`06` may evolve with the active phase; historical claims inside them must not be silently rewritten.

## Working rules for agents

* Modify only the currently active phase unless explicitly instructed otherwise (`docs/05_engineering_and_implementation.md`); do not implement deferred mechanisms (Phases 2–10) prematurely.
* Never alter published numerical results, checkpoints, benchmark definitions, or experimental protocols as part of unrelated work.
* New experiment reports follow the required 19-section structure in the policy (§8) and record runner, analysis, tests, data artifacts, checkpoints (with hashes where relevant), figures, and exact reproduction commands — using repository-relative paths.
* After any documentation change, run: `pytest tests/test_documentation_governance.py -v` (and the full suite `pytest tests/`).
* Corrections are appended amendments, never silent edits; failed experiments and negative results are preserved, with rejected approaches recorded in the ledger and the status document.
* **Abbreviations and acronyms:** When an abbreviation or acronym is introduced in repository documentation, code comments, or research artifacts, add it to `docs/ABBREVIATIONS.md` with its expansion and a brief project context. Spell the term out on first use where practical, and clarify distinct meanings rather than overwriting an existing definition.
* Maintain the repository-global plan file `PLAN.md` and rewrite it after every agent task completion (rules in the section below).

## Global plan file (trackable, current plan only)

* **File:** `PLAN.md` in the repository root. It is a normal repository file — keep it in the working tree so it stays trackable; never keep the global plan in session scratch or machine-local agent plan folders.
* **Content:** strictly the current to-be-done plan — pending tasks, their status, owner, gating condition, and open decisions. It is not a history, not a research record, and not a status synthesis; do not duplicate `docs/RESEARCH_LOG.md` (append-only chronology) or `docs/RESEARCH_STATUS.md` (current validated state).
* **Update timing:** rewrite it after every agent task completion — drop completed items, keep still-pending ones, add newly planned work, and replace its contents with the updated current plan. Do not accumulate history; if nothing is pending, state that plainly.
* **Format:** keep it short — a title, a last-updated date, and a compact list of the current planned items, citing canonical repository documents (repository-relative paths) instead of restating them.
* **Enforcement:** `tests/test_documentation_governance.py` checks that `PLAN.md` exists, carries a `Last updated:` line, and contains no machine-specific absolute paths.
* **Git:** committable like any other repository file; this rule does not change the repository's Git-authorization requirements (no staging, commits, or history changes without explicit instruction).

## Protocol Rigor, Metric Precision, and Epistemic Guardrails

### 1. Exact Preregistered Thresholds
* Always retrieve quantitative thresholds, seed-count requirements, rejection boundaries, and advancement criteria directly from the controlling research specification (`docs/reports/phase_1_6_architectural_reformulation_proposal.md` or active proposal), rather than recalling them from conversational memory or informal roundoffs.
* Cite the authoritative document and section (and line references where helpful, noting that line numbers may shift).
* If a contradiction exists between specifications, runner scripts, results artifacts, or status documents, report and reconcile the discrepancy explicitly before taking action. Never select a convenient threshold or retroactively alter a decision gate.
* *Project Example:* In Phase 1.6 Stage 1 Track 1A, the authoritative threshold is `acc_val >= 0.950 across >= 4/5 seeds, with zero seeds < 0.250` (`docs/reports/phase_1_6_architectural_reformulation_proposal.md` §5.3, §6.1, §8 Table 1); informal approximations such as `0.9000` are strictly prohibited.

### 2. Strict Epistemic Classification
Maintain unambiguous distinctions across all statements and reports:
* **Verified implementation facts:** Directly established by source inspection, formal assertions, or deterministic unit tests (e.g., input dimensions, tensor shapes, target slice semantics).
* **Official experimental observations:** Derived strictly from the designated, hash-pinned or authoritative results artifact (e.g., metrics stored in `experiments/phase1_6_stage1_pilot_results.json`).
* **Exploratory observations:** Obtained from interactive diagnostic loops, timing benchmarks, or command executions outside the official preregistered artifact. These must never be promoted to or conflated with official pilot results.
* **Interpretations and hypotheses:** Proposed explanatory mechanisms (e.g., memorization overfitting, logit explosion). These must be explicitly labeled as hypotheses unless definitively established across all official runs.
* Always declare telemetry limitations explicitly (e.g., when an artifact logs validation metrics only, do not assert unlogged training-loss dynamics as official facts).

### 3. Cross-Track Metric Incommensurability
* Preserve the evaluation modality and scope of every result.
* An isolated component unit test (e.g., a parameter-free Givens rotation ALU verifying exact arithmetic on Track 1B) evaluates a mathematical component in isolation; it does not validate learned read, write, or gating networks and must never be cited as empirical proof of end-to-end neural sequence capability.
* A deterministic analytical reference (e.g., MSM) demonstrates algorithmic behavior by construction, not learned neural generalization.
* Results from disparate evaluation tracks must never be ranked against one another without an approved, scientifically justified comparison framework. Explicitly state both what an experiment establishes and what it does not establish.

### 4. Preregistered Protocol Integrity
* Strictly adhere to the controlling research proposal regarding candidate rosters, seed counts, hyperparameters, split boundaries, advancement gates, and contingency rules.
* Never silently prune candidates, add training stages, change evaluation thresholds, introduce extra runs, or alter the experimental design in response to intermediate or negative results.
* Distinguish clearly between:
  1. **Actions authorized by the registered protocol** (e.g., completing the planned feasibility pilot across all registered models).
  2. **Actions governed by registered contingency rules** (e.g., deterministic contingency ladders).
  3. **Actions requiring a formal protocol amendment** and explicit PI approval before execution.
* Preserve negative outcomes, failed gates, rejected approaches, and protocol deviations. Never modify code, hyperparameters, or benchmark definitions merely to produce a passing result.

### 5. Git Authorization and State Discipline
* Respect the latest explicit user authorization and current task boundaries for all Git operations.
* A subsequent explicit user command (e.g., "create a git commit") overrides earlier general prohibitions; when this occurs, execute the specific request and record the deviation in the audit log.
* Maintain strict separation between distinct permissions:
  - Modifying files
  - Staging files
  - Creating commits
  - Pushing to remote branches
  - Amending or rewriting Git history
  - Resetting or discarding working-tree changes
* Authorization for one action never implies authorization for another. Always inspect working-tree status prior to edits, preserve uncommitted user changes, and report the post-action Git state accurately.

### 6. Standard Engineering & Research Operating Practices
Incorporate these established successful practices as standard operating procedure:
* **Pre-Execution Audits:** Conduct independent, read-only audits before granting pre-implementation or pre-pilot readiness clearance.
* **Contracts First:** Establish mathematical formulations, exact parameter counts, tensor shapes, and evaluation boundaries before writing model or runner code.
* **Domain-Oriented Naming:** Use permanent, meaningful, domain-driven filenames (e.g., `src/re_reason/core/register_machine_models.py`) rather than temporary milestone names. When renaming modules, verify imports, exports, and stale references across the repository.
* **Dual-Track Verification:** Decouple deterministic analytical component tests from empirical, trained neural evaluations.
* **Validation Cadence:** Run focused unit tests and documentation-governance checks (`pytest tests/test_documentation_governance.py -v`) before and after consequential modifications.
* **Read-Only Diagnostics:** Investigate failures with deterministic forward passes, assertions, and read-only telemetry audits before modifying source code or consuming compute.
* **Strict Evaluation Firewalls:** Strictly firewall held-out test splits from model selection, early stopping, or diagnostic inspection. Ensure all experiment artifacts are reproducible from repository-relative paths.
* **Explicit Gate Progression:** Treat each experimental stage as an explicit gate requiring separate authorization; completing one stage never implies authorization for the next.
* **Auditable Handoff Reports:** Conclude every phase with an auditable report documenting confirmed facts, limitations, exact test results, governance deviations, and explicit next decisions.
