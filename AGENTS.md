# AGENTS.md — Repository Agent & Developer Instructions

RE:Reason is an experimental research project (candidate recurrent inference architecture), not a finalized architecture. Start with `PROJECT_PLAN.md` and the specification suite in `docs/`. The project operates under the empirical cycle BUILD → MEASURE → ABLATE → INTERPRET → KEEP OR KILL → EXTEND.

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
