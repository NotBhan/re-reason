# RE:Reason — Research Documentation Policy (Permanent)

**Effective date:** 2026-10-05
**Status:** Canonical and permanent. This document governs how research documentation in this repository is written, corrected, and maintained.

> **CORE PRINCIPLE**
> **HISTORY IS APPENDED. CURRENT STATUS IS SYNTHESIZED. OLD RESULTS ARE NEVER SILENTLY REWRITTEN.**

---

## 0. Purpose and Scope

RE:Reason is an empirical research project whose conclusions have already been refined multiple times by later experiments (for example: terminal-only supervision → dense supervision → candidate/local supervision; branching considered → measured → not adopted; scalar gate analysis → shown insufficient by vector-level analysis). Preserving the actual trajectory of these conclusions is a scientific requirement, not an aesthetic one.

This policy applies to:

* every experiment report under `docs/reports/`,
* the chronological ledger `docs/RESEARCH_LOG.md`,
* the current-state synthesis `docs/RESEARCH_STATUS.md`,
* this policy document itself,
* agent/developer instruction files at the repository root.

It does **not** relax the project's other rules (see `docs/05_engineering_and_implementation.md` — *"AI coding agents may modify only the currently active phase unless explicitly instructed otherwise"* — and the Phase-2 exclusions in `docs/06_risks_research_questions_and_future.md`).

## 1. Documentation Layers (three separate purposes)

```
docs/
├── DOCUMENTATION_POLICY.md     ← permanent rules (this file; append amendments only)
├── RESEARCH_LOG.md             ← chronological append-only ledger (never reordered)
├── RESEARCH_STATUS.md          ← current validated state (mutable synthesis)
├── 01_project_overview.md      ← project specification suite (may evolve with the project)
├── 02_core_architecture.md
├── 03_development_roadmap.md
├── 04_experiments_and_evaluation.md
├── 05_engineering_and_implementation.md
├── 06_risks_research_questions_and_future.md
└── reports/
    ├── HISTORICAL_BASELINE.json    ← SHA-256 immutability anchor for Layer A
    └── <experiment reports>        ← EVIDENCE: append-only historical record
```

| Layer | Artifact | Editing rule |
|---|---|---|
| **A — Historical research record (evidence)** | `docs/reports/*.md` (and their machine-readable artifacts) | **Immutable after completion.** Additions occur only as dated appends (amendments, cross-experiment notes). Never rewrite results, conclusions, or numbers. |
| **B — Chronology** | `docs/RESEARCH_LOG.md` | **Append-only.** New entries go at the end. Existing entries are never edited, reordered, or deleted. Later interpretation is a *new* entry. |
| **C — Current state (synthesis)** | `docs/RESEARCH_STATUS.md` | Mutable as evidence arrives, but must preserve the record that earlier conclusions were different, and every claim must cite its underlying report(s). |

The specification suite (`docs/01`–`06`) may evolve with the project's active phase and roadmap; however, historical claims inside it must not be silently rewritten. When a project-level document materially changes a previously published interpretation, the change is recorded as a new `RESEARCH_LOG.md` entry (with the reason), and the log preserves what the earlier interpretation was.

## 2. Historical Immutability of Completed Reports

Completed experiment reports are historical records. Once a report is marked completed:

* do **not** rewrite its results, statistics, decision, or interpretation section because later work changes the picture;
* do **not** adjust it to appear that it predicted a later result;
* do **not** collapse multiple experiments into one rewritten narrative;
* do **not** delete negative results or falsified hypotheses.

**Worked example (this repository).**

*Bad:*

> Experiment G's report is edited after Experiment H so that it reads "G proved coordinate-wise gating."

*Correct:*

> Experiment G's report remains unchanged and says it localized the residual as mixed (dilution + blend/readout signatures, classification C). Experiment H's report additionally states: *"The scalar interpolation path is monotone … the mean-gate scalarization collapses accuracy … H5 strongly supported: gate-vector structure matters."* The sharpening is recorded in `docs/reports/experiment_h_lambda_interpolation.md` and in the log — never by editing G.

## 3. Append-Only Research Record

New scientific information is appended, never retrofitted into old narratives. Preferred forms:

* `## Amendment — YYYY-MM-DD` (correction or clarification of a completed document)
* `## Cross-Experiment Interpretation — YYYY-MM-DD`
* `## Follow-up Result — YYYY-MM-DD`
* a new dated entry in `docs/RESEARCH_LOG.md`
* a new experiment report under `docs/reports/`

Appends go **at the end** of the affected document. Do not insert new interpretation into the middle of an old completed report to make it read better.

## 4. Corrections Are Append-Only Amendments

If an earlier document contains an error, **do not silently edit the old conclusion**. Append a correction/amendment (at the end of the affected report and/or as a new ledger entry) that states:

1. **Date**
2. **Document / experiment affected**
3. **Exact issue**
4. **Original statement** (quoted)
5. **Corrected statement**
6. **Evidence for the correction**
7. **Whether previously reported numerical results change** (yes/no; if yes, which)
8. **Whether downstream conclusions change** (yes/no; if yes, which)

Corrections themselves are append-only: they must not be applied by editing historical text inside the original report. If a corrected number is needed for future work, state both values in the amendment ("originally reported X (retained in the historical section); corrected value Y; reason: …").

**Historical precedent for this workflow:** `docs/reports/parameterization_isolation_audit.md` records a FLOP measurement bug in earlier reports and its corrected accounting. This amendment-style handling is now the required pattern for all future corrections.

## 5. Failed Hypotheses and Negative Results Are Preserved

A falsified hypothesis is research evidence. Never delete or soften:

* failed experiments, negative results, abandoned mechanisms;
* rejected objectives, rejected architectures, rejected training methods;
* dead ends and superseded interpretations.

Record them as completed research outcomes with the reason for rejection. The log's `Decision` fields and `docs/RESEARCH_STATUS.md`'s *Rejected approaches* section exist precisely to keep these outcomes visible.

## 6. Numerical Immutability

Once a result is published in a completed report, do not silently change: accuracy, loss, FLOPs, parameter counts, timings, p-values, confidence intervals, per-seed results, dataset sizes, checkpoint identifiers, or hashes.

If a bug is discovered, follow §4 and record:

* original result (retained),
* discovered issue,
* corrected methodology,
* corrected result,
* impact on conclusions (or an explicit statement that the original result remains valid).

Machine-readable artifacts (`experiments/*.json`, `*.npz`) referenced by a completed report are part of the record: regenerate only via the documented reproduction command, and if regeneration changes a published value, treat it as a correction (§4).

## 7. Experiment Numbering

* Use the next available identifier; never reuse one.
* Never rename a completed experiment for aesthetics.
* Repetitions due to bugs or protocol problems get a new identifier or an explicit `correction`/`reproduction` label.
* A rerun that is scientifically identical and only verifies reproducibility is documented as a **verification/reproduction**, not as a replacement of the original experiment.

Current sequence: … Experiment G (`docs/reports/experiment_g_add_commit_gap.md`) → Experiment H (`docs/reports/experiment_h_lambda_interpolation.md`) → next available is **Experiment I**.

## 8. Required Structure of New Experiment Reports

Every new report under `docs/reports/` contains, at minimum:

1. Date · 2. Status · 3. Research question · 4. Existing baseline · 5. Hypothesis ·
6. What is frozen · 7. What changed · 8. Dataset/protocol · 9. Experimental procedure ·
10. Results · 11. Statistical analysis · 12. Interpretation ·
13. What the experiment DOES establish · 14. What it DOES NOT establish ·
15. Decision · 16. Next experiment · 17. Reproduction commands · 18. Artifact list ·
19. Scientific guardrails.

The distinction between **empirical observation**, **supported interpretation**, **causal claim**, and **unresolved hypothesis** must remain explicit in every report. Causal language is permitted only when the experimental design supports it (e.g., interventional controls); measurement-only and correlational results never receive causal phrasing.

## 9. Code and Data Traceability

Each experiment report identifies, using **repository-relative paths**:

* runner script(s) and analysis script(s);
* tests covering the pipeline;
* machine-readable results (`experiments/*.json`, `*.npz`);
* checkpoints used, with SHA-256 hashes where relevant;
* generated figures (`docs/reports/figures/…`);
* the exact reproduction command(s).

No report may reference an ephemeral result without recording where the reproducible artifact lives. Machine-specific absolute paths must not appear in new documentation (legacy pre-policy reports containing machine-specific absolute links are preserved as-is under §2).

## 10. Decisions Are Documented

When a research decision is made (adopt / reject / freeze / defer), record:

* **Decision**
* **Evidence**
* **Alternatives considered**
* **Why rejected / why chosen**
* **Consequence**
* **What remains unresolved**

Rejected approaches are part of the record and must remain discoverable.

## 11. Never Retroactively Promote a Hypothesis

If an experiment weakens or falsifies an earlier hypothesis, the earlier hypothesis stays where it was written. Support is updated by appending new interpretation — e.g. `docs/reports/experiment_h_lambda_interpolation.md` states that Experiment H reduces the adequacy of the scalar mean-gate description used as a proxy in Experiment G; it does not edit G, and G is not rewritten to have always anticipated H.

## 12. No Duplication of Results Without Context

High-level documents summarize; they do not copy large tables out of experiment reports. Prefer concise statements with references, e.g.:

> Candidate→commit residual: see `docs/reports/experiment_g_add_commit_gap.md`.

The detailed numerical record lives in the experiment report plus its machine-readable artifact.

## 13. Documentation Workflow After Every Experiment

1. Create the new experiment report (structure per §8).
2. Preserve all historical reports (Layer A untouched).
3. Append a new entry to `docs/RESEARCH_LOG.md`.
4. Update `docs/RESEARCH_STATUS.md` with the current validated state.
5. If the result changes an earlier interpretation, append an amendment / cross-experiment synthesis **to the relevant document** (§3) — never edit the earlier result section.
6. Update roadmap / open-question documentation only where appropriate, preserving historical claims.
7. Record all new artifacts and reproduction commands in the report.
8. Run the documentation consistency checks (§14).
9. Verify that no completed report was silently rewritten (baseline hash check, §14).

## 14. Documentation Consistency Checks (enforced)

`tests/test_documentation_governance.py` enforces this policy:

* required governance files exist and contain the mandated sections;
* every report in `docs/reports/*.md` is referenced by `docs/RESEARCH_LOG.md`;
* every `RESEARCH_LOG.md` entry carries its required fields, with non-decreasing dates;
* no machine-specific absolute paths appear in governance documents;
* **`docs/reports/HISTORICAL_BASELINE.json`** stores the SHA-256 of every completed report. A mismatch fails the check until the maintainer either reverts the change or (for an intentional appended amendment) updates the hash **and** records the amendment in the log and the baseline's `history` array.

The baseline is the enforcement mechanism for §2/§6: reports cannot be silently rewritten without tripping a test.

## 15. Research Synthesis Rules (`docs/RESEARCH_STATUS.md`)

The status document is a synthesis, not a history rewrite. It must clearly distinguish:

* **Current validated state** (each claim cited to its report),
* **Historical progression** (how the current state was reached, including superseded conclusions),
* **Rejected approaches** (with reasons),
* **Frozen components**,
* **Open questions**,
* **Active / next experiment**.

It must never present current conclusions as though they were always known, and it must never erase the fact that earlier conclusions were different.

## 16. Project-Specific Scientific Guardrails

All documentation preserves the established RE:Reason guardrails:

* Do not claim novelty merely from combining known mechanisms (see the Model E audit's Category B classification).
* Do not claim AGI / general reasoning capability from Phase-1 results.
* Do not confuse parameter efficiency with intrinsic architectural superiority.
* Do not describe learned gates as symbolic memory addresses without causal evidence.
* Do not confuse probe decodability with exact state preservation.
* Do not confuse transition-level oracle results with sequence-level capability.
* Do not use scalar gate summaries as descriptions of the actual vector-gated mechanism when vector structure matters (Experiment H, H5).
* Do not call a measurement-based localization causal proof.
* Do not promote a candidate architecture to "final" solely because it currently wins an ablation.
* Preserve negative results.

## 17. Permanent Instruction for Future Agents

The concise version of this policy is placed in the repository-level agent/instruction file (`AGENTS.md` at the repository root). If that file gains other content, this rule is retained or extended — never replaced by a weaker policy:

> **RESEARCH DOCUMENTATION POLICY:**
> Completed research documentation is historical and append-only.
> Never silently rewrite or delete historical experiment results, conclusions, failed hypotheses, or numerical measurements.
> New evidence, corrections, and reinterpretations must be appended as dated amendments or new experiment entries.
> Maintain the current validated state separately from the historical research record.

## 18. Amending This Policy

This policy is itself append-only. Changes are made by appending dated sections (e.g., `## Amendment — YYYY-MM-DD`) rather than rewriting the rules above. If a rule is superseded, the superseding section states what it replaces and why.
