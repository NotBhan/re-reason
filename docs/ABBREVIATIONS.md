# Abbreviations and Glossary

**Last updated:** 2026-10-18
**Status:** Curated registry. Maintenance rules are in `AGENTS.md` ("Working rules for agents").

This file records abbreviations, acronyms, and initialisms established in this repository, with their expansions and brief project context. It is curated, not exhaustive; every expansion below is grounded in the cited repository source. Ordinary all-caps prose, opcodes, variable names, stage or seed labels, and file extensions are intentionally excluded.

## Registered terms

- **COW** — *copy-on-write*. Memory-management pattern. Named in this repository to distinguish it from what is actually implemented: RE:Reason does not claim a hardware-level COW mechanism; state isolation is described as "isolated tensor cloning", "state snapshot", or "speculative state isolation" (`docs/02_core_architecture.md`; `docs/05_engineering_and_implementation.md`; `README.md`; `src/re_reason/state/container.py`).
- **ECE** — *Expected Calibration Error*. Verifier metric: bin-wise deviation between verifier confidence scores and true transition validity (`docs/04_experiments_and_evaluation.md`, §4.5).
- **EM** — *exact match*. Primary task metric, also written "Exact Match Accuracy"; e.g. "T=10 EM" denotes exact match at step T=10 (`docs/04_experiments_and_evaluation.md`, §4.1; `README.md`; `docs/RESEARCH_STATUS.md`).
- **FAR** — *False Acceptance Rate*. Verifier metric: invalid candidates incorrectly accepted, over all invalid candidates (`docs/04_experiments_and_evaluation.md`, §4.5).
- **FLOPs** — *floating-point operations*. Compute-accounting quantity ("Inference FLOPs"); compute-parity comparisons and the corrected FLOP-accounting amendment depend on this metric (`docs/04_experiments_and_evaluation.md`, §4.2; `docs/DOCUMENTATION_POLICY.md`, §4).
- **FRR** — *False Rejection Rate*. Verifier metric: valid candidates incorrectly rejected, over all valid candidates (`docs/04_experiments_and_evaluation.md`, §4.5).
- **MGU** — *Minimal Gated Unit* (Zhou et al., 2016). Gated recurrent unit family; Model E is classified as an "Update-Gate RNN / Minimal Gated Unit (MGU) family architecture" with no novelty claim (`README.md`; `docs/reports/model_e_novelty_and_mechanism_audit.md`).
- **MSM** — *Modular Slot Machine*. Phase 1.6 Candidate 3: a parameter-free deterministic analytical reference — exact by construction, never a learned-model result, and distinct from SCLM (`docs/reports/phase_1_6_architectural_reformulation_proposal.md`, §4.3; `src/re_reason/core/register_machine_models.py`; `docs/research/model_e_heldout_generalization_design.md`).
- **PI** — *Principal Investigator*. Governance role; "explicit PI approval" gates protocol amendments, stage progression, and execution authorization (`AGENTS.md`; `docs/RESEARCH_LOG.md`).
- **RNN** — *Recurrent Neural Network*. Recurrent architecture family referenced in the Model E novelty audit (`docs/reports/model_e_novelty_and_mechanism_audit.md`).
- **SCLM** — *Slot-Cyclic Latent Model* (provisional name). Four-cell Phase 1.6 candidate combining explicit register slots with a fixed cyclic arithmetic operation; implementation complete, no training executed as of 2026-10-18 (`docs/research/model_e_heldout_generalization_design.md`; `src/re_reason/core/sclm_models.py`; `docs/RESEARCH_LOG.md`).
- **SPI** — *State Pollution Index*. Diagnostic geometric metric (one minus cosine similarity between corrupted and clean-reference states); it measures latent displacement, not semantic state integrity (`docs/04_experiments_and_evaluation.md`, §4.6; `PROJECT_PLAN.md`).
- **UGRNN** — *Update-Gate RNN* (Collins, Sohl-Dickstein, & Sussillo, ICLR 2017). Recurrent unit family; Model E's recurrence is classified as a parameterization variant of UGRNN/MGU (`docs/reports/model_e_novelty_and_mechanism_audit.md`; `README.md`).

## Not registered (expansion not stated in repository sources)

The following abbreviations appear in repository artifacts but no repository source states their expansions. They are intentionally left undefined rather than guessed; register a sourced entry here when one is established:

`ALU`, `GELU`, `GRU`, `LSTM`, `MLP`, `TCN`.
