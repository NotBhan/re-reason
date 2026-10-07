# RE:Reason — Experiment Q: Candidate-Generation Origin / Recoverability

**Date:** October 7, 2026
**Status:** Completed — Measurement only (frozen checkpoints; no training of the model; no optimizer on model weights; no architecture/candidate/core/gate/commit/readout modification; no branching/verifier/oracle; no counterfactual rollout). The only fitted objects are analysis-only probes trained on the disjoint TRAIN split, tuned on VAL, and evaluated once on TEST; they never enter model inference. Classification: **Q-C — weak recoverability** for E and F.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned (hashes re-verified against G/L/M/N/O/P metadata)
**Data:** `experiments/experiment_q_candidate_origin.npz`, `experiments/experiment_q_candidate_origin_meta.json`, `experiments/experiment_q_results.json`
**Runners:** `experiments/run_experiment_q.py`, `experiments/analyze_experiment_q.py`, tests `tests/test_experiment_q.py`

---

## 1. Title / Date / Status

Experiment Q: Candidate-Generation Origin / Recoverability — Q-C (weak recoverability). Measurement-only, frozen; no intervention; no Phase 2.

## 2. Executive Answer

P established that the frozen model's sequence-level first-error budget is dominated by candidate-generation errors (Class A ~60% at T=10, ~67% at T=20). Q asks whether that dominant failure is a **candidate-representation/arithmetic limitation** or more consistent with **exposure/decoding/training alignment**, by measuring how much of the true written value is **recoverable from the frozen candidate vector `C_t`** on the transitions where the frozen candidate head is wrong.

**On the Class-A population (ADD transitions whose frozen candidate decode is wrong), the target is only weakly recoverable.** An analysis-only linear probe (fit on the disjoint TRAIN split, λ tuned on VAL, evaluated once on TEST) recovers the true written value on **26.7% (E) / 25.3% (F)** of the hard transitions, versus **0% for the frozen head** (by construction) and **10% chance**. A small fixed-capacity nonlinear probe recovers **23.3% (E) / 26.1% (F)** — i.e. **no material gain over linear** (nonlinear − linear = −0.035 (E) / +0.008 (F)). The pattern is the same for wrap (27.3% / 27.1%) and no-wrap (26.3% / 23.8%) and slightly worse for repeated ADD (22.2% / 21.2%). The raw arithmetic sum is **no easier** to decode than the modular target (26.0% vs 26.7%, E). On the candidate-**correct** control population the same probe reaches 93.4% (E), confirming it is competent where the representation is intact. Thus **the recoverability threshold is not met** (0.267 ≪ 0.50), the nonlinear probe does not reveal hidden linear information, and wrap/repeated-ADD do not behave materially differently.

Under the pre-registered rule this is **Q-C — weak recoverability**: little target information is recoverable from the frozen candidate representation on the dominant Class-A population. **This is the strongest available evidence for a candidate-generation representation limitation** — but it is an **observed** limitation, it **does not establish causality**, and it **does not justify an intervention**.

## 3. Existing Baseline

* **G–I**: the commit is bit-exact and off the scalar path; a real candidate→commit residual (~5.5% of candidate-correct adds).
* **L/M/N/O**: the off-path third basin is latent placement (B), a systematic **parity** (modular) geometry (M-A), **state-alignment-dominant** (N-B); removing the W-derived parity direction is **non-specific** and harms sequence EM (**O-D**).
* **P**: the sequence-level first-error budget is **candidate-generation-dominated** (A 59.7% T=10 / 67.2% T=20; B ~32%/30%; C ~8%/2.6%; untouched-register 0%). P established only an *observed* budget and licensed one candidate-generation origin experiment — Q.
* **Generator-repair (historical, governed)**: candidate-local supervision raised candidate written-register accuracy from **0.830 (A, terminal-only) to 0.956 (E)** and terminal EM 0.540 → 0.755 — evidence that the *training objective* is a lever for candidate quality.

## 4. Research Question

Is the dominant Class-A candidate-generation failure primarily a **candidate-representation/arithmetic limitation** — specifically, insufficient **linearly accessible** representation of modular wrap / repeated ADD in `C_t` — or more consistent with exposure/decoding/training effects?

## 5. Hypotheses

* **H-A (representation absent).** On Class-A transitions neither a linear nor a nonlinear probe recovers the true value much above chance → candidate representation limitation.
* **H-B (linear geometry).** A linear probe recovers most of the gap → information is present and linearly accessible; exposure/alignment implicated.
* **H-C (nonlinear-only).** Linear weak but nonlinear strong → information present but not linearly organized.
* **H-D (wrap-specific).** Wrap failures show materially weaker recoverability than matched no-wrap.

## 6. Preregistration (declared before aggregate inspection)

Populations (test ADD): **Class-A** = frozen candidate head wrong; **control** = frozen candidate head correct. `recoverability = (probe_acc − frozen_acc)/(1 − frozen_acc)`.
Decision rule:
* **Q-A linearly recoverable** — mean linear recoverability ≥ 0.50 AND wrap & repeated in the same direction (≥0.30).
* **Q-B nonlinear-only** — linear < 0.50 AND (nonlinear − linear) ≥ 0.20.
* **Q-C weak recoverability** — linear < 0.50 AND (nonlinear − linear) < 0.20 AND nonlinear < 0.50.
* **Q-MIX** — wrap/no-wrap/repeated disagree (spread ≥ 0.15).
Probes: ridge (closed-form, λ grid on val) and a fixed-capacity MLP (212→32→10, weight decay, early stop on val, fixed seed). Bars (0.50/0.20/0.15) are **declared conventions**. Inference: five E checkpoints are independent units; F is a control. Continuous effect sizes reported alongside thresholds.

## 7. Frozen Components

Recurrent core, candidate network, observation encoder, gate network, commit rule, head, benchmark, task generator, optimizer, loss. **Nothing modified.** All model execution under `torch.no_grad()`; checkpoints SHA-256 verified before and after. The only fitted objects are analysis-only probes; they are separate Python objects and never wired into `forward`, `decision_head`, or any rollout.

## 8. Dataset / Split

The **DisjointSplitManager** splits (no sequence overlap by construction): TRAIN 6,000 sequences, VAL 1,200, TEST = the canonical strict split (1,200 × 10, stream `90042+seed`). Probe fitting uses **TRAIN only**; λ/capacity selection uses **VAL only**; the final measurement uses **TEST only**. Captured per ADD transition: `C_t`, frozen `W,b`, candidate logits/decode/margin, `‖C_t‖`, `‖C_t−S_prev‖`, off-path distance, operands, wrap, repeated-ADD, t, seq, seed. TEST: all ADD transitions (15,763 over 5 seeds per model); TRAIN/VAL: deterministic subsamples (1,500 / 800 per seed). Split separation is enforced by the DisjointSplitManager streams and asserted in the tests.

## 9. Population Definitions

Primary: **test ADD transitions with frozen candidate decode wrong (Class-A)** — n ≈ 361 / seed (E). Subtypes: **wrap** (n≈160), **no-wrap** (n≈201), **repeated ADD** (n≈157; defined by "previous transition was ADD", the existing benchmark criterion), **isolated** (n≈204). Matched control: **candidate-correct ADD** (n≈2,792 / seed) split into wrap/no-wrap. Raw and matched analyses are both reported; difficult examples are not discarded.

## 10. Frozen-Head Baseline

Frozen candidate written-register accuracy on all test ADD: **0.886 (E) / 0.884 (F)**; Class-A (candidate-wrong) fraction **14% / 15%**. On the Class-A population the frozen head is **0.000 by definition** (the population *is* its errors); on the matched control it is **1.000**. The frozen head is reproduced exactly from the stored `W,b` and `C_t` in the tests.

## 11. Linear-Probe Recoverability

| Population (E) | n/seed | frozen | linear | nonlinear | raw-sum |
|---|---:|---:|---:|---:|---:|
| **Class-A** | 361 | 0.000 | **0.267** | 0.233 | 0.260 |
| Class-A wrap | 160 | 0.000 | 0.273 | 0.232 | 0.233 |
| Class-A no-wrap | 201 | 0.000 | 0.263 | 0.235 | 0.282 |
| Class-A repeated | 157 | 0.000 | 0.222 | 0.181 | 0.225 |
| Class-A isolated | 204 | 0.000 | 0.301 | 0.272 | 0.286 |
| control (cand-correct) | 2,792 | 1.000 | 0.934 | 0.949 | 0.922 |

F: Class-A linear **0.253**, nonlinear 0.261, wrap 0.271, no-wrap 0.238, repeated 0.212. Sanity: probe validation accuracies 0.862 (linear) / 0.879 (MLP) on E; all-test accuracy 0.857 / 0.867 vs frozen head 0.886 — the probes are competent where the representation is intact. **The target is therefore linearly recoverable on only ~26% of the dominant Class-A population** — above chance (10%) but far below the 0.50 recoverability bar.

## 12. Raw-Sum vs Modular Probe

Separate analysis-only linear probes for the modular target `(a+b) mod 10` (10-way) and the raw arithmetic sum `a+b` (19-way): Class-A accuracy **0.267 (mod) vs 0.260 (raw)** for E (0.253 vs ? for F). **The raw sum is not easier to decode than the modular target.** This is **not** a "present but inconveniently organized" signature: there is no evidence that the arithmetic is well represented in `C_t` but merely mis-organized for modular decoding.

## 13. Candidate Geometry

Frozen-vector geometry on test ADD (E, seed-pooled):

| group | n | candidate margin | ‖C_t‖ | ‖C_t−S_prev‖ | off-path distance |
|---|---:|---:|---:|---:|---:|
| candidate-correct | 13,958 | 9.35 | 12.82 | 14.42 | 5.57 |
| **Class-A** | 1,805 | **1.89** | 12.48 | 15.20 | 6.03 |
| Class-A wrap | 800 | 2.45 | 12.60 | 15.42 | 6.16 |
| Class-A no-wrap | 1,005 | 1.44 | 12.38 | 15.02 | 5.93 |
| Class-A repeated | 784 | 1.63 | 12.37 | 14.67 | 5.74 |
| Class-A isolated | 1,021 | 2.08 | 12.56 | 15.60 | 6.25 |

Class-A transitions are a **margin-degraded** region (margin 1.9 vs 9.3) but **not a cleanly geometrically separated region**: the centroid distance between the candidate-correct and Class-A populations is only **2.94** (comparable to the within-population spread), norms are nearly identical, and the linear-probe confusion on Class-A is **diffuse** (diagonal recall 0.21–0.37 per value; the largest off-diagonal confusions are 33/1805 ≈ 2%). Class-A failures are thus broadly distributed low-margin states rather than a single "wrong region" or a single arithmetic error mode.

## 14. Nonlinear Information Probe

A deliberately small, fixed-capacity MLP probe (212→32→10, L2, early stop on VAL, fixed seed) was trained on TRAIN and evaluated once on TEST on the same population. On Class-A it reaches **0.233 (E) / 0.261 (F)** — **no material improvement** over the linear probe (Δ = −0.035 / +0.008, both within seed noise). The nonlinear probe does **not** recover substantial information that is invisible to the linear probe. (As required, nonlinear-probe success is not interpreted as evidence that a nonlinear readout would improve sequence EM — and here it is not even observed.)

## 15. Wrap / Repeated-ADD Analysis

Wrap vs no-wrap recoverability is essentially the **same** (0.273 vs 0.263, E; 0.271 vs 0.238, F; spread 0.051 / 0.059, below the 0.15 mixed bar). Repeated ADD is **slightly weaker** than isolated ADD (0.222 vs 0.301, E), but in the same low band. There is **no wrap-specific or repeated-ADD-specific collapse** that would single out one arithmetic sub-mechanism; the weak recoverability is uniform across the ADD subtypes.

## 16. Historical Objective Cross-Check (read-only)

The governed generator-repair results (`experiments/generator_repair_results.json`, hash-pinned) show that **candidate-local supervision previously increased candidate written-register accuracy from 0.830 (A, terminal-only) to 0.956 (E)**, with terminal EM 0.540 → 0.755 and T20 0.233 → 0.407. This is **observational context only**: it shows the training objective is a historical lever for candidate quality; it does **not** establish that credit assignment caused the current Class-A residual, and Q does not create or run any training experiment. Q's own measurement is that the *remaining* Class-A errors are only ~26% recoverable from the frozen representation — consistent with a residual that is hard for the frozen representation to expose, whether for representation or exposure reasons.

## 17. E vs F

| Class-A (paired, 5 seeds) | E | F | paired p |
|---|---:|---:|---:|
| linear accuracy | 0.267 | 0.253 | 0.60 |
| nonlinear accuracy | 0.233 | 0.261 | 0.32 |
| raw-sum accuracy | 0.260 | 0.233 | 0.24 |

E and F are statistically indistinguishable on every recoverability measure; both classify **Q-C**. Per-seed Class-A linear accuracy: E [0.252, 0.245, 0.278, 0.278, 0.282] (tight); F [0.218, 0.308, 0.221, 0.302, 0.217] (more variable). No category is significant.

## 18. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a control, never pooled. Probes are fit on TRAIN, tuned on VAL, evaluated **once** on TEST; no test row enters fitting or tuning. All fractions are reported per seed with mean ± SD; paired differences use the five seeds (transition counts are descriptive and are never used as independent *n*). Continuous effect sizes are reported alongside the threshold decisions. Collector and analysis are byte-identical across two runs; checkpoint hashes are unchanged.

## 19. Causal Status

**Observation only.** Q measures recoverability of the frozen representation under analysis-only probes. It does **not** show that candidate generation *causes* the sequence failures, that the representation is incapable of arithmetic, that the state stores a symbolic sum, or that any readout change would help. The wording used throughout is "the target is / is not linearly recoverable from `C_t`" and "the result is consistent with a candidate-representation limitation". The observed weakness is not taken as a causal proof, and the historical-training comparison is explicitly retrospective.

## 20. Recoverability Decision

**Q-C — weak recoverability** (E primary; F agrees).
* Linear Class-A recoverability **0.267 (E) / 0.253 (F)** — below the 0.50 bar.
* Nonlinear − linear **−0.035 (E) / +0.008 (F)** — no nonlinear gain, so not Q-B.
* Wrap / no-wrap / repeated spread 0.051 / 0.059 — consistent, so not Q-MIX.
* Interpretation: **little target information is recoverable from the frozen candidate representation on the dominant Class-A population**; this is the strongest available evidence for a **candidate-generation representation limitation** (with the caveats that ~26% is above chance and that no causal claim is made). **Recoverability is not demonstrated at the pre-registered level.**

## 21. What Q Establishes

* A leakage-controlled, deterministic **recoverability measurement** for the frozen candidate vector on the dominant Class-A population, reproducing the frozen head exactly.
* That the true written value is **only weakly recoverable from `C_t`** (linear 26.7% / 25.3%, nonlinear 23.3% / 26.1%) on the transitions where the frozen head is wrong — versus 0% for the frozen head and 10% chance.
* That a **nonlinear probe adds no material information** over a linear probe.
* That **wrap and no-wrap behave the same**, repeated ADD is only slightly weaker, and the **raw sum is not easier** than the modular target.
* That the Class-A population is a **margin-degraded but not cleanly separated** region of `C_t`.
* That the result is **robust across E and F**.

## 22. What Q Does Not Establish

* It does **not** establish causality (no intervention; only an observed recoverability measurement).
* It does **not** show the candidate network "cannot do arithmetic" or that a symbolic representation is absent; it measures linear/weak-nonlinear *accessibility* under weak analysis-only probes.
* It does **not** show a better/nonlinear readout would improve sequence EM.
* It does **not** distinguish an information limit from an exposure/probe-distribution limit at the level needed for a fix; the probes were trained on the (mostly easy) TRAIN distribution and evaluated on the hard tail.
* It does **not** license any inference-time intervention and does **not** begin Phase 2.

## 23. Final Answers

1. **How much of the dominant Class-A ADD failure is linearly recoverable from `C_t`?** ~**26.7% (E) / 25.3% (F)** (vs 0% frozen, 10% chance) — weak.
2. **Does wrap behave differently from matched no-wrap?** **No** — 0.273 vs 0.263 (E); 0.271 vs 0.238 (F); spread < the mixed bar.
3. **Does repeated ADD behave differently?** **Slightly weaker** (0.222 vs isolated 0.301, E) but in the same low band.
4. **Is raw arithmetic sum easier to decode than modular target?** **No** — 0.260 vs 0.267 (E).
5. **Does a nonlinear probe recover substantially more information?** **No** — 0.233 vs 0.267 (E); 0.261 vs 0.253 (F).
6. **Is the candidate representation demonstrably recoverable without modifying the model?** **No** — recoverability is below the pre-registered 0.50 bar on the dominant Class-A population.
7. **Does E agree with F?** **Yes** — statistically indistinguishable; both Q-C.
8. **Is the evidence more consistent with…?** **Weak candidate representation** (Q-C): little linearly or nonlinearly recoverable target information on the hard transitions. It is *not* linearly-recoverable-information (Q-A) nor nonlinear-only (Q-B), and the subtypes do not disagree (not Q-MIX).
9. **Does the evidence justify an inference-time intervention?** **No.** Recoverability is **not** established, so under the pre-registered rule Q does **not** intervene. Q establishes only the *absence* of a demonstrated recoverable target; it does not license any change and does not begin Phase 2.

## 24. Exactly ONE Next Scientific Experiment

Because Q finds **weak recoverability** (not established), Q must **not** intervene. The one experiment justified by Q's measured outcome is a measurement that resolves the **remaining ambiguity in Q-C**: whether the weak recoverability is a genuine **information limit** of `C_t` or a **probe-distribution / capacity artifact** (Q's probes were trained on the mostly-easy TRAIN distribution and evaluated on the hard tail).

**Experiment R — controlled candidate-representation information test on the Class-A tail (measurement-only).** In one controlled study: (i) a **capacity sweep** (linear → increasing small-nonlinear capacity, each tuned on VAL, evaluated once on TEST) to see whether *any* bounded probe recovers the Class-A target — if none does on held-out sequences, information is absent; (ii) probes trained **directly on the hard (candidate-wrong) distribution** in a disjoint hard-train / hard-val / hard-test design with **matched-difficulty controls**, to separate an information limit from a probe-distribution/transfer artifact; (iii) the same raw-sum-vs-modular and wrap/no-wrap contrasts on the hard distribution. R is measurement-only, touches no commit/gate/readout/candidate weights, is not a re-run of J/K/O/P, and does **not** begin Phase 2. Only if R **establishes** recoverability would a further candidate-side study be licensed — and even then, Q/R would only have identified a recoverable target, not a validated repair.

## Required Figures

`docs/reports/figures/experiment_q_*.png` (12): `head_vs_linear` (frozen head vs linear probe by population), `wrap_nowrap`, `repeated`, `rawsum_vs_modular`, `cand_margin`, `norms`, `value_geometry`, `confusion` (Class-A linear-probe confusion), `nonlinear_vs_linear`, `per_seed` (per-seed recoverability), `gap_by_op`, `historical` (historical objective cross-check).

## Tests

`tests/test_experiment_q.py` — **22 tests**: collector determinism / no-grad / no-mutation / representation capture; checkpoint immutability; split separation; frozen-head reproduction; linear probe separation + closed-form correctness + determinism; nonlinear probe separation; no probe insertion into the model; probe shapes + E/F alignment; wrap / repeated / raw-sum / modular target generation; recoverability & metric arithmetic; the decision rule (synthetic Q-A/Q-B/Q-C); a deliberate leakage failure path + identity-overlap detector; malformed-data rejection; JSON determinism; figure creation; full-split counts; frozen-head accuracy band; gap arithmetic; historical cross-check; operand-pair coverage + constructible held-out-operand protocol; repeated-ADD subset; metadata integrity; historical-report immutability.

## Reproduction

```bash
python experiments/run_experiment_q.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_q_candidate_origin.npz
python experiments/analyze_experiment_q.py \
    --npz experiments/experiment_q_candidate_origin.npz \
    --output experiments/experiment_q_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_q
pytest tests/test_experiment_q.py -v      # 22 tests
pytest tests/                             # full suite
```

Determinism: collector arrays bit-identical across two runs; analysis JSON identical (excluding the timestamp). Checkpoint SHA-256 verified before/after. No model parameter, readout, gate, commit, or candidate weight is modified.

## Scientific Guardrails

Measurement-only; no causal claim. The report says "the earliest observed error occurs at candidate decoding", "the target is only weakly linearly recoverable from `C_t`", and "the result is consistent with a candidate-generation representation limitation". It never says candidate generation *causes* the failures, that the state stores the exact sum, that the representation is symbolic, that the architecture cannot do modular arithmetic, that a better readout will solve it, or that the next intervention will fix it. The observed error budget (P), causal mechanism, and intervention leverage are kept separate; recoverability is **not** demonstrated, so no intervention is proposed, and Phase 2 is not begun. Negative/inconclusive findings are preserved. Experiments G–P reports and artifacts are untouched.
