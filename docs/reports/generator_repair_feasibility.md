# RE:Reason — Phase-1 Generator Repair Feasibility: Is the Ceiling Caused by Terminal-Only Supervision?

**Date:** October 5, 2026
**Status:** Completed — Feasibility Experiment (strong positive result for dense supervision)
**Frozen Architecture:** Nonlinear Candidate MGU (226,358 params) — architecture, state dim, candidate MLP, gate, encoder, readout, and parameter budget all unchanged. **Only the training objective varies.**
**Data References:** [`experiments/generator_repair_results.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/generator_repair_results.json), [`experiments/generator_repair_supplement.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/generator_repair_supplement.json)
**Runners:** [`experiments/run_generator_repair_feasibility.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/run_generator_repair_feasibility.py), [`experiments/analyze_generator_repair_supplement.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/analyze_generator_repair_supplement.py)
**Figures:** [`generator_repair_horizons.png`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/reports/figures/generator_repair_horizons.png), [`generator_repair_stepwise.png`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/reports/figures/generator_repair_stepwise.png), [`generator_repair_em.png`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/reports/figures/generator_repair_em.png)

---

## Executive Answer

**The ~54–58 % terminal-EM ceiling is not an architectural ceiling. With architecture, parameter budget, data, optimizer, and training budget held fixed, changing only the supervision lifted T=10 EM from 0.540 to 0.755 and zero-shot T=20 EM from 0.233 to 0.407.**

1. **Terminal-only supervision was a first-order bottleneck.** All six dense-supervision variants beat terminal-only on the primary metrics: T=10 EM +5.1 to +21.5 pts (paired by seed, t-test p ≤ 0.03, Cohen's dz 1.5–12.3; Wilcoxon p = 0.0625, the minimum possible at n = 5), T=20 EM +4.1 to +17.3 pts, first-error rate 0.70 → 0.38–0.53, drift halved. Candidate/committed decoding improves for every dense variant (C3 marginal on candidate decoding). The frozen seed-42 checkpoint reference (0.533 on the canonical stream) and the fresh terminal-only baseline (0.540 ± 0.018) match the published parameterization-isolation result (0.543 ± 0.028), confirming the baseline.
2. **Candidate/local supervision is the strongest single objective.** E (terminal + auxiliary loss on the candidate vector's *written* registers) reaches **0.755 ± 0.019** T=10 EM (every seed ≥ 0.72) and **0.407 ± 0.023** T=20 EM, with candidate written-register decoding at 0.956 vs 0.830 (dz = 18.0). It works by fixing the generator's local arithmetic: isolated-ADD candidate accuracy rises 0.364 → 0.798; ADD-wrap 0.140 → 0.485.
3. **Late-step supervision is the strongest simple committed-state fix.** D (supervise t ≥ 6 only) reaches 0.699 T=10, 0.346 T=20, eliminates the t = 1 readout ghosts entirely (T1 EM 0.755 → 1.000), and has the lowest real first-error rate among stepwise variants (0.362 vs 0.559 for A). Full stepwise (B) ≈ hybrid λ = 0.25 (C1), with D slightly above both on T10/T20; hybrids degrade monotonically as λ → 1 (C3 ≈ terminal).
4. **The two dense signals are complementary, not redundant.** Stepwise supervision calibrates the *committed-state readout* at every step (kills ghost readout errors; first strict-error 0.700 → 0.417, T1 1.000) and improves candidate write decoding modestly (0.830 → 0.887). Candidate supervision reshapes the *generator* (candidate written 0.956) while leaving committed readout ghosts partly present (E: 4/40 T1 ghost episodes vs 0 for B/D; 38.8 % of E's remaining first errors are commit/readout-origin vs ~16 % for all other variants).
5. **What remains unsolved:** ADD with wrap-around (E transition-level 0.657; candidate@add 0.485; terminal EM 0.493 on wrap-ending sequences), repeated-ADD chains (terminal EM 0.148), the residual commit/readout gap (E's candidate is *better than its committed state* on ADD: 0.798 vs 0.670 isolated; 0.485 vs 0.355 wrap), and long-horizon decay (T=20 EM 0.407). The dominant limitation was supervision; a smaller residual is arithmetic-(wrap)- and readout-related.
6. **No architecture change, no new parameters, no Phase-2 machinery.** All variants share one model class and 226,358 parameters; training cost differs by ≤ ~1.2× (analytical FLOPs +4 %; observed wall-clock +18 %, E −14 % under contention). No branching, verifier, recursion, halting, or online learning was introduced.

**Scientific guardrails:** this does not prove architecture sufficiency; it does not show terminal supervision causes all errors (residual error mass remains); candidate generation, state commitment and readout are reported separately; no claim is inferred from a single metric.

---

## Method

### Frozen architecture and protocol

* Model: `NonlinearCandidateMGUModel` (input 21, state 212, encoder 64→256, candidate hidden 146) — 226,358 params, identical to the parameterization-isolation checkpoint.
* Data: `DisjointSplitManager(RegisterTrackingTask(seed=42), train 6000 / val 1200 / test 1200, base_seed=seed)`; seeds **42, 43, 44, 45, 46**; the test stream (90042+seed) is never used for training, checkpoint selection, or any decision.
* Optimization (identical for all variants): AdamW, lr 1e-3, weight decay 0.01, batch 64, grad-clip 1.0, 3000 steps, validation every 100 steps on 1000 val samples, best-val checkpoint, final evaluation on the held-out test split.
* Model initialization is seeded per run (`set_seed(seed)` before construction) — a minor, documented cleanup over the earlier protocol, whose unseeded init made runs non-reproducible while leaving the baseline distribution unchanged (see Baseline Reproduction).

### Supervision variants (only the loss changes)

| Key | Supervision | Loss |
|---|---|---|
| A | Terminal | `CE(y_T)` |
| B | Full stepwise | `(1/T) Σ_t CE(y_t)` |
| C1/C2/C3 | Hybrid λ ∈ {0.25, 0.50, 0.75} | `λ·CE(y_T) + (1−λ)·(1/T)Σ_t CE(y_t)` |
| D | Late-step | `(1/5) Σ_{t=6..10} CE(y_t)` (1-indexed; last 5 steps) |
| E | Candidate/local | `CE(y_T) + β·mean_{t, written registers} CE(head(C_t))`, β = 1 |

Model E uses the **existing** decision head on the **existing** candidate vectors `C_t = tanh(cand_mlp([f_t ⊙ S_(t−1), e_t]))`; the committed update remains `S_t = (1−f_t)S_(t−1) + f_t C_t`. No parameters are added. The auxiliary term is restricted to the registers the instruction writes (derived from the observation one-hots), i.e. the "immediate post-instruction target register(s)".

### Evaluation

Canonical T=10 test: terminal EM, per-register accuracy. Zero-shot horizon sweep T ∈ {1,2,3,5,8,10,15,20} on fresh disjoint streams (T=1 has only 40 reachable sequences — all are used; no retraining per length). Per-transition committed and candidate decoding (full-state and written-register), gates, first-error chains, and controlled ADD sequences (isolated / ADD→SET / 4 consecutive / no-wrap / wrap; Round-2 audit builders). Statistics: 5-seed means ± SD, paired t-test, Wilcoxon signed-rank, and paired Cohen's dz vs Model A.

### Baseline reproduction

| Reference | T=10 EM | Note |
|---|---:|---|
| Model A (this experiment, 5 seeds) | **0.540 ± 0.018** | per seed: 0.5175 / 0.5592 / 0.5558 / 0.5400 / 0.5283 |
| Published parameterization-isolation ("6_nonlinear_candidate_mgu", 5 seeds) | 0.543 ± 0.028 | same architecture/protocol, unseeded init |
| Frozen seed-42 checkpoint on the same 5 test streams | 0.555 ± 0.021 | 0.533 on the canonical stream 42 (1,200-sequence protocol; 0.5367 known on its 1,500-sequence extension) |

All three anchors agree within ~1 pt; the baseline reproduces.

---

## Primary Results Table

| Model | Supervision | T1 EM | T2 EM | T5 EM | T10 EM | T20 EM | Overall EM |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Terminal | 0.755 | 0.964 | 0.850 | 0.540 | 0.233 | 0.660 |
| B | Full stepwise | **1.000** | **1.000** | 0.926 | 0.672 | 0.335 | 0.768 |
| C1 | Hybrid λ=.25 | **1.000** | **1.000** | 0.916 | 0.662 | 0.319 | 0.757 |
| C2 | Hybrid λ=.50 | **1.000** | **1.000** | 0.904 | 0.629 | 0.291 | 0.742 |
| C3 | Hybrid λ=.75 | **1.000** | **1.000** | 0.881 | 0.591 | 0.274 | 0.723 |
| D | Late-step | **1.000** | **1.000** | 0.933 | 0.699 | 0.346 | 0.778 |
| E | Candidate/local | 0.880 | 0.982 | **0.952** | **0.755** | **0.407** | **0.790** |

*Means over 5 seeds. "Overall" = mean over all eight evaluated horizons {1,2,3,5,8,10,15,20}. Full horizon table (mean ± SD) below. Training at T=10; all other horizons are zero-shot.*

### Full horizon sweep (mean ± SD over 5 seeds)

| Model | T1 | T2 | T3 | T5 | T8 | T10 | T15 | T20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.755±.062 | 0.964±.013 | 0.954±.010 | 0.850±.015 | 0.641±.034 | 0.540±.018 | 0.338±.040 | 0.233±.029 |
| B | 1.000±.000 | 1.000±.000 | 0.988±.003 | 0.926±.008 | 0.761±.016 | 0.672±.015 | 0.463±.014 | 0.335±.018 |
| C1 | 1.000±.000 | 1.000±.000 | 0.984±.002 | 0.916±.006 | 0.739±.007 | 0.662±.033 | 0.438±.022 | 0.319±.024 |
| C2 | 1.000±.000 | 1.000±.000 | 0.979±.002 | 0.904±.004 | 0.718±.009 | 0.629±.028 | 0.416±.019 | 0.291±.019 |
| C3 | 1.000±.000 | 1.000±.000 | 0.974±.006 | 0.881±.011 | 0.687±.022 | 0.591±.041 | 0.380±.026 | 0.274±.028 |
| D | 1.000±.000 | 1.000±.000 | 0.988±.004 | 0.933±.005 | 0.781±.010 | 0.699±.019 | 0.475±.012 | 0.346±.016 |
| E | 0.880±.045 | 0.982±.013 | 0.985±.004 | 0.952±.011 | 0.821±.010 | 0.755±.019 | 0.542±.019 | 0.407±.023 |

---

## Second Table — T=10 EM by Final Operation of the Sequence

| Model | SET | SWAP | ADD no-wrap | ADD wrap | Mean |
|---|---:|---:|---:|---:|---:|
| Terminal (A) | 0.644 | 0.570 | 0.475 | 0.226 | 0.540 |
| Full stepwise (B) | 0.769 | 0.702 | 0.622 | 0.344 | 0.672 |
| Best Hybrid (C1, λ=.25) | 0.761 | 0.692 | 0.612 | 0.323 | 0.662 |
| Late-step (D) | 0.793 | 0.733 | 0.647 | 0.369 | 0.699 |
| Candidate/local (E) | **0.836** | **0.778** | **0.709** | **0.493** | **0.755** |

*Sequences grouped by the operation type at the final timestep (wrap determined from the true pre-operation operands). Mean = overall EM.*

### Supplementary — transition-level written-register accuracy by operation (committed state)

| Model | SET | SWAP | ADD no-wrap | ADD wrap | Mean |
|---|---:|---:|---:|---:|---:|
| Terminal (A) | 1.000 | 0.885 | 0.759 | 0.405 | 0.887 |
| Full stepwise (B) | 1.000 | 0.929 | 0.866 | 0.528 | 0.927 |
| Best Hybrid (C1) | 1.000 | 0.925 | 0.850 | 0.490 | 0.921 |
| Late-step (D) | 1.000 | 0.935 | 0.877 | 0.561 | 0.933 |
| Candidate/local (E) | 1.000 | **0.948** | **0.903** | **0.657** | **0.947** |

*Fraction of transitions whose committed state decodes all written registers correctly. SET transitions are at ceiling for every model; the entire op-level gradient is in ADD/SWAP.*

---

## Third Table — Candidate vs Committed Decoding and Error Accumulation

| Model | Candidate EM | Committed EM | First-error rate | Mean state drift |
|---|---:|---:|---:|---:|
| Terminal (A) | 0.125 ± 0.021 | 0.771 ± 0.012 | 0.700 | 0.0681 |
| Full stepwise (B) | 0.194 ± 0.004 | 0.873 ± 0.009 | 0.417 | 0.0370 |
| Best Hybrid (C1) | 0.192 ± 0.006 | 0.862 ± 0.013 | 0.436 | 0.0402 |
| Late-step (D) | 0.187 ± 0.007 | 0.883 ± 0.008 | 0.385 | 0.0341 |
| Candidate/local (E) | 0.039 ± 0.006 | **0.891 ± 0.012** | 0.426 | **0.0313** |

*"EM" here = full-state decode exact (head vs true state) averaged over transitions t = 1..10, mean ± SD over seeds. First-error rate = fraction of sequences with ≥ 1 strict (any-register) per-step committed error; the stricter real-error (written registers only) rate is A 0.559 → B 0.395 → D 0.362 → E 0.303. Drift = mean per-register committed error across timesteps (1 − per-register accuracy).*

**The E candidate-EM anomaly is the key mechanism.** E's candidates become *specialists*: written-register decoding rises to 0.956 (vs 0.830), while full-state candidate decoding collapses to 0.039 because untouched registers are deliberately left to the committed state (the convex commit preserves them). E converts the candidate from a noisy "full next state" proposal into an accurate "written-register update" proposal. `candidate EM` for E is therefore not comparable on the full-state metric — it must be read jointly with candidate-written (0.956) and committed full-state (0.891).

---

## Task-by-Task Findings

### Q1/Q2 — Stepwise supervision and horizon generalization

Yes on both, with monotone dose-response for hybrids: T5 0.850 → 0.881–0.952; T10 0.540 → 0.591–0.755; T20 0.233 → 0.274–0.407. Parameter count is identical (226,358); nothing was retrained per horizon. Late-step supervision (D) beats full stepwise (B) slightly on T10/T20 despite supervising only 5 of 10 steps — early-timestep supervision is mostly redundant except for early readout calibration (T1: A 0.755 → B/C/D 1.000; E 0.880).

### Q3 — Operation-level effects

Every operation improves; ADD wrap remains the hardest. Sequence-level EM by final op (E vs A): SET +19.2, SWAP +20.8, ADD no-wrap +23.4, ADD wrap +26.7 pts. Transition-level written accuracy (E vs A): SWAP 0.885 → 0.948, ADD no-wrap 0.759 → 0.903, ADD wrap 0.405 → 0.657. SET is at ceiling for all models at transition level.

### Q4/Q5 — First errors and drift

The fraction of sequences with no strict step error rises from 30.0 % (A) to 58.3–61.5 % (B/D) and 57.5 % (E); real (written-register) first-error rate: 0.559 (A) → 0.303 (E). First errors essentially vanish at t = 1–2 for B/C/D (ghost readout was the dominant early failure: A fails 23.8 % of sequences at t = 1, all of them ghost-only mismatches on untouched zero registers — verified directly: 10/10 A first-step failures, 4/4 E first-step failures are on untouched registers). Mean state drift halves: 0.0681 → 0.0313 (E).

### Q6 — Candidate quality before the convex commit

Candidate written-register decoding improves for every dense variant and dramatically for E: A 0.830 → B 0.887 / D 0.888 / **E 0.956** (dz = 18.0 vs A). Controlled ADD candidate-at-ADD-step accuracy (last ADD of the sequence): isolated 0.364 (A) → 0.451 (B) → 0.463 (D) → **0.798 (E)**; wrap 0.140 → 0.173/0.246 → **0.485 (E)**. This confirms the generator itself was under-trained for local transition correctness.

### Q7 — Gate behavior

Modest but significant changes; no qualitative restructuring. Mean gate 0.344 (A) → 0.397 (B) / 0.360 (E). E's gate entropy drops markedly (0.224 → 0.157) — more decisive/binarized gates. Gate↔displacement correlation drops 0.685 (A) → 0.596–0.673. Destination-register decodability from the gate is ~1.0 for every model (the gate trivially carries the instruction route; not a differentiator).

### Q8 — Actual candidate arithmetic (controlled ADD, mean over seeds)

| Builder | A termEM / comm@ADD / cand@ADD | B | D | E |
|---|---|---|---|---|
| Isolated ADD | 0.405 / 0.419 / 0.364 | 0.513 / 0.517 / 0.451 | 0.550 / 0.557 / 0.463 | **0.669 / 0.670 / 0.798** |
| ADD→SET | 0.533 / 0.408 / 0.361 | 0.629 / 0.525 / 0.456 | 0.649 / 0.566 / 0.464 | **0.734 / 0.667 / 0.799** |
| 4× repeated ADD | 0.057 / 0.313 / 0.286 | 0.094 / 0.363 / 0.329 | 0.104 / 0.380 / 0.345 | **0.148 / 0.412 / 0.437** |
| ADD no-wrap | 0.320 / 0.334 / 0.230 | 0.536 / 0.537 / 0.437 | 0.510 / 0.516 / 0.365 | **0.606 / 0.607 / 0.676** |
| ADD wrap | 0.095 / 0.103 / 0.140 | 0.219 / 0.219 / 0.173 | 0.239 / 0.254 / 0.246 | **0.355 / 0.355 / 0.485** |

Note the E signature: its candidate is *more accurate than its own committed state* on ADD (0.798 vs 0.670 isolated; 0.485 vs 0.355 wrap) — the residual error has moved into the commit/readout stage. Repeated ADD chains remain weak for every model (terminal EM ≤ 0.148).

### Q9 — Compositionality

See the full horizon sweep. Dense supervision improves every length except E's small T=1 dip (0.880 vs 1.000 for B/C/D and 0.755 for A). E's dip is entirely ghost-readout on untouched zero registers (4/40 episodes, 0 on the touched register) and is the flip side of its generator-first optimization.

### Failure-chain analysis (A vs B vs D vs E)

At each sequence's first real (written-register) error: operation (0 % SET / 7–12 % SWAP / 41–54 % ADD no-wrap / 36–49 % ADD wrap across variants), wrong registers (uniform across R0–R3), and origin:

| Model | Candidate-origin | Commit/readout-origin |
|---|---:|---:|
| A | 2791 (83.2 %) | 565 (16.8 %) |
| B | 1983 (83.7 %) | 385 (16.3 %) |
| D | 1807 (83.2 %) | 364 (16.8 %) |
| E | 1113 (61.2 %) | **705 (38.8 %)** |

The absolute error count roughly halves for B/D/E; E's *composition* of residuals shifts toward commit/readout — consistent with its candidate@ADD > committed@ADD gap. This is a precision boundary of the current commit/interpolation, not a candidate-generation failure.

### Compute accounting

| Variant | Optimization steps (5 runs) | Head evals / train step | Approx. train FLOPs (5 runs) | Ratio vs A | Observed wall-clock (5 runs)* | Inference FLOPs / episode |
|---|---:|---:|---:|---:|---:|---:|
| A | 15,000 | 1 | 12,594 GFLOP | 1.00× | 682 s | 4.53 MFLOP |
| B/C1/C2/C3/D | 15,000 | 11 | 13,084 GFLOP | 1.04× | 810 ± 4 s | 4.53 MFLOP |
| E | 15,000 | 11 | 13,084 GFLOP | 1.04× | 589 s | 4.53 MFLOP |

*\*Wall-clock totals measured under 5-way parallel worker contention (2 threads each), so absolute values are load-dependent; E's lower total reflects lighter competing load in its batches, not a cheaper objective. The analytical FLOPs ratio (+4 % — extra per-step head evaluations, 3× fwd+bwd approximation) is the parity reference; the empirical bound from the worst clean comparison (B vs A) is ≈ 1.18×. A +13–21 pt EM gain at ≤ 1.2× compute is not a compute effect. Inference cost is identical by construction (same architecture).*

---

## Critical Distinction — Supervision vs Capability

The experiment separates the two hypotheses directly:

* **The transition function was capable of more than terminal-only training extracted (supported).** With the identical model, dense supervision improves the *generator* (candidate written 0.830 → 0.956), the *committed state* (0.771 → 0.891 full-state), the *readout* (T1 ghosts 0.755 → 1.000), error accumulation (drift halved), every operation, and every horizon, with paired dz up to 18.
* **A residual capability boundary remains (partially supported).** After the best objective, ADD wrap still decodes at 0.657 (candidates 0.485), repeated-ADD chains at 0.148 terminal EM, and 24.5 % of T=10 sequences still fail. These concentrated, op-specific residuals are the natural next probe — but they are now a minority of the error budget, and nothing here yet implicates the transition equation itself (no architecture change was tested).

---

## Final Q&A

**Q1. Is terminal-only supervision the problem?**
It was a first-order bottleneck, not the whole problem. Changing only the objective lifted T=10 EM by +5 to +21.5 pts across the dense variants (B/D/E: +13.2 to +21.5, dz 6.8–12.3; all paired p ≤ 0.03) and improved every horizon and operation. A smaller, structured residual error mass remains (wrap arithmetic, commit/readout).

**Q2. Can the same recurrent transition learn better local updates with intermediate supervision?**
Yes. Identical architecture and parameter count: candidate written-register decoding 0.830 → 0.956, isolated-ADD candidate accuracy 0.364 → 0.798, committed full-state decode 0.771 → 0.891.

**Q3. Does intermediate supervision improve modular arithmetic?**
Substantially: ADD no-wrap transition accuracy 0.759 → 0.903, ADD wrap 0.405 → 0.657, controlled wrap terminal EM 0.095 → 0.355 (E). It improves but does not solve wrap-around or repeated ADD chains (0.148).

**Q4. Does it reduce long-horizon error accumulation?**
Yes. First real-error rate 0.559 → 0.303 (E); sequences with no strict step error 30 % → 58–62 %; drift 0.068 → 0.031; T20 EM 0.233 → 0.407 (E), 0.346 (D).

**Q5. Does it improve candidate generation itself?**
Yes — most dramatically under E (candidate written 0.830 → 0.956; candidates exceed the committed state on ADD). All dense variants improve candidate write decoding; E turns the candidate into a specialized written-register update.

**Q6. Does it improve the actual latent state?**
Yes: committed full-state decode 0.771 → 0.891, committed written 0.887 → 0.947, higher at every timestep from t ≥ 3 (E) or every timestep (B/D), with halved drift.

**Q7. Does it change the gate's role?**
Only modestly: mean gate 0.344 → 0.360–0.397; gate entropy drops sharply for E (0.224 → 0.157, more decisive gates); gate/displacement correlation 0.685 → 0.596–0.673; destination decodability ~1.0 for all. Same routing role, different assertiveness.

**Q8. What remains unsolved after improving supervision?**
(i) ADD with wrap-around (transition 0.657; candidate 0.485; terminal 0.493 by final op); (ii) repeated-ADD chains (0.148 terminal EM); (iii) the commit/readout residual (E's candidate beats its committed state; 38.8 % of E's first errors are commit/readout-origin); (iv) long-horizon decay (T20 0.407).

**Q9. Is the current recurrent architecture still the right frozen core?**
Yes — more clearly than before. The unchanged core now reaches 0.755 T=10 / 0.407 T=20; the earlier ceiling was largely an artifact of the objective. No evidence yet that the transition equation is the binding constraint at this new operating point; the residual wrap/commit questions are now the right probes.

**Q10. What is the single most informative next experiment?**
**Combine the two dense signals under the same protocol:** a stepwise committed-readout loss (deghosts the readout — B/D) plus the candidate-local auxiliary loss (fixes local arithmetic — E), with the terminal loss retained. This one new variant × 5 seeds tests whether the residual commit/readout gap and wrap arithmetic close when both failure modes are supervised directly — and, if it holds, establishes the new Phase-1 training protocol. If wrap accuracy stays ≈ 0.5–0.65 under both signals, the follow-up is a representation-level probe of modular wrap-around (e.g., operand-value structure in the state), not an architecture rewrite.

---

## Final Decision

**A. ADOPT BETTER SUPERVISION AND CONTINUE PHASE 1.**

The decision tree's first branch applies: stepwise/hybrid supervision produces a large improvement with the architecture unchanged (B/D/E: T=10 EM +13.2 to +21.5 pts, dz 6.8–12.3; T=20 EM +10.1 to +17.3 pts, dz 3.7–6.3; all dense variants paired t p ≤ 0.03; worst-case observed compute ratio ≈ 1.2×; all 5 seeds improve). Training objective / credit assignment was a major limitation of the previous Phase-1 baseline, and the improvement is clearly sufficient to justify making dense supervision the new Phase-1 protocol:

* **Adopt (pending the combined-objective confirmation):** terminal loss + candidate-local loss (E's term) + late/full stepwise committed-state loss (B/D's term). D and E are individually strong and mechanistically complementary; C1 (λ = 0.25) shows near-full-stepwise performance with a small terminal weight.
* **Freeze as-is:** the recurrent transition, encoder, gate, readout, parameter budget, and inference path — all unchanged by this experiment.
* **Do not proceed to Phase-2 machinery** (branching, verifier, transaction): the single transition was under-trained; with dense supervision the deterministic K=1 path now reaches 0.755 and dominates everything Phase-2 ever measured, at zero architectural cost.
* **Do not rewrite the core yet:** residual failures (wrap, repeated ADD, commit/readout ghosts) are now localized enough to attack with objective- and representation-level probes first.

---

## Appendix — Reproducibility and Artifacts

* Runner: `experiments/run_generator_repair_feasibility.py` (worker mode per (variant, seed), parallel coordinator, aggregation mode with statistics and figures). Supplement: `experiments/analyze_generator_repair_supplement.py`.
* Results: `experiments/generator_repair_results.json` (summary, per-seed runs, paired statistics vs A, per-op/horizon/controlled-ADD metrics, compute accounting); `experiments/generator_repair_supplement.json` (per-op transition-level decoding).
* Trained checkpoints (35, ≈ 0.9 MB each) were written to the session scratch directory; any run can be regenerated deterministically from the runner with the same seeds.
* Independent checks: terminal EM was recomputed from saved weights for A/B/D/E (seed 42 and E seed 46) and matched exactly; the frozen seed-42 checkpoint was evaluated on all 5 test streams; T=1 failure modes were verified register-by-register (all ghost-only).
