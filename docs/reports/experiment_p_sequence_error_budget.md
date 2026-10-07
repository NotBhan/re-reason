# RE:Reason — Experiment P: Sequence-Level Error-Budget Attribution

**Date:** October 7, 2026
**Status:** Completed — Measurement only (frozen checkpoints; no training; no optimizer; no parameter/readout/gate/commit modification; no intervention; no branching; no verifier; no oracle repair; no counterfactual rollout). Classification: **candidate-generation (Class A) dominates the first-error budget** for E and F at both T=10 and T=20.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned
**Data:** `experiments/experiment_p_error_budget.npz`, `experiments/experiment_p_error_budget_meta.json`, `experiments/experiment_p_results.json`
**Runners:** `experiments/run_experiment_p.py`, `experiments/analyze_experiment_p.py`, tests `tests/test_experiment_p.py`
**Checkpoints:** `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against G/L/M/N/O metadata)

---

## 1. Executive Answer

**The frozen model's sequence-level error budget is dominated by candidate-generation errors.** Of the sequences that fail at T=10, the **earliest observed task-level error** is a candidate written-register error (Class A) in **59.7% (E) / 59.4% (F)** of failures (5-seed mean), a candidate→commit residual (Class B) in **32.5% / 31.8%**, and a terminal/readout residual (Class C) in only **8.0% / 8.7%** (C1 untouched 0%, C2 written 2.0% / 3.1%, C3 near-boundary 5.8% / 5.6%); Class D is empty. Class A dominates in **5/5 seeds** with a mean gap of **+0.27** over Class B at T=10, and the dominance strengthens at T=20 (**A 67.2% / 67.6%**, B 30.2% / 29.9%, C 2.6% / 2.5%; gap +0.37, 5/5 seeds). The result is thus **not a pooled artifact** — it is a seed-consistent ordering, and E and F are statistically indistinguishable (A p=0.90, B p=0.75, C3 p=0.74). The T=10 and T=20 terminal EM values reproduce the established headline numbers exactly (E 0.7547 / 0.4068; F 0.7370 / 0.3898).

**Terminology (strict, per the brief).** This is an **observed error budget**: "X% of failed sequences have candidate generation as their earliest observed task-level error." It is **not** a causal claim that candidate generation causes X% of failures. The three categories answer only *where a sequence first becomes observably wrong*.

The required final questions are answered in §30.

## 2. Existing Baseline (what G–O established, and what P adds)

* **G/H/I**: the element-wise commit is bit-exact and off the scalar path; a real candidate→commit residual (~5.5% of candidate-correct adds, concentrated in general two-register adds) with coordinate-wise under-movement plus a placement component.
* **J/K**: commit-only magnitude (γ) and shape (κ) interventions move local margin delivery but improve no sequence endpoint; commit-rule modification frozen.
* **L/M/N**: the off-path third basin is latent-state placement; a systematic **parity** structure; the parity signature is **state-alignment-dominant** (N-B), with the readout a small non-dominant substrate.
* **O**: removing the W-derived parity component of the candidate is a **non-specific**, broadly damaging local perturbation that **reduces** sequence EM — parity is a correlated diagnostic, not a causal mechanism.
* **P (this report)** asks where the *actual* T=10/T=20 error budget lives by attributing each failed sequence to its earliest observed failure site, and finds **candidate generation (Class A) dominates**.

## 3. Research Question

> Where does the frozen model's T=10/T=20 exact-match error budget live? Among failed sequences, what fraction first become observably wrong at candidate generation (A), at the candidate→commit residual (B), or only at the terminal/state readout (C)?

## 4. Hypotheses

* **H-A.** Candidate-generation errors dominate the first-error budget (consistent with the audits localizing the hard arithmetic — modular wrap — upstream of the commit).
* **H-B.** The candidate→commit residual dominates (consistent with G–L's transition-level localization).
* **H-C.** Terminal/readout residuals dominate at long horizons (decoder-adjacent ambiguity accumulating).
* **H-mix.** The budget is mixed / not cleanly dominated, or its composition changes with horizon.

## 5. Pre-registration (declared before inspecting aggregate results)

Recorded in `run_experiment_p.py` / `analyze_experiment_p.py`; per failed sequence exactly one class:

```
A  earliest t with candidate written-register decode != true written value
B  earliest t with candidate written correct AND committed written-register decode wrong
C  no A/B anywhere, but the terminal full-state decode is wrong:
     C3  near-boundary: min normalized true-vs-winner margin over wrong registers > -0.5
     C1  otherwise, a wrong register was never written
     C2  otherwise, all wrong registers were written at some point
D  failure not cleanly attributable
```

Priority A > B > C; C3 > C1 > C2. **Dominance** = largest mean per-seed fraction among T=10 terminal failures AND largest in ≥4/5 seeds; if the top two are within 0.05 or swap ordering, the budget is "mixed / not cleanly dominated". The first-error budget is over **terminal failures** (the established definition of a failed sequence). Inference: five E checkpoints are independent units; F is a secondary control. Transition/sequence counts are descriptive.

## 6. Frozen Components

Recurrent core, candidate network, observation encoder, gate network, commit rule, readout, benchmark, task generator, optimizer, loss, and all Phase-2 machinery. No parameters, weights, readout, gate, commit, or benchmark modified; all analysis is on the **actual frozen trajectory** under `torch.no_grad()`. No counterfactual rollout, branching, repair, or oracle.

## 7. Dataset / Sequence Definition

* Canonical strict test split (1,200 × 10; stream `90042+seed`) for T=10 — the same split as G–O; horizon sets `horizon_test_set(seed, L)` for T ∈ {1,2,3,5,8,15,20}.
* Unit of analysis: the **sequence**. Per-step trajectory (`S_prev`, `C`, `S_commit`) and the exact frozen decodes are captured for T=10 and T=20; per-sequence first-error classes are computed for every horizon.
* Per seed: 1,200 sequences at T=10 (6,000 pooled over 5 seeds); 1,000 at T=20 (5,000 pooled). T=10 terminal EM reproduces 0.7547 (E) / 0.7370 (F) and T=20 0.4068 / 0.3898.

## 8. Task-State Reconstruction

The true register state after every step is the benchmark-provided deterministic `step_targets` (reconstructed from the operation sequence; verified: for ADD steps `true_written == (operand_a + operand_b) mod 10` exactly). For each step P compares the frozen decodes against the true written value and the true full state, distinguishing candidate written error, committed written error, untouched-register error, and terminal full-state error.

## 9. First-Error Classification

Exclusive, deterministic, chronological. A sequence is **failed** if its terminal full-state decode is wrong; among failures the class is the earliest observed error: A (candidate written wrong) > B (candidate correct, committed written wrong) > C (no A/B; terminal readout residual, C1/C2/C3) > D. The classification makes **A/B/C exhaustive for failures** under the declared priority, so D is empty in this population (reported, not hidden).

## 10. T=10 Error Budget (primary)

| Class (E primary) | mean fraction ± SD | per-seed | count (mean/seed) |
|---|---:|---|---:|
| **A candidate generation** | **0.597 ± 0.038** | [0.62, 0.54, 0.63, 0.62, 0.59] | ~176 |
| B candidate→commit residual | 0.325 ± 0.032 | [0.30, 0.36, 0.31, 0.32, 0.34] | ~96 |
| C1 untouched | 0.000 | — | 0 |
| C2 written | 0.020 ± 0.006 | — | ~6 |
| C3 near-boundary | 0.058 ± 0.016 | — | ~17 |
| D mixed | 0.000 | — | 0 |

E T=10 EM 0.7547 ± 0.0191, 294 failures/seed. **Dominator = A (5/5 seeds, gap +0.272).** F T=10: A 0.594 ± ?, B 0.318, C2 0.031, C3 0.056; EM 0.7370; **dominator A (5/5)**.

## 11. T=20 Error Budget

| Class (E) | mean fraction ± SD |
|---|---:|
| **A candidate generation** | **0.672 ± 0.047** |
| B candidate→commit residual | 0.302 ± 0.049 |
| C2 written | 0.009 |
| C3 near-boundary | 0.017 |
| C1, D | 0.000 |

E T=20 EM 0.4068 ± 0.0225, 593 failures/seed. **Dominator = A (5/5, gap +0.371).** F T=20: A 0.676, B 0.299, C2 0.009, C3 0.016; EM 0.3898; dominator A (5/5). **The composition shifts toward candidate generation as the horizon grows** (A 59.7% → 67.2%; B ~32.5% → 30.2%; C 8.0% → 2.6%).

## 12. Horizon Error Budget

| Horizon | EM (E) | A | B | C3 |
|---:|---:|---:|---:|---:|
| T=3 | 0.985 | 0.20 | 0.39 | 0.39 |
| T=5 | 0.952 | 0.52 | 0.37 | 0.10 |
| T=8 | 0.821 | 0.55 | 0.35 | 0.07 |
| T=10 | 0.755 | 0.60 | 0.33 | 0.06 |
| T=15 | 0.542 | 0.66 | 0.30 | 0.03 |
| T=20 | 0.407 | 0.67 | 0.30 | 0.02 |

At the shortest non-trivial horizon (T=3) the commit residual and readout residual are relatively larger (A 0.20, B 0.39, C3 0.39); from T=5 onward **candidate generation dominates and its share grows monotonically with horizon**, while the readout/terminal share shrinks. This is an empirical error-budget trajectory, not a causal hazard.

## 13. Candidate-Generation Profile (Class A)

For E, the first-error operations of Class-A sequences (T=10, pooled): **SET 0, SWAP 93, ADD no-wrap 568, ADD wrap 542** — i.e. candidate-origin first errors are almost entirely **ADD** (~1,110 of 1,203), split roughly evenly between no-wrap and **wrap** (~49%), plus a small SWAP count. First-error timesteps are spread over t=2…9 (histogram: t2 51, t3 117, t4 190, t5 245, t6 296, t7 285, t8 362, t9 327), i.e. candidate errors accumulate across the trajectory rather than concentrating at a single step. Class-A off-path distance ≈ 6.3 and mean gate ≈ 0.36 (comparable to the general residual population).

## 14. Candidate→Commit Residual Profile (Class B)

For E Class-B first failures (T=10): candidate margin 4.47, committed margin 0.73, **ρ_margin 0.73**, ‖C−S_prev‖ off-path distance 6.39. The operations: SWAP 19, ADD no-wrap 274, **ADD wrap 377** — the commit residual is **even more wrap-weighted** than Class A. This reproduces the G–L transition-level residual (candidate-correct/commit-wrong with ρ ≈ 0.73–0.90) but places it at **~32% of the sequence-level first-error budget** — far below the ~68% transition-level candidate-correct share, confirming that transition-level and sequence-level budgets are different objects (as the brief insists).

## 15. Readout / Decoding Profile (Class C)

Class C is small: E T=10 C1 = 0, C2 = 29, C3 = 85 (pooled). C3 (near-boundary, normalized true-vs-winner distance between −0.5 and 0) is the majority; C1 (a wrong register that was never written) is **empty**. The C failures are therefore **decoder-adjacent ambiguous states** (small margins) rather than deep wrong regions, described with the exact L-style linear-readout decomposition; this is a *decoding signature*, not a causal readout failure. Note also that Class C requires *no* candidate or commit written error anywhere — so a sequence that had any earlier written error is A/B even if it also ends wrong.

## 16. Untouched-Register Retention

Across all 6,000 (seq, seed) T=10 rows, the fraction with a terminal error on a register that was **never written** is **0.000** (both E and F); the number of failed sequences with an untouched-register terminal error is **0**. Every terminal error is on a register that was written at some point. This reproduces the earlier "100% retention on untouched registers" finding and shows that **untouched-register corruption does not contribute to the T=10/T=20 budget** in this population.

## 17. Error-Recovery / Propagation

After a first observed error, the frozen system often recovers the *written-register* correctness on a later step but rarely the *full-state* correctness:

| First class (E) | recover candidate written | recover committed written | recover full-state | mean subsequent full-error steps |
|---|---:|---:|---:|---:|
| A | 0.746 | 0.743 | 0.311 | 2.92 |
| B | 0.794 | 0.796 | 0.416 | 3.04 |
| C3 | 0.000 | 0.000 | 0.000 | — |

So candidate and commit written errors are **not** terminal in themselves — ~75–80% are followed by a later correctly-written step — but only ~31–42% of the sequences recover full-state correctness, i.e. errors accumulate across registers. C3 (terminal) cannot recover by construction. This is observational; no propagation is called causal.

## 18. T10→T20 Transition Analysis

On the **same sequences** (T=20 dataset, prefix at step 10):

|  | count (E) |
|---|---:|
| T10 success → T20 success | 1,678 |
| T10 success → T20 failure | 1,998 |
| T10 failure → T20 failure | 968 |
| T10 failure → T20 success | 356 |

P(T20 fail | T10 fail) = **0.731**; P(T20 fail | T10 succ) = **0.544** (odds ratio ≈ 2.3). A T=10 failure is thus a **descriptive predictor** of T=20 failure, but the majority of T=20 failures come from sequences that were still correct at T=10 (long-horizon accumulation). No causal reading.

## 19. E-vs-F

| Class (T=10) | E | F | paired p |
|---|---:|---:|---:|
| A | 0.597 | 0.594 | 0.90 |
| B | 0.325 | 0.318 | 0.75 |
| C3 | 0.058 | 0.056 | 0.74 |

E and F are statistically indistinguishable on the budget; both classify **A-dominant** at T=10 and T=20, with the same operation concentration and the same near-zero C1 and untouched-register rates.

## 20. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a secondary control, never pooled. Sequence-level fractions are reported per seed with mean ± SD; Class A dominates in 5/5 seeds with a gap far exceeding the declared 0.05 near-mixed bar. Transition-level and per-step counts are descriptive. The T10→T20 association is a declared descriptive conditional probability / odds ratio, treated as observational prediction. No significance is manufactured from the thousands of transitions.

## 21. Historical Intervention Cross-check

Using J/K/O as historical evidence only: the sequence-level budget **explains why those interventions did not improve the endpoint**. J/K intervened on the **commit** — which is the earliest error in only ~32% of failures — and O intervened on a **candidate-representation parity component**, which O itself showed to be non-specific and broadly damaging. The dominant earliest error (**candidate generation, ~60–67%**) is upstream of the commit and was not the target of any of J/K/O; the interventions could therefore at best touch the minority (B) component, consistent with their local-only effects and no sequence gain. This is an interpretation, not a re-analysis; J/K/O are not modified or rerun.

## 22. Recoverable-Target Assessment

| Observed mass | Class | Recoverable-target label |
|---|---|---|
| ~60% (T10) / ~67% (T20) | A candidate generation | **A1** — large error mass, concentrated in ADD (wrap/repeated), **not falsified** by J/K/O (which targeted commit/representation-parity, not candidate generation) |
| ~32% | B candidate→commit residual | **B1** — large error mass, **already falsified** by J/K (simple commit interventions give no sequence improvement) |
| ~3–8% | C terminal/readout residual | **C1** — readout-adjacent mass with **no causal intervention evidence**; mostly near-boundary (C3), and L already showed the fixed bias is negligible |
| ~0% | D | unresolved/mixed |

## 23. Interpretation / Causal Status

**Observation.** The earliest observed task-level failure is a candidate-generation error in ~60% (T=10) / ~67% (T=20) of failed sequences, a candidate→commit residual in ~32% / ~30%, and a terminal/readout residual in ~3–8% / ~2.6%; untouched-register terminal errors do not occur; candidate/commit written errors recover ~75–80% of the time at the written register but only ~31–42% at the full state; and T=10 failure predicts T=20 failure (OR ≈ 2.3).

**Causal status.** **Observation only.** P attributes sequences to their earliest observed failure site; it does **not** show that candidate generation *causes* the failures, that any category is *recoverable* at inference time, or that fixing it would improve the endpoint. The word "candidate-origin" means only "the earliest observed error occurs when the frozen candidate is already decoded incorrectly."

**Unresolved.** P does not distinguish, among Class-A sequences, whether the candidate error is an arithmetic-representation limit of the candidate MLP (the wrapped sum is not linearly represented) or a training/credit-assignment effect (candidate-local supervision was shown in the generator-repair experiment to raise candidate accuracy) — and it does not test any change to candidate generation.

## 24. What P Establishes

* A deterministic, leakage-free, measurement-only **sequence-level error budget** for the frozen E model (and the F control) at T=10 and T=20, reproducing the headline EM exactly.
* That **candidate-generation errors (Class A) dominate** the first-error budget: **59.7% at T=10, 67.2% at T=20** for E (59.4% / 67.6% for F), 5/5 seeds, gap ≫ the mixed bar.
* That the **candidate→commit residual (Class B)** is the second-largest earliest site (~32% / ~30%), *lower than its transition-level candidate-correct share*, confirming transition-level ≠ sequence-level budgets.
* That **terminal/readout residuals (Class C)** are small (~8% / ~2.6%) and mostly **near-boundary (C3)**; the untouched-register subtype (C1) is empty.
* That **untouched-register corruption is zero** at T=10 (retention is preserved) and that candidate/commit written errors are frequently followed by later correct steps (~75–80%) yet rarely recover full-state correctness (~31–42%).
* That the budget composition **shifts toward candidate generation as the horizon grows**, and that **T=10 failure predicts T=20 failure** (OR ≈ 2.3) descriptively.
* That the budget is **robust across E and F**.

## 25. What P Does NOT Establish

* It does **not** establish causality; the classes are *earliest observed* sites, not causes.
* It does not show that any category is **recoverable** at inference time, nor that fixing the dominant class would improve sequence EM.
* It does not re-open the commit question: J/K already showed that simple commit interventions do not help (Class B is B1, already falsified).
* It does not prove that terminal/readout residuals are causally readout failures — they are *decoding signatures*, and L already closed the readout-bias hypothesis.
* It does not, by itself, justify an intervention; a dominant numerical mass is not yet a validated recoverable target.
* It does not generalize beyond the register-tracking benchmark, the E/F objectives, or the frozen K=1 trajectory; and it does not begin Phase 2.

## 26. Final Decision

**Dominant observed failure source: Class A — candidate generation** (E primary; F control agrees; 5/5 seeds at both T=10 and T=20). The candidate→commit residual (B) is second, and terminal/readout residuals (C) are a small near-boundary minority. This is an **observed** budget, not a causal attribution. Recoverable-target labels: **A1** (dominant, localized in ADD wrap/repeated-ADD, not falsified by J/K/O), B1 (already falsified), C1 (readout-adjacent, no causal evidence), D (empty).

**Answers to the required final questions.**

1. **Fraction of T=10 failures first arising from candidate generation?** **59.7% (E) / 59.4% (F)**.
2. **From candidate→commit residual?** **32.5% / 31.8%**.
3. **Terminal/state-readout residuals?** **8.0% / 8.7%** (C3 near-boundary 5.8% / 5.6%, C2 2.0% / 3.1%, C1 0%).
4. **Mixed/unresolved?** **0%** (the declared priority makes A/B/C exhaustive for failures).
5. **Which class dominates for E?** **A (candidate generation)**, 5/5 seeds, gap +0.27 (T=10), +0.37 (T=20).
6. **Same ranking for F?** **Yes** — A 0.594 (T=10) / 0.676 (T=20); statistically indistinguishable from E.
7. **Does the composition change between T=10 and T=20?** **Yes** — A rises 59.7% → 67.2%, B ~flat (32.5% → 30.2%), C falls 8.0% → 2.6%.
8. **At what timestep do first errors usually occur?** Spread over t=2…9, peaking ~t=8–9 (none at t=0–1).
9. **Largest candidate-origin operation budget?** **ADD** (no-wrap 568 + wrap 542 of ~1,203), then SWAP (93); SET 0.
10. **Largest commit-residual operation budget?** **ADD**, especially **wrap** (377 vs 274 no-wrap); SWAP 19; SET 0.
11. **How much of the remaining error is untouched-register corruption?** **0%** (no untouched-register terminal errors).
12. **How often does the model recover after its first observed error?** Written-register recovery 74–80%; full-state recovery only 31–42%.
13. **How often does a first error propagate to multi-register terminal failure?** Mean 2.9–3.0 subsequent full-error steps; only ~31–42% recover full-state.
14. **Does the T10 error budget predict T20 failure?** Descriptively yes — P(T20 fail | T10 fail) = 0.731 vs 0.544 given T10 success (OR ≈ 2.3).
15. **Do the budgets explain why J/K/O failed to improve sequence EM?** Yes as an interpretation: J/K targeted the commit (earliest site in only ~32% of failures) and O targeted a non-specific candidate-representation parity component; the dominant earliest error (candidate generation, ~60–67%) was never their target.
16. **Is the dominant observed source the most plausible recoverable target?** It is the **largest and most localized** component (A1, ADD wrap/repeated), and it is **not already falsified** by J/K/O — but P does not demonstrate recoverability (it is an observed mass, not an intervention result).
17. **Enough evidence to justify another intervention on the frozen model?** For an **inference-time** intervention, **no** — J/K (commit) and O (candidate-representation parity) are falsified, and the dominant class-A errors are arithmetic-generation errors, which are not obviously inference-time recoverable. The single defensible next step is a **candidate-generation origin** experiment (below), which must establish recoverability before any intervention is adopted.

## 27. Exactly ONE Next Scientific Experiment

**Experiment Q — one controlled candidate-generation origin experiment targeting the dominant Class-A (wrap / repeated-ADD) budget.** P's budget is dominated by candidate-generation errors (Class A ~60–67%), concentrated in ADD (wrap and no-wrap) — a component **not** targeted by J/K (commit) or O (candidate-representation parity) and therefore **not already falsified**. Q is the single controlled study that P licenses: it asks **whether the dominant Class-A candidate-generation failure is an arithmetic-representation limit of the frozen candidate MLP (the wrapped sum is not linearly represented in `C`) or a training/credit-assignment effect (candidate-local supervision was already shown to raise candidate accuracy in the generator-repair experiment)** — using the exact per-transition candidate representation and the L/N linear-readout decomposition tools, on the frozen trajectory, measurement-first. Q must not touch the commit, gate, or readout; it must not be a re-run of J/K/O; and it must establish recoverability (not assume it) before any intervention is adopted. Per the governing loop, P is MEASURE: it determines the budget and names the one next experiment; it does **not** design or implement an intervention and does not begin Phase 2.

## Reproduction

```bash
# 1. Collect the frozen trajectory + first-error classes (~30 s)
python experiments/run_experiment_p.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_p_error_budget.npz

# 2. Analysis + 12 figures (~3 s)
python experiments/analyze_experiment_p.py \
    --npz experiments/experiment_p_error_budget.npz \
    --output experiments/experiment_p_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_p

# 3. Tests
pytest tests/test_experiment_p.py -v      # 14 tests
pytest tests/                             # full suite
```

Determinism: the collector was run twice (arrays bit-identical, max |Δ| = 0.0); the analysis twice (JSON identical excluding the timestamp). Checkpoint SHA-256 verified before/after; the T=10 terminal EM reproduces the established headline values exactly.

## Artifact List

* `experiments/run_experiment_p.py` — collector: per-step frozen trajectory, decodes, task metadata, first-error classification.
* `experiments/analyze_experiment_p.py` — T10/T20 budgets, horizon trajectory, survival/hazard, per-class profiles, untouched-register, recovery/propagation, T10→T20 cross-tab, E-vs-F, 12 figures.
* `tests/test_experiment_p.py` — 14 tests (collector determinism, checkpoint hashes, no-grad/no-mutation, split reproduction, task-state reconstruction, candidate/committed/full correctness, classification, mutually exclusive categories, budget/all-error/horizon/T10-T20 accounting, untouched/recovery/residual/margin, E/F alignment, JSON determinism, figures, malformed data, historical immutability).
* `experiments/experiment_p_error_budget.npz` — per-step trajectory arrays (T=10, T=20) + per-sequence first-error classes for all horizons.
* `experiments/experiment_p_error_budget_meta.json` — horizons, class codes, c3 bar, checkpoint hashes.
* `experiments/experiment_p_results.json` — budgets, dominators, survival, profiles, cross-tab, E-vs-F.
* `docs/reports/figures/experiment_p_*.png` — 12 figures: `budget_t10`, `budget_t20`, `budget_horizons`, `survival`, `hazard`, `first_error_t`, `classA_ops`, `classB_profile`, `classC_readout`, `untouched`, `recovery`, `E_vs_F`.
* Checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`.
* Historical references (read only, unchanged): `experiments/experiment_o_representation_intervention.npz`, `experiments/experiment_n_parity_origin.npz`, `experiments/experiment_m_geometry.npz`, `experiments/experiment_l_readout.npz`, `experiments/experiment_i_coordinate.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_g_transitions.npz`.

## Scientific Guardrails

Measurement-only, observational attribution; **no causal claim** and **no fix**. The classes are *earliest observed* failure sites, and the report uses "candidate-origin sequence" strictly to mean "the earliest observed task-level error occurs when the frozen candidate is already decoded incorrectly." C3 is called a **readout-adjacent / decoding signature**, never a causal readout failure. The observed error budget, causal mechanism, and intervention leverage are kept explicitly separate; a numerically dominant category is not treated as a validated recoverable target. No counterfactual rollout, branching, repair, verifier, or oracle is used. Negative and inconclusive findings (empty C1/D, zero untouched-register corruption) are preserved. Experiments G/H/I/J/K/L/M/N/O reports and artifacts are untouched.
