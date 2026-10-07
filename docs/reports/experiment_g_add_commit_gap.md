# RE:Reason — Experiment G: Localizing the ADD Candidate→Commit Gap

**Date:** October 5, 2026
**Status:** Completed — Measurement-only localization (no training, no model changes, no rollouts)
**Frozen models:** `E_candidate` (official Phase-1 objective) and `F_combined` (control), 5 seeds each, 226,358 parameters, unchanged
**Data References:** [`experiments/experiment_g_transitions.npz`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/experiment_g_transitions.npz), [`experiment_g_transitions_meta.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/experiment_g_transitions_meta.json), [`experiment_g_results.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/experiment_g_results.json)
**Runners:** [`experiments/run_experiment_g.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/run_experiment_g.py), [`experiments/analyze_experiment_g.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/analyze_experiment_g.py), tests [`tests/test_experiment_g.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/tests/test_experiment_g.py)
**Checkpoints:** `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (SHA-256 hashes recorded in the transitions metadata)

---

## Executive Answer

**The ADD candidate→commit gap is real, small, concentrated, and mechanistically mixed. It is confined almost entirely to two-register ADDs with nonzero operands; within that stratum a partial gate transfer leaves a convex blend that the readout decodes to neither the old nor the new value. Evidence for gate dilution and for blend/readout failure is both present — classification: C (both materially contribute). No causal claim is made.**

1. **Size and location.** 5.5% of candidate-correct ADD transitions commit wrong (E: 764/13,958; F: 770/13,940; seed-level 5.47% ± 0.48% vs 5.52% ± 0.74%, paired p = 0.89). The gap is **not uniformly distributed**: 95.8% (E) / 96.1% (F) of these failures are in the *general* stratum (destination ≠ source register, source operand > 0). No-op ADDs (source operand 0, correct result equals the old value) and same-register ADDs are essentially gap-free (0.25–0.61%).
2. **Gate relationship (within the general stratum).** The gap rate falls monotonically with the full-vector mean gate: **34.7% / 42.1% (f < 0.3) → 16.3% / 18.6% (0.3 ≤ f < 0.4) → 1.2% / 2.1% (f ≥ 0.4)** for E / F. Within this stratum the gate is the strongest single discriminator of commit failure among candidate-correct adds (oriented AUC **0.76 (E) / 0.77 (F)**), better than candidate displacement. However, the model never opens the ADD gate widely (observed mean-gate range ≈ 0.23–0.51), so the high-transfer regime is essentially unobserved.
3. **Decode signature.** Only **31.8% (E) / 28.6% (F)** of committed failures decode to the *old* value (the classical dilution signature); **68.2% / 71.4%** decode to a third value — the committed state moved toward the correct candidate but the readout maps the partial blend to neither endpoint. The dominant corruption mode is therefore blend/readout, not simple old-state dominance.
4. **Displacement contributes.** Pooled over all ADDs, `||C_t − S_{t-1}||` predicts the gap better than the gate (AUC **0.689 / 0.684** vs **0.582 / 0.626**); adding the gate to displacement yields **no** gain for E (ΔAUC −0.0001 [−0.0003, +0.0002]) and a small gain for F (+0.0127 [+0.0076, +0.0175]). Within the general stratum, the ordering flips (gate AUC 0.76/0.77 beats displacement 0.56/0.61) — the pooled displacement advantage is largely a stratum-composition effect documented below.
5. **E vs F are identical on this residual.** On the 13,202 transitions where both models' candidates are correct: E-only failures 303, F-only failures 299 (paired seed-level 3.90% vs 3.87%, p = 0.96). F's higher ADD gate (0.372 vs 0.354, p = 0.002) does not reduce the commitment residual; F's previously measured committed-state advantage lies outside this ADD-destination mechanism.
6. **Next experiment (single): an analysis-only λ-interpolation decode sweep** (Experiment H) that decodes `head((1−λ)·S_(t−1) + λ·C_t)` for λ ∈ [0, 1] on candidate-correct general ADD transitions, to determine whether the readout can decode blends at *any* λ (blend/readout) or only above a threshold λ* (transfer sufficiency), and how λ* compares to the observed f_t. This is a measurement, not a fix.

---

## Data and Method

### Data collection (frozen models, no training)

* For each of E and F × seeds {42, 43, 44, 45, 46}: one `forward_with_states(..., return_components=True)` pass over (a) the canonical strict test split (1,200 sequences × 10 steps; stream 90042+seed, never trained on) and (b) the five existing controlled ADD builders (isolated / ADD→SET / 4× repeated / no-wrap / wrap; 400 sequences each).
* **31,763 ADD transitions per model** (15,763 test-split + 16,000 controlled). For every transition the following were recorded: model, seed, sequence id, timestep, source/destination registers, operand values, wrap flag, isolated/repeated flag, candidate/committed/previous-state destination correctness, candidate/committed/previous decoded values, full-vector gate mean/min/max, `||C−S_prev||`, `||S−S_prev||`, `||S−C||`, cos(S,C), candidate/committed destination CE and probabilities, and `comm_eq_old` (committed decode equals the old value).
* Collector validation: ADD `committed_written_all_acc` reproduces the independently produced per-op supplement values to within ±0.0005 for all four models/categories (e.g., E wrap 0.6564 vs 0.6568; F wrap 0.6616 vs 0.6617).

### Gate statistic (explicit statement, as required)

`f_t` is a **212-dimensional per-coordinate interpolation vector**. The existing implementation reports only full-vector gate summaries (mean/std/entropy); **there is no principled destination- or register-specific gate coordinate**, and none was invented. Throughout this analysis the per-transition scalar is the **full-vector mean gate f̄_t** (consistent with the existing `mean_gate` convention), recorded alongside full-vector min/max. All conclusions are therefore about *global* transfer, not about a register-specific gate value.

### Binning and statistical discipline

* Fixed 0.1 grid on f̄ ∈ [0,1]; sparse bins (< 200 transitions) are merged deterministically into the less-populated neighbour until all bins ≥ 200 or 3 bins remain. The realized edges are **0.00 / 0.30 / 0.40 / 1.00** for both models (reported in JSON).
* **Transition-level tables are descriptive** — transitions within a sequence are dependent. Independent inference is seed-level (n = 5): per-seed rates, mean ± SD, paired t-test / Wilcoxon for E vs F on identical seeds and datasets (means ± SD for key metrics are in the JSON).

### Confound documented before interpretation

The raw pooled view is confounded by ADD arithmetic sub-types. Two strata require separation:

| Stratum | Definition | Why it matters |
|---|---|---|
| `no_op_src0` | source operand value = 0 | correct result equals the old value; committed decode of the old value is automatically correct; no transfer needed |
| `same_register_src_gt0` | destination = source register, value > 0 | self-referential update (`2x mod 10`) |
| `general` | destination ≠ source, source > 0 | ordinary two-register modular addition |

The low-gate region is over-represented by no-op/same-register adds; conditioning changes the picture materially (see Analysis 1).

---

## Analysis 1 — Gate-Dilution Decomposition (descriptive; test-split ADDs, 5 seeds pooled)

### Per-bin table

| Model | bin [lo, hi) | n | cand acc | comm acc | prev acc | gap (cand−comm) | P(commit ok \| cand ok) | P(prev ok \| cand ok & fail) | mean ‖C−S_prev‖ | wrap frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | [0.0, 0.3) | 2,341 | 0.928 | 0.928 | 0.947 | +0.000 | 0.989 | 0.739 | 15.28 | 0.361 |
| E | [0.3, 0.4) | 10,909 | 0.854 | 0.805 | 0.971 | +0.048 | 0.921 | 0.958 | 14.66 | 0.219 |
| E | [0.4, 1.0) | 2,513 | 0.983 | 0.982 | 0.994 | +0.002 | 0.996 | 1.000 | 13.13 | 0.012 |
| F | [0.0, 0.3) | 829 | 0.912 | 0.907 | 0.938 | +0.005 | 0.976 | 0.667 | 15.40 | 0.381 |
| F | [0.3, 0.4) | 10,075 | 0.837 | 0.791 | 0.966 | +0.046 | 0.914 | 0.965 | 15.10 | 0.287 |
| F | [0.4, 1.0) | 4,859 | 0.977 | 0.975 | 0.994 | +0.002 | 0.994 | 0.931 | 13.47 | 0.014 |

### Conditional view (candidate-correct transitions only)

| Metric | E | F |
|---|---:|---:|
| n candidate-correct | 13,958 | 13,940 |
| candidate-correct & committed-wrong | 764 | 770 |
| P(commit ok \| cand ok) | 0.9453 | 0.9448 |
| mean f̄ (commit ok / commit wrong) | 0.3566 / 0.3452 | 0.3757 / 0.3543 |
| f̄ AUC for failure (oriented) | 0.4025 (0.5975) | 0.3548 (0.6452) |
| failures below f = 0.35 vs exposure | 56.7% vs 41.1% | 44.0% vs 33.5% |
| failures at/above f = 0.50 vs exposure | 0.0% vs 0.0% | 0.0% vs 0.0% |
| P(prev state correct \| failure) | 0.952 | 0.957 |
| committed decode equals old value \| failure | 0.318 | 0.286 |

### The confound-free view (what actually carries the gap)

| Model | Stratum | n | cand acc | comm acc | P(fail \| cand ok) | failures | share of all failures | old-value decode \| fail | f̄ AUC for fail (oriented) | d_cs AUC for fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | no_op_src0 | 7,000 | 0.969 | 0.970 | 0.35% | 24 | 3.1% | 0.00 | — | — |
| E | same_register | 2,241 | 0.951 | 0.954 | 0.38% | 8 | 1.0% | 0.375 | — | — |
| E | **general** | **6,522** | **0.773** | **0.690** | **14.5%** | **732** | **95.8%** | **0.328** | **0.240 (0.760)** | 0.611 |
| F | no_op_src0 | 7,000 | 0.966 | 0.969 | 0.25% | 17 | 2.2% | 0.00 | — | — |
| F | same_register | 2,241 | 0.956 | 0.957 | 0.61% | 13 | 1.7% | 0.385 | — | — |
| F | **general** | **6,522** | **0.772** | **0.694** | **14.7%** | **740** | **96.1%** | **0.291** | **0.228 (0.772)** | 0.563 |

**General-stratum gradient by gate bin** (this is the dilution-relevant gradient):

| Model | bin | n | cand acc | P(fail \| cand ok) | failures | f̄ AUC (oriented) | d_cs AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| E | [0.0, 0.3) | 113 | 0.434 | 34.7% | 17 | 0.449 (0.551) | 0.410 |
| E | [0.3, 0.4) | 5,742 | 0.758 | 16.3% | 707 | 0.274 (0.726) | 0.588 |
| E | [0.4, 1.0) | 667 | 0.964 | 1.2% | 8 | 0.215 (0.785) | 0.794 |
| F | [0.0, 0.3) | 38 | 0.500 | 42.1% | 8 | 0.625 (0.375) | 0.523 |
| F | [0.3, 0.4) | 5,224 | 0.730 | 18.6% | 707 | 0.286 (0.714) | 0.536 |
| F | [0.4, 1.0) | 1,260 | 0.953 | 2.1% | 25 | 0.325 (0.675) | 0.727 |

**Failure decode composition (all test ADDs):** E — old value 31.8%, third value 68.2%, new value 0.0% (by construction); F — 28.6% / 71.4% / 0.0%.

**Reading (empirical).** Within the general stratum the gap rate is steeply and monotonically gate-dependent (≈35–42% → 16–19% → 1–2%), and the gate is the best single discriminator there. Yet only ~30% of failures decode to the old value; the majority show a third value — i.e. transfer occurred but the readout of the partial blend was wrong. The observed ADD mean-gate range is narrow (E: 0.230–0.477; F: 0.248–0.505), so predictions about f ≳ 0.5 are outside the observed operating range.

---

## Analysis 2 — Oracle-Commit Upper Bound (analysis-only decode of `head(C_t)`)

All numbers are **transition-level destination decoding** (not rollouts, not sequence-level bounds). "Lost at commitment" counts candidate-correct/committed-wrong transitions.

| Category | Model | n | committed acc | oracle candidate acc | oracle − committed | lost at commitment | lost / candidate-correct |
|---|---|---:|---:|---:|---:|---:|---:|
| isolated ADD (controlled) | E | 2,000 | 0.669 | 0.797 | +0.128 | 311 | 16.3% |
| ADD no-wrap (controlled) | E | 2,000 | 0.608 | 0.676 | +0.069 | 213 | 15.7% |
| ADD wrap (controlled) | E | 2,000 | 0.355 | 0.484 | +0.129 | 354 | 26.6% |
| 4× repeated ADD (controlled) | E | 8,000 | 0.530 | 0.602 | +0.072 | 886 | 15.5% |
| test, all ADDs | E | 15,763 | 0.852 | 0.885 | +0.034 | 764 | 5.5% |
| test, no-wrap | E | 12,492 | 0.903 | 0.920 | +0.017 | 344 | 3.0% |
| test, wrap | E | 3,271 | 0.656 | 0.755 | +0.099 | 420 | 11.1% |
| isolated ADD (controlled) | F | 2,000 | 0.642 | 0.782 | +0.140 | 350 | 18.3% |
| ADD no-wrap (controlled) | F | 2,000 | 0.600 | 0.721 | +0.121 | 297 | 19.4% |
| ADD wrap (controlled) | F | 2,000 | 0.398 | 0.524 | +0.126 | 371 | 23.5% |
| 4× repeated ADD (controlled) | F | 8,000 | 0.517 | 0.590 | +0.073 | 942 | 16.0% |
| test, all ADDs | F | 15,763 | 0.854 | 0.884 | +0.030 | 770 | 5.5% |
| test, no-wrap | F | 12,492 | 0.904 | 0.916 | +0.012 | 330 | 2.6% |
| test, wrap | F | 3,271 | 0.662 | 0.762 | +0.100 | 440 | 11.1% |

The transition-level commitment loss is bounded by this gap (E: +3.4 pts overall, +12.8 pts isolated, +12.9 pts wrap). **This is an upper bound for transition decoding only** — it is not a sequence-level EM bound, and the controlled-builder numbers cover harder synthetic constructions than the natural test distribution.

---

## Analysis 3 — Predictability of the Gap

Target A: `candidate correct & committed wrong` over all test ADDs (base rate 4.85% E / 4.88% F).
Target B: `committed wrong | candidate correct` (base rate 5.47% / 5.52%).

Transparent L2 logistic regression (single/dual features, standardized), **grouped 5-fold CV with sequence groups** (no transition leakage within sequences), plus leave-seeds-out transfer and a grouped-bootstrap CI on AUC differences.

| Metric | E | F |
|---|---:|---:|
| Spearman(f̄, target A) | −0.063 | −0.096 |
| Spearman(‖C−S_prev‖, target A) | +0.141 | +0.138 |
| Spearman(f̄, ‖C−S_prev‖) | −0.335 | −0.363 |
| CV AUC f̄ only (A / B) | 0.582 / 0.595 | 0.626 / 0.645 |
| CV AUC ‖C−S_prev‖ only (A / B) | 0.689 / 0.701 | 0.684 / 0.697 |
| CV AUC both (A / B) | 0.689 / 0.701 | 0.696 / 0.712 |
| CV PR-AUC both (A) | 0.085 (base 0.049) | 0.075 (base 0.049) |
| Leave-seeds-out (both, A) | 0.713 | 0.704 |
| ΔAUC both − f̄ (bootstrap CI, A) | +0.107 [+0.086, +0.126] | +0.070 [+0.052, +0.087] |
| ΔAUC both − d_cs (bootstrap CI, A) | −0.0001 [−0.0003, +0.0002] | +0.0127 [+0.0076, +0.0175] |

**Reading (empirical).** Pooled, displacement is the stronger single predictor and the gate adds nothing over it for E; for F the gate adds a small amount. Within the general stratum the ordering reverses (gate AUC 0.76/0.77 vs displacement 0.56/0.61). The pooled displacement advantage is partly a composition effect: displacement is larger at low gate (15.3 vs 13.1 mean) and in the harder wrap sub-population. Discriminative skill is moderate either way (AUC ≤ 0.71 out-of-seed) — this is a weak-to-moderate, not near-perfect, signal, and it is descriptive.

---

## Required E vs F Comparisons (seed-level, paired n = 5)

| Comparison | Result |
|---|---|
| Similar ADD gate distributions? | Yes in shape (unimodal ≈ 0.30–0.40); F shifted up: mean 0.372 ± 0.008 vs 0.354 ± 0.004 (p = 0.002); observed range 0.248–0.505 vs 0.230–0.477. |
| F's lower error from better candidate generation? | No. Candidate destination accuracy 0.8842 vs 0.8856 (paired p = 0.88); gap rate 3.04% vs 3.39% (p = 0.50). |
| F's committed-state advantage ↔ different gate behavior? | No measurable effect *on this residual*: F is slightly more gate-open yet the ADD commit gap is identical (paired transition comparison: E-only 303 vs F-only 299, p = 0.96). F's committed-state advantage (from Experiment F) is not in the ADD destination mechanism. |
| Does the gap persist when the candidate is correct? | Yes, for both (5.5% of candidate-correct adds). |
| Do low-gate transitions account for most failures? | Partially. Raw: 57% (E) / 44% (F) of failures below f = 0.35 (vs 41% / 34% exposure). Unconfounded (general stratum): the f < 0.3 band has the highest *rate* (35–42%) but only 2–3% of failures; 96% of failures sit in the mass band 0.3 ≤ f < 0.4. |
| Do high-gate failures remain substantial (blend/readout)? | Within the observed range, failures at f ≥ 0.4 are rare (1.1% / 3.2% of failures); f ≥ 0.5 is essentially unobserved (max f̄ = 0.48–0.51). The test is therefore limited by the model's operating range, not able to rule out blend/readout at wide gates. |

---

## Required Figures (`docs/reports/figures/`)

1. `experiment_g_gate_vs_commit_acc.png` — gate f̄ vs committed (and candidate) accuracy for ADD.
2. `experiment_g_gate_histograms.png` — gate distribution split by cand-ok/commit-ok vs cand-ok/commit-wrong.
3. `experiment_g_cand_vs_comm.png` — candidate vs committed accuracy across gate bins.
4. `experiment_g_dcs_vs_gate.png` — `||C−S_prev||` vs gate, split by commit outcome.
5. `experiment_g_oracle_vs_committed.png` — oracle candidate vs actual committed accuracy by ADD category.
6. `experiment_g_prediction.png` — ROC curves for f̄ / ‖C−S_prev‖ / both (grouped CV, both targets).
7. `experiment_g_confound_strata.png` — gap rate by gate bin and arithmetic stratum (the confound-free view).

---

## Decision Logic — Classification

**C. Both materially contribute.**

* **Gate dilution is supported as a rate-controlling factor**: within the unconfounded general stratum the failure rate rises steeply as the transfer fraction falls (1–2% at f ≥ 0.4; 16–19% at 0.3–0.4; 35–42% at f < 0.3), there are no failures where the gate is at the top of its observed range, ~30% of failures retain the old value, and the previous state correctly decodes the old value in 95–96% of failures.
* **Blend/readout failure is supported as the dominant corruption mode**: 68–71% of failures decode to a third value (transfer happened; the readout of the partial blend was wrong), failures require a *displaced* correct candidate (`||C−S_prev||` adds strongly pooled; d_cs predicts the rare high-gate failures within the general stratum, AUC 0.73–0.79), and F's stepwise committed-state supervision — which trained the readout at every step — does not reduce this residual at all (paired p = 0.96).
* **Neither explanation alone is adequate**: pure dilution would predict failures concentrated at the lowest gates with old-value decodes (the lowest band has the highest rate but only 2–3% of the failures, and old-value decodes are a minority); pure "sufficient transfer + unreadable blend" would predict gate-independence once transferred (the gate is instead the strongest within-stratum discriminator, and there are no failures at the top of the observed gate range).

**Unresolved hypotheses** (not testable from this data): (i) whether the readout could decode a *fully* correct blend at some intermediate λ — no λ-sweep was run; (ii) whether the observed gate ceiling (f̄ ≤ ~0.5 on ADD) is a property of the learned policy, of the gate's per-coordinate structure, or of the global-mean statistic; (iii) whether the third-value decodes are near-miss interpolations in value space or unrelated attractors (decoded values were recorded; their geometry was not analyzed here).

---

## Next Experiment (single, measurement-only)

**Experiment H — λ-interpolation decode sweep of the ADD commitment.**
On the frozen E and F checkpoints, for every candidate-correct **general-stratum** ADD transition in the test split, construct `S(λ) = (1−λ)·S_(t−1) + λ·C_t` over a fixed grid λ ∈ {0, 0.05, …, 1.0} and decode `head(S(λ))[dest]` against the true new value. Report:
1. **Decode-accuracy vs λ** (pooled and by wrap/no-wrap): monotone increase to the candidate level ⇒ pure transfer-sufficiency (dilution) with a well-behaved readout; a dip below both endpoints or a plateau below 1.0 ⇒ blend/readout limitation.
2. **λ*(transition)** = smallest λ with correct decode; its distribution vs `||C−S_prev||`, wrap, and the actual f̄_t. Failures with f̄_t ≥ λ* are blend/readout cases; failures with f̄_t < λ* are transfer-insufficiency cases — a per-transition causal classification that this experiment could not provide.
3. **Counterfactual check (analysis only):** decode `head` at the *actual* f̄_t vs at λ = 1 to confirm the recorded failure attribution.

Outcome: a quantitative split of the residual into transfer-insufficiency vs blend-decode components, and the transfer fraction needed for reliable ADD commitment — motivating (but not implementing) any future commit/readout change.

**Explicitly out of scope:** no training, no loss changes, no architecture change, no Phase-2 machinery, no fix implementation. This experiment localizes an existing residual only.

---

## Reproduction

```bash
# 1. Collect per-ADD-transition records from the frozen checkpoints (fast, ~20 s)
python experiments/run_experiment_g.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_g_transitions.npz

# 2. Run the full analysis + figures (~30 s)
python experiments/analyze_experiment_g.py \
    --transitions experiments/experiment_g_transitions.npz \
    --output experiments/experiment_g_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_g

# 3. Tests for the analysis pipeline
pytest tests/test_experiment_g.py -v
```

* The collector writes `experiments/experiment_g_transitions_meta.json` (columns, source ids, checkpoint paths + SHA-256 hashes, row counts); the analysis JSON embeds the full transition metadata.
* Pipeline tests: 14 passing (`tests/test_experiment_g.py`), covering bin merging determinism, decomposition values, conditional analysis, confound stratification, oracle table, logistic regression + grouped CV + grouped bootstrap, seed-level and paired statistics, a collector smoke test on a real repo checkpoint, and an end-to-end synthetic run of the analysis + all seven figures.
* Reconciliation: the collector reproduces the independent per-op supplement (`experiments/combined_objective_supplement.json`) to ±0.0005 on all four ADD categories and both models.

## Scientific Guardrails

This experiment **localizes** a residual and does not establish causation. It does not claim that the gate is a symbolic memory mechanism, that the convex commit is understood, or that the architecture is sufficient; it provides empirical observations (counts and rates), a supported interpretation (mixed dilution/blend-readout evidence, classification C), and explicitly unresolved hypotheses (the λ sweep, the gate ceiling, the geometry of third-value decodes). The ADD candidate→commit gap affects ~5.5% of candidate-correct ADD transitions; it is one part of the remaining Phase-1 error budget, alongside wrap arithmetic and repeated-ADD chains documented earlier. No fix was implemented.
