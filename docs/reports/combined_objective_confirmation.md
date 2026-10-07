# RE:Reason — Phase-1 Combined Objective Confirmation (Model F)

**Date:** October 5, 2026
**Status:** Completed — Confirmation Experiment (F does not beat E on the primary metric; adopt candidate/local)
**Frozen Architecture:** Nonlinear Candidate MGU (226,358 params) — unchanged. One new training objective (F), five seeds; all other variants reused from the preceding experiment under the identical protocol.
**Data References:** [`experiments/combined_objective_confirmation_results.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/combined_objective_confirmation_results.json), [`combined_objective_analysis.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/combined_objective_analysis.json), [`combined_objective_supplement.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/combined_objective_supplement.json)
**Runners:** [`run_generator_repair_feasibility.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/run_generator_repair_feasibility.py) (new variant `F_combined` + per-step loss-component logging), [`analyze_combined_objective.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/analyze_combined_objective.py)
**Figures:** [`combined_objective_horizons.png`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/reports/figures/combined_objective_horizons.png), [`combined_objective_stepwise.png`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/reports/figures/combined_objective_stepwise.png), [`combined_objective_em.png`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/reports/figures/combined_objective_em.png)

---

## Executive Answer

**The two supervision signals are not complementary at the terminal-EM level. F matches E's generator quality and beats E's committed-state calibration, but it does not beat E on T=10 or T=20 EM — so per the decision rule, the simpler candidate/local objective (E) is adopted, and objective sweeps stop.**

1. **F vs E (primary/secondary): no improvement, small consistent negative trend.** T=10: **0.737 ± 0.015 vs 0.755 ± 0.019** (paired Δ = −1.77 pts; 4/5 seeds negative; p = 0.17; dz = −0.74). T=20: 0.390 ± 0.014 vs 0.407 ± 0.023 (Δ = −1.70; p = 0.24). T=15 is the only nominally significant horizon (Δ = −2.18; p = 0.017). No horizon shows an F advantage ≥ +0.6 pts; T=5 is tied (0.951 vs 0.952).
2. **F does fix the ghost readout failure mode.** T=1 EM reaches **1.000 with zero ghost errors** (0/200 episodes across seeds; written-register errors also 0); E keeps 24 ghost-only failures and T1 = 0.880. F's strict first-error rate is the best of all models (0.341 vs E 0.425, D 0.385, A 0.700) with drift also lowest (0.028).
3. **F matches E on the generator and slightly beats it on the committed state.** Candidate written-register decoding: F 0.954 vs E 0.956 (ns); committed written: 0.946 vs 0.947 (ns); committed full-state: **0.903 vs 0.891** (+1.2 pts, p = 0.089, dz = +1.0). Static (evaluated-at-checkpoint) losses show the trade: E = terminal 0.013 / stepwise 0.075 / candidate 0.014; F = 0.023 / 0.018 / 0.027 — the only model low on all three, but at the cost of some terminal/candidate fit.
4. **The candidate→commit gap is NOT closed on ADD; it is ADD-specific.** Isolated ADD: candidate − committed = **+0.14 (F)** / +0.13 (E), vs +0.00 (SWAP) and −0.005 (SWAP,D). F improves ADD-wrap transition accuracy over every other model (candidate 0.524, committed 0.398, transition 0.662 — best), but the gap remains, exactly where the task predicted "the next problem is probably the commit/interpolation mechanism."
5. **F beats D decisively.** T=10 +3.8 pts (p = 0.023), T=20 +4.4 (p = 0.014), T=15 +4.5 (p = 0.012), candidate written +6.6 pts (dz = 8.8), real first-error −5.5 pts (p = 0.007). Every dense-supervision claim from the preceding experiment is confirmed; combined supervision dominates late-step supervision on every axis.
6. **Compute and normalization are clean.** Same 226,358 parameters; identical inference path (4.53 MFLOP/episode). Training FLOPs: F/E ratio **1.04×** (extra supervised candidate heads: 21 head-evals/step vs E's 11). F's three loss components are all mean per-prediction cross-entropies (identical averaging convention): training-time means terminal 0.375 ± 0.006, stepwise 0.217 ± 0.002, candidate 0.262 ± 0.005, total 0.854 ± 0.012 — shares ≈ 44 : 25 : 31; no component dominates for scale reasons.

**Decision (per the stated rule): B. ADOPT CANDIDATE/LOCAL SUPERVISION.** F is approximately equal to E on the primary metrics and does not improve the candidate→commit gap, so the extra loss term is not adopted ("do not add complexity merely because F contains more loss terms"). The residual commitment gap on ADD becomes the next target — measurement-only, no architecture change.

---

## Method

### Model F — combined supervision (the only new objective)

Exact loss (λ_step = 1, β = 1, no sweep):

```
L_F = CE(y_T)                                      # terminal readout
    + 1.0 * (1/T) Σ_{t=1..T} CE( head(S_t),  y_t )  # committed-state stepwise (full, uniform)
    + 1.0 * (1/|W|) Σ_{t} Σ_{r ∈ W_t} CE( head(C_t)_r, y_{t,r} )   # candidate-local, written regs
```

* `W_t` = registers written at step t (from the observation one-hots).
* `head` is the existing `TypedDecisionHead`; `C_t` are the existing candidate vectors; the committed update `S_t = (1−f_t)S_(t−1) + f_t C_t` is untouched. No new parameters.
* Component normalization: all three terms are **mean cross-entropy per scalar prediction** (terminal: mean over B×4 register predictions; stepwise: mean over steps of the per-step B×4 CE; candidate: mean over written-register positions). Equal weights λ = β = 1 are therefore numerically meaningful.

### Loss-scale audit (required)

**Training-time component means over all 3,000 steps** (F, per seed and aggregated):

| Seed | L_terminal | L_stepwise | L_candidate | L_combined |
|---|---:|---:|---:|---:|
| 42 | 0.3737 | 0.2161 | 0.2629 | 0.8527 |
| 43 | 0.3688 | 0.2155 | 0.2554 | 0.8398 |
| 44 | 0.3750 | 0.2157 | 0.2613 | 0.8520 |
| 45 | 0.3740 | 0.2171 | 0.2598 | 0.8509 |
| 46 | 0.3840 | 0.2202 | 0.2680 | 0.8722 |
| **mean ± sd** | **0.375 ± 0.006** | **0.217 ± 0.002** | **0.262 ± 0.005** | **0.854 ± 0.012** |

Shares of the combined loss: terminal ≈ 44 %, candidate ≈ 31 %, stepwise ≈ 25 %. The components are the same order of magnitude (0.22–0.38); the objective is not dominated by any single term through averaging conventions.

**Static component losses at the best-val checkpoints (train split, mean over seeds)** — context for what each objective drives:

| Model | terminal | stepwise (committed) | candidate |
|---|---:|---:|---:|
| A (terminal) | 0.049 | 0.167 | 0.523 |
| D (late) | 0.037 | 0.025 | 0.233 |
| E (candidate/local) | 0.013 | 0.075 | 0.014 |
| F (combined) | 0.023 | 0.018 | 0.027 |

E fits terminal + candidate best but leaves intermediate committed-state readout at 0.075 (the ghost source); D fits stepwise best but leaves the candidate at 0.233; F is the only model simultaneously low on all three — with terminal/candidate slightly worse than E's extremes.

### Protocol and reference reproduction

* Identical protocol to the preceding experiment: train 6000 / val 1200 / test 1200, 3000 steps, batch 64, AdamW lr 1e-3, wd 0.01, clip 1.0, val every 100 steps, best-val checkpoint, seeds 42–46.
* Models A, D and E references are the stored runs/checkpoints from the preceding experiment (same code path for their losses; deterministic protocol). Values re-verified in the new aggregate: A 0.540 ± 0.018, D 0.699 ± 0.019, E 0.755 ± 0.019 — identical to the published results.
* F checkpoints were independently re-evaluated from saved weights (seeds 43, 45): 0.7183 / 0.7592 — exact match.

---

## Primary Results Table

| Model | T1 | T5 | T10 | T15 | T20 | Candidate Written | Committed Written | Real First Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Terminal A | 0.755 | 0.850 | 0.540 | 0.338 | 0.233 | 0.830 | 0.887 | 0.559 |
| Late D | 1.000 | 0.933 | 0.699 | 0.475 | 0.346 | 0.888 | 0.933 | 0.362 |
| Candidate E | 0.880 | 0.952 | **0.755** | **0.542** | **0.407** | **0.956** | 0.947 | **0.303** |
| Combined F | **1.000** | 0.951 | 0.737 | 0.520 | 0.390 | 0.954 | 0.946 | 0.307 |

*Means over 5 seeds. "Candidate/Committed Written" = per-transition written-register decode accuracy averaged over t = 1..10. "Real First Error" = fraction of sequences with ≥ 1 written-register committed error (strict any-register rates: A 0.700, D 0.385, E 0.425, F 0.341).*

### Full horizon sweep (mean ± SD)

| Model | T1 | T2 | T3 | T5 | T8 | T10 | T15 | T20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.755±.062 | 0.964±.013 | 0.954±.010 | 0.850±.015 | 0.641±.034 | 0.540±.018 | 0.338±.040 | 0.233±.029 |
| D | 1.000±.000 | 1.000±.000 | 0.988±.004 | 0.933±.005 | 0.781±.010 | 0.699±.019 | 0.475±.012 | 0.346±.016 |
| E | 0.880±.045 | 0.982±.013 | 0.985±.004 | 0.952±.011 | 0.821±.010 | 0.755±.019 | 0.542±.019 | 0.407±.023 |
| F | 1.000±.000 | 1.000±.000 | 0.993±.002 | 0.951±.010 | 0.821±.011 | 0.737±.015 | 0.520±.020 | 0.390±.014 |

**F − E at the required horizons:** T5 −0.001 (p = 0.74) · T10 −0.018 (p = 0.17) · T15 −0.022 (p = 0.017) · T20 −0.017 (p = 0.24). The difference is a roughly constant small offset, not a slope change: F does **not** improve the long-horizon slope over E (it does improve it over D: +1.8/+3.8/+4.5/+4.4 pts at T5/T10/T15/T20, all p ≤ 0.02).

---

## ADD Table — Controlled ADD Diagnostics (candidate / committed destination accuracy)

| Model | Isolated ADD | ADD no-wrap | ADD wrap | 4× repeated ADD |
|---|---|---|---|---|
| A | 0.364 / 0.419 | 0.230 / 0.334 | 0.140 / 0.103 | 0.286 / 0.313 |
| D | 0.463 / 0.557 | 0.365 / 0.516 | 0.246 / 0.254 | 0.345 / 0.380 |
| E | **0.798** / **0.670** | 0.676 / 0.607 | 0.485 / 0.355 | 0.437 / 0.412 |
| F | 0.783 / 0.643 | **0.721** / **0.599** | **0.524** / **0.398** | **0.439** / 0.410 |

*Cells are "candidate destination accuracy / committed destination accuracy" at the last ADD step (400 fixed procedural sequences per builder × 5 seeds). Terminal EM on the same builders: F = 0.642 isolated / 0.599 no-wrap / 0.397 wrap / 0.138 repeated; E = 0.669 / 0.606 / 0.355 / 0.148.*

**Candidate→commit gap (candidate − committed):**

| Model | Isolated ADD | ADD no-wrap | ADD wrap | 4× ADD | SWAP (test split) |
|---|---:|---:|---:|---:|---:|
| A | −0.054 | −0.104 | +0.037 | −0.027 | −0.040 |
| D | −0.094 | −0.151 | −0.008 | −0.035 | −0.006 |
| E | +0.128 | +0.069 | +0.129 | +0.025 | **+0.000** |
| F | **+0.140** | **+0.121** | +0.126 | +0.029 | −0.002 |

**F does not close the ADD candidate→commit gap — it is slightly wider than E's.** On SWAP both E and F have essentially closed it (candidate ≈ committed). The gap is ADD-specific and is now the largest measured residual inefficiency: closing it would recover ~+12–14 pts of transition-level ADD accuracy (~one third of the remaining ADD error). Notably, F's absolute wrap accuracy improved over E in both candidate (0.524 vs 0.485) and committed (0.398 vs 0.355) terms, and F's wrap transition-level committed accuracy (0.662) is the best of all models — yet F's sequence-level wrap EM (0.463) is below E's (0.493), a composition effect worth keeping in mind.

---

## Q5 — T=1 Ghost Diagnostic

| Model | T1 EM | Written-register errors | Ghost-only failures (of 200 episodes) |
|---|---:|---:|---:|
| A | 0.755 | 0 | 49 |
| D | 1.000 | 0 | 0 |
| E | 0.880 | 0 | 24 |
| F | **1.000** | **0** | **0** |

F reaches 100% on all 40 unique single-SET sequences in every seed (standard terminal readout — no evaluation shortcut). Combined supervision eliminates the ghost readout failure mode entirely, as the stepwise term predicts.

## Q7 — Failure Chain (4-way classification of first real errors, mean count per 1,200-sequence test run)

| Model | A. Candidate generation (SWAP) | C. Arithmetic (ADD) | B. Commit/readout | D. Other | Total |
|---|---:|---:|---:|---:|---:|
| A | 53 (7.9 %) | 505 (75.3 %) | 113 (16.8 %) | 0 | 671 |
| D | 30 (6.9 %) | 332 (76.3 %) | 73 (16.8 %) | 0 | 435 |
| E | 34 (9.3 %) | 189 (51.9 %) | 141 (38.7 %) | 0 | 364 |
| F | 44 (12.0 %) | 185 (50.3 %) | 139 (37.8 %) | 0 | 368 |

The combined objective does **not** narrow the residual into a single class: F's mix is statistically indistinguishable from E's (slightly more SWAP candidate-generation failures, slightly fewer ADD arithmetic failures) at an equal total count. Relative to D, both E and F halve the arithmetic class and roughly double the commit/readout share — the residual has migrated from generation to commitment.

## Q8 — Gate

| Model | Mean gate | Gate entropy | Gate↔displacement corr | Destination decodability |
|---|---:|---:|---:|---:|
| A | 0.344 | 0.224 | 0.685 | 1.000 |
| D | 0.377 | 0.222 | 0.637 | 0.998 |
| E | 0.360 | 0.157 | 0.673 | 0.998 |
| F | **0.415** | 0.194 | 0.603 | 0.998 |

Combined supervision pushes the gate most open (0.415) and to intermediate entropy between E's near-binary gates (0.157) and the stepwise variants (~0.22). No causal claim about gates is made; destination decodability from the gate remains ~1.0 for every model and is not a differentiator.

## Q9 — Compute

| Variant | Params | Head evals / train step | Train FLOPs (5 runs) | Ratio vs E/D | Inference FLOPs / episode | Wall-clock (5 runs)* |
|---|---:|---:|---:|---:|---:|---:|
| A | 226,358 | 1 | 12,594 GFLOP | 0.96× | 4.53 MFLOP | 682 s |
| D | 226,358 | 11 | 13,084 GFLOP | 1.000× | 4.53 MFLOP | 807 s |
| E | 226,358 | 11 | 13,084 GFLOP | 1.000× | 4.53 MFLOP | 589 s |
| F | 226,358 | 21 | 13,573 GFLOP | **1.037×** | 4.53 MFLOP | 622 s |

*\*Wall-clock totals under parallel-worker contention; load-dependent. The analytical FLOPs ratio (extra T candidate-head evaluations; 3× fwd+bwd approximation) is the parity reference. Inference architecture and cost are identical by construction.*

---

## Final Q&A

**Q1. Does combined supervision beat candidate/local supervision?**
No. T10 0.737 vs 0.755 (Δ = −1.8 pts, 4/5 seeds negative, p = 0.17); T20 0.390 vs 0.407 (Δ = −1.7, p = 0.24); T15 −2.2 (p = 0.017). F never beats E by more than 0.6 pts at any horizon; T5 is tied. Under the stated decision rule this is "approximately equal", and F is *worse* on the primary metric in trend — so E is retained.

**Q2. Does combined supervision beat late-step supervision?**
Yes, decisively: +3.8 pts T10 (p = 0.023), +4.4 T20 (p = 0.014), +4.5 T15 (p = 0.012), +1.8 T5 (p = 0.017), candidate written +6.6 pts (dz = 8.8, p < 0.001), real first-error −5.5 pts (p = 0.007), drift −0.006 (p = 0.019). Every F vs D comparison favors F.

**Q3. Does it improve candidate generation?**
Relative to D and A, overwhelmingly (candidate written 0.954 vs 0.888/0.830; isolated-ADD candidate 0.783 vs 0.463/0.364; wrap candidate 0.524 vs 0.246/0.140). Relative to E: no — identical within noise (0.954 vs 0.956; isolated 0.783 vs 0.798; wrap 0.524 vs 0.485, F nominally better on wrap).

**Q4. Does it improve committed-state quality?**
Yes, relative to both references: committed full-state 0.903 (E 0.891, D 0.883), committed written 0.946 (E 0.947 tie; D 0.933), strict first-error 0.341 (best), drift 0.028 (best), and the ghost class is eliminated (T1 1.000). This is the clearest F advantage — calibration, not terminal accuracy.

**Q5. Does it close the candidate→commit gap?**
No. On ADD the gap *persists and is slightly larger* than E's (isolated +0.140 vs +0.128; no-wrap +0.121 vs +0.069; wrap +0.126 vs +0.129). On SWAP the gap was already closed by E and remains closed. The gap is ADD-specific. Per the task's own criterion, if the candidate remains substantially better than the committed state, the next problem is the commit/interpolation mechanism — that is exactly where the residuals now concentrate.

**Q6. Does it improve ADD wrap-around?**
Transition-level: yes, and F is the best model on it (candidate 0.524, committed 0.398, transition written 0.662; E 0.485 / 0.355 / 0.657; D 0.246 / 0.254 / 0.561). Sequence-level wrap EM: F 0.463 vs E 0.493 — F's absolute wrap gains at the transition level do not fully pass through to sequence outcomes (composition effects). Wrap remains the hardest operation for every model.

**Q7. Does it improve repeated ADD?**
Marginal: 4×-ADD candidate 0.439 vs E 0.437, committed 0.410 vs 0.412, terminal EM 0.138 vs 0.148. Repeated ADD chains remain essentially unsolved (all models ≤ 0.15 terminal EM).

**Q8. Does it improve T20 zero-shot generalization?**
No, not over E (0.390 vs 0.407). Over D and A: yes (+4.4 and +15.7 pts). F also does not change the long-horizon slope relative to E (the F−E offset is roughly constant across T5–T20).

**Q9. What fraction of remaining failures are now candidate-origin vs commit/readout-origin?**
For F: 62.2 % candidate-origin (12.0 % SWAP candidate generation + 50.3 % ADD arithmetic) vs 37.8 % commit/readout-origin (0 % other). E: 61.2 % / 38.8 %. The composition is indistinguishable; the combined objective lowered the total error count to E's level (368 vs 364 per run) but did not shift the residual mix.

**Q10. Which supervision protocol should become the official Phase-1 training objective?**
**Candidate/local (E), per the decision rule** — F is approximately equal on the primary metrics and does not improve the candidate→commit gap, and the rule says not to add complexity (a third loss term) without a demonstrated gain. F's ghost-free calibration is a real capability, and the stepwise term remains the tool to recover it if future work re-prioritizes intermediate-state calibration — but it is not adopted as the training objective now. Objective sweeps stop here.

**Exact adopted loss (E):**

```
L_E = CE( y_T )  +  (1/|W|) Σ_{t} Σ_{r ∈ W_t} CE( head(C_t)_r , y_{t,r} )      (β = 1)
```

where `head` is the existing decision head, `C_t` the existing candidate vectors, and `W_t` the registers written at step t.

---

## Final Decision

**B. ADOPT CANDIDATE/LOCAL SUPERVISION.**

The decision rule's "approximately equal" branch applies: F does not beat E on T=10 or T=20 (small consistent negative trend), and it does not close the candidate→commit gap; it therefore does not justify the extra loss term. The measured F advantages (ghost elimination, best committed-state calibration, best strict first-error and drift) are documented for future reference but are not the adoption criterion.

### Single next scientific experiment

**Experiment G — Localize the ADD candidate→commit gap (measurement only; no training, no architecture change).**

On the frozen E and F checkpoints, over all ADD transitions in the test splits:
1. **Gate-dilution vs blend-readout decomposition.** Bin transitions by gate value `f_t`; measure candidate decode, committed decode, and previous-state decode as functions of `f_t`. Attribute committed errors to (i) dilution (`head(S_t)` tracks the old state; low `f_t`) vs (ii) blend-readout error (`S_t` is a genuine convex combination but `head(S_t)` misdecodes it).
2. **Oracle-commit upper bound (analysis only).** Quantify how much transition-level ADD accuracy the gap is worth by decoding `head(C_t)` in place of `head(S_t)` at ADD steps (no model change, no rollout).
3. **Predictability of the gap.** Test whether the gap is predictable from `(f_t, ||C_t − S_(t-1)||)` — i.e. whether a commit-side calibration signal exists at all.

Outcome: determines whether the eventual fix belongs to the commit/gate (open wider on arithmetic steps) or to the readout (decode blends vs candidates), before any architecture decision is made.

**Explicitly out of scope (unchanged):** no Phase-2 machinery, no architecture change, no loss-weight sweeps, no new candidate heads, no representation redesign. This result determines only how much of the existing capability different training objectives expose; it does not claim the architecture is sufficient (wrap-around and repeated-ADD ceilings remain, and the commit gap is now measured, not explained).

---

## Appendix — Artifacts and Verification

* `experiments/combined_objective_confirmation_results.json` — full aggregate (8 variants; A–E references identical to the preceding experiment).
* `experiments/combined_objective_analysis.json` — 4-way failure-chain classification, T=1 ghost diagnostics, static loss components.
* `experiments/combined_objective_supplement.json` — per-op transition-level decoding incl. SWAP gap.
* F trained runs: 5 seeds (42–46); T10 EM 0.7408 / 0.7183 / 0.7342 / 0.7592 / 0.7325; all checkpoints saved. Independent recomputation from saved weights for F seeds 43 and 45 matched exactly (0.7183 / 0.7592).
* Loss-component logs were added to the runner this experiment; the reuse of A/D/E references is exact because their stored runs/checkpoints are deterministic outputs of the identical protocol.
