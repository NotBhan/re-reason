# RE:Reason — Final Phase-1 Experiment: Oracle Branching Feasibility

**Date:** October 5, 2026  
**Status:** Completed — Feasibility Probe (Offline)  
**Target Model:** Nonlinear Candidate MGU (Seed-42 checkpoint, 212-dim state, 256-dim obs, 226,358 params) — frozen, not retrained  
**Data Reference:** [`experiments/branching_feasibility_results.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/branching_feasibility_results.json)  
**Runner:** [`experiments/run_branching_feasibility.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/run_branching_feasibility.py)  
**Benchmark:** Register Tracking, frozen test split — 1,500 sequences × 10 transitions = 15,000 transitions (`DisjointSplitManager(RegisterTrackingTask(seed=42), base_seed=42)`, identical protocol to the Round-2 audit)

---

## Executive Answer

**Alternative candidate states containing the correct answer exist, but they are lottery tickets, not hidden computational paths.**

1. **Existence (positive):** At K = 8, a perfect per-step verifier that samples stochastic candidates (with the gate and S frozen, noise only inside the candidate MLP) recovers **241 / 695 = 34.7 %** of the deterministic terminal failures (test EM 0.5367 → **0.6913**). On real (target-register) transition failures, **32.2 %** contain at least one fully-correct candidate at K = 8.
2. **Structure (negative):** Each stochastic draw decodes the written register on a real failure at **≈ 0.10 = chance** (ADD no-wrap slightly above chance ≈ 0.12, ADD wrap *below* chance ≈ 0.06). The candidate distribution carries no reliable information about the correct next value; the oracle is selecting a lucky re-sample, not a better computation.
3. **Control (negative):** A magnitude-matched **random perturbation of S_candidate** — explicitly not a valid branching mechanism — still recovers 119 / 695 terminal failures (EM → 0.6080), i.e. **≈ 46 % of the primary mechanism's EM gain** and 18.6 % of real transition failures. The candidate-generator-specific advantage is real but modest.
4. **Selection is the bottleneck:** No non-oracle selector exists. Even with a *perfect* selector, a **single** repair followed by deterministic continuation reaches only EM 0.5953 (+5.9 pts; 110/695 failures recovered, 22 successes broken). Most of the oracle gain needs re-selection at every step.
5. **Error anatomy:** half of all per-transition strict "failures" (1,717 / 3,403) are **ghost readout mismatches** on untouched registers. Real write errors occur only in ADD (73.9 %) and SWAP (26.1 %); SET writes never fail. ADD wrap-around is *less* recoverable than average (23.6 %), so it is not the recovery engine.

---

## Protocol and Rules Enforced

* Frozen model: `NonlinearCandidateMGUModel` loaded from `experiments/checkpoints/6_nonlinear_candidate_mgu_seed42_3000steps.pt`, eval mode, no retraining, no core modification.
* For every transition the deterministic pass freezes `S_(t-1)`, `e_t`, `f_t` (gate never perturbed).
* Candidates: `C_0` = deterministic candidate; `C_1..C_7` = stochastic draws, noise injected **only** on the candidate MLP hidden activation `h` (primary: Gaussian, σ_d = c · per-dim std of h, c = 1.0; sensitivity: c ∈ {0.25, 0.5, 2.0}, MC dropout p ∈ {0.1, 0.2}). Draw prefixes are nested across K.
* Branch state `B_k = (1 − f_t) · S_(t-1) + f_t · C_k`; decoding always through the model's own `decision_head`.
* Ground truth is used **only** to score candidates (oracle verifier) and to define errors. It never alters gate, state, generation, or sampling.
* This is an offline feasibility probe. Nothing here is "Phase-2 branching", and the oracle EM is an upper bound under a perfect selector, not achievable performance.

### Validation

* Deterministic reconstruction: max |recomputed − `model.forward`| logit difference = 0.0.
* K=1 oracle EM and K=1 one-shot EM exactly equal the deterministic test EM (0.5367).
* Deterministic candidate (index 0) decode equals the committed deterministic decode at all 15,000 transitions.
* All aggregated JSON statistics reproduce exactly on independent recomputation; a hand-checked transition (seq 3, t=2: true R2 = 6, deterministic predicts 4, draws land on {0,4,7,8,9,6,…}) confirms the lottery behavior.

---

## Failure Anatomy (deterministic K=1 model)

| Quantity | Value |
|---|---|
| Test exact match (terminal) | 0.5367 (805 / 1500) |
| Per-transition target-register failures | 1,686 / 15,000 (11.2 %) |
| Per-transition full-state failures | 3,403 / 15,000 (22.7 %) |
| — of which ghost-only readout mismatches | 1,717 (50.5 % of failures) |
| Terminal failures | 695 |
| — with ≥ 1 real target-register error | 669 |
| — ghost-only (never a real target error) | 26 |
| SET write errors | **0** |
| Real write errors | ADD no-wrap 734, ADD wrap 512, SWAP 440 |

**Interpretation:** the Round-2 "failure" totals overstate real computational slips by ~2×. Strict per-step decoding of intermediate states contains substantial readout noise on registers that the operation did not touch; those ghosts are not latent corruption (many sequences still end correct). All SET full-state failures are of this ghost type.

---

## Required Table 1 — Recovery vs K (primary mechanism)

| K | Candidate diversity | Any correct target register | Any correct full state | Oracle recoverable failures | Oracle EM upper bound |
|---|---:|---:|---:|---:|---:|
| 1 | 0.000 / cos 1.000 | 0.8876 | 0.7731 | 0/695 (0.0 %) | 0.5367 |
| 2 | L2 8.593 / cos 0.785 | 0.8981 | 0.7925 | 57/695 (8.2 %) | 0.5693 |
| 4 | L2 9.735 / cos 0.723 | 0.9153 | 0.8175 | 162/695 (23.3 %) | 0.6360 |
| 8 | L2 10.297 / cos 0.693 | 0.9339 | 0.8448 | 241/695 (34.7 %) | 0.6913 |

* "Any correct…" columns are over all 15,000 transitions (oracle over K candidates incl. the deterministic one).
* "Oracle recoverable failures" = terminal K=1 failures that become correct under the perfect per-step verifier (K=8 also breaks 13–9 originally-correct sequences, so the net is slightly smaller than the recovered count).
* Diversity = mean pairwise L2 distance / mean pairwise cosine similarity among raw candidate states; branch-state diversity is smaller (K=8: L2 5.610 / cos 0.868). Cosine distance at K=8 is 0.307, above the project's Phase-4 diversity gate of 0.20 — but see the lottery diagnostic.

### Table 1b — conditioned on real (target-register) failures (1,686 transitions)

| K | Any candidate target ok | Any candidate full state ok | Oracle EM (trajectory) | Recovered terminal failures | Lost successes |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.5367 | 0 | 0 |
| 2 | 0.0937 | 0.0765 | 0.5693 | 57 | 8 |
| 4 | 0.2461 | 0.1934 | 0.6360 | 162 | 13 |
| 8 | 0.4122 | 0.3215 | 0.6913 | 241 | 9 |

Per-candidate lottery diagnostic on the same 1,686 failures (branch-state written-register accuracy; index 0 = deterministic, 1–7 = stochastic; chance = 0.10):

```
[0.000, 0.094, 0.104, 0.106, 0.092, 0.099, 0.106, 0.104]
```

Every stochastic draw is at chance. The candidate generator does not encode the correct next value in a retrievable way on failures; it re-samples the readout.

---

## Required Table 2 — Failure-type breakdown (K = 8)

Using **strict full-state failures** as the measured failure surface (1,075 / 3,403 recoverable; "Branch recoverable" = ≥ 1 candidate fully correct at that transition):

| Failure Type | Count | % of failures | Branch recoverable | Non-recoverable |
|---|---:|---:|---:|---:|
| SET/routing | 1,095 | 32.2 % | 438 (40.0 %) | 657 (60.0 %) |
| ADD no wrap | 976 | 28.7 % | 299 (30.6 %) | 677 (69.4 %) |
| ADD wrap | 570 | 16.7 % | 124 (21.8 %) | 446 (78.2 %) |
| SWAP | 762 | 22.4 % | 214 (28.1 %) | 548 (71.9 %) |
| Other | 0 | 0.0 % | 0 | 0 |
| **Total** | **3,403** | **100 %** | **1,075 (31.6 %)** | **2,328 (68.4 %)** |

### Table 2b — real failures only (target register wrong; ghosts excluded)

| Failure Type | Count | % of real failures | Branch recoverable (full state) | Target-register-only fix (partial) |
|---|---:|---:|---:|---:|
| SET/routing | 0 | 0.0 % | — | — |
| ADD no wrap | 734 | 43.5 % | 256 (34.9 %) | 329 (44.8 %) |
| ADD wrap | 512 | 30.4 % | 121 (23.6 %) | 160 (31.3 %) |
| SWAP | 440 | 26.1 % | 165 (37.5 %) | 206 (46.8 %) |
| **Total** | **1,686** | **100 %** | **542 (32.2 %)** | **695 (41.2 %)** |

Failure taxonomy on the 1,686 real failures at K=8: **A** no candidate correct 991 (58.8 %), **B** exactly one full-correct 307 (18.2 %), **C** multiple full-correct 235 (13.9 %), **D** target-only partial fix 153 (9.1 %).

---

## Required Table 3 — First-error timestep (K = 8)

Two definitions are reported because they differ materially.

**(a) Strict any-register first error** (Round-2/Q9 convention; includes ghosts):

| First error timestep | Failures | Branch recoverable | Recovery % |
|---|---:|---:|---:|
| 1 | 210 | 133 | 63.3 % |
| 2 | 7 | 7 | 100.0 % |
| 3 | 28 | 14 | 50.0 % |
| 4 | 36 | 19 | 52.8 % |
| 5 | 65 | 29 | 44.6 % |
| 6 | 49 | 15 | 30.6 % |
| 7 | 80 | 37 | 46.2 % |
| 8 | 75 | 26 | 34.7 % |
| 9 | 66 | 30 | 45.5 % |
| 10 | 79 | 35 | 44.3 % |

*Critical caveat:* **every** t=1 first error is a ghost readout mismatch — the modified register was decoded correctly (target-register recovery at t=1 is 210/210). This table mostly measures readout dithering, not computational errors.

**(b) First real target-register error** (computational slips only):

| First error timestep | Failures | Branch recoverable | Recovery % |
|---|---:|---:|---:|
| 3 | 40 | 19 | 47.5 % |
| 4 | 54 | 26 | 48.1 % |
| 5 | 75 | 30 | 40.0 % |
| 6 | 81 | 22 | 27.2 % |
| 7 | 96 | 38 | 39.6 % |
| 8 | 108 | 38 | 35.2 % |
| 9 | 104 | 48 | 46.2 % |
| 10 | 111 | 51 | 45.9 % |

(Timesteps 1–2 have zero real errors: the task forces SET operations, and SET writes never fail. The 26 ghost-only terminal failures are excluded here.)

End-to-end oracle EM recovery grouped by first strict error timestep (K=8): t=1: 70/210, t=2: 1/7, t=3: 9/28, t=4: 11/36, t=5: 22/65, t=6: 16/49, t=7: 28/80, t=8: 20/75, t=9: 27/66, t=10: 37/79.

---

## Required Table 4 — Control and selection diagnostics

**EXP 7 control (matched-norm random S_candidate perturbation; not a valid branching mechanism):**

| K | any target reg | any full state | oracle EM | recoverable failures |
|---|---:|---:|---:|---:|
| 1 | 0.8876 | 0.7731 | 0.5367 | 0/695 |
| 2 | 0.8949 | 0.7907 | 0.5493 | 27/695 |
| 4 | 0.9039 | 0.8111 | 0.5827 | 79/695 |
| 8 | 0.9141 | 0.8304 | 0.6080 | 119/695 |

Conditioned on real failures at K=8: control target-ok 0.2361, full-ok 0.1862 vs primary 0.4122 / 0.3215. Matched displacement norm = 8.59 (scale 0.754); control per-draw written-register accuracy ≈ 0.06–0.07 (below chance).

**One-shot repair (single oracle fix at the first failing transition, then fully deterministic):**

| K | fixed sequences | recovered failures | lost successes | oracle EM |
|---|---:|---:|---:|---:|
| 1 | 0 | 0/695 | 0 | 0.5367 |
| 2 | 248 | 40/695 | 12 | 0.5553 |
| 4 | 503 | 93/695 | 18 | 0.5867 |
| 8 | 706 | 110/695 | 22 | 0.5953 |

A single repair retains only ~38 % of the full per-step oracle EM gain (0.5953 vs 0.6913) and breaks 22 originally-correct sequences. Most of the oracle profit comes from re-rolling the lottery at every step, not from repaired states carrying correct computation forward.

**Sensitivity (oracle EM at K=8):** Gaussian c=0.25 → 0.6093; c=0.5 → 0.6593; **c=1.0 (primary) → 0.6913**; c=2.0 → 0.6613; dropout p=0.1 → 0.6047; p=0.2 → 0.6393; random-S_cand matched → 0.6080; random-S_cand c=1.0 → 0.6387. The primary scale (fixed a priori at the activation scale) is the most favorable of the sweep; the effect saturates around c≈1 and declines beyond, as untouched-register corruption grows. Target-register commit at K=8 (weaker verifier) → 0.6667.

**ADD wrap-around specific:** 512 real failures (30.4 % of real failures), local full-state recoverability 23.6 %, and 68/241 = **28.2 % of all oracle-recovered terminal failures** (own recovery 26.0 % by first-error operation). Wrap-around is slightly *under*-represented in recoverable mass relative to its failure share. Top deterministic confusions (true→predicted): 0→8, 2→4, 0→6, 2→8, 0→2, 1→7, 1→9, 1→3.

**Draw-correctness histogram (of 7 stochastic draws per failure transition):** ADD no-wrap: 55 % of transitions have 0 correct draws, max ≠ 7; ADD wrap: 69 % have 0 correct draws, never ≥ 5; SWAP: 53 % have 0 correct draws. Mildly overdispersed relative to independent resampling (per-transition heterogeneity), but no transition has all draws correct and only 3.1 % have ≥ 4/7 — again consistent with lottery resampling rather than a structured alternative path.

---

## Answers to the Required Q&A

**Q1. Can the current candidate generator produce multiple plausible next states?**  
Yes for diversity (K=8: raw pairwise L2 ≈ 10.3, cosine ≈ 0.69; branch-state cosine ≈ 0.87), but no for informativeness. On real failures, each stochastic draw decodes the written register at ≈ chance (0.10; ADD no-wrap ≈ 0.12, ADD wrap ≈ 0.06). Diverse but content-free with respect to the correct answer.

**Q2. Do those alternative states contain correct answers often enough to matter?**  
Under a perfect selector: yes — 32.2 % of real transition failures and 34.7 % of terminal failures contain a fully-correct candidate at K=8 (K=2: 7.7 %/8.2 %; K=4: 19.3 %/23.3 %). But the correct states are lucky draws, not structured alternatives: a matched random perturbation of S_candidate recovers roughly half as much (18.6 % of real failures; EM 0.608 vs 0.691), and no non-oracle selector has been demonstrated.

**Q3. What fraction of current K=1 failures are theoretically recoverable?**  
Terminal, oracle per-step: **34.7 %** (241/695; EM 0.5367 → 0.6913). Terminal, single repair: **15.8 %** (110/695 → 0.5953). Real transition failures, oracle: 32.2 % full-state / 41.2 % written-register at K=8. None of these is achievable without ground-truth selection.

**Q4. How much of the potential recovery comes specifically from ADD wrap-around?**  
Little. ADD wrap is 30.4 % of real failures but only 23.6 % locally recoverable and 28.2 % of oracle-recovered terminal failures (68/241); its own recovery rate (26.0 %) is below ADD no-wrap (37.0 %) and SWAP (53.0 %). Wrap-around is one of the *least* recoverable error types; branching does not specifically solve it.

**Q5. Would a perfect verifier have anything useful to select?**  
Formally yes: 735/1500 sequences receive ≥ 1 oracle fix and 241 terminal failures flip to correct (target-only commit variant: 207/695, EM 0.6667), at the cost of breaking 9–22 already-correct sequences. But what exists to select is resampling luck (per-draw information = chance), and ~62 % of the per-step oracle gain disappears when a single repair must carry the trajectory. The perfect verifier is doing rejection sampling against ground truth, not exploiting a hidden computation.

**Q6. Does the evidence justify implementing speculative branching?**  
Not as specified. The hypothesis's implicit premise — that candidate diversity contains *useful alternative computations* — is falsified: on failures the diversity is information-free (chance-level per draw), the generator-specific advantage over matched random latent noise is modest, single repairs rarely stick, and no deployable selector exists. The project's own Phase-4 failure semantics ("branches… behave as unhelpful random noise") is met on the informativeness axis even though the numeric diversity gate (cosine distance 0.307 > 0.20) is passed.

**Q7. If yes, what should actually branch? (choose one)**  
**C. discrete register hypotheses** (conditional on a future, *learned* selector). The only oracle-recoverable object measured is the discrete register value emitted by the readout. Branching over latent candidate states (B) is empirically equivalent to matched random noise; full trajectories (D) are unsupported by the one-shot durability result; A is not a branch target.

**Q8. What should NOT branch?**  
The gate (never perturbed here; the earlier gate-scaling experiment degraded K=4 exact match from 0.504 to 0.322 with zero oracle recovery — gate perturbation is an invalid mechanism); the state update/commit equation; and no stochastic branch should be committed without a non-oracle selection signal. The deterministic K=1 trunk should remain the single trajectory.

**Q9. Minimum Phase-2 experiment that tests this hypothesis (redesigned):**  
Train a candidate **selector** on frozen-model candidate sets from the train/val splits only (no test truth at inference): input `(S_(t-1), e_t, candidate decode/state)`, target = per-step correctness; then measure test-split selection accuracy / FAR / FRR / ECE against chance and compare compute-matched "K=4 + selector" vs "K=1" EM. Branching is justified only if the selector ranks correct candidates above chance and the compute-parity EM improves. If the selector fails, the correct follow-up is readout/commit calibration on the K=1 trunk, not branching.

---

## Final Scientific Classification

1. **"Branching can recover existing errors." — ESTABLISHED FACT (oracle-only).** Direct measurement: 34.7 % of deterministic terminal failures are recovered by a perfect per-step selector over K=8 sampled candidates (EM 0.5367 → 0.6913); 32.2 % of real transition failures contain a fully-correct candidate. The claim is established strictly in the sense "the candidate sets contain answers a perfect selector prefers", not as deployable recovery.
2. **"The verifier can select useful alternatives." — NOT SUPPORTED.** No verifier was trained or tested. The only selector used is ground truth; the oracle is not a verifier. Moreover, the selectable objects are chance-level draws (~10 % per draw), and even perfect selection yields only +5.9 EM points when a single repair must carry the trajectory, while breaking 22 successes. No evidence that a learnable selector can isolate the lucky draws.
3. **"The current errors contain enough ambiguity for branching to matter." — SUPPORTED INTERPRETATION, with a decisive caveat.** Recoverable ambiguity exists (241/695 terminal failures), but it is *unstructured resampling ambiguity*, not latent representational ambiguity: matched random perturbation recovers 119/695 (46 % of the EM gain), per-draw information is at chance, and one-shot repairs retain ~38 % of the gain. It is not supported that the ambiguity is a usable computational resource without oracle-grade selection.
4. **"Gate perturbation is a valid branching mechanism." — NOT SUPPORTED.** This experiment froze the gate by design, and the audited gate-scaling variant (Round-2 Q17) degraded accuracy (K=4 exact 0.504 → 0.322) while providing no oracle recovery. Perturbing the gate is a different, destructive mechanism.
5. **"Candidate-level branching is preferable to gate perturbation." — SUPPORTED INTERPRETATION.** With the gate held fixed, candidate-only stochasticity produces monotone oracle recovery (0.5367 → 0.6913, no losses to the deterministic baseline under oracle commitment beyond 8–13 sequences), whereas gate perturbation degraded accuracy. The preference holds only under oracle selection; without a selector, candidate sampling has no measured value.

---

## Final Decision

**2. FREEZE PHASE 1 BUT REDESIGN THE PHASE-2 HYPOTHESIS.**

**Why the branching hypothesis must change:**

* The premise "sampled candidate states provide useful alternative computations to search over" is falsified. Diversity is large but its informativeness on failures is at chance (per-draw written-register accuracy 0.05–0.13 vs 0.10 chance); the oracle's 34.7 % recovery is rejection-sampling a near-uniform readout distribution, not retrieving a hidden computation.
* Half of the oracle EM gain (and ~58 % of the local real-failure recovery) is matched by an explicitly invalid control (magnitude-matched random perturbation of S_candidate). The candidate generator's internal stochasticity is only modestly more useful than generic latent noise.
* The binding constraint is **selection/commit**, not generation or search. The deterministic model's failures are concentrated in ADD/SWAP writes and in readout ghosts; a single oracle repair carries only ~38 % of the oracle gain and can break correct trajectories. A Phase-2 program built around "more branches" attacks the wrong bottleneck and pays 8× candidate compute for it.
* Redesign direction: (i) treat the problem first as **verification and readout calibration on the single K=1 trunk** (learned confidence/selection signal, no ground truth at inference); (ii) if branching is revisited, it must branch **discrete register hypotheses (C)** with a *learned, compute-parity-audited selector* — and the minimum Phase-2 experiment in Q9 is the gate for that decision. Branching is not killed outright: the oracle bound shows recoverable mass exists; what is killed is the assumption that the current candidate generation makes that mass usable.

**Do not** implement speculative branching as specified; do **not** treat the oracle EM bound (0.6913) as achievable performance; do **not** use gate perturbation as the branching mechanism.
