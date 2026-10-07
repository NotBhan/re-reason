# RE:Reason — Phase-2 Verifier Feasibility: Can a Learned Selector Recognize a Good Transition Without Ground Truth?

**Date:** October 5, 2026  
**Status:** Completed — Feasibility Experiment (strong negative result)  
**Target Model:** Nonlinear Candidate MGU (Seed-42 checkpoint, 226,358 params) — frozen core, encoder, candidate network, gate, readout; no weight updates  
**Data Reference:** [`experiments/verifier_feasibility_results.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/verifier_feasibility_results.json)  
**Runner:** [`experiments/run_verifier_feasibility.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/run_verifier_feasibility.py)  
**Figures:** [`docs/reports/figures/verifier_reliability.png`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/reports/figures/verifier_reliability.png), [`verifier_methods_em.png`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/reports/figures/verifier_methods_em.png)

---

## Executive Answer

**A learned selector cannot usefully recognize a good transition from inference-only information. The apparent signal is candidate-identity / base-rate structure, not validity detection.**

1. **Headline metrics look healthy, but they are an artifact.** The MLP selector reaches AUROC 0.879, top-1 selection 0.7747, and ECE 0.061. But the *always-commit-the-deterministic-candidate* policy already scores top-1 = **0.7731** (max possible, oracle = 0.8179). The selector captures **+0.0016 (+0.16 pt)** — 3.6 % of the available headroom. A 6-feature magnitude model reaches AUROC 0.781 (89 % of the full model), and the same AUROC (0.877) transfers to **pure matched random-noise candidates**.
2. **End-to-end, learned selection loses to doing nothing.** K=4 + MLP selector: EM **0.4940**; K=4 + linear: 0.5293; K=1 deterministic: **0.5367**; oracle: 0.6247. The selector improves over random K=4 (0.1707) only by learning to commit the deterministic candidate 86–99.5 % of the time — i.e., it degenerates into the K=1 baseline, and its rare deviations are net-negative.
3. **The decision-relevant signal is weak and does not survive.** On transitions where the deterministic candidate is wrong, deciding which candidate to commit is at random-pick level (MLP overall top-1 success 5.85 % vs 5.3 % random-pick; best-draw selection 10.8 % vs ~7.1 % chance). Draws-only AUROC restricted to those transitions is 0.639; a selector trained on random noise reaches 0.592 within its own family, and the genuine-candidate signal does not transfer to noise (0.518 ≈ chance).
4. **The verifier collapses exactly where errors live.** Top-1 by operation: SET 0.859 (deterministic baseline 0.844), SWAP 0.801 (0.807), ADD no-wrap 0.678 (0.693), **ADD wrap 0.329 (0.339)** — at or below the trivial baseline on every error-heavy op.
5. **Verification cannot manufacture the missing information.** At a failing transition, no inference-available feature distinguishes the correct candidate from incorrect ones beyond prior-level statistics; the selector's only reliably learnable policy is "trust the deterministic candidate". Even a perfect selector (oracle) is not achievable, and its ceiling for this loop (K=4: 0.6247) remains far below what a better generator could offer.

---

## Method (what was frozen, what was trained)

* **Frozen:** recurrent core, observation encoder, candidate network, gate network, terminal readout. No weight updates to any of these.
* **Candidate sets:** same stochastic mechanism as the Phase-1 branching experiment — Gaussian activation noise on the candidate MLP hidden (σ = activation scale, estimated on the verifier-train split); candidate index 0 = deterministic candidate, 1…K−1 = stochastic draws. Branch state `B_k = (1 − f_t)·S_(t−1) + f_t·C_k`. K = 4 primary; K = 8 ranking secondary.
* **Label (VALID/INVALID):** the committed branch state `B_k` decodes (model's own head) to the exact true 4-register next state — primary; target-register match — secondary. Base rates (test): det full 0.773, draws full 0.622, all candidates 0.660; target-register 0.778.
* **Features (inference-only):** candidate state, previous state, gate, instruction embedding + raw observation one-hots, deterministic baseline candidate, candidate-vs-state/baseline deltas, distances/cosines/norms, decoded logits/probabilities/entropy/margins of the candidate and baseline. 1,907 dims total; standardized on train. **No ground truth, no future observations, no oracle labels.**
* **Splits (strictly disjoint):** verifier-train = 1,200 sequences (seed stream 2042), verifier-val = 500 sequences (stream 51042), test = the frozen model's untouched test split (stream 90042, 1,500 episodes / 15,000 transitions). No test-set information was used for training, architecture, feature, or threshold choice.
* **Models:** logistic linear (lr 5e-3) and small MLP (256→128, lr 1e-3, GELU), AdamW (wd 1e-4), batch 4,096, ≤30 epochs, early stop on val AUROC. Decode threshold τ = 0.5 (primary); val-tuned τ* = 0.9 (secondary). Policy: commit argmax score if ≥ τ, else deterministic fallback.
* **Oracle/verifier language:** the ground-truth selector is called the *oracle*, never a verifier; oracle EM is not achievable performance; stochastic candidate diversity is not called alternative reasoning.

Validation: deterministic K=1 test EM reproduces exactly (0.5367); candidate-0 labels equal the committed deterministic-state labels on train and test; K=1 rollout equals the deterministic model exactly; all aggregate partitions and headroom calculations re-verified independently.

---

## Required Primary Table

| Method | EM | Per-Reg | Top-1 Candidate Selection | AUROC | FAR | FRR | ECE | FLOPs/Episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K=1 | 0.5367 | 0.8470 | N/A | N/A | N/A | N/A | N/A | 4.34M |
| K=4 Random | 0.1707 | 0.6353 | 0.6600 (expected) | N/A | N/A | N/A | N/A | 6.20M |
| K=4 Oracle | 0.6247 | 0.8705 | 100% (solvable rate 0.8179) | 1.0 | 0 | 0 | 0 | 6.20M |
| K=4 Linear Selector | 0.5293 | 0.8442 | 0.7723 | 0.8549 | 0.3060 | 0.1595 | 0.0563 | 7.01M |
| K=4 MLP Selector | 0.4940 | 0.8280 | 0.7747 | 0.8791 | 0.3220 | 0.1173 | 0.0606 | 44.22M |

* Top-1 = fraction of transitions where the committed candidate is a valid one (unconditional). Always-deterministic baseline: **0.7731**; random candidate: 0.6600; oracle (∃ valid): 0.8179. FLOPs are episode totals for the frozen model + candidates + selector (verifier forward included); oracle compute excludes verification by definition.
* Selector hyperparameters and the τ = 0.5 threshold were fixed; τ* = 0.9 was selected on val only and yields EM 0.5107 (still below K=1).

## Task 1 — Candidate ranking (test, K=4)

| Metric | MLP | Linear | Reference |
|---|---:|---:|---|
| Top-1 selection accuracy | 0.7747 | 0.7723 | always-det 0.7731; random 0.6600; oracle 0.8179 |
| Top-2 recovery | 0.7987 | 0.7950 | oracle 0.8179 |
| MRR | 0.9688 | 0.9664 | — |
| AUROC (all candidates) | 0.8791 | 0.8549 | 0.5 chance |
| AUPRC (all candidates) | 0.9304 | 0.9147 | base rate 0.660 |
| Top-1 given a valid candidate exists | 0.9472 | 0.9442 | always-det 0.9452 |
| **Decision-relevant draws AUROC (det wrong)** | **0.6388** | **0.6395** | 0.5 chance |
| **Best-of-draws top-1 (det wrong)** | **0.1084** | **0.0967** | ~0.071 chance |
| **Overall top-1 success (det wrong)** | **0.0585** | **0.0012** | ~0.053 random-pick |
| Target-register label top-1 | 0.8775 | 0.8864 | always-det 0.8876; oracle 0.9148 |

K=8 secondary (same selector, 8 candidates): top-1 0.7745, top-1|exists 0.9184 (always-det 0.9168), ∃-rate 0.8433. No K-dependence: consistent with the selector picking the deterministic candidate and ignoring extra draws.

**Reading:** the headline ranking skill is the ability to identify *which candidate is the deterministic baseline* and the base rate of the transition — not candidate validity. The decision-relevant numbers (+0.2 pt top-1 over doing nothing; 5.85 % vs 5.3 % rescue rate) show no practical selection ability.

## Task 2 — Do full trajectories show a benefit?

| Method | EM | Per-Reg | Continuation* | Draw commits | Latency/ep | FLOPs/ep |
|---|---:|---:|---:|---:|---:|---:|
| A. K=1 | 0.5367 | 0.8470 | 0.7934 | 0 | 0.034 ms | 4.34M |
| B. K=4 Random | 0.1707 | 0.6353 | 0.3810 | 75.1% | 0.141 ms | 6.20M |
| C. K=4 Oracle | 0.6247 | 0.8705 | 0.8824 | 4.4% | 0.153 ms | 6.20M |
| D. K=4 Linear sel. | 0.5293 | 0.8442 | 0.7917 | 0.53% | 0.413 ms | 7.01M |
| D. K=4 MLP sel. | 0.4940 | 0.8280 | 0.7667 | 14.3% | 0.563 ms | 44.22M |
| (val-tuned τ*) MLP | 0.5107 | 0.8358 | 0.7815 | 10.2% | 0.572 ms | 44.22M |

\* P(terminal correct | committed state fully correct at t=5), a small-sample diagnostic (120/1500 episodes).

**Critical comparison D vs B:** learned selection beats random selection by +32.3 pts (MLP) / +35.9 pts (linear). **D vs K=1:** learned selection *loses* by −4.3 pts (MLP) / −0.7 pts (linear). The more the policy deviates from the deterministic candidate, the worse the EM — the MLP commits 14.3 % draws and loses 4.3 pts; the linear commits 0.5 % and loses 0.7 pts.

## Task 3 — Compute-matched controls

| Control | EM | Compute (FLOPs) | Latency/ep |
|---|---:|---:|---:|
| 1× K=1 | 0.5367 | 4.34M (1.0×) | 0.034 ms |
| 4 identical deterministic runs (ensemble) | 0.5367 | 17.4M (4.0×) | — (trivially identical) |
| 4 independent stochastic K=1 runs + vote | 0.2413 | 17.4M (4.0×) | 0.253 ms |
| K=1 with equivalent recurrent compute | N/A — frozen architecture has no R>1 path; adding one would violate the freeze | | |
| K=4 + MLP verifier | 0.4940 | 44.2M (10.2×) | 0.563 ms (16.6×) |

No structural advantage anywhere: 4× compute on stochastic ensembles is far *worse* than 1× K=1 (0.241 vs 0.537); the learned selector at 10.2× compute is still worse than K=1; the linear verifier at 1.6× compute reaches "always-deterministic" behavior (0.529) but not K=1. Peak process RSS was shared (~2.5 GB, dominated by feature matrices); per-step analytical working set ≈ 5 MB (generation) vs ≈ 46 MB (feature scoring).

## Task 4 — Calibration

| | FAR | FRR | ECE | Brier |
|---|---:|---:|---:|---:|
| Linear | 0.306 | 0.160 | 0.056 | 0.147 |
| MLP | 0.322 | 0.117 | 0.061 | 0.136 |

Reliability (MLP) is monotone but systematically overconfident above 0.5: the ~0.98-confidence bin (25,073 of 60,000 candidates) is 94 % accurate; the 0.7 bin is 61 % accurate. Constant base-rate Brier = 0.224, so there is modest skill. Calibration is *tolerable*; discrimination is the failure, not calibration.

## Task 5 — What signal is the verifier using? (feature ablations, MLP AUROC)

| Feature set | Dims | AUROC (all) | Decision-relevant AUROC |
|---|---:|---:|---:|
| A. candidate only | 212 | 0.7869 | 0.560 |
| B. candidate + previous state | 424 | 0.8423 | 0.622 |
| C. candidate + gate | 424 | 0.8438 | 0.599 |
| D. candidate + instruction | 489 | 0.7967 | 0.570 |
| E. candidate + decoded probabilities | 304 | 0.7976 | 0.587 |
| F. all available | 1907 | 0.8791 | 0.639 |
| diag: norms only (6 dims) | 6 | 0.7813 | 0.571 |
| diag: candidate direction only | 212 | 0.7881 | 0.568 |

Previous state and gate provide most of the lift; the decoded register probabilities add almost nothing (0.7976 vs 0.7869 over candidate-only); and **six magnitude-distance scalars reproduce 89 % of the full-model AUROC**. The apparent ranking skill is predominantly candidate-identity and magnitude structure, not validity information. Linear ablations show the same ordering at slightly lower levels.

## Task 6 — Random-baseline sanity check (matched random-noise candidates)

| Evaluation | AUROC (all) | Decision-relevant AUROC | Top-1 | Valid rate |
|---|---:|---:|---:|---:|
| Primary-trained MLP on genuine candidates | 0.8791 | 0.6388 | 0.7747 | 0.660 |
| Primary-trained MLP on matched noise | 0.8766 | **0.5179** | 0.7759 | 0.757 |
| Noise-trained MLP on matched noise | **0.8866** | 0.5923 | 0.7758 | 0.757 |

The headline AUROC is unchanged on pure random-noise candidates (0.877 vs 0.879) — it is not measuring genuine-candidate validity. The weak decision-relevant signal does not transfer (0.518 ≈ chance), and the noise family is equally learnable within-family (0.592). Additionally, matched-noise draws are *more* often valid (0.752) than the genuine stochastic draws (0.622): the candidate mechanism's stochasticity is not a better source of correct states than a matched random perturbation.

## Task 7 — Generalization

**By operation (MLP):**

| Op | Selector top-1 | Deterministic baseline | Best draw (det wrong) |
|---|---:|---:|---:|
| SET | 0.859 | 0.844 | 0.836 |
| SWAP | 0.801 | 0.807 | 0.669 |
| ADD no wrap | 0.678 | 0.693 | 0.536 |
| ADD wrap (a+b ≥ 10) | **0.329** | 0.339 | 0.232 |

At or below the trivial baseline on SWAP and both ADD splits; worst exactly on the wrap case that dominates the Phase-1 error budget. **By first-error timestep:** AUROC is flat (0.83–0.88) — no timestep-specific competence. **By episode outcome:** 0.886 on terminal-fail episodes vs 0.857 on successes (more invalid candidates to score). **By label type:** target-register top-1 (0.8775) is below the always-deterministic target baseline (0.8876); oracle 0.9148. **By candidate rank:** in 11,621/12,269 solvable transitions the first valid candidate sits at rank 1 of the selector's ordering (the deterministic candidate); the 648 det-failure transitions are not rescued (success ≈ random-pick).

## Task 8 — Failure analysis (MLP, τ = 0.5; 15,000 test transitions)

| Category | Count | % of all | % of solvable (12,269) |
|---|---:|---:|---:|
| A. correct candidate found and chosen | 11,033 | 73.6 % | 89.9 % |
| B. correct candidate found, wrong choice | 365 | 2.4 % | 3.0 % |
| C. no correct candidate exists | 2,731 | 18.2 % | — |
| D. selector rejects all candidates | 871 | 5.8 % | 7.1 % |
| E. confidently wrong choice (⊂ B, p ≥ 0.9) | 170 | 1.1 % | 1.4 % |

A ≈ "the deterministic candidate was valid and committed". The selector's residual failures are abstentions and confident picks of the *deterministic* candidate on transitions where it was wrong — not recoveries of the available draws. Linear: A 10,298 / B 229 / C 2,731 / D 1,742 / E 176.

## Task 9 — One-shot vs recursive verification

The selector is purely local by construction (features at `(S_(t-1), e_t, candidate)` only; no trajectory context). The test: can a correct candidate at a single timestep be recognized? **Only weakly** — decision-relevant AUROC 0.639 and best-draw top-1 of 10.8 % vs ~7.1 % chance — and that weak signal does not survive the det-vs-draw commit decision or end-to-end rollouts. Local one-shot verification is therefore **not** supported; a multi-step verifier would be required, but it would have to *create* information (recompute or cross-check), not merely aggregate the same features.

---

## Decision-Criteria Evaluation

| # | Criterion | Verdict |
|---|---|---|
| 1 | Beats random candidate selection | **Yes** (top-1 0.7747 vs 0.660; EM 0.494 vs 0.171) — but only by learning to commit the deterministic candidate |
| 2 | Generalizes to held-out test | **Yes** (val AUROC 0.885 → test 0.879) |
| 3 | Meaningfully above chance in ranking | **No** (decision-relevant AUROC 0.639; +0.2 pt top-1 over always-det; +0.5 pt rescue rate over random-pick) |
| 4 | Improves EM over random K=4 | **Yes trivially**, but **No vs K=1** (−4.3 pts MLP, −0.7 pts linear) |
| 5 | Survives compute-matched controls | **No** (1× K=1 beats 10.2× learned K=4 and 4× stochastic ensemble) |
| 6 | Calibration acceptable | **Marginal** (ECE 0.06; overconfident at high confidence) |
| 7 | Does not collapse on ADD-wrap | **No** (top-1 0.329, at/below the 0.339 baseline; zero end-to-end wrap rescues) |
| 8 | Does not exploit norm / magnitude | **No** (6 norm features = 89 % of full AUROC) |
| 9 | Better on genuine than matched noise | **No** (AUROC equal; decision-relevant signal fails to transfer; noise equally learnable within-family) |

The selector is not promising under the stated criteria: it fails 3, 5, 7, 8, 9 and only trivially passes 1 and 4.

---

## Final Q&A

**Q1. Can candidate correctness be predicted without ground truth?** Not usefully. Overall AUROC 0.88 reflects candidate identity and base rates (a 6-feature magnitude model gets 0.78; the same AUROC transfers to random noise). The decision-relevant signal — which candidate is correct when the deterministic one is wrong — is weak (AUROC 0.64) and produces no end-to-end gain.

**Q2. Does the latent state contain enough information for a local verifier?** No. At failing transitions the state/observation/candidate features carry only prior-level hints (e.g., which ops are draw-friendly), not evidence that a specific candidate matches the unobserved true next state.

**Q3. Does a learned verifier outperform random selection?** Yes, trivially: it learns to commit the deterministic candidate (86–99.5 % of transitions), which random selection does not.

**Q4. Does learned selection recover a meaningful portion of the oracle gap?** No. Top-1 headroom above always-deterministic is +4.5 pts; the selector captures +0.16 pts (3.6 %). EM headroom is +8.8 pts (0.5367 → 0.6247); the selector captures **−4.3 pts** (negative).

**Q5. Is verification more promising than raw branching?** As formulated, no. Both fail to beat K=1; verification's only learnable policy collapses to the K=1 baseline, and its deviations are negative-expected-value.

**Q6. Does candidate validity remain locally identifiable at a single timestep?** Weakly, but insufficiently. Best-draw selection beats chance by ~3.7 pts (10.8 % vs 7.1 %) and decision-relevant AUROC is 0.64; this does not survive the commit decision or the rollout.

**Q7. Does the verifier generalize across operation types and ADD wrap-around?** No. It is at/below the deterministic baseline on SWAP (0.801 vs 0.807), ADD no-wrap (0.678 vs 0.693) and ADD wrap (0.329 vs 0.339) — collapsing on the Phase-1 error epicentre.

**Q8. Does compute-matched selection still provide an advantage?** No. 1× K=1 (0.5367) beats 4× stochastic ensembles (0.2413) and the 10.2×-compute learned K=4 (0.4940). The verifier's extra compute buys a loss.

**Q9. What information does the verifier actually use?** Predominantly (i) candidate identity/magnitude relative to the deterministic baseline (delta = 0, zero distance) and (ii) step-level context (previous state, gate, operation) that predicts *when draws tend to be valid*. Decoded register probabilities add essentially nothing. It is not using, and cannot access, per-candidate validity evidence.

**Q10. Is the current Phase-2 hypothesis now: A. viable, B. viable but requires redesign, C. unsupported?** **C. Unsupported.**

---

## Final Decision

**3. ABANDON VERIFIER/BRANCHING FOR NOW.**

**What the experiments indicate the real bottleneck is:**

* **Information, not selection.** At a failing transition, the information needed to recognize a correct candidate — the true next state — is absent from everything available at inference. Validation with ground truth confirms the channel is empty: the genuine decision-relevant signal is weak (AUROC 0.639), does not transfer to matched noise (0.518), and is matched within-family by a noise-trained selector (0.592). A classifier over available features cannot recover an answer the model never computed; verification cannot manufacture information.
* **The learnable optimum is the K=1 baseline.** The deterministic candidate is valid 0.773 of the time and no draw feature predicts validity well enough to beat it; every policy that deviates more (MLP, 14.3 % draws) loses more EM than the conservative linear policy (0.5 % draws). The best "verifier" is "commit the deterministic candidate" — i.e., no branching at all.
* **The bottleneck is upstream, in the base model's error structure.** The errors that matter (ADD wrap 8 % accuracy, SWAP, and the ghost readout mismatches documented in Phase 1) are representation/computation failures of the frozen core. Search and verification can only recombine what the core already represents; here, they recombine chance-level draws. The productive direction is Phase-1-side work on the generator's error structure — not Phase-2 verification machinery.
* The oracle K=4 (0.6247) and K=8 (0.6913) recoveries remain oracle-only artifacts of selecting lottery tickets with ground truth; they are not achievable and should not motivate Phase-2 work.

**Explicitly not supported:** that candidate ranking is meaningful, that a local verifier can select useful alternatives, that learned selection improves outcomes, that compute-matched selection has an advantage, and that verification is the right next mechanism. No minimal Phase-2 verifier should be implemented; no redesign of the local-selector hypothesis is supported by these data.
