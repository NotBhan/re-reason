# RE:Reason — Experiment L: Readout-Side Localization of the Off-Path Component

**Date:** October 7, 2026
**Status:** Completed — Measurement only (frozen checkpoints; no training; no readout/gate/commit modification; no intervention; no recursive counterfactual trajectories). Classification: **B — primarily latent-state placement** (E primary; the F control agrees).
**Frozen models:** `E_candidate` (official Phase-1 objective, primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned
**Frozen readout:** `TypedDecisionHead` = a single `nn.Linear(212 → 40)`; `z_{d,v}(S) = W[d,v]·S + b[d,v]` exactly
**Data:** `experiments/experiment_l_readout.npz`, `experiments/experiment_l_readout_meta.json`, `experiments/experiment_l_results.json`
**Runners:** `experiments/run_experiment_l.py`, `experiments/analyze_experiment_l.py`, tests `tests/test_experiment_l.py`
**Checkpoints:** `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against Experiment G metadata)

---

## 1. Executive Answer

**The off-path failures are a latent-state placement problem, not a readout/bias problem.** On the exact primary population (candidate-correct, committed-wrong, `ρ_margin ≥ α*_full`), the committed state has *cleared the strongest competitor that was binding at `S_prev` in 100% of cases* (`m(S_commit; a_prev) ≥ 0`, median **+1.96**), yet the committed argmax is a **different competitor in 100% of cases** and decodes to a **third value in 100% of cases** (never the old value, never the correct value). The fixed readout bias is essentially never decisive: removing it (analysis-only) flips the decode to correct in only **1.2% (E) / 3.1% (F)** of off-path failures, and the bias accounts for a median **7.1%** of the winning-pair margin. Classification: **B — primarily latent-state placement** (r_place = 0.988 E / 0.970 F; r_bias = 0.012 / 0.031).

The required answers:

1. **Boundary-shortfall fraction:** **0.0%** (the deterministic category is empty; by measurement, 100% of off-path failures clear the intended competitor — they do not merely stop short of the `S_prev` boundary).
2. **Readout/calibration signature fraction:** **1.24% (E) / 3.12% (F)** — the share for which removing the fixed bias changes the committed decision to the correct value.
3. **Latent-state-placement signature fraction:** **98.76% (E) / 96.88% (F)** — the committed state cleared the intended competitor but a *different* competitor is the committed argmax.
4. **Mixed/unresolved fraction:** **0.0%**.
5. **Does the fixed bias materially matter?** **No.** Bias share of the winning-pair margin: median **7.1%**, p95 47.2% (E); removing bias changes the committed decision on only **2.8%** of off-path failures (vs 0.7% of all primary transitions) and flips only ~1–3% to correct. The bias is small and rarely decisive.
6. **Does the committed state move toward a competing basin?** **Yes.** The intended competitor is cleared (normalized hyperplane distance `+0.52` at commit vs `−3.13` at `S_prev`), but the state sits **below** the winning competitor's boundary (normalized distance `−0.168`, E), the committed competitor changes relative to `S_prev` in **100%** of off-path failures (vs **65%** of successes), and perturbations around `S_commit` rarely reach the correct basin (perturbed-decode-correct 1.7% / 7.9% / 13.1% at ε = 0.01 / 0.05 / 0.10) — the state is locally stable *inside the wrong basin*, not near a boundary.
7. **E versus F:** statistically indistinguishable on every signature (all paired p ≥ 0.40; competitor-change and cleared-intended are identical at 1.0 in both). The localization does not depend on the training objective.
8. **Dominant explanation:** **B — latent-state placement.**
9. **Exactly one next experiment (§16):** **Experiment M — representation geometry of the third basin** (measurement-only): characterize *what* the third (winning) basin is and whether the misplacement is a systematic modular/wrap geometry, so that any later representation-side change is grounded in a measured mechanism rather than a search.

L is a **measurement-only localization**. It does not implement, and does not propose, a fix; and it does not claim that the readout or the representation is "causing" the failure.

---

## 2. Existing Baseline (what G–K established, and what L resolves)

* **G** (`experiment_g_add_commit_gap.md`): a real ADD candidate→commit residual (~5.5% of candidate-correct adds), concentrated in general two-register adds; gate-dilution vs blend/readout evidence mixed (C); scalar mean gate a proxy.
* **H** (`experiment_h_lambda_interpolation.md`): the element-wise vector-gate commit is bit-exact; the committed state lies off the scalar `S_prev→C` path (mean off-path distance ≈ 6.3–6.5 on a segment ≈ 15); scalar mean-gate reasoning is inadequate.
* **I** (`experiment_i_coordinate_commit.md`): coordinate-wise decision-margin analysis; failures under-deliver the decision-helping coordinates and start from weaker candidate margins, **but 44–48% delivered at least the scalar-equivalent transfer needed (`ρ_margin ≥ α*_full`) yet still failed off-path** — the off-path/placement component that L now localizes.
* **J** (`experiment_j_commit_intervention.md`): commit-only magnitude scaling raises local margin delivery but improves no sequence endpoint; γ≥1.25 harmful; coordinate identity essential.
* **K** (`experiment_k_gate_balance.md`): commit-only *shape/balance* remapping achieves the favorable delivery profile (helpful delivery up, harmful delivery down) yet still improves no sequence endpoint and leaves the off-path failures (≈6% rescued; 100% competitor change).
* **L (this report)** measures the frozen linear readout directly on the off-path population and finds the residual is latent-state placement (a third, wrong basin), not fixed readout bias/calibration.

## 3. Research Question

> When the candidate is correct and the actual committed state contains substantial useful decision movement, why can the frozen readout still decode the wrong value: **readout-side bias/calibration** (adequate latent placement, unfavorable fixed offsets/class geometry) or **latent-state placement geometry** (the committed state itself lies in the wrong region of latent space)?

## 4. Hypotheses and Pre-Registration

**Hypotheses (declared before the formal analysis).**

* **L-A (readout/calibration).** A substantial share of off-path failures are corrected by removing the fixed readout bias while holding the latent state fixed, and/or fixed bias accounts for a large share of the required margin — i.e. the committed latent placement is adequate but the frozen readout's offsets make the decision unfavorable.
* **L-B (latent placement).** A substantial share of off-path failures show the committed state entering a *different* competitor's basin (competitor change and/or fully delivered intended-direction movement yet failure), with only negligible bias sensitivity.
* **L-C (both).** Both signatures are non-trivial.
* **L-D (inconclusive).** Neither dominates.

**Pre-registered category rules** (fixed in `analyze_experiment_l.py` before the aggregate was inspected; per transition):

```
bias_sig = the bias-removed readout argmax at S_commit equals the true value
moved    = <w_y − w_{a_prev}, S_commit − S_prev> > 0          (intended-direction movement)
cleared  = m(S_commit; a_prev) ≥ 0                            (intended competitor cleared)
changed  = committed strongest competitor != S_prev strongest competitor
READOUT (readout_calibration)  = bias_sig
PLACEMENT (latent_misplacement)= not bias_sig and cleared and changed
SHORTFALL (boundary_shortfall) = not bias_sig and not PLACEMENT
MIXED                          = non-finite inputs
```

**Pre-registered localization decision** (three-way, from converging evidence): `r_bias` = share with `bias_sig`, `r_place` = share with PLACEMENT; **readout** if `r_bias ≥ 0.30` and `r_place < 0.60`; **latent placement** if `r_place ≥ 0.60` and `r_bias < 0.30`; **both** if both bars are met; **inconclusive** otherwise. The bars (0.30 / 0.60) are declared conventions, not data-tuned cutoffs; the raw shares are reported so the verdict can be judged independently.

**Primary population** (the exact established I/J/K off-path definition, **not redefined after seeing results**): strict test split | ADD transition | general stratum (`dest ≠ src` AND source operand `> 0`) | candidate correct | baseline commit wrong | `ρ_margin ≥ α*_full`.

## 5. What Is Frozen

Recurrent core, candidate network, observation encoder, gate network, **readout**, benchmark, task generator, optimizer, loss, and all Phase-2 machinery. No training, no weight modification, no optimizer, no gate/commit rescaling or reshaping, no readout change; all inference under `torch.no_grad()`. Every counterfactual (bias removal, perturbation) is a **one-step analysis-only** computation on frozen states, never used in a rollout, and never a new prediction.

## 6. What Changed

New code and artifacts only (`run_experiment_l.py`, `analyze_experiment_l.py`, `tests/test_experiment_l.py`, the readout npz + meta, results JSON, 12 figures, this report). No completed report was modified; Experiments G/H/I/J/K reports and artifacts are untouched. Experiment I's `experiment_i_coordinate.npz` and Experiment H's `experiment_h_interp.npz` are read (read-only) for cross-experiment alignment.

## 7. Dataset and Protocol

* **Data.** For each of E and F × seeds {42..46}: one `forward_with_states(..., return_components=True)` pass over the **canonical strict test split** (1,200 sequences × 10 steps, never trained on; stream `90042+seed`).
* **Scale.** 15,763 ADD test transitions per model; the off-path primary population is **323 transitions (E)** and **353 (F)** — 2.05% / 2.24% of test ADDs (mean per seed; ≈65 / ≈71 per seed).
* **Frozen readout actual form (verified, not assumed).** `decision_head` is `TypedDecisionHead` whose `proj` is a single `nn.Linear(212, 40)`; reshaped, `W ∈ R^{4×10×212}` and `b ∈ R^{4×10}`, so `z_{d,v}(S) = W[d,v]·S + b[d,v]` **exactly** (no nonlinearity precedes the logits).
* **Conventions reused verbatim.** `a* = argmax_{a≠y} z_a(S_prev)` (fixed reference, Experiment I); `m(S) = z_y(S) − z_{a*}(S)`; `ρ_margin = Δm_commit/Δm_cand` (defined when `Δm_cand > EPS_ABS = 1e-3`); `α*_full` exact (linear head); `λ_eff = <S_t−S_prev, C−S_prev>/‖C−S_prev‖²` and `offpath_dist = ‖S_t − [S_prev + λ_eff(C−S_prev)]‖` (Experiment H).
* **IMPORTANT DISTINCTION (explicit).** `ρ_margin ≥ α*_full` is **not** proof that the commit was sufficient. It means the delivered decision-margin *fraction* was at least as large as the *scalar-path* fraction required by the local candidate-direction analysis. Experiments H/I established the actual vector-gated state is generally off the scalar path, so L analyzes the **actual latent state and the actual frozen linear readout directly** (raw vectors and logits), not the scalar proxy.
* **Cross-experiment alignment.** L's re-collected test rows reproduce Experiment I's stored test rows and Experiment H's stored test rows **exactly** on every shared column (`max |Δ| = 0.0`; see §9), and match Experiment G's test rows exactly on all integer columns.

## 8. Experimental Procedure

1. Load a frozen checkpoint; verify its SHA-256 against Experiment G metadata.
2. One forward pass with components; capture `S_prev`, `C`, `S_commit`, the 212-dim gate, and (for every ADD transition) the readout decomposition: `z_y`, `z_a`, state margins `(w_y−w_a)·S`, bias margins `b_y−b_a`, bias-removed argmax decodes, pairwise decision-direction dot products, normalized hyperplane distances, `λ_eff`, `offpath_dist`.
3. Select the primary off-path population; store its raw `S_prev/C/S_commit` vectors plus the per-seed readout `(W, b)`.
4. Run the ten analyses (logit/margin decomposition; bias vs state; counterfactuals; pairwise geometry; projection error; hyperplane distance; perturbation stability; candidate-vs-commit geometry; class/value effects; E vs F), the category assignment, and the decision.
5. Verify checkpoint hashes before/after; re-run the whole pipeline a second time and require deterministic equality.

## 9. Results — Baseline Reproduction and Determinism

* **Readout decomposition is exact.** `head(x)` equals `einsum(W, x)+b` to float32 tolerance (tested); `z_y − z_a = (w_y−w_a)·S + (b_y−b_a)` exact; `m_commit = state_av_commit + bias_av` and `m_commit_alt = state_ac_commit + bias_ac` reconstructed from the stored columns; normalized hyperplane distances equal `margin/‖w_y−w_a‖`.
* **Projection conventions verified.** From the stored off-path vectors, `λ_eff` and `offpath_dist` recompute to the stored values (tolerances 1e-3).
* **Cross-experiment alignment.** L vs Experiment I: all 22 shared columns identical (`max |Δ| = 0.0`); L vs Experiment H: all shared columns identical (`max |Δ| = 0.0`); L vs Experiment G test rows: 0 exact mismatches on 17 integer columns.
* **Determinism.** The collector produced bit-identical arrays across two full runs (all array keys, `max |Δ| = 0.0`), and the analysis JSON is byte-identical across two runs (excluding the timestamp). Checkpoint SHA-256 values verified unchanged before and after every run.
* **No training path.** The collector uses only `load_state_dict` + `forward_with_states` under `torch.no_grad()`; no optimizer exists in the pipeline and no parameter changes (tested).

## 10. Results — Analysis 1–2: Margin and Bias/State Decomposition (E primary, medians over off-path failures)

| Quantity | E | F |
|---|---:|---:|
| `m_prev` (vs a_prev) | −12.20 | −11.56 |
| `m_cand` (vs a_prev) | **+5.74** | +4.60 |
| `m_commit` (vs a_prev) | **+1.96** | +1.94 |
| `m_commit` (vs committed competitor) | **−0.63** | −0.60 |
| `Δm_cand` | 18.06 | 16.35 |
| `Δm_commit` | 14.44 | 13.43 |
| `ρ_margin` | 0.817 | 0.838 |
| `α*_full` | 0.696 | 0.723 |
| `bias_av` | +0.007 | +0.014 |
| `state_av_commit` | **+1.96** | +1.91 |
| `bias_ac` | +0.013 | +0.006 |
| `state_ac_commit` | **−0.65** | −0.61 |
| bias share of winning-pair margin (median / p95) | **7.1% / 47.2%** | 6.7% / 54.8% |
| `state_ac_commit < 0` rate | 98.5% | 96.9% |

**Reading.** The candidate is genuinely helpful (candidate margin +5.7) and the commit moves in the intended direction (the committed state *clears* the intended competitor, ending at a **positive** margin against `a_prev`, +1.96). The failure is that a **different** competitor is, at that same state, ahead of the true class (`−0.63`). The **bias terms are tiny** (`+0.007`, `+0.013`), while the **state-projection term is what places the state** — and it is negative against the winner. The state alone (without bias) already puts the true class behind the winner in ~98% of off-path failures.

## 11. Results — Analysis 3: Readout-Calibration Counterfactuals (E primary)

| Counterfactual | E | F |
|---|---:|---:|
| actual off-path committed accuracy | 0.000 | 0.000 |
| **bias-removed decode correct rate** | **0.012** | 0.031 |
| bias-removed decode changes the decision (off-path) | 0.028 | 0.043 |
| bias-removed decode changes the decision (all primary) | 0.007 | 0.008 |

Removing the fixed bias (analysis only, latent state unchanged) flips only ~1–3% of off-path failures to correct. Even accounting for the bias changes the committed decision on fewer than 5% of off-path failures. This is direct evidence against readout/calibration as the dominant cause.

## 12. Results — Analysis 4–6: Pairwise Geometry, Projection, and Hyperplane Distance (E primary)

| Quantity | E | F |
|---|---:|---:|
| `⟨w_y−w_{a_prev}, S_commit−S_prev⟩` (intended movement) | +14.44 | +13.43 |
| `⟨w_y−w_{a_prev}, C−S_prev⟩` (candidate-available) | +18.06 | +16.35 |
| `⟨w_y−w_{a_commit}, S_commit−S_prev⟩` | **−0.19** | +0.00 |
| moved-intended rate | **1.000** | 1.000 |
| competitor-changed rate (off-path) | **1.000** | 1.000 |
| competitor-changed rate (primary successes) | 0.653 | 0.657 |
| hyperplane dist vs a_prev: `S_prev` → `C` → `S_commit` | −3.13 → +1.45 → **+0.52** | −2.86 → +1.12 → +0.47 |
| hyperplane dist vs committed competitor at `S_commit` | **−0.168** | −0.154 |
| cleared-intended rate (off-path / non-off-path failure / success) | **1.000 / 0.178 / 1.000** | 1.000 / 0.191 / 1.000 |
| `λ_eff` (off-path / success) | 0.323 / 0.324 | 0.329 / 0.342 |
| `offpath_dist` (off-path / success) | 6.57 / 6.36 | 6.63 / 6.62 |
| `offpath_dist / ‖C−S_prev‖` (off-path / success) | 0.415 / 0.420 | 0.416 / 0.428 |

**Reading.** Off-path failures move ~80% of the candidate's intended-direction movement; they **cross** the `S_prev` boundary (normalized distance becomes positive) and clear the intended competitor in **100%** of cases (versus **17.8%** for non-off-path failures). But at that committed state the winning competitor's boundary is violated (normalized distance `−0.168`): the state has entered a **different** basin. **Importantly, `off-path distance` and `λ_eff` do not distinguish off-path failures from successes** (6.57 vs 6.36; 0.415 vs 0.420) — so "off-path distance" alone is not the signature, and is explicitly not over-read.

## 13. Results — Analysis 7–9: Perturbation Stability, Candidate-vs-Commit, Class/Value Effects (E primary)

* **Local perturbation stability (analysis only, deterministic isotropic directions, `ε·‖C−S_prev‖`, 32 directions):** decision stability (same decode) = **0.977 / 0.891 / 0.809** at ε = 0.01 / 0.05 / 0.10; perturbed-decode-correct rate = **0.017 / 0.079 / 0.131**. The committed state is **locally stable and deep inside the wrong basin** — small perturbations do not reach the correct region.
* **Candidate vs commit (off-path failures vs candidate-correct successes):** candidate margin 5.74 vs 8.52; commit margin (vs a_prev) +1.96 vs +6.61; margin loss 3.18 vs 1.84; non-off-path failures have commit margin **−0.61** (they do **not** clear the intended competitor). Off-path failures thus have a **weaker candidate margin** than successes *and* wrong final placement — the two are not mutually exclusive.
* **Class/value effects (descriptive):** wrap rate **54.8%** (E) / 55.8% (F) among off-path failures; the wrong decode is a **third value** in **100%** (E) / 99.4% (F) of cases (never the old value); off-path failures are spread across destination registers and values, so the placement is not confined to a single register/value geometry.

## 14. Results — Analysis 10: E versus F

| Signature (mean ± SD over 5 seeds) | E | F | paired p |
|---|---:|---:|---:|
| off-path rate | 0.0205 ± 0.0019 | 0.0224 ± 0.0027 | — |
| r_bias | 0.013 ± 0.022 | 0.030 ± 0.031 | 0.40 |
| r_place | 0.987 ± 0.022 | 0.970 ± 0.031 | 0.40 |
| competitor changed | 1.000 ± 0.000 | 1.000 ± 0.000 | — |
| cleared intended | 1.000 ± 0.000 | 1.000 ± 0.000 | — |
| third-basin decode | 1.000 ± 0.000 | 0.994 ± 0.008 | 0.18 |
| bias share (median) | 0.0745 ± 0.020 | 0.0745 ± 0.028 | 1.00 |
| off-path ratio | 0.414 ± 0.006 | 0.416 ± 0.005 | 0.72 |
| hyperplane dist vs winner | −0.164 ± 0.030 | −0.155 ± 0.040 | 0.72 |

**Reading.** E and F are statistically indistinguishable on every off-path signature. The localization is not a property of the training objective; both objectives produce the same off-path mechanism.

## 15. Results — Statistical Analysis and Sanity Checks

* **Independent units:** the five frozen E checkpoints. Transition-level tables (n = 323 for E) are **descriptive**; seed-level summaries (mean ± SD over 5 seeds) and the paired E-vs-F comparison carry the inference. No thousand-transition significance claim is made.
* **E seed-level** (per-seed values in the results JSON): `r_place` = 0.960–1.000 in every seed; `r_bias` = 0.000–0.031; competitor-changed = 1.000; cleared-intended = 1.000; third-basin = 1.000 in every seed. The verdict is seed-invariant.
* **Sanity checks (all pass):** (1) checkpoint SHA-256 unchanged before/after; (2) no optimizer/training path invoked; (3) no gradients (tests); (4) frozen readout weights unchanged (hash + no-parameter-change tests); (5) G/H/I row alignment exact; (6) linear `z = W S + b` decomposition exact; (7) pairwise-margin reconstruction exact; (8) projection formulas agree with direct computation; (9) deterministic rerun (collector bit-identical; analysis JSON identical excluding timestamp); (10) no historical artifacts modified.

## 16. Interpretation — Observation, Interpretation, Causal Status, Unresolved

**Observation (measurement).** On the exact off-path population, the committed state (i) delivers ~80% of the candidate's intended-direction movement and clears the strongest competitor that mattered at `S_prev` in 100% of cases, yet (ii) the committed argmax is a *different* competitor and decodes to a *third* value in 100% of cases, (iii) the fixed bias is nearly never decisive (bias-removed flip 1–3%; bias share of the winning margin median 7%), and (iv) small perturbations around `S_commit` almost never reach the correct basin.

**Supported interpretation.** The off-path residual is **readout-adjacent but not readout-caused**: the frozen linear readout is not mis-scoring an adequately-placed state (bias is not the cause), and the state is not merely stopping short of a fixed boundary (it clears the intended one); rather, the committed latent state is **placed in a different competitor's region** — a latent-state placement phenomenon. This is consistent with Experiment H (the commit is off the scalar path) and with Experiments J/K (commit-side reweighting does not help), and it explains why: the residual is a *placement* property of the frozen state trajectory, which commit-side magnitudes/shapes cannot reach.

**Causal status.** Measurement only — **no causal claim**. L localizes the off-path component to latent placement by a reproducible, converging set of properties; it does not show that the placement *causes* the sequence-level failures, nor that any representation change would fix them.

**Unresolved hypothesis.** L does not explain *why* the frozen trajectory places the state in the third basin — whether it is a systematic arithmetic/modular geometry (off-path failures are ~55% wrap; the third value may be a wrap/adjacent-value region), a general off-axis collapse, or a readout-tolerance interaction with the (weaker) candidate margin. It also does not determine whether the *candidate* margin weakness partly drives the placement.

## 17. What the Experiment DOES Establish

* A clean, reproducible, frozen, measurement-only decomposition of the off-path failures using the **actual** linear readout (`z = W S + b` exact).
* That the off-path population is characterized by **clearing the intended competitor (100%) while landing on a different competitor (100%)** — a latent-placement signature, not a boundary shortfall (0%) and not a bias/calibration signature (1–3%).
* That the **fixed bias is nearly never decisive** (removing it flips ~1–3% to correct), which directly rules out the simple "the readout's offsets are wrong" explanation.
* That `off-path distance` alone does **not** distinguish off-path failures from successes (6.57 vs 6.36) — a negative result that guards against over-reading the off-path geometry.
* That the committed state is **locally stable inside the wrong basin** (perturbations rarely reach correct), consistent with placement rather than boundary proximity.
* That the localization is **robust across E and F** (statistically indistinguishable signatures), and aligned exactly with Experiments G/H/I.

## 18. What the Experiment DOES NOT Establish

* It does **not** show the readout or the representation "causes" the failures (no causality); it is a localization.
* It does not propose, test, or select any fix (no readout adjustment, no bias change, no weights, no gate/commit change, no retraining, no repair rule).
* It does **not** show the third basin is a wrap-arithmetic region — that is an unresolved hypothesis and the subject of the next experiment.
* It does **not** separate "weak candidate margin" from "wrong placement" as independent causes (both appear; they are not mutually exclusive).
* It is restricted to the ADD candidate→commit off-path population on the register-tracking benchmark under the E/F objectives; it does not generalize to Phase-2 machinery, other tasks, or sequence-level capability.

## 19. Decision

**B — primarily latent-state placement (E primary; F control agrees).** The off-path candidate→commit residual is a **latent-state placement** phenomenon: the committed state delivers enough intended-direction movement to clear the boundary that was binding at `S_prev`, but enters a different competitor's region; the fixed readout bias is nearly never decisive, `off-path distance` does not distinguish these failures, and the state is locally stable inside the wrong basin. Per the pre-registered decision rule, `r_place = 0.988 ≥ 0.60` and `r_bias = 0.012 < 0.30` → **B**. This closes the readout-bias hypothesis for the off-path component and redirects the next step to **representation/state geometry** — as required, not to a readout fix.

## Final Decision (required answers)

1. **Fraction primarily boundary-shortfall:** 0.0% (deterministic category; measured by `m(S_commit; a_prev) < 0`, which is never the case for the off-path set).
2. **Fraction showing readout/calibration signatures:** 1.24% (E) / 3.12% (F) (bias-removed decode flips to correct).
3. **Fraction showing latent-state-placement signatures:** 98.76% (E) / 96.88% (F) (cleared intended competitor, different committed competitor).
4. **Mixed/unresolved:** 0.0%.
5. **Does the fixed bias materially matter?** No — bias share of the winning margin median 7.1% (p95 47.2%); removing bias changes the committed decision on 2.8% of off-path failures and flips only ~1–3% to correct.
6. **Does the committed state move toward a competing basin?** Yes — competitor change 100% (vs 65% of successes), third-value decode 100%, cleared the intended competitor 100%, and sits below the winning competitor's boundary (normalized −0.168); perturbations rarely reach the correct basin.
7. **Consistent across E and F?** Yes — every signature is statistically indistinguishable (paired p ≥ 0.40); no objective-specific mechanism.
8. **Dominant explanation:** **B — latent-state placement.**
9. **Exactly one next scientific experiment:** **Experiment M — representation geometry of the third basin (measurement-only, frozen checkpoints).**

## Next Experiment (exactly one)

**Experiment M — representation geometry of the third basin (measurement-only, frozen checkpoints; no training, no intervention).** L shows off-path failures clear the intended competitor and land in a *different* competitor's region, without a bias cause. M must characterize *what* that third basin is, so any later representation-side change is grounded in a measured mechanism rather than a search. On the same frozen E/F checkpoints and the same off-path population, M should: (i) identify, from the full 40-way readout at `S_commit`, the exact committed (wrong) competitor register/value and its relation to the task arithmetic (old value, source operand, `(old + operand) mod 10`, wrap flag); (ii) test whether the third (winning) basin is systematically the **wrap/modular** region (~55% of off-path failures are wrap) or a structurally adjacent value; (iii) measure the direction of the committed state's off-axis displacement relative to the candidate direction and test whether it is a **consistent** off-axis direction (a systematic misplacement) or unsystematic; (iv) report the distance from `S_commit` to the candidate-direction decodable region and to the winning-basin region. It is measurement only; it must not implement a fix, and it must not begin Phase 2. If M shows a systematic modular geometry, the following step would be a tightly controlled representation-side intervention; if M shows unsystematic collapse, the residual is a general representation bandwidth/placement limit to be documented rather than repaired.

## Reproduction

```bash
# 1. Collect readout decompositions + off-path vectors from frozen checkpoints (~10 s)
python experiments/run_experiment_l.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_l_readout.npz

# 2. Analysis + 12 figures (~4 s)
python experiments/analyze_experiment_l.py \
    --readout experiments/experiment_l_readout.npz \
    --output experiments/experiment_l_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_l

# 3. Tests
pytest tests/test_experiment_l.py -v      # 17 tests
pytest tests/                             # full suite
```

Determinism: the collector was run twice (arrays bit-identical, `max |Δ| = 0.0`) and the analysis twice (JSON identical excluding the timestamp). Checkpoint SHA-256 verified before/after.

## Artifact List

* `experiments/run_experiment_l.py` — collector: exact linear-readout decomposition, off-path filtering, vector capture.
* `experiments/analyze_experiment_l.py` — analyses 1–10, deterministic categories, pre-registered decision, E-vs-F, G/H/I alignment, 12 figures.
* `tests/test_experiment_l.py` — 17 tests (linear/pairwise/bias-state decomposition, hyperplane distance, projection/residual, competitor identification, off-path filtering, category + decision logic, collector determinism/no-grad/no-param-change, artifact reproducibility, checkpoint hashes, G/I/H alignment, end-to-end analysis + figures, invalid inputs).
* `experiments/experiment_l_readout.npz` — per-ADD-transition scalars (test split) + off-path vectors + per-seed `(W, b)`.
* `experiments/experiment_l_readout_meta.json` — columns, off-path definition, checkpoint hashes, G verification.
* `experiments/experiment_l_results.json` — analyses 1–10, categories, decision, E-vs-F paired, alignment.
* `docs/reports/figures/experiment_l_*.png` — 12 figures: `decomposition`, `margin_progression`, `bias_vs_state`, `offpath_dist`, `hyperplane_dist`, `competitor_switch`, `margin_dist`, `projection_residual`, `categories`, `value_breakdown`, `E_vs_F`, `perturbation`.
* Checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against `experiments/experiment_g_transitions_meta.json`).
* Historical references (read only, unchanged): `experiments/experiment_i_coordinate.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_g_transitions.npz`.

## Scientific Guardrails

This is a **measurement-only** localization; it is not causal proof and does not implement a fix. `ρ_margin ≥ α*_full` is not treated as proof that the commit was sufficient — it is the established subset definition, and L analyzes the actual latent state and the actual frozen linear readout directly. Bias-removal and perturbation counterfactuals are explicitly labeled analysis-only one-step computations on frozen states; they are never used as predictions or in a rollout. `off-path distance` is explicitly reported as non-distinguishing rather than over-read. The categories (boundary shortfall / readout calibration / latent misplacement / mixed) are analytical labels, not causal labels. The result is stated as a localization ("consistent with / placed in a different basin"), never as "the readout causes the failure" or "the representation causes the failure." No novelty, AGI, or generalization claim is made. Negative results are preserved; Experiments G/H/I/J/K reports and artifacts are untouched, and L's alignment against G/H/I is exact.
