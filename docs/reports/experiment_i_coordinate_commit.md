# RE:Reason — Experiment I: Coordinate-Wise Decision-Margin Commit Analysis

**Date:** October 6, 2026
**Status:** Completed — Measurement-only (no training, no model changes, no rollouts; frozen checkpoints)
**Frozen models:** `E_candidate` (official Phase-1 objective) and `F_combined` (control), 5 seeds each, 226,358 parameters, SHA-256-pinned
**Data:** `experiments/experiment_i_coordinate.npz`, `experiments/experiment_i_coordinate_meta.json`, `experiments/experiment_i_results.json`
**Runners:** `experiments/run_experiment_i.py`, `experiments/analyze_experiment_i.py`, tests `tests/test_experiment_i.py`
**Checkpoints:** `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against Experiment G metadata)

---

## 1. Executive Answer

**When the ADD candidate is already correct, the commit fails not because the gate delivers nothing, but because it delivers ~78–80% of the candidate's available decision-margin movement (versus ~90% on successes), preferentially suppressing the coordinates that contribute positively to the correct-vs-competitor decision and leaving the state just short — and, additionally, because ~44–48% of failures delivered at least the scalar-equivalent transfer the linear path would need, yet the *element-wise* commit landed in a region where a different competitor wins. Both signatures are present; classification: C (both).**

1. **The residual is small, localized, and reproduced exactly.** Candidate-correct general-stratum test ADDs: E 5,042 transitions (4,310 success / 732 failure); F 5,032 (4,292 / 740). Every one of these transitions has a positive candidate decision margin (`m_cand > 0`, by definition of argmax) and positive available movement (`Δm_candidate > 0`). No-op and same-register adds are essentially gap-free (P(fail | cand ok) = 0.4% / 0.4% E, 0.3% / 0.6% F); the gap is confined to ordinary two-register adds (14.5% / 14.7%) and wrap (17.0% / 17.7%).
2. **The decision margin is defined against the strongest competing class.** The frozen readout is a single `nn.Linear(212→40)`, so `z_{d,v}(S) = w_{d,v}·S + b_{d,v}` exactly, and with `a* = argmax_{a≠y} z_a(S_prev)` the margin `m(S) = z_y(S) − z_a*(S)` is exactly affine in `S`. Consequently the coordinate decomposition is **exact** (not a Jacobian approximation): `Σ_i (w_y−w_a*)_i f_i (C_i−S_prev_i) = Δm_commit` (max residual ~2e-6 in float32).
3. **Under-movement is systematically present.** On failures the delivered margin fraction `ρ_margin = Δm_commit/Δm_candidate` is **0.780 (E) / 0.798 (F)** versus **0.897 / 0.903** on successes (seed-paired fail−succ: −0.117, dz = −6.34, 5/5 seeds; F: −0.102, dz = −5.96, 5/5). The contribution-weighted gate on the *positive* (decision-helping) coordinates is lower on failures (0.617 / 0.623 vs 0.702 / 0.712; dz = −7.6 / −11.1, 5/5), while on *negative* (decision-harming) coordinates it is slightly higher on failures (0.281 / 0.305 vs 0.268 / 0.271). The gate therefore does not merely under-deliver globally — it relatively favors harmful coordinate movement at the expense of helpful movement.
4. **But the candidate's own margin is also much weaker on failures.** Successes have median candidate margin 8.52 (E) / 8.35 (F); failures 4.38 / 3.86 (seed-paired dz = −9.5 / −19.1). Failure rate by candidate-margin decile spans 53% → 0.4% (E). The candidate is only *just* on the correct side in failure cases.
5. **And a substantial share of failures is not explained by insufficient delivered amount.** `α*_full` (the exact scalar transfer needed along `S_prev→C` to clear *all* competitors) has median 0.774 / 0.782 on failures, while `ρ_margin` delivered ≥ `α*_full` for **44.1% (E) / 47.7% (F)** of failures. Those commits delivered enough movement in scalar-equivalent terms yet still failed — the element-wise commit lands off the scalar path (as Experiment H established), and ~66–70% of failures end nearer a *different* strongest competitor than the one at `S_prev`.
6. **The decision-boundary statistic agrees with Experiment H.** `α*_full` computed here from the exact linear readout and Experiment H's independently measured `λ*` (21-point sweep) agree almost perfectly on the same primary transitions (Spearman 0.997 / 0.994; mean |Δ| = 0.025; 100% within 0.05). `α*_margin`/`α*_full` are **empirical local decision-boundary statistics, not model thresholds**.
7. **E and F are indistinguishable on this residual.** On primary failures every paired seed-level comparison is statistically null (ρ_margin p = 0.14, Δm_commit p = 0.72, gw_pos p = 0.36, α* p = 0.27); overall primary failure rate E 0.145 vs F 0.147 (p = 0.84). F's previously reported committed-state calibration advantage (`combined_objective_confirmation.md`) does not show up in the ADD decision-margin delivery mechanism, consistent with Experiments G and H.
8. **Decision (§15): C. Both.** Under-movement (gate delivery + weak candidate margin) is the larger, more consistent effect; readout/placement precision is additionally required to explain the ≈45% of failures with sufficient scalar-equivalent delivery. **No causal claim** is made. The single next experiment (§14/§16) is a **controlled inference-time commit-rule intervention** on the frozen checkpoints.

---

## 2. Existing Baseline (what Experiments G and H left open)

* **Experiment G** (`experiment_g_add_commit_gap.md`) localized the residual to the general ADD stratum (~5.5% of candidate-correct adds) and reported mixed evidence for gate dilution vs blend/readout (classification C), flagging the scalar mean gate as a proxy only.
* **Experiment H** (`experiment_h_lambda_interpolation.md`) showed the scalar old→candidate path is monotone and decodable only late (median λ* = 0.80 on failures), that the element-wise vector-gate replay reproduces the commit bit-exactly while its mean-gate scalarization collapses (0.855→0.182), and that the actual committed state lies off the scalar path (mean off-path distance 6.3–6.5 on a segment ≈15). It explicitly specified Experiment I: **decompose the actual element-wise gate at the decision level**, using the strongest competing class rather than the true logit alone.
* Experiment I is the response to that specification. It is **measurement only** and does not implement any fix.

## 3. Research Question

> When the candidate state is correct but the committed state is wrong, does the element-wise gate fail to deliver enough movement on coordinates relevant to the decoder's actual decision boundary (**commit/gate under-movement**), or does the frozen readout require unusually precise latent alignment even when useful decision-direction movement has been delivered (**readout/representation precision/alignment**)?

## 4. Hypotheses (declared before interpretation)

For candidate-correct general ADD transitions:

* **I-U (under-movement).** Failures deliver a smaller fraction of the candidate's decision-margin movement, a smaller absolute delivered margin, and a lower contribution-weighted gate on the decision-helping coordinates; the pattern is seed-consistent.
* **I-R (readout/precision).** A non-trivial share of failures delivers at least the scalar-equivalent transfer needed (`ρ_margin ≥ α*_full`) yet still fails, and/or the candidate direction is less aligned with the decision direction on failures — implicating placement/off-path geometry rather than amount delivered.
* **I-C (both)** if both signatures are present; **I-D (inconclusive)** if neither is.

## 5. What Is Frozen

Frozen and untouched: recurrent core, observation encoder, gate network, candidate MLP, convex commit, terminal readout, optimizer protocol, and the E/F checkpoints; the task generator; the benchmark and unified split; all Phase-2 machinery (branching, verifier, recursion, transactional commit); NLP/NLG. No training, no optimizer, no weight modification; all inference under `torch.no_grad()`. Checkpoint SHA-256 values are re-verified before and after the run.

## 6. What Changed

Only new measurement code and artifacts were added (`run_experiment_i.py`, `analyze_experiment_i.py`, `tests/test_experiment_i.py`, the coordinate artifact, results JSON, figures, and this report). No model, data, protocol, or previous result was modified. Experiment G and H reports are untouched.

## 7. Dataset and Protocol

* **Data.** For each of E and F × seeds {42..46}: one `forward_with_states(..., return_components=True)` pass over (a) the canonical strict test split (1,200 sequences × 10 steps, stream `90042+seed`, never trained on) and (b) the five existing controlled ADD builders (isolated / ADD→SET / 4× repeated / no-wrap / wrap; 400 sequences each).
* **Scale.** 31,763 ADD transitions per model (15,763 test + 16,000 controlled), identical ordering to Experiments G and H; the runner verifies row-level equality against the Experiment G dataset on 17 integer columns (0 mismatches) and re-verifies the Experiment H alignment (`f_mean` max diff 0.0).
* **Primary analysis set.** candidate-correct ∧ general-stratum ADD ∧ strict test split, where general = (destination ≠ source register) ∧ (source operand > 0) — the Experiment G/H definition. Secondary controls (no-op, same-register, candidate-wrong, wrap/no-wrap, controlled builders) are reported separately and never pooled into the primary result.

### Decision-measure definition (documented, primary)

For a destination register `d` and true value `y`, `z(S) = head(S)` reshaped `(R,V)`:

```
a*        = argmax_{a != y} z_a(S_(t-1))          (fixed local decision reference — PRIMARY)
m(S)      = z_y(S) − z_(a*)(S)                     (exactly affine in S)
m_prev    = m(S_(t-1)),  m_cand = m(C_t),  m_commit = m(S_t)
Δm_candidate = m_cand − m_prev ,  Δm_commit = m_commit − m_prev
rho_margin   = Δm_commit / Δm_candidate            (defined only when Δm_candidate > EPS_ABS = 1e-3)
margin_shortfall = m_cand − m_commit
```

* The strongest competitor is fixed at `S_(t-1)` as the local reference. **Sensitivity** to recomputing `a*` at `S_t` is reported separately (`a_star_commit`, `m_commit_altstar`) — the primary definition is never silently changed.
* The true-logit movement `⟨w_y, S − S_prev⟩` was retained as a **secondary** diagnostic and is not the primary criterion.
* `alpha*_margin` = minimal `α ≥ 0` with `m(S_prev + α(C−S_prev)) ≥ 0` against the fixed `a*`; `alpha*_full` = minimal `α` with the *full* argmax equal to `y` (all competitors cleared). Both are exact closed forms for the linear head and are **empirical local decision-boundary statistics, not model thresholds**.
* The commit is `S_t = (1−f_t)⊙S_prev + f_t⊙C_t`, i.e. `S_t − S_prev = f_t⊙(C−S_prev)`; hence
  `q_i = (w_y − w_a*)_i f_i (C_i − S_prev_i)` and `Σ_i q_i = Δm_commit` **exactly**. `q_cand_i = (w_y − w_a*)_i (C_i − S_prev_i)` is the candidate-available contribution, with `Σ_i q_cand_i = Δm_candidate`.

### Statistical discipline

Transitions within a sequence are dependent, so transition-level tables and plots are **descriptive**. Independent inference is **seed-level** (n = 5): paired failure-vs-success within model and paired E-vs-F. Effect sizes use the conventional Cohen's dz bands (negligible < 0.2, small < 0.5, medium < 0.8, large ≥ 0.8), with sign consistency reported as k/5. These bands are declared conventions, not data-tuned cutoffs.

## 8. Experimental Procedure

1. Re-run inference on the frozen checkpoints and record, per ADD transition: `S_prev`, `C_t`, `S_t`, the 212-dim gate `f_t`; `m_prev/m_cand/m_commit`; `Δm_candidate/Δm_commit`; `ρ_margin`; sign-crossing flags; `α*_margin`, `α*_full`; gate/contribution-weighted gate statistics; geometry; decodes; task metadata.
2. Store the full per-coordinate arrays (`f_i`, `move_i = f_i(C_i−S_prev_i)`, `q_cand_i`, `|C_i−S_prev_i|`) for the **primary set only** (5,042 / 5,032 rows).
3. Merge Experiment H scalars (`λ_eff`, `λ*`) by verified row alignment; run the ten analyses, sanity checks, and figures.

## 9. Analysis 1 — Decision-Margin Movement

Primary set (pooled seeds), medians:

| Quantity | E success | E failure | F success | F failure |
|---|---:|---:|---:|---:|
| n | 4,310 | 732 | 4,292 | 740 |
| m_prev | −9.74 | −12.41 | −10.62 | −12.14 |
| m_cand | 8.52 | 4.38 | 8.35 | 3.86 |
| m_commit | 6.61 | 0.30 | 6.62 | 0.53 |
| Δm_candidate | 17.66 | 16.61 | 19.21 | 16.02 |
| Δm_commit | 15.70 | 12.83 | 17.35 | 12.69 |
| margin_shortfall | 1.84 | 3.59 | 1.83 | 3.17 |
| ρ_margin | 0.897 | 0.780 | 0.903 | 0.798 |

* 100% of candidate-correct transitions have `m_cand > 0` and `Δm_candidate > 0`; `ρ_margin` is defined for 100% of them.
* Failures lose ~3.2–3.6 margin units at commitment (≈2× the success shortfall).
* Only 54.1% (E) / 57.7% (F) of failures have `m_commit > 0` against the fixed `a*`; the rest still have the wrong side of the fixed reference. Almost none (0.3% / 0.1%) have `Δm_commit ≤ 0`.
* **Sensitivity:** the strongest competitor changes at the commit in 65.4% / 66.4% of all primary transitions (66.3% / 70.1% of failures); among failures the sign of `m_commit` flips when `a*` is recomputed at `S_t` in 54.1% / 57.7% of cases, and `m_commit_altstar` has median −0.894 / −0.835 (negative by construction: the committed argmax is not `y`). This is the first indication that the fixed-reference margin understates what the actual decision does.

## 10. Analysis 2 — Decision-Boundary Movement (`α*`)

* Solution fraction 100% (every candidate-correct transition admits an `α*` along the line, since `α = 1` is the candidate).
* Failure `α*_margin` median 0.738 (E) / 0.753 (F); p25 0.671 / 0.700, p75 0.811 / 0.817, p95 0.929 / 0.927.
* Failure `α*_full` median 0.774 / 0.782 (needs to clear *all* competitors — slightly later than the fixed `a*`).
* Primary `α*` by wrap: wrap p50 0.687 / 0.710 vs no-wrap p50 0.511 / 0.542 — wrap transitions need later (more) transfer, consistent with wrap being the hardest class.
* **Agreement with Experiment H:** `α*_full` vs H's independently measured `λ*` on the same rows — Spearman 0.997 / 0.994, mean |Δ| = 0.025, 100% within 0.05. Two structurally different constructions (exact closed form vs 21-point sweep) agree, validating both.
* The comparison `ρ_margin` vs `α*` is **descriptive only**: the actual commit is element-wise and off the scalar line (H), so `ρ_margin ≥ α*_full` does not imply a correct commit. This is made explicit in the results rather than treated as a threshold test.

## 11. Analysis 3 — Actual Vector-Gate Delivery

* The head is demonstrably linear (single `nn.Linear`), so the decomposition is **exact**: `max |Σ_i q_i − Δm_commit| ≈ 2e-6` (float32) and `max |Σ_i q_cand_i − Δm_candidate| ≈ 2e-6` over all 31,763 transitions.
* `(1−f)⊙S_prev + f⊙C` reproduces the recorded `S_t` bit-exactly (max |Δ| = 0.0).
* Mean gate on the primary set ≈ 0.33–0.37.
* No Jacobian approximation is needed; the identity is exact because no nonlinearity precedes the logits.

## 12. Analysis 4 — Coordinate Attribution (success vs failure)

Per-coordinate arrays, primary set (E / F):

| Quantity | success | failure |
|---|---:|---:|
| delivered fraction on positive `q_cand` coords | 0.705 / 0.716 | 0.625 / 0.631 |
| delivered fraction on negative `q_cand` coords | 0.270 / 0.274 | 0.288 / 0.311 |
| aggregate decision-delivered fraction | 0.879 / 0.890 | — |

* Failures under-move the **positive** decision-contributing coordinates (gate 0.62–0.63) relative to successes (0.70–0.72), while their movement on negative coordinates is not reduced to the same degree.
* The top decision-contributing coordinates (by |available contribution|) are reported in the JSON; they are described as **readout-relevant / decision-contributing coordinates**, never as register coordinates (no such mapping was established).
* The effect is a relative one: failures lose ≈12% of the positive-coordinate delivery and gain ≈5–13% on the negative side relative to successes.

## 13. Analysis 5 — Available vs Delivered Decision Movement (seed-level)

Seed-paired failure − success (mean diff, dz, sign k/5):

| Metric | E | F |
|---|---|---|
| ρ_margin (median) | −0.117, dz −6.34, 5/5 | −0.102, dz −5.96, 5/5 |
| Δm_commit (median) | −2.76, dz −3.48, 5/5 | −4.66, dz −11.83, 5/5 |
| Δm_candidate (median) | −1.13, dz −1.79, 5/5 | −3.04, dz −5.33, 5/5 |
| m_cand (median) | −4.18, dz −9.52, 5/5 | −4.45, dz −19.07, 5/5 |
| gw_pos (mean) | −0.086, dz −7.61, 5/5 | −0.088, dz −11.13, 5/5 |
| align_ratio (median) | −0.100, dz −3.98, 5/5 | −0.219, dz −6.00, 5/5 |
| α*_margin (median) | +0.202, dz +10.78, 0/5 | +0.190, dz +12.62, 0/5 |

Failures deliver less absolute margin, less of the available margin, less positive-coordinate gate, and start from a weaker candidate margin, in **every seed**. Since candidate-correct force `m_cand>0`, `Δm_candidate>0`, and `ρ_margin` defined, no `Δm_candidate ≈ 0` masking is triggered on the primary set; the documented rule applies to secondary categories (e.g., no-op adds, where `Δm_candidate` is tiny, or candidate-wrong transitions where it is negative — those are excluded from the ratio by the `EPS_ABS` rule and reported by count).

Failure rate by decile (E): m_cand 0.533 → 0.004; ρ_margin 0.610 → 0.014; gw_pos 0.574 → 0.004; align_ratio 0.253 → 0.069. All are monotone; none is a sharp threshold.

## 14. Analysis 6 — Minimum Movement to Flip the Decision

`α*_margin`/`α*_full` distributions are in §10 and Figure 4/5. By subset (E / F medians of `α*_margin`): success 0.539 / 0.561; failure 0.738 / 0.753; wrap 0.687 / 0.710; no-wrap 0.511 / 0.542. Relations on failures: `α*` vs `‖C−S_prev‖` and vs mean gate and vs delivered margin are all reported in `analysis_6_alpha_star` (descriptive). `α*` is an analysis metric only and is never used as a threshold.

## 15. Analysis 7 — Coordinate-Wise Gate Effect

Contribution-weighted gate means (E / F):

| Weighting | success | failure |
|---|---:|---:|
| gw_all (all \|q_cand\|) | 0.602 / 0.613 | 0.539 / 0.544 |
| gw_pos (positive contributions) | 0.702 / 0.712 | 0.617 / 0.623 |
| gw_neg (negative contributions) | 0.268 / 0.271 | 0.281 / 0.305 |
| gw_mag (weight \|C−S_prev\|) | 0.341 / 0.354 | 0.306 / 0.317 |

Exact formulas: `gw_all = Σ|q_cand_i| f_i / Σ|q_cand_i|`; `gw_pos = Σ_{q_cand>0} q_cand_i f_i / Σ_{q_cand>0} q_cand_i`; `gw_neg = Σ_{q_cand<0} (−q_cand_i) f_i / Σ_{q_cand<0} (−q_cand_i)`; `gw_mag = Σ|C_i−S_prev_i| f_i / Σ|C_i−S_prev_i|`. Where a side has no contribution the statistic is undefined (NaN) by the `EPS_ABS` rule.

The gate is not uniformly closed in failures: it delivers **disproportionately more on harmful (negative) coordinates and less on helpful (positive) coordinates** than in successes. This is coordinate-selective under-movement, not a global dilution.

## 16. Analysis 8 — Candidate/Commit Four-Way Classification

Test-split ADDs (fractions; E / F): A (cand✓ commit✓) 0.837 / 0.836; B (cand✓ commit✗) 0.0485 / 0.0488; C (cand✗ commit✓) 0.0146 / 0.0184; D (cand✗ commit✗) 0.0999 / 0.0973. B is the transition-level commitment residual.

A-vs-B central comparison (seed-level, paired; E / F dz): Δm_commit +3.48 / +11.83 (failures lower), Δm_candidate +1.79 / +5.33 (failures lower), align_ratio +3.98 / +6.00 (failures lower), gw_pos +7.61 / +11.13 (failures lower). Candidate margin is the dominant single separator (dz −9.5 / −19.1). All effects consistent across 5/5 seeds.

## 17. Analysis 9 — E vs F

Paired seed-level, primary failures (E vs F): ρ_margin 0.781±0.019 vs 0.801±0.021 (diff −0.020, p = 0.14); Δm_commit 12.94 vs 12.74 (p = 0.72); gw_pos 0.617 vs 0.625 (p = 0.36); gw_mag 0.306 vs 0.318 (p = 0.06); align_ratio 1.055 vs 1.010 (p = 0.13); α*_margin 0.739 vs 0.751 (p = 0.27); overall primary failure rate E 0.145 vs F 0.147 (p = 0.84). On successes F has marginally lower Δm_commit/Δm_candidate/align_ratio (p ≈ 0.001–0.03) but the same `ρ_margin`. **F's better committed-state calibration does not change the ADD decision-margin delivery mechanism** — the residual is identical, consistent with G and H.

## 18. Analysis 10 — Evidence Classification

Conservative, pre-declared signature tests (see `analysis_10_evidence` for the full numbers):

* **Under-movement (U):** 3/3 sub-signatures hold for both models (lower delivered fraction, lower absolute delivered margin, lower positive-weighted gate), each with 5/5 seed sign-consistency and large effect sizes. **U supported.**
* **Readout/precision (R):**
  * R1 — share of failures whose delivered fraction ≥ the exact scalar-equivalent transfer `α*_full`: **0.441 (E) / 0.477 (F)**. These commits delivered sufficient *amount* in scalar-equivalent terms yet failed, implicating off-path placement rather than amount. **Supported** (far above any permissive non-triviality bar).
  * R2 — candidate decision-direction alignment lower on failures (align_ratio dz −3.98 / −6.00, 5/5). **Supported.**
  * Additionally, the strongest competitor changes at the commit in 66–70% of failures, and the fixed-reference margin is positive in ~54–58% of failures — the element-wise commit reaches the reference boundary but not the correct basin.
* **Verdict (both models): `C_both`.** Under-movement is the larger, more consistent effect; a roughly equal-sized additional component is not explained by delivered amount and requires the readout/placement (off-path) account.

Conventions used (declared in `rule_note`): U supported if ≥2 of the three sub-signatures hold with ≥4/5 sign consistency; R supported if the sufficient-scalar-transfer share is ≥ 0.10 (deliberately permissive; the observed ≈0.45 makes the verdict insensitive to any bar in ~[0.1, 0.4]) or if alignment is lower on failures with ≥4/5 consistency. No cutoff is tuned on the data.

## 19. Sanity Checks

1. Checkpoint SHA-256 unchanged before/after; matches Experiment G metadata. **Pass.**
2. No optimizer/training path invoked; measurement code only; `torch.no_grad()` throughout. **Pass.**
3. Vector-gate replay reproduces the recorded state bit-exactly (max |Δ| = 0.0). **Pass.**
4. `Σ_i q_i = Δm_commit` and `Σ_i q_cand_i = Δm_candidate` exact to float32 (~2e-6). **Pass.**
5. Primary filtering matches Experiments G/H (candidate-correct ∧ general ∧ test); row-level equality to the G dataset (0 mismatches) and H alignment (`f_mean` diff 0.0). **Pass.**
6. E/F counts reconcile with G/H (31,763 transitions/model; primary 5,042 / 5,032). **Pass.**
7. Deterministic rerun: re-running the collector produces boolean/int arrays identical and float arrays with max diff 0.0. **Pass.**
8. `α*_margin` closed form verified against a brute-force 6,001-point line scan on real transitions (max |Δ| < 0.01). **Pass** (`tests/test_experiment_i.py`).

---

## Decision at the End

1. **Primary empirical findings.** For candidate-correct general ADD transitions, the commit fails on 732/5,042 (E) and 740/5,032 (F); failures deliver a systematically smaller fraction of the candidate's decision-margin movement (0.78–0.80 vs 0.90) and a smaller absolute delivered margin, with the deficit concentrated on the decision-helping coordinates; failures also start from a much weaker candidate margin (median ~4 vs ~8.5).
2. **E vs F comparison.** Statistically indistinguishable on this residual (all primary comparison p ≥ 0.06; failure rate 0.145 vs 0.147, p = 0.84). F's committed-state calibration advantage does not act through the ADD decision-margin delivery mechanism.
3. **Decision-margin findings.** `m_cand > 0` and `Δm_candidate > 0` for 100% of the primary set; failures lose 3.2–3.6 margin units; ~45% of failures delivered at least the exact scalar-equivalent transfer needed along the candidate direction yet still failed, and 66–70% of failures end nearer a different strongest competitor. The boundary statistic `α*_full` reproduces Experiment H's `λ*` (Spearman ≥ 0.994).
4. **Coordinate/gate findings.** The element-wise gate is coordinate-selective: on failures it delivers ~12% less on positive decision-contributing coordinates and ~5–13% more on negative ones, relative to successes; the exact linear decomposition shows this is not a global dilution.
5. **Which explanation is favored: C. Both.** Under-movement (gate delivery + weak candidate margin) is the dominant, seed-consistent component; readout/representation placement precision is additionally required for the ≈45% of failures with sufficient scalar-equivalent delivery that still land in the wrong basin. No causal claim.
6. **Exactly one next experiment:** **Experiment J — controlled inference-time commit-rule intervention on the frozen E checkpoints (interventional, compute-parity-audited).** A pre-registered, small family of commit rules (e.g. monotone gate rescaling `f→f^γ` and a coordinate-selective rule that at most preserves/redirects decision-helping movement) is applied at inference only; the causal test is whether increasing/redirecting gate delivery reduces `ρ_margin` shortfall and the B-class failure rate, and whether the off-path failures (those with `ρ_margin ≥ α*_full`) persist — which would confirm the readout/placement component. It must respect the Round-2/branching negative result that naive gate perturbation is not a valid branching mechanism, and must not be presented as a fix.
7. **What Experiment I does NOT establish.**
   * It is **observational**; it does not show that the gate *causes* the failures, that any coordinate stores a register, or that the readout is *the* bottleneck.
   * `α*_margin`/`α*_full` are local decision-boundary statistics for a linear readout, **not model thresholds** and not achievable mechanisms.
   * It does not show that reducing under-movement or sharpening the readout would improve sequence-level exact match (transition-level only; no rollouts).
   * It does not resolve the wrap/no-wrap or repeated-ADD failure budgets (reported as secondary controls only).
   * It makes no claim about F's committed-state advantage elsewhere, about other operations (SET/SWAP), or about any Phase-2 mechanism.

---

## Reproduction

```bash
# 1. Collect decision-margin + coordinate records from the frozen checkpoints (~13 s)
python experiments/run_experiment_i.py \
    --checkpoints-dir experiments/checkpoints \
    --g-transitions experiments/experiment_g_transitions.npz \
    --output experiments/experiment_i_coordinate.npz

# 2. Analysis + figures (~8 s)
python experiments/analyze_experiment_i.py \
    --coord experiments/experiment_i_coordinate.npz \
    --interp experiments/experiment_h_interp.npz \
    --output experiments/experiment_i_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_i

# 3. Tests
pytest tests/test_experiment_i.py -v      # 11 tests
pytest tests/                             # full suite
```

## Artifact List

* `experiments/run_experiment_i.py` — collector (measurement only).
* `experiments/analyze_experiment_i.py` — analyses 1–10, sanity checks, figures.
* `tests/test_experiment_i.py` — 11 tests (helpers, algebra, α* vs brute force, collector determinism, G-verification, sanity pass/fail, evidence rule, synthetic end-to-end + 12 figures).
* `experiments/experiment_i_coordinate.npz` — scalar table (61 columns) + primary per-coordinate arrays (`f`, `move`, `qcand`, `abscand`).
* `experiments/experiment_i_coordinate_meta.json` — columns, thresholds, checkpoint hashes, G-verification.
* `experiments/experiment_i_results.json` — all analyses + sanity checks + classification.
* `docs/reports/figures/experiment_i_*.png` — 12 figures: `margin_dist`, `avail_vs_deliv`, `rho_success_fail`, `alpha_star_dist`, `alpha_vs_delivered`, `coord_contrib`, `weighted_gate`, `seed_movement`, `E_vs_F`, `failure_classes`, `align_ratio`, `categories`.
* Checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`.

## Scientific Guardrails

This experiment **localizes and measures** a transition-level residual and does not establish causation. It does not claim that the element-wise gate is a symbolic memory mechanism or that the convex commit is understood. The evidence is correlational and is reported as "consistent with / associated with / localized to"; causal language is reserved for the proposed interventional Experiment J. Probe/decision-margin decodability is not conflated with exact state preservation; `α*` is not called a threshold; the scalar mean gate is not used as the explanatory mechanism (per Experiment H). Negative results are preserved (Experiment G/H reports untouched; the G/H transition data reused unchanged). No fix was implemented. The ADD candidate→commit residual (~4.9% of test ADDs, ~14.5% of candidate-correct general adds) remains one part of the Phase-1 error budget alongside wrap arithmetic and repeated-ADD chains.
