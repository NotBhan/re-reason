# RE:Reason — Experiment K: Coordinate-Preserving Gate-Balance Ablation

**Date:** October 7, 2026
**Status:** Completed — Interventional / inference-only; frozen checkpoints; no training. Classification: **B (locally beneficial, sequence-level neutral)** for the primary scientific model `E_candidate`; the secondary control `F_combined` touches the harmful band at T=20 and is reported separately as a control only.
**Frozen models:** `E_candidate` (official Phase-1 objective, primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned
**Data:** `experiments/experiment_k_records.json`, `experiments/experiment_k_results.json`
**Runners:** `experiments/run_experiment_k.py`, `experiments/analyze_experiment_k.py`, tests `tests/test_experiment_k.py`
**Checkpoints:** `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against Experiment G metadata)

---

## 1. Executive Answer

**Changing the *shape/balance* of the element-wise gate — a coordinate-preserving monotone remap of `f_i` about 0.5 that does NOT increase aggregate transfer — reproduces the local mechanistic effect that global amplification (Experiment J) produced, and does so more cleanly: it *raises* delivery on the decision-helping coordinates and *lowers* delivery on the harmful ones (which uniform amplification raised), while the mean gate does not increase. Yet it still does not improve T=10 or T=20 sequence EM, its net candidate→commit recovery is negative (regressions exceed rescues), and the off-path (placement) failures survive. Classification: **B — locally beneficial but sequence-level neutral.****

The pre-registered questions and their answers:

1. **Does sharpening increase useful decision-coordinate transfer?** **Yes, locally.** On the fixed baseline candidate-correct general-ADD set (E), delivered-margin fraction `ρ_margin` rises 0.8836 → 0.9022 (paired dz +12.15, 5/5 seeds, p < 1e-4), the contribution-weighted gate on the positive (decision-helping) coordinates rises `gw_pos` 0.6900 → 0.6977 (dz +7.71, 5/5, p = 1e-4), delivered decision margin `Δm_commit` rises (+0.369, dz +14.04, 5/5), and the candidate→commit margin shortfall falls (2.078 → 1.761, dz −7.45, 5/5).
2. **Does it change harmful-coordinate transfer?** **Yes — it *decreases* it**, which is the opposite of uniform amplification. `gw_neg` falls 0.2698 → 0.2637 (dz −4.25, 5/5, p = 7e-4). For comparison, Experiment J's uniform scaling *raised* the same statistic monotonically (γ=1.0 → 2.0: 0.2698 → 0.2843 → 0.2994 → 0.3451). This is exactly the qualitative outcome the experiment was designed to look for: positive-coordinate delivery improves while negative-coordinate delivery does not increase.
3. **Does the gate really change shape rather than magnitude?** **Yes.** Aggregate gate statistics (all coordinates, E): Δmean(f) = −0.0035, Δmedian(f) = −0.0235, Δ gate entropy = −0.0827 (all sign-consistent 5/5; paired dz −27.1 / −7.6 / −44.6). The mean transfer does **not** rise (it falls slightly); the fraction of near-saturated coordinates rises on both ends (`f < 0.25`: 0.6089 → 0.6267; `f ≥ 0.75`: 0.3230 → 0.3394) and the mid-range bins shrink — mid-range gates are pushed to their existing side of 0.5.
4. **Does it change the candidate→commit gap?** **Mixed.** The decision-margin shortfall shrinks and margin delivery improves, but the *committed accuracy on the fixed baseline set* falls slightly (0.8550 → 0.8474; dz −1.44, 5/5 down, p = 0.033), and the strongest competitor changes at the commit somewhat more often (0.655 → 0.691, dz +3.93, 5/5). More margin is delivered, but into slightly more competing basins.
5. **Does it change sequence-level capability?** **No.** E T=10 EM 0.7547 → 0.7528 (−0.18 pp; dz −0.52; 4/5 down; p = 0.31) and T=20 EM 0.4068 → 0.4122 (+0.54 pp; dz +0.71; 3/5 up; p = 0.19). Neither endpoint shows a sign-consistent, conventionally large effect (|dz| ≥ 0.8, p < 0.05). No beneficial sequence-level effect exists at any horizon.
6. **Does it introduce regressions?** **At the transition level, yes (descriptively).** Baseline candidate-correct general-ADD transitions: rescued 14.6/seed vs regressed 22.2/seed (net −7.6; dz −1.50, 5/5, p = 0.029). Sequence level: rescued 14.8/seed vs broken 17.0/seed. The strict first-real-error rate rises slightly (0.3030 → 0.3078, dz +2.68, 5/5, p = 0.004). **Operation-level, no.** SET (−0.55 pp), SWAP (+0.37 pp), ADD-no-wrap (−0.20 pp) and ADD-wrap (−0.66 pp) are all statistically null, and the controlled wrap builder actually *improves* (+6.75 pp, dz +1.40, 5/5, p = 0.035). Sharpening does not buy ADD at the expense of other operations — it simply fails to move the sequence endpoint.
7. **K vs J (the central comparison).** K κ=2.0 and J γ=1.10 reach comparable margin-delivery gains (`ρ_margin` +0.019 for both; `gw_pos` +0.008 vs +0.018) but differ in *where* the extra delivery goes: K lowers `gw_neg`, J raises it. At the transition level K is **not** safer than mild amplification — K rescues more (14.6 vs 12.4) but also regresses far more (22.2 vs 7.4), so K's net recovery (−7.6) is worse than J γ=1.10's (+5.0). At the sequence level both are flat at T=10 (−0.18 pp vs −0.22 pp) and neither improves.
8. **Decision (§15): B for E (primary).** The sharpening remap is mechanistically real and directionally as hypothesized (helpful delivery up, harmful delivery down, mean transfer flat), but it does not improve sequence-level capability and its transition-level collateral slightly exceeds its benefit. Per the pre-registered rule, a local-only improvement with T10/T20 unchanged is **B, not A**. The `F_combined` control shows the same local mechanism and a marginally significant T=20 decline (−0.30 pp, dz −1.41, 4/5, p = 0.034), i.e. it touches **D**; this is reported as a control, not the classification of K.
9. **The single next experiment (§16):** a measurement-only, frozen-checkpoint readout-side localization of the off-path component — decomposing the off-path failures into a readout-bias term and a latent state-projection term — because commit-side reweighting (magnitude in J, shape in K) is now demonstrated not to resolve them. Commit-rule modification is frozen; no further search over remapping functions.

K is an intervention test, **not** architecture adoption. No fix is proposed.

---

## 2. Existing Baseline (what G/H/I/J established, and what K adds)

* **Experiment G** (`experiment_g_add_commit_gap.md`): the ADD candidate→commit gap affects ~5.5% of candidate-correct adds, concentrated in general two-register adds; gate-dilution and blend/readout evidence both present (classification C); scalar mean gate is a proxy only.
* **Experiment H** (`experiment_h_lambda_interpolation.md`): the element-wise vector-gate replay is bit-exact; the committed state lies off the scalar old→candidate path; commit-side reasoning must be coordinate-wise.
* **Experiment I** (`experiment_i_coordinate_commit.md`): decision-margin and exact coordinate decomposition. Failures deliver ~78–80% of the candidate's margin movement (vs ~90% on successes), preferentially **under-delivering the positive (decision-helping) coordinates** while relatively over-delivering the negative ones; 44–48% of failures delivered at least the scalar-equivalent transfer needed yet still failed off-path. Verdict C (both).
* **Experiment J** (`experiment_j_commit_intervention.md`): a commit-only intervention on the full 212-dim gate (`f' = clip(γ f, 0, 1)`, recursive rollout) monotonically raises margin delivery and nets a small transition rescue only in a narrow window (γ=1.10: net +5.0/seed), but no γ>1 improves T10/T20 EM; γ≥1.25 degrades it; γ<1 is harmful; a mean-matched **coordinate-scrambled** gate collapses EM to ≈0.01. Verdict **B**; the commit/gate is a real but sub-dominant local bottleneck, and coordinate identity is functionally essential.
* **K (this report)** intervenes on the *shape* of the gate (a coordinate-preserving monotone remap about 0.5) rather than its magnitude, and finds a locally favorable, mechanistically distinct delivery profile (helpful up, harmful down) that nonetheless does not improve sequence capability.

## 3. Research Question

> Does changing the SHAPE / BALANCE of the existing element-wise gate — while preserving coordinate identity and without simply increasing aggregate transfer everywhere — improve inference-time performance?

Framed as a controlled intervention on the frozen checkpoints, this tests whether a routing-preserving (balance-aware) commit change can help where uniform magnitude scaling (Experiment J) could not.

## 4. Hypothesis and Pre-Registration

**Hypothesis (K).** If Experiment I's coordinate-selective under-movement reflects a *balance* problem in the learned gate (helpful coordinates suppressed, harmful coordinates relatively over-delivered), then a coordinate-preserving monotone remap of `f_i` about 0.5 that sharpens the gate toward its existing side of 0.5 — and therefore does not increase aggregate transfer — should improve decision-coordinate delivery and reduce the candidate→commit residual, and (if the commit bottleneck is consequential) improve sequence EM.

**Pre-registered before any result was inspected** (fixed in `run_experiment_k.py`; stored in the records metadata):

* **Family (exactly one):** `f'_i = sigmoid(kappa * logit(clip(f_i, eps, 1-eps)))`, `eps = 1e-6`.
* **Grid (exactly two conditions):** `kappa = 2.0` (sharpening) vs `kappa = 1.0` (identity baseline). **Closed:** no kappa value is added, removed, tuned, or searched after seeing results.
* **Transform properties (documented, and tested):** `f < 0.5 ⇒ f' < 0.5`; `f = 0.5 ⇒ f' = 0.5` exactly; `f > 0.5 ⇒ f' > 0.5`; coordinate identity/order preserved (element-wise, no permutation); already-near-0 / near-1 gates become more saturated; mid-range gates are pushed to their existing side of 0.5; the remap is **not** equivalent to `f → γ f` and does not intentionally increase every coordinate's transfer uniformly.
* **Intervention depends only on `f_i`.** No label, no logits, no decision margin, no register identity, no operation type, no candidate correctness, and no analysis quantity from Experiments I/J enters it.
* **No new random/scramble control.** Experiment J already established coordinate scrambling is catastrophic; K's question is gate *shape* while preserving coordinate identity, so an additional control would complicate interpretation without answering the central question.
* **Decision rule:** A requires a beneficial, sign-consistent (≥4/5 seeds) and conventionally large (|Cohen's dz| ≥ 0.8, p < 0.05) paired **sequence-level** effect at T=10 or T=20, together with net candidate→commit rescue and no large operation-level regression. B = the local commit mechanism moves (margin/coordinate delivery) or net rescue is positive, but there is no beneficial sequence-level effect. C = no meaningful benefit. D = harmful sequence-level effect. **A local-only improvement with T10/T20 unchanged classifies as B, not A.** No pp threshold is tuned on the data.

## 5. What Is Frozen

Recurrent core, candidate network, observation encoder, gate network, readout, benchmark, task generator, optimizer, loss, model dimensions, and all Phase-2 machinery. No retraining, no weight modification, no added trainable parameters, no optimizer; every inference under `torch.no_grad()`. Only the commit combination `(1−f)S + fC → (1−f')S + f'C` is changed, with `f'` a function of `f` alone, and with `f` and `C` recomputed from the modified state exactly as the frozen core does (complete recursive rollout, reusing Experiment J's `rollout` verbatim).

## 6. What Changed

New code and artifacts only (`run_experiment_k.py`, `analyze_experiment_k.py`, `tests/test_experiment_k.py`, the records/results JSON, 14 figures, this report). No completed report was modified; Experiments G/H/I/J reports and their artifacts are untouched. Experiment J's historical results are read (read-only) by the analysis for the comparison figure and are not regenerated or changed.

## 7. Dataset and Protocol

* **Data.** For each of E and F × seeds {42..46}: the canonical strict test split (1,200 sequences × 10 steps, never trained on) and the horizon sets `horizon_test_set(seed, L)` for `L ∈ {1,2,3,5,8,10,15,20}` (identical to the generator-repair / Experiment J protocol); plus the five existing controlled ADD builders (isolated / ADD→SET / 4× repeated / no-wrap / wrap; 400 sequences each). **All kappa conditions use the same sequence sets** (paired comparison).
* **Configurations.** E: 2 kappas × 5 seeds; F: 2 kappas × 5 seeds — 20 full recursive rollouts total.
* **Primary analysis set.** candidate-correct ∧ general-stratum ADD ∧ strict test split (`general = dest ≠ src ∧ source operand > 0`), the Experiment G/H/I definition; on E this is 987 / 1,084 / 960 / 1,018 / 993 rows for seeds 42–46 (1,008/seed on average; **5,042 pooled**, matching Experiment I). Secondary controls (no-op, same-register, candidate-wrong, wrap/no-wrap, controlled builders) are reported separately and never pooled into the primary result.
* **Endpoints.** Primary: T=10 and T=20 terminal EM (sequence-level). Secondary: T=5, T=15, T=1 ghost errors, full horizon curve, per-register accuracy, written-register accuracy, candidate accuracy, committed accuracy, strict/real first-error rate, gate-shape statistics, decision-margin delivery, positive/negative coordinate delivery, candidate→commit gap, candidate-margin strata, off-path subset, ADD no-wrap/wrap/repeated-ADD, SET/SWAP, controlled builders, and rescue/regression vs the kappa=1 baseline.

### Decision-measure and coordinate definitions (reused exactly from Experiments I/J)

For destination register `d` and true value `y`, `z(S) = head(S)` is affine (single `nn.Linear`), so with `a* = argmax_{a≠y} z_a(S_{t−1})` fixed at the pre-commit state:

```
m(S)       = z_y(S) − z_{a*}(S)                    (exact affine in S)
Δm_candidate = m(C_t) − m(S_{t−1}),  Δm_commit = m(S_t) − m(S_{t−1})
ρ_margin   = Δm_commit / Δm_candidate              (defined only when Δm_candidate > EPS_ABS = 1e-3)
margin_shortfall = m(C_t) − m(S_t)
q_cand_i   = (w_y − w_{a*})_i (C_i − S_prev_i)      (candidate-available contribution)
q_commit_i = (w_y − w_{a*})_i f'_i (C_i − S_prev_i) (gate-delivered contribution; uses the MODIFIED gate)
gw_pos = Σ_{q_cand>0} q_cand_i f'_i / Σ_{q_cand>0} q_cand_i
gw_neg = Σ_{q_cand<0} (−q_cand_i) f'_i / Σ_{q_cand<0} (−q_cand_i)
gw_all = Σ |q_cand_i| f'_i / Σ |q_cand_i|,  gw_mag = Σ |C_i−S_prev_i| f'_i / Σ |C_i−S_prev_i|
α*_full  = minimal α with the full argmax of head(S_prev + α(C−S_prev)) equal to y (exact, linear head)
```

`Σ_i q_commit_i = Δm_commit` holds exactly (max residual ~1e-6 in float32). `α*` values are empirical local decision-boundary statistics for the linear readout, not model thresholds.

### Gate-shape statistics

For the full 212-dim gate and per ADD transition: mean, median, entropy, and the coordinate fractions `f < 0.25`, `0.25 ≤ f < 0.5`, `0.5 ≤ f < 0.75`, `f ≥ 0.75`, before and after the remap; the coordinate-wise absolute change `|f′−f|`, the signed change `f′−f`, the mean transfer change `mean(f′) − mean(f)`, and per-coordinate identity preservation. Aggregate Δ mean(f), Δ median(f), Δ entropy are reported.

### Statistical discipline

Seeds (the 5 frozen E checkpoints) are the independent units. Transition-level rescue/regression counts and gate-shape distributions are **descriptive** (transitions within a sequence are dependent). Primary inference is **seed-level paired** (κ=2 vs κ=1; mean ± SD, Cohen's dz, paired t, sign k/5). Cohen's dz bands (negligible < 0.2, small < 0.5, medium < 0.8, large ≥ 0.8) are declared conventions, not data-tuned cutoffs.

## 8. Experimental Procedure

1. Load a frozen checkpoint; verify SHA-256 against Experiment G metadata.
2. κ=1 rollout on the canonical test set; verify it reproduces `forward_with_states` and the established headline metrics; at κ=1 the remap is the exact identity.
3. For κ=2: recursive rollout; horizon EM; transition-level decoding; per-ADD decision margins, coordinate decomposition and gate-shape statistics; controlled builders; rescue/regression vs the κ=1 baseline.
4. Re-run the entire configuration a second time (and a third after the final code freeze); require deterministic equality of every record.

## 9. Results — Baseline Reproduction and Determinism

* **κ=1 is the exact identity:** `tighten(f, 1.0)` returns `f` unchanged (tested bit-exact), so the κ=1 rollout is the frozen forward. `_verify_baseline_reproduction` passes for **all 10** checkpoint-seeds (states, candidates, gates, decoded states and terminal predictions identical; max float difference < 1e-5).
* **κ=1 headline metrics reproduce the frozen baselines:** E T=10 EM **0.7547 ± 0.0191** (report 0.755 ± 0.019), E T=20 **0.4068**; F T=10 **0.7370** (report 0.737 ± 0.015), F T=20 **0.3898**. At κ=1 the gate-shape deltas are exactly 0 and rescue/regression counts are exactly 0.
* **Determinism:** the full 20-record configuration was run twice; metadata (excluding the timestamp and wall-clock) and **all 20 records** are identical (0 mismatches). A third run after the final code freeze is again identical. Checkpoint SHA-256 verified before and after every run.

## 10. Results — Sequence-Level (E, κ=2 vs κ=1, mean over 5 seeds)

| Horizon | κ=1 EM | κ=2 EM | Δ (pp) | dz | sign | p |
|---:|---:|---:|---:|---:|---:|---:|
| T=1 | 0.8800 | 0.8450 | −3.50 | −1.23 | 4/5 ↓ | 0.052 |
| T=2 | 0.9816 | 0.9794 | −0.22 | −1.20 | 4/5 ↓ | 0.055 |
| T=3 | 0.9850 | 0.9834 | −0.16 | −1.13 | 3/5 ↓ | 0.13 |
| T=5 | 0.9524 | 0.9502 | −0.22 | −0.82 | 3/5 ↓ | 0.14 |
| T=8 | 0.8212 | 0.8176 | −0.36 | −1.38 | 4/5 ↓ | 0.043 |
| **T=10** | **0.7547** | **0.7528** | **−0.18** | **−0.52** | **4/5 ↓** | **0.31** |
| T=15 | 0.5420 | 0.5438 | +0.18 | +0.27 | 3/5 ↑ | 0.58 |
| **T=20** | **0.4068** | **0.4122** | **+0.54** | **+0.71** | **3/5 ↑** | **0.19** |

No endpoint attains the pre-registered beneficial signature (sign-consistent ≥4/5 **and** |dz| ≥ 0.8 **and** p < 0.05). T=10 is flat-to-marginally-down; T=20 is flat-to-marginally-up; short horizons (T=1–T=8) are marginally down (T=1 ghost errors rise: 0.880 → 0.845). The F control is qualitatively identical (T=10 0.7355, T=20 0.3868).

## 11. Results — Local Mechanistic Effect (E, κ=2 vs κ=1, primary set)

| Quantity | κ=1 | κ=2 | Δ | dz | sign | p |
|---|---:|---:|---:|---:|---:|---:|
| ρ_margin (median) | 0.8836 | 0.9022 | +0.0186 | +12.15 | 5/5 ↑ | <1e-4 |
| gw_pos (helpful) | 0.6900 | 0.6977 | +0.0077 | +7.71 | 5/5 ↑ | 1e-4 |
| gw_neg (harmful) | 0.2698 | 0.2637 | **−0.0061** | −4.25 | 5/5 ↓ | 7e-4 |
| gw_all | 0.5931 | 0.5967 | +0.0036 | +3.79 | 5/5 ↑ | 1e-3 |
| gw_mag | 0.3356 | 0.3318 | −0.0038 | −4.58 | 5/5 ↓ | 5e-4 |
| m_cand (median) | 7.905 | 7.822 | −0.083 | −0.73 | 4/5 ↓ | 0.18 |
| m_commit (median) | 5.840 | 6.065 | +0.224 | +2.14 | 5/5 ↑ | 0.009 |
| Δm_commit (mean) | 15.522 | 15.891 | +0.369 | +14.04 | 5/5 ↑ | <1e-4 |
| margin_shortfall (median) | 2.078 | 1.761 | −0.317 | −7.45 | 5/5 ↓ | 1e-4 |
| committed acc (primary) | 0.8550 | 0.8474 | −0.0077 | −1.44 | 5/5 ↓ | 0.033 |
| competitor changed at commit | 0.655 | 0.691 | +0.037 | +3.93 | 5/5 ↑ | 9e-4 |
| first-real-error rate | 0.3030 | 0.3078 | +0.0048 | +2.68 | 5/5 ↑ | 0.004 |

**Reading.** Sharpening delivers more of the candidate's decision margin and concentrates the additional delivery on the decision-helping coordinates while *reducing* delivery on the harmful ones — the direction Experiment I identified as the deficit. The candidate→commit margin shortfall shrinks. **But** committed accuracy on the same fixed set falls slightly and the committed competitor changes more often: more margin is delivered, and slightly more often into a different competitor's basin. This is the first interventional evidence that the positive/negative delivery asymmetry can be moved in the *favorable* direction without global amplification — and that doing so is nonetheless insufficient at the sequence level.

## 12. Results — Gate Shape / Balance (E, κ=2)

Aggregate over all gate coordinates (5-seed means, paired):

| Quantity | κ=1 | κ=2 | Δ | dz | sign |
|---|---:|---:|---:|---:|---:|
| mean(f) | 0.3603 | 0.3569 | **−0.0035** | −27.1 | 5/5 ↓ |
| median(f) | — | — | **−0.0235** | −7.6 | 5/5 ↓ |
| entropy(f) | — | — | **−0.0827** | −44.6 | 5/5 ↓ |
| frac(f < 0.25) | 0.6089 | 0.6267 | +0.0178 | +44.9 | 5/5 ↑ |
| frac(f ≥ 0.75) | 0.3230 | 0.3394 | +0.0164 | +26.6 | 5/5 ↑ |
| per-transition |f′−f| (mean) | 0 | 0.0299 | +0.0299 | +117 | 5/5 ↑ |
| per-transition (f′−f) (mean) | 0 | −0.0020 | −0.0020 | −3.6 | 5/5 ↓ |
| per-transition transfer change | 0 | −0.0020 | −0.0020 | −3.6 | 5/5 ↓ |

The mean and median transfer **decrease**, the entropy **decreases**, and both saturation tails grow while the mid-range shrinks: this is a balance/shape change, not an amplification. Coordinate identity is preserved structurally (the remap is element-wise; tested to commute with coordinate permutation). The per-coordinate signed change `f′−f` is small and mixed-signed (per-dimension means range roughly ±0.01), which is why large positive *and* negative changes cancel in the aggregate while |f′−f| ≈ 0.03.

## 13. Results — Rescue / Regression (E, κ=2 vs κ=1)

* **Transition level (baseline candidate-correct general ADD):** rescued **14.6**/seed, regressed **22.2**/seed, **net −7.6** (dz −1.50, 5/5, p = 0.029). Sharpening recovers a minority of baseline failures but breaks more baseline successes.
* **Sequence level:** rescued **14.8**/seed, broken **17.0**/seed. Sequence damage exceeds sequence recovery.
* **Off-path subset** (baseline failures with `ρ_margin ≥ α*_full`; n ≈ **63/seed**): only **6.35%** are rescued, and **100%** of them end nearer a *different* strongest competitor at the commit under both κ=1 and κ=2. The scalar-sufficient-transfer failures are essentially untouched by sharpening — consistent with Experiment I's placement/readout component and with Experiment J's observation that amplification cannot resolve them either.

## 14. Results — Candidate-Margin Strata and Operation-Level Safety (E)

* **Candidate-margin strata** (baseline terciles, pre-declared from the κ=1 distribution): committed accuracy and rescue rates change only marginally and in no strong direction within low/mid/high strata; sharpening does not selectively rescue weak-margin states.
* **Operation-level (final-op EM):** SET 0.8365 → 0.8310 (−0.55 pp, dz −0.62, n.s.), SWAP 0.7776 → 0.7813 (+0.37 pp, n.s.), ADD-no-wrap 0.7094 → 0.7075 (−0.20 pp, n.s.), ADD-wrap 0.4930 → 0.4864 (−0.66 pp, n.s.).
* **Controlled builders:** repeated-ADD 0.1475 → 0.1540 (+0.65 pp, n.s.); wrap 0.3550 → **0.4225** (+6.75 pp, dz +1.40, 5/5, p = 0.035); isolated ADD 0.6685 → 0.6745 (+0.60 pp, n.s.). No significant operation-level regression; the wrap builder improves.

## 15. Results — Statistical Analysis and Compute

* Primary inference (E, κ=2 vs κ=1) uses paired seed-level tests (n = 5) with Cohen's dz and sign consistency, reported in the tables above. Representative values: `ρ_margin` +0.0186 (dz +12.15, 5/5, p < 1e-4); `gw_pos` +0.0077 (dz +7.71, 5/5, p = 1e-4); `gw_neg` −0.0061 (dz −4.25, 5/5, p = 7e-4); T=10 −0.18 pp (dz −0.52, 4/5, p = 0.31); T=20 +0.54 pp (dz +0.71, 3/5, p = 0.19); net transition recovery −7.6 (dz −1.50, 5/5, p = 0.029); first-real-error +0.0048 (dz +2.68, 5/5, p = 0.004). The F control reproduces the local pattern; its only "harmful" endpoint is T=20 (Δ = −0.30 pp, dz −1.41, 4/5, p = 0.034).
* **Compute / parity:** identical parameter count (**226,358**), **zero additional trainable parameters**, no optimizer, no training, inference only. The intervention adds only element-wise operations per state coordinate per step (clamp/log/log1p/sub/mul/exp/div): ≈ 2,120 element-wise ops/step against ≈ 206,976 MACs/step, i.e. ≈ **1.0%** of per-step operations in MAC terms. A clean T=10 rollout micro-benchmark (1,200 sequences, median of 5, 8 CPU threads) gives 65.9 ms (κ=1) vs 71.1 ms (κ=2), a measured **1.08×** wall-clock ratio. Total wall clock for the 20-configuration run ≈ **37 s** (CPU).

## 16. Interpretation — Empirical Result, Causal Reading, Unresolved

**Empirical result.** A coordinate-preserving monotone remap of the learned element-wise gate about 0.5 (`κ=2`) changes the gate's shape (lower mean/median/entropy; more saturated tails) and produces a locally favorable delivery profile on the fixed baseline failure set: more decision-margin delivery, more delivery on the decision-helping coordinates, and *less* delivery on the harmful ones. It nonetheless leaves T=10 and T=20 sequence EM unchanged (no beneficial effect) and produces a small negative net transition recovery and a small increase in the first-real-error rate. No operation-level regression appears; the controlled wrap builder improves.

**Supported causal reading (interventional, commit rule only).** Changing the gate's *shape* is causally connected to decision-margin delivery and to the positive/negative delivery split — a commit-only intervention moves those quantities in the hypothesized direction with large, seed-consistent effects. This is a second, distinct causal handle on the local residual (magnitude in J, shape in K), and it confirms Experiment I's asymmetry account at the interventional level: the gate can be re-shaped to raise helpful delivery and lower harmful delivery simultaneously.

**But the commit-side residual is not the sequence-level bottleneck.** Improving the *balance* of delivery does not improve sequence capability. The relevant limiting interpretation is unchanged from J and is now stronger: the commit/gate performs a balancing function, and neither increasing its magnitude (J) nor re-shaping its balance (K) recovers long-horizon capability. Two facts reinforce this. First, the primary committed accuracy *decreased* slightly even as margins improved — extra delivery lands in competing basins more often. Second, the off-path failures (baseline ρ ≥ α*_full) remain ≈ 94% unresolved and change competitor 100% of the time; the placement component is not a deliverable-amount problem and is not addressable by commit-side reweighting.

**Unresolved explanation.** K cannot say *why* a favorable delivery profile fails to convert into sequence capability. Candidates: (i) extra delivered margin is spent entering competing basins (supported by the competitor-change increase and the small committed-accuracy drop); (ii) the sequence-level error budget is dominated not by candidate→commit failures but by wrap arithmetic, repeated-ADD chains and candidate *generation* errors upstream of the commit (supported by the fact that the wrap builder's local improvement does not move sequence EM); (iii) the off-path placement component is a frozen-readout tolerance property that commit-side changes cannot reach. K does not discriminate among (i)–(iii); all three are consistent with the data.

## 17. What the Experiment DOES Establish

* A clean, reproducible, compute-parity-audited interventional test of the **shape** of the commit gate (κ=1 is the exact identity and reproduces the frozen baseline bit-for-bit at decode level and to float32 round-off at state level).
* That a coordinate-preserving monotone remap of the gate about 0.5, which does **not** increase aggregate transfer (mean/median/entropy all decrease), is causally sufficient to **raise decision-margin delivery and move the positive/negative delivery split in the favorable direction** (`gw_pos` up, `gw_neg` down; ρ_margin up; shortfall down), in every seed.
* That this favorable local delivery profile does **not** translate into sequence-level capability: T=10/T=20 EM are unchanged (no beneficial sign-consistent large effect), and net transition recovery is negative.
* That the shape intervention is **operationally safe at the category level** — no significant SET/SWAP/ADD-no-wrap/ADD-wrap regression, and a significant improvement in the controlled wrap builder — so its failure is not "improved ADD at the cost of others."
* That the **off-path** (placement) failures survive the shape intervention (≈6% rescued; 100% competitor change), reinforcing that the residual is not purely a delivery-amount problem.
* That K's delivery profile is mechanistically distinct from Experiment J's uniform amplification: K lowers `gw_neg` while J raises it (J γ=1.0→2.0: 0.2698→0.2843→0.2994→0.3451).

## 18. What the Experiment DOES NOT Establish

* It does not show that gate re-shaping is a **fix**; it improves no sequence endpoint and has negative net transition recovery.
* It is interventional only with respect to the commit rule; it does **not** show that the gate is a balanced routing mechanism, that the representation is understood, or that the readout is uninvolved.
* It does not claim any coordinate stores a register, nor that the commit is the dominant long-horizon bottleneck.
* It does not show that *some other* remap could not help — it tests exactly one pre-registered remap (κ=2) plus its identity, on the register-tracking benchmark and the E/F objectives.
* It does not resolve the off-path placement component, the wrap/repeated-ADD budgets, or candidate-generation errors; and it does not generalize to Phase-2 machinery (branching, verifier, recursion), which remains out of scope.
* Transition-level rescue/regression and gate-shape distributions are descriptive, not inference at the independent unit.

## 19. Decision

**B — Locally beneficial but sequence-level neutral (E primary; F control shows the same local mechanism and touches the harmful band at T=20).** The sharpening remap is a genuine, seed-consistent causal handle on decision-margin delivery and on the helpful/harmful delivery split — it achieves the favorable delivery profile that Experiment I identified as the deficit, and it does so *without* the harmful-coordinate over-delivery produced by uniform amplification. But it does not improve T=10/T=20 sequence EM, its net transition recovery is negative (regressions exceed rescues), and the off-path placement failures persist. Per the pre-registered rule, a local-only improvement with T10/T20 unchanged is **B, not A**.

**Consequence for the commit rule: freeze it.** Both magnitude (J) and shape (K) interventions on the element-wise commit gate have now been shown to move local delivery without moving sequence capability, with comparable-or-worse collateral. Commit-rule modification does not merit further study in Phase 1; no search over remapping functions will be opened.

## Final Decision (required answers)

1. **Does sharpening increase useful decision-coordinate transfer?** Yes, locally and seed-consistently (ρ_margin 0.8836 → 0.9022; gw_pos 0.6900 → 0.6977; Δm_commit +0.369; 5/5, p < 1e-3).
2. **Does it change harmful-coordinate transfer?** Yes — it *decreases* it (gw_neg 0.2698 → 0.2637, 5/5), unlike uniform amplification, which raised it.
3. **Does it change the candidate→commit gap?** Margin shortfall falls (2.078 → 1.761) but committed accuracy on the fixed set falls slightly (0.8550 → 0.8474, 5/5, p = 0.033) and the committed competitor changes more often (0.655 → 0.691).
4. **Does it change sequence-level capability?** No (T=10 −0.18 pp, p = 0.31; T=20 +0.54 pp, p = 0.19; no beneficial endpoint).
5. **Does it introduce regressions?** Transition: rescued 14.6 vs regressed 22.2 (net −7.6); sequence: 14.8 rescued vs 17.0 broken; first-real-error +0.48 pp (5/5). Operation-level: no significant regression; controlled wrap improves (+6.75 pp).
6. **Does gate shape behave differently from magnitude?** Yes: mean/median/entropy decrease and `gw_neg` decreases, whereas J's magnitude scaling increased all three. Shape and magnitude are mechanistically distinct handles.
7. **Is the evidence A / B / C / D?** **B** for `E_candidate` (primary); the `F_combined` control reproduces the local mechanism and registers a marginal T=20 decline (D-band at that endpoint), reported as a control.
8. **Exactly one next scientific experiment:** **Experiment L — readout-side localization of the off-path component (measurement-only, frozen readout).** Decompose the off-path failures (baseline candidate-correct failures with `ρ_margin ≥ α*_full`) into a readout-bias term and a latent state-projection term using the frozen linear head, to determine whether the placement component is a readout-threshold/calibration property or a genuine latent misplacement — the one component neither J (magnitude) nor K (shape) can reach.

## Reproduction

```bash
# 1. Run the pre-registered two-condition intervention (E + F control, 5 seeds) (~40 s)
python experiments/run_experiment_k.py --out experiments/experiment_k_records.json

# 2. Analysis + 14 figures (~10 s)
python experiments/analyze_experiment_k.py \
    --records experiments/experiment_k_records.json \
    --output experiments/experiment_k_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_k

# 3. Tests
pytest tests/test_experiment_k.py -v      # 18 tests
pytest tests/                             # full suite
```

Determinism: the full 20-record configuration was run twice (and a third time after the code freeze); every record is identical and the metadata is identical after excluding the timestamp and wall-clock fields.

## Artifact List

* `experiments/run_experiment_k.py` — pre-registered intervention runner (κ grid, recursive rollout, horizon/transition/margin/gate-shape/controlled evaluation, rescue accounting).
* `experiments/analyze_experiment_k.py` — paired seed-level analysis, gate-balance accounting, strata, off-path subset, operation-level safety, A/B/C/D decision, comparison to Experiment J, 14 figures.
* `tests/test_experiment_k.py` — 18 tests (identity, unit-interval, sign/monotonicity/fixed-point, coordinate-identity preservation, non-equivalence to scaling, checkpoint hashes, no-grad/no-param-change, determinism, baseline-frozen reproduction, rescue/margin/gate-shape accounting, paired alignment, synthetic end-to-end + figure generation, invalid config, verdict logic).
* `experiments/experiment_k_records.json` — 20 per-(model, seed, κ) records + pre-registration metadata + gate-shape statistics.
* `experiments/experiment_k_results.json` — summaries, paired statistics, decision, compute parity, J reference.
* `docs/reports/figures/experiment_k_*.png` — 14 figures: `t10_t20_em`, `horizons_em`, `rescue_regression`, `gate_distribution`, `gate_entropy_shape`, `margin_delivery`, `gw_pos_vs_neg`, `commit_gap`, `margin_strata`, `add_ops`, `set_swap_regression`, `offpath_rescue`, `coord_gate_change`, `vs_j_reference`.
* Checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against `experiments/experiment_g_transitions_meta.json`).
* Historical reference (read only, unchanged): `experiments/experiment_j_results.json`.

## Scientific Guardrails

This is an interventional test of the commit rule only; it is not a production fix and does not establish that the remap is correct or that the representation is understood. "Sharpening" and "balance" are used as hypothesis labels, not as claims that the learned gate is a balanced routing mechanism. No claim is made that the gate is a symbolic memory mechanism, that coordinates represent registers, that the architecture is sufficient for general reasoning, or that the readout is uninvolved. Transition-level rescue is explicitly separated from sequence-level capability; `α*` is not called a threshold; the scalar mean gate is not used as the explanatory mechanism (Experiment H). The `F_combined` result is reported as a control, not as the classification of K. Negative results are preserved; Experiments G/H/I/J reports and artifacts are untouched, and Experiment J's history is read-only.
