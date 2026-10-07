# RE:Reason — Experiment H: λ-Interpolation Decode Sweep of the ADD Commitment

**Date:** October 5, 2026
**Status:** Completed — Measurement-only (no training, no model changes, no rollouts, frozen checkpoints)
**Frozen models:** `E_candidate` (official Phase-1 objective) and `F_combined` (control), 5 seeds each, 226,358 parameters, SHA-256-pinned
**Data:** [`experiments/experiment_h_interp.npz`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/experiment_h_interp.npz), [`experiment_h_interp_meta.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/experiment_h_interp_meta.json), [`experiment_h_results.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/experiment_h_results.json)
**Runners:** [`experiments/run_experiment_h.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/run_experiment_h.py), [`experiments/analyze_experiment_h.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/analyze_experiment_h.py), tests [`tests/test_experiment_h.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/tests/test_experiment_h.py)

---

## Executive Answer

**The scalar old→candidate path is monotone and smooth, but it is decodable almost only near the candidate end (median λ* = 0.80 on failures; only 1.2% of failures are decodable at λ ≤ 0.5). The actual commit is not a point on that path: the element-wise gate already decodes at 0.855, while replacing it by its own mean gate collapses decoding to 0.182 / 0.042 (E / F) — so the gate-vector structure governs the commit and scalar f̄ is not a faithful summary. Classification: C (both mechanisms present), with the sharper statement that any commit-side reasoning must be coordinate-wise.**

1. **Sanity checks are exact.** λ=0 reproduces the previous-state decode and λ=1 the candidate decode (1.000 both models); the vector-gate replay `(1−f_t)⊙S_(t−1) + f_t⊙C_t` reproduces the recorded committed state **bit-exactly** (max |Δ| = 0.0) and its decode equals the recorded committed decode (1.000); the stored λ* column matches the recomputed value (0.0 diff); checkpoint hashes unchanged; the re-collected transitions match Experiment G's dataset exactly (0 mismatches; float columns 0.0 diff).
2. **Scalar interpolation path (candidate-correct general ADDs, primary).** Accuracy rises smoothly and monotonically from ≈0 (λ ≤ 0.2) to 1.0 (λ = 1): E/F at λ = 0.4: 0.200 / 0.039; λ = 0.5: 0.356 / 0.254; λ = 0.6: 0.546 / 0.514; λ = 0.7: 0.749 / 0.712; λ = 0.8: 0.903 / 0.887; λ = 0.9: 0.973 / 0.964. No monotonicity violations, no interior peak, zero revert trajectories. Wrong intermediate slots are the **old value 92% of the time** (near-path interpolation 1.2%, other 6.5%): the readout holds the old value and snaps to the correct value late — it does not wander through "third" values on this path.
3. **Actual vector-gate behavior.** The replay equals the recorded commit (accuracy 0.855 (E) / 0.853 (F) on primary, identical to recorded). The **mean-gate counterfactual** `head((1−f̄)S + f̄C)` decodes at **0.182 (E) / 0.042 (F)** — a 67–81 pt collapse — and disagrees with the replay decode on 77% / 90% of transitions. Off-path distance of the actual state from the S_prev→C segment: mean 6.34 / 6.47 (segment length ≈ 15). **H5 supported: the element-wise gate does something the scalar cannot represent.**
4. **Failures (candidate-correct, committed-wrong; n = 732 (E) / 740 (F)).** λ* exists for 100% (λ = 1 is the candidate decode by selection), median λ* = 0.80 (p25 0.70–0.75, p95 1.0); only **1.2% have λ* ≤ 0.5**, 5.7% / 8.0% need λ* = 1.0 exactly. The actual committed state's projection onto the path is λ_eff ≈ 0.30, and **λ* ≤ λ_eff for only 0.3% (E) / 0.1% (F)** of failures; λ* ≤ f̄ for 0.4% / 0.1%. In scalar terms the commit stops far short of the region where this readout decodes.
5. **Candidate-wrong transitions (secondary).** 81.9% (E) / 78.3% (F) have **no λ at all** that decodes the true value — full scalar transfer cannot recover a wrong candidate, consistent with the earlier information argument. (No claim beyond that: this is a readout-level statement.)
6. **Category quantification (transition-level).** Candidate → committed gaps: test all 8.85→8.52 (−3.4 pts); test general 7.73→6.90 (−8.3); test wrap 7.55→6.56 (−9.9); controlled isolated 7.97→6.69 (−12.8); controlled wrap 4.84→3.55 (−12.9); controlled 4× repeated 6.02→5.30 (−7.2). The scalar mean-gate counterfactual sits at 0.14–0.50 in every category — always far below the actual commit.
7. **E vs F.** Primary curves and λ* are nearly identical (median λ* = 0.80 in both; commit/replay accuracy difference p = 0.84). One descriptive difference: E's scalar path is more decodable in the mid-range (λ = 0.5: 35.6% vs 25.4%, seed-paired p = 0.008); F's start-old failures need slightly more transfer (λ* mean 0.810 vs 0.797, p = 0.087). Neither changes the commit residual (Experiment G's paired result stands).

**Decision (§13): C. Both** — see the precise decomposition below. **Next single experiment (§14): coordinate-wise commit analysis (gate-vector level)** — the scalar view is now shown to be inadequate (H5), so the next measurement must decompose the actual element-wise gate.

---

## Method

* Runner collects, for every ADD transition of the strict test split and the five controlled builders (same data/order as Experiment G): `S_(t−1)`, `C_t`, `S_t`, the 212-dim gate `f_t`, scalar summaries (`f̄`, min, max, distances), decodes of `S_prev`, `C`, `S_t`, the vector-gate replay `(1−f)⊙S_prev + f⊙C`, the mean-gate counterfactual, λ_eff = projection of `S_t` onto the `S_prev→C` segment, off-path distance, and the 21-point λ sweep (λ ∈ {0.00,…,1.00}) of decoded destination digits and correctness.
* 31,763 transitions per model; the runner re-runs inference **only because G stored scalar summaries, not the vectors**; every re-collected row is verified against G's dataset (exact match on all integer columns, 0.0 diff on `f̄`, `d_cs`, `d_ss`, `d_sc`).
* Primary analysis set: **candidate-correct ∧ general-stratum ADD** (destination ≠ source register, source operand > 0; source = test split). No-op and same-register adds are secondary controls; controlled builders are used for categories.
* **λ* is an empirical decodable-point statistic along the scalar path — not a causal threshold.** `S(f̄_t)` is a counterfactual scalarization, never described as the actual commit. All comparisons between scalar λ and the vector-gate state are descriptive.
* Statistical discipline: transition-level tables are descriptive (transitions within sequences are dependent); independent inference is seed-level (n = 5, paired E vs F).

---

## 1. Scalar interpolation-path behavior (primary set)

| λ | 0.0 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E accuracy | 0.001 | 0.024 | 0.086 | 0.200 | 0.356 | 0.546 | 0.749 | 0.903 | 0.973 | 1.000 |
| F accuracy | 0.001 | 0.002 | 0.005 | 0.039 | 0.254 | 0.514 | 0.712 | 0.887 | 0.964 | 1.000 |

* Curve shape: smooth sigmoid, strictly monotone (0 violations; max consecutive drop +0.0006), final = candidate accuracy (by construction and verified).
* λ* on all primary transitions: E mean 0.591, median 0.60 (p5 0.30, p95 0.90), 35.6% decodable by λ = 0.5; F mean 0.637, median 0.60, 25.4% by λ = 0.5.
* λ* on failures: median 0.80 (both models); 1.2% ≤ 0.5; p95 = 1.0.
* Wrap vs no-wrap (primary): no-wrap transitions decode earlier than wrap transitions throughout the mid-range (E at λ = 0.5: 0.469 vs 0.024; λ = 0.6: 0.685 vs 0.137; λ = 0.7: 0.833 vs 0.501; λ = 0.8: 0.935 vs 0.810) — wrap failures require near-full transfer to flip, consistent with wrap being the hardest arithmetic class.
* Conditionals on the committed outcome: failures that decoded to the **old value** need *more* transfer than failures that decoded to a **third value** (e.g., E at λ = 0.8: 0.383 old vs 0.687 third) — consistent with "old value still dominating the readout" vs "state already moved off-path".
* Controls: no-op transitions decode correctly from λ = 0 (acc 0.995–0.996; λ* median 0.0) — the sweep semantics are validated; same-register adds sit between (λ* median 0.50–0.55).

## 2. Actual vector-gate behavior (candidate-correct general ADDs)

| Quantity | E | F |
|---|---:|---:|
| candidate `head(C)` | 1.000 | 1.000 |
| recorded commit `head(S_t)` | 0.855 | 0.853 |
| **vector-gate replay** `head((1−f)⊙S+f⊙C)` | **0.855** | **0.853** |
| **mean-gate scalar** `head((1−f̄)S+f̄C)` | **0.182** | **0.042** |
| λ = 1 | 1.000 | 1.000 |
| λ = 0 (previous state) | 0.0008 | 0.0008 |
| replay vs recorded decode agreement | 1.000 | 1.000 |
| mean-gate vs replay decode agreement | 0.230 | 0.086 |
| max replay state difference | 0.0 | 0.0 |
| λ_eff (projection of S_t) mean | 0.305 | 0.317 |
| off-path distance mean (segment ≈ 15) | 6.34 | 6.47 |

The actual commit cannot be faithfully represented by a scalar point on the interpolation path — the element-wise replay is **bit-exact**, while the mean-gate scalarization is 67–81 points worse and disagrees on 77–90% of decodes.

## 3. Candidate→commit loss (transition level, all transitions of each category)

| Category | n | cand | committed | replay | mean-gate | λ=1 | cand−comm gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| test all ADDs | 15,763 | 0.885 | 0.852 | 0.852 | 0.499 | 0.885 | −0.034 |
| test general | 6,522 | 0.773 | 0.690 | 0.690 | 0.144 | 0.773 | −0.083 |
| test isolated | 11,270 | 0.909 | 0.874 | 0.874 | 0.516 | 0.909 | −0.035 |
| test repeated | 4,493 | 0.826 | 0.795 | 0.795 | 0.456 | 0.826 | −0.031 |
| test wrap | 3,271 | 0.755 | 0.656 | 0.656 | 0.020 | 0.755 | −0.099 |
| test no-wrap | 12,492 | 0.920 | 0.903 | 0.903 | 0.624 | 0.920 | −0.017 |
| controlled isolated ADD | 2,000 | 0.797 | 0.669 | 0.669 | 0.110 | 0.797 | −0.128 |
| controlled ADD→SET | 2,000 | 0.799 | 0.667 | 0.667 | 0.113 | 0.799 | −0.132 |
| controlled 4× repeated | 8,000 | 0.602 | 0.530 | 0.530 | 0.105 | 0.602 | −0.072 |
| controlled no-wrap | 2,000 | 0.676 | 0.608 | 0.608 | 0.213 | 0.676 | −0.069 |
| controlled wrap | 2,000 | 0.484 | 0.355 | 0.355 | 0.000 | 0.484 | −0.129 |

(F values differ by ≤0.7 pt except `mean-gate`, which is even lower: test general 0.035, controlled wrap 0.000.) "Scalar oracle gap" (fraction with any correct λ minus committed accuracy) is +3.2 to +15.9 pts — an oracle over λ, not an achievable mechanism. Transition-level only; no sequence-level claims are made here (consistent with prior sequence-level results in Experiments E/F/G).

## 4. Does a decodable state exist between S_prev and C?

Yes, but only near the candidate end. For candidate-correct transitions, a correct λ always exists (λ = 1), and among failures the first correct λ has median 0.80, p5 0.60, p95 1.0; only 1.2% of failures are decodable at λ ≤ 0.5 and 5.7–8.0% require λ = 1.0 exactly. In other words, the decodable region on this scalar path is not a broad "sweet spot" in the middle — it is a late, narrow band. Meanwhile 78–82% of *candidate-wrong* transitions have no correct λ at all.

## 5. Does the observed actual gate reach that region?

Not in scalar projection terms (λ* ≤ λ_eff for 0.3% / 0.1% of failures), **but this question is itself not well-posed at the scalar level**: the actual state lies off the scalar path (mean off-path distance 6.3–6.5 on a segment of length ≈15), and its decode accuracy (0.855) far exceeds any scalar blend at comparable mean transfer (mean-gate: 0.182/0.042). The genuine open question is therefore coordinate-wise: which coordinates the gate moves and how the readout scores that movement — Experiment I below.

---

## Hypothesis Evaluations (as requested)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Scalar accuracy increases smoothly toward candidate accuracy as λ→1 ⇒ transfer sufficiency | **Supported** | Strictly monotone, smooth sigmoid; end point = candidate accuracy (1.000); control: no-op adds decode from λ = 0 (λ* = 0) |
| H2 | Accuracy peaks at an intermediate λ then degrades ⇒ nontrivial readout geometry | **Not supported** | Argmax at λ = 1.0 on every curve; 0 monotone violations; 0 of 732/740 failure trajectories revert after reaching correct |
| H3 | Many candidate-correct/commit-wrong transitions have a valid λ* but the commit is wrong ⇒ commit fails to land in a decodable region | **Supported with refinement** | 100% of failures have λ*; 99.7%/99.9% have λ* > λ_eff; median λ* = 0.80 vs λ_eff ≈ 0.30. Refinement: λ_eff is a projection; the actual commit is off-path and decodes far better than its scalarization (H5) |
| H4 | Many such transitions have no valid λ* ⇒ increasing scalar transfer cannot help | **Not supported for candidate-correct** (0 no-solution, by construction) — but **supported as an information statement for candidate-wrong transitions**: 81.9%/78.3% have no λ* at all |
| H5 | If vector-gate replay differs materially from scalar mean-gate interpolation, gate-vector structure matters | **Strongly supported** | Replay = recorded commit (0.855/0.853) vs mean-gate scalar 0.182/0.042; decode disagreement 77%/90%; off-path distance 6.3–6.5 |

---

## Statistical Discipline

* All transition-level tables/figures are descriptive (transitions within sequences are dependent). Independent inference is seed-level (n = 5, paired E vs F).
* Seed-level (E): n_primary per seed 960–1,084; failures 124–165; committed accuracy 0.841–0.871; median λ* among failures = 0.80 in **every** seed; fraction λ* ≤ λ_eff = 0.0–0.7% per seed; accuracy at λ = 0.5 = 32.5–38.6%.
* Paired E vs F (mean ± SD over seeds): committed/replay accuracy 0.855 vs 0.853 (p = 0.84); λ* mean on failures 0.797 vs 0.810 (p = 0.087); fraction λ* ≤ λ_eff 0.26% vs 0.14% (p = 0.64); accuracy at λ = 0.5 35.6% vs 25.4% (p = 0.008); λ = 0/λ = 1 accuracies identical. E's scalar path is modestly more mid-range decodable; the commit residual is unchanged.

## Required Figures (`docs/reports/figures/`)

1. `experiment_h_acc_vs_lambda.png` — accuracy vs λ, pooled E/F (primary + all-test).
2. `experiment_h_acc_vs_lambda_wrap.png` — accuracy vs λ, wrap vs no-wrap (primary).
3. `experiment_h_acc_vs_lambda_failures.png` — failures only; split by old-value vs third-value decode.
4. `experiment_h_lambda_star_dist.png` — λ* distribution (failures).
5. `experiment_h_lambda_star_vs_dcs.png` — λ* vs ‖C−S_prev‖ (decile means, Spearman −0.10/−0.06).
6. `experiment_h_lambda_star_vs_gate.png` — λ* vs actual mean gate (Spearman +0.03/+0.09).
7. `experiment_h_replay_vs_scalar.png` — candidate / recorded / vector replay / mean-gate scalar / λ=1 / λ=0.
8. `experiment_h_trajectories.png` — trajectory-class frequencies + example digit trajectories.
9. `experiment_h_category_gap.png` — candidate vs commit vs replay vs counterfactuals by ADD category.

## Sanity Checks (§12) — all pass

* λ=0 reproduces previous-state decoding (1.000); λ=1 reproduces candidate decoding (1.000) and equals candidate correctness (1.000).
* Vector-gate replay reproduces the recorded committed state (max |Δ| = 0.0) and decode (1.000).
* Stored λ* equals recomputed λ* (0.0 max diff); recorded committed decode equals replay decode (1.000).
* Re-collected transitions verified against Experiment G: 0 exact mismatches across 17 columns; float summaries 0.0 max diff; 31,763 rows/model.
* Checkpoint SHA-256 values unchanged before/after runs (runner rechecks and aborts otherwise); no weights modified, no optimizer exists in the pipeline, all inference under `torch.no_grad()`.
* Deterministic rerun: see report footnote after the reproduction run (arrays and figures byte-identical).

---

## Decision (§13)

**C. Both.** Reported separately as required:

1. **Scalar interpolation-path behavior:** monotone old→correct with no interior peak; decodable band is late and narrow (median λ* = 0.80 on failures; only ~1% decodable at λ ≤ 0.5); intermediate blends are *not* reliably decodable. This is a genuine readout-tolerance limitation for partial states.
2. **Actual vector-gate behavior:** the commit is element-wise, not scalar; bit-exact replay; mean-gate scalarization collapses accuracy by 67–81 points. The vector structure is decisive.
3. **Candidate→commit loss:** 14.5% of candidate-correct general adds; 8.3 pts on test-general; up to 12.9 pts on controlled wrap. Same residual for E and F.
4. **Whether a decodable state exists between S_prev and C:** yes, but only near the candidate end (λ ≳ 0.7–0.8 typically). There is no broad decodable intermediate region that a small extra transfer would reach.
5. **Whether the observed actual gate reaches it:** in scalar projection, no (λ_eff ≈ 0.30 « median λ* = 0.80); but this comparison is descriptive only — the actual state is off-path and outperforms its own scalarization, so the residual is a coordinate-wise question, not a scalar-transfer shortfall.

This is **not causal proof**. The two contributing factors — insufficient transfer relative to a late threshold, and low readout tolerance for partial blends — are entangled at the scalar level and must be disentangled at the coordinate level (below).

## Next Step (§14) — one experiment, no fix

**Experiment I — coordinate-wise commit analysis (gate-vector level; measurement only).**
On the same frozen E/F checkpoints, for candidate-correct general ADD transitions, decomposed by committed outcome:
1. **Decision-direction movement.** For the destination register's true-value logit (using the frozen head's linear readout weights), compute the delivered movement `⟨w_true, S_t − S_(t−1)⟩` versus the candidate's available movement `⟨w_true, C_t − S_(t−1)⟩`; measure the delivered fraction on failures vs successes, and the fraction of failures explained by a delivered fraction below the value needed to flip the argmax.
2. **Per-coordinate attribution.** Attribute the delivered movement to gate coordinates (largest contributors `f_i ⊙ (C−S)_i`); test whether failures are characterized by under-movement on the readout-relevant coordinates while total movement (‖S−S_prev‖) is normal.
3. **Local counterfactual (analysis only).** Along the decision direction, compute the minimal movement scale that flips the decode; compare it to the delivered movement — a decision-level analogue of λ* that is faithful to the vector mechanism.

Outcome: determines whether the residual is (a) the gate under-moving the readout-relevant coordinates (a commit/gate issue), or (b) the readout requiring implausibly precise coordinate alignment (a readout/representation issue) — before any architecture or training change is contemplated. No Phase-2 machinery; no fix implementation.

---

## Reproduction

```bash
# 1. Collect interpolation vectors + λ sweep from frozen checkpoints (~15 s)
python experiments/run_experiment_h.py \
    --checkpoints-dir experiments/checkpoints \
    --g-transitions experiments/experiment_g_transitions.npz \
    --output experiments/experiment_h_interp.npz

# 2. Analysis + figures (~7 s)
python experiments/analyze_experiment_h.py \
    --interp experiments/experiment_h_interp.npz \
    --output experiments/experiment_h_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_h

# 3. Tests
pytest tests/test_experiment_h.py -v        # 10 tests
pytest tests/                                # full suite
```

## Scientific Guardrails

This experiment measures decoding along a constructed scalar path and on the actual vector-gate state; it does not establish causation and does not claim the commit mechanism is understood. No training, no representation changes, no Phase-2 machinery. λ* is a path statistic, not a threshold in the model; `S(f̄)` is a counterfactual, not the commit. The element-wise gate was shown to be necessary for accurate description — any future commit-side reasoning must therefore be coordinate-wise. The ADD candidate→commit residual (~8 pts on test-general, ~13 pts on controlled wrap) remains one part of the Phase-1 error budget alongside wrap arithmetic and repeated-ADD chains.
