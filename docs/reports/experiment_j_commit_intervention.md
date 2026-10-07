# RE:Reason — Experiment J: Controlled Inference-Time Commit-Rule Intervention

**Date:** October 6, 2026
**Status:** Completed — Interventional / causal test; frozen checkpoints; no training. Classification: **B (partial causal support)** for both E (primary) and F (control).
**Frozen models:** `E_candidate` (official Phase-1 objective, primary) and `F_combined` (control), seeds 42–46, 226,358 parameters, SHA-256-pinned
**Data:** `experiments/experiment_j_records.json`, `experiments/experiment_j_results.json`
**Runners:** `experiments/run_experiment_j.py`, `experiments/analyze_experiment_j.py`, tests `tests/test_experiment_j.py`
**Checkpoints:** `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against Experiment G metadata)

---

## 1. Executive Answer

**Intervening only on the element-wise commit gate is causally effective at the local level and causally insufficient at the sequence level. Increasing gate transfer (`γ > 1`) monotonically increases decision-margin delivery and rescues some baseline candidate→commit failures, but it never improves T=10 or T=20 terminal EM; beyond `γ = 1.10` it significantly degrades sequence EM, worsens wrap/repeated-ADD, and breaks more baseline-correct sequences than it rescues. Reducing transfer (`γ < 1`) is strongly harmful. A mean-matched coordinate-scrambled gate collapses EM to ≈0.01, so the coordinate identity of gate delivery is functionally essential. Classification: B (partial causal support) — the commit/gate is a real local bottleneck (under-movement, Experiment I) but it is not the dominant long-horizon bottleneck, and untuned global amplification is not an adequate intervention.**

The intervention is interventional with respect to the commit rule (frozen checkpoint, frozen inputs, only the commit combination changed), but it does **not** establish that the modified rule is the correct architecture or that the underlying representation is understood.

1. **Baseline parity is exact.** `γ = 1` reproduces the frozen forward (decoded states and terminal predictions bit-identical; max float difference < 1e-5, pure float32 round-off) and reproduces the established headline metrics: E T=10 EM **0.7547 ± 0.0191** (report `0.755 ± 0.019`), E T=20 **0.4068**; F T=10 **0.7370 ± 0.0149** (report `0.737 ± 0.015`), F T=20 **0.3898**. At `γ = 1` the rescue/regression counts are exactly 0, and re-running the whole grid is bit-identical.
2. **Sequence-level: no benefit, then harm.** E T=10/T=20 EM (mean over 5 seeds): `γ = 1.10` 0.7525 / 0.4064 (Δ = −0.22 pp / −0.04 pp); `γ = 1.25` 0.7463 / 0.3924 (−0.83 / −1.44 pp); `γ = 1.50` 0.7380 / 0.3724; `γ = 2.00` 0.7107 / 0.3350. No `γ > 1` improves the sequence endpoint; `γ ≥ 1.25` degrades it with sign-consistent, conventionally large paired effects. `γ < 1` (transfer reduction) is strongly harmful (E T=10: 0.7240 at 0.90, 0.5860 at 0.75).
3. **Local decision-margin delivery moves exactly as hypothesized.** On the fixed baseline candidate-correct general-ADD set, raising `γ` monotonically raises the delivered margin fraction and the contribution-weighted gate on decision-helping coordinates: E ρ_margin 0.884 → 0.903 → 0.912 → 0.920 → 0.930, gw_pos 0.690 → 0.708 → 0.719 → 0.730 → 0.743 for `γ = 1.0 → 2.0` (paired vs baseline dz +15 to +59, 5/5 seeds, p < 1e-4). Candidate→commit committed accuracy on that set rises slightly at `γ = 1.10` (0.855 → 0.860).
4. **Net transition rescue exists but is small and non-monotone in usefulness.** E rescued/regressed counts per seed: `γ = 1.10` 12.4 / 7.4 (net **+5.0**); `γ = 1.25` 18.2 / 17.2 (net +1.0); `γ = 1.50` 24.8 / 28.8 (net −4.0); `γ = 2.00` 34.0 / 45.6 (net −11.6). The intervention recovers a minority of baseline candidate-correct failures but breaks a comparable or larger number of baseline candidate-correct successes.
5. **Sequence-level rescue is outweighed by collateral damage.** E baseline-failure sequences rescued / baseline-success sequences broken: `γ = 1.10` 7.8 / 10.4; `γ = 1.25` 12.2 / 22.2; `γ = 2.00` 19.8 / 72.6.
6. **Coordinate identity is essential (mean-matched scramble control).** Permuting the gate coordinates and rescaling to the *same* mean gate (matched within ≈0.005) collapses E T=10 EM from 0.586–0.755 (coordinate-preserving) to **0.008–0.013** at every `γ`, including `γ = 1.0`. Aggregate transfer is not the mechanism; the specific coordinate routing is.
7. **The off-path component is not resolved by amplification.** Of the baseline candidate-correct failures that already delivered ≥ the scalar-equivalent transfer needed (`ρ_margin ≥ α*_full`; n = 65/seed), only **6.4%** are rescued at `γ = 1.10` and **22.6%** even at `γ = 2.0`; the majority remain failures under every `γ`. Consistent with Experiment I's placement/readout component.
8. **Operation-level regressions appear for `γ ≥ 1.25`.** Final-op EM: ADD-wrap 0.493 → 0.475 (γ=1.25) → 0.433 (γ=2.0); controlled repeated-ADD 0.148 → 0.132 → 0.117; SET and SWAP regress too. `γ = 1.10` leaves SET/SWAP/ADD-no-wrap essentially unchanged (|Δ| ≤ 0.6 pp).
9. **Secondary monotone family agrees.** The pre-registered `f' = 1 − (1−f)^γ` family behaves like the primary one (E T=10 0.7547 → 0.7533 → 0.7503 → 0.7315 for γ = 1.0 → 1.1 → 1.25 → 2.0): margin delivery up, no sequence benefit.
10. **E and F agree.** Both classify B; F's transition rescues are slightly larger (`γ = 1.10` net +10.0) and its sequence EM is equally flat-to-worse. The commit intervention's effect is not specific to the official E objective.

## 2. Research Question

> If we intervene **only** on the actual element-wise commit gate `f_t ∈ R^212` — leaving architecture, weights, optimizer, readout, data, and task generator frozen — can we recover sequence-level capability, or does the Experiment I under-movement residual prove to be a local symptom rather than the sufficient cause of long-horizon failure?

## 3. Existing Baseline (what G/H/I established)

* **Experiment G** (`experiment_g_add_commit_gap.md`): the ADD candidate→commit gap affects ~5.5% of candidate-correct adds, concentrated in general two-register adds; evidence for gate dilution and blend/readout both present (C).
* **Experiment H** (`experiment_h_lambda_interpolation.md`): the element-wise vector-gate commit is bit-exact and must be reasoned about coordinate-wise; scalar mean-gate reasoning is inadequate; the committed state lies off the scalar path.
* **Experiment I** (`experiment_i_coordinate_commit.md`): coordinate-wise decision-margin analysis. Failures deliver less decision-margin movement (0.78–0.80 vs 0.90), concentrated on decision-helping coordinates, and start from weaker candidate margins; but 44–48% of failures had sufficient scalar-equivalent transfer, indicating a placement/readout component. Verdict C (both); next experiment = this controlled intervention.

## 4. Hypothesis and Pre-Registration

**Hypothesis (J).** If the ADD candidate→commit residual is causally gate-limited, then increasing the coordinate-wise commit transfer `f_i` should increase delivered decision margin, reduce candidate-correct/commit-wrong transitions, and — if the commit bottleneck dominates long-horizon error — improve sequence-level EM.

**Pre-registered before any result was inspected** (fixed in `run_experiment_j.py`; stored in the records metadata):

* Primary family: `f'_i = clip(γ f_i, 0, 1)` on the **full 212-dim gate vector**, grid `γ ∈ {0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00}`, `γ = 1.00` the identity baseline. The grid is closed: no γ added or removed after seeing outcomes.
* One control: coordinate-scrambled gate `permute(f)`, deterministic per (base_seed, seed), rescaled to match the coordinate-preserving scaled gate's **mean per row**.
* One optional secondary monotone family: `f'_i = 1 − (1 − f_i)^γ`.
* Complete **recursive** rollout: `S'_t = (1−f'_t)⊙S'_{t-1} + f'_t⊙C_t`, with `f_t` and `C_t` recomputed from `S'_{t-1}` exactly as the frozen core does (candidate generation tracks the modified trajectory).
* Decision rule: A requires a beneficial sign-consistent (≥4/5 seeds) large (|Cohen's dz| ≥ 0.8, p < 0.05) **sequence-level** effect at T=10/T=20, with net candidate→commit rescue and no large operation regression; B = the local mechanism moves causally but there is no beneficial sequence-level effect (or mixed); D = harmful sequence effect with neither margin nor rescue benefit; C = neither. No pp threshold is tuned on the data.
* No readout/decision-margin/label information enters the intervention; those quantities are used only in analysis.

## 5. What Is Frozen

Recurrent core, candidate network, observation encoder, gate network, readout, benchmark, task generator, optimizer, loss, model dimensions, all Phase-2 machinery. No retraining, no weight modification, no added trainable parameters, no optimizer; all inference under `torch.no_grad()`. Only the commit combination `(1−f)S + fC → (1−f')S + f'C` is changed, and only `f` is transformed, never `S` directly or the readout.

## 6. What Changed

New code and artifacts only (`run_experiment_j.py`, `analyze_experiment_j.py`, `tests/test_experiment_j.py`, the records/results JSON, figures, this report). No completed report was modified. One test-infrastructure fix: `tests/test_experiment_i.py` contained a flaky same-second timestamp comparison in its analysis-determinism assertion (metadata embeds a wall-clock `date`); the comparison now excludes the timestamp. This is a test-only change and does not alter any Experiment I result or artifact.

## 7. Dataset, Protocol, and Evaluation

* **Data.** For each of E and F × seeds {42..46}: the canonical strict test split (1,200 sequences × 10 steps, never trained on) and the horizon sets `horizon_test_set(seed, L)` for `L ∈ {1,2,3,5,8,10,15,20}` (identical to the generator-repair protocol); plus the five existing controlled ADD builders. All γ and all seeds use the **same** sequence sets (paired comparison).
* **Grid.** E: 3 families × 7 γ = 21 configs/seed; F: primary family only = 7 configs/seed. 140 config-runs total; each is a full recursive rollout.
* **Endpoints.** Primary: T=10 and T=20 terminal EM (sequence-level). Secondary: T=5, T=15, T=1 ghost errors, per-register accuracy, transition written-register accuracy, candidate accuracy, committed accuracy, strict first-error rate, operation-level EM (SET/SWAP/ADD-no-wrap/ADD-wrap), controlled builders (isolated / ADD→SET / 4× repeated / no-wrap / wrap), candidate→commit gap, decision-margin delivery, and candidate-margin strata.
* **Rescue accounting.** Baseline candidate-correct general-ADD transitions that are committed-wrong are **rescued** if the intervention commits correctly; baseline candidate-correct committed-correct transitions become **regressions** if the intervention commits wrong. Sequence-level: baseline-failure sequences rescued / baseline-success sequences broken.
* **Strata.** Baseline candidate-margin terciles (pre-declared, computed from the γ=1 distribution), so bin choice is not selected after observing γ responses.

## 8. Experimental Procedure

1. Load frozen checkpoint; verify SHA-256 against Experiment G metadata.
2. `γ = 1` rollout on the canonical test set; verify it reproduces `forward_with_states` (decoded states and terminal predictions identical, max float diff < 1e-5).
3. For every (family, γ): recursive rollout; horizon EM; transition-level decoding; per-ADD decision margins; controlled builders; rescue/regression vs the baseline arrays.
4. Re-run the entire grid a second time; require deterministic equality of every record.

## 9. Results — Baseline Reproduction and Determinism

* γ=1 reproduces the frozen forward: `decode_equal = True`, `terminal_equal = True`, `max_state_diff ≤ 1.1e-6` (float32 round-off) for every checkpoint.
* γ=1 metrics equal the frozen model's: E T=10 0.7547 ± 0.0191, T=20 0.4068; F T=10 0.7370 ± 0.0149, T=20 0.3898 — matching the official E/F headline values.
* Second full run: meta identical (excluding timestamp and wall clock) and **0/140** record mismatches.

## 10. Results — Primary Family (`scale`, coordinate-preserving), E

| γ | T=5 | T=10 | T=15 | T=20 | ρ_margin | gw_pos | primary comm acc | rescued / regressed (net) | seq rescued / broken |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.908 | 0.5860 | 0.352 | 0.2462 | 0.685 | 0.522 | 0.627 | 9.6 / 240.0 (−230.4) | 13.8 / 216.2 |
| 0.90 | 0.941 | 0.7240 | 0.510 | 0.3772 | 0.803 | 0.626 | 0.799 | 10.8 / 67.8 (−57.0) | 13.0 / 49.8 |
| 1.00 | 0.952 | **0.7547** | 0.542 | **0.4068** | 0.884 | 0.690 | 0.855 | 0 / 0 (0) | 0 / 0 |
| 1.10 | 0.951 | 0.7525 | 0.537 | 0.4064 | 0.903 | 0.708 | 0.860 | 12.4 / 7.4 (**+5.0**) | 7.8 / 10.4 |
| 1.25 | 0.950 | 0.7463 | 0.527 | 0.3924 | 0.912 | 0.719 | 0.856 | 18.2 / 17.2 (+1.0) | 12.2 / 22.2 |
| 1.50 | 0.947 | 0.7380 | 0.512 | 0.3724 | 0.920 | 0.730 | 0.851 | 24.8 / 28.8 (−4.0) | 16.8 / 36.8 |
| 2.00 | 0.941 | 0.7107 | 0.479 | 0.3350 | 0.930 | 0.743 | 0.844 | 34.0 / 45.6 (−11.6) | 19.8 / 72.6 |

Reading: margin delivery rises monotonically with γ, the local commit residual shrinks slightly at γ=1.1, but no sequence endpoint improves and every endpoint worsens for γ ≥ 1.25. Rescue counts grow but regressions grow faster.

## 11. Results — Operation-Level and ADD Strata (E, final-op EM)

| γ | SET | SWAP | ADD no-wrap | ADD wrap | controlled repeated ADD | controlled wrap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.712 | 0.578 | 0.542 | 0.272 | 0.077 | 0.077 |
| 0.90 | 0.818 | 0.744 | 0.675 | 0.435 | 0.137 | 0.261 |
| 1.00 | 0.836 | 0.778 | 0.709 | 0.493 | 0.148 | 0.355 |
| 1.10 | 0.835 | 0.778 | 0.704 | 0.490 | 0.151 | 0.347 |
| 1.25 | 0.828 | 0.776 | 0.697 | 0.475 | 0.152 | 0.328 |
| 1.50 | 0.819 | 0.765 | 0.688 | 0.480 | 0.144 | 0.291 |
| 2.00 | 0.789 | 0.741 | 0.667 | 0.433 | 0.117 | 0.203 |

γ ≥ 1.25 produces sign-consistent, large regressions in SET, SWAP and ADD-no-wrap; wrap and repeated-ADD degrade. `γ = 1.10` is essentially neutral for every operation. The intervention is therefore a system-level inference rule evaluated on all operations, and it does not improve ADD at the expense of SET/SWAP — it fails to improve anything at the sequence level.

## 12. Results — Control and Secondary Family (E)

* **Scrambled control (mean-matched).** T=10 EM at matched mean gate: coordinate-preserving 0.586–0.755 vs scrambled **0.008–0.013** across all γ (including γ=1.0, where scrambling is not an identity). Mean gate matches within ≈0.005. → coordinate identity of gate delivery is functionally essential.
* **Secondary monotone family `1 − (1−f)^γ`.** T=10 EM 0.7502 / 0.7550 / 0.7547 / 0.7533 / 0.7503 / 0.7445 / 0.7315 for γ = 0.75 … 2.0: same qualitative behaviour (no sequence benefit; mild decline for γ > 1).

## 13. Results — Off-Path Subset and Candidate-Margin Strata (E)

* **Off-path baseline failures** (baseline candidate-correct general ADDs with `ρ_margin ≥ α*_full`; n = 65/seed): rescued at 6.4% (γ=1.10), 11.3% (1.25), 15.6% (1.50), 22.6% (2.00). The majority remain failures under every γ — gate amplification alone does not resolve the placement component.
* **Candidate-margin strata** (baseline terciles): rescue rate rises with γ in every stratum but most steeply in the high-margin stratum (γ=2.0: low 0.241, mid 0.184, high 0.400), while regressions also rise. Scaling does not selectively rescue near-boundary candidates; it rescues some candidates at all margins and harms others.

## 14. Statistical Analysis

* Independent units are the 5 frozen E checkpoints (F as control). Transition-level rescue/regression counts are descriptive; all primary inference is **seed-level paired vs γ=1** (mean ± SD over seeds, Cohen's dz, paired t, sign k/5).
* E `γ = 1.10` vs baseline: T=10 −0.22 pp (dz −1.25, 5/5 down, p = 0.049); T=20 −0.04 pp (dz −0.11, 2/5, p = 0.82); ρ_margin +0.019 (dz +15.2, 5/5 up, p < 1e-4); gw_pos +0.018 (dz +42.6, p < 1e-4); primary committed acc +0.50 pp (dz +1.33, p = 0.041).
* E `γ = 1.25`: T=10 −0.83 pp (dz −2.72, p = 0.0037); T=20 −1.44 pp (dz −3.46, p = 0.0015). `γ = 1.50`: T=10 −1.67 pp (p = 0.0004), T=20 −3.44 pp (p = 0.0004). `γ = 2.00`: T=10 −4.40 pp, T=20 −7.18 pp (both p < 1e-4).
* No `γ > 1` yields a beneficial sign-consistent large sequence-level effect. All margin-delivery effects are sign-consistent and large in the direction hypothesized.
* F (control) reproduces the same pattern; verdict B for both.

## 15. Interpretation — Supported Causal Reading

* **Causal claim supported at the local level.** Changing only the commit rule changes decision-margin delivery monotonically and in the predicted direction, changes the candidate→commit failure rate (net rescue positive at γ=1.10/1.25), and changes the sequence endpoint (harmfully for large γ). The commit rule is therefore causally connected to both the local residual and the sequence outcome — it is not epiphenomenal.
* **But the commit/gate is not the dominant long-horizon bottleneck.** Increasing transferred margin improves the local residual while sequence EM stays flat or falls; the rescued transitions are outnumbered by newly broken ones as γ grows; wrap and repeated-ADD worsen. This is the pre-registered outcome "gate scaling improves margins but sequence EM unchanged → the commit affects a real local bottleneck, but it is not the dominant long-horizon bottleneck."
* **The element-wise gate performs a balancing function.** Both directions hurt: decreasing transfer is strongly harmful, and increasing it beyond a small window is harmful. The learned per-coordinate gate values are not simply "too small"; they balance useful and harmful coordinate movement, so global monotone amplification is not an adequate intervention.
* **Coordinate identity is essential** (scrambled control), so any commit-side change must respect the learned per-coordinate routing, not just aggregate transfer.
* **Remaining uncertainty.** This experiment tests one monotone family (plus one secondary family) and one frozen-objective pair; it does not prove no commit rule could help. It does not identify why amplification breaks baseline-correct sequences (hypotheses: off-path placement, over-movement into competing basins, loss of the gate's balancing role). It cannot separate gate routing from readout tolerance because the readout was deliberately left frozen and out of the intervention.

This experiment is **interventional with respect to the commit rule**, but it does not establish that the modified rule is the correct architecture or that the underlying representation is understood.

## 16. What J Establishes

* A clean, reproducible, compute-parity-audited causal intervention on the full 212-dim commit gate (γ=1 reproduces baseline bit-identically at the decode level).
* That commit-margin delivery is a monotone, directly manipulable function of the gate scale, with the Experiment I residual responding in the predicted direction.
* That net candidate→commit rescue is positive only in a narrow window (γ ≈ 1.10–1.25) and small, while sequence-level capability does not improve and degrades for γ ≥ 1.25.
* That the coordinate identity of gate delivery is functionally essential (scrambled control).
* That the majority of "sufficient scalar-equivalent transfer" (off-path) failures survive all γ.

## 17. What J Does NOT Establish

* It does not show gate amplification is a **fix**; it recovers a minority (~6–10%) of baseline candidate-correct failures and improves no sequence endpoint.
* It does not show the readout is uninvolved, nor that the commit mechanism is solved; the off-path component persists.
* It does not claim the gate is a symbolic memory mechanism, that coordinates store registers, or that the architecture is sufficient.
* It does not generalize beyond the tested monotone families, the register-tracking benchmark, or the E/F objectives.
* Improved transition-level rescue does **not** imply improved sequence reasoning; the two are explicitly separated here.

## 18. Decision

**B — Partial causal support (E primary; F control agrees).** The commit/gate is causally involved in the local ADD candidate→commit residual (margin delivery and rescue respond as hypothesized to a commit-only intervention), but the intervention does not improve and (for γ ≥ 1.25) degrades sequence-level EM, with comparable collateral breakage; γ < 1 is harmful; coordinate scrambling is catastrophic. The residual is a real local bottleneck but not the dominant long-horizon one, and untuned global gate amplification is not an adequate commit intervention.

## Final Decision (required answers)

1. **Does increasing gate transfer improve T10/T20 EM?** No. E T=10/T=20 are flat at best (γ=1.10: −0.22 pp / −0.04 pp) and decline for γ ≥ 1.25 (γ=2.0: −4.40 pp / −7.18 pp). No beneficial sequence-level effect at any γ.
2. **Does it reduce the candidate→commit failure rate?** Yes, mildly and in a narrow window: net transition rescue +5.0/seed (E) and +10.0/seed (F) at γ=1.10, +1.0 (E) at γ=1.25; negative beyond.
3. **Does it increase decision-margin delivery?** Yes, strongly and monotonically (ρ_margin 0.884 → 0.930; gw_pos 0.690 → 0.743), sign-consistent 5/5, p < 1e-4.
4. **Does it preferentially rescue low-margin candidate states?** No. Rescue occurs at all baseline candidate-margin strata; at large γ the high-margin stratum is rescued most (0.40 at γ=2.0) and regressions rise everywhere.
5. **Does it reduce ADD-wrap / repeated-ADD errors?** No — both worsen for γ ≥ 1.25 (ADD-wrap 0.493 → 0.433; repeated-ADD 0.148 → 0.117 at γ=2.0); γ=1.10 is neutral.
6. **What regressions does it introduce?** SET, SWAP and ADD-no-wrap regress for γ ≥ 1.25 (sign-consistent, large); baseline-correct sequences are broken (up to 72.6/seed at γ=2.0); regressions exceed rescues beyond γ≈1.1.
7. **Does coordinate identity matter relative to the scrambled control?** Yes, decisively: mean-matched scrambling collapses T=10 EM to ≈0.01 versus 0.586–0.755 coordinate-preserving.
8. **Is there causal evidence for the commit/gate hypothesis?** Yes at the local level (the residual responds to a commit-only intervention as hypothesized); no at the sequence level (capability does not improve).
9. **Is the evidence A / B / C / D?** **B — partial causal support** (E primary; F control agrees).
10. **Exactly one next scientific experiment:** **Experiment K — a controlled, paired "commit-rule vs gate-balance" ablation on the frozen E checkpoints that intervenes on the *balance* rather than the magnitude**: a coordinate-wise rule that leaves coordinates the gate already opens (or closes) near saturation unchanged while adjusting only the mid-range coordinates (e.g. a per-coordinate sigmoid-sharpening / monotone remap of `f_i` about 0.5 that preserves the sign of the deviation from 0.5 and is applied at inference only), evaluated with the same rescue/regression and sequence-level protocol. This directly tests whether a *routing-preserving* commit change can help where uniform magnitude scaling could not, and whether the gating mechanism's balancing role is the reason uniform amplification fails. It must be pre-registered, compute-parity-audited, and must not be treated as a fix.

## Reproduction

```bash
# 1. Run the full pre-registered grid (E: 3 families x 7 gamma; F: 1 family x 7 gamma; 5 seeds) (~2 min)
python experiments/run_experiment_j.py --out experiments/experiment_j_records.json

# 2. Analysis + figures (~5 s)
python experiments/analyze_experiment_j.py \
    --records experiments/experiment_j_records.json \
    --output experiments/experiment_j_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_j

# 3. Tests
pytest tests/test_experiment_j.py -v      # 15 tests
pytest tests/                             # full suite
```

Determinism: the grid was run twice; every one of the 140 records is identical and the metadata is identical after excluding the timestamp and wall-clock fields.

## Artifact List

* `experiments/run_experiment_j.py` — pre-registered intervention runner (rollout, horizon/transition/margin/controlled evaluation, rescue accounting).
* `experiments/analyze_experiment_j.py` — paired seed-level analysis, rescue/regression, strata, scrambled control comparison, A/B/C/D decision, 12 figures.
* `tests/test_experiment_j.py` — 15 tests (identity, clipping, [0,1] bound, checkpoint hashes, baseline reproduction, determinism, no-grad, no param change, rescue/margin/sequence accounting, paired alignment, permutation determinism + mean match, synthetic end-to-end + figures, invalid config).
* `experiments/experiment_j_records.json` — 140 per-(model, seed, family, γ) records + pre-registration metadata.
* `experiments/experiment_j_results.json` — summaries, paired statistics, decision.
* `docs/reports/figures/experiment_j_*.png` — 12 figures: `em_vs_gamma`, `paired_seed_changes`, `rescue_regression`, `commit_gap`, `margin_vs_gamma`, `weighted_gate_vs_gamma`, `margin_strata`, `add_strata`, `op_regression`, `rescue_examples`, `coord_vs_scramble`, `compute_parity`.
* Checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`.

## Compute / Parity

Identical parameter count (226,358), **zero additional trainable parameters**, no optimizer, no training, inference only. The intervention adds one multiply and one clamp per state coordinate per step (~424 elementwise ops/step) against ≈206,976 MACs/step — about **0.20%** of per-step operations. Total wall clock for the 140-config grid ≈ 118 s (CPU).

## Scientific Guardrails

This is an interventional test of the commit rule only; it is not a production fix and does not establish that the modified rule is correct or that the representation is understood. `γ > 1` is not called a fix; the result is stated as "the intervention recovered a minority of baseline candidate-correct failures." No claim is made that the gate is a symbolic memory mechanism, that coordinates represent registers, that the architecture is sufficient for general reasoning, or that the readout is uninvolved. Transition-level rescue is explicitly separated from sequence-level capability; candidate generation, commit, readout, and long-horizon error accumulation are kept as distinct mechanisms. The scrambled control is a control, not a candidate mechanism. Negative results are preserved; Experiments G/H/I reports are untouched.
