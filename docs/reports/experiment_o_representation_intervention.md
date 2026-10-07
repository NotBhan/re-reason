# RE:Reason — Experiment O: Controlled Representation-Side Test of Parity Alignment

**Date:** October 7, 2026
**Status:** Completed — Interventional (inference-time state computation only), frozen checkpoints; no training, no optimizer, no weight/readout/gate/commit-rule/benchmark modification; no learned policy; no verifier; no branching; no Phase 2. Classification: **O-D — no specific parity causality** (E primary; the F control agrees).
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned
**Data:** `experiments/experiment_o_representation_intervention.npz`, `experiments/experiment_o_representation_intervention_meta.json`, `experiments/experiment_o_results.json`
**Runners:** `experiments/run_experiment_o.py`, `experiments/analyze_experiment_o.py`, tests `tests/test_experiment_o.py`
**Checkpoints:** `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against G/L/M/N metadata)

---

## 1. Executive Answer

**Removing the W-derived parity component from the candidate representation does *not* specifically reduce the third-basin failure.** On the exact off-path population it rescues a small number of transitions (net **+8.0/seed** for E and **+8.8/seed** for F at κ=1), **but a norm-matched orthogonal control rescues essentially the same number** (E +8.6, F +8.0), so the local rescue is **not parity-specific**. Both parity interventions also cause **broad collateral damage** — they break **24.8% (E) / 22.2% (F)** of baseline-correct successes (versus 7.8% / 6.6% for the matched control) — and **reduce sequence-level exact match** (E T=10 0.7547 → 0.6505, −10.4 pp, 5/5 seeds, dz −2.49, p = 0.005; T=20 0.4068 → 0.3224). The commit-side stage control behaves the same way (non-specific, worse collateral). Classification: **O-D — no specific parity causality.** Parity alignment is therefore a **correlated state/readout diagnostic feature, not a demonstrated causal mechanism** for the frozen inference computation.

The required final questions are answered in §23.

## 2. Existing Baseline (what G–N established, and what O resolves)

* **G/H/I**: a real ADD candidate→commit residual; the element-wise commit is bit-exact and off the scalar path; coordinate-wise under-movement plus a placement component.
* **J/K**: commit-only *magnitude* (γ) and *shape* (κ) interventions move local margin delivery but improve no sequence endpoint; commit-rule modification frozen.
* **L** (`experiment_l_readout_localization.md`): the off-path residual is latent-state **placement** — the committed state clears the intended competitor (100%) but a different competitor wins (100%); fixed bias negligible.
* **M** (`experiment_m_representation_geometry.md`): the third basin is a systematic **parity** region — the wrong value preserves the target's parity 73.7% (E)/64.0% (F) vs a 50% null; wrap and ±1 adjacency rejected.
* **N** (`experiment_n_parity_origin_localization.md`): the parity signature is **state-alignment-dominant** (N-B) — the readout has a small non-dominant parity geometry, and breaking the state↔label pairing collapses the effect to ≈0.50. N left the *locus* of that alignment unresolved between upstream candidate representation and downstream commit placement.
* **O (this report)** makes the controlled causal test of the state-side mechanism and finds **no specific parity causality** (O-D).

## 3. Research Question

> Is the parity-aligned error produced **upstream in the candidate representation** (N-A/O-A) or introduced by the **candidate→commit placement mechanism** (N-B/O-B)? Does removing the W-derived parity component of the candidate — through the original frozen gate/commit — specifically reduce the off-path third-basin failure?

## 4. Hypotheses

* **O-A upstream candidate.** Candidate-side parity removal produces a reproducible off-path improvement that exceeds the norm-matched orthogonal control and is not explained by broad degradation.
* **O-B downstream commit.** The candidate-side intervention does not materially improve the population, but the commit-side parity removal does (and exceeds its matched control).
* **O-C both.** Both stages show specific improvements exceeding their controls.
* **O-D no specific parity causality.** Neither intervention exceeds its matched control, or improvements are dominated by generic degradation, or effects are inconsistent across seeds.

## 5. Pre-registration (declared before inspecting aggregate results)

Recorded in `run_experiment_o.py` / `analyze_experiment_o.py`:

```
p_dest = normalize(mean normalised W[dest, even v] - mean normalised W[dest, odd v])   (from W alone)
deltaC = C - S_prev;  delta_p = <deltaC, p_dest> p_dest
candidate-side  : C' = C - kappa delta_p          (kappa in {0, 0.5, 1.0})
commit-side     : S_commit' = S_commit - kappa <(S_commit - S_prev), p_dest> p_dest
matched control : same magnitude kappa|<delta,p>| along a deterministic unit q orthogonal to p_dest
primary         : candidate-side kappa=1 vs baseline kappa=0 on the off-path population
```

Each seed s (five frozen E checkpoints = independent units) yields `net(s) = rescued - regressed`
on the off-path population. Pre-registered specificity bars: a side is **specific** iff
`mean net > 0` **and** `≥4/5 seeds net > 0` **and** `mean(net - matched-control net) > 0` **and**
`≥4/5 seeds net > matched control` **and** the baseline-success breakage is not worse than the
matched control (+0.02). Classification: O-A if only the candidate side is specific, O-B if only
the commit side, O-C if both, O-D otherwise. The bars are declared conventions, not tuned cutoffs.
Sequence endpoints (T=5/10/15/20 EM) are the confirmatory endpoints; a local rescue without a
sequence improvement is labelled "causally effective locally, sequence-level insufficient".

## 6. Frozen Components

Recurrent core, candidate network, observation encoder, **gate network**, **commit rule**, readout, benchmark, task generator, optimizer, loss, and all Phase-2 machinery. No parameters, weights, readout, gate, or commit rule are modified; the intervention lives entirely in inference-time state computation under `torch.no_grad()`. The modified candidate flows through the **original frozen gate** `f = sigmoid(W_f e + U_f S)` and the **original frozen commit** `S' = (1-f)S + f C'`.

## 7. Intervention Definition

The parity axis is derived **only from the frozen readout** (per destination register), never from the target/wrong value, failure labels, or arithmetic metadata. The candidate-side intervention removes the signed parity component of the candidate movement:

```
C' = C - kappa <C - S_prev, p_dest> p_dest
```

and then commits through the unchanged frozen mechanism. The commit-side stage control applies the identical operation to the committed displacement, `S_commit' = S_commit - kappa <S_commit - S_prev, p_dest> p_dest`, after the original commit. Both are applied **at ADD transitions only**, deterministically and state-locally, at every ADD step of the recursive rollout. The matched orthogonal control removes the same magnitude `kappa|<delta,p>|` along a deterministic unit vector orthogonal to `p_dest` (derived from the current state — no arithmetic labels).

## 8. Leakage Controls

The intervention direction is computable from the **current frozen state, current candidate, destination register, and frozen readout weights only**. It never uses `y_true`, the old value, the operand, the wrap flag, the wrong decoded value, future observations, future states/gates, oracle branches, or failure labels; the test target is used only to evaluate outcomes. The rollout function's signature contains no labels, and the tests verify that changing the labels does not change the rollout. No future-state leakage: truncating the horizon leaves earlier committed states unchanged (tested).

## 9. Dataset / Primary Population

* Canonical strict test split (1,200 × 10) for E and F × seeds 42–46; horizon sets T ∈ {1,2,3,5,8,10,15,20}.
* Primary off-path population (exact M/L/N definition, not redefined): **323 (E) / 353 (F)**. Controls: candidate-correct successes (4,310 / 4,292), candidate-correct non-off-path failures (409 / 387), all test ADD transitions (15,763).
* 9 configs (none κ=0; cand κ∈{0.5,1.0}; orth κ∈{0.5,1.0}; commit κ∈{0.5,1.0}; commit_orth κ∈{0.5,1.0}) × 5 seeds × 2 models; 8 horizon rollouts + canonical rollout per config.

## 10. Baseline Reproduction

**κ=0 reproduces the frozen forward exactly** for all 10 checkpoint-seeds: max state difference 0.0 and identical terminal decodes (`decode_equal = True`); κ=0 is a mathematical and operational no-op for every intervention family (tested). Checkpoint SHA-256 verified before/after and unchanged. Full recursive rollout is deterministic (arrays bit-identical across two runs; analysis JSON identical excluding timestamp).

## 11. Candidate-Side Intervention (E primary)

| Config | off-path net rescue (mean/seed) | off-path committed acc | baseline-success broken rate | all-ADD committed acc |
|---|---:|---:|---:|---:|
| baseline κ=0 | 0.0 | 0.000 | 0.000 | 0.855 |
| cand κ=0.5 | +7.2 | 0.111 | 0.112 | — |
| **cand κ=1.0** | **+8.0** | **0.190** | **0.248** | **0.653** |
| orth κ=0.5 | +6.0 | — | 0.033 | — |
| **orth κ=1.0** (control) | **+8.6** | — | **0.078** | **0.797** |
| commit κ=1.0 | +9.8 | 0.206 | 0.269 | 0.638 |
| commit_orth κ=1.0 | +7.2 | — | 0.177 | — |

Per-seed candidate net rescue (E): cand κ=1 [12, 7, 4, 12, 5]; orthogonal control [11, 9, 8, 8, 7]. **The parity removal rescues *fewer* off-path transitions than the matched orthogonal control on average (8.0 vs 8.6; only 2/5 seeds positive)** and breaks **3× more** baseline successes (24.8% vs 7.8%). The effect scales weakly with κ (7.2 → 8.0).

## 12. Matched Orthogonal Control

The control uses the identical geometric operation with the same displacement magnitude (matched per transition at intervention time; the recorded rollout sums agree to ≈2% because the trajectories diverge after the first intervention). It produces an off-path rescue of the **same order** (E +8.6, F +8.0) as parity removal (E +8.0, F +8.8). **The rescue is therefore not specific to the parity direction**: any candidate perturbation of that magnitude rescues a similar minority of off-path failures.

## 13. Commit-Side Stage Control

| Config | off-path net rescue (mean/seed) | baseline-success broken rate |
|---|---:|---:|
| commit κ=1.0 (E) | +9.8 | 26.9% |
| commit_orth κ=1.0 (E) | +7.2 | 17.7% |
| commit κ=1.0 (F) | +10.0 | 24.9% |
| commit_orth κ=1.0 (F) | +7.8 | 13.1% |

The commit-side mean net exceeds its matched control (+2.6 E, +2.2 F) but **not consistently across seeds (3/5 positive for E)** and with **substantially worse collateral** (26.9% vs 17.7% successes broken), so it fails the pre-registered specificity bars. The commit-side condition is a **stage-localization control**, not a better intervention; the primary causal interpretation comes from the candidate-side condition.

## 14. Transition-Level Rescue/Regression (E primary)

* Off-path (all baseline-wrong): candidate κ=1 rescues **8.0/seed**, regresses **0** (none can regress). Orthogonal control rescues 8.6.
* Baseline successes: candidate κ=1 **breaks 24.8%** (≈214/seed of 862 successes), orthogonal control breaks 7.8%. This is the decisive **collateral** signature — the parity intervention is not selective.
* Non-off-path failures: candidate κ=1 net +? (small; the intervention rescues a few).

Rescue/regression is reported as paired counts per seed; the pre-registered verdict is based on specificity vs the matched control and on collateral, not on a single pooled transition-level p-value.

## 15. Sequence-Level Results (E primary)

| Endpoint | baseline | cand κ=1 | orth κ=1 | commit κ=1 |
|---|---:|---:|---:|---:|
| T=5 EM | 0.9524 | 0.9000 | 0.9410 | 0.8992 |
| **T=10 EM** | **0.7547** | **0.6505** | 0.7228 | 0.6478 |
| T=15 EM | 0.5420 | 0.4442 | 0.5174 | 0.4484 |
| **T=20 EM** | **0.4068** | **0.3224** | 0.3774 | 0.3220 |

Paired seed-level (E, candidate κ=1 vs baseline): T=10 **−10.4 pp** (5/5 seeds down, dz −2.49, p = 0.005); T=20 **−8.4 pp**. Per-register accuracy on all ADD transitions falls 0.925 → 0.889 (cand κ=1). The matched orthogonal control also degrades T=10 (−3.2 pp), confirming that these interventions perturb the representation broadly. **The local off-path rescue does not translate to any sequence-level improvement** — it is "causally effective locally, sequence-level insufficient" (and on net, sequence-level harmful).

Per-register accuracy, Hamming-distance distributions, and operation-level EM all degrade consistently (see §17 and the figures).

## 16. Continuous Geometry Effects

* κ=0 gives exactly zero displacement (`C'−C = 0`, `S_commit'−S_commit = 0`; tested).
* κ=1 gives `C'−C = −<ΔC, p_dest> p_dest` exactly (within float tolerance), and the intervention displacement is linear in κ (κ=0.5 gives half the magnitude).
* The matched orthogonal control removes the same magnitude `κ|<ΔC,p>|` along an orthogonal unit direction.
* Applied magnitudes: candidate-side |ΔC removal| ≈ 4000 summed over the canonical test (E), matched by the control to ≈2%.

## 17. Failure-Mode Analysis

The operation-level behavior (E, final-op EM) degrades under every intervention:

| Operation | baseline | cand κ=1 | commit κ=1 |
|---|---:|---:|---:|
| SET | 0.836 | 0.745 | 0.746 |
| SWAP | 0.778 | 0.685 | 0.688 |
| ADD no-wrap | 0.709 | 0.587 | 0.577 |
| ADD wrap | 0.493 | 0.346 | 0.330 |

The intervention does not selectively remove the M-observed third-basin mechanism: it breaks all operations and shifts errors broadly (the terminal Hamming-distance distribution spreads toward larger distances for cand κ=1 versus baseline). Rescued off-path transitions are outnumbered by broken baseline successes; the net effect on the frozen model is negative.

## 18. E-versus-F

Both models classify **O-D**. F shows the same pattern: candidate net +8.8 (orth +8.0; 2/5 seeds positive), commit net +10.0 (commit-orth +7.8; 4/5 but collateral 24.9% vs 13.1%), sequence T=10 0.737 → 0.6595 (cand κ=1) / 0.6578 (commit κ=1), T=20 0.3898 → 0.3342 / 0.3314. The negative result reproduces across the two training objectives.

## 19. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a secondary control (not pooled). Transition-level counts are descriptive. Rescue/regression is reported as per-seed paired counts; sequence endpoints use paired seed-level comparisons (mean ± SD, Cohen's dz, sign consistency, paired t). Effect sizes and raw per-seed values are reported. κ is not optimized; the family {0, 0.5, 1.0} is pre-registered.

## 20. Causal Interpretation

**Intervention-level causal evidence.** The candidate-side parity removal *does* change the frozen inference computation: it alters the committed states, rescues a small number of off-path transitions, and degrades sequence EM. This is a genuine inference-time causal effect.

**But there is no *specific* parity causality.** The rescue does not exceed a norm-matched orthogonal perturbation; the parity intervention breaks far more baseline successes than its control; and the sequence endpoints get significantly worse. The effect is a **generic consequence of perturbing the candidate representation of this magnitude**, not a parity-direction-specific mechanism. The commit-side stage control shows the same non-specific pattern. Therefore, at the level of this experiment, **parity alignment is a correlated diagnostic feature of the off-path state, not a demonstrated causal mechanism** for the third-basin failure.

**Status.** Interventional evidence, restricted to the frozen model and the measured consequence. No architectural claim; no claim that parity is a designed feature; no claim of exact arithmetic encoding.

## 21. What O Establishes

* A clean, deterministic, leakage-free, weight-frozen **interventional** test: κ=0 reproduces the frozen forward exactly; the parity axis is derived from W alone; the modified candidate flows through the original gate/commit.
* That removing the W-derived parity component of the candidate (κ=1) produces only a **small, non-specific** off-path rescue (E +8.0/seed vs matched orthogonal +8.6/seed; F +8.8 vs +8.0).
* That the parity intervention causes **broad collateral damage** (24.8% of baseline successes broken, ~3× the matched control) and **significantly reduces sequence EM** (E T=10 −10.4 pp, T=20 −8.4 pp, 5/5 seeds; all operations degrade).
* That the commit-side stage control behaves the same way (non-specific; worse collateral), so **neither stage shows a parity-specific causal effect**.
* That the negative result **reproduces across E and F**.

## 22. What O Does NOT Establish

* It does **not** establish causality for the *architecture* (only for the frozen inference computation), and it is not a "fix".
* It does not show that parity is meaningless — only that it is a **correlated diagnostic**, not a demonstrated causal lever for the third-basin failure in this intervention family.
* It does not establish exact arithmetic representation, a designed parity feature, or generalization to NLP/Phase-2/AGI.
* It does not prove that *no* representation-side intervention could help — only that this specific measured parity direction, removed by this specific geometry-matched operation, does not.
* It does not resolve the remaining sequence-level error budget (wrap/repeated-ADD arithmetic and candidate-generation errors upstream).

## 23. Final Decision

**O-D — no specific parity causality** (E primary; F control agrees). Pre-registered components (E): candidate side net **+8.0** but `mean(net − matched control) = −0.6` with only 2/5 seeds positive → **not specific**, collateral 24.8% vs 7.8% → **fails**; commit side net +9.8 over control +7.2 (+2.6) but only 3/5 seeds positive and collateral 26.9% vs 17.7% → **fails**. Neither side is specific. **Parity-directed intervention work stops; parity is documented as a correlated state/readout feature.**

**Answers to the required final questions.**

1. **Does removing the W-derived parity component causally reduce the third-basin failure?** Locally it rescues ~8/seed, but **not specifically** — it does not exceed a matched orthogonal perturbation, so no *specific* causal reduction.
2. **Is any improvement specific to the parity direction rather than generic perturbation?** **No** — the matched orthogonal control rescues essentially the same (E 8.6 vs 8.0; F 8.0 vs 8.8).
3. **Does the effect occur upstream at the candidate representation?** The intervention acts on the candidate and changes outcomes, but the effect is not parity-specific, so it does not establish a specific upstream candidate parity mechanism.
4. **Does the same geometric intervention work only after commit?** **No** — the commit-side condition is likewise non-specific (fails the same bars).
5. **Which stage has the larger causal effect?** **Neither** (both fail the specificity/collateral bars); descriptively the commit side exceeds its control slightly more but inconsistently and with worse collateral.
6. **Does the effect survive full recursive rollout?** The intervention changes recursive states deterministically, but sequence endpoints **degrade**.
7. **Does local transition rescue translate to T=10/T=20 improvement?** **No** — E T=10 −10.4 pp, T=20 −8.4 pp (5/5 seeds, large effects).
8. **Does it preserve SET/SWAP?** **No** — SET −9.1 pp, SWAP −9.3 pp.
9. **Does it preserve wrap/repeated-ADD?** **No** — ADD no-wrap −12.2 pp, ADD wrap −14.7 pp.
10. **Does the effect reproduce across E and F?** **Yes** — both O-D, same pattern.
11. **Is there enough evidence to treat parity alignment as a genuine causal mechanism?** **No.** Parity alignment is a correlated diagnostic feature, not a demonstrated causal mechanism for the frozen inference computation in this intervention family.

## 24. Exactly One Next Scientific Experiment

**Experiment P — sequence-level error-budget attribution across the frozen rollout (measurement-only).** O joins J and K in showing that local commit-side and representation-side perturbations do not improve the sequence endpoint, so the remaining question is *where the sequence-level error budget actually lives*. Experiment P attributes the frozen E model's T=10/T=20 exact-match failures, per seed, to (i) **candidate-generation errors** (candidate decodes the written register wrong — upstream of the commit), (ii) the **candidate→commit residual**, and (iii) **readout/decoding** — decomposed along the recursive trajectory, with the same population definitions used throughout G–O. The purpose is to identify the dominant *recoverable* target now that commit-side (J/K) and representation-side parity (O) interventions have been ruled out; it is measurement-only, and it decides whether any further intervention on the frozen model is warranted or whether the remaining ceiling is a genuine representation limit. Per the research standard, O stops here: it **determines the origin** (no specific parity causality) and names the single follow-up; it does not implement an intervention or begin Phase 2.

## Reproduction

```bash
# 1. Run the full interventional grid (9 configs x 5 seeds x 2 models; ~3 min)
python experiments/run_experiment_o.py \
    --output experiments/experiment_o_representation_intervention.npz

# 2. Analysis + 12 figures (~10 s)
python experiments/analyze_experiment_o.py \
    --npz experiments/experiment_o_representation_intervention.npz \
    --output experiments/experiment_o_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_o

# 3. Tests
pytest tests/test_experiment_o.py -v      # 16 tests
pytest tests/                             # full suite
```

Determinism: the collector was run twice (arrays bit-identical, max |Δ| = 0.0); the analysis twice (JSON identical excluding the timestamp). Checkpoint SHA-256 verified before/after; κ=0 reproduces the frozen forward exactly.

## Artifact List

* `experiments/run_experiment_o.py` — interventional runner (candidate/commit/orthogonal families, recursive rollout, transitions, horizons, geometry).
* `experiments/analyze_experiment_o.py` — rescue/regression, specificity vs matched control, sequence endpoints, geometry, failure modes, E-vs-F, pre-registered O-A/B/C/D decision, 12 figures.
* `tests/test_experiment_o.py` — 16 tests (κ=0 reproduction, determinism, no-grad/no-mutation, target/future leakage, parity-axis construction, projection/removal correctness, matched control, stage separation, population groups, accounting, sequence reproduction, E/F alignment, JSON determinism, figures, malformed input, historical immutability, checkpoint hashes).
* `experiments/experiment_o_representation_intervention.npz` — per-config transition committed/candidate arrays, sequence EM, group masks, per-seed readout inputs.
* `experiments/experiment_o_representation_intervention_meta.json` — config, checkpoint hashes, reproduction checks, geometry, horizons, per-config seq metrics.
* `experiments/experiment_o_results.json` — specificity, sequence, groups, geometry, failure modes, decision.
* `docs/reports/figures/experiment_o_*.png` — 12 figures: `rescue_regression`, `cand_vs_orth`, `commit_vs_orth`, `sequence_em`, `t10_t20`, `group_accuracy`, `collateral`, `geometry`, `by_op`, `hamming`, `kappa_monotone`, `E_vs_F`.
* Checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`.
* Historical references (read only, unchanged): `experiments/experiment_n_parity_origin.npz`, `experiments/experiment_m_geometry.npz`, `experiments/experiment_l_readout.npz`, `experiments/experiment_i_coordinate.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_g_transitions.npz`.

## Compute and Safety

The intervention adds only a projection onto `p_dest` and a subtraction per ADD step (plus the frozen gate/commit), and no search, branching, verifier, extra model passes, or learned optimization; all conditions execute identical rollout logic. No parameters changed, no gradients, no optimizer, no `model.train()`, no checkpoint overwrite (tested). Full run ≈ 180 s (CPU).

## Scientific Guardrails

Interventional evidence, restricted to the frozen model and the measured consequence. No oracle target information, no failure labels, no future information enters the intervention. Parity is not claimed to be a designed feature; no exact arithmetic-encoding claim; no architectural proof; the intervention is not called a "fix". A local causal effect is not called an architectural proof, and a negative specificity result is preserved. Experiments G/H/I/J/K/L/M/N reports and artifacts are untouched.
