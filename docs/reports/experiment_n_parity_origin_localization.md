# RE:Reason — Experiment N: Parity-Origin Localization

**Date:** October 7, 2026
**Status:** Completed — Measurement only (frozen checkpoints; no training; no optimizer; no weight/readout/gate/commit modification; no intervention; no recursive counterfactual rollout; no learned classifier as primary evidence). Classification: **N-B — state-dominant parity alignment** (E primary; the F control agrees).
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned
**Data:** `experiments/experiment_n_parity_origin.npz`, `experiments/experiment_n_parity_origin_meta.json`, `experiments/experiment_n_results.json`
**Runners:** `experiments/run_experiment_n.py`, `experiments/analyze_experiment_n.py`, tests `tests/test_experiment_n.py`
**Checkpoints:** `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against G/L/M metadata)

---

## 1. Executive Answer

**The same-parity signature M found lives predominantly in the state↔label alignment, not in a state-independent preference of the frozen readout.** The frozen 40-way head **does** contain a statistically significant — but small — parity organization in its weight geometry (within-register same-parity value vectors are more similar: mean pairwise cosine −0.0417 same vs −0.1518 different parity; effect **+0.110 (E) / +0.111 (F)**, significant in **4/5** seeds, permutation p ≈ 0.005–0.011). However, that geometry produces almost no **state-independent** same-parity preference: a pairing-break null that holds the frozen readout and the observed `S_commit` distribution fixed but randomizes the state↔label pairing gives a same-parity rate of **0.521 (E) / 0.504 (F)** versus a chance rate of 0.50, while the observed rate is **0.737 / 0.640**. The **gap (0.216 / 0.137)** is the alignment contribution. Quantitatively, the readout's marginal contribution is only **8.9% (E) / 2.5% (F)** of the excess over chance; the state↔label alignment accounts for **91.1% / 97.5%**.

Classification: **N-B — state-dominant parity alignment** (E primary; F control agrees). The readout's parity geometry is the necessary *substrate* (without same-parity weight vectors near `w_y`, proximity to the true class would not favor same-parity values), but the *driver* of the observed signature is where the committed state sits relative to the true class — i.e. the state side.

The required final questions are answered in §23.

## 2. Existing Baseline (what G–M established, and what N resolves)

* **G/H/I**: a real ADD candidate→commit residual; the element-wise vector-gate commit is bit-exact and off the scalar path; coordinate-wise under-movement plus a placement component.
* **J/K**: commit-only *magnitude* and *shape* interventions move local delivery but no sequence endpoint; the commit rule is frozen.
* **L** (`experiment_l_readout_localization.md`): the off-path residual is **latent-state placement** — the committed state clears the intended competitor (100%) but a different competitor wins (100%); the fixed readout bias is negligible.
* **M** (`experiment_m_representation_geometry.md`): the third basin is a **systematic modular (parity)** region — the wrong value preserves the target's parity in 73.7% (E)/64.0% (F) vs a 50% permutation null — with a seed-consistent but weak off-axis direction; wrap/carry and ±1 adjacency rejected; the parity structure's **origin** was left unresolved.
* **N (this report)** resolves the origin: the parity signature is **state-alignment-dominant**, with the readout weight geometry serving as a small, non-dominant substrate.

## 3. Research Question

> Does the same-parity signature live in the frozen readout's **value-geometry** (N-A), in the committed **state↔label alignment** (N-B), in **both** (N-C), or is it **not localizable** from the frozen measurements (N-D)?

## 4. Hypotheses

* **N-A readout value-geometry.** The frozen 40-way linear head has a parity-organized geometry, so same-parity classes are intrinsically favored by the decision geometry, independent of the state distribution.
* **N-B state-displacement geometry.** The readout does not produce the effect by itself; the committed state/displacement preferentially occupies parity-organized regions (its decode is systematically same-parity as the target).
* **N-C joint.** Both contribute materially and neither control removes the majority of the effect.
* **N-D inconclusive.** Controls disagree or the effect cannot be localized.

## 5. Pre-registration (declared before inspecting aggregate results)

Recorded in `analyze_experiment_n.py`:

```
W_effect(seed) = mean cosine(W_u, W_v | same parity) - mean cosine(... | different parity)   (within register)
W permutation null : within each register, randomly permute the 10 value labels; N_PERM = 2000
obs             = P(dec(S_commit) mod 2 == y mod 2)                       (committed decode)
null_pair       = pairing-break null: permute y relative to the fixed decode within seed x dest
                  stratum (readout-only), equivalently permute the decode relative to y
                  (state-permutation) — state vectors and readout preserved
readout_fraction   = (null_pair - 0.5) / (obs - 0.5)
alignment_fraction = 1 - readout_fraction

W_strong = mean W_effect >= 0.05 AND >= 4/5 seeds with permutation p <= 0.05
N-A : W_strong AND readout_fraction >= 0.50
N-C : W_strong AND readout_fraction >= 0.30 AND alignment_fraction >= 0.30
N-B : alignment_fraction >= 0.60
N-D : otherwise
```

The 0.05 / 4-of-5 / 0.50 / 0.30 / 0.60 bars are declared conventions, not data-tuned cutoffs; all raw quantities are reported. The **parity axis is derived from W alone** (`p = normalize(μ_even − μ_odd)` over normalized value weight vectors), never from the wrong-value labels — avoiding the circularity warned about in the brief. Permutation controls are **analysis-only counterfactuals on frozen objects, not interventions**.

## 6. Frozen Components

Recurrent core, candidate network, observation encoder, gate network, **readout**, benchmark, task generator, optimizer, loss, and all Phase-2 machinery. No training, no weight modification, no optimizer, no gate/commit rescaling, no readout change; all inference under `torch.no_grad()`. No learned classifier is used as primary evidence.

## 7. Dataset and Primary Population

* **Data.** Canonical strict test split (1,200 × 10; stream `90042+seed`) for E and F × seeds 42–46 — the same split as G/H/I/J/K/L/M.
* **Primary population** (exact M/L definition, not redefined): strict test | ADD | general (`dest ≠ src` AND `operand_b > 0`) | candidate correct | baseline commit wrong | `ρ_margin ≥ α*_full` → **323 (E) / 353 (F)**; controls: candidate-correct successes (4,310 / 4,292) and candidate-correct non-off-path failures (409 / 387).
* **Alignment.** N's re-collected primary rows/groups are **identical** to M's (`max |Δ| = 0` on shared identifying/scalar columns), and the group-0/2 raw vectors (S_prev/C/S_commit) match M's stored failure vectors **bit-exactly** (`max |Δ| = 0`). Checkpoint SHA-256 verified before/after and against G/L/M metadata.

## 8. Readout Weight Geometry (W alone)

`z_{d,v}(S) = W[d,v]·S + b[d,v]`; the head is a single `nn.Linear(212→40)`. Per register, the ten value weight vectors are normalized and compared pairwise.

| Quantity (5-seed mean) | E | F |
|---|---:|---:|
| mean pairwise cosine, same parity | −0.0417 | −0.0411 |
| mean pairwise cosine, different parity | −0.1518 | −0.1518 |
| **same − different effect** | **+0.110** | **+0.111** |
| effect SD over seeds | 0.058 | 0.042 |
| seeds significant (perm p ≤ 0.05) | **4/5** | **4/5** |
| parity-axis separation ‖μ_even − μ_odd‖ | 0.796 | 0.797 |
| ‖W_v − W_u‖, same vs diff parity | 3.91 vs 3.92 | — |
| mean \|bias_y − bias_wrong\|, same vs diff parity | 0.059 vs 0.057 | 0.060 vs 0.057 |

Per-seed W effects (E): 0.008 (p=0.42), 0.128 (p=0.006), 0.147 (p=0.011), 0.130 (p=0.009), 0.137 (p=0.007). The parity geometry is present in four of five checkpoints; the weakest checkpoint (seed 42) is not significant. **Cross-register control:** the same−different effect is **+0.110 within register** but only **+0.024 across registers**, so the parity geometry is a genuine within-register *value* property, not an artifact of register structure. **Bias** shows no parity structure (same vs diff parity nearly equal), so the effect is in the weight vectors, not the offsets.

## 9. W-only Parity Null

The permutation null (randomly permuting value labels within each register, preserving all norms and pairwise geometry) has mean ≈ 0.001 and p95 ≈ 0.045–0.053. The observed per-seed effect (0.13–0.16 for the four significant seeds) lies well above the null (p ≈ 0.006–0.011). Inference is **seed-level**: the effect is present in 4/5 independent checkpoints, mean +0.110 ± 0.058 (E) and +0.111 ± 0.042 (F). The readout therefore does contain a real, reproducible, if small, parity organization.

## 10. State Projection onto Parity Axes

Using the W-derived parity axis per transition (`p_{dest}`), the committed displacement `Δ_commit = S_commit − S_prev` projects onto it with:

| Group | median ‖par disp‖ / ‖Δ‖ | median signed par disp | median par proj S_commit |
|---|---:|---:|---:|
| E off-path | 0.198 | −0.181 | −0.217 |
| E success | 0.214 | −0.472 | −0.192 |
| E non-off-path failure | 0.143 | +0.286 | −0.074 |
| F off-path | 0.145 | −0.027 | −0.124 |
| F success | 0.161 | −0.401 | +0.021 |
| F non-off-path failure | 0.121 | +0.171 | −0.095 |

The **parity-axis displacement magnitude is not off-path-specific**: the off-path ratio (0.198 E / 0.145 F) is comparable to, and slightly *lower* than, the success ratio (0.214 / 0.161). So a simple "the state moves along the parity axis" account does **not** explain the failure signature. The informative diagnostic is the **alignment** (§11), and the arithmetic conditioning (§15): the signed parity displacement tracks the target's parity (correlation −0.488 E / −0.463 F; even targets → positive displacement, odd targets → negative displacement), consistent with the committed state decoding a same-parity value.

## 11. Readout-only Null (and the equivalent State-Permutation Null)

Primary pairing-break control — keep the frozen readout and the observed `S_commit` distribution, permute the arithmetic labels relative to the decode within each seed × destination stratum:

| Quantity | E | F |
|---|---:|---:|
| observed same-parity rate `obs` | **0.737** | **0.640** |
| readout-only (label-permuted) null | **0.521** | **0.504** |
| state-permutation null | 0.522 | 0.503 |
| chance | 0.500 | 0.500 |
| gap `obs − null` | **+0.216** | **+0.137** |
| readout marginal excess `null − 0.5` | 0.021 | 0.004 |

Per-seed (E): gaps 0.063 (seed 42, p=0.21), 0.243, 0.322, 0.124, 0.349 (p ≤ 0.025 for 4/5 seeds); (F) all five gaps positive (0.113–0.162, p ≤ 0.06). The **label-permuted (readout-only) and state-permuted nulls coincide** (as expected — both break the same state↔label pairing) and both collapse to ≈0.50. Under the observed state distribution, the frozen readout with **no arithmetic alignment** produces essentially **no** same-parity bias (0.521 / 0.504).

## 12. State-Permutation Control

Holding the arithmetic labels fixed and permuting the committed decode relative to them (equivalently, permuting `S_commit` relative to the labels) yields the same collapse to ≈0.50 (state-permutation null 0.522 E / 0.503 F). **The parity signature does not survive breaking the state↔label pairing**, which is the direct signature that the effect depends on where the committed state sits relative to the true class, not on a state-independent readout property.

## 13. Readout-Permutation Control (and its degeneracy)

Permuting the ten value identities assigned to `W` and `b` within each register, leaving `S_commit` untouched, is **degenerate as a parity test**: mapping the decoded label back to its original identity is an exact no-op (`readout_perm_mapped_back = obs`, s.d. 0 in every replicate), while not mapping back trivially randomizes parity to ≈0.5 (`readout_perm_unmapped = 0.499 E / 0.501 F`). Neither variant can distinguish readout geometry from state geometry, because the parity of a predicted value is a property of the value label itself. **We therefore report it as a necessary-condition control only** (it confirms the parity effect requires the learned label assignment), and we do **not** rely on it as the primary discriminator — consistent with the brief's instruction to report ambiguity rather than force a result.

## 14. Third-Basin Pairwise Geometry (Analysis 14)

For the actual (true, wrong) class pair, per off-path failure:

| Pair | n | ‖w_y − w_wrong‖ | cos(w_y, w_wrong) | bias_y − bias_wrong | hyper dist S_prev / C / S_commit |
|---|---:|---:|---:|---:|---|
| same-parity wrong (E) | 238 | 3.73 | −0.008 | +0.012 | −0.223 / +0.374 / −0.185 |
| different-parity wrong (E) | 85 | 3.86 | −0.055 | +0.018 | +0.173 / +0.433 / −0.106 |
| same-parity wrong (F) | 226 | 3.87 | −0.009 | +0.013 | −0.170 / +0.330 / −0.140 |
| different-parity wrong (F) | 127 | 4.02 | −0.075 | +0.012 | −0.194 / +0.337 / −0.164 |

Same-parity competitors have weight vectors marginally **more aligned with the true class** (`cos` −0.008 vs −0.055, E) and a slightly **more negative** committed hyperplane distance (−0.185 vs −0.106), i.e. they sit closer to `w_y` — exactly the substrate that turns "the state is near the true class" into "the decode is a same-parity value". The differences are modest; the bias terms are negligible. This connects M's behavioral parity signature to the frozen linear decision geometry without claiming the readout dominates.

## 15. Arithmetic-Conditioned State Geometry

Descriptive only (no arithmetic-encoding claim). For off-path failures, the signed parity-axis displacement `Δ_commit·p` splits by arithmetic metadata:

| Condition | E | F |
|---|---:|---:|
| target even / odd | +0.781 / −1.338 | +0.542 / −1.007 |
| operand even / odd | +0.037 / −0.722 | +0.128 / −0.716 |
| wrap / no-wrap | +0.079 / −0.908 | −0.266 / −0.403 |
| corr(Δ·p, target parity) | −0.488 | −0.463 |

The displacement tracks the **target parity** (strongly) far more than the operand parity — consistent with the committed state decoding a same-parity value, not with an operand-parity code.

## 16. Control Populations

| Statistic | off-path failure | success | non-off-path failure |
|---|---:|---:|---:|
| same-parity rate (E) | 0.737 | 1.000 (offset 0, trivial) | 0.589 |
| same-parity rate (F) | 0.640 | 1.000 | 0.532 |
| parity displacement ratio (E) | 0.198 | 0.214 | 0.143 |
| W-only parity effect (broad, all states) | +0.110 | (same W) | +0.110 |

The readout's parity geometry is **broad** (a property of W for every state), but the **same-parity error** signature is specific to failures and stronger in the off-path population (0.737) than in non-off-path failures (0.589). So the readout substrate is universal, while the error signature is produced by the failure-specific state placement.

## 17. E-versus-F Comparison

| Quantity (5-seed) | E | F |
|---|---:|---:|
| mean W parity effect | 0.110 ± 0.058 | 0.111 ± 0.042 |
| seeds W-significant | 4/5 | 4/5 |
| observed same-parity | 0.737 | 0.640 |
| pairing-break null | 0.521 | 0.504 |
| alignment gap | 0.216 | 0.137 |
| readout fraction | 0.089 | 0.025 |
| alignment fraction | 0.911 | 0.975 |

Both models classify **N-B**. The alignment gap is somewhat larger for E (paired diff +0.083, dz +0.66, 4/5 seeds, p = 0.21 — descriptive, not a model-level difference). The localization is **reproducible across the two training objectives**.

## 18. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a secondary control and is never pooled. Transition-level rates are descriptive. The key probability is tested by **permutation/within-seed-sign-flip nulls** (2000 per-seed permutations for the W geometry; label/decode permutations within seed × destination stratum for the pairing-break controls). No threshold is tuned on the data; the pre-registered bars are declared in advance. Diagnostic quantities that are degenerate (the readout value-label permutation, §13) are reported as such rather than used to force a classification.

## 19. Localization Decision

Pre-registered components (E primary):
- `W_strong = True` (mean effect 0.110 ≥ 0.05; 4/5 seeds p ≤ 0.05).
- `readout_fraction = 0.089`, `alignment_fraction = 0.911`.
- `readout_dominant (≥0.50) = False`; `joint (≥0.30 both) = False`; `alignment_fraction ≥ 0.60 = True`.

→ **N-B — state-dominant parity alignment.** F control: `readout_fraction 0.025`, `alignment_fraction 0.975` → **N-B** as well.

The readout weight geometry is real but **non-dominant**: it produces only a ~2% state-independent same-parity preference, while the observed ~24 pp excess is produced by the state↔label alignment.

## 20. Interpretation / Causal Status

**Observation.** The frozen readout's value-weight geometry has a small, reproducible within-register parity organization (effect +0.110, 4/5 seeds, p ≈ 0.006–0.011). Holding the readout and the committed-state distribution fixed and breaking the state↔label pairing collapses the same-parity rate from 0.737/0.640 to ≈0.50; the readout's *state-independent* same-parity preference is therefore ≈0.02 (E) / 0.004 (F). The parity-axis displacement magnitude is comparable across groups; the signed displacement tracks the target parity.

**Supported interpretation.** The same-parity signature is **state-alignment-dominant**: it is produced by the committed state's decode being systematically same-parity as the target — i.e. by where the committed state sits relative to the true class — acting **through** the readout's small parity geometry (which is the necessary substrate: same-parity weight vectors sit marginally closer to `w_y`). Neither the readout alone (its independent behavioral contribution is ≈2–9% of the excess) nor a simple "state moves along the parity axis" displacement explains the effect.

**Causal status.** Measurement only — **no causal claim**. N localizes the *origin* of the parity signature; it does not show that the state alignment causes the sequence-level failures, nor that changing it would help. Permutation controls are counterfactual analyses of frozen objects, **not** interventions on the trained model.

**Unresolved.** N does not resolve whether the state alignment is a property of the *candidate/commit trajectory* (the committed state is a blend near `C` and `C` is near `w_y` because the candidate is correct) or of the *encoder/representation* that produces `S_commit`. It also does not resolve whether the small readout parity geometry is incidental (an incidental correlate of training) or functional.

## 21. What N Establishes

* A clean, deterministic, frozen localization of the parity signature's origin, with the primary parity axis derived from **W alone** (no circular use of the wrong-value labels).
* That the frozen readout **does** contain a small but real, reproducible (4/5 seeds) within-register parity organization in its value weight geometry (+0.110 / +0.111), **not** driven by register structure (+0.024 cross-register) and **not** by bias terms.
* That this geometry produces **almost no state-independent same-parity preference**: the readout-only / state-permutation pairing-break null is 0.521 / 0.504 vs chance 0.50.
* That the observed signature (0.737 / 0.640) is **overwhelmingly state↔label alignment** (91.1% / 97.5% of the excess over chance).
* That the **readout value-label permutation is degenerate** for parity (mapped-back = no-op, unmapped ≈ 0.5) and cannot serve as the primary discriminator.
* That the parity-axis **displacement magnitude** is not off-path-specific (off-path 0.198/0.145 vs success 0.214/0.161), so a naive "movement along the parity axis" account is not the mechanism.
* That the localization is **reproducible across E and F** (both N-B).

## 22. What N Does NOT Establish

* It does **not** establish causality and does **not** design or test a fix.
* It does **not** show the readout is parity-structured *because* the predictions are (the axis is W-derived and a permutation null is used), and it does not claim the parity geometry is a designed feature.
* It does **not** claim exact arithmetic encoding, nor that the parity geometry reflects the arithmetic (parity is a mod-2 property, not the mod-10 carry — M already rejected the wrap hypothesis).
* It does **not** attribute the state alignment to a specific representation stage (encoder vs candidate vs commit blend).
* It does not generalize beyond the ADD off-path population, the register-tracking benchmark, or the E/F objectives; and it does not begin Phase 2.

## 23. Final Decision

**N-B — state-dominant parity alignment** (E primary; F control agrees). Pre-registered components: `W_strong = True` (mean effect 0.110 ≥ 0.05; 4/5 seeds significant) but `readout_fraction = 0.089 < 0.50`; `alignment_fraction = 0.911 ≥ 0.60`. The readout's parity geometry is a small, real, non-dominant substrate; the observed same-parity signature is produced by the state↔label alignment.

**Answers to the required final questions.**

1. **Does the frozen readout weight geometry itself contain parity organization?** **Yes** — within-register same-parity value vectors are more similar (cos −0.042 vs −0.152; effect +0.110 / +0.111); it is a within-register value property (+0.024 cross-register), not register or bias structure.
2. **Is it stronger than a value-label permutation null?** **Yes in 4/5 seeds** (p ≈ 0.006–0.011; the weakest seed, 42, is not significant), mean effect +0.110 ± 0.058 (E) / +0.111 ± 0.042 (F).
3. **Does S_commit move along readout-defined parity directions?** **Partly** — the signed parity displacement tracks the target parity (corr −0.49 / −0.46), but the displacement *magnitude* ratio (0.198/0.145) is modest.
4. **Is that movement stronger in off-path failures than controls?** **No** — the ratio is comparable to or slightly *lower* than for successes (0.214/0.161); the parity-axis displacement is not off-path-specific.
5. **Does breaking state/arithmetic alignment eliminate the parity signature?** **Yes** — the pairing-break null collapses to ≈0.50 (0.521/0.504) from 0.737/0.640 (gap +0.216/+0.137).
6. **Does breaking readout/value alignment eliminate it?** The readout-label permutation is **degenerate**; mapped-back it is an exact no-op (= obs), unmapped it trivially randomizes parity to ≈0.5. It cannot cleanly answer and is reported as a necessary-condition control only.
7. **Which control attenuates the observed effect more strongly?** The **state/label pairing-break** removes 91% (E) / 97% (F) of the excess; the readout-label permutation is degenerate (0% when mapped back). The state side dominates.
8. **Is parity specific to the third basin or broadly present in the readout?** The readout parity *geometry* is **broad** (a property of W for all states); the same-parity *error signature* is failure-specific and stronger off-path (0.737) than in non-off-path failures (0.589).
9. **Is the same localization present in E and F?** **Yes** — both **N-B** (alignment fraction 0.911 / 0.975; W effect 0.110 / 0.111; 4/5 seeds each).
10. **Is the evidence strong enough to justify ONE targeted intervention?** The localization is **state-dominant**, which licenses a **representation-side** follow-up and argues *against* a readout-side fix (the readout's independent effect is ≈2–9%). N does **not** design that experiment; it only determines the origin.

## 24. Exactly One Next Scientific Experiment

**Experiment O — controlled representation-side test of the parity alignment (state-dominant follow-up; N does not design the intervention).** Because N localizes the parity signature to the **state↔label alignment** rather than the readout, the single next experiment is a tightly controlled **representation-side** study of that alignment. Its purpose is to determine whether the committed state's same-parity decode is a property of the committed-state *placement relative to the true-class region* (the measured lever) or of the upstream candidate/encoder representation that produces `S_commit` — a controlled, frozen-checkpoint, measurement-first test on the representation side. Per the research standard, N stops here: it **determines the origin** and names the representation-side experiment the localization licenses; it does **not** design an intervention, does not alter any frozen component, and does not begin Phase 2.

## Reproduction

```bash
# 1. Collect parity-origin records from the frozen checkpoints (~10 s)
python experiments/run_experiment_n.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_n_parity_origin.npz

# 2. Analysis + 12 figures (~45 s; permutation controls)
python experiments/analyze_experiment_n.py \
    --npz experiments/experiment_n_parity_origin.npz \
    --output experiments/experiment_n_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_n

# 3. Tests
pytest tests/test_experiment_n.py -v      # 16 tests
pytest tests/                             # full suite
```

Determinism: the collector was run twice (arrays bit-identical, `max |Δ| = 0.0`); the analysis twice (JSON identical excluding the timestamp). Checkpoint SHA-256 verified before/after; N/M row and vector alignment exact.

## Artifact List

* `experiments/run_experiment_n.py` — collector: per-seed W,b, primary population, W-derived parity-axis projections, failure vectors (M-aligned).
* `experiments/analyze_experiment_n.py` — W-geometry + permutation null, cross-register control, state projections, pairing-break/readout-permutation controls, value-pair geometry, arithmetic conditioning, controls, E-vs-F, pre-registered N-A/B/C/D decision, 12 figures.
* `tests/test_experiment_n.py` — 16 tests (collection determinism, checkpoint hashes, no-grad/no-mutation, population filtering, W shape/indexing, parity-axis construction/normalization, same/diff pair construction, permutation-null determinism, state/displacement projection correctness, state/readout permutation controls, control groups, E/F alignment, JSON determinism, figures, malformed data, historical immutability).
* `experiments/experiment_n_parity_origin.npz` — per-seed `(W,b)`, primary scalars + parity projections, M-aligned failure vectors.
* `experiments/experiment_n_parity_origin_meta.json` — columns, checkpoint hashes, M alignment.
* `experiments/experiment_n_results.json` — W geometry, nulls, controls, value-pair geometry, groups, classification.
* `docs/reports/figures/experiment_n_*.png` — 12 figures: `w_geometry`, `w_null`, `parity_axis`, `cross_register`, `state_proj`, `par_delta`, `readout_null`, `controls`, `value_pair_geometry`, `arithmetic`, `control_groups`, `E_vs_F`.
* Checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`.
* Historical references (read only, unchanged): `experiments/experiment_m_geometry.npz`, `experiments/experiment_l_readout.npz`, `experiments/experiment_i_coordinate.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_g_transitions.npz`.

## Scientific Guardrails

Measurement-only localization; **no causal claim** and **no fix**. The readout is the actual frozen implementation; the parity axis is derived from W alone (no circular use of the wrong-value labels) and tested against a value-label permutation null. Pairing-break controls are explicitly labeled analysis-only counterfactuals on frozen objects, **not interventions**. Degenerate controls (readout value-label permutation) are reported as ambiguous rather than used to force a verdict. Parity is a mod-2 property; no exact arithmetic-encoding claim is made, and M's rejection of the wrap/carry hypothesis stands. The seed is the independent unit; transition-level rates are descriptive. Negative results are preserved; Experiments G/H/I/J/K/L/M reports and artifacts are untouched.
