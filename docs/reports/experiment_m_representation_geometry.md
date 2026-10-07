# RE:Reason — Experiment M: Representation Geometry of the Third Basin

**Date:** October 7, 2026
**Status:** Completed — Measurement only (frozen checkpoints; no training; no optimizer; no weight/readout/gate/commit modification; no intervention; no recursive counterfactual trajectories; no learned classifier as primary evidence). Classification: **M-A — systematic modular geometry** (E primary; the F control agrees), with the explicit qualification that the modular structure is **parity (mod 2)**, *not* wrap-around and *not* ±1 adjacency.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned
**Data:** `experiments/experiment_m_geometry.npz`, `experiments/experiment_m_geometry_meta.json`, `experiments/experiment_m_results.json`
**Runners:** `experiments/run_experiment_m.py`, `experiments/analyze_experiment_m.py`, tests `tests/test_experiment_m.py`
**Checkpoints:** `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt` (hashes re-verified against Experiment G/L metadata)

---

## 1. Executive Answer

**The third basin is a *systematic* modular region — but the modular structure is parity (mod 2), not the wrap-around carry structure that was the natural hypothesis.** On the exact Experiment-L off-path population (E n = 323; F n = 353), the wrong destination value **preserves the parity of the true target in 73.7% (E) / 64.0% (F) of cases against a 50.2% / 49.9% permutation null (p ≈ 0)** — i.e. it is a same-parity value lying at an **even** modular offset (dominantly ±2 and ±4) from the target, *not* ±1. It is **not driven by wrap**: `P(v_wrong | wrap)` and `P(v_wrong | no-wrap)` differ only mildly (TV 0.240 / 0.212, below the pre-registered 0.30 bar), and the mean modular distances to the target are essentially equal for wrap and no-wrap (3.09 vs 3.19 E). The off-axis committed displacement is **directionally consistent at the seed level** (per-seed mean resultant R = 0.256 ± 0.013 (E) and 0.306 ± 0.017 (F), all five seeds significant against a within-seed sign-flip null, p ≈ 0), weakly arithmetic-localized (each wrap/no-wrap group aligns more with its own mean direction; within-operand groups R ≈ 0.19–0.27), and the wrong region is **entered deeply** (normalized true-vs-flat-winner distance −2.27 (E) vs −0.90 for successes). The wrong *register* is not localized (the flat 40-way winner is essentially never the destination; it is "neither" destination nor source in 84% / 89% of off-path failures).

Classification: **M-A — systematic modular geometry** (E and F alike). The qualification matters: the measured modular axis is **parity**, not wrap/adjacency, so the finding is *not* "the third basin is the modular-carry basin".

Answers to the required decision questions are given in §21.

## 2. Existing Baseline (what G–L established, and what M resolves)

* **G** (`experiment_g_add_commit_gap.md`): a real ADD candidate→commit residual, concentrated in general two-register adds; gate-dilution vs blend/readout evidence mixed (C).
* **H** (`experiment_h_lambda_interpolation.md`): the element-wise vector-gate commit is bit-exact; the committed state lies off the scalar `S_prev→C` path.
* **I** (`experiment_i_coordinate_commit.md`): coordinate-wise under-movement plus a placement component; ~45% of failures deliver at least the scalar-equivalent transfer yet still fail off-path.
* **J** (`experiment_j_commit_intervention.md`): commit-only *magnitude* scaling moves local margin delivery but no sequence endpoint; coordinate identity essential.
* **K** (`experiment_k_gate_balance.md`): commit-only *shape* remapping achieves the favorable delivery profile yet still no sequence endpoint; commit rule frozen.
* **L** (`experiment_l_readout_localization.md`): the off-path residual is **latent-state placement** — the committed state clears the intended competitor (100%) but a different competitor wins (100%), the decoded value is a third value, and fixed readout bias is negligible (bias-removal flips 1–3%).
* **M (this report)** asks *what* that third basin is: a systematic modular geometry, a systematic non-modular placement, or an unsystematic collapse. It finds a systematic **parity**-modular signature plus a seed-consistent but weak directional structure.

## 3. Research Question

> Is the third (wrong) basin a **systematic arithmetic/modular geometry** — in particular wrap-around or an adjacent modular value — or an **unsystematic off-axis representation collapse**?

## 4. Hypotheses

* **M-A (systematic modular geometry).** Wrong-basin identity is strongly tied to modular arithmetic structure and/or wrap status, together with directional representation structure.
* **M-B (systematic non-modular placement).** Off-axis displacement is directionally consistent, but wrong-basin identity does not map cleanly onto modular arithmetic.
* **M-C (unsystematic representation collapse).** Wrong-basin identities and off-axis directions show little systematic structure beyond the already-known failure population.
* **M-D (mixed / inconclusive).** Multiple signatures appear but no explanation clearly dominates.

## 5. Pre-registration (declared before inspecting aggregate results)

Recorded in `analyze_experiment_m.py`; the pre-registered quantities are:

```
d_mod(a,b) = min((a-b) mod 10, (b-a) mod 10)                       # declared modular distance
modular_identity      = same-parity rate among off-path failures >= 0.60 AND permutation p <= 0.01
wrap_specific         = TV(P(v_wrong|wrap), P(v_wrong|no-wrap)) >= 0.30
directional_systematic = seed-level mean resultant R >= 0.15 AND every seed's permutation p <= 0.01
directional_weak       = any seed's permutation p <= 0.01   (reported as a sub-finding)
```

Classification: **M-A** if `modular_identity and directional_systematic`; **M-B** if `directional_systematic and not modular_identity`; **M-C** if `not modular_identity and not directional_weak`; else **M-D**. The directional test uses the **seed-level** mean resultant R (the five checkpoints are the independent units and each has its own frozen readout); the pooled R is reported descriptively only. The 0.60 / 0.01 / 0.30 / 0.15 bars are declared conventions, not data-tuned cutoffs; all raw quantities are reported so the verdict can be judged independently.

## 6. Frozen Components

Recurrent core, candidate network, observation encoder, gate network, **readout**, benchmark, task generator, optimizer, loss, and all Phase-2 machinery. No training, no weight modification, no optimizer, no gate/commit rescaling, no readout change; all inference under `torch.no_grad()`. No learned classifier is used as primary evidence. The only counterfactual-like construct is the pre-declared analysis-only **flat 40-way winner** and the **permutation nulls**; neither is used in a rollout.

## 7. Dataset and Primary Population

* **Data.** For each of E and F × seeds {42..46}: one `forward_with_states(..., return_components=True)` pass over the canonical strict test split (1,200 × 10; stream `90042+seed`), the same split as G/H/I/J/K/L.
* **Scale.** 15,763 ADD transitions/model. Primary population (exact Experiment-L definition, **not redefined**): strict test | ADD | general (`dest ≠ src` AND `operand_b > 0`) | candidate correct | baseline commit wrong | `ρ_margin ≥ α*_full` → **323 (E) / 353 (F)** off-path failures; 4,310 / 4,292 candidate-correct successes (control B); 409 / 387 candidate-correct non-off-path failures (control C).
* **Cross-experiment alignment.** M's re-collected rows reproduce Experiment L's stored test rows **exactly** (`max |Δ| = 0` on all 21 shared identifying/scalar columns) and the off-path counts match (323/353). Checkpoint SHA-256 verified before/after and against G/L metadata.
* **Arithmetic.** `y = (operand_a + operand_b) mod 10`; `wrap = (operand_a + operand_b) ≥ 10` (reconstructed exactly from the stored operands; tested).

## 8. Experimental Procedure

1. Load a frozen checkpoint; verify its SHA-256.
2. One forward pass; capture `S_prev`, `C`, `S_commit`, and the exact frozen-readout objects: the flat 40-way logits at each state, the true-vs-flat-winner margins, the destination-register margins, the off-axis decomposition of `Δ_commit` vs `Δ_candidate`, and the arithmetic mapping columns.
3. Construct the three control groups (A off-path failure, B success, C non-off-path failure).
4. Run the core analyses A–F + candidate-vs-commit geometry, the permutation nulls, the control and E-vs-F comparisons, the pre-registered classification, and the figures.
5. Re-run the collector and the analysis; require deterministic equality.

## 9. Reproduction and Determinism

* Collector: two full runs produce **bit-identical** arrays (all 20 stored arrays, `max |Δ| = 0.0`); metadata identical excluding the timestamp.
* Analysis: two runs produce **identical** JSON (excluding the timestamp).
* Off-axis decomposition verified against the stored columns from the raw `S_prev/C/S_commit` vectors (orthogonality of the residual to `Δ_candidate` checked).
* Checkpoint SHA-256 verified unchanged before/after; no parameter mutation (tested).

## 10. Third-value Arithmetic Mapping (E primary, off-path; F in parentheses)

| Quantity | E | F |
|---|---:|---:|
| same-parity rate (v_wrong ≡ y mod 2) | **0.737** | 0.640 |
| permutation null mean / p | 0.502 / **p ≈ 0** | 0.499 / **p ≈ 0** |
| wrong == old value | 0.000 | 0.006 |
| wrong == operand value | 0.164 | 0.136 |
| \|offset\| = 1 / 2 / 3 / 4 / 5 counts | 32 / 101 / 18 / **137** / 35 | 39 / 115 / 32 / **111** / 56 |
| mean d_mod(v_wrong, y) | 3.13 | 3.09 |
| mean d_mod(v_wrong, old) | 2.78 | 2.76 |
| mean d_mod(v_wrong, operand) | 2.32 | 2.35 |
| TV(observed v_wrong, reference y-distribution) | 0.167 | 0.249 |
| TV(observed v_wrong, uniform) | 0.066 | 0.090 |

Signed-offset counts (E, in [-5,4]): −5:35, −4:60, −3:11, −2:52, −1:16, +1:16, +2:49, +3:7, +4:77. The wrong value sits overwhelmingly at **even** modular offsets (same parity): ±2 and ±4 dominate, while ±1 and ±3 (parity-flipping) are rare. The reference baseline is the **empirical true-value distribution in the same primary population** (not a uniform assumption); the observed distribution is closer to the reference than to uniform, but the parity dependence is the dominant structure. This is a **modular (mod-2) signature**, not the ±1 adjacency and not the mod-10 carry.

## 11. Register Mapping (flat 40-way committed winner)

| Group | winner == dest | == source | neither |
|---|---:|---:|---:|
| E off-path failure | 0.003 | 0.155 | 0.842 |
| E success | 0.072 | 0.247 | 0.682 |
| F off-path failure | 0.000 | 0.110 | 0.890 |
| F success | 0.094 | 0.194 | 0.712 |

The committed global argmax is essentially **never** the destination register for off-path failures and is "neither destination nor source" in 84% / 89% of cases. Successes also mostly have a non-destination global argmax (0.68 / 0.71), so the *register* is not a failure-specific localization; the failure-specific shift is that the global winner moves *away* from the destination/source more often. **No clean register-specific third basin is found.** (The destination value is decoded per-register; the flat winner is a diagnostic of global readout geometry.)

## 12. Wrap / Modular Analysis

| Quantity | E | F |
|---|---:|---:|
| wrap fraction | 0.548 | 0.558 |
| TV(P(v_wrong\|wrap), P(v_wrong\|no-wrap)) | **0.240** | **0.212** |
| same-parity \| wrap / no-wrap | 0.706 / 0.774 | 0.624 / 0.660 |
| d_mod(v_wrong, y) \| wrap / no-wrap | 3.085 / 3.185 | 3.076 / 3.096 |

**The wrong-value identity does not change materially with wrap status** (TV below the pre-registered 0.30 bar; the mean modular distance to the target is essentially identical for wrap and no-wrap). The parity structure is present in both wrap and no-wrap transitions. Wrap-around is therefore **not** the organizing geometry of the third basin. This is a direct negative result for the "modular carry" hypothesis.

## 13. Off-axis Directional Geometry

Decomposition of `Δ_commit = S_commit − S_prev` into the candidate direction `Δ_candidate = C − S_prev` (parallel) and its orthogonal complement:

| Group | median ‖parallel‖ | median ‖orthogonal‖ | median cos | median ‖orth‖/‖Δ_cand‖ |
|---|---:|---:|---:|---:|
| E off-path | 5.17 | 6.59 | 0.613 | 0.415 |
| E success | 4.97 | 6.36 | 0.611 | 0.420 |
| E non-off-path failure | 4.58 | 6.24 | 0.588 | 0.399 |

The **off-axis ratio does not distinguish** off-path failures from successes (0.415 vs 0.420) — consistent with Experiment L's negative result. The informative structure is **directional**, not magnitude-based:

| Directional statistic | E | F |
|---|---:|---:|
| seed-level mean resultant R (5 checkpoints) | **0.256 ± 0.013** | **0.306 ± 0.017** |
| per-seed sign-flip permutation p | ≈ 0 (all 5 seeds) | ≈ 0 (all 5 seeds) |
| within-seed null mean R (n ≈ 65) | ≈ 0.12 | ≈ 0.12 |
| pooled R (descriptive only) | 0.110 | 0.138 |
| pooled mean pairwise cosine | 0.009 | 0.016 |

**Within each model the off-axis direction is systematic** (R ≈ 2–2.5× the within-seed null; significant in every seed). The pooled R is smaller only because the five checkpoints have different readout weight vectors, so their mean off-axis directions differ — pooling across seeds is an invalid unit for a directional test. Per the pre-registration, the seed is the independent unit.

## 14. Does Off-axis Direction Track Arithmetic? / Candidate-vs-Commit Geometry

* **Arithmetic localization (Core E):** each wrap/no-wrap group aligns more with **its own** mean direction than the other group's (E: wrap group cos to wrap-mean 0.124 vs to no-wrap-mean 0.063; no-wrap group 0.134 vs 0.058; F: 0.145/0.108 and 0.155/0.101). Within-operand groups show higher consistency than the pooled set (E R = 0.19–0.27; F 0.19–0.29). The directional structure is therefore **mildly arithmetic-localized** (operand/wrap subgroups), though the correlation between the signed offset and the orthogonal magnitude is negligible (−0.031 / −0.040).
* **Third-basin depth (Core F):** median true-vs-flat-winner margin at `S_commit` = **−8.27 (E) / −8.46 (F)** for off-path failures vs **−3.39 / −3.24** for successes; normalized distance −2.27 / −2.23 vs −0.90 / −0.84; margin against the committed competitor −0.17 / −0.15. The third basin is **broad and deeply entered**, not boundary-adjacent.
* **Candidate-vs-commit (Core 8):** off-path failures start from a **weak candidate margin** (m_cand 5.74 / 4.60 vs 8.52 / 8.35 for successes), lose margin at the commit (loss 3.18 / 2.63), and end with only a small positive margin against the original competitor (m_commit 1.96 / 1.94). Non-off-path failures have a *negative* commit margin (−0.61 / −0.53) — they never clear the intended competitor. So off-path failures are characterized by a **weaker candidate margin plus wrong placement**, not by one or the other alone.

## 15. Control Comparisons

| Statistic | A off-path failure | B success | C non-off-path failure |
|---|---:|---:|---:|
| same-parity rate (E) | 0.737 | 1.000 (trivial) | 0.589 |
| off-axis ratio (E) | 0.415 | 0.420 | 0.399 |
| off-axis cos (E) | 0.613 | 0.611 | 0.588 |
| hyper true-vs-flat-winner (E) | −2.27 | −0.90 | −2.32 |
| same-parity rate (F) | 0.640 | 1.000 (trivial) | 0.532 |

The parity signature is **strongest in the off-path population** (0.737 / 0.640) and falls for non-off-path failures (0.589 / 0.532) and vanishes trivially for successes (offset 0 is even). The third-basin depth separates off-path and non-off-path failures from successes. The off-axis *ratio* (magnitude) does **not** separate the groups — the discriminating structure is the parity signature and the directional/statistical depth, not the off-path distance.

## 16. E-versus-F Comparison

| Statistic (seed-level, mean ± SD) | E | F | paired diff / p |
|---|---:|---:|---|
| same-parity rate | 0.741 ± 0.123 | 0.640 ± 0.068 | +0.101, p = 0.15 |
| seed-level R | 0.256 ± 0.013 | 0.306 ± 0.017 | −0.050, 5/5, p < 0.01 |
| mean pairwise cosine | 0.051 ± 0.007 | — | — |

Both E and F classify **M-A**. The parity signature is somewhat larger for E; the directional consistency is somewhat larger for F. The mechanism is **robust across the two training objectives** (no qualitative difference). E is the primary scientific model; F is a control and is never pooled.

## 17. Statistical Discipline

The five frozen E checkpoints are the independent units. Transition-level tables are descriptive. The key nulls are **permutation tests**: (i) the same-parity rate is compared against a permutation of `v_wrong` relative to `y` (2,000 permutations; p ≈ 0), which controls for the marginal distributions; (ii) the directional resultant is compared against a per-seed sign-flip null (500–2,000 permutations). The reference distribution for the wrong-value histogram is the empirical true-value distribution in the same population (not a uniform assumption). No threshold is tuned on the data; the pre-registered bars are declared in advance and every raw quantity is reported.

## 18. Interpretation and Causal Status

**Observation.** The off-path wrong value systematically preserves the **parity** of the target (74% / 64% vs 50% null), sitting at even modular offsets (dominantly ±2 and ±4); wrap status does not change the wrong-value identity; the off-axis committed displacement is directionally consistent within each model (R ≈ 0.26–0.31, all seeds significant) and mildly arithmetic-localized; the wrong region is deeply entered; and the wrong *register* is not localized.

**Supported interpretation.** The third basin is a **systematic modular geometry**, but the modular axis is **parity (mod 2)**, not the wrap-around carry (mod 10) and not ±1 adjacency. This is a genuine, reproducible, seed-consistent structure, and it is not a generic collapse.

**Causal status.** Measurement only — **no causal claim**. M localizes the *geometry* of the third basin; it does not show that the parity structure causes the failures, nor whether it originates in the state representation or in the frozen readout's value geometry.

**Unresolved.** M does not determine **where** the parity structure lives: it could be a property of the committed **state** (the state lies in a same-parity region of latent space) or of the frozen **readout's value geometry** (the 40 class weight vectors cluster by parity, so any committed state is more likely to decode to a same-parity value). It also does not resolve whether the modest off-axis direction is state-geometry or readout-geometry. This attribution is the single most important unresolved question and the subject of the one next experiment.

## 19. What M Establishes

* A frozen, exact, measurement-only characterization of the third basin across the full off-path population and two control groups.
* That the off-path wrong value has a **strong same-parity (mod-2) modular signature** (0.737 E / 0.640 F vs 0.50 permutation null, p ≈ 0) — a systematic, reproducible modular structure.
* A **direct negative result for the wrap/modular-carry hypothesis**: wrap status does not materially change the wrong-value identity (TV 0.24 / 0.21, below the declared bar), and the modular distance to the target is essentially equal for wrap and no-wrap.
* A **negative result for ±1 adjacency**: only ~10% of failures are ±1; the offsets concentrate at ±2, ±4.
* That the **wrong register is not localized** (the flat winner is "neither" destination nor source in 84% / 89% of failures).
* That the off-axis committed displacement is **directionally consistent within each frozen model** (seed-level R = 0.256 / 0.306, all seeds significant vs a within-seed permutation null), mildly arithmetic-localized — while the pooled R is small because the checkpoints' readout geometries differ.
* That the third basin is **deeply entered** (normalized true-vs-flat-winner distance −2.27 vs −0.90 for successes), and that the off-path *distance* magnitude does **not** separate groups (re-confirming Experiment L).
* That the geometry is **robust across E and F** (both M-A).

## 20. What M Does NOT Establish

* It does **not** establish causality, and it does **not** design or test a fix.
* It does **not** show the third basin is the modular-carry basin: the modular structure is **parity**, and the wrap hypothesis is not supported.
* It does **not** attribute the parity structure to the state representation versus the frozen readout value-geometry (the central unresolved question).
* It does **not** show exact arithmetic encoding from decodability — the parity signature is a decision-level measurement, not an arithmetic-encoding claim.
* It does **not** use a learned classifier, and it does not generalize beyond the ADD off-path population, the register-tracking benchmark, or the E/F objectives.
* It does not begin Phase 2 (no branching, verifier, recursion, transactional commit).

## 21. Final Decision

**M-A — systematic modular geometry** (E primary; F control agrees). Pre-registered components: `modular_identity = True` (same-parity 0.737 E / 0.640 F ≥ 0.60, permutation p ≈ 0) and `directional_systematic = True` (seed-level R 0.256 / 0.306 ≥ 0.15, every seed p ≤ 0.01); `wrap_specific = False` (TV 0.24 / 0.21 < 0.30). The verdict is **M-A**, **with the essential qualification that the modular axis is parity (mod 2), not wrap-around or ±1 adjacency.**

**Answers to the required decision questions.**

1. **Is the third wrong value systematically related to modular addition?** **Yes** — it preserves the target's parity (74% / 64% vs 50% null, p ≈ 0); a genuine modular (mod-2) structure.
2. **Specifically related to wrap-around?** **No** — wrap status does not change the wrong-value identity (TV 0.24 / 0.21 below bar; mean modular distance to target 3.09 / 3.19 wrap/no-wrap).
3. **Usually an adjacent modular value (±1)?** **No** — only 9.9% (E) / 11.0% (F) are ±1; the offsets concentrate at ±2 and ±4 (\|offset\| ∈ {2,4}: 73.7% / 64.0%).
4. **Is the wrong register systematic?** **No clean localization** — the flat 40-way winner is at the destination ≈0%, at the source 16% / 11%, and "neither" 84% / 89%; the failure-specific shift is that the global winner moves away from destination/source.
5. **Is the off-axis displacement direction consistent?** **Yes at the seed level** (R = 0.256 ± 0.013 E / 0.306 ± 0.017 F; all seeds significant vs within-seed null), **though weak in magnitude** (pooled R 0.11 / 0.14).
6. **Does off-axis direction correlate with arithmetic structure?** **Weakly, yes** — subgroups align with their own mean direction (wrap/no-wrap) and within-operand groups are more consistent (R ≈ 0.19–0.27) than the pooled set.
7. **Is the third basin broad/deep or boundary-adjacent?** **Deep** — normalized true-vs-flat-winner distance −2.27 / −2.23 vs −0.90 / −0.84 for successes.
8. **Is the same geometry present in E and F?** **Yes** — both classify M-A; both show strong parity and seed-level directional structure.
9. **Is the evidence strong enough to justify a representation-side intervention?** **Not yet, as a direct step.** The modular structure is real, but **where it lives (state geometry vs frozen readout value-geometry) is not localized**; a representation-side intervention now would risk targeting the wrong stage (cf. the project's "trace effects to the right stage" guardrail). One more measurement is required first.

## 22. Exactly One Next Scientific Experiment

**Experiment N — Parity-Origin Localization (measurement-only, frozen checkpoints; no training, no intervention).** M established a strong same-parity modular signature and a weak-but-seed-consistent off-axis direction, but could not attribute the parity structure to the **state representation** or to the **frozen readout's value geometry**. N must resolve exactly this, so that any later intervention targets the right stage:

1. **Readout value-geometry.** Measure the frozen readout's 40 class weight vectors: their pairwise geometry, and specifically whether the weight vectors for the 10 values (at each register) form a **parity-organized** structure (e.g. same-parity value directions are more aligned; a parity axis is recoverable from `{W}` alone). Quantify how much of the same-parity wrong-value rate a **readout-only** null (a committed state drawn from the observed `S_commit` distribution, decoded by the frozen head) would reproduce.
2. **State geometry.** Project `S_commit`, `C`, `S_prev` onto the parity-organized readout axes (and onto the parity-preserving vs parity-flipping direction pairs) and test whether the committed **state displacement** `Δ_commit` is parity-organized (moves preferentially within same-parity decision axes) versus the readout being parity-structured irrespective of the state.
3. Decompose the same-parity rate into a **readout-geometry** component and a **state-displacement** component with a pre-declared, exactly-computable split (e.g. compare the observed same-parity rate to the rate from state-permuted and readout-permuted controls), and report which component dominates.

**Outcome rule.** If the parity structure is dominated by the readout value-geometry, the next step after N (not now) would be a tightly controlled **readout-side** probe of that geometry; if it is dominated by the state displacement, it would be a **representation-side** measurement. If neither dominates, document the parity signature as a joint property and do not proceed to an intervention. **M designs no intervention**; it only determines which mechanism is sufficiently supported for N.

## Reproduction

```bash
# 1. Collect representation-geometry records from the frozen checkpoints (~10 s)
python experiments/run_experiment_m.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_m_geometry.npz

# 2. Analysis + 12 figures (~5 s)
python experiments/analyze_experiment_m.py \
    --geometry experiments/experiment_m_geometry.npz \
    --output experiments/experiment_m_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_m

# 3. Tests
pytest tests/test_experiment_m.py -v      # 19 tests
pytest tests/                             # full suite
```

Determinism: the collector was run twice (arrays bit-identical, `max |Δ| = 0.0`); the analysis twice (JSON identical excluding the timestamp). Checkpoint SHA-256 verified before/after and against G/L metadata; M/L row alignment exact.

## Artifact List

* `experiments/run_experiment_m.py` — collector: flat 40-way winner, arithmetic mapping, off-axis decomposition, directional units.
* `experiments/analyze_experiment_m.py` — core analyses A–F, controls, E-vs-F, permutation nulls, pre-registered M-A/B/C/D classification, 12 figures.
* `tests/test_experiment_m.py` — 19 tests (collection determinism, checkpoint hashes, no-grad/no-mutation, primary-population filtering, arithmetic reconstruction, modular-distance correctness, wrong-value/register identification, vector decomposition + orthogonality, directional/parity nulls, wrap grouping, control groups, E/F alignment, JSON determinism, figures, malformed data, historical immutability).
* `experiments/experiment_m_geometry.npz` — per-ADD-transition scalars + primary unit off-axis + failure vectors + per-seed readout `(W, b)`.
* `experiments/experiment_m_geometry_meta.json` — columns, checkpoint hashes, L alignment.
* `experiments/experiment_m_results.json` — core analyses, classification, seed-level stats, E-vs-F.
* `docs/reports/figures/experiment_m_*.png` — 12 figures: `wrong_value_hist`, `wrong_offset`, `modular_distance`, `wrap_split`, `wrong_register`, `offaxis_decomposition`, `directional_consistency`, `offaxis_arithmetic`, `third_basin_depth`, `candidate_vs_commit`, `control_comparison`, `E_vs_F`.
* Checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`.
* Historical references (read only, unchanged): `experiments/experiment_l_readout.npz`, `experiments/experiment_i_coordinate.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_g_transitions.npz`.

## Scientific Guardrails

Measurement-only localization; **no causal claim** and **no fix**. The element-wise/readout machinery is the actual frozen implementation, not an assumed form. `d_mod` is defined explicitly; the reference distribution for the wrong-value histogram is the empirical true-value distribution, not an assumed uniform; the same-parity rate is compared against a permutation null, and the directional statistic against a within-seed sign-flip null (the seed is the independent unit). No learned classifier is used as primary evidence; decodability is never conflated with exact arithmetic encoding. "Basin" here means the region of latent space decoded to a class under the frozen linear readout — **not** a nonlinear dynamical attractor. Modularity is **not** claimed merely because wrap is common (the wrap hypothesis is in fact not supported). The result is stated as a measured geometry ("the third basin is parity-organized"), never as a mechanism that causes the failures, and no generalization to NLP/Phase-2/AGI is made. Negative results are preserved; Experiments G/H/I/J/K/L reports and artifacts are untouched.
