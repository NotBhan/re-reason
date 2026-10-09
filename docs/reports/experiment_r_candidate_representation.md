# RE:Reason — Experiment R: Candidate-Representation Information Test on the Class-A Tail

**Date:** October 7, 2026
**Status:** Completed — Measurement only (frozen checkpoints; no model training; no optimizer on model weights; no architecture/candidate/core/gate/commit/readout modification; no inference-time intervention; no branching/verifier/oracle; no counterfactual rollout). Analysis-only probes are fit on the regime's train split, tuned on its validation split, and evaluated once on the untouched canonical-TEST hard tail; they never enter model inference. Classification: **R-D — memorization / transfer artifact** for E and F.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned (hashes re-verified against G/L/M/N/O/P/Q metadata)
**Data:** `experiments/experiment_r_candidate_representation.npz`, `experiments/experiment_r_candidate_representation_meta.json`, `experiments/experiment_r_results.json`
**Runners:** `experiments/run_experiment_r.py`, `experiments/analyze_experiment_r.py`, tests `tests/test_experiment_r.py`

---

## 1. Title / Date / Status

Experiment R — Candidate-Representation Information Test on the Class-A Tail. Measurement-only, frozen; **R-D (memorization / transfer artifact)**. No intervention; no Phase 2; Q's Q-C left untouched as historical evidence.

## 2. Executive Answer

Q found only weak recoverability of the true written value from the frozen candidate vector `C_t` on the Class-A (candidate-wrong) population, and could not tell whether that reflects a genuine information limit of `C_t` or an artifact of probe capacity / training on the mostly-easy distribution. R resolves the ambiguity with a 2-D design — capacity (linear, MLP-16/32/64/128) × training distribution (ordinary, size-matched ordinary, designated hard-tail, augmented hard-tail) — evaluated on the **same** untouched canonical-TEST hard tail.

**The weak recoverability survives controlled changes and is not a capacity or (size-matched) distribution artifact; the apparent augmented-hard gains are memorization.** On E's hard tail (n≈360/seed):

* **Capacity alone does nothing.** Under ordinary training the hard-test accuracy is **0.244 → 0.236** across the whole capacity family (train 0.89 → 0.99): added capacity fits the training data, not unseen hard transitions.
* **Size-matched hard-tail training is not material.** Training directly on the designated hard-tail pool (n=144/seed; the frozen model memorized its train split, so almost no candidate errors exist there) gives hard-test **0.213 → 0.248** vs the size-matched ordinary control **0.179 → 0.189** — Δ = **+0.068** (5/5 seeds), below the +0.15 materiality bar.
* **An augmented i.i.d. hard pool lifts apparent accuracy but not by recoverability.** Training on an i.i.d. hard pool disjoint from train/val/test (n≈1700/seed) raises hard-test to **0.396 → 0.480** (Δ vs ordinary **+0.225**), which would look like distribution-sensitive recoverability — **except that it does not generalize to unseen operand combinations**: a probe fit while holding out 20 operand pairs scores **0.176** on held-out-pair hard transitions vs **0.493** on seen-pair hard transitions (gap **0.317**; chance = 0.10), while training accuracy climbs to **0.97**. The apparent gain is operand-combination memorization, not arithmetic recoverability.
* Capacity, wrap/no-wrap, repeated/isolated, raw-sum-vs-modular, candidate-correct controls and E-vs-F all behave consistently (details below).

Under the pre-registered rule this is **R-D — memorization / transfer artifact**: apparent capacity/distribution gains cannot be interpreted as recoverability. This **strengthens the information-limit interpretation** (Q's Q-C conclusion survives: no genuine candidate-side recoverability is demonstrated under the tested probe family), and **no inference-time intervention is justified**. This remains an **observational** result.

## 3. Existing Baseline

* **G–I**: bit-exact commit; a real candidate→commit residual (~5.5% of candidate-correct adds).
* **L/M/N/O**: the third basin is latent placement (B); a systematic **parity/modular** geometry (M-A), **state-alignment-dominant** (N-B); parity-direction removal is non-specific and harmful (**O-D**).
* **P**: the sequence-level first-error budget is **candidate-generation-dominated** (A 59.7% T=10 / 67.2% T=20).
* **Q**: on the dominant Class-A population the frozen candidate vector is only **weakly recoverable** — linear 0.267/0.253, nonlinear 0.233/0.261, no nonlinear gain, wrap ≈ no-wrap, raw-sum not easier; **Q-C**. Q explicitly flagged that its probes were trained on the mostly-easy TRAIN distribution and evaluated on the hard tail. R tests whether that limitation survives.

## 4. Research Question

Does Q's weak recoverability of `C_t` reflect a genuine information limitation of the frozen candidate representation, or is it partly an artifact of probe capacity and of training probes on the mostly-easy distribution before evaluating them on the hard Class-A tail?

## 5. Hypotheses

* **H-A|R-A**: ordinary-vs-hard training distribution explains part of the weakness → distribution artifact.
* **H-B|R-B**: bounded capacity was the limiting factor → capacity-limited recoverability.
* **H-C|R-C**: neither helps; recoverability stays weak → robust information limit.
* **H-D|R-D**: larger probes/hard training produce gains that do not generalize (memorization).

## 6. Preregistration (declared before aggregate inspection)

Probe family (**fixed, not extended after results**): `P1` linear ridge (closed-form, class-balanced, λ∈{1e-2,1e-1,1,10,100} on val); `P2/P3/P4/P5` = MLP 212→{16,32,64,128}→10 (tanh, Adam lr 3e-3, weight-decay 1e-2, 200 epochs, early stop on val, fixed seed, balanced class weights). Training distributions: **D0** ordinary train (1500, Q protocol), **D0m** size-matched ordinary (= |hard-train|), **D1** designated hard-train (all candidate-wrong train ADD), **D1x** augmented i.i.d. hard pool. Evaluation: the canonical-TEST hard tail. `best_hardtest` = max over bounded probes. Decision bars: Δ_distribution ≥ **0.15**; recoverability **0.50**; capacity gain **0.10**; held-out-operand gap **0.20** with held-out accuracy ≤ 0.25 for memorization. Inference: five E checkpoints are independent units; F is a control.

## 7. Frozen Components

Recurrent core, candidate network, encoder, gate, commit, head, benchmark, optimizer, loss. **Nothing modified.** All model execution under `torch.no_grad()`; checkpoints SHA-256 verified before/after; the frozen `W,b` are stored and the frozen head is reproduced exactly in the tests. The only fitted objects are analysis-only probes — separate objects, never wired into `forward`, `decision_head`, or any rollout.

## 8. Dataset / Split

The **DisjointSplitManager** splits: TRAIN 6,000 / VAL 1,200 / canonical TEST 1,200×10 (stream `90042+seed`), sequence-disjoint by construction. A **fresh i.i.d. stream** (seed base `810000+seed`, 5,000 sequences) supplies the augmented hard pool; it is verified sequence-disjoint from train/val/test (hash check in the tests). Probes fit on the regime's train, tune on its val, evaluate **once** on TEST.

## 9. Hard-Tail Population Definition

Q's Class-A definition, unchanged: an **ADD transition with frozen candidate written-register decode ≠ true written value**. Frozen candidate written accuracy on canonical-TEST ADD is ~0.86, so the hard tail is ~360/seed (E 448/327/363/330/337; F 367/347/358/328/423). The designated **hard-train** pool is the class-A subset of the TRAIN split; the frozen model has memorized that split (0.991 ADD accuracy there), so it contains only **144** (E) / 239–509 (F) rows per seed — a data-scarcity fact reported, not hidden.

## 10. Ordinary-vs-Hard Probe Regimes

| Regime | Train source | n/seed (E) | Val | Purpose |
|---|---|---:|---|---|
| D0 | ordinary train (all ADD) | 1,500 | ordinary val | Q reproduction |
| D0m | ordinary train (first |hard-train|) | 144 | ordinary val | size-matched control |
| D1 | designated hard-train | 144 | hard val | spec hard-tail training |
| D1x | augmented i.i.d. hard pool | 1,696 | hard val | data-adequate hard training |

## 11. Capacity Sweep

Hard-test accuracy by (regime × capacity), E (mean over 5 seeds):

| Probe (params) | D0 | D0m | D1 | D1x |
|---|---:|---:|---:|---:|
| linear (2,130) | 0.244 | 0.179 | 0.213 | 0.396 |
| MLP-16 (3,578) | 0.244 | 0.170 | 0.251 | 0.453 |
| MLP-32 (7,146) | 0.238 | 0.194 | 0.261 | 0.463 |
| MLP-64 (14,282) | 0.237 | 0.185 | 0.263 | 0.473 |
| MLP-128 (28,554) | 0.236 | 0.189 | 0.248 | 0.480 |

Train accuracy: D0 0.89→0.99; D1x 0.57→0.97 (memorization). Val (D1x): 0.374→0.476. F is analogous (D1x 0.373→0.441).

## 12. Hard-Test Recoverability

Best bounded hard-test accuracy: **0.480 (D1x/MLP-128, E)** and **0.441 (F)** — below the 0.50 recoverability bar. Capacity gain (best nonlinear − linear): D0 **−0.006**, D0m +0.015, D1 +0.048, D1x **+0.066** — small everywhere. The best probes remain in the weak regime.

## 13. Matched-Difficulty Controls

A deterministic nearest-neighbour control matched Class-A rows to candidate-correct rows on frozen-only metadata (wrap, repeated, t, |margin|, ‖C_t‖, ‖C_t−S_prev‖), one-to-one. Matched candidate-correct decodability under D1x (MLP-32): **0.571** vs Class-A 0.463; under D0: 0.93. The Class-A deficit persists after matching (the matched control is easier even when difficulty-matched), so the hard tail is not merely "a difficult corner of state space" — but nor is it recovered by capacity.

## 14. Held-Out Operand/Context Generalization

Holding out 20 operand pairs (b∈{0,5}; all 10 target values represented) from probe fitting and evaluating on hard-test rows with held-out pairs (D1x, E):

| probe | held-out pairs | seen pairs | gap |
|---|---:|---:|---:|
| linear | 0.148 | 0.422 | 0.274 |
| MLP-32 | **0.176** | **0.493** | **0.317** |

Held-out-pair accuracy collapses to near chance (0.10) while seen-pair accuracy approaches 0.5 — the signature of **operand-combination memorization** rather than arithmetic recoverability. (Under D0 the gap is small, ~0.03–0.09, because the ordinary probe is weak everywhere.) All benchmark operand pairs are enumerated in every split, so a natural held-out-operand split is *constructed* by exclusion; this is the required generalization control.

## 15. Raw-Sum vs Modular

Under D1x, hard-test accuracy for the modular target `(a+b) mod 10` vs the raw sum `a+b` (19-way): linear **0.396 vs 0.184**, MLP-32 **0.463 vs 0.440**. The raw sum is **not** easier to decode — as in Q, there is no evidence of arithmetic present-but-mis-organised.

## 16. Wrap / No-Wrap / Repeated Analysis

D1x / MLP-128 hard-test (E): wrap **0.542**, no-wrap **0.432**, repeated **0.375**, isolated **0.560**. Wrap is modestly easier and repeated ADD modestly harder, but all remain in the weak band; no subtype reaches recoverability, and the same ordering holds for F (0.530 / 0.374 / 0.355 / 0.506).

## 17. Candidate-Correct Controls

Candidate-correct ADD decodability: **D0** 0.844 / 0.945 / 0.948 (linear/MLP-32/MLP-128) — strong, as expected. **D1x** 0.496 / 0.562 / 0.524 — degraded: a probe trained only on hard examples specializes and loses calibration on the easy majority. This neither indicates pathological instability nor supports recoverability; it shows the hard-trained gains are hard-specific and do not reflect a broadly better readout.

## 18. E vs F

| hard-test (D1x), paired | E | F | diff | p |
|---|---:|---:|---:|---:|
| linear | 0.396 | 0.373 | +0.023 | 0.25 |
| MLP-32 | 0.463 | 0.428 | +0.035 | 0.19 |
| MLP-128 | 0.480 | 0.441 | +0.039 | 0.11 |

E and F agree (both **R-D**; all paired differences n.s.). The held-out-operand collapse is present in both (F held-out 0.166 vs seen 0.461).

## 19. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a control, never pooled. Every fraction is reported per seed with mean ± SD; Δ_distribution/Δ_capacity use paired seed-level differences (5 seeds). Probe capacity is a fixed, pre-declared family — every capacity is reported, including failures; the grid is not extended after seeing results. Transition counts are descriptive.

## 20. Leakage / Validity

Enforced and tested: no sequence overlap between ordinary TRAIN/VAL/TEST; the augmented stream is sequence-disjoint (hash-verified); probe fitting never sees the hard-test rows; λ / early-stopping / capacity use only the regime's val; test accuracy is never used to select anything; class-balanced weights prevent majority-class exploitation (testing `test_probe_determinism_and_no_test_tuning` shows permuting test labels leaves the fitted λ and train accuracy unchanged); probes never enter model inference; checkpoints and the readout are unmodified. A deliberate leakage failure path (fitting on the test set) is included and shown to inflate test accuracy, and the disjoint-split identity-overlap guard is tested. Collector and analysis are byte-identical across two runs.

## 21. Causal Status

**Observation only.** R measures recoverability under analysis-only probes across a bounded capacity/distribution grid. It does not show that candidate generation *causes* the failures, that the representation is incapable of arithmetic, that the model "needs more dimensions", or that a better/nonlinear readout would solve the task. The wording used is "target information was / was not recoverable under the tested probe family", "recoverability did not increase with bounded capacity", "hard-tail training did not produce a generalizing improvement", "generalization to unseen operands was not preserved". The finding **strengthens the information-limit interpretation** and **weakens** the distribution-artifact and capacity-limited explanations, but it is not a causal proof.

## 22. Decision Classification

**R-D — memorization / transfer artifact** (E primary; F agrees).
* Held-out-operand gap 0.317 (E) / 0.294 (F) with held-out accuracy 0.176 / 0.166 ≤ 0.25 → the D1x gains are operand-combination memorization.
* Size-matched distribution effect +0.068 / +0.111 < the 0.15 bar → not R-A.
* Best bounded probe 0.480 / 0.441 < 0.50 and capacity gain 0.066 / 0.055 < 0.10 → not R-B.
* Consequently the R-D evidence means **apparent capacity/distribution gains cannot be interpreted as recoverability**, which **coincides with R-C's conclusion** (no genuine recoverability) and strengthens the information-limit reading. No subtype or E/F divergence rose to R-MIX.

## 23. What R Establishes

* A deterministic, leakage-controlled **capacity × distribution recoverability matrix** for the frozen `C_t` on the Class-A hard tail, reproducing Q's D0 baseline (~0.24).
* That **bounded capacity does not help** (D0 flat 0.244→0.236; train→0.99).
* That **size-matched hard-tail training is not material** (+0.068 E).
* That an **augmented hard pool lifts apparent accuracy to 0.48 but the gain does not survive held-out operand combinations** (held-out 0.176 vs seen 0.493) — a **memorization/transfer artifact**.
* That **raw-sum is not easier than modular**, wrap ≈ no-wrap (modestly easier), repeated ADD modestly harder; candidate-correct decodability degrades for hard-specialized probes; and E and F agree.

## 24. What R Does Not Establish

* It does **not** establish causality or that the frozen representation contains no arithmetic information.
* It does **not** prove a specific architecture is required, or that more dimensions/neurons would help, or that an inference-time intervention would work.
* It does **not** exclude the possibility that some *other* probe family (outside the pre-declared bounded family) could recover more; the tested family is bounded by design.
* It does **not** explain *why* the candidate representation is weak (the projection/geometry vs deeper computation distinction is untested here) — that is the next experiment.

## 25. Final Answers

1. **Does increasing bounded capacity recover substantially more of the Class-A target?** **No** — D0 0.244→0.236; the D1x gain (0.396→0.480) is memorization.
2. **Does training directly on the hard candidate-wrong distribution improve held-out hard-test?** **Only the augmented pool** (D1x 0.48); the designated size-matched hard tail gives only **+0.068** (E) — not material.
3. **Does any improvement survive sequence-disjoint testing?** At the sequence level yes (D1x tests on disjoint canonical-TEST sequences).
4. **Does it survive held-out operand/context testing?** **No** — 0.176 vs 0.493 seen (gap 0.317).
5. **Does the effect persist in wrap?** Wrap is modestly easier (0.542) but in the same weak band.
6. **Does it persist in repeated ADD?** Present but weaker (0.375 vs isolated 0.560).
7. **Does raw-sum decoding become easier under any regime?** **No** (0.440 vs 0.463 modular, D1x/MLP-32).
8. **Do candidate-correct matched controls remain strongly decodable?** Under D0 yes (0.93–0.95); hard-specialized D1x probes degrade to ~0.56.
9. **Is there evidence of memorization?** **Yes** — train 0.97 vs held-out-operand 0.18, and capacity helps training but not held-out.
10. **Do E and F agree?** **Yes** (both R-D; paired p ≥ 0.11).
11. **Which does the evidence strengthen?** **Memorization / transfer artifact** (R-D), which coincides with the **information-limitation** reading; it does *not* support probe-distribution artifact or capacity limitation.
12. **Is candidate-side recoverability now demonstrated?** **No.**
13. **Is an inference-time intervention justified?** **No** — held-out recoverability remains weak; the apparent gains fail generalization. No intervention; no Phase 2.

## 26. Exactly ONE Next Experiment

R-D shows the augmented-hard gains are memorization and that **no genuine recoverability from `C_t` is demonstrated**, but it does not localise *where in the candidate computation* the arithmetic information is lost. The one justified next experiment is a **measurement-only representation investigation**, not an intervention:

**Experiment S — controlled candidate-internal representation investigation (measurement-only).** Extend R's capacity × distribution × held-out-operand design from `C_t` to the **frozen candidate network's internal activations** (forward-only, `torch.no_grad()`, no weight change): probe the candidate MLP's hidden features with the same family. If the sum is decodable from an internal feature but not from `C_t`, the loss is in the candidate's **output projection/geometry**; if no internal feature encodes it, the limitation is deeper in the candidate **computation**. S uses the same hard tail, the same held-out-operand control, and the same pre-registered capacity family. It conditions whether any candidate-side intervention is even theoretically available — and it does **not** implement a repair, select an architecture, or begin Phase 2. The governing loop remains MEASURE → INTERPRET → KEEP/KILL → NEXT CONTROLLED TEST.

## Required Figures

`docs/reports/figures/experiment_r_*.png` (14): `capacity_vs_hardtest`, `ordinary_vs_hard`, `capacity_distribution_heatmap`, `train_val_hardtest`, `hard_vs_matched`, `heldout_operand`, `wrap_nowrap`, `repeated_isolated`, `rawsum_vs_modular`, `per_seed`, `params_vs_heldout`, `distribution_shift`, `confusion_best`, `candidate_correct_control`.

## Tests

`tests/test_experiment_r.py` — **24 tests**: collector determinism / no-grad / no-mutation; checkpoint SHA + immutability; exact candidate-vector reproduction; sequence-split disjointness; hard-tail construction and hard-split disjointness; augmented-stream disjointness (hash); ordinary/hard regime construction; target balancing; capacity ordering + parameter accounting; probe determinism and no-test-tuning; deliberate leakage failure path; held-out operand split and nearest-neighbour matching; wrap/no-wrap and repeated/isolated classification; raw-sum target generation; candidate-correct controls and metric separation; capacity × distribution matrix completeness; regime split sizes; hard-tail minority + raw-sum range; no probe insertion into model inference; E/F alignment; decision validity; malformed-data rejection; JSON determinism; figure creation; historical immutability.

## Reproduction

```bash
python experiments/run_experiment_r.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_r_candidate_representation.npz
python experiments/analyze_experiment_r.py \
    --npz experiments/experiment_r_candidate_representation.npz \
    --output experiments/experiment_r_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_r
pytest tests/test_experiment_r.py -v      # 24 tests
pytest tests/                             # full suite
```

Determinism: collector arrays bit-identical across two runs; analysis JSON identical (excluding the timestamp). Checkpoint SHA-256 verified before/after. No model, readout, gate, commit, or candidate weight is modified.

## Scientific Guardrails

Measurement-only; observational. The report says "target information was not recoverable under the tested probe family", "recoverability did not increase with bounded capacity", "hard-tail training did not produce a generalizing improvement", "generalization to unseen operands was not preserved", and "the result strengthens the information-limit interpretation". It never says the representation definitely contains no arithmetic, that the architecture cannot calculate, that the model needs more dimensions, that training is proven to be the cause, or that a better probe implies a better readout. The observed budgets (P), recoverability (Q), and intervention leverage remain separate. Q is **not** edited; R is appended as new evidence. No Phase 2. Negative and inconsistent findings (flat capacity curves, chunk-specific degradation) are preserved.
