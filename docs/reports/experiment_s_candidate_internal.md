# RE:Reason — Experiment S: Controlled Candidate-Internal Representation Investigation

**Date:** October 7, 2026
**Status:** Completed — Measurement only (frozen checkpoints; no model training; no optimizer on model weights; no architecture/candidate/recurrent-core/encoder/gate/commit/readout modification; no inference-time intervention; no branching/verifier/oracle; no counterfactual recursive rollout; no J/K/O/P/Q/R rerun; no Phase 2). Analysis-only probes are fit on a regime's train split, tuned on its validation split, and evaluated once on the untouched canonical-TEST hard tail; they never enter model inference. Classification: **S-C — computation-limited** for E and F.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned (hashes re-verified against G/L/M/N/O/P/Q/R metadata)
**Data:** `experiments/experiment_s_candidate_internal.npz`, `experiments/experiment_s_candidate_internal_meta.json`, `experiments/experiment_s_results.json`
**Runners:** `experiments/run_experiment_s.py`, `experiments/analyze_experiment_s.py`, tests `tests/test_experiment_s.py`

---

## 1. Title / Date / Status

Experiment S — Controlled Candidate-Internal Representation Investigation. Measurement-only, frozen; **S-C (computation-limited)**. No intervention; no Phase 2. Q and R reports are untouched; S is appended as new evidence.

## 2. Executive Answer

Q measured that the true written value is only weakly recoverable from the frozen candidate vector `C_t` on the Class-A population (Q-C); R showed that survives bounded capacity and size-matched hard-tail training and that the only regime raising apparent accuracy is operand-combination memorization (R-D). Neither could say **where in the candidate computation** the information is lost — an output-projection/geometry limit (present internally, absent from `C_t`) or a deeper candidate-computation limit (absent throughout).

S probes the **actual internal activations** of the frozen candidate network — captured read-only with forward hooks inside the existing `forward_with_states` — under R's capacity × distribution design, on the **same** untouched canonical-TEST hard tail, with held-out-operand controls.

**No internal activation provides generalizing recoverability; the loss is not localised to the output.** Under the signal-bearing hard-trained regime (D1x), recoverability *increases weakly* from the candidate input toward `C_t` rather than declining (`cand_input` 0.246 → `h1_pre` 0.378 → `h1_post` 0.427 → `out_pre` 0.432 → `C_t` 0.463; mlp32, mean over 5 E seeds), so **`C_t` is the single most "recoverable" representation probed, not the weakest**. No internal stage materially beats `C_t` (best internal `out_pre` = 0.432; paired gain over `C_t` **−0.031**, 0–1/5 seeds positive). Under ordinary training (D0) the same chain is flat-to-weak (0.128 → 0.217 → 0.215 → 0.220 → 0.238). Crucially, **held-out-operand generalization fails at every stage**: D1x-seen accuracy reaches 0.47–0.49 at the later stages but held-out-operand accuracy is only **0.045–0.176** (all ≤ 0.25, seen−heldout gaps 0.24–0.35). The preactivation/postactivation comparisons show **no drop across a nonlinear boundary** (GELU or tanh; postactivations are if anything slightly *more* decodable). `C_t` reproduces R exactly (D0 linear 0.244, D0/mlp32 0.238, D1x/mlp32 0.463).

Under the pre-registered rule this is **S-C — computation-limited**: no tested internal activation materially exceeds the established weak regime, bounded probe capacity does not rescue held-out performance, hard-trained probes still fail held-out operands, and no internal "island" of robust information appears. The result is **consistent with the loss occurring before or throughout the candidate computation**, not at the output projection alone; it is **observational**, **not causal**, and **no intervention is justified**.

## 3. Existing Baseline

* **G–I**: bit-exact commit; a real candidate→commit residual (~5.5% of candidate-correct adds).
* **L/M/N/O**: the off-path third basin is latent placement (B), a systematic **parity/modular** geometry (M-A), **state-alignment-dominant** (N-B); parity-direction removal is non-specific and harmful (**O-D**).
* **P**: the sequence-level first-error budget is **candidate-generation-dominated** (A 59.7% T=10 / 67.2% T=20).
* **Q**: on the Class-A population the frozen `C_t` is only **weakly recoverable** — linear 0.267/0.253, nonlinear 0.233/0.261, no nonlinear gain, wrap ≈ no-wrap, raw-sum not easier; **Q-C**.
* **R**: bounded capacity is flat (D0 0.244→0.236), size-matched hard-tail training is immaterial (+0.068), and the augmented-hard gain (0.480) is **operand-combination memorization** (held-out 0.176 vs seen 0.493, gap 0.317) — **R-D**. R leaves unresolved **where in the candidate computation** the information is lost; S measures exactly that on the candidate network's internal activations.

## 4. Research Question

> Does the true ADD written value become recoverable from an **internal activation** of the frozen candidate network even though it is not recoverable from the final candidate vector `C_t`?

S distinguishes (i) **S-A** information survives internally while `C_t` loses it (output/geometry-localised), (ii) **S-B** recoverability declines progressively toward `C_t` (deeper limitation), (iii) **S-C** no internal recoverability anywhere (weak throughout), (iv) **S-MIX** incompatible stage/subtype conclusions. These are localisation labels, not causal claims.

## 5. Hypotheses

* **H-A|S-A**: an internal candidate activation contains generalizable target information while `C_t` does not.
* **H-B|S-B**: partial information at early stages declining monotonically toward `C_t`.
* **H-C|S-C**: no tested internal activation generalizes above the weak regime.
* **H-D|S-MIX**: different stages or arithmetic subtypes disagree.

## 6. Preregistration (declared before aggregate inspection)

**Activation set (from the actual implementation — see §8):** `S_prev(212)`, `f_gate(212)`, `h_reset(212)`, `cand_input(468)`, `h1_pre(146)`, `h1_post(146)`, `out_pre(212)`, `C_t(212)`. Primary candidate chain: `cand_input → h1_pre → h1_post → out_pre → C_t`. All 8 stages are reported; no stage is dropped.

**Probe family (R's, fixed, not extended after results):** `P1` linear ridge (closed-form, class-balanced, λ∈{1e-2,1e-1,1,10,100} on val); `P2..P5` MLP 16/32/64/128 (tanh, Adam 3e-3, weight-decay 1e-2, 200 epochs, early stop on val, fixed seed, balanced class weights).

**Reduced capacity design (predeclared):** ALL stages × {linear, mlp16, mlp32}; subset `{cand_input, h1_post, out_pre, C_t}` × {mlp64, mlp128}.

**Distributions (R's):** `D0` ordinary train (1500/seed), `D0m` size-matched ordinary (= |hard-train|), `D1` designated hard-train, `D1x` augmented i.i.d. hard pool. **Evaluation is always the same untouched canonical-TEST Class-A hard tail.**

**Held-out operand protocol:** 20 pairs `{(a,b): b∈{0,5}}` (all 10 targets represented); the probe is fit excluding them, evaluated once on hard-test rows with held-out vs seen pairs (D0 and D1x, linear + mlp32).

**Localization verdict (checked in order; bars are declared conventions):**
* **S-MIX** if subtype spread |wrap−no-wrap| ≥ 0.15 at ≥2 stages, or E/F verdict categories disagree.
* **S-A (output/downstream)** if an internal chain stage A (not `C_t`) beats `C_t` by ≥ 0.10 (D1x/mlp32) reproduced ≥4/5 seeds, with held-out-operand accuracy ≥ 0.30 and ≥ `C_t` held-out + 0.10, and seen−heldout gap < 0.20.
* **S-B (progressive)** if the D1x/mlp32 chain scalar is ~non-increasing (tol 0.05) with max adjacent *drop* ≥ 0.10 and earliest stage ≥ 0.30.
* **S-C (computation-limited)** if max internal gain over `C_t` < 0.10 and max internal held-out ≤ 0.25.
* else **S-MIX**.

Recoverability = `(acc − chance)/(1 − chance)` (chance = 0.10) is reported alongside raw accuracy (on Class-A the frozen accuracy is 0, so recoverability = acc/0.9). Inference: five E checkpoints are independent units; F is a control.

## 7. Frozen Components

Recurrent core, candidate network, observation encoder, gate, commit rule, head, benchmark, optimizer, loss. **Nothing modified.** All model execution under `torch.no_grad()`; checkpoints SHA-256 verified before and after. Internal activations are captured with **read-only forward hooks** registered on the existing `cand_mlp` submodules (and removed in a `finally`); they cannot alter the frozen computation (verified bit-exact against a manual recomputation in the tests). The only fitted objects are analysis-only probes; they are separate Python objects and never wired into `forward`, `decision_head`, or any rollout.

## 8. Candidate Architecture / Activation Map

The frozen Model E/F candidate network is `NonlinearCandidateMGUCore` (**not** a two-hidden-layer MLP):

```
f_t        = sigmoid(W_f_ih e_t + W_f_hh S_(t-1) + b_f)     # gate                (212)
h_reset    = f_t ⊙ S_(t-1)                                  # pre-gated state     (212)
cand_input = [h_reset, e_t]                                 # candidate input     (468)
h1_pre     = cand_mlp[0](cand_input)                        # hidden-1 preact     (146)
h1_post    = GELU(h1_pre)                                   # hidden-1 postact    (146)
out_pre    = cand_mlp[2](h1_post)                           # candidate logits    (212)
C_t        = tanh(out_pre)                                  # final candidate     (212)
```

`cand_mlp = Sequential(Linear(468→146), GELU, Linear(146→212))` — **one** hidden layer. The predeclared stage map above is read from this implementation; there is no hidden-2. Captured stages are exactly those in the module: `cand_input` (the layer-0 input), `h1_pre` (layer-0 output), `h1_post` (GELU output), `out_pre` (layer-2 output), `C_t = tanh(out_pre)`; plus the candidate-adjacent reference stages `S_prev`, `f_gate`, `h_reset`. The activation set and its dimensions are validated against the actual modules in `tests/test_experiment_s.py`.

## 9. Dataset / Split

The **DisjointSplitManager** splits (no sequence overlap by construction): TRAIN 6,000 / VAL 1,200 / canonical TEST 1,200×10 (stream `90042+seed`), sequence-disjoint. A fresh i.i.d. stream (base `810000+seed`, 5,000 sequences) supplies the augmented hard pool; it is verified sequence-disjoint from train/val/test. `S_prev` is the pre-transition committed state (`S_{t-1}`); `seq` is the sequence index; the collector is chunked (512 sequences) to bound peak memory. Collection is bit-identical across two runs.

## 10. Class-A Population

Q's Class-A definition, unchanged: an **ADD transition whose frozen candidate written-register decode ≠ the true written value**. Per-seed n (E): 448 / 330 / 363 / 327 / 337 (mean 361); (F): 367 / 347 / 358 / 328 / 423 (mean 365). Test ADD totals reproduce R exactly (15,763 per model over 5 seeds); the designated hard-train split is the Class-A subset of TRAIN (E 747 total; F 1,827). Matched candidate-correct controls are constructed by deterministic nearest-neighbour matching on frozen-only metadata.

## 11. Frozen `C_t` Baseline

Reproduced exactly from the frozen `W,b` stored in the artifact: E `C_t` D0 linear **0.244**, D0/mlp32 **0.238**, D1x/mlp32 **0.463** (F: 0.230 / 0.259 / 0.428). These match R's published 5-seed means (0.244 / 0.238 / 0.463) — S's `C_t` row is a correct baseline bridge to R. The frozen candidate head is 0% on Class-A by construction. Candidate-correct decodability (E): D0 0.945 → D1x 0.562 (hard-specialised degradation, as in R).

## 12. Internal Activation Recoverability

Hard-test exact accuracy, mean over 5 E seeds (mlp32):

| Stage | Dim | D0/mlp32 | D0m | D1 | D1x | D1x/lin | D1x/mlp128 |
|---|---:|--:|--:|--:|--:|--:|--:|
| S_prev | 212 | 0.085 | 0.102 | 0.130 | 0.167 | 0.136 | – |
| f_gate | 212 | 0.102 | 0.101 | 0.145 | 0.213 | 0.155 | – |
| h_reset | 212 | 0.138 | 0.100 | 0.161 | 0.239 | 0.189 | – |
| cand_input | 468 | 0.128 | 0.114 | 0.155 | 0.246 | 0.190 | 0.322 |
| h1_pre | 146 | 0.217 | 0.122 | 0.167 | 0.378 | 0.190 | – |
| h1_post | 146 | 0.215 | 0.130 | 0.218 | 0.427 | 0.373 | 0.474 |
| out_pre | 212 | 0.220 | 0.162 | 0.236 | 0.432 | 0.377 | 0.488 |
| C_t | 212 | 0.238 | 0.194 | 0.261 | **0.463** | 0.396 | 0.480 |

Chance 0.10; recoverability bar 0.50. **No internal stage reaches the bar or materially beats `C_t`.** Under D1x the best internal stage (`out_pre`) is 0.432 vs `C_t` 0.463 (paired gain **−0.031**, 0/5 seeds positive; `h1_post` −0.035, 1/5; `h1_pre` −0.084, 0/5; `cand_input` −0.217, 0/5). Macro accuracy tracks exact accuracy (D1x/mlp32 `C_t` 0.461; `out_pre` 0.426). Validation accuracy is highest at `C_t` (0.451); train accuracy is high everywhere (D0/mlp32 → 0.99), i.e. the probes fit training, not the unseen hard tail.

## 13. Capacity × Distribution

Capacity (E, D1x) rises only weakly and never to recovery: `cand_input` 0.190→0.322 (lin→mlp128), `h1_post` 0.373→0.474, `out_pre` 0.377→0.488, `C_t` 0.396→0.480. Under D0 the curves are flat (e.g. `C_t` 0.244→0.236), reproducing R. The hard-tailed D1x pool raises all stages but — as §14 shows — by memorization. Δ_D (D1x − D0, mlp32) is roughly constant across stages (≈ +0.10 for the later stages), so the distribution sensitivity is not stage-localised; it is a property of the whole chain. No stage shows a capacity gain that is both material and generalization-preserving.

## 14. Held-Out Operand Generalization

Probe fit excluding 20 operand pairs, evaluated on Class-A hard-test rows with held-out vs seen pairs (mlp32, E):

| Stage | D1x held-out | D1x seen | gap | D0 held-out | D0 seen |
|---|--:|--:|--:|--:|--:|
| cand_input | 0.045 | 0.286 | 0.241 | 0.097 | 0.146 |
| h1_post | 0.131 | 0.478 | 0.347 | 0.166 | 0.255 |
| out_pre | 0.164 | 0.471 | 0.307 | 0.184 | 0.241 |
| C_t | 0.176 | 0.493 | 0.317 | 0.158 | 0.275 |

**Every stage collapses on held-out operands** (all held-out ≤ 0.25; gaps 0.24–0.35) — the same signature R found for `C_t`. There is **no internal "island"** where the hard-trained gain generalizes. F is analogous (`C_t` held-out 0.166 vs seen 0.461).

## 15. Raw-Sum vs Modular

Under D1x/mlp32 (E) the modular target is slightly *harder* at most stages (`C_t` 0.463 modular vs 0.440 raw; `h1_pre` 0.378 vs 0.342); under D0 the raw sum is slightly *easier* at `C_t` (0.267 vs 0.238). There is **no stage at which the raw arithmetic sum is clearly more recoverable** than the modular target — i.e. no evidence of arithmetic present-but-mis-organised for the modular label at any internal stage, consistent with Q/R. F behaves the same (`C_t` raw 0.435 vs modular 0.428, D1x).

## 16. Wrap / No-Wrap / Repeated Analysis

D0/mlp32 (E), per stage: wrap vs no-wrap spread is small everywhere (max 0.100 at `f_gate`; 0.016 at `C_t`), far below the 0.15 mixed bar. Wrap is modestly easier at `C_t` under D1x (0.526 vs 0.414) but in the same weak band. Repeated vs isolated: at `C_t` D0, repeated 0.182 vs isolated 0.280 (repeated harder); at the internal stages the two are within ~0.06. There is **no wrap-specific or repeated-specific internal collapse** that would single out one arithmetic sub-mechanism.

## 17. Preactivation vs Postactivation

The GELU (hidden) and tanh (output) boundaries do **not** destroy recoverability; if anything the postactivations are slightly more decodable (D1x/mlp32, E): `h1_pre` 0.378 → `h1_post` 0.427 (**+0.049**); `out_pre` 0.432 → `C_t` 0.463 (**+0.031**). Under D0: `h1_pre` 0.217 → `h1_post` 0.215 (−0.002); `out_pre` 0.220 → `C_t` 0.238 (+0.018). **No nonlinear boundary is identified as the site of a recoverability drop** — consistent with S-C (weak throughout) and inconsistent with an output-nonlinearity localisation.

## 18. Candidate-Input Control

Probing the candidate-network input directly answers whether the information is present before candidate computation. It is **not**: `cand_input` is among the *weakest* stages (D0 0.128; D1x 0.246; D0m 0.114; held-out 0.045). The pre-candidate state `S_prev` (D1x 0.167) and the gate `f_gate` (0.213) are also weak, while `h1_pre` jumps to 0.378 under D1x — i.e. the first candidate layer *creates* most of the (memorised) separability rather than receiving it. Since the candidate input is itself weak, the candidate network does not appear to be receiving a strong recoverable representation from upstream. (This is a localisation statement only; it does **not** prove the encoder is causal.)

## 19. Activation Geometry

Class-A test-tail geometry (E, mean over seeds): `fisher` (between/within class variance) rises weakly toward the output (`cand_input` 0.033 → `C_t` 0.061); the Class-A↔candidate-correct centroid **cosine falls** toward the output (`cand_input` 0.983 → `out_pre` 0.633 → `C_t` 0.623) and the normalised centroid distance rises (0.189 → 0.871); activation norms differ strongly by stage (`cand_input` ~15, `h1_pre` ~33, `out_pre`/`C_t` ~12–45). So the Class-A mean rotates *away* from the candidate-correct mean as the candidate computation proceeds and the value classes remain only weakly separable (fisher ≪ 1) at every stage — Class-A does **not** become cleanly separable at an internal activation.

## 20. Information-Drop Localization

Using the D1x/mlp32 chain scalar there is **no decline**: the adjacent changes `cand_input→h1_pre` **+0.133**, `h1_pre→h1_post` +0.049, `h1_post→out_pre` +0.005, `out_pre→C_t` +0.031 are all non-negative (a weak *rise*). Under D0: +0.089, −0.002, +0.005, +0.018. The largest observed change is a **gain** at the candidate's first layer, not a drop. There is therefore **no localised loss to attribute** within the candidate network's activations: recoverability is weak throughout and, if anything, mildly *improves* toward the output.

## 21. E vs F

Paired (5 seeds) D1x/mlp32 E−F: `S_prev` +0.001 (p=0.89), `f_gate` −0.003 (p=0.84), `h_reset` +0.035 (p=0.10), `cand_input` +0.031 (p=0.08), `h1_pre` +0.018 (p=0.51), `h1_post` +0.038 (p=0.20), `out_pre` +0.023 (p=0.38), `C_t` +0.035 (p=0.19). **No stage differs significantly**; F reproduces the same increasing chain (`cand_input` 0.215 → `C_t` 0.428), the same held-out collapse, and the same **S-C** category. E and F agreement is explicit and `E_F_disagreement = false`.

## 22. Statistical Discipline / Multiple Comparisons

The five frozen E checkpoints are the independent units; F is a control, never pooled. Every fraction is reported per seed with mean ± SD; all stage comparisons are paired by seed. The activation set and capacity design were **predeclared**; every stage is reported, including negative results. The stage family is not searched for significance — the primary conclusion rests on the predeclared stage-level summary (max internal gain over `C_t`), and the many paired comparisons are reported descriptively. Recoverability (normalised) and raw accuracy coincide on Class-A (frozen = 0) and both are shown.

## 23. Leakage / Validity

Enforced and tested: no sequence overlap between TRAIN/VAL/TEST; the augmented stream is sequence-disjoint (hash-verified); hard-train/hard-val/hard-test are disjoint by construction (trainhard/valhard are Class-A subsets of the disjoint train/val splits; hard-test is the canonical test hard tail and is never used to fit or tune); λ / early stopping / capacity use only the regime's val; permuting the TEST labels leaves the fitted λ, train accuracy and val accuracy unchanged (no test tuning); held-out operand pairs are excluded from fitting and shown to have been present in the pool; probes never enter model inference (forward is bit-identical before/after probe fitting); `W,b` are stored and reproduced from the checkpoint; a deliberate leakage failure path (fit on test) is included and shown to inflate test accuracy; the disjoint-split identity-overlap guard is tested; the collector is bit-identical across two runs; the analysis JSON is identical (excluding the timestamp) and all figures are byte-identical.

## 24. Causal Status

**Observation only.** S measures recoverability under analysis-only probes across a bounded capacity/distribution grid and a fixed activation map. It does **not** show that the candidate network computes or fails to compute arithmetic, that the output projection "destroys" the answer, that a hidden layer is a symbolic representation, or that adding/altering a layer would help. The wording used is "target information was / was not recoverable from this activation under the tested probe family", "recoverability was weakly higher at `C_t` than at any internal stage", "held-out-operand generalization was not preserved at any stage", "the result is consistent with the loss occurring before or throughout the candidate computation". No intervention is performed or licensed.

## 25. Decision Classification

**S-C — computation-limited** (E primary; F agrees).
* Max internal gain over `C_t` = **−0.031** (`out_pre`, D1x/mlp32) < 0.10 bar → not S-A.
* Chain is non-decreasing (max adjacent change +0.133, a gain) → not S-B.
* Max internal held-out accuracy = **0.176** ≤ 0.25 → no generalizing internal recoverability.
* Subtype spread < 0.15 at every stage (max 0.100) and E/F agree → not S-MIX.
* Interpretation: **no tested internal activation provides generalizing recoverability above the established weak regime; the final `C_t` is the most (weakly) recoverable stage, not an output bottleneck.** This is **consistent with a deeper candidate-computation limitation** (or with the information never being present in a probe-recoverable form upstream of the candidate network), and it does **not** support an output-projection/geometry localisation.

## 26. What S Establishes

* A deterministic, leakage-controlled **candidate-internal recoverability map** over the real `NonlinearCandidateMGU` candidate network (8 activation stages; single hidden layer), reproducing R's `C_t` baseline exactly.
* That **no internal candidate activation materially exceeds `C_t`** on the Class-A hard tail; the chain weakly *increases* toward the output under both D0 and D1x.
* That **held-out-operand generalization fails at every stage** (held-out 0.045–0.176; seen−heldout gaps 0.24–0.35) — the internal "gains" are the same memorization R found for `C_t`.
* That **no nonlinear boundary (GELU/tanh) is the site of a drop**; postactivations are slightly more decodable than preactivations.
* That the **candidate-network input is itself weak** (0.128 D0 / 0.246 D1x), so the candidate network is not receiving a strong recoverable representation.
* That the result is **robust across E and F**.

## 27. What S Does Not Establish

* It does **not** establish causality; there is no intervention.
* It does **not** show the candidate architecture "cannot" do modular arithmetic, nor that no representation (outside the predeclared bounded probe family) could encode it.
* It does **not** show that the loss occurs *in* the candidate network as opposed to upstream of it; `cand_input` is already weak, which is *consistent with* an upstream limitation but does not localise it (S did not probe the encoder's internals beyond the stages feeding the candidate).
* It does **not** prove a specific projection/geometry limit; it **rules out** the simplest output-projection localisation (internal stages are not more recoverable than `C_t`).
* It does **not** identify the upstream source of the weakness, and it licenses **no** candidate-side modification.

## 28. Final Answers

1. **First stage at which target information becomes recoverable?** None reaches the recoverability bar. A weak, non-generalizing signal appears from `h1_pre` onward under D1x (memorization); no genuine onset.
2. **More recoverable internally than from `C_t`?** **No** — `C_t` is the highest under D1x (0.463); best internal `out_pre` 0.432.
3. **Generalize to unseen sequences?** The evaluation is on disjoint canonical-TEST sequences (yes at the sequence level), but that is not sufficient — see 4.
4. **Generalize to unseen operand combinations?** **No** — best internal held-out 0.176 (≤ 0.25); `C_t` 0.176.
5. **Does the hard-trained improvement survive those controls?** **No** — the D1x chain rise collapses on held-out operands at every stage.
6. **Does increasing probe capacity materially improve internal recoverability?** Only weakly and without generalization (`cand_input` 0.190→0.322; `h1_post` 0.373→0.474; `C_t` 0.396→0.480); no stage reaches the bar or survives held-out operands.
7. **Largest observed decline across candidate stages?** There is **no decline**; the largest adjacent change is a **gain** (+0.133 at `cand_input→h1_pre`, D1x).
8. **Pre/post comparison identifies a nonlinear boundary drop?** **No** — postactivations are slightly *more* decodable (GELU +0.049; tanh +0.031, D1x).
9. **Is arithmetic information present at the candidate input?** **Weakly / not materially** — `cand_input` is among the weakest stages (D0 0.128; D1x 0.246).
10. **Does wrap differ from no-wrap at internal stages?** **No** — spread < 0.15 everywhere (max 0.100).
11. **Does repeated ADD differ from isolated ADD?** Only weakly and inconsistently (at `C_t` D0, repeated 0.182 vs isolated 0.280; elsewhere within ~0.06).
12. **Is the raw sum easier than the modular target at any stage?** **No consistent signature** — modular slightly harder under D1x, raw slightly easier under D0 at `C_t`.
13. **Do E and F agree?** **Yes** — all paired p ≥ 0.08; both S-C.
14. **What is the observed localisation?** **Weak throughout** (S-C) — not output/projection-side, not progressive loss.
15. **Is candidate-side recoverability genuinely demonstrated under held-out operand testing?** **No.**
16. **Does S justify an intervention?** **No.** Weak recoverability at every stage ⇒ under the conservative rule, **no intervention**; the next step is measurement upstream, not a candidate-side modification.

## 29. Exactly ONE Next Experiment

S-C (weak recoverability at every candidate stage) directs the next step **upstream**, not to the output or the candidate MLP: the candidate network does not receive a strong recoverable representation, so the remaining question is *where before the candidate the modular/operand information becomes inaccessible* — or whether it was ever present in a probe-recoverable form.

**Experiment T — upstream-representation localisation of the Class-A information (measurement-only, frozen checkpoints).** On the same untouched Class-A hard tail, with the same pre-registered probe family, the same D0/D0m/D1/D1x regimes and the same held-out-operand control, probe the frozen pipeline **upstream of the candidate network**: the observation encoder's inner activations (`e_t` and the encoder MLP's hidden layer), the raw pre-transition state `S_{t-1}`, the gate preactivation and the gate postactivation `f_t`, and the pre-gated state `f_t ⊙ S_{t-1}` — i.e. decompose `cand_input` into its three additive/concatenated sources and localise which of them fails. Measurement-only; no weight change, no architecture change, no intervention; it conditions whether any candidate-side limitation is even reachable from the current frozen interface.

## Required Figures

`docs/reports/figures/experiment_s_*.png` (16): `recoverability_by_stage`, `capacity_curves_by_stage`, `ordinary_vs_hard_by_stage`, `capacity_distribution_heatmap` and `capacity_distribution_heatmap_out_pre`, `heldout_operand_by_stage`, `seen_vs_heldout_gap`, `rawsum_vs_modular_by_stage`, `wrap_nowrap_by_stage`, `repeated_isolated_by_stage`, `preactivation_vs_postactivation`, `information_trajectory`, `candidate_correct_vs_classA_geometry`, `activation_geometry`, `per_seed_stage_recoverability`, `adjacent_information_drop`.

## Tests

`tests/test_experiment_s.py` — **21 tests**: frozen-checkpoint SHA/immutability; exact activation capture (bit-exact vs manual recomputation of every stage); deterministic capture; no-grad; no parameter mutation; activation dimensions matched to the actual model (single hidden layer); C_t→readout reproduction; W,b match the checkpoint; npz stage completeness; sequence-split and augmented-stream disjointness; hard-split candidate-wrong construction; hard-test = untouched canonical Class-A; probe determinism; λ from val only (no test tuning); deliberate leakage failure path; operand-pair leakage exclusion; no probe insertion into inference; every declared stage analyzed; no test-based activation selection (fixed predeclared stage/capacity sets); decision validity; JSON determinism; malformed-artifact rejection; figure creation; historical-report immutability.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_s.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_s_candidate_internal.npz
PYTHONPATH=src python experiments/analyze_experiment_s.py \
    --npz experiments/experiment_s_candidate_internal.npz \
    --output experiments/experiment_s_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_s
pytest tests/test_experiment_s.py -v      # 21 tests
pytest tests/                             # full suite
```

Determinism: collector arrays bit-identical across two runs; analysis JSON identical (excluding the timestamp); all 16 figures byte-identical. Checkpoint SHA-256 verified before/after. No model, readout, gate, commit, or candidate weight is modified. The analyzer fixes `torch.set_num_threads(8)` for speed and bit-reproducibility.

## Scientific Guardrails

Measurement-only; observational. The report says "the target was not recoverable from this internal activation under the tested probe family", "recoverability was weakly higher at `C_t` than at any internal stage", "generalization to unseen operands was not preserved at any stage", and "the result is consistent with the loss occurring before or throughout the candidate computation". It never says the candidate network computes or fails to compute arithmetic, that the final projection destroys the answer, that a hidden layer is a symbolic arithmetic representation, that adding a layer would help, or that the loss is *caused* by the output layer. Probe decodability is not exact state preservation; a failed bounded probe is not proof that no representation exists; the internal (D1x) gains that fail held-out operands are labelled memorization, not recoverability. Q and R are **not** edited; S is appended as new evidence. Seed is the independent unit (pooling across frozen readouts is invalid). No Phase 2. Negative and inconsistent findings (weak throughout, no internal gain, held-out collapse) are preserved.
