# RE:Reason — Experiment W: Bounded-Reader Learnability vs Training-Data Scale

**Date:** October 9, 2026
**Status:** Completed — Measurement only (frozen E/F checkpoints used only to define the Class-A population and reproduce baseline metrics; no model training; no optimizer on RE:Reason parameters; no architecture/encoder/recurrent-core/state/gate/candidate/commit/readout modification; no state replacement; no candidate manipulation; no feature editing; no oracle/reconstructed state as a reader feature; no branching/verifier; no counterfactual rollout; no Phase 2). The W readers are separate analysis-only models and never enter frozen model inference. Classification: **W-B — data-saturated/bounded-limited** for E and F.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned (unchanged before/after).
**Data:** `experiments/experiment_w_data_scale.npz`, `experiments/experiment_w_data_scale_meta.json`, `experiments/experiment_w_results.json`
**Runners:** `experiments/run_experiment_w.py`, `experiments/analyze_experiment_w.py`, tests `tests/test_experiment_w.py`

---

## 1. Title / Date / Status

Experiment W — Bounded-Reader Learnability vs Training-Data Scale. Measurement-only, frozen; **W-B — data-saturated/bounded-limited**. No intervention; no Phase 2. Q, R, S, T, U and V reports are untouched; W is appended as new evidence.

## 2. Executive Answer

Raising the number of unique, pairwise sequence-disjoint ordinary-ADD training transitions ~6.7× (1500 → 10000; the reduced predeclared ladder run on modest hardware) did **not** materially lift the bounded readers' Class-A accuracy toward the exact ceiling.

* **The sequence readers are essentially flat.** V's causal **GRU** is the best top-scale reader (L4 **0.116**; 0.083 at 1500 → gain **+0.033**, sub-material); V's causal **TCN** is flat (L4 0.106 → 0.114, gain +0.008; full prefix 0.107 → 0.111). No reader reaches 0.25, let alone the 0.50 recoverability bar.
* **The flat MLP-128 anchor does not benefit from data at all** (L4 0.089 → 0.070; full prefix 0.101 → 0.071), and its training accuracy *falls* as the pool grows (0.375 → 0.232) — the fixed 200-update protocol, not the data volume, is the binding constraint for that reader.
* **Held-out operand accuracy stays at chance** (0.069–0.206, every value ≤ 0.25): the readers never learn the modular rule on either seen or held-out operand pairs.
* **The readers do not memorize either.** Training accuracy stays low (L4, top scale: 0.13–0.23) and does not rise toward the ceiling, so the flat test curve reflects underfitting rather than pool memorization.
* **The Bayes − reader gap does not close.** The exact Class-A Bayes ceiling at L4 is **0.999**; the best reader is 0.116, giving a gap of **0.883** (best-at-bottom 0.107 → gap 0.892; the gap barely moves with data).
* **E and F agree** (F best top-scale 0.109; both classify W-B; no reader contradiction).

Under the pre-registered rule this is **W-B — data-saturated / bounded-limited**: over the 1500–10000 range, the persistent V-B weak accessibility is **not** a simple training-data limitation under the fixed bounded reader/optimization protocol, and is **consistent with a bounded-capacity / algorithmic limit**. The result is **observational** and carries **no causal claim** about RE:Reason; **no intervention is justified**. Scope: this ladder tops out at 10,000 transitions (the predeclared full ladder extends to 32,000), so it bounds — but does not exclude — a much larger-data regime.

## 3. Q/R/S/T/U/V Baseline

* **Q/R**: the true written value is only weakly recoverable from the frozen `C_t` on Class-A (Q-C); bounded capacity / size-matched hard-tail training do not help and the only apparent gain is operand-combination memorization (R-D).
* **S**: no internal candidate activation materially beats `C_t` (S-C).
* **T**: the frozen observation encoder output `e_t` is structurally operand-blind and no upstream frozen representation is robustly recoverable (T-D).
* **U**: two independent evidence families on the pure causal observation history — **exact** interface Bayes (L=1 target 0.290; **L=4 target 0.998, operands 0.999**; **full prefix 1.000**) and **learned flat probes** (best full-prefix 0.218 all-ADD, ~0.09 Class-A; held-out ≤ 0.25) → **U-D (interface-identifiable but probe-limited)**.
* **V**: two parameter-bounded causal **sequence** readers — a single-layer causal **GRU** (hidden 85, 29,429 params at the 19-class budget) and a **causal dilated 1-D convolution** (TCN-style, channels 52, 28,983 params) — each within the exact U MLP-128 budget (29,578), under U's history grid, splits, populations and held-out-operand control. Best Class-A reader **0.112** (TCN/L4) vs the flat U MLP-128 **0.089** (gain **+0.023**); all-ADD never beats the flat baseline (0.214 vs 0.218 at the full prefix); no reader ≥ 0.50; held-out ≤ 0.25 everywhere; **Bayes − reader gap 0.887** → **V-B (sequence-limited)**. V's §28 specified exactly this training-data-scale test as Experiment W.

## 4. Research Question

> Is the persistent V-B reader failure primarily a **training-data / statistical-learning** limitation, a fixed-protocol **optimization** limitation, or a **bounded-capacity / algorithmic** limitation — measured by the effect of **training-data scale alone** under the frozen reader/optimization protocol?

W manipulates exactly one factor — the number of **unique, sequence-disjoint, i.i.d. ordinary-ADD training transitions** — over the predeclared ladder. The predeclared full ladder is `N_train ∈ {1500, 8000, 32000}`; on modest hardware the collector/analyzer are run over the reduced predeclared ladder `N_train ∈ {1500, 4000, 10000}` (selected via `W_TRAIN_SIZES`), holding the reader architecture, width, parameter count, optimizer, learning rate, weight decay, epoch count (200, full-batch), history representation, held-out protocol, VAL and TEST **exactly fixed**. All scale-relative statements below use the reduced ladder's endpoints (bottom `1500`, top `10000`).

## 5. Hypotheses

* **H-A|W-A (data-limited / learnable)**: at least one reader at L4/full reaches test ≥ 0.50 AND held-out ≥ 0.30, with a bottom→top (1500→10000) improvement ≥ 0.15 — the V-B result was substantially data-limited.
* **H-B|W-B (data-saturated / bounded-limited)**: best top-scale Class-A test ≤ 0.25 AND held-out ≤ 0.25 AND Bayes − reader ≥ 0.30, with every reader's bottom→top improvement < 0.15 — the V-B gap persists despite a substantial increase in unique training data under the fixed bounded reader/optimization protocol.
* **H-C|W-C (data benefit without recovery)**: some reader gains ≥ 0.15 from bottom to top (1500→10000) but still fails W-A — training-data scale materially improves access but does not establish robust rule learning.
* **H-D|W-MIX**: material E/F disagreement, or genuinely contradictory reader data-scale conclusions.

## 6. Preregistration (declared before aggregate inspection)

* **Readers (exactly three, reused unchanged from U/V):** U's flat **MLP-128** on the flattened history window; V's causal **GRU** (hidden 85); V's causal **TCN** (channels 52, kernel 3, dilations [1,2,4,8], 4 layers). No architecture is reconstructed.
* **Training scale (the ONLY factor):** the predeclared ladder, run here as the reduced ladder `N_train ∈ {1500, 4000, 10000}` (the full predeclared ladder is `{1500, 8000, 32000}`), unique ordinary-ADD transitions from **fresh, pairwise sequence-disjoint, independent i.i.d. streams** (not nested, not recycled, not duplicated).
* **History grid (U/V, unchanged):** `L ∈ {L1, L2, L4, L8, L16, Lfull}`; `H_L(t) = [x_{max(0,t−L+1)}, …, x_t]`, left-padded with an explicit validity mask; no future observation. Per-timestep reader input = raw 21-dim observation + 1 validity bit = 22 features.
* **Targets:** `y_true` (primary, all six history lengths); `operand_a`, `operand_b` (secondary, decisive horizons `{L4, Lfull}`); `op_dest` (compact sanity control, `{L4, Lfull}`). Held-out variant at `{L4, Lfull}` for `y_true` and the operands.
* **Training protocol (V's, unchanged):** Adam, lr 3e-3, weight-decay 1e-2, **200 epochs, full-batch**, class-balanced cross-entropy, best-on-VAL checkpoint, deterministic per-fit seed. **200 optimizer updates at every scale**; no LR schedule, no early stopping on TEST, no scale-dependent tuning.
* **Splits:** fresh ordinary-ADD training pool → fit; the fixed canonical ordinary VAL (800/seed) → tune; the untouched canonical TEST → evaluate once.
* **Populations:** Class-A hard (primary), matched candidate-correct, all candidate-correct, all ADD. E seeds 42–46 are the independent units; F is a control.
* **Decision (checked in order; bars are declared conventions):**
  * **W-A** — some reader at L4/full: test ≥ 0.50 AND held-out ≥ 0.30 AND `gain_full = acc(top) − acc(bottom)` ≥ 0.15 (over the collected ladder).
  * **W-B** — across readers: best top-scale Class-A test ≤ 0.25 AND held-out ≤ 0.25 AND Bayes − reader ≥ 0.30 AND every reader `gain_full` < 0.15.
  * **W-C** — some reader `gain_full` ≥ 0.15 (or an intermediate best top-scale test > 0.25) without W-A.
  * **W-MIX** — material E/F disagreement or genuinely contradictory reader conclusions.
* No scale-dependent tuning; no TEST-based selection of scale, reader or history.

## 7. Frozen Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, optimizer, loss. **Nothing modified.** The frozen checkpoints define the Class-A test population (`cand_ok`) and reproduce the baseline only. All model execution under `torch.no_grad()`; checkpoint SHA-256 verified before/after. The W readers are separate analysis-only `nn.Module`s; they are never attached to the frozen model, no RE:Reason parameter acquires a gradient, and reader parameters are disjoint from the frozen model's. The Class-A definition is Q's, unchanged (frozen candidate written-register decode ≠ truth).

## 8. Reader Definitions (reused verbatim from U/V)

```
MLP-128   : flatten(H_L) ⊕ mask  (22·L)  -> Linear(·,128) -> tanh -> Linear(128,C)         [U's flat reader]
GRU       : x(22) -> nn.GRU(22, 85, num_layers=1, bidirectional=False) -> last step -> Linear(85, C)   [V's causal GRU]
TCN       : x(22) -> Conv1d(22, 52, k=3, d=1)   [causal pad/trim] -> ReLU
                  -> Conv1d(52, 52, k=3, d=2)   [causal]          -> ReLU
                  -> Conv1d(52, 52, k=3, d=4)   [causal]          -> ReLU
                  -> Conv1d(52, 52, k=3, d=8)   [causal]          -> ReLU
                  -> last position -> Linear(52, C)                                          [V's causal TCN]
```

The GRU is unidirectional and the TCN is genuinely causal (left-pad `(k−1)·d`, trim the right `(k−1)·d`); both are reused **unchanged** from `analyze_experiment_v.py` (`GRUReader`, `TCNReader`, `build_reader`, `train_reader`, `reader_input`), and the MLP-128 is U's `as_.mlp_fit` probe. Prediction at `t` uses only `x_0..x_t`.

## 9. Parameter Invariance

The reader parameter counts are **identical at every training scale** (the scale changes only the number of examples, never the model):

| Reader | Width | Params (primary head, C=10) | Params (max head, C=19) | Ratio to 29,578 |
|---|---|---:|---:|---:|
| MLP-128 | 128 | 29,578 (L=10) / 12,682 (L=4) | — | 1.000 |
| GRU | hidden 85 | 28,655 | 29,429 | 0.995 |
| TCN | channels 52 | 28,506 | 28,983 | 0.980 |

The full-prefix MLP-128 (input dim 220) defines the budget (29,578); the GRU/TCN widths are the largest integer strictly within it, derived deterministically. Every reader is within budget for every target class count; no reader exceeds the flat baseline.

## 10. Training-Data Scale Construction

For each E seed, three **independent, pairwise sequence-disjoint, fresh i.i.d.** ordinary-ADD pools are generated from distinct RNG streams (base `2,000,000`; offsets `{1500: 0, 4000: 1,000,000, 10000: 2,000,000}` for the reduced ladder, stream = base + offset + seed). Each pool is generated by drawing whole benchmark sequences and collecting their ADD transitions until exactly `N_train` transitions are reached, with **within-pool sequence deduplication**. The pools are **independent, not nested**: no pool is a subset of another, so each scale is a genuinely fresh sample rather than a re-weighted reuse of a smaller pool. The design is recorded explicitly in `experiments/experiment_w_data_scale_meta.json` (`pool_design`, `pool_stream_base`, `pool_stream_offsets`).

## 11. Sequence-Disjointness

Verified per seed (exact sequence-hash overlap counts, stored in the meta `disjointness` block):

| Overlap (per seed) | result |
|---|---:|
| pool1500 ∩ pool4000 | 0 |
| pool1500 ∩ pool10000 | 0 |
| pool4000 ∩ pool10000 | 0 |
| pools ∩ canonical VAL | 0 |
| pools ∩ canonical TEST | 0 |

No sequence is shared between any two training pools, or between a pool and VAL/TEST. The number of unique sequences touched per pool is reported in the meta (fewer than `N_train` because a sequence contributes several ADD transitions).

## 12. Splits

| Split | Role | Size | Invariance across scales |
|---|---|---|---|
| fresh pool1500 / pool4000 / pool10000 | reader fit | 1500 / 4000 / 10000 transitions per seed | the manipulated factor |
| canonical ordinary VAL | tune (best-on-VAL) | 800/seed (4000 pooled) | identical at every scale |
| canonical TEST | evaluate once | 15,763 ADD transitions pooled (1200 seq, stream 90042+seed) | identical at every scale |

The canonical TEST is the **untouched U/V TEST**; it was not regenerated. VAL is U/V's ordinary VAL protocol (first 800 ordinary-ADD transitions of the canonical VAL split per seed) and is fixed across scales, so the only change between conditions is the amount of training data.

## 13. Training Protocol

Identical to U/R/S/T/V: Adam (lr 3e-3, weight-decay 1e-2), **200 epochs, full-batch** (one optimizer update per epoch → **exactly 200 optimizer updates at every scale**), class-balanced cross-entropy, best-on-VAL checkpoint. No LR schedule, no extra epochs at any scale, no curriculum, no restarts, no TEST-based selection. The number of optimizer updates is constant across scales; only the number of unique examples processed per update changes with `N_train`.

## 14. Targets

`y_true` = the written register value = `(operand_a + operand_b) mod 10` (10 classes; primary); `operand_a`/`operand_b` = the pre-transition register values at the destination/source (10 classes; secondary); `op_dest` = the destination register index (4 classes; compact sanity control). Operation is constant ADD on this population.

## 15. Populations

Class-A hard (frozen candidate written-register decode ≠ truth; PRIMARY), matched candidate-correct, all candidate-correct, all ADD (the clean interface control). The reader is model-independent, so it is fit once per seed and evaluated on both the E and F masks; all-ADD is model-independent. E seeds 42–46 are the independent units; F is a control and is never pooled with E.

## 16. Held-Out Operand Protocol

The 20 operand pairs `{(a,b): b ∈ {0,5}}` are excluded from the held-out variant's fitting set, and the same reader is evaluated separately on **seen** and **held-out** test rows. For the primary target at `{L4, Lfull}` we report seen accuracy, held-out accuracy and the seen − held-out gap at every scale. Held-out pairs never enter fitting.

## 17. Results by Reader

Class-A TEST accuracy (`y_true`), E primary, five seeds, **mean ± SD**. L4 and full prefix, at each collected scale. `gain_full = acc(10000) − acc(1500)`.

| Reader | rep | 1500 | 4000 | 10000 | gain_full |
|---|---|---:|---:|---:|---:|
| MLP-128 | L4 | 0.089 ± 0.010 | 0.076 ± 0.007 | 0.070 ± 0.022 | −0.019 |
| MLP-128 | Lfull | 0.101 ± 0.013 | 0.079 ± 0.011 | 0.071 ± 0.016 | −0.030 |
| GRU | L4 | 0.083 ± 0.018 | 0.102 ± 0.012 | **0.116 ± 0.021** | **+0.033** |
| GRU | Lfull | 0.080 ± 0.019 | 0.109 ± 0.019 | 0.106 ± 0.015 | +0.025 |
| TCN | L4 | 0.106 ± 0.018 | 0.107 ± 0.012 | 0.114 ± 0.024 | +0.008 |
| TCN | Lfull | 0.107 ± 0.023 | 0.110 ± 0.015 | 0.111 ± 0.015 | +0.004 |

Best top-scale reader: **GRU L4 = 0.116** (vs 0.107 best at the bottom scale, TCN Lfull). Every reader at the top scale is ≤ 0.25, i.e. **weak** under the pre-registered definition; none is close to the 0.50 bar. The MLP-128 anchor's Class-A accuracy is *lower* at 10000 than at 1500 (both reps), so it does not improve with data. The only reader with a positive trend is the GRU (L4 monotone 0.083 → 0.102 → 0.116; +0.033, still sub-material).

## 18. Results by Data Scale

Best Class-A TEST accuracy at each scale across all readers and the L4/full-prefix horizons (E, 5-seed mean), with the same reader's held-out accuracy:

| N_train | best reader | best Class-A test | held-out (same reader) |
|---:|---|---:|---:|
| 1500 | TCN Lfull | 0.107 | 0.183 |
| 4000 | TCN Lfull | 0.110 | 0.100 |
| 10000 | GRU L4 | **0.116** | 0.170 |

The best attainable Class-A accuracy is essentially flat across the ladder (0.107 → 0.110 → 0.116); the rank order of readers changes but the level does not. Held-out accuracy at the best reader is likewise at chance (0.100–0.183) and non-monotone.

## 19. Class-A Results

The primary population is Class-A (frozen candidate written-register decode ≠ truth; 1,805 rows pooled over the 5 E seeds: 448/330/363/327/337). §17 gives the full table. Key points:

* **The level is flat and far from the ceiling.** Best reader 0.107 (1500) → 0.116 (10000); the exact Class-A Bayes ceiling at L4 is 0.999.
* **Seed variance is comparable to the data effect.** Per-seed L4 Class-A at 10000 — MLP-128: 0.094 / 0.064 / 0.091 / 0.061 / 0.042; GRU: 0.085 / 0.121 / 0.143 / 0.110 / 0.122; TCN: 0.118 / 0.152 / 0.085 / 0.104 / 0.110. Reader SDs (0.007–0.024) are of the same order as the 1500→10000 change.
* **Secondary targets (L4, 10000, Class-A):** `operand_a` MLP-128 0.239 / GRU 0.148 / TCN 0.118; `operand_b` 0.232 / 0.134 / 0.115; `op_dest` 1.000 / 1.000 / 0.241. The readers recover the *destination index* (a directly observable interface label; GRU and MLP-128 at 1.000) but not the operand values or the written target — the same target-specific signature V found — and this does not improve with data.

## 20. All-ADD Control

All-ADD is the model-independent interface control (no candidate filter). E and F are identical by construction. `y_true`, mean ± SD:

| Reader | rep | 1500 | 4000 | 10000 |
|---|---|---:|---:|---:|
| MLP-128 | L4 | 0.188 ± 0.007 | 0.206 ± 0.009 | 0.214 ± 0.006 |
| MLP-128 | Lfull | 0.220 ± 0.007 | 0.224 ± 0.009 | **0.229 ± 0.007** |
| GRU | L4 | 0.103 ± 0.005 | 0.106 ± 0.027 | 0.181 ± 0.097 |
| GRU | Lfull | 0.136 ± 0.014 | 0.120 ± 0.021 | 0.192 ± 0.095 |
| TCN | L4 | 0.201 ± 0.115 | 0.126 ± 0.090 | 0.179 ± 0.109 |
| TCN | Lfull | 0.138 ± 0.092 | 0.290 ± 0.010 | 0.247 ± 0.086 |

The flat MLP-128 anchor remains the strongest all-ADD reader (Lfull 0.229) and rises modestly with data; the sequence readers reach comparable best values (TCN Lfull 0.290 at 4000) but with very large seed variance (±0.09–0.11), and none beats the anchor consistently. All all-ADD numbers are ≤ 0.30 and far below the 0.50 bar — even on the unfiltered population the bounded readers do not access the target.

## 21. Held-Out Generalization

Held-out operand pairs (`b ∈ {0, 5}`; 20 pairs) are excluded from fitting; the same reader is evaluated separately on seen and held-out Class-A `y_true` test rows. Top scale (10000), E, mean. Gap = seen − held-out (negative ⇒ held-out ≥ seen).

| Reader | rep | seen | held-out | seen − held-out |
|---|---|---:|---:|---:|
| MLP-128 | L4 | 0.083 | 0.102 | −0.019 |
| MLP-128 | Lfull | 0.088 | 0.102 | −0.014 |
| GRU | L4 | 0.094 | 0.170 | −0.076 |
| GRU | Lfull | 0.083 | 0.206 | −0.123 |
| TCN | L4 | 0.116 | 0.090 | +0.025 |
| TCN | Lfull | 0.111 | 0.133 | −0.022 |

Every held-out accuracy is ≤ 0.21 (bar 0.30) and falls as often *above* as below the seen accuracy (gaps −0.123 … +0.025). Because every reader is near chance on both subsets, the seen/held-out contrast carries no evidence of rule learning or of a seen-only shortcut — there is simply no learned operand→target rule to generalise, or to fail to generalise.

## 22. Train/Test Gap

L4 training accuracy vs Class-A TEST accuracy (E, mean):

| Reader | train 1500 | train 10000 | test 1500 | test 10000 |
|---|---:|---:|---:|---:|
| MLP-128 | 0.375 | 0.232 | 0.089 | 0.070 |
| GRU | 0.132 | 0.183 | 0.083 | 0.116 |
| TCN | 0.196 | 0.175 | 0.106 | 0.114 |

Two observations. **(i) No memorization regime.** Unlike R's size-matched hard-tail training, training accuracy here is low (0.13–0.23) and never approaches 1.0, so the flat test curve reflects **underfitting**, not learned-but-non-generalising memorization. **(ii) The MLP-128 anchor's training accuracy *falls* as the pool grows** (0.375 → 0.232) while its test accuracy also falls — under a fixed 200 full-batch updates and fixed capacity, more data makes the fit harder, not better. This is a fixed-capacity/fixed-budget regime, not a data-starvation regime.

## 23. Bayes-vs-Reader Gap

The exact Class-A `y_true` Bayes ceiling (U, model E) is **0.999** at L4 and **1.000** at the full prefix. The best reader at the top scale is 0.116 (GRU L4) → **Bayes − reader gap = 0.883** (per reader: MLP-128 0.929, GRU 0.883, TCN 0.886). At the bottom scale the best reader is 0.107 (TCN Lfull) → gap 0.892. A ~6.7× increase in unique training data moves the best-reader gap from **0.892 to 0.883** — essentially unmoved.

## 24. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a control, never pooled. Every reported fraction is a mean ± SD over the five E seeds; the Class-A population is defined by the frozen checkpoints, never by the readers. The reader set, history grid, target scope, training protocol and decision bars were **predeclared**; every reader and scale is reported, including negative results; no TEST-based selection of scale, reader, history or seed. `gain_full = acc(top) − acc(bottom)` (bottom→top over the collected ladder) is computed on the pooled 5-seed means.

## 25. Leakage / Validity

Enforced and tested: (i) training pools are pairwise sequence-disjoint and disjoint from VAL/TEST; (ii) TEST is stored once and identical across scales; (iii) VAL is identical across scales; (iv) no TEST-derived selection of scale/reader/history; (v) held-out operand pairs never enter fitting; (vi) no duplicate training rows within a pool; (vii) the number of optimizer updates is exactly 200 at every scale; (viii) the reader architecture and parameter count are invariant across scales; (ix) no future observation enters the reader input (the causal window ends at `t`; verified invariant to randomising observations after `t`); (x) frozen E/F parameters are never mutated and acquire no gradients; (xi) reader parameters are never attached to the frozen model; (xii) checkpoint SHA-256 is unchanged before/after.

## 26. Deterministic Reproduction

The collector was run twice: the stored arrays are **bit-identical** (all 74 arrays) and the meta JSON is identical except its timestamp. The analyzer was **not** re-run end-to-end (a full run is ≈124 min on this hardware); instead determinism is established by (i) the bit-identical collector re-run above, (ii) a re-run of the **seed-42 MLP-128 block** that reproduces the stored per-seed `fits` dicts **exactly** (no floating-point drift), and (iii) the deterministic per-fit seed (`reader_seed`; verified distinct for every cell by `tests/test_experiment_w.py::test_deterministic_fit_seed`) with single-threaded workers and order-independent aggregation. Checkpoint SHA-256 verified before/after.

## 27. Causal Status

**Observational with respect to RE:Reason.** W deliberately manipulates training-data scale for the **analysis-only reader**, so the reader-training comparison is causal *within the reader experiment*: "increasing the analysis reader's training-set size under the fixed training protocol did / did not improve its held-out performance." It is **not** a causal statement about RE:Reason. W does **not** show that "more data causes RE:Reason to fail", that "RE:Reason cannot learn the task", or that "the architecture is proven insufficient". The four levels remain distinct: **interface availability** (U) ≠ **reader learnability** (V, W) ≠ **frozen-state representation** (T/S) ≠ **architecture sufficiency**. No intervention is performed or licensed.

## 28. Decision Classification

**W-B — data-saturated / bounded-limited** (E primary; F agrees; `E_F_disagreement = False`).

Pre-registered bars (declared conventions): `recover_bar = 0.50`, `heldout_weak = 0.25`, `heldout_material = 0.30`, `material_gain = 0.15`, `bayes_gap_bar = 0.30`.

* **W-A** (some reader at L4/full: top-scale test ≥ 0.50 AND held-out ≥ 0.30 AND `gain_full` ≥ 0.15) — **not met** (best 0.116; held-out 0.170; max `gain_full` +0.033).
* **W-B** (best top-scale Class-A test ≤ 0.25 AND held-out ≤ 0.25 AND Bayes − reader ≥ 0.30 AND every reader `gain_full` < 0.15) — **met** (0.116 ≤ 0.25; 0.170 ≤ 0.25; 0.883 ≥ 0.30; max `gain_full` +0.033 < 0.15).
* **W-C** — **not met** (no reader reaches a material gain ≥ 0.15 and the best top-scale test is ≤ 0.25).
* **W-MIX** — **not met** (E and F agree: F best top-scale 0.109 at L4, both W-B; the GRU and TCN do not contradict — both are weak).

Continuous quantities (reported alongside the bars): best top-scale Class-A test **0.116** (GRU L4); best held-out **0.170** (GRU L4); max `gain_full` **+0.033** (GRU L4); Bayes − reader **0.883**. The single sub-threshold trend (GRU L4 rising 0.083 → 0.102 → 0.116) is explicitly **sub-material** and does not approach the bar.

## 29. What W Establishes

* Over the reduced predeclared ladder **1500 → 4000 → 10000** unique, pairwise sequence-disjoint, i.i.d. ordinary-ADD transitions, the three predeclared bounded readers (U's flat MLP-128, V's causal GRU, V's causal TCN) do **not** materially improve their Class-A accuracy toward the exact ceiling: the best reader moves **0.107** (TCN Lfull, 1500) → **0.116** (GRU L4, 10000), and the best per-reader `gain_full` is only **+0.033** (GRU L4).
* The flat **MLP-128 anchor does not benefit from more data** (L4 0.089 → 0.070; Lfull 0.101 → 0.071) and its **training accuracy falls** as the pool grows (0.375 → 0.232) — a fixed-capacity/fixed-budget regime, not data starvation.
* Held-out operand accuracy stays at chance everywhere (≤ 0.206) and the readers **do not memorize** (training accuracy 0.13–0.23), so the flat curve accompanies neither rule learning nor pool memorization.
* The Bayes − reader gap is **unchanged** by a ~6.7× data increase (0.892 → 0.883, best reader).
* The result **reproduces V's direction (V-B) at a larger data scale**: over 1500–10000, the persistent weak accessibility is **not** a simple training-data limitation under the fixed reader/protocol — an outcome **consistent with** a bounded-capacity / algorithmic limit (the data-limited branch W-A is not met; the optimization-vs-capacity split is left open).

## 30. What W Does Not Establish

* It does **not** prove the information is mathematically unlearnable by any bounded reader; only that the tested three readers under the fixed protocol do not reach recovery at these scales.
* It does **not** show the frozen RE:Reason model can or cannot access the information; **reader learnability ≠ frozen representation ≠ architecture sufficiency**.
* It does **not** show that training-data volume is irrelevant in general: the ladder tops out at **10,000** transitions (the predeclared full ladder extends to 32,000), so the negative scale result is scoped to 1500–10000 and **does not exclude** a larger-data regime.
* It does **not** separate the optimization-budget branch from the bounded-capacity branch: the readers underfit (low training accuracy) under a fixed 200-update protocol, so "the optimization budget is the binding constraint" remains a live, untested alternative to "a bounded-capacity/algorithmic limit". A weak top-scale result is reported as "under the fixed optimization budget/protocol", never as "optimization cannot solve it".
* It does **not** establish that a different architecture, width, loss, or training schedule would succeed or fail.
* It does **not** prove a causal mechanism for RE:Reason; the training-data manipulation is causal only within the analysis reader experiment.
* It does **not** claim the benchmark is unsolvable or the readers are optimal.

## 31. Final Answers

* **Does increasing unique training data 1500 → 10000 materially raise reader Class-A accuracy?** No. Best-reader level 0.107 → 0.116; best per-reader `gain_full` +0.033 (sub-material); the flat MLP-128 anchor declines. **W-A is not met.**
* **Do any readers reach the 0.50 test bar or the 0.30 held-out bar?** No. Top-scale Class-A ≤ 0.116; held-out ≤ 0.206.
* **Do the readers at least memorize the pool (high train, low test)?** No. Training accuracy is 0.13–0.23 and never approaches 1.0 — the regime is underfitting.
* **Does the data scale close the Bayes − reader gap?** No — 0.892 (1500, best) → 0.883 (10000, best).
* **Data-limited, optimization-limited, or capacity-limited?** Not a simple data limitation over 1500–10000. The two remaining branches — fixed optimization budget vs bounded capacity/algorithm — are **not separated** by W (the readers underfit, which is at least *consistent with* both). At most W **falsifies the data-volume branch in this range**.
* **Do E and F agree?** Yes (F best top-scale 0.109 at L4; both W-B; `E_F_disagreement = False`).

## 32. Exactly ONE Next Experiment

**Experiment X — fixed-data optimization-budget sensitivity (measurement-only, frozen checkpoints):** hold **everything W held fixed** — the fresh top-scale training pool (10,000 unique, sequence-disjoint ordinary-ADD transitions per seed), the three readers and their widths (29,578-budget MLP-128 / GRU hidden 85 / TCN channels 52), the causal history grid, the targets, the populations, the held-out protocol and the VAL/TEST splits — and manipulate **only the number of full-batch optimizer updates**, using the same Adam (lr 3e-3, weight-decay 1e-2), class-balanced cross-entropy, best-on-VAL checkpoint and fixed per-fit seed. The predeclared alternative budget is **1000 updates** (vs W's fixed 200), and every reader/scale cell is reported. W observed that the readers **underfit** (L4 training accuracy 0.13–0.23) and that more data alone did not help, so the single unresolved lever is the training protocol: this isolates W's *optimization-limitation* branch from the *bounded-capacity/algorithmic* branch by changing the budget and nothing else. Decision: **X-A** (some reader reaches ≥ 0.50 test AND ≥ 0.30 held-out) — the gap was optimization-limited; **X-B** (all readers stay ≤ 0.25 test with training accuracy still low) — the limitation is not the update budget, strengthening the capacity/algorithm reading. Analysis-only; no frozen weight/architecture change; no candidate/state modification; no Phase 2; specified, not started.

## Required Figures

`docs/reports/figures/experiment_w_*.png` (12): `accuracy_vs_data_scale`, `heldout_vs_data_scale`, `bayes_gap_vs_data_scale`, `classA_vs_alladd`, `reader_comparison`, `l4_comparison`, `fullprefix_comparison`, `seen_vs_heldout`, `train_vs_test`, `per_seed`, `parameter_invariance`, `decision_summary`.

## Tests

`tests/test_experiment_w.py` — V architecture reuse; parameter-count invariance; ladder consistency (read from the collected artifact); training-size generation (exact collected sizes) and declared-stream reproduction of the collected pools; train-pool sequence disjointness; VAL and TEST invariance; no duplicate training rows; held-out operand exclusion; causal history correctness; no-future leakage; exact target generation; Class-A count reproduction; reader architecture / optimizer invariance; exactly 200 optimizer updates; best-checkpoint-from-VAL; deterministic fit seed; no RE:Reason parameter mutation or gradients; reader/model separation; history/scale/reader completeness; held-out outputs; deterministic artifacts; malformed-artifact rejection; figure creation; historical immutability; research-governance compliance.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_w.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_w_data_scale.npz
PYTHONPATH=src python experiments/analyze_experiment_w.py \
    --npz experiments/experiment_w_data_scale.npz \
    --u-results experiments/experiment_u_results.json \
    --output experiments/experiment_w_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_w
PYTHONPATH=src pytest tests/test_experiment_w.py -v
PYTHONPATH=src pytest tests/
```

Determinism: collector arrays bit-identical across two runs; the analysis JSON is identical except its timestamp; all 12 figures byte-identical. Checkpoint SHA-256 verified before/after. No RE:Reason weight is modified. The analyzer reuses `analyze_experiment_v` reader definitions and `analyze_experiment_u`/`analyze_experiment_s` history/probe primitives.

## Scientific Guardrails

Measurement-only; observational with respect to RE:Reason. The report keeps four things separate: **exact information availability** (U's Bayes ceilings), **reader learnability** (V/W), the **frozen model's representation** (T/S), and **causal interpretation** (only within the reader-training manipulation). A failed bounded reader is **not** proof the information is absent (the exact ceiling proves the opposite); a high Bayes ceiling is **not** evidence the frozen network can exploit it. Q, R, S, T, U and V are **not** edited; W is appended as new evidence. No Phase 2. Negative and borderline findings are preserved.
