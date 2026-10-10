# RE:Reason — Experiment X: Fixed-Data Optimization-Budget Sensitivity

**Date:** 2026-10-09
**Status:** Completed — Measurement only (frozen E/F checkpoints used only to define the Class-A population and reproduce baseline metrics; no model training; no optimizer on RE:Reason parameters; no architecture/encoder/recurrent-core/state/gate/candidate/commit/readout modification; no state replacement; no candidate manipulation; no feature editing; no oracle/reconstructed state as a reader feature; no branching/verifier; no counterfactual rollout; no Phase 2). The X readers are separate analysis-only models and never enter frozen model inference. Classification: **X-B — persistent bounded-reader limitation** for E and F.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, SHA-256-pinned (unchanged before/after).
**Data:** `experiments/experiment_x_optimization_budget.npz`, `experiments/experiment_x_optimization_budget_meta.json`, `experiments/experiment_x_results.json`
**Runners:** `experiments/run_experiment_x.py`, `experiments/analyze_experiment_x.py`, tests `tests/test_experiment_x.py`

---

## 1. Title / Date / Status

Experiment X — Fixed-Data Optimization-Budget Sensitivity. Measurement-only, frozen; **X-B — persistent bounded-reader limitation**. Exactly one factor is manipulated: `optimizer_updates ∈ {200, 1000}`. No intervention on the frozen model; no Phase 2. Q–W reports are untouched; X is appended as new evidence.

## 2. Executive Answer

Raising the bounded readers' optimization budget **5× (200 → 1,000 full-batch updates)** with the data, model, optimizer and splits held **exactly fixed did not improve target or held-out generalization** — it is X-B, a persistent bounded-reader limitation over the tested conditions.

* **No reader improves.** Best 1,000-budget Class-A `y_true` accuracy is **0.111** (TCN Lfull) versus **0.116** at 200 (GRU L4); the paired best-rep changes are **GRU −0.006**, **TCN −0.003**, **MLP-128 +0.004** — all within seed noise and sub-material. No reader reaches 0.25.
* **Held-out operand accuracy stays at chance** (0.094–0.212 at 1,000; every value ≤ 0.25; bar 0.30). The three largest held-out values are the GRU/TCN L4 cells (≈0.21), still far below the material bar.
* **The readers still do not fit the training pool.** Training accuracy at 1,000 updates is **0.07–0.15** for the GRU/TCN (near/below the 0.10 chance) and reaches only **0.499** for the flat MLP-128 at long histories (L16/Lfull). The pre-registered "still low" bar is 0.50, and the max (0.499) sits *just* below it — a borderline that is reported explicitly rather than relied upon.
* **The validation trajectory is flat-to-noisy.** Across the 200→1,000 milestones the mean validation accuracy of the sequence readers is non-monotone and stays ≤ 0.16 (e.g. GRU L4: 0.097 → 0.061 → 0.068 → 0.069 → 0.069); the independent checkpoints selected by best-on-VAL are scattered across the budget (best indices 36–394 at 1,000), confirming there is no later-checkpoint improvement to select.
* **The Bayes − reader gap does not close.** The exact Class-A Bayes ceiling at L4 is **0.999**; the best reader is 0.111 → gap **0.888** (0.883–0.931 across readers).
* **E and F agree exactly** (all-ADD accuracy identical to 0, Class-A differences within seed noise; both X-B).
* **The 200-update bridge against W is exact** (`max |X₂₀₀ − W₂₀₀| = 0.0` over every primary reader × history Class-A cell), so the paired comparison is internally valid.

Interpretation: over the fixed data/model/protocol, the W-B/V-B weak accessibility is **not** relieved by a 5× larger update budget — this *strengthens* (but does not prove) a bounded-capacity/algorithmic reading, and it does not identify what would. The result is **observational** and carries **no causal claim** about RE:Reason; **no intervention is justified**.

## 3. Q–W Evidence Context

* **Q/R**: the true written value is only weakly recoverable from the frozen candidate `C_t` on Class-A (Q-C); bounded capacity and size-matched hard-tail training do not help, and the only apparent gain is operand-combination memorization (R-D).
* **S**: no internal candidate activation materially beats `C_t` (S-C).
* **T**: the frozen observation-encoder output `e_t` is structurally operand-blind and no upstream frozen representation is robustly recoverable (T-D).
* **U**: on the raw causal observation history, the target is exactly identifiable (**L=4 Bayes 0.998, operands 0.999; full prefix 1.000**) yet bounded flat probes stay weak (best full-prefix 0.218 all-ADD, ~0.09 Class-A; held-out ≤ 0.25) → **U-D (interface-identifiable but probe-limited)**.
* **V**: two parameter-bounded causal **sequence** readers (GRU hidden 85; TCN channels 52), each within the exact U MLP-128 budget, are also weak (best Class-A 0.112; held-out ≤ 0.25; Bayes − reader 0.887) → **V-B (sequence-limited)**.
* **W**: at a **fixed 200 full-batch updates**, 1,500 / 4,000 / 10,000 fresh sequence-disjoint training pools did not lift Class-A accuracy (best 0.107 → 0.116, best gain +0.033), the flat MLP-128 anchor declined, held-out stayed ≤ 0.21, the Bayes − reader gap barely moved (0.892 → 0.883), and **training accuracy stayed low (0.13–0.23)** → **W-B (data-saturated/bounded-limited)**. W's §32 specified exactly this optimization-budget test as Experiment X.

## 4. Research Question and Hypotheses

> Does increasing the optimization budget from **200 to 1,000** full-batch updates improve target generalization and held-out operand generalization when the data, model, and all other training conditions remain fixed?

* **H-A|X-A (optimization-limited):** at least one reader at L4 or the full prefix reaches test ≥ 0.50 AND held-out ≥ 0.30 at 1,000 updates.
* **H-B|X-B (persistent bounded-reader limitation):** all three readers stay ≤ 0.25 test with held-out accuracy still weak and training accuracy still low after 1,000 updates.
* **Otherwise (inconclusive/partial):** neither rule is met; the continuous results are reported and the explanation is left unresolved.

## 5. Preregistration (declared before aggregate inspection)

* **Factor (the ONLY one):** `optimizer_updates ∈ {200, 1000}`. One 1,000-update trajectory is run per cell; the **200** budget is the best-on-VAL checkpoint over the first 200 updates (W's exact protocol) and the **1,000** budget is the best-on-VAL checkpoint over all 1,000. No repetition, duplication or reweighting of examples.
* **Readers (exactly three, reused unchanged from U/V/W):** U's flat **MLP-128** on the flattened history window; V's causal **GRU** (hidden 85); V's causal **TCN** (channels 52, kernel 3, dilations [1,2,4,8], 4 layers).
* **Training protocol (W's, unchanged):** Adam (lr 3e-3, weight-decay 1e-2), **full-batch**, class-balanced cross-entropy, best-on-VAL checkpoint, deterministic per-fit seed. **No** LR schedule, scheduler, batch-size change, curriculum, restarts, or TEST-based selection.
* **History grid:** `L ∈ {L1, L2, L4, L8, L16, Lfull}`; `H_L(t) = [x_{max(0,t−L+1)}, …, x_t]`, left-padded with an explicit validity mask; no future observation. Per-timestep reader input = raw 21-dim observation + 1 validity bit = 22 features.
* **Targets:** `y_true` (primary, all six lengths); `operand_a`, `operand_b` (secondary, `{L4, Lfull}`); `op_dest` (sanity control, `{L4, Lfull}`). Held-out variant at `{L4, Lfull}` for `y_true` and the operands.
* **Splits (fixed):** W's fresh top-scale pool → fit; the fixed canonical ordinary VAL (800/seed) → tune; the untouched canonical TEST → evaluate once.
* **Populations:** Class-A hard (primary), matched candidate-correct, all candidate-correct, all ADD. E seeds 42–46 are the independent units; F is a control.
* **Decision (checked in order; bars are declared conventions):**
  * **X-A** — some reader at L4/full: 1,000-budget test ≥ 0.50 AND held-out ≥ 0.30.
  * **X-B** — across readers: best 1,000-budget test ≤ 0.25 AND held-out ≤ 0.25 AND training accuracy still low (`< 0.50`).
  * **else** — inconclusive / partial improvement (continuous results reported).

## 6. Frozen Model Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, optimizer, loss. **Nothing modified.** The frozen E/F checkpoints define the Class-A test population (`cand_ok`) and reproduce the baseline only. All model execution under `torch.no_grad()`; checkpoint SHA-256 verified before/after. The X readers are separate analysis-only `nn.Module`s; they are never attached to the frozen model, no RE:Reason parameter acquires a gradient, and reader parameters are disjoint from the frozen model's.

## 7. Reader Definitions (reused verbatim from U/V)

```
MLP-128   : flatten(H_L) ⊕ mask  (22·L)  -> Linear(·,128) -> tanh -> Linear(128,C)         [U's flat reader]
GRU       : x(22) -> nn.GRU(22, 85, num_layers=1, bidirectional=False) -> last step -> Linear(85, C)   [V's causal GRU]
TCN       : x(22) -> Conv1d(22, 52, k=3, d=1)   [causal pad/trim] -> ReLU
                  -> Conv1d(52, 52, k=3, d=2)   [causal]          -> ReLU
                  -> Conv1d(52, 52, k=3, d=4)   [causal]          -> ReLU
                  -> Conv1d(52, 52, k=3, d=8)   [causal]          -> ReLU
                  -> last position -> Linear(52, C)                                          [V's causal TCN]
```

X builds every reader through V's `build_reader`/`GRUReader`/`TCNReader` and U's MLP-128 definition — no reconstruction. The GRU is unidirectional and the TCN is genuinely causal; prediction at `t` uses only `x_0..x_t`.

| Reader | Width | Params (primary head, C=10) | Params (max head, C=19) | Ratio to 29,578 |
|---|---|---|---|---|
| MLP-128 | 128 | 29,578 (L=10) | — | 1.000 |
| GRU | hidden 85 | 28,655 | 29,429 | 0.995 |
| TCN | channels 52 | 28,506 | 28,983 | 0.980 |

Parameter counts are byte-identical at both update budgets (only the update count changes).

## 8. Fixed Data and Split Identity

X reuses **W's existing fresh top-scale pool** (10,000 unique ordinary-ADD transitions per E seed), W's canonical ordinary VAL, and W's untouched canonical TEST (E and F). Nothing is regenerated; the collector copies the arrays verbatim and records per-array SHA-256 plus the source-artifact SHA-256.

| Property | Value |
|---|---|
| Source artifact | `experiments/experiment_w_data_scale.npz` (SHA-256 recorded in the X meta) |
| Training pool | pool10000: 10,000 transitions/seed, 50,000 pooled (5 seeds) — byte-identical to W |
| Unique sequences (seed 42) | 3,634 (from 10,000 transitions; no duplicate rows) |
| Duplicate training rows | 0 (by `(sequence, t)` fingerprint) |
| Pool ∩ VAL / Pool ∩ TEST / VAL ∩ TEST | 0 / 0 / 0 |
| Held-out operand rows (seed 42) | 5,026 of 10,000 excluded in the held-out variant |
| Canonical VAL | 4,000 pooled (800/seed) — identical to W |
| Canonical TEST | 15,763 pooled (E and F) — identical to W |
| Class-A counts (E, per seed) | 448 / 330 / 363 / 327 / 337 |

## 9. Optimization Protocol

Identical to U/V/W except the update count: Adam (lr 3e-3, weight-decay 1e-2), full-batch, class-balanced cross-entropy, best-on-VAL checkpoint, deterministic per-fit seed. Each cell runs a single **1,000-update** trajectory and evaluates VAL every 10 updates; the 200-update condition is the best-on-VAL checkpoint over the first 200 updates, so the paired conditions share the same initialization, data and optimizer state by construction. The number of optimizer updates actually executed is exactly 1,000 per cell (verified in `tests/test_experiment_x.py`); no example is repeated or reweighted to simulate more updates.

## 10. Validation Checkpoint Selection

The reported model for each budget is the **best-on-VAL** checkpoint (validation accuracy only). The bridge requirement is satisfied exactly: X's 200-budget best-on-VAL state and metrics reproduce W's stored 200-update result for the equivalent seed/reader/target/history/pool, with **`max |X₂₀₀ − W₂₀₀| = 0.0`** over every primary reader × history Class-A cell, and the model-level trajectory reproduces `av.train_reader` / `as_.mlp_fit` exactly at 200 updates (`tests/test_experiment_x.py::test_update_200_bridge_against_w_procedure`). No TEST information enters selection.

## 11. Target and Population Definitions

`y_true` = written register value = `(operand_a + operand_b) mod 10` (10 classes; primary); `operand_a`/`operand_b` = pre-transition register values at destination/source (10 classes; secondary); `op_dest` = destination register index (4 classes; sanity control). Populations: Class-A hard (frozen candidate written-register decode ≠ truth; PRIMARY), matched candidate-correct, all candidate-correct, all ADD (interface control). Operation is constant ADD.

## 12. Primary Results

Class-A TEST accuracy (`y_true`), E primary, five seeds, mean ± SD. `Bayes L4 = 0.999` (U, model E).

| Reader | Updates | L4 test | L4 held-out | Full test | Full held-out | Bayes L4 | Bayes − reader |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP-128 | 200 | 0.070 ± 0.022 | 0.102 | 0.071 ± 0.016 | 0.102 | 0.999 | 0.929 |
| MLP-128 | 1,000 | 0.068 ± 0.019 | 0.094 | 0.075 ± 0.019 | 0.094 | 0.999 | 0.931 |
| GRU | 200 | 0.116 ± 0.021 | 0.170 | 0.106 ± 0.015 | 0.206 | 0.999 | 0.883 |
| GRU | 1,000 | 0.110 ± 0.015 | 0.212 | 0.102 ± 0.018 | 0.205 | 0.999 | 0.889 |
| TCN | 200 | 0.114 ± 0.024 | 0.090 | 0.111 ± 0.015 | 0.133 | 0.999 | 0.885 |
| TCN | 1,000 | 0.110 ± 0.015 | 0.212 | 0.111 ± 0.015 | 0.186 | 0.999 | 0.889 |

`Bayes − reader` uses the row's L4 test value against the exact L4 ceiling (0.999). Best 1,000-budget reader: **TCN Lfull = 0.111** (best 200-budget: GRU L4 = 0.116). **No reader improves materially**; the paired best-rep changes are MLP-128 **+0.004**, GRU **−0.006**, TCN **−0.003**. The all-ADD control (model-independent) at 1,000 updates is MLP-128 0.227 (L4) / 0.245 (Lfull), GRU 0.290 / 0.253, TCN 0.290 / 0.247 — still far below the 0.50 bar.

## 13. Training and Validation Trajectories

Mean training accuracy (at the 1,000-update endpoint) and mean validation accuracy at the predeclared milestones (`y_true`, E). The sequence readers' validation curves are flat-to-noisy with no upward trend; the best-on-VAL checkpoint indices are scattered (36–394 at 1,000), i.e. there is no later checkpoint worth selecting.

| Reader | rep | val@200 | val@400 | val@600 | val@800 | val@1000 | train@1000 | best idx (200 / 1000) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP-128 | L4 | 0.217 | 0.215 | 0.224 | 0.229 | 0.231 | 0.256 | 88 / 396 |
| MLP-128 | Lfull | 0.228 | 0.225 | 0.233 | 0.237 | 0.235 | 0.499 | 76 / 588 |
| GRU | L4 | 0.097 | 0.061 | 0.068 | 0.069 | 0.069 | 0.071 | 44 / 250 |
| GRU | Lfull | 0.078 | 0.072 | 0.061 | 0.069 | 0.100 | 0.102 | 74 / 146 |
| TCN | L4 | 0.078 | 0.077 | 0.079 | 0.060 | 0.156 | 0.149 | 36 / 394 |
| TCN | Lfull | 0.155 | 0.106 | 0.160 | 0.127 | 0.122 | 0.116 | 90 / 90 |

Training accuracy over all six history lengths (1,000 updates, E, mean): MLP-128 max **0.499** (Lfull; L16 0.497), GRU max **0.102** (Lfull), TCN max **0.149** (L4). The GRU/TCN stay near/below the 0.10 chance level on the training pool — they do not fit it at all — while the flat MLP-128 partially fits long-history windows but does not generalize (test ≈ 0.07). The pre-registered "training still low" bar (0.50) is met only because the maximum sits at 0.499; this borderline is reported explicitly and is not load-bearing (the sequence readers, which drive the headline, are at 0.07–0.15).

## 14. Test Generalization

Across the 200→1,000 budget increase the best Class-A `y_true` test accuracy moves **0.116 → 0.111** (best-rep paired: GRU −0.006, TCN −0.003, MLP-128 +0.004). Every reader stays ≤ 0.25 at both budgets, and the exact Class-A Bayes ceiling at L4 is 0.999 — the accessible fraction of the gap does not increase with the update budget. Per-seed 1,000-budget Class-A L4 accuracy: MLP-128 0.094 / 0.067 / 0.074 / 0.061 / 0.042; GRU 0.123 / 0.121 / 0.085 / 0.110 / 0.110; TCN 0.123 / 0.121 / 0.085 / 0.110 / 0.110 (E seeds 42–46).

## 15. Held-Out Operand Generalization

Held-out operand pairs (`b ∈ {0, 5}`; 20 pairs) are excluded from fitting; the same reader is evaluated separately on seen and held-out Class-A `y_true` rows. Both budgets, E, mean.

| Reader | rep | 200 seen | 200 held-out | 1000 seen | 1000 held-out |
|---|---|---:|---:|---:|---:|
| MLP-128 | L4 | 0.083 | 0.102 | 0.093 | 0.094 |
| MLP-128 | Lfull | 0.088 | 0.102 | 0.087 | 0.094 |
| GRU | L4 | 0.094 | 0.170 | 0.084 | 0.212 |
| GRU | Lfull | 0.083 | 0.206 | 0.090 | 0.205 |
| TCN | L4 | 0.116 | 0.090 | 0.084 | 0.212 |
| TCN | Lfull | 0.111 | 0.133 | 0.097 | 0.186 |

Every held-out value is ≤ 0.25 (bar 0.30), and the largest (≈0.21, GRU/TCN L4) is still far below the material bar. Because every reader is near chance on both seen and held-out rows, the contrast carries no evidence of learned operand→target generalization at either budget.

## 16. Bayes-versus-Reader Gaps

The exact Class-A `y_true` Bayes ceiling (U, model E) is **0.999** at L4 and **1.000** at the full prefix. Best-reader Bayes − reader gaps at 1,000 updates: MLP-128 0.931, GRU 0.889, TCN 0.889 (best reader TCN Lfull 0.111 → gap 0.888). The gap is essentially unchanged from W's 200-update value (0.883–0.892) — a 5× larger update budget does not move it.

## 17. E/F Controls

The all-ADD population is model-independent; E and F reader accuracies on all-ADD are **identical to 0** (e.g. 1,000-update L4 all-ADD: MLP-128 0.22694, GRU 0.28972, TCN 0.28972 for both E and F). Class-A differs only through the frozen candidate mask, and both models classify **X-B** with no reader-type contradiction. F is a control and is never pooled with E.

## 18. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a control, never pooled. Every reported fraction is a mean ± SD over the five E seeds; per-seed results are preserved. The reader set, history grid, target scope, training protocol and decision bars were **predeclared**; every reader and budget is reported, including negative results; no TEST-based selection of reader, history or checkpoint. The paired change from W's 200-update baseline and from X's own 200-budget condition is reported alongside the bars.

## 19. Leakage and Validity

Enforced and tested: (i) the training pool is byte-identical to W's and sequence-disjoint from VAL/TEST; (ii) VAL and TEST are byte-identical to W's; (iii) no TEST-derived selection of budget/reader/history; (iv) held-out operand pairs `{(a,b): b ∈ {0,5}}` never enter fitting; (v) no duplicate training rows; (vi) the 1,000-update run executes exactly 1,000 updates (no example reweighting/repetition); (vii) the reader architecture and parameter count are invariant across budgets; (viii) no future observation enters the reader input; (ix) frozen E/F parameters are never mutated and acquire no gradients; (x) reader parameters are never attached to the frozen model; (xi) checkpoint SHA-256 is unchanged before/after.

## 20. Deterministic Reproduction

The collector copies W's arrays and verifies byte identity (per-array SHA-256 recomputed and matched, plus the source-artifact SHA-256). The analyzer's live fits are deterministic per fit (fixed seed) and executed single-threaded per worker; the 200-budget condition reproduces W exactly (`max |X₂₀₀ − W₂₀₀| = 0.0`), which is the strongest available determinism check since it ties X to a previously published artifact. The full analyzer was not re-run end-to-end (a single run is ≈6.6 h on this 14 GB / 16-core host under memory pressure); determinism is instead enforced by the deterministic per-fit seed (`_fit_seed` = W's `reader_seed` at the top scale), the single-threaded workers, order-independent aggregation, and the exact bridge. Checkpoint SHA-256 verified before/after.

## 21. Causal Status

**Observational with respect to RE:Reason.** X deliberately manipulates the update budget for the **analysis-only reader**, so the comparison is causal *within the reader experiment*: "increasing the analysis reader's optimizer budget under the fixed data/model/protocol did / did not improve its held-out performance." It is **not** a causal statement about RE:Reason. The four levels remain distinct: **interface availability** (U) ≠ **reader learnability** (V, W, X) ≠ **frozen-state representation** (T/S) ≠ **architecture sufficiency**. No intervention is performed or licensed.

## 22. Decision Classification

**X-B — persistent bounded-reader limitation** (E primary; F agrees).

Pre-registered bars (declared conventions): `recover_bar = 0.50`, `heldout_weak = 0.25`, `heldout_material = 0.30`, `train_low_bar = 0.50`.

* **X-A** (some reader at L4/full: 1,000-budget test ≥ 0.50 AND held-out ≥ 0.30) — **not met** (best test 0.111; best held-out 0.212).
* **X-B** (every reader's best 1,000-budget test ≤ 0.25 AND held-out ≤ 0.25 AND training accuracy < 0.50) — **met** (0.111 ≤ 0.25; 0.212 ≤ 0.25; max training accuracy 0.499 < 0.50).
* Continuous quantities: best 1,000-budget Class-A test **0.111** (TCN Lfull); best held-out **0.212** (GRU L4); max training accuracy **0.499** (MLP-128 Lfull); Bayes − reader **0.888**; paired changes from 200: MLP-128 +0.004, GRU −0.006, TCN −0.003.

## 23. What X Establishes

* Over the fixed data (W's 10,000-transition pool), model (U/V/W readers), optimizer and splits, a **5× increase in full-batch optimizer updates (200 → 1,000) does not improve target or held-out generalization**: the best Class-A `y_true` accuracy moves 0.116 → 0.111, held-out stays ≤ 0.212, and the Bayes − reader gap is unmoved.
* The answer is **not "the update budget was too small"** — the readers do not trend upward over the trajectory, and the best-on-VAL checkpoint is not preferentially selected late.
* The sequence readers **do not even fit the training pool** (training accuracy 0.07–0.15 at 1,000 updates, near/below chance), while the flat MLP-128 partially fits long-history windows (≤ 0.499) without generalizing — a memorization-without-generalization signature on the flat reader and non-fitting on the sequence readers.
* This **reproduces and extends W-V's direction** at a larger budget: the weak accessibility is not a simple training-data limitation (W) and not a simple update-budget limitation (X). It is **consistent with** a bounded-reader/algorithmic limitation under the tested conditions.
* The X pipeline is internally valid: the 200-budget condition reproduces W **exactly** (`max diff = 0.0`), and E/F agree.

## 24. What X Does Not Establish

* It does **not** prove the information is mathematically unlearnable by any bounded reader; only that the tested readers under the fixed data/protocol behave as measured at these budgets.
* It does **not** show the frozen RE:Reason model can or cannot access the information; **reader learnability ≠ frozen representation ≠ architecture sufficiency**.
* It does **not** establish that a different architecture, width, loss, LR, or schedule would fail or succeed.
* It does **not** prove the optimization budget is irrelevant in general — only that under the tested fixed protocol/data, 1,000 updates is not the lever; a different optimizer/objective could behave differently.
* It does **not** prove a causal mechanism for RE:Reason; the update-budget manipulation is causal only within the analysis reader experiment.
* It does **not** claim the benchmark is unsolvable or the readers optimal.

## 25. Final Answers

* **Does 200 → 1,000 updates improve target generalization?** No. Best Class-A test 0.116 → 0.111 (paired GRU −0.006, TCN −0.003, MLP-128 +0.004). **X-A is not met.**
* **Does it improve held-out operand generalization?** No. All held-out ≤ 0.212 (< 0.30 bar).
* **Do the readers fit the training pool after 1,000 updates?** Essentially no: GRU/TCN training accuracy 0.07–0.15 (chance ≈ 0.10); MLP-128 reaches 0.499 only at L16/Lfull.
* **Is the W-B/V-B gap explained by the optimization budget (under the tested conditions)?** No — it is X-B: the update budget alone does not resolve the limitation, which strengthens (without proving) a bounded-reader/algorithmic reading.
* **Do E and F agree?** Yes (all-ADD identical; both X-B).
* **Was the paired comparison valid?** Yes — the 200-budget condition reproduces W exactly (`max diff = 0.0`).

## 26. Exactly ONE Next Experiment

**Experiment Y — bounded-reader fitting control at fixed capacity and data (measurement-only, frozen checkpoints).** X found the sequence readers **do not fit the training pool at all** (training accuracy 0.07–0.15 at 1,000 updates, near/below chance) under the fixed protocol, which is the one fact that can distinguish "the ADD task is the obstacle" from "the training pipeline is the obstacle". Hold the data (W's 10,000-transition pool), the readers and widths, the splits and the optimizer exactly fixed, and change **only the fitting target**: predeclare an *overfitting / memorization control* in which the same readers are trained with the identical protocol on the fixed pool and evaluated on a small fixed held-out subset versus the **training** rows themselves. Decision — **Y-A**: if the readers drive TRAINING accuracy to ≥ 0.90 on the fixed pool (they can fit it) while test/held-out stay at chance, the limitation is *generalization* (a representation/algorithmic obstacle), and the next step is a representation-level measurement; **Y-B**: if the readers *cannot* fit even the training pool under any predeclared fitting budget, the bottleneck is the reader training procedure/objective itself, and the reader family must be re-examined as an instrument **before** any ADD-specific conclusion. Measurement-only; no frozen weight/architecture change; no candidate/state modification; no Phase 2; specified, not started.

## Required Figures

`docs/reports/figures/experiment_x_*.png` (12): `accuracy_vs_updates`, `heldout_vs_updates`, `bayes_gap_vs_updates`, `train_val_test_vs_updates`, `reader_comparison`, `l4_comparison`, `fullprefix_comparison`, `seen_vs_heldout`, `per_seed`, `classA_vs_alladd`, `update_trajectory`, `decision_summary`.

## Tests

`tests/test_experiment_x.py` — V/W reader reuse and architecture identity; fixed training-pool identity; fixed VAL and TEST identity; parameter-count invariance; exact update counts of 200 and 1,000; deterministic paired seeds; update-200 bridge against W (procedure + stored values); validation-only checkpoint selection; held-out operand exclusion; sequence disjointness; no duplicate data; causal history construction; future-observation leakage rejection; causal reader guards; target-generation correctness; Class-A construction and count reproduction; E/F control; reader/model separation; no gradients on frozen RE:Reason parameters; no parameter mutation; deterministic results and artifacts; figure generation; historical-report immutability; documentation-governance compliance.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_x.py \
    --w-npz experiments/experiment_w_data_scale.npz \
    --output experiments/experiment_x_optimization_budget.npz
PYTHONPATH=src python experiments/analyze_experiment_x.py \
    --npz experiments/experiment_x_optimization_budget.npz \
    --w-results experiments/experiment_w_results.json \
    --u-results experiments/experiment_u_results.json \
    --output experiments/experiment_x_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_x
PYTHONPATH=src pytest tests/test_experiment_x.py -v
PYTHONPATH=src pytest tests/
```

## Artifacts

* `experiments/run_experiment_x.py` — fixed-data collector (copies W's pool10000/VAL/TEST verbatim; records identity hashes)
* `experiments/analyze_experiment_x.py` — analysis (paired 200/1,000-update trajectories)
* `experiments/experiment_x_optimization_budget.npz`, `experiments/experiment_x_optimization_budget_meta.json`
* `experiments/experiment_x_results.json`
* `docs/reports/figures/experiment_x_*.png` (12)
* `tests/test_experiment_x.py`

## Scientific Guardrails

Measurement-only; observational with respect to RE:Reason. The report keeps four things separate: **exact information availability** (U's Bayes ceilings), **reader learnability** (V/W/X), the **frozen model's representation** (T/S), and **causal interpretation** (only within the reader-training manipulation). A failed bounded reader is **not** proof the information is absent; a high Bayes ceiling is **not** evidence the frozen network can exploit it; a negative optimization-budget result is **not** proof that no training change could help. Q–W are **not** edited; X is appended as new evidence. No Phase 2. Negative and borderline findings are preserved.
