# RE:Reason — Experiment AC: TCN Re-qualification on the ADD `y_true` Target

**Date:** 2026-10-10
**Status:** Completed — Measurement only (frozen E/F checkpoints used only to define the test populations and reproduce baseline metrics; no model training on RE:Reason; no optimizer on RE:Reason parameters; no architecture/encoder/recurrent-core/state/gate/candidate/commit/readout modification; no state replacement; no candidate manipulation; no feature editing; no oracle/reconstructed state as a reader feature; no branching/verifier; no counterfactual rollout; no Phase 2). The AC readers are separate analysis-only models and never enter frozen model inference. Classification: **AC-A (a qualified recipe fits the ADD target) · fit-without-generalization on the separate transfer axis**.
**Frozen models:** `E_candidate` (primary) and `F_combined` (control), seeds 42–46, SHA-256-pinned (unchanged before/after).
**Data:** `experiments/experiment_ac_ytrue_requalification.npz`, `experiments/experiment_ac_ytrue_meta.json`, `experiments/experiment_ac_results.json` (+ resumable raw fits `experiments/experiment_ac_grid_raw.json`)
**Runners:** `experiments/run_experiment_ac.py`, `experiments/analyze_experiment_ac.py`, tests `tests/test_experiment_ac.py`

---

## 1. Title / Date / Status

Experiment AC — TCN Re-qualification on the ADD `y_true` Target. Measurement-only, frozen; **AC-A** on the fitting axis and **fit_without_generalization** on the separate transfer axis. The three AB conditions (Adam/default reference, AdamW/default, Adam/xavier) are applied to the existing causal TCN (subject) and causal GRU (positive control) on the actual register-tracking target `y_true = (operand_a + operand_b) mod 10` over Y's fixed 256-transition subset. No intervention on the frozen model; no Phase 2. Reports Q–AB are untouched; AC is appended as new evidence.

## 2. Executive Answer

On Y's fixed 256-example ADD subset at fixed lr 3e-3 / 5,000 updates, the AB-qualified recipes let the TCN fit the ADD target — but the fit does not transfer:

* **`adamw` (AdamW, default init) fits the TCN at both histories.** Maximum training accuracy **1.0000 ± 0.0000 (5/5 seeds)** at L4 and Lfull (reached by update ≈96 / ≈76), while `adam_ref` reproduces Y's near-chance TCN exactly (**0.1016 ± 0.0000, 0/5**). Changing only the optimizer family (Adam → AdamW) flips the TCN from chance to a perfect subset fit.
* **`adam_xavier` (Adam, xavier init) fits only at the full prefix.** Lfull **1.0000 ± 0.0000 (5/5)** (by update ≈142), but L4 is **split and non-robust** — **0.5258 ± 0.4329**, per-seed [0.215, 0.199, 1.000, 1.000, 0.215], i.e. 2/5 seeds reach 1.000 and 3/5 stall near chance.
* **The fit does not generalize — it is fit-without-generalization.** At the best-on-VAL state of the qualified recipes the TCN's `y_true` TEST accuracy is **≈chance** (Class-A 0.1032–0.1145, held-out operand pairs 0.1179–0.1251, all-ADD 0.1118–0.1204) versus chance 0.10; Y's historical TCN was 0.110/0.111 and Y's historical GRU 0.105/0.093. A perfect 256-example fit that transfers nothing is memorization, not arithmetic recovery.
* **The reproduction bridge is exact.** AC's `adam_ref` first-1,000-update trajectory equals a fresh run of Y's own trainer bit-for-bit (`max |Δ| = 0.0`, every seed/reader/history).
* **The GRU control fits everywhere** (`adam_ref`/`adamw`/`adam_xavier`, 1.0000–0.9992, 5/5), validating the labels/data/target/protocol.

Interpretation: the recipe that lets the TCN learn the trivial, directly observable `op_dest` control (AB) **also** lets it fit the actual ADD `y_true` target on a small fixed subset (`adamw` at both histories) — so the Y/Z/AA "the TCN cannot fit `y_true`" statement is an **optimizer-instrument artifact**, not an ADD-representation limit. But the ADD fit is a **small-subset memorization that does not transfer**: on untouched VAL/TEST the fitted TCN is at chance. AC therefore carries **no** claim about RE:Reason latent-state sufficiency, frozen-representation content, architecture sufficiency, or arithmetic learnability, and licenses no intervention.

## 3. Y / AB Evidence Context

* **Y (Y-MIX)**: on the fixed 256-example `y_true` subset the GRU fits (train-max 0.9992 / 0.9922) but does not transfer (Class-A ≈ 0.10), while the **TCN does not fit even 256 rows** (train-max 0.1016) — the TCN's non-fit was classified as a training/optimization **instrument** concern.
* **Z (Z-B)** and **AA (AA-B)** narrowed that concern to the training recipe (not the target, not the learning rate, not the update budget, not alignment/indexing/gradient defects).
* **AB (AB-A)** resolved it: the TCN fits the directly observable `op_dest` control under **AdamW** or a **xavier initialization under Adam**, while Adam/default reproduces AA exactly and RMSprop/SGD fail.
* **AC** applies the AB-qualified recipes to the actual ADD `y_true` target on Y's subset and asks whether the fit transfers.

## 4. Research Question and Hypotheses

> Does either AB-qualified recipe let the TCN fit the actual ADD `y_true` target on Y's fixed 256-transition subset — and does any fit transfer to the untouched VAL/TEST populations?

* **AC-A — qualified ADD-target fit**: some qualified recipe (`adamw` or `adam_xavier`) is robust at both L4 and Lfull.
* **AC-MIX — partial fitting**: a qualified recipe is robust at only one history, or reaches mean ≥ 0.90 without the 4/5-seed criterion.
* **AC-B — no qualified fit**: neither qualified recipe is robust at either history.
* **AC-INVALID — validity failure**: a preflight/data/checkpoint/determinism/protocol invariant fails (incl. the GRU control or the Y reproduction bridge). Affected results are not evidence about reader capability.
* **Generalization (separate axis, not part of the fitting classification):** `fit_without_generalization`, `material_transfer`, `partial_transfer`, or `not_fitted`.

## 5. Preregistration (declared before aggregate inspection)

* **Conditions** (each fully specifies optimizer family + initialization), all at lr 3e-3, weight decay 1e-2 where supported, 5,000 updates, matched deterministic init per (seed, reader, history):

| Condition | Optimizer | Initialization | Role |
|---|---|---|---|
| `adam_ref` | Adam (betas 0.9/0.999, eps 1e-8) | PyTorch default | reference bridge (must reproduce Y; not a qualified recipe) |
| `adamw` | AdamW (betas 0.9/0.999, eps 1e-8) | PyTorch default | AB-qualified recipe |
| `adam_xavier` | Adam | AB xavier-uniform variant | AB-qualified recipe |

* **Qualified recipes** are `adamw` and `adam_xavier`; `adam_ref` is reported separately and never determines whether a qualified recipe succeeds. The initialization question is tested only under Adam.
* **Data**: Y's exact fixed 256-transition subset and canonical VAL/TEST (carried verbatim in Z's artifact), byte-verified.
* **Fit criterion**: `fit_bar = 0.90`; a recipe is **robust** at a history iff the five-seed mean maximum training accuracy ≥ 0.90 **and** ≥ 4/5 seeds reach 0.90.
* **Fitting decision**: AC-INVALID if preflight/invariant/protocol fails, the `adam_ref` GRU control is not robust at both histories, or `adam_ref` does not reproduce Y's trajectory; else AC-A if some qualified recipe is robust at both histories; else AC-MIX if a qualified recipe is robust at exactly one history or reaches the mean level without the seed criterion; else AC-B.
* **Generalization** is reported separately: at each qualified recipe's best-on-VAL state — `fit_without_generalization` if Class-A ≤ 0.25 AND held-out ≤ 0.25; `material_transfer` if Class-A ≥ 0.50 AND held-out ≥ 0.30; else `partial_transfer`. Bars reuse Y's conventions (weak 0.25, material 0.30, reader-recovery 0.50); no new thresholds are introduced.
* Checkpoint selection is VAL-only; best-on-TRAIN is diagnostic; TEST never selects. No recipe/architecture/hyperparameter change after inspecting results.

## 6. Frozen Model Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, official optimizer, and loss. **Nothing modified.** Checkpoints are read only to define the test populations; no RE:Reason parameter acquires a gradient, and reader parameters are disjoint from the frozen model's.

## 7. Mandatory Trainable-Parameter Preflight (non-training)

A mandatory preflight was run before the grid on **disposable** models (TCN subject and GRU control, both histories) — it verifies that the loss produces gradients on all registered trainable parameters and that the optimizer actually updates them. It **passed** for all four (diagnostic batch = 256 real subset examples, seed 42).

| Check | TCN L4 | TCN Lfull | GRU L4 | GRU Lfull |
|---|---|---|---|---|
| `n_trainable_params` = expected | **True** (28,506) | **True** (28,506) | **True** (28,655) | **True** (28,655) |
| `all_in_optimizer` / `all_requires_grad` | **True** / **True** | **True** / **True** | **True** / **True** | **True** / **True** |
| `loss_decreased` (finite CE) | **True** (2.3062 → 2.3028) | **True** (2.3060 → 2.3028) | **True** (2.3059 → 2.2570) | **True** (2.3064 → 2.2564) |
| `aggregate_grad_norm` | 3.02e-02 | 3.07e-02 | 5.27e-02 | 5.62e-02 |
| `dead_stages` | `[]` | `[]` | `[]` | `[]` |
| per-stage `grad_norm` / update moved | all > 0 / all moved | all > 0 / all moved | all > 0 / all moved | all > 0 / all moved |
| **`passed`** | **True** | **True** | **True** | **True** |

A test-level **negative control** (`tests/test_experiment_ac.py::test_grad_stats_flags_dead_stage`) confirms the dead-stage detector can fail (a fully frozen model is flagged all-dead; a trainable model is not). The preflight found no trainable-parameter registration, finiteness or update defect. Consequence: the grid proceeded; no AC-INVALID on this account.

## 8. Data Identities, Target Derivation, Split Checks, and Checkpoint Hashes

| Property | Value |
|---|---|
| Source artifact | `experiments/experiment_z_op_dest.npz` (SHA-256 `917dcb373d4e2cecbc0a3a2a5c888efed3d97a4dc155d8d0f411dbe2e8364931`) |
| Upstream Y | `experiments/experiment_y_fitting_control.npz` (SHA-256 `d22f881a973aeb6925c2a66edab09c34a5625041dde40fc9eb669501de0a6de5`) |
| Upstream W | `experiments/experiment_w_data_scale.npz` (SHA-256 `fd7b32535401f9b70f7433bb1c8ef9ea66dfa0da21960dfc9a3d600c7df7efe6`) |
| Subset | Y's exact 256 transitions/seed (1,280 pooled); selection seed `20261010` |
| Arrays | 53 data arrays copied verbatim from Z; byte-identical, per-array SHA-256 recorded |
| `y_true` derivation | `(operand_a + operand_b) mod 10`, verified **exactly** in train / VAL / TEST (E and F) |
| Held-out exclusion | fitting subset has **0** rows with `b ∈ {0,5}` (held-out operand protocol) |
| `y_true` train pooled | [130, 130, 130, 130, 130, 130, 125, 125, 125, 125] (256/seed; 6 classes ×26, 4 ×25) |
| `y_true` VAL pooled | [1194, 269, 412, 231, 380, 241, 400, 240, 400, 233] |
| `y_true` TEST (E and F) | [4565, 959, 1551, 972, 1550, 967, 1586, 974, 1642, 997] |
| Class coverage | all 10 classes present in train (per seed + pooled), VAL (per seed + pooled) and TEST |
| Subset ∩ VAL / Subset ∩ TEST / VAL ∩ TEST | 0 / 0 / 0 |
| Checkpoints | `E_candidate`/`F_combined` seeds 42–46, SHA-256 re-verified unchanged before/after |

`y_true` is a computed label (not a directly recoverable one); the verified invariant is the exact derivation rule above, and no target-derived value is used as an input feature (reader inputs are the raw 21-dim observation + validity bit only).

## 9. Reader Architectures, Conditions, and Protocol

* **Readers (reused verbatim from V, unchanged):** causal **TCN** — 52 channels, kernel 3, dilations `[1, 2, 4, 8]`, four causal conv layers, ReLU, last-position linear head; causal **GRU** — single-layer unidirectional, hidden 85, last-step hidden → linear head. Output heads are 10-class (AB used 4-class `op_dest`, so the head alone differs: TCN 28,506 vs 28,188; GRU 28,655 vs 28,139). Per-timestep input = raw 21-dim observation + validity bit = 22 features. No architecture, width, input, causal padding or mask change.
* **Trainer/optimizer/initialization:** AC reuses AB's **exact** implementations (`analyze_experiment_ab.train_ab`, `make_optimizer`, `apply_init`); tests assert object identity. The xavier variant is AB's: `xavier_uniform_` on Conv1d/Linear weights + zero biases (and, for the GRU control, the analogous treatment of `weight_ih/hh` + zero `bias_ih/hh`). The `adam_ref` condition is bit-identical in ops/order to Y's `train_trajectory`.
* **Seeds:** the `y_true` fit seed `aw.reader_seed(seed, reader, "y_true", rep, top, "main")` (Y's exact convention). Full-batch class-balanced cross-entropy, eval every 10 updates, 5,000 updates. Grid = 3 conditions × 2 readers × 2 histories × 5 seeds = **60 fits**. Checkpoint selection VAL-only; best-on-TRAIN diagnostic.
* **Determinism note (reported explicitly):** the reader fits are executed in worker processes pinned to one torch thread (`torch.set_num_threads(1)`). At a fixed thread count the fits are deterministic across processes; the GRU full-prefix trajectory is **thread-sensitive** and numerically chaotic (see §15). This affects only the intermediate milestone record, never the fit maxima or the generalization metrics.

## 10. Checkpoint Reporting

For every condition the **final** state, the **best-on-VAL** state (validation-only selection) and the **best-on-TRAIN** state (diagnostic only) are reported. No TEST information enters selection.

## 11. Fitting and Generalization Metrics (separate axes)

Training accuracy (max observed and final) on the 256-example subset; training/validation loss and accuracy at updates 200, 400, 600, 800, 1,000 and every subsequent 1,000 through 5,000; validation accuracy; and `y_true` TEST accuracy on all-ADD, Class-A (frozen `cand_ok` mask) and held-out operand pairs in the best-on-VAL state. **Fitting and generalization are classified separately** — test generalization never enters the fitting verdict.

## 12. Fit Results — TCN (Subject)

Five-seed mean ± SD of the maximum training accuracy on the 256-example subset. `chance = 0.10`, `fit_bar = 0.90`, `nfit` = seeds ≥ 0.90.

| Condition | L4 mean ± SD | nfit | L4 final | full mean ± SD | nfit | full final |
|---|---:|---:|---:|---:|---:|---:|
| `adam_ref` (reference) | 0.1016 ± 0.0000 | 0 | 0.1000 | 0.1016 ± 0.0000 | 0 | 0.1008 |
| `adamw` (**qualified**) | **1.0000 ± 0.0000** | **5** | 1.0000 | **1.0000 ± 0.0000** | **5** | 1.0000 |
| `adam_xavier` (**qualified**) | 0.5258 ± 0.4329 | 2 | 0.4609 | **1.0000 ± 0.0000** | **5** | 1.0000 |

Per-seed maximum training accuracy:
* `adam_ref` (both histories): [0.102, 0.102, 0.102, 0.102, 0.102].
* `adamw` (both histories): [1.0, 1.0, 1.0, 1.0, 1.0] — reached by update ≈96 (L4) / ≈76 (Lfull).
* `adam_xavier` L4: **[0.215, 0.199, 1.000, 1.000, 0.215]** (bimodal; 2/5 seeds fit); Lfull: [1.0, 1.0, 1.0, 1.0, 1.0] (by update ≈142).

The TCN `adam_ref` value reproduces Y's near-chance TCN (0.1016), and `adamw` is the robust qualified recipe at **both** histories.

## 13. Fit Results — GRU (Positive Control)

The GRU fits the subset at **every** condition and history: `adam_ref` 1.0000 (L4) / 0.9992 ± 0.0017 (Lfull), `adamw` 1.0000 / 1.0000, `adam_xavier` 1.0000 / 1.0000 (all 5/5 seeds). The control therefore validates the labels/data/target/protocol/grid under all three conditions, including the conditions under which the TCN's L4 xavier arm fails.

## 14. Training/Validation Trajectories and Best-Validation Checkpoints

* **TCN `adamw`**: training accuracy is 1.000 from the first milestone (200) through 5,000 at both histories; validation stays ≈0.10–0.12 (chance) throughout — no transfer develops.
* **TCN `adam_xavier` Lfull**: 1.000 from update 200; **L4**: flat ≈0.46 for the seed-mean (with two seeds at 1.0 and three stalled near 0.20).
* **TCN `adam_ref`**: flat near chance (0.10) at every milestone.
* Best-on-VAL checkpoint indices are early for the fitting arms; there is no later checkpoint worth selecting. All checkpoints are validation-selected; TEST is never used.

## 15. Reproduction of Y (`adam_ref` Bridge) and Determinism

`adam_ref` at 5,000 updates is Y's original recipe (Adam 3e-3 / wd 1e-2, default init, Y's seed). Two bridges are reported:

* **Trainer identity (gating).** AC's per-seed `adam_ref` first-1,000-update milestones vs a **fresh run of Y's own trainer** (`analyze_experiment_y.train_trajectory`) with the identical seed/data/optimizer/init: **`max |Δ| = 0.0` at every seed, reader and history** (`identical = True`). The trainer and initialization are verified identical, so AC's `adam_ref` **is** Y's protocol.
* **Stored-record bridge (informational, not gating).** AC's `adam_ref` 5-seed milestone means vs Y's **stored JSON** means: **max |Δ| = 0.091 (train) / 0.102 (val)**, worst at GRU\|Lfull\|800 (train) and TCN\|Lfull\|400 (val). This residual is **cross-environment numerical drift**, not a protocol mismatch: the GRU full-prefix fitting trajectory is thread-sensitive and chaotic (a single seed's milestone-200 train accuracy moves from 0.473 to 0.566 between 8-thread and 1-thread execution), and the near-chance TCN's validation argmax is correspondingly unstable. It affects only the intermediate milestone record; the fit maxima (GRU Lfull train-max 0.9992 vs Y 0.9922) and every generalization metric are stable.

This is reported explicitly per the experiment specification; the reproduction criterion used for validity is the exact trainer-identity check, not the environment-sensitive stored record.

## 16. Generalization Assessment (Separate Axis)

At each qualified recipe's best-on-VAL state:

| Recipe | history | train-max | all-ADD TEST | Class-A TEST | seen (Class-A) | held-out (Class-A) |
|---|---|---:|---:|---:|---:|---:|
| `adamw` | L4 | 1.000 | 0.1204 | 0.1032 | 0.0998 | 0.1179 |
| `adamw` | Lfull | 1.000 | 0.1118 | 0.1069 | 0.1041 | 0.1182 |
| `adam_xavier` | Lfull | 1.000 | 0.1165 | 0.1145 | 0.1125 | 0.1251 |

All values are at chance (0.10 for the 10-class target). For comparison, Y's historical TCN: Class-A 0.1099 (L4) / 0.1110 (Lfull), held-out 0.2123 / 0.2152; Y's historical GRU: Class-A 0.1054 / 0.0934, held-out 0.1000 / 0.0990. **Verdict: fit_without_generalization** — a perfect 256-example subset fit (train-max 1.000) with Class-A and held-out accuracy ≤ 0.125. A fit without meaningful transfer is **not** arithmetic recovery.

## 17. E/F Controls

The `y_true` test arrays are identical for E and F and the readers are model-independent, so E and F reader accuracies on all-ADD are **exactly identical** (max |E − F| = 0.0; `tests/test_experiment_ac.py::test_ef_control`). F is a control and is never pooled with E.

## 18. Decision Classification

**Fitting: AC-A — a qualified recipe fits the ADD target.** Pre-registered rule as in §5.
* **TCN qualified robust:** `adamw` at **both** L4 and Lfull; `adam_xavier` at **Lfull only**. → `both = [adamw]` → **AC-A**.
* **GRU control:** robust at both histories for all three conditions → `control_ok = True`.
* **Reproduction:** trainer identity exact (`max |Δ| = 0.0`) → `reproduce_y = True`.
* **AC-MIX / AC-B / AC-INVALID** — not met. **AC-A** — met.

**Generalization: fit_without_generalization** (reported separately; §16).

## 19. What AC Establishes

* The TCN **can fit the actual ADD `y_true` target** on Y's fixed subset under the AB-qualified recipe — `adamw` at both histories (1.0000, 5/5), `adam_xavier` at the full prefix (1.0000, 5/5) — while the original Adam/default recipe reproduces Y's near-chance TCN exactly. So Y's "the TCN does not fit even 256 rows" is an **optimizer-instrument artifact**, not an ADD-representation limit.
* The recipe lever is the **optimizer/initialization**, matching AB: the change that qualifies the TCN holds lr/budget/init/data fixed and varies only the optimizer family (Adam → AdamW) or the initialization (default → xavier under Adam).
* The ADD fit **does not transfer**: at chance on all-ADD, Class-A and held-out operand pairs — a fit-without-generalization (memorization) signature identical in kind to Y's GRU.
* The **protocol is validated**: a fresh run of Y's own trainer reproduces AC's `adam_ref` bit-for-bit, the GRU control fits everywhere, and the trainable-parameter preflight passes.

## 20. What AC Does Not Establish

* It does **not** establish arithmetic generalization or ADD-task recovery: the fit transfers nothing (Class-A/held-out ≤ 0.125).
* It does **not** license any RE:Reason claim — latent-state sufficiency, frozen-representation content, or architecture sufficiency — the readers are separate analysis models on a computed label.
* It does **not** establish that the ADD target is mathematically unlearnable, nor that it is "inherently different" from `op_dest` in any way AC has identified; AC only reports that these tested recipes fit without transfer.
* It does **not** isolate the mechanism (weight-decay coupling vs initialization scale) — AB's confound persists; `adam_xavier`'s L4 arm is also split (2/5 seeds), so xavier alone is not a robust L4 recipe.
* It does **not** show that `adamw` is robust to hyperparameters/architectures beyond the predeclared cells; no grid expansion was performed.
* It does **not** re-measure Y/V/W/X; those remain valid historical results under the **unqualified Adam/default** recipe. Whether the qualified recipe changes the V/W/X Class-A ADD-target access conclusions is the next experiment.

## 21. Interpretation Limits and Guardrails

AC measures analysis-reader behavior, not RE:Reason parameter learning. A successful small-subset fit does **not** establish arithmetic generalization, and fit-without-generalization must **not** be read as arithmetic recovery. Failure to fit a subset would **not** establish information absence or unlearnability (AC does fit, so this is moot but preserved). The GRU full-prefix trajectory is thread-sensitive/chaotic and Y's stored milestone record is not bit-reproducible in this environment; the reproduction criterion is therefore the exact trainer-identity check, and the drift is reported. All negative results (e.g. `adam_xavier` L4) are preserved.

## 22. Final Answers

* **Does a qualified recipe fit the ADD `y_true` target?** Yes — `adamw` at both histories (1.0000, 5/5); `adam_xavier` at Lfull only (1.0000, 5/5), split at L4 (0.5258, 2/5).
* **Does the original Adam/default recipe fit it?** No (0.1016, 0/5), reproducing Y's TCN exactly.
* **Does either AB-qualified recipe transfer to VAL/TEST?** No — Class-A and held-out accuracy ≈ chance (0.10–0.125); **fit_without_generalization**.
* **Is AC's `adam_ref` Y's protocol?** Yes — a fresh run of Y's own trainer reproduces it bit-for-bit (`max |Δ| = 0.0`).
* **Fitting classification / generalization classification?** **AC-A** / **fit_without_generalization**.
* **Does this license a RE:Reason or arithmetic claim?** No.

## 23. Exactly ONE Next Experiment

**Experiment AD — qualified-recipe re-run of the Class-A ADD-target access test (V's design; measurement-only, frozen checkpoints).** AC shows the AB-qualified recipe (`adamw`) lets the TCN fit the ADD `y_true` subset, but the fit memorizes rather than transfers. The V/W/X ADD-target-access conclusions were produced with the **unqualified** Adam/default TCN instrument. AD holds V's exact design fixed (ordinary TRAIN ADD pool, canonical VAL, untouched TEST, Class-A / held-out populations, the predeclared GRU/TCN readers and histories) and re-runs the TCN under the **`adamw` recipe** with the GRU control under the same recipe, at V's full-pool training scale and update protocol. Decision — **AD-A**: the qualified TCN materially improves Class-A accuracy or held-out operand accuracy over V's unqualified TCN result (the earlier V/W/X TCN-based conclusions were confounded by the optimizer instrument and must be re-examined); **AD-B**: it does not (the weak ADD-target accessibility is not an optimizer artifact of the TCN, strengthening the bounded-reader/algorithmic reading). Measurement-only; no frozen weight/architecture change; no RE:Reason modification; no Phase 2; specified, not started.

## Required Figures

`docs/reports/figures/experiment_ac_*.png` (8): `tcn_by_condition`, `gru_control`, `training_trajectory`, `reproduce_y`, `fit_vs_generalization`, `heldout_generalization`, `per_seed`, `decision_summary`.

## Tests

`tests/test_experiment_ac.py` (37 tests) — the mandatory trainable-parameter preflight (parameter groups/`requires_grad`, finiteness, per-stage gradient/update evidence, dead-stage detection with a **frozen negative control**); reuse of Y's `y_true` dataset (byte identity); the `(operand_a + operand_b) mod 10` derivation; held-out-pair exclusion; data/split/class/checkpoint invariants; the three conditions and their reuse of AB's exact optimizer/initialization implementations; the **byte-identical-to-Y trainer**; determinism; the fitting decision rule (AC-A/AC-MIX/AC-B/AC-INVALID) and the separate generalization rule, both encoded and tested on synthetic grids; the `adam_ref`↔Y reproduction bridge; the GRU control; the E/F control; VAL-only selection; frozen-model protection; malformed-artifact rejection; figure generation; historical immutability; documentation-governance compliance.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_ac.py \
    --z-npz experiments/experiment_z_op_dest.npz \
    --output experiments/experiment_ac_ytrue_requalification.npz
# grid is resumable: collect raw fits, then assemble
PYTHONPATH=src python experiments/analyze_experiment_ac.py --collect-only \
    --raw experiments/experiment_ac_grid_raw.json
PYTHONPATH=src python experiments/analyze_experiment_ac.py --use-raw \
    --raw experiments/experiment_ac_grid_raw.json \
    --output experiments/experiment_ac_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_ac
PYTHONPATH=src pytest tests/test_experiment_ac.py -v
PYTHONPATH=src pytest tests/
```

## Artifacts

* `experiments/run_experiment_ac.py` — `y_true` collector + mandatory trainable-parameter preflight
* `experiments/analyze_experiment_ac.py` — condition/recipe analysis (resumable: `--collect-only` / `--use-raw`)
* `experiments/experiment_ac_ytrue_requalification.npz`, `experiments/experiment_ac_ytrue_meta.json`
* `experiments/experiment_ac_grid_raw.json` — raw per-seed fits (resumable evidence; retained alongside the report)
* `experiments/experiment_ac_results.json`
* `docs/reports/figures/experiment_ac_*.png` (8)
* `tests/test_experiment_ac.py`

## Scientific Guardrails

Measurement-only; observational with respect to RE:Reason. AC is a **reader training/optimization diagnostic on a small fixed ADD subset**, not a RE:Reason capability experiment: fitting the 256-example subset does not establish arithmetic generalization (the fit transfers nothing), and the AB/AC recipe qualification is a statement about the analysis reader. The GRU full-prefix trajectory is thread-sensitive/chaotic and Y's stored milestone record is not bit-reproducible here; the reproduction criterion is the exact trainer identity, reported with the drift. Q–AB are **not** edited; AC is appended as new evidence. No Phase 2. Negative and inconclusive results are preserved.
