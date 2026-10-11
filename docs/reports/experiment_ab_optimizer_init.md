# RE:Reason — Experiment AB: TCN Optimizer-Family and Initialization Diagnostic on the `op_dest` Control

**Date:** 2026-10-10
**Status:** Completed — Measurement only (frozen E/F checkpoints used only to define the test populations and reproduce baseline metrics; no model training; no optimizer on RE:Reason parameters; no architecture/encoder/recurrent-core/state/gate/candidate/commit/readout modification; no state replacement; no candidate manipulation; no feature editing; no oracle/reconstructed state as a reader feature; no branching/verifier; no counterfactual rollout; no Phase 2). The AB readers are separate analysis-only models and never enter frozen model inference. Classification: **AB-A — a predeclared optimizer/initialization qualifies the TCN**.
**Frozen models:** `E_candidate` (primary) and `F_combined` (control), seeds 42–46, SHA-256-pinned (unchanged before/after).
**Data:** `experiments/experiment_ab_optimizer_init.npz`, `experiments/experiment_ab_optimizer_init_meta.json`, `experiments/experiment_ab_results.json` (+ resumable raw fits `experiments/experiment_ab_grid_raw.json`)
**Runners:** `experiments/run_experiment_ab.py`, `experiments/analyze_experiment_ab.py`, tests `tests/test_experiment_ab.py`

---

## 1. Title / Date / Status

Experiment AB — TCN Optimizer-Family and Initialization Diagnostic. Measurement-only, frozen; **AB-A — a predeclared optimizer/initialization qualifies the TCN**. A predeclared optimizer-family set (AdamW, SGD+momentum, SGD+Nesterov, RMSprop) and a single alternative initialization (xavier-uniform) are applied to the existing causal TCN (subject) and causal GRU (positive control) on the directly observable `op_dest` target at fixed learning rate 3e-3 and 5,000 updates. No intervention on the frozen model; no Phase 2. Reports Q–AA are untouched; AB is appended as new evidence.

## 2. Executive Answer

At fixed learning rate 3e-3 and 5,000 updates, two predeclared conditions drive the TCN to a **perfect training fit** while the Adam/default reference reproduces AA exactly; the result is **AB-A**:

* **AdamW (decoupled weight decay) fits the TCN.** Maximum training accuracy **1.000 ± 0.000** (5/5 seeds) at both L4 and Lfull (reached by update ≈38–40), with validation 0.996 (L4) / 0.985 (Lfull) and held-out `op_dest` TEST accuracy **0.996 / 0.987** (all-ADD). Changing only the optimizer family (Adam → AdamW, identical lr/budget/init/data) flips the TCN from 0.287 to 1.000.
* **An xavier-uniform initialization under Adam fits the TCN.** Maximum training accuracy **1.000 ± 0.000** (5/5) at both histories (by update ≈32–36), validation **1.000** and `op_dest` TEST accuracy **1.000** at both histories.
* **The Adam/default reference reproduces AA exactly.** Its five-seed mean maximum training accuracy is **0.28671875** (per-seed [0.26171875, 0.3125, 0.26953125, 0.296875, 0.29296875]) at both histories — bit-identical to AA's lr 3e-3 / 5,000 cell — so the AB trainer, data and seed convention are the same instrument as AA.
* **The remaining predeclared conditions do not fit.** SGD(momentum 0.9) reaches 0.409 ± 0.125 (L4) / 0.388 ± 0.038 (Lfull); SGD(Nesterov) 0.410 ± 0.124 / 0.387 ± 0.036; RMSprop 0.287 ± 0.021 at both histories (RMSprop behaves like Adam, not like a rescue). None reaches 0.90.
* **The GRU control fits `op_dest` at every condition and history** (1.000, 5/5), validating the labels/data/target/protocol/grid.

Interpretation: the TCN's non-fitting on this directly observable control is an **optimizer/initialization artifact under the fixed Adam/default recipe**, **not** an intrinsic limitation of the TCN body or parameterization; with a qualified recipe the TCN **does** fit `op_dest` and generalizes to held-out test. This is an **instrument-qualification** statement about the analysis reader; it carries **no** latent-state, ADD-learnability or RE:Reason architecture claim, and the recipe is qualified **on `op_dest` only**. **No intervention is justified.**

## 3. Z/AA Evidence Context

* **Z (Z-B)**: on the directly observable four-way `op_dest` target the GRU control fits to 1.000 while the TCN reaches only 0.287 training accuracy under the fixed Adam/default protocol; Z classified the TCN non-fitting as a **target-agnostic instrument concern** and recommended diagnosing the training/optimization recipe.
* **AA (AA-B)**: a predeclared learning-rate × update-budget grid (`{3e-4…3e-2} × {1,000,5,000}`) does **not** rescue the TCN (mean 0.256–0.311, no recipe ≥ 0.90) while the GRU fits at every recipe; AA ruled out learning rate, update budget, and (via a non-training implementation audit) alignment/indexing/gradient-path defects, and named **optimizer type** and **initialization scheme** as the single remaining optimization-side hypotheses.
* **AB** runs exactly that predeclared optimizer-family-and-initialization test.

## 4. Research Question and Hypotheses

> Can a predeclared change in **optimizer family** or **initialization scheme** drive the existing causal TCN to fit the directly observable `op_dest` target — or does the TCN remain unqualified under the tested conditions?

* **AB-A**: some predeclared condition is robust for the TCN at both L4 and Lfull (the non-fitting is an optimization-recipe artifact → the TCN is usable with the qualified recipe).
* **AB-MIX**: a condition is robust at only one history, or reaches the mean level without the 4/5-seed robust criterion.
* **AB-B**: no condition meets the robust criterion at either history.
* **AB-INVALID**: the trainable-parameter preflight, a data/split invariant, or the GRU control fails, or the `adam_ref` condition does not reproduce AA's Adam baseline.

## 5. Preregistration (declared before aggregate inspection)

* **Conditions** (each fully specifies optimizer family + initialization), all at lr 3e-3, budget 5,000, weight decay 1e-2 where supported:

| Condition | Optimizer | Initialization |
|---|---|---|
| `adam_ref` | Adam (betas 0.9/0.999, eps 1e-8) | PyTorch default (deterministic reference; must reproduce AA) |
| `adamw` | AdamW (betas 0.9/0.999, eps 1e-8) | PyTorch default |
| `sgd_mom` | SGD (momentum 0.9, nesterov False) | PyTorch default |
| `sgd_nesterov` | SGD (momentum 0.9, nesterov True) | PyTorch default |
| `rmsprop` | RMSprop (alpha 0.99, eps 1e-8, momentum 0, centered False) | PyTorch default |
| `adam_xavier` | Adam | xavier-uniform conv/head weights + zero biases |

* **Separation of questions:** the initialization question is tested **only under Adam** (`adam_ref` vs `adam_xavier`), so the optimizer-family and initialization questions stay separately interpretable. Optimizer-specific settings are recorded; effective regularization is **not** claimed identical across families.
* **Matched initialization:** every condition of a (seed, reader, history) is reset to the same deterministic initialization seed (`aw.reader_seed(seed, reader, "op_dest", rep, top, "main")`).
* **Data:** Z's exact `op_dest` dataset (Y's fixed 256-example subset, canonical VAL/TEST), copied verbatim and byte-verified.
* **Fitting criterion:** `fit_bar = 0.90`. A condition is **robust** at a history iff the five-seed mean maximum training accuracy ≥ 0.90 **and** ≥ 4/5 seeds reach 0.90.
* **Decision:** AB-INVALID if the preflight/invariant/GRU-control fails or `adam_ref` does not reproduce AA; else AB-A if some condition is robust at both histories; else AB-MIX if robust at exactly one history or reaches the mean level without the seed criterion; else AB-B. Checkpoint selection is VAL-only; the best-on-TRAIN state is diagnostic; TEST never selects condition/reader/history.

## 6. Frozen Model Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, official optimizer, and loss. **Nothing modified.** Checkpoints are read only to define the test populations; no RE:Reason parameter acquires a gradient, and reader parameters are disjoint from the frozen model's.

## 7. Mandatory Trainable-Parameter Preflight (non-training)

A mandatory trainable-parameter preflight was run before the grid on a **disposable** TCN (never used in the grid). It verifies that the loss produces gradients on **all** registered trainable parameters and that the optimizer actually updates them (distinguishing an absent computational path from a numerically/gradient-dead or merely-untrained model). It **passed** at both histories (diagnostic batch = 256 real subset examples, seed 42).

| Check | L4 | Lfull | Meaning |
|---|---|---|---|
| `n_trainable_params` = expected | **True** (28,188) | **True** (28,188) | parameter count matches the reader definition |
| `n_param_tensors` | 10 | 10 | 4 convs (weight+bias) + head (weight+bias) |
| `all_in_optimizer` / `all_requires_grad` | **True** / **True** | **True** / **True** | no parameter is detached or frozen |
| `loss_finite` / `loss_decreased` | **True** / **True** (1.3900 → 1.3863) | **True** / **True** (1.3900 → 1.3863) | finite class-balanced CE that decreases |
| `aggregate_grad_norm` | 5.01e-02 | 5.07e-02 | nontrivial aggregate gradient |
| `grad_finite` / `dead_stages` | **True** / `[]` | **True** / `[]` | no NaN/Inf; no stage with identically-zero gradient |
| per-stage `grad_norm` | all > 0 (conv0 2.2e-3 … head 4.6e-2) | all > 0 | every stage receives gradient |
| per-stage update `effectively_unchanged` | all **False** | all **False** | every stage is moved by the optimizer |
| `logit_change` after 20 diagnostic steps | 2.72 | 2.71 | the optimizer demonstrably changes the output |
| **`passed`** | **True** | **True** | — |

A test-level **negative control** (`tests/test_experiment_ab.py::test_grad_stats_flags_dead_stage`) confirms the dead-stage detector can *fail*: a fully frozen model is flagged all-dead while a trainable model is not. The preflight therefore does not trivially pass, and it found **no** trainable-parameter registration, finiteness or update defect. Consequence: the optimizer grid proceeded; no AB-INVALID.

## 8. Data Identities, Split Checks, and Checkpoint Hashes

| Property | Value |
|---|---|
| Source artifact | `experiments/experiment_z_op_dest.npz` (SHA-256 `917dcb373d4e2cecbc0a3a2a5c888efed3d97a4dc155d8d0f411dbe2e8364931`) |
| Upstream Y | `experiments/experiment_y_fitting_control.npz` (SHA-256 `d22f881a973aeb6925c2a66edab09c34a5625041dde40fc9eb669501de0a6de5`) |
| Upstream W | `experiments/experiment_w_data_scale.npz` (SHA-256 `fd7b32535401f9b70f7433bb1c8ef9ea66dfa0da21960dfc9a3d600c7df7efe6`) |
| Subset | Y's/Z's exact 256 transitions/seed (1,280 pooled); selection seed `20261010` |
| Arrays | 53 data arrays copied verbatim from Z; byte-identical, per-array SHA-256 recorded |
| `op_dest` train pooled | [306, 315, 300, 359] |
| `op_dest` VAL pooled | [1032, 1000, 995, 973] |
| `op_dest` TEST (E and F) | [3928, 3975, 3943, 3917] |
| Class coverage | all four classes present in train (per seed), VAL (per seed) and TEST |
| `op_dest` recoverability | train / VAL / TEST = 1.000 (exact `argmax(x_t[3:7])`) |
| Subset ∩ VAL / Subset ∩ TEST / VAL ∩ TEST | 0 / 0 / 0 |
| Checkpoints | `E_candidate`/`F_combined` seeds 42–46, SHA-256 re-verified unchanged before/after |

## 9. Optimization Protocol and Initialization

Full-batch, class-balanced cross-entropy, evaluation every 10 updates, exactly 5,000 updates; weight decay 1e-2 where supported; **only** the optimizer family and initialization scheme are varied (learning rate and budget are held fixed at the AB-qualified values 3e-3 / 5,000, which AA's grid covered). The `adam_ref` condition uses the identical Adam/default recipe as AA; its trainer is byte-identical in ops/order to `ay.train_trajectory` at lr = 3e-3 and reproduces AA's value exactly (`tests/test_experiment_ab.py::test_reproduce_aa_in_results`). The alternative initialization (**xavier-uniform** on conv/head weights, **zero** biases) rescales the conv0 weight magnitude modestly (mean |w| 6.2e-2 → 8.1e-2) and zeroes the biases; the default initialization is left as PyTorch supplies it.

## 10. Checkpoint Reporting

For every condition the **final** state (end of the budget), the **best-on-VAL** state (validation-only selection) and the **best-on-TRAIN** state (diagnostic only) are reported. No TEST information enters selection.

## 11. Fitting and Generalization Metrics

Training accuracy (max observed and final) on the 256-example subset; training/validation loss and accuracy at updates 200, 400, 600, 800, 1,000 and each subsequent 1,000-update milestone through 5,000; validation accuracy; and `op_dest` TEST accuracy on all-ADD and Class-A populations in the best-on-VAL state.

## 12. Full Results — TCN (Subject)

Five-seed mean ± SD of the maximum training accuracy on the 256-example subset. `chance = 0.25`, `fit_bar = 0.90`, `nfit` = seeds ≥ 0.90.

| Condition | L4 mean ± SD | nfit | L4 final | L4 test (all-ADD) | full mean ± SD | nfit | full final | full test (all-ADD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `adam_ref` (Adam, default) | 0.287 ± 0.021 | 0 | 0.261 | 0.251 | 0.287 ± 0.021 | 0 | 0.248 | 0.251 |
| `adamw` (AdamW, default) | **1.000 ± 0.000** | **5** | 1.000 | **0.996** | **1.000 ± 0.000** | **5** | 1.000 | **0.987** |
| `sgd_mom` (SGD+mom, default) | 0.409 ± 0.125 | 0 | 0.263 | 0.373 | 0.388 ± 0.038 | 0 | 0.224 | 0.338 |
| `sgd_nesterov` (SGD+nesterov, default) | 0.410 ± 0.124 | 0 | 0.263 | 0.373 | 0.387 ± 0.036 | 0 | 0.224 | 0.335 |
| `rmsprop` (RMSprop, default) | 0.287 ± 0.021 | 0 | 0.267 | 0.251 | 0.287 ± 0.021 | 0 | 0.238 | 0.251 |
| `adam_xavier` (Adam, xavier) | **1.000 ± 0.000** | **5** | 1.000 | **1.000** | **1.000 ± 0.000** | **5** | 1.000 | **1.000** |

Per-seed maximum training accuracy:
* `adam_ref` (both histories): [0.2617, 0.3125, 0.2695, 0.2969, 0.2930].
* `adamw` (both histories): [1.0, 1.0, 1.0, 1.0, 1.0] — reached by update ≈38 (L4) / ≈40 (Lfull).
* `adam_xavier` (both histories): [1.0, 1.0, 1.0, 1.0, 1.0] — reached by update ≈32 (L4) / ≈36 (Lfull).
* `sgd_mom` L4 [0.4961, 0.2695, 0.3438, 0.3594, 0.5781]; Lfull [0.4023, 0.3359, 0.4141, 0.4258, 0.3633].
* `sgd_nesterov` L4 [0.4961, 0.2695, 0.3438, 0.3633, 0.5781]; Lfull [0.4023, 0.3359, 0.4102, 0.4219, 0.3633].
* `rmsprop` (both histories): [0.2617, 0.3125, 0.2695, 0.2969, 0.2930].

Every condition is numerically finite (no divergence); every run's `op_dest` TEST accuracy is reported in the best-on-VAL state.

## 13. Full Results — GRU (Positive Control)

The GRU reaches **1.000 ± 0.000** maximum training accuracy (5/5 seeds), validation 1.000 and `op_dest` TEST accuracy 1.000 at **every** one of the six conditions at both histories (L4 final 1.000, full final 1.000). The control therefore robustly fits `op_dest` under every optimizer family and under both initializations, validating the labels/data/target/protocol/grid across the whole AB design — including the conditions under which the TCN fails (`adam_ref`, `rmsprop`).

## 14. Training/Validation Curves and Best-Validation Checkpoints

* **`adamw` (TCN)**: training accuracy is already 1.000 at the first milestone (update 200) and stays there; validation rises from 0.992 → 0.996 (L4) and 0.961 → 0.985 (Lfull) across 200–5,000. The fit is immediate and stable.
* **`adam_xavier` (TCN)**: training **and** validation are 1.000 at every milestone (200–5,000) at both histories.
* **`adam_ref` (TCN)**: training accuracy is flat near chance (e.g. L4 0.235 → 0.268 → 0.261 at updates 200 → 1,000 → 5,000); there is no upward trend.
* **`sgd_mom`/`sgd_nesterov` (TCN)**: training accuracy climbs slowly and partially (≈0.39–0.41), never approaching the bar.
* Best-on-VAL checkpoint indices for the qualified conditions are early (the fit is reached well before 200 updates); there is no later checkpoint worth selecting. All checkpoints are validation-selected; TEST is never used.

## 15. Optimizer-Family and Initialization Effects

* **Optimizer family (default init) matters decisively.** Adam 0.287 and RMSprop 0.287 (both at chance) versus AdamW 1.000; SGD(momentum/Nesterov) reaches only ≈0.39–0.41 (partial). The Adam→AdamW change alters **only** the weight-decay coupling (identical lr, budget, init, data) and flips the TCN from chance to a perfect fit.
* **Initialization under Adam matters decisively.** Default (0.287) versus xavier-uniform (1.000) under the identical Adam recipe; the xavier change is a modest weight rescale (mean |w| 6.2e-2 → 8.1e-2) plus zeroed biases.
* **Two independent levers qualify the TCN.** Decoupled weight decay (AdamW+default) and a re-scaled initialization (Adam+xavier) both reach the bar — so the non-fitting is not intrinsic to the TCN body, but the **precise mechanism is not isolated** by AB: there is no AdamW+xavier cell, and the xavier condition changes weights *and* biases together.
* **Plausible (not established) hypothesis.** With Adam's coupled L2 weight decay the decay term `wd·p` is added to the raw gradient *before* the adaptive normalization; for the TCN's small default-initialized parameters this may dominate the small data gradient and suppress the effective data step, whereas AdamW applies decay directly to the parameters and xavier rescaling moves the network out of that regime. AB is measurement-only and does **not** test this; it is recorded as an open hypothesis.
* **Control:** the GRU is at ceiling under every condition, so the TCN's flat surface under Adam/RMSprop is not a grid or pipeline artifact.

## 16. E/F Controls

The `op_dest` test arrays are identical for E and F and the readers are model-independent, so E and F reader accuracies on all-ADD are **exactly identical** (max |E − F| = 0.0; `tests/test_experiment_ab.py::test_ef_control`). F is a control and is never pooled with E.

## 17. Decision Classification

**AB-A — a predeclared optimizer/initialization qualifies the TCN.** Pre-registered rule: AB-INVALID if the preflight/invariant/GRU-control fails or `adam_ref` does not reproduce AA; else AB-A if some condition is robust at both histories; else AB-MIX if robust at exactly one history or reaches the mean level without the 4/5-seed criterion; else AB-B. Robust = five-seed mean max-train ≥ 0.90 **and** ≥ 4/5 seeds ≥ 0.90.

* **TCN robust conditions:** `adamw` and `adam_xavier`, at **both** L4 and Lfull (mean 1.000, 5/5 seeds). **TCN mean-only conditions:** none. → AB-A.
* **GRU control:** robust at both histories for all six conditions → `control_ok = True`.
* **Reproduce AA:** `adam_ref` TCN mean = 0.28671875 = AA's lr 3e-3 / 5,000 value (per-seed identical) → `reproduce_aa = True`.
* **AB-MIX / AB-B / AB-INVALID** — not met. **AB-A** — met.

## 18. What AB Establishes

* The **non-fitting is not intrinsic to the TCN body/parameterization.** Two predeclared conditions (AdamW with default init; AdamW-alternative xavier init under Adam) drive the TCN to a perfect training fit (1.000, 5/5) and near-perfect held-out `op_dest` TEST accuracy (0.987–1.000) at both histories, while the Adam/default reference reproduces AA exactly.
* The **lever is optimizer-family/initialization**, not learning rate or update budget (AA) — the change that qualifies the TCN holds lr/budget/init/data fixed and varies only the optimizer family (Adam → AdamW) or the initialization (default → xavier under Adam).
* The **instrument is valid**: the GRU control fits `op_dest` under every condition, and the trainable-parameter preflight passes (with a negative control), so the fitted case is not a pipeline artifact.
* With a qualified recipe the TCN **is** usable as an instrument for the `op_dest` control — resolving the Z-B/AA-B "TCN cannot fit at all" state to "TCN fits with a qualified optimizer/initialization".

## 19. What AB Does Not Establish

* It does **not** isolate the mechanism: whether the critical factor is the weight-decay coupling (Adam vs AdamW) or the initialization scale (and, within xavier, weights vs biases) is not separated — there is no AdamW+xavier cell.
* It does **not** show the TCN fits the **ADD `y_true` target**; AB only re-qualifies the TCN on the directly observable `op_dest` control. `op_dest` is a label readable from the current observation, not the ADD target.
* It does **not** license any latent-state, ADD-task-learnability, frozen-representation or RE:Reason architecture-sufficiency claim.
* It does **not** re-measure Y/V/W/X; those were run under the **unqualified** Adam/default recipe and remain valid historical results **under that recipe**. Whether their TCN-based conclusions change under the qualified recipe is exactly the next experiment.
* It does **not** prove universal capability: RMSprop also fails, and SGD is only partial, so "fits `op_dest`" is recipe-specific.
* It does **not** re-measure Z/Y/AA; those results are reused unchanged.

## 20. Interpretation Limits and Guardrails

AB concerns the analysis reader, not RE:Reason. Fitting `op_dest` (a directly observable control) does **not** establish arithmetic generalization, ADD-target access, or latent-state sufficiency. The clean split — AdamW/xavier fit, Adam/RMSprop fail — is a statement about the TCN **instrument** under this predeclared recipe set, and the mechanism behind it is left unresolved (weight-decay coupling vs initialization are confounded by the available cells). The qualified recipe re-qualifies the TCN **on `op_dest` only**; any extension to the ADD target is a separate, pre-registered test. All negative results (RMSprop, SGD, `adam_ref`) are preserved.

## 21. Final Answers

* **Does any predeclared optimizer/initialization let the TCN fit `op_dest`?** Yes — AdamW (default init) and xavier-uniform init (under Adam) both reach 1.000 training accuracy (5/5 seeds) at both histories (test `op_dest` 0.987–1.000).
* **Does the GRU control fit `op_dest` across the grid?** Yes, at every condition and history (1.000).
* **Does the Adam/default reference still fail?** Yes (0.287, reproducing AA exactly), as does RMSprop (0.287); SGD is partial (≈0.39–0.41).
* **Is the TCN's non-fitting intrinsic to its body/parameterization?** No — it is an optimizer/initialization artifact under the fixed Adam/default recipe.
* **Is the mechanism identified?** No — decoupled weight decay and initialization rescaling are confounded by the predeclared cells.
* **Is the TCN qualified as an instrument for ADD-target claims?** Not yet — AB qualifies it on `op_dest` only; the ADD `y_true` target must be re-tested under the qualified recipe.

## 22. Exactly ONE Next Experiment

**Experiment AC — TCN re-qualification on the ADD `y_true` target under the AB-qualified recipe (measurement-only, frozen checkpoints).** AB shows the TCN **can** fit the directly observable `op_dest` control once the optimizer/initialization is changed (AdamW, or xavier under Adam), while the Adam/default recipe does not. The open question is whether the TCN's earlier `y_true` non-fitting (Y) — which Z/AA showed is **not** target-specific — also disappears under the qualified recipe. Hold Y's fixed 256-transition subset, the Y `y_true` target, the TCN body/width/head, the splits and the deterministic seeds **exactly fixed**, and re-run Y's fitting control with the analysis readers under the AB-qualified recipe (AdamW, fixed lr 3e-3 and 5,000 updates; and the xavier-under-Adam variant), with the GRU control under the same recipe. Decision — **AC-A**: if the TCN fits the `y_true` subset (robust, 4/5 seeds) the Y/V/W/X TCN-based conclusions are re-examined under the qualified recipe; **AC-B**: if it does not, the `y_true` target is genuinely different from `op_dest` and the instrument- vs representation-limit split for the ADD target stays open. Measurement-only; no frozen weight/architecture change; no RE:Reason modification; no Phase 2; specified, not started.

## Required Figures

`docs/reports/figures/experiment_ab_*.png` (8): `optimizer_comparison`, `init_comparison`, `gru_control`, `per_seed`, `train_val_curves`, `tcn_trajectories`, `reproduce_aa`, `decision_summary`.

## Tests

`tests/test_experiment_ab.py` (36 tests) — the mandatory trainable-parameter preflight (parameter groups/`requires_grad` registration, loss/gradient finiteness, per-stage gradient and update evidence, dead-stage detection with a **frozen negative control**); reuse of Z's `op_dest` dataset (byte identity); data/split/class/recoverability invariants; checkpoint hashes; optimizer-family construction and recorded defaults; deterministic xavier initialization (bounds and zero biases); the Adam/default trainer's protocol identity with Y and exact update counts; determinism; the predeclared robust-fit decision rule (AB-A/AB-MIX/AB-B/AB-INVALID) encoded and tested on synthetic grids; condition completeness; the `adam_ref`↔AA reproduction bridge; the GRU control; the E/F control; VAL-only selection; frozen-model protection; malformed-artifact rejection; figure generation; historical immutability; documentation-governance compliance.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_ab.py \
    --z-npz experiments/experiment_z_op_dest.npz \
    --output experiments/experiment_ab_optimizer_init.npz
# grid is resumable: collect raw fits, then assemble
PYTHONPATH=src python experiments/analyze_experiment_ab.py --collect-only \
    --raw experiments/experiment_ab_grid_raw.json
PYTHONPATH=src python experiments/analyze_experiment_ab.py --use-raw \
    --raw experiments/experiment_ab_grid_raw.json \
    --output experiments/experiment_ab_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_ab
PYTHONPATH=src pytest tests/test_experiment_ab.py -v
PYTHONPATH=src pytest tests/
```

## Artifacts

* `experiments/run_experiment_ab.py` — `op_dest` collector + mandatory trainable-parameter preflight
* `experiments/analyze_experiment_ab.py` — optimizer-family/initialization analysis (resumable: `--collect-only` / `--use-raw`)
* `experiments/experiment_ab_optimizer_init.npz`, `experiments/experiment_ab_optimizer_init_meta.json`
* `experiments/experiment_ab_grid_raw.json` — raw per-seed grid fits (resumable evidence)
* `experiments/experiment_ab_results.json`
* `docs/reports/figures/experiment_ab_*.png` (8)
* `tests/test_experiment_ab.py`

## Scientific Guardrails

Measurement-only; observational with respect to RE:Reason. AB is an **optimizer/initialization diagnostic of the analysis reader**, not a RE:Reason capability experiment: qualifying the TCN on `op_dest` does not prove universal capability (RMSprop and Adam still fail), does not establish arithmetic generalization or ADD-target access, and does not isolate the mechanism (weight-decay coupling vs initialization are confounded). The result **re-qualifies the TCN on the `op_dest` control only**; the earlier Y/V/W/X TCN-based results were produced under the unqualified Adam/default recipe and remain valid historical results under that recipe, and are **not** edited. Q–AA are **not** edited; AB is appended as new evidence. No Phase 2. Negative and inconclusive results are preserved.
