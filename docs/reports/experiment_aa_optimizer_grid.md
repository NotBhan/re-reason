# RE:Reason — Experiment AA: TCN Optimization-Recipe Diagnostic on the `op_dest` Control

**Date:** 2026-10-10
**Status:** Completed — Measurement only (frozen E/F checkpoints used only to define the test populations and reproduce baseline metrics; no model training; no optimizer on RE:Reason parameters; no architecture/encoder/recurrent-core/state/gate/candidate/commit/readout modification; no state replacement; no candidate manipulation; no feature editing; no oracle/reconstructed state as a reader feature; no branching/verifier; no counterfactual rollout; no Phase 2). The AA readers are separate analysis-only models and never enter frozen model inference. Classification: **AA-B — no recipe qualifies**.
**Frozen models:** `E_candidate` (primary) and `F_combined` (control), seeds 42–46, SHA-256-pinned (unchanged before/after).
**Data:** `experiments/experiment_aa_optimizer_grid.npz`, `experiments/experiment_aa_optimizer_grid_meta.json`, `experiments/experiment_aa_results.json` (+ resumable raw fits `experiments/experiment_aa_grid_raw.json`)
**Runners:** `experiments/run_experiment_aa.py`, `experiments/analyze_experiment_aa.py`, tests `tests/test_experiment_aa.py`

---

## 1. Title / Date / Status

Experiment AA — TCN Optimization-Recipe Diagnostic. Measurement-only, frozen; **AA-B — no recipe qualifies**. A predeclared crossed optimizer grid (learning rate × update budget) is applied to the existing causal TCN (subject) and causal GRU (positive control) on the directly observable `op_dest` target. No intervention on the frozen model; no Phase 2. Reports Q–Z are untouched; AA is appended as new evidence.

## 2. Executive Answer

Over the full predeclared grid the GRU control fits `op_dest` perfectly at **every** recipe, while the TCN **never** approaches the fitting bar; the result is **AA-B**:

* **The GRU positive control is robust everywhere.** Across all 10 (learning-rate, budget) combinations and both histories it reaches maximum training accuracy **1.000 ± 0.000** (5/5 seeds), validation 1.000 and `op_dest` TEST accuracy 1.000. The labels, data path, target construction, training protocol and grid machinery are therefore valid.
* **The TCN does not fit `op_dest` at any predeclared recipe.** Its five-seed mean maximum training accuracy ranges **0.256–0.311** across the whole grid (chance = 0.25), with **no** recipe reaching 0.90 and **no** recipe reaching even the mean-only (seed-dependent) level. The best cell is L4, lr = 3e-4, budget = 5,000 (mean **0.311 ± 0.064**; per-seed [0.258, 0.312, 0.270, 0.297, 0.418]); every other cell is 0.256–0.292. Test `op_dest` accuracy is 0.251–0.276, all runs are numerically finite, and increasing the budget from 1,000 to 5,000 or the learning rate from 3e-4 to 3e-2 does not move the TCN materially.

Interpretation: under this tested optimizer grid and implementation the TCN **fails to fit** the directly observable `op_dest` control. This does **not** prove the TCN's body or parameterization is uniquely responsible, and it carries **no** latent-state, ADD-learnability or RE:Reason architecture-sufficiency claim; **no intervention is justified**.

## 3. Z/AA Evidence Context

* **U/V/W/X**: the raw interface identifies the target exactly (U-D) but bounded flat and sequence readers stay weak; the weakness is not a data-scale (W-B) or optimization-budget (X-B) matter over the tested conditions.
* **Y**: on a fixed 256-example `y_true` subset the GRU fits but does not transfer, while the TCN does not fit even the small subset → **Y-MIX**.
* **Z**: on the directly observable four-way `op_dest` target the GRU control fits to 1.000 while the TCN reaches only 0.287 training accuracy → **Z-B**; Z named exactly this optimizer-recipe grid as Experiment AA.

## 4. Research Question and Hypotheses

> Does a predeclared optimizer grid (learning rate × update budget) allow the existing TCN implementation to fit the directly observable `op_dest` target, or does the TCN remain unqualified under the tested conditions?

* **AA-A**: some recipe meets the robust fitting criterion at both L4 and Lfull.
* **AA-MIX**: a recipe meets it at only one history, or reaches the mean level without the 4/5-seed robust criterion.
* **AA-B**: no recipe meets the robust criterion at either history.
* **AA-INVALID**: the implementation audit, a data/split invariant, deterministic execution, or the GRU control fails.

## 5. Preregistration (declared before aggregate inspection)

* **Grid (full cross):** learning rate `{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}` × update budget `{1,000, 5,000}` × readers `{TCN, GRU}` × histories `{L4, Lfull}` × seeds `42–46` = 200 fits. Adam with the existing weight decay 1e-2, full-batch class-balanced cross-entropy; **only** the learning rate and update budget change.
* **Matched initialization:** every LR/budget cell of a (seed, reader, history) is reset to the same deterministic initialization (`aw.reader_seed(seed, reader, "op_dest", rep, top, "main")`), so LR/budget comparisons share one init.
* **Data:** Z's exact `op_dest` dataset (Y's fixed 256-example subset, canonical VAL/TEST), copied verbatim and byte-verified.
* **Fitting criterion:** `fit_bar = 0.90`. A recipe is **robust** at a history iff the five-seed mean maximum training accuracy ≥ 0.90 **and** ≥ 4/5 seeds reach 0.90.
* **Decision:** AA-INVALID if audit/invariant/control fails; else AA-A if some recipe is robust at both histories; else AA-MIX if some recipe is robust at exactly one history or reaches the mean level without the seed criterion; else AA-B. Checkpoint selection is VAL-only; the best-on-TRAIN state is diagnostic; TEST never selects LR/budget/reader/history.

## 6. Frozen Model Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, optimizer, loss. **Nothing modified.** Checkpoints are read only to define the test populations; no RE:Reason parameter acquires a gradient, and reader parameters are disjoint from the frozen model's.

## 7. Mandatory TCN Implementation Audit (non-training)

A non-training validity audit of the existing TCN implementation was run before the grid (no repair, no modification). It distinguishes an absent computational path, numerical/gradient failure, and a merely untrained model. It **passed** at both histories (diagnostic batch = 64 real subset examples).

| Check | L4 | Lfull | Meaning |
|---|---|---|---|
| `layer_alignment` | **True** (`[4,4,4,4]`) | **True** (`[10,10,10,10]`) | every causal conv + right-trim preserves the temporal length |
| `causal_alignment` | **True** | **True** | pre-head output at position *i* is invariant to inputs at positions > *i* |
| `final_step_effect` | **True** | **True** | the final logits depend on the final input timestep |
| `logits_finite` | **True** | **True** | no NaN/Inf in the forward output |
| `grad_finite` | **True** | **True** | autograd input gradient is finite |
| `dest_grad_path` | **True** (|g|sum = 8.61e-02) | **True** (|g|sum = 7.77e-02) | the current step's `x_t[3:7]` has a nontrivial gradient path to the logits (all 64 rows nonzero) |
| `dest_change_effect` | **True** | **True** | a controlled change to `x_t[3:7]` changes the output |
| **`passed`** | **True** | **True** | — |

A separate **negative control** (in `tests/test_experiment_aa.py`) confirms the audit's causal check can *fail*: a future-leaking representation (centered convolution with no trim) is flagged `causal_alignment = False`, while the real TCN is `True`. The audit therefore does not trivially pass, and it found **no** alignment, indexing, gradient-path or numerical defect. Consequence: the optimizer sweep proceeded; no AA-INVALID.

## 8. Data Identities, Split Checks, and Checkpoint Hashes

| Property | Value |
|---|---|
| Source artifact | `experiments/experiment_z_op_dest.npz` (SHA-256 `917dcb373d4e2cecbc0a3a2a5c888efed3d97a4dc155d8d0f411dbe2e8364931`) |
| Upstream Y | `experiments/experiment_y_fitting_control.npz` (SHA-256 `d22f881a973aeb6925c2a66edab09c34a5625041dde40fc9eb669501de0a6de5`) |
| Upstream W | `experiments/experiment_w_data_scale.npz` (SHA-256 `fd7b32535401f9b70f7433bb1c8ef9ea66dfa0da21960dfc9a3d600c7df7efe6`) |
| Subset | Y's/Z's exact 256 transitions/seed (1,280 pooled); selection seed `20261010` |
| Arrays | 53 arrays copied verbatim from Z; byte-identical, per-array SHA-256 recorded |
| `op_dest` train pooled | [306, 315, 300, 359] |
| `op_dest` VAL pooled | [1032, 1000, 995, 973] |
| `op_dest` TEST (E and F) | [3928, 3975, 3943, 3917] |
| Class coverage | all four classes present in train (per seed), VAL (per seed) and TEST |
| `op_dest` recoverability | train / VAL / TEST = 1.000 (exact `argmax(x_t[3:7])`) |
| Subset ∩ VAL / Subset ∩ TEST / VAL ∩ TEST | 0 / 0 / 0 |
| Checkpoints | `E_candidate`/`F_combined` seeds 42–46, SHA-256 re-verified unchanged before/after |

## 9. Optimization Protocol and Initialization

Adam (weight decay 1e-2), full-batch, class-balanced cross-entropy, evaluation every 10 updates. Only the learning rate and the update budget are varied. Each (seed, reader, history) resets to a deterministic initialization shared by all ten LR/budget cells; because the model is rebuilt under the same `torch.manual_seed`, the LR/budget comparison is init-matched (no unavoidable initialization difference). The trainer is byte-identical in ops/order to Y's `train_trajectory` at lr = 3e-3 (`tests/test_experiment_aa.py::test_train_grid_matches_y_protocol`).

## 10. Checkpoint Reporting

For every grid cell the **final** state (end of the budget), the **best-on-VAL** state (validation-only selection) and the **best-on-TRAIN** state (diagnostic only) are reported. No TEST information enters selection.

## 11. Fitting and Generalization Metrics

Training accuracy (max observed and final) on the 256-example subset; training/validation loss and accuracy at updates 200, 400, 600, 800, 1,000 and each subsequent 1,000-update milestone through 5,000; validation accuracy; and `op_dest` TEST accuracy on all-ADD and Class-A populations in the best-on-VAL state.

## 12. Full Grid Results — TCN (Subject)

Five-seed mean ± SD of the maximum training accuracy on the 256-example subset. `chance = 0.25`, `fit_bar = 0.90`, `nfit` = seeds ≥ 0.90.

| LR | budget | L4 mean ± SD | nfit | L4 final | L4 test (all-ADD) | full mean ± SD | nfit | full final | full test (all-ADD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3e-4 | 1,000 | 0.281 ± 0.080 | 0 | 0.235 | 0.276 | 0.277 ± 0.033 | 0 | 0.267 | 0.257 |
| 3e-4 | 5,000 | **0.311 ± 0.064** | 0 | 0.263 | 0.276 | 0.292 ± 0.015 | 0 | 0.263 | 0.259 |
| 1e-3 | 1,000 | 0.256 ± 0.031 | 0 | 0.235 | 0.251 | 0.277 ± 0.033 | 0 | 0.251 | 0.252 |
| 1e-3 | 5,000 | 0.286 ± 0.022 | 0 | 0.256 | 0.251 | 0.291 ± 0.016 | 0 | 0.268 | 0.254 |
| 3e-3 | 1,000 | 0.287 ± 0.021 | 0 | 0.268 | 0.251 | 0.287 ± 0.021 | 0 | 0.246 | 0.251 |
| 3e-3 | 5,000 | 0.287 ± 0.021 | 0 | 0.261 | 0.251 | 0.287 ± 0.021 | 0 | 0.248 | 0.251 |
| 1e-2 | 1,000 | 0.287 ± 0.021 | 0 | 0.247 | 0.251 | 0.287 ± 0.021 | 0 | 0.258 | 0.251 |
| 1e-2 | 5,000 | 0.287 ± 0.021 | 0 | 0.251 | 0.251 | 0.287 ± 0.021 | 0 | 0.245 | 0.251 |
| 3e-2 | 1,000 | 0.287 ± 0.021 | 0 | 0.252 | 0.251 | 0.287 ± 0.021 | 0 | 0.248 | 0.251 |
| 3e-2 | 5,000 | 0.287 ± 0.021 | 0 | 0.232 | 0.251 | 0.287 ± 0.021 | 0 | 0.235 | 0.251 |

Per-seed maximum training accuracy at the best cell (L4, lr 3e-4, budget 5,000): **[0.258, 0.312, 0.270, 0.297, 0.418]** — one seed reaches 0.418, still far below 0.90. Every cell is numerically finite (no divergence).

## 13. Full Grid Results — GRU (Positive Control)

Exactly the same grid; the GRU reaches **1.000 ± 0.000** maximum training accuracy (5/5 seeds), validation 1.000 and `op_dest` TEST accuracy 1.000 at **every** one of the ten (LR, budget) cells at both histories (L4 final 1.000, full final 1.000). The control therefore robustly fits `op_dest`, validating the labels/data/protocol/grid at every recipe — including the extreme learning rates 3e-4 and 3e-2.

## 14. Training/Validation Curves and Best-Validation Checkpoints

The TCN's training accuracy is flat near chance throughout (e.g. L4, lr 3e-3, budget 1,000: train 0.235 → 0.235 → 0.252 → 0.263 → 0.268 at updates 200–1000; validation 0.255 → 0.255 → 0.248 → 0.249 → 0.251). At lr 3e-4/5,000 the L4 train accuracy is 0.251 → 0.235 → 0.235 → 0.263 at updates 200/1000/3000/5000 — no upward trend. The GRU is at 1.000 from the first milestone. Best-on-VAL checkpoint indices are early and scattered; there is no later checkpoint worth selecting. All checkpoints are validation-selected; TEST is never used.

## 15. Learning-Rate, Budget, and Interaction Effects

* **Learning-rate effect (TCN):** essentially none over `{3e-4 … 3e-2}` — L4 means 0.256–0.311, full means 0.277–0.292; the mild maximum at lr 3e-4 is within seed noise and far below the bar. Neither a too-small nor a too-large rate is the explanation.
* **Budget effect (TCN):** increasing 1,000 → 5,000 updates changes the mean by at most +0.030 (L4 3e-4: 0.281 → 0.311) and usually by ~0.000 (e.g. lr 3e-3: 0.287 → 0.287). The TCN is budget-insensitive.
* **Interaction:** no LR × budget combination lifts the TCN; the (best) corner (small LR, large budget) is the same order of magnitude as every other cell.
* **Control:** the GRU is at ceiling for every point, so the flat TCN surface is not a grid/pipeline artifact.

## 16. E/F Controls

The `op_dest` test arrays are identical for E and F and the readers are model-independent, so E and F reader accuracies on all-ADD are **exactly identical** (`< 1e-9`; `tests/test_experiment_aa.py::test_ef_control`). F is a control and is never pooled with E.

## 17. Decision Classification

**AA-B — no recipe qualifies.** Pre-registered rule: AA-INVALID if the audit/invariant/GRU-control fails; else AA-A if some recipe is robust at both histories; else AA-MIX if some recipe is robust at exactly one history or reaches the mean level without the 4/5-seed criterion; else AA-B. Robust = five-seed mean max-train ≥ 0.90 **and** ≥ 4/5 seeds ≥ 0.90.

* **TCN robust recipes:** none at L4; none at Lfull. **TCN mean-only recipes:** none. Best TCN mean **0.311** (never ≥ 0.90; no seed ≥ 0.90 anywhere).
* **GRU control:** robust at **both** histories for all ten recipes → `control_ok = True`.
* **AA-A / AA-MIX / AA-INVALID** — not met. **AA-B** — met.

## 18. What AA Establishes

* The **implementation is valid**: the TCN's causal alignment, final-timestep path, `x_t[3:7]` computational/gradient path and controlled-change effect all hold, so the non-fitting is not an indexing/padding/gradient defect.
* The **control is valid**: the GRU robustly fits `op_dest` at every recipe, so the data, labels, target, protocol and grid are sound.
* Over the tested grid `{3e-4 … 3e-2} × {1,000, 5,000}`, the TCN **fails to fit** the directly observable `op_dest` control at every recipe — the failure is not relieved by learning-rate or update-budget changes in this range.

## 19. What AA Does Not Establish

* It does **not** prove the TCN's body or parameterization is uniquely responsible: only that this finite grid does not qualify it.
* It does **not** prove the TCN is universally incapable; a different optimizer, objective, initialization scheme or parameterization could behave differently.
* It does **not** license any latent-state, ADD-task-learnability, frozen-representation or RE:Reason architecture-sufficiency claim; `op_dest` is a direct-observation control, not the ADD target.
* It does **not** identify the mechanism of the TCN's non-fitting (optimizer-recipe vs parameterization vs expressivity); the audit rules out only alignment/indexing/gradient-path defects.
* It does **not** re-measure Z/Y; those results are reused unchanged.

## 20. Interpretation Limits and Guardrails

AA concerns the analysis reader, not RE:Reason. A failed finite optimizer grid does not establish universal architectural incapacity; fitting `op_dest` (which the GRU does) would not establish arithmetic generalization or latent-state sufficiency. The clean split — GRU fits at every recipe, TCN at none — is a statement about the TCN **instrument** under this grid. All negative results are preserved.

## 21. Final Answers

* **Does any predeclared recipe let the TCN fit `op_dest`?** No (best mean 0.311; no recipe ≥ 0.90; no seed ≥ 0.90).
* **Does the GRU control fit `op_dest` across the grid?** Yes, at every recipe (1.000).
* **Is the failure explained by the learning rate or the update budget?** No, over `{3e-4 … 3e-2} × {1,000, 5,000}`.
* **Is the TCN implementation defective (alignment/indexing/gradient)?** No — the audit passes, with a negative control confirming the causal test can fail.
* **Is the TCN qualified as an instrument for ADD-target claims?** No (unchanged from Z).

## 22. Exactly ONE Next Experiment

**Experiment AB — TCN optimization-robustness and initialization diagnostic on the `op_dest` control (measurement-only, frozen checkpoints).** AA shows the TCN is insensitive to the learning rate and update budget in `{3e-4 … 3e-2} × {1,000, 5,000}`; the remaining optimization-side hypotheses are the **optimizer type** and the **initialization scheme**. Hold Z/Y's fixed subset, the `op_dest` target, the TCN body/width/head, the splits and the seeds **exactly fixed**, and change only: (i) the optimizer family along a predeclared set — AdamW, SGD with momentum 0.9, SGD with Nesterov, RMSprop — at the fixed lr 3e-3 and 5,000 updates; and (ii) a predeclared initialization scale (PyTorch default vs a fixed-orthogonal/`xavier_uniform` variant) with Adam. Decision — **AB-A**: if some predeclared optimizer/initialization drives the TCN's `op_dest` training accuracy to ≥ 0.90 (robust, 4/5 seeds) the non-fitting is an optimization-recipe artifact and the TCN is usable with the qualified recipe; **AB-B**: if none does, the constraint lies in the TCN body/parameterization under this design, and a separately reviewed implementation-level investigation is required before the TCN is used at all. Measurement-only; no frozen weight/architecture change; no RE:Reason modification; no Phase 2; specified, not started.

## Required Figures

`docs/reports/figures/experiment_aa_*.png` (10): `lr_curve_tcn`, `lr_curve_gru`, `budget_effect`, `fit_heatmap`, `fit_heatmap_Lfull`, `per_seed`, `train_val_curves`, `reader_comparison`, `gru_control`, `decision_summary`.

## Tests

`tests/test_experiment_aa.py` — implementation-audit checks (alignment, causal path, gradient path, controlled change, and a **negative control** proving the causal check can fail); reuse of Z's `op_dest` dataset (byte identity); data/split/class/recoverability invariants; checkpoint hashes; grid-trainer protocol identity with Y and exact update counts; matched deterministic initialization; the predeclared robust-fit decision rule (AA-A/AA-MIX/AA-B/AA-INVALID) encoded and tested on synthetic grids; full grid completeness; per-seed reporting; milestones; VAL-only selection; E/F control; deterministic trajectories; frozen-model protection; malformed-artifact rejection; figure generation; historical immutability; documentation-governance compliance.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_aa.py \
    --z-npz experiments/experiment_z_op_dest.npz \
    --output experiments/experiment_aa_optimizer_grid.npz
# grid is resumable: collect raw fits, then assemble
PYTHONPATH=src python experiments/analyze_experiment_aa.py --collect-only \
    --raw experiments/experiment_aa_grid_raw.json
PYTHONPATH=src python experiments/analyze_experiment_aa.py --use-raw \
    --raw experiments/experiment_aa_grid_raw.json \
    --output experiments/experiment_aa_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_aa
PYTHONPATH=src pytest tests/test_experiment_aa.py -v
PYTHONPATH=src pytest tests/
```

## Artifacts

* `experiments/run_experiment_aa.py` — `op_dest` collector + mandatory TCN implementation audit
* `experiments/analyze_experiment_aa.py` — optimizer-grid analysis (resumable: `--collect-only` / `--use-raw`)
* `experiments/experiment_aa_optimizer_grid.npz`, `experiments/experiment_aa_optimizer_grid_meta.json`
* `experiments/experiment_aa_grid_raw.json` — raw per-seed grid fits (resumable evidence)
* `experiments/experiment_aa_results.json`
* `docs/reports/figures/experiment_aa_*.png` (10)
* `tests/test_experiment_aa.py`

## Scientific Guardrails

Measurement-only; observational with respect to RE:Reason. AA is an **optimizer-recipe diagnostic of the analysis reader**, not a RE:Reason capability experiment: a failed finite grid does not prove universal architectural incapacity, and fitting `op_dest` would not establish arithmetic generalization or latent-state sufficiency. The implementation audit establishes only the absence of alignment/indexing/gradient-path defects — not a mechanism, and not a positive statement about the TCN body or parameterization. Q–Z are **not** edited; AA is appended as new evidence. No Phase 2. Negative and inconclusive results are preserved.
