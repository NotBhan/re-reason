# RE:Reason — Experiment Z: TCN Instrument Qualification on the Directly Observable `op_dest` Target

**Date:** 2026-10-10
**Status:** Completed — Measurement only (frozen E/F checkpoints used only to define the Class-A / all-ADD test populations and reproduce baseline metrics; no model training; no optimizer on RE:Reason parameters; no architecture/encoder/recurrent-core/state/gate/candidate/commit/readout modification; no state replacement; no candidate manipulation; no feature editing; no oracle/reconstructed state as a reader feature; no branching/verifier; no counterfactual rollout; no Phase 2). The Z readers are separate analysis-only models and never enter frozen model inference. Classification: **Z-B — TCN-specific instrument concern** (E primary; F control).
**Frozen models:** `E_candidate` (primary) and `F_combined` (control), seeds 42–46, SHA-256-pinned (unchanged before/after).
**Data:** `experiments/experiment_z_op_dest.npz`, `experiments/experiment_z_op_dest_meta.json`, `experiments/experiment_z_results.json`
**Runners:** `experiments/run_experiment_z.py`, `experiments/analyze_experiment_z.py`, tests `tests/test_experiment_z.py`

---

## 1. Title / Date / Status

Experiment Z — TCN Instrument Qualification with a Directly Observable Target. Measurement-only, frozen; **Z-B — TCN-specific instrument concern**. The only thing trained is the analysis-only readers, on the directly observable four-way `op_dest` target, using Y's exact fixed subset and protocol. No intervention on the frozen model; no Phase 2. Q–Y reports are untouched; Z is appended as new evidence.

## 2. Executive Answer

On the directly observable `op_dest` control the two readers separate cleanly, and the result is **Z-B**:

* **The GRU positive control fits the control perfectly.** On the 256-example subset it reaches training accuracy **1.000 ± 0.000** at both L4 and the full prefix (by update 44), validation accuracy **1.000**, and `op_dest` TEST accuracy **1.000** (all-ADD and Class-A; held-out operand accuracy 1.000). The labels, data path, target construction and training protocol therefore support fitting this directly observable target.
* **The TCN does not fit it.** Its maximum training accuracy on the same 256 examples is **0.287 ± 0.021** at L4 and **0.287 ± 0.021** at the full prefix (final 0.268 / 0.246) — above the 0.25 four-way chance but far below the 0.90 fitting bar — and its `op_dest` validation and TEST accuracies sit at ≈0.25 (test all-ADD 0.251, Class-A 0.241–0.259). The TCN's non-fitting on `y_true` (Experiment Y) therefore **extends to a directly observable target** under the same protocol.

Interpretation: under the tested protocol the TCN's failure is **not target-specific** — it is a **TCN training/optimization/expressivity instrument concern**, not an ADD-representation or generalization statement. The TCN must not be used to draw ADD-target representation conclusions. This is **observational** and carries **no causal claim** about RE:Reason; **no intervention is justified**. A successful `op_dest` fit (which the GRU shows) would not have qualified the TCN for ADD-target claims anyway, and a failed fit does not prove the architecture is universally incapable.

## 3. Y/Z Evidence Context

* **Q/R**: the true written value is only weakly recoverable from the frozen candidate `C_t` on Class-A (Q-C); bounded capacity and size-matched hard-tail training do not help (R-D).
* **S/T**: no internal candidate activation materially beats `C_t` (S-C); the frozen observation encoder is structurally operand-blind (T-D).
* **U**: on the raw causal observation history the target is exactly identifiable (L=4 Bayes 0.998, operands 0.999; full prefix 1.000) yet bounded flat probes stay weak (best full-prefix 0.218 all-ADD, ~0.09 Class-A) → **U-D**.
* **V/W/X**: two bounded causal sequence readers (GRU hidden 85; TCN channels 52) stay weak across history, data scale and optimization budget (V-B, W-B, X-B); X found the sequence readers do not fit even the training pool.
* **Y**: on a fixed, class-balanced 256-example `y_true` subset the **GRU fits but does not transfer**, while the **TCN does not fit even the small subset** (training accuracy flat at chance) → **Y-MIX**, which named exactly this `op_dest` control as Experiment Z.

## 4. Research Question and Hypotheses

> Can the TCN fit the directly observable four-way `op_dest` target using the same input data, subset, temporal reader, optimizer protocol and history conditions as Y?

* **H-A|Z-A:** the GRU control reaches the fitting threshold and the TCN reaches ≥ 0.90 training accuracy at **both** L4 and Lfull.
* **H-B|Z-B:** the GRU reaches the threshold but the TCN is below 0.90 at **both** histories.
* **H-MIX|Z-MIX:** the TCN meets the threshold at one history but not the other (or another predeclared materially different outcome).
* **Z-INVALID:** the GRU control fails the threshold, or a class/split/recoverability invariant fails.

## 5. Preregistration (declared before aggregate inspection)

* **Fitting set:** Y's **exact** fixed 256-transition class-balanced subset per seed (copied verbatim; not regenerated, resampled, rebalanced or enlarged); Y's canonical VAL and TEST copied verbatim. The subset's `(sequence_id, timestep)` provenance digest is re-verified against Y's record.
* **Readers (exactly two, reused verbatim from U/V/W/X/Y):** causal **GRU** hidden 85 (positive control) and causal **TCN** channels 52, kernel 3, dilations [1,2,4,8], 4 layers (subject). The temporal reader body, causal padding, input features (21-dim observation + validity bit = 22) and validity-mask semantics are unchanged; the target has **4 classes**, so only the output head changes to `Linear(width, 4)`.
* **Histories:** exactly `L4` and `Lfull`.
* **Target:** `op_dest` = destination register index = `argmax(x_t[3:7])` of the current observation, generated by the existing helpers; used as a **label only** (never an input feature). The collector verifies it is exactly recoverable from the current observation (100%) in train, VAL and TEST.
* **Training protocol (Y's, unchanged):** Adam (lr 3e-3, weight-decay 1e-2), full-batch, class-balanced cross-entropy, **exactly 1,000 updates**, VAL evaluated every 10 updates, deterministic per-fit seed = `aw.reader_seed(seed, reader, "op_dest", rep, TOP, "main")`. No scheduler/curriculum/restart/tuning/early stopping.
* **States reported per (reader, history):** final (after exactly 1,000 updates), best-on-VAL (validation-only selection), best-on-TRAIN (diagnostic only; never a deployable selection). TEST is never used to select or adjust.
* **Decision (checked in order; `fit_bar = 0.90` on the maximum observed training accuracy):** Z-INVALID (invariant/control failure) → Z-A (TCN ≥ 0.90 both) → Z-B (TCN < 0.90 both) → Z-MIX (one history only).

## 6. Frozen Model Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, optimizer, loss. **Nothing modified.** Checkpoints are read only to define the test populations (`cand_ok`) and the all-ADD population; no RE:Reason parameter acquires a gradient, and reader parameters are disjoint from the frozen model's.

## 7. Readers and the Four-Class Output Head

```
GRU : x(22) -> nn.GRU(22, 85, num_layers=1, bidirectional=False) -> last step -> Linear(85, C)
TCN : x(22) -> Conv1d(22, 52, k=3, d=1) [causal] -> ReLU
              -> Conv1d(52, 52, k=3, d=2) [causal] -> ReLU
              -> Conv1d(52, 52, k=3, d=4) [causal] -> ReLU
              -> Conv1d(52, 52, k=3, d=8) [causal] -> ReLU
              -> last position -> Linear(52, C)
```

Only the output head width `C` changes with the target (4 classes for `op_dest` vs 10 for `y_true`); the temporal body is byte-identical in structure.

| Reader | Width | Params (`op_dest`, C=4) | Params (`y_true`, C=10) | Head delta |
|---|---|---:|---:|---:|
| GRU | hidden 85 | 28,139 | 28,655 | (85+1)·6 = 516 |
| TCN | channels 52 | 28,188 | 28,506 | (52+1)·6 = 318 |

The unavoidable difference is the output-head width only; no other architecture change is introduced.

## 8. Fixed Data: Reused Subset, Provenance, and `op_dest` Class Coverage

Z reuses Experiment Y's exact fixed subset and canonical VAL/TEST. Every copied array is byte-identical to the Y artifact, and the subset provenance digest matches Y's recorded value.

| Property | Value |
|---|---|
| Source artifact | `experiments/experiment_y_fitting_control.npz` (SHA-256 `d22f881a973aeb6925c2a66edab09c34a5625041dde40fc9eb669501de0a6de5`) |
| Upstream source | `experiments/experiment_w_data_scale.npz` (SHA-256 `fd7b32535401f9b70f7433bb1c8ef9ea66dfa0da21960dfc9a3d600c7df7efe6`) |
| Subset | Y's exact 256 unique transitions/seed (1,280 pooled); selection seed `20261010` |
| `op_dest` train counts / seed | [65,66,58,67] / [53,60,63,80] / [60,66,69,61] / [61,68,51,76] / [67,55,59,75] |
| `op_dest` train pooled | [306, 315, 300, 359] |
| `op_dest` VAL pooled (800/seed) | [1032, 1000, 995, 973] |
| `op_dest` TEST (E and F) | [3928, 3975, 3943, 3917] |
| All four classes present | train (per seed), VAL (per seed), TEST — verified |
| `op_dest` direct recoverability | train / VAL / TEST = 1.000 (exact `argmax(x_t[3:7])`) |
| Subset ∩ VAL / Subset ∩ TEST / VAL ∩ TEST | 0 / 0 / 0 |

## 9. The `op_dest` Target Is Directly Observable

The destination register at the current timestep is `argmax` of the current 21-dim observation's `reg_a` block `x_t[3:7]`; the collector verifies this holds for **every** row in train, VAL and TEST (recoverability 1.000). `op_dest` is therefore a pure direct-observation control — a reader that cannot fit it cannot fit a target that is readable from the current input alone, independent of history or recurrence. `op_dest` enters the pipeline as a label only; the reader input remains the raw observation + validity bit (no derived/oracle/reconstructed feature).

## 10. Optimization Protocol

Identical to Y/U/V/W/X except the target and the output-head width: Adam (lr 3e-3, weight-decay 1e-2), full-batch, class-balanced cross-entropy, best-on-VAL checkpoint, deterministic per-fit seed. Each cell runs exactly **1,000 updates**, evaluating VAL every 10 updates; training/validation accuracy and loss are recorded at updates 200 / 400 / 600 / 800 / 1,000, along with the final and maximum training accuracy and the update at which the maximum occurred. No tuning of any protocol parameter within Z. The GRU is a positive control for the labels/data/protocol.

## 11. Checkpoint Reporting

Two deployable states are reported per (reader, history): the **final** state after exactly 1,000 updates and the **best-on-VAL** state (validation accuracy only). The **best-on-TRAIN** state (the update at which the running maximum training accuracy was observed) is reported as a diagnostic only and is never claimed to be the best generalizing model. No TEST information enters selection.

## 12. Fitting Metrics

Accuracy/loss on the 256 training examples, at the predeclared milestones, and the final and maximum training accuracy with its update number.

## 13. Generalization / Test Metrics

The same fitted readers are evaluated on the untouched canonical VAL, the untouched canonical TEST, the Class-A subset of canonical TEST, and the held-out operand subset (`b ∈ {0,5}`); target accuracy and seen-versus-held-out accuracy are reported separately. For the directly observable `op_dest` target, TEST accuracy is a direct instrument read-out: a reader that genuinely fits the control should also decode it on TEST.

## 14. Primary Results: Can the Readers Fit `op_dest`?

Training accuracy on the 256-example subset, E primary, five seeds, mean ± SD. Four-way chance = 0.25.

| Reader | History | Train acc max | (update) | Train acc final | Train loss final | Fit? |
|---|---|---:|---:|---:|---:|---|
| GRU | L4 | **1.000 ± 0.000** | 44 | 1.000 | 0.042 | yes |
| GRU | full | **1.000 ± 0.000** | 44 | 1.000 | 0.038 | yes |
| TCN | L4 | 0.287 ± 0.021 | 46 | 0.268 | 1.386 | **no** |
| TCN | full | 0.287 ± 0.021 | 134 | 0.246 | 1.386 | **no** |

Per-seed maximum training accuracy: GRU 1.000 on every seed at both histories; TCN L4/full [0.262, 0.312, 0.270, 0.297, 0.293]. The GRU reaches the 0.90 bar by update 44; the TCN never approaches it (its class-balanced loss stays near ln 4 ≈ 1.386).

## 15. Training and Validation Trajectories

Mean training (subset) and validation (canonical VAL) accuracy at the predeclared milestones (`op_dest`, E).

| Reader | History | tr@200 | tr@400 | tr@600 | tr@800 | tr@1000 | val@200 | val@400 | val@600 | val@800 | val@1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | L4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU | full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TCN | L4 | 0.235 | 0.235 | 0.252 | 0.263 | 0.268 | 0.255 | 0.255 | 0.248 | 0.249 | 0.251 |
| TCN | full | 0.271 | 0.253 | 0.258 | 0.248 | 0.246 | 0.248 | 0.252 | 0.257 | 0.253 | 0.244 |

The GRU is at ceiling from the first milestone; the TCN hovers around the 0.25 chance level throughout at both histories, with no upward trend.

## 16. Test and Held-Out Accuracy

`op_dest` TEST accuracy by reported state (E, mean). The best-on-VAL state is the deployable selection.

| Reader | History | best-VAL test (all-ADD) | best-VAL test (Class-A) | best-VAL held-out | final test (all-ADD) | best-TRAIN test (all-ADD) |
|---|---|---:|---:|---:|---:|---:|
| GRU | L4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |
| GRU | full | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 |
| TCN | L4 | 0.251 | 0.241 | 0.248 | 0.249 | 0.251 |
| TCN | full | 0.251 | 0.241 | 0.248 | 0.255 | 0.251 |

The GRU generalizes the directly observable target perfectly (TEST 1.000 at both histories, all populations). The TCN stays at ≈chance (0.24–0.26) on every state and population — consistent with its inability to fit the control at all. Per-seed TCN test all-ADD = [0.259, 0.244, 0.252, 0.250, 0.253]; per-seed GRU test all-ADD = 1.000 on every seed.

## 17. Control Role of the GRU

The GRU is the positive control: under the identical data, subset, protocol and history conditions it reaches 1.000 training accuracy (by update 44) and 1.000 TEST accuracy. This establishes that the labels, data path, target construction and training pipeline support fitting this directly observable target, so the TCN's failure cannot be attributed to a broken control. Every GRU comparison is reported; no TCN-specific conclusion is drawn from a failing control.

## 18. E/F Controls

The all-ADD population is model-independent and the E/F `op_dest` test arrays are identical, so E and F reader accuracies on all-ADD are **exactly identical** (`< 1e-9`; verified in `tests/test_experiment_z.py::test_ef_control`). F is a control and is never pooled with E; the E/F pair does not change the Z-B classification.

## 19. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a control, never pooled. Every reported fraction is a mean ± SD over the five E seeds, and per-seed results are preserved. The reader set, histories, target, fitting protocol and decision bars were **predeclared**; every reader and history is reported, including negative results; no TEST-based selection of reader/history/checkpoint. The best-on-TRAIN state is a diagnostic only.

## 20. Leakage and Validity

Enforced and tested: (i) the subset is Y's exact fixed subset, sequence-disjoint from VAL/TEST (overlap 0); (ii) VAL and TEST are byte-identical to Y's; (iii) no TEST-derived selection of checkpoint/history/reader; (iv) `op_dest` is used as a label only and is verified directly recoverable from the current observation; (v) no future observation enters the reader input; (vi) exactly 1,000 updates are executed (no reweighting/repetition); (vii) all four `op_dest` classes are present in train, VAL and TEST; (viii) frozen E/F parameters are never mutated and acquire no gradients; (ix) reader parameters are never attached to the frozen model; (x) checkpoint SHA-256 unchanged before/after.

## 21. Deterministic Reproduction

Z reuses Y's collector output and Y's exact trainer; each fit is deterministic per fit (fixed seed) and executed single-threaded per worker. An independent analyzer re-run reproduced identical readers, trajectory and decision. Checkpoint SHA-256 verified before/after. The collector verifies byte identity of every copied array and the subset provenance digest.

## 22. Causal Status

**Observational with respect to RE:Reason.** Z manipulates only the analysis reader's fitting target, so the comparison is causal *within the reader experiment* ("the GRU can fit `op_dest`; the TCN cannot"). It is **not** a causal statement about RE:Reason. The four levels remain distinct: **interface availability** (U) ≠ **reader learnability** (V/W/X/Y/Z) ≠ **frozen-state representation** (T/S) ≠ **architecture sufficiency**. No intervention is performed or licensed.

## 23. Decision Classification

**Z-B — TCN-specific instrument concern** (E primary; F control).

Pre-registered bar: `fit_bar = 0.90` on the maximum observed training accuracy.

* **GRU control:** train-max 1.000 at L4 and full → reaches the threshold → **control passes**.
* **TCN:** train-max 0.287 (L4) and 0.287 (full), both < 0.90 → **below the bar at both histories**.
* **Z-INVALID** (invariant/control failure) — not met (all class/split/recoverability invariants hold; GRU control passes).
* **Z-A** (GRU passes and TCN ≥ 0.90 at both histories) — **not met**.
* **Z-B** (GRU passes and TCN < 0.90 at both histories) — **met**.
* **Z-MIX** (TCN meets the threshold at one history only) — not met.

Continuous quantities: TCN train-max 0.287 ± 0.021 (both histories); TCN best-VAL `op_dest` TEST 0.251 (all-ADD); GRU train-max 1.000 and TEST 1.000.

## 24. What Z Establishes

* **The control is valid.** The GRU fits `op_dest` to 1.000 training and TEST accuracy under the fixed protocol, so the labels, data path, target construction and training pipeline support fitting a directly observable target.
* **The TCN's non-fitting is not target-specific under the tested protocol.** The TCN stays below the 0.90 bar at both histories on a target that is readable from the current observation alone (train-max 0.287; TEST ≈chance).
* **Consequently the TCN cannot be used as an instrument for ADD-target representation conclusions** until its fitting is qualified; its behavior on `y_true` (Y) is confounded by this instrument concern.
* The Z pipeline is internally valid: subset/provenance identity, byte-identical VAL/TEST, 100% `op_dest` recoverability, four-class coverage, split disjointness and the GRU control all verified.

## 25. What Z Does Not Establish

* It does **not** prove the TCN architecture is universally incapable of learning a directly observable target; the result is specific to the tested protocol (one optimizer, one LR/WD, full-batch, 1,000 updates, this parameterization).
* It does **not** identify the cause of the TCN's non-fitting (optimization recipe vs parameterization vs a gradient-flow/receptive-field implementation issue) — that is the next experiment.
* It does **not** say anything ADD-specific: it is an instrument qualification, not a capability experiment.
* It does **not** license any conclusion about RE:Reason's frozen representation, information content, or architecture sufficiency; a successful `op_dest` fit would not have validated the TCN for ADD-target claims, and this failed fit does not prove architectural incapacity.
* It does **not** re-measure Y/W/X; those results are reused unchanged.

## 26. Final Answers

* **Does the GRU control fit `op_dest`?** Yes (train-max 1.000 at L4 and full; TEST 1.000).
* **Does the TCN fit `op_dest`?** No (train-max 0.287 at both L4 and full; TEST ≈0.25).
* **Is the TCN's non-fitting target-specific?** No — it extends to the directly observable control.
* **Can the TCN be used for ADD-target representation claims?** No; the instrument is unqualified.
* **Do E and F agree?** Yes (all-ADD identical to `< 1e-9`).
* **Was the protocol valid?** Yes — the subset/provenance, VAL/TEST identity, `op_dest` recoverability, class coverage, disjointness and the GRU control are all verified.

## 27. Exactly ONE Next Experiment

**Experiment AA — TCN optimization-recipe diagnostic on the `op_dest` control (measurement-only, frozen checkpoints).** Z-B leaves one question: is the TCN's failure an **optimization-recipe** artifact or a **parameterization/expressivity** limitation? Hold the data (Y's fixed 256-subset), the target (`op_dest`), the reader architecture/body, the splits and the deterministic seeds **exactly fixed**, and change **only the optimizer recipe** for the analysis readers along a **predeclared finite grid** — learning rate `{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}` × update budget `{1,000, 5,000}` (the GRU control run under the same grid). Decision — **AA-A**: if some predeclared recipe drives the TCN's `op_dest` training accuracy to ≥ 0.90, the non-fitting is an **optimization-recipe** artifact and the TCN is usable as an instrument with the qualified recipe; **AA-B**: if no predeclared recipe fits the TCN, the constraint is the **TCN body/parameterization** under this design, and the reader family must be re-examined as an instrument (no ADD conclusion either way). Measurement-only; no frozen weight/architecture change; no RE:Reason modification; no Phase 2; specified, not started.

## Required Figures

`docs/reports/figures/experiment_z_*.png` (8): `training_fit`, `training_trajectory`, `reader_comparison`, `l4_vs_full`, `per_seed`, `test_accuracy`, `op_dest_control`, `decision_summary`.

## Tests

`tests/test_experiment_z.py` (36 tests) — reuse of Y's exact subset and provenance; byte-identical VAL/TEST; four-way `op_dest` class coverage in every population; direct recoverability of `op_dest` from the current observation; split/leakage invariants; reader architecture reuse and four-class output-head parameter counts; exactly 1,000 updates; milestone recording; maximum-training-accuracy calculation; final-checkpoint integrity; validation-only checkpoint selection; no TEST-based selection; reader-seed convention; deterministic trajectories; GRU trainer-protocol reuse; E/F control; decision validity and rule consistency; frozen-model protection (no gradients / no mutation); reader/model separation; checkpoint SHA verification; malformed-artifact rejection; figure generation; historical-report immutability; documentation-governance compliance.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_z.py \
    --y-npz experiments/experiment_y_fitting_control.npz \
    --output experiments/experiment_z_op_dest.npz
PYTHONPATH=src python experiments/analyze_experiment_z.py \
    --npz experiments/experiment_z_op_dest.npz \
    --output experiments/experiment_z_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_z
PYTHONPATH=src pytest tests/test_experiment_z.py -v
PYTHONPATH=src pytest tests/
```

## Artifacts

* `experiments/run_experiment_z.py` — `op_dest` collector (copies Y's fixed subset/VAL/TEST verbatim; verifies provenance, class coverage and direct recoverability)
* `experiments/analyze_experiment_z.py` — instrument-qualification analysis (GRU control + TCN on `op_dest`)
* `experiments/experiment_z_op_dest.npz`, `experiments/experiment_z_op_dest_meta.json`
* `experiments/experiment_z_results.json`
* `docs/reports/figures/experiment_z_*.png` (8)
* `tests/test_experiment_z.py`

## Scientific Guardrails

Measurement-only; observational with respect to RE:Reason. Z is an **instrument qualification**, not a capability experiment: a successful `op_dest` fit would only show the TCN can learn a direct-observation control under the tested setup, and would **not** validate it for ADD-target representation claims; this failed fit does **not** prove the architecture is universally incapable, that the arithmetic information is absent from RE:Reason, that it is mathematically unlearnable, or any causal mechanism. The four levels stay separate: **interface availability** ≠ **reader learnability** ≠ **frozen-state representation** ≠ **architecture sufficiency**. Q–Y are **not** edited; Z is appended as new evidence. No Phase 2. Negative and inconclusive results are preserved.
