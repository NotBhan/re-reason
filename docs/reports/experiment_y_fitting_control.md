# RE:Reason — Experiment Y: Bounded-Reader Fitting Control

**Date:** 2026-10-10
**Status:** Completed — Measurement only (frozen E/F checkpoints used only to define the Class-A population and reproduce baseline metrics; no model training; no optimizer on RE:Reason parameters; no architecture/encoder/recurrent-core/state/gate/candidate/commit/readout modification; no state replacement; no candidate manipulation; no feature editing; no oracle/reconstructed state as a reader feature; no branching/verifier; no counterfactual rollout; no Phase 2). The Y readers are separate analysis-only models and never enter frozen model inference. Classification: **Y-MIX — reader-specific fitting result** (E primary; F control).
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, SHA-256-pinned (unchanged before/after).
**Data:** `experiments/experiment_y_fitting_control.npz`, `experiments/experiment_y_fitting_control_meta.json`, `experiments/experiment_y_results.json`
**Runners:** `experiments/run_experiment_y.py`, `experiments/analyze_experiment_y.py`, tests `tests/test_experiment_y.py`

---

## 1. Title / Date / Status

Experiment Y — Bounded-Reader Fitting Control. Measurement-only, frozen; **Y-MIX — reader-specific fitting result**. Exactly one factor is manipulated for the fitting diagnostic: the fitting subset is a deterministic, fixed, class-balanced small subset of **256 unique ordinary-ADD transitions per E seed** (drawn from W's existing 10,000-transition pool), versus W/X's full-pool conditions. No intervention on the frozen model; no Phase 2. Q–X reports are untouched; Y is appended as new evidence.

## 2. Executive Answer

The two bounded sequence readers behave **materially differently** on the small fixed subset, so Y is **Y-MIX**, not a clean single branch:

* **The GRU can fit the small subset but does not generalize.** At L4 it drives training accuracy on the 256 examples to **0.999 ± 0.002** (per-seed 1.000/1.000/1.000/1.000/0.996; reached by update ≈ 492) and at the full prefix to **0.992 ± 0.005** (by update ≈ 924; final 0.912). Yet its Class-A target accuracy on the untouched TEST is **0.096–0.105** and its held-out-operand Class-A accuracy is **0.092–0.111** — at/below the 0.10 chance, essentially unchanged from the full-pool GRU (W 200-update 0.116, X 1,000-update 0.110). Fitting is *possible* for this reader; it simply does not transfer.
* **The TCN does not fit the small subset at all.** Its training accuracy on the same 256 examples stays at chance — **0.102 ± 0.000** at both L4 and the full prefix (final 0.099), i.e. it never learns the tiny sample, exactly as it never learned the 10,000-transition pool in X (X training accuracy 0.07–0.15). The TCN's held-out Class-A accuracy is the highest of the three readers (0.212 / 0.215) but this is a validation/prior artifact of a non-fitting model, not learned target information; its Class-A TEST accuracy is 0.110 / 0.111.
* **The flat MLP-128 control fits perfectly and also fails to generalize.** It reaches training accuracy **1.000** at L4 and full prefix (by update 68 / 38) while its held-out Class-A accuracy is **0.076 / 0.123** and its Class-A TEST accuracy is 0.092 / 0.102 — the same fit-without-transfer signature as the GRU.

Interpretation: Y separates the two possibilities X could not. For the **GRU**, local fitting succeeds and generalization fails, so the GRU's large-pool failure is **not** an inability to fit a small sample; it reinforces a generalization / task-structure / representation reading. For the **TCN**, local fitting itself fails, so the TCN cannot be used to make ADD-specific representation claims — the bottleneck is the **reader training/optimization instrument**, not the task representation. The result is **observational** and carries **no causal claim** about RE:Reason; **no intervention is justified**. A successful small-set fit does **not** demonstrate that the reader learned a generally valid ADD rule.

## 3. Q–X Evidence Context

* **Q/R**: the true written value is only weakly recoverable from the frozen candidate `C_t` on Class-A (Q-C); bounded capacity and size-matched hard-tail training do not help (R-D).
* **S/T**: no internal candidate activation materially beats `C_t` (S-C); the frozen observation encoder is structurally operand-blind (T-D).
* **U**: on the raw causal observation history the target is exactly identifiable (L=4 Bayes 0.998, operands 0.999; full prefix 1.000) yet bounded flat probes stay weak (best full-prefix 0.218 all-ADD, ~0.09 Class-A) → **U-D**.
* **V**: two parameter-bounded causal **sequence** readers (GRU hidden 85; TCN channels 52) are also weak (best Class-A 0.112; held-out ≤ 0.25; Bayes − reader 0.887) → **V-B**.
* **W**: at a fixed 200 full-batch updates, 1,500 / 4,000 / 10,000 fresh sequence-disjoint pools did not lift Class-A accuracy, and training accuracy stayed low (0.13–0.23) → **W-B**.
* **X**: raising the budget to 1,000 updates on W's fixed pool did not lift Class-A accuracy (best 0.111) or the Bayes − reader gap (0.888), and the **sequence readers did not fit even the training pool** (training accuracy 0.07–0.15, near/below the 0.10 chance) → **X-B**. X §26 specified exactly this fitting control as Experiment Y.

## 4. Research Question and Hypotheses

> Can the existing bounded readers fit the actual register-tracking target on a small, fixed training subset under the same fixed optimizer protocol?

* **H-A|Y-A (small-set fitting succeeds):** some GRU/TCN at L4/full reaches training accuracy ≥ 0.90 on the 256-subset while its held-out Class-A target accuracy stays ≤ 0.25.
* **H-B|Y-B (small-set fitting fails):** neither the GRU nor the TCN reaches 0.90 training accuracy on the small subset at either history condition after 1,000 updates.
* **Y-MIX (reader-specific fitting):** the GRU and TCN produce materially different fitting outcomes that change the interpretation.
* **Otherwise (inconclusive):** any case between the above is reported explicitly rather than re-categorised.

## 5. Preregistration (declared before aggregate inspection)

* **Fitting set (fixed):** 256 unique ordinary-ADD transitions per E seed, drawn from W's existing top-scale pool (10,000/seed). Stratified by `y_true` over its 10 classes as evenly as possible (base 25 + 1 to the first 6 classes = **6 × 26 + 4 × 25 = 256**). Held-out operand pairs `(a,b)` with `b ∈ {0,5}` are excluded. No duplication. Selection is deterministic (fixed seed `20261010`), performed before any fitting, and never depends on reader performance. If a seed cannot supply 256 eligible unique examples with the requested per-class coverage, the collector raises an explicit validation error (it never silently shrinks the subset). The **same subset** is used for all three readers and both history conditions within a seed.
* **Readers (exactly three, reused unchanged from U/V/W/X):** U's flat **MLP-128** on the flattened history window; V's causal **GRU** (hidden 85); V's causal **TCN** (channels 52, kernel 3, dilations [1,2,4,8], 4 layers). No architecture/width/parameter-count/input/head change.
* **History conditions:** exactly two, `L4` and `Lfull`; per-timestep input = raw 21-dim observation + 1 validity bit = 22 features; left-padded causal window with validity mask; no future observation.
* **Target:** `y_true` = written register value = `(operand_a + operand_b) mod 10` (10 classes) generated by the existing helpers. No new target types.
* **Training protocol (X's, unchanged):** Adam (lr 3e-3, weight-decay 1e-2), full-batch, class-balanced cross-entropy, **exactly 1,000 optimizer updates**, deterministic per-fit seed = W/X's `reader_seed` at the top scale, VAL evaluated every 10 updates. No scheduler, curriculum, restart, or tuning.
* **Checkpoint reporting:** the **final** state (after exactly 1,000 updates), the **best-on-VAL** state (validation-only selection), and the **best-on-TRAIN** state (a diagnostic statistic; never a deployable selection rule). TEST is never used to select a checkpoint or adjust the experiment.
* **Decision (checked in order; bars are declared conventions):** Y-MIX first (materially different GRU/TCN fitting outcomes), else Y-A (some sequence reader ≥ 0.90 train **and** held-out Class-A ≤ 0.25), else Y-B (neither sequence reader ≥ 0.90 train at either history), else Y-inconclusive. Bars: `fit_bar = 0.90`, `heldout_weak = 0.25`, `heldout_material = 0.30`, `test_weak = 0.25`.
* **Controls:** MLP-128 is reported as a control; the headline decision focuses on the two sequence readers. W's 200-update and X's 1,000-update full-pool results are reused **unchanged** for comparison; no new full-pool sweep is run. `op_dest` is cited from prior experiments as the directly-observable-target control, not re-run.

## 6. Frozen Model Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, optimizer, loss. **Nothing modified.** The frozen E/F checkpoints define the Class-A test population (`cand_ok`) and reproduce baseline metrics only. All frozen-model execution is read-only (arrays copied from W); checkpoint SHA-256 verified before/after. The Y readers are separate analysis-only `nn.Module`s; they are never attached to the frozen model, no RE:Reason parameter acquires a gradient, and reader parameters are disjoint from the frozen model's.

## 7. Reader Definitions (reused verbatim from U/V)

```
MLP-128   : flatten(H_L) ⊕ mask  -> Linear(·,128) -> tanh -> Linear(128,10)                 [U's flat reader]
GRU       : x(22) -> nn.GRU(22, 85, num_layers=1, bidirectional=False) -> last step -> Linear(85, 10)   [V's causal GRU]
TCN       : x(22) -> Conv1d(22, 52, k=3, d=1)   [causal] -> ReLU
                  -> Conv1d(52, 52, k=3, d=2)   [causal] -> ReLU
                  -> Conv1d(52, 52, k=3, d=4)   [causal] -> ReLU
                  -> Conv1d(52, 52, k=3, d=8)   [causal] -> ReLU
                  -> last position -> Linear(52, 10)                                          [V's causal TCN]
```

Y builds every reader through `ax.build_model` → V's `build_reader`/`GRUReader`/`TCNReader` and U's MLP-128 definition — no reconstruction. The GRU is unidirectional and the TCN genuinely causal; prediction at `t` uses only `x_0..x_t`.

| Reader | Width | Params (L4 head, input 88) | Params (full head, input 220) | Ratio to 29,578 |
|---|---|---|---|---|
| MLP-128 | 128 | 12,682 | 29,578 | 1.000 (full) |
| GRU | hidden 85 | 28,655 | 28,655 | 0.969 |
| TCN | channels 52 | 28,506 | 28,506 | 0.964 |

The sequence-reader widths are L-independent (22 input features at every history length); the flat reader's input dimension scales with the window (22·L). Parameter counts are byte-identical across the two history conditions for the sequence readers.

## 8. Fixed Data: the 256-Example Fitting Subset

Y reuses **W's existing fresh top-scale pool** (10,000 unique ordinary-ADD transitions per E seed), W's canonical ordinary VAL, and W's untouched canonical TEST (E and F). Nothing is regenerated; the collector copies those arrays verbatim and records per-array SHA-256 plus the source-artifact SHA-256. The 256-row subset is selected from the pool and its exact `(sequence_id, timestep)` provenance is recorded.

| Property | Value |
|---|---|
| Source artifact | `experiments/experiment_w_data_scale.npz` (SHA-256 recorded in the Y meta) |
| Subset size | 256 unique transitions/seed (1,280 pooled) |
| Selection seed | 20261010 (fixed, declared before fitting) |
| Per-class target | 6 × 26 + 4 × 25 = 256 by `y_true` |
| Per-class actual (all seeds) | [26, 26, 26, 26, 26, 26, 25, 25, 25, 25] |
| Eligible rows / seed (held-out pairs excluded) | 4,974 / 5,095 / 5,007 / 4,924 / 4,932 |
| Unique sequences in subset / seed | 249 / 245 / 249 / 250 / 251 |
| Duplicate `(sequence_id, timestep)` | 0 |
| Held-out operand pairs in subset | 0 |
| Pool ∩ VAL / Pool ∩ TEST / VAL ∩ TEST | 0 / 0 / 0 |
| Subset ∩ VAL / Subset ∩ TEST | 0 / 0 |
| Canonical VAL | 4,000 pooled (800/seed) — identical to W |
| Canonical TEST | 15,763 pooled (E and F) — identical to W |
| Class-A counts (E, per seed) | 448 / 330 / 363 / 327 / 337 |

## 9. Optimization Protocol

Identical to U/V/W/X except the fitting-set size: Adam (lr 3e-3, weight-decay 1e-2), full-batch, class-balanced cross-entropy, best-on-VAL checkpoint, deterministic per-fit seed. Each cell runs exactly **1,000 updates**, evaluating VAL (and the training subset) every 10 updates; the training/validation trajectory is recorded at the predeclared milestones 200 / 400 / 600 / 800 / 1,000. The number of optimizer updates actually executed is exactly 1,000 per cell (verified in `tests/test_experiment_y.py`); no example is repeated or reweighted.

## 10. Checkpoint Reporting

Two deployable states are reported per (reader, history): the **final** state after exactly 1,000 updates and the **best-on-VAL** state (validation accuracy only). The **best-on-TRAIN** state (the update at which the running maximum training accuracy was observed) is additionally reported as a diagnostic; it is **not** claimed to be the best generalizing model and is never used to select a deployable checkpoint. No TEST information enters selection.

## 11. Primary Fitting Metric

For every reader, seed and history length, Y records the accuracy on the 256 training examples, the training loss, the accuracy and loss at each predeclared milestone, the final accuracy after 1,000 updates, and the maximum training accuracy observed during the trajectory together with its update number.

## 12. Generalization Metrics

The same fitted readers are evaluated on the untouched canonical VAL, the untouched canonical TEST, the Class-A subset of canonical TEST, and the held-out operand subset of canonical TEST defined by `(a,b)` with `b ∈ {0,5}`. Target accuracy and seen-versus-held-out accuracy are reported separately. The small-set control is deliberately not expected to match the full-pool baseline; the purpose is to establish whether local fitting is possible and whether it transfers.

## 13. Primary Results: Can the Readers Fit the 256-Example Subset?

Training accuracy (`y_true`) on the 256-example subset, E primary, five seeds, mean ± SD. `chance = 0.10`.

| Reader | History | Train acc max | (update) | Train acc final | Train loss final | Fit? |
|---|---|---:|---:|---:|---:|---|
| MLP-128 | L4 | **1.000 ± 0.000** | 68 | 1.000 | 0.160 | yes |
| MLP-128 | full | **1.000 ± 0.000** | 38 | 1.000 | 0.078 | yes |
| GRU | L4 | **0.999 ± 0.002** | 492 | 0.999 | 0.247 | yes |
| GRU | full | **0.992 ± 0.005** | 924 | 0.912 | 0.402 | yes |
| TCN | L4 | 0.102 ± 0.000 | 12 | 0.099 | 2.303 | **no** |
| TCN | full | 0.102 ± 0.000 | 46 | 0.099 | 2.303 | **no** |

Per-seed maximum training accuracy: MLP-128 L4/full 1.000 (all seeds); GRU L4 1.000 / 1.000 / 1.000 / 1.000 / 0.996; GRU full 0.992 / 0.992 / 0.988 / 1.000 / 0.988; TCN 0.102 on every seed at both histories.

The GRU and the flat MLP-128 fit the small sample cleanly; the TCN does not fit it at all (it never leaves the 0.10 chance level and its class-balanced training loss stays at ln 10 ≈ 2.302).

## 14. Training and Validation Trajectories

Mean training (subset) and validation (canonical VAL) accuracy at the predeclared milestones (`y_true`, E).

| Reader | History | tr@200 | tr@400 | tr@600 | tr@800 | tr@1000 | val@200 | val@400 | val@600 | val@800 | val@1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP-128 | L4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.116 | 0.118 | 0.117 | 0.118 | 0.122 |
| GRU | L4 | 0.553 | 0.984 | 0.998 | 0.999 | 0.999 | 0.109 | 0.112 | 0.114 | 0.117 | 0.117 |
| GRU | full | 0.335 | 0.579 | 0.818 | 0.896 | 0.912 | 0.112 | 0.116 | 0.128 | 0.122 | 0.125 |
| TCN | L4 | 0.100 | 0.102 | 0.100 | 0.098 | 0.099 | 0.079 | 0.070 | 0.068 | 0.113 | 0.135 |

The GRU's training curve rises monotonically to ≈1.0 by update ≈ 400–600 while its validation accuracy is flat at 0.11–0.12 throughout — the classic fit-without-generalization signature. The TCN's training curve is flat at chance and its validation curve is noisy (0.068–0.135) with no trend. The best-on-VAL checkpoint indices are early and scattered for every reader (MLP-128 436/190; GRU 250/378; TCN 302/86), i.e. there is no late-checkpoint generalization to select. For the GRU, the best-on-VAL state is an **under-fit** state (L4: train 0.433, val 0.135), whereas the best-on-TRAIN state (train 0.999) is the fitting diagnostic.

## 15. Test Generalization

Class-A TEST accuracy (`y_true`), E primary, five seeds, mean. Both the best-on-VAL state and the best-on-TRAIN (fit-diagnostic) state are reported; the final state is also recorded.

| Reader | History | best-VAL train | best-VAL Class-A test | best-VAL all-ADD | best-TRAIN train | best-TRAIN Class-A test | final Class-A test |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP-128 | L4 | 0.964 | 0.092 | 0.122 | 1.000 | 0.091 | 0.093 |
| MLP-128 | full | 0.998 | 0.102 | 0.129 | 1.000 | 0.102 | 0.107 |
| GRU | L4 | 0.433 | 0.105 | 0.125 | 0.999 | 0.103 | 0.096 |
| GRU | full | 0.690 | 0.093 | 0.130 | 0.992 | 0.096 | 0.098 |
| TCN | L4 | 0.102 | 0.110 | 0.290 | 0.102 | 0.112 | 0.124 |
| TCN | full | 0.101 | 0.111 | 0.247 | 0.102 | 0.087 | 0.098 |

Every reader stays at/below ~0.12 Class-A TEST accuracy; no reader approaches the 0.50 bar. The GRU's ability to fit the 256 examples transfers to **no** test improvement (its Class-A test accuracy is 0.093–0.105, essentially the same as X's full-pool 0.110). The TCN's nominally higher all-ADD accuracy (0.290 at L4, best-VAL) reflects that its best-on-VAL state is a near-prior classifier, not a fitted rule — its Class-A test accuracy is 0.110.

## 16. Held-Out Operand Generalization

Held-out operand pairs (`b ∈ {0,5}`; 20 pairs) are excluded from the fitting subset; the same reader is evaluated separately on the seen and held-out Class-A TEST rows. E, mean; best-on-VAL state.

| Reader | History | seen | held-out | final held-out | best-TRAIN held-out |
|---|---|---:|---:|---:|---:|
| MLP-128 | L4 | 0.096 | 0.076 | 0.083 | 0.082 |
| MLP-128 | full | 0.095 | 0.123 | 0.121 | 0.119 |
| GRU | L4 | 0.106 | 0.100 | 0.097 | 0.092 |
| GRU | full | 0.093 | 0.099 | 0.091 | 0.111 |
| TCN | L4 | 0.084 | 0.212 | 0.123 | 0.108 |
| TCN | full | 0.085 | 0.215 | 0.142 | 0.094 |

Every held-out value is ≤ 0.25 (bar 0.30); the material bar (0.30) is never reached. For the GRU — the reader that *does* fit — both seen and held-out accuracy are at chance (≈0.09–0.11), so the fit carries **no** evidence of learned operand→target generalization. The TCN's held-out 0.21 (per-seed 0.216 / 0.266 / 0.188 / 0.264 / 0.129) is an artifact of a non-fitting model on a small masked subset (n = 20 pairs) and does not indicate learned target information; its seen accuracy is *below* chance (0.084).

## 17. E/F Controls

The all-ADD population is model-independent (the raw observations and targets are identical for E and F); E and F reader accuracies on all-ADD are **exactly identical** (`< 1e-9`; verified in `tests/test_experiment_y.py::test_ef_control`). Class-A differs only through the frozen candidate mask (`cand_ok`). F is a control and is never pooled with E; the E/F pair does not change the Y-MIX classification.

## 18. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a control, never pooled. Every reported fraction is a mean ± SD over the five E seeds; per-seed results are preserved. The reader set, history conditions, target scope, fitting protocol and decision bars were **predeclared**; every reader and history is reported, including negative results; no TEST-based selection of reader, history or checkpoint. The best-on-VAL state is selected purely on canonical VAL; the best-on-TRAIN state is reported as a diagnostic only.

## 19. Leakage and Validity

Enforced and tested: (i) the fitting subset is drawn from W's pool and is sequence-disjoint from VAL/TEST (overlap 0); (ii) VAL and TEST are byte-identical to W's; (iii) no TEST-derived selection of checkpoint/history/reader; (iv) held-out operand pairs `{(a,b): b ∈ {0,5}}` never enter the fitting subset; (v) no duplicate `(sequence_id, timestep)` rows; (vi) the run executes exactly 1,000 updates (no example reweighting/repetition); (vii) the reader architecture and parameter count are invariant across the two history conditions for the sequence readers; (viii) no future observation enters the reader input; (ix) frozen E/F parameters are never mutated and acquire no gradients; (x) reader parameters are never attached to the frozen model; (xi) checkpoint SHA-256 unchanged before/after; (xii) the subset is fixed before any fitting and does not depend on reader performance.

## 20. Deterministic Reproduction

The collector copies W's arrays and verifies byte identity (per-array SHA-256 recomputed and matched, plus the source-artifact SHA-256), and rebuilds the subset deterministically (its provenance digest is recomputed in the tests). Each fit is deterministic per fit (fixed seed) and executed single-threaded per worker; the training loop is byte-identical in ops/order to X's `train_trajectory`, so the best-on-VAL state reproduces X's bit-for-bit (`tests/test_experiment_y.py::test_trainer_matches_x_protocol`). Checkpoint SHA-256 verified before/after. The full analyzer run took ≈ 81 s on this 14 GB / 16-core host.

## 21. Causal Status

**Observational with respect to RE:Reason.** Y deliberately manipulates the **analysis-only reader's fitting-set size**, so the comparison is causal *within the reader experiment*: "fitting the 256-example subset did / did not transfer, and the sequence readers did / did not fit it." It is **not** a causal statement about RE:Reason. The four levels remain distinct: **interface availability** (U) ≠ **reader learnability** (V, W, X, Y) ≠ **frozen-state representation** (T/S) ≠ **architecture sufficiency**. No intervention is performed or licensed.

## 22. Decision Classification

**Y-MIX — reader-specific fitting result** (E primary; F control).

Pre-registered bars (declared conventions): `fit_bar = 0.90`, `heldout_weak = 0.25`, `heldout_material = 0.30`, `test_weak = 0.25`.

* **Fit outcome by pair:** `GRU|L4` train-max 0.999 (fits), held-out-Class-A max over states 0.100 (≤ 0.25); `GRU|full` train-max 0.992 (fits), held-out 0.111 (≤ 0.25); `TCN|L4` train-max 0.102 (does **not** fit); `TCN|full` train-max 0.102 (does **not** fit).
* **Y-MIX condition** (the GRU and TCN produce materially different fitting outcomes — one family ≥ 0.90, the other not) — **met** (`family_fits = {GRU: true, TCN: false}`).
* **Y-A condition** (some sequence reader ≥ 0.90 train **and** held-out ≤ 0.25) — **met for the GRU** (train 0.999; held-out 0.100) but is not the headline because of the TCN divergence.
* **Y-B condition** (neither sequence reader ≥ 0.90 train) — **not met** (the GRU fits).

Continuous quantities: best sequence train-max **0.999** (GRU L4); smallest sequence max-held-out **0.100** (GRU L4); GRU L4 Class-A test 0.096–0.105; TCN train-max 0.102 at both histories.

## 23. What Y Establishes

* **Fitting is possible for the GRU on a small sample.** The GRU drives training accuracy on 256 unique examples to 0.999 (L4) / 0.992 (full) within the fixed protocol. So the GRU/MLP-128 family's large-pool non-fitting (X) is **not** a hard inability of the optimizer/architecture to fit *any* sample.
* **Fitting does not transfer.** The GRU's fit leaves Class-A TEST accuracy at 0.093–0.105 and held-out-operand accuracy at 0.092–0.111 — at/below chance and unchanged from the full-pool result. This points at **generalization / task structure / representation**, not at local optimizability, for the GRU.
* **The TCN fails at fitting itself.** Training accuracy stays at chance (0.102) on the small subset at both histories, with a flat class-balanced loss. The TCN therefore cannot support ADD-specific representation claims under this protocol; its bottleneck is the **reader training/optimization instrument** (a TCN-specific issue), and the reader family must be re-examined as an instrument.
* **The control (MLP-128) fits perfectly and does not generalize** (train 1.000; held-out 0.076/0.123), reproducing the same fit-without-transfer signature.
* The Y pipeline is internally valid: the subset is fixed and balanced, byte-identical VAL/TEST, disjoint, and the best-on-VAL state reproduces X's protocol bit-for-bit.

## 24. What Y Does Not Establish

* It does **not** show the GRU learned a generally valid ADD rule; a ≥ 0.90 fit on 256 examples that does not transfer is memorization of a small sample, not rule learning.
* It does **not** prove the information is mathematically unlearnable by any bounded reader, nor that the frozen RE:Reason model can or cannot access it; **reader fitting ≠ frozen representation ≠ architecture sufficiency**.
* It does **not** prove the TCN is architecturally incapable: the TCN non-fitting is measured under the tested fixed protocol (one optimizer, one LR/WD, full-batch), and a different optimizer/objective/parameterization could behave differently.
* It does **not** establish that any particular intervention will help; the fitting-set manipulation is causal only within the analysis reader experiment.
* It does **not** claim the small-sample fit makes the large-pool failure a pure "generalization problem" for the *family* — that holds for the GRU and MLP-128, while the TCN is an instrument failure.
* It does **not** re-measure W/X's full-pool results; those are reused unchanged as external baselines.

## 25. Comparison with the Unchanged W/X Full-Pool Baselines

W's 200-update and X's 1,000-update full-pool results (E, Class-A `y_true`, 5-seed mean) are reused **unchanged**:

| Reader | History | W 200-upd Class-A | W train acc | X 1,000-upd Class-A | X train acc | Y 256-set train-max | Y held-out Class-A |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP-128 | L4 | 0.070 | 0.232 | 0.068 | 0.256 | 1.000 | 0.076 |
| GRU | L4 | 0.116 | 0.183 | 0.110 | 0.071 | 0.999 | 0.100 |
| GRU | full | 0.106 | 0.193 | 0.102 | 0.102 | 0.992 | 0.111 |
| TCN | L4 | 0.114 | 0.175 | 0.110 | 0.149 | 0.102 | 0.212 |
| TCN | full | 0.111 | 0.249 | 0.111 | 0.116 | 0.102 | 0.215 |

The contrast is the headline: on the full 10,000-transition pool the GRU/TCN training accuracy is 0.07–0.25 (X/W); shrinking the fitting set to 256 lets the **GRU** reach 0.99+ training accuracy while its test accuracy is unchanged (0.10–0.11), whereas the **TCN** stays at chance even on 256 examples. `op_dest` (destination register, directly observable at `t`) is the prior directly-observable control that the sequence readers *can* learn (V: GRU `op_dest` 1.000 at the full prefix); it is cited, not re-run in Y.

## 26. Final Answers

* **Can the readers fit the actual target on a small fixed subset?** Reader-specific. The **GRU** can (train-max 0.999 L4 / 0.992 full) and the control **MLP-128** can (1.000); the **TCN** cannot (0.102 ≈ chance at both histories).
* **Does the small-sample fit transfer?** No. The GRU's Class-A TEST accuracy is 0.093–0.105 and held-out-operand accuracy 0.092–0.111 — at/below chance and unchanged from the full-pool result.
* **Does the result change the X-B reading?** It **splits** it. For the GRU/MLP-128 the obstacle is **generalization / task structure** (they can fit a small sample but not transfer); for the TCN the obstacle is the **training/optimization instrument** (it cannot fit even 256 rows), so the TCN cannot justify ADD-specific claims.
* **Do E and F agree?** Yes (all-ADD identical to `< 1e-9`; both contribute to Y-MIX).
* **Was the protocol valid?** Yes — best-on-VAL reproduces X's protocol bit-for-bit, the subset is fixed/balanced/disjoint, and no TEST information enters selection.

## 27. Exactly ONE Next Experiment

**Experiment Z — bounded-reader instrument qualification for the non-fitting TCN (measurement-only, frozen checkpoints).** Y's headline is that the **TCN cannot fit even 256 examples** under the fixed protocol (train-max 0.102 at both histories), so *any* TCN-based ADD conclusion is confounded by the instrument. Hold the data (the same fixed 256-subset), the TCN architecture/width/parameter count, the splits and the optimizer **exactly fixed**, and change **only the fitting target** to a directly observable one: train the same TCN (and, as a control, the same GRU) on the 4-way `op_dest` target on the identical subset at L4/full under the identical protocol. Decision — **Z-A**: if the TCN drives `op_dest` training accuracy to ≥ 0.90 (it can fit an *observable* target but not `y_true`), the TCN non-fitting is **target-specific**, and the reader family is a valid instrument for the ADD question; the immediate next step is a representation-level measurement. **Z-B**: if the TCN cannot fit even the directly-observable `op_dest` target, the bottleneck is the **TCN training/optimization instrument itself**, and the reader family must be re-examined (or replaced within the fixed-budget design) **before** any ADD-specific conclusion. Measurement-only; no frozen weight/architecture change; no candidate/state modification; no Phase 2; specified, not started.

## Required Figures

`docs/reports/figures/experiment_y_*.png` (8): `training_fit`, `training_trajectory`, `train_vs_test`, `heldout_generalization`, `reader_comparison`, `l4_vs_full`, `per_seed`, `decision_summary`.

## Tests

`tests/test_experiment_y.py` (43 tests) — V/X reader reuse and architecture identity; exact 256-row subset construction and class balance; deterministic selection; target-generation correctness; held-out operand-pair exclusion; no duplicate `(sequence_id, timestep)`; sequence disjointness from canonical VAL/TEST; fixed VAL/TEST identity; causal history construction; future-observation leakage rejection; causal reader guards; fixed model/optimizer configuration; exactly 1,000 optimizer updates; milestone recording; maximum-training-accuracy calculation; final-checkpoint integrity; validation-only checkpoint selection; no TEST-based selection; trainer protocol identity with X; deterministic trajectories; fit-seed convention; fitting/generalization separation; E/F control; decision validity; reused W/X baselines; frozen-model protection (no gradients / no mutation); reader/model separation; checkpoint SHA verification; deterministic artifacts and figure generation; historical-report immutability; documentation-governance compliance.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_y.py \
    --w-npz experiments/experiment_w_data_scale.npz \
    --output experiments/experiment_y_fitting_control.npz
PYTHONPATH=src python experiments/analyze_experiment_y.py \
    --npz experiments/experiment_y_fitting_control.npz \
    --w-results experiments/experiment_w_results.json \
    --x-results experiments/experiment_x_results.json \
    --output experiments/experiment_y_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_y
PYTHONPATH=src pytest tests/test_experiment_y.py -v
PYTHONPATH=src pytest tests/
```

## Artifacts

* `experiments/run_experiment_y.py` — fixed-subset collector (draws the 256-row class-balanced subset from W's pool; copies W's VAL/TEST verbatim; records identity hashes and provenance)
* `experiments/analyze_experiment_y.py` — fitting-control analysis (fit vs generalization; three reported states)
* `experiments/experiment_y_fitting_control.npz`, `experiments/experiment_y_fitting_control_meta.json`
* `experiments/experiment_y_results.json`
* `docs/reports/figures/experiment_y_*.png` (8)
* `tests/test_experiment_y.py`

## Scientific Guardrails

Measurement-only; observational with respect to RE:Reason. The report keeps four things separate: **exact information availability** (U's Bayes ceilings), **reader learnability** (V/W/X/Y), the **frozen model's representation** (T/S), and **causal interpretation** (only within the reader-fitting manipulation). A successful small-sample fit is **not** evidence the reader learned the general ADD rule; a failed small-sample fit for one reader (the TCN) is a statement about that **instrument**, not about the task or the frozen model; a fit that does not transfer is **not** proof the information is absent; a high Bayes ceiling is **not** evidence a bounded reader can exploit it. Q–X are **not** edited; Y is appended as new evidence. No Phase 2. Negative, borderline and reader-specific findings are preserved.
