# RE:Reason — Experiment V: Bounded Sequence-Aware Access Test

**Date:** October 8, 2026
**Status:** Completed — Measurement only (frozen E/F checkpoints used only to define the Class-A population and reproduce baseline metrics; no model training; no optimizer on RE:Reason parameters; no architecture/encoder/recurrent-core/state/gate/candidate/commit/readout modification; no state replacement; no candidate manipulation; no feature editing; no oracle/reconstructed state as a reader feature; no branching/verifier; no counterfactual rollout; no Phase 2). The V readers are separate analysis-only models and never enter frozen model inference. Classification: **V-B — sequence-limited** for E and F.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned (unchanged before/after).
**Data:** `experiments/experiment_v_sequence_access.npz`, `experiments/experiment_v_sequence_access_meta.json`, `experiments/experiment_v_results.json`
**Runners:** `experiments/run_experiment_v.py`, `experiments/analyze_experiment_v.py`, tests `tests/test_experiment_v.py`

---

## 1. Title / Date / Status

Experiment V — Bounded Sequence-Aware Access Test. Measurement-only, frozen; **V-B (sequence-limited)**. No intervention; no Phase 2. Q, R, S, T and U reports are untouched; V is appended as new evidence.

## 2. Executive Answer

U classified the raw-interface result as **U-D (interface-identifiable but probe-limited)**: the raw causal observation history uniquely determines the ADD operands/target (target Bayes 0.290 → 0.998 at L=4 → 1.000 at the full prefix), yet the bounded **flat** probe family (linear / MLP-16..128 on the flattened one-hot window) stayed weak (best full-prefix 0.218 all-ADD, ~0.09 Class-A; held-out ≤ 0.25). U explicitly did **not** establish that the information is inaccessible to all bounded readers. V tests exactly one question:

> Does a parameter-bounded causal **sequence-aware** reader recover the information that is exactly identifiable from the raw L=4 history, or does the U-D gap survive when temporal structure is represented explicitly?

V fits **two** predeclared families — a single-layer **causal GRU** and a **causal dilated 1-D convolution (TCN-style)** — each with **fewer trainable parameters than U's MLP-128 baseline**, on the raw 22-feature per-timestep stream (`x_t` ⊕ validity bit), across U's exact history grid and populations, with U's exact splits, held-out-operand control, target definitions and no-future discipline.

**The U-D gap survives: explicit temporal structure does not resolve the weak accessibility.**

* **Both readers are weak on the primary population.** On **Class-A** (E), the best sequence reader reaches **0.112** (TCN at L=4) versus U's flat MLP-128 **0.089** — a sub-material gain of **+0.023**. At the **full prefix**: GRU **0.080**, TCN **0.109**, U MLP-128 **0.086**. On **all-ADD**: L=4 GRU 0.115 / TCN 0.176 / U 0.198; full prefix GRU 0.147 / TCN 0.214 / U 0.218 — the sequence readers never beat the flat baseline materially.
* **No reader reaches the recoverability bar.** Best reader Class-A accuracy **0.112 ≪ 0.50**; the exact Class-A Bayes ceiling at the same horizon is **0.999** (L=4) / **1.000** (full), giving a **Bayes − reader gap of 0.887**.
* **Held-out operand generalisation is weak everywhere.** Class-A held-out accuracy is **0.105–0.212** (all ≤ 0.25); the best reader's held-out accuracy is **0.105**.
* **The readers memorise rather than learn the rule.** They train to high training accuracy but stay near chance on the untouched test, and per-seed all-ADD accuracy is highly variable (TCN full-prefix: 0.093–0.306).
* **But the readers do learn *some* structure.** At the full prefix, the GRU recovers `op_dest` at **1.000** and the operands at **0.248 / 0.207** (Class-A) — far above its target accuracy (0.080) but still far below the operand Bayes ceiling of 1.000. So the limit is not "the reader learns nothing"; it is that the **target** is not accessed.
* **No subtype or model divergence.** E and F agree (all-ADD accuracy identical; `E_vs_F` Class-A differences ≤ 0.016; both classify V-B), and the two families do not produce a preregistered disagreement.

Under the pre-registered rule this is **V-B — sequence-limited**: the best sequence-aware reader has test ≤ 0.25, held-out ≤ 0.25, and Bayes − reader ≥ 0.30. The weak accessibility is **not resolved merely by providing temporal structure**. The result is **observational**, **not causal**, and **no intervention is justified**.

## 3. Q/R/S/T/U Baseline

* **Q/R**: the true written value is only weakly recoverable from the frozen `C_t` on Class-A (Q-C); bounded capacity / size-matched hard-tail training do not help and the only apparent gain is operand-combination memorization (R-D).
* **S**: no internal candidate activation materially beats `C_t` (S-C).
* **T**: the frozen observation encoder output `e_t` is structurally operand-blind and no upstream frozen representation is robustly recoverable (T-D).
* **U**: two independent evidence families on the **raw** causal observation history — **exact** interface Bayes (current observation operand-blind: L=1 target 0.290; **L=4 target 0.998, operands 0.999**; **full prefix 1.000**) and **learned flat probes** (best full-prefix 0.218 all-ADD, ~0.09 Class-A; held-out ≤ 0.25) → **U-D (interface-identifiable but probe-limited)**. U's §29 specified exactly this sequence-aware test as Experiment V.

## 4. Research Question

> Does a parameter-bounded causal **sequence-aware** reader recover the task-interface information that is exactly identifiable from the raw L=4 history, or does the U-D gap persist when temporal structure is represented explicitly?

V reports **exact information availability** (U's Bayes ceilings, unchanged) separately from **sequence-reader accessibility** (the V readers), and never conflates either with the **frozen model's representation** or a **causal mechanism**.

## 5. Hypotheses

* **H-A|V-A (sequence-accessible)**: at least one bounded sequence reader reaches test ≥ 0.50 AND held-out ≥ 0.30 at L=4 or full prefix.
* **H-B|V-B (sequence-limited)**: the best sequence reader stays ≤ 0.25 test AND ≤ 0.25 held-out with a Bayes − reader gap ≥ 0.30 — the U-D gap persists.
* **H-C|V-C (partial-sequence-benefit)**: a sequence reader improves over the U MLP-128 baseline by ≥ 0.15 without reaching V-A.
* **H-D|V-MIX**: material E/F disagreement, or a genuine GRU-vs-TCN disagreement that changes the conclusion.

## 6. Preregistration (declared before aggregate inspection)

**Reader families (exactly two):** (A) single-layer **causal GRU** (unidirectional); (B) **causal dilated 1-D convolution** TCN-style, kernel 3, dilation pattern [1, 2, 4, 8], four layers, one classification head. Per-timestep input = raw 21-dim observation `x_t` + 1 validity bit = **22 features**. No attention, no transformer, no bidirectional processing, no extra recurrence, no symbolic-arithmetic or reconstructed-state features. Prediction at `t` uses only `x_0..x_t`.

**Parameter budget:** every reader must satisfy `trainable_params ≤ U_MLP128_parameter_count` (the exact full-prefix U MLP-128, input dim 22·10 = 220 → **29,578**). Widths are the **largest integer strictly within budget**, derived deterministically against the **largest target class count** (raw_sum, 19) so that every target's reader fits with one shared width. **No width chosen after seeing results.**

**History grid (U's, unchanged):** `L ∈ {L1, L2, L4, L8, L16, Lfull}`; `H_L(t) = [x_{max(0,t−L+1)}, …, x_t]`, zero left-padded with an explicit validity mask; no future observation.

**Targets:** `y_true` (primary, 10 classes); `operand_a`/`operand_b` (10); `raw_sum` (19); `op_dest` (4). Operation is constant (ADD) and not probed. Variants: `main` (all ordinary-train ADD rows) and `heldout` (20 operand pairs `{(a,b): b∈{0,5}}` excluded).

**Scope (predeclared):** main fits for all five targets; `raw_sum`/`op_dest` at the decisive horizons `{L4, Lfull}`; held-out fits for `y_true` (all L) and the operands (decisive horizons).

**Training:** Adam 3e-3, weight-decay 1e-2, 200 epochs, class-balanced cross-entropy, best-on-VAL checkpoint, fixed deterministic seed. No tuning of LR/WD/epochs/width/kernel/dilation/layers.

**Splits:** ordinary TRAIN ADD (cap 1500/seed) → fit; ordinary VAL (cap 800/seed) → tune; canonical TEST → evaluate once.

**Populations:** Class-A hard (primary), matched candidate-correct, all candidate-correct, all ADD. E seeds 42–46 are the independent units; F is a control.

**Decision (checked in order; bars are declared conventions):**
* **V-A** — some reader test ≥ 0.50 AND held-out ≥ 0.30 at L4 or full.
* **V-B** — best reader test ≤ 0.25 AND held-out ≤ 0.25 AND Bayes − reader ≥ 0.30.
* **V-C** — improves over U MLP-128 by ≥ 0.15 without V-A.
* **V-MIX** — material E/F disagreement or a genuine GRU-vs-TCN disagreement.

No test-based history/width/architecture selection.

## 7. Frozen Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, optimizer, loss. **Nothing modified.** The frozen checkpoints define the Class-A population (`cand_ok`) and reproduce the baseline only. All model execution under `torch.no_grad()`; checkpoint SHA-256 verified before/after. The V readers are separate analysis-only `nn.Module`s; they are never attached to the frozen model, no RE:Reason parameter acquires a gradient, and the reader parameters are disjoint from the frozen model's.

## 8. Reader Architectures

```
GRU reader:   x(22) -> nn.GRU(22, 85, num_layers=1, bidirectional=False) -> last-step hidden -> Linear(85, C)
              prediction uses only x_0..x_t (causal)
TCN reader:   x(22) -> Conv1d(22, 52, k=3, dilation=1)   [causal pad then trim] -> ReLU
                    -> Conv1d(52, 52, k=3, dilation=2)  [causal] -> ReLU
                    -> Conv1d(52, 52, k=3, dilation=4)  [causal] -> ReLU
                    -> Conv1d(52, 52, k=3, dilation=8)  [causal] -> ReLU
                    -> last-position feature -> Linear(52, C)
```

The TCN is genuinely causal: each convolution left-pads by `(k−1)·d` and trims `(k−1)·d` from the right, so the feature at position `i` depends only on inputs `≤ i` (verified in tests: changing `x_{≥t+1}` leaves positions `≤ t` bit-identical). The GRU is unidirectional (verified `bidirectional is False`).

The final widths are **GRU hidden = 85** and **TCN channels = 52** (kernel 3, dilations [1,2,4,8], 4 layers).

## 9. Exact Parameter-Budget Derivation

The U MLP-128 baseline at the full-prefix representation (input dim 220) has

```
params = 220·128 + 128 + 128·10 + 10 = 28,160 + 128 + 1,280 + 10 = 29,578.
```

Widths are chosen as the largest integer whose total trainable parameter count (including the classification head, evaluated at the largest class count 19) is ≤ 29,578:

| Reader | Formula | Width | Params (19-class head) | Params (10-class primary) | Ratio to 29,578 |
|---|---|---:|---:|---:|---:|
| GRU | `3h·22 + 3h² + 6h + 19h + 19` | **85** | **29,429** | 28,655 | **0.99496** |
| TCN | `(22·c·3+c) + 3(c²·3+c) + 19c+19` | **52** | **28,983** | 28,506 | **0.97988** |

Both readers are **within** the U MLP-128 budget (never exceeding it); the architecture-selection algorithm is deterministic and unit-tested (width + 1 must exceed the budget).

## 10. Causal History Representation

For every ADD transition the reader receives `H_L(t) = [x_{max(0,t−L+1)}, …, x_t]` as an `L × 22` tensor (`x` = the raw 21-dim one-hot, plus the validity bit). Windows are constructed by U's exact routine and are invariant to any modification of observations after `t`. `L16` and `Lfull` encode the same real observations for T=10 (different padding) and are both preserved.

## 11. Data / Splits

The **DisjointSplitManager** splits (sequence-disjoint by construction): TRAIN 6,000 / VAL 1,200 / canonical TEST 1,200×10 (stream `90042+seed`). Per seed: ordinary TRAIN ADD (1500) → fit; ordinary VAL ADD (800) → tune; canonical TEST → evaluate once. Collection reuses U's `capture_split` verbatim; the V npz arrays are bit-identical to U's stored columns, and checkpoint SHA-256 is verified before/after.

## 12. Populations

Class-A hard (frozen candidate written-register decode ≠ truth; E per-seed 448/330/363/327/337), matched candidate-correct (nearest-neighbour on frozen-only metadata), all candidate-correct, all ADD. The reader is model-independent, so it is trained once per seed and evaluated on both E and F masks; the all-ADD and candidate-correct populations are model-independent (verified: E/F all-ADD accuracies identical to 1e-6).

## 13. Training Protocol

Identical to U/R/S/T: Adam (lr 3e-3, weight-decay 1e-2), 200 epochs, full-batch, class-balanced cross-entropy, best-on-VAL checkpoint, deterministic per-fit seed `SEED_BASE + seed·100000 + family·10000 + target·1000 + rep·10 + variant`. No early-stopping on TEST; no TEST-based tuning.

## 14. Target Definitions

`y_true` = the written register value = `(operand_a + operand_b) mod 10`; `operand_a`/`operand_b` = the pre-transition register values at the destination/source; `raw_sum` = `operand_a + operand_b` (19 classes); `op_dest` = the destination register index (4 classes). Operation is constant ADD.

## 15. Results

TEST accuracy, mean over 5 E seeds (`main` variant). **Class-A** (primary):

| History | GRU | TCN | U MLP-128 | Class-A Bayes |
|---|---:|---:|---:|---:|
| L1 | 0.098 | 0.110 | 0.097 | 0.161 |
| L2 | 0.104 | 0.115 | 0.100 | 0.501 |
| L4 | 0.083 | **0.112** | 0.089 | **0.999** |
| L8 | 0.081 | 0.111 | 0.089 | 1.000 |
| L16 | 0.088 | 0.096 | 0.081 | 1.000 |
| full | 0.080 | 0.109 | 0.086 | 1.000 |

**All-ADD** (interface control):

| History | GRU | TCN | U MLP-128 |
|---|---:|---:|---:|
| L1 | 0.090 | 0.212 | 0.100 |
| L2 | 0.078 | 0.128 | 0.168 |
| L4 | 0.115 | 0.176 | 0.198 |
| L8 | 0.149 | 0.179 | 0.219 |
| L16 | 0.147 | 0.086 | 0.220 |
| full | 0.147 | 0.214 | 0.218 |

**The sequence-aware readers are weak on Class-A (≈0.08–0.12) and never beat the flat MLP-128 materially on all-ADD.** Per-seed full-prefix all-ADD accuracy is variable (GRU 0.120–0.185; TCN 0.093–0.306), consistent with memorisation rather than rule learning. Target-vs-operand at the full prefix (Class-A): GRU `op_dest` 1.000, `operand_a` 0.248, `operand_b` 0.207, `y_true` 0.080; TCN `op_dest` 0.228, `operand_a` 0.086, `operand_b` 0.109, `y_true` 0.109 — the readers recover the destination (and, for the GRU, the operands partially) but not the target.

## 16. Bayes-vs-Reader Comparison

| History | Class-A Bayes | best reader | Bayes − reader |
|---|---:|---:|---:|
| L1 | 0.161 | 0.110 | 0.051 |
| L2 | 0.501 | 0.115 | 0.386 |
| L4 | **0.999** | **0.112** | **0.887** |
| full | **1.000** | 0.109 | **0.891** |

At L=4 the exact ceiling is **0.999** but the best reader reaches **0.112**; the gap **0.887 ≥ 0.30**. The information is present — the readers do not access it. (Bayes ceilings are U's, reused unmodified; the readers never influence them.)

## 17. Held-Out Operand Generalization

Probe fit excluding the 20 pairs `{(a,b): b∈{0,5}}`, evaluated on Class-A `y_true`:

| History | GRU seen | GRU held-out | TCN seen | TCN held-out |
|---|---:|---:|---:|---:|
| L4 | 0.085 | 0.146 | 0.107 | 0.105 |
| full | 0.095 | 0.148 | 0.087 | 0.175 |

All held-out accuracies are **≤ 0.25** (0.105–0.212 across all L), and the best reader's held-out accuracy is **0.105** — no sequence reader achieves operand generalisation. There is no seen ≫ held-out memorisation signature on Class-A because the reader never learns the rule well enough to over-fit it.

## 18. Candidate-Correct Controls

On candidate-correct ADD the readers are likewise near chance (the reader is model-independent; the candidate-correct subset is simply the complement of Class-A on the shared all-ADD test). The all-ADD control (§15) is the clean interface measurement: the TCN approaches the flat baseline (0.214 vs 0.218 at the full prefix) but neither is recoverable.

## 19. E/F Control

The raw interface is model-independent, so the reader results are shared: all-ADD accuracies are numerically identical between E and F (to 1e-6), and the Class-A `E_vs_F` paired differences are ≤ 0.016 (GRU|L4 −0.016, GRU|full −0.002, TCN|L4 +0.008, TCN|full −0.015). Both models classify **V-B**; `E_F_disagreement = false`.

## 20. Leakage / Causality Validation

Enforced and tested: (i) reader input is invariant to randomising every observation after `t`; (ii) the full prefix ends exactly at `t`; (iii) a deliberately leaky builder that consumes `x_{t+1}` is shown to differ from the causal input; (iv) a **bidirectional** GRU is shown to leak the future and is rejected — the declared GRU is unidirectional; (v) the **TCN** output at each position is verified invariant to observations after it; (vi) TRAIN/VAL/TEST sequences are disjoint; (vii) held-out operand pairs never enter fitting (the held-out fit's `n_train` excludes them exactly); (viii) TEST labels never tune the architecture, hyper-parameters or training; (ix) frozen RE:Reason parameters acquire no gradients and are not mutated by reader training; (x) reader parameters are never attached to the frozen model; (xi) a deliberate test-fitting leakage path is included; (xii) collector and reader outputs are deterministic across reruns; (xiii) checkpoint SHA-256 unchanged.

## 21. Parameter Parity

`U MLP-128 = 29,578`; `GRU (h=85) = 29,429 (0.995×)`; `TCN (c=52) = 28,983 (0.980×)` — both **within** budget for every target class count (max 19). No reader exceeds the flat baseline's parameter count.

## 22. Deterministic Reproduction

Two full collection/analysis runs produce bit-identical collected arrays, an analysis JSON identical except the timestamp, and byte-identical figures (verified). Reader training is deterministic per fit (fixed seed), and the architecture selection is a pure function of the budget.

## 23. Causal Status

**Observation only.** V measures whether bounded causal sequence-aware readers access the information U showed is exactly identifiable at the raw interface. It does **not** show the frozen recurrent state contains the information, that the encoder should have learned it, that the architecture is insufficient, or that any intervention will solve the task. The wording used is "the tested bounded sequence readers do not access the L4-identifiable information", "the U-D gap persists", and "this does not establish that the information is inaccessible to all bounded readers". No causal mechanism has been established and no intervention is performed or licensed.

## 24. Decision Classification

**V-B — sequence-limited** (E primary; F agrees).
* Best reader (Class-A) = **0.112** (TCN/L4) — not V-A (test < 0.50).
* Best reader held-out = **0.105** ≤ 0.25, best reader test **0.112** ≤ 0.25, Bayes − reader = **0.887** ≥ 0.30 → **V-B**.
* Gain over the U MLP-128 baseline = **+0.023** (< 0.15) → not V-C.
* E and F categories agree; GRU and TCN do not produce a preregistered disagreement (both weak) → not V-MIX.
* Interpretation: **the U-D gap persists for the tested bounded causal sequence readers; weak accessibility is not resolved merely by providing temporal structure.**

## 25. What V Establishes

* A deterministic, leakage-controlled **sequence-aware access test** with two predeclared, parameter-bounded causal readers (GRU h=85, TCN c=52), each **within** the exact U MLP-128 budget, with a predeclared architecture-selection algorithm.
* That **neither bounded sequence reader materially beats the flat MLP-128** on Class-A (best 0.112 vs 0.089) or all-ADD (TCN 0.214 vs 0.218 at the full prefix).
* That **no sequence reader reaches the 0.50 bar** and held-out stays ≤ 0.25, giving a **0.887 Bayes − reader gap** at L=4.
* That the readers **recover the destination (GRU 1.000) and partially the operands (0.21–0.25) but not the target** — the failure is target-specific, not a total reader failure.
* That **E and F agree**, and the two families do not disagree — classification **V-B**.

## 26. What V Does Not Establish

* It does **not** show the information is inaccessible to **all** bounded readers; only that the tested GRU/TCN family does not access it.
* It does **not** establish that the frozen recurrent state contains (or fails to contain) the information; **availability ≠ reader accessibility ≠ frozen representation**.
* It does **not** show the encoder should have learned the mapping, that the architecture is insufficient, or that a specific intervention would help.
* It does **not** prove a causal mechanism; it is measurement-only.
* It does **not** claim the benchmark is unsolvable or that the readers are optimal.

## 27. Final Answers

1. **Does the current observation `x_t` determine the operands/target?** **No** (U: L=1 target Bayes 0.290; L=1 remains the same here).
2. **Is the L=4 raw history exactly identifiable?** **Yes** — Class-A target Bayes **0.999** (U, reused); full prefix **1.000**.
3. **Does a bounded causal sequence reader reach test ≥ 0.50?** **No** — best Class-A reader **0.112**.
4. **Does it reach held-out ≥ 0.30?** **No** — best held-out **0.105** (all held-out ≤ 0.25).
5. **Does a sequence reader beat the flat U MLP-128 materially?** **No** — Class-A gain **+0.023**; all-ADD TCN 0.214 vs 0.218.
6. **What is the Bayes − reader gap at L=4?** **0.887**.
7. **Do the readers learn anything?** **Yes** — `op_dest` 1.000 (GRU), operands 0.21–0.25 (GRU), but the target stays weak.
8. **Do E and F agree?** **Yes** — identical all-ADD reader accuracy, Class-A differences ≤ 0.016.
9. **Do GRU and TCN disagree materially?** **No** — both weak.
10. **Classification?** **V-B — sequence-limited.**
11. **Does V justify an intervention?** **No.** No reader is recoverable; the result is measurement-only and licenses no architectural change.

## 28. Exactly ONE Next Experiment

V-B shows that neither the flat probe family (U) nor two parameter-bounded **causal sequence** readers (V) access information that is **exactly identifiable from a 4-step raw history**. The readers train to high training accuracy but generalise near chance, i.e. they **memorise** training windows rather than learning the register-machine rule. The single unresolved question is therefore whether the persistent V-B gap is a **training-data / optimisation limit** of the bounded readers or a **bounded-capacity / algorithmic** limit.

**Experiment W — bounded-reader learnability vs training-data scale (measurement-only, frozen checkpoints).** Train the *same* predeclared readers (U's flat MLP-128 anchor, V's GRU and TCN) on progressively larger **sequence-disjoint i.i.d.** ordinary ADD training pools (e.g. 1.5k → 8k → 32k transitions from fresh disjoint streams), with the same held-out-operand and sequence-disjoint controls and the same targets/populations. If test/held-out recoverability rises toward the exact Bayes ceiling as data grows, the V-B gap is **data/optimisation-limited**; if it stays flat while training accuracy rises, it is a **bounded-capacity / algorithmic** limit (the interface information, though present, is not statistically learnable at this budget). W is measurement-only, changes no weights/architecture, reuses U/V collection and the V readers, and begins no Phase 2.

The governing loop remains **MEASURE → LOCALIZE → INTERPRET → KEEP/KILL → ONE CONTROLLED NEXT TEST**.

## Required Figures

`docs/reports/figures/experiment_v_*.png` (12): `reader_vs_history`, `reader_comparison`, `bayes_vs_reader`, `heldout_vs_history`, `seen_vs_heldout`, `l4_comparison`, `fullprefix_comparison`, `per_seed`, `classA_vs_correct`, `target_vs_operands`, `parameter_budget`, `decision_summary`.

## Tests

`tests/test_experiment_v.py` — **34 tests**: U baseline loading and bridge; exact parameter-budget calculation; GRU/TCN within the MLP-128 budget; deterministic architecture construction; raw 22-dim reader input; causal window construction and explicit validity mask; no-future leakage; deliberate leaky-builder rejection; bidirectional-GRU rejection; causal TCN verification; GRU last-step causality; deterministic reader output; no RE:Reason parameter mutation; no gradients on the frozen checkpoint; reader-parameter separation; sequence disjointness; held-out operand exclusion; train/val/test separation; best-checkpoint-from-VAL; target generation; Class-A count reproduction; candidate-correct and all-ADD controls; E/F agreement; history-grid completeness; exact reader parameter budget; deterministic artifacts; JSON determinism; malformed-artifact rejection; figure creation (≥12); historical-report immutability; research-governance compliance.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_v.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_v_sequence_access.npz
PYTHONPATH=src python experiments/analyze_experiment_v.py \
    --npz experiments/experiment_v_sequence_access.npz \
    --u-results experiments/experiment_u_results.json \
    --output experiments/experiment_v_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_v
PYTHONPATH=src pytest tests/test_experiment_v.py -v      # 34 tests
PYTHONPATH=src pytest tests/                             # full suite
```

Determinism: collector arrays bit-identical across two runs; analysis JSON identical (excluding the timestamp); all 12 figures byte-identical. Checkpoint SHA-256 verified before/after. No RE:Reason weight is modified. The analyzer reuses `analyze_experiment_s` probe primitives and `analyze_experiment_u` history/Bayes/population utilities.

## Scientific Guardrails

Measurement-only; observational. The report keeps four things separate: **exact information availability** (U's Bayes ceilings: L=4 0.999, full 1.000), **sequence-reader accessibility** (V: ≤ 0.112 Class-A), the **frozen model's representation** (T/S: weak; not re-measured here), and **causal interpretation** (none). It says "the tested bounded causal sequence readers do not access the L4-identifiable information" and "the U-D gap persists", never that the information is inaccessible to all bounded readers, that the frozen state contains it, that the encoder should have learned it, that the architecture is insufficient, or that a particular intervention will solve the task. A failed bounded reader is **not** proof the information is mathematically absent (the exact ceiling proves the opposite); a high Bayes ceiling is **not** evidence the frozen network can exploit it. Q, R, S, T and U are **not** edited; V is appended as new evidence. No Phase 2. Negative and borderline findings (weak sequence readers, high per-seed variance, partial operand recovery, no subtype benefit) are preserved.
