# RE:Reason — Experiment U: Raw-Observation / Task-Encoding Localisation

**Date:** October 8, 2026
**Status:** Completed — Measurement only (frozen checkpoints used only to define the Class-A population and reproduce baseline metrics; no model training; no optimizer on model weights; no architecture/encoder/recurrent-core/state/gate/candidate/commit/readout modification; no inference-time intervention; no state replacement; no feature editing; no branching/verifier/oracle; no counterfactual rollout; no T/S/R/Q rerun; no Phase 2). The raw-observation/history probes are analysis-only objects; they never enter model inference. Classification: **U-D — interface-identifiable but probe-limited** for E and F.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned (hashes re-verified; unchanged before/after).
**Data:** `experiments/experiment_u_task_interface.npz`, `experiments/experiment_u_task_interface_meta.json`, `experiments/experiment_u_results.json`
**Runners:** `experiments/run_experiment_u.py`, `experiments/analyze_experiment_u.py`, tests `tests/test_experiment_u.py`

---

## 1. Title / Date / Status

Experiment U — Raw-Observation / Task-Encoding Localisation. Measurement-only, frozen; **U-D (interface-identifiable but probe-limited)**. No intervention; no Phase 2. Q, R, S and T reports are untouched; U is appended as new evidence.

## 2. Executive Answer

T concluded T-D: the frozen observation encoder is structurally operand-blind and no tested upstream representation is robustly recoverable. U asks the representation-independent question that precedes the model entirely: **does the raw task interface itself contain enough causal information to determine the current operands/target, and how much observation history is required before that information becomes (a) exactly identifiable and (b) accessible to a bounded learned probe?**

U separates two evidence families (spec §3): **exact interface evidence** (identical-history collisions, unique-history counts, Bayes-optimal accuracy, collision/ambiguity mass, target entropy — computed with no learned probe) and **learned probe evidence** (R/S/T's exact probe family on the raw observation history).

**The raw task interface does contain the target information, and remarkably little history is required — but the bounded flat-probe family cannot access it.**

* **The current observation is operand-blind and target-ambiguous.** At L=1 the observation determines the destination register exactly (`op_dest` Bayes = **1.000**) but not the operands: `operand_a` **0.437**, `operand_b` **0.444**, written/modular target **0.290** (all-ADD; Class-A target **0.161**) — every L=1 history is a collision (`collision_rate` **1.000**, `ambiguous_mass` **1.000**), with 16 unique histories over 15,763 rows. This is the exact representation-independent confirmation of T's encoder operand-blindness.
* **A four-step causal window makes the target identifiable.** The Bayes ceiling rises **0.290 → 0.402 → 0.998** (L=1 → L=2 → L=4) for the target and **0.437/0.444 → 0.537/0.538 → 0.999/0.999** for the operands; at L=4 the target collision rate falls to **0.016** and the ambiguous mass to **0.004**. The **interface-identifiable history horizon is L\* = L4** (the minimum tested window with Bayes ≥ 0.99), and the **full causal prefix is uniquely sufficient** (target/operands/raw-sum Bayes = **1.000**, zero ambiguous mass).
* **The bounded learned probes stay weak.** The best full-prefix probe on all-ADD reaches only **0.218** (MLP-128, mean of 5 seeds; MLP-32 0.205; linear 0.164), and on the Class-A tail **~0.09** (chance 0.10). Held-out *operand* accuracy of the full-prefix probe is **0.121** (all-ADD) / **0.082** (Class-A) — all ≤ 0.25. The full-prefix probe memorizes its training windows (train accuracy **0.913**) but does not learn the rule. The **Bayes-minus-probe gap at the full prefix is 0.782**.
* **The probe limitation is not confined to the modular composition.** Even the *directly identifiable* operands are weakly recovered by a probe (operand_a MLP-32: 0.077 → 0.290 across L=1 → full), far below their Bayes ceiling of 1.0 — so the limitation is the flat-reader's access to the raw one-hot interface, not the modular operation specifically.
* **No subtype or model divergence.** Wrap/no-wrap/repeated/isolated all reach target Bayes ≈ 1.0 by L=4 (max spread 0.096 < 0.15 bar); raw-sum is not easier than modular; **E and F are identical** (the raw interface and the all-ADD population are model-independent).

Under the pre-registered rule this is **U-D — interface-identifiable but probe-limited**: the information is mathematically present at the raw task interface (uniquely at the full prefix, and ≥0.99-identifiable from a 4-step window), but the tested bounded probe family does not exploit it. The strong **history-dependence** (U-C: L1 0.290 → L4 0.998) is the mechanism. The result is **observational**, **not causal**, and **no intervention is justified**.

## 3. Existing Baseline

* **G–I**: bit-exact commit; a real candidate→commit residual (~5.5% of candidate-correct adds).
* **L/M/N/O**: the off-path third basin is latent placement (B), a systematic **parity/modular** geometry (M-A), **state-alignment-dominant** (N-B); parity-direction removal is non-specific and harmful (**O-D**).
* **P**: the sequence-level first-error budget is **candidate-generation-dominated** (A 59.7% T=10 / 67.2% T=20).
* **Q/R**: the true written value is only weakly recoverable from the frozen `C_t` on the Class-A population (Q-C), robust to bounded capacity / size-matched hard-tail training; the only apparent gain is operand-combination memorization (R-D).
* **S**: no internal candidate activation materially beats `C_t`; held-out-operand generalisation fails at every candidate stage (S-C).
* **T**: the frozen observation encoder output `e_t` is **structurally operand-blind** (destination 1.000, `operand_a`/`operand_b` at chance 0.110/0.136), the state path is only descriptively stronger and non-generalising, and no upstream stage reaches the 0.50 bar (T-D). **T's §32 specified exactly this raw-interface test as Experiment U.**

## 4. Research Question

> Does the raw causal observation stream `x_0…x_t` uniquely determine the current operands and target without any latent state, and at what causal history length does that information become exactly identifiable and accessible to a bounded learned probe?

U separates **information availability at the task interface** (exact collision/Bayes) from **accessibility to the frozen neural representation** (learned probes), and reports both (spec §3). It runs independently on E Class-A and F Class-A (seed is the unit) with the shared all-ADD population as a model-independent interface control.

## 5. Hypotheses

* **H-A|U-A (interface-complete)**: the full causal history has a high Bayes ceiling for operands and target AND learned probes approach it with held-out generalisation.
* **H-B|U-B (interface-incomplete)**: the full causal history retains substantial collisions / low Bayes ceiling.
* **H-C|U-C (history-dependent)**: the current observation is insufficient, and increasing causal history systematically raises the Bayes ceiling / probe accuracy until a finite horizon makes the target identifiable.
* **H-D|U-D (interface-identifiable but probe-limited)**: the Bayes ceiling is high but learned probes remain weak.
* **H-E|U-MIX**: different targets/subtypes/history regimes disagree materially.

## 6. Preregistration (declared before aggregate inspection)

**Causal history grid (spec §4):** `L ∈ {1, 2, 4, 8, 16}` and the **full causal prefix** (benchmark T=10). Each window ends at the current timestep t; early timesteps use the available causal prefix, left-padded with an explicit binary validity mask. **No future observation is used.**

**Representations:** flattened raw one-hot observation window + mask (dims 22·L). The primary probe input is **raw x vectors only** — no latent state, register values, reconstructed operands, task labels beyond the recorded targets, candidate logits, gate values, or hidden model features.

**Probe family (R/S/T's, fixed, not extended after results; imported verbatim from `analyze_experiment_s.py`):** linear ridge (closed-form, class-balanced, λ∈{1e-2,1e-1,1,10,100} on val); MLP 16/32/64/128 (tanh, Adam 3e-3, weight-decay 1e-2, 200 epochs, best-on-val state, fixed seed, balanced class weights).

**Splits:** DisjointSplitManager; fit on ordinary TRAIN ADD transitions (cap 1500/seed), tune on ordinary VAL (cap 800/seed), evaluate **once** on the untouched canonical TEST.

**Populations:** Class-A hard (frozen candidate wrong), matched candidate-correct (deterministic nearest-neighbour on frozen-only metadata), all candidate-correct, all ADD. **Targets:** `y_true` (primary), `operand_a`, `operand_b`, `raw_sum`, `op_dest`. (Operation is constant = ADD on this population and is reported as structurally non-informative rather than probed.)

**Exact interface evidence (no learned probe):** identical-history collisions, unique-history count, Bayes-optimal accuracy, collision rate, ambiguous mass, max multiplicity, mean target entropy.

**Held-out operand protocol:** 20 pairs `{(a,b): b∈{0,5}}` excluded from fitting.

**Decision classes (checked in order; bars are declared conventions):**
* **U-MIX** — E/F categories disagree, or wrap/no-wrap target-Bayes spread ≥ 0.15 at ≥ 2 lengths.
* **U-B** — `Bayes_target(full prefix) < 0.99`.
* **U-D** — `Bayes_target(full) ≥ 0.99` AND best full-prefix learned probe held-out ≤ 0.25 AND (Bayes − probe) gap ≥ 0.30.
* **U-A** — `Bayes_target(full) ≥ 0.99` AND best full-prefix learned probe ≥ 0.50 AND its held-out accuracy ≥ 0.30.
* **U-C** — `Bayes_target(L=1) ≤ 0.50` AND the Bayes curve is non-decreasing (tol 0.02) and reaches ≥ 0.99 at a finite length.
* else **U-MIX**.

No test-based history/capacity selection: every predeclared history length and capacity is reported.

## 7. Frozen Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, optimizer, loss. **Nothing modified.** The frozen checkpoints are used **only** to define the Class-A population (`cand_ok`) and reproduce the baseline; no model feature is stored as a probe input. All model execution under `torch.no_grad()`; checkpoint SHA-256 verified before/after. The only fitted objects are analysis-only probes — separate Python objects, never wired into `forward`, `decision_head`, or any rollout. Exact state reconstruction (spec §21) is a **validity control only** and is never a probe input.

## 8. Actual Task-Observation Schema

The benchmark encodes each step as a 21-dim one-hot:

```
x_t[0:3]   op one-hot      (SET=0, SWAP=1, ADD=2)
x_t[3:7]   reg_a one-hot   (destination register for SET/ADD; first register for SWAP)
x_t[7:11]  reg_b one-hot   (source register for SWAP/ADD; unused for SET)
x_t[11:21] immediate one-hot (value for SET; all-zero for SWAP/ADD)
```

The generator (`RegisterTrackingTask`) initialises all registers to 0, forces the first two steps to be SET, then draws SET/SWAP/ADD uniformly; ADD computes `reg_a = (reg_a + reg_b) mod 10`. The observation at an ADD step therefore contains the operation, the destination and source register indices, and **no operand values** — the operands live only in the running register state. U confirms this exactly: at L=1 the destination is Bayes-determined (1.000) while the operands are not.

## 9. Dataset / Split

The **DisjointSplitManager** splits (no sequence overlap by construction): TRAIN 6,000 / VAL 1,200 / canonical TEST 1,200×10 (stream `90042+seed`), sequence-disjoint. Per model, per seed: ordinary TRAIN ADD subsample (1500/seed) → probe fitting; ordinary VAL ADD subsample (800/seed) → probe tuning; canonical TEST → evaluation. Captured per ADD transition: the raw observation sequence `x_0…x_9` (210 dims), `t`, `seq`, and the metadata `y_true, raw_sum, cand_ok, cand_pred, op_dest, src, operand_a, operand_b, wrap, repeated, m_cand, offpath, norm_C, norm_dC`. Collection is bit-identical across two runs.

## 10. Class-A Population

Q's Class-A definition, unchanged: an **ADD transition whose frozen candidate written-register decode ≠ the true written value**. Per-seed n (E): **448 / 330 / 363 / 327 / 337**; canonical test ADD totals (E): **3,199 / 3,196 / 3,171 / 3,125 / 3,072** — reproducing Q/R/S/T exactly. The shared all-ADD test population (15,763 rows over 5 seeds) is model-independent and serves as the interface control. The five E checkpoints are the independent units; F is a control.

## 11. Current-Observation Analysis

Exactly, at L=1 (pooled over 5 seeds, E; Class-A in parentheses):

| Target | L=1 Bayes (all-ADD) | L=1 Bayes (Class-A) | L=1 collision rate | L=1 ambiguous mass |
|---|---:|---:|---:|---:|
| written/modular target | 0.290 | 0.161 | 1.000 | 1.000 |
| operand_a | 0.437 | 0.148 | 1.000 | 0.994 |
| operand_b | 0.444 | 0.145 | 1.000 | 0.994 |
| raw sum | 0.249 | — | 1.000 | 1.000 |
| destination | 1.000 | 1.000 | (by construction) | — |

There are **16 unique L=1 histories** (`(dest, src)` combinations) over 15,763 rows. **Every L=1 history is ambiguous for the target** (`ambiguous_mass = 1.000`): the current observation contains the operation and destination but **not** the operand values, so identical `x_t` maps to many different targets. The Class-A tail is *more* current-observation-ambiguous than all-ADD (target Bayes 0.161 vs 0.290) — the frozen model's failures concentrate where the current observation is even less informative. This is the representation-independent form of T's `e_t` operand-blindness.

## 12. History-Window Construction

For a representation length L the window is `H_L(t) = [x_{max(0,t−L+1)}, …, x_t]`, left-padded to L slots with zeros and an explicit L-bit validity mask (the mask encodes the real causal-prefix length, which is known at t). `cand_input`-style feature = flattened window ⊕ mask (dims 22·L). The full causal prefix is represented over the benchmark maximum T=10 (dims 220). Construction is tested to use only observation indices ≤ t, to be deterministic, and to be invariant to modifications of observations strictly after t. (Because T=10, the L=16 and full-prefix representations encode the same real observations; they are reported separately for completeness and behave identically.)

## 13. Exact Collision / Bayes Analysis

Pooled over 5 seeds (E), all-ADD. Bayes ceiling:

| History | unique histories | collision rate | ambiguous mass | target Bayes | operand_a Bayes | operand_b Bayes | raw-sum Bayes | mean target entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L=1 | 16 | 1.000 | 1.000 | 0.290 | 0.437 | 0.444 | 0.249 | 2.915 |
| L=2 | 1,088 | 1.000 | 0.972 | 0.402 | 0.537 | 0.538 | 0.373 | 2.277 |
| L=4 | 15,634 | 0.016 | 0.004 | **0.998** | **0.999** | **0.999** | **0.998** | 0.004 |
| L=8 | 15,672 | 0.011 | 0.000 | **1.000** | 1.000 | 1.000 | 1.000 | 0.000 |
| L=16 | 15,672 | 0.011 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| full prefix | 15,672 | 0.011 | 0.000 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |

**The full causal prefix is uniquely sufficient** (target Bayes = 1.000, ambiguous mass 0.000). The residual 1.1% "collision rate" at L≥4 is rows sharing an identical prefix (e.g. repeated windows across seeds) that are **target-consistent** — i.e. the same window maps to the same target, which is exactly the deterministic task generator. Class-A target Bayes reaches **0.999 at L=4** and 1.000 at L≥8.

## 14. Learned Probe Results

TEST accuracy (mean of 5 E seeds). All-ADD:

| Representation | linear | MLP-16 | MLP-32 | MLP-64 | MLP-128 |
|---|---:|---:|---:|---:|---:|
| L1 (raw obs) | 0.060 | 0.128 | 0.150 | 0.128 | 0.100 |
| L2 | 0.114 | 0.153 | 0.155 | 0.165 | 0.168 |
| L4 | 0.161 | 0.182 | 0.191 | 0.195 | 0.198 |
| L8 | 0.166 | 0.195 | 0.203 | 0.214 | 0.219 |
| L16 | 0.164 | 0.195 | 0.208 | 0.214 | 0.220 |
| full prefix | 0.164 | 0.197 | 0.205 | 0.212 | **0.218** |

Class-A: MLP-32 **0.107 → 0.084** across L1 → full prefix (chance 0.10); no capacity reaches the 0.50 bar. The full-prefix probe **memorizes** its training windows (MLP-128 train accuracy **0.913**) but does not learn the register-machine rule (test 0.218). Operand probes (MLP-32, all-ADD) rise 0.077 → 0.290 (operand_a) / 0.081 → 0.278 (operand_b) — still far below their Bayes ceiling of 1.0, so the limitation is not confined to the modular composition.

## 15. Bayes-vs-Probe Comparison

| Representation | Bayes (target, all-ADD) | best probe (all-ADD) | gap |
|---|---:|---:|---:|
| L1 | 0.290 | 0.150 | 0.140 |
| L2 | 0.402 | 0.168 | 0.234 |
| L4 | 0.998 | 0.198 | 0.800 |
| L8 | 1.000 | 0.219 | 0.781 |
| L16 | 1.000 | 0.220 | 0.780 |
| full prefix | 1.000 | 0.218 | **0.782** |

The gap **widens** with history: exactly where the exact interface becomes identifiable (L=4), the bounded probe fails to follow. This is the **Bayes-high / probe-low** signature of **U-D** (information present in raw history, not accessible to the chosen probe family). The probes at L≥4 are also capacity-flat (MLP-16 0.197 ≈ MLP-128 0.218), so the failure is not a capacity deficiency within the bounded family.

## 16. Held-Out Operand Generalization

Probe fit excluding 20 operand pairs `{(a,b): b∈{0,5}}`, evaluated on the untouched test (all-ADD, MLP-32):

| Representation | seen pairs | held-out pairs | gap |
|---|---:|---:|---:|
| L1 | 0.097 | 0.142 | −0.045 |
| L2 | 0.120 | 0.125 | −0.005 |
| L4 | 0.131 | 0.141 | −0.010 |
| L8 | 0.160 | 0.117 | 0.043 |
| L16 | 0.152 | 0.121 | 0.031 |
| full prefix | 0.157 | **0.121** | 0.036 |

Held-out accuracy is **≤ 0.25 at every length** (Class-A held-out 0.082–0.124). The history probe does **not** achieve operand generalisation: no representation is "recoverable" on learned-probe evidence, and there is no seen≫held-out memorization signature here — the probe is weak on seen and held-out alike (it never learns the rule well enough to memorise operand combinations either).

## 17. History Minimality

The **interface-identifiable history horizon is L\* = L4** for the target, `operand_a`, `operand_b` and the raw sum — the minimum predeclared window at which the exact Bayes ceiling ≥ 0.99 (target 0.998, operands 0.999). L=2 is partially informative (target 0.402, operands 0.54), and L=1 is the fully ambiguous current observation. This is the **minimum tested raw-observation-history length needed to make the target nearly identifiable**; it is **not** the model's required memory depth, and the frozen recurrent state (T) does not demonstrate this horizon.

## 18. Exact State-Reconstruction Control

Replaying the register machine from `x_0…x_{t-1}` and reading the operands/target at t reproduces the stored ground truth **exactly for every row** (`operand_a`, `operand_b`, `y_true`, `dest`, `src` all 100% equal, both models, all seeds). This is the required validity control showing the benchmark generator is internally deterministic and that the target is a deterministic function of the causal observation prefix. The reconstructed state is **never** used as a probe input.

## 19. Raw-Sum vs Modular

For ADD the written value and the modular target coincide: `y_true = (a+b) mod 10`. The exact ceiling is identical for both (0.998 at L=4, 1.000 at L≥8). Under learned probes neither is easy: raw-sum MLP-32 0.043 → 0.161 and modular MLP-32 0.150 → 0.205 (L1 → full). There is **no upstream arithmetic-encoding issue**: both the operands and both target forms become exactly identifiable at L=4; the probes simply fail to reach either.

## 20. Wrap / No-Wrap / Repeated

Target Bayes (all-ADD, pooled):

| History | wrap | no-wrap | repeated | isolated |
|---|---:|---:|---:|---:|
| L1 | 0.218 | 0.314 | 0.277 | 0.295 |
| L2 | 0.505 | 0.451 | 0.319 | 0.435 |
| L4 | 0.999 | 0.998 | 0.997 | 0.999 |
| Lfull | 1.000 | 1.000 | 1.000 | 1.000 |

Every subtype reaches identifiability by L=4; the wrap/no-wrap spread is at most **0.096** (L1), below the 0.15 mixed bar, and vanishes at L≥4. **No arithmetic subtype is materially harder at the task interface.**

## 21. E vs F

The raw observation history and the all-ADD test population are model-independent, so the exact Bayes curves are **identical** between E and F (all-ADD target full-prefix Bayes 1.000 for both). The learned probes are also identical (same raw inputs, same D0 training), so every `E_vs_F` MLP-32 difference is **0.000**; F reproduces the same weak-probe pattern (MLP-32 full-prefix 0.205, held-out 0.121) and the same **U-D** category. E and F agreement is explicit and `E_F_disagreement = false`. (The Class-A *subsets* differ by model — E 448/330/363/327/337, F 367/347/358/328/423 — but the interface conclusion is population-independent.)

## 22. Statistical Discipline

The five frozen E checkpoints are the independent units; F is a control, never pooled. Every learned-probe fraction is reported per seed with mean ± SD (e.g. full-prefix MLP-128 0.218 ± 0.009); the exact Bayes collision statistics are descriptive counts over the pooled test rows (never used as inferential *n*). The representation grid, capacity family and decision rule were **predeclared**; every representation and capacity is reported, including negative results; no test-based selection.

## 23. Leakage / Validity

Enforced and tested: (i) **no future observation** enters `H_L(t)` — the window ends at t by construction and is invariant to randomising every observation after t; a deliberately leaky builder using `x_{t+1}` is shown to differ; (ii) the full prefix ends exactly at t; (iii) TRAIN/VAL/TEST sequences are disjoint; (iv) held-out operand pairs never enter probe fitting; (v) TEST labels never tune λ, capacity, or the reported representation (permuting TEST labels leaves λ/train/val unchanged); (vi) the exact Bayes ceiling is computed independently of any probe prediction; (vii) the reconstructed state is never a probe feature; (viii) no latent model state enters the raw-history probes; (ix) probes never enter model inference (forward bit-identical before/after); (x) a deliberate test-fitting leakage path is included; (xi) collector arrays are bit-identical across two runs, the analysis JSON is deterministic (excluding the timestamp), and all 17 figures are byte-identical; (xii) checkpoint SHA-256 unchanged. Q/R/S/T reports remain byte-identical (baseline-hash test).

## 24. Causal Status

**Observation only.** U measures the raw task interface's exact information content and the accessibility of that content to bounded analysis-only probes. It does **not** show that the frozen model "should" be able to solve the task because the history contains the answer; it does **not** show the recurrent core "failed to remember"; it does **not** show the encoder "lost" information; and it does **not** show the task encoding causes the Class-A failure. The wording used is "the raw causal history has / does not have a unique target mapping", "the information becomes identifiable at history length L", "learned-probe accessibility stays weak", and "this establishes task-interface availability under the tested representation — it does not establish that the frozen model can exploit it". No causal mechanism has been established and no intervention is performed or licensed.

## 25. Decision Classification

**U-D — interface-identifiable but probe-limited** (E primary; F agrees).
* `Bayes_target(full prefix)` = **1.000** ≥ 0.99 → **not U-B**.
* Best full-prefix learned probe = **0.218** (< 0.50) so **not U-A**; its held-out accuracy **0.121 ≤ 0.25** and the Bayes−probe gap **0.782 ≥ 0.30** → **U-D**.
* U-C is also *satisfied* as a mechanism (`Bayes_target(L=1)` = 0.290 ≤ 0.50; the curve is non-decreasing; L\* = L4 is finite), and the pre-registered priority selects **U-D** as the headline because the dominant unresolved fact is the **0.78 Bayes-vs-probe gap**: the interface is identifiable, the bounded reader is not.
* Subtype spread ≤ 0.096 at every length and E/F categories agree → **not U-MIX**.
* Interpretation: **the necessary information exists at the raw task interface — uniquely at the full prefix and ≥0.99-identifiable from a four-step causal window — but the tested bounded probe family does not exploit it.** This is a task-interface-availability statement, **not** a claim that the frozen model can access the information and **not** a claim about what any architectural change would achieve.

## 26. What U Establishes

* A deterministic, leakage-controlled **raw-interface information map** with two independent evidence families (exact collision/Bayes + bounded learned probes) over the predeclared history grid.
* That the **current observation x_t does not determine the operands or target** (L=1 target Bayes 0.290; every L=1 history ambiguous; only 16 unique histories) but **does determine the destination** (1.000).
* That the **full causal prefix uniquely determines** the operands/target/raw-sum (Bayes 1.000, zero ambiguity) and that a **four-step causal window already yields Bayes ≥ 0.99** — the interface-identifiable history horizon **L\* = L4**.
* That the **bounded learned probes stay weak** (best full-prefix 0.218 all-ADD, ~0.09 Class-A) with held-out operand accuracy ≤ 0.25, creating a **0.78 Bayes-vs-probe gap** — classification **U-D**.
* That the deterministic generator is **exactly reconstructible** from the causal prefix (validity control).
* That no arithmetic subtype is materially harder at the interface, and **E and F agree**.

## 27. What U Does Not Establish

* It does **not** show the frozen model can (or should) exploit the interface information; **availability ≠ accessibility by the frozen network**.
* It does **not** show the recurrent core "failed to remember" or the encoder "lost" information, nor that the task encoding causes the Class-A failure.
* It does **not** prove no bounded probe could recover the information (the tested family is bounded flat probes by design); the failure is a property of the tested reader, not of mathematical absence.
* It does **not** show that the history probe's weakness is a capacity limit within the family — but it also does **not** rule out that a different, still-bounded reader (e.g. sequence-aware) could do better; that is the specified next measurement.
* It does **not** claim the target is "easy" or that the benchmark is solvable by the frozen architecture.

## 28. Final Answers

1. **Does the current observation x_t contain operand values?** **No** — `operand_a`/`operand_b` L=1 Bayes 0.437/0.444, every L=1 history ambiguous (ambiguous mass 1.000).
2. **Does x_t uniquely determine the written target?** **No** — target L=1 Bayes 0.290 (all-ADD), 0.161 (Class-A).
3. **Exact Bayes ceiling at L=1?** Target **0.290** (Class-A 0.161); operands 0.437/0.444; destination 1.000.
4. **At what history length does operand_a become identifiable?** **L=4** (Bayes 0.999).
5. **At what history length does operand_b become identifiable?** **L=4** (0.999).
6. **At what history length does the written target become identifiable?** **L=4** (0.998); 1.000 at L≥8.
7. **Is the full causal prefix uniquely sufficient?** **Yes** — Bayes 1.000, zero ambiguous mass.
8. **Do learned probes approach the exact Bayes ceiling?** **No** — 0.218 vs 1.000 (gap 0.782).
9. **Does held-out operand generalisation survive?** **No** — held-out ≤ 0.25 at every length (full-prefix 0.121).
10. **Is the target information distributed across time?** **Yes** — L* = L4; L=1 0.290 → L=4 0.998 (non-decreasing).
11. **Does raw-sum vs modular reveal anything different?** **No** — both share the same ceiling (identifiable at L=4) and both probes stay weak.
12. **Does wrap/repeated behaviour differ?** **No** — all subtypes reach Bayes ≈ 1.0 by L=4 (spread ≤ 0.096).
13. **Do E and F Class-A populations agree?** **Yes** — identical interface/probe results; `E_F_disagreement = false`.
14. **Earliest task-interface point at which operand information becomes accessible?** Exactly identifiable at **L=4**; only partially at L=1–2; **learned-probe accessible nowhere in the tested family**.
15. **Classification?** **U-D — interface-identifiable but probe-limited** (with strong history-dependence as the mechanism).
16. **Does U justify an intervention?** **No.** The information is available at the raw interface but no intervention is licensed: U is measurement-only and conservative. When raw history is identifiable but the frozen state is not (T), this establishes an **information-retention/accessibility** problem, not a repair; when the Bayes ceiling is high but probes fail, the information is **not** concluded absent.

## 29. Exactly ONE Next Experiment

U-D localises the remaining gap precisely: the target/operands are **exactly identifiable from a 4-step raw causal window** (and uniquely at the full prefix), yet the bounded **flat** probe family (linear/MLP on the flattened one-hot window) reaches only ~0.22 on all-ADD and ~0.09 on Class-A, with held-out ≤ 0.25. Per §36 (U-D), the next step is **one stronger but still controlled information-access measurement specifically targeted at the remaining probe limitation** — not an intervention.

**Experiment V — bounded sequence-aware access test (measurement-only, frozen checkpoints).** Fit a **predeclared, parameter-bounded family of small causal readers** over the raw observation stream — e.g. a fixed-width GRU and a causal 1-D convolutional reader, each capped at the R/S/T MLP-128 parameter budget — on the same TRAIN → VAL → TEST discipline, the same held-out-operand control, and the same populations as U. The question is narrow and controlled: **does the L4-identifiable task-interface information become accessible to a bounded sequence-aware reader, or is the U-D gap a flat-representation artefact?** V is analysis-only (never in the frozen model), changes no weights/architecture, uses the same splits, and begins no Phase 2. It conditions whether the interface information is *readable in principle* by a bounded sequence reader before any statement about the frozen recurrent state is warranted.

The governing loop remains **MEASURE → LOCALIZE → INTERPRET → KEEP/KILL → ONE CONTROLLED NEXT TEST**.

## Required Figures

`docs/reports/figures/experiment_u_*.png` (17): `bayes_ceiling_vs_history`, `probe_accuracy_vs_history`, `bayes_vs_probe_gap`, `operand_a_bayes_vs_history`, `operand_b_bayes_vs_history`, `target_bayes_vs_history`, `modular_vs_rawsum_vs_history`, `current_observation_collision`, `history_collision_by_L`, `heldout_operand_vs_history`, `seen_vs_heldout_gap_vs_history`, `classA_vs_correct_vs_alladd`, `wrap_nowrap_vs_history`, `repeated_isolated_vs_history`, `per_seed_probe`, `full_prefix_comparison`, `earliest_identifiable_summary`.

## Tests

`tests/test_experiment_u.py` — **30 tests**: frozen-checkpoint SHA/immutability; no-grad; exact raw observation capture; causal history construction and window-ends-at-t; no-future-observation leakage (randomised-future invariance) and a deliberate leaky-builder rejection; padding/early-prefix determinism; history-length ordering; exact history hashing/grouping; Bayes-ceiling correctness; target/operand collision detection; current-observation structural insufficiency; sequence-split separation and history-overlap accounting; probe determinism; probe train/val/test separation (λ from val only; no test tuning); held-out operand exclusion; no probe insertion into inference; exact state-reconstruction control; Class-A count reproduction (448/330/363/327/337; test totals 3199/3196/3171/3125/3072); candidate-correct control; E/F alignment (model-independent all-ADD Bayes); modular and raw-sum target generation; wrap/repeated classification; the full-prefix Bayes=1.0 result; deliberate leakage failure path; malformed-artifact rejection; JSON determinism; figure creation (≥17); historical-report immutability.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_u.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_u_task_interface.npz
PYTHONPATH=src python experiments/analyze_experiment_u.py \
    --npz experiments/experiment_u_task_interface.npz \
    --output experiments/experiment_u_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_u
PYTHONPATH=src pytest tests/test_experiment_u.py -v      # 30 tests
PYTHONPATH=src pytest tests/                             # full suite
```

Determinism: collector arrays bit-identical across two runs; analysis JSON identical (excluding the timestamp); all 17 figures byte-identical. Checkpoint SHA-256 verified before/after. No model, encoder, readout, gate, commit, or candidate weight is modified. The analyzer reuses `analyze_experiment_s`'s probe primitives verbatim.

## Scientific Guardrails

Measurement-only; observational. The report says "the raw causal history has a unique target mapping from L≈4", "the information becomes identifiable at history length L=4", "learned-probe accessibility stays weak even where the exact ceiling is 1.0", and "this establishes task-interface availability under the tested representation". It never says the model should be able to solve the task because the history contains the answer, that the recurrent core failed to remember, that the encoder lost the information, that the task encoding is sufficient for the model, or that the interface causes the failure. A failed bounded probe is **not** proof that the information is mathematically absent (here the exact ceiling proves the opposite); a high Bayes ceiling is **not** evidence the frozen network can exploit it. Q, R, S and T are **not** edited; U is appended as new evidence. No Phase 2. Negative and borderline findings (weak flat probes, held-out failure, capacity-flat probes, Class-A vs all-ADD gap) are preserved.
