# Phase 1 Evidence Synthesis and Formal Go/No-Go Review — RE:Reason

**Date:** 2026-10-13  
**Status:** Canonical Phase 1 Synthesis Report (consolidating Experiments U through AH within the broader Phase 1 investigation).  
**Subject:** Formal empirical synthesis, reconciliation of evidence, and two separate formal governance decisions:
1. **Decision A (Empirical Cycle Closure):** GO (Close Phase 1 diagnostic cycle).
2. **Decision B (Post-Phase-1 Action):** NO-GO on demonstrated arithmetic rule generalisation; NO-GO on core architecture / gating / candidate / commit / readout interventions; NO-GO on immediate Phase 2 transition.

---

## 1. Executive Decision

This document delivers the formal Phase 1 empirical synthesis for the RE:Reason project, consolidating the investigation across Experiments U through AH in the context of the full Phase 1 sequence (A–T). RE:Reason is an empirical research project evaluating a candidate recurrent inference architecture with bounded convex state updates. The synthesis separates measurement from causal inference, reconciles apparent tensions between empirical task identifiability and bounded-learner accessibility, audits documentation and test governance, and establishes two distinct decisions.

### Decision A — Phase 1 Empirical Cycle Closure: GO

**The Phase 1 empirical cycle is formally CLOSED.**

* **Rationale:** The pre-registered diagnostic questions for Phase 1 have been systematically bounded across thirty-four conceptual experiments (A through AH), documented in twenty-eight formal letter-designated experiment reports (Experiments G through AH, preceded by eleven foundational architecture, feasibility, diagnostic, and ablation reports covering Models A through F, comprising thirty-nine prior completed research reports in the repository baseline).
* **Saturation:** Continued external bounded-reader diagnostics under the existing task interface, data regime, and optimization protocol have reached empirical saturation on the sampled benchmark histories. Across data volume sweeps (W), optimization budget expansions (X), instrument qualifications (Y–AC), optimizer recipe adjustments (AD), fit-transfer decompositions (AE), task difficulty stratifications (AF), training-pool resampling (AG), and reader capacity ladders (AH), the central finding remains consistent: external bounded sequence readers fit the observed training distribution via sample memorization but fail to achieve held-out-operand rule transfer (balanced accuracy 0.072–0.143 vs. 0.10 chance; 0/5 seeds achieving the +0.15 material bar).
* **Sufficiency:** The empirical cycle BUILD → MEASURE → ABLATE → INTERPRET → KEEP OR KILL has completed its cycle for Phase 1. Within the scope of the frozen Model E checkpoints, tested reader families, and fixed protocol, the pre-registered diagnostic questions have been answered. Further external reader sweeps within this specific parameter envelope and training paradigm provide diminishing returns and do not justify keeping the Phase 1 diagnostic cycle open.

### Decision B — Post-Phase-1 Direction: NO-GO

**NO-GO on Demonstrated Arithmetic Rule Generalisation; NO-GO on Core Architecture / Gating / Readout / Candidate Interventions; NO-GO on Immediate Phase 2 Transition.**

* **No demonstrated arithmetic rule generalisation:** The frozen RE:Reason model (`NonlinearCandidateMGUModel`, candidate/local objective Model E) achieves strong overall register tracking accuracy on short horizons (T=10 Exact Match 0.755 ± 0.019; T=20 Exact Match 0.407 ± 0.023; per-register linear decodability ~78%). However, on arithmetic transitions where the candidate network makes an error (Class-A error-conditioned test ADD transitions), candidate accuracy is 0.000 by construction and linear probe recoverability from $C_t$ is weak (26.7%). On repeated-ADD chains, terminal exact match collapses to ~0.15 under all objectives. The frozen model does not possess demonstrated generalizable modular arithmetic capability across held-out operands or extended arithmetic chains.
* **No ad-hoc model intervention is justified:** The sequence of interventional ablations (Experiments J, K, and O) demonstrates that modifying gate magnitudes (`f' = clip(gamma * f, 0, 1)`), sharpening gate distributions (`kappa=2`), or subtracting readout-aligned parity components from candidate vectors produces at best narrow, non-specific local transition rescues while consistently degrading sequence-level exact match (e.g., -10.4 pp EM in Experiment O). The failure is not an isolated gate dilution, not an artifact of linear readout bias (L showed removing bias flips only 1.2% to correct), and not localized to candidate output-projection geometry (S showed no decodability drop across candidate internal layers). Slicing or tweaking frozen submodules without a validated representational foundation produces negative transfer.
* **No immediate Phase 2 transition:** Roadmap Phase 2 machinery (parallel hypothesis branching, learned verifiers, transactional rollback, adaptive halting, recursion) was conceived under the premise that the underlying persistent core reliably produces valid candidate computations that merely require selection or multi-hypothesis pursuit. However, earlier feasibility studies (`branching_feasibility_oracle.md` and `verifier_feasibility.md`) proved that verifier skill is indistinguishable from deterministic argmax selection and that branching gains are largely matched-noise artifacts. Introducing speculative multi-path mechanisms over an unverified recurrent arithmetic core would compound error propagation rather than resolve representational failure.
* **Exactly one recommended next action:** Formulate a separate, standalone research proposal for architectural reformulation (Phase 1.6), evaluating fundamental sequence-to-arithmetic inductive biases (such as explicit symbolic binding, modular representation geometries, or alternative recurrence formulations trained from scratch) before writing or modifying any model code.

---

## 2. Scope and Evidence Provenance

### Investigation Scope

This review synthesizes the empirical evidence established across the entire Phase 1 investigation, with primary analytical focus on the external bounded-reader access program (Experiments U through AH).

```
Phase 1 Diagnostic Progression:
  [A–F]  Foundational Architecture & Objectives
         └── Model E (Bounded Convex MGU) adopted; Candidate/Local supervision (E) locked.
  [G–K]  Commit Gate & Transition Dynamics
         └── Coordinate-wise gating matters; scalar proxies fail; commit modifications neutral/harmful.
  [L–O]  Latent State Representation Geometry & Parity
         └── Latent misplacement dominates (L); modular parity geometry (M, N); parity intervention non-causal (O).
  [P–T]  Error Budgeting & Upstream Pipeline Probing
         └── Earliest errors are candidate-generation (P); C_t weakly recoverable (Q, R); internal activations
             no better than C_t (S); encoder output structurally operand-blind (T).
  [U–AH] Task Interface Identifiability & Bounded Sequence-Reader Program
         └── Raw history carries target at L*=4 (U); bounded sequence readers fail transfer (V); data scale (W)
             and update budget (X) do not fix transfer; instrument qualification (Y–AC) enables fit but not
             generalization; qualified re-evaluation (AD, AE) confirms persistent fit without transfer;
             raw held-out target is identifiable (AF); training-pool balancing (AG) and capacity ladder (AH)
             scale fitting without yielding rule transfer.
```

### Frozen Artifacts and Provenance

All findings across U–AH are grounded in immutable, hash-verified artifacts:

* **Frozen Checkpoints:**
  * Official Candidate/Local Model E: `experiments/checkpoints/E_candidate_seed42_3000steps.pt` through `seed46` (5 seeds).
  * Control Combined Objective Model F: `experiments/checkpoints/F_combined_seed42_3000steps.pt` through `seed46` (5 seeds).
  * SHA-256 hashes are recorded in `experiments/experiment_g_transitions_meta.json` and verified prior to each measurement run.
* **Data Splits:**
  * `DisjointSplitManager`: TRAIN (6,000 sequences), VAL (1,200 sequences), TEST (1,200 sequences x 10 streams, `90042+seed`). Strict sequence-level disjointness enforced with zero sequence overlap and no label leakage.
* **Historical Baseline Anchor:**
  * `docs/reports/HISTORICAL_BASELINE.json`: Enforces SHA-256 immutability of all 39 prior completed reports (and 40 total reports upon registration of this Phase 1 synthesis).
  * Verified by `tests/test_documentation_governance.py`.

---

## 3. U–AH Evidence Matrix

The table below summarizes the fourteen experiments comprising the external bounded-reader program (U–AH), categorised by research question, experimental object, primary empirical findings, pre-registered decisions, supported conclusions, and boundaries.

| Exp | Research Question | Experimental Object | Frozen Controls & Data | Primary Numerical Result | Decision | What Evidence Supports | What It Does NOT Establish | Impact on Phase 1 Decision |
|---|---|---|---|---|---|---|---|---|
| **U** | Is the arithmetic target identifiable from raw observation history $x_0..x_t$ without latent state? | Raw history identifiability vs flat learned probes | Frozen E/F test masks; raw one-hot observations (21-dim) + validity bit | Bayes ceiling: L1=0.290, L2=0.402, L4=0.998, full=1.000. Best full-prefix probe: all-ADD 0.218, Class-A ~0.09. Held-out <= 0.25. Bayes-probe gap 0.782. | **U-D** | On sampled benchmark evaluation streams, target is empirically identifiable from a 4-step raw window ($L^*=4$, Bayes >= 0.998; full prefix 1.000). Current observation is operand-blind (16 unique histories, collision rate 1.0). | Does not establish that a practical neural learner will discover or generalize the modular rule; does not prove latent state lacks information; does not guarantee identifiability under arbitrary unseen lengths. | Reframes failure: missing information is not at the task interface. |
| **V** | Does explicit temporal structure (causal GRU, causal TCN within 29,578 params) close the U-D gap? | Externally trained causal sequence readers | Budget matched to U MLP-128 (GRU h=85, TCN c=52). Disjoint splits; 20 held-out pairs ($b \in \{0,5\}$). | Class-A test accuracy: GRU 0.083 (L4) / 0.080 (full); TCN 0.112 (L4) / 0.109 (full) vs flat 0.089. Held-out <= 0.25. Bayes-reader gap 0.887. | **V-B** | Explicit temporal structure alone does not resolve weak accessibility under a bounded parameter budget. Readers decode `op_dest` (1.000) and operands partially (0.21–0.25) but not modular target. | Does not establish that sequence readers are universally incapable or that RE:Reason representation is defective. | Demonstrates that sequence inductive bias alone does not solve the arithmetic extraction problem. |
| **W** | Does scaling training data (1,500 -> 4,000 -> 10,000 transitions) under a fixed 200-update budget rescue readers? | Externally trained analysis readers across data scale | Unique, pairwise sequence-disjoint i.i.d. pools; fixed 200 updates; identical architectures. | Class-A best reader: 0.107 -> 0.116 (+0.033 gain, sub-material). MLP-128 declines (0.089 -> 0.070). Training accuracy 0.13–0.23 (underfitting). Bayes-reader gap 0.883. | **W-B** | Weak accessibility is not relieved by a 6.7x data increase under fixed 200 full-batch updates. Readers underfit the training data. | Does not exclude benefits of larger data under different optimization regimes; left open fixed-budget vs capacity limit. | Rules out data volume as the primary limiting lever under the tested training protocol. |
| **X** | Does a 5x larger update budget (200 -> 1,000 updates) on fixed 10,000-example pool improve accessibility? | Externally trained analysis readers across update budget | Fixed W 10k pool; Adam (lr 3e-3, wd 1e-2); eval every 10 updates; VAL-selected checkpoints. | Best Class-A: 0.111 (1,000) vs 0.116 (200). Paired deltas near zero. Training accuracy stays at 0.07–0.15 for sequence readers (near chance). | **X-B** | Increasing update iterations 5x does not improve test or held-out accuracy. Sequence readers fail to fit the training set itself. | Does not explain why readers fail to fit; whether it is an optimization artifact or structural representation hurdle. | Sharpens the problem: readers cannot even optimize the training sample under default Adam. |
| **Y** | Can readers fit a small, class-balanced 256-transition sample? (Fitting vs generalization split) | Fitting control on 256-example subset | Selection seed 20261010; held-out pairs excluded; 1,000 updates; VAL/TEST canonical. | GRU fits training set to 0.999 (L4) / 0.992 (full) by update ~492; TEST stays near chance (0.096–0.105). TCN fails to fit even 256 rows (train acc 0.102, loss ln 10). | **Y-MIX** | Splits the failure: GRU suffers from generalization / task structure (fits small pool via memorization, zero transfer); TCN suffers from an optimization instrument defect. | GRU small-sample fit does not establish rule learning. TCN non-fitting is not proven to be target-specific. | Isolates instrument confound: TCN cannot be used for ADD conclusions until qualified. |
| **Z** | Can TCN fit a directly observable target (`op_dest` = argmax(x_t[3:7])) on Y's subset? | Instrument qualification on 4-class direct control | Reuses Y 256 subset; GRU positive control; `op_dest` verified 100% readable from x_t. | GRU fits perfectly (train 1.000, test 1.000). TCN fails to fit (max train 0.287, test 0.251, chance 0.25). | **Z-B** | TCN non-fitting is an instrument qualification defect, not an ADD-specific representation problem. Confound identified. | Does not prove intrinsic architectural invalidity of TCN; does not reflect on RE:Reason. | Formally disqualifies unqualified TCN as a measurement instrument for ADD representation claims. |
| **AA** | Does sweeping LR {3e-4..3e-2} x updates {1k, 5k} qualify TCN on `op_dest`? | Implementation audit + hyperparameter grid | 200 fits; audit verified causal padding, final-step path, finite gradients. | Audit passes. GRU fits at every recipe (1.000). TCN never reaches 0.90 (mean 0.256–0.311 across entire grid). | **AA-B** | TCN non-fitting is not an implementation bug (indexing/gradient flow) nor an LR/budget tuning issue under Adam/default init. | Does not establish intrinsic parameterization limit (other optimizer families untested). | Narrows instrument investigation to optimizer family and weight initialization. |
| **AB** | Do alternative optimizers (AdamW, SGD, RMSprop) or initialization (xavier) qualify TCN on `op_dest`? | Optimizer-family & initialization diagnostic | Fixed lr 3e-3, 5,000 updates; matched init per seed; trainable preflight passed. | AdamW drives TCN to 1.000 train (5/5 seeds, test 0.987–0.996). Adam+xavier drives TCN to 1.000 (5/5). Adam_ref reproduces AA (0.2867). | **AB-A** | Z-B/AA-B non-fitting was an optimizer/initialization artifact under Adam/default, not an intrinsic parameterization limit. TCN qualified under AdamW. | Does not isolate weight decay vs init scale mechanism; does not establish ADD learnability. | Re-qualifies TCN as a valid instrument for sequence reading when trained with AdamW. |
| **AC** | Does the AB-qualified recipe enable TCN to fit ADD `y_true` on Y's 256-example subset? | Re-qualification on ADD target | Y 256 subset; AdamW and Adam+xavier vs Adam_ref; preflight passed; y_true derivation exact. | AdamW drives TCN to 1.000 train (5/5 seeds) at both L4 and Lfull. Adam+xavier fits Lfull (1.000) but split at L4 (2/5). TEST accuracy at chance (0.103–0.125). Trainer-identity bridge against fresh Y trainer is exact (`max |Δ| = 0.0`), while stored-record comparison exhibits thread-sensitive drift up to 0.091/0.102 on intermediate milestones. | **AC-A** (fit) / **fit_without_generalization** | TCN non-fitting on ADD was an optimizer artifact; with AdamW it fits. However, small-subset fit is pure memorization with zero transfer to held-out test. | Small-subset ADD fit does not establish arithmetic rule acquisition. | Proves readers can fit the modular label, but fitting is entirely disconnected from arithmetic generalization. |
| **AD** | Does re-running V's full-scale access test under AdamW change target access or held-out generalization? | Full-scale qualified access test | V design fixed (ordinary TRAIN 1500, VAL 800, TEST canonical); 240 fits; AdamW vs Adam_ref bridge. | Largest mean paired delta is +0.0165 (far below +0.15 bar). AdamW lifts training fit (0.44–0.86) without improving test access (0.09–0.14). Reference bridge (`adam_ref`) reproduces V almost exactly (max Class-A drift 0.0080, held-out drift 0.0000). | **AD-B** (V-B unchanged) | V's sequence-limited accessibility verdict (V-B) was NOT an optimizer artifact. AdamW produces fit-without-transfer at full scale. | Observational with respect to RE:Reason; does not show whether high-train solutions contain partial structure. | Confirms V-B weak accessibility is robust to optimizer recipe at full training scale. |
| **AE** | Do high-train-accuracy AdamW solutions contain partial transferable structure or pure pool memorization? | Baseline-aware fit-transfer diagnostic | Exact deterministic re-evaluation of AD AdamW fits; balanced accuracy, per-class recall, training-derived majority baseline. | Train accuracy 0.44–0.86 far exceeds majority baseline (0.285). Held-out Class-A accuracy (0.089–0.136) at/below baseline (0.116). Balanced accuracy near chance (0.093–0.114). Metrics match AD retained metrics exactly (`max |Δ| = 0.0`). | **Inconclusive_mixed** (no partial transfer) | Readers fit the pool without acquiring the general arithmetic rule. Apparent positive raw margins are seed/pair inconsistent noise. | Does not determine whether task-side label geometry/difficulty causes held-out failure. | Establishes fit-without-transfer as the dominant, robust empirical regime of the readers. |
| **AF** | Do held-out operand pairs ($b \in \{0,5\}$) suffer from an interface-level identifiability loss or label-geometry skew? | Task difficulty & label geometry stratification | U exact-interface Bayes machinery on V dataset; stratified across b=0/5, wrap, a<5, a==b. | Every held-out stratum maintains ~1.0 raw-interface target ceiling (0.9991–1.0000). Largest subgroup gap <= 0. Class-0 skew observed (21.2% held-out vs 8.4% seen). | **task_side_limitation_not_indicated** | Held-out reader failure is NOT caused by loss of task identifiability at raw interface. Skew is explained by $(0,0)$ and $(5,5)$ both mapping to 0. | Does not prove sparse held-out pairs (2–10 rows/seed) are practically learnable. | Eliminates task-side identifiability loss as an explanation for held-out generalization failure. |
| **AG** | Does training-pool sampling (natural, pair-balanced, target-balanced bootstrap) resolve the fit-transfer gap? | Training-pool coverage & generalization stress | 200 fits across 4 pool constructions; held-out pairs excluded; matched model init. | Held-out balanced accuracy stays at chance across all pools (0.077–0.119). Training fit is sampling-sensitive only for seeds that fit (2/5 seeds never fit). | **Inconclusive_mixed** (persistent fit without transfer) | Re-balancing pair coverage or target distribution changes fit for fitting seeds but produces zero held-out transfer. | Does not test reader capacity sensitivity. | Demonstrates that dataset distribution skew is not the root cause of transfer failure. |
| **AH** | Does scaling reader capacity over a 5-level width ladder (0.29x to 3.71x baseline) produce rule generalization? | Bounded-reader capacity ladder | 100 fits; GRU (h=42..170) & TCN (c=26..104); native eligible pool; AdamW protocol; V source arrays. | Capacity scales training fit (TCN Lfull 0.374 -> 0.770; GRU L4 0.306 -> 0.484). Held-out balanced accuracy stays at chance (0.072–0.143; 0/5 seeds reach +0.15 gain). Baseline level reproduces AE bit-exactly (`max |Δ| = 0.00e+00`). | **H2_fit_without_transfer (AH-B)** | H1 falsified for tested ladder/protocol. Capacity scales memorization, not rule generalization. V-A not reached; V-B unchanged (Bayes-reader gap 0.895). | Does not show capacity cannot solve task under different objectives/architectures; observational; initialization-seed stride reuse across levels noted as methodological constraint. | Closes the reader-side diagnostic program: capacity within this family expands memorization without inducing transfer. |

---

## 4. Cross-Experiment Synthesis by Research Question

### 4.1 Frozen-Model Task Performance and Earliest Observed Error Types

Phase 1 established that the frozen RE:Reason model (`NonlinearCandidateMGUModel`, Model E) achieves a test Exact Match of 0.755 ± 0.019 at T=10 under candidate/local supervision, but degrades to 0.407 ± 0.023 at T=20 zero-shot. On ordinary test addition transitions, the frozen candidate head frequently succeeds. However, when evaluating the specific subset of active arithmetic transitions where the candidate network makes an error (the **Class-A error-conditioned test population**), candidate accuracy is 0.000 by construction.

In Experiment P, a fine-grained sequence-level first-error budget revealed that when sequences fail at T=10:
* **59.7% (E) / 59.4% (F)** of failures originate as **candidate-generation errors (Class A)** on written registers.
* **32.5% (E) / 31.8% (F)** originate as **candidate-to-commit residuals (Class B)**.
* **8.0% (E) / 8.7% (F)** originate as **terminal readout residuals (Class C)**.
* **Untouched-register corruption is 0.0%**.

At T=20, candidate-generation errors expand to **67.2% / 67.6%**. The earliest observed failure along the temporal execution trace is overwhelmingly localized to candidate generation during arithmetic operations (almost entirely ADD transitions, split between modular wrap-around and non-wrap additions). Crucially, this first-error attribution is an empirical descriptive measurement of failure onset, not a causal proof that the candidate generator is an independent, repairable failure module decoupled from upstream representations or recurrent dynamics.

### 4.2 Candidate-Generation Errors vs. State Persistence and Register Corruption

Passive state persistence is remarkably robust in RE:Reason: unwritten registers remain clean across the entire sequence, and linear probes decode unwritten register values with ~78% accuracy at t=10. The system does not suffer from passive register drift or uncoordinated state corruption. Instead, descriptive measurement localization across intermediate stages and representations reveals that failures concentrate during active updates:
1. **Candidate State ($C_t$):** Experiments Q, R, and S established that when the frozen candidate head fails on ADD transitions (Class A), the true written value is only weakly recoverable from $C_t$ (linear probe 26.7%, nonlinear probe 23.3%). Probing the internal activations of the candidate network (S) demonstrated that recoverability weakly increases toward the output (`cand_input` 0.246 -> `h1_post` 0.427 -> `C_t` 0.463); no internal stage outperforms $C_t$. The loss is not localized to an output projection defect.
2. **Upstream Representations ($T$):** Experiment T showed that the input representation from the encoder is structurally operand-blind at step $t$, lacking immediate value bits prior to recurrence.
3. **Commit State ($S_t$):** Experiments G, H, and I demonstrated that even when the candidate is correct, ~5.5% of ADD transitions fail to commit the correct value. Experiment L proved that this residual is primarily latent-state misplacement (98.8%), not linear readout bias (removing the bias flips only 1.2% to correct). Experiments M and N demonstrated that the committed state lands in a systematic same-parity modular basin (73.7% preserve target parity), which is driven by state-to-label alignment (91.1%) rather than readout geometry.
4. **Interventional Falsification:** Experiments J and K demonstrated that scaling or sharpening the gate vector fails to rescue sequence EM. Experiment O demonstrated that interventional subtraction of the parity component from the candidate vector is non-specific (matched by an orthogonal control) and catastrophic to sequence accuracy (-10.4 pp EM). These findings confirm that localized stage measurements describe where representations degrade, but do not imply that isolated architectural patches can restore arithmetic function.

### 4.3 Training Objectives and Candidate-Local Supervision Effects

The foundational Phase 1 investigations demonstrated that supervision structure is a first-order determinant of performance:
* Terminal-only supervision (Model A) suffered massive credit assignment failure, yielding only 0.540 EM at T=10.
* Step-wise state supervision improved EM, but candidate-local supervision (Model E, $L_E = CE(y_T) + \beta \sum CE(head(C_t), y_{t,r})$ with $\beta=1$) achieved the strongest performance (0.755 EM).
* However, Model F (combined terminal + stepwise + candidate supervision) achieved 0.737 EM, failing to improve upon Model E and failing to close the candidate-to-commit residual gap. Model E was locked as the official objective.
* Crucially, candidate-local supervision successfully trained the candidate vector to act as a delta patch, but failed to instill modular wrap-around logic or general arithmetic computation.

### 4.4 Information Availability at the Raw Task Interface

Experiment U answered the fundamental upstream question: is the modular arithmetic target mathematically present in the input observation stream?
* The current observation $x_t$ is strictly operand-blind: at step $t$, the one-hot observation encodes only the operation (ADD) and destination/source register indices ($r_a, r_b$). It contains zero immediate value bits. Target Bayes accuracy at $L=1$ is 0.290 (all-ADD) and 0.161 (Class-A), with a collision rate of 1.000 across 16 unique histories.
* However, across causal history windows on the sampled benchmark evaluation streams, target Bayes accuracy climbs monotonically: $L=1 (0.290) \to L=2 (0.402) \to L=4 (0.998) \to \text{full prefix } (1.000)$.
* By $L^* = 4$, collision rate drops to 0.016 and ambiguous mass to 0.004. At the full prefix, the causal history uniquely determines the target with zero ambiguity (Bayes 1.000) within the evaluated sequence distribution.
* Experiment AF further proved that this raw-interface identifiability holds uniformly across every held-out operand stratum ($b=0$, $b=5$, wrap, non-wrap, $a < 5$, $a \ge 5$, $a = b$, $a \ne b$), with ceilings between 0.9991 and 1.0000.
* **Scope and Distinction:** This result establishes mathematical identifiability of the target within the benchmark generator's sampled histories; it demonstrates that failure does not stem from information absence at the task interface. However, raw-history target identifiability must be strictly distinguished from practical neural learnability. Proving that $P(Y \mid X_{0..t})$ is near-deterministic does not guarantee that a gradient-based neural network trained on continuous sequence representations will discover, represent, or generalize the underlying modular arithmetic rule.

### 4.5 Bounded Readers' Fitting Ability vs. Held-Out Generalisation

Across Experiments V through AH, externally trained analysis readers (causal GRUs and causal TCNs) were evaluated to determine whether explicit sequence processing could extract the arithmetic target from the raw observation history:
* **Initial Observation (V):** Sequence readers matched to U's parameter budget (29,578 parameters) remained near chance on Class-A test targets (0.080–0.112), leaving an empirical Bayes-reader gap of 0.887.
* **Fitting vs. Transfer Separation (Y, AC, AD, AE):** When evaluated on small (256-example) or full-pool training sets under qualified AdamW optimization, readers achieved high training fit:
  * GRU training accuracy reached 0.999 on 256 examples (Y) and 0.858 on full pools (AD/AE).
  * TCN training accuracy reached 1.000 on 256 examples (AC) and 0.655 on full pools (AD/AE).
* **Persistent Lack of Held-Out Generalisation:** Despite high training fit, test accuracy on held-out operand pairs ($b \in \{0,5\}$) remained near chance:
  * In Experiment AE, held-out Class-A balanced accuracy spanned 0.093–0.114 (chance 0.10), sitting at or below training-derived majority-class baselines (0.116).
  * In Experiment AH, held-out balanced accuracy spanned 0.072–0.143 across all widths (chance 0.10).
  * Across every condition and width, 0/5 seeds demonstrated material held-out rule transfer (+0.15 gain).
* **Scope and Boundaries:**
  1. *Model Family and Protocol Scope:* This finding is strictly scoped to the tested model families (causal GRUs and causal TCNs up to 105k parameters) trained with the specified Adam/AdamW protocols and optimization budgets on the sampled benchmark datasets. It does not prove that sequence models in general are universally incapable of learning modular addition under all conceivable architectures, loss formulations, or curricula.
  2. *External Reader vs. Internal Latent State:* External sequence readers are separate analysis models trained on raw inputs. While they prove that sequence inductive bias alone is insufficient to extract modular rules from raw histories under empirical risk minimization, their failure cannot be cited as causal proof that RE:Reason's internal latent space mathematically lacks information.
  3. *Near-Chance vs. Literal Chance:* The observed performance is near-chance (balanced accuracy ~0.07–0.14 around 0.10 chance), reflecting empirical failure to exceed baseline rather than a mathematical identity to exact chance.

### 4.6 Data Volume, Optimization, Initialization, Sampling, and Capacity

Systematic experiments investigated every standard deep learning lever to determine if the fit-transfer gap was an experimental artifact:
1. **Data Volume (W):** Expanding unique training transitions from 1,500 to 10,000 (6.7x) produced a sub-material gain of +0.033 in the best reader and left the Bayes-reader gap at 0.883.
2. **Optimization Budget (X):** Expanding training updates from 200 to 1,000 (5x) yielded zero improvement (paired deltas -0.006 to +0.004).
3. **Optimization Algorithm & Initialization (Z, AA, AB):** Discovered that the TCN's total failure to fit was an optimizer artifact of standard Adam with default initialization; switching to decoupled weight decay (AdamW) or Xavier initialization drove training accuracy to 1.000 on observable controls (`op_dest`) and the ADD target. However, fixing the optimizer instrument improved training fit without improving generalization (AC, AD).
4. **Training-Pool Resampling (AG):** Evaluating natural bootstrap, pair-balanced bootstrap, and target-balanced bootstrap pools demonstrated that uniformizing operand pairs or target class distributions influenced training fit for fitting seeds, but left held-out balanced accuracy at chance (0.077–0.119).
5. **Reader Capacity Ladder (AH):** Scaling reader parameters across a pre-registered 5-level ladder (0.29x to 3.71x baseline, within a 4.0x cap) markedly increased training fit (TCN Lfull reaching 0.770 ± 0.334), but held-out balanced accuracy remained locked at chance (0.072–0.143). Hypothesis H1 was formally falsified.

---

## 5. Reconciled Findings and Apparent Tensions

A rigorous synthesis requires directly reconciling the primary empirical tensions identified across the campaign:

### Reconciliation 1: Raw-History Identifiability (U, AF) vs. Learner Generalisation (V–AH)
* **Apparent Tension:** Experiment U proved that a 4-step raw history window determines the arithmetic target with Bayes probability 0.998 (and full prefix 1.000), and Experiment AF proved this holds across every held-out operand stratum. Yet every learned flat probe and sequence reader (V–AH) remained locked at near-chance accuracy on held-out operands (~0.10).
* **Reconciliation:** Information identifiability is a mathematical property of the joint distribution $P(Y \mid X_{0..t})$ under the benchmark generator; it establishes that the causal observation history contains sufficient information to determine the arithmetic target. However, mathematical identifiability does not imply practical learnability or appropriate inductive bias. Standard sequence models (GRUs, TCNs, MLPs) minimize empirical risk over continuous representations. In the absence of an explicit variable-binding mechanism or modular arithmetic inductive bias, gradient descent in standard parameter spaces fits the training set by memorizing specific one-hot trajectory co-occurrences rather than discovering the algorithmic rule $(a + b) \pmod{10}$. Identifiability establishes the information ceiling; it does not guarantee that a learner will acquire the generating rule.

### Reconciliation 2: High Training Fit vs. Zero Held-Out Transfer (AC, AD, AE, AH)
* **Apparent Tension:** Under AdamW, readers drive training accuracy far above baseline (up to 1.000 on subsets, up to 0.86 on full pools, and up to 0.77 on wide capacity ladders), yet test accuracy on held-out operand pairs ($b \in \{0,5\}$) fails to exceed chance.
* **Reconciliation:** This is the classical signature of *interpolation without rule induction* (Power et al., 2022). The readers treat the 22-dimensional observation history as a high-dimensional pattern matching task. Because the training pool covers 80 of the 100 possible operand pairs, the network constructs an overparameterized lookup table that correctly classifies seen pairs but contains zero algorithmic structure for held-out combinations. The fit-transfer gap is not an artifact of underfitting or weak optimization; it is consistent with the behavior of empirical risk minimization on unstructured sequence representations without modular inductive bias. While this explains how overparameterized networks construct lookup tables over seen operand pairs without modular generalization, ERM's role is framed here as a supported empirical interpretation and prominent hypothesis, rather than a proven causal necessity for all optimization regimes.

### Reconciliation 3: The Role of Capacity in Experiment AH
* **Boundaries of the AH Finding:**
  1. Hypothesis H1 (Capacity-associated generalisation) was falsified **specifically for the pre-registered 5-level capacity ladder and fixed AdamW training protocol**, not for every conceivable architecture or infinite capacity.
  2. The descriptive outcome is **H2 (Fit without transfer)** / AH-B.
  3. Experiment V's sequence accessibility criterion **V-A was not reached**, and **V-B remains unchanged**.
  4. The result demonstrates that within the tested parameter envelope (up to 105k parameters, ~3.7x baseline), capacity expands memorization capacity (training fit rising 0.37 -> 0.77) without inducing rule transfer.
  5. The disclosed initialization-seed reuse limitation (per-level seed offsets reused the per-seed stride `(level_idx - 2) * 100000`, causing some `(seed, level)` combinations across different levels to share an initialization seed while each level's five seeds remained distinct) is preserved as a recorded methodological constraint. Per-seed paired deltas confirm that 0 out of 5 seeds exhibited transfer at any width, and this limitation is not claimed to be immaterial beyond what the recorded data justify.
  6. The finding characterizes external analysis readers; it cannot be cited as direct evidence that RE:Reason's internal latent space lacks information.

### Reconciliation 4: Class-Balanced vs. Raw Accuracy Metrics
* **Measurement Precision:** In Experiments AE, AF, AG, and AH, raw accuracy on held-out operand rows occasionally fluctuated between 0.08 and 0.16. In Experiment AF, task-side stratification proved that held-out operand rows exhibit significant class-0 skew (~21.2% in Class-A held-out vs 8.4% in seen), caused by equal-operand pairs $(0,0)$ and $(5,5)$ both mapping to modular target 0. When evaluated with class-balanced accuracy and per-class recall, held-out accuracy collapsed uniformly to chance (0.072–0.119 in AG; 0.072–0.143 in AH). Preserving the distinction between raw and class-balanced accuracy prevents misinterpreting label-geometry artifacts as partial rule acquisition.

---

## 6. Supported Claims, Unsupported Claims, and Unresolved Questions

### What the Evidence Strictly SUPPORTS

1. **State Persistence without Register Corruption:** RE:Reason's bounded convex update maintains unwritten register states cleanly over 10 to 20 timesteps without passive register drift.
2. **Localized Failure Regime:** Errors are predominantly concentrated in active arithmetic updates (ADD transitions), particularly modular wrap-around and multi-step addition chains.
3. **Complete Raw-Interface Information:** The raw causal observation stream contains mathematically sufficient information ($L^* = 4$, Bayes >= 0.998; full prefix Bayes 1.000) to recover operand values and modular targets across all strata.
4. **Failure of Standard Sequence Readers:** Externally trained causal GRUs and TCNs, parameter-matched to U's baseline, fail to extract the modular arithmetic rule from raw history and fail to generalize to held-out operand combinations.
5. **Robustness of the Fit-Transfer Gap:** The failure to transfer is not an artifact of training data volume (W), optimization budget (X), optimizer algorithm / initialization (AB, AD), dataset sampling / balance (AG), or reader capacity within a 3.7x parameter ladder (AH).
6. **Ineffectiveness of Ad-Hoc Interventions:** Scalar gate scaling, gate distribution sharpening, and latent parity subtraction fail to produce sequence-level capability improvements.

### What the Evidence Strictly DOES NOT SUPPORT

1. **No Demonstrated Modular Rule Generalisation:** The evidence does *not* support any claim that the frozen RE:Reason model possesses general modular arithmetic capability.
2. **No Proof of Latent State Emptiness:** Weak external reader performance does *not* prove that RE:Reason's latent state mathematically lacks all arithmetic information; external readers are separate analysis models.
3. **No Proof of Universal Sequence Inability:** AH does *not* prove that no neural network can ever learn modular arithmetic from sequence observations; it proves that the tested GRU/TCN architectures do not generalize under the tested protocol.
4. **No Single Locus of Failure:** The empirical record does *not* support localizing the failure to an isolated "bad gate," "biased readout," or "defective candidate head."
5. **No Justification for Architectural Tweaks:** The evidence does *not* justify tweaking gate constants, adding LayerNorm, or altering loss weights in the frozen model.
6. **No Justification for Phase 2 Deployment:** The evidence does *not* support activating branching, verifiers, or recursion over the current core.

### Fundamental Unresolved Questions

1. **Variable Binding Inductive Bias:** How can a recurrent network maintain explicit, disentangled bindings between register IDs (addresses) and scalar values without tensor-product representations or symbolic routing?
2. **Modular Representation Geometry:** Does gradient-based learning on sequential register operations inherently favor continuous interpolation over cyclical modular arithmetic topologies?
3. **Alternative Training Formulations:** Would learning modular arithmetic require pre-training on isolated arithmetic primitives before end-to-end register tracking, or dense step-level algebraic supervision?

---

## 7. Phase 1 Closure Decision (Decision A)

### Formal Decision: GO (Close Phase 1 Empirical Cycle)

The Phase 1 empirical cycle is formally closed.

### Detailed Justification

1. **Hypothesis Bounding Complete:** Every primary architectural, mechanistic, and diagnostic question pre-registered for Phase 1 has been executed and evaluated against declared criteria.
2. **Diminishing Returns on External Reader Diagnostics:** Experiments U through AH systematically tested flat probes, sequence architectures, data scale, optimization schedules, optimizer families, initialization regimes, training-pool constructions, and capacity scaling. All converged on the same empirical reality: persistent fit without transfer under empirical risk minimization. Within the pre-registered Phase 1 scope and the tested class of external reader diagnostic sweeps, returns have diminished, and additional sweeps within this parameter envelope and training paradigm will yield no new insight into RE:Reason's core representation. (This characterizes the exhausted reader-side diagnostic program, without asserting that no hypothetical diagnostic could ever exist.)
3. **Historical Record Preserved:** The empirical trajectory from Model A through Model E, through interventional ablations G–O, error budgeting P, and reader diagnostics U–AH, spans thirty-four conceptual experiments (A through AH) documented in twenty-eight letter-designated experiment reports (G through AH) preceded by eleven foundational architecture, feasibility, diagnostic, and ablation reports covering Models A through F, comprising 39 prior completed reports registered in `docs/reports/HISTORICAL_BASELINE.json` (and 40 total registered reports upon registration of this synthesis), alongside a complete chronological ledger (`docs/RESEARCH_LOG.md`) and a validated status synthesis (`docs/RESEARCH_STATUS.md`).
4. **Governance Criteria Satisfied:** All historical reports match their SHA-256 baselines, regression tests pass (with the documented Experiment L floating-point tolerance exclusion noted), and no active research run is in progress.

---

## 8. Post-Phase-1 Go/No-Go Decision (Decision B)

### Formal Decision: NO-GO on Core Intervention, Model Modification, or Immediate Phase 2

### Detailed Justification

1. **No Demonstrated Generalisation:** The frozen model cannot perform modular wrap-around arithmetic reliably. Deploying it as an algorithmic reasoning core is unwarranted.
2. **Interventions are Destructive:** Experiments J, K, and O conclusively showed that heuristic interventions on the existing weights or gates break sequence-level exact match.
3. **Phase 2 Prerequisites are Unmet:** Phase 2 mechanisms (branching, verifiers, recursion) require a baseline generator that reliably samples near-correct solutions. In RE:Reason, candidate-generation errors represent fundamental representational gaps, not stochastic near-misses that a verifier or branching search can filter.
4. **Requirement for Ground-Up Reformulation:** Any further investigation cannot be an incremental patch to RE:Reason. It must be a new, separately specified research program (Phase 1.6 / architectural reformulation) addressing the core representation of modular arithmetic and variable binding from scratch.

---

## 9. Reproducibility and Governance Audit

### Historical Report Immutability Audit

* All 39 prior completed experiment reports under `docs/reports/` were verified against `docs/reports/HISTORICAL_BASELINE.json`.
* Every report matches its registered SHA-256 hash bit-for-bit.
* Zero unauthorized edits, deletions, or historical retrofits were detected.

### Verification of Reproducibility Bridges & Limitations

To ensure rigorous auditability, reproducibility claims across Experiments AC, AD, AE, and AH are explicitly delineated by reference, protocol, and numerical discrepancy:

1. **Experiment AC (Trainer Identity Bridge vs. Stored Milestone Drift):**
   * *Protocol:* Re-evaluates Y's 256-example subset under `adam_ref` (Adam 3e-3, weight decay 1e-2, default initialization) for 1,000 updates.
   * *Trainer Identity:* AC's `adam_ref` trajectory equals a freshly executed run of Y's own trainer bit-for-bit (`max |Δ| = 0.0` across every seed, reader, and history).
   * *Stored Record Comparison:* Comparing AC's `adam_ref` against Y's historical stored milestone JSON record exhibited intermediate cross-environment numerical drift of up to `max |Δ| = 0.091` (train) and `0.102` (val), localized to PyTorch CPU GRU multi-threading sensitivity (8-thread vs. 1-thread execution) and near-chance TCN validation argmax instability. AC establishes exact trainer identity, not bit-exact reproduction of Y's stored milestone JSON.
2. **Experiment AD (Reference Bridge vs. Stored V Results):**
   * *Protocol:* Re-evaluates V's full-scale protocol (1,500 train, 800 val, 15,763 test) with an `adam_ref` reference bridge condition on byte-identical copies of V's source arrays.
   * *Reference & Drift:* `adam_ref` closely reproduces V's stored test metrics: max Class-A drift is **0.0080** (in TCN/L8) and max held-out drift is **0.0000**, with all other cells $\le 0.0006$. This confirms that V's weak accessibility verdict was not an optimizer artifact, while establishing a close reference bridge rather than identical floating-point trajectories.
3. **Experiment AE (Deterministic Re-evaluation of AD AdamW Fits):**
   * *Protocol:* Deterministically re-executes the exact AdamW fits using AD's trainer (`ab.train_ab`), V's exact source arrays, and V's reader-seed convention (AD retained best-on-VAL scalar metrics but no state dictionaries).
   * *Metric Reproduction:* AE's reproduced `adamw` metrics match AD's retained `adamw` metrics **exactly** (`max |Δ| = 0.0` for train accuracy, Class-A test accuracy, and held-out Class-A test accuracy).
4. **Experiment AH (Baseline Level Bridge vs. AE & Disclosed Limitation):**
   * *Protocol:* The Baseline level (Level 2 of the 5-level capacity ladder, matching AE's parameter count: 28,655 / 28,506) re-runs AE's exact heldout-variant fit under the same AdamW recipe on the same eligible training pool with level offset 0.
   * *Numerical Reproduction:* Matches AE's stored fits **bit-exactly** (`max |Δ| = 0.00e+00` for train fit, Class-A, held-out Class-A, and best-val accuracy), with identical init hashes and parameter counts.
   * *Methodological Constraint Disclosed:* Per-level initialization-seed offsets reused the per-seed stride (`(level_idx - 2) * 100000`), meaning some `(seed, level)` combinations across different ladder levels shared an initialization seed, although within each level the five seeds remained distinct. This limitation is preserved as a recorded methodological constraint and is not claimed to be immaterial beyond what the recorded data justify.

### Test Governance & Regression Audit

* **Documentation Governance Suite:** `pytest tests/test_documentation_governance.py -v` passes 12 of 12 tests (100% pass rate).
* **Focused U–AH Test Suites:** Tests covering Experiments U, V, W, X, Y, Z, AA, AB, AC, AD, AE, AF, AG, and AH all pass cleanly.
* **Broad Regression Suite Audit:**
  * Full regression runs across the repository pass across all test modules when `tests/test_experiment_l.py` is excluded.
  * **Audit Finding on Experiment L (`tests/test_experiment_l.py`):** In `test_stored_artifact_is_reproducible_from_checkpoints`, a floating-point tolerance check (`assert np.allclose(t.scalars[seed_mask], fresh["scalars"], atol=1e-4)`) fails on row-by-row comparisons due to slight numerical drift across BLAS/PyTorch runtime environments on CPU (e.g. differences on the order of $10^{-4}$ across 3,199 rows x 64 scalar columns).
  * Per documentation policy and project instructions, this is reported transparently as a **passing regression run with `test_experiment_l` excluded**, not as an unqualified full-suite pass. This tolerance issue is an existing numerical reproducibility note from Experiment L and is strictly independent of the Experiment AH or Phase 1 synthesis results.

### Working Tree and Code Integrity

* **Exact Git Status:**
  * Untracked: `docs/reports/phase_1_evidence_synthesis.md`
  * Modified (unstaged in working tree): `docs/reports/HISTORICAL_BASELINE.json`, `docs/RESEARCH_LOG.md`, `docs/RESEARCH_STATUS.md`
  * Staged: 0 files. Committed: 0 files.
* **Code & Checkpoint Invariants:**
  * Zero RE:Reason architecture files (`src/re_reason/*`) were modified.
  * Zero model checkpoints were modified or overwritten.
  * Zero historical experiment result arrays or JSON files were altered.

---

## 10. Recommended Next Action

Per project governance, exactly **ONE** next action is recommended:

### Recommended Next Action

**Draft a formal, pre-registered Research Proposal for Phase 1.6 (Architectural Reformulation for Algorithmic Binding and Modular Arithmetic).**

* **Scope:** Specify a ground-up investigation evaluating how neural architectures can natively represent cyclical modular arithmetic and dynamic address-value variable binding under sequence supervision.
* **Constraints:** No code modifications to RE:Reason, no model retraining, and no Phase 2 mechanism implementation shall occur until Phase 1.6 is formally specified, peer-reviewed, and pre-registered with explicit falsification criteria.
* **Alternative:** If resources or project priorities do not justify a ground-up architectural reformulation, the project should be formally archived at Phase 1 with the findings published as a definitive negative result on bounded recurrent register machines.
