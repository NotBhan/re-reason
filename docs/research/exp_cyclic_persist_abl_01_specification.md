# EXP-CYCLIC-PERSIST-ABL-01: Authoritative Experiment Specification (Persistence-Only Ablation)

> [!IMPORTANT]
> **DOCUMENT STATUS: PREREGISTERED SPECIFICATION (PI APPROVED — FROZEN PROTOCOL; STAGE 1 VERIFIED COMPLETE; STAGE 2 UNAUTHORIZED)**
> This document specifies the authoritative, preregistered research protocol for `EXP-CYCLIC-PERSIST-ABL-01`.
> Protocol approved and frozen by the Principal Investigator (PI) on 2026-10-16 using Option 1 ($\lambda_{\text{persist}} = 0.5$).
> Stage 1 execution (runner implementation, focused unit tests, and 100-update seed-42 smoke run) is **VERIFIED COMPLETE** (2026-10-16).
> Stage 2 official five-seed training remains **STRICTLY UNAUTHORIZED** pending separate explicit PI authorization following independent review of Stage 1 outcomes.

---

## 1. Metadata and Proposal Identity

* **Experiment Identifier:** `EXP-CYCLIC-PERSIST-ABL-01`
* **Controlling Specification:** Phase 1.6 Architectural Reformulation Proposal (`docs/reports/phase_1_6_architectural_reformulation_proposal.md`)
* **Primary Reference Baseline:**
  * `EXP-CYCLIC-PERSIST-01` (`docs/reports/exp_cyclic_persist_01_report.md`; `experiments/exp_cyclic_persist_01_results.json`) — Joint destination + persistence supervision ($\lambda_{\text{dest}} = 1.0, \lambda_{\text{persist}} = 0.5$; macro validation accuracy 0.9875 on `VAL-SEEN`; outcome `BEHAVIORAL_PASS_CANONICAL_FULL`).
* **Predecessor Context Baselines:**
  * `EXP-CYCLIC-ADD-01` (`docs/reports/exp_cyclic_add_01_report.md`; `experiments/exp_cyclic_add_01_results.json`) — Terminal-only supervision baseline ($acc_{\text{val}} = 0.0625$, rejection outcome).
  * `EXP-CYCLIC-STEPWISE-01` (`docs/reports/exp_cyclic_stepwise_01_report.md`; `experiments/exp_cyclic_stepwise_01_results.json`) — Destination-only intermediate supervision ($\lambda_{\text{aux}} = 1.0$, $acc_{\text{val}} = 0.6750$, inconclusive intermediate outcome driven by $R0/R1$ read asymmetry).
* **Document Status:** Preregistered Specification (Authoritative — Finalized Protocol; Stage 1 Verified Complete; Stage 2 Unauthorized)
* **Approval Date:** 2026-10-16 (Formal PI Approval — Decision A & Decision B)
* **Runner Path:** `experiments/run_exp_cyclic_persist_abl_01.py`
* **Focused Test Suite:** `tests/test_exp_cyclic_persist_abl_01.py`
* **Official Results Target Artifact (Deferred):** `experiments/exp_cyclic_persist_abl_01_results.json`
* **Smoke Test Artifact (Stage 1):** `experiments/exp_cyclic_persist_abl_01_SMOKE_ONLY_seed42.json`

---

## 2. Research Context, Scientific Questions, and Epistemic Separation

### 2.1 Empirical Facts Established by Prior Experiments
The protocol proposal for `EXP-CYCLIC-PERSIST-ABL-01` builds upon four verified, immutable empirical anchors:
1. **Fact 1 (Terminal-Only Failure):** In `EXP-CYCLIC-ADD-01`, supervising Candidate 2 (`CyclicMGUPhase16Model`) with terminal cross-entropy loss alone failed completely: macro validation accuracy on `VAL-SEEN` was $0.0625$ (near the $0.10$ random baseline; 0/5 seeds passing), meeting the preregistered rejection boundary.
2. **Fact 2 (Destination Supervision Lift and Asymmetry):** In `EXP-CYCLIC-STEPWISE-01`, auxiliary supervision on destination registers at write steps ($r0 \leftarrow a$ at step 0; $r1 \leftarrow b$ at step 1) lifted validation accuracy to $0.6750$ (+0.6125), but resulted in severe read asymmetry at terminal step 2: register $r1$ aligned strongly ($\bar{C}_{\text{read}}(r1) = 0.8368 \ge 0.80$), while unaddressed register $r0$ degraded severely ($\bar{C}_{\text{read}}(r0) = 0.2573 < 0.30$), bottlenecking the parameter-free Givens ALU ($\bar{C}_{\text{ALU}} = 0.5374$).
3. **Fact 3 (Joint Supervision Resolution):** In `EXP-CYCLIC-PERSIST-01`, adding auxiliary supervision for unaddressed-register persistence at step 1 alongside destination supervision ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{terminal}} + 1.0 \cdot \mathcal{L}_{\text{dest}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$) completely eliminated the asymmetry ($\bar{C}_{\text{read}}(r0) = 0.8879$ and $\bar{C}_{\text{read}}(r1) = 0.9303$, both $\ge 0.80$), lifted macro validation accuracy to $0.9875$ (15.8/16; 4/5 seeds at 16/16, 1 seed at 15/16), and achieved the preregistered `BEHAVIORAL_PASS_CANONICAL_FULL` outcome without altering model architecture (219,018 trainable parameters invariant).
4. **Fact 4 (Factorial Incompleteness):** While `EXP-CYCLIC-PERSIST-01` established that the joint objective $\mathcal{L}_{\text{terminal}} + \mathcal{L}_{\text{dest}} + \mathcal{L}_{\text{persist}}$ is sufficient for canonical coordinate preservation and arithmetic generalization, it did not isolate whether destination supervision is causally necessary when persistence supervision is present.

### 2.2 The $2 \times 2$ Supervision-Condition Grid
Across Phase 1.6 cyclic modular addition experiments, the auxiliary supervision space is structured by two binary conditions:
* **Condition D (Destination Supervision):** Intermediate supervision on addressed register writes ($r0$ at step 0, $r1$ at step 1).
* **Condition P (Persistence Supervision):** Intermediate supervision on unaddressed register persistence ($r0$ at step 1).

The four cells of this $2 \times 2$ grid map to distinct experiments:

| Supervision Cell | Destination ($\lambda_{\text{dest}}$) | Persistence ($\lambda_{\text{persist}}$) | Experiment Identifier | Status & Macro $acc_{\text{val}}$ |
| :--- | :--- | :--- | :--- | :--- |
| **Cell (0, 0)** | 0.0 (None) | 0.0 (None) | `EXP-CYCLIC-ADD-01` | Completed: $0.0625$ (`FAIL_REJECTION`) |
| **Cell (1, 0)** | 1.0 (Steps 0, 1) | 0.0 (None) | `EXP-CYCLIC-STEPWISE-01` | Completed: $0.6750$ (`INCONCLUSIVE_INTERMEDIATE`) |
| **Cell (1, 1)** | 1.0 (Steps 0, 1) | 0.5 (Step 1) | `EXP-CYCLIC-PERSIST-01` | Completed: $0.9875$ (`BEHAVIORAL_PASS_CANONICAL_FULL`) |
| **Cell (0, 1)** | **0.0 (None)** | **0.5 (Step 1)** | **`EXP-CYCLIC-PERSIST-ABL-01`** | **Proposed Ablation (Cell 0, 1)** |

Completing Cell (0, 1) provides the empirical data to complete the intended $2 \times 2$ supervision-condition grid. Across the shared cohort of five random seeds (42–46), this grid enables descriptive per-seed and macro-averaged contrasts between conditions. Such comparability strictly depends on holding all background experimental conditions invariant (identical model architecture, dataset partitions, batch size, optimizer hyperparameters, update counts, and evaluation cadence). These comparisons provide descriptive empirical characterizations across the matched 5-seed cohort; they do not constitute formal inferential claims of statistical significance, and no unapproved post-hoc hypothesis tests or arbitrary significance thresholds are assumed.

### 2.3 Research Questions
1. **Sufficiency of Persistence Alone:** Is unaddressed-register persistence supervision at step 1, combined solely with terminal task supervision ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{terminal}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{persist}}$), sufficient to induce canonical coordinate preservation and generalization ($acc_{\text{val}} \ge 0.950$, $\bar{C}_{\text{read}} \ge 0.80$, $\bar{C}_{\text{ALU}} \ge 0.80$)?
2. **Causal Necessity of Destination Supervision:** Or is intermediate destination supervision causally necessary to anchor the write representations in canonical coordinate space, such that omitting destination supervision causes either training divergence, representation drift, or an inverted read asymmetry?

### 2.4 Falsifiable Hypotheses
* **Hypothesis $H_{\text{persist-sufficient}}$ (Persistence Sufficiency):**
  Persistence supervision at step 1 provides sufficient inductive bias to constrain the continuous recurrent trajectory $S_1 \to S_2$. Because $r0$ is penalized if its coordinates dissolve during the $r1$ write, and terminal loss penalizes incorrect modular addition at step 2, `mlp_gate` and `mlp_write` learn to place $r1$ in orthogonal coordinate subspaces without requiring direct intermediate destination losses at steps 0 and 1.
  * *Predicted Empirical Outcome:* Macro validation accuracy $acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds (`BEHAVIOR_PROGRESSION_PASS`); $\bar{C}_{\text{read}} \ge 0.80$; $\bar{C}_{\text{ALU}} \ge 0.80$; asymmetry resolved ($\bar{C}_{\text{read}}(r0) \ge 0.80$ and $\bar{C}_{\text{read}}(r1) \ge 0.80$).
* **Hypothesis $H_{\text{dest-necessary}}$ (Destination Necessity / Joint Synergy):**
  Direct intermediate destination supervision is causally necessary to anchor the coordinate basis. In the absence of direct destination supervision at steps 0 and 1:
  * *Sub-hypothesis $H_{\text{inv-asymmetry}}$ (Inverted Read Asymmetry):* Register $r0$ receives direct supervision at step 1 (persistence loss) and survives ($\bar{C}_{\text{read}}(r0) \ge 0.80$), but register $r1$ receives zero auxiliary supervision anywhere and its representation fails to align to the canonical basis ($\bar{C}_{\text{read}}(r1) < 0.80$), inverting the asymmetry observed in `EXP-CYCLIC-STEPWISE-01`.
  * *Sub-hypothesis $H_{\text{basis-drift}}$ (Coordinate Basis Drift):* Without destination supervision at step 0 ($r0 \leftarrow a$), the network fails to ground operand $a$ in the fixed Fourier basis $\Phi$ early in the sequence, causing persistence and terminal losses to compete destructively.
  * *Predicted Empirical Outcome:* Progression gate fails ($acc_{\text{val}} < 0.950$ across seeds, or $\ge 4$ seeds intermediate/rejected); representation metrics fail canonical alignment (`REP_PARTIAL` or `REP_FAILED`).

---

## 3. Single-Intervention Definition and Frozen Invariants

### 3.1 Single Intervention Definition
The single proposed intervention in `EXP-CYCLIC-PERSIST-ABL-01` relative to `EXP-CYCLIC-PERSIST-01` is:
**The complete removal of intermediate destination supervision terms from the training loss ($\lambda_{\text{dest}} = 0$).**

The intermediate destination losses at step 0 ($\text{CE}(\text{logits}_{\text{dest}, 0}, a)$) and step 1 ($\text{CE}(\text{logits}_{\text{dest}, 1}, b)$) are excluded from backpropagation. They are recorded in telemetry for diagnostic tracking only.

### 3.2 Frozen Architectural and System Invariants
To ensure rigorous comparability across the entire experimental suite, all other components are strictly frozen:
1. **Model Architecture:** `CyclicMGUPhase16Model` (`src/re_reason/core/register_machine_models.py`).
2. **Trainable Parameter Count:** Exactly **219,018** trainable parameters across all seeds at initialization and after training:
   * `mlp_read`: 29,066 parameters (`Linear(216, 128)` + `ReLU` + `Linear(128, 10)`).
   * `mlp_write`: 52,152 parameters (`Linear(32, 212)` + `ReLU` + `Linear(212, 212)`).
   * `mlp_gate`: 137,800 parameters (`Linear(436, 212)` + `ReLU` + `Linear(212, 212)`).
3. **State Vector:** Monolithic continuous state $S \in \mathbb{R}^{212}$, initialized to $\mathbf{0}$ ($S_0 = \mathbf{0}$). No discrete structural slot partitions.
4. **Arithmetic Core:** Fixed, analytical parameter-free Givens rotation ALU with fixed 5-harmonic Fourier basis $\Phi \in \mathbb{R}^{10 \times 10}$. Zero trainable parameters in the ALU. The cyclic ALU remains fixed and analytical; this experiment does not test whether arithmetic is learned by the ALU.
5. **Optimizer:** AdamW (`torch.optim.AdamW`), constant learning rate $3.0 \times 10^{-3}$, weight decay $1.0 \times 10^{-2}$, betas $(0.9, 0.999)$, eps $1.0 \times 10^{-8}$. No warmup, no decay, no gradient clipping.
6. **Batch Size and Duration:** Batch size 64 (full `TRAIN-FIT` batch); exactly 3,000 updates.
7. **Random Seeds:** Invariant 5-seed cohort: `[42, 43, 44, 45, 46]`.
8. **Evaluation Cadence:** Evaluated every 50 updates on both full `TRAIN-FIT` (64 examples) and full `VAL-SEEN` (16 examples) splits, yielding exactly 60 checkpoints per seed (300 total across 5 seeds).
9. **Hardware Execution Target:** Host CPU (retaining parity with prior experiments).

---

## 4. Dataset Partitions and Strict Evaluation Firewall

### 4.1 Canonical 3-Instruction Program Trace
The task is 3-instruction modular addition modulo 10:
* Step 0 ($t=0$): `SET r0 a` $\implies S_0 \xrightarrow{x_0} S_1$. Destination write: $r0 \leftarrow a$.
* Step 1 ($t=1$): `SET r1 b` $\implies S_1 \xrightarrow{x_1} S_2$. Destination write: $r1 \leftarrow b$; Unaddressed persistence: $r0 \leftarrow a$.
* Step 2 ($t=2$): `ADD r0 r1` $\implies S_2 \xrightarrow{x_2} S_3$. Givens rotation of read vectors $u_a, u_b$; Terminal target: $(a + b) \bmod 10$.

### 4.2 Partitions
* **TRAIN-FIT (64 operand pairs):** Operand pairs $(a, b)$ with $a \in \{0..9\}$ and $b \in \{1, 2, 3, 4, 6, 7, 8, 9\} \setminus \text{val\_pairs}$.
* **VAL-SEEN (16 operand pairs):** Sequence-disjoint held-out operand combinations formed from seen digits:
  `[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6), (3, 4), (3, 6), (3, 7), (4, 7), (4, 8), (5, 8), (5, 9), (6, 9), (9, 1)]`
* **TEST-HELDOUT (20 operand pairs, Protected):** Operand pairs with $b \in \{0, 5\}$.

### 4.3 Strict Protected TEST Firewall
The protected `TEST-HELDOUT` split must **never** be loaded, constructed, queried, evaluated, or logged at any stage of protocol design, smoke testing, checkpoint selection, or diagnostic analysis.
* All gating decisions, checkpoint selections, and model rankings are evaluated exclusively on `VAL-SEEN`.
* The runner and analysis scripts must record `queries = 0` for the protected partition.

---

## 5. Mathematical Formulation of the Proposed Ablation Objective

### 5.1 Readout and Query Semantics
All queries utilize the existing shared `mlp_read` network decoded against the fixed Fourier basis $\Phi \in \mathbb{R}^{10 \times 10}$:
$$\text{logits}(S, e_r) = \operatorname{project}(\text{mlp\_read}([S, e_r])) \cdot \Phi^{\top} \in \mathbb{R}^{10}$$
where $\operatorname{project}(v) = v / (\|v\|_2 + \epsilon)$ for 10-dimensional output vectors.

1. **Step 0 Destination Query ($t=0$ on state $S_1$):**
   $$\text{logits}_{\text{dest}, 0} = \operatorname{project}(\text{mlp\_read}([S_1, e_{r0}])) \cdot \Phi^{\top}$$
   *Target:* $y_{\text{dest}, 0} = a$.
   *Role in Ablation:* **Telemetry only.** Not backpropagated ($\lambda_{\text{dest}, 0} = 0$).
2. **Step 1 Destination Query ($t=1$ on state $S_2$):**
   $$\text{logits}_{\text{dest}, 1} = \operatorname{project}(\text{mlp\_read}([S_2, e_{r1}])) \cdot \Phi^{\top}$$
   *Target:* $y_{\text{dest}, 1} = b$.
   *Role in Ablation:* **Telemetry only.** Not backpropagated ($\lambda_{\text{dest}, 1} = 0$).
3. **Step 1 Persistence Query ($t=1$ on state $S_2$):**
   $$\text{logits}_{\text{persist}, 1} = \operatorname{project}(\text{mlp\_read}([S_2, e_{r0}])) \cdot \Phi^{\top}$$
   *Target:* $y_{\text{persist}, 1} = a$.
   *Role in Ablation:* **Active auxiliary loss term.** Backpropagated with weight $\lambda_{\text{persist}}$.
   *Note:* The persistence query on $S_2$ uses address $e_{r0} = [1, 0, 0, 0]^{\top}$, querying the unaddressed register that was written at step 0 and was NOT addressed by the step 1 instruction `SET r1 b`.
4. **Step 2 Terminal Task Query ($t=2$ on state $S_3$):**
   $$\text{logits}_{\text{terminal}} = \operatorname{project}(\text{mlp\_read}([S_3, e_{r0}])) \cdot \Phi^{\top}$$
   *Target:* $y_{\text{terminal}} = (a + b) \bmod 10$.
   *Role in Ablation:* **Primary task loss.** Backpropagated with weight $1.0$.

### 5.2 Ablation Objective Function
$$\mathcal{L}_{\text{total}}^{\text{abl}} = \mathcal{L}_{\text{terminal}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{persist}}$$

Where:
$$\mathcal{L}_{\text{terminal}} = \text{CE}(\text{logits}_{\text{terminal}}, (a + b) \bmod 10)$$
$$\mathcal{L}_{\text{persist}} = \text{CE}(\text{logits}_{\text{persist}, 1}, a)$$
$$\mathcal{L}_{\text{dest}} = \frac{1}{2} \left[ \text{CE}(\text{logits}_{\text{dest}, 0}, a) + \text{CE}(\text{logits}_{\text{dest}, 1}, b) \right] \quad (\text{logged for telemetry; coefficient } \lambda_{\text{dest}} = 0.0)$$

### 5.3 Coefficient Selection Analysis and Recommendation
The proposal evaluates two candidate settings for $\lambda_{\text{persist}}$ prior to observing any ablation validation outcomes:

#### Option 1 (Recommended): $\lambda_{\text{persist}} = 0.5$ (Nominal Scalar Weight Invariance)
* **Mathematical Rationale:** In `EXP-CYCLIC-PERSIST-01`, expanding the objective yielded:
  $$\mathcal{L} = \mathcal{L}_{\text{terminal}} + 0.5 \cdot \text{CE}(\text{dest}_0, a) + 0.5 \cdot \text{CE}(\text{dest}_1, b) + 0.5 \cdot \text{CE}(\text{persist}_1, a)$$
  Setting $\lambda_{\text{persist}} = 0.5$ preserves the **nominal scalar weighting of the persistence-loss term relative to `EXP-CYCLIC-PERSIST-01`**.
* **Epistemic Precision on Gradient Dynamics:** Preserving the nominal scalar coefficient does *not* guarantee identical actual gradient magnitudes across training runs. Because intermediate destination supervision is removed ($\lambda_{\text{dest}} = 0$), the model's intermediate hidden states, parameter update trajectories, and backpropagation pathways will necessarily diverge from `EXP-CYCLIC-PERSIST-01`. However, preserving the nominal scalar coefficient ($0.5$) ensures that the objective-level contribution of persistence loss per assertion is held constant, rather than simultaneously modifying both the supervision topology (omitting destination loss) and the persistence loss scalar weight.
* **Scientific Merit:** This isolates the removal of destination supervision as the *sole* structural change in the objective formulation. Setting $\lambda_{\text{persist}} = 0.5$ avoids confounding the removal of destination supervision with an artificial amplification of the persistence loss scalar weight.
* **Non-Tuning Guarantee:** The coefficient recommendation is established entirely prior to observing any ablation validation results and will not be tuned against validation performance.

#### Option 2 (Alternative Contingency): $\lambda_{\text{persist}} = 1.0$ (Unit Auxiliary Weighting)
* **Mathematical Rationale:** In `EXP-CYCLIC-STEPWISE-01`, the total auxiliary weight was $\lambda_{\text{aux}} = 1.0$. Setting $\lambda_{\text{persist}} = 1.0$ makes the total auxiliary weight equal to the terminal loss weight ($1.0 : 1.0$).
* **Scientific Limitation:** Setting $\lambda_{\text{persist}} = 1.0$ doubles the nominal scalar weight of the persistence term ($1.0$ vs $0.5$) compared to `EXP-CYCLIC-PERSIST-01`. If the ablation were to succeed with $\lambda=1.0$, one could not determine whether success required doubling the nominal persistence weighting or whether persistence alone at the original weight was sufficient.

**Proposal Recommendation:** Recommend **Option 1 ($\lambda_{\text{persist}} = 0.5$)** for PI approval. Present Option 2 as a formal alternative for PI review.

---

## 6. Checkpoint Selection Policy and Comparability Analysis

### 6.1 Checkpoint Selection Rule
To preserve fidelity to standard empirical practice without biasing the selection policy:
* **Proposed Selection Criterion:** Minimum total validation loss under the native ablation objective:
  $$\mathcal{L}_{\text{val, total}}^{\text{abl}} = \mathcal{L}_{\text{val, terminal}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{val, persist}}$$
* **Tie-Breaking Rules (Invariant):**
  1. Ties broken by highest validation accuracy ($acc_{\text{val}}$ on `VAL-SEEN`).
  2. Further ties broken by earliest update step.

### 6.2 Cross-Experiment Comparability and Dual Telemetry Contract
* **Optimization Objective Divergence:** Because $\mathcal{L}_{\text{val, total}}^{\text{abl}}$ omits $\lambda_{\text{dest}} \cdot \mathcal{L}_{\text{dest}}$, the numerical value of the ablation validation loss cannot be directly compared to the validation loss of `EXP-CYCLIC-PERSIST-01` ($\mathcal{L}_{\text{val, total}} = \mathcal{L}_{\text{term}} + 1.0 \cdot \mathcal{L}_{\text{dest}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$).
* **Dual Telemetry Specification:** To ensure complete cross-experiment observability, the runner will record both native and composite loss metrics across training and validation splits:
  1. **Native Ablation Validation Loss (`val_loss_total`):** $\mathcal{L}_{\text{val, terminal}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{val, persist}}$, used strictly for official checkpoint selection.
  2. **Composite Reference Validation Loss (`val_loss_composite_reference`):** $\mathcal{L}_{\text{val, terminal}} + 1.0 \cdot \mathcal{L}_{\text{val, dest}} + 0.5 \cdot \mathcal{L}_{\text{val, persist}}$, matching the primary experiment's objective formula. Required at all 60 validation checkpoints.
  3. **Composite Reference Training Loss (`train_loss_composite_reference`):** $\mathcal{L}_{\text{train, terminal}} + 1.0 \cdot \mathcal{L}_{\text{train, dest}} + 0.5 \cdot \mathcal{L}_{\text{train, persist}}$, serialized at all 60 training checkpoints for split symmetry.
* **Strict Selection Isolation:** Official checkpoint selection is strictly governed by `val_loss_total` (native ablation objective). The composite reference loss is recorded purely as an unselected diagnostic telemetry field. It is **never** used to select checkpoints, rank models, or govern progression gates.
* **Accuracy Comparability:** Validation accuracy ($acc_{\text{val}}$ out of 16), read alignment ($\bar{C}_{\text{read}}$), ALU alignment ($\bar{C}_{\text{ALU}}$), and persistence alignment ($\bar{C}_{\text{persist}}$) are completely objective-independent and directly comparable across all experiments.

---

## 7. Diagnostic Representation Metrics and Reconciled Telemetry Contract

All diagnostic metrics are computed at every checkpoint following the exact mathematical specifications of `EXP-CYCLIC-PERSIST-01`:

1. **Register-Specific Read Alignment at Step 2 ($C_{\text{read}}(r0), C_{\text{read}}(r1), C_{\text{read}}$):**
   $$C_{\text{read}}(r) = \frac{1}{N} \sum_{i=1}^N \frac{u_{\text{read}, i}^{(r)} \cdot \Phi(v_{\text{true}, i}^{(r)})}{\|u_{\text{read}, i}^{(r)}\|_2 \|\Phi(v_{\text{true}, i}^{(r)})\|_2 + \epsilon}, \quad C_{\text{read}} = \frac{C_{\text{read}}(r0) + C_{\text{read}}(r1)}{2}$$
2. **Asymmetry Resolution Criterion (Diagnostic Gate):**
   The $R0/R1$ read asymmetry is formally resolved if and only if:
   $$\bar{C}_{\text{read}}(r0) \ge 0.80 \quad \text{AND} \quad \bar{C}_{\text{read}}(r1) \ge 0.80$$
3. **Step 1 Persistence Read Alignment ($C_{\text{persist}}(r0)$):**
   Cosine similarity between unaddressed readout on $S_2$ with $e_{r0}$ and canonical codeword $\Phi(a)$.
4. **Cross-Context Read Consistency ($S_{\text{context}}$):**
   Mean pairwise cosine similarity of read vectors representing the same digit across different operand roles and contexts at step 2.
5. **ALU Output Phase Alignment ($C_{\text{ALU}}$):**
   Cosine similarity between raw Givens rotation output $v_{\text{ALU}}$ and canonical sum codeword $\Phi((a+b) \bmod 10)$.
6. **Destination Read Alignment ($C_{\text{dest}}$):**
   Cosine similarities between post-update states and destination targets at steps 0, 1, and 2.
7. **Step-Wise Accuracies:**
   * Step 0 destination accuracy: $\operatorname{argmax}(\text{logits}_{\text{dest}, 0}) == a$.
   * Step 1 destination accuracy: $\operatorname{argmax}(\text{logits}_{\text{dest}, 1}) == b$.
   * Step 1 persistence accuracy: $\operatorname{argmax}(\text{logits}_{\text{persist}, 1}) == a$.
   * Step 2 terminal accuracy: $\operatorname{argmax}(\text{logits}_{\text{terminal}}) == (a + b) \bmod 10$.
8. **Loss Telemetry Serialized at Every Checkpoint (Reconciled Contract):**
   Every checkpoint in the serialized artifact must record exactly the following 12 loss and accuracy fields:
   * `train_loss_total`: Native ablation training loss ($\mathcal{L}_{\text{train, terminal}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{train, persist}}$).
   * `train_loss_terminal`: Terminal training cross-entropy.
   * `train_loss_dest`: Destination training cross-entropy (diagnostic, unweighted in backward pass).
   * `train_loss_persist`: Persistence training cross-entropy.
   * `train_loss_composite_reference`: Diagnostic reference matching the primary experiment formula ($\mathcal{L}_{\text{train, term}} + 1.0 \cdot \mathcal{L}_{\text{train, dest}} + 0.5 \cdot \mathcal{L}_{\text{train, persist}}$).
   * `train_acc`: Task training accuracy on `TRAIN-FIT` (64 examples).
   * `val_loss_total`: Native ablation validation loss ($\mathcal{L}_{\text{val, terminal}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{val, persist}}$), used for official checkpoint selection.
   * `val_loss_terminal`: Terminal validation cross-entropy.
   * `val_loss_dest`: Destination validation cross-entropy (diagnostic, unweighted in backward pass).
   * `val_loss_persist`: Persistence validation cross-entropy.
   * `val_loss_composite_reference`: Diagnostic reference matching the primary experiment formula ($\mathcal{L}_{\text{val, term}} + 1.0 \cdot \mathcal{L}_{\text{val, dest}} + 0.5 \cdot \mathcal{L}_{\text{val, persist}}$).
   * `val_acc`: Task validation accuracy on `VAL-SEEN` (16 examples).
9. **Numerical Invariants:**
   All accuracies bounded in $[0.0, 1.0]$; all cosine similarities bounded in $[-1.0, 1.0]$; all loss terms finite and non-negative; parameter gradients verified non-None, non-zero, and non-NaN across `mlp_read`, `mlp_write`, and `mlp_gate`.

---

## 8. Proposed Comparison Gates and Decision Rules

### 8.1 Per-Seed Behavioral Gates and 5-Seed Aggregation
Evaluated over the 16 validation pairs ($N_{\text{val}} = 16$) across the 5 official seeds:
* For each seed $s$, let $k_s \in \{0, \dots, 16\}$ denote the number of correct validation predictions at its selected checkpoint, and $acc_s = k_s / 16$.
* **Per-Seed Classification:**
  * **Pass seed:** $acc_s \ge 0.950 \iff k_s = 16$ ($acc_s = 1.0000$).
  * **Reject seed:** $acc_s < 0.250 \iff k_s \le 3$ ($acc_s \le 0.1875$).
  * **Intermediate seed:** $0.250 \le acc_s < 0.950 \iff 4 \le k_s \le 15$ ($0.2500 \le acc_s \le 0.9375$).
* **Five-Seed Aggregate Categories:**
  1. `BEHAVIOR_PROGRESSION_PASS`: $N_{\text{pass}} \ge 4$ AND $N_{\text{reject}} = 0$.
  2. `BEHAVIOR_REJECTION`: $N_{\text{reject}} \ge 4$.
  3. `BEHAVIOR_INTERMEDIATE`: All other 5-seed distributions.

### 8.2 Representation Categories
Macro-averaged metrics across selected checkpoints:
1. `REP_CANONICAL_FULL`: $\bar{C}_{\text{read}} \ge 0.80$ AND $\bar{S}_{\text{context}} \ge 0.80$ AND $\bar{C}_{\text{ALU}} \ge 0.80$.
2. `REP_CANONICAL_INPUT_ONLY`: $\bar{C}_{\text{read}} \ge 0.80$ AND $\bar{S}_{\text{context}} \ge 0.80$ AND $\bar{C}_{\text{ALU}} < 0.80$.
3. `REP_FAILED`: $\bar{C}_{\text{read}} < 0.30$ AND $\bar{C}_{\text{ALU}} < 0.30$.
4. `REP_PARTIAL`: All other metric triples.

### 8.3 Proposed Exhaustive 3 × 4 Decision Matrix (Ablation-Specific Epistemic Mapping)

| Behavioral State ($N_{\text{val}}=16$) | Representation State | Scientific Category (`enum`) | Proposed Epistemic Meaning for Ablation |
| :--- | :--- | :--- | :--- |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_CANONICAL_FULL` | **`BEHAVIORAL_PASS_CANONICAL_FULL`** | **Persistence alone is sufficient.** Destination supervision is not causally necessary; persistence supervision alone provides adequate inductive constraint to preserve $r0$ and enable full arithmetic generalization. |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_CANONICAL_INPUT_ONLY` | **`BEHAVIORAL_PASS_CANONICAL_INPUT_ONLY`** | **Input alignment without raw ALU alignment.** Operands align canonically in registers, but raw ALU output is sub-threshold; progression achieved via non-linear read/write adaptation. |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_PARTIAL` | **`BEHAVIORAL_PASS_PARTIAL_ALIGNMENT`** | **Behavioral success with partial alignment.** Progression pass met, but representations are only partially canonical. |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_FAILED` | **`BEHAVIORAL_PASS_UNALIGNED`** | **Unaligned progression.** Behavioral success without canonical alignment; disconfirms canonical routing hypothesis. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_CANONICAL_FULL` | **`INTERMEDIATE_CANONICAL_FULL`** | **Aligned representation, intermediate behavior.** Canonical representations form, but sequence execution does not achieve 5-seed consensus. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_CANONICAL_INPUT_ONLY` | **`INTERMEDIATE_CANONICAL_INPUT_ONLY`** | **Input alignment, intermediate behavior.** Operands align, but ALU output degrades and sequence behavior fails progression gate. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_PARTIAL` | **`INCONCLUSIVE_INTERMEDIATE`** | **Inconclusive outcome.** Omitting destination supervision degrades performance relative to `EXP-CYCLIC-PERSIST-01`; persistence alone is partially helpful but insufficient for reliable progression. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_FAILED` | **`INCONCLUSIVE_UNALIGNED`** | **Inconclusive unaligned.** Intermediate behavior with unaligned representations. |
| **`BEHAVIOR_REJECTION`** | `REP_CANONICAL_FULL` | **`REJECTION_CANONICAL_FULL_DECOUPLED`** | **Decoupled representations.** Representations align, but sequence behavior fails completely. |
| **`BEHAVIOR_REJECTION`** | `REP_CANONICAL_INPUT_ONLY` | **`REJECTION_CANONICAL_INPUT_DECOUPLED`** | **Decoupled input alignment.** Input reads align, but sequence behavior fails completely. |
| **`BEHAVIOR_REJECTION`** | `REP_PARTIAL` | **`FALSIFIED_PARTIAL_ALIGNMENT`** | **Destination supervision causally necessary.** Rejection boundary reached; persistence alone cannot maintain coordinate stability without destination write anchoring. |
| **`BEHAVIOR_REJECTION`** | `REP_FAILED` | **`FALSIFIED_FULL`** | **Full rejection of persistence sufficiency.** Without destination supervision, both representations and task behavior collapse to near-chance baseline. |
| **ANY** (Numerical Failure) | **ANY** | **`IMPLEMENTATION_INVALID`** | **Execution invalid.** Numerical check failure, NaN/Inf, non-finite loss/gradients, or test failure. |

---

## 9. Protocol Analysis: Causal Isolation, Optimization Confounders, and Interpretive Limits

### 9.1 Descriptive Factorial Contrasts across the Shared 5-Seed Cohort
The $2 \times 2$ grid enables descriptive paired contrasts across the shared 5-seed cohort (`[42, 43, 44, 45, 46]`):
1. **Descriptive Contrast 1 (Destination Contribution under Persistence):** Comparing `EXP-CYCLIC-PERSIST-01` (Cell 1, 1) with `EXP-CYCLIC-PERSIST-ABL-01` (Cell 0, 1) characterizes the descriptive change in validation accuracy and alignment when destination supervision is omitted while persistence supervision is present.
2. **Descriptive Contrast 2 (Persistence Contribution without Destination):** Comparing `EXP-CYCLIC-PERSIST-ABL-01` (Cell 0, 1) with `EXP-CYCLIC-ADD-01` (Cell 0, 0) characterizes the descriptive change when persistence supervision is added to terminal-only training.
3. **Descriptive Contrast 3 (Persistence vs Destination Supervision):** Comparing `EXP-CYCLIC-PERSIST-ABL-01` (Cell 0, 1) with `EXP-CYCLIC-STEPWISE-01` (Cell 1, 0) contrasts supervising unaddressed persistence ($r0$ at step 1) versus supervising addressed destination writes ($r0$ at step 0, $r1$ at step 1).

*Epistemic Limitation on Statistical Inference:* These comparisons represent descriptive empirical contrasts within a matched 5-seed design under identical protocol conditions. With $N=5$ seeds, they do not constitute formal inferential claims of statistical significance, nor do they establish asymptotic distributional properties. Any formal hypothesis test or inferential analysis would require separate protocol formulation and explicit PI approval.

### 9.2 Optimization Differences and Potential Confounders
Omitting destination supervision introduces three critical optimization shifts that must be accounted for during scientific interpretation:
1. **Step 0 Gradient Starvation (Credit-Assignment Horizon Lengthening):**
   * In `EXP-CYCLIC-STEPWISE-01` and `EXP-CYCLIC-PERSIST-01`, step 0 received a direct gradient signal through $\text{CE}(\text{logits}_{\text{dest}, 0}, a)$. The transition $S_0 \xrightarrow{x_0} S_1$ was directly anchored to place operand $a$ into $r0$ coordinates.
   * In `EXP-CYCLIC-PERSIST-ABL-01`, step 0 receives **zero immediate gradient signal**. The gradient reaching $S_1$ must backpropagate through:
     (a) Step 1 persistence loss (a 2-transition path: $S_0 \to S_1 \to S_2$), and
     (b) Step 2 terminal task loss (a 3-transition path: $S_0 \to S_1 \to S_2 \to S_3$).
   * *Confound:* If the ablation fails, is it because persistence supervision is fundamentally incapable of preserving unaddressed registers, or because the network struggled with the longer credit-assignment horizon to discover the initial $r0$ encoding at step 0?
2. **Register 1 ($b$) Asymmetry / Inversion Risk:**
   * In `EXP-CYCLIC-PERSIST-ABL-01`, register $r0$ receives direct auxiliary supervision at step 1 (persistence loss), but register $r1$ receives **zero auxiliary supervision anywhere in the sequence**.
   * Register $r1$ is written at step 1 and must be read at step 2 to perform modular addition, but its write is supervised only through the composite non-linear Givens rotation and terminal classification loss at step 2.
   * *Proposed Diagnostic Hypothesis:* This asymmetry could induce an **inverted read asymmetry**, where $\bar{C}_{\text{read}}(r0) \ge 0.80$ while $\bar{C}_{\text{read}}(r1) < 0.80$. The telemetry contract explicitly tracks both per-register alignments to test this hypothesis.
3. **Total Auxiliary Gradient Magnitude Reduction:**
   * Under Option 1 ($\lambda_{\text{persist}} = 0.5$), the nominal auxiliary loss weight drops from $1.5$ in `EXP-CYCLIC-PERSIST-01` ($1.0 \times \mathcal{L}_{\text{dest}} + 0.5 \times \mathcal{L}_{\text{persist}}$) to $0.5$.
   * Terminal loss ($\lambda_{\text{terminal}} = 1.0$) will exert a proportionally higher influence on early gradient updates ($1.0 : 0.5$ ratio vs $1.0 : 1.5$ ratio).

### 9.3 Interpretive Boundaries and Epistemic Guardrails
1. **No Arithmetic Learning Claim:** Candidate 2 uses a fixed, parameter-free cyclic Givens rotation ALU. Neither a positive nor a negative outcome in this ablation reflects whether the ALU "learned" arithmetic from data. The experiment strictly evaluates whether routing and memory persistence networks learn coordinate preservation under persistence-only supervision.
2. **Domain Scope (`VAL-SEEN` vs `TEST-HELDOUT`):** Validation metrics evaluate generalization to unseen combinations of seen training digits. A passing outcome establishes in-distribution sequence generalization; it does **not** evaluate out-of-distribution generalization to the firewalled digits $b \in \{0, 5\}$.
3. **Telemetric Limitations:** All claims regarding loss trajectories must be grounded in recorded checkpoint telemetry. Training loss dynamics between checkpoints must not be asserted as verified facts.

---

## 10. Compute and Verification Budget

* **Platform:** Host CPU execution (retaining parity with prior experiments).
* **Reference Workload:** In `EXP-CYCLIC-PERSIST-01`, total wall time across 5 seeds was 3 min 14 s (approx 194 seconds; per seed: 32.56, 30.32, 29.60, 33.38, 54.06 s), representing approximately 180 seconds of aggregate active computation.
* **Stage 1 Smoke Test Budget (Deferred):**
  * 1 seed (Seed 42) $\times$ 100 updates $\approx$ **1.5 seconds** CPU runtime.
  * Storage footprint: $\approx 25$ KB (`experiments/exp_cyclic_persist_abl_01_SMOKE_ONLY_seed42.json`).
* **Stage 2 Official Five-Seed Budget (Deferred):**
  * 5 seeds $\times$ 3,000 updates = 15,000 updates.
  * With destination cross-entropy computation omitted from backward passes (retained for forward diagnostic telemetry only), per-seed CPU runtime is estimated at approximately **30–36 seconds**.
  * Total aggregate CPU workload across all 5 seeds is estimated at approximately **170–190 seconds** (total wall clock $\approx 3.0 - 3.5$ minutes on host CPU).
  * *Engineering Disclaimer:* This runtime figure is an engineering estimate based on comparable workloads, not a guarantee; actual wall times are subject to host CPU frequency scaling, background process activity, and OS thread scheduling.
  * Storage footprint: $\approx 1.6$ MB (`experiments/exp_cyclic_persist_abl_01_results.json`).
* **Mandatory Pre-Execution Verification Battery (for future Stage 1):**
  1. Focused unit tests: `pytest tests/test_exp_cyclic_persist_abl_01.py -v`.
  2. Documentation governance: `pytest tests/test_documentation_governance.py -v`.
  3. Pre-flight parameter count assertion: exactly 219,018 trainable parameters.
  4. Pre-flight firewall assertions: `assert_protected_domain_absent()` and `assert_canonical_trace()`.
  5. Loss decomposition assertion: $\mathcal{L}_{\text{total}}^{\text{abl}} == \mathcal{L}_{\text{terminal}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{persist}}$.

---

## 11. Authorization Status and Governance Gates

### 11.1 Current Governance Status
* **Document Status:** PREREGISTERED SPECIFICATION (AUTHORITATIVE PROTOCOL).
* **Protocol Freeze Status:** **FROZEN & APPROVED (2026-10-16, PI Decision A).**
  * Frozen Objective: $\mathcal{L}_{\text{total}}^{\text{abl}} = \mathcal{L}_{\text{terminal}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$ ($\lambda_{\text{persist}} = 0.5$, Option 1).
  * Destination loss terms excluded from backward pass ($\lambda_{\text{dest}} = 0.0$); recorded for diagnostic telemetry only.
* **Stage 1 (Implementation, Focused Tests, Smoke Test):** **VERIFIED COMPLETE (2026-10-16).**
  * Runner implemented: `experiments/run_exp_cyclic_persist_abl_01.py`.
  * Focused unit tests implemented: `tests/test_exp_cyclic_persist_abl_01.py` (26/26 passing in 0.41s).
  * 100-update seed-42 smoke run executed in 0.96s CPU runtime.
  * Smoke artifact serialized: `experiments/exp_cyclic_persist_abl_01_SMOKE_ONLY_seed42.json`.
  * Invariants verified: 219,018 trainable parameters, exact loss decomposition ($\mathcal{L}_{\text{total}}^{\text{abl}} = \mathcal{L}_{\text{terminal}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$), finite gradients to `mlp_gate`, `mlp_read`, `mlp_write`, protected test firewall intact (`queries = 0`).
* **Stage 2 (Official Five-Seed Training):** **STRICTLY UNAUTHORIZED (PI Decision C).**
  * Full 3,000-update 5-seed training across seeds 42–46 remains on hold pending independent review of Stage 1 results.
* **Protected TEST Partition ($b \in \{0, 5\}$):** **STRICTLY FIREWALLED (`queries = 0`).**

### 11.2 Record of PI Decisions
1. **Decision A (Protocol Approval & Coefficient Freeze — 2026-10-16):** **APPROVED.** Option 1 ($\lambda_{\text{persist}} = 0.5$) selected and frozen to preserve nominal scalar weighting invariance relative to `EXP-CYCLIC-PERSIST-01`.
2. **Decision B (Stage 1 Execution Authorization — 2026-10-16):** **AUTHORIZED.** Implementation of runner, test suite, and seed-42 smoke test authorized.
3. **Decision C (Stage 2 Official Run Authorization):** **NOT AUTHORIZED.** Requires separate PI authorization following Stage 1 audit.

### 11.3 Stage 1 Execution and Verification Outcome (2026-10-16)
Stage 1 execution was conducted under explicit PI Decision B and completed with all verification checks satisfied:
1. **Runner Implementation:** `experiments/run_exp_cyclic_persist_abl_01.py` faithfully implements the frozen objective $\mathcal{L}_{\text{total}}^{\text{abl}} = \mathcal{L}_{\text{terminal}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$, excludes destination losses from backpropagation ($\lambda_{\text{dest}} = 0.0$), serializes `val_loss_composite_reference` and `train_loss_composite_reference` as unselected diagnostic reference fields, and enforces that checkpoint selection is strictly governed by native ablation validation loss (`val_loss_total`).
2. **Focused Test Suite:** `tests/test_exp_cyclic_persist_abl_01.py` defines 26 deterministic tests covering loss decomposition, zero destination gradient contribution, parameter-count invariance (219,018), gradient flow to all blocks, representation metrics, and strict evaluation firewalls. All 26 tests passed in 0.41s.
3. **Smoke Run Telemetry:** Executed 100 updates on seed 42 in 0.96s CPU runtime. Artifact serialized to `experiments/exp_cyclic_persist_abl_01_SMOKE_ONLY_seed42.json`.
   - Update 50: `train_loss_total = 2.2460`, `val_loss_total = 3.5583` (`val_loss_composite_reference = 5.1459`), `val_acc = 0.0625`.
   - Update 100: `train_loss_total = 0.4761`, `val_loss_total = 2.5969` (`val_loss_composite_reference = 3.6822`), `val_acc = 0.3125`.
   - Selected checkpoint: update 100 (`val_loss_total = 2.5969`, `val_acc = 0.3125`).
   - Trainable parameters: exactly 219,018 before and after training.
   - Gradient checks: all trainable blocks (`mlp_gate`, `mlp_read`, `mlp_write`) received finite, non-zero gradients.
   - Protected test firewall: `queries = 0`.
4. **Epistemic Classification:** The smoke artifact is strictly an implementation and instrumentation validation check; it does not evaluate or falsify the scientific hypothesis. Stage 2 remains strictly unauthorized.
