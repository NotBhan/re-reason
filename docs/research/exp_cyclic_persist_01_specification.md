# EXP-CYCLIC-PERSIST-01: Authoritative Experiment Specification

## 1. Metadata and Preregistration Identity

* **Experiment Identifier:** `EXP-CYCLIC-PERSIST-01`
* **Controlling Proposal:** Phase 1.6 Architectural Reformulation Proposal (`docs/reports/phase_1_6_architectural_reformulation_proposal.md`)
* **Predecessor Baselines:**
  * `EXP-CYCLIC-ADD-01` (`docs/reports/exp_cyclic_add_01_report.md`; `experiments/exp_cyclic_add_01_results.json`) — Terminal-only supervision baseline ($acc_{\text{val}} = 0.0625$, rejection outcome).
  * `EXP-CYCLIC-STEPWISE-01` (`docs/reports/exp_cyclic_stepwise_01_report.md`; `experiments/exp_cyclic_stepwise_01_results.json`) — Destination-only intermediate supervision ($acc_{\text{val}} = 0.6750$, inconclusive intermediate outcome).
* **Document Status:** Preregistered Specification (Authoritative — Finalized Protocol; Stage 1 Implementation Pending Authorization)
* **Date:** 2026-10-16
* **Runner Target Path:** `experiments/run_exp_cyclic_persist_01.py`
* **Focused Test Suite Target:** `tests/test_exp_cyclic_persist_01.py`
* **Official Results Target Artifact:** `experiments/exp_cyclic_persist_01_results.json`
* **Smoke Test Target Artifact:** `experiments/exp_cyclic_persist_01_SMOKE_ONLY_seed42.json`

---

## 2. Research Context, Questions, and Epistemic Separation

### 2.1 Empirical Facts Established by Predecessors
The protocol for `EXP-CYCLIC-PERSIST-01` is strictly grounded in verified facts from prior experiments:
1. **Fact 1 (Immediate Write Readout Alignment):** In `EXP-CYCLIC-STEPWISE-01`, auxiliary destination supervision enabled high-fidelity readout of the freshly written register immediately after update: macro $C_{\text{dest}}(\text{step } 0) = 0.9856$ and $C_{\text{dest}}(\text{step } 1) = 0.8368$.
2. **Fact 2 (Severe Register Read Asymmetry):** At terminal step 2, reading register $r1$ (which was written at step 1 immediately preceding step 2) exhibits strong canonical alignment across all 5 seeds: $\bar{C}_{\text{read}}(r1) = 0.8368 \ge 0.80$ (range 0.8158–0.8568). In contrast, reading register $r0$ (which was written at step 0 and subjected to an intervening state update at step 1) exhibits severe degradation across all 5 seeds: $\bar{C}_{\text{read}}(r0) = 0.2573 < 0.30$ (range 0.1718–0.3003).
3. **Fact 3 (Downstream Arithmetic Degradation):** The parameter-free Givens ALU achieves macro $C_{\text{ALU}} = 0.5374$ at step 2, which is an exact mathematical consequence of receiving a corrupted $u_a$ input vector ($r0$).
4. **Fact 4 (Step-Wise Accuracy Profile):** Validation accuracies across sequence steps in `EXP-CYCLIC-STEPWISE-01` are: Step 0: 1.0000 (5/5 seeds); Step 1: 0.9875 (4 seeds 1.0000, 1 seed 0.9375); Step 2: 0.6750 (range 0.4375–0.8750). Execution failures concentrate exclusively at terminal step 2.
5. **Fact 5 (Overfitting Dynamics):** All 5 seeds in `EXP-CYCLIC-STEPWISE-01` reached 100% training accuracy ($64/64$) within 100–150 updates, while terminal validation loss diverged later.

### 2.2 Candidate Explanations (Hypotheses vs. Facts)
The following are candidate explanations under empirical investigation, not established facts:
* **Hypothesis A (Destructive Continuous State Interference):** In a monolithic continuous state $S \in \mathbb{R}^{212}$ lacking structural slot isolation, writing $r1 \leftarrow b$ at step 1 via continuous gating $g_1$ and candidate $C_1$ interferes with or overwrites the coordinates where $r0 \leftarrow a$ was written at step 0.
* **Hypothesis B (Supervision Asymmetry / Credit-Assignment Horizon):** The auxiliary loss in `EXP-CYCLIC-STEPWISE-01` supervised only the destination register at step 1 ($r1 \to b$). Because $r0$ received zero direct gradient signal at step 1, the network faced no penalty for letting $r0$ drift or dissolve in the continuous state during the $S_1 \to S_2$ transition.

### 2.3 Research Question
Does adding auxiliary supervision for unaddressed-register persistence at step 1 (requiring $r0$ to preserve operand $a$ during the $S_1 \to S_2$ transition) eliminate the R0/R1 read asymmetry, restore canonical ALU input alignment ($\bar{C}_{\text{read}} \ge 0.80$), and enable `CyclicMGUPhase16Model` to pass the validation progression gate ($acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds) without altering model architecture?

### 2.4 Falsifiable Hypotheses
* **Hypothesis ($H_1$):** Insufficient persistence supervision (Hypothesis B) is the primary cause of the $r0/r1$ asymmetry. When step 1 is supervised to simultaneously write $r1 \leftarrow b$ and persist $r0 \leftarrow a$, the learned recurrent gating pathway (`mlp_gate`) learns to preserve $r0$ coordinates in $S_2$. Consequently:
  1. $\bar{C}_{\text{read}}(r0) \ge 0.80$ emerges at step 2, eliminating the asymmetry ($\bar{C}_{\text{read}} \ge 0.80$).
  2. $\bar{C}_{\text{ALU}} \ge 0.80$ emerges because both ALU inputs are canonically aligned.
  3. Validation accuracy passes the progression gate: $acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds with 0 seeds $< 0.250$.
* **Null Hypothesis ($H_0$):** Continuous state interference (Hypothesis A) is an intrinsic structural limitation of monolithic continuous state routing in `CyclicMGUPhase16Model`. Even with explicit persistence supervision at step 1, the model cannot maintain independent coordinate slots in continuous space while performing non-linear state updates. As a result:
  1. Validation accuracy fails the progression gate ($acc_{\text{val}} < 0.950$ across seeds), OR
  2. The network trades off destination writing accuracy for persistence, failing to achieve joint alignment.

---

## 3. Single-Intervention Constraint and Model Invariants

### 3.1 Single Intervention Definition
The single authorized intervention in `EXP-CYCLIC-PERSIST-01` is:
**Auxiliary supervision of unaddressed-register persistence during the intermediate state transition at step 1.**

The following mechanisms are **strictly excluded**:
* Adding new trainable readout heads, linear projections, or auxiliary layers.
* Altering the parameter-free cyclic Givens rotation ALU or its 5-harmonic basis.
* Partitioning the monolithic state vector $S \in \mathbb{R}^{212}$ into discrete structural slots (retaining monolithic continuous state).
* Introducing weight decay changes, learning rate schedules, dropout, or gradient clipping beyond the invariant optimizer contract.
* Any persistent input reinjection, recurrent loop unrolling alterations, or multi-pass computation.

### 3.2 Model Invariants
* **Model Class:** `CyclicMGUPhase16Model` (`src/re_reason/core/register_machine_models.py`).
* **Trainable Parameter Count:** Exactly **219,018** trainable parameters:
  * `mlp_read`: 29,066 parameters (`Linear(216, 128)` + `ReLU` + `Linear(128, 10)`).
  * `mlp_write`: 52,152 parameters (`Linear(32, 212)` + `ReLU` + `Linear(212, 212)`).
  * `mlp_gate`: 137,800 parameters (`Linear(436, 212)` + `ReLU` + `Linear(212, 212)`).
* **State Dimension:** Monolithic continuous state $S \in \mathbb{R}^{212}$ with zero initialization ($S_0 = \mathbf{0}$).
* **Zero Trainable Head Overhead:** The persistence query uses the exact existing `mlp_read` network with weights shared across read, destination, and persistence queries, decoded against the identical fixed prototype basis $\Phi \in \mathbb{R}^{10 \times 10}$.
* **Forward Evaluation Invariance:** Under default inference mode (`return_step_diagnostics=False`), the model executes the exact forward pass and produces bit-identical logits compared to pre-intervention code.

---

## 4. Dataset Partitions and Strict Evaluation Firewall

### 4.1 Canonical Program Trace
The task is 3-instruction modular addition modulo 10:
* Step 0 ($t=0$): `SET r0 a` $\implies S_0 \xrightarrow{x_0} S_1$, destination $r0 \leftarrow a$.
* Step 1 ($t=1$): `SET r1 b` $\implies S_1 \xrightarrow{x_1} S_2$, destination $r1 \leftarrow b$, unaddressed persistence $r0 \leftarrow a$.
* Step 2 ($t=2$): `ADD r0 r1` $\implies S_2 \xrightarrow{x_2} S_3$, terminal target $(a + b) \bmod 10$.

### 4.2 Dataset Partitions
* **TRAIN-FIT:** 64 operand pairs $(a, b)$ with $a \in \{0..9\}$ and $b \in \{1, 2, 3, 4, 6, 7, 8, 9\} \setminus \text{val\_pairs}$.
* **VAL-SEEN:** 16 sequence-disjoint operand pairs $(a, b)$:
  `[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6), (3, 4), (3, 6), (3, 7), (4, 7), (4, 8), (5, 8), (5, 9), (6, 9), (9, 1)]`
* **TEST-HELDOUT (Protected):** 20 operand pairs characterized by held-out second operand $b \in \{0, 5\}$.

### 4.3 Strict Test-Split Firewall
The protected `TEST-HELDOUT` split must **never** be loaded, constructed, queried, evaluated, or logged during training, smoke testing, checkpoint selection, or diagnostic analysis. All gating decisions are evaluated exclusively against `VAL-SEEN`.

---

## 5. Mathematical Formulation of the Persistence Objective

### 5.1 Querying Persistence via Existing Model Readout
At step 1 ($t=1$, instruction `SET r1 b`), the state transitions from $S_1$ to $S_2$.
The existing model readout is queried twice on state $S_2$:
1. **Addressed Destination Query ($r1$):**
   Input: $[S_2, e_{r1}] \in \mathbb{R}^{216}$, where $e_{r1} = [0, 1, 0, 0]^{\top}$.
   $$u_{\text{dest}, 1} = \operatorname{project}(\text{mlp\_read}([S_2, e_{r1}])) \in S^1 \subset \mathbb{R}^{10}$$
   $$\text{logits}_{\text{dest}, 1} = u_{\text{dest}, 1} \cdot \Phi^{\top} \in \mathbb{R}^{10}$$
   Target: $y_{\text{dest}, 1} = b$.
2. **Unaddressed Persistence Query ($r0$):**
   Input: $[S_2, e_{r0}] \in \mathbb{R}^{216}$, where $e_{r0} = [1, 0, 0, 0]^{\top}$.
   $$u_{\text{persist}, 1} = \operatorname{project}(\text{mlp\_read}([S_2, e_{r0}])) \in S^1 \subset \mathbb{R}^{10}$$
   $$\text{logits}_{\text{persist}, 1} = u_{\text{persist}, 1} \cdot \Phi^{\top} \in \mathbb{R}^{10}$$
   Target: $y_{\text{persist}, 1} = a$.

*Key Mathematical Equivalence:* Vector $u_{\text{persist}, 1}$ is computationally identical to the operand read vector $u_a = \operatorname{project}(\text{mlp\_read}([S_2, e_{r0}]))$ extracted at the start of step 2 prior to ALU injection. Supervising $u_{\text{persist}, 1}$ directly penalizes the corruption of $u_a$.

### 5.2 Loss Decomposition and Objective Function
The total training loss is:
$$\mathcal{L} = \mathcal{L}_{\text{terminal}} + \lambda_{\text{dest}} \cdot \mathcal{L}_{\text{dest}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{persist}}$$

Where:
1. **Terminal Task Loss:**
   $$\mathcal{L}_{\text{terminal}} = \text{CE}(\text{logits}_{\text{terminal}}, (a + b) \bmod 10)$$
2. **Intermediate Destination Loss:**
   $$\mathcal{L}_{\text{dest}} = \frac{1}{2} \left[ \text{CE}(\text{logits}_{\text{dest}, 0}, a) + \text{CE}(\text{logits}_{\text{dest}, 1}, b) \right]$$
   Identical to $\mathcal{L}_{\text{aux}}$ in `EXP-CYCLIC-STEPWISE-01`.
3. **Intermediate Persistence Loss:**
   $$\mathcal{L}_{\text{persist}} = \text{CE}(\text{logits}_{\text{persist}, 1}, a)$$
   Evaluated strictly at step 1 on register $r0$. (Step 0 is omitted because initial state $S_0 = \mathbf{0}$ contains no prior register allocations).

### 5.3 Finalized Preregistered Loss Weighting (PI Approved 2026-10-16)
The objective coefficients are formally finalized and preregistered:
$$\lambda_{\text{dest}} = 1.0, \quad \lambda_{\text{persist}} = 0.5$$

* **Mathematical Rationale (Equal Assertion Weighting):** Expanding the objective yields:
  $$\mathcal{L} = \mathcal{L}_{\text{terminal}} + 0.5 \cdot \text{CE}(\text{dest}_0, a) + 0.5 \cdot \text{CE}(\text{dest}_1, b) + 0.5 \cdot \text{CE}(\text{persist}_1, a)$$
  Every single intermediate register assertion ($r0$ at step 0, $r1$ at step 1, $r0$ at step 1) receives an **identical effective weight of 0.5**.
* **Scale Preservation:** The destination gradient scale is identical to `EXP-CYCLIC-STEPWISE-01` ($0.5$ per assertion); adding persistence does not dilute or attenuate the write signals.
* **Status:** Formally selected and approved by PI (2026-10-16). No coefficient tuning on validation data permitted.

---

## 6. Training Protocol and Hyperparameters

* **Optimizer:** AdamW (`torch.optim.AdamW`).
* **Learning Rate:** $3.0 \times 10^{-3}$ (constant schedule, no warmup, no decay).
* **Weight Decay:** $1.0 \times 10^{-2}$.
* **Betas:** $(0.9, 0.999)$.
* **Batch Size:** 64 (full training batch).
* **Training Updates:** Exactly 3,000 updates.
* **Seeds:** `[42, 43, 44, 45, 46]` (5 seeds, invariant).
* **Evaluation Cadence:** Every 50 updates on both full `TRAIN-FIT` and full `VAL-SEEN` splits.
* **Checkpoint Selection Policy:** Checkpoint exhibiting minimum total validation loss:
  $$\mathcal{L}_{\text{val, total}} = \mathcal{L}_{\text{terminal}} + \lambda_{\text{dest}} \cdot \mathcal{L}_{\text{dest}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{persist}}$$
  Ties broken by highest validation accuracy ($acc_{\text{val}}$), then earliest update step.

---

## 7. Diagnostic Representation Metrics

All representation metrics operate on 10-dimensional Fourier vectors with cosine similarity guarded by $\epsilon = 10^{-8}$ and norm floor $10^{-7}$ (returning 0.0 if either norm is $< 10^{-7}$):

1. **Register-Specific Read Alignment at Step 2 ($C_{\text{read}}(r0), C_{\text{read}}(r1)$):**
   $$C_{\text{read}}(r) = \frac{1}{N} \sum_{i=1}^N \frac{u_{\text{read}, i}^{(r)} \cdot \Phi(v_{\text{true}, i}^{(r)})}{\|u_{\text{read}, i}^{(r)}\|_2 \|\Phi(v_{\text{true}, i}^{(r)})\|_2 + \epsilon}, \quad C_{\text{read}} = \frac{C_{\text{read}}(r0) + C_{\text{read}}(r1)}{2}$$
2. **Step 1 Persistence Alignment ($C_{\text{persist}}(r0)$):**
   Cosine similarity between post-update persistence vector $u_{\text{persist}, 1}$ and canonical codeword $\Phi(a)$ at step 1:
   $$C_{\text{persist}}(r0) = \frac{1}{N} \sum_{i=1}^N \frac{u_{\text{persist}, i}^{(r0)} \cdot \Phi(a_i)}{\|u_{\text{persist}, i}^{(r0)}\|_2 \|\Phi(a_i)\|_2 + \epsilon}$$
3. **Cross-Context Read Consistency ($S_{\text{context}}$):**
   Evaluated at step 2 across operand roles and sequence contexts for all digits $v$ with $|\mathcal{R}(v)| \ge 2$:
   $$S_{\text{context}}(v) = \frac{1}{\binom{|\mathcal{R}(v)|}{2}} \sum_{j < k} \frac{u_j \cdot u_k}{\|u_j\|_2 \|u_k\|_2 + \epsilon}$$
4. **ALU Output Phase Alignment ($C_{\text{ALU}}$):**
   Evaluated at step 2 against canonical sum codeword $\Phi((a+b) \bmod 10)$:
   $$C_{\text{ALU}} = \frac{1}{N} \sum_{i=1}^N \frac{v_{\text{ALU}, i} \cdot \Phi((a_i + b_i) \bmod 10)}{\|v_{\text{ALU}, i}\|_2 \|\Phi((a_i + b_i) \bmod 10)\|_2 + \epsilon}$$
5. **Post-Update Destination Alignment ($C_{\text{dest}}$):**
   Evaluated at steps $t \in \{0, 1, 2\}$ between post-update destination vector and target codeword.
6. **Per-Step and Persistence Accuracies:**
   * Step 0 destination accuracy: $\operatorname{argmax}(\text{logits}_{\text{dest}, 0}) == a$.
   * Step 1 destination accuracy: $\operatorname{argmax}(\text{logits}_{\text{dest}, 1}) == b$.
   * Step 1 persistence accuracy: $\operatorname{argmax}(\text{logits}_{\text{persist}, 1}) == a$.
   * Step 2 terminal accuracy: $\operatorname{argmax}(\text{logits}_{\text{terminal}}) == (a + b) \bmod 10$.
7. **Loss Components Serialized at Every Checkpoint:**
   `train_loss_total`, `train_loss_terminal`, `train_loss_dest`, `train_loss_persist`, `train_acc`, `val_loss_total`, `val_loss_terminal`, `val_loss_dest`, `val_loss_persist`, `val_acc`.
8. **Numerical Contract & Metric Bounds Validation:**
   * All accuracy metrics validated in $[0.0, 1.0]$.
   * All cosine metrics validated in $[-1.0, 1.0]$.
   * All loss metrics validated finite and non-negative ($\ge 0.0$).
   * Parameter gradients verified non-None, non-zero, and non-NaN across `mlp_read`, `mlp_write`, and `mlp_gate`.

---

## 8. Two-Stage Execution Protocol and Authorization Boundaries

Execution is strictly partitioned into two gated stages:

### Stage 1: Implementation, Focused Tests, and Smoke Test
* **Scope:** 
  1. Implement persistence query hook and loss calculation in new runner `experiments/run_exp_cyclic_persist_01.py`.
  2. Implement unit tests in `tests/test_exp_cyclic_persist_01.py` verifying parameter invariance (219,018), loss decomposition, gradient flow to unaddressed state coordinates, and protected split firewall.
  3. Execute 100-update smoke test on Seed 42 via `--smoke`.
* **Mandatory Smoke Test Validations:**
  The 100-update smoke test artifact must explicitly satisfy and log all five validation criteria:
  1. **Loss Decomposition:** Exact numerical equality $\mathcal{L}_{\text{total}} == \mathcal{L}_{\text{terminal}} + 1.0 \times \mathcal{L}_{\text{dest}} + 0.5 \times \mathcal{L}_{\text{persist}}$ across every update and logged checkpoint.
  2. **Gradient Flow:** Non-zero, finite gradients verified across `mlp_read`, `mlp_write`, and `mlp_gate` on every update, confirming direct gradient propagation to the unaddressed register ($r0$) coordinate space during the step 1 transition.
  3. **Parameter-Count Invariance:** Total trainable parameters strictly equal to exactly 219,018 throughout instantiation and forward execution (zero new trainable heads or parameters).
  4. **Diagnostic Metric Correctness:** All representations ($C_{\text{read}}(r0), C_{\text{read}}(r1), C_{\text{persist}}(r0), S_{\text{context}}, C_{\text{ALU}}, C_{\text{dest}}$) strictly bounded in $[-1.0, 1.0]$, accuracies in $[0.0, 1.0]$, and losses finite and non-negative.
  5. **Protected TEST Split Firewall:** Zero queries to held-out test split ($b \in \{0, 5\}$), with `queries = 0` verified in the smoke artifact.
* **Artifact:** `experiments/exp_cyclic_persist_01_SMOKE_ONLY_seed42.json`.
* **Hard Stop Condition:** DeepSeek must stop immediately after the smoke test and deliver the Stage 1 report. The five-seed experiment is strictly unauthorized.

### Stage 2: Official Five-Seed Experiment
* **Gating:** Requires explicit, separate PI authorization after reviewing the Stage 1 report.
* **Scope:** Train seeds `[42, 43, 44, 45, 46]` for 3,000 updates using `--stage2-authorized`.
* **Artifact:** `experiments/exp_cyclic_persist_01_results.json`.

---

## 9. Comparison Gates and Decision Rules

### 9.1 Per-Seed Behavioral Gates and Five-Seed Aggregation
Evaluated over the 16 validation pairs ($N_{\text{val}} = 16$) across the 5 official seeds:
* For each seed $s$, let $k_s \in \{0, \dots, 16\}$ denote the number of correct validation predictions at its selected checkpoint, and $acc_s = k_s / 16$.
* **Per-Seed Classification:**
  * **Pass seed:** $acc_s \ge 0.950 \iff k_s = 16$ ($acc_s = 1.0000$).
  * **Reject seed:** $acc_s < 0.250 \iff k_s \le 3$ ($acc_s \le 0.1875$).
  * **Intermediate seed:** $0.250 \le acc_s < 0.950 \iff 4 \le k_s \le 15$ ($0.2500 \le acc_s \le 0.9375$).
* **Five-Seed Aggregate Counts:**
  Let $N_{\text{pass}} = \sum_{s=1}^5 \mathbb{I}[k_s = 16]$, $N_{\text{reject}} = \sum_{s=1}^5 \mathbb{I}[k_s \le 3]$, and $N_{\text{mid}} = \sum_{s=1}^5 \mathbb{I}[4 \le k_s \le 15]$.
* **Five-Seed Behavioral Categories (Mutually Exclusive & Collectively Exhaustive):**
  1. `BEHAVIOR_PROGRESSION_PASS`: $N_{\text{pass}} \ge 4$ AND $N_{\text{reject}} = 0$ (at least 4 seeds at 16/16, zero seeds $\le 3/16$).
  2. `BEHAVIOR_REJECTION`: $N_{\text{reject}} \ge 4$ (at least 4 seeds $\le 3/16$).
  3. `BEHAVIOR_INTERMEDIATE`: All other 5-seed distributions.

### 9.2 Representation Aggregation across Seeds
Macro-averaged metrics across selected checkpoints:
$$\bar{C}_{\text{read}} = \frac{1}{5} \sum_{s=1}^5 C_{\text{read}}^{(s)}, \quad \bar{S}_{\text{context}} = \frac{1}{5} \sum_{s=1}^5 S_{\text{context}}^{(s)}, \quad \bar{C}_{\text{ALU}} = \frac{1}{5} \sum_{s=1}^5 C_{\text{ALU}}^{(s)}$$

* **Representation Categories (Mutually Exclusive & Collectively Exhaustive):**
  1. `REP_CANONICAL_FULL`: $\bar{C}_{\text{read}} \ge 0.80$ AND $\bar{S}_{\text{context}} \ge 0.80$ AND $\bar{C}_{\text{ALU}} \ge 0.80$.
  2. `REP_CANONICAL_INPUT_ONLY`: $\bar{C}_{\text{read}} \ge 0.80$ AND $\bar{S}_{\text{context}} \ge 0.80$ AND $\bar{C}_{\text{ALU}} < 0.80$.
  3. `REP_FAILED`: $\bar{C}_{\text{read}} < 0.30$ AND $\bar{C}_{\text{ALU}} < 0.30$.
  4. `REP_PARTIAL`: All other triples $(\bar{C}_{\text{read}}, \bar{S}_{\text{context}}, \bar{C}_{\text{ALU}})$.

* **Asymmetry Resolution Metric (Diagnostic):**
  In addition to macro $\bar{C}_{\text{read}}$, the report must explicitly record $\bar{C}_{\text{read}}(r0)$ and $\bar{C}_{\text{read}}(r1)$. The R0/R1 asymmetry is defined as resolved if and only if:
  $$\bar{C}_{\text{read}}(r0) \ge 0.80 \quad \text{AND} \quad \bar{C}_{\text{read}}(r1) \ge 0.80$$

### 9.3 Exhaustive Decision Table (3 × 4 Matrix)

| Behavioral State (5 Seeds, $N_{\text{val}}=16$) | Representation State ($\bar{C}_{\text{read}}, \bar{S}_{\text{context}}, \bar{C}_{\text{ALU}}$) | Scientific Category (`enum`) | Epistemic Status & Interpretation |
| :--- | :--- | :--- | :--- |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_CANONICAL_FULL` | **`BEHAVIORAL_PASS_CANONICAL_FULL`** | **Hypothesis $H_1$ fully supported.** Persistence supervision eliminates $r0$ degradation; full canonical harmonic alignment and arithmetic generalization achieved. |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_CANONICAL_INPUT_ONLY` | **`BEHAVIORAL_PASS_CANONICAL_INPUT_ONLY`** | **Generalization with input alignment.** Both input operands align canonically ($\bar{C}_{\text{read}} \ge 0.80, \bar{S}_{\text{context}} \ge 0.80$), but raw ALU output alignment is sub-threshold, compensated by routing networks. |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_PARTIAL` | **`BEHAVIORAL_PASS_PARTIAL_ALIGNMENT`** | **Generalization with partial alignment.** Behavioral progression achieved, but representation metrics remain partially aligned; mechanistic support for $H_1$ is intermediate. |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_FAILED` | **`BEHAVIORAL_PASS_UNALIGNED`** | **Generalization without canonical alignment.** Progression pass met while canonical alignment fails; $H_1$ disconfirmed; alternative mechanism uncharacterized. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_CANONICAL_FULL` | **`INTERMEDIATE_CANONICAL_FULL`** | **Intermediate behavior despite full alignment.** Canonical representations emerge across both registers, but sequence execution does not achieve 5-seed progression consensus. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_CANONICAL_INPUT_ONLY` | **`INTERMEDIATE_CANONICAL_INPUT_ONLY`** | **Intermediate behavior with input alignment.** Input reads align canonically across both registers, but ALU output is sub-threshold and behavior fails progression gate. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_PARTIAL` | **`INCONCLUSIVE_INTERMEDIATE`** | **Inconclusive outcome.** Intermediate behavioral accuracy and partial representation alignment; persistence supervision improved coordinates partially but insufficiently. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_FAILED` | **`INCONCLUSIVE_UNALIGNED`** | **Inconclusive unaligned.** Intermediate behavioral accuracy with unaligned representations; neither progression nor complete rejection. |
| **`BEHAVIOR_REJECTION`** | `REP_CANONICAL_FULL` | **`REJECTION_CANONICAL_FULL_DECOUPLED`** | **Rejection with full alignment.** Canonical representations emerge, but sequence behavior fails ($\le 3/16$ in $\ge 4/5$ seeds); representations decoupled from task execution. |
| **`BEHAVIOR_REJECTION`** | `REP_CANONICAL_INPUT_ONLY` | **`REJECTION_CANONICAL_INPUT_DECOUPLED`** | **Rejection with input alignment.** Input reads align, but sequence behavior fails; representations decoupled from task execution. |
| **`BEHAVIOR_REJECTION`** | `REP_PARTIAL` | **`FALSIFIED_PARTIAL_ALIGNMENT`** | **Falsified for persistence intervention.** Rejection boundary met with incomplete alignment; persistence supervision fails to restore generalization. |
| **`BEHAVIOR_REJECTION`** | `REP_FAILED` | **`FALSIFIED_FULL`** | **Falsified: full rejection ($H_0$ supported).** Both behavioral accuracy and canonical representation alignment fail. Continuous state interference cannot be overcome by persistence supervision. |
| **ANY** (Numerical Failure) | **ANY** | **`IMPLEMENTATION_INVALID`** | **Invalid execution.** Numerical check failure, NaN/Inf, non-finite loss/gradients, invalid seed count $\ne 5$, or test failure. |

### 9.4 Validation Set Limitations & Epistemic Boundaries
* $N_{\text{val}} = 16$ pairs provides a binomial resolution of $\approx 6.25\%$ per instance.
* All validation pairs are sequence-disjoint combinations of seen training operands ($b \in \{1, 2, 3, 4, 6, 7, 8, 9\}$).
* A pass outcome on `VAL-SEEN` establishes modular addition generalization on unobserved combinations of seen digits; it does **not** evaluate out-of-distribution generalization to unseen digits. The protected `TEST-HELDOUT` split ($b \in \{0, 5\}$) remains strictly firewalled.

---

## 10. Compute and Execution Budget

* **Platform Basis:** Execution on host CPU (matching `EXP-CYCLIC-STEPWISE-01`).
* **Stage 1 Smoke Test Budget:** 1 seed (Seed 42) $\times$ 100 updates $\approx$ **1.5 seconds** CPU runtime.
* **Stage 2 Official Five-Seed Budget:**
  * 5 seeds $\times$ 3,000 updates = 15,000 updates.
  * In `EXP-CYCLIC-STEPWISE-01`, per-seed CPU runtime was $25.1 \pm 0.4$ seconds (total 125.7 s).
  * Adding one additional `mlp_read` query and cross-entropy calculation at step 1 adds an estimated 10–15% computational overhead per update.
  * **Estimated Per-Seed Runtime:** $\approx 28 - 30$ seconds on host CPU.
  * **Estimated Total Runtime:** $\approx 140 - 150$ seconds ($\approx 2.5$ minutes) on host CPU.
* **Storage Footprint:**
  * Smoke test artifact: $\approx 25$ KB.
  * Official Stage 2 results artifact: $\approx 1.6$ MB.
* **Verification Battery:**
  1. Focused unit tests: `pytest tests/test_exp_cyclic_persist_01.py -v` (expected: ~40 tests).
  2. Governance checks: `pytest tests/test_documentation_governance.py -v` (13 tests).
  3. Pre-flight parameter count assertion: `model.count_parameters() == 219_018`.
  4. Pre-flight firewall assertion: `assert_protected_domain_absent()` and `assert_canonical_trace()`.
  5. Checkpoint loss consistency assertion: $\mathcal{L}_{\text{val, total}} == \mathcal{L}_{\text{term}} + \lambda_{\text{dest}}\mathcal{L}_{\text{dest}} + \lambda_{\text{persist}}\mathcal{L}_{\text{persist}}$.

---

## 11. Authorization Status and Governance Gates

* **Objective Weighting Decision:** **RESOLVED.** The PI formally approved Option 1 ($\lambda_{\text{dest}} = 1.0, \lambda_{\text{persist}} = 0.5$) on 2026-10-16. This establishes equal effective weighting ($0.5$) across all three intermediate register assertions ($r0$ at step 0, $r1$ at step 1, $r0$ at step 1) while preserving the individual destination gradient scale from `EXP-CYCLIC-STEPWISE-01` without dilution.
* **Ablation Arm Decision:** **DEFERRED.** The persistence-only ablation arm ($\mathcal{L}_{\text{term}} + \mathcal{L}_{\text{persist}}$) is deferred until the primary multi-objective outcome is established.
* **Next Authorization Gate (Stage 1 Authorization):** **PENDING.** Stage 1 implementation (code changes in `experiments/run_exp_cyclic_persist_01.py`, focused unit tests in `tests/test_exp_cyclic_persist_01.py`, and execution of the 100-update seed-42 smoke test) remains strictly unauthorized pending separate explicit PI authorization.
* **Stage 2 Authorization:** **BLOCKED.** The five-seed experiment (`--stage2-authorized`) remains blocked until Stage 1 completion and independent audit review.
