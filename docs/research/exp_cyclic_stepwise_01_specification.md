# EXP-CYCLIC-STEPWISE-01: Authoritative Experiment Specification

## 1. Metadata and Preregistration Identity

* **Experiment Identifier:** `EXP-CYCLIC-STEPWISE-01`
* **Controlling Proposal:** Phase 1.6 Architectural Reformulation Proposal (`docs/reports/phase_1_6_architectural_reformulation_proposal.md`)
* **Predecessor Baseline:** `EXP-CYCLIC-ADD-01` (`experiments/exp_cyclic_add_01_results.json`)
* **Document Status:** Preregistered Specification (Authoritative)
* **Date:** 2026-10-16
* **Runner Path:** `experiments/run_exp_cyclic_stepwise_01.py`
* **Focused Test Suite:** `tests/test_exp_cyclic_stepwise_01.py`
* **Official Results Target Artifact:** `experiments/exp_cyclic_stepwise_01_results.json`

---

## 2. Research Question and Falsifiable Hypothesis

### 2.1 Research Question
Does auxiliary supervision of intermediate destination-register values during 3-step sequence rollout enable `CyclicMGUPhase16Model` to generalize modular addition and align its learned latent representations with the canonical harmonic coordinate system expected by its fixed cyclic ALU, without modifying model architecture?

### 2.2 Falsifiable Hypothesis
* **Hypothesis ($H_1$):** Under terminal-only supervision (`EXP-CYCLIC-ADD-01`), the learned read/write/gate pathways fit the final training targets by utilizing arbitrary, non-canonical internal coordinates, thereby supplying misaligned inputs to the fixed cyclic ALU. Supervising intermediate destination values ($r0$ at step 0, $r1$ at step 1) provides sufficient direct gradient signal to align intermediate read representations with canonical harmonic codewords, restoring ALU generalization to unseen validation pairs.
* **Null Hypothesis ($H_0$):** Intermediate destination supervision does not prevent coordinate drift or provide sufficient representational inductive bias. Even with intermediate destination supervision, validation accuracy fails to reach the progression gate ($acc_{\text{val}} < 0.950$ across seeds), and canonical representation alignment fails to emerge across the learned routing pathways ($C_{\text{read}} < 0.80$).
* **Scientific Scope:** The cyclic ALU is a fixed analytical Givens rotation operator (0 trainable parameters). Any performance change is attributable strictly to the supervision of surrounding learned routing pathways (`mlp_read`, `mlp_write`, `mlp_gate`), not to learned arithmetic.

---

## 3. Single-Intervention Constraint and Model Invariants

### 3.1 Single Intervention Only
The only intervention authorized by this specification is **auxiliary supervision of intermediate destination registers**.
The following mechanisms are **strictly excluded**:
* All-register state supervision (Arm B).
* Structural coordinate clamping or projections within recurrent loops.
* Auxiliary prediction heads or altered linear readouts.
* Recursive latent refinement, adaptive computation, or persistent input reinjection from TinyRecursiveModels (TRM).
* Any alteration of the fixed cyclic ALU or 5-harmonic basis.

### 3.2 Model Invariants
* **Model Class:** `CyclicMGUPhase16Model` (`src/re_reason/core/register_machine_models.py`).
* **Trainable Parameter Count:** Exactly **219,018** trainable parameters:
  * `mlp_read`: 29,066 parameters.
  * `mlp_write`: 52,152 parameters.
  * `mlp_gate`: 137,800 parameters.
* **State Dimension:** Monolithic state $S \in \mathbb{R}^{212}$ with zero initialization ($S_0 = \mathbf{0}$).
* **Behavioral Invariance:** Default forward execution (`return_step_diagnostics=False`) must produce bit-identical logits and final states compared to pre-intervention code. Diagnostic tensor capture adds zero trainable parameters and must not consume random number generator states.

---

## 4. Dataset Partitions and Evaluation Firewall

### 4.1 Canonical Task and Traces
The task is 3-instruction modular addition modulo 10:
1. Step 0 ($t=0$): `SET r0 a`
2. Step 1 ($t=1$): `SET r1 b`
3. Step 2 ($t=2$): `ADD r0 r1`
Ground truth final target: $(a + b) \bmod 10$.

### 4.2 Dataset Partitions
* **TRAIN-FIT:** 64 operand pairs $(a, b)$ with $a \in \{0..9\}$ and $b \in \{1, 2, 3, 4, 6, 7, 8, 9\} \setminus \text{val\_pairs}$.
* **VAL-SEEN:** 16 sequence-disjoint operand pairs $(a, b)$:
  `[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6), (3, 4), (3, 6), (3, 7), (4, 7), (4, 8), (5, 8), (5, 9), (6, 9), (9, 1)]`
* **TEST-HELDOUT (Protected):** 20 operand pairs characterized by held-out second operand $b \in \{0, 5\}$.

### 4.3 Strict Test-Split Firewall
The protected `TEST-HELDOUT` split must **never** be loaded, constructed, queried, evaluated, or logged during training, smoke testing, checkpoint selection, or diagnostic analysis. All gating decisions are evaluated exclusively against `VAL-SEEN`.

---

## 5. Target Mapping and Objective Function

### 5.1 Dynamic Destination-Target Extraction
Destination addresses are extracted dynamically from the instruction tensor:
$$\text{dest\_idx}_t = \operatorname{argmax}(\text{inputs}[:, t, 4:8])$$
The scalar target is gathered from the ground-truth register states `step_targets` of shape $(B, T, 4)$:
$$y_{\text{dest}, t} = \text{step\_targets}[b, t, \text{dest\_idx}_t]$$
For the canonical Stage 1 trace, this extracts strictly:
* $t=0$: Destination $r0$, target $a$.
* $t=1$: Destination $r1$, target $b$.
* $t=2$: Destination $r0$, target $(a + b) \bmod 10$.

### 5.2 Objective Function
$$\mathcal{L} = \mathcal{L}_{\text{terminal}} + \lambda \cdot \mathcal{L}_{\text{aux}}$$
Where:
$$\mathcal{L}_{\text{terminal}} = \text{CE}(\text{out.logits}, y_{\text{terminal}})$$
$$\mathcal{L}_{\text{aux}} = \frac{1}{2} \left[ \text{CE}(\text{out.step_logits}[:, 0, :], y_{\text{dest}, 0}) + \text{CE}(\text{out.step_logits}[:, 1, :], y_{\text{dest}, 1}) \right]$$
* **Auxiliary Coefficient:** Fixed at $\lambda = 1.0$. No hyperparameter tuning permitted.
* **Double-Counting Exclusion:** Step 2 ($t=2$) is omitted from $\mathcal{L}_{\text{aux}}$ because $\text{out.logits} \equiv \text{out.step_logits}[:, 2, :]$; supervising step 2 in both terms would artificially distort the terminal weighting.

---

## 6. Training Protocol and Hyperparameters

* **Optimizer:** AdamW (`torch.optim.AdamW`).
* **Learning Rate:** $3.0 \times 10^{-3}$ (constant schedule).
* **Weight Decay:** $1.0 \times 10^{-2}$.
* **Betas:** $(0.9, 0.999)$.
* **Batch Size:** 64 (full training batch).
* **Training Updates:** Exactly 3,000 updates.
* **Seeds:** `[42, 43, 44, 45, 46]`.
* **Evaluation Cadence:** Every 50 updates on both full `TRAIN-FIT` and full `VAL-SEEN` splits.
* **Checkpoint Selection Policy:** Checkpoint exhibiting minimum validation total loss ($\mathcal{L}_{\text{val}}$). Ties are broken by highest validation accuracy, then earliest update step.

---

## 7. Complete Diagnostic Metric Definitions

All representation metrics operate on 10-dimensional Fourier vectors with cosine similarity guarded by $\epsilon = 10^{-8}$ and norm floor $10^{-7}$ (returning 0.0 if either vector norm is below $10^{-7}$).

1. **Canonical Operand-Read Alignment ($C_{\text{read}}$):**
   Evaluated at step 2 ($t=2$) before ALU injection. Cosine similarity between pre-update read vector $u_a$ and canonical Fourier codeword $\Phi(a)$ for operand 1 ($r0$), and between $u_b$ and $\Phi(b)$ for operand 2 ($r1$):
   $$C_{\text{read}}(r) = \frac{1}{N} \sum_{i=1}^N \frac{v_{\text{read}, i}^{(r)} \cdot \Phi(v_{\text{true}, i}^{(r)})}{\|v_{\text{read}, i}^{(r)}\|_2 \|\Phi(v_{\text{true}, i}^{(r)})\|_2 + \epsilon}, \quad C_{\text{read}} = \frac{C_{\text{read}}(r0) + C_{\text{read}}(r1)}{2}$$
2. **Cross-Context Read Consistency ($S_{\text{context}}$):**
   Evaluated at step 2. For each digit $v \in \{0..9\}$, pool all read vectors where true value is $v$ (across operand roles $r0$ and $r1$ and sequence contexts). If $|\mathcal{R}(v)| \ge 2$, compute mean pairwise cosine similarity over all distinct pairs:
   $$S_{\text{context}}(v) = \frac{1}{\binom{|\mathcal{R}(v)|}{2}} \sum_{j < k} \frac{u_j \cdot u_k}{\|u_j\|_2 \|u_k\|_2 + \epsilon}$$
   $S_{\text{context}}$ is the unweighted mean over all digits $v$ where $|\mathcal{R}(v)| \ge 2$.
3. **ALU Output Phase Alignment ($C_{\text{ALU}}$):**
   Evaluated at step 2. Cosine similarity between raw ALU output $v_{\text{ALU}} = \text{cyclic\_alu}(u_a, u_b)$ and canonical Fourier codeword $\Phi((a+b)\bmod 10)$:
   $$C_{\text{ALU}} = \frac{1}{N} \sum_{i=1}^N \frac{v_{\text{ALU}, i} \cdot \Phi((a_i + b_i) \bmod 10)}{\|v_{\text{ALU}, i}\|_2 \|\Phi((a_i + b_i) \bmod 10)\|_2 + \epsilon}$$
4. **Post-Update Destination Alignment ($C_{\text{dest}}$):**
   Evaluated at steps $t \in \{0, 1, 2\}$. Cosine similarity between post-update destination query vector $u_{\text{dest\_post}, t}$ and canonical codeword $\Phi(y_{\text{dest}, t})$, averaged across all 3 steps.
5. **Per-Step Accuracy:**
   Prediction accuracy computed as $\operatorname{argmax}(\text{step\_logits}[:, t, :])$ matching ground truth destination integer at steps 0, 1, and 2.
6. **Loss and Accuracy Components:**
   `train_loss_total`, `train_loss_terminal`, `train_loss_aux`, `train_acc`, `val_loss_total`, `val_loss_terminal`, `val_loss_aux`, `val_acc`.
7. **Finite Value & Gradient Checks:**
   All loss scalars and parameter gradient tensors must be finite (zero NaNs or Infs), with non-zero gradients verified across `mlp_read`, `mlp_write`, and `mlp_gate`.

---

## 8. Two-Stage Execution and Authorization Boundaries

Execution is strictly partitioned into two gated stages:

### Stage 1: Implementation, Focused Tests, and Smoke Test
* **Scope:** Implement dynamic loss and diagnostic tensor hooks; run focused unit tests (`tests/test_exp_cyclic_stepwise_01.py`); run 100-update smoke test on Seed 42 via `--smoke`.
* **Artifact:** `experiments/exp_cyclic_stepwise_01_SMOKE_ONLY_seed42.json`.
* **Hard Stop Condition:** DeepSeek must stop immediately after the smoke test and deliver the Stage 1 report. The five-seed experiment is strictly unauthorized.

### Stage 2: Official Five-Seed Experiment
* **Gating:** Requires explicit, separate PI authorization after reviewing the Stage 1 report.
* **Scope:** Train seeds `[42, 43, 44, 45, 46]` for 3,000 updates using `--stage2-authorized`.
* **Artifact:** `experiments/exp_cyclic_stepwise_01_results.json`.

---

## 9. Comparison Gates and Decision Rules

Evaluated against the 16 validation pairs ($N_{\text{val}} = 16$) on `VAL-SEEN`:

### 9.1 Per-Seed Behavioral Gates and Five-Seed Aggregation
Evaluated over the 16 validation pairs ($N_{\text{val}} = 16$) across the 5 official seeds $s \in \{1..5\}$.
For each seed $s$, let $k_s \in \{0, \dots, 16\}$ denote the number of correct validation predictions at its selected checkpoint, and $acc_s = k_s / 16$.
* **Per-Seed Classification:**
  * **Pass seed:** $acc_s \ge 0.950 \iff k_s = 16$ ($acc_s = 1.0000$).
  * **Reject seed:** $acc_s < 0.250 \iff k_s \le 3$ ($acc_s \le 0.1875$).
  * **Intermediate seed:** $0.250 \le acc_s < 0.950 \iff 4 \le k_s \le 15$ ($0.2500 \le acc_s \le 0.9375$).
* **Five-Seed Aggregate Counts:**
  Let $N_{\text{pass}} = \sum_{s=1}^5 \mathbb{I}[k_s = 16]$, $N_{\text{reject}} = \sum_{s=1}^5 \mathbb{I}[k_s \le 3]$, and $N_{\text{mid}} = \sum_{s=1}^5 \mathbb{I}[4 \le k_s \le 15]$, where $N_{\text{pass}} + N_{\text{mid}} + N_{\text{reject}} = 5$.
* **Five-Seed Behavioral Categories (Mutually Exclusive & Collectively Exhaustive):**
  1. `BEHAVIOR_PROGRESSION_PASS`: $N_{\text{pass}} \ge 4$ AND $N_{\text{reject}} = 0$ (at least 4 seeds at 16/16, zero seeds $\le 3/16$).
  2. `BEHAVIOR_REJECTION`: $N_{\text{reject}} \ge 4$ (at least 4 seeds $\le 3/16$).
  3. `BEHAVIOR_INTERMEDIATE`: All other 5-seed distributions (neither progression pass nor rejection boundary met; covers high-pass with outlier rejection $N_{\text{pass}} = 4, N_{\text{reject}} = 1$, partial consensus $N_{\text{pass}} \in \{1, 2, 3\}$, and broad intermediate performance $N_{\text{mid}} \ge 1$).

### 9.2 Checkpoint Selection, Metric Extraction, and Five-Seed Representation Aggregation
Representation alignment is evaluated on the designated `VAL-SEEN` split (the protected `TEST-HELDOUT` split is strictly firewalled and never evaluated).
* **Checkpoint Selection Policy:** For each seed $s \in \{1..5\}$, the evaluation checkpoint $c_s^*$ is selected strictly from the periodic validation evaluations (every 50 updates) by:
  $$c_s^* = \arg\min_{c} \left( \mathcal{L}_{\text{val, total}}(c), -acc_{\text{val}}(c), \text{update}(c) \right)$$
  exhibiting minimum validation total loss ($\mathcal{L}_{\text{val, total}} = \mathcal{L}_{\text{terminal}} + 1.0 \cdot \mathcal{L}_{\text{aux}}$), breaking ties by highest validation accuracy ($acc_{\text{val}}$), then earliest update step.
* **Metric Extraction:** From each seed's selected checkpoint $c_s^*$, three diagnostic metrics are extracted:
  * $C_{\text{read}}^{(s)}$: step-2 read alignment with canonical Fourier basis.
  * $S_{\text{context}}^{(s)}$: step-2 cross-context read consistency.
  * $C_{\text{ALU}}^{(s)}$: step-2 raw ALU output alignment with canonical Fourier sum.
* **Macro-Average Aggregation across Seeds:** Experiment-level representation metrics are computed as the unweighted arithmetic mean across all 5 official seeds:
  $$\bar{C}_{\text{read}} = \frac{1}{5} \sum_{s=1}^5 C_{\text{read}}^{(s)}, \quad \bar{S}_{\text{context}} = \frac{1}{5} \sum_{s=1}^5 S_{\text{context}}^{(s)}, \quad \bar{C}_{\text{ALU}} = \frac{1}{5} \sum_{s=1}^5 C_{\text{ALU}}^{(s)}$$
* **Representation Contract Categories (Mutually Exclusive & Collectively Exhaustive):**
  1. `REP_CANONICAL_FULL`: $\bar{C}_{\text{read}} \ge 0.80$ AND $\bar{S}_{\text{context}} \ge 0.80$ AND $\bar{C}_{\text{ALU}} \ge 0.80$.
  2. `REP_CANONICAL_INPUT_ONLY`: $\bar{C}_{\text{read}} \ge 0.80$ AND $\bar{S}_{\text{context}} \ge 0.80$ AND $\bar{C}_{\text{ALU}} < 0.80$.
  3. `REP_FAILED`: $\bar{C}_{\text{read}} < 0.30$ AND $\bar{C}_{\text{ALU}} < 0.30$.
  4. `REP_PARTIAL`: All other triples $(\bar{C}_{\text{read}}, \bar{S}_{\text{context}}, \bar{C}_{\text{ALU}})$ (i.e. neither full canonical, input-only canonical, nor failed; includes partial alignments in $[0.30, 0.80)$ and mixed coordinate profiles).
* **Numerical Contract & Metric Bounds Validation:**
  * **Accuracy Metrics ($acc_{\text{val}}$):** Validated to be finite and in $[0.0, 1.0]$.
  * **Cosine Representation Metrics ($C_{\text{read}}, S_{\text{context}}, C_{\text{ALU}}$):** Validated to be finite and in $[-1.0, 1.0]$, accepting all valid finite cosine-similarity values (including negative alignment values $\le 0.0$ and exact boundaries $\pm 1.0$). If any value is non-finite or outside $[-1.0, 1.0]$, execution is classified as `IMPLEMENTATION_INVALID`.
  * **Loss Metrics ($\mathcal{L}_{\text{val, total}}, \mathcal{L}_{\text{terminal}}, \mathcal{L}_{\text{aux}}$):** Validated to be finite and strictly non-negative ($\ge 0.0$).
  * **Negative Cosine Alignment Semantics:** When $\bar{C}_{\text{read}} < 0.30$ and $\bar{C}_{\text{ALU}} < 0.30$, valid negative cosine alignments strictly satisfy the `REP_FAILED` definition, mapping to `BEHAVIORAL_PASS_UNALIGNED` under progression pass, `INCONCLUSIVE_UNALIGNED` under intermediate behavior, and `FALSIFIED_FULL` under rejection boundary.
* **Telemetry Preservation:** Both the macro-averaged metrics $(\bar{C}_{\text{read}}, \bar{S}_{\text{context}}, \bar{C}_{\text{ALU}})$ and the individual per-seed records $[C_{\text{read}}^{(s)}, S_{\text{context}}^{(s)}, C_{\text{ALU}}^{(s)}]_{s=1}^5$ are serialized in the official results artifact for complete post-run auditability.

### 9.3 Exhaustive Decision Table (3 × 4 Matrix) with Epistemic Discipline
The decision matrix partitions all empirical outcomes across behavioral progression and macro-averaged representation alignment. It adheres to strict epistemic boundaries: behavioral generalization is decoupled from mechanistic claims, and no speculative mechanism ('shortcut', 'memorization', 'alternative manifold') is asserted without direct empirical proof.

| Behavioral State (5 Seeds, $N_{\text{val}}=16$) | Representation State ($\bar{C}_{\text{read}}, \bar{S}_{\text{context}}, \bar{C}_{\text{ALU}}$) | Scientific Category (`enum`) | Epistemic Status & Interpretation |
| :--- | :--- | :--- | :--- |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_CANONICAL_FULL` | **`BEHAVIORAL_PASS_CANONICAL_FULL`** | **Hypothesis $H_1$ fully supported.** Behavioral generalization achieved with full canonical harmonic alignment across reads, context, and ALU output. |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_CANONICAL_INPUT_ONLY` | **`BEHAVIORAL_PASS_CANONICAL_INPUT_ONLY`** | **Behavioral generalization with input alignment.** Reads align canonically ($\bar{C}_{\text{read}} \ge 0.80, \bar{S}_{\text{context}} \ge 0.80$), but raw ALU output alignment is sub-threshold ($\bar{C}_{\text{ALU}} < 0.80$), compensated by downstream routing networks. |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_PARTIAL` | **`BEHAVIORAL_PASS_PARTIAL_ALIGNMENT`** | **Behavioral generalization with partial alignment.** Behavioral gate passed, but evidence for canonical harmonic coordinate alignment is intermediate/incomplete; $H_1$ mechanistic claim is inconclusive. |
| **`BEHAVIOR_PROGRESSION_PASS`** | `REP_FAILED` | **`BEHAVIORAL_PASS_UNALIGNED`** | **Behavioral generalization with unaligned representations.** Behavioral gate passed while canonical alignment fails ($\bar{C}_{\text{read}} < 0.30 \land \bar{C}_{\text{ALU}} < 0.30$). $H_1$ mechanistic claim is disconfirmed; computational mechanism remains uncharacterized without direct evidence. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_CANONICAL_FULL` | **`INTERMEDIATE_CANONICAL_FULL`** | **Intermediate behavior despite full canonical alignment.** Canonical harmonic representations emerge, but sequence execution does not achieve 5-seed progression consensus. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_CANONICAL_INPUT_ONLY` | **`INTERMEDIATE_CANONICAL_INPUT_ONLY`** | **Intermediate behavior with canonical input alignment.** Input reads align canonically, but ALU output is sub-threshold and sequence execution does not achieve progression consensus. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_PARTIAL` | **`INCONCLUSIVE_INTERMEDIATE`** | **Inconclusive outcome.** Intermediate behavioral accuracy ($4..15/16$) and partial representation alignment ($0.30 \le \bar{C} < 0.80$); sub-threshold on both dimensions. |
| **`BEHAVIOR_INTERMEDIATE`** | `REP_FAILED` | **`INCONCLUSIVE_UNALIGNED`** | **Inconclusive unaligned.** Intermediate behavioral accuracy with unaligned representations ($\bar{C}_{\text{read}} < 0.30 \land \bar{C}_{\text{ALU}} < 0.30$); neither progression nor complete rejection. |
| **`BEHAVIOR_REJECTION`** | `REP_CANONICAL_FULL` | **`REJECTION_CANONICAL_FULL_DECOUPLED`** | **Behavioral rejection with full canonical alignment.** Canonical harmonic coordinates emerge at intermediate reads, but sequence behavior fails ($\le 3/16$ in $\ge 4/5$ seeds); representations are decoupled from task execution. |
| **`BEHAVIOR_REJECTION`** | `REP_CANONICAL_INPUT_ONLY` | **`REJECTION_CANONICAL_INPUT_DECOUPLED`** | **Behavioral rejection with canonical input alignment.** Canonical input reads emerge, but sequence behavior fails; read representations are decoupled from task execution. |
| **`BEHAVIOR_REJECTION`** | `REP_PARTIAL` | **`FALSIFIED_PARTIAL_ALIGNMENT`** | **Falsified for this intervention.** Behavioral rejection with incomplete representation alignment; auxiliary supervision fails to restore generalization. |
| **`BEHAVIOR_REJECTION`** | `REP_FAILED` | **`FALSIFIED_FULL`** | **Falsified for this intervention: full rejection.** Both behavioral accuracy ($\le 3/16$ in $\ge 4/5$ seeds) and canonical representation alignment ($\bar{C}_{\text{read}} < 0.30 \land \bar{C}_{\text{ALU}} < 0.30$) fail. Null hypothesis $H_0$ supported. |
| **ANY** (Numerical / Check Failure) | **ANY** | **`IMPLEMENTATION_INVALID`** | **Implementation-invalid execution.** Numerical check failure, NaN/Inf, non-finite loss/gradients, invalid seed count $\ne 5$, or test failure. |


