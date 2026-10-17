# RE:Reason — Architecture-to-Implementation Gap Audit & Milestone Proposal

**Date:** 2026-10-17  
**Status:** Completed Architecture Audit & Research Proposal (Read-only audit; implementation pending explicit PI authorization)  
**Authoritative Reference:** `docs/reports/architecture_gap_audit_and_milestone_proposal.md`  

---

## 1. Executive Assessment: Gap from Current Repository to Target Architecture

**Current repository reality:** RE:Reason currently operates as a straight-line, fixed-depth, single-trajectory recurrent baseline model (`CyclicMGUPhase16Model`, 219,018 parameters; `src/re_reason/core/register_machine_models.py`), evaluated on narrow 3-instruction modular addition traces (`src/re_reason/benchmarks/register_machine_benchmark.py`).

**Target conceptual architecture:** A recurrent latent state machine for multi-step algorithmic reasoning featuring:
1. Isolated speculative hypothesis exploration ($K > 1$).
2. Weight-tied recursive cognition with adaptive halting ($R \ge 1$, PonderNet).
3. Deterministic transition verification ($V_{\text{det}}$).
4. Transactional commit/rollback gating with clean master state snapshots.

**Distance to target: Substantial architectural divergence.**
* **Missing core machinery:** 5 of 7 core mechanisms in target conceptual pipeline (`docs/02_core_architecture.md`) are **not implemented** in active model code:
  * Hypothesis brancher module: Directory `src/re_reason/hypothesis/` contains only an empty `__init__.py`.
  * Typed decision subsystem: Directory `src/re_reason/decision/` contains only an empty `__init__.py`.
  * In-the-loop transition verifier: Directory `src/re_reason/verification/` contains only an abstract base class (`base.py`).
  * Transaction commit/rollback gate: Completely absent. State update in active models is unconditional continuous gating ($S_t = (1 - f_t) \odot S_{t-1} + f_t \odot C_t$).
  * Adaptive halting: No halting head, halting probability, or dynamic stopping loss implemented in `src/re_reason/`.
* **State of the art in repository:** Recent engineering focused on stabilizing coordinate preservation in continuous monolithic recurrent cells across a 3-step arithmetic trace (`EXP-CYCLIC-PERSIST-01`, macro val acc 0.9875). This validated that joint auxiliary destination and persistence supervision anchors register coordinate representations in continuous state. It did **not** validate hypothesis branching, verification, error containment, adaptive compute, or baseline superiority.
* **Strategic risk:** Continuing along Phase 1.6 straight-line modular addition (e.g. Stage 3 composed sequences) refines a conventional RNN cell on deterministic arithmetic; it does **not** exercise or advance the latent state machine architecture.

---

## 2. Architecture-to-Implementation Matrix

Classification of the 12 target conceptual components against repository evidence into 7 standard maturity states:
1. *Not implemented*
2. *Implemented but not adequately tested*
3. *Unit-tested in isolation*
4. *Experimentally evaluated in isolation*
5. *Integrated end to end*
6. *End-to-end behavior experimentally validated*
7. *Comparatively validated against appropriate baselines*

| Component | Target Specification | Repository Implementation Path | Relevant Tests | Experiment & Report Evidence | Maturity Classification | Notes & Grounded Facts |
|---|---|---|---|---|---|---|
| **1. Observation Encoder** | Maps $x_t \to e_t \in \mathbb{R}^D$ via lightweight MLP/linear projection | `src/re_reason/core/model.py`, `src/re_reason/core/register_machine_models.py` (`CyclicMGUPhase16Model.encoder`) | `tests/test_phase1.py`, `tests/test_phase1_6_models.py` | `EXP-CYCLIC-PERSIST-01` (`docs/reports/exp_cyclic_persist_01_report.md`) | **End-to-end behavior experimentally validated** | Linear projection encodes one-hot instruction to 256-dim embedding. In modular addition, $e_t$ is operand-blind by design; verified in Exp T (`docs/reports/experiment_t_upstream_representation.md`). |
| **2. Persistent Committed State & Isolation** | Clean master state $S_t$; isolated snapshot buffer allocation via explicit tensor cloning | `src/re_reason/state/container.py` (`StateContainer`); monolithic tensor in `src/re_reason/core/register_machine_models.py` | `tests/test_state.py`, `tests/test_contracts.py` | Flat continuous state validated in `EXP-CYCLIC-PERSIST-01`; Container isolation unexercised in training | **Unit-tested in isolation** (Container isolation) / **End-to-end behavior experimentally validated** (Monolithic recurrent state) | `StateContainer` supports `save_checkpoint()`, `restore_checkpoint()`, `torch.clone()`. Active models do not use `StateContainer`; state is unrolled as raw tensor `S` across fixed steps. No speculative branch isolation. |
| **3. Weight-tied Recursive Core ($R > 1$)** | Repeated application of single neural block $\theta$ across $r = 1 \dots R$ iterations per step | `src/re_reason/cognition/core.py` (`CognitionCore`); `src/re_reason/cognition/ablation_cores.py` | `tests/test_contracts.py`, `tests/test_phase1.py` | Exploratory parity trees (`experiments/run_benchmark.py`). In Phase 1.6, $R=1$ strictly. | **Implemented but not adequately tested** | `CognitionCore` unrolls $R$ iterations with tied weights. Excluded from Phase 1.5 and Phase 1.6 models. All active models use single-step recurrence ($R=1$). |
| **4. Adaptive Halting Controller** | PonderNet halting gate $p_{t,r} = \sigma(W_h s_{t,r} + b_h)$ with dynamic compute modulation | None in `src/re_reason/` | None | None | **Not implemented** | No halting head, halting probability loss, or variable-depth iteration loop exists in repository. |
| **5. Hypothesis Branching ($K > 1$)** | Generation of $K$ alternative candidate working states from master state $S_t$ | `src/re_reason/hypothesis/__init__.py` (empty docstring only) | None | `experiments/run_branching_feasibility.py`; `docs/reports/branching_feasibility_oracle.md` | **Experimentally evaluated in isolation** (Historical negative result) | Evaluated Gaussian noise perturbation $\delta \sim \mathcal{N}(0, \sigma^2)$ on frozen Phase 1 checkpoints. Falsified: oracle $K=8$ had per-draw information at chance; matched noise recovered ~46% of gain. No brancher code in `src/`. |
| **6. Typed Decision Heads & Confidence** | Output heads projecting latent states to structured actions, bounds, calibrated confidence | Integrated ad-hoc as `nn.Linear` in `src/re_reason/core/register_machine_models.py`; `src/re_reason/decision/__init__.py` (empty) | `tests/test_phase1_6_models.py` | `EXP-CYCLIC-PERSIST-01` | **Integrated end to end** (Linear classification) / **Not implemented** (Confidence calibration & typed structures) | Readout is standard linear layer mapping 212-dim state to 10-class modular output. No structured confidence head ($c^{(k)}$) or multi-head typed validation implemented. |
| **7. Deterministic Transition Verifier** | Programmatic invariant rule checker: $V_{\text{det}}(\hat{y}^{(k)}, x_t) \in \{0, 1\}$ | `src/re_reason/verification/base.py` (`BaseVerifier` ABC only) | `tests/test_contracts.py` | Benchmark evaluators only (`src/re_reason/benchmarks/algorithmic_tasks.py`); never used in model forward pass | **Unit-tested in isolation** (Abstract interface only) | Concrete rule verifier never connected inside model execution loop. Rules exist only in test assertions and dataset generation. |
| **8. Learned Verifier** | Auxiliary critic network scoring state transition consistency: $V_{\text{learned}}(s^{(k)}, e_t) \in [0, 1]$ | None in `src/re_reason/` | None | `experiments/run_verifier_feasibility.py`; `docs/reports/verifier_feasibility.md` | **Experimentally evaluated in isolation** (Historical negative result) | Evaluated MLP verifier on Phase 1 transitions. AUROC skill collapsed to deterministic baseline; decision-relevant signal was near chance. Formally rejected in Phase 1 Synthesis (`docs/RESEARCH_STATUS.md` line 146). |
| **9. Transaction Commit / Rollback Gate** | Coord finalization: $S_{t+1} \leftarrow s^{(k^*)}$ if valid, else $S_{t+1} \leftarrow S_t$ (abort/rollback) | None in `src/re_reason/` | None | None | **Not implemented** | No module arbitrates between candidate verification outputs, commits winning candidate, or aborts/rolls back master state. |
| **10. Multi-step Episode Orchestration** | Execution loop coordinating input, branching, unrolling, verification, commit/rollback | Inline sequential loop in `experiments/run_exp_cyclic_persist_01.py` and `src/re_reason/core/register_machine_models.py` | `tests/test_exp_cyclic_persist_01.py` | `EXP-CYCLIC-PERSIST-01` | **Integrated end to end** (Fixed straight-line sequence only) | Current orchestration runs fixed 3-step loop (`SET r0 a`, `SET r1 b`, `ADD r0 r1`). Zero support for variable horizons, conditional branches, or state rollback. |
| **11. Benchmarks & Baselines** | Algorithmic tasks exercising dead ends, verification, rollback; compute-matched controls | `src/re_reason/benchmarks/register_machine_benchmark.py`, `src/re_reason/benchmarks/register_swap.py`; `src/re_reason/core/baseline_gru.py` | `tests/test_phase1_6_benchmarks.py` | Stage 1 Pilot (`experiments/phase1_6_stage1_pilot_results.json`) | **Integrated end to end** (Straight-line arithmetic only) | Current benchmark is straight-line modular addition. No active benchmark exercises dead ends, candidate pruning, or rollback. Compute-matched GRU baseline evaluated only on Stage 1 Track 1A (failed). |
| **12. Instrumentation & Diagnostics** | Diagnostic probes for candidate diversity, duplicate rates, FAR/FRR, rollback integrity, FLOPs | Representation probes in `experiments/run_exp_cyclic_persist_01.py`; parameter counters | `tests/test_phase1_6_stage1_telemetry.py` | `EXP-CYCLIC-PERSIST-01` ($C_{\text{read}}, S_{\text{context}}, C_{\text{ALU}}, C_{\text{dest}}$) | **Integrated end to end** (Representation telemetry only) | Representation probes track coordinate alignment on unaddressed registers. Telemetry for branch diversity, rollback contamination (State Pollution Index), and verified execution latency is missing. |

---

## 3. Verified Research Baseline and Explicit Limitations

### 3.1 Registered Experimental Results
The four cyclic supervision experiments establish the official research baseline for the monolithic recurrent core (`CyclicMGUPhase16Model`, 219,018 trainable parameters, analytical Givens rotation ALU):

| Experiment ID | Objective Function Formulation | Macro Val Accuracy | Registered Outcome | C_read(r0) | C_read(r1) | Primary Empirical Finding | Authoritative Reference |
|---|---|---:|---|---:|---:|---|---|
| `EXP-CYCLIC-ADD-01` | $\mathcal{L}_{\text{term}}$ (terminal CE only) | 0.0625 | `FAIL_REJECTION` | — | — | Complete generalization failure (< 0.250 on 5/5 seeds). Monolithic continuous state cannot ground arithmetic routing without intermediate supervision. | `docs/reports/exp_cyclic_stepwise_01_report.md` |
| `EXP-CYCLIC-STEPWISE-01` | $\mathcal{L}_{\text{term}} + 1.0 \cdot \mathcal{L}_{\text{dest}}$ | 0.6750 | `INCONCLUSIVE_INTERMEDIATE` | 0.2573 | 0.8368 | Destination supervision substantially lifts accuracy (+0.6125), but fails progression gate ($0.950$) due to severe read asymmetry degrading unaddressed $r0$. | `docs/reports/exp_cyclic_stepwise_01_report.md` |
| `EXP-CYCLIC-PERSIST-01` | $\mathcal{L}_{\text{term}} + 1.0 \cdot \mathcal{L}_{\text{dest}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$ | 0.9875 | `BEHAVIORAL_PASS_CANONICAL_FULL` | 0.8879 | 0.9303 | Joint supervision eliminates read asymmetry, grounding both operands ($\ge 0.80$). Passes progression gate on `VAL-SEEN`. | `docs/reports/exp_cyclic_persist_01_report.md` |
| `EXP-CYCLIC-PERSIST-ABL-01` | $\mathcal{L}_{\text{term}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$ | 0.5375 | `INCONCLUSIVE_INTERMEDIATE` | 0.7710 | 0.2597 | Persistence supervision alone preserves $r0$ but fails to ground write register $r1$ (write acc 0.1250), inverting read asymmetry. Confirms joint supervision is necessary. | `docs/reports/exp_cyclic_persist_abl_01_report.md` |

### 3.2 Explicit Limitations of Baseline
1. **Narrow benchmark domain:** Evaluated exclusively on a 3-instruction modular addition sequence (`SET r0 a`, `SET r1 b`, `ADD r0 r1`).
2. **Fixed analytical ALU:** Arithmetic computation $(a + b) \bmod 10$ is executed by a parameter-free Givens rotation kernel (`src/re_reason/core/register_machine_models.py`). Model does not learn arithmetic; it only learns coordinate routing.
3. **Protected test split firewall:** Held-out operand partition ($b \in \{0, 5\}$) remains strictly unevaluated across all official runs. Out-of-distribution generalization is unproven.
4. **No evidence of state machine operation:** Baseline models execute straight-line traces with $K=1, R=1$, no branching, no verifier, and no rollback.
5. **No demonstrated baseline superiority:** In Phase 1.6 Stage 1 Track 1A, parameter-matched `VanillaGRUPhase16Model` failed ($0.0000$ val acc) under terminal supervision, but neither model was trained with the joint persistence objective on equal footing. Superiority over compute-matched baselines is unproven.

---

## 4. Critical Gaps and Ranked Dependencies

The gap between the repository and the target architecture decomposes into five ranked dependencies:

```
[Rank 1: Meaningful Hypothesis Branching (K > 1)]
                       │
                       ▼
[Rank 2: Deterministic Verifier & Transaction Gate]
                       │
                       ▼
[Rank 3: Task Benchmarks with Branching & Dead Ends]
                       │
                       ▼
[Rank 4: Speculative State Container Integration]
                       │
                       ▼
[Rank 5: Recursive Cognition & Adaptive Halting (R > 1)]
```

### Rank 1: Meaningful Hypothesis Branching ($K > 1$) in Continuous Latent Space
* **Nature:** Fundamental scientific blocker.
* **Evidence:** Phase 2 feasibility study (`docs/reports/branching_feasibility_oracle.md`) demonstrated that adding stochastic noise perturbations $\delta \sim \mathcal{N}(0, \sigma^2)$ produces duplicate or destructive trajectories. Oracle $K=8$ had per-draw information at chance.
* **Requirement:** Must formulate structured branch proposal mechanisms (e.g. discrete action-conditioned latent proposals or learned orthogonal perturbation vectors) that maintain branch diversity ($\text{cos\_dist} > 0.20$) and low duplicate trajectory rates ($< 30\%$).

### Rank 2: Deterministic In-the-Loop Verifier & Transaction Commit/Rollback Gate
* **Nature:** Architecture & software engineering gap.
* **Evidence:** Abstract interface exists (`src/re_reason/verification/base.py`), but zero concrete verifiers or transaction gates exist in model forward passes.
* **Requirement:** Implement a hard rule checker $V_{\text{det}}$ that validates candidate state transitions, and a transaction arbiter that commits the highest-confidence valid candidate or aborts to $S_t$ when all candidates violate constraints.

### Rank 3: Benchmarks with Branching Decision Points, Dead Ends, and Rollback
* **Nature:** Benchmark design gap.
* **Evidence:** Modular addition traces are strictly deterministic and straight-line. There are no alternative paths, no dead ends, no invalid transitions, and zero utility for state rollback.
* **Requirement:** A structured algorithmic benchmark where greedy choices lead to verifiable dead ends, requiring candidate exploration and state rollback to achieve task completion.

### Rank 4: Speculative State Container Integration
* **Nature:** Software integration gap.
* **Evidence:** `StateContainer` (`src/re_reason/state/container.py`) is unit-tested in isolation (`tests/test_state.py`), but active models unroll state as a raw PyTorch tensor.
* **Requirement:** Integrate `StateContainer` into model inference execution to manage speculative candidate allocations (`torch.clone()`), graph detachment, and atomic pointer updates without memory leakage.

### Rank 5: Recursive Cognition & Adaptive Halting ($R > 1$, PonderNet)
* **Nature:** Secondary architectural optimization.
* **Evidence:** `CognitionCore` exists (`src/re_reason/cognition/core.py`) but is unused in Phase 1.6; adaptive halting is not implemented.
* **Requirement:** Add weight-tied internal recurrence and PonderNet halting gate only after branching and verification are functional.

---

## 5. Candidate End-to-End Task Evaluation & Recommendation

To validate the target workflow, candidate tasks are evaluated against the 10 criteria specified in the audit charter:

| Candidate Task | Multi-Step Dependencies | Alternative Hypotheses (K > 1) | Speculative State Isolation | Trustworthy Verification Rule | Dead Ends & Rejection | Testable Rollback Recovery | Verifiable Goal State | Procedural Generation | Feasible on RX 6600M / CPU | Overall Suitability |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Extended Modular Addition** (Stage 3 composed sequences) | Yes (Chained steps) | **No** (Deterministic arithmetic) | **No** (Single state) | Yes (ALU equality) | **No** (No dead ends) | **No** (No rollback) | Yes (Final sum) | Yes (Exact pairs) | Yes (< 1 GB VRAM) | **Unsuitable** (Does not exercise state machine workflow) |
| **B. Parity Trees / DAGs** (Phase 2 benchmark) | Yes (Graph depth) | **No** (Greedy eval) | **No** (Bottom-up) | Yes (Boolean logic) | **No** (Deterministic reduction) | **No** (No backtracking) | Yes (Root bit) | Yes (Random DAGs) | Yes (< 1 GB VRAM) | **Unsuitable** (Tests recursion depth $R$, not branching/rollback) |
| **C. Register Program with Fault Injections** | Yes (Sequential instructions) | Partial (Alt branches) | Yes (Register snapshot) | Yes (Memory assertions) | Partial (Injected illegal ops) | Yes (Revert to snapshot) | Yes (Target registers) | Yes (Code generation) | Yes (< 1 GB VRAM) | **Moderate** (Good, but discrete state space obscures latent search) |
| **D. Constrained Graph Pathfinding with Dynamic Blockades** | **Yes** (Multi-hop path) | **Yes** (Multiple path options at nodes) | **Yes** (Candidate path states) | **Yes** (Adjacency & obstacle collision: $V_{\text{det}} \in \{0, 1\}$) | **Yes** (Unseen blocked corridors / dead ends) | **Yes** (Rollback to junction node) | **Yes** (Reach target node) | **Yes** (Procedural grid/graph generation, known A* ground truth) | **Yes** (Extremely fast on host CPU / 8 GB GPU) | **RECOMMENDED** (Fully exercises entire state-machine lifecycle) |

### Recommended Task: Constrained Graph Pathfinding with Dynamic Dead Ends
* **Task formulation:** An agent navigates a directed graph or 2D maze from start node $v_0$ to target node $v_{\text{goal}}$. At each node, 2–4 outgoing edges exist. Dynamic blockades are placed on certain paths (observable only upon attempting transition or via local constraint checks).
* **Why it exercises the intended workflow:**
  1. **Speculative isolation:** Model generates $K=2..4$ candidate movements from committed state $S_t$.
  2. **Deterministic verification:** $V_{\text{det}}$ checks edge connectivity and obstacle collision against ground-truth map constraints.
  3. **Transaction commit & rollback:** If Candidate 1 hits a dead end/blockade ($V_{\text{det}} = 0$) but Candidate 2 is open ($V_{\text{det}} = 1$), Candidate 2 is committed ($S_{t+1} \leftarrow s^{(2)}$). If all candidates fail, the system rolls back to $S_t$ and explores alternative actions without corrupting the committed position.
  4. **Ground-truth baseline:** Optimal A* or BFS provides exact oracle path length and failure boundaries.
  5. **Direct comparison:** Can be compared directly against non-branching recurrent baselines ($K=1$ GRU), which become trapped in dead ends due to lack of rollback capability.

---

## 6. Staged Implementation & Evaluation Roadmap

```
Current Baseline (EXP-CYCLIC-PERSIST-01)
                    │
                    ▼
[Milestone 1: Speculative State Isolation & Transaction Gate Engine] (Code & Tests)
                    │
                    ▼
[Milestone 2: Constrained Algorithmic Pathfinding Benchmark Suite] (Benchmark & Oracle)
                    │
                    ▼
[Milestone 3: Hypothesis Proposal & Diversity Exploration] (Research Gate 1)
                    │
                    ▼
[Milestone 4: End-to-End Latent State Machine Validation] (Research Gate 2)
                    │
                    ▼
[Milestone 5: Recursive Cognition & Adaptive Halting Integration] (Research Gate 3)
```

---

### Milestone 1: Speculative State Isolation & Transaction Gate Engine
* **Purpose:** Establish the core software infrastructure for isolated speculative branching and atomic commit/rollback.
* **Prerequisites:** Existing `StateContainer` (`src/re_reason/state/container.py`) and `BaseVerifier` (`src/re_reason/verification/base.py`).
* **Scope:**
  * Implement `TransactionCommitGate` managing candidate evaluation, winning branch commit, and clean rollback.
  * Update `StateContainer` to support $K$ detached speculative working buffers.
  * Implement deterministic transition verification harness.
  * **No neural network training in this milestone.**
* **Evaluation:** Unit tests verifying bit-exact state invariance upon rollback, zero memory leaks across 1,000 simulated rollbacks, and clean tensor graph detachment.
* **Controls:** Mock neural tensors with simulated valid/invalid verification signals.
* **Success criteria:** 100% test pass rate on state restoration and memory leak diagnostics.
* **Failure criteria:** Retained computation graphs or monotonic allocation growth in PyTorch allocator.
* **Compute implications:** Negligible (CPU unit tests run in < 2 seconds).
* **Dependencies:** None. Immediate next engineering step.

---

### Milestone 2: Constrained Algorithmic Pathfinding Benchmark Suite
* **Purpose:** Create procedural benchmark that demands candidate exploration, verification, and rollback.
* **Prerequisites:** Milestone 1 interfaces.
* **Scope:**
  * Implement `ConstrainedPathfindingBenchmark` generating procedural graphs with controllable dead-end density (e.g. 20% dead-end branches).
  * Implement deterministic oracle solver (A*/BFS) and rule verifier $V_{\text{det}}$ (checking edge connectivity and blockade status).
  * Establish train, validation, and firewalled test splits with known optimal path lengths.
* **Evaluation:** Exact match path completion rate, minimum step efficiency, and verification execution latency (< 0.5 ms/step).
* **Controls:** Random walk baseline, greedy non-backtracking agent.
* **Success criteria:** Zero unsolvable graphs; deterministic verifier executes in < 0.2 ms per step.
* **Failure criteria:** Verification rules ambiguous or task solvable by greedy feedforward policy without backtracking.
* **Compute implications:** Procedural dataset generation runs in < 30 seconds on CPU.
* **Dependencies:** Milestone 1 completion.

---

### Milestone 3: Hypothesis Proposal & Diversity Exploration (PI Research Gate 1)
* **Purpose:** Resolve the fundamental scientific failure of Phase 2 (noise perturbation collapse). Determine how to generate diverse, non-duplicate candidate transitions in continuous state.
* **Prerequisites:** Milestones 1 and 2.
* **Scope:**
  * Formulate and evaluate two structured candidate proposal mechanisms:
    1. *Action-conditioned proposal:* Candidate branches explicitly conditioned on discrete action embeddings ($K$ top actions).
    2. *Learned orthogonal perturbation:* Generating candidate state offsets $\delta^{(k)}$ via orthogonal projection heads.
  * Measure branch diversity and duplicate trajectory rates across $K \in \{2, 4\}$.
* **Evaluation:**
  * Branch Diversity: Mean pairwise cosine distance between candidate branches ($\bar{d}_{\text{cos}} \ge 0.20$ required).
  * Duplicate Trajectory Rate: Percentage of identical candidate outputs ($< 30\%$ required).
* **Controls:** Matched Gaussian noise injection ($\delta \sim \mathcal{N}(0, \sigma^2)$) as negative control; single-trajectory $K=1$ control.
* **Success criteria:** Preregistered threshold: $\bar{d}_{\text{cos}} \ge 0.20$ and duplicate rate $< 25\%$ across 5 seeds, statistically outperforming Gaussian noise ($p < 0.01$).
* **Failure criteria:** Candidate branches collapse into identical outputs or degenerate into destructive representations.
* **Compute implications:** Lightweight pilot training (~2,000 updates, 5 seeds, ~15 minutes on RX 6600M / CPU).
* **Dependencies:** Explicit PI authorization of proposal specification and preregistered thresholds.

---

### Milestone 4: End-to-End Latent State Machine Validation (PI Research Gate 2)
* **Purpose:** Empirically validate the complete workflow (Encoder $\to$ Speculative Isolation $\to$ Recurrent Core $\to$ Deterministic Verifier $\to$ Transaction Commit/Rollback).
* **Prerequisites:** Milestone 3 passing Gate 1.
* **Scope:**
  * Train full model ($K=2..4$) on Constrained Pathfinding benchmark.
  * Evaluate error containment and dead-end recovery.
* **Evaluation:**
  * Task Success Rate: Exact match arrival at goal node.
  * Recovery Rate: Percentage of successful goal completions after entering $\ge 1$ dead end.
  * State Pollution Index (SPI): Cosine similarity between master state $S_t$ before entering dead end and after rollback (must equal 1.000).
  * Compute-adjusted efficiency: Task completion per total executed FLOPs.
* **Controls:**
  * Control 1: Parameter-matched Vanilla GRU ($K=1$).
  * Control 2: Compute-matched unrolled GRU (given $K \times$ compute steps).
  * Ablation: Ungated model (state updates unconditionally without transaction rollback).
* **Success criteria:** Preregistered threshold: Gated model achieves statistically significant higher completion rate ($p < 0.01$, paired t-test) and recovery rate over compute-matched $K=1$ GRU and ungated ablation across 5 seeds.
* **Failure criteria:** Gated rollback fails to improve continuation accuracy over ungated baseline, or transaction gate causes permanent deadlock/stagnation.
* **Compute implications:** 5-seed training run (~3,000 updates, ~45 minutes on RX 6600M).
* **Dependencies:** Formal PI review of Milestone 3 report and authorization of Milestone 4 protocol.

---

### Milestone 5: Recursive Cognition & Adaptive Halting Integration (PI Research Gate 3)
* **Purpose:** Add weight-tied recurrence ($R > 1$) and PonderNet adaptive halting to dynamically modulate compute per transition.
* **Prerequisites:** Milestone 4 validation.
* **Scope:** Integrate `CognitionCore` and learned halting gate $p_{t,r}$. Evaluate on mixed-difficulty pathfinding graphs.
* **Evaluation:** Accuracy retained vs mean inference steps $\bar{R}$.
* **Controls:** Fixed-depth baseline ($R=4$).
* **Success criteria:** Retains $\ge 95\%$ accuracy of fixed $R=4$ while reducing average executed steps by $\ge 25\%$ on simple graphs.
* **Failure criteria:** Halting controller collapses to constant 1 or constant $R_{\max}$.
* **Compute implications:** ~1 hour training across 5 seeds.
* **Dependencies:** Milestone 4 completion.

---

## 7. Assessment of Phase 1.6 Stage 3 Composed Sequence Benchmark

**Question:** Should Phase 1.6 Stage 3 remain the next milestone?

**Assessment: Phase 1.6 Stage 3 is insufficient and should NOT remain on the critical path to the target architecture.**

### Grounded Rationale:
1. **Narrow model-specific scope:** Stage 3 was designed to evaluate whether `CyclicMGUPhase16Model` can chain multiple modular additions (e.g. $(a + b + c) \bmod 10$ across 5–10 steps) under the joint supervision objective established in `EXP-CYCLIC-PERSIST-01`.
2. **Zero workflow overlap:** Stage 3 contains:
   * No speculative candidate branching ($K=1$).
   * No deterministic transition verifier.
   * No transaction commit gate.
   * No dead ends, alternatives, or state rollback.
3. **Diminishing scientific return:** Executing Stage 3 would consume substantial protocol formulation, runner implementation, and compute to answer a narrow recurrence horizon question about monolithic Givens-rotation cells. It does not advance the core research question: *Can a latent state machine with isolated exploration and verifier-gated commits outperform standard models?*
4. **Recommendation:** De-prioritize Stage 3. Re-scope it as an optional, secondary model diagnostic. Redirect engineering and research effort immediately to Milestone 1 (State Isolation & Transaction Engine) and Milestone 2 (Constrained Pathfinding Benchmark).

---

## 8. Recommendations on the Five Pending PI Decisions in PLAN.md

Detailed evaluation and recommendations for the five unresolved decisions currently listed in `PLAN.md`:

### 1. Post-Ablation Strategy
* **Recommendation: Formally close the primitive modular addition 2×2 supervision grid.**
* **Rationale:** The 2×2 factorial design across terminal, destination, and persistence supervision is empirically complete:
  * Terminal only (`EXP-CYCLIC-ADD-01`): 0.0625 (`FAIL_REJECTION`).
  * Destination only (`EXP-CYCLIC-STEPWISE-01`): 0.6750 (`INCONCLUSIVE_INTERMEDIATE`, $r0$ read asymmetry).
  * Joint destination + persistence (`EXP-CYCLIC-PERSIST-01`): 0.9875 (`BEHAVIORAL_PASS_CANONICAL_FULL`).
  * Persistence only (`EXP-CYCLIC-PERSIST-ABL-01`): 0.5375 (`INCONCLUSIVE_INTERMEDIATE`, $r1$ write ungrounded).
* The scientific conclusion is definitive: continuous monolithic gating requires joint destination and persistence auxiliary supervision to preserve multi-register coordinates. Further hyperparameter sweeps or coefficient tuning on 3-step modular addition yield no new architectural insights.

### 2. Phase 1.6 Stage 3 Composed Sequence Benchmark Formulation
* **Recommendation: Do not formulate Stage 3 at this time.**
* **Rationale:** As established in Section 7, Stage 3 does not advance the state-machine architecture. Deferring Stage 3 avoids allocating project bandwidth to straight-line RNN evaluation.

### 3. Protected TEST-HELDOUT Evaluation ($b \in \{0, 5\}$)
* **Recommendation: Maintain strict firewall; do NOT evaluate protected TEST at this time.**
* **Rationale:** The protected TEST partition is governed by a strict single-evaluation firewall. Evaluating it on `CyclicMGUPhase16Model` now would permanently burn the held-out partition on a narrow 3-step arithmetic model with a fixed analytical ALU. The firewall should be preserved until an integrated, end-to-end latent state machine candidate is ready for definitive out-of-distribution evaluation.

### 4. Official Runner Overwrite Hardening
* **Recommendation: AUTHORIZE immediately.**
* **Rationale:** In `EXP-CYCLIC-PERSIST-ABL-01`, the official runner was flagged for lacking an explicit overwrite refusal safeguard when executing official runs. Adding a defensive check (`if os.path.exists(output_path) and not args.allow_overwrite: raise FileExistsError`) to runner scripts is standard research governance and software hygiene. It prevents accidental overwrites of hash-pinned artifacts.

### 5. Pre-existing Unrelated Test Drift — Experiment L
* **Recommendation: Authorize a dedicated documentation-governance audit and amendment.**
* **Rationale:** `tests/test_experiment_l.py::test_stored_artifact_is_reproducible_from_checkpoints` fails due to pre-existing artifact reproduction drift dated 2026-10-07. Per `docs/DOCUMENTATION_POLICY.md` §4, historical records must not be silently modified. A read-only reproduction audit followed by a dated amendment in the historical baseline ledger will resolve the test failure cleanly without altering research conclusions.

---

## 9. Risks, Unresolved Questions, & Information Requiring PI Judgment

### 9.1 Critical Scientific & Engineering Risks
1. **Continuous Candidate Branch Collapse (High Risk):**
   * *Risk:* If continuous candidate proposals are insufficiently distinct, the model explores identical states, wasting compute without search benefit.
   * *Mitigation:* Explicitly enforce branch diversity loss during training and evaluate discrete action-conditioned proposals in Milestone 3 before full integration.
2. **Deterministic Verifier Latency Overhead (Moderate Risk):**
   * *Risk:* Programmatic verification calls inside the inner PyTorch loop could degrade training throughput.
   * *Mitigation:* Ensure verification rules are vectorized over batch tensors in PyTorch or NumPy. Milestone 2 pre-flight gate requires $< 0.5\text{ ms}$ per step.
3. **Transactional Deadlock / Stagnation (Moderate Risk):**
   * *Risk:* If all $K$ candidates are rejected repeatedly, rollback $S_{t+1} \leftarrow S_t$ causes an infinite loop or zero progress.
   * *Mitigation:* Define maximum rollback retry budget (e.g. 3 attempts), after which an emergency fallback or termination token is triggered.
4. **Parameter & Compute Overhead (Moderate Risk):**
   * *Risk:* Maintaining $K$ candidate branches multiplies forward FLOPs by $K$. If the state machine does not outperform a compute-matched baseline with $K \times$ depth or parameters, the architectural thesis fails.
   * *Mitigation:* Enforce strict compute-matched baselines at every research gate (Milestones 4 and 5).

### 9.2 Key Decisions Requiring Explicit PI Judgment
1. **Task Domain Authorization:** Does the PI approve pivoting from modular addition to Constrained Graph Pathfinding as the primary end-to-end evaluation vehicle?
2. **Branching Formulation:** Does the PI prefer discrete action-conditioned proposals or continuous orthogonal learned perturbation for Milestone 3 preregistration?
3. **Execution Sequencing:** Does the PI confirm executing Milestone 1 (code & unit tests) immediately, while holding experimental training until Milestone 3 protocol approval?

---

## 10. Explicit Declaration of Boundaries Respected

* **Zero models trained:** No training commands or subprocesses were invoked.
* **Zero experiment runners called:** No runners in `experiments/` were executed.
* **Zero model code modified:** No neural network code in `src/` was edited.
* **`latent-thing-git/` untouched:** Directory was completely excluded from all operations.
* **Protected TEST partition firewalled:** Operand combinations $b \in \{0, 5\}$ were not accessed, enumerated, or evaluated.
