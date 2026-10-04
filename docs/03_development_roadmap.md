# Development Roadmap

This document defines the 11 sequential implementation phases for RE:Reason. In accordance with project philosophy, each phase tests a single architectural mechanism in isolation. Complexity from later phases is strictly excluded from earlier phases.

```
Phase 0: Architecture Contracts & Test Infrastructure
   │
Phase 1: Minimal Persistent-State Core (R=1, K=1)
   │
Phase 2: Fixed-Depth Recursive Cognition (R in [1, 2, 4, 8])
   │
Phase 3: Adaptive Halting (PonderNet)
   │
Phase 4: Parallel Hypothesis Branching (K in [1, 2, 4, 8] - Unverified)
   │
Phase 5: Deterministic Verification (Transition Rules)
   │
Phase 6: Learned Verifier (Auxiliary Trajectory Scorer)
   │
Phase 7: Transactional State Commit & Error Containment
   │
Phase 8: Test-Time Latent Adaptation (Optional Research Branch)
   │
Phase 9: Optional Frozen Multimodal Adapters
   │
Phase 10: Future NLP & Semantic Interface Boundary
```

### Staging Rationale
The sequencing of Phases 4 through 7 is deliberate:
* Phase 4 tests branching and search independently, without verification.
* Phase 5 introduces deterministic verification of candidate transitions.
* Phase 6 evaluates whether an auxiliary learned verifier adds value over deterministic checks.
* Phase 7 connects verified candidates to transactional state commit and evaluates error containment.
Evaluating these mechanisms separately ensures that failures in one component (e.g., verifier reward hacking) are not conflated with the behavior of other components (e.g., hypothesis branching).

Any numerical go/no-go threshold below is a **predefined project decision rule** for engineering triage, not a universal law of machine intelligence.

## Phase 0: Architecture Contracts and Test Infrastructure

* Objective: Establish typed contracts, abstract interfaces, and continuous integration tests before model development.
* What is Implemented: Abstract base classes (`BaseTask`, `BaseStateContainer`, `BaseCognitionCore`, `BaseVerifier`), test runner, PyTorch memory leakage diagnostics.
* Memory Diagnostics: Tests check for retained tensor references, uncollected autograd computation graphs, reference cycles, monotonic allocation growth across repeated runs, and proper cleanup of speculative branches. Allocator statistics are used as diagnostics rather than expecting reserved GPU memory to return to zero.
* Deliberately Excluded: Any neural network implementations, benchmark task code.
* Benchmark: Unit test suite with mock objects.
* Metrics: Test coverage (100% on interfaces), execution time, absence of retained tensor references across repeated loop iterations.
* Baseline: Python standard library unittest / pytest mocks.
* Experiment: Run 1,000 iterations of mock state snapshot and rollback; verify that memory allocations do not display monotonic growth.
* Go/No-Go Decision Rule: All tests pass cleanly on both CPU and AMD ROCm GPU; memory allocation diagnostics remain flat across 1,000 iterations.
* What Failure Means: Test infrastructure or tensor isolation logic leaks references or computation graphs.
* What Happens if it Fails: Fix reference cycles and container implementations before touching any neural layers.

## Phase 1: Minimal Persistent-State Core

* Objective: Determine whether the proposed persistent-state update formulation can achieve basic sequential state tracking without being materially worse than a standard GRU at comparable parameter count.
* What is Implemented: Minimal core ($R=1, K=1$): $e_t = \text{Encoder}(x_t)$, $S_t = \text{LayerNorm}(S_{t-1} + f_\theta(S_{t-1}, e_t))$, $\hat{y}_t = \text{Head}(S_t)$.
* Deliberately Excluded: Recursion ($R>1$), branching ($K>1$), adaptive halting, verification, transactional commit, multimodality, NLP.
* Benchmark: Register Tracking (4 registers, sequence length $T=10$ assignments/swaps).
* Metrics: Exact match accuracy across all registers at step $T=10$, parameter count, training wall-clock time.
* Baseline: Compact `nn.GRUCell` with matched parameter count ($\sim 250\text{K}$ parameters) trained under identical optimizer, batch size, and seed count.
* Experiment: Train System C ($R=1, K=1$) and Baseline A (GRU) across 5 random seeds.
* Go/No-Go Decision Rule: System C matches Baseline A within $\pm 2\%$ accuracy at equal parameter count.
* Interpretation of Success: Matching a GRU is strictly evidence that the proposed state update is viable for basic sequential state tracking. It is **not** evidence of architectural superiority.
* What Failure Means: The chosen persistent state update equation is fundamentally less expressive or less learnable than a standard GRU cell.
* What Happens if it Fails: Re-examine the state update equation (e.g., test gated residual addition vs. standard GRU gating) or terminate the persistent core approach.

## Phase 2: Fixed-Depth Recursive Cognition

* Objective: Test whether applying the same cognition module $f_\theta$ repeatedly ($R > 1$) improves performance on deeper problems without adding parameters.
* What is Implemented: Unrolled recursive iteration for $R \in \{1, 2, 4, 8\}$ using tied weights $\theta$. Backpropagation through time across recursive steps.
* Deliberately Excluded: Adaptive halting (depth is fixed per run), branching, verification, transactional commit.
* Benchmark: Parity Decision Trees (binary logic DAGs evaluated at depths 2, 4, and 6).
* Metrics: Exact match accuracy at root node, inference FLOPs, gradient norm during training.
* Baseline: System C ($R=1$) and an unrolled feedforward network with equivalent total FLOPs.
* Experiment: Train and evaluate System D across $R \in \{1, 2, 4, 8\}$ over 5 seeds.
* Go/No-Go Decision Rule: System D ($R=4$) achieves a statistically significant accuracy advantage ($p < 0.01$, paired t-test) over System C ($R=1$) on depth-4 DAGs at equal parameter count.
* What Failure Means: Weight-tied recursion does not provide genuine computational depth on this task family; additional unrolled iterations simply add gradient noise or vanishing signals.
* What Happens if it Fails: Remove depth recursion. Maintain static depth ($R=1$) for remaining phases.

## Phase 3: Adaptive Halting

* Objective: Test whether a learned halting gate (PonderNet formulation) can dynamically allocate compute to retain task accuracy while reducing average executed steps.
* What is Implemented: PonderNet halting head $p_{t,r} = \sigma(W_h s_{t,r} + b_h)$ with geometric prior regularization loss.
* Deliberately Excluded: Branching, verification, transactional commit.
* Benchmark: Mixed-Depth Arithmetic DAGs (50% depth-1, 30% depth-3, 20% depth-6).
* Evaluation Criteria:
  * Primary Evaluation: Task accuracy retained relative to fixed-depth baseline ($R=4$), reduction in total inference compute/FLOPs, average executed steps $\bar{R}$.
  * Secondary Diagnostic: Correlation between task depth and halting steps. Note that graph depth is only an operational proxy for problem difficulty, not identical to computational difficulty.
* Baseline: System D with fixed $R=4$.
* Experiment: Train adaptive model across 5 seeds. Measure accuracy, total compute, and step distribution.
* Go/No-Go Decision Rule: Retains $\ge 95\%$ of fixed-depth accuracy while reducing mean inference steps by at least 25% on easy problems, with no step collapse to constant 1 or constant $R_{\max}$.
* What Failure Means: The halting controller fails to modulate compute constructively, either collapsing to fixed depth or degrading task accuracy.
* What Happens if it Fails: Discard adaptive halting. Revert to fixed compute depth ($R=\text{constant}$) for future phases.

## Phase 4: Parallel Hypothesis Branching

* Objective: Test whether branching and exploring $K$ alternative candidate working states provides search value over single-trajectory decoding under equalized compute budgets.
* Staging Note: Phase 4 evaluates branching in isolation. It does not test correctness or verification. Candidate selection uses unverified confidence scores $c^{(k)}$ to evaluate search independently.
* What is Implemented: Hypothesis brancher generating $K \in \{1, 2, 4, 8\}$ candidate states via isolated tensor cloning.
* Critical Caveat: The equation $S + \delta$ is treated only as one candidate implementation. Meaningful branch generation remains an engineering unknown. Action-conditioned proposals and learned perturbation generators will both be tested.
* Deliberately Excluded: Verification (deterministic or learned), transactional commit.
* Benchmark: Dead-End Mazes (grid graphs where greedy local steps lead to dead ends requiring exploration of alternatives).
* Metrics: Maze completion rate, Branch Diversity (mean pairwise cosine distance between branches), Duplicate Trajectory Rate (% identical outputs), total compute-adjusted efficiency.
* Compute Matching: $K=4$ must be compared against compute-matched $K=1$ controls (e.g., $K=1$ given $4\times$ deeper recursion or an ensemble of 4 runs).
* Baseline: Compute-matched System D ($K=1$).
* Experiment: Compare $K=1, 2, 4, 8$ on identical maze suites under compute parity across 5 seeds.
* Go/No-Go Decision Rule: $K=4$ achieves statistically significant higher completion rates than compute-matched $K=1$, with Duplicate Trajectory Rate $< 30\%$ and Branch Diversity $> 0.20$.
* What Failure Means: Candidate branches collapse into duplicate trajectories or behave as unhelpful random noise, failing to justify branching overhead.
* What Happens if it Fails: Kill parallel hypothesis branching. Maintain a single trajectory ($K=1$).

## Phase 5: Deterministic Verification

* Objective: Implement hard programmatic constraint checks on candidate transitions and outputs before state selection.
* What is Implemented: `DeterministicVerifier` validating domain invariants, physical collision boundaries, and transition legality: $V_{\text{det}}(\hat{y}^{(k)}, x_t) \in \{0, 1\}$.
* Deliberately Excluded: Learned verifier networks, transactional rollback (selection picks the highest confidence valid candidate).
* Benchmark: Constrained Graph Pathfinding (paths must avoid dynamic obstacle nodes).
* Metrics: Constraint violation rate (%), candidate survival rate (%), execution latency overhead (ms).
* Baseline: System E without verification (unfiltered top confidence selection).
* Experiment: Measure violation rate and task success across 1,000 test graphs.
* Go/No-Go Decision Rule: Deterministic verifier achieves 0.0% constraint violation rate on surviving candidates with latency overhead $< 0.5\text{ ms}$ per step.
* What Failure Means: Rule checking is too slow or constraint checks fail to prevent invalid transitions.
* What Happens if it Fails: Refactor programmatic rules for vectorized execution.

## Phase 6: Learned Verifier

* Objective: Evaluate an auxiliary neural trajectory scorer $V_{\text{learned}}(s^{(k)}, e_t) \in [0, 1]$ that estimates candidate transition consistency when deterministic rules are insufficient.
* Operational Rules: The learned verifier evaluates candidate outputs, transitions, and state consistency. It must **never** receive ground-truth answers during inference. Ground truth is restricted strictly to the evaluation harness.
* What is Implemented: Auxiliary critic network trained on state-transition consistency.
* Benchmark: Constrained Graph Pathfinding with injected corrupted/suboptimal candidate states.
* Comparisons Evaluated:
  * Setup A: Deterministic verification only.
  * Setup B: Learned verification only.
  * Setup C: Deterministic hard filter + Learned ranking.
* Metrics: False Acceptance Rate (FAR), False Rejection Rate (FRR), Expected Calibration Error (ECE), susceptibility to shortcut exploitation.
* Go/No-Go Decision Rule: Setup C achieves FAR $< 10\%$ and FRR $< 15\%$ without exhibiting shortcut reward-hacking.
* What Failure Means: The learned verifier is easily deceived by corrupted states or overfits to superficial statistical cues.
* What Happens if it Fails: Discard the learned verifier. RE:Reason continues using deterministic verification exclusively.

## Phase 7: Transactional State Commit & Error Containment

* Objective: Test whether rejecting unverified candidates and rolling back state ($S_{t+1} \leftarrow S_t$) prevents error contamination during long-horizon computation.
* What is Implemented: Transaction commit gate: if all candidate branches fail verification, abort step and retain $S_t$ (master state unchanged; rejected candidate not committed).
* Deliberately Excluded: Weight updates during inference.
* Benchmark: 20-step Sequential Planning with injected adversarial/corrupted inputs at step $t=5$.
* Metrics: Downstream task accuracy at steps $t=6 \dots 20$, recovery rate (%), continuation accuracy, State Pollution Index (SPI) as a supplementary geometric diagnostic.
* Baseline: Ungated System (candidate state always updates master state regardless of verification).
* Experiment: Run 500 adversarial episodes comparing Gated vs. Ungated master state across 5 seeds.
* Go/No-Go Decision Rule: Gated system maintains statistically significant higher downstream continuation accuracy ($p < 0.01$) over the ungated baseline across steps 15-20.
* What Failure Means: Transactional rollback does not preserve downstream recovery, or state abort causes permanent deadlock/stagnation.
* What Happens if it Fails: Terminate the transactional state commit hypothesis; investigate continuous soft gating instead of binary rollbacks.

## Phase 8: Test-Time Latent Adaptation (Optional Research Branch)

* Status: Explicitly an **optional research branch**. Failure in Phase 8 does not invalidate RE:Reason as a whole if Phases 1–7 succeed.
* Objective: Evaluate whether freezing model weights $\theta$ and performing gradient descent directly on the latent state vector $S_t$ via verifier loss improves performance on out-of-distribution problems.
* Scope Boundaries:
  * Latent-state adaptation: Optimization of continuous vector $S_t$ during inference.
  * Weight updates: Parameter modifications ($\Delta \theta$). Strictly excluded.
  * Continual learning: Permanent updates across task distributions. Strictly excluded.
* Benchmark: Out-of-distribution algorithmic DAGs.
* Metrics: Accuracy improvement vs. optimization step count, optimization latency, stability of latent norms.
* Baseline: System F without test-time latent optimization.
* Experiment: Sweep latent optimization steps ($N \in \{1, 5, 10\}$) across 5 seeds.
* Go/No-Go Decision Rule: Latent optimization achieves statistically significant accuracy gain ($p < 0.01$) without causing latent norm divergence.
* What Failure Means: Gradient descent in latent space leads to degenerate vectors or reward hacking.
* What Happens if it Fails: Discard test-time latent adaptation.

## Phase 9: Optional Frozen Multimodal Adapters

* Objective: Test whether the RE:Reason core can process heterogeneous observation vectors from external frozen perception models.
* What is Implemented: Linear/MLP input adapter mapping frozen external features (e.g., ResNet-18 or AST audio embeddings) into latent dimension $D$.
* Deliberately Excluded: Multimodal representation learning, end-to-end vision/audio training.
* Benchmark: Multimodal state tracking (tracking register values encoded as visual glyphs or audio tones).
* Metrics: Tracking accuracy, adapter parameter overhead, latency.
* Baseline: Direct end-to-end recurrent baseline.
* Experiment: Evaluate whether core inference dynamics remain stable when driven by external embeddings.
* Go/No-Go Decision Rule: Core achieves within $5\%$ of clean symbolic performance when using frozen visual embeddings.
* What Failure Means: The latent core is overly sensitive to embedding noise from perception adapters.
* What Happens if it Fails: Restrict RE:Reason to structured symbolic domain.

## Phase 10: Future NLP & Semantic Interface Boundary

* Objective: Specify the input/output interface schema for future external semantics integration ("WHAT / WHY Semantics").
* What is Implemented: Type-checked schema definitions for consuming structured semantic operator vectors (WHAT, WHY, WHO, WHERE, WHEN, HOW, WHICH).
* Deliberately Excluded: Any NLP model, tokenization, text decoding, or Swarm Semantics implementation.
* Deliverable: Interface specification document only.

For detailed benchmark specifications and metrics, see [docs/04_experiments_and_evaluation.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/04_experiments_and_evaluation.md).  
For the technical engineering and testing strategy, see [docs/05_engineering_and_implementation.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/05_engineering_and_implementation.md).
