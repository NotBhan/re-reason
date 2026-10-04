# Experiments and Evaluation

## 1. Synthetic Benchmark Suite

All core experiments in RE:Reason rely on procedural synthetic tasks. Synthetic benchmarks eliminate natural language ambiguity, tokenization artifacts, and internet training data contamination, while providing mathematically verifiable ground-truth transitions.

We do not call these benchmarks "reasoning" merely because they involve multiple steps. Each benchmark tests a specific operational property:

```
┌─────────────────────────────────┬───────────────────┬────────────────────────────────────────────────────────┐
│ Benchmark Family                │ Primary Phase     │ Operational Property Tested                            │
├─────────────────────────────────┼───────────────────┼────────────────────────────────────────────────────────┤
│ 1. Register Tracking            │ Phase 1 (Core)    │ State retention & sequential memory updates            │
│ 2. Parity DAGs                  │ Phase 2 (Recur)   │ Combinatorial depth evaluation via weight-tied steps   │
│ 3. Mixed Arithmetic             │ Phase 3 (Halt)    │ Compute allocation across variable-depth expressions   │
│ 4. Dead-End Mazes               │ Phase 4 (Branch)  │ Non-greedy search & branch exploration in mazes        │
│ 5. Constrained Path Verification│ Phase 5 & 6 (Ver) │ Rule compliance & candidate transition filtering       │
│ 6. Adversarial Long-Horizon     │ Phase 7 (Commit)  │ State integrity preservation under corrupted inputs    │
└─────────────────────────────────┴───────────────────┴────────────────────────────────────────────────────────┘
```

### Benchmark 1: Register Tracking
* Description: Tracks the values of 4 integer registers across $T=10$ sequential assign and swap operations (e.g., `R1 = 4`, `SWAP R1 R2`, `R3 = R1 + R4`).
* Input: Discrete vector encoding register indices and arithmetic operations.
* Target: Value of all 4 registers at step $T=10$.
* Loss Formulation: Terminal supervision at step $t=10$. Step-wise predictions and losses may be logged only as diagnostics.
* What it Actually Tests: Memory retention and sequential update fidelity in a persistent latent vector. Does not test complex reasoning.

### Benchmark 2: Parity DAGs
* Description: Evaluates binary Boolean expression trees (depths 2, 4, and 6) where each internal node is an XOR or AND operation over leaf values.
* Input: Flattened tree adjacency matrix and leaf truth values.
* Target: Single binary truth value at the root node.
* What it Actually Tests: Whether recursive forward passes evaluate compositional logic trees better than a shallow single-step pass.

### Benchmark 3: Mixed Arithmetic
* Description: Evaluates arithmetic expression graphs with variable branch depths (50% depth-1, 30% depth-3, 20% depth-6).
* Input: Expression graph vector.
* Target: Numeric output.
* What it Actually Tests: Whether adaptive halting reduces compute while preserving accuracy. Graph depth serves as an operational proxy for problem difficulty, though not identical to computational difficulty.

### Benchmark 4: Dead-End Mazes
* Description: Grid graph navigation where greedy shortest-path choices lead into walled dead ends, requiring exploration of alternative paths.
* Input: Local grid observation (current coordinate, wall adjacency, goal coordinate).
* Target: Sequence of directional movements to goal.
* What it Actually Tests: Whether parallel latent branching explores alternative paths that greedy single-trajectory decoding misses, evaluated without verifier intervention.

### Benchmark 5: Constrained Graph Pathfinding
* Description: Graph traversal where paths must obey strict adjacency rules and avoid dynamically placed forbidden obstacle nodes.
* Input: Graph edge list, start node, target node, forbidden node set.
* Target: Valid sequence of node transitions.
* What it Actually Tests: Verification efficiency and false acceptance rates when evaluating candidate actions and transitions against hard constraints.

### Benchmark 6: Adversarial Long-Horizon Tasks
* Description: 20-step sequential state tracking where an intentionally invalid, corrupted, or contradictory observation is injected at step $t=5$.
* Input: Sequence of state transition instructions with one corrupted step.
* Target: Correct final state at $t=20$.
* What it Actually Tests: Error containment. Verifies whether transactional rollback preserves downstream continuation accuracy after an invalid transition.

## 2. Baseline Models

To prevent fabricated wins, every architectural addition is benchmarked against compute- and parameter-matched baselines:

### Baseline A: Compact GRU
* Standard PyTorch `nn.GRUCell`.
* Serves as the established floor for recurrent latent-state memory. Parameter count matched to Phase 1 model ($\pm 5\%$).
* Matching Baseline A in Phase 1 demonstrates basic viability for state tracking, not architectural superiority.

### Baseline B: Small Transformer Decoder
* 2-to-4 layer causal Transformer decoder with multi-head attention.
* Parameter- and FLOP-matched to evaluate whether persistent latent recurrence can match standard attention over past context.

### Baseline C: Persistent Core (Phase 1)
* Minimal latent update: $R=1, K=1$, flat tensor $S \in \mathbb{R}^{B \times 256}$, terminal loss at $t=10$, fixed single pass, no branching, no verifier, no halting, no transactional commit. Parameter-matched to Baseline A (`nn.GRUCell`).

### Baseline D: Recursive Core (Phase 2)
* Weight-tied recursion: $R \in [1..8], K=1$.

### Baseline E: Branching Core (Phase 4)
* Parallel candidate exploration: $K \in [1..8]$, evaluated unverified under compute parity.

### Baseline F: Full Architecture (Phase 7)
* Complete pipeline: Branching + Dual Verifier + Transactional Commit Gate.

## 3. System Matrix & Ablation Design

```
┌───────────┬─────────────┬──────────────┬─────────────┬───────────┬────────────┬─────────────┐
│ System    │ State Type  │ Recur Depth  │ Halting     │ Branches  │ Verifier   │ Commit Gate │
├───────────┼─────────────┼──────────────┼─────────────┼───────────┼────────────┼─────────────┤
│ Baseline A│ GRU State   │ R = 1        │ Static      │ K = 1     │ None       │ Standard    │
│ Baseline B│ Attention   │ Fixed Layers │ Static      │ K = 1     │ None       │ Standard    │
│ System C  │ LatentState │ R = 1        │ Static      │ K = 1     │ None       │ Standard    │
│ System D  │ LatentState │ R in [1..8]  │ Static      │ K = 1     │ None       │ Standard    │
│ System D+ │ LatentState │ R in [1..8]  │ Adaptive    │ K = 1     │ None       │ Standard    │
│ System E  │ LatentState │ R in [1..8]  │ Adaptive    │ K in [2..8]None       │ Standard    │
│ System E+ │ LatentState │ R in [1..8]  │ Adaptive    │ K in [2..8]Det + Learn │ Gated       │
│ System F  │ LatentState │ R in [1..8]  │ Adaptive    │ K in [2..8]Det + Learn │ Transaction │
└───────────┴─────────────┴──────────────┴─────────────┴───────────┴────────────┴─────────────┘
```

## 4. Evaluation Metrics

### 4.1 Task Performance Metrics
* Exact Match Accuracy (%): Binary success rate across test instances.
* Continuation Accuracy (%): Step-by-step accuracy across sequence length $T$ to measure error decay over time.
* Recovery Rate (%): Proportion of episodes where the model successfully answers steps $t > t_{\text{corrupt}}$ following an injected error.
* Post-Corruption Performance: Accuracy curve on steps $t=6 \dots 20$ after error injection at $t=5$.
* Contamination Duration: Number of sequential steps required for the model to recover baseline accuracy after an error event.

### 4.2 Compute & Efficiency Metrics
* Parameter Efficiency: Task Accuracy divided by parameter count ($\% / \text{MParam}$).
* Inference FLOPs: Total floating-point operations per completed test episode.
* Latency (ms): Wall-clock time per step measured on the target AMD RX 6600M GPU.
* Peak VRAM Allocation (MB): High-water memory mark reported by PyTorch.
* Average Recursion Depth ($\bar{R}$): Mean recursive iterations executed before halting.

### 4.3 Adaptive Halting Evaluation
* Primary Metrics:
  * Task accuracy retained relative to fixed-depth baseline ($R=4$).
  * Total compute reduction (% reduction in FLOPs / executed steps).
  * Average executed steps $\bar{R}$ per problem category.
* Secondary Diagnostic:
  * Pearson correlation $r(\text{depth}, \bar{R})$ between problem depth and executed steps.

### 4.4 Branch Diversity Metrics
* Branch Diversity ($D_{\text{branch}}$): Mean pairwise cosine distance among $K$ candidate latent states:
  $$D_{\text{branch}} = \frac{2}{K(K-1)} \sum_{i < j} \left(1.0 - \frac{s^{(i)} \cdot s^{(j)}}{\|s^{(i)}\| \|s^{(j)}\|}\right)$$
* Duplicate Trajectory Rate (%): Percentage of candidate branches that produce identical typed decision outputs $\hat{y}$.

### 4.5 Verification Metrics
Verification is strictly defined over candidate transitions, outputs, and state invariants.

* Definitions:
  * Positive = Candidate is classified as valid/accepted.
  * Negative = Candidate is classified as invalid/rejected.
* False Acceptance Rate (FAR):
  $$\text{FAR} = \frac{\text{Invalid candidates incorrectly accepted}}{\text{All invalid candidates}}$$
* False Rejection Rate (FRR):
  $$\text{FRR} = \frac{\text{Valid candidates incorrectly rejected}}{\text{All valid candidates}}$$
* Expected Calibration Error (ECE): Bin-wise deviation between verifier confidence scores and true transition validity.
* Operational Rule: The learned verifier must **never** receive ground-truth answers during inference. Ground truth is used strictly by the evaluation harness.

### 4.6 State Integrity & Diagnostic Metrics
* State Pollution Index (SPI) [Diagnostic Geometric Metric]:
  $$\text{SPI} = 1.0 - \text{CosineSimilarity}(S_{\text{after\_error}}, S_{\text{clean\_reference}})$$
* Critical Interpretation Rule: SPI measures geometric latent displacement; it does **not** prove semantic correctness or state integrity. Primary evidence for state integrity must come from downstream continuation accuracy and recovery rates.

## 5. Experimental Methodology & Rigor

### Multi-Seed Testing
Every primary comparison must be run across a minimum of 5 independent random seeds (different parameter initializations and procedural dataset seeds). Results must be reported as $\text{Mean} \pm \text{Standard Deviation}$ with 95% bootstrap confidence intervals.

### Explicit Compute Matching Rules
Whenever $K > 1$ is compared against $K = 1$, total inference compute must be explicitly equalized:
* Compare $K=4$ against a $K=1$ baseline given equivalent total recursive computation (e.g., $4\times$ deeper recursion).
* Compare $K=4$ against four sequential $K=1$ runs (ensemble control).
* Do not claim a branching advantage simply because four parallel branches received $4\times$ the compute.
* Always report FLOPs, latency, parameter count, and memory alongside accuracy.

### Parameter Parity Rules
Models being compared must be within $\pm 5\%$ parameter count, unless explicitly running a parameter scaling ablation.

### Statistical Rigor vs. Engineering Decision Rules
* Hypothesis testing uses paired two-tailed t-tests (or non-parametric Wilcoxon signed-rank tests for non-normal metrics) with a significance threshold of $p < 0.01$.
* Numerical go/no-go thresholds are predefined project decision rules for triage, not universal scientific laws.

## 6. Examples of Valid vs. Invalid Comparisons

### Invalid Comparison 1:
* Invalid: Running $K=4$ branches and comparing accuracy against a single-pass $K=1$ model with no compute adjustment, then claiming "branching improves intelligence".
* Valid: Comparing $K=4$ against $K=1$ where $K=1$ is allocated an equal FLOP budget (e.g., $4\times$ deeper sequential steps or an ensemble of 4 runs), reporting accuracy, latency, and memory.

### Invalid Comparison 2:
* Invalid: Measuring SPI (cosine distance) after an adversarial injection, observing high similarity, and claiming "the state is completely uncorrupted".
* Valid: Measuring whether the model actually solves steps $t=6 \dots 20$ after the injection, with SPI reported only as supplementary geometric context.

### Invalid Comparison 3:
* Invalid: Letting the learned verifier access ground-truth graph labels at test time, then reporting 99% task accuracy.
* Valid: The learned verifier only accesses the candidate latent state $s^{(k)}$ and observation embedding $e_t$, with ground truth used strictly by the evaluation harness.

For engineering implementation details and memory management, see [docs/05_engineering_and_implementation.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/05_engineering_and_implementation.md).  
For the explicit kill criteria and research risks, see [docs/06_risks_research_questions_and_future.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/06_risks_research_questions_and_future.md).
