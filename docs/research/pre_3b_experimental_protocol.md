# Pre-3B Experimental Protocol Specification — Neural Hypothesis Branching & Baseline Comparisons

**Date:** 2026-10-17  
**Status:** Finalized Research Protocol Specification (Clarification Audit Complete — Implementation & Experimentation Strictly Unauthorized)  
**Governing Documents:** `AGENTS.md`, `docs/DOCUMENTATION_POLICY.md`, `docs/reports/architecture_gap_audit_and_milestone_proposal.md`  
**Target Milestone:** Pre-3B Protocol Finalization (Prerequisite for Milestone 3B Authorization)  
**Authoritative Reference:** `docs/research/pre_3b_experimental_protocol.md`

---

## 1. Executive Context & Objective

Following the independent code audit and formal acceptance by the Principal Investigator (PI) of **Milestone 3A** (`src/re_reason/hypothesis/pathfinding_branching.py`, `src/re_reason/hypothesis/two_phase_orchestrator.py`, and additive `LatentState.restore_checkpoint` in `src/re_reason/state/container.py`), this document establishes the authoritative, finalized experimental protocol for evaluating RE:Reason against competitive baselines.

The core research question of RE:Reason is:
> *Can a small, weight-tied latent recurrent core with isolated hypothesis exploration and verifier-gated state commits solve multi-step algorithmic problems with higher parameter efficiency and error containment than standard recurrent baselines?*

This protocol formalizes the PI-approved experimental-design decisions governing:
1. Baseline comparator specifications and compute budget matching.
2. The primary Goal-Reaching Rate (GRR) metric and statistical decision rules.
3. Decoupled, independent random number generator (RNG) namespaces for training, benchmark generation, and stochastic rollout sampling.
4. The initial training objective (BFS shortest-path behavioral cloning) with explicit teacher-forcing semantics and retry/abort supervision rules.
5. Operational definitions for descriptive, non-gating secondary metrics and dual compute accounting.

**Strict Governance Boundary:** This specification is documentation-only. It authorizes **no** neural training, experiment execution, runner implementation, baseline coding, or held-out test evaluation.

---

## 2. Approved Experimental Design & Baseline Comparators

The experimental evaluation tests RE:Reason against three distinct comparators, each addressing an isolated scientific question:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              RE:REASON COMPARATOR ROSTER                               │
├──────────────────────────────┬────────────────────────────┬────────────────────────────┤
│ Comparator                   │ Primary Scientific Purpose │ Evaluation Mechanism       │
├──────────────────────────────┼────────────────────────────┼────────────────────────────┤
│ 1. Primary Baseline:         │ Error containment against  │ Single-trajectory residual │
│    Un-Gated Recurrent        │ un-gated state corruption  │ recurrent core without     │
│    Residual Baseline         │                            │ branching/gate (T_max = 24)│
├──────────────────────────────┼────────────────────────────┼────────────────────────────┤
│ 2. Additional Comparator:    │ Speculative simulation vs. │ Policy-level action mask   │
│    Action-Masking Recurrent  │ shallow action filtering   │ on legal moves             │
│    Baseline                  │                            │ (T_max = 24)               │
├──────────────────────────────┼────────────────────────────┼────────────────────────────┤
│ 3. Additional Comparator:    │ Intra-step branching vs.   │ Two independent stochastic │
│    Episode-Budget Matched    │ recurrent compute budget   │ rollouts with fixed τ=1.0  │
│    (Two-Rollout) Baseline    │                            │ (T1=26, T2=27)             │
└──────────────────────────────┴────────────────────────────┴────────────────────────────┘
```

*Scientific Exclusion:* The sequential **Beam-3 Recurrent Baseline** remains **permanently excluded** as an algorithmically redundant control (it generates identical candidate tensors using the same shared core and verifier, differing solely in loop vectorization).

### 2.1 Primary Baseline: Un-Gated Recurrent Residual Baseline (Ablation Control)
* **Architecture:** To isolate the empirical effect of speculative hypothesis branching, transition verification, and transaction rollback while holding parameter count, receptive field, and recurrent state capacity strictly constant, the primary baseline employs the **exact identical neural modules** as RE:Reason:
  - Shared node embedding table: `nn.Embedding(12, 32)` (384 params).
  - Observation encoder: `ObservationEncoder(220 -> 128 -> 256)` (61,824 params).
  - Policy head: `Linear(512 -> 12)` (6,156 params).
  - Recurrent transition core: `Linear(544 -> 128) -> GELU -> Linear(128 -> 256) -> LayerNorm(256)` (103,296 params).
* **Parameters:** Exactly **171,660 trainable parameters** (matched bit-for-bit to RE:Reason).
* **Execution:** Single-trajectory execution ($K=1$, $T_{\max} = 24$ steps). Actions are selected directly by policy argmax, evaluated through the recurrent transition core, and committed unconditionally to the hidden state ($S_{t+1} = \text{Core}(S_t, e_t, E_{\text{node}}[a_t])$) with zero verification, zero candidate branching, and zero transaction rollback.
* **Inference Compute:** $341,016\text{ FLOPs}$ per decision step (122,880 encoder + 12,312 policy head + 205,824 recurrent core); nominal peak $8.19\text{ MFLOPs}$ per episode ($24 \times 341\text{K}$).
* **Scope of Claims:** Tests whether verifier-gated speculative commitment and transaction rollback protect the recurrent latent state from fatal error accumulation and state pollution compared to an identical single-trajectory recurrent architecture. This comparator is strictly an **un-gated recurrent residual baseline** and must **not** be described as a conventional GRU.

### 2.2 Additional Comparator: Action-Masking Recurrent Baseline
* **Architecture:** Identical parameter-matched residual recurrent architecture (exactly 171,660 trainable parameters).
* **Execution:** Single-trajectory execution ($K=1$, $T_{\max} = 24$ steps). At each step, policy logits are masked with the public `move_mask` prior to argmax:
  $$a_t = \arg\max_{a \in \text{AllowedMoves}(x_t)} \pi(a \mid S_t, e_t)$$
  The chosen action is committed unconditionally to state ($S_{t+1} = \text{Core}(S_t, e_t, E_{\text{node}}[a_t])$).
* **Inference Compute:** $341,016\text{ FLOPs}$ per decision step; nominal peak $8.19\text{ MFLOPs}$ per episode.
* **Scope of Claims:** Tests whether simulating candidate successor states in latent space and verifying transitions offers an empirical advantage over shallow, policy-level action filtering at the input boundary.

### 2.3 Additional Comparator: Episode-Budget Matched Baseline (Two-Rollout Recurrent Baseline)
To test intra-step hypothesis branching ($K=3$) against recurrent exploration under a comparable computational budget, the PI has approved the **Two-Rollout Recurrent Baseline**:

* **Architecture & Parameters:** Preserves the parameter-matched residual recurrent architecture (exactly 171,660 trainable parameters). The wider recurrent comparator (Option C, $\approx 390\text{K}$ parameters) is formally **deferred** to avoid confounding compute matching with parameter capacity.
* **Transition Budget Allocation:** Allocates a nominal budget of **53 recurrent transitions** across two independent trajectories:
  - Trajectory 1: Nominal horizon $T_1 = 26$ transitions ($26 \times 341,016 = 8,866,416\text{ FLOPs}$).
  - Trajectory 2: Nominal horizon $T_2 = 27$ transitions ($27 \times 341,016 = 9,207,432\text{ FLOPs}$).
  - **Total Nominal Episode Compute:** $8,866,416 + 9,207,432 = \mathbf{18,073,848\text{ FLOPs}} \approx \mathbf{18.07\text{ MFLOPs}}$.
  - *Comparison:* Directly comparable to RE:Reason's nominal peak budget of $24 \times 752,664 = \mathbf{18,063,936\text{ FLOPs}} \approx \mathbf{18.06\text{ MFLOPs}}$ (at $T_{\max}=24$, $K=3$, zero retries).
* **Static Horizon Partitioning:** Each rollout receives a fixed, independent step limit ($T_1=26, T_2=27$). Unused steps from an early-terminating Trajectory 1 are not dynamically banked into Trajectory 2, preserving deterministic step horizons.
* **Fixed Action-Selection Policy & Temperature ($\tau = 1.0$):**
  To ensure Trajectory 2 explores alternative paths rather than duplicating Trajectory 1 from the identical initial state, the Two-Rollout baseline employs stochastic action selection during evaluation:
  $$a_t \sim \text{Categorical}(\text{Softmax}(\text{logits} / \tau)) \quad \text{for } a \in \text{AllowedMoves}(x_t)$$
  The temperature is **fixed at $\tau = 1.0$** for the initial experiment. **No temperature sweep is permitted in the initial protocol.** Any subsequent temperature tuning would require a separately authorized experiment.
* **Trajectory-Selection Rule (Inference Time):**
  At episode termination, the agent selects the winning trajectory using purely public verification:
  1. Both trajectories are verified using `PathfindingVerifier` to confirm arrival at `goal_node` without invalid edges or blockades.
  2. If Trajectory 1 succeeds and Trajectory 2 fails $\implies$ Select Trajectory 1.
  3. If Trajectory 2 succeeds and Trajectory 1 fails $\implies$ Select Trajectory 2.
  4. If both trajectories succeed $\implies$ Select the shorter valid trajectory (fewer executed transitions). If lengths are tied, select Trajectory 1.
  5. If neither trajectory reaches the goal $\implies$ Record episode failure (GRR = 0).
  *Strict Prohibition:* Zero BFS oracle or ground-truth shortest-path information may be used to select trajectories at inference time.
* **Scope of Claims:** Tests whether intra-step parallel candidate branching and verifier gating provide algorithmic search advantages over allocating the equivalent compute budget to stochastic sequential exploration.

---

## 3. Observation Surface & Feature Encoding

### 3.1 Public Surface Guarantee
* **Benchmark:** Constrained Graph Pathfinding with Dynamic Blockades (`src/re_reason/benchmarks/pathfinding_benchmark.py`).
* **Environment Visibility:** Fully observable with strict public encapsulation:
  - *Public View (`PathfindingObservation`):* `current: int`, `goal: int`, and `effective_adjacency: tuple[tuple[int, ...], ...]`.
  - *Private Ground Truth (`PathfindingEpisode`):* Underlying graph topology, dynamic blockade subsets, and reference BFS oracle paths are strictly private.
  - No privileged graph topology, ground-truth blockade identities, or oracle shortest-path data are exposed to any agent or model.

### 3.2 220-Dimensional Feature Transformation Semantics
The observation representation implemented in `src/re_reason/hypothesis/pathfinding_branching.py` constructs a **220-dimensional feature tensor** via `ActionConditionedBrancher.observation_features(observation)`:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│              220-DIMENSIONAL PUBLIC OBSERVATION FEATURE VECTOR                         │
├────────────────────┬────────────────────┬───────────────────────┬──────────────────────┤
│ E_node[current]    │ E_node[goal]       │ Allowed Move Mask     │ Effective Adjacency  │
│ (32 dimensions)    │ (32 dimensions)    │ (12 dimensions)       │ (144 dimensions)     │
├────────────────────┼────────────────────┼───────────────────────┼──────────────────────┤
│ Learned 32-dim     │ Learned 32-dim     │ Binary 1-hot vector   │ Flattened 12x12      │
│ embedding of       │ embedding of       │ indicating allowed    │ binary adjacency     │
│ current node id    │ goal node id       │ moves from current    │ matrix (edges minus  │
│ from shared table  │ from shared table  │ node                  │ blockades)           │
└────────────────────┴────────────────────┴───────────────────────┴──────────────────────┘
```

* **Exact Dimension Breakdown:**
  $$D_{\text{obs}} = 32 + 32 + 12 + (12 \times 12) = 220\text{ features}$$
* **Encoder Architecture:** `ObservationEncoder` processes the 220-dim tensor via `Linear(220 -> 128) -> GELU -> Linear(128 -> 256) -> LayerNorm(256)`, outputting the 256-dim encoded observation vector $e_t \in \mathbb{R}^{256}$ (61,824 trainable parameters).
* **Total Model Parameters:** Exactly **171,660 trainable parameters** for RE:Reason (384 node embeddings + 61,824 encoder + 6,156 policy head + 69,760 branch input + 33,024 branch output + 512 branch LayerNorm).

---

## 4. Behavioral-Cloning Training Contract & Supervision Semantics

### 4.1 Relationship Between Policy Logits, Candidate Selection, and Branch Core
In `ActionConditionedBrancher`, there are two neural stages:
1. **The Policy Head** (`nn.Linear(512, 12)`): produces 12 unnormalized action logits $\ell_t = \text{PolicyHead}([S_t \parallel e_t])$.
2. **The Branch Core** (`branch_input` $544 \to 128$, `branch_output` $128 \to 256$, `branch_norm` $\text{LN}(256)$): takes $[S_t \parallel e_t \parallel E_{\text{node}}[a]]$ and computes candidate successor state $s^{(k)} \in \mathbb{R}^{256}$.

*Audit Finding:* At inference time, `propose_candidates` sorts eligible legal moves by descending policy logit and evaluates the branch core on only the top $K=3$ actions. During early training, the policy is unaligned, so the optimal demonstration action $a^*_t$ may rank 4th or lower and thus be absent from the top $K=3$ candidate set.

### 4.2 Explicit Training Contract: Teacher-Forced Sequence Unrolling
To resolve the candidate-ranking disconnect during training without altering the Milestone 3A inference contract, behavioral cloning (BC) training is governed by an explicit **Teacher-Forced Sequence Unrolling Contract**:

```
Training Step t along BFS Shortest Path (v_0, v_1, ..., v_L):
1. Compute Observation Embedding:
   e_t = ObservationEncoder(observation_features(obs_t))
2. Compute Policy Logits & Policy Loss:
   logits_t = PolicyHead([S_t || e_t])
   L_policy,t = CrossEntropy(logits_t, a*_t)   where a*_t = v_{t+1}
3. Advance Latent State via Branch Core on Demonstration Action:
   S_{t+1} = BranchCore(S_t, e_t, E_node[a*_t])
4. Recurrent Gradient Flow:
   L_policy,t+1 at step t+1 backpropagates through S_{t+1} directly into BranchCore!
```

*Key Technical Properties of the Training Contract:*
1. **Policy Head Supervision is Always Defined:** Cross-entropy loss $\mathcal{L}_{\text{policy}, t} = \text{CE}(\ell_t, a^*_t)$ evaluates the full 12-dimensional logit vector. It pulls up the logit for the demonstration action $a^*_t$ regardless of whether $a^*_t$ ranks in the top $K$ at that iteration.
2. **Branch Core Receives Downstream Policy Gradients:** Because the successor state along the demonstration path is generated as $S_{t+1} = \text{BranchCore}(S_t, e_t, E_{\text{node}}[a^*_t])$, the downstream policy loss $\mathcal{L}_{\text{policy}, t+1}$ backpropagates directly through $S_{t+1}$ into `branch_input`, `branch_output`, `branch_norm`, and upstream state $S_t$.
3. **No Candidate-Absence Failure:** Because teacher forcing uses the known demonstration action $a^*_t$ to advance $S_{t+1}$, the branch core is never starved of training signal when $a^*_t$ is absent from the policy's top-$K$ ranking.
4. **Inference Contract Preserved Bit-for-Bit:** Candidate proposal ranking (`propose_candidates`), top-$K$ selection, and the two-phase transaction gate operate strictly during **evaluation and inference** as implemented in Milestone 3A. No training-only hacks (such as dynamic candidate injection into `BranchProposal`) are injected into model code.

### 4.3 Sequence Loss Aggregation, Padding Masking, & Boundary Conditions
For a batch of $B$ demonstration episodes with variable shortest-path lengths $L_1, \dots, L_B$ ($L_b \le T_{\max}$):

1. **Transition-Weighted Batch Reduction:**
   Demonstration sequences are padded to the maximum path length in the batch $T_{\text{batch}} = \max_b L_b$. A binary step validity mask is defined as $M_{b, t} = \mathbf{1}[t < L_b]$. The batch loss is computed as the total cross-entropy over all valid demonstration transitions divided by the total number of valid transitions across the batch:
   $$\mathcal{L}_{\text{batch}} = \frac{1}{\sum_{b=1}^B L_b} \sum_{b=1}^B \sum_{t=0}^{T_{\text{batch}}-1} M_{b, t} \cdot \text{CrossEntropy}(\ell_{b, t}, a^*_{b, t})$$
2. **Padding Masking Contract:**
   Padding steps ($t \ge L_b$) are zero-masked by $M_{b, t} = 0$; they contribute zero loss and zero gradient to model parameters.
3. **Solvability Guarantee & Defensive Filtering:**
   In the benchmark generator (`src/re_reason/benchmarks/pathfinding_benchmark.py`), every generated episode is constructed around a guaranteed unblocked spine and validated against `solve_shortest_path`. Consequently, 100% of generated instances in `train`, `validation`, and `test` partitions possess valid BFS demonstration paths ($L_b \ge 1$, `solved == True`). If an external or synthetic episode with `solved == False` is ever encountered, the training runner must skip it (loss = 0) without raising an exception or accessing protected partitions.
4. **Terminal Demonstration Step Gradient Boundary:**
   At the final demonstration step $t = L_b - 1$, the demonstration transition arrives at `goal_node` ($v_{L_b}$). Because no subsequent action $a^*_{b, L_b}$ exists, step $L_b - 1$ produces policy loss $\mathcal{L}_{b, L_b - 1}$, but its successor state $S_{b, L_b}$ has no downstream step or policy loss. Consequently, the branch core evaluated on the final transition receives **zero backpropagated policy loss**.

### 4.4 Exposure Bias & Epistemic Boundaries of Behavioral Cloning
* **Exposure / Distribution Mismatch:** Under teacher forcing, the recurrent core experiences only on-distribution latent states generated by executing ground-truth optimal moves. At inference time, closed-loop execution is prone to exposure bias: if the policy ranks a suboptimal legal move into the top 3 and commits it, the latent state transitions to an out-of-distribution state never encountered during training.
* **Absence of Direct Transition Supervision:** Gradients through downstream policy losses adjust the branch core to produce latent representations that facilitate next-step action prediction. They do **not** provide direct supervision of transition quality (e.g. no next-observation prediction or state-reconstruction objective) and do not train candidate discrimination (evaluating valid vs. invalid branches).
* **Epistemic Principle:** *Teacher forcing alone does not guarantee or establish successful closed-loop inference behavior.* Successful goal-reaching at inference depends on the interaction between learned policy priors, runtime verification, and transaction gating.

### 4.5 Retry & Abort Supervision Contract
* **Demonstration Alignment:** In the BC training regime, models are trained on valid shortest-path demonstration sequences. Retries and transaction rollbacks do not occur along optimal demonstration paths.
* **Zero Rollback Backpropagation:** In validation rollouts and future multi-step training, checkpoint rollback (`LatentState.restore_checkpoint`) is a non-differentiable tensor state restoration. Gradients are never backpropagated through rollback operations or aborted rounds.
* **Full Telemetry Accounting:** All retry attempts ($R \le R_{\max} = 3$), rollbacks, and deadlock/stagnation terminations are recorded explicitly in evaluation logs and FLOP accounting; zero events are omitted.

---

## 5. Primary Metric & Statistical Decision Procedure

### 5.1 Primary Evaluation Metric: Goal-Reaching Rate (GRR)
* **Definition:** Empirical probability that an agent completes an episode by arriving at `goal_node` within its allocated transition budget without encountering an invalid edge or dynamic blockade:
  $$\text{GRR} = \frac{1}{N_{\text{eval}}} \sum_{i=1}^{N_{\text{eval}}} \mathbf{1}[\text{Episode } i \text{ reached goal}]$$
* **Unit of Statistical Analysis:** The independent unit of replication is the **paired model training seed** ($s$). Individual graph episodes are common evaluation benchmarks, not independent degrees of freedom.
* **Paired Difference:** For seed $s \in \{1, \dots, S\}$:
  $$\Delta_s = \text{GRR}_{\text{RE:Reason}, s} - \text{GRR}_{\text{baseline}, s}$$
  with sample mean $\bar{\Delta}$, sample standard deviation $s_\Delta$, and standard error $\text{SE}(\bar{\Delta}) = s_\Delta / \sqrt{S}$.

### 5.2 Confidence Interval & Decision Boundaries
All statistical decisions evaluate a two-sided $95\%$ Student's $t$ confidence interval:
$$\text{CI}_{95\%}(\bar{\Delta}) = [\text{CI}_{\text{lower}}, \; \text{CI}_{\text{upper}}] = [\bar{\Delta} - t_{\text{crit}} \cdot \text{SE}(\bar{\Delta}), \; \bar{\Delta} + t_{\text{crit}} \cdot \text{SE}(\bar{\Delta})]$$
where $t_{\text{crit}} = 2.776$ ($S=5, df=4$) and $t_{\text{crit}} = 2.262$ ($S=10, df=9$).

The observed confidence interval is mapped into three mutually exclusive and exhaustive categories relative to the ratified margin $\delta = +0.10$:

1. **Practical Improvement Supported:**
   $$\text{CI}_{\text{lower}} \ge +0.10$$
   *Decision:* The lower confidence bound meets or exceeds $+0.10$. Empirical evidence demonstrates at least a 10.0 percentage point advantage over the comparator.
2. **Practical Improvement Threshold Not Supported:**
   $$\text{CI}_{\text{upper}} < +0.10$$
   *Decision:* The upper confidence bound is strictly below $+0.10$, ruling out a practical advantage of $+0.10$ or greater. This category partitions into three mutually exclusive subcases:
   - *Subcase 2a (Statistically significant positive effect, but sub-threshold):*
     $$0.00 < \text{CI}_{\text{lower}} < \text{CI}_{\text{upper}} < +0.10$$
   - *Subcase 2b (Positive effect not established; zero plausible):*
     $$\text{CI}_{\text{lower}} \le 0.00 \le \text{CI}_{\text{upper}} < +0.10$$
   - *Subcase 2c (Statistically significant negative effect):*
     $$\text{CI}_{\text{upper}} < 0.00$$
3. **Inconclusive (Spanning Margin):**
   $$\text{CI}_{\text{lower}} < +0.10 \le \text{CI}_{\text{upper}}$$
   *Decision:* The confidence interval spans $+0.10$. The evidence neither confirms that the effect is $\ge +0.10$ nor rules out that it could be $\ge +0.10$.

*Sequential Inspection Qualification:* Applying nominal 95% confidence intervals sequentially at $S=5$ and, if inconclusive, at $S=10$ constitutes a two-stage group sequential evaluation without alpha correction. The effective family-wise type I error rate may slightly exceed 5% (typically $\approx 7\text{–}8\%$). This limitation is formally acknowledged; nominal 95% coverage is not preserved across the sequential procedure.

---

## 6. Seed Pairing, RNG Namespaces, & Held-Out Test Firewall

### 6.1 Independent, Reproducible RNG Namespaces
To ensure complete experimental reproducibility without cross-subsystem RNG coupling, the protocol establishes three independent namespaces:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        INDEPENDENT REPRODUCIBLE RNG NAMESPACES                         │
├───────────────────────┬───────────────────────────────┬────────────────────────────────┤
│ Namespace             │ Deterministic Seed Rule       │ Governed Scope & Subsystem     │
├───────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ 1. Graph Benchmark    │ base_seed = 42                │ Common benchmark partitions    │
│    (RNG_GRAPH)        │ train: base_seed + 1000       │ (Train: 200, Val: 50, Test: 50)│
│                       │ val:   base_seed + 50000      │ Independent of model training  │
│                       │ test:  base_seed + 90000      │ seeds or rollout RNG.          │
├───────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ 2. Model Training     │ seed = s in {1 .. S}          │ PyTorch weight initialization  │
│    (RNG_TRAIN)        │ PyTorch: manual_seed(s)       │ and training batch shuffle     │
│                       │ NumPy: default_rng(100000 + s)│ order. Paired across RE:Reason │
│                       │ Python: seed(100000 + s)      │ and each baseline comparator.  │
├───────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ 3. Stochastic Rollout │ seed_rollout(s, i, traj, part)│ Action sampling for Two-Rollout│
│    Action Sampling    │ = 500000 + (s * 10000)        │ baseline during evaluation.    │
│    (RNG_EVAL_SAMPLE)  │ + part_offset + (i * 10)      │ Uses isolated torch.Generator. │
│                       │ + traj_id                     │ Does not touch RNG_TRAIN.      │
│                       │ (val: 0, test: 200000)        │                                │
└───────────────────────┴───────────────────────────────┴────────────────────────────────┘
```

*Properties of the Namespace Design:*
1. **Decoupled Inference RNG with Partition Scoping:**
   Stochastic action sampling for the Two-Rollout baseline is evaluated via an isolated generator instance:
   $$\text{seed\_rollout}(s, i, \text{traj\_id}, \text{part}) = 500000 + (s \times 10000) + \text{part\_offset} + (i \times 10) + \text{traj\_id}$$
   where $\text{part\_offset} = 0$ for `validation` and $\text{part\_offset} = 200000$ for `test`, $s \in \{1, \dots, S\}$ is the paired model training seed, $i \in \{0, \dots, N-1\}$ is the 0-indexed episode position within the partition, and $\text{traj\_id} \in \{1, 2\}$ distinguishes the two sequential rollouts.
   This generator (`torch.Generator().manual_seed(seed_rollout)`) is completely isolated; it never advances or perturbs the global training RNG stream (`RNG_TRAIN`), ensuring bit-exact determinism across repeated evaluations of the same frozen model without seed collisions between validation and test partitions.
2. **Common Evaluation Suite:** All models are evaluated on the exact same fixed canonical `validation` suite ($N=50$ episodes).
3. **Paired Training Replication:** Seed $s$ pairs RE:Reason and each comparator on training initialization and data ordering; the statistical unit is the model seed pair ($S=5$ or $S=10$).

### 6.2 Held-Out Test Firewall Decoupling
To protect the held-out `test` partition ($N=50$ episodes, `base_seed + 90000`) from optional-stopping leakage:
* **The 5-to-10 seed expansion rule is evaluated strictly on the `validation` partition.**
* If the validation result at $S=5$ is inconclusive, training expands to $S=10$ and validation is re-evaluated.
* **Once the validation phase terminates, all checkpoints, hyperparameters, and the sample size ($S=5$ or $S=10$) are permanently frozen.**
* The held-out `test` partition is evaluated **exactly once** on the frozen models at the final sample size.
* Zero tuning, zero sample expansion, and zero re-evaluation occur on the `test` partition.
* The protected modular addition held-out partition ($b \in \{0, 5\}$) remains strictly untouched.

---

## 7. Secondary Metrics & Compute Telemetry Protocol

All secondary metrics are confirmed as **strictly descriptive, non-gating diagnostics** (carrying zero veto power over the primary GRR gate):

1. **Path Optimality Ratio ($\bar{\rho}_{\text{opt}}$):**
   - Evaluated on *successful episodes only*:
     $$\bar{\rho}_{\text{opt}} = \frac{1}{|S_{\text{success}}|} \sum_{i \in S_{\text{success}}} \frac{\text{Length}(P_i)}{\text{Length}(P^*_{\text{BFS}, i})}$$
2. **Mean Episode Steps ($\bar{T}$):**
   - Average executed steps across all episodes: $\bar{T} = \frac{1}{N} \sum_{i=1}^N T_i$.
3. **Transaction Rollback Rate ($\bar{B}$):**
   - Mean count of candidate rounds rejected by the verifier triggering checkpoint rollback per episode.
4. **Deadlock / Retry Exhaustion Rate ($r_{\text{deadlock}}$):**
   - Fraction of episodes terminating due to retry exhaustion ($R > R_{\max} = 3$) at an intermediate node.
5. **Dual Compute Accounting Protocol:**
   - *Theoretical Peak FLOPs:* $T_{\max} \times \text{FLOPs/step}$ (assuming maximum transition budget and zero retries: $18.06\text{ MFLOPs}$ for RE:Reason, $18.07\text{ MFLOPs}$ for Two-Rollout baseline, $8.19\text{ MFLOPs}$ for single un-gated recurrent baseline).
   - *Actual Executed FLOPs:* Sum of operations actually executed during episode $i$:
     $$\text{FLOPs}_i = \sum_{t=1}^{T_i} \left[ \text{EncoderFLOPs} + \sum_{r=1}^{R_{i,t}} (\text{PolicyFLOPs} + K \cdot \text{BranchCoreFLOPs}) \right]$$
   - *Mean Executed FLOPs per Episode:* $\overline{\text{FLOPs}} = \frac{1}{N} \sum_{i=1}^N \text{FLOPs}_i$.
   - *Mean Executed FLOPs per Successful Episode:* $\overline{\text{FLOPs}}_{\text{success}} = \frac{1}{|S_{\text{success}}|} \sum_{i \in S_{\text{success}}} \text{FLOPs}_i$.
   - *CPU Wall-Clock Latency:* Mean milliseconds per episode ($\text{ms/ep}$) and per decision step ($\text{ms/step}$), measured on identical CPU hardware.

---

## 8. Specific Governance Bounds & Explicit Pending Decisions

### 8.1 Clarified Implementation Parameters (Resolved by PI Direction)
1. **Baseline A (Un-Gated Recurrent Residual Baseline):** Parameter-matched at exactly **171,660 trainable parameters** using the exact node embeddings (384), observation encoder (61,824), policy head (6,156), and residual MLP transition core (103,296). Executes single-trajectory ($K=1$) unconditional commitment without candidate branching, verification, or rollback. Isolates mechanisms against an identical residual recurrent system; not described as a GRU.
2. **Behavioral Cloning Loss & Masking Contract:** Formalized as transition-weighted batch mean with a binary step validity mask $M_{b, t} = \mathbf{1}[t < L_b]$:
   $$\mathcal{L}_{\text{batch}} = \frac{1}{\sum_{b=1}^B L_b} \sum_{b=1}^B \sum_{t=0}^{T_{\text{batch}}-1} M_{b, t} \cdot \text{CrossEntropy}(\ell_{b, t}, a^*_{b, t})$$
   Padding steps ($t \ge L_b$) contribute zero loss and zero gradient. Step $L_b - 1$ produces policy loss, but its successor state $S_{b, L_b}$ has no downstream loss and receives zero backpropagated gradient. Exposure mismatch and absence of direct transition supervision are formally documented.
3. **Two-Rollout Baseline Fixed Temperature:** Fixed at $\tau = 1.0$ for the initial experiment. Sweeps prohibited in initial protocol; subsequent tuning requires separate authorization.
4. **RNG Namespaces & Partition Scoping:** Partitioned into `RNG_GRAPH` (`base_seed=42`), `RNG_TRAIN` (paired seed $s$), and `RNG_EVAL_SAMPLE` with explicit partition offsets (`validation: 0`, `test: 200000`) for isolated per-rollout action sampling without cross-partition or training RNG coupling.

### 8.2 Explicit Pending Decisions (Awaiting PI Direction)
1. **Formal Milestone 3B Authorization:** Explicit PI work order authorizing baseline model implementation, training runner, and validation execution.
2. **Remaining Project-Level Roadmap Choices:**
   - Stage 3 composed sequence evaluation formal de-prioritization.
   - Confirmation of held-out test firewalls.
   - Official runner artifact overwrite safeguard authorization.
   - Experiment L reproduction drift audit scope and authorization.

---

## 9. Explicit Declaration of Governance Boundaries

* **Zero lines of implementation code touched:** No source files in `src/` were modified.
* **Zero models trained or evaluated:** No training runs, optimization steps, or benchmark evaluations were executed.
* **Zero access to protected partitions:** Modular addition held-out partition ($b \in \{0, 5\}$) and pathfinding `test` partition were not loaded, queried, or inspected.
* **External repos untouched:** `latent-thing-git/` was not accessed.
* **Git state:** Zero staging, commit, push, reset, or history-modifying commands were executed.
