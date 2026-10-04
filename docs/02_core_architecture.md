# Core Architecture

## 1. Status & Scope of the Architecture

RE:Reason is an experimental research framework. The full architecture described in Section 2 represents the **target conceptual pipeline**. It is not a finalized or frozen architecture. Components are added strictly across sequential phases and only become part of the final system if empirical ablation experiments justify their computational cost.

## 2. Target Conceptual Architecture & Operational Domains

The target conceptual pipeline operates across two distinct memory domains to isolate intermediate speculative computation from committed persistent memory:

```
                            INPUT OBSERVATION
                                   x_t
                                    │
                                    ▼
                           Observation Encoder
                              e_t = E(x_t)
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ COMMITTED DOMAIN                                                      │
│ Master Latent State: S_t (Clean, Checkpointable, Read-Only in Step)   │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
               Speculative State Isolation (Tensor Cloning)
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
   [ Candidate 1 ]           [ Candidate 2 ]           [ Candidate K ]
 s^(1)_0 = Clone(S_t)      s^(2)_0 = Clone(S_t)      s^(K)_0 = Clone(S_t)
          │                         │                         │
          ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Recursive Core   │      │ Recursive Core   │      │ Recursive Core   │
│ Step r = 1..R    │      │ Step r = 1..R    │      │ Step r = 1..R    │
│  - Residual f_θ  │      │  - Residual f_θ  │      │  - Residual f_θ  │
│  - Halting Gate  │      │  - Halting Gate  │      │  - Halting Gate  │
└─────────┬────────┘      └─────────┬────────┘      └─────────┬────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Typed Decision   │      │ Typed Decision   │      │ Typed Decision   │
│  - Output y^(1)  │      │  - Output y^(2)  │      │  - Output y^(K)  │
│  - Conf c^(1)    │      │  - Conf c^(2)    │      │  - Conf c^(K)    │
└─────────┬────────┘      └─────────┬────────┘      └─────────┬────────┘
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ TRANSITION VALIDATION & VERIFICATION                                  │
│ 1. Deterministic Rule Verifier: V_det(y^(k), x_t) in {0, 1}           │
│ 2. Auxiliary Learned Scorer:   V_learned(s^(k), e_t) in [0, 1]        │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ TRANSACTION COMMIT GATE                                               │
│ Decision: ACCEPT (k*) or REJECT (all)                                 │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [ All Candidates Fail ]             [ Best Candidate Wins ]
                  │                                   │
                  ▼                                   ▼
        S_(t+1) = S_t (Abort)               S_(t+1) = s^(k*) (Commit)
        Master State Unchanged              Atomic State Update
        (Rejected candidate not committed)
```

### Committed Domain
Contains the persistent master state vector $S_t \in \mathbb{R}^{B \times D}$ (or slotted representation $S_t \in \mathbb{R}^{B \times M \times D}$). $S_t$ is strictly read-only during the execution of time step $t$. It is checkpointable and represents the established belief state.

### Speculative / Working Domain
Contains up to $K$ candidate latent paths $\{s^{(1)}_{t}, \dots, s^{(K)}_{t}\}$. Candidate states are mutated in-place or across recursive passes. If an invalid or unverified candidate is rejected, its working buffer is discarded without modifying $S_t$.

### Speculative State Isolation vs. Copy-on-Write
PyTorch does not natively provide hardware-level copy-on-write (COW) memory paging. In RE:Reason, state isolation is achieved through explicit tensor cloning (`torch.clone()`) and detached buffer allocations. We use the terms "isolated tensor cloning", "state snapshot", or "speculative state isolation" rather than claiming a novel COW implementation. Only if an actual lazy copy-on-write mechanism is implemented should the term copy-on-write be applied.

## 3. Component Specifications

### 3.1 Observation Encoder
Maps external task observations $x_t \in \mathbb{R}^{d_{\text{in}}}$ to continuous embeddings $e_t \in \mathbb{R}^D$:
$$e_t = \text{Encoder}_\phi(x_t)$$
Implemented as a lightweight 2-layer MLP or linear projection with LayerNorm.

### 3.2 Latent State Container
Maintains $S_t$. Encapsulates methods for state snapshotting, branch instantiation, and atomic commit/rollback. Manages detached reference drops to ensure PyTorch's caching allocator does not leak computation graphs.

### 3.3 Recursive Cognition Core
Applies a single, weight-tied neural block $\theta$ across $r = 1, \dots, R$ recursive iterations:
$$s^{(k)}_{t, r} = \text{LayerNorm}\left(s^{(k)}_{t, r-1} + f_\theta(s^{(k)}_{t, r-1}, e_t)\right)$$
$f_\theta$ is a compact gated residual block (LayerNorm -> Linear -> GELU -> Linear). Weights are strictly tied across all $R$ steps.

### 3.4 Adaptive Halting Controller
Implements PonderNet-style dynamic iteration depth. At recursive step $r$, the controller outputs a scalar halting probability:
$$p^{(k)}_{t, r} = \sigma\left(W_h s^{(k)}_{t, r} + b_h\right)$$
Iterations stop when cumulative halting probability $\sum_{j=1}^r p^{(k)}_{t,j} \ge 1 - \epsilon$ or maximum depth $R_{\max}$ is reached.

### 3.5 Hypothesis Brancher
Generates $K$ candidate working states from parent $S_t$. 

Crucially, how to generate genuinely meaningful alternative hypotheses in continuous latent space remains an **engineering unknown**. The equation $s^{(k)}_{t,0} = S_t + \delta^{(k)}_t$ is only one candidate implementation, not an established solution. Candidate proposal mechanisms to evaluate experimentally include:
* Action-conditioned branching: Candidate vectors explicitly conditioned on top-$K$ discrete action proposals.
* Orthogonal learned perturbations: Generating candidate offsets $\delta^{(k)}_t$ via orthogonal linear projection heads.
* Stochastic perturbation: Controlled latent noise injection.

The brancher must be explicitly evaluated on branch diversity and duplicate trajectory rates to verify that branches do not degenerate into identical copies.

### 3.6 Typed Decision Heads
Projects internal states to bounded, structured spaces:
* Categorical actions: $\hat{y}^{(k)} = \text{Softmax}(W_y s^{(k)}_{t, \text{halt}})$
* Scalar bounds: $\hat{v}^{(k)} = \text{Clamp}(W_v s^{(k)}_{t, \text{halt}}, v_{\min}, v_{\max})$
* Internal confidence: $c^{(k)} = \sigma(W_c s^{(k)}_{t, \text{halt}})$

Free-form natural language generation is excluded from decision heads.

### 3.7 Candidate Transition Validation & Verifier Subsystem
Verification in RE:Reason is strictly defined as validating:
* Candidate outputs ($\hat{y}^{(k)}$)
* Candidate state transitions ($s^{(k)}_{t,0} \to s^{(k)}_{t, \text{halt}}$)
* Domain-specific state consistency and invariants

Verification does **not** imply that the system can determine the intrinsic semantic correctness of an arbitrary continuous latent vector.

* Deterministic Verifier: Hard rule checker validating task invariants, physical constraints, and syntax:
  $$V_{\text{det}}(\hat{y}^{(k)}, x_t) \in \{0, 1\}$$
  On synthetic benchmarks, deterministic verification is the primary trusted ground truth.
* Learned Verifier: Auxiliary trajectory scorer predicting state transition consistency:
  $$V_{\text{learned}}(s^{(k)}_{t, \text{halt}}, e_t) \in [0, 1]$$
  The learned verifier never accesses ground-truth answers during inference and is never treated as an infallible oracle.

### 3.8 Transaction Commit Gate
Coordinates state finalization:
$$S_{t+1} = \begin{cases} s^{(k^*)}_{t, \text{halt}} & \text{if at least one candidate is VALID} \\ S_t & \text{if all candidates are REJECTED (Rollback)} \end{cases}$$
where $k^* = \arg\max_{k \in \text{valid}} c^{(k)}$.

The transaction gate guarantees isolation of the committed master state from the rejected candidate. It ensures the master state remains unchanged when transitions fail, preventing direct injection of unverified states. It does not prove that the model's internal representation has suffered zero degradation.

## 4. Step-by-Step Inference Lifecycle

1. Ingest: Observation $x_t$ is encoded into $e_t$.
2. Snapshot: Clean state $S_t$ is preserved. $K$ speculative working states $\{s^{(1)}_{t,0}, \dots, s^{(K)}_{t,0}\}$ are instantiated via isolated tensor cloning.
3. Unroll Recursion: For each branch, apply $f_\theta$ for $r = 1 \dots R$. Check halting condition $p_{t,r}$.
4. Output Extraction: Extract candidate predictions $\hat{y}^{(k)}$ and confidence estimates $c^{(k)}$.
5. Verification: Apply deterministic assertions $V_{\text{det}}$. If enabled, compute auxiliary learned score $V_{\text{learned}}$.
6. Commit / Rollback:
   * If a valid candidate exists: commit winning branch to $S_{t+1}$.
   * If all candidates fail: abort transition, retain $S_{t+1} = S_t$ (master state unchanged; rejected candidate not committed).
7. Cleanup: Delete speculative tensor references to allow allocation reuse by the PyTorch caching allocator.

## 5. Minimal Phase 1 Architecture (Strict Baseline)

In Phase 1, all speculative branching, recursive iteration, adaptive halting, verification, and transactional gating are completely excluded.

```
                           INPUT OBSERVATION
                                  x_t
                                   │
                                   ▼
                          Observation Encoder
                             e_t = E(x_t)
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │ Persistent State      │
                       │ Update: S_t           │
                       └───────────┬───────────┘
                                   │
                                   ▼
                         Typed Decision Head
                                 y_hat
```

### Phase 1 Mathematical Formulation:
$$e_t = \text{Encoder}(x_t)$$
$$S_t = \text{LayerNorm}\left(S_{t-1} + f_\theta(S_{t-1}, e_t)\right)$$
$$\hat{y}_t = \text{DecisionHead}(S_t)$$

### Phase 1 Locked Decisions:
The engineering and architectural decisions for Phase 1 are locked as follows:
* **State Representation:** Flat tensor $S \in \mathbb{R}^{B \times 256}$.
* **State Update:** $S_t = \text{LayerNorm}\left(S_{t-1} + f_\theta(S_{t-1}, e_t)\right)$.
* **Recursion Depth:** $R = 1$ (strictly single pass; no recursion).
* **Branch Count:** $K = 1$ (strictly single trajectory; no branching).
* **Halting:** Fixed single pass (no adaptive halting / PonderNet).
* **Verifier:** None (no deterministic or learned verifier).
* **Transactional Commit:** None (standard sequential recurrent state update).
* **Loss & Supervision:** Terminal supervision at step $t = 10$. Step-wise predictions and losses may be logged only as diagnostics.
* **Direct Baseline:** Parameter-matched PyTorch `nn.GRUCell` ($\sim 250\text{K}$ parameters).
* **Phase 1 Pipeline:** Strictly $\text{Input} \to \text{Observation Encoder} \to \text{Persistent State Update} \to \text{Typed Prediction}$.
* **Exclusions:** No recursion, branching, halting, verification, transactional commit, multimodality, NLP, or online weight updates.

This minimal architecture exists purely to determine whether the proposed persistent-state update formulation can achieve basic sequential state tracking without being materially worse than a standard GRU at comparable parameter count.

For the progression from Phase 1 through Phase 10, see [docs/03_development_roadmap.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/03_development_roadmap.md).  
For the benchmark suite and baseline evaluation details, see [docs/04_experiments_and_evaluation.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/04_experiments_and_evaluation.md).
