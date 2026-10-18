# Research Proposal — Phase 1.6: Architectural Reformulation for Algorithmic Binding and Modular Arithmetic

**Date:** 2026-10-13  
**Status:** Proposal Under Review — Implementation NOT Authorized  
**Author:** RE:Reason Project Team  
**Governing Documents:** `AGENTS.md`, `docs/DOCUMENTATION_POLICY.md`, `docs/reports/phase_1_evidence_synthesis.md`  
**Target Phase:** Phase 1.6 (Architectural Reformulation)

---

## 1. Executive Summary and Precise Research Questions

### 1.1 Context and Scientific Motivation

Phase 1 of RE:Reason investigated whether a recurrent inference architecture with bounded convex updates (`NonlinearCandidateMGUModel`, Model E) could serve as a persistent algorithmic core for multi-step reasoning. The Phase 1 empirical synthesis (`docs/reports/phase_1_evidence_synthesis.md`) formally closed the Phase 1 diagnostic cycle (**Decision A: GO**) and established a definitive stop on ad-hoc model patching or immediate transition to Phase 2 speculative machinery (**Decision B: NO-GO**).

The empirical record across thirty-four conceptual experiments (A through AH) demonstrated that:
1. **Passive state persistence is functional:** Model E maintains untouched registers cleanly over 10 to 20 steps with 0.0% unwritten register corruption, achieving 0.755 ± 0.019 Exact Match (EM) at T=10 and 0.407 ± 0.023 at T=20 under candidate/local supervision (`experiments/checkpoints/E_candidate_seed{42..46}_3000steps.pt`).
2. **Active arithmetic updates fail systematically:** Failures concentrate heavily when active addition transitions occur. In Experiment P (`docs/reports/experiment_p_sequence_error_budget.md`), candidate-generation errors (Class A) account for 59.7% of first errors at T=10 and 67.2% at T=20, with repeated-ADD sequence exact match collapsing to ~0.15.
3. **The failure is not an interface information limit:** Experiments U and AF (`docs/reports/experiment_u_task_interface.md`, `docs/reports/experiment_af_task_difficulty.md`) proved that the raw causal observation stream carries mathematically sufficient information by horizon $L^*=4$ (Bayes ceiling $\ge 0.998$; full prefix 1.000) across every held-out operand stratum.
4. **Standard sequence learners fail rule generalisation:** Exhaustive diagnostic sweeps across data scale (W), update budgets (X), optimizer families (Z–AC), training-pool balancing (AG), and capacity scaling (AH) demonstrated that bounded sequence readers (GRUs, TCNs, MLPs) under empirical risk minimization fit training pools via sample memorization (training accuracy up to 0.77–1.00) while held-out operand balanced accuracy ($b \in \{0,5\}$) remains locked at chance (0.072–0.143 vs. 0.10 chance).
5. **Local interventions are destructive:** Interventions on existing gates (J, K) or candidate vectors (O) degrade sequence accuracy (e.g., -10.4 pp EM in O; `docs/reports/experiment_o_representation_intervention.md`).

Within the tested benchmark, model families, and training protocols of Phase 1, the evidence demonstrates that generic continuous recurrent representations under empirical risk minimization fit training distributions via sample memorization rather than acquiring general modular arithmetic mechanisms.

### 1.2 The Two Distinct Research Problems

Phase 1.6 addresses two distinct structural requirements that were conflated in the original monolithic state representation:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE DUAL BOTTLENECK                                    │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│ Problem 1: Address-Value Binding & Routing│ Problem 2: Modular Arithmetic Computation  │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ * Observation at step t is operand-blind: │ * Task requires computing:                 │
│   encodes (op, dest, src) with no values. │     y_true = (a + b) mod 10                │
│ * Model must store 4 discrete register    │ * Under ERM, continuous representations    │
│   states without mutual interference.     │   memorize seen pairs (80/100).            │
│ * Upstream representation probing (T)     │ * On held-out pairs (b in {0,5}), accuracy │
│   shows state pathway is weak and fails   │   collapses to chance (~0.10).             │
│   held-out generalisation.                │ * Cyclic wrap-around (sum >= 10) is a non- │
│ * Gating acts on continuous blend, not    │   convex boundary in Euclidean space.      │
│   discrete address slots.                 │                                            │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

These two problems are logically independent: an architecture could possess perfect discrete register addressing while failing modular addition, or it could compute modular addition perfectly on scalar inputs while corrupting register bindings during multi-step execution.

### 1.3 Precise Research Questions

Phase 1.6 evaluates three explicit architectural research questions:

* **RQ1 (Address-Value Binding):** Does structuring the recurrent state into explicit, disentangled register address slots eliminate cross-register interference and provide robust, held-out-generalising operand routing compared to a monolithic continuous state vector?
* **RQ2 (Modular Arithmetic Inductive Bias):** Does embedding scalar register values into a cyclic topological manifold (e.g. rotary/harmonic phase embeddings) enable zero-shot rule generalisation on held-out modular addition pairs ($b \in \{0,5\}$), where generic Euclidean MLPs fail?
* **RQ3 (Composed-Program Generalisation, Length-Extrapolation, and Persistence):** Do the two trainable candidate architectures (Slot-MGU with explicit slot addressing, and Cyclic-MGU with cyclic arithmetic inductive bias) achieve the specified composed-program generalisation ($acc_{heldout} \ge 0.800$, $EM_{heldout} \ge 0.700$), length extrapolation (with $T=20$ as the primary gate requiring $EM_{T=20} \ge 0.700$, and $T=30, 50$ as secondary diagnostic probes), and extended persistence ($EM_{read} \ge 0.950$ at $K=20$) thresholds, compared with the Vanilla GRU baseline and deterministic MSM exact-rule reference?
  - *Explicit Scope and Non-Claim:* The approved study evaluates the two trainable candidates independently against the baseline and deterministic reference; it does **not** test a trainable neural architecture combining explicit slot addressing and cyclic arithmetic units. No claims of evidence for that untested trainable composition are made.

---

## 2. Phase 1 Evidence and Motivation

Every proposal element is grounded directly in the finalized Phase 1 evidence base:

| Experiment / Report | Established Finding | Direct Architectural Implication for Phase 1.6 |
|---|---|---|
| **E / G–I** (`docs/reports/experiment_g_add_commit_gap.md`, `docs/reports/experiment_i_coordinate_commit.md`) | Bounded convex update maintains state cleanly, but ADD transitions suffer ~5.5% candidate-to-commit loss, primarily coordinate-selective under-delivery and off-path basin landing. | Scalar gating is inadequate; register updates must be routed to isolated coordinate blocks rather than blending across a shared state vector. |
| **L–N** (`docs/reports/experiment_l_readout_localization.md`, `docs/reports/experiment_m_representation_geometry.md`, `docs/reports/experiment_n_parity_origin_localization.md`) | Off-path errors land in a systematic same-parity modular basin (73.7%); failure is latent-state misplacement (98.8%), driven by state-to-label alignment (91.1%). | The continuous Euclidean latent space naturally organizes by broad coarse clusters (parity) but lacks metric precision for cyclic wrap-around arithmetic. |
| **J, K, O** (`docs/reports/experiment_j_commit_intervention.md`, `docs/reports/experiment_o_representation_intervention.md`) | Heuristic gate scaling, sharpening, and candidate parity subtraction degrade sequence exact match (-10.4 pp EM). | Post-hoc geometric interventions on frozen weights cannot fix representational deficits; a new architecture must be trained from scratch. |
| **P** (`docs/reports/experiment_p_sequence_error_budget.md`) | First-error budget is dominated by candidate generation during ADD (59.7% at T=10, 67.2% at T=20); untouched-register corruption is 0.0%. | Address retention is already stable; the design focus must be the ALU candidate generation and input routing. |
| **Q–S** (`docs/reports/experiment_q_candidate_origin.md`, `docs/reports/experiment_s_candidate_internal.md`) | True written value is weakly recoverable from $C_t$ (26.7%); internal candidate activations do not exceed $C_t$; no internal stage is an isolated defect. | The candidate MLP never computes the correct arithmetic representation; the deficit is computational and representational throughout. |
| **T** (`docs/reports/experiment_t_upstream_representation.md`) | Observation encoder $e_t$ is operand-blind (operands at chance); state pathway is weak and fails held-out generalisation. | The architecture requires an explicit mechanism to route stored register states to the arithmetic unit based on the instruction's address fields. |
| **U, AF** (`docs/reports/experiment_u_task_interface.md`, `docs/reports/experiment_af_task_difficulty.md`) | Raw observation history determines target with Bayes probability $\ge 0.998$ by $L^*=4$ (and 1.000 at full prefix) across every held-out stratum. | The task environment contains complete causal information; failure to generalize indicates an architectural and inductive-bias limitation within the tested reader families, training protocols, and optimization budgets. |
| **V–AH** (`docs/reports/experiment_v_sequence_access.md`, `docs/reports/experiment_ah_capacity_ladder.md`) | External sequence readers (GRUs, TCNs up to 105k params) fit training sets via memorization but collapse to near-chance on held-out pairs ($b \in \{0,5\}$ balanced acc 0.072–0.143). | Generic recurrent/convolutional deep learning levers (data scale, budget, optimizer, sampling, capacity) cannot induce modular arithmetic rules without architectural priors. |

---

## 3. Competing Architectural Hypotheses and Scope Resolution

We formulate and evaluate three distinct, testable architectural hypotheses:

```
                                  ARCHITECTURAL CANDIDATES
                                             │
         ┌───────────────────────────────────┴───────────────────────────────────┐
         ▼                                                                       ▼
   [Hypothesis A]                                                          [Hypothesis B]
Explicit Address-Value Binding                                           Modular Inductive Bias
 ├── Disentangled memory slots                                            ├── Cyclic/Rotary phase space
 ├── Hard address-based routing                                           ├── Trigonometric group action
 └── Continuous MLP ALU                                                   └── Monolithic state with continuous query
```

### Hypothesis A: Explicit Address-Value Binding (Disentangled Register Slots)

* **Proposed Mechanism:** Instead of representing all registers in a single coupled state vector $S_t \in \mathbb{R}^{212}$, partition the state space into $M=4$ explicit register memory slots $S_t = [s_{t,0}, s_{t,1}, s_{t,2}, s_{t,3}]$, where each $s_{t,r} \in \mathbb{R}^d$ ($d \approx 53$, maintaining total parameter/state budget). Transition instructions $(op, r_{dest}, r_{src})$ are executed via discrete or hard-sparse address routing:
  1. Operand retrieval: $u_a = s_{t-1, r_{dest}}$, $u_b = s_{t-1, r_{src}}$.
  2. Candidate computation: $c_t = \text{ALU}(u_a, u_b, op)$.
  3. Selective write: $s_{t, r_{dest}} = (1 - g_t) \odot s_{t-1, r_{dest}} + g_t \odot c_t$.
  4. Identity preservation: $s_{t, r} = s_{t-1, r}$ for all $r \neq r_{dest}$.
* **Theoretical Grounding:** Smolensky (1990) tensor product variable binding; Graves et al. (2014) Neural Turing Machines; Goyal et al. (2021) Recurrent Independent Mechanisms.
* **Expected Benefit:** Eliminates cross-register interference by architectural construction. Unwritten registers have zero representational drift. Solves Experiment T's operand-blindness by directly piping $u_a$ and $u_b$ into the ALU.
* **Failure Modes / Vulnerabilities:** Does not address the arithmetic representation. If the ALU is a standard continuous MLP, it will still memorize seen operand pairs and fail on held-out combinations ($b \in \{0,5\}$).
* **Implementation Complexity:** Low-to-moderate. Fully differentiable; parameter-neutral.
* **Disconfirming Evidence:** If explicit slots achieve $\ge 0.990$ behavioral register retention, but held-out arithmetic balanced accuracy remains at chance (~0.100).

### Hypothesis B: Modular Arithmetic Inductive Bias (Cyclic Group / Harmonic Manifold)

* **Proposed Mechanism:** Scalar register values are represented within a cyclic topological manifold $S^1$ or a multi-frequency Fourier basis, rather than an unconstrained Euclidean vector:
  $$\Phi(v) = \left[ \cos\left(\frac{2\pi k v}{10}\right), \sin\left(\frac{2\pi k v}{10}\right) \right]_{k=1}^K \in \mathbb{R}^{2K}$$
  Modular addition $(a + b) \pmod{10}$ is computed via group multiplication (phase addition / rotation in $\mathbb{C}^K$):
  $$\theta_{a+b} = (\theta_a + \theta_b) \pmod{2\pi}$$
  The candidate update computes rotation matrices or complex products rather than non-linear ReLU/tanh hyperplanes.
* **Theoretical Grounding:** Gromov (2023) Grokking Modular Arithmetic; Mallinar et al. (2025) block-circulant features in non-neural models; Swaroop (2026) latent algorithmic structures in modular MLPs.
* **Expected Benefit:** Eliminates the modular wrap-around boundary. In Euclidean space, $(4 + 5 = 9)$ and $(5 + 5 = 0)$ represent a discontinuous jump; on the circle $S^1$, addition is a continuous isometric rotation. Held-out operand generalisation is enforced by group structure.
* **Failure Modes / Vulnerabilities:** Phase drift and dispersion over long recurrence horizons. When embedded in a monolithic coupled state, continuous retrieval of operands $r_a, r_b$ can experience cross-register interference.
* **Implementation Complexity:** Moderate. Requires complex-valued or 2D-block linear algebra layers.
* **Disconfirming Evidence:** If harmonic representations fail to achieve $\ge 0.800$ accuracy on isolated scalar modular addition primitives, or if phase dispersion corrupts multi-step register reads.

### Hypothesis C: Scope Resolution and Staged Algorithmic Curriculum

Hypothesis C proposes a multi-stage training curriculum: (1) pre-training the ALU on isolated scalar modular addition, (2) pre-training routing controllers on SET/SWAP sequences, and (3) fine-tuning composed traces.

* **Methodological Decision:** **Hypothesis C is formally DEFERRED from the primary Phase 1.6 architectural investigation.**
* **Scientific Rationale:**
  1. *Confound Separation:* The core research question of Phase 1.6 is representational and architectural (RQ1, RQ2). Combining new architectures with a multi-stage curriculum would confound inductive bias with training protocol, making it impossible to attribute success or failure to the architecture.
  2. *Compute & Complexity Control:* A multi-stage curriculum multiplies training runs, loss schedules, and hyperparameter grids by $3\times$, exceeding the project's frugal budget.
  3. *Primary Focus:* The two trainable candidates (**Candidate 1: Slot-MGU** and **Candidate 2: Cyclic-MGU**) are evaluated under the **matched end-to-end optimizer protocol** across composed sequence stages. **Vanilla GRU** serves as the recurrent baseline. The Modular Slot Machine (**MSM**) is a parameter-free deterministic analytical reference and receives zero parameter-training runs.
  4. *Contingent Status:* Hypothesis C is preregistered as a contingent secondary investigation to be triggered *only* if the relevant trainable candidates (Slot-MGU or Cyclic-MGU) pass pilot stages (proving representational capability) but fail composed trace training due to documented empirical evidence of credit-assignment failure. The curriculum remains strictly deferred; no new training stages or candidate expansions are added to the active specification.

---

## 4. Recommended Candidate Shortlist and Full Specification

We specify two trainable candidate architectures, complying with the registered Model E parameter envelope ($226,358 \pm 10\%$, ~204k–249k parameters), alongside the dedicated 10-class Vanilla GRU baseline, and one parameter-free deterministic analytical reference (MSM) plus historical controls:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               ARCHITECTURAL CANDIDATES                                 │
├──────────────────────────────┬─────────────────────────────┬───────────────────────────┤
│ Candidate 1: Slot-MGU        │ Candidate 2: Cyclic-MGU     │ Candidate 3: Modular Slot │
│ (Trainable Candidate)        │ (Trainable Candidate)       │ Machine (MSM) (Reference) │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ * M=4 discrete memory slots  │ * Monolithic state S_t      │ * M=4 discrete slots      │
│   (d=53 each, total 212).    │   (212-dim).                │   (d=10 value coords each,│
│ * Hard address routing.      │ * Continuous query readout. │   total 40-dim state).    │
│ * Generic continuous MLP ALU.│ * Cyclic/Harmonic ALU.      │ * Hard address routing.   │
│ * 223,414 trainable params.  │ * 219,018 trainable params. │ * Parameter-free cyclic   │
│ * Isolates Hypothesis A.     │ * Isolates Hypothesis B.    │   ALU reference (0 params)│
└──────────────────────────────┴─────────────────────────────┴───────────────────────────┘
```

### 4.1 Candidate 1: Slot-MGU (Explicit Binding, Generic ALU)
* **Architecture:** State partitioned into 4 discrete slots $S_t = [s_{t,0}, s_{t,1}, s_{t,2}, s_{t,3}] \in \mathbb{R}^{4 \times 53}$.
* **Benchmark Instruction Token Contract ($x_t \in \{0, 1\}^{22}$):**
  The benchmark instruction token is uniformly defined across all operations as:
  $$x_t = [e_{op}, e(r_{dest}), e(r_{src}), e(v)] \in \{0, 1\}^{4 + 4 + 4 + 10} = \{0, 1\}^{22}$$
  where:
  - $e_{op} \in \{0, 1\}^4$ is the one-hot operation type (`SET`, `SWAP`, `READ`, `ADD`).
  - $e(r_{dest}) \in \{0, 1\}^4$ is the one-hot destination register address index ($r_0..r_3$).
  - $e(r_{src}) \in \{0, 1\}^4$ is the one-hot source register address index ($r_0..r_3$, or $\mathbf{0}_4$ when no source register is involved, such as for `SET` and `READ`).
  - $e(v) \in \{0, 1\}^{10}$ is the one-hot categorical indicator for immediate literal value $v \in \{0..9\}$ when $op = \text{SET}$, and is strictly the zero vector $\mathbf{0}_{10}$ for operations without an immediate scalar literal (`SWAP`, `READ`, `ADD`).
* **Explicit Encoding Justification (No Harmonic Prior Leakage):**
  The literal value $v$ reaches Candidate 1 exclusively as a standard discrete one-hot vector $e(v) \in \{0, 1\}^{10}$. Candidate 1 strictly **does not use** a fixed circular or harmonic encoding ($\Phi(v)$ is prohibited). Candidate 1 is explicitly designed to isolate **Hypothesis A** (evaluating whether discrete memory slots resolve variable binding in the presence of a generic continuous MLP ALU). Silently equipping Candidate 1 with a fixed cyclic harmonic representation would confound Hypothesis A with Hypothesis B, invalidating the scientific comparison.
* **Operand Retrieval & Unified Input Contract:**
  Discrete hard multiplexing routes source and destination slot vectors:
  $$u_a = s_{t-1, r_{dest}} \in \mathbb{R}^{53}, \quad u_b = \begin{cases} s_{t-1, r_{src}} \in \mathbb{R}^{53} & \text{if } op \in \{\text{ADD}, \text{SWAP}\} \\ \mathbf{0}_{53} & \text{if } op \in \{\text{SET}, \text{READ}\} \end{cases}$$
  The candidate generator $\text{MLP}_{cand}$ and recurrent convex gate $\text{MLP}_{gate}$ both receive the unified concatenated vector:
  $$I_t = [u_a, u_b, x_t] \in \mathbb{R}^{53 + 53 + 22} = \mathbb{R}^{128}$$
* **Operation Execution Semantics:**
  - **`SET r_dest v`:** $I_t = [s_{t-1, r_{dest}}, \mathbf{0}_{53}, e_{SET}, e(r_{dest}), \mathbf{0}_4, e(v)] \in \mathbb{R}^{128}$.
    Candidate vector $c_t = \text{MLP}_{cand}(I_t) \in \mathbb{R}^{53}$, gate $g_t = \sigma(\text{MLP}_{gate}(I_t)) \in (0, 1)^{53}$.
    Convex update applied exclusively to destination slot:
    $$s_{t, r_{dest}} = (1 - g_t) \odot s_{t-1, r_{dest}} + g_t \odot c_t$$
    All non-addressed slots maintain exact mathematical identity: $s_{t, r} = s_{t-1, r}$ $\forall r \neq r_{dest}$ ($\|\Delta s_r\| = 0.0$ guaranteed).
  - **`SWAP r_dest r_src`:** Slot contents are atomically exchanged via discrete routing:
    $$s_{t, r_{dest}} = s_{t-1, r_{src}}, \quad s_{t, r_{src}} = s_{t-1, r_{dest}}$$
    All unswapped slots maintain exact mathematical identity: $s_{t, r} = s_{t-1, r}$ $\forall r \notin \{r_{dest}, r_{src}\}$ ($\|\Delta s_r\| = 0.0$ guaranteed).
  - **`READ r_dest`:** State undergoes zero mutation across all slots ($S_t = S_{t-1}$, $\|\Delta s_r\| = 0.0$ $\forall r$).
    The linear decision head maps retrieved destination contents $u_a = s_{t-1, r_{dest}}$ to class logits:
    $$\hat{y}_t = \arg\max_{c \in \{0..9\}} \text{Head}(u_a)$$
  - **`ADD r_dest r_src`:** $I_t = [s_{t-1, r_{dest}}, s_{t-1, r_{src}}, e_{ADD}, e(r_{dest}), e(r_{src}), \mathbf{0}_{10}] \in \mathbb{R}^{128}$.
    Candidate vector $c_t = \text{MLP}_{cand}(I_t)$, gate $g_t = \sigma(\text{MLP}_{gate}(I_t))$.
    Convex update applied exclusively to destination slot:
    $$s_{t, r_{dest}} = (1 - g_t) \odot s_{t-1, r_{dest}} + g_t \odot c_t$$
    All non-addressed slots maintain exact mathematical identity: $s_{t, r} = s_{t-1, r}$ $\forall r \neq r_{dest}$ ($\|\Delta s_r\| = 0.0$ guaranteed).
* **Target Parameter Envelope:** $226,358 \pm 10\%$ parameters (~203,722 to 248,994).
* **Trainable Parameter Reconciliation & Layer Breakdown:**
  - Memory routing: Discrete hard multiplexing over register address indices ($0$ trainable parameters).
  - Unified input representation: $[u_a, u_b, x_t] \in \mathbb{R}^{128}$ ($53 + 53 + 22 = 128$).
  - Candidate generator $\text{MLP}_{cand}$: Linear($128 \to 612$) + GELU + Linear($612 \to 53$):
    - Linear($128 \to 612$): $128 \times 612 + 612 = 78,948$ parameters.
    - Linear($612 \to 53$): $612 \times 53 + 53 = 32,489$ parameters.
    - Subtotal: $111,437$ parameters.
  - Convex gate network $\text{MLP}_{gate}$: Linear($128 \to 612$) + GELU + Linear($612 \to 53$):
    - Linear($128 \to 612$): $128 \times 612 + 612 = 78,948$ parameters.
    - Linear($612 \to 53$): $612 \times 53 + 53 = 32,489$ parameters.
    - Subtotal: $111,437$ parameters.
  - Output decision head: Linear($53 \to 10$):
    - Linear($53 \to 10$): $53 \times 10 + 10 = 540$ parameters.
  - Total Trainable Parameters: $111,437 + 111,437 + 540 = \mathbf{223,414}$ parameters.
  - Parameter Envelope Compliance: 223,414 parameters is within $-1.30\%$ of the 226,358 reference target, strictly inside the registered envelope of $[203,722, 248,994]$. (Envelope compliance is established; exact parameter matching is not claimed).
* **Scientific Role:** Isolates Hypothesis A. Evaluates whether discrete memory slots resolve variable binding in the absence of arithmetic priors.

---

### 4.2 Candidate 2: Cyclic-MGU (Monolithic State, Continuous Retrieval, Cyclic ALU)

Candidate 2 intentionally maintains Model E's monolithic, continuous state vector $S_t \in \mathbb{R}^{212}$ without structural slot partitions. It serves as an essential comparator isolating the effect of cyclic arithmetic representations when embedded in a shared, coupled state.

#### Complete Read-Compute-Write Operation and Atomic SWAP
1. **Instruction Representation:** The instruction token $x_t$ is encoded as:
   $$e_{inst} = [e_{op}, e(r_{dest}), e(r_{src})] \in \mathbb{R}^{d_{op} + 4 + 4} = \mathbb{R}^{12}$$
   where $e_{op} \in \{0, 1\}^4$ is the one-hot operation type (`SET`, `SWAP`, `READ`, `ADD`), $e(r_{dest}) \in \{0, 1\}^4$ is the one-hot destination address, and $e(r_{src}) \in \{0, 1\}^4$ is the one-hot source address.
2. **Pre-Update Snapshot Operand Retrieval from Monolithic State:**
   To guarantee read consistency and prevent read-after-write corruption, both operand representations are retrieved simultaneously from the *pre-update* state $S_{t-1} \in \mathbb{R}^{212}$ before any state mutation:
   $$u_a = \text{MLP}_{read}([S_{t-1}, e(r_{dest})]), \quad u_b = \text{MLP}_{read}([S_{t-1}, e(r_{src})])$$
   where $\text{MLP}_{read}: \mathbb{R}^{212 + 4} \to \mathbb{R}^{2K}$ ($2K = 10$, $K=5$ harmonics). Each 2D harmonic frequency block $k \in \{1..K\}$ is normalized to the unit circle:
   $$\tilde{u}_a^{(k)} = \frac{u_a^{(k)}}{\max(\|u_a^{(k)}\|_2, \epsilon)}, \quad \tilde{u}_b^{(k)} = \frac{u_b^{(k)}}{\max(\|u_b^{(k)}\|_2, \epsilon)}, \quad \epsilon = 10^{-6}$$
   Here $\tilde{u}_a \in \mathbb{R}^{10}$ is the pre-update value of $r_{dest}$, and $\tilde{u}_b \in \mathbb{R}^{10}$ is the pre-update value of $r_{src}$.
3. **Fixed Cyclic ALU Computation and Dual-Payload Formulation:**
   The writer receives two structured write payloads $(c_{1, t}, e(r_{1, t}))$ and $(c_{2, t}, e(r_{2, t}))$:
   * **For `ADD r_dest r_src`:**
     The fixed cyclic ALU receives the concatenated 20-dimensional operand vector $[\tilde{u}_a, \tilde{u}_b] \in \mathbb{R}^{20}$ (composed of pre-update 10-dimensional harmonic embeddings $\tilde{u}_a, \tilde{u}_b \in \mathbb{R}^{10}$, each containing $K=5$ two-dimensional blocks) and computes the 10-dimensional harmonic rotation result $c_{1, t} \in \mathbb{R}^{10}$ across $K=5$ block-diagonal 2D Givens rotations:
     $$c_{1, t}^{(k)} = R(\theta_b^{(k)}) \tilde{u}_a^{(k)} = \begin{bmatrix} \tilde{u}_{a,1}^{(k)} \tilde{u}_{b,1}^{(k)} - \tilde{u}_{a,2}^{(k)} \tilde{u}_{b,2}^{(k)} \\ \tilde{u}_{a,2}^{(k)} \tilde{u}_{b,1}^{(k)} + \tilde{u}_{a,1}^{(k)} \tilde{u}_{b,2}^{(k)} \end{bmatrix}, \quad r_{1, t} = r_{dest}$$
     The secondary write payload is inactive: $c_{2, t} = \mathbf{0} \in \mathbb{R}^{10}, e(r_{2, t}) = \mathbf{0} \in \mathbb{R}^4$.
   * **For `SET r_dest v`:**
     Primary payload receives the canonical harmonic embedding: $c_{1, t} = \Phi(v) \in \mathbb{R}^{10}, r_{1, t} = r_{dest}$. Secondary payload is inactive: $c_{2, t} = \mathbf{0}, e(r_{2, t}) = \mathbf{0}$.
   * **For `SWAP r_dest r_src` (Atomic Dual-Target Transition):**
     Both writes are formulated from the pre-update snapshot values:
     - Primary target: $r_{1, t} = r_{dest}$ receives pre-update source value $c_{1, t} = \tilde{u}_b$.
     - Secondary target: $r_{2, t} = r_{src}$ receives pre-update destination value $c_{2, t} = \tilde{u}_a$.
     Because $\tilde{u}_a$ and $\tilde{u}_b$ are both captured from the pre-update state $S_{t-1}$, neither register's value is overwritten before the other is extracted.
   * **For `READ r_dest`:**
     Readout pass: $c_{1, t} = \tilde{u}_a, c_{2, t} = \mathbf{0}$; state update gate is forced to zero ($g_t = \mathbf{0}$).
4. **Primary Writer Specification: Destination-Conditioned Full Replacement Vector with Convex Gating:**
   The write payloads are projected into the monolithic state space via a joint write candidate generator $\text{MLP}_{write}: \mathbb{R}^{32} \to \mathbb{R}^{212}$, generating a full state replacement candidate $C_t \in \mathbb{R}^{212}$:
   $$\tilde{C}_t = \text{MLP}_{write}([c_{1, t}, e(r_{1, t}), c_{2, t}, e(r_{2, t}), e_{op}]), \quad C_t = \tanh(\tilde{C}_t) \in [-1, 1]^{212}$$
   where the input vector is the concatenation of both payloads and the operation type ($10 + 4 + 10 + 4 + 4 = 32$ dimensions).
   The recurrent convex gate vector $g_t \in (0, 1)^{212}$ is generated via:
   $$g_t = \sigma(\text{MLP}_{gate}([S_{t-1}, C_t, e_{inst}])) \in (0, 1)^{212}$$
   where the input is the concatenation of pre-update state, write candidate, and instruction token ($212 + 212 + 12 = 436$ dimensions). For `READ`, $g_t$ is hard-clamped to $\mathbf{0}$.
   The single atomic recurrent convex state update is:
   $$S_t = (1 - g_t) \odot S_{t-1} + g_t \odot C_t \in [-1, 1]^{212}$$
5. **Execution Mechanics of Atomic SWAP and Preservation of Non-Target Registers:**
   * *Atomicity and No Mutual Overwrite:* In `SWAP`, $C_t$ provides the replacement coordinates for both $r_{dest}$ (encoding $\tilde{u}_b$) and $r_{src}$ (encoding $\tilde{u}_a$) in a single vector. The convex gating $S_t = (1 - g_t) \odot S_{t-1} + g_t \odot C_t$ applies both updates in one vector operation. Neither register undergoes an intermediate state that could overwrite the other.
   * *Behavioral Preservation of Remaining Registers ($r \notin \{r_{dest}, r_{src}\}$):* The gating network $\text{MLP}_{gate}$ is conditioned on $e_{inst}$ and must learn to output $g_{t, k} \approx 0$ on the coordinate subspaces allocated to unswapped registers, preserving $S_{t, k} \approx S_{t-1, k}$ so that subsequent queries satisfy:
     $$\text{MLP}_{read}([S_t, e(r)]) \approx \text{MLP}_{read}([S_{t-1}, e(r)]) \quad \forall r \notin \{r_{dest}, r_{src}\}$$
6. **Architectural Guarantees vs. Learned Preservation Behavior:**
   * *Guaranteed by Architecture:*
     1. *Coordinate Boundedness:* Because $S_{t-1} \in [-1, 1]^{212}$ and $C_t = \tanh(\tilde{C}_t) \in [-1, 1]^{212}$, the convex combination $S_t = (1 - g_t) \odot S_{t-1} + g_t \odot C_t$ strictly guarantees $S_t \in [-1, 1]^{212}$ for all $t \ge 0$, eliminating coordinate explosion over $T=50$ rollouts.
     2. *Pre-Update Snapshot Atomicity:* DAG execution order guarantees that both $u_a$ and $u_b$ are extracted prior to any state modification, precluding sequential overwrite artifacts.
   * *Learned (Empirical) Preservation (Zero Structural Guarantee):*
     Unlike Candidate 1 and Candidate 3 (which provide structural register isolation $\|\Delta s_r\| = 0.0$ by slice assignment), Candidate 2 maintains a monolithic, unpartitioned 212-dimensional state. There is **zero structural guarantee** that unswapped registers remain unperturbed. Coordinate isolation relies entirely on learned orthogonal superposition and localized gating. This empirical hypothesis is directly tested and falsifiable in Stage 2 (Address Binding) and Stage 6 (Long Persistence).
7. **Architectural Components and Trainable Parameter Reconciliation:**
   * *Fixed by Architecture:* Harmonic basis $\Phi(v) \in \mathbb{R}^{10}$, block rotation ALU mapping concatenated operand input $[\tilde{u}_a, \tilde{u}_b] \in \mathbb{R}^{20} \to c_{1, t} \in \mathbb{R}^{10}$ via $R(\theta_k)$, unit-norm projection, prototype cosine bank $\Phi(c)^\top \in \mathbb{R}^{10 \times 10}$.
   * *Learned Networks & Exact Parameter Counts:*
     - Readout query network $\text{MLP}_{read}$: Linear($216 \to 128$) + ReLU + Linear($128 \to 10$) = $27,776 + 1,290 = 29,066$ parameters.
     - Dual-payload write network $\text{MLP}_{write}$: Linear($32 \to 212$) + ReLU + Linear($212 \to 212$) = $6,996 + 45,156 = 52,152$ parameters.
     - Recurrent convex gate network $\text{MLP}_{gate}$: Linear($436 \to 212$) + ReLU + Linear($212 \to 212$) = $92,644 + 45,156 = 137,800$ parameters.
     - Total Trainable Parameters: $29,066 + 52,152 + 137,800 = 219,018$ parameters.
   * *Parameter Envelope Compliance:* 219,018 parameters is within $-3.24\%$ of the 226,358 reference target and strictly inside the registered $226,358 \pm 10\%$ envelope ($[203,722, 248,994]$). (Envelope compliance is established; exact parameter matching is not claimed).
* **Scientific Role:** Isolates Hypothesis B in the presence of continuous state coupling.

---

### 4.3 Candidate 3: Modular Slot Machine (MSM — Deterministic Analytical Reference)

Per PI Decision 1, Candidate 3 adopts **Option B**, removing the causally disconnected 43-dimensional auxiliary branch and classifying MSM as a parameter-free deterministic analytical reference for the documented task.

#### State Layout and Discrete Slot Memory
* **State Layout:** Recurrent state partitioned into $M=4$ discrete register slots $S_t = [s_{t,0}, s_{t,1}, s_{t,2}, s_{t,3}] \in \mathbb{R}^{4 \times 10}$. Each slot consists strictly of 10 harmonic cyclic value coordinates:
  $$s_{t,r} = z_{t,r} \in \mathbb{R}^{10} \quad (2K = 10, K = 5 \text{ frequencies})$$
  The total recurrent state dimension is exactly $4 \times 10 = 40$ dimensions. The 43-dimensional continuous auxiliary context coordinates ($h_{t,r}$), auxiliary networks ($\text{MLP}_{aux\_cand}, \text{MLP}_{aux\_gate}$), timestamps, and recency approximations proposed in earlier drafts are completely eliminated.
* **Initialization ($t=0$):** All register slots start canonically at value zero:
  $$z_{0,r} = \Phi(0) \in \mathbb{R}^{10} \quad \forall r \in \{0..3\}$$

#### Complete State Transitions
1. **`SET r_dest v`:**
   * Harmonic value update: $z_{t, r_{dest}} = \Phi(v) \in \mathbb{R}^{10}$.
   * Non-destination slot preservation: $z_{t, r} = z_{t-1, r}$ for all $r \ne r_{dest}$ ($\|\Delta z_r\| = 0.0$ guaranteed by discrete slot assignment).
2. **`SWAP r_dest r_src`:**
   * Harmonic value swap: $z_{t, r_{dest}} = z_{t-1, r_{src}}$ and $z_{t, r_{src}} = z_{t-1, r_{dest}}$.
   * Non-destination slot preservation: $z_{t, r} = z_{t-1, r}$ for all $r \notin \{r_{dest}, r_{src}\}$ ($\|\Delta z_r\| = 0.0$ guaranteed).
3. **`READ r_dest`:**
   * State preservation: $S_t = S_{t-1}$ across all slots (zero state mutation; reading does not alter stored values).
   * Deterministic output decoding: $\hat{y}_t = \arg\max_{c \in \{0..9\}} \Phi(c)^\top z_{t-1, r_{dest}}$.
4. **`ADD r_dest r_src`:**
   * Hard multiplexing extracts operand slots: $u_a = z_{t-1, r_{dest}}$, $u_b = z_{t-1, r_{src}}$.
   * Harmonic ALU computes 2D block rotations: $c_t^{(k)} = R(\theta_b^{(k)}) u_a^{(k)}$ across $k \in \{1..5\}$.
   * Unit-norm projection bounds numerical drift: $\tilde{z}_{t, r_{dest}}^{(k)} = c_t^{(k)} / \max(\|c_t^{(k)}\|_2, \epsilon)$.
   * Harmonic write: $z_{t, r_{dest}} = \tilde{z}_{t, r_{dest}}$.
   * Non-destination slot preservation: $z_{t, r} = z_{t-1, r}$ for all $r \ne r_{dest}$ ($\|\Delta z_r\| = 0.0$ guaranteed).

#### Parameter Reconciliation and Classification
* **Trainable Parameter Count:** Exactly **0 parameters**:
  - Harmonic cyclic ALU: Fixed analytical 2D block rotations ($0$ trainable parameters).
  - Output decision head: Fixed prototype cosine bank $\Phi(c)^\top$ ($0$ trainable parameters).
  - Memory routing: Discrete hard multiplexing ($0$ trainable parameters).
  - Total Trainable Parameters: **0 parameters**.
* **Parameter Envelope Status:** **Exempt** from the neural trainable parameter envelope ($226,358 \pm 10\%$) because it is a parameter-free analytical reference.
* **Scientific Role & Rigorous Classification:**
  - Classified strictly as a **parameter-free deterministic analytical reference / exact-rule control** for the documented task.
  - Defines the analytical upper performance standard achievable by composing discrete slot routing with exact cyclic arithmetic on the documented Register Tracking task.
  - *Explicit Terminology Restrictions:* Per PI Decision 1, MSM is **NOT** described as a trainable neural candidate, nor is it characterized as a general Bayes-optimal solution across arbitrary tasks. It provides a task-specific exact-rule reference against which trainable candidates (Slot-MGU, Cyclic-MGU) and the baseline Vanilla GRU are evaluated.

---

### 4.4 Mandatory Scientific Controls
* **Control 1: Frozen Model E (Historical Reference):** `experiments/checkpoints/E_candidate_seed{42..46}_3000steps.pt`. Evaluated strictly read-only on all benchmark stages to provide direct continuity with Phase 1. Parameter count: 226,358 (reference baseline).
* **Control 2: Standard Recurrent Baseline (Vanilla GRU):** Parameter-envelope-compliant standard GRU baseline trained from scratch under identical protocols. Verifies that the task requires specialized inductive biases.
  - *Observation Encoder ($22 \to 64 \to 256$):* Linear($22 \to 64$) [weight $22 \times 64 = 1,408$, bias $64 \implies 1,472$] + GELU + Linear($64 \to 256$) [weight $64 \times 256 = 16,384$, bias $256 \implies 16,640$] + LayerNorm($256$) [weight $256$, bias $256 \implies 512$] = $1,472 + 16,640 + 512 = \mathbf{18,624}$ parameters. (Reconciled from Phase 1's 21-dim observation: $22 \times 64 + 64$ vs $21 \times 64 + 64$, adding +64 parameters).
  - *PyTorch GRU Cell ($H=160$):* Standard 3-gate recurrent update with input dimension $256$ and hidden size $H=160$: $3 \times (256 \times 160 + 160 + 160^2 + 160) = 3 \times (40,960 + 160 + 25,600 + 160) = 3 \times 66,880 = \mathbf{200,640}$ parameters.
  - *Interface-Specific Readout Heads & Total Parameter Counts:*
    1. **Stage 1 Dedicated Interface (10-class Head):** Linear($160 \to 10$) [weight $160 \times 10 = 1,600$, bias $10 \implies 1,610$]. Total trainable parameters: $18,624 + 200,640 + 1,610 = \mathbf{220,874}$ parameters ($-2.42\%$ vs Model E $226,358$).
    2. **General Composed Sequence Interface (40-logit Head for Stages 2–6):** Linear($160 \to 40$) (predicting 4 register values $\times 10$ classes each = 40 logits) [weight $160 \times 40 = 6,400$, bias $40 \implies 6,440$]. Total trainable parameters: $18,624 + 200,640 + 6,440 = \mathbf{225,704}$ parameters ($-0.29\%$ vs Model E $226,358$).
    3. **Historical Phase 1 Baseline Note:** The figure of 225,640 parameters previously cited reflected Phase 1's 21-dimensional input token ($18,560 + 200,640 + 6,440$). Updating to Phase 1.6's 22-dimensional token shifts the general configuration to 225,704 parameters (+64 parameters).
  - *Parameter Envelope Compliance:* Both configurations ($220,874$ and $225,704$ parameters) fall strictly inside the registered envelope of $[203,722, 248,994]$ ($226,358 \pm 10\%$). (Envelope compliance is established; exact parameter matching is not claimed).
  - *Ratified Interface Configuration (PI Decision 5):* The Principal Investigator has formally selected and ratified the dedicated 10-class readout head configuration for Stage 1, documented at **220,874 trainable parameters** ($-2.42\%$ vs Model E $226,358$), preserving the parameter-accounting rationale ($18,624$ encoder + $200,640$ GRU + $1,610$ head). The shared 40-logit configuration (225,704 parameters) is documented for reference but not adopted for Stage 1.
* **Control 3: Deterministic Symbolic Oracle:** A programmatic Python reference implementing the exact Register Machine rules. Verifies dataset validity, split disjointness, and ceiling metrics (Bayes = 1.000).

---

### 4.5 Complete Mathematical Specification of the Cyclic Manifold

#### Scalar Encoding
A scalar integer register value $v \in \{0, \dots, 9\}$ is embedded into $K=5$ harmonic frequencies:
$$\phi_k(v) = \begin{bmatrix} \cos\left(\frac{2\pi k v}{10}\right) \\ \sin\left(\frac{2\pi k v}{10}\right) \end{bmatrix} \in \mathbb{R}^{2}, \quad k \in \{1, \dots, K\}$$
$$\Phi(v) = [\phi_1(v)^\top, \phi_2(v)^\top, \dots, \phi_K(v)^\top]^\top \in \mathbb{R}^{2K} \quad (2K = 10)$$

#### ALU Operation (Modular Addition)
Modular addition $(a + b) \pmod{10}$ is computed via 2D rotation matrix multiplication across each harmonic frequency $k$:
$$R(\theta_k) = \begin{bmatrix} \cos\theta_k & -\sin\theta_k \\ \sin\theta_k & \cos\theta_k \end{bmatrix}$$
Given input 2D blocks $u_a^{(k)} = [\cos\alpha_k, \sin\alpha_k]^\top$ and $u_b^{(k)} = [\cos\beta_k, \sin\beta_k]^\top$:
$$c_t^{(k)} = \begin{bmatrix} \cos\alpha_k \cos\beta_k - \sin\alpha_k \sin\beta_k \\ \sin\alpha_k \cos\beta_k + \cos\alpha_k \sin\beta_k \end{bmatrix} = \begin{bmatrix} \cos(\alpha_k + \beta_k) \\ \sin(\alpha_k + \beta_k) \end{bmatrix}$$

#### Analytical Prior vs. Learned Function
* **Explicit Specification:** In Candidates 2 and 3, group addition on $S^1$ is hardwired into the ALU as an **explicit algebraic architectural prior**, NOT an emergent function induced from data.
* **Separation of Stage 1 Verification Tracks:**
  - *Candidate 1 (Slot-MGU) & Control 2 (Vanilla GRU):* Evaluated under **Stage 1A (Trained End-to-End Pilot)**, testing whether generic recurrent memory and learned ALUs can acquire modular arithmetic across actual register-state and convex gating pathways.
  - *Candidates 2 (Cyclic-MGU) & 3 (MSM):* Evaluated under **Stage 1B (Deterministic Component Pre-Flight Unit Test)**, verifying strictly the algebraic correctness and phase scaling of the fixed cyclic rotation ALU ($R(\theta_b) \Phi(a) = \Phi((a + b) \pmod{10})$) with zero training steps.
  - *Explicit Scope Limitation for Candidate 2:* Stage 1B exercises **strictly the algebraic rotation core**; it does **NOT** exercise or validate Candidate 2's learned read queries ($\text{MLP}_{read}$), monolithic write candidate generator ($\text{MLP}_{write}$), or convex gating ($\text{MLP}_{gate}$). The complete learned read-compute-write pathway of Candidate 2 is evaluated empirically for the first time in **Stage 2 (Address Binding)** and **Stage 3 (Composed Sequence Execution)**. Passing Stage 1B is never presented or claimed as evidence of end-to-end learned capability.

#### State Normalization and Numerical Drift Mitigation
To prevent numerical drift or magnitude collapse across recurrence horizons ($T=50$), every 2D harmonic block is projected back to the unit circle $S^1$ after each write:
$$\tilde{z}_k = \frac{z_k}{\max(\|z_k\|_2, \epsilon)}, \quad \epsilon = 10^{-6}$$
This guarantees isometric norm preservation ($\|\tilde{z}_k\|_2 = 1.0$) and eliminates amplitude explosion or decay over extended trajectories.

#### Value Decoding
A prototype cosine bank extracts the categorical distribution:
$$P(y = c \mid z) = \text{softmax}\left(\frac{\Phi(c)^\top z}{\tau}\right), \quad \hat{y} = \arg\max_{c \in \{0..9\}} \Phi(c)^\top z$$

---

## 5. Staged Benchmark and Split Design

To prevent conflating distinct computational capabilities into a single aggregate score, Phase 1.6 defines a rigorous **6-stage progressive evaluation ladder**:

```
                               PROGRESSIVE EVALUATION LADDER
                                             │
    Stage 1: Primitive Arithmetic ───────────┴───────────► Can it do modular wrap? (Disjoint 64/16/20 split)
    Stage 2: Address-Value Binding ──────────┴───────────► Can it route registers cleanly? (Behavioral Read EM)
    Stage 3: Composed Short (T=10) ──────────┴───────────► Can it compose short traces? (Sequence EM)
    Stage 4: Systematic Generalisation ──────┴───────────► Does it generalize to held-out pairs? (b in {0,5})
    Stage 5: Length Extrapolation (T=20..50) ┴───────────► Does it execute long horizons? (T=20..50 EM)
    Stage 6: Long Persistence ───────────────┴───────────► Are untouched registers preserved? (K=20 Read EM)
```

### Stage 1: Primitive Modular Arithmetic (Separation of Verification Tracks: 1A vs. 1B)
* **Objective & Structural Role:** Evaluates whether candidate inductive biases compute primitive modular addition $(a + b) \pmod{10}$, with strict, principled separation between learned end-to-end trace induction and deterministic algebraic component unit testing.
  * **Separation of Verification Tracks:**
    To eliminate protocol asymmetry, Stage 1 is partitioned into two structurally distinct evaluation tracks. Passing an isolated algebraic unit test is **never** presented, claimed, or counted as equivalent to passing a trained end-to-end evaluation.
    - **Track 1A (Trained End-to-End Pilot):** For architectures with learned ALUs or generic recurrent memory (Candidate 1: Slot-MGU; Control 2: Vanilla GRU). Tests whether the full learned memory, gating, and update pathways can acquire modular arithmetic end-to-end across a canonical 3-step execution trace.
    - **Track 1B (Deterministic Component Pre-Flight Unit Test):** For architectures incorporating hardwired cyclic group priors (Candidate 2: Cyclic-MGU; Candidate 3: MSM). Evaluates analytical correctness and numerical implementation of the parameter-free $S^1$ rotation arithmetic ($R(\theta_b) \Phi(a) = \Phi((a + b) \pmod{10})$) with zero training steps.
  * **Canonical 3-Step Execution Trace (Track 1A & Track 1B-C3):** For each operand pair $(a, b) \in \{0..9\}^2$, the addition task is formulated as a standardized minimal register execution trace:
    1. Step 1 ($t=1$): $\text{SET } r_0 \; a$ (loads immediate scalar $a$ into slot 0).
    2. Step 2 ($t=2$): $\text{SET } r_1 \; b$ (loads immediate scalar $b$ into slot 1).
    3. Step 3 ($t=3$): $\text{ADD } r_0 \; r_1$ (computes modular sum into slot 0).
* **Architecture-Specific Interface Mapping and Track Assignment:**
  * **Track 1A — Candidate 1 (Slot-MGU):**
    - *Nature:* Empirical parameter-training across 5 random seeds (5 runs).
    - *State Initialization:* $S_0 = \mathbf{0}_{4 \times 53}$.
    - *Input Construction:* Three successive instruction tokens $x_1, x_2, x_3 \in \{0, 1\}^{22}$:
      $$x_1 = [e_{\text{SET}}, e(r_0), \mathbf{0}_4, e(a)], \quad x_2 = [e_{\text{SET}}, e(r_1), \mathbf{0}_4, e(b)], \quad x_3 = [e_{\text{ADD}}, e(r_0), e(r_1), \mathbf{0}_{10}]$$
    - *Update Pathway:*
      - At $t=1$: $u_a = s_{0, 0} = \mathbf{0}_{53}, u_b = \mathbf{0}_{53}$. Input $I_1 = [u_a, u_b, x_1] \in \mathbb{R}^{128}$. Convex update modifies slot 0: $s_{1, 0} = (1 - g_1) \odot \mathbf{0}_{53} + g_1 \odot \text{MLP}_{cand}(I_1)$.
      - At $t=2$: $u_a = s_{1, 1} = \mathbf{0}_{53}, u_b = \mathbf{0}_{53}$. Input $I_2 = [u_a, u_b, x_2] \in \mathbb{R}^{128}$. Convex update modifies slot 1 ($s_{2, 1}$), while slot 0 is preserved identically ($s_{2, 0} = s_{1, 0}$).
      - At $t=3$: Discrete routing retrieves $u_a = s_{2, 0}$ and $u_b = s_{2, 1}$. Input $I_3 = [u_a, u_b, x_3] \in \mathbb{R}^{128}$. Convex update produces $s_{3, 0} = (1 - g_3) \odot s_{2, 0} + g_3 \odot \text{MLP}_{cand}(I_3)$. Non-addressed slots ($r \in \{1, 2, 3\}$) undergo zero mutation ($\|\Delta s_r\| = 0.0$).
    - *Readout & Prediction:* Evaluated strictly at $t=3$ from destination slot $r_0$ via linear classification head:
      $$\hat{y} = \arg\max_{c \in \{0..9\}} \text{Head}(s_{3, 0}) \in \{0..9\}, \quad \text{Head}: \mathbb{R}^{53} \to \mathbb{R}^{10}$$
    - *Training Objective:* Multi-class cross-entropy loss applied exclusively to step 3 logits on the 64 seen training pairs (zero loss on steps $t=1, 2$):
      $$\mathcal{L} = -\log \text{softmax}(\text{Head}(s_{3, 0}))_y, \quad y = (a + b) \pmod{10}$$
    - *Gating & Pass Criterion:* Minimum validation CE loss checkpoint on `VAL-SEEN` (16 pairs) must achieve $acc_{val} \ge 0.950$ across $\ge 4/5$ seeds (zero seeds $< 0.250$). Pilot weights are discarded post-pilot.
  * **Track 1A — Mandatory Control 2 (Vanilla GRU Baseline):**
    - *Nature:* Empirical parameter-training across 5 random seeds (5 runs).
    - *Input Sequence:* Receives the identical sequence of instruction tokens $x_1, x_2, x_3 \in \{0, 1\}^{22}$.
    - *State Initialization:* $h_0 = \mathbf{0}_{160}$.
    - *Update Pathway:* Token encoder ($22 \to 64 \to 256$ + LayerNorm($256$): 18,624 parameters), standard PyTorch GRU cell recurrence ($256 \to H=160$: 200,640 parameters).
    - *Readout & Head Contract (Ratified Dedicated Head, PI Decision 5):*
      Evaluated at $t=3$ via dedicated classification head ($160 \to 10$: 1,610 parameters) on $h_3$. Total trainable parameters: $18,624 + 200,640 + 1,610 = \mathbf{220,874}$ parameters ($-2.42\%$ vs Model E $226,358$), strictly complying with the $[203,722, 248,994]$ envelope. Per PI Decision 5, this dedicated configuration is formally selected and ratified. (The alternative 40-logit sliced head at 225,704 parameters is not adopted for Stage 1).
    - *Training Objective:* Cross-entropy loss $\mathcal{L}_{CE}(\text{Head}(h_3), (a + b) \pmod{10})$ on the 64 seen training pairs at step $t=3$ (zero loss on steps $t=1, 2$).
    - *Gating & Pass Criterion:* Checkpoint on `VAL-SEEN` (16 pairs) must achieve $acc_{val} \ge 0.950$ across $\ge 4/5$ seeds (zero seeds $< 0.250$). Pilot weights discarded post-pilot.
  * **Track 1B — Architectural Component Pre-Flight Unit Tests (Candidates 2 & 3):**
    - *Nature:* Deterministic analytical unit test (0 training steps, forward numerical check, $< 1$ second).
    - *Architectural Asymmetry & Distinct Test Scopes:* Candidates 2 and 3 feature fundamentally different architectural mechanisms. Their Track 1B tests are structurally distinct and must not be conflated as identical evaluations.
    - **Track 1B-C2 (Isolated Cyclic ALU Operator Unit Test — Candidate 2 Cyclic-MGU):**
      1. *Components and Data Pathways Exercised:* Strictly the isolated parameter-free cyclic rotation ALU mapping concatenated 20-dimensional harmonic operand inputs $[\Phi(a), \Phi(b)] \in \mathbb{R}^{20}$ (composed of two 10-dimensional harmonic embeddings $\Phi(a), \Phi(b) \in \mathbb{R}^{10}$ across $K=5$ two-dimensional blocks) to the 10-dimensional harmonic rotation output $\Phi((a + b) \pmod{10}) \in \mathbb{R}^{10}$ via block-diagonal 2D rotations $R(\theta_b^{(k)}) \Phi(a)^{(k)}$, and the fixed 10-dimensional cosine prototype value decoding bank $\Phi(c)^\top \in \mathbb{R}^{10 \times 10}$.
      2. *Components Not Exercised:* Learned read query network ($\text{MLP}_{read}: 216 \to 128 \to 10$), monolithic write candidate generator ($\text{MLP}_{write}: 32 \to 212 \to 212$), convex gating network ($\text{MLP}_{gate}: 436 \to 212 \to 212$), continuous state-memory vector $S_t \in \mathbb{R}^{212}$, write blending, and register persistence across steps.
      3. *Test Input Domain & Pass Criterion:* Strictly the 16 validation pairs $(a, b) \in \text{VAL-SEEN}$, as deterministically partitioned from the 80-pair seen domain ($b \in \{1..4, 6..9\}, a \in \{0..9\}$) in Section 5.1 (two distinct pairs per eligible column $b$, spanning all ten target classes). The remaining 64 seen pairs comprise `TRAIN-SEEN` (unexercised in Track 1B), and the 20-pair `TEST-HELDOUT` ($b \in \{0, 5\}$) remains strictly withheld. Pass criterion is exact numerical accuracy $acc = 1.000$ (16/16 correct). Any floating-point error resulting in incorrect class argmax is an immediate implementation failure.
      4. *What It Establishes vs. Does Not Establish:*
         - *Establishes:* The mathematical formula and code implementation of the harmonic phase embedding $\Phi(\cdot)$ and 2D rotation blocks $R(\theta_k)$ correctly satisfy the abelian group identity $R(\theta_b) \Phi(a) = \Phi((a + b) \pmod{10})$ under cosine decoding.
         - *Does NOT Establish:* It does **NOT** establish that Candidate 2 can read register values from its monolithic continuous state, route them into the ALU, generate write candidates, update state via convex gating, preserve unwritten register values, or execute multi-step programs. **Candidate 2's learned state-memory and update pathways are completely unexercised and unvalidated by this test.**
      5. *Advancement Effect:* Passing Track 1B-C2 is a necessary pre-flight sanity check permitting progression to Stage 2. Candidate 2's learned read, write, and gating networks face their first empirical capability gate in **Stage 2 (Address Binding)** ($EM_{read, val} \ge 0.950$) and **Stage 3 (Composed Short Traces)** ($EM_{heldout} \ge 0.700$). If Candidate 2 fails Stage 2, it is immediately halted.
    - **Track 1B-C3 (Deterministic Value-Coordinate Slot-and-Arithmetic Pipeline Unit Test — Candidate 3 MSM Reference):**
      1. *Components and State Coordinates Covered:* Strictly the **40-dimensional deterministic slot-and-arithmetic pipeline**: the 4 discrete register slots $S_t = [s_{t,0}, s_{t,1}, s_{t,2}, s_{t,3}] \in \mathbb{R}^{4 \times 10}$ consisting purely of 10-dimensional harmonic cyclic value coordinates $z_{t,r} \in \mathbb{R}^{10}$ ($2K=10$, $K=5$ frequencies), hardwired instruction parsing and discrete routing for SET and ADD ($u_a = z_{t-1, r_{dest}}$, $u_b = z_{t-1, r_{src}}$), the canonical 3-step execution pipeline ($t=1: \text{SET } r_0 \; a; t=2: \text{SET } r_1 \; b; t=3: \text{ADD } r_0 \; r_1$), deterministic slot-to-slot cyclic rotation ALU with unit-norm projection, step-by-step measured preservation of non-destination slots (exact zero value-coordinate mutation $\|\Delta z_r\| = 0.0$ on unwritten slots), and the fixed cosine prototype decoding bank on destination slot $r_0$.
      2. *Auxiliary Branch Elimination (Option B):* Per PI Decision 1 adopting Option B, the 43-dimensional auxiliary coordinates ($h_{t,r}$), auxiliary networks ($\text{MLP}_{aux\_cand}, \text{MLP}_{aux\_gate}$), timestamps, and recency approximations are completely eliminated from the architecture. Candidate 3 is evaluated strictly as a parameter-free deterministic analytical reference.
      3. *Test Input Domain & Step-by-Step Measured Preservation Invariants:* Evaluated on the canonical 3-step execution trace across the 16 pairs $(a, b) \in \text{VAL-SEEN}$ (deterministically partitioned from the 80-pair seen domain in Section 5.1; `TEST-HELDOUT` strictly withheld). The exact measured preservation invariant for each step of the trace is:
         - Step 1 ($t=1: \text{SET } r_0 \; a$): destination slot $r_0$ is written ($z_{1, 0} = \Phi(a)$); non-destination slots $r_1$, $r_2$, and $r_3$ maintain zero value-coordinate mutation ($\|\Delta z_r\| = 0.0$ for $r \in \{1, 2, 3\}$).
         - Step 2 ($t=2: \text{SET } r_1 \; b$): destination slot $r_1$ is written ($z_{2, 1} = \Phi(b)$); non-destination slots $r_0$, $r_2$, and $r_3$ maintain zero value-coordinate mutation ($\|\Delta z_r\| = 0.0$ for $r \in \{0, 2, 3\}$, preserving slot $r_0 = \Phi(a)$).
         - Step 3 ($t=3: \text{ADD } r_0 \; r_1$): destination slot $r_0$ is written ($z_{3, 0} = \Phi((a + b) \pmod{10})$); source slot $r_1$ is read as operand but NOT written, so non-destination slots $r_1$, $r_2$, and $r_3$ maintain zero value-coordinate mutation ($\|\Delta z_r\| = 0.0$ for $r \in \{1, 2, 3\}$).
      4. *Explicit Pass Criterion:* Evaluated across all 16 pairs in `VAL-SEEN`:
         - Exact numerical accuracy $acc = 1.000$ (16/16 correct) at step $t=3$ decoded from destination slot $r_0$.
         - Exact zero value-coordinate mutation $\|\Delta z_r\| = 0.0$ on slots $r \in \{1, 2, 3\}$ after `SET r0 a`.
         - Exact zero value-coordinate mutation $\|\Delta z_r\| = 0.0$ on slots $r \in \{0, 2, 3\}$ after `SET r1 b`.
         - Exact zero value-coordinate mutation $\|\Delta z_r\| = 0.0$ on slots $r \in \{1, 2, 3\}$ after `ADD r0 r1`.
      5. *What It Establishes vs. Does Not Establish:*
         - *Establishes:* The deterministic slot-and-arithmetic pipeline (discrete routing, value-coordinate persistence, cyclic rotation ALU, and untouched-slot value preservation across read and write operations) functions without software defects or numerical drift.
         - *Does NOT Establish:* It does **NOT** establish that neural sequence models trained with gradient descent can learn or reproduce this discrete algorithmic execution. MSM serves as a deterministic reference standard, not as a trained neural candidate.
      6. *Advancement Effect:* Passing Track 1B-C3 verifies the deterministic infrastructure of the exact-rule reference. MSM advances to subsequent evaluation stages as a parameter-free deterministic baseline.
    - *Prohibition of Unsupported Cross-Candidate Conclusions:* Because Track 1A measures empirical statistical learning under gradient descent, Track 1B-C2 evaluates an isolated algebraic operator, and Track 1B-C3 evaluates a deterministic primary value-coordinate slot pipeline, **all cross-candidate capability rankings and performance comparisons on Stage 1 are strictly prohibited**. Passing one's designated track is an independent prerequisite for advancement, establishing no cross-candidate superiority.
  * **Ratified Stage 1 Protocol (PI Decision 3):**
    - The Principal Investigator has formally ratified **Approach 2 (Structural Separation of Tracks)** as the approved protocol: Track 1A is retained for the trained Slot-MGU and Vanilla GRU pilot, and Track 1B for deterministic verification of Cyclic-MGU and MSM. (The alternative Approach 1 expanding the pilot by +5 trained Candidate 2 runs is not adopted).
* **Deterministic Partitioning Algorithm (Zero Overlap across 100 Pairs):**
  * **Held-Out Test (TEST-HELDOUT, 20 pairs):** Strictly all pairs with operand value $b \in \{0, 5\}$ across all $a \in \{0..9\}$ ($2 \times 10 = 20$ pairs). Strictly withheld from all training, validation, early stopping, pilot progression gating, and hyperparameter tuning. It serves exclusively for the final preregistered systematic generalization evaluation.
  * **Seen Domain (80 pairs):** All pairs where $b \in \{1, 2, 3, 4, 6, 7, 8, 9\}$ and $a \in \{0..9\}$.
  * **Deterministic Seen-Domain Partition:**
    Using the sorted eligible operand values $b = [1, 2, 3, 4, 6, 7, 8, 9]$ indexed by $i \in \{0, \dots, 7\}$:
    1. For each column index $i \in \{0, \dots, 7\}$, select two distinct validation target classes:
       $$y_0 = (2 \cdot i) \pmod{10}, \quad y_1 = (2 \cdot i + 1) \pmod{10}$$
    2. Derive their corresponding input operands using modular subtraction:
       $$a_0 = (y_0 - b) \pmod{10}, \quad a_1 = (y_1 - b) \pmod{10}$$
    3. The resulting pairs $(a_0, b)$ and $(a_1, b)$ form the validation set (**VAL-SEEN**, exactly $8 \times 2 = 16$ pairs). Because the target sequence $(y_0, y_1)$ for $i=0..7$ spans $\{0, 1\}, \{2, 3\}, \{4, 5\}, \{6, 7\}, \{8, 9\}, \{0, 1\}, \{2, 3\}, \{4, 5\}$, **all ten target classes $y \in \{0..9\}$ are represented in the validation set**, with exact non-uniform class frequencies:
       - Target classes $0, 1, 2, 3, 4, 5$ occur **twice** (12 pairs total).
       - Target classes $6, 7, 8, 9$ occur **once** (4 pairs total).
       Total: $12 + 4 = 16$ validation pairs.
    4. The remaining eight values of $a$ per column form the training set (**TRAIN-FIT**, exactly $8 \times 8 = 64$ pairs). Training fit is reported strictly as training convergence, never described as an independent test set.
  * **Disjointness & Verification Specification:**
    A mandatory lightweight split-verification test checks:
    1. Exact partition counts: $|D_{\text{train}}| = 64$, $|D_{\text{val}}| = 16$, $|D_{\text{heldout}}| = 20$.
    2. Zero pairwise intersections: $D_{\text{train}} \cap D_{\text{val}} = \emptyset$, $D_{\text{train}} \cap D_{\text{heldout}} = \emptyset$, $D_{\text{val}} \cap D_{\text{heldout}} = \emptyset$.
    3. Complete domain coverage: $D_{\text{train}} \cup D_{\text{val}} \cup D_{\text{heldout}} = \{0..9\}^2$ (all 100 pairs).
    4. Full representation of all ten target classes $\{0..9\}$ in validation.
    5. Exact per-class frequencies in validation match $\{0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1\}$.
* **Metrics:**
  * *Pilot Progression:* Validation accuracy ($acc_{val}$) on VAL-SEEN ($\ge 0.950$).
  * *Final Reporting:* Macro-averaged class-balanced accuracy ($acc_{heldout}$) on TEST-HELDOUT ($\ge 0.950$).

### Stage 2: Address-Value Binding & Variable Routing
* **Objective:** Test multi-register storage, retrieval, and identity preservation without arithmetic.
* **Format:** Multi-step sequences ($T=10$) consisting strictly of `SET r_i v`, `SWAP r_i r_j`, and `READ r_i`. Zero addition operations.
* **Splits:** Sequence-disjoint TRAIN (6,000 sequences), VAL (1,200 sequences), TEST (1,200 sequences). The TEST split is strictly withheld from all pilot progression gating, model selection, early stopping, and hyperparameter tuning.
* **Metrics:**
  * *Pilot Progression:* Behavioral Read Exact Match on the validation split ($EM_{read, val}$) at $t=10$ ($\ge 0.950$).
  * *Final Reporting:* Behavioral Read Exact Match on the test split ($EM_{read}$) at $t=10$ ($\ge 0.990$). Untouched slot internal representation drift ($\|\Delta s\|$) is reported as an auxiliary diagnostic.

### Stage 3: Composed Short Register Execution (T=10, Regime A In-Distribution)
* **Objective:** Standard multi-step register execution composed of SET, SWAP, and ADD across 4 registers ($T=10$) under matched in-distribution training conditions to historical Model E.
* **Training Regime & Splits (Regime A — In-Distribution Composed Training Pool):**
  * *Training Set:* 6,000 sequence-disjoint sequences ($T=10$) where operand values $a, b \in \{0..9\}$ are sampled uniformly across all operations without masking.
  * *Validation Set:* 1,200 sequence-disjoint in-distribution sequences (governing validation loss and early checkpoint selection).
  * *Test Set:* Canonical 1,200 sequences $\times$ 10 independent test streams (matching the exact canonical evaluation set used for Model E in Phase 1).
* **Metrics:** Sequence Exact Match ($EM_{T=10}$: all outputs correct across all 10 steps).
* **Role in Research Program:** Provides an unconfounded, directly comparable benchmark against historical Model E ($0.755 \pm 0.019$) and produces the foundation checkpoints evaluated zero-shot in Stage 5 (Length Extrapolation) and Stage 6 (Long Persistence).

### Stage 4: Systematic Generalization (Held-Out Operands on Composed Traces, Regime B Masked Training)
* **Objective:** Test out-of-distribution systematic generalization and rule transfer on held-out operand combinations in multi-step composed sequences without data leakage.
* **Clarification of Notation:** In instructions $(op, r_{dest}, r_{src})$, $b$ denotes the **runtime source-register value** $v(r_{src}) \in \{0, 5\}$, NOT the register address index $r_{src} \in \{0..3\}$.
* **Training Regime & Splits (Regime B — Held-Out Masked Composed Training Pool):**
  * *Training Set:* 6,000 sequence-disjoint sequences ($T=10$) generated under strict held-out exclusion: sequences **never** contain an ADD transition where the source register holds operand value $b \in \{0, 5\}$ at execution time.
  * *Validation Set:* 1,200 sequence-disjoint masked sequences (satisfying $b \notin \{0, 5\}$ for checkpoint selection).
  * *Test Set:* 1,200 held-out test sequences strictly conditioned to contain at least one ADD transition where $v(r_{src}) \in \{0, 5\}$.
* **Reconciliation of Stage 3 and Stage 4 Training Regimes (Leakage Prevention & Protocol Integrity):**
  * *Non-Equivalence of Datasets and Checkpoints:* Stages 3 and 4 **cannot** share a single training dataset or trained checkpoint.
    1. *Leakage in Stage 4 under In-Distribution Training:* If a model trained on the in-distribution pool (Regime A) were evaluated on Stage 4, transitions with $b \in \{0, 5\}$ would have occurred during training (~20% frequency). Stage 4 would suffer from catastrophic training leakage, invalidating the held-out generalization claim.
    2. *Confounded Baseline Comparison in Stage 3 under Masked Training:* If a model trained on the masked pool (Regime B) were evaluated on Stage 3, the model would have been trained with 20% of operand transitions missing, introducing a severe training support disadvantage relative to Model E (which had 100% operand coverage).
    3. *Separate Training Regimes Mandated:* To satisfy both research goals simultaneously—unconfounded Model E comparison on composed execution and leak-free systematic transfer on held-out operands—**separate training regimes and checkpoints (Regime A and Regime B) are required**.
  * *Alternative Design for Review Gate 1 (Option B — Unified Masked Regime):* If compute constraints strictly prohibit running two composed training regimes, the project may adopt a single masked training regime (Regime B only, 45 total training runs across the campaign). Under Option B, Stage 3 is evaluated on canonical test sequences with the explicit qualification that historical Model E serves as an asymmetric upper-bound reference. (The primary preregistered specification adopts the Dual-Regime design A+B to preserve complete scientific rigor).
* **Mandatory Joint Metrics:** Both transition balanced accuracy AND sequence exact match are co-primary requirements:
  1. *Held-Out Transition Balanced Accuracy ($acc_{heldout}$):* Macro-averaged recall on ADD transitions with operand value $b \in \{0,5\}$ in composed traces. Threshold: $\ge 0.800$.
  2. *Held-Out Sequence Exact Match ($EM_{heldout}$):* Whole-sequence exact match on held-out test traces. Threshold: $\ge 0.700$.
* **Error-Conditioned Evaluation Population:** Evaluated on test traces containing at least one ADD transition where $v(r_{src}) \in \{0, 5\}$ ($N=1,200$ sequences). Both metrics must pass; if transition accuracy is $\ge 0.800$ but sequence EM $< 0.700$, the architecture fails systematic generalization due to sequence accumulation errors.

### Stage 5: Length Extrapolation (T=20 Mandatory Gate; T=30, T=50 Diagnostic Probes)
* **Objective:** Zero-shot execution on sequence horizons longer than the training length ($T_{train}=10$).
* **Splits:** Models trained strictly on $T \le 10$; evaluated zero-shot on $T \in \{20, 30, 50\}$.
* **Primary Success Gate:** Evaluated strictly at horizon $T=20$: Sequence Exact Match $EM_{T=20} \ge 0.700$.
* **Secondary Diagnostic Probes:** Horizons $T=30$ (diagnostic benchmark: $EM_{T=30} \ge 0.500$) and $T=50$ (diagnostic benchmark: $EM_{T=50} \ge 0.300$) are reported as empirical degradation curves, but do NOT block Phase 2 authorization if the primary $T=20$ gate is satisfied.

### Stage 6: Long-Horizon Persistence & Interference Resistance
* **Objective:** Measure memory decay of an untouched register under active interference.
* **Format:** Register $r_0$ is initialized with value $v_0$ at $t=0$. For $t=1..K$, active ADD and SET operations are performed exclusively on registers $r_1, r_2, r_3$. At step $t=K+1$, register $r_0$ is queried via `READ r_0`.
* **Horizons:** $K \in \{5, 10, 20, 30\}$. Primary threshold evaluated at $K=20$.
* **Metric:** Behavioral Read Exact Match ($EM_{read}$) of $r_0$ at step $K+1$.

---

## 6. Preregistered Evaluation Protocol

### 6.1 Protocol Invariants
1. **Validation-Only Model Selection:** Best checkpoint selection is governed strictly by validation loss or validation accuracy on the designated VAL split. Test data (canonical TEST, held-out splits, extrapolation sets) shall NEVER be used for model selection, early stopping, or hyperparameter tuning.
2. **Five Independent Seeds:** All conditions are executed across 5 independent seeds (seeds 42, 43, 44, 45, 46). The seed governs data ordering, initialization, and minibatch sampling.
3. **Statistical Units:** The paired difference across seeds ($\Delta_s = \text{Candidate}_s - \text{Baseline}_s$) forms the primary statistical unit where comparable baselines exist. Means, standard deviations, per-seed sign consistency (e.g. 5/5 or 4/5 positive), and paired tests shall be reported.
4. **Data Disjointness:** Strictly enforced by `DisjointSplitManager`. Zero sequence overlap, zero common trajectory IDs, and byte-verified array hashes.

### 6.2 Single Normalized Training Protocol
To prevent optimization confounders across candidate architectures:
* **Optimizer:** AdamW with decoupled weight decay ($\text{lr} = 3 \times 10^{-3}$, $\text{weight\_decay} = 10^{-2}$), matching the qualified recipe established in Experiments AB–AH.
* **Minibatch Size:** Exactly 64 sequences per update.
* **Total Optimizer Updates:** Exactly 3,000 updates. Total training exposure: $3,000 \times 64 = 192,000$ sequence transitions.
* **Validation Cadence:** Validation loss and accuracy evaluated every 50 updates on the designated VAL split.
* **Checkpoint Selection:** Checkpoint corresponding to the minimum validation cross-entropy loss.
* **Composed Training Regimes:**
  - *Regime A (In-Distribution Composed Training):* 6,000 unmasked sequences ($a, b \in \{0..9\}$ sampled uniformly). Trained checkpoints are evaluated on Stage 3 (Canonical $T=10$ EM vs Model E), Stage 5 (Length Extrapolation $T=20..50$), and Stage 6 (Long Persistence $K=20$).
  - *Regime B (Held-Out Masked Composed Training):* 6,000 masked sequences with runtime $b \in \{0, 5\}$ strictly excluded from all ADD transitions. Trained checkpoints are evaluated on Stage 4 (Systematic Generalization on held-out operands).
  Both regimes use identical optimizer hyperparameters, minibatch sizes (64), update budgets (3,000 updates), and validation cadences (every 50 updates on their respective VAL split).

### 6.3 Pilot Stage Implementation & Timing Check
To conserve compute and enforce rapid falsification, a mandatory **cheap pilot stage** precedes full composed training:
* **Purpose of the Pilot:** The pilot functions strictly as an **inexpensive feasibility filter** to discard mechanically non-viable architectures prior to investing in composed sequence training. It provides evidence of basic representation and numerical stability, NOT proof of final generalization.
* **Firewall Against Final-Test Data Leakage:** In accordance with the protocol invariants (§6.1), held-out test data (`TEST-HELDOUT` with $b \in \{0, 5\}$ in Stage 1; `TEST` in Stage 2) shall NEVER be evaluated or unblinded during the pilot stage. Model progression, checkpoint selection, and feasibility gating are governed **strictly by validation metrics on the designated validation splits (`VAL-SEEN` for Stage 1, `VAL` for Stage 2)**.
* **Discarding of Pilot Parameters:** Because the multi-stage training curriculum (Hypothesis C) is explicitly deferred, **pilot-trained model weights are discarded**. Primary models for composed stages (Stages 3–6) are trained from **fresh random initialization** directly on composed sequences, preventing any architectural or parametric leakage from pilot training.
* **Candidate & Control Pilot Runs:**
  * **Candidate 1 (Slot-MGU):** Evaluated under **Track 1A (Trained End-to-End Pilot)** on Stage 1 (5 parameter-training runs fitting seen pairs across 5 seeds) and 5 training runs on Stage 2 (learning address binding across 5 seeds) = **10 runs**.
  * **Candidate 2 (Cyclic-MGU):** Evaluated under **Track 1B (Deterministic Component Pre-Flight Unit Test)** on Stage 1 (0 training steps, 1 forward pass, $< 1$ second) verifying exact $S^1$ rotation arithmetic on `VAL-SEEN`, and 5 training runs on Stage 2 (learning address binding across 5 seeds) = **5 runs + 1 deterministic check**.
    - *Scope Limitation & Non-Claim:* Track 1B evaluates strictly the isolated cyclic rotation core. It does **NOT** exercise or validate Candidate 2's learned read queries ($\text{MLP}_{read}$), monolithic write candidate generator ($\text{MLP}_{write}$), or convex gating ($\text{MLP}_{gate}$). The learned state-update pathway of Candidate 2 is evaluated empirically for the first time in Stage 2 (Address Binding, requiring $EM_{read, val} \ge 0.950$).
  * **Candidate 3 (MSM):** Evaluated under **Track 1B (Deterministic Component Pre-Flight Unit Test)** on Stage 1 (0 training steps, forward pass through parameter-free slot rotation pipeline, $< 1$ second, exact $acc = 1.000$ on `VAL-SEEN`, exact zero value-coordinate mutation $\|\Delta z_r\| = 0.0$ on unwritten slots), and forward-only deterministic verification on Stage 2 address binding (0 training steps) = **0 parameter-training runs + 2 deterministic checks**.
    - *Scope Limitation & Classification:* Per PI Decision 1 adopting Option B, MSM is a parameter-free deterministic analytical reference (0 trainable parameters). It undergoes deterministic numerical verification on the pilot splits without seed training.
  * **Mandatory Control 2 (Vanilla GRU):** Dedicated 10-class baseline (220,874 parameters) evaluated under **Track 1A (Trained End-to-End Pilot)** on Stage 1 across 5 seeds (5 runs) and Stage 2 across 5 seeds (5 runs) = **10 runs**.
* **Total Pilot Training Runs:** Exactly **25 parameter-training runs** ($5 \text{ (Cand 1 Stage 1)} + 5 \text{ (Ctrl 2 Stage 1)} + 5 \text{ (Cand 1 Stage 2)} + 5 \text{ (Cand 2 Stage 2)} + 5 \text{ (Ctrl 2 Stage 2)} = 25$), plus deterministic verification checks.
  - *Wall Time Estimate:* **25 runs** $\times$ ~45 sec = **~18.75 minutes wall time**.
* **Timing & Resource Check:** During the pilot run, exact wall-clock time per 100 updates will be measured on local hardware to calibrate resource projections before any composed training runs are launched.
* **Candidate-Advancement Policy (Elimination of Unprincipled Ranking):**
  Earlier planning drafts advanced the "top 2" pilot candidates into composed regimes without defining a reproducible ranking metric. That formulation is scientifically flawed:
  1. *Incommensurable Metric Modalities:* Candidate 2 features a deterministic algebraic ALU achieving $acc=1.000$ on Track 1B with zero training steps, whereas Candidate 1 and Control 2 train empirical MLUs requiring gradient descent to reach $acc_{val} \ge 0.950$ on Track 1A. Ranking them across tracks by raw accuracy conflates analytical verification of an isolated component with learned end-to-end rule induction.
  2. *Distinct Hypothesis Roles:* The candidates test distinct scientific hypotheses regarding memory partitioning and arithmetic inductive bias. Pruning an arbitrary candidate based on raw pilot metrics breaks the ability to isolate Hypothesis A from Hypothesis B.
  3. *Advancement Rule:* Candidate progression from Gate 1 to composed training is governed by **independent non-comparative threshold satisfaction**:
     - Candidate 1 advances iff Stage 1 Track 1A $acc_{val} \ge 0.950$ AND Stage 2 $EM_{read, val} \ge 0.950$.
     - Candidate 2 advances iff Stage 1 Track 1B exact $acc = 1.000$ AND Stage 2 $EM_{read, val} \ge 0.950$ (where its learned read/write/gate mechanics are first validated).
     - Candidate 3 (MSM) is evaluated forward-only across all stages as a parameter-free deterministic analytical reference.
     - Control 2 (Vanilla GRU) advances iff Stage 1 Track 1A $acc_{val} \ge 0.950$ AND Stage 2 $EM_{read, val} \ge 0.950$.
     Every candidate satisfying its respective gate advances into the composed regimes under the deterministic contingency ladder (§9).
* **Ratified Stage 1 Protocol (PI Decision 3):**
  - The Principal Investigator has formally ratified **Approach 2 (Structural Separation of Verification Tracks)**: Track 1A is retained for the trained Slot-MGU and Vanilla GRU pilot (10 runs, 7.5 min), and Track 1B for deterministic verification of Cyclic-MGU and MSM (0 training steps). (The alternative Approach 1 expanding the pilot by +5 trained Candidate 2 runs is not adopted).
* **Methodological Classification of Research Design (PI Decision 2):**
  Earlier drafts explored whether the study could be structured as a balanced $2 \times 2$ factorial experiment across Memory Architecture $\in \{\text{Monolithic, Explicit Slots}\}$ and Arithmetic Mechanism $\in \{\text{Learned MLP, Fixed Cyclic Group Prior}\}$.
  - *PI Decision 2 Ratification:* The Principal Investigator has formally ratified **Factorial Option 1**:
    1. The study is structured around **two trainable candidate architectures** (Slot-MGU and Cyclic-MGU), the **Vanilla GRU baseline**, and the **deterministic MSM reference**.
    2. Candidate 4 (Monolithic-MLP-MGU) is formally **REJECTED / NOT ADOPTED**, preventing compute expansion to 90 runs / ~8.13 hours.
    3. The study is rigorously classified as an **unbalanced architectural comparison**, NOT as a balanced $2 \times 2$ factorial experiment. Main-effect ANOVA claims are prohibited; scientific evaluation is conducted via targeted pairwise comparisons:
       - Candidate 1 vs. Control 2 evaluates the effect of explicit slot memory equipped with a generic learned ALU over standard recurrence.
       - Candidate 2 vs. Control 2 evaluates the effect of an embedded cyclic arithmetic prior in a monolithic state over standard recurrence.
       - Candidate 1 vs. Candidate 2 compares discrete slot partitioning with generic learning against continuous cyclic prior embedding.
       - Candidates 1 & 2 vs. Deterministic MSM evaluates how close trainable neural architectures approach the exact algorithmic rule standard.
* **Ratified Candidate-Advancement Policy (PI Decision 4):**
  - The Principal Investigator has formally approved **Advancement Option A**:
    - All candidates that satisfy their preregistered pilot feasibility requirements advance to composed training (Regime A and Regime B).
    - Candidates are NOT eliminated before their applicable gates have been evaluated. (The alternative Advancement Option B proposing advance pruning of Candidate 1 is formally rejected).
* **Deterministic Contingency Ladder (for Trainable Candidates):**
  To prevent ad-hoc fallback rules, pilot advancement for trainable candidates follows a strict deterministic contingency ladder:
  - *Case 1 (Both trainable candidates pass):* Both Candidate 1 and Candidate 2 advance to composed sequence training under both Regime A and Regime B alongside Control 2 Vanilla GRU (20 candidate runs + 10 control runs = 30 composed training runs). Total parameter-training runs across the entire campaign: **55 runs** (**258.75 minutes wall time**, ~4.31 hours).
  - *Case 2 (Exactly 1 trainable candidate passes):* The single passing candidate advances to composed training alongside Control 2 Vanilla GRU (10 candidate runs + 10 control runs = 20 composed training runs). Total campaign parameter-training runs: **45 runs** (**178.75 minutes wall time**, ~2.98 hours). The failing candidate is halted and classified as primitive-stage falsified.
  - *Case 3 (0 trainable candidates pass):* Zero candidates advance to composed training. Composed training is canceled. Control 2 Vanilla GRU completes pilot evaluation. The campaign terminates with an immediate negative falsification report documenting that both proposed inductive biases failed primitive feasibility gates (**25 parameter-training runs**, **~18.75 minutes**).
  In all cases, Candidate 3 (MSM) is evaluated forward-only on the prescribed benchmark datasets as a deterministic reference.
* **Formal Freeze and Preregistration Status:**
  With PI Decisions 1–6 recorded, the architectural specifications, research design, protocol tracks, advancement rules, and run budgets are formally frozen for preregistration review. Implementation and training remain unauthorized until formal sign-off of Review Gates 1–5.

---

## 7. Canonical Evaluation Thresholds and Baseline Grounding

### 7.1 Canonical Evaluation Thresholds Table

The following canonical table defines the evaluation metrics, baselines, and total decision partitions for each stage. Every metric interval across $[0, 1]$ is strictly partitioned without uncovered gaps:

| Stage & Benchmark | Primary & Joint Metrics | Historical Baseline Provenance & Comparability | Pilot Pass Threshold | Material Improvement ($\Delta_{mat}$) | Final Success Bar | Below Final Bar / Underperforming (Phase 2 Unauthorized) | Inconclusive / Ambiguous Region (Phase 2 Unauthorized) | Rejection / Falsification Threshold | Seed Consistency Rule |
|---|---|---|---:|---:|---:|---|---|---:|---|
| **Stage 1: Primitive ADD** | Track 1A: Final Held-Out Balanced Acc ($acc_{heldout}$) on 20 test pairs ($b \in \{0,5\}$); Pilot Validation Acc ($acc_{val}$) on 16 `VAL-SEEN` pairs (Cand 1 & Ctrl 2). Track 1B: Exact Numerical Unit Test Acc ($acc = 1.000$) on `VAL-SEEN` (Track 1B-C2 for Cand 2 isolated rotation ALU; Track 1B-C3 for Cand 3 deterministic primary value-coordinate slot pipeline with unwritten slot value-coordinate preservation $\|\Delta z_r\| = 0.0$; unit tests only, learned pathways unexercised) | `N/A — no comparable historical baseline` (Chance: 0.100 for learned heads; Model E was never trained on isolated scalar pairs) | Track 1A: $acc_{val} \ge 0.950$ on `VAL-SEEN`; Track 1B (C2 & C3): exact $acc = 1.000$ on `VAL-SEEN` | $\ge 0.500$ (+0.400 over chance on TEST for Track 1A) | Track 1A: **$acc_{heldout} \ge 0.950$**; Track 1B (C2 & C3): **$acc = 1.000$** (exact by construction) | $0.800 \le \text{acc} < 0.950$ (Track 1A) | $0.250 \le \text{acc} < 0.800$ (Track 1A) | $< 0.250$ (Track 1A); $< 1.000$ (Track 1B) | $\ge 4/5$ seeds independently satisfy threshold (Track 1A); zero seeds $< 0.250$ |
| **Stage 2: Address Binding** | Behavioral Read EM ($EM_{read}$) on 1,200 TEST sequences (Final); Validation Read EM ($EM_{read, val}$) on 1,200 VAL sequences (Pilot) | `N/A — no comparable historical behavioral baseline` (Bayes Ceiling: 1.000 from Exp U; Model E was not benchmarked on pure SET/SWAP) | $EM_{read, val} \ge 0.950$ on `VAL` (TEST strictly unqueried) | N/A (absolute target governs) | **$EM_{read} \ge 0.990$** | $0.950 \le EM < 0.990$ | $0.850 \le EM < 0.950$ | $< 0.850$ (on VAL for pilot; on TEST for final) | $\ge 4/5$ seeds independently satisfy threshold; zero seeds $< 0.850$ |
| **Stage 3: Composed Short (T=10, Regime A In-Distribution)** | Sequence EM ($EM_{T=10}$): Fraction of $T=10$ test sequences with 100% correct transition outputs | **Model E: 0.755 ± 0.019** (Exp W Table 1 / Exp P Table 2; valid direct comparison under Regime A) | N/A (post-pilot) | Paired $\Delta_{EM} \ge +0.100$ ($\implies EM \ge 0.855$) | **$\ge 0.900$** | $0.855 \le EM < 0.900$ (Material gain met, but misses Final Bar) | $0.755 < EM < 0.855$ (Exceeds baseline, lacks material gain) | $EM \le 0.755$ (Fails baseline) | Paired $\Delta_s > 0$ in $\ge 4/5$ seeds, mean paired $\Delta \ge +0.100$; zero seeds $\le 0.755$ |
| **Stage 4: Systematic Transfer (Regime B Masked Training)** | **Joint Gate:** (1) Held-Out Transition Balanced Acc ($acc_{heldout}$), AND (2) Held-Out Sequence EM ($EM_{heldout}$) | `N/A — no comparable historical Model E baseline` (External GRU: 0.116 ± 0.020 in Exp W Table 2; Chance: 0.100; zero training leakage under Regime B) | N/A (post-pilot) | $acc_{heldout} \ge 0.500$ AND $EM_{heldout} \ge 0.500$ | **$acc \ge 0.800$ AND $EM \ge 0.700$** | $0.500 \le acc < 0.800$ OR $0.500 \le EM < 0.700$ | $0.250 \le acc < 0.500$ OR $0.150 < EM < 0.500$ | $acc < 0.250$ OR $EM \le 0.150$ | $\ge 4/5$ seeds satisfy both joint thresholds; zero seeds in rejection |
| **Stage 5: Extrapolation (T=20)** | Sequence EM ($EM_{T=20}$): Zero-shot $T=20$ sequence EM ($T=30, 50$ diagnostic) | **Model E: 0.407 ± 0.023** (Exp W Table 1 / Exp P Table 2; valid direct comparison) | N/A (post-pilot) | Paired $\Delta_{EM} \ge +0.200$ ($\implies EM \ge 0.607$) | **$\ge 0.700$** | $0.607 \le EM < 0.700$ (Material gain met, but misses Final Bar) | $0.407 < EM < 0.607$ (Exceeds baseline, lacks material gain) | $EM \le 0.407$ (Fails baseline) | Paired $\Delta_s > 0$ in $\ge 4/5$ seeds, mean paired $\Delta \ge +0.200$; zero seeds $\le 0.407$ |
| **Stage 6: Persistence (K=20)** | Behavioral Read EM ($EM_{read}$): Decoding accuracy of $r_0$ at step $K+1$ after $K=20$ active writes | `N/A — no comparable historical behavioral baseline` (Diagnostic linear probe: ~0.780 in Exp M/T; non-behavioral) | N/A (post-pilot) | N/A (absolute target governs) | **$\ge 0.950$** | $0.900 \le EM < 0.950$ | $0.750 \le EM < 0.900$ | $< 0.750$ | $\ge 4/5$ seeds independently satisfy threshold; zero seeds $< 0.750$ |

---

### 7.2 Historical Baseline Provenance and Comparability Audit

To preserve empirical integrity, every historical baseline cited above is audited against its source report:

1. **Stage 1 (Primitive Modular Arithmetic):**
   * *Historical Baseline:* `N/A — no comparable historical baseline`.
   * *Provenance & Track Distinction:* Model E was never evaluated on isolated scalar arithmetic in Phase 1 (Model E is a sequence model trained on multi-step traces). Theoretical chance accuracy on 10 balanced classes is 0.100 for learned classification heads. For Track 1A (Slot-MGU and Vanilla GRU), the absolute success threshold ($acc_{heldout} \ge 0.950$) governs evaluation across the trained 3-step trace. For Track 1B, verification is divided into **Track 1B-C2** (Candidate 2 isolated cyclic ALU unit test on 16-pair `VAL-SEEN`, verifying the 20-dim input to 10-dim output rotation ALU) and **Track 1B-C3** (Candidate 3 deterministic primary value-coordinate slot-and-arithmetic pipeline unit test with per-step value-coordinate preservation checks $\|\Delta z_r\| = 0.0$ on non-destination slots $r \in \{1, 2, 3\}$), with exact $acc = 1.000$ by construction. Passing Track 1B-C2 verifies the algebraic rotation prior only, completely bypassing Candidate 2's learned read, write, and gating networks ($\text{MLP}_{read}, \text{MLP}_{write}, \text{MLP}_{gate}$); Candidate 2's learned state-machine mechanics face empirical evaluation for the first time in Stage 2 ($EM_{read, val} \ge 0.950$). Passing Track 1B-C3 verifies strictly the deterministic parameter-free slot routing and cyclic rotation pipeline of the MSM reference (Option B, 40-dimensional state, 0 trainable parameters), with exact $acc = 1.000$ and $\|\Delta z_r\| = 0.0$ on unwritten slots. Cross-track capability comparisons are strictly prohibited.
2. **Stage 2 (Address-Value Binding):**
   * *Historical Baseline:* `N/A — no comparable historical behavioral baseline`.
   * *Provenance:* In Experiment U (`docs/reports/experiment_u_task_interface.md`), the causal observation stream was shown to contain theoretical Bayes ceiling information of $\ge 0.998$ (1.000 at full prefix). However, Model E was never benchmarked on pure SET/SWAP/READ sequences without arithmetic. The 0.999 value in earlier drafts reflected theoretical information ceiling, not Model E output. Absolute target threshold ($\ge 0.990$) governs evaluation.
3. **Stage 3 (Composed Short Register Execution, T=10):**
   * *Historical Baseline:* **0.755 ± 0.019 Sequence EM**.
   * *Provenance:* Experiment W Table 1 (`docs/reports/experiment_w_scaling_baseline.md`) and Experiment P Table 2 (`docs/reports/experiment_p_sequence_error_budget.md`), evaluated on checkpoint `experiments/checkpoints/E_candidate_seed{42..46}_3000steps.pt`.
   * *Direct Comparability:* **Directly valid under Regime A (In-Distribution Composed Training)**. Evaluated on identical canonical test sequences ($N=1,200 \times 10$) following training on the unmasked 6,000 sequence pool matching Model E's training support. Both absolute bar ($EM \ge 0.900$) and paired material increment ($\Delta \ge +0.100$) are preregistered. (If evaluated under the alternative single-regime Option B, Model E represents an asymmetric upper-bound reference due to training support differences).
4. **Stage 4 (Systematic Generalization on Held-Out Operands):**
   * *Historical Baseline:* `N/A — no comparable historical Model E baseline`.
   * *Provenance:* The figure $0.116 \pm 0.020$ reported in Experiment W Table 2 measured an **external vanilla GRU sequence reader** (105k parameters) trained on the sequence stream, NOT Model E. Model E was trained on the standard training pool without held-out operand masking in Phase 1.
   * *Direct Comparability & Leakage Elimination:* Evaluated strictly under **Regime B (Held-Out Masked Composed Training)**, where runtime transitions with $b \in \{0, 5\}$ are strictly withheld from training sequences. This guarantees zero training data leakage. Absolute target thresholds ($acc_{heldout} \ge 0.800$ AND $EM_{heldout} \ge 0.700$) govern evaluation.
5. **Stage 5 (Length Extrapolation, T=20):**
   * *Historical Baseline:* **0.407 ± 0.023 Sequence EM**.
   * *Provenance:* Experiment W Table 1 and Experiment P Table 2, evaluated on checkpoint `experiments/checkpoints/E_candidate_seed{42..46}_3000steps.pt`.
   * *Direct Comparability:* **Directly valid**. Evaluated zero-shot on identical $T=20$ sequences ($N=1,200$). Both absolute bar ($EM \ge 0.700$) and paired material increment ($\Delta \ge +0.200$) are preregistered. ($T=30$ and $T=50$ are secondary diagnostic probes).
6. **Stage 6 (Long-Horizon Persistence, K=20):**
   * *Historical Baseline:* `N/A — no comparable historical behavioral baseline`.
   * *Provenance:* The ~0.780 figure cited in earlier drafts was a *diagnostic linear probe accuracy* decoding frozen hidden state representations in Experiment M and T (`docs/reports/experiment_m_representation_geometry.md`, `docs/reports/experiment_t_upstream_representation.md`). It measured latent linear separability via auxiliary supervised logistic regression, NOT behavioral readout from the native output head under active interference. Model E was never tested on a dedicated active-interference persistence protocol. Absolute target threshold ($EM_{read} \ge 0.950$) governs evaluation.

---

## 8. Falsification Criteria and Decision Gates

```
                                  PHASE 1.6 DECISION GATES
                                             │
      [Gate 1: Pilot Stage] ─────────────────┴──────────────────► Primitive pass on validation?
       ├── PASS: Proceed to composed training                     (Track 1A VAL >= 0.950 / Track 1B pass, Stage 2 VAL >= 0.950)
       └── FAIL: Reject inductive bias; halt candidate            (Track 1A VAL < 0.250 / Track 1B fail, Stage 2 VAL < 0.850)
                                             │
      [Gate 2: Systematic Transfer] ─────────┴──────────────────► Held-Out rule transfer?
       ├── PASS: Proceed to extrapolation                         (Stage 4 acc >= 0.800 AND EM >= 0.700)
       └── FAIL: Declare fit without transfer; reject             (Stage 4 acc < 0.250 or EM <= 0.150)
                                             │
      [Gate 3: Conjunction Verification] ────┴──────────────────► Full algorithmic core?
       ├── PASS: AUTHORIZE Phase 2 Roadmap                        (All 6 stages satisfy Final Success Bar)
       └── FAIL: Maintain implementation hold; archive
```

### 8.1 Total Decision Rules and Conjunction Formulation

#### Mutually Exclusive Outcome Classifications
For each stage, every possible empirical outcome is strictly partitioned into one of five mutually exclusive categories:
1. **Final Success:** The candidate satisfies the preregistered final success bar and all five-seed consistency criteria.
2. **Underperforming / Below Final Bar:** The candidate achieves material improvement or surpasses baseline, but fails to reach the final success bar (e.g. Stage 3 $EM_{T=10} \in [0.855, 0.900)$). **Phase 2 remains unauthorized.**
3. **Inconclusive / Ambiguous:** The candidate falls within the ambiguity region where neither success nor falsification can be declared (e.g. Stage 4 $acc_{heldout} \in [0.250, 0.500)$). **Phase 2 remains unauthorized.**
4. **Falsification / Rejection:** The candidate falls into the prespecified rejection region in $\ge 4/5$ seeds (e.g. Stage 1 $< 0.250$ on VAL for pilot / TEST for final, Stage 2 $< 0.850$, Stage 3 $\le 0.755$, Stage 4 $acc < 0.250$ or $EM \le 0.150$). The core inductive bias hypothesis is formally declared **FALSIFIED**.
5. **Statistical Seed Inconsistency:** The 5-seed mean meets a numerical threshold, but $< 4/5$ seeds satisfy it, or any single seed falls into the rejection region. The result is classified as unstable. **Phase 2 remains unauthorized.**

#### Five-Seed Consistency Rules
1. **Absolute Target Thresholds:** $\ge 4/5$ independent seeds must meet the absolute threshold, and zero seeds may fall into the rejection region.
2. **Paired Increment Thresholds (Stages 3 & 5):** The candidate must achieve paired improvement $\Delta_s = \text{Candidate}_s - \text{ModelE}_s > 0$ in $\ge 4/5$ seeds AND the 5-seed mean paired difference must satisfy $\bar{\Delta} \ge \Delta_{mat}$.
3. **Rejection Handling:** If $\ge 4/5$ seeds enter the rejection region, the architecture is rejected. If 1 to 3 seeds enter the rejection region, the candidate is classified as seed-inconsistent / failed stability.

#### Formal Conjunction Rule for Success (GO to Phase 2)
Authorization to initiate Phase 2 (branching, verifiers, recursion) is granted **ONLY IF** a trainable candidate architecture (Candidate 1 Slot-MGU or Candidate 2 Cyclic-MGU) satisfies the **complete conjunction across all six stages** (with Candidate 3 MSM serving as the deterministic analytical reference across all stages):

$$\text{Success} = (\text{Stage 1: } acc \ge 0.950 \text{ [Track 1A: } acc_{heldout} \ge 0.950 \text{; Track 1B: exact } 1.000]) \land (\text{Stage 2} \ge 0.990) \land (\text{Stage 3} \ge 0.900) \land (\text{Stage 4: } acc_{heldout} \ge 0.800 \land EM_{heldout} \ge 0.700) \land (\text{Stage 5: } EM_{T=20} \ge 0.700) \land (\text{Stage 6} \ge 0.950)$$

*Scope Boundary:* Passing the Stage 1 Track 1B algebraic unit test alone establishes only component correctness of the cyclic prior; authorization to proceed to Phase 2 requires demonstrated learned capability across all empirical stages (Stages 2–6). If any single stage in the conjunction misses its Final Success Bar, **Phase 2 transition is strictly NO-GO**.

---

## 9. Resource and Compute Budget

Phase 1.6 is designed for strict computational frugality and rapid execution on existing repository infrastructure:

* **Hardware Target:** Single local workstation CPU / AMD RX 6600M GPU (ROCm Linux environment with native PyTorch primitives; automated CPU fallback).
* **Trainable Parameter Envelope Compliance:** Strictly bounded to $226,358 \pm 10\%$ parameters (envelope: 203,722 to 248,994 parameters) for all trainable neural models:
  - *Historical Reference (Model E):* 226,358 parameters (reference baseline).
  - *Candidate 1 (Slot-MGU):* 223,414 parameters ($-1.30\%$ vs. reference).
  - *Candidate 2 (Cyclic-MGU):* 219,018 parameters ($-3.24\%$ vs. reference).
  - *Candidate 3 (MSM — Deterministic Reference):* **0 parameters** (Option B ratified by PI Decision 1; parameter-free exact-rule reference; exempt from neural parameter envelope).
  - *Control 2 (Vanilla GRU Baseline):* **220,874 parameters** ($-2.42\%$ vs. reference) for dedicated 10-class interface ($22 \to 64 \to 256 \to H=160 \to 10$) as ratified by PI Decision 5 ($18,624$ encoder + $200,640$ GRU + $1,610$ head); general composed sequence interface is 225,704 parameters ($-0.29\%$).
  - *Candidate 4 (Monolithic-MLP-MGU):* Rejected / Not Adopted per PI Decision 2.

| Model Identifier | Architectural Mechanism | Key Trainable Layers & Dimensions | Trainable Parameters | $\Delta$ vs Model E | Envelope Compliance Status |
|---|---|---|---:|---:|---|
| **Model E (Reference)** | Monolithic Non-linear MGU | Hidden size $H=212$, candidate/gate MLPs | 226,358 | 0.0% | Reference ($226,358$) |
| **Candidate 1 (Slot-MGU)** | 4 Discrete Slots + Learned MLP ALU | $\text{MLP}_{cand}$ ($128 \to 612 \to 53$), $\text{MLP}_{gate}$ ($128 \to 612 \to 53$), Head ($53 \to 10$) | 223,414 | -1.30% | Compliant ($203,722 \le N \le 248,994$) |
| **Candidate 2 (Cyclic-MGU)** | Monolithic State + Cyclic ALU | $\text{MLP}_{read}$ ($216 \to 128 \to 10$), $\text{MLP}_{write}$ ($32 \to 212 \to 212$), $\text{MLP}_{gate}$ ($436 \to 212 \to 212$) | 219,018 | -3.24% | Compliant ($203,722 \le N \le 248,994$) |
| **Candidate 3 (MSM Reference)** | 4 Slots + Cyclic ALU (Option B) | Parameter-Free Discrete Slot Routing + 2D Cyclic Rotations ($4 \times 10 = 40$-dim state) | 0 | -100.0% | Exempt (Deterministic Reference; 0 params) |
| **Control 2 (Vanilla GRU)** | Monolithic Standard GRU | Stage 1 Dedicated: Enc ($22 \to 64 \to 256$), GRU ($256 \to H=160$), Head ($160 \to 10$); General: Head ($160 \to 40$) | 220,874 (Stage 1 Dedicated, Ratified) / 225,704 (General) | -2.42% / -0.29% | Compliant ($203,722 \le N \le 248,994$) (Historical 21-in: 225,640) |
| *Candidate 4 (Monolithic-MLP-MGU)* | *Monolithic State + Learned MLP ALU* | *Rejected / Not Adopted per PI Decision 2* | *N/A (221,138)* | *-2.31%* | *Not Adopted (Factorial Option 1 Ratified)* |

* **Comprehensive Compute Allocation and Execution Accounting (Ratified Design):**
  Per PI Decisions 1–6, all training runs, evaluation passes, and deterministic checks are explicitly reconciled across two trainable candidate architectures (Slot-MGU, Cyclic-MGU), the dedicated Vanilla GRU baseline, and the forward-only MSM deterministic reference:

  * **Parameter-Training Runs (55 total runs):**
    - *Pilot Stage (Stages 1 & 2):*
      - Candidate 1 (Slot-MGU): 5 runs (Stage 1 Track 1A) + 5 runs (Stage 2) = 10 runs.
      - Candidate 2 (Cyclic-MGU): 0 runs (Stage 1 Track 1B-C2 deterministic check) + 5 runs (Stage 2) = 5 runs.
      - Candidate 3 (MSM): 0 runs (Stage 1 Track 1B-C3 deterministic check) + 0 runs (Stage 2 deterministic check) = 0 parameter-training runs.
      - Mandatory Control 2 (Vanilla GRU): 5 runs (Stage 1 Track 1A) + 5 runs (Stage 2) = 10 runs.
      - *Subtotal Pilot Training Runs:* **25 runs** $\times$ ~45 sec = **~18.75 minutes wall time**.
    - *Composed Sequence Stage — Regime A (In-Distribution, Stages 3, 5, 6):*
      - Passing trainable candidates: 2 candidates $\times$ 5 seeds = 10 runs $\times$ ~8 min = ~80 minutes wall time.
      - Mandatory Control 2 (Vanilla GRU Baseline): 1 control $\times$ 5 seeds = 5 runs $\times$ ~8 min = ~40 minutes wall time.
      - Candidate 3 (MSM): 0 training runs (deterministic reference; evaluated forward-only).
      - *Subtotal Regime A Composed Training Runs:* **15 runs** $\times$ ~8 min = **~120.00 minutes wall time** (2.00 hours).
    - *Composed Sequence Stage — Regime B (Held-Out Masked, Stage 4):*
      - Passing trainable candidates: 2 candidates $\times$ 5 seeds = 10 runs $\times$ ~8 min = ~80 minutes wall time.
      - Mandatory Control 2 (Vanilla GRU Baseline): 1 control $\times$ 5 seeds = 5 runs $\times$ ~8 min = ~40 minutes wall time.
      - Candidate 3 (MSM): 0 training runs (deterministic reference; evaluated forward-only).
      - *Subtotal Regime B Composed Training Runs:* **15 runs** $\times$ ~8 min = **~120.00 minutes wall time** (2.00 hours).
    - *Total Parameter-Training Runs:* Exactly **55 training runs** ($25 \text{ pilot} + 15 \text{ Regime A} + 15 \text{ Regime B} = 55$).
    - *Total Parameter-Training Wall Time:* $18.75\text{ min} + 120.00\text{ min} + 120.00\text{ min} = \mathbf{258.75\text{ minutes}}$ (~4.31 hours).

  * **Forward-Only Evaluations and Deterministic Reference Passes:**
    Per PI Decision 6, the deterministic reference is evaluated on the prescribed evaluation datasets and conditions without treating it as a seed-trained model:
    - *Unique Evaluation Suites Enumeration:*
      1. Stage 1: Primitive Modular Arithmetic (`VAL-SEEN` 16 pairs, `TEST-HELDOUT` 20 pairs).
      2. Stage 2: Address-Value Binding (1,200 validation and 1,200 test sequences).
      3. Stage 3: Composed Short Register Execution ($T=10$, 1,200 sequences $\times$ 10 steps).
      4. Stage 4: Systematic Generalization on Held-Out Operands (1,200 masked sequences).
      5. Stage 5: Length Extrapolation ($T=20$ primary, $T=30, 50$ diagnostic).
      6. Stage 6: Long-Horizon Persistence under Active Interference ($K=20$).
    - *Forward-Only Pass Accounting:*
      - Seed-Trained Evaluation Passes: 35 passes (15 Regime A passes: 2 candidates $\times$ 5 seeds + 1 control $\times$ 5 seeds; 15 Regime B passes: 2 candidates $\times$ 5 seeds + 1 control $\times$ 5 seeds; 5 Control 1 Frozen Model E passes across stages).
      - Deterministic Reference (MSM Option B): Evaluated forward-only across the 6 unique evaluation suites (parameter-free, seed-invariant; 0 training steps).
      - Control 3 (Deterministic Symbolic Oracle): Programmatic verification of all dataset partitions and splits (0 sec).
      - *Provisional Evaluation Allowance:* Per PI Decision 6, the existing **35.00-minute evaluation allowance is retained provisionally** until runtime accounting is verified.

  * **Total Campaign Wall-Clock Time:**
    $258.75\text{ min (parameter training)} + 35.00\text{ min (provisional evaluation allowance)} = \mathbf{293.75\text{ minutes wall time}}$ (**~4.90 hours**).
    The primary campaign compute budget strictly satisfies the preregistered envelope of **$< 5.0$ hours wall time** on the target local workstation CPU / AMD RX 6600M GPU.

  * **Storage Estimate:** 55 checkpoints (~55 MB) + evaluation NPZ arrays and logs (~120 MB) + figures/JSONs $\approx < 250$ MB (strictly within $< 500\text{ MB}$ limit).

  * **Supersession of Alternative Design Options:**
    - *Candidate Advancement Policy (PI Decision 4):* Approved **Advancement Option A** (advance all passing candidates meeting pilot gates). The compute-constrained alternative proposing advance pruning of Candidate 1 (the earlier Option B compute alternative, 50 runs) was formally **REJECTED**.
    - *Research Design (PI Decision 2):* Ratified **Factorial Option 1** (unbalanced comparison of two trainable candidates, Vanilla GRU baseline, and MSM reference). Candidate 4 (Monolithic-MLP-MGU; 90 runs, ~8.13 hrs under Factorial Option 2) was formally **REJECTED / NOT ADOPTED**.
    - *MSM Architectural Reformulation (PI Decision 1):* Adopted **Option B for Candidate 3**, eliminating the disconnected 43-dim auxiliary branch and classifying MSM as a 0-parameter deterministic reference, reducing parameter-training runs from 70 to 55 and training time by 83.75 minutes.

* **Deterministic Contingency Rules for Compute Execution (for Trainable Candidates):**
  - *Case 1 (Both trainable candidates pass pilot):* Both Candidate 1 and Candidate 2 advance to composed sequence training under Regimes A and B alongside Control 2 Vanilla GRU (20 candidate runs + 10 control runs = 30 composed training runs). Total campaign parameter-training runs: **55 runs** (**258.75 minutes wall time**; total campaign wall time: **293.75 minutes**, ~4.90 hours).
  - *Case 2 (Exactly 1 trainable candidate passes pilot):* The single passing candidate advances to composed training alongside Control 2 Vanilla GRU (10 candidate runs + 10 control runs = 20 composed training runs). Total campaign parameter-training runs: **45 runs** (**178.75 minutes wall time**; total campaign wall time: ~210 minutes, ~3.5 hours). The failing candidate is halted and classified as primitive-stage falsified.
  - *Case 3 (0 trainable candidates pass pilot):* Zero candidates advance to composed training. Composed training is canceled. Control 2 Vanilla GRU completes pilot evaluation. The campaign terminates with an immediate negative falsification report documenting that both proposed inductive biases failed primitive feasibility gates (**25 parameter-training runs**, **~18.75 minutes**).
  In all cases, Candidate 3 (MSM) is evaluated forward-only on the prescribed benchmark datasets as a deterministic reference.
* **Storage Budget:** All checkpoints, evaluation NPZ arrays, metadata JSONs, and summary figures shall consume $< 500\text{ MB}$ disk space.

---

## 10. Risks, Limitations, and Alternative Explanations

1. **Differentiable vs. Hard Slot Addressing:** Hard discrete addressing is non-differentiable if address indices are learned; however, in RE:Reason's Register Tracking task, register addresses are provided explicitly as one-hot fields in the instruction token $x_t[3:7]$. Hard multiplexing is fully differentiable with respect to slot contents.
2. **Phase Dispersion in Cyclic Representations:** In recurrent chains of length $T=50$, repeated phase additions could accumulate numerical roundoff error, leading to phase dispersion on the circle $S^1$. *Mitigation:* Explicit normalization to unit norm ($\tilde{z}_k = z_k / \|z_k\|_2$) after each step bounds numerical drift.
3. **The "Why Not a Transformer?" Alternative:** A standard Transformer with RoPE or causal multi-head self-attention might solve this benchmark with sufficient data. *Scientific Boundary:* RE:Reason is explicitly an investigation into **compact, recurrent, constant-memory inference cores** ($< 250\text{k}$ parameters) that maintain bounded state without quadratic KV-cache growth. Proving that an attention cache solves the task does not answer whether recurrent state machines can do so.

---

## 11. Literature Assessment and Scoped Contribution Claims

### 11.1 Grounding in Primary Literature
* **Grokking and Modular Arithmetic (Power et al. 2022; Gromov 2023; Swaroop 2026):**
  * *Established Fact:* Small MLPs and transformers trained with weight decay exhibit delayed generalization (grokking) on modular arithmetic by discovering discrete Fourier feature representations (Gromov, 2023).
  * *Gap Unresolved by Prior Art:* Grokking studies examine static $(a, b) \to (a+b) \pmod{p}$ tasks presented simultaneously in an input context. They do not examine sequential register machines where operands must be dynamically retrieved from memory, updated in-place, and persisted across intervening operations.
* **Tensor-Product Variable Binding (Smolensky 1990; Graves et al. 2014; Goyal et al. 2021):**
  * *Established Fact:* Distributed connectionist models suffer from catastrophic binding interference unless address vectors and value vectors are bound using tensor products or explicit slot structures.
  * *Project Adaptation:* Slot-MGU adopts Smolensky's role-filler separation by dedicating orthogonal subspaces to register slots, tested under recurrent convex updates.
* **Recurrent Algorithmic Reasoning (Veličković & Blundell 2021; Schwarzschild et al. 2021):**
  * *Established Fact:* Neural networks can learn algorithmic execution, but standard architectures fail to extrapolate across problem difficulty and sequence length without explicit inductive biases.

### 11.2 Carefully Scoped Contribution Claims
* **What Phase 1.6 WILL Claim if Successful:**
  * Evaluating whether explicit slot addressing (Candidate 1: Slot-MGU) or cyclic group arithmetic inductive bias (Candidate 2: Cyclic-MGU) individually achieve systematic generalisation on held-out modular arithmetic, length extrapolation, and multi-step persistence within a compact ($< 250\text{k}$ param) trainable recurrent core, compared against the Vanilla GRU recurrent baseline and the parameter-free deterministic MSM exact-rule reference.
* **What Phase 1.6 WILL NOT Claim:**
  * We will NOT claim to have tested, trained, or validated a neural architecture combining explicit slot addressing and cyclic arithmetic units into a single trainable recurrent model; Candidate 4 was rejected, MSM is a deterministic exact-rule reference with zero trainable parameters, and the study makes no claims for that untested trainable composition.
  * We will NOT claim to have solved general AGI reasoning, human-like abstraction, arbitrary mathematical theorem proving, or unbounded Turing computation.
  * We will NOT claim novelty for slot memory (known from NTM/Slot Attention) or cyclic embeddings (known from Fourier features/RoPE) in isolation, nor claim that this study constitutes a balanced $2 \times 2$ factorial evaluation.

---

## 12. Explicit Review Requirements Before Implementation

In accordance with `AGENTS.md` and `docs/DOCUMENTATION_POLICY.md`, **no model code may be implemented, no model trained, and no repository files modified** until the following five review gates are formally signed off:

1. **Review Gate 1 (Architectural Shortlist Sign-Off):** Peer review and alignment on the exact mathematical specifications of Candidates 1, 2, and 3.
2. **Review Gate 2 (Benchmark Generator Audit):** Independent audit of the data generation scripts to guarantee zero label leakage, strict sequence disjointness, and exact held-out operand masking.
3. **Review Gate 3 (Pre-Registration Registry):** Registration of the final proposal, exact parameter budgets, and evaluation metrics in `docs/reports/HISTORICAL_BASELINE.json`.
4. **Review Gate 4 (Governance Check):** Verification that all documentation governance tests pass.
5. **Review Gate 5 (Formal Authorization):** Explicit instruction from the user/PI to transition from proposal status to active implementation.

---

## 13. Final Recommendation and Implementation Go/No-Go Checklist

### Final Recommendation

**RECOMMENDATION: PROPOSAL UNDER REVIEW — MAINTAIN IMPLEMENTATION HOLD.**

The proposal provides a mathematically grounded, empirically motivated, and computationally frugal roadmap for Phase 1.6. It directly addresses the dual bottlenecks identified in Phase 1 (address-value binding and cyclic modular arithmetic) without relaxing scientific rigor or violating project governance.

### Implementation Go/No-Go Checklist

```
[X] 1. Phase 1 empirical synthesis formally closed (Decision A: GO, Decision B: NO-GO).
[X] 2. Dual bottlenecks (binding vs arithmetic) explicitly separated.
[X] 3. Competing hypotheses (A, B, C) rigorously evaluated; Hypothesis C deferred to preserve scientific isolation.
[X] 4. Two trainable candidate architectures (Slot-MGU: 223,414; Cyclic-MGU: 219,018), dedicated 10-class Vanilla GRU baseline (220,874 parameters, PI Decision 5 ratified), and deterministic MSM reference (Option B, 40-dim state, 0 parameters, exempt from neural parameter envelope, PI Decision 1 ratified; Candidate 4 rejected, PI Decision 2) fully specified mathematically: Candidate 1 unified input contract with categorical one-hot immediate value e(v) (no harmonic prior leakage); Candidate 2 atomic SWAP with pre-update read snapshots and convex gating; Candidate 3 deterministic slot routing with exact cyclic rotation and unit-circle projection, non-mutating READ, and auxiliary branch completely eliminated.
[X] 5. Progressive 6-stage evaluation ladder defined with deterministic non-overlapping splits (64/16/20) covering all 10 target classes in validation with exact frequency audit.
[X] 6. Single unified canonical threshold table reconciled across all sections with total decision partitions.
[X] 7. Pilot stage (< 20 min compute, 25 parameter-training runs across trainable candidates and GRU control, 18.75 min wall time) designed to cheaply prune flawed inductive biases with validation-only gating, deterministic contingency ladder (for 0–2 passing trainable candidates), and timing check; Stage 1 protocol reconciled into Track 1A (trained end-to-end pilot for Candidate 1 & GRU control) vs Track 1B (deterministic architectural unit test: Track 1B-C2 isolated rotation ALU for Candidate 2 on 16-pair `VAL-SEEN`, Track 1B-C3 deterministic primary value-coordinate slot pipeline for Candidate 3 with per-step value-coordinate preservation invariants $\|\Delta z_r\| = 0.0$ on non-destination slots $r \in \{1, 2, 3\}$, with explicit non-claims for unexercised learned pathways) under Approach 2 (ratified by PI Decision 3); research design ratified under Factorial Option 1 (two trainable candidates, Vanilla GRU baseline, MSM reference; Candidate 4 rejected; PI Decision 2); candidate advancement ratified under Advancement Option A (advance all passing candidates; PI Decision 4).
[X] 8. Conjunction rule for success formally pre-registered with mandatory joint Stage 4 gates and T=20 primary extrapolation.
[X] 9. Strict compute budget reconciled: primary campaign budgeted at exactly 55 parameter-training runs (258.75 min) + provisional 35.00 min evaluation allowance = 293.75 minutes total wall time (~4.90 hours, < 5.0 hrs envelope) across 2 trainable candidates and Vanilla GRU baseline, with Candidate 3 MSM evaluated forward-only across 6 stages; parameter envelope (226,358 +/- 10%) verified across all neural models, with MSM classified as an exempt parameter-free reference.
[ ] 10. Formal peer review and user sign-off on proposal document.
[ ] 11. User authorization to begin Phase 1.6 implementation.

CURRENT STATUS: GATES 1-9 COMPLETE. GATES 10-11 PENDING.
IMPLEMENTATION IS NOT AUTHORIZED.
```

---

## Amendment — 2026-10-13 (Pilot Interface Validity & Parameter Arithmetic Audit)

* **Stage 1 Interface Validity:** Clarified that Stage 1 (Primitive Modular Arithmetic) is an end-to-end evaluation across the candidate's actual register memory, gating, and update pathways via canonical 3-step execution traces (`SET r_0 a`, `SET r_1 b`, `ADD r_0 r_1` evaluated at $t=3$ on destination slot 0), strictly avoiding ad-hoc scalar proxy encoders while preserving the `TEST-HELDOUT` ($b \in \{0, 5\}$) evaluation firewall.
* **Candidate 4 Parameter Recomputation:** Recomputed trainable parameter count for optional Candidate 4 (Monolithic-MLP-MGU) directly from layer dimensions ($\text{MLP}_{read}$: 29,066; $\text{MLP}_{cand}$: 54,272; $\text{MLP}_{gate}$: 137,800), resolving arithmetic from 221,424 to exactly **221,138** parameters ($-2.31\%$ vs Model E). Documented the 42-dimensional candidate input contract and three preregistration decisions requiring PI approval.
* **Governance Status:** Proposal updated; SHA-256 recalculated and registered; implementation, training, and benchmark execution remain strictly unauthorized.

---

## Amendment — 2026-10-13 (Stage 1 Protocol Resolution: Track 1A vs. Track 1B Separation)

* **Root Cause of Protocol Inconsistency:** Audited Stage 1 (Primitive Modular Arithmetic) and resolved an experimental asymmetry: Candidate 1 (Slot-MGU) and Control 2 (Vanilla GRU) were evaluated via 5 trained runs on a 3-step execution trace (`SET r_0 a`, `SET r_1 b`, `ADD r_0 r_1`), whereas Candidates 2 (Cyclic-MGU) and 3 (MSM) were evaluated via 0 training steps checking the algebraic rotation identity $R(\theta_b) \Phi(a) = \Phi((a + b) \pmod{10})$. In Candidate 2, this algebraic check bypassed its learned read queries ($\text{MLP}_{read}: 216 \to 128 \to 10$), monolithic write candidate generator ($\text{MLP}_{write}: 32 \to 212 \to 212$), and convex gate ($\text{MLP}_{gate}: 436 \to 212 \to 212$), creating a false equivalence between component unit testing and end-to-end capability evaluation.
* **Resolution Adopted (Approach 2 — Structural Separation of Verification Tracks):**
  - Formally partitioned Stage 1 into **Track 1A (Trained End-to-End Pilot)** and **Track 1B (Deterministic Component Pre-Flight Unit Test)**.
  - *Track 1A:* Evaluates Candidate 1 and Control 2 on the canonical 3-step trace across 5 training seeds with validation gating ($acc_{val} \ge 0.950$ on `VAL-SEEN`) and pilot parameter discarding.
  - *Track 1B:* Evaluates Candidates 2 and 3 strictly as an analytical component sanity check on `VAL-SEEN` with zero training steps (exact $acc = 1.000$).
  - *Explicit Scope Limitations & Non-Claims:* Documented that Track 1B does **NOT** exercise or validate Candidate 2's learned read, write, or gate pathways, nor does it establish register preservation. Candidate 2's learned mechanics are tested empirically for the first time in Stage 2 (Address Binding) ($EM_{read, val} \ge 0.950$) and Stage 3 (Composed Short Traces).
  - *Metric Incommensurability:* Prohibited ranking candidates across Track 1A and Track 1B.
  - *PI Decision Option Formulated (Approach 1):* Documented an open decision for the Principal Investigator at Review Gate 1: if Candidate 2 is required to undergo trained end-to-end pilot evaluation under Track 1A, it will add 5 training runs (~3.75 min), expanding the pilot budget from 30 to 35 runs and the campaign budget from 70 to 75 runs (~6.35 hrs, strictly within the $< 6.5$ hrs envelope).
* **Governance Status:** Proposal updated; SHA-256 recalculated and registered in `docs/reports/HISTORICAL_BASELINE.json`; implementation, training, and benchmark execution remain strictly unauthorized.

---

## Amendment — 2026-10-13 (Documentation Consistency Audit: Historical Supersession, Verification Coverage Distinction, and Vanilla GRU Parameter Reconciliation)

* **Clarification and Historical Supersession of Earlier Stage 1 Statements:**
  - *Context:* The earlier amendment titled `Pilot Interface Validity & Parameter Arithmetic Audit` (above) described Stage 1 (Primitive Modular Arithmetic) broadly as an end-to-end evaluation across each candidate's memory, gating, and update pathways via canonical 3-step execution traces (`SET r_0 a`, `SET r_1 b`, `ADD r_0 r_1`).
  - *Explicit Clarification of Supersession:* In accordance with the append-only research documentation policy (`docs/DOCUMENTATION_POLICY.md` §2–4), the earlier amendment text is preserved verbatim as an immutable historical record. However, this amendment explicitly clarifies that the earlier description of Stage 1 as a uniform end-to-end evaluation across all candidates is **historical and superseded** by the subsequent amendment (`Stage 1 Protocol Resolution: Track 1A vs. Track 1B Separation`) and the authoritative current specification in Section 5.
  - *Authoritative Current Protocol:* Under the adopted baseline protocol (Approach 2), Candidate 1 (Slot-MGU) and Control 2 (Vanilla GRU) are evaluated via trained end-to-end traces under **Track 1A**, whereas Candidates 2 (Cyclic-MGU) and 3 (MSM) undergo deterministic component verification under **Track 1B**. All current proposal sections (Executive Summary, Section 4.4, Section 5, Section 6, Section 7, Section 9, and Section 13 checklist) consistently specify this partitioned protocol.
* **Separation and Formal Distinction of Track 1B Verification Coverage by Candidate:**
  - The deterministic checks for Candidates 2 and 3 test fundamentally different architectural scopes and are formally distinguished into separate unit tests:
  - **Track 1B-C2 (Isolated Cyclic ALU Operator Unit Test — Candidate 2 Cyclic-MGU):**
    1. *Components and Data Pathways Exercised:* Strictly the parameter-free 20-dimensional cyclic rotation ALU operator $R(\theta_b) \Phi(a) = \Phi((a + b) \pmod{10})$ across harmonic frequencies $k \in \{1..5\}$ and the fixed cosine prototype value decoding bank.
    2. *Components Not Exercised:* Learned read query network ($\text{MLP}_{read}: 216 \to 128 \to 10$), monolithic write candidate generator ($\text{MLP}_{write}: 32 \to 212 \to 212$), convex gating network ($\text{MLP}_{gate}: 436 \to 212 \to 212$), continuous state-memory vector $S_t \in \mathbb{R}^{212}$, write blending, and register persistence across steps.
    3. *Test Input Domain & Pass Criterion:* All 16 pairs $(a, b) \in \text{VAL-SEEN}$ ($b \in \{1..4, 6..9\}, a \in \{0..9\}$). Pass criterion is exact numerical accuracy $acc = 1.000$ (16/16 correct).
    4. *What It Establishes vs. Does Not Establish:* Establishes that the mathematical formula and code implementation of the harmonic phase embedding $\Phi(\cdot)$ and 2D rotation blocks $R(\theta_k)$ correctly satisfy the abelian group identity $R(\theta_b) \Phi(a) = \Phi((a + b) \pmod{10})$ under cosine decoding. It does **NOT** establish that Candidate 2 can read register values from its monolithic continuous state, route them into the ALU, generate write candidates, update state via convex gating, preserve unwritten register values, or execute multi-step programs. **Candidate 2's learned state-memory and update pathways are completely unexercised and unvalidated by this test.**
    5. *Advancement Effect:* Passing Track 1B-C2 is a necessary pre-flight sanity check permitting progression to Stage 2. Candidate 2's learned read, write, and gating networks face their first empirical capability gate in **Stage 2 (Address Binding)** ($EM_{read, val} \ge 0.950$) and **Stage 3 (Composed Short Traces)** ($EM_{heldout} \ge 0.700$). If Candidate 2 fails Stage 2, it is immediately halted.
  - **Track 1B-C3 (Deterministic Slot-and-Arithmetic Pipeline Unit Test — Candidate 3 MSM):**
    1. *Components and Data Pathways Exercised:* The 4 discrete register slots $S_t = [s_{t,0}, s_{t,1}, s_{t,2}, s_{t,3}]$, hardwired instruction parsing and discrete routing for SET and ADD ($u_a = s_{t-1, r_{dest}}$, $u_b = s_{t-1, r_{src}}$), the 3-step execution pipeline ($t=1: \text{SET } r_0 \; a; t=2: \text{SET } r_1 \; b; t=3: \text{ADD } r_0 \; r_1$), deterministic slot-to-slot cyclic rotation ALU, untouched slot identity preservation ($\|\Delta s_r\| = 0.0$ for non-addressed slots $r \in \{1, 2, 3\}$), and fixed cosine prototype bank on destination slot $r_0$.
    2. *Components Not Exercised:* Trainable auxiliary context candidate network ($\text{MLP}_{aux\_cand}: 120 \to 680 \to 43$), trainable auxiliary gating network ($\text{MLP}_{aux\_gate}: 120 \to 680 \to 43$), auxiliary coordinate dynamics, recency approximation, and timestamp processing.
    3. *Test Input Domain & Pass Criterion:* The canonical 3-step execution trace on `VAL-SEEN` (16 pairs). Pass criterion is exact numerical accuracy $acc = 1.000$ (16/16 correct) at step $t=3$ on slot 0, and exact zero mutation $\|\Delta s_r\| = 0.0$ on unaddressed slots $r \in \{2, 3\}$.
    4. *What It Establishes vs. Does Not Establish:* Establishes that the primary deterministic slot-and-arithmetic pipeline (discrete routing, slot state persistence, cyclic rotation ALU, untouched slot identity) functions end-to-end without software defects or numerical drift. It does **NOT** establish that Candidate 3's trainable auxiliary state networks learn useful contextual features or remain stable under gradient descent during composed multi-step tasks.
    5. *Advancement Effect:* Passing Track 1B-C3 verifies the deterministic infrastructure of the hybrid architecture, permitting progression to Stage 2. Auxiliary pathway learning and stability are evaluated empirically in Stages 2–6.
  - *Prohibition of Unsupported Cross-Candidate Conclusions:* Because Track 1A measures empirical statistical learning under gradient descent, Track 1B-C2 evaluates an isolated algebraic operator, and Track 1B-C3 evaluates a deterministic slot-routing pipeline, all cross-candidate capability rankings and performance comparisons on Stage 1 are strictly prohibited.
* **Vanilla GRU Architecture, Parameter Count, and Interface Reconciliation:**
  - *Input Dimension:* Reconciled from Phase 1's 21-dimensional observation to Phase 1.6's 22-dimensional instruction token $x_t = [e_{op}, e(r_{dest}), e(r_{src}), e(v)] \in \{0, 1\}^{4 + 4 + 4 + 10} = \{0, 1\}^{22}$. Linear($22 \to 64$) has $1,472$ parameters ($22 \times 64 + 64$), increasing the encoder subtotal from $18,560$ to $18,624$ (+64 parameters).
  - *PyTorch GRU Cell ($H=160$):* Input dimension 256, hidden size 160: $3 \times (256 \times 160 + 160 + 160^2 + 160) = 3 \times 66,880 = \mathbf{200,640}$ parameters.
  - *Readout Heads and Total Parameter Counts:*
    1. **Stage 1 Dedicated Interface (10-class Head):** Linear($160 \to 10$) ($160 \times 10 + 10 = 1,610$ parameters). Total trainable parameters: $18,624 + 200,640 + 1,610 = \mathbf{220,874}$ parameters ($-2.42\%$ vs Model E $226,358$). Matches Candidate 1's dedicated stage head structure.
    2. **General Composed Sequence Interface (40-logit Head for Stages 2–6):** Linear($160 \to 40$) ($160 \times 40 + 40 = 6,440$ parameters). Total trainable parameters: $18,624 + 200,640 + 6,440 = \mathbf{225,704}$ parameters ($-0.29\%$ vs Model E $226,358$).
    3. **Historical Phase 1 Baseline Note:** The figure of 225,640 parameters previously cited reflected Phase 1's 21-dimensional input token ($18,560 + 200,640 + 6,440$). The +64 parameter difference to 225,704 is due to Phase 1.6's 22-dimensional instruction token.
  - *Parameter Envelope Compliance:* Both configurations ($220,874$ and $225,704$ parameters) fall strictly inside the registered envelope of $[203,722, 248,994]$ ($226,358 \pm 10\%$).
  - *Technical Contract Choice for PI Sign-Off:* Whether Stage 1 Vanilla GRU instantiates the dedicated 10-class head (220,874 parameters) or shares the general 40-logit head sliced on slot 0 (225,704 parameters) is documented as an open technical contract choice for PI determination at Review Gate 1.
* **Audit of Related References and Non-Contradiction Verification:**
  - Audited all occurrences of Vanilla GRU parameter counts across proposal sections (Section 4.4, Section 5, Section 7.1, Section 7.2, Section 9, Section 13), reconciling stale references to 225,640 with the dual-interface accounting (220,874 / 225,704) while preserving historical baseline provenance.
  - Confirmed that no model source code, training scripts, datasets, or checkpoints have been modified; zero files staged in git.
* **Governance Status:** Proposal updated; SHA-256 recalculated and registered in `docs/reports/HISTORICAL_BASELINE.json`; implementation, training, and benchmark execution remain strictly unauthorized.

## Amendment — 2026-10-13 (Final Track 1B Acceptance Correction: MSM Slot Preservation Criterion, Value-Coordinate Scope Narrowing, and Candidate 2 Domain Clarification)

* **Correction of MSM Slot Preservation Criterion (Track 1B-C3):**
  - *Context:* The previous amendment and draft specification identified slots $\{1, 2, 3\}$ as non-destination slots during the final instruction `ADD r0 r1`, but the explicit pass criterion checked exact zero mutation ($\|\Delta s_r\| = 0.0$) only on slots $\{2, 3\}$.
  - *Root Cause & Resolution:* The check on $\{2, 3\}$ inadvertently omitted source operand slot $r_1$. In a discrete register machine with hard multiplexed routing, reading a register does not mutate it. The instruction `ADD r0 r1` writes exclusively to destination register $r_0$ ($u_a = z_{t-1, 0}$, $u_b = z_{t-1, 1}$, $z_{3, 0} = \Phi((a + b) \pmod{10})$). Slot $r_1$ is read as $r_{src}$ and must remain unchanged ($s_{3, 1} = s_{2, 1}$). Consequently, all non-destination slots $r \in \{1, 2, 3\}$ must maintain exact zero mutation ($\|\Delta s_r\| = 0.0$).
  - *Authoritative Per-Step Preservation Invariants:* For each step of the canonical 3-step trace on `VAL-SEEN`:
    1. Step 1 ($t=1: \text{SET } r_0 \; a$): Destination is $r_0$. Non-destination slots $r_1, r_2, r_3$ remain unchanged ($\|\Delta s_r\| = 0.0$ for $r \in \{1, 2, 3\}$).
    2. Step 2 ($t=2: \text{SET } r_1 \; b$): Destination is $r_1$. Non-destination slots $r_0, r_2, r_3$ remain unchanged ($\|\Delta s_r\| = 0.0$ for $r \in \{0, 2, 3\}$; preserving slot $r_0 = \Phi(a)$).
    3. Step 3 ($t=3: \text{ADD } r_0 \; r_1$): Destination is $r_0$; source is $r_1$ (read-only). Non-destination slots $r_1, r_2, r_3$ remain unchanged ($\|\Delta s_r\| = 0.0$ for $r \in \{1, 2, 3\}$).
  - *Explicit Pass Criterion:* Exact numerical accuracy $acc = 1.000$ (16/16 correct) at step $t=3$ on slot $r_0$, AND exact zero mutation $\|\Delta s_r\| = 0.0$ across all non-destination slots at each respective step:
    - After Step 1 (`SET r0 a`): $\|\Delta s_r\| = 0.0$ for slots $r \in \{1, 2, 3\}$.
    - After Step 2 (`SET r1 b`): $\|\Delta s_r\| = 0.0$ for slots $r \in \{0, 2, 3\}$.
    - After Step 3 (`ADD r0 r1`): $\|\Delta s_r\| = 0.0$ for slots $r \in \{1, 2, 3\}$.
* **Narrowing of MSM Verification Scope (Primary Value-Coordinate Pathway Only):**
  - *Explicit Non-Claim:* Track 1B-C3 verifies strictly the **primary value-coordinate slot-and-arithmetic pathway** ($z_{t,r} \in \mathbb{R}^{10}$, discrete routing, cyclic ALU rotation, and untouched-slot preservation).
  - *Unexercised Components:* The test does **NOT** exercise or validate Candidate 3's trainable auxiliary candidate network ($\text{MLP}_{aux\_cand}: 120 \to 680 \to 43$), trainable auxiliary gating network ($\text{MLP}_{aux\_gate}: 120 \to 680 \to 43$), auxiliary coordinate dynamics ($h_{t,r} \in \mathbb{R}^{43}$), timestamp processing ($\psi(t_{slot\_write}), \psi(t_{val\_birth})$), or learned non-linear recency approximation.
  - *Boundary:* Passing Track 1B-C3 establishes only that the deterministic discrete slot substrate and cyclic arithmetic operate correctly without numerical drift. It does not validate the full 53-dimensional MSM state transition or complete candidate architecture. Auxiliary pathways face their first empirical capability gate in **Stage 2 (Address Binding)** ($EM_{read, val} \ge 0.950$) and subsequent composed sequence stages.
* **Clarification of Candidate 2 Input Domain (Track 1B-C2):**
  - *Domain Distinction:* Clarified that Track 1B-C2 evaluates strictly the 16 validation pairs $(a, b) \in \text{VAL-SEEN}$, as deterministically partitioned from the 80-pair seen domain ($b \in \{1..4, 6..9\}, a \in \{0..9\}$) in Section 5.1 (2 pairs per column $b$, spanning all ten target classes).
  - *Data Split Integrity:* The remaining 64 seen pairs comprise `TRAIN-SEEN` (unexercised in Track 1B), and the 20-pair `TEST-HELDOUT` ($b \in \{0, 5\}$) remains strictly withheld until post-pilot preregistered evaluation. No data split is altered.
* **Governance Status:** Proposal updated; SHA-256 recalculated and registered in `docs/reports/HISTORICAL_BASELINE.json`; implementation, training, and benchmark execution remain strictly unauthorized under the research hold.

---

## Amendment — 2026-10-13 (Final Scope and Dimensionality Reconciliation: Measured Value-Coordinate Preservation Criterion and Candidate 2 ALU Interface Contract)

* **Reconciliation of MSM Measured Preservation Criterion with Test Scope (Track 1B-C3):**
  - *Context & Root Cause:* The Track 1B-C3 pre-flight unit test verifies strictly the deterministic primary value-coordinate slot-and-arithmetic pathway ($z_{t,r} \in \mathbb{R}^{10}$) and explicitly excludes the 43 auxiliary coordinates ($h_{t,r} \in \mathbb{R}^{43}$) and trainable auxiliary networks ($\text{MLP}_{aux\_cand}, \text{MLP}_{aux\_gate}$). However, earlier text expressed its step-wise invariants and pass criteria using $\|\Delta s_r\| = 0.0$, where $s_{t,r} = [z_{t,r}, h_{t,r}] \in \mathbb{R}^{53}$.
  - *Measured Preservation Criterion:* The measured preservation pass criterion across all three trace steps on `VAL-SEEN` (16 pairs) is now explicitly formulated using $\|\Delta z_r\| = 0.0 \in \mathbb{R}^{10}$ for non-destination slots:
    1. Step 1 ($t=1: \text{SET } r_0 \; a$): destination slot $r_0$ is written ($z_{1, 0} = \Phi(a)$); non-destination slots $r \in \{1, 2, 3\}$ maintain $\|\Delta z_r\| = 0.0$.
    2. Step 2 ($t=2: \text{SET } r_1 \; b$): destination slot $r_1$ is written ($z_{2, 1} = \Phi(b)$); non-destination slots $r \in \{0, 2, 3\}$ maintain $\|\Delta z_r\| = 0.0$ (preserving slot $r_0 = \Phi(a)$).
    3. Step 3 ($t=3: \text{ADD } r_0 \; r_1$): destination slot $r_0$ is written ($z_{3, 0} = \Phi((a + b) \pmod{10})$); source slot $r_1$ is read as an operand but NOT written, so non-destination slots $r \in \{1, 2, 3\}$ maintain $\|\Delta z_r\| = 0.0$.
  - *Design-Level Invariant vs. Test Observation:* Exact full-slot identity ($s_{t, r} = s_{t-1, r} \implies \|\Delta s_r\| = 0.0 \in \mathbb{R}^{53}$) is identified separately as an architectural design-level invariant guaranteed by discrete slice non-assignment, NOT as an observed result of this value-coordinate-only test.
  - *Explicit Non-Claims:* The 43 auxiliary coordinates ($h_{t,r} \in \mathbb{R}^{43}$), auxiliary candidate and gating networks ($\text{MLP}_{aux\_cand}, \text{MLP}_{aux\_gate}$), write and provenance timestamps ($\psi(t_{slot\_write}), \psi(t_{val\_birth})$), and learned non-linear recency dynamics are completely unexercised and unmeasured by Track 1B-C3. They face their first empirical evaluation in Stage 2 (Address Binding, requiring $EM_{read, val} \ge 0.950$) and subsequent composed sequence stages.
* **Clarification of Candidate 2 ALU Dimensionality and Interface Contract:**
  - *Disambiguation of "20-dimensional cyclic rotation ALU":* Replaced the ambiguous phrasing with an explicit input/output contract consistent with the architectural equations:
    - *Concatenated Operand Input:* $[\Phi(a), \Phi(b)] \in \mathbb{R}^{20}$ (or pre-update operand snapshot $[\tilde{u}_a, \tilde{u}_b] \in \mathbb{R}^{20}$), formed by concatenating two 10-dimensional harmonic embeddings ($\Phi(v) \in \mathbb{R}^{10}$, each partitioned into $K=5$ two-dimensional harmonic frequency blocks).
    - *Block-Diagonal 2D Rotation ALU:* Computes 2D Givens rotations $c_{1, t}^{(k)} = R(\theta_b^{(k)}) \tilde{u}_a^{(k)} \in \mathbb{R}^2$ for each frequency $k \in \{1..5\}$.
    - *Harmonic Rotation Result Output:* $c_{1, t} = \Phi((a + b) \pmod{10}) \in \mathbb{R}^{10}$.
    - *Decoded Class Logits:* Evaluated via fixed prototype cosine bank $\Phi(c)^\top \in \mathbb{R}^{10 \times 10}$.
  - *Interface Contract:* The ALU operator is a fixed mapping $\mathbb{R}^{20} \to \mathbb{R}^{10}$ taking a 20-dimensional concatenated operand input and producing a 10-dimensional harmonic rotation output. No mathematical behavior, representation, or parameter count is modified.
* **Supersession of Earlier Historical Statements:**
  - In accordance with the append-only research documentation policy (`docs/DOCUMENTATION_POLICY.md`), earlier amendment text mentioning $\|\Delta s_r\| = 0.0$ or "20-dimensional cyclic rotation ALU operator" is preserved as immutable historical record. This amendment explicitly clarifies that the current authoritative protocol specifies $\|\Delta z_r\| = 0.0$ for the measured Track 1B-C3 pass criteria and the explicit $\mathbb{R}^{20} \to \mathbb{R}^{10}$ ALU input/output contract for Candidate 2.
* **Governance Status:** Proposal updated; SHA-256 recalculated and registered in `docs/reports/HISTORICAL_BASELINE.json`; implementation, training, and benchmark execution remain strictly unauthorized under the active research hold.

---

## Amendment — 2026-10-14 (PI Design Decision and Documentation Freeze: Adoption of Option B for MSM, Research Design Realignment, and Campaign Budget Reconciliation)

* **PI Decision 1 — Candidate 3 Architectural Reformulation (Option B Adoption):**
  - Formally adopted Option B for Candidate 3 (Modular Slot Machine). Removed the causally disconnected 43-dimensional auxiliary branch ($h_{t,r}$), auxiliary candidate network ($\text{MLP}_{aux\_cand}: 120 \to 680 \to 43$), auxiliary gating network ($\text{MLP}_{aux\_gate}: 120 \to 680 \to 43$), write/provenance timestamps ($\psi(t_{slot\_write}), \psi(t_{val\_birth})$), and non-linear recency approximations.
  - Recurrent state dimension reduced from $4 \times 53 = 212$ to strictly $M=4$ slots $\times 10$ harmonic cyclic value coordinates = **40 dimensions** ($z_{t,r} \in \mathbb{R}^{10}$, $K=5$ frequencies).
  - Trainable parameters reduced from $223,126$ to exactly **0 parameters**.
  - Classified strictly as a **parameter-free deterministic analytical reference / exact-rule control** for the documented task, exempt from the neural trainable parameter envelope ($226,358 \pm 10\%$). Prohibited describing MSM as a trainable neural candidate or as a general Bayes-optimal solution.
* **PI Decision 2 — Research Design Realignment:**
  - Structured the study around **two trainable candidate architectures** (Candidate 1 Slot-MGU and Candidate 2 Cyclic-MGU), the **Vanilla GRU baseline** (dedicated 10-class readout head), and the **deterministic MSM reference**.
  - Formally rejected Candidate 4 (Monolithic-MLP-MGU; 221,138 parameters). Prohibited claiming that the resulting study is a balanced $2 \times 2$ factorial experiment; scientific evaluation is conducted via targeted pairwise comparisons against baseline and reference.
* **PI Decision 3 — Stage 1 Protocol Ratification:**
  - Ratified Approach 2: retained Track 1A for the trained Slot-MGU and Vanilla GRU pilot (10 runs, 7.5 min) and Track 1B for deterministic pre-flight verification of Cyclic-MGU (Track 1B-C2) and MSM (Track 1B-C3) with 0 training steps.
* **PI Decision 4 — Candidate Advancement Policy:**
  - Ratified Advancement Option A: all candidates satisfying their preregistered advancement criteria advance to composed sequence training under Regimes A and B. Advance pruning (the earlier Option B compute alternative) was formally rejected.
* **PI Decision 5 — Vanilla GRU Interface Ratification:**
  - Formally selected the dedicated 10-class readout head configuration for Stage 1 Vanilla GRU, documented at **220,874 trainable parameters** ($18,624$ encoder + $200,640$ GRU + $1,610$ head; $-2.42\%$ vs Model E $226,358$). Preserved the parameter-accounting rationale.
* **PI Decision 6 — MSM Evaluation and Campaign Budget Reconciliation:**
  - Evaluated the deterministic reference on the prescribed evaluation datasets and conditions without treating it as a seed-trained model. Enumerated the six unique evaluation suites (Stages 1–6) for forward-only reference evaluation.
  - Reconciled parameter-training runs to exactly **55 runs** ($25 \text{ pilot} + 15 \text{ Regime A} + 15 \text{ Regime B} = 55$) totaling **258.75 minutes wall time** (~4.31 hours).
  - Retained the existing **35.00-minute evaluation allowance provisionally** across 35 seed-trained evaluation passes ($15 \text{ Regime A} + 15 \text{ Regime B} + 5 \text{ Control 1}$) plus deterministic reference checks.
  - Reconciled total campaign wall-clock time to $258.75 + 35.00 = \mathbf{293.75\text{ minutes}}$ (**~4.90 hours**), strictly satisfying the primary campaign compute envelope of $< 5.0$ hours.
* **Governance Status:** Proposal updated; documentation freeze enacted; SHA-256 recalculated and registered in `docs/reports/HISTORICAL_BASELINE.json`. Implementation, training, and experimental execution remain strictly unauthorized under the active research hold pending formal authorization.

---

## Amendment — 2026-10-14 (Scientific Consistency Audit: RQ3 Formulation, Deferred Curriculum Realignment, and Composition Scope Clarification)

* **RQ3 Formulation and Non-Claim on Untested Neural Composition:**
  - *Context:* The initial formulation of RQ3 asked whether composing explicit slot addressing (RQ1) and cyclic arithmetic units (RQ2) into a recurrent update achieved systematic generalisation.
  - *Correction & Alignment with Approved Design:* Under the approved research design (PI Decisions 1 & 2), the study evaluates two separate trainable neural candidates (Candidate 1 Slot-MGU and Candidate 2 Cyclic-MGU), the Vanilla GRU baseline, and a parameter-free deterministic analytical reference (Candidate 3 MSM Option B). There is no trainable candidate combining explicit slot memory and cyclic arithmetic units (Candidate 4 Monolithic-MLP-MGU was rejected per PI Decision 2, and Candidate 3 MSM is a parameter-free deterministic reference).
  - *Authoritative RQ3 Definition:* RQ3 is formally reformulated to ask whether the two trainable candidate architectures (Slot-MGU with explicit slot addressing, and Cyclic-MGU with cyclic arithmetic inductive bias) achieve the specified composed-program generalisation ($acc_{heldout} \ge 0.800$, $EM_{heldout} \ge 0.700$), length extrapolation (with $T=20$ as the primary gate requiring $EM_{T=20} \ge 0.700$, and $T=30, 50$ as secondary diagnostic probes), and extended persistence ($EM_{read} \ge 0.950$ at $K=20$) thresholds, compared against the Vanilla GRU baseline and deterministic MSM exact-rule reference.
  - *Explicit Non-Claim:* The approved study does **not** test a trainable neural architecture combining explicit slots and cyclic arithmetic. No empirical claims are made regarding that untested neural composition.
* **Deferred Curriculum Realignment (Section 3 Hypothesis C):**
  - Realigned Section 3's Hypothesis C text with the active specification:
    1. Slot-MGU and Cyclic-MGU are the trainable candidate architectures evaluated under the matched end-to-end optimizer protocol across composed sequence stages.
    2. Vanilla GRU serves as the recurrent baseline (220,874 trainable parameters for Stage 1 dedicated 10-class head; 225,704 general sequence).
    3. MSM (Option B) is a parameter-free deterministic analytical reference and receives zero parameter-training runs.
    4. Any contingent curriculum trigger must refer strictly to the relevant trainable candidates (Slot-MGU or Cyclic-MGU) upon documented empirical evidence of credit-assignment failure.
    5. The curriculum remains strictly deferred; no new training stages or candidate expansions are added to the active specification.
* **Clarification of Historical Supersession Regarding Candidate 3 Auxiliary Networks:**
  - *Context:* In accordance with the append-only documentation policy (`docs/DOCUMENTATION_POLICY.md` §2–4), earlier amendments dated 2026-10-13 preserve mentions of Candidate 3's auxiliary networks ($\text{MLP}_{aux\_cand}, \text{MLP}_{aux\_gate}$) and 43-dimensional context dynamics being unexercised by Track 1B-C3 and facing evaluation in Stages 2–6.
  - *Explicit Clarification of Supersession:* This amendment explicitly clarifies that all such statements are **historical and superseded** by PI Decision 1 adopting Option B. Under the operative specification, the auxiliary branch is completely eliminated, recurrent state dimension is strictly 40 dimensions ($4 \times 10$), trainable parameters are exactly 0, and MSM receives zero parameter-training runs across all stages (evaluated forward-only as a deterministic exact-rule reference).
* **Scoped Contribution and Factorial Balance Claims Reaffirmed:**
  - Section 11.2 was updated to align with the absence of a trainable composite architecture.
  - Reaffirmed that the study is an unbalanced architectural comparison under Factorial Option 1, not a balanced $2 \times 2$ factorial experiment.
* **Governance Status:** Proposal updated; documentation freeze enacted; SHA-256 recalculated and registered in `docs/reports/HISTORICAL_BASELINE.json`. Implementation, training, and benchmark execution remain strictly unauthorized under the active research hold pending formal authorization.

---

## Amendment — 2026-10-18 (Model E Held-Out Generalization: Four-Cell SCLM Roster, `b = 3` Development Proxy, Masked {0, 3, 5} Regime, Unified Objective, and Two-Tier Protected-Test Firewall)

**Status of this amendment.** Append-only registration, under explicit Principal Investigator authorization, of the Model E held-out generalization study (four-cell SCLM benchmark) as a Phase 1.6 extension. It formally registers the revised four-cell candidate roster, the updated Regime B masking rule, the sequence-disjoint `PROXY-DEV` (`b = 3`) development split, the unified objective, strict validation-only checkpoint selection, the registered five-seed set, the progression thresholds, the runtime stop-gate boundary, and the two-tier evaluation firewall. Earlier proposal text — including the earlier three-candidate roster, the rejection of historical Candidate 4, and all completed measurements — is preserved unchanged as the historical record and is superseded only within the scope of this study. No Phase 2+ mechanism (branching, verifiers, recursion) is authorized by this amendment.

### A. Architectural Note — Revised Four-Cell Roster (exact interfaces and parameter counts)

1. **Model A — SCLM (Slot-Cyclic Latent Model; 8,255 trainable parameters).** Four explicit register slots `z_{t,r} ∈ R^53` (total latent budget 4 × 53 = 212). Initialization `z_{0,r} = Write(one_hot(0)) ∈ [−1, 1]^53`. Shared decoder `Dec: 53 → 64 → 10` (GELU; raw logits; softmax only to form operand distributions; 4,106 parameters). Shared writer `Write: 10 → 64 → 53` (GELU; tanh output; 4,149 parameters). Dispatch: `SET r_dest v` → `z_{t,r_dest} = Write(e(v))`; `SWAP r_dest r_src` → atomic slot exchange from the pre-update snapshot; `READ r_dest` → identity (zero mutation); `ADD r_dest r_src` → fixed exact mod-10 circular convolution over pre-update decoded operand distributions, `p_sum[k] = Σ_i p_{t−1,r_dest}[i] · p_{t−1,r_src}[(k − i) mod 10]`, then `z_{t,r_dest} = Write(p_sum)`. Untouched slots satisfy `||Δz_r|| = 0` by construction.
2. **Model B — Slot-MLP control (10,249).** Identical to Model A except the fixed convolution is replaced by a learned ALU `[p_{t−1,r_dest}, p_{t−1,r_src}] ∈ R^20 → 64 → 10` (GELU; softmax output; 1,994 parameters) on `ADD`. A vs B is a **targeted within-family comparison of complete ALU modules** (+1,994 parameters, +24.2% on slot models); it is not a parameter-matched pure causal comparison.
3. **Model C — Cyclic-MGU control (227,062).** Monolithic state `S_t ∈ R^212`, `S_0 = 0_212`. Frozen composition reusing the existing Model E architecture components (`src/re_reason/core/ablation_models.py`, `NonlinearCandidateMGUModel` family; `src/re_reason/cognition/ablation_cores.py`, `NonlinearCandidateMGUCore`): shared pre-state/output head `212 → 40` (8,520 parameters); per-step ALU feature `f_ALU ∈ R^10` computed by the fixed mod-10 circular convolution over pre-state softmax distributions on `ADD` and `f_ALU = 0_10` otherwise; observation encoder `[x_t, f_ALU] ∈ R^32 → 64 → 256` + LayerNorm (GELU; 19,264 parameters); recurrent core `f_t = σ(W_f e_t + U_f S_{t−1})`, `h_reset = f_t ⊙ S_{t−1}`, `C_t = tanh(MLP_cand([h_reset, e_t]))` (`468 → 146 → 212`, GELU), `S_t = (1 − f_t) ⊙ S_{t−1} + f_t ⊙ C_t ∈ [−1, 1]^212` (199,278 parameters); post-transition readout `Head(S_t) ∈ R^40`. The `h_reset`/`tanh` candidate semantics are those of the existing live `NonlinearCandidateMGUCore` implementation; the parameter arithmetic is verified against it (8,520 + 19,264 + 199,278 = 227,062).
4. **Model D — Monolithic-MLP control (229,056).** Identical to Model C except `f_ALU = softmax(ALU_learned([p_{t−1,r_dest}, p_{t−1,r_src}]))` on `ADD` (same 1,994-parameter ALU architecture as Model B). Trained fresh from scratch under the masked protocol; historical Model E checkpoints are preserved as historical baselines and are **never** evaluated on the protected test. C vs D is a targeted within-family comparison of complete ALU modules (+1,994 parameters, +0.88% on monolithic models).
5. **Comparative epistemics.** A vs C and B vs D are **bundled full-architecture comparisons** (memory organization, routing, operational semantics, and capacity confounded; 27.5× parameter disparity). Claiming a balanced 2 × 2 factorial main-effect decomposition is prohibited.

### B. Dataset, Generator, and Masking Registration

* **TRAIN-MASKED (6,000 sequences, T = 10)** and **VAL-SEEN (1,200 sequences)**: 100% of `ADD` transitions must satisfy the runtime source-register invariant `v(r_src) ∉ {0, 3, 5}`, verified by deterministic replay. Immediate literals in `SET` remain unrestricted (`0..9`); masking applies strictly to the runtime source value of `ADD` transitions. The existing `b ∉ {0, 5}` Regime B behavior remains available unchanged for prior artifacts.
* **PROXY-DEV (proposed new specification; 1,200 sequences)**: exactly one designated `ADD` transition per sequence with runtime source `v(r_src) = 3`; zero `ADD` transitions with runtime source in `{0, 5}`; any other `ADD` transitions must have runtime sources in `{1, 2, 4, 6, 7, 8, 9}`; post-`ADD` target class `(v(r_dest) + 3) mod 10` exactly balanced at **120 sequences per class**; designated `ADD` step position, register addresses, intervening ops, and `SET` literals randomized. PROXY-DEV construction and split checks are frozen before training. This is a development proxy only: proxy success is not a claim about `b ∈ {0, 5}`.
* **Bounded attempts**: `max_attempts = 100` per sequence for TRAIN/VAL generation and `max_attempts = 100` per proxy sequence. Attempt exhaustion raises a fatal `RuntimeError` before training; silent sample-count reduction and mask relaxation are prohibited.
* **Deterministic split derivation**: split base seed 42; TRAIN-MASKED seed = base + 1000; VAL-SEEN seed = base + 50000 (inherited Regime B namespaces); PROXY-DEV seed = base + 70000; TEST-PROTECTED remains the existing sealed derivation (base + 90000) and is not regenerated during development.
* **TEST-PROTECTED (`b ∈ {0, 5}`; 1,200 sequences)**: permanently sealed; no queries or content inspections during planning and development. Split-disjointness verification is metadata-only (SHA-256 hash-manifest collision check outputting only pass/fail status and overlap counts, which must be 0) and must never stream, browse, or inspect protected examples. Establishing and approving the protected-set hash manifest remains a Review Gate 2 item that must be resolved **before training** (not before this implementation stage).

### C. Objective and Checkpoint-Selection Registration

* **Unified objective (all four models):** `L_total = (1/B) Σ_i (L_term^(i) + 1.0 · L_step^(i))`, with `L_term^(i) = (1/4) Σ_r CE(ŷ_{T,r}, y_{T,r})` at the final step `T = 10`, and per-step unique-written-register cross-entropy supervised on `W_t` (`SET`/`ADD` → `{r_dest}`; `SWAP` → `{r_dest, r_src}`; `READ` → `∅`), normalized by `N_writes^(i) = Σ_t |W_t^(i)|` and defined as 0 when `N_writes = 0`. No internal candidate-state, gate, or ALU intermediate losses; fixed coefficient `λ_step = 1.0`.
* **Checkpoint selection:** minimum validation cross-entropy on `VAL-SEEN` only, evaluated every 50 updates; on exact ties, the earliest update step is selected (preregistered refinement; secondary EM tie-breaking omitted). `PROXY-DEV` and `TEST-PROTECTED` are never used for training, early stopping, tuning, or checkpoint selection.

### D. Seeds, Thresholds, Gating, Runtime Boundary, and Final-Test Firewall

* **Seeds and budget:** seeds 42–46 (5 seeds per model); 4 models × 5 seeds = 20 registered runs; AdamW (lr = 3×10⁻³, weight decay = 10⁻²); batch size 64; 3,000 optimizer updates; validation every 50 updates.
* **Tier 1 (PROXY-DEV gate; Model A only):** joint pass requires `acc_heldout ≥ 0.800` AND `EM_terminal ≥ 0.700` on ≥ 4/5 seeds with zero seeds in the rejection zone (`acc < 0.250` OR `EM ≤ 0.150`). `acc_heldout` is the macro recall across true output classes present among eligible held-out `ADD` transitions (runtime source = 3 on PROXY-DEV); `EM_terminal` is the fraction of sequences whose predicted final four-register state exactly matches the target state at `t = T`. If Model A fails Tier 1, execution halts and the protected split remains sealed permanently.
* **Tier 2 (single protected pass):** only if Tier 1 passes, evaluate all four frozen models (A, B, C, D) together in a single program pass over `TEST-PROTECTED`; historical Model E is excluded. The protected evaluation is consumed once and is never re-run; artifact overwrite refusal is unconditional for the Tier 2 result.
* **Runtime boundary:** aggregate wall-clock for all 20 runs plus scheduled validation overhead must be `< 12.0 h`. The first 100 updates of all 20 runs (including the scheduled validations at updates 50 and 100) form the profiling stop-gate; the projected campaign duration is `Σ 30 · t_100` over the 20 runs. If the projection is ≥ 12.0 h, the campaign stops after profiling and escalates without altering the registered schedule; a hard cumulative 12.0-h stop is enforced during continuation.

### E. Governance Conformance for This Amendment

* SHA-256 of this amended proposal updated in `docs/reports/HISTORICAL_BASELINE.json` with a new `history` entry; a dated entry appended to `docs/RESEARCH_LOG.md`; `pytest tests/test_documentation_governance.py -v` must pass.
* Registered implementation artifacts: `src/re_reason/core/sclm_models.py` (four candidates), `src/re_reason/benchmarks/register_machine_benchmark.py` (additive generator/split extensions), `src/re_reason/training/heldout_objective.py` (unified objective, held-out metrics, Tier 1 gate classification), `experiments/run_model_e_heldout_campaign.py` (guarded campaign runner); focused tests `tests/test_sclm_models.py`, `tests/test_heldout_benchmark_splits.py`, `tests/test_heldout_objective.py`, `tests/test_heldout_campaign_runner.py`.
* This implementation stage authorizes code, tests, and the guarded runner only. No training, no profiling run, no dataset generation, no checkpoint loading or inspection, no protected-split access, no held-out evaluation, and no push are authorized by this amendment; each execution stage requires its own explicit PI authorization.


