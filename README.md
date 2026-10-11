# RE:Reason

> **An experimental research framework and candidate recurrent latent inference architecture for multi-step algorithmic reasoning.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/target-AMD%20ROCm%20%2F%20CPU-purple.svg)](https://rocm.docs.amd.com/)
[![Governance](https://img.shields.io/badge/governance-append--only-red.svg)](docs/DOCUMENTATION_POLICY.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🔬 Core Mantra

$$\text{BUILD} \longrightarrow \text{MEASURE} \longrightarrow \text{ABLATE} \longrightarrow \text{INTERPRET} \longrightarrow \text{KEEP OR KILL} \longrightarrow \text{EXTEND}$$

RE:Reason is **an active research framework, not a finalized or frozen architecture**. Every architectural component is treated as an empirical hypothesis subject to strict ablation, compute-matched benchmarking, and potential falsification. No mechanism belongs to the architecture unless empirical ablations justify its computational cost under controlled parameter budgets.

---

## 📌 Problem Statement & Motivation

Standard autoregressive sequence models display three structural inefficiencies when solving multi-step algorithmic problems:

1. **High Token Overhead:** Intermediate reasoning requires emitting full natural-language token sequences (Chain-of-Thought), consuming high memory bandwidth and quadratic or linear sequence compute.
2. **Irreversible Context Contamination:** Once an incorrect intermediate token is emitted into the context window, all subsequent self-attention attends to that faulty token. Autoregressive inference has no native transactional rollback mechanism.
3. **Static Computation Depth:** Standard feedforward transformer layers allocate an identical compute budget to every token regardless of whether a step is trivial or computationally hard.

### Central Research Hypothesis
> *Can a compact (500K to 15M parameters), weight-tied recurrent latent core with dynamic halting, speculative branch exploration, and verifier-gated state commits solve multi-step algorithmic problems with higher parameter efficiency, better error containment, or lower inference compute than compute-matched feedforward, recurrent, and sequence baselines?*

---

## 🚫 Boundaries: What RE:Reason Is and Is Not

| What RE:Reason **IS** | What RE:Reason **IS NOT** |
| :--- | :--- |
| **Latent State Machine:** Maintains an evolving continuous internal state vector $S_t \in \mathbb{R}^D$ across time without emitting discrete text tokens. | **Not a Large Language Model:** No text generation, no vocabulary, no BPE tokenizers. |
| **Speculative State Isolation:** Explores parallel candidate trajectories in isolated tensor branches (`torch.clone()`). | **Not an Autonomous Agent Framework:** No external tool use, no web scraping, no OS integration. |
| **Controlled Algorithmic Testbed:** Evaluated on synthetic algorithmic benchmarks with exact mathematical ground truth. | **Not Proven Superior to Transformers:** Candidate architecture under empirical test; hypotheses are subject to falsification. |
| **Hardware-Targeted:** Engineered for rapid execution on consumer GPUs (AMD ROCm 8 GB VRAM) and CPU fallback. | **Not an Online Continual Learner:** Model parameters are frozen during inference; test-time adaptation is latent-only. |
| **Append-Only Research Record:** Governed by immutable, SHA-256 hashed experiment reports and negative-result preservation. | **Not "Hallucination-Proof":** Latent representations can drift, collapse, or misdecode unless constrained. |

---

## 🏛️ Target Conceptual Architecture

The conceptual pipeline operates across two distinct memory domains to isolate intermediate speculative computation from committed persistent memory:

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
        Rollback to Clean S_t               Commit S_{t+1} = s^(k*)_R
```

### Reality-Check Nomenclature
None of these individual mechanisms represent ungrounded inventions; each maps directly to established machine learning literature:

| Proposed Term | Underlying Technique | Precedents & Analogues | Category |
| :--- | :--- | :--- | :--- |
| **Persistent Latent State** | Recurrent hidden state vector | RNNs, GRUs, LSTMs, MGU, State-Space Models (S4, Mamba), Dreamer | Known technique |
| **Recursive Cognition** | Weight-tied depth recurrence | Universal Transformers, ALBERT depth-recurrence, Recurrent Highway Networks, TRM | Known technique |
| **Adaptive Halting** | Learned dynamic depth | Adaptive Computation Time (ACT), PonderNet, Early-Exit networks | Known technique |
| **Isolated Hypotheses** | Latent candidate rollouts / search | Beam search, Monte Carlo Tree Search (MCTS), particle filters | Known technique |
| **Typed Decisions** | Multi-head output projections | Classification logits, bounded scalar heads, confidence estimates | Known technique |
| **Transactional Commit** | Snapshotting and rollback | Copy-on-write memory patterns, isolated tensor cloning (`torch.clone()`) | Engineering pattern |

---

## 📐 Mathematical Specification: Active Validated Model

Per [docs/RESEARCH_STATUS.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/RESEARCH_STATUS.md), the currently validated active system is **Phase 1 Model E** (`NonlinearCandidateMGUModel`), an Update-Gate RNN / Minimal Gated Unit (MGU) family architecture:

### 1. Mathematical Formulation
For observation $x_t \in \mathbb{R}^{D_{\text{in}}}$ and previous state $S_{t-1} \in \mathbb{R}^D$:

$$\begin{aligned}
e_t &= \text{MLP}_{\text{enc}}(x_t) \in \mathbb{R}^{D_e} \\
f_t &= \sigma(W_f [S_{t-1}, e_t] + b_f) \in (0, 1)^D \quad &&\text{(Element-wise Update Gate)} \\
h_{\text{reset}, t} &= f_t \odot S_{t-1} \in \mathbb{R}^D \quad &&\text{(Pre-gated State)} \\
C_t &= \tanh(\text{MLP}_{\text{cand}}([h_{\text{reset}, t}, e_t])) \in (-1, 1)^D \quad &&\text{(Candidate Delta Vector)} \\
S_t &= (1 - f_t) \odot S_{t-1} + f_t \odot C_t \in (-1, 1)^D \quad &&\text{(Convex-Bounded Commit)} \\
y_t &= W_{\text{out}} S_t + b_{\text{out}} \in \mathbb{R}^{N_{\text{reg}} \times N_{\text{classes}}} \quad &&\text{(Linear Readout)}
\end{aligned}$$

### 2. Parameter Budget Breakdown (226,358 Parameters)
* **Observation Encoder $\text{MLP}_{\text{enc}}$:** Linear(21, 256) $\to$ GELU $\to$ Linear(256, 256) (71,424 params)
* **Update Gate $W_f$:** Linear(212 + 256, 212) (99,428 params)
* **Candidate Network $\text{MLP}_{\text{cand}}$:** Linear(212 + 256, 146) $\to$ GELU $\to$ Linear(146, 212) (99,566 params)
* **Linear Readout $W_{\text{out}}$:** Linear(212, 40) (8,520 params)
* **State Dimension:** $D = 212$, Embedding: $D_e = 256$, Candidate Hidden: $D_c = 146$.

### 3. Supervised Objective: Candidate/Local Supervision ($L_E$)
The model is trained via local supervision on candidate activations rather than terminal-only loss:

$$L_E = \text{CE}(y_T, y^*_T) + \frac{1}{|W|} \sum_{t=1}^T \sum_{r \in W_t} \text{CE}(\text{head}(C_t)_r, y^*_{t, r})$$

where $W_t$ is the set of registers written at step $t$, $\text{head}(C_t)_r$ is the linear decision head applied directly to candidate vector $C_t$, and $y^*_{t, r}$ is the ground-truth value.

* **Impact:** Lifted Register Tracking $T=10$ Exact Match from **$0.540 \pm 0.018$** (Terminal-only $L_A$) to **$0.755 \pm 0.019$** (Candidate/Local $L_E$).
* **Extrapolation:** Zero-shot $T=20$ Exact Match reaches **$0.407 \pm 0.023$**.

---

## 🎯 Benchmark Suite: Synthetic Algorithmic Reasoning

To isolate computational mechanisms without the confounding variables of natural language pretraining, RE:Reason evaluates on procedural synthetic tasks with unambiguous ground truth:

### 1. Register Tracking (Active Primary Benchmark)
Tracks internal state transitions over $M=4$ registers ($R_0 \dots R_3$) holding modular digits $v \in \{0 \dots 9\}$.
* **Step Transitions:**
  * `SET r v`: Sets register $r \leftarrow v$.
  * `SWAP r1 r2`: Swaps contents of registers $r_1$ and $r_2$.
  * `ADD r1 r2`: Modular addition $r_1 \leftarrow (r_1 + r_2) \bmod 10$.
* **Evaluation Windows:** Standard training and testing at $T=10$ steps; out-of-distribution length extrapolation evaluated zero-shot at $T=20$ steps.

### 2. Secondary & Deferred Benchmarks
* **Parity DAGs:** Multi-hop boolean evaluation across directed acyclic graphs.
* **Constrained Pathfinding & Mazes:** Spatial grid traversal requiring backtracking across dead ends (evaluating speculative rollouts).
* **Mixed Arithmetic:** Multi-step algebraic expressions evaluating operator precedence and intermediate registers.

---

## 🔬 The Empirical Research Arc: Experiments A through AE

Rather than assuming theoretical mechanisms work, the repository follows a rigorous empirical investigation chain documenting why components succeed, fail, or remain bounded:

```
[Phases 1 & 1.5] ──► [Early Feasibility] ──► [Residual Isolation] ──► [Representation Probes] ──► [Interface & Scale] ──► [Optimizer & Transfer]
  Model E Bounded       Branching Oracle       Exp G–K: Gate Vector      Exp P: Class-A Budget    Exp U: Bayes=1.0        Exp Z–AC: Qualification
  Convex Update         & Verifiers Fail       Exp L–N: Parity Basin     Exp Q–S: Activations     Exp V: Sequence Readers Exp AD: High Train Fit
  (0.755 EM)            (Goodhart Artifact)    Exp O: Parity Causal? No  Exp T: Operand-Blind e_t Exp W: Data Scale Flat  Exp AE: Fit-Without-Transfer
```

| Experiment / Report | Research Question & Intervention | Key Result & Metric | Decision & Classification |
| :--- | :--- | :--- | :--- |
| **Phase 1 / 1.5 Ablations** | Does bounded convex update beat standard recurrent baselines? | Model E reaches 45.55% vs GRU (11.25%) under initial protocol; LayerNorm ruled out. | **Adopt Model E**; classified as MGU variant (no novelty claim). |
| **Branching Feasibility** | Can oracle branch sampling ($K=8$) solve algorithmic failures? | Oracle reaches 0.6913, but per-draw information is at chance; noise recovers 46% of gain. | **Branching unvalidated**; oracle signal cannot be extracted. |
| **Verifier Feasibility** | Can an auxiliary learned scorer distinguish valid transitions? | Headline AUROC is an artifact of top-1 class balance; verifier provides no ranking signal. | **Learned verifiers deferred**; avoid Goodharting on weak states. |
| **Supervision Repair** | Does candidate-level supervision resolve credit assignment? | Candidate loss $L_E$ raises $T=10$ EM from 0.540 to 0.755; $T=20$ from 0.22 to 0.407. | **Adopt $L_E$ as official objective**; frozen checkpoints locked. |
| **Exp G & H (Gate Interpolation)** | Does the commit failure stem from scalar dilution? | Vector gate reproduces commit bit-exactly; scalar mean gate collapses accuracy (0.855 $\to$ 0.182). | **Scalar summaries rejected**; coordinate-wise structure is mandatory. |
| **Exp I, J, K (Commit Interventions)** | Can gate scaling ($\gamma$) or balance remapping ($\kappa$) fix ADD? | Gate scaling monotonic in local margin delivery, but degrades sequence EM at $\gamma \ge 1.25$. | **Interventions rejected**; commit rule frozen; bottleneck is upstream. |
| **Exp L, M, N (Geometry & Parity)** | Where does the off-path state land when ADD fails? | 100% of failures clear previous competitor, landing in a 3rd basin preserving parity (74% vs 50% null). | **State-dominant parity alignment** (91–98% from state $\leftrightarrow$ label alignment). |
| **Exp O (Representation Test)** | Is modular parity alignment causally responsible for errors? | Subtracting parity component fails: matched control rescues same transitions; breaks 24% of successes. | **Parity causality rejected (O-D)**; parity is a correlated diagnostic. |
| **Exp P (Error Budgeting)** | Which step in the inference pipeline causes sequence failures? | Earliest task failure is **Candidate Generation (Class A)** in 59.7% (T=10) and 67.2% (T=20). | **Class A dominates**; commit residual is secondary (32%); readout is minor (8%). |
| **Exp Q & R (Candidate Probing)** | Is target information recoverable from frozen candidate $C_t$? | Linear probes recover only 26.7%; apparent gains on hard pools collapse on held-out operands (0.176 vs 0.493). | **Memorization artifact (R-D)**; candidate representation information-limited. |
| **Exp S & T (Upstream Probing)** | Where is operand information lost in the network? | Probes across internal layers ($h1$, out-pre) show $C_t$ is best stage; encoder $e_t$ is operand-blind (11–13%). | **Encoder operand-blind (T-D)**; single-step observation cannot see operands. |
| **Exp U (Interface Bayes)** | Does the raw task interface determine the target? | Current observation is ambiguous ($L=1$ Bayes 0.29), but history reaches **Bayes 0.998 at $L=4$** and **1.0 at full prefix**. | **Interface-identifiable (U-D)**; Bayes ceiling = 1.0, but flat probes fail (gap 0.782). |
| **Exp V (Sequence Readers)** | Do causal sequence readers (GRU, causal TCN) resolve the gap? | Parameter-matched GRU (0.083) and TCN (0.112) fail to solve Class A; Bayes-reader gap is 0.887. | **Sequence-limited (V-B)**; temporal inductive bias alone does not fix access. |
| **Exp W (Data Scale)** | Does scaling training transitions ($1.5\text{k} \to 10\text{k}$) unlock rules? | Readers remain flat (0.08 $\to$ 0.11); training accuracy remains low (0.13–0.23); Bayes gap unchanged. | **Data-saturated / bounded-limited (W-B)**; underfitting persists. |
| **Exp X & Y (Budget Controls)** | Are sequence readers simply under-optimized? | Scaled update budgets (200 $\to$ 1,000) and fitting controls show readers struggle with convergence. | Optimization budget isolated as a key lever. |
| **Exp Z, AA, AB (Qualification)** | What training recipe allows readers to fit sequence histories? | AdamW with tuned weight decay (3e-3, 1e-2) achieves high training fit (GRU 0.858, TCN 0.655). | **AdamW recipe qualified**; fitting obstacle removed. |
| **Exp AC & AD (Qualified Access)** | Does high training fit translate to canonical test accuracy? | Despite training accuracy climbing to 0.44–0.86, canonical Class-A test accuracy stays ~0.09. | **Qualified readers fail transfer (AD-B)**; large train-to-test gap. |
| **Exp AE (Fit vs. Transfer)** | Do qualified readers learn transferable rules on held-out operands? | GRU and TCN fit the training pool far above baseline, but held-out operand accuracy stays at/below baseline (0.116). | **Fit-without-transfer (AE)**; readers memorize training pairs without general rule induction. |

---

## 🗺️ Development Roadmap: 11 Sequential Phases

To avoid conflating failure modes, each architectural component is introduced and evaluated in complete isolation:

```
Phase 0: Architecture Contracts & Test Infrastructure
   │
Phase 1: Minimal Persistent-State Core (R=1, K=1) ── [ACTIVE VALIDATED PHASE]
   │
Phase 2: Fixed-Depth Recursive Cognition (R in [1, 2, 4, 8])
   │
Phase 3: Adaptive Halting (PonderNet / ACT)
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

### Staging Rationale & Decision Gates
1. **Phases 4 through 7 are decoupled:** Branching is evaluated alone (Phase 4), before deterministic verification (Phase 5), before learned verifiers (Phase 6), before transactional commits (Phase 7).
2. **Current Decision Rule:** Phase 2+ machinery remains **deferred in code** while the core representation and sequence learning bottlenecks exposed by Experiments G–AE are analyzed and resolved.

---

## 💻 Hardware Profile & Engineering Guardrails

The framework is strictly engineered for reproducible execution on consumer workstation hardware:

* **Primary Target GPU:** AMD Radeon RX 6600M (8 GB VRAM, Navi 23, Linux ROCm 6.x).
* **Host RAM:** 16 GB.
* **Framework:** PyTorch 2.3+ (native operations only; zero CUDA-exclusive dependencies).
* **Execution Rule:** Mandatory CPU fallback across all modules and tests.
* **VRAM Ceiling:** Active allocation capped strictly under 4.5 GB VRAM.
* **Memory Diagnostics:** CI and unit tests continuously check for uncollected autograd computation graphs, reference cycles, and monotonic PyTorch caching allocator growth.

---

## 📂 Repository Structure

```
re-reason/
├── docs/                               # Master documentation suite & research record
│   ├── 01_project_overview.md          # Motivation, boundaries, and nomenclature
│   ├── 02_core_architecture.md         # Full conceptual architecture blueprint
│   ├── 03_development_roadmap.md       # 11-phase development roadmap (Phase 0–10)
│   ├── 04_experiments_and_evaluation.md# Benchmarks, metrics, and ablation matrix
│   ├── 05_engineering_and_implementation.md # Tech stack, memory, and code conventions
│   ├── 06_risks_research_questions_and_future.md # Kill criteria & failure modes
│   ├── DOCUMENTATION_POLICY.md         # Canonical append-only research policy
│   ├── RESEARCH_LOG.md                 # Chronological research ledger
│   ├── RESEARCH_STATUS.md              # Current validated synthesis
│   └── reports/                        # Immutable, SHA-256 pinned experiment reports
├── src/re_reason/                      # Core framework implementation
│   ├── benchmarks/                     # Procedural algorithmic benchmark suites
│   │   ├── base.py                     # BaseTask and TaskBatch abstractions
│   │   ├── register_tracking.py        # Register Tracking benchmark implementation
│   │   └── dataset_split.py            # Train / Val / Test splitting utilities
│   ├── cognition/                      # Recurrent cognition interfaces & base cores
│   ├── core/                           # Models (NonlinearCandidateMGU, GRU baseline)
│   │   ├── model.py                    # ReReasonPhase1Model & decision heads
│   │   ├── ablation_models.py          # Phase 1/1.5 Model variants (Models A–F)
│   │   └── baseline_gru.py             # Baseline A GRU implementation
│   ├── decision/                       # Typed decision projection heads
│   ├── hypothesis/                     # Speculative state isolation interfaces
│   ├── state/                          # Latent state container abstractions
│   ├── training/                       # Loss formulations and training protocols
│   ├── utils/                          # Caching, seed control, and logging
│   └── verification/                   # Deterministic and learned verifier contracts
├── experiments/                        # Experiment runners, analyzers, and outputs
│   ├── checkpoints/                    # SHA-256 pinned model checkpoints
│   ├── run_experiment_*.py             # Deterministic experiment runners
│   └── analyze_experiment_*.py         # Statistical analysis and figure generation
├── tests/                              # Unit, integration, and governance tests
│   ├── test_documentation_governance.py# SHA-256 hash baseline integrity test
│   └── test_experiment_*.py            # Experiment validation suites
└── PROJECT_PLAN.md                     # Master project index and operational status
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/NotBhan/re-reason.git
cd re-reason

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e .

# Or using standard pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run Test Suite & Governance Verification
Run the complete test suite:

```bash
# Full test suite
pytest tests/ -v

# Documentation governance and immutability ledger test
pytest tests/test_documentation_governance.py -v
```

### 3. Run Benchmark Diagnostics
Run the diagnostic evaluation suite on frozen Phase 1 checkpoints:

```bash
PYTHONPATH=src python experiments/run_qa_diagnostics.py
```

### 4. Reproduce an Experiment (Example: Experiment AE)
```bash
# Run experiment runner
PYTHONPATH=src python experiments/run_experiment_ae.py

# Run analysis and figure generation
PYTHONPATH=src python experiments/analyze_experiment_ae.py

# Execute verification tests
PYTHONPATH=src pytest tests/test_experiment_ae.py -v
```

---

## 📜 Research Governance & Documentation Policy

Completed research documentation in this repository is **historical and append-only**:

* **Immutable Evidence (`docs/reports/*.md`):** Once completed, an experiment report is never rewritten, deleted, or back-edited. All numerical results, conclusions, and failed hypotheses are frozen.
* **SHA-256 Ledger Anchor (`HISTORICAL_BASELINE.json`):** Every completed report is hashed and tracked in `docs/reports/HISTORICAL_BASELINE.json`, enforced continuously by `tests/test_documentation_governance.py`.
* **Chronological Ledger (`docs/RESEARCH_LOG.md`):** Append-only record tracking the chronological sequence of experiments. Existing entries are never edited or reordered.
* **Current Validated Synthesis (`docs/RESEARCH_STATUS.md`):** Mutable synthesis of what the project currently believes, preserving historical context and citing specific reports.
* **Amendments over Silent Edits:** Any correction, reinterpretation, or bug fix is appended as a dated amendment (`## Amendment — YYYY-MM-DD`).

For complete policy specifications, see [docs/DOCUMENTATION_POLICY.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/DOCUMENTATION_POLICY.md).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
