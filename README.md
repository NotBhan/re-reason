# RE:Reason

> **An experimental research framework and candidate recurrent latent inference architecture for multi-step algorithmic reasoning.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/target-AMD%20ROCm%20%2F%20CPU-purple.svg)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🔬 Core Mantra

$$\text{BUILD} \longrightarrow \text{MEASURE} \longrightarrow \text{ABLATE} \longrightarrow \text{INTERPRET} \longrightarrow \text{KEEP OR KILL} \longrightarrow \text{EXTEND}$$

RE:Reason is **an active research framework, not a finalized or frozen architecture**. Every architectural component is treated as an empirical hypothesis subject to strict ablation, compute-matched benchmarking, and potential falsification.

---

## 📌 Project Motivation

Standard autoregressive language models display three fundamental structural inefficiencies when solving multi-step algorithmic problems:

1. **High Token Overhead:** Multi-step reasoning requires emitting full natural-language token sequences (Chain-of-Thought), consuming high memory bandwidth and quadratic or linear sequence compute.
2. **Irreversible Context Contamination:** Once a faulty token is generated into the context window, all subsequent self-attention attends to that error without any native transactional rollback.
3. **Static Computation Depth:** Standard feedforward architectures allocate an identical compute budget to every token regardless of step complexity.

### The Central Research Hypothesis
> *Can a compact (500K to 15M parameters), weight-tied recurrent latent core with dynamic halting, speculative branch exploration, and verifier-gated state commits solve multi-step algorithmic problems with higher parameter efficiency, better error containment, or lower inference compute than compute-matched feedforward, recurrent, and sequence baselines?*

---

## 🚫 Boundaries: What RE:Reason Is and Is Not

| What RE:Reason **IS** | What RE:Reason **IS NOT** |
| :--- | :--- |
| An experimental latent-state machine with speculative state isolation and transition verification. | **Not a Large Language Model** (no discrete tokenizers, no natural language text generation). |
| A compact research testbed (500K–15M params) optimized for local consumer hardware (AMD ROCm / CPU). | **Not an autonomous agent framework** (no web browsing, OS tools, or prompt chaining). |
| Evaluated on controlled, synthetic algorithmic benchmarks with mathematical ground truth. | **Not proven superior to Transformers** or claimed to be "hallucination-free." |
| Governed by strict append-only research documentation and empirical falsification. | **Not an online continual learner** (model weights are frozen during inference). |

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
None of these mechanisms are novel inventions in isolation; each maps directly to established machine learning literature:
* **Persistent Latent State:** Recurrent hidden state vector (RNNs, GRUs, LSTMs, MGU, S4/Mamba).
* **Recursive Cognition:** Weight-tied depth recurrence (Universal Transformers, ALBERT, TRM).
* **Adaptive Halting:** Learned dynamic depth (Adaptive Computation Time, PonderNet).
* **Isolated Hypotheses:** Speculative candidate search (Beam search, MCTS, particle filters).
* **Transactional Commit:** Copy-on-write snapshotting and rollback (`torch.clone()`).

---

## 📊 Current Validated State & Research Progress

Per [docs/RESEARCH_STATUS.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/RESEARCH_STATUS.md), the repository follows strict staging rules. Later-phase machinery (branching, learned verifiers, transactional rollbacks) is **intentionally deferred** until representation bottlenecks are diagnosed and resolved.

### Active Validated Model: Phase 1 (Model E)
* **Architecture:** `NonlinearCandidateMGUModel` (226,358 parameters, 212-dim state, 256-dim embedding, 146-dim candidate hidden layer).
* **Commit Mechanism:** Convex-bounded element-wise interpolation:
  $$S_t = (1 - f_t) \odot S_{t-1} + f_t \odot C_t$$
  where $C_t = \tanh(\text{MLP}_{\text{cand}}([f_t \odot S_{t-1}, e_t]))$.
* **Benchmark Performance (Register Tracking):**
  * $T=10$ Exact Match: **$0.755 \pm 0.019$** (Candidate/Local supervision) vs $0.540$ (Terminal-only).
  * Zero-shot $T=20$ Exact Match: **$0.407 \pm 0.023$**.

### Systematic Diagnostic Journey (Experiments G through AE)
When early feasibility tests revealed that learned verifiers and naive branch sampling provided no useful signal on raw latent states, the project executed an extensive mechanistic investigation:
* **Error Budgeting (Exp P):** Demonstrated that **candidate-generation errors (Class A)** account for ~60–67% of all sequence-level failures, concentrated primarily in addition operations.
* **Probing & Localization (Exp Q–T):** Internal MLP layer probes showed information loss is not an output-projection artifact; observation encoder $e_t$ was found to be structurally operand-blind to single-step additions.
* **Interface & Access (Exp U–W):** Proved mathematical target information is 100% present in input history by step $L=4$ (Bayes ceiling = 1.000), but bounded flat probes and standard sequence readers underfit even with $6.7\times$ data scaling.
* **Instrument Qualification & Generalization (Exp X–AE):** Evaluated optimization schedules, architectural initialization, and out-of-distribution operand generalization across candidate architectures.

---

## 🛠️ Hardware Profile & Environment

The framework is strictly engineered for reproducible execution on consumer workstation hardware:
* **Primary Target GPU:** AMD Radeon RX 6600M (8 GB VRAM, Navi 23, Linux ROCm 6.x).
* **Host RAM:** 16 GB.
* **Software:** Python 3.11+, PyTorch 2.3+ (native operations only; zero CUDA-exclusive dependencies).
* **Execution Constraint:** Mandatory CPU fallback across all modules and tests.
* **Memory Budget:** Active memory ceiling capped under 4.5 GB VRAM.

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
│   ├── cognition/                      # Recurrent cognition interfaces & base cores
│   ├── core/                           # Models (NonlinearCandidateMGU, GRU baseline)
│   ├── decision/                       # Typed decision projection heads
│   ├── hypothesis/                     # Speculative state isolation interfaces
│   ├── state/                          # Latent state container abstractions
│   ├── training/                       # Loss formulations and training protocols
│   ├── utils/                          # Caching, seed control, and logging
│   └── verification/                   # Deterministic and learned verifier contracts
├── experiments/                        # Experiment runners, analyzers, and outputs
├── tests/                              # Unit, integration, and governance tests
└── PROJECT_PLAN.md                     # Master project index and operational status
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install dependencies using `uv` or `pip`:

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
Run the complete test suite, including architecture unit tests and documentation governance checks:

```bash
pytest tests/ -v
```

To run only the governance integrity suite:
```bash
pytest tests/test_documentation_governance.py -v
```

### 3. Run Benchmark Baselines
Execute the Phase 1 Register Tracking task benchmark:

```bash
python experiments/run_qa_diagnostics.py
```

---

## 📜 Research Governance & Documentation Policy

Completed research documentation in this repository is **historical and append-only**:
* **Completed reports (`docs/reports/*.md`) are immutable:** Every finished experiment report is hashed with SHA-256 and verified by [`tests/test_documentation_governance.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/tests/test_documentation_governance.py).
* **Failed experiments and negative results are preserved:** Numerical measurements and rejected hypotheses are never deleted or rewritten.
* **Corrections are dated amendments:** If an interpretation changes, an amendment is appended (`## Amendment — YYYY-MM-DD`).

For details, see [docs/DOCUMENTATION_POLICY.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/DOCUMENTATION_POLICY.md).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
