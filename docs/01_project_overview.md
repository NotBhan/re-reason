# Project Overview

## 1. Project Purpose & Status

RE:Reason is currently a research framework and experimental inference architecture, not a finalized architecture. It is designed to test whether persistent latent state, weight-tied recursive computation, isolated hypothesis exploration, verification, and transactional state commits can provide measurable advantages in parameter efficiency, compute allocation, or error containment over compute-matched baselines.

Individual components do not automatically belong to the final architecture. A component only becomes part of the final system if its empirical ablations justify keeping it under controlled compute and parameter budgets.

The system performs iterative latent computation without autoregressive natural-language token generation. It does not output discrete token sequences to conduct intermediate reasoning (such as natural-language chain-of-thought). Instead, it maintains a continuous state vector, speculatively branches candidate trajectories, validates transitions, and transactionally commits winning states to long-term memory.

The immediate goal is not AGI, an LLM replacement, natural language generation, or a multimodal foundation model. The immediate goal is to establish whether this core inference loop works on controlled algorithmic tasks on local consumer hardware.

## 2. Problem Statement

Standard autoregressive sequence models display three structural inefficiencies in multi-step problem solving:

1. High Token Overhead: Intermediate reasoning requires generating full natural-language token sequences, consuming high memory bandwidth and quadratic or linear sequence compute.
2. Irreversible Context Contamination: Once an incorrect token is generated into the context window, subsequent self-attention attends to that faulty token. Standard inference has no native transactional rollback mechanism.
3. Static Computation Depth: Standard feedforward models allocate a static amount of compute to every token regardless of problem difficulty.

RE:Reason explores whether maintaining an isolated internal state machine with speculative execution and verification can address these specific limitations in a purely latent representation.

## 3. Core Hypothesis

The central research hypothesis is:

> Can a small, weight-tied latent recurrent core with isolated hypothesis exploration and verifier-gated state commits solve multi-step algorithmic problems with higher parameter efficiency, better error containment, or lower inference compute than compute-matched feedforward, recurrent, and sequence baselines?

We do not know the answer. Recurrent models historically suffer from optimization instability, vanishing gradients, representational drift, and mode collapse. If testing shows this configuration underperforms standard baselines, the architecture will be deemed non-viable.

## 4. Reality-Check Nomenclature

None of the individual mechanisms in RE:Reason represent new inventions. Each maps directly to established literature:

| Proposed Term | Underlying Technique | Historical Precedents | Category |
| :--- | :--- | :--- | :--- |
| Persistent Latent State | Recurrent hidden state vector | RNNs, GRUs, LSTMs, State-Space Models (S4, Mamba), Memory Networks, World Models (Dreamer). | Known technique |
| Recursive Cognition | Weight-tied depth recurrence | Universal Transformers, ALBERT depth-recurrence, Recurrent Highway Networks, TRM. | Known technique |
| Adaptive Halting | Learned dynamic depth | Adaptive Computation Time (ACT), PonderNet, Early-Exit networks. | Known technique |
| Isolated Hypotheses | Latent candidate rollouts / search | Beam search, Monte Carlo Tree Search (MCTS), particle filters, speculative execution. | Known technique |
| Typed Decisions | Multi-head output projections | Standard softmax classification heads, sigmoid heads, bounded scalar regression. | Known technique |
| Transactional Commit | State snapshotting and rollback | Copy-on-write memory patterns, isolated tensor cloning, speculative state isolation. | Engineering pattern |

The only possible research question is whether this specific combination produces a measurable Pareto improvement under controlled compute and parameter constraints.

## 5. What RE:Reason Is and Is Not

### What RE:Reason IS:
* An experimental research framework and candidate inference architecture (500K to 15M parameters).
* An iterative latent-state machine with speculative state isolation and transition verification.
* A testbed evaluated on controlled, synthetic algorithmic benchmarks with ground-truth verification.
* A system designed for rapid local execution on an 8 GB consumer GPU.

### What RE:Reason IS NOT:
* Not a finalized architecture (components remain provisional until justified by ablations).
* Not a Large Language Model (no text generation, no token vocabulary, no BPE tokenizers).
* Not an autonomous agent framework (no external tool use, no web scraping, no OS integration).
* Not "hallucination-free" or proven superior to Transformers.
* Not an online continual learner (model weights are frozen at inference).
* Not a perception system (no custom vision or audio encoders).

## 6. Project Philosophy

The project operates strictly under the empirical cycle:

BUILD -> MEASURE -> ABLATE -> INTERPRET -> KEEP OR KILL -> EXTEND

* We do not design the entire system upfront under the assumption that it works.
* We do not add components because they sound intelligent.
* Every component must test a specific hypothesis, undergo ablation, and be discarded if it fails to justify its computational overhead.
* A rigorous negative result (proving a mechanism does not work under equal compute) is a successful scientific outcome.

## 7. Research vs. Engineering Boundary

We explicitly divide the project into three categories:

### Category A: Known Techniques (Buildable Now)
* Standard PyTorch recurrent tensor operations.
* Gated residual MLP and LayerNorm blocks.
* Multi-head classification and regression projections.
* Isolated tensor cloning (`torch.clone()`) and state rollbacks.
* Deterministic programmatic constraint checkers.
* Synthetic procedural dataset generators.

### Category B: Engineering Problems (Feasible, but design unclear)
* Stabilizing PonderNet/ACT training dynamics to prevent step-count collapse.
* Generating genuinely diverse hypothesis branches rather than duplicate trajectories.
* Calibrating learned verifiers to avoid shortcut exploitation and reward hacking.
* Eliminating retained references and computation graphs in PyTorch caching allocators on AMD ROCm.

### Category C: Research Hypotheses (Unproven assumptions)
* Whether searching in continuous latent space provides any benefit over discrete beam search under equal compute.
* Whether transactional state rollback prevents long-horizon drift without causing deadlock.
* Whether weight-tied recursion generalizes to depths unseen during training.

## 8. Hardware Constraints

Development and benchmarking are constrained to consumer hardware:
* GPU: AMD Radeon RX 6600M (8 GB GDDR6 VRAM, Navi 23 architecture, Linux ROCm 6.x).
* Host RAM: 16 GB DDR4/DDR5.
* Tooling Rules: Native PyTorch operations only. No NVIDIA CUDA-exclusive libraries (no `flash-attn`, no CUDA Triton kernels). Mandatory CPU fallback for every test and experiment.
* Model Size Budget: 500K to 15M parameters.

## 9. Overall Architecture Overview

RE:Reason organizes inference into two memory domains:
1. Committed Domain: Holds the clean, persistent master state $S_t$.
2. Speculative Domain: Spawns $K$ isolated candidate branches via isolated tensor cloning, executes $R$ recursive cognition passes with adaptive halting, validates candidate transitions, and atomically commits the winning candidate to $S_{t+1}$ or aborts to $S_t$ (leaving master state unchanged).

For detailed component mathematics, transition equations, and architectural views, see [docs/02_core_architecture.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/02_core_architecture.md).  
For the stage-gated implementation phases, see [docs/03_development_roadmap.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/03_development_roadmap.md).
