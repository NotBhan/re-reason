# RE:Reason Project Index & Specification

## Project Overview & Status

RE:Reason is currently an experimental machine learning research framework and candidate inference architecture, not a finalized architecture. It tests whether a recurrent latent inference architecture can perform multi-step computation more efficiently and robustly than compute-matched baselines. Individual components only become part of the final architecture if empirical ablations justify keeping them.

The system performs iterative latent computation without autoregressive natural-language token generation. It maintains a persistent internal state, executes weight-tied recursive computation with dynamic halting, speculatively explores candidate branches using isolated tensor cloning, validates candidate transitions, and commits state updates transactionally.

Every architectural idea is treated as an empirical hypothesis subject to strict ablation and falsification.

## Core Mantra

BUILD -> MEASURE -> ABLATE -> INTERPRET -> KEEP OR KILL -> EXTEND

## Target Platform & Hardware

* Primary Hardware: AMD Radeon RX 6600M (8 GB VRAM, Navi 23, Linux ROCm 6.x), 16 GB Host RAM
* Framework: PyTorch 2.3+ (native operations only; zero CUDA-exclusive dependencies)
* Execution: Mandatory CPU fallback across all modules and tests
* Parameter Range: 500K to 15M parameters

## Master Documentation Suite

The complete engineering specification, architectural blueprints, experimental methodology, and risk assessments are organized into six modular documents under `docs/`:

### [1. Project Overview](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/01_project_overview.md)
* Status as an experimental research framework, not a finalized architecture.
* Project purpose, problem statement, and narrow core hypothesis.
* Reality-check nomenclature mapping proposed terms to established ML techniques (RNNs, Universal Transformers, beam search, speculative execution).
* Explicit boundaries defining what RE:Reason is and is not.
* Research vs. engineering categorization.

### [2. Core Architecture](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/02_core_architecture.md)
* Full conceptual architecture: Committed master domain vs. speculative working domain.
* Component specifications: Observation encoder, latent state container, recursive core, adaptive halting, hypothesis brancher, typed decision heads, verifiers, and commit gate.
* Clarification of speculative state isolation via isolated tensor cloning (`torch.clone()`) vs. copy-on-write memory.
* Candidate/transition validation rather than arbitrary latent semantic correctness.
* Target conceptual pipeline diagram and separate minimal Phase 1 architecture diagram.

### [3. Development Roadmap](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/03_development_roadmap.md)
* 11 sequential implementation phases (Phase 0 through Phase 10).
* Explicit staging rationale: Phase 4 evaluates branching in isolation; Phase 5 introduces deterministic verification; Phase 6 evaluates learned verification; Phase 7 connects verification to transactional commit.
* Phase 0 contracts and CI leak detection.
* Phase 1 minimal persistent core ($R=1, K=1$, locked decisions: flat $S \in \mathbb{R}^{B \times 256}$, terminal loss at $t=10$, single pass, no branching/recursion/halting/verification/commit) benchmarked directly against `nn.GRUCell` (matching GRU shows viability, not superiority).
* Phase 2 recursive cognition ($R \in [1, 2, 4, 8]$).
* Phase 3 adaptive halting evaluated primarily on compute reduction and accuracy retention.
* Phase 4 parallel hypothesis branching evaluated under compute parity.
* Phase 8 optional test-time latent adaptation branch.
* Concrete objectives, benchmarks, metrics, baselines, and predefined project decision rules for every phase.

### [4. Experiments and Evaluation](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/04_experiments_and_evaluation.md)
* Procedural synthetic benchmark suite: Register Tracking, Parity DAGs, Mixed Arithmetic, Dead-End Mazes, Constrained Pathfinding, and Adversarial Long-Horizon Tasks.
* Baseline models: Compact GRU (Baseline A), Small Transformer Decoder (Baseline B), and Systems C through F.
* System matrix and ablation plan.
* Evaluation metrics: Task accuracy, FLOPs/latency, branch diversity, duplicate rate, and verifier FAR/FRR/ECE (with positive = valid/accepted, negative = invalid/rejected).
* State Pollution Index (SPI) reframed as a geometric diagnostic; downstream continuation accuracy and recovery rate serve as primary proof of state integrity.
* Strict experimental methodology: Multi-seed runs ($\ge 5$ seeds), explicit compute matching rules ($K=4$ vs. compute-matched $K=1$), and statistical significance criteria.

### [5. Engineering and Implementation](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/05_engineering_and_implementation.md)
* Technology stack: Python 3.11+, PyTorch 2.3+, NumPy, Pydantic v2, pytest, Polars.
* Target hardware profile, VRAM ceiling (4.5 GB active allocation), and AMD ROCm operational rules.
* Repository layout and module responsibilities.
* State container design, tensor lifecycle, and proper PyTorch caching allocator leak diagnostics (checking for uncollected graphs, reference cycles, and monotonic allocation growth).
* Recursive unrolling, gradient clipping, hypothesis batching ($B \times K$), and testing strategy.
* AI coding-agent workflow boundary: **AI coding agents may modify only the currently active phase unless explicitly instructed otherwise.** Division of labor between AI agents and human engineering review.

### [6. Risks, Research Questions, and Future Work](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/06_risks_research_questions_and_future.md)
* Engineering risks: Vanishing/exploding gradients, hypothesis mode collapse, halting collapse, verifier Goodharting, and ROCm driver edge cases.
* Unanswered research questions: Continuous latent search vs. discrete search, long-horizon state drift, and recursion depth extrapolation.
* Concrete failure modes and explicit project kill criteria.
* Criteria defining meaningful positive and negative results.
* Explicitly deferred research: Natural language generation, tokenizers, Swarm Semantics, continual weight updates, and autonomous agents.
* Conceptual external NLP interface boundary schema.

## Current Project Status & Immediate Next Step

* Status: Documentation synchronized and fully corrected; codebase clean slate.
* Phase: Ready for Phase 0 (Architecture contracts, abstract base classes, test runner, and tensor isolation tests).
* Execution Directive: Do not implement model code or advance beyond Phase 0 until the test contracts are established and passing.
