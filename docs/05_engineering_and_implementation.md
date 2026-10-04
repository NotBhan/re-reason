# Engineering and Implementation

## 1. Technology Stack

RE:Reason relies on standard, production-ready open-source tools:

* Language: Python 3.11+
* Deep Learning Framework: PyTorch 2.3+ (standard build with ROCm 6.x compatibility)
* Mathematical & Data Operations: NumPy 1.26+, Polars 0.20+
* Configuration & Validation: Pydantic v2, PyYAML
* Testing: pytest 8.0+, pytest-benchmark, hypothesis (property-based testing)
* Diagnostics & Profiling: PyTorch Profiler (`torch.profiler`), psutil

### Strict Dependencies Constraint
Zero CUDA-exclusive libraries. The codebase must not import or depend on NVIDIA-specific packages such as `flash-attn`, `triton` CUDA kernels, or `bitsandbytes` CUDA builds. All operations rely strictly on standard PyTorch primitives (`torch.nn.functional.scaled_dot_product_attention`, native linear/norm layers). Every module must execute cleanly on both `--device cpu` and `--device cuda/hip`.

## 2. Hardware Profile & AMD ROCm Constraints

Target system: AMD Radeon RX 6600M (8 GB VRAM, Navi 23, Linux ROCm 6.x), 16 GB Host RAM.

### Operational Guardrails:
* VRAM High-Water Cap: Active model allocations must remain under 4.5 GB to prevent PCIe page-spilling into system RAM.
* Contiguous Allocations: Avoid high-frequency allocations of dynamic shapes within recursive loops to prevent ROCm memory fragmentation.
* Profiling Diagnostics: Use `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` to track tensor footprints.

## 3. Repository Structure

```
re-reason/
├── PROJECT_PLAN.md             # Concise project index & entry point
├── pyproject.toml              # Build config, dependencies, pytest settings
├── docs/                       # Project documentation
│   ├── 01_project_overview.md
│   ├── 02_core_architecture.md
│   ├── 03_development_roadmap.md
│   ├── 04_experiments_and_evaluation.md
│   ├── 05_engineering_and_implementation.md
│   └── 06_risks_research_questions_and_future.md
├── configs/                    # Strict type-checked YAML configs
│   ├── base.yaml
│   ├── phase1_core.yaml
│   ├── phase2_recursion.yaml
│   └── baselines/
├── src/
│   └── re_reason/
│       ├── core/               # Orchestration & high-level inference runtime
│       │   ├── model.py        # ReReasonModel wrapper
│       │   └── engine.py       # Step-by-step stateful runtime
│       ├── state/              # Latent state management & speculative isolation
│       │   ├── container.py    # LatentState dataclass (clone, rollback, commit)
│       │   └── tracker.py      # State drift and norm diagnostics
│       ├── cognition/          # Recurrent computation modules
│       │   ├── core.py         # Weight-tied residual block
│       │   └── halting.py      # Adaptive halting gate (PonderNet)
│       ├── hypothesis/         # Branching & candidate generation
│       │   ├── brancher.py     # K-candidate proposal generator
│       │   └── diversity.py    # Pairwise diversity & duplicate metrics
│       ├── verification/       # Dual-layer verification subsystem
│       │   ├── deterministic.py# Hard programmatic constraint checkers
│       │   ├── learned.py      # Auxiliary trajectory scoring network
│       │   └── composite.py    # Staged verification coordinator
│       ├── decision/           # Output heads
│       │   └── heads.py        # Multi-head categorical & bounded scalar projections
│       ├── benchmarks/         # Procedural synthetic benchmarks
│       │   ├── base.py         # Abstract benchmark contract
│       │   ├── register_swap.py# Phase 1 Register Tracking
│       │   ├── parity_dag.py   # Phase 2 Parity Decision Trees
│       │   ├── mixed_arith.py  # Phase 3 Mixed Arithmetic
│       │   ├── deadend_maze.py # Phase 4 Backtracking Mazes
│       │   └── adversarial.py  # Phase 7 Distractor injection suite
│       ├── training/           # Training loops & loss functions
│       │   ├── trainer.py      # Supervised BPTT training loop
│       │   └── losses.py       # Multi-task loss (Task + Ponder penalty + Verifier)
│       └── utils/              # Utilities
│           ├── logging.py      # Structured JSONL logger
│           └── profiling.py    # VRAM, latency, and FLOP counter
├── tests/                      # Automated test suite
│   ├── test_state.py           # Unit tests for isolation & rollback
│   ├── test_cognition.py       # Unit tests for recursion & halting
│   ├── test_branching.py       # Unit tests for candidate generation & diversity
│   ├── test_verification.py    # Unit tests for deterministic & learned verifiers
│   └── test_baselines.py       # Equivalence & sanity tests for baselines
└── experiments/                # Executable benchmark and ablation scripts
    ├── run_benchmark.py        # Runs model against task suite
    ├── run_ablation.py         # Executes specific ablation sweep
    └── analyze_results.py      # Generates summary tables and statistical tests
```

## 4. State Container Design & Speculative State Isolation

### Tensor Lifecycle
The `LatentState` container manages two distinct operational tensors:
1. `_committed_state`: Persistent master tensor $S_t \in \mathbb{R}^{B \times D}$.
2. `_working_branches`: Speculative candidate tensor $s^{(k)} \in \mathbb{R}^{B \times K \times D}$.

### Isolated Tensor Cloning (Speculative State Isolation)
Isolation is achieved by cloning tensors before branching:
```python
def spawn_branches(self, k: int) -> torch.Tensor:
    # Isolated tensor cloning: creates an independent memory buffer
    # without retaining computation graphs across time steps.
    self._working_branches = self._committed_state.unsqueeze(1).repeat(1, k, 1).clone()
    return self._working_branches
```
We do not describe `torch.clone()` as copy-on-write. True copy-on-write involves deferred lazy allocation, which PyTorch does not implement natively. We describe this strictly as **isolated tensor cloning**, **state snapshots**, or **speculative state isolation**.

### Memory Management & Leak Prevention
PyTorch uses a caching memory allocator that retains reserved memory pools even after tensors are freed. Therefore, demanding that GPU memory return to zero is technically incorrect.

Leak detection tests must specifically check:
* No Retained Graph References: Ensure `.detach()` is called when committing states across time steps, preventing unrolled autograd graphs from chaining infinitely.
* Reference Cycle Elimination: Explicitly clear candidate buffers after step finalization:
  ```python
  def clear_working_memory(self):
      self._working_branches = None
  ```
* Cleanup of Speculative Branches: Verify that rejected candidate buffers are promptly unreferenced.
* Monotonic Allocation Growth Diagnostics: In automated leak tests, measure `torch.cuda.memory_allocated()` over 1,000 sequential steps. After a warm-up phase (steps 1–10), the net allocated tensor bytes must exhibit zero monotonic drift between step $N$ and step $N+100$. Allocator statistics serve as diagnostics rather than proof of the complete absence of leaks.

## 5. Training Loop & Recursive Unrolling

### Backpropagation Through Time (BPTT)
* Recursion steps $r = 1 \dots R$ are unrolled within the standard PyTorch autograd graph for time step $t$.
* Gradients flow back through all $R$ recursive steps to the shared parameter set $\theta$.
* Gradient Clipping: All training loops enforce `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to mitigate exploding gradients inherent in unrolled weight-tied loops.

### Batching Hypothesis Branches
To optimize GPU core utilization on the AMD RX 6600M, the $K$ candidate branches are flattened into the batch dimension during recursive cognition:
$$\text{Effective Batch Size} = B \times K$$
The tensor $\mathbb{R}^{B \times K \times D}$ is reshaped to $\mathbb{R}^{(B \cdot K) \times D}$, passed through the vectorised cognition core, and reshaped back to $\mathbb{R}^{B \times K \times D}$ before evaluation.

## 6. Testing Strategy

The test suite enforces mathematical invariants and operational isolation:

### Unit Tests
* `test_state_isolation`: Verifies that mutating candidate branch $k$ does not alter $S_t$ or other branches $j \ne k$.
* `test_rollback_bit_exact`: Verifies that aborting a transaction leaves $S_t$ numerically identical to its pre-step value (`torch.equal(S_t_after, S_t_before)`).
* `test_weight_tying`: Verifies that gradients from all unrolled recursive steps accumulate into the identical parameter tensors in `CognitionCore`.

### Property-Based Tests (Hypothesis Library)
* Random Tensor Dimensions: Test arbitrary batch sizes $B \in [1, 64]$ and latent dimensions $D \in [64, 512]$ to ensure dimension invariance.
* Extreme Value Stability: Test NaN, Inf, and near-zero inputs to ensure LayerNorm and halting controllers do not produce numerical exceptions.

### Deterministic Reproducibility
All test scripts and benchmark runners must expose a `--seed` flag that sets:
```python
import torch, numpy as np, random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

## 7. AI Coding-Agent Workflow

To maintain development velocity without compromising scientific rigor, the collaboration workflow follows strict boundary rules:

### Strict Rule for AI Agents:
> **AI coding agents may modify only the currently active phase unless explicitly instructed otherwise.** Later phases must not be implemented prematurely or leaked into earlier code.

### Appropriate Tasks for AI Agents:
* Generating boilerplate dataclasses, Pydantic schemas, and PyTorch module skeletons.
* Writing comprehensive pytest suites and mock classes.
* Writing procedural synthetic dataset generators (e.g., random graph generators, arithmetic trees).
* Implementing CLI benchmark runners and structured JSONL logging.
* Profiling memory footprints and identifying un-vectorized loops.
* Refactoring code for PEP 8, Ruff, and Black compliance.

### Decisions Strictly Requiring Human Engineering Review:
* Benchmark Validity: Inspecting generated benchmarks to confirm that tasks do not permit trivial shortcut memorization.
* Baseline Fairness: Reviewing baseline architectures to ensure they are not hobbled or undertrained.
* Interpreting Results: Determining whether a performance difference represents a genuine architectural property or a hyperparameter tuning discrepancy.
* Architectural Changes: Adding or removing structural components.
* Enforcing Kill Rules: Making the final decision to kill a component or terminate an entire phase.

For research hypotheses, failure modes, and kill criteria, see [docs/06_risks_research_questions_and_future.md](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/docs/06_risks_research_questions_and_future.md).
