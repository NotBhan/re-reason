# RE:Reason Experiment Runners

This directory contains executable entry points for benchmarks and ablations.

## Requirements
All experiment runners must strictly support:
- `--seed <int>`: Deterministic seed setting across all random generators.
- `--device <str>`: Target execution device (`cpu` or `cuda`).
- Standardized logging format.

## Phase Status
- Phase 0: Contract validation and test infrastructure active.
- Phase 1+: Benchmark execution logic will be implemented in subsequent phases.
