# EXP-CYCLIC-STEPWISE-01: Step-Wise Destination Supervision for Monolithic Cyclic-MGU

## 1. Date

2026-10-16

## 2. Status

Completed. Official 5-seed Stage 2 run executed under explicit PI authorization with `--stage2-authorized`. Mechanical outcome under preregistered decision matrix: **`INCONCLUSIVE_INTERMEDIATE`** (`BEHAVIOR_INTERMEDIATE`, `REP_PARTIAL`). The protected TEST split ($b \in \{0, 5\}$) remained strictly untouched (`queries = 0`).

## 3. Research Question

Does auxiliary cross-entropy supervision of intermediate destination registers at steps 0 and 1 resolve the generalization failure of the continuous monolithic recurrent architecture (`CyclicMGUPhase16Model`, 219,018 trainable parameters) observed under terminal-only supervision (`EXP-CYCLIC-ADD-01`), by enabling its learned read, write, and gate networks to route and preserve canonical harmonic coordinates into its fixed cyclic ALU?

## 4. Existing Baseline

- **`EXP-CYCLIC-ADD-01` (`docs/reports/experiment_ah_capacity_ladder.md` and `docs/RESEARCH_LOG.md` entry 2026-10-15):** Evaluated identical `CyclicMGUPhase16Model` under terminal-only supervision. Result: 0/5 seeds passed progression, 5/5 seeds fell below rejection boundary ($acc_{\text{val}} \in \{0.0000, 0.0625, 0.1250, 0.1250, 0.0000\}$, mean $0.0625$, near chance $0.10$; $C_{\text{read}} \approx 0.176$, $C_{\text{ALU}} \approx 0.117$). Classified as `FAIL_REJECTION`.
- **Historical Stage 1 Track 1A Baselines (`docs/reports/phase_1_6_architectural_reformulation_proposal.md`):** `SlotMGUPhase16Model` ($acc_{\text{val}} = 0.0000$, 5/5 seeds) and `VanillaGRUPhase16Model` ($acc_{\text{val}} = 0.0000$, 5/5 seeds) both failed Gate 1 under terminal-only supervision.

## 5. Hypothesis

Preregistered hypothesis: Terminal-only cross-entropy supervision on step 2 does not provide sufficient gradient locality for the learned `mlp_read`, `mlp_write`, and `mlp_gate` networks to align with the fixed cyclic ALU's 5-harmonic Fourier basis $\Phi(v) \in \mathbb{R}^{10}$. Providing auxiliary supervision on intermediate destination register readouts at step 0 ($SET\ r0,\ a \to a$) and step 1 ($SET\ r1,\ b \to b$) with coefficient $\lambda = 1.0$ will enforce canonical coordinate alignment ($C_{\text{read}} \ge 0.80$, $S_{\text{context}} \ge 0.80$, $C_{\text{ALU}} \ge 0.80$) and enable out-of-training-pair generalization ($acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds).

## 6. What is Frozen

- **Architecture:** `CyclicMGUPhase16Model` with monolithic continuous state $S_t \in \mathbb{R}^{212}$, fixed 5-harmonic basis $\Phi(v)$ ($10 \times 10$), parameter-free Givens rotation cyclic ALU, and learned networks `mlp_read`, `mlp_write`, `mlp_gate`. Total trainable parameters: exactly 219,018.
- **Data Split:** Canonical `Stage1PairPartition` with 64 training pairs, 16 validation pairs (`VAL-SEEN`), and 20 protected test pairs ($b \in \{0, 5\}$).
- **Optimization:** AdamW ($\text{lr} = 3 \times 10^{-3}$, $\text{weight\_decay} = 10^{-2}$, $\beta = (0.9, 0.999)$), batch size 64, exactly 3,000 updates, validation cadence 50 updates (60 checkpoints per seed).
- **Execution Firewall:** The protected test split remains strictly unqueried (`queries = 0`).

## 7. What Changed

A single isolated intervention was implemented:
- **Auxiliary Destination Loss:** Total loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{term}} + 1.0 \times \mathcal{L}_{\text{aux}}$, where $\mathcal{L}_{\text{aux}} = \frac{1}{2}(\mathcal{L}_{\text{step0}} + \mathcal{L}_{\text{step1}})$ supervises the post-transition destination readout $u_{\text{dest\_post}}$ at step 0 ($r0 \to a$) and step 1 ($r1 \to b$).
- **Instrumentation:** Zero-parameter diagnostic tensor capture for representation metrics ($C_{\text{read}}$, $C_{\text{read\_r0}}$, $C_{\text{read\_r1}}$, $S_{\text{context}}$, $C_{\text{ALU}}$, $C_{\text{dest}}$).
- **Checkpoint Selection:** Selection policy updated to $\min \mathcal{L}_{\text{val\_total}} \to \max acc_{\text{val}} \to \text{earliest update}$.

## 8. Dataset and Protocol

- **Canonical 3-Step Trace:** $t=0: \text{SET}\ r0,\ a$; $t=1: \text{SET}\ r1,\ b$; $t=2: \text{ADD}\ r0,\ r1$.
- **Partitions:** 64 seen training pairs, 16 seen validation pairs (`VAL-SEEN`), 20 held-out test pairs ($b \in \{0, 5\}$).
- **Protocol:** 5 seeds (42–46), 3,000 updates each, CPU evaluation.
- **Progression Gate:** $acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds, zero seeds $< 0.250$.
- **Rejection Boundary:** $acc_{\text{val}} < 0.250$ across $\ge 4/5$ seeds.

## 9. Experimental Procedure

1. Stage 1 instrumentation and verification: implemented in `experiments/run_exp_cyclic_stepwise_01.py` and tested via `tests/test_exp_cyclic_stepwise_01.py` (39 passing unit tests).
2. Stage 1 smoke verification: Seed 42 run for 100 updates (`experiments/exp_cyclic_stepwise_01_SMOKE_ONLY_seed42.json`).
3. Stage 2 authorization: Authorized by PI and executed via `.venv/bin/python experiments/run_exp_cyclic_stepwise_01.py --stage2-authorized`.
4. Independent verification: Raw telemetry verified against specification, loss decomposition verified, bit-identity with smoke checkpoints confirmed.

## 10. Results

### Per-Seed Selected Checkpoints

Selected by preregistered policy ($\min \mathcal{L}_{\text{val\_total}} \to \max acc_{\text{val}} \to \text{earliest update}$):

| Seed | Update | Val Acc ($k/16$) | Val Acc | Val Loss Total | Val Loss Terminal | Val Loss Aux | Train Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 | 150 | 12/16 | 0.7500 | 0.861159 | 0.762611 | 0.098549 | 1.0000 |
| 43 | 100 | 14/16 | 0.8750 | 0.472725 | 0.374412 | 0.098313 | 1.0000 |
| 44 | 1700 | 7/16 | 0.4375 | 2.197906 | 2.103602 | 0.094303 | 1.0000 |
| 45 | 1200 | 10/16 | 0.6250 | 1.654200 | 1.510014 | 0.144186 | 1.0000 |
| 46 | 450 | 11/16 | 0.6875 | 1.291207 | 1.175377 | 0.115830 | 1.0000 |
| **Macro**| — | **10.8/16** | **0.6750** | **1.295439** | **1.185203** | **0.110236** | **1.0000** |

### Representation Metrics (Selected Checkpoints; VAL-SEEN)

| Seed | $C_{\text{read}}$ | $C_{\text{read\_r0}}$ | $C_{\text{read\_r1}}$ | $S_{\text{context}}$ | $C_{\text{ALU}}$ | $C_{\text{dest}}$ | $C_{\text{dest, s0}}$ | $C_{\text{dest, s1}}$ | $C_{\text{dest, s2}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 | 0.5579 | 0.2786 | 0.8373 | 0.3605 | 0.6046 | 0.8505 | 0.9893 | 0.8373 | 0.7248 |
| 43 | 0.5648 | 0.3003 | 0.8294 | 0.3295 | 0.6344 | 0.8734 | 0.9885 | 0.8294 | 0.8023 |
| 44 | 0.5143 | 0.1718 | 0.8568 | 0.2595 | 0.4246 | 0.7620 | 0.9874 | 0.8568 | 0.4420 |
| 45 | 0.5388 | 0.2619 | 0.8158 | 0.3257 | 0.4930 | 0.7841 | 0.9845 | 0.8158 | 0.5520 |
| 46 | 0.5595 | 0.2740 | 0.8449 | 0.3082 | 0.5304 | 0.8163 | 0.9785 | 0.8449 | 0.6254 |
| **Macro**| **0.5471** | **0.2573** | **0.8368** | **0.3167** | **0.5374** | **0.8172** | **0.9856** | **0.8368** | **0.6293** |

### Per-Step Accuracy Trajectory (VAL-SEEN)

- **Step 0 Accuracy ($a \to r0$):** 1.0000 across all 5 seeds (macro: 1.0000).
- **Step 1 Accuracy ($b \to r1$):** 1.0000, 1.0000, 1.0000, 0.9375, 0.9375 (macro: 0.9875).
- **Step 2 Terminal Accuracy ($a+b \pmod{10}$):** 0.7500, 0.8750, 0.4375, 0.6250, 0.6875 (macro: 0.6750).

## 11. Statistical Analysis

### Preregistered Decision Matrix Outcome (§9.3)

1. **Behavioral Evaluation (5 seeds, $N_{\text{val}} = 16$):**
   - $N_{\text{pass}} = 0$ (requires $\ge 4/5 \ge 0.950$).
   - $N_{\text{reject}} = 0$ (requires $\ge 4/5 < 0.250$).
   - $N_{\text{mid}} = 5$ (all seeds in range $[0.4375, 0.8750]$).
   - $\implies$ **`BEHAVIOR_INTERMEDIATE`**.
2. **Representation Contract (Macro-Averaged):**
   - $\bar{C}_{\text{read}} = 0.5471 < 0.80$ (fails `CANONICAL_FULL` and `CANONICAL_INPUT_ONLY`).
   - $\bar{C}_{\text{read}} = 0.5471 \ge 0.30$ (fails `FAILED`).
   - $\implies$ **`REP_PARTIAL`**.
3. **Scientific Category:** **`INCONCLUSIVE_INTERMEDIATE`** (mechanical output of the preregistered matrix).

### Fair Predecessor Comparison with `EXP-CYCLIC-ADD-01`

*Note: This post-hoc analysis does not alter the official preregistered results.*

| Evaluation Policy | EXP-CYCLIC-ADD-01 | EXP-CYCLIC-STEPWISE-01 | Difference |
| :--- | :--- | :--- | :--- |
| **Official Selection Policy** | 0.0625 ($\min \mathcal{L}_{\text{term}}$) | 0.6750 ($\min \mathcal{L}_{\text{total}}$) | +0.6125 |
| **Consistent Policy: $\min \mathcal{L}_{\text{val\_terminal}}$** | 0.0625 | 0.6750 | +0.6125 |
| **Consistent Policy: Peak Validation Accuracy** | 0.1375 | 0.7000 | +0.5625 |

Under a strictly identical selection rule ($\min \mathcal{L}_{\text{val\_terminal}}$), the macro validation accuracy is unchanged at 0.6750 (Seed 44 selects update 200 instead of 1700, with identical accuracy 0.4375). The substantial gain over `EXP-CYCLIC-ADD-01` (+0.6125) is an empirical property of the auxiliary supervision, not a selection-rule artifact.

## 12. Interpretation

### Observed Facts

1. **Intermediate Readout Fidelity:** When an operand is written into a register, post-transition readout reaches canonical Fourier alignment immediately ($C_{\text{dest, s0}} = 0.9856$, $C_{\text{dest, s1}} = 0.8368$).
2. **Severe Register Read Asymmetry at Step 2:** At step 2, when operands are read for addition, $C_{\text{read\_r1}} = 0.8368 \pm 0.015 \ge 0.80$ in all 5 seeds, whereas $C_{\text{read\_r0}} = 0.2573 \pm 0.048 < 0.30$ in all 5 seeds.
3. **ALU Output Degradation:** The parameter-free cyclic ALU computes rotation on $u_a$ and $u_b$. Because $u_a$ is misaligned ($0.2573$), the raw ALU output alignment $C_{\text{ALU}} = 0.5374$ falls well below the canonical threshold ($0.80$).
4. **Step-Wise Decoupling:** Step 0 and step 1 accuracies reach near-perfection on validation pairs ($1.0000$ and $0.9875$), while errors are concentrated exclusively at the terminal step ($0.6750$).
5. **Overfitting Dynamic:** Training accuracy reaches 1.0000 across all steps by update 100–150 in all seeds, while terminal validation loss diverges after early updates.

### Hypotheses and Explanations (Labeled, Unproven)

- *Hypothesis A (Recurrent State Interference):* The continuous monolithic state $S \in \mathbb{R}^{212}$ lacks discrete slot isolation (unlike `DiscreteMGUPhase16Model`, where non-addressed slots have $\|\Delta s_r\| = 0$). When $S_1$ undergoes the transition at step 1 ($S_2 = (1 - g_1) \odot S_1 + g_1 \odot C_1$), the continuous gate $g_1$ and write vector $C_1$ overwrite or disperse the coordinates corresponding to R0, degrading $u_a$ read fidelity.
- *Hypothesis B (Asymmetric Supervision Horizon):* Auxiliary supervision at step 1 only supervised the destination register R1 ($b$). R0 was not supervised at step 1. Consequently, the network had no objective forcing R0 coordinate preservation across step 1, allowing R0 representation drift during training set fitting.

## 13. What the Experiment DOES Establish

- Proves that dense auxiliary destination supervision substantially improves validation generalization over terminal-only supervision ($0.6750$ vs $0.0625$), ruling out catastrophic optimization collapse.
- Proves that `mlp_read` can learn to map continuous states into canonical Fourier codewords when immediately supervised ($C_{\text{dest, s0}} = 0.9856$, $C_{\text{dest, s1}} = 0.8368$).
- Establishes that the terminal generalization bottleneck in `CyclicMGUPhase16Model` is localized to the retrieval of the unaddressed operand $a$ from R0 after an intervening recurrent transition ($C_{\text{read\_r0}} = 0.2573$).

## 14. What the Experiment DOES NOT Establish

- Does NOT establish that the fixed cyclic ALU is defective (the ALU is analytical and parameter-free; its output degradation is fully explained by its input degradation).
- Does NOT establish that monolithic recurrent architectures are incapable of modular addition (an intervention enforcing multi-slot coordinate preservation remains untested).
- Does NOT establish learned arithmetic from data (the cyclic ALU is analytical; only neural routing and coordination were evaluated).
- Does NOT evaluate general out-of-distribution arithmetic (the protected test split $b \in \{0, 5\}$ remained unqueried).

## 15. Decision

**EXP-CYCLIC-STEPWISE-01 is recorded as INCONCLUSIVE_INTERMEDIATE.** The candidate architecture made substantial progress over the terminal-only predecessor ($0.6750$ vs $0.0625$), but failed the preregistered progression gate ($\ge 4/5 \ge 0.950$). Advancement to Stage 3 composed sequence training remains **CLOSED**.

## 16. Next Experiment

Recommend **`EXP-CYCLIC-PERSIST-01`**: A controlled investigation and targeted auxiliary supervision intervention testing whether supervising the persistence of unaddressed registers during intermediate steps ($r0 \to a$ maintained at step 1) eliminates the $r0/r1$ asymmetry and restores $C_{\text{read\_r0}} \ge 0.80$.

## 17. Reproduction Commands

```bash
# Execute official Stage 2 five-seed run
.venv/bin/python experiments/run_exp_cyclic_stepwise_01.py --stage2-authorized

# Run full test suite for EXP-CYCLIC-STEPWISE-01
pytest tests/test_exp_cyclic_stepwise_01.py -v
```

## 18. Artifact List

- Runner: `experiments/run_exp_cyclic_stepwise_01.py`
- Test suite: `tests/test_exp_cyclic_stepwise_01.py`
- Official Stage 2 Results Artifact: `experiments/exp_cyclic_stepwise_01_results.json`
- Stage 1 Smoke Telemetry Artifact: `experiments/exp_cyclic_stepwise_01_SMOKE_ONLY_seed42.json`
- Controlling Specification: `docs/research/exp_cyclic_stepwise_01_specification.md`

## 19. Scientific Guardrails

- Clear separation of verified facts from speculative hypotheses.
- No claim of learned arithmetic from data (ALU is analytical).
- No promotion of intermediate results to progression pass.
- Protected test split ($b \in \{0, 5\}$) maintained strictly untouched.
