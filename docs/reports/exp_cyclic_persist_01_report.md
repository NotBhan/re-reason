# EXP-CYCLIC-PERSIST-01: Multi-Register Persistence Supervision for Monolithic Cyclic-MGU

## 1. Date

2026-10-16

## 2. Status

Completed. Official 5-seed Stage 2 run executed under explicit PI authorization with `--stage2-authorized`. Mechanical outcome under preregistered decision matrix: **`BEHAVIORAL_PASS_CANONICAL_FULL`** (behavioral: `BEHAVIOR_PROGRESSION_PASS`, representation: `REP_CANONICAL_FULL`). Macro validation accuracy on `VAL-SEEN`: **0.9875** (15.8/16 pairs; 4/5 seeds at 1.0000, 1 seed at 0.9375). R0/R1 read asymmetry resolved ($\bar{C}_{\text{read}}(r0) = 0.8879$, $\bar{C}_{\text{read}}(r1) = 0.9303$, both $\ge 0.80$). The protected TEST split ($b \in \{0, 5\}$) remained strictly firewalled (`queries = 0`).

## 3. Research Question

Does auxiliary supervision of unaddressed-register persistence during the intermediate state transition at step 1 (requiring register $r0$ to retain operand $a$ during the $S_1 \to S_2$ transition under instruction `SET r1 b`) eliminate the severe $r0/r1$ read asymmetry observed in `EXP-CYCLIC-STEPWISE-01`, restore canonical harmonic alignment into the fixed parameter-free cyclic ALU ($\bar{C}_{\text{read}} \ge 0.80$, $\bar{S}_{\text{context}} \ge 0.80$, $\bar{C}_{\text{ALU}} \ge 0.80$), and enable `CyclicMGUPhase16Model` (219,018 trainable parameters) to pass the preregistered validation progression gate ($acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds) without altering model architecture?

## 4. Existing Baseline

- **`EXP-CYCLIC-ADD-01` (`docs/reports/experiment_ah_capacity_ladder.md` and `docs/RESEARCH_LOG.md` entry 2026-10-15):** Evaluated `CyclicMGUPhase16Model` under terminal-only cross-entropy supervision. Outcome: `FAIL_REJECTION` (0/5 seeds passed progression, 5/5 seeds in rejection zone $< 0.250$, macro validation accuracy $0.0625$, near chance $0.10$; $C_{\text{read}} \approx 0.176$, $C_{\text{ALU}} \approx 0.117$).
- **`EXP-CYCLIC-STEPWISE-01` (`docs/reports/exp_cyclic_stepwise_01_report.md`):** Added auxiliary supervision of intermediate destination registers at step 0 ($r0 \to a$) and step 1 ($r1 \to b$) with coefficient $\lambda = 1.0$. Outcome: `INCONCLUSIVE_INTERMEDIATE` (macro validation accuracy lifted to $0.6750$, but 0/5 seeds reached progression $\ge 0.950$). The failure was localized to a severe register read asymmetry at terminal step 2: $C_{\text{read}}(r1) = 0.8368 \ge 0.80$, whereas $C_{\text{read}}(r0) = 0.2573 < 0.30$.
- **Historical Stage 1 Track 1A Baselines (`docs/reports/phase_1_6_architectural_reformulation_proposal.md`):** `SlotMGUPhase16Model` ($acc_{\text{val}} = 0.0000$, 5/5 seeds) and `VanillaGRUPhase16Model` ($acc_{\text{val}} = 0.0000$, 5/5 seeds) both failed Gate 1 under terminal-only supervision.

## 5. Hypothesis

Preregistered hypothesis $H_1$ (`docs/research/exp_cyclic_persist_01_specification.md` §2.4): Insufficient persistence supervision is the primary cause of the $r0/r1$ read asymmetry in `EXP-CYCLIC-STEPWISE-01`, because $r0$ received zero direct gradient signal during the $S_1 \to S_2$ transition (`SET r1 b`). Supervising the persistence of unaddressed register $r0$ at step 1 using the existing shared `mlp_read` network decoded against the fixed prototype basis $\Phi \in \mathbb{R}^{10 \times 10}$ will:
1. Eliminate the read asymmetry at step 2 ($\bar{C}_{\text{read}}(r0) \ge 0.80$ and $\bar{C}_{\text{read}}(r1) \ge 0.80$);
2. Achieve full canonical representation alignment ($\bar{C}_{\text{read}} \ge 0.80$, $\bar{S}_{\text{context}} \ge 0.80$, $\bar{C}_{\text{ALU}} \ge 0.80$);
3. Enable sequence validation accuracy to pass the progression gate ($acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds with 0 seeds $< 0.250$).

Null hypothesis $H_0$: Monolithic continuous state routing in `CyclicMGUPhase16Model` is structurally incapable of preserving independent coordinates across non-linear state updates without discrete slot isolation, resulting in continued progression failure ($acc_{\text{val}} < 0.950$) or a trade-off between destination writing and persistence.

## 6. What is Frozen

- **Architecture:** `CyclicMGUPhase16Model` (`src/re_reason/core/register_machine_models.py`) with monolithic continuous state $S_t \in \mathbb{R}^{212}$, fixed 5-harmonic basis $\Phi(v)$ ($10 \times 10$), parameter-free Givens rotation cyclic ALU, and learned networks `mlp_read` (29,066 params), `mlp_write` (52,152 params), and `mlp_gate` (137,800 params). Total trainable parameters: exactly **219,018** (invariant before and after training).
- **Zero Trainable Readout Overhead:** The persistence query uses the exact existing `mlp_read` network with shared weights, decoded against the identical fixed basis $\Phi$. No new trainable projection, layer, or head is introduced.
- **Dataset Partitions:** Canonical `Stage1PairPartition` with 64 training pairs (`TRAIN-FIT`), 16 validation pairs (`VAL-SEEN`), and 20 protected test pairs ($b \in \{0, 5\}$).
- **Optimization:** AdamW ($\text{lr} = 3 \times 10^{-3}$, $\text{weight\_decay} = 10^{-2}$, $\beta = (0.9, 0.999)$), batch size 64, exactly 3,000 updates, validation cadence 50 updates (60 checkpoints per seed).
- **Evaluation Firewall:** The protected TEST split ($b \in \{0, 5\}$) remains strictly unqueried (`queries = 0`).

## 7. What Changed

A single isolated intervention was added to the training objective:
- **Persistence Supervision at Step 1:** During the $S_1 \to S_2$ transition (`SET r1 b`), unaddressed register $r0$ is queried via $[S_2, e_{r0}] \in \mathbb{R}^{216}$ through `mlp_read` and supervised against target $a$:
  $$\mathcal{L}_{\text{persist}} = \text{CE}(\text{logits}_{\text{persist}, 1}, a)$$
  Note that vector $u_{\text{persist}, 1} = \operatorname{project}(\text{mlp\_read}([S_2, e_{r0}]))$ is mathematically identical to operand read vector $u_a$ extracted at the start of step 2 before ALU injection.
- **Preregistered Objective Function:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{terminal}} + \lambda_{\text{dest}} \cdot \mathcal{L}_{\text{dest}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{persist}}$$
  With PI-approved coefficients $\lambda_{\text{dest}} = 1.0$ and $\lambda_{\text{persist}} = 0.5$. Because $\mathcal{L}_{\text{dest}} = 0.5 \cdot [\text{CE}(\text{dest}_0, a) + \text{CE}(\text{dest}_1, b)]$, every intermediate register assertion ($r0$ at step 0, $r1$ at step 1, $r0$ at step 1) carries an identical effective weight of $0.5$.
- **Checkpoint Selection Policy:** Preregistered policy identical to specification: minimum total validation loss ($\min \mathcal{L}_{\text{val\_total}}$), ties broken by maximum validation accuracy ($\max acc_{\text{val}}$), then earliest update step.

## 8. Dataset and Protocol

- **Canonical 3-Step Trace:** $t=0: \text{SET}\ r0,\ a \implies S_0 \to S_1$; $t=1: \text{SET}\ r1,\ b \implies S_1 \to S_2$; $t=2: \text{ADD}\ r0,\ r1 \implies S_2 \to S_3$, terminal target $(a + b) \bmod 10$.
- **Partitions:** 64 seen training pairs, 16 seen validation pairs (`VAL-SEEN`), 20 held-out test pairs ($b \in \{0, 5\}$).
- **Protocol:** 5 seeds (42–46), 3,000 updates each, CPU evaluation.
- **Progression Gate:** $acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds, zero seeds $< 0.250$ (on `VAL-SEEN`, $N_{\text{val}} = 16$).
- **Rejection Boundary:** $acc_{\text{val}} < 0.250$ across $\ge 4/5$ seeds.

## 9. Experimental Procedure

1. **Stage 1 (Implementation, Focused Tests, Smoke Test):** Implemented in `experiments/run_exp_cyclic_persist_01.py` with 26 focused tests in `tests/test_exp_cyclic_persist_01.py`. Seed 42 ran for 100 updates (`experiments/exp_cyclic_persist_01_SMOKE_ONLY_seed42.json`). All 5 mandatory smoke checks passed (loss decomposition identity, gradient flow to unaddressed coordinates, parameter count invariance 219,018, diagnostic bounds, and TEST firewall `queries = 0`). Audited and documented.
2. **Stage 2 (Official Five-Seed Training):** Authorized by PI on 2026-10-16. Executed via `.venv/bin/python experiments/run_exp_cyclic_persist_01.py --stage2-authorized`. All five seeds (42–46) completed exactly 3,000 updates with 60 checkpoints per seed.
3. **Independent Artifact Audit:** Conducted independently by Antigravity. Recomputed loss decomposition across all 300 checkpoints, verified parameter count invariance, re-derived checkpoint selection from raw telemetry, and confirmed strict firewall integrity.

## 10. Results

### Per-Seed Selected Checkpoints

Selected by preregistered policy ($\min \mathcal{L}_{\text{val\_total}} \to \max acc_{\text{val}} \to \text{earliest update}$):

| Seed | Update | Val Acc ($k/16$) | Val Acc | Val Loss Total | Val Loss Terminal | Val Loss Dest | Val Loss Persist | Train Loss Total | Train Acc | Final Val Acc | Runtime (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 | 950 | 16/16 | 1.0000 | 0.130573 | 0.046342 | 0.046334 | 0.075795 | 0.097305 | 1.0000 | 1.0000 | 32.56 |
| 43 | 550 | 16/16 | 1.0000 | 0.271447 | 0.116045 | 0.060323 | 0.190159 | 0.097168 | 1.0000 | 0.8750 | 30.32 |
| 44 | 500 | 16/16 | 1.0000 | 0.197333 | 0.051828 | 0.063181 | 0.164648 | 0.097369 | 1.0000 | 1.0000 | 29.60 |
| 45 | 400 | 15/16 | 0.9375 | 0.384802 | 0.258801 | 0.047650 | 0.156703 | 0.098254 | 1.0000 | 0.8750 | 33.38 |
| 46 | 500 | 16/16 | 1.0000 | 0.210393 | 0.064120 | 0.095144 | 0.102257 | 0.098455 | 1.0000 | 0.9375 | 54.06 |
| **Macro** | — | **15.8/16** | **0.9875** | **0.238910** | **0.107427** | **0.062526** | **0.137912** | **0.097710** | **1.0000** | **0.9375** | **179.91 (total)** |

### Representation Metrics (Selected Checkpoints; VAL-SEEN)

| Seed | $C_{\text{read}}$ | $C_{\text{read}}(r0)$ | $C_{\text{read}}(r1)$ | $C_{\text{persist}}(r0)$ | $S_{\text{context}}$ | $C_{\text{ALU}}$ | $C_{\text{dest}}$ | $C_{\text{dest, s0}}$ | $C_{\text{dest, s1}}$ | $C_{\text{dest, s2}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 | 0.9493 | 0.9348 | 0.9638 | 0.9348 | 0.9044 | 0.9053 | 0.9812 | 0.9994 | 0.9638 | 0.9804 |
| 43 | 0.8859 | 0.8417 | 0.9302 | 0.8417 | 0.8346 | 0.8263 | 0.9473 | 0.9998 | 0.9302 | 0.9118 |
| 44 | 0.8960 | 0.8727 | 0.9193 | 0.8727 | 0.8246 | 0.8294 | 0.9628 | 0.9997 | 0.9193 | 0.9693 |
| 45 | 0.9166 | 0.8772 | 0.9561 | 0.8772 | 0.8835 | 0.8449 | 0.9508 | 0.9996 | 0.9561 | 0.8967 |
| 46 | 0.8976 | 0.9131 | 0.8822 | 0.9131 | 0.8449 | 0.8419 | 0.9459 | 0.9994 | 0.8822 | 0.9559 |
| **Macro** | **0.9091** | **0.8879** | **0.9303** | **0.8879** | **0.8584** | **0.8496** | **0.9576** | **0.9996** | **0.9303** | **0.9428** |

### Per-Step Accuracy Trajectory (VAL-SEEN)

- **Step 0 Destination Accuracy ($a \to r0$):** 1.0000 across all 5 seeds (macro: 1.0000).
- **Step 1 Destination Accuracy ($b \to r1$):** 1.0000 across all 5 seeds (macro: 1.0000).
- **Step 1 Persistence Accuracy ($a$ preserved in $r0$):** 1.0000 across all 5 seeds (macro: 1.0000).
- **Step 2 Terminal Accuracy ($(a + b) \bmod 10$):** 1.0000, 1.0000, 1.0000, 0.9375, 1.0000 (macro: 0.9875).

## 11. Statistical Analysis

### Preregistered Decision Matrix Outcome (§9.3)

1. **Behavioral Evaluation (5 seeds, $N_{\text{val}} = 16$):**
   - Seed 42: $k = 16$ ($acc = 1.0000$) $\implies$ Pass
   - Seed 43: $k = 16$ ($acc = 1.0000$) $\implies$ Pass
   - Seed 44: $k = 16$ ($acc = 1.0000$) $\implies$ Pass
   - Seed 45: $k = 15$ ($acc = 0.9375$) $\implies$ Intermediate
   - Seed 46: $k = 16$ ($acc = 1.0000$) $\implies$ Pass
   - Aggregate counts: $N_{\text{pass}} = 4$, $N_{\text{mid}} = 1$, $N_{\text{reject}} = 0$.
   - Gate criterion ($N_{\text{pass}} \ge 4 \land N_{\text{reject}} = 0$) is satisfied.
   - $\implies$ **`BEHAVIOR_PROGRESSION_PASS`**.
2. **Representation Contract (Macro-Averaged):**
   - $\bar{C}_{\text{read}} = 0.9091 \ge 0.80$
   - $\bar{S}_{\text{context}} = 0.8584 \ge 0.80$
   - $\bar{C}_{\text{ALU}} = 0.8496 \ge 0.80$
   - All three criteria $\ge 0.80$ satisfied simultaneously.
   - $\implies$ **`REP_CANONICAL_FULL`**.
3. **Asymmetry Resolution Criterion (§9.2):**
   - $\bar{C}_{\text{read}}(r0) = 0.8879 \ge 0.80$
   - $\bar{C}_{\text{read}}(r1) = 0.9303 \ge 0.80$
   - Both register reads satisfy $\ge 0.80$. The severe asymmetry from `EXP-CYCLIC-STEPWISE-01` ($0.2573$ vs $0.8368$) is **fully resolved**.
4. **Scientific Category:** **`BEHAVIORAL_PASS_CANONICAL_FULL`** (mechanical output of the preregistered 3 × 4 matrix).

### Comparison with Predecessor Cyclic-MGU Experiments

| Metric | EXP-CYCLIC-ADD-01 (terminal only) | EXP-CYCLIC-STEPWISE-01 (dest only) | EXP-CYCLIC-PERSIST-01 (dest + persist) | Difference vs STEPWISE | Difference vs ADD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Objective** | $\mathcal{L}_{\text{term}}$ | $\mathcal{L}_{\text{term}} + 1.0\mathcal{L}_{\text{dest}}$ | $\mathcal{L}_{\text{term}} + 1.0\mathcal{L}_{\text{dest}} + 0.5\mathcal{L}_{\text{persist}}$ | $+0.5\mathcal{L}_{\text{persist}}$ | $+1.0\mathcal{L}_{\text{dest}} + 0.5\mathcal{L}_{\text{persist}}$ |
| **Parameters** | 219,018 | 219,018 | 219,018 (invariant) | 0 | 0 |
| **Macro Val Acc** | 0.0625 | 0.6750 | **0.9875** | **+0.3125** | **+0.9250** |
| **Pass Seeds ($k=16$)** | 0/5 | 0/5 | **4/5** | +4 | +4 |
| **Reject Seeds ($k\le 3$)**| 5/5 | 0/5 | **0/5** | 0 | −5 |
| **Behavioral Gate** | `BEHAVIOR_REJECTION` | `BEHAVIOR_INTERMEDIATE` | **`BEHAVIOR_PROGRESSION_PASS`** | Advanced | Advanced |
| **Macro $C_{\text{read}}$** | ~0.176 | 0.5471 | **0.9091** | **+0.3620** | +0.733 |
| **$C_{\text{read}}(r0)$** | ~0.17 | 0.2573 | **0.8879** | **+0.6306** | +0.718 |
| **$C_{\text{read}}(r1)$** | ~0.18 | 0.8368 | **0.9303** | **+0.0935** | +0.750 |
| **Macro $S_{\text{context}}$** | ~0.15 | 0.3167 | **0.8584** | **+0.5417** | +0.708 |
| **Macro $C_{\text{ALU}}$** | ~0.117 | 0.5374 | **0.8496** | **+0.3122** | +0.733 |
| **Macro $C_{\text{dest}}$** | — | 0.8172 | **0.9576** | **+0.1404** | — |
| **Representation State** | `REP_FAILED` | `REP_PARTIAL` | **`REP_CANONICAL_FULL`** | Advanced | Advanced |
| **Scientific Category** | `FAIL_REJECTION` | `INCONCLUSIVE_INTERMEDIATE` | **`BEHAVIORAL_PASS_CANONICAL_FULL`** | Resolved | Resolved |

## 12. Interpretation

### Observed Facts

1. **R0 Retention Fully Restored:** Auxiliary supervision of unaddressed persistence during the $S_1 \to S_2$ transition directly restored R0 read fidelity at step 2 from $0.2573$ to $0.8879$ ($+0.6306$ gain).
2. **Canonical Basis Alignment Across All Stages:** With both input operands canonically aligned into the cyclic ALU ($C_{\text{read}}(r0) = 0.8879, C_{\text{read}}(r1) = 0.9303$), the analytical Givens rotation ALU produced canonically aligned sum output ($C_{\text{ALU}} = 0.8496 \ge 0.80$), resolving the downstream arithmetic degradation observed in `EXP-CYCLIC-STEPWISE-01` ($0.5374$).
3. **Context Independence:** Cross-context read consistency rose from $0.3167$ to $0.8584$, confirming that representations of digit operands have become stable across roles and sequence contexts.
4. **Progression Gate Passed:** 4 of 5 seeds achieved perfect validation accuracy ($16/16$), and 1 seed achieved $15/16$ ($0.9375$), yielding a macro accuracy of $0.9875$ and satisfying the preregistered progression gate.
5. **Architectural Sufficiency:** The identical continuous monolithic architecture (`CyclicMGUPhase16Model`, 219,018 parameters) that failed under terminal supervision ($0.0625$) and stalled under destination-only supervision ($0.6750$) reached near-perfect validation generalization ($0.9875$) under joint destination and persistence supervision without changing a single parameter or architectural component.

### Hypotheses and Explanations (Labeled, Plausible Mechanistic Reading)

- *Hypothesis $H_1$ Supported:* Insufficient persistence supervision was indeed the primary bottleneck in `EXP-CYCLIC-STEPWISE-01`. When the unaddressed register received explicit gradient signal during the intervening state update, `mlp_gate` learned to route and preserve the $r0$ coordinate subspace in continuous state $S_2$ rather than allowing it to be overwritten by the candidate vector $C_1$.
- *Rejection of Null Hypothesis $H_0$:* Monolithic continuous state representations do **not** inherently require physical slot partitioning (such as discrete slot indexing) to maintain coordinate independence during a 3-step sequence; continuous gating can learn coordinate preservation if appropriate multi-register supervision is supplied.
- *Remaining Causal Question:* Because this experiment trained with joint destination and persistence supervision ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{term}} + 1.0\mathcal{L}_{\text{dest}} + 0.5\mathcal{L}_{\text{persist}}$), the individual contribution of persistence supervision alone (without intermediate destination supervision) remains an untested question, formalized as the deferred ablation arm in specification §11.

## 13. What the Experiment DOES Establish

- Proves that multi-register persistence supervision eliminates the $r0/r1$ read asymmetry in `CyclicMGUPhase16Model`, raising $\bar{C}_{\text{read}}(r0)$ from $0.2573$ to $0.8879$.
- Proves that `CyclicMGUPhase16Model` can achieve full canonical harmonic representation alignment ($\bar{C}_{\text{read}} \ge 0.80$, $\bar{S}_{\text{context}} \ge 0.80$, $\bar{C}_{\text{ALU}} \ge 0.80$) and pass the preregistered validation progression gate ($0.9875$ macro accuracy; 4/5 seeds at 1.0000).
- Confirms that the failure of `EXP-CYCLIC-ADD-01` and `EXP-CYCLIC-STEPWISE-01` was an objective and supervision horizon limitation, not an intrinsic expressivity or routing failure of the continuous MGU architecture.

## 14. What the Experiment DOES NOT Establish

- Does NOT evaluate or establish generalization to out-of-distribution digits. The protected TEST split ($b \in \{0, 5\}$) was strictly firewalled (`queries = 0`). The reported result concerns sequence-disjoint combinations of seen training digits (`VAL-SEEN`) only.
- Does NOT establish learned arithmetic from data. The cyclic Givens rotation ALU is fixed and analytical; this experiment validates neural routing, gating, and coordinate preservation into that component, not the emergence of arithmetic circuits from scratch.
- Does NOT establish that persistence supervision alone is sufficient without destination supervision (the persistence-only ablation arm remains unexecuted and deferred).
- Does NOT evaluate sequence lengths beyond 3 steps or composed sequence tasks (Stage 3 remains closed).

## 15. Decision

**EXP-CYCLIC-PERSIST-01 is recorded as BEHAVIORAL_PASS_CANONICAL_FULL.** The candidate architecture passed both the behavioral progression gate ($N_{\text{pass}} = 4, N_{\text{mid}} = 1, N_{\text{reject}} = 0$) and the representation contract ($\bar{C}_{\text{read}} = 0.9091$, $\bar{S}_{\text{context}} = 0.8584$, $\bar{C}_{\text{ALU}} = 0.8496$). The result validates Candidate 2 (`CyclicMGUPhase16Model`) under the Phase 1.6 research program. Advancing to subsequent stages (such as the persistence-only ablation, composed sequence training, or protected test evaluation) requires separate explicit PI authorization.

## 16. Next Experiment

Pending explicit PI direction, candidate next steps are:
1. **Persistence-Only Ablation Arm (`EXP-CYCLIC-PERSIST-ABL-01`):** Train with $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{terminal}} + \mathcal{L}_{\text{persist}}$ (omitting $\mathcal{L}_{\text{dest}}$) to determine whether persistence supervision alone drives coordinate preservation or whether joint destination supervision is strictly necessary (recorded as deferred in specification §11).
2. **Phase 1.6 Stage 2 / Stage 3 Formulation:** Formulate address-binding or composed sequence evaluation for `CyclicMGUPhase16Model`.
3. **Protected TEST Split Evaluation:** Subject to explicit PI authorization and strict single-evaluation governance.

## 17. Reproduction Commands

```bash
# Execute official Stage 2 five-seed experiment
.venv/bin/python experiments/run_exp_cyclic_persist_01.py --stage2-authorized

# Run full focused test suite (26 tests)
pytest tests/test_exp_cyclic_persist_01.py -v

# Run documentation governance test suite (13 tests)
pytest tests/test_documentation_governance.py -v
```

## 18. Artifact List

- Runner: `experiments/run_exp_cyclic_persist_01.py`
- Focused Test Suite: `tests/test_exp_cyclic_persist_01.py`
- Official Results Artifact: `experiments/exp_cyclic_persist_01_results.json`
- Stage 1 Smoke Telemetry Artifact: `experiments/exp_cyclic_persist_01_SMOKE_ONLY_seed42.json`
- Controlling Specification: `docs/research/exp_cyclic_persist_01_specification.md`

## 19. Scientific Guardrails

- Clear separation of verified empirical observations from explanatory hypotheses.
- No claim of learned arithmetic from data (the cyclic ALU is parameter-free and analytical).
- No claim of out-of-distribution generalization; `TEST-HELDOUT` ($b \in \{0, 5\}$) remains strictly firewalled (`queries = 0`).
- No retroactive modification of preregistered thresholds, gates, or decision rules.
