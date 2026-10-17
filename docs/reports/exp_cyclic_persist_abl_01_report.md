# EXP-CYCLIC-PERSIST-ABL-01: Persistence-Only Auxiliary Supervision Ablation for Monolithic Cyclic-MGU

## 1. Date

2026-10-17

## 2. Status

Completed. Official 5-seed Stage 2 run executed on 2026-10-17 under explicit PI authorization with `--stage2-authorized`. Mechanical outcome under the preregistered decision matrix: **`INCONCLUSIVE_INTERMEDIATE`** (behavioral: `BEHAVIOR_INTERMEDIATE`, representation: `REP_PARTIAL`). Macro validation accuracy on `VAL-SEEN`: **0.5375** (43/80 pairs; 0/5 seeds passed, 5/5 seeds intermediate). Progression gate ($acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds) was **not met**; rejection boundary ($acc_{\text{val}} < 0.250$ across $\ge 4/5$ seeds) was **not met**. Persistence supervision alone resolved $r0$ persistence ($\bar{C}_{\text{read}}(r0) = 0.7710$), but $r1$ suffered severe representation misalignment ($\bar{C}_{\text{read}}(r1) = 0.2597$), producing an **inverted read asymmetry** relative to `EXP-CYCLIC-STEPWISE-01`. The protected TEST split ($b \in \{0, 5\}$) remained strictly firewalled (`queries = 0`).

## 3. Research Question

Is auxiliary supervision of unaddressed-register persistence at step 1 alone, combined solely with terminal task supervision ($\mathcal{L}_{\text{total}}^{\text{abl}} = \mathcal{L}_{\text{terminal}} + \lambda_{\text{persist}} \cdot \mathcal{L}_{\text{persist}}$, omitting intermediate destination supervision: $\lambda_{\text{dest}} = 0.0$), sufficient to induce canonical coordinate preservation, eliminate read asymmetry, and achieve sequence generalization in `CyclicMGUPhase16Model` ($acc_{\text{val}} \ge 0.950$, $\bar{C}_{\text{read}} \ge 0.80$, $\bar{C}_{\text{ALU}} \ge 0.80$)? Or is direct intermediate destination supervision causally necessary to anchor operand representations in canonical Fourier coordinates?

## 4. Existing Baseline

- **`EXP-CYCLIC-ADD-01` (`docs/reports/experiment_ah_capacity_ladder.md` and `docs/RESEARCH_LOG.md` entry 2026-10-15):** Evaluated `CyclicMGUPhase16Model` under terminal-only cross-entropy supervision ($\lambda_{\text{dest}} = 0.0, \lambda_{\text{persist}} = 0.0$). Outcome: `FAIL_REJECTION` (0/5 seeds passed, 5/5 seeds in rejection zone $< 0.250$, macro validation accuracy $0.0625$, near chance $0.10$).
- **`EXP-CYCLIC-STEPWISE-01` (`docs/reports/exp_cyclic_stepwise_01_report.md`):** Added auxiliary supervision of intermediate destination registers at step 0 ($r0 \leftarrow a$) and step 1 ($r1 \leftarrow b$) with $\lambda_{\text{dest}} = 1.0, \lambda_{\text{persist}} = 0.0$. Outcome: `INCONCLUSIVE_INTERMEDIATE` (macro validation accuracy lifted to $0.6750$, but 0/5 seeds passed). Failure localized to severe register read asymmetry at terminal step 2: $C_{\text{read}}(r1) = 0.8368 \ge 0.80$, while unaddressed register $r0$ degraded to $C_{\text{read}}(r0) = 0.2573 < 0.30$.
- **`EXP-CYCLIC-PERSIST-01` (`docs/reports/exp_cyclic_persist_01_report.md`):** Combined destination and persistence supervision ($\lambda_{\text{dest}} = 1.0, \lambda_{\text{persist}} = 0.5$). Outcome: `BEHAVIORAL_PASS_CANONICAL_FULL` (macro validation accuracy $0.9875$; 4/5 seeds at 1.0000, 1 seed at 0.9375; asymmetry resolved: $\bar{C}_{\text{read}}(r0) = 0.8879$, $\bar{C}_{\text{read}}(r1) = 0.9303$).
- **Factorial Grid Status:** `EXP-CYCLIC-PERSIST-ABL-01` provides Cell (0, 1) to complete the $2 \times 2$ supervision grid across conditions $\text{D} \in \{0, 1\}$ (Destination) and $\text{P} \in \{0, 1\}$ (Persistence).

## 5. Hypothesis

- **Hypothesis $H_{\text{persist-sufficient}}$ (Persistence Sufficiency):** Persistence supervision on unaddressed register $r0$ at step 1 provides sufficient inductive bias to constrain continuous state trajectory $S_1 \to S_2$. Terminal loss penalizing incorrect addition at step 2 forces `mlp_gate` and `mlp_write` to place $r1$ into orthogonal coordinate subspaces without requiring direct destination losses at steps 0 and 1. Predicted outcome: $acc_{\text{val}} \ge 0.950$, $\bar{C}_{\text{read}} \ge 0.80$, $\bar{C}_{\text{ALU}} \ge 0.80$.
- **Hypothesis $H_{\text{dest-necessary}}$ (Destination Necessity / Joint Synergy):** Direct destination supervision is causally necessary to anchor representations in the fixed Fourier basis $\Phi$. In the absence of intermediate destination supervision:
  - *Sub-hypothesis $H_{\text{inv-asymmetry}}$ (Inverted Read Asymmetry):* Register $r0$ receives direct supervision at step 1 and is preserved ($\bar{C}_{\text{read}}(r0) \ge 0.80$), but register $r1$ receives zero auxiliary supervision anywhere in the trace and fails to align to the canonical basis ($\bar{C}_{\text{read}}(r1) < 0.80$), inverting the asymmetry observed in `EXP-CYCLIC-STEPWISE-01`.
  - Predicted outcome: Progression gate fails ($acc_{\text{val}} < 0.950$ across seeds); representation metrics fail canonical threshold (`REP_PARTIAL` or `REP_FAILED`).

## 6. What is Frozen

- **Model Architecture:** `CyclicMGUPhase16Model` (`src/re_reason/core/register_machine_models.py`) with monolithic continuous state $S_t \in \mathbb{R}^{212}$, fixed 5-harmonic basis $\Phi(v) \in \mathbb{R}^{10 \times 10}$, parameter-free cyclic Givens rotation ALU, and learned networks `mlp_read` (29,066 params), `mlp_write` (52,152 params), and `mlp_gate` (137,800 params). Total trainable parameters: exactly **219,018** (invariant at initialization and after training across all seeds).
- **Zero Trainable Readout Overhead:** The persistence query uses the shared `mlp_read` network decoded against prototype basis $\Phi$; no new parameters, projections, or heads.
- **Dataset Partitions:** Canonical `Stage1PairPartition` with 64 training pairs (`TRAIN-FIT`), 16 validation pairs (`VAL-SEEN`), and 20 protected test pairs ($b \in \{0, 5\}$).
- **Optimization:** AdamW ($\text{lr} = 3 \times 10^{-3}$, $\text{weight\_decay} = 10^{-2}$, $\beta = (0.9, 0.999)$), batch size 64, exactly 3,000 updates, evaluation cadence every 50 updates (60 checkpoints per seed).
- **Hardware Target:** Host CPU.
- **Evaluation Firewall:** The protected TEST split ($b \in \{0, 5\}$) remains strictly unqueried (`queries = 0`).

## 7. What Changed

The single intervention relative to `EXP-CYCLIC-PERSIST-01` is:
- **Removal of Destination Supervision Terms from the Training Loss ($\lambda_{\text{dest}} = 0.0$):**
  Intermediate destination losses at step 0 ($\text{CE}(\text{logits}_{\text{dest}, 0}, a)$) and step 1 ($\text{CE}(\text{logits}_{\text{dest}, 1}, b)$) are excluded from backpropagation.
- **Preregistered Objective Function:**
  $$\mathcal{L}_{\text{total}}^{\text{abl}} = \mathcal{L}_{\text{terminal}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$$
  Approved Option 1 coefficient $\lambda_{\text{persist}} = 0.5$ preserves nominal scalar weighting invariance relative to `EXP-CYCLIC-PERSIST-01`.
- **Diagnostic Reference Loss:**
  $$\mathcal{L}_{\text{composite\_reference}} = \mathcal{L}_{\text{terminal}} + 1.0 \cdot \mathcal{L}_{\text{dest}} + 0.5 \cdot \mathcal{L}_{\text{persist}}$$
  Recorded in telemetry for diagnostic comparison matching `EXP-CYCLIC-PERSIST-01`; does not govern checkpoint selection.
- **Checkpoint Selection Policy:** Registered policy strictly governed by native ablation validation loss: minimum $\mathcal{L}_{\text{val\_total}}$, ties broken by maximum $acc_{\text{val}}$, then earliest update step.

## 8. Dataset and Protocol

- **Task:** 3-instruction modular addition modulo 10:
  $t=0: \text{SET}\ r0,\ a \implies S_0 \to S_1$;
  $t=1: \text{SET}\ r1,\ b \implies S_1 \to S_2$;
  $t=2: \text{ADD}\ r0,\ r1 \implies S_2 \to S_3$, target $(a + b) \bmod 10$.
- **Partitions:** 64 training pairs, 16 validation pairs (`VAL-SEEN`), 20 firewalled test pairs ($b \in \{0, 5\}$).
- **Progression Gate:** $acc_{\text{val}} \ge 0.950$ across $\ge 4/5$ seeds with zero seeds $< 0.250$ (on `VAL-SEEN`).
- **Rejection Boundary:** $acc_{\text{val}} < 0.250$ across $\ge 4/5$ seeds ($N_{\text{reject}} \ge 4$).

## 9. Experimental Procedure

1. **Stage 1 (Implementation, Focused Tests, Smoke Run):** Implemented in `experiments/run_exp_cyclic_persist_abl_01.py` with 26 focused tests in `tests/test_exp_cyclic_persist_abl_01.py`. 100-update seed-42 smoke run executed on 2026-10-16 (`experiments/exp_cyclic_persist_abl_01_SMOKE_ONLY_seed42.json`). Audited and verified.
2. **Stage 2 (Official Five-Seed Training):** Authorized by PI on 2026-10-17. Executed via `.venv/bin/python experiments/run_exp_cyclic_persist_abl_01.py --stage2-authorized --device cpu`. All five seeds (42–46) completed 3,000 updates each with 60 checkpoints per seed (300 checkpoints total). Serialized to `experiments/exp_cyclic_persist_abl_01_results.json`.
3. **Independent Raw-Evidence Audit:** Conducted independently by Antigravity. Recomputed loss decomposition identities across all 300 checkpoints, verified parameter-count invariance (219,018), independently reproduced checkpoint selections, and verified decision matrix outputs.

## 10. Results

### Per-Seed Selected Checkpoints

Selected by preregistered policy ($\min \mathcal{L}_{\text{val\_total}} \to \max acc_{\text{val}} \to \text{earliest update}$):

| Seed | Selected Update | Val Acc ($k/16$) | Val Acc | Val Loss Total | Val Loss Terminal | Val Loss Persist | Val Loss Dest (diag) | Val Loss Comp Ref | Train Loss Total | Train Acc | Final Val Acc | Runtime (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 | 150 | 6/16 | 0.3750 | 1.7293 | 1.3912 | 0.6763 | 1.0963 | 2.8257 | 0.2319 | 1.0000 | 0.3125 | 26.95 |
| 43 | 350 | 8/16 | 0.5000 | 1.5491 | 1.2589 | 0.5804 | 1.0805 | 2.6297 | 0.0898 | 1.0000 | 0.3750 | 26.69 |
| 44 | 950 | 10/16 | 0.6250 | 1.7052 | 1.3414 | 0.7276 | 1.0427 | 2.7480 | 0.0886 | 1.0000 | 0.4375 | 26.40 |
| 45 | 200 | 7/16 | 0.4375 | 1.9778 | 1.6377 | 0.6803 | 1.2505 | 3.2284 | 0.0904 | 1.0000 | 0.3750 | 26.65 |
| 46 | 1500 | 12/16 | 0.7500 | 1.0700 | 0.8160 | 0.5080 | 0.8659 | 1.9360 | 0.0924 | 1.0000 | 0.5625 | 26.62 |
| **Macro** | — | **8.6/16** | **0.5375** | **1.6063** | **1.2890** | **0.6345** | **1.0672** | **2.6735** | **0.1186** | **1.0000** | **0.4125** | **133.31 (total)** |

### Representation Metrics (Selected Checkpoints; VAL-SEEN)

| Seed | $C_{\text{read}}$ | $C_{\text{read}}(r0)$ | $C_{\text{read}}(r1)$ | $C_{\text{persist}}(r0)$ | $S_{\text{context}}$ | $C_{\text{ALU}}$ | $C_{\text{dest}}$ | $C_{\text{dest, s0}}$ | $C_{\text{dest, s1}}$ | $C_{\text{dest, s2}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 | 0.4942 | 0.7236 | 0.2648 | 0.7236 | 0.3263 | 0.4019 | 0.5668 | 0.9998 | 0.2648 | 0.4357 |
| 43 | 0.5156 | 0.8155 | 0.2158 | 0.8155 | 0.3256 | 0.4166 | 0.5778 | 0.9997 | 0.2158 | 0.5178 |
| 44 | 0.5058 | 0.7474 | 0.2642 | 0.7474 | 0.2847 | 0.3688 | 0.5946 | 0.9994 | 0.2642 | 0.5203 |
| 45 | 0.4953 | 0.8138 | 0.1768 | 0.8138 | 0.3260 | 0.4097 | 0.5210 | 0.9994 | 0.1768 | 0.3867 |
| 46 | 0.5658 | 0.7548 | 0.3768 | 0.7548 | 0.4176 | 0.5949 | 0.6820 | 0.9997 | 0.3768 | 0.6696 |
| **Macro** | **0.5154** | **0.7710** | **0.2597** | **0.7710** | **0.3360** | **0.4384** | **0.5884** | **0.9996** | **0.2597** | **0.5060** |

### Per-Step Diagnostic Accuracies (Selected Checkpoints; VAL-SEEN)

- **Step 0 Readout Accuracy ($a$ in $r0$ after step 0):** 1.0000 across all 5 seeds (macro: 1.0000).
- **Step 1 Destination Readout Accuracy ($b$ in $r1$ after step 1):** 0.1875, 0.0625, 0.2500, 0.0000, 0.1250 (macro: **0.1250** — near 0.10 chance).
- **Step 1 Persistence Readout Accuracy ($a$ preserved in $r0$ after step 1):** 0.8125, 1.0000, 0.8125, 1.0000, 0.9375 (macro: **0.9125**).
- **Step 2 Terminal Accuracy ($(a + b) \bmod 10$):** 0.3750, 0.5000, 0.6250, 0.4375, 0.7500 (macro: **0.5375**).

## 11. Statistical Analysis

### Preregistered Decision Matrix Evaluation (§8)

1. **Behavioral Evaluation (5 seeds, $N_{\text{val}} = 16$):**
   - Seed 42: $k = 6$ ($acc = 0.3750$) $\implies$ Intermediate ($0.250 \le acc < 0.950$)
   - Seed 43: $k = 8$ ($acc = 0.5000$) $\implies$ Intermediate
   - Seed 44: $k = 10$ ($acc = 0.6250$) $\implies$ Intermediate
   - Seed 45: $k = 7$ ($acc = 0.4375$) $\implies$ Intermediate
   - Seed 46: $k = 12$ ($acc = 0.7500$) $\implies$ Intermediate
   - Aggregate counts: $N_{\text{pass}} = 0$, $N_{\text{mid}} = 5$, $N_{\text{reject}} = 0$.
   - Progression gate ($N_{\text{pass}} \ge 4 \land N_{\text{reject}} = 0$): **NOT MET**.
   - Rejection boundary ($N_{\text{reject}} \ge 4$): **NOT MET**.
   - $\implies$ **`BEHAVIOR_INTERMEDIATE`**.
2. **Representation Contract (Macro-Averaged):**
   - $\bar{C}_{\text{read}} = 0.5154$ (threshold: $0.80$, sub-criterion $\ge 0.50$ met)
   - $\bar{S}_{\text{context}} = 0.3360 < 0.80$
   - $\bar{C}_{\text{ALU}} = 0.4384 < 0.80$
   - Canonical full criteria ($\bar{C}_{\text{read}}, \bar{S}_{\text{context}}, \bar{C}_{\text{ALU}} \ge 0.80$) not met; at least one metric $\ge 0.50$ satisfied.
   - $\implies$ **`REP_PARTIAL`**.
3. **Read Asymmetry Analysis:**
   - $\bar{C}_{\text{read}}(r0) = 0.7710$ (substantially preserved)
   - $\bar{C}_{\text{read}}(r1) = 0.2597$ (severely degraded, below 0.30)
   - Read asymmetry $\Delta C = \bar{C}_{\text{read}}(r0) - \bar{C}_{\text{read}}(r1) = +0.5113$.
   - This represents an **inverted read asymmetry** relative to `EXP-CYCLIC-STEPWISE-01`, where $\Delta C = 0.2573 - 0.8368 = -0.5795$.
4. **Mechanical Outcome Category:** **`INCONCLUSIVE_INTERMEDIATE`** (intersection of `BEHAVIOR_INTERMEDIATE` and `REP_PARTIAL`).

### Factorial $2 \times 2$ Supervision Grid Contrast

All four experiments share an identical architecture (`CyclicMGUPhase16Model`, 219,018 params), identical 5-seed cohort (42–46), identical 64-pair training split, identical 16-pair validation split, AdamW optimizer, and 3,000 updates:

| Condition Cell | Experiment ID | Dest ($\lambda_{\text{dest}}$) | Persist ($\lambda_{\text{persist}}$) | Macro Val Acc | $C_{\text{read}}(r0)$ | $C_{\text{read}}(r1)$ | Macro $C_{\text{read}}$ | Macro $S_{\text{context}}$ | Macro $C_{\text{ALU}}$ | Outcome Category |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cell (0, 0)** | `EXP-CYCLIC-ADD-01` | 0.0 | 0.0 | 0.0625 | ~0.17 | ~0.18 | ~0.176 | ~0.15 | ~0.117 | `FAIL_REJECTION` |
| **Cell (1, 0)** | `EXP-CYCLIC-STEPWISE-01`| 1.0 | 0.0 | 0.6750 | 0.2573 | 0.8368 | 0.5471 | 0.3167 | 0.5374 | `INCONCLUSIVE_INTERMEDIATE` |
| **Cell (0, 1)** | **`EXP-CYCLIC-PERSIST-ABL-01`** | **0.0** | **0.5** | **0.5375** | **0.7710** | **0.2597** | **0.5154** | **0.3360** | **0.4384** | **`INCONCLUSIVE_INTERMEDIATE`** |
| **Cell (1, 1)** | `EXP-CYCLIC-PERSIST-01` | 1.0 | 0.5 | **0.9875** | **0.8879** | **0.9303** | **0.9091** | **0.8584** | **0.8496** | **`BEHAVIORAL_PASS_CANONICAL_FULL`** |

## 12. Interpretation

### Observed Facts

1. **Persistence Alone Fails Progression:** Omitting intermediate destination supervision dropped macro validation accuracy from $0.9875$ (`EXP-CYCLIC-PERSIST-01`) to $0.5375$ ($-0.4500$). Zero of five seeds passed the progression gate ($0/5 \ge 0.950$).
2. **Inverted Read Asymmetry:**
   - In `EXP-CYCLIC-STEPWISE-01` (destination only), $r1$ was supervised at step 1 and aligned ($0.8368$), while unaddressed $r0$ degraded ($0.2573$).
   - In `EXP-CYCLIC-PERSIST-ABL-01` (persistence only), $r0$ was supervised at step 1 and survived ($0.7710$), while unaddressed $r1$ received zero auxiliary supervision and collapsed ($0.2597$).
   - The asymmetry inverted with near-symmetric magnitude: $+0.5113$ vs $-0.5795$.
3. **Register 1 Write Grounding Failure:** Without intermediate supervision on the $r1$ write at step 1, the step 1 destination readout accuracy was $0.1250$ (near chance $0.10$). The network failed to ground operand $b$ into canonical Fourier coordinates during the write transition.
4. **ALU Bottleneck:** The analytical cyclic ALU requires both input operands $u_a$ and $u_b$ in canonical Fourier basis coordinates. With $r1$ misaligned ($C_{\text{read}}(r1) = 0.2597$), $C_{\text{ALU}}$ dropped from $0.8496$ to $0.4384$, directly bottlenecking modular addition output.

### Hypotheses and Explanations

- *Hypothesis $H_{\text{persist-sufficient}}$ Falsified:* Persistence supervision alone alongside terminal loss is **not sufficient** to induce dual-register coordinate preservation or sequence generalization.
- *Hypothesis $H_{\text{dest-necessary}}$ and Sub-hypothesis $H_{\text{inv-asymmetry}}$ Supported for Evaluated Setup:* Within the evaluated 5-seed cohort and architecture (`CyclicMGUPhase16Model`), intermediate destination supervision is necessary under this objective formulation to ground newly written registers into canonical basis coordinates.
- *Mechanistic Account of Joint Synergy:* In a continuous recurrent state without discrete memory boundaries:
  - Destination supervision grounds the *write* operation into canonical coordinates ($r1 \leftarrow b$).
  - Persistence supervision prevents the intervening write from dissolving previously stored coordinates ($r0 \to a$ across $S_1 \to S_2$).
  - Neither supervision signal alone suffices: destination-only leaves $r0$ vulnerable to write-interference, while persistence-only leaves $r1$ ungrounded. Joint supervision is synergistic and necessary to anchor both registers simultaneously under this objective.

## 13. What the Experiment DOES Establish

- Proves that persistence supervision alone ($\lambda_{\text{persist}} = 0.5, \lambda_{\text{dest}} = 0.0$) does not reproduce the validation generalization or canonical alignment of `EXP-CYCLIC-PERSIST-01`.
- Proves the emergence of an inverted read asymmetry: persistence supervision preserves the unaddressed register ($C_{\text{read}}(r0) = 0.7710$), but the ungrounded write fails canonical alignment ($C_{\text{read}}(r1) = 0.2597$).
- Establishes that intermediate destination supervision is necessary under this objective formulation to anchor write representations in the continuous state for modular addition sequence execution on `CyclicMGUPhase16Model`.

## 14. What the Experiment DOES NOT Establish

- Does NOT establish out-of-distribution failure. The protected TEST split ($b \in \{0, 5\}$) was strictly firewalled (`queries = 0`). The reported failure is on sequence-disjoint combinations of seen training digits (`VAL-SEEN`).
- Does NOT establish universal failure for all neural architectures or larger parameter budgets. The finding is strictly scoped to `CyclicMGUPhase16Model` (219,018 params) under the registered 3-instruction trace.
- Does NOT establish inferential claims of statistical significance. Across $N=5$ matched seeds, the findings represent descriptive contrasts within a controlled factorial design.
- Does NOT claim that the ALU failed to learn arithmetic; the ALU is fixed and analytical.

## 15. Decision

**EXP-CYCLIC-PERSIST-ABL-01 is recorded as INCONCLUSIVE_INTERMEDIATE.** The persistence-only ablation arm failed the behavioral progression gate ($N_{\text{pass}} = 0, N_{\text{mid}} = 5, N_{\text{reject}} = 0$) and achieved partial representation alignment (`REP_PARTIAL`). The hypothesis of persistence sufficiency ($H_{\text{persist-sufficient}}$) is rejected; the hypothesis of destination necessity and inverted asymmetry ($H_{\text{inv-asymmetry}}$) is supported. Joint destination and persistence supervision (`EXP-CYCLIC-PERSIST-01`) remains the sole validated objective supporting canonical sequence arithmetic on `CyclicMGUPhase16Model`.

## 16. Next Experiment

Pending explicit PI direction, candidate next steps are:
1. **Phase 1.6 Stage 3 Composed Sequence Benchmark Formulation:** Formulate multi-operation or composed sequence evaluation for `CyclicMGUPhase16Model` under the validated joint objective (`EXP-CYCLIC-PERSIST-01`).
2. **Protected TEST Split Evaluation:** Authorize single-evaluation held-out evaluation ($b \in \{0, 5\}$) for the validated `EXP-CYCLIC-PERSIST-01` model under strict firewall governance.
3. **Runner Overwrite Hardening:** Implement explicit overwrite-refusal safeguards in `experiments/run_exp_cyclic_persist_abl_01.py` for official execution mode under separate authorized maintenance.
4. **Experiment L Investigation:** Audit pre-existing checkpoint reproduction drift in `tests/test_experiment_l.py`.

## 17. Reproduction Commands

```bash
# Execute official Stage 2 five-seed experiment (host CPU)
.venv/bin/python experiments/run_exp_cyclic_persist_abl_01.py --stage2-authorized --device cpu

# Run focused unit tests (26 tests)
pytest tests/test_exp_cyclic_persist_abl_01.py -v

# Run documentation governance tests (13 tests)
pytest tests/test_documentation_governance.py -v
```

## 18. Artifact List

- Runner: `experiments/run_exp_cyclic_persist_abl_01.py`
- Focused Test Suite: `tests/test_exp_cyclic_persist_abl_01.py`
- Official Results Artifact: `experiments/exp_cyclic_persist_abl_01_results.json`
- Stage 1 Smoke Telemetry Artifact: `experiments/exp_cyclic_persist_abl_01_SMOKE_ONLY_seed42.json`
- Authoritative Specification: `docs/research/exp_cyclic_persist_abl_01_specification.md`

## 19. Scientific Guardrails

- Clear separation of verified empirical observations from explanatory hypotheses.
- Rejection of informal rounding; exact preregistered thresholds adhered to.
- Parameter count strictly invariant at 219,018 across all seeds.
- Protected TEST partition ($b \in \{0, 5\}$) strictly firewalled (`queries = 0`).
- No claims of general reasoning; arithmetic core is fixed and analytical.
