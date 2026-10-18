# Model E Held-Out Generalization: Slot-Cyclic Latent Model Design

**Date:** 2026-10-18
**Status:** Accepted design proposal; protocol amendment and guarded implementation committed; training not started
**Model name:** Slot-Cyclic Latent Model (SCLM, registered candidate)
**Scope:** Improve held-out operand generalization on the modular register-tracking task.

## 1. Goal and decision

Design a trainable latent-state model intended to replace Model E in the goal system only if it clears the held-out transfer gate. Preserve Model E's frozen checkpoint and all historical measurements as the baseline record.

The co-primary success criteria are:

- Held-out ADD-transition balanced accuracy **≥ 0.800**.
- Held-out sequence exact match **≥ 0.700**.

The protected final test uses source-register values `b ∈ {0, 5}` and receives one final evaluation only. Before that evaluation, use a development proxy holdout from currently seen operands. The recommended proxy is `b = 3`, reserved from training and placed in sequence-disjoint proxy validation. Proxy success is not a claim about `b ∈ {0, 5}`.

Each training run must take less than 12 hours on the current local machine. No runtime estimate is accepted until measured on that machine. The user permits a larger parameter or training budget within that per-run cap.

## 2. Evidence and interpretation

### Directly supported findings

- Model E is the active validated `NonlinearCandidateMGUModel`, with 226,358 parameters, a 212-dimensional state, and a candidate/local objective. Its reported exact match is `0.755 ± 0.019` at T=10 and `0.407 ± 0.023` at T=20 (`README.md:126–165`; `docs/RESEARCH_STATUS.md:13–36`).
- The ADD observation encoder is operand-blind by construction; the recurrent state pathway is weak on held-out operand transfer (`docs/RESEARCH_STATUS.md:59–61`).
- On candidate-wrong ADD transitions, an analysis-only linear probe recovered the true written value 26.7% for Model E; nonlinear probing did not materially improve it. A hard-pool probe reached 0.493 on seen combinations and 0.176 on held-out combinations (`docs/RESEARCH_STATUS.md:56–58`).
- The raw causal prefix contains enough information to identify the target, while bounded readers fail to extract it; a GRU can fit a small sample and still fail held-out transfer (`docs/RESEARCH_STATUS.md:60–65`).
- The first-error breakdown reports candidate-origin errors as the largest observed category. It explicitly does **not** establish candidate-module causality (`docs/RESEARCH_STATUS.md:55`).

### Working hypothesis

Held-out failure reflects coupled weaknesses in register-value access/state representation and learned arithmetic over seen operand combinations. Explicit slots address the former; a fixed cyclic operation supplies an arithmetic prior for the latter. Neither component's benefit in the combined learned model has been demonstrated.

## 3. Proposed model

SCLM stores one continuous latent vector per register, with four 53-dimensional slots (212 dimensions total). Address routing uses the operation's explicit register address. Both ADD operands are read from a pre-update snapshot.

A shared learned decoder maps each slot `z_r` to a distribution `p_r` over the ten values. A shared learned writer maps a ten-value distribution back into a 53-dimensional slot state. For `ADD`, the fixed cyclic group operation computes:

```text
p_sum[k] = Σ_i p_dest[i] · p_src[(k − i) mod 10]
```

For one-hot operands this yields the exact one-hot modular sum. The writer stores the result only in the destination slot. `SET` uses the same writer, `SWAP` exchanges slot states atomically, and `READ` decodes a slot without mutating state. Non-addressed slots remain unchanged by construction.

**Learned:** slot-state encoding/writing and decoding.
**Fixed:** address dispatch, operation semantics, untouched-slot identity, and cyclic arithmetic.

A successful SCLM result would show generalization of learned latent memory around an explicit arithmetic prior. It would not show that the model learned modular addition. The deterministic Modular Slot Machine (MSM) remains a separate exact reference, never a learned-model result. At design time, exact adapter dimensions, parameter counts, and training-loss weighting still required reconciliation in the implementation contract; the 2026-10-18 amendment now records the controlling values.

## 4. Controlled comparisons

Use a four-cell comparison to distinguish memory organization from arithmetic mechanism:

| Memory | Arithmetic | Role |
|---|---|---|
| Monolithic latent state | Learned MLP | Masked-trained Model E-architecture control |
| Four explicit slots | Learned MLP | Slot-MGU control |
| Monolithic latent state | Fixed cyclic operation | Cyclic-MGU control |
| Four explicit slots | Fixed cyclic operation | SCLM candidate |

All four conditions use the same training and validation data, seed set, optimizer/update schedule, checkpoint-selection rule, and 212-dimensional total state budget. Report each parameter count; the fixed operator has no trainable parameters, so parameter equality is not asserted. Comparisons support the planned factor contrasts only at the level the matched implementation permits; do not claim a balanced factorial main-effect result without a parameter and implementation audit.

The historical Model E checkpoint was trained with operand exposure that makes it an invalid held-out test control. Preserve it for historical and seen-distribution comparisons; train the monolithic/MLP control from scratch under the masked protocol. Do not load or evaluate the old checkpoint on the protected held-out test.

## 5. Data, training, and gates

The existing Phase 1.6 Regime B protocol uses 6,000 T=10 masked training sequences and 1,200 validation sequences, excluding `b ∈ {0, 5}` from all ADD transitions; its 1,200 test sequences contain at least one such transition (`phase_1_6_architectural_reformulation_proposal.md:476–492`). That validation set cannot measure transfer to the excluded values.

Proposed amendment:

1. Reserve `b = 3` for a development-only proxy set; exclude it, along with `b ∈ {0, 5}`, from ADD transitions in training. Keep proxy and training sequences disjoint. Retain a seen-only validation split for convergence diagnostics. Freeze the proxy construction and split checks before training.
2. Train all four cells under the same masked regime. Preserve the registered five-seed set (42–46), AdamW configuration, batch size 64, 3,000-update schedule, 50-update validation cadence, and validation-only checkpoint selection unless a formally approved amendment changes them (`phase_1_6_architectural_reformulation_proposal.md:508–526`). The local runtime check must confirm every run is under 12 hours before the full campaign.
3. Apply both transfer thresholds to proxy sequences containing at least one ADD with source value `b = 3`. Use the proxy only for development selection. Keep exact seed consistency and rejection rules from the controlling protocol unless the amendment explicitly revises them.
4. If SCLM passes the preregistered proxy gate, freeze the selected checkpoints and perform one final evaluation on the protected `b ∈ {0, 5}` split. Evaluate SCLM only in that final pass. Apply the same joint thresholds; if it fails, do not tune against or re-query the protected set.
5. A final pass of the held-out transfer gate qualifies SCLM for the goal system as directed by the user. It does not satisfy the repository's full six-stage conjunction and does not authorize deferred Phase 2+ mechanisms (`phase_1_6_architectural_reformulation_proposal.md:658–663`; `AGENTS.md:24`).

## 6. Design-time governance and required amendment

At design time, the existing Phase 1.6 proposal ratified an unbalanced candidate roster and explicitly rejected a fourth monolithic-MLP candidate; it also deferred the curriculum (`phase_1_6_architectural_reformulation_proposal.md:120–129, 553–560`). This design changed the roster and split plan. Before implementation or training, an append-only protocol amendment was required to define the new candidate, the `b=3` proxy, the masked training regime, data disjointness, exact loss, checkpoint selection, seed consistency, metrics, parameter accounting, timing procedure, and the one-pass test procedure. The required amendment was subsequently appended; the original proposal and prior evidence remain preserved.

No branching, learned verifier, or other deferred Phase 2+ machinery is included. No tests, training, data generation, test-split inspection, checkpoint inspection, or implementation were performed for this proposal.

## 7. Antigravity plan review

Antigravity's evidence-only review recommended combining explicit slots with cyclic arithmetic. That synthesis is plausible and aligns with the observed representation and transfer gaps. Its numerical runtime estimates and proposed thresholds (including entropy, seen-accuracy, probe-recovery, and candidate-error cutoffs) were unsupported and are not adopted. Its description of candidate-first errors as evidence of an observational cause overstates the source report, which marks the statistic associational. The fixed-arithmetic candidate is explicitly classified here as a learned latent-memory model with a hand-specified arithmetic prior, distinct from deterministic MSM.

At the time this design was accepted, CommandCode had not implemented or generated code; implementation was gated on the required pre-implementation review and protocol amendment. The subsequent implementation is documented in §10.

## 8. Decision log

| Decision | Alternatives considered | Rationale |
|---|---|---|
| Optimize held-out generalization on Model E's register-tracking task. | Pivot directly to the broader end-to-end RE:Reason architecture. | The user selected Model E transfer as the immediate objective. |
| Keep Model E as a preserved research checkpoint and baseline. | Replace it immediately. | Preserve historical comparability; qualify a new goal-system candidate only after the agreed gate. |
| Combine explicit slots with fixed cyclic arithmetic and use matched ablations. | Slot-MGU alone; monolithic Cyclic-MGU alone; retain the old roster. | Targets both observed bottlenecks while controlled comparisons expose their separate contributions. |
| Keep learned memory interfaces distinct from fixed arithmetic. | Count a deterministic exact slot machine as a learned model. | Avoid claiming the prior itself was learned. |
| Use a `b=3` proxy and retain `b∈{0,5}` for one final pass. | Iterate directly on the protected split; or make no held-out proxy. | Enables development iteration while preserving the existing firewall. |
| Keep each run below 12 hours on local hardware. | Estimate runtime from the CLI memo. | The memo's timing estimate was not measured; local feasibility must be demonstrated. |
| Do not begin implementation or training in this design step. | Start coding immediately. | The user requested staged design review; the repo requires a protocol amendment and contracts first. |

## 9. Design-time open implementation-contract items (resolved by the amendment in §10)

Before code was written, the following items required independent review and freezing: exact shared encoder/decoder/writer shapes and loss weighting; exact four-cell interfaces and parameter counts; whether the 212-dimensional state budget is retained for every condition; the `b=3` proxy sample construction and size; proxy checkpoint-selection metric; seed consistency; maximum total campaign budget; and runtime measurement procedure on the local machine. The amendment cited in §10 records the resolved contract.

## 10. Implementation-stage update — 2026-10-19

The implementation gate described above was subsequently completed through the append-only amendment in `docs/reports/phase_1_6_architectural_reformulation_proposal.md` (2026-10-18). That amendment is authoritative for the final candidate roster, interfaces, parameter counts, masked training and proxy splits, objective, checkpoint selection, seed set, runtime boundary, and protected-test firewall; it resolves the design-time open contract items in §9. The four candidate implementations, additive benchmark/split support, objective/metrics, and guarded runner are committed in `3db8183`. The smoke-check scope correction is recorded in `3908a46` and `docs/RESEARCH_LOG.md`.

Implementation-stage verification is recorded in the research log: 35 focused pure tests and the documentation-governance suite passed; the generator/split tests were authored but not executed at that stage. A bounded in-memory smoke check did generate exploratory sequences, which exceeded the then-active no-generation instruction; its scope is recorded as a process deviation, not empirical evidence. No official campaign dataset, training run, profiling stop-gate, checkpoint inspection, proxy evaluation, or protected-test access has occurred. Therefore, SCLM remains an untrained candidate; no held-out generalization claim or replacement of the validated Model E baseline is supported.

The dated decision table above records the original design-time choices, including the decision not to implement during that planning step. The later amendment and implementation do not change those recorded choices; they document the subsequent authorized stage. Execution still requires the separate gates stated in the controlling Phase 1.6 amendment.
