# RE:Reason — Research Ledger (Append-Only Chronological Record)

**Established:** 2026-10-05 (at adoption of `docs/DOCUMENTATION_POLICY.md`)
**Rules:** This file is append-only. New entries go at the end. Existing entries are never edited, reordered, or deleted; later interpretation belongs in a **new** entry. Corrections are appended amendments (see `docs/DOCUMENTATION_POLICY.md` §4).

**Scope & provenance:** The entries below reconstruct the completed research steps from the reports under `docs/reports/` and their machine-readable artifacts. Dates and facts follow the reports; each entry cites its sources. Where an entry's *Next experiment* field names the next executed step, it reflects the documented sequence; where a report explicitly prescribed its own next step, the wording follows that report. Nothing in this ledger is invented beyond the cited documentation.

**Entry format fields:** Status · Research question · Configuration · Key result · Decision · Evidence · Changes to project state · Open questions · Next experiment.

---

## 2026-10-04 — Phase 1 Overfitting / Underfitting Diagnostic Suite

**Status:** Completed — diagnostic investigation (10 axes), negative-result-preserving.
**Research question:** Why does the Phase-1 persistent-state model (and the GRU baseline) underperform under terminal-only supervision — underfitting, overfitting/memorization, representation interference / credit assignment, or genuine architectural disadvantage?
**Configuration:** Register Tracking (N=4 registers, T=10, V=10; SET/SWAP/ADD); strict SHA-256-verified procedural splits (train 6,000 / val 1,200 / test 1,200, zero leakage); terminal supervision at t=10; training-budget sweep 1,200 → 3,000 → 6,000 → 12,000 steps; step-wise supervision diagnostic.
**Key result:** Massive generalization gap under terminal supervision (RE:Reason train 98.44% vs val 6.70%; GRU train 100.00% vs val 11.90%) — finite fixed-split training induces sequence memorization. Under step-wise supervision, RE:Reason test exact match rises ~4× to 20.67% (GRU 22.50%; delta −1.83 pp, inside the project's ±2% parity band), showing the gap was credit assignment under terminal supervision, not an inability of the 256-dim state to represent register values.
**Decision:** Do **not** declare architectural failure. Lock the diagnostics; investigate gating/routing mechanisms (Phase 1.5) before considering Phase-2 mechanisms. *(Report §11, verbatim: "Do NOT declare architectural failure… To move toward parity under terminal-only supervision, Phase 1.5 or Phase 2 should investigate gating/routing mechanisms… rather than abandoning persistent latent state.")*
**Evidence:**
- report: `docs/reports/phase1_overfitting_underfitting_diagnosis.md`
- data: `experiments/diagnostic_results.json`
- figures: `docs/reports/figures/01_state_norm_over_time.png` … `08_failure_rate_by_operation_type.png`
- code: `experiments/run_full_diagnostics.py`
**Changes to project state:** Phase-1 gate review: proceed to Phase 1.5 gating ablations; Phase 1 not abandoned.
**Open questions:** Which gating/routing mechanism fixes terminal-supervision credit assignment?
**Next experiment:** Phase 1.5 Causal Architectural Ablation (`docs/reports/phase1_5_causal_ablation_report.md`).

---

## 2026-10-04 — Phase 1.5 Causal Architectural Ablation (Models A–D)

**Status:** Completed — controlled causal ablation, 4 parameter-matched models × 5 seeds.
**Research question:** Is RE:Reason losing because of LayerNorm, because it lacks gating, because additive recurrence itself is inadequate, or because terminal-only supervision exposes a broader temporal-credit problem?
**Configuration:** Models A (LayerNorm additive), B (residual no-LN), C (unbounded gated residual `S + g⊙f`), D (parameter-matched `nn.GRUCell`); identical encoders/heads; AdamW lr 1e-3, wd 0.01, batch 64, 3,000 steps; seeds 42–46; terminal supervision at t=10.
**Key result:** A 5.83±1.27%, B 5.93±1.18%, C 4.73±0.91%, D 10.78±0.44% test exact match. LayerNorm is not the primary bottleneck; the tested gated variant was an **unbounded additive** gate, so bounded convex interpolation `(1−z)S + z·tanh(f)` remained untested. Terminal credit assignment across unsupervised recurrent steps is the primary difficulty for un-gated additive updates.
**Decision:** Execute a dedicated **bounded convex** ablation (the missing hypothesis) before any Phase-2 mechanism.
**Evidence:**
- report: `docs/reports/phase1_5_causal_ablation_report.md`
- data: `experiments/ablation_1_5_results.json`
- code: `experiments/run_ablation_1_5.py`; tests: `tests/test_phase1_5_ablation.py`
**Changes to project state:** None adopted; bounded convex flagged as the open hypothesis.
**Open questions:** Does bounded convex state updating materially improve the persistent core?
**Next experiment:** Phase 1.5 Bounded Convex Residual Ablation (`docs/reports/phase1_5_bounded_convex_ablation_report.md`).

---

## 2026-10-04 — Phase 1.5 Bounded Convex Residual Ablation (Model E)

**Status:** Completed — mechanism confirmation; **adopted**.
**Research question:** Does bounded convex state updating, without a GRU-style reset gate, materially improve RE:Reason under the official terminal-supervision benchmark?
**Configuration:** Model E `S_t = (1−z_t)⊙S_{t−1} + z_t⊙tanh(f_θ(S_{t−1}, e_t))` vs Model A and Model D (GRU); ±1.2% parameter parity; 5 seeds; identical splits and optimizer.
**Key result:** Model E 45.55% ± 4.38% test exact match (per-register 81.35%); Model D 11.25% ± 1.29%; Model A 6.13% ± 0.73%. Delta vs A +39.42 pp; vs GRU +34.30 pp; E has fewer parameters than both. Interpretation matrix: Case 1 (strong mechanism confirmation).
**Decision:** Formally adopt the **Bounded Convex Residual Update** as the foundational state-update mechanism; do not introduce later-phase complexity prematurely. *(Report §9 gate decision, verbatim.)*
**Evidence:**
- report: `docs/reports/phase1_5_bounded_convex_ablation_report.md`
- data: `experiments/ablation_1_5_bounded_results.json`
- code: `experiments/run_ablation_1_5_bounded.py`; tests: `tests/test_phase1_5_ablation.py`
**Changes to project state:** Foundational recurrence adopted (bounded convex interpolation with learned gate).
**Open questions:** Is this recurrence novel? What is the causal driver among boundedness, convexity, and parameterization?
**Next experiment:** Model E Novelty & Mechanism Audit (`docs/reports/model_e_novelty_and_mechanism_audit.md`).

---

## 2026-10-04 — Model E Novelty & Mechanism Audit (7-way mechanism isolation)

**Status:** Completed — literature audit + mechanism decomposition; **no novelty claim** established.
**Research question:** Is Model E a novel recurrent architecture or a rediscovery/variant of prior gated units (UGRNN/MGU); and which ingredient (boundedness, convexity, parameterization) causes the gain?
**Configuration:** Literature equation-matching audit; 7 parameter-matched variants × 5 seeds on the Phase-1 benchmark plus algorithmic generality probes (horizons T=5–30; OOD algorithmic tasks without retraining).
**Key result:** Official classification **Category B** — a known recurrence (Update-Gate RNN / MGU family; Collins et al. 2017; Zhou et al. 2016) with an MLP parameterization variant. Boundedness alone ineffective (Model B ≈ baseline); convex decay is the primary driver (Model C 24.57% ± 3.15% with unbounded candidate); bounded+convex best (43.27% ± 4.23%). Canonical UGRNN 9.52% vs MGU 37.07% — pre-gating supplies bilinear feature filtering. OOD algorithmic tasks show near-zero zero-shot transfer (0–5%): no general-reasoning claim.
**Decision:** Adopt as the official state-update core; make **no novelty claims** for the recurrence equation; document attribution.
**Evidence:**
- report: `docs/reports/model_e_novelty_and_mechanism_audit.md`
- data: `experiments/model_e_mechanism_results.json`
- code: `experiments/run_mechanism_audit.py`
**Changes to project state:** Core formally attributed to known literature; mechanism drivers recorded.
**Open questions:** Which parameterization components (nonlinear mixing, trunk sharing, pre-gating) drive performance? (→ next audit)
**Next experiment:** Parameterization-Isolation Audit (`docs/reports/parameterization_isolation_audit.md`).

---

## 2026-10-04 — Parameterization-Isolation Audit (7 matched variants × 5 seeds)

**Status:** Completed — architectural-driver isolation; **core locked**; includes an FLOP-accounting **correction** of earlier telemetry.
**Research question:** Isolate the architectural drivers of state tracking: nonlinear feature mixing, shared vs independent trunks, MGU-style pre-gating, and audited FLOP efficiency.
**Configuration:** 7 parameter-matched variants (Canonical UGRNN, Independent MLPs, Shared Deep MLP Trunk, Shared Linear Trunk, Canonical MGU, Nonlinear Candidate MGU, Model D), 35 runs across seeds 42–46; corrected FLOP accounting.
**Key result:** (i) **FLOP measurement bug discovered and corrected** (GRU 49,800 → 460,846 FLOPs/step; Model D 452,264) — parity verified; (ii) nonlinear state–observation mixing is decisive (+35.67% vs linear trunk); (iii) trunk sharing does not matter (+0.02%, but ~15% faster training); (iv) MGU pre-gating is a powerful bilinear filter (+26.23% vs UGRNN); (v) **Nonlinear Candidate MGU** reaches **54.28% ± 2.81%** test exact match (+7.02% over Model D) — the best core of the audit.
**Decision:** Lock the Nonlinear Candidate MGU as the base recurrent core; Phase-1 benchmark officially closed at 54.28%; advance toward Phase-2 hypotheses (branching, recursion, verification) now that a sound core exists. *(Report §7.)*
**Evidence:**
- report: `docs/reports/parameterization_isolation_audit.md`
- data: `experiments/parameterization_isolation_results.json`
- checkpoint: `experiments/checkpoints/6_nonlinear_candidate_mgu_seed42_3000steps.pt`
- code: `experiments/run_parameterization_isolation.py`; tests: `tests/test_parameterization_isolation.py`
**Changes to project state:** Official core = Nonlinear Candidate MGU (~226K params); FLOP accounting corrected (amendment precedent for future corrections).
**Open questions:** What is the mechanism of the gate/candidate pair, and where exactly do the remaining ~46% errors live?
**Next experiment:** Mechanism Q&A Diagnostic (`docs/reports/mechanism_qa_diagnostic.md`).

---

## 2026-10-04 — Mechanism Q&A Diagnostic (Phase-1 interrogation)

**Status:** Completed — mechanistic diagnostic investigation.
**Research question:** What is actually stored in the latent state, what the gate does, how the candidate is composed, what the readout can extract, and whether training/compute changes the ceiling.
**Configuration:** Frozen Nonlinear Candidate MGU (seed-42 checkpoint, 212-dim state); probes and controlled measurements on the 1,200-sequence test split; training-budget sweep as an axis.
**Key result:** State carries entangled but linearly decodable register information (~78.2% linear at t=10; MLP probe +2.3 pp); the gate stabilizes near 0.33 and correlates with displacement (r=0.69); **100% retention on untouched registers across 8 intervening steps**; deep readout heads give **0.00% exact-match gain** (linear readout extracts available information); quadrupling training compute moves ~54% → ~58% at 100% train accuracy.
**Decision:** **FREEZE AND ADVANCE.** The remaining errors are not optimization/readout failures; the project specified its single next Phase-2 experiment: **speculative branching K∈{1,2,4} with a terminal scoring/rollout mechanism**, against a matched K=1 baseline. *(Report §8–§9.)*
**Evidence:**
- report: `docs/reports/mechanism_qa_diagnostic.md`
- data: `experiments/qa_diagnostics_results.json`
- figures: `docs/reports/figures/01…08_*.png`
- code: `experiments/run_qa_diagnostics.py`
**Changes to project state:** Phase-1 mechanism description frozen; Phase-2 branching experiment specified (not implemented).
**Open questions:** Do single-step errors (72.5% isolated single-register slips) justify branching, and can any selector pick correct branches?
**Next experiment:** Mechanism Audit Round 2 (`docs/reports/mechanism_audit_round2.md`), which re-interrogated the same core before Phase 2.

---

## 2026-10-04 — Mechanism Audit Round 2 (second-pass interrogation)

**Status:** Completed second-pass audit; decision **FREEZE AND ADVANCE**.
**Research question:** Second-pass mechanistic interrogation of state geometry, gate function, candidate function, error anatomy, and where the remaining ceiling lives.
**Configuration:** Frozen Nonlinear Candidate MGU (seed-42 checkpoint); test-split measurements; probing, ablations, controlled ADD diagnostics, and budget sweeps.
**Key result:** State behaves as a cyclic ~106-effective-dimensional manifold (participation ratio 47.67); gate carries 100% linear op/destination decodability; the candidate is a **delta patch** (untouched registers at chance in the candidate; they survive via the convex residual); the bottleneck is **upstream modular wrap-around arithmetic** (8.00% wrap vs 39.25% no-wrap accuracy); 72.5% of failures are isolated single-register slips in the irreversible single-pass chain.
**Decision:** **FREEZE AND ADVANCE.** Phase-2 target identified empirically: provide parallel speculative states for high-entropy operations (wrap-around) and prune them with a verifier before committing. *(Q20, verbatim.)*
**Evidence:**
- report: `docs/reports/mechanism_audit_round2.md`
- data: `experiments/round2_audit_results.json`
- checkpoint: `experiments/checkpoints/6_nonlinear_candidate_mgu_seed42_3000steps.pt`
- code: `experiments/run_round2_audit.py`
**Changes to project state:** Phase-2 hypothesis narrowed to branching + verification at wrap-around/high-entropy steps.
**Open questions:** Does an oracle selector over sampled candidate states actually contain recoverable terminal failures (and do learned selectors exist)?
**Next experiment:** Final Phase-1 Experiment — Oracle Branching Feasibility (`docs/reports/branching_feasibility_oracle.md`).

---

## 2026-10-05 — Final Phase-1 Experiment: Oracle Branching Feasibility

**Status:** Completed — feasibility probe (offline); **hypothesis not validated**; negative result preserved.
**Research question:** Can multiple speculative candidate states, sampled by injecting stochasticity only into the candidate generator (frozen state, gate, and core), recover errors the deterministic K=1 model cannot?
**Configuration:** Frozen Nonlinear Candidate MGU; K∈{1,2,4,8} nested draws (Gaussian activation noise on candidate MLP hidden; gate never perturbed); branch state `B_k = (1−f)S + f·C_k`; oracle per-step scoring; magnitude-matched random-`S_candidate` perturbation control; one-shot repair; sensitivity sweep.
**Key result:** At K=8 the per-step oracle recovers **241/695 = 34.7%** of deterministic terminal failures (EM 0.5367 → **0.6913**); K=2: 8.2% (0.5693); K=4: 23.3% (0.6360). **However:** each draw decodes written registers at ≈chance (0.094–0.106 vs 0.10) on failures; the matched random-noise control recovers 119/695 (46% of the EM gain; EM 0.6080); a single oracle repair then deterministic continuation reaches only 0.5953 and breaks 22 originally-correct sequences. The selectable objects are rejection-sampling lottery tickets, not structured alternative computations.
**Decision:** **Freeze Phase 1, redesign the Phase-2 hypothesis.** Do not treat the oracle EM bound as achievable; gate perturbation is invalid; any revisiting of branching must use a *learned, compute-parity-audited selector*. Q9 of the report specified the minimum next experiment (learned selector on frozen candidate sets).
**Evidence:**
- report: `docs/reports/branching_feasibility_oracle.md`
- data: `experiments/branching_feasibility_results.json`
- checkpoint: `experiments/checkpoints/6_nonlinear_candidate_mgu_seed42_3000steps.pt`
- code: `experiments/run_branching_feasibility.py`
**Changes to project state:** Branching not adopted; oracle numbers recorded as oracle-only facts; a selector feasibility study prescribed.
**Open questions:** Can a learned selector/verifier recognize correct candidates without ground truth?
**Next experiment:** Learned-Verifier Feasibility (`docs/reports/verifier_feasibility.md`) — executed as the prescribed selector probe.

---

## 2026-10-05 — Learned-Verifier Feasibility (prescribed Phase-2-gated selector probe)

**Status:** Completed — feasibility study; **strong negative result**; verifier/branching abandoned for now.
**Research question:** Can a learned selector identify better candidates without access to ground truth, using only inference-available features over frozen candidate sets (K=4 primary)?
**Configuration:** Frozen core/encoder/candidate network/gate/readout; the same stochastic candidate mechanism as the branching experiment; disjoint verifier train/val (streams 2042/51042) and the untouched test split (90042); linear + small MLP selectors; ranking metrics, trajectories, calibration, ablations, noise control.
**Key result:** Headline metrics are artifacts: MLP AUROC 0.879 and top-1 0.7747, but the always-commit-deterministic baseline is 0.7731 (headroom to oracle 0.8179); decision-relevant (det-wrong) AUROC 0.6388 and rescue rate 5.85% vs 5.3% random; the same AUROC (0.8766) transfers to matched random noise (decision-relevant 0.5179); end-to-end: K=1 0.5367 > learned K=4 (linear 0.5293, MLP 0.4940) > random K=4 0.1707; oracle-only 0.6247; 6 magnitude features reproduce 89% of full-model AUROC. The learnable optimum is "commit the deterministic candidate".
**Decision:** **Abandon verifier/branching for now (option 3).** The bottleneck is information/representation upstream in the core, not selection; return to the generator. No Phase-2 machinery.
**Evidence:**
- report: `docs/reports/verifier_feasibility.md`
- data: `experiments/verifier_feasibility_results.json`, `experiments/verifier_feasibility_preds.npz`
- figures: `docs/reports/figures/verifier_reliability.png`, `verifier_methods_em.png`
- checkpoint: `experiments/checkpoints/6_nonlinear_candidate_mgu_seed42_3000steps.pt`
- code: `experiments/run_verifier_feasibility.py`
**Changes to project state:** Branching/verification deferred; effort redirected to the frozen core's own error structure.
**Open questions:** Is the ~54–58% terminal-EM ceiling caused by training objective rather than the transition mechanism?
**Next experiment:** Generator-Repair Feasibility — training-objective comparison (`docs/reports/generator_repair_feasibility.md`).

---

## 2026-10-05 — Generator-Repair Feasibility (Supervision Objectives A–F)

**Status:** Completed — feasibility study; **strong positive result for dense supervision**.
**Research question:** Is the ~54–58% ceiling caused by terminal-only supervision rather than by the recurrent transition itself? (Test the same frozen architecture under different training objectives only.)
**Configuration:** Nonlinear Candidate MGU (226,358 params, frozen architecture); identical protocol (train 6,000 / val 1,200 / test 1,200, base_seed=seed; 3,000 steps; AdamW lr 1e-3, wd 0.01; batch 64; grad-clip 1.0; seeds 42–46). Variants: A terminal-only; B full stepwise; C1–C3 hybrids (λ=0.75/0.50/0.25); D late-step (t≥6); E candidate/local written-register auxiliary loss; F combined (terminal + stepwise + candidate).
**Key result:** T=10 EM: A 0.540 ± 0.018; B 0.672; C1 0.662; C2 0.629; C3 0.591; D 0.699; **E 0.755 ± 0.019** (T=20 0.407; per-every-seed ≥ 0.72). Dense supervision lifts every horizon and operation; E's candidate written-register accuracy 0.956 vs 0.830 (A); isolated-ADD candidate accuracy 0.364 → 0.798; ADD-wrap 0.140 → 0.485; first real-error rate 0.559 → 0.303; drift halved. E = strongest individual objective; residuals concentrate in wrap arithmetic, repeated ADD, and a candidate→commit mismatch.
**Decision:** Dense supervision is a major limitation of the previous baseline. Adopt better supervision and continue Phase 1, pending a combined-objective confirmation of the two complementary signals (B/D-style stepwise + E-style candidate).
**Evidence:**
- report: `docs/reports/generator_repair_feasibility.md`
- data: `experiments/generator_repair_results.json`, `experiments/generator_repair_supplement.json`
- figures: `docs/reports/figures/generator_repair_{horizons,stepwise,em}.png`
- code: `experiments/run_generator_repair_feasibility.py`, `experiments/analyze_generator_repair_supplement.py`
**Changes to project state:** Supervision objectives become the active research axis; architecture stays frozen.
**Open questions:** Are stepwise commitment calibration and candidate-local supervision complementary? Does the combined objective dominate E?
**Next experiment:** Combined-Objective Confirmation (`docs/reports/combined_objective_confirmation.md`).

---

## 2026-10-05 — Combined-Objective Confirmation (Model F)

**Status:** Completed — confirmation experiment (F ≈ E on the primary metric; **E adopted**; objective sweeps stop).
**Research question:** Do the two successful supervision signals combine — does F = terminal + stepwise + candidate-local supervision beat E on the primary terminal metric and close the candidate→commit gap?
**Configuration:** One new objective F (`L = CE(y_T) + 1.0·(1/T)Σ_t CE(head(S_t)) + 1.0·mean_{written} CE(head(C_t))`); same protocol and 5 seeds; E/D references from the previous experiment reused exactly; loss-component scales audited (terminal 0.375 / stepwise 0.217 / candidate 0.262).
**Key result:** F T10 0.737 ± 0.015 vs E 0.755 ± 0.019 (paired Δ = −1.77 pp; 4/5 seeds negative; p = 0.17); T20 0.390 vs 0.407 (p = 0.24); T15 −2.18 pp (p = 0.017). F **does** eliminate ghost readout errors (T1 = 1.000, zero ghosts; strict first-error 0.341 vs E 0.425) and slightly improves committed full-state decoding (0.903 vs 0.891), and dominates D on every axis — but it does not beat E on the primary/secondary metrics and does not close the ADD candidate→commit gap (isolated +0.140 vs E +0.128).
**Decision:** Per the stated rule, **adopt candidate/local supervision (Model E) as the official Phase-1 training objective**; F's combined objective is documented as a capability (deghosting) but not adopted; transition-level objective sweeps stop. Next target: the ADD candidate→commit residual.
**Evidence:**
- report: `docs/reports/combined_objective_confirmation.md`
- data: `experiments/combined_objective_confirmation_results.json`, `experiments/combined_objective_analysis.json`, `experiments/combined_objective_supplement.json`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_generator_repair_feasibility.py` (variant `F_combined`), `experiments/analyze_combined_objective.py`
**Changes to project state:** E is the official objective (exact loss recorded in the report and in `docs/RESEARCH_STATUS.md`); architecture and checkpoints frozen.
**Open questions:** Is the ADD candidate→commit gap caused by gate dilution or by blend/readout failure?
**Next experiment:** Experiment G — Localizing the ADD candidate→commit gap (`docs/reports/experiment_g_add_commit_gap.md`).

---

## 2026-10-05 — Experiment G: Localizing the ADD Candidate→Commit Gap

**Status:** Completed — measurement-only localization (no training; frozen E/F checkpoints); classification **C (both mechanisms present)**.
**Research question:** When a candidate state contains the correct ADD result but the committed state does not decode it, is the failure primarily gate dilution (transfer too small) or blend/readout failure (partial blends undecodable)?
**Configuration:** Frozen E_candidate (official) and F_combined (control), seeds 42–46; strict test split + existing controlled ADD builders; 31,763 ADD transitions per model; analysis restricted (primary) to candidate-correct general-stratum adds (dest ≠ src, src operand > 0); transition-level tables descriptive, seed-level paired inference.
**Key result:** 5.5% of candidate-correct ADDs commit wrong (E 764/13,958; F 770/13,940), with **96% of failures in the general stratum**; no-op and same-register adds nearly gap-free. Within the general stratum the gap rate falls monotonically with the mean gate (34.7% → 16.3% → 1.2% across bins) — but only ~30% of failures decode to the old value while 68–71% decode to a third value; displacement predicts pooled failures better than the gate; the gate-vector mean is only a proxy (no register-specific gate coordinate exists). Paired E/F show an identical residual (303 vs 299 E/F-only failures, p = 0.96).
**Decision:** **C — both gate dilution and blend/readout materially contribute**; no causal claim. Next measurement must respect the fact that the actual commit is an element-wise vector operation that the scalar mean-gate proxy cannot represent.
**Evidence:**
- report: `docs/reports/experiment_g_add_commit_gap.md`
- data: `experiments/experiment_g_transitions.npz`, `experiments/experiment_g_transitions_meta.json`, `experiments/experiment_g_results.json`
- reconciliation reference: `experiments/combined_objective_supplement.json` (per-op supplement used to cross-check the collector)
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_g.py`, `experiments/analyze_experiment_g.py`
- tests: `tests/test_experiment_g.py`
**Changes to project state:** Residual localized; scalar gate proxies flagged as descriptive only.
**Open questions:** Does the old→candidate state path contain a decodable point, and does the actual vector-gate commit reach it?
**Next experiment:** Experiment H — λ-interpolation decode sweep (`docs/reports/experiment_h_lambda_interpolation.md`).

---

## 2026-10-05 — Experiment H: λ-Interpolation Decode Sweep of the ADD Commitment

**Status:** Completed — measurement-only (no training; frozen E/F checkpoints); classification **C (both)**, sharpened by the vector-gate finding.
**Research question:** When an ADD candidate is correct but the committed state is wrong, does the old→candidate state path contain a point the existing readout can decode — and is the actual element-wise commit equivalent to any scalar point on that path?
**Configuration:** Frozen E/F, seeds 42–46; 21-point scalar sweep `head((1−λ)S_prev + λC)`; **actual vector-gate replay** `head((1−f)⊙S_prev + f⊙C)` verified bit-exactly against recorded states; mean-gate counterfactual `head((1−f̄)S + f̄C)` (analysis-only); λ* (first correct λ); λ_eff (projection of the committed state onto the segment); trajectory and near-miss classes; all sanity checks passed; deterministic rerun verified.
**Key result:** Scalar path accuracy is monotone and smooth but decodable only near the candidate end (E: 0.20 at λ=0.4 → 0.90 at λ=0.8 → 1.0 at λ=1.0); on failures, median λ* = 0.80 and only 1.2% are decodable at λ ≤ 0.5; wrong slots stay at the old value (92%). **The vector-gate structure is decisive:** replay = recorded commit (0.855/0.853) while the mean-gate scalarization collapses to 0.182/0.042 (decode disagreement 77%/90%); 99.7%/99.9% of failures have λ* > λ_eff, so in scalar projection the commit stops far short of the decodable band — but the actual state is off-path (mean distance ≈ 6.3 vs segment ≈ 15), making the scalar comparison descriptive only. Candidate-wrong transitions: 78–82% admit no correct λ at all.
**Decision:** **C — both mechanisms present**, with the added finding that any commit-side reasoning must be **coordinate-wise**; scalar mean-gate reasoning is exhausted. No fix implemented.
**Evidence:**
- report: `docs/reports/experiment_h_lambda_interpolation.md`
- data: `experiments/experiment_h_interp.npz`, `experiments/experiment_h_interp_meta.json`, `experiments/experiment_h_results.json`
- figures: `docs/reports/figures/experiment_h_{acc_vs_lambda,acc_vs_lambda_wrap,acc_vs_lambda_failures,lambda_star_dist,lambda_star_vs_dcs,lambda_star_vs_gate,replay_vs_scalar,trajectories,category_gap}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_h.py`, `experiments/analyze_experiment_h.py`
- tests: `tests/test_experiment_h.py`
**Changes to project state:** Scalar gate summaries formally demoted to descriptive proxies; the next measurement is specified at the coordinate level.
**Open questions:** Which coordinates does the gate move, how much of the readout's decision-direction movement is delivered on failures vs successes, and is the residual under-movement or readout precision?
**Next experiment:** **Experiment I — coordinate-wise commit analysis (gate-vector level; measurement only)** — specified in the Experiment H report §14; not yet executed.

---

*End of ledger (initial population at policy adoption, 2026-10-05). All future entries append below this line.*

---

## 2026-10-06 — Experiment I: Coordinate-Wise Decision-Margin Commit Analysis

**Status:** Completed — measurement-only (no training; frozen E/F checkpoints; classification **C (both)**).
**Research question:** When the ADD candidate is correct but the committed state is wrong, does the element-wise gate fail to deliver enough movement on the decoder's actual decision boundary (commit/gate under-movement), or does the frozen readout require unusually precise latent alignment even when useful decision-direction movement has been delivered (readout/representation precision)?
**Configuration:** Frozen `E_candidate` (official) and `F_combined` (control), seeds 42–46; strict test split + the five controlled ADD builders; 31,763 ADD transitions per model. Decision margin `m(S) = z_y(S) − z_a*(S)` against the strongest competitor `a*` chosen at `S_(t-1)` (readout is a single linear layer, so margins and the coordinate decomposition `q_i = (w_y−w_a*)_i f_i (C_i−S_prev_i)` are exact); `ρ_margin = Δm_commit/Δm_candidate`; exact `α*_margin`/`α*_full` along `S_prev+α(C−S_prev)`. Primary set: candidate-correct ∧ general-stratum ∧ test (E 5,042 / F 5,032). Transition-level tables descriptive; inference seed-level (n=5, paired).
**Key result:** Under-movement is present and seed-consistent: on failures `ρ_margin` is 0.780 (E) / 0.798 (F) vs 0.897 / 0.903 on successes (paired fail−succ dz −6.34 / −5.96, 5/5), the positive-contribution-weighted gate is lower on failures (0.617 / 0.623 vs 0.702 / 0.712) while the negative-contribution gate is higher, absolute delivered margin is lower, and candidate margins are about half (median 4.38 / 3.86 vs 8.52 / 8.35). But 44.1% / 47.7% of failures delivered at least the exact scalar-equivalent transfer needed (`ρ_margin ≥ α*_full`) yet still failed, and the strongest competitor changes at the commit in 66–70% of failures — a readout/off-path placement component. `α*_full` reproduces Experiment H's λ* (Spearman ≥ 0.994). E and F are statistically indistinguishable on this residual (failure rate 0.145 vs 0.147, p = 0.84).
**Decision:** **C — both mechanisms materially contribute.** Under-movement (gate delivery + weak candidate margin) is the larger, more consistent component; the readout/placement (off-path) account is required for the ≈45% of failures with sufficient scalar-equivalent delivery. No causal claim; no fix implemented.
**Evidence:**
- report: `docs/reports/experiment_i_coordinate_commit.md`
- data: `experiments/experiment_i_coordinate.npz`, `experiments/experiment_i_coordinate_meta.json`, `experiments/experiment_i_results.json`
- figures: `docs/reports/figures/experiment_i_{margin_dist,avail_vs_deliv,rho_success_fail,alpha_star_dist,alpha_vs_delivered,coord_contrib,weighted_gate,seed_movement,E_vs_F,failure_classes,align_ratio,categories}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_i.py`, `experiments/analyze_experiment_i.py`
- tests: `tests/test_experiment_i.py` (11 tests; α* closed form verified against a 6,001-point brute-force scan; deterministic-rerun verified)
- reused unchanged: `experiments/experiment_g_transitions.npz`, `experiments/experiment_h_interp.npz` (row-aligned, re-verified)
**Changes to project state:** The coordinate-wise/decision-margin decomposition is now the validated description of the ADD candidate→commit residual; the residual is bounded as mixed (under-movement + off-path placement). Scalar gate analysis remains demoted (H stands).
**Open questions:** Would a controlled change to the commit rule reduce the under-movement component, and do the off-path failures (delivered ≥ scalar-equivalent needed) persist under it?
**Next experiment:** **Experiment J — controlled inference-time commit-rule intervention on the frozen E checkpoints (interventional, compute-parity-audited)**, respecting the Round-2/branching negative result that naive gate perturbation is not a valid mechanism; specified, not started.

---

## 2026-10-06 — Experiment J: Controlled Inference-Time Commit-Rule Intervention

**Status:** Completed — interventional / causal test (frozen E primary and F control checkpoints; no training; no weight modification); classification **B (partial causal support)**.
**Research question:** If we intervene only on the actual element-wise commit gate `f_t ∈ R^212` (architecture, weights, readout, data, optimizer frozen), can we recover sequence-level capability — or is the Experiment I under-movement residual a local symptom rather than the sufficient cause?
**Configuration:** Frozen E (primary) and F (control), seeds 42–46. Pre-registered primary family `f' = clip(γ f, 0, 1)` on the full gate vector over `γ ∈ {0.75,0.90,1.00,1.10,1.25,1.50,2.00}`; one mean-matched coordinate-scrambled control; one pre-registered secondary monotone family `f' = 1−(1−f)^γ`. Complete **recursive** rollout (gate and candidate recomputed from the modified state). Evaluated at T ∈ {1,2,3,5,8,10,15,20} (canonical T=10 test + horizon sets), 31,763-transition-style transition/margin analysis, operation-level EM, controlled ADD builders, candidate-margin strata, and rescue/regression vs γ=1. 140 config-runs; γ=1 reproduces the frozen forward (decoded states and terminal predictions bit-identical); the whole grid was run twice with 0/140 record mismatches.
**Key result:** Increasing γ monotonically raises decision-margin delivery (E ρ_margin 0.884→0.930, gw_pos 0.690→0.743; paired dz +15 to +59, 5/5) and yields a small net candidate→commit rescue only in a narrow window (net +5.0 transitions/seed at γ=1.10, +1.0 at 1.25; negative beyond), but **no γ>1 improves T=10 or T=20 EM** and γ≥1.25 significantly degrades it (E T=10 0.7525→0.7107, T=20 0.4064→0.3350 for γ=1.10→2.00), worsening wrap/repeated-ADD and SET/SWAP, and breaking more baseline-correct sequences than it rescues. γ<1 (transfer reduction) is strongly harmful. A mean-matched coordinate-scrambled gate collapses EM to ≈0.01, so coordinate identity is essential. Of baseline failures with sufficient scalar-equivalent transfer, only 6–23% are rescued under any γ.
**Decision:** **B — partial causal support.** The commit/gate is causally involved in the local ADD candidate→commit residual (a commit-only intervention moves margin delivery and rescue as predicted) but it is not the dominant long-horizon bottleneck, and untuned global gate amplification is not an adequate commit intervention; the learned element-wise gate performs a balancing function. Interventional with respect to the commit rule only; not a fix.
**Evidence:**
- report: `docs/reports/experiment_j_commit_intervention.md`
- data: `experiments/experiment_j_records.json`, `experiments/experiment_j_results.json`
- figures: `docs/reports/figures/experiment_j_{em_vs_gamma,paired_seed_changes,rescue_regression,commit_gap,margin_vs_gamma,weighted_gate_vs_gamma,margin_strata,add_strata,op_regression,rescue_examples,coord_vs_scramble,compute_parity}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_j.py`, `experiments/analyze_experiment_j.py`
- tests: `tests/test_experiment_j.py` (15 tests)
**Changes to project state:** The commit/gate is established as a genuine but sub-dominant local bottleneck; global magnitude scaling of the gate is rejected as a recovery mechanism; coordinate identity of gate delivery is established as functionally essential.
**Open questions:** Can a routing-preserving (balance-aware) commit change help where uniform magnitude scaling could not? Does the off-path placement component require a readout-side or representation-side change rather than a commit-side one?
**Next experiment:** **Experiment K — commit-rule "balance" ablation (routing-preserving monotone remap of `f_i` about 0.5, inference only, frozen checkpoints, pre-registered)**, testing whether a coordinate-preserving remap that does not uniformly amplify transfer can recover the local residual without the sequence-level and operational regressions seen under uniform scaling; specified, not started.

---

## 2026-10-07 — Experiment K: Coordinate-Preserving Gate-Balance Ablation

**Status:** Completed — interventional / inference-only (frozen E primary and F control checkpoints; no training; no weight modification); classification **B (locally beneficial, sequence-level neutral)** for the primary model E; the F control touches the harmful band at T=20 and is reported as a control only.
**Research question:** Does changing the SHAPE / BALANCE of the existing element-wise gate — a coordinate-preserving monotone remap of `f_i` about 0.5 that does not simply increase aggregate transfer everywhere — improve inference-time performance, given that Experiment J showed uniform magnitude amplification does not?
**Configuration:** Frozen E (primary) and F (control), seeds 42–46. Pre-registered **two-condition** intervention `f'_i = sigmoid(kappa · logit(clip(f_i, eps, 1-eps)))`, `eps = 1e-6`, `kappa = 2.0` vs the identity `kappa = 1.0` (the identity is returned exactly); applied to the **full 212-dim gate vector**, then a complete recursive rollout `S'_t = (1−f'_t)⊙S'_(t-1) + f'_t⊙C_t` with gate and candidate recomputed from the modified state (reusing Experiment J's `rollout`). The intervention depends only on `f_i` (no label/logit/margin/register/operation/candidate-correctness/analysis information). No kappa grid search; no new scramble control (Experiment J already established coordinate scrambling is catastrophic). Evaluated at T ∈ {1,2,3,5,8,10,15,20}, transition-level decision-margin and exact coordinate decomposition, gate-shape statistics, operation-level EM, controlled ADD builders, candidate-margin strata, off-path subset, and rescue/regression vs kappa=1. 20 config-runs; kappa=1 reproduces the frozen forward (states/candidates/gates/decodes/terminal identical); the whole configuration was run twice with 0/20 record mismatches.
**Key result:** The remap changes gate *shape*, not magnitude (E: Δ mean(f) = −0.0035, Δ median(f) = −0.0235, Δ entropy = −0.0827; saturation tails grow, mid-range shrinks; mean transfer does **not** increase). It *raises* delivery on the decision-helping coordinates and *lowers* delivery on the harmful ones — the favorable profile Experiment I identified — unlike uniform amplification, which raised the harmful-coordinate gate monotonically (J γ=1.0→2.0: gw_neg 0.2698→0.2843→0.2994→0.3451). On the fixed baseline candidate-correct general-ADD set (E): ρ_margin 0.8836→0.9022 (dz +12.15, 5/5), gw_pos 0.6900→0.6977 (dz +7.71, 5/5), gw_neg 0.2698→**0.2637** (dz −4.25, 5/5), Δm_commit +0.369 (dz +14.04, 5/5), margin shortfall 2.078→1.761 (dz −7.45, 5/5). **But** no sequence endpoint improves: E T=10 0.7547→0.7528 (−0.18 pp, dz −0.52, 4/5, p=0.31), T=20 0.4068→0.4122 (+0.54 pp, dz +0.71, 3/5, p=0.19); net transition recovery −7.6/seed (rescued 14.6 vs regressed 22.2; dz −1.50, 5/5); sequence rescue 14.8 vs breakage 17.0; primary committed accuracy *falls* slightly 0.8550→0.8474 (5/5, p=0.033) while margin delivery rises; first-real-error rate rises slightly (0.3030→0.3078, 5/5). No significant operation-level regression (SET/SWAP/ADD-no-wrap/ADD-wrap all null); the controlled wrap builder improves (+6.75 pp, 5/5, p=0.035). Off-path failures (ρ≥α*_full, n≈63/seed) survive (6.35% rescued; 100% competitor change).
**Decision:** **B — locally beneficial but sequence-level neutral.** The gate-shape remap is a genuine, seed-consistent causal handle on decision-margin delivery and on the helpful/harmful delivery split, and is operationally safe at the category level, but it does not improve T=10/T=20 EM and its transition-level collateral slightly exceeds its benefit. The commit rule is **frozen**: neither magnitude (J) nor shape (K) commit-side intervention recovers sequence capability; no search over remapping functions will be opened. Interventional with respect to the commit rule only; not a fix.
**Evidence:**
- report: `docs/reports/experiment_k_gate_balance.md`
- data: `experiments/experiment_k_records.json`, `experiments/experiment_k_results.json`
- figures: `docs/reports/figures/experiment_k_{t10_t20_em,horizons_em,rescue_regression,gate_distribution,gate_entropy_shape,margin_delivery,gw_pos_vs_neg,commit_gap,margin_strata,add_ops,set_swap_regression,offpath_rescue,coord_gate_change,vs_j_reference}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_k.py`, `experiments/analyze_experiment_k.py`
- tests: `tests/test_experiment_k.py` (18 tests)
- historical reference (read only, unchanged): `experiments/experiment_j_results.json`
**Changes to project state:** The gate's *shape/balance* is established as a second, distinct causal handle on the local commit residual (alongside magnitude in J); a favorable delivery profile (helpful up, harmful down) is demonstrated to be achievable **without** net amplification yet still does not convert into sequence capability; commit-rule modification is frozen; the off-path placement component is confirmed to be unreachable from the commit side.
**Open questions:** Is the off-path placement component (baseline failures with ρ_margin ≥ α*_full, ~100% competitor change) a frozen-readout bias/threshold property or a genuine latent misplacement? Does the remaining long-horizon error budget lie in wrap arithmetic, repeated-ADD chains, and candidate generation upstream of the commit?
**Next experiment:** **Experiment L — readout-side localization of the off-path component (measurement-only, frozen readout)**: decompose the off-path failures into a readout-bias term and a latent state-projection term using the frozen linear head, to determine whether the placement component is a readout calibration property or a representation property; specified, not started.

---

## 2026-10-07 — Experiment L: Readout-Side Localization of the Off-Path Component

**Status:** Completed — measurement only (frozen E primary and F control checkpoints; no training; no readout/gate/commit modification; no intervention; no recursive counterfactual trajectories). Classification: **B — primarily latent-state placement** (E primary; F control agrees).
**Research question:** When the candidate is correct and the committed state contains substantial useful decision movement, why can the frozen linear readout still decode the wrong value — readout-side bias/calibration, or latent-state placement geometry?
**Configuration:** Frozen E (primary) and F (secondary control), seeds 42–46, strict test split (1,200 × 10), 15,763 ADD transitions/model. Primary population = strict test | ADD | general (`dest ≠ src` AND source operand `> 0`) | candidate correct | baseline commit wrong | `ρ_margin ≥ α*_full` (n = 323 E / 353 F). The frozen readout is a single `nn.Linear(212→40)`; its exact decomposition `z_{d,v}(S) = W[d,v]·S + b[d,v]` was verified (`head(x) = W x + b` to float32). Per transition the collector recorded the state margins `(w_y−w_a)·S`, bias margins `b_y−b_a` (at `S_prev`/`C`/`S_commit`, for both the S_prev reference `a_prev` and the committed competitor `a_commit`), bias-removed argmax decodes, pairwise decision-direction dot products, normalized hyperplane distances, and Experiment H's `λ_eff`/`offpath_dist`. Analyses 1–10 plus deterministic per-transition categories and a pre-registered three-way decision (bars `r_bias ≥ 0.30`, `r_place ≥ 0.60`, fixed in code before the aggregate). All counterfactuals are one-step analysis-only computations on frozen states. Collector bit-identical across two runs; analysis JSON identical (excluding timestamp); G/H/I row alignment exact (max |Δ| = 0.0).
**Key result:** The off-path failures are **latent-state placement**, not readout bias/calibration. In **100%** of E off-path failures the committed state **clears the strongest competitor that was binding at `S_prev`** (`m(S_commit; a_prev) ≥ 0`, median **+1.96**) yet the committed argmax is a **different** competitor in **100%** of cases and decodes to a **third value** in **100%** of cases (never the old value). The fixed bias is nearly never decisive: removing it (analysis only) flips only **1.24% (E) / 3.12% (F)** to correct, and the bias accounts for a median **7.1%** of the winning-pair margin. The state-projection alone is negative against the committed competitor in 98.5% of cases. `off-path distance` does **not** distinguish off-path failures from successes (6.57 vs 6.36; ratio 0.415 vs 0.420). Perturbations around `S_commit` rarely reach the correct basin (perturbed-correct 1.7% / 7.9% / 13.1% at ε = 0.01 / 0.05 / 0.10) — the state is locally stable inside the wrong basin. Category fractions (E / F): boundary_shortfall 0.0% / 0.0%, readout_calibration 1.2% / 3.1%, latent_misplacement 98.8% / 96.9%, mixed 0.0% / 0.0%. E and F are statistically indistinguishable on every signature (paired p ≥ 0.40).
**Decision:** **B — primarily latent-state placement.** The off-path residual is a placement property of the frozen state trajectory (the committed state enters a different competitor's basin after clearing the intended one); the simple "readout offsets are wrong" explanation is ruled out. Measurement only, no causal claim. Redirects the next step to representation/state geometry (not a readout fix).
**Evidence:**
- report: `docs/reports/experiment_l_readout_localization.md`
- data: `experiments/experiment_l_readout.npz`, `experiments/experiment_l_readout_meta.json`, `experiments/experiment_l_results.json`
- figures: `docs/reports/figures/experiment_l_{decomposition,margin_progression,bias_vs_state,offpath_dist,hyperplane_dist,competitor_switch,margin_dist,projection_residual,categories,value_breakdown,E_vs_F,perturbation}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_l.py`, `experiments/analyze_experiment_l.py`
- tests: `tests/test_experiment_l.py` (17 tests)
- reused unchanged (read only): `experiments/experiment_g_transitions.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_i_coordinate.npz`
**Changes to project state:** The off-path candidate→commit residual is localized to **latent-state placement**; the readout-bias/calibration explanation is measured away as the dominant cause; `off-path distance` is recorded as non-distinguishing; the next measurement moves to representation geometry. No fix; commit rule remains frozen.
**Open questions:** Is the third basin a systematic **modular/wrap** geometry (~55% of off-path failures are wrap) or an unsystematic off-axis collapse? Does the (weaker) off-path candidate margin partly drive the placement? What representation-side quantity would need to change?
**Next experiment:** **Experiment M — representation geometry of the third basin (measurement-only, frozen checkpoints)**: identify the exact committed (wrong) competitor and its relation to the task arithmetic, test whether the winning basin is systematically the wrap/modular region, and measure whether the off-axis displacement is a consistent direction — so any later representation-side change is grounded in a measured mechanism; specified, not started.

---

## 2026-10-07 — Experiment M: Representation Geometry of the Third Basin

**Status:** Completed — measurement only (frozen E primary and F control checkpoints; no training; no optimizer; no weight/readout/gate/commit modification; no intervention; no recursive counterfactuals; no learned classifier as primary evidence). Classification: **M-A — systematic modular geometry** (E primary; F control agrees), with the qualification that the modular axis is **parity (mod 2)**, not wrap-around and not ±1 adjacency.
**Research question:** Is the third (wrong) basin that Experiment L localized a systematic arithmetic/modular geometry — in particular wrap-around or an adjacent modular value — or an unsystematic off-axis representation collapse?
**Configuration:** Frozen E (primary) and F (control), seeds 42–46, strict test split (1,200 × 10). Exact Experiment-L primary population (not redefined): strict test | ADD | general (`dest ≠ src` AND `operand_b > 0`) | candidate correct | baseline commit wrong | `ρ_margin ≥ α*_full` → n = 323 (E) / 353 (F); controls: candidate-correct successes (4,310 / 4,292) and candidate-correct non-off-path failures (409 / 387). Per transition the collector captured the flat 40-way committed winner `(register, value)`, the arithmetic mapping of the wrong destination value, the candidate/commit off-axis decomposition, and the true-vs-flat-winner margins. `d_mod(a,b) = min((a-b) mod 10, (b-a) mod 10)`; the wrong-value reference is the empirical true-value distribution (not uniform); the same-parity rate and the directional resultant R are tested against permutation/within-seed sign-flip nulls. Pre-registered `modular_identity = same-parity ≥ 0.60 AND perm p ≤ 0.01`; `wrap_specific = TV(wrap,no-wrap) ≥ 0.30`; `directional_systematic = seed-level R ≥ 0.15 AND every seed p ≤ 0.01`. M/L row alignment exact (max |Δ| = 0). Collector bit-identical across runs; analysis JSON identical (excluding timestamp).
**Key result:** The third basin is a **systematic modular geometry**, but the modular axis is **parity**, not wrap. The wrong destination value **preserves the target's parity** in **73.7% (E) / 64.0% (F)** vs a **50% permutation null (p ≈ 0)** — i.e. an even modular offset (±2, ±4 dominate; |offset| ∈ {2,4} = 73.7% / 64.0%), **not** ±1 (only 9.9% / 11.0%). Wrap status does **not** change the wrong-value identity (TV 0.240 / 0.212 < 0.30 bar; mean d_mod to target 3.09 vs 3.19 wrap/no-wrap) → the wrap/carry hypothesis is **not supported**. The off-axis committed displacement is **directionally consistent within each model** (seed-level R = 0.256 ± 0.013 (E) / 0.306 ± 0.017 (F); all 5 seeds p ≈ 0 vs a within-seed sign-flip null; pooled R only 0.110 / 0.138 because the checkpoints' readout geometries differ), mildly arithmetic-localized (each wrap/no-wrap group aligns more with its own mean direction; within-operand R ≈ 0.19–0.27). The wrong *register* is **not** localized (flat winner == dest ≈ 0%, "neither" 84% / 89%). The third basin is **deep** (normalized true-vs-flat-winner distance −2.27 / −2.23 vs −0.90 / −0.84 for successes). Off-path magnitude (off-axis ratio 0.415 / 0.416) does **not** separate groups (re-confirming L). E and F both classify M-A (same-parity 0.741 / 0.640; seed R 0.256 / 0.306).
**Decision:** **M-A — systematic modular geometry.** The off-path third basin is a reproducible, seed-consistent modular (parity) region with a seed-consistent off-axis direction; it is **not** an unsystematic collapse (M-C), not a non-modular placement (M-B), and not wrap/adjacency. Measurement only; **no causal claim**. Central unresolved: **where** the parity structure lives — the committed **state** representation or the frozen **readout's value geometry** — which M does not attribute; a representation-side intervention is therefore **not yet** justified.
**Evidence:**
- report: `docs/reports/experiment_m_representation_geometry.md`
- data: `experiments/experiment_m_geometry.npz`, `experiments/experiment_m_geometry_meta.json`, `experiments/experiment_m_results.json`
- figures: `docs/reports/figures/experiment_m_{wrong_value_hist,wrong_offset,modular_distance,wrap_split,wrong_register,offaxis_decomposition,directional_consistency,offaxis_arithmetic,third_basin_depth,candidate_vs_commit,control_comparison,E_vs_F}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_m.py`, `experiments/analyze_experiment_m.py`
- tests: `tests/test_experiment_m.py` (19 tests)
- reused unchanged (read only): `experiments/experiment_l_readout.npz`, `experiments/experiment_i_coordinate.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_g_transitions.npz`
**Changes to project state:** The third basin is characterized as a **parity-organized modular region** with a seed-consistent (weak) off-axis direction and deep basin entry; the wrap/carry and ±1-adjacency hypotheses are rejected; the wrong register is not localized. The next measurement must attribute the parity structure to the state vs the frozen readout before any intervention. No fix; commit rule frozen.
**Open questions:** Is the parity structure a property of the committed **state geometry** or of the frozen **readout value-geometry**? Is the modest off-axis direction state-geometry or readout-geometry? Does the weaker off-path candidate margin contribute to the parity-organized placement?
**Next experiment:** **Experiment N — parity-origin localization (measurement-only, frozen checkpoints)**: measure the frozen readout's 40 class weight-vector geometry for parity organization, project `S_commit`/`C`/`S_prev` onto parity-organized decision axes, and decompose the same-parity rate into a readout-geometry component and a state-displacement component (with pre-declared state/readout permutation controls), to decide whether a later intervention should target the readout or the representation; specified, not started.

---

## 2026-10-07 — Experiment N: Parity-Origin Localization

**Status:** Completed — measurement only (frozen E primary and F control checkpoints; no training; no optimizer; no weight/readout/gate/commit modification; no intervention; no recursive counterfactual rollout; no learned classifier as primary evidence). Classification: **N-B — state-dominant parity alignment** (E primary; F control agrees).
**Research question:** Does the same-parity signature M found live in the frozen readout's value-geometry (N-A), in the committed state↔label alignment (N-B), in both (N-C), or is it not localizable (N-D)?
**Configuration:** Frozen E (primary) and F (control), seeds 42–46, strict test split. Exact M/L population (not redefined): off-path n = 323 (E) / 353 (F); controls = candidate-correct successes (4,310 / 4,292) and non-off-path failures (409 / 387). The primary parity axis is derived from **W alone** (`p = normalize(μ_even − μ_odd)` over normalized value weight vectors); the W-only same−different-parity cosine effect is tested against a within-register value-label permutation null (2,000 perms, per seed); pairing-break controls permute the arithmetic labels relative to the fixed decode (readout-only) and the decode relative to the labels (state-permutation) within seed × destination stratum. Pre-registered: `W_strong = mean effect ≥ 0.05 AND ≥4/5 seeds p ≤ 0.05`; `readout_fraction = (null_pair − 0.5)/(obs − 0.5)`; N-A if `W_strong and readout_fraction ≥ 0.50`, N-C if `W_strong and both fractions ≥ 0.30`, N-B if `alignment_fraction ≥ 0.60`, else N-D. N/M row and vector alignment exact (max |Δ| = 0). Collector bit-identical across runs; analysis JSON identical (excluding timestamp).
**Key result:** The frozen readout **does** contain a small, reproducible within-register parity organization (mean pairwise cosine same −0.042 vs different −0.152; **effect +0.110 (E) / +0.111 (F)**, significant in **4/5** seeds, p ≈ 0.006–0.011; cross-register effect only +0.024, so it is a value property not register structure; bias shows no parity structure). **But** it produces almost no state-independent same-parity preference: the pairing-break (readout-only / state-permutation) null is **0.521 / 0.504** vs chance 0.50, while the observed rate is **0.737 / 0.640** (gap +0.216 / +0.137). The readout's marginal contribution is **8.9% / 2.5%** of the excess; the **state↔label alignment contributes 91.1% / 97.5%** → **N-B**. The parity-axis displacement magnitude is **not** off-path-specific (off 0.198/0.145 vs success 0.214/0.161), but the signed displacement tracks the target parity (corr −0.49/−0.46). Same-parity competitors are marginally closer to `w_y` (cos −0.008 vs −0.055). The readout value-label permutation is **degenerate** (mapped-back = exact no-op = obs; unmapped ≈ 0.50) and is reported as a necessary-condition control only. E and F both classify N-B.
**Decision:** **N-B — state-dominant parity alignment.** The readout's parity geometry is a small, real, non-dominant substrate; the observed same-parity signature is produced by the committed state's decode being systematically same-parity as the target — i.e. by where the committed state sits relative to the true class, acting through the readout geometry. Measurement only; **no causal claim**. The localization licenses a **representation-side** follow-up and argues against a readout-side fix; N does not design the intervention.
**Evidence:**
- report: `docs/reports/experiment_n_parity_origin_localization.md`
- data: `experiments/experiment_n_parity_origin.npz`, `experiments/experiment_n_parity_origin_meta.json`, `experiments/experiment_n_results.json`
- figures: `docs/reports/figures/experiment_n_{w_geometry,w_null,parity_axis,cross_register,state_proj,par_delta,readout_null,controls,value_pair_geometry,arithmetic,control_groups,E_vs_F}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_n.py`, `experiments/analyze_experiment_n.py`
- tests: `tests/test_experiment_n.py` (16 tests)
- reused unchanged (read only): `experiments/experiment_m_geometry.npz`, `experiments/experiment_l_readout.npz`, `experiments/experiment_i_coordinate.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_g_transitions.npz`
**Changes to project state:** The same-parity signature is localized to the **state↔label alignment** (N-B); the readout's parity geometry is established as small and **non-dominant** (≈2–9% of the excess); the readout value-label permutation is recorded as degenerate for parity; a representation-side follow-up is licensed but not designed. No fix; commit rule frozen.
**Open questions:** Is the state↔class alignment a property of the committed-state placement (a blend near `C`, with `C` near `w_y` because the candidate is correct) or of the upstream encoder/candidate representation? Is the small readout parity geometry incidental or functional?
**Next experiment:** **Experiment O — controlled representation-side test of the parity alignment (state-dominant follow-up)**: a single tightly controlled, frozen-checkpoint, measurement-first representation-side study of whether the committed state's same-parity decode is a property of its placement relative to the true-class region or of the upstream candidate/encoder representation; N determines the origin only and does not design an intervention; specified, not started.

---

## 2026-10-07 — Experiment O: Controlled Representation-Side Test of Parity Alignment

**Status:** Completed — interventional (inference-time state computation only; frozen checkpoints; no training; no optimizer; no weight/readout/gate/commit-rule/benchmark modification; no learned policy; no verifier; no branching; no Phase 2). Classification: **O-D — no specific parity causality** (E primary; F control agrees).
**Research question:** Is the off-path parity-aligned error produced upstream in the candidate representation or by the candidate→commit placement mechanism? Does removing the W-derived parity component of the candidate — through the original frozen gate/commit — specifically reduce the third-basin failure?
**Configuration:** Frozen E (primary) and F (control), seeds 42–46, strict test split; exact M/L/N off-path population (n = 323 E / 353 F) plus controls. Pre-registered 9 configs: baseline κ=0; candidate-side `C' = C − κ<C−S_prev, p_dest> p_dest` for κ∈{0.5,1.0}; a norm-matched orthogonal control (`C' = C − κ|<ΔC,p>| q`, q ⟂ p_dest deterministic); commit-side `S_commit' = S_commit − κ<S_commit−S_prev, p_dest> p_dest` for κ∈{0.5,1.0}; and its orthogonal control. `p_dest` derived from W ALONE. The modified candidate flows through the **original frozen gate and commit**; applied at ADD transitions only, deterministically, state-locally. κ=0 is an exact no-op (reproduces the frozen forward: max state diff 0.0, identical decodes, all 10 checkpoint-seeds); collector bit-identical across two runs; analysis JSON identical (excluding timestamp).
**Key result:** **No specific parity causality.** The candidate-side parity removal rescues a small number of off-path transitions (net **+8.0/seed (E) / +8.8 (F)**) but the **matched orthogonal control rescues essentially the same** (E +8.6, F +8.0) — only 2/5 seeds exceed the control — so the rescue is **not parity-specific**. The parity interventions break **24.8% (E) / 22.2% (F)** of baseline-correct successes (versus 7.8% / 6.6% for the matched control) and **reduce sequence EM** (E T=10 0.7547→0.6505, −10.4 pp, 5/5 seeds, dz −2.49, p=0.005; T=20 0.4068→0.3224; all operations degrade: SET −9.1 pp, SWAP −9.3 pp, ADD-no-wrap −12.2 pp, ADD-wrap −14.7 pp). The commit-side stage control shows the same non-specific pattern (net +9.8 vs control +7.2, but only 3/5 seeds positive and worse collateral). The effect is a generic consequence of perturbing the candidate of that magnitude, not a parity-direction mechanism.
**Decision:** **O-D — no specific parity causality.** Parity alignment is a **correlated state/readout diagnostic feature, not a demonstrated causal mechanism** for the third-basin failure in this intervention family. Parity-directed intervention work stops. Interventional evidence restricted to the frozen model and the measured consequence; no architectural claim; not a fix.
**Evidence:**
- report: `docs/reports/experiment_o_representation_intervention.md`
- data: `experiments/experiment_o_representation_intervention.npz`, `experiments/experiment_o_representation_intervention_meta.json`, `experiments/experiment_o_results.json`
- figures: `docs/reports/figures/experiment_o_{rescue_regression,cand_vs_orth,commit_vs_orth,sequence_em,t10_t20,group_accuracy,collateral,geometry,by_op,hamming,kappa_monotone,E_vs_F}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_o.py`, `experiments/analyze_experiment_o.py`
- tests: `tests/test_experiment_o.py` (16 tests)
- reused unchanged (read only): `experiments/experiment_n_parity_origin.npz`, `experiments/experiment_m_geometry.npz`, `experiments/experiment_l_readout.npz`, `experiments/experiment_i_coordinate.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_g_transitions.npz`
**Changes to project state:** The parity hypothesis is **closed as a causal mechanism** (O-D): the measured parity direction is a correlated diagnostic, not a recoverable lever. Commit-side (J/K) and representation-side parity (O) interventions are both ruled out for improving the frozen sequence endpoint; the remaining question is the sequence-level error budget.
**Open questions:** Where does the frozen model's sequence-level error budget live now — candidate generation (upstream), the candidate→commit residual, or readout/decoding? Is any further intervention on the frozen model warranted, or is the remaining ceiling a genuine representation limit?
**Next experiment:** **Experiment P — sequence-level error-budget attribution across the frozen rollout (measurement-only)**: attribute the E model's T=10/T=20 exact-match failures, per seed, to candidate-generation errors vs the candidate→commit residual vs readout/decoding along the recursive trajectory, to identify the dominant recoverable target now that commit-side and representation-side parity interventions have been ruled out; specified, not started.

---

## 2026-10-07 — Experiment P: Sequence-Level Error-Budget Attribution

**Status:** Completed — measurement only (frozen E primary and F control checkpoints; no training; no optimizer; no parameter/readout/gate/commit modification; no intervention; no branching; no verifier; no oracle repair; no counterfactual rollout). Result: **candidate generation (Class A) dominates the first-error budget** for E and F at T=10 and T=20.
**Research question:** Where does the frozen model's T=10/T=20 exact-match error budget live? Among failed sequences, what fraction first become observably wrong at candidate generation (A), the candidate→commit residual (B), or only the terminal/state readout (C)?
**Configuration:** Frozen E (primary) and F (control), seeds 42–46; canonical strict test split (1,200×10) for T=10 and horizon sets for T∈{1,2,3,5,8,15,20}; per-step frozen trajectory (S_prev/C/S_commit) and exact decodes captured; task state reconstructed deterministically from `step_targets`. Each failed sequence (terminal full-state decode wrong) is assigned its earliest observed task-level failure site by a pre-registered exclusive rule: **A** candidate written-register error; **B** candidate-correct/committed-written-wrong; **C** no A/B but terminal readout residual (C1 untouched / C2 written / C3 near-boundary by normalized true-vs-winner distance > −0.5); **D** mixed. Dominance = largest mean per-seed fraction AND largest in ≥4/5 seeds. Collector bit-identical across two runs; analysis JSON identical (excluding timestamp); T=10 EM reproduces 0.7547 (E) / 0.7370 (F) and T=20 0.4068 / 0.3898 exactly.
**Key result:** The T=10 first-error budget (E) is **A 59.7% ± 3.8, B 32.5% ± 3.2, C2 2.0%, C3 5.8%, C1 0%, D 0%** — **Class A dominates in 5/5 seeds (gap +0.27)**; at T=20 the budget is **A 67.2%, B 30.2%, C 2.6%** (A dominant 5/5, gap +0.37). F is statistically indistinguishable (A 59.4%/67.6%, B 31.8%/29.9%, paired p ≥ 0.75). Candidate-origin first errors are almost entirely **ADD** (no-wrap 568 + wrap 542 of ~1,203), plus SWAP 93, SET 0; commit-residual first errors are even more wrap-weighted (377 wrap vs 274 no-wrap). **Untouched-register terminal corruption is 0** (retention preserved). Written-register errors recover later in 74–80% of cases but full-state correctness in only 31–42%; first errors occur across t=2…9. T=10 failure predicts T=20 failure descriptively (P=0.731 vs 0.544; OR ≈2.3). The composition shifts toward candidate generation as the horizon grows (A 0.52→0.67 for T=5→20).
**Decision:** **Observation — candidate generation (Class A) dominates the observed first-error budget** (~60% T=10, ~67% T=20; 5/5 seeds; robust E/F). Recoverable-target labels: **A1** (dominant, localized in ADD wrap/repeated, **not falsified** by J/K/O), **B1** (large mass, already falsified by J/K), **C1** (readout-adjacent, no causal evidence), D (empty). This is an **observed** budget, **not a causal claim**; the dominant mass is not assumed recoverable.
**Evidence:**
- report: `docs/reports/experiment_p_sequence_error_budget.md`
- data: `experiments/experiment_p_error_budget.npz`, `experiments/experiment_p_error_budget_meta.json`, `experiments/experiment_p_results.json`
- figures: `docs/reports/figures/experiment_p_{budget_t10,budget_t20,budget_horizons,survival,hazard,first_error_t,classA_ops,classB_profile,classC_readout,untouched,recovery,E_vs_F}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_p.py`, `experiments/analyze_experiment_p.py`
- tests: `tests/test_experiment_p.py` (14 tests)
- reused unchanged (read only): `experiments/experiment_o_representation_intervention.npz`, `experiments/experiment_n_parity_origin.npz`, `experiments/experiment_m_geometry.npz`, `experiments/experiment_l_readout.npz`, `experiments/experiment_i_coordinate.npz`, `experiments/experiment_h_interp.npz`, `experiments/experiment_g_transitions.npz`
**Changes to project state:** The sequence-level first-error budget is established: candidate generation dominates (~60–67%), the candidate→commit residual is second (~30%), and terminal/readout residuals are a small near-boundary minority (~3–8%). This is consistent with (and helpfully explains) the J/K/O failure to improve sequence EM — those interventions never targeted the dominant earliest site. The observed budget is kept strictly separate from causal mechanism and intervention leverage.
**Open questions:** Within the dominant Class-A candidate-generation budget, is the failure an arithmetic-representation limit of the frozen candidate MLP (the wrapped sum is not linearly represented in `C`) or a training/credit-assignment effect (candidate-local supervision already raised candidate accuracy in the generator-repair experiment)? Is the dominant observed source actually recoverable?
**Next experiment:** **Experiment Q — one controlled candidate-generation origin experiment targeting the dominant Class-A (wrap/repeated-ADD) budget (measurement-first)**: determine whether the dominant candidate-generation failure is a candidate-representation (arithmetic) limit or a training/objective effect, using the frozen candidate representation and the L/N linear-readout decomposition tools; Q must not touch the commit/gate/readout, must not re-run J/K/O, and must establish recoverability before any intervention is adopted; specified, not started.

---

## 2026-10-07 — Experiment Q: Candidate-Generation Origin / Recoverability

**Status:** Completed — measurement only (frozen E primary and F control checkpoints; no model training; no optimizer on model weights; no architecture/candidate/core/gate/commit/readout modification; no branching/verifier/oracle; no counterfactual rollout). Result: **Q-C — weak recoverability** for E and F.
**Research question:** Is the dominant Class-A candidate-generation failure primarily a candidate-representation/arithmetic limitation (insufficient linearly accessible representation of modular wrap / repeated ADD in `C_t`) or more consistent with exposure/decoding/training effects?
**Configuration:** Frozen E and F, seeds 42–46; DisjointSplitManager TRAIN (6,000) / VAL (1,200) / canonical TEST (1,200, stream `90042+seed`); all test ADD transitions captured (15,763/model over 5 seeds), with deterministic TRAIN/VAL subsamples (1,500/800 per seed). Three distinct quantities measured: (1) frozen-head candidate error, (2) analysis-only **linear** recoverability (closed-form ridge, λ tuned on VAL), (3) analysis-only **nonlinear** recoverability (fixed-capacity MLP 212→32→10, early stop on VAL). Also raw-sum vs modular probes, candidate geometry, wrap/repeated-ADD breakdowns, matched candidate-correct controls. **All model execution under `torch.no_grad()`; probes are separate objects and never enter inference; no checkpoint modified** (SHA-256 verified before/after). Collector bit-identical across two runs; analysis JSON identical (excluding timestamp).
**Key result:** On the Class-A population (test ADD where the frozen candidate head is wrong, n≈361/seed) the frozen head is 0% by construction, the **linear probe recovers the true written value 26.7% (E) / 25.3% (F)** and the **nonlinear probe 23.3% / 26.1%** — i.e. **no nonlinear gain** (Δ = −0.035 / +0.008). Wrap ≈ no-wrap (0.273/0.263 E; 0.271/0.238 F); repeated ADD slightly weaker (0.222 vs isolated 0.301, E); **raw sum not easier than modular** (0.260 vs 0.267, E). On the candidate-correct control the same probe reaches 93.4% (E), confirming competence where the representation is intact. Class-A states are margin-degraded (margin 1.9 vs 9.3) but **not cleanly separated** (centroid distance 2.94; diffuse confusion). E and F are statistically indistinguishable.
**Decision:** **Q-C — weak recoverability** (E primary; F agrees; bars 0.50/0.20/0.15 declared pre-registration). The true value is only weakly recoverable from `C_t` on the dominant Class-A population, and a small nonlinear probe reveals no hidden linear information. This is the **strongest available evidence for a candidate-generation representation limitation** — an **observed** limitation, **not** a causal claim; recoverability is **not** demonstrated and **no intervention is licensed**.
**Evidence:**
- report: `docs/reports/experiment_q_candidate_origin.md`
- data: `experiments/experiment_q_candidate_origin.npz`, `experiments/experiment_q_candidate_origin_meta.json`, `experiments/experiment_q_results.json`
- figures: `docs/reports/figures/experiment_q_{head_vs_linear,wrap_nowrap,repeated,rawsum_vs_modular,cand_margin,norms,value_geometry,confusion,nonlinear_vs_linear,per_seed,gap_by_op,historical}.png`
- checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
- code: `experiments/run_experiment_q.py`, `experiments/analyze_experiment_q.py`
- tests: `tests/test_experiment_q.py` (22 tests)
- historical cross-check (read only): `experiments/generator_repair_results.json`
**Changes to project state:** The dominant Class-A candidate-generation failure is **not demonstrably recoverable** from the frozen candidate vector under analysis-only probes — the target is only weakly linearly recoverable and a nonlinear probe adds nothing. This is the strongest evidence so far for a **candidate-representation limitation**, kept strictly separate from causality and from intervention leverage; no fix is proposed and no inference-time intervention is licensed.
**Open questions:** Is Q-C a genuine **information** limit of `C_t`, or a probe-distribution/capacity artifact (the probes were trained on the mostly-easy TRAIN distribution and evaluated on the hard tail)? Does a bounded **capacity sweep** on the hard distribution, or probes trained directly on the hard population with matched-difficulty controls, recover more?
**Next experiment:** **Experiment R — controlled candidate-representation information test on the Class-A tail (measurement-only)**: a capacity sweep (linear → increasing small-nonlinear capacity) plus probes trained directly on the hard (candidate-wrong) distribution in a disjoint hard-split design with matched-difficulty controls, to separate a genuine information limit from a probe-distribution/transfer artifact, with the same raw-sum-vs-modular and wrap/no-wrap contrasts; measurement-only, touches no weights; only if R establishes recoverability would a further candidate-side study be licensed; specified, not started.
