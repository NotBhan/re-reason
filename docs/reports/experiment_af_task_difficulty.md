# Experiment AF — Held-Out Operand Task-Difficulty Stratification

1. **Date:** 2026-10-11
2. **Status:** Completed — measurement-only task-side analysis (U's existing representation-independent
   exact-interface machinery on the unchanged V dataset; no reader training; no RE:Reason latent-state
   probe; no task change; no RE:Reason modification; no state replacement; no intervention; no Phase 2).
   Result: **task_side_limitation_not_indicated** — every held-out operand stratum retains
   ≈1.0 raw-interface target identifiability, so AE's reader failure is not explained by an evident loss
   of target identifiability at the raw interface (which still does not identify the reader-side
   limitation).
3. **Research question:** Do AE's weak held-out-operand results (the 20 pairs `(a,b)` with `b ∈ {0,5}`)
   coincide with identifiable differences in task-side label geometry or raw-interface target
   recoverability? AF is descriptive and conditional on these measurements; it does not reinterpret AE
   as evidence for or against RE:Reason's frozen representation.
4. **Existing baseline:** U (U-D) established that the raw causal observation history determines the ADD
   operands/target: the interface-identifiable horizon is L\*=4 (target Bayes ≈0.998), the full prefix is
   uniquely sufficient (1.000), yet the bounded flat probes stay weak. V/W/X/AD showed bounded causal
   readers (GRU/TCN) do not access that information; AE showed the AdamW-qualified readers fit the
   training pool far above its baseline without transferring to the held-out operand pairs.
5. **Hypothesis:** if the held-out pairs were harder at the raw interface (sparse support, ambiguous
   histories, or a target not identifiable from the causal history), their exact-interface ceiling
   should fall materially below the overall U baseline; if the ceiling stays high, the reader failure is
   not explained by an evident loss of raw-interface target identifiability.
6. **What is frozen:** V's exact source arrays (via AD's artifact; V SHA-256 and byte-identity
   re-verified); `y_true = (operand_a + operand_b) mod 10`; canonical TRAIN/VAL/TEST assignments; the
   Class-A definition (frozen candidate written-register decode wrong) and the all-ADD population; E
   primary and F control; U's exact-interface estimator, history construction, support conventions and
   Bayes-ceiling implementation; the frozen E/F checkpoints (SHA-256 verified before and after).
7. **What changed:** nothing — AF adds measurement only. No reader is trained, no predictor is fit, no
   optimizer/architecture/target/loss is introduced, and no rule, history length or subgroup is selected
   from outcomes.
8. **Dataset / protocol:** the canonical TEST populations per frozen seed (E and F, seeds 42–46);
   held-out condition = all 20 pairs `(a,b)` with `b ∈ {0,5}`; the `b ∉ {0,5}` rows are the seen-operand
   reference group. Predefined strata (from operand values and the arithmetic rule, before inspecting
   outcomes): `b=0` vs `b=5`; wrap (`a+b ≥ 10`) vs no-wrap; `a<5` vs `a ≥ 5`; `a==b` vs `a≠b`; target
   class `y_true`. Primary histories L4 and Lfull; other lengths reported descriptively.
9. **Experimental procedure:** for each model/seed/population/rep/stratum, U's `exact_bayes` is applied
   to the causal-history keys of the masked rows, for `y_true`, `operand_a` and `operand_b`, returning
   the ceiling together with support (n), unique-history counts, collision rate, ambiguous mass, max
   multiplicity and label entropy. Per-seed estimates are aggregated (mean ± sd over the five E
   checkpoints) and a pooled (concatenated-seed) estimate is also computed. Geometry (label counts and
   entropy) is reported per stratum for all-ADD and Class-A separately, with per-seed and pooled counts.
10. **Results.**
    * **Reproduction of U's baseline.** AF's pooled estimates reproduce U's published ceilings:
      Class-A L4 **0.9994** and all-ADD L4 **0.9981** (n = 1,805 / 15,763; unique histories
      1,803 / 15,634).
    * **Raw-interface identifiability is unchanged in every held-out stratum (E, pooled).** all-ADD L4:
      held-out **0.9991**, seen **0.9991**; `b=0` 0.9991, `b=5` 1.0000, wrap 1.0000, no-wrap 0.9991,
      `a<5` 0.9997, `a≥5` 1.0000, `a==b` 1.0000, `a≠b` 0.9995. all-ADD Lfull: 1.0000 everywhere.
      Class-A L4: overall **0.9994**, held-out **1.0000**, seen **1.0000**, and every stratum 1.0000;
      Class-A Lfull 1.0000 everywhere. `operand_a` recoverability tracks the target (0.9991–1.0000) in
      every stratum; `operand_b` is trivially ≥1.0 in `b=0` / `b=5` strata (constant operand). The
      largest per-seed subgroup gap versus the same-population overall ceiling is **≤0** (all strata at
      or above the overall level); the F control gives the same picture.
    * **Support.** all-ADD strata are large (pooled n = 459–7,483; Class-A 76–292). Class-A strata are
      thinner (`wrap` 76 pooled ≈15/seed, `a==b` 78 pooled ≈16/seed) and the 20 individual held-out
      pairs are sparse (≈2–10 Class-A rows per seed, pooled ≈9–49). These are reported explicitly; the
      per-pair estimates are flagged as low-support and are not interpreted as practical learnability.
    * **Task-side label geometry (Class-A, pooled over seeds).** held-out target counts
      `[78,21,37,29,38,36,30,35,34,30]` (class 0 ≈ 21.2%, entropy 3.12) vs seen
      `[121,148,153,112,172,127,183,121,169,131]` (class 0 ≈ 8.4%, entropy 3.28). Two structural facts
      dominate: **equal operands in the held-out set are only `(0,0)` and `(5,5)`, both of which map to
      target 0** — the `a==b` stratum has entropy 0 and carries the entire held-out class-0 mass; and the
      `a≠b` stratum can never have target 0 (`a≠b` counts start at class 1). The `wrap` stratum is
      entirely `b=5, a≥5` (targets {0,1,2,3,4}, entropy 1.98) and **`b=0` never wraps** — wrap is
      structurally tied to `b=5`/`a≥5` in this held-out set, so these are descriptive strata, **not**
      independent causal factors.
11. **Statistical analysis:** five frozen E checkpoints are the independent units. Per-seed and pooled
    estimates, support (n), unique-history fractions, collision rates and ambiguous masses are retained
    per stratum in `experiments/experiment_af_results.json` and
    `experiments/experiment_af_task_difficulty_raw.json`. Individual transitions and the 20 operand
    pairs are **not** treated as independent replicates. No multiplicity correction is applied; the
    result is reported continuously.
12. **Interpretation (observation vs interpretation).**
    * *Empirical observation:* the raw-interface `y_true` Bayes ceiling is ≈1.0 in every predefined
      held-out stratum (and for `operand_a`), essentially identical to the overall population; the
      held-out Class-A label distribution is class-0-skewed and the wrap/equality strata are structurally
      constrained.
    * *Supported interpretation:* **task-side limitation not indicated** — the reader failure on
      held-out operands is not explained by an evident loss of raw-interface target identifiability;
      the information remains exactly identifiable from the L4 causal history for every held-out
      subgroup. The label-geometry differences are real but affect *population composition / baselines*,
      not *identifiability*.
    * *Not claimed:* this does **not** identify which reader-side limitation explains AE, does not
      establish anything about RE:Reason's frozen state, and does not contradict U (it reproduces U).
13. **What the experiment DOES establish:** (i) held-out operand strata retain ≈1.0 exact-interface
    target (and `operand_a`) identifiability at L4 and Lfull, matching U's overall baseline (pooled
    Class-A L4 0.9994 / all-ADD 0.9981); (ii) the held-out Class-A population has a distinct label
    geometry (class-0 skew from the `(0,0)`/`(5,5)` equal-operand pairs; `wrap` confined to
    `b=5, a≥5`); (iii) the five E checkpoints and F control agree; (iv) per-pair support is small and is
    reported so no sparse estimate is over-read.
14. **What the experiment DOES NOT establish:** that the readers could access the information (they do
    not); which reader-side limitation explains AE; any property of RE:Reason's frozen representation,
    latent state or architecture; any causal role for wrap or operand equality; that sparse per-pair
    ceilings imply practical learnability.
15. **Decision:** **task_side_limitation_not_indicated** (E primary; F control identical). No
    held-out subgroup shows a material (≥0.15) or seed-consistent drop below the overall ceiling; the
    largest subgroup gap is ≤0. The V-A/V-B accessibility criterion is unchanged (this experiment does
    not re-run readers).
16. **Next experiment:** **Experiment AG — bounded-reader inductive-bias / generalization stress at V's
    scale (measurement-only, frozen checkpoints):** AF removes the task-identifiability explanation for
    the held-out-reader failure, and AE showed the AdamW readers fit without transferring. The single
    remaining reader-side question, without probing RE:Reason, is whether the fit–transfer gap tracks a
    reader-side inductive-bias limitation — e.g. whether the training pool's operand-pair coverage
    (which pairs are present, class balance) explains the held-out failure under the same reader and
    protocol. Specified here and **not started** in this run.
17. **Reproduction commands:**
    ```
    PYTHONPATH=src python experiments/run_experiment_af.py
    PYTHONPATH=src python experiments/analyze_experiment_af.py
    PYTHONPATH=src uv run python -m pytest tests/test_experiment_af.py -v
    ```
18. **Artifact list:**
    * report: `docs/reports/experiment_af_task_difficulty.md`
    * code: `experiments/run_experiment_af.py`, `experiments/analyze_experiment_af.py`
    * data: `experiments/experiment_af_task_difficulty_meta.json`,
      `experiments/experiment_af_results.json`, `experiments/experiment_af_task_difficulty_raw.json`
      (per-seed + pooled exact-interface estimates, support/unique-history diagnostics, label geometry,
      and the 20 per-pair estimates)
    * figures: `docs/reports/figures/experiment_af_{bayes_by_group,operand_a_by_group,support_by_group,per_pair_support,target_distribution,entropy_by_group,ceiling_gap,decision_summary}.png`
    * tests: `tests/test_experiment_af.py`
    * checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
    * reused unchanged (read only): `experiments/experiment_ad_qualified_access.npz`;
      `analyze_experiment_u.py` (exact-interface estimator / history construction / population masks);
      `experiments/experiment_u_results.json` (U baseline); `experiments/experiment_v_sequence_access.npz`.
19. **Scientific guardrails:** AF is measurement-only and task-side; it does not train a reader, probe
    RE:Reason's latent state, change the task, or license a candidate-side/architecture change. Wrap and
    operand-equality are descriptive strata tied to `b ∈ {0,5}`, not independent causal factors; "digit
    overlap" is not used and no overlap metric was added. Raw-interface identifiability is kept separate
    from bounded-reader learnability and from AE's train-derived baselines; sparse per-pair estimates are
    not over-read. Negative/inconclusive findings are preserved. No RE:Reason modification; no Phase 2.
    All prior reports and artifacts unchanged.
