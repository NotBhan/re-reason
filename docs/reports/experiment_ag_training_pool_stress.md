# Experiment AG — Bounded-Reader Training-Pool Coverage and Generalization Stress

1. **Date:** 2026-10-11
2. **Status:** Completed — measurement-only **analysis-reader training** experiment (frozen E/F
   checkpoints used only to define test populations and reproduce baseline metrics; no RE:Reason
   training; no optimizer on RE:Reason parameters; no representation probe; no architecture / encoder /
   recurrent-core / state / gate / candidate / commit / readout modification; no state replacement; no
   intervention; no Phase 2). Result: **inconclusive_mixed** — no training-pool sampling policy
   improves held-out-operand transfer (which stays at chance in every condition), while training fit is
   sampling-sensitive only for the seeds that fit at all (seed-heterogeneous). V's accessibility
   criterion is unchanged and not reached.
3. **Research question:** Is AE's fit–transfer gap sensitive to the sampling of the ordinary-ADD
   training pool — in particular operand-pair coverage and target-label composition — with the
   held-out operand pairs (`b ∈ {0,5}`) excluded from every condition? Any finding is conditional on
   the readers, data and protocol tested.
4. **Existing baseline:** U (U-D) — the raw causal history determines the operands/target (L\*=4
   target Bayes ≈0.998), yet bounded flat probes stay weak. V (V-B) — bounded causal GRU/TCN readers
   are also weak (Class-A 0.083/0.112 at L4; V-A not reached). AD (AD-B) — the AB-qualified `adamw`
   recipe does not materially change access. AE — `adamw` readers fit the training pool far above its
   baseline with no consistent transfer. AF — every held-out stratum retains ≈1.0 raw-interface
   identifiability.
5. **Hypothesis:** if the fit–transfer gap were driven by training-pool construction, re-balancing
   operand-pair coverage or target composition (with held-out pairs still excluded) should change
   training fit and/or held-out transfer; if transfer stays at chance under all samplings, the gap
   persists at the reader level for these tested conditions.
6. **What is frozen:** V's exact source arrays/splits (via AD's artifact; V SHA-256 and byte-identity
   re-verified); target `y_true = (operand_a + operand_b) mod 10`; Class-A mask; frozen E/F checkpoints
   (SHA-256 before/after); the canonical VAL and untouched canonical TEST; readers (GRU hidden 85;
   causal TCN 52 channels, kernel 3, dilations `[1,2,4,8]`); AdamW lr 3e-3 / wd 1e-2 / AB defaults;
   200 full-batch updates; class-balanced CE; best-on-VAL selection; deterministic reader-init seeds;
   histories L4 and Lfull (primary).
7. **What changed:** only the **training-pool construction**. Four prespecified conditions draw from
   each seed's eligible pool (V `trainord` rows with `b ∉ {0,5}`; held-out pairs never trained on):
   `ae_native` (the exact eligible pool, AE's anchor, no resampling); `natural_bootstrap` (sample with
   replacement, empirical distribution); `pair_balanced_bootstrap` (draw eligible operand pairs
   uniformly, then an example from that pair); `target_balanced_bootstrap` (draw the 10 target classes
   uniformly, then an example from that class). Conditions 2–4 keep `n` = the seed's `ae_native` size,
   with three deterministic resampling replicates per seed. Sampling seeds are separate from the
   model-initialization seed; model initialization is **matched across conditions** (same
   `av.reader_seed`; verified identical init hashes).
8. **Dataset / protocol:** frozen E, seeds 42–46. The held-out exclusion removes ~half of each seed's
   1,500-row `trainord` cap (operand_b is concentrated on 0/5): eligible pools 741–781 rows. Support
   verified: all 10 target classes present in every eligible pool; 75–80 of the 80 non-held-out
   operand pairs present, with the absent rare pairs **reported as a limitation**
   (`5_3,8_3` / `6_3,6_7,9_7` / `1_9,2_1,2_7,4_1,5_2`) and never patched from VAL/TEST. Grid = 4
   conditions (1 native + 3 bootstrap × 3 replicates = 10 pools) × 2 readers × 2 histories × 5 seeds =
   **200 fits**.
9. **Experimental procedure:** each pool is fitted with the fixed AdamW protocol; the best-on-VAL
   checkpoint is evaluated once on the canonical TEST. Reported per condition/reader/history/seed:
   training accuracy on the fitted pool and on the fixed eligible base pool, validation accuracy,
   held-out-operand Class-A and all-ADD raw **and balanced** accuracy with per-class recall and the
   training-derived majority baseline, seen-operand performance (descriptive), per-pair held-out
   accuracy with support, and the realized pair-coverage/target-composition of every generated pool.
   The three replicates are nested draws, not independent checkpoints.
10. **Results.**
    * **Realized sampling.** `natural_bootstrap` mirrors the empirical pool (class entropy 3.16,
      class-min share 0.037, pair-min 0.0); `target_balanced_bootstrap` uniformizes the target
      composition (class entropy **3.31**, class-min share **0.082**); `pair_balanced_bootstrap`
      uniformizes pair coverage (pair-min **1.1**, pair-max **17.3** vs native 46.2 / natural 49.3) and
      also flattens the class composition (entropy 3.31). Pair and target balancing are coupled through
      the arithmetic rule, so outcomes are not attributed to one factor.
    * **Training fit is sampling-sensitive only where the reader fits.** For the seeds that fit the
      native pool, resampling changes training accuracy substantially (TCN L4 per-seed native
      `[0.78,0.14,0.13,0.88,0.81]` vs natural `[0.45,0.16,0.11,0.26,0.47]`; TCN L4 mean fit
      0.55→0.29 natural, →0.25 pair-balanced, →0.30 target-balanced). But 2/5 seeds never fit any pool,
      so the change is **not** 4/5-consistent.
    * **Held-out transfer never improves.** Held-out Class-A balanced accuracy stays at chance in every
      condition (0.077–0.119; all conditions within ≈0.02 of `ae_native` and of chance); no transfer
      signal fires (best mean paired Δ balanced +0.019, TCN/GRU ≤3/5 seeds positive, ≤11/20 pairs
      improved). Raw accuracy tracks the (training-derived) majority-baseline composition, not rule
      recovery (e.g. GRU Lfull target-balanced raw 0.116 / balanced 0.112 vs native 0.089 / 0.093). The
      per-seed paired deltas and per-pair outcomes are retained.
    * **Accessibility.** The best Class-A accuracy across all conditions is 0.107 (TCN/Lfull,
      pair-balanced) — far below 0.50; **V-A is not reached** and V-B is unchanged.
11. **Statistical analysis:** five frozen E checkpoints are the independent units; the three
    resampling replicates are nested draws and are never treated as extra checkpoints. Per-seed means
    over replicates drive the paired comparisons against `ae_native` (and the replicate variation is
    reported within each cell). Per-pair and per-class results are retained. No multiplicity correction
    is applied; the outcome is reported continuously.
12. **Interpretation (observation vs interpretation).**
    * *Empirical observation:* no sampling policy moves held-out-operand balanced accuracy above chance;
      training fit responds to sampling for the seeds that fit, but that response does not transfer.
    * *Supported interpretation (descriptive categories):* the dominant, unambiguous finding is
      **persistent fit without transfer where fit occurs, with no transfer signal in any condition**.
      Because the fit is itself seed-heterogeneous (2/5 seeds never fit any pool) and the fit changes are
      not 4/5-consistent, the strict overall category is **inconclusive_mixed** (fit changes vary
      substantially across seeds); the sampling-sensitivity of the fit does not extend to transfer.
    * *Not claimed:* sampling sensitivity would not prove an architectural inductive-bias mechanism
      even if present; here it would only show the tested reader outcomes depend on pool construction.
      No RE:Reason frozen-representation claim is made; no architecture change is licensed.
13. **What the experiment DOES establish:** (i) neither operand-pair balancing nor target-composition
    balancing improves held-out-operand transfer for the tested readers under this protocol (balanced,
    raw, and per-pair held-out performance all stay at chance); (ii) training fit is sensitive to pool
    construction for the seeds that fit (a reader-side, not task-side, effect); (iii) the majority
    baseline shifts with the training composition and raw-accuracy changes track it, not rule recovery;
    (iv) V-A/V-B are unchanged and V-A is not reached.
14. **What the experiment DOES NOT establish:** an architectural inductive-bias mechanism; any RE:Reason
    frozen-representation, latent-state or architecture property; that some other reader/loss/protocol
    would not transfer; that the held-out information is absent. AG is conditional on V's readers, data
    and training protocol.
15. **Decision:** **inconclusive_mixed** (no transfer signal; training fit sampling-sensitive only for
    the seeds that fit). Robust sub-finding: **persistent fit without transfer** — held-out
    balanced/raw accuracy stays at chance in every condition. Accessibility: **V-A not reached; V-B
    unchanged**.
16. **Next experiment:** **Experiment AH — bounded-reader capacity / inductive-bias sensitivity at V's
    scale (measurement-only, frozen checkpoints):** AG shows training-pool construction changes fit but
    not transfer; the open reader-side question is whether the fit–transfer gap is sensitive to reader
    **capacity/width** at the same data/held-out protocol (a controlled width ladder within a fixed
    parameter range), or whether it is insensitive to capacity — distinguishing an optimization/coverage
    effect from a representation-capacity effect. Specified here and **not started** in this run.
17. **Reproduction commands:**
    ```
    PYTHONPATH=src python experiments/run_experiment_ag.py
    PYTHONPATH=src python experiments/analyze_experiment_ag.py --jobs 16 --reps L4 Lfull
    PYTHONPATH=src python experiments/analyze_experiment_ag.py --jobs 16 --use-raw
    PYTHONPATH=src uv run python -m pytest tests/test_experiment_ag.py -v
    ```
18. **Artifact list:**
    * report: `docs/reports/experiment_ag_training_pool_stress.md`
    * code: `experiments/run_experiment_ag.py`, `experiments/analyze_experiment_ag.py`
    * data: `experiments/experiment_ag_training_pool_stress_meta.json`,
      `experiments/experiment_ag_results.json`, `experiments/experiment_ag_grid_raw.json`
      (200 fits: per-draw pool composition, train/val/test metrics, balanced accuracy + per-class
      recall, training-derived baselines, per-pair held-out accuracy with support)
    * figures: `docs/reports/figures/experiment_ag_{heldout_bal_by_condition,fit_vs_transfer,paired_deltas,pool_target_composition,pool_pair_coverage,per_pair_ae_native,per_pair_pair_balanced_bootstrap,per_pair_target_balanced_bootstrap,heldout_raw_vs_balanced,decision_summary}.png`
    * tests: `tests/test_experiment_ag.py`
    * checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
    * reused unchanged (read only): `experiments/experiment_ad_qualified_access.npz`;
      `analyze_experiment_ad.py` (features/conditions/seeds), `analyze_experiment_ae.py` (population
      summariser), `analyze_experiment_ab.py` (trainer), `analyze_experiment_u.py` (populations),
      `analyze_experiment_v.py` (readers, seeds).
19. **Scientific guardrails:** AG is an analysis-reader training experiment, not a RE:Reason training or
    representation experiment; findings are conditional on the tested readers/data/protocol. No reader
    or representation claim is made about RE:Reason. The held-out pairs are excluded from every
    condition; absent rare pairs are reported, never patched. Balanced accuracy and per-class recall are
    reported so composition-driven raw-accuracy changes are not mistaken for rule recovery; the
    training-derived majority baseline is kept separate from test-label prevalence. The three replicates
    are nested draws, not independent units. Sampling sensitivity is not claimed to prove an
    architectural mechanism. No RE:Reason modification; no representation probe; no Phase 2. All prior
    reports and artifacts unchanged.
