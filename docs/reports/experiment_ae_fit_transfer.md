# Experiment AE — Baseline-Aware Fit–Transfer Diagnostic

1. **Date:** 2026-10-11
2. **Status:** Completed — measurement-only analysis-reader experiment (frozen E/F checkpoints used
   only to define test populations and reproduce baseline metrics; no RE:Reason training; no optimizer
   on RE:Reason parameters; no architecture / encoder / recurrent-core / state / gate / candidate /
   commit / readout modification; no state replacement; no candidate manipulation; no feature editing;
   no branching/verifier; no counterfactual rollout; no Phase 2). AE tests **only** the analysis
   readers' fit–transfer behaviour; it does **not** test RE:Reason's frozen representation. Result:
   **no evidence of partial transfer**; GRU cells are **fit-without-transfer**, the TCN cells are
   predominantly fit-without-transfer with one borderline cell, and the overall descriptive verdict is
   **inconclusive_mixed**; V's **V-B** accessibility criterion is unchanged.
3. **Research question:** Do AD's AdamW-qualified readers' high-training-accuracy solutions show
   evidence of **transferable structure on held-out operand pairs**, or do they predominantly **fit the
   observed training pool**? AE is explicitly **not** a representation-shift experiment; no
   representation probe was designed or approved.
4. **Existing baseline:** AD (AD-B) showed the AB-qualified `adamw` recipe removes the readers' fitting
   obstacle at V's full scale (best-on-VAL training accuracy 0.44–0.86) with no transfer to the
   untouched canonical TEST; V (V-B) reported Class-A 0.083 (GRU) / 0.112 (TCN) at L4 and held-out
   ≤0.25 everywhere under Adam. V-A/V-B accessibility bars are unchanged here.
5. **Hypothesis:** if the readers learned a weak but real ADD rule under `adamw`, held-out-operand
   accuracy should exceed the variant-matched majority-class baseline consistently across seeds and
   operand pairs (partial transfer); if they fit only the observed pool, training accuracy should rise
   while held-out accuracy stays at/below its baseline (fit without transfer).
6. **What is frozen:** V's source arrays, observation histories, population masks, splits and seed
   assignments (via AD's artifact); AD's `adamw` condition and settings; the readers (GRU hidden 85;
   causal TCN 52 channels, kernel 3, dilations `[1,2,4,8]`); seeds 42–46; histories
   `{L1,L2,L4,L8,L16,Lfull}`; variants `main` / `heldout`; target `y_true = (operand_a + operand_b)
   mod 10`; Class-A definition; the VAL-only best-on-checkpoint rule; U's Bayes ceilings; the frozen
   E/F checkpoints (SHA-256 verified before and after).
7. **What changed:** nothing in the experimental conditions — AE adds only measurement. Because AD
   retained the best-on-VAL scalar metrics and the train/val trajectories **but no state
   dictionaries**, AE **deterministically reproduces the exact AdamW fits** with AD's trainer
   (`ab.train_ab`), V's arrays and V's reader-seed convention. No optimizer condition, initialization,
   target, loss, architecture or tuning is added; only `adamw` is reproduced. Reused-state vs
   reproduction is recorded in `experiments/experiment_ae_fit_transfer_meta.json`
   (`reused_retained_states = false`, `inspection.has_state_dict = false`).
8. **Dataset / protocol:** V's exact arrays (via `experiments/experiment_ad_qualified_access.npz`; V
   SHA-256 and byte-identity re-verified by `run_experiment_ae.py`). Ordinary TRAIN ADD cap 1,500/seed
   (heldout variant excludes operand pairs `b ∈ {0,5}`), ordinary VAL cap 800/seed, canonical TEST
   15,763 ADD transitions. Each seed/reader/history/variant is fit with the `adamw` recipe (lr 3e-3,
   wd 1e-2, 200 full-batch updates, class-balanced CE) and evaluated once at the best-on-VAL state.
   Grid = 2 readers × 6 histories × 2 variants × 5 seeds = **120 reproduced fits**. Test results never
   select checkpoints, horizons or variants.
9. **Experimental procedure:** at the best-on-VAL state, evaluate training (fit pool), validation, and
   test populations (Class-A, matched candidate-correct, all candidate-correct, all ADD), and — for
   the `heldout` variant — test rows with `b ∈ {0,5}` (held-out operands) versus other rows (seen),
   reported separately. For each population report raw accuracy, balanced accuracy with per-class
   recall, a majority-class baseline derived **exclusively from the corresponding training labels**,
   and the test-label frequencies. For the `heldout` variant also report per-pair accuracy for the 20
   excluded operand pairs and the target-class confusion. The `main` variant's held-out/seen breakdown
   is reported as a **subgroup analysis** (those operands were present in its training pool — not an
   unseen-operand test). Five seeds are the independent units.
10. **Results.**
    * **Reproduction bridge.** AE's reproduced `adamw` metrics match AD's retained `adamw` metrics
      **exactly** (`max |Δ| = 0.0` for train accuracy, Class-A and held-out Class-A), confirming the
      reproduction is the same set of fits.
    * **Fit.** Best-on-VAL training accuracy (fit pool): GRU 0.858 (L4) / 0.649 (full); TCN 0.442 /
      0.655, versus the training-derived majority-class baseline 0.285 — all readers fit far above
      their baseline. Validation accuracy is much lower (GRU 0.221/0.225; TCN 0.152/0.187).
    * **Transfer (main variant).** Class-A test 0.095/0.094 (GRU) and 0.094/0.097 (TCN); all-ADD
      0.207/0.213 (GRU) and 0.147/0.185 (TCN). Train-to-test gaps are large (GRU 0.763/0.555; TCN
      0.348/0.558).
    * **Held-out vs seen operands (`heldout` variant, Class-A).** GRU seen/held-out 0.091/0.106 (L4)
      and 0.095/0.089 (full); TCN 0.089/0.122 and 0.096/0.136. The variant-matched held-out
      majority-class baseline is **0.116** (derived from the held-out-filtered training labels); the
      GRU is **below** it, the TCN is slightly **above** it. Balanced accuracy on held-out Class-A is
      ≈chance (GRU 0.103/0.093; TCN 0.114/0.103).
    * **Baseline-aware transfer signal.** Per-seed held-out-minus-baseline gain: GRU L4
      `[+0.0098, +0.0759, −0.0312, −0.0189, −0.0857]` (2/5 positive); GRU full
      `[+0.049, −0.0127, −0.0156, −0.1132, −0.0429]` (1/5); TCN L4
      `[+0.0784, +0.0380, +0.0312, −0.1321, +0.0143]` (4/5, mean **+0.0060**); TCN full
      `[+0.0588, +0.1139, +0.0625, −0.1321, 0.0000]` (3/5, mean **+0.0206**). Reproducibility across
      the 20 held-out operand pairs is weak: 7/7 pairs above chance for the GRU, 9/8 for the TCN; per
      target class 4/3 (GRU) and 5/3 (TCN) of 10 above chance. Class-A held-out label distribution is
      skewed (class 0 ≈ 15.6/73.6); the per-seed majority class from the training labels varies, giving
      baseline 0.116.
11. **Statistical analysis:** five frozen E checkpoints are the paired units. Per-seed transfer gains,
    per-pair means and per-class recalls are retained with their pooled means and standard deviations
    in `experiments/experiment_ae_results.json` and `experiments/experiment_ae_grid_raw.json`. The
    only positive mean held-out–baseline margins are the TCN's +0.006 (L4) and +0.021 (full); both are
    **inconsistent across seeds** (4/5 and 3/5) and **not reproducible across operand pairs** (9/8 of
    20 above chance) — i.e. within noise and pair-dependent, not a consistent signal. No multiplicity
    correction is applied; the result is reported continuously, not thresholded.
12. **Interpretation (observation vs interpretation).**
    * *Empirical observation:* under the `adamw` recipe the readers fit the training pool far above
      their training-derived baseline while held-out-operand Class-A accuracy is at or barely above
      the variant-matched majority-class baseline, ≈chance (balanced), with weak per-pair/class
      reproducibility.
    * *Supported interpretation (baseline-aware categories):* **no cell shows partial transfer**; the
      GRU cells are **fit-without-transfer** (held-out at/below baseline); the TCN's small positive
      held-out margins are seed- and pair-inconsistent, so the TCN is **predominantly
      fit-without-transfer with one borderline cell**, and the overall descriptive verdict is
      **inconclusive_mixed** rather than transfer. In particular, the TCN's apparent held-out advantage
      is **not** evidence of a transferable ADD rule.
    * *Unresolved / not claimed:* AE does **not** show the ADD rule is unlearnable, that the information
      is absent from RE:Reason's frozen state, or that some other reader could not access it. AE is not
      a representation-shift experiment and makes no frozen-representation claim.
13. **What the experiment DOES establish:** (i) the `adamw` readers fit the training pool far above
    their training-derived baseline while transfer stays at/below the variant-matched majority-class
    baseline; (ii) no cell meets the descriptive partial-transfer criterion — the small TCN held-out
    margins are seed- and pair-inconsistent; (iii) the reproduction is exact against AD's retained
    `adamw` metrics (`max |Δ| = 0.0`); (iv) V's V-B accessibility verdict is unchanged (best Class-A
    0.097, held-out 0.136, Bayes−reader ≈0.90).
14. **What the experiment DOES NOT establish:** any property of RE:Reason's frozen representation,
    latent state, architecture or sufficiency; that the ADD rule is unlearnable or the information
    absent; a causal claim about the optimizer or the readers; a generalization from a subgroup
    breakdown of the `main` variant (its operands were in training); or that individual transitions are
    independent replicates (five seeds are the units).
15. **Decision:** **no partial transfer.** Descriptive categories: GRU/L4, GRU/Lfull,
    TCN/Lfull → **fit_without_transfer**; TCN/L4 → **inconclusive_mixed** (mean gain +0.006, 4/5 seeds,
    9/20 pairs). Overall verdict **inconclusive_mixed** (mixed across readers), with the dominant
    pattern fit-without-transfer and no cell reaching partial transfer. Accessibility remains **V-B
    (sequence-limited)** unchanged.
16. **Next experiment:** **Experiment AF — held-out-operand difficulty stratification of the ADD task
    itself (measurement-only; no new reader):** AE shows the readers cannot transfer to held-out
    operand pairs, but AE's per-pair cells are small (n ≈ 3–15). Before any further reader-side probe,
    quantify how much of the held-out-operand difficulty is a property of the **task/label geometry**
    (e.g. carry/wrap structure, digit overlap, class balance across operand pairs) using U's
    representation-independent exact-interface machinery, so that a future reader/representation probe
    (if any) is designed against a known task-side baseline. Specified here and **not started** in
    this run.
17. **Reproduction commands:**
    ```
    PYTHONPATH=src python experiments/run_experiment_ae.py
    PYTHONPATH=src python experiments/analyze_experiment_ae.py --jobs 16
    # reassemble from the retained raw per-seed evidence without re-running the fits:
    PYTHONPATH=src python experiments/analyze_experiment_ae.py --jobs 16 --use-raw
    PYTHONPATH=src uv run python -m pytest tests/test_experiment_ae.py -v
    ```
18. **Artifact list:**
    * report: `docs/reports/experiment_ae_fit_transfer.md`
    * code: `experiments/run_experiment_ae.py`, `experiments/analyze_experiment_ae.py`
    * data: `experiments/experiment_ae_fit_transfer_meta.json`,
      `experiments/experiment_ae_results.json`, `experiments/experiment_ae_grid_raw.json`
      (120 reproduced AdamW fits: training/validation/test accuracy, balanced accuracy + per-class
      recall, training-derived majority baselines, label frequencies, the 20 held-out-pair accuracies,
      and the Class-A confusion, per seed)
    * figures: `docs/reports/figures/experiment_ae_{fit_vs_transfer,train_to_test_gap,heldout_vs_seen,per_seed_heldout,per_pair_GRU_L4,per_pair_GRU_Lfull,per_pair_TCN_L4,per_pair_TCN_Lfull,confusion_classA,balanced_vs_raw,per_class_recall_heldout,decision_summary}.png`
    * tests: `tests/test_experiment_ae.py`
    * checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
    * reused unchanged (read only): `experiments/experiment_ad_qualified_access.npz`;
      `experiments/experiment_ad_grid_raw.json`; `analyze_experiment_ad.py` (features/conditions/seeds);
      `analyze_experiment_ab.py` (trainer); `analyze_experiment_u.py` (history/populations);
      `analyze_experiment_v.py` (readers, U baseline); `experiments/experiment_u_results.json`.
19. **Scientific guardrails:** AE is a measurement-only analysis-reader experiment and makes no
    representation-shift or RE:Reason claim. The readers never enter frozen model inference and no
    RE:Reason parameter was trained or mutated. Poor external-reader performance does **not** establish
    absence of information in RE:Reason's frozen state. Negative and inconclusive findings are
    preserved; individual transitions are not treated as independent replicates; no new target, loss,
    probe, architecture, optimizer or training sweep was introduced. No dataset/target change, no
    RE:Reason modification, no Phase 2. All prior reports and artifacts are unchanged.
