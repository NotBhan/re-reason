# Experiment AH — Bounded-Reader Capacity and Rule-Generalisation Sensitivity

1. **Date:** 2026-10-12
2. **Status:** Completed — measurement-only **analysis-reader training** experiment (frozen E/F
   checkpoints used only to define test populations and reproduce baseline metrics; no RE:Reason
   training; no optimizer on RE:Reason parameters; no representation probe; no architecture / encoder /
   recurrent-core / state / gate / candidate / commit / readout modification; no state replacement; no
   candidate manipulation; no feature editing; no branching/verifier; no counterfactual rollout; no
   Phase 2). Result: **H2_fit_without_transfer (AH-B)** — increasing reader width improves training
   fit markedly where fitting occurs (TCN Lfull 0.374 → 0.770 across the ladder; GRU L4 0.306 → 0.484
   at Medium-large) while held-out-operand transfer stays locked at chance at every width (balanced
   0.072–0.143, chance 0.10; 0/5 seeds reach a material gain at any width). The fit–transfer gap is
   capacity-insensitive over this ladder and **H1 is falsified**. V's **V-A is not reached; V-B is
   unchanged**.
3. **Research question:** Does increasing the capacity of the external causal sequence readers improve
   generalisation of the modular arithmetic rule, or does it primarily improve training-set fit without
   held-out generalisation? Any finding is conditional on the tested readers, ladder, data and protocol.
4. **Existing baseline:** V (**V-B**) — bounded causal GRU/TCN readers are weak at V's scale (Class-A
   0.083/0.112 at L4; V-A not reached). AD (**AD-B**) — the AB-qualified `adamw` recipe does not
   materially change ADD-target access. AE (**inconclusive_mixed**) — `adamw` readers fit the training
   pool far above its baseline with no consistent transfer (persistent fit without transfer). AF
   (**task_side_limitation_not_indicated**) — every held-out operand stratum retains ≈1.0
   raw-interface target identifiability, so AE's failure is not a task-identifiability loss. AG
   (**inconclusive_mixed**) — training-pool construction changes fit only for the seeds that fit and
   never improves held-out transfer (sampling-sensitive fit, no transfer).
5. **Hypothesis.**
   * **H1 — Capacity-associated generalisation:** some width's held-out Class-A **balanced** accuracy
     improves by ≥ +0.15 over the baseline width in ≥ 4/5 seeds AND reaches ≥ 0.30.
   * **H2 — Fit without transfer:** training fit is present at some level (mean train accuracy ≥ 0.50,
     X's "training accuracy still low" boundary used as the fit bar, or a ≥ +0.15 gain over the
     baseline width in ≥ 4/5 seeds) while every fitting level's held-out balanced accuracy stays
     ≤ 0.25 (chance ≈ 0.10).
   * **H3 — Capacity-insensitive fitting failure:** no level in the ladder fits the training pool.
   * **H4 — Mixed / inconclusive:** everything else (borderline / seed-dependent); continuous
     quantities are reported and no verdict is forced.
   V's V-A/V-B accessibility criterion is evaluated separately and unchanged.
6. **What is frozen:** all E/F checkpoints (SHA-256 verified before and after; no weights modified);
   the RE:Reason architecture (no RE:Reason parameter trained, mutated or probed); V's exact source
   arrays/splits via AD's artifact (V SHA-256 `b985c386653538cd8312f6fdd0983330f5bee6b8c256718145492c1ae2a55494`
   re-verified; AD==V byte-identity; 108 source arrays re-checked by SHA-256/shape/dtype); target
   `y_true = (operand_a + operand_b) mod 10`; the Class-A mask; the canonical VAL and the untouched
   canonical TEST; the native eligible training pool (V `trainord` rows with `b ∉ {0,5}`, no
   resampling); the AdamW protocol (lr 3e-3, wd 1e-2, 200 full-batch updates, class-balanced CE,
   evaluated every 10 updates, best-on-VAL selection with a cloned state); histories L4 and Lfull;
   seeds 42–46; V's exact reader classes (GRU / causal TCN, kernel 3, dilations `[1,2,4,8]`, 4 causal
   layers, 22-dim input), with the output head 10-class `y_true`.
7. **What changed:** only the **reader width ladder** — five pre-registered levels, each within 4.0×
   the baseline level's exact parameter count:
   * **Small:** GRU hidden 42 — 8,746 params (0.31×) / TCN channels 26 — 8,174 params (0.29×)
   * **Medium-small:** GRU 60 — 15,730 (0.55×) / TCN 37 — 15,291 (0.54×)
   * **Baseline:** GRU 85 — 28,655 (1.00×) / TCN 52 — 28,506 (1.00×)
   * **Medium-large:** GRU 120 — 53,050 (1.85×) / TCN 74 — 55,214 (1.94×)
   * **Large:** GRU 170 — 100,650 (3.51×) / TCN 104 — 105,674 (3.71×)
   The maximum ratio is 3.71× (≤ the declared 4.0× cap, under both the 10-class and 19-class output
   widths); the **Baseline** level reproduces V's deterministic width selection exactly (GRU 85 /
   TCN 52, same 28,655 / 28,506 counts). The ladder was fixed in `run_experiment_ah.py` **before any
   fit**; no width, history, target or seed was selected after seeing a result.
8. **Dataset / protocol:** frozen E (primary), seeds 42–46; native eligible pools of 741–781 rows
   (E: 746 / 741 / 759 / 781 / 753 for seeds 42–46; E and F draw identical eligible pools); the 20
   held-out operand pairs (`b ∈ {0,5}`) are excluded from every fit and never trained on. Grid = **2
   readers × 5 widths × 2 histories (L4, Lfull) × 5 seeds = 100 fits** (all finite).
9. **Experimental procedure:** each cell is fitted with AB's exact `adamw` protocol at the level's
   width, on the seed's native eligible pool, with a deterministic per-cell init seed (V's
   heldout-variant reader seed plus a per-level offset of `(level_idx − 2) × 100000`; the Baseline
   level's offset is 0 and therefore matches AE's exact heldout-variant init seed). The best-on-VAL
   state is then evaluated **once** on the untouched canonical TEST; per-cell measurements use AE's
   baseline-aware population summarizer (`train_fit`, `val`, `test_classA`, `test_all_add`,
   `heldout_classA`, `seen_classA`, `heldout_all`, `seen_all` — raw + balanced accuracy, per-class
   recall, a majority-class baseline derived exclusively from the fitted pool's labels, and label
   frequencies), plus per-pair accuracy for the 20 held-out pairs and the init hash / exact parameter
   count per cell. TEST never selects checkpoints, widths, histories or seeds. Disclosure: the
   per-level seed offsets reuse the per-seed stride, so some `(seed, level)` cells share an init seed
   across levels while each level's five seeds remain distinct; every cell is deterministic and its
   init hash is recorded.
10. **Results.**
    * **AE baseline bridge (reported, never gating).** The Baseline level re-runs AE's exact
      `heldout`-variant fit seed on the same eligible pool under the same recipe with level offset 0:
      train fit, Class-A, held-out Class-A and best-val accuracy all match AE's stored fits
      **bit-exactly** (`max |Δ| = 0.00e+00`), with identical init hashes and identical parameter
      counts (28,655 / 28,506).
    * **Training fit improves with capacity where fitting occurs.** TCN Lfull: 0.374 ± 0.223 (Small)
      → 0.452 ± 0.486 (Baseline) → 0.553 ± 0.412 (Medium-large) → **0.770 ± 0.334** (Large); GRU L4:
      0.306 ± 0.091 (Small) / 0.302 ± 0.033 (Baseline) → **0.484 ± 0.303** (Medium-large). The 0.50
      fit bar is met at TCN L4/Baseline (0.550), GRU Lfull/Medium-small (0.550), TCN
      Lfull/Medium-large (0.553) and TCN Lfull/Large (0.770). The response is not monotone at every
      cell and remains seed-heterogeneous (e.g., TCN Lfull/Baseline 0.452 ± 0.486; GRU Lfull/Medium-large
      0.358 ± 0.063), so fit changes are not 4/5-consistent.
    * **Held-out transfer stays locked at chance across all widths.** Held-out Class-A **balanced**
      accuracy spans 0.072–0.143 (chance 0.10) and **raw** accuracy 0.078–0.164 across all 20
      reader × history × width cells; no cell's mean reaches the 0.30 material bar, and in **every**
      one of the 16 non-baseline cells **0/5 seeds** reach the +0.15 paired material gain. The largest
      mean paired Δ balanced is +0.041 (TCN Lfull/Medium-small); the largest mean paired Δ fit is
      +0.319 (TCN Lfull/Large). No transfer signal fires under the pre-registered rule.
    * **Accessibility unchanged.** Best Class-A raw accuracy across the ladder is **0.105**
      (TCN/Lfull/Baseline); best held-out Class-A raw accuracy is **0.163** (TCN/Lfull/Medium-small;
      0.136 at the best-accessibility cell); the Bayes−reader gap is **0.895** against U's Class-A
      Bayes ceiling of 1.000 at that history. V-A is **not reached** and V-B is **unchanged**.
11. **Statistical analysis:** the five frozen E checkpoints (seeds 42–46) are the independent units;
    the two readers, two histories and five widths are conditions, not replicates. All comparisons are
    per-seed **paired deltas against the Baseline width** (mean ± SD across seeds, per-seed values and
    seed counts retained in `experiments/experiment_ah_results.json` /
    `experiments/experiment_ah_grid_raw.json`). No multiplicity correction is applied; the outcome is
    reported continuously alongside the pre-registered descriptive categories. Thresholds are declared
    conventions reused from U/V/X, not new acceptance criteria: material gain 0.15, fit bar 0.50, weak
    0.25, held-out material 0.30, 4-of-5 seeds.
12. **Interpretation (observation vs interpretation).**
    * *Empirical observation:* raising reader width raises training fit where fitting occurs, while
      held-out-operand Class-A accuracy stays at chance for every width (0.072–0.143 balanced;
      0/5-seed material gains at every non-baseline cell); the Baseline level reproduces AE bit-exactly.
    * *Supported interpretation (pre-registered descriptive categories):* **H2 — fit without transfer
      — is selected**, framing **AH-B: the fit–transfer gap is capacity-insensitive over this ladder**.
      H1 is **falsified** (no capacity-associated generalisation at any width); H3 is **not observed**
      (several ladder cells fit the pool); H4 is **not selected**. The only 4/5-consistent pattern is
      the *absence* of held-out transfer.
    * *Not claimed:* AH does not show that capacity "cannot" solve the task, that some other
      reader/optimizer/data/objective regime would not transfer, or that the held-out information is
      absent. No RE:Reason frozen-representation, latent-state, architecture or accessibility claim
      follows; the ladder is an analysis-reader instrument, and the result is observational with
      respect to any mechanism.

**Literature connections.**
* Power et al. (2022) motivate treating training fit and held-out generalisation as distinguishable
  outcomes; AH's observation — fit rising with capacity while held-out stays at chance — is
  *consistent with* that distinction and does not show that more capacity or training fixes anything
  here.
* Gromov (2023) reports structured generalising solutions for modular arithmetic in small fully
  connected networks; AH's sequential register-update observations, bounded causal readers and
  held-out operand pairs differ, so that work is a hypothesis source, not directly comparable
  evidence.
* He et al. (2024) frame out-of-distribution generalisation across modular arithmetic functions;
  AH's held-out operand pairs are a systematic-generalisation-style condition, but that work studies
  transformers and in-context learning, not RE:Reason's recurrent register machine or these readers.
* Veličković & Blundell (2021) position algorithm learning and generalisation as a research area; AH
  measures one bounded-reader family's behaviour within it and licenses no algorithmic-reasoning
  capability claim.
* Schwarzschild et al. (2021) study recurrent networks that generalise to harder instances via
  additional inference-time computation; AH varies reader *width* (a static capacity axis), not
  per-instance recurrence depth, so it does not test that mechanism.
* Smolensky (1990) supplies the variable-binding framing relevant in principle to register IDs,
  operand values and binding interference; AH identifies no binding mechanism and does not require
  tensor-product representations.

**Answers to the required final questions.**

1. **Does increasing reader capacity produce held-out-operand generalisation of the modular
   arithmetic rule?** **No.** Across all 16 non-baseline ladder cells, held-out Class-A balanced
   accuracy stays at chance (0.072–0.143; chance 0.10; raw 0.078–0.164), no cell reaches 0.30, and
   **0/5 seeds** reach the +0.15 paired material gain at any width — H1 is falsified.
2. **Does increasing capacity improve training fit?** **Yes where fitting occurs.** TCN Lfull
   0.374 → 0.452 → 0.553 → 0.770 (Small → Baseline → Medium-large → Large); GRU L4 0.306 → 0.484
   (Small → Medium-large); the 0.50 fit bar is met at four cells (TCN L4/Baseline 0.550, GRU
   Lfull/Medium-small 0.550, TCN Lfull/Medium-large 0.553, TCN Lfull/Large 0.770), but the fit
   response is not 4/5-consistent (large per-cell SDs).
3. **Is the fit–transfer gap capacity-insensitive over this ladder?** **Yes** (descriptive category):
   every fitting cell keeps held-out balanced accuracy ≤ 0.25 (all cells ≤ 0.144) → **H2/AH-B**.
   The gap does not close at any width.
4. **Is the fitting failure itself capacity-insensitive (H3)?** **Not selected** — some ladder
   cells fit, so AH does not observe a uniform fitting failure; the dominant category is
   fit-without-transfer.
5. **Is the result consistent across the independent units and conditions?** For transfer, yes:
   0/5 seeds meet the material gain in every non-baseline cell. For fit, no: the fitting response is
   seed-heterogeneous; only the absence of transfer is uniform.
6. **Does any width change V's accessibility verdict?** **No.** Best Class-A 0.105; best held-out
   0.163; Bayes−reader gap 0.895 → **V-A not reached, V-B unchanged**.
7. **Does AH license any RE:Reason modification, intervention or Phase-2 step?** **No.** AH trains
   only external analysis readers on frozen E populations; no RE:Reason parameter was trained or
   mutated; all E/F checkpoints are hash-verified unchanged; all findings are conditional
   analysis-reader statements. The recommended next step is the Phase-1 evidence synthesis and
   go/no-go decision, not another fit grid.

13. **What the experiment DOES establish:** (i) **H1 is falsified** — no width in the ladder produces
    materially or seed-consistently better held-out-operand accuracy (stays at chance; 0/5 seeds
    reach the material gain at every width); (ii) reader capacity improves training fit markedly
    where fitting occurs (TCN Lfull 0.374 → 0.770; GRU L4 0.306 → 0.484), so the reader-side
    fit–transfer gap persists and grows with capacity; (iii) the ladder is a valid controlled
    capacity manipulation (exact implementation-derived parameter counts, every level within 4.0×
    the baseline, Baseline reproducing V's deterministic width selection, and the Baseline level
    reproducing AE's `heldout`-variant fits bit-exactly, `max |Δ| = 0.00e+00`, with matching init
    hashes); (iv) V-A is not reached and V-B is unchanged (0.105 / 0.163, gap 0.895).
14. **What the experiment DOES NOT establish:** that capacity "cannot" solve the task, or that a
    different reader family, objective, optimizer, data regime or wider ladder would not transfer;
    that the held-out information is absent; any RE:Reason frozen-representation, latent-state or
    architecture property; any causal claim about width (the capacity manipulation is observational
    with respect to any mechanism); and anything beyond frozen E, this target, these histories and
    this protocol.
15. **Decision:** **H2_fit_without_transfer (AH-B)** — the fit–transfer gap is capacity-insensitive
    over this ladder. Pre-registered rule trace: H1 cells = none; fit records = 4 (including the
    Baseline TCN L4 cell); all fitting cells are held-out-weak → H2; H3 not selected; H4 not
    selected. Accessibility: **V-B_sequence_limited** (unchanged; V-A not reached).
16. **Next experiment / recommendation:** **close the Phase-1 empirical cycle with an evidence
    synthesis across U–AH and an explicit go/no-go decision** (no new fit grid, no new reader or
    width family, no RE:Reason modification; the synthesis is specified here and not started in this
    run).
17. **Reproduction commands:**
    ```
    PYTHONPATH=src python experiments/run_experiment_ah.py
    PYTHONPATH=src python experiments/analyze_experiment_ah.py --jobs 16
    # reassemble from the retained raw per-fit evidence without re-running the fits:
    PYTHONPATH=src python experiments/analyze_experiment_ah.py --jobs 16 --use-raw
    PYTHONPATH=src uv run python -m pytest tests/test_experiment_ah.py -v
    ```
18. **Artifact list:**
    * report: `docs/reports/experiment_ah_capacity_ladder.md`
    * code: `experiments/run_experiment_ah.py`, `experiments/analyze_experiment_ah.py`
    * data: `experiments/experiment_ah_capacity_ladder_meta.json` (source identity, ladder with exact
      parameter counts, eligible-pool support, checkpoint hashes), `experiments/experiment_ah_results.json`
      (aggregates, paired deltas, decision, accessibility, AE bridge),
      `experiments/experiment_ah_grid_raw.json` (100 fits: per-cell population metrics with balanced
      accuracy, per-class recall and training-derived baselines, per-pair held-out accuracy, init
      hash, exact parameter count, best-val update)
    * figures: `docs/reports/figures/experiment_ah_width_vs_fit.png`,
      `experiment_ah_width_vs_heldout_bal.png`, `experiment_ah_paired_deltas.png`,
      `experiment_ah_per_seed_heldout.png`, `experiment_ah_param_vs_acc.png`,
      `experiment_ah_decision_summary.png`
    * tests: `tests/test_experiment_ah.py`
    * checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
      (SHA-256 verified before/after; hashes recorded in the meta artifact)
    * reused unchanged (read only): `experiments/experiment_ad_qualified_access.npz`;
      `experiments/experiment_ad_qualified_access_meta.json`; `experiments/experiment_v_sequence_access.npz`;
      `experiments/experiment_u_results.json`; `experiments/experiment_ae_grid_raw.json`;
      `analyze_experiment_ad.py` (source loading/features/conditions), `analyze_experiment_ae.py`
      (population summarizer), `analyze_experiment_ab.py` (trainer), `analyze_experiment_u.py`
      (histories/populations), `analyze_experiment_v.py` (readers, seeds, U baseline)
19. **Scientific guardrails:** AH is an analysis-reader training experiment, not a RE:Reason training
    or representation experiment; findings are conditional on the tested readers, ladder, data and
    protocol. The analysis readers never enter frozen model inference, and no RE:Reason parameter was
    trained or mutated; frozen E/F checkpoints are SHA-256-verified unchanged. The capacity
    insensitivity is scoped to **the tested ladder under the fixed optimization protocol and budget**,
    not stated as "capacity cannot solve it" or as an architectural limit. Balanced accuracy and
    per-class recall are reported so composition-driven raw-accuracy changes are not mistaken for rule
    recovery; majority-class baselines are derived exclusively from the corresponding training labels.
    The five seeds are the independent units; readers/histories/widths are conditions. Every ladder
    level is reported, including non-fitting ones; no family was added after seeing results; the
    pre-registered ladder, histories, target and seeds are fixed. The AE baseline bridge is reported,
    never gating. Observational; no causal claim; no intervention licensed; no RE:Reason modification;
    no Phase 2. All prior reports and artifacts are unchanged.
