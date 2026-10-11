# Experiment AD — Qualified-Recipe Re-evaluation of the ADD-Target Access Test

1. **Date:** 2026-10-11
2. **Status:** Completed — measurement-only analysis-reader experiment (frozen E primary and F control
   checkpoints used only to define test populations and reproduce baseline metrics; no RE:Reason
   training; no optimizer on RE:Reason parameters; no architecture / encoder / recurrent-core / state /
   gate / candidate / commit / readout modification; no state replacement; no candidate manipulation;
   no feature editing; no oracle/reconstructed state as a reader feature; no branching/verifier; no
   counterfactual rollout; no Phase 2). Result: **AD-B (no material optimizer effect)** on the
   optimizer axis, with the **V-B (sequence-limited)** accessibility verdict unchanged under the
   qualified `adamw` recipe.
3. **Research question:** Experiments V/W/X evaluated the predeclared parameter-bounded causal readers
   (GRU hidden 85; causal TCN channels 52) with **Adam** and PyTorch's default initialization. AB/AC
   established that the causal TCN's fitting behaviour is optimizer-recipe sensitive: the AB-qualified
   `adamw` recipe lets the TCN fit the directly observable `op_dest` control and the small-subset ADD
   `y_true` target (though the small-subset fit transfers nothing). AD asks: **does changing Adam to
   the AB-qualified AdamW recipe change the TCN's access to the ADD target under V's original full
   training-data and evaluation protocol?** Four questions are kept separate: (1) does the optimizer
   change materially affect Class-A target access? (AD-A/AD-B); (2) does it improve generalization to
   held-out operand pairs? (paired-delta axis); (3) does it cross V's existing sequence-accessibility
   criterion? (V-A/V-B axis); (4) does any result support a claim about RE:Reason's frozen
   representation? — **out of scope and left unanswered**.
4. **Existing baseline:** V (V-B) fit the two bounded causal readers with Adam/3e-3/1e-2, 200
   full-batch epochs, class-balanced CE, best-on-VAL checkpointing, on ordinary-ADD TRAIN (cap
   1,500/seed), tuned on ordinary VAL (cap 800/seed), evaluated once on the untouched canonical TEST,
   and reported Class-A (E) 0.083 (GRU) / 0.112 (TCN) at L=4 and 0.080 / 0.109 at the full prefix,
   held-out ≤0.25 everywhere, Bayes−reader gap 0.887. W/X reused the same readers at larger data and
   budget without change. AC then showed the AB-qualified `adamw` recipe lifts the TCN's *fitting*
   (1.0000 train accuracy on Y's 256-subset) without transfer. V/W/X's ADD-target-access conclusions
   were produced with the **unqualified** Adam/default TCN instrument.
5. **Hypothesis:** if the V/W/X weak accessibility were a TCN optimizer instrument artifact, then
   re-running V's exact access test with the AB-qualified `adamw` recipe should lift Class-A and/or
   held-out operand accuracy materially (≥ +0.15 paired) over the contemporaneous Adam/default
   reference; if it were not, the accessibility stay weak and/or non-transferring.
6. **What is frozen:** the RE:Reason E/F checkpoints (SHA-256 verified before and after); V's raw
   observation history construction (`x_t` (21) + 1 validity bit = 22 features; zero left-padding;
   validity mask; no future observation); V's reader implementations (`av.build_reader`: causal GRU
   hidden 85; causal dilated TCN 52 channels, kernel 3, dilations `[1,2,4,8]`, four causal conv
   layers; 10-class heads); V's splits, populations, history grid `{L1,L2,L4,L8,L16,Lfull}` and
   per-seed assignment; V's target `y_true = (operand_a + operand_b) mod 10`; U's Bayes ceilings
   (reused unmodified); V's stored historical results (read-only).
7. **What changed:** exactly one factor — the optimizer family. Two contemporaneous conditions were
   run for both readers: `adam_ref` (Adam, default init; a reference bridge, not a replacement for
   V's historical result) and `adamw` (AdamW, default init; the AB-qualified recipe). Learning rate
   3e-3, weight decay 1e-2, AB's optimizer defaults (betas `(0.9,0.999)`, eps 1e-8), **200 full-batch
   updates** (V's epoch count), class-balanced CE, best-on-VAL selection, V's deterministic reader-seed
   convention with the optimizer condition **not** part of the seed. No scheduler, no additional
   optimizer, no alternative initialization, no hyperparameter search. Training variant `main` (all
   ordinary-ADD train pool rows) and `heldout` (excluding the 20 operand pairs `(a,b)`, `b ∈ {0,5}`)
   were both run across all six histories. Grid: 2 readers × 2 conditions × 6 histories × 2 variants ×
   5 seeds = **240 fits**.
8. **Dataset / protocol:** V's exact artifact `experiments/experiment_v_sequence_access.npz` was
   reused verbatim (copied byte-for-byte into the AD artifact). Verified before and after: source
   SHA-256; exact array identity; 10-class coverage in trainord/valord/test for both E and F;
   `y_true` derivation exact; zero trainord/valord/test sequence overlap and no test leakage; V's
   causal history construction (padded slots zero, no observation index > t, cross-checked against an
   independent builder); frozen E/F checkpoint SHA-256. Splits: ordinary TRAIN ADD (cap 1,500/seed) →
   fit, ordinary VAL (cap 800/seed) → tune, canonical TEST (15,763 ADD transitions pooled over five
   seeds) → evaluate once. Populations: Class-A (primary; E `cand_ok == 0`), matched candidate-correct,
   all candidate-correct, all ADD. E seeds 42–46 are the independent units; F is a control.
9. **Experimental procedure:** For each `(seed, reader, history, variant)` a single deterministic
   initialization is produced from V's `av.reader_seed` and shared by both conditions (verified by
   state-dictionary hash equality); the two conditions then differ only by `torch.optim.Adam` vs
   `torch.optim.AdamW`. Features are V's exact `av.reader_input`; the trainer is AB's exact
   `ab.train_ab` (identical ops/order to V's `train_reader`); evaluation uses `ax.predict_state` on the
   best-on-VAL state. Selection is VAL-only; TEST is never used to select a checkpoint, history,
   reader or condition. Paired `adamw − adam_ref` deltas are computed per seed.
   Preregistered decision (declared before aggregate inspection): **AD-A** if at L4 or Lfull the mean
   paired improvement in Class-A `y_true` accuracy **or** held-out Class-A accuracy is ≥ +0.15 (V's
   material-gain convention); **AD-B** otherwise. Sequence accessibility is a **separate** axis using
   V's unchanged bar: **V-A** iff Class-A test ≥ 0.50 **and** held-out Class-A ≥ 0.30 at L4 or Lfull.
10. **Results.**
    * **Optimizer-effect axis (AD-B).** The largest mean paired improvement anywhere is **+0.0165**
      (TCN, L4, held-out Class-A) — an order of magnitude below the +0.15 bar; all other paired means
      are smaller or negative. TCN Class-A paired deltas are **negative** at both decisive horizons
      (L4 **−0.0180**; Lfull **−0.0124**); TCN held-out is **+0.0165** (L4) and **−0.0383** (Lfull).
      GRU Class-A is +0.0120 (L4) / +0.0145 (Lfull); GRU held-out is −0.0400 (L4) / −0.0598 (Lfull).
      The held-out results are seed-dependent (e.g. TCN Lfull per-seed deltas
      `[−0.049, +0.114, −0.016, −0.226, −0.014]`). **No condition/metric/history crosses the bar.**
    * **Access metrics (main, E).** Class-A: TCN `adam_ref` 0.112 (L4) / 0.109 (full) vs `adamw`
      0.094 / 0.097; GRU `adam_ref` 0.083 / 0.080 vs `adamw` 0.095 / 0.094. All-ADD: GRU rises under
      `adamw` (L4 0.116 → 0.207; full 0.147 → 0.213) while TCN is mixed (L4 0.176 → 0.147; full
      0.214 → 0.185). Candidate-correct and matched populations stay ≈ chance–0.23.
    * **Fitting (main, E; best-on-VAL train accuracy).** The optimizer recipe changes *fitting*
      substantially: TCN 0.174 → 0.442 (L4) and 0.216 → 0.655 (full); GRU 0.142 → 0.858 (L4) and
      0.181 → 0.649 (full). The readers now fit the ordinary-ADD pool far better, yet Class-A and
      held-out accuracy do not rise — a **full-scale fit-without-transfer** signature (AC's
      small-subset result reproduced at V's scale).
    * **Held-out operand protocol.** TCN `adamw` held-out Class-A 0.122 (L4) / 0.136 (full), seen
      0.089 / 0.096; `adam_ref` held-out 0.105 / 0.175, seen 0.107 / 0.087. No cell approaches the
      0.30 material or 0.50 recovery bars.
    * **Reference bridge.** `adam_ref` reproduces V essentially exactly: max Class-A drift **0.0080**
      (TCN/L8), max held-out drift **0.0000**; all other cells ≤ 0.0006. E and F are identical on
      all-ADD (max |E−F| = 0.0). Every run finite.
    * **Bayes ceiling / accessibility.** U's Class-A ceiling is 0.9994 (L4) / 1.000 (full). The best
      reader under the qualified recipe is TCN/Lfull at Class-A 0.0968 (held-out 0.1364), Bayes−reader
      gap 0.903; under `adam_ref` the best is TCN/L4 at 0.112 (held-out 0.105), gap 0.887. No reader
      reaches 0.50; no reader reaches 0.30 held-out.
11. **Statistical analysis:** five frozen E checkpoints are the paired units. For each history the
    `adamw − adam_ref` difference is computed per seed; pooled means, standard deviations, sign counts
    and per-seed deltas are retained for every reader/history/metric in
    `experiments/experiment_ad_results.json` and `experiments/experiment_ad_grid_raw.json`. The largest
    effect (+0.0165) is well inside the ±0.15 material band, and its per-seed signs are mixed
    (2 positive, 1 zero, 2 negative), so it is not a consistent effect. No multiplicity correction is
    applied because no comparison approaches the preregistered bar.
12. **Interpretation (observation vs interpretation).**
    * *Empirical observation:* under V's exact full-pool protocol, switching the reader optimizer from
      Adam to the AB-qualified AdamW does **not** materially change Class-A or held-out ADD-target
      accuracy for either reader (AD-B), even though it substantially increases the readers' ability to
      fit the training pool.
    * *Supported interpretation:* the V-B weak ADD-target accessibility is **not** a TCN optimizer-recipe
      artifact; V's conclusion is robust to the AB-qualified recipe. The earlier V/W/X TCN-based
      conclusions are therefore **not** optimizer-confounded in the way AC raised as a possibility.
    * *Unresolved / not claimed:* this does **not** show the ADD rule is unlearnable, that the
      information is absent from RE:Reason's latent state, or that a larger/other reader family could
      not access it. The readers are separate analysis-only models on a computed label.
13. **What the experiment DOES establish:** (i) the AB-qualified `adamw` recipe does not materially
    improve the two readers' Class-A or held-out ADD-target access under V's exact protocol (AD-B);
    (ii) the readers' *fitting* of the ordinary-ADD pool improves substantially under `adamw`, but the
    improvement does not transfer to the untouched TEST — reproducing AC's fit-without-generalization
    at V's scale; (iii) the contemporaneous Adam/default reference reproduces V's stored result almost
    exactly (max Class-A drift 0.008), so the comparison is made on identical source arrays, readers,
    histories, seeds and protocol; (iv) V's sequence-accessibility verdict remains V-B under the
    qualified recipe.
14. **What the experiment DOES NOT establish:** (i) any property of RE:Reason's frozen representation,
    latent state, architecture or sufficiency (question 4 is unanswered by construction); (ii) that the
    Class-A information is absent from the raw history or from RE:Reason — U's Bayes ceiling is 0.9994
    at L4; (iii) that ADD is unlearnable, or that the bounded-reader family is the only possible reader;
    (iv) a causal claim about the optimizer mechanism (AB/AC already note the weight-decay-coupling vs
    initialization factors are not separated, and AD runs only `adam_ref` vs `adamw`); (v) any
    generalization from a fitting improvement alone — fitting without transfer is explicitly **not**
    arithmetic recovery.
15. **Decision:** **AD-B — no material optimizer effect** (neither Class-A `y_true` nor held-out
    Class-A accuracy improves by +0.15 at L4 or Lfull; largest mean paired delta +0.0165, TCN L4
    held-out, seed-dependent). Separately, the sequence-accessibility assessment is **V-B
    (sequence-limited) for both conditions** (best Class-A 0.0968 qualified / 0.112 reference; held-out
    0.136 / 0.105; Bayes−reader gap 0.887–0.903). AD-A and the V-A recoverability bar are both unmet;
    the two verdicts are reported as separate axes as preregistered.
16. **Next experiment:** **Experiment AE — bounded-reader training-objective / representation-shift
    diagnostic at V's scale (measurement-only, frozen checkpoints):** AD shows the optimizer recipe
    removes the *fitting* obstacle at full scale (train accuracy 0.44–0.86 under `adamw`) without any
    transfer to the untouched TEST. The single most informative next question is **what the readers fit
    under `adamw`** — whether the high-training-accuracy solutions are generic memorization of the
    1,500-transition pool (held-out-operand and sequence-disjoint splits stay at chance) or a weak
    partial rule that a different target/label organization or a probe of the learned reader
    representation could expose — holding V's arrays, readers, histories and seeds fixed and varying
    only the reported object (train-vs-held-out-operand gap, per-seed, at the best-on-VAL state). It is
    specified here and **not started** in this run.
17. **Reproduction commands:**
    ```
    PYTHONPATH=src python experiments/run_experiment_ad.py \
        --v-npz experiments/experiment_v_sequence_access.npz \
        --output experiments/experiment_ad_qualified_access.npz
    PYTHONPATH=src python experiments/analyze_experiment_ad.py --jobs 16
    # reassemble from the retained raw per-seed evidence without re-running the fits:
    PYTHONPATH=src python experiments/analyze_experiment_ad.py --jobs 16 --use-raw
    PYTHONPATH=src uv run python -m pytest tests/test_experiment_ad.py -v
    ```
18. **Artifact list:**
    * report: `docs/reports/experiment_ad_qualified_access.md`
    * code: `experiments/run_experiment_ad.py`, `experiments/analyze_experiment_ad.py`
    * data: `experiments/experiment_ad_qualified_access.npz`,
      `experiments/experiment_ad_qualified_access_meta.json`,
      `experiments/experiment_ad_results.json`, `experiments/experiment_ad_grid_raw.json`
      (raw per-seed evidence: 120 records × 2 optimizer conditions, retaining full trajectories and
      per-population metrics so the report can be independently reassembled)
    * figures: `docs/reports/figures/experiment_ad_{classA_vs_history,heldout_vs_history,optimizer_delta_classA,optimizer_delta_heldout,per_seed_delta,training_trajectory,fit_vs_generalization,bayes_vs_reader,sequence_access_summary,v_bridge,decision_summary}.png`
    * tests: `tests/test_experiment_ad.py`
    * checkpoints: `experiments/checkpoints/{E_candidate,F_combined}_seed{42..46}_3000steps.pt`
    * reused unchanged (read only): `experiments/experiment_v_sequence_access.npz`;
      `analyze_experiment_v.py` (readers, seeds, U baseline loader); `analyze_experiment_u.py`
      (history/population utilities); `analyze_experiment_ab.py` (exact optimizer/init/trainer);
      `analyze_experiment_x.py` (`build_model`/`predict_state`); `experiments/experiment_v_results.json`;
      `experiments/experiment_u_results.json`.
19. **Scientific guardrails:** AD is a controlled analysis-reader experiment. AdamW qualification on
    `op_dest` (AB) and a small-/full-pool `y_true` fit (AC/AD) do **not** establish arithmetic
    generalization or any property of RE:Reason's frozen latent state. The readers never enter frozen
    model inference and no RE:Reason parameter was trained or mutated. Negative and inconclusive
    findings are preserved (no optimizer/access gain; seed-dependent held-out deltas). No causal claim
    is made about the optimizer mechanism. No dataset/target change, no optimizer-grid expansion, no
    RE:Reason modification, no Phase 2. All historical reports Q–AC are unchanged.

## Amendment — 2026-10-11

Recorded after completion: the AD run's own reference bridge is the contemporaneous `adam_ref`
condition, not a rewrite of V; V's stored results remain untouched in
`experiments/experiment_v_results.json` and `docs/reports/experiment_v_sequence_access.md`. No
completed report was modified to accommodate this result.
