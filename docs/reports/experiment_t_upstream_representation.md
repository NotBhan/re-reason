# RE:Reason — Experiment T: Upstream Representation Localisation of the Class-A Information

**Date:** October 8, 2026
**Status:** Completed — Measurement only (frozen checkpoints; no model training; no optimizer on model weights; no architecture/encoder/recurrent-core/gate/commit/readout modification; no inference-time intervention; no hidden-state replacement; no feature editing; no branching/verifier/oracle; no counterfactual rollout; no Q/R/S rerun; no Phase 2). Analysis-only probes are fit on a regime's train split, tuned on its validation split, and evaluated once on the untouched canonical-TEST Class-A hard tail; they never enter model inference. Classification: **T-D — weak throughout** for E and F.
**Frozen models:** `E_candidate` (primary) and `F_combined` (secondary control), seeds 42–46, 226,358 parameters, SHA-256-pinned (hashes re-verified against G/L/M/N/O/P/Q/R/S metadata)
**Data:** `experiments/experiment_t_upstream_representation.npz`, `experiments/experiment_t_upstream_representation_meta.json`, `experiments/experiment_t_results.json`
**Runners:** `experiments/run_experiment_t.py`, `experiments/analyze_experiment_t.py`, tests `tests/test_experiment_t.py`

---

## 1. Title / Date / Status

Experiment T — Upstream Representation Localisation of the Class-A Information. Measurement-only, frozen; **T-D (weak throughout)**. No intervention; no Phase 2. Q, R and S reports are untouched; T is appended as new evidence.

## 2. Executive Answer

S closed the candidate-side question negatively (S-C): no internal candidate activation provides generalizing recoverability, and the candidate input `cand_input = [h_reset, e_t]` is itself weak. T asks the upstream question: **which of the frozen pipeline's representations before the candidate network — the observation encoder's inner features, its output `e_t`, the raw pre-transition state `S_prev`, the gate preactivation `gate_pre` / postactivation `f_gate`, the pre-gated state `h_reset`, or `cand_input` — first fails to carry generalizable information about the true written ADD value, and which source of `cand_input` is responsible?**

T captures every predeclared upstream stage read-only (forward hooks inside the existing `forward_with_states`) and probes them with R/S's exact probe family under R/S's D0/D0m/D1/D1x design, on the **same** untouched canonical-TEST Class-A hard tail, with a held-out-operand control.

**No upstream stage provides generalizable (held-out-operand) recoverability, and no stage reaches the recoverability regime.**

* **The encoder output is structurally incapable of carrying the operands.** In-distribution `e_t` decodes the **destination register at 1.000** but `operand_a` at **0.110** and `operand_b` at **0.136** (chance = 0.10). `e_t` is a function of the current observation `x_t` alone, and an ADD step's observation contains no operand values; the probe result confirms the architecture-level expectation. Consequently the target information for an ADD must come from the **state pathway**, not the observation encoding.
* **The state pathway is descriptively stronger in-distribution but only marginally.** Along `S_prev → gate_pre → f_gate → h_reset` the D1x/MLP-32 hard-test accuracy rises **0.167 → 0.191 → 0.213 → 0.239** (state-path best `h_reset` = **0.239**), versus the encoder-path best `enc_h1_pre` = **0.141**. The state-minus-encoder difference is **+0.098** — *just below* the pre-registered 0.10 material bar — and is **not** accompanied by a held-out advantage (state best held-out 0.113 vs encoder best held-out 0.118, Δ = **−0.005**).
* **The state-path "gain" is memorization.** Every state-path stage **collapses on held-out operand combinations**: `gate_pre` 0.208→0.078, `f_gate` 0.245→0.065, `h_reset` 0.261→0.046, `cand_input` 0.286→0.045 (seen→held-out). The encoder stages, which cannot memorize operands, barely move (raw_obs 0.127→0.118, `e_t` 0.147→0.110). `C_t` reproduces R/S exactly (D1x/MLP-32 **0.463**) and also collapses (0.493→0.176).
* **The sigmoid is not a bottleneck.** `gate_pre → f_gate` changes recoverability by **+0.022** (D1x) / **+0.024** (D0) in-distribution (slightly *more* decodable after the sigmoid) and **−0.013** on held-out operands; no material drop occurs across the gate nonlinearity.
* **No stage reaches the 0.50 recoverability bar** (max stage D1x = `C_t` 0.463); **no stage reaches 0.25 held-out** (max held-out = `C_t` 0.176). `C_t` reproduces R exactly (D0 linear 0.244, D0/MLP-32 0.238, D1x/MLP-32 0.463). Wrap ≈ no-wrap, raw-sum not easier than modular, E and F agree.

Under the pre-registered rule this is **T-D — weak throughout**: the limitation is already present across the tested upstream representations, the state path's in-distribution advantage is sub-material and does not survive held-out operands, and no upstream representation is robustly (held-out) recoverable. The result is **observational**, **not causal**, and **no intervention is justified**.

## 3. Existing Baseline

* **G–I**: the commit is bit-exact; a real candidate→commit residual (~5.5% of candidate-correct adds).
* **L/M/N/O**: the third basin is latent placement (B), a systematic **parity/modular** geometry (M-A), **state-alignment-dominant** (N-B); parity-direction removal is non-specific and harmful (**O-D**).
* **P**: the sequence-level first-error budget is **candidate-generation-dominated** (A 59.7% T=10 / 67.2% T=20).
* **Q/R**: the true written value is only weakly recoverable from the frozen `C_t` on the Class-A population (linear 0.267/0.253, nonlinear 0.233/0.261; **Q-C**), and bounded capacity / size-matched hard-tail training do not help; the only regime raising apparent accuracy is **operand-combination memorization** (**R-D**).
* **S**: probing the candidate network's **internal** activations, no internal stage materially beats `C_t` (best internal gain −0.031, 0/5 seeds), the hard-trained chain weakly *increases* toward the output, and held-out-operand generalisation fails at every stage (**S-C**). The candidate input is itself weak (`cand_input` D0 0.128 / D1x 0.246). S explicitly left the question **upstream of the candidate network** open — the observation encoder's inner features, `S_prev`, the gate pre/postactivation, and `h_reset`.

## 4. Research Question

> Where, before the candidate network, does the Class-A target information first become inaccessible in a probe-recoverable form — the observation encoder's internal stages or its output `e_t`, the raw pre-transition state `S_prev`, the gate preactivation/prediction, or the pre-gated state `h_reset` — and which source of `cand_input = [h_reset, e_t]` is responsible for the weakness?

T distinguishes (i) **T-A** encoder-limited, (ii) **T-B** state-limited, (iii) **T-C** source-fusion/candidate-input limited, (iv) **T-D** weak throughout, (v) **T-MIX**. These are localisation labels, not causal claims.

## 5. Hypotheses

* **H-A|T-A**: the encoder output/internal representation is weak while the state pathway is comparatively informative.
* **H-B|T-B**: the encoder carries useful target/operand information that is lost in `S_prev`/`gate_pre`/`f_gate`/`h_reset`.
* **H-C|T-C**: `e_t` and `h_reset` each retain material recoverability but the fused `cand_input` does not.
* **H-D|T-D**: no upstream source provides robust generalizable recoverability — the limitation is already present before the tested upstream representations.
* **H-E|T-MIX**: different sources or arithmetic subtypes disagree.

## 6. Preregistration (declared before aggregate inspection)

**Activation set (from the actual implementation — see §8):** `raw_obs(21)`, `enc_h1_pre(64)`, `enc_h1_post(64)`, `enc_out_pre(256)`, `e_t(256)`, `S_prev(212)`, `gate_pre(212)`, `f_gate(212)`, `h_reset(212)`, `cand_input(468)`, `C_t(212)`. All 11 stages are reported; none is dropped. Encoder path = `raw_obs → enc_h1_pre → enc_h1_post → enc_out_pre → e_t`; state path = `S_prev → gate_pre → f_gate → h_reset`; bridge = `cand_input → C_t`. KEY stages = `{e_t, S_prev, gate_pre, f_gate, h_reset, cand_input, C_t}`.

**Probe family (R/S's, fixed, not extended after results):** `P1` linear ridge (closed-form, class-balanced, λ∈{1e-2,1e-1,1,10,100} tuned on val); `P2..P5` MLP 16/32/64/128 (tanh, Adam 3e-3, weight-decay 1e-2, 200 epochs, best-on-val state, fixed seed, balanced class weights). The probe primitives are imported **verbatim** from `analyze_experiment_s.py`, so the family is literally identical.

**Capacity × distribution (predeclared):** NON-key stages × {linear, MLP-32} × {D0, D0m, D1, D1x}; KEY stages × {linear, MLP-16, MLP-32} × {D0, D0m, D1, D1x} plus {MLP-64, MLP-128} × {D1x} (the bounded full sweep on the key stages). **Distributions (R/S's):** `D0` ordinary train (1500/seed), `D0m` size-matched ordinary (=|hard-train|), `D1` designated hard-train, `D1x` augmented i.i.d. hard pool. **Evaluation is always the same untouched canonical-TEST Class-A hard tail.**

**Held-out operand protocol:** 20 pairs `{(a,b): b∈{0,5}}` (all 10 targets represented); the probe is fit excluding them and evaluated once on hard-test rows with held-out vs seen pairs (D0 and D1x).

**Explanatory secondary targets (D1x/MLP-32):** destination (4-way), `operand_a` (10-way), `operand_b` (10-way). **Operation is constant (= ADD) on this ADD-only population** and is therefore reported as structurally non-informative rather than probed.

**Localization verdict (checked in order; bars are declared conventions):**
* **T-MIX** — subtype spread |wrap−no-wrap| ≥ 0.15 at ≥ 2 KEY stages, or E/F verdict categories disagree.
* **T-C** — `e_t` ≥ 0.50 AND `h_reset` ≥ 0.50 AND `cand_input` ≤ min(e_t, h_reset) − 0.10, with the ordering preserved on held-out operands.
* **T-B** — encoder-path best ≥ state-path best + 0.10 AND encoder-path best ≥ 0.50 AND the ordering preserved on held-out operands.
* **T-A** — encoder-path best ≤ 0.25 (weak) AND state-path best ≥ encoder best + 0.10 AND state best held-out ≥ encoder best held-out + 0.05 AND reproduced ≥ 4/5 seeds.
* **T-D** — both pathways weak (≤ 0.25) or no material difference, AND no stage reaches the 0.50 bar, AND held-out stays ≤ 0.25 at every stage.
* else **T-MIX**.

Recoverability = `(acc − 0.10)/(1 − 0.10)` is reported alongside raw accuracy. Inference: the five E checkpoints are independent units; F is a control. Every predeclared stage and capacity is reported; no test-based stage/capacity selection.

## 7. Frozen Components

Recurrent core, observation encoder, gate network, candidate MLP, convex commit rule, terminal readout, benchmark, task generator, optimizer, loss. **Nothing modified.** All model execution under `torch.no_grad()`; checkpoints SHA-256 verified before and after. Upstream activations are captured with **read-only forward hooks** registered on the existing encoder submodules (`encoder.net[0..3]`), gate submodules (`w_f_ih`, `w_f_hh`) and candidate submodules (`cand_mlp[0..2]`), and removed in a `finally`; they cannot alter the frozen computation (verified bit-exact against manual recomputation in the tests). The only fitted objects are analysis-only probes — separate objects, never wired into `forward`, `decision_head`, or any rollout.

## 8. Actual Upstream Architecture / Activation Map

The frozen Model E/F pipeline (`NonlinearCandidateMGUModel`, 226,358 params; state_dim 212, encoder hidden 64, encoder out 256, candidate hidden 146):

```
    raw_obs      = x_t                           (21)
    enc_h1_pre   = net[0](x_t)      Linear(21,64) (64)
    enc_h1_post  = net[1]           GELU          (64)
    enc_out_pre  = net[2]  Linear(64,256)         (256)
    e_t          = net[3]  LayerNorm(256)         (256)   # observation-encoder output
    S_prev       = S_(t-1)                        (212)   # pre-transition committed state
    gate_pre     = W_f_ih e_t + W_f_hh S_prev      (212)   # gate preactivation
    f_gate       = sigmoid(gate_pre)               (212)   # gate postactivation
    h_reset      = f_gate ⊙ S_prev                 (212)   # pre-gated state
    cand_input   = [h_reset, e_t]                  (468)   # candidate input
    C_t          = tanh(cand_mlp(cand_input))      (212)   # final candidate (downstream bridge)
```

`ObservationEncoder.net = Sequential(Linear(21,64), GELU, Linear(64,256), LayerNorm(256))`. The predeclared stage map is read from this implementation; the activation set and dimensions are validated against the actual modules in `tests/test_experiment_t.py`. `gate_pre` is captured from the actual submodule outputs (`w_f_ih(e_t) + w_f_hh(S_prev)`) inside the same forward pass; the bit-exactness of the capture is tested (`sigmoid(gate_pre) == f_gate` to float tolerance; `cand_input == [h_reset, e_t]` exactly; every stage equals its manual recomputation).

## 9. Dataset / Split

The **DisjointSplitManager** splits (no sequence overlap by construction): TRAIN 6,000 / VAL 1,200 / canonical TEST 1,200×10 (stream `90042+seed`), sequence-disjoint. A fresh i.i.d. stream (base `810000+seed`, 5,000 sequences) supplies the augmented hard pool; it is verified sequence-disjoint from train/val/test. `S_prev` is the pre-transition committed state (`S_{t-1}`); `seq` is the sequence index; the collector is chunked (512 sequences) to bound peak memory. Collection is bit-identical across two runs. Captured per ADD transition: all 11 stage vectors + `y_true, raw_sum, cand_pred, cand_ok, m_cand, offpath, norm_C, norm_dC, op_dest, src, operand_a, operand_b, wrap, repeated, t, seq, seed`.

## 10. Class-A Population

Q's Class-A definition, unchanged: an **ADD transition whose frozen candidate written-register decode ≠ the true written value**. Per-seed n (E): 448 / 330 / 363 / 327 / 337 (mean 361); canonical test ADD totals (E): 3,199 / 3,196 / 3,171 / 3,125 / 3,072 (reproducing R/S exactly). The designated **hard-train** split is the Class-A subset of TRAIN (E: 144 / 177 / 200 / 162 / 64); the augmented hard pool (D1x) supplies 1,696–1,365 Class-A rows/seed. Matched candidate-correct controls are constructed by deterministic nearest-neighbour matching on frozen-only metadata (as in R/S).

## 11. Frozen Baseline Bridge

Reproduced exactly from the frozen `W,b` stored in the artifact: `C_t` D0 linear **0.244**, D0/MLP-32 **0.238**, D1x/MLP-32 **0.463** (E) — matching R's and S's published 5-seed means (0.244 / 0.238 / 0.463). The frozen candidate head is 0% on Class-A by construction. `cand_input = [h_reset, e_t]` is reproduced bit-exactly. **T's `C_t` row is a correct baseline bridge to S and R**; the new probes do not redefine the baseline.

## 12. Encoder Internal Recoverability

Hard-test accuracy, mean over 5 E seeds (D1x/MLP-32):

| Encoder stage | Dim | D0/lin | D1x/lin | D1x/MLP-32 | D1x held-out |
|---|---:|---:|---:|---:|---:|
| raw_obs | 21 | 0.087 | 0.095 | 0.124 | 0.118 |
| enc_h1_pre | 64 | 0.087 | 0.095 | **0.141** | 0.107 |
| enc_h1_post | 64 | 0.101 | 0.102 | 0.131 | 0.104 |
| enc_out_pre | 256 | 0.098 | 0.104 | 0.131 | 0.108 |
| e_t | 256 | 0.095 | 0.105 | 0.137 | 0.110 |

The encoder path is **flat-to-weak across depth** (0.087 → 0.137 for the D1x scalar; best 0.141 at `enc_h1_pre`), well below the 0.50 bar, and its held-out accuracy sits at chance (0.10–0.12). **Increasing probe capacity does not help**: `e_t` D1x goes 0.105 (linear) → 0.137 (MLP-32) → 0.121 (MLP-128). The encoder's apparent weakness is therefore not a capacity artifact and is not a learned "loss" along depth — it is the ceiling of what the current observation can determine.

## 13. Encoder Output e_t

`e_t` reaches D1x/MLP-32 **0.137** (D0/linear 0.095; D1x/linear 0.105) with held-out 0.110 — near chance. The **explanatory targets** localise *why*: at `e_t` the **destination register is fully decodable (1.000)** while **`operand_a` is 0.110 and `operand_b` is 0.136** (chance 0.10). The ADD observation `x_t` contains the op-type, the two register indices and an immediate (zero at ADD steps) — **not** the operand *values*. `e_t = f(x_t)` therefore cannot, by construction, carry the information the target requires; the target information must come from the state pathway. This is the cleanest structural fact of T: **the encoder output is a valid task-encoding of the current operation and destination but structurally devoid of the operands.**

## 14. S_prev / State-Path Recoverability

Hard-test accuracy, mean over 5 E seeds (D1x/MLP-32):

| State-path stage | Dim | D0/lin | D1x/lin | D1x/MLP-32 | D1x held-out |
|---|---:|---:|---:|---:|---:|
| S_prev | 212 | 0.074 | 0.136 | 0.167 | 0.113 |
| gate_pre | 212 | 0.072 | 0.149 | 0.191 | 0.078 |
| f_gate | 212 | 0.082 | 0.155 | 0.213 | 0.065 |
| h_reset | 212 | 0.090 | 0.189 | **0.239** | 0.046 |

The state path rises monotonically in-distribution (`S_prev → h_reset`: 0.167 → 0.239; the state-path best is `h_reset` = 0.239), but every stage **collapses on held-out operands** (0.113 → 0.046). The state-path explanatory targets confirm it carries the operands: `operand_a` rises `S_prev` 0.364 → `h_reset` 0.785, `operand_b` 0.364 → 0.553, while `destination` is mostly injected via the gate (`gate_pre`/`f_gate` = 1.000). The in-distribution advantage over the encoder (state 0.239 vs encoder 0.141, **+0.098**) is **sub-material** and does **not** generalise to held-out operands (state held-out 0.046–0.113 vs encoder held-out 0.104–0.118).

## 15. Gate Pre/Post Analysis

| Regime | gate_pre | f_gate (post) | Δ (post − pre) |
|---|---:|---:|---:|
| D0/MLP-32 | 0.078 | 0.102 | +0.024 |
| D1x/MLP-32 | 0.191 | 0.213 | +0.022 |
| D1x held-out | 0.078 | 0.065 | −0.013 |

`sigmoid(gate_pre) = f_gate` is reproduced bit-exactly. The sigmoid does **not** materially change information accessibility: in-distribution the postactivation is *slightly more* decodable (+0.022/+0.024), and on held-out operands the difference is tiny and negative (−0.013). No nonlinear boundary is identified as a site of recoverability loss — consistent with S's GELU/tanh finding. (`gate_pre` itself carries `operand_a` at 0.533 and `destination` at 1.000, inheriting the state and encoder components respectively.)

## 16. h_reset

`h_reset = f_gate ⊙ S_prev` is the strongest **upstream state** stage (D1x/MLP-32 **0.239**; D0/MLP-32 0.138) and carries `operand_a` at 0.785, `operand_b` at 0.553, `destination` at 0.927. It is the second-strongest stage overall, behind only `C_t`. Yet it **collapses to 0.046 on held-out operand combinations** (seen 0.261 → held-out 0.046; gap 0.215) — the same memorization signature R found for `C_t`. `h_reset` improves in-distribution recoverability relative to `S_prev` (+0.072), but the gain does not generalise.

## 17. Source Decomposition

Primary source comparison (E, D1x/MLP-32):

| Source | Dim | hard-test | held-out |
|---|---:|---:|---:|
| e_t (encoder portion) | 256 | 0.137 | 0.110 |
| S_prev | 212 | 0.167 | 0.113 |
| f_gate | 212 | 0.213 | 0.065 |
| h_reset (state portion) | 212 | 0.239 | 0.046 |
| cand_input = [h_reset, e_t] | 468 | 0.246 | 0.045 |
| C_t | 212 | **0.463** | 0.176 |

`cand_input` (0.246) is essentially `h_reset` (0.239) plus a non-informative `e_t` (0.137): the joint representation does **not** preserve more than `h_reset` alone (D1x/MLP-32 gain over `h_reset` +0.007; the `e_t` component contributes destination but not operands). `C_t` is much stronger than both of its inputs (0.463), consistent with S's finding that the candidate MLP creates most of the (memorised) separability. This is a **recoverability measurement on the joint representation only** — it is **not** evidence that the fusion operation is causally defective, and no fusion model is trained. The pattern is **neither** "encoder strong / state weak" **nor** "both strong / fused weak": the encoder is structurally operand-blind, the state path is weakly informative in-distribution and non-generalising, and the fused input is no better than its state component on held-out operands.

## 18. Capacity × Distribution

D1x capacity sweep (E, hard-test accuracy):

| Stage | linear | MLP-16 | MLP-32 | MLP-64 | MLP-128 |
|---|---:|---:|---:|---:|---:|
| e_t | 0.105 | 0.124 | 0.137 | 0.130 | 0.121 |
| enc_h1_pre | 0.095 | – | 0.141 | – | – |
| S_prev | 0.136 | 0.155 | 0.167 | 0.186 | 0.210 |
| gate_pre | 0.149 | 0.168 | 0.191 | 0.209 | 0.238 |
| f_gate | 0.155 | 0.179 | 0.213 | 0.220 | 0.253 |
| h_reset | 0.189 | 0.218 | 0.239 | 0.296 | 0.332 |
| cand_input | 0.190 | 0.207 | 0.246 | 0.293 | 0.322 |
| C_t | 0.396 | 0.453 | 0.463 | 0.473 | 0.480 |

The encoder path is **flat across capacity** (e_t 0.105→0.121); the state path rises weakly with capacity but **no stage reaches 0.50**, and the larger-capacity state gains are the same memorization (they collapse on held-out operands). Distribution regimes (E, MLP-32): D0 0.093 (e_t) / 0.138 (h_reset) / 0.238 (C_t); D0m 0.110 / 0.100 / 0.194; D1 0.108 / 0.161 / 0.261; D1x 0.137 / 0.239 / 0.463. The distribution effect appears **globally**, not at one isolated stage, and never lifts any upstream stage into the recoverable regime.

## 19. Held-Out Operand Generalization

Probe fit excluding 20 operand pairs, evaluated on Class-A hard-test rows with held-out vs seen pairs (D1x/MLP-32, E):

| Stage | held-out | seen | gap |
|---|---:|---:|---:|
| raw_obs | 0.118 | 0.127 | 0.009 |
| enc_h1_pre | 0.107 | 0.147 | 0.040 |
| e_t | 0.110 | 0.147 | 0.037 |
| S_prev | 0.113 | 0.164 | 0.051 |
| gate_pre | 0.078 | 0.208 | 0.130 |
| f_gate | 0.065 | 0.245 | 0.180 |
| h_reset | 0.046 | 0.261 | 0.215 |
| cand_input | 0.045 | 0.286 | 0.241 |
| C_t | 0.176 | 0.493 | 0.317 |

**Every stage fails held-out generalisation** (all held-out ≤ 0.25; state-path gaps 0.05–0.24; `C_t` gap 0.317 reproduces R exactly). The encoder stages have small gaps only because they are weak everywhere (no operand information to memorise); the state-path stages show the memorization signature — high seen, collapsed held-out. No stage is "recoverable" on the strength of seen-combination performance.

## 20. Raw-Sum vs Modular

D1x/MLP-32 accuracy for the modular target `(a+b) mod 10` vs the raw sum `a+b` (19-way), E:

| Stage | modular | raw sum |
|---|---:|---:|
| e_t | 0.137 | 0.067 |
| S_prev | 0.167 | 0.162 |
| gate_pre | 0.191 | 0.163 |
| f_gate | 0.213 | 0.215 |
| h_reset | 0.239 | 0.266 |
| cand_input | 0.246 | 0.258 |
| C_t | 0.463 | 0.440 |

There is **no upstream stage at which the raw sum is clearly easier** than the modular target (the encoder stages show the raw-sum probe *below* its 1/19 chance because the encoder has no arithmetic information at all). As in Q/R/S, there is no "arithmetic present but modular-mis-organised" signature anywhere upstream.

## 21. Wrap / No-Wrap / Repeated

D0/MLP-32 (E), hard-test accuracy:

| Stage | wrap | no-wrap | repeated | isolated |
|---|---:|---:|---:|---:|
| raw_obs | 0.088 | 0.104 | 0.092 | 0.102 |
| e_t | 0.071 | 0.110 | 0.094 | 0.092 |
| S_prev | 0.048 | 0.116 | 0.077 | 0.090 |
| f_gate | 0.046 | 0.146 | 0.117 | 0.089 |
| h_reset | 0.105 | 0.165 | 0.160 | 0.121 |
| cand_input | 0.080 | 0.166 | 0.142 | 0.117 |
| C_t | 0.228 | 0.245 | 0.182 | 0.280 |

The wrap/no-wrap subtype spread is below the 0.15 mixed bar at every KEY stage (max 0.100 at `f_gate`; `C_t` spread 0.016); repeated vs isolated differ only weakly (`C_t` repeated 0.182 vs isolated 0.280). **No subtype reaches the recoverability bar**, and no subtype shows a wrap-specific or repeated-specific collapse at an upstream stage. E and F behave the same (F D1x `/` wrap 0.530 / no-wrap 0.374 as in R; the upstream stages are uniformly weak).

## 22. Candidate-Input Bridge

Explicit comparison (E, D1x/MLP-32 hard-test / D1x held-out): `e_t` 0.137 / 0.110, `h_reset` 0.239 / 0.046, `cand_input = [h_reset, e_t]` 0.246 / 0.045, `C_t` 0.463 / 0.176.

The four pre-registered patterns:
* *Encoder strong / state weak* — **no** (encoder structurally operand-blind).
* *State strong / encoder weak* — **in-distribution only, weakly** (state 0.239 vs encoder 0.141 = +0.098, sub-material; held-out reverses).
* *Both weak* — **closest description**: encoder weak everywhere; state path weak and non-generalising.
* *Both strong / fused weak* — **no** (`cand_input` is not materially weaker than `h_reset`).

The fused candidate input is **no more recoverable than its state component** and is not a fusion-localised collapse. `C_t`'s higher recovery (0.463) is downstream of the candidate MLP, already shown by S not to be an output-projection limit.

## 23. Information Trajectory

D1x/MLP-32 along the full upstream trajectory (E): raw_obs 0.124 → enc_h1_pre 0.141 → enc_h1_post 0.131 → enc_out_pre 0.131 → **e_t 0.137** → S_prev 0.167 → gate_pre 0.191 → f_gate 0.213 → h_reset 0.239 → cand_input 0.246 → **C_t 0.463**. Adjacent changes: +0.017, −0.009, −0.000, +0.006, +0.030, +0.024, +0.023, +0.026, +0.007, +0.217.

The trajectory is **weakly rising, not falling**: there is **no material drop** anywhere between the encoder output and `C_t` (the largest adjacent *increase* is `cand_input → C_t` +0.217; the largest *decrease* is −0.009). The only "early loss" is structural: the encoder output never contained the operands to begin with. **No localised drop in probe recoverability is identified**; the whole upstream path is weak, and the state-path rise is memorization.

## 24. E vs F

Paired (5 seeds) D1x/MLP-32 E−F differences: raw_obs +0.002 (p=0.89), enc_h1_pre +0.012 (p=0.25), enc_h1_post −0.000 (p=1.00), enc_out_pre 0.000 (p=0.99), e_t +0.004 (p=0.81), S_prev +0.001 (p=0.90), gate_pre +0.004 (p=0.73), f_gate −0.003 (p=0.84), h_reset +0.035 (p=0.10), cand_input +0.031 (p=0.08), C_t +0.035 (p=0.19). **No stage differs significantly.** F reproduces the same weak encoder path (best 0.131), the same weakly-rising state path (0.166 → 0.204), the same `C_t` bridge (D0 linear 0.230, D0/MLP-32 0.259, D1x/MLP-32 0.428, held-out 0.166), the same held-out collapse, and the same **T-D** category. F's destination/operand explanatory pattern matches E (`e_t` dest 1.000 / oa 0.126 / ob 0.105). E and F agreement is explicit and `E_F_disagreement = false`.

## 25. Statistical / Multiple-Comparison Discipline

The five frozen E checkpoints are the independent units; F is a control, never pooled. Every fraction is reported per seed with mean ± SD; all stage/source comparisons are paired by seed. The activation set, capacity design and decision rule were **predeclared**; every stage and capacity is reported, including negative results. The stage family is not searched for significance — the primary conclusion rests on the predeclared source-level summary (encoder-path best vs state-path best, and held-out performance). Individual paired comparisons are reported descriptively. Transition counts are descriptive and never used as inferential *n*.

## 26. Leakage / Validity

Enforced and tested: no sequence overlap between TRAIN/VAL/TEST; the augmented stream is sequence-disjoint (hash-verified); hard-train/hard-val/hard-test are disjoint by construction (trainhard/valhard are Class-A subsets of the disjoint train/val splits; hard-test is the canonical untouched Class-A tail and is never used to fit or tune); λ / capacity / best-state selection use only the regime's val; permuting the TEST labels leaves the fitted λ, train accuracy and val accuracy unchanged; held-out operand pairs are excluded from fitting and shown to have been present in the pool; probes never enter model inference (forward is bit-identical before/after probe fitting); hooks are removed after capture and the capture is bit-exact against manual recomputation; `W,b` are stored and reproduced from the checkpoint; a deliberate leakage failure path (fit on test) is included and shown to inflate test accuracy; the disjoint-split identity-overlap guard is tested; the collector is bit-identical across two runs; the analysis JSON is identical (excluding the timestamp); checkpoint SHA-256 hashes are unchanged.

## 27. Causal Status

**Observation only.** T measures recoverability under analysis-only probes across a bounded capacity/distribution grid and a fixed activation map. It does **not** show that the encoder, the state pathway, the gate, or the fusion "causes" the Class-A failure; it does **not** show the model "cannot do arithmetic"; it does **not** show a hidden-state replacement or feature edit would help. The wording used is "target/operand information was / was not recoverable from this representation under the tested probe family", "the state path was descriptively stronger in-distribution but the difference was sub-material and did not survive held-out operands", "the encoder output is structurally operand-blind", "no upstream representation is robustly (held-out) recoverable". **No causal mechanism has been established and no intervention is performed or licensed.**

## 28. Decision Classification

**T-D — weak throughout** (E primary; F agrees).
* Encoder-path best `enc_h1_pre` = **0.141** ≤ 0.25 (weak) — but the state-path advantage (state best `h_reset` 0.239 − 0.141 = **+0.098**) is **below** the 0.10 material bar, and the held-out difference is **negative** (state 0.113 vs encoder 0.118, Δ −0.005), so **T-A does not trigger** (its held-out requirement fails and the margin is sub-material). This is a *documented borderline*: descriptive direction favours the state path, but it is neither material nor generalising.
* Encoder-path best (0.141) < state-path best (0.239) and encoder best < 0.50 → **not T-B**.
* `e_t` 0.137 < 0.50 and `h_reset` 0.239 < 0.50 → **not T-C**.
* Nothing reaches the 0.50 bar (max stage `C_t` 0.463); held-out ≤ 0.25 at every stage (max `C_t` 0.176); both pathways weak; subtype spread < 0.15 at every KEY stage; E/F categories agree → **T-D**.
* Interpretation: **the limitation is already present across the tested upstream representations.** The observation encoding is structurally operand-blind; the state pathway carries operand information weakly in-distribution but not in a generalisable form; the candidate input is no better than its state component. This is a localisation, not a mechanism, and it licenses no change.

## 29. What T Establishes

* A deterministic, leakage-controlled **upstream recoverability map** over the real frozen pipeline (11 stages; observation-encoder internals + output, `S_prev`, gate pre/postactivation, `h_reset`, `cand_input`, `C_t`), reproducing R/S's `C_t` baseline exactly (0.244 / 0.238 / 0.463).
* That the **observation encoder output `e_t` is structurally operand-blind**: destination decodes at 1.000 while `operand_a`/`operand_b` are at chance (0.110 / 0.136) — the ADD observation contains no operand values, so the target information must come from the state pathway.
* That the **state pathway is only descriptively stronger** (best `h_reset` 0.239 vs encoder best 0.141; +0.098, sub-material) and **does not generalise**: every state-path stage collapses on held-out operand combinations (0.113 → 0.046).
* That the **sigmoid gate boundary does not materially change accessibility** (`gate_pre → f_gate` +0.022 in-distribution, −0.013 held-out).
* That **no upstream stage reaches the 0.50 recoverability bar** (max `C_t` 0.463) or 0.25 held-out (max `C_t` 0.176); raw-sum is not easier than modular; wrap ≈ no-wrap; and **E and F agree** — classification **T-D**.

## 30. What T Does Not Establish

* It does **not** establish causality; there is no intervention and no hidden-state replacement.
* It does **not** show the encoder "destroys" information, the state "should" carry more, the gate is a bottleneck (it is not), or the concatenation is broken.
* It does **not** show the model cannot do modular arithmetic, that any representation is definitively absent, or that adding dimensions / changing the encoder / editing features would help.
* It does **not** test representations outside the predeclared bounded probe family, nor task-level generalisation of a *different* model.
* It does **not** prove that the state-path operand information is only memorization in an absolute sense; it measures that it does not survive the pre-registered held-out-operand control under the tested probe family.

## 31. Final Answers

1. **Is the target recoverable from the observation encoder?** **No** — encoder path ≤ 0.141 (chance 0.10); the encoder is structurally operand-blind.
2. **Which encoder stage is most recoverable?** `enc_h1_pre` (0.141), marginally above `e_t` (0.137); all near chance.
3. **Is `e_t` stronger than `S_prev`?** **No** — `e_t` 0.137 < `S_prev` 0.167 (D1x/MLP-32); on held-out they are comparable (0.110 vs 0.113).
4. **Is the target more recoverable from `gate_pre` than `f_gate`?** **No** — `f_gate` (0.213) ≥ `gate_pre` (0.191) in-distribution; the sigmoid does not remove information.
5. **Does `h_reset` retain useful target information?** **Weakly, in-distribution only** (0.239; second-strongest stage) — it **collapses to 0.046 on held-out operands**.
6. **Which source contributes more transferable information, `e_t` or `h_reset`?** On the primary (held-out) endpoint, **neither is transferable** (`e_t` 0.110, `h_reset` 0.046) — the difference is not meaningful; in-distribution `h_reset` is descriptively stronger (0.239 vs 0.137) but sub-material relative to the encoder best and non-generalising.
7. **Does `[e_t, h_reset]` preserve more than either source alone?** **No** — `cand_input` 0.246 ≈ `h_reset` 0.239 (+0.007); held-out 0.045.
8. **Where is the largest observed decrease in recoverability?** **There is none** along the tested upstream path; the trajectory weakly *rises* (largest adjacent increase +0.217 at `cand_input→C_t`; largest decrease −0.009).
9. **Does the decrease survive held-out operand testing?** There is no in-distribution decrease; the state-path *increases* do **not** survive held-out operands (they collapse).
10. **Does raw-sum vs modular reveal a specific arithmetic encoding issue upstream?** **No** — no stage shows raw-sum easier than modular.
11. **Does wrap differ from no-wrap?** **No** — spread < 0.15 at every KEY stage (max 0.100).
12. **Does repeated ADD differ from isolated ADD?** Only weakly (`C_t` repeated 0.182 vs isolated 0.280); no upstream-specific collapse.
13. **Do E and F agree?** **Yes** — all paired p ≥ 0.08; both T-D.
14. **Is the result best classified encoder-limited, state-limited, source-fusion-limited, weak throughout, or mixed?** **Weak throughout (T-D)** — with a documented borderline: the state path is descriptively stronger in-distribution (+0.098, just under the 0.10 bar) but does not generalise.
15. **Is any upstream representation robustly recoverable under held-out operands?** **No.**
16. **Does T justify an intervention?** **No.** Weak held-out recoverability everywhere ⇒ under the conservative rule, **no intervention**; the next step is measurement, not a candidate/state modification.

## 32. Exactly ONE Next Experiment

T-D localises the earliest observed limitation and rules out a candidate-side or state-side intervention. The single unresolved discrepancy is the **borderline state-vs-encoder difference** and, more fundamentally, the fact that the only source that *could* carry the operands (the state/history pathway) holds them **only non-generalisingly**. Per T-D, the next study moves **farther upstream — to the raw observation / task encoding — rather than changing the candidate/state system**.

**Experiment U — Raw-observation / task-encoding localisation (measurement-only, frozen checkpoints).** Test the *task interface itself*: (i) probe the raw 21-dim observation interface and the raw observation *history window* (the task input stream, state-free) with R/S's exact probe family and held-out-operand control on the same Class-A hard tail, to establish whether the operand/target information is ever made available at the task interface at all, and (ii) quantify the earliest trajectory point at which the operand values enter a probe-recoverable representation (the state), separating "the task interface structurally omits the operands" from "the operands are present in the interface but compressed away". U is measurement-only, touches no weights, does not modify the candidate/state/gate/commit/readout, and does not begin Phase 2. It conditions whether **any** upstream route to the Class-A information exists before the recurrent state — the last upstream measurement before the loop would have to accept the state-encoding limitation as the current frontier.

The governing loop remains **MEASURE → LOCALIZE → INTERPRET → KEEP/KILL → ONE CONTROLLED NEXT TEST**.

## Required Figures

`docs/reports/figures/experiment_t_*.png` (16): `upstream_recoverability_by_stage`, `encoder_capacity_curves`, `state_capacity_curves`, `source_decomposition`, `ordinary_vs_hard_by_stage`, `heldout_operand_by_stage`, `seen_vs_heldout_gap`, `gate_pre_vs_gate_post`, `h_reset_vs_S_prev`, `rawsum_vs_modular_by_stage`, `wrap_nowrap`, `repeated_isolated`, `candidate_correct_vs_classA`, `information_trajectory`, `per_seed_source`, `localization_summary`.

## Tests

`tests/test_experiment_t.py` — **28 tests**: frozen-checkpoint SHA/immutability; exact upstream capture (bit-exact vs manual recomputation of every stage — encoder internals, `e_t`, `gate_pre`, `f_gate`, `h_reset`, `cand_input`, `C_t`); `cand_input == [h_reset, e_t]` identity; `sigmoid(gate_pre) == f_gate`; deterministic capture; no-grad; no parameter mutation; activation dimensions matched to the actual model; `C_t → readout` reproduction; `W,b` match the checkpoint; npz stage completeness; sequence-split and augmented-stream disjointness; hard-split candidate-wrong construction; hard-test = untouched canonical Class-A; wrap/repeated classification; raw-sum target generation; probe determinism; probe train/val/test separation (λ from val only, no test tuning); deliberate leakage failure path; operand-pair leakage exclusion; no probe insertion into inference; every declared stage analyzed; no test-based stage/capacity selection (pure `probes_for`); decision validity; `C_t` bridge vs R/S; candidate-correct control separation; E/F alignment; JSON determinism; malformed-artifact rejection; figure creation (≥16); historical-report immutability.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_experiment_t.py \
    --checkpoints-dir experiments/checkpoints \
    --output experiments/experiment_t_upstream_representation.npz
PYTHONPATH=src python experiments/analyze_experiment_t.py \
    --npz experiments/experiment_t_upstream_representation.npz \
    --output experiments/experiment_t_results.json \
    --fig-dir docs/reports/figures --fig-prefix experiment_t
PYTHONPATH=src pytest tests/test_experiment_t.py -v      # 28 tests
PYTHONPATH=src pytest tests/                             # full suite
```

Determinism: collector arrays bit-identical across two runs; analysis JSON identical (excluding the timestamp); all 16 figures byte-identical. Checkpoint SHA-256 verified before/after. No model, encoder, readout, gate, commit, or candidate weight is modified. The analyzer reuses `analyze_experiment_s`'s probe primitives verbatim and fixes the torch thread count for bit-reproducibility.

## Scientific Guardrails

Measurement-only; observational. The report says "target/operand information was not recoverable from this representation under the tested probe family", "the encoder output is structurally operand-blind", "the state path was descriptively stronger in-distribution but the difference was sub-material and did not survive held-out operands", "no upstream representation is robustly recoverable", and "the result localises the earliest observed weakness". It never says the encoder/state/gate/fusion *causes* the failure, that the gate "destroys" information, that the concatenation is broken, that the representation is absent, that the model cannot arithmetic, that the candidate should receive more dimensions, or that the next repair is obvious. Probe decodability ≠ exact state preservation; a failed bounded probe ≠ proof that no representation exists; apparent capacity/distribution gains that fail held-out operands are labelled memorization, not recoverability; the seed is the independent unit. Q, R and S are **not** edited; T is appended as new evidence. No Phase 2. Negative and borderline findings (sub-material state advantage, no localised drop, held-out collapse, constant operation) are preserved.
