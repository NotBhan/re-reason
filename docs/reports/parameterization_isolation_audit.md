# Parameterization-Isolation Audit Report: Dissecting the Architectural Drivers of Recurrent State Tracking

**Date:** 2026-10-04  
**Status:** Completed  
**Objective:** Resolve the primary mechanistic hypotheses regarding Model D's performance on the Phase 1 Register Tracking benchmark by isolating non-linear feature mixing, shared vs. independent trunks, MGU-style pre-gating, and audited FLOP efficiency across 7 parameter-matched variants (35 total runs across seeds 42–46).

---

## Executive Summary & Core Verdicts

1. **FLOP Audit Verdict: Measurement Bug Discovered & Corrected.**
   - In previous reports, GRU was incorrectly reported at **49,800 FLOPs/step** while Model D was reported at **451,240 FLOPs/step**.
   - **Root Cause:** A measurement bug in the module traversal loop `for m in model.modules(): if isinstance(m, nn.Linear): ...`. In PyTorch, native `nn.GRUCell` encapsulates its weight matrices (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`) directly as `nn.Parameter` tensors without child `nn.Linear` modules. The loop completely skipped the GRU recurrent cell and only counted the linear layers of the observation encoder and decision head.
   - **Corrected Accounting:** When accounting for all matrix multiplications ($6 H I + 6 H^2 + 6 H$) and pointwise gate activations ($17 H$), Canonical GRU requires **460,846 FLOPs/step** — matching or slightly exceeding Model D (**452,264 FLOPs/step**). FLOP parity is verified across all models.

2. **Causal Driver 1: Non-Linear State-Observation Mixing is Decisive (+35.67% Gain).**
   - Comparing **Model D** (47.27% ± 3.33% exact match) against **Shared Linear Trunk** (11.60% ± 1.98% exact match) reveals an enormous drop of **35.67% ± 4.19%** ($t = 19.05, p = 4.47 \times 10^{-5}$, Cohen's $d = 8.52$).
   - Both models share the identical 192-dimensional bottleneck structure and parameter count (226,152). Removing solely the GELU activation causes performance to collapse to baseline levels. Low-rank linear bottleneck compression provides zero advantage; **non-linear feature mixing between state and observation is the fundamental enabler of tracking performance**.

3. **Causal Driver 2: Trunk Sharing Does NOT Matter (+0.02% Difference).**
   - Comparing **Model D** (shared trunk, 47.27% ± 3.33%) against **Independent MLPs** (separate candidate MLP and gate MLP, 47.25% ± 4.19%) yields a paired delta of **+0.02% ± 6.46%** ($t = 0.01, p = 0.996$).
   - Coupling gate and candidate representations through a shared trunk does not improve tracking accuracy over separate MLPs. However, the shared trunk is **~15% faster in training wall-clock time** (52.6s vs. 62.2s) and requires fewer matrix dispatch calls.

4. **Causal Driver 3: MGU Pre-Gating is a Powerful Bilinear Feature Filter (+26.23% Gain).**
   - Comparing **Canonical UGRNN** (9.85% ± 0.30%) against **Canonical MGU** (36.08% ± 2.18%) shows a massive **+26.23%** advantage for MGU without any non-linear MLP.
   - MGU's elementwise multiplication $f_t \odot S_{t-1}$ acts as a bilinear interaction (cross-term) that resets unselected state coordinates prior to candidate projection, providing substantial feature selectivity even in purely affine layers.

5. **Discovery: Nonlinear Candidate MGU Surpasses Model D (+7.02% Gain, 54.28% Exact Match).**
   - When MGU's pre-gated state is processed by a non-linear candidate MLP (**Model 6: Nonlinear Candidate MGU**), performance reaches **54.28% ± 2.81% test exact match** and **85.05% per-register accuracy** (best seed: **57.92%** test exact match).
   - This decisively outperforms Model D by **+7.02% ± 1.38%** ($t = -11.37, p = 3.41 \times 10^{-4}$, Cohen's $d = -5.08$) across all 5 seeds.
   - Pre-gating and non-linear feature synthesis are orthogonal and complementary mechanisms.

---

## 1. Audited FLOP Accounting & Computation Definition

### 1.1 Root-Cause Analysis of Previous Telemetry Bug
In the previous audit report, the following FLOP calculation was used:
```python
for m in model.modules():
  if isinstance(m, nn.Linear):
    flops += 2 * m.in_features * m.out_features + m.out_features
```
Because PyTorch's native `nn.GRUCell` stores `weight_ih` ($768 \times 256$) and `weight_hh` ($768 \times 256$) directly as internal `nn.Parameter` attributes without child `nn.Linear` instances, `isinstance(m, nn.Linear)` evaluated to `False` for the GRU cell. Consequently, the function counted only the 2 linear layers in `ObservationEncoder` and 1 linear layer in `TypedDecisionHead`, yielding 49,800 FLOPs.

### 1.2 Rigorous FLOP Definition
To guarantee fairness, the audited FLOP metric is defined as:
1. **Linear Layer ($\text{in} \to \text{out}$):**  
   $$\text{FLOPs} = 2 \times \text{in} \times \text{out} + (\text{out if bias else } 0)$$
   (1 multiply and 1 accumulate per weight, plus 1 addition per output bias).
2. **LayerNorm (dimension $D$):**  
   $$\text{FLOPs} = 8 \times D$$
   (Mean: $D$; Variance: $3D$; Normalize: $2D$; Affine scale/shift $\gamma x + \beta$: $2D$).
3. **`nn.GRUCell(input_size=I, hidden_size=H)`:**  
   $$\text{FLOPs} = 6 H I + 6 H^2 + 23 H$$
   - 3 input linear projections: $2 \times 3H \times I + 3H = 6 H I + 3H$.
   - 3 hidden linear projections: $2 \times 3H \times H + 3H = 6 H^2 + 3H$.
   - Pre-reset multiplication $r_t \odot h_{t-1}$: $H$ FLOPs.
   - Pointwise activations ($2 \times \sigma, 1 \times \tanh$): $12 H$ FLOPs.
   - Convex update $(1 - z_t) \odot h_{t-1} + z_t \odot n_t$: $4 H$ FLOPs.
   - For $I=256, H=162$: $411,046$ cell FLOPs + $49,800$ enc/head FLOPs = **460,846 FLOPs/step**.

### 1.3 Audited FLOPs Across Parameter-Matched Models

| Model Architecture | Total Parameters | Core Parameters | Audited FLOPs / Step | Training Wall-Clock (3k steps) | Latency (batch=1) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Canonical UGRNN** | 226,360 | 199,280 | 452,536 | 47.1s | 0.085 ms |
| **2. Independent MLPs** | 226,216 | 197,376 | 452,328 | 62.2s | 0.104 ms |
| **3. Shared Deep MLP Trunk** | 224,512 | 195,672 | 448,848 | 57.8s | 0.103 ms |
| **4. Shared Linear Trunk** | 226,152 | 197,312 | 452,264 | 49.2s | 0.080 ms |
| **5. Canonical MGU** | 226,360 | 199,280 | 452,536 | 62.0s | 0.102 ms |
| **6. Nonlinear Candidate MGU** | 226,358 | 199,278 | 452,598 | 64.7s | 0.112 ms |
| **7. Current Model D** | 226,152 | 197,312 | 452,264 | 52.6s | 0.089 ms |

*Conclusion on Compute Fairness:* All 7 models have identical theoretical computational requirements (~449K–453K FLOPs per step, within $\pm 0.8\%$). Differences in empirical latency (0.080 ms to 0.112 ms) stem entirely from kernel fusion and sequential dependency chains (such as MGU's pre-gating dependency $f_t \to h_\text{reset} \to \text{candidate}$).

---

## 2. Parameterization-Isolation Experimental Protocol

- **Task:** Official Phase 1 Register Tracking Benchmark ($T=10$, 4 registers, values in $[-10, 10]$).
- **Splits:** Strict procedural split generation (6,000 train sequences, 1,200 validation, 1,200 test sequences) with sequence hash verification to guarantee zero overlap.
- **Seeds:** 5 identical seeds across all models: `[42, 43, 44, 45, 46]`.
- **Training Budget:** 3,000 steps, batch size 64, AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-2}$, grad clipping 1.0).
- **Supervision:** Terminal-only loss at step $t=10$.
- **Selection:** Best checkpoint selected strictly by validation exact match; held-out test evaluation performed only on selected checkpoints.
- **Hardware:** AMD Radeon RX 6600M GPU (ROCm 6.4).

### 2.1 The 7 Isolated Architectural Variants

| Model | Variant Name | Candidate $\tilde{S}_t$ | Gate $z_t$ | Pre-Gated State? | Intermediate Non-Linearity | Trunk Architecture | Parameters |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|
| **1** | **Canonical UGRNN** | $\tanh(W_c e + U_c S)$ | $\sigma(W_g e + U_g S)$ | No | None (Pure Affine) | None (Independent Affine) | 226,360 |
| **2** | **Independent MLPs** | $\tanh(\text{MLP}_c([S, e]))$ | $\sigma(\text{MLP}_g([S, e]))$ | No | GELU in both MLPs | None (Independent $H=128$) | 226,216 |
| **3** | **Shared Deep Trunk** | $\tanh(W_c h_t)$ | $\sigma(W_g h_t)$ | No | GELU in 2 layers | 2-layer MLP ($H=164$) | 224,512 |
| **4** | **Shared Linear Trunk** | $\tanh(W_c h_t)$ | $\sigma(W_g h_t)$ | No | None (Linear bottleneck) | 1-layer Linear ($H=192$, No GELU) | 226,152 |
| **5** | **Canonical MGU** | $\tanh(W_c e + U_c (f \odot S))$ | $f = \sigma(W_f e + U_f S)$ | **Yes** ($f \odot S$) | None (Pure Affine) | None (Independent Affine) | 226,360 |
| **6** | **Nonlinear MGU** | $\tanh(\text{MLP}_c([f \odot S, e]))$ | $f = \sigma(W_f e + U_f S)$ | **Yes** ($f \odot S$) | GELU in candidate | Affine gate + 2-layer cand MLP | 226,358 |
| **7** | **Current Model D** | $\tanh(W_c h_t)$ | $\sigma(W_g h_t)$ | No | GELU in 1 layer | 1-layer MLP ($H=192$, GELU) | 226,152 |

---

## 3. Empirical Results across 5 Seeds (35 Runs)

### 3.1 Primary Benchmark Summary

| Model Variant | Test Exact Match (%) | Test Per-Reg Acc (%) | Val Exact Match (%) | Gradient Attenuation ($\frac{\|g_1\|}{\|g_{10}\|}$) | Training Time (s) | Inference Latency (ms) | Peak VRAM (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Canonical UGRNN** | 9.85% ± 0.30% | 55.53% ± 0.36% | 10.22% ± 0.58% | 0.443 ± 0.080 | 47.1s | 0.085 ms | 299.1 MB |
| **2. Independent MLPs** | 47.25% ± 4.19% | 82.17% ± 1.83% | 46.28% ± 4.16% | 0.673 ± 0.024 | 62.2s | 0.104 ms | 299.1 MB |
| **3. Shared Deep MLP Trunk** | 47.62% ± 3.93% | 82.45% ± 1.63% | 49.08% ± 3.65% | 0.719 ± 0.045 | 57.8s | 0.103 ms | 299.1 MB |
| **4. Shared Linear Trunk** | 11.60% ± 1.98% | 57.28% ± 2.05% | 11.86% ± 1.34% | 0.601 ± 0.023 | 49.2s | 0.080 ms | 299.1 MB |
| **5. Canonical MGU** | 36.08% ± 2.18% | 77.57% ± 1.15% | 35.14% ± 1.68% | 0.523 ± 0.060 | 62.0s | 0.102 ms | 299.1 MB |
| **6. Nonlinear Candidate MGU** | **54.28% ± 2.81%** | **85.05% ± 1.02%** | **52.92% ± 1.67%** | 0.583 ± 0.052 | 64.7s | 0.112 ms | 299.1 MB |
| **7. Current Model D** | 47.27% ± 3.33% | 82.17% ± 1.48% | 45.72% ± 3.09% | 0.665 ± 0.040 | 52.6s | 0.089 ms | 299.1 MB |

### 3.2 Paired Statistical Significance (Model D vs. Others Across 5 Seeds)

| Comparison Pair | Paired Mean Delta ($\Delta$) | Paired $\text{Std}(\Delta)$ | Paired $t$-stat | $p$-value | Cohen's $d$ | Decisive Effect? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model D vs. 1. Canonical UGRNN** | **+37.42%** | 3.43% | 24.37 | $1.68 \times 10^{-5}$ | 10.90 | Yes (Decisive) |
| **Model D vs. 2. Independent MLPs** | **+0.02%** | 6.46% | 0.01 | $0.9957$ | 0.00 | **No Difference** |
| **Model D vs. 3. Shared Deep Trunk** | **-0.35%** | 2.13% | -0.37 | $0.7314$ | -0.16 | **No Difference** |
| **Model D vs. 4. Shared Linear Trunk** | **+35.67%** | 4.19% | 19.05 | $4.47 \times 10^{-5}$ | 8.52 | Yes (Decisive) |
| **Model D vs. 5. Canonical MGU** | **+11.18%** | 4.43% | 5.65 | $0.0048$ | 2.53 | Yes (Significant) |
| **Model D vs. 6. Nonlinear MGU** | **-7.02%** | 1.38% | -11.37 | $3.41 \times 10^{-4}$ | -5.08 | **Model 6 Wins Decisively** |

*Statistical Note:* With $N=5$ matched seeds, paired t-tests yield $p < 0.001$ for the primary causal contrasts (D vs. Linear Trunk, D vs. Nonlinear MGU). Confidence intervals are compact (e.g. Model 6 outscored Model D by between +5.83% and +8.83% on every seed).

### 3.3 Seed-by-Seed Trajectories (Test Exact Match %)

| Seed | 1. UGRNN | 2. Indep MLPs | 3. Deep Trunk | 4. Linear Trunk | 5. MGU | 6. Nonlinear MGU | 7. Model D |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | 9.33% | 42.58% | 41.58% | 12.50% | 35.00% | **52.00%** | 42.92% |
| **43** | 9.75% | 54.33% | 45.75% | 10.42% | 36.75% | **50.17%** | 43.50% |
| **44** | 9.92% | 44.92% | 47.08% | 10.25% | 39.75% | **55.25%** | 49.42% |
| **45** | 10.17% | 49.50% | 51.00% | 15.08% | 33.17% | **56.08%** | 50.25% |
| **46** | 10.08% | 44.92% | 52.67% | 9.75% | 35.75% | **57.92%** | 50.25% |

---

## 4. Gate Telemetry & Mechanistic Diagnostics

| Model Architecture | Mean Gate ($\bar{z}$) | Shannon Entropy (bits) | Fraction $< 0.10$ | Fraction $> 0.90$ | Gate-Displacement Corr ($r$) | SET Mean Gate | SWAP Mean Gate | ADD Mean Gate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Canonical UGRNN** | 0.324 | 0.245 | 55.8% | 21.7% | 0.719 | 0.387 | 0.306 | 0.229 |
| **2. Independent MLPs** | 0.328 | 0.120 | 61.6% | 27.5% | 0.719 | 0.376 | 0.347 | 0.224 |
| **3. Shared Deep Trunk** | 0.276 | 0.117 | 67.0% | 22.5% | 0.709 | 0.344 | 0.269 | 0.160 |
| **4. Shared Linear Trunk** | 0.297 | 0.129 | 64.2% | 24.1% | 0.701 | 0.355 | 0.306 | 0.186 |
| **5. Canonical MGU** | 0.328 | 0.221 | 56.6% | 23.0% | 0.722 | 0.380 | 0.336 | 0.227 |
| **6. Nonlinear MGU** | 0.343 | 0.223 | 55.0% | 24.3% | 0.686 | 0.379 | 0.361 | 0.259 |
| **7. Current Model D** | 0.313 | 0.151 | 61.6% | 24.6% | 0.707 | 0.379 | 0.313 | 0.194 |

### Mechanistic Insights from Telemetry:
1. **Gate Polarization vs. Accuracy:** Models with non-linear MLP candidate/gate networks (Models 2, 3, 7) display lower gate entropy (~0.12–0.15 bits) and sharper bimodal saturation (>61% near 0, >24% near 1) than affine units (~0.22–0.25 bits).
2. **Predictive State Displacement ($r \approx 0.70$):** In all models, gate activation correlates strongly ($r = 0.69 \text{ to } 0.72$) with coordinate-wise state displacement $|S_t - S_{t-1}|$.
3. **Operational Hierarchy:** Every model preserves the operational hierarchy: $\text{SET} > \text{SWAP} > \text{ADD}$, confirming that the gate actively modulates state retention based on the arithmetic or assignment demand of the instruction.

---

## 5. Incremental Causal Analysis

To determine which architectural mechanism produces the largest marginal improvement, we rank the incremental step deltas:

| Incremental Step | Contrast | Performance Impact | Causal Mechanism |
|:---|:---|:---:|:---|
| **1. Non-linear Feature Mixing** | Model D vs. Shared Linear Trunk | **+35.67%** ($\Delta = 11.6\% \to 47.3\%$) | Non-linear cross-product of observation and memory allows register matching. |
| **2. State Pre-Gating (Affine)** | Canonical MGU vs. Canonical UGRNN | **+26.23%** ($\Delta = 9.8\% \to 36.1\%$) | Bilinear elementwise product $f_t \odot S_{t-1}$ isolates relevant register dimensions. |
| **3. Non-linear Candidate in MGU** | Nonlinear MGU vs. Canonical MGU | **+18.20%** ($\Delta = 36.1\% \to 54.3\%$) | Adding non-linear candidate MLP on top of pre-gated state resolves compound arithmetic. |
| **4. Pre-gating over Shared Trunk** | Nonlinear MGU vs. Model D | **+7.02%** ($\Delta = 47.3\% \to 54.3\%$) | Pre-gating state before MLP extraction outperforms un-gated MLP extraction. |
| **5. Deepening Trunk ($1 \to 2$ layers)** | Shared Deep Trunk vs. Model D | **+0.35%** ($\Delta = 47.3\% \to 47.6\%$) | Extra trunk layer adds parameter overhead without statistically meaningful gain. |
| **6. Trunk Sharing vs. Independent** | Model D vs. Independent MLPs | **+0.02%** ($\Delta = 47.25\% \to 47.27\%$) | Shared representations between gate and candidate do not alter tracking accuracy. |

---

## 6. Answers to the Final Research Questions

### Question 1: Does non-linear state-observation mixing explain D's advantage?
**YES.**
This is the single largest causal factor in the entire parameterization study. When the non-linear GELU activation is removed from Model D's 192-dimensional bottleneck (Model 4: Shared Linear Trunk), performance drops from **47.27% to 11.60%** (a 35.67% collapse, $p < 10^{-4}$). Pure affine combinations ($W e + U S$) cannot express the multiplicative interaction required to look up a register value and compute an arithmetic delta. The GELU non-linearity synthesizes the joint state-observation features needed to route information into the state.

### Question 2: Does sharing the trunk between gate and candidate matter?
**NO.**
Model D (shared trunk, 47.27% ± 3.33%) and Model 2 (independent candidate and gate MLPs, 47.25% ± 4.19%) perform identically ($\Delta = +0.02\%$, $p = 0.996$). Representation sharing between gate and candidate is neither beneficial nor detrimental to accuracy. However, a shared trunk is practically preferable because it executes a single joint linear matrix multiplication instead of two smaller independent passes, saving wall-clock training time (52.6s vs. 62.2s).

### Question 3: Does MGU's pre-gating mechanism explain most of its advantage?
**YES.**
In purely affine models, Canonical MGU achieves **36.08%** exact match while Canonical UGRNN achieves only **9.85%** (a +26.23% delta). The sole mathematical difference between them is that MGU computes its candidate as $\tanh(W e + U (f \odot S))$ rather than $\tanh(W e + U S)$. Pre-multiplying the state by the forget gate $f_t$ provides a bilinear gating cross-term that filters out irrelevant register states before the linear transformation, giving MGU a decisive edge over standard affine RNNs.

### Question 4: Is D's remaining advantage over MGU robust?
**NO — MGU with non-linear candidate processing decisively beats Model D.**
While Model D outperforms *Canonical* (affine) MGU (47.27% vs. 36.08%), this was an unfair comparison between a non-linear network (Model D) and an affine network (Canonical MGU). When MGU is upgraded to use an equal-parameter non-linear candidate MLP (Model 6), it achieves **54.28% ± 2.81% exact match**, outperforming Model D by **+7.02% ± 1.38%** ($p = 0.00034$) across every seed. MGU's state pre-gating combined with a non-linear candidate generator is empirically superior to Model D's un-gated shared trunk.

### Question 5: What exact mechanism should become the baseline persistent-state core for RE:Reason?
**Nonlinear Candidate MGU (`NonlinearCandidateMGUCore`) or Model D (`BoundedConvexCore`).**
- **Recommendation:** Adopt **`NonlinearCandidateMGUCore`** as the new primary state transition core. It achieves the project's highest recorded benchmark accuracy (**54.28% test exact match, 85.05% per-register accuracy**), maintains exact parameter parity (226K params), and provides both structural pre-gating ($f \odot S$) and non-linear candidate synthesis.
- Both architectures belong to the classical convex-bounded gated recurrent family (Zhou et al., 2016; Collins et al., 2017) augmented with modern non-linear MLP feature mixing. RE:Reason adopts this as a reliable, gradient-stable substrate for subsequent reasoning phases.

---

## 7. Next Steps for Phase 2

1. **Lock the Base Recurrent Core:**
   - Designate `NonlinearCandidateMGUCore` as `PersistentStateCore` (or retain `BoundedConvexCore` as an equivalent 47%+ alternative).
   - Document both cores in `docs/02_core_architecture.md` with explicit attribution to Zhou et al. (2016) and Collins et al. (2017).
2. **Phase 1 Benchmark Officially Closed:**
   - Initial Phase 1 RE:Reason baseline: 5.95% exact match.
   - Audited final recurrent persistent state baseline: **54.28% exact match** (an absolute gain of **+48.33%**).
3. **Advance to Phase 2 (Branching & Deliberation):**
   - With a verified, gradient-stable persistent state core in place, the project can now proceed to test RE:Reason's true research hypotheses: speculative hypothesis branching ($K > 1$), recursive deliberation ($R > 1$), isolated state verification, and transactional commits.
