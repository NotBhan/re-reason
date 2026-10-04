# RE:Reason Phase 1.5: Bounded Convex Residual Update Ablation Report

**Date:** 2026-10-04  
**Hardware:** AMD Radeon RX 6600M (8 GB VRAM, ROCm 6.x / PyTorch 2.14, `HSA_OVERRIDE_GFX_VERSION=10.3.0`)  
**Scope:** Controlled causal ablation evaluating Model E (Bounded Convex Residual Update) against Model A (Current RE:Reason) and Model D (GRU Baseline) across 5 independent random seeds  
**Official Benchmark:** Register Tracking ($N=4$ registers, $T=10$ steps, $V=10$ values, terminal supervision strictly at $t=10$)  
**Machine-Readable Telemetry:** [`experiments/ablation_1_5_bounded_results.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/ablation_1_5_bounded_results.json)  
**Code Implementations:**
- Core: [`BoundedConvexCore` in `src/re_reason/cognition/ablation_cores.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/src/re_reason/cognition/ablation_cores.py)
- Model: [`ReReasonBoundedConvexModel` in `src/re_reason/core/ablation_models.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/src/re_reason/core/ablation_models.py)
- Test Suite: [`tests/test_phase1_5_ablation.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/tests/test_phase1_5_ablation.py)
- Runner: [`experiments/run_ablation_1_5_bounded.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/run_ablation_1_5_bounded.py)

---

## 1. Executive Summary & Causal Verdict

### The Primary Causal Question:
> *"Does bounded convex state updating, without a GRU-style reset gate, materially improve RE:Reason under the official terminal-supervision benchmark?"*

### The Verdict:
**YES, emphatically and decisively.**

Under strict parameter parity ($\pm 1.2\%$), identical cryptographic procedural splits, and identical terminal-only supervision at $t=10$, **Model E (Bounded Convex Residual Update)** completely alters the performance landscape of RE:Reason:

```
Test Exact Match (test_at_best_val_checkpoint across 5 seeds):
Model A (Current RE:Reason - LayerNorm):      6.13% ± 0.73%  (Per-Reg: 47.36% ± 0.46%)
Model D (Parameter-matched GRU baseline):    11.25% ± 1.29%  (Per-Reg: 57.34% ± 1.08%)
Model E (Bounded Convex Residual Update):   45.55% ± 4.38%  (Per-Reg: 81.35% ± 1.94%)
```

- **Delta vs. Current RE:Reason (Model A):** **+39.42% absolute (+643% relative)** test exact match improvement.
- **Delta vs. GRU Baseline (Model D):** **+34.30% absolute (+305% relative)** test exact match improvement.
- **Parameters:** Model E has **226,152 parameters**—which is 320 parameters (-0.14%) *fewer* than Model A and 3,048 parameters (-1.33%) *fewer* than Model D.

### Interpretation Matrix Classification: **CASE 1 (Strong Mechanism Confirmation)**
> **Case 1 Definition:** *"E substantially improves over A and approaches GRU. Conclusion: bounded convex state transition is strong evidence for the proposed mechanism."*

In fact, Model E does not merely approach the GRU baseline—it **drastically outperforms both Model A and the GRU baseline**. It demonstrates that neither a complex GRU reset gate ($r_t$) nor later-phase mechanisms (recursion $R>1$, branching $K>1$, verifier, or transactional commits) are required to solve the terminal-credit bottleneck on Register Tracking. The fatal flaw in previous models was the **unbounded additive state formulation**, which forced a destructive trade-off between unconstrained norm divergence (Models B and C) and global dimensional coupling via LayerNorm (Model A).

---

## 2. Architectural Formulations & Mathematical Boundedness

To understand why Model E succeeded where previous models failed, we compare the exact update equations:

```
Model Formulations:
---------------------------------------------------------------------------------------------------------
Model A (Current RE:Reason):  S_t = LayerNorm(S_(t-1) + f_theta(S_(t-1), e_t))
Model B (No LayerNorm):       S_t = S_(t-1) + f_theta(S_(t-1), e_t)
Model C (Gated Residual):     S_t = S_(t-1) + g_theta(S_(t-1), e_t) ⊙ f_theta(S_(t-1), e_t)
Model D (GRU Baseline):       h_t = (1 - z_t) ⊙ h_(t-1) + z_t ⊙ tanh(W e_t + U(r_t ⊙ h_(t-1)) + b)
Model E (Bounded Convex):     S_t = (1 - z_t) ⊙ S_(t-1) + z_t ⊙ tanh(f_theta(S_(t-1), e_t))
                              where z_t = sigmoid(g_theta(S_(t-1), e_t))
---------------------------------------------------------------------------------------------------------
```

### Mathematical Guarantees of Model E:
1. **Bounded Candidate State:**
   $$\tilde{S}_t = \tanh(f_\theta(S_{t-1}, e_t)) \in (-1, 1)^d$$
   The candidate update is strictly confined to the hypercube $(-1, 1)^{256}$.
2. **Coordinate-wise Convex Interpolation Factor:**
   $$z_t = \sigma(g_\theta(S_{t-1}, e_t)) \in (0, 1)^d$$
   Each latent coordinate $i \in \{1, \dots, 256\}$ has an independent interpolation coefficient $z_{t, i}$.
3. **Element-wise Convexity & Universal State Boundedness:**
   $$S_{t, i} = (1 - z_{t, i}) S_{t-1, i} + z_{t, i} \tilde{S}_{t, i}$$
   Because $S_{0} = 0 \in [-1, 1]^d$ and $\tilde{S}_t \in (-1, 1)^d$, by induction every future state is a convex combination of points in $[-1, 1]^d$:
   $$S_{t, i} \in [-1, 1] \quad \forall i, \forall t$$
   $$\|S_t\|_2 \le \sqrt{d} = \sqrt{256} = 16.0 \quad \forall t$$
   **State norm boundedness is mathematically guaranteed without LayerNorm**, preventing both the unconstrained norm explosion of Models B/C and the global inter-register interference caused by LayerNorm in Model A.
4. **No Reset Gate, Fixed Recurrence:**
   Model E contains no reset gate $r_t$, no adaptive halting, no recursion ($R=1$), and no branching ($K=1$).

---

## 3. Parameter Parity & Experimental Controls

All models were evaluated under identical training and data conditions:
- **Optimizer:** AdamW ($\text{LR} = 10^{-3}, \text{weight\_decay} = 0.01, \text{grad\_clip} = 1.0$)
- **Batch size:** 64
- **Training budget:** 3,000 steps
- **Validation interval:** 100 steps (model selection strictly on peak validation exact match)
- **Seeds:** 5 independent runs per model (`[42, 43, 44, 45, 46]`)
- **Procedural data splits:** 6,000 train, 1,200 val, 1,200 test with cryptographic SHA-256 collision verification (0 cross-split leakage).
- **Metric naming:** Final held-out test evaluation is strictly named `test_at_best_val_checkpoint`.

| Model Architecture | Update Equation | Exact Parameters | Delta vs. Model A | Under 250K Ceiling? |
|:---|:---|:---:|:---:|:---:|
| **Model A (Current RE:Reason)** | $S_t = \text{LayerNorm}(S_{t-1} + f_\theta)$ | 226,472 | — | **YES** (-9.4%) |
| **Model E (Bounded Convex)** | $S_t = (1-z_t)S_{t-1} + z_t\tanh(f_\theta)$ | 226,152 | -320 (-0.14%) | **YES** (-9.5%) |
| **Model D (GRU Baseline)** | $h_t = \text{GRUCell}(e_t, h_{t-1})$ | 229,200 | +2,728 (+1.20%) | **YES** (-8.3%) |

Parameter counts match within $\pm 1.2\%$, well within the required $\pm 5\%$ parity window.

---

## 4. Aggregate Performance Across 5 Independent Seeds

All values report `mean ± standard deviation` over seeds `[42, 43, 44, 45, 46]`:

| Metric | Model A (LayerNorm) | Model E (Bounded Convex) | Model D (GRU Baseline) |
|:---|:---:|:---:|:---:|
| **Test Exact (`test_at_best_val_checkpoint`)** | 6.13% ± 0.73% | **45.55% ± 4.38%** | 11.25% ± 1.29% |
| **Test Per-Register Accuracy** | 47.36% ± 0.46% | **81.35% ± 1.94%** | 57.34% ± 1.08% |
| **Best Validation Exact Match** | 6.20% ± 0.43% | **43.90% ± 4.77%** | 11.28% ± 0.72% |
| **Final Training Exact Match** | 54.69% ± 4.74% | **100.00% ± 0.00%** | 79.38% ± 7.93% |
| **Training Time (3,000 steps)** | 43.0s | 49.1s | **36.3s** |
| **Inference Latency (per batch)** | 0.073 ms | 0.081 ms | **0.055 ms** |
| **Peak GPU Memory** | 18.2 MB | 18.5 MB | **16.5 MB** |
| **Gradient Attenuation ($\|g_1\| / \|g_{10}\|$)** | 0.404 ± 0.049 | **0.715 ± 0.023** | 0.419 ± 0.081 |

### Seed-by-Seed Consistency:
| Seed | Model A Test Exact (Per-Reg) | Model E Test Exact (Per-Reg) | Model D Test Exact (Per-Reg) |
|:---:|:---:|:---:|:---:|
| **42** | 4.92% (47.46%) | **41.00% (79.60%)** | 9.67% (55.92%) |
| **43** | 6.00% (46.46%) | **46.33% (81.58%)** | 12.83% (58.23%) |
| **44** | 6.42% (47.62%) | **40.00% (78.71%)** | 9.83% (56.17%) |
| **45** | 6.17% (47.56%) | **50.67% (83.65%)** | 12.33% (58.50%) |
| **46** | 7.17% (47.71%) | **49.75% (83.19%)** | 11.58% (57.88%) |

Model E's advantage is not a seed artifact: in every single run, Model E achieves between 40.0% and 50.7% exact match, outperforming Model A by 6x to 8x and GRU by 3x to 5x.

---

## 5. Latent State Norms & Candidate Norm Dynamics

We measured state norms $\|S_t\|_2$ and candidate norms $\|\tilde{S}_t\|_2$ across sequence steps $t \in \{1, \dots, 10\}$:

| Step $t$ | Model A State Norm $\|S_t\|$ | Model E State Norm $\|S_t\|$ | Model E Candidate Norm $\|\tilde{S}_t\|$ | Model D State Norm $\|h_t\|$ |
|:---:|:---:|:---:|:---:|:---:|
| **$t=1$** | 15.91 | 10.29 | 15.42 | 6.91 |
| **$t=2$** | 15.92 | 12.22 | 15.30 | 8.67 |
| **$t=3$** | 15.95 | 12.69 | 14.82 | 9.22 |
| **$t=4$** | 15.98 | 12.96 | 14.82 | 9.57 |
| **$t=5$** | 16.00 | 13.12 | 14.80 | 9.77 |
| **$t=6$** | 16.01 | 13.23 | 14.80 | 9.91 |
| **$t=7$** | 16.02 | 13.29 | 14.79 | 10.01 |
| **$t=8$** | 16.03 | 13.30 | 14.78 | 10.07 |
| **$t=9$** | 16.04 | 13.30 | 14.77 | 10.11 |
| **$t=10$** | **16.04** | **13.31** | **14.76** | **10.15** |
| **Dynamic Behavior** | **Enforced Constant** | **Smooth Bounded Saturation** | **Strictly Bounded Hypercube** | **Saturating Sigmoid/Tanh** |

### Analysis:
- In Model E, the latent state starts at 0, absorbs the first updates smoothly, and converges to a stable norm of **13.31**, well below the theoretical ceiling of 16.0.
- Candidate states $\|\tilde{S}_t\|$ remain bounded at ~14.76 across all steps.
- Unlike Model B and Model C where norms exploded to 20.7 and 30.0, Model E achieves natural asymptotic stability without any global normalization layer.

---

## 6. Temporal Credit Assignment & Gradient Flow

Backpropagated gradient norms $\|\partial \mathcal{L}_{\text{term}} / \partial S_t\|$ reveal a profound change in backward credit assignment:

| Step $t$ | Model A (LayerNorm) | Model E (Bounded Convex) | Model D (GRU Baseline) |
|:---:|:---:|:---:|:---:|
| **$t=10$** | $1.238 \times 10^{-3}$ | $2.141 \times 10^{-3}$ | $2.825 \times 10^{-3}$ |
| **$t=8$** | $0.994 \times 10^{-3}$ | $2.272 \times 10^{-3}$ | $2.410 \times 10^{-3}$ |
| **$t=6$** | $0.846 \times 10^{-3}$ | $2.262 \times 10^{-3}$ | $2.080 \times 10^{-3}$ |
| **$t=4$** | $0.731 \times 10^{-3}$ | $2.098 \times 10^{-3}$ | $1.798 \times 10^{-3}$ |
| **$t=2$** | $0.631 \times 10^{-3}$ | $1.851 \times 10^{-3}$ | $1.534 \times 10^{-3}$ |
| **$t=1$** | $0.503 \times 10^{-3}$ | **$1.534 \times 10^{-3}$** | $1.237 \times 10^{-3}$ |
| **Attenuation Ratio ($\|g_1\| / \|g_{10}\|$)** | **0.404 ± 0.049** | **0.715 ± 0.023** | **0.419 ± 0.081** |

```
Backward Gradient Retention:
Model A (LayerNorm):      [0.00124  -->  0.00050] (40.4% retained; severe decay)
Model D (GRU Baseline):    [0.00282  -->  0.00124] (41.9% retained; steady decay)
Model E (Bounded Convex): [0.00214  -->  0.00153] (71.5% retained; superior preservation)
```

### Why Does Model E Preserve Gradients Far Better?
In Model E, the backward Jacobian of the state transition contains a direct multiplicative path:
$$\frac{\partial S_t}{\partial S_{t-1}} = \text{diag}(1 - z_t) + \dots$$
When gate coordinates $z_{t, i}$ are low (retaining state), the gradient flows backward with multiplier $(1 - z_{t, i}) \approx 1.0$. Because there is no LayerNorm dividing gradients by global state standard deviations, gradients retain **71.5%** of their terminal magnitude at step 1 (over 3x higher than Model A's $0.503 \times 10^{-3}$).

---

## 7. Learned Gate Dynamics & Selectivity Diagnostics

We tested whether the learned gate $z_t$ becomes selective across operations (`SET`, `SWAP`, `ADD`) and whether coordinates map to symbolic registers.

### A. Gate Statistics Across Sequence Steps
| Step $t$ | Mean Gate Activation | Gate Std Dev | Gate Min | Gate Max |
|:---:|:---:|:---:|:---:|:---:|
| **$t=1$** | **0.466** | 0.461 | $1.0 \times 10^{-14}$ | 1.000 |
| **$t=2$** | 0.414 | 0.456 | $4.5 \times 10^{-17}$ | 1.000 |
| **$t=3$** | 0.314 | 0.424 | $1.2 \times 10^{-17}$ | 1.000 |
| **$t=5$** | 0.284 | 0.414 | $6.1 \times 10^{-18}$ | 1.000 |
| **$t=8$** | 0.272 | 0.410 | $9.7 \times 10^{-18}$ | 1.000 |
| **$t=10$** | **0.266** | 0.407 | $6.9 \times 10^{-17}$ | 1.000 |

1. **Early vs. Late Plasticity:** At steps 1–2 (when $S_0=0$ is empty), the gate mean is high (~0.44) to rapidly write initial register values. By step 10, the gate mean drops to ~0.27, reflecting preservation of existing state.
2. **Bimodal Binarization:** Notice the large standard deviation (~0.41 to 0.46) with $\min \approx 0.0$ and $\max = 1.0$. The gate distribution is strongly **bimodal**, with coordinates saturating near 0 (retain) or 1 (overwrite).

### B. Operation-Specific Gate Selectivity
| Operation Type | Semantic Function | Mean Gate Activation | Std Dev | Gate Range |
|:---|:---|:---:|:---:|:---:|
| **`SET`** | Overwrite single register | **0.3755** | 0.450 | $[1.5 \times 10^{-18}, 1.0]$ |
| **`SWAP`** | Permute two registers | **0.3229** | 0.432 | $[3.1 \times 10^{-14}, 1.0]$ |
| **`ADD`** | In-place modular accumulation | **0.1905** | 0.348 | $[2.0 \times 10^{-16}, 1.0]$ |

The learned gate shows clear functional differentiation:
- `SET` operations trigger the highest gate activation (mean 0.3755), actively overwriting the state with new values.
- `ADD` operations trigger the lowest gate activation (mean 0.1905), preserving more of the existing accumulated state coordinates.

### C. Empirical Test of Register Coordinate Mapping:
> **Question:** *"Do coordinate-wise gate values correspond to 4 contiguous 64-dimensional slices for registers R0, R1, R2, R3?"*

We measured the difference between the mean gate of the targeted register slice ($[r \cdot 64 : (r+1) \cdot 64]$) and non-targeted slices during `SET` operations:
$$\Delta_{\text{target - other}} = -0.0050 \pm 0.0538$$
**Empirical Conclusion:** The difference is statistically zero. The gate coordinates do **NOT** align with naive contiguous symbolic register slices. Instead, the 256 coordinates form a **distributed representation** where register identities and values are encoded in non-orthogonal linear subspaces.

---

## 8. Zero-Shot Horizon Generalization ($T=5, 10, 15, 20$)

Models trained strictly on $T=10$ sequences were evaluated zero-shot on sequence horizons $T \in \{5, 10, 15, 20\}$:

| Horizon | Model A (LayerNorm) Exact / Per-Reg | Model E (Bounded Convex) Exact / Per-Reg | Model D (GRU Baseline) Exact / Per-Reg |
|:---:|:---:|:---:|:---:|
| **$T=5$** | 13.97% / 58.65% | **79.95% ± 3.82% / 94.55% ± 1.10%** | 27.22% / 72.02% |
| **$T=10$** | 5.45% / 46.94% | **44.07% ± 3.91% / 80.56% ± 1.89%** | 10.68% / 56.46% |
| **$T=15$** | 2.75% / 39.87% | **24.67% ± 3.27% / 68.53% ± 2.19%** | 5.52% / 47.78% |
| **$T=20$** | 1.72% / 36.49% | **16.43% ± 2.02% / 61.80% ± 1.58%** | 4.23% / 44.06% |

```
Horizon Scaling (Exact Match):
T=5:   Model A: 14.0%  | Model D: 27.2%  | Model E: 80.0%  (Near-complete solution!)
T=10:  Model A:  5.5%  | Model D: 10.7%  | Model E: 44.1%  (4x above GRU)
T=15:  Model A:  2.8%  | Model D:  5.5%  | Model E: 24.7%  (4.5x above GRU)
T=20:  Model A:  1.7%  | Model D:  4.2%  | Model E: 16.4%  (almost 10x above Model A!)
```

At $T=5$, Model E achieves **79.95% exact match** and **94.55% per-register accuracy** without any intermediate supervision. Even at double the training sequence length ($T=20$), Model E maintains **16.43% exact match**, outperforming Model A's $T=10$ performance by nearly 3x.

---

## 9. Causal Synthesis & Phase 1.5 Gate Decision

### What We Learned Across All Phase 1.5 Ablations:
1. **Model A ($S_t = \text{LayerNorm}(S_{t-1} + f)$):** LayerNorm prevents explosion, but its global variance-normalization divides all coordinates equally, destroying selective register persistence and causing severe gradient attenuation (ratio 0.404). Result: **6.13% exact match**.
2. **Model B ($S_t = S_{t-1} + f$):** Removing LayerNorm without bounding the update causes state norm drift ($8.9 \to 20.7$) and fails on longer horizons. Result: **5.93% exact match**.
3. **Model C ($S_t = S_{t-1} + g \odot f$):** Multiplying an unbounded candidate by a gate without convex decay causes rapid norm explosion ($11.2 \to 30.0$). Result: **4.73% exact match**.
4. **Model D ($\text{GRUCell}$):** Combines bounded candidate ($\tanh$), coordinate update gate ($z$), and reset gate ($r$). Result: **11.25% exact match**.
5. **Model E ($S_t = (1-z)S_{t-1} + z\tanh(f)$):** Combines strictly bounded candidate ($\tanh$) with coordinate convex interpolation ($(1-z)$). Boundedness is guaranteed, gradient flow is preserved (ratio 0.715), and the network learns functional operation gating. Result: **45.55% exact match**.

### Phase 1.5 Gate Decision:
- **The causal bottleneck of Phase 1 has been identified and resolved.**
- **Decision:** Formally adopt the **Bounded Convex Residual Update (Model E)** as the foundational state update mechanism for the RE:Reason cognition core:
  $$S_t = (1 - z_t) \odot S_{t-1} + z_t \odot \tanh(f_\theta(S_{t-1}, e_t))$$
- **Do not introduce later-phase complexity prematurely:** We now have a recurrent baseline that achieves 45.55% exact match (and 80.0% at $T=5$) with $R=1, K=1$, no verifier, and no branching. Any Phase 2 mechanisms (such as recursive deliberation $R>1$) can now be evaluated against a sound, high-performing foundation.
