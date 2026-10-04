# RE:Reason Phase 1.5: Causal Architectural Ablation Report

**Date:** 2026-10-04  
**Hardware:** AMD Radeon RX 6600M (8 GB VRAM, ROCm 6.x / PyTorch 2.14, `HSA_OVERRIDE_GFX_VERSION=10.3.0`)  
**Scope:** Phase 1.5 Causal Architectural Ablation across 4 parameter-matched models and 5 seeds  
**Official Benchmark:** Register Tracking ($N=4$ registers, $T=10$ steps, $V=10$ values, terminal supervision strictly at $t=10$)  
**Machine-Readable Telemetry:** [`experiments/ablation_1_5_results.json`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/ablation_1_5_results.json)  
**Code Implementations:**
- Cores: [`src/re_reason/cognition/ablation_cores.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/src/re_reason/cognition/ablation_cores.py)
- Models: [`src/re_reason/core/ablation_models.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/src/re_reason/core/ablation_models.py)
- Test Suite: [`tests/test_phase1_5_ablation.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/tests/test_phase1_5_ablation.py)
- Runner: [`experiments/run_ablation_1_5.py`](file:///home/chandrabhan/Documents/Personal%20Projects/re-reason/experiments/run_ablation_1_5.py)

---

## 1. Executive Summary & Causal Diagnosis

To definitively resolve why RE:Reason trailed the parameter-matched GRU baseline under terminal-only supervision, we executed a controlled causal ablation comparing four strictly parameter-matched models across five independent random seeds (`42, 43, 44, 45, 46`).

```
Model Comparison Architectures:
Model A (Current RE:Reason):     S_t = LayerNorm(S_(t-1) + f_theta(S_(t-1), e_t))
Model B (Residual No LayerNorm): S_t = S_(t-1) + f_theta(S_(t-1), e_t)
Model C (Gated Residual):        S_t = S_(t-1) + g_theta(S_(t-1), e_t) ⊙ f_theta(S_(t-1), e_t)
Model D (GRU Baseline):          h_t = GRUCell(e_t, h_(t-1))
```

### The Central Causal Question:
> *"Is RE:Reason losing because of LayerNorm, because it lacks gating, because additive recurrence itself is inadequate, or because terminal-only supervision exposes a broader temporal-credit problem?"*

### The Empirical Findings:
1. **LayerNorm is not the sole or primary performance bottleneck:**
   - Removing LayerNorm (Model B) yielded **$5.93\% \pm 1.18\%$** test exact match, virtually identical to Model A's **$5.83\% \pm 1.27\%$** (+0.10% delta, well within seed variance).
   - Removing LayerNorm allowed latent state norms to drift monotonically from 8.9 to 20.7, with greater degradation on longer horizons ($T=20$: 0.9% vs. 2.1%).
2. **Model C tested an unbounded gated residual, NOT a bounded convex transition:**
   - Model C implemented $S_t = S_{t-1} + g_\theta(S_{t-1}, e_t) \odot f_\theta(S_{t-1}, e_t)$ with unbounded candidate $f_\theta$. It achieved **$4.73\% \pm 0.91\%$** test exact match.
   - This result demonstrates only that adding a multiplicative gate to an unbounded additive residual is insufficient in this configuration. It does NOT evaluate or falsify bounded convex interpolation ($(1-z)S + z\tanh(f)$).
3. **Additive recurrence under terminal supervision vs. step-wise supervision:**
   - Under **terminal-only supervision**, additive models (A, B, C) achieve 4.7%–5.9% test exact match, trailing the GRU (10.78%).
   - Under **step-wise supervision**, the 256-dimensional representation supports substantial partial tracking (**20.67%** test exact match, trailing GRU by 1.83%).
   - This indicates that terminal credit assignment across multiple unsupervised recurrent steps remains a primary difficulty for un-gated additive updates.

---

## 2. Experimental Controls & Parameter Parity

All models utilized identical 2-layer MLP observation encoders, linear typed decision heads, batch size 64, AdamW ($\text{LR}=10^{-3}, \text{WD}=0.01$), 3,000 training steps, and identical cryptographic data splits:

| Model Architecture | Update Equation | Exact Parameters | Delta vs. Model A | Under 250K Ceiling? |
|:---|:---|:---:|:---:|:---:|
| **Model A (Current RE:Reason)** | $S_t = \text{LayerNorm}(S_{t-1} + f_\theta)$ | 226,472 | — | **YES** (-9.4%) |
| **Model B (Residual No LayerNorm)** | $S_t = S_{t-1} + f_\theta$ | 225,960 | -512 (-0.23%) | **YES** (-9.6%) |
| **Model C (Gated Residual)** | $S_t = S_{t-1} + g_\theta \odot f_\theta$ | 226,152 | -320 (-0.14%) | **YES** (-9.5%) |
| **Model D (GRU Baseline)** | $h_t = \text{GRUCell}(e_t, h_{t-1})$ | 229,200 | +2,728 (+1.20%) | **YES** (-8.3%) |

All parameter counts match within **$\pm 1.2\%$**—far tighter than the $\pm 5\%$ parity target.

---

## 3. Aggregate Performance Across 5 Independent Seeds

All metrics report `mean ± standard deviation` over seeds `[42, 43, 44, 45, 46]`.  
`test_at_best_val_checkpoint` denotes strictly held-out test evaluation using the checkpoint selected by peak validation exact match:

| Metric | Model A (LayerNorm) | Model B (No LayerNorm) | Model C (Gated Residual) | Model D (GRU Baseline) |
|:---|:---:|:---:|:---:|:---:|
| **Test Exact (`test_at_best_val_checkpoint`)** | 5.83% ± 1.27% | 5.93% ± 1.18% | 4.73% ± 0.91% | **10.78% ± 0.44%** |
| **Test Per-Register Accuracy** | 46.90% ± 0.99% | 47.53% ± 1.52% | 45.88% ± 0.97% | **55.66% ± 1.05%** |
| **Best Validation Exact Match** | 6.16% ± 0.42% | 6.02% ± 0.59% | 5.82% ± 0.53% | **10.68% ± 0.75%** |
| **Final Training Exact Match** | 52.81% ± 2.69% | 55.94% ± 8.11% | 78.12% ± 3.42% | **76.56% ± 6.77%** |
| **Training Time (3,000 steps)** | 47.7s | 44.7s | 50.4s | **42.8s** |
| **Inference Latency (per batch)** | 0.051 ms | 0.048 ms | 0.052 ms | **0.046 ms** |
| **Peak GPU Memory** | 18.2 MB | 17.8 MB | 18.4 MB | **16.5 MB** |

---

## 4. Temporal Credit-Assignment & Gradient Dynamics

We measured backpropagated gradient norms $\|\partial \mathcal{L}_{\text{term}} / \partial S_t\|$ and state norms $\|S_t\|$ across timesteps $t \in \{1, \dots, 10\}$ on held-out test batches:

### A. Gradient Attenuation Across Sequence Timesteps
| Metric / Timestep | Model A (LayerNorm) | Model B (No LayerNorm) | Model C (Gated Residual) | Model D (GRU Baseline) |
|:---|:---:|:---:|:---:|:---:|
| **Gradient at $t=10$ ($\|\partial \mathcal{L}/\partial S_{10}\|$)** | $1.352 \times 10^{-3}$ | $1.364 \times 10^{-3}$ | $1.406 \times 10^{-3}$ | $3.150 \times 10^{-3}$ |
| **Gradient at $t=5$ ($\|\partial \mathcal{L}/\partial S_5\|$)** | $0.954 \times 10^{-3}$ | $1.551 \times 10^{-3}$ | $1.561 \times 10^{-3}$ | $2.211 \times 10^{-3}$ |
| **Gradient at $t=1$ ($\|\partial \mathcal{L}/\partial S_1\|$)** | $0.663 \times 10^{-3}$ | $1.800 \times 10^{-3}$ | $1.814 \times 10^{-3}$ | $1.450 \times 10^{-3}$ |
| **Attenuation Ratio ($\|\partial \mathcal{L}/\partial S_1\| / \|\partial \mathcal{L}/\partial S_{10}\|$)** | **0.468 ± 0.134** | **1.279 ± 0.260** | **1.197 ± 0.325** | **0.458 ± 0.030** |
| **Temporal Gradient Status** | Smooth Decay (~53%) | Slight Growth (~28%) | Slight Growth (~20%) | Smooth Decay (~54%) |

### B. State Norm Drift Across Timesteps ($\|S_t\|$)
| Timestep | Model A (LayerNorm) | Model B (No LayerNorm) | Model C (Gated Residual) | Model D (GRU Baseline) |
|:---:|:---:|:---:|:---:|:---:|
| **$t=1$** | 15.96 | 8.91 | 11.20 | 7.00 |
| **$t=2$** | 15.96 | 12.32 | 15.74 | 8.78 |
| **$t=4$** | 16.02 | 14.10 | 18.67 | 9.67 |
| **$t=6$** | 16.05 | 16.18 | 22.13 | 10.02 |
| **$t=8$** | 16.07 | 18.44 | 25.90 | 10.18 |
| **$t=10$** | **16.08** | **20.72** | **29.98** | **10.26** |
| **Dynamic Stability** | **Strictly Constant** | **Monotonic Growth (+132%)** | **Rapid Explosion (+168%)** | **Bounded Saturation (+46%)** |

```
State Norm Evolution:
Model A (LayerNorm):  [16.0 -> 16.0 -> 16.0 -> 16.1] (Strictly preserved by LayerNorm)
Model B (No LN):      [ 8.9 -> 14.1 -> 18.4 -> 20.7] (Continuous unconstrained accumulation)
Model C (Gated Res):  [11.2 -> 18.7 -> 25.9 -> 30.0] (Rapid unchecked divergence)
Model D (GRU):        [ 7.0 ->  9.7 -> 10.2 -> 10.3] (Bounded saturation via tanh + convex gate)
```

> [!NOTE]
> **Key Gradient & Stability Observations:**
> 1. Under this specific aggregate terminal measurement, backpropagated state gradient norms in Model A and Model D attenuate by comparable amounts (~53% vs. ~54%). This indicates similar aggregate gradient attenuation across the 10-step recurrent chain rather than an extreme vanishing gradient discrepancy between A and D, though path-specific or coordinate-level gradient dynamics may still differ.
> 2. Model B and Model C exhibit backward gradient amplification (ratio > 1.0), yet their generalization performance remains poor (5.93% and 4.73%), demonstrating that higher backpropagated gradient magnitude alone does not produce better generalization.
> 3. LayerNorm in Model A plays a stabilizing role: it prevents the state norm from growing unconstrained. Without LayerNorm (Model B and C), the state norm drifts upwards monotonically, leading to severe degradation over longer horizons.

---

## 5. Horizon Generalization Diagnostics ($T=5, 10, 15, 20$)

Models trained on $T=10$ sequences were evaluated zero-shot on sequence horizons $T \in \{5, 10, 15, 20\}$:

| Horizon | Model A (LayerNorm) | Model B (No LayerNorm) | Model C (Gated Residual) | Model D (GRU Baseline) |
|:---:|:---:|:---:|:---:|:---:|
| **$T=5$ (Exact / Per-Reg)** | 14.5% / 59.7% | 15.9% / 60.8% | 12.6% / 56.1% | **27.4% / 72.0%** |
| **$T=10$ (Exact / Per-Reg)** | 5.2% / 45.8% | 5.9% / 46.9% | 4.6% / 45.1% | **9.9% / 55.6%** |
| **$T=15$ (Exact / Per-Reg)** | 2.4% / 39.6% | 2.2% / 36.9% | 1.9% / 34.6% | **5.1% / 46.9%** |
| **$T=20$ (Exact / Per-Reg)** | **2.1% / 36.3%** | 0.9% / 29.2% | 0.7% / 26.0% | **3.7% / 43.4%** |

### Findings:
- At $T=5$, the GRU achieves 27.4% exact match (72.0% per-register accuracy), substantially higher than the additive models.
- At $T=20$, Model B (0.9%) and Model C (0.7%) experience near-total collapse because their state norms explode across long horizons.
- Model A maintains 2.1% at $T=20$ due to LayerNorm stabilization, but remains behind the GRU (3.7%).

---

## 6. Synthesis: Distinguishing Empirical Findings from Causal Hypotheses

The ablation provides empirical evidence on several structural factors:
- **LayerNorm (Model A vs. Model B):** Removing LayerNorm changes test exact match from 5.83% to 5.93% (statistically indistinguishable), while destabilizing state norms ($8.9 \to 20.7$). LayerNorm is therefore not the primary cause of RE:Reason's performance gap.
- **Unbounded Residual Gating (Model C):** Adding coordinate-wise residual gating ($S_t = S_{t-1} + g \odot f$) without bounding the candidate update results in 4.73% test exact match and causes state norm explosion ($11.2 \to 30.0$). **Important Caveat:** Model C does NOT test bounded convex interpolation ($S_t = (1-z)S_{t-1} + z\tanh(f)$); its result only shows that an unbounded additive gated residual is insufficient.
- **Gradient Attenuation (Model A vs. Model D):** Aggregate attenuation ratios are comparable (0.468 vs. 0.458), indicating that aggregate vanishing gradients do not explain the gap under this metric.

### Causal Hypotheses Regarding the GRU Advantage:
The GRU baseline's recurrent update features two architectural mechanisms absent in Models A, B, and C:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tanh(W x_t + U (r_t \odot h_{t-1}) + b)$$

1. **Convex Multiplicative Retention:** The $(1 - z_t)$ factor provides a direct multiplicative decay/retention mechanism in hidden-state coordinates, with candidate updates strictly bounded by $\tanh \in (-1, 1)$.
2. **Selective Gating in Hidden Space:** Update ($z_t$) and reset ($r_t$) gates operate in distributed hidden-state coordinates, enabling selective state modulation. (Note: These gates operate in continuous hidden coordinates, not direct 1:1 symbolic register slots).

In contrast, unconstrained additive recurrence $S_t = S_{t-1} + \Delta_t$ allows state norms to drift without bound unless stabilized by LayerNorm. When LayerNorm is applied, the global normalization couples all latent dimensions.

---

## 7. Phase 1.5 Diagnostic Status & Next Step

```
Ablation Summary:
- Model A (LayerNorm):      5.83% ± 1.27% (Test Exact) | State norm: 16.0 (Stable)
- Model B (No LayerNorm):   5.93% ± 1.18% (Test Exact) | State norm: 20.7 (Drifting)
- Model C (Unbounded Gate): 4.73% ± 0.91% (Test Exact) | State norm: 30.0 (Exploding)
- Model D (GRU Baseline):  10.78% ± 0.44% (Test Exact) | State norm: 10.3 (Bounded)
```

### Critical Gap in Previous Ablation:
Model C evaluated an unbounded additive gated residual ($S + g \odot f$). It did **NOT** evaluate a bounded convex interpolation:
$$S_t = (1 - z_t) \odot S_{t-1} + z_t \odot \tanh(f_\theta(S_{t-1}, e_t))$$
Therefore, the hypothesis that bounded convex state updating improves persistent latent recurrence remains **untested**.

### Action:
Execute a dedicated Phase 1.5 follow-up ablation implementing **Model E (Bounded Convex Residual Update)** to directly test this missing hypothesis before any Phase 2 mechanisms are considered.
