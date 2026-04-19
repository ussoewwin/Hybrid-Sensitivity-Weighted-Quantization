
# HSWQ: Hybrid Sensitivity Weighted Quantization — Technical Overview

**Document version:** 3.0  
**Date:** 2026-04-20  
**Targets:** SDXL (v1.3), Flux.1-dev (v1.6), Z Image Turbo / Base (v1.92)

---

## 1. Philosophy

Naive FP8 cast applies **one global rounding rule to every weight in every layer**. This blows up on diffusion UNets / DiTs because:

- A few **adaLN / time-embedder / FFN.w2** style layers have *extreme* outliers (kurtosis ≫ 20, abs_max ≫ FP8 safe range). They cannot fit in unscaled FP8 at all — the dense band in the center collapses.
- A different set of layers are **structurally critical** (their output variance dominates downstream activations). Even small rounding error here destroys global SSIM.
- The remaining "well-behaved" layers can be aggressively compressed if the **clipping amax** is chosen to minimize an **importance-weighted** error, not raw L2.

HSWQ treats these three populations differently. It is *not* a single quantizer; it is a **three-axis decision system**:

1. **Distribution Profile** (static, weight-only): kurtosis, outlier-ratio, abs_max → Hard VETO + dynamic per-layer search-range.
2. **Sensitivity** (dynamic, calibration-time): output variance via DualMonitor → Top-`keep_ratio` kept in FP16.
3. **Importance** (dynamic + structural): per-channel input mean-abs (V1 hist) **or** SVD leverage × RMS magnitude hybrid (V4 hist) → weights of the histogram MSE.

The output stays **standard FP8 E4M3** (`torch.float8_e4m3fn`) — no custom loader needed for the V1-compatible mode. Critical layers are simply preserved as FP16 inside the same Safetensors.

---

## 2. Family Status (current implementations)

| Family | Script | Histogram backend | Strategy engine | Notes |
|---|---|---|---|---|
| SDXL | `quantize_sdxl_hswq_v1.3.py` | `weighted_histogram_mse_fast` (searchsorted) | DualMonitor → keep_ratio FP16; channel-importance weighted MSE | 10–50× faster amax search vs. brute-force grid. SA2 attention optional during calibration. |
| Flux.1-dev | `quantize_flux_hswq_v1.6.py` | `weighted_histogram_mse_fast` | DualMonitor + **Adaptive search_range** per layer | Adaptive lower bound prevents over-clipping on Flux's spiky activation layers. |
| Z Image Turbo / Base | `quantize_zit_hswq_v1.6.py` / `quantize_zib_hswq_v1.92.py` | `weighted_histogram_mse_v4` (SVD + RMS hybrid) | **Pure Data-Driven Autonomous Engine** (Profile → Alpha/Beta + Hard VETO + Dynamic search_low) | Full V4 stack. Profile JSON is generated automatically by `analyze/analyze_zib_distribution.py` if missing. |

All families share the same FP8 E4M3 physical-grid simulator and emit `comfy_quant` / `weight_scale` metadata so the result loads in standard ComfyUI/Diffusers FP8 paths.

---

## 3. Architecture

```mermaid
graph TD
    A[Calibration Inputs] --> B[Diffusers Pipeline + Hooks]
    B --> C{Dual Monitor}
    C --> C1[Sensitivity = Var(output)]
    C --> C2[Importance = Mean(|X|_c)]
    W[FP16 weights] --> P[Static Distribution Profile]
    P --> P1[kurtosis]
    P --> P2[outlier_ratio]
    P --> P3[abs_max]

    P1 --> V{Hard VETO?}
    P2 --> V
    P3 --> V
    V -- yes --> KFP16[Force FP16]

    P1 --> S[Autonomous Strategy]
    P2 --> S
    P3 --> S
    S --> SAlpha[Alpha = SVD weight]
    S --> SBeta[Beta = RMS weight]
    S --> SLow[get_dynamic_search_low]

    C1 --> Sel{Top keep_ratio<br/>by sensitivity}
    Sel --> KFP16
    Sel -- not selected --> Q[FP8 candidate]

    Q --> H[Weighted Histogram MSE]
    C2 --> H
    SAlpha --> H
    SBeta --> H
    SLow --> H
    H --> Amax[Per-layer optimal amax]

    Amax --> Cast[clamp(±amax) → cast to fp8_e4m3fn]
    KFP16 --> Save[(Safetensors)]
    Cast --> Save
```

### 3.1 DualMonitor (calibration time)

Forward hooks on every Linear/Conv2d collect, per layer:

- `output_sum`, `output_sq_sum` → running variance ⇒ **Sensitivity**.
- `channel_importance = mean(|input|, dim=batch+spatial)` → per-input-channel weight ⇒ **Importance**.

Hardening (V1.92):

- Outputs are clamped to ±65504 before squaring to prevent FP16 overflow.
- `math.isfinite()` guards drop NaN/Inf batches instead of poisoning the running stats.
- `get_sensitivity()` returns `0.0` when variance is non-finite.

### 3.2 Static Distribution Profile (weight-only, pre-pass)

A separate analysis pass over the FP16 state-dict produces a JSON profile keyed by weight name with three numbers per tensor:

| Field | Meaning | Why it matters in FP8 E4M3 |
|---|---|---|
| `kurtosis` | 4th-moment heaviness of the weight distribution | High kurtosis = a thin spike + long tail; a flat amax cannot resolve both. |
| `outlier_ratio` | `abs_max / std` | High ratio = the bulk of values is much smaller than the extremes; clipping to `abs_max` crushes the bulk. |
| `abs_max` | Largest absolute weight | Beyond ~20 the safe E4M3 representation budget is gone even with optimal clip. |

For Z Image, this profile is **mandatory**; if it is missing, the quantize script auto-invokes `analysis/analyze_zib_distribution.py`. For SDXL/Flux it is optional (used only when present, otherwise on-the-fly stats are computed).

### 3.3 Hard VETO

Layers whose static profile satisfies any of

```
kurtosis      > 20
outlier_ratio > 40
abs_max       > 20
```

are **force-promoted to FP16 before sensitivity selection ever runs**. Rationale: these distributions are mathematically unrepresentable in unscaled FP8 — no amax can save them. By taking them out *first*, the Dynamic Sensitivity pool is no longer dominated by a handful of pathological layers (the V1.9 → V1.92 fix that pushed Z Image Turbo SSIM from 0.59 to its current level).

### 3.4 Dynamic Sensitivity Selection

After VETO, the remaining layers form the Dynamic pool. They are ranked by a continuous profile score

```
profile_score = kurtosis + 2.0 * outlier_ratio + 0.5 * abs_max
```

(falling back to DualMonitor sensitivity when no profile exists). The top `keep_ratio` are added to the FP16 set.

```
final_FP16 = VETO_layers ∪ top_keep_ratio_dynamic   (no overlap)
```

`keep_ratio` defaults are 0.10–0.25 depending on family.

### 3.5 Per-layer Adaptive `search_low`

Instead of a global `search_range=(0.55, 1.0)` for amax, V1.92 / Flux v1.6 derive the lower bound **per layer** from the same profile:

```
k_penalty = min(kurtosis      / 100, 0.49)
o_penalty = min(outlier_ratio /  60, 0.49)
search_low = clip(0.50 + max(k_penalty, o_penalty), 0.50, 0.99)
```

Result: a calm layer is allowed to clip aggressively (search_low ≈ 0.50, finer center quantization), while a borderline-VETO layer is forced to stay close to its native amax (search_low ≈ 0.95, preserving outliers).

### 3.6 Importance backends

#### V1 / Fast — channel importance (SDXL, Flux)

```
importance_c = mean(|X|, dim=batch+spatial)        # per input channel
H(b)         = Σ importance_c · 1[bin(|w|) = b]    # weighted histogram
```

Cheap, calibration-driven, drop-in replacement for unweighted histograms.

#### V4 — SVD Leverage × RMS Magnitude Hybrid (Z Image)

For 2D weight `W = U Σ V^T`:

```
L(i,j) = (U_i · σ)^2 · (V_j)^2          # SVD leverage (structural)
M(i,j) = W_ij^2                          # RMS magnitude (energy)
Score(i,j) = α · L/‖L‖₂ + β · M/‖M‖₂     # Blended, L2-normalized
```

`α` and `β` are *not* hardcoded. They are derived from the model's **average kurtosis**:

```
k_factor = min(avg_kurtosis / 50, 0.30)
α = clip(0.50 + k_factor, 0.50, 0.80)    # SVD weight grows with global kurtosis
β = 1.0 − α                               # RMS weight shrinks accordingly
```

Reasoning: heavier-tailed models benefit from protecting the **principal subspace** (SVD); flatter models benefit from preserving raw **energy** (RMS). Z Image's `compute_optimal_amax(..., use_svd_leverage=True)` builds this matrix once and feeds it as per-element importance into the same weighted histogram MSE search as the simpler families.

Full mathematical derivation (σ²-weighted bilateral leverage, L2 normalization, per-element histogram), line-by-line walkthrough of `compute_hybrid_leverage_scores`, and the V1.5 → V1.9 → V1.92 failure history that motivated the hybrid model are documented separately in [HSWQ V4 SVD-RMS — Technical Guide](HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md).

### 3.7 Rigorous FP8 Grid Simulation

All amax search candidates are evaluated against the **physical** FP8 E4M3 grid:

```python
all_bytes = torch.arange(256, dtype=torch.uint8, device=device)
grid      = all_bytes.view(torch.float8_e4m3fn).float()
grid      = grid[~grid.isnan()]
```

Rounding to the nearest grid point uses `torch.searchsorted` on the sorted positive grid (Fast / V4): O(N log G) instead of O(N · G). The histogram and bin centers are kept in **float64**; only the per-candidate dequantization step touches float32. This guarantees that the optimizer's MSE matches what the runtime FP8 cast actually produces — no theoretical-formula drift.

### 3.8 Weighted Histogram MSE Search

For each candidate amax `Δ`:

```
J(Δ) = Σ_i H(i) · ( q(x_i, Δ) − x_i )²       # weighted sum over histogram bins
Δ*   = argmin_Δ  J(Δ)
```

`Δ` is searched over `num_candidates` linspace points in `[search_low, 1.0] · max(|W|)`, then refined for `refinement_iterations` rounds (Z Image: 1000 candidates × 10 refinements × 8192 bins). Tie-breaking between identical FP8 grid neighbors does not affect the chosen `Δ*` in practice.

---

## 4. Modes

| Mode | `scaled` | Output format | Loader required | Status |
|---|---|---|---|---|
| **V1 Standard Compatible** | `False` | Plain FP8 E4M3 weights, FP16-mixed for kept layers | None — works in any FP8 loader (ComfyUI, Diffusers) | **Production**. All HF releases use this. |
| **V2 High-Performance Scaled** | `True` | FP8 weights + `.scale` metadata in Safetensors | Custom `HSWQLoader` / `HSWQLinear` / `HSWQConv2d` | Algorithm in place; **not usable until a dedicated loader ships**. |

V1 optimizes only the clipping threshold (no per-tensor scale). V2 would additionally pack `S = amax/448` into Safetensors and dequantize on the fly. Until the loader exists, V1 is the only deployable mode.

---

## 5. Recommended Parameters

| Parameter | SDXL v1.3 | Flux.1-dev v1.6 | Z Image Turbo/Base v1.92 |
|---|---|---|---|
| `samples` | 32 | 32 | 256 (default in script) |
| `steps` | 20–25 | 20–25 | 20 |
| `keep_ratio` | 0.10 (often enough); 0.25 for safety | 0.15–0.25 | 0.25 (VETO is on top of this) |
| `latent` | 128 | 128 | 128 |
| Histogram | Fast | Fast | V4 (SVD+RMS hybrid) |
| Profile | optional | optional | **mandatory** (auto-generated) |
| Adaptive `search_low` | n/a | yes | yes |
| Hard VETO | n/a | n/a | yes |

For SDXL, 5–10 % FP16 retention combined with the Fast weighted histogram already reaches SSIM ≈ 0.94–0.99 against the FP16 reference (see `test/benchmark_test.md`). For Z Image, the VETO + V4 stack is what unlocks usable SSIM on extreme-distribution checkpoints (e.g. JANKU-trained Noobai derivatives).

---

## 6. Benchmark Results (reference)

See per-family benchmark documents for exact numbers; this is the high-level picture.

| Variant | SSIM (typical) | File size vs FP16 | Compatibility |
|---|---|---|---|
| Original FP16 | 1.0000 | 100 % | High |
| Naive FP8 cast | 0.75 – 0.93 | 50 % | High |
| **HSWQ V1 (SDXL Fast)** | **0.94 – 0.99** | 60–70 % (FP16 mixed) | **High — standard FP8 loader** |
| **HSWQ V1 (Flux Fast + Adaptive)** | **0.93 – 0.99** | 60–70 % | **High** |
| **HSWQ V1 (Z Image V4 + VETO)** | **0.86 – 0.97** on extreme checkpoints, **0.94+** on typical | 60–70 % | **High** |
| HSWQ V2 (Scaled) | — (not measurable yet) | 60–70 % | Custom loader required |

Detailed per-model tables:

- SDXL: [`test/benchmark_test.md`](../test/benchmark_test.md)
- Z Image: [`test/benchmark_zit.md`](../test/benchmark_zit.md)
- Flux: [`md/Flux_Benchmark_MSE_SSIM.md`](Flux_Benchmark_MSE_SSIM.md)

---

## 7. Related Documents

- [Dual Monitor System — Technical Guide](Dual_Monitor_System_Technical_Guide.md)
- [Weighted Histogram MSE — Technical Guide](Weighted_Histogram_MSE_Technical_Guide.md)
- **[HSWQ V4 SVD-RMS — Technical Guide](HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md)** — Full V4 optimizer reference: SVD leverage derivation, RMS magnitude, hybrid blending, line-by-line `compute_hybrid_leverage_scores`, integration with the V1.92 pipeline.
- [SDXL V1.3 + Histogram Fast — Full Explanation](SDXL_V1.3_and_Histogram_Fast_Explanation.md)
- [Adaptive Search Range — Technical Guide](Adaptive_Search_Range_Technical_Guide.md)
- [Flux v1.6 Adaptive Search Range](Flux1_v1.6_Adaptive_Search_Range_Explanation.md)
- [Z Image V1.5 — Latent and Mixed-Precision Calibration](ZI_V1.5_Latent_and_MixedPrecision_Calibration.md)
- [Z Image V1.9 → V1.92 Changes (VETO + V4 Hybrid)](V1.9_to_V1.92_Changes.md)
- [How to quantize SDXL](How%20to%20quantize%20SDXL.md) / [How to quantize Z Image](How%20to%20quantize%20Z%20Image.md)
