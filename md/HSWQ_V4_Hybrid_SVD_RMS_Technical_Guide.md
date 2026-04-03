# HSWQ V4: SVD–Magnitude Hybrid Weighted Histogram MSE Optimizer — Technical Guide

**Source:** `histogram/weighted_histogram_mse_v4.py`

This document provides a complete mathematical and implementation-level description of the HSWQ Weighted Histogram MSE Optimizer V4, which determines the optimal per-layer clipping threshold \(\Delta^*\) (amax) for FP8 E4M3 post-training quantization. V4 introduces a **hybrid per-element importance model** that blends Singular Value Decomposition (SVD) structural leverage with Root Mean Square (RMS) magnitude preservation, superseding the per-channel importance model of earlier versions.

---

## 0. HSWQ Fundamental Design Philosophy

### 0.1 The Problem: Why Naive FP8 Cast Fails

Standard FP8 quantization (Naive Cast) applies a uniform conversion to all layers: every weight tensor is cast from FP16 to FP8 E4M3 using a single global strategy (e.g., min-max clipping). This ignores a critical reality of neural networks:

- **Not all layers are equally sensitive.** Some layers (e.g., attention projections, modulation layers) produce output distributions where even small perturbations cause catastrophic degradation in generated image quality. Other layers (e.g., mid-network feed-forward blocks) are robust to quantization noise.
- **Not all channels within a layer are equally important.** Input channels with high activation magnitudes contribute disproportionately to the forward-pass computation. Quantization error on high-activation channels propagates with proportionally larger impact.

Naive Cast achieves SSIM of only 0.75–0.93 against the FP16 original, because it wastes precision on robust layers while insufficiently protecting sensitive ones.

### 0.2 The Name: Hybrid Sensitivity Weighted Quantization

Each word in **HSWQ** corresponds to a core design choice:

| Word | Meaning | Implementation |
|------|---------|----------------|
| **Hybrid** | Combines two orthogonal monitoring perspectives — **sensitivity** (output-side) and **importance** (input-side) — in a single calibration pass. Neither alone is sufficient. | Dual Monitor System: one hook per layer collects both metrics simultaneously. |
| **Sensitivity** | Output variance \(\mathrm{Var}(Y)\) identifies which layers, if corrupted, would hurt image quality the most. | Layer Selection: top 5–25% by sensitivity are kept in FP16; the rest are quantized. |
| **Weighted** | Per-element importance weights \(\alpha_{m,n}\) control how quantization error is distributed. The MSE objective is not uniform — it penalizes errors on important elements more heavily. | Weighted Histogram MSE: importance-weighted histogram \(H(i)\), not frequency histogram, drives the amax search. |
| **Quantization** | FP8 E4M3 post-training quantization with a physically accurate grid simulation. No retraining. | FP8E4M3Quantizer: grid built from all 256 byte patterns, ensuring MSE matches runtime behavior exactly. |

### 0.3 Three-Component Architecture

HSWQ operates in three stages:

```
Stage 1: Dual Monitor (Calibration)
    ├── Sensitivity Monitor → output variance Var(Y) per layer
    └── Importance Monitor  → per-channel input activation I_c per layer

Stage 2: Layer Selection
    ├── Sort layers by Sensitivity (descending)
    ├── Top keep_ratio (5–25%) → Keep in FP16 (no quantization)
    └── Remaining layers → Proceed to Stage 3

Stage 3: Weighted MSE Optimization (per FP8 layer)
    ├── Build importance-weighted histogram H(i) from weights + importance
    ├── Search clipping threshold Δ* that minimizes Σ H(i)·(q(x_i,Δ)−x_i)²
    └── Clip and cast to FP8 E4M3
```

**Stage 1** answers the question *"which layers matter most?"* (sensitivity) and *"which channels within each layer matter most?"* (importance).

**Stage 2** uses only sensitivity. It is a binary decision: FP16 or FP8. This prevents catastrophic quality loss on the most fragile layers.

**Stage 3** uses only importance. For the layers that will be quantized, it determines *how* to quantize — specifically, where to set the clipping threshold so that the importance-weighted quantization error is minimized.

This separation of concerns — **what to protect** vs. **how to quantize** — is the core architectural insight of HSWQ.

### 0.4 V1 vs V2: Two Quantization Modes

| | V1: Standard Compatible | V2: High Performance Scaled |
|---|---|---|
| **Mechanism** | Optimize clipping threshold \(\Delta\) only. No scale factor stored. | Scale weights to fill FP8 range; store inverse scale \(S = \Delta/448\). |
| **Quantize–dequantize** | \(q(x, \Delta) = \mathrm{round}_{\mathrm{FP8}}(\mathrm{clip}(x, -\Delta, \Delta))\) | \(q(x, \Delta) = \mathrm{round}_{\mathrm{FP8}}(x \cdot 448/\Delta) / (448/\Delta)\) |
| **Compatibility** | Any standard FP8 loader (ComfyUI, etc.) | Requires dedicated HSWQLoader (not available) |
| **Status** | **Active** — all production quantization uses V1. | Theoretical — currently unmeasurable. |

V1 is the only mode used in practice. The V4 optimizer supports both modes via the `scaled` parameter, but V1.92 invokes it with `scaled=False`.

### 0.5 Why Full SVD Was Necessary — The Z Image Failure History

The full SVD hybrid (V4) was not introduced out of theoretical interest. It was a direct response to the **catastrophic failure** of the original HSWQ approach when applied to Z Image (NextDiT architecture).

#### The SDXL Baseline: Per-Channel Importance Worked

For SDXL's UNet, the original three-stage HSWQ (V1.3) with per-channel 1D importance \(I_c\) from DualMonitor achieved SSIM 0.86–0.98. The UNet's weight distributions are relatively well-behaved: layers have moderate kurtosis, outlier ratios are manageable, and the DualMonitor's output variance reliably identifies the most sensitive layers.

#### Z Image V1.5: The Same Approach Applied

V1.5 applied the identical Dual Monitor + per-channel importance framework to Z Image (NextDiT), adding only latent resolution control and mixed-precision calibration. The architecture difference was treated as irrelevant.

#### Z Image V1.9: SSIM 0.59 — Catastrophic Failure

V1.9 introduced the distribution profile (kurtosis, outlier_ratio, abs_max) and the V4 SVD hybrid optimizer. Despite these additions, the result was **SSIM 0.59** — worse than naive FP8 cast. The root causes were all in **Stage 2 (Layer Selection)**, not Stage 3:

| Failure | Mechanism | Impact |
|---------|-----------|--------|
| **DualMonitor NaN contamination** | Squaring large FP16 outputs overflowed to NaN/Inf; these propagated into sensitivity scores. | Sensitivity ranking was corrupted — layers were selected essentially at random. |
| **w2 variance domination** | `feed_forward.w2` layers produce outputs with enormous variance (due to activation functions), consuming all FP16 slots. | Structurally critical layers (`attention.out`, `adaLN_modulation`) were forced into FP8, destroying image quality. |
| **Profile key mismatch** | Profile keys retained prefixes (`model.diffusion_model.layers.0...`) while module names were stripped (`layers.0...`). Profile data was effectively unused. | The distribution-aware search range and VETO logic had no effect — the profile was dead code. |

**Critical insight:** V1.9 already had the V4 SVD optimizer (Stage 3), but it couldn't help because **Stage 2 was broken**. The wrong layers were being kept in FP16, so even perfect per-element importance in Stage 3 couldn't compensate for quantizing layers that should never have been quantized.

#### Z Image V1.92: Fixing Stage 2

V1.92 repaired Stage 2 with four targeted fixes:

1. **DualMonitor overflow protection** — clamp outputs to \(\pm 65504\), skip NaN/Inf batches, return 0 for non-finite variance.
2. **Hard VETO** — layers with kurtosis >20, outlier_ratio >40, or abs_max >20 are **unconditionally** kept in FP16 (34 layers for a typical Z Image model). These layers have weight distributions that are mathematically incompatible with FP8 E4M3 — no clipping threshold can preserve their information.
3. **Profile key normalization** — automatic prefix detection and stripping, so profile data actually reaches the selection logic.
4. **Profile-based composite score** — FP16 selection uses \(k + 2o + 0.5m\) instead of DualMonitor output variance. This is scale-independent and not susceptible to the w2 domination problem.

With Stage 2 fixed, the V4 SVD optimizer in Stage 3 could finally operate on the correct set of layers.

#### Why Per-Channel Importance Was Insufficient for NextDiT

Even with Stage 2 fixed, the original per-channel 1D importance \(I_c\) from DualMonitor was inadequate for NextDiT's weight structure:

- **NextDiT weights are internally heterogeneous.** Unlike SDXL's UNet, NextDiT layers (especially `attention.qkv`, `feed_forward.w1/w3`) have highly non-uniform internal distributions — some rows/columns participate in dominant singular modes while others carry near-zero signal.
- **Per-channel averaging destroys within-layer structure.** \(I_c = \mathrm{mean}(|X_c|)\) treats all output dimensions identically. It cannot distinguish between a weight element that is structurally critical (high SVD leverage) and one that is noise-level, as long as they are in the same input channel.
- **The SVD reveals structural criticality.** Full SVD decomposition identifies which elements participate in the principal modes (\(\sigma_k^2\)-weighted leverage). The RMS magnitude adds energetic significance. Together, they provide per-element importance that captures NextDiT's complex internal structure.

#### The Full Picture: SVD Was Necessary But Not Sufficient

| Component | What it solves | Alone sufficient? |
|-----------|---------------|-------------------|
| V4 SVD hybrid (Stage 3) | Per-element importance for heterogeneous weight structure | No — useless if wrong layers are quantized |
| Hard VETO (Stage 2) | Excludes mathematically unquantizable layers | No — remaining layers still need intelligent quantization |
| Profile-based selection (Stage 2) | Scale-independent FP16 layer ranking | No — doesn't address within-layer quantization quality |
| DualMonitor NaN fix (Stage 1) | Prevents calibration corruption | No — fixes the tool, not the strategy |

All four components are necessary; none is sufficient alone. The V1.92 system works because **Stage 2 correctly identifies what to protect** and **Stage 3 (V4) correctly determines how to quantize the rest**.

#### Empirical Outcome

Preliminary benchmarks on Z Image Turbo models show that the integrated V1.92 system (all four components working together) achieves SSIM scores substantially higher than both Naive FP8 and officially distributed FP8 baselines, with keep ratios as low as 5–10%. The improvement from V1.9 (SSIM ~0.59) to V1.92 confirms that the four-component design is not just theoretically motivated but empirically necessary: removing any one component collapses the quality back toward V1.9 levels. Full benchmark results are being collected and will be published separately.

### 0.6 V4's Position Within This Framework

V4 replaces the **importance model** in Stage 3. It does not modify Stage 1 or Stage 2:

| Version | Importance granularity | Importance source |
|---------|----------------------|-------------------|
| V1 (original) | Per-channel 1D (\(I_c\)) | Calibration activations only |
| **V4 (current)** | **Per-element 2D** (\(S_{m,n}\)) | SVD structural leverage + RMS magnitude + (optional) calibration |

V4 provides a fundamentally richer importance signal to the weighted histogram, capturing not just "which channels are active" but "which individual weight elements are structurally and energetically critical." This matters most for architectures like NextDiT where the within-layer weight structure is highly non-uniform.

---

## 1. V4 Optimizer — Technical Details

### 1.1 Problem Statement

Given a trained weight matrix \(\mathbf{W} \in \mathbb{R}^{M \times N}\) and a quantization function \(q(\cdot, \Delta)\) that maps real values to the FP8 E4M3 representable set, find the clipping threshold \(\Delta^*\) that minimizes the **importance-weighted mean squared error**:

$$\Delta^* = \arg\min_{\Delta} \sum_{i=0}^{B-1} H(i) \cdot \bigl(q(x_i,\, \Delta) - x_i\bigr)^2$$

where \(B\) is the number of histogram bins, \(H(i)\) is the normalized weighted histogram at bin \(i\), \(x_i\) is the bin center, and \(q(x, \Delta)\) is the quantize–dequantize operator.

The critical design choice lies in how the per-element importance weights \(\alpha_{m,n}\) — which determine \(H(i)\) — are constructed. V4 formulates \(\alpha_{m,n}\) as a **hybrid of two orthogonal objectives**, each capturing a distinct aspect of weight significance.

### 1.2 V4 Hybrid Importance Model

From the module docstring (lines 1–14):

```1:14:histogram/weighted_histogram_mse_v4.py
"""
HSWQ Weighted Histogram MSE Optimizer V4 (SVD & RMS Magnitude Hybrid Blended Edition)
=====================================================================================

Compatible with standard environments (no custom loaders). Blends two orthogonal
objectives with L2 normalization: (1) minimization of projection error onto
principal components (SVD), (2) preservation of constant-bias absolute energy (RMS Magnitude).

Hybrid model:
    L(i,j) = (U_i_norm^2) * (V_j_norm^2)  # SVD Leverage
    M(i,j) = X_ij^2                       # RMS Magnitude
    Score(i,j) = alpha * (L / ||L||_2) + beta * (M / ||M||_2)

Assigns per-element (2D/4D) importance and builds weighted histogram MSE at high precision.
"""
```

The two objectives are:

1. **SVD Leverage** \(\mathbf{L}\): Measures each element's contribution to the principal subspace of \(\mathbf{W}\). Elements with high leverage participate strongly in the dominant singular modes; quantization error on these elements disproportionately distorts the low-rank structure that carries most of the layer's representational capacity.

2. **RMS Magnitude** \(\mathbf{M}\): Measures each element's squared absolute value. Large-magnitude weights carry more energy in the forward pass; errors on them propagate with proportionally larger impact on downstream activations.

These two matrices are L2-normalized to equalize their scales, then blended with coefficients \((\alpha, \beta)\) into a single per-element importance map \(\mathbf{S} \in \mathbb{R}^{M \times N}\).

### 1.3 Notation

| Symbol | Meaning |
|--------|---------|
| \(\mathbf{W} \in \mathbb{R}^{M \times N}\) | Weight matrix (2D view; Conv2d is reshaped to \((O, I \cdot K^2)\)). |
| \(\mathbf{U} \in \mathbb{R}^{M \times K}\), \(\mathbf{S} \in \mathbb{R}^{K}\), \(\mathbf{V}_h \in \mathbb{R}^{K \times N}\) | Compact SVD: \(\mathbf{W} = \mathbf{U}\,\mathrm{diag}(\boldsymbol{\sigma})\,\mathbf{V}_h\), \(K = \min(M, N)\). |
| \(\sigma_k\) | \(k\)-th singular value (\(k = 1, \ldots, K\)). |
| \(L_{m,n}\) | SVD leverage score for element \((m, n)\). |
| \(M_{m,n}\) | RMS magnitude score for element \((m, n)\). |
| \(\alpha, \beta\) | Blend coefficients (\(\alpha + \beta = 1\) in practice). |
| \(S_{m,n}\) | Final hybrid importance for element \((m, n)\). |
| \(I_c\) | 1D channel importance from calibration (DualMonitor). Optional. |
| \(\Delta\) (amax) | Clipping threshold — the single parameter optimized. |
| \(B\) | Number of histogram bins. |
| \(H(i)\) | Normalized weighted histogram at bin \(i\); \(\sum_i H(i) = 1\). |
| \(q(x, \Delta)\) | Quantize–dequantize function (FP8 E4M3 grid rounding). |
| \(x_i\) | Bin center for bin \(i\). |

### 1.4 Component Summary

| Component | Responsibility |
|-----------|----------------|
| **`compute_hybrid_leverage_scores`** | Compute per-element hybrid importance \(\mathbf{S}\) via full SVD and RMS blending. |
| **`FP8E4M3Quantizer`** | Physical FP8 E4M3 grid; quantize–dequantize \(q(x, \Delta)\). |
| **`WeightedHistogram`** | Build \(H(i)\) from \(\mathbf{W}\) and \(\mathbf{S}\); normalize; provide \(x_i\). |
| **`MSEOptimizer`** | Evaluate \(\sum_i H(i)(q(x_i, \Delta) - x_i)^2\); search \(\Delta\) via iterative refinement. |
| **`HSWQWeightedHistogramOptimizerV4`** | Compose: hybrid scores → histogram → MSE search → optimal \(\Delta^*\). |

### 1.5 V1 → V4 Evolution

| Aspect | V1 (Baseline) | V4 (Hybrid SVD–RMS) |
|--------|---------------|---------------------|
| Importance granularity | Per-channel 1D (\(I_c\)) | Per-element 2D (\(S_{m,n}\)) |
| Importance source | Calibration activations only | SVD structure + magnitude + (optional) calibration |
| Default bins | 4096 | 8192 |
| Default candidates | 200 | 1000 |
| Default refinement iterations | 3 | 10 |
| Blend parameters | N/A | \(\alpha\), \(\beta\) (autonomous from model profile) |

---

## 2. FP8 E4M3 Quantizer

### 2.1 Specification

FP8 E4M3 (IEEE-draft compliant, PyTorch `float8_e4m3fn`):

| Field | Bits | Details |
|-------|------|---------|
| Sign | 1 | 0 = positive, 1 = negative |
| Exponent | 4 | Bias = 7; range \(e \in [0, 15]\) |
| Mantissa | 3 | 8 representable fractions per exponent |
| Normalized range | — | \(2^{e-7}(1 + m/8)\) for \(e \in [1, 15]\), \(m \in [0, 7]\) |
| Denormalized range | — | \(2^{-6}(m/8)\) for \(e = 0\), \(m \in [1, 7]\) |
| Special values | — | NaN: `0x7F`, `0xFF`; \(\pm 0\); no \(\pm\infty\) |
| Representable range | — | \(\pm [2^{-9}, 448]\) (including denormals) |

### 2.2 Grid Construction — `_build_fp8_grid` (lines 35–52)

The grid \(\mathcal{G}\) is the set of all distinct positive float32 values obtainable by interpreting each byte pattern \(b \in [0, 255]\) as `float8_e4m3fn`:

$$\mathcal{G}^+ = \bigl\{ \mathrm{float32}\bigl(\mathrm{view}_{\mathrm{E4M3}}(b)\bigr) \;\big|\; b \in [0, 255],\; \text{not NaN},\; \geq 0 \bigr\}$$

```35:52:histogram/weighted_histogram_mse_v4.py
    def _build_fp8_grid(self):
        """Build full representable positive grid for FP8 E4M3 (PyTorch native behavior)."""
        all_bytes = torch.arange(256, dtype=torch.uint8, device=self.device)
        fp8_vals = all_bytes.view(torch.float8_e4m3fn)
        f32_vals = fp8_vals.float()
        
        valid_mask = ~f32_vals.isnan()
        valid_vals = f32_vals[valid_mask]
        
        pos_vals = valid_vals[valid_vals >= 0]
        unique_vals = pos_vals.unique().sort().values
        
        self._positive_grid = unique_vals
        
        negative_values = -unique_vals[unique_vals > 0].flip(0)
        self._full_grid = torch.cat([negative_values, unique_vals])
        
        self.max_representable = self._positive_grid.max().item()  # 448.0
```

The full grid is symmetric: \(\mathcal{G} = \{-g : g \in \mathcal{G}^+, g > 0\} \cup \mathcal{G}^+\). The maximum representable value is \(\mathcal{G}_{\max} = 448.0\).

This construction guarantees that the rounding simulation exactly matches PyTorch's native `float8_e4m3fn` cast behavior, since the grid is derived from the same byte-level representation.

### 2.3 Quantize–Dequantize — `quantize_dequantize` (lines 54–70)

**Signature:** `quantize_dequantize(values, amax, scaled)` → implements \(q(x, \Delta)\).

Two modes are supported:

**Mode 1 — Scaled (\(\texttt{scaled=True}\)):**

$$s = \frac{\mathcal{G}_{\max}}{\Delta} = \frac{448}{\Delta}$$

$$q(x, \Delta) = \frac{\mathrm{round}_{\mathrm{FP8}}\bigl(\mathrm{clip}(x \cdot s,\; -448,\; 448)\bigr)}{s}$$

This maps the full \([-\Delta, \Delta]\) range onto \([-448, 448]\), quantizes in the scaled domain, then maps back. The effective quantization step size is proportional to \(\Delta / 448\).

```59:65:histogram/weighted_histogram_mse_v4.py
        if scaled:
            scale = self.max_representable / amax
            scaled_vals = values * scale
            scaled_vals = scaled_vals.clamp(-self.max_representable, self.max_representable)
            quantized = self._round_to_fp8_grid(scaled_vals)
            dequantized = quantized / scale
            return dequantized
```

**Mode 2 — Non-scaled (\(\texttt{scaled=False}\)):**

$$q(x, \Delta) = \mathrm{round}_{\mathrm{FP8}}\bigl(\mathrm{clip}(\mathrm{clip}(x,\, -\Delta,\, \Delta),\; -448,\; 448)\bigr)$$

Values are clipped to \([-\Delta, \Delta]\) (then additionally capped at \(\pm 448\)), and rounded directly to the nearest FP8 grid point without rescaling.

```66:70:histogram/weighted_histogram_mse_v4.py
        else:
            clipped = values.clamp(-amax, amax)
            clipped = clipped.clamp(-self.max_representable, self.max_representable)
            dequantized = self._round_to_fp8_grid(clipped)
            return dequantized
```

Guard: if \(\Delta \leq 0\), returns a zero tensor (lines 56–57).

V1.92 quantizer invokes this with `scaled=False`, meaning weights are clipped and stored directly in the FP8 native range.

### 2.4 Nearest-Grid Rounding — `_round_to_fp8_grid` (lines 72–88)

**Definition:** For each value \(v\), decompose into sign and magnitude, then find the nearest positive grid point:

$$\mathrm{round}_{\mathrm{FP8}}(v) = \mathrm{sign}(v) \cdot \arg\min_{g \in \mathcal{G}^+} |g - |v||$$

```72:88:histogram/weighted_histogram_mse_v4.py
    def _round_to_fp8_grid(self, values: torch.Tensor) -> torch.Tensor:
        """Round values to nearest FP8 grid points."""
        signs = torch.sign(values)
        abs_values = values.abs()
        
        abs_flat = abs_values.reshape(-1)
        batch_size = 10000
        result = torch.zeros_like(abs_flat)
        
        for i in range(0, len(abs_flat), batch_size):
            batch = abs_flat[i:i+batch_size]
            distances = (batch.unsqueeze(1) - self._positive_grid.unsqueeze(0)).abs()
            nearest_indices = distances.argmin(dim=1)
            result[i:i+batch_size] = self._positive_grid[nearest_indices]
        
        result = result.reshape(abs_values.shape)
        return result * signs
```

**Computational note:** For a batch of \(P\) values and a grid of \(G\) points, the distance matrix has shape \((P, G)\). Batching at 10000 elements bounds peak memory to \(\mathcal{O}(10^4 \times G)\). For FP8 E4M3, \(|\mathcal{G}^+| = 120\) (after deduplication), so each batch consumes approximately \(10^4 \times 120 \times 4 \approx 4.8\,\text{MB}\).

---

## 3. SVD Leverage Scores — Theoretical Foundation

### 3.1 Motivation

Consider a linear layer \(\mathbf{y} = \mathbf{W}\mathbf{x}\). The compact SVD gives \(\mathbf{W} = \mathbf{U}\,\mathrm{diag}(\boldsymbol{\sigma})\,\mathbf{V}_h\), where the singular values \(\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_K > 0\) represent the "importance" of each singular mode. A perturbation \(\mathbf{E}\) to \(\mathbf{W}\) causes an output perturbation:

$$\|\delta\mathbf{y}\|^2 = \|\mathbf{E}\mathbf{x}\|^2$$

When the perturbation is structured (as in quantization), elements \(W_{m,n}\) that participate strongly in the top singular modes contribute disproportionately to the output distortion. The **statistical leverage** of element \((m, n)\) quantifies this participation.

### 3.2 Classical Leverage in Linear Algebra

In the classical Hat matrix formulation for least-squares regression \(\mathbf{y} = \mathbf{X}\boldsymbol{\beta}\), the leverage of observation \(i\) is:

$$h_{ii} = \sum_{k=1}^{K} U_{ik}^2$$

where \(\mathbf{U}\) comes from the SVD of \(\mathbf{X}\). This measures how much observation \(i\) influences the fitted values.

### 3.3 Generalized Bilateral Leverage for Weight Matrices

HSWQ V4 extends this concept to **bilateral leverage** for a 2D weight matrix, incorporating singular value magnitudes:

$$L_{m,n} = \underbrace{\left(\sum_{k=1}^{K} U_{mk}^2 \, \sigma_k^2\right)}_{\text{row leverage } r_m} \cdot \underbrace{\left(\sum_{k=1}^{K} V_{nk}^2 \, \sigma_k^2\right)}_{\text{column leverage } c_n}$$

where \(V_{nk} = (V_h^T)_{nk}\) (columns of \(\mathbf{V}_h^T\)).

**Interpretation:**

- **Row leverage** \(r_m = \sum_k U_{mk}^2 \sigma_k^2\): Measures how strongly output dimension \(m\) participates in the principal modes, weighted by mode strength \(\sigma_k^2\). High \(r_m\) means row \(m\) carries critical information across the dominant singular components.

- **Column leverage** \(c_n = \sum_k V_{nk}^2 \sigma_k^2\): Measures how strongly input dimension \(n\) contributes to the principal modes. High \(c_n\) means input channel \(n\) feeds into the dominant components.

- **Element leverage** \(L_{m,n} = r_m \cdot c_n\): The outer product of row and column leverages. An element \(W_{m,n}\) has high leverage when **both** its row and column participate in important singular modes. Quantization error at such elements distorts the most significant directions of the linear map.

### 3.4 Why \(\sigma_k^2\) Weighting?

Without weighting (\(\sigma_k^2 = 1\)), all singular modes contribute equally to leverage, which treats noise-level singular components as important as dominant ones. With \(\sigma_k^2\) weighting:

- Modes with large singular values dominate the leverage computation.
- Near-zero singular modes (numerical noise) have negligible influence.
- The leverage score naturally concentrates on the "structurally important" subspace.

This is mathematically equivalent to computing leverage on the weighted matrix \(\mathbf{U} \cdot \mathrm{diag}(\boldsymbol{\sigma})\), i.e., the left singular vectors scaled by their corresponding singular values.

### 3.5 Compact Matrix Form

In matrix notation, define:

$$\mathbf{r} = (\mathbf{U} \odot \mathbf{U}) \, \boldsymbol{\sigma}^2 \in \mathbb{R}^{M \times 1}, \qquad \mathbf{c} = (\mathbf{V}_h^T \odot \mathbf{V}_h^T) \, \boldsymbol{\sigma}^2 \in \mathbb{R}^{N \times 1}$$

where \(\odot\) denotes element-wise (Hadamard) product and \(\boldsymbol{\sigma}^2 = [\sigma_1^2, \ldots, \sigma_K^2]^T\). Then:

$$\mathbf{L} = \mathbf{r} \, \mathbf{c}^T \in \mathbb{R}^{M \times N}$$

This is a rank-1 outer product, making the leverage map inherently low-rank regardless of the weight matrix rank.

---

## 4. RMS Magnitude Scores

### 4.1 Definition

$$M_{m,n} = W_{m,n}^2$$

The squared magnitude of each weight element. This is equivalent to the contribution of element \((m, n)\) to the Frobenius norm:

$$\|\mathbf{W}\|_F^2 = \sum_{m,n} M_{m,n}$$

### 4.2 Motivation

Magnitude-based importance captures a complementary signal to SVD leverage:

- **SVD leverage is structural**: It measures participation in principal modes regardless of actual element values. An element can have high leverage but small magnitude (structurally important position, small value), or low leverage but large magnitude (isolated outlier in a non-principal direction).

- **RMS magnitude is energetic**: It directly measures the element's contribution to the layer's output energy. In the forward pass \(\mathbf{y} = \mathbf{W}\mathbf{x}\), the expected squared output is proportional to \(\sum_{m,n} W_{m,n}^2 \, \mathbb{E}[x_n^2]\). Elements with large \(W_{m,n}^2\) contribute more to the signal power.

The combination of both ensures that neither structurally critical low-magnitude elements nor energetically significant outliers are neglected during quantization.

---

## 5. Hybrid Importance Computation — `compute_hybrid_leverage_scores`

### 5.1 Signature and Parameters (line 228)

```228:231:histogram/weighted_histogram_mse_v4.py
def compute_hybrid_leverage_scores(weight: torch.Tensor, alpha: float = 0.7, beta: float = 0.3, top_p: float = 1.0, min_k: int = 1, max_k: int = 4096) -> torch.Tensor:
    """
    Outputs blended importance matrix: SVD-based structural leverage and RMS magnitude,
    each L2-normalized and combined with weights (alpha, beta).
    """
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `weight` | — | Input weight tensor (2D, 4D, or other). |
| `alpha` | 0.7 | Blend weight for SVD leverage. |
| `beta` | 0.3 | Blend weight for RMS magnitude. |
| `top_p` | 1.0 | Fraction of singular values to retain (1.0 = full SVD). |
| `min_k` / `max_k` | 1 / 4096 | Bounds on number of singular components. |

In V1.92, \(\alpha\) and \(\beta\) are **not hardcoded** but autonomously derived from the global model kurtosis profile (see §8).

### 5.2 Preprocessing (lines 233–250)

```236:250:histogram/weighted_histogram_mse_v4.py
    # Flatten to 2D
    if weight.ndim > 2:
        w_float = weight.detach().float().view(weight.shape[0], -1)
    elif weight.ndim == 2:
        w_float = weight.detach().float()
    else:
        return torch.ones_like(weight, dtype=torch.float32)

    if torch.all(w_float == 0):
        return torch.ones_like(weight, dtype=torch.float32)

    M, N = w_float.shape
    max_rank = min(M, N)
    k = min(max_k, max(min_k, int(math.floor(top_p * max_rank))))
    k = min(k, max_rank)
```

- **4D tensors** (Conv2d with shape \((O, I, K_h, K_w)\)) are reshaped to \((O, I \cdot K_h \cdot K_w)\).
- **1D tensors** (bias, norms) receive uniform importance \(\mathbf{1}\).
- **Zero tensors** receive uniform importance \(\mathbf{1}\) (degenerate case).
- The effective rank \(k = \min(\texttt{max\_k}, \lfloor \texttt{top\_p} \cdot \min(M, N) \rfloor)\). With defaults \(\texttt{top\_p}=1.0, \texttt{max\_k}=4096\), this is simply \(k = \min(M, N)\): **full SVD**.

### 5.3 Step 1: Full SVD and Leverage Computation (lines 256–266)

```256:266:histogram/weighted_histogram_mse_v4.py
    # --- 1. SVD Leverage (full: σ^2 weighted) ---
    # Use full SVD (linalg.svd), not low-rank approximation
    U, S, Vh = torch.linalg.svd(w_float, full_matrices=False)
    
    # Full leverage formula with σ^2 weighting
    # weighted_U_k = U * S
    # leverage = (U_ik^2 * σ_k^2) * (V_jk^2)
    S_sq = S ** 2
    row_scores = (U ** 2) @ S_sq.unsqueeze(1)    # (M, k) @ (k, 1) -> (M, 1)
    col_scores = (Vh.T ** 2) @ S_sq.unsqueeze(1) # (N, k) @ (k, 1) -> (N, 1)
    leverage_2d = row_scores * col_scores.T      # (M, N)
```

**Mathematical correspondence:**

1. `torch.linalg.svd(w_float, full_matrices=False)` returns \(\mathbf{U} \in \mathbb{R}^{M \times K}\), \(\boldsymbol{\sigma} \in \mathbb{R}^{K}\), \(\mathbf{V}_h \in \mathbb{R}^{K \times N}\) with \(K = \min(M, N)\).

2. `S_sq` = \(\boldsymbol{\sigma}^2 \in \mathbb{R}^K\).

3. Row scores:

$$\mathbf{r} = (\mathbf{U} \odot \mathbf{U}) \, \boldsymbol{\sigma}^2 = \begin{bmatrix} \sum_{k} U_{1k}^2 \sigma_k^2 \\ \vdots \\ \sum_{k} U_{Mk}^2 \sigma_k^2 \end{bmatrix} \in \mathbb{R}^{M \times 1}$$

4. Column scores:

$$\mathbf{c} = (\mathbf{V}_h^T \odot \mathbf{V}_h^T) \, \boldsymbol{\sigma}^2 = \begin{bmatrix} \sum_{k} V_{h,k1}^2 \sigma_k^2 \\ \vdots \\ \sum_{k} V_{h,kN}^2 \sigma_k^2 \end{bmatrix} \in \mathbb{R}^{N \times 1}$$

5. Leverage matrix:

$$\mathbf{L} = \mathbf{r} \, \mathbf{c}^T \in \mathbb{R}^{M \times N}$$

**Complexity:** \(\mathcal{O}(M N K)\) for the SVD (dominant cost), plus \(\mathcal{O}(MK + NK)\) for the leverage computation. For a typical transformer Linear layer with \(M = N = 3840\), the SVD operates on a \(3840 \times 3840\) matrix.

### 5.4 Step 2: RMS Magnitude (lines 268–269)

```268:269:histogram/weighted_histogram_mse_v4.py
    # --- 2. RMS Magnitude ---
    magnitude_2d = w_float ** 2  # (M, N)
```

$$\mathbf{M} = \mathbf{W} \odot \mathbf{W} \in \mathbb{R}^{M \times N}$$

Element-wise squared weights. \(\mathcal{O}(MN)\).

### 5.5 Step 3: L2 Normalization (lines 271–277)

```271:277:histogram/weighted_histogram_mse_v4.py
    # --- 3. L2 normalize (equal impact per score matrix) ---
    lev_norm = torch.norm(leverage_2d, p=2)
    mag_norm = torch.norm(magnitude_2d, p=2)
    
    # Avoid division by zero
    if lev_norm > 0: leverage_2d = leverage_2d / lev_norm
    if mag_norm > 0: magnitude_2d = magnitude_2d / mag_norm
```

$$\hat{\mathbf{L}} = \frac{\mathbf{L}}{\|\mathbf{L}\|_2}, \qquad \hat{\mathbf{M}} = \frac{\mathbf{M}}{\|\mathbf{M}\|_2}$$

where \(\|\cdot\|_2\) is the Frobenius norm (treated as a flattened vector norm):

$$\|\mathbf{L}\|_2 = \sqrt{\sum_{m,n} L_{m,n}^2}$$

**Rationale:** Without normalization, the leverage matrix and magnitude matrix have entirely different scales. Leverage values are products of squared singular-value-weighted projections (can span many orders of magnitude), while magnitude values are raw squared weights. L2 normalization maps both matrices to the unit sphere \(\|\hat{\mathbf{L}}\|_2 = \|\hat{\mathbf{M}}\|_2 = 1\), ensuring that the blend coefficients \(\alpha, \beta\) control the relative contribution as intended.

### 5.6 Step 4: \(\alpha/\beta\) Blending (lines 279–280)

```279:280:histogram/weighted_histogram_mse_v4.py
    # --- 4. Alpha/Beta blend ---
    hybrid_importance = (alpha * leverage_2d) + (beta * magnitude_2d)
```

$$\mathbf{S}_{\mathrm{raw}} = \alpha \, \hat{\mathbf{L}} + \beta \, \hat{\mathbf{M}}$$

Since \(\|\hat{\mathbf{L}}\|_2 = \|\hat{\mathbf{M}}\|_2 = 1\), the blend coefficients directly control the relative influence:

- \(\alpha = 0.7, \beta = 0.3\): SVD structure dominates (70% weight).
- \(\alpha = \beta = 0.5\): Equal contribution.

In V1.92, \(\alpha\) is autonomously derived from the model's average kurtosis, ranging from 0.5 (low kurtosis, smooth distributions) to 0.8 (high kurtosis, heavy-tailed distributions where structural preservation is critical).

### 5.7 Step 5: Scale Normalization and Baseline (lines 282–289)

```282:289:histogram/weighted_histogram_mse_v4.py
    # --- 5. Histogram scale normalization ---
    # Scale so mean ~1.0 and histogram area matches weight count
    avg_score = hybrid_importance.mean()
    if avg_score > 0:
        hybrid_importance = hybrid_importance / avg_score

    # V2-style mild baseline (avoid 0-div and full collapse)
    hybrid_importance = 0.5 + 0.5 * hybrid_importance
```

**Two-stage normalization:**

**Stage A — Mean normalization:**

$$\mathbf{S}_{\mathrm{norm}} = \frac{\mathbf{S}_{\mathrm{raw}}}{\overline{S}_{\mathrm{raw}}}$$

where \(\overline{S}_{\mathrm{raw}} = \frac{1}{MN}\sum_{m,n} S_{\mathrm{raw},m,n}\). After this, \(\overline{S}_{\mathrm{norm}} = 1.0\), which ensures that the total weight in the histogram equals the element count (as if each element contributed weight 1 on average).

**Stage B — Baseline offset:**

$$S_{m,n} = 0.5 + 0.5 \cdot S_{\mathrm{norm},m,n}$$

This affine transform has two effects:

1. **Floor at 0.5:** No element has importance below 0.5, preventing complete neglect of any weight during histogram construction. Even structurally unimportant, low-magnitude elements receive a minimum weight of 0.5.

2. **Compression to \([0.5, \infty)\):** The dynamic range of importance is halved and shifted, preventing extreme concentration of histogram mass on a few elements.

After mean normalization, \(\overline{S}_{\mathrm{norm}} = 1.0\), so the final mean is \(0.5 + 0.5 \times 1.0 = 1.0\) — the histogram scale is preserved.

### 5.8 Return Value (line 291)

```291:291:histogram/weighted_histogram_mse_v4.py
    return hybrid_importance.view(original_shape)
```

The importance map is reshaped back to the original tensor shape (e.g., \((O, I, K_h, K_w)\) for Conv2d), so it can be passed directly to `WeightedHistogram.build()` as per-element weights.

---

## 6. Weighted Histogram with Per-Element Importance

### 6.1 Role

The `WeightedHistogram` class constructs a discrete histogram \(H(i)\) that approximates the importance-weighted distribution of absolute weight values. V4 extends the V1 histogram to accept **full per-element importance maps** (same shape as the weight tensor), in addition to the legacy per-channel 1D importance.

```91:94:histogram/weighted_histogram_mse_v4.py
class WeightedHistogram:
    """
    HSWQ-compliant weighted histogram (SVD 2D/4D importance).
    """
```

### 6.2 Construction — `build` (lines 103–160)

**Inputs:** Weight tensor \(\mathbf{W}\), optional importance \(\boldsymbol{\alpha}\).

**Step 1 — Absolute maximum:**

$$w_{\max} = \max_{m,n} |W_{m,n}|, \qquad \text{guard: } w_{\max} \leftarrow \max(w_{\max}, 10^{-7})$$

```108:113:histogram/weighted_histogram_mse_v4.py
        weight = weight.detach().float().to(self.device)
        w_abs = weight.abs()
        
        self.max_val = w_abs.max().item()
        if self.max_val == 0:
            self.max_val = 1e-7
```

**Step 2 — Importance expansion:**

V4 supports three importance modes:

| Mode | Condition | Behavior |
|------|-----------|----------|
| Per-element (V4) | `importance.shape == weight.shape` | Use directly as \(\alpha_{m,n}\). |
| Per-channel (V1) | `importance` is 1D | Broadcast to match weight shape. |
| Uniform | `importance is None` | \(\alpha_{m,n} = 1 \;\forall\; m,n\). |

```118:120:histogram/weighted_histogram_mse_v4.py
            # V2: when shapes match exactly, use as per-element importance (e.g. SVD scores)
            if importance.shape == weight.shape:
                imp_expanded = importance
```

For per-channel (1D) importance with a 2D weight \((O, I)\):

```136:143:histogram/weighted_histogram_mse_v4.py
                elif weight.dim() == 2:  # Linear: (Out, In)
                    in_features = weight.shape[1]
                    if importance.numel() >= in_features:
                        importance = importance[:in_features]
                    else:
                        padding = torch.ones(in_features - importance.numel(), device=self.device)
                        importance = torch.cat([importance, padding])
                    imp_expanded = importance.view(1, -1).expand_as(weight)
```

The 1D vector is trimmed or padded to match the input dimension, then broadcast across the output dimension.

**Step 3 — Binning and weighted accumulation:**

$$\text{bin\_width} = \frac{w_{\max}}{B}$$

$$b_{m,n} = \mathrm{clamp}\!\left(\left\lfloor \frac{|W_{m,n}|}{\text{bin\_width}} \right\rfloor,\; 0,\; B-1\right)$$

$$H_{\mathrm{raw}}(i) = \sum_{\{(m,n) : b_{m,n} = i\}} \alpha_{m,n}$$

```149:155:histogram/weighted_histogram_mse_v4.py
        # Bin indices
        bin_width = self.max_val / self.bins
        bin_indices = (w_abs / bin_width).long().clamp(0, self.bins - 1)
        
        # Build weighted histogram (add weights to bins)
        self.histogram = torch.zeros(self.bins, dtype=torch.float64, device=self.device)
        self.histogram.scatter_add_(0, bin_indices.reshape(-1), imp_expanded.reshape(-1).double())
```

The `scatter_add_` operation is the computational equivalent of the summation: for each element, its importance \(\alpha_{m,n}\) is added to the bin \(b_{m,n}\). This runs in \(\mathcal{O}(MN)\) and uses `float64` precision to avoid accumulation errors over millions of elements.

**Step 4 — Normalization:**

$$H(i) = \frac{H_{\mathrm{raw}}(i)}{\sum_{j=0}^{B-1} H_{\mathrm{raw}}(j)}, \qquad \sum_{i=0}^{B-1} H(i) = 1$$

```157:160:histogram/weighted_histogram_mse_v4.py
        self.total_weight = self.histogram.sum().item()
        
        if self.total_weight > 0:
            self.histogram = self.histogram / self.total_weight
```

### 6.3 Bin Centers (lines 162–170)

$$x_i = \left(i + \tfrac{1}{2}\right) \cdot \text{bin\_width}, \qquad i = 0, 1, \ldots, B-1$$

```162:170:histogram/weighted_histogram_mse_v4.py
    def get_bin_centers(self) -> torch.Tensor:
        bin_width = self.max_val / self.bins
        return torch.linspace(
            0.5 * bin_width,
            self.max_val - 0.5 * bin_width,
            self.bins,
            device=self.device,
            dtype=torch.float64
        )
```

Each bin center represents the "typical" absolute weight value for that bin. The histogram and bin centers together form a discrete approximation of the importance-weighted distribution of \(|\mathbf{W}|\).

---

## 7. MSE Optimizer

### 7.1 Weighted MSE Evaluation — `compute_weighted_mse` (lines 183–186)

**Formula:** For a candidate threshold \(\Delta\):

$$\mathrm{MSE}(\Delta) = \sum_{i=0}^{B-1} H(i) \cdot \bigl(q(x_i,\, \Delta) - x_i\bigr)^2$$

```183:186:histogram/weighted_histogram_mse_v4.py
    def compute_weighted_mse(self, histogram: torch.Tensor, bin_centers: torch.Tensor, amax: float, scaled: bool = True) -> float:
        dequantized = self.fp8_quantizer.quantize_dequantize(bin_centers.float(), amax, scaled=scaled).double()
        error_sq = (dequantized - bin_centers) ** 2
        return (histogram * error_sq).sum().item()
```

The bin centers \(x_i\) are passed through the full quantize–dequantize pipeline, and the squared error at each bin is weighted by \(H(i)\). Since \(\sum_i H(i) = 1\), this is a proper weighted average.

### 7.2 Multi-Stage Search — `find_optimal_amax` (lines 188–222)

**Parameters:**

| Parameter | V4 Default | Meaning |
|-----------|------------|---------|
| `num_candidates` | 1000 | Candidates per iteration. |
| `search_range` | \((r_{\mathrm{lo}}, r_{\mathrm{hi}})\) | Fraction of \(w_{\max}\); set per-layer by `get_dynamic_search_low`. |
| `refinement_iterations` | 10 | Number of narrowing iterations after the initial sweep. |
| `scaled` | `True`/`False` | Quantization mode. |

**Algorithm:**

**Initialization:**

$$\ell_0 = w_{\max} \cdot r_{\mathrm{lo}}, \qquad h_0 = w_{\max} \cdot r_{\mathrm{hi}}$$

$$\Delta^*_0 = w_{\max}, \qquad \mathrm{MSE}^*_0 = +\infty$$

**For** \(t = 0, 1, \ldots, T\) (where \(T = \texttt{refinement\_iterations}\)):

1. Generate \(N\) candidates: \(\Delta_1, \ldots, \Delta_N = \mathrm{linspace}(\ell_t, h_t, N)\)

2. Evaluate: \(\Delta^*_t = \arg\min_{\Delta \in \{\Delta_j\}} \mathrm{MSE}(\Delta)\)

3. If \(t < T\), refine:

$$w = \frac{h_t - \ell_t}{4}$$

$$\ell_{t+1} = \max(0.1 \cdot w_{\max},\; \Delta^*_t - w), \qquad h_{t+1} = \min(1.2 \cdot w_{\max},\; \Delta^*_t + w)$$

```206:222:histogram/weighted_histogram_mse_v4.py
        for iteration in range(refinement_iterations + 1):
            candidates = torch.linspace(low, high, num_candidates, device=self.device)
            
            for amax_tensor in candidates:
                amax = amax_tensor.item()
                mse = self.compute_weighted_mse(histogram, bin_centers, amax, scaled=scaled)
                
                if mse < min_mse:
                    min_mse = mse
                    best_amax = amax
            
            if iteration < refinement_iterations:
                range_width = (high - low) / 4
                low = max(max_val * 0.1, best_amax - range_width)
                high = min(max_val * 1.2, best_amax + range_width)
        
        return best_amax
```

**Convergence analysis:** Each refinement halves the effective search width (the new range is \(2w = (h_t - \ell_t)/2\)). After \(T\) iterations, the resolution is:

$$\frac{h_0 - \ell_0}{2^T \cdot N}$$

With \(N = 1000\) and \(T = 10\): initial resolution is \((h_0 - \ell_0) / 1000\), final resolution is \((h_0 - \ell_0) / (1000 \times 2^{10}) \approx (h_0 - \ell_0) / 10^6\). For a typical \(w_{\max} = 1.0\), this gives sub-microsecond precision on the amax.

**Total candidates evaluated:** \(N \times (T + 1) = 1000 \times 11 = 11{,}000\).

**Bound guards:** The lower bound is floored at \(0.1 \cdot w_{\max}\) (prevents degenerate near-zero amax) and the upper bound is capped at \(1.2 \cdot w_{\max}\) (allows slight expansion beyond max weight).

---

## 8. HSWQWeightedHistogramOptimizerV4 — Top-Level API

### 8.1 Constructor (lines 295–307)

```295:307:histogram/weighted_histogram_mse_v4.py
class HSWQWeightedHistogramOptimizerV4:
    """
    HSWQ weighted histogram optimizer (V4: SVD-Magnitude Hybrid).
    """
    
    def __init__(self, bins: int = 8192, num_candidates: int = 1000, refinement_iterations: int = 10, device: str = "cuda", alpha: float = 0.7, beta: float = 0.3):
        self.bins = bins
        self.num_candidates = num_candidates
        self.refinement_iterations = refinement_iterations
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.mse_optimizer = MSEOptimizer(device)
```

| Parameter | Default | Role |
|-----------|---------|------|
| `bins` | 8192 | Histogram resolution (\(B\)). |
| `num_candidates` | 1000 | Candidates per search iteration (\(N\)). |
| `refinement_iterations` | 10 | Narrowing iterations (\(T\)). |
| `alpha` | 0.7 | SVD leverage blend weight. |
| `beta` | 0.3 | RMS magnitude blend weight. |

### 8.2 `compute_optimal_amax` (lines 309–364)

This is the primary entry point. The full pipeline for a single layer:

```
Weight W ──┬──> compute_hybrid_leverage_scores(W, α, β) ──> Hybrid S (2D)
           │                                                      │
           │   DualMonitor.channel_importance (1D, optional) ─────┤
           │                                                      ↓
           │                                            Multiply: S × I_expanded
           │                                                      │
           ├──────────────────────────────────────────────────────→↓
           │                                              WeightedHistogram.build(W, combined)
           │                                                      │
           │                                                      ↓
           └──────────────────────────────────────────> MSEOptimizer.find_optimal_amax(H, search_range)
                                                                  │
                                                                  ↓
                                                           Δ* (optimal amax)
```

**Step 1 — Hybrid importance:**

```316:318:histogram/weighted_histogram_mse_v4.py
        if use_svd_leverage and weight.ndim >= 2:
            # Compute hybrid importance matrix (SVD + RMS)
            hybrid_importance = compute_hybrid_leverage_scores(weight, alpha=self.alpha, beta=self.beta)
```

**Step 2 — Calibration importance fusion (optional):**

If a 1D `importance` vector (from DualMonitor calibration) is provided, it is broadcast to the weight shape and **element-wise multiplied** with the hybrid scores:

$$\alpha_{m,n}^{\mathrm{final}} = S_{m,n} \cdot I_{\mathrm{expanded},m,n}$$

```322:339:histogram/weighted_histogram_mse_v4.py
            # If 1D channel importance (e.g. from DualMonitor) exists, multiply
            if importance is not None:
                importance = importance.float().to(self.device)
                
                # Broadcast to match weight dimensions
                if weight.ndim == 4:
                    in_channels = weight.shape[1]
                    pad_len = max(0, in_channels - importance.numel())
                    imp_1d = torch.cat([importance[:in_channels], torch.ones(pad_len, device=self.device)])
                    imp_expanded = imp_1d.view(1, -1, 1, 1).expand_as(weight)
                elif weight.ndim == 2:
                    in_features = weight.shape[1]
                    pad_len = max(0, in_features - importance.numel())
                    imp_1d = torch.cat([importance[:in_features], torch.ones(pad_len, device=self.device)])
                    imp_expanded = imp_1d.view(1, -1).expand_as(weight)
                else:
                    imp_expanded = importance.expand_as(weight)
                    
                combined_importance = hybrid_importance * imp_expanded
```

This multiplicative fusion means that channels with high activation importance **and** high structural importance receive the highest weight in the histogram. Neither signal can override the other; both must agree for maximum importance.

**Step 3 — Build histogram and search:**

```345:362:histogram/weighted_histogram_mse_v4.py
        # Build weighted histogram (2D/4D combined_importance gives per-pixel weights)
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, combined_importance)
        
        # Search for optimal amax
        ...
        optimal_amax = self.mse_optimizer.find_optimal_amax(
            weighted_hist,
            num_candidates=self.num_candidates,
            search_range=search_range,
            refinement_iterations=self.refinement_iterations,
            scaled=scaled
        )
```

### 8.3 `compute_optimal_amax_with_stats` (lines 366–389)

Identical optimization pipeline, but additionally returns diagnostic statistics:

| Key | Formula / Meaning |
|-----|-------------------|
| `optimal_amax` | \(\Delta^*\) |
| `max_val` | \(w_{\max} = \max_{m,n} |W_{m,n}|\) |
| `compression_ratio` | \(\Delta^* / w_{\max}\) — how aggressively the range is clipped. |
| `estimated_mse` | \(\mathrm{MSE}(\Delta^*)\) — residual quantization error at the optimum. |

---

## 9. Integration with V1.92 Quantization Pipeline

### 9.1 Autonomous \(\alpha/\beta\) Derivation

In the V1.92 quantizer (`quantize_zib_hswq_v1.92.py`), \(\alpha\) and \(\beta\) are derived from the model's global kurtosis profile:

$$k_{\mathrm{factor}} = \min\!\left(\frac{\overline{\kappa}}{50},\; 0.3\right), \qquad \alpha = \mathrm{clip}(0.5 + k_{\mathrm{factor}},\; 0.5,\; 0.8), \qquad \beta = 1 - \alpha$$

where \(\overline{\kappa}\) is the mean Fisher kurtosis across all profiled layers.

| \(\overline{\kappa}\) | \(\alpha\) | \(\beta\) | Interpretation |
|---|---|---|---|
| 0 (Gaussian) | 0.50 | 0.50 | Equal blend; no structural bias needed. |
| 15 (moderate tails) | 0.80 | 0.20 | Prioritize SVD structure preservation. |
| \(\geq 50\) | 0.80 | 0.20 | Maximum SVD protection (capped). |

### 9.2 Per-Layer Search Range

The search range \((r_{\mathrm{lo}}, 1.0)\) is computed per layer from its kurtosis \(\kappa\) and outlier ratio \(o\):

$$r_{\mathrm{lo}} = \mathrm{clip}\!\left(0.50 + \max\!\left(\frac{\kappa}{100},\; \frac{o}{60}\right),\; 0.50,\; 0.99\right)$$

Layers with extreme distributions get narrow search ranges (e.g., \(r_{\mathrm{lo}} = 0.99\) means amax is nearly fixed at \(w_{\max}\)), while smooth layers have wide ranges (e.g., \(r_{\mathrm{lo}} = 0.50\)) allowing aggressive clipping.

### 9.3 DualMonitor Channel Importance

During calibration, `DualMonitor` hooks record the mean absolute activation per input channel:

$$I_c = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} |x_{t,c,s}|$$

where \(T\) is the number of calibration samples and \(\mathcal{S}\) ranges over spatial/sequence dimensions. This 1D vector is passed to `compute_optimal_amax` as the `importance` parameter, where it is multiplied with the 2D hybrid scores (§8.2, Step 2).

### 9.4 End-to-End Data Flow per FP8 Layer

$$\mathbf{W} \xrightarrow{\text{SVD}} (\mathbf{U}, \boldsymbol{\sigma}, \mathbf{V}_h) \xrightarrow{\sigma^2\text{-leverage}} \mathbf{L} \xrightarrow{\text{L2 norm}} \hat{\mathbf{L}}$$

$$\mathbf{W} \xrightarrow{(\cdot)^2} \mathbf{M} \xrightarrow{\text{L2 norm}} \hat{\mathbf{M}}$$

$$\hat{\mathbf{L}},\, \hat{\mathbf{M}} \xrightarrow{\alpha,\beta} \mathbf{S}_{\mathrm{raw}} \xrightarrow{\text{mean norm}} \mathbf{S}_{\mathrm{norm}} \xrightarrow{0.5+0.5\cdot} \mathbf{S}$$

$$\mathbf{S} \times I_{\mathrm{expanded}} = \boldsymbol{\alpha}^{\mathrm{final}}$$

$$\mathbf{W},\, \boldsymbol{\alpha}^{\mathrm{final}} \xrightarrow{\text{histogram}} H(i) \xrightarrow{\text{MSE search}} \Delta^*$$

$$\mathbf{W},\, \Delta^* \xrightarrow{\text{clip + FP8 cast}} \mathbf{W}_{\mathrm{FP8}}$$

---

## 10. Self-Test (lines 392–429)

The module includes a self-test that validates the full pipeline:

```392:409:histogram/weighted_histogram_mse_v4.py
if __name__ == "__main__":
    print("HSWQ V4: Hybrid SVD-Magnitude Blended Histogram MSE - Self Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test: random weight tensor with 1D and hybrid importance
    print("\n[Test] 2D Weight Matrix Hybrid Extraction and Amax Optimization")
    
    # Dummy (128, 256) tensor
    U_true = torch.randn(128, 16, device=device)
    V_true = torch.randn(256, 16, device=device)
    weight = U_true @ V_true.T
    
    # Add intentional outliers
    weight[5, 5] = 20.0
    weight[10, 100] = -25.0
```

The test constructs a rank-16 matrix with injected outliers, then compares amax values obtained with and without hybrid importance:

1. **Without hybrid** (`use_svd_leverage=False`): Uniform importance; pure histogram MSE. The optimizer is unaware of structural significance.

2. **With hybrid** (`use_svd_leverage=True`): SVD+RMS importance. The optimizer preserves structurally and energetically important elements at the cost of higher error on unimportant ones.

The hybrid-aware amax typically differs from the uniform amax, reflecting the optimizer's preference for preserving the principal subspace.

---

## 11. Summary

### 11.1 Formula Index

| Formula | Section |
|---------|---------|
| \(\Delta^* = \arg\min_\Delta \sum_i H(i)(q(x_i,\Delta)-x_i)^2\) | §1.1 |
| \(L_{m,n} = \bigl(\sum_k U_{mk}^2\sigma_k^2\bigr)\bigl(\sum_k V_{nk}^2\sigma_k^2\bigr)\) | §3.3 |
| \(\mathbf{L} = \bigl[(\mathbf{U}\odot\mathbf{U})\boldsymbol{\sigma}^2\bigr]\bigl[(\mathbf{V}_h^T\odot\mathbf{V}_h^T)\boldsymbol{\sigma}^2\bigr]^T\) | §3.5 |
| \(M_{m,n} = W_{m,n}^2\) | §4.1 |
| \(\hat{\mathbf{L}} = \mathbf{L}/\|\mathbf{L}\|_2\), \(\hat{\mathbf{M}} = \mathbf{M}/\|\mathbf{M}\|_2\) | §5.5 |
| \(\mathbf{S}_{\mathrm{raw}} = \alpha\hat{\mathbf{L}} + \beta\hat{\mathbf{M}}\) | §5.6 |
| \(S_{m,n} = 0.5 + 0.5 \cdot S_{\mathrm{norm},m,n}\) | §5.7 |
| \(\alpha_{m,n}^{\mathrm{final}} = S_{m,n} \cdot I_{\mathrm{expanded},m,n}\) | §8.2 |
| \(H_{\mathrm{raw}}(i) = \sum_{\{(m,n):b_{m,n}=i\}} \alpha_{m,n}^{\mathrm{final}}\) | §6.2 |
| \(H(i) = H_{\mathrm{raw}}(i) / \sum_j H_{\mathrm{raw}}(j)\) | §6.2 |
| \(x_i = (i+0.5) \cdot w_{\max}/B\) | §6.3 |
| \(q(x,\Delta)\) scaled: \(\mathrm{round}_{\mathrm{FP8}}(x\cdot 448/\Delta)/(448/\Delta)\) | §2.3 |
| \(q(x,\Delta)\) non-scaled: \(\mathrm{round}_{\mathrm{FP8}}(\mathrm{clip}(x,-\Delta,\Delta))\) | §2.3 |
| \(\mathrm{round}_{\mathrm{FP8}}(v) = \mathrm{sign}(v)\cdot\arg\min_{g\in\mathcal{G}^+}|g-|v||\) | §2.4 |
| Refinement: \(w=(h-\ell)/4\); \(\ell' = \max(0.1w_{\max}, \Delta^*-w)\); \(h' = \min(1.2w_{\max}, \Delta^*+w)\) | §7.2 |
| \(\alpha = \mathrm{clip}(0.5 + \min(\overline{\kappa}/50, 0.3),\; 0.5,\; 0.8)\), \(\beta = 1-\alpha\) | §9.1 |
| \(r_{\mathrm{lo}} = \mathrm{clip}(0.50 + \max(\kappa/100, o/60),\; 0.50,\; 0.99)\) | §9.2 |

### 11.2 Component Table

| Component | Responsibility |
|-----------|----------------|
| **`compute_hybrid_leverage_scores`** | Full SVD → \(\sigma^2\)-weighted bilateral leverage; RMS magnitude; L2 normalization; \(\alpha/\beta\) blend; mean normalization; baseline offset. Returns per-element \(\mathbf{S}\). |
| **`FP8E4M3Quantizer`** | Physical FP8 E4M3 grid from byte patterns; `quantize_dequantize` (scaled/non-scaled); nearest-grid rounding. |
| **`WeightedHistogram`** | Per-element weighted histogram \(H(i)\) via `scatter_add_` in float64; supports 2D/4D importance; normalize to \(\sum=1\). |
| **`MSEOptimizer`** | Evaluate \(\mathrm{MSE}(\Delta) = \sum_i H(i)(q(x_i,\Delta)-x_i)^2\); iterative refinement search (1000×11 candidates). |
| **`HSWQWeightedHistogramOptimizerV4`** | Compose all components: hybrid importance → (optional) calibration fusion → histogram → MSE search → \(\Delta^*\). |

### 11.3 Design Principles

1. **Orthogonal dual objectives.** SVD leverage captures structural significance (principal subspace participation); RMS magnitude captures energetic significance (forward-pass signal power). Neither alone is sufficient; their combination provides robust importance estimates across diverse weight distributions.

2. **Scale-invariant blending.** L2 normalization ensures that \(\alpha, \beta\) operate on commensurate quantities regardless of the absolute scale of leverage or magnitude matrices.

3. **Physical FP8 fidelity.** The quantizer uses the actual PyTorch FP8 E4M3 grid (not a simplified model), ensuring that the MSE computed during optimization matches the actual quantization error at inference time.

4. **Autonomous parameterization.** Both the blend ratio \((\alpha, \beta)\) and the per-layer search range are derived from measurable statistical properties (kurtosis, outlier ratio) of the weight distribution, eliminating manual tuning.
