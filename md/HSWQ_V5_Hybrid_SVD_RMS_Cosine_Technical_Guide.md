# HSWQ V5: SVD–Magnitude Hybrid Weighted Histogram Cosine Optimizer — Technical Guide

**Source:** `histogram/weighted_histogram_cosine_v5.py`

**Companion (MSE objective):** `md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md`  
**Companion (MSE source):** `histogram/weighted_histogram_mse_v4.py`

This document provides a complete mathematical and implementation-level description of the HSWQ Weighted Histogram Optimizer **V5**, which determines the optimal per-layer clipping threshold \(\Delta^*\) (amax) for FP8 E4M3 post-training quantization. V5 keeps the **same hybrid per-element importance model** family as V4 (SVD structural leverage + RMS magnitude), but replaces the amax search objective: the loss is **Cosine Similarity Loss** on the importance-weighted magnitude histogram, not Weighted Mean Squared Error.

A dedicated comparison (§12–§13) derives, without hand-waving, why cosine loss and MSE are **not interchangeable surrogates** for quantization-threshold search, and when cosine is the superior fidelity criterion for FP8 grid alignment under heavy-tailed weight distributions.

---

## 0. Position of V5 Inside HSWQ

### 0.1 What Is Unchanged From the HSWQ Framework

HSWQ still answers three questions in three stages (see V4 guide §0.3):

| Stage | Question | Signal |
|-------|----------|--------|
| 1 Dual Monitor | Which layers / channels matter? | Sensitivity \(\mathrm{Var}(Y)\), importance \(I_c\) |
| 2 Layer Selection | Keep FP16 or quantize to FP8? | Keep ratio, VETO, profile scores |
| 3 Histogram Optimizer | **Given** an FP8 layer, where is \(\Delta^*\)? | Weighted histogram + **objective** |

V5 is exclusively a **Stage 3 objective swap**. It does not redefine Stage 1 or Stage 2. Layer selection, Hard VETO, DualMonitor NaN protection, and keep-ratio policy remain the responsibility of the quantizer entrypoint (e.g. Z Image / SDXL pipelines). Calling V5 on the wrong layer set cannot repair a broken Stage 2 — the same lesson as V4 (§0.5 of the V4 guide).

### 0.2 What V5 Changes

| Aspect | V4 (`weighted_histogram_mse_v4.py`) | V5 (`weighted_histogram_cosine_v5.py`) |
|--------|--------------------------------------|----------------------------------------|
| Amax objective | \(\sum_i H(i)\,(q(x_i,\Delta)-x_i)^2\) | \(1 - \dfrac{\sum_i H(i)\,x_i\,q_i}{\sqrt{(\sum_i H(i)\,x_i^2)(\sum_i H(i)\,q_i^2)}}\) |
| Optimizer class | `MSEOptimizer` | `CosineOptimizer` |
| Top-level class | `HSWQWeightedHistogramOptimizerV4` | `HSWQWeightedHistogramOptimizerV5` |
| Default `loss_type` | (implicit MSE) | `"cosine"` (hard requirement) |
| Importance | SVD + RMS hybrid | Same API surface; **leverage GEMM form differs** (§5.3) |
| Defaults \(B,N,T\) | 8192 / 1000 / 10 | 8192 / 1000 / 10 |

### 0.3 Module Docstring Contract (Source of Truth)

```1:21:histogram/weighted_histogram_cosine_v5.py
"""
HSWQ Weighted Histogram Optimizer V5 (SVD & RMS Magnitude Hybrid + Cosine Loss)
================================================================================

Amax search objective is Cosine Similarity Loss only:

  L = 1 - (x · q) / (||x|| ||q||)

with histogram weights:
  dot    = Σ h · x · q
  norm_x = Σ h · x²
  norm_q = Σ h · q²

Importance for the histogram comes from SVD leverage + RMS magnitude hybrid.
Cosine loss measures how faithfully the FP8 grid reproduces the original
weight magnitude distribution under the chosen amax. A higher amax clips
fewer outliers but coarsens the grid for small magnitudes; a lower amax
preserves small-magnitude resolution but truncates large values. The
optimizer finds the amax that minimizes this distributional distortion,
weighted by per-element SVD+RMS importance.
"""
```

**Operational reading of that contract:**

1. The search never mixes MSE and cosine in one run (`loss_type != "cosine"` raises).
2. The vectors being compared are **histogram bin centers** (absolute magnitudes) and their **FP8 reconstructs**, not the signed full tensor \(\mathbf{W}\) vs \(q(\mathbf{W})\).
3. The histogram mass \(H(i)\) is importance-weighted, so cosine is **not** uniform over elements; it is cosine in an \(H\)-induced discrete measure.

### 0.4 Why a Second Optimizer Exists

MSE asks: *“How large is the squared residual after quantize–dequantize?”*  
Cosine asks: *“Does the reconstructed magnitude vector point in the same direction as the original, in the importance-weighted \(L^2\) sense?”*

For FP8 threshold search these questions diverge:

- Clipping a few extreme outliers can **reduce MSE** while **tilting** the relative shape of the remaining mass (grid steps coarsen across the bulk).
- Stretching the grid to protect the bulk can **raise MSE** on the tails while **preserving angular fidelity** of the bulk distribution that carries most forward energy under importance weights.

V5 exists so Stage 3 can optimize the second criterion when the deployment goal is **distributional / directional fidelity of magnitudes under the FP8 grid**, not absolute squared residual alone.

---

## 1. Problem Statement

### 1.1 Formal Optimization Problem

Given a trained weight tensor viewed as \(\mathbf{W} \in \mathbb{R}^{M \times N}\) (Conv2d flattened to \((O, I\cdot K_h\cdot K_w)\)), an FP8 E4M3 quantize–dequantize map \(q(\cdot, \Delta)\), and a normalized importance-weighted histogram \(H\) of absolute weights with bin centers \(x_i \geq 0\):

$$\Delta^* = \arg\min_{\Delta} \; L_{\mathrm{cos}}(\Delta)$$

$$L_{\mathrm{cos}}(\Delta) = 1 - \cos\bigl(\mathbf{x},\; \mathbf{q}(\Delta)\bigr)_{H}$$

where the **importance-weighted cosine similarity** is

$$\cos(\mathbf{x}, \mathbf{q})_{H}
  = \frac{\displaystyle\sum_{i=0}^{B-1} H(i)\, x_i\, q_i}
         {\displaystyle\sqrt{\Bigl(\sum_{i=0}^{B-1} H(i)\, x_i^2\Bigr)
                             \Bigl(\sum_{i=0}^{B-1} H(i)\, q_i^2\Bigr)}}$$

with \(q_i = q(x_i, \Delta)\) and \(\sum_i H(i) = 1\), \(H(i) \geq 0\).

Equivalently, define the discrete weighted inner product

$$\langle \mathbf{a}, \mathbf{b} \rangle_H = \sum_i H(i)\, a_i\, b_i,$$

then

$$\cos(\mathbf{x}, \mathbf{q})_{H}
  = \frac{\langle \mathbf{x}, \mathbf{q} \rangle_H}
         {\|\mathbf{x}\|_H \|\mathbf{q}\|_H},
  \qquad
  \|\mathbf{v}\|_H = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle_H}.$$

### 1.2 Relation to Unweighted Cosine

If \(H(i) \equiv 1/B\) (uniform bins, **not** the HSWQ case), the formula reduces to ordinary cosine of the vectors \((x_0,\ldots,x_{B-1})\) and \((q_0,\ldots,q_{B-1})\). HSWQ instead places mass where hybrid importance says elements matter. Two bins with the same \(|W|\) but different total importance contribute differently to \(\langle\cdot,\cdot\rangle_H\).

### 1.3 What Is Being Matched

Because the histogram is built on \(w_{\mathrm{abs}} = |\mathbf{W}|\) (§7), the optimizer matches **magnitude shapes**, not signed geometry of \(\mathbf{W}\). Sign is restored only inside \(q\) via nearest-grid rounding on absolute values (§3.4). This is intentional: FP8 clipping thresholds for PTQ are almost always chosen on absolute ranges; cosine on signed weights would be dominated by cancellation across positive/negative lobes and would not match ComfyUI / PyTorch FP8 cast practice.

### 1.4 Notation

| Symbol | Meaning |
|--------|---------|
| \(\mathbf{W} \in \mathbb{R}^{M \times N}\) | Weight matrix (2D view). |
| \(\mathbf{U}, \boldsymbol{\sigma}, \mathbf{V}_h\) | Compact SVD: \(\mathbf{W} = \mathbf{U}\,\mathrm{diag}(\boldsymbol{\sigma})\,\mathbf{V}_h\). |
| \(L_{m,n}\) | SVD leverage (V5 GEMM form, §5.3). |
| \(M_{m,n}\) | RMS magnitude \(W_{m,n}^2\). |
| \(S_{m,n}\) | Hybrid importance after blend + normalization. |
| \(I_c\) | Optional 1D DualMonitor channel importance. |
| \(\Delta\) (amax) | Clipping / scale threshold optimized. |
| \(B\) | Histogram bins (default 8192). |
| \(H(i)\) | Normalized weighted histogram; \(\sum_i H(i)=1\). |
| \(x_i\) | Bin center (absolute). |
| \(q_i = q(x_i,\Delta)\) | FP8 reconstruct of bin center. |
| \(L_{\mathrm{cos}}(\Delta)\) | Cosine loss \(1 - \cos_H(\mathbf{x},\mathbf{q})\). |
| \(L_{\mathrm{MSE}}(\Delta)\) | V4 objective \(\sum_i H(i)(q_i-x_i)^2\) (comparison only). |

---

## 2. Component Architecture

| Component | Responsibility |
|-----------|----------------|
| **`FP8E4M3Quantizer`** | Physical E4M3 grid; \(q(x,\Delta)\) scaled / non-scaled. |
| **`WeightedHistogram`** | Build \(H(i)\) from \(\mathbf{W}\) and importance; bin centers. |
| **`compute_hybrid_leverage_scores`** | SVD+RMS hybrid \(\mathbf{S}\) (V5 leverage GEMM). |
| **`CosineOptimizer`** | Evaluate \(L_{\mathrm{cos}}(\Delta)\); iterative grid search. |
| **`HSWQWeightedHistogramOptimizerV5`** | Compose: hybrid → optional \(I_c\) fuse → histogram → cosine search. |

Pipeline:

```
W ──> compute_hybrid_leverage_scores ──> S
         optional I_c broadcast ──> α_final = S ⊙ I_exp
W, α_final ──> WeightedHistogram.build ──> H(i), x_i
H, x ──> CosineOptimizer.find_optimal_amax ──> Δ*
```

---

## 3. FP8 E4M3 Quantizer

### 3.1 Specification

Identical physical format to V4: PyTorch `float8_e4m3fn`, max finite magnitude \(448\), 4-bit exponent (bias 7), 3-bit mantissa, NaN encodings `0x7F` / `0xFF`, no infinities. Representable range includes denormals down to \(2^{-9}\).

### 3.2 Grid Construction — `_build_fp8_grid` (lines 43–60)

```43:60:histogram/weighted_histogram_cosine_v5.py
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

$$\mathcal{G}^+ = \bigl\{ \mathrm{float32}(\mathrm{view}_{\mathrm{E4M3}}(b))
  \;\big|\; b\in[0,255],\; \text{finite},\; \ge 0 \bigr\}_{\mathrm{unique}}$$

### 3.3 Quantize–Dequantize

**Scaled (`scaled=True`):**

$$s = 448/\Delta,\qquad
q(x,\Delta)=\frac{\mathrm{round}_{\mathrm{FP8}}(\mathrm{clip}(x\cdot s,-448,448))}{s}$$

**Non-scaled (`scaled=False`):**

$$q(x,\Delta)=\mathrm{round}_{\mathrm{FP8}}\bigl(\mathrm{clip}(\mathrm{clip}(x,-\Delta,\Delta),-448,448)\bigr)$$

```62:78:histogram/weighted_histogram_cosine_v5.py
    def quantize_dequantize(self, values: torch.Tensor, amax: float, scaled: bool = True) -> torch.Tensor:
        """Full quantize-then-dequantize function q(x, delta)."""
        if amax <= 0:
            return torch.zeros_like(values)
        
        if scaled:
            scale = self.max_representable / amax
            scaled_vals = values * scale
            scaled_vals = scaled_vals.clamp(-self.max_representable, self.max_representable)
            quantized = self._round_to_fp8_grid(scaled_vals)
            dequantized = quantized / scale
            return dequantized
        else:
            clipped = values.clamp(-amax, amax)
            clipped = clipped.clamp(-self.max_representable, self.max_representable)
            dequantized = self._round_to_fp8_grid(clipped)
            return dequantized
```

Production HSWQ V1 paths typically call Stage 3 with `scaled=False` (amax = clipping threshold; stored `weight_scale=1.0`). The optimizer still accepts `scaled=True` for V2-style analysis.

### 3.4 Nearest-Grid Rounding

$$\mathrm{round}_{\mathrm{FP8}}(v)=\mathrm{sign}(v)\cdot\arg\min_{g\in\mathcal{G}^+}|g-|v||$$

Batched at 10000 elements to bound the \((P,|\mathcal{G}^+|)\) distance matrix (§2.4 of the V4 guide applies verbatim).

---

## 4. RMS Magnitude Scores

$$M_{m,n}=W_{m,n}^2$$

Same energetic interpretation as V4 §4: contribution to \(\|\mathbf{W}\|_F^2\) and, under isotropic input assumptions, to expected output power of \(\mathbf{y}=\mathbf{W}\mathbf{x}\).

---

## 5. SVD Leverage — V5 Exact Per-Mode Form

### 5.1 Motivation (Shared With V4)

For \(\mathbf{y}=\mathbf{W}\mathbf{x}\) and compact SVD \(\mathbf{W}=\mathbf{U}\,\mathrm{diag}(\boldsymbol{\sigma})\,\mathbf{V}_h\), elements that participate in large-\(\sigma_k\) modes dominate output geometry. Quantization error on those elements is structurally more damaging than the same absolute error on near-nullspace coordinates.

### 5.2 Critical Difference From V4’s Outer-Product Leverage

**V4** (`weighted_histogram_mse_v4.py`) implements **bilateral outer-product leverage**:

$$L^{\mathrm{V4}}_{m,n}
  = \underbrace{\Bigl(\sum_k U_{mk}^2\sigma_k^2\Bigr)}_{r_m}
    \underbrace{\Bigl(\sum_k V_{h,kn}^2\sigma_k^2\Bigr)}_{c_n}
  = r_m\, c_n$$

**V5** implements the **mode-coupled (diagonal) form** documented in code comments:

$$L^{\mathrm{V5}}_{m,n}
  = \sum_{k=1}^{K} U_{mk}^2\, \sigma_k^2\, V_{h,kn}^2$$

```287:302:histogram/weighted_histogram_cosine_v5.py
    # --- 1. SVD Leverage (top-k: σ^2 weighted) ---
    # Full SVD then truncate to top-k components (k controlled by max_k/top_p/min_k).
    # This makes the max_k parameter functional: large matrices use only the
    # top-k singular directions for leverage, reducing compute from O(MN·r) to O(MN·k).
    U, S, Vh = torch.linalg.svd(w_float, full_matrices=False)
    if k < S.shape[0]:
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]

    # Leverage formula: leverage[i,j] = Σ_k U[i,k]² · σ_k² · Vh[k,j]²
    # Implemented as a single GEMM: (U² · S²) @ (Vh²)  →  (M, k) @ (k, N) = (M, N)
    S_sq = S ** 2
    US_sq = (U ** 2) * S_sq.unsqueeze(0)         # (M, k) each col scaled by σ_k²
    Vh_sq = Vh ** 2                               # (k, N)
    leverage_2d = US_sq @ Vh_sq                   # (M, N) exact per-element leverage
```

Expanding the outer product shows the algebraic gap:

$$L^{\mathrm{V4}}_{m,n}
  = \sum_k\sum_{k'} U_{mk}^2\sigma_k^2\, V_{h,k'n}^2\sigma_{k'}^2
  = L^{\mathrm{V5}}_{m,n}
    + \underbrace{\sum_{k\neq k'} U_{mk}^2\sigma_k^2\, V_{h,k'n}^2\sigma_{k'}^2}_{\text{cross-mode terms}}.$$

So \(L^{\mathrm{V4}}\) **includes cross-mode products** while \(L^{\mathrm{V5}}\) **retains only same-mode couplings**. V5’s form is the natural elementwise contribution of each singular mode’s rank-1 factor \(\sigma_k\,\mathbf{u}_k\mathbf{v}_k^\top\) to a \(\sigma_k^2\)-weighted participation score without mixing unrelated modes through an outer product of aggregated scores.

**Interpretation of \(L^{\mathrm{V5}}_{m,n}\):** Element \((m,n)\) scores high when there exists at least one large mode \(k\) for which both \(U_{mk}\) and \(V_{h,kn}\) are large. It does **not** inflate scores merely because row \(m\) is important in mode \(k\) and column \(n\) is important in an unrelated mode \(k'\).

### 5.3 Compact Matrix Form (V5)

$$\mathbf{L}^{\mathrm{V5}}
  = \bigl((\mathbf{U}\odot\mathbf{U})\odot\boldsymbol{\sigma}^{2\top}\bigr)\,
    (\mathbf{V}_h\odot\mathbf{V}_h)
  = (\mathbf{U}^{\odot 2}\mathrm{diag}(\boldsymbol{\sigma}^2))\,(\mathbf{V}_h^{\odot 2})$$

in code: `US_sq @ Vh_sq`.

### 5.4 Top-\(k\) Truncation (Functional `max_k`)

```278:281:histogram/weighted_histogram_cosine_v5.py
    M, N = w_float.shape
    max_rank = min(M, N)
    k = min(max_k, max(min_k, int(math.floor(top_p * max_rank))))
    k = min(k, max_rank)
```

After full `svd`, factors are truncated to the first \(k\) components. Defaults (`top_p=1.0`, `max_k=4096`) keep full rank for typical layers with \(\min(M,N)\le 4096\). For larger matrices, \(k=\texttt{max\_k}\) makes the leverage GEMM \(\mathcal{O}(MNk)\) instead of \(\mathcal{O}(MN\cdot\min(M,N))\).

### 5.5 Complexity

- SVD: \(\mathcal{O}(MN\min(M,N))\) dominant.
- Leverage GEMM: \(\mathcal{O}(MNk)\).
- Memory peak during SVD is the main practical cost for wide Linear layers.

---

## 6. Hybrid Importance — `compute_hybrid_leverage_scores`

### 6.1 Signature

```258:263:histogram/weighted_histogram_cosine_v5.py
def compute_hybrid_leverage_scores(weight: torch.Tensor, alpha: float = 0.7, beta: float = 0.3, top_p: float = 1.0, min_k: int = 1, max_k: int = 4096) -> torch.Tensor:
    """
    Blended importance: SVD structural leverage and RMS magnitude,
    each L2-normalized then combined with (alpha, beta). Used as histogram weights
    for Cosine amax search.
    """
```

| Parameter | Default | Role |
|-----------|---------|------|
| `alpha` | 0.7 | Weight on \(\hat{\mathbf{L}}\) |
| `beta` | 0.3 | Weight on \(\hat{\mathbf{M}}\) |
| `top_p` / `min_k` / `max_k` | 1.0 / 1 / 4096 | Singular truncation |

### 6.2 Preprocessing

- `ndim > 2` → view as `(shape[0], -1)`.
- `ndim < 2` or all-zero → return ones (uniform importance).

### 6.3 L2 Normalization, Blend, Mean Norm, Baseline

Identical affine pipeline to V4:

$$\hat{\mathbf{L}}=\frac{\mathbf{L}}{\|\mathbf{L}\|_2},\quad
\hat{\mathbf{M}}=\frac{\mathbf{M}}{\|\mathbf{M}\|_2}$$

$$\mathbf{S}_{\mathrm{raw}}=\alpha\hat{\mathbf{L}}+\beta\hat{\mathbf{M}}$$

$$\mathbf{S}_{\mathrm{norm}}=\mathbf{S}_{\mathrm{raw}}/\overline{S}_{\mathrm{raw}}$$

$$S_{m,n}=0.5+0.5\,S_{\mathrm{norm},m,n}$$

```307:327:histogram/weighted_histogram_cosine_v5.py
    # --- 3. L2 normalize (equal impact per score matrix) ---
    lev_norm = torch.norm(leverage_2d, p=2)
    mag_norm = torch.norm(magnitude_2d, p=2)
    
    # Avoid division by zero
    if lev_norm > 0: leverage_2d = leverage_2d / lev_norm
    if mag_norm > 0: magnitude_2d = magnitude_2d / mag_norm

    # --- 4. Alpha/Beta blend ---
    hybrid_importance = (alpha * leverage_2d) + (beta * magnitude_2d)

    # --- 5. Histogram scale normalization ---
    # Scale so mean ~1.0 and histogram area matches weight count
    avg_score = hybrid_importance.mean()
    if avg_score > 0:
        hybrid_importance = hybrid_importance / avg_score

    # V2-style mild baseline (avoid 0-div and full collapse)
    hybrid_importance = 0.5 + 0.5 * hybrid_importance

    return hybrid_importance.view(original_shape)
```

**Floor at 0.5:** No element is erased from the histogram. Cosine still “sees” low-importance bins; they simply pull less.

---

## 7. Weighted Histogram

### 7.1 Absolute Binning

$$w_{\max}=\max|W_{m,n}|,\quad
b_{m,n}=\mathrm{clamp}\!\left(\Big\lfloor\frac{|W_{m,n}|}{w_{\max}/B}\Big\rfloor,0,B-1\right)$$

$$H_{\mathrm{raw}}(i)=\sum_{\{(m,n):b_{m,n}=i\}}\alpha_{m,n},\qquad
H(i)=\frac{H_{\mathrm{raw}}(i)}{\sum_j H_{\mathrm{raw}}(j)}$$

Implemented with `scatter_add_` in `float64` (lines 157–168).

### 7.2 Importance Modes

| Mode | Condition | Behavior |
|------|-----------|----------|
| Per-element | `importance.shape == weight.shape` | Direct \(\alpha_{m,n}\) |
| Per-channel 1D | Legacy | Broadcast on input dim |
| Uniform | `None` | Ones |

### 7.3 Bin Centers

$$x_i=\Bigl(i+\tfrac12\Bigr)\frac{w_{\max}}{B},\qquad i=0,\ldots,B-1$$

These \(x_i\) are the only values passed through \(q(\cdot,\Delta)\) during the search. The optimizer never re-quantizes every individual weight; it optimizes against a **compressed sufficient statistic** of the importance-weighted magnitude law.

---

## 8. Cosine Optimizer

### 8.1 Class Contract

```184:192:histogram/weighted_histogram_cosine_v5.py
class CosineOptimizer:
    """Cosine-loss amax grid search over a weighted histogram (V5).

    Minimizes 1 - cos_sim(|w|, q(|w|, amax)) over histogram bins, where q is
    the FP8 E4M3 quantize-dequantize function. This measures how well the FP8
    grid reproduces the original weight magnitude distribution under amax,
    weighted by SVD+RMS per-element importance. Completely separate from V4
    MSEOptimizer / compute_weighted_mse.
    """
```

### 8.2 Evaluation — `compute_weighted_cosine` (lines 198–213)

```198:213:histogram/weighted_histogram_cosine_v5.py
    def compute_weighted_cosine(self, histogram: torch.Tensor, bin_centers: torch.Tensor, amax: float, scaled: bool = True, loss_type: str = "cosine") -> float:
        """Cosine loss: 1 - (dot / sqrt(norm_x * norm_q)) under histogram weights."""
        if loss_type is None:
            loss_type = DEFAULT_LOSS_TYPE
        if loss_type != "cosine":
            raise ValueError(
                f"V5 CosineOptimizer supports Cosine only (got loss_type={loss_type!r})"
            )
        dequantized = self.fp8_quantizer.quantize_dequantize(bin_centers.float(), amax, scaled=scaled).double()
        # Cosine Loss: 1 - (dot / sqrt(norm_x * norm_q))
        dot = (histogram * bin_centers * dequantized).sum()
        norm_x = (histogram * (bin_centers ** 2)).sum()
        norm_q = (histogram * (dequantized ** 2)).sum()
        denom = torch.sqrt(norm_x * norm_q) + 1e-12
        cos_sim = dot / denom
        return (1.0 - cos_sim).item()
```

**Exact discrete identities:**

$$\mathrm{dot}=\langle\mathbf{x},\mathbf{q}\rangle_H,\quad
\mathrm{norm}_x=\|\mathbf{x}\|_H^2,\quad
\mathrm{norm}_q=\|\mathbf{q}\|_H^2$$

$$L_{\mathrm{cos}}=1-\frac{\mathrm{dot}}{\sqrt{\mathrm{norm}_x\,\mathrm{norm}_q}+10^{-12}}$$

Properties:

1. **Scale invariance in a joint sense:** If \(\mathbf{x}\) and \(\mathbf{q}\) are scaled by the same positive constant, cosine is unchanged. Under `scaled=True` quantization, changing \(\Delta\) changes the **relative** mapping of \(x_i\) into the FP8 lattice; cosine tracks that relative distortion.
2. **Range:** For non-negative \(x_i,q_i\) (true for magnitude histograms), \(\cos_H\in[0,1]\) in exact arithmetic when norms are positive; loss \(\in[0,1]\).
3. **Epsilon:** \(10^{-12}\) prevents division by zero if \(\mathbf{q}=\mathbf{0}\) (pathological \(\Delta\le 0\) already returns zeros earlier).

### 8.3 Geometric Picture

Think of \(\mathbf{x}\) and \(\mathbf{q}\) as points in the positive orthant of \(\mathbb{R}^B\) with the inner product \(\langle\cdot,\cdot\rangle_H\). Then:

$$\cos_H = \cos\theta_H$$

where \(\theta_H\) is the angle between them in that inner-product space. Minimizing \(L_{\mathrm{cos}}\) is maximizing alignment of reconstructed magnitudes with original magnitudes **under importance measure \(H\)**.

MSE instead minimizes the squared Euclidean distance \(\|\mathbf{q}-\mathbf{x}\|_H^2\) in the **same** inner-product space:

$$L_{\mathrm{MSE}}(\Delta)=\|\mathbf{q}(\Delta)-\mathbf{x}\|_H^2
  =\|\mathbf{x}\|_H^2+\|\mathbf{q}\|_H^2-2\langle\mathbf{x},\mathbf{q}\rangle_H.$$

§12 expands this identity into a rigorous MSE↔cosine comparison.

### 8.4 Multi-Stage Search — `find_optimal_amax` (lines 215–252)

Same refinement skeleton as V4 MSE search:

- Initialize \(\ell=w_{\max} r_{\mathrm{lo}}\), \(h=w_{\max} r_{\mathrm{hi}}\).
- For \(t=0,\ldots,T\): evaluate \(N\) linspace candidates; keep \(\Delta\) with minimal \(L_{\mathrm{cos}}\).
- Narrow: \(w=(h-\ell)/4\), \(\ell\leftarrow\max(0.1 w_{\max},\Delta^*-w)\), \(h\leftarrow\min(1.2 w_{\max},\Delta^*+w)\).

Defaults in V5 top-level: \(N=1000\), \(T=10\) → \(11{,}000\) objective evaluations per layer.

```236:252:histogram/weighted_histogram_cosine_v5.py
        for iteration in range(refinement_iterations + 1):
            candidates = torch.linspace(low, high, num_candidates, device=self.device)
            
            for amax_tensor in candidates:
                amax = amax_tensor.item()
                loss = self.compute_weighted_cosine(histogram, bin_centers, amax, scaled=scaled, loss_type=loss_type)
                
                if loss < min_loss:
                    min_loss = loss
                    best_amax = amax
            
            if iteration < refinement_iterations:
                range_width = (high - low) / 4
                low = max(max_val * 0.1, best_amax - range_width)
                high = min(max_val * 1.2, best_amax + range_width)
        
        return best_amax
```

---

## 9. Top-Level API — `HSWQWeightedHistogramOptimizerV5`

### 9.1 Constructor

```337:349:histogram/weighted_histogram_cosine_v5.py
    def __init__(self, bins: int = 8192, num_candidates: int = 1000, refinement_iterations: int = 10, device: str = "cuda", alpha: float = 0.7, beta: float = 0.3, loss_type: str = None):
        self.bins = bins
        self.num_candidates = num_candidates
        self.refinement_iterations = refinement_iterations
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type if loss_type is not None else DEFAULT_LOSS_TYPE
        self.cosine_optimizer = CosineOptimizer(device)
        print(
            f"[HSWQ V5] Optimizer initialized on {device} "
            f"(alpha={alpha}, beta={beta}, loss={self.loss_type})"
        )
```

### 9.2 `compute_optimal_amax`

1. If `use_svd_leverage` and `weight.ndim >= 2`: compute hybrid \(\mathbf{S}\).
2. If 1D `importance` present: broadcast and **multiply** \(\alpha^{\mathrm{final}}=S\odot I_{\mathrm{exp}}\).
3. Cache `_last_combined_importance` (avoids second SVD in stats path).
4. Build histogram; run cosine search.

### 9.3 `compute_optimal_amax_with_stats`

Returns:

| Key | Meaning |
|-----|---------|
| `optimal_amax` | \(\Delta^*\) |
| `max_val` | \(w_{\max}\) |
| `compression_ratio` | \(\Delta^*/w_{\max}\) |
| `estimated_cosine` | \(L_{\mathrm{cos}}(\Delta^*)\) (loss, not similarity) |
| `estimated_loss` | Alias of `estimated_cosine` |
| `loss_type` | `"cosine"` |

**Naming caution:** `estimated_cosine` is the **loss** \(1-\cos_H\), not the similarity itself.

---

## 10. Self-Test

```443:480:histogram/weighted_histogram_cosine_v5.py
if __name__ == "__main__":
    print("HSWQ V5: Hybrid SVD-Magnitude + Cosine Loss amax search - Self Test")
    ...
    weight = U_true @ V_true.T
    weight[5, 5] = 20.0
    weight[10, 100] = -25.0
    ...
    result_v1 = optimizer.compute_optimal_amax_with_stats(..., use_svd_leverage=False)
    result_v2 = optimizer.compute_optimal_amax_with_stats(..., use_svd_leverage=True)
```

Compares uniform-importance cosine search vs hybrid-aware cosine search on a rank-16 matrix with injected outliers. Differences in \(\Delta^*\) demonstrate that importance reweighting moves the cosine optimum — the same structural lesson as V4’s MSE self-test, under a different objective.

---

## 11. Amax Trade-Off Under Cosine (Operational Intuition)

For a fixed FP8 lattice \(\mathcal{G}\):

| \(\Delta\) choice | Effect on \(q(x,\Delta)\) | Typical cosine effect |
|-------------------|---------------------------|------------------------|
| Larger \(\Delta\) (near \(w_{\max}\)) | Less clipping of outliers; coarser effective step for small \(x\) under scaled mode; under non-scaled mode, small values keep native E4M3 spacing but outliers stay representable until 448 | Protects tail bins; may misalign dense bulk bins if scaled mode stretches steps |
| Smaller \(\Delta\) | Aggressive clip: large \(x_i\) collapse toward \(\Delta\) then to grid | Bulk bins may align better; tail bins collapse → \(\mathbf{q}\) loses high-\(x\) direction |

Cosine loss is sensitive to **relative pattern** across bins. Destroying the ordering / relative heights of high-\(H\) bins hurts \(\cos_H\) even if absolute squared errors look acceptable.

---

## 12. Mathematical Comparison: Cosine Loss vs Weighted MSE

This section is the core comparison requested against V4 MSE. All statements are with respect to the **same** histogram \(H\), bin centers \(\mathbf{x}\), and FP8 map \(q(\cdot,\Delta)\). Only the scalar objective differs.

### 12.1 Common Geometry

Work in the real Hilbert space \(\mathbb{R}^B\) equipped with

$$\langle \mathbf{a},\mathbf{b}\rangle_H=\sum_i H(i)\,a_i b_i.$$

Define \(\mathbf{q}(\Delta)=(q(x_0,\Delta),\ldots,q(x_{B-1},\Delta))\). Both optimizers see the same family of curves \(\{\mathbf{q}(\Delta)\}_{\Delta>0}\) in this space.

### 12.2 Exact Algebraic Bridge

Expand MSE in the \(H\)-inner product:

\begin{align}
L_{\mathrm{MSE}}(\Delta)
&= \|\mathbf{q}-\mathbf{x}\|_H^2 \\
&= \|\mathbf{x}\|_H^2 + \|\mathbf{q}\|_H^2 - 2\langle\mathbf{x},\mathbf{q}\rangle_H.
\end{align}

Cosine loss:

\begin{align}
L_{\mathrm{cos}}(\Delta)
&= 1 - \frac{\langle\mathbf{x},\mathbf{q}\rangle_H}{\|\mathbf{x}\|_H\|\mathbf{q}\|_H} \\
&= 1 - \cos\theta_H(\Delta).
\end{align}

Introduce normalized vectors \(\hat{\mathbf{x}}=\mathbf{x}/\|\mathbf{x}\|_H\) and \(\hat{\mathbf{q}}=\mathbf{q}/\|\mathbf{q}\|_H\). Then:

$$L_{\mathrm{cos}}=1-\langle\hat{\mathbf{x}},\hat{\mathbf{q}}\rangle_H
  =\tfrac12\|\hat{\mathbf{x}}-\hat{\mathbf{q}}\|_H^2.$$

So **cosine loss is exactly half the squared Euclidean distance between unit vectors** in the \(H\)-geometry. MSE is the squared distance between **unnormalized** vectors.

### 12.3 When MSE and Cosine Agree (Necessary Conditions)

Suppose \(\|\mathbf{q}(\Delta)\|_H\) is constrained to equal \(\|\mathbf{x}\|_H\) for all candidate \(\Delta\) (isometry / fixed-norm reconstructs). Then:

$$L_{\mathrm{MSE}}
  = 2\|\mathbf{x}\|_H^2\bigl(1-\cos\theta_H\bigr)
  = 2\|\mathbf{x}\|_H^2\, L_{\mathrm{cos}}.$$

Under a **constant-norm constraint**, \(\arg\min L_{\mathrm{MSE}}=\arg\min L_{\mathrm{cos}}\).  
FP8 quantize–dequantize **does not** enforce \(\|\mathbf{q}\|_H=\|\mathbf{x}\|_H\): clipping and rounding change energy. Therefore the argmins **generally differ**.

### 12.4 How \(\Delta\) Moves \(\|\mathbf{q}\|_H\)

Qualitative regimes (non-scaled FP8, typical heavy-tailed weights):

1. **\(\Delta\) too small:** Large \(x_i\) clipped → \(\|\mathbf{q}\|_H\) collapses; \(\mathbf{q}\) becomes flatter / truncated → both MSE and cosine worsen, but cosine especially penalizes loss of high-bin direction.
2. **\(\Delta\) moderate:** Trade-off between clip error on tails and rounding error on bulk.
3. **\(\Delta\) too large (scaled mode):** Mapping \(x\mapsto x\cdot 448/\Delta\) compresses dynamic use of the mantissa grid for the bulk → \(\mathbf{q}\) can stay closer in absolute value on tails while **angularly** drifting on the dense region that owns most \(H\)-mass.

MSE’s extra terms \(\|\mathbf{q}\|_H^2\) and cross term make it prefer \(\Delta\) that reduce **absolute** residuals, even if that scales the whole reconstruct. Cosine **quotients out** joint scale and focuses on shape.

### 12.5 Decomposition: Radial vs Angular Error

Any reconstruct admits the polar decomposition in \(H\)-space:

$$\mathbf{q}
  = \|\mathbf{q}\|_H\cdot\hat{\mathbf{q}}
  = \underbrace{\|\mathbf{q}\|_H}_{\text{radial / energy}}
    \cdot
    \underbrace{\hat{\mathbf{q}}}_{\text{angular / shape}}.$$

Then:

$$L_{\mathrm{MSE}}
  = \|\mathbf{x}\|_H^2 + \|\mathbf{q}\|_H^2
    - 2\|\mathbf{x}\|_H\|\mathbf{q}\|_H\cos\theta_H.$$

Fix \(\theta_H\) and vary radial gain \(r=\|\mathbf{q}\|_H\):

$$L_{\mathrm{MSE}}(r)=\|\mathbf{x}\|_H^2 + r^2 - 2\|\mathbf{x}\|_H r\cos\theta_H.$$

This quadratic in \(r\) is minimized at \(r^*=\|\mathbf{x}\|_H\cos\theta_H\) (for \(\cos\theta_H>0\)), with minimum value \(\|\mathbf{x}\|_H^2\sin^2\theta_H\).

**Consequence:** MSE can improve by changing **energy** \(r\) even when the angle \(\theta_H\) is fixed or slightly worse. Cosine **ignores** \(r\) entirely and only sees \(\theta_H\).

For FP8 amax search, many \(\Delta\) moves are **approximately radial** (global stretch of reconstruct magnitudes under scaled quantization) mixed with **angular** distortions (bin-dependent rounding/clip). Cosine isolates the angular part — the part that corresponds to “does the importance-weighted magnitude **profile** still look like the original?”

### 12.6 Scale Invariance and PTQ Semantics

Let \(\lambda>0\). For a pure scaling of the weight tensor \(\mathbf{W}\mapsto\lambda\mathbf{W}\):

- Histogram centers scale: \(x_i\mapsto\lambda x_i\), \(w_{\max}\mapsto\lambda w_{\max}\).
- Optimal thresholds typically scale: \(\Delta^*(\lambda\mathbf{W})\approx\lambda\Delta^*(\mathbf{W})\) under homogeneous \(q\) modes.

Cosine similarity of \((\mathbf{x},\mathbf{q})\) is invariant to replacing \((\mathbf{x},\mathbf{q})\) by \((\lambda\mathbf{x},\lambda\mathbf{q})\). MSE scales as \(\lambda^2\). Therefore:

- Cosine loss values are **comparable across layers** without renormalization.
- MSE values are **not** comparable across layers with different \(\|\mathbf{W}\|_F\) without dividing by \(\|\mathbf{x}\|_H^2\) or similar.

V4 already normalizes \(H\) to sum to 1, but \(L_{\mathrm{MSE}}\) still scales with the square of typical magnitudes. \(L_{\mathrm{cos}}\) does not. This matters when analyzing optimizer diagnostics (`estimated_mse` vs `estimated_cosine`) across heterogeneous NextDiT layers.

### 12.7 Outlier Sensitivity

Consider a single high bin \(i_\star\) with center \(x_\star\gg x_{\mathrm{bulk}}\) and importance mass \(H_\star\).

**MSE contribution** of that bin after clipping to \(\Delta\ll x_\star\):

$$H_\star\,(q_\star-x_\star)^2 \approx H_\star\,(\Delta-x_\star)^2 \sim H_\star\, x_\star^2.$$

One outlier bin with moderate \(H_\star\) can dominate the entire MSE sum if \(x_\star\) is huge — classic heavy-tail pathology that pushes \(\Delta^*\) upward even when that bin is **not** structurally important (unless importance already down-weights it).

**Cosine contribution** is softer: the same clip changes \(\langle\mathbf{x},\mathbf{q}\rangle_H\) by about \(H_\star x_\star\Delta\) instead of \(H_\star x_\star^2\), and also changes \(\|\mathbf{q}\|_H\). Relative influence of the outlier is order \(H_\star x_\star / \|\mathbf{x}\|_H\) on the normalized vectors, not order \(H_\star x_\star^2\).

With hybrid importance, structural outliers in non-principal directions already receive lower \(S_{m,n}\). Cosine **additionally** resists letting residual absolute tail error blackmail the threshold away from bulk alignment.

### 12.8 Information / Coding View of the FP8 Grid

Treat the positive E4M3 grid as a nonuniform codebook \(\mathcal{G}^+\). For a fixed \(\Delta\), each bin center \(x_i\) is encoded as \(q_i\in\mathcal{G}^+\) (after clip/scale).

- **MSE** is the classical **mean distortion** of a quantizer under measure \(H\):
  $$D_{\mathrm{MSE}}=\mathbb{E}_{i\sim H}\bigl[(q_i-x_i)^2\bigr].$$
- **Cosine loss** is one minus the **correlation** between the original and coded sequences under the same measure:
  $$D_{\mathrm{cos}}=1-\frac{\mathbb{E}_H[xq]}{\sqrt{\mathbb{E}_H[x^2]\mathbb{E}_H[q^2]}}.$$

In rate–distortion terms, MSE is \(L^2\) distortion. Cosine is a **normalized correlation distortion**. For generative model weights, preserving the **shape** of the importance-weighted magnitude law often correlates more tightly with retaining singular-mode energy ratios than minimizing raw \(L^2\) on absolute values — because forward maps care about relative channel/element contributions (angles in activation space), not an arbitrary global scale that LayerNorm / RMSNorm / residual scales may absorb.

### 12.9 Effect of Importance Weighting on Both Objectives

Importance enters **only** through \(H\). Both V4 and V5 use the same conceptual chain “hybrid \(\mathbf{S}\) → \(H\)”. Therefore:

- Switching V4↔V5 does **not** by itself change which elements are important.
- It changes **which mismatch on those elements** is penalized: absolute vs angular.

If hybrid importance is wrong, both fail. If hybrid importance is right, cosine asks Stage 3 for a threshold that keeps the **shape** of important magnitudes, while MSE asks for minimal **absolute** residual.

### 12.10 Monotonicity and Multiple Local Minima

Neither \(L_{\mathrm{MSE}}(\Delta)\) nor \(L_{\mathrm{cos}}(\Delta)\) is guaranteed unimodal in \(\Delta\) for discrete FP8 grids. Rounding induces piecewise-constant plateaus and jumps when bin centers cross Voronoi boundaries of \(\mathcal{G}\). Both V4 and V5 rely on dense multi-stage grid search rather than gradient descent for this reason. Cosine does not remove multimodality; it changes the landscape’s valleys.

### 12.11 Signed Weights vs Magnitude Histograms (Shared Limitation)

Both V4 and V5 build \(H\) on \(|W|\). Neither directly optimizes:

$$\cos\bigl(\mathrm{vec}(\mathbf{W}),\,\mathrm{vec}(q(\mathbf{W},\Delta))\bigr).$$

A full signed cosine would couple positive and negative lobes and could prefer thresholds that cancel errors across signs. HSWQ deliberately avoids that. Comparison “cosine vs MSE” in this guide means **histogram-magnitude cosine vs histogram-magnitude MSE**, matching the two source files.

### 12.12 Summary Table — Objective Calculus

| Property | Weighted MSE (V4) | Weighted Cosine Loss (V5) |
|----------|-------------------|---------------------------|
| Formula | \(\|\mathbf{q}-\mathbf{x}\|_H^2\) | \(1-\langle\hat{\mathbf{x}},\hat{\mathbf{q}}\rangle_H\) |
| Penalizes radial (energy) error | Yes | No (invariant to joint scale) |
| Penalizes angular (shape) error | Yes (mixed with radial) | Yes (pure) |
| Scale of loss across layers | \(\propto\) magnitude\(^2\) | Dimensionless, in \([0,1]\) |
| Outlier absolute leverage | \(\sim H_\star x_\star^2\) | \(\sim H_\star x_\star/\|\mathbf{x}\|_H\) (normalized) |
| Argmin vs other | Differs unless \(\|\mathbf{q}\|_H\) fixed | Differs unless \(\|\mathbf{q}\|_H\) fixed |
| Matches FP8 grid fidelity notion | Absolute reconstruct error | Directional magnitude-profile fidelity |

---

## 13. Quantization-Accuracy Advantages of Cosine (Detailed)

“Accuracy” here means **faithfulness of the quantized layer as a linear map / energy distributor**, not a claim that cosine always raises SSIM. SSIM is an end-to-end metric depending on Stage 2 and the rest of the stack. The advantages below are **Stage-3 / local quantization** advantages that follow from the mathematics in §12.

### 13.1 Advantage A — Separation From Absorbable Global Scale

Many diffusion transformer blocks apply normalization and residual scaling around Linear layers. A near-global gain error on a weight matrix is often **partially absorbable** by surrounding scales, whereas **relative** distortion among important weights is not.

Cosine’s scale invariance (§12.5–§12.6) aligns the Stage-3 objective with “preserve relative importance-weighted magnitudes,” which is closer to preserving the action of \(\mathbf{W}\) **up to a scalar**, i.e. preserving the projective geometry of rows/columns in the magnitude histogram sense.

MSE treats a global gain error as first-class damage equal to shape error of the same \(L^2\) size. For PTQ threshold search, that over-penalizes radial mismatch and can select \(\Delta\) that are worse for shape.

### 13.2 Advantage B — Resistance to Tail Absolutism

Heavy-tailed layers (high kurtosis, high `abs_max`) are exactly the layers where VETO / narrow search ranges appear in HSWQ pipelines. When a layer **is** quantized, MSE’s \(x^2\) weighting on residual tails (§12.7) still biases \(\Delta^*\) toward protecting absolute outliers.

Cosine reduces that blackmail. Combined with hybrid importance (outliers in non-principal directions already down-weighted), Stage 3 can keep more FP8 resolution on the **structurally energetic bulk**.

### 13.3 Advantage C — Compatibility With SVD-Centric Importance

Hybrid importance is built from singular modes. Singular modes define **directions**. An objective that scores **angular** fidelity of the magnitude profile is philosophically aligned with “protect principal subspace geometry,” whereas pure MSE is aligned with “minimize entrywise energy of the error tensor.”

More formally, let \(\mathbf{E}=q(\mathbf{W})-\mathbf{W}\) (elementwise, after a chosen \(\Delta\)). MSE on the histogram approximates a reweighted \(\|\mathbf{E}\|_F^2\) surrogate. Cosine approximates preservation of the **correlation** between \(|\mathbf{W}|\) and \(|q(\mathbf{W})|\) under importance. Correlation preservation is a tighter proxy for keeping the **ranking and relative sizes** of important magnitudes — the quantities that determine which singular directions remain strong after quantization.

### 13.4 Advantage D — Cross-Layer Diagnostics

Because \(L_{\mathrm{cos}}\in[0,1]\) roughly, `estimated_cosine` can be compared across layers as a **pure fidelity score**. `estimated_mse` cannot without normalization by \(\|\mathbf{x}\|_H^2\). This matters for profiling, regression tests, and choosing per-layer search ranges.

### 13.5 Advantage E — Scaled FP8 Mode Semantics

Under `scaled=True`, \(\Delta\) primarily sets the **gain** \(s=448/\Delta\) that maps weights into the E4M3 cube. Changing \(\Delta\) is almost a **radial** control in reconstruct space, with secondary angular effects from clip and rounding. Cosine is designed to see those secondary effects clearly; MSE confounds them with the intentional gain change.

Even under `scaled=False` (HSWQ V1 production), clip boundaries create a piecewise radial collapse of the tail; cosine still focuses on whether the surviving bulk profile remains aligned.

### 13.6 Advantage F — Softmax / Attention Adjacent Intuition

Although Stage 3 never runs attention, Linear layers feeding Q/K/V are sensitive to **relative** row scales. Softmax is shift-invariant in logits but **not** invariant to arbitrary per-channel stretches that destroy relative geometry. Preserving magnitude **shape** among important weights is closer to preserving those relative geometries than minimizing absolute MSE on weight entries.

### 13.7 Non-Advantages / Honesty Bounds

Cosine is **not** universally superior:

1. If the deployment loss truly is entrywise \(L^2\) (e.g. some classic PTQ papers’ proxy), MSE is the matching objective.
2. If importance weights are uniform and weights are light-tailed, argmins often nearly coincide (§12.3).
3. Cosine on **magnitude histograms** still ignores sign errors; it will not fix a bad signed rounding model.
4. End-to-end SSIM can still be dominated by Stage 2 mistakes (V4 guide §0.5).

V5’s advantage is conditional: **when Stage 2 is correct and the goal is importance-weighted magnitude-profile fidelity under a physical FP8 grid, cosine is the more surgically correct Stage-3 risk functional than MSE.**

### 13.8 Practical Decision Rule

| Situation | Prefer |
|-----------|--------|
| Need minimal entrywise reconstruct error; reporting MSE proxy | V4 MSE |
| Heavy tails; hybrid SVD importance; care about bulk/profile fidelity | **V5 Cosine** |
| Comparing optimizer quality across layers with different scales | **V5 Cosine** |
| Debugging absolute cast error in units of weight | V4 MSE |

---

## 14. End-to-End Formula Chain (V5)

$$\mathbf{W}\xrightarrow{\mathrm{SVD}}\mathbf{U},\boldsymbol{\sigma},\mathbf{V}_h$$

$$L_{m,n}=\sum_k U_{mk}^2\sigma_k^2 V_{h,kn}^2$$

$$M_{m,n}=W_{m,n}^2$$

$$\hat L=L/\|L\|_2,\;\hat M=M/\|M\|_2$$

$$S_{\mathrm{raw}}=\alpha\hat L+\beta\hat M$$

$$S=0.5+0.5\,S_{\mathrm{raw}}/\overline{S_{\mathrm{raw}}}$$

$$\alpha^{\mathrm{final}}=S\odot I_{\mathrm{exp}}\quad(\text{optional})$$

$$H\leftarrow\mathrm{hist}(|W|;\alpha^{\mathrm{final}})$$

$$q_i=q(x_i,\Delta)$$

$$\Delta^*=\arg\min_\Delta\Bigl(1-\frac{\sum_i H(i)x_i q_i}{\sqrt{(\sum_i H(i)x_i^2)(\sum_i H(i)q_i^2)}}\Bigr)$$

---

## 15. Formula Index

| Formula | Section |
|---------|---------|
| \(L_{\mathrm{cos}}=1-\langle\mathbf{x},\mathbf{q}\rangle_H/(\|\mathbf{x}\|_H\|\mathbf{q}\|_H)\) | §1.1, §8.2 |
| \(L_{\mathrm{MSE}}=\|\mathbf{q}-\mathbf{x}\|_H^2\) | §12.2 |
| \(L_{\mathrm{cos}}=\tfrac12\|\hat{\mathbf{x}}-\hat{\mathbf{q}}\|_H^2\) | §12.2 |
| \(L_{\mathrm{MSE}}=2\|\mathbf{x}\|_H^2 L_{\mathrm{cos}}\) if \(\|\mathbf{q}\|_H=\|\mathbf{x}\|_H\) | §12.3 |
| \(L^{\mathrm{V5}}_{m,n}=\sum_k U_{mk}^2\sigma_k^2 V_{h,kn}^2\) | §5.2 |
| \(L^{\mathrm{V4}}_{m,n}=r_m c_n\) (outer product; cross terms) | §5.2 |
| \(M_{m,n}=W_{m,n}^2\) | §4 |
| \(S=0.5+0.5\,S_{\mathrm{norm}}\) | §6.3 |
| \(H(i)\) scatter-add normalization | §7.1 |
| \(q\) scaled / non-scaled | §3.3 |
| Search refinement \(w=(h-\ell)/4\) | §8.4 |

---

## 16. Component Table (Audit)

| Component | Responsibility |
|-----------|----------------|
| **`compute_hybrid_leverage_scores`** | SVD → mode-coupled \(\sigma^2\) leverage GEMM; RMS; L2; \(\alpha/\beta\); mean norm; 0.5 baseline. |
| **`FP8E4M3Quantizer`** | Physical E4M3 grid; scaled/non-scaled \(q\); nearest rounding. |
| **`WeightedHistogram`** | Importance-weighted \(|W|\) histogram in float64. |
| **`CosineOptimizer`** | \(L_{\mathrm{cos}}(\Delta)\); iterative refinement search. |
| **`HSWQWeightedHistogramOptimizerV5`** | Full Stage-3 composition; stats API; SVD cache. |

---

## 17. Design Principles

1. **Objective–geometry match.** Stage 3 should optimize the risk functional that matches the fidelity notion you care about. V5 chooses angular fidelity of importance-weighted magnitudes.
2. **Importance first, objective second.** Cosine without hybrid importance is still a uniform-bin shape match; hybrid importance focuses that shape match on structurally energetic elements.
3. **Physical FP8 grid.** Same byte-derived \(\mathcal{G}\) as V4 — optimizer error matches runtime cast.
4. **No silent MSE fallback.** `loss_type` must be `"cosine"`; the module refuses to pretend to be V4.
5. **Honest leverage form.** V5’s GEMM leverage is documented as mode-coupled, not silently equated to V4’s outer-product form.

---

## 18. Relationship to the V4 Technical Guide

Readers should treat:

- `md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md` as the reference for **MSE Stage 3**, DualMonitor/Stage-2 history, and outer-product leverage.
- **This document** as the reference for **Cosine Stage 3**, mode-coupled leverage as implemented in `weighted_histogram_cosine_v5.py`, and the MSE↔cosine mathematical comparison.

Shared subsystems (FP8 grid construction, histogram binning, refinement search scaffolding, \(\alpha/\beta\) blend philosophy) are intentionally parallel so the two guides can be read as a pair.

---

## 19. Closing Summary

HSWQ V5 keeps Hybrid Sensitivity Weighted Quantization’s Stage-3 architecture — per-element SVD+RMS importance feeding a high-resolution weighted histogram of absolute weights — and replaces the amax risk functional with importance-weighted cosine loss against a physical FP8 E4M3 quantize–dequantize map.

Relative to V4 MSE, cosine loss:

- equals half the squared distance between **unit** vectors in \(H\)-space;
- ignores absorbable radial gain and focuses on magnitude-profile angle;
- reduces absolute-tail blackmail of \(\Delta^*\);
- yields dimensionless, cross-layer-comparable diagnostics;
- aligns more cleanly with SVD-centric notions of structural fidelity.

It does not replace Stage 2, does not remove the need for hybrid importance, and does not guarantee higher SSIM in isolation. It is the correct Stage-3 instrument when the quantization-accuracy target is **importance-weighted magnitude-distribution fidelity under FP8**, stated and derived with the same rigor as the V4 MSE guide.
