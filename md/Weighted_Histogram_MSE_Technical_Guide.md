# Weighted Histogram MSE — Technical Guide

**Source:** `histogram/weighted_histogram_mse.py`  
This document is a complete, line-by-line technical explanation of the HSWQ weighted histogram MSE optimization module used to find the optimal clipping threshold (amax) that minimizes quantization error under importance weighting.

---

## 1. Overview and Core Formula

### 1.1 Optimization Objective

The module implements the HSWQ optimization objective (see module docstring lines 7–13):

$$\Delta^* = \arg\min_\Delta \sum_{i=0}^{B-1} H(i) \cdot \bigl(q(x_i,\, \Delta) - x_i\bigr)^2$$

Where \(B\) is the number of bins. In code this is expressed as:

```7:13:histogram/weighted_histogram_mse.py
Core formula:
    Δ* = argmin_Δ Σ_i H(i) · (q(x_i, Δ) - x_i)²

Where:
    - H(i): weighted histogram (weighted by input importance I_c)
    - q(x, Δ): quantize-dequantize function
    - Δ: clipping value (amax)
```

### 1.2 Notation

| Symbol | Meaning |
|--------|--------|
| \(\Delta\) (amax) | Clipping threshold; the single parameter we optimize. |
| \(B\) | Number of histogram bins (`bins`). |
| \(H(i)\) | **Weighted histogram** at bin \(i\): after normalization, \(\sum_{i=0}^{B-1} H(i) = 1\). Derived from input importance \(I_c\) (per-channel). |
| \(q(x, \Delta)\) | Quantize–dequantize function: clip by \(\Delta\), round to FP8 grid, return dequantized value. |
| \(x_i\) | Representative value for bin \(i\) (bin center). |
| \(I_c\) | Per-channel importance (e.g. input mean absolute value); \(\alpha_{k,c}=I_c\) for weight element in channel \(c\). |

So we search for the **amax** that minimizes the **importance-weighted** mean squared error between original and quantized weights.

### 1.3 Component Summary

| Component | Responsibility |
|-----------|----------------|
| **FP8E4M3Quantizer** | Physical FP8 E4M3 grid; \(q(x,\Delta)\) with scaled / non-scaled modes. |
| **WeightedHistogram** | Build \(H(i)\) from weight tensor and optional \(I_c\); normalize; provide \(x_i\). |
| **MSEOptimizer** | Compute \(\sum_i H(i)(q(x_i,\Delta)-x_i)^2\); search \(\Delta\) (linear candidates + refinement). |
| **HSWQWeightedHistogramOptimizer** | Compose: build histogram → find optimal amax (and optional stats). |

---

## 2. FP8E4M3Quantizer

### 2.1 Role and FP8 E4M3 Spec

The quantizer simulates real FP8 E4M3 behavior so that MSE in this module matches runtime behavior. Spec from class docstring:

```26:36:histogram/weighted_histogram_mse.py
class FP8E4M3Quantizer:
    """
    Accurate quantize-dequantize simulator for FP8 E4M3 format.

    FP8 E4M3 spec:
        - Sign: 1 bit
        - Exponent: 4 bits (bias = 7)
        - Mantissa: 3 bits
        - Range: ±[2^-6, 448] (incl. denormals)
        - Special: NaN (0x7F, 0xFF), ±0
    """
```

- **Representable range:** \(\pm[2^{-6},\, 448]\) (including denormals).
- **Normalized values:** \(2^{e-7}(1 + m/8)\), \(e\in[1,15]\), \(m\in[0,7]\).
- **Denormals:** \(2^{-6}(m/8)\), \(m\in[1,7]\) (comment lines 38–39).

### 2.2 Grid Construction — `_build_fp8_grid` (lines 46–60)

**Formula:** The grid is the set of all distinct float32 values obtained by interpreting each byte in \([0,255]\) as `torch.float8_e4m3fn` and converting to float. NaNs and duplicates are removed; positives are sorted.

**Code (exact):**

```46:60:histogram/weighted_histogram_mse.py
    def _build_fp8_grid(self):
        """Build full representable positive FP8 E4M3 grid (PyTorch native behavior)."""
        # All byte patterns (0-255) on device to avoid transfer cost
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

- `max_representable = 448.0` (FP8 E4M3 maximum positive value).
- This **physical grid** ensures rounding matches PyTorch's actual FP8 behavior.

### 2.3 Quantize–Dequantize — `quantize_dequantize` (lines 62–92)

**Signature:** `quantize_dequantize(values, amax, scaled=True)`. Implements \(q(x,\,\Delta)\).

**Mathematical definition:**

- **scaled=True (V2):**
  - \(s = 448 / \Delta\)
  - \(x' = \mathrm{clip}(x \cdot s,\; -448,\; 448)\)
  - \(q(x,\Delta) = \mathrm{round}_{\mathrm{FP8}}(x') / s\)
- **scaled=False (V1, standard-compatible):**
  - \(x' = \mathrm{clip}(x,\; -\Delta,\; \Delta)\), then \(\mathrm{clip}(x', -448, 448)\)
  - \(q(x,\Delta) = \mathrm{round}_{\mathrm{FP8}}(x')\) (no scale; output in same units as input)

**Code (scaled branch):**

```81:87:histogram/weighted_histogram_mse.py
        if scaled:
            scale = self.max_representable / amax
            scaled_vals = values * scale
            scaled_vals = scaled_vals.clamp(-self.max_representable, self.max_representable)
            quantized = self._round_to_fp8_grid(scaled_vals)
            dequantized = quantized / scale
            return dequantized
```

**Code (non-scaled branch):**

```88:92:histogram/weighted_histogram_mse.py
        else:
            clipped = values.clamp(-amax, amax)
            clipped = clipped.clamp(-self.max_representable, self.max_representable)
            dequantized = self._round_to_fp8_grid(clipped)
            return dequantized
```

Guard: if `amax <= 0`, returns zeros (lines 78–79).

### 2.4 Rounding to FP8 Grid — `_round_to_fp8_grid` (lines 94–109)

**Formula:** For each absolute value \(a \ge 0\), find the nearest positive grid point:
\(\mathrm{round}_{\mathrm{FP8}}(a) = \mathrm{argmin}_{g \in \textit{positive\_grid}} |a - g|\), then reapply sign.

**Code (core):**

```94:109:histogram/weighted_histogram_mse.py
    def _round_to_fp8_grid(self, values: torch.Tensor) -> torch.Tensor:
        """Round values to nearest FP8 grid point."""
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

- Batching by 10000 elements limits memory for large tensors.
- Distance matrix: `batch` shape `(N,)`, `_positive_grid` shape `(G,)` → `distances` shape `(N, G)`; `argmin(dim=1)` gives index of nearest grid point per element.

### 2.5 Single-Value Error — `compute_quantization_error` (lines 111–115)

**Formula:** \(\mathrm{error}(x,\,\Delta) = |q(x,\,\Delta) - x|\).

```111:115:histogram/weighted_histogram_mse.py
    def compute_quantization_error(self, value: float, amax: float, scaled: bool = True) -> float:
        """Compute quantization error for a single value."""
        val_tensor = torch.tensor([value], device=self.device)
        dequant = self.quantize_dequantize(val_tensor, amax, scaled=scaled)
        return (dequant - val_tensor).abs().item()
```

Used for debugging; the main optimization uses the full histogram and bin centers.

---

## 3. WeightedHistogram

### 3.1 Role and Formula

From the class docstring (lines 118–123): \(\alpha_{k,c} = I_c\); \(H(b) = \sum \alpha_{k,c}\) over all weight elements \((k,c)\) whose bin is \(b\).

```118:123:histogram/weighted_histogram_mse.py
class WeightedHistogram:
    """
    HSWQ spec-compliant weighted histogram.
    α_{k,c} = I_c; H(b) = Σ α_{k,c} over bin_b.
    Counts sum of importance per bin (not frequency).
    """
```

So the histogram is **not** a count of elements per bin; it is the **sum of importance** per bin. After building, it is **normalized** so that \(\sum_{i=0}^{B-1} H(i) = 1\).

### 3.2 Constructor and State (lines 125–131)

```125:131:histogram/weighted_histogram_mse.py
    def __init__(self, bins: int = 4096, device: str = "cuda"):
        """Args: bins (affects precision), device."""
        self.bins = bins
        self.device = device
        self.histogram = None
        self.max_val = 0.0
        self.total_weight = 0.0
```

- `bins` \(= B\): number of histogram bins (e.g. 4096).
- `histogram`, `max_val`, `total_weight` are set in `build()`.

### 3.3 Building the Histogram — `build` (lines 133–177)

**Inputs:** `weight` (tensor), optional `importance` \(I_c\) (per-channel, shape \([I]\)).

**Step 1 — Preprocess and max:**

$$\text{max\_val} = \max_{k,c} |W_{k,c}|,\qquad \text{guard: if } \text{max\_val}=0 \text{ then set } 10^{-7}.$$

```135:139:histogram/weighted_histogram_mse.py
        weight = weight.detach().float().to(self.device)
        w_abs = weight.abs()
        self.max_val = w_abs.max().item()
        if self.max_val == 0:
            self.max_val = 1e-7  # Prevent division by zero
```

**Step 2 — Importance expansion:**

- **4D** (Conv2d \((O, I, K, K)\)): \(I_c\) is trimmed/padded to length \(I\), then expanded to shape \((1, I, 1, 1)\) and broadcast to `weight.shape`.
- **2D** (Linear \((O, I)\)): trim/pad to length \(I\), then `(1, -1)` and expand.
- If `importance is None` or other shape: \(\alpha \equiv 1\) (uniform).

**Code (4D and 2D):**

```144:165:histogram/weighted_histogram_mse.py
            if weight.dim() == 4:  # Conv2d: (Out, In, K, K)
                in_channels = weight.shape[1]
                if importance.numel() >= in_channels:
                    importance = importance[:in_channels]
                else:
                    # Padding
                    padding = torch.ones(in_channels - importance.numel(),
                                        device=self.device)
                    importance = torch.cat([importance, padding])
                imp_expanded = importance.view(1, -1, 1, 1).expand_as(weight)
                
            elif weight.dim() == 2:  # Linear: (Out, In)
                in_features = weight.shape[1]
                if importance.numel() >= in_features:
                    importance = importance[:in_features]
                else:
                    padding = torch.ones(in_features - importance.numel(),
                                        device=self.device)
                    importance = torch.cat([importance, padding])
                imp_expanded = importance.view(1, -1).expand_as(weight)
            else:
                imp_expanded = torch.ones_like(weight)
```

**Step 3 — Binning and scatter-add:**

- Bin width: \(\mathit{bin\_width} = \text{max\_val} / B\).
- Bin index for each element: \(b = \bigl\lfloor |W_{k,c}| / \mathit{bin\_width} \bigr\rfloor\), clamped to \([0,\, B-1]\).
- Raw histogram: \(H_{\mathrm{raw}}(b) = \sum_{(k,c):\ \text{bin}=b} \alpha_{k,c}\) (scatter-add).

**Code:**

```169:173:histogram/weighted_histogram_mse.py
        bin_width = self.max_val / self.bins
        bin_indices = (w_abs / bin_width).long().clamp(0, self.bins - 1)
        self.histogram = torch.zeros(self.bins, dtype=torch.float64, device=self.device)
        self.histogram.scatter_add_(0, bin_indices.reshape(-1), 
                                    imp_expanded.reshape(-1).double())
```

**Step 4 — Normalize:**

$$H(i) \leftarrow \frac{H_{\mathrm{raw}}(i)}{\sum_j H_{\mathrm{raw}}(j)},\qquad \sum_i H(i) = 1.$$

```175:177:histogram/weighted_histogram_mse.py
        self.total_weight = self.histogram.sum().item()
        if self.total_weight > 0:
            self.histogram = self.histogram / self.total_weight
```

### 3.4 Bin Centers and Histogram Getter

**Bin center formula:** \(x_i = (i + 0.5) \cdot \mathit{bin\_width}\) for \(i = 0,\,\ldots,\, B-1\), i.e. center of each bin in \([0,\,\text{max\_val}]\).

```179:188:histogram/weighted_histogram_mse.py
    def get_bin_centers(self) -> torch.Tensor:
        """Return center value of each bin."""
        bin_width = self.max_val / self.bins
        return torch.linspace(
            0.5 * bin_width,
            self.max_val - 0.5 * bin_width,
            self.bins,
            device=self.device,
            dtype=torch.float64
        )
```

**`get_histogram()`** (lines 190–192): Returns the normalized \(H(i)\) tensor (used by MSEOptimizer).

---

## 4. MSEOptimizer

### 4.1 Role

Finds \(\Delta^* = \arg\min_\Delta \sum_i H(i)(q(x_i,\Delta)-x_i)^2\) using the histogram and bin centers. Holds an `FP8E4M3Quantizer` instance for \(q\).

```195:203:histogram/weighted_histogram_mse.py
class MSEOptimizer:
    """
    HSWQ spec-compliant MSE optimizer.
    Δ* = argmin_Δ Σ_i H(i)·(q(x_i,Δ)-x_i)². Finds optimal amax given full quantization error.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fp8_quantizer = FP8E4M3Quantizer(device)
```

### 4.2 Weighted MSE — `compute_weighted_mse` (lines 205–217)

**Formula:** For a candidate \(\Delta = \text{amax}\),

$$\mathrm{MSE}(\Delta) = \sum_{i=0}^{B-1} H(i) \cdot \bigl(q(x_i,\,\Delta) - x_i\bigr)^2.$$

**Code:**

```205:217:histogram/weighted_histogram_mse.py
    def compute_weighted_mse(self, 
                             histogram: torch.Tensor,
                             bin_centers: torch.Tensor,
                             amax: float,
                             scaled: bool = True) -> float:
        """Compute weighted MSE for given amax. Returns Σ H(i)·(q(x_i,amax)-x_i)²."""
        dequantized = self.fp8_quantizer.quantize_dequantize(
            bin_centers.float(), amax, scaled=scaled
        ).double()
        error_sq = (dequantized - bin_centers) ** 2
        weighted_mse = (histogram * error_sq).sum().item()
        
        return weighted_mse
```

- `bin_centers` are evaluated with \(q(\cdot,\,\text{amax})\); squared error per bin is then multiplied by \(H(i)\) and summed.

### 4.3 Optimal amax Search — `find_optimal_amax` (lines 219–254)

**Inputs:** `weighted_hist`, `num_candidates`, `search_range` \((r_{\mathrm{lo}}, r_{\mathrm{hi}})\), `refinement_iterations`, `scaled`.

**Initial range:**

$$\mathit{low}_0 = \text{max\_val} \cdot r_{\mathrm{lo}},\qquad \mathit{high}_0 = \text{max\_val} \cdot r_{\mathrm{hi}}.$$

Example: `search_range=(0.5, 1.0)` → search in \([0.5\,\text{max\_val},\, 1.0\,\text{max\_val}]\).

**Code (guard and initial bounds):**

```226:237:histogram/weighted_histogram_mse.py
        if weighted_hist.histogram is None or weighted_hist.max_val <= 0:
            return weighted_hist.max_val
        
        histogram = weighted_hist.get_histogram()
        bin_centers = weighted_hist.get_bin_centers()
        max_val = weighted_hist.max_val
        low = max_val * search_range[0]
        high = max_val * search_range[1]
        if not scaled:
            pass  # compatible mode: 448 cap applied in quantizer
        best_amax = max_val
        min_mse = float('inf')
```

**Search loop:** For iteration \(t = 0,\,1,\,\ldots,\,\texttt{refinement\_iterations}\):

1. **Candidate set:** \(\Delta_1,\,\ldots,\,\Delta_N\) = `torch.linspace(low, high, num_candidates)`.
2. **Minimize:** \(\Delta^*_t = \arg\min_{\Delta \in \{\Delta_j\}} \mathrm{MSE}(\Delta)\); keep `best_amax` and `min_mse`.
3. **Refinement (if \(t < \texttt{refinement\_iterations}\)):** Narrow range around \(\Delta^*_t\):

$$\mathit{range\_width} = \frac{\mathit{high} - \mathit{low}}{4},$$

$$\mathit{low}_{t+1} = \max\bigl(0.1\,\text{max\_val},\; \Delta^*_t - \mathit{range\_width}\bigr),\qquad \mathit{high}_{t+1} = \min\bigl(1.2\,\text{max\_val},\; \Delta^*_t + \mathit{range\_width}\bigr).$$

**Code (loop and refinement):**

```239:254:histogram/weighted_histogram_mse.py
        for iteration in range(refinement_iterations + 1):
            # Debug: log search bounds (initial + each refinement update)
            print(f"  [MSE SEARCH DEBUG] max_val: {max_val:.6f} | range: {search_range[0]:.3f}-{search_range[1]:.3f} | BOUNDS: {low:.6f} to {high:.6f} (iter {iteration})")
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

**Return value:** The amax that achieved the minimum weighted MSE over all iterations.

---

## 5. HSWQWeightedHistogramOptimizer

### 5.1 Role

High-level API: given a weight tensor and optional per-channel importance, build the weighted histogram and run the MSE optimizer to get the best amax (and optionally stats).

```257:273:histogram/weighted_histogram_mse.py
class HSWQWeightedHistogramOptimizer:
    """
    HSWQ weighted histogram optimizer: WeightedHistogram + FP8E4M3Quantizer + MSEOptimizer.
    Example: optimizer.compute_optimal_amax(weight_tensor, importance)
    """

    def __init__(self,
                 bins: int = 4096,
                 num_candidates: int = 200,
                 refinement_iterations: int = 3,
                 device: str = "cuda"):
        """Args: bins, num_candidates, refinement_iterations, device."""
        self.bins = bins
        self.num_candidates = num_candidates
        self.refinement_iterations = refinement_iterations
        self.device = device
        self.mse_optimizer = MSEOptimizer(device)
```

- A `WeightedHistogram` is **not** created at init; one is created per call in `compute_optimal_amax` / `compute_optimal_amax_with_stats`.

### 5.2 Constructor

Parameters: `bins`, `num_candidates`, `refinement_iterations`, `device`. A single `MSEOptimizer(device)` is stored as `self.mse_optimizer`.

### 5.3 `compute_optimal_amax` (lines 275–290)

**Formula:** \(\Delta^* = \arg\min_\Delta \sum_i H(i)(q(x_i,\Delta)-x_i)^2\) where \(H\) and \(x_i\) are built from `weight` and `importance`.

**Code:**

```275:290:histogram/weighted_histogram_mse.py
    def compute_optimal_amax(self,
                             weight: torch.Tensor,
                             importance: Optional[torch.Tensor] = None,
                             scaled: bool = True) -> float:
        """Compute optimal amax: build weighted hist from I_c, then minimize MSE. scaled=False for compatible."""
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, importance)
        
        # Find optimal amax
        optimal_amax = self.mse_optimizer.find_optimal_amax(
            weighted_hist,
            num_candidates=self.num_candidates,
            refinement_iterations=self.refinement_iterations,
            scaled=scaled
        )
        return optimal_amax
```

This is the one-liner entry point used by quantizer scripts.

### 5.4 `compute_optimal_amax_with_stats` (lines 292–318)

Same optimization as above; additionally computes **estimated MSE** at the chosen amax and returns a dict.

**Returned values:**

| Key | Formula / meaning |
|-----|---------------------|
| `optimal_amax` | \(\Delta^*\) |
| `max_val` | \(\max_{k,c}|W_{k,c}|\) |
| `compression_ratio` | \(\Delta^* / \text{max\_val}\) (or 1.0 if max_val=0) |
| `estimated_mse` | \(\sum_i H(i)(q(x_i,\Delta^*)-x_i)^2\) |

**Code:**

```292:318:histogram/weighted_histogram_mse.py
    def compute_optimal_amax_with_stats(self,
                                        weight: torch.Tensor,
                                        importance: Optional[torch.Tensor] = None,
                                        scaled: bool = True
                                        ) -> dict:
        """Return optimal_amax, max_val, compression_ratio, estimated_mse."""
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, importance)
        
        optimal_amax = self.mse_optimizer.find_optimal_amax(
            weighted_hist,
            num_candidates=self.num_candidates,
            refinement_iterations=self.refinement_iterations,
            scaled=scaled
        )
        histogram = weighted_hist.get_histogram()
        bin_centers = weighted_hist.get_bin_centers()
        estimated_mse = self.mse_optimizer.compute_weighted_mse(
            histogram, bin_centers, optimal_amax, scaled=scaled
        )
        
        return {
            'optimal_amax': optimal_amax,
            'max_val': weighted_hist.max_val,
            'compression_ratio': optimal_amax / weighted_hist.max_val if weighted_hist.max_val > 0 else 1.0,
            'estimated_mse': estimated_mse
        }
```

---

## 6. Self-Test (`if __name__ == "__main__"`)

The script runs four checks (lines 322–367):

1. **FP8 grid:** Build quantizer, print positive grid size, max representable, sample values.
2. **Quantize–dequantize:** Run on a small tensor with amax=448, print original, dequantized, and errors.
3. **Weighted histogram:** Build from a random Conv2d-shaped weight and random importance; print max_val, total_weight, and histogram sum (should be 1.0).
4. **MSE optimization:** Run `compute_optimal_amax_with_stats` on the same weight/importance and print optimal amax, max_val, compression ratio, estimated MSE.

**Code (tests 1–2):**

```329:344:histogram/weighted_histogram_mse.py
    # Test 1: FP8 grid construction
    print("\n[Test 1] FP8 E4M3 Grid Construction")
    quantizer = FP8E4M3Quantizer(device)
    print(f"  Positive grid size: {len(quantizer._positive_grid)}")
    ...
    # Test 2: Quantize-dequantize
    print("\n[Test 2] Quantize-Dequantize")
    test_values = torch.tensor([0.1, 0.5, 1.0, 2.0, 100.0, 400.0], device=device)
    amax = 448.0
    dequant = quantizer.quantize_dequantize(test_values, amax)
```

**Code (tests 3–4):**

```346:364:histogram/weighted_histogram_mse.py
    # Test 3: Weighted histogram
    ...
    hist = WeightedHistogram(bins=1024, device=device)
    hist.build(weight, importance)
    ...
    # Test 4: MSE optimization
    optimizer = HSWQWeightedHistogramOptimizer(device=device)
    result = optimizer.compute_optimal_amax_with_stats(weight, importance)
```

This verifies the pipeline end-to-end without a full quantization run.

---

## 7. Summary

### 7.1 Formula Index

| Formula | Section |
|--------|--------|
| \(\Delta^* = \arg\min_\Delta \sum_i H(i)(q(x_i,\Delta)-x_i)^2\) | §1.1 |
| \(q(x,\Delta)\) scaled: \(q = \mathrm{round}_{\mathrm{FP8}}(x\cdot s)/s\), \(s=448/\Delta\) | §2.3 |
| \(q(x,\Delta)\) non-scaled: \(q = \mathrm{round}_{\mathrm{FP8}}(\mathrm{clip}(x,-\Delta,\Delta))\) | §2.3 |
| \(\mathrm{round}_{\mathrm{FP8}}(a) = \mathrm{argmin}_{g\in\text{grid}}|a-g|\) (then sign) | §2.4 |
| \(H(b) = \sum \alpha_{k,c}\) over bin \(b\); \(\alpha_{k,c}=I_c\); then normalize \(\sum_i H(i)=1\) | §3.1, §3.3 |
| Bin index \(b = \lfloor |W|/\mathit{bin\_width}\rfloor\), clamp \([0,B-1]\) | §3.3 |
| Bin center \(x_i = (i+0.5)\cdot \mathit{bin\_width}\) | §3.4 |
| \(\mathrm{MSE}(\Delta) = \sum_i H(i)(q(x_i,\Delta)-x_i)^2\) | §4.2 |
| Refinement: \(\mathit{range\_width}=(high-low)/4\); new bounds around best_amax | §4.3 |

### 7.2 Component Table

| Component | Responsibility |
|-----------|----------------|
| **FP8E4M3Quantizer** | Physical FP8 grid; `quantize_dequantize` with scaled / non-scaled modes; nearest-grid rounding. |
| **WeightedHistogram** | Build \(H(i)\) from weight + importance (2D/4D); normalize; provide bin centers and histogram. |
| **MSEOptimizer** | Compute \(\sum_i H(i)(q(x_i,\Delta)-x_i)^2\); search amax (linear candidates + refinement). |
| **HSWQWeightedHistogramOptimizer** | Compose: build histogram → find_optimal_amax (and optional stats). |

Together, they implement the HSWQ objective: **find the clipping threshold that minimizes importance-weighted quantization error**, using an exact FP8 simulation and multi-stage search.
