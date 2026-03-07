# Weighted Histogram MSE — Technical Guide

**Source:** `histogram/weighted_histogram_mse.py`  
This document is a complete, line-by-line technical explanation of the HSWQ weighted histogram MSE optimization module used to find the optimal clipping threshold (amax) that minimizes quantization error under importance weighting.

---

## 1. Overview and Core Formula

The module implements the HSWQ optimization objective:

$$\Delta^* = \arg\min_\Delta \sum_i H(i) \cdot \bigl(q(x_i, \Delta) - x_i\bigr)^2$$

Where:

| Symbol | Meaning |
|--------|--------|
| \(\Delta\) (amax) | Clipping threshold; the parameter we optimize. |
| \(H(i)\) | **Weighted histogram**: weight of the \(i\)-th bin, derived from input importance \(I_c\) (per-channel importance). |
| \(q(x, \Delta)\) | Quantize–dequantize function: clip by \(\Delta\), round to FP8 grid, return dequantized value. |
| \(x_i\) | Representative value for bin \(i\) (e.g. bin center). |

So we search for the **amax** that minimizes the **importance-weighted** mean squared error between original and quantized weights.

The module provides three main components:

1. **FP8E4M3Quantizer** — Accurate FP8 E4M3 quantize/dequantize simulation (physical grid, no theoretical formula).
2. **WeightedHistogram** — Build \(H(i)\) from weight tensor and optional per-channel importance.
3. **MSEOptimizer** — Evaluate weighted MSE for a given amax and search for optimal amax (with optional refinement).

The top-level **HSWQWeightedHistogramOptimizer** composes these: build histogram from weight + importance, then run MSEOptimizer to get optimal amax.

---

## 2. FP8E4M3Quantizer

### 2.1 Role and FP8 E4M3 Spec

The quantizer simulates real FP8 E4M3 behavior so that MSE in this module matches runtime behavior.

**FP8 E4M3 format (from docstring):**

- Sign: 1 bit  
- Exponent: 4 bits (bias = 7)  
- Mantissa: 3 bits  
- Representable range: \(\pm[2^{-6}, 448]\) (including denormals)  
- Special: NaN (0x7F, 0xFF), \(\pm 0\)

### 2.2 Grid Construction — `_build_fp8_grid` (lines 45–59)

```python
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

- Every byte 0–255 is interpreted as `float8_e4m3fn` and converted to float32.
- NaNs are dropped; only valid numeric values are kept.
- **Positive** representable values are collected, uniquified, and sorted → `_positive_grid`.
- Negatives are the symmetric negatives of the positive grid (excluding 0) → `_full_grid`.
- `max_representable` is the maximum positive value (448.0).

This **physical grid** ensures rounding and representable set match PyTorch’s actual FP8 behavior.

### 2.3 Quantize–Dequantize — `quantize_dequantize` (lines 60–91)

**Signature:** `quantize_dequantize(values, amax, scaled=True)`

- **`scaled=True` (V2-style, full range):**
  1. Scale: \(x_{\text{scaled}} = x \cdot (448/\text{amax})\).
  2. Clamp to \([-448, 448]\).
  3. Round to nearest FP8 value (`_round_to_fp8_grid`).
  4. Dequantize: divide by the same scale.
- **`scaled=False` (V1, standard-compatible):**
  1. Clip: \(x \mapsto \text{clip}(x, -\text{amax}, \text{amax})\).
  2. Further clamp to \([-448, 448]\) (FP8 range).
  3. Round to nearest FP8 value; **no** scaling, so output is in the same scale as input.

So `scaled=False` only changes the clip threshold; no per-tensor scale factor is applied (compatible with standard loaders).

### 2.4 Rounding to FP8 Grid — `_round_to_fp8_grid` (lines 93–108)

```python
signs = torch.sign(values)
abs_values = values.abs()
# ... batch loop over abs_flat ...
distances = (batch.unsqueeze(1) - self._positive_grid.unsqueeze(0)).abs()
nearest_indices = distances.argmin(dim=1)
result[i:i+batch_size] = self._positive_grid[nearest_indices]
# ...
return result * signs
```

- Absolute values are compared to `_positive_grid`; each value is mapped to the **nearest** representable positive value (argmin of distances).
- Batching (e.g. 10000 elements at a time) avoids excessive memory for large tensors.
- Signs are reapplied so the result is symmetric for negative inputs.

### 2.5 Single-Value Error — `compute_quantization_error` (lines 110–115)

Helper that computes \(|q(x, \text{amax}) - x|\) for a single float. Used for debugging or small tests; the main optimization uses the full histogram and bin centers.

---

## 3. WeightedHistogram

### 3.1 Role and Formula

From the spec: \(\alpha_{k,c} = I_c\); \(H(b) = \sum \alpha_{k,c}\) over elements in bin \(b\).

So the histogram is **not** a count of elements per bin; it is the **sum of importance** per bin. After building, the histogram is **normalized** so that \(\sum_i H(i) = 1\).

### 3.2 Constructor and State (lines 124–130)

- `bins`: number of histogram bins (e.g. 4096).
- `device`: `"cuda"` or `"cpu"`.
- `histogram`, `max_val`, `total_weight` are set in `build()`.

### 3.3 Building the Histogram — `build` (lines 132–177)

**Inputs:** `weight` (tensor), optional `importance` (per-channel, shape \([I]\)).

1. **Preprocess weight:** `weight.detach().float().to(device)`, then use `w_abs = weight.abs()`.  
   `max_val = w_abs.max().item()` (with a small guard if 0 to avoid division by zero).

2. **Importance handling:**
   - If `importance` is `None`, use `imp_expanded = torch.ones_like(weight)` (uniform weight).
   - If **4D** (Conv2d `(Out, In, K, K)`):  
     - `importance` is trimmed or padded to length `in_channels`.  
     - Expanded to `(1, in_channels, 1, 1)` and broadcast to `weight.shape` → `imp_expanded`.
   - If **2D** (Linear `(Out, In)`):  
     - Same idea: trim/pad to `in_features`, then `(1, -1)` and expand to `weight.shape`.
   - Other shapes: fallback to `torch.ones_like(weight)`.

3. **Binning:**
   - `bin_width = max_val / bins`.
   - `bin_indices = (w_abs / bin_width).long().clamp(0, bins - 1)` (0-based indices).
   - Histogram array: `torch.zeros(bins, dtype=torch.float64, device=device)`.
   - **Scatter-add:** `histogram.scatter_add_(0, bin_indices.reshape(-1), imp_expanded.reshape(-1).double())`  
     So each element contributes its **importance** to its bin.

4. **Normalize:**  
   `total_weight = histogram.sum()`; if \(> 0\), `histogram = histogram / total_weight` so \(\sum_i H(i) = 1\).

### 3.4 Bin Centers and Histogram Getter

- **`get_bin_centers()`** (lines 179–188): Returns the **center** of each bin in \([0, \text{max\_val}]\):  
  \(x_i = (i + 0.5) \cdot \text{bin\_width}\) for \(i = 0, \ldots, \text{bins}-1\), as float64 on the same device.
- **`get_histogram()`** (lines 190–192): Returns the normalized histogram tensor (used by MSEOptimizer).

---

## 4. MSEOptimizer

### 4.1 Role

Finds \(\Delta^* = \arg\min_\Delta \sum_i H(i)(q(x_i,\Delta)-x_i)^2\) using the histogram and bin centers. Holds an `FP8E4M3Quantizer` instance for \(q\).

### 4.2 Weighted MSE — `compute_weighted_mse` (lines 206–218)

**Inputs:** `histogram`, `bin_centers`, `amax`, `scaled`.

1. Run quantize–dequantize on `bin_centers` with the given `amax` and `scaled` flag:  
   `dequantized = fp8_quantizer.quantize_dequantize(bin_centers.float(), amax, scaled=scaled).double()`.
2. Squared error per bin: `error_sq = (dequantized - bin_centers)**2`.
3. Weighted sum: `weighted_mse = (histogram * error_sq).sum().item()`.

So we get exactly \(\sum_i H(i)(q(x_i,\Delta)-x_i)^2\) for the candidate \(\Delta=\text{amax}\).

### 4.3 Optimal amax Search — `find_optimal_amax` (lines 220–255)

**Inputs:** `weighted_hist`, `num_candidates`, `search_range`, `refinement_iterations`, `scaled`.

1. **Guard:** If histogram not built or `max_val \le 0`, return `max_val`.
2. **Range:**  
   `low = max_val * search_range[0]`, `high = max_val * search_range[1]` (e.g. (0.5, 1.0) → search in \([0.5\,\text{max\_val}, 1.0\,\text{max\_val}]\)).
3. **Coarse search:**  
   For each iteration (initial + `refinement_iterations`):
   - Build `num_candidates` candidates with `torch.linspace(low, high, num_candidates)`.
   - For each candidate amax, compute `compute_weighted_mse(...)` and keep the amax with smallest MSE.
4. **Refinement:**  
   For iterations after the first: narrow the range around the current best amax (e.g. shrink to a quarter width, clamp to `[0.1*max_val, 1.2*max_val]`), then repeat the linear sweep.

So we get a multi-stage search: coarse over `[low, high]`, then finer around the best amax. The optional debug print logs `max_val`, `search_range`, and the current bounds each iteration.

**Return value:** The amax that achieved the minimum weighted MSE.

---

## 5. HSWQWeightedHistogramOptimizer

### 5.1 Role

High-level API: given a weight tensor and optional per-channel importance, build the weighted histogram and run the MSE optimizer to get the best amax (and optionally stats).

### 5.2 Constructor (lines 264–274)

- `bins`, `num_candidates`, `refinement_iterations`, `device` are stored.
- Creates a single `MSEOptimizer(device)` as `self.mse_optimizer`.  
  (It does not create a WeightedHistogram at init; one is created per call.)

### 5.3 `compute_optimal_amax` (lines 276–291)

1. Build: `WeightedHistogram(bins, device).build(weight, importance)`.
2. Call `mse_optimizer.find_optimal_amax(weighted_hist, num_candidates=..., refinement_iterations=..., scaled=scaled)`.
3. Return the optimal amax.

So this is the one-liner entry point used by quantizer scripts: “build histogram from this weight/importance, then minimize weighted MSE.”

### 5.4 `compute_optimal_amax_with_stats` (lines 293–319)

Same as above, but after finding optimal amax it also:

- Recomputes weighted MSE at that amax → `estimated_mse`.
- Returns a dict: `optimal_amax`, `max_val`, `compression_ratio` (= optimal_amax / max_val), `estimated_mse`.

Useful for logging and diagnostics.

---

## 6. Self-Test (`if __name__ == "__main__"`)

The script runs four checks:

1. **FP8 grid:** Build quantizer, print positive grid size, max representable, sample values.
2. **Quantize–dequantize:** Run on a small tensor with amax=448, print original, dequantized, and errors.
3. **Weighted histogram:** Build from a random Conv2d-shaped weight and random importance; print max_val, total_weight, and histogram sum (should be 1.0).
4. **MSE optimization:** Run `compute_optimal_amax_with_stats` on the same weight/importance and print optimal amax, max_val, compression ratio, estimated MSE.

This verifies the pipeline end-to-end without depending on a full quantization run.

---

## 7. Summary Table

| Component | Responsibility |
|-----------|----------------|
| **FP8E4M3Quantizer** | Physical FP8 grid; `quantize_dequantize` with scaled / non-scaled modes; nearest-grid rounding. |
| **WeightedHistogram** | Build \(H(i)\) from weight + importance (2D/4D); normalize; provide bin centers and histogram. |
| **MSEOptimizer** | Compute \(\sum_i H(i)(q(x_i,\Delta)-x_i)^2\); search amax (linear candidates + refinement). |
| **HSWQWeightedHistogramOptimizer** | Compose: build histogram → find_optimal_amax (and optional stats). |

Together, they implement the HSWQ objective: **find the clipping threshold that minimizes importance-weighted quantization error**, using an exact FP8 simulation and optional multi-stage search.
