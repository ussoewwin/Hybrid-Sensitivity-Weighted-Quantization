# SDXL HSWQ V1.3 and Weighted Histogram MSE Fast — Full Explanation

This document explains in full detail what changed from the **original histogram module (`weighted_histogram_mse`) + SDXL V1.2** to **V1.3 + Fast histogram (`weighted_histogram_mse_fast`)**, including code.

---

## 1. Overview of Changes

| Item | Before (V1.2 + original histogram) | After (V1.3 + Histogram Fast) |
|------|------------------------------------|-------------------------------|
| **SDXL quantization script** | `quantize_sdxl_hswq_v1.2.py` | `quantize_sdxl_hswq_v1.3.py` |
| **Histogram / MSE module** | `weighted_histogram_mse.py` | `weighted_histogram_mse_fast.py` |
| **FP8 grid rounding** | Full distance matrix (brute force) | Binary search (`searchsorted`) |
| **Histogram / bin centers** | float64 | float64 (precision preserved) |
| **Device transfer** | Always `.to(device)` | Skip if already on same device |
| **MSE search** | No logs | Debug logs + Fast init logs |
| **Formula / API** | Same | Same (drop-in compatible) |

**VRAM optimization** (drop pipeline → load state_dict on GPU → clamp/cast) was already in place in V1.2.  
In V1.3 the main change is **switching the histogram module used for amax computation to the Fast version**, plus minor differences such as message language.

---

## 2. Original Module (`weighted_histogram_mse.py`) — Specification

### 2.1 Role and Core Formula

Implements HSWQ-compliant **weighted histogram MSE optimization**.

- **Core formula**
  - \( \Delta^* = \arg\min_\Delta \sum_i H(i) \cdot (q(x_i, \Delta) - x_i)^2 \)
  - \( H(i) \): weighted histogram (weighted by input importance \( I_c \))
  - \( q(x, \Delta) \): quantize–dequantize (clip threshold amax = Δ)
  - \( \Delta \): clipping value (amax)

### 2.2 Class Layout

- **FP8E4M3Quantizer**  
  Accurate FP8 E4M3 quantization simulation.  
  Builds the positive representable grid in `_build_fp8_grid()` and rounds to the nearest grid point in `_round_to_fp8_grid()`.
- **WeightedHistogram**  
  Builds the weighted histogram: \( \alpha_{k,c} = I_c \), and for bin \( b \), \( H(b) = \sum \alpha_{k,c} \) (sum of importance, not counts). Normalized.
- **MSEOptimizer**  
  Computes the weighted MSE above and searches for optimal amax via grid search over candidate amax values plus refinement.
- **HSWQWeightedHistogramOptimizer**  
  Single API wrapping the above. `compute_optimal_amax(weight, importance, scaled=False)` returns the optimal amax in compatibility mode.

### 2.3 Original FP8 Grid Rounding (Bottleneck)

`_round_to_fp8_grid` computes the distance from each value to every positive grid point and picks the minimum (brute force).

```python
# from weighted_histogram_mse.py
def _round_to_fp8_grid(self, values: torch.Tensor) -> torch.Tensor:
    """Round values to nearest FP8 grid point."""
    signs = torch.sign(values)
    abs_values = values.abs()
    abs_flat = abs_values.reshape(-1)
    batch_size = 10000
    result = torch.zeros_like(abs_flat)

    for i in range(0, len(abs_flat), batch_size):
        batch = abs_flat[i:i+batch_size]
        # [N] vs [grid_size] → [N, grid_size] distance matrix — heavy in memory and compute
        distances = (batch.unsqueeze(1) - self._positive_grid.unsqueeze(0)).abs()
        nearest_indices = distances.argmin(dim=1)
        result[i:i+batch_size] = self._positive_grid[nearest_indices]

    result = result.reshape(abs_values.shape)
    return result * signs
```

- Complexity: \( O(N \cdot G) \) for \( N \) values and \( G \) grid points.  
- Memory: \( O(N \cdot G) \) temporary tensor.  
For large layers or many bins, this dominates amax computation time.

### 2.4 Original Histogram Build

- `weight` and `importance` are always transferred with `weight.detach().float().to(self.device)` and `importance.float().to(self.device)`.
- Histogram is stored and normalized in **float64** (`torch.float64`, `.double()` in `scatter_add_`).
- Bin centers from `get_bin_centers()` are also **float64**.

### 2.5 Original MSE Search

- `find_optimal_amax`: builds `num_candidates` amax candidates in `search_range` via `torch.linspace`, and calls `compute_weighted_mse` for each.
- Refinement: narrows the range around the best amax and repeats.
- No log output.

---

## 3. Fast Module (`weighted_histogram_mse_fast.py`) — What Changed

### 3.1 Same Formula and API

Core formula and public API are unchanged.

- \( \Delta^* = \arg\min_\Delta \sum_i H(i) \cdot (q(x_i, \Delta) - x_i)^2 \)
- `HSWQWeightedHistogramOptimizerFast` exposes `compute_optimal_amax` and `compute_optimal_amax_with_stats` with the same signatures — **drop-in replacement**.

### 3.2 Faster FP8 Grid Rounding (Binary Search)

The positive grid is sorted, so `torch.searchsorted` gives the insertion index and only the left/right neighbors are compared.

```python
# from weighted_histogram_mse_fast.py
def _round_to_fp8_grid_optimized(self, values: torch.Tensor) -> torch.Tensor:
    """
    Optimized FP8 grid rounding using binary search.
    Performance: 10-50x faster than brute force method.
    Memory: O(n) instead of O(n * grid_size)
    """
    signs = torch.sign(values)
    abs_values = values.abs()
    original_shape = abs_values.shape
    abs_flat = abs_values.reshape(-1)

    # Binary search on sorted _positive_grid
    indices = torch.searchsorted(self._positive_grid, abs_flat)
    indices = indices.clamp(1, len(self._positive_grid) - 1)

    left_indices = indices - 1
    right_indices = indices
    left_vals = self._positive_grid[left_indices]
    right_vals = self._positive_grid[right_indices]
    left_dist = (abs_flat - left_vals).abs()
    right_dist = (abs_flat - right_vals).abs()
    result = torch.where(left_dist < right_dist, left_vals, right_vals)

    result = result.reshape(original_shape)
    return result * signs
```

- Complexity: \( O(N \log G) \).  
- Memory: \( O(N) \) only.  
Roughly 10–50× faster on large layers.

### 3.3 Original Rounding Kept for Verification

The Fast module keeps `_round_to_fp8_grid_original`, which reproduces the same batch distance-matrix → argmin behavior. The self-test checks the maximum difference between optimized and original rounding results.

### 3.4 Device Optimization in Histogram Build

Skips `.to(device)` when the tensor is already on the target device.

```python
# from weighted_histogram_mse_fast.py (WeightedHistogramOptimized.build)
if weight.device.type != self.device:
    weight = weight.detach().float().to(self.device)
else:
    weight = weight.detach().float()
# importance: same — no transfer if already on device
```

The histogram itself remains **float64** (comment states float64 to avoid information loss).  
Bin centers are also **float64**, preserving the same precision as the original.

### 3.5 MSE Search Debug Logs

- **Search range log**  
  Inside `find_optimal_amax`, prints `max_val` and search bounds to stdout.  
  Example: `[MSE SEARCH DEBUG] max_val: ... | range: ... | BOUNDS: ... to ...`
- **Fast init log**  
  In `HSWQWeightedHistogramOptimizerFast.__init__`, prints device, bins, candidates, and refinement iterations.  
  Example: `[HSWQ] HSWQWeightedHistogramOptimizer (Fast Path) initialized on cuda`  
  Example: `Bins: 4096 | Candidates: 200 | Refinement: 3 iterations`

### 3.6 Benchmark Mode

`benchmark_performance()` runs `compute_optimal_amax` with the original and Fast modules on the same weight and importance, and compares runtime and the resulting amax.

---

## 4. SDXL Script Diff: V1.2 → V1.3

### 4.1 Import Difference (Actual Functional Change)

**V1.2 (`archives/quantize_sdxl_hswq_v1.2.py`)**

```python
# HSWQ module
from weighted_histogram_mse import HSWQWeightedHistogramOptimizer
```

**V1.3 (`quantize_sdxl_hswq_v1.3.py`)**

```python
# HSWQ module (Fast)
from weighted_histogram_mse_fast import HSWQWeightedHistogramOptimizerFast as HSWQWeightedHistogramOptimizer
```

In V1.3 the Fast class is imported under the name `HSWQWeightedHistogramOptimizer`, so **the rest of the script is unchanged** and only the amax computation uses the Fast implementation.

### 4.2 Docstring / Message Differences

- V1.2: English (e.g. "Quantize SDXL model to FP8 (HSWQ V1.2: GPU-accelerated conversion)").
- V1.3: Some Japanese messages (e.g. "VRAMを最大限活用し...", "キャリブレーション準備中").  
Algorithm and conversion flow are the same.

### 4.3 Same Conversion Flow

Both scripts follow the same steps:

1. Load ComfyUI-style state_dict and detect UNet layout.
2. Build Diffusers pipeline and register Dual Monitor hooks for calibration.
3. Run calibration (collect Sensitivity and Importance). Optional SageAttention2 (`--sa2`) support.
4. Sort by sensitivity; keep top `keep_ratio` in FP16, rest as FP8 targets.
5. **HSWQ optimizer** computes optimal amax from each FP8 layer’s `weight` and importance.  
   → V1.2 uses the original module here; V1.3 uses the Fast module.
6. Delete pipeline and optimizer to free VRAM.
7. Move `original_state_dict` to GPU in one go.
8. For each key: keep FP16 as-is; for FP8 targets, `clamp(±amax)` then cast to `torch.float8_e4m3fn`.
9. Save with `save_file` (fallback: move to CPU then save).

So **only “which histogram module is used for amax” changes**; the conversion loop and VRAM strategy are the same as V1.2.

---

## 5. Class / Method Mapping

| Original (`weighted_histogram_mse`) | Fast (`weighted_histogram_mse_fast`) | Notes |
|-------------------------------------|--------------------------------------|------|
| `FP8E4M3Quantizer` | `FP8E4M3QuantizerOptimized` | Same grid construction |
| `_round_to_fp8_grid` | `_round_to_fp8_grid_optimized` | 10–50× faster via binary search |
| (none) | `_round_to_fp8_grid_original` | Original implementation kept for verification |
| `WeightedHistogram` | `WeightedHistogramOptimized` | Device check in build; histogram still float64 |
| `MSEOptimizer` | `MSEOptimizerOptimized` | Debug logs added; search logic same |
| `HSWQWeightedHistogramOptimizer` | `HSWQWeightedHistogramOptimizerFast` | Init log added; API same |

---

## 6. Summary

- **Original (`weighted_histogram_mse`) + SDXL V1.2**  
  - Optimal amax from weighted histogram MSE.  
  - FP8 rounding uses full grid distance (brute force), costly on large layers.  
  - Conversion already runs on GPU (VRAM-optimized).
- **Fast (`weighted_histogram_mse_fast`) + SDXL V1.3**  
  - Same formula and API; FP8 rounding switched to binary search, shortening amax computation by roughly 10–50×.  
  - Histogram and bin centers stay float64 for precision.  
  - Skips unnecessary device transfers and adds debug/init logs.  
- **SDXL-side change**  
  - V1.3 only swaps the import to the Fast module; conversion algorithm and VRAM strategy match V1.2.

This completes the full explanation of the change from the original histogram + V1.2 to V1.3 + Histogram Fast.
