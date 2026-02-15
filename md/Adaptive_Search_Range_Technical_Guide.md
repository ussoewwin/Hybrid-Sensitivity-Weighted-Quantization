# HSWQ Adaptive Search Range — Technical Guide (Flux.1)

## 1. Overview

This feature is a quantization-quality protection logic **tuned for Flux.1**, optimized for its weight distribution (extreme spikes and outliers).

It was developed to address the “SSIM 0.79” issue in Flux.1: severe quality drop when certain layers are quantized. It analyzes weight statistics (kurtosis, outlier ratio), automatically detects risky layers, and avoids aggressive clipping.

> **Note**: This logic is designed specifically for the Flux.1 architecture. On other models (e.g. SDXL, Z-Image Turbo), it can **worsen** SSIM or generation quality. Use it **only for Flux.1**.

## 2. Background and Problem (Flux.1 Specific)

### 2.1 Flux.1’s unusual distribution

Flux.1 (especially attention outputs and FFN layers) often has **heavy-tailed** or **spike-like** weight distributions, unlike SDXL and similar models.

### 2.2 Why SSIM drops to 0.79

With the default HSWQ `search_range=(0.55, 1.0)`, the optimizer minimizes MSE by clipping those spikes (outliers) and representing the bulk of values more finely. In Flux.1, those “clipped outliers” are **critical** for quality; removing them can cause a large SSIM drop (e.g. 0.95 → 0.79).

## 3. Algorithm

### 3.1 Statistics

For each layer’s weight tensor $W$ we compute:

1. **Kurtosis**: Sharpness of the distribution. Flux layers with spikes can exceed ~50.
2. **Outlier ratio**: $Ratio = \frac{\max(|W|)}{|P_{99}|}$ — how many times the max is larger than the 99th percentile.

### 3.2 Decision logic (Flux-tuned)

From these, we choose the layer’s `search_range` (lower bound of the search interval).

| Condition (priority) | Label | Search range | Effect in Flux |
| :--- | :--- | :--- | :--- |
| `Ratio > 3.0` | **Extreme Outlier** | `(1.0, 1.0)` | **Full protection**: no clipping; quantize at max. This avoids the 0.79 issue. |
| `Kurtosis > 50.0` | Very Sharp | `(0.95, 1.0)` | Almost no clipping; only slight adjustment. |
| `Kurtosis > 20.0` | Sharp | `(0.8, 1.0)` | Cautious search. |
| Otherwise | Normal | `(0.55, 1.0)` | Standard HSWQ optimization. |

## 4. Code (Flux V1.5)

### 4.1 Shared logic

Implemented in `quantize_flux_hswq_v1.5.py` (or equivalent Flux script).

```python
def get_adaptive_search_range(weight_tensor: torch.Tensor) -> tuple[float, float]:
    w_abs = weight_tensor.abs().float()
    max_val = w_abs.max().item()
    if max_val == 0: return (0.5, 1.0)

    # Sample for speed when > 1M elements
    if w_abs.numel() > 1_000_000:
        indices = torch.randint(0, w_abs.numel(), (1_000_000,), device=w_abs.device)
        sample = w_abs.flatten()[indices]
        p99 = torch.quantile(sample, 0.99).item()
        mean = sample.mean().item()
        std = sample.std().item()
        kurtosis = ((sample - mean)**4).mean().item() / (std**4 + 1e-6)
    else:
        # ... (full tensor)

    outlier_ratio = max_val / (p99 + 1e-6)
    
    # Flux-tuned logic
    if outlier_ratio > 3.0: return (1.0, 1.0)
    elif kurtosis > 50.0: return (0.95, 1.0)
    elif kurtosis > 20.0 or outlier_ratio > 2.2: return (0.8, 1.0)
    elif outlier_ratio > 1.8: return (0.65, 1.0)
    else: return (0.55, 1.0)
```

### 4.2 Usage: inject range via component

Without changing `weighted_histogram_mse.py`, the adaptive range is passed through the optimizer component (`hswq_optimizer.mse_optimizer`).

```python
# Inside quantize_flux_hswq_*.py
adaptive_range = get_adaptive_search_range(module.weight.data)

optimal_amax = hswq_optimizer.mse_optimizer.find_optimal_amax(
    weighted_hist,
    # ...
    search_range=adaptive_range,  # dynamic range per layer
    scaled=False
)
```

## 5. Usage notes

* **Not recommended for SDXL / ZIT**:
  * SDXL and ZIT tend not to have such extreme spikes.
  * Using this logic there can over-protect layers that would benefit from clipping, **increasing** quantization error and worsening scores.
  * For SDXL/ZIT, use standard HSWQ (fixed range, `scaled=False`) or `scaled=True` instead.

## 6. Summary

Adaptive Search Range is a **Flux.1–specific** tuning that respects the importance of outliers. The thresholds are tuned for Flux; do not apply them unchanged to other architectures.
