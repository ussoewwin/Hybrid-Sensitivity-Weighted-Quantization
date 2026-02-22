# Flux V1.6 Adaptive Search Range — Full Explanation

This document explains the **outlier-handling logic** implemented in `quantize_flux_hswq_v1.6.py`, and why it **dramatically improved quality on Flux.1 (0.79→0.95)** while **hurting or breaking ZIB/SDXL** — with code and structural mechanisms.

---

## 1. Core Logic for Outlier Handling (Code Walkthrough)

The main feature of Flux V1.6 is a mechanism that **computes each layer’s weight distribution (kurtosis, outlier ratio) in real time and automatically adjusts how aggressively HSWQ (histogram MSE) clips** when quantizing.

### Relevant Code: `get_adaptive_search_range`

The core function is:

```python
# --- Adaptive search range (New in v1.5/1.6) ---
def get_adaptive_search_range(weight_tensor: torch.Tensor) -> tuple[float, float]:
    """Choose optimal search range from weight distribution."""
    w_abs = weight_tensor.abs().float()
    max_val = w_abs.max().item()
    
    if max_val == 0:
        return (0.5, 1.0)

    # For very large tensors (>1M elements), use random sampling for speed
    if w_abs.numel() > 1_000_000:
        indices = torch.randint(0, w_abs.numel(), (1_000_000,), device=w_abs.device)
        sample = w_abs.flatten()[indices]
        p99 = torch.quantile(sample, 0.99).item()  # 99th percentile (upper 1% bound)
        mean = sample.mean().item()
        std = sample.std().item()
        # Kurtosis: how "heavy-tailed" (outlier-rich) the distribution is
        kurtosis = ((sample - mean)**4).mean().item() / (std**4 + 1e-6)
    else:
        # (full-element path without sampling... omitted)

    # Outlier ratio: how far the max is from the 99th percentile
    outlier_ratio = max_val / (p99 + 1e-6)
    
    # --- Decision logic ---
    # Very high kurtosis = heavy tails = outliers matter a lot = do not clip aggressively
    if outlier_ratio > 3.0:
        return (1.0, 1.0)  # Severe outliers: no search, force quant at abs_max
    elif kurtosis > 50.0:
        return (0.95, 1.0)  # Very conservative: lower bound 0.95
    elif kurtosis > 20.0 or outlier_ratio > 2.2:
        return (0.8, 1.0)   # Conservative: lower bound 0.8
    elif outlier_ratio > 1.8:
        return (0.65, 1.0)  # Some outliers: lower bound 0.65
    else:
        return (0.55, 1.0)  # Normal (near-Gaussian): allow deeper MSE search down to 0.55
```

### How It Is Used

The `search_range` (e.g. `(0.8, 1.0)`) from this function is passed to the dedicated `AdaptiveHSWQOptimizer`.  
That way, **layers with many outliers are forced to use an Amax in the upper band (e.g. 0.8–1.0) even when histogram MSE would prefer a more aggressive clip.** This is the “outlier protection” in practice.

---

## 2. Why It Worked So Well on Flux.1 (0.79 → 0.95)

### Architecture: “Uniform DiT Blocks Everywhere”

Flux.1 does not have the kind of **extreme bottlenecks** (e.g. UNet-style mid/neck) that tightly restrict information flow. It is built from repeated blocks (Double/Single) with similar structure.

### What Actually Breaks Flux in FP8: “Cutting Off Large Spikes”

On Flux.1, the main cause of broken images is that **occasional very large outliers (spikes)** in some layers get **clipped away** by normal histogram-MSE search, and their information is lost.  
FP8 (e4m3) has only 3 mantissa bits, but **with 4 exponent bits, if Amax (scale) is kept large enough, even large values can be represented.**

* **V1.6’s effect:** High kurtosis is detected and the Amax search range for that layer is fixed at `1.0` (no clipping). As a result, large spikes are preserved and quality jumps back up to about **0.95**.

---

## 3. Why It Backfires on SDXL / ZIB (Quality Collapse)

Applying the same V1.6 logic to ZIB or SDXL can cause SSIM to drop to around **0.7**. Why?

### Architecture: “Extreme Input/Output Bottlenecks (Lens Layers)”

SDXL and ZI (NextDiT with UNet-like I/O) have **special “lens” layers** such as `x_embedder`, `t_embedder`, and `final_layer`. These layers map high-dimensional vectors into/out of latent space and routinely carry **very extreme values**.

### What Actually Breaks ZI/SDXL in FP8: “3-bit Mantissa → Frosted-Glass Effect”

The reason ZIB/SDXL lens layers die when quantized to FP8 is **not** “outliers were clipped.”  
It is that **FP8’s mantissa (only 3 bits for fine gradients) is too coarse for these lenses — they effectively become “frosted glass,”** and the smooth mapping is destroyed.

* Once the lens is frosted, every layer downstream (no matter how high-precision) sees a blurred signal.
* **Why V1.6 fails here:** V1.6 says “this layer has many outliers, so let’s use a wide Amax and quantize gently.” But **as soon as the layer is in FP8, the 3-bit mantissa makes it frosted glass anyway** — so the model is still destroyed.

---

## 4. Conclusion and the Role of V1.7 / V1.8

Flux V1.6’s logic is: **“Assume we quantize all layers to FP8; protect outliers from aggressive clipping.”** For Flux, where large spikes were the main cause of failure, this was the right fix.

For ZI/SDXL, the failure mode is different: **“The lens is destroyed by the limited precision of FP8 (mantissa), not by clipping.”**

That is why the **V1.7 / V1.8 hardcoded protection (critical-layer keep)** for ZIB is the correct approach:

```python
# V1.7/1.8 hard protection for ZIB/SDXL
if any(k in name for k in ['embedder', 'final_layer', 'time_in', 'vector_in', 'guidance_in', 'txt_in', 'img_in']):
    keep_layers.add(name)
    print(f"\n  [V1.6 Protection] {name} explicitly excluded to prevent SSIM drop.")
    continue  # Skip FP8 conversion; keep in FP16 (clear lens).
```

### Different Architectures → Different Strategies

* **Flux.1:** Quantize most layers to FP8 and only soften clipping for spikes **(Adaptive Search Range: V1.6)**.
* **ZIB / SDXL:** Keep the “heart” (I/O lens layers) in FP16 at all costs; compress the rest with full SVD **(Hardcoded protection + Full-SVD: V1.8)**.

This architecture-dependent difference in where quantization hurts most is the full picture of why the Flux V1.6 feature helps Flux but worsens ZI.
