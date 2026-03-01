# Why SageAttention2 (SA2) Was Removed from SDXL Calibration

**Summary:** SA2 was removed from SDXL quantization because it **slightly lowers calibration scores (SSIM)** and provides **no meaningful speed gain**. The same option remains available for Z Image Turbo (ZIT), where it **does not degrade scores** (speed gain is not significant there either). This document explains the reasoning from both engineering and mathematical viewpoints.

---

## 1. The Decision

- **SDXL:** SA2 has been **removed** from the calibration pipeline. Calibration uses **native PyTorch SDPA only**.
- **ZIT:** The **`--sa2`** option is **kept**. It does not degrade calibration scores (unlike on SDXL); speed improvement is not significant.

**Reasons for removing SA2 from SDXL:**

1. SA2 **slightly lowers** calibration scores (SSIM).
2. **No meaningful speed improvement** was observed for SDXL calibration.
3. Keeping it would add complexity and noise with no benefit.

This is a “subtraction for purity”: removing a non-essential component that only degrades the precision of calibration.

---

## 2. Why Does SA2 Lower Scores? (Noise in Calibration)

Optimized attention implementations (SA2, FlashAttention, etc.) trade **speed and VRAM** for:

- **Changed computation order** (e.g. block-wise processing).
- **Custom rounding** inside Triton kernels.

For normal **inference**, the resulting numerical differences are negligible to the human eye. So for “generating an image and looking at it,” SA2 is fine.

HSWQ calibration is different. It **measures**:

- **Sensitivity** — output variance per layer (which layers matter most if perturbed).
- **Importance** — per-channel input statistics (mean absolute values).

These are accumulated over **many steps and many prompts**. Any systematic difference between “SA2 output” and “native PyTorch SDPA output” is **not** negligible here: it acts as **statistical noise** that:

- Shifts layer rankings (which layers get FP16).
- Slightly distorts the weighted histogram used for amax optimization.

So in a pipeline that is fighting for **every bit of rounding precision** (e.g. the kind of sensitivity that showed up with a single character change in the Fast histogram), inserting a **black box** that does **not** produce bit-identical results to native PyTorch is harmful. For SDXL calibration, SA2 is pure noise with no upside.

---

## 3. Why SA2 Hurts SDXL but Not ZIT

Empirically:

- **SDXL:** SA2 **lowers** calibration scores.
- **ZIT:** SA2 **does not** lower scores (speed improvement is not significant).

This fits a clear picture from signal/noise and architecture.

### 3.1 Signal-to-Noise Ratio (S/N)

- **SDXL** weight and activation statistics are relatively “well-behaved” — like a “quiet lake.” The small numerical differences introduced by SA2’s Triton kernels (e.g. ULP-level) are on the same order as the signals we care about. So in DualMonitor and histogram aggregation, **SA2’s differences show up as noise** and blur the calibration.
- **ZIT** has much larger variance and more extreme activations — like a “stormy sea.” The scale of the “real” signal (layer outputs, outliers) is so large that **SA2’s tiny numerical differences are effectively drowned out**. They don’t show up as a measurable score drop.

So: same SA2, but different S/N in the two models → different impact on calibration.

### 3.2 Block-Wise Processing and Outliers (ZIT)

FlashAttention-style kernels use **block-wise max scaling** and custom accumulators to save VRAM. For a model like ZIT, where outputs can approach FP16 limits (e.g. 65504), this block-wise processing can **accidentally** act like a mild soft-clipping or stabilization. So in ZIT, SA2 might slightly **tame** extreme spikes that native SDPA would pass through. That can make ZIT’s calibration **more** stable rather than noisier. For SDXL, there is no such beneficial effect; only the extra noise matters.

### 3.3 Architecture Fit: DiT vs UNet

SageAttention (and SA2) were designed and tuned for **DiT-style** models (large self-attention, diffusion transformers). ZIT (e.g. NextDiT) is exactly that target. SDXL is a **UNet** with a different pattern (e.g. cross-attention–heavy). So:

- **ZIT (DiT):** SA2 is a good fit — numerically stable; it does not degrade calibration (no meaningful speed gain).
- **SDXL (UNet):** SA2 is a poor fit — lowers scores and no real gain; the numerical differences show up as calibration noise.

So “SA2 hurts SDXL but not ZIT” reflects both **noise level** and **architecture alignment**.

---

## 4. Engineering Conclusion

| Model | SA2 in calibration | Reason |
|-------|--------------------|--------|
| **SDXL** | **Removed** | Slightly lowers SSIM; no speed gain; only adds noise. Purity and reproducibility are better with native SDPA only. |
| **ZIT** | **Optional `--sa2`** | Does not lower scores; no significant speed gain. Kept as an option for those who prefer it. |

So:

- **SDXL:** “Pure native computation” for calibration — no SA2.
- **ZIT:** Keep `--sa2` for users who prefer it; it does not degrade scores (speed gain is not significant).

**In one sentence:** *Depending on the model (UNet vs DiT, “quiet” vs “stormy” statistics), the same optimization (SA2) can either degrade calibration (SDXL) or not (ZIT) — we keep it only where it does not hurt (ZIT).*

This is fully consistent with the empirical finding: **SA2 slightly lowers scores on SDXL with no meaningful speed benefit**; on ZIT it does not degrade scores (and speed gain is not significant). Removing it from SDXL and keeping it as an optional, non-degrading choice for ZIT is the right engineering choice.
