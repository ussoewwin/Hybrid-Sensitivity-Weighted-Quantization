# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity FP8 quantization for **SDXL**, **Flux1.dev**, and **Z Image** diffusion models. HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast. It offers two modes: standard-compatible (V1) and high-performance scaled (V2). **V2 requires a dedicated loader and is not usable at the current time.**

**Technical details:** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md)

**SDXL models:** [Hugging Face — Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3)

---

## How to quantize

- **SDXL:** [How to quantize SDXL](md/How%20to%20quantize%20SDXL.md)
- **Z Image:** [How to quantize Z Image](md/How%20to%20quantize%20ZI.md)

---

## Overview

| Feature | V1: Standard Compatible | V2: High Performance Scaled |
| :--- | :--- | :--- |
| **Compatibility** | Full (100%), any FP8 loader | Requires dedicated loader — **not usable at present** |
| **File format** | Standard FP8 (`torch.float8_e4m3fn`) | Extended FP8 (weights + `.scale` metadata) |
| **Image quality (SSIM)** | ~0.98 (max) | Unmeasurable (no dedicated loader) |
| **Mechanism** | Optimal clipping (smart clipping) | Full-range scaling (dynamic scaling) |
| **Benchmark** | Measurable | Currently unmeasurable (no dedicated loader) |
| **Use case** | Distribution, general users | Unavailable until a dedicated loader exists |

File size is reduced by about **40–45%** vs FP16 while keeping best quality per use case.

---

## Architecture

1. **Dual Monitor System** — During calibration, two metrics are collected:
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted → top 10–25% kept in FP16 (for SDXL, 10% is often sufficient).
   - **Importance** (input mean absolute value): per-channel contribution → used as weights in the weighted histogram.

2. **Rigorous FP8 Grid Simulation** — Uses a physical grid (all 0–255 values cast to `torch.float8_e4m3fn`) instead of theoretical formulas, so MSE matches real runtime.

3. **Weighted MSE Optimization** — Finds parameters that minimize quantization error using the importance histogram.

---

## Modes

- **V1** (`scaled=False`): No scaling; only the clipping threshold (amax) is optimized. Output is standard FP8 weights. **Use this mode** — full compatibility with any FP8 loader.
- **V2** (`scaled=True`): Weights are scaled to FP8 range, quantized, and inverse scale `S` is stored in Safetensors (`.scale`). Requires a dedicated loader; **not usable at the current time.**

---

## Recommended Parameters

- **Samples:** 256 (recommended). In practice, a sample size of 32 is sufficient to maintain adequate precision.
- **Keep ratio:** 10–25% — keeps critical layers in FP16. For SDXL, 10% often gives sufficient quality; for Z Image, 25% is recommended.
- **Steps:** 20–25 — to include early denoising sensitivity.

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% (6.5GB) | High |
| Naive FP8 | 0.81–0.93 | 50% | High |
| **HSWQ V1** | **0.86–0.98** | 55–60% (FP16 mixed) | **High** |
| **HSWQ V2** | — (currently unmeasurable) | 55–60% (FP16 mixed) | Not usable (no dedicated loader) |

HSWQ V1 gives a clear gain over Naive FP8 with full compatibility. V2 would offer higher quality but requires a dedicated loader; benchmark is currently unmeasurable and V2 is not usable at the current time.

---

## Changelog

### 1.0.4
- **Quantization guides** — Published step-by-step procedures for [SDXL](md/How%20to%20quantize%20SDXL.md) and [ZIT (Z Image Turbo)](md/How%20to%20quantize%20ZIT.md).  
  **Release notes:** [v1.0.4](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.4)

### 1.0.3
- **SDXL SageAttention2** — V1.2 (standard) and V1.6 (high precision) add optional SageAttention2-accelerated calibration via `--sa2`. Same FP8 output; SA2 used only during calibration.  
  **Release notes:** [v1.0.3](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.3) *(to be published)*

### 1.0.2
- **SDXL HSWQ V1.5** — High-precision quantization script: bins=8192, candidates=1000, refinement_iterations=10 (ZIT V1.5 methodology). Same standard-compatible FP8 output as V1.1; higher quality, ~27× longer run. V1.1 script moved to `archives/`.  
  **Release notes:** [v1.0.2](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.2)

### 1.0.1
- **DualMonitor 2D input support** — Fixed handling of 2D input tensors `(B, C)` in `DualMonitor.update()`. Previously, 2D inputs (e.g. embedding layers, `adaLN_modulation` in Z-Image Turbo) fell back to uniform importance `1.0`; now per-channel importance `(C,)` is computed via `mean(dim=0)`. This improves weighted histogram MSE for time_embedding, add_embedding (SDXL) and adaLN / t_embedder / cap_embedder (ZIT).  
  **Release notes:** [v1.0.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.1)

---

## Base Repositories

This project is built upon the following repositories:

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** — The most powerful and modular diffusion model GUI, API and backend with a graph/nodes interface by [@Comfy-Org](https://github.com/Comfy-Org).
