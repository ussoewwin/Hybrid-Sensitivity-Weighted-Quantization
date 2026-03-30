# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity FP8 quantization for **SDXL**, **Flux1.dev**, and **Z Image Turbo** diffusion models. HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast. It offers two modes: standard-compatible (V1) and high-performance scaled (V2). **V2 requires a dedicated loader and is not usable at the current time.**

**Technical details:** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md)

**SDXL models:** [Hugging Face — Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/logo.png" width="400">
</p>

---

## How to quantize

- **SDXL:** [How to quantize SDXL](md/How%20to%20quantize%20SDXL.md)
- **Z Image Turbo:** [How to quantize Z Image Turbo](md/How%20to%20quantize%20ZIT.md)

**Benchmark results:** [SDXL (MSE / SSIM)](test/benchmark_test.md)

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

File size is reduced by about **30–40%** vs FP16 while keeping best quality per use case.

---

## Architecture

1. **Dual Monitor System** — During calibration, two metrics are collected:
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted → top 10–25% kept in FP16 (for SDXL, 10% often gives sufficient quality; for ZIT, r1.0 is sufficient).
   - **Importance** (input mean absolute value): per-channel contribution → used as weights in the weighted histogram.
   **Technical details:** [Dual Monitor System — Technical Guide](md/Dual_Monitor_System_Technical_Guide.md).

2. **Rigorous FP8 Grid Simulation** — Uses a physical grid (all 0–255 values cast to `torch.float8_e4m3fn`) instead of theoretical formulas, so MSE matches real runtime.

3. **Weighted MSE Optimization** — Finds parameters that minimize quantization error using the importance histogram. **Technical details:** [Weighted Histogram MSE — Technical Guide](md/Weighted_Histogram_MSE_Technical_Guide.md).

---

## Modes

- **V1** (`scaled=False`): No scaling; only the clipping threshold (amax) is optimized. Output is standard FP8 weights. **Use this mode** — full compatibility with any FP8 loader.
- **V2** (`scaled=True`): Weights are scaled to FP8 range, quantized, and inverse scale `S` is stored in Safetensors (`.scale`). Requires a dedicated loader; **not usable at the current time.**

---

## Recommended Parameters

- **Samples:** 32 (recommended) — number of calibration samples.
- **Steps:** 25 — number of inference steps per sample during calibration.
- **Keep ratio:** 10–25% — keeps critical layers in FP16. For SDXL, 10% often gives sufficient quality.
- **Latent:** 32–256, default 128 — calibration latent size (H/W). Use `--latent 32` for faster calibration, `--latent 256` for higher fidelity.

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| Naive FP8 | 0.75–0.93 | 50% | High |
| **HSWQ V1** | **0.86–0.98** | 60-70% (FP16 mixed) | **High** |
| **HSWQ V2** | — (currently unmeasurable) | 60-70% (FP16 mixed) | Not usable (no dedicated loader) |

HSWQ V1 gives a clear gain over Naive FP8 with full compatibility. V2 would offer higher quality but requires a dedicated loader; benchmark is currently unmeasurable and V2 is not usable at the current time.

---

## Changelog

Version history and release notes are in [CHANGELOG.md](CHANGELOG.md).

---

## Base Repositories

This project is built upon the following repositories:

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** — The most powerful and modular diffusion model GUI, API and backend with a graph/nodes interface by [@Comfy-Org](https://github.com/Comfy-Org).
