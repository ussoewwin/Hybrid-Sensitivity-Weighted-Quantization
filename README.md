# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity FP8 quantization for **SDXL** diffusion models. HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast. It offers two modes: standard-compatible (V1) and high-performance scaled (V2). **V2 requires a dedicated loader and is not usable at the current time.**

**Technical details:** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md)

**SDXL models:** [Hugging Face — Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3)

---

## Overview

| Feature | V1: Standard Compatible | V2: High Performance Scaled |
| :--- | :--- | :--- |
| **Compatibility** | Full (100%), any FP8 loader | Requires dedicated loader — **not usable at present** |
| **File format** | Standard FP8 (`torch.float8_e4m3fn`) | Extended FP8 (weights + `.scale` metadata) |
| **Image quality (SSIM)** | ~0.95 (theoretical limit) | ~0.96+ (close to FP16) |
| **Mechanism** | Optimal clipping (smart clipping) | Full-range scaling (dynamic scaling) |
| **Use case** | Distribution, general users | Unavailable until a dedicated loader exists |

File size is reduced by about **50%** vs FP16 while keeping best quality per use case.

---

## Architecture

1. **Dual Monitor System** — During calibration, two metrics are collected:
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted → top 25% kept in FP16.
   - **Importance** (input mean absolute value): per-channel contribution → used as weights in the weighted histogram.

2. **Rigorous FP8 Grid Simulation** — Uses a physical grid (all 0–255 values cast to `torch.float8_e4m3fn`) instead of theoretical formulas, so MSE matches real runtime.

3. **Weighted MSE Optimization** — Finds parameters that minimize quantization error using the importance histogram.

---

## Modes

- **V1** (`scaled=False`): No scaling; only the clipping threshold (amax) is optimized. Output is standard FP8 weights. **Use this mode** — full compatibility with any FP8 loader.
- **V2** (`scaled=True`): Weights are scaled to FP8 range, quantized, and inverse scale `S` is stored in Safetensors (`.scale`). Requires a dedicated loader; **not usable at the current time.**

---

## Files in This Repo

| File | Description |
|------|-------------|
| `quantize_sdxl_hswq_v1.1.py` | V1 SDXL conversion: standard-compatible FP8 (no scaling). |
| `quantize_sdxl_hswq_v2.1_scaled.py` | V2 SDXL conversion: FP8 with `.scale` metadata. Not usable at present (no dedicated loader). |
| `quantize_zit_hswq_v1.py` | Z-Image Turbo (ZIT) conversion: HSWQ FP8 for ZIT models. |
| `weighted_histogram_mse.py` | Core optimization: weighted histogram MSE (PyTorch native grid). |
| `verify_fp8_grid.py` | Verifies FP8 grid accuracy. |
| `fp8bench.py` | FP8 benchmarking utilities. |
| `archives/` | Older scripts: `quantize_sdxl_hswq_v1.py`, `quantize_sdxl_hswq_v2_scaled.py`. |
| `sample/` | Calibration prompt sets (`calibration_prompts_128.txt`, `_256`, `_512`). |
| `md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md` | Full technical spec (algorithm, process flow, benchmarks). |
| `md/HSWQ_DualMonitor_Fix_Report.md` | DualMonitor 2D input fix report (v1.0.1). |

---

## Recommended Parameters

- **Samples:** 256 (minimum for reliable stats; 128 is insufficient).
- **Keep ratio:** 0.25 (25%) — keeps critical layers in FP16; 0.10 has higher degradation risk.
- **Steps:** 20–25 — to include early denoising sensitivity.

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% (6.5GB) | High |
| Naive FP8 | 0.81–0.93 | 50% | High |
| **HSWQ V1** | **0.86–0.95** | 55% (FP16 mixed) | **High** |
| **HSWQ V2** | **0.87–0.96** | 55% (FP16 mixed) | Not usable (no dedicated loader) |

HSWQ V1 gives a clear gain over Naive FP8 with full compatibility. V2 would offer higher quality but requires a dedicated loader and is not usable at the current time.

---

## Changelog

### 1.0.1
- **DualMonitor 2D input support** — Fixed handling of 2D input tensors `(B, C)` in `DualMonitor.update()`. Previously, 2D inputs (e.g. embedding layers, `adaLN_modulation` in Z-Image Turbo) fell back to uniform importance `1.0`; now per-channel importance `(C,)` is computed via `mean(dim=0)`. This improves weighted histogram MSE for time_embedding, add_embedding (SDXL) and adaLN / t_embedder / cap_embedder (ZIT).  
  **Release notes:** [v1.0.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.1)
