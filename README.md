# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity FP8 quantization for diffusion models (SDXL / SD1.5 / Flux.1). HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast, and offers two modes: standard-compatible (V1) and high-performance scaled (V2).

**Technical details:** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md)

---

## Overview

| Feature | V1: Standard Compatible | V2: High Performance Scaled |
| :--- | :--- | :--- |
| **Compatibility** | Full (100%), any FP8 loader | Custom loader (HSWQLoader) required |
| **File format** | Standard FP8 (`torch.float8_e4m3fn`) | Extended FP8 (weights + `.scale` metadata) |
| **Image quality (SSIM)** | ~0.95 (theoretical limit) | ~0.96+ (close to FP16) |
| **Mechanism** | Optimal clipping (smart clipping) | Full-range scaling (dynamic scaling) |
| **Use case** | Distribution, general users | In-house, max quality, server-side |

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

- **V1** (`scaled=False`): No scaling; only the clipping threshold (amax) is optimized. Output is standard FP8 weights. Use when you need maximum compatibility.
- **V2** (`scaled=True`): Weights are scaled to FP8 range, quantized, and inverse scale `S` is stored in Safetensors (`.scale`). Use with HSWQLoader for best quality.

---

## Files in This Repo

| File | Description |
|------|-------------|
| `quantize_sdxl_hswq_v1.py` | V1 conversion: standard-compatible FP8 (no scaling). |
| `quantize_sdxl_hswq_v2_scaled.py` | V2 conversion: high-performance FP8 with `.scale` metadata. |
| `weighted_histogram_mse.py` | Core optimization: weighted histogram MSE (PyTorch native grid). |
| `verify_fp8_grid.py` | Verifies FP8 grid accuracy. |
| `md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md` | Full technical spec (algorithm, process flow, benchmarks). |

*(ComfyUI loader for V2: `hswq_loader_node.py` — not in this repo; see technical doc.)*

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
| Naive FP8 | ~0.81 | 50% | High |
| **HSWQ V1** | **0.86–0.88** | 55% (FP16 mixed) | **High** |
| **HSWQ V2** | **0.90–0.94** | 55% (FP16 mixed) | Low (custom loader) |

HSWQ V1 gives a clear gain over Naive FP8 with full compatibility; V2 targets maximum quality with a custom loader.
