
# HSWQ: Hybrid Sensitivity Weighted Quantization — Technical Overview

**Version:** 2.0  
**Date:** 2026-01-29  
**Models:** SDXL / SD1.5 / Flux.1 (Proposed)

---

## 1. Executive Summary

**HSWQ (Hybrid Sensitivity Weighted Quantization)** is a high-fidelity FP8 quantization algorithm optimized for diffusion models (especially SDXL).

Whereas conventional FP8 methods apply uniform compression (Naive Cast), HSWQ analyzes model **sensitivity** and **importance** to provide two modes:

| Feature | V1: Standard Compatible | V2: High Performance Scaled |
| :--- | :--- | :--- |
| **Compatibility** | **Full (100%)** | Custom loader required (HSWQLoader) |
| **File format** | Standard FP8 (`torch.float8_e4m3fn`) | Extended FP8 (Weights + `.scale` metadata) |
| **Image quality (SSIM)** | **~0.87** (theoretical limit) | **~0.92+** (close to FP16) |
| **Mechanism** | Optimal clipping (Smart Clipping) | Full-range scaling (Dynamic Scaling) |
| **Primary use** | Distribution, general users | In-house use, maximum quality, server-side |

This approach reduces file size by about 50% (vs. FP16) while achieving the best quality for each use case.

---

## 2. Architecture

The HSWQ core system consists of three components.

### 2.1. Dual Monitor System
During calibration inference, statistics are collected from two perspectives.

1.  **Sensitivity Monitor (output variance)**:
    *   **Purpose**: Identify layers whose corruption severely degrades image quality.
    *   **Metric**: Output tensor variance $\text{Var}(Y)$.
    *   **Action**: **Keep top 25% (recommended) in FP16** for protection.

2.  **Importance Monitor (input importance)**:
    *   **Purpose**: Identify which input channels contribute most to the computation.
    *   **Metric**: Input tensor mean absolute value $\text{Mean}(|X|_c)$.
    *   **Action**: Used as **weights** in the weighted histogram.

### 2.2. Rigorous FP8 Grid Simulation
Instead of theoretical formulas (`2 ** E * ...`), a **physical grid** is used: all byte values (0–255) are cast to PyTorch’s `torch.float8_e4m3fn` type.
This simulates implementation-dependent rounding and special values exactly, ensuring MSE calculations match the real runtime environment.

### 2.3. Weighted MSE Optimization
Parameters that minimize quantization error are searched using the collected importance histogram.

---

## 3. Algorithm Details

### 3.1. V1: Standard Compatible Strategy
Standard loaders do not support FP8 scaling factors (metadata). Therefore, V1 must not scale (multiply) weight values.

*   **Strategy**: MSE optimization in `scaled=False` mode.
*   **Behavior**:
    *   No scaling; only the **clipping threshold (amax)** is optimized.
    *   Determine how much to clip outliers so that overall MSE (importance-weighted) is minimized.
    *   This yields much higher quality than simple min-max clipping.

### 3.2. V2: High Performance Scaled Strategy
With a custom loader, FP8 dynamic range is fully utilized.

*   **Strategy**: MSE optimization in `scaled=True` mode.
*   **Behavior**:
    *   $$W_{fp8} = \text{Round}(W_{fp16} \times \frac{448.0}{\text{amax}})$$
    *   $$S = \frac{\text{amax}}{448.0}$$
    *   Weights are scaled up to FP8 maximum (448.0), quantized, and the inverse scale $S$ is stored in Safetensors under the `.scale` key.
*   **At inference**:
    *   `HSWQLoader` loads `.scale` at load time.
    *   `HSWQLinear` / `HSWQConv2d` layers compute $$W_{fp16} \approx W_{fp8} \times S$$ on-the-fly in VRAM (or use FP8 Gemm).

---

## 4. Process Flow

```mermaid
graph TD
    A[Calibration Input] --> B{Dual Monitor}
    B --> C[Sensitivity Map]
    B --> D[Importance Map]
    
    C --> E{Layer Selection}
    E -->|Top 25%| F[Keep FP16]
    E -->|Others| G[Weighted MSE Opt]
    
    G --> H{Optimization Mode}
    D --> G
    
    H -->|V1: Compatible| I[Find Best Clip (amax)]
    I --> J[Clip & Cast (No Scaling)]
    J --> K[Standard FP8 Weights]
    K --> L[Save .safetensors]
    F --> L
    
    H -->|V2: Scaled| M[Find Best Scale (amax)]
    M --> N[Scale & Cast]
    N --> O[FP8 Weights + .scale Meta]
    O --> P[Save .safetensors]
    F --> P
```

---

## 5. Implementation Specs and Recommended Settings

### 5.1. Recommended Parameters
*   **Samples**: `256` (HSWQ default)
    *   Minimum for statistical reliability; 128 is insufficient.
*   **Keep Ratio**: `0.25` (25%)
    *   Safety margin to protect critical layers; 0.10 carries higher degradation risk.
*   **Steps**: `20–25`
    *   To include sensitivity from the early denoising stages.

### 5.2. Package Layout (`hswq/`)
*   `quantize_sdxl_hswq_v1.py`: V1 compatible-mode conversion script
*   `quantize_sdxl_hswq_v2_scaled.py`: V2 high-performance conversion script
*   `weighted_histogram_mse.py`: Core optimization engine (PyTorch native grid)
*   `hswq_loader_node.py`: ComfyUI custom node for V2
*   `verify_fp8_grid.py`: FP8 grid accuracy verification tool

---

## 6. Benchmark Results (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| **Original FP16** | 1.0000 | 100% (6.5GB) | High |
| **Naive FP8** | ~0.81 | 50% | High |
| **HSWQ V1** | **0.86–0.88** | 55% (FP16 mixed) | **High** |
| **HSWQ V2** | **0.90–0.94** | 55% (FP16 mixed) | Low (custom loader) |

HSWQ V1 provides a clear quality gain over Naive FP8 while keeping full compatibility, establishing it as a practical standard distribution format.
