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

### Analysis & Key Findings (HSWQ V1.3)

The comprehensive benchmark results demonstrate that **HSWQ V1.3** provides a profound advantage over standard naive FP8 conversion, proving essential for structurally vulnerable SDXL models.

1. **Rescuing "Unstable" Models from Native FP8 Collapse**  
   Standard naive FP8 casting frequently destroys models with irregular weight distributions or extreme outliers. Several models in this test—such as **JANKUTrainedNoobaiRouwei_v69** (SSIM drops to **0.8872**), **unholyDesireMixSinister_v60** (**0.8694**), and **waiIllustriousSDXL_v160** (**0.8864**)—suffered catastrophic structural collapse under Native FP8. **HSWQ V1.3**'s Dual-Monitor engine dynamically identifies high-variance layers and protects them in FP16, successfully pulling these models back to highly usable states (SSIM **0.93–0.96+**) while drastically reducing Mean Squared Error (MSE).

2. **The Raw Power of Weighted Histogram Optimization (Even at r=0)**  
   The results for **uwazumimixILL_v50** (tested at **r0**, meaning 0% FP16 protection) highlight the core strength of the V1.3 algorithm. Even when forcing the entire UNet into FP8 without any protective fallback, HSWQ achieved an SSIM of **0.9641** (outperforming the naive baseline's **0.9542**) while significantly lowering MSE. This proves that HSWQ's exact-grid MSE optimization, weighted by input activations, is fundamentally superior to naive casting on its own.

3. **Near-Lossless Fidelity for Modern Architectures**  
   Highly evolved architectures like Pony and Illustrious derivatives (e.g., **cottonnoob_v50**, **obsessionIllustrious_vPredV20**, **epicrealismXL_pureFix**) show exceptional tolerance to HSWQ, frequently scoring between **0.97** and **0.98+** SSIM. Their optimized parameter distributions allow HSWQ to compress them aggressively with virtually zero human-perceivable degradation.

4. **PTQ Limitations vs. Official FP8 Releases**  
   In one specific instance (**asianRealismByStable_v30FP16**), the officially distributed FP8 model outperforms HSWQ. This transparently illustrates the natural limitations of Post-Training Quantization (PTQ) when compared to Quantization-Aware Training (QAT) or a publisher's manually curated FP8 release. However, for the vast majority of community merges and finetunes that lack an official FP8 release, HSWQ clearly stands as the definitive quantization solution.

**Conclusion:**  
HSWQ V1.3 offers a highly efficient, structurally safe quantization strategy for SDXL. By relying on activation variance for targeted FP16 protection and importance-weighted histogram clipping for FP8 optimization, it consistently prevents structural collapse, maximizes VRAM efficiency, and outperforms naive conversion without the overhead of heavy computational operations.

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
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted → top 10–25% kept in FP16 (for SDXL, 10% is often sufficient).
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
