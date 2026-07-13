# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity **FP8** and **INT8** quantization for **SDXL**, **Flux1.dev**, **Z Image Turbo**, and **Z-Anime** diffusion models. HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast.

- **FP8:** two modes — standard-compatible (V1) and high-performance scaled (V2). **V2 requires a dedicated loader and is not usable at the current time.**
- **INT8 (SDXL V3.0):** ComfyUI-compatible `int8_tensorwise` pack with DualMonitor + V4 weighted-histogram ranking for FP16 protection under a fixed budget. Keep ratio is **0** (r0); critical layers stay FP16 via automatic analysis, not a keep-ratio percentage.

**Technical details:** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md)

**SDXL models (FP8):** [Hugging Face — Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3)

**SDXL models (INT8):** [Hugging Face — Hybrid-Sensitivity-Weighted-Quantization-SDXL-INT8](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-INT8)

**Z Image / Z-Anime models (FP8):** [Hugging Face — HSWQ-Z-Image-fp8e4m3](https://huggingface.co/ussoewwin/HSWQ-Z-Image-fp8e4m3)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/logo.png" width="400">
</p>

---

## How to quantize

- **SDXL (FP8):** [How to quantize SDXL](md/How%20to%20quantize%20SDXL.md)
- **SDXL (INT8):** `quantize_sdxl_hswq_v3.0.py` (keep ratio **0**; Card 3 `--per_channel_int8` optional)
- **Z Image / Z-Anime (FP8):** [How to quantize Z Image](md/How%20to%20quantize%20Z%20Image.md)

**Benchmark results:**
- **SDXL (FP8):** [MSE / SSIM](test/benchmark_test.md)
- **SDXL (INT8):** [MSE / SSIM](test/benchmark_sdxl_int8.md)
- **Z Image / Z-Anime (FP8):** [MSE / SSIM](test/benchmark_zit.md)

---

## Overview

| Feature | FP8 V1: Standard Compatible | FP8 V2: High Performance Scaled | INT8 (SDXL V3.0) |
| :--- | :--- | :--- | :--- |
| **Compatibility** | Full (100%), any FP8 loader | Requires dedicated loader — **not usable at present** | ComfyUI `int8_tensorwise` / QUANT_ALGOS compatible |
| **File format** | Standard FP8 (`torch.float8_e4m3fn`) | Extended FP8 (weights + `.scale` metadata) | INT8 weights + scale (`int8_tensorwise`) |
| **Image quality (SSIM)** | **0.94–0.98** | Unmeasurable (no dedicated loader) | **0.94–0.98** |
| **Mechanism** | Optimal clipping (smart clipping) | Full-range scaling (dynamic scaling) | Absmax pack + DualMonitor / V4 FP16 protect (r0) |
| **Keep ratio** | 5–25% (see How-to; SDXL/ZIT often 10%) | 5–25% (see How-to) | **0 (fixed)** |
| **Benchmark** | Measurable | Currently unmeasurable (no dedicated loader) | Measurable |
| **Use case** | Distribution, general users | Unavailable until a dedicated loader exists | SDXL INT8 distribution / kitchen loaders |

File size is reduced by about **30–40%** vs FP16 while keeping best quality per use case.

---

## Architecture

1. **Dual Monitor System** — During calibration, two metrics are collected:
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted → kept in FP16 when selected by HSWQ (**FP8:** top 5–25% by keep ratio; for SDXL and ZIT, 10% often gives sufficient quality. **INT8:** keep ratio **0**; FP16 set comes from automatic analysis / budget ranking, not a keep-ratio %).
   - **Importance:** V1 uses per-channel input mean-abs; V4 uses per-element SVD leverage × RMS magnitude hybrid → weights of the weighted histogram.
   **Technical details:** [Dual Monitor System — Technical Guide](md/Dual_Monitor_System_Technical_Guide.md).

2. **Rigorous grid / pack simulation**
   - **FP8:** physical grid (all 0–255 values cast to `torch.float8_e4m3fn`) instead of theoretical formulas, so MSE matches real runtime.
   - **INT8:** natural absmax pack point for the symmetric INT8 grid; V4 weighted-histogram MSE ranks FP16 protection candidates (does not choose pack amax).

3. **Weighted MSE Optimization** — Finds parameters that minimize quantization error using an importance-weighted histogram (not a plain frequency histogram).
   - **V1 / Fast:** per-channel importance (activation mean-abs) drives the histogram amax search. **Technical details:** [Weighted Histogram MSE — Technical Guide](md/Weighted_Histogram_MSE_Technical_Guide.md).
   - **V4 (SVD × RMS hybrid):** per-element importance blends **SVD structural leverage** \(L(i,j)=(U_i\cdot\sigma)^2\cdot(V_j)^2\) with **RMS magnitude**; \(\alpha\) tilts toward SVD on heavy-tailed layers. Used by Z Image / Z-Anime FP8 for amax search, and by SDXL INT8 for FP16-candidate ranking at the absmax pack point. **Technical details:** [HSWQ V4 SVD-RMS — Technical Guide](md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md).

---

## Modes

### FP8

- **V1** (`scaled=False`): No scaling; only the clipping threshold (amax) is optimized. Output is standard FP8 weights. **Use this mode** — full compatibility with any FP8 loader.
- **V2** (`scaled=True`): Weights are scaled to FP8 range, quantized, and inverse scale `S` is stored in Safetensors (`.scale`). Requires a dedicated loader; **not usable at the current time.**

### INT8 (SDXL)

- **Tensorwise pack (default):** per-tensor absmax; format tag `int8_tensorwise`.
- **Card 3** (`--per_channel_int8`): per-output-channel amax / scale. Mutually exclusive with `--asymmetric_int8`.
- **Keep ratio:** **0 (fixed)** — FP16 protection is automatic (analyze Hard VETO + DualMonitor + V4 ranking inside the FP16 budget), not a percentage keep-ratio.

---

## Recommended Parameters

- **Samples:** 32 (recommended) — number of calibration samples (**same for FP8 and INT8**).
- **Steps:** 25 — number of inference steps per sample during calibration (**same for FP8 and INT8**).
- **Keep ratio (FP8):** 5–25% — keeps critical layers in FP16. For SDXL and ZIT, 10% often gives sufficient quality.
- **Keep ratio (INT8):** **0 (fixed)** — do not use a non-zero keep-ratio percentage for the INT8 SDXL V3.0 path.
- **Latent:** 32–256, default 128 — calibration latent size (H/W). Use `--latent 32` for faster calibration, `--latent 256` for higher fidelity.

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| Naive FP8 | 0.75–0.93 | 50% | High |
| **HSWQ FP8 V1** | **0.94–0.98** | 60-70% (FP16 mixed) | **High** |
| **HSWQ INT8** | **0.94–0.98** | 60-70% (FP16 mixed) | **High** (ComfyUI INT8) |
| **HSWQ FP8 V2** | — (currently unmeasurable) | 60-70% (FP16 mixed) | Not usable (no dedicated loader) |

HSWQ FP8 V1 and HSWQ INT8 target **SSIM 0.94–0.98** with full loader compatibility on their respective formats. FP8 V2 would require a dedicated loader; benchmark is currently unmeasurable and V2 is not usable at the current time.

---

## Changelog

Version history and release notes are in [CHANGELOG.md](CHANGELOG.md).

---

## Base Repositories

This project is built upon the following repositories:

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** — The most powerful and modular diffusion model GUI, API and backend with a graph/nodes interface by [@Comfy-Org](https://github.com/Comfy-Org).
