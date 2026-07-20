# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity **ConvRot INT8** and **ConvRot NVFP4** quantization for **SDXL**, **Flux1.dev**, and **Z Image Turbo** diffusion models. HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast.

- **ConvRot INT8 (SDXL V3.1):** ComfyUI-compatible `int8_tensorwise` pack with **FULL ConvRot** on remaining Linear/Conv2d after DualMonitor + V4 weighted-histogram FP16 protection under a fixed **300 MiB** budget. Keep ratio is **0** (r0); critical layers stay FP16 via automatic analysis, not a keep-ratio percentage. Pack path matches `native_convert_int8_convrot.py`.
- **Z Image INT8 (HSWQ):** **Development and public release ended.** For Z Image, **native ConvRot INT8** already reaches roughly **SSIM > 0.99** in general, so a separate HSWQ Z Image 8-bit line is no longer developed or published. Use native ConvRot INT8 for Z Image 8-bit; HSWQ INT8 work continues for **SDXL**.

**Technical details (FP8):** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md)  
**Technical details (INT8 FP16-protect / pack overview — ConvRot pack guide not published yet):** [md/HSWQ_INT8_SDXL_Technical_Guide.md](md/HSWQ_INT8_SDXL_Technical_Guide.md)

**SDXL models (FP8):** [Hugging Face — Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-fp8e4m3)

**SDXL models (ConvRot INT8):** [Hugging Face — Hybrid-Sensitivity-Weighted-Quantization-SDXL-ConvRot-INT8](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-ConvRot-INT8)

**ComfyUI Loader for ConvRot INT8 / INT8:** To use these models in ComfyUI, please use this custom node: [ComfyUI-nunchaku-unofficial-loader](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/logo.png" width="400">
</p>

---

## How to quantize

- **SDXL (ConvRot INT8):** [How to quantize SDXL ConvRot INT8](md/How%20to%20quantize%20SDXL.md)
- **Z Image / Z-Anime (native ConvRot INT8):** [How to quantize Z Image](md/How%20to%20quantize%20Z%20Image.md)

**Benchmark results:**
- **SDXL (FP8):** [MSE / SSIM](test/benchmark_test.md)
- **SDXL (ConvRot INT8):** [MSE / SSIM](test/benchmark_sdxl_int8.md)

---

## Overview

| Feature | FP8 V1: Standard Compatible | FP8 V2: High Performance Scaled | ConvRot INT8 (SDXL V3.1) |
| :--- | :--- | :--- | :--- |
| **Compatibility** | Full (100%), any FP8 loader | Requires dedicated loader — **not usable at present** | ComfyUI `int8_tensorwise` / QUANT_ALGOS compatible |
| **File format** | Standard FP8 (`torch.float8_e4m3fn`) | Extended FP8 (weights + `.scale` metadata) | INT8 weights + scale (`int8_tensorwise`); SDXL V3.1 packs remainder with **FULL ConvRot** |
| **Image quality (SSIM)** | **0.94–0.98** | Unmeasurable (no dedicated loader) | **0.94–0.98** |
| **Mechanism** | Optimal clipping (smart clipping) | Full-range scaling (dynamic scaling) | Absmax + DualMonitor / V4 FP16 protect (r0); then FULL ConvRot on Linear/Conv2d remainder |
| **Keep ratio** | 5–25% (see How-to; SDXL/ZIT often 10%) | 5–25% (see How-to) | **0 (fixed)** |
| **Benchmark** | Measurable | Currently unmeasurable (no dedicated loader) | Measurable |
| **Use case** | Distribution, general users | Unavailable until a dedicated loader exists | SDXL ConvRot INT8 distribution / kitchen loaders |

**Note (Z Image 8-bit):** HSWQ Z Image INT8 development and publication **ended**. Native ConvRot INT8 is sufficient for Z Image (typically **SSIM > 0.99**). HSWQ INT8 remains the SDXL path.

File size is reduced by about **30–40%** vs FP16 while keeping best quality per use case.

---

## Architecture

1. **Dual Monitor System** — During calibration, two metrics are collected:
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted → kept in FP16 when selected by HSWQ (**FP8:** top 5–25% by keep ratio; for SDXL and ZIT, 10% often gives sufficient quality. **ConvRot INT8 (SDXL):** keep ratio **0**; FP16 set comes from automatic analysis / budget ranking, not a keep-ratio %).
   - **Importance:** V1 uses per-channel input mean-abs; V4 uses per-element SVD leverage × RMS magnitude hybrid → weights of the weighted histogram.
   **Technical details:** [Dual Monitor System — Technical Guide](md/Dual_Monitor_System_Technical_Guide.md).

2. **Rigorous grid / pack simulation**
   - **FP8:** physical grid (all 0–255 values cast to `torch.float8_e4m3fn`) instead of theoretical formulas, so MSE matches real runtime.
   - **ConvRot INT8 (SDXL):** natural absmax pack point for the symmetric INT8 grid; V4 weighted-histogram MSE ranks FP16 protection candidates (does not choose pack amax). **SDXL V3.1** then applies **FULL ConvRot** (Hadamard rotate → channelwise absmax) on remaining Linear/Conv2d, identical to `native_convert_int8_convrot.py`.

3. **Weighted MSE Optimization** — Finds parameters that minimize quantization error using an importance-weighted histogram (not a plain frequency histogram).
   - **V1 / Fast:** per-channel importance (activation mean-abs) drives the histogram amax search. **Technical details:** [Weighted Histogram MSE — Technical Guide](md/Weighted_Histogram_MSE_Technical_Guide.md).
   - **V4 (SVD × RMS hybrid):** per-element importance blends **SVD structural leverage** \(L(i,j)=(U_i\cdot\sigma)^2\cdot(V_j)^2\) with **RMS magnitude**; \(\alpha\) tilts toward SVD on heavy-tailed layers. Used by Z Image / Z-Anime FP8 for amax search, and by **SDXL ConvRot INT8** for FP16-candidate ranking at the absmax pack point. **Technical details:** [HSWQ V4 SVD-RMS — Technical Guide](md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md).

---

## Modes

### FP8

- **V1** (`scaled=False`): No scaling; only the clipping threshold (amax) is optimized. Output is standard FP8 weights. **Use this mode** — full compatibility with any FP8 loader.
- **V2** (`scaled=True`): Weights are scaled to FP8 range, quantized, and inverse scale `S` is stored in Safetensors (`.scale`). Requires a dedicated loader; **not usable at the current time.**

### ConvRot INT8 (SDXL)

- **Script:** `quantize_sdxl_hswq_v3.1.py` (SDXL ConvRot INT8; **300 MiB** FP16 budget; FULL ConvRot default ON).
- **SDXL V3.1 order:** (1) FP16 keep via DualMonitor + analyze + V4 under the 300 MiB budget → (2) remaining Linear/Conv2d **FULL ConvRot INT8** (`native_convert_int8_convrot.py` pack path). Card 1 / Card 2 forced OFF.
- **Tensorwise / channelwise:** format tag `int8_tensorwise`. ConvRot layers use rotate → channelwise absmax; non-ConvRot remainder uses per-tensor absmax unless Card 3 is enabled.
- **Card 3** (`--per_channel_int8`): per-output-channel amax / scale for non-ConvRot plain packs (SDXL).
- **Keep ratio:** **0 (fixed)** — FP16 protection is automatic (analyze Hard VETO + DualMonitor + V4 ranking inside the FP16 budget), not a percentage keep-ratio.
- **Z Image INT8 (HSWQ):** **Ended** — no further HSWQ Z Image 8-bit development or Hugging Face publication. Prefer **native ConvRot INT8** for Z Image (typically **SSIM > 0.99**).

---

## Recommended Parameters

- **Samples:** 32 (recommended) — number of calibration samples (**same for FP8 and ConvRot INT8**).
- **Steps:** 25 — number of inference steps per sample during calibration (**same for FP8 and ConvRot INT8**).
- **Keep ratio (FP8):** 5–25% — keeps critical layers in FP16. For SDXL and ZIT, 10% often gives sufficient quality.
- **Keep ratio (ConvRot INT8, SDXL):** **0 (fixed)** — do not use a non-zero keep-ratio percentage for the SDXL V3.1 path.
- **Latent:** 32–256, default 128 — calibration latent size (H/W). Use `--latent 32` for faster calibration, `--latent 256` for higher fidelity.
- **ConvRot group size (SDXL V3.1):** power of 4 (default from script CLI); `--no-convrot` disables FULL ConvRot (plain pack only).

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| Naive FP8 | 0.75–0.93 | 50% | High |
| **HSWQ FP8 V1** | **0.94–0.98** | 60-70% (FP16 mixed) | **High** |
| **HSWQ ConvRot INT8** | **0.94–0.98** | 60-70% (FP16 mixed) | **High** (ComfyUI INT8) |
| **HSWQ FP8 V2** | — (currently unmeasurable) | 60-70% (FP16 mixed) | Not usable (no dedicated loader) |

HSWQ FP8 V1 and HSWQ ConvRot INT8 target **SSIM 0.94–0.98** with full loader compatibility on their respective formats. FP8 V2 would require a dedicated loader; benchmark is currently unmeasurable and V2 is not usable at the current time.

---

## Changelog

Version history and release notes are in [CHANGELOG.md](CHANGELOG.md).

---

## Base Repositories

This project is built upon the following repositories:

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** — The most powerful and modular diffusion model GUI, API and backend with a graph/nodes interface by [@Comfy-Org](https://github.com/Comfy-Org).
