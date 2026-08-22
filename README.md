# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity **ConvRot INT8** and **ConvRot NVFP4** quantization for **SDXL**, **Flux1.dev**, and **Z Image Turbo** diffusion models. HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast.

- **ConvRot INT8 (SDXL V3.1):** ComfyUI-compatible `int8_tensorwise` pack with **FULL ConvRot** on remaining Linear/Conv2d after DualMonitor + V4 weighted-histogram FP16 protection under a fixed **300 MiB** budget. Keep ratio is **0** (r0); critical layers stay FP16 via automatic analysis, not a keep-ratio percentage. Pack path matches `native_convert_int8_convrot.py`.
- **ConvRot NVFP4 (SDXL):** ComfyUI Load Diffusion Model `nvfp4` pack with **FULL ConvRot** (Linear→NVFP4, Conv2d→INT8 `int8_tensorwise`) after DualMonitor + V4 pack-MSE FP16 protection under a fixed **600 MiB** budget. Keep ratio is **0** (r0); calib writes NVFP4 `.input_scale`. Script: `hswq_convert_nvfp4_convrot_1.0.py`.
- **Z Image INT8 (HSWQ):** **Development and public release ended.** For Z Image, **native ConvRot INT8** already reaches roughly **SSIM > 0.99** in general, so a separate HSWQ Z Image 8-bit line is no longer developed or published. Use native ConvRot INT8 for Z Image 8-bit; HSWQ INT8 work continues for **SDXL**.
- **Z Image Hybrid ConvRot NVFP4:** Built from a complete **native ConvRot INT8** UNet via the **reverse method** — layers are converted to NVFP4 in ascending order of per-layer impact (lowest-impact first). Unlike the conventional "protect top-important layers" approach, this stays in the low-error regime where single-layer ranking is valid. The number of NVFP4 layers varies per model (e.g. nv60–nv110). Validated at all-seed **decoded SSIM >= 0.97** with the unmodified native bench. Script: `Z_Image/diag_impact.py`.

**Technical details (FP8):** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md) — **FP8 development has ended**; this document is retained as a technical asset.  
**Technical details (INT8 FP16-protect / pack overview — ConvRot pack guide not published yet):** [md/HSWQ_INT8_SDXL_Technical_Guide.md](md/HSWQ_INT8_SDXL_Technical_Guide.md)  
**Technical details (V5 histogram cosine):** [md/HSWQ_V5_Hybrid_SVD_RMS_Cosine_Technical_Guide.md](md/HSWQ_V5_Hybrid_SVD_RMS_Cosine_Technical_Guide.md) — Stage-3 amax search with the same SVD×RMS hybrid importance as V4, but **cosine similarity loss** on the importance-weighted magnitude histogram (not weighted MSE); includes a full MSE↔cosine mathematical comparison for quantization fidelity.

**ComfyUI Loader for ConvRot INT8 / INT8:** To use these models in ComfyUI, please use this custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)  
**ComfyUI Loader for ConvRot NVFP4:** To use these models in ComfyUI, please use this custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/logo.png" width="400">
</p>

---

## How to quantize

> **ComfyUI Node Integration:** Progressive support for direct in-graph quantization inside ComfyUI via custom nodes (`comfyui_nodes/`) is currently underway. Support is already available for **native ConvRot INT8** (`Native ConvRot INT8 Quantize`), and additional model architectures / quantization formats will be rolled out sequentially.

- **SDXL (ConvRot INT8):** [How to quantize SDXL ConvRot INT8](md/How%20to%20quantize%20SDXL.md)
- **SDXL (ConvRot NVFP4):** [How to quantize SDXL ConvRot NVFP4](md/How%20to%20quantize%20SDXL%20NVFP4.md)
- **Z Image (native ConvRot INT8):** [How to quantize Z Image](md/How%20to%20quantize%20Z%20Image.md) — CLI and ComfyUI custom node (`Native ConvRot INT8 Quantize`) quantization guide. HSWQ-specific Z Image development has **ended**; this How-to introduces the **general** ConvRot INT8 quantization method.
- **Z Image (Hybrid NVFP4, reverse method):** [How to quantize Z Image - Hybrid NVFP4](md/How%20to%20quantize%20Z%20Image%20-%20Hybrid%20NVFP4.md) — build a hybrid NVFP4 model from the native ConvRot INT8 by converting the lowest-impact layers first (reverse method); validated at all-5-seed SSIM >= 0.97 with the unmodified native bench.

**Benchmark results:**
- **SDXL (ConvRot INT8):** [MSE / SSIM](benchmark%20result/benchmark_sdxl_int8.md)
- **SDXL (ConvRot NVFP4):** [MSE / SSIM](benchmark%20result/benchmark_convrotnvfp4.md)
- **Krea2 (ConvRot INT8):** [MSE / SSIM](benchmark%20result/benchmark_krea2_int8.md)
- **Krea2 (Hybrid NVFP4):** [MSE / SSIM](benchmark%20result/benchmark_krea2_nvfp4.md)
- **Z Image (ConvRot INT8):** [MSE / SSIM](benchmark%20result/benchmark_zi_int8.md)
- **Z Image (Hybrid NVFP4):** [MSE / SSIM](benchmark%20result/benchmark_zi_nvfp4.md)

---

## Overview

| Feature | ConvRot INT8 (SDXL V3.1) | ConvRot NVFP4 (SDXL) | Z Image Hybrid ConvRot NVFP4 |
| :--- | :--- | :--- | :--- |
| **Compatibility** | ComfyUI `int8_tensorwise` / QUANT_ALGOS compatible | ComfyUI Load Diffusion Model / QUANT_ALGOS `nvfp4` compatible | ComfyUI Load Diffusion Model / QUANT_ALGOS `nvfp4` compatible |
| **File format** | INT8 weights + scale (`int8_tensorwise`); SDXL V3.1 packs remainder with **FULL ConvRot** | Linear **NVFP4** + Conv2d **INT8** (`int8_tensorwise`); **FULL ConvRot** on eligible layers | Hybrid: lowest-impact layers → **NVFP4**, remaining layers stay **ConvRot INT8**; built from complete native ConvRot INT8 UNet |
| **Image quality (SSIM)** | **0.94-0.98** | **0.92-0.98** | **0.97-0.99** |
| **Mechanism** | Absmax + DualMonitor / V4 FP16 protect (r0); then FULL ConvRot on Linear/Conv2d remainder | Absmax + DualMonitor / V4 FP16 protect (r0, **600 MiB**); FULL ConvRot (Linear→NVFP4, Conv2d→INT8) | **Reverse method**: start from ConvRot INT8 (error ≈ 0), convert layers to NVFP4 in ascending per-layer impact order |
| **Keep ratio** | **0 (fixed)** | **0 (fixed)** | N/A (layer count varies per model, e.g. nv60-nv110) |
| **Benchmark** | Measurable | Measurable | Measurable |
| **Use case** | SDXL ConvRot INT8 distribution / kitchen loaders | SDXL ConvRot NVFP4 distribution / native ComfyUI load | Z Image Turbo Hybrid NVFP4 distribution / native ComfyUI load |

**Note (Z Image 8-bit):** HSWQ Z Image INT8 development and publication **ended**. Native ConvRot INT8 is sufficient for Z Image (typically **SSIM > 0.99**). HSWQ INT8 remains the SDXL path.

File size is reduced by about **30-40%** vs FP16 while keeping best quality per use case.

---

## Architecture

1. **Dual Monitor System** — During calibration, two metrics are collected:
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted → kept in FP16 when selected by HSWQ. **ConvRot INT8 / NVFP4 (SDXL):** keep ratio **0**; FP16 set comes from automatic analysis / budget ranking, not a keep-ratio %.
   - **Importance:** V1 uses per-channel input mean-abs; V4 uses per-element SVD leverage × RMS magnitude hybrid → weights of the weighted histogram.
   **Technical details:** [Dual Monitor System — Technical Guide](md/Dual_Monitor_System_Technical_Guide.md).

2. **Rigorous grid / pack simulation**
   - **ConvRot INT8 (SDXL):** natural absmax pack point for the symmetric INT8 grid; V4 weighted-histogram MSE ranks FP16 protection candidates (does not choose pack amax). **SDXL V3.1** then applies **FULL ConvRot** (Hadamard rotate → channelwise absmax) on remaining Linear/Conv2d, identical to `native_convert_int8_convrot.py`.
   - **ConvRot NVFP4 (SDXL):** absmax pack point for Linear→NVFP4 and Conv2d→INT8; V4 pack-MSE ranks FP16 protection under the **600 MiB** budget, then **FULL ConvRot** on eligible remainder.

3. **Weighted Histogram Optimization** — Finds parameters that minimize quantization error using an importance-weighted histogram (not a plain frequency histogram).
   - **V1 / Fast:** per-channel importance (activation mean-abs) drives the histogram. **Technical details:** [Weighted Histogram MSE — Technical Guide](md/Weighted_Histogram_MSE_Technical_Guide.md).
   - **V4 (SVD × RMS hybrid, MSE):** per-element importance blends **SVD structural leverage** \(L(i,j)=(U_i\cdot\sigma)^2\cdot(V_j)^2\) with **RMS magnitude**; \(\alpha\) tilts toward SVD on heavy-tailed layers. Used by **SDXL ConvRot INT8** and **ConvRot NVFP4** for FP16-candidate ranking at the absmax pack point. **Technical details:** [HSWQ V4 SVD-RMS — Technical Guide](md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md).
   - **V5 (SVD × RMS hybrid, cosine):** same hybrid importance family as V4; Stage-3 objective is **importance-weighted cosine loss** \(L=1-\langle x,q\rangle_H/(\|x\|_H\|q\|_H)\) against the physical FP8 E4M3 quantize–dequantize map (scale-invariant angular fidelity; reduces absolute-tail blackmail of \(\Delta^*\)). Source: `histogram/weighted_histogram_cosine_v5.py`. **Technical details:** [HSWQ V5 SVD-RMS Cosine — Technical Guide](md/HSWQ_V5_Hybrid_SVD_RMS_Cosine_Technical_Guide.md).
4. **Trajectory-Sensitivity Impact Ranking** — Ranks each layer by the divergence its
   quantization error actually causes after propagating through the full model and sampler
   (dynamical importance, replacing static weight-space saliency).
   - **Reverse method:** start from the complete high-precision pack (error ≈ 0) and convert
     layers to lower precision in ascending impact order; single-layer ranking stays valid in the
     low-error additivity regime.
   - **Universal theory:** error interaction (Taylor cross terms, error cancellation), nonlinear
     amplification (Lyapunov-style growth), marginal effects, and Shapley-style attribution —
     why per-layer static measures (histogram MSE / cosine / SVD) cannot predict joint quantization
     error; applies to any iterative sampling system, not a specific model. Source:
     `Z_Image/diag_impact.py`. **Technical details:** [Trajectory-Sensitivity Impact Ranking —
     Technical Guide](md/diag_impact_trajectory_sensitivity_technical_guide.md).

---

## Modes

### ConvRot INT8 (SDXL)

- **Script:** `quantize_sdxl_hswq_v3.1.py` (SDXL ConvRot INT8; **300 MiB** FP16 budget; FULL ConvRot default ON).
- **SDXL V3.1 order:** (1) FP16 keep via DualMonitor + analyze + V4 under the 300 MiB budget → (2) remaining Linear/Conv2d **FULL ConvRot INT8** (`native_convert_int8_convrot.py` pack path). Card 1 / Card 2 forced OFF.
- **Tensorwise / channelwise:** format tag `int8_tensorwise`. ConvRot layers use rotate → channelwise absmax; non-ConvRot remainder uses per-tensor absmax unless Card 3 is enabled.
- **Card 3** (`--per_channel_int8`): per-output-channel amax / scale for non-ConvRot plain packs (SDXL).
- **Keep ratio:** **0 (fixed)** — FP16 protection is automatic (analyze Hard VETO + DualMonitor + V4 ranking inside the FP16 budget), not a percentage keep-ratio.
- **Z Image INT8 (HSWQ):** **Ended** — no further HSWQ Z Image 8-bit development or Hugging Face publication. Prefer **native ConvRot INT8** for Z Image (typically **SSIM > 0.99**).

### ConvRot NVFP4 (SDXL)

- **Script:** `hswq_convert_nvfp4_convrot_1.0.py` (SDXL ConvRot NVFP4; **600 MiB** FP16 budget; FULL ConvRot default ON).
- **Order:** (1) FP16 keep via DualMonitor + analyze + V4 under the **600 MiB** hard ceiling → (2) remaining eligible layers **FULL ConvRot** — Linear → **NVFP4**, Conv2d → **INT8** (`int8_tensorwise`).
- **Loader / format:** ComfyUI Load Diffusion Model / QUANT_ALGOS `nvfp4`; NVFP4 `.input_scale` from PTQ calib (`--calib_file`).
- **Keep ratio:** **0 (fixed)** — FP16 protection is automatic (analyze Hard VETO + DualMonitor + V4 pack-MSE ranking inside the FP16 budget), not a percentage keep-ratio.
- **Image quality (SSIM):** **0.92-0.98**.

### Z Image Hybrid ConvRot NVFP4

- **Script:** `Z_Image/diag_impact.py` (per-layer impact diagnosis + automatic NVFP4 conversion).
- **Prerequisite:** A complete **native ConvRot INT8** UNet created by `native_convert_int8_convrot_zi.py`.
- **Method:** **Reverse method** — start from the complete ConvRot INT8 model (error ≈ 0) and convert layers to NVFP4 in **ascending order of per-layer impact** (lowest-impact first). Unlike the conventional "protect top-important layers" approach, this stays in the low-error regime where single-layer ranking is valid.
- **Loader / format:** ComfyUI Load Diffusion Model / QUANT_ALGOS `nvfp4`; same pack format as SDXL ConvRot NVFP4.
- **NVFP4 layer count:** Varies per model (e.g. nv60-nv110); determined automatically by impact diagnosis.
- **Image quality (SSIM):** **0.97-0.99**.

---

## Recommended Parameters

- **Samples:** 32 (recommended) — number of calibration samples (**same for FP8 and ConvRot INT8**).
- **Steps:** 25 — number of inference steps per sample during calibration (**same for FP8 and ConvRot INT8**).
- **ConvRot group size (SDXL V3.1):** power of 4 (default from script CLI); `--no-convrot` disables FULL ConvRot (plain pack only).

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| Naive FP8 | 0.75-0.93 | 50% | High |
| **HSWQ ConvRot INT8** | **0.94-0.98** | **68%** (FP16 mixed) | **High** (ComfyUI INT8) |
| **HSWQ ConvRot NVFP4** | **0.92-0.98** | **60%** (FP16 mixed) | **High** (ComfyUI NVFP4) |
| **Z Image Hybrid NVFP4** | **0.97-0.99** | **60%** (FP16 mixed) | **High** (ComfyUI NVFP4) |

HSWQ ConvRot INT8 targets **SSIM 0.94-0.98**; HSWQ ConvRot NVFP4 targets **SSIM 0.92-0.98**; Z Image Hybrid NVFP4 targets **SSIM 0.97-0.99**. All keep full loader compatibility on their respective formats.

---

## Changelog

Version history and release notes are in [CHANGELOG.md](CHANGELOG.md).

---

## Base Repositories

This project is built upon the following repositories:

| Repository | In-repo path | Upstream |
| :--- | :--- | :--- |
| **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** | `ComfyUI-master/` | [@Comfy-Org](https://github.com/Comfy-Org) — The most powerful and modular diffusion model GUI, API and backend with a graph/nodes interface. |

