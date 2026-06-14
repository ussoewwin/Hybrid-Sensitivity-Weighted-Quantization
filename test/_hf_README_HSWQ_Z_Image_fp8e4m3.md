---
license: other
tags:
- text-to-image
- comfyui
- fp8
- quantized
- z-image-turbo
---

# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity FP8 quantization for diffusion models (Z Image Turbo family). HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast, and offers two modes: standard-compatible (V1) and high-performance scaled (V2).

**Technical details:** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md)

**How to quantize:** [How to quantize Z Image.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Z%20Image.md)

**Z Image Benchmark Test Results:** [Z Image Benchmark Test Results.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/test/benchmark_zit.md)

---

## Overview

| Feature | V1: Standard Compatible | V2: High Performance Scaled |
| :--- | :--- | :--- |
| **Compatibility** | Full (100%), any FP8 loader | The scaled model does not perform well in the current ComfyUI.
| **File format** | Standard FP8 (`torch.float8_e4m3fn`) | Extended FP8 (weights + `.scale` metadata) |
| **Image quality (SSIM)** | ~0.96 (theoretical limit) | ~Unable to measure at this time |
| **Mechanism** | Optimal clipping (smart clipping) | Full-range scaling (dynamic scaling) |
| **Use case** | Distribution, general users | In-house, max quality, server-side |

File size is reduced by about **60-70%** vs FP16 while keeping best quality per use case.

---

## Architecture

1. **Dual Monitor System** - During calibration, two metrics are collected:
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted - top fraction kept in FP16 per keep ratio.
   - **Importance** (input mean absolute value): per-channel contribution - used as weights in the weighted histogram.

2. **Rigorous FP8 Grid Simulation** - Uses a physical grid (all 0-255 values cast to `torch.float8_e4m3fn`) instead of theoretical formulas, so MSE matches real runtime.

3. **Weighted MSE Optimization** - Finds parameters that minimize quantization error using the importance histogram.

---

## Modes

- **V1** (`scaled=False`): No scaling; only the clipping threshold (amax) is optimized. Output is standard FP8 weights. Use when you need maximum compatibility.
- **V2** (`scaled=True`): Weights are scaled to FP8 range, quantized, and inverse scale `S` is stored in Safetensors (`.scale`). Unavailable until a dedicated loader exists.

---

## Recommended Parameters

- **Samples:** 32 (recommended).
- **Keep ratio:** 0.05-0.25 (5-25%) - keeps critical layers in FP16; for Z Image Turbo (ZIT), 5-10% often gives sufficient quality.
- **Steps:** 25 (recommended) - to include early denoising sensitivity.

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% (6.5GB) | High |
| Naive FP8 | 0.75-0.92 | 50% | High |
| **HSWQ V1** | **0.88-0.99** | 60-70% (FP16 mixed) | **High** |
| **HSWQ V2** | **Unable to measure at this time** | 60-70% (FP16 mixed) | Low (custom loader) |

HSWQ V1 gives a clear gain over Naive FP8 with full compatibility; V2 is unavailable until a dedicated loader exists.

## Available Models

Quantized checkpoints use suffix **`_hswq_r32_r0.05_v1`** (R32 calibration samples, keep ratio **r0.05**, HSWQ **v1**).

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `beyondREALITY_V30_hswq_r32_r0.05_v1.safetensors` | [beyondREALITY_V30](https://civitai.red/models/1090420/beyond-reality?modelVersionId=2648189) | v3.0 | Apache 2.0 |
| `darkBeastMar2126Latest_dbzit8SDAFOK_hswq_r32_r0.05_v1.safetensors` | [darkBeastMar2126Latest_dbzit8SDAFOK](https://civitai.com/models/2242173?modelVersionId=2774410) | v8 | Apache 2.0 |
| `harukiMIX_zit2603_hswq_r32_r0.05_v1.safetensors` | [harukiMIX_zit2603](https://civitai.com/models/856375/harukimix?modelVersionId=2815582) | v2603 | Apache 2.0 |
| `moodyProMix_zitV12DPO_hswq_r32_r0.05_v1.safetensors` | [moodyProMix_zitV12DPO](https://civitai.red/models/620406?modelVersionId=2966200) | v12 | Apache 2.0 |
| `moodyRealMix_zitV5DPO_hswq_r32_r0.05_v1.safetensors` | [moodyRealMix_zitV5DPO](https://civitai.red/models/621441?modelVersionId=2824098) | v5 | Apache 2.0 |
| `moodyRealMix_zitV6DPO_hswq_r32_r0.05_v1.safetensors` | [moodyRealMix_zitV6DPO](https://civitai.red/models/621441?modelVersionId=2922447) | v6 | Apache 2.0 |
| `moodyRealMix_zitV7_hswq_r32_r0_v1.safetensors` | [moodyRealMix_zitV7](https://civitai.red/models/621441) | v7 | Apache 2.0 |
| `moodyWildMix_v02_hswq_r32_r0.05_v1.safetensors` | [moodyWildMix_v02](https://civitai.red/models/2384856?modelVersionId=2698792) | v0.2 | Apache 2.0 |
| `unstableRevolution_V2Fp16_hswq_r32_r0.05_v1.safetensors` | [unstableRevolution_V2Fp16](https://civitai.com/models/2193942/unstable-revolution-zit?modelVersionId=2564070) | v2 | Apache 2.0 |
| `za_hswq_r32_r0.05_v1.safetensors` | [Z-Anime](https://civitai.red/models/2483351/z-anime) | Base | Apache 2.0 |
| `zit_hswq_R32_r0.05_v1.safetensors` | Official **Z Image Turbo** weights redistributed as Comfy split checkpoints: [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) (diffusion file used for quantization: `split_files/diffusion_models/z_image_turbo_bf16.safetensors`) | Turbo official | See upstream repo |

---

## Credits & License

### Base Models

- **beyondREALITY_V30**: Created by [Nurburgring](https://civitai.red/user/Nurburgring) ([BEYOND REALITY](https://civitai.red/models/1090420/beyond-reality?modelVersionId=2648189) on Civitai).
- **darkBeastMar2126Latest_dbzit8SDAFOK**: Created by [AiMetatron](https://civitai.com/user/AiMetatron).
- **harukiMIX_zit2603**: Created by [HARUKI3](https://civitai.com/user/HARUKI3).
- **moodyProMix_zitV12DPO**: Created by [catlover1937](https://civitai.red/user/catlover1937) ([Moody Pro Mix](https://civitai.red/models/620406) on Civitai).
- **moodyRealMix_zitV5DPO** / **moodyRealMix_zitV6DPO** / **moodyRealMix_zitV7**: Created by [catlover1937](https://civitai.red/user/catlover1937) ([Moody Real Mix](https://civitai.red/models/621441) on Civitai).
- **moodyWildMix_v02**: Created by [catlover1937](https://civitai.red/user/catlover1937).
- **unstableRevolution_V2Fp16**: Created by [Peli86](https://civitai.com/user/Peli86).
- **Z-Anime**: Created by [SeeSeeLP](https://civitai.red/user/SeeSeeLP).
- **Official Z Image Turbo (ZIT)**: Distribution [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) on Hugging Face - follow upstream terms.

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
