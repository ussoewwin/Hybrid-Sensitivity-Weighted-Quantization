---
license: other
tags:
- text-to-image
- sdxl
- nunchaku
- svdq
- quantized
- w4a4
- fp4
- int4
- r128
- bluepencil
- illustrious
- realvisxl
- photorealistic
- comfyui
- controlnet
- anime
library_name: nunchaku
---


# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity FP8 quantization for diffusion models (SDXL). HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast, and offers two modes: standard-compatible (V1) and high-performance scaled (V2).

**Technical details:** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md)

**How to quantize:** [md/HSWQ_ How to quantize SDXL.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL.md)

**SDXL Benchmark Test Results:** [md/SDXL Benchmark Test Results.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/test/benchmark_test.md)

---

## Overview

| Feature | V1: Standard Compatible | V2: High Performance Scaled |
| :--- | :--- | :--- |
| **Compatibility** | Full (100%), any FP8 loader | Custom loader (HSWQLoader) required |
| **File format** | Standard FP8 (`torch.float8_e4m3fn`) | Extended FP8 (weights + `.scale` metadata) |
| **Image quality (SSIM)** | ~0.98 (theoretical limit) | Unmeasurable (no dedicated loader) |
| **Mechanism** | Optimal clipping (smart clipping) | Full-range scaling (dynamic scaling) |
| **Use case** | Distribution, general users | In-house, max quality, server-side |

File size is reduced by about **60-70%** vs FP16 while keeping best quality per use case.

---

## Architecture

1. **Dual Monitor System** — During calibration, two metrics are collected:
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted → top 5–25% kept in FP16 (for SDXL and ZIT, 10% often gives sufficient quality).
   - **Importance** (input mean absolute value): per-channel contribution → used as weights in the weighted histogram.

2. **Rigorous FP8 Grid Simulation** — Uses a physical grid (all 0–255 values cast to `torch.float8_e4m3fn`) instead of theoretical formulas, so MSE matches real runtime.

3. **Weighted MSE Optimization** — Finds parameters that minimize quantization error using the importance histogram.

---

## Modes

- **V1** (`scaled=False`): No scaling; only the clipping threshold (amax) is optimized. Output is standard FP8 weights. Use when you need maximum compatibility.
- **V2** (`scaled=True`): Weights are scaled to FP8 range, quantized, and inverse scale `S` is stored in Safetensors (`.scale`). Unavailable until a dedicated loader exists.

---

## Recommended Parameters

- **Samples:** 32 (recommended).
- **Keep ratio:** 0.25 (25%) in the example; the valid range is typically `0.05`–`0.25` (5–25%). For SDXL, 0.25 often gives sufficient quality. Adjust if you want to trade off quality vs. memory/speed.
- **Steps:** 25(recommended). — to include early denoising sensitivity.
---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| Naive FP8 | 0.81-0.93 | 50% | High |
| **HSWQ V1** | **0.86–0.98** | 60-70% (FP16 mixed) | **High** |
| **HSWQ V2** | Unmeasurable (no dedicated loader) | 60-70% (FP16 mixed) | Low (custom loader) |

HSWQ V1 gives a clear gain over Naive FP8 with full compatibility; V2 targets maximum quality with a custom loader.

### 2. Setup
- **VAE:** Use standard SDXL VAE (place in `models/vae/`)

---

## 📦 Available Models

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `realvisxlV50_v50Bakedvae_r32_r0.1.safetensors` | [RealVisXL V5.0 (BakedVAE)](https://civitai.red/models/139562) | v5.0 | CreativeML Open RAIL++-M |
| `waiREALCN_v150_hswq_r32_r0.15_v1.safetensors` | [WAI-REAL_CN](https://civitai.red/models/469902) | v15.0 | Pony License |
| `waiANIPONYXL_v140_hswq_r32_r0.15_v1.safetensors` | [WAI-ANI-PONYXL](https://civitai.red/models/404154) | v14.0 | Pony License |
| `waiIllustriousSDXL_v160_hswq_r32_r0.1_v1.safetensors` | [WAI-illustrious-SDXL](https://civitai.red/models/827184/wai-illustrious-sdxl) | v16.0 | Illustrious License |
| `waiIllustriousSDXL_v170_hswq_r32_r0.1_v1.safetensors` | [Illustrious-XL v1.7 (WAI-illustrious-SDXL)](https://civitai.red/models/827184/wai-illustrious-sdxl) | v17.0 (HF weight) | Illustrious License |
| `waiREALISM_v10_hswq_r32_r0.1_v1.safetensors` | [WAI-REALISM-Illustrious](https://civitai.red/models/2233797) | v1.0 | Illustrious License |
| `novaAsianXL_illustriousV70_r32_r0.1.safetensors` | [Nova Asian XL](https://civitai.red/models/641919/nova-asian-xl) | Illustrious v7.0 | Illustrious License |
| `perfectionAsianILXL_v10_r32_r0.1.safetensors` | [Perfection Asian [ILXL / Illustrious XL]](https://civitai.red/models/1518448/perfection-asian-ilxl-illustrious-xl--sfw-checkpoint) | v1.0 | Illustrious License |
| `perfectionRealisticILXL_60_r32_r0.1.safetensors` | [Perfection Realistic [ILXL / Illustrious XL]](https://civitai.red/models/1257570) | v6.0 | Illustrious License |
| `prefectIllustriousXL_v70_r32_r0.1.safetensors` | [Prefect illustrious XL](https://civitai.red/models/1224788) | v7.0 | Illustrious License |
| `unholyDesireMixSinister_v80_hswq_r32_r0.1_v1.safetensors` | [Unholy Desire Mix - Sinister Aesthetic (Illustrious)](https://civitai.red/models/1307857/unholy-desire-mix-sinister-aesthetic-illustrious) | v8.0 | Illustrious License |
| `JANKUTrainedChenkinNoobai_v777_hswq_r32_r0.1_v1.safetensors` | [JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)](https://civitai.red/models/1277670/janku-trained-chenkin-and-noobai-rouwei-illustrious-xl) | v777 | Illustrious License |
| `animagineXLV31_v30_hswq_r32_r0.1_v1.safetensors` | [Animagine XL 3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1) | v3.1 | CreativeML Open RAIL++-M |

---

## 📜 Credits & License

### 🏆 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **RealVisXL V5.0**: Created by [SG_161222](https://civitai.red/user/SG_161222).
- **WAI-REAL_CN / WAI-ANI-PONYXL / WAI-illustrious-SDXL / WAI-REALISM-Illustrious**: Created by [WAI0731](https://civitai.red/user/WAI0731).
- **Nova Asian XL** (Illustrious v7.0): Created by [Crody](https://civitai.red/user/Crody).
- **Perfection Asian [ILXL / Illustrious XL]**: Created by [6tZ](https://civitai.red/user/6tZ) (Illustrious XL checkpoint merge).
- **Perfection Realistic [ILXL / Illustrious XL]**: Created by [6tZ](https://civitai.red/user/6tZ) (Illustrious XL checkpoint merge).
- **Prefect illustrious XL**: Created by [Goofy_Ai](https://civitai.red/user/Goofy_Ai).
- **Unholy Desire Mix - Sinister Aesthetic (Illustrious)**: Created by [UnholyDesiresStudio](https://civitai.red/user/UnholyDesiresStudio).
- **JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)**: Created by [janxd](https://civitai.red/user/janxd).
- **Koronemix Vpred v2.0**: Created by [koronen](https://civitai.red/user/koronen).

---
**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.