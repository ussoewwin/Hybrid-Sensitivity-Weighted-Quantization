---
license: other
tags:
- text-to-image
- comfyui
- int8
- quantized
- z-image-turbo
---

# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity INT8 quantization for diffusion models (Z Image Turbo family). HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast.

**Technical details:** [md/HSWQ_INT8_SDXL_Technical_Guide.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_INT8_SDXL_Technical_Guide.md)

**How to quantize:** [How to quantize Z Image.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Z%20Image.md)

**Z Image INT8 Benchmark Test Results:** [test/benchmark_zi_int8.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/test/benchmark_zi_int8.md)

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% (6.5GB) | High |
| Naive INT8 | 0.93-0.97 | 50% | High |
| **HSWQ INT8** | **0.99** | 55% (FP16 mixed) | **High** |

---

## 📦 Available Models

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `darkBeastINT8Convrot2_dbzit9DIMRclaw_hswq_r32_int8.safetensors` | [darkBeastINT8Convrot2_dbzit9DIMRclaw](https://civitai.com/models/2242173) | dbzit9 | Apache 2.0 |
| `moodyProMix_zitV13_hswq_r32_int8.safetensors` | [moodyProMix_zitV13](https://civitai.red/models/620406) | v13 | Apache 2.0 |
| `moodyRealMix_zitV7_hswq_r32_int8_v1.safetensors` | [moodyRealMix_zitV7](https://civitai.red/models/621441) | v7 | Apache 2.0 |

---

## 📜 Credits & License

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **darkBeastINT8Convrot2_dbzit9DIMRclaw**: Created by [AiMetatron](https://civitai.com/user/AiMetatron).
- **moodyProMix_zitV13** / **moodyRealMix_zitV7**: Created by [catlover1937](https://civitai.red/user/catlover1937) (on Civitai).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
