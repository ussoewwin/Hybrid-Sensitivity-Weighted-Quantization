---
license: other
tags:
- text-to-image
- sdxl
- nunchaku
- svdq
- quantized
- int8
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

High-fidelity INT8 quantization for diffusion models (SDXL). HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast.

**Technical details:** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md)

**How to quantize:** [md/HSWQ_ How to quantize SDXL.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL.md)

**SDXL Benchmark Test Results:** [test/benchmark_sdxl_int8.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/test/benchmark_sdxl_int8.md)

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| Naive INT8 | 0.96 | 50% | High |
| **HSWQ INT8** | **0.98** | 60% (FP16 mixed) | **High** |

---

## 📦 Available Models

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `waiIllustriousSDXL_v170_hswq_r32_r0_int8.safetensors` | [Illustrious-XL v1.7 (WAI-illustrious-SDXL)](https://civitai.red/models/827184/wai-illustrious-sdxl) | v17.0 (HF weight) | Illustrious License |
| `prefectIllustriousXL_v8_hswq_r32_r0_int8.safetensors` | [Prefect illustrious XL](https://civitai.red/models/1224788) | v8.0 | Illustrious License |
| `JANKUTrainedChenkinNoobai_v777_hswq_r32_r0_int8.safetensors` | [JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)](https://civitai.red/models/1277670/janku-trained-chenkin-and-noobai-rouwei-illustrious-xl) | v777 | Illustrious License |
| `unholyDesireMixSinister_v80_hswq_r32_r0_int8.safetensors` | [Unholy Desire Mix - Sinister Aesthetic (Illustrious)](https://civitai.red/models/1307857/unholy-desire-mix-sinister-aesthetic-illustrious) | v8.0 | Illustrious License |

---

## 📜 Credits & License

### 🏆 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **WAI-illustrious-SDXL**: Created by [WAI0731](https://civitai.red/user/WAI0731).
- **Prefect illustrious XL**: Created by [Goofy_Ai](https://civitai.red/user/Goofy_Ai).
- **JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)**: Created by [janxd](https://civitai.red/user/janxd).
- **Unholy Desire Mix - Sinister Aesthetic (Illustrious)**: Created by [UnholyDesiresStudio](https://civitai.red/user/UnholyDesiresStudio).

---
