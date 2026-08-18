---
license: other
tags:
- text-to-image
- z-image
- zi
- nunchaku
- svdq
- quantized
- nvfp4
- comfyui
- controlnet
- photorealistic
- creativeml-openrail-m
library_name: nunchaku
---

# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity Hybrid ConvRot NVFP4 quantization for Z-Image Turbo diffusion models. Built from a complete **native ConvRot INT8** UNet via the **reverse method** (converting lowest-impact layers to NVFP4 in ascending order of trajectory impact). This is highly useful for users who need to strictly manage their VRAM resources (~53–58% savings) while maintaining high image fidelity (**SSIM >= 0.97-0.99**).

**Technical details:** [https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

**How to quantize (Z-Image Hybrid NVFP4):** [md/How to quantize Z Image - Hybrid NVFP4.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Z%20Image%20-%20Hybrid%20NVFP4.md)

**ComfyUI Loader for ConvRot NVFP4:** To use these models in ComfyUI, please use this custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

**Z Image ConvRot NVFP4 Benchmark Test Results (published tables):** [benchmark result/benchmark_zi_nvfp4.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/benchmark%20result/benchmark_zi_nvfp4.md)

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| **HSWQ Z-Image Hybrid ConvRot NVFP4** | **0.97-0.99** | 60% (FP16 mixed) | **High** (ComfyUI NVFP4) |

---

## 📦 Available Models

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `moodyProMix_zitV13_hswq_hybrid_nv80_convrot_nvfp4.safetensors` | Moody Pro Mix | zit v1.3 (nv80) | CreativeML Open RAIL++-M |
| `moodyProMix_collectorsEdition_hswq_hybrid_nv90_convrot_nvfp4.safetensors` | Moody Pro Mix | Collector's Edition (nv90) | CreativeML Open RAIL++-M |
| `moodyRealMix_zitV7_hswq_hybrid_nv100_convrot_nvfp4.safetensors` | Moody Real Mix | zit v7.0 (nv100) | CreativeML Open RAIL++-M |
| `moodyRealMix_xhsEdition_hswq_hybrid_nv110_convrot_nvfp4.safetensors` | Moody Real Mix | XHS Edition (nv110) | CreativeML Open RAIL++-M |
| `darkBeast30BF16INT8_dbzit9DIMRclaw_hswq_hybrid_nv100_convrot_nvfp4.safetensors` | Dark Beast | dbzit9 DIMRclaw (nv100) | CreativeML Open RAIL++-M |
| `unstableRevolution_V3Fp16_hswq_hybrid_nv90_convrot_nvfp4.safetensors` | Unstable Revolution | v3.0 FP16 (nv90) | CreativeML Open RAIL++-M |
| `gonzalomoZpop_insta2_hswq_hybrid_nv80_convrot_nvfp4.safetensors` | gonzalomo Z-Pop | insta2 (nv80) | CreativeML Open RAIL++-M |
| `zimageTurboByStable_2602BF16_hswq_hybrid_nv100_convrot_nvfp4.safetensors` | Z-Image Turbo | 2602 BF16 (nv100) | CreativeML Open RAIL++-M |

---

## 📜 Credits & License

### 🏆 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **Moody Pro Mix / Moody Real Mix / Dark Beast**: Created by [catlover1937](https://civitai.com/user/catlover1937).
- **Unstable Revolution**: Created by [Yamer](https://civitai.com/user/Yamer).
- **gonzalomo Z-Pop**: Created by [gonzalomo](https://civitai.com/user/gonzalomo).
- **Z-Image Turbo**: Created by [Tongyi-MAAS / Civitai Community](https://civitai.com/).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
