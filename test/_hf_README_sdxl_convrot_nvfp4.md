---
license: other
tags:
- text-to-image
- sdxl
- nunchaku
- svdq
- quantized
- nvfp4
- illustrious
- comfyui
- controlnet
- anime
- faipl-1.0-sd
library_name: nunchaku
---

# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity ConvRot NVFP4 quantization for diffusion models (SDXL). HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast. This is highly useful for users who need to strictly manage their VRAM resources while maintaining maximum image quality.

ComfyUI Load Diffusion Model `nvfp4` pack with **FULL ConvRot** (Linear→NVFP4, Conv2d→INT8 `int8_tensorwise`) after DualMonitor + V4 pack-MSE FP16 protection under a fixed **600 MiB** budget. Keep ratio is **0** (r0); calib writes NVFP4 `.input_scale`. Script: `hswq_convert_nvfp4_convrot_1.0.py`.

**Technical details:** [https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

**How to quantize (SDXL ConvRot NVFP4):** [md/How to quantize SDXL NVFP4.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL%20NVFP4.md)

**ComfyUI Loader for ConvRot NVFP4:** To use these models in ComfyUI, please use this custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

**SDXL ConvRot NVFP4 Benchmark Test Results:** [test/benchmark_convrotnvfp4.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/test/benchmark_convrotnvfp4.md)

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| **HSWQ ConvRot NVFP4** | **0.92-0.98** | 60% (FP16 mixed) | **High** (ComfyUI NVFP4) |

---

## 📦 Available Models

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `waiIllustriousSDXL_v170_hswq_r32_convrot_nvfp4.safetensors` | [Illustrious-XL v1.7 (WAI-illustrious-SDXL)](https://civitai.com/models/827184/wai-illustrious-sdxl) | v17.0 (HF weight) | Fair AI Public License 1.0-SD |
| `JANKUTrainedChenkinNoobai_v777_hswq_r32_+200_nvfp4.safetensors` | [JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)](https://civitai.com/models/1277670/janku-trained-chenkin-and-noobai-rouwei-illustrious-xl) | v777 | Fair AI Public License 1.0-SD |
| `animemix_v80_hswq_r32_nvfp4.safetensors` | AnimeMix | v8.0 | Fair AI Public License 1.0-SD |
| `koronemixIllustrious_v70_hswq_r32_convrot_nvfp4.safetensors` | koronemixIllustrious | v70 | Fair AI Public License 1.0-SD |

---

## 📜 Credits & License

### 🏆 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **WAI-illustrious-SDXL**: Created by [WAI0731](https://civitai.com/user/WAI0731).
- **JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)**: Created by [janxd](https://civitai.com/user/janxd).
- **AnimeMix / koronemixIllustrious**: Created by [koronen](https://civitai.com/user/koronen).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
