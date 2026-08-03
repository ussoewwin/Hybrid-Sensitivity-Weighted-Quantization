---
license: other
tags:
- text-to-image
- z-image
- zi
- sdxl
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

High-fidelity ConvRot NVFP4 quantization for diffusion models (Z-Image / SDXL). HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast. This is highly useful for users who need to strictly manage their VRAM resources while maintaining maximum image quality.

ComfyUI Load Diffusion Model `nvfp4` pack with **FULL ConvRot** (Linear→NVFP4, Conv2d→INT8 `int8_tensorwise`) after DualMonitor + V4 pack-MSE FP16 protection under a fixed budget. calib writes NVFP4 `.input_scale`. Z-Image pack scripts: `hswq_convert_nvfp4_1.0.py` (HSWQ) and `native_convert_nvfp4.py` (native).

**Technical details:** [https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

**How to quantize (SDXL/Z-Image ConvRot NVFP4):** [md/How to quantize SDXL NVFP4.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL%20NVFP4.md)

**ComfyUI Loader for ConvRot NVFP4:** To use these models in ComfyUI, please use this custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

**Post-convert fidelity bench (integrated, default ON):** After save, `hswq_convert_nvfp4_1.0.py` and `native_convert_nvfp4.py` **clear parent VRAM**, then automatically run `benchmark/nvfp4bench_sdxl.py` with `--fp16` = the FP16 input, `--nvfp4` = the saved pack, and a fixed `--prompt` / `--seed` (not inventable parent CLI overrides). Pass `--no-bench` to skip.

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| **HSWQ Z-Image ConvRot NVFP4** | **0.97** | 60% (FP16 mixed) | **High** (ComfyUI NVFP4) |

---

## 📦 Available Models

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `moodyProMix_zitV13_hswq_int8protect60_convrot_nvfp4.safetensors` | Moody Pro Mix | v1.3 | CreativeML Open RAIL++-M |
| `moodyRealMix_zitV7_hswq_int8protect60_convrot_nvfp4.safetensors` | Moody Real Mix | v7.0 | CreativeML Open RAIL++-M |

---

## 📜 Credits & License

### 🏆 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **Moody Pro Mix / Moody Real Mix**: Created by [catlover1937](https://civitai.com/user/catlover1937).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
