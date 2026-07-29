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

ComfyUI Load Diffusion Model `nvfp4` pack with **FULL ConvRot** (Linear→NVFP4, Conv2d→INT8 `int8_tensorwise`) after DualMonitor + V4 pack-MSE FP16 protection under a fixed **600 MiB** budget. Keep ratio is **0** (r0); calib writes NVFP4 `.input_scale`. SDXL pack scripts: `hswq_convert_nvfp4_1.0.py` (HSWQ) and `native_convert_nvfp4.py` (native).

**Technical details:** [https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

**How to quantize (SDXL ConvRot NVFP4):** [md/How to quantize SDXL NVFP4.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL%20NVFP4.md)

**ComfyUI Loader for ConvRot NVFP4:** To use these models in ComfyUI, please use this custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

**Post-convert fidelity bench (integrated, default ON):** After save, `hswq_convert_nvfp4_1.0.py` and `native_convert_nvfp4.py` **clear parent VRAM**, then automatically run `benchmark/nvfp4bench_sdxl.py` with `--fp16` = the FP16 input, `--nvfp4` = the saved pack, and a fixed `--prompt` / `--seed` (not inventable parent CLI overrides). Pass `--no-bench` to skip. Standalone re-runs use the same `nvfp4bench_sdxl.py` command shape as in the How-to.

**SDXL ConvRot NVFP4 Benchmark Test Results (published tables):** [test/benchmark_convrotnvfp4.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/test/benchmark_convrotnvfp4.md)

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
| `animemix_v80_hswq_r32_1off_nvfp4.safetensors` | AnimeMix | v8.0 | Fair AI Public License 1.0-SD |
| `epicrealismXL_pureFix_hswq_r32_1off_convrot_nvfp4.safetensors` | [epiCRealism XL](https://civitai.com/models/277058) | pureFix | CreativeML Open RAIL++-M |
| `koronemixIllustrious_v70_hswq_r32_1off_convrot_nvfp4.safetensors` | koronemixIllustrious | v70 | Fair AI Public License 1.0-SD |
| `koronemixVpred_v20_hswq_r32_1on_convrot_nvfp4.safetensors` | koronemixVpred | v2.0 | CreativeML Open RAIL++-M |
| `realvisxlV50_v40Bakedvae_hswq_r32_1off_nvfp4.safetensors` | [RealVisXL V5.0 (Lightning)](https://civitai.com/models/139562/realvisxl-v50) | v4.0 BakedVAE | CreativeML Open RAIL++-M |
| `realvisxlV50_v50Bakedvae_hswq_r32_1off_convrot_nvfp4.safetensors` | [RealVisXL V5.0 (Lightning)](https://civitai.com/models/139562/realvisxl-v50) | v5.0 BakedVAE | CreativeML Open RAIL++-M |
| `unholyDesireMixSinister_v80_hswq_r32_1off_nvfp4.safetensors` | Unholy Desire Mix Sinister | v8.0 | Fair AI Public License 1.0-SD |
| `waiIllustriousSDXL_v170_hswq_r32_1off_convrot_nvfp4.safetensors` | [Illustrious-XL v1.7 (WAI-illustrious-SDXL)](https://civitai.com/models/827184/wai-illustrious-sdxl) | v17.0 (HF weight) | Fair AI Public License 1.0-SD |

---

## 📜 Credits & License

### 🏆 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **AnimeMix / koronemixIllustrious / koronemixVpred**: Created by [koronen](https://civitai.com/user/koronen).
- **epiCRealism XL**: Created by [epinikion](https://civitai.com/user/epinikion).
- **RealVisXL V5.0**: Created by [SG_161222](https://civitai.com/user/SG_161222).
- **Unholy Desire Mix Sinister**: Created by [UnholyDesiresStudio](https://civitai.com/user/UnholyDesiresStudio).
- **WAI-illustrious-SDXL**: Created by [WAI0731](https://civitai.com/user/WAI0731).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
