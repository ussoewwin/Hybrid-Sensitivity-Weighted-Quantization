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
- faipl-1.0-sd
- creativeml-openrail-m
library_name: nunchaku
---


# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity ConvRot INT8 quantization for diffusion models (SDXL). HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast. This is highly useful for users who need to strictly manage their VRAM resources while maintaining maximum image quality.

ComfyUI-compatible `int8_tensorwise` pack with **FULL ConvRot** on remaining Linear/Conv2d after DualMonitor + V4 weighted-histogram FP16 protection under a fixed **300 MiB** budget. Keep ratio is **0** (r0); critical layers stay FP16 via automatic analysis, not a keep-ratio percentage. SDXL pack scripts: `quantize_sdxl_hswq_v3.1.py` (HSWQ) and `native_convert_int8_sdxl.py` (native).

**Technical details:** [https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

**How to quantize (SDXL ConvRot INT8):** [md/How to quantize SDXL.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL.md)

**ComfyUI Loader for ConvRot INT8 / INT8:** To load these INT8 models in ComfyUI, please use the unofficial loader node: [ComfyUI-nunchaku-unofficial-loader](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader)

**Post-quantize fidelity bench (integrated, default ON):** After save, `quantize_sdxl_hswq_v3.1.py` and `native_convert_int8_sdxl.py` **clear parent VRAM**, then automatically run `benchmark/int8bench_sdxl.py` with `--fp16` = the FP16 input, `--int8` = the saved pack, and a fixed `--prompt` / `--seed` (not inventable parent CLI overrides). Pass `--no-bench` to skip. Standalone re-runs use the same `int8bench_sdxl.py` command shape as in the How-to.

**SDXL ConvRot INT8 Benchmark Test Results (published tables):** [test/benchmark_sdxl_int8.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/test/benchmark_sdxl_int8.md)

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| Naive INT8 | 0.95-0.97 | 50% | High |
| **HSWQ ConvRot INT8** | **0.94-0.98** | 68% (FP16 mixed) | **High** (ComfyUI INT8) |

---

## 📦 Available Models

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `JANKUTrainedChenkinNoobai_v777_hswq_r32_1off_convrot_int8.safetensors` | [JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)](https://civitai.com/models/1277670/janku-trained-chenkin-and-noobai-rouwei-illustrious-xl) | v777 | Fair AI Public License 1.0-SD |
| `bluePencilXL_v031_hswq_r32_1off_convrot_int8.safetensors` | [blue_pencil-XL](https://civitai.com/models/119012) | v0.3.1 | CreativeML Open RAIL++-M |
| `epicrealismXL_pureFix_hswq_r32_1off_convrot_int8.safetensors` | [epiCRealism XL](https://civitai.com/models/277058) | pureFix | CreativeML Open RAIL++-M |
| `koronemixIllustrious_v70_sci_1on_covrot_int8.safetensors` | koronemixIllustrious | v70 | Fair AI Public License 1.0-SD |
| `novaAnimeXL_ilV190_hswq_r32_1on_convrot_int8.safetensors` | Nova Anime XL | ilV190 | Fair AI Public License 1.0-SD |
| `prefectIllustriousXL_v8_hswq_r32_1on_convrot_int8.safetensors` | Prefect Illustrious XL | v8 | Fair AI Public License 1.0-SD |
| `waiIllustriousSDXL_v170hswq_r32_1off_convrot_int8.safetensors` | [Illustrious-XL v1.7 (WAI-illustrious-SDXL)](https://civitai.com/models/827184/wai-illustrious-sdxl) | v17.0 (HF weight) | Fair AI Public License 1.0-SD |

---

## 📜 Credits & License

### 🏆 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)**: Created by [janxd](https://civitai.com/user/janxd).
- **blue_pencil-XL**: Created by [Euge_us](https://civitai.com/user/Euge_us).
- **epiCRealism XL**: Created by [epinikion](https://civitai.com/user/epinikion).
- **WAI-illustrious-SDXL**: Created by [WAI0731](https://civitai.com/user/WAI0731).
- **koronemixIllustrious**: Created by [koronen](https://civitai.com/user/koronen).
- **Nova Anime XL**: Original creator on Civitai.
- **Prefect Illustrious XL**: Created by [Goofy_Ai](https://civitai.com/user/Goofy_Ai).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
