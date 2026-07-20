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

High-fidelity ConvRot INT8 quantization for diffusion models (SDXL). HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast.

**Technical details:** [md/HSWQ_INT8_SDXL_Technical_Guide.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_INT8_SDXL_Technical_Guide.md)

**How to quantize:** [md/HSWQ_ How to quantize SDXL.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL.md)

**ComfyUI Loader for INT8:** To load these INT8 models in ComfyUI, please use the unofficial loader node: [ComfyUI-nunchaku-unofficial-loader](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader)

**SDXL Benchmark Test Results:** [test/benchmark_sdxl_int8.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/test/benchmark_sdxl_int8.md)

---

## Benchmark (Reference)

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| Naive INT8 | 0.95-0.97 | 50% | High |
| **HSWQ INT8** | **0.96-0.98** | 67% (FP16 mixed) | **High** |

---

## 📦 Available Models

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `JANKUTrainedChenkinNoobai_v777_hswq_r32_convrot_int8.safetensors` | [JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)](https://civitai.red/models/1277670/janku-trained-chenkin-and-noobai-rouwei-illustrious-xl) | v777 | Illustrious License |
| `bluePencilXL_v031_hswq_r32_convrot_int8.safetensors` | [blue_pencil-XL](https://civitai.red/models/119012) | v0.3.1 | CreativeML Open RAIL++-M |
| `epicrealismXL_pureFix_hswq_r32_convrot_int8.safetensors` | [epiCRealism XL](https://civitai.red/models/277058) | pureFix | CreativeML Open RAIL++-M |
| `oneObsession_v23_hswq_r32_convrot_int8.safetensors` | [OneObsession](https://civitai.red/models/691062) | v23 | CreativeML Open RAIL++-M |
| `waiIllustriousSDXL_v170_hswq_r32_convrot_int8.safetensors` | [Illustrious-XL v1.7 (WAI-illustrious-SDXL)](https://civitai.red/models/827184/wai-illustrious-sdxl) | v17.0 (HF weight) | Illustrious License |

---

## 📜 Credits & License

### 🏆 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)**: Created by [janxd](https://civitai.red/user/janxd).
- **blue_pencil-XL**: Created by [Euge_us](https://civitai.red/user/Euge_us).
- **epiCRealism XL**: Created by [epinikion](https://civitai.red/user/epinikion).
- **OneObsession**: Created by [Polyhedron](https://civitai.red/user/Polyhedron).
- **WAI-illustrious-SDXL**: Created by [WAI0731](https://civitai.red/user/WAI0731).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
