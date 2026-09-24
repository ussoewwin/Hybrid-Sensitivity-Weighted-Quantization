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

High-fidelity **ConvRot INT8 reverse hybrid** quantization for **SDXL** diffusion models. HSWQ uses per-layer **trajectory-impact** measurement instead of naive uniform cast, converting only the **K lowest-impact layers** to ConvRot INT8 while keeping every other layer at FP16. This is highly useful for users who need to strictly manage their VRAM resources while maintaining maximum image quality.

## Method

**Reverse hybrid (diag → reverse):** The FP16 checkpoint is the only input. A per-layer trajectory-impact measurement (`sdxl/diag_impact_sdxl.py`) injects each candidate layer's ConvRot INT8 reconstruction into the FP16 model one at a time and runs the production sampler — recording the final-latent drift. The **K lowest-impact layers** are then packed as FULL ConvRot INT8 (`int8_tensorwise`) while **every other layer stays FP16** (`sdxl/gen_reverse_int8_sdxl.py`). The V3.1 selector (DualMonitor + V4 weighted-histogram MSE + full SVD, fixed **300 MiB** FP16 protection budget) provides static protection, and the reverse step provides the dynamic trajectory-based criterion for the remaining pool.

Validated by the **deterministic 25-seed latent-trajectory comparison** (per-step cosine + bifurcation detection); production gate = **cosine mean ≥ 0.95 and 0/25 bifurcated**.

The quantized file does not embed a VAE (`first_stage_model.*` is removed at conversion): load it with a separate SDXL VAE.

**Technical details:** [https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

**How to quantize (SDXL ConvRot INT8):** [md/How to quantize SDXL.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL.md)

**Diag → Reverse SDXL Technical Guide:** [md/Diag_Reverse_SDXL_v1.1_Technical_Guide.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Diag_Reverse_SDXL_v1.1_Technical_Guide.md)

**ComfyUI Loader for ConvRot INT8 / INT8:** To load these INT8 models in ComfyUI, please use the custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

**SDXL ConvRot INT8 Benchmark Test Results (published tables):** [benchmark result/benchmark_sdxl_int8.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/benchmark%20result/benchmark_sdxl_int8.md)

---

## Benchmark (Reference)

**Production gate: deterministic 25-seed latent-trajectory comparison** (`benchmark/sdxl_int8_traj_compare.py`). PASS = final-cosine mean ≥ 0.95 and 0/25 bifurcated.

| Configuration | 25-seed cosine mean | Note |
| :--- | :--- | :--- |
| Original FP16 | 1.000 | identity |
| Native ConvRot INT8 (all layers) | ≈ 0.93 | the floor the hybrid must beat |
| **HSWQ Reverse Hybrid ConvRot INT8** | **≥ 0.95 (gate)** | K lowest-impact layers INT8, rest FP16 |

**Legacy decoded-image reference (not the gate):**

| Model | SSIM (Avg) | File size | Compatibility |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| Naive INT8 | 0.95-0.97 | 50% | High |
| **HSWQ ConvRot INT8** | **0.94-0.98** | 68% (FP16 mixed) | **High** (ComfyUI INT8) |

---

## 📦 Available Models

Filename convention: `<model>_hswq_1on_re<K>_convrot_int8.safetensors` — reverse hybrid with K lowest-impact layers converted to ConvRot INT8, bias correction ON (`1on`), everything else FP16.

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `JANKUTrainedChenkinNoobai_v777_hswq_1on_re550_convrot_int8.safetensors` | [JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)](https://civitai.red/models/1277670/janku-trained-chenkin-and-noobai-rouwei-illustrious-xl) | v777 | Fair AI Public License 1.0-SD |
| `bluePencilXL_v031_hswq_1on_re570_convrot_int8.safetensors` | [blue_pencil-XL](https://civitai.red/models/119012) | v0.3.1 | CreativeML Open RAIL++-M |
| `epicrealismXL_pureFix_hswq_1on_re570_convrot_int8.safetensors` | [epiCRealism XL](https://civitai.red/models/277058) | pureFix | CreativeML Open RAIL++-M |
| `koronemixIllustrious_v70_hswq_1on_re550_convrot_int8.safetensors` | koronemixIllustrious | v70 | Fair AI Public License 1.0-SD |
| `koronemixVpred_v20_hswq_1on_re550_convrot_int8.safetensors` | koronemixVpred | v2.0 | CreativeML Open RAIL++-M |
| `novaAnimeXL_ilV190_hswq_1on_re599_convrot_int8.safetensors` | Nova Anime XL | ilV190 | Fair AI Public License 1.0-SD |
| `novaAsianXL_illustriousV70_hswq_1on_re550_convrot_int8.safetensors` | Nova Asian XL | v7.0 | Fair AI Public License 1.0-SD |
| `oneObsession_v24_hswq_1on_re572_convrot_int8.safetensors` | [OneObsession](https://civitai.red/models/691062) | v24 | CreativeML Open RAIL++-M |
| `prefectIllustriousXL_v8_hswq_1on_re610_convrot_int8.safetensors` | Prefect Illustrious XL | v8 | Fair AI Public License 1.0-SD |
| `realvisxlV30_v30TurboBakedvae_hswq_1on_re650_convrot_int8.safetensors` | [RealVisXL V3.0 (Turbo)](https://civitai.red/models/139562?modelVersionId=361593) | v3.0 Turbo | CreativeML Open RAIL++-M |
| `realvisxlV50_v40Bakedvae_hswq_1on_re550_convrot_int8.safetensors` | [RealVisXL V5.0 (Lightning)](https://civitai.red/models/139562/realvisxl-v50) | v4.0 BakedVAE | CreativeML Open RAIL++-M |
| `realvisxlV50_v50Bakedvae_hswq_1on_re550_convrot_int8.safetensors` | [RealVisXL V5.0 (Lightning)](https://civitai.red/models/139562/realvisxl-v50) | v5.0 BakedVAE | CreativeML Open RAIL++-M |
| `uwazumimixILL_v50_hswq_1on_re720_convrot_int8.safetensors` | UwazumiMix | v5.0 | Fair AI Public License 1.0-SD |
| `waiIllustriousSDXL_v170_hswq_1on_re597_convrot_int8.safetensors` | [Illustrious-XL v1.7 (WAI-illustrious-SDXL)](https://civitai.red/models/827184/wai-illustrious-sdxl) | v17.0 | Fair AI Public License 1.0-SD |
| `waiREALCN_v150_hswq_1on_re630_convrot_int8.safetensors` | WAI-REAL_CN | v15.0 | Fair AI Public License 1.0-SD |
| `waiREALISM_v10_hswq_r32_1on_convrot_int8.safetensors` | WAI-REALISM | v1.0 | Fair AI Public License 1.0-SD |

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
- **WAI-illustrious-SDXL / WAI-REAL_CN / WAI-REALISM**: Created by [WAI0731](https://civitai.red/user/WAI0731).
- **koronemixIllustrious / koronemixVpred**: Created by [koronen](https://civitai.red/user/koronen).
- **Nova Anime XL / Nova Asian XL**: Original creator on Civitai.
- **Prefect Illustrious XL**: Created by [Goofy_Ai](https://civitai.red/user/Goofy_Ai).
- **OneObsession**: Created by [Polyhedron](https://civitai.red/user/Polyhedron).
- **RealVisXL**: Created by [SG_161222](https://civitai.red/user/SG_161222).
- **UwazumiMix**: Created by [UWAZUMI](https://civitai.red/user/UWAZUMI).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
