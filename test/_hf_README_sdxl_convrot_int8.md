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

## 📦 Available Models
Filename convention: `<model>_hswq_1on_re<K>_convrot_int8.safetensors` — reverse hybrid with K lowest-impact layers converted to ConvRot INT8, bias correction ON (`1on`), everything else FP16.

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| `JANKUTrainedChenkinNoobai_v777_hswq_1on_re550_convrot_int8.safetensors` | [JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)](https://civitai.red/models/1277670/janku-trained-chenkin-and-noobai-rouwei-illustrious-xl) | v777 | Fair AI Public License 1.0-SD |
| `bluePencilXL_v031_hswq_1on_re570_convrot_int8.safetensors` | [blue_pencil-XL](https://civitai.red/models/119012) | v0.3.1 | CreativeML Open RAIL++-M |
| `epicrealismXL_pureFix_hswq_1on_re570_convrot_int8.safetensors` | [epiCRealism XL](https://civitai.red/models/277058) | pureFix | CreativeML Open RAIL++-M |
| `koronemixIllustrious_v70_hswq_1on_re550_convrot_int8.safetensors` | [koronemixIllustrious](https://civitai.red/user/koronen) | v70 | Fair AI Public License 1.0-SD |
| `koronemixVpred_v20_hswq_1on_re550_convrot_int8.safetensors` | [koronemixVpred](https://civitai.red/user/koronen) | v2.0 | CreativeML Open RAIL++-M |
| `novaAnimeXL_ilV190_hswq_1on_re599_convrot_int8.safetensors` | [Nova Anime XL](https://civitai.red/user/Crody) | ilV190 | Fair AI Public License 1.0-SD |
| `novaAsianXL_illustriousV70_hswq_1on_re550_convrot_int8.safetensors` | [Nova Asian XL](https://civitai.red/user/Crody) | v7.0 | Fair AI Public License 1.0-SD |
| `oneObsession_v24_hswq_1on_re572_convrot_int8.safetensors` | [OneObsession](https://civitai.red/models/691062) | v24 | CreativeML Open RAIL++-M |
| `prefectIllustriousXL_v8_hswq_1on_re610_convrot_int8.safetensors` | [Prefect Illustrious XL](https://civitai.red/user/Goofy_Ai) | v8 | Fair AI Public License 1.0-SD |
| `realvisxlV30_v30TurboBakedvae_hswq_1on_re650_convrot_int8.safetensors` | [RealVisXL V3.0 (Turbo)](https://civitai.red/models/139562?modelVersionId=361593) | v3.0 Turbo | CreativeML Open RAIL++-M |
| `realvisxlV50_v40Bakedvae_hswq_1on_re550_convrot_int8.safetensors` | [RealVisXL V5.0 (Lightning)](https://civitai.red/models/139562/realvisxl-v50) | v4.0 BakedVAE | CreativeML Open RAIL++-M |
| `realvisxlV50_v50Bakedvae_hswq_1on_re550_convrot_int8.safetensors` | [RealVisXL V5.0 (Lightning)](https://civitai.red/models/139562/realvisxl-v50) | v5.0 BakedVAE | CreativeML Open RAIL++-M |
| `unholyDesireMixSinister_v90_hswq_1on_re590_convrot_int8.safetensors` | [Unholy Desire Mix - Sinister](https://civitai.red/models/1307857) | v9.0 | Fair AI Public License 1.0-SD |
| `uwazumimixILL_v50_hswq_1on_re720_convrot_int8.safetensors` | [UwazumiMix](https://civitai.red/user/UWAZUMI) | v5.0 | Fair AI Public License 1.0-SD |
| `waiANIPONYXL_v140_hswq_1on_re650_convrot_int8.safetensors` | [WAI-ANI-PONY-XL](https://civitai.red/user/WAI0731) | v14.0 | Fair AI Public License 1.0-SD |
| `waiANIPONYXL_v90_hswq_1on_re650_convrot_int8.safetensors` | [WAI-ANI-PONY-XL](https://civitai.red/user/WAI0731) | v9.0 | Fair AI Public License 1.0-SD |
| `waiIllustriousSDXL_v170_hswq_1on_re597_convrot_int8.safetensors` | [Illustrious-XL v1.7 (WAI-illustrious-SDXL)](https://civitai.red/models/827184/wai-illustrious-sdxl) | v17.0 | Fair AI Public License 1.0-SD |
| `waiREALCN_v150_hswq_1on_re630_convrot_int8.safetensors` | [WAI-REAL_CN](https://civitai.red/user/WAI0731) | v15.0 | Fair AI Public License 1.0-SD |
| `waiREALISM_v10_hswq_1on_re590_convrot_int8.safetensors` | [WAI-REALISM](https://civitai.red/user/WAI0731) | v1.0 | Fair AI Public License 1.0-SD |

## 📜 Credits & License

### 🏆 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **JANKU Trained Chenkin & Noobai-Rouwei (Illustrious-XL)**: Created by [janxd](https://civitai.red/user/janxd).
- **blue_pencil-XL**: Created by [Euge_us](https://civitai.red/user/Euge_us).
- **epiCRealism XL**: Created by [epinikion](https://civitai.red/user/epinikion).
- **WAI-illustrious-SDXL / WAI-REAL_CN / WAI-REALISM / WAI-ANI-PONY-XL**: Created by [WAI0731](https://civitai.red/user/WAI0731).
- **koronemixIllustrious / koronemixVpred**: Created by [koronen](https://civitai.red/user/koronen).
- **Nova Anime XL / Nova Asian XL**: Created by [Crody](https://civitai.red/user/Crody).
- **Prefect Illustrious XL**: Created by [Goofy_Ai](https://civitai.red/user/Goofy_Ai).
- **OneObsession**: Created by [Polyhedron](https://civitai.red/user/Polyhedron).
- **RealVisXL**: Created by [SG_161222](https://civitai.red/user/SG_161222).
- **Unholy Desire Mix - Sinister**: Created by [UnholyDesiresStudio](https://civitai.red/user/UnholyDesiresStudio).
- **UwazumiMix**: Created by [UWAZUMI](https://civitai.red/user/UWAZUMI).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.

## 📚 Detailed documentation

- [md/How to quantize SDXL.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL.md)
- [md/Diag_Reverse_SDXL_v1.1_Technical_Guide.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Diag_Reverse_SDXL_v1.1_Technical_Guide.md)
- [benchmark result/benchmark_sdxl_int8.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/benchmark%20result/benchmark_sdxl_int8.md)

## 🔒 Security

**Backdoor audit.** Each file's audit status - verdict strength, weight coverage, the record behind it,
and the limits of what that measurement can show - is in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/backdoor/Backdoor_Audit_Verdicts.md). The audit code, the procedure and the raw records are in the same
repository.
