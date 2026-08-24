---
license: other
tags:
- text-to-image
- krea2
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

High-fidelity Hybrid ConvRot NVFP4 quantization for **Krea2** architecture models. Built from a complete **native ConvRot INT8** UNet via the **reverse method** (converting lowest-impact layers to NVFP4 in ascending order of trajectory impact). This is highly useful for users who need to strictly manage their VRAM resources while maintaining high image fidelity and avoiding catastrophic trajectory bifurcations.

**Technical details:** [https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

**ComfyUI Loader for ConvRot NVFP4:** To use these models in ComfyUI, please use this custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

**Krea2 ConvRot NVFP4 Benchmark Test Results (published tables):** [benchmark result/benchmark_krea2_nvfp4.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/benchmark%20result/benchmark_krea2_nvfp4.md)

---

## Benchmark (Reference)

| Model | Final Cosine (Avg) | Trajectory Stability | VRAM |
| :--- | :--- | :--- | :--- |
| Original FP16 | 1.0000 | 100% | High |
| **HSWQ Krea2 Hybrid ConvRot NVFP4** | **~0.916 - 0.925** | **4.3x less bifurcation** | **~50% Savings** |

---

## 🟢 Available Models

| Filename | Base Model | Version | License |
| :--- | :--- | :--- | :--- |
| moodyKrea2Mix_v70_hswq_nv103_hybrid_convrot_nvfp4.safetensors | Moody Krea2 Mix | v7.0 (nv103) | CreativeML Open RAIL++-M |
| moodyCutieMixKrea2_v30_hswq_nv108_hybrid_convrot_nvfp4.safetensors | Moody Cutie Mix Krea2 | v3.0 (nv108) | CreativeML Open RAIL++-M |

---

## 🤝 Credits & License

### 💡 Special Acknowledgement
We extend our deepest respect and gratitude to the **Nunchaku Team** for their groundbreaking work on SVDQ quantization and for sharing their models with the community. This collection relies heavily on their research and original implementation.
- **Original Repository:** [nunchaku-tech/nunchaku-sdxl](https://huggingface.co/nunchaku-tech/nunchaku-sdxl)

### Base Models
These models are derivatives of their respective creators. All credit for aesthetic tuning and model training belongs to the original creators.
- **Moody Krea2 Mix / Moody Cutie Mix Krea2**: Created by [catlover1937](https://civitai.com/user/catlover1937).

---

**Disclaimer:** These models are provided for optimization and research purposes. Please adhere to the original licenses of the base models.
