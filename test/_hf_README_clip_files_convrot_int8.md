---
license: other
tags:
- clip
- text-encoder
- t5
- qwen3
- openclip
- text-to-image
- image-to-image
- quantized
- int8
- convrot
- hswq
- comfyui
pipeline_tag: feature-extraction
---

# Text Encoder & CLIP Models (HSWQ ConvRot INT8)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity **HSWQ ConvRot INT8** (Hybrid-Sensitivity-Weighted-Quantization) quantized weights for primary Text Encoders, CLIP, and compact language models across generative diffusion architectures (**SD 1.5**, **SDXL**, **FLUX.1**, **SD3 / SD3.5**, and **Next-Gen Diffusion** pipelines).

- **Technical Details & Quantization Framework:** [Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **ComfyUI Loader for ConvRot INT8:** [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

---

## 🌟 Model Overview

This repository hosts production-ready **HSWQ ConvRot INT8** quantized weights for essential text encoders and conditioning backbones. By combining orthogonal Hadamard rotation ($W_{rot} = W \cdot H^T$) with layer-wise sensitivity protection, activation outlier spikes are suppressed while critical attention and normalization boundaries remain protected:

- **`clip_l_hswq_convrot_int8.safetensors`**: Standalone OpenAI CLIP ViT-L/14 text encoder quantized via HSWQ ConvRot INT8 (~196.8 MB), offering massive memory savings while maintaining 100% prompt parsing integrity across SD 1.5, SDXL, and FLUX workflows.
- **`CLIP-SAE-ViT-L-14-hswq_convrot_int8_protected.safetensors`**: High-fidelity ViT-L/14 text encoder with Sparse Autoencoder (SAE) feature representation and HSWQ sensitivity protection (~1.03 GB), preserving sensitive alignment layers for intricate semantic steering.
- **`flan_t5_xxl_hswq_convrot_int8.safetensors`**: The 11B-parameter dense Google Flan-T5 XXL text encoder quantized with HSWQ ConvRot INT8 (~11.70 GB vs ~22.4 GB FP16 baseline), reducing VRAM footprint by ~50% to enable high-capacity prompt conditioning on consumer GPUs.
- **`Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors`**: Compact 0.6B abliterated Qwen3 language model quantized with HSWQ ConvRot INT8 (~928.9 MB), providing fast, lightweight prompt expansion and text conditioning with uncensored safety vectors.

---

## 📦 Available Models

| Filename | Architecture | Upstream Base Model | Quantization | File Size | Baseline Size | VRAM Savings | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clip_l_hswq_convrot_int8.safetensors` | OpenAI CLIP ViT-L/14 | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | HSWQ ConvRot INT8 | **~196.8 MB** | ~492 MB (FP16) | **~60% Reduction** | Available |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8_protected.safetensors` | CLIP ViT-L/14 (SAE Protected) | OpenAI CLIP ViT-L/14 (SAE) | HSWQ ConvRot INT8 (Protected) | **~1.03 GB** | ~1.90 GB (FP32/FP16) | **~46% Reduction** | Available |
| `flan_t5_xxl_hswq_convrot_int8.safetensors` | Google Flan-T5 XXL (11B) | [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) | HSWQ ConvRot INT8 | **~11.70 GB** | ~22.40 GB (FP16) | **~48% Reduction** | Available |
| `Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors` | Qwen3 0.6B Abliterated | Josiefied-Qwen3-0.6B-abliterated-v1 | HSWQ ConvRot INT8 | **~928.9 MB** | ~1.30 GB (FP16) | **~28% Reduction** | Available |

---

## 🛠️ Key Features

- **Orthogonal Hadamard Pre-Rotation (ConvRot):** Weight matrices are pre-rotated group-wise along the input feature dimension ($W_{rot} = W \cdot H^T$) using power-of-4 normalized Hadamard blocks (`groupsize=256`), eliminating catastrophic dynamic range compression caused by outlier channels.
- **Hybrid Sensitivity Weighting (HSWQ):** Automatically measures layer sensitivity to prevent semantic collapse on critical projection and attention layers, maintaining high prompt coherence.
- **Native ComfyUI Compatibility:** Encodes standard `comfy_quant` header metadata (`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`) and per-channel scale tensors, loading automatically in ComfyUI with standard `CLIPLoader`, `DualCLIPLoader`, or `TripleCLIPLoader` without custom nodes.
- **Full Precision Preservation for Critical Weights:** Non-2D weights (token embeddings, position embeddings, layer normalizations, biases) are maintained in native floating-point precision to guarantee prompt parsing integrity and boundary consistency.
- **Hardware Acceleration:** Fully compatible with Tensor Core INT8 matrix multiplication across modern GPUs (NVIDIA Turing, Ampere, Ada Lovelace, Blackwell).

---

## 🚀 Usage in ComfyUI

### Model Placement
Download the desired `.safetensors` files and place them into your ComfyUI text encoder directory:
```
ComfyUI/models/clip/
```
*(or `ComfyUI/models/LLM/` for Qwen3 language models)*

### Loading
Load directly via standard ComfyUI CLIP loader nodes:
- **Single CLIP:** `Load CLIP` / `CLIPLoader` $\rightarrow$ select `clip_l_hswq_convrot_int8.safetensors` or `CLIP-SAE-ViT-L-14-hswq_convrot_int8_protected.safetensors`.
- **Dual CLIP (SDXL):** `DualCLIPLoader` $\rightarrow$ set `clip_name1` to `clip_l_hswq_convrot_int8.safetensors` (type `sdxl`).
- **Triple CLIP (SD3 / FLUX.1):** `TripleCLIPLoader` $\rightarrow$ assign CLIP-L and `flan_t5_xxl_hswq_convrot_int8.safetensors` accordingly.
- **HSWQ Custom Node:** Use [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools) for dedicated HSWQ graph execution and precision management.

ComfyUI automatically recognizes the `comfy_quant` stamp and dispatches optimized INT8 execution kernels seamlessly.

---

## 📜 Credits & License

- **OpenAI CLIP ViT-L/14:** [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) (MIT License)
- **Google Flan-T5 XXL:** [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) (Apache-2.0 License)
- **Josiefied Qwen3 0.6B Abliterated:** Qwen Team / Community Abliterated Releases (Apache-2.0 License)
- **Quantization Pipeline:** [HSWQ / ConvRot INT8 Suite](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **Base Licenses:** Please adhere to the individual upstream licenses of the respective foundation models.
