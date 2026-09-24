---
license: other
tags:
- clip
- text-encoder
- t5
- openclip
- text-to-image
- image-to-image
- quantized
- int8
- convrot
- comfyui
pipeline_tag: feature-extraction
---

# Text Encoder & CLIP Models (ConvRot INT8)

High-fidelity **Native ConvRot INT8** quantized weights for primary Text Encoders and Vision-Language CLIP models across state-of-the-art generative diffusion architectures (**SDXL**, **FLUX.1**, **SD 1.5 / SD 2.1**, **SD3 / SD3.5**, and **Next-Gen Diffusion** pipelines).

---

## 🌟 Model Overview

This repository hosts high-quality **Native ConvRot INT8** quantized weights for essential text encoders and CLIP vision-language backbones. By applying orthogonal Hadamard rotation ($W_{rot} = W \cdot H^T$) prior to per-channel symmetric INT8 quantization, activation outlier spikes in transformer feed-forward and projection layers are redistributed uniformly across channels. This eliminates quantization-induced semantic drift and cuts VRAM and storage footprint by ~50% while preserving precise prompt conditioning fidelity:

- **CLIP-SAE-ViT-L-14-FP32**: Primary ViT-L/14 text encoder with high-precision FP32/SAE baseline representation, quantized to native ConvRot INT8 for ultra-fast, lightweight prompt encoding across SD 1.5, SDXL, and FLUX workflows.
- **CLIP-ViT-bigG-14-laion2B-39B-b160k**: The massive OpenCLIP ViT-bigG/14 text encoder (laion2B) for SDXL and multi-encoder generative pipelines, reducing disk and VRAM demands from 3.69 GB down to 1.85 GB.
- **flan_t5_xxl**: The 11B-parameter dense Flan-T5 XXL text encoder for next-generation text-to-image and multimodal architectures, compressed from ~22.4 GB down to ~11.28 GB to allow inference on consumer GPUs.

---

## 📦 Available Models

| Filename | Base Architecture | Base Model / Upstream | Quantization | File Size | Baseline Size | VRAM Savings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CLIP-SAE-ViT-L-14-FP32_native_convrot_int8.safetensors` | OpenAI CLIP ViT-L/14 | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | ConvRot INT8 (`int8_tensorwise`) | **~0.43 GB** (412 MB) | ~0.93 GB | **~54% Reduction** |
| `CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` | OpenCLIP ViT-bigG/14 | [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) | ConvRot INT8 (`int8_tensorwise`) | **~1.85 GB** (1,765 MB) | ~3.69 GB | **~50% Reduction** |
| `flan_t5_xxl_convrot_int8.safetensors` | Google Flan-T5 XXL (11B) | [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) | ConvRot INT8 (`int8_tensorwise`) | **~11.28 GB** (10.50 GB) | ~22.40 GB | **~50% Reduction** |

---

## 🛠️ Key Features

- **Orthogonal Hadamard Pre-Rotation (ConvRot):** Weight matrices are pre-rotated group-wise along the input feature dimension ($W_{rot} = W \cdot H^T$) using power-of-4 normalized Hadamard blocks (`groupsize=256`), eliminating catastrophic dynamic range compression caused by outlier channels.
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

### Loading
Load directly via standard ComfyUI CLIP loader nodes:
- **Single CLIP:** `Load CLIP` / `CLIPLoader` $\rightarrow$ select `<model_name>_convrot_int8.safetensors`.
- **Dual CLIP (SDXL):** `DualCLIPLoader` $\rightarrow$ set `clip_name1` to `clip_l` (or `CLIP-SAE-ViT-L-14-FP32_native_convrot_int8.safetensors`) and `clip_name2` to `CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` (type `sdxl`).
- **Triple CLIP (SD3 / FLUX):** `TripleCLIPLoader` $\rightarrow$ assign CLIP-L, CLIP-G, and T5-XXL ConvRot INT8 weights accordingly.

ComfyUI automatically recognizes the `comfy_quant` stamp and dispatches optimized INT8 execution kernels seamlessly.

---

## 📜 Credits & License

- **OpenAI CLIP ViT-L/14:** [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) (MIT License)
- **OpenCLIP ViT-bigG/14:** [LAION / OpenCLIP](https://github.com/mlfoundations/open_clip) (MIT / Apache-2.0 License)
- **Google Flan-T5 XXL:** [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) (Apache-2.0 License)
- **Quantization Pipeline:** [HSWQ / ConvRot INT8 Suite](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **Base Licenses:** Please adhere to the individual upstream licenses of the respective foundation models.
