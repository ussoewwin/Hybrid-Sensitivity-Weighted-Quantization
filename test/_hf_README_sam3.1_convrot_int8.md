---
license: other
license_name: sam-license
license_link: https://github.com/facebookresearch/segment-anything/blob/main/LICENSE
base_model:
- facebook/sam3.1
tags:
- sam
- sam3
- sam3.1
- segment-anything
- segmentation
- video-segmentation
- multiplex
- quantized
- int8
- convrot
- comfyui
pipeline_tag: image-segmentation
---

# SAM 3.1 Multiplex (ConvRot INT8)

High-fidelity **Native ConvRot INT8** quantized weights for the **SAM 3.1 Multiplex** (Segment Anything Model 3.1) architecture.

---

## 🌟 Model Overview

This repository provides **Native ConvRot INT8** quantized weights for the **SAM 3.1 Multiplex** foundation model. By applying orthogonal Hadamard rotation before per-channel INT8 quantization, activation outliers are evenly distributed across matrix dimensions, drastically suppressing quantization noise while halving VRAM and storage footprint:

- **Architecture:** SAM 3.1 Multiplex (Vision Backbone + FPN Neck + Decoupled Memory Attention + 16-Object Multiplex Mask Decoder)
- **Base Checkpoint:** [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) / [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) (`sam3.1_multiplex_fp16.safetensors`)
- **Native ComfyUI Support:** Features native `int8_tensorwise` and `comfy_quant` metadata stamps, loading seamlessly with ComfyUI's standard `UNetLoader` / `DiffusionModelLoader` without requiring external custom quantization nodes.

---

## 📦 Model Details & Comparison

| Filename | Base Architecture | Precision / Format | File Size | Memory Footprint | Native ComfyUI Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sam3.1_multiplex_convrot_int8.safetensors` | SAM 3.1 Multiplex | **ConvRot INT8** (`int8_tensorwise`) | **~0.84 GB** (899 MB) | **~50% VRAM Reduction** | ✅ Supported natively (`UNetLoader`) |
| `sam3.1_multiplex_fp16.safetensors` (Reference) | SAM 3.1 Multiplex | FP16 | ~1.63 GB (1,746 MB) | Baseline | ✅ Supported natively |

### Quantization Breakdown
- **ConvRot INT8 Layers:** 647 Linear projection layers (Hadamard rotation `groupsize=256` + per-out-channel scale)
- **Plain INT8 Fallback Layers:** 4 layers (non-power-of-4 channel dimensions safely quantized with row-wise scaling)
- **Preserved High-Precision Tensors:** 1,183 tensors (LayerNorm, Embedding, Biases, and 4D Conv2d weights preserved in full float precision for boundary and spatial accuracy)

---

## 🛠️ Key Highlights

- **Hadamard Orthogonal Rotation (ConvRot):** Pre-rotates weight matrices ($W_{rot} = W \cdot H^T$) along the input channel axis using normalized Sylvester Hadamard blocks, eliminating channel-wise outlier spikes and preventing mask degradation.
- **Full Multiplex Support:** Preserves the complete 16-object simultaneous tracking, interactive segmentation, and decoupled cross-attention capabilities of the SAM 3.1 architecture.
- **50% Disk & VRAM Savings:** Reduces weight storage from 1.63 GB down to 0.84 GB, enabling lightweight deployment on consumer hardware and multi-model video segmentation workflows.

---

## 🚀 Usage in ComfyUI

### Model Placement
Download `sam3.1_multiplex_convrot_int8.safetensors` and place it in your ComfyUI models directory:
```
ComfyUI/models/unet/sam3.1_multiplex_convrot_int8.safetensors
```

### Loading
1. Load the model using the standard **`UNetLoader`** (or **`Load Diffusion Model`**) node in ComfyUI.
2. Connect the loaded model output directly into your SAM3 image / video segmentation workflow nodes.
3. ComfyUI automatically detects the `comfy_quant` header metadata and executes optimized INT8 Tensor Core operations natively.

---

## 📜 Credits & License

- **Base Architecture & Model:** [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) / [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) (Meta SAM / Segment Anything Team)
- **Quantization Framework:** [HSWQ / ConvRot INT8 Pipeline](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **License:** [SAM License (Meta SAM License)](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)
