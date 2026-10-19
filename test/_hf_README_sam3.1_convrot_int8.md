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
- hswq
- comfyui
pipeline_tag: image-segmentation
---

# SAM 3.1 Multiplex (HSWQ ConvRot INT8)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity **HSWQ ConvRot INT8** (Hybrid-Sensitivity-Weighted-Quantization) quantized weights for the **SAM 3.1 Multiplex** (Segment Anything Model 3.1) architecture.

- **Technical Details & Quantization Framework:** [Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **ComfyUI Loader for ConvRot INT8:** [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

---

## 🌟 Model Overview

This repository provides production-grade **HSWQ ConvRot INT8** quantized weights for the **SAM 3.1 Multiplex** foundation model. By combining orthogonal Hadamard rotation with HSWQ sensitivity profiling, activation outlier spikes are suppressed across projection layers while critical memory attention, boundary refinement, and mask decoder modules are protected in higher precision (FP16/BF16):

- **Architecture:** SAM 3.1 Multiplex (Vision Backbone + FPN Neck + Decoupled Memory Attention + 16-Object Multiplex Mask Decoder)
- **Base Checkpoint:** [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) / [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) (`sam3.1_multiplex_fp16.safetensors`)
- **Native ComfyUI Support:** Features native `int8_tensorwise` and `comfy_quant` metadata stamps, loading seamlessly with ComfyUI's standard `UNetLoader` / `DiffusionModelLoader` or `HSWQ SAM3 Loader` without requiring external manual dequantization workflows.

---

## 📦 Model Details & Comparison

| Filename | Base Architecture | Precision / Format | File Size | Memory Footprint | Native ComfyUI Support | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sam3.1_multiplex_hswq_convrot_int8.safetensors` | SAM 3.1 Multiplex | **HSWQ ConvRot INT8** (`int8_tensorwise`) | **~1.29 GB** (1,287 MB) | **~26% VRAM Reduction** | ✅ Supported natively (`UNetLoader`) | Available |
| `sam3.1_multiplex_fp16.safetensors` (Reference) | SAM 3.1 Multiplex | FP16 | ~1.63 GB (1,746 MB) | Baseline | ✅ Supported natively | Reference |

### Quantization Breakdown
- **HSWQ ConvRot INT8 Linear Layers:** Bulk transformer and feedforward projection layers quantized with orthogonal Sylvester Hadamard rotation (`groupsize=256`) and per-channel scaling.
- **Sensitivity-Protected High-Precision Tensors:** Critical cross-attention heads, memory encoder/decoder boundaries, LayerNorm, Embeddings, Biases, and 4D Conv2d weights preserved in full float precision under HSWQ sensitivity analysis to prevent mask jitter, boundary collapse, and video tracking loss.

---

## 🛠️ Key Highlights

- **Hadamard Orthogonal Rotation (ConvRot):** Pre-rotates weight matrices ($W_{rot} = W \cdot H^T$) along the input channel axis using normalized Sylvester Hadamard blocks, eliminating channel-wise outlier spikes and preventing mask degradation.
- **Hybrid Sensitivity Protection (HSWQ):** Specifically designed to prevent precision loss in multi-object video segmentation, guaranteeing sharp mask boundaries and long-term tracking stability across all 16 multiplex object slots.
- **Optimized Storage & Memory Footprint:** Reduces model weight footprint from 1.63 GB down to 1.29 GB while maintaining near-lossless mIoU segmentation metrics.

---

## 🚀 Usage in ComfyUI

### Model Placement
Download `sam3.1_multiplex_hswq_convrot_int8.safetensors` and place it in your ComfyUI models directory:
```
ComfyUI/models/unet/sam3.1_multiplex_hswq_convrot_int8.safetensors
```

### Loading
1. Load the model using the standard **`UNetLoader`** (or **`Load Diffusion Model`**) node in ComfyUI, or use **`HSWQ SAM3 Loader`** from [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools).
2. Connect the loaded model output directly into your SAM3 image / video segmentation workflow nodes.
3. ComfyUI automatically detects the `comfy_quant` header metadata and executes optimized INT8 Tensor Core operations natively.

---

## 📜 Credits & License

- **Base Architecture & Model:** [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) / [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) (Meta SAM / Segment Anything Team)
- **Quantization Framework:** [HSWQ / ConvRot INT8 Pipeline](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **License:** [SAM License (Meta SAM License)](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)
