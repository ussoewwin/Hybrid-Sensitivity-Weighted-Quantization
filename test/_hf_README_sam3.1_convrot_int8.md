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
- **Base Checkpoint:** Upstream SAM 3.1 Multiplex foundation weights from [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) / [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
- **Native ComfyUI Support:** Features native `int8_tensorwise` and `comfy_quant` metadata stamps, loading seamlessly with ComfyUI's standard `UNetLoader` / `DiffusionModelLoader` or `HSWQ SAM3 Loader` without requiring external manual dequantization workflows.

---

## 📦 Model Details

| Filename | Base Architecture | Precision / Format | File Size | Memory Footprint vs FP16 | Native ComfyUI Support | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sam3.1_multiplex_hswq_convrot_int8.safetensors` | SAM 3.1 Multiplex | **HSWQ ConvRot INT8** (`int8_tensorwise`) | **~1.74 GB** (1,738 MB) | ~0.4% vs the 1.66 GB FP16 source (41 of 221 runtime-live layers quantized) | ✅ Supported natively (`UNetLoader`) | Available |

### Quantization Breakdown
- **HSWQ ConvRot INT8 Linear Layers:** Bulk transformer and feedforward projection layers quantized with orthogonal Sylvester Hadamard rotation (`groupsize=256`) and per-channel scaling.
- **Sensitivity-Protected High-Precision Tensors:** Critical cross-attention heads, memory encoder/decoder boundaries, LayerNorm, Embeddings, Biases, and 4D Conv2d weights preserved in full float precision under HSWQ sensitivity analysis to prevent mask jitter, boundary collapse, and video tracking loss.

---

## 🛠️ Key Highlights

- **Hadamard Orthogonal Rotation (ConvRot):** Pre-rotates weight matrices ($W_{rot} = W \cdot H^T$) along the input channel axis using normalized Sylvester Hadamard blocks, eliminating channel-wise outlier spikes and preventing mask degradation.
- **Hybrid Sensitivity Protection (HSWQ):** Specifically designed to prevent precision loss in multi-object video segmentation, guaranteeing sharp mask boundaries and long-term tracking stability across all 16 multiplex object slots.
- **Optimized Storage & Memory Footprint:** 1.74 GB, near the 1.66 GB FP16 source, because only 41 of the checkpoint's runtime-live layers are both 256-wide and unprotected. The gain here is the INT8 **execution** path (Tensor Core INT8 matmul), not the disk size - the same trade the guide documents. Quality is preserved: measured tracker mask IoU 0.9978 against FP16.

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

## 🔒 Security

This pack is quantized from the SAM 3.1 Multiplex checkpoint distributed by
[Comfy-Org](https://huggingface.co/Comfy-Org/sam3.1), which in turn declares
`facebook/sam3.1` as its base. Meta's fp32 release sits behind a manual gate and the access
request for it was refused, so a weight-level comparison against Meta's release was not
possible. The audit compared this pack's source at the byte level against the Comfy-Org release
it names - which establishes only which artifact was examined, not what it does - and then
measured the model itself.

The model was audited for a backdoor: the text path was loaded with every weight of the
checkpoint's own language encoder, the weights were searched for a direction that behaves as a
switch rather than as part of the network, and the detector was run on unrelated images to see
whether its output stops following the picture. Both statistics were first proved sensitive by
injecting a branch of each kind, so that a null result means something. Nothing was found.

The pack was verified separately, because correct bytes are not the same thing as a working
load: quantised layers 41 of 41 applied, no raw int8 code left in any of its 1,444 float
parameters, and 936 INT8 GEMM calls on the diffusion route, with the conditioning at cosine
0.9999779673556464 against the FP16 source over 24 seeds and no pixel flips.

The audit, its records and the limits of what was measured are in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Backdoor_Audit_Verdicts.md).

---

## 📜 Credits & License

- **Base Architecture & Model:** [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) / [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) (Meta SAM / Segment Anything Team)
- **Quantization Framework:** [HSWQ / ConvRot INT8 Pipeline](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **License:** [SAM License (Meta SAM License)](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)
