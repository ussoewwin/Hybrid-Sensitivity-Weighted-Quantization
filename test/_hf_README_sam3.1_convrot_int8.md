---
license: other
license_name: sam-license
license_link: https://github.com/facebookresearch/segment-anything/blob/main/LICENSE
base_model:
- facebook/sam3.1
- Comfy-Org/sam3.1
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

## 📦 Model Details
| Filename | Base Architecture | Precision / Format | File Size | Size Reduction vs FP16 | Native ComfyUI Support | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sam3.1_multiplex_hswq_convrot_int8.safetensors` | SAM 3.1 Multiplex | **HSWQ ConvRot INT8** (`int8_tensorwise`) | **~1,738.2 MB** (1.74 GB) | **~0.4% Reduction** vs the ~1,745.5 MB (1.75 GB) FP16 source - 41 of the 221 runtime-live Linear layers are quantised, the rest of the checkpoint stays FP16 | ✅ Supported natively (`UNetLoader`) | Available |

Sizes are decimal MB/GB computed from the published byte counts, and the reduction is computed from those same byte counts against the source file named in the same row.

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

## 🔒 Security

**Backdoor audit.** Each file's audit status - verdict strength, weight coverage, the record behind it,
and the limits of what that measurement can show - is in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/backdoor/Backdoor_Audit_Verdicts.md). The audit code, the procedure and the raw records are in the same
repository.

## 📜 Credits & License
- **Base Architecture & Model:** [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) / [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) (Meta SAM / Segment Anything Team)
- **Quantization Framework:** [HSWQ / ConvRot INT8 Pipeline](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **License:** [SAM License (Meta SAM License)](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)
