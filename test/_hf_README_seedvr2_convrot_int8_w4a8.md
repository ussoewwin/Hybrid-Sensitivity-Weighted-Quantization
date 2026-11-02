---
license: apache-2.0
tags:
- seedvr2
- dit
- video-upscaler
- comfyui
- tensorrt
- quantization
- w4a8
- int4
- int8
- convrot
- asym_w4a8_int8
base_model: ByteDance-Seed/SeedVR2
pipeline_tag: video-to-video
---

# SeedVR2 DiT — ConvRot INT8 & W4A8 Quantization (`asym_w4a8_int8` & `int8_tensorwise`)

This repository provides the official **ConvRot INT8** and **ConvRot W4A8 (`asym_w4a8_int8`)** quantized diffusion transformer (DiT) weights for [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) / [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT).

It brings the original **15.35 GB** FP16 7B DiT down to **4.44 GB** (~71.1% memory & disk reduction) for W4A8, and **8.53 GiB** VRAM footprint for INT8, enabling native low-bit inference directly inside ComfyUI with comprehensive VRAM savings.

## Available Models & Checkpoint Specifications
| Model Checkpoint | Architecture | Precision / Layout | Disk Size | Peak VRAM | Hugging Face Direct URL |
|:---|:---|:---|:---:|:---:|:---|
| **`seedvr2_7b_convrot_w4a8.safetensors`** | 7B DiT Base | `asym_w4a8_int8` + ConvRot | **4.44 GB** | **5.69 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_7b_convrot_w4a8.safetensors) |
| **`seedvr2_7b_sharp_convrot_w4a8.safetensors`** | 7B DiT Sharp | `asym_w4a8_int8` + ConvRot | **4.44 GB** | **5.69 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_7b_sharp_convrot_w4a8.safetensors) |
| **`seedvr2_3b_convrot_w4a8.safetensors`** | 3B DiT Base | `asym_w4a8_int8` + ConvRot | **2.21 GB** | **5.69 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_3b_convrot_w4a8.safetensors) |
| **`seedvr2_7b_convrot_int8.safetensors`** | 7B DiT Base | `int8_tensorwise` + ConvRot | **8.12 GB** | **8.53 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_7b_convrot_int8.safetensors) |
| **`seedvr2_7b_sharp_convrot_int8.safetensors`** | 7B DiT Sharp | `int8_tensorwise` + ConvRot | **8.12 GB** | **8.53 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_7b_sharp_convrot_int8.safetensors) |
| **`seedvr2_3b_convrot_int8.safetensors`** | 3B DiT Base | `int8_tensorwise` + ConvRot | **4.08 GB** | **5.69 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_3b_convrot_int8.safetensors) |

## Installation & Usage in ComfyUI

### 1. Model Placement
Place the downloaded `.safetensors` file into your ComfyUI models directory:

```text
ComfyUI/
└── models/
    └── SEEDVR2/
        ├── seedvr2_7b_convrot_w4a8.safetensors
        ├── seedvr2_7b_sharp_convrot_w4a8.safetensors
        ├── seedvr2_3b_convrot_w4a8.safetensors
        ├── seedvr2_7b_convrot_int8.safetensors
        └── seedvr2_3b_convrot_int8.safetensors
```

### 2. ComfyUI Node Setup
In the custom node repository [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT):
1. In the standard **`SeedVR2 (Down)Load DiT Model`** node, select your desired model (e.g. `seedvr2_7b_convrot_w4a8.safetensors`).
2. Connect to the **`SeedVR2 Video Upscaler`** node.
3. Run inference — enjoy substantial VRAM reductions with preserved FP16 precision on sensitive layers.

> **Important Compatibility Note**:
> W4A8 models are strictly supported via the standard DiT loader (**`SeedVR2 (Down)Load DiT Model`**). The DisTorch2 loader (**`SeedVR2 (Down)Load DiT Model with Distorch2`**) does **not** support W4A8 models due to blockwise CPU RAM streaming and weight-dispatch constraints specific to asymmetric 4-bit packed weights. For DisTorch2 offloading, use the ConvRot INT8 models.

## 📚 Detailed documentation

- [`benchmark/seedvr2_w4a8_traj_compare.py`](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/benchmark/seedvr2_w4a8_traj_compare.py)
- [`seedvr2_int8_traj_compare.py`](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/benchmark/seedvr2_int8_traj_compare.py)
- [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR)
- [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
- [ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT)

## 🔒 Security

**Backdoor audit.** Each file's audit status - verdict strength, weight coverage, the record behind it,
and the limits of what that measurement can show - is in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Backdoor_Audit_Verdicts.md). The audit code, the procedure and the raw records are in the same
repository.
