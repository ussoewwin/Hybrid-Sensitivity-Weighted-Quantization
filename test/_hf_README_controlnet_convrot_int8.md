---
license: other
tags:
- controlnet
- text-to-image
- image-to-image
- inpainting
- sdxl
- stable-diffusion-xl
- illustrious-xl
- z-image
- z-image-turbo
- qwen-image
- qwen-image-2512
- flux
- flux.1-dev
- diffsynth
- diffsynth-studio
- quantized
- int8
- convrot
- comfyui
pipeline_tag: image-to-image
---

# ControlNet Models (ConvRot INT8)

High-fidelity **ConvRot INT8** quantized weights for multi-condition and dedicated ControlNet models across diverse generative architectures (**Illustrious-XL / SDXL**, **SDXL 1.0**, **Z-Image-Turbo**, **Qwen-Image / Qwen-Image-2512**, and **FLUX.1-dev**).

---

## 🌟 Model Overview

This repository hosts high-quality **ConvRot INT8** quantized weights for all-in-one ControlNet Union and dedicated condition-specific ControlNet models. By applying orthogonal Hadamard rotation prior to per-channel INT8 quantization, these models effectively eliminate activation outlier distortion and drastically reduce VRAM and disk footprint while preserving precise structural control fidelity:

- **CN-anytest4_illustrious2 (Variants A & B)**: Multi-purpose all-in-one Anytest v4 ControlNet models fine-tuned for **Illustrious-XL / SDXL** by **2vXpSwA7**, offering high-precision anime and illustration structure guiding.
- **controlnet-union-pro-max-sdxl-1.0**: The comprehensive all-in-one ControlNet Union model for **SDXL 1.0** by **xinsir**, supporting 10+ control conditions in a single compact file.
- **Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps**: Multi-condition ControlNet Union model (lite 6-block architecture, 8-step distilled) for the **Z-Image-Turbo** architecture by **alibaba-pai**, providing lightweight and fast structure control (Canny, Depth, Pose, Color, etc.).
- **Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps**: Dedicated high-resolution Tile and upscaling ControlNet model (lite architecture, 8-step distilled) for the **Z-Image-Turbo** architecture by **alibaba-pai**, optimized for super-resolution detail enhancement.
- **Qwen-Image-2512-Fun-Controlnet-Union-2602**: Multi-condition ControlNet Union model (5 layer blocks) for the **Qwen-Image-2512** architecture.
- **Qwen-Image-ControlNet-Inpainting**: Dedicated inpainting and editing ControlNet model for the **Qwen-Image** architecture.
- **Qwen-Image DiffSynth ControlNet Suite**: Dedicated single-condition ControlNet models for the **Qwen-Image** architecture developed by **DiffSynth-Studio**:
  - **`qwen_image_canny_diffsynth_controlnet_convrot_int8`**: High-precision Canny edge structural control.
  - **`qwen_image_depth_diffsynth_controlnet_convrot_int8`**: Geometric spatial depth guidance.
  - **`qwen_image_inpaint_diffsynth_controlnet_convrot_int8`**: Masked inpainting and localized semantic replacement.
- **FLUX.1-dev-ControlNet-Union-Pro-2.0**: Next-generation unified 7-in-1 ControlNet for the **FLUX.1-dev** architecture by **Shakker Labs**.

---

## 📦 Available Models

| Filename | Base Architecture | Base Model / Author | Supported Conditions | Quantization | File Size | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CN-anytest4_illustrious2_A_convrot_int8.safetensors` | Illustrious-XL / SDXL | [2vXpSwA7/iroiro-lora (Anytest v4 Variant A)](https://huggingface.co/2vXpSwA7/iroiro-lora) | Multi-condition (Canny, Lineart, Depth, Pose, Structure) | ConvRot INT8 | ~1.40 GB | Fair AI / OpenRAIL++-M |
| `CN-anytest4_illustrious2_B_convrot_int8.safetensors` | Illustrious-XL / SDXL | [2vXpSwA7/iroiro-lora (Anytest v4 Variant B)](https://huggingface.co/2vXpSwA7/iroiro-lora) | Multi-condition (Canny, Lineart, Depth, Pose, Structure) | ConvRot INT8 | ~1.40 GB | Fair AI / OpenRAIL++-M |
| `controlnet-union-pro-max-sdxl-1.0_convrot_int8.safetensors` | SDXL 1.0 | [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) | OpenPose, Depth, Canny, Lineart, Anime Lineart, Scribble, Soft Edge, Normal, Segment, Tile, Inpaint | ConvRot INT8 | ~1.40 GB | OpenRAIL++-M / Apache-2.0 |
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps_convrot_int8.safetensors` | Z-Image-Turbo | [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | Multi-condition (Canny, Depth, Pose, Color, Scribble, Inpaint) | ConvRot INT8 | ~1.01 GB | Apache-2.0 |
| `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps_convrot_int8.safetensors` | Z-Image-Turbo | [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | Tile, Super-Resolution, Detail Enhancing | ConvRot INT8 | ~1.01 GB | Apache-2.0 |
| `Qwen-Image-2512-Fun-Controlnet-Union-2602_convrot_int8.safetensors` | Qwen-Image-2512 | [alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union) | Canny, HED, Depth, Pose, MLSD, Scribble, Gray, Inpaint | ConvRot INT8 | ~1.76 GB | Apache-2.0 |
| `Qwen-Image-ControlNet-Inpainting_convrot_int8.safetensors` | Qwen-Image | [alibaba-pai/Qwen-Image-ControlNet-Inpainting](https://huggingface.co/alibaba-pai/Qwen-Image-ControlNet-Inpainting) | Inpainting, Image Editing | ConvRot INT8 | ~2.12 GB | Apache-2.0 |
| `qwen_image_canny_diffsynth_controlnet_convrot_int8.safetensors` | Qwen-Image | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Canny Edge Detection & Structural Guidance | ConvRot INT8 | ~1.14 GB | Apache-2.0 |
| `qwen_image_depth_diffsynth_controlnet_convrot_int8.safetensors` | Qwen-Image | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Geometric Depth Map Guidance | ConvRot INT8 | ~1.14 GB | Apache-2.0 |
| `qwen_image_inpaint_diffsynth_controlnet_convrot_int8.safetensors` | Qwen-Image | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Masked Inpainting & Localized Editing | ConvRot INT8 | ~1.14 GB | Apache-2.0 |
| `FLUX.1-dev-ControlNet-Union-Pro-2.0_convrot_int8.safetensors` | FLUX.1-dev | [Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0) | Canny, Depth, Pose, Blur, Gray, Soft Edge, Low Quality | ConvRot INT8 | ~2.14 GB | Other / Non-Commercial |

---

## 🛠️ Key Features

- **All-in-One & Dedicated Condition Control**: Unified multi-condition architectures alongside specialized single-condition models (Canny, Depth, Tile, Inpainting) for fine-grained workflow flexibility.
- **ConvRot INT8 Precision**: Leverages orthogonal Hadamard rotations to redistribute channel-wise outlier spikes uniformly across dimensions, preventing quantization error buildup in deep control layers.
- **VRAM & Storage Optimization**: Slashes VRAM consumption and disk footprint by ~50% compared to unquantized FP16 checkpoints, allowing seamless multi-ControlNet workflows on consumer GPUs.

---

## 🚀 Usage in ComfyUI

To load and execute these ConvRot INT8 ControlNet models in ComfyUI, please use the dedicated loader node from the **ComfyUI-HSWQ-Loader-and-Tools** extension:

- **Extension Repository:** [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

### Installation

Clone the repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools.git
```

Place the downloaded `.safetensors` files into your ComfyUI `models/controlnet/` directory and load them using the dedicated ControlNet loader node.

---

## 📜 Credits & License

### Base Models & Research
- **Illustrious Anytest ControlNet:** [2vXpSwA7/iroiro-lora](https://huggingface.co/2vXpSwA7/iroiro-lora) by **2vXpSwA7**
- **SDXL ControlNet Union:** [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) by **xinsir**
- **Z-Image-Turbo ControlNet Models:** [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) & [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (Apache-2.0)
- **Qwen-Image ControlNet Models:** [alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union), [alibaba-pai/Qwen-Image-ControlNet-Inpainting](https://huggingface.co/alibaba-pai/Qwen-Image-ControlNet-Inpainting) & [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (Apache-2.0)
- **Qwen-Image DiffSynth ControlNet Models:** [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) by **ModelScope / Alibaba Group** (Apache-2.0)
- **FLUX.1-dev ControlNet Union Pro 2.0:** [Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0) & [InstantX Team](https://huggingface.co/InstantX) (FLUX.1-dev Non-Commercial License)
- **Base Architectures:** [Illustrious-XL](https://civitai.red/models/675574/illustrious-xl), [Stability AI SDXL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) & [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

---

**Disclaimer:** These models are provided for optimization, workflow acceleration, and research purposes. Please adhere to the licenses and terms of the respective base models.
