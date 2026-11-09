---
license: other
tags:
- controlnet
- text-to-image
- image-to-image
- inpainting
- sd-1-5
- stable-diffusion
- sdxl
- stable-diffusion-xl
- illustrious-xl
- noobai-xl
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

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity **ConvRot INT8** quantized weights for multi-condition and dedicated ControlNet models across diverse generative architectures (**SD1.5**, **Illustrious-XL / SDXL**, **SDXL 1.0**, **NoobAI-XL**, **Z-Image-Turbo**, **Qwen-Image / Qwen-Image-2512**, and **FLUX.1-dev**).

- **Technical Details & Quantization Framework:** [Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **ComfyUI Loader for ConvRot INT8:** [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

## 📦 Available Models
Sizes are decimal (1 GB = 10\^9 bytes) and exact; the `SHA256` column is the pack's own hash, so a download can be verified against it.

| Filename | Base Architecture | Base Model / Author | Supported Conditions | Quantization | File Size | SHA256 | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CN-anytest4_illustrious2_A_hswq_convrot_int8.safetensors` | Illustrious-XL / SDXL | [2vXpSwA7/iroiro-lora (Anytest v4 Variant A)](https://huggingface.co/2vXpSwA7/iroiro-lora) | Multi-condition (Canny, Lineart, Depth, Pose, Structure) | ConvRot INT8 | 1425.1 MB | `be9ff80b2f4895684c7114afff12220394f838179b5c82eadd8ebc46f5d82c25` | Fair AI / OpenRAIL++-M |
| `CN-anytest4_illustrious2_B_hswq_convrot_int8.safetensors` | Illustrious-XL / SDXL | [2vXpSwA7/iroiro-lora (Anytest v4 Variant B)](https://huggingface.co/2vXpSwA7/iroiro-lora) | Multi-condition (Canny, Lineart, Depth, Pose, Structure) | ConvRot INT8 | 1423.5 MB | `7e34dc3cc3334afa2b9fe5461131a1f61e289a83b1a11413d6c687e94b9dde6e` | Fair AI / OpenRAIL++-M |
| `controlnet-union-pro-max-sdxl-1.0_hswq_convrot_int8.safetensors` | SDXL 1.0 | [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) | OpenPose, Depth, Canny, Lineart, Anime Lineart, Scribble, Soft Edge, Normal, Segment, Tile, Inpaint | ConvRot INT8 | 1435.3 MB | `dcdf6f2037205c4a40b1f8863a91827ca3a5b55ec3261e07bb13d2dc855b47fb` | OpenRAIL++-M / Apache-2.0 |
| `noobaiXLControlnet_epsLineartRealistic_hswq_convrot_int8.safetensors` | NoobAI-XL / SDXL | [Eugeoter/noob-sdxl-controlnet-lineart_realistic](https://huggingface.co/Eugeoter/noob-sdxl-controlnet-lineart_realistic) | Lineart (Realistic) | ConvRot INT8 | 1420.2 MB | `86e795137dcf8d2876784eeecc2d61365c385f6caf4a5176d225ecd57fdcc914` | Fair AI Public 1.0-SD |
| `FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_convrot_int8.safetensors` | FLUX.1-dev | [Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0) | Canny, Depth, Pose, Blur, Gray, Soft Edge, Low Quality | ConvRot INT8 | 2710.6 MB | `fbcba3c76a221fba461c83680f682d3ff7aa813ef554c42e3a5449b981090c46` | Other / Non-Commercial |
| `Qwen-Image-2512-Fun-Controlnet-Union-2602_hswq_convrot_int8.safetensors` | Qwen-Image-2512 | [alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union) | Canny, HED, Depth, Pose, MLSD, Scribble, Gray, Inpaint | ConvRot INT8 | 1758.7 MB | `5aec39a53f583bade3b7db18f38ed5a01905a0610c2ad2f9f792802178596bc4` | Apache-2.0 |
| `Qwen-Image-ControlNet-Inpainting_hswq_convrot_int8.safetensors` | Qwen-Image | [alibaba-pai/Qwen-Image-ControlNet-Inpainting](https://huggingface.co/alibaba-pai/Qwen-Image-ControlNet-Inpainting) | Inpainting, Image Editing | ConvRot INT8 | 2120.3 MB | `38f0ab7495029517d4b0bb4187cedcbdfa090a29f0bb50dfa5733f5076a0c451` | Apache-2.0 |
| `control_v11f1e_sd15_tile_hswq_convrot_int8.safetensors` | SD1.5 | [lllyasviel/ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1) | Tile, Super-Resolution, Detail Enhancing | ConvRot INT8 | 607.6 MB | `e90a2cb010ea2e180b75ccc722dff476d8617f6b474dbadcb82423a34102de60` | OpenRAIL / CreativeML Open RAIL-M |
| `control_v11p_sd15_canny_hswq_convrot_int8.safetensors` | SD1.5 | [lllyasviel/ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1) | Canny Edge Detection | ConvRot INT8 | 607.6 MB | `f7e20fa625863f0225d39a6c8feddbfeb77b2ba6a95dc13d4cf144c70c2bdb36` | OpenRAIL / CreativeML Open RAIL-M |
| `control_v11f1p_sd15_depth_hswq_convrot_int8.safetensors` | SD1.5 | [lllyasviel/ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1) | Depth Map Guidance | ConvRot INT8 | 607.6 MB | `45086721b1ff4643a3ac50553eacc287e28d83fd7684bdc0dee7a8313cb6272b` | OpenRAIL / CreativeML Open RAIL-M |
| `control_v11e_sd15_ip2p_hswq_convrot_int8.safetensors` | SD1.5 | [lllyasviel/ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1) | Instruct Pix2Pix (Image Editing) | ConvRot INT8 | 607.6 MB | `43e0884402b1801f2fdfdf05e187a581090bbc0a679a187b7d7626133a0f94bd` | OpenRAIL / CreativeML Open RAIL-M |
| `qwen_image_canny_diffsynth_controlnet_hswq_convrot_int8.safetensors` | Qwen-Image | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Canny Edge Detection & Structural Guidance | ConvRot INT8 | 1135.9 MB | `c2899255252b81fd64ef2fae0fbfd86e1496ba04cf10cc95fe45679120f1c996` | Apache-2.0 |
| `qwen_image_depth_diffsynth_controlnet_hswq_convrot_int8.safetensors` | Qwen-Image | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Geometric Depth Map Guidance | ConvRot INT8 | 1135.9 MB | `7b82a9360169fa99524c03bcf913ce63713930174ad1fc332e553e5646bad44c` | Apache-2.0 |
| `qwen_image_inpaint_diffsynth_controlnet_hswq_convrot_int8.safetensors` | Qwen-Image | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Masked Inpainting & Localized Editing | ConvRot INT8 | 1135.9 MB | `471b6affc7fcedafb9940fafd6354dd6eb5fac5705e647029c90598c407c82d3` | Apache-2.0 |
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps_hswq_convrot_int8.safetensors` | Z-Image-Turbo | [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | Multi-condition (Canny, Depth, Pose, Color, Scribble, Inpaint) | ConvRot INT8 | 1009.7 MB | `1dd12b2d028b12df6b2a4f93856f8ab161c1edc05125e3b1c987a1694e99dba4` | Apache-2.0 |
| `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps_hswq_convrot_int8.safetensors` | Z-Image-Turbo | [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | Tile, Super-Resolution, Detail Enhancing | ConvRot INT8 | 1009.7 MB | `2998f83ef65903b4b6b22fb7ac270ed66f9a2443220563aa43db4188024763a8` | Apache-2.0 |

## 🚀 Usage in ComfyUI
Install the loader extension:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools.git
```

Then place each file in the directory its type requires (see the table at the top):

- **ControlNet** files go in `ComfyUI/models/controlnet/` and load with the **HSWQ ControlNet Loader** node.
- **Model patch** files go in `ComfyUI/models/model_patches/` and load with the **HSWQ Model Patch Loader** node, then apply with `ZImageFunControlnet` / `QwenImageDiffsynthControlnet`.

## 📜 Credits & License

### Base Models & Research
- **Illustrious Anytest ControlNet:** [2vXpSwA7/iroiro-lora](https://huggingface.co/2vXpSwA7/iroiro-lora) by **2vXpSwA7**
- **NoobAI-XL ControlNet:** [Eugeoter/noob-sdxl-controlnet-lineart_realistic](https://huggingface.co/Eugeoter/noob-sdxl-controlnet-lineart_realistic) by **Eugeoter** (Fair AI Public 1.0-SD)
- **SDXL ControlNet Union:** [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) by **xinsir**
- **SD1.5 ControlNets (tile, canny, depth, ip2p):** [lllyasviel/ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1) by **lllyasviel** (OpenRAIL)
- **Z-Image-Turbo ControlNet Models:** [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) & [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (Apache-2.0)
- **Qwen-Image ControlNet Models:** [alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union), [alibaba-pai/Qwen-Image-ControlNet-Inpainting](https://huggingface.co/alibaba-pai/Qwen-Image-ControlNet-Inpainting) & [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (Apache-2.0)
- **Qwen-Image DiffSynth ControlNet Models:** [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) by **ModelScope / Alibaba Group** (Apache-2.0)
- **FLUX.1-dev ControlNet Union Pro 2.0:** [Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0) & [InstantX Team](https://huggingface.co/InstantX) (FLUX.1-dev Non-Commercial License)
- **Base Architectures:** [Illustrious-XL](https://civitai.red/models/675574/illustrious-xl), [Stability AI SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [Stability AI SDXL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) & [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

---

**Disclaimer:** These models are provided for optimization, workflow acceleration, and research purposes. Please adhere to the licenses and terms of the respective base models.

## 📚 Detailed documentation

- [ControlNet ConvRot INT8 trajectory impact guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/ControlNet_ConvRot_INT8_Trajectory_Impact_Technical_Guide.md)
- [SD1.5 tile ControlNet guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/CN_SD15_Tile_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 canny ControlNet guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/CN_SD15_Canny_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 depth ControlNet guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/CN_SD15_Depth_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 ip2p ControlNet guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/CN_SD15_Ip2p_ConvRot_INT8_Impact_Technical_Guide.md)
- [Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/backdoor/Backdoor_Audit_Verdicts.md)

## 🔒 Security

**Backdoor audit.** Each file's audit status - verdict strength, weight coverage, the record behind it,
and the limits of what that measurement can show - is in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/backdoor/Backdoor_Audit_Verdicts.md). The audit code, the procedure and the raw records are in the same
repository.
