---
license: other
base_model:
  - comfyanonymous/flux_text_encoders
  - zer0int/CLIP-SAE-ViT-L-14
  - zer0int/CLIP-GmP-ViT-L-14
  - zer0int/LongCLIP-GmP-ViT-L-14
  - google/flan-t5-xxl
  - Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1
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

**HSWQ ConvRot INT8** (Hybrid-Sensitivity-Weighted-Quantization) quantized weights for text encoders, CLIP, and compact language models used by **SD 1.5**, **SDXL** and **FLUX.1** workflows.

- **Technical Details & Quantization Framework:** [Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **ComfyUI Loader for ConvRot INT8:** [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

## 📦 Available Models
| Filename | Architecture | Upstream Base Model | Quantization | File Size | Baseline Size | Size Reduction | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clip_l_hswq_convrot_int8.safetensors` | OpenAI CLIP ViT-L/14 (FLUX.1 CLIP-L) | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) | HSWQ ConvRot INT8 | **~142.0 MB** | ~246.1 MB (FP16) | **~42.3% Reduction** | Available |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` | CLIP ViT-L/14, text + vision (SAE) | [zer0int/CLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14) | HSWQ ConvRot INT8 | **~690.9 MB** | ~1.71 GB (FP32) | **~59.6% Reduction** | Available |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` | CLIP ViT-L/14, extended text, text + vision | [zer0int/LongCLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14) | HSWQ ConvRot INT8 | **~514.4 MB** | ~1.71 GB (FP32) | **~69.9% Reduction** | Available |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors` | CLIP ViT-L/14 (HF layout), text + vision | [zer0int/CLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) | HSWQ ConvRot INT8 | **~678.1 MB** | ~931.4 MB (FP16) | **~27.2% Reduction** | Available |
| `flan_t5_xxl_hswq_convrot_int8.safetensors` | Google Flan-T5 XXL (11B), encoder + decoder | [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) | HSWQ ConvRot INT8 | **~14.74 GB** | ~22.53 GB (FP16) | **~34.6% Reduction** | Available |
| `Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors` | Qwen3 0.6B Abliterated | [Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1](https://huggingface.co/Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1) | HSWQ ConvRot INT8 | **~687.0 MB** | ~1.19 GB (BF16) | **~42.4% Reduction** | Available |

Sizes are decimal MB/GB computed from the published byte counts, and each reduction is computed from those same byte counts against the source file in the same row.

## 🚀 Usage in ComfyUI

### Model Placement
Download the desired `.safetensors` files and place them into your ComfyUI text encoder directory:
```
ComfyUI/models/clip/
```
*(or `ComfyUI/models/LLM/` for Qwen3 language models)*

### Loading
Load directly via standard ComfyUI CLIP loader nodes:
- **Single CLIP:** `Load CLIP` / `CLIPLoader` $\rightarrow$ the CLIP-family packs in this repository are **not** applied on this route (the loading note below has the measurements), so use the float original here; the Qwen3-0.6B pack is applied on this loader (`CLIPLoader` type `boogu`).
- **Dual CLIP (SDXL):** `DualCLIPLoader` $\rightarrow$ the `clip_l` pack is **not** applied on this route either; set `clip_name1` to the float `clip_l.safetensors` (type `sdxl`).
- **Triple CLIP (SD3 / FLUX.1):** `TripleCLIPLoader` $\rightarrow$ assign the float CLIP-L and `flan_t5_xxl_hswq_convrot_int8.safetensors`; the T5 pack is applied on the FLUX `clip_text` route.
- **HSWQ Custom Node:** Use [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools) for dedicated HSWQ graph execution and precision management.

Route-by-route measurements, including which load paths apply the INT8 and which do not:
[Clip-files INT8 load path](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/ClipFiles_INT8_TextEncoder_Load_Path_2026-10-28.md).

## 🔒 Security

**Backdoor audit.** Each file's audit status - verdict strength, weight coverage, the record behind it,
and the limits of what that measurement can show - is in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Backdoor_Audit_Verdicts.md). The audit code, the procedure and the raw records are in the same
repository.

## 📜 Credits & License
- **OpenAI CLIP ViT-L/14:** [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) (MIT License)
- **CLIP-SAE and CLIP-GmP checkpoints:** [zer0int/CLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14) and [zer0int/CLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) (MIT License each)
- **LongCLIP-GmP checkpoint:** [zer0int/LongCLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14) (a CLIP model by OpenAI, MIT License); its declared base [BeichenZhang/LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L) is Apache-2.0 License
- **Google Flan-T5 XXL:** [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) (Apache-2.0 License)
- **Josiefied Qwen3 0.6B Abliterated:** [Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1](https://huggingface.co/Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1); its base `Qwen/Qwen3-0.6B` is Apache-2.0 License
- **Quantization Pipeline:** [HSWQ / ConvRot INT8 Suite](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **Base Licenses:** Please adhere to the individual upstream licenses of the respective foundation models.
