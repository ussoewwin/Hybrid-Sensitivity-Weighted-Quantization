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

- **`clip_l_hswq_convrot_int8.safetensors`**: Standalone OpenAI CLIP ViT-L/14 text encoder quantized via HSWQ ConvRot INT8 (~142.0 MB), offering massive memory savings while maintaining 100% prompt parsing integrity across SD 1.5, SDXL, and FLUX workflows.
- **`CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors`**: High-fidelity ViT-L/14 text encoder with Sparse Autoencoder (SAE) feature representation, covering both the text and vision towers, with HSWQ sensitivity protection (~690.9 MB).
- **`Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors`**: CLIP ViT-L/14 with an extended text context (248-token position embedding), both towers, HSWQ ConvRot INT8 (~677.6 MB).
- **`ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors`**: CLIP ViT-L/14 in Hugging-Face layout, both towers, HSWQ ConvRot INT8 (~678.1 MB).
- **`flan_t5_xxl_hswq_convrot_int8.safetensors`**: The 11B-parameter dense Google Flan-T5 XXL text encoder (encoder + decoder) quantized with HSWQ ConvRot INT8 (~14.74 GB vs ~21.49 GB FP16 baseline), reducing the footprint by ~31% to enable high-capacity prompt conditioning on consumer GPUs.
- **`Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors`**: Compact 0.6B abliterated Qwen3 language model quantized with HSWQ ConvRot INT8 (~687.0 MB), providing fast, lightweight prompt expansion and text conditioning with uncensored safety vectors.

---

## 📦 Available Models

| Filename | Architecture | Upstream Base Model | Quantization | File Size | Baseline Size | VRAM Savings | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clip_l_hswq_convrot_int8.safetensors` | OpenAI CLIP ViT-L/14 (FLUX.1 CLIP-L) | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) | HSWQ ConvRot INT8 | **~142.0 MB** | ~234.7 MB (FP16) | **~40% Reduction** | Available |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` | CLIP ViT-L/14, text + vision (SAE) | OpenAI CLIP ViT-L/14 (SAE) | HSWQ ConvRot INT8 | **~690.9 MB** | ~1.63 GB (FP32) | **~58% Reduction** | Available |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` | CLIP ViT-L/14, extended text, text + vision | `Long-ViT-L-14-BEST-GmP-smooth-ft-full` | HSWQ ConvRot INT8 | **~490.6 MB** | ~1.63 GB (FP32) | **~71% Reduction** | Available |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors` | CLIP ViT-L/14 (HF layout), text + vision | `ViT-L-14-BEST-smooth-GmP-HF-format` | HSWQ ConvRot INT8 | **~678.1 MB** | ~888.3 MB (FP16) | **~24% Reduction** | Available |
| `flan_t5_xxl_hswq_convrot_int8.safetensors` | Google Flan-T5 XXL (11B), encoder + decoder | [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) | HSWQ ConvRot INT8 | **~14.74 GB** | ~21.49 GB (FP16) | **~31% Reduction** | Available |
| `Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors` | Qwen3 0.6B Abliterated | Josiefied-Qwen3-0.6B-abliterated-v1 | HSWQ ConvRot INT8 | **~687.0 MB** | ~1.14 GB (BF16) | **~40% Reduction** | Available |

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
- **Single CLIP:** `Load CLIP` / `CLIPLoader` $\rightarrow$ select `clip_l_hswq_convrot_int8.safetensors` or `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors`.
- **Dual CLIP (SDXL):** `DualCLIPLoader` $\rightarrow$ set `clip_name1` to `clip_l_hswq_convrot_int8.safetensors` (type `sdxl`).
- **Triple CLIP (SD3 / FLUX.1):** `TripleCLIPLoader` $\rightarrow$ assign CLIP-L and `flan_t5_xxl_hswq_convrot_int8.safetensors` accordingly.
- **HSWQ Custom Node:** Use [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools) for dedicated HSWQ graph execution and precision management.

**Loading note (measured 2026-10-28).** The INT8 execution result depends on the route, and it
was measured rather than assumed. Before 2026-10-28 the CLIP-shaped files in this repository
(`clip_l_hswq_convrot_int8.safetensors`, `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors`,
`Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors`,
`ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors`) reached `CLIPLoader` with no
quantisation metadata - ComfyUI core only looks for the `comfy_quant` stamp in its MiniMax and
Ming branches - so the raw int8 codes landed in float16 parameters and the encoder produced
output uncorrelated with its own source, with no error and no warning. [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)
now supplies that metadata on the CLIP text and clip-vision routes, and the packs were re-measured
through it:

| pack | declared layers applied | raw int8 codes left | INT8 GEMM calls |
| :--- | ---: | ---: | ---: |
| `clip_l_hswq_convrot_int8` | 63 of 63 | 0 | 372 |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8` | 44 text + 84 vision of 129 | 0 | 258 |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8` | 45 text + 84 vision of 130 | 0 | 258 |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8` | 43 text + 84 vision of 127 | 0 | 252 |
| `Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8` | 157 of 157 | 0 | 936 |

Without that loader version installed, use the float files; the INT8 packs do not execute as
INT8 on the stock CLIP route. The stored data was correct in both cases: every int8 matrix
matches a control re-quantisation of its source (63 of 63, ratio 1.000) and every float-kept
tensor is bit-identical to the source (133 of 133). Measurements, root cause and the load-path
gate that now enforces this: [Clip-files INT8 load path](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/ClipFiles_INT8_TextEncoder_Load_Path_2026-10-28.md).

---

## 🔒 Security

Every text encoder hosted here is audited for a backdoor **before** the pack is published, and
the audit runs on the source model the pack is quantised from - a pack cannot carry behaviour
its source does not carry.

`clip_l.safetensors`, the source of `clip_l_hswq_convrot_int8.safetensors`, was verified to be
the official OpenAI ViT-L/14 text encoder: all 196 tensors equal the official release rounded to
fp16, with no tensor differing by more than that rounding. The pack was then verified against
it: 133 float-kept tensors bit-exact, 63 quantised matrices identical to a control
re-quantisation of the official file, and the reconstructed weights producing embeddings within
0.9999 cosine of the official encoder on every prompt tested, including trigger-style phrases.

The audits, the measurements they rest on, and the limits of what each one can show are in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Backdoor_Audit_Verdicts.md).

---

## 📜 Credits & License

- **OpenAI CLIP ViT-L/14:** [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) (MIT License)
- **Google Flan-T5 XXL:** [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) (Apache-2.0 License)
- **Josiefied Qwen3 0.6B Abliterated:** Qwen Team / Community Abliterated Releases (Apache-2.0 License)
- **Quantization Pipeline:** [HSWQ / ConvRot INT8 Suite](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **Base Licenses:** Please adhere to the individual upstream licenses of the respective foundation models.
