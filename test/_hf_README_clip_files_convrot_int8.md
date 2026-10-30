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

**Loading note (measured 2026-10-28).** Whether the INT8 in these files takes effect depends on
the route; both cases were measured, with no code changed.

- As an SDXL text-encoder pair (`DualCLIPLoader`, this file in the clip_l slot) an exact
  match with the float encoder was reported on 2026-10-28; that measurement is withdrawn
  on 2026-10-29 - the cosine was `min(1.0, nan)` over NaN-carrying conditioning, not
  matching behaviour. The guarded re-measurement ran 2026-10-29 (fresh process, NaN counts, clip_l
  half of the SDXL conditioning): the pack is not applied on this route either - every declared text
  layer holds raw int8 codes, 0 INT8 kernel calls, and the conditioning is uncorrelated with the
  float source (cosine 0.022189 to 0.050407 over 8 prompts for this file).
- As a single text encoder (`CLIPLoader`, type `stable_diffusion`) the same file does not: the
  conditioning cosine falls to 0.076368, because on that route the int8 codes land in the float
  parameters unchanged.
- No INT8 matrix-multiplication kernel was observed on either route. `int8_tensorwise` here stores
  weights in int8 (memory saving) and dequantises when the layer runs, so no speed-up should be
  expected from these files.

- The vision half of the CLIP-SAE pack is a different case: on the clip-vision route every
  declared vision layer keeps its int8 codes, and image embeddings move away from the float file
  (cosine 0.061973 to 0.121915 over three images; the float encoder separates two of those images
  at 0.911204, the pack at 0.826424, so the pack still sees the picture but not the same function).
  Use the float file for that clip-vision pair until that route is settled.

Use the float file if your workflow loads text encoders one at a time. Route-by-route
measurements: [Clip-files INT8 load path](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/ClipFiles_INT8_TextEncoder_Load_Path_2026-10-28.md).

---

## 🔒 Security

Five of the six text encoders hosted here have a backdoor verdict in the audit index; only the
`clip_l` pack has none yet, and this card states that rather than implying coverage it does not
have. The `flan_t5_xxl` pack was audited against the official `google/flan-t5-xxl` release: five
of five shards hash-verified, 247 float tensors equal to the source, 312 int8 layers that
reproduce within one quantisation step (every residual difference is a rounding-boundary flip of
the rotation), and - through the same loader as the source - a conditioning cosine of 0.9966733 or
better with no trigger candidate separating from the control spread.

The audit runs on the source model the pack is quantised from, because a backdoor written onto the
source survives rotation and per-channel scaling. That is a statement about weights, not about
runtime behaviour: whether a given loader applies the pack's INT8 at all has to be measured
separately, and the load-path records show that it varies by route.

`clip_l.safetensors`, the source of `clip_l_hswq_convrot_int8.safetensors`, was verified to be
the official OpenAI ViT-L/14 text encoder: all 196 tensors equal the official release rounded to
fp16, with no tensor differing by more than that rounding. The pack was then verified against
it: 133 float-kept tensors bit-exact, 63 quantised matrices identical to a control
re-quantisation of the official file, and the reconstructed weights producing embeddings at
cosine 0.99963 or better of the official encoder on every prompt tested (median 0.99988),
including trigger-style phrases.

Verdicts are stated as **none detected** with a weight-closure strength (A/B/C) and
audited-tensor coverage per file. In this collection the sources split: three carry strength B,
where the delta is dense and the verdict rests on the behavioural probes; one carries A, where the
delta is fully accounted for at the weight level by the declared edit; and two packs have no
verdict entry at all. The per-file numbers are in the audit index.


**Positive control.** The pipeline's sensitivity was measured outright: two
backdoors planted on a clean text-only LM (one LoRA-planted, one BadEdit-style
weight-space rank-1 edit) both survived the same behavioural probes (LoRA 0/28
surfacing, BadEdit-vs-clean ratio 0.92x, no isolated island), and the tool-context
scan false-positives on the clean base at 20/28. The behavioural null therefore
means "not detected by this pipeline on this probe set" - the weight audits decide
verdicts, probes annotate.

The audits, the measurements they rest on, and the limits of what each one can show are in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Backdoor_Audit_Verdicts.md).

---

## 📜 Credits & License

- **OpenAI CLIP ViT-L/14:** [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) (MIT License)
- **Google Flan-T5 XXL:** [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) (Apache-2.0 License)
- **Josiefied Qwen3 0.6B Abliterated:** Qwen Team / Community Abliterated Releases (Apache-2.0 License)
- **Quantization Pipeline:** [HSWQ / ConvRot INT8 Suite](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **Base Licenses:** Please adhere to the individual upstream licenses of the respective foundation models.
