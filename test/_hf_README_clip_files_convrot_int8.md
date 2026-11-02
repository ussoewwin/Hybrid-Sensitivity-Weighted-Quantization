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

---

## 🌟 Model Overview

This repository hosts **HSWQ ConvRot INT8** quantized weights for text encoders and conditioning backbones. By combining orthogonal Hadamard rotation ($W_{rot} = W \cdot H^T$) with a per-model sensitivity measurement, the layers that are quantised are chosen from that model's own measured drift:

- **`clip_l_hswq_convrot_int8.safetensors`**: Standalone OpenAI CLIP ViT-L/14 text encoder quantized via HSWQ ConvRot INT8 (~142.0 MB, 42.3 % smaller than its 246.1 MB FP16 source); the loading note below records on which routes its INT8 is actually applied.
- **`CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors`**: ViT-L/14 text encoder with Sparse Autoencoder (SAE) feature representation, covering both the text and vision towers, with HSWQ sensitivity protection (~690.9 MB, 59.6 % smaller than its 1.71 GB FP32 source).
- **`Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors`**: CLIP ViT-L/14 with an extended text context (248-token position embedding), both towers, HSWQ ConvRot INT8 (~514.4 MB, 69.9 % smaller than its 1.71 GB FP32 source). This build was rebuilt from zer0int's published `BEST-GmP-smooth-ft`; the earlier build published under this name (677,643,528 bytes) came from a local file that matched no published checkpoint and was replaced.
- **`ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors`**: CLIP ViT-L/14 in Hugging-Face layout, both towers, HSWQ ConvRot INT8 (~678.1 MB, 27.2 % smaller than its 931.4 MB FP16 source).
- **`flan_t5_xxl_hswq_convrot_int8.safetensors`**: The 11B-parameter dense Google Flan-T5 XXL text encoder (encoder + decoder) quantized with HSWQ ConvRot INT8 (~14.74 GB vs ~22.53 GB FP16 source, 34.6 % smaller), for high-capacity prompt conditioning on consumer GPUs. Its INT8 path is applied on the FLUX `clip_text` route: 128 of the 128 layers that route instantiates are applied, none is left holding raw int8 code, and the kernel fires 256 times.
- **`Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors`**: Compact 0.6B abliterated Qwen3 language model quantized with HSWQ ConvRot INT8 (~687.0 MB, 42.4 % smaller than its 1.19 GB BF16 source), for fast, lightweight prompt expansion and text conditioning with uncensored safety vectors. Its INT8 path is applied on the `CLIPLoader` type `boogu` route: 157 of 157 declared layers applied, no raw int8 code left in a float parameter, 2,496 kernel calls, conditioning cosine 0.999071 to 0.999574 against the source.

---

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

---

## 🛠️ Key Features

- **Orthogonal Hadamard Pre-Rotation (ConvRot):** Weight matrices are pre-rotated group-wise along the input feature dimension ($W_{rot} = W \cdot H^T$) using power-of-4 normalized Hadamard blocks (`groupsize=256`), eliminating catastrophic dynamic range compression caused by outlier channels.
- **Hybrid Sensitivity Weighting (HSWQ):** The FP16 protection set is chosen per model from that model's own measured per-layer drift, and the result is accepted on a fixed 256-item set (seed 42) with the gate mean >= 0.99 and min >= 0.98.
- **Native ComfyUI Compatibility:** Encodes standard `comfy_quant` header metadata (`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`) and per-channel scale tensors, so the files load in ComfyUI with standard `CLIPLoader`, `DualCLIPLoader`, or `TripleCLIPLoader` and no custom node. Whether the INT8 is then executed depends on the route - see the loading note below.
- **Full Precision Preservation for Critical Weights:** The 1-D tensors (layer normalizations and biases) are kept in native floating-point precision; the 2-D weight matrices are quantised, including the token embedding in `clip_l`, which reproduces to 1.3e-7 against the source.
- **Hardware Acceleration:** Where the INT8 path runs, the weights are executed as INT8 matrix multiplications on the Tensor Core path - 256 kernel calls for the T5 pack on the FLUX `clip_text` route, 2,496 for the Qwen3-0.6B pack on the `boogu` route. The four CLIP-family files ran no INT8 kernel on the routes measured (0 calls), so no speed-up should be expected from them as shipped.

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
- **Single CLIP:** `Load CLIP` / `CLIPLoader` $\rightarrow$ the CLIP-family packs in this repository are **not** applied on this route (the loading note below has the measurements), so use the float original here; the Qwen3-0.6B pack is applied on this loader (`CLIPLoader` type `boogu`).
- **Dual CLIP (SDXL):** `DualCLIPLoader` $\rightarrow$ the `clip_l` pack is **not** applied on this route either; set `clip_name1` to the float `clip_l.safetensors` (type `sdxl`).
- **Triple CLIP (SD3 / FLUX.1):** `TripleCLIPLoader` $\rightarrow$ assign the float CLIP-L and `flan_t5_xxl_hswq_convrot_int8.safetensors`; the T5 pack is applied on the FLUX `clip_text` route.
- **HSWQ Custom Node:** Use [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools) for dedicated HSWQ graph execution and precision management.

**Loading note (measured 2026-10-28, re-measured under a guard 2026-10-29).** Whether the
INT8 in these files takes effect depends on the route. Every row below was measured with the loader
node imported and no code changed; the guarded runs used a fresh process per artifact, a source
sha256 re-verified before the load, a float64 cosine that returns no value unless every sum is
finite, and the NaN count of both sides recorded per prompt.

| file | route | INT8 applied | conditioning cosine vs the float source | INT8 kernel calls |
| :--- | :--- | :--- | :--- | ---: |
| `clip_l_hswq_convrot_int8.safetensors` | `CLIPLoader`, `stable_diffusion` | no - 63 of 63 layers hold raw int8 code | 0.076368 | 0 |
| `clip_l_hswq_convrot_int8.safetensors` | `DualCLIPLoader` (SDXL pair) | no - 63 of 63 raw | 0.022189 to 0.050407 (8 prompts) | 0 |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` | `CLIPLoader` (text tower) | no - 44 of 45 raw | -0.009499 to 0.085984 (6 prompts) | 0 |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` | `DualCLIPLoader` (SDXL pair) | no - 44 of 45 raw | -0.005174 to 0.005855 (8 prompts) | 0 |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` | `comfy.clip_vision.load` | no - 84 of 84 raw | image embeddings 0.061973 to 0.121915 (3 images) | - |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` | `CLIPLoader` (text tower) | no - 45 of 46 raw | -0.005666 to 0.057231 (8 prompts) | 0 |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` | `DualCLIPLoader` (SDXL pair) | no - 45 of 46 raw | 0.029275 to 0.047834 (8 prompts) | 0 |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` | `comfy.clip_vision.load` | no - 84 of 84 raw | image embeddings 0.084569 to 0.087250 (3 images) | - |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors` | `CLIPLoader` | no - 43 of 43 raw | -0.023504 to 0.103258 (8 prompts) | 0 |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors` | `DualCLIPLoader` (SDXL pair) | no - 43 of 43 raw | -0.006227 to 0.003722 (8 prompts) | 0 |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors` | `comfy.clip_vision.load` | no - 84 of 84 raw | image embeddings 0.032509 to 0.040646 (3 images) | - |
| `flan_t5_xxl_hswq_convrot_int8.safetensors` | `comfy.sd.load_clip`, `clip_type=flux` | yes - 128 of the 128 layers the route builds (the 184 decoder layers are not built by it) | 0.9966733 or better (37 prompts) | 256 |
| `Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors` | `CLIPLoader`, type `boogu` | yes - 157 of 157 | 0.999071 to 0.999574 (8 prompts) | 2,496 |

In the `CLIP-SAE-ViT-L-14` and `Long-ViT-L-14-BEST-GmP-smooth-ft-full` text rows one declared layer is counted unmatched:
`text_projection`, which the checkpoint declares quantised and the runtime holds without a
`.weight` suffix.

- An exact match with the float encoder was reported for the SDXL pair route on 2026-10-28 and is
  **withdrawn**: the cosine was `min(1.0, nan)` over NaN-carrying conditioning, not matching
  behaviour. Every `1.000000` that came from a pair route was that clamp. The guarded
  re-measurement of 2026-10-29 replaced those rows with the measurements above.
- On the CLIP routes the int8 codes land in the float parameters unchanged - no scale is applied and
  the rotation is not inverted - so the conditioning is uncorrelated with the float file.
  `int8_tensorwise` stores the weights in int8 (smaller file) and dequantises when the layer runs,
  so a file that is not applied saves disk size but delivers neither a memory saving nor a speed-up.
- On the clip-vision route the CLIP-SAE pack still separates the three probe images (float encoder
  0.911204, pack 0.826424), so it sees the picture but not the same function. The Long pack goes
  further: the float encoder separates the three images at 0.9245, 0.9245 and 1.0000, while the
  pack returns 1.0000 for all three - its vision output stops distinguishing them.
- The `HSWQCheckpointLoader` route (keys `conditioner.embedders.*`) was closed as a boundary rather
  than measured: a scan of every `.safetensors` under `models/checkpoints` found 3 checkpoints and
  none with `conditioner.embedders` quantisation metadata, so there is no such route to measure.

For the four CLIP-family files use the float originals on these routes -
[comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) for
`clip_l`, [zer0int/CLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14),
[zer0int/LongCLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14) and
[zer0int/CLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) for the other three.
The T5 and Qwen3-0.6B packs are usable as published on the routes in the table. Route-by-route
measurements: [Clip-files INT8 load path](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/ClipFiles_INT8_TextEncoder_Load_Path_2026-10-28.md).

---

## 🔒 Security

All six files hosted here have a backdoor verdict in the audit index - for `clip_l` and `flan_t5_xxl` the verdict covers the pack file itself, for the other four it covers the source checkpoint the pack was quantised from. The `flan_t5_xxl` pack was audited against the official `google/flan-t5-xxl` release: five of five
shards hash-verified, 247 float tensors equal to the source, 312 int8 layers that reproduce
within one quantisation step (every residual difference is a rounding-boundary flip of the
rotation), and - through the same loader as the source - a conditioning cosine of 0.9966733 or
better with no trigger candidate separating from the control spread. The `clip_l` pack was
audited against the official OpenAI ViT-L/14 encoder: all 196 source tensors are the official
fp16 rounding (sha256 matching the Hub), 133 float tensors equal the source, 63 int8 layers
reproduce bit for bit under the control re-quantisation (19 one-LSB rounding-boundary flips
under TF32 arithmetic, scale 2.4e-7), and the token embedding - which this pack quantises -
reproduces to 1.3e-7.

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
audited-tensor coverage per file. In this collection the strengths split three A (`clip_l`,
`flan_t5_xxl`, and the Josiefied source) and three B (`CLIP-SAE`, `GmP-HF` and `Long-GmP`, where
the delta against the declared base is dense and the verdict rests on the behavioural probes); no
file here lacks a verdict entry. The per-file numbers are in the audit index.

- `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` - **none detected (B, 221/590 audited
  tensors)** on its source `CLIP-SAE-ViT-L-14-FP32` checkpoint
  (`zer0int/CLIP-SAE-ViT-L-14`, sha256 prefix `820afa79`), whose base is
  `openai/clip-vit-large-patch14`.
- `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors` - **none detected (B, 74/587
  audited tensors)** on its source `ViT-L-14-BEST-smooth-GmP-HF-format` checkpoint
  (`zer0int/CLIP-GmP-ViT-L-14`, sha256 prefix `46a201af`), whose base is
  `openai/clip-vit-large-patch14`.
- `Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors` - **none detected (A, 54/310
  audited tensors)** on its source `Josiefied-Qwen3-0.6B-abliterated-v1` checkpoint
  (`Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1`, sha256 prefix `538d7e68`), whose base is
  `Qwen/Qwen3-0.6B`.
- `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` - **none detected (B, 74/590
  audited tensors)** on the source file of the same name (sha256 prefix `4d3e6c8c`), which is a
  fine-tune of the declared base `BeichenZhang/LongCLIP-L` and not zer0int's published checkpoint
  (589 of its 590 tensors differ); the verdict rests on the delta against that base, not on
  identity with the publisher's file. The pack published here was rebuilt from zer0int's published
  `BEST-GmP-smooth-ft`, and its FP16-kept tensors match that checkpoint bit for bit.


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
- **CLIP-SAE and CLIP-GmP checkpoints:** [zer0int/CLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14) and [zer0int/CLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) (MIT License each)
- **LongCLIP-GmP checkpoint:** [zer0int/LongCLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14) (a CLIP model by OpenAI, MIT License); its declared base [BeichenZhang/LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L) is Apache-2.0 License
- **Google Flan-T5 XXL:** [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) (Apache-2.0 License)
- **Josiefied Qwen3 0.6B Abliterated:** [Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1](https://huggingface.co/Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1); its base `Qwen/Qwen3-0.6B` is Apache-2.0 License
- **Quantization Pipeline:** [HSWQ / ConvRot INT8 Suite](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **Base Licenses:** Please adhere to the individual upstream licenses of the respective foundation models.
