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

---

## 🌟 Model Overview

This repository provides production-grade **HSWQ ConvRot INT8** quantized weights for the **SAM 3.1 Multiplex** foundation model. By combining orthogonal Hadamard rotation with a per-model sensitivity measurement, the layers that are quantised are chosen from this checkpoint's own measured behaviour, and everything the measurement did not release stays in FP16:

- **Architecture:** SAM 3.1 Multiplex (Vision Backbone + 3-Level FPN Neck + Decoupled Memory Attention + 16-Object Multiplex Interactive Segmentation & Tracking)
- **Base Checkpoint:** Upstream SAM 3.1 Multiplex foundation weights from [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) / [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
- **Native ComfyUI Support:** Features native `int8_tensorwise` and `comfy_quant` metadata stamps, loading seamlessly with ComfyUI's standard `UNetLoader` / `DiffusionModelLoader` or `HSWQ SAM3 Loader` without requiring external manual dequantization workflows.

---

## 📦 Model Details

| Filename | Base Architecture | Precision / Format | File Size | Size Reduction vs FP16 | Native ComfyUI Support | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sam3.1_multiplex_hswq_convrot_int8.safetensors` | SAM 3.1 Multiplex | **HSWQ ConvRot INT8** (`int8_tensorwise`) | **~1,738.2 MB** (1.74 GB) | **~0.4% Reduction** vs the ~1,745.5 MB (1.75 GB) FP16 source - 41 of the 221 runtime-live Linear layers are quantised, the rest of the checkpoint stays FP16 | ✅ Supported natively (`UNetLoader`) | Available |

Sizes are decimal MB/GB computed from the published byte counts, and the reduction is computed from those same byte counts against the source file named in the same row.

### Quantization Breakdown
- **HSWQ ConvRot INT8 Linear Layers:** 41 of the checkpoint's 221 runtime-live Linear layers carry the INT8 path - all of them on the detector side, 23 of them attention output projections. Each is rotated group-wise along the input axis with normalized Sylvester Hadamard blocks (`groupsize=256`) and quantised per output channel.
- **FP16-kept Tensors:** Everything else stays in FP16 - the 457 tracker tensors (interactive mask decoder, plain mask decoder, memory fuser, prompt encoder), all 55 4D Conv2d weights, the 998 1D tensors (LayerNorms, biases, buffers), the CLIP token embedding and the position embedding. The CLIP text encoder is FP16 except for one resizer projection. No tensor in the pack is BF16.
- **How the protection set was chosen:** From this checkpoint's own measurement, then validated by a release search run through the real loader path: of the layers kept in FP16, none could be handed to INT8 without breaking the quality gate by an order of magnitude, so the shipped set is the smallest that holds the output.

---

## 🛠️ Key Highlights

- **Hadamard Orthogonal Rotation (ConvRot):** Weight matrices are pre-rotated group-wise along the input feature dimension ($W_{rot} = W \cdot H^T$) using power-of-4 normalized Hadamard blocks (`groupsize=256`), eliminating catastrophic dynamic range compression caused by outlier channels.
- **Hybrid Sensitivity Protection (HSWQ):** The FP16-kept set is this checkpoint's own, and it was validated by a release search through the real loader path - none of the kept layers could be handed to INT8 without breaking the quality gate by an order of magnitude.
- **Size and Execution Path:** ~1,738.2 MB against a ~1,745.5 MB FP16 source, a ~0.4% reduction, because only 41 of the 221 runtime-live Linear layers are both 256-wide and unprotected - the rest of the checkpoint is FP16 by design. The gain here is the INT8 **execution** path (Tensor Core INT8 matmul), not the disk size - the same trade the guide documents. Quality on the measured paths: tracker mask IoU 0.9978 against FP16 at the shipped protection threshold, and on the detector path the masks sit at median IoU 0.982682 (stock loader) / 0.958022 (HSWQ loader) with mask cosine never below 0.967490 / 0.979220. The measurement exercised the tracker point-prompt path and the detector text path; the video and propagation paths were not exercised and are not covered by these figures.

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

The pack was also verified against that source at byte level: all 1,590 source tensors are
present (none missing, none unexplained), the 41 quantised matrices reproduce the source's own
re-quantisation exactly (0 mismatching codes, scales exact), and every FP16-kept tensor equals
the source. The published pack is sha256 `2bcbcba10d462a7a`, the audited source is
`9ba99c92703c2e8b`.

Both loaders a user can pick were measured on 2026-10-29: the stock `UNetLoader` /
`load_diffusion_model` route and `HSWQ SAM3 Loader` each apply all 41 declared layers and
execute the INT8 kernel (468 calls over a 12-forward detector probe - 3 prompts x 4 images), with identical weight-state
counts. Against the FP16 source, detection masks over 3 prompts x 4 images sit at median IoU
0.982682 (stock) / 0.958022 (HSWQ loader) and mask cosine never below 0.967490 / 0.979220;
one comparison pair reads IoU 0.0 because both masks are empty there - a degenerate ratio,
recorded rather than dropped. These masks are near-empty (reference areas 3.8e-05 to 0.0206),
so the cosine figures carry the weight of the comparison.

Strength and coverage: this pack's source file carries **none detected (C, 96/1,590
audited tensors)** - 6% weight coverage is the weakest class in the audit table, so the
behavioural probes carry more of the weight here than on the strength-A files. The
positive control that scopes what that is worth is in the audit index.


**Positive control.** The pipeline's sensitivity was measured outright: two
backdoors planted on a clean text-only LM (one LoRA-planted, one BadEdit-style
weight-space rank-1 edit) both survived the same behavioural probes (LoRA 0/28
surfacing, BadEdit-vs-clean ratio 0.92x, no isolated island), and the tool-context
scan false-positives on the clean base at 20/28. The behavioural null therefore
means "not detected by this pipeline on this probe set" - the weight audits decide
verdicts, probes annotate.

The audit, its records and the limits of what was measured are in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Backdoor_Audit_Verdicts.md).

---

## 📜 Credits & License

- **Base Architecture & Model:** [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) / [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) (Meta SAM / Segment Anything Team)
- **Quantization Framework:** [HSWQ / ConvRot INT8 Pipeline](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **License:** [SAM License (Meta SAM License)](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)
