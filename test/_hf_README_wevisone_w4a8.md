---
base_model:
  - tencent/WeVisOne-VL-4B
tags:
  - wevisone
  - text-encoder
  - safetensors
  - w4a8
  - int4
  - convrot
  - hswq
  - comfyui
  - vision-language
library_name: comfyui
pipeline_tag: image-to-text
---

# WeVisOne-VL-4B - HSWQ ConvRot W4A8

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

A single-file ConvRot W4A8 pack of the **WeVisOne-VL-4B** text encoder, built with
**HSWQ** (Hybrid-Sensitivity-Weighted-Quantization).

The pack holds the same tensors as the checkpoint it derives from: 326 linear layers stored as
`asym_w4a8_int8` (4-bit weight, group size 16, ConvRot group 256) and 708 tensors kept in their
original dtype. It replaces the BF16 checkpoint at **48.47 %** of its size.

- **Technical Details & Quantization Framework:** [Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **Verification report:** [WeVisOneVL4B_W4A8_Pack.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/backdoor/verdicts/WeVisOneVL4B_W4A8_Pack.md)

## 🟢 Available Files

| Filename | Architecture | Format / Precision | Size | Upstream Base Model | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `wevisone_vl_4b_hswq_convrot_w4a8.safetensors` | WeVisOne-VL 4B (Qwen3-VL + alpha-matte decoder) | HSWQ ConvRot W4A8 | ~4.70 GB | `tencent/WeVisOne-VL-4B` | Available |

## 📐 Quantization Summary

| Item | Value |
| :--- | :--- |
| Source | BF16, 9,699,729,600 B |
| Source sha256 | `063c653ab1169be31adb796804def025a954c6c079b67f88bb1eeae1b94fa9cd` |
| Pack | 4,701,528,058 B |
| Pack sha256 | `435b8819b363f70126cb9e21753cbc69a2e3d9ce0a9c7f8851944ec8ebaf3b64` |
| W4A8 layers | 326 |
| Kept layers | 708 tensors (32 by drift rank, plus `embed_tokens`, `lm_head`, `mask_decoder`) |
| Weight reproduction | median relative L2 5.335e-03 against the source weight |
| Output acceptance | 256 items: cosine mean 0.995055, min 0.987840 |

## 💻 Usage

Place the pack in `ComfyUI/models/clip/` and load it with the standard ComfyUI `CLIPLoader`.
The W4A8 layout is applied by ComfyUI itself; no additional loader is required.

## 🔒 Security

This pack carries no tensor that is absent from its source and drops none; its 708 kept tensors
are identical to the source tensors element by element, and its 326 W4A8 layers reproduce the
source weights to a median relative L2 of 5.335e-03. No backdoor was found in the pack.

The verdict covers the pack as a transformation of the source checkpoint. Whether the upstream
release itself carries a trained trigger is outside the reach of a pack audit.

## 🤝 Credits & Acknowledgments

- Upstream model: `tencent/WeVisOne-VL-4B`
