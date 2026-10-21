# CLIP-ViT-H-14 — ConvRot INT8 Trajectory-Impact Technical Guide

**Model.** `models/clip_vision/CLIP-ViT-H-14.safetensors` (2411.2 MB, FP32, 632,077,057 parameters)
**Pack.** `models/clip_vision/CLIP-ViT-H-14_hswq_convrot_int8.safetensors` (949.5 MB, **-60.6 %**)
**SHA256.** `52B6DB65B2D413897B8BB78133BCD8B2679EDCEE91E91AB8576FB46885B3F7D3`
**Method.** impact-driven (measured drift of the model's own output), ConvRot group size 256.

---

## 1. The model

CLIP-ViT-H-14 is a **vision-only** CLIP checkpoint: 32 transformer layers, hidden
size 1280, image size 224, patch size 14.

| item | measured |
| :--- | :--- |
| checkpoint keys | 521 |
| parameters | 632,077,057 |
| dtypes | F32 (the 257 I64 tensors are `position_ids`) |
| runtime Linear modules | **192** (32 layers x 6) |
| per layer | `self_attn.q_proj / k_proj / v_proj / out_proj` (1280x1280), `mlp.fc1` (5120x1280), `mlp.fc2` (1280x5120) |
| the config ComfyUI picks | `comfy/clip_vision_config_h.json` (from `layers.30.layer_norm1.weight`) |

---

## 2. Impact measurement

Script: `sdxl/diag_impact_clip_vision.py` -> `impact/impact_vit_h14_vision.json`

The observable is the image-embedding output for a fixed synthetic image batch.
Each candidate layer's ConvRot INT8 reconstruction is injected and the relative
MSE against the pristine run is recorded.

| item | value |
| :--- | :--- |
| layers measured | **192 / 192** |
| runs | 1 (single completion) |
| wall time | 19 s |
| interruptions | 0 |
| reference output | finite |

**Drift distribution**

| min | p10 | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| 2.20e-03 | 1.36e-02 | 3.34e-02 | 2.42e-01 | 7.97e-01 |

---

## 3. Threshold selection

| threshold | INT8 layers | cos mean | cos min | pack |
| :--- | :--- | :--- | :--- | :--- |
| **0.05** | **127** | **0.999919** | **0.999866** | **949.5 MB** |
| 0.20 | 170 | (the pack is not smaller than the 0.05 one in quality) | | |

**Threshold 0.05 was adopted**: 127 of the 192 layers are quantized, the five
sensitive ones stay in FP16, and the measured output cosine is 0.999919.

The pack is 949.5 MB against the 2411.2 MB source (**-60.6 %**).

---

## 4. A measurement trap that had to be cleared first

The first pack built for this model measured a cosine of **0.945** and no
threshold changed it. The cause was not the quantization:

- The stock `CLIPVisionLoader` builds the vision tower with
  `comfy.ops.manual_cast`, while a quantized pack is built with
  `mixed_precision_ops`. **Those two paths are not numerically equivalent for
  this model**: the same FP16 source through the two paths gives a cosine of
  0.94-0.96.
- Restoring **every** quantized layer of the pack back to FP16 left the cosine
  at 0.9588 - the loss was not in the quantized layers at all.
- The pack itself was correct throughout: the stored int8 payload and the
  scale are byte-identical to a fresh reconstruction, all 127 loaded modules
  hold the pack's qdata and scale, and the kernel reproduces the FP16 result
  for the pack's tensors with a relative MSE of 6.9e-06.

So the baseline has to be built through the **same ops class**, otherwise the
comparison measures the ops path instead of the quantization. With the baseline
on the same path the pack measures 0.999919 / 0.999866.

The same correction applies to the other vision packs, which were re-measured
on the same-path baseline:

| pack | cos mean | cos min | size |
| :--- | :--- | :--- | :--- |
| CLIP-ViT-H-14 | 0.999919 | 0.999866 | 949.5 MB |
| CLIP-ViT-bigG-14 | 0.999755 | 0.999596 | 1935.7 MB |
| sigclip_vision_patch14_384 | 0.998563 | 0.997798 | 633.5 MB |

---

## 5. Verification

| item | value |
| :--- | :--- |
| loaded | `CLIPVisionModel` through `mixed_precision_ops` |
| the quantized layers at runtime | 127 (all `QuantizedTensor`, group size 256) |
| the module qdata / scale vs the pack | identical (127 / 127) |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42, the same ops path) | **0.999919 / 0.999866 / 0.999899** |
| gate (mean >= 0.99, min >= 0.98) | **PASS** |

---

## 6. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_clip_vision.py` | vision impact measurement (the image-embedding output) |
| `clip_convert/convert_clip_impact_convrot_int8.py` | impact-driven pack builder (`--clip_type vision`, per-layer group size, the runtime names) |
| `sdxl/verify_vision.py` | the acceptance check with the baseline on the same ops path |
| `impact/impact_vit_h14_vision.json` | the measured per-layer drift (git-ignored) |
| `models/clip_vision/CLIP-ViT-H-14_hswq_convrot_int8.safetensors` | the pack |
