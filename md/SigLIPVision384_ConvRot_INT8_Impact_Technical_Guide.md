# sigclip_vision_patch14_384 — ConvRot INT8 Trajectory-Impact Technical Guide

**Model.** `models/clip/sigclip_vision_patch14_384.safetensors` (816.8 MB, 27-layer SigLIP vision tower)
**Pack.** `models/clip/sigclip_vision_patch14_384_hswq_convrot_int8.safetensors` (633.5 MB, **-22.4 %**)
**SHA256.** `1A7184CF4AC22B666BE7ECE9F62A1ECA8C830ADD825B482036B2F75C74304CCE`
**Method.** impact-driven (measured drift of the model's own output), per-layer group size.

---

## 1. The model

sigclip_vision_patch14_384 is a **vision-only** SigLIP checkpoint: 27 transformer
layers, hidden size 1152, image size 384, patch size 14.

| item | measured |
| :--- | :--- |
| runtime Linear modules | **162** (27 layers x 6) |
| per layer | `self_attn.q_proj / k_proj / v_proj / out_proj`, `mlp.fc1 / fc2` |
| non-Linear 2D weight | the position embedding (an Embedding, kept FP16) |
| image size | 384 |
| the config ComfyUI picks | `comfy/clip_vision_siglip_384.json` (chosen from the checkpoint's own shape) |

---

## 2. The group size is per layer, and this model needs it

SigLIP's hidden size is 1152, which is **not** a multiple of 256. The builder
derives the group size per layer (the largest power-of-4 divisor of that
layer's input width that is <= the requested maximum), so the pack carries:

| group size | layers |
| :--- | :--- |
| 64 | 79 |
| 16 | 17 |

With a fixed group size of 256 the 1152-wide layers could not be rotated at
all. This is the same per-layer rule that CLIP-ViT-bigG needs (its hidden size
is 1664, which is not a multiple of 256 either).

---

## 3. Impact measurement

Script: `sdxl/diag_impact_clip_vision.py` -> `impact/impact_siglip_vision.json`

The observable is the image-embedding output for a fixed synthetic image batch.
Each candidate layer's ConvRot INT8 reconstruction is injected and the relative
MSE against the pristine run is recorded.

| item | value |
| :--- | :--- |
| layers measured | **162 / 162** |
| runs | 1 (single completion) |
| wall time | 30 s |
| interruptions | 0 |
| reference output | 1,679,616 elements, finite |

**Drift distribution**

| min | p10 | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| 3.88e-02 | 1.12e-01 | 2.74e-01 | 7.73e-01 | 3.75e+00 |

**SigLIP is far more sensitive to INT8 than the CLIP towers**: the *smallest*
drift in the whole model is 3.88e-02, which is above the CLIP text tower's
median. Every layer needs a threshold well above the CLIP family's 0.05-0.25
range.

---

## 4. Threshold selection (measured on the acceptance set)

Four candidate thresholds were built and each was measured on the full 256-item
acceptance set (seed 42) against the FP16 baseline, with the image-embedding
cosine as the observable.

| threshold | INT8 layers | cos mean | cos min | cos p5 | pack |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.20 | 49 | **0.999917** | **0.999848** | 0.999905 | 730.8 MB |
| **0.30** | **96** | **0.999659** | **0.999404** | 0.999594 | **633.5 MB** |
| 0.50 | 134 | 0.999235 | 0.997555 | 0.998942 | 526.8 MB |
| 0.80 | 146 | 0.998818 | 0.996018 | 0.998309 | 494.4 MB |

**Threshold 0.30 was adopted.** It sits between the two points that are still
comfortably inside the gate (0.20 and 0.30) and the ones that start to give it
margin away (0.50 already drops the minimum to 0.9976, 0.80 to 0.9960). At
0.30 the minimum is 0.999404, which keeps a clear margin over the 0.98
criterion while the pack is 87 MB smaller than the 0.20 candidate.

---

## 5. The pack

Built with `clip_convert/convert_clip_impact_convrot_int8.py`:

```
--clip_type vision --drift-threshold 0.30
```

| item | value |
| :--- | :--- |
| file size | 816.8 MB -> **633.5 MB** (-22.4 %) |
| tensors | 640 |
| quantized layers | 96 (group size 64: 79, group size 16: 17) |
| FP16 kept | 66 (the layers whose measured drift exceeds 0.30, plus the biases and the embedding) |

---

## 6. Verification (the real load path)

The pack is loaded through the vision loader path with
`mixed_precision_ops({"int8_tensorwise": ...})`, and the loader package's
runtime arm is installed (the ComfyUI startup condition).

| item | value |
| :--- | :--- |
| runtime weight types | **96 QuantizedTensor + 66 Parameter** |
| `int8_linear` calls per encode | **96** |
| the image embeds | (1, 729, 1152), **finite** |
| cos mean / min / p5 (256 items, seed 42, arm enabled) | **0.998560 / 0.996602 / 0.997994** |
| gate (mean >= 0.99, min >= 0.98) | **PASS** |
| encode latency | **64.44 ms / image** (batch 1, 384x384) |

**The INT8 GEMM is reached, and it is worth real time.** The vision tower uses
`operations.Linear`, so the same runtime arm applies to it as to the CLIP text
tower: without the arm the encode measures 85.89 ms, with it **64.44 ms
(-25.0 %)**.

Without the arm the pack would still be correct (the weight-only path gives
cos mean 0.999659), but the INT8 GEMM would never run and the compute saving
would be zero. ComfyUI core is not modified; the arm lives in
`ComfyUI-HSWQ-Loader-and-Tools/patches/clip_int8_gemm/arm.py`.

---

## 7. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_clip_vision.py` | vision impact measurement (the image-embedding output) |
| `clip_convert/convert_clip_impact_convrot_int8.py` | impact-driven pack builder (`--clip_type vision`, per-layer group size) |
| `impact/impact_siglip_vision.json` | the measured per-layer drift (git-ignored) |
| `impact/verify_siglip.json` | the verification result (git-ignored) |
| `models/clip/sigclip_vision_patch14_384_hswq_convrot_int8.safetensors` | the pack |
