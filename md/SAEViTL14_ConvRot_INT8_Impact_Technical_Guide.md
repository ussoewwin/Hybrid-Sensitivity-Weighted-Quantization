# CLIP-SAE-ViT-L-14 ConvRot INT8 — Technical Guide

**Target**: `CLIP-SAE-ViT-L-14-FP32.safetensors` — an OpenAI CLIP ViT-L/14 checkpoint
carrying **two towers** (`text_model.*`, `vision_model.*`), stored in FP32.

**What this document covers**: how the checkpoint is quantized to ConvRot INT8 with a
**measured** FP16 protection set covering **both towers**, the pack layout, the load
paths, and the measured fidelity / size / latency.

- Source: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\CLIP-SAE-ViT-L-14-FP32.safetensors`
  (1631.30 MB, SHA256 `820AFA79F4BA17399E740B4B5630AB74BC3A196E471D7321123EFA8356B62F79`)
- Pack: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\CLIP-SAE-ViT-L-14-hswq_convrot_int8_protected.safetensors`
  (413.24 MB, SHA256 `64CB47FF0C31A9FE82D06F337143E421B8C1097B7C14CA9E7124C4956E6FA64D`)

---

## 0. Why this pack was rebuilt

The previous `CLIP-SAE-ViT-L-14-hswq_convrot_int8_protected.safetensors` (dated
2026-10-19, 985.4 MB) was produced by the **old method**: its protection set came from
static weight statistics (kurtosis / outlier ratio), and there was **no impact JSON**
for it — i.e. its layers were never measured against the model's own output.

Two consequences:

1. Static weight statistics are not a proxy for output sensitivity.
2. The kurtosis values used were computed through a broken safetensors header offset
   (`f.seek(offs[0])` instead of `8 + header_len + offs[0]`), so the protection basis
   was invalid.

This document describes the **rebuild from the FP32 source** with the impact-driven
method, both towers measured.

---

## 1. The checkpoint: two towers, two load paths

| tower | 2D weight modules | ComfyUI loader | operations used |
| :--- | :--- | :--- | :--- |
| `text_model.*` | 74 | `CLIPLoader` -> `SD1ClipModel` | `mixed_precision_ops` when `quantization_metadata` is supplied, else `manual_cast` |
| `vision_model.*` | 145 | `CLIPVisionLoader` -> `ClipVisionModel` | **always `manual_cast`** (hard-coded) |
| `logit_scale`, `text_projection` | 2 | — | not quantized |

`comfy_quant` is consumed **only** by the `MixedPrecisionOp` family in `comfy/ops.py`;
`manual_cast` ignores it. Therefore:

- **Text** quantizes when `model_options["quantization_metadata"]` is present, because
  `SDClipModel.__init__` then builds `mixed_precision_ops`.
- **Vision** quantizes only when the tower is constructed with `mixed_precision_ops`.
  The stock `CLIPVisionLoader` does not do this; building it with `manual_cast` reads
  int8 weights as raw int8 and the output collapses.

Unlike the other ViT-L/14 checkpoint, **this source carries both `text_projection.weight`
and `vision_model.embeddings.class_embedding`**, so no completion step is required.

---

## 2. Method: impact-driven, per tower

For each candidate 2D layer, the tower is run with that one layer replaced by its
ConvRot INT8 reconstruction (Hadamard rotation along the input axis + per-out-channel
INT8), and the relative MSE of the tower's **own output** against the pristine run is
recorded. The layer is then restored.

| tower | observable | script | layers measured |
| :--- | :--- | :--- | :--- |
| text | text embedding (`encode_from_tokens_scheduled`) | `sdxl/diag_impact_clip.py` | 73 |
| vision | image embedding (`encode_image`) | `sdxl/diag_impact_clip_vision.py` | 144 |

### 2.1 Measured drift distribution

| tower | layers | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| text | 73 | **1.4819e-01** | 1.3529e+00 | **1.9112** |
| vision | 144 | **7.2335e-02** | 3.7557e-01 | **0.8939** |

Note the contrast with the other ViT-L/14 checkpoint: this vision tower is markedly
less sensitive (median 0.072 vs 0.226; max 0.894 vs 1.251), which is why it tolerates
near-full quantization.

---

## 3. Threshold selection (measured on the acceptance set)

The acceptance set is fixed and deterministic (no network):
- text: 256 prompts composed from a fixed 34-word vocabulary with `random.Random(42)`
- vision: 256 synthetic 224×224 images from `torch.Generator().manual_seed(42)`

Acceptance: **mean >= 0.99 AND min >= 0.98** for both towers.

Three thresholds were built and measured end to end:

| thresholds (text / vision) | INT8 | FP16 | size | text min | vision min | verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1.30 / 0.50 | 203 | 18 | 552.9 MB | 0.999816 | 0.999359 | PASS |
| 1.80 / 0.85 | 216 | 5 | 438.7 MB | 0.999780 | 0.999361 | PASS |
| **1.95 / 0.90 (shipped)** | **219** | **2** | **413.2 MB** | **0.999755** | **0.999353** | **PASS** |

**Shipped choice: text 1.95 / vision 0.90** — above both towers' measured maxima
(text max 1.9112, vision max 0.8939), i.e. the maximum quantization available. Every
qualifying 2D layer is quantized, and the acceptance minimum still clears 0.98 with a
wide margin (0.9998 / 0.9994). Lower thresholds protect more layers without any
acceptance benefit, so the maximum-compression point is the optimum here.

---

## 4. Pack layout

```
text_model.*.weight            int8 (ConvRot)
text_model.*.weight_scale      float32  [out, 1]
text_model.*.comfy_quant       uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
vision_model.*.weight          int8 (ConvRot)
vision_model.*.weight_scale    float32  [out, 1]
vision_model.*.comfy_quant     uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata         JSON {"format_version":"1.0","layers":{...}}
```

| | value |
| :--- | :--- |
| ConvRot INT8 layers | **219** (text 73 + vision 144 + 2 misc) |
| FP16 retained (2D) | 2 |
| `comfy_quant` tensors | 219 |
| group size | 256 (fixed) |

The 2 FP16-retained 2D layers are those that cannot use the INT8 GEMM path
(input width not a multiple of 256, or output rows not a multiple of 4 — e.g.
`vision_model.embeddings.position_embedding`, 257 rows).

---

## 5. Measured results

### 5.1 Embedding fidelity (acceptance set, 256 items, seed 42)

| tower | n | mean | min | p5 | p50 | verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| text | 256 | **0.999872** | **0.999755** | — | — | **PASS** |
| vision | 256 | **0.999676** | **0.999353** | — | — | **PASS** |

### 5.2 Size / weight memory / latency (same machine, same batch)

| metric | FP32 | INT8 | change |
| :--- | :--- | :--- | :--- |
| file size | 1631.30 MB | **413.24 MB** | **-74.7 %** |
| text tower weight bytes | 235.65 MB | 118.41 MB | -49.8 % |
| vision tower weight bytes | 577.75 MB | 290.60 MB | -49.7 % |
| text encode latency | 55.58 ms | 90.66 ms | +63.1 % |
| vision encode latency | 83.42 ms | 104.43 ms | +25.2 % |

**Honest note on latency.** File and weight memory shrink by ~75% / ~50%, but
per-encode latency is **higher**: the `mixed_precision_ops` path adds per-op
dequant/rotation overhead. The benefit here is **VRAM / file size**, not speed.

### 5.3 Downstream smoke test

| check | result |
| :--- | :--- |
| text encode | shape (1, 77, 768), finite, norm 252.2140 |
| vision encode | shape (3, 1024), finite, norms 32.40 / 32.33 / 32.28 |
| FP16-vs-INT8 consistency | text **0.999883**, vision **0.999692** |

---

## 6. Loading the pack

### 6.1 Text tower

```python
sd, meta = comfy.utils.load_torch_file(pack, safe_load=True, return_metadata=True)
mo = {"quantization_metadata": json.loads(meta["_quantization_metadata"])}
clip = comfy.sd.load_clip(ckpt_paths=[pack], embedding_directory=None,
                          clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=mo)
```

This checkpoint carries `text_projection.weight`, so no completion is needed.

### 6.2 Vision tower (needs `mixed_precision_ops`)

```python
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS
ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                            torch.float16, full_precision_mm=True)
model = IMAGE_ENCODERS["clip_vision_model"](config, dtype, device, ops)
model.load_state_dict({k: v for k, v in sd.items() if k.startswith("vision_model.")},
                      strict=False)
```

This checkpoint carries `class_embedding`, so no completion is needed.

---

## 7. Reproduce

```bash
# 1) measure text drift (73 layers)
python sdxl/diag_impact_clip.py --model CLIP-SAE-ViT-L-14-FP32.safetensors \
  --out impact/impact_sae_vitl14_text.json --comfy_path <ComfyUI> --full
# 2) measure vision drift (144 layers)
python sdxl/diag_impact_clip_vision.py --model CLIP-SAE-ViT-L-14-FP32.safetensors \
  --out impact/impact_sae_vitl14_vision.json --comfy_path <ComfyUI> --full
# 3) build (thresholds text=1.95 vision=0.90)
python sdxl/build_sae_vitl14_convrot_int8.py --thr_t 1.95 --thr_v 0.90 --out <pack>
# 4) verify (256 items each, seed 42)
python sdxl/verify_sae_vitl14.py --pack <pack> --n 256 --seed 42 --out impact/verify_sae_vitl14.json
# 5) downstream smoke
python sdxl/smoke_sae_vitl14.py
```

Expected: text mean >= 0.9998 / min >= 0.9997; vision mean >= 0.9996 / min >= 0.9993.

---

## 8. Summary

| Aspect | Choice |
| :--- | :--- |
| Target | CLIP-SAE ViT-L/14 (FP32 source, text + vision) |
| Previous pack | old method (static kurtosis, no impact measurement) — rebuilt |
| Quantized | **both towers**: text 73 + vision 144 = 219 INT8 (of 221 2D) |
| Protection criterion | Measured drift of each tower's own output |
| Thresholds | text **1.95**, vision **0.90** (maximum quantization; both above measured maxima) |
| Acceptance fidelity | text mean 0.999872 / min 0.999755; vision mean 0.999676 / min 0.999353 |
| Size | 1631.30 MB -> **413.24 MB** (-74.7 %) |
| Latency | higher (reported, not hidden) |
| Load | text: standard `CLIPLoader` + metadata; vision: `mixed_precision_ops` required |
