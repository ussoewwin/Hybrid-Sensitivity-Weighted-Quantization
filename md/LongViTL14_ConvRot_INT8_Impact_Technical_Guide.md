# Long-ViT-L-14-BEST-GmP-smooth-ft-full ConvRot INT8 — Technical Guide

**Target**: `Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors` — an OpenAI CLIP
ViT-L/14 checkpoint with an extended text context, stored in FP32, carrying **two
towers** (`text_model.*`, `vision_model.*`).

**What this document covers**: how it is quantized to ConvRot INT8 with a **measured**
FP16 protection set covering **both towers**, the pack layout, the load paths, and the
measured fidelity / size / latency.

- Source: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors`
  (1631.80 MB, SHA256 `4D3E6C8C21881F13E532C33A809538B51132BAE9FF8A40147FBDD7D57DC8C13E`)
- Pack: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors`
  (413.20 MB, SHA256 `EF3E0A30DC1C41BAC4C5F35047944E86EE6CAB9D4CD63D4F3BB25C8FCFAC0E20`)

---

## 0. The checkpoint

| item | value |
| :--- | :--- |
| architecture | OpenAI CLIP ViT-L/14, long-text variant |
| 2D weight modules | 221 (text 74 + vision 145 + 2 misc) |
| text context | position embedding **248 × 768** (vs 77 in the standard ViT-L/14) |
| input widths | 1024 (122), 768 (63), 4096 (24), 3072 (12) |
| dtypes | FP32 |
| projection layers | `text_projection.weight` and `vision_model.embeddings.class_embedding` are **present** |

Because both projection tensors exist, no completion step is required when loading
either tower.

---

## 1. Method: impact-driven, per tower

For each candidate 2D layer, the layer's weight is replaced by its ConvRot INT8
reconstruction (Hadamard rotation along the input axis + per-out-channel INT8), the
tower is run on a fixed input, and the relative MSE of the tower's **own output**
against the pristine run is recorded. The layer is then restored.

| tower | observable | script | layers measured |
| :--- | :--- | :--- | :--- |
| text | text embedding (`encode_from_tokens_scheduled`) | `sdxl/diag_impact_clip.py` | 73 |
| vision | image embedding (`encode_image`) | `sdxl/diag_impact_clip_vision.py` | 144 |

### 1.1 Measured drift distribution

| tower | layers | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| text | 73 | **2.2013e-01** | 1.4511e+00 | **1.9328** |
| vision | 144 | **4.9761e-02** | 2.8525e-01 | **8.5660e-01** |

The profile matches the other ViT-L/14 checkpoints: a handful of text layers dominate
the drift while the vision tower is far less sensitive.

---

## 2. Threshold selection (measured on the acceptance set)

The acceptance set is fixed and deterministic (no network):
- text: 256 prompts composed from a fixed 34-word vocabulary with `random.Random(42)`
- vision: 256 synthetic 224×224 images from `torch.Generator().manual_seed(42)`

Acceptance: **mean >= 0.99 AND min >= 0.98** for both towers.

| thresholds (text / vision) | INT8 | FP16 | size | text min | vision min | verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.95 / 0.90 (shipped)** | **220** | **1** | **413.2 MB** | **0.999687** | **0.999210** | **PASS** |

Both thresholds sit just above the measured maxima of their tower (text 1.9328,
vision 0.8566), i.e. the maximum quantization available. The acceptance minimum still
clears 0.98 with a wide margin, so protecting more layers would only add size back
without a measured benefit.

The single FP16-retained 2D module is `vision_model.embeddings.position_embedding`
(257 rows — not a multiple of 4, so it cannot use the INT8 GEMM path).

---

## 3. Pack layout

```
text_model.*.weight          int8 (ConvRot)
text_model.*.weight_scale    float32  [out, 1]
text_model.*.comfy_quant     uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
vision_model.*.weight        int8 (ConvRot)
vision_model.*.weight_scale  float32  [out, 1]
vision_model.*.comfy_quant   uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata       JSON {"format_version":"1.0","layers":{...}}
```

| | value |
| :--- | :--- |
| ConvRot INT8 layers | **220** (text 74 + vision 144 + 2 misc) |
| FP16 retained (2D) | 1 |
| `comfy_quant` tensors | 220 |
| group size | 256 (fixed) |

---

## 4. Measured results

### 4.1 Embedding fidelity (256 items, seed 42)

| tower | mean | min | p5 | verdict |
| :--- | :--- | :--- | :--- | :--- |
| text | **0.999809** | **0.999687** | 0.999738 | **PASS** |
| vision | **0.999772** | **0.999210** | 0.999699 | **PASS** |

### 4.2 Size / latency (same machine, same batch)

| metric | FP32 | INT8 | change |
| :--- | :--- | :--- | :--- |
| file size | 1631.80 MB | **413.20 MB** | **-74.7 %** |
| text encode latency | 15.94 ms | 31.38 ms | +96.9 % |
| vision encode latency | 38.50 ms | 55.02 ms | +42.9 % |

**Honest note on latency.** The file shrinks by ~75 %, but encode latency roughly
doubles: the `mixed_precision_ops` path adds per-op dequant/rotation overhead that is
not amortized at these batch sizes. The benefit here is **VRAM / file size**, not
speed.

---

## 5. Loading the pack

### 5.1 Text tower (standard loader)

```python
sd, meta = comfy.utils.load_torch_file(pack, safe_load=True, return_metadata=True)
mo = {"quantization_metadata": json.loads(meta["_quantization_metadata"])}
clip = comfy.sd.load_clip(ckpt_paths=[pack], embedding_directory=None,
                          clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=mo)
```

`text_projection.weight` is present in this checkpoint, so no completion is needed.

### 5.2 Vision tower (needs `mixed_precision_ops`)

The stock `CLIPVisionLoader` builds the tower with `manual_cast`, which does **not**
consume `comfy_quant`. Construct it with `mixed_precision_ops`:

```python
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS
ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                            torch.float16, full_precision_mm=True)
model = IMAGE_ENCODERS["clip_vision_model"](config, dtype, device, ops)
model.load_state_dict({k: v for k, v in sd.items() if k.startswith("vision_model.")},
                      strict=False)
```

`class_embedding` is present in this checkpoint, so no completion is needed.

---

## 6. Reproduce

```bash
# 1) measure text drift (73 layers)
python sdxl/diag_impact_clip.py --model Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors \
  --out impact/impact_long_vitl14_text.json --comfy_path <ComfyUI> --full
# 2) measure vision drift (144 layers)
python sdxl/diag_impact_clip_vision.py --model Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors \
  --out impact/impact_long_vitl14_vision.json --comfy_path <ComfyUI> --full
# 3) build (thresholds text=1.95 vision=0.90)
python sdxl/build_long_vitl14_convrot_int8.py --thr_t 1.95 --thr_v 0.90 --out <pack>
# 4) verify (256 items each, seed 42)
python sdxl/verify_long_vitl14.py --pack <pack> --n 256 --seed 42 \
  --out impact/verify_long_vitl14.json
```

Expected: text mean >= 0.9997 / min >= 0.9996; vision mean >= 0.9997 / min >= 0.9991.

---

## 7. Summary

| Aspect | Choice |
| :--- | :--- |
| Target | Long-ViT-L-14-BEST-GmP-smooth-ft-full (FP32, text + vision, 248-token text context) |
| Quantized | **both towers**: text 74 + vision 144 = 220 INT8 |
| Protection criterion | Measured drift of each tower's own output |
| Thresholds | text **1.95**, vision **0.90** (both above the measured maxima) |
| Acceptance fidelity | text mean 0.999809 / min 0.999687; vision mean 0.999772 / min 0.999210 |
| Size | 1631.80 MB -> **413.20 MB** (-74.7 %) |
| Latency | text +96.9 %, vision +42.9 % — reported, not hidden |
| Load | text: standard `CLIPLoader` + metadata; vision: `mixed_precision_ops` required |
