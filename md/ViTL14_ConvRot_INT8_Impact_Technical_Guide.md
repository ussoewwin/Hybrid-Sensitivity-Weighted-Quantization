# ViT-L-14-BEST-smooth-GmP ConvRot INT8 — Technical Guide

**Target**: `ViT-L-14-BEST-smooth-GmP-HF-format.safetensors` — the OpenAI CLIP
ViT-L/14 checkpoint in Hugging-Face layout, carrying **two towers** (`text_model.*`
and `vision_model.*`).

**What this document covers**: how the checkpoint is quantized to ConvRot INT8 with a
**measured** FP16 protection set covering **both towers**, how the pack is laid out,
how it loads, and the measured fidelity / size / latency of the result.

- Source: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\ViT-L-14-BEST-smooth-GmP-HF-format.safetensors`
  (888.30 MB, SHA256 `46A201AFAE2516C95B6D0CC798ACA637161F07B8D0D7DC7776FB6F691F5476F1`)
- Pack: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors`
  (570.27 MB, SHA256 `40AF368223C63F489DC155904150F5FB1DDD0872758AC5D71A2D047C14218D3F`)

---

## 0. The checkpoint: two towers, two load paths

| tower | 2D weight modules | ComfyUI loader | operations used |
| :--- | :--- | :--- | :--- |
| `text_model.*` | 74 | `CLIPLoader` -> `SD1ClipModel` | `mixed_precision_ops` when `quantization_metadata` is supplied, else `manual_cast` |
| `vision_model.*` | 145 | `CLIPVisionLoader` -> `ClipVisionModel` | **always `manual_cast`** (hard-coded in `ClipVisionModel.__init__`) |

This asymmetry is the single most important fact in this document. `comfy_quant` is
consumed **only** by the `MixedPrecisionOp` family in `comfy/ops.py`; `manual_cast`
(`disable_weight_init`) ignores it. Therefore:

- The **text** tower quantizes whenever `model_options["quantization_metadata"]` is
  present, because `SDClipModel.__init__` then builds `mixed_precision_ops`.
- The **vision** tower quantizes only if the tower is constructed with
  `mixed_precision_ops` explicitly. The stock `CLIPVisionLoader` does not do this.

**Consequence.** Both towers *can* be quantized, but a quantized vision tower needs
`mixed_precision_ops` at construction. Building it with the stock `manual_cast` reads
the int8 weights as raw int8 and the output collapses (measured vision cos `-0.0117`).
With `mixed_precision_ops` the same pack gives vision cos **0.998939** (and
**0.999451** mean / **0.996486** min on the 256-image acceptance set).

> An earlier revision of this work concluded "the vision tower cannot be quantized".
> That was wrong: the limitation was the loader's operation choice, not the pack.

---

## 1. Method: impact-driven, per tower

Static weight statistics (kurtosis, outlier ratio) are **not** used as the protection
criterion. Instead, for each candidate 2D layer the tower's **own output drift** is
measured:

1. Run the pristine tower on a fixed input; keep the output.
2. For one layer at a time, replace its weight with the ConvRot INT8 reconstruction
   (Hadamard rotation along the input axis + per-out-channel INT8).
3. Re-run the tower; record the relative MSE of the output against the pristine run.
4. Restore the layer. Repeat for every layer.

Layers whose drift exceeds the threshold stay FP16; the rest become ConvRot INT8.

| tower | observable | script | layers measured |
| :--- | :--- | :--- | :--- |
| text | text embedding (`encode_from_tokens_scheduled`) | `sdxl/diag_impact_clip.py` | 73 |
| vision | image embedding (`encode_image`) | `sdxl/diag_impact_clip_vision.py` | 144 |

### 1.1 Measured drift distribution

| tower | layers | min | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| text | 73 | 0.0 | 1.526e-01 | 1.35e+00 | **1.9276** |
| vision | 144 | — | 2.258e-01 | 9.83e-01 | **1.2505** |

Text's most fragile layers:

| drift | layer |
| :--- | :--- |
| 1.9276 | `text_model.encoder.layers.0.mlp.fc2` |
| 1.8766 | `text_model.encoder.layers.0.mlp.fc1` |
| 1.5789 | `text_model.encoder.layers.2.mlp.fc1` |

---

## 2. Threshold selection (measured, not assumed)

The threshold decides how many layers stay FP16. It is chosen so that the
**acceptance set's minimum cosine stays above 0.98**, at the highest quantization
that still satisfies it.

### 2.1 Text tower

| threshold | INT8 | text cos (256) | min |
| :--- | :--- | :--- | :--- |
| **1.30 (shipped)** | **65** | mean 0.999914 | 0.999851 |
| 1.90 | 67 | mean 0.999859 | — |

Text 1.30 sits just below the two max-drift layers (1.9276, 1.8766), keeping them
FP16.

### 2.2 Vision tower (measured on the acceptance set, 256 images)

| threshold | INT8 | min | mean | min >= 0.98 |
| :--- | :--- | :--- | :--- | :--- |
| 0.80 | 116 | 0.9747 | — | FAIL |
| **0.50 (shipped)** | **99** | **0.996486** | 0.999451 | **PASS** |
| 0.30 | 84 | 0.997061 | 0.999628 | PASS |
| 0.15 | 52 | 0.999386 | 0.999915 | PASS |
| 0.00 | 0 | 1.000000 | 1.000000 | (no quantization) |

**Vision 0.50 is the optimum**: the highest quantization whose acceptance-set minimum
still clears 0.98 with margin. 0.80 fails (min 0.9747); lowering to 0.30/0.15 protects
more layers for no acceptance benefit.

> Note: an intermediate sweep on a 64-image subset suggested 0.80 was safe. The full
> 256-image acceptance set showed 0.80 fails. The shipped threshold is therefore set
> from the acceptance set itself, not a subset.

---

## 3. Pack layout

```
text_model.*.weight            int8 (ConvRot) or fp16 if protected
text_model.*.weight_scale      float32  [out, 1]
text_model.*.comfy_quant       uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
vision_model.*.weight          int8 (ConvRot) or fp16 if protected
vision_model.*.weight_scale    float32  [out, 1]
vision_model.*.comfy_quant     uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata         JSON {"format_version":"1.0","layers":{...}}
```

| | value |
| :--- | :--- |
| ConvRot INT8 layers | **164** (text 65 + vision 99) |
| FP16 retained (2D) | 55 |
| `comfy_quant` tensors | 164 |
| group size | 256 (fixed) |

The FP16-retained 2D layers are those whose measured drift exceeded the threshold,
plus any layer that cannot use the INT8 GEMM path (input width not a multiple of 256,
or output rows not a multiple of 4 — e.g. `vision_model.embeddings.position_embedding`,
257 rows).

---

## 4. Measured results

### 4.1 Embedding fidelity (acceptance set, 256 fixed items, seed 42)

| tower | n | mean | min | p5 | p50 | verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| text | 256 | **0.999914** | **0.999851** | 0.999890 | 0.999916 | **PASS** |
| vision | 256 | **0.999451** | **0.996486** | 0.998791 | 0.999554 | **PASS** |

Acceptance: mean >= 0.99 AND min >= 0.98 for both towers. Both pass.

The acceptance set is generated deterministically (no network):
- text: 256 prompts composed from a fixed 34-word vocabulary with `random.Random(42)`
- vision: 256 synthetic 224×224 images from `torch.Generator().manual_seed(42)`

### 4.2 Size / weight memory / latency (same machine, same batch)

| metric | FP16 | INT8 | change |
| :--- | :--- | :--- | :--- |
| file size | 888.30 MB | **570.27 MB** | **-35.8 %** |
| text tower weight bytes | 235.65 MB | 133.52 MB | -43.3 % |
| vision tower weight bytes | 577.75 MB | 434.15 MB | -24.9 % |
| text encode latency | 62.99 ms | 78.72 ms | +25.0 % |
| vision encode latency | 73.19 ms | 83.70 ms | +14.4 % |

**Honest note on latency.** The file and weight memory shrink substantially, but
per-encode latency is **higher** on this machine: the `mixed_precision_ops` path adds
per-op dequant/rotation overhead that dominates a model of this size. The benefit
here is **VRAM / file size**, not speed. This is reported as measured, not claimed as
an improvement.

### 4.3 Downstream smoke test

| check | result |
| :--- | :--- |
| text encode | shape (1, 77, 768), finite, norm 248.9973 |
| vision encode | shape (3, 1024), finite, norms 31.19 / 31.60 / 31.38 |
| FP16-vs-INT8 consistency | text **0.999921**, vision **0.999275** |

---

## 5. Loading the pack

### 5.1 Text tower (standard loader works)

`CLIPLoader` -> `comfy.sd.load_clip` -> `SDClipModel`. To activate INT8, the pack's
`_quantization_metadata` must reach `model_options["quantization_metadata"]`; then
`SDClipModel` builds `mixed_precision_ops` and consumes the `comfy_quant` stamps.

```python
sd, meta = comfy.utils.load_torch_file(pack, safe_load=True, return_metadata=True)
mo = {"quantization_metadata": json.loads(meta["_quantization_metadata"])}
clip = comfy.sd.load_clip(ckpt_paths=[pack], embedding_directory=None,
                          clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=mo)
```

`text_projection` is absent in this checkpoint; under mixed ops its weight stays
`None` and the forward crashes (`'NoneType' object has no attribute 'device'`).
Complete it with fp16 zeros (768×768).

### 5.2 Vision tower (needs `mixed_precision_ops`)

The stock `CLIPVisionLoader` uses `manual_cast` and will **not** quantize. Construct
the tower with `mixed_precision_ops`:

```python
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS
ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                            torch.float16, full_precision_mm=True)
model = IMAGE_ENCODERS["clip_vision_model"](config, dtype, device, ops)
```

`vision_model.embeddings.class_embedding` is absent in the checkpoint; complete it
with zeros (dim 1024) when loading the vision tower standalone.

---

## 6. Reproduce

```bash
# 1) measure text drift (73 layers)
python sdxl/diag_impact_clip.py --model <vitl>.safetensors \
  --out impact/impact_vitl14_best.json --comfy_path <ComfyUI> --full
# 2) measure vision drift (144 layers)
python sdxl/diag_impact_clip_vision.py --model <vitl>.safetensors \
  --out impact/impact_vitl14_best_vision.json --comfy_path <ComfyUI> --full
# 3) build the pack (thresholds text=1.30 vision=0.50)
python sdxl/build_vitl14_convrot_int8.py
# 4) verify (256 items each, seed 42)
python sdxl/verify_vitl14_text.py   --n 256 --seed 42 --out impact/verify_vitl14_text.json
python sdxl/verify_vitl14_vision.py --n 256 --seed 42 --out impact/verify_vitl14_vision.json
# 5) downstream smoke
python sdxl/smoke_vitl14.py
```

Expected: text mean >= 0.9999 / min >= 0.9998; vision mean >= 0.9990 / min >= 0.9960.

---

## 7. Summary

| Aspect | Choice |
| :--- | :--- |
| Target | OpenAI CLIP ViT-L/14 (HF layout, text + vision) |
| Quantized | **both towers**: text 65 INT8, vision 99 INT8 (164 total) |
| Protection criterion | Measured drift of each tower's own output |
| Thresholds | text **1.30**, vision **0.50** (chosen from the acceptance set) |
| Acceptance fidelity | text mean 0.999914 / min 0.999851; vision mean 0.999451 / min 0.996486 |
| Size | 888.30 MB -> **570.27 MB** (-35.8 %) |
| Latency | higher (reported, not hidden) |
| Load | text: standard `CLIPLoader` + metadata; vision: `mixed_precision_ops` required |

**Related documents**: `md/ViTL14_ConvRot_INT8_Verification_Report.md` (verification),
`md/ViTL14_ConvRot_INT8_Runbook.md` (reproduction / rollback),
`impact/ledger_vitl14.json` (experiment ledger).
