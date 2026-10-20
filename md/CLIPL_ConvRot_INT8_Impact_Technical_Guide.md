# CLIP-L ConvRot INT8 - Impact-Driven Quantization (Technical Guide)

This document describes how a **standalone CLIP-L text encoder** (`clip_l.safetensors`)
is quantized to ConvRot INT8 with a **measured** FP16 protection set, and how it is
made to load and run as true INT8 inside ComfyUI.

It is the CLIP counterpart of the ControlNet / SAM guides: the protection set comes
from measuring the model's **own output drift**, not from static weight statistics.

---

## 1. What changed and why (the new method)

### 1.1 The old method and its invalidity

The earlier `clip_l_hswq_convrot_int8.safetensors` was produced by
`clip_convert/convert_clip_l_convrot_int8.py`, which chose the protected layers from
**static kurtosis / outlier-ratio** measured on the weights. Two problems:

1. Static weight statistics are not a proxy for output sensitivity.
2. The kurtosis numbers themselves were computed through a **broken safetensors
   header offset** (`f.seek(offs[0])` instead of `8 + header_len + offs[0]`), so the
   measured values were garbage. The old protection set therefore had **no valid basis**.

### 1.2 The new method

For each candidate Linear layer, its ConvRot INT8 reconstruction is injected, the CLIP
forward is run on a fixed prompt set, and the **relative MSE of the text-embedding
output** against the pristine run is recorded. Layers above the threshold stay FP16.

This is the same "perturb one layer, measure the real output drift" principle used for
ControlNet (latent trajectory) and SAM (mask logits + detector outputs); only the
observable differs (a text embedding).

- Measurement script: `sdxl/diag_impact_clip.py`
- Result: `impact/impact_clip_l.json` (73 Linear layers measured)

### 1.3 Measured drift distribution (73 layers)

| Statistic | Value |
| :--- | :--- |
| layers measured | 73 |
| min | 0.0 |
| median | **1.58e-01** |
| p90 | 1.29e+00 |
| max | **1.89e+00** |

The heaviest layers are the early MLP / attention projections
(`encoder.layers.0.mlp.fc2` = 1.89, `layers.0.mlp.fc1` = 1.86, `layers.0.self_attn.v_proj`
= 1.62, `layers.0.self_attn.out_proj` = 1.59). The `text_projection` layer measures
exactly 0 (it is unused for the standard SD1.5 conditioning path).

### 1.4 Threshold

| Threshold | Protected layers |
| :--- | :--- |
| 0.10 | 59 |
| **0.15 (shipped)** | **37** |
| 0.20 | 27 |
| 0.30 | 22 |
| 0.50 | 16 |

---

## 2. Why the pack needed a loader fix (the critical part)

A correct pack alone is **not enough**. ComfyUI builds a text encoder with
`comfy.ops.manual_cast` unless `model_options["quantization_metadata"]` is set
(`SDClipModel.__init__`, `comfy/sd1_clip.py`). The **standalone CLIP load path**
(`comfy.sd.load_clip` -> `load_text_encoder_state_dicts`) never sets it for the
CLIP-L / CLIP-G branches (only MINIMAX and MING do). So an int8 CLIP pack loaded as
**raw fp16** and the conditioning collapsed.

Measured, before the fix (stock `CLIPLoader`):

| | value |
| :--- | :--- |
| Linear with QuantizedTensor | **0 / 146** |
| conditioning cos vs FP16 | **0.029** |

Two fixes, both taken from the SDXL CLIP record
(`md/SDXL_ConvRot_INT8_CLIP_Technical_Record_2026-10-15.md`, failures 1 and 2) and
applied to the standalone loader:

1. **Inject `quantization_metadata`** (from the pack's own `.comfy_quant` tensors)
   into `model_options` so `SDClipModel` builds the Linear layers with
   `mixed_precision_ops`.
2. **Complete `text_projection`** with fp16 zeros. CLIP-L stores no `text_projection`
   in the checkpoint; under `mixed_precision_ops` its weight stays `None` and the
   forward crashes with `'NoneType' object has no attribute 'device'`.

The fix lives in a dedicated module, armed at package import:

- `ComfyUI-HSWQ-Loader-and-Tools/nodes/clip_convrot_int8/clip_quant.py`
  (`arm_clip_quant_metadata()`)
- armed from `ComfyUI-HSWQ-Loader-and-Tools/__init__.py`

It wraps `comfy.sd.load_clip` and **only acts when the loaded file actually carries
CLIP `.comfy_quant` layers**, so every other text-encoder family (SDXL-embedded,
SD3, FLUX, Qwen, T5, ...) is untouched. It is deliberately separate from the
SDXL-checkpoint-embedded CLIP fix (`nodes/sdxl_convrot_int8/sdxl_clip_quant.py`),
which keys on `conditioner.embedders.*`.

---

## 3. The converter

`sdxl/diag_impact_clip.py` measures; `clip_convert/convert_clip_impact_convrot_int8.py`
converts. Rules (the same runtime constraints established for ControlNet and SAM):

1. Only modules the runtime actually has are quantized, and only 2D `nn.Linear`
   layers - ComfyUI's INT8 load path attaches a QuantizedTensor to Linear layers only.
2. The group size is fixed at **256**; a Linear whose input width is not a multiple
   of 256 is kept in original precision (the ConvRot kernel is 2D-only and fused only
   at 256).
3. The pack is written under the checkpoint's own parameter names, which is what the
   CLIP load path expects.

Pack layout:

```
<layer>.weight           int8
<layer>.weight_scale     float32  ([out,1])
<layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata   JSON  {"format_version":"1.0","layers":{...}}
```

---

## 4. Measured results

### 4.1 Embedding fidelity (armed CLIPLoader path, 3 fixed prompts)

| Threshold | ConvRot INT8 | Protected | cos vs FP16 | relMSE | maxabs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.15 (shipped)** | 35 | 37 | **0.999957** | 8.87e-05 | 0.066 |
| 0.20 | 45 | 27 | 0.999939 | 1.25e-04 | 0.068 |

### 4.2 Edge cases (long text, Japanese, symbols, repetition)

| Threshold | finite | absmax | zero fraction |
| :--- | :--- | :--- | :--- |
| 0.15 | True | 33.06 | 0.000 |
| 0.20 | True | 33.06 | 0.000 |

No blow-up on long / multilingual / symbol-heavy inputs.

### 4.3 Size, memory, latency

| | FP16 | INT8 (0.15) |
| :--- | :--- | :--- |
| file size | 234.7 MB | **201.7 MB** (-14.1 %) |
| loaded weight bytes | 235.6 MB | 202.6 MB (-14.0 %) |
| per-encode latency | 45.4 ms | 98.5 ms |

**Honest note on latency.** The INT8 CLIP-L is **slower** per encode on this machine.
CLIP-L is small, and the `mixed_precision_ops` path adds per-op dequant/rotation
overhead that dominates a model this size; the benefit here is **VRAM / file size**,
not speed. This is reported as measured, not as a claimed improvement.

### 4.4 Reproducibility

| | SHA256 |
| :--- | :--- |
| `clip_l.safetensors` (source) | `660C6F5B1ABAE9DC498AC2D21E1347D2ABDB0CF6C0C0C8576CD796491D9A6CDD` |
| `clip_l_hswq_convrot_int8.safetensors` (0.15) | `52B538A24C02605F5763CB6ECE88848A44858482EC8624F2567D71AA91B68D05` |

Re-running the converter with the same inputs and `--drift-threshold 0.15` reproduces
the **identical** SHA256.

---

## 5. Reproduce

```bash
# 1) measure the output drift of every Linear layer (73 layers)
python sdxl/diag_impact_clip.py \
  --model clip_l.safetensors \
  --out impact/impact_clip_l.json \
  --comfy_path <ComfyUI checkout> \
  --full

# 2) convert with the measured protection set
python clip_convert/convert_clip_impact_convrot_int8.py \
  --input clip_l.safetensors \
  --output clip_l_hswq_convrot_int8.safetensors \
  --impact impact/impact_clip_l.json \
  --comfy_path <ComfyUI checkout> \
  --clip_type stable_diffusion \
  --drift-threshold 0.15
```

The loader fix must be present (`nodes/clip_convrot_int8/clip_quant.py` armed from the
package `__init__.py`), otherwise the pack loads as raw fp16.

---

## 6. Summary

| Aspect | Choice |
| :--- | :--- |
| Protection criterion | Measured drift of the CLIP's own text-embedding output |
| Measurement scope | Full sweep of all 73 Linear layers, one run |
| Perturbation | ConvRot INT8 reconstruction (Hadamard rotation + per-channel INT8) |
| Threshold | **0.15** |
| Quantized | 35 ConvRot INT8 Linear (of 73) |
| Loader requirement | `quantization_metadata` injection + `text_projection` completion (standalone CLIP arm) |
| Measured embedding cos vs FP16 | **0.999957** |
| Size | 234.7 MB -> 201.7 MB |
| Latency | slower (reported, not hidden) |
