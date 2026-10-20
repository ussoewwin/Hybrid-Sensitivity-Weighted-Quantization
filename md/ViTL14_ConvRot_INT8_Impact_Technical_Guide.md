# ViT-L-14-BEST-smooth-GmP ConvRot INT8 - Impact-Driven Quantization (Technical Guide)

How the **OpenAI CLIP ViT-L/14** checkpoint
(`ViT-L-14-BEST-smooth-GmP-HF-format.safetensors`, HF layout, text + vision) is
quantized to ConvRot INT8 with a **measured** FP16 protection set, covering **both
towers**, and how it loads through the **standard ComfyUI path**.

---

## 0. Scope: both towers

This checkpoint carries two towers:

| side | 2D weights | loaded by |
| :--- | :--- | :--- |
| `text_model.*` | 74 | `CLIPLoader` (SD1ClipModel) |
| `vision_model.*` | 145 | `CLIPVisionLoader` |

**Both towers are quantized.** An earlier revision claimed the vision tower could not
be quantized; that was wrong. The reason it first appeared broken is a loading detail,
not a limit:

- The standard `ClipVisionModel.__init__` builds the tower with `comfy.ops.manual_cast`,
  which does **not** consume `comfy_quant`. Loading the int8 vision weights that way
  reads them as raw int8 and the output collapses.
- Building the tower with `comfy.ops.mixed_precision_ops` (the same operations the
  INT8 path uses elsewhere) **does** consume `comfy_quant`, and the quantized vision
  tower then runs correctly.

Measured: quantized vision via `mixed_precision_ops` gives **vision cos 0.998939**.
So the vision tower is quantized, not skipped.

---

## 1. The method (impact-driven, per tower)

Each tower is measured separately against its own output:

- **Text**: the text-embedding output (`encode_from_tokens_scheduled`).
- **Vision**: the image embedding (`encode_image`).

For each candidate layer, its ConvRot INT8 reconstruction is injected, the tower
forward is run on a fixed input, and the relative MSE against the pristine run is
recorded. Layers above the threshold stay FP16.

- Text: `sdxl/diag_impact_clip.py` -> `impact/impact_vitl14_best.json` (73 layers)
- Vision: `sdxl/diag_impact_clip_vision.py` -> `impact/impact_vitl14_best_vision.json` (144 layers)

### 1.1 Measured drift distribution

| tower | layers | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| text | 73 | 1.53e-01 | 1.35e+00 | **1.9276** |
| vision | 144 | 2.26e-01 | 9.83e-01 | **1.2505** |

Text heaviest: `layers.0.mlp.fc2` (1.9276), `layers.0.mlp.fc1` (1.8766).
Vision heaviest cluster near 1.25.

---

## 2. Thresholds (measured)

| tower | threshold | rationale |
| :--- | :--- | :--- |
| text | **1.30** | just below the two max-drift layers (1.9276, 1.8766); keeps them FP16 |
| vision | **1.25** | at the vision max (1.2505); keeps the single most fragile vision layer FP16 |

---

## 3. The pack

```
text_model.*.weight           int8 (ConvRot) or fp16 if protected
vision_model.*.weight         int8 (ConvRot) or fp16 if protected
<layer>.weight_scale          float32 ([out,1])
<layer>.comfy_quant           uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata        JSON {"format_version":"1.0","layers":{...}}
```

| | value |
| :--- | :--- |
| ConvRot INT8 | **208** |
| FP16 retained | 11 |
| size | 888.3 MB -> **430.7 MB** (-51.5 %) |

---

## 4. Measured results

| tower | load path | cos vs FP16 | finite |
| :--- | :--- | :--- | :--- |
| text | `CLIPLoader` (SD1ClipModel) + quantization_metadata | **0.999907** | yes |
| vision | `ClipVisionModel` with `mixed_precision_ops` | **0.998939** | yes |

---

## 5. Reproduce

```bash
# text drift
python sdxl/diag_impact_clip.py --model <vitl>.safetensors \
  --out impact/impact_vitl14_best.json --comfy_path <ComfyUI> --full
# vision drift
python sdxl/diag_impact_clip_vision.py --model <vitl>.safetensors \
  --out impact/impact_vitl14_best_vision.json --comfy_path <ComfyUI> --full
```

---

## 6. Summary

| Aspect | Choice |
| :--- | :--- |
| Target | OpenAI CLIP ViT-L/14 (HF layout, text + vision) |
| Quantized | **both towers** (text 65 INT8, vision 143 INT8) |
| Protection criterion | Measured drift of each tower's own output |
| Thresholds | text 1.30, vision 1.25 |
| Measured cos vs FP16 | text **0.999907**, vision **0.998939** |
| Size | 888.3 MB -> **430.7 MB** (-51.5 %) |
