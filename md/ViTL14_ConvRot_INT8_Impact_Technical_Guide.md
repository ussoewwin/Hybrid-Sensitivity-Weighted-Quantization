# ViT-L-14-BEST-smooth-GmP ConvRot INT8 - Impact-Driven Quantization (Technical Guide)

How the **OpenAI CLIP ViT-L/14** checkpoint
(`ViT-L-14-BEST-smooth-GmP-HF-format.safetensors`, HF layout, text + vision) is
quantized to ConvRot INT8 with a **measured** FP16 protection set, and how it loads
through the **standard ComfyUI path**.

---

## 0. Scope and the text/vision split

This checkpoint carries **two towers**:

| side | 2D weights | loaded by | ComfyUI path |
| :--- | :--- | :--- | :--- |
| `text_model.*` | 74 | `CLIPLoader` (SD1ClipModel) | standard |
| `vision_model.*` | 145 | `CLIPVisionLoader` | standard |

**Only the text tower is quantized.** The reason is measured, not assumed: the
standard `CLIPVisionLoader` does **not** consume the `comfy_quant` metadata (the same
structural fact established for the standalone FLUX CLIP), so an int8 vision tower
would be read as raw int8 and destroyed. Measured: quantizing the vision tower gives
`vision cos = -0.0117`; keeping it FP16 gives `vision cos = 1.000000`.

So the shipped pack is: **text tower ConvRot INT8, vision tower FP16**.

---

## 1. The new method (impact-driven)

Every eligible Linear layer of each tower is measured separately:

- **Text**: the observable is the text-embedding output (`encode_from_tokens_scheduled`).
- **Vision**: the observable is the image embedding (`encode_image`).

For each candidate layer, its ConvRot INT8 reconstruction is injected, the tower
forward is run on a fixed input, and the relative MSE against the pristine run is
recorded. Layers above the threshold stay FP16.

- Text measurement: `sdxl/diag_impact_clip.py` -> `impact/impact_vitl14_best.json` (73 layers)
- Vision measurement: `sdxl/diag_impact_clip_vision.py` -> `impact/impact_vitl14_best_vision.json` (144 layers)

### 1.1 Measured drift distribution

| tower | layers | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| text | 73 | 1.53e-01 | 1.35e+00 | **1.9276** |
| vision | 144 | 2.26e-01 | 9.83e-01 | 1.2505 |

Text heaviest: `layers.0.mlp.fc2` (1.9276), `layers.0.mlp.fc1` (1.8766),
`layers.2.mlp.fc1` (1.5789).

---

## 2. Threshold sweep (text tower, measured)

| Threshold | INT8 (text) | FP16 (text) | text cos |
| :--- | :--- | :--- | :--- |
| 0.15 | 36 | 38 | 0.999955 |
| 0.50 | 57 | 17 | 0.999930 |
| 1.00 | 60 | 14 | 0.999924 |
| 1.20 | 65 | 9 | 0.999922 |
| **1.30 (shipped)** | **65** | **9** | **0.999907** |
| 1.50 | 67 | 7 | 0.999882 |

**Why 1.30.** It sits inside the measured text distribution, just below the two
most fragile layers (`layers.0.mlp.fc2` 1.9276, `layers.0.mlp.fc1` 1.8766), keeping
them FP16. It is the highest-quality-per-byte point for the text tower.

---

## 3. The pack

```
text_model.*.weight           int8  (ConvRot)
text_model.*.weight_scale     float32  ([out,1])
text_model.*.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
vision_model.*                FP16 (unchanged)
_quantization_metadata        JSON  {"format_version":"1.0","layers":{...}}
```

| | value |
| :--- | :--- |
| text ConvRot INT8 | **65** |
| text FP16 retained | 9 |
| vision | FP16 (unchanged) |
| size | 888.3 MB -> **713.8 MB** |

---

## 4. Measured results (standard loaders)

| side | loader | absmax FP16 | absmax INT8 | cos vs FP16 |
| :--- | :--- | :--- | :--- | :--- |
| text | `CLIPLoader` (SD1ClipModel) | 33.0481 | 33.0479 | **0.999907** |
| vision | `CLIPVisionLoader` | 8.6714 | 8.6714 | **1.000000** |

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

The output loads in the standard `CLIPLoader` (text) and `CLIPVisionLoader` (vision).

---

## 6. Summary

| Aspect | Choice |
| :--- | :--- |
| Target | OpenAI CLIP ViT-L/14 (HF layout, text + vision) |
| Quantized side | **text tower only** (vision is FP16; standard CLIPVisionLoader does not consume comfy_quant) |
| Protection criterion | Measured drift of each tower's own output |
| Threshold | **1.3** (text tower) |
| Quantized | 65 ConvRot INT8 (text), 9 FP16, vision FP16 |
| Measured cos vs FP16 | text **0.999907**, vision **1.000000** |
| Size | 888.3 MB -> 713.8 MB |
