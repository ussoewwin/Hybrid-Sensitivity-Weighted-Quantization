# ConvRot INT8 — Text Encoder / CLIP Quantization Index

Every ConvRot INT8 pack built for a **text encoder or CLIP** checkpoint, which
method produced it, and where the details live.

---

## 1. Two methods

| method | protection set | status |
| :--- | :--- | :--- |
| **impact-driven** (current) | measured drift of the model's **own output**, per layer | current standard |
| **static kurtosis** (old) | kurtosis / outlier-ratio of the weights | superseded - the kurtosis values were computed through a broken safetensors header offset (`f.seek(offs[0])` instead of `8 + header_len + offs[0]`), so the protection basis was invalid |

Any pack produced by the old method is being rebuilt with the impact-driven method.
The measurement script, the builder and the verification script are shipped per model
so the result is reproducible.

---

## 2. Current state per checkpoint

| checkpoint | towers quantized | thresholds | cos (mean / min) | size | guide |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `clip_l.safetensors` (FLUX.1 CLIP-L) | text | 1.8 | 0.999857 / 0.999851 | 122.6 MB | `md/CLIPL_ConvRot_INT8_Impact_Technical_Guide.md` |
| `ViT-L-14-BEST-smooth-GmP-HF-format.safetensors` | text + vision | 1.30 / 0.50 | text 0.999914 / 0.999851, vision 0.999451 / 0.996486 | 570.3 MB | `md/ViTL14_ConvRot_INT8_Impact_Technical_Guide.md` |
| `CLIP-SAE-ViT-L-14-FP32.safetensors` | text + vision | 1.95 / 0.90 | text 0.999872 / 0.999755, vision 0.999676 / 0.999353 | 413.2 MB | `md/SAEViTL14_ConvRot_INT8_Impact_Technical_Guide.md` |
| `Josiefied-Qwen3-0.6B-abliterated-v1.safetensors` | text | full (all 197) | 0.997572 / 0.992079 | 570.5 MB | `md/JosiefiedQwen3_06B_ConvRot_INT8_Impact_Technical_Guide.md` |

Acceptance criterion used throughout: **cos mean >= 0.99 AND cos min >= 0.98**, on a
fixed 256-item set (seed 42).

---

## 3. Why the thresholds differ per model

The threshold is not a global constant - it is chosen from the measured drift
distribution of that specific checkpoint, on that model's own acceptance set.

| model family | drift profile | consequence |
| :--- | :--- | :--- |
| CLIP-L / ViT-L / SAE ViT-L | a few layers dominate (e.g. ViT-L text: max 1.93 vs median 0.15) | protect the outliers, quantize the rest |
| Josiefied-Qwen3-0.6B | **flat** (min 0.75, median 3.05, max 3.43; IQR 2.94-3.17) | no layer is safer than another, so **selective protection cannot help** - quantize everything |

A pack built for one checkpoint must not be reused for another: the protection set is
model-specific by construction.

---

## 4. Load paths (measured)

| tower | loader | how INT8 gets enabled |
| :--- | --- | --- |
| CLIP text (SD1ClipModel) | `CLIPLoader` / `DualCLIPLoader` | `model_options["quantization_metadata"]` -> `SDClipModel` builds `mixed_precision_ops` |
| llama-family text (Qwen3 / T5) | `CLIPLoader` / `DualCLIPLoader` | automatic: `llama_detect` -> `detect_layer_quantization` -> `llama_quantization_metadata` |
| CLIP vision | `CLIPVisionLoader` | the stock loader uses `manual_cast`, which does **not** consume `comfy_quant`; the tower must be built with `mixed_precision_ops` |
| SDXL-embedded CLIP | `CheckpointLoader` | the SDXL-only loader arm (`sdxl_clip_quant.py`) |

---

## 5. Related records (different scope, still valid)

| document | scope |
| --- | --- |
| `md/SDXL_ConvRot_INT8_CLIP_Technical_Record_2026-10-15.md` | the **SDXL-checkpoint-embedded** CLIP (rule-based, SDXL only) - the loader arm lives there |
| `md/SD1.5_ConvRot_INT8_CLIP_Technical_Record_2026-10-15.md` | the SD1.5 text-encoder path |
| `md/ControlNet_ConvRot_INT8_Trajectory_Impact_Technical_Guide.md` | ControlNet packs (impact-driven) |
| `md/SAM31_ConvRot_INT8_Trajectory_Impact_Technical_Guide.md` | SAM 3.1 pack (impact-driven) |

---

## 6. Tooling

| script | purpose |
| --- | --- |
| `sdxl/diag_impact_clip.py` | text-tower drift (CLIP family) |
| `sdxl/diag_impact_clip_vision.py` | vision-tower drift (CLIP family) |
| `sdxl/diag_impact_qwen3.py` | llama-family text-encoder drift |
| `clip_convert/convert_clip_impact_convrot_int8.py` | CLIP converter (impact-driven) |
| `sdxl/verify_*.py`, `sdxl/smoke_*.py`, `sdxl/perf_*.py` | per-model verification / smoke / performance |
