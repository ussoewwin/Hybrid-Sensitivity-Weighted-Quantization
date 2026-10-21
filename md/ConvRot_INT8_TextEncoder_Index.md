# ConvRot INT8 — Text Encoder / CLIP Quantization Index

Every ConvRot INT8 pack built for a **standalone text encoder / CLIP** checkpoint,
which method produced it, and where the details live.

**Scope.** Standalone text-encoder checkpoints only. The CLIP inside an SD1.5 or SDXL
checkpoint is part of the UNet (a single-file, already-finished model) and is
**out of scope here** - it is not touched by this work.

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
| `flan_t5_xxl_fp16.safetensors` (FLAN-T5 XXL encoder) | text | 0.25 (163 of 168 INT8) | 0.999878 / 0.999716 | 5026.1 MB | `md/T5XXL_ConvRot_INT8_Impact_Technical_Guide.md` |

Acceptance criterion used throughout: **cos mean >= 0.99 AND cos min >= 0.98**, on a
fixed 256-item set (seed 42).

**How to read the size column.** The source checkpoints are often FP32 while
ComfyUI holds a text encoder in `float16`, so the file reduction and the runtime
reduction are different numbers. Measured on Long-ViT-L-14 (the source is
**F32 for all 427,747,841 parameters**, 590 tensors):

| | source | pack |
| :--- | :--- | :--- |
| file size | 1631.80 MB | 413.20 MB (-74.7 %) |
| resident weights at runtime | 235.90 MB | 118.48 MB (-49.8 %) |

The file figure is the plain FP32 -> INT8 factor of 1/4 (427,747,841 x 1 byte =
407.9 MB, and the pack measures 413.20 MB with the scales and the one FP16-kept
weight on top). **No tensor is missing**: the pack holds 428,119,001 parameters
against the source's 427,747,841. The runtime figure is smaller because the
source is already held in fp16 in memory and the token embedding (144.8 MB as
FP32) has to stay in FP16. See
`md/LongViTL14_ConvRot_INT8_Impact_Technical_Guide.md` sections 0.1 and 0.2.

**Two model-specific facts, measured.**

- **FLAN-T5 XXL carries a decoder the runtime never loads.** The runtime T5 module
  (`comfy/text_encoders/t5.py`) holds `encoder` and `shared` only; the 339
  `decoder.*` tensors (11.87 GB of the 20.99 GB checkpoint) are never loaded. The pack
  omits them, which is why the size reduction is 76.6 % rather than the 18 % the
  quantized encoder alone gives. The encoder that actually runs is bit-identical.
- **The INT8 GEMM needs the loader's runtime arm.** On the stock ComfyUI path an
  INT8 pack is dequantized for every matmul (`int8_tensorwise` is a weight-only
  algorithm) and `int8_linear` is called 0 times. With the arm installed the real
  INT8 GEMM runs: for FLAN-T5 XXL that is 163 calls per forward and **+75.9 % to
  +86.7 %** against the stock path (3.9x to 7.2x against FP16). See
  `md/T5XXL_ConvRot_INT8_Impact_Technical_Guide.md` section 7.

---

## 3. Why the thresholds differ per model

The threshold is not a global constant - it is chosen from the measured drift
distribution of that specific checkpoint, on that model's own acceptance set.

| model family | drift profile | consequence |
| :--- | :--- | :--- |
| CLIP-L / ViT-L / SAE ViT-L | a few layers dominate (e.g. ViT-L text: max 1.93 vs median 0.15) | protect the outliers, quantize the rest |
| Josiefied-Qwen3-0.6B | **flat** (min 0.75, median 3.05, max 3.43; IQR 2.94-3.17) | no layer is safer than another, so **selective protection cannot help** - quantize everything |
| FLAN-T5 XXL | **rising with depth** (block 0 median 0.12 -> block 23 median 0.24), top drift in the deepest blocks' `attn.v` / `attn.o` / `ff.wi_0` | protect the deep-block outliers, quantize the other 163 of 168 |

A pack built for one checkpoint must not be reused for another: the protection set is
model-specific by construction.

---

## 4. Load paths (measured)

| tower | loader | how INT8 gets enabled |
| :--- | --- | --- |
| CLIP text (SD1ClipModel) | `CLIPLoader` / `DualCLIPLoader` | `model_options["quantization_metadata"]` -> `SDClipModel` builds `mixed_precision_ops` |
| llama-family text (Qwen3) | `CLIPLoader` / `DualCLIPLoader` | automatic: `llama_detect` -> `detect_layer_quantization` -> `llama_quantization_metadata` |
| FLAN-T5 XXL text | `CLIPLoader` / `DualCLIPLoader` | automatic: `t5_xxl_detect` -> `detect_layer_quantization` -> `t5xxl_quantization_metadata` -> `model_options["quantization_metadata"]` |
| CLIP vision | `CLIPVisionLoader` | the stock loader uses `manual_cast`, which does **not** consume `comfy_quant`; the tower must be built with `mixed_precision_ops` |

---

## 5. Tooling

| script | purpose |
| --- | --- |
| `sdxl/diag_impact_clip.py` | text-tower drift (CLIP family) |
| `sdxl/diag_impact_clip_vision.py` | vision-tower drift (CLIP family) |
| `sdxl/diag_impact_qwen3.py` | llama-family text-encoder drift |
| `sdxl/diag_impact_t5.py` | FLAN-T5 XXL encoder drift (encoder hidden state) |
| `clip_convert/convert_t5_impact_convrot_int8.py` | FLAN-T5 XXL pack builder (impact-driven) |
| `sdxl/verify_t5.py` | FLAN-T5 XXL acceptance verification + latency |
| `clip_convert/convert_clip_impact_convrot_int8.py` | CLIP converter (impact-driven) |
| `sdxl/verify_*.py`, `sdxl/smoke_*.py`, `sdxl/perf_*.py` | per-model verification / smoke / performance |
