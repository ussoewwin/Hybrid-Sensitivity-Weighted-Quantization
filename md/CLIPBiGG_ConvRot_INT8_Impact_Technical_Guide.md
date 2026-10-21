# CLIP-ViT-bigG-14 — ConvRot INT8 Trajectory-Impact Technical Guide

**Model.** `models/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` (3519.0 MB, FP16, 1,844,907,521 parameters)
**Pack.** `models/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` (1935.7 MB, **-45.0 %**)
**SHA256.** `6C8BF27AE72415B5CCC2ECF51A21D5654D4C39342AB5D57436CAE822B9121EC8`
**Method.** impact-driven (measured drift of the model's own output), **group size 64**.

---

## 1. What this model is

CLIP-ViT-bigG-14 is a **vision-only** checkpoint: 776 `vision_model.*` tensors plus
`visual_projection.weight`. There is no `text_model.*` at all, so the text tower
is not part of this work.

| item | measured |
| :--- | :--- |
| tensors | 777 |
| parameters | 1,844,907,521 |
| dtypes | F16 (the 257 I64 tensors are `position_ids`) |
| hidden size | 1664 |
| transformer layers | 48 |
| attention heads | 16 |
| FF size | 8192 |
| patch size | 14, image size 224 |
| projection dim | 1280 |

The runtime is built by `comfy.clip_vision.load_clipvision_from_sd`, which picks
`clip_vision_config_g.json` from the checkpoint's own shape (48 layers ->
`layers.47.layer_norm1.weight`). The `CLIPVisionLoader` uses the same path.

---

## 2. Runtime model inventory (measured)

| item | measured |
| :--- | :--- |
| runtime `Linear` modules | **288** (48 layers x 6) |
| per layer | `self_attn.q_proj / k_proj / v_proj / out_proj` (1664x1664), `mlp.fc1` (8192x1664), `mlp.fc2` (1664x8192) |
| non-Linear 2D weight | `vision_model.embeddings.position_embedding` (257 x 1664) - an Embedding, kept FP16 |
| runtime names vs checkpoint names | the runtime drops the `vision_model.` prefix |

---

## 3. The group size is 64, not 256

The ConvRot rotation works group-wise along the input width, and the group size
must be a power of 4 that divides the input width. bigG's widths are:

| layer | in_features | largest power-of-4 divisor <= 256 |
| :--- | :--- | :--- |
| `self_attn.*` | 1664 | **64** (1664 = 64 x 26) |
| `mlp.fc1` | 1664 | **64** |
| `mlp.fc2` | 8192 | 256 (8192 = 256 x 32) |

1664 is not a multiple of 256, so **240 of the 288 layers cannot be rotated with
a group size of 256** - with that group size only the 48 `mlp.fc2` layers would
be quantized, which is what a first run of the builder produced (47 layers, the
48th excluded by the `of % 4` rule). The builder now derives the group size
per layer: the largest power-of-4 divisor of that layer's input width that is
<= the requested maximum. The pack therefore uses group size 64 for 275 layers
and 256 is never needed for bigG.

---

## 4. Impact measurement

Script: `sdxl/diag_impact_clip_vision.py` -> `impact/impact_bigg_vision.json`

The observable is the image-embedding output for a fixed synthetic image batch.
Each candidate layer's ConvRot INT8 reconstruction is injected and the relative
MSE against the pristine run is recorded.

| item | value |
| :--- | :--- |
| layers measured | **288 / 288** |
| runs | 1 (single completion) |
| wall time | 86 s |
| interruptions | 0 |
| reference output | finite |

**Drift distribution**

| min | p10 | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| 1.44e-06 | 1.74e-03 | 2.65e-02 | 1.93e-01 | 1.15e+00 |

**By layer kind**

| kind | n | min | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| attn.q | 48 | 1.44e-06 | 1.10e-02 | 3.00e-02 | 1.35e-01 |
| attn.k | 48 | 1.88e-05 | 1.37e-02 | 4.54e-02 | 1.12e-01 |
| attn.v | 48 | 1.52e-05 | 2.65e-02 | 5.51e-02 | 1.24e-01 |
| attn.o | 48 | 1.44e-05 | 2.22e-02 | 5.66e-02 | 1.34e-01 |
| mlp.fc1 | 48 | 1.18e-02 | **2.01e-01** | 8.95e-01 | **1.15e+00** |
| mlp.fc2 | 48 | 1.17e-03 | 4.66e-02 | 1.78e-01 | 3.06e-01 |

**`mlp.fc1` dominates** - its median (0.20) is seven times the attention
layers' median, and the twelve largest drifts are all `mlp.fc1`. The outliers
also cluster in the deep layers: layers 40-44 hold drifts of 0.70-0.92, while
layer 9's `mlp.fc1` is the single worst at 1.15.

---

## 5. Threshold selection (measured on the acceptance set)

Four candidate thresholds were built and each was measured on the full 256-item
acceptance set (seed 42) against the FP16 baseline, with the image-embedding
cosine as the observable.

| threshold | INT8 layers | cos mean | cos min | cos p5 | pack |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.05 | 204 | 0.999978 | 0.999964 | 0.999972 | 2691.7 MB |
| 0.20 | 260 | 0.999926 | 0.999893 | 0.999909 | 2130.3 MB |
| **0.30** | **275** | **0.999916** | **0.999880** | **0.999900** | **1935.7 MB** |
| 0.50 | 277 | 0.999916 | 0.999874 | 0.999900 | 1909.8 MB |

**Threshold 0.30 was adopted.** All four pass the gate (mean >= 0.99, min >=
0.98). 0.30 and 0.50 measure the same cosine to six decimals, but 0.30 keeps
two more layers in FP16 for 26 MB - the margin is taken on the side that
protects the model.

---

## 6. The pack

Built with `clip_convert/convert_clip_impact_convrot_int8.py`:

```
--clip_type vision --drift-threshold 0.30 --groupsize 256
```

(The builder derives the per-layer group size: 64 for the 1664-wide layers.)

| item | value |
| :--- | :--- |
| file size | 3519.0 MB -> **1935.7 MB** (-45.0 %) |
| quantized layers | 275 (of 288) |
| FP16 protected | 13 (drift > 0.30) |
| group size | 64 (all 275 layers) |
| kept as-is | the Embedding, the 1D weights, the layer norms |

---

## 7. Verification (real load path)

Script: `sdxl/verify_t5.py`'s vision counterpart; the pack is loaded through
`comfy.clip_vision` with `mixed_precision_ops`, the path the `CLIPVisionLoader`
uses.

| item | value |
| :--- | :--- |
| output finite | **yes** |
| cos mean (256 items, seed 42) | **0.999916** |
| cos min | **0.999880** |
| cos p5 | 0.999900 |
| gate (mean >= 0.99, min >= 0.98) | **PASS** |
| QuantizedTensor Linear layers at runtime | **275** |
| `int8_linear` calls per encode | **275** |
| encode latency | 92.12 ms / image (batch 1) |

**The INT8 GEMM is reached.** On the stock path the `int8_tensorwise` algorithm
is weight-only (`quantize_input: False`) and `int8_linear` is never called; the
loader package's runtime arm
(`ComfyUI-HSWQ-Loader-and-Tools/patches/clip_int8_gemm/arm.py`) routes the
`TensorWiseINT8Layout` weights into the real kernel, with the online ConvRot
activation rotation. **ComfyUI core is not modified.**

---

## 8. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_clip_vision.py` | vision impact measurement (the image-embedding output) |
| `clip_convert/convert_clip_impact_convrot_int8.py` | impact-driven pack builder (`--clip_type vision`, `--groupsize`) |
| `impact/impact_bigg_vision.json` | the measured per-layer drift (git-ignored) |
| `impact/_t5_packs/bigg_threshold_sweep.json` | the threshold sweep (git-ignored) |
| `models/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` | the pack |
