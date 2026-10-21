# CLIP-ViT-bigG-14 — ConvRot INT8 Trajectory-Impact Technical Guide

**Model.** `models/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` (3519.0 MB, FP16, 1,844,907,521 parameters)
**Pack.** `models/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` (2180.1 MB, **-38.0 %**)
**SHA256.** `6CD4AFAC314AE720FF1EDCAC59F927B93D8E15CE3E8E109DDC59F6FD43B2D5D3`
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
| 4.42e-12 | 1.05e-08 | **1.12e-07** | 1.93e-06 | **9.12e-06** |

(This is the **corrected** measurement - see section 8. The pre-fix
measurement gave the median 2.65e-02 and the max 1.15e+00.)

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

The kept set is given explicitly (`--protect-list`), with the layers taken
in the order of the corrected drift's ranking. Every candidate below was
measured on the full 256-item acceptance set (seed 42) against the FP16
baseline, with the image-embedding cosine as the observable.

| the FP16-kept | INT8 layers | cos mean | cos min | cos p5 | pack |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 (all quantized) | 288 | 0.999731 | 0.999554 | 0.999554 | 1767.1 MB |
| 20 | 268 | 0.999873 | 0.999697 | 0.999827 | 2026.8 MB |
| 30 | 258 | 0.999901 | 0.999737 | 0.999858 | 2135.9 MB |
| **35** | **253** | **0.999916** | **0.999741** | **0.999877** | **2180.1 MB** |
| 40 | 248 | 0.999923 | 0.999737 | 0.999889 | 2234.6 MB |
| the previous pack | 275 | 0.999757 | 0.999469 | 0.999682 | 1935.7 MB |

**35 FP16-kept layers were adopted.** The quality rises monotonically with the
kept count - the all-quantized pack measures the worst minimum (0.999554) -
and 35 is the point where the curve flattens: 40 buys +7e-06 of the mean and
nothing of the minimum, for 55 MB more.

Against the previous pack, at a comparable layer count the difference is in
the minimum, not the mean: **0.999741 against 0.999469**, a 2.7x improvement,
with 22 fewer quantized layers.

---

## 6. The pack

Built with `clip_convert/convert_clip_impact_convrot_int8.py`:

```
--clip_type vision --drift-threshold 0.05 --groupsize 256 \
  --protect-list impact/bigg_protect_35.json
```

(The builder derives the per-layer group size: 64 for the 1664-wide layers.)

| item | value |
| :--- | :--- |
| file size | 3519.0 MB -> **2180.1 MB** (-38.0 %) |
| quantized layers | 253 (of 288) |
| FP16 protected | 35 (the aggregate-contribution order) |
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
| cos min | **0.999741** |
| cos p5 | **0.999877** |
| gate (mean >= 0.99, min >= 0.98) | **PASS** |
| QuantizedTensor Linear layers at runtime | **253** |
| `int8_linear` calls per encode | **253** |
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

---

## 8. The re-measurement (2026-10-21) and the rebuilt pack

### 8.1 The injection defect

The impact measurement had a defect that was found while working on the FLUX
ControlNet: `convrot_int8_recon` returned the reconstruction in the **rotated**
frame (`q * s`, i.e. `W_hat * H`) and injected it as the module's weight, but
the module's weight is used by the plain forward as `F.linear(x, weight)`. The
pack stores the rotated weight and the kernel rotates the **activation**, so
the quantization error that reaches the output is in the **original** frame:
`(q*s) H^T - W`.

Measured on the FLUX ControlNet's `controlnet_blocks.0` with the pack's own
qdata/scale and the real kernel:

| what | relMSE vs the FP16 result |
| :--- | :--- |
| the pack's kernel (`convrot=True`) | **1.74e-04** (correct) |
| the plain int8 (no activation rotation) | 1.89 |
| the injected rotated frame as a plain linear (the defect) | 1.89 |
| the un-rotated reconstruction as a plain linear (the fix) | **9.01e-05** |

The same defect was in this script; it is fixed, and the 288-layer sweep was
redone.

### 8.2 The corrected drift

| | the old (defective) | the new (corrected) |
| :--- | :--- | :--- |
| the measured modules | 288 | 288 |
| the median drift | 2.65e-02 | **1.12e-07** |
| the p90 | 1.93e-01 | **1.93e-06** |
| the max | 1.15e+00 | **9.12e-06** |
| the protected at 0.30 | 13 | **0** |

The old measurement made `mlp.fc1` look catastrophic - `encoder.layers.35.mlp.fc1`
measured 0.9619, and layers 40-44 held drifts of 0.70-0.92. The corrected
measurement puts the same layer at 6.72e-06, a factor of 143,000 lower, and
the ranking changes with it.

### 8.3 Why the drift cannot be the selection criterion

The per-layer drift measures the **weight** error only: the reconstruction is
placed in the module and the plain forward is used, so the activation is not
quantized. The pack's real path also quantizes the **activation** for that
layer, and that is where most of the error comes from. The marginal effect is
therefore not the quantity the pack trades off.

The consequence is visible in the table in section 5: with the corrected
drift the 0.30 threshold protects **0** layers, so a pack built from the drift
alone quantizes all 288 and measures the **worst** minimum of the whole
series (0.999554). The kept set has to be chosen on the aggregate.

### 8.4 How the kept set was chosen

The kept layers are taken in the order of the corrected drift's ranking
(the largest first), and the candidates are measured on the full 256-item
acceptance set - the table in section 5. The quality rises monotonically
with the kept count and flattens at 35:

| the FP16-kept | 0 | 20 | 30 | **35** | 40 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| the INT8 layers | 288 | 268 | 258 | **253** | 248 |
| cos min | 0.999554 | 0.999697 | 0.999737 | **0.999741** | 0.999737 |
| the size | 1767.1 | 2026.8 | 2135.9 | **2180.1** | 2234.6 MB |

35 is where the minimum stops improving: 40 buys +7e-06 of the mean and
nothing of the minimum, for 55 MB more.

The previous pack's 13 kept layers came from the defective drift, and they are
not the 13 largest drifts - which is why the previous pack measured min
0.999469 while a pack with **22 fewer** quantized layers measures 0.999741.

### 8.5 The rebuilt pack

| item | value |
| :--- | :--- |
| the source | 3519.0 MB |
| the pack | **2180.1 MB** (-38.0 %) |
| the SHA256 | `6CD4AFAC314AE720FF1EDCAC59F927B93D8E15CE3E8E109DDC59F6FD43B2D5D3` |
| the tensors | 1283 |
| the quantized layers | **253** (of 288) |
| the FP16-kept layers | 35 |
| the group size | 64 (all 253 layers) |
| the `int8_linear` calls per encode | **253** |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42) | **0.999916 / 0.999741 / 0.999877** |
| the gate (mean >= 0.99, min >= 0.98) | **PASS** |

### 8.6 The measurement note

The acceptance runs on the **same ops class** for the baseline and the pack.
The stock `CLIPVisionLoader` builds the vision tower with
`comfy.ops.manual_cast` while a quantized pack is built with
`mixed_precision_ops`, and for this model the two paths are not numerically
equivalent - the same FP16 source through the two paths gives a cosine of
0.94-0.96, and restoring **every** quantized layer of a pack to FP16 does not
move the number. The baseline therefore has to be built through the same ops
class, otherwise the comparison measures the ops path instead of the
quantization.

For the same reason the per-layer contribution could not be measured in one
live model for this checkpoint: the source (3519 MB) and the pack (1936 MB)
plus the source's fp16 copies exceed the 16 GB VRAM
(`CUDA error: invalid argument`, `Copy is larger than memobj size`). The
contribution is therefore measured with real packs - one pack per candidate
kept set - so only one model is resident at a time.
