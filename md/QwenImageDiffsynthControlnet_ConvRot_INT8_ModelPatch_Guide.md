# Qwen-Image diffsynth ControlNet (canny / depth / inpaint) - ConvRot INT8 Model Patch Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors` (2161.8 MB, BF16, 362 tensors)
**Pack.** `models/model_patches/qwen_image_canny_diffsynth_controlnet_convrot_int8.safetensors` (1083.3 MB, **−49.9 %**)
**SHA256.** `C2899255252B81FD64EF2FAE0FBFD86E1496BA04CF10CC95FE45679120F1C996`

The same architecture backs the `depth` and `inpaint` siblings
(`qwen_image_depth_diffsynth_controlnet`, `qwen_image_inpaint_diffsynth_controlnet`,
both 1083.8 MB packs). The method and the layer layout are identical; the
ranking below is this checkpoint's.

---

## Table of contents

1. [What the score is](#1-what-the-score-is)
2. [This is a model patch, and it needs the HSWQ loader](#2-this-is-a-model-patch-and-it-needs-the-hswq-loader)
3. [The model](#3-the-model)
4. [The impact measurement](#4-the-impact-measurement)
5. [The candidates, and the adopted set](#5-the-candidates-and-the-adopted-set)
6. [The pack](#6-the-pack)
7. [The acceptance through the real loader](#7-the-acceptance-through-the-real-loader)
8. [Reproduction](#8-reproduction)
9. [Error sources and limits](#9-error-sources-and-limits)
10. [The experiment ledger](#10-the-experiment-ledger)
11. [Files](#11-files)

---

## 1. What the score is

The score is **not** a static weight statistic (kurtosis, absmax, outlier ratio).
It is the **observed change of the model's own output** when a layer is
perturbed - a behavioural measurement, taken in the frame the runtime uses.

**The metric.** For a fixed set of inputs, the score of a pack is the cosine
similarity between the pack's output and the FP16 source's output. The pack's
output and the source's output are both lists of 60 per-block residuals
(`control_block(img, cond, i)` for i in 0..59). For block `i`:

```
cos_i = <r_source_i, r_pack_i> / (||r_source_i|| * ||r_pack_i||)
1 - cos_i  is that block's per-item error
```

A pack's score over the acceptance set is the **mean over blocks** of
`1 - cos_i`, then averaged over items (the mean of the per-block `1 - cos`);
the reported `cos.mean` / `cos.min` are `1 - (the mean error)` and
`1 - (the worst item's error)`.

**The definition matters, and it is not arbitrary.** Two choices are forced by
the runtime, and both were measured:

- **The observable is a list of 60 residuals, not one concatenated vector.**
  The apply node injects each block's residual into the DiT at that block
  (`img[:, :n] += model.control_block(img[:, :n], cond, i) * strength`), so the
  60 residuals are 60 independent contributions. Pooling them into one
  concatenation lets the largest block's norm decide the whole number and hides
  the rest. Each block is scored on its own.
- **The arithmetic is float64.** The residuals run to ~3e5 in magnitude, so a
  float32 dot product over 47,185,920 elements accumulates ~1e-3 of rounding
  noise. The signal of the single most important layer is 1.12e-05 and the
  smallest is 2.6e-09 - **float32 rounds the entire signal away.** The first
  measurement run in float32 returned every value negative.

**The gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98`
over the fixed 256-item set (seed 42), with a finite output. Measured values are
quoted to 10 decimal places because the differences between candidates are at
1e-07.

---

## 2. This is a model patch, and it needs the HSWQ loader

The file name says "controlnet" but the runtime object is a **model patch**.
ComfyUI dispatches it on the key `controlnet_blocks.0.y_rms.weight` into
`comfy_extras.nodes_model_patch.QwenImageBlockWiseControlNet`: 60 blocks of
`[x_rms, y_rms, input_proj, gelu, output_proj]` plus one shared `img_in`.

**ComfyUI's stock `ModelPatchLoader` does not recognise the INT8 layout.** The
stock loader builds the patch with `comfy.ops.manual_cast`, which reads the
pack's INT8 tensors as plain weights. The pack must be loaded with the
**`HSWQ Model Patch Loader`** (`HSWQModelPatchLoaderCustom`,
`custom_nodes/ComfyUI-HSWQ-Loader-and-Tools`), which detects the
`int8_tensorwise` `comfy_quant` stamps and builds the patch with
`mixed_precision_ops`: the weights stay INT8 in memory and the forward uses the
comfy-kitchen `int8_linear` kernel with online ConvRot rotation.

`load_model_patch(name, cpu_offload)` addresses packs by **name** under
`models/model_patches` and returns a `ModelPatcher`; the module is
`patcher.model`.

---

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 2161.8 MB, BF16, 362 tensors |
| the 2D weights | **121** (all ConvRot-eligible) |
| the layer families | `img_in` + `controlnet_blocks.{0..59}` (`input_proj` / `output_proj`; `x_rms` / `y_rms` are 1D norms) |
| the group size | 256 (all 121) |

Unlike the Z-Image Fun ControlNet patch, **no key conversion happens on load**
(only the Z-Image branch runs `z_image_convert`), so the pack carries the plain
checkpoint names.

---

## 4. The impact measurement

`sdxl/measure_qwen_image_diffsynth_patch_contrib.py` measures the **aggregate
contribution**: hold the FP16 weights, replace ONE layer by its ConvRot INT8
reconstruction in the **original frame** (`dequant(INT8(W @ H^T))` = `(q*s) @ H`),
and rank by the change in the patch's own output over 8 fixed seeds.

The injection frame is the whole point. The pack stores the **rotated** weight
and the kernel rotates the **activation**, so the error that reaches the output
is `(q*s) @ H - W` - the **original** frame. Injecting in the rotated frame
(`q*s`) would measure the rotation itself, not the error, and inverts the
ranking.

| rank | layer | the aggregate contribution |
| :--- | :--- | :--- |
| 1 | **`img_in`** | **1.1247e-05** |
| 2 | `controlnet_blocks.58.output_proj` | 1.8809e-08 |
| 3 | `controlnet_blocks.48.output_proj` | 1.8321e-08 |
| 4 | `controlnet_blocks.13.output_proj` | 1.8170e-08 |
| 5 | `controlnet_blocks.21.output_proj` | 1.7917e-08 |
| … | (the 60 blocks span 2.6e-09 .. 1.9e-08; the smallest is `controlnet_blocks.43.input_proj`) | … |

**`img_in` is 598x above the runner-up and carries 90.9 % of the summed contribution** (1.1247e-05 of 1.2373e-05; the other 120 layers average 9.39e-09). That is structural, not incidental:
`img_in` is the single entry point that converts the encoded control image into
the 3072-wide feature all 60 blocks consume, so every block's residual inherits
its error. The 60 blocks are cross-cancelling in the aggregate; each one's own
contribution is two to three orders below `img_in`'s.

---

## 5. The candidates, and the adopted set

Built from the contribution ranking (the top-K kept in FP16) and measured on
the fixed **256-item** acceptance set (seed 42), through the real loader. The
gate is the patch's own output - the same instrument the ranking was measured
with.

| the candidate | the INT8 | the kept | cos mean | cos min | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 121 | 0 | 0.9999555935 | 0.9999507946 | 1083.10 MB |
| **k=1 (adopted)** | **120** | **1 (`img_in`)** | **0.9999924905** | **0.9999924096** | **1083.28 MB** |
| k=5 | 116 | 5 | 0.9999926819 | 0.9999926039 | 1119.23 MB |
| k=10 | 111 | 10 | 0.9999929108 | 0.9999928415 | 1164.17 MB |

**k=1 was adopted.** Protecting `img_in` alone moves the minimum from
0.9999507946 to 0.9999924096 - a **597x cut of the error** (3.7e-05 to 6.3e-08)
- for **+0.18 MB**. Every later candidate is flat by comparison: k=1 to k=10
buys only +4.2e-07 of the mean for **+80.9 MB**, which is the gate-satisfied
region and not worth the bytes.

This is the sharpest form of the size/quality rule measured so far: the curve is
not a plateau, it is a **single step**. One layer carries 90.9 % of the summed
contribution, so the answer follows directly.

---

## 6. The pack

| item | value |
| :--- | :--- |
| the source | 2161.8 MB (BF16) |
| the pack | **1083.3 MB** (-49.9 %) |
| the SHA256 | `C2899255252B81FD64EF2FAE0FBFD86E1496BA04CF10CC95FE45679120F1C996` |
| the quantized layers | **120 of 121** (`img_in` kept in FP16) |
| the kept tensors | `img_in.weight` + the 1D norms/biases, stored **FP16** (Rule 5) |
| the gate (256 items, seed 42) | **PASS** (mean >= 0.99, min >= 0.98) |

### What the rebuild changed against the previous pack

| | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the protected | 0 | **1 (`img_in`)** |
| the kept dtypes | 121 BF16 + **120 F32** | 241 **FP16** |
| the cos mean / min | 0.9999555935 / 0.9999507946 | **0.9999924905 / 0.9999924096** |
| the size | 1083.80 MB | **1083.28 MB** |

Two defects are fixed at once:

1. **The missing protection.** `img_in` was quantized; the measurement says it
   is 598x the runner-up, so it must stay FP16. The minimum improves 597x.
2. **The FP32 kept tensors.** The previous pack stored 120 kept tensors in FP32,
   which the runtime casts to FP16 at load - the bytes were wasted (Rule 5). The
   rebuilt pack stores every kept tensor FP16.

Both fixes point the same way, and the rebuilt pack is **0.52 MB smaller** than
the one it replaces while being strictly better on quality.

---

## 7. The acceptance through the real loader

The numbers above were produced by two independent harnesses, and both agree:

| the harness | how it builds the pack | cos mean / min (k=1) |
| :--- | :--- | :--- |
| `sdxl/verify_qwen_image_diffsynth_patch.py` | the module graph built directly with `mixed_precision_ops` | 0.9999924905 / 0.9999924096 |
| `sdxl/verify_qwen_image_diffsynth_patch_loader.py` | `HSWQModelPatchLoaderCustom.load_model_patch(name, False)` - the node ComfyUI runs | 0.9999924905 / 0.9999924096 |

The loader harness stages a candidate into `models/model_patches` under a unique
name (so the real loader can address it), loads it, measures, then removes it.
The loaded module is confirmed to be `QwenImageBlockWiseControlNet` with 120
INT8 layers and `img_in` kept - i.e. the pack really is the INT8 layout the
loader consumes, not a pack that happens to load as FP16.

This matters because the stock `ModelPatchLoader` would silently read the INT8
tensors as plain weights. The loader harness is what rules that out.

---

## 8. Reproduction

```powershell
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe
#   COMFY   = <ComfyUI>  (the checkout ComfyUI runs from)

# 1. the criterion: the aggregate contribution (8 seeds, float64)
python sdxl/measure_qwen_image_diffsynth_patch_contrib.py `
    --patch <ComfyUI>/models/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors `
    --out impact/contrib_qwen_image_canny_diffsynth.json --n 8

# 2. the builder (top-k kept in FP16)
python clip_convert/convert_qwen_image_diffsynth_patch_impact_convrot_int8.py `
    --input <ComfyUI>/models/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors `
    --output impact/_work/qds_k1.safetensors `
    --contrib impact/contrib_qwen_image_canny_diffsynth.json --top-k 1

# 3. the acceptance through the REAL loader (256 items, seed 42)
$env:QDS_N = "256"
python sdxl/verify_qwen_image_diffsynth_patch_loader.py `
    <ComfyUI>/models/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors `
    impact/_work/qds_k1.safetensors
```

**Determinism.** The builder is exact integer/float arithmetic over the source
tensors and the contribution list, so it is byte-reproducible: re-running step 2
on the same inputs reproduced the placed pack byte for byte
(SHA256 `C2899255252B81FD64EF2FAE0FBFD86E1496BA04CF10CC95FE45679120F1C996`, both
times). The acceptance is deterministic too: the 256 inputs are generated from
`torch.Generator(device="cpu").manual_seed(42 + i)`, and the two harnesses
returned identical numbers to 10 decimal places.

**Timing.** The contribution pass takes ~55 min on the target machine (121
layers x 9 forwards over 60 blocks, all in float64). The acceptance takes ~45 min
for the 256 items. The builder takes under a minute.

---

## 9. Error sources and limits

**Error sources in the measurement itself.**

- **float32 arithmetic is fatal here.** The residuals run to ~3e5; a float32 dot
  product over 47M elements accumulates ~1e-3 of noise, larger than the entire
  signal (< 1.2e-05). Every cosine must be taken in float64. Symptom of the
  mistake: values that are all negative or all equal.
- **Concatenating the 60 residuals hides the ranking.** A single vector lets the
  largest block's norm dominate. Score per block and average.
- **`cond` must be recomputed inside the perturbation loop.** `cond` is produced
  by `process_input_latent_image`, which runs through `img_in` - itself one of
  the layers under test. Precomputing `cond` once (from the unperturbed model)
  understates `img_in`'s contribution by an order of magnitude.
- **The baseline must use the same ops class as the pack.** The stock path is
  `manual_cast`, the INT8 path is `mixed_precision_ops`; they are not
  numerically equal. Both harnesses build the baseline with the INT8 path's
  `mixed_precision_ops`.

**Limits of the score.**

- **The score is input-distribution dependent.** It is measured on fixed random
  tensors of the correct shape (the VAE / text encoder are not on this machine).
  The *shape and the graph* are exact; the *content* is synthetic. This is the
  same substitution the ControlNet sweeps use, and it is sound for ranking
  layers, but a pack adopted on this score should be spot-checked once on a real
  image before release.
- **The score covers the model patch's own output only.** It does not include
  the DiT, the sampler, or the VAE decoder. A patch that is perfect in its own
  output can still interact with a specific base model or sampler; the trajectory
  gate (`verify_*_traj.py`) exists for that and should be run when a base model
  is available.
- **It is a 256-item set.** `cos.min` is the worst of 256 draws, not a bound.
  Differences below ~1e-07 between candidates are not resolved; that is why
  k=5 and k=10 are reported as effectively tied.
- **A single-layer contribution is marginal by construction.** The ranking says
  which layers are individually expensive, not what happens when many are
  quantized together (they can partially cancel). The acceptances are therefore
  run on the *complete pack*, not on a sum of contributions.

**Cases where this method does not apply.**

- **Layers that are not 2D Linear.** Conv2d, embeddings and norms are not
  ConvRot-eligible here and stay FP16; they are outside the ranking.
- **Widths that are not a power-of-4 multiple.** The group size falls back from
  256 to 64 to 4 and stops; a width with no power-of-4 divisor cannot be rotated
  and stays FP16 (it is reported as protected).
- **A checkpoint whose runtime has no INT8 load path.** The pack is only useful
  where the loader consumes `int8_tensorwise` `comfy_quant` stamps. For these
  Qwen-Image diffsynth patches that path is the HSWQ model-patch loader; the
  stock `ModelPatchLoader` does not consume them.

---

## 10. The experiment ledger

| # | when (JST) | what | the setting | the score (cos mean / min) | adopted |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2026-10-23 11:20 | the aggregate contribution | 121 layers, 8 seeds, float64, per-block | - (a ranking) | - |
| 2 | 2026-10-23 11:33 | candidate k=0 | 121 INT8, 0 kept | 0.9999555935 / 0.9999507946 | no |
| 3 | 2026-10-23 11:33 | candidate k=1 | 120 INT8, 1 kept (`img_in`) | 0.9999924905 / 0.9999924096 | **yes** |
| 4 | 2026-10-23 11:33 | candidate k=5 | 116 INT8, 5 kept | 0.9999926819 / 0.9999926039 | no (size) |
| 5 | 2026-10-23 11:33 | candidate k=10 | 111 INT8, 10 kept | 0.9999929108 / 0.9999928415 | no (size) |
| 6 | 2026-10-23 11:33 | the previous pack | 121 INT8, 0 kept, BF16/F32 kept | 0.9999555935 / 0.9999507946 | replaced |
| 7 | 2026-10-23 13:55 | the loader acceptance | k=0 and k=1 through `HSWQModelPatchLoaderCustom`, 256 items | k=0: 0.9999555935 / 0.9999507946; k=1: 0.9999924905 / 0.9999924096 | confirms #3 |
| 8 | 2026-10-23 13:00 | the reproducibility check | re-run the builder (k=1) | byte-identical SHA256 | confirms #3 |

Every row is a real run on this machine; the raw JSON is kept under `impact/`
(git-ignored). The runtime was the one described in section 8 (a single
consumer GPU; 256 items per acceptance).

---

## 11. Files

| file | role |
| :--- | :--- |
| `sdxl/measure_qwen_image_diffsynth_patch_contrib.py` | the criterion (the aggregate contribution) |
| `clip_convert/convert_qwen_image_diffsynth_patch_impact_convrot_int8.py` | the builder (`--top-k`, FP16 kept) |
| `sdxl/verify_qwen_image_diffsynth_patch.py` | the acceptance, module graph built directly |
| `sdxl/verify_qwen_image_diffsynth_patch_loader.py` | the acceptance through the real HSWQ loader (the path ComfyUI uses) |
| `impact/contrib_qwen_image_canny_diffsynth.json` | the contribution (git-ignored) |
| `impact/verify_qds_256.json`, `impact/verify_qds_loader_both.json` | the acceptance results (git-ignored) |
