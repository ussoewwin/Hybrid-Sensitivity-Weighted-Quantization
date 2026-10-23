# Qwen-Image diffsynth ControlNet Depth (model patch) - ConvRot INT8 Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/model_patches/qwen_image_depth_diffsynth_controlnet.safetensors` (2161.8 MB, BF16, 362 tensors)
**Pack.** `models/model_patches/qwen_image_depth_diffsynth_controlnet_hswq_convrot_int8.safetensors` (1083.3 MB, **−49.9 %**)
**SHA256.** `7B82A9360169FA99524C03BCF913CE63713930174AD1FC332E553E5646BAD44C`

The `canny` and `inpaint` siblings share this architecture and the same load
path; see `md/QwenImageDiffsynthControlnet_ConvRot_INT8_ModelPatch_Guide.md` for
the full method walkthrough. This guide carries this checkpoint's numbers.

---

## 1. What the score is

The score is the **observed change of the model patch's own output**, not a
static weight statistic. For block `i` of the 60 control blocks:

```
cos_i = <r_source_i, r_pack_i> / (||r_source_i|| * ||r_pack_i||)
1 - cos_i  = that block's per-item error
```

A pack's score over the acceptance set is the **mean over blocks** of
`1 - cos_i`, then averaged over items. The reported `cos.mean` / `cos.min` are
`1 - (the mean error)` and `1 - (the worst item's error)`.

Two choices are forced by the runtime and both were measured:

- **The observable is a list of 60 residuals, not one concatenated vector.** The
  apply node injects each block's residual at that block (`img[:, :n] +=
  control_block(img[:, :n], cond, i) * strength`), so the 60 residuals are
  independent contributions. Pooling them lets the largest block's norm decide
  the whole number.
- **The arithmetic is float64.** The residuals run to ~3e5, so a float32 dot
  product over 47,185,920 elements accumulates ~1e-3 of rounding noise - larger
  than the signal, whose smallest layers measure ~2.2e-09.

The gate: `cos.mean >= 0.99` and `cos.min >= 0.98` over the fixed 256-item set
(seed 42), finite output. Values are quoted to 10 decimals because candidates
differ at 1e-07.

---

## 2. This is a model patch, and it needs the HSWQ loader

ComfyUI dispatches the checkpoint on `controlnet_blocks.0.y_rms.weight` into
`comfy_extras.nodes_model_patch.QwenImageBlockWiseControlNet`: 60 blocks of
`[x_rms, y_rms, input_proj, gelu, output_proj]` plus one shared `img_in`.

**ComfyUI's stock `ModelPatchLoader` does not recognise the INT8 layout** - it
builds the patch with `manual_cast`, which would read the pack's INT8 tensors as
plain weights. Load the pack with the **HSWQ Model Patch Loader**
(`HSWQModelPatchLoaderCustom`), which detects the `int8_tensorwise`
`comfy_quant` stamps and builds the patch with `mixed_precision_ops`: weights
stay INT8 and the forward uses the comfy-kitchen `int8_linear` kernel with
online ConvRot rotation.

---

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 2161.8 MB, BF16, 362 tensors |
| the 2D weights | **121** (all ConvRot-eligible) |
| the layer families | `img_in` + `controlnet_blocks.{0..59}` (`input_proj` / `output_proj`; `x_rms` / `y_rms` are 1D norms) |
| the group size | 256 (all 121) |

`img_in.weight` is `[3072, 64]`, i.e. **`additional_in_dim = 0`** (16 latent
channels, no mask) - the same shape as the canny variant, unlike inpaint's
`[3072, 68]`. Both measurement harnesses derive the latent channel count from
`img_in.in_features // 4`, so one script serves the whole family.

No key conversion happens on load (only the Z-Image branch runs
`z_image_convert`), so the pack carries the plain checkpoint names.

---

## 4. The impact measurement

`sdxl/measure_qwen_image_diffsynth_patch_contrib.py` measures the **aggregate
contribution**: hold the FP16 weights, replace ONE layer by its ConvRot INT8
reconstruction in the **original frame** (`dequant(INT8(W @ H^T))` = `(q*s) @ H`),
and rank by the change in the patch's own output over 8 fixed seeds.

The injection frame is the whole point: the pack stores the **rotated** weight
and the kernel rotates the **activation**, so the error that reaches the output
is `(q*s) @ H - W` - the original frame. Injecting in the rotated frame measures
the rotation itself and inverts the ranking.

| rank | layer | the aggregate contribution |
| :--- | :--- | :--- |
| 1 | **`img_in`** | **5.7632e-06** |
| 2 | `controlnet_blocks.17.output_proj` | 1.5335e-08 |
| 3 | `controlnet_blocks.19.output_proj` | 1.5097e-08 |
| 4 | `controlnet_blocks.42.output_proj` | 1.4998e-08 |
| 5 | `controlnet_blocks.0.output_proj` | 1.4310e-08 |
| … | (the 60 blocks span 2.2e-09 .. 1.5e-08; the smallest is `controlnet_blocks.44.input_proj`) | … |

**`img_in` is 376x above the runner-up and carries 85.8 % of the summed
contribution.** That is structural: `img_in` is the single entry point that
converts the encoded control latent into the 3072-wide feature all 60 blocks
consume, so every block's residual inherits its error. The 60 blocks
cross-cancel in the aggregate.

---

## 5. The candidates, and the adopted set

Built from the contribution ranking (the top-K kept in FP16) and measured on the
fixed **256-item** acceptance set (seed 42), through the real loader. The gate is
the patch's own output - the same instrument the ranking was measured with.

| the candidate | the INT8 | the kept | cos mean | cos min | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 121 | 0 | 0.9999695153 | 0.9999653491 | 1083.10 MB |
| **k=1 (adopted)** | **120** | **1 (`img_in`)** | **0.9999924109** | **0.9999923262** | **1083.28 MB** |
| k=5 | 116 | 5 | 0.9999925941 | 0.9999925040 | 1119.23 MB |
| k=10 | 111 | 10 | 0.9999928184 | 0.9999927315 | 1164.17 MB |

**k=1 was adopted.** Protecting `img_in` alone moves the minimum from
0.9999653491 to 0.9999923262 - a **642x cut of the error** (3.5e-05 to 5.4e-08)
- for **+0.18 MB**. Every later candidate is flat by comparison: k=1 to k=10
buys only +4.1e-07 of the mean for **+80.9 MB**, which is the gate-satisfied
region and not worth the bytes.

The curve is a **single step**, not a plateau: one layer carries 85.8 % of the
summed contribution, so the answer follows directly.

---

## 6. The pack

| item | value |
| :--- | :--- |
| the source | 2161.8 MB (BF16) |
| the pack | **1083.3 MB** (-49.9 %) |
| the SHA256 | `7B82A9360169FA99524C03BCF913CE63713930174AD1FC332E553E5646BAD44C` |
| the quantized layers | **120 of 121** (`img_in` kept in FP16) |
| the kept tensors | `img_in.weight` + the 1D norms/biases, stored **FP16** (Rule 5) |
| the gate (256 items, seed 42) | **PASS** (mean >= 0.99, min >= 0.98) |

**This pack is new.** The checkpoint had no ConvRot INT8 pack before; the
sources list carried only the `canny` and `inpaint` packs. The pack is written
with the current method from the first commit, so there is no "what changed"
delta - the k=0 row above is the all-INT8 baseline that any alternative would
have to beat, and k=1 beats it by 642x of the error for 0.18 MB.

**Determinism.** The builder is exact arithmetic over the source tensors and the
contribution list, so it is byte-reproducible: re-running it reproduced the
placed pack byte for byte (SHA256 `7B82A936...D44C`, both times).

---

## 7. Acceptance through the real loader

The numbers above were produced by the loader harness
(`sdxl/verify_qwen_image_diffsynth_patch_loader.py`), which drives
`HSWQModelPatchLoaderCustom.load_model_patch(name, False)` - the node ComfyUI
runs - over the 256-item set. The loaded module is confirmed to be
`QwenImageBlockWiseControlNet` with 120 INT8 layers and `img_in` kept, so the
pack really is the INT8 layout the loader consumes; the stock
`ModelPatchLoader` would silently read it as FP16.

The acceptance is deterministic: the 256 inputs come from
`torch.Generator(device="cpu").manual_seed(42 + i)`.

---

## 8. Reproduction

```powershell
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe
#   COMFY   = <ComfyUI>  (the checkout ComfyUI runs from)

# 1. the criterion: the aggregate contribution (8 seeds, float64)
python sdxl/measure_qwen_image_diffsynth_patch_contrib.py `
    --patch <ComfyUI>/models/model_patches/qwen_image_depth_diffsynth_controlnet.safetensors `
    --out impact/contrib_qwen_image_depth_diffsynth.json --n 8

# 2. the builder (top-k kept in FP16)
python clip_convert/convert_qwen_image_diffsynth_patch_impact_convrot_int8.py `
    --input <ComfyUI>/models/model_patches/qwen_image_depth_diffsynth_controlnet.safetensors `
    --output impact/_work/qdd_k1.safetensors `
    --contrib impact/contrib_qwen_image_depth_diffsynth.json --top-k 1

# 3. the acceptance through the REAL loader (256 items, seed 42)
$env:QDS_N = "256"
python sdxl/verify_qwen_image_diffsynth_patch_loader.py `
    <ComfyUI>/models/model_patches/qwen_image_depth_diffsynth_controlnet.safetensors `
    impact/_work/qdd_k1.safetensors
```

**Timing.** The contribution pass takes ~55 min on the target machine (121
layers x 9 forwards over 60 blocks, all in float64). The acceptance takes
~45 min for the 256 items (longer when ComfyUI shares the GPU). The builder takes
under a minute.

---

## 9. Error sources and limits

**Error sources in the measurement itself.**

- **float32 arithmetic is fatal here.** The residuals run to ~3e5; a float32 dot
  product over 47M elements accumulates ~1e-3 of noise, larger than the whole
  signal (< 6e-06). Every cosine must be in float64. Symptom of the mistake:
  values that are all negative or all equal.
- **Concatenating the 60 residuals hides the ranking.** A single vector lets the
  largest block's norm dominate. Score per block and average.
- **`cond` must be recomputed inside the perturbation loop.** `cond` is produced
  by `process_input_latent_image`, which runs through `img_in` - itself under
  test. Precomputing it (from the unperturbed model) understates `img_in` by an
  order of magnitude.
- **The baseline must use the same ops class as the pack.** Stock is
  `manual_cast`, INT8 is `mixed_precision_ops`; they are not numerically equal.
- **The latent channel count differs per checkpoint.** canny and depth are 16
  channels (`img_in` width 64); inpaint is 17 (`img_in` width 68,
  `additional_in_dim = 4`). Both harnesses derive it from
  `img_in.in_features // 4`; a hardcoded value gives
  `mat1 and mat2 shapes cannot be multiplied` on the wrong variant.

**Limits of the score.**

- **Input-distribution dependent.** Measured on fixed random tensors of the
  correct shape (no VAE / text encoder on this machine). Sound for ranking
  layers; spot-check once on a real image before release.
- **The patch's own output only.** No DiT, sampler or VAE. Use the trajectory
  gate when a base model is available.
- **A 256-item set.** `cos.min` is the worst of 256 draws, not a bound.
  Differences below ~1e-07 between candidates are not resolved; that is why k=5
  and k=10 are reported as effectively tied.
- **Single-layer contributions are marginal by construction.** Many quantized
  together can partially cancel; the acceptances run on the *complete pack*.

**Cases where this method does not apply.**

- Layers that are not 2D Linear (Conv2d, embeddings, norms) stay FP16.
- Widths with no power-of-4 divisor cannot be rotated; they stay FP16.
- A checkpoint whose runtime has no INT8 load path (these need the HSWQ loader).

---

## 10. The experiment ledger

| # | when (JST) | what | the setting | the score (cos mean / min) | adopted |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2026-10-23 15:30 | the aggregate contribution | 121 layers, 8 seeds, float64, per-block | - (a ranking) | - |
| 2 | 2026-10-23 15:40 | candidate k=0 | 121 INT8, 0 kept | 0.9999695153 / 0.9999653491 | no |
| 3 | 2026-10-23 15:40 | candidate k=1 | 120 INT8, 1 kept (`img_in`) | 0.9999924109 / 0.9999923262 | **yes** |
| 4 | 2026-10-23 15:40 | candidate k=5 | 116 INT8, 5 kept | 0.9999925941 / 0.9999925040 | no (size) |
| 5 | 2026-10-23 15:40 | candidate k=10 | 111 INT8, 10 kept | 0.9999928184 / 0.9999927315 | no (size) |
| 6 | 2026-10-23 15:45 | the reproducibility check | re-run the builder (k=1) | byte-identical SHA256 | confirms #3 |

Every row is a real run on this machine; the raw JSON is kept under `impact/`
(git-ignored).

---

## 11. Files

| file | role |
| :--- | :--- |
| `sdxl/measure_qwen_image_diffsynth_patch_contrib.py` | the criterion (the aggregate contribution) |
| `clip_convert/convert_qwen_image_diffsynth_patch_impact_convrot_int8.py` | the builder (`--top-k`, FP16 kept) |
| `sdxl/verify_qwen_image_diffsynth_patch.py` | the acceptance, module graph built directly |
| `sdxl/verify_qwen_image_diffsynth_patch_loader.py` | the acceptance through the real HSWQ loader (the path ComfyUI uses) |
| `md/QwenImageDepthDiffsynthControlnet_ConvRot_INT8_Guide.md` | this guide |
| `impact/contrib_qwen_image_depth_diffsynth.json` | the contribution (git-ignored) |
| `impact/verify_qdd_loader_256.json` | the acceptance results (git-ignored) |
