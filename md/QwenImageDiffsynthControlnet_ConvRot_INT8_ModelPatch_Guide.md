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

## 0. This is a model patch, and it needs the HSWQ loader

The file name says "controlnet" but the runtime object is a **model patch**.
ComfyUI dispatches it on the key `controlnet_blocks.0.y_rms.weight` into
`comfy_extras.nodes_model_patch.QwenImageBlockWiseControlNet` (60 blocks of
`[x_rms, y_rms, input_proj, gelu, output_proj]` plus a shared `img_in`).

**ComfyUI's stock `ModelPatchLoader` does not recognise the INT8 layout.** The
stock loader builds the patch with `comfy.ops.manual_cast`, which would read the
pack's INT8 tensors as plain weights. The pack must be loaded with the
**`HSWQ Model Patch Loader`** (`custom_nodes/ComfyUI-HSWQ-Loader-and-Tools`,
`HSWQModelPatchLoader`), which detects the `int8_tensorwise` `comfy_quant`
stamps and builds the patch with `mixed_precision_ops` so the weights stay INT8
and the forward uses the comfy-kitchen `int8_linear` kernel.

The apply node calls, once per DiT block `i`:

```
img[:, :n] += model.control_block(img[:, :n], cond, i) * strength
```

with `cond = model.process_input_latent_image(vae_encode(control_image))`.

---

## 1. The model

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

## 2. The impact measurement

`sdxl/measure_qwen_image_diffsynth_patch_contrib.py` measures the **aggregate
contribution**: hold the FP16 weights, replace ONE layer by its ConvRot INT8
reconstruction in the **original frame** (`dequant(INT8(W @ H^T))` = `(q*s) @ H`),
and rank by the change in the patch's own output over 8 fixed seeds.

Two things the measurement has to get right:

1. **The observable is a LIST of 60 residuals, not one concatenated vector.**
   The runtime injects each block's residual into the DiT at that block, so the
   60 residuals are independent contributions. Pooling them into one
   concatenation lets the largest block's norm decide the number and hides the
   rest; each block is therefore scored on its own and the score is the mean of
   the per-block `1 - cos`.
2. **float64 is required.** The residuals run to ~3e5 in magnitude, so a float32
   dot product over 47185920 elements accumulates ~1e-3 of rounding noise - which
   **swamps the entire signal** (the smallest layers measure ~2.6e-09). The first
   run in float32 returned every value negative; the corrected run is below.

| rank | layer | the aggregate contribution |
| :--- | :--- | :--- |
| 1 | **`img_in`** | **1.1247e-05** |
| 2 | `controlnet_blocks.58.output_proj` | 1.8809e-08 |
| 3 | `controlnet_blocks.48.output_proj` | 1.8321e-08 |
| 4 | `controlnet_blocks.13.output_proj` | 1.8170e-08 |
| 5 | `controlnet_blocks.21.output_proj` | 1.7917e-08 |
| … | (the 60 blocks are all within 2.6e-09 .. 1.9e-08; the smallest is `controlnet_blocks.43.input_proj`) | … |

**`img_in` is 598x above the runner-up.** That is the structural reason: `img_in`
is the single entry point that converts the encoded control image into the
3072-wide feature the 60 blocks all consume, so every block's residual inherits
its error. The 60 blocks are cross-cancelling in the aggregate - each one's own
contribution is two to three orders below `img_in`'s.

---

## 3. The candidates, and the adopted set

Built from the contribution ranking (the top-K kept in FP16) and measured on the
fixed **256-item** acceptance set (seed 42). The gate is the patch's **own
output** - the same instrument the ranking was measured with.

| the candidate | the INT8 | the kept | cos mean | cos min | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 121 | 0 | 0.99995559 | 0.99995079 | 1083.10 MB |
| **k=1 (adopted)** | **120** | **1 (`img_in`)** | **0.99999249** | **0.99999241** | **1083.28 MB** |
| k=5 | 116 | 5 | 0.99999268 | 0.99999260 | 1119.23 MB |
| k=10 | 111 | 10 | 0.99999291 | 0.99999284 | 1164.17 MB |

**k=1 was adopted.** Protecting `img_in` alone moves the minimum from
0.99995079 to 0.99999241 - a **597x reduction of the error** (3.7e-05 to 6.3e-08)
- for **+0.18 MB**. Every later candidate is flat by comparison: k=1 to k=10
buys only +4.2e-07 of the mean for **+80.9 MB**, which is the gate-satisfied
region and not worth the bytes.

This is the sharpest version of the floor/plateau rule measured so far: the
curve is not a plateau, it is a **single step**. One layer carries essentially
all of the measurable contribution, and the correct answer follows directly.

Note that the previous pack had no protection at all (`img_in` at INT8), which
is why it sat at 0.99995079.

---

## 4. The pack

| item | value |
| :--- | :--- |
| the source | 2161.8 MB (BF16) |
| the pack | **1083.3 MB** (-49.9 %) |
| the SHA256 | `C2899255252B81FD64EF2FAE0FBFD86E1496BA04CF10CC95FE45679120F1C996` |
| the quantized layers | **120 of 121** (`img_in` kept in FP16) |
| the kept tensors | `img_in.weight` + the 1D norms/biases, stored **FP16** (Rule 5) |
| the output | finite (int8_linear 150784 calls over the 256 items) |
| the gate (256 items, seed 42) | **PASS** (mean >= 0.99, min >= 0.98) |

### What the rebuild changed against the previous pack

| | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the protected | 0 | **1 (`img_in`)** |
| the kept dtypes | 121 BF16 + **120 F32** | 241 **FP16** |
| the cos mean / min | 0.99995559 / 0.99995079 | **0.99999249 / 0.99999241** |
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

## 5. Files

| file | role |
| :--- | :--- |
| `sdxl/measure_qwen_image_diffsynth_patch_contrib.py` | the aggregate contribution (the criterion) |
| `clip_convert/convert_qwen_image_diffsynth_patch_impact_convrot_int8.py` | the builder (`--top-k`, FP16 kept) |
| `sdxl/verify_qwen_image_diffsynth_patch.py` | the acceptance on the fixed 256-item set (the patch's own output) |
| `impact/contrib_qwen_image_canny_diffsynth.json` | the contribution (git-ignored) |
| `impact/verify_qds_256.json` | the acceptance results (git-ignored) |
