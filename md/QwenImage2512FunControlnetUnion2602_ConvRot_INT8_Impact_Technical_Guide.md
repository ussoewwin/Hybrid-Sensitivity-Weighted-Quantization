# Qwen-Image-2512 Fun ControlNet Union 2602 — ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors` (3349.7 MB, BF16, 174 tensors)
**Pack.** `Qwen-Image-2512-Fun-Controlnet-Union-2602_convrot_int8.safetensors` (1677.2 MB, **−49.9 %**)
**SHA256.** `5AEC39A53F583BADE3B7DB18F38ED5A01905A0610C2AD2F9F792802178596BC4`

---

## 1. The load path

This is a **ControlNet**, not a model patch. ComfyUI dispatches it on the key
`control_blocks.0.after_proj.weight` into `load_controlnet_qwen_fun`, which
builds `comfy.ldm.qwen_image.controlnet.QwenImageFunControlNetModel` and wraps
it in `QwenFunControlNet`.

| step | what happens |
| :--- | :--- |
| the dispatch | `control_blocks.0.after_proj.weight` + `control_img_in.weight` → the Qwen Fun branch |
| the builder | `load_controlnet_qwen_fun` |
| the class | `QwenImageFunControlNetModel` (5 `control_blocks`, injection layers 0/12/24/36/48) |
| the runtime | `QwenFunControlNet.pre_run` injects `base_model` = the Qwen-Image DiT; the forward calls the base's `process_img` / `pe_embedder` / `img_in` / `txt_norm` / `txt_in` / `time_text_embed` |

The source checkpoint is **already in the runtime layout** (`control_blocks.*`,
`control_img_in.*`) - unlike an SDXL ControlNet, no diffusers→comfy name remap
happens, so the pack is written under the same names.

---

## 2. The model

| item | measured |
| :--- | :--- |
| the source | 3349.7 MB, BF16, 174 tensors |
| the 2D weights | **77** (all ConvRot-eligible) |
| the layer families | `control_blocks.{0..4}` (attn / img_mlp / txt_mlp / img_mod / txt_mod / before_proj / after_proj) + `control_img_in` |
| the group size | 256 (76 layers) / 4 (`control_img_in`, 132-wide) |

`control_img_in` is `[3072, 132]`: 33 channels × 2×2 patch grid × 1 frame. 132
is not a multiple of 256, so it falls back to group size 4.

---

## 3. The impact measurement

`sdxl/measure_qwen_image_fun_cn_contrib.py` measures the **aggregate
contribution**: hold the FP16 weights, replace ONE layer by its ConvRot INT8
reconstruction in the **original frame** (`dequant(INT8(W @ H^T))` = `(q*s) @ H`),
and rank by `1 - cos` of the ControlNet's **own output** over 8 fixed seeds.

The ControlNet's forward needs a Qwen-Image base for the entry path. That base
is built once from the local BF16 entry tensors (`img_in`, `txt_in`,
`txt_norm`, `time_text_embed`) and used **identically** in the baseline and in
every perturbed run, so it cancels out of the comparison.

| rank | layer | the aggregate contribution |
| :--- | :--- | :--- |
| 1 | `control_blocks.0.img_mod.1` | 3.3806e-04 |
| 2 | `control_blocks.0.txt_mod.1` | 8.9832e-05 |
| 3 | `control_blocks.0.attn.to_k` | 6.3278e-05 |
| 4 | `control_img_in` | 5.4806e-05 |
| 5 | `control_blocks.0.attn.add_v_proj` | 4.8310e-05 |
| … | (block 0 dominates; block 4 is the smallest) | … |

**Block 0 dominates the ranking** - its modulation and attention projections
carry the whole spread. The measured set is 77 of 77 layers (none skipped).

---

## 4. The candidates, and the adopted set

Built from the contribution ranking (the top-K kept in FP16) and measured on
the fixed **256-item** acceptance set (seed 42). The gate is the ControlNet's
**own output** - the same instrument the ranking was measured with.

| the candidate | the INT8 | the kept | mean | min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **k=0 (adopted)** | **77** | **0** | **0.999158** | **0.998215** | **0.998685** | **1677.2 MB** |
| k=10 | 67 | 10 | 0.999732 | 0.999458 | 0.999596 | 1893.3 MB |
| k=20 | 57 | 20 | 0.999891 | 0.999782 | 0.999855 | 2154.0 MB |
| k=40 | 37 | 40 | 0.999938 | 0.999898 | 0.999926 | 2558.6 MB |

**k=0 was adopted - the floor case.** Every ConvRot-eligible layer is already
quantized at k=0, and the gate is cleared with a wide margin (min 0.998215
against the 0.98 gate). The protection curve here is a genuine quality/size
trade, not the "floor drop" shape: each protected layer does raise the cosine.
But the size rule decides it:

- the pack being replaced is **1677.2 MB**, and every protected candidate is
  **larger** (k=10 alone is +216 MB);
- the gate is already cleared at 0.9982, so +216 MB to reach 0.9995 is paying
  hundreds of MB for accuracy past the gate.

Both rules point the same way: **no protection**. The whole measured curve is
reported above rather than one number.

---

## 5. The pack

| item | value |
| :--- | :--- |
| the source | 3349.7 MB (BF16) |
| the pack | **1677.2 MB** (-49.9 %) |
| the SHA256 | `5AEC39A53F583BADE3B7DB18F38ED5A01905A0610C2AD2F9F792802178596BC4` |
| the quantized layers | **77 of 77** (group size 256; `control_img_in` at group size 4) |
| the kept tensors | the 1D biases / norms, stored **FP16** (Rule 5) |
| the output | finite (int8_linear 19712 calls over the 256 items) |
| the gate (256 items, seed 42) | **PASS** (mean >= 0.99, min >= 0.98) |

### What the rebuild changed against the previous pack

The previous pack was the same layer set (77 INT8, no protection) but stored
its 97 kept tensors in **BF16**. The rebuild stores them **FP16** per Rule 5.
The size is identical (BF16 and FP16 are both 2 bytes) and the acceptance is
identical to the last digit (min 0.998215 both), which is the expected result
when only the kept-tensor dtype changes.

The value of the rebuild is the verification: the previous pack's layer set was
produced by the older static selection. This pack is re-derived under the
current method (aggregate contribution on the ControlNet's own output, the
size/quality balance) and the method **confirms** all-INT8 - the ranking was
never needed, because nothing is protected.

---

## 6. Files

| file | role |
| :--- | :--- |
| `sdxl/measure_qwen_image_fun_cn_contrib.py` | the aggregate contribution (the criterion) |
| `clip_convert/convert_qwen_image_fun_cn_impact_convrot_int8.py` | the builder (`--top-k`, FP16 kept) |
| `sdxl/verify_qwen_image_fun_cn.py` | the acceptance on the fixed 256-item set (the ControlNet's own output) |
| `impact/contrib_qwen_image_fun_cn_2602.json` | the contribution (git-ignored) |
| `impact/verify_qcn_256.json` | the acceptance results (git-ignored) |
