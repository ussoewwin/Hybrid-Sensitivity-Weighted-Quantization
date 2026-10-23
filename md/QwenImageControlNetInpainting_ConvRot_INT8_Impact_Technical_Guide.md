# Qwen-Image ControlNet Inpainting — ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `Qwen-Image-ControlNet-Inpainting.safetensors` (4038.4 MB, BF16, 215 tensors)
**Pack.** `Qwen-Image-ControlNet-Inpainting_convrot_int8.safetensors` (2022.1 MB, **−49.9 %**)
**SHA256.** `38F0AB7495029517D4B0BB4187CEDCBDFA090A29F0BB50DFA5733F5076A0C451`

---

## 1. The load path

This is a **ControlNet** of the InstantX family, not a model patch and not a
Fun ControlNet. ComfyUI dispatches it on the key
`controlnet_blocks.0.weight` into `load_controlnet_qwen_instantx`.

| step | what happens |
| :--- | :--- |
| the dispatch | `controlnet_blocks.0.weight` + `transformer_blocks.0.img_mlp.net.0.proj.weight` → the Qwen InstantX branch |
| the builder | `load_controlnet_qwen_instantx` |
| the class | `comfy.ldm.qwen_image.controlnet.QwenImageControlNetModel` (a `QwenImageTransformer2DModel` subclass, `final_layer=False`) |
| the mask branch | `controlnet_x_embedder.weight` is `[3072, 68]` → `extra_condition_channels = 4`, `concat_mask = True` (the inpaint path) |

**No base model is needed.** Unlike the Fun ControlNet, this class carries its
own 60-block transformer, so its forward runs self-contained - the measurement
and the acceptance have no external tensor dependency at all.

The source is already in the runtime layout, so no name remap happens.

---

## 2. The model

| item | measured |
| :--- | :--- |
| the source | 4038.4 MB, BF16, 215 tensors |
| the 2D weights | **95** (all ConvRot-eligible) |
| the layer families | `transformer_blocks.{0..59}` (attn / img_mlp / txt_mlp / img_mod / txt_mod), `controlnet_blocks.{0..5}`, `img_in`, `txt_in`, `controlnet_x_embedder`, `time_text_embed` |
| the group size | 256 (93 layers) / 64 / 4 (the narrow embedder line) |

---

## 3. The impact measurement

`sdxl/measure_qwen_image_inpaint_cn_contrib.py` measures the **aggregate
contribution**: hold the FP16 weights, replace ONE layer by its ConvRot INT8
reconstruction in the **original frame** (`dequant(INT8(W @ H^T))` = `(q*s) @ H`),
and rank by `1 - cos` of the ControlNet's **own output** over 8 fixed seeds.

| rank | layer | the aggregate contribution |
| :--- | :--- | :--- |
| 1 | `transformer_blocks.0.img_mod.1` | 1.7072e-04 |
| 2 | `img_in` | 1.4843e-04 |
| 3 | `transformer_blocks.0.attn.to_k` | 1.3775e-04 |
| 4 | `time_text_embed.timestep_embedder.linear_1` | 1.3355e-04 |
| 5 | `time_text_embed.timestep_embedder.linear_2` | 1.2890e-04 |
| 6 | `controlnet_x_embedder` | 1.2597e-04 |
| … | (block 0 and the entry path dominate) | … |

The measured set is 95 of 95 layers (none skipped). The checkpoint's whole
measurable spread is roughly one order of magnitude (1.71e-04 down to ~1e-05).

---

## 4. The candidates, and the adopted set

Built from the contribution ranking (the top-K kept in FP16) and measured on
the fixed **256-item** acceptance set (seed 42). The gate is the ControlNet's
**own output** - the same instrument the ranking was measured with.

| the candidate | the INT8 | the kept | mean | min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **k=0 (adopted)** | **95** | **0** | **0.999545** | **0.998887** | **0.999297** | **2022.1 MB** |
| k=10 | 85 | 10 | 0.999730 | 0.999651 | 0.999688 | 2177.5 MB |
| k=20 | 75 | 20 | 0.999867 | 0.999827 | 0.999852 | 2483.1 MB |
| k=40 | 55 | 40 | 0.999886 | 0.999860 | 0.999876 | 2959.6 MB |

**k=0 was adopted - the floor case.** Every ConvRot-eligible layer is already
quantized at k=0 and the gate is cleared with a wide margin (min 0.998887
against the 0.98 gate). The protection curve is a genuine trade (each protected
layer does raise the cosine), but k=20 to k=40 is nearly flat (+1.9e-05 of the
mean for +477 MB) and the size rule decides:

- the pack being replaced is **2022.1 MB**, and every protected candidate is
  **larger** (k=10 alone is +155 MB);
- the gate is already cleared at 0.9989, so +155 MB to reach 0.9997 is paying
  hundreds of MB for accuracy past the gate.

Both rules point the same way: **no protection**. The whole measured curve is
reported above rather than one number.

---

## 5. The pack

| item | value |
| :--- | :--- |
| the source | 4038.4 MB (BF16) |
| the pack | **2022.1 MB** (-49.9 %) |
| the SHA256 | `38F0AB7495029517D4B0BB4187CEDCBDFA090A29F0BB50DFA5733F5076A0C451` |
| the quantized layers | **95 of 95** (group size 256/64/4) |
| the kept tensors | the 1D biases / norms / embeddings, stored **FP16** (Rule 5) |
| the output | finite (int8_linear 24320 calls over the 256 items) |
| the gate (256 items, seed 42) | **PASS** (mean >= 0.99, min >= 0.98) |

### What the rebuild changed against the previous pack

The previous pack was the same layer set (95 INT8, no protection). The rebuild
differs in exactly one respect: the **120 kept tensors went from BF16 to FP16**
per Rule 5. The 95 F32 tensors in both packs are the INT8 `weight_scale`, which
stays FP32 by design.

| | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the kept dtypes | 120 BF16 | **120 FP16** |
| the quant layer set | 95 | 95 (identical) |
| the mean / min | 0.999545 / 0.998887 | 0.999545 / 0.998887 |
| the size | 2022.1 MB | 2022.1 MB |

The size and the acceptance are identical to the last digit, which is the
expected result when only the kept-tensor dtype changes (BF16 and FP16 are both
2 bytes, and the runtime runs FP16 either way). The value of the rebuild is the
verification: the previous pack's layer set came from the older static
selection, and this pack is re-derived under the current method - the method
**confirms** all-INT8, because nothing is protected.

---

## 6. Files

| file | role |
| :--- | :--- |
| `sdxl/measure_qwen_image_inpaint_cn_contrib.py` | the aggregate contribution (the criterion) |
| `clip_convert/convert_qwen_image_inpaint_cn_impact_convrot_int8.py` | the builder (`--top-k`, FP16 kept) |
| `sdxl/verify_qwen_image_inpaint_cn.py` | the acceptance on the fixed 256-item set (the ControlNet's own output) |
| `impact/contrib_qwen_image_inpaint_cn.json` | the contribution (git-ignored) |
| `impact/verify_qin_256.json` | the acceptance results (git-ignored) |
