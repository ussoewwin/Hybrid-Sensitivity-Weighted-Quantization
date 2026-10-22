# Qwen3-VL-4B Thinking — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Source.** `models/clip/Qwen3_VL_4B_Thinking_abliterated.safetensors` (8464.5 MB, 713 keys, 358 finite 2D Linear layers)
**Pack.** `models/clip/Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` (5958.9 MB, **−29.6 %**)
**SHA256.** `49C19DAD012A8CF4152CE01D1FF6948989C6C0AD1B7AF45F673B84B6FBB1D522`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate** output.

---

## 1. The model

A Qwen3-VL text encoder is **not** a denoiser: it produces a text embedding
(and, for a multimodal prompt, the visual tokens spliced into it). So what is
measured is the drift of the model's own output, on both towers:

| tower | the layers | the observable |
| :--- | :--- | :--- |
| `model.language_model` | 36 layers x 7 Linear = 252 | the hidden state for a fixed prompt |
| `model.visual` | 24 blocks x 4 Linear = 96, plus the merger (2) and the deepstack mergers (6) | the image embeddings and the deepstack features |

The two towers together make the 358 candidates.

The runtime is ComfyUI's `comfy.text_encoders.qwen3vl`: the clip is
`Qwen3VLTEModel_`, the `Qwen3VLClipModel` is at `.qwen3vl_4b`, the language
model is at `.qwen3vl_4b.transformer.model` (`Llama2_`) and the vision tower
at `.qwen3vl_4b.transformer.visual` (`Qwen3VLVisionModel`).

---

## 2. The previous pack was built by an invalid method

The earlier pack of this model was quantized with the **old method**: the
FP16-kept set came from **static kurtosis / outlier-ratio** on the weights,
with a hard-coded pattern list. That method is invalid here:

- the kurtosis numbers of the earlier revision were themselves computed
  through a **broken safetensors header offset**, so those packs had no valid
  protection basis;
- a static weight statistic is not a proxy for the output's sensitivity.

Measured against the FP16 source on the 256-item acceptance set:

| pack | INT8 | the FP16-kept | cos mean | cos min | cos p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| the previous pack | 197 | 359 | 0.998340 | **0.988435** | 0.997495 | 6146.1 MB |

The previous pack's minimum is **0.988435**, an order of magnitude below the
gate's neighbourhood, and it is the **largest** of the packs measured here.

---

## 3. The impact measurement

Script: `sdxl/diag_impact_qwen3vl.py` -> `impact/impact_qwen3vl_4b.json`

Each candidate layer's ConvRot INT8 reconstruction is injected and the
relative MSE of the concatenated outputs (the language model's hidden state,
the vision tower's merged embeddings and the deepstack features) against the
pristine run is recorded.

| item | value |
| :--- | :--- |
| the measured layers | **358 / 358** |
| the reference output | 81,920 elements, finite |

**The injection is the un-rotated reconstruction.** The pack stores the
rotated weight (`W_hat = W H^T`) and the kernel rotates the **activation**, so
the result is `(x H) W_hat^T = x W^T`; the quantization error that reaches the
output is `(q*s) H^T - W` in the **original** frame. Injecting the rotated
frame measures a destroyed layer (measured on the FLUX ControlNet with the
real kernel: the pack's kernel 1.74e-04, the rotated-frame injection 1.89).

**Drift distribution**

| min | median | p90 | max |
| :--- | :--- | :--- | :--- |
| 1.92e-07 | 3.39e-05 | 5.58e-05 | 2.92e-04 |

**The language tower is about eight times more sensitive than the vision
tower** - a fact a static weight statistic could not have produced:

| layer kind | n | median | max |
| :--- | :--- | :--- | :--- |
| `layers.N.self_attn.k_proj` | 36 | 4.39e-05 | 6.82e-05 |
| `layers.N.self_attn.o_proj` | 36 | 4.34e-05 | 1.10e-04 |
| `layers.N.mlp.gate_proj` | 36 | 4.33e-05 | **2.92e-04** |
| `layers.N.mlp.up_proj` | 36 | 4.29e-05 | 9.02e-05 |
| `layers.N.self_attn.q_proj` | 36 | 4.22e-05 | 6.07e-05 |
| `layers.N.self_attn.v_proj` | 36 | 4.22e-05 | 9.66e-05 |
| `layers.N.mlp.down_proj` | 36 | 4.19e-05 | 1.43e-04 |
| `blocks.N.attn.qkv` | 24 | 5.56e-06 | 1.86e-05 |
| `blocks.N.attn.proj` | 24 | 4.89e-06 | 1.95e-05 |
| `blocks.N.mlp.linear_fc2` | 24 | 4.52e-06 | 2.34e-05 |

The most sensitive single layer is `layers.35.mlp.gate_proj` (2.92e-04), and
`embed_tokens` (1.07e-04) is the most sensitive of the non-Linear tensors.

---

## 4. The selection criterion, and the rebuilt pack

The per-layer drift is a **screen**, not the selection criterion. It measures
the **weight** error only (the reconstruction is placed in the module and the
plain forward is used), while the pack's real path also quantizes the
**activation** for that layer, and that is where most of the error comes from.
The FP16-kept set is therefore given explicitly, with the layers taken in
the corrected drift's ranking.

The candidates below were measured on the full 256-item acceptance set (seed
42) against the FP16 source, on the **same ops class** for the baseline and
the pack, with the **arm** active (the loader package's
`patches/clip_int8_gemm`, which makes the real `int8_linear` fire).

| the FP16-kept | INT8 layers | cos mean | cos min | cos p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 60 | 298 | 0.999640 | 0.998863 | 0.999531 | 5415.9 MB |
| **100** | **258** | **0.999649** | **0.998927** | **0.999525** | **5958.9 MB** |
| 150 | 208 | 0.999671 | 0.998898 | 0.999555 | 6676.8 MB |
| the previous pack | 197 | 0.998340 | 0.988435 | 0.997495 | 6146.1 MB |

**100 FP16-kept layers were adopted.** The quality rises with the kept count
and flattens at 100: 150 buys +2.2e-05 of the mean and nothing of the
minimum, for 718 MB more. The chosen set is at the **same size** as the
previous pack (5958.9 MB against 6146.1 MB) with a **10x better minimum**
(0.998927 against 0.988435).

---

## 5. The pack

Built with `clip_convert/convert_qwen3_vl_4b_impact_convrot_int8.py`:

```
--impact impact/impact_qwen3vl_4b.json --protect-list impact/qwen3vl_protect_100.json
```

The pack is written under the **checkpoint's own names**, which is what
ComfyUI's loader reads (it renames `model.language_model.` -> `model.` and
`model.visual.` -> `visual.` when it loads). The protect list is expressed in
the measurement's relative names (`layers.3.mlp.up_proj`,
`blocks.3.attn.qkv`, `merger.*`, `embed_tokens`, `pos_embed`) and mapped onto
the checkpoint's names by the converter.

| item | value |
| :--- | :--- |
| the source | 8464.5 MB |
| the pack | **5958.9 MB** (-29.6 %) |
| the SHA256 | `49C19DAD012A8CF4152CE01D1FF6948989C6C0AD1B7AF45F673B84B6FBB1D522` |
| the tensors | 1229 |
| the quantized layers | **258** (of 358) |
| the FP16-kept layers | 100 |
| the group size | 256 (all 258 layers) |
| the `int8_linear` calls over the 256-item set | **65792** |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42) | **0.999649 / 0.998927 / 0.999525** |
| the gate (mean >= 0.99, min >= 0.98) | **PASS** |

---

## 6. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_qwen3vl.py` | the impact measurement (both towers, the un-rotated injection) |
| `clip_convert/convert_qwen3_vl_4b_impact_convrot_int8.py` | the impact-driven builder (`--protect-list`) |
| `sdxl/verify_qwen3vl.py` | the 256-item acceptance, the loader's own path |
| `impact/impact_qwen3vl_4b.json` | the measured per-layer drift (git-ignored) |
| `impact/qwen3vl_protect_100.json` | the adopted FP16-kept list (git-ignored) |
