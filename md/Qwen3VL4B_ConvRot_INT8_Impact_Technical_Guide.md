# Qwen3-VL-4B Thinking — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Source.** `models/clip/Qwen3_VL_4B_Thinking_abliterated.safetensors` (8464.5 MB, 713 keys, 358 finite 2D Linear layers)
**Pack.** `models/clip/Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` (5297.3 MB, **−37.4 %**)
**SHA256.** `07BD6D89F0712C91C00D649463973D2D64364FF502184B053D136FA4D797E5A5`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate** output.

**The withdrawn build.** The pack that stood here before this one (sha
`49C19DAD012A8CF4152CE01D1FF6948989C6C0AD1B7AF45F673B84B6FBB1D522`, 5958.9 MB, 100
FP16-kept) is withdrawn. Its protection set was this checkpoint's own drift ranking -
rebuilding it at K=100 from the same ranking reproduces those bytes exactly - so the
defect was the **K choice**, not the origin of the set: K=100 was adopted from a
three-point curve (60/100/150) whose "flattens at 100" reading was never tested below
60. The curve below is the full one, measured on nine candidates.

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

## 2. The first pack was built by an invalid method

The first pack of this model was quantized with the **old method**: the
FP16-kept set came from **static kurtosis / outlier-ratio** on the weights,
with a hard-coded pattern list. That method is invalid here:

- the kurtosis numbers of the earlier revision were themselves computed
  through a **broken safetensors header offset**, so those packs had no valid
  protection basis;
- a static weight statistic is not a proxy for the output's sensitivity.

Measured against the FP16 source on the 256-item acceptance set:

| pack | INT8 | the FP16-kept | cos mean | cos min | cos p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| that pack | 197 | 359 | 0.998340 | **0.988435** | 0.997495 | 6146.1 MB |

Its minimum is **0.988435**, an order of magnitude below the gate's
neighbourhood, and it was the **largest** of the packs measured for this model.

---

## 3. The impact measurement

Script: `sdxl/diag_impact_qwen3vl.py` -> `impact/impact_qwen3vl_thinking.json`
(the shipped copy is `md/verdicts/records/impact_qwen3vl_thinking.json`)

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
| 2.04e-07 | 3.39e-05 | 5.58e-05 | 2.92e-04 |

**The language tower is about ten times more sensitive than the vision
tower** - a fact a static weight statistic could not have produced:

| layer kind | n | median | max |
| :--- | :--- | :--- | :--- |
| `layers.N.self_attn.k_proj` | 36 | 4.39e-05 | 6.82e-05 |
| `layers.N.self_attn.o_proj` | 36 | 4.34e-05 | 1.10e-04 |
| `layers.N.mlp.gate_proj` | 36 | 4.33e-05 | **2.92e-04** |
| `layers.N.mlp.up_proj` | 36 | 4.28e-05 | 9.02e-05 |
| `layers.N.self_attn.q_proj` | 36 | 4.22e-05 | 6.06e-05 |
| `layers.N.self_attn.v_proj` | 36 | 4.21e-05 | 9.66e-05 |
| `layers.N.mlp.down_proj` | 36 | 4.19e-05 | 1.43e-04 |
| `blocks.N.attn.qkv` | 24 | 4.46e-06 | 1.84e-05 |
| `blocks.N.mlp.linear_fc2` | 24 | 4.21e-06 | 3.30e-05 |
| `blocks.N.attn.proj` | 24 | 3.96e-06 | 1.71e-05 |
| `blocks.N.mlp.linear_fc1` | 24 | 3.82e-06 | 1.61e-05 |

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
| 20 | 338 | 0.999602 | 0.997877 | 0.999442 | 4914.0 MB |
| 30 | 328 | 0.999613 | 0.996474 | 0.999487 | 5073.8 MB |
| 40 | 318 | 0.999625 | 0.998508 | 0.999486 | 5171.2 MB |
| **50** | **308** | **0.999634** | **0.998849** | **0.999487** | **5297.3 MB** |
| 60 | 298 | 0.999638 | 0.998873 | 0.999489 | 5415.9 MB |
| 80 | 278 | 0.999643 | 0.998794 | 0.999500 | 5666.8 MB |
| 100 | 258 | 0.999647 | 0.998933 | 0.999502 | 5958.9 MB |
| 120 | 238 | 0.999653 | 0.998841 | 0.999534 | 6209.8 MB |
| 150 | 208 | 0.999669 | 0.998911 | 0.999523 | 6676.8 MB |
| the withdrawn build | 258 | 0.999647 | 0.998933 | 0.999502 | 5958.9 MB |

Five candidates (K=20/30/40/50/100) were re-measured on the **1024-item** set,
so the worst prompt is placed on a larger sample:

| the FP16-kept | INT8 layers | cos mean | cos min | cos p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 20 | 338 | 0.999610 | 0.997877 | 0.999479 | 4914.0 MB |
| 30 | 328 | 0.999625 | 0.996474 | 0.999514 | 5073.8 MB |
| 40 | 318 | 0.999633 | 0.998508 | 0.999509 | 5171.2 MB |
| **50** | **308** | **0.999640** | **0.998849** | **0.999522** | **5297.3 MB** |
| 100 | 258 | 0.999654 | 0.998933 | 0.999542 | 5958.9 MB |

**50 FP16-kept layers were adopted**, and the reason is in the two tables rather
than in one number:

- the worst-prompt cosine of the candidates with K >= 60 lies in the band
  0.998794 to 0.998933; K=50's 0.998849 is inside that band, while K=40's
  0.998508 sits below it and K=30's 0.996474 and K=20's 0.997877 further below;
- the fifth percentile and the mean are flat from K=30 upward (p5 0.999514,
  0.999509, 0.999522, 0.999542), so the quality stops paying for the bytes
  well before K=60;
- against the withdrawn K=100 build, K=50 gives up 8.4e-05 of minimum and
  2.0e-05 of fifth percentile for **661.6 MB less**.

---

## 5. The pack

Built with `clip_convert/convert_qwen3_vl_4b_impact_convrot_int8.py`:

```
--impact impact/impact_qwen3vl_thinking.json --protect-list impact/qwen3vl_thinking_protect_50.json
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
| the pack | **5297.3 MB** (-37.4 %) |
| the SHA256 | `07BD6D89F0712C91C00D649463973D2D64364FF502184B053D136FA4D797E5A5` |
| the tensors | 1329 |
| the quantized layers | **308** (of 358) |
| the FP16-kept layers | 50 |
| the group size | 256 (all 308 layers) |
| the `int8_linear` calls over the 256-item set | **78592** |
| the `int8_linear` calls over the 1024-item set | **314368** |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42) | **0.999634 / 0.998849 / 0.999487** |
| cos mean / min / p5 (1024 items, seed 42) | **0.999640 / 0.998849 / 0.999522** |
| the gate (mean >= 0.99, min >= 0.98) | **PASS** |

**On the shipped code.** Loaded with `CLIPLoader` type `kandinsky5`, with the
loader node imported and no code changed: 308 of 308 declared layers applied, 0
parameters holding raw int8 codes, 0 unmatched layers, 1624 INT8 GEMM calls over
8 prompt encodes, and the conditioning cosine against the source 0.9999075
(worst) to 0.9999310 (best).

**Provenance.** Re-quantising the source under the converter's own arithmetic
(GPU fp32 with TF32 matmul) reproduces the pack's codes except for 652 of them,
each differing by exactly one step, every one of them on a .5 rounding boundary.
The scale's largest relative difference is 3.4e-07; the dequantised layers sit at
0.0087500 median / 0.0104068 maximum relative L2 against the source.

---

## 6. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_qwen3vl.py` | the impact measurement (both towers, the un-rotated injection) |
| `clip_convert/convert_qwen3_vl_4b_impact_convrot_int8.py` | the impact-driven builder (`--protect-list`) |
| `sdxl/verify_qwen3vl.py` | the 256-item / 1024-item acceptance, the loader's own path |
| `impact/impact_qwen3vl_thinking.json` | the measured per-layer drift (git-ignored) |
| `impact/qwen3vl_thinking_protect_50.json` | the adopted FP16-kept list (git-ignored) |
| `md/verdicts/records/impact_qwen3vl_thinking.json` | the shipped copy of the drift record |
| `md/verdicts/records/thinking_pack.json` | the pack record: identity, provenance, gates, load path |
| `md/verdicts/records/thinking_boundary.json` | the rounding-boundary classification of the residuals |
| `md/verdicts/records/qwen3vl_thinking_protect_50.json` | the adopted FP16-kept list, shipped |
| `md/verdicts/ThinkingQwen3VL4B_Pack.md` | the backdoor verdict for this pack |
