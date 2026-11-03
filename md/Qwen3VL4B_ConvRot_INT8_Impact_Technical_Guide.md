# Qwen3-VL-4B Thinking — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Source.** `models/clip/Qwen3_VL_4B_Thinking_abliterated.safetensors` (8464.5 MB, 713 keys, 358 finite 2D Linear layers)
**Pack.** `models/clip/Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` (4914.0 MB, **−41.9 %**)
**SHA256.** `865948A2E5029E67548A946CF019733A56EA04EEF29F11378B6816E7045C5CF8`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate** output.

**The withdrawn builds.** Two earlier versions of this pack are withdrawn:

1. the build that stood here before the first one (sha
   `49C19DAD012A8CF4152CE01D1FF6948989C6C0AD1B7AF45F673B84B6FBB1D522`, 5958.9 MB, 100
   FP16-kept) - its protection set was this checkpoint's own drift ranking, but the kept count was
   taken from a three-point curve (60/100/150) whose "flattens at 100" reading was never tested
   below 60;
2. the build published in commits `494fd8f` / `63503d8` / `160ffb2` (sha `07BD6D89…`,
   5297.3 MB, 50 FP16-kept) - the same protection set, but the kept count was chosen on a band
   criterion ("the worst-prompt cosine of the candidates with K >= 60") that the method does not
   define, spending 383.3 MiB on +9.7e-05 of minimum and +4.3e-05 of fifth percentile over the
   smallest pack that passes.

The pack below is the **smallest pack that passes the gate** on this checkpoint's own measured
curve, and the floor under it is measured, not assumed.

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

Script: `clip_convert/diag_impact_qwen3vl.py` -> `impact/impact_qwen3vl_thinking.json`
(the shipped copy is `md/backdoor/verdicts/records/impact_qwen3vl_thinking.json`)

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

| the FP16-kept | INT8 layers | cos mean | cos min | cos p5 | size | gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 358 | 0.997745 | 0.955591 | 0.997012 | 4240.3 MB | FAIL |
| 10 | 348 | 0.999200 | 0.915881 | 0.999303 | 4767.9 MB | FAIL |
| **20** | **338** | **0.999602** | **0.997877** | **0.999442** | **4914.0 MB** | **PASS** |
| 30 | 328 | 0.999613 | 0.996474 | 0.999487 | 5073.8 MB | PASS |
| 40 | 318 | 0.999625 | 0.998508 | 0.999486 | 5171.2 MB | PASS |
| 50 | 308 | 0.999634 | 0.998849 | 0.999487 | 5297.3 MB | PASS |
| 60 | 298 | 0.999638 | 0.998873 | 0.999489 | 5415.9 MB | PASS |
| 80 | 278 | 0.999643 | 0.998794 | 0.999500 | 5666.8 MB | PASS |
| 100 | 258 | 0.999647 | 0.998933 | 0.999502 | 5958.9 MB | PASS |
| 120 | 238 | 0.999653 | 0.998841 | 0.999534 | 6209.8 MB | PASS |
| 150 | 208 | 0.999669 | 0.998911 | 0.999523 | 6676.8 MB | PASS |

Seven candidates (K=0/10/20/30/40/50/100) were re-measured on the **1024-item** set, so the
worst prompt is placed on a larger sample:

| the FP16-kept | INT8 layers | cos mean | cos min | cos p5 | size | gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 358 | 0.997853 | 0.955591 | 0.997024 | 4240.3 MB | FAIL |
| 10 | 348 | 0.999451 | 0.915881 | 0.999333 | 4767.9 MB | FAIL |
| **20** | **338** | **0.999610** | **0.997877** | **0.999479** | **4914.0 MB** | **PASS** |
| 30 | 328 | 0.999625 | 0.996474 | 0.999514 | 5073.8 MB | PASS |
| 40 | 318 | 0.999633 | 0.998508 | 0.999509 | 5171.2 MB | PASS |
| 50 | 308 | 0.999640 | 0.998849 | 0.999522 | 5297.3 MB | PASS |
| 100 | 258 | 0.999654 | 0.998933 | 0.999542 | 5958.9 MB | PASS |

**20 FP16-kept layers were adopted.** The floor under it is measured, not assumed: K=0 fails
with min 0.955591 and K=10 fails with min 0.915881, on both acceptance sets, so K=20 is the first
count that clears the gate. Everything the larger candidates buy is below the resolution the gate
is about - from K=20 to K=100 the mean moves +4.4e-05, the fifth percentile +6.3e-05 and the worst
prompt +1.1e-04, for 1044.9 MiB more. Against the withdrawn K=50 build, K=20 gives up 9.7e-05 of
minimum and 4.3e-05 of fifth percentile for **383.3 MiB less**.

The worst prompt is not monotone in K (K=30's minimum, 0.996474, is below K=20's 0.997877): the
minimum is one prompt out of 256 or 1024, and a different protection count puts the error on a
different prompt. The mean and the fifth percentile are the stable columns, and they are flat from
K=20 upward.

---

## 5. The pack

Built with `clip_convert/convert_qwen3_vl_4b_impact_convrot_int8.py`:

```
--impact impact/impact_qwen3vl_thinking.json --protect-list impact/qwen3vl_thinking_protect_20.json
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
| the pack | **4914.0 MB** (-41.9 %) |
| the SHA256 | `865948A2E5029E67548A946CF019733A56EA04EEF29F11378B6816E7045C5CF8` |
| the tensors | 1389 |
| the quantized layers | **338** (of 358) |
| the FP16-kept layers | 20 |
| the group size | 256 (all 338 layers) |
| the `int8_linear` calls over the 256-item set | **86272** |
| the `int8_linear` calls over the 1024-item set | **345088** |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42) | **0.999602 / 0.997877 / 0.999442** |
| cos mean / min / p5 (1024 items, seed 42) | **0.999610 / 0.997877 / 0.999479** |
| the gate (mean >= 0.99, min >= 0.98) | **PASS** |

**On the shipped code.** Loaded with `CLIPLoader` type `kandinsky5`, with the
loader node imported and no code changed: 338 of 338 declared layers applied, 0
parameters holding raw int8 codes, 0 unmatched layers, 1864 INT8 GEMM calls over
8 prompt encodes, and the conditioning cosine against the source 0.9999003
(worst) to 0.9999313 (best).

**Provenance.** Re-quantising the source under the converter's own arithmetic
(GPU fp32 with TF32 matmul) reproduces the pack's codes except for 751 of the
3,728,211,968 int8 codes, each differing by exactly one step, every one of them
on a .5 rounding boundary. The scale's largest relative difference is 3.4e-07;
the dequantised layers sit at 0.0087500 median / 0.0106538 maximum relative L2
against the source.

---

## 6. Files

| file | role |
| :--- | :--- |
| `clip_convert/diag_impact_qwen3vl.py` | the impact measurement (both towers, the un-rotated injection) |
| `clip_convert/convert_qwen3_vl_4b_impact_convrot_int8.py` | the impact-driven builder (`--protect-list`) |
| `clip_convert/verify_qwen3vl.py` | the 256-item / 1024-item acceptance, the loader's own path |
| `impact/impact_qwen3vl_thinking.json` | the measured per-layer drift (git-ignored) |
| `impact/qwen3vl_thinking_protect_20.json` | the adopted FP16-kept list (git-ignored) |
| `md/backdoor/verdicts/records/impact_qwen3vl_thinking.json` | the shipped copy of the drift record |
| `md/backdoor/verdicts/records/qwen3vl_thinking_protect_20.json` | the shipped copy of the adopted FP16-kept list |
| `md/backdoor/verdicts/records/thinking_pack.json` | the pack record: identity, provenance, gates, load path, environment, integrity |
| `md/backdoor/verdicts/records/thinking_boundary.json` | the rounding-boundary classification of the residuals |
| `md/backdoor/verdicts/ThinkingQwen3VL4B_Pack.md` | the backdoor verdict for this pack |
