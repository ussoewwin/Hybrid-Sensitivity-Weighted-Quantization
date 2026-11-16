# SDXL IP-Adapter Plus (ViT-H) - ConvRot INT8 Impact Technical Guide

> **The conversion record** for this pack (what was quantised, the protection ranking, the
> conversion arguments, the faithfulness replay and the score) is
> [`IPAdapter_Plus_SDXL_ViTH_ConvRot_INT8_Quantization_and_Score_Report.md`](IPAdapter_Plus_SDXL_ViTH_ConvRot_INT8_Quantization_and_Score_Report.md).
>
> **The method this pack was built under** (the injection frame, the behavioural observable, the
> size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section
> *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter-plus_sdxl_vit-h.safetensors` (847,517,512 B, 191 tensors)
**Pack.** `ip-adapter-plus_sdxl_vit-h_hswq_convrot_int8.safetensors` (424,750,694 B, **-49.9 %**)
**SHA256 (source).** `3F5062B8400C94B7159665B21BA5C62ACDCD7682262743D7F2AEFEDEF00E6581`
**SHA256 (pack).** `CA0FF3706DC3673538EEC2683774E384FE3145F2191F2006E03862A7D3C2DB7A`

---

## Table of contents

1. [What the score is](#1-what-the-score-is)
2. [The loader requirement](#2-the-loader-requirement)
3. [The model](#3-the-model)
4. [The impact measurement](#4-the-impact-measurement)
5. [The candidates, and the adopted set](#5-the-candidates-and-the-adopted-set)
6. [The pack](#6-the-pack)
7. [The acceptance](#7-the-acceptance)
8. [Reproduction](#8-reproduction)
9. [Error sources and limits](#9-error-sources-and-limits)
10. [The experiment ledger](#10-the-experiment-ledger)
11. [Files](#11-files)

---

## 1. What the score is

The score is **not** a static weight statistic. It is the **observed change of the model's own
output** when one weight group is replaced by its ConvRot INT8 reconstruction.

**This adapter is the Plus layout, not the basic one.** Its `image_proj` is a `Resampler` of depth 4,
not a single Linear, and that changes both how the impact is measured and what the measurement can
see:

| part | what the runtime does | how a layer's impact is measured |
| :--- | :--- | :--- |
| `image_proj` (22 Linears: `proj_in`, 4 attention blocks, 4 feed-forwards, `proj_out`) | runs once inside `ipadapter_execute`, consumes the CLIP-H penultimate hidden states and emits 16 tokens x 2048, which are LayerNorm'd and frozen into `patch_kwargs["cond"]` | swap one weight, run the whole Resampler, compare the 16x2048 latent stack **per latent** and aggregate by the worst latent |
| `ip_adapter` (140 Linear: `to_k_ip` / `to_v_ip`, 2048 in) | executed on **every** forward of the UNet cross-attention | swap one weight and measure that Linear's own output on a fixed 2048-wide input |

**The input to `image_proj` is the CLIP-H penultimate hidden states - the whole (1, 257, 1280)
sequence, not a single token.** A basic SDXL adapter takes one 1280-d token; the Plus Resampler takes
the sequence, and feeding it a single token would measure a different function.

**The injection frame matters.** The pack stores the **rotated** weight and the kernel rotates the
**activation**; the error that reaches the output is therefore `(q*s) @ H - W`, the **original**
frame.

**The two surfaces are measured separately and are not interchangeable.** The `ip_adapter`
projections consume the UNet cross-attention K/V, not the image embedding, so their output does not
depend on the image at all; they are measured once on a fixed input of each projection's own width.
Only the `image_proj` surface is image-conditioned, and only it can be moved by an image trigger.

**The acceptance gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98` over the
fixed 256-item set (seed 42), on both surfaces, with finite output. The item statistic is one value
per item (the mean of its 16 latents) - the same statistic the other IP-Adapter packs were judged on.
The worst single latent is reported beside the gate and not folded into it; Section 9 says why.

---

## 2. The loader requirement

The pack's weights are `int8_tensorwise` with `convrot` stamps and a group size of 256.

| loader | what it does with this pack | measured |
| :--- | :--- | :--- |
| A1111 | `modules/ipadapter_int8_convrot.py`: strips the sidecar keys, installs the int8 payload as an instance-level forward bound to `comfy_kitchen.int8_linear(convrot=True, groupsize)`, rebuilds `To_KV` from the declared shapes | **162 of 162 layers installed** (22 `image_proj` + 140 `ip_adapter`), forward `[1,16,2048]` finite, **22 kernel calls** for one `image_proj` forward |
| ComfyUI `comfyui_ipadapter_plus` | `ipadapter_model_loader` rebuilds the nested dict with `load_torch_file(safe_load=True)` and never reads `comfy_quant` | **162 raw int8 codes**, 0 quantised-backed parameters: this route does not apply the pack |
| Forge | `backend/operations_mixed_precision.py` reads `comfy_quant` and dispatches to the same kernel | not separately measured for this pack |

The route census is `records/cn_ipadapter_plus_vit_h_route.json`. The ComfyUI numbers describe the
stock extension; the HSWQ loader package carries its own IP-Adapter runtime
(`patches/ipadapter_convrot_int8.py`) which installs this pack in place, and that path is not the one
counted here.

---

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 847,517,512 B, 191 tensors |
| the weight groups | **162** (22 `image_proj` + 140 `ip_adapter`) |
| the `image_proj` part | a `Resampler`: `proj_in` 1280x1280, `proj_out` 2048x1280, 4 blocks of (`to_q`, `to_kv`, `to_out` + a 2-layer feed-forward), `latents` (1, 16, 1280), `norm_out` (2048) |
| the Resampler parameters | read off the checkpoint and checked against the node's constructor: `dim` 1280, `depth` 4, `dim_head` 64, `heads` 20, `num_queries` 16, `embedding_dim` 1280, `output_dim` 2048, `ff_mult` 4 |
| the token layout | 16 tokens x 2048 (SDXL cross-attention width) |
| the `ip_adapter` part | 70 blocks x (`to_k_ip`, `to_v_ip`), 2048 in, outputs 640 / 1280 |
| the group size | 256, all INT8 layers |
| the tensors kept as-is | 29 (`latents`, the LayerNorm vectors, the biases) |

The Resampler parameters are not assumed. `diag_impact_ipadapter_plus_vit_h.py` derives them from the
tensor shapes (`latents` gives `num_queries` and `dim`, `layers.*.0.to_q` gives `dim_head * heads`,
`layers.*.1.1` gives `ff_mult`) and loads them into the node's own `image_proj_models.Resampler`, so
the measured path is the runtime path. A shape that does not match raises instead of being coerced.

---

## 4. The impact measurement

`clip_convert/diag_impact_ipadapter_plus_vit_h.py` replaces one group at a time with its ConvRot INT8
reconstruction and measures the relative MSE of that part's own output, in float32.

| rank | layer | drift (rel_mse) |
| :--- | :--- | ---: |
| 1 | `ip_adapter.25.to_k_ip` | 8.488220e-05 |
| 2 | `ip_adapter.21.to_k_ip` | 6.873459e-05 |
| 3 | `ip_adapter.117.to_k_ip` | 6.717949e-05 |
| 4 | `ip_adapter.131.to_k_ip` | 6.683299e-05 |
| 5 | `ip_adapter.61.to_v_ip` | 6.667952e-05 |
| 6 | `ip_adapter.17.to_k_ip` | 6.618118e-05 |

| summary | measured |
| :--- | :--- |
| groups measured | **162** (0 not quantisable, 0 unmeasured) |
| largest drift | **8.488220e-05** (`ip_adapter.25.to_k_ip`) |
| median | 4.826412e-05 |
| p90 | 5.861301e-05 |
| smallest | 1.370495e-08 |
| groups above 0.15 | **0** |

**Every group is three to four orders of magnitude below the 0.15 threshold.** This is a screen, not
a criterion: the spread is wide enough (the smallest is four orders below the largest) that the
ranking is informative, but the drift of a layer does not decide whether that layer is protected.
Section 5 carries the criterion.

---

## 5. The candidates, and the adopted set

The criterion is the **aggregate contribution** (`HSWQ_High_Quality_Packs_ConvRot_INT8.md`, Rule 2):
replace **one** layer by its ConvRot INT8 reconstruction and measure the change in the stage output
that layer feeds.

**On this graph the "restore one layer to float" form of that measurement ranks nothing.** Measured:
starting from the all-INT8 pack and restoring a single `image_proj` layer to its float weight makes
the worst-latent divergence **worse for all 22 layers** (gains from -1.7e-02 to -9.1e-01). A single
layer's precision is not a small perturbation of a residual attention block; changing it changes the
whole block's arithmetic.

So the measurement is made from the other end: the **all-float source** is the reference, and **one**
layer at a time is replaced by its INT8 reconstruction. The contribution of that layer is how much
distance to the source it adds on its own.

| rank | layer | contribution (worst-latent divergence) |
| :--- | :--- | ---: |
| 1 | `image_proj.proj_in` | 1.502143e-04 |
| 2 | `ip_adapter.25.to_k_ip` | 4.238458e-05 |
| 3 | `ip_adapter.21.to_k_ip` | 3.436726e-05 |
| 4 | `ip_adapter.117.to_k_ip` | 3.356947e-05 |
| 5 | `ip_adapter.131.to_k_ip` | 3.340939e-05 |
| 6 | `ip_adapter.61.to_v_ip` | 3.333808e-05 |

| summary | measured |
| :--- | :--- |
| layers measured | **162** |
| largest contribution | **1.502143e-04** (`image_proj.proj_in`) |
| median | 2.409374e-05 |
| smallest | 1.9e-09 (`image_proj.layers.1.0.to_q`) |

**The curve.** Each candidate is built with the converter, measured on the fixed 256-item set through
the A1111 ConvRot INT8 route, and deleted. `K` is the number of layers kept in FP16, taken from the
contribution ranking above.

| K | size | gate mean | gate min | worst single latent | `ip_adapter` mean / min | gate |
| ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| **0** | **405.1 MB** | **0.999739679** | **0.998887861** | **0.986089059** | 0.999922318 / 0.999842750 | **PASS** |
| 1 | 406.6 MB | 0.999837537 | 0.995995511 | 0.938652834 | 0.999922318 / 0.999842750 | PASS |
| 20 | 449.0 MB | 0.999837537 | 0.995995511 | 0.938652834 | 0.999933195 / 0.999842750 | PASS |
| 60 | 541.3 MB | 0.999837537 | 0.995995511 | 0.938652834 | 0.999955528 / 0.999842750 | PASS |
| 162 | 808.3 MB | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 / 1.000000000 | PASS |

K = 0 is the adopted pack, so its row is the acceptance record
(`records/cn_ipadapter_plus_vit_h_gate_256.json`); K = 1 / 20 / 60 are the curve record
(`records/cn_ipadapter_plus_vit_h_curve.json`); K = 162 is the curve's upper end, where every layer is
FP16 and the pack equals the source. The `image_proj` gate column is identical for K = 1 / 20 / 60
because those builds protect `ip_adapter` layers, which that surface does not read.

**The adopted set is K = 0: 162 layers INT8, 0 layers protected, 405.1 MB.**

The curve says three things, and they point the same way:

- **K = 0 already passes with room.** The mean clears the 0.99 gate by 9.74e-03, the min clears the
  0.98 gate by 1.89e-03, and 0 of the 162 groups sits anywhere near the drift threshold.
- **Protection buys almost nothing on the surface it is for, and costs the other one.** Keeping
  `proj_in` in FP16 (K = 1, the largest contribution) adds 1.5 MB and moves the gate mean by +9.8e-05
  - one percent of the gate's own margin - while the worst single latent drops from 0.986 to 0.939.
  K = 20 and K = 60 change nothing on the `image_proj` surface (they protect `ip_adapter` layers,
  which that surface does not read) and add 44 MB and 136 MB.
- **The floor is K = 0.** There is nothing below it: every eligible layer is already INT8.

The `ip_adapter` mean does rise with K (0.999922318 -> 0.999955528 at K = 60), which is the other
surface's own arithmetic, not the image path's. It is reported because it was measured, and it does
not change the choice: 136 MB for +3.3e-05 on a surface already at 0.99992.

---

## 6. The pack

| item | measured |
| :--- | :--- |
| the file | `ip-adapter-plus_sdxl_vit-h_hswq_convrot_int8.safetensors` |
| the size | 424,750,694 B (405.08 MiB) |
| against the source | 847,517,512 B - **-49.9 %** |
| the tensors | 515 |
| the INT8 layers | 162 |
| the FP16 kept layers | 0 |
| the group size | 256 |
| the verdict | FAITHFUL |

| check | measured |
| :--- | :--- |
| declared / found | **162 / 162** |
| integers differing from the replay | **0** |
| largest integer delta | **0** |
| scale exact relative max | **0.0** |
| non-finite scales | 0 |
| float tensors checked | **29** |
| float mismatches | **0** (bit-identical, max abs diff 0.0) |
| dtype differences | 0 |
| coverage: missing / extra / shape mismatch | **0 / 0 / 0** |

The conversion itself is in `records/cn_ipadapter_plus_vit_h_conversion.json` -> `arguments`: the arguments it ran with and the per-layer outcome. Re-running it reproduced this pack bit for bit.

---

## 7. The acceptance

**CN-5a, on the A1111 ConvRot INT8 route** (162 layers installed, kernel counted):

| surface | element | cos mean | cos min | kernel calls |
| :--- | :--- | ---: | ---: | ---: |
| `image_proj` | one latent of the 16x2048 injected stack | **0.995676220** | 0.933289702 | 648 over 4 batches + the base |
| `ip_adapter` | one projection's output on a fixed input of its width | **0.999922318** | 0.999842750 | included above |

The `image_proj` mean is 0.9957 and its worst latent is 0.933. That is the Resampler's own arithmetic:
four attention blocks in series accumulate the per-layer error, and the error of a residual block is
not the sum of its layers' errors. Section 9 states what that does and does not say.

**CN-5b, the trigger search on the image-dependent surface.** 56 candidate image transforms against
the natural-transform control distribution:

| item | measured |
| :--- | :--- |
| candidates | 56 (45 patch, 6 frequency, 5 natural controls including the identity) |
| control max | 6.671030e-02 (the identity - the image unmodified) |
| control p99 (max x 1.5) | **1.000654e-01** |
| candidates above p99 | **0** |
| largest candidate response | 3.147203e-02 (`patch_solid_center_64`), 31 % of the threshold |
| positive control, `image_proj` | code ladder on `proj_out`: 10 % of the codes shifted |
| detection floor, `image_proj` | **+64 LSB** (1-cos 1.770971e-01 against a 1.000654e-01 threshold) |
| positive control, `ip_adapter` | code ladder on `1.to_k_ip` |
| detection floor, `ip_adapter` | **+1 LSB** (1-cos 3.329122e-04) |

Both surfaces are armed. The `image_proj` ladder runs 4.46e-02 (8 LSB) -> 9.48e-03 (16) -> 6.92e-03
(32) -> **1.77e-01 (64, detected)** -> 2.42e-01 (96) -> 2.80e-01 (127): it resolves, and it is
monotone over the range that matters. The `ip_adapter` ladder runs 3.33e-04 (1 LSB, detected) ->
5.90e-01 (128), monotone throughout.

**The scale axis was not usable on this graph and is not reported as if it were.** Every Resampler
layer is followed by a LayerNorm, so scaling a layer's output uniformly is removed by the
normalisation that follows it. The code axis is the instrument's axis.

**CN-6, the gate.**

| item | `image_proj` | `ip_adapter` |
| :--- | ---: | ---: |
| items | 256 | 140 elements |
| statistic | mean of the item's 16 latents | the element's own cosine |
| mean | **0.999739679** | **0.999922318** |
| min | **0.998887861** | **0.999842750** |
| p5 / p50 | 0.999466956 / 0.999780970 | - |
| worst single latent (reported, not gated) | **0.986089059** (`c5_o0_b3` latent 12) | - |
| finite | yes (0 non-finite) | yes |
| INT8 kernel calls over the gate | **5772** | included |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** | **PASS** |

**The render is not measured for this pack.** Section 9 states it as a limit rather than leaving it
implicit.

---

## 8. Reproduction

```
#   python = <ComfyUI>/python_embeded/python.exe
#   COMFY  = <ComfyUI>

# 1. the drift screen (162 groups, seed 42, float32, the node's own Resampler)
python clip_convert/diag_impact_ipadapter_plus_vit_h.py \
    --input <src>/ip-adapter-plus_sdxl_vit-h.safetensors \
    --out   records/cn_ipadapter_plus_vit_h_protection.json --seed 42 \
    --node  <COMFY>/custom_nodes/comfyui_ipadapter_plus

# 2. the criterion: one layer at a time against the all-float source
python clip_convert/measure_ipadapter_plus_vit_h_contrib.py \
    --source <src>/ip-adapter-plus_sdxl_vit-h.safetensors \
    --out    records/cn_ipadapter_plus_vit_h_contrib.json --seed 42 \
    --node   <COMFY>/custom_nodes/comfyui_ipadapter_plus

# 3. the builder (groupsize 256, no protection: K = 0)
python clip_convert/convert_ipadapter_plus_vit_h_convrot_int8.py \
    --input  <src>/ip-adapter-plus_sdxl_vit-h.safetensors \
    --impact records/cn_ipadapter_plus_vit_h_protection.json \
    --output <out>/ip-adapter-plus_sdxl_vit-h_hswq_convrot_int8.safetensors \
    --drift-threshold 0.15

# 4. CN-1 / CN-2 (the replay and the loader's view)
python audit/ipadapter_plus_vit_h_pack_audit.py \
    --source <src>/ip-adapter-plus_sdxl_vit-h.safetensors --pack <out>/<pack>.safetensors \
    --comfy <COMFY> --node <COMFY>/custom_nodes/comfyui_ipadapter_plus \
    --json records/cn_ipadapter_plus_vit_h_pack_audit.json

# 5. the route census (ComfyUI and A1111, with the kernel counted)
python audit/ipadapter_plus_vit_h_route.py \
    --source <src>/ip-adapter-plus_sdxl_vit-h.safetensors --pack <out>/<pack>.safetensors \
    --comfy <COMFY> --root <A1111> --json records/cn_ipadapter_plus_vit_h_route.json

# 6. CN-5a / CN-5b / CN-6 on the INT8 route
python audit/ipadapter_plus_vit_h_acceptance.py \
    --source <src>/ip-adapter-plus_sdxl_vit-h.safetensors --pack <out>/<pack>.safetensors \
    --image <ref>.png --clip-vision <COMFY>/models/clip_vision/clip_vision_h.safetensors \
    --root <A1111> --comfy <COMFY> \
    --json-5a records/cn_ipadapter_plus_vit_h_pack_source_output_diff.json \
    --json-trigger records/cn_ipadapter_plus_vit_h_image_trigger.json \
    --json-gate records/cn_ipadapter_plus_vit_h_gate_256.json

# 7. the curve (K = 0 .. 162), candidate packs built and deleted
python clip_convert/sweep_ipadapter_plus_vit_h.py \
    --source <src>/ip-adapter-plus_sdxl_vit-h.safetensors \
    --contrib records/cn_ipadapter_plus_vit_h_contrib.json \
    --impact  records/cn_ipadapter_plus_vit_h_protection.json \
    --image <ref>.png --clip-vision <COMFY>/models/clip_vision/clip_vision_h.safetensors \
    --root <A1111> --comfy <COMFY> --work <scratch> \
    --out records/cn_ipadapter_plus_vit_h_curve.json --ks 0,1,20,60,162
```

---

## 9. Error sources and limits

- **The ComfyUI extension does not take this pack.** 162 raw int8 codes reach the model and 0 layers
  are quantised-backed. All acceptance numbers come from the A1111 route.
- **The HSWQ loader's own IP-Adapter runtime was not exercised for the acceptance numbers.** It
  installs this pack in place of the stock path; the numbers here are the stock path plus the A1111
  runtime patch.
- **Forge was not separately measured for this pack.**
- **No render.** The A1111 install carries no IP-Adapter model registration and the ComfyUI extension
  refuses the pack, so there is no pair of images to look at for this checkpoint. Nothing here says
  what the pack looks like on canvas; that statement is missing, not implied.
- **The trigger search resolves on a coarser perturbation than the drift it is looking for.** The
  `image_proj` floor is 10 % of the codes shifted by 64 LSB; the largest natural candidate response
  (3.15e-02) is 31 % of the 1.000654e-01 threshold. A trigger smaller than that floor would not be
  seen on this surface. The `ip_adapter` surface resolves at 1 LSB but does not consume the image, so
  it cannot carry an image trigger either.
- **The worst single latent is not monotone in the layers' precision.** Measured: the all-INT8 pack
  measures a worse worst-latent (0.986) than every pack with protected layers (0.939). That is the
  Resampler's error arithmetic - the all-INT8 errors partly cancel at that latent - and it is why the
  worst latent is reported beside the gate rather than used as a pass criterion. It does not mean the
  all-INT8 pack is more accurate layer by layer; the layer-by-layer number is the gate mean, and it
  moves the other way.
- **`image_proj` mean 0.9957 with worst latent 0.933.** Four attention blocks in series accumulate
  error; a residual block's error is not the sum of its layers' errors. The pack is measured, not
  characterised: what a 0.933 worst latent does to a given picture is not answered here.
- **The sweep is one seed and one fixed input per layer.** The contribution spread is wide (1.5e-04 to
  1.9e-09), so the top of the ranking is stable, but the ordering inside the middle is not.
- **`image_proj` cannot be protected part-way.** Each of its 22 weights is one ConvRot layer.

---

## 10. The experiment ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `diag_impact_ipadapter_plus_vit_h.py` | 162 groups, max 8.488220e-05, median 4.826412e-05, p90 5.861301e-05, min 1.370495e-08, 0 above 0.15, 2.2 s |
| 2 | `measure_ipadapter_plus_vit_h_contrib.py` | 162 layers against the all-float source, max 1.502143e-04 (`image_proj.proj_in`), median 2.409374e-05, 38.8 s |
| 3 | `convert_ipadapter_plus_vit_h_convrot_int8.py` | 162 INT8, 0 protected, 29 non-quantiser tensors, 424,750,694 B, -49.9 % |
| 4 | `ipadapter_plus_vit_h_pack_audit.py` | CN-1 FAITHFUL 162/162, 0 integer differences, scale 0.0, float 29/29 bit-identical, coverage 0/0/0; CN-2 ComfyUI 162 raw codes |
| 5 | `ipadapter_plus_vit_h_route.py` | A1111: 162/162 installed (22 + 140), forward `[1,16,2048]` finite, 22 kernel calls; ComfyUI: 162 raw codes, applies_int8 false |
| 6 | `ipadapter_plus_vit_h_acceptance.py` | CN-5a `image_proj` 0.995676220 / 0.933289702, `ip_adapter` 0.999922318 / 0.999842750, 648 kernel calls; CN-5b 56 candidates, control p99 1.000654e-01, flagged 0, floors 64 LSB / 1 LSB; CN-6 mean 0.999739679, min 0.998887861, worst latent 0.986089059, 5772 kernel calls, PASS |
| 7 | `sweep_ipadapter_plus_vit_h.py` | K = 0 / 1 / 20 / 60 / 162 -> 405.1 / 406.6 / 449.0 / 541.3 / 808.3 MB, all PASS, K = 0 adopted |

---

## 11. Files

| file | what it is |
| :--- | :--- |
| `records/cn_ipadapter_plus_vit_h_pack_audit.json` | the quantisation replay (162 declared / 162 found, 0 differing) |
| `records/cn_ipadapter_plus_vit_h_route.json` | which loaders take the pack, and what they do with it |
| `records/cn_ipadapter_plus_vit_h_pack_source_output_diff.json` -> `image_proj_surface` | CN-5a on the INT8 route, both surfaces |
| `records/cn_ipadapter_plus_vit_h_gate_256.json` | CN-6, the 256-item gate, both surfaces |
| `records/cn_ipadapter_plus_vit_h_image_trigger.json` | CN-5b, the 56 candidates and both positive-control ladders |
| `records/cn_ipadapter_plus_vit_h_protection.json` | the 162-group drift screen and its ranking |
| `records/cn_ipadapter_plus_vit_h_contrib.json` | the 162-layer aggregate contribution and its ranking |
| `records/cn_ipadapter_plus_vit_h_curve.json` -> `curve` | the measured size/quality curve |
