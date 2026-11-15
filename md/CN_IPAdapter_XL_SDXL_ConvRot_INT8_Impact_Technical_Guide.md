# SDXL IP-Adapter (basic) - ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the behavioural observable, the
> size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section
> *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter_xl.pth` (702,585,021 B, 144 tensors)
**Pack.** `ip-adapter_xl_hswq_convrot_int8.safetensors` (352,063,023 B, **-49.9 %**)
**SHA256 (source).** `C2E9F1ED4FF231B9F78DE8158D63AC4B031E599D26D524E9171CCB9C16305FC3`
**SHA256 (pack).** `727A7D265F29EF94A8E51F0184415D556130807DA7CD9B00803EDF57F97C3923`

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

**This adapter is the basic SDXL layout**: an `ImageProjModel` and 140 cross-attention projections.
Both parts execute in different ways, so each is measured on the output it actually feeds:

| part | what the runtime does | how a layer's impact is measured |
| :--- | :--- | :--- |
| `image_proj` (1 Linear: `proj` 1280 -> 8192) | runs once inside `ipadapter_execute`, reshaped to 4 tokens x 2048 and LayerNorm'd, then frozen into `patch_kwargs["cond"]` | swap the weight, run the projection, compare the 4x2048 token stack |
| `ip_adapter` (140 Linear: `to_k_ip` / `to_v_ip`, 2048 in) | executed on **every** forward of the UNet cross-attention | swap one weight and measure that Linear's own output on a fixed 2048-wide input |

**The input to `image_proj` is the CLIP-H penultimate CLS token (1280 wide)**, not the 1024-wide
`image_embeds` projection. That distinction matters: the pack's `proj.weight` is 8192 x 1280, so a
1024-d input is rejected by the shape. The measurement below feeds the 1280-d token.

**The injection frame matters.** The pack stores the **rotated** weight and the kernel rotates the
**activation**; the error that reaches the output is therefore `(q*s) @ H - W`, the **original**
frame.

**The acceptance gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98` over the
fixed 256-item set (seed 42), with finite output.

---

## 2. The loader requirement

The pack's weights are `int8_tensorwise` with `convrot` stamps and a group size of 256.

| loader | what it does with this pack | measured |
| :--- | :--- | :--- |
| A1111 | `modules/ipadapter_int8_convrot.py`: strips the sidecar keys, installs the int8 payload as an instance-level forward bound to `comfy_kitchen.int8_linear(convrot=True, groupsize)`, rebuilds `To_KV` from the declared shapes | **141 of 141 layers installed** (1 `image_proj` + 140 `ip_adapter`), 144 float parameters left untouched, forward `[1,4,2048]` finite, **2 kernel calls** |
| ComfyUI `comfyui_ipadapter_plus` | `ipadapter_model_loader` rebuilds the nested dict with `load_torch_file(safe_load=True)` and never reads `comfy_quant` | **140 raw int8 codes**, 0 quantised-backed parameters: this route does not apply the pack |
| Forge | `backend/operations_mixed_precision.py` reads `comfy_quant` at L75-76 and dispatches to the same kernel | not separately measured for this pack |

The route census is `records/cn_ipadapter_xl_hswq_route.json`.

---

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 702,585,021 B, 144 tensors |
| the weight groups | **141** (1 `image_proj.proj` + 140 `ip_adapter`) |
| the `image_proj` part | `proj` 8192x1280, plus `norm` (2048) and `proj.bias` (8192) |
| the token layout | 4 tokens x 2048 (SDXL cross-attention width) |
| the `ip_adapter` part | 70 blocks x (`to_k_ip`, `to_v_ip`), 2048 in, outputs 640 / 1280 |
| the group size | 256, all INT8 layers |
| the tensors kept as-is | 3 (`norm.weight`, `norm.bias`, `proj.bias`) |

---

## 4. The impact measurement

`clip_convert/diag_impact_ipadapter_xl.py` replaces one group at a time with its ConvRot INT8
reconstruction and measures the relative MSE of that part's own output, in float32.

| rank | layer | drift (rel_mse) |
| :--- | :--- | ---: |
| 1 | `ip_adapter.63.to_k_ip` | 7.817284e-05 |
| 2 | `ip_adapter.91.to_k_ip` | 7.633302e-05 |
| 3 | `ip_adapter.105.to_k_ip` | 7.104035e-05 |
| 4 | `ip_adapter.137.to_k_ip` | 7.060644e-05 |
| 5 | `ip_adapter.83.to_k_ip` | 6.912488e-05 |
| 6 | `ip_adapter.85.to_v_ip` | 6.774781e-05 |
| - | `image_proj.proj` | 3.094009e-05 |

| summary | measured |
| :--- | :--- |
| groups measured | **141** (0 not quantisable, 0 unmeasured) |
| largest drift | **7.817284e-05** (`ip_adapter.63.to_k_ip`) |
| median | 5.050932e-05 |
| p90 | 6.119710e-05 |
| smallest | 2.709644e-05 |
| groups above 0.15 | **0** |

**Every group is three to four orders of magnitude below the 0.15 threshold, and the spread is
narrow** - the largest is under three times the smallest. This adapter has no drift-sensitive layer
at the scale the method uses.

---

## 5. The candidates, and the adopted set

| item | measured |
| :--- | :--- |
| groups swept | 141 |
| above 0.15 | **0** |
| INT8 quantised | **141** |
| FP16 kept | **0** |
| kept by the shape rule | **0** |
| tensors kept that are not quantiser inputs | 3 |

**Nothing is protected, and the shape rule did not have to protect anything.** The inner dimensions
are 1280 (`proj`) and 2048 (`to_k_ip` / `to_v_ip`), both divisible by 256, so every measured group
could be quantised. The 3 kept tensors are the two LayerNorm vectors and `proj.bias`.

---

## 6. The pack

| item | measured |
| :--- | :--- |
| the file | `ip-adapter_xl_hswq_convrot_int8.safetensors` |
| the size | 352,063,023 B (335.75 MiB) |
| against the source | 702,585,021 B - **-49.9 %** |
| the tensors | 426 |
| the INT8 layers | 141 |
| the FP16 kept layers | 0 |
| the group size | 256 |
| the verdict | FAITHFUL |

| check | measured |
| :--- | :--- |
| declared / found | **141 / 141** |
| integers differing from the replay | **0** |
| largest integer delta | **0** |
| scale exact relative max | **0.0** |
| non-finite scales | 0 |
| float tensors checked | **3** |
| float mismatches | **0** (bit-identical) |

---

## 7. The acceptance

**On the A1111 ConvRot INT8 route** (141 layers installed, kernel counted):

| item | measured |
| :--- | :--- |
| `image_proj` token-stack cos | **0.999944299** |
| INT8 kernel calls behind it | **8** |
| batches 0 / 1 / 2 / 3 | 0.999939796 / 0.999941850 / 0.999943407 / 0.999941472 |
| 256-item cos mean | **0.999932818** |
| 256-item cos min | **0.999910579** |
| p5 / p50 | 0.999920450 / 0.999933149 |
| items scored | **256** |
| INT8 kernel calls over the gate | **512** |
| finite | yes |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** |

**The trigger set, on the INT8 route.** 55 candidate image transforms against the
natural-transform control distribution:

| item | measured |
| :--- | :--- |
| candidates | 55 (45 patch, 6 frequency, 4 natural controls) |
| control max | 7.377439e-05 |
| control p99 (max x 1.5) | **1.106616e-04** |
| candidates above p99 | **0** |
| positive control | `weight_scale` ladder on `image_proj_model.proj`, then an int8 code ladder |
| detection floor | **0.1 % of the layer's codes shifted by 8** (1-cos 7.594861e-04) |

The scale ladder did not resolve: x1.0001 through x1.10 all stayed between 5.49e-05 and 5.95e-05,
below the 1.106616e-04 threshold. The code ladder did: 0.1 % of the codes shifted by 8 gives
7.594861e-04, 1 % gives 6.152763e-03, 10 % gives 3.589648e-02. **The instrument is armed on the code
axis, not on the scale axis**, and that distinction is stated rather than collapsed.

**The render, on A1111.** 1024x1024, 12 steps, cfg 7.0, DPM++ 2M Karras, seed 42, weight 0.8,
preprocessor `ip-adapter_clip_sdxl`, base `pornmasterPro_realismILV4_r32_r0.2`, VAE
`sdxl_vae.safetensors`:

| arm | adapter | PNG | wall |
| :--- | :--- | ---: | ---: |
| source | `ip-adapter_xl` | 1,438,325 B | 155.8 s |
| pack | `ip-adapter_xl_hswq_convrot_int8` | 1,437,344 B | 232.9 s |

| source against pack | measured |
| :--- | :--- |
| mean absolute pixel difference | **8.76 / 255** |
| largest single-channel difference | 240 |
| pixels that changed at all | 87.65 % |

Both images were inspected: the same street scene, the same figure, colour normal, no artefacts.
The surfaces differ, which is what a weight-only quantisation of the conditioning path does.

---

## 8. Reproduction

```
#   python = <A1111>/venv/Scripts/python.exe  (or <ComfyUI>/python_embeded/python.exe)
#   COMFY  = <ComfyUI>

# 1. the impact sweep (141 groups, seed 42, float32)
python clip_convert/diag_impact_ipadapter_xl.py \
    --input <src>/ip-adapter_xl.pth \
    --out   impact/impact_ipadapter_xl.json --seed 42

# 2. the builder
python clip_convert/convert_ipadapter_convrot_int8.py \
    --input  <src>/ip-adapter_xl.pth \
    --impact impact/impact_ipadapter_xl.json \
    --output <out>/ip-adapter_xl_hswq_convrot_int8.safetensors \
    --drift-threshold 0.15

# 3. CN-1..CN-3 (faithfulness, loader census, kernel)
python audit/ipadapter_faceid_pack_audit.py \
    --source <src>/ip-adapter_xl.pth --pack <out>/<pack>.safetensors \
    --comfy <COMFY> --node <COMFY>/custom_nodes/comfyui_ipadapter_plus \
    --json records/cn_ipadapter_xl_pack_audit.json

# 4. the route census
python audit/ipadapter_faceid_route_probe.py \
    --pack <out>/<pack>.safetensors --root <A1111> --which a1111 \
    --json records/cn_ipadapter_xl_hswq_route.json

# 5. the render (A1111 API, source and pack arms; SDXL + sdxl_vae)
python audit/ipadapter_faceid_a1111_render.py \
    --image <ref>.png --source-model "ip-adapter_xl [hash]" \
    --pack-model "ip-adapter_xl_hswq_convrot_int8 [hash]" \
    --module ip-adapter_clip_sdxl --width 1024 --height 1024 \
    --out-dir <dir> --json records/cn_ipadapter_xl_a1111_image_test.json

# 6. CN-5a / 5b / 6 on the INT8 route
python audit/ipadapter_faceid_int8_route_acceptance.py \
    --root <A1111> --comfy <COMFY> \
    --source <src>/ip-adapter_xl.pth --pack <out>/<pack>.safetensors \
    --image <ref>.png --clip-vision <COMFY>/models/clip_vision/clip_vision_h.safetensors \
    --json-5a records/cn_ipadapter_xl_pack_source_output_diff.json \
    --json-trigger records/cn_ipadapter_xl_image_trigger.json \
    --json-gate records/cn_ipadapter_xl_gate_256.json
```

---

## 9. Error sources and limits

- **The ComfyUI route does not take this pack.** 140 raw int8 codes reach the model and 0 layers are
  quantised-backed. All acceptance numbers come from the A1111 route; the render does too.
- **Forge was not separately measured for this pack.**
- **The trigger's detection floor is on the code axis only.** The scale ladder did not resolve even
  at x1.10, so a statement about "a 10 % weight change" cannot be made for this instrument; the
  measured floor is 0.1 % of the codes shifted by 8.
- **The render is one prompt, one seed, one checkpoint.** A mean absolute difference of 8.76 / 255
  says the two images are not identical; it does not say whether that matters for a given use.
- **The sweep is one seed and one fixed input per group.** The spread across the 141 groups is under
  three-fold, so the ranking is stable, but the ordering inside the top decile is not.
- **`image_proj` is a single group.** Its 8192x1280 weight is one ConvRot layer; the method cannot
  protect part of it.

---

## 10. The experiment ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `diag_impact_ipadapter_xl.py` | 141 groups, max 7.817284e-05, median 5.050932e-05, p90 6.119710e-05, min 2.709644e-05, 0 above 0.15, 1.0 s |
| 2 | `convert_ipadapter_convrot_int8.py` | 141 INT8, 0 protected, 3 non-quantiser tensors, 352,063,023 B, -49.9 % |
| 3 | `ipadapter_faceid_pack_audit.py` | CN-1 FAITHFUL 141/141, 0 integer differences, scale 0.0, float 3/3 bit-identical; CN-2 ComfyUI 140 raw codes; CN-3 141 INT8 GEMM, finite |
| 4 | `ipadapter_faceid_route_probe.py` | A1111: 141/141 installed (1 + 140), 144 float params, forward `[1,4,2048]` finite, 2 kernel calls |
| 5 | `ipadapter_faceid_a1111_render.py` | source 1,438,325 B / pack 1,437,344 B, both inspected, mean abs diff 8.76 / 255 |
| 6 | `ipadapter_faceid_int8_route_acceptance.py` | CN-5a 0.999944299 over 8 kernel calls; CN-5b 55 candidates, control p99 1.106616e-04, flagged 0, floor 0.1 % codes +8; CN-6 mean 0.999932818, min 0.999910579, n 256, 512 kernel calls, PASS |

---

## 11. Files

| file | what it is |
| :--- | :--- |
| `records/cn_ipadapter_xl_pack_audit.json` | the quantisation replay (141 declared / 141 found, 0 differing) |
| `records/cn_ipadapter_xl_protection.json` | the 141-group sweep and its ranking |
| `records/cn_ipadapter_xl_pack_source_output_diff.json` | CN-5a on the INT8 route |
| `records/cn_ipadapter_xl_gate_256.json` | CN-6, the 256-item gate |
| `records/cn_ipadapter_xl_image_trigger.json` | CN-5b, the 55 candidates and both positive-control ladders |
| `records/cn_ipadapter_xl_a1111_image_test.json` | the two render arms and their comparison |
| `records/cn_ipadapter_xl_a1111_source.png` | the source render |
| `records/cn_ipadapter_xl_a1111_pack.png` | the pack render |
| `records/cn_ipadapter_xl_hswq_route.json` | which loaders take the pack |
