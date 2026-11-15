# SD1.5 IP-Adapter FaceID - ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the behavioural observable, the
> size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section
> *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter-faceid_sd15.bin` (96,740,574 B, 294 tensors, 162 weight groups)
**Pack.** `ip-adapter-faceid_sd15_hswq_convrot_int8.safetensors` (65,047,130 B, **-32.8 %**)
**SHA256 (source).** `201344E22E6F55849CF07CA7A6E53D8C3B001327C66CB9710D69FD5DA48A8DA7`
**SHA256 (pack).** `35B5942EE163494B82EC4B9D0797EF511A2552E780D44F9E15A6BAE9EE40FCC6`

The source is distributed by `h94/IP-Adapter-FaceID`; its Hub LFS oid equals the sha256 above, so
the file measured here is the published object, not a local variant.

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

**The observable.** This adapter has no forward of its own. `image_proj` maps a 512-wide InsightFace
embedding through `proj.0` (512 -> 1024), a GELU, and `proj.2` (1024 -> 3072) into the token stack
the cross-attention consumes; the 32 blocks of `to_{q,k,v,out}_lora.{down,up}` are low-rank updates
the node adds into a UNet cross-attention projection. The observable is therefore the stage output
the weight feeds:

- for an `image_proj` weight, the relative MSE of the 3072-wide token stack;
- for a LoRA pair, the relative MSE of that pair's low-rank contribution on a fixed input of the
  block's own input width.

A sampled-image number would require running the UNet with the adapter attached. That is the
node's job and a different measurement, and section 7 records what happens when the node is asked
to do it.

**Why there is no trajectory sweep here.** The ControlNet family measures drift along a full
sampling trajectory because a ControlNet emits ten feature maps that a sampler consumes directly.
An IP-Adapter emits conditioning and a set of LoRA deltas; its "output at step N" is a function of
the UNet it is attached to, not of the adapter alone. Measuring the adapter's own stage outputs is
the only construction that isolates the quantisation from the base model. The InstantID checkpoint
in this repository is a plain ControlNet and is measured with the trajectory instrument for exactly
that reason.

**The injection frame matters.** The pack stores the **rotated** weight and the kernel rotates the
**activation**; the error that reaches the output is therefore `(q*s) @ H - W`, the **original**
frame.

**The acceptance gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98` over the
fixed 256-item set (seed 42), with finite output.

---

## 2. The loader requirement

The pack's weights are `int8_tensorwise` with `convrot` stamps and a group size of 256. A loader
that reads the file with `load_torch_file(safe_load=True)` and never interprets `comfy_quant` hands
the int8 code tensors to the model's float modules.

**Measured on this machine, ComfyUI path:** `comfyui_ipadapter_plus.utils.ipadapter_model_loader`
reads with `load_torch_file(safe_load=True)` and rebuilds `image_proj` / `ip_adapter` from the flat
key names without looking at `comfy_quant`, so **100 of 100 declared ip_adapter weights arrive as raw
int8 codes** (`records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN2.loaded_as_raw_int8_codes`).
The load then fails before a forward: `MLPProjModelFaceId.load_state_dict` raises on the int8
tensors. That path does not take the pack.

**Two other loaders on this machine do take it**, and both were run against this exact file
(`records/cn_ipadapter_faceid_sd15_route.json`):

| loader | what it does | measured |
| :--- | :--- | :--- |
| A1111 | `modules/ipadapter_int8_convrot.py` (12,449 B), IP-Adapter-only: strips `*.weight_scale` / `*.comfy_quant` before the float load, installs the int8 payload as an instance-level forward bound to `comfy_kitchen.int8_linear(convrot=True, groupsize)` | **102 of 102 layers installed** (2 `image_proj` + 100 `ip_adapter`), forward `[1,4,768]` finite, **4 kernel calls** |
| Forge | `backend/operations_mixed_precision.py` reads the marker at L75-76, sets the convrot flag at L118-119, dispatches to the same kernel | marker decoded `int8_tensorwise` / convrot true / groupsize 256, **102 of 102 loaded**, 3-layer forward finite, **3 kernel calls** |

**Correction.** An earlier version of this section said no route existed. It was written after
inspecting ComfyUI alone. A1111 and Forge were not inspected, and both carry a route that runs this
pack. `ComfyUI-HSWQ-Loader-and-Tools` has no IP-Adapter node (measured: zero matches for `ipadapter`
in its `__init__.py`) and ComfyUI stock ships none - that part stands - but the machine is not
limited to ComfyUI.

---

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 96,740,574 B, 294 tensors |
| the weight groups (down/up paired) | 162 |
| the 2D weights | 290 |
| the `image_proj` part | `proj.0` 1024x512, `proj.2` 3072x1024, `norm` 768 (1-D) |
| the `ip_adapter` part | 32 blocks x (`to_q_lora`, `to_k_lora`, `to_v_lora`, `to_out_lora`) |
| the `to_k_ip` / `to_v_ip` projections | 16 + 16, shapes 640/1280 x 768 or 320 x 768 |
| the LoRA rank | 128 (`down` 128 x in, `up` out x 128) |
| the group size | 256, all INT8 layers |

A FaceID adapter carries `to_k_lora.up` **without** a matching `down`: the UNet owns the down
projection for those blocks. The sweep pairs a `down` with its `up` where both exist and measures a
bare weight on its own input width where only one does.

---

## 4. The impact measurement

`clip_convert/diag_impact_ipadapter_faceid_sd15.py` runs each stage, replaces one weight group with
its ConvRot INT8 reconstruction, runs again, and records the relative MSE.

| rank | layer | drift (rel_mse) |
| :--- | :--- | :--- |
| 1 | `ip_adapter.13.to_out_lora.down` | 1.407497e-04 |
| 2 | `ip_adapter.13.to_k_ip` | 1.29e-04 (top decile) |
| 3 | `ip_adapter.15.to_out_lora.down` | 1.23e-04 (top decile) |

| summary | measured |
| :--- | :--- |
| weight groups measured | **162** (0 not quantisable, 0 unmeasured) |
| largest drift | **1.407497e-04** (`ip_adapter.13.to_out_lora.down`) |
| median | 6.752623e-05 |
| p90 | 1.061702e-04 |
| min | below 1e-06 |
| groups above 0.15 | **0** |

**Every group is two to three orders of magnitude below the 0.15 threshold.** Nothing in this
adapter is drift-sensitive at the scale the method uses. The ranking is recorded in
`records/cn_ipadapter_faceid_sd15_protection.json` with the full per-group table.

This is the opposite of the ControlNet case, where `input_hint_block.2` reached 46.85. An adapter
whose largest group moves the observable by 1.4e-04 has no layer that a quality-first selection
would protect.

---

## 5. The candidates, and the adopted set

| item | measured |
| :--- | :--- |
| weight groups swept | 162 |
| groups above 0.15 | **0** |
| INT8 quantised | **102** |
| FP16 kept | **60** |
| kept for drift | **0** |
| kept by the shape rule | **60** |
| tensors kept that are not quantiser inputs | 132 |

The 60 kept groups are kept by the **shape rule**, not by drift: a ConvRot group must be a power of
four in the inner dimension, and these weights have inner dimensions of 128, 320 or 640 - none of
which carries a 256-way group. Their split:

| inner dimension | what it is | groups |
| :--- | :--- | ---: |
| 128 | every LoRA `up` (out x 128) | 32 |
| 320 | `to_*_lora.up` in the 320-wide blocks, `to_q/v_lora.down` in those blocks | 20 |
| 640 | the corresponding weights in the 640-wide blocks | 8 |

**No layer was kept for quality reasons.** The boundary is the shape rule alone, and the sweep says
no layer would have needed protection anyway.

---

## 6. The pack

| item | measured |
| :--- | :--- |
| the file | `ip-adapter-faceid_sd15_hswq_convrot_int8.safetensors` |
| the size | 65,047,130 B (62.0 MB) |
| against the source | 96,740,574 B - **-32.8 %** |
| the tensors | 498 |
| the INT8 layers | 102 |
| the FP16 kept layers | 60 |
| the group size | 256 |
| the verdict | FAITHFUL |

The audit replays the converter's own arithmetic on the source weights and compares element by
element:

| check | measured |
| :--- | :--- |
| INT8 layers declared / found | 102 / 102 |
| integers differing from the replay | **0** |
| largest integer delta | **0** |
| scale exact relative max | **0.0** |
| non-finite scales | 0 |
| float tensors checked | **192** |
| float mismatches | **0**, max abs diff 0.0 |

The float side is bit-identical, not merely close: the 192 kept tensors are the source cast to
float16 with zero difference.

---

## 7. The acceptance

**The observable comparison.** Source against pack, both dequantised into the adapter's own float
modules, four perturbed face embeddings plus every LoRA pair:

| item | measured |
| :--- | :--- |
| `image_proj` token-stack cos | **0.999999525** |
| LoRA pairs checked | **68** |
| LoRA cos min | **0.999961097** |
| LoRA cos mean | 0.99999 |

**The 256-item gate.** 8 crops x 4 orientations x 8 brightness steps from one real face image,
each put through InsightFace and then both adapters:

| item | measured |
| :--- | :--- |
| cos mean | **0.999999655** |
| cos min | **0.999999533** |
| items scored | **222** of 256 |
| finite | yes |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** |

**34 of the 256 transformed crops produced no face detection and are excluded, not scored as
passes.** The count is stated because a gate that silently drops items is not a gate.

**The trigger set.** The divergence `1 - cos` of the token stack between source and pack, under 55
candidate image transforms (45 patch patterns, 6 frequency bands, 4 natural transforms), against
the natural-transform control distribution:

| item | measured |
| :--- | :--- |
| candidates | 55 (45 patch, 6 frequency, 4 natural controls) |
| control max | 4.830383e-07 |
| control p99 (max x 1.5) | **7.240575e-07** |
| candidates above p99 | **0** |
| positive control | 7 rank-1 implants into `image_proj.proj.2` |
| detection floor | **0.01** relative to the weight's Frobenius norm (1-cos 1.049906e-06) |

The instrument is shown to see a 1 %-of-norm implant, and no candidate reaches the control
distribution's edge.

**The render.** 512x512, 12 steps, cfg 7.0, dpmpp_2m karras, seed 42, weight 0.8, on
`cyberrealistic_final`, CLIP vision `CLIP-ViT-H-14`:

| arm | measured |
| :--- | :--- |
| source | latent `[1,4,64,64]`, finite, NaN 0, Inf 0, PNG 384,606 B, 2.5 s - inspected, a photographed-style street scene with the identity applied, no artefacts |
| pack | **fails before sampling**: `MLPProjModelFaceId.load_state_dict` raises on the raw int8 codes |

The image was inspected, not inferred. **No image exists for the pack.** The pack does execute on
A1111 and Forge (section 2), but no full sampling pass with the adapter attached was run, so this
guide records no render for it.

---

## 8. Reproduction

```
#   python = <ComfyUI>/python_embeded/python.exe
#   COMFY  = <ComfyUI>

# 1. the impact sweep (162 weight groups, seed 42)
python clip_convert/diag_impact_ipadapter_faceid_sd15.py \
    --ipadapter <src>/ip-adapter-faceid_sd15.bin \
    --out       impact/impact_ipadapter_faceid_sd15.json --seed 42

# 2. the builder
python clip_convert/convert_ipadapter_convrot_int8.py \
    --input  <src>/ip-adapter-faceid_sd15.bin \
    --impact impact/impact_ipadapter_faceid_sd15.json \
    --output <out>/ip-adapter-faceid_sd15_hswq_convrot_int8.safetensors \
    --drift-threshold 0.15

# 3. CN-1..CN-3 (faithfulness, loader route, kernel)
python audit/ipadapter_faceid_pack_audit.py \
    --source <src>/ip-adapter-faceid_sd15.bin --pack <out>/<pack>.safetensors \
    --comfy <COMFY> --node <COMFY>/custom_nodes/comfyui_ipadapter_plus \
    --json records/cn_ipadapter_faceid_sd15_pack_audit.json

# 4. CN-4 (render, source and pack arms)
python audit/ipadapter_faceid_image_test.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter-faceid_sd15.bin \
    --image <ref>.png --ckpt <base>.safetensors \
    --clipvision <COMFY>/models/clip_vision/CLIP-ViT-H-14.safetensors \
    --comfy <COMFY> --out-dir <dir> --json records/cn_ipadapter_faceid_sd15_image_test.json

# 5. CN-5a, CN-5b, CN-6
python audit/ipadapter_faceid_gate_and_trigger.py \
    --source <src>/ip-adapter-faceid_sd15.bin --pack <out>/<pack>.safetensors \
    --image <ref>.png --comfy <COMFY> \
    --json-5a records/cn_ipadapter_faceid_sd15_pack_source_output_diff.json \
    --json-trigger records/cn_ipadapter_faceid_sd15_image_trigger.json \
    --json-gate records/cn_ipadapter_faceid_sd15_gate_256.json
```

---

## 9. Error sources and limits

- **The ComfyUI IP-Adapter Plus path does not take the pack; A1111 and Forge do.** CN-2 measures 100
  of 100 declared ip_adapter weights arriving as raw int8 codes on the ComfyUI path, and that
  model's `load_state_dict` rejects them. On A1111 all 102 layers are installed and the kernel fires
  (4 calls in one forward); on Forge all 102 load and the kernel fires (3 calls over 3 layers). The
  acceptance numbers in section 7 were produced with the pack's codes dequantised into the source
  structure - they measure the **quantisation**, and they are not re-labelled as numbers from an
  INT8-route forward.
- **The observable is the adapter's stage output, not a sampled image.** It isolates the
  quantisation from the base model. It does not bound the pack's effect on a render, because no
  render of the pack exists here.
- **One face, one base model, one prompt.** The gate is 222 scored items derived from a single
  photograph; a different face or a different base model is not covered.
- **34 of 256 gate items had no detectable face** and are excluded. The gate passes on the 222 that
  were scored; the exclusion itself is a property of the transform set, not of the pack.
- **The control distribution is very tight.** Natural transforms move the token stack by 4.8e-07 or
  less, which makes the 1.5x threshold 7.2e-07. A detector with that threshold is sensitive to
  1 %-of-norm implants, but the tightness also means the control set - four transforms on one image
  - is a thin description of "natural variation".
- **The sweep is one seed.** The ranking comes from one fixed input per weight group. With every
  group two to three orders of magnitude below the threshold, a different seed cannot plausibly
  move a group across 0.15, but the ordering within the top decile is not stable.
- **The shape rule, not drift, sets the boundary.** 60 groups are kept because their inner
  dimension cannot carry a power-of-four group. A different group size would change the count and
  therefore the size/quality trade of this pack.

---

## 10. The experiment ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `diag_impact_ipadapter_faceid_sd15.py` | 162 groups, max 1.407497e-04, median 6.752623e-05, p90 1.061702e-04, 0 above 0.15, 0 not quantisable, 5 s |
| 2 | `convert_ipadapter_convrot_int8.py` | 102 INT8, 60 kept, 132 not quantiser inputs, 65,047,130 B, -32.8 % |
| 3 | `ipadapter_faceid_pack_audit.py` | CN-1 FAITHFUL 102/102, q_mismatch 0, max_q_delta 0, scale 0.0, float 192/192 bit-identical; CN-2 ComfyUI path not applied (100 raw codes); CN-3 102 INT8 calls, all finite |
| 3b | `ipadapter_faceid_route_probe.py` | A1111: 102/102 installed, 4 kernel calls, finite; Forge: 102/102 loaded, 3 kernel calls, finite |
| 4 | `ipadapter_faceid_image_test.py` | CN-4 source renders (384,606 B, 2.5 s, inspected); pack fails at `load_state_dict` |
| 5 | `ipadapter_faceid_gate_and_trigger.py` | CN-5a image_proj cos 0.999999525, 68 LoRA pairs cos_min 0.999961097; CN-5b 55 candidates, control p99 7.240575e-07, flagged 0, floor 0.01; CN-6 mean 0.999999655, min 0.999999533, n 222, PASS |

---

## 11. Files

| file | what it is |
| :--- | :--- |
| `records/cn_ipadapter_faceid_sd15_pack_audit.json` | CN-1 / CN-2 / CN-3 and the per-layer code deltas |
| `records/cn_ipadapter_faceid_sd15_protection.json` | the 162-group sweep and its ranking |
| `records/cn_ipadapter_faceid_sd15_pack_source_output_diff.json` | CN-5a, four batches plus the LoRA pairs |
| `records/cn_ipadapter_faceid_sd15_gate_256.json` | CN-6, the 256-item acceptance |
| `records/cn_ipadapter_faceid_sd15_image_trigger.json` | CN-5b, the 55 candidates and the positive-control ladder |
| `records/cn_ipadapter_faceid_sd15_image_test.json` | CN-4, the source render and the pack's load failure |
| `records/cn_ipadapter_faceid_sd15_image_test.png` | the source render |
| `records/cn_ipadapter_faceid_sd15_route.json` | the route census: ComfyUI does not take the pack, A1111 and Forge do (both measured) |

### Retraction

The version of this guide dated 2026-11-14 is withdrawn. It cited
`records/cn_ipadapter_faceid_sd15_gate_256.json`, which did not exist; it recorded
`294/294` as audited when 102 layers were quantised; it reported `protected: 0` against the
converter's 60; it named a `control_p99` of `0.0`; and it omitted the loader route, the trigger
search and the render entirely.

A second correction, from 2026-11-15: the first rewrite called the loader route "not applied" for
the whole machine. That was measured on ComfyUI only. A1111 and Forge both run this pack with the
kernel firing (section 2); that is now what the record says. Every number above is a fresh measurement against the file whose
sha256 is stated at the top.
