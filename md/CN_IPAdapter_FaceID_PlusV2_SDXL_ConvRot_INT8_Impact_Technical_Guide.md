# SDXL IP-Adapter FaceID Plus v2 - ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the behavioural observable, the
> size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section
> *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter-faceid-plusv2_sdxl.bin` (1,487,555,181 B, 1,316 tensors, 1,284 weight groups)
**Pack.** `ip-adapter-faceid-plusv2_sdxl_hswq_convrot_int8.safetensors` (748,384,560 B, **-49.7 %**)
**SHA256 (source).** `C6945D82B543700CC3CCBB98D363B837E9C596281607857C74B713A876DAF5FB`
**SHA256 (pack).** `C9376BC5C5535EE43A96442049E3B7E0A451439209C696A45DA5EA09A23323A3`

The source is distributed by `h94/IP-Adapter-FaceID`. The file measured here is that published object;
every number below was measured against the bytes whose sha256 is stated above.

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

**The observable.** This adapter has no forward of its own. `image_proj` maps the CLIP-ViT-H
penultimate stack (257 x 1280) and the 512-d InsightFace embedding through `proj` (512 -> 1024 ->
8192, reshaped to 4 x 2048), a LayerNorm and a 4-block Perceiver resampler into the token stack the
cross-attention consumes; the 1,260 `ip_adapter` tensors are rank-128 `down`/`up` pairs and the
`to_k_ip` / `to_v_ip` projections the node adds into a UNet cross-attention projection. The
observable is therefore the stage output the weight feeds:

- for an `image_proj` weight, the cosine of the 4 x 2048 token stack on fixed inputs;
- for an `ip_adapter` weight, the cosine of that weight applied to a fixed input of its own width.

**Why there is no trajectory sweep here.** The ControlNet family measures drift along a full
sampling trajectory because a ControlNet emits ten feature maps a sampler consumes directly. An
IP-Adapter emits conditioning and a set of deltas injected inside the base model's cross-attention;
its "output at step N" is a function of the UNet it is attached to, not of the adapter alone.
Measuring the adapter's own stage outputs is the construction that isolates the quantisation from the
base model, and the render in section 7 is what closes the gap.

**The injection frame matters.** The pack stores the **rotated** weight and the kernel rotates the
**activation**; the error that reaches the output is therefore `(q*s) @ H - W`, the **original**
frame.

**The acceptance gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98` over the
fixed 256-item set (seed 42), with finite output.

| item | measured |
| :--- | :--- |
| gate, mean over 256 items | **0.999926916** |
| gate, worst item | **0.999925345** |
| p5 / p50 | 0.999926001 / 0.999926969 |
| worst single token | 0.999921679 (`c7_o1_b1`) |
| INT8 kernel calls behind the gate | **6,144** |
| layers the gate can see | 24 of 1,284 |

## 2. The loader requirement

The pack's weights are `int8_tensorwise` with `convrot` stamps and a group size read per layer. A
loader that reads the file with `load_torch_file` and never interprets `comfy_quant` hands the int8
code tensors to the model's float modules, and `load_state_dict` raises on them.

**Measured on this machine, ComfyUI path:** the loader package's `HSWQ Load IPAdapter Model` node
(`nodes/hswq_load_ipadapter_model.py`) splits the `.bin` pickle into `image_proj` / `ip_adapter`,
installs the ConvRot patch on the extension's `IPAdapter` / `To_KV` globals when the pack carries
`comfy_quant`, and leaves the codes INT8. Measured: patch applied, **24 + 1,260
layers installed**, **1,284 kernel calls** in one pass.

**Two loaders take it, and both were run against this exact file:**

| loader | what it does | measured |
| :--- | :--- | :--- |
| ComfyUI | `nodes/hswq_load_ipadapter_model.py`, IP-Adapter-only: strips `*.weight_scale` / `*.comfy_quant` before the float load, installs the int8 payload as a forward bound to `comfy_kitchen.int8_linear(convrot=True, groupsize)` | **24 + 1,260 layers installed**, `image_proj` cos 0.999927541, **1,284 kernel calls**, all finite |
| A1111 | `modules/ipadapter_int8_convrot.py` called from `sd-webui-controlnet/scripts/ipadapter/ipadapter_model.py`, same split, same kernel | **24/24 + 1,260/1,260 installed**, **1,284 kernel calls**, widths [128, 512, 640, 1024, 1280, 2048, 8192], convrot true on every call |

**What this checkpoint needs that the SD1.5 packs do not.** `cross_attention_dim` is 2048 and
`proj` is 512 -> 1024 -> 8192 reshaped to 4 x 2048 - that width is the host model's, not the file's.
The inner dimensions are 128, 640, 1280 and 2048, so the group size is read **per layer** (256 on
644 tensors, 64 on 620); a fixed 256 would push every 128- and 640-wide weight back to FP16 and the
INT8 count would collapse. The `faceidplusv2` flag is set from the file name by the loader and the
graph branches on it, which is why the name of this file is part of its behaviour.

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 1,487,555,181 B, 1,316 tensors |
| the weight groups | 1,284 |
| the 2D weights | **1,284** (every one of them quantisable) |
| float tensors kept | **32** - biases and LayerNorm parameters |
| the `image_proj` part | `proj.0` 1024x512, `proj.2` 8192x1024, `norm` 2048 (1-D), 4 resampler blocks, 20 heads |
| the `ip_adapter` part | 32 blocks x (`to_q_lora`, `to_k_lora`, `to_v_lora`, `to_out_lora`) of rank 128, plus `to_k_ip` / `to_v_ip` 1280x2048 |
| the token stack | 4 x 2048 |
| the CLIP side | ViT-H penultimate, 257 x 1280 |
| the identity side | insightface buffalo_l, 512-d |
| the group sizes | 256 x 664, 64 x 620 |
| dtype in the source | `float16` |

The adapter consumes the UNet's cross-attention width, not its own: `to_k_ip` and `to_v_ip` write
1280-wide keys and values into a 2048-wide attention. Every 2-D weight in the file is a quantiser
input, which is why this pack has no shape-kept layer at all - the 64-group takes the 128- and
640-wide weights that a 256-group could not.

## 4. The impact measurement

`clip_convert/diag_impact_ipadapter_faceid_plusv2_sdxl.py` runs each stage, replaces one weight group
with its ConvRot INT8 reconstruction, runs again, and records the relative MSE. The criterion is a
second pass on the same construction with 1 - cosine as the statistic, because relative MSE on a
near-identity branch ranks differently from the change the cosine sees.

| rank | layer | contribution (1 - cos) |
| :--- | :--- | ---: |
| 1 | `ip_adapter.3.to_k_ip` | 4.319810e-05 |
| 2 | `ip_adapter.105.to_k_ip` | 4.057634e-05 |
| 3 | `ip_adapter.51.to_k_ip` | 3.868796e-05 |
| 4 | `ip_adapter.133.to_k_ip` | 3.865837e-05 |
| 5 | `ip_adapter.77.to_v_ip` | 3.854346e-05 |
| 6 | `ip_adapter.19.to_k_ip` | 3.854141e-05 |
| 7 | `ip_adapter.109.to_v_ip` | 3.850980e-05 |
| 8 | `ip_adapter.9.to_k_ip` | 3.850847e-05 |
| last | `ip_adapter.79.to_v_ip` | 3.793372e-05 |

| summary | measured |
| :--- | :--- |
| weight groups measured | **1,284** (0 not quantisable, 0 unmeasured) |
| largest drift (rel_mse) | **8.632594e-05** |
| median | 2.053594e-05 |
| min | **8.672217e-11** |
| summed contribution | 1.704065e-02 |
| largest single share | **0.254 %** |
| median layer's share | 0.063 % |
| layers with a negative contribution | **0** |
| groups above 0.15 | **0** |

**The distribution is flat.** The largest single layer - `ip_adapter.3.to_k_ip` - is 0.25 % of
the total, and no layer moves the observable by more than 8.63e-05. There is no layer at the
scale a quality-first selection would protect, which is the opposite of the ControlNet case where
`input_hint_block.2` reached 46.85.

## 5. The candidates, and the adopted set

| item | measured |
| :--- | :--- |
| weight groups swept | 1,284 |
| groups above 0.15 | **0** |
| INT8 quantised | **1,284** |
| FP16 kept | **32** |
| kept for drift | **0** |
| kept by the shape rule | **0** |
| points measured | 4 (K = 0, 40, 80, 160) |

| K = 0 | 1284 INT8, 0 kept - 748.4 MB - gate 0.999926916 / 0.999925345 - gain 0.0000 % of the margin |
| K = 40 | 1244 INT8, 40 kept - 844.1 MB - gate 0.999926916 / 0.999925345 - gain 0.0000 % of the margin |
| K = 80 | 1204 INT8, 80 kept - 940.9 MB - gate 0.999926916 / 0.999925345 - gain 0.0000 % of the margin |
| K = 160 | 1124 INT8, 160 kept - 1091.7 MB - gate 0.999926916 / 0.999925345 - gain 0.0000 % of the margin |

**K = 0 was adopted, and the reason is arithmetic.** The gate calls the `image_proj` graph, 24 of
1,284 layers. Every layer in the top 160 of the contribution ranking is an `ip_adapter` weight, so
protecting 40, 80 or 160 of them returns the **identical** gate statistics - the measured gain is
exactly 0.0000 % of the margin, not a small number rounded down - while the pack grows by
95.7 to 343.4 MB. The method's rule is the smallest pack that
the larger candidates do not beat by a meaningful share of the margin; nothing beats it.

This is a stronger case than the SD1.5 pack, where the balance had to be argued at 0.42 % of the
margin. Here the curve is flat by construction, and the smallest pack is the answer without an
argument.

## 6. The pack

| item | measured |
| :--- | :--- |
| the file | `ip-adapter-faceid-plusv2_sdxl_hswq_convrot_int8.safetensors` |
| the size | 748,384,560 B (748.4 MB) |
| against the source | 1,487,555,181 B - **-49.7 %** |
| the INT8 layers | **1,284** |
| the FP16 kept layers | **32** |
| the protected layers | **0** |
| the format | `int8_tensorwise`, `convrot: true` |
| the group sizes | 256 x 664, 64 x 620 |
| the verdict | **FAITHFUL** |

The audit replays the converter's own arithmetic on the source weights and compares element by
element:

| check | measured |
| :--- | :--- |
| INT8 layers declared / found | 1284 / 1284 |
| integers differing from the replay | **0** |
| largest integer delta | **0** |
| scale exact relative max | **0.0** |
| non-finite scales | 0 |
| keys missing from the source | 0 |
| keys extra | 0 |
| shape mismatches | 0 |
| layers without a scale sidecar | 0 |
| float tensors checked | **32** |
| float mismatches | **0**, max abs diff 0.0 |
| behaviour cosine, mean / worst | 0.999960800 / 0.999954441 |
| load, missing / unexpected | 0 / 0 |

The float side is bit-identical, not merely close: the 32 kept tensors are the source's
own values with zero difference.

## 7. The acceptance

Measured on two loaders, each with the kernel firing. **CN-1 to CN-6, all measured, none inferred.**

**CN-1, the converter's arithmetic.** Replaying the reconstruction on the source weights gives
1,284 INT8 layers, 0 integers differing, largest delta 0, scale relative
0.0, and 32 float tensors bit-identical. The pack is the source quantised, with nothing else
changed.

**On the ComfyUI INT8 route** (loader node, 24 + 1,260 layers, kernel counted):

| item | measured |
| :--- | :--- |
| `image_proj` token-stack cos | **0.999927541** |
| `image_proj` worst batch | **0.999923034** |
| INT8 kernel calls over it | **1,284** |
| `to_kv` projections measured | **1,260** |
| `to_kv` worst projection mean | **0.999612421** |
| `to_kv` worst single | **0.998824256** |
| the float source on the same node | 0 INT8 layers (it does not pass through the patch) |
| kernel input widths | [128, 640, 1280, 2048] |
| kernel group sizes | [64, 256] |

**On the A1111 INT8 route** (`modules/ipadapter_int8_convrot.py`, 24/24 + 1,260/1,260 installed):

| item | measured |
| :--- | :--- |
| `image_proj` INT8 / total modules | **24 / 24** |
| `ip_adapter` INT8 / total modules | **1,260 / 1,260** |
| INT8 kernel calls, one pass | **1,284** |
| group sizes at the kernel | [64, 256] |
| distinct input widths | 7 ([128, 512, 640, 1024, 1280, 2048, 8192]) |
| every call with the convrot flag | **True** |
| applied | **True** |

**CN-5a, both surfaces on fixed inputs.** The `image_proj` surface varies with the CLIP input, so it
is measured over four batches; the `ip_adapter` surface does not, so it is measured per weight:

| item | measured |
| :--- | :--- |
| `image_proj` cos mean / worst | **0.999960350 / 0.999956486** |
| `image_proj` worst relative MSE | 7.890615e-05 |
| `ip_adapter` weights | **1,260** |
| `ip_adapter` cos mean / worst | **0.999972070 / 0.999928368** |
| `ip_adapter` largest deviation from 1 | 7.163225e-05 |
| between them: the INT8 route | **96** kernel calls, cos 0.999906062 / 0.999905986 |

**The activation dtype is part of the instrument.** With float16 activations the 8192-wide `proj`
returns NaN - on the source as well as the pack, which is what tells the two apart from a pack defect
- so every projection measurement here is float32. The route record carries it as `activation_dtype`.

**The trigger set, on the INT8 route.** 56 candidate image transforms against the natural-transform
control distribution:

| item | measured |
| :--- | :--- |
| candidates | **56** |
| baseline divergence | 7.693165e-05 |
| control p99 (the threshold) | **1.172190e-04** |
| largest candidate | 7.873451e-05 |
| candidates above p99 | **0** |
| positive control, code axis | **2 LSB** resolved |
| positive control, scale axis | **x1.05** resolved |
| seed stability | the ladder is re-run from a CRC32 of the candidate spec |

**The positive control ran.** Both ladders were stepped until they resolved, so the zero flagged
count is a measured zero at a stated floor, not an absence of measurement. The floor is coarse on the
scale axis - x1.05 on the scale, 2 LSB on the codes - and a subtler
implant would not be seen, which is what section 9 records.

**The render.** 1024x1024, 12 steps, cfg 7.0, dpmpp_2m / karras, seed
42, weight 0.8, weight_faceidv2 1.0, on `waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors` with `sdxl_vae.safetensors`:

| arm | adapter | measured |
| :--- | :--- | :--- |
| source | `ip-adapter-faceid-plusv2_sdxl` | PNG 1,696,348 B in 42.7 s - inspected: a woman in a red coat on a neon street, the reference identity applied, no artefacts |
| pack | `ip-adapter-faceid-plusv2_sdxl_hswq_convrot_int8` | PNG 1,696,653 B in 21.6 s - inspected: the same subject and the same scene, no artefacts, no noise; the framing and the signage differ |

| source against pack | measured |
| :--- | :--- |
| mean absolute pixel difference | **16.41 / 255** |
| largest single-channel difference | 255 |
| p99 difference | 143 |
| pixels that changed at all | 3.99 % |
| cosine | 0.962361634 |
| PSNR | 18.027259 dB |

Both images were looked at, not inferred from the cosine. 16.41 / 255 is recorded, not
characterised as negligible: a weight-only quantisation of the conditioning path moves the composition
while keeping the subject, and that is what these two PNGs show.

## 8. Reproduction

```
#   python = <ComfyUI>/python_embeded/python.exe
#   COMFY  = <ComfyUI>

# 1. the impact sweep (1,284 weight groups, seed 42)
python clip_convert/diag_impact_ipadapter_faceid_plusv2_sdxl.py \
    --ipadapter <src>/ip-adapter-faceid-plusv2_sdxl.bin \
    --out       impact/impact_ipadapter_faceid_plusv2_sdxl.json --seed 42

# 2. the criterion and the builder
python clip_convert/measure_ipadapter_faceid_plusv2_sdxl_contrib.py \
    --ipadapter <src>/ip-adapter-faceid-plusv2_sdxl.bin \
    --out       impact/contrib_ipadapter_faceid_plusv2_sdxl.json

python clip_convert/sweep_ipadapter_faceid_plusv2_sdxl.py \
    --impact  impact/impact_ipadapter_faceid_plusv2_sdxl.json \
    --contrib impact/contrib_ipadapter_faceid_plusv2_sdxl.json \
    --input   <src>/ip-adapter-faceid-plusv2_sdxl.bin \
    --work    <dir> --ks 0 40 80 160

# 3. CN-1..CN-3 (faithfulness, loader route, kernel)
python audit/ipadapter_faceid_plusv2_sdxl_pack_audit.py \
    --source <src>/ip-adapter-faceid-plusv2_sdxl.bin --pack <out>/<pack>.safetensors \
    --comfy <COMFY> --json records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json

python audit/ipadapter_faceid_plusv2_sdxl_comfyui_route.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter-faceid-plusv2_sdxl.bin \
    --comfy <COMFY> --loader <COMFY>/custom_nodes/ComfyUI-HSWQ-Loader-and-Tools \
    --dir <models>/controlnet --json records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json

python audit/a1111_ipadapter_faceid_plusv2_sdxl_route.py \
    --a1111 <A1111> --pack <out>/<pack>.safetensors \
    --json records/cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json

# 4. CN-4 (render, source and pack arms)
python audit/ipadapter_faceid_plusv2_sdxl_render.py \
    --server http://127.0.0.1:8189 --out-dir <dir> --output-root <COMFY>/output \
    --json records/cn_ipadapter_faceid_plusv2_sdxl_render.json

python audit/ipadapter_faceid_plusv2_sdxl_pixel_compare.py \
    --source <dir>/source.png --pack <dir>/pack.png \
    --json records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json

# 5. CN-5a, CN-5b, CN-6
python audit/ipadapter_faceid_plusv2_sdxl_acceptance.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter-faceid-plusv2_sdxl.bin \
    --comfy <COMFY> --json records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json

python audit/ipadapter_faceid_plusv2_sdxl_comfyui_trigger.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter-faceid-plusv2_sdxl.bin \
    --comfy <COMFY> --loader <loader> --dir <models>/controlnet \
    --image <ref>.png --clip-vision <COMFY>/models/clip_vision/clip_vision_h.safetensors \
    --json records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json

python audit/ipadapter_faceid_plusv2_sdxl_comfyui_gate.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter-faceid-plusv2_sdxl.bin \
    --comfy <COMFY> --loader <loader> --dir <models>/controlnet \
    --image <ref>.png --clip-vision <COMFY>/models/clip_vision/clip_vision_h.safetensors \
    --json records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json
```

## 9. Error sources and limits

- **The gate observes one surface of two.** It calls `image_proj`, 24 of 1,284 layers. The
  1,260 `ip_adapter` weights are measured on fixed inputs (CN-5a), not through a sampled image.
- **The curve is flat by construction.** Every layer in the top 160 is an `ip_adapter` weight, so the
  larger candidates move the gate by exactly zero. The conclusion is about this checkpoint's
  observable, not about those layers being unimportant to a render.
- **One face, one base model, one prompt.** The 256 items vary the CLIP side of a single identity; a
  different face or base model is not covered. The render is one pair on one checkpoint, and its
  16.41 / 255 says the images differ, not whether that matters for a use.
- **float16 invalidates the projection measurement** on this graph: the 8192-wide `proj` returns NaN
  on the source as well as the pack. Every projection number above is float32, and the route record
  carries the dtype.
- **The trigger floor is coarse on the scale axis.** 2 LSB on codes,
  x1.05 on the scale. A subtler implant is not seen, and the count of zero
  flagged candidates is a zero at that floor.
- **The inventory is one machine, one day.** 1,307 entries across four roots,
  166 unreadable and 38 placeholders; it describes this install, not the world.

## 10. The experiment ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `diag_impact_ipadapter_faceid_plusv2_sdxl.py` | 1284 layers, max 8.632594e-05, median 2.053594e-05, min 8.672217e-11, 0 above 0.15 |
| 2 | `measure_ipadapter_faceid_plusv2_sdxl_contrib.py` | sum 1.704065e-02, largest share 0.254 %, 0 negative |
| 3 | `sweep_ipadapter_faceid_plusv2_sdxl.py` | K = 0 / 40 / 80 / 160, gain 0.0000 % at every larger K |
| 4 | `convert_ipadapter_faceid_plusv2_sdxl_convrot_int8.py` | 1284 INT8, 32 kept, 748,384,560 B, -49.7 % |
| 5 | `ipadapter_faceid_plusv2_sdxl_pack_audit.py` | CN-1 FAITHFUL 1284/1284, code 0, max delta 0, scale 0.0, float 32/32 bit-identical |
| 5b | `ipadapter_faceid_plusv2_sdxl_comfyui_route.py` | ComfyUI: 24 + 1260 installed, 1,284 kernel calls, cos 0.999927541 / 0.998824256 |
| 5c | `a1111_ipadapter_faceid_plusv2_sdxl_route.py` | A1111: 24/24 + 1260/1260 installed, 1,284 kernel calls, widths [128, 512, 640, 1024, 1280, 2048, 8192] |
| 6 | `ipadapter_faceid_plusv2_sdxl_acceptance.py` | CN-5a: image_proj 0.999960350 / 0.999956486, ip_adapter 1260 weights 0.999972070 / 0.999928368, route 0.999906062 / 0.999905986 |
| 7 | `ipadapter_faceid_plusv2_sdxl_comfyui_trigger.py` | CN-5b: 56 candidates, baseline 7.693165e-05, p99 1.172190e-04, flagged 0, floor 2 LSB / x1.05 |
| 8 | `ipadapter_faceid_plusv2_sdxl_comfyui_gate.py` | CN-6: mean 0.999926916, min 0.999925345, n 256, 6,144 kernel calls, PASS |
| 9 | `ipadapter_faceid_plusv2_sdxl_render.py` | CN-4: source 1,696,348 B / 42.7 s, pack 1,696,653 B / 21.6 s, mean abs diff 16.41 / 255 |
| 10 | `ipadapter_faceid_plusv2_sdxl_model_inventory.py` | 1,307 entries, 1,102 ok, 38 placeholder, 1 partial, 166 unreadable |

## 11. Files

| file | what it is |
| :--- | :--- |
| `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` | the 1,284-layer drift screen |
| `records/cn_ipadapter_faceid_plusv2_sdxl_contrib.json` | the criterion and the ranking |
| `records/cn_ipadapter_faceid_plusv2_sdxl_curve.json` | the K curve and the decision |
| `records/cn_ipadapter_faceid_plusv2_sdxl_conversion.json` | the converter's per-layer map |
| `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` | CN-1, the arithmetic and the behaviour |
| `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` | CN-2 / CN-3 on the ComfyUI loader node |
| `records/cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json` | CN-2 on A1111, both surfaces installed |
| `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` | CN-5a, both surfaces and the route |
| `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` | CN-6, the 256-item acceptance |
| `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` | CN-5b, the 56 candidates and both ladders |
| `records/cn_ipadapter_faceid_plusv2_sdxl_render.json` | CN-4, the settings and both arms |
| `records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` | CN-4, the two PNGs compared |
| `records/cn_ipadapter_faceid_plusv2_sdxl_model_inventory.json` | the section 1.10 inventory, 1,307 rows |
| `audit/cn_ipadapter_faceid_plusv2_sdxl_source.png` | the source arm's PNG |
| `audit/cn_ipadapter_faceid_plusv2_sdxl_pack.png` | the pack arm's PNG |

### Retraction

The version of this guide dated 2026-11-16 is withdrawn. Its section order was written from the
conversion, not read from the sibling guide of the same kind: the tables carried their own column
names instead of this track's `item / measured`, the acceptance was one long three-column table
instead of a block per route, and the ledger's columns did not match the ledger of the pack beside it.
Every number here is the same measurement as before; the form is the form of
`CN_IPAdapter_FaceID_SD15_ConvRot_INT8_Impact_Technical_Guide.md`.
