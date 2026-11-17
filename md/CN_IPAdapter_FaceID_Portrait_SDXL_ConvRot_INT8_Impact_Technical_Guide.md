# SDXL IP-Adapter FaceID Portrait (unnorm) - ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the behavioural observable, the
> size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section
> *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter-faceid-portrait_sdxl_unnorm.bin` (1,009,523,411 B, 191 tensors, 162 weight groups)
**Pack.** `ip-adapter-faceid-portrait-unnorm_sdxl_hswq_convrot_int8.safetensors` (423,767,654 B, **-58.0 %**)
**SHA256 (source).** `220BB86E205393A3D0411631CB473CADDBF35FD371BE2905CA9008818170DB55`
**SHA256 (pack).** `54B10126EFD6205510AF3764C6EE53216064B52EB48267F5234E21C005994C38`

The source is distributed by `h94/IP-Adapter-FaceID`; the file measured here is the published object.

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

The score is **not** a static weight statistic. It is the **observed change of the model's own output**
when one weight group is replaced by its ConvRot INT8 reconstruction.

**The observable.** This adapter has no forward of its own. `image_proj` is a 16-query `Resampler` over the
unnormalised 512-d ArcFace embedding: `proj_in` (1280, 512), four Perceiver blocks with 20 heads, `proj_out`
(2048, 1280); the 140 `ip_adapter` tensors are bare `to_k_ip` / `to_v_ip` projections (2048 -> 640 or 1280)
the node adds into a UNet cross-attention projection. The observable is therefore the stage output the
weight feeds:

- for an `image_proj` weight, the cosine of the 16 x 2048 token stack on a fixed embedding;
- for a projection, the cosine of that weight applied to a fixed 16-token input of its own width.

**Why there is no trajectory sweep here.** The ControlNet family measures drift along a full sampling
trajectory because a ControlNet emits feature maps a sampler consumes directly. An IP-Adapter emits
conditioning; its "output at step N" is a function of the UNet it is attached to, not of the adapter alone.
Measuring the adapter's own stage outputs isolates the quantisation from the base model, and the render in
section 7 is what closes the gap.

**The injection frame matters.** The pack stores the **rotated** weight and the kernel rotates the
**activation**; the error that reaches the output is therefore `(q*s) @ H - W`, the **original** frame.

**The acceptance gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98` over the fixed
256-item set (seed 42), with finite output.

| item | measured |
| :--- | :--- |
| gate, mean over 256 items | **0.999986747** |
| gate, worst item | **0.999923475** |
| p5 / p50 | 0.999971982 / 0.999989115 |
| worst single token | 0.999555409 (`c0_o2_b0`) |
| INT8 kernel calls behind the gate | **5,632** |
| layers the gate can see | 22 of 162 |

## 2. The loader requirement

The pack's weights are `int8_tensorwise` with `convrot` stamps and a group size of 256 everywhere. A loader
that reads the file without interpreting `comfy_quant` hands the int8 codes to the float modules, and the
load raises on them.

**Measured on this machine, ComfyUI path:** the loader package's `HSWQ Load IPAdapter Model` node reads
`portraitunnorm` from the file name, installs the ConvRot patch, and keeps the codes INT8. Measured: patch
applied, **22 + 140 layers installed**, **162 kernel calls** in one pass.

| loader | what it does | measured |
| :--- | :--- | :--- |
| ComfyUI | `nodes/hswq_load_ipadapter_model.py`, IP-Adapter-only: strips the sidecars, installs the int8 payload bound to `comfy_kitchen.int8_linear(convrot=True, groupsize)` | **22 + 140 layers installed**, `image_proj` cos 0.999984396, **162 kernel calls** |
| A1111 | `sd-webui-controlnet`'s `IPAdapterModel.load` derives `is_plus` from `"latents" in image_proj` and builds `ProjModelFaceIdPlus` | **cannot build**: `load mismatch: missing=['proj.0.weight', ...] unexpected=['latents', ...]` |

**The name decides the graph.** `IPAdapterPlus.py` sets `is_portrait_unnorm = "portraitunnorm" in ipadapter`, and
from that derives `cross_attention_dim = 1280`, `clip_extra_context_tokens = 16`, the `Resampler` branch and the
unnormalised embedding. The pack's file name carries `unnorm` for exactly this reason.

**The input is the unnormalised embedding.** `proj_in` is (1280, 512), so the graph consumes the raw 512-d
ArcFace vector, not a CLIP sequence: measured norm 24.355612 against a normalised
counterpart of 1.000000.

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 1,009,523,411 B, 191 tensors |
| the weight groups | 162 |
| the 2D weights | **162** (every one quantisable) |
| float tensors kept | **29** - biases, LayerNorm parameters and the `latents` query |
| the `image_proj` part | `proj_in` 1280x512, `proj_out` 2048x1280, 4 blocks x (`to_q`, `to_kv`, `to_out`, `1.1`, `1.3`), `latents` (1, 16, 1280) |
| the `ip_adapter` part | 70 `to_k_ip` + 70 `to_v_ip`, 640 or 1280 x 2048 |
| the token stack | 16 x 2048 |
| the identity side | insightface buffalo_l, 512-d unnormalised |
| the group size | 256 on all 162 layers |

There are no LoRA pairs in this checkpoint: every `ip_adapter` entry is a bare projection, which is why the
sweep measures each weight on its own input width rather than pairing a `down` with an `up`.

## 4. The impact measurement

`clip_convert/diag_impact_ipadapter_faceid_portrait_sdxl.py` runs each stage, replaces one weight group with
its ConvRot INT8 reconstruction, runs again, and records the relative MSE. The criterion is a second pass on
the same construction with 1 - cosine as the statistic.

| rank | layer | drift |
| ---: | :--- | ---: |
| 1 | `ip_adapter.105.to_k_ip` (0.877 % of the total) | 4.450684e-05 |
| 2 | `ip_adapter.87.to_k_ip` (0.804 % of the total) | 4.079308e-05 |
| 3 | `ip_adapter.127.to_k_ip` (0.792 % of the total) | 4.018840e-05 |
| 4 | `ip_adapter.5.to_k_ip` (0.785 % of the total) | 3.983512e-05 |
| 5 | `ip_adapter.131.to_k_ip` (0.783 % of the total) | 3.974688e-05 |
| 6 | `ip_adapter.119.to_k_ip` (0.780 % of the total) | 3.961915e-05 |
| 7 | `ip_adapter.113.to_k_ip` (0.774 % of the total) | 3.930200e-05 |
| 8 | `ip_adapter.19.to_k_ip` (0.769 % of the total) | 3.903459e-05 |

| summary | measured |
| :--- | :--- |
| weight groups measured | **162** (0 not quantisable, 0 unmeasured) |
| largest drift | **8.387730e-05** |
| median | 7.013413e-05 |
| min | **6.028451e-10** |
| summed contribution | 5.076462e-03 |
| largest single layer's share | **0.008767294678320766** |
| layers with a negative contribution | **0** |
| groups above 0.15 | **0** |

**The distribution is flat.** The largest single layer is 0.88 % of the total and no layer moves
the observable by more than 8.39e-05. There is no layer at the scale a quality-first selection would protect.

## 5. The candidates, and the adopted set

| item | measured |
| :--- | :--- |
| weight groups swept | 162 |
| groups above 0.15 | **0** |
| INT8 quantised | **162** |
| FP16 kept | **29** |
| kept for drift | **0** |
| kept by the shape rule | **0** |
| points measured | 4 (K = 0, 20, 40, 80) |

| inner dimension | what the group is | groups |
| :--- | :--- | ---: |
| K = 0 | 162 INT8, 0 kept - 423.8 MB - gate 0.999986747 / 0.999923475 - protected share 0.0 % - `54b10126efd6` | 162 |
| K = 20 | 142 INT8, 20 kept - 466.9 MB - gate 0.999986747 / 0.999923475 - protected share 15.4 % - `956fa87c77b4` | 142 |
| K = 40 | 122 INT8, 40 kept - 515.3 MB - gate 0.999986747 / 0.999923475 - protected share 30.1 % - `a7b5dbd4e4ab` | 122 |
| K = 80 | 82 INT8, 80 kept - 616.0 MB - gate 0.999986747 / 0.999923475 - protected share 58.8 % - `3b5add6ab111` | 82 |

**K = 0 is adopted.** The gate calls the `image_proj` graph only, 22 of 162 layers, and every layer in
the top 80 of the contribution ranking is an `ip_adapter` projection the gate never calls. Protecting 20, 40 or
80 of them returns the identical gate statistics while the pack grows from 423.8 to 616.0 MB.
The measured gain is exactly zero, not a small number rounded down.

## 6. The pack

| item | measured |
| :--- | :--- |
| the file | `ip-adapter-faceid-portrait-unnorm_sdxl_hswq_convrot_int8.safetensors` |
| the size | 423,767,654 B (423.8 MB) |
| against the source | 1,009,523,411 B - **-58.0 %** |
| the INT8 layers | **162** |
| the FP16 kept layers | **29** |
| the protected layers | **0** |
| the format | `int8_tensorwise`, `convrot: true` |
| the group sizes | 256 x 162 |
| the verdict | **FAITHFUL** |

| check | measured |
| :--- | :--- |
| INT8 layers declared / found | 162 / 162 |
| integers differing from the replay | **0** |
| largest integer delta | **0** |
| scale exact relative max | **0.0** |
| non-finite scales | 0 |
| keys missing / extra / shape mismatch | 0 / 0 / 0 |
| float tensors bit-identical | **29** |
| of those that differ | **0** |
| behaviour cosine, mean / worst | 0.999994832 / 0.999976210 |

## 7. The acceptance

**CN-1 to CN-6, all measured, none inferred.**

**On the ComfyUI INT8 route** (22 + 140 layers, kernel counted):

| item | measured |
| :--- | :--- |
| `image_proj` token-stack cos | **0.999984396** |
| `image_proj` worst batch | **0.999906724** |
| INT8 kernel calls over it | **162** |
| projections measured | **140** |
| projection worst mean / worst single | **0.999919083** / **0.999862444** |
| the float source on the same node | 0 INT8 layers |

**CN-5a, both surfaces on fixed inputs:**

| item | measured |
| :--- | :--- |
| `image_proj` cos mean / worst | **0.999995405 / 0.999971881** |
| `image_proj` worst relative MSE | 2.346997e-05 |
| `ip_adapter` weights | **140** |
| `ip_adapter` cos mean / worst | **0.999964875 / 0.999962064** |
| between them: the INT8 route | **88** kernel calls, cos 0.999971191 / 0.999967509 |

**CN-5b, three passes, each recorded.** The first compared against a natural-transform band and flagged
2 of 56; the band was too tight (the unperturbed baseline, 7.103098e-05, sat inside it).
The second used the graph re-run as the floor and flagged 39 of 39, because that floor is 1e-07.
The third - the one that decides - compares against a **second honest quantisation of the same checkpoint**
(142 INT8 layers, 20 kept, sha256 `4314b2833a309175`):

| item | measured |
| :--- | :--- |
| candidates | **41** |
| scored (a face was found) | **39** |
| no face - recorded, not scored | 2 |
| the band (reference divergence max) | **8.093933e-05** |
| shipped divergence, median | 7.103098e-05 |
| flagged against the band | **0** |

**The render.** 1024x1024, 12 steps, cfg 7.0, dpmpp_2m / karras, seed 42,
weight 0.5, weight_faceidv2 1.0, on `waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors`:

| arm | adapter | measured |
| :--- | :--- | :--- |
| source | `ip-adapter-faceid-portrait_sdxl_unnorm` | PNG 1,980,889 B in 18.5 s - inspected: a girl in a long red coat on a blue-lit street, finished, no artefacts |
| pack | `ip-adapter-faceid-portrait-unnorm_sdxl_hswq_convrot_int8` | PNG 2,004,275 B in 15.5 s - inspected: the same subject and scene, finished; the coat pattern, signage and reflections differ |

| source against pack | measured |
| :--- | :--- |
| mean absolute pixel difference | **55.717063** / 255 |
| largest single-channel difference | 255 |
| p99 difference | 244 |
| pixel cosine | 0.875642756 |
| PSNR | 9.698528 dB |
| pixels that changed at all | 98.74 % |

Both images were looked at, not inferred from the cosine. Both are finished pictures; 55.717063 / 255 is
recorded, not characterised as negligible.

## 8. Reproduction

```
#   python = <ComfyUI>/python_embeded/python.exe
#   COMFY  = <ComfyUI>

# 1. the impact sweep (162 weight groups, seed 42)
python clip_convert/diag_impact_ipadapter_faceid_portrait_sdxl.py \
    --ipadapter <src>/ip-adapter-faceid-portrait_sdxl_unnorm.bin \
    --out       impact/impact_ipadapter_faceid_portrait_sdxl.json --comfy <COMFY>

# 2. the criterion and the builder
python clip_convert/measure_ipadapter_faceid_portrait_sdxl_contrib.py \
    --ipadapter <src>/ip-adapter-faceid-portrait_sdxl_unnorm.bin \
    --out       impact/contrib_ipadapter_faceid_portrait_sdxl.json --comfy <COMFY>

python clip_convert/sweep_ipadapter_faceid_portrait_sdxl.py \
    --impact  impact/impact_ipadapter_faceid_portrait_sdxl.json \
    --contrib impact/contrib_ipadapter_faceid_portrait_sdxl.json \
    --input   <src>/ip-adapter-faceid-portrait_sdxl_unnorm.bin \
    --work    <dir> --ks 0 20 40 80

# 3. CN-1..CN-3 (faithfulness, loader route, kernel)
python audit/ipadapter_faceid_portrait_sdxl_pack_audit.py \
    --source <src>/ip-adapter-faceid-portrait_sdxl_unnorm.bin --pack <out>/<pack>.safetensors \
    --comfy <COMFY> --json records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json

python audit/ipadapter_faceid_portrait_sdxl_comfyui_route.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter-faceid-portrait_sdxl_unnorm.bin \
    --comfy <COMFY> --loader <COMFY>/custom_nodes/ComfyUI-HSWQ-Loader-and-Tools \
    --dir <models>/controlnet --json records/cn_ipadapter_faceid_portrait_sdxl_comfyui_route.json

python audit/a1111_ipadapter_faceid_portrait_sdxl_route.py \
    --a1111 <A1111> --pack <out>/<pack>.safetensors \
    --json records/cn_ipadapter_faceid_portrait_sdxl_a1111_route.json

# 4. CN-4 (render, source and pack arms)
python audit/ipadapter_faceid_portrait_sdxl_render.py \
    --server http://<comfyui-host> --out-dir <dir> --output-root <COMFY>/output \
    --adapter-dir <models>/controlnet \
    --json records/cn_ipadapter_faceid_portrait_sdxl_render.json

python audit/ipadapter_faceid_plusv2_sdxl_pixel_compare.py \
    --source <dir>/source.png --pack <dir>/pack.png \
    --json records/cn_ipadapter_faceid_portrait_sdxl_pixel_compare.json

# 5. CN-5a, CN-5b, CN-6
python audit/ipadapter_faceid_portrait_sdxl_acceptance.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter-faceid-portrait_sdxl_unnorm.bin \
    --comfy <COMFY> --json records/cn_ipadapter_faceid_portrait_sdxl_pack_source_output_diff.json

python audit/ipadapter_faceid_portrait_sdxl_baseline.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter-faceid-portrait_sdxl_unnorm.bin \
    --comfy <COMFY> --loader <loader> --dir <models>/controlnet --image <ref>.png \
    --work <dir> --impact impact/impact_ipadapter_faceid_portrait_sdxl.json \
    --json records/cn_ipadapter_faceid_portrait_sdxl_trigger_baseline.json

python audit/ipadapter_faceid_portrait_sdxl_comfyui_gate.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter-faceid-portrait_sdxl_unnorm.bin \
    --comfy <COMFY> --loader <loader> --dir <models>/controlnet --image <ref>.png \
    --json records/cn_ipadapter_faceid_portrait_sdxl_gate_256.json
```

## 9. Error sources and limits

- **A1111 cannot build this graph.** Its loader derives `is_plus` from `latents` and builds `ProjModelFaceIdPlus`;
  the load raises. ComfyUI through the loader node is the route measured here.
- **The adapter saturates above weight 0.5 on the generic FaceID node.** 0.0 and 0.5 give finished pictures;
  0.8 gives a lattice of artefacts in **both** arms; 1.0 collapses the source arm. The isolation record shows
  base and weight-0 arms clean, so the break belongs to the conditioning strength, not to the pack.
- **The gate observes one surface of two.** It calls `image_proj`, 22 of 162 layers; the 140 projections are
  measured on fixed inputs.
- **The K curve is flat by construction.** Every layer in the top 80 is a projection the gate never calls.
- **One face, one base model, one prompt, one weight.** The 256 items vary the identity side of one reference.
- **The trigger band is set by another quantisation, not by theory.** A subtler implant than 8.093933e-05 is not
  separated by this instrument; two candidates could not be scored at all because no face was detected on them.

## 10. The experiment ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `diag_impact_ipadapter_faceid_portrait_sdxl.py` | 162 layers, max 8.387730e-05, median 7.013413e-05, min 6.028451e-10
| 2 | `measure_ipadapter_faceid_portrait_sdxl_contrib.py` | sum 5.076462e-03, largest share 0.008767294678320766, 0 negative
| 3 | `sweep_ipadapter_faceid_portrait_sdxl.py` | K = 0 / 20 / 40 / 80, gate identical at every K
| 4 | `convert_ipadapter_faceid_plusv2_sdxl_convrot_int8.py` | 162 INT8, 29 kept, 423,767,654 B, -58.0 %
| 5 | `ipadapter_faceid_portrait_sdxl_pack_audit.py` | CN-1 FAITHFUL 162/162, code 0, scale 0.0, float 29/29
| 6 | `ipadapter_faceid_portrait_sdxl_comfyui_route.py` | ComfyUI 22 + 140 installed, 162 kernel calls
| 7 | `a1111_ipadapter_faceid_portrait_sdxl_route.py` | A1111 model_built False - the plus-v2 graph is built for this file
| 8 | `ipadapter_faceid_portrait_sdxl_acceptance.py` | CN-5a image_proj 0.999995405, ip_adapter 140 weights 0.999962064
| 9 | `ipadapter_faceid_portrait_sdxl_baseline.py` | CN-5b band 8.093933e-05, 0 of 39 flagged
| 10 | `ipadapter_faceid_portrait_sdxl_comfyui_gate.py` | CN-6 mean 0.999986747, min 0.999923475, n 256, 5632 kernel calls
| 11 | `ipadapter_faceid_portrait_sdxl_render.py` | CN-4 source 1,980,889 B / 18.5 s, pack 2,004,275 B / 15.5 s

## 11. Files

| file | what it is |
| :--- | :--- |
| `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` | CN-1, the arithmetic and the behaviour |
| `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_route.json` | CN-2 / CN-3 on the loader node |
| `records/cn_ipadapter_faceid_portrait_sdxl_a1111_route.json` | CN-2 on A1111, and the graph it cannot build |
| `records/cn_ipadapter_faceid_portrait_sdxl_pack_source_output_diff.json` | CN-5a, both surfaces |
| `records/cn_ipadapter_faceid_portrait_sdxl_gate_256.json` | CN-6 at K = 0 |
| `records/cn_ipadapter_faceid_portrait_sdxl_trigger_baseline.json` | CN-5b against a second quantisation |
| `records/cn_ipadapter_faceid_portrait_sdxl_trigger_normalised.json` | CN-5b against the graph re-run |
| `records/cn_ipadapter_faceid_portrait_sdxl_isolation.json` | base, weight-0, source and pack arms |
| `records/cn_ipadapter_faceid_portrait_sdxl_embprobe.json` | the embedding statistics and the weight ladder |
| `records/cn_ipadapter_faceid_portrait_sdxl_render.json` | CN-4, the settings and both arms |
| `records/cn_ipadapter_faceid_portrait_sdxl_pixel_compare.json` | CN-4, the two PNGs compared |
| `records/cn_ipadapter_faceid_portrait_sdxl_source.png` | the source arm |
| `records/cn_ipadapter_faceid_portrait_sdxl_pack.png` | the pack arm |

### Retraction

None. This is the first record for this checkpoint.
