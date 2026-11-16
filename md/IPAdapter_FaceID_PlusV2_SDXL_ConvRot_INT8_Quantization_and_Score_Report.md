# SD1.5→SDXL IP-Adapter FaceID Plus v2 (`ip-adapter-faceid-plusv2_sdxl`) ConvRot INT8 - Quantization and Score Report

The pack is `ip-adapter-faceid-plusv2_sdxl_hswq_convrot_int8.safetensors`, 748,384,560 bytes, sha256
`c9376bc5c5535ee43a96442049e3b7e0a451439209c696a45da5ea09a23323a3`. Every 2-D weight of the adapter is INT8 ConvRot (1284 layers); the 32 float tensors
that remain are the biases and the LayerNorm parameters. The floating-point source is
`ip-adapter-faceid-plusv2_sdxl.bin`, 1,487,555,181 bytes, sha256 `c6945d82b543700cc3ccbb98d363b837e9c596281607857c74b713a876daf5fb`. The pack is
50.3 % of the source's size.

## Table of contents

0. [The load path is the whole story](#0-the-load-path-is-the-whole-story)
1. [What is quantized](#1-what-is-quantized)
2. [What the protection set is based on](#2-what-the-protection-set-is-based-on)
3. [The score table (256 items, seed 42, through the real loader)](#3-the-score-table-256-items-seed-42-through-the-real-loader)
4. [Visible-sample check](#4-visible-sample-check)
5. [Reproduction](#5-reproduction)
6. [Error sources and limits](#6-error-sources-and-limits)
7. [Ledger](#7-ledger)
7a. [Registration](#7a-registration)
8. [Files](#8-files)

---

## 0. The load path is the whole story

### 0.1 The checkpoint carries its own CLIP, and the SDXL CLIP is not optional

The `.bin` is a torch pickle holding two dicts, `image_proj` (56 tensors) and `ip_adapter` (1,260
tensors), all `float16`. The loader splits them on those prefixes and sets `faceidplusv2` from the file
name. The SDXL graph then needs `cross_attention_dim = 2048`, `output_cross_attention_dim = 2048`,
`clip_embeddings_dim = 1280` and `clip_extra_context_tokens = 4` - the SD1.5 width of 768 builds a
different module and fails on the first weight.

### 0.2 The projection is 8192-wide, and float16 overflows it

`proj` is 512 -> 1024 -> 8192, reshaped to 4 x 2048, and the resampler runs 20 heads. With synthetic
`N(0,1)` CLIP inputs in float16 the projection returns NaN - and it does so on the **float source** as
well, so it is the graph's range, not the pack. Every projection measurement in this report therefore
runs in float32; the gate, which uses real CLIP features, does the same. The route record carries this as
`activation_dtype`.

### 0.3 A single group size does not fit this graph

The shared converter fixes `GROUP_SIZE = 256`. This graph's inner widths are 128, 512, 640, 1024, 1280,
2048 and 8192: a fixed 256 leaves every 128- and 640-wide layer in FP16 and silently produces a pack with
far fewer INT8 layers than the graph has. The group is read per layer as the largest power of four that
divides the inner width, which here is 256 for 664 layers and 64 for the other 620.

### 0.4 Both hosts execute it

ComfyUI through the loader package's `HSWQ Load IPAdapter Model` node; A1111 through
`sd-webui-controlnet`'s IP-Adapter path plus `modules/ipadapter_int8_convrot.py`. Both were measured by
importing each host's own classes, not by re-implementing them.

### 0.5 What the harness got wrong before the pack existed

| what happened | the number that showed it | what it was |
| :--- | :--- | :--- |
| the first projection comparison came back NaN | `image_proj` cos `NaN`, finite false | float16 activations on the 8192-wide projection; the source returns NaN too |
| the A1111 census read zero INT8 layers on a firing kernel | `installed` 0 while the kernel fired 1,284 times | `isinstance` against a class imported through another package object |
| a single group size produced a thin pack | INT8 count well below the 1,284 2-D weights | the fixed 256 leaves 128/640 widths in FP16 |

## 1. What is quantized

| module group | weight shape | modules | INT8 |
| :--- | :--- | ---: | ---: |
| `image_proj.proj.0` / `proj.2` / resampler | 1024x512, 8192x1024, up to 2048x2048 | 24 | 24 |
| `ip_adapter` low-rank pairs (rank 128) | down 128x2048, up 2048x128 | 1,152 | 1,152 |
| `ip_adapter` `to_k_ip` / `to_v_ip` | 1280x2048 | 108 | 108 |
| total 2-D weights | - | **1,284** | **1,284** |
| float tensors kept | biases, LayerNorm parameters | **32** | 0 |

Coverage and identity, from the pack's own audit: 1284 INT8 layers
(`records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `coverage.int8_layers`), 32 non-2-D
float tensors (`coverage.non_2d_float_tensors`), 0 missing / 0 extra / 0 shape-mismatched keys
(`coverage.missing`), 0 layers without a `weight_scale` sidecar (`coverage.missing_scale`).

## 2. What the protection set is based on

The protection set is this checkpoint's own measurement. Two instruments were run on it: the drift screen
(one weight reconstructed as ConvRot INT8, the stage output's relative MSE) and the criterion (1 - cosine
on the same stage output). The criterion decides.

| statistic | measured | record |
| :--- | ---: | :--- |
| largest layer drift (`ip_adapter.3.to_k_ip`) | 8.632594e-05 | `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `max` |
| median layer drift | 2.053594e-05 | `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `median` |
| smallest layer drift | 8.672217e-11 | `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `min` |
| layers measured | 1284 | `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `layers_measured` |
| summed contribution | 1.704065e-02 | `records/cn_ipadapter_faceid_plusv2_sdxl_contrib.json` -> `sum` |
| largest single layer's share | 0.002535003 | `records/cn_ipadapter_faceid_plusv2_sdxl_contrib.json` -> `distribution.largest_share` |
| layers with a negative contribution | 0 | `records/cn_ipadapter_faceid_plusv2_sdxl_contrib.json` -> `distribution.negative_gain_layers` |
| layers above the 0.15 drift threshold | 0 | `records/cn_ipadapter_faceid_plusv2_sdxl_conversion.json` -> `arguments.protect_count` |

No layer crosses the threshold, so nothing is kept by drift, and the distribution is flat: the largest
single layer is 0.25 % of the total, the median 1.081513e-05.

The curve, measured on the fixed 256-item set through the INT8 route (K = layers kept in FP16):

| K | INT8 layers | FP16 kept | size | gate mean | gate min | p5 | worst single token | mean gain / gate margin | pack sha256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **0** | 1284 | 0 | 748.4 MB | 0.999926916 | 0.999925345 | 0.999926001 | 0.999921679 | - | `c9376bc5c553` |
| 40 | 1244 | 40 | 844.1 MB | 0.999926916 | 0.999925345 | 0.999926001 | 0.999921679 | 0.0 / 9.93e-03 = 0.0 % | `bb0fbe978a57` |
| 80 | 1204 | 80 | 940.9 MB | 0.999926916 | 0.999925345 | 0.999926001 | 0.999921679 | 0.0 / 9.93e-03 = 0.0 % | `022a181c10d1` |
| 160 | 1124 | 160 | 1091.7 MB | 0.999926916 | 0.999925345 | 0.999926001 | 0.999921679 | 0.0 / 9.93e-03 = 0.0 % | `6fe95da2ad58` |

**K = 0 is adopted, and here the reason is exact rather than proportional.** Every layer in the top 160 of
the contribution ranking is an `ip_adapter` weight, and the gate calls the `image_proj` graph only - so
protecting 40, 80 or 160 of them moves the gate by **exactly zero** while adding
95.7 to 343.4 MB. There is no quality to buy at any price on this curve.

## 3. The score table (256 items, seed 42, through the real loader)

Instrument: 8 crops x 4 orientations x 8 brightness steps of one reference image, each through CLIP-H;
one fixed face embedding from that image (insightface buffalo_l, 512-d, extracted once for all 256
items); per item the mean cosine over the 4 injected tokens; `mean >= 0.99`, `min >= 0.98`.

**Two surfaces, reported separately.** The gate calls the `image_proj` graph, 24 INT8 layers of it. The
1,260 `ip_adapter` weights consume the UNet cross-attention K/V and the identity/CLIP stack rather than
the pixel image, so their agreement does not move with the item (`depends_on_the_image: false`) and they
are measured on a fixed input of each weight's own width.

| item | `image_proj` (24 layers) | `ip_adapter` (1,260 weights) |
| :--- | ---: | ---: |
| items | 256 items x 4 tokens | 1,260 2-D weights, one cosine each |
| statistic | mean cosine over the item's 4 injected tokens | the weight's own cosine on a fixed input |
| mean | **0.999926916** | **0.999972070** |
| min | **0.999925345** | **0.999928368** |
| p5 / p50 | 0.999926001 / 0.999926969 | - |
| worst single value | **0.999921679**, `c7_o1_b1`, its worst token | largest distance from 1: 7.163225e-05 |
| through the loader node (CN-5a) | 96 INT8 kernel calls, 0.999906062 mean / 0.999905986 worst batch | - |
| finite | yes | yes |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** | not gated: no item dependence to gate over |

6144 = 256 x 24: the gate calls the `image_proj` graph per item and 24 of its layers are INT8
(`records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `int8_kernel_calls`). The CN-5a figure is a
separate run over four CLIP inputs, which is why its count is 96.

Source against the pack dequantised from its own sidecars
(`records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `ip_adapter_surface`):

| surface | elements | mean | min | relative MSE |
| :--- | ---: | ---: | ---: | ---: |
| `image_proj` | 4 tokens | 0.999960350 | 0.999956486 | 7.890615e-05 |
| `ip_adapter` | 1,260 weights | 0.999972070 | 0.999928368 | - |

## 4. Visible-sample check

The base model is an HSWQ ConvRot INT8 SDXL UNet, `waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors`, loaded through `HSWQ Checkpoint Loader` with
`weight_dtype: SDXL ConvRot INT8`; it carries its own SDXL CLIP (`conditioner.embedders.0/1`), and the VAE
is a separate `sdxl_vae.safetensors`. That it carries a CLIP is a property of the file read from its header, not of its
name - see section 6.

The two arms are `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_source.png` and
`records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_pack.png`; `_comfyui_` rather than the family's
`_a1111_`, because this render ran on ComfyUI.

| item | value |
| :--- | ---: |
| settings | 1024x1024, 12 steps, cfg 7.0, dpmpp_2m / karras, seed 42, weight 0.8, weight_faceidv2 1.0 |
| source arm | 1,696,348 B in 42.7 s |
| pack arm | 1,696,653 B in 21.6 s |
| mean abs diff / max / p99 | 16.408394 / 255 / 143 |
| pixel cosine / PSNR | 0.962361634 / 18.027259 dB |
| identical-pixel ratio | 0.039916 |

Both PNGs were read directly and looked at. Both are finished illustrations of the same subject in the
same red coat on the same neon street; the difference is local - the framing is slightly tighter on the
pack arm, the shop signs differ, the bystanders are placed differently. Neither is noise, and neither is a
collapse. The cosine of 0.962362 says the two are not the same picture, which a 12-step chain
amplifies; the judgement that both are pictures comes from the look, not from the number.

## 5. Reproduction

```
python clip_convert/diag_impact_ipadapter_faceid_plusv2_sdxl.py --ipadapter <f.bin> --out <impact.json> --comfy <ComfyUI>
python clip_convert/measure_ipadapter_faceid_plusv2_sdxl_contrib.py --ipadapter <f.bin> --out <contrib.json> --comfy <ComfyUI>
python clip_convert/sweep_ipadapter_faceid_plusv2_sdxl.py --impact <impact.json> --contrib <contrib.json> --input <f.bin> --work <dir> --ks 0 40 80 160
python audit/ipadapter_faceid_plusv2_sdxl_pack_audit.py --source <f.bin> --pack <pack> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_comfyui_route.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader root> --dir <folder> --json <record>
python audit/a1111_ipadapter_faceid_plusv2_sdxl_route.py --a1111 <A1111 root> --pack <pack> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_acceptance.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_comfyui_gate.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader root> --dir <folder> --image <ref.png> --clip-vision <f> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_comfyui_trigger.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader root> --dir <folder> --image <ref.png> --clip-vision <f> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_render.py --server http://127.0.0.1:8189 --out-dir <dir> --output-root <server output> --json <record>
```

Notes that decide whether a re-run matches:

* the projection measurements are **float32**; float16 returns NaN on the 8192-wide `proj`, on the source
  too (section 0.2);
* the gate needs the reference image's face (insightface buffalo_l, CPU) and `clip_vision_h`;
* the render needs an SDXL base that carries its own CLIP, plus `sdxl_vae`, plus the loader package's
  `HSWQ Checkpoint Loader` set to `SDXL ConvRot INT8`;
* the converter must read the group size per layer (section 0.3).

## 6. Error sources and limits

* **The gate observes one surface of two.** It calls the `image_proj` graph, 24 of 1,284 layers. The 1,260
  `ip_adapter` weights are measured on fixed inputs, not on the 256-item set.
* **The K curve is flat by construction.** All four points return the identical gate statistics because the
  protected layers are `ip_adapter` and the gate cannot see them. The curve is still measured and reported:
  a zero measured is not the same as a zero assumed.
* **One face, one base model, one prompt.** The 256 items vary the CLIP side of a single identity against a
  single reference face.
* **One base model for the picture.** `waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors` is an Illustrious-family SDXL UNet; a different checkpoint
  would give a different picture and is not covered.
* **The trigger search does not model a trained trigger.** It injects synthetic perturbations and compares
  them against natural-transform controls; a trigger learned into the weights before quantisation is
  outside what this can see.
* **The instrument's floor is coarse on the scale axis.** The code axis resolves at 2 LSB
  (1.320351e-04 against a 1.172190e-04 threshold) and the scale axis at x1.05
  (1.530211e-04); a subtler implant is not seen.
* **The model inventory is a snapshot of one machine.** 1307 entries across four model roots, read
  by container and key prefixes; a different machine will differ.

## 7. Ledger

The clock times below are the records' own file timestamps as the steps wrote them, not the copies.

| time | step | setting it ran under | record |
| :--- | :--- | :--- | :--- |
| 2026-11-16 | drift screen, 1284 layers | seed 42, one weight at a time, the node's own `ProjModelFaceIdPlus` at 2048 | `cn_ipadapter_faceid_plusv2_sdxl_impact.json` |
| 2026-11-16 | the criterion: the aggregate contribution | one layer at a time, 1 - cosine on the stage output | `cn_ipadapter_faceid_plusv2_sdxl_contrib.json` |
| 16:45:35 | K curve, four packs built | K = 0 / 40 / 80 / 160, each from this checkpoint's own ranking | `cn_ipadapter_faceid_plusv2_sdxl_curve.json` |
| 16:50:26 | CN-1 pack audit | the source and the adopted pack | `cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` |
| 16:46:48 | CN-6 gate at K=0 | 256 items, identity from `cn_clipvit_ref.png`, CLIP-H | `cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` |
| 16:47:45 | CN-6 gate at K=40 | the same 256 items | `cn_ipadapter_faceid_plusv2_sdxl_gate_256_k40.json` |
| 16:48:41 | CN-6 gate at K=80 | the same 256 items | `cn_ipadapter_faceid_plusv2_sdxl_gate_256_k80.json` |
| 16:49:35 | CN-6 gate at K=160 | the same 256 items | `cn_ipadapter_faceid_plusv2_sdxl_gate_256_k160.json` |
| 16:54:45 | CN-2 / CN-3, ComfyUI | the loader node, float32 activations, kernel counted | `cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` |
| 16:55:14 | CN-2, A1111 | `sd-webui-controlnet` + `modules/ipadapter_int8_convrot.py` | `cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json` |
| 16:55:54 | CN-5a | both surfaces, four CLIP inputs, and the INT8 route | `cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` |
| 16:57:02 | CN-5b trigger | 56 candidates, 5 natural controls, both ladders | `cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` |
| 18:02:01 | CN-4 render | both arms, 1024x1024, 12 steps, cfg 7.0 | `cn_ipadapter_faceid_plusv2_sdxl_render.json` + the two PNGs |
| 18:02:11 | CN-4 pixel comparison | the two decoded PNGs | `cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` |
| 2026-11-16 | the model inventory (section 1.10) | every model path, no name filter; 1307 entries | `cn_ipadapter_faceid_plusv2_sdxl_model_inventory.json` |

## 7a. Registration

Both tables carry this pack. The verdict index row is in `md/backdoor/Backdoor_Audit_Verdicts.md`
(the gate fails a verdict that the index does not link), and the finished-pack row is in
`md/HSWQ_High_Quality_Packs_ConvRot_INT8.md` - `ip-adapter-faceid-plusv2_sdxl_hswq_convrot_int8`,
SDXL / IP-Adapter, gate 0.999927 / 0.999925, 748.4 MB, sha `C9376BC5C5535EE4`, linking this pack's
guide. The row followed the shape of the rows already in that table.

## 8. Files

| file | role |
| :--- | :--- |
| `clip_convert/diag_impact_ipadapter_faceid_plusv2_sdxl.py` | the drift screen |
| `clip_convert/measure_ipadapter_faceid_plusv2_sdxl_contrib.py` | the criterion |
| `clip_convert/sweep_ipadapter_faceid_plusv2_sdxl.py` | the K curve |
| `clip_convert/convert_ipadapter_faceid_plusv2_sdxl_convrot_int8.py` | the converter |
| `audit/ipadapter_faceid_plusv2_sdxl_pack_audit.py` | CN-1 |
| `audit/ipadapter_faceid_plusv2_sdxl_comfyui_route.py` | CN-2 / CN-3 on ComfyUI |
| `audit/a1111_ipadapter_faceid_plusv2_sdxl_route.py` | CN-2 on A1111 |
| `audit/ipadapter_faceid_plusv2_sdxl_acceptance.py` | CN-5a on both surfaces and through the INT8 route |
| `audit/ipadapter_faceid_plusv2_sdxl_comfyui_gate.py` | CN-6 |
| `audit/ipadapter_faceid_plusv2_sdxl_comfyui_trigger.py` | CN-5b, both positive-control axes |
| `audit/ipadapter_faceid_plusv2_sdxl_render.py` | CN-4 |
| `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `max` | the drift screen numbers |
| `records/cn_ipadapter_faceid_plusv2_sdxl_contrib.json` -> `distribution.largest_share` | the contribution numbers |
| `records/cn_ipadapter_faceid_plusv2_sdxl_curve.json` -> `decision.k` | the four K points with their gates |
| `records/cn_ipadapter_faceid_plusv2_sdxl_protection.json` -> `decision.protected_layers` | the protection decision |
| `records/cn_ipadapter_faceid_plusv2_sdxl_conversion.json` -> `pack_sha256` | the conversion arguments and both hashes |
| `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `verdict` | CN-1 |
| `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `int8_kernel_calls` | CN-2 / CN-3 on ComfyUI |
| `records/cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json` -> `int8_kernel_calls` | CN-2 on A1111 |
| `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `int8_route` | CN-5a |
| `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `mean` | CN-6 |
| `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` -> `control_p99` | CN-5b |
| `records/cn_ipadapter_faceid_plusv2_sdxl_render.json` -> `settings.seed` | CN-4 and the two arms |
| `records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` -> `mean_abs_diff` | CN-4 pixel statistics |
| `records/cn_ipadapter_faceid_plusv2_sdxl_model_inventory.json` -> `counts.entries` | section 1.10 |
| `md/backdoor/verdicts/CN_IPAdapter_FaceID_PlusV2_SDXL_Pack.md` | the verdict |
| `md/CN_IPAdapter_FaceID_PlusV2_SDXL_ConvRot_INT8_Impact_Technical_Guide.md` | the runbook |
