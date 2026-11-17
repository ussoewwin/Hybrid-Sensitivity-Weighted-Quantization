# SDXL IP-Adapter FaceID Portrait (`ip-adapter-faceid-portrait_sdxl_unnorm`) ConvRot INT8 - Quantization and Score Report

The pack is `ip-adapter-faceid-portrait-unnorm_sdxl_hswq_convrot_int8.safetensors`, 423,767,654 bytes, sha256
`54b10126efd6205510af3764c6ee53216064b52eb48267f5234e21c005994c38`. Every 2-D weight of the adapter is INT8 ConvRot (162 layers); the 29 float
tensors that remain are the biases and the LayerNorm parameters. The floating-point source is
`ip-adapter-faceid-portrait_sdxl_unnorm.bin`, 1,009,523,411 bytes, sha256 `220bb86e205393a3d0411631cb473caddbf35fd371be2905ca9008818170db55`.

The route measured here is ComfyUI with the loader package's `HSWQ Load IPAdapter Model` node: the node reads
`portraitunnorm` from the file name, builds the portrait graph, installs all 162 INT8 layers, and the
INT8 kernel fires 162 times on one projection pass. The 256-item gate returns mean 0.999986747 /
min 0.999923475 against mean >= 0.99 / min >= 0.98.

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

### 0.1 The checkpoint is a Resampler, and its input is the unnormalised ArcFace embedding

The `.bin` holds two dicts, `image_proj` (74 entries) and `ip_adapter` (140 entries). `image_proj` is a
`Resampler`: `proj_in` (1280, 512), four Perceiver blocks with 20 heads, `proj_out` (2048, 1280), and a
`latents` query of shape (1, 16, 1280). `proj_in`'s input width of 512 is the unnormalised ArcFace
embedding, not the CLIP sequence the other FaceID graphs take.

The node reads `portraitunnorm` from the file name - the pack's name carries `unnorm` for that reason, and
the measurement below is on the pack as shipped.

### 0.2 `IPAdapterPlus.py` builds the portrait graph from that flag

`apply_ipadapter` sets `is_portrait_unnorm = "portraitunnorm" in ipadapter`, derives
`cross_attention_dim = 1280` and `clip_extra_context_tokens = 16` from it (L242-243), and `IPAdapter.__init__`
takes the `init_proj_plus()` branch rather than `init_proj_faceid()` because `is_faceid and not
is_portrait_unnorm` is false (L61-68). The face embedding branch that follows keeps the vector unnormalised
for this flag only (L293-296).

### 0.3 The loader package's node keeps the codes INT8

Measured on the pack: patch applied, 22 INT8 layers in `image_proj` and
140 in `ip_adapter`, 162 INT8 kernel calls on one projection pass, kernel widths
[2048], group sizes [256]. The same node on the float source installs 0 INT8 layers.

### 0.4 A single group size fits this graph

Every inner width in `ip_adapter` is 2048 and `image_proj`'s are 512, 1280 and 8192 - all divisible by 256.
The converter's per-layer group reading returns 256 everywhere (162 layers), so the FP16-kept count is
29 (biases and LayerNorm only), not a shape rule.

### 0.5 A1111 cannot build this graph

`sd-webui-controlnet`'s `IPAdapterModel.load` derives `is_plus` from `"latents" in image_proj`, so it takes
`init_proj_faceid()` and builds `ProjModelFaceIdPlus` - `proj.*` keys this checkpoint does not have. Measured:
`RuntimeError: IP-Adapter ConvRot INT8 load mismatch: missing=['proj.0.weight', ...] unexpected=['latents',
'layers.0.0.to_q.weight', ...]`. The run is recorded in `records/cn_ipadapter_faceid_portrait_sdxl_a1111_route.json` with `model_built: false`.
This is a property of that loader's branch table, not of the pack.

### 0.6 What the harness got wrong before the pack existed

| what happened | the number that identifies it | what it was |
| :--- | :--- | :--- |
| the impact screen crashed on load | `size mismatch for proj_in.weight: copying [1280, 512], model [1280, 1280]` | the harness assumed the 1280-wide CLIP input; the checkpoint's input is 512 |
| the same graph returned a 2-D latent | `Tensors must have same number of dimensions: got 2 and 3` | the embedding was passed as (1, 512); the Resampler wants (1, 1, 512) |
| the trigger search could not see `cv2` | `cannot access free variable 'cv2'` | a function-local import shadowed the module-level one |
| the render broke at weight 0.8 | 格子状の破綻 in both arms | the adapter saturates above 0.5 on this node; see section 6 |
| the A1111 census could not run | `load mismatch: missing=['proj.0.weight', ...]` | the extension builds the plus-v2 graph for this file (0.5) |

## 1. What is quantized

| module group | weight shape | modules | INT8 |
| :--- | :--- | ---: | ---: |
| `image_proj.proj_in` | 1280 x 512 | 1 | 1 |
| `image_proj.proj_out` | 2048 x 1280 | 1 | 1 |
| 4 x `layers.N.0.to_q` / `to_kv` / `to_out` | 2048 / 4096 / 2048 x 2048 | 12 | 12 |
| 4 x `layers.N.1.1` / `layers.N.1.3` | 8192 x 2048 / 2048 x 8192 | 8 | 8 |
| 70 x `to_k_ip` | 640 or 1280 x 2048 | 70 | 70 |
| 70 x `to_v_ip` | 640 or 1280 x 2048 | 70 | 70 |
| **total** | | **162** | **162** |

29 tensors are not 2-D weights (biases, the LayerNorm weights and the `latents` query) and stay float;
the record's `coverage.kept_float` is 0, so no 2-D weight was left in float. Every INT8 layer carries its own
`weight_scale` (float32, `[out, 1]`) and a `comfy_quant` sidecar `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`.

## 2. What the protection set is based on

The protection set is this checkpoint's own measurement. Two instruments were run on it: the drift screen
(one weight reconstructed as ConvRot INT8, the stage output's relative MSE) and the criterion (the same sweep
scored as the aggregate contribution, 1 - cosine). The criterion decides.

| layer | drift (relative MSE) | share of the contribution |
| :--- | ---: | ---: |
| `ip_adapter.105.to_k_ip` | 4.450684e-05 | 0.877 % |
| `ip_adapter.87.to_k_ip` | 4.079308e-05 | 0.804 % |
| `ip_adapter.127.to_k_ip` | 4.018840e-05 | 0.792 % |
| `ip_adapter.5.to_k_ip` | 3.983512e-05 | 0.785 % |
| `ip_adapter.131.to_k_ip` | 3.974688e-05 | 0.783 % |
| `ip_adapter.119.to_k_ip` | 3.961915e-05 | 0.780 % |
| `ip_adapter.113.to_k_ip` | 3.930200e-05 | 0.774 % |
| `ip_adapter.19.to_k_ip` | 3.903459e-05 | 0.769 % |
| **layers above the 0.15 drift threshold** | **0** | - |
| **162 layers** | max 8.387730e-05 / median 7.013413e-05 / min 6.028451e-10 | sum 5.076462e-03 |

No layer crosses the 0.15 drift threshold, so nothing is kept by drift. The distribution is flat: the
largest single layer is 0.88 % of the sum, the median 0.706 %, and
0 layers have a negative contribution.

The curve, measured on the fixed 256-item set through the INT8 route (K = layers kept in FP16):

| K | INT8 layers | FP16 kept | size | gate mean | gate min | p5 | worst single token | `ip_adapter` mean | mean gain over the floor | pack sha256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **0** | 162 | 0 | 423.8 MB | 0.999986747 | 0.999923475 | 0.999971982 | 0.999971982 | 0.999964875 | 0.0e+00 | `54b10126efd6` |
| **20** | 142 | 20 | 466.9 MB | 0.999986747 | 0.999923475 | 0.999971982 | 0.999971982 | - | 0.0e+00 | `956fa87c77b4` |
| **40** | 122 | 40 | 515.3 MB | 0.999986747 | 0.999923475 | 0.999971982 | 0.999971982 | - | 0.0e+00 | `a7b5dbd4e4ab` |
| **80** | 82 | 80 | 616.0 MB | 0.999986747 | 0.999923475 | 0.999971982 | 0.999971982 | - | 0.0e+00 | `3b5add6ab111` |

**K = 0 is adopted.** The gate calls the `image_proj` graph, 22 of 162 layers; every layer in the
top 80 of the contribution ranking is an `ip_adapter` weight, so protecting 20, 40 or 80 of them returns the
identical gate statistics while the pack grows from 423.8 to 616.0 MB. The measured gain is
exactly zero.

## 3. The score table (256 items, seed 42, through the real loader)

Instrument: 8 crops x 4 orientations x 8 brightness steps of one reference image, each with its own
unnormalised ArcFace embedding (insightface buffalo_l); per item the mean cosine over the 16 injected tokens;
`mean >= 0.99`, `min >= 0.98`.

**Two surfaces, reported separately.** The gate calls the `image_proj` graph, 22 INT8 layers of it. The 140
`ip_adapter` projections consume the UNet cross-attention K/V, so their agreement does not move with the item
and they are measured on a fixed input of each weight's own width.

| item | `image_proj` (24 layers) | `ip_adapter` (288 weights) |
| :--- | ---: | ---: |
| elements | 256 items x 16 tokens | 140 2-D weights, one cosine each |
| statistic | mean cosine over the item's 16 injected tokens | the weight's own cosine on a fixed 16-token input |
| mean | **0.999986747** | **0.999964875** |
| min | **0.999923475** | **0.999962064** |
| p5 / p50 | 0.999971982 / 0.999989115 | - |
| worst single value | **0.999555409**, `c0_o2_b0`, its worst token | largest distance from 1: 3.793634e-05 |
| through the loader node (CN-5a) | 88 INT8 kernel calls, 0.999971191 mean / 0.999967509 worst batch | - |
| finite | yes (0 non-finite) | yes |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** | not gated: no item dependence to gate over |

5632 = 256 x 22: the gate calls the `image_proj` graph per item. Items without a detectable face
would be excluded and counted; here `skipped_no_face` is 0.

The source against the pack dequantised from its own sidecars, the two surfaces side by side:

| surface | elements | mean | min | relative MSE |
| :--- | ---: | ---: | ---: | ---: |
| `image_proj` | 16 tokens | 0.999995405 | 0.999971881 | 2.346997e-05 |
| `ip_adapter` | 140 weights | 0.999964875 | 0.999962064 | - |

## 4. Visible-sample check

The base model is `waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors`, loaded through `HSWQ Checkpoint Loader` with `weight_dtype: SDXL ConvRot INT8`; the VAE is
`sdxl_vae.safetensors`.

The two arms are `records/cn_ipadapter_faceid_portrait_sdxl_source.png` and `records/cn_ipadapter_faceid_portrait_sdxl_pack.png`.

| item | value |
| :--- | ---: |
| settings | 1024x1024, 12 steps, cfg 7.0, dpmpp_2m / karras, seed 42, weight 0.5, weight_faceidv2 1.0 |
| source arm | 1,980,889 B in 18.5 s |
| pack arm | 2,004,275 B in 15.5 s |
| mean abs diff / max / p99 | 55.717063 / 255 / 244 |
| pixel cosine / PSNR | 0.875642756 / 9.698528 dB |
| identical-pixel ratio | 0.012557983 |

Both PNGs were read directly and looked at. Both are finished illustrations - a girl with a long red coat in
a blue-lit street, no noise and no artefacts - and they are **not the same picture**: the coat pattern, the
signage and the floor reflections differ. The cosine of 0.875642756 and PSNR of 9.698528 dB say the same.

## 5. Reproduction

| setting | value used | why it matters |
| :--- | :--- | :--- |
| weight | 0.5 | the adapter saturates above 0.5 on this node; 0.8 breaks the picture in both arms (6) |
| pack name | contains `unnorm` | the loader sets `portraitunnorm` from the name |
| input | the 512-d unnorm ArcFace embedding | `proj_in` is (1280, 512) |
| group size | read per layer | 256 everywhere here |
| gate instrument | the loader node, INT8 route | a dequantised comparison would not exercise the kernel |
| seed | 42 | the gate, the trigger ladder and the render share it |

```
python clip_convert/diag_impact_ipadapter_faceid_portrait_sdxl.py --ipadapter <f.bin> --out <impact.json> --comfy <ComfyUI>
python clip_convert/measure_ipadapter_faceid_portrait_sdxl_contrib.py --ipadapter <f.bin> --out <contrib.json> --comfy <ComfyUI>
python clip_convert/sweep_ipadapter_faceid_portrait_sdxl.py --impact <impact.json> --contrib <contrib.json> --input <f.bin> --work <dir> --ks 0 20 40 80
python audit/ipadapter_faceid_portrait_sdxl_pack_audit.py --source <f.bin> --pack <pack> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_portrait_sdxl_comfyui_route.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader> --dir <dir> --json <record>
python audit/a1111_ipadapter_faceid_portrait_sdxl_route.py --a1111 <A1111 root> --pack <pack> --json <record>
python audit/ipadapter_faceid_portrait_sdxl_acceptance.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_portrait_sdxl_comfyui_gate.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader> --dir <dir> --image <ref.png> --json <record>
python audit/ipadapter_faceid_portrait_sdxl_baseline.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader> --dir <dir> --image <ref.png> --work <dir> --impact <impact.json> --json <record>
python audit/ipadapter_faceid_portrait_sdxl_render.py --server http://<comfyui-host> --out-dir <dir> --output-root <server output> --adapter-dir <dir> --json <record>
```

## 6. Error sources and limits

- **The A1111 route cannot build this graph.** Its `IPAdapterModel.load` builds `ProjModelFaceIdPlus` for a
  file carrying `latents`; the load raises. ComfyUI through the loader node is the route measured here.
- **The adapter saturates above weight 0.5 on the generic FaceID node.** Measured: 0.0 and 0.5 produce
  finished pictures; 0.8 produces a lattice of artefacts in **both** arms; 1.0 collapses the source arm
  entirely. The isolation run (`records/cn_ipadapter_faceid_portrait_sdxl_isolation.json`) records base and weight-0 arms as clean, so the break
  belongs to the conditioning at that strength, not to the pack.
- **The gate observes one surface of two.** It calls `image_proj`, 22 of 162 layers; the 140
  `ip_adapter` weights are measured on fixed inputs.
- **The K curve is flat by construction.** All four points return the identical gate statistics because the
  protected layers are `ip_adapter` weights the gate never calls.
- **One face, one base model, one prompt.** The 256 items vary the identity side of one reference image.
- **The render is one pair at one weight.** 55.717063 / 255 is recorded, not characterised.
- **The trigger band is set by another quantisation, not by theory.** Its value is the largest divergence a
  second honest pack showed over the same candidates (8.093933e-05); a subtler implant than that band is
  not separated by this instrument.

## 7. Ledger

| time | step | setting it ran under | record |
| :--- | :--- | :--- | :--- |
| - | drift screen | 162 layers, seed 42, one weight at a time | `records/cn_ipadapter_faceid_portrait_sdxl_impact.json` -> `layers_measured` |
| - | the criterion | 1 - cosine on the stage output | `records/cn_ipadapter_faceid_portrait_sdxl_contrib.json` -> `sum` |
| - | the K curve | K = 0 / 20 / 40 / 80, each converted from this ranking | `records/cn_ipadapter_faceid_portrait_sdxl_curve.json` -> `points` |
| - | CN-1 | replay of the converter's arithmetic on the adopted pack | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `verdict` |
| - | CN-2 / CN-3, ComfyUI | the loader node, kernel counted | `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_route.json` -> `int8_kernel_calls` |
| - | CN-2, A1111 | the extension's own factory; the graph cannot be built | `records/cn_ipadapter_faceid_portrait_sdxl_a1111_route.json` -> `model_built` |
| - | CN-5a | both surfaces and the INT8 route | `records/cn_ipadapter_faceid_portrait_sdxl_pack_source_output_diff.json` -> `ip_adapter_surface` |
| - | CN-6 | 256 fixed items at K = 0 | `records/cn_ipadapter_faceid_portrait_sdxl_gate_256.json` -> `mean` |
| - | CN-5b | 41 candidates, another honest pack as the band | `records/cn_ipadapter_faceid_portrait_sdxl_trigger_baseline.json` -> `band` |
| - | CN-4 | two arms at 1024x1024, weight 0.5 | `records/cn_ipadapter_faceid_portrait_sdxl_render.json` -> `arms` |

## 7a. Registration

| pack | row before | row after | line |
| :--- | :--- | :--- | :--- |
| `ip-adapter-faceid-portrait-unnorm_sdxl_hswq_convrot_int8` | absent | present | catalogue, IP-Adapter family |

## 8. Files

| file | role |
| :--- | :--- |
| `clip_convert/diag_impact_ipadapter_faceid_portrait_sdxl.py` | the drift screen |
| `clip_convert/measure_ipadapter_faceid_portrait_sdxl_contrib.py` | the criterion |
| `clip_convert/sweep_ipadapter_faceid_portrait_sdxl.py` | the K curve |
| `audit/ipadapter_faceid_portrait_sdxl_pack_audit.py` | CN-1 |
| `audit/ipadapter_faceid_portrait_sdxl_comfyui_route.py` | CN-2 / CN-3 on ComfyUI |
| `audit/a1111_ipadapter_faceid_portrait_sdxl_route.py` | CN-2 on A1111, and why it cannot build |
| `audit/ipadapter_faceid_portrait_sdxl_acceptance.py` | CN-5a |
| `audit/ipadapter_faceid_portrait_sdxl_comfyui_gate.py` | CN-6 |
| `audit/ipadapter_faceid_portrait_sdxl_comfyui_trigger.py` | CN-5b, first pass |
| `audit/ipadapter_faceid_portrait_sdxl_trigger_normalised.py` | CN-5b, the floor-normalised pass |
| `audit/ipadapter_faceid_portrait_sdxl_baseline.py` | CN-5b, the second-quantisation band |
| `audit/ipadapter_faceid_portrait_sdxl_isolation.py` | the weight-0 / base isolation |
| `audit/ipadapter_faceid_portrait_sdxl_embprobe.py` | the embedding probe |
| `audit/ipadapter_faceid_portrait_sdxl_render.py` | CN-4 |
| `records/cn_ipadapter_faceid_portrait_sdxl_trigger_baseline.json` | the band and all 41 candidates |
| `records/cn_ipadapter_faceid_portrait_sdxl_isolation.json` | base, weight-0, source and pack arms |
| `records/cn_ipadapter_faceid_portrait_sdxl_embprobe.json` | the embedding statistics and the weight ladder |
