# SD1.5 IP-Adapter FaceID Plus v2 (`ip-adapter-faceid-plusv2_sd15`) ConvRot INT8 - Quantization and Score Report

The pack is `ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors`, 79,143,188 bytes, sha256
`3984cb16f3ed7507b60e4ac9b5ee21b10dcf8ea02aed9414adb3535a8104c15e`. Every 2-D weight of the adapter is
INT8 ConvRot (312 layers); the 32 float tensors that remain are the biases and the LayerNorm
parameters. The floating-point source is `ip-adapter-faceid-plusv2_sd15.bin`, 156,558,509 bytes, sha256
`26d0d86a1d60d6cc811d3b8862178b461e1eeb651e6fe2b72ba17aa95411e313`.

The path this pack is used on is ComfyUI with the loader package's `HSWQ Load IPAdapter Model` node
feeding `IPAdapterFaceID`. That route was measured here: the node installs all 312 INT8 layers, the
INT8 kernel fires 312 times on a single projection pass, and the 256-item acceptance gate returns
mean 0.999945044 / min 0.999930516 against mean >= 0.99 / min >= 0.98.

## Table of contents

0. The load path is the whole story
1. What is quantized
2. What the protection set is based on
3. The score table (256 items, seed 42, through the real loader)
4. Visible-sample check
5. Reproduction
6. Error sources and limits
7. Ledger
8. Files

## 0. The load path is the whole story

### 0.1 The checkpoint is the node's own nested layout, and the filename carries the v2 flag

The `.bin` holds two dicts, `image_proj` (56 tensors) and `ip_adapter` (288 tensors). The node flattens
them to `image_proj.*` / `ip_adapter.*`, and on the way back it splits on those prefixes. Nothing else
is stored: there is no `faceidplusv2` key in the file, so the v2 behaviour is decided by the filename -
the node sets `model["faceidplusv2"] = True` when the name contains `plusv2` (measured in
`nodes/hswq_load_ipadapter_model.py`, `_ipadapter_model_loader`). The pack keeps `plusv2` in its name,
so the flag is the same on both arms; the two arms were confirmed to carry it (route record,
`node.faceidplusv2_flag` and `node.source_faceidplusv2_flag`, both true).

### 0.2 The graph is the FaceID one, not the SDXL one

`ProjModelFaceIdPlus` is `proj` (512 -> 1024 -> 3072, reshaped to 4 x 768), a LayerNorm, then a
`FacePerceiverResampler` (proj_in 1280 -> 768, four blocks, proj_out, norm_out) fed by the CLIP ViT-H
penultimate. The node builds it when `is_faceid and is_plus`, with `clip_extra_context_tokens = 4` and
`cross_attention_dim = output_cross_attention_dim = 768` - the token count and width the SDXL path
does not use, so the SDXL argument set would build the wrong module here.

`ip_adapter` is 32 blocks of bias-free LoRA pairs (`to_q_lora`, `to_k_lora`, `to_v_lora`,
`to_out_lora`, rank 128) plus 16 `to_k_ip` / `to_v_ip` projections. `To_KV` builds one Linear per entry
of that dict, so 288 modules are built, with inner widths of 320, 640, 768 and 1280 and outer widths
of 128, 320, 640, 768 and 1280 - a set of shapes a single group size does not fit.

### 0.3 The loader package's node keeps the codes INT8

The node patches the extension's `IPAdapter` / `To_KV` globals and builds `ConvRotInt8Linear` for the
quantised weights. Measured on the pack: patch applied (both globals replaced), 24 INT8 layers in
`image_proj` and 288 in `ip_adapter`, 312 INT8 kernel calls on one projection pass. The same node on
the float source installs 0 INT8 layers, which is the negative control for the patch being conditional.

### 0.4 A single group size does not fit this graph

The shared IP-Adapter converter fixes `GROUP_SIZE = 256` and leaves any layer whose inner width is not
a multiple of it in FP16. On this checkpoint that is every 320 / 640 / 1280 layer, and the first run of
that converter produced **0 INT8 layers on a 312-layer graph** with 344 tensors kept - a pack that
looks quantised and is not. This pack's converter reads the group per layer: the largest power of four
that divides the inner width, which here is 256 for 124 layers and 64 for the other 188. The measured
split is in the pack audit (`arithmetic.groupsizes`).

### 0.5 A1111 installs the same payload into its own IP-Adapter graph

A1111's `sd-webui-controlnet` builds `ProjModelFaceIdPlus` when `is_faceid and is_plus`
(`ipadapter/ipadapter_model.py`), and `modules/ipadapter_int8_convrot.py` installs the ConvRot INT8
payload for that graph. Measured on the pack: 24 / 24 `image_proj` Linear modules INT8, 288 / 288
`ip_adapter` Linear modules INT8, 312 INT8 kernel calls on one projection pass, group sizes 64 and 256
both reaching the kernel, input widths 128 / 320 / 512 / 640 / 768 / 1024 / 1280 / 3072, every call with
`convrot` set, and finite output from the 4 x 768 stack.

Both hosts therefore carry the pack; neither is a quality claim, and the acceptance below is measured
on ComfyUI (the render's route) with CN-5a measured off the weights themselves.

### 0.6 What the harness got wrong before the pack existed

Recorded because "the pack was correct and the harness was not" is the shape this track fails in most
often, and because each of these produced a wrong number before it was read:

| what happened | the number that identifies it | what it was |
| :--- | :--- | :--- |
| the shared converter produced a pack with no quantisation | **0 INT8 layers** on a 312-layer graph, 344 tensors kept | the fixed `GROUP_SIZE = 256` leaves every 320 / 640 / 1280 inner width in FP16; the pack looked quantised and was not |
| the criterion's keys did not match the converter's | 0 layers ranked | the screen wrote `layer.weight`, the converter looks up the module name without `.weight` |
| the ComfyUI route raised on the first projection | `ConvRotInt8Linear expected 320 input features, got 768` | the harness fed the 4 x 768 stack to every `to_kvs` entry; the LoRA entries take 320 / 640-wide inputs |
| the A1111 route raised before the census | `expected scalar type Half but found Float` | A1111 runs float32 activations; the harness fed float16 into a LayerNorm |
| the layer count read as zero on a working route | `installed_int8_layers` 0 while the kernel fired 312 times | `isinstance` against a class imported through another package object; the census has to key on the class name |
| the trigger candidates changed between runs | largest candidate 8.36e-05 then 9.59e-05 on the same pack | the perturbation seed came from Python's salted `hash()`; it is `zlib.crc32` now and two runs agree |

## 1. What is quantized

| module group | weight shape | modules | INT8 |
| :--- | :--- | ---: | ---: |
| `image_proj.proj.0` | 1024 x 512 | 1 | 1 |
| `image_proj.proj.2` | 3072 x 1024 | 1 | 1 |
| `image_proj.perceiver_resampler.proj_in` | 768 x 1280 | 1 | 1 |
| 4 x `layers.N.0.to_q` | 768 x 768 | 4 | 4 |
| 4 x `layers.N.0.to_kv` | 1536 x 768 | 4 | 4 |
| 4 x `layers.N.0.to_out` | 768 x 768 | 4 | 4 |
| 4 x `layers.N.1.1` | 3072 x 768 | 4 | 4 |
| 4 x `layers.N.1.3` | 768 x 3072 | 4 | 4 |
| `image_proj.perceiver_resampler.proj_out` | 768 x 768 | 1 | 1 |
| 128 x LoRA `down` (32 blocks x 4) | 128 x 320/640/768/1280 | 128 | 128 |
| 128 x LoRA `up` (32 blocks x 4) | 320/640/768/1280 x 128 | 128 | 128 |
| 16 x `to_k_ip` | 320 x 768 | 16 | 16 |
| 16 x `to_v_ip` | 320 x 768 | 16 | 16 |
| **total** | | **312** | **312** |

32 tensors are not 2-D weights (biases and the LayerNorm weights) and stay float; the record's
`coverage.kept_float` is 0, so no 2-D weight was left in float. Group sizes on the pack are 256 for
124 layers and 64 for 188. Every INT8 layer carries its own `weight_scale` (float32, `[out, 1]`) and a
`comfy_quant` sidecar `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":<g>}`.

## 2. What the protection set is based on

The protection set is this checkpoint's own measurement. Two instruments were run on it: the drift
screen (one weight reconstructed as ConvRot INT8, the stage output's relative MSE) and the criterion
(the same sweep scored as the aggregate contribution, 1 - cosine). The criterion is what the decision
is made on; the screen only says which layers are worth ranking.

| layer | drift (relative MSE) | share of the contribution |
| :--- | ---: | ---: |
| `image_proj.proj.0` | 6.150512e-04 | 5.182 % |
| `image_proj.perceiver_resampler.layers.0.0.to_q` | 4.153478e-04 | 3.722 % |
| `image_proj.proj.2` | 3.400411e-04 | 2.536 % |
| `image_proj.perceiver_resampler.proj_in` | 2.664621e-04 | 2.366 % |
| `ip_adapter.17.to_out_lora.down` | 7.449577e-05 | 0.543 % |
| `ip_adapter.21.to_out_lora.down` | 7.043230e-05 | 0.493 % |
| `ip_adapter.29.to_v_ip` | 7.032828e-05 | 0.482 % |
| `ip_adapter.1.to_k_ip` | 6.521934e-05 | 0.490 % |
| `ip_adapter.29.to_k_ip` | 6.444912e-05 | 0.593 % |
| `ip_adapter.25.to_v_ip` | 6.304151e-05 | 0.435 % |
| `ip_adapter.17.to_k_ip` | 6.215214e-05 | 0.534 % |
| `ip_adapter.12.to_out_lora.down` | 6.190367e-05 | 0.520 % |
| `ip_adapter.31.to_k_ip` | 6.182971e-05 | 0.498 % |
| `ip_adapter.19.to_v_ip` | 6.133644e-05 | 0.542 % |
| `ip_adapter.19.to_k_ip` | 6.129583e-05 | 0.506 % |
| `ip_adapter.31.to_v_ip` | 6.085698e-05 | 0.506 % |
| `ip_adapter.21.to_k_ip` | 6.056366e-05 | 0.505 % |
| `ip_adapter.5.to_k_ip` | 6.050388e-05 | 0.514 % |
| `ip_adapter.20.to_out_lora.down` | 6.028744e-05 | 0.529 % |
| **layers above the 0.15 drift threshold** | **0** | - |
| **312 layers** | max 6.150512e-04 / median 2.952240e-05 / min 4.697038e-08 | sum 5.621867e-03 |

No layer crosses the 0.15 drift threshold, so the threshold protects nothing here and the decision
is the K curve. The distribution is flat: no layer with a negative contribution, the largest single
layer 5.18 % of the sum.

The curve, measured on the fixed 256-item set through the INT8 route (K = layers kept in FP16):

| K | INT8 layers | FP16 kept | size | gate mean | gate min | p5 | worst single token | `ip_adapter` mean | mean gain / gate margin | pack sha256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **0** | 312 | 0 | 79.1 MB | 0.999945044 | 0.999930516 | 0.999938630 | 0.999895573 | - | - | `3984cb16f3ed` |
| 20 | 292 | 20 | 89.0 MB | 0.999986390 | 0.999976352 | 0.999982875 | 0.999951005 | - | 4.1e-05 / 9.9e-03 = 0.42 % | `984ec3eb941a` |
| 40 | 272 | 40 | 101.1 MB | 0.999986390 | 0.999976352 | 0.999982875 | 0.999951005 | - | 4.1e-05 / 9.9e-03 = 0.42 % | `05442afd0991` |
| 60 | 252 | 60 | 106.5 MB | 0.999986390 | 0.999976352 | 0.999982875 | 0.999951005 | - | 4.1e-05 / 9.9e-03 = 0.42 % | `cea6b34ca1d9` |

Three points of K = 20, 40 and 60 return the same statistics to the last digit because the gate
observes the `image_proj` graph, and the first four layers of the ranking - the four that carry 13.8 %
of the contribution - are already protected at K = 20; protecting more `ip_adapter` weights does not
move that graph. The `ip_adapter` mean is left as `-` for those three packs: the surface exists for the
adopted pack only, and a value for the others was not measured.

**K = 0 is adopted.** It is the smallest pack and it passes with 9.9e-03 of mean margin and 2.0e-02 of
min margin. The larger candidates buy 4.1e-05 of mean - 0.42 % of that margin - for 9.8 to 27.3 MB
more size, and 0.42 % is movement the gate cannot see.

## 3. The score table (256 items, seed 42, through the real loader)

Instrument: 8 crops x 4 orientations x 8 brightness steps of one reference image, each through CLIP-H;
one fixed face embedding from that image (insightface buffalo_l, 512-d, extracted once and held for all
256 items); per item the mean cosine over the 4 injected tokens; `mean >= 0.99`, `min >= 0.98`.

**Two surfaces, reported separately.** The gate calls the `image_proj` graph, 24 INT8 layers of it.
The `ip_adapter` projections consume the UNet cross-attention K/V and the identity/CLIP stack rather
than the pixel image, so their agreement does not move with the item - the record states it as
`"depends_on_the_image": false` - and they are measured on a fixed input of each weight's own width.
Both surfaces are in the table; only the first is a 256-item surface.

| item | `image_proj` (24 layers) | `ip_adapter` (288 weights) |
| :--- | ---: | ---: |
| elements | 256 items x 4 tokens | 288 2-D weights, one cosine each |
| statistic | mean cosine over the item's 4 injected tokens | the weight's own cosine on a fixed input of its width |
| mean | **0.999945044** | **0.999974131** |
| min | **0.999930516** | **0.999959882** |
| p5 / p50 | 0.999938630 / 0.999945320 | - |
| worst single value | **0.999895573**, `c6_o1_b1`, its worst token (reported, not gated) | largest distance from 1: 4.011769e-05 |
| through the loader node (CN-5a) | 96 INT8 kernel calls, 0.999232120 mean / 0.997380123 worst batch | - |
| finite | yes (0 non-finite) | yes |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** | not gated: no item dependence to gate over |

6,144 = 256 x 24: the gate calls the `image_proj` graph per item, and 24 of its layers are INT8. The
CN-5a figure is a separate run of that graph over four CLIP inputs, which is why its kernel count is 96
and not 6,144; its worst batch is the one whose CLIP input differs most from the reference image.

The source against the pack dequantised from its own sidecars, the two surfaces side by side:

| surface | elements | mean | min | relative MSE |
| :--- | ---: | ---: | ---: | ---: |
| `image_proj` | 4 tokens | 0.998930287 | 0.985520761 | 9.773901e-03 |
| `ip_adapter` | 288 weights | 0.999974131 | 0.999959882 | - |

The `image_proj` minimum belongs to one token of one of the four CLIP inputs; the same four batches sit
at 0.999949706 / 0.999851169 / 0.999116154 on their worst token.

## 4. Visible-sample check

| item | value |
| :--- | ---: |
| source arm | `source.png`, 402459 B |
| pack arm | `pack.png`, 403685 B |
| settings | 512x512, 12 steps, cfg 7.0, dpmpp_2m / karras, seed 42, weight 0.8, weight_faceidv2 1.0 |
| mean abs diff / max / p99 | 12.981 / 244 / 92 |
| pixel cosine / PSNR | 0.987881 / 21.008 dB |
| identical-pixel ratio | 1.38 % |
| how it was looked at | image-recognition skill first (code 500, empty text), the PNGs read directly second |
| what was seen | both arms are finished portraits of the same subject in a red coat; the difference is local (hair ends, a shoulder detail, the car plate); neither is noise |

The two arms are not the same picture - a 12-step chain amplifies a weight change - but they are the
same subject rendered twice, not a collapse.

## 5. Reproduction

```
python clip_convert/diag_impact_ipadapter_faceid_plusv2.py --ipadapter <f.bin> --out <impact.json> --comfy <ComfyUI>
python clip_convert/measure_ipadapter_faceid_plusv2_contrib.py --ipadapter <f.bin> --out <contrib.json> --comfy <ComfyUI>
python clip_convert/convert_ipadapter_faceid_plusv2_convrot_int8.py --input <f.bin> --impact <impact.json> --output <pack.safetensors> --record <conversion.json>
python clip_convert/sweep_ipadapter_faceid_plusv2.py --impact <impact.json> --contrib <contrib.json> --input <f.bin> --work <dir> --ks 0 20 40 60
python audit/ipadapter_faceid_plusv2_pack_audit.py --source <f.bin> --pack <pack> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_plusv2_comfyui_route.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader root> --dir <folder> --json <record>
python audit/a1111_ipadapter_faceid_plusv2_route.py --a1111 <A1111 root> --pack <pack> --json <record>
python audit/ipadapter_faceid_plusv2_acceptance.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_plusv2_comfyui_gate.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader root> --dir <folder> --image <ref.png> --clip-vision <clip_vision_h> --json <record>
python audit/ipadapter_faceid_plusv2_comfyui_trigger.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader root> --dir <folder> --image <ref.png> --clip-vision <clip_vision_h> --json <record>
python audit/ipadapter_faceid_plusv2_render.py --server <ComfyUI :port> --out-dir <dir> --json <record> --output-root <server output dir>
```

Notes that decide whether a re-run matches:

| setting | value used | why it matters |
| :--- | :--- | :--- |
| checkpoint | `realisticVisionV60B1_v60B1VAE.safetensors` (SD1.5, VAE baked in) | FaceID Plus v2 is an SD1.5 graph; an SDXL checkpoint would build a different model |
| VAE | the checkpoint's baked VAE, no separate VAE node | nothing in the pack's arithmetic involves the VAE; it only decodes the acceptance image |
| clip vision | `clip_vision_h.safetensors` | FaceID Plus v2 reads the ViT-H penultimate (1280 wide); a ViT-L file would not fit |
| identity | one insightface buffalo_l embedding from the reference image | the same face for all 256 gate items and both render arms |
| loader | the loader package's `HSWQ Load IPAdapter Model` node | the stock loader reads int8 codes as float magnitudes |
| adapter node | `IPAdapterAdvanced` / `IPAdapterFaceID` with `insightface` supplied | the apply node raises `insightface model is required for FaceID models` without it |

The render needs a server; it was run on a second instance on port 8189 with `--lowvram
--disable-assets` while another instance was already running, so the comparison is two arms on one
host with identical settings.

## 6. Error sources and limits

* **The gate observes the `image_proj` stack.** That is the surface the injected conditioning travels
  on, but the 288 `ip_adapter` layers are outside it. They are covered by CN-1 (replay arithmetic and
  the adapter's own forward), CN-2 (installed layers and per-layer cosine) and CN-5a (per-layer cosine
  against the source), not by the 256-item gate.
* **The face embedding is one identity.** The 256 items vary the CLIP side, not the identity; a second
  identity would be a second instrument, and this report makes no claim beyond the one measured.
* **K=20/40/60 gate numbers are identical.** That is a property of the instrument, not a copy: those
  three protect the same four `image_proj` layers, and the gate cannot see the `ip_adapter` layers
  they also protect.
* **The trigger search does not model a trained trigger.** It injects synthetic perturbations (45
  patches, 6 frequency perturbations) and compares them against natural-transform controls; a trigger
  learned into the weights before quantisation is outside what this can see.
* **The A1111 route is measured at the projection, not rendered.** `modules/ipadapter_int8_convrot.py`
  installs all 312 layers (24 `image_proj`, 288 `ip_adapter`), the kernel fires 312 times on one pass
  with group sizes 64 and 256 and input widths 128 / 320 / 512 / 640 / 768 / 1024 / 1280 / 3072, and the
  injected stack comes back finite. What A1111 did not do for this pack is draw the picture: the two
  render arms are ComfyUI renders, so the pixel comparison says nothing about an A1111 image.
* **`proj` widens 512 -> 3072 and the resampler mixes 1280 into 768.** INT8 error concentrates in
  `image_proj.proj.0` (drift 6.15e-04, 5.18 % of the contribution); the measured effect on the injected
  stack under four CLIP inputs is 0.998930 mean / 0.985521 min cosine.

## 7. Ledger

The clock times below are the records' own file timestamps on this machine's clock, not typed in.

| time | step | setting it ran under | record |
| :--- | :--- | :--- | :--- |
| 14:28:40 | drift screen, 312 layers | seed 42, float32, the node's own `ProjModelFaceIdPlus` | `cn_ipadapter_faceid_plusv2_impact.json` |
| 14:29:19 | conversion, K=0 | drift threshold 0.15, group read per layer, no protection | `cn_ipadapter_faceid_plusv2_conversion.json` |
| 14:30:29 | CN-1 pack audit | the same two files | `cn_ipadapter_faceid_plusv2_pack_audit.json` |
| 14:36:32 | CN-6 gate at K=0 | 256 items, identity from `cn_clipvit_ref.png`, CLIP-H | `cn_ipadapter_faceid_plusv2_gate_256.json` |
| 14:37:34 | the criterion | one layer at a time, ConvRot INT8 reconstruction, 1 - cosine | `cn_ipadapter_faceid_plusv2_contrib.json` |
| 14:37:57 | K curve, four packs built | K = 0 / 20 / 40 / 60, each converted from this checkpoint's ranking | `sweep.json` behind `cn_ipadapter_faceid_plusv2_curve.json` |
| 14:39:46 | CN-6 gate at K=20 | same 256 items | `cn_ipadapter_faceid_plusv2_gate_256_k20.json` |
| 14:40:49 | CN-6 gate at K=40 | same 256 items | `cn_ipadapter_faceid_plusv2_gate_256_k40.json` |
| 14:41:40 | CN-6 gate at K=60 | same 256 items | `cn_ipadapter_faceid_plusv2_gate_256_k60.json` |
| 14:51:55 | CN-4 render | both arms, 512x512, 12 steps, cfg 7.0, dpmpp_2m / karras, seed 42, weight 0.8, weight_faceidv2 1.0 | `cn_ipadapter_faceid_plusv2_render.json` and the two PNGs |
| 14:52:20 | CN-4 pixel comparison | the two decoded PNGs | `cn_ipadapter_faceid_plusv2_pixel_compare.json` |
| 14:53:50 | CN-5a | both surfaces, four CLIP inputs | `cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` |
| 14:54:03 | the curve record | the four points joined with their gate numbers | `cn_ipadapter_faceid_plusv2_curve.json` |
| 15:08:16 | CN-2 route, second host | A1111, `sd-webui-controlnet` + `modules/ipadapter_int8_convrot.py` | `cn_ipadapter_faceid_plusv2_a1111_route.json` |
| 16:09:30 | CN-2 / CN-3 route, re-run | ComfyUI, `HSWQ Load IPAdapter Model`; the same numbers, now with the input widths and group sizes in the record | `cn_ipadapter_faceid_plusv2_comfyui_route.json` |
| 16:10:51 | CN-5b trigger, second seeded run | 56 candidates, 5 natural controls; identical candidate list and largest candidate, recorded as `seed_stability` | `cn_ipadapter_faceid_plusv2_comfyui_trigger.json` |
| 2026-11-16 | publication | Hub upload, README row and guide link | `cn_ipadapter_faceid_plusv2_hf_upload.json` |

Two points about the order, because the timestamps show it and a reader will notice:

* the drift screen and the first conversion were written before the criterion, because the first run
  of the shared converter (fixed group 256) produced a pack with 0 INT8 layers; the per-layer group
  and the criterion's corrected keys came after, and the adopted pack was rebuilt with them.
* the A1111 route was measured after the publication, when the two-host requirement was honoured in
  full; the record is timestamped 15:08 and the commit that carries it follows the first one.
* the CN-2 and CN-5b records were re-run at 16:09 and 16:10, to carry the input widths and group sizes
  the route saw and to put a second seeded trigger run beside the first. Both reproduced their earlier
  numbers - 312 kernel calls, largest candidate 8.364785264813879e-05, 0 flagged - so the times above
  are the re-runs' and the earlier runs are the reproducibility evidence, not a different measurement.

## 7a. Registration

The verdict index row for this pack is written in `md/backdoor/Backdoor_Audit_Verdicts.md`, because the
gate fails a verdict report that the index does not link. **The finished-pack table
(`md/HSWQ_High_Quality_Packs_ConvRot_INT8.md`) was not edited**: a catalogue row is added only on
instruction, so this run records the decision not to add one rather than adding it because the work
is finished.

## 8. Files

| file | role |
| :--- | :--- |
| `clip_convert/diag_impact_ipadapter_faceid_plusv2.py` | the drift screen |
| `clip_convert/measure_ipadapter_faceid_plusv2_contrib.py` | the criterion |
| `clip_convert/sweep_ipadapter_faceid_plusv2.py` | the K curve |
| `clip_convert/convert_ipadapter_faceid_plusv2_convrot_int8.py` | the converter |
| `audit/ipadapter_faceid_plusv2_pack_audit.py` | CN-1 |
| `audit/ipadapter_faceid_plusv2_comfyui_route.py` | CN-2 / CN-3 on ComfyUI |
| `audit/a1111_ipadapter_faceid_plusv2_route.py` | CN-2 on A1111 |
| `audit/ipadapter_faceid_plusv2_acceptance.py` | CN-5a, dequantised and through the INT8 route |
| `audit/ipadapter_faceid_plusv2_comfyui_gate.py` | CN-6 |
| `audit/ipadapter_faceid_plusv2_comfyui_trigger.py` | CN-5b, both positive-control axes |
| `audit/ipadapter_faceid_plusv2_render.py` | CN-4 |
| `records/cn_ipadapter_faceid_plusv2_impact.json` | the drift screen numbers |
| `records/cn_ipadapter_faceid_plusv2_contrib.json` | the contribution numbers and their distribution |
| `records/cn_ipadapter_faceid_plusv2_curve.json` | the four K points, with p5 and the margin ratios |
| `records/cn_ipadapter_faceid_plusv2_protection.json` | the protection decision and its reason |
| `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` | CN-5a, both surfaces and the INT8 route |
| `records/cn_ipadapter_faceid_plusv2_hf_upload.json` | the published bytes and the card checks |
| `md/backdoor/verdicts/CN_IPAdapter_FaceID_PlusV2_SD15_Pack.md` | the verdict |
| `md/CN_IPAdapter_FaceID_PlusV2_SD15_ConvRot_INT8_Impact_Technical_Guide.md` | the runbook |
