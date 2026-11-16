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

## 1. What is quantized

| group | 2-D weights | INT8 | float kept |
|---|---:|---:|---:|
| `image_proj` (proj 512/1024/3072, resampler 768/1280) | 24 | 24 | 8 (norm + biases) |
| `ip_adapter` (LoRA down/up 128 pairs, to_k_ip / to_v_ip 16 each) | 288 | 288 | 24 (none - all 2-D) |
| total | 312 | 312 | 32 |

Group sizes measured on the pack: 256 for 124 layers, 64 for 188 layers. Every INT8 layer carries its
own `weight_scale` (float32, `[out, 1]`) and a `comfy_quant` sidecar
`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":<g>}`.

## 2. What the protection set is based on

The protection set is this checkpoint's own measurement, not another pack's. Two instruments were run
on it:

* the drift screen (`clip_convert/diag_impact_ipadapter_faceid_plusv2.py`): one weight at a time
  reconstructed as ConvRot INT8, the stage output's relative MSE recorded. 312 layers measured, max
  6.150512e-04, median 2.952240e-05, min 4.697038e-08.
* the criterion (`clip_convert/measure_ipadapter_faceid_plusv2_contrib.py`): the same sweep scored as
  the aggregate contribution to the stage output (1 - cosine), which is what the protection decision
  is made on. Sum of contributions 5.621867e-03; the largest are `image_proj.proj.0` (5.18 %),
  `image_proj.perceiver_resampler.layers.0.0.to_q` (3.72 %), `image_proj.proj.2` (2.54 %),
  `image_proj.perceiver_resampler.proj_in` (2.37 %).

The K curve was measured on this checkpoint, four points, each converted and each scored with the same
256-item gate through the same node:

| K (protected) | pack MB | INT8 | gate mean | gate min | pass | share of contribution |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 79.143 | 312 | 0.999945044 | 0.999930516 | yes | 0.0000 |
| 20 | 88.962 | 292 | 0.999986390 | 0.999976352 | yes | 0.2302 |
| 40 | 101.098 | 272 | 0.999986390 | 0.999976352 | yes | 0.3348 |
| 60 | 106.471 | 252 | 0.999986390 | 0.999976352 | yes | 0.4296 |

K=0 is adopted. It holds 0.009945 of margin over the 0.99 threshold; the best larger candidate gains
4.135e-05 of mean (0.42 % of that margin) and costs +27.328 MB, so it does not displace the smallest
passing pack. K=20, 40 and 60 return the same gate numbers because the gate observes the
`image_proj` stack and those three protect the same four `image_proj` layers; the budget they add goes
into `ip_adapter` layers, which the gate does not see and which CN-5a measures separately.

## 3. The score table (256 items, seed 42, through the real loader)

Instrument: 8 crops x 4 orientations x 8 brightness steps of one reference image, each through CLIP-H;
one fixed face embedding from that image (insightface buffalo_l, 512-d, extracted once and held for all
256 items); per item the mean cosine over the 4 injected tokens; `mean >= 0.99`, `min >= 0.98`.

| statistic | value |
|---|---:|
| n | 256 |
| mean | 0.999945044 |
| min | 0.999930516 |
| p5 | 0.999938630 |
| p50 | 0.999945320 |
| worst item | see record |
| INT8 kernel calls | 6,144 |
| pass | yes |

6,144 = 256 x 24: the gate calls the `image_proj` graph per item, and 24 of its layers are INT8.

## 4. Visible-sample check

Two images, same settings, same seed, the source arm and the pack arm, both through the loader node and
`IPAdapterFaceID` (512 x 512, 12 steps, cfg 7.0, dpmpp_2m / karras, weight 0.8, weight_faceidv2 1.0,
identity from the reference image).

Read as images: the source arm is a finished portrait of a woman in a red coat in a city street; the
pack arm is the same scene, the same character, the same coat and pose, with the hair ends, a shoulder
detail and the car plate differing locally. Neither arm is noise.

Pixel comparison of the two: mean |diff| 12.981, max 244, p99 92, cosine 0.987881, PSNR 21.008 dB,
identical-pixel ratio 1.38 %. The two arms are not the same picture - the adapter is part of a sampling
chain and 12 steps amplify any change - but they are the same subject rendered twice, not a collapse.

## 5. Reproduction

```
python clip_convert/diag_impact_ipadapter_faceid_plusv2.py --ipadapter <f.bin> --out <impact.json> --comfy <ComfyUI>
python clip_convert/measure_ipadapter_faceid_plusv2_contrib.py --ipadapter <f.bin> --out <contrib.json> --comfy <ComfyUI>
python clip_convert/convert_ipadapter_faceid_plusv2_convrot_int8.py --input <f.bin> --impact <impact.json> --output <pack.safetensors> --record <conversion.json>
python clip_convert/sweep_ipadapter_faceid_plusv2.py --impact <impact.json> --contrib <contrib.json> --input <f.bin> --work <dir> --ks 0 20 40 60
python audit/ipadapter_faceid_plusv2_pack_audit.py --source <f.bin> --pack <pack> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_plusv2_comfyui_route.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader root> --dir <folder> --json <record>
python audit/ipadapter_faceid_plusv2_acceptance.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_plusv2_comfyui_gate.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader root> --dir <folder> --image <ref.png> --clip-vision <clip_vision_h> --json <record>
python audit/ipadapter_faceid_plusv2_comfyui_trigger.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader root> --dir <folder> --image <ref.png> --clip-vision <clip_vision_h> --json <record>
python audit/ipadapter_faceid_plusv2_render.py --server <ComfyUI :port> --out-dir <dir> --json <record> --output-root <server output dir>
```

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
* **The A1111 route was not measured for this pack.** The host instruction for this run is ComfyUI;
  the A1111 IP-Adapter runtime that serves the SDXL pack does not carry this graph's module set, and
  building it there was not part of the instruction. This is a scope statement, not a pass.
* **`proj` widens 512 -> 3072 and the resampler mixes 1280 into 768.** INT8 error concentrates in
  `image_proj.proj.0` (drift 6.15e-04, 5.18 % of the contribution); the measured effect on the injected
  stack under four CLIP inputs is 0.998930 mean / 0.985521 min cosine.

## 7. Ledger

| item | value |
|---|---|
| source | `ip-adapter-faceid-plusv2_sd15.bin`, 156,558,509 B, sha256 `26d0d86a...e313` |
| pack | `ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors`, 79,143,188 B, sha256 `3984cb16...c15e` |
| size ratio | 0.5056 of the source |
| INT8 layers | 312 of 312 measured 2-D weights |
| float kept | 32 tensors that are not 2-D weights (biases, LayerNorm weights); `coverage.kept_float` = 0 |
| group sizes | 256 (124 layers), 64 (188 layers) |
| CN-1 | FAITHFUL: 312/312 replay, code delta 0, scale mismatch 0, non-finite scale 0, float side 32/32 bit-identical |
| CN-1 behaviour | injected stack cosine 0.999968 mean / 0.999955 min |
| CN-2 route | patch applied, 312 INT8 installed (24 + 288), 312 kernel calls, source arm 0 |
| CN-5a | `image_proj` 0.998930 / 0.985521 over four CLIP inputs (relMSE max 9.774e-03); `ip_adapter` 288 layers 0.999974 / 0.999960 |
| CN-6 gate | mean 0.999945044, min 0.999930516, p5 0.999938630, pass |
| CN-5b trigger | 56 candidates, 0 flagged, control p99 1.079e-04, max candidate 8.364e-05, positive control floor 4 LSB |
| CN-4 render | both arms finished images; mean |diff| 12.981, cosine 0.987881, PSNR 21.008 dB |

## 8. Files

* converter: `clip_convert/convert_ipadapter_faceid_plusv2_convrot_int8.py`
* sweep: `clip_convert/sweep_ipadapter_faceid_plusv2.py`
* drift screen: `clip_convert/diag_impact_ipadapter_faceid_plusv2.py`
* criterion: `clip_convert/measure_ipadapter_faceid_plusv2_contrib.py`
* audits: `audit/ipadapter_faceid_plusv2_pack_audit.py`, `audit/ipadapter_faceid_plusv2_comfyui_route.py`,
  `audit/ipadapter_faceid_plusv2_acceptance.py`, `audit/ipadapter_faceid_plusv2_comfyui_gate.py`,
  `audit/ipadapter_faceid_plusv2_comfyui_trigger.py`, `audit/ipadapter_faceid_plusv2_render.py`
* records: `md/backdoor/verdicts/records/cn_ipadapter_faceid_plusv2_*`
* verdict: `md/backdoor/verdicts/CN_IPAdapter_FaceID_PlusV2_SD15_Pack.md`
