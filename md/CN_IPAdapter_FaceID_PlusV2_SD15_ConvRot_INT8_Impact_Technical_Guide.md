# SD1.5 IP-Adapter FaceID Plus v2 - ConvRot INT8 Impact Technical Guide

> **The conversion record** for this pack (what was quantised, the protection ranking, the conversion
> arguments, the faithfulness replay and the score) is
> [`IPAdapter_FaceID_PlusV2_SD15_ConvRot_INT8_Quantization_and_Score_Report.md`](IPAdapter_FaceID_PlusV2_SD15_ConvRot_INT8_Quantization_and_Score_Report.md).
>
> **The method this pack was built under** is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section
> *How a pack is selected (the current method)*. This guide carries the measured numbers for this
> checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter-faceid-plusv2_sd15.bin` (156,558,509 B, 344 tensors)
**Pack.** `ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors` (79,143,188 B, **-49.4 %**)
**SHA256 (source).** `26D0D86A1D60D6CC811D3B8862178B461E1EEB651E6FE2B72BA17AA95411E313`
**SHA256 (pack).** `3984CB16F3ED7507B60E4AC9B5EE21B10DCF8EA02AED9414ADB3535A8104C15E`

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

The score is the **observed change of the adapter's own output** when one weight is replaced by its
ConvRot INT8 reconstruction, not a static statistic of the weights.

| surface | what the runtime does | how a weight's impact is measured |
| :--- | :--- | :--- |
| `image_proj` | runs once per `ipadapter_execute`, consumes the 512-d identity embedding and the CLIP ViT-H penultimate sequence `[1, 257, 1280]`, emits 4 tokens x 768 | swap one weight, run the whole projection, compare the 4x768 stack |
| `ip_adapter` | the LoRA pairs apply a low-rank delta inside the UNet cross-attention; `to_k_ip` / `to_v_ip` project the injected stack to the K/V width | swap one weight and apply the weight (or its pair) to a fixed input of that layer's own width |

**This is v2, and the v2 branch is decided by the filename.** The file carries no `faceidplusv2` key;
the node sets `model["faceidplusv2"] = True` when the name contains `plusv2`
(`_ipadapter_model_loader`). The pack keeps `plusv2` in its name, and both arms were measured to carry
the flag, so both arms run the same branch.

**The injection frame.** The pack stores the **rotated** weight and the kernel rotates the
**activation**, so the error that reaches the output is `(q*s) @ H - W`, the original frame.

**The gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98` over the fixed 256-item
set, with finite output; the item statistic is the mean of its 4 injected tokens.

---

## 2. The loader requirement

ComfyUI needs the loader package's `HSWQ Load IPAdapter Model` node; the stock loader reads int8 codes
as magnitudes. A1111 needs `modules/ipadapter_int8_convrot.py`, which installs the same payload into
the `sd-webui-controlnet` IP-Adapter graph. What each route did with this pack, measured:

| host | measured | value |
| :--- | :--- | :--- |
| ComfyUI | patch applied to `IPAdapter` / `To_KV` | True / True |
| ComfyUI | INT8 layers installed (image_proj / ip_adapter) | 24 / 288 |
| ComfyUI | INT8 kernel calls, one projection pass | 312 |
| ComfyUI | the float source through the same node | 0 INT8 layers |
| A1111 | INT8 layers installed (image_proj / ip_adapter) | 24 / 288 |
| A1111 | INT8 kernel calls, one projection pass | 312 |
| A1111 | group sizes the kernel saw | 64, 256 |

---

## 3. The model

`image_proj` is `ProjModelFaceIdPlus`: `proj` (512 -> 1024 -> 3072, reshaped to 4 x 768), a LayerNorm
over 768, then `FacePerceiverResampler` (proj_in 1280 -> 768, four blocks of 12-head attention at head
64 plus a 4x feed-forward, proj_out, norm_out). `forward(id_embeds, clip_embeds, scale, shortcut)`
returns `x + scale * resampler(x, clip)` on the v2 path.

The node builds it with `cross_attention_dim = output_cross_attention_dim = 768`,
`clip_embeddings_dim = 1280`, `clip_extra_context_tokens = 4`, `is_faceid = True`, `is_plus = True`.
The SDXL argument set (`1280` / `2048` / `16 tokens`) builds a different module and would not load.

| module group | weight shape | modules | INT8 | group size |
| :--- | :--- | ---: | ---: | :--- |
| `image_proj.proj.0` | 1024 x 512 | 1 | 1 | 256 |
| `image_proj.proj.2` | 3072 x 1024 | 1 | 1 | 256 |
| `image_proj.perceiver_resampler.proj_in` | 768 x 1280 | 1 | 1 | 256 |
| four `layers.N.0.to_q` | 768 x 768 | 4 | 4 | 256 |
| four `layers.N.0.to_kv` | 1536 x 768 | 4 | 4 | 256 |
| four `layers.N.0.to_out` | 768 x 768 | 4 | 4 | 256 |
| four `layers.N.1.1` (feed-forward in) | 3072 x 768 | 4 | 4 | 256 |
| four `layers.N.1.3` (feed-forward out) | 768 x 3072 | 4 | 4 | 256 |
| `image_proj.perceiver_resampler.proj_out` | 768 x 768 | 1 | 1 | 256 |
| `ip_adapter.N.to_*_lora.down` (32 blocks x 4) | 128 x 320 / 640 / 768 / 1280 | 128 | 128 | 64 / 256 |
| `ip_adapter.N.to_*_lora.up` (32 blocks x 4) | 320 / 640 / 768 / 1280 x 128 | 128 | 128 | 64 / 256 |
| `ip_adapter.N.to_k_ip` | 320 x 768 | 16 | 16 | 256 |
| `ip_adapter.N.to_v_ip` | 320 x 768 | 16 | 16 | 256 |
| **total** | | **312** | **312** | 256 (124) / 64 (188) |

---

## 4. The impact measurement

One seed (42), one fixed input set per surface. The drift screen replaces one weight and records the
relative MSE of the stage output; 312 layers were measured, max `6.150512e-04`, median `2.952240e-05`,
min `4.697038e-08`. The twelve largest:

| layer | relative MSE |
| :--- | ---: |
| `image_proj.proj.0` | 6.150512e-04 |
| `image_proj.perceiver_resampler.layers.0.0.to_q` | 4.153478e-04 |
| `image_proj.proj.2` | 3.400411e-04 |
| `image_proj.perceiver_resampler.proj_in` | 2.664621e-04 |
| `ip_adapter.17.to_out_lora.down` | 7.449577e-05 |
| `ip_adapter.21.to_out_lora.down` | 7.043230e-05 |
| `ip_adapter.29.to_v_ip` | 7.032828e-05 |
| `ip_adapter.1.to_k_ip` | 6.521934e-05 |
| `ip_adapter.29.to_k_ip` | 6.444912e-05 |
| `ip_adapter.25.to_v_ip` | 6.304151e-05 |
| `ip_adapter.17.to_k_ip` | 6.215214e-05 |
| `ip_adapter.12.to_out_lora.down` | 6.190367e-05 |
| `ip_adapter.31.to_k_ip` | 6.182971e-05 |
| `ip_adapter.19.to_v_ip` | 6.133644e-05 |
| `ip_adapter.19.to_k_ip` | 6.129583e-05 |

The observable the ranking is taken from is the stage output, not the weight; the screen only says
which layers are worth ranking, and no layer crosses the 0.15 drift threshold.

---

## 5. The candidates, and the adopted set

The criterion is the aggregate contribution of one weight to the stage it feeds (1 - cosine), summed
over the 312 layers to `5.621867e-03`. The eight largest, with their share:

| layer | one minus cosine | share of the sum |
| :--- | ---: | ---: |
| `image_proj.proj.0` | 2.913358e-04 | 5.18 % |
| `image_proj.perceiver_resampler.layers.0.0.to_q` | 2.092311e-04 | 3.72 % |
| `image_proj.proj.2` | 1.425838e-04 | 2.54 % |
| `image_proj.perceiver_resampler.proj_in` | 1.330230e-04 | 2.37 % |
| `ip_adapter.14.to_v_lora.down` | 3.871607e-05 | 0.69 % |
| `ip_adapter.8.to_out_lora.down` | 3.455514e-05 | 0.61 % |
| `ip_adapter.14.to_out_lora.down` | 3.373657e-05 | 0.60 % |
| `ip_adapter.29.to_k_ip` | 3.336113e-05 | 0.59 % |

The K curve holds the top-K of that ranking in FP16 and rebuilds the pack, then runs the same 256-item
gate through the same node:

| K | INT8 | FP16 kept | size MB | gate mean | gate min | pass | share of contribution held out |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| 0 | 312 | 0 | 79.143 | 0.999945044 | 0.999930516 | yes | 0.0000 |
| 20 | 292 | 20 | 88.962 | 0.999986390 | 0.999976352 | yes | 0.2302 |
| 40 | 272 | 40 | 101.098 | 0.999986390 | 0.999976352 | yes | 0.3348 |
| 60 | 252 | 60 | 106.471 | 0.999986390 | 0.999976352 | yes | 0.4296 |

**K = 0 adopted**, by the method's own rule: the smallest pack whose cosine is not beaten by the
smaller ones, decided against the gate's margin. K=0 holds `0.009945` of mean margin; the best
larger candidate gains `4.135e-05` of mean (%,(gain_ratio).2f % of that margin) and costs `+27.328`
MB. The three larger points return identical gate numbers because the gate observes the `image_proj`
stack and they protect the same four `image_proj` layers; their extra budget goes to `ip_adapter`,
which the gate does not see and which CN-5a measures instead.

---

## 6. The pack

| item | value |
| :--- | ---: |
| 2-D weights measured | 312 |
| INT8 layers | 312 |
| 2-D weights left in float | 0 |
| tensors that are not 2-D weights, kept float | 32 |
| group size 256 | 124 layers |
| group size 64 | 188 layers |
| int8 codes differing from the replay | 0 |
| maximum code delta | 0 |
| scale mismatches | 0 |
| maximum relative scale difference | 0.0 |
| non-finite scales | 0 |
| float tensors bit-identical to the source | 32 |
| float tensors differing | 0 |
| tensors missing from the pack | 0 |
| tensors added by the pack | 0 |
| shape mismatches | 0 |
| pack bytes | 79143188 |
| source bytes | 156558509 |
| pack / source | 0.5055 |

The group size has to be read per layer: the shared IP-Adapter converter fixes 256 and leaves every
layer whose inner width is not a multiple of 256 in FP16, which on this graph is every 320 / 640 / 1280
layer - the first run of that converter produced 0 INT8 layers on a 312-layer graph. This pack's
converter takes the largest power of four that divides the inner width, which here is 256 or 64.

---

## 7. The acceptance

Both hosts were measured. A1111 builds the graph from the pack and installs the payload; ComfyUI does
the same through the loader node. Neither is a quality claim: the acceptance is CN-5a / CN-5b / CN-6
and the render, below.

| item | value |
| :--- | ---: |
| A1111: image_proj Linear modules / INT8 | 24 / 24 |
| A1111: ip_layers Linear modules / INT8 | 288 / 288 |
| A1111: INT8 kernel calls on one projection pass | 312 |
| A1111: group sizes the kernel saw | 64, 256 |
| A1111: input widths the kernel saw | 128, 320, 512, 640, 768, 1024, 1280, 3072 |
| A1111: every call with convrot set | True |
| A1111: injected stack output | [1, 4, 768] |
| A1111: finite | True |
| ComfyUI: patch applied to IPAdapter / To_KV | True / True |
| ComfyUI: INT8 layers installed (image_proj / ip_adapter) | 24 / 288 |
| ComfyUI: INT8 kernel calls on one projection pass | 312 |
| ComfyUI: image_proj cosine (mean / min) | 0.999930672 / 0.999916662 |
| ComfyUI: to_kv projections / cosine mean min / min min | 288 / 0.999905985 / 0.999821822 |
| ComfyUI: the float source through the same node | 0 INT8 layers |
| gate n | 256 |
| gate mean | 0.999945044 |
| gate min | 0.999930516 |
| gate p5 | 0.999938630 |
| gate p50 | 0.999945320 |
| gate kernel calls | 6144 |
| gate pass | True |
| CN-5a image_proj cosine (mean / min) over four CLIP inputs | 0.998930 / 0.985521 |
| CN-5a image_proj relative MSE max | 9.774e-03 |
| CN-5a ip_adapter elements / cosine mean / min | 288 / 0.999974 / 0.999960 |
| trigger candidates | 56 |
| trigger flagged | 0 |
| trigger control band (natural control max x 1.5) | 1.079442e-04 |
| trigger largest candidate | 8.364785e-05 |
| trigger negative control (source against itself) | 6.960186e-05 |
| trigger detection floor, code axis | 4 LSB on 10 % of the codes |
| trigger detection floor, scale axis | x1.05 of the layer's weight_scale |
| render pixel mean abs diff / max / p99 | 12.981 / 244 / 92 |
| render pixel cosine / PSNR | 0.987881 / 21.008 dB |
| render identical-pixel ratio | 1.38 % |

---

## 8. Reproduction

```
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe
#   COMFY   = <ComfyUI>
#   LOADER  = the ComfyUI-HSWQ-Loader-and-Tools checkout
#   A1111   = the A1111 root

python clip_convert/diag_impact_ipadapter_faceid_plusv2.py --ipadapter <f.bin> --out impact.json --comfy <COMFY>
python clip_convert/measure_ipadapter_faceid_plusv2_contrib.py --ipadapter <f.bin> --out contrib.json --comfy <COMFY>
python clip_convert/sweep_ipadapter_faceid_plusv2.py --impact impact.json --contrib contrib.json --input <f.bin> --work <dir> --ks 0 20 40 60
python clip_convert/convert_ipadapter_faceid_plusv2_convrot_int8.py --input <f.bin> --impact impact.json --output <pack.safetensors> --record conversion.json

python audit/ipadapter_faceid_plusv2_pack_audit.py --source <f.bin> --pack <pack> --comfy <COMFY> --json cn1.json
python audit/ipadapter_faceid_plusv2_comfyui_route.py --pack <pack> --source <f.bin> --comfy <COMFY> --loader <LOADER> --dir <dir> --json cn2_comfyui.json
python audit/a1111_ipadapter_faceid_plusv2_route.py --a1111 <A1111> --pack <pack> --json cn2_a1111.json
python audit/ipadapter_faceid_plusv2_acceptance.py --pack <pack> --source <f.bin> --comfy <COMFY> --json cn5a.json
python audit/ipadapter_faceid_plusv2_comfyui_gate.py --pack <pack> --source <f.bin> --comfy <COMFY> --loader <LOADER> --dir <dir> --image <ref.png> --clip-vision <clip_vision_h> --json cn6.json
python audit/ipadapter_faceid_plusv2_comfyui_trigger.py --pack <pack> --source <f.bin> --comfy <COMFY> --loader <LOADER> --dir <dir> --image <ref.png> --clip-vision <clip_vision_h> --json cn5b.json
python audit/ipadapter_faceid_plusv2_render.py --server <host:port> --out-dir <dir> --json cn4.json --output-root <server output dir>
```

The render needs a running server; it ran on a second instance (`--port 8189 --lowvram
--disable-assets`) while another instance was already running, so both arms ran on one host with
identical settings. The A1111 route measurement imports the install's classes and does not start a
server.

---

## 9. Error sources and limits

| limit | what it means |
| :--- | :--- |
| the gate observes `image_proj` only | the 288 `ip_adapter` weights are outside it and are covered by CN-1, CN-2 and CN-5a |
| one identity | the 256 items vary the CLIP side; a second face would be a second instrument |
| K=20/40/60 gate numbers are identical | a property of the instrument (same four `image_proj` layers), not three confirmations |
| the trigger search | cannot see a trigger trained into the weights before quantisation |
| the positive-control floor | code axis 4 LSB on 10 % of the codes; scale axis x1.05 of the layer's own scale; the weaker axis is the scale axis |
| `image_proj.proj.0` | widens 512 -> 3072 and carries 5.18 % of the contribution; the measured effect on the injected stack is %.6f mean / %.6f min over four CLIP inputs |
| the render | one seed, one prompt, 12 steps, 512x512; the pixel cosine %.6f is larger than the gate margin |
| the drift threshold | protects nothing here: the largest drift is %.2e against 0.15, so the decision is the K curve |

---

## 10. The experiment ledger

| step | what ran | outcome |
| :--- | :--- | :--- |
| 1 | drift screen, 312 layers | max %.6e / median %.6e / min %.6e |
| 2 | contribution criterion, 312 layers | sum %.6e, top share %.2f % |
| 3 | K curve 0 / 20 / 40 / 60 | all pass; K=0 adopted |
| 4 | conversion, K=0 | %d INT8 / %d protected / %d float kept |
| 5 | CN-1 replay | %d differing integers, %d scale mismatches, %d/%d float bit-identical |
| 6 | CN-1 behaviour | %.6f mean / %.6f min on the injected stack |
| 7 | CN-2 ComfyUI | %d installed, %d kernel calls |
| 8 | CN-2 A1111 | %d installed, %d kernel calls |
| 9 | CN-5a | image_proj %.6f / %.6f, ip_adapter %.6f / %.6f |
| 10 | CN-6 gate | mean %.9f / min %.9f, %d kernel calls |
| 11 | CN-5b trigger | %d candidates, %d flagged |
| 12 | CN-4 render | both arms finished images, pixel cosine %.6f |

---

## 11. Files

| file | role |
| :--- | :--- |
| `clip_convert/diag_impact_ipadapter_faceid_plusv2.py` | the drift screen, 312 layers |
| `clip_convert/measure_ipadapter_faceid_plusv2_contrib.py` | the criterion: the aggregate contribution |
| `clip_convert/sweep_ipadapter_faceid_plusv2.py` | the K curve, four packs built and gated |
| `clip_convert/convert_ipadapter_faceid_plusv2_convrot_int8.py` | the converter, group read per layer |
| `audit/ipadapter_faceid_plusv2_pack_audit.py` | CN-1 |
| `audit/ipadapter_faceid_plusv2_comfyui_route.py` | CN-2 / CN-3 on ComfyUI |
| `audit/a1111_ipadapter_faceid_plusv2_route.py` | CN-2 on A1111 |
| `audit/ipadapter_faceid_plusv2_acceptance.py` | CN-5a |
| `audit/ipadapter_faceid_plusv2_comfyui_gate.py` | CN-6 |
| `audit/ipadapter_faceid_plusv2_comfyui_trigger.py` | CN-5b, both positive-control axes |
| `audit/ipadapter_faceid_plusv2_render.py` | CN-4 |
| `records/cn_ipadapter_faceid_plusv2_impact.json` | the drift screen numbers |
| `records/cn_ipadapter_faceid_plusv2_contrib.json` | the contribution numbers |
| `records/cn_ipadapter_faceid_plusv2_curve.json` | the four K points with size and gate |
| `records/cn_ipadapter_faceid_plusv2_protection.json` | the protection decision and its reason |
| `records/cn_ipadapter_faceid_plusv2_conversion.json` | the per-layer outcome and both hashes |
| `records/cn_ipadapter_faceid_plusv2_pack_audit.json` | CN-1 numbers |
| `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` | CN-2 / CN-3 on ComfyUI |
| `records/cn_ipadapter_faceid_plusv2_a1111_route.json` | CN-2 on A1111 |
| `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` | CN-5a |
| `records/cn_ipadapter_faceid_plusv2_gate_256.json` | CN-6 at K=0 |
| `records/cn_ipadapter_faceid_plusv2_gate_256_k20.json` | CN-6 at K=20 |
| `records/cn_ipadapter_faceid_plusv2_gate_256_k40.json` | CN-6 at K=40 |
| `records/cn_ipadapter_faceid_plusv2_gate_256_k60.json` | CN-6 at K=60 |
| `records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` | CN-5b, both axes |
| `records/cn_ipadapter_faceid_plusv2_render.json` | CN-4, the two arms and how they were looked at |
| `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` | CN-4 pixel statistics |
| `records/cn_ipadapter_faceid_plusv2_hf_upload.json` | the published bytes and the Hub's own size and LFS digest |
