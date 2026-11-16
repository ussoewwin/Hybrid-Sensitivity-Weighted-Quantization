# SDXL IP-Adapter FaceID Plus v2 - ConvRot INT8 Impact Technical Guide

**Pack.** `ip-adapter-faceid-plusv2_sdxl_hswq_convrot_int8.safetensors` (748,384,560 B,
50.3 % of the source)
(`records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `identity.pack_bytes`)

**SHA256 (source).** `C6945D82B543700CC3CCBB98D363B837E9C596281607857C74B713A876DAF5FB`

**SHA256 (pack).** `C9376BC5C5535EE43A96442049E3B7E0A451439209C696A45DA5EA09A23323A3`

This guide is the runbook for this pack: what it is, where it loads, what was measured, and how to
reproduce it. The verdict is `md/backdoor/verdicts/CN_IPAdapter_FaceID_PlusV2_SDXL_Pack.md`; the
quantisation record is `md/IPAdapter_FaceID_PlusV2_SDXL_ConvRot_INT8_Quantization_and_Score_Report.md`.

## 1. What the pack is

| item | value |
| :--- | :--- |
| source | `ip-adapter-faceid-plusv2_sdxl.bin`, 1,487,555,181 B |
| pack | 748,384,560 B, 1,284 INT8 layers + 32 float tensors |
| format | `int8_tensorwise`, `convrot: true`, group size read per layer |
| group sizes | 256 for 664 layers, 64 for 620 |
| protected | 0 (nothing crossed the drift threshold) |
| graph | `cross_attention_dim` 2048, `clip_embeddings_dim` 1280, `clip_extra_context_tokens` 4 |

## 2. Why 2048 and not 768

The SDXL adapter's `proj` is 512 -> 1024 -> 8192 reshaped to 4 x 2048, and the resampler runs 20 heads.
The SD1.5 width of 768 builds a different module and fails on the first weight. The loader decides the
width from the host model, not from the file.

## 3. Why the group size is read per layer

This graph's inner widths are 128, 512, 640, 1024, 1280, 2048 and 8192. A fixed 256 leaves every 128- and
640-wide layer in FP16 and produces a pack far smaller in INT8 coverage than the graph - 1,284 2-D weights
here, 1284 of them quantised. The converter reads the largest
power of four that divides the width.

## 4. Where it loads

| host | node / module | what it does |
| :--- | :--- | :--- |
| ComfyUI | `HSWQ Load IPAdapter Model` (`nodes/hswq_load_ipadapter_model.py`) + `IPAdapterFaceID` | installs the ConvRot patch on `IPAdapter` / `To_KV`, builds `ConvRotInt8Linear`, keeps the codes INT8 |
| A1111 | `extensions-builtin/sd-webui-controlnet` IP-Adapter path + `modules/ipadapter_int8_convrot.py` | strips the sidecars, installs the int8 payload in place, binds `comfy_kitchen.int8_linear(convrot=True)` |

Measured on both: 1,260 + 24 layers installed,
1,284 kernel calls on one projection pass, group sizes 64/256, 7 distinct
input widths (128 … 8192), every call with the convrot flag.

## 5. What was measured, and how

| step | instrument | the number it produced |
| :--- | :--- | :--- |
| the drift screen | one weight at a time, ConvRot INT8 reconstruction, stage output relMSE | max 8.632594e-05, median 2.053594e-05 |
| the criterion | the same, scored as 1 - cosine on the stage output | sum 0.01704065 over 1,284 layers |
| CN-1 | replay of the converter's arithmetic | 0 code differences, 0 float-side differences, FAITHFUL |
| CN-2 / CN-3 | both hosts, classes imported not re-implemented | 1,284 kernel calls on each |
| CN-5a | source against pack, both surfaces | 0.999960350 / 0.999972070 mean |
| CN-6 | 256 fixed items, CLIP-H, one fixed face embedding | mean 0.999926916 / min 0.999925345 |
| CN-5b | 56 candidates against natural-transform controls | 0 flagged, floor 2 LSB / x1.05 |
| CN-4 | two arms, one prompt, both looked at | pixel cosine 0.962361634 |

## 6. Two traps, both measured

**float16 overflows the projection.** With `N(0,1)` CLIP inputs the 8192-wide `proj` returns NaN in
float16 - on the source as well as the pack. Every projection measurement here is float32. A NaN on this
graph is not a pack defect until the source has been measured the same way.

**The gate cannot see the `ip_adapter` layers.** It calls `image_proj`, 24 layers. That is why K=40, 80 and
160 return identical gate statistics: they protect `ip_adapter` weights. The curve is real, and it is flat.

## 7. Choosing the base model

The render used `waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors` through `HSWQ Checkpoint Loader` with
`weight_dtype: SDXL ConvRot INT8`, plus a separate `sdxl_vae.safetensors`. That file carries its own
SDXL CLIP (`conditioner.embedders.0/1`), which is why no CLIP loader appears in the graph - a property read
from its header, not from its name. The inventory behind that choice is
`records/cn_ipadapter_faceid_plusv2_sdxl_model_inventory.json`
(`records/cn_ipadapter_faceid_plusv2_sdxl_model_inventory.json` -> `counts.entries`): 1,307 entries
across four model roots, classified by container and key prefixes, with 38 placeholders and
166 unreadable entries counted rather than dropped.

## 8. Reproduction

```
python clip_convert/diag_impact_ipadapter_faceid_plusv2_sdxl.py --ipadapter <f.bin> --out <impact.json> --comfy <ComfyUI>
python clip_convert/measure_ipadapter_faceid_plusv2_sdxl_contrib.py --ipadapter <f.bin> --out <contrib.json> --comfy <ComfyUI>
python clip_convert/sweep_ipadapter_faceid_plusv2_sdxl.py --impact <impact.json> --contrib <contrib.json> --input <f.bin> --work <dir> --ks 0 40 80 160
python audit/ipadapter_faceid_plusv2_sdxl_pack_audit.py --source <f.bin> --pack <pack> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_comfyui_route.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader> --dir <dir> --json <record>
python audit/a1111_ipadapter_faceid_plusv2_sdxl_route.py --a1111 <A1111> --pack <pack> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_acceptance.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_comfyui_gate.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader> --dir <dir> --image <ref.png> --clip-vision <f> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_comfyui_trigger.py --pack <pack> --source <f.bin> --comfy <ComfyUI> --loader <loader> --dir <dir> --image <ref.png> --clip-vision <f> --json <record>
python audit/ipadapter_faceid_plusv2_sdxl_render.py --server http://127.0.0.1:8189 --out-dir <dir> --output-root <server output> --json <record>
```

## 9. Error sources and limits

* the gate observes `image_proj` only; the 1,260 `ip_adapter` weights are measured on fixed inputs
* one face, one base model, one prompt
* float16 projection measurements are invalid on this graph (section 6)
* the trigger instrument's floor is 2 LSB on the code axis and
  x1.05 on the scale axis; the scale axis is the weak one
* the model inventory is a snapshot of one machine on one day

## 10. Ledger

| time | step | record |
| :--- | :--- | :--- |
| 2026-11-16 | drift screen, criterion, curve, CN-1, both routes, CN-5a, CN-6 x4, CN-5b, CN-4 | the seventeen records beside the verdict |

## 11. Files

| file | role |
| :--- | :--- |
| `clip_convert/diag_impact_ipadapter_faceid_plusv2_sdxl.py` | the drift screen |
| `clip_convert/measure_ipadapter_faceid_plusv2_sdxl_contrib.py` | the criterion |
| `clip_convert/sweep_ipadapter_faceid_plusv2_sdxl.py` | the K curve |
| `clip_convert/convert_ipadapter_faceid_plusv2_sdxl_convrot_int8.py` | the converter, group size read per layer |
| `audit/ipadapter_faceid_plusv2_sdxl_pack_audit.py` | CN-1 |
| `audit/ipadapter_faceid_plusv2_sdxl_comfyui_route.py` | CN-2 / CN-3 on ComfyUI |
| `audit/a1111_ipadapter_faceid_plusv2_sdxl_route.py` | CN-2 on A1111 |
| `audit/ipadapter_faceid_plusv2_sdxl_acceptance.py` | CN-5a |
| `audit/ipadapter_faceid_plusv2_sdxl_comfyui_gate.py` | CN-6 |
| `audit/ipadapter_faceid_plusv2_sdxl_comfyui_trigger.py` | CN-5b |
| `audit/ipadapter_faceid_plusv2_sdxl_render.py` | CN-4 |
| `records/cn_ipadapter_faceid_plusv2_sdxl_model_inventory.json` -> `counts.entries` | the section 1.10 inventory |

## Retraction

None. This is the first record for this checkpoint.
