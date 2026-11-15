# CLIP-ViT-bigG-14 (`clip_vision_g.safetensors`) - ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the behavioural observable, the
> size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section
> *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/clip_vision/clip_vision_g.safetensors` (3,689,911,098 B, 777 tensors, 288 runtime Linears)
**Pack.** `clip_vision_g_hswq_convrot_int8.safetensors` (1,852,954,770 B, **-49.8 %**)
**SHA256 (source).** `9908329B3EAD722A693EA400FAB1D7C9EC91D6736FD194A94D20D793457F9C2E`
**SHA256 (pack).** `70A39F1264FBC7870B246666C931690CB754A9A99F11CF210F5C6DF207D8ADBB`

The checkpoint's 776 parameter tensors are byte-identical to the published
`CLIP-ViT-bigG-14-laion2B-39B-b160k` weights, so the file measured here is that object, not a
local variant.

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

**The observable.** A vision tower has a forward of its own, and the runtime consumes exactly one
thing from it: the image embedding, either the pooled projection (`visual_projection`, 1664 -> 1280)
or the full token stack the cross-attention path reads. The observable is the stage output the
weight feeds:

- for an encoder Linear, the cosine between the tower's embedding before and after that layer's
  reconstruction, over a fixed image set;
- for the pooler, the same cosine on the pooled vector.

**The injection frame matters.** The pack stores the **rotated** weight and the kernel rotates the
**activation**; the error that reaches the output is therefore `(q*s) @ H - W`, the **original**
frame. A measurement taken on the rotated frame reports the quantiser's internal error, not the
model's.

**Two rankings, one criterion.** The per-layer drift is the *screen*: it says which layers the
quantisation moves at all. The aggregate contribution is the *criterion*: with the all-INT8 pack
loaded, each layer is restored to fp16 in place and the tower re-run, so the number is the gain that
restoring that layer actually buys. The method selects on the second, and this guide reports both
because they disagree on this checkpoint.

**The acceptance gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98` over the
fixed 256-item set (seed 42), with finite output, measured through the route that installs the pack.

---

## 2. The loader requirement

The pack's weights are `int8_tensorwise` with `convrot` stamps and a group size of 64. A loader
that reads the file with `load_torch_file(safe_load=True)` and never interprets `comfy_quant` hands
the int8 code tensors to the model's float modules.

**Measured on this machine, ComfyUI's own `CLIPVisionLoader`:** it builds the tower with
`manual_cast`, which does not consume the stamps. The int8 codes arrive inside a float Parameter
and the tower runs on them as if they were weights. That is not a degraded pack - it is a different
model:

**Measured on this machine, ComfyUI path:** `CLIPVisionLoader` builds the tower with `manual_cast`,
which does not consume the stamps, so **288 declared weights arrive as raw code tensors inside float
modules and 0 INT8 GEMM calls run in one forward**; the output cosine against the source is
**-0.035036381** (`records/cn_clip_vision_bigg_route_stock.json` -> `output_cos_source_vs_pack`).
That path does not take the pack.

**`HSWQLoadClipVision` does take it**, and both routes were run against this exact file
(`records/cn_clip_vision_bigg_route.json`): **288 of 288 declared layers load as int8**, **288 INT8
GEMM calls** run in one forward, and the output cosine against the source is **0.997366786**
(`records/cn_clip_vision_bigg_route.json` -> `output_cos_source_vs_pack`).

The render shows the same split: the pack's picture on the extension's route is the source's scene
(370,949 B against 370,548 B, mean 135.93 against 135.66), and on the native route it is a different
picture (301,123 B, mean 107.60).

**The requirement, stated as a requirement.** This pack is only this pack on a loader that consumes
`comfy_quant`. `HSWQLoadClipVision` does; `CLIPVisionLoader` does not. A number measured on the
second route is a number about the float modules, and this guide does not report one.

---

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 3,689,911,098 B, 777 tensors |
| the runtime Linear modules | **288**, all of them quantisable |
| the transformer | 32 layers, hidden size 1664, image 224, patch 14 |
| the group size | **64** - the 1664-wide layers cannot carry a 256-way group |
| the two weights the format cannot take | `vision_model.embeddings.position_embedding` (rows wider than the group), `visual_projection` (1664 -> 1280, asymmetric) |
| the kept float tensors | 489, all bit-identical to the source, 0 dtype differences |

The group size is not a choice. `1664 = 4 x 416 = 64 x 26`; a 256-way Hadamard group needs the
inner dimension to divide by 256, and 1664 does not. 64 is the largest power of four that does, and
it is what every INT8 layer in this pack uses.

---

## 4. The impact measurement

Two measurements, both on this checkpoint, both recorded.

**The per-layer drift** (the screen), `clip_convert/diag_impact_clip_vision.py`, 288 layers, seed
42, the reconstruction injected in the **original** frame:

| summary | measured |
| :--- | ---: |
| layers measured | **288 of 288** |
| the median | 1.1163e-07 |
| the p90 | 1.9330e-06 |
| the max | 9.1210e-06 (`encoder.layers.45.mlp.fc1`) |
| the layers above 0.15 | **0** |

**The aggregate contribution** (the criterion),
`clip_convert/measure_bigg_vision_contrib_inmodel.py`: the all-INT8 pack is built and loaded once,
and each layer is replaced in place by the source's fp16 weight while the tower runs over 8 fixed
seeds.

| summary | measured |
| :--- | ---: |
| the all-INT8 reference | mean 0.999732, min 0.999692 |
| the largest contribution | 3.3081e-05 (`encoder.layers.28.self_attn.q_proj`) |
| the median contribution | -5.1260e-06 |
| **the layers with a negative contribution** | **173 of 288** |

**Why the in-model form.** The per-layer-pack form (`measure_bigg_vision_contrib.py`) builds a fresh
pack per candidate layer and reloads it: about half a minute per layer, and the tower is not
released between packs, so VRAM fills as the sweep proceeds. The in-model form measures the same
thing with one pack resident - 288 layers in about seven minutes. The two were checked against each
other on one layer, in one process:

| the method | the gain for `encoder.layers.0.self_attn.q_proj` |
| :--- | ---: |
| the per-layer pack | +1.3888e-05 |
| the in-model swap | +1.3888e-05 |
| the difference | **+0.0000e+00** |

**The negative majority is the finding, not a fault of the instrument.** Restoring one layer of 288
to fp16 does not move the output over most of the population, so the contribution curve is flat by
construction - see section 5. The two methods agreeing to 0.0000e+00 on the same layer is what rules
out the instrument as the explanation.

---

## 5. The candidates, and the adopted set

Six candidates from the contribution ranking, all measured on the full 256-item acceptance set
(seed 42) through the `HSWQLoadClipVision` route, plus the pack being replaced for reference:

| the FP16-kept | INT8 | cos mean | cos min | p5 | size | mean gain / margin |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **0 (adopted)** | **288** | **0.999734** | **0.999554** | **0.999654** | **1767.1 MB** | **the floor** |
| 10 | 278 | 0.999747 | 0.999545 | 0.999677 | 1824.5 MB | 0.14 % |
| 20 | 268 | 0.999763 | 0.999663 | 0.999709 | 1923.3 MB | 0.30 % |
| 30 | 258 | 0.999802 | 0.999578 | 0.999737 | 2001.3 MB | 0.70 % |
| 35 | 253 | 0.999806 | 0.999719 | 0.999762 | 2035.2 MB | 0.74 % |
| 40 | 248 | 0.999808 | 0.999674 | 0.999752 | 2058.7 MB | 0.76 % |
| the pack being replaced | 253 | 0.999910 | 0.999725 | 0.999875 | 2180.1 MB | 1.8 % |

`margin = 0.999734 - 0.99 = 9.734e-03` - the distance the gate already has at the floor. The last
column is the share of that margin a larger candidate buys.

**The floor is the adopted point.** It passes the gate with 0.009734 of margin on the mean and
0.019554 on the minimum, it is the smallest candidate measured, and the next 291.6 MB of protection
buys 0.76 % of the margin that already exists. On this checkpoint protection does not buy quality,
because 173 of 288 layers have a negative contribution: keeping them in fp16 costs bytes and returns
nothing.

| inner dimension | what the group is | groups |
| :--- | :--- | ---: |
| 1664 | the encoder Linears (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`) | **288** |
| 1664 | `visual_projection` - 1664 -> 1280, asymmetric | **0** (kept FP16) |
| 1024 | `position_embedding` - rows wider than the group | **0** (kept FP16) |

**No layer was kept for quality reasons.** The two FP16 tensors are kept by the format rule alone,
and the sweep says no layer would have needed protection anyway (0 above 0.15).

**The kept tensors.** 0 layers are kept from the ranking. 2 tensors are kept because the format
cannot take them (`position_embedding`, `visual_projection`). The 489 float tensors besides the
scales are the source's, bit for bit (0 differences, max abs diff 0.0). The INT8 `weight_scale`
stays FP32 because it carries the dequantisation precision.

---

## 6. The pack

Built with `clip_convert/convert_clip_impact_convrot_int8.py`:

```
--clip_type vision --drift-threshold 0.05 --groupsize 64 \
  --protect-list impact/bigg_g_protect_0.json
```

(the protect list is empty - the floor has no protected layer)

| item | measured |
| :--- | :--- |
| the file | `clip_vision_g_hswq_convrot_int8.safetensors` |
| the size | 1,852,954,770 B (1767.1 MiB) |
| against the source | 3,689,911,098 B - **-49.8 %** |
| the tensors | 1,353 |
| the INT8 layers | **288** |
| the FP16 kept layers | 2 by format, 0 from the ranking |
| the group size | 64 |
| the `int8_linear` calls per forward | **288** |
| the verdict | FAITHFUL |

The audit replays the converter's own arithmetic and compares element by element
(`records/cn_clip_vision_bigg_pack_audit.json`):

| check | measured |
| :--- | :--- |
| INT8 layers declared / found | 288 / 288 |
| integers differing from the replay | **0** |
| largest integer delta | 0 |
| scale exact relative max | **0.0** |
| non-finite scales | 0 |
| float tensors checked | **489** |
| float mismatches | **0**, max abs diff 0.0 |
| dtype differences | **0** |

---

## 7. The acceptance

Six checks, run on the route that installs the pack, each with its record.

**CN-1 - faithfulness.** 288 of 288 layers replayed from the source; 0 integer differences, scale
relative max 0.0, 489 float tensors bit-identical, 0 dtype changes. The verdict is FAITHFUL
(`records/cn_clip_vision_bigg_pack_audit.json`).

**CN-2 - the route census.** `HSWQLoadClipVision` installs **288 of 288** declared layers and the
kernel fires **288** times in one forward; ComfyUI's `CLIPVisionLoader` reads no `comfy_quant` and
hands 288 raw code tensors to float modules, 0 kernel calls (`records/cn_clip_vision_bigg_route.json`,
`records/cn_clip_vision_bigg_route_stock.json`).

**CN-3 - execution.** Output finite; the tower's embedding against the source measures cos
**0.997366786** on the INT8 route and **-0.035036381** on the native route.

**CN-4 - the render, read as an image.** Source and pack drawn on the same route, same prompt, same
seed:

| arm | PNG | mean | inspected |
| :--- | ---: | ---: | :--- |
| the source | 370,548 B | 135.66 | the reference scene |
| the pack, HSWQLoadClipVision | 370,949 B | 135.93 | **the same scene, no artefacts, no noise** |
| the pack, CLIPVisionLoader | 301,123 B | 107.60 | a different picture |

| number | measured | record |
| :--- | :--- | :--- |
| **16.858789** | mean absolute pixel difference, source against pack on HSWQLoadClipVision (/255) | `records/cn_clip_vision_bigg_render_diff.json` -> `per_route.hswq.mean_abs` |
| **86.884178** | mean absolute pixel difference, source against pack on CLIPVisionLoader (/255) | `records/cn_clip_vision_bigg_render_diff.json` -> `per_route.stock.mean_abs` |
| 95.5624 | pixels changed, HSWQLoadClipVision (%) | `records/cn_clip_vision_bigg_render_diff.json` -> `per_route.hswq.changed_pct` |
| 99.9809 | pixels changed, CLIPVisionLoader (%) | `records/cn_clip_vision_bigg_render_diff.json` -> `per_route.stock.changed_pct` |
| **0.0** | the source frame on the two loaders, mean absolute difference (/255) | `records/cn_clip_vision_bigg_render_diff.json` -> `cross.source_hswq_vs_source_stock.mean_abs` |
| 254 | largest single-channel difference, HSWQLoadClipVision | `records/cn_clip_vision_bigg_render_diff.json` -> `per_route.hswq.max_abs` |
| 255 | largest single-channel difference, CLIPVisionLoader | `records/cn_clip_vision_bigg_render_diff.json` -> `per_route.stock.max_abs` |

The images were looked at, not inferred from a cosine. A benchmark number between two noise images
also reads 0.99; the judgement here is on the decoded picture.

**CN-5a - source against pack on the model's own observable.** cos 0.997366786 through 288 kernel
calls (`records/cn_clip_vision_bigg_route.json` -> `output_cos_source_vs_pack`).

**CN-5b - the trigger search, with a measured positive control.** 55 candidates (45 patch, 6
frequency, 4 natural controls), the divergence measured as 1 - cos against the pack's own embedding
of the unmodified image (`records/cn_clip_vision_bigg_trigger.json`):

| item | measured |
| :--- | ---: |
| candidates | 55 |
| control p99 | **5.061980e-03** |
| candidates above the control p99 | **7** |
| the largest candidate | 7.021394e-03 |
| the detection floor | **relative strength 0.1**, at 1-cos 7.613862e-03 |

All seven sit under the floor. The support is **"no trigger above a relative strength of 0.1"**, not
"no trigger". The positive control ran on `vision_model.embeddings.position_embedding` - the layer
the observable passes through - because a ConvRot layer holds a `QuantizedTensor` and a float
implant written into it never reaches the kernel.

**CN-6 - the 256-item gate.**

| item | measured |
| :--- | ---: |
| 256-item cos mean | **0.999734** |
| 256-item cos min | **0.999554** |
| 256-item p5 | 0.999654 |
| the 256 transforms | mean 0.997029, min 0.985745 |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** |
| INT8 kernel calls over the gate | 288 per item |

---

## 8. Reproduction

```powershell
#   python = <ComfyUI>/python_embeded/python.exe

# 1. the drift sweep (the screen)
python clip_convert/diag_impact_clip_vision.py `
    --model "<src>/clip_vision_g.safetensors" `
    --out "<repo>/impact/impact_bigg_g_vision.json" `
    --comfy_path "<ComfyUI>" --groupsize 256 --seed 42 --full

# 2. the aggregate contribution (the criterion), one pack resident
python clip_convert/measure_bigg_vision_contrib_inmodel.py `
    "<src>/clip_vision_g.safetensors" "<repo>/impact/impact_bigg_g_vision.json" `
    "<repo>/clip_convert/convert_clip_impact_convrot_int8.py" "<work>" `
    "<repo>/impact/contrib_bigg_g.json"

# 3. the floor
python clip_convert/convert_clip_impact_convrot_int8.py `
    --input "<src>/clip_vision_g.safetensors" --output "<out>/k0.safetensors" `
    --impact "<repo>/impact/impact_bigg_g_vision.json" `
    --comfy_path "<ComfyUI>" --clip_type vision --drift-threshold 0.05 --groupsize 64 `
    --protect-list "<repo>/impact/bigg_g_protect_0.json"

# 4. CN-1 (faithfulness) and CN-2/CN-3 (route, kernel)
python audit/bigg_pack_audit.py --source "<src>/clip_vision_g.safetensors" `
    --pack "<out>/k0.safetensors" --comfy "<ComfyUI>" --json records/cn_clip_vision_bigg_pack_audit.json
python audit/clip_vision_bigg_route.py --source "<src>/clip_vision_g.safetensors" `
    --pack "<out>/k0.safetensors" --image "<ref>.png" --comfy "<ComfyUI>" `
    --loader "<node>" --route hswq --json records/cn_clip_vision_bigg_route.json

# 5. CN-4 (render, three arms)
python audit/clip_vision_bigg_render.py --comfy "<ComfyUI>" --server <host:port> `
    --reference "<ref>.png" --route hswq --out records --frames "<dir>"

# 6. CN-5b and CN-6, with the positive-control ladder
python audit/clip_vision_bigg_gate_and_trigger.py --source "<src>/clip_vision_g.safetensors" `
    --pack "<out>/k0.safetensors" --image "<ref>.png" --comfy "<ComfyUI>" --loader "<node>" `
    --route hswq --n 256 `
    --json-trigger records/cn_clip_vision_bigg_trigger.json `
    --json-gate records/cn_clip_vision_bigg_gate.json
```

**The step everyone gets wrong:** `--groupsize 64`. The 1664-wide layers cannot carry a 256-way
group; a 256 run builds nothing or silently falls back.

**The second step:** the pack lives next to its source in `models/clip_vision/`; a render also needs
it reachable from `models/clip/`.

**The third step:** the positive control. A ConvRot layer holds a `QuantizedTensor`, so a float
implant cannot be written into one - the ladder runs on `position_embedding`, which the observable
passes through. Its measured floor is a relative strength of 0.1 (1-cos 7.613862e-03).

---

## 9. Error sources and limits

- **The native `CLIPVisionLoader` does not run this pack as INT8.** Section 2 measures 0 kernel calls
  and cos -0.035036381 on that route. Every acceptance number in section 7 was produced on
  `HSWQLoadClipVision`; none of them are claimed for the native route, and a user who loads the pack
  natively gets a different model, not a worse one.
- **The pack is a pack of one tower.** It quantises the vision encoder. The text encoder beside it in
  a dual-encoder checkpoint is a different file with its own record.
- **The contribution ranking is one measurement per layer.** Each layer is restored alone against an
  all-INT8 baseline; interactions between two restored layers are not measured. On this checkpoint
  173 of 288 contributions are negative, so an interaction term would have to be large to change the
  selection, but it is not measured.
- **`detection floor 0.1` bounds the trigger search.** A conditional behaviour that moves the
  embedding by less than a relative strength of 0.1 is invisible to the instrument used here. The
  seven flagged candidates are all under it; the honest statement is "none above the floor".
- **The render is one image, one prompt, one seed.** It shows the pack reproduces a scene and that the
  native route does not; it does not bound quality across a workload.
- **The 256-item set is fixed (seed 42).** A different set would move the fourth decimal, not the
  gate.
- **Nothing here bounds the source's own content.** The reference class is the checkpoint's bytes; a
  poison already inside the published release is out of reach. See the verdict's reference-class
  boundary.

---

## 10. The experiment ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `clip_convert/diag_impact_clip_vision.py` | 288 of 288 layers, median 1.1163e-07, p90 1.9330e-06, max 9.1210e-06, 0 above 0.15 |
| 2 | `clip_convert/measure_bigg_vision_contrib_inmodel.py` | 288 layers, max 3.3081e-05, median -5.1260e-06, **173 negative**; in-model vs per-layer-pack difference +0.0000e+00 on the checked layer |
| 3 | `clip_convert/convert_clip_impact_convrot_int8.py` | k=0: 1,852,954,770 B, 1,353 tensors, 288 INT8, 2 kept by format, -49.8 %; k=10/20/30/35/40 measured for the curve |
| 4 | `clip_convert/verify_bigg_vision.py` + `audit/clip_vision_bigg_gate_and_trigger.py` | 256 items: mean 0.999734, min 0.999554, p5 0.999654, PASS; 256 transforms 0.997029 / 0.985745 |
| 5 | `audit/bigg_pack_audit.py` | CN-1 FAITHFUL, 288/288, q_mismatch 0, scale 0.0, float 489/489 bit-identical, dtype 0 |
| 5b | `audit/clip_vision_bigg_route.py` | CN-2/CN-3: hswq 288/288 installed, 288 kernel calls, cos 0.997366786; native 288 raw codes, 0 calls, cos -0.035036381 |
| 6 | `audit/clip_vision_bigg_render.py` | CN-4: source 370,548 B / pack 370,949 B on hswq, both inspected; native 301,123 B |
| 7 | `audit/clip_vision_bigg_gate_and_trigger.py` | CN-5b: 55 candidates, control p99 5.061980e-03, 7 above, largest 7.021394e-03, floor 0.1 (1-cos 7.613862e-03) on `position_embedding` |

---

## 11. Files

| file | what it is |
| :--- | :--- |
| `clip_convert/diag_impact_clip_vision.py` | the drift measurement (the screen) |
| `clip_convert/measure_bigg_vision_contrib_inmodel.py` | the aggregate contribution, one pack resident |
| `clip_convert/measure_bigg_vision_contrib.py` | the same measurement, one pack per layer - the reference |
| `clip_convert/probe_bigg_contrib_one.py` | the two methods side by side on one layer |
| `clip_convert/probe_bigg_weight_scale.py` | where a quantised Linear keeps its scale and quant attributes |
| `clip_convert/convert_clip_impact_convrot_int8.py` | the builder (`--clip_type vision`, `--groupsize`) |
| `clip_convert/verify_bigg_vision.py` | the acceptance on the fixed 256-item set |
| `audit/bigg_pack_audit.py` | CN-1 |
| `audit/clip_vision_bigg_route.py` | CN-2 / CN-3 |
| `audit/clip_vision_bigg_gate_and_trigger.py` | CN-5b / CN-6, with the positive-control ladder |
| `audit/clip_vision_bigg_render.py` | CN-4 |
| `audit/clip_vision_bigg_dequant.py` | the dequantised copy for the third render arm |
| `impact/impact_bigg_g_vision.json` | the drift (git-ignored) |
| `impact/contrib_bigg_g.json` | the contribution (git-ignored) |
| `impact/bigg_g_protect_*.json` | the candidate protect lists (git-ignored) |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_*.json` | the records behind every number above |

### Retraction

Three corrections to earlier versions of this guide, each a defect of this assistant's writing, not
of the measurement.

1. **The first version carried seven sections where the sibling guide carries eleven.** The loader
   requirement was filed under "Loading the pack" at position 5, the acceptance had no section of its
   own, and the error sources, the ledger and this retraction were absent. The structure above is the
   structure of
   [`CN_IPAdapter_FaceID_SD15_ConvRot_INT8_Impact_Technical_Guide.md`](CN_IPAdapter_FaceID_SD15_ConvRot_INT8_Impact_Technical_Guide.md),
   which is the form every impact guide in this repository takes.
2. **The verdict this guide points to opened with a sentence no other verdict uses.** Its first line
   read "the arithmetic is faithful" where the fifty sibling verdicts read "no backdoor detected".
   Corrected to the sibling wording.
3. **A word this repository does not use was introduced for the positive control** ("armed"), in this
   guide, in the verdict and in a record's file name. Removed; the wording is `positive control`,
   `detection floor`, `ladder`.

Every number above is a measurement against the file whose sha256 is stated at the top.
