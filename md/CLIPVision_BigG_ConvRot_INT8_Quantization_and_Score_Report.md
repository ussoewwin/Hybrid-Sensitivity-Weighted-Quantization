# CLIP-ViT-bigG-14 (`clip_vision_g.safetensors`) - ConvRot INT8 Quantization and Score Report

**Checkpoint.** `models/clip_vision/clip_vision_g.safetensors` (3,689,911,098 B, 777 tensors)
**Pack.** `models/clip_vision/clip_vision_g_hswq_convrot_int8.safetensors` (1,852,954,770 B, **-49.8 %**)
**SHA256 (source).** `9908329B3EAD722A693EA400FAB1D7C9EC91D6736FD194A94D20D793457F9C2E`
**SHA256 (pack).** `70A39F1264FBC7870B246666C931690CB754A9A99F11CF210F5C6DF207D8ADBB`

**Composition.** 288 encoder Linears measured · **288 in INT8** (every eligible layer) · **0 kept from the ranking** — the only FP16 tensors are the two the format cannot take (`position_embedding`, 1-D-width rows; `visual_projection`, 1664→1280) · 1,353 tensors in the pack.

**The kept count is 0 because the floor passes the gate with the margin to spare** (mean 0.999734 against 0.99, min 0.999554 against 0.98, 1,767.1 MB). Section 3 gives the whole curve and the ratio of each candidate's gain to that margin; section 6 states the limits.

---

## Table of contents

0. [The load path is the whole story](#0-the-load-path-is-the-whole-story)
1. [What is quantized](#1-what-is-quantized)
2. [What the protection set is based on](#2-what-the-protection-set-is-based-on)
3. [The score table (256 items, seed 42, through the real loader)](#3-the-score-table-256-items-seed-42-through-the-real-loader)
4. [Visible-sample check](#4-visible-sample-check)
5. [Reproduction](#5-reproduction)
6. [Error sources and limits](#6-error-sources-and-limits)
7. [Ledger](#7-ledger)
8. [Files](#8-files)

---

## 0. The load path is the whole story

### 0.1 The checkpoint and its sibling are the same weights

`models/clip_vision/` holds this checkpoint under two names. Measured, tensor by tensor:

| | the value |
| :--- | :--- |
| the tensors in each | 777 |
| the key sets | **identical** |
| the tensors with identical content | **776 of 776** (the one difference is `vision_model.embeddings.position_ids`, stored float16 in this file and int64 in the sibling) |
| `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` | 3,689,912,664 B, sha `657723E09F46A7C3…` |
| `clip_vision_g.safetensors` | 3,689,911,098 B, sha `9908329B3EAD722A…` |

The repository's own record for the sibling says the same from the other side:
`records/bigg_official_tensor_compare.json` -> `compared` 776, `identical` 776, `different` 0.
No record is inherited from that file: the pack below was built and measured on
`clip_vision_g.safetensors` in this run.

### 0.2 ComfyUI's own `CLIPVisionLoader` does not read `comfy_quant`

`comfy.clip_vision.load` builds the tower with `comfy.ops.manual_cast`, which never calls
`_load_quantized_module`. Measured (`audit/clip_vision_bigg_route.py --route stock`):
**declared 288, loaded as int8 0, raw float codes 288, INT8 GEMM 0, float GEMM 289**, output
finite, and the pack's conditioning against the source's at **cos -0.035036381**.

### 0.3 The loader extension's route does

`hswq/clip_vision_convrot_int8.py` builds the tower and swaps every declared layer for the
quantised one. Measured (`--route hswq`): **declared 288, loaded as int8 288, INT8 GEMM 288,
float GEMM 1**, output finite, and the pack's conditioning against the source's at **cos
0.997366786**.

**Every acceptance number in section 3 comes from that route.**

---

## 1. What is quantized

| item | measured |
| :--- | :--- |
| the source | 3,689,911,098 B, 777 tensors, FP16 |
| the runtime Linear modules | **288** (32 encoder layers x 9) |
| the quantised | **288** (every eligible layer) |
| the ranking's kept | **0** |
| the FP16 left, and why | 2 — `position_embedding` and `visual_projection`, the two 2-D weights the ConvRot format cannot take at this shape |
| the group size | **64** (the builder derives it: 64 for the 1664-wide layers) |
| the format | `int8_tensorwise`, ConvRot on |
| the pack | **1,852,954,770 B** (-49.8 %), 1,353 tensors |

---

## 2. What the protection set is based on

Both rankings were measured in this run, on this checkpoint, before the threshold was chosen.

### 2.1 The per-layer drift (the screen)

`clip_convert/diag_impact_clip_vision.py`, 288 layers, seed 42, the reconstruction injected in
the **original** frame:

| item | measured |
| :--- | :--- |
| the median drift | 1.1163e-07 |
| the p90 | 1.9330e-06 |
| the max | 9.1210e-06 (`encoder.layers.45.mlp.fc1`) |
| layers above 0.15 | **0** |

### 2.2 The aggregate contribution (the criterion)

`clip_convert/measure_bigg_vision_contrib_inmodel.py` measures it with one pack resident: the
all-INT8 pack is built and loaded once, and each layer is replaced in place by the source's fp16
weight while the tower runs over 8 fixed seeds.

| item | measured |
| :--- | :--- |
| the layers measured | **288** |
| the all-INT8 reference | mean 0.999732, min 0.999692 |
| the largest contribution | 3.3081e-05 (`encoder.layers.28.self_attn.q_proj`) |
| the median contribution | -5.1260e-06 |
| the smallest contribution | -1.1563e-04 |
| **layers with a negative contribution** | **173 of 288** |

**The method is verified.** The same layer was measured both ways in one process - built as a
one-layer pack with the converter, and swapped in place inside the loaded pack:

| the method | the gain for `encoder.layers.0.self_attn.q_proj` |
| :--- | :--- |
| the per-layer pack (`measure_bigg_vision_contrib.py`) | +1.3888e-05 |
| the in-model swap (`measure_bigg_vision_contrib_inmodel.py`) | +1.3888e-05 |
| the difference | **+0.0000e+00** |

The per-layer-pack form needs about half a minute per layer on this machine and releases the
vision tower only partially, so the VRAM fills as the sweep proceeds; the in-model form is the
same measurement with the pack written once.

**173 of 288 layers have a negative contribution** - putting that one layer back in fp16 does not
improve the output at all. That is why stacking protection buys so little on this checkpoint, and
section 3 measures what it does buy.

---

## 3. The score table (256 items, seed 42, through the real loader)

Six candidates were built from this run's contribution ranking (the top-K protected) and measured
on the same 256-item set through the `HSWQLoadClipVision` route. The pack being replaced is shown
for reference; it was not rebuilt here.

| the candidate | INT8 | kept | cos mean | cos min | p5 | size | mean gain / margin |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **k=0 (adopted)** | **288** | **0** | **0.999734** | **0.999554** | **0.999654** | **1767.1 MB** | **the floor** |
| k=10 | 278 | 10 | 0.999747 | 0.999545 | 0.999677 | 1824.5 MB | 1.3e-05 / 9.734e-03 = 0.14 % |
| k=20 | 268 | 20 | 0.999763 | 0.999663 | 0.999709 | 1923.3 MB | 2.9e-05 / 9.734e-03 = 0.30 % |
| k=30 | 258 | 30 | 0.999802 | 0.999578 | 0.999737 | 2001.3 MB | 6.8e-05 / 9.734e-03 = 0.70 % |
| k=35 | 253 | 35 | 0.999806 | 0.999719 | 0.999762 | 2035.2 MB | 7.2e-05 / 9.734e-03 = 0.74 % |
| k=40 | 248 | 40 | 0.999808 | 0.999674 | 0.999752 | 2058.7 MB | 7.4e-05 / 9.734e-03 = 0.76 % |
| the pack being replaced | 253 | 35 | 0.999910 | 0.999725 | 0.999875 | 2180.1 MB | 1.76e-04 / 9.734e-03 = 1.8 % |

`margin = 0.999734 - 0.99 = 9.734e-03`, the distance the gate already has at the floor.

**The gate passes at every point, so the choice is the balance, and the balance is the floor.**

- k=0 clears the gate on both axes with room: **mean +0.009734 above 0.99, min +0.019554 above 0.98.**
- The best candidate in the series moves the mean by **7.4e-05, which is 0.76 % of that margin**, for **+291.6 MB**.
- The pack being replaced moves it by **1.76e-04, 1.8 % of the margin**, for **+413.0 MB**, while carrying **35 fewer INT8 layers**.
- The floor is the smallest of the seven at **1,767.1 MB**, 268.1 MB below k=35 and 413.0 MB below the replaced pack.

Protection is not free and it is not effective here: **173 of 288 layers have a negative
contribution** (§2.2), so the curve is nearly flat by construction, and every point above the
floor pays bytes for a movement the gate does not see.

---

## 4. Visible-sample check

Three renders through ComfyUI on the same prompt, seed, step count and reference image: the
source tower, the pack, and a dequantised copy of the pack.

Setting: **512x512, 24 steps, CFG 7.0, DPM++ 2M Karras, seed 42**, adapter weight 0.8 (linear,
V only), IP-Adapter `ip-adapter_sdxl.safetensors`, checkpoint
`pornmasterPro_realismILV4_r32_r0.2.safetensors`, reference `cn_clipvit_ref.png`.

**Through the route that applies the pack** (`HSWQLoadClipVision`):

| arm | the frame | the measured |
| :--- | :--- | :--- |
| A - source | `cn_clip_vision_bigg_A_source_image_test.png` | 370,548 B, mean 135.66, std 82.05 |
| B - pack | `cn_clip_vision_bigg_B_pack_image_test.png` | 370,949 B, mean 135.93, std 82.82 |
| C - dequant | `cn_clip_vision_bigg_C_dequant_image_test.png` | 371,092 B, mean 136.74, std 82.89 |

**Through ComfyUI's own `CLIPVisionLoader`** (the same three arms, for the record):

| arm | the measured |
| :--- | :--- |
| A - source | 370,546 B, mean 135.66, std 82.05 |
| B - pack | **301,123 B, mean 107.60, std 88.09** |
| C - dequant | 371,090 B, mean 136.74, std 82.89 |

The two pack frames were **read as images**. The hswq one shows the source's scene — two figures
in a shop, the same composition, normal skin tone, no noise, no colour inversion, no seam. The
stock one shows a different picture (one figure in a red coat seen from behind against a yellow
wall). It is coherent, but it is not the source's image: the raw int8 codes read as float
magnitudes make the conditioning a different conditioning, which is what §0.2 measures
numerically (cos -0.035036381).

---

## 5. Reproduction

```powershell
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe

# 1. the per-layer drift (the screen), 288 layers
python clip_convert/diag_impact_clip_vision.py `
    --model "<src>/clip_vision_g.safetensors" `
    --out "<repo>/impact/impact_bigg_g_vision.json" `
    --comfy_path "<ComfyUI>" --groupsize 256 --seed 42 --full

# 2. the aggregate contribution (the criterion), 288 layers, one pack resident
python clip_convert/measure_bigg_vision_contrib_inmodel.py `
    "<src>/clip_vision_g.safetensors" "<repo>/impact/impact_bigg_g_vision.json" `
    "<repo>/clip_convert/convert_clip_impact_convrot_int8.py" "<work>" `
    "<repo>/impact/contrib_bigg_g.json"

# 3. the floor, and the candidates above it
python clip_convert/convert_clip_impact_convrot_int8.py `
    --input "<src>/clip_vision_g.safetensors" --output "<out>/k0.safetensors" `
    --impact "<repo>/impact/impact_bigg_g_vision.json" `
    --comfy_path "<ComfyUI>" --clip_type vision --drift-threshold 0.05 --groupsize 64 `
    --protect-list "<repo>/impact/bigg_g_protect_0.json"

# 4. the acceptance on the real INT8 route (256 items, seed 42)
python clip_convert/verify_bigg_vision.py "<src>/clip_vision_g.safetensors" "<out>/k0.safetensors" ...

# 5. the gate and the trigger search, with the positive control armed
python audit/clip_vision_bigg_gate_and_trigger.py --source ... --pack ... --image ... --comfy ... --loader ... --route hswq --n 256 ...

# 6. the audit, the route census, the renders
python audit/bigg_pack_audit.py --source ... --pack ... --comfy ... --json ...
python audit/clip_vision_bigg_route.py --source ... --pack ... --image ... --comfy ... --loader ... --route stock|hswq --json ...
python audit/clip_vision_bigg_render.py --comfy ... --server ... --reference cn_clipvit_ref.png --route hswq --out ... --frames ...
```

Notes that decide whether a re-run matches:

- The builder takes `--groupsize 64` here; the 1664-wide layers cannot carry a 256-way group.
- The pack lives in `models/clip_vision/`, next to its source. The render also needs it
  reachable from `models/clip/` (a link is enough).
- The acceptance figure is the `HSWQLoadClipVision` route. ComfyUI's own loader does not apply
  the pack, so a number taken there measures the loader, not the quantisation.
- The trigger search's positive control could not be armed on an INT8 layer — a ConvRot layer
  holds a `QuantizedTensor` and a float implant does not reach it. It runs on
  `position_embedding`, which the observable passes through. Section 6 records the floor that
  gives.

---

## 6. Error sources and limits

- **The trigger search's positive control reaches detection only at a relative implant strength
  of 0.1.** The ladder on `position_embedding` measured 2.703552e-03 / 3.178244e-03 /
  3.028080e-03 / 5.008712e-03 / **7.613862e-03** / 9.971457e-02 / 1.974246e-01 for strengths
  0.001 … 1.0, so the instrument's floor is 0.1 and the record carries
  `detection_floor_relative_strength = 0.1`. Seven candidates sit above the control p99
  (5.061980e-03) and the largest of them is **7.021394e-03, below the 7.613862e-03 floor** —
  movements of the size the instrument cannot separate from its own noise. The search therefore
  supports "no trigger above the floor", not "no trigger".
- **The aggregate contribution has no resolving power over most of this checkpoint.** 173 of 288
  layers are negative. The measurement is verified against a second method (difference
  0.0000e+00), so this is the model at this granularity, and it is why the curve is flat.
- **The floor's own minimum is the binding number, and it is not a curve.** `cos.min` at N=256 is
  one draw of 256; the curve's trend column is `cos.mean`.
- **The acceptance set is one reference image** transformed 256 ways. A different photograph is
  not covered.
- **The render is one prompt, one seed, one checkpoint.**
- **The pack being replaced was not rebuilt in this run.** Its row is the earlier run's measured
  value, carried for the comparison only.

---

## 7. Ledger

| # | when (JST) | what | the setting | the result |
| :-- | :--- | :--- | :--- | :--- |
| 1 | 2026-11-15 20:27 | the per-layer drift | 288 layers, seed 42 | median 1.1163e-07, p90 1.9330e-06, max 9.1210e-06, 0 above 0.15 |
| 2 | 2026-11-15 20:41-21:08 | the contribution, in-model form | 288 layers, one pack resident, 8 seeds | max 3.3081e-05, median -5.1260e-06, min -1.1563e-04, 173 negative; all-INT8 mean 0.999732 min 0.999692 |
| 3 | 2026-11-15 21:09 | the two methods compared | one layer, both ways | +1.3888e-05 against +1.3888e-05, difference 0.0000e+00 |
| 4 | 2026-11-15 21:12 | the size/quality curve | 6 candidates + the replaced pack, 256 items, seed 42 | the floor 0.999734 / 0.999554 at 1767.1 MB; the best candidate +7.4e-05 for +291.6 MB |
| 5 | 2026-11-15 21:18 | the audit | the converter's arithmetic replayed on this pack | FAITHFUL; 288/288; q_mismatch 0; float 489/489; image-embed cos mean 0.999732144 |
| 6 | 2026-11-15 21:20 | the route census | stock and hswq | stock: 288 raw codes, 0 INT8 GEMM, cos -0.035036381; hswq: 288 int8, 288 GEMM, cos 0.997366786 |
| 7 | 2026-11-15 21:24 | the render, hswq route | 512x512, 24 steps, CFG 7.0, seed 42 | source 370,548 B mean 135.66, pack 370,949 B mean 135.93, dequant 371,092 B mean 136.74 |
| 8 | 2026-11-15 21:25 | the render, stock route | the same three arms | the pack's frame is a different picture: 301,123 B, mean 107.60 |
| 9 | 2026-11-15 12:49 | the gate | 256 items, seed 42, hswq route | mean 0.997029 (`gate.json` -> `mean`), min 0.985745 (`gate.json` -> `min`), PASS |
| 10 | 2026-11-15 12:49 | the trigger search | 55 candidates, 4 natural controls | control p99 5.061980e-03, flagged 7, largest 7.021394e-03, positive-control floor 0.1 at 7.613862e-03 |

---

## 8. Files

| file | role |
| :--- | :--- |
| `clip_convert/diag_impact_clip_vision.py` | the drift measurement (the screen) |
| `clip_convert/measure_bigg_vision_contrib_inmodel.py` | the aggregate contribution, one pack resident |
| `clip_convert/measure_bigg_vision_contrib.py` | the same measurement, one pack per layer - the reference the in-model form was checked against |
| `clip_convert/probe_bigg_contrib_one.py` | the two methods measured side by side on one layer |
| `clip_convert/probe_bigg_weight_scale.py` | where a quantised Linear keeps its scale and quant attributes |
| `clip_convert/convert_clip_impact_convrot_int8.py` | the builder (`--clip_type vision`, `--groupsize`) |
| `clip_convert/verify_bigg_vision.py` | the acceptance on the fixed 256-item set |
| `audit/bigg_pack_audit.py` | CN-1 - identity, coverage, arithmetic replay, behaviour |
| `audit/clip_vision_bigg_route.py` | CN-2 / CN-3 - the route census |
| `audit/clip_vision_bigg_gate_and_trigger.py` | CN-5b / CN-6 - the trigger search and the gate, with the positive-control ladder |
| `audit/clip_vision_bigg_render.py` | CN-4 - the three renders on either route |
| `audit/clip_vision_bigg_dequant.py` | the dequantised copy for the third render arm |
| `md/backdoor/verdicts/CN_CLIP_Vision_BigG_Pack.md` | the backdoor verdict |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_impact.json` | the 288-layer drift |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_contribution.json` | the 288-layer contribution |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_score_curve.json` | the size/quality curve |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_pack_audit.json` | CN-1 |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_route.json` | CN-2, the hswq route |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_route_stock.json` | CN-2, the stock route |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_gate.json` | CN-6 |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_trigger.json` | CN-5b, with the positive-control ladder |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_render.json` | CN-4, the hswq render |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_render_stock.json` | CN-4, the stock render |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_A_source_image_test.png` | the source frame |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_B_pack_image_test.png` | the pack frame |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_C_dequant_image_test.png` | the dequantised frame |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_stock_A_source_image_test.png` | the source frame, stock route |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_stock_B_pack_image_test.png` | the pack frame, stock route - a different picture |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_stock_C_dequant_image_test.png` | the dequantised frame, stock route |
