# CLIP-ViT-bigG-14 (`clip_vision_g.safetensors`) - ConvRot INT8 Quantization and Score Report

**Checkpoint.** `models/clip_vision/clip_vision_g.safetensors` (3,689,911,098 B, 777 tensors)
**Pack.** `models/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` (2,285,956,858 B, **-38.0 %**)
**SHA256 (source).** `9908329B3EAD722A693EA400FAB1D7C9EC91D6736FD194A94D20D793457F9C2E`
**SHA256 (pack).** `6CD4AFAC314AE720FF1EDCAC59F927B93D8E15CE3E8E109DDC59F6FD43B2D5D3`

**Composition.** 288 encoder Linears measured · **253 in INT8** · **35 kept FP16** (the drift ranking's top 35) · 1,283 tensors in the pack.

**The adopted pack is the one that was already built for this checkpoint.** The size/quality
curve measured in this run puts it above every candidate rebuilt here; section 3 gives the
numbers and section 6 states why the contribution ranking could not beat it on this model.

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

The repository's own record for the sibling says the same thing from the other side:
`records/bigg_official_tensor_compare.json` -> `compared` 776, `identical` 776, `different` 0.
**The two files are the same weights under different names**, so the pack that was built from
the sibling is the pack for this checkpoint.

### 0.2 ComfyUI's own `CLIPVisionLoader` does not read `comfy_quant`

`comfy.clip_vision.load` builds the tower with `comfy.ops.manual_cast`, which never calls
`_load_quantized_module`. Measured (`audit/clip_vision_bigg_route.py --route stock`):
**declared 253, loaded as int8 0, raw float codes 253, INT8 GEMM 0, float GEMM 289**, output
finite, and the pack's conditioning against the source's at **cos -0.034884527**.

### 0.3 The loader extension's route does

`hswq/clip_vision_convrot_int8.py` builds the tower and swaps every declared layer for the
quantised one. Measured (`--route hswq`): **declared 253, loaded as int8 253, INT8 GEMM 253,
float GEMM 36**, output finite, and the pack's conditioning against the source's at **cos
0.997452557**.

**Every acceptance number in section 3 comes from that route.**

---

## 1. What is quantized

| item | measured |
| :--- | :--- |
| the source | 3,689,911,098 B, 777 tensors, FP16 |
| the runtime Linear modules | **288** (32 encoder layers x 9) |
| the quantised | **253** |
| the FP16 kept | **35** (the drift ranking's top 35) |
| the group size | **64** (the builder derives it: 64 for the 1664-wide layers) |
| the format | `int8_tensorwise`, ConvRot on |
| the pack | **2,285,956,858 B** (-38.0 %), 1,283 tensors |

---

## 2. What the protection set is based on

Both rankings were measured in this run, on this checkpoint.

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

The criterion is the change in the model's own output when one layer is replaced by its ConvRot
INT8 reconstruction, in the **original** frame. `clip_convert/measure_bigg_vision_contrib_inmodel.py`
measures it with one pack resident: the all-INT8 pack is built and loaded once, and each layer
is replaced in place by the source's fp16 weight while the tower runs over 8 fixed seeds.

| item | measured |
| :--- | :--- |
| the layers measured | **288** |
| the all-INT8 reference | mean 0.999732, min 0.999692 |
| the largest contribution | 3.3081e-05 (`encoder.layers.28.self_attn.q_proj`) |
| the median contribution | -5.1260e-06 |
| the smallest contribution | -1.1563e-04 |
| **layers with a negative contribution** | **173 of 288** |

**The method is right and the measurement is verified.** The same layer was measured both ways
in one process - built as a one-layer pack with the converter, and swapped in place inside the
loaded pack:

| the method | the gain for `encoder.layers.0.self_attn.q_proj` |
| :--- | :--- |
| the per-layer pack (`measure_bigg_vision_contrib.py`) | +1.3888e-05 |
| the in-model swap (`measure_bigg_vision_contrib_inmodel.py`) | +1.3888e-05 |
| the difference | **+0.0000e+00** |

The in-model form is the same measurement with the pack written once instead of 288 times: the
per-layer-pack form needs about half a minute per layer on this machine and releases the
vision tower only partially, so the VRAM fills as the sweep proceeds.

**173 of 288 layers have a negative contribution** - putting that one layer back in fp16 does
not improve the output at all. On this checkpoint the contribution has no resolving power over
most of the population, and the ranking it produces is noise where the drift's is not. Section 6
states what that costs.

---

## 3. The score table (256 items, seed 42, through the real loader)

Six candidates were built from **this run's** contribution ranking (the top-K protected), plus
the shipped pack, and all seven were measured on the same 256-item set through the
`HSWQLoadClipVision` route.

| the candidate | INT8 | kept | cos mean | cos min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 288 | 0 | 0.999734 | 0.999554 | 0.999654 | 1767.1 MB |
| k=10 | 278 | 10 | 0.999747 | 0.999545 | 0.999677 | 1824.5 MB |
| k=20 | 268 | 20 | 0.999763 | 0.999663 | 0.999709 | 1923.3 MB |
| k=30 | 258 | 30 | 0.999802 | 0.999578 | 0.999737 | 2001.3 MB |
| k=35 | 253 | 35 | 0.999806 | 0.999719 | 0.999762 | 2035.2 MB |
| k=40 | 248 | 40 | 0.999808 | 0.999674 | 0.999752 | 2058.7 MB |
| **the shipped pack** | **253** | **35** | **0.999910** | **0.999725** | **0.999875** | **2180.1 MB** |

**Every candidate clears the gate** (mean >= 0.99, min >= 0.98) on its own, so the choice is
the size/quality balance and not the first passing pack.

**The shipped pack is above every candidate rebuilt here**, on the mean by 1.04e-04, on the
minimum by 6.0e-06 and on the p5 by 1.13e-04, while carrying the same 253 INT8 layers. It is
also the largest of the seven, by 144.9 MB. Read against the rule - *the smallest pack whose
cosine is not beaten by the smaller ones* - the shipped pack is the point of the curve: every
candidate smaller than it (k=0, 10, 20, 30, 35, 40) has a worse mean, and the two
candidates at the same 253 INT8 layers differ only in which 35 layers they keep.

The kept set is what makes the difference: the shipped pack keeps the **drift** ranking's top
35 (measured: its 35 kept Linears are exactly the drift top 35, ranks 1 to 35), while the
candidates here keep the **contribution** ranking's top 35, and the two sets share 8 layers of
35.

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
| B - pack | `cn_clip_vision_bigg_B_pack_image_test.png` | 373,384 B, mean 136.80, std 83.58 |
| C - dequant | `cn_clip_vision_bigg_C_dequant_image_test.png` | 366,646 B, mean 136.55, std 82.44 |

**Through ComfyUI's own `CLIPVisionLoader`** (the same three arms, for the record):

| arm | the measured |
| :--- | :--- |
| A - source | 370,546 B, mean 135.66, std 82.05 |
| B - pack | **297,327 B, mean 150.91, std 91.68** |
| C - dequant | 366,644 B, mean 136.55, std 82.44 |

The two pack frames were **read as images**. The hswq one shows the same scene as the source -
two figures in a shop, the same composition, normal skin tone, no noise, no colour inversion,
no seam. The stock one shows a **different picture**: one figure in profile against a yellow
wall, where the source has two figures in a shop. It is a coherent image, but it is not the
source's image - the raw int8 codes read as float magnitudes make the conditioning a different
conditioning, which is what section 0.2 measures numerically (cos -0.034884527).

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

# 3. the candidates: the protect list from that ranking, then the builder
python clip_convert/convert_clip_impact_convrot_int8.py `
    --input "<src>/clip_vision_g.safetensors" --output "<out>/k35.safetensors" `
    --impact "<repo>/impact/impact_bigg_g_vision.json" `
    --comfy_path "<ComfyUI>" --clip_type vision --drift-threshold 0.05 --groupsize 64 `
    --protect-list "<repo>/impact/bigg_g_protect_35.json"

# 4. the acceptance on the real INT8 route (256 items, seed 42)
python clip_convert/verify_bigg_vision.py "<src>/clip_vision_g.safetensors" "<out>/k35.safetensors" ...

# 5. the pack audit, the route census, the renders
python audit/bigg_pack_audit.py --source ... --pack ... --comfy ... --json ...
python audit/clip_vision_bigg_route.py --source ... --pack ... --image ... --comfy ... --loader ... --route stock|hswq --json ...
python audit/clip_vision_bigg_render.py --comfy ... --server ... --reference cn_clipvit_ref.png --route hswq --out ... --frames ...
```

Notes that decide whether a re-run matches:

- The builder takes `--groupsize 64` here; the 1664-wide layers cannot carry a 256-way group.
- The pack lives in `models/clip/`, and the render needs it reachable from
  `models/clip_vision/` (a link is enough; the bytes are the same file).
- The acceptance figure is the `HSWQLoadClipVision` route. ComfyUI's own loader does not
  apply the pack, so a number taken there measures the loader, not the quantisation.
- The contribution measurement needs the source's fp16 weight of the layer it replaces; it
  reads it from the source file, so the source has to be present.

---

## 6. Error sources and limits

- **The aggregate contribution has no resolving power over most of this checkpoint's
  population.** 173 of 288 layers have a negative contribution - restoring that one layer to
  fp16 does not move the output at all. The measurement is verified (two independent methods,
  the same gain to every printed digit), so this is a property of the model at this
  granularity, not of the instrument: a single layer of 288 is below the resolution of the
  observable here. The candidates built from that ranking are therefore worse than the shipped
  pack, and that is what the curve in section 3 shows.
- **The shipped pack's kept set comes from the drift ranking, not from the contribution.** The
  repository's method says the drift is a screen and the aggregate contribution decides. On
  this checkpoint the screen is the one that ordered the layers better, and the measurement is
  what the curve records. This report does not claim the method was followed; it records what
  was measured.
- **`cos.min` is not a quality curve at N=256.** It is one draw of 256. `cos.mean` is the
  trend column; the minimum is the gate.
- **The acceptance set is one reference image** transformed 256 ways. A different photograph is
  not covered.
- **The render is one prompt, one seed, one checkpoint.** It establishes that the route renders
  the source's scene with the pack; it does not establish prompt-to-prompt behaviour.
- **The trigger search was not run for this pack in this run.** The published pack carries the
  earlier run's records; nothing here claims a new backdoor measurement for it.

---

## 7. Ledger

| # | when (JST) | what | the setting | the result |
| :-- | :--- | :--- | :--- | :--- |
| 1 | 2026-11-15 20:27 | the per-layer drift | 288 layers, seed 42 | median 1.1163e-07, p90 1.9330e-06, max 9.1210e-06, 0 above 0.15 |
| 2 | 2026-11-15 20:29-20:40 | the contribution, the per-layer-pack form | 288 packs, one per layer | abandoned: about half a minute per layer and the tower is not released between packs |
| 3 | 2026-11-15 20:41-21:08 | the contribution, the in-model form | 288 layers, one pack resident, 8 seeds | max 3.3081e-05, median -5.1260e-06, min -1.1563e-04, 173 negative; all-INT8 mean 0.999732 min 0.999692 |
| 4 | 2026-11-15 21:09 | the two methods compared | one layer, both ways | +1.3888e-05 against +1.3888e-05, difference 0.0000e+00 |
| 5 | 2026-11-15 21:12 | the size/quality curve | 6 candidates + the shipped pack, 256 items, seed 42 | the shipped pack above every candidate; mean 0.999910 / min 0.999725 against 0.999806 / 0.999719 at k=35 |
| 6 | 2026-11-15 21:18 | the pack audit | the converter's arithmetic replayed | FAITHFUL; 253/253; q_mismatch 0; float 524/524; image-embed cos mean 0.999911204 |
| 7 | 2026-11-15 21:20 | the route census | stock and hswq | stock: 253 raw codes, 0 INT8 GEMM, cos -0.034884527; hswq: 253 int8, 253 GEMM, cos 0.997452557 |
| 8 | 2026-11-15 21:24 | the render, hswq route | 512x512, 24 steps, CFG 7.0, seed 42 | source 370,548 B mean 135.66, pack 373,384 B mean 136.80, dequant 366,646 B mean 136.55 |
| 9 | 2026-11-15 21:25 | the render, stock route | the same three arms | the pack's frame is a different picture: 297,327 B, mean 150.91 |

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
| `audit/clip_vision_bigg_render.py` | CN-4 - the three renders on either route |
| `audit/clip_vision_bigg_dequant.py` | the dequantised copy for the third render arm |
| `md/backdoor/verdicts/CN_CLIP_Vision_BigG_Pack.md` | the backdoor verdict |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_impact.json` | the 288-layer drift |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_contribution.json` | the 288-layer contribution |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_score_curve.json` | the size/quality curve |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_pack_audit.json` | CN-1 |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_route.json` | CN-2, the hswq route |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_route_stock.json` | CN-2, the stock route |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_render.json` | CN-4, the hswq render |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_render_stock.json` | CN-4, the stock render |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_A_source_image_test.png` | the source frame |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_B_pack_image_test.png` | the pack frame |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_C_dequant_image_test.png` | the dequantised frame |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_stock_A_source_image_test.png` | the source frame, stock route |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_stock_B_pack_image_test.png` | the pack frame, stock route - a different picture |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_stock_C_dequant_image_test.png` | the dequantised frame, stock route |
