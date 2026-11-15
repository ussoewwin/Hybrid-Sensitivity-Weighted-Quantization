# CLIP-ViT-H-14 (laion2B-s32B-b79K) - ConvRot INT8 Quantization and Score Report

**Checkpoint.** `models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` (2,528,373,448 B, 521 tensors)
**Pack.** `models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K_hswq_convrot_int8.safetensors` (1,064,907,988 B, **-57.9 %**)
**SHA256 (source).** `6CA9667DA1CA9E0B0F75E46BB030F7E011F44F86CBFB8D5A36590FCD7507B030`
**SHA256 (pack).** `477E5271A94349109D31A482EE870581F55F39059036CDFCC13DC6314CCE7323`

**Composition.** 192 encoder Linears measured · **92 in INT8** · **100 kept FP16** (the contribution's top 100) · 705 tensors in the pack.

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

This checkpoint has one route that installs the pack's INT8 payload and one that does not, and
the difference decides where every acceptance number comes from.

### 0.1 ComfyUI's own `CLIPVisionLoader` does not read `comfy_quant`

`comfy.clip_vision.load` builds the tower with `comfy.ops.manual_cast`, which never calls
`_load_quantized_module`. A pack loaded that way keeps its int8 codes in a float Parameter.

Measured on this pack (`audit/clip_vit_h14_route.py --route stock`): **declared 92, loaded as
int8 0, raw float codes 92, INT8 GEMM 0, float GEMM 193**, output finite, and the pack's
conditioning against the source's at **cos -0.018628847** - the conditioning is unusable.

### 0.2 The loader extension's `HSWQLoadClipVision` does

`nodes/hswq_load_clip_vision.py` calls `hswq/clip_vision_convrot_int8.py`, which builds the
tower and then swaps every layer the pack declares for `mixed_precision_ops`' own Linear and
loads its `{weight, weight_scale, bias, comfy_quant}` into it.

Measured (`--route hswq`): **declared 92, loaded as int8 92, INT8 GEMM 92, float GEMM 101**,
output finite, and the pack's conditioning against the source's at **cos 0.998942256**.

### 0.3 A third route exists on this machine and does not apply the pack

A1111's ControlNet extension has no CLIP Vision model input at all - it consumes pre-computed
embeddings. There is no ComfyUI-IP-Adapter-style path that would read the stamps either.

**Every acceptance number below is measured on the `HSWQLoadClipVision` route.**

### 0.4 The scripts change the working directory

`measure_h14_vision_contrib.py` does `os.chdir(COMFY)` before it runs. A relative `--out` then
resolves against the ComfyUI tree, and the run fails with `FileNotFoundError` **after** the
measurement has been paid for. This cost one full 192-layer sweep on the first attempt.

---

## 1. What is quantized

| item | measured |
| :--- | :--- |
| the source | 2,528,373,448 B, 521 tensors, FP32 |
| the runtime Linear modules | **192** (32 encoder layers x 6) |
| the ConvRot-eligible Linears | **192** |
| the quantised | **92** |
| the FP16 kept | **100** (the contribution's top 100) |
| the group size | 256 |
| the format | `int8_tensorwise`, ConvRot on |
| the pack | **1,064,907,988 B** (-57.9 %), 705 tensors |

Inner dimensions: `mlp.fc1` 1280->5120, `mlp.fc2` 5120->1280, the four projections 1280->1280.
All are multiples of 256, so **no layer was kept for a shape reason** - the kept 100 are the
contribution's top 100 and nothing else.

The kept tensors are stored **FP16**. The source is FP32, but the runtime runs in FP16, so an
FP32 kept tensor is cast at load time and only pays for the bytes. The INT8 `weight_scale` stays
FP32 because it carries the dequantisation precision.

---

## 2. What the protection set is based on

The criterion is the **aggregate contribution**: one layer is replaced by its ConvRot INT8
reconstruction and the change in the stage output that layer feeds is measured, as `1 - cos`.
`clip_convert/measure_h14_vision_contrib.py` measures exactly that - it puts the reconstruction
of `proj` back through the tower and takes the embedding's cosine against the FP16 baseline,
and it does the same for each of the 192 Linears on that Linear's own output.

The baseline is built with the **same** ops class the pack is loaded with
(`mixed_precision_ops`); ComfyUI's `manual_cast` is not numerically equivalent for this model,
so a baseline on the wrong ops path measures the ops path.

Measured on this checkpoint (192 layers, 8 seeds, seed 42, float32 weights):

| rank | layer | the contribution | the share |
| :--- | :--- | :--- | :--- |
| 1 | `encoder.layers.1.mlp.fc2` | 1.9073486328125e-06 | 1.8667 % |
| 2 | `encoder.layers.24.mlp.fc1` | 1.899898e-06 | 1.8594 % |
| 3 | `encoder.layers.1.mlp.fc1` | 1.817942e-06 | 1.7792 % |
| 4 | `encoder.layers.28.mlp.fc1` | 1.817942e-06 | 1.7792 % |
| 5 | `encoder.layers.27.mlp.fc1` | 1.743436e-06 | 1.7063 % |
| 6 | `encoder.layers.21.mlp.fc2` | 1.549721e-06 | 1.5157 % |

| item | measured |
| :--- | :--- |
| the largest contribution | 1.9073486328125e-06 (`encoder.layers.1.mlp.fc2`) |
| the median | 4.209578037261963e-07 |
| the p90 | 1.3113021850585938e-06 |
| the smallest | -3.725290298461914e-08 (`encoder.layers.4.mlp.fc2`) |
| the sum over the 192 layers | 0.00010217726230621338 |
| layers with a negative contribution | **8** |
| layers above 0.15 | **0** |

**The curve is flat, and the share column is the number that says so.** The largest layer
carries 1.87 % of the summed contribution and the fifth carries 1.71 %; a network with a real
hot layer would show one group at tens of percent. Eight layers have a **negative**
contribution - protecting them would not help, and the ranking does not.

**The protection set is empty at the 0.15 threshold** because the measurement returned nothing
above it: 0 of 192 layers are above 0.15 and 0 are above 1e-4. The 100 kept layers are the
contribution's top 100, chosen on the size/quality balance below.

**The tie at the boundary.** Two layers share 1.817942e-06 and several more share 1.3113021850585938e-06,
so the membership of the top-100 set depends on the sort order among equal values. A pack
built from a protect list that breaks those ties differently differs from this one in **10
layers** (measured: the INT8/FP16 split differs on `layers.4.mlp.fc2`, `layers.5.mlp.fc1`,
`layers.7.self_attn.v_proj`, `layers.8.mlp.fc2`, `layers.10.self_attn.out_proj`,
`layers.17.self_attn.out_proj`, `layers.25.self_attn.v_proj`, `layers.29.self_attn.k_proj`,
`layers.29.self_attn.q_proj`, `layers.29.self_attn.out_proj`). The protect list is written out
in full, so the set is reproducible from the record.

---

## 3. The score table (256 items, seed 42, through the real loader)

Six candidates were built from the ranking (the top-K protected) and measured on the fixed
256-item acceptance set through the `HSWQLoadClipVision` route.

| the candidate | INT8 | kept FP16 | cos mean | cos min | p5 | size | INT8 GEMM calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 192 | 0 | 0.998089762 | 0.988531062 | 0.994754711 | 636,631,336 B | 49,152 |
| k=15 | 177 | 15 | 0.998473382 | 0.993083664 | 0.996227620 | 729,830,411 B | 45,312 |
| k=30 | 162 | 30 | 0.998538490 | 0.993156433 | 0.996141628 | 823,013,318 B | 41,472 |
| k=60 | 132 | 60 | 0.998692746 | 0.993619517 | 0.996385444 | 965,264,724 B | 33,792 |
| **k=100 (adopted)** | **92** | **100** | **0.999044517** | **0.995568918** | **0.997158609** | **1,064,907,988 B** | **23,552** |
| k=140 | 52 | 140 | 0.999702289 | 0.995363910 | 0.998700873 | 1,179,312,148 B | 13,312 |

**Every candidate clears the gate** (mean >= 0.99, min >= 0.98) on its own, so the choice is
the size/quality balance and not the first passing pack.

`cos.mean` is monotone across the whole curve. `cos.min` is **not**: it peaks at k=100
(0.995568918) and then falls at k=140 (0.995363910). At N=256 the minimum is one draw of 256,
so it moves with which item happens to be worst; the mean is the trend column and the minimum
is the gate.

k=140 buys **+6.58e-04** of the mean and **-2.05e-04** of the minimum for **+114,404,160 B**.
That is a column already far past the gate, bought with bytes and paid for in the worst case,
so **k=100 is the adopted point**: it keeps the 100 highest-contribution layers in FP16 while
the pack stays at 57.9 % below the source.

**The pack audit** (`audit/clip_vit_h14_pack_audit.py`, replaying the converter's arithmetic):

| check | measured |
| :--- | :--- |
| INT8 layers found / declared | 92 / 92 |
| integers differing from the replay | **0** |
| the largest integer delta | 1 |
| scale exact relative max | 3.282655e-04 |
| non-finite scales | 0 |
| source weights missing from the pack | **0** |
| weights added by the pack | **0** |
| shape mismatches | **0** |
| float tensors checked | 329 |
| float mismatches | **0**, max abs diff 0.0 |
| behaviour, token stack cos | 0.9999468728489679 (rel L2 0.010308990015502785) |
| verdict | **FAITHFUL** |

---

## 4. Visible-sample check

Three renders through ComfyUI on the same prompt, seed, step count and reference image, the
source tower against the pack and against a dequantised copy of the pack.

Setting: **512x512, 24 steps, CFG 7.0, DPM++ 2M Karras, seed 42**, adapter weight 0.8 (linear,
V only), IP-Adapter `ip-adapter-plus_sdxl_vit-h.safetensors`, checkpoint
`pornmasterPro_realismILV4_r32_r0.2.safetensors`, reference `cn_clipvit_ref.png`.

**Through the route that applies the pack** (`HSWQLoadClipVision`):

| arm | the frame | the measured |
| :--- | :--- | :--- |
| A - source | `cn_clip_vit_h14_laion2b_A_source_image_test.png` | 388,714 B, mean 145.70, std 81.26 |
| B - pack | `cn_clip_vit_h14_laion2b_B_pack_image_test.png` | 382,349 B, mean 143.61, std 80.99 |
| C - dequant | `cn_clip_vit_h14_laion2b_C_dequant_image_test.png` | 380,184 B, mean 143.58, std 81.20 |

**Through ComfyUI's own `CLIPVisionLoader`** (the same three arms, for the record):

| arm | the measured |
| :--- | :--- |
| A - source | 388,712 B, mean 145.70, std 81.26 |
| B - pack | **2,447 B, mean 0.00, std 0.00 - exactly black** |
| C - dequant | 380,182 B, mean 143.58, std 81.20 |

All three frames of the first table were **looked at as images**: each shows the same figure in
the same room, with coherent facial features and hands, normal skin tone, no noise, no colour
inversion and no seam. The source and the pack differ in the background furniture and in the
hair strands - a per-pixel shift, not a structural change. The black frame in the second table
is the stock loader reading the int8 codes as float magnitudes, which is what section 0.1
measures.

---

## 5. Reproduction

```powershell
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe

# 1. the criterion: the aggregate contribution (192 layers, 8 seeds, seed 42)
#    --out must be ABSOLUTE: the script does os.chdir(COMFY)
python clip_convert/measure_h14_vision_contrib.py `
    --src "<src>/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" `
    --out "<repo>/impact/contrib_vit_h14_laion2b.json" --n 8 --seed 42 --groupsize 256

# 2. the protect list (top-K of that ranking), then the builder
python clip_convert/convert_clip_impact_convrot_int8.py `
    --input "<src>/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" `
    --output "<out>/CLIP-ViT-H-14-laion2B-s32B-b79K_hswq_convrot_int8.safetensors" `
    --impact "<repo>/impact/contrib_vit_h14_laion2b.json" `
    --comfy_path "<ComfyUI>" --clip_type vision --drift-threshold 1.0 --groupsize 256 `
    --protect-list "<repo>/impact/h14_laion2b_protect_100.json"

# 3. the acceptance on the real INT8 route (256 items, seed 42)
python audit/clip_vit_h14_gate_and_trigger.py `
    --source "<src>/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" `
    --pack   "<out>/CLIP-ViT-H-14-laion2B-s32B-b79K_hswq_convrot_int8.safetensors" `
    --image  "<ComfyUI>/input/cn_clipvit_ref.png" --comfy "<ComfyUI>" `
    --loader "<ComfyUI>/custom_nodes/ComfyUI-HSWQ-Loader-and-Tools" `
    --route hswq --n 256 `
    --json-gate    md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_gate_256.json `
    --json-trigger md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_image_trigger.json

# 4. the audit, the route census, the positive controls, the render
python audit/clip_vit_h14_pack_audit.py --source ... --pack ... --image ... --comfy ... --json ...
python audit/clip_vit_h14_route.py --source ... --pack ... --image ... --comfy ... --loader ... --route stock --json ...
python audit/clip_vit_h14_route.py --source ... --pack ... --image ... --comfy ... --loader ... --route hswq --json ...
python audit/clip_vit_h14_int8_control.py --source ... --pack ... --image ... --comfy ... --loader ... --threshold 0.004198130934271849 --seeds 5 --json ...
python audit/clip_vit_h14_int8_sweep.py  --pack ... --image ... --comfy ... --loader ... --threshold 0.004198130934271849 --seeds 5 --json ...
python audit/clip_vit_h14_render.py --comfy ... --server http://127.0.0.1:8188 --reference cn_clipvit_ref.png --route hswq --out ... --frames ...
```

Notes that decide whether a re-run matches:

- The pack is written **next to its source**, in `models/clip_vision/`.
- The render needs the SDXL checkpoint and an SDXL IP-Adapter (`ip-adapter-plus_sdxl_vit-h`);
  the SD1.5 ones are 768-wide and this tower is 1280-wide.
- The gate figure is the `HSWQLoadClipVision` route. The stock `CLIPVisionLoader` does not
  apply the pack at all, so a number taken there measures the loader, not the quantisation.
- The dequantised copy in `models/clip_vision/` exists only for the third render arm.

---

## 6. Error sources and limits

- **61 of the 92 INT8 layers are blind to a rank-1 implant at 100 % of the weight norm.** They
  are almost entirely `q_proj` (30 of 32) and `k_proj` (28 of 32). A weight-shaped backdoor in
  those layers would not have been visible to this search at any strength it can inject. The
  verdict covers the 31 layers that resolved, not the 61 that did not.
- **28 of the 92 layers do not resolve a 64 LSB code edit on 10 % of rows** either - the blind
  end for codes.
- **Below the noise floor nothing is measurable.** A ConvRot INT8 layer dequantised and
  requantised with nothing added already moves its output by 1.365459e-03 at the worst layer.
  An implant smaller than that cannot be separated from requantisation noise.
- **The trigger search flagged 1 of 55 candidates** (`patch_qrblock_bl_64`, 1-cos
  4.828287e-03, 1.15x the control threshold). It is an image-level content change - a 64 px
  block in the lower-left corner - and it sits below every floor measured inside the INT8
  layers. It is recorded as flagged, not explained away.
- **The sweep is one seed set per layer.** All 92 layers were swept, with 3 implant directions
  per rung and 5 seeds; a floor at the edge of resolution could move with more draws.
- **The measurement is one fixed input** (a synthetic 224x224 tensor, seed 42) for the
  contribution, and one real reference image for the acceptance set. A different image moves
  the layer ordering inside the top decile.
- **The `cos.min` column is not a quality curve at N=256.** It is one draw of 256; the mean is
  the trend column.
- **The catalogue is not touched by this report.** The pack is built, measured and recorded
  here; adding a row to the shared table is a separate act.

---

## 7. Ledger

| # | when (JST) | what | the setting | the result |
| :-- | :--- | :--- | :--- | :--- |
| 1 | 2026-11-15 19:30:38 | the aggregate contribution | 192 layers, 8 seeds, seed 42 | max 1.9073486328125e-06, median 4.209578037261963e-07, p90 1.3113021850585938e-06, min -3.725290298461914e-08, sum 0.00010217726230621338, 8 negative |
| 2 | 2026-11-15 19:31:13 | the protect lists | top-K of that ranking, K = 0/15/30/60/100/140 | the top-100 set written out in full |
| 3 | 2026-11-15 19:31-19:33 | the threshold sweep | 6 packs, groupsize 256 | 192/177/162/132/92/52 INT8, 636.6 MB .. 1,179.3 MB |
| 4 | 2026-11-15 19:35 | k=0 | 192 INT8 | 0.998089762 / 0.988531062, 49,152 GEMM calls, PASS |
| 5 | 2026-11-15 19:36 | k=15 | 177 INT8 | 0.998473382 / 0.993083664, 45,312 GEMM calls, PASS |
| 6 | 2026-11-15 19:37 | k=30 | 162 INT8 | 0.998538490 / 0.993156433, 41,472 GEMM calls, PASS |
| 7 | 2026-11-15 19:38 | k=60 | 132 INT8 | 0.998692746 / 0.993619517, 33,792 GEMM calls, PASS |
| 8 | 2026-11-15 19:39 | **k=100 (adopted)** | 92 INT8 | **0.999044517 / 0.995568918**, 23,552 GEMM calls, PASS |
| 9 | 2026-11-15 19:40 | k=140 | 52 INT8 | 0.999702289 / 0.995363910, 13,312 GEMM calls, PASS |
| 10 | 2026-11-15 19:54 | the positive control | one layer, 5 seeds, 3 arms | floors: float 0.3, pre_quantise 0.3, codes 32 LSB (512 rows) / 64 LSB (64 rows); noise floor 9.191147e-04 |
| 11 | 2026-11-15 19:57 | the positive control sweep | all 92 INT8 layers, 5 seeds | worst noise 1.365459e-03, worst float floor 1.0, worst codes floor 64; 6 layers at 0.3, 25 at 1.0, 61 unresolved |
| 12 | 2026-11-15 19:58 | the pack audit | the converter's own arithmetic replayed | FAITHFUL; 92/92; q_mismatch 0; float 329/329; tokens cos 0.999946873 |
| 13 | 2026-11-15 19:58 | the route census | stock and hswq | stock: 92 raw codes, 0 INT8 GEMM, cos -0.018628847; hswq: 92 int8, 92 GEMM, cos 0.998942256 |
| 14 | 2026-11-15 20:03 | the render | 512x512, 24 steps, CFG 7.0, seed 42, weight 0.8 | hswq: source 388,714 B, pack 382,349 B, dequant 380,184 B; stock: pack 2,447 B, mean 0.00 |
| 15 | 2026-11-15 20:04 | the score curve record | the six candidates | `records/cn_clip_vit_h14_laion2b_score_curve.json` |

---

## 8. Files

| file | role |
| :--- | :--- |
| `clip_convert/measure_h14_vision_contrib.py` | the criterion — the aggregate contribution |
| `clip_convert/convert_clip_impact_convrot_int8.py` | the builder (`--protect-list`, FP16 kept) |
| `audit/clip_vit_h14_pack_audit.py` | CN-1 — identity, coverage, arithmetic replay, behaviour |
| `audit/clip_vit_h14_route.py` | CN-2 — the route census, stock and hswq |
| `audit/clip_vit_h14_gate_and_trigger.py` | CN-5b / CN-6 — the trigger search and the 256-item gate |
| `audit/clip_vit_h14_int8_control.py` | the positive control with a measured noise floor |
| `audit/clip_vit_h14_int8_sweep.py` | the positive control over all 92 INT8 layers |
| `audit/clip_vit_h14_render.py` | CN-4 — the three renders on either route |
| `md/CN_CLIPViTH14_laion2B_ConvRot_INT8_Impact_Technical_Guide.md` | the runbook for this checkpoint |
| `md/backdoor/verdicts/CN_CLIP_ViT_H14_laion2B_Pack.md` | the backdoor verdict |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_protection.json` | the 192-layer contribution and its ranking |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_pack_audit.json` | CN-1 — FAITHFUL, the arithmetic replay, the behaviour |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_route.json` | CN-2 — the hswq route |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_route_stock.json` | CN-2 — the stock route, which does not apply the pack |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_gate_256.json` | CN-6 — the 256-item gate |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_score_curve.json` | the six candidates |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_image_trigger.json` | CN-5b — the 55 candidates and the trigger ladders |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_int8_control.json` | the positive control, one layer, three arms |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_int8_sweep.json` | the positive control over all 92 INT8 layers |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_render.json` | CN-4 — the renders and their pixel statistics |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_A_source_image_test.png` | the source render |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_B_pack_image_test.png` | the pack render |
