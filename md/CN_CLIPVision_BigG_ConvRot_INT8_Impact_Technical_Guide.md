# CLIP-ViT-bigG-14 (`clip_vision_g.safetensors`) — ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) — Section *How a pack is selected (the current method)*.
> This guide is the runbook for this checkpoint. The measured numbers are in
> [`CLIPVision_BigG_ConvRot_INT8_Quantization_and_Score_Report.md`](CLIPVision_BigG_ConvRot_INT8_Quantization_and_Score_Report.md);
> the verdict is in
> [`backdoor/verdicts/CLIP_Vision_G_Pack.md`](backdoor/verdicts/CLIP_Vision_G_Pack.md).

**Model.** `models/clip_vision/clip_vision_g.safetensors` (3,689,911,098 B, 777 tensors) — the same weights as `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` (776 of 777 tensors byte-identical).
**Pack.** `models/clip_vision/clip_vision_g_hswq_convrot_int8.safetensors` (1,852,954,770 B, **−49.8 %**)
**SHA256 (source).** `9908329B3EAD722A693EA400FAB1D7C9EC91D6736FD194A94D20D793457F9C2E`
**SHA256 (pack).** `70A39F1264FBC7870B246666C931690CB754A9A99F11CF210F5C6DF207D8ADBB`

---

## 1. What this model is

A vision-only CLIP checkpoint: 32 transformer layers, hidden size 1664, image size 224, patch
size 14, 288 encoder Linear modules — all of them quantisable.

| item | measured |
| :--- | :--- |
| checkpoint keys | 777 (1 int64: `vision_model.embeddings.position_ids`) |
| the runtime Linear modules | **288** |
| the group size | **64** — the 1664-wide layers cannot carry a 256-way group |
| the two weights the format cannot take | `position_embedding` (rows wider than the group), `visual_projection` (1664→1280, asymmetric) |

---

## 2. The measurement

Two rankings, both measured on this checkpoint.

**The per-layer drift** (the screen), `clip_convert/diag_impact_clip_vision.py`, 288 layers,
seed 42, the reconstruction injected in the **original** frame:

| | value |
| :--- | :--- |
| the median | 1.1163e-07 |
| the p90 | 1.9330e-06 |
| the max | 9.1210e-06 (`encoder.layers.45.mlp.fc1`) |
| the layers above 0.15 | **0** |

**The aggregate contribution** (the criterion), `clip_convert/measure_bigg_vision_contrib_inmodel.py`:
the all-INT8 pack is built and loaded once, and each layer is replaced in place by the source's
fp16 weight while the tower runs over 8 fixed seeds.

| | value |
| :--- | :--- |
| the all-INT8 reference | mean 0.999732, min 0.999692 |
| the largest contribution | 3.3081e-05 (`encoder.layers.28.self_attn.q_proj`) |
| the median contribution | -5.1260e-06 |
| **the layers with a negative contribution** | **173 of 288** |

**Why the in-model form.** The per-layer-pack form (`measure_bigg_vision_contrib.py`) builds a
fresh pack per candidate layer and reloads it: about half a minute per layer, and the vision
tower is not released between packs, so the VRAM fills as the sweep proceeds. The in-model form
measures the same thing with one pack resident — 288 layers in about seven minutes. It was
checked against the per-layer-pack form on one layer, in one process:

| the method | the gain for `encoder.layers.0.self_attn.q_proj` |
| :--- | :--- |
| the per-layer pack | +1.3888e-05 |
| the in-model swap | +1.3888e-05 |
| the difference | **+0.0000e+00** |

**The negative majority is the finding.** Restoring one layer of 288 to fp16 does not move the
output over most of the population, so the curve is flat by construction — see §3.

---

## 3. The candidates, and the adopted point

Six candidates from the contribution ranking, all measured on the full 256-item acceptance set
(seed 42) through the `HSWQLoadClipVision` route, plus the pack being replaced for reference:

| the FP16-kept | INT8 | cos mean | cos min | p5 | size | mean gain / margin |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0 (adopted)** | **288** | **0.999734** | **0.999554** | **0.999654** | **1767.1 MB** | **the floor** |
| 10 | 278 | 0.999747 | 0.999545 | 0.999677 | 1824.5 MB | 0.14 % |
| 20 | 268 | 0.999763 | 0.999663 | 0.999709 | 1923.3 MB | 0.30 % |
| 30 | 258 | 0.999802 | 0.999578 | 0.999737 | 2001.3 MB | 0.70 % |
| 35 | 253 | 0.999806 | 0.999719 | 0.999762 | 2035.2 MB | 0.74 % |
| 40 | 248 | 0.999808 | 0.999674 | 0.999752 | 2058.7 MB | 0.76 % |
| the pack being replaced | 253 | 0.999910 | 0.999725 | 0.999875 | 2180.1 MB | 1.8 % |

`margin = 0.999734 − 0.99 = 9.734e-03` — the distance the gate already has at the floor.

**The floor is the adopted point.** It passes the gate with 0.009734 of margin on the mean and
0.019554 on the minimum, it is the smallest candidate measured, and the next 291.6 MB of
protection buys 0.76 % of that margin. On this checkpoint protection does not buy quality
because 173 of 288 layers have a negative contribution.

### 3.1 The kept tensors are stored in FP16

The pack keeps 489 float tensors besides the scales; their values are the source's (0
differences, max abs diff 0.0, 0 dtype differences). The INT8 `weight_scale` stays FP32 because
it carries the dequantisation precision.

---

## 4. The pack

Built with `clip_convert/convert_clip_impact_convrot_int8.py`:

```
--clip_type vision --drift-threshold 0.05 --groupsize 64 \
  --protect-list impact/bigg_g_protect_0.json
```

(the protect list is empty — the floor has no protected layer)

| item | value |
| :--- | :--- |
| the source | 3,689,911,098 B |
| the pack | **1,852,954,770 B** (−49.8 %) |
| the SHA256 | `70A39F1264FBC7870B246666C931690CB754A9A99F11CF210F5C6DF207D8ADBB` |
| the tensors | 1,353 |
| the quantised layers | **288** (of 288) |
| the FP16-kept layers | 0 from the ranking; 2 by format (`position_embedding`, `visual_projection`) |
| the group size | 64 |
| the `int8_linear` calls per forward | **288** |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42) | **0.999734 / 0.999554 / 0.999654** |
| the gate (mean ≥ 0.99, min ≥ 0.98) | **PASS** |

---

## 5. Loading the pack

The vision tower is loaded with `mixed_precision_ops` so the `comfy_quant` stamps are consumed:

```python
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS
ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                            torch.float16, full_precision_mm=False)
```

ComfyUI's own `CLIPVisionLoader` uses `manual_cast`, which does **not** consume the stamps: a
pack loaded that way keeps its int8 codes in a float Parameter, and the conditioning becomes a
different conditioning (measured output cosine −0.035036381, and the render shows a different
picture). The loader extension's `HSWQLoadClipVision` does consume them.

---

## 6. Reproduction

```powershell
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe

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

# 4. the acceptance on the real INT8 route
python clip_convert/verify_bigg_vision.py "<src>/clip_vision_g.safetensors" "<out>/k0.safetensors" ...

# 5. the gate and the trigger search, positive control armed
python audit/clip_vision_bigg_gate_and_trigger.py --source ... --pack ... --image ... `
    --comfy ... --loader ... --route hswq --n 256 ...

# 6. the audit, the route census, the render
python audit/bigg_pack_audit.py --source ... --pack ... --comfy ... --json ...
python audit/clip_vision_bigg_route.py --source ... --pack ... --image ... --comfy ... --loader ... --route stock|hswq --json ...
python audit/clip_vision_bigg_render.py --comfy ... --server ... --reference cn_clipvit_ref.png --route hswq --out ... --frames ...
```

**The step everyone gets wrong:** `--groupsize 64`. The 1664-wide layers cannot carry a
256-way group.

**The second step:** the pack lives next to its source in `models/clip_vision/`; a render also
needs it reachable from `models/clip/`.

**The third step:** the positive control. A ConvRot layer holds a `QuantizedTensor`, so a float
implant cannot be written into it — the ladder runs on `position_embedding`, which the observable
passes through. Its measured floor is a relative strength of 0.1 (1−cos 7.613862e-03).

---

## 7. Files

| file | role |
| :--- | :--- |
| `clip_convert/diag_impact_clip_vision.py` | the drift measurement (the screen) |
| `clip_convert/measure_bigg_vision_contrib_inmodel.py` | the aggregate contribution, one pack resident |
| `clip_convert/measure_bigg_vision_contrib.py` | the same measurement, one pack per layer — the reference |
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
