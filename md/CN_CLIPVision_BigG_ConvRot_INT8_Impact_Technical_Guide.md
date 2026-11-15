# CLIP-ViT-bigG-14 (`clip_vision_g.safetensors`) — ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) — Section *How a pack is selected (the current method)*.
> This guide is the runbook for this checkpoint. The measured numbers are in
> [`CLIPVision_BigG_ConvRot_INT8_Quantization_and_Score_Report.md`](CLIPVision_BigG_ConvRot_INT8_Quantization_and_Score_Report.md);
> the verdict is in
> [`backdoor/verdicts/CN_CLIP_Vision_BigG_Pack.md`](backdoor/verdicts/CN_CLIP_Vision_BigG_Pack.md).

**Model.** `models/clip_vision/clip_vision_g.safetensors` (3,689,911,098 B, 777 tensors) — the same weights as `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` (776 of 777 tensors byte-identical).
**Pack.** `models/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` (2,285,956,858 B, **−38.0 %**)
**SHA256 (source).** `9908329B3EAD722A693EA400FAB1D7C9EC91D6736FD194A94D20D793457F9C2E`
**SHA256 (pack).** `6CD4AFAC314AE720FF1EDCAC59F927B93D8E15CE3E8E109DDC59F6FD43B2D5D3`

---

## 1. What this model is

A vision-only CLIP checkpoint: 32 transformer layers, hidden size 1664, image size 224, patch
size 14, 288 encoder Linear modules. The checkpoint carries no `visual_projection` under the
`vision_model.` prefix; the tower is built through `CLIPVisionModelProjection`.

| item | measured |
| :--- | :--- |
| checkpoint keys | 777 (1 int64: `vision_model.embeddings.position_ids`) |
| the runtime Linear modules | **288** |
| the group size | **64** — the 1664-wide layers cannot carry a 256-way group |

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
the all-INT8 pack is built and loaded once, and each layer is replaced in place by the
source's fp16 weight while the tower runs over 8 fixed seeds.

| | value |
| :--- | :--- |
| the all-INT8 reference | mean 0.999732, min 0.999692 |
| the largest contribution | 3.3081e-05 (`encoder.layers.28.self_attn.q_proj`) |
| the median contribution | -5.1260e-06 |
| **the layers with a negative contribution** | **173 of 288** |

**Why the in-model form.** The per-layer-pack form (`measure_bigg_vision_contrib.py`) builds a
fresh pack per candidate layer and reloads it: about half a minute per layer, and the vision
tower is not released between packs, so the VRAM fills as the sweep proceeds. The in-model form
measures the same thing with one pack resident - 288 layers in about seven minutes. It was
checked against the per-layer-pack form on one layer, in one process:

| the method | the gain for `encoder.layers.0.self_attn.q_proj` |
| :--- | :--- |
| the per-layer pack | +1.3888e-05 |
| the in-model swap | +1.3888e-05 |
| the difference | **+0.0000e+00** |

**The negative majority is the finding.** Restoring one layer of 288 to fp16 does not move the
output at all over most of the population, so the contribution ranking is noise where the
drift's is not. The candidates built from it measure worse than the shipped pack - see the
report, section 3.

---

## 3. The candidates, and the adopted set

Six candidates from the contribution ranking, plus the shipped pack, all measured on the full
256-item acceptance set (seed 42) through the `HSWQLoadClipVision` route:

| the FP16-kept | INT8 | cos mean | cos min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 (all quantised) | 288 | 0.999734 | 0.999554 | 0.999654 | 1767.1 MB |
| 10 | 278 | 0.999747 | 0.999545 | 0.999677 | 1824.5 MB |
| 20 | 268 | 0.999763 | 0.999663 | 0.999709 | 1923.3 MB |
| 30 | 258 | 0.999802 | 0.999578 | 0.999737 | 2001.3 MB |
| 35 | 253 | 0.999806 | 0.999719 | 0.999762 | 2035.2 MB |
| 40 | 248 | 0.999808 | 0.999674 | 0.999752 | 2058.7 MB |
| **the shipped pack** | **253** | **0.999910** | **0.999725** | **0.999875** | **2180.1 MB** |

**The shipped pack is the adopted one.** It is above every candidate rebuilt from the
contribution ranking — mean by 1.04e-04, p5 by 1.13e-04 — at the same 253 INT8 layers. Its kept
set is the **drift** ranking's top 35 (measured: the pack's 35 kept Linears occupy drift ranks
1 to 35), and the two top-35 sets share 8 layers of 35.

### 3.1 The kept tensors are stored in FP16

The source stores its tensors in FP32 or FP16 and the runtime runs in FP16. The pack keeps 524
float tensors; their values are the source's (0 differences, max abs diff 0.0). Five of them are
stored with a different dtype than the source — four LayerNorm tensors in FP32 and
`position_ids` — which is recorded in the audit record.

---

## 4. The pack

Built with `clip_convert/convert_clip_impact_convrot_int8.py`:

```
--clip_type vision --drift-threshold 0.05 --groupsize 64 \
  --protect-list impact/bigg_g_protect_35.json
```

(The builder derives the per-layer group size: 64 for the 1664-wide layers.)

| item | value |
| :--- | :--- |
| the source | 3,689,911,098 B |
| the pack | **2,285,956,858 B** (−38.0 %) |
| the SHA256 | `6CD4AFAC314AE720FF1EDCAC59F927B93D8E15CE3E8E109DDC59F6FD43B2D5D3` |
| the tensors | 1,283 |
| the quantised layers | **253** (of 288) |
| the FP16-kept layers | 35 |
| the group size | 64 |
| the `int8_linear` calls per forward | **253** |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42) | **0.999910 / 0.999725 / 0.999875** |
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
different conditioning (measured output cosine −0.034884527, and the render shows a different
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

# 3. the candidates
python clip_convert/convert_clip_impact_convrot_int8.py `
    --input "<src>/clip_vision_g.safetensors" --output "<out>/k35.safetensors" `
    --impact "<repo>/impact/impact_bigg_g_vision.json" `
    --comfy_path "<ComfyUI>" --clip_type vision --drift-threshold 0.05 --groupsize 64 `
    --protect-list "<repo>/impact/bigg_g_protect_35.json"

# 4. the acceptance on the real INT8 route
python clip_convert/verify_bigg_vision.py "<src>/clip_vision_g.safetensors" "<out>/k35.safetensors" ...

# 5. the audit, the route census, the render
python audit/bigg_pack_audit.py --source ... --pack ... --comfy ... --json ...
python audit/clip_vision_bigg_route.py --source ... --pack ... --image ... --comfy ... --loader ... --route stock|hswq --json ...
python audit/clip_vision_bigg_render.py --comfy ... --server ... --reference cn_clipvit_ref.png --route hswq --out ... --frames ...
```

**The step everyone gets wrong:** `--groupsize 64`. The 1664-wide layers cannot carry a
256-way group, and the builder has to be told, or the shapes do not match.

**The second step:** the pack lives in `models/clip/`, and a render needs it reachable from
`models/clip_vision/` as well.

---

## 7. Files

| file | role |
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
| `audit/clip_vision_bigg_render.py` | CN-4 |
| `audit/clip_vision_bigg_dequant.py` | the dequantised copy for the third render arm |
| `impact/impact_bigg_g_vision.json` | the drift (git-ignored) |
| `impact/contrib_bigg_g.json` | the contribution (git-ignored) |
| `impact/bigg_g_protect_*.json` | the candidate protect lists (git-ignored) |
| `md/backdoor/verdicts/records/cn_clip_vision_bigg_*.json` | the records behind every number above |
