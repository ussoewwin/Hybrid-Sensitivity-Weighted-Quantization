# CLIP-ViT-H-14 (laion2B-s32B-b79K) — ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) — Section *How a pack is selected (the current method)*.
> This guide is the runbook for this checkpoint. The measured numbers are in
> [`CLIPViTH14_laion2B_ConvRot_INT8_Quantization_and_Score_Report.md`](CLIPViTH14_laion2B_ConvRot_INT8_Quantization_and_Score_Report.md);
> the backdoor verdict is in
> [`backdoor/verdicts/CN_CLIP_ViT_H14_laion2B_Pack.md`](backdoor/verdicts/CN_CLIP_ViT_H14_laion2B_Pack.md).

**Model.** `models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` (2,528,373,448 B, 521 tensors)
**Pack.** `models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K_hswq_convrot_int8.safetensors` (1,064,907,988 B, **−57.9 %**)
**SHA256 (source).** `6CA9667DA1CA9E0B0F75E46BB030F7E011F44F86CBFB8D5A36590FCD7507B030`
**SHA256 (pack).** `477E5271A94349109D31A482EE870581F55F39059036CDFCC13DC6314CCE7323`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate** output.

---

## 1. The model

CLIP-ViT-H-14 is a **vision-only** CLIP checkpoint: 32 transformer layers, hidden size 1280,
image size 224, patch size 14. The checkpoint carries no `visual_projection`.

| item | measured |
| :--- | :--- |
| checkpoint keys | 521 |
| the runtime Linear modules | **192** (32 layers × 6) |
| the script | `clip_convert/measure_h14_vision_contrib.py` |

The source is stored in FP32. The kept tensors are written back **FP16**, because the runtime
runs in FP16 and an FP32 kept tensor is cast at load time — it only pays for the bytes. The
INT8 `weight_scale` stays FP32 because it carries the dequantisation precision.

`vision_model.embeddings.class_embedding` is absent from the checkpoint and is completed with
zeros when the tower is built standalone; the measurement script takes the dimension from the
checkpoint rather than assuming one.

---

## 2. The criterion: the aggregate contribution

The per-layer drift is a **screen**, not the selection criterion. It measures the **weight**
error only, while the pack's real path also quantises the **activation** for that layer, and
that is where most of the error comes from.

The selection is made on the **aggregate output**: start from the FP16 weights, replace ONE
layer by its ConvRot INT8 reconstruction (rotate → row-wise quantise → dequantise → un-rotate),
and measure the tower's embedding against the FP16 output. The layer's **contribution** is
`1 − cos`.

`1 − cos` over the 192 layers, 8 seeds, seed 42, measured with the baseline on the **same** ops
class (`mixed_precision_ops`, not ComfyUI's own `manual_cast`):

| | value |
| :--- | :--- |
| the min | −3.725290e-08 |
| the median | 4.209578037261963e-07 |
| the p90 | 1.3113021850585938e-06 |
| the max | **1.9073486328125e-06** (`encoder.layers.1.mlp.fc2`) |
| the sum over the 192 layers | 0.00010217726230621338 |

**The top of the ranking is the MLP blocks.** The share column is what says the curve is flat:
the largest layer carries **1.87 %** of the summed contribution and the fifth carries 1.71 %.
Eight layers have a **negative** contribution — protecting them would not help.

| the layer | 1 − cos | the share |
| :--- | :--- | :--- |
| `encoder.layers.1.mlp.fc2` | 1.9073486328125e-06 | 1.8667 % |
| `encoder.layers.24.mlp.fc1` | 1.899898e-06 | 1.8594 % |
| `encoder.layers.1.mlp.fc1` | 1.817942e-06 | 1.7792 % |
| `encoder.layers.28.mlp.fc1` | 1.817942e-06 | 1.7792 % |
| `encoder.layers.27.mlp.fc1` | 1.743436e-06 | 1.7063 % |
| `encoder.layers.21.mlp.fc2` | 1.549721e-06 | 1.5157 % |

The ranking file the converter consumed is `impact/contrib_vit_h14_laion2b.json`, copied to
`md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_protection.json`.

---

## 3. The candidates, and the adopted set

The packs are built from the contribution's ranking (the top-K protected) and measured on the
fixed **256-item** acceptance set (seed 42) through the loader's ConvRot INT8 route:

| the candidate | the INT8 | the kept | cos mean | cos min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 192 | 0 | 0.998089762 | 0.988531062 | 0.994754711 | 636,631,336 B |
| k=15 | 177 | 15 | 0.998473382 | 0.993083664 | 0.996227620 | 729,830,411 B |
| k=30 | 162 | 30 | 0.998538490 | 0.993156433 | 0.996141628 | 823,013,318 B |
| k=60 | 132 | 60 | 0.998692746 | 0.993619517 | 0.996385444 | 965,264,724 B |
| **k=100 (adopted)** | **92** | **100** | **0.999044517** | **0.995568918** | **0.997158609** | **1,064,907,988 B** |
| k=140 | 52 | 140 | 0.999702289 | 0.995363910 | 0.998700873 | 1,179,312,148 B |

**100 FP16-kept layers were adopted.** k=140 buys +6.58e-04 of the mean but **loses** 2.05e-04
of the min, for 114,404,160 B more; the extra size buys a column that is already past the gate
and costs the worst case. k=60 is 99.6 MB smaller but its min (0.993620) is 1.95e-03 below
k=100's, so k=100 is where the quality per byte stops improving sharply.

Every candidate clears the gate (mean ≥ 0.99, min ≥ 0.98) on its own, so the choice is the
size/quality balance, not the first passing pack.

### 3.1 The tie at the boundary

Two layers share the value 1.817942e-06 and several more share 1.3113021850585938e-06, so the boundary of
the top-100 set is decided by the sort order among equal values. A pack built from a protect
list that breaks those ties differently differs from this one in **10 layers** (measured: the
INT8/FP16 split differs on `layers.4.mlp.fc2`, `layers.5.mlp.fc1`, `layers.7.self_attn.v_proj`,
`layers.8.mlp.fc2`, `layers.10.self_attn.out_proj`, `layers.17.self_attn.out_proj`,
`layers.25.self_attn.v_proj`, `layers.29.self_attn.k_proj`, `layers.29.self_attn.q_proj`,
`layers.29.self_attn.out_proj`). The protect list is written out in full, so the set is
reproducible from the record.

---

## 4. The pack

| item | value |
| :--- | :--- |
| the source | 2,528,373,448 B (521 tensors) |
| the pack | **1,064,907,988 B** (−57.9 %), 705 tensors |
| the quantised layers | **92** (group size 256) |
| the FP16-kept layers | 100 (the contribution's top 100) |
| the group size | 256 |
| the format | `int8_tensorwise`, ConvRot on |
| the output | finite |

---

## 5. Loading the pack

The vision tower is loaded with `mixed_precision_ops` so the `comfy_quant` stamps are consumed:

```python
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS
ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                            torch.float16, full_precision_mm=False)
```

ComfyUI's own `CLIPVisionLoader` uses `manual_cast`, which does **not** consume the stamps; a
pack loaded that way keeps its int8 codes in a float Parameter and produces an unusable
conditioning. The loader extension's `HSWQLoadClipVision` does consume them.

Measured on this pack: the stock route presents **92** raw int8 codes and fires **0** INT8
kernels; the `HSWQLoadClipVision` route loads **92** as int8 and fires **92** INT8 GEMMs per
forward.

---

## 6. Reproduction

```powershell
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe

# 1. the criterion: the aggregate contribution (192 layers, 8 seeds, seed 42)
#    NOTE: this script does os.chdir(COMFY), so --out must be an absolute path
python clip_convert/measure_h14_vision_contrib.py `
    --src "<src>/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" `
    --out "<repo>/impact/contrib_vit_h14_laion2b.json" --n 8 --seed 42 --groupsize 256

# 2. the protect list, from that ranking
#    (top-K by contribution; see 3.1 for the tie at the boundary)

# 3. the builder (threshold 0.15, groupsize 256, FP16 kept)
python clip_convert/convert_clip_impact_convrot_int8.py `
    --input "<src>/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" `
    --output "<out>/CLIP-ViT-H-14-laion2B-s32B-b79K_hswq_convrot_int8.safetensors" `
    --impact "<repo>/impact/contrib_vit_h14_laion2b.json" `
    --comfy_path "<ComfyUI>" --clip_type vision --drift-threshold 1.0 --groupsize 256 `
    --protect-list "<repo>/impact/h14_laion2b_protect_100.json"

# 4. the acceptance on the real INT8 route (256 items, seed 42)
python audit/clip_vit_h14_gate_and_trigger.py `
    --source "<src>/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" `
    --pack   "<out>/CLIP-ViT-H-14-laion2B-s32B-b79K_hswq_convrot_int8.safetensors" `
    --image  "<ComfyUI>/input/cn_clipvit_ref.png" --comfy "<ComfyUI>" `
    --loader "<ComfyUI>/custom_nodes/ComfyUI-HSWQ-Loader-and-Tools" `
    --route hswq --n 256 `
    --json-gate    md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_gate_256.json `
    --json-trigger md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_image_trigger.json

# 5. the pack audit, the route census, the positive controls and the render
python audit/clip_vit_h14_pack_audit.py --source ... --pack ... --image ... --comfy ... --json ...
python audit/clip_vit_h14_route.py       --source ... --pack ... --image ... --comfy ... --loader ... --route stock|hswq --json ...
python audit/clip_vit_h14_int8_control.py --source ... --pack ... --image ... --comfy ... --loader ... --threshold <control p99> --json ...
python audit/clip_vit_h14_int8_sweep.py  --pack ... --image ... --comfy ... --loader ... --threshold <control p99> --json ...
python audit/clip_vit_h14_render.py       --comfy ... --server ... --reference cn_clipvit_ref.png --route hswq --out ... --frames ...
```

**The step everyone gets wrong:** `--out` and `--json` must be absolute paths for the scripts
that `os.chdir` into the ComfyUI tree, and `--route hswq` needs `--loader` pointing at the
loader extension. A relative `--out` fails with `FileNotFoundError` at the very end of a long
run, after the measurement has already been paid for.

---

## 7. Files

| file | role |
| :--- | :--- |
| `clip_convert/measure_h14_vision_contrib.py` | the criterion — the aggregate contribution |
| `clip_convert/convert_clip_impact_convrot_int8.py` | the builder (`--protect-list`, FP16 kept) |
| `audit/clip_vit_h14_pack_audit.py` | CN-1 — identity, coverage, arithmetic replay, behaviour |
| `audit/clip_vit_h14_route.py` | CN-2 — the route census (`--route stock` / `hswq`) |
| `audit/clip_vit_h14_gate_and_trigger.py` | CN-5b / CN-6 — the trigger search and the 256-item gate |
| `audit/clip_vit_h14_int8_control.py` | the positive control with a measured noise floor |
| `audit/clip_vit_h14_int8_sweep.py` | the positive control over all 92 INT8 layers |
| `audit/clip_vit_h14_render.py` | CN-4 — the three renders (`--route stock` / `hswq`) |
| `impact/contrib_vit_h14_laion2b.json` | the contribution (git-ignored) |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_protect_100.json` | the adopted FP16-kept list (git-ignored) |
| `md/backdoor/verdicts/records/cn_clip_vit_h14_laion2b_*.json` | the records behind every number above |
