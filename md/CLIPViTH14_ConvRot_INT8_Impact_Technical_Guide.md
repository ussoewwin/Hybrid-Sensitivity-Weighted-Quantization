# CLIP-ViT-H-14 — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/clip_vision/CLIP-ViT-H-14.safetensors` (2411.2 MB, FP32, 632,077,057 parameters)
**Pack.** `models/clip_vision/CLIP-ViT-H-14_hswq_convrot_int8.safetensors` (1015.6 MB, **−57.9 %**)
**SHA256.** `2F057DED08D8FA5E4A965749F241F10A3E654C9682970BCE292020130625DA80`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate** output.

---

## 1. The model

CLIP-ViT-H-14 is a **vision-only** CLIP checkpoint: 32 transformer layers,
hidden size 1280, image size 224, patch size 14.

| item | measured |
| :--- | :--- |
| checkpoint keys | 521 |
| parameters | 632,077,057 |
| dtypes | F32 (the 257 I64 tensors are `position_ids`) |
| runtime Linear modules | **192** (32 layers × 6) |
| the script | `sdxl/diag_impact_clip_vision.py` |

---

## 2. The impact measurement, re-measured

The earlier measurement had the injection defect found while working on the
FLUX ControlNet: the reconstruction was injected in the **rotated** frame
(`q * s`, i.e. `W_hat * H`) instead of the **original** one. The pack stores the
rotated weight and the kernel rotates the **activation**, so the quantization
error that reaches the output is `(q*s) H^T − W`.

| | the old (defective) | the new (corrected) |
| :--- | :--- | :--- |
| the min | 2.20e-03 | 3.39e-10 |
| the median | 3.34e-02 | **6.38e-07** |
| the p90 | 2.42e-01 | 1.79e-06 |
| the max | 7.97e-01 | **3.80e-06** |

The old measurement made the worst layer look like 0.797. The corrected values
are five orders of magnitude smaller, and with them **the ranks** — which is
what makes a drift-threshold protection set meaningless here. At the old
threshold (0.05) the corrected drift protects **nothing**; the shipped pack's
127-layer protection set had come from the defective values.

---

## 3. The selection criterion: the aggregate contribution

The per-layer drift is a **screen**, not the selection criterion. It measures
the **weight** error only, while the pack's real path also quantizes the
**activation** for that layer, and that is where most of the error comes from.

The selection is therefore made on the **aggregate output**: start from the
FP16 weights, replace ONE layer by its ConvRot INT8 reconstruction, and measure
the vision embedding against the FP16 embedding. The layer's **contribution**
is `1 − cos`. A layer whose quantization barely moves the output can stay INT8;
a layer that moves it must be protected.

`1 − cos` over the 192 layers, measured with the baseline on the **same** ops
class (`mixed_precision_ops`, not stock `manual_cast`):

| | value |
| :--- | :--- |
| the min | −8.20e-08 |
| the p10 | 1.49e-08 |
| the median | 4.06e-07 |
| the p90 | 1.31e-06 |
| the max | **1.92e-06** |

**The top of the ranking is the MLP blocks.** The 20 largest contributions are
17 MLP layers (`mlp.fc1` / `mlp.fc2`) and 3 attention `v_proj`:

| the layer | 1 − cos |
| :--- | :--- |
| `encoder.layers.1.mlp.fc2` | 1.92e-06 |
| `encoder.layers.24.mlp.fc1` | 1.87e-06 |
| `encoder.layers.1.mlp.fc1` | 1.83e-06 |
| `encoder.layers.28.mlp.fc1` | 1.83e-06 |
| `encoder.layers.27.mlp.fc1` | 1.75e-06 |
| `encoder.layers.24.mlp.fc2` | 1.57e-06 |

The smallest are the attention `q_proj` / `k_proj` of the early layers, several
of them with a **negative** contribution (protecting them would not help).

---

## 4. The candidates, and the adopted set

The packs were built from the contribution's ranking (the top-K protected) and
measured on the fixed **256-item** acceptance set (seed 42):

| the candidate | the INT8 | the kept | cos mean | cos min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| all-INT8 (k=0) | 192 | 0 | 0.999798 | 0.999705 | 0.999750 | **609.6 MB** |
| k=15 | 177 | 15 | 0.999849 | 0.999743 | 0.999815 | 698.5 MB |
| k=30 | 162 | 30 | 0.999885 | 0.999797 | 0.999856 | 787.4 MB |
| k=60 | 132 | 60 | 0.999932 | 0.999885 | 0.999907 | 920.5 MB |
| **k=100 (adopted)** | **92** | **100** | **0.999968** | **0.999906** | **0.999951** | **1015.6 MB** |
| k=140 | 52 | 140 | 0.999993 | 0.999981 | 0.999988 | 1127.2 MB |

**100 FP16-kept layers were adopted.** k=140 buys +2.5e-05 of the mean and
+7.5e-05 of the min for 112 MB more; the extra size is better spent on the
remaining models than on the last increment here. k=60 is 95 MB smaller but
its min (0.999885) is below k=100's mean-gap, so k=100 is where the quality per
byte stops improving sharply.

### 4.1 The kept tensors are stored in FP16

The source stores its tensors in **FP32**, so the kept tensors were being
written back FP32 — the same defect found on the CLIP-SAE-ViT-L-14 and the
Long-ViT-L-14 packs. ComfyUI's CLIP runs in FP16, so an FP32 kept tensor is
cast to FP16 when the model is built and nothing at runtime sees the extra
precision; storing FP32 only pays for the bytes.

All floating-point kept tensors are stored FP16 now, and the INT8 scales stay
FP32 (they carry the dequantization precision). On this checkpoint the source
is FP32 throughout, but the vision runtime state dict already reports its
tensors in FP16 except `visual_projection` (5.0 MB), so the storage change is
small here: measured on the k=100 pack, 1018.1 MB (FP32 kept) -> **1015.6 MB**
(FP16 kept), with the acceptance unchanged (0.999968 / 0.999906 in both).

---

## 5. The pack

| item | value |
| :--- | :--- |
| the source | 2411.2 MB (FP32) |
| the pack | **1015.6 MB** (-57.9 %) |
| the SHA256 | `2F057DED08D8FA5E4A965749F241F10A3E654C9682970BCE292020130625DA80` |
| the tensors | 705 |
| the quantized layers | **92** (group size 256, all) |
| the FP16-kept layers | 100 |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42) | **0.999968 / 0.999906 / 0.999951** |
| the gate (mean ≥ 0.99, min ≥ 0.98) | **PASS** |

Against the previous pack (both on the same 256-item set, seed 42):

| | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the cos mean / min | 0.999920 / 0.999866 | **0.999968 / 0.999906** |
| the size | 949.5 MB | 1015.6 MB |

The rebuilt pack is 66 MB larger for a 2.5× smaller mean error and a 3.4×
smaller worst-case error. The previous pack's protection set had protected
67 layers chosen on the defective drift (47 of them overlapping the corrected
top-67), i.e. the wrong layers.

---

## 6. Loading the pack

The vision tower is loaded with `mixed_precision_ops` so the `comfy_quant`
stamps are consumed:

```python
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS
ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                            torch.float16, full_precision_mm=False)
```

The stock `CLIPVisionLoader` uses `manual_cast`, which does **not** consume the
stamps; a pack loaded that way keeps its quantized tensors as int8 storage.
`vision_model.embeddings.class_embedding` is absent in the checkpoint and is
completed with zeros (dim 1280) when the tower is built standalone.

---

## 7. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_clip_vision.py` | the drift measurement (the screen) |
| `sdxl/measure_h14_vision_contrib.py` | the aggregate contribution (the criterion) |
| `clip_convert/convert_clip_impact_convrot_int8.py` | the builder (`--protect-list`, FP16 kept) |
| `sdxl/verify_vision.py` | the acceptance on the fixed 256-item set |
| `impact/impact_vit_h14_vision_fixed.json` | the corrected drift (git-ignored) |
| `impact/contrib_vit_h14.json` | the contribution (git-ignored) |
| `impact/h14_protect_100.json` | the adopted FP16-kept list (git-ignored) |
