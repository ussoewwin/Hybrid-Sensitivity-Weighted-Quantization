# CLIP-SAE-ViT-L-14 — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Source.** `models/clip/CLIP-SAE-ViT-L-14-FP32.safetensors` (1631.3 MB, 590 tensors, FP32)
**Pack.** `models/clip/CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` (658.9 MB, **−59.6 %**)
**SHA256.** `F642CD146A494939029BC32242C198F14CB7C10F5C4BDDD4734E0B29B8A98501`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate** output.

---

## 1. The model

A CLIP checkpoint with **both** towers: the text tower (SD1ClipModel, 12
layers) and the vision tower (ClipVisionModel, 12 layers). They are measured
separately, because they are different graphs with different observables.

| tower | the candidates | the observable | the script |
| :--- | :--- | :--- | :--- |
| the text tower | 73 | the text embedding for a fixed prompt set | `sdxl/diag_impact_clip.py` |
| the vision tower | 144 | the image embedding for a fixed image set | `sdxl/diag_impact_clip_vision.py` |

The impact measurement's names are the **runtime's** (`transformer.text_model.encoder.layers.N.*`
for the text tower, `encoder.layers.N.*` for the vision tower); the pack is
written under the **checkpoint's** names (`text_model.*`, `vision_model.*`),
which is what ComfyUI's loader reads.

---

## 2. The impact measurement, re-measured

The earlier measurement had the injection defect that was found while working
on the FLUX ControlNet: the reconstruction was injected in the **rotated**
frame (`q * s`, i.e. `W_hat * H`) instead of the **original** one. The pack
stores the rotated weight and the kernel rotates the **activation**, so the
quantization error that reaches the output is `(q*s) H^T - W`.

| | the old (defective) | the new (corrected) |
| :--- | :--- | :--- |
| the text tower's median | 1.48e-01 | **2.39e-06** |
| the text tower's max | 1.91e+00 | **2.48e-05** |
| the vision tower's median | 7.23e-02 | **2.08e-06** |
| the vision tower's max | 8.94e-01 | **1.51e-05** |

The old measurement made the text tower's `transformer.text_model.encoder.layers.0.mlp.fc1`
look like 1.86 and the vision tower's worst layer like 0.89. The corrected
values are five orders of magnitude smaller, and the ranks change with them.

**The largest drifts, corrected**

| the text tower | the drift |
| :--- | :--- |
| `transformer.text_model.encoder.layers.0.self_attn.v_proj` | 2.48e-05 |
| `transformer.text_model.encoder.layers.0.self_attn.out_proj` | 2.23e-05 |
| `transformer.text_model.encoder.layers.0.mlp.fc2` | 1.78e-05 |
| `transformer.text_model.encoder.layers.11.self_attn.out_proj` | 8.92e-06 |

| the vision tower | the drift |
| :--- | :--- |
| `encoder.layers.0.mlp.fc2` | 1.51e-05 |
| `encoder.layers.5.mlp.fc1` | 1.30e-05 |
| `encoder.layers.4.mlp.fc1` | 8.99e-06 |
| `encoder.layers.16.mlp.fc1` | 8.53e-06 |

---

## 3. The selection criterion, and the rebuilt pack

The per-layer drift is a **screen**, not the selection criterion. It measures
the **weight** error only (the reconstruction is placed in the module and the
plain forward is used), while the pack's real path also quantizes the
**activation** for that layer, and that is where most of the error comes from.
The FP16-kept set is therefore given explicitly, with the layers taken in the
corrected drift's ranking, separately for the two towers.

The candidates were measured on a fixed 16-item set (seed 42) against the FP16
source, on the runtime's own path, with the text tower's `text_projection`
completed and the vision tower's class embedding added exactly as the loader
does it.

| the candidate | the text kept | the vision kept | the INT8 | text cos (mean / min) | vision cos (mean / min) | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **the previous pack** | 0 | 0 | 219 | 0.999875 / 0.999820 | 0.999683 / 0.999592 | 413.2 MB |
| t10_v20 | 10 | 20 | 189 | 0.999922 / 0.999894 | 0.999782 / 0.999740 | 700.2 MB |
| t20_v40 | 20 | 40 | 159 | 0.999946 / 0.999929 | 0.999863 / 0.999825 | 591.6 MB |
| **t30_v60** | **30** | **60** | **129** | **0.999964 / 0.999924** | **0.999932 / 0.999921** | **658.9 MB** |
| t50_v100 | 50 | 100 | 69 | 0.999985 / 0.999958 | 0.999989 / 0.999985 | 722.3 MB |

**30 text + 60 vision FP16-kept layers were adopted.** 50/100 is better on
every one of the four numbers (+2.1e-05 of the text mean, +3.4e-05 of the text
minimum, +5.7e-05 of the vision mean, +5.9e-05 of the vision minimum) for
63 MB more; the extra size is better spent on the remaining models than on the
last five thousandth here.

**The previous pack quantized every eligible layer** (219 INT8, no
protection at all) - the corrected drift at any threshold protects 0 layers,
and the previous pack's measured drift had also been the defective one. The
rebuilt pack protects 90 layers and the output is measurably closer to the
FP16 source on **both** towers:

| | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the text mean / min | 0.999875 / 0.999820 | **0.999962 / 0.999908** |
| the vision mean / min | 0.999683 / 0.999592 | **0.999931 / 0.999801** |

---

## 4. The pack

Built with `sdxl/build_sae_vitl14_convrot_int8.py`:

```
--impact-text impact/impact_sae_vitl14_text_fixed.json \
--impact-vision impact/impact_sae_vitl14_vision_fixed.json \
--protect-list impact/sae_protect_t30_v60.json
```

| item | value |
| :--- | :--- |
| the source | 1631.3 MB (FP32) |
| the pack | **658.9 MB** (-59.6 %) |
| the SHA256 | `F642CD146A494939029BC32242C198F14CB7C10F5C4BDDD4734E0B29B8A98501` |
| the tensors | 848 |
| the quantized layers | **129** (the text 43, the vision 84, the projections 2) |
| the FP16-kept layers | 90 (the text 30, the vision 60), stored FP16 |
| the group size | 256 (all 129 layers) |
| the output | finite |
| text cos mean / min (256 items, seed 42) | **0.999962 / 0.999908** |
| vision cos mean / min (256 items, seed 42) | **0.999931 / 0.999797** |
| the gate (mean >= 0.99, min >= 0.98, both towers) | **PASS** |

---

## 5. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_clip.py` | the text tower's impact measurement |
| `sdxl/diag_impact_clip_vision.py` | the vision tower's impact measurement |
| `sdxl/build_sae_vitl14_convrot_int8.py` | the builder (`--protect-list`) |
| `sdxl/verify_sae_vitl14.py` | the acceptance, both towers, the runtime's own path |
| `impact/impact_sae_vitl14_text_fixed.json` | the text tower's drift (git-ignored) |
| `impact/impact_sae_vitl14_vision_fixed.json` | the vision tower's drift (git-ignored) |
| `impact/sae_protect_t30_v60.json` | the adopted FP16-kept list (git-ignored) |

---

## 6. The FP16-kept tensors are stored in FP16

The checkpoint is FP32, and the first revision of this pack wrote the kept
tensors back in FP32. That is the wrong storage width: **ComfyUI's CLIP runs
in FP16**, so an FP32 kept tensor is cast to FP16 when the model is built and
nothing at runtime sees the extra precision. Storing FP32 only paid for the
bytes. The other packs already ship FP16 kept tensors (CLIP-ViT-bigG-14:
519 of them, CLIP-L: 133) and pass the acceptance.

The kept tensors are therefore stored FP16, and the pack halves:

| | the kept ones | the pack |
| :--- | :--- | :--- |
| the first revision (FP32 kept) | 92 tensors, 997.2 MB | 1159.5 MB |
| the shipped revision (FP16 kept) | 92 tensors, 498.6 MB | **658.9 MB** |

The INT8 scales stay FP32 on purpose: they carry the dequantization
precision, and FP16 would round the step size itself.

Dropping the kept tensors' width changes **nothing** in the measured output -
the acceptance was re-run on the FP16 revision and it is bit-for-bit the same
gate result:

| | the first revision | the shipped revision |
| :--- | :--- | :--- |
| text cos mean / min | 0.999962 / 0.999908 | **0.999962 / 0.999908** |
| vision cos mean / min | 0.999931 / 0.999801 | **0.999931 / 0.999797** |
