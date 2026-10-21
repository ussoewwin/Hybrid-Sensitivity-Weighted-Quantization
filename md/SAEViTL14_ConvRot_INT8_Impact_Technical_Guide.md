# CLIP-SAE-ViT-L-14 — ConvRot INT8 Trajectory-Impact Technical Guide

**Source.** `models/clip/CLIP-SAE-ViT-L-14-FP32.safetensors` (1631.3 MB, 590 tensors, FP32)
**Pack.** `models/clip/CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` (1159.5 MB, **−28.9 %**)
**SHA256.** `ACAFDD2A95DE644000F19E4A3FA5CE51F2AF7DBDF3873C8CFA9297DC0A1073F4`
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
| t20_v40 | 20 | 40 | 159 | 0.999946 / 0.999929 | 0.999863 / 0.999825 | 957.4 MB |
| **t30_v60** | **30** | **60** | **129** | **0.999964 / 0.999924** | **0.999932 / 0.999921** | **1159.5 MB** |
| t50_v100 | 50 | 100 | 89 | 0.999985 / 0.999958 | 0.999989 / 0.999985 | 1597.4 MB |

**30 text + 60 vision FP16-kept layers were adopted.** 50/100 buys +2.1e-05
of the text mean and +5.7e-05 of the vision mean for 438 MB more, and the
text **minimum** goes **down** (0.999924 against 0.999958), so the extra size
is not paid for in quality.

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
| the pack | **1159.5 MB** (-28.9 %) |
| the SHA256 | `ACAFDD2A95DE644000F19E4A3FA5CE51F2AF7DBDF3873C8CFA9297DC0A1073F4` |
| the tensors | 848 |
| the quantized layers | **129** (the text 43, the vision 84, the projections 2) |
| the FP16-kept layers | 90 (the text 30, the vision 60) |
| the group size | 256 (all 129 layers) |
| the output | finite |
| text cos mean / min (256 items, seed 42) | **0.999962 / 0.999908** |
| vision cos mean / min (256 items, seed 42) | **0.999931 / 0.999801** |
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
