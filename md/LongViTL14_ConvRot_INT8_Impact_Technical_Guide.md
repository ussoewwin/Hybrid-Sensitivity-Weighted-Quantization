# Long-ViT-L-14-BEST-GmP-smooth-ft-full — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Source.** `models/clip/Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors` (1631.8 MB, 590 tensors, FP32).

**Provenance correction.** The previous pack was measurably a quantisation of a different
checkpoint from the one this name publishes: its 167 FP16-kept tensors matched the local
file bit for bit and disagreed with the published checkpoint in all 167 (median 8.507e-4).
The pack was therefore rebuilt from zer0int's published `BEST-GmP-smooth-ft` (sha256
`8e259dc8...`, 1,711,063,028 bytes), re-measured on that source, and re-accepted. The audit
of the retired unidentified checkpoint, which closed the backdoor question but found no
published identity, is in `md/verdicts/LongCLIP_ViT_L14.md`.
**Pack.** `models/clip/Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` (490.6 MB, sha256 `3ED7D2CC...1AD89`), rebuilt from the published checkpoint
**SHA256.** `3ED7D2CCB7DE29D6AD5DBE5661746234F8CC3CF579DCDB58D78F71DD8BB1AD89` (the retired pack's `9F8BA355...` remains in the retired-checkpoint record)
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate** output.

---

## 1. The model

An OpenAI CLIP ViT-L/14 checkpoint with an **extended text context** (position
embedding 248×768 against the standard 77) and **both** towers. They are
measured separately, because they are different graphs with different
observables.

| tower | the candidates | the observable | the script |
| :--- | :--- | :--- | :--- |
| the text tower | 73 | the text embedding for a fixed prompt set | `sdxl/diag_impact_clip.py` |
| the vision tower | 144 | the image embedding for a fixed image set | `sdxl/diag_impact_clip_vision.py` |

`text_projection.weight` and `vision_model.embeddings.class_embedding` are both
present, so no completion step is needed on either load path.

The impact measurement's names are the **runtime's** (`transformer.text_model.encoder.layers.N.*`
for the text tower, `encoder.layers.N.*` for the vision tower); the pack is
written under the **checkpoint's** names (`text_model.*`, `vision_model.*`),
which is what ComfyUI's loader reads.

---

## 2. The impact measurement, re-measured

The earlier measurement had the injection defect found while working on the
FLUX ControlNet: the reconstruction was injected in the **rotated** frame
(`q * s`, i.e. `W_hat * H`) instead of the **original** one. The pack stores
the rotated weight and the kernel rotates the **activation**, so the
quantization error that reaches the output is `(q*s) H^T - W`.

| | the old (defective) | the new (corrected) |
| :--- | :--- | :--- |
| the text tower's median | 2.20e-01 | **4.14e-06** |
| the text tower's max | 1.93e+00 | **3.72e-05** |
| the vision tower's median | 4.98e-02 | **1.54e-06** |
| the vision tower's max | 8.57e-01 | **3.70e-05** |

The old measurement made the text tower's
`transformer.text_model.encoder.layers.0.self_attn.v_proj` look like 1.33 and
the vision tower's worst layer like 0.86. The corrected values are five orders
of magnitude smaller, and the ranks change with them.

**The largest drifts, corrected**

| the text tower | the drift |
| :--- | :--- |
| `transformer.text_model.encoder.layers.0.self_attn.v_proj` | 3.72e-05 |
| `transformer.text_model.encoder.layers.0.self_attn.out_proj` | 1.73e-05 |
| `transformer.text_model.encoder.layers.0.mlp.fc2` | 1.63e-05 |

| the vision tower | the drift |
| :--- | :--- |
| `encoder.layers.0.mlp.fc2` | 3.70e-05 |
| `encoder.layers.6.mlp.fc1` | 2.64e-05 |
| `encoder.layers.5.mlp.fc1` | 2.06e-05 |

---

## 3. The selection criterion, and the rebuilt pack

The per-layer drift is a **screen**, not the selection criterion. It measures
the **weight** error only (the reconstruction is placed in the module and the
plain forward is used), while the pack's real path also quantizes the
**activation** for that layer, and that is where most of the error comes from.
The FP16-kept set is therefore given explicitly, with the layers taken in the
corrected drift's ranking, separately for the two towers.

The candidates, measured on a fixed 16-item set (seed 42) against the FP16
source, on the runtime's own path:

| the candidate | the text kept | the vision kept | the INT8 | text cos (mean / min) | vision cos (mean / min) | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **the previous pack** | 0 | 0 | 220 | 0.999806 / 0.999725 | 0.999757 / 0.999589 | 413.2 MB |
| t10_v20 | 10 | 20 | 190 | 0.999872 / 0.999819 | 0.999819 / 0.999764 | 486.5 MB |
| t20_v40 | 20 | 40 | 160 | 0.999913 / 0.999870 | 0.999866 / 0.999833 | 576.1 MB |
| **t30_v60** (the retired checkpoint) | **30** | **60** | **130** | **0.999937 / 0.999898** | **0.999927 / 0.999897** | **646.3 MB** |
| t50_v100 | 50 | 100 | 70 | 0.999974 / 0.999960 | 0.999989 / 0.999983 | 722.3 MB |

**30 text + 60 vision FP16-kept layers were adopted on the published checkpoint**, measured
by the same procedure. The gate is the same: mean >= 0.99 and min >= 0.98 on both towers,
256 items, seed 42. The rebuilt pack measures

| | text mean / min | vision mean / min |
| :--- | :--- | :--- |
| the rebuilt pack (published source) | **0.999803 / 0.999684** | **0.999687 / 0.999367** |

(the 646.3 MB numbers quoted earlier belong to the retired unidentified checkpoint; they are
the record of that pack, superseded by the pack shipped now).

**The previous pack quantized every eligible layer** (220 INT8, one 2D weight
kept). The corrected drift at any threshold protects almost nothing - the
previous pack's whole protection set came from the defective measurement. The
rebuilt pack protects 90 layers and the output is measurably closer to the
FP16 source on **both** towers:

| | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the text mean / min | 0.999806 / 0.999725 | **0.999941 / 0.999888** |
| the vision mean / min | 0.999757 / 0.999589 | **0.999925 / 0.999790** |

---

## 4. The pack

Built with `sdxl/build_long_vitl14_convrot_int8.py`:

```
--impact-text impact/impact_long_vitl14_text_fixed.json \
--impact-vision impact/impact_long_vitl14_vision_fixed.json \
--protect-list impact/longvitl14_protect_t30_v60.json
```

| item | value |
| :--- | :--- |
| the source | zer0int's published `BEST-GmP-smooth-ft` (`8e259dc8...`, 1,711,063,028 bytes, FP32) |
| the pack | **490.6 MB** |
| the SHA256 | `3ED7D2CCB7DE29D6AD5DBE5661746234F8CC3CF579DCDB58D78F71DD8BB1AD89` |
| the tensors | 850 |
| the quantized layers | **130** (the text 44, the vision 84, the projections 2) |
| the FP16-kept layers | 90 (the text 30, the vision 60), re-measured on the published source |
| the group size | 256 (all 130 layers) |
| the output | finite |
| text cos mean / min (256 items, seed 42) | **0.999803 / 0.999684** |
| vision cos mean / min (256 items, seed 42) | **0.999687 / 0.999367** |
| the gate (mean >= 0.99, min >= 0.98, both towers) | **PASS** |
| the identity check | the pack's 167 FP16-kept tensors match the published checkpoint bit for bit (`results/verify_long_vitl14_published.json` holds the acceptance; the retired unidentified checkpoint is recorded in `md/verdicts/LongCLIP_ViT_L14.md`) |

---

## 5. The FP16-kept tensors are stored in FP16

The checkpoint is FP32, and the previous revision of this pack kept the
unquantized tensors in FP32. That is the wrong storage width: **ComfyUI's CLIP
runs in FP16**, so an FP32 kept tensor is cast to FP16 when the model is built
and nothing at runtime sees the extra precision. Storing FP32 only paid for
the bytes.

The kept tensors are stored FP16 and the INT8 scales stay FP32 (they carry the
dequantization precision, and FP16 would round the step size itself). The
packs that already ship this way - CLIP-ViT-bigG-14, CLIP-L, sigclip, T5,
CLIP-SAE-ViT-L-14 - all pass their acceptance.

---

## 6. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_clip.py` | the text tower's impact measurement |
| `sdxl/diag_impact_clip_vision.py` | the vision tower's impact measurement |
| `sdxl/build_long_vitl14_convrot_int8.py` | the builder (`--protect-list`, FP16 kept) |
| `sdxl/verify_long_vitl14.py` | the acceptance, both towers, the runtime's own path |
| `impact/impact_long_vitl14_text_fixed.json` | the text tower's drift (git-ignored) |
| `impact/impact_long_vitl14_vision_fixed.json` | the vision tower's drift (git-ignored) |
| `impact/longvitl14_protect_t30_v60.json` | the adopted FP16-kept list (git-ignored) |
