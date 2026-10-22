# sigclip_vision_patch14_384 — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/clip/sigclip_vision_patch14_384.safetensors` (816.8 MB, 27-layer SigLIP vision tower)
**Pack.** `models/clip/sigclip_vision_patch14_384_hswq_convrot_int8.safetensors` (502.9 MB, **−38.4 %**)
**SHA256.** `C9448A890697ED74674AFBD70919F0CFFF0DD115B9E0D207478B9586C3EB62A5`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate output**.

---

## 1. The model

sigclip_vision_patch14_384 is a **vision-only** SigLIP checkpoint: 27 transformer
layers, hidden size 1152, image size 384, patch size 14.

| item | measured |
| :--- | :--- |
| the runtime Linear modules | **162** (27 layers × 6) |
| the config ComfyUI picks | `comfy/clip_vision_siglip_384.json` (chosen from the checkpoint's own shape) |
| the load path | `ClipVisionModel` with `mixed_precision_ops` |

SigLIP's hidden size is 1152, which is **not** a multiple of 256. The builder
falls back per layer: 1664-wide layers use group size 64, and the rest use the
largest power-of-4 divisor available (16 for some). This is unchanged from the
earlier pack.

---

## 2. The impact measurement, re-measured

The earlier measurement had the injection defect found while working on the
FLUX ControlNet: the reconstruction was injected in the **rotated** frame
(`q * s`, i.e. `W_hat * H`) instead of the **original** one.

| | the old (defective) | the new (corrected) |
| :--- | :--- | :--- |
| the min | 9.76e-01 | 9.76e-07 |
| the median | 2.74e-01 | **1.10e-05** |
| the p90 | 3.75e+00 | 7.32e-05 |
| the max | 3.75e+00 | **1.43e-03** |

At the old threshold (0.30) the corrected drift protects **nothing**. The
shipped pack's 66-layer protection set had come from the defective values.

---

## 3. The selection criterion: the aggregate contribution

The per-layer drift is a **screen**, not the selection criterion. The decision
is the **aggregate output**: hold the weights, replace ONE layer by its ConvRot
INT8 reconstruction, and compare the image embedding against the unmodified
output. The layer's contribution is `1 − cos`.

`1 − cos` over the 162 runtime Linear layers:

| | value |
| :--- | :--- |
| the min | −3.27e-05 |
| the median | −2.77e-05 |
| the p90 | −8.72e-06 |
| the max | **8.22e-04** |

**Almost every layer is negative** — the cosine rounds to 1.0 within float
precision, so the measured `1 − cos` is a negative rounding residual, not a
real contribution. Only the top few layers are measurable:

| the layer | 1 − cos |
| :--- | :--- |
| `encoder.layers.0.mlp.fc2` | 8.22e-04 |
| `encoder.layers.0.mlp.fc1` | 1.54e-04 |
| `encoder.layers.0.self_attn.out_proj` | 8.70e-05 |
| `encoder.layers.1.mlp.fc2` | 6.94e-05 |
| `encoder.layers.1.mlp.fc1` | 2.50e-05 |

**SigLIP-384 is extremely robust to single-layer quantization**: the effect of
quantizing any one layer is at or below the measurement floor, except for the
first two blocks. This is a property of the model, not of the measurement.

---

## 4. The candidates, and the adopted set

Built from the contribution's ranking (the top-K protected) and measured on
the fixed **256-item** acceptance set (seed 42):

| the candidate | the INT8 | the kept | cos mean | cos min | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| the previous pack (the defective drift) | 96 | 66 | 0.998560 | 0.996602 | 633.5 MB |
| k=0 (the floor) | 162 | 0 | 0.984356 | 0.975729 | **425.9 MB** |
| k=10 | 152 | 10 | 0.995935 | 0.989390 | 466.2 MB |
| **k=20 (adopted)** | **142** | **20** | **0.996794** | **0.993237** | **502.9 MB** |
| k=40 | 122 | 40 | 0.997729 | 0.995073 | 569.6 MB |
| k=100 | 62 | 100 | 0.999715 | 0.999209 | 731.7 MB |

**20 FP16-kept layers were adopted.** The knee is between k=0 and k=10 (the
mean moves +1.2e-02 for 40 MB) - the floor's 0.984356 is below the 0.99 gate,
which is exactly the "do not sit at the floor" case. Past k=10 the curve
climbs slowly: k=20 → k=40 buys +9.4e-04 for 67 MB, and k=100 buys +2.9e-03
for 229 MB more, which is not this pack's trade.

Against the previous pack:

| | the previous pack | the rebuilt pack (k=20) |
| :--- | :--- | :--- |
| the cos mean / min | 0.998560 / 0.996602 | 0.996794 / 0.993237 |
| the size | 633.5 MB | **502.9 MB** |

**The rebuilt pack is 130.6 MB smaller, and its cosine is lower.** Both clear
the 0.99 / 0.98 gate. This is a deliberate trade, not a win on both axes, and
it is stated that way:

- The previous pack protected **66 of 162 layers** - a third of the model kept
  in FP16. That came from the defective drift, and it is far more than this
  model's measured sensitivity needs (section 3: only the first two blocks have
  a measurable single-layer contribution).
- The gate is 0.99 / 0.98. k=20 sits at 0.996794 / 0.993237, well above it.
- **k=10 (466.2 MB) is the smallest pack that clears the gate**
  (0.995935 / 0.989390). k=20 buys +8.6e-04 of the mean and +3.8e-03 of the
  minimum for 36.7 MB. k=20 was adopted for that margin; k=10 is the
  defensible minimum.
- **k=100 (731.7 MB, 0.999715 / 0.999209) is the point that beats the previous
  pack's cosine**, at 98 MB *larger* than the previous pack - not adopted,
  because it is bigger than what it replaces and the gate does not require it.

---

## 5. The pack

| item | value |
| :--- | :--- |
| the source | 816.8 MB |
| the pack | **502.9 MB** (-38.4 %) |
| the SHA256 | `C9448A890697ED74674AFBD70919F0CFFF0DD115B9E0D207478B9586C3EB62A5` |
| the tensors | 732 |
| the quantized layers | **142** (group size 64/16) |
| the FP16-kept layers | 20, stored FP16 |
| the output | finite |
| the gate (256 items, seed 42) | **PASS** (mean ≥ 0.99, min ≥ 0.98) |

The kept tensors are FP16 (the runtime runs in FP16; the INT8 scales stay
FP32).

---

## 6. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_clip_vision.py` | the drift measurement (the screen) |
| `sdxl/measure_sigclip384_contrib.py` | the aggregate contribution (the criterion) |
| `clip_convert/convert_clip_impact_convrot_int8.py` | the builder (`--clip_type vision`, `--protect-list`, FP16 kept) |
| `sdxl/verify_vision.py` | the acceptance on the fixed 256-item set |
| `impact/impact_sigclip384_vision_fixed.json` | the corrected drift (git-ignored) |
| `impact/contrib_sigclip384.json` | the contribution (git-ignored) |
| `impact/sig_protect_20.json` | the adopted FP16-kept list (git-ignored) |
