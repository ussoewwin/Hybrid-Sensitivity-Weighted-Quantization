# Qwen3-4B-abliterated — ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/clip/qwen3_4b_abliterated_fp16_converted.safetensors` (7672.3 MB, BF16, 36 layers, hidden 2560, vocab 151936)
**Pack.** `models/clip/qwen3_4b_abliterated_hswq_convrot_int8.safetensors` (4168.3 MB, **−45.7 %**)
**SHA256.** `1A7E13E6F899CD8369A29A04DACD21B1A04605DACF968EC8954357F67B54A431`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate output**.

---

## 0. What changed

The previous pack was produced by a **static kurtosis / outlier-ratio rule**
protecting four layer families:

```
model.embed_tokens.weight                  (the embedding)
model.layers.*.self_attn.k_proj.weight     (all 36)
model.layers.*.self_attn.q_proj.weight     (all 36)
model.layers.*.mlp.down_proj.weight        (all 36)
```

That rule is not a proxy for output sensitivity, and the measurement behind it
predates the injection fix. The pack was rebuilt from scratch.

---

## 1. The model

This is **Z Image's text encoder**. ComfyUI builds it through
`comfy.text_encoders.z_image.te` (not `anima.te`, which is the 0.6B), so the
load path is `llama_detect` → `z_image.te` → `ZImageTEModel` → `Qwen3_4BModel`.

| item | measured |
| :--- | :--- |
| the source | 7672.3 MB, BF16, 397 tensors |
| the 2D weight modules | **253** (36 layers × 7 + the embedding) |
| the runtime Linear modules | **252** (the embedding is not a runtime Linear) |
| the layer families | `self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj` |

---

## 2. The drift measurement (the screen)

`diag_impact_qwen3.py` measures the per-layer drift with the reconstruction
injected in the **original** frame. Over the 252 runtime Linear modules:

| | value |
| :--- | :--- |
| the median | 6.15e-04 |
| the p90 | 2.61e-03 |
| the max | 2.98e-03 |

The drift is a **screen**, not the criterion (it measures the weight error
only).

---

## 3. The aggregate contribution (the criterion)

`measure_qwen3_4b_contrib.py` holds the weights, replaces ONE layer by its
ConvRot INT8 reconstruction, and measures the encoder's hidden state against
the unmodified output. The layer's contribution is `1 − cos`.

`1 − cos` over the 252 runtime Linear layers:

| | value |
| :--- | :--- |
| the min | 6.65e-06 |
| the p10 | 2.39e-05 |
| the median | 6.25e-05 |
| the p90 | 1.05e-04 |
| the max | **6.34e-04** |

**The spread is 95×.** The top is layer 35's MLP (`mlp.gate_proj` 6.34e-04)
and layer 6's `mlp.down_proj` (2.91e-04), then layers 0–3's attention and MLP
blocks.

---

## 4. The candidates, and the adopted set

Built from the contribution's ranking (the top-K protected) and measured on
the fixed **256-item** acceptance set (seed 42):

| the candidate | the INT8 | the kept | cos mean | cos min | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| the previous pack (the kurtosis rule) | 144 | 109 | 0.999198 | 0.997798 | 5515.5 MB |
| k=0 (the floor) | 253 | 0 | 0.999000 | 0.996968 | **3841.3 MB** |
| **k=20 (adopted)** | **233** | **20** | **0.999630** | **0.997430** | **4168.3 MB** |
| k=40 | 213 | 40 | 0.999648 | 0.997050 | 4434.3 MB |
| k=80 | 173 | 80 | 0.999667 | 0.997775 | 4941.1 MB |
| k=150 | 103 | 150 | 0.999698 | 0.998418 | 5909.9 MB |

**20 FP16-kept layers were adopted.** The knee is between k=0 and k=20: the
mean moves +6.3e-04 for 327 MB. Past k=20 the curve is a plateau (k=20 → k=40
buys +1.8e-05 for 266 MB), and k=40 is **dominated** - k=20 is smaller *and*
has a better minimum (0.997430 against 0.997050). k=150 reaches the best
cosine but at 5909.9 MB, larger than the pack it replaces.

Against the previous pack the rebuilt one is **better and smaller**:

| | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the cos mean / min | 0.999198 / 0.997798 | **0.999630 / 0.997430** |
| the size | 5515.5 MB | **4168.3 MB** |

1347 MB smaller with a 2.2× smaller mean error. The **minimum** is 1.2×
larger (0.997430 against 0.997798) - the kurtosis rule happened to protect one
item better - but both sit far above the 0.98 gate, and the mean over the set
is what a text encoder feeds into every generation. The previous pack's 109
kept layers came from the kurtosis rule and were the wrong ones.

---

## 5. The pack

| item | value |
| :--- | :--- |
| the source | 7672.3 MB (BF16) |
| the pack | **4168.3 MB** (-45.7 %) |
| the SHA256 | `1A7E13E6F899CD8369A29A04DACD21B1A04605DACF968EC8954357F67B54A431` |
| the tensors | 864 |
| the quantized layers | **233** (group size 256, all) |
| the FP16-kept layers | 20, stored FP16 |
| the output | finite |
| the gate (256 items, seed 42) | **PASS** (mean ≥ 0.99, min ≥ 0.98) |

The kept tensors are FP16 (the runtime runs in FP16; the INT8 scales stay
FP32).

---

## 6. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_qwen3.py` | the drift measurement (the screen); `--model_type qwen3_4b` selects the `z_image` build path |
| `sdxl/measure_qwen3_4b_contrib.py` | the aggregate contribution (the criterion) |
| `clip_convert/convert_qwen3_4b_convrot_int8.py` | the builder (`--protect-list`, FP16 kept) |
| `sdxl/verify_qwen3_4b.py` | the acceptance on the fixed 256-item set |
| `impact/impact_qwen3_4b_fixed.json` | the drift (git-ignored) |
| `impact/contrib_qwen3_4b.json` | the contribution (git-ignored) |
| `impact/q4b_protect_20.json` | the adopted FP16-kept list (git-ignored) |
