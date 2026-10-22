# Josiefied-Qwen3-0.6B ConvRot INT8 — Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Target**: `Josiefied-Qwen3-0.6B-abliterated-v1.safetensors` — a Qwen3-0.6B text
encoder (llama-family architecture, 28 layers, hidden 1024, vocab 151936),
stored in BF16.

**What this document covers**: how it is quantized to ConvRot INT8 with a
**measured** FP16 protection set chosen on the **aggregate contribution**, the
pack layout, the load path, and the measured fidelity / size.

- Source: `models/clip/Josiefied-Qwen3-0.6B-abliterated-v1.safetensors`
  (1136.91 MB, SHA256 `538D7E689B6A57AED42139298D88A3876634E2DD583A3F6B5C381BA60C640E67`)
- Pack: `models/clip/Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors`
  (655.19 MB, **−42.4 %**, SHA256 `40BB0C7F8862EE3AC4A5F345120CE5F23E1DFF15FFF020D57F00FAE8A9736D9C`)

---

## 0. What changed

The previous pack was produced by a **static kurtosis / outlier-ratio rule**
protecting three layer families:

```
model.layers.*.self_attn.k_proj.weight     (all 28)
model.layers.*.self_attn.q_proj.weight     (all 28)
model.layers.*.mlp.down_proj.weight        (all 28)
model.embed_tokens.weight                  (the embedding)
```

That rule is not a proxy for output sensitivity, and the kurtosis values behind
it were computed through a broken safetensors header offset, so the protection
basis was invalid twice over. The previous guide also concluded that
**"selective protection cannot help"** because the drift distribution looked
flat — that conclusion was drawn from the **defective** drift, and it is wrong
(section 3).

---

## 1. The model

| item | measured |
| :--- | :--- |
| 2D weight modules | **197** (text tower only) |
| the runtime Linear modules | 196 |
| the source dtype | BF16 (all 310 tensors) |
| the load path | `llama_detect` → `comfy.text_encoders.anima.te` |

`model.embed_tokens.weight` (151936 × 1024) is a 2D Linear-shaped tensor and is
**not** a runtime Linear; it is handled as an embedding (see section 4).

---

## 2. The impact measurement, re-measured

The earlier measurement had the injection defect found while working on the
FLUX ControlNet: the reconstruction was injected in the **rotated** frame
(`q * s`, i.e. `W_hat * H`) instead of the **original** one. The pack stores
the rotated weight and the kernel rotates the **activation**, so the
quantization error that reaches the output is `(q*s) H^T − W`.

| | the old (defective) | the new (corrected) |
| :--- | :--- | :--- |
| the min | 7.52e-01 | 3.43e-04 |
| the median | 3.05e+00 | **2.35e-03** |
| the p90 | 3.24e+00 | 6.10e-03 |
| the max | 3.43e+00 | **6.42e-03** |

The old values were all clustered around 3.0–3.4 — the number was the rotation
itself, not the quantization error, so the distribution looked flat and the
old guide concluded that selective protection could not help. The corrected
distribution has a **19× spread over the whole set** (and 241× over the
contribution ranking below), and the ranking is usable.

---

## 3. The selection criterion: the aggregate contribution

The per-layer drift is a **screen**, not the selection criterion. It measures
the **weight** error only, while the pack's real path also quantizes the
**activation** for that layer, and that is where most of the error comes from.

The selection is therefore made on the **aggregate output**: build the encoder
the way ComfyUI builds it, hold the weights, replace ONE layer by its ConvRot
INT8 reconstruction, and compare the encoder's hidden state against the
unmodified output. The layer's **contribution** is `1 − cos`.

`1 − cos` over the 196 runtime Linear layers:

| | value |
| :--- | :--- |
| the min | 4.08e-06 |
| the p10 | 2.26e-05 |
| the median | 8.10e-05 |
| the p90 | 1.40e-04 |
| the max | **9.82e-04** |

**The spread is 241×.** The top of the ranking is dominated by layer 27's MLP
(`up_proj` 9.82e-04, `gate_proj` 4.14e-04) and by layer 0–2's attention and MLP
blocks; the smallest are the late layers' attention projections
(`layers.27.self_attn.k_proj` 4.08e-06). Protecting the top of this ranking is
what the size buys.

---

## 4. The candidates, and the adopted set

The packs were built from the contribution's ranking (the top-K protected) and
measured on the fixed **256-item** acceptance set (seed 42):

| the candidate | the INT8 | the kept | cos mean | cos min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **the previous pack** | 197 | 0 | 0.997572 | 0.992079 | 0.996506 | 570.5 MB |
| all-INT8 (k=0) | 197 | 0 | 0.997720 | 0.993213 | 0.996827 | 0.56 GB |
| k=10 | 187 | 10 | 0.999410 | 0.998609 | 0.999125 | 0.58 GB |
| k=20 | 177 | 20 | 0.999455 | 0.998794 | 0.999182 | 0.60 GB |
| **k=40 (adopted)** | **157** | **40** | **0.999502** | **0.998980** | **0.999286** | **655.2 MB** |
| k=80 | 117 | 80 | 0.999550 | 0.998821 | 0.999349 | 0.72 GB |
| k=140 | 57 | 140 | 0.999644 | 0.999016 | 0.999494 | 0.85 GB |

**40 FP16-kept layers were adopted.** k=80 and k=140 give a marginally better
mean but their **min** is *not* monotone (k=80's min 0.998821 is below k=40's
0.998980), so k=40 is where the quality per byte stops improving cleanly. Note
the **huge** jump from k=0 to k=10: protecting just the ten worst layers moves
the min from 0.993213 to 0.998609, which is exactly the selective protection
the old guide said could not help.

### 4.1 The kept tensors are stored in FP16

The source is BF16 and ComfyUI's text encoder runs in FP16, so a kept tensor is
cast either way when the model is built. The kept tensors are stored FP16 and
the INT8 scales stay FP32 (they carry the dequantization precision).

---

## 5. The pack

| item | value |
| :--- | :--- |
| the source | 1136.91 MB (BF16) |
| the pack | **655.19 MB** (-42.4 %) |
| the SHA256 | `40BB0C7F8862EE3AC4A5F345120CE5F23E1DFF15FFF020D57F00FAE8A9736D9C` |
| the tensors | 624 |
| the quantized layers | **157** (group size 256, all) |
| the FP16-kept layers | 40 |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42) | **0.999502 / 0.998980 / 0.999286** |
| the gate (mean ≥ 0.99, min ≥ 0.98) | **PASS** |

Against the previous pack (both on the same 256-item set, seed 42):

| | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the cos mean / min | 0.997572 / 0.992079 | **0.999502 / 0.998980** |
| the size | 570.5 MB | 655.2 MB |

The rebuilt pack is 85 MB larger for a **4.9× smaller mean error** and a
**7.8× smaller worst-case error**. The previous pack's protection set came
from the invalid kurtosis rule and protected the wrong layers — most notably it
protected every `k_proj` / `q_proj` / `down_proj`, while the contribution's top
is layer 27's `mlp.up_proj` / `mlp.gate_proj`.

---

## 6. Loading the pack

The pack is loaded through the text-encoder path the model is built with:

```python
sd = comfy.utils.load_torch_file(pack, safe_load=True)
detect = comfy.text_encoders.hunyuan_video.llama_detect(sd)
te = comfy.text_encoders.anima.te(**detect)
clip = te(device=device)
clip.load_sd(sd)
```

The `comfy_quant` stamps are consumed by the ops class the encoder is built
with; the `_quantization_metadata` in the pack's header carries the per-layer
`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`.

---

## 7. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_qwen3.py` | the drift measurement (the screen) |
| `sdxl/measure_josiefied_qwen3_06b_contrib.py` | the aggregate contribution (the criterion) |
| `clip_convert/convert_josiefied_qwen3_06b_convrot_int8.py` | the builder (`--protect-list`, FP16 kept) |
| `sdxl/verify_josiefied_qwen3_06b.py` | the acceptance on the fixed 256-item set |
| `impact/impact_josiefied_qwen3_06b_fixed.json` | the corrected drift (git-ignored) |
| `impact/contrib_josiefied_qwen3_06b.json` | the contribution (git-ignored) |
| `impact/jq_protect_40.json` | the adopted FP16-kept list (git-ignored) |
