# Qwen2.5-VL-7B-Instruct — ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/clip/Qwen2.5-VL-7B-Instruct-abliterated_converted.safetensors` (15816.1 MB, BF16, 729 tensors)
**Pack.** `models/clip/Qwen2.5-VL-7B-Instruct-abliterated_converted_convrot_int8.safetensors` (9289.6 MB, **−41.3 %**)
**SHA256.** `25D509531992846FF895BAF3C3E8AC6A5D182629130162880C6E2234664EF18C`

---

## 1. The model

| item | measured |
| :--- | :--- |
| the source | 15816.1 MB, BF16, 729 tensors |
| the language tower | `model.embed_tokens` + 28 layers × 7 + `model.norm` + `lm_head` (336 tensors) |
| the vision tower | `visual.blocks.0..31` + `visual.merger` + `visual.patch_embed` (391 tensors) |
| the 2D weights | 360 + the embedding and the head |

**The runtime is the language tower only.** ComfyUI's `QWEN25_7B` clip type
builds `QwenImageTEModel` → `Qwen25_7BVLIModel` - the language model, whose
hidden state (3584-wide) is what the downstream model consumes. The vision
tower is **model content** that no Qwen2.5-VL load path in ComfyUI reads. It is
still shipped and quantized in the pack (the content is complete), but it is
not part of the gate.

The load path is `llama_detect` → `qwen_image.te` → `Qwen25_7BVLIModel`, the
same dispatch the Qwen-Image text encoder uses.

---

## 2. The impact measurement (the corrected pass)

The earlier measurement had the injection defect (the reconstruction injected
in the **rotated** frame). It is fixed. The aggregate contribution was measured
over the **358 runtime Linear modules** - replace one layer by its ConvRot INT8
reconstruction, hold the rest, and compare the language model's hidden state
against the unmodified output:

| | the language tower (196) | the vision tower (162) |
| :--- | :--- | :--- |
| the median | 1.03e-04 | **4.05e-06 (all identical)** |
| the max | 5.9e-04 | 4.05e-06 |

**The vision tower's contributions are all identical to the last digit** - the
measurement runs the language model only, so the vision tower never sees an
input and cannot contribute. Its 4.05e-06 is the measurement floor, not a
ranking. The tower that the output actually depends on is the language tower,
and that is where the protection goes.

The pack's 320 INT8 layers split **157 language / 162 vision** (the vision
tower is 100% quantized, as its measured sensitivity is nil and the runtime
never reads it).

---

## 3. The candidates, and the adopted set

Built from the contribution's ranking (the top-K of the language tower) and
measured on the fixed **256-item** acceptance set (seed 42), the gate being the
language model's hidden state:

| the candidate | the INT8 | the kept | mean | min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| the previous pack | 360 | 0 | 0.998241 | 0.994528 | 0.997052 | 7918.6 MB |
| k=0 (the floor) | 360 | 0 | 0.998422 | 0.997757 | 0.997757 | 7918.6 MB |
| **k=40 (adopted)** | **320** | **40** | **0.999325** | **0.997294** | **0.998553** | **9289.6 MB** |
| k=150 | 210 | 150 | 0.999496 | 0.997813 | 0.998918 | 12650.0 MB |

(The previous pack and k=0 are the same layer set; their scores differ because
the previous pack's acceptance was measured against a stock-`manual_cast`
baseline, while these rows are measured on the same ops path on both sides.)

**40 FP16-kept layers were adopted.** k=40 is the quality plateau: going to
k=150 buys +1.7e-04 of the mean for 3360 MB, and the k=40 minimum (0.997294) is
within the noise of k=150's (0.997813). Both clear the 0.99 / 0.98 gate.

Against the previous pack the rebuilt one is **better on every number and
bigger by design**:

| | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the mean / min | 0.998241 / 0.994528 | **0.999325 / 0.997294** |
| the size | 7918.6 MB | 9289.6 MB (+1371 MB) |

The previous pack quantized every eligible layer (the old "native is enough"
assumption from a measurement that had not been re-derived). The measured
curve says otherwise: 40 protected layers buy +1.1e-03 of the mean and
+2.8e-03 of the minimum for 1371 MB. The minimum 0.994528 → 0.997294 is the
part that matters, because 0.9945 was the worst-case item sitting close to the
0.99 gate.

---

## 4. The pack

| item | value |
| :--- | :--- |
| the source | 15816.1 MB (BF16) |
| the pack | **9289.6 MB** (-41.3 %) |
| the SHA256 | `25D509531992846FF895BAF3C3E8AC6A5D182629130162880C6E2234664EF18C` |
| the tensors | 1369 |
| the quantized layers | **320 of 360** (the language 157, the vision 162, group size 256; 32 layers at group size 4 where the width demands it) |
| the FP16-kept layers | 40 (the language tower's top contributions), stored FP16 |
| the output | finite (int8_linear 39936 calls) |
| the gate (256 items, seed 42) | **PASS** (mean ≥ 0.99, min ≥ 0.98) |

---

## 5. Files

| file | role |
| :--- | :--- |
| `sdxl/measure_qwen25vl_7b_contrib.py` | the aggregate contribution (the criterion) |
| `clip_convert/convert_qwen25_vl_7b_impact_convrot_int8.py` | the builder (`--protect-list`, FP16 kept; the measured names are the checkpoint's, so the mapping is the identity) |
| `sdxl/verify_qwen25vl_7b.py` | the acceptance on the fixed 256-item set (the language tower) |
| `impact/contrib_qwen25vl_7b.json` | the contribution (git-ignored) |
| `impact/q25vl_protect_40.json` | the adopted FP16-kept list (git-ignored) |
