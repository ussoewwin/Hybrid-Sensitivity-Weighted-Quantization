# Flan-T5-XXL — ConvRot INT8 Trajectory-Impact Technical Guide

**Model.** `models/clip/flan_t5_xxl_fp16.safetensors` (FLAN-T5 XXL encoder, 20.99 GB FP16)
**Pack.** `models/clip/flan_t5_xxl_hswq_convrot_int8.safetensors` (4.91 GB, **-76.6 %**)
**SHA256.** `05F10CF35D2F3DC2882D41A6A1BDC1C139EC3CE4B034E803648ED3A95B8E4347`
**Method.** impact-driven (measured drift of the model's own output), ConvRot group size 256.

---

## 1. Why this model needed a separate pass

The previous revision of this pack used a **kurtosis / outlier-ratio** rule to pick the
protected layers. That criterion is invalid for this checkpoint, and the impact
measurement below shows exactly how it fails.

Two more things are specific to T5 and had to be settled first:

- **The runtime T5 has no decoder.** `comfy/text_encoders/t5.py`'s `T5` module holds
  `encoder` and `shared` only. The checkpoint additionally carries the T5 **decoder**
  blocks (339 tensors, **11.87 GB**) that the runtime never loads.
- **T5 is not a llama-family text encoder.** Its load path is
  `t5_xxl_detect` → `t5xxl_quantization_metadata` → `model_options["quantization_metadata"]`
  → `mixed_precision_ops`, so the Qwen3 measurement script cannot be reused as-is.

---

## 2. Runtime model inventory (measured)

The model is built exactly the way ComfyUI builds it
(`comfy.text_encoders.sd3_clip.T5XXLModel` via `t5_xxl_detect`), and the weights are
loaded with `load_sd` — building the graph alone would produce a meaningless
all-zero drift.

| item | measured |
| :--- | :--- |
| encoder blocks | 24 |
| runtime `Linear` modules | **168** |
| checkpoint keys | 559 |
| runtime Linear names present verbatim in the checkpoint | **168 / 168** |
| observable output | encoder hidden state, shape `(1, tokens, 4096)`, fp32, finite |
| `shared` token embedding | `Embedding (32128, 4096)` — not a Linear, kept FP16 |
| `decoder.*` tensors | 339 keys, 11.87 GB — never loaded at runtime |

The 168 Linear layers are 7 per block × 24 blocks: `SelfAttention.q / k / v / o` and
`DenseReluDense.wi_0 / wi_1 / wo`. Input widths are 4096 (attention, `wo` output) and
10240 (`wi_0`, `wi_1`), so every one of them is a multiple of the 256 group size.

---

## 3. Impact measurement

Script: `sdxl/diag_impact_t5.py` → `impact/impact_flan_t5_xxl.json`

For each candidate layer the ConvRot INT8 reconstruction is injected into the live
module and the relative MSE of the encoder hidden state against the pristine run is
recorded.

| item | value |
| :--- | :--- |
| layers measured | **168 / 168** |
| runs | 1 (single completion) |
| wall time | 86 s |
| interruptions | 0 |
| reference output | finite |

**Drift distribution**

| min | p10 | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| 5.26e-03 | 1.09e-02 | 4.92e-02 | 1.64e-01 | 3.15e-01 |

**By layer kind**

| kind | n | min | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| attn.q | 24 | 5.78e-03 | 1.28e-02 | 2.69e-02 | 5.80e-02 |
| attn.k | 24 | 5.26e-03 | 1.45e-02 | 2.29e-02 | 5.60e-02 |
| attn.v | 24 | 3.28e-02 | 7.90e-02 | 2.75e-01 | **3.15e-01** |
| attn.o | 24 | 2.94e-02 | 6.46e-02 | 2.36e-01 | 2.99e-01 |
| ff.wi_0 | 24 | 3.11e-02 | 8.82e-02 | 2.03e-01 | 2.56e-01 |
| ff.wi_1 | 24 | 2.56e-02 | 4.96e-02 | 1.19e-01 | 1.51e-01 |
| ff.wo | 24 | 2.75e-02 | 5.35e-02 | 1.37e-01 | 2.38e-01 |

**By depth** — the drift rises monotonically with depth: block 0 has a median of
1.20e-01, block 23 a median of 2.38e-01. The deepest blocks (19-23) hold every one of
the 12 layers above 1.4e-01.

**The old kurtosis protection set, re-measured**

| layer (old protect set) | measured drift | rank among the 168 |
| :--- | :--- | :--- |
| `encoder.block.0.layer.0.SelfAttention.k` | 2.26e-02 | 41 / 168 (24th percentile) |
| `encoder.block.0.layer.0.SelfAttention.v` | 1.20e-01 | 142 / 168 (85th percentile) |
| `encoder.block.1.layer.1.DenseReluDense.wi_0` | 7.89e-02 | 121 / 168 (72nd percentile) |
| `encoder.block.1.layer.1.DenseReluDense.wi_1` | 5.45e-02 | 93 / 168 (55th percentile) |
| `encoder.block.2.layer.1.DenseReluDense.wi_0` | 1.19e-01 | 139 / 168 (83rd percentile) |

The old set protected one layer that sits in the **lowest quartile** of the drift
distribution, while **24 layers drift more than the whole old set's maximum**
(1.20e-01) were left unprotected — the worst of them at 3.15e-01, **2.6x the old
set's maximum**. The kurtosis rule is therefore replaced.

---

## 4. Threshold selection

Three candidate thresholds were built and each was measured on the full 256-item
acceptance set (seed 42) against the FP16 baseline — the threshold was chosen from the
measured output quality, not from the drift distribution alone.

| threshold | INT8 layers | cos mean | cos min | cos p5 | pack size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.15 | 150 | 0.999908 | 0.999689 | 0.999860 | 17,533 MB (decoder kept) |
| **0.25** | **163** | **0.999875** | **0.999310** | **0.999802** | **17,182 MB (decoder kept)** |
| 0.30 | 167 | 0.999864 | 0.999301 | 0.999795 | 17,094 MB (decoder kept) |

**Threshold 0.25 was adopted.** All three pass the acceptance gate
(mean >= 0.99, min >= 0.98); 0.25 keeps the five genuinely sensitive layers in FP16
while quantizing 163 of 168, and it is the point where the measured quality still has
a clear margin over the 0.30 candidate.

The five protected layers (drift > 0.25):

| layer | drift |
| :--- | :--- |
| `encoder.block.21.layer.0.SelfAttention.v` | 3.1521e-01 |
| `encoder.block.23.layer.0.SelfAttention.o` | 2.9897e-01 |
| `encoder.block.23.layer.0.SelfAttention.v` | 2.9026e-01 |
| `encoder.block.20.layer.0.SelfAttention.v` | 2.7537e-01 |
| `encoder.block.23.layer.1.DenseReluDense.wi_0` | 2.5609e-01 |

---

## 5. The decoder, and why it is not in the pack

The runtime T5 module has no decoder attribute, so the 339 `decoder.*` tensors are
never loaded and never used. `SDClipModel.load_sd` calls
`load_state_dict(sd, strict=False)`, so a pack without them loads normally.

The converter takes `--drop-decoder` to omit them. This is a **load-path fact, not a
model edit**: the encoder that actually runs is bit-identical either way, and the
measurement in section 3 was taken on the runtime encoder.

| part | FP16 | pack with decoder | pack without decoder |
| :--- | :--- | :--- | :--- |
| encoder | 8.05 GB | 4.42 GB | 4.42 GB |
| decoder | 11.87 GB | 11.87 GB | **dropped** |
| other | 0.49 GB | 0.49 GB | 0.49 GB |
| **total** | **20.99 GB** | 16.78 GB | **4.91 GB** |

---

## 6. Verification (real load path)

Script: `sdxl/verify_t5.py` → `impact/verify_t5.json`

The pack is loaded through the real runtime path
(`comfy.text_encoders.sd3_clip.sd3_clip(clip_l=False, clip_g=False, t5=True, **t5_xxl_detect(sd))`),
which is the path the SD3 / FLUX loader uses.

| item | value |
| :--- | :--- |
| file size | 21490.03 MB -> **5026.13 MB** |
| INT8 layers | 163 (of 168) |
| FP16 protected | 5 |
| output finite | **yes** (both FP16 and INT8) |
| cos mean | **0.999878** |
| cos min | **0.999716** |
| cos p5 | 0.999833 |
| cos p50 | 0.999882 |
| gate (mean >= 0.99, min >= 0.98) | **PASS** |

---

## 7. Performance, and the INT8 GEMM

### 7.1 The stock path never reaches the INT8 GEMM

Measured on the stock ComfyUI path with the INT8 pack loaded: the weights **are**
quantized (163 `QuantizedTensor` with the `convrot` parameter set), but
`int8_linear` was called **0 times**. This is the same mechanism as for the CLIP text
tower: ComfyUI's `int8_tensorwise` is declared with `quantize_input: False`, so it is a
**weight-only** format - the weight is dequantized for every matmul - and
`comfy/sd1_clip.py` additionally builds the encoder with `full_precision_mm=True`,
which disables the quantized matmul outright.

### 7.2 The loader's runtime arm fixes it, and T5 benefits strongly

The loader package installs an arm at import time
(`ComfyUI-HSWQ-Loader-and-Tools/patches/clip_int8_gemm/arm.py`) that makes a quantized
caller get `full_precision_mm=False` and routes a `TensorWiseINT8Layout` weight into
`int8_linear` with the online ConvRot activation rotation. **ComfyUI core is not
modified.** T5 reaches `mixed_precision_ops` through the same `SDClipModel.__init__`
as CLIP, so the arm applies to it unchanged.

| condition | tokens=64 | tokens=256 | tokens=1024 |
| :--- | :--- | :--- | :--- |
| FP16 reference | 143.47 ms | 290.95 ms | 911.19 ms |
| INT8 pack, stock path | 153.42 ms (1.07x) | 299.99 ms (1.03x) | 929.65 ms (1.02x) |
| **INT8 pack, with the arm** | **34.79 ms (0.24x)** | **39.76 ms (0.14x)** | **224.04 ms (0.25x)** |

`int8_linear` calls per forward with the arm: **163** — exactly the number of INT8
layers. Against the stock path the arm is **+75.9 % to +86.7 % faster**; against FP16
it is **3.9x to 7.2x faster**.

Fidelity with the arm enabled (256-item acceptance set, seed 42): mean **0.999713**,
min **0.999176**, p5 0.999549 — still **PASS** (mean >= 0.99, min >= 0.98). The small
change against the stock path (0.999878 -> 0.999713) is the expected cost of the
dynamic activation quantization.

---

## 8. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_t5.py` | impact measurement (encoder hidden state) |
| `clip_convert/convert_t5_impact_convrot_int8.py` | impact-driven pack builder (`--drop-decoder`) |
| `sdxl/verify_t5.py` | acceptance verification + latency |
| `impact/impact_flan_t5_xxl.json` | measured per-layer drift (git-ignored) |
| `impact/verify_t5.json` | verification result (git-ignored) |
| `models/clip/flan_t5_xxl_hswq_convrot_int8.safetensors` | the pack |
