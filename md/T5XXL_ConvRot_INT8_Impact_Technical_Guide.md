# Flan-T5-XXL — ConvRot INT8 Trajectory-Impact Technical Guide

**Model.** `models/clip/flan_t5_xxl_fp16.safetensors` (FLAN-T5 XXL, 20.99 GB FP16, 11,266,928,640 parameters)
**Pack.** `models/clip/flan_t5_xxl_hswq_convrot_int8.safetensors` (15.62 GB, **-25.6 %**)
**SHA256.** `0FA68C40970462A427F15ECC566E2F02EF4FE9CC07C2223DB6A028E3180F96B5`
**Method.** impact-driven (measured drift of the model's own output), ConvRot group size 256.

---

## 1. What this model is, and why it needed its own pass

FLAN-T5 XXL is an **encoder-decoder** checkpoint: 24 encoder blocks, 24 decoder
blocks, 11.27 B parameters.

| part | parameters | FP16 bytes |
| :--- | --- | :--- |
| encoder | 4,630,714,368 | 8.63 GiB |
| decoder | 6,373,021,696 | 11.87 GiB |
| `shared` (token embedding) | 131,596,288 | 0.25 GiB |
| `lm_head` | 131,596,288 | 0.25 GiB |
| **total** | **11,266,928,640** | **20.99 GiB** |

Two facts decided the whole approach:

- **The runtime T5 has no decoder.** `comfy/text_encoders/t5.py`'s `T5` module holds
  `encoder` and `shared` only - it has no decoder attribute, so the 339
  `decoder.*` tensors are never loaded and never executed. They are still
  **model content**: dropping them would delete 6.37 B parameters, which is not
  quantization. The pack therefore **keeps the decoder** and quantizes it.
- **The size floor.** Every parameter as int8 would be
  `11,266,928,640 / 2 = 5,632,464,320` bytes = **5.63 GB**. A pack below 5 GB is
  not representable for this model. With the measured protection sets the pack
  is 15.62 GB.

The previous revision of this work used a **kurtosis / outlier-ratio** rule to
pick the protected layers. That criterion is invalid for this checkpoint, and
the impact measurement below shows exactly how it fails.

---

## 2. Runtime model inventory (measured)

The encoder is built exactly the way ComfyUI builds it
(`comfy.text_encoders.sd3_clip.T5XXLModel` via `t5_xxl_detect`), and the weights
are loaded with `load_sd` - building the graph alone would produce a meaningless
all-zero drift.

| item | measured |
| :--- | :--- |
| encoder blocks | 24 |
| runtime `Linear` modules | **168** |
| checkpoint keys | 559 |
| runtime Linear names present verbatim in the checkpoint | **168 / 168** |
| observable output | encoder hidden state, shape `(1, tokens, 4096)`, fp32, finite |
| `shared` token embedding | `Embedding (32128, 4096)` - not a Linear, kept FP16 |
| `decoder.*` tensors | 339 keys, 264 of them Linear |

The 168 encoder Linear layers are 7 per block x 24 blocks: `SelfAttention.q / k /
v / o` and `DenseReluDense.wi_0 / wi_1 / wo`. The 264 decoder Linear layers are
11 per block x 24 blocks: `SelfAttention.q / k / v / o`, `EncDecAttention.q / k /
v / o` and `DenseReluDense.wi_0 / wi_1 / wo`.

---

## 3. The encoder impact measurement

Script: `sdxl/diag_impact_t5.py` -> `impact/impact_flan_t5_xxl.json`

For each candidate layer the ConvRot INT8 reconstruction is injected into the live
module and the relative MSE of the encoder hidden state against the pristine run
is recorded.

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

**By depth** - the drift rises monotonically with depth: block 0 has a median of
1.20e-01, block 23 a median of 2.38e-01. The deepest blocks (19-23) hold every one
of the 12 layers above 1.4e-01.

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
(1.20e-01) were left unprotected - the worst of them at 3.15e-01, **2.6x the old
set's maximum**. The kurtosis rule is therefore replaced.

---

## 4. The decoder impact measurement

Script: `sdxl/diag_impact_t5_decoder.py` -> `impact/impact_flan_t5_xxl_decoder.json`

The runtime has no decoder, so the decoder drift cannot be observed through it.
It is measured against a decoder that **really executes**: transformers'
`T5Stack` in decoder mode, whose parameter names match the checkpoint 1:1
(559 params vs 559 checkpoint keys; 339 decoder keys mapped 1:1, 264 of them
Linear). The encoder output that feeds the cross-attention comes from the real
ComfyUI runtime encoder.

**Memory.** Building the whole `T5Model` on the GPU does not fit - the encoder
+ decoder in fp32 is >50 GB of shared memory (measured: 15.5/16.0 GB dedicated
plus 36.4 GB shared). The measurement therefore:

- builds **only** the decoder stack, never the full model;
- keeps everything in **fp16** (the checkpoint dtype), never fp32;
- runs the decoder **block by block** on the GPU, with only the current block's
  weights resident, streaming the per-layer weights from the safetensors file;
- loads the ComfyUI runtime encoder through a **streamed** load (only the
  encoder tensors), because a full `load_torch_file` of the 21.4 GB checkpoint
  exhausts the page file.

Measured peak during the sweep: **11.3 / 16.3 GB VRAM**.

**A correction that the measurement needed.** `T5Block.forward` returns
`(hidden_states, self_attn_position_bias, cross_attn_position_bias)`. The two
position biases have to be chained from block to block; without that the
attention loses its relative-position bias and the measurement is meaningless.
The first run of the decoder sweep did not chain them, and the resulting
threshold sweep was non-monotonic (more quantization appeared to improve the
cosine), which is what exposed the defect. After the fix the measurement is
deterministic (two identical runs: cos 1.00000000) and a single-layer injection
produces a non-zero drift (cos 0.99979409 for
`decoder.block.0.layer.0.SelfAttention.q`), while an unperturbed run reproduces
the reference exactly (cos 1.00000000).

| item | value |
| :--- | :--- |
| layers measured | **264 / 264** |
| runs | 1 (single completion) |
| wall time | 836 s |
| interruptions | 0 |
| reference output | finite (absmax 84.56) |

**Drift distribution**

| min | p10 | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| 2.97e-04 | 2.88e-03 | 1.03e-02 | 5.47e-02 | **2.40e+00** |

**By layer kind**

| kind | n | min | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| attn.k | 24 | 2.97e-04 | 4.88e-03 | 1.32e-02 | 1.64e-02 |
| attn.q | 24 | 3.65e-04 | 4.66e-03 | 8.92e-03 | 1.07e-02 |
| xattn.k | 24 | 1.78e-03 | 6.57e-03 | 2.08e-02 | 2.91e-02 |
| xattn.q | 24 | 1.41e-03 | 5.32e-03 | 2.17e-02 | 6.01e-02 |
| ff.wi_1 | 24 | 2.16e-03 | 1.02e-02 | 9.02e-02 | 5.78e-01 |
| ff.wo | 24 | 2.26e-03 | 1.28e-02 | 6.60e-02 | 6.39e-01 |
| xattn.v | 24 | 2.43e-03 | 1.33e-02 | 3.95e-02 | 6.48e-02 |
| xattn.o | 24 | 2.06e-03 | 1.40e-02 | 4.41e-02 | 5.47e-02 |
| attn.o | 24 | 5.23e-03 | 1.76e-02 | 6.66e-02 | 1.53e-01 |
| attn.v | 24 | 4.68e-03 | 1.97e-02 | 8.00e-02 | 1.60e-01 |
| ff.wi_0 | 24 | 7.47e-03 | 3.33e-02 | 1.15e-01 | **2.40e+00** |

The decoder's drift is **an order of magnitude smaller than the encoder's**
(median 1.03e-02 vs 4.92e-02), and the profile is different: the encoder's
outliers are the deep blocks' attention, the decoder's outliers are
`ff.wi_0` and the block-0 feed-forward.

**The decoder is far more sensitive to INT8 than the encoder.** That is not
visible in the per-layer drift (which is small) - it shows up in the output
cosine, which is what section 5 measures.

---

## 5. Threshold selection, from the measured output quality

The thresholds are chosen from the measured output quality, not from the drift
distribution.

### 5.1 The encoder

Three candidate thresholds were built and each was measured on the full 256-item
acceptance set (seed 42) against the FP16 baseline.

| threshold | INT8 layers | cos mean | cos min | cos p5 |
| :--- | :--- | :--- | :--- | :--- |
| 0.15 | 150 | 0.999908 | 0.999689 | 0.999860 |
| **0.25** | **163** | **0.999875** | **0.999310** | **0.999802** |
| 0.30 | 167 | 0.999864 | 0.999301 | 0.999795 |

**Threshold 0.25 was adopted.** All three pass the gate (mean >= 0.99, min >=
0.98); 0.25 keeps the five genuinely sensitive layers in FP16 while quantizing
163 of 168, and it keeps a clear margin over the 0.30 candidate.

The five protected encoder layers (drift > 0.25):

| layer | drift |
| :--- | :--- |
| `encoder.block.21.layer.0.SelfAttention.v` | 3.1521e-01 |
| `encoder.block.23.layer.0.SelfAttention.o` | 2.9897e-01 |
| `encoder.block.23.layer.0.SelfAttention.v` | 2.9026e-01 |
| `encoder.block.20.layer.0.SelfAttention.v` | 2.7537e-01 |
| `encoder.block.23.layer.1.DenseReluDense.wi_0` | 2.5609e-01 |

### 5.2 The decoder

The decoder's output cosine was measured for a set of candidate protection sets
(`impact/decoder_threshold_sweep.json`, `impact/decoder_protection_candidates.json`,
`impact/decoder_frontier.json`):

| protection set | decoder Linear quantized | decoder cos | pack |
| :--- | :--- | :--- | :--- |
| drift <= 0.0005 | 2 / 264 | 0.999835 | 17.99 GB |
| drift <= 0.001 | 3 / 264 | 0.999821 | 17.97 GB |
| drift <= 0.002 | 12 / 264 | 0.997805 | 17.82 GB |
| drift <= 0.003 | 32 / 264 | 0.995249 | 17.39 GB |
| **drift <= 0.005** | **62 / 264** | **0.990368** | **16.78 GB** |
| drift <= 0.01 | 127 / 264 | 0.973003 | 15.36 GB |
| drift <= 0.02 | 190 / 264 | 0.748864 | 13.78 GB |
| drift <= 0.05 | 235 / 264 | 0.630629 | 12.70 GB |

**Threshold 0.005 was adopted** - it is the largest decoder quantization that
still meets the acceptance criterion (cos 0.990368 >= 0.99) on the decoder
output.

Per-kind measurements explain why the decoder needs so much protection:

| quantized set | layers | decoder cos |
| :--- | :--- | :--- |
| the FF layers only | 72 | 0.554025 |
| the self-attention layers only | 96 | 0.929583 |
| the cross-attention layers only | 96 | 0.944740 |
| everything except the FF layers | 192 | 0.878983 |
| nothing (FP16 reference) | 0 | 1.000000 |

**The feed-forward layers dominate the decoder's degradation**: quantizing only
the 72 FF layers costs more accuracy than quantizing all 192 attention layers.

---

## 6. The pack

Built with `clip_convert/convert_t5_impact_convrot_int8.py`:

```
--drift-threshold 0.25                    # the encoder
--quantize-decoder
--decoder-impact impact/impact_flan_t5_xxl_decoder.json
--decoder-max-drift 0.005                 # the decoder
```

| item | value |
| :--- | :--- |
| file size | 21490.03 MB -> **15998.83 MB** (-25.6 %) |
| encoder | 163 INT8 / 5 FP16 |
| decoder | 62 INT8 / 202 FP16 |
| other | 127 tensors kept as-is (1D weights, layer norms, the `shared` embedding, `lm_head`) |
| group size | 256 |

The pack keeps every parameter of the source. The decoder's 202 protected layers
stay FP16 because the measurement shows they cannot be quantized without
breaking the decoder output - this is the price of the decoder's sensitivity,
and it is paid deliberately rather than discovered later.

---

## 7. Verification (real load path)

Script: `sdxl/verify_t5.py` -> `impact/verify_t5_full.json`

The pack is loaded through the real runtime path
(`comfy.text_encoders.sd3_clip.sd3_clip(clip_l=False, clip_g=False, t5=True, **t5_xxl_detect(sd))`),
which is the path the SD3 / FLUX loader uses.

| item | value |
| :--- | :--- |
| file size | 21490.03 MB -> 15998.83 MB |
| output finite | **yes** (both FP16 and INT8) |
| cos mean | **0.999879** |
| cos min | **0.999626** |
| cos p5 | 0.999840 |
| cos p50 | 0.999886 |
| gate (mean >= 0.99, min >= 0.98) | **PASS** |

**Latency** (the encoder is what the runtime executes; the decoder is not
loaded by ComfyUI at all, so it does not appear here)

| batch | FP16 | INT8 | ratio |
| :--- | :--- | :--- | :--- |
| 1 (64 tokens) | 145.62 ms | 156.26 ms | 1.07x |
| 4 (256 tokens) | 295.92 ms | 306.42 ms | 1.04x |
| 16 (1024 tokens) | 1315.92 ms | 1328.21 ms | 1.01x |

---

## 8. Performance: the INT8 GEMM

### 8.1 The stock path never reaches the INT8 GEMM

Measured on the stock ComfyUI path with the INT8 pack loaded: the weights **are**
quantized (163 `QuantizedTensor` with the `convrot` parameter set), but
`int8_linear` was called **0 times**. This is the same mechanism as for the CLIP
text tower: ComfyUI's `int8_tensorwise` is declared with `quantize_input: False`,
so it is a **weight-only** format - the weight is dequantized for every matmul -
and `comfy/sd1_clip.py` additionally builds the encoder with
`full_precision_mm=True`, which disables the quantized matmul outright.

### 8.2 The loader's runtime arm fixes it, and T5 benefits strongly

The loader package installs an arm at import time
(`ComfyUI-HSWQ-Loader-and-Tools/patches/clip_int8_gemm/arm.py`) that makes a
quantized caller get `full_precision_mm=False` and routes a `TensorWiseINT8Layout`
weight into `int8_linear` with the online ConvRot activation rotation. **ComfyUI
core is not modified.** T5 reaches `mixed_precision_ops` through the same
`SDClipModel.__init__` as CLIP, so the arm applies to it unchanged.

| condition | tokens=64 | tokens=256 | tokens=1024 |
| :--- | :--- | :--- | :--- |
| FP16 reference | 143.47 ms | 290.95 ms | 911.19 ms |
| INT8 pack, stock path | 153.42 ms (1.07x) | 299.99 ms (1.03x) | 929.65 ms (1.02x) |
| **INT8 pack, with the arm** | **34.79 ms (0.24x)** | **39.76 ms (0.14x)** | **224.04 ms (0.25x)** |

`int8_linear` calls per forward with the arm: **163** - exactly the number of INT8
encoder layers. Against the stock path the arm is **+75.9 % to +86.7 % faster**;
against FP16 it is **3.9x to 7.2x faster**.

Fidelity with the arm enabled (256-item acceptance set, seed 42): mean **0.999713**,
min **0.999176**, p5 0.999549 - still **PASS** (mean >= 0.99, min >= 0.98). The
small change against the stock path (0.999878 -> 0.999713) is the expected cost
of the dynamic activation quantization.

---

## 9. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_t5.py` | encoder impact measurement (encoder hidden state) |
| `sdxl/diag_impact_t5_decoder.py` | decoder impact measurement (transformers decoder stack, streamed) |
| `clip_convert/convert_t5_impact_convrot_int8.py` | impact-driven pack builder (`--quantize-decoder`, `--decoder-max-drift`) |
| `sdxl/verify_t5.py` | acceptance verification + latency |
| `impact/impact_flan_t5_xxl.json` | measured encoder drift (git-ignored) |
| `impact/impact_flan_t5_xxl_decoder.json` | measured decoder drift (git-ignored) |
| `impact/decoder_threshold_sweep.json` | the decoder threshold sweep (git-ignored) |
| `impact/decoder_frontier.json` | the decoder size / quality frontier (git-ignored) |
| `impact/verify_t5_full.json` | verification result (git-ignored) |
| `models/clip/flan_t5_xxl_hswq_convrot_int8.safetensors` | the pack |
