# FLAN-T5 XXL — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Source.** `models/clip/flan_t5_xxl_fp16.safetensors` (21490.0 MB, 559 keys, 11,266,928,640 params)
**Pack.** `models/clip/flan_t5_xxl_hswq_convrot_int8.safetensors` (14056.6 MB, **−34.5 %**)
**SHA256.** `DE3072A97C710EF2B1C661C3EEA0FBAA5759A0454007163228AF6D489ADE6BB0`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate** output.

---

## 1. The model

FLAN-T5 XXL is an **encoder-decoder** checkpoint: 24 encoder blocks, 24 decoder
blocks, and the `shared` embedding. ComfyUI's runtime
(`comfy/text_encoders/t5.py`) has only `encoder` and `shared` — **it has no
decoder** — so the decoder weights are never loaded by the runtime, but they
are still part of the model content and are shipped (and quantized) so the
checkpoint stays complete.

| tower | the runtime Linear modules | the observable |
| :--- | :--- | :--- |
| the encoder | **168** | the encoder's hidden state for a fixed token batch |
| the decoder | 264 (measured separately) | the decoder's output, with the position biases chained |

The two towers are measured by different scripts, because the runtime has no
decoder: `sdxl/diag_impact_t5.py` (the encoder, through the real T5 load
path) and `sdxl/diag_impact_t5_decoder.py` (the decoder, against a decoder
that really executes).

### 1.1 The decoder's position biases must be chained

`T5Block.forward` returns `(hidden_states, self_attn_position_bias,
cross_attn_position_bias)`. Both biases have to be **chained** from block to
block; without that the measurement is not monotonic and the threshold is
meaningless. The decoder script does chain them.

---

## 2. The impact measurement, re-measured

The earlier measurement had the injection defect that was found while working
on the FLUX ControlNet: the reconstruction was injected in the **rotated**
frame (`q * s`, i.e. `W_hat * H`) instead of the **original** one. The pack
stores the rotated weight and the kernel rotates the **activation**, so the
quantization error that reaches the output is `(q*s) H^T - W`.

| | the old (defective) | the new (corrected) |
| :--- | :--- | :--- |
| the encoder's median | 4.92e-02 | **8.74e-07** |
| the encoder's max | 3.15e-01 | **1.18e-05** |
| the decoder's median | 1.03e-02 | **6.26e-06** |
| the decoder's max | 2.40e+00 | **6.17e-05** |

The old measurement made the encoder's `layers.0.mlp.fc1` look like 1.86 and
the decoder's worst layer like 2.40. The corrected values are five orders of
magnitude smaller.

**The structure the old measurement found still holds**: the decoder's
feed-forward layers dominate.

| the decoder's layer kind | the largest drift |
| :--- | :--- |
| `decoder.block.9.layer.2.DenseReluDense.wi_0` | 6.17e-05 |
| `decoder.block.10.layer.2.DenseReluDense.wo` | 4.30e-05 |
| `decoder.block.7.layer.2.DenseReluDense.wi_1` | 4.16e-05 |
| `decoder.block.10.layer.2.DenseReluDense.wi_0` | 3.85e-05 |
| `decoder.block.8.layer.1.EncDecAttention.v` | 3.22e-05 |

---

## 3. The selection criterion, and the rebuilt pack

The per-layer drift is a **screen**, not the selection criterion: it measures
the **weight** error only (the reconstruction is placed in the module and the
plain forward is used), while the pack's real path also quantizes the
**activation** for that layer, and that is where most of the error comes from.
The FP16-kept set is therefore given explicitly, with the layers taken in
the corrected drift's ranking, separately for the encoder and the decoder.

The candidates were measured on the 256-item acceptance set (seed 42) against
the FP16 source, with the baseline built through the **same ops class**.

| the candidate | the encoder kept | the decoder kept | the INT8 | cos | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| e0_d0 | 0 | 0 | 168 + 264 | 0.999750 | 11131.2 MB |
| e5_d10 | 5 | 10 | 163 + 254 | 0.999785 | 11538.9 MB |
| e10_d20 | 10 | 20 | 158 + 244 | 0.999811 | 11898.6 MB |
| e20_d40 | 20 | 40 | 148 + 224 | 0.999891 | 12498.0 MB |
| **e40_d80** | **40** | **80** | **128 + 184** | **0.999934** | **14056.6 MB** |
| e80_d160 | 80 | 160 | 88 + 104 | 0.999967 | 17245.9 MB |
| the previous pack | 5 | 202 | 163 + 62 | 0.999774 | 15998.8 MB |

**40 encoder + 80 decoder FP16-kept layers were adopted.** The quality rises
monotonically with the kept count; 80/160 buys +3.3e-05 of the cosine for
3.2 GB more, which is not the trade this pack is for.

Against the previous pack the new one is **better and smaller**: the cosine
0.999934 against 0.999774 (a 6.1x smaller error) and 14056.6 MB against
15998.8 MB (**-1942 MB**).

The previous pack's set came from the defective measurement, and it kept
**202** of the decoder's 264 layers against the encoder's 5 — the opposite
balance from what the output needs.

---

## 4. The pack

Built with `clip_convert/convert_t5_impact_convrot_int8.py`:

```
--drift-threshold 0.25 --quantize-decoder \
--decoder-impact impact/impact_flan_t5_xxl_decoder_fixed.json \
--decoder-max-drift 0.25 --protect-list impact/t5_protect_e40_d80.json
```

| item | value |
| :--- | :--- |
| the source | 21490.0 MB |
| the pack | **14056.6 MB** (-34.5 %) |
| the SHA256 | `DE3072A97C710EF2B1C661C3EEA0FBAA5759A0454007163228AF6D489ADE6BB0` |
| the tensors | 1183 |
| the quantized layers | **312** (the encoder 128, the decoder 184) |
| the FP16-kept layers | 120 (the encoder 40, the decoder 80) |
| the group size | 256 (all 312 layers) |
| the output | finite |
| the cos (256 items, seed 42) | **0.999934** |
| the previous pack's cos | 0.999774 |

**The model content is complete.** The decoder is not dropped: it is shipped
and quantized, so the checkpoint keeps every weight the source had. Only the
**runtime** never loads it (ComfyUI's T5 has no decoder), which is a load-path
fact and not an edit of the model.

---

## 5. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_t5.py` | the encoder's impact measurement |
| `sdxl/diag_impact_t5_decoder.py` | the decoder's impact measurement (the position biases chained) |
| `clip_convert/convert_t5_impact_convrot_int8.py` | the builder (`--protect-list`) |
| `sdxl/verify_t5.py` | the 256-item acceptance, the runtime's own path |
| `impact/impact_flan_t5_xxl_fixed.json` | the encoder's drift (git-ignored) |
| `impact/impact_flan_t5_xxl_decoder_fixed.json` | the decoder's drift (git-ignored) |
| `impact/t5_protect_e40_d80.json` | the adopted FP16-kept list (git-ignored) |
