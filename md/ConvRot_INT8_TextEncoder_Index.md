# ConvRot INT8 — Text Encoder / CLIP Quantization Index

> The rules the thresholds and the kept sets below were produced under are stated
> once in [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md)
> - Section *How a pack is selected (the current method)*.

Every ConvRot INT8 pack built for a **standalone text encoder / CLIP** checkpoint,
which method produced it, and where the details live.

**Scope.** Standalone text-encoder checkpoints only. The CLIP inside an SD1.5 or SDXL
checkpoint is part of the UNet (a single-file, already-finished model) and is
**out of scope here** - it is not touched by this work.

---

## 1. Two methods

| method | protection set | status |
| :--- | :--- | :--- |
| **impact-driven** (current) | measured drift of the model's **own output**, per layer | current standard |
| **static kurtosis** (old) | kurtosis / outlier-ratio of the weights | superseded - the kurtosis values were computed through a broken safetensors header offset (`f.seek(offs[0])` instead of `8 + header_len + offs[0]`), so the protection basis was invalid |

Any pack produced by the old method is being rebuilt with the impact-driven method.
The measurement script, the builder and the verification script are shipped per model
so the result is reproducible.

---

## 2. Current state per checkpoint

| checkpoint | towers quantized | thresholds | cos (mean / min) | size | guide |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `clip_l.safetensors` (FLUX.1 CLIP-L) | text | the FP16-kept set by the aggregate contribution (10 of 73 kept, 63 INT8, group size 256) | 0.999876 / 0.999841 | 135.5 MB | `md/CLIPL_ConvRot_INT8_Impact_Technical_Guide.md` |
| `ViT-L-14-BEST-smooth-GmP-HF-format.safetensors` | text + vision | the FP16-kept set by the aggregate contribution (30 text + 60 vision kept in FP16, 127 INT8, group size 256) | text 0.999964 / 0.999906, vision 0.999803 / 0.996378 | 646.7 MB | `md/ViTL14_ConvRot_INT8_Impact_Technical_Guide.md` |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors` | text + vision (the extended text context) | the FP16-kept set by the aggregate contribution (30 text + 60 vision kept in FP16, 130 INT8, group size 256) | text 0.999941 / 0.999888, vision 0.999925 / 0.999790 | 646.3 MB | `md/LongViTL14_ConvRot_INT8_Impact_Technical_Guide.md` |
| `CLIP-SAE-ViT-L-14-FP32.safetensors` | text + vision | the FP16-kept set by the aggregate contribution (30 text + 60 vision kept in FP16, 129 INT8, group size 256) | text 0.999962 / 0.999908, vision 0.999931 / 0.999797 | 658.9 MB | `md/SAEViTL14_ConvRot_INT8_Impact_Technical_Guide.md` |
| `Josiefied-Qwen3-0.6B-abliterated-v1.safetensors` | text | the FP16-kept set by the aggregate contribution (40 of 197 kept in FP16, 157 INT8, group size 256) | 0.999502 / 0.998980 | 655.2 MB | `md/JosiefiedQwen3_06B_ConvRot_INT8_Impact_Technical_Guide.md` |
| `qwen3_4b_abliterated_fp16_converted.safetensors` (Z Image's text encoder) | text | the FP16-kept set by the aggregate contribution (20 of 253 kept in FP16, 233 INT8, group size 256) | 0.999630 / 0.997430 | 4168.3 MB | `md/Qwen3_4B_ConvRot_INT8_Impact_Technical_Guide.md` |
| `Qwen3_VL_4B_Thinking_abliterated.safetensors` | text (the language + the vision tower) | the FP16-kept set by the aggregate contribution (100 of 358 kept, 258 INT8, group size 256) | 0.999649 / 0.998927 | 5958.9 MB | `md/Qwen3VL4B_ConvRot_INT8_Impact_Technical_Guide.md` |
| `flan_t5_xxl_fp16.safetensors` (FLAN-T5 XXL, encoder + decoder) | text | the FP16-kept set by the aggregate contribution (40 encoder + 80 decoder kept, 312 INT8 of 432, group size 256) | 0.999934 | 14056.6 MB | `md/T5XXL_ConvRot_INT8_Impact_Technical_Guide.md` |
| `sigclip_vision_patch14_384.safetensors` | vision | the FP16-kept set by the aggregate contribution (20 of 162 kept in FP16, 142 INT8, group size 64/16) | 0.996794 / 0.993237 | 502.9 MB | `md/SigLIPVision384_ConvRot_INT8_Impact_Technical_Guide.md` |
| `CLIP-ViT-H-14.safetensors` | vision | the FP16-kept set by the aggregate contribution (100 of 192 kept in FP16, 92 INT8, group size 256) | 0.999968 / 0.999906 | 1015.6 MB | `md/CLIPViTH14_ConvRot_INT8_Impact_Technical_Guide.md` |
| `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` | vision | the FP16-kept set by the aggregate contribution (35 of 288 kept, 253 INT8, group size 64) | 0.999916 / 0.999741 | 2180.1 MB | `md/CLIPBiGG_ConvRot_INT8_Impact_Technical_Guide.md` |

Acceptance criterion used throughout: **cos mean >= 0.99 AND cos min >= 0.98**, on a
fixed 256-item set (seed 42).

**A note on the ControlNet packs.** The ControlNet family follows the same
impact-driven method, but two FLUX-specific facts apply and are measured, not
assumed:

- **The fused `qkv` layers of a FLUX ControlNet can be quantized.** The
  runtime builds one Linear out of `cat([to_q, to_k, to_v], dim=0)`, and the
  three halves have value ranges that differ by ~7x (absmax 2.48 / 0.53 /
  0.375). On the **ConvRot** path the fused tensor reconstructs at **relMSE
  7.49e-05** (the plain int8 path is 2.16e-04) - the rotation mixes the
  columns, so one per-out-channel scale on the rotated tensor does not have
  to serve the three halves separately. Nine of the twelve are quantized in
  the shipped pack.
- **A FLUX ControlNet's pack carries only the tensors the runtime reads**
  (174 for the Union Pro 2.0). The source checkpoint is in the diffusers
  layout and the loader converts it; the converted state dict holds every
  tensor twice, so writing it verbatim would carry the weights twice.
- **The FLUX ControlNet's FP16-kept set is chosen by the aggregate
  contribution**, not by the per-layer drift - 21 of the 75 runtime Linear
  modules are kept in FP16 (54 quantized). See
  `md/FLUXControlNetUnionPro2_ConvRot_INT8_Impact_Technical_Guide.md`.

**How to read the size column.** The source checkpoints are often FP32 while
ComfyUI holds a text encoder in `float16`, so the file reduction and the runtime
reduction are different numbers. Measured on Long-ViT-L-14 (the source is
**F32 for all 427,747,841 parameters**, 590 tensors):

| | source | pack |
| :--- | :--- | :--- |
| file size | 1631.80 MB | 413.20 MB (-74.7 %) |
| resident weights at runtime | 235.90 MB | 118.48 MB (-49.8 %) |

The file figure is the plain FP32 -> INT8 factor of 1/4 (427,747,841 x 1 byte =
407.9 MB, and the pack measures 413.20 MB with the scales and the one FP16-kept
weight on top). **No tensor is missing**: the pack holds 428,119,001 parameters
against the source's 427,747,841. The runtime figure is smaller because the
source is already held in fp16 in memory and the token embedding (144.8 MB as
FP32) has to stay in FP16. See
`md/LongViTL14_ConvRot_INT8_Impact_Technical_Guide.md` sections 0.1 and 0.2.

**Two model-specific facts, measured.**

- **FLAN-T5 XXL carries a decoder the runtime never loads - and the pack keeps
  it.** The runtime T5 module (`comfy/text_encoders/t5.py`) holds `encoder` and
  `shared` only; the 339 `decoder.*` tensors (11.87 GB of the 20.99 GB
  checkpoint) are never loaded. They are still model content: dropping them
  would delete 6.37 B parameters, which is not quantization. The pack therefore
  keeps the decoder and quantizes it, using a separately measured decoder drift
  (the runtime has no decoder, so it is measured against a decoder that really
  executes). The size floor is structural: every parameter as int8 would be
  11,266,928,640 / 2 = **5.63 GB**, so a pack below 5 GB is not representable.
  The pack measures 15.62 GB (the decoder needs heavy protection - see the
  drift profile above).
- **The vision acceptance check has to use the same ops path for the
  baseline.** The stock `CLIPVisionLoader` builds the vision tower with
  `comfy.ops.manual_cast`, while a quantized pack is built with
  `mixed_precision_ops`, and for some models the two paths are **not
  numerically equivalent**: the same FP16 source through the two paths gives a
  cosine of 0.94-0.96 on CLIP-ViT-H-14, and restoring every quantized layer of
  the pack back to FP16 does not move the number. The pack was correct
  throughout. The baseline therefore has to be built through the same ops
  class, otherwise the comparison measures the ops path instead of the
  quantization.
- **The impact measurement must inject the reconstruction in the ORIGINAL
  frame.** The pack stores the rotated weight (`W_hat = W H^T`) and the kernel
  rotates the activation, so the result is `(x H) W_hat^T = x W^T`; the
  quantization error that reaches the output is `(q*s) H^T - W`, not
  `q*s - W`. Injecting the rotated frame into a plain `F.linear` measures a
  destroyed layer (relMSE 1.89 against the real kernel's 1.74e-04). The
  effect on a full sweep is 1.0e+01 to 1.0e-04 too large, and the ranking is
  inverted. This defect was present in 8 of the 9 impact scripts; all are
  fixed.
- **The FP16-kept set is chosen by the AGGREGATE contribution, not by the
  per-layer drift.** The per-layer drift is a marginal effect and does not
  rank the layers the way the aggregate output does. Measure it through the
  real pack path: load the all-INT8 pack, restore one layer's FP16 weight,
  and measure the aggregate cosine. On the FLUX ControlNet the two criteria
  disagree completely - the drift says every layer is equivalent
  (~1e-04 each), while the aggregate says 21 of the 75 layers carry the
  output. At the same 54 INT8 layers the aggregate-selected set measures min
  cos 0.999698 against the drift-selected set's 0.999160.
- **The `y` conditioning must be the production value.** For FLUX it is the
  CLIP-L pooled output, `vec_in_dim` wide (768). With `y = zeros` the
  `vector_in` layers have no input and a layer that changes the output
  measures 0.0000.
- **The per-layer drift is a SCREEN, not the selection criterion.** The
  drift measures the WEIGHT error only (the reconstruction is placed in the
  module and the plain forward is used), while the pack's real path also
  quantizes the ACTIVATION - and that is where most of the error comes from.
  Measured on the FLUX ControlNet: the drift-selected pack measures the relMSE
  1.11e-02, the aggregate-selected one 4.3e-05. Measured on SAM 3.1: the
  drift-selected pack measures 1.11e-02 against the shipped pack's 3.51e-05.
  The FP16-kept set has to be chosen by the **aggregate** effect on the
  output, measured through the real pack path.
- **A smaller protection set is not automatically better.** For SAM 3.1 every
  one of the 401 kept layers, when released, breaks the accepted quality by an
  order of magnitude (the cheapest release costs +9.67e-04 against a 1.0e-04
  gate) - the protection set cannot be reduced at all.
- **The group size follows the layer's input width** - this is not a
  CLIP-specific rule. SigLIP's hidden size is 1152, which is not a multiple of
  256 either, so 79 of its 162 layers are rotated with a group size of 64 and
  17 with 16. The builder derives it per layer, so no model is stuck with a
  group size its shapes cannot take.
- **The group size follows the layer's input width.** CLIP-ViT-bigG's hidden size
  is 1664, which is not a multiple of 256, so 240 of its 288 layers cannot be
  rotated with a group size of 256 - only the 48 `mlp.fc2` layers would be
  quantized. The builder derives the group size per layer (the largest
  power-of-4 divisor of the input width), so bigG uses 64 for 275 layers. The
  same rule applies to every other pack.
- **The INT8 GEMM needs the loader's runtime arm.** On the stock ComfyUI path an
  INT8 pack is dequantized for every matmul (`int8_tensorwise` is a weight-only
  algorithm) and `int8_linear` is called 0 times. With the arm installed the real
  INT8 GEMM runs: for FLAN-T5 XXL that is 163 calls per forward and **+75.9 % to
  +86.7 %** against the stock path (3.9x to 7.2x against FP16). See
  `md/T5XXL_ConvRot_INT8_Impact_Technical_Guide.md` section 7.

---

## 3. Why the thresholds differ per model

The threshold is not a global constant - it is chosen from the measured drift
distribution of that specific checkpoint, on that model's own acceptance set.

| model family | drift profile | consequence |
| :--- | :--- | :--- |
| CLIP-L / ViT-L / SAE ViT-L | a few layers dominate (e.g. ViT-L text: max 1.93 vs median 0.15) | protect the outliers, quantize the rest |
| Josiefied-Qwen3-0.6B | **flat** (min 0.75, median 3.05, max 3.43; IQR 2.94-3.17) | no layer is safer than another, so **selective protection cannot help** - quantize everything |
| FLAN-T5 XXL encoder | **rising with depth** (block 0 median 0.12 -> block 23 median 0.24), top drift in the deepest blocks' `attn.v` / `attn.o` / `ff.wi_0` | protect the deep-block outliers, quantize the other 163 of 168 |
| FLAN-T5 XXL decoder | **low per-layer drift (median 0.010) but extremely sensitive in the output**: quantizing the 72 FF layers alone drops the decoder cosine to 0.554, while quantizing all 192 attention layers keeps 0.879 | protect the FF layers: 62 of 264 quantized at the measured quality limit |
| CLIP-ViT-bigG-14 (vision) | **`mlp.fc1` dominates** (median 0.20 against ~0.012-0.047 for the other kinds), the outliers sit in the deep layers (40-44) | protect the 13 worst `mlp.fc1` layers, quantize the other 275 of 288 |
| SigLIP vision (sigclip_vision_patch14_384) | **uniformly sensitive**: even the smallest drift is 3.9e-02, above the CLIP text tower's median, and the median is 0.27 | the threshold has to be much higher than the CLIP family's; 96 of 162 quantized at 0.30 |

A pack built for one checkpoint must not be reused for another: the protection set is
model-specific by construction.

---

## 4. Load paths (measured)

| tower | loader | how INT8 gets enabled |
| :--- | --- | --- |
| CLIP text (SD1ClipModel) | `CLIPLoader` / `DualCLIPLoader` | `model_options["quantization_metadata"]` -> `SDClipModel` builds `mixed_precision_ops` |
| llama-family text (Qwen3) | `CLIPLoader` / `DualCLIPLoader` | automatic: `llama_detect` -> `detect_layer_quantization` -> `llama_quantization_metadata` |
| FLAN-T5 XXL text | `CLIPLoader` / `DualCLIPLoader` | automatic: `t5_xxl_detect` -> `detect_layer_quantization` -> `t5xxl_quantization_metadata` -> `model_options["quantization_metadata"]` |
| CLIP vision | `CLIPVisionLoader` | the stock loader uses `manual_cast`, which does **not** consume `comfy_quant`; the tower must be built with `mixed_precision_ops` |

---

## 5. Tooling

| script | purpose |
| --- | --- |
| `sdxl/diag_impact_clip.py` | text-tower drift (CLIP family) |
| `sdxl/diag_impact_clip_vision.py` | vision-tower drift (CLIP family) |
| `sdxl/diag_impact_qwen3.py` | llama-family text-encoder drift |
| `sdxl/diag_impact_t5.py` | FLAN-T5 XXL encoder drift (encoder hidden state) |
| `clip_convert/convert_t5_impact_convrot_int8.py` | FLAN-T5 XXL pack builder (impact-driven) |
| `sdxl/verify_t5.py` | FLAN-T5 XXL acceptance verification + latency |
| `sdxl/diag_impact_t5_decoder.py` | FLAN-T5 XXL decoder drift (transformers decoder stack, streamed block by block) |
| `clip_convert/convert_clip_impact_convrot_int8.py` | CLIP converter (impact-driven) |
| `sdxl/verify_*.py`, `sdxl/smoke_*.py`, `sdxl/perf_*.py` | per-model verification / smoke / performance |
