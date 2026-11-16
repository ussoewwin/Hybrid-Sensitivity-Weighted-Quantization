# SD1.5 IP-Adapter FaceID Plus v2 - ConvRot INT8 Impact Technical Guide

> **The conversion record** for this pack (what was quantised, the protection ranking, the conversion
> arguments, the faithfulness replay and the score) is
> [`IPAdapter_FaceID_PlusV2_SD15_ConvRot_INT8_Quantization_and_Score_Report.md`](IPAdapter_FaceID_PlusV2_SD15_ConvRot_INT8_Quantization_and_Score_Report.md).
>
> **The method this pack was built under** (the injection frame, the behavioural observable, the
> size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section
> *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter-faceid-plusv2_sd15.bin` (156,558,509 B, 344 tensors)
**Pack.** `ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors` (79,143,188 B, **-49.4 %**)
**SHA256 (source).** `26D0D86A1D60D6CC811D3B8862178B461E1EEB651E6FE2B72BA17AA95411E313`
**SHA256 (pack).** `3984CB16F3ED7507B60E4AC9B5EE21B10DCF8EA02AED9414ADB3535A8104C15E`

---

## Table of contents

1. [What the score is](#1-what-the-score-is)
2. [The loader requirement](#2-the-loader-requirement)
3. [The model](#3-the-model)
4. [The impact measurement](#4-the-impact-measurement)
5. [The candidates, and the adopted set](#5-the-candidates-and-the-adopted-set)
6. [The pack](#6-the-pack)
7. [The acceptance](#7-the-acceptance)
8. [Reproduction](#8-reproduction)
9. [Error sources and limits](#9-error-sources-and-limits)
10. [The experiment ledger](#10-the-experiment-ledger)
11. [Files](#11-files)

---

## 1. What the score is

The score is the **observed change of the adapter's own output** when one weight is replaced by its
ConvRot INT8 reconstruction, not a static statistic of the weights.

This checkpoint has two kinds of 2-D weights and they are measured differently:

| part | what the runtime does | how a weight's impact is measured |
| :--- | :--- | :--- |
| `image_proj` (24 Linears: `proj.0` 512 -> 1024, `proj.2` 1024 -> 3072, `proj_in` 1280 -> 768, four blocks of `to_q` / `to_kv` / `to_out` / feed-forward, `proj_out` 768 -> 768) | runs once inside `ipadapter_execute`, consumes the identity embedding (512-d) and the CLIP ViT-H penultimate sequence (1, 257, 1280), and emits 4 tokens x 768 | swap one weight, run the whole projection, compare the 4x768 stack |
| `ip_adapter` (288 Linears: 128 bias-free LoRA `down`/`up` pairs at rank 128, plus 16 `to_k_ip` and 16 `to_v_ip`) | the pairs are applied as a low-rank delta inside the UNet cross-attention; the `to_k_ip` / `to_v_ip` projections run on every forward | swap one weight and apply the weight (or its pair) to a fixed input of that layer's own width |

**This is v2, and the v2 branch is decided by the filename.** The file itself has no `faceidplusv2`
key; the node sets `model["faceidplusv2"] = True` when the filename contains `plusv2`
(`_ipadapter_model_loader`). The pack keeps `plusv2` in its name, and both arms were measured to carry
the flag, so both arms run the same branch.

**The injection frame matters.** The pack stores the **rotated** weight and the kernel rotates the
**activation**, so the error that reaches the output is `(q*s) @ H - W`, the original frame.

**The two surfaces are not interchangeable.** The `ip_adapter` weights consume the UNet
cross-attention K/V, not the image embedding; they are measured once on a fixed input of each layer's
own width. Only the `image_proj` surface is image-conditioned, and only it can be moved by an image
trigger.

**The acceptance gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98` over the
fixed 256-item set, with finite output. The item statistic is the mean of its 4 injected tokens.

---

## 2. The loader requirement

ComfyUI needs the loader package's `HSWQ Load IPAdapter Model` node. The stock
`comfyui_ipadapter_plus` loader reads int8 codes as magnitudes - a pack loaded through it is not INT8,
it is a wrong float tensor. The node installs `ConvRotInt8Linear` for the layers that carry a
`comfy_quant` sidecar, and the patch lives on the extension's `IPAdapter` / `To_KV` globals.

Measured on the pack: patch applied (`IPAdapter` and `To_KV` both replaced), 312 INT8 layers installed
(24 + 288), 312 INT8 kernel calls on one projection pass. Measured on the float source through the same
node: 0 INT8 layers - the patch is conditional on the sidecars.

---

## 3. The model

`image_proj` is `ProjModelFaceIdPlus`: `proj` (512 -> 1024 -> 3072, reshaped to `num_tokens x 768`),
a LayerNorm over 768, then `FacePerceiverResampler` (proj_in 1280 -> 768, four blocks of 12-head
attention at head 64 plus a 4x feed-forward, proj_out, norm_out). `forward(id_embeds, clip_embeds,
scale, shortcut)` returns `x + scale * resampler(x, clip)` when `shortcut` - the v2 path.

The node builds it with `cross_attention_dim = output_cross_attention_dim = 768`,
`clip_embeddings_dim = 1280`, `clip_extra_context_tokens = 4`, `is_faceid = True`, `is_plus = True`.
The SDXL argument set (`1280` / `2048` / `16 tokens`) builds a different module and would not load.

`ip_adapter` is 32 blocks. Each block carries `to_q_lora`, `to_k_lora`, `to_v_lora`, `to_out_lora` as
bias-free `down` (rank 128) / `up` pairs, with inner widths 320, 640, 768 and 1280 and outer widths
128, 320, 640, 768 and 1280. `To_KV` builds one module per entry, so 288 modules are built and the
inner widths span four values - which is why the group size is decided per layer (Section 6).

---

## 4. The impact measurement

Two instruments, both on this checkpoint, one seed (42), one fixed input set per surface.

**The drift screen** (`clip_convert/diag_impact_ipadapter_faceid_plusv2.py`): replace one weight by its
ConvRot INT8 reconstruction, run the stage, record the relative MSE of the stage output. 312 layers,
max `6.150512e-04`, median `2.952240e-05`, min `4.697038e-08`. The largest five:

| layer | relative MSE |
| :--- | ---: |
| `image_proj.proj.0.weight` | 6.150512e-04 |
| `image_proj.perceiver_resampler.layers.0.0.to_q.weight` | 4.153478e-04 |
| `image_proj.proj.2.weight` | 3.400411e-04 |
| `image_proj.perceiver_resampler.proj_in.weight` | 2.664621e-04 |
| `ip_adapter.17.to_out_lora.down.weight` | 7.449577e-05 |

**The criterion** (`clip_convert/measure_ipadapter_faceid_plusv2_contrib.py`): the same sweep scored as
the aggregate contribution (1 - cosine) to the stage output. Sum `5.621867e-03`; the largest
contributors are `image_proj.proj.0` (5.18 %), `layers.0.0.to_q` (3.72 %), `proj.2` (2.54 %),
`proj_in` (2.37 %), then `ip_adapter.14.to_v_lora.down` (0.69 %) and `ip_adapter.8.to_out_lora.down`
(0.61 %).

**No layer crosses the 0.15 drift threshold.** The protection decision is therefore made by the K curve,
not by the threshold - see Section 5.

---

## 5. The candidates, and the adopted set

The K curve: for each K, the top-K layers of this checkpoint's own contribution ranking are held in
FP16, the pack is rebuilt, and the same 256-item gate is run through the same node.

| K | pack MB | INT8 | protected | gate mean | gate min | pass | contribution captured |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | 79.143 | 312 | 0 | 0.999945044 | 0.999930516 | yes | 0.0000 |
| 20 | 88.962 | 292 | 20 | 0.999986390 | 0.999976352 | yes | 0.2302 |
| 40 | 101.098 | 272 | 40 | 0.999986390 | 0.999976352 | yes | 0.3348 |
| 60 | 106.471 | 252 | 60 | 0.999986390 | 0.999976352 | yes | 0.4296 |

**Adopted: K = 0.** The rule is the smallest passing pack; a larger candidate must beat it by a
meaningful share of the gate margin. K=0 holds `0.009945` of margin, the best larger candidate gains
`4.135e-05` of mean (0.42 % of that margin) and costs `+27.328 MB`.

**The three larger points return the same gate numbers by construction**: the gate observes the
`image_proj` stack, and K=20/40/60 protect the same four `image_proj` layers (the rest of their budget
goes to `ip_adapter`, which the gate does not see). They are reported as measured, not as three
independent confirmations. The `ip_adapter` surface is covered by CN-1 (0.999968 / 0.999955 through the
adapter's own forward) and CN-5a (0.999974 / 0.999960 over 288 weights).

---

## 6. The pack

| item | value |
|---|---|
| tensors | 344 in, 344 out |
| 2-D weights | 312, all INT8 |
| float kept | 32 tensors that are not 2-D weights (biases and LayerNorm weights) |
| group sizes | 256 for 124 layers, 64 for 188 layers |
| sidecars | one `weight_scale` (float32, `[out, 1]`) and one `comfy_quant` per INT8 layer |
| size | 79,143,188 B, 0.5056 of the source |

**The group size has to be read per layer.** The shared IP-Adapter converter fixes 256 and leaves any
layer whose inner width is not a multiple of 256 in FP16; on this graph that is every 320 / 640 / 1280
layer, and the first run of that converter produced **0 INT8 layers on a 312-layer graph** with 344
tensors kept - a pack that looks quantised and is not. This pack's converter takes the largest power of
four that divides the inner width (256, 64, 16 or 4), which here is 256 or 64.

**CN-1 (faithfulness).** The converter's arithmetic replayed on the source for all 312 layers: 0
differing integers, 0 scale mismatches, maximum relative scale difference `0.0`, 0 non-finite scales,
and the 32 float tensors bit-identical. Through the adapter's own forward, the injected stack cosine is
`0.999968` mean / `0.999955` min.

---

## 7. The acceptance

256 items (8 crops x 4 orientations x 8 brightness steps of the reference image), one fixed identity
embedding from that image (insightface buffalo_l, 512-d), each item through CLIP-H, gate
`mean >= 0.99 and min >= 0.98`:

| statistic | value |
|---|---:|
| n | 256 |
| mean | 0.999945044 |
| min | 0.999930516 |
| p5 | 0.999938630 |
| p50 | 0.999945320 |
| INT8 kernel calls | 6,144 |
| pass | yes |

Against the source (`records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json`): `image_proj`
`0.998930` mean / `0.985521` min over four CLIP inputs (relative MSE max `9.774e-03`), and the 288
`ip_adapter` weights `0.999974` / `0.999960`.

**Trigger search (CN-5b).** 56 candidates on the image-dependent surface, 5 natural-transform controls;
a candidate is flagged above the control maximum x 1.5. **0 flagged**: control band `1.079442e-04`,
largest candidate `8.364785e-05`, negative control (source against itself) `6.960186e-05`. Positive
control: 10 % of the first installed `image_proj` layer's codes shifted by +4 LSB is the first ladder
step above the band.

**Render (CN-4).** Two arms, 512 x 512, 12 steps, cfg 7.0, dpmpp_2m / karras, weight 0.8,
weight_faceidv2 1.0, identity from the reference image, both through the loader node and
`IPAdapterFaceID`. Both arms read as finished portraits of the same subject; pixel cosine `0.987881`,
mean |diff| `12.981`, PSNR `21.008 dB`.

---

## 8. Reproduction

```
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe
#   COMFY   = <ComfyUI>
#   LOADER  = the ComfyUI-HSWQ-Loader-and-Tools checkout

# 1. the drift screen (312 layers, seed 42)
python clip_convert/diag_impact_ipadapter_faceid_plusv2.py --ipadapter <f.bin> --out impact.json --comfy <COMFY>

# 2. the criterion
python clip_convert/measure_ipadapter_faceid_plusv2_contrib.py --ipadapter <f.bin> --out contrib.json --comfy <COMFY>

# 3. the curve (K = 0 / 20 / 40 / 60), each pack built and gated
python clip_convert/sweep_ipadapter_faceid_plusv2.py --impact impact.json --contrib contrib.json --input <f.bin> --work <dir> --ks 0 20 40 60

# 4. the adopted pack (K = 0)
python clip_convert/convert_ipadapter_faceid_plusv2_convrot_int8.py --input <f.bin> --impact impact.json --output <pack.safetensors> --record conversion.json

# 5. CN-1 / CN-2 / CN-5a
python audit/ipadapter_faceid_plusv2_pack_audit.py --source <f.bin> --pack <pack> --comfy <COMFY> --json cn1.json
python audit/ipadapter_faceid_plusv2_comfyui_route.py --pack <pack> --source <f.bin> --comfy <COMFY> --loader <LOADER> --dir <dir> --json cn2.json
python audit/ipadapter_faceid_plusv2_acceptance.py --pack <pack> --source <f.bin> --comfy <COMFY> --json cn5a.json

# 6. CN-6 / CN-5b / CN-4
python audit/ipadapter_faceid_plusv2_comfyui_gate.py --pack <pack> --source <f.bin> --comfy <COMFY> --loader <LOADER> --dir <dir> --image <ref.png> --clip-vision <clip_vision_h> --json cn6.json
python audit/ipadapter_faceid_plusv2_comfyui_trigger.py --pack <pack> --source <f.bin> --comfy <COMFY> --loader <LOADER> --dir <dir> --image <ref.png> --clip-vision <clip_vision_h> --json cn5b.json
python audit/ipadapter_faceid_plusv2_render.py --server <host:port> --out-dir <dir> --json cn4.json --output-root <server output dir>
```

The render needs a running server; it was run on a second instance (`--port 8189 --lowvram
--disable-assets`) while another instance was already running, so both arms ran on one host with
identical settings.

---

## 9. Error sources and limits

- **The gate observes the `image_proj` stack only.** The 288 `ip_adapter` weights are outside it and
  are covered by CN-1, CN-2 and CN-5a.
- **The identity embedding is one face.** The 256 items vary the CLIP side; a second identity would be
  a second instrument.
- **K=20/40/60 gate numbers are identical by construction** (Section 5).
- **A1111 was not measured for this pack.** The instruction for this run is ComfyUI; the A1111
  IP-Adapter runtime that serves the SDXL pack does not build this module set. Scope statement, not a
  pass.
- **The trigger search cannot see a trigger trained into the weights before quantisation.**
- **`image_proj.proj.0` concentrates the error** (512 -> 3072, drift 6.15e-04, 5.18 % of the
  contribution); the measured effect on the injected stack is 0.998930 / 0.985521 over four CLIP
  inputs.
- **The drift threshold protects nothing here**: the largest drift is 6.15e-04 against 0.15, so the
  protection decision is the K curve.

---

## 10. The experiment ledger

| step | what ran | outcome |
|---|---|---|
| 1 | drift screen, 312 layers, seed 42 | max 6.150512e-04 / median 2.952240e-05 / min 4.697038e-08 |
| 2 | contribution criterion, 312 layers | sum 5.621867e-03, top share 5.18 % |
| 3 | K curve 0 / 20 / 40 / 60, each converted and gated | all pass; K=0 adopted (margin 0.009945, best larger gain 4.135e-05) |
| 4 | conversion, K=0 | 312 INT8 / 0 protected / 32 float kept, 79,143,188 B |
| 5 | CN-1 replay | 0 differing integers, 0 scale mismatch, 32/32 float bit-identical |
| 6 | CN-1 behaviour | 0.999968 mean / 0.999955 min on the injected stack |
| 7 | CN-2 route | 312 installed (24 + 288), 312 kernel calls, source arm 0 |
| 8 | CN-5a | `image_proj` 0.998930 / 0.985521 (4 CLIP inputs), `ip_adapter` 0.999974 / 0.999960 |
| 9 | CN-6 gate | mean 0.999945044 / min 0.999930516, 6,144 kernel calls, PASS |
| 10 | CN-5b trigger | 56 candidates, 0 flagged, band 1.079442e-04, floor 4 LSB |
| 11 | CN-4 render | both arms finished images, cosine 0.987881, PSNR 21.008 dB |

---

## 11. Files

* `clip_convert/diag_impact_ipadapter_faceid_plusv2.py`
* `clip_convert/measure_ipadapter_faceid_plusv2_contrib.py`
* `clip_convert/sweep_ipadapter_faceid_plusv2.py`
* `clip_convert/convert_ipadapter_faceid_plusv2_convrot_int8.py`
* `audit/ipadapter_faceid_plusv2_pack_audit.py`
* `audit/ipadapter_faceid_plusv2_comfyui_route.py`
* `audit/ipadapter_faceid_plusv2_acceptance.py`
* `audit/ipadapter_faceid_plusv2_comfyui_gate.py`
* `audit/ipadapter_faceid_plusv2_comfyui_trigger.py`
* `audit/ipadapter_faceid_plusv2_render.py`
* `md/backdoor/verdicts/CN_IPAdapter_FaceID_PlusV2_SD15_Pack.md`
* `md/backdoor/verdicts/records/cn_ipadapter_faceid_plusv2_*`
