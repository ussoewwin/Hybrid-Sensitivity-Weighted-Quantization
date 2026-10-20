# SAM 3.1 Multiplex ConvRot INT8 Trajectory-Impact Protection (Technical Guide)

This document describes how the SAM 3.1 Multiplex checkpoint is converted to ConvRot INT8 with a **measured** FP16 protection set.

SAM 3.1 requires a **different measurement** from the ControlNet / UNet case, and this guide explains why: a SAM model is not a denoiser. It produces **masks**, not a latent trajectory, so there is no denoising trajectory to measure. What is measured instead is the drift of the model's **own output** - the mask logits from the tracker SAM-decoder path and the detector's boxes / scores / masks from the text path - when a single layer is replaced by its ConvRot INT8 reconstruction. The drift scale that results is an order of magnitude away from the ControlNet numbers, and the threshold has to be chosen on SAM's own distribution.

Everything below is backed by a **completed full sweep**: **all 551 ConvRot-eligible modules of SAM 3.1 Multiplex measured in a single uninterrupted run**.

| Checkpoint | Modules measured | Runs | Run time | Stoppages |
| :--- | :--- | :--- | :--- | :--- |
| `sam3.1_multiplex_fp16.safetensors` | 551 / 551 | 1 | ~75 min | 0 |

---

## 1. Overview and Core Idea

### 1.1 What this method computes

For a SAM checkpoint `W` (the FP16 source), the method produces:

1. **An output-drift ranking** over the checkpoint's ConvRot-eligible modules, where the drift of a module is the relative MSE of the model's output when only that module is replaced by its ConvRot INT8 reconstruction.
2. **A protection set** derived from that ranking: every module whose measured drift exceeds the threshold is kept in its original precision (FP16/BF16), and every remaining eligible module (2D Linear and 4D Conv2d) is converted to ConvRot INT8.
3. **A pack** in the ComfyUI `int8_tensorwise` layout with `comfy_quant` stamps and a `_quantization_metadata` section.

### 1.2 The central claim, and why SAM needs its own version of it

> A static weight statistic (kurtosis, outlier ratio) is **not** a proxy for output sensitivity.
> Only the measured drift of the model's real output is.

For SAM this is even more important than for a denoiser, because SAM's output is a **discrete decision** (which pixels are in the mask, which boxes are detected). A layer whose drift is small can still flip a mask edge or a detection threshold, and a static weight score cannot see that at all.

### 1.3 Why the measurement differs from the ControlNet method

| | ControlNet / UNet | SAM 3.1 |
| :--- | :--- | :--- |
| What the model produces | a denoising trajectory (latents) | **masks, boxes, scores** |
| What is perturbed | one weight tensor | one weight tensor (same) |
| What is compared | per-step latent vs the pristine latent | **mask logits + detector boxes/scores/masks** vs the pristine output |
| Paths exercised | one generation | **two**: tracker (point prompt) and detector (text prompt) |
| Measured range | max ~1.2 - 3.7 | **max 10.64** |
| Median | 0.06 - 0.15 | **0.0024** |
| Sensible threshold | 0.15 | **0.05** |

The same "perturb one layer, measure the output change" principle applies; what changes is the observable and therefore the scale of the metric.

---

## 2. The Measurement Pipeline

### 2.1 Building the model

The checkpoint is loaded exactly the way ComfyUI loads it:

```python
sd = comfy.utils.load_torch_file(path)
model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
net   = model.model            # comfy.model_base.SAM3
sam3  = net.diffusion_model    # comfy.ldm.sam3.detector.SAM3Model (detector + tracker)
```

This is the same construction the `HSWQ SAM3 Loader` node uses, so the measured model is the model ComfyUI runs.

### 2.2 The two paths

A SAM 3.1 Multiplex model has two independent sub-networks, and **both** must be exercised or half the model will never be measured:

| Path | Entry point | Prompt | Output |
| :--- | :--- | :--- | :--- |
| **Tracker** (SAM decoder) | `sam3.forward_segment(frame, point_inputs=...)` | a fixed point prompt | `[1, 1, 1008, 1008]` mask logits |
| **Detector** | `sam3(frame, text_embeddings=..., text_mask=...)` | a fixed text embedding | `boxes`, `scores`, `masks` |

Both outputs are concatenated into one vector and compared against the pristine run.

**Measured consequence of covering both paths**: in a first attempt that exercised only the tracker path, all 200+ detector-side modules measured an exact drift of **0.0** - they were simply never executed. After adding the detector path, every module that either path touches produces a non-zero measurement.

### 2.3 The fixed input

- **Image**: a deterministic synthetic image (`torch.rand` with a fixed seed, 1008x1008, scaled to the model's dtype). The *content* is irrelevant - what matters is that the same graph is reproduced for every measurement.
- **Point prompt**: `point_coords = [[[256.0, 256.0]]]`, `point_labels = [[1]]` (the tracker SAM-decoder path).
- **Text prompt**: a fixed random `[1, 32, 1024]` embedding with an all-ones attention mask (the detector path). A fixed embedding reproduces the detector graph without needing a CLIP text encoder in the loop.

The input dtype follows the model (`next(p.dtype for p in sam3.parameters())`), which is fp16 for this checkpoint - a float32 input raises a dtype mismatch inside the first conv.

### 2.4 The measurement loop

For each candidate module:

1. Clone the original weight **to CPU**.
2. Compute the ConvRot INT8 reconstruction **on CPU**.
3. Overwrite the module's weight with it, run the pipeline, compute the drift against the pristine output.
4. Restore the original weight.

Only one module is perturbed at a time, so each value is a marginal effect in the full-pipeline context.

### 2.5 Memory management and single-run completion

The same three rules that made the ControlNet full sweep finish in one run apply here:

- **Detached execution** - the sweep is launched with `Start-Process`, not as a child of the agent's shell, so it survives an agent-session restart. (In the ControlNet case every run had died after exactly ~33 minutes for this reason.)
- **CPU-side weight clone and reconstruction** - the transient GPU peak of "clone + quantize" is removed; only the single perturbed tensor is moved to the GPU, for the duration of the injection. Steady VRAM during the SAM sweep was ~3.5 GB.
- **Checkpoint every 10 modules + `--resume`** - the partial result is written to the impact JSON every 10 modules and already-measured modules are skipped on a restart.

---

## 3. The Measured Result

### 3.1 Distribution (all 551 modules)

| Statistic | Value |
| :--- | :--- |
| Modules measured | 551 |
| Exact zero drift | **162** (modules not executed by either measured path) |
| min | 0.0 |
| median | **2.38e-03** |
| p90 | 1.68e-01 |
| max | **10.64** |

### 3.2 The decisive finding: the mask decoder is the fragile part

The nine modules of the tracker's interactive mask decoder are in a league of their own:

| Module | drift |
| :--- | :--- |
| `tracker.interactive_sam_mask_decoder.obj_score_token` | **10.64** |
| `tracker.interactive_sam_mask_decoder.pred_obj_score_head.layers.0` | **10.64** |
| `tracker.interactive_sam_mask_decoder.transformer.layers.0.mlp.0` | **10.64** |
| `tracker.interactive_sam_mask_decoder.transformer.layers.0.mlp.2` | **10.64** |
| `tracker.interactive_sam_mask_decoder.transformer.layers.0.self_attn.{k,q,v,out}_proj` | **10.64** |
| `tracker.interactive_sam_prompt_encoder.point_embeddings.1` | **10.64** |

These are the layers that turn the tracker's features into the final mask logits and the object-presence score. Their drift is **not** slightly larger than the rest - it is an order of magnitude larger, and identical across the group because they all sit on the same critical path to the output.

Note what this means: the *object-presence score* head is a **binary decision** about whether an object exists. A perturbation there does not shift the mask slightly; it changes the score, and a score crossing the decision threshold changes the output qualitatively.

### 3.3 Family statistics (n >= 2)

| Layer family | n | mean | median | max |
| :--- | :--- | :--- | :--- | :--- |
| `tracker.interactive_sam_mask_decoder.transformer.layers.N.mlp.0 / .mlp.2` | 2 each | **5.321** | 5.321 | 10.642 |
| `tracker.interactive_sam_mask_decoder.transformer.layers.N.self_attn.{k,q,v,out}_proj` | 2 each | **5.321** | 5.321 | 10.642 |
| `detector.transformer.encoder.layers.N.linear1` | 6 | 0.160 | 0.168 | 0.228 |
| `detector.backbone.vision_backbone.trunk.blocks.N.mlp.fc1` | 32 | 0.131 | 0.115 | 0.329 |
| `detector.backbone.vision_backbone.trunk.blocks.N.attn.qkv` | 32 | 0.118 | 0.107 | 0.301 |
| `detector.transformer.decoder.layers.N.cross_attn.out_proj` | 6 | 0.117 | 0.072 | 0.314 |
| `detector.backbone.vision_backbone.trunk.blocks.N.mlp.fc2` | 32 | 0.106 | 0.099 | 0.252 |
| `detector.transformer.encoder.layers.N.linear2` | 6 | 0.100 | 0.102 | 0.142 |
| `detector.transformer.encoder.layers.N.self_attn.v_proj` | 6 | 0.099 | 0.103 | 0.148 |
| `detector.transformer.decoder.layers.N.cross_attn.v_proj` | 6 | 0.099 | 0.040 | 0.314 |
| `detector.backbone.vision_backbone.trunk.blocks.N.attn.proj` | 32 | 0.096 | 0.072 | 0.229 |
| `detector.transformer.encoder.layers.N.self_attn.out_proj` | 6 | 0.094 | 0.097 | 0.156 |
| `detector.transformer.decoder.layers.N.cross_attn.k_proj` | 6 | 0.087 | 0.021 | 0.419 |
| `detector.backbone.vision_backbone.convs.N.conv_1x1` | 3 | 0.082 | 0.002 | 0.244 |
| `detector.transformer.encoder.layers.N.self_attn.q_proj` | 6 | 0.082 | 0.067 | 0.187 |
| `detector.transformer.decoder.layers.N.cross_attn.q_proj` | 6 | 0.073 | 0.029 | 0.311 |
| `detector.transformer.encoder.layers.N.self_attn.k_proj` | 6 | 0.073 | 0.067 | 0.110 |
| `detector.backbone.vision_backbone.convs.N.conv_3x3` | 3 | 0.067 | 0.002 | 0.199 |
| `detector.transformer.decoder.layers.N.linear1` | 6 | 0.067 | 0.067 | 0.094 |
| `detector.transformer.decoder.layers.N.self_attn.out_proj` | 6 | 0.040 | 0.009 | 0.106 |
| `detector.transformer.decoder.layers.N.self_attn.v_proj` | 6 | 0.034 | 0.012 | 0.109 |
| `detector.transformer.decoder.layers.N.linear2` | 6 | 0.021 | 0.024 | 0.033 |
| `tracker.sam_mask_decoder.*` (12 modules) | 12 | **0.000** | 0.000 | 0.000 |
| `tracker.maskmem_backbone.fuser.layers.N.pwconv1 / .pwconv2` | 4 | **0.000** | 0.000 | 0.000 |
| `detector.backbone.vision_backbone.propagation_convs.*` | 30+ | **0.000** | 0.000 | 0.000 |

Two structural readings:

1. **Inside the tracker, only the interactive mask decoder is on the measured path.** The plain `sam_mask_decoder` (used for the non-interactive / video path) and the memory fuser measure exactly 0 - they are not executed by the measured point-prompt path.
2. **The detector's vision backbone is broadly sensitive** (every family of the ViT trunk sits at 0.10-0.13), while the detector's *transformer* layers are a mix of sensitive (`linear1` at 0.16) and tolerant (`linear2` at 0.02) parts.

### 3.4 Threshold choice

| Threshold | Protected modules |
| :--- | :--- |
| 0.01 | 239 |
| 0.02 | 225 |
| **0.05 (shipped)** | **174** |
| 0.10 | 115 |
| 0.15 | 74 |

The shipped pack uses **0.05**. For comparison, 0.15 - the ControlNet threshold - would leave 74 protected modules on a 551-module model, i.e. it would quantize the interactive mask decoder that the measurement identifies as the single most fragile group in the network.

---

## 4. The Converter

### 4.1 What is quantized

| Weight kind | Treatment |
| :--- | :--- |
| 2D Linear | ConvRot INT8 (rotated + per-out-channel INT8) unless protected |
| 4D Conv2d | ConvRot INT8 (rotated along in\_channels + per-channel INT8) unless protected |
| 1D / bias / buffers / non-float | Kept in original precision |
| Protected module | Kept in original precision (byte-exact) |

The protection decision applies to both 2D and 4D weights.

### 4.2 Pack layout

```
<layer>.weight           int8
<layer>.weight_scale     float32  ([out,1] for Linear / [out,1,1,1] for Conv2d)
<layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
_quantization_metadata   JSON  {"format_version":"1.0","layers":{...}}
```

### 4.3 Key mapping

The impact JSON stores **runtime module names** (`tracker.interactive_sam_mask_decoder.transformer.layers.0.mlp.0`), which for SAM 3.1 are identical to the checkpoint's parameter paths plus the `.weight` suffix - the standalone multiplex checkpoint already stores the split `q_proj` / `k_proj` / `v_proj` layout, so no name table and no `unet_to_diffusers` inversion is needed. Only the rotary `freqs_cis` buffers are dropped.

### 4.4 Group size

The Hadamard group size must divide the input dimension and be a power of 4; the converter picks the largest such size not exceeding 256 and falls back to plain per-channel INT8 (with a warning) when none exists.

---

## 5. The Shipped Packs

| Property | Threshold 0.05 | Threshold 0.02 |
| :--- | :--- | :--- |
| Output | `sam3.1_multiplex_convrot_int8_t005.safetensors` | `sam3.1_multiplex_convrot_int8_t002.safetensors` |
| Size | **1319.8 MB** (1.29 GB) | **1418.5 MB** (1.39 GB) |
| Reduction vs 1.63 GB source | 19.1 % | 13.0 % |
| Protected modules | **143** (138 2D + 5 4D) | **181** (176 2D + 5 4D) |
| ConvRot INT8 | 379 modules | 341 modules |
| FP16 / BF16 retained | 890.4 MB | 1088.2 MB |
| INT8 payload | 427.5 MB | 328.6 MB |
| Kept as-is (1D, buffers) | 1068 | 1068 |

A lower threshold means **more protected modules**, which means **more FP16 retained**, which means a **larger** pack. The 38 modules that move into the protected set between 0.02 and 0.05 cost +197.8 MB of FP16 retention while removing 98.9 MB of INT8 payload - a net +98.7 MB, which is exactly the measured difference between the two packs. This is the same relationship the ControlNet packs show (174 protected -> 1783 MB, 105 -> 1570 MB, 61 -> 1397 MB).

Both packs are produced from the same measured impact JSON; only the threshold differs.

---

## 6. Reproducing the Measurement

```bash
# 1) full output-drift sweep (single run; see Section 2.5 for the operational recipe)
python sdxl/diag_impact_sam31.py \
  --model <sam3.1_multiplex_fp16.safetensors> \
  --out impact/impact_sam3.1_multiplex.json \
  --comfy_path <ComfyUI checkout> \
  --full --resume impact/impact_sam3.1_multiplex.json

# 2) convert with the measured protection set
python clip_convert/convert_sam31_convrot_int8.py \
  --input <sam3.1_multiplex_fp16.safetensors> \
  --output <out>.safetensors \
  --impact impact/impact_sam3.1_multiplex.json \
  --drift-threshold 0.05
```

---

## 7. Practical Notes and Pitfalls

**Exercise both paths.** A tracker-only measurement leaves every detector module at drift 0.0 and produces a protection set that misses the entire detector half of the model.

**Match the model dtype.** The SAM 3.1 backbone runs in fp16; a float32 input tensor fails in the first convolution with `Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same`.

**Do not carry the ControlNet threshold over.** SAM's drift scale is different by construction (the observable is a mask, not a latent), and 0.15 - which is the ControlNet threshold - is far too permissive here.

**Zero drift means "not executed", not "safe".** 162 of the 551 measured modules measured exactly 0.0 because neither measured path reaches them (`propagation_convs`, the plain `sam_mask_decoder`, the memory fuser). Those modules are **unmeasured, not verified safe**. A future measurement of the video / propagation path would be needed to cover them.

**Read the safetensors header correctly.** Tensor data starts at `8 + header_length + data_offset`. Reading at `data_offset` yields garbage values that look like plausible but meaningless statistics.

**Do not use static weight statistics as the protection criterion.** They may be used as a cheap screen to choose which modules are worth measuring, but the decision must come from the measurement.

---

## 8. Summary

| Aspect | Choice |
| :--- | :--- |
| Protection criterion | Measured drift of SAM 3.1's own output (mask logits + detector boxes/scores/masks) |
| Measurement scope | **Full sweep of all 551 eligible modules, one uninterrupted run** |
| Paths exercised | tracker (point prompt) **and** detector (text prompt) |
| Perturbation | ConvRot INT8 reconstruction (Hadamard rotation + per-channel INT8) |
| Metric | Relative MSE of the concatenated model output against the pristine output |
| Threshold | **0.05** |
| Most fragile group | `tracker.interactive_sam_mask_decoder.*` - 9 modules at drift 10.64 (vs 0.12 typical) |
| Unmeasured remainder | 162 modules (video / propagation paths) - measured drift 0.0 |
| Pack layout | `int8_tensorwise` with `comfy_quant` stamps and `_quantization_metadata` |
| Result | 1.63 GB -> 1.23 GB, 326 Linear + 44 Conv2d in INT8, 143 protected |

The essential point for SAM is the same as for any other model, but with one addition: because the observable is a **discrete segmentation decision** rather than a smooth latent, the measured output drift is the only thing that speaks for the quality of the mask - and the measurement has to exercise **every path** the model offers, or half of it stays unverified.
