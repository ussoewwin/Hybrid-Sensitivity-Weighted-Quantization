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
2. **A protection set** derived from that ranking: every module whose measured drift exceeds the threshold is kept in its original precision (FP16/BF16), and every remaining eligible module is converted to ConvRot INT8. Only 2D `nn.Linear` layers whose input width is a multiple of 256 are eligible - see Section 4 for why the scope is exactly that.
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
| 2D Linear, input width a multiple of 256, not protected | ConvRot INT8 (rotated + per-out-channel INT8) |
| 2D Linear, protected by drift | Kept in original precision (byte-exact) |
| 2D Linear, input width not a multiple of 256 | Kept in original precision |
| 4D Conv2d | Kept in original precision |
| 1D / bias / buffers / non-float | Kept in original precision |

The scope is narrower than the ControlNet case, and that is a hard requirement of the runtime rather than a choice. Three facts were established by a measured quality gate against the FP16 baseline **through the real ComfyUI loader path** (`apply_comfy_quant_int8_patches` + `MixedPrecisionOps(int8_tensorwise)`, i.e. the `HSWQ SAM3 Loader` node):

1. **Only Linear layers are quantized.** ComfyUI's INT8 load path attaches an INT8 `QuantizedTensor` to Linear layers; Conv2d layers stay FP16. A pack that also quantized Conv2d therefore loaded as a **mixed** model - the Conv2d entries were read back as unrotated INT8 while the runtime kept FP16 - and the tracker mask came out inverted (measured pixel flip 0.966, IoU 0.000).
2. **The group size is fixed at 256.** ComfyUI's ConvRot kernel is 2D-only and its fused path activates only for `convrot_groupsize == 256` (`comfy/ops.py`, `comfy_kitchen/backends/cuda`). A Linear whose input width is not a multiple of 256 cannot be rotated by that kernel and is read back unrotated, so it corrupts the output; such layers stay in their original precision.
3. **Only modules the runtime actually has are considered.** The checkpoint stores more parameter names than the ComfyUI runtime exposes (many are folded or initialized from scratch). Quantizing a name the runtime does not have produces a mixed model; the converter therefore asks ComfyUI for the runtime module list and only converts a layer that is a real `nn.Linear` whose weight is present in the runtime state dict.

With this scope the measured gate is: **tracker mask flip 0.0016 / IoU 0.9983 / logit relMSE 4.3e-04**. The remaining detector-path difference is a **dtype effect, not a quantization effect**: comparing FP16 weights with bfloat16 activations against INT8 weights with bfloat16 activations gives tracker flip 0.0012 and detector IoU 0.951, i.e. the same magnitude as the FP16-fp16 vs FP16-bf16 difference.

### 4.2 Pack layout

```
<layer>.weight           int8
<layer>.weight_scale     float32  ([out,1])
<layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata   JSON  {"format_version":"1.0","layers":{...}}
```

### 4.3 Key mapping

The impact JSON stores **runtime module names** (`tracker.interactive_sam_mask_decoder.transformer.layers.0.mlp.0`), which for SAM 3.1 are identical to the checkpoint's parameter paths plus the `.weight` suffix - the standalone multiplex checkpoint already stores the split `q_proj` / `k_proj` / `v_proj` layout, so no name table and no `unet_to_diffusers` inversion is needed. Only the rotary `freqs_cis` buffers are dropped.

### 4.4 Group size

The group size is **fixed at 256**. It must divide the input dimension and be a power of 4, and the ComfyUI ConvRot kernel's fused path only activates at 256; any other value would be read back unrotated. A Linear whose input width is not a multiple of 256 is therefore not quantized at all rather than quantized at a smaller group size.

---

## 5. The Shipped Packs

| Property | Threshold 0.05 | Threshold 0.02 |
| :--- | :--- | :--- |
| Output | `sam3.1_multiplex_convrot_int8_t005.safetensors` | `sam3.1_multiplex_convrot_int8_t002.safetensors` |
| Size | **1623.6 MB** (1.586 GB) | **1657.6 MB** (1.619 GB) |
| Reduction vs 1664.7 MB source | 2.5 % | 0.4 % |
| ConvRot INT8 Linear | **65** | **41** |
| INT8 payload | 41.3 MB | 7.1 MB |
| Protected (drift or shape) | 156 | 180 |
| Kept as-is (1D, buffers) | 1068 | 1068 |
| Measured tracker mask IoU vs FP16 | **0.9983** | - |

A lower threshold means **more protected modules**, hence a **larger** pack - the same relationship the ControlNet packs show (174 protected -> 1783 MB, 105 -> 1570 MB, 61 -> 1397 MB).

**Why the pack is only 2.5 % smaller.** The size is dominated by the parameter names the runtime *does not* use. Of the 489 Linear layers the ComfyUI runtime has, only **221 receive a weight from this checkpoint** (the other 268 are initialized by the runtime itself); of those 221, only **184 have an input width that is a multiple of 256**, and 119 of them are protected at the 0.05 threshold. That leaves exactly the 65 quantized layers. The checkpoint's own 468 2D weights include many names the runtime never loads (folds), and the bulk of the remaining bytes are the fp16 weights the runtime keeps in full precision plus the 1D/buffer tensors. This is not a limitation of the threshold but of how many layers of this checkpoint are both runtime-live and 256-wide.

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
| Result | 1664.7 MB -> 1623.6 MB; 65 ConvRot INT8 Linear; 156 protected; measured tracker mask IoU 0.9983 |

The essential point for SAM is the same as for any other model, but with one addition: because the observable is a **discrete segmentation decision** rather than a smooth latent, the measured output drift is the only thing that speaks for the quality of the mask - and the measurement has to exercise **every path** the model offers, or half of it stays unverified.
