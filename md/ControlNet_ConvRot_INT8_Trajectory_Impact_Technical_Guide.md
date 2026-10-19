# ControlNet ConvRot INT8 Trajectory-Impact Protection (Technical Guide)

This document describes, in full, how a ControlNet checkpoint is converted to ConvRot INT8 with a **measured** FP16 protection set, using the HSWQ trajectory-sensitivity method.

The defining property of this method is that the protection set is **not** inferred from static weight statistics. It is measured by running the ControlNet inside a real ComfyUI generation pipeline and observing how much the denoising trajectory moves when a given layer is replaced by its ConvRot INT8 reconstruction. Layers whose measurement shows a large trajectory drift are kept in their original precision.

The reference implementation in this repository was validated on a **complete sweep of all 400 ConvRot-eligible layers** of an SDXL ControlNet, and the shipped pack uses a **drift threshold of 0.15**.

---

## 1. Overview and Core Idea

### 1.1 What this method computes

For a ControlNet checkpoint `W` (the FP16/BF16 source), the method produces:

1. **A trajectory-impact ranking** over the checkpoint's ConvRot-eligible layers, where the impact of a layer is the relative MSE drift of the final latent when only that layer is replaced by its ConvRot INT8 reconstruction.
2. **A protection set** derived from that ranking: every layer whose measured drift exceeds the threshold is kept in original precision (FP16/BF16), and every remaining eligible layer (2D Linear and 4D Conv2d) is converted to ConvRot INT8.
3. **A pack** in the ComfyUI `int8_tensorwise` layout with `comfy_quant` stamps and a `_quantization_metadata` section.

### 1.2 The central claim

> A static weight statistic (kurtosis, outlier ratio, absmax/std) is **not** a proxy for trajectory impact.
> Only the measured drift of the actual denoising trajectory is.

This is not a theoretical preference - it was measured. Section 5 shows a concrete case where the two rankings disagree on the most important layer, and Section 6 shows how a partial measurement would have produced a different (and unsafe) protection set.

---

## 2. Mathematical Setup

### 2.1 Notation

| Symbol | Meaning |
| :--- | :--- |
| $\Phi$ | the sampling map of one denoising step, $\Phi(x_t, t, c)$ |
| $x_T \to x_0$ | the full trajectory produced by the production sampler |
| $W_\ell$ | the weight tensor of candidate layer $\ell$ |
| $H_g$ | the normalized regular Hadamard matrix of group size $g$ (power of 4) |
| $\hat{W}_\ell$ | the ConvRot INT8 reconstruction of $W_\ell$ |

### 2.2 The perturbation model

For a candidate layer with weight $W_\ell \in \mathbb{R}^{O \times I}$ (2D Linear) or $\mathbb{R}^{O \times I \times k_h \times k_w}$ (4D Conv2d):

**Step 1 - orthogonal rotation.** The weight is rotated along its input dimension in groups of size $g$:

$$W^{\text{rot}} = W\, H_g^{\top} \quad\text{(group-wise, Linear)}$$

$$W^{\text{rot}}[o, i, :, :] = \sum_{j} W[o, j, :, :]\, H_g[i, j] \quad\text{(Conv2d, along in\_channels)}$$

The rotation is orthogonal ($H_g H_g^{\top} = I$), so it preserves the weight exactly while spreading activation outliers over the group.

**Step 2 - per-channel INT8.** Each output channel gets its own scale:

$$\text{scale}_i = \frac{\max_j |W^{\text{rot}}_{ij}|}{127}, \qquad W^{q}_{ij} = \text{round}\!\left(\frac{W^{\text{rot}}_{ij}}{\text{scale}_i}\right)$$

**Step 3 - the reconstruction used for measurement.** The value injected into the network during the impact measurement is the dequantized rotated tensor:

$$\hat{W}_\ell = \text{dequant}\!\left(\text{INT8}(W^{\text{rot}}_\ell)\right)$$

No inverse rotation is applied, matching exactly what the shipped INT8 pack does at runtime (the online activation rotation is paired with it).

### 2.3 The impact metric

Let $x^{(0)}$ be the trajectory of the unmodified ControlNet and $x^{(\ell)}$ the trajectory with layer $\ell$ replaced by $\hat{W}_\ell$. The drift of step $t$ is the scale-invariant relative MSE:

$$\text{rel\_mse}(x^{(\ell)}_t, x^{(0)}_t) = \frac{\lVert x^{(\ell)}_t - x^{(0)}_t \rVert^2}{\lVert x^{(0)}_t \rVert^2}$$

and the layer impact is the mean over all sampled steps:

$$\text{impact}(\ell) = \frac{1}{T} \sum_{t=1}^{T} \text{rel\_mse}\left(x^{(\ell)}_t, x^{(0)}_t\right)$$

**Ascending impact = the layer survives INT8 without breaking the trajectory. Descending impact = it must stay FP16.**

---

## 3. The Measurement Pipeline

### 3.1 Pipeline construction

The measurement is performed inside a real ComfyUI generation pipeline, not in isolation:

| Component | How it is built |
| :--- | :--- |
| Base model | `comfy.sd.load_checkpoint_guess_config` (ComfyUI-native) |
| ControlNet | `comfy.controlnet.load_controlnet` |
| Control hint | A fixed deterministic tensor installed with `set_cond_hint`; the *content* is irrelevant for the measurement, only its presence and the resulting graph matter |
| Conditioning | CLIP positive/negative exactly as the benchmark builds them, with the ControlNet injected into the positive conditioning (as the apply node does) |
| Latent | `comfy.sample.fix_empty_latent_channels` |
| Sampler | `comfy.sample.sample` with `dpmpp_2m` / `karras` / CFG 7.0 / fixed seed |

Because the production sampler is used, the measured ranking follows exactly the trajectory that a quality gate would measure.

### 3.2 The measurement loop

For each candidate layer $\ell$:

1. Clone the original weight $W_\ell$.
2. Compute $\hat{W}_\ell$ (Section 2.2).
3. Overwrite `module.weight.data` with $\hat{W}_\ell$ (converted back to the original dtype).
4. Re-run the trajectory with the **same** seed, prompt, cfg and step count.
5. Record $\text{impact}(\ell)$.
6. Restore the original weight.

Only one layer is perturbed at a time, so every impact value is a marginal effect measured in the full-pipeline context - not a static score of the weight alone.

### 3.3 Candidate selection

A full sweep over every eligible layer of a ControlNet is expensive: each candidate costs one full denoising run. The implementation supports two modes:

| Mode | Flag | Candidates | Purpose |
| :--- | :--- | :--- | :--- |
| Focused | *(default)* | the fat-tail group found by a cheap static screen, **plus the 20 layers immediately below it** (the decision boundary) | fast exploration of the region where the protect/quantize decision is actually made |
| Full | `--full` | **every eligible module** | the definitive measurement, used for the shipped pack |

The shipped pack in this repository was produced with the **full sweep**.

### 3.4 Memory management and resumability

A full sweep of 400 layers runs for roughly 70-90 minutes and is memory-bound: each iteration allocates a weight clone plus the ConvRot INT8 reconstruction, and the sampler keeps a per-step latent history.

Two implementation details make the full sweep survive on a 16 GB card:

- **Aggressive release**: the reconstruction is freed immediately after the injection, the per-step latents are copied to CPU with `torch.float32` (contiguous), the perturbed run's trajectory is deleted right after the metric is computed, and `torch.cuda.empty_cache()` runs after every layer.
- **Checkpoint + resume**: the partial result is written to the impact JSON every 10 layers, and `--resume <json>` loads already-measured layers and skips them. A run that dies (VRAM exhaustion, out-of-memory kill) resumes from the last checkpoint instead of starting over.

The reference run of 400 layers was executed in three segments (180 + 180 + 60 layers) using this mechanism; the first two segments died from memory exhaustion before the release/resume logic was added.

### 3.5 Runtime-module to checkpoint-key mapping

The impact JSON stores **ComfyUI runtime module names** (`input_blocks.2.0.in_layers.2`), because that is what the loaded model exposes. The checkpoint stores **diffusers-format keys** (`down_blocks.0.resnets.1.conv1`). The two are related by ComfyUI's own converter, `comfy.utils.unet_to_diffusers`, applied to the checkpoint's inferred UNet config. The converter uses that mapping in reverse, so **no hand-maintained name table exists** and the mapping cannot drift from ComfyUI's behaviour.

---

## 4. The Converter

### 4.1 What is quantized

| Weight kind | Treatment |
| :--- | :--- |
| 2D Linear (`ndim == 2`) | ConvRot INT8 (rotated + per-out-channel INT8) unless protected |
| 4D Conv2d (`ndim == 4`) | ConvRot INT8 (rotated along in\_channels + per-channel INT8) unless protected |
| 1D / bias / buffers / non-float | Kept in original precision |
| Protected layer | Kept in original precision (byte-exact) |

The protection decision applies to both 2D and 4D weights; a Conv2d whose measured drift is large stays in FP16 exactly like a Linear.

### 4.2 Pack layout

```
<layer>.weight           int8
<layer>.weight_scale     float32  ([out,1] for Linear / [out,1,1,1] for Conv2d)
<layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
_quantization_metadata   JSON  {"format_version":"1.0","layers":{...}}
```

### 4.3 Group size

The Hadamard group size must divide the input dimension and be a power of 4. The converter picks the largest such size not exceeding 256. When no valid group size exists for a layer (for example a 3-channel conv\_in), that layer falls back to plain per-channel INT8 and is reported as a warning - it is never silently skipped.

---

## 5. Why the Measurement Matters: A Measured Case

Measured on `CN-anytest4_illustrious2_B.safetensors` (SDXL ControlNet, 844 keys, 401 eligible weights, **all 400 measured**).

### 5.1 The static screen and the trajectory disagree

The static screen (kurtosis of the weight distribution) and the measured trajectory drift rank the layers differently, and crucially they disagree about the most dangerous layer:

| Layer (checkpoint key) | Static kurtosis | Measured trajectory drift | Verdict |
| :--- | :--- | :--- | :--- |
| `down_blocks.0.resnets.1.conv1` | 2.4 (looks harmless) | **1.131** | **must stay FP16** |
| `down_blocks.0.resnets.0.conv1` | 1.1 | **0.716** | **must stay FP16** |
| `down_blocks.0.resnets.1.time_emb_proj` | 5.0 | 0.223 | must stay FP16 |
| `add_embedding.linear_1` | 21.1 (top static) | 0.105 | protected (trajectory) |
| `middle_block.1.transformer_blocks.4.attn2.to_v` | low | 0.015 | safe to quantize |

The Conv2d layers (`conv1`, runtime name `in_layers.2` / `out_layers.3`) look *mild* to a static kurtosis screen - their weight distribution is nearly Gaussian - yet they are the single most damaging layers in the whole ControlNet when quantized. A static-only selection would have quantized exactly the layers that break the trajectory.

### 5.2 Measured family statistics (all 400 layers)

| Layer family | n | mean drift | median | min | max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `input_blocks.*.0.in_layers.2` | 6 | **0.456** | 0.240 | 0.160 | 1.131 |
| `input_blocks.*.0.out_layers.3` | 6 | **0.433** | 0.313 | 0.160 | 1.209 |
| `input_blocks.*.0.op` | 2 | 0.273 | 0.273 | 0.195 | 0.352 |
| `input_blocks.*.1.proj_in` | 4 | 0.263 | 0.262 | 0.245 | 0.283 |
| `input_blocks.*.1.proj_out` | 4 | 0.257 | 0.240 | 0.194 | 0.354 |
| `input_blocks.*.0.skip_connection` | 2 | 0.208 | 0.208 | 0.182 | 0.233 |
| `input_blocks.*.0.emb_layers.1` | 6 | 0.178 | 0.176 | 0.134 | 0.223 |
| `zero_convs.*.0` | 9 | 0.177 | 0.175 | 0.089 | 0.443 |
| `middle_block.*.out_layers.3` | 2 | 0.176 | 0.176 | 0.174 | 0.179 |
| `input_blocks.*.1.transformer_blocks.*.attn2.to_q` | 24 | 0.168 | 0.177 | 0.058 | 0.242 |
| `input_blocks.*.1.transformer_blocks.*.attn2.to_k` | 24 | 0.151 | 0.148 | 0.062 | 0.257 |
| `middle_block.*.in_layers.2` | 2 | 0.132 | 0.132 | 0.106 | 0.157 |
| `input_blocks.*.1.transformer_blocks.*.ff.net.2` | 24 | 0.131 | 0.130 | 0.065 | 0.172 |
| `input_blocks.*.1.transformer_blocks.*.attn1.to_v` | 24 | 0.131 | 0.132 | 0.067 | 0.275 |
| `input_blocks.*.1.transformer_blocks.*.ff.net.*.proj` | 24 | 0.122 | 0.122 | 0.060 | 0.185 |
| `input_blocks.*.1.transformer_blocks.*.attn1.to_out.0` | 24 | 0.117 | 0.110 | 0.046 | 0.203 |
| `middle_block.*.transformer_blocks.*.attn2.to_k` | 10 | 0.111 | 0.092 | 0.057 | 0.219 |
| `middle_block.*.transformer_blocks.*.attn2.to_q` | 10 | 0.100 | 0.103 | 0.038 | 0.172 |
| `input_blocks.*.1.transformer_blocks.*.attn1.to_k` | 24 | 0.095 | 0.084 | 0.049 | 0.194 |
| `input_blocks.*.1.transformer_blocks.*.attn2.to_v` | 24 | 0.094 | 0.082 | 0.020 | 0.202 |
| `input_blocks.*.1.transformer_blocks.*.attn1.to_q` | 24 | 0.093 | 0.085 | 0.044 | 0.183 |
| `middle_block.*.transformer_blocks.*.attn2.to_out.0` | 10 | **0.030** | 0.027 | 0.005 | 0.069 |
| `middle_block.*.transformer_blocks.*.attn2.to_v` | 10 | **0.026** | 0.019 | 0.012 | 0.061 |

The ordering is clear and not intuitive from the architecture alone: the **residual Conv2d path** (`in_layers.2` / `out_layers.3`) is by far the most sensitive part of a ControlNet, while the **cross-attention value path** (`attn2.to_v` / `attn2.to_out.0`) is the most tolerant.

### 5.3 The threshold choice

The drift threshold converts the ranking into a protection set. Measured on this checkpoint:

| Threshold | Protected layers | Pack size | Reduction vs 2.34 GB |
| :--- | :--- | :--- | :--- |
| 0.10 | 218 | 1.86 GB | 20.6 % |
| **0.15 (shipped)** | **105** | **1.46 GB** | **37.3 %** |
| 0.20 | 44 | - | - |

The shipped pack uses **0.15**: it keeps the measured worst offenders (the residual Conv2d path and the attention projection families) in FP16, while the 129 layers whose drift falls in the 0.10-0.15 band are quantized.

### 5.4 The shipped pack

| Property | Value |
| :--- | :--- |
| Source | `CN-anytest4_illustrious2_B.safetensors` (2.34 GB, BF16) |
| Output | `CN-anytest4_illustrious2_B_hswq_convrot_int8_t015.safetensors` (1.46 GB / 1570 MB) |
| Protected (drift > 0.15) | **105 layers**, kept byte-exact |
| ConvRot INT8 | 277 Linear + 18 Conv2d |
| INT8 payload | 934.4 MB |
| FP16/BF16 retained | 633.2 MB |
| Load check | `comfy.utils.load_torch_file` -> 1436 keys, 296 `comfy_quant` entries |

---

## 6. Why a Full Sweep Was Necessary

The focused mode measured 29 layers (the static fat-tail group plus the 20 layers just below it). That measurement produced a 13-layer protection set.

The full sweep of 400 layers showed that this was **unsafe**: the layers that actually matter most (the residual Conv2d path, mean drift 0.43-0.46) sit **outside** the static fat-tail group, because their weight distributions are near-Gaussian and therefore look mild to a kurtosis screen. In the focused measurement the highest-drift layers that were visible were already at 0.15-1.13, but 92 additional layers in the 0.15-0.35 band - including `proj_in` / `proj_out`, `skip_connection`, `zero_convs` and the `emb_layers.1` family - were never measured at all.

The practical consequence: a focused measurement can only be trusted for the band it actually covered. For a shipped artifact, the full sweep is the only defensible basis.

---

## 7. Relationship to the UNet Reverse-Hybrid Method

The measurement engine is the same one used for the UNet/DiT reverse-hybrid method (`sdxl/diag_impact_sdxl.py`): same perturbation model, same drift metric, same production sampler. What differs is the host:

| | UNet / DiT | ControlNet |
| :--- | :--- | :--- |
| Host model | The generation model itself | The ControlNet branch, injected into an SDXL generation |
| Baseline trajectory | The model's own denoising | The generation with the ControlNet applied |
| Candidate keys | `model.diffusion_model.*` | `down_blocks.*` (diffusers) / `input_blocks.*` (ComfyUI runtime) |
| Name mapping | Direct | `comfy.utils.unet_to_diffusers`, inverted |

A ControlNet is a structural copy of the UNet backbone, so the drift it induces in the generation trajectory is comparable in meaning to the UNet's own drift - but the ControlNet contributes a *residual* added to the UNet's prediction, so its sensitivity profile is genuinely its own and has to be measured on its own. Section 5.2 shows the profile is in fact quite different from what one would guess from the architecture.

---

## 8. Reproducing the Measurement

```bash
# 1) full trajectory sweep (resumable; safe to interrupt)
python sdxl/diag_impact_controlnet.py \
  --base <base_sdxl.safetensors> \
  --controlnet <controlnet.safetensors> \
  --out impact/impact_<stem>_full.json \
  --comfy_path <ComfyUI checkout> \
  --steps 20 --seed 42 --cfg 7.0 --full \
  --resume impact/impact_<stem>_full.json

# 2) convert with the measured protection set
python clip_convert/convert_controlnet_convrot_int8.py \
  --input <controlnet.safetensors> \
  --output <controlnet>_hswq_convrot_int8_t015.safetensors \
  --impact impact/impact_<stem>_full.json \
  --comfy_path <ComfyUI checkout> \
  --drift-threshold 0.15
```

The converter reads the impact JSON, keeps every layer whose measured drift exceeds `--drift-threshold` in original precision, and converts the rest.

---

## 9. Practical Notes and Pitfalls

**Read the safetensors header correctly.** The data of a tensor starts at `8 + header_length + data_offset`, not at `data_offset`. Reading at the wrong offset yields garbage values that look like plausible but meaningless statistics (e.g. a kurtosis of 9,494 for a tensor whose true kurtosis is 21). Use `safetensors.safe_open(...).get_tensor(...)` or account for the header length.

**Do not use static statistics as the protection criterion.** Section 5.1 shows the disagreement directly. Static statistics may be used as a cheap *screen* to choose which candidates are worth measuring, but the decision itself must come from the measurement.

**The Conv2d scope is a design decision, not an accident.** Some ControlNet converters quantize only 2D Linear weights and leave every Conv2d in FP16. That is a valid choice, but it is a different pack from "everything ConvRot-eligible is quantized, with the measured exceptions". Both exist in the wild; they are not comparable by size, because the protected-set size differs.

**Budget the measurement for interruptions.** A full sweep is a long job with a real chance of dying from memory exhaustion. Use the checkpointing and `--resume` mechanism rather than assuming a single clean run.

**Verify the load path.** A pack with `comfy_quant` stamps is loaded natively by ComfyUI for ControlNet only when the layer set and the `comfy_quant` metadata are consistent. Always confirm with `comfy.utils.load_torch_file` that the pack loads and that the expected number of `comfy_quant` entries is present.

---

## 10. Summary

| Aspect | Choice |
| :--- | :--- |
| Protection criterion | Measured trajectory drift of the real ControlNet in a real ComfyUI generation |
| Measurement scope | **Full sweep of all 400 eligible layers** for the shipped pack |
| Perturbation | ConvRot INT8 reconstruction (Hadamard rotation + per-channel INT8) |
| Metric | Mean relative MSE of the per-step latent against the pristine trajectory |
| Sampler | Production sampler (`dpmpp_2m` / `karras` / CFG 7.0), fixed seed |
| Threshold | **0.15** -> 105 protected layers |
| Name mapping | ComfyUI's own `unet_to_diffusers`, inverted - no hand-written table |
| Quantized scope | 2D Linear + 4D Conv2d, minus the protected set |
| Result | 2.34 GB -> 1.46 GB, 277 Linear + 18 Conv2d in INT8 |
| Pack layout | `int8_tensorwise` with `comfy_quant` stamps and `_quantization_metadata` |

The essential point: for a ControlNet, the set of layers that must stay in FP16 is a property of *that checkpoint*, discoverable only by measuring *that checkpoint* in the pipeline it will actually run in. Everything else in the pipeline is a mechanism for making that measurement cheap, reproducible and survivable.
