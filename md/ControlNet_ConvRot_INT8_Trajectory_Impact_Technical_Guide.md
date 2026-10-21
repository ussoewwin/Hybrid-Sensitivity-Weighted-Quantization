# ControlNet ConvRot INT8 Trajectory-Impact Protection (Technical Guide)

This document describes, in full, how a ControlNet checkpoint is converted to ConvRot INT8 with a **measured** FP16 protection set, using the HSWQ trajectory-sensitivity method.

The defining property of this method is that the protection set is **not** inferred from static weight statistics. It is measured by running the ControlNet inside a real ComfyUI generation pipeline and observing how much the denoising trajectory moves when a given layer is replaced by its ConvRot INT8 reconstruction. Layers whose measurement shows a large trajectory drift are kept in their original precision.

Everything below is backed by **completed full sweeps**: two SDXL ControlNet checkpoints, **all 400 ConvRot-eligible layers measured in a single uninterrupted run each**, with the measured numbers quoted verbatim.

| Checkpoint | Layers measured | Runs | Run time | Stoppages |
| :--- | :--- | :--- | :--- | :--- |
| `CN-anytest4_illustrious2_A.safetensors` | 400 / 400 | 2 | 62 + 103 min | 0 |
| `CN-anytest4_illustrious2_B.safetensors` | 400 / 400 | 1 | 71 min | 0 |
| `controlnet-union-pro-max-sdxl-1.0.safetensors` | 407 / 407 | 1 | 97 min | 0 |

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

This is not a theoretical preference - it was measured, twice, on two different checkpoints. Section 5 shows the disagreement; Section 6 shows how a partial measurement would have produced a different, unsafe protection set; Section 9 describes the operational procedure that makes a complete sweep finish in one run.

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
| Base model | `comfy.sd.load_checkpoint_guess_config` (ComfyUI-native), `waiIllustriousSDXL_v170.safetensors` |
| ControlNet | `comfy.controlnet.load_controlnet` |
| Control hint | A fixed deterministic tensor installed with `set_cond_hint`; the *content* is irrelevant for the measurement, only its presence and the resulting graph matter |
| Conditioning | CLIP positive/negative exactly as the benchmark builds them, with the ControlNet injected into the positive conditioning (as the apply node does) |
| Latent | `comfy.sample.fix_empty_latent_channels` |
| Sampler | `comfy.sample.sample` with `dpmpp_2m` / `karras` / CFG 7.0 / 20 steps / seed 42 |

Because the production sampler is used, the measured ranking follows exactly the trajectory that a quality gate would measure.

### 3.2 The measurement loop

For each candidate layer $\ell$:

1. Clone the original weight $W_\ell$ **to CPU**.
2. Compute $\hat{W}_\ell$ **on CPU** (Section 2.2).
3. Overwrite `module.weight.data` with $\hat{W}_\ell$ (converted back to the original dtype).
4. Re-run the trajectory with the **same** seed, prompt, cfg and step count.
5. Record $\text{impact}(\ell)$.
6. Restore the original weight.

Only one layer is perturbed at a time, so every impact value is a marginal effect measured in the full-pipeline context - not a static score of the weight alone.

### 3.3 Candidate selection

The implementation supports two modes:

| Mode | Flag | Candidates | Purpose |
| :--- | :--- | :--- | :--- |
| Focused | *(default)* | the static fat-tail group plus the 20 layers immediately below it (the decision boundary) | fast exploration of the region where the protect/quantize decision is actually made |
| **Full** | `--full` | **every eligible module (400 on these checkpoints)** | the definitive measurement, used for the shipped packs |

**The shipped packs in this repository were produced with the full sweep, in one uninterrupted run each.**

### 3.4 Runtime-module to checkpoint-key mapping

The impact JSON stores **ComfyUI runtime module names** (`input_blocks.2.0.in_layers.2`), because that is what the loaded model exposes. The checkpoint stores **diffusers-format keys** (`controlnet_down_blocks.N.weight`, `controlnet_cond_embedding.*`), which ComfyUI remaps at load time to the comfy names (`zero_convs.N.0`, `input_hint_block.N`, `input_blocks.*`). The pack must be written under the **runtime (comfy) names** - see Section 4.4 for why writing under the checkpoint's own diffusers names produces a black image.

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

**Why the scope is exactly this.** Three facts were established by a measured gate against the FP16 ControlNet **through the real ComfyUI loader path** (`apply_comfy_quant_int8_patches` + `MixedPrecisionOps(int8_tensorwise)`, i.e. the `HSWQ ControlNet Loader` node):

1. **Only 2D Linear layers are quantized.** ComfyUI's INT8 load path attaches a QuantizedTensor to Linear layers only; Conv2d layers stay FP16. Quantizing Conv2d produced a mixed model.
2. **The group size is fixed at 256.** The ConvRot kernel is 2D-only and its fused path activates only at `convrot_groupsize == 256`. A Linear whose input width is not a multiple of 256 is read back unrotated and corrupts the output.
3. **The pack must be written under the runtime (comfy) key names - not the checkpoint's diffusers names.** This was the cause of a shipped pack that produced a **black image** (see Section 4.4).

### 4.2 Pack layout

```
<layer>.weight           int8
<layer>.weight_scale     float32  ([out,1])
<layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata   JSON  {"format_version":"1.0","layers":{...}}
```

`<layer>` is a **runtime (comfy) module name** - `input_blocks.*`, `zero_convs.N.0`, `input_hint_block.N`, ... - not the checkpoint's diffusers name.

### 4.3 Group size

The group size is **fixed at 256** (it must divide the input dimension and be a power of 4; the fused ConvRot path only activates at 256). A Linear whose input width is not a multiple of 256 is not quantized at all rather than quantized at a smaller group size.

### 4.4 Key naming: the diffusers-to-comfy remap

ComfyUI loads an SDXL ControlNet through its **diffusers** branch, which remaps the checkpoint's diffusers-style names to the comfy-style names the runtime actually builds:

| Checkpoint (diffusers) | Runtime (comfy) |
| :--- | :--- |
| `controlnet_down_blocks.N.weight` | `zero_convs.N.0.weight` |
| `controlnet_cond_embedding.conv_in` / `.blocks.N` / `.conv_out` | `input_hint_block.2N` / `.2N+2` |
| `controlnet_mid_block.weight` | `middle_block_out.0.weight` |
| `controlnet_cond_embedding.*` (the rest) | `input_blocks.*` |

A quantized pack **must be written under the runtime names**. If the quantized weights are written under the checkpoint's diffusers names, the remap moves only `.weight` and leaves `.weight_scale` / `.comfy_quant` behind; the runtime then sees a bare int8 tensor with no quant metadata and the model outputs NaN - a **black image**.

Measured on the broken pack: **340 of 346** quantized modules did not exist in the runtime module graph, and only **3** of the 369 runtime Linear layers received a QuantizedTensor. The converter now builds the ControlNet the way the runtime does, takes the runtime module graph as the source of truth, and writes the pack under the runtime names.

---

## 5. Why the Measurement Matters: Three Measured Checkpoints

All three are SDXL ControlNets of the same architecture. Their measured sensitivity profiles are **completely different from each other**, which is the central point of this section.

### 5.1 The static screen and the trajectory disagree

The static screen (kurtosis of the weight distribution) and the measured trajectory drift rank the layers differently, and critically they disagree about the most dangerous layer:

| Layer (checkpoint key) | Static kurtosis | Measured drift (B) | Verdict |
| :--- | :--- | :--- | :--- |
| `down_blocks.0.resnets.1.conv1` | 2.4 (looks harmless) | **1.131** | **must stay FP16** |
| `down_blocks.0.resnets.0.conv1` | 1.1 | **0.716** | **must stay FP16** |
| `down_blocks.0.resnets.1.time_emb_proj` | 5.0 | 0.223 | must stay FP16 |
| `add_embedding.linear_1` | 21.1 (top static) | 0.105 | protected (trajectory) |
| `middle_block.1.transformer_blocks.4.attn2.to_v` | low | **0.018** | safe to quantize |

The Conv2d layers (`conv1`, runtime name `in_layers.2` / `out_layers.3`) look *mild* to a static kurtosis screen - their weight distribution is nearly Gaussian - yet they are among the most damaging layers in the whole ControlNet when quantized. A static-only selection would have quantized exactly the layers that break the trajectory.

### 5.2 Measured family statistics (all layers, all three checkpoints)

Mean drift per layer family:

| Layer family | n | A mean | A max | B mean | B max | U mean | U max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `input_blocks.*.0.in_layers.2` | 6 | **0.406** | 0.714 | **0.455** | 1.131 | 0.398 | 0.729 |
| `input_blocks.*.0.out_layers.3` | 6 | **0.396** | 0.549 | **0.433** | 1.209 | **0.430** | 0.963 |
| `input_blocks.*.1.proj_in` | 4 | 0.278 | 0.331 | 0.263 | 0.283 | 0.291 | 0.373 |
| `input_blocks.*.1.proj_out` | 4 | 0.272 | 0.305 | 0.257 | 0.354 | 0.276 | 0.351 |
| `input_blocks.*.0.emb_layers.1` | 6 | 0.235 | 0.289 | 0.178 | 0.223 | 0.262 | 0.438 |
| `zero_convs.*.0` | 9 | 0.219 | 0.511 | 0.177 | 0.443 | 0.321 | 1.200 |
| `input_blocks.*.1.transformer_blocks.*.attn2.to_q` | 24 | 0.212 | 0.354 | 0.168 | 0.241 | 0.030 | 0.095 |
| `input_blocks.*.1.transformer_blocks.*.attn2.to_v` | 24 | 0.094 | 0.202 | 0.082 | 0.176 | **0.014** | 0.055 |
| `middle_block.*.transformer_blocks.*.attn2.to_out.0` | 10 | **0.030** | 0.069 | **0.028** | 0.066 | 0.015 | 0.060 |
| `middle_block.*.transformer_blocks.*.attn2.to_v` | 10 | **0.026** | 0.061 | **0.022** | 0.055 | **0.004** | 0.009 |

Two structural facts hold across all three checkpoints:

1. **The residual Conv2d path dominates.** `in_layers.2` / `out_layers.3` are the most sensitive families everywhere (mean 0.40-0.46), roughly **an order of magnitude** above the attention projections.
2. **The cross-attention value path is the most tolerant.** `attn2.to_v` / `attn2.to_out.0` sit at mean 0.004-0.030 - up to **60x less sensitive** than the Conv2d path.

### 5.3 Whole-distribution comparison

| Statistic | A | B | U (union-pro-max) |
| :--- | :--- | :--- | :--- |
| Layers measured | 400 | 400 | **407** |
| min drift | 0.0122 | 0.0053 | 0.0000 |
| **median drift** | **0.1478** | **0.1119** | **0.0601** |
| p90 drift | 0.2512 | 0.2026 | 0.1960 |
| **max drift** | 0.7141 | **1.2090** | **3.7339** |
| drift > 0.10 | 356 | 233 | 131 |
| **drift > 0.15** | **190** | **117** | **74** |
| drift > 0.20 | 86 | 44 | 40 |
| drift > 0.25 | 40 | 16 | 30 |

The three checkpoints have **fundamentally different shapes**:

- **A** is uniformly sensitive (highest median, moderate tail).
- **B** has a lower median but a heavy tail (max 1.209).
- **U** (union-pro-max) is the opposite of A: its *typical* layer is the least sensitive of the three (median 0.060, less than half of A), yet it owns by far the most extreme individual layer (**3.734**).

The extreme layers in U are its own:

| Layer (U) | drift |
| :--- | :--- |
| `input_hint_block.2` | **3.734** |
| `input_hint_block.8` | **2.432** |
| `zero_convs.0.0` | 1.200 |
| `input_blocks.1.0.out_layers.3` | 0.963 |
| `input_blocks.1.0.in_layers.2` | 0.729 |

`input_hint_block.*` is not even a sensitive family in A or B (0.15-0.35 there) - in U it is the single most dangerous object in the model. This is exactly the kind of structure that no architecture-derived or inherited rule can predict.

**At the same 0.15 threshold the required protection ranges over a factor of three (190 / 117 / 74 measured layers).** Any approach that derives the protection set from the architecture, from a static score, or from another checkpoint's numbers gets at least two of the three wrong.

### 5.4 The threshold choice

| Threshold | A protected | B protected | U protected |
| :--- | :--- | :--- | :--- |
| 0.10 | 356 | 233 | 131 |
| **0.15 (the earlier measurement)** | **190** | **117** | **74** |
| **0.010 (A, re-measured)** | **222** | - | - |
| **0.005 (U, re-measured)** | - | - | **41** |
| 0.20 | 86 | 44 | 40 |
| 0.25 | 40 | 16 | 30 |

The shipped packs use **0.15**.

### 5.5 The shipped packs

| Property | A | B | U (union-pro-max) |
| :--- | :--- | :--- | :--- |
| Source | `CN-anytest4_illustrious2_A.safetensors` (2386.2 MB) | `CN-anytest4_illustrious2_B.safetensors` (2386.2 MB) | `controlnet-union-pro-max-sdxl-1.0.safetensors` (2396.9 MB) |
| Output | `CN-anytest4_illustrious2_A_hswq_convrot_int8.safetensors` | `CN-anytest4_illustrious2_B_hswq_convrot_int8_t015.safetensors` | `controlnet-union-pro-max-sdxl-1.0_hswq_convrot_int8.safetensors` |
| Size | **1959.5 MB** | **1533.2 MB** | **1368.8 MB** |
| Reduction vs source | **17.9 %** | 35.7 % | **42.9 %** |
| Protected | **222 (drift>0.010)** | 117 (drift>0.15) | **41 (drift>0.005)** |
| ConvRot INT8 Linear | **140** | **257** | **328** |
| INT8 payload | 669.4 MB | 855.3 MB | 955.9 MB |
| Load check | 1236 keys, 196 `comfy_quant` | 1358 keys, 257 `comfy_quant` | 1459 keys, 298 `comfy_quant` |
| Measured gate (cos vs FP16, real loader path) | **0.999879** (relMSE 5.96e-04) | **1.000044** (relMSE 1.28e-04) | **0.999961** (relMSE 7.74e-06) |

The size ordering follows the protected-layer ordering: A protects the most layers and is therefore the largest pack; U protects the fewest and is the smallest. The pack size is a direct consequence of the measured protection set - not of the quantization scope, which is identical for all three.

## 6. Why a Full Sweep Was Necessary

An earlier focused measurement covered 29 layers (the static fat-tail group plus the 20 layers just below it) and produced a 13-layer protection set for checkpoint B.

The full sweep of 400 layers showed that this was **unsafe**. The layers that actually matter most - the residual Conv2d path, mean drift 0.43-0.46 - sit **outside** the static fat-tail group, because their weight distributions are near-Gaussian and therefore look mild to a kurtosis screen. The focused measurement did happen to catch the single worst layer (1.131), but it never measured the families at 0.15-0.35 (`proj_in` / `proj_out`, `skip_connection`, `zero_convs`, the `emb_layers.1` family), and the full sweep found **92 additional layers** in that band that the focused run had never looked at.

With a 0.15 threshold, the two answers differ by a factor of nine (13 vs 117 protected layers). A focused measurement can only be trusted for the band it actually covered; for a shipped artifact, the full sweep is the only defensible basis.

The third checkpoint makes the point sharper. Union-pro-max's most dangerous layers are its `input_hint_block.2` (drift 3.734) and `input_hint_block.8` (2.432) - a family that sits at 0.15-0.35 in A and B and would never have been reached by a static fat-tail screen at all. A focused measurement of that checkpoint would have missed the worst layer in the model by more than an order of magnitude.

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
# 1) full trajectory sweep (single run, see Section 9 for the operational recipe)
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
  --output <controlnet>_hswq_convrot_int8.safetensors \
  --impact impact/impact_<stem>_full.json \
  --comfy_path <ComfyUI checkout> \
  --drift-threshold 0.15
```

The converter reads the impact JSON, keeps every layer whose measured drift exceeds `--drift-threshold` in original precision, and converts the rest.

---

## 9. Making a Complete Sweep Finish in One Run

This is the part that decides whether a full sweep is practical. The first attempts died repeatedly; the reference runs above completed in one pass. Three changes made the difference, and all three are load-bearing.

### 9.1 Do not tie the sweep to the agent session

The initial implementation ran the sweep as a child process of the agent's shell. Every single attempt died after **exactly ~33 minutes** - not at a memory limit, not at a layer boundary, and always at the same wall-clock offset. The diagnosis was direct: an instrumented run showed **no memory growth at all** (RSS 3.45 -> 3.70 GB flat, VRAM 8.28 GB flat across 15 consecutive trajectories), so the process was not exhausting anything - it was being **terminated from outside**, when the agent session restarted.

The fix is to launch the sweep as a **detached process** that is not a child of the agent's shell:

```powershell
$runner = "run_sweep.ps1"
@'
param([string]$Tag, [string]$ControlNet, [string]$Steps = "20")
$py   = "D:\USERFILES\ComfyUI\python_embeded\python.exe"
$repo = "D:\USERFILES\GitHub\hswq"
$base = "D:\USERFILES\ComfyUI\ComfyUI\models\unet\waiIllustriousSDXL_v170.safetensors"
$out  = "$repo\impact\impact_${Tag}_full.json"
$log  = "$repo\impact\sweep_${Tag}.log"
$env:PYTHONIOENCODING = "utf-8"
& $py -u "$repo\sdxl\diag_impact_controlnet.py" --base $base --controlnet $ControlNet --out $out --comfy_path "$repo\ComfyUI-master" --steps $Steps --full --resume $out *> $log
'@ | Set-Content -Path $runner -Encoding UTF8

$p = Start-Process -FilePath "powershell.exe" `
  -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$runner,"-Tag",$Tag,"-ControlNet",$ControlNet) `
  -WindowStyle Hidden -PassThru
```

The log is written directly by the detached process, so progress is readable with `Get-Content` at any time, and the process survives anything that happens to the agent session.

### 9.2 Keep the weight clone and the reconstruction off the GPU

The measurement perturbs one weight tensor at a time. Doing the clone and the ConvRot INT8 reconstruction on the GPU adds a transient peak on top of the base model (6.5 GB), the ControlNet (2.3 GB) and the sampler's activations (8.28 GB resident on a 16 GB card). Performing both on the **CPU** removes that peak entirely; only the single perturbed tensor is moved to the GPU, and only for the duration of the injection:

```python
w0 = m.weight.data.detach().to("cpu", copy=True)
try:
    rec = convrot_int8_recon(w0, gs)          # CPU
    m.weight.data.copy_(rec.to(w0.dtype).to(w_dev))
    del rec
    torch.cuda.empty_cache()
    xt = run_traj()
    imp = sum(rel_mse(xt[j], x_ref[j]) for j in range(len(x_ref))) / len(x_ref)
    del xt
finally:
    m.weight.data.copy_(w0.to(w_dev))
    del w0
    torch.cuda.empty_cache()
```

With this in place the sweep held a steady **10.0-10.2 GB of VRAM** for 400 consecutive layers, and the per-step latents are copied to CPU as contiguous `float32` tensors so the trajectory history itself does not sit on the card.

### 9.3 Checkpoint every 10 layers and support resume

Even with 9.1 and 9.2 in place, a long sweep should not be able to lose its work:

- the partial result is written to the impact JSON **every 10 layers**;
- `--resume <json>` loads the already-measured layers and skips them, so a run that dies resumes from the last checkpoint instead of restarting;
- the reference runs in this repository were executed as **single uninterrupted runs** (400 layers, 0 stoppages), but the resume path remains available for environments where a sweep still has to be split.

### 9.4 Reference run characteristics

| | Checkpoint A | Checkpoint B | U (union-pro-max) |
| :--- | :--- | :--- | :--- |
| Layers measured | 400 | 400 | 407 |
| Stoppages | **0** | **0** | **0** |
| Wall time | 62 min | 71 min | 82 min |
| Steady VRAM | ~10.0 GB | ~10.1 GB | ~9.1 GB |
| Peak RSS | ~3.7 GB | ~3.7 GB | ~3.4 GB |
| Speed | ~10 s/layer | ~10 s/layer | ~12 s/layer |

All three completed as single uninterrupted runs. The wall-time differences track the number of layers and the GPU contention present during each run (other servers were resident on the same card in some runs); no run was ever interrupted.

### 9.5 Reproducibility of the measurement

Re-running the full sweep on the same checkpoint reproduces the distribution, not the individual values:

| | Checkpoint A (run 1 vs run 2) | Checkpoint B (run 1 vs run 2) |
| :--- | :--- | :--- |
| Layers | 400 = 400 | 400 = 400 |
| Layers with a differing value | 255 (all at the 1e-4 level) | 277 (all at the 1e-4 level) |
| median drift | 0.1478 vs 0.1478 | 0.1119 vs 0.1117 |
| max drift | 0.7141 vs 0.7141 | 1.2090 vs 1.2090 |
| **drift > 0.15** | 190 vs 190 | 118 vs 117 |
| **drift > 0.20** | 86 vs 86 | 44 vs 44 |

The residual per-layer differences come from non-deterministic GPU reductions (cuDNN / TF32 atomics), not from the method. The **population statistics and the threshold decision are stable**: the >0.20 counts are identical and the >0.15 counts differ by at most one layer.

---

## 10. Practical Notes and Pitfalls

**Read the safetensors header correctly.** The data of a tensor starts at `8 + header_length + data_offset`, not at `data_offset`. Reading at the wrong offset yields garbage values that look like plausible but meaningless statistics (e.g. a kurtosis of 9,494 for a tensor whose true kurtosis is 21). Use `safetensors.safe_open(...).get_tensor(...)` or account for the header length.

**Do not use static statistics as the protection criterion.** Section 5.1 shows the disagreement directly. Static statistics may be used as a cheap *screen* to choose which candidates are worth measuring, but the decision itself must come from the measurement.

**Do not reuse another checkpoint's protection set.** Three checkpoints of the same architecture require **190 / 117 / 74** protected layers at the same 0.15 threshold - a factor of 2.6 between the most and the least (Section 5.3). One of them additionally has a single layer with drift 3.734, which the other two do not have at all.

**The Conv2d scope is a hard requirement, not a design decision.** ComfyUI's INT8 load path attaches a QuantizedTensor to 2D Linear layers only; Conv2d layers stay FP16. Quantizing Conv2d produced a mixed model that output NaN.

**Write the pack under the runtime (comfy) key names.** The single most damaging pitfall: ComfyUI remaps the checkpoint's diffusers names to comfy names, so a pack written under the diffusers names loses its `.weight_scale` / `.comfy_quant` companions in the remap and the runtime reads a bare int8 tensor - the model outputs NaN (a black image). Measured on the broken pack: 340 of 346 quantized modules were absent from the runtime graph. Always build the pack from the runtime module graph (Section 4.4).

**Verify the load path.** Load the pack through the `HSWQ ControlNet Loader` path (`apply_comfy_quant_int8_patches` + `MixedPrecisionOps(int8_tensorwise)`), not with a bare `comfy.sd.load_diffusion_model_state_dict(sd, model_options={})` - the bare path does not arm the ConvRot patch and makes even a correct pack look broken.

---

## 11. Summary

| Aspect | Choice |
| :--- | :--- |
| Protection criterion | Measured trajectory drift of the real ControlNet in a real ComfyUI generation |
| Measurement scope | **Full sweep of every eligible layer, one uninterrupted run per checkpoint (400 / 400 / 407)** |
| Perturbation | ConvRot INT8 reconstruction (Hadamard rotation + per-channel INT8) |
| Metric | Mean relative MSE of the per-step latent against the pristine trajectory |
| Sampler | Production sampler (`dpmpp_2m` / `karras` / CFG 7.0), 20 steps, seed 42 |
| Threshold | **0.15** |
| Name mapping | Pack written under the **runtime (comfy) names** - the checkpoint's diffusers names are remapped by ComfyUI |
| Quantized scope | 2D Linear with input width a multiple of 256, minus the protected set |
| A pack | 2386.2 MB -> 1718.6 MB, 196 Linear in INT8, 190 protected, gate cos 0.999875 |
| B pack | 2386.2 MB -> 1533.2 MB, 257 Linear in INT8, 117 protected, gate cos 1.000044 |
| U pack | 2396.9 MB -> 1443.5 MB, 298 Linear in INT8, 74 protected, gate cos 0.999646 |
| Pack layout | `int8_tensorwise` with `comfy_quant` stamps and `_quantization_metadata`, under runtime key names |
| Operational requirement | Detached process, CPU-side weight clone + reconstruction, 10-layer checkpoint with resume |

The essential point: for a ControlNet, the set of layers that must stay in FP16 is a property of *that checkpoint*, discoverable only by measuring *that checkpoint* in the pipeline it will actually run in. The measurement itself is a full sweep, and the engineering around it - detachment, CPU-side reconstruction, checkpointing - is what makes the full sweep a routine operation rather than a multi-hour, multi-restart ordeal.


---

## 10. The union-pro-max re-measurement (the injection fix)

The trajectory-impact sweep of `controlnet-union-pro-max-sdxl-1.0` was re-run
after the injection fix. The fix matters here in the same way as elsewhere:
the measurement injects the ConvRot INT8 reconstruction into the module, and
the reconstruction has to be in the **original** frame (`(q*s) H^T - W`), not
the rotated one — the pack stores the rotated weight and the kernel rotates the
**activation**.

The 407-layer sweep was re-run in one uninterrupted pass (97 min). The
ConvRot-eligible layers (the 364 the measurement can perturb) moved:

| | the earlier measurement | the re-measured |
| :--- | :--- | :--- |
| the min | 6.15e-04 | 3.30e-04 |
| the median | 5.49e-02 | **1.64e-03** |
| the p90 | 1.64e-01 | 2.76e-03 |
| the max | 4.38e-01 | **6.01e-03** |

The 43 Conv2d layers are unchanged by the fix (they are not perturbed), which
is why the global max still reads 3.73 (that is `input_hint_block.2`, a Conv2d).

Because the scale changed by ~33×, the old `0.15` threshold selects a
different set on the corrected values. The shipped pack is re-derived at
**0.005**, which protects 41 layers:

| the candidate | the INT8 | the protected | cos | relMSE | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| the earlier pack (0.15 on the defective values) | 298 | 74 | 0.999961 | 7.67e-06 | 1443.5 MB |
| t=0.002 | 215 | 154 | 0.999962 | 7.64e-06 | 1764.0 MB |
| t=0.0025 | 277 | 92 | 0.999961 | 7.63e-06 | 1539.3 MB |
| t=0.003 | 310 | 59 | 0.999961 | 7.71e-06 | 1421.2 MB |
| t=0.004 | 322 | 47 | 0.999961 | 7.72e-06 | 1386.5 MB |
| **t=0.005 (adopted)** | **328** | **41** | **0.999961** | **7.74e-06** | **1368.8 MB** |

All candidates sit at the same cosine (0.999961) — the SDXL ControlNet's
trajectory is not sensitive to which of these layers is protected, so the
smallest pack that holds the gate was adopted. The earlier pack and the
rebuilt pack are numerically indistinguishable on the gate, and the rebuilt
one is 75 MB smaller.

The corrected drift's ranking (the ConvRot-eligible layers) is dominated by
`input_blocks.*.transformer_blocks.*` and `middle_block.*` — the transformer
blocks — while the earlier (defective) ranking was dominated by the same
families for the wrong reason.


---

## 11. The anytest4 A re-measurement (the injection fix)

`CN-anytest4_illustrious2_A` was re-measured after the injection fix, in one
uninterrupted pass (103 min, 400 / 400).

The 362 ConvRot-eligible layers moved:

| | the earlier measurement | the re-measured |
| :--- | :--- | :--- |
| the min | 1.22e-02 | 2.71e-03 |
| the median | 1.44e-01 | **1.06e-02** |
| the p90 | 2.29e-01 | 1.54e-02 |
| the max | 3.54e-01 | **3.38e-02** |

The 38 Conv2d layers are unchanged by the fix. The corrected ranking's top is
`input_blocks.7.1.transformer_blocks.4.attn1.to_v` (3.38e-02),
`input_blocks.7.1.proj_in` (3.38e-02) and
`input_blocks.4.1.transformer_blocks.1.attn1.to_v` (3.19e-02).

Unlike the union-pro-max checkpoint, **A's trajectory is sensitive to the
protection set**: the candidates are not flat.

| the candidate | the INT8 | the protected | cos | relMSE | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| the earlier pack (0.15 on the defective values) | 196 | 190 | 0.999841 | 6.78e-04 | 1718.6 MB |
| t=0.030 | 326 | 36 | 0.999626 | 1.11e-03 | 1359.1 MB |
| t=0.020 | 324 | 38 | 0.999600 | 1.16e-03 | 1366.9 MB |
| t=0.015 | 297 | 65 | 0.999626 | 1.13e-03 | 1479.4 MB |
| t=0.012 | 220 | 142 | 0.999779 | 7.97e-04 | 1717.2 MB |
| t=0.011 | 186 | 176 | 0.999824 | 7.08e-04 | 1825.8 MB |
| **t=0.010 (adopted)** | **140** | **222** | **0.999879** | **5.96e-04** | **1959.5 MB** |

There is a clear knee between 0.012 and 0.010: the cosine steps from 0.999779
to 0.999879 and the relMSE from 7.97e-04 to 5.96e-04. The adopted pack is the
only candidate that **beats the earlier pack on quality** (0.999879 against
0.999841, relMSE 5.96e-04 against 6.78e-04) — at 241 MB more, because it
protects 222 layers instead of 190, chosen on the corrected ranking.

The corrected values sit ~14× lower than the defective ones, so the earlier
`0.15` threshold would protect almost nothing on the corrected scale; the
protection set has to be re-derived, and the knee in the measured gate is what
sets it.
