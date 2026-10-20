# ControlNet ConvRot INT8 Trajectory-Impact Protection (Technical Guide)

This document describes, in full, how a ControlNet checkpoint is converted to ConvRot INT8 with a **measured** FP16 protection set, using the HSWQ trajectory-sensitivity method.

The defining property of this method is that the protection set is **not** inferred from static weight statistics. It is measured by running the ControlNet inside a real ComfyUI generation pipeline and observing how much the denoising trajectory moves when a given layer is replaced by its ConvRot INT8 reconstruction. Layers whose measurement shows a large trajectory drift are kept in their original precision.

Everything below is backed by **completed full sweeps**: two SDXL ControlNet checkpoints, **all 400 ConvRot-eligible layers measured in a single uninterrupted run each**, with the measured numbers quoted verbatim.

| Checkpoint | Layers measured | Runs | Run time | Stoppages |
| :--- | :--- | :--- | :--- | :--- |
| `CN-anytest4_illustrious2_A.safetensors` | 400 / 400 | 1 | 62 min | 0 |
| `CN-anytest4_illustrious2_B.safetensors` | 400 / 400 | 1 | 71 min | 0 |

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

## 5. Why the Measurement Matters: Two Measured Checkpoints

Both checkpoints are SDXL ControlNets with the same architecture (400 ConvRot-eligible modules each).

### 5.1 The static screen and the trajectory disagree

The static screen (kurtosis of the weight distribution) and the measured trajectory drift rank the layers differently, and critically they disagree about the most dangerous layer:

| Layer (checkpoint key) | Static kurtosis | Measured drift (B) | Verdict |
| :--- | :--- | :--- | :--- |
| `down_blocks.0.resnets.1.conv1` | 2.4 (looks harmless) | **1.131** | **must stay FP16** |
| `down_blocks.0.resnets.0.conv1` | 1.1 | **0.716** | **must stay FP16** |
| `down_blocks.0.resnets.1.time_emb_proj` | 5.0 | 0.223 | must stay FP16 |
| `add_embedding.linear_1` | 21.1 (top static) | 0.105 | protected (trajectory) |
| `middle_block.1.transformer_blocks.4.attn2.to_v` | low | **0.018** | safe to quantize |

The Conv2d layers (`conv1`, runtime name `in_layers.2` / `out_layers.3`) look *mild* to a static kurtosis screen - their weight distribution is nearly Gaussian - yet they are the single most damaging layers in the whole ControlNet when quantized. A static-only selection would have quantized exactly the layers that break the trajectory.

### 5.2 Measured family statistics (all 400 layers, both checkpoints)

| Layer family | n | A mean | A max | B mean | B max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `input_blocks.*.0.in_layers.2` | 6 | **0.406** | 0.714 | **0.455** | 1.131 |
| `input_blocks.*.0.out_layers.3` | 6 | **0.396** | 0.549 | **0.433** | 1.209 |
| `input_blocks.*.1.proj_in` | 4 | 0.278 | 0.331 | 0.263 | 0.283 |
| `input_blocks.*.1.proj_out` | 4 | 0.272 | 0.305 | 0.257 | 0.354 |
| `input_blocks.*.0.emb_layers.1` | 6 | 0.235 | 0.289 | 0.178 | 0.223 |
| `zero_convs.*.0` | 9 | 0.219 | 0.511 | 0.177 | 0.443 |
| `input_blocks.*.1.transformer_blocks.*.attn2.to_q` | 24 | 0.212 | 0.354 | 0.168 | 0.241 |
| `middle_block.*.transformer_blocks.*.attn2.to_v` | 10 | **0.026** | 0.061 | **0.022** | 0.055 |
| `middle_block.*.transformer_blocks.*.attn2.to_out.0` | 10 | **0.030** | 0.069 | **0.028** | 0.066 |

The ordering is clear and not intuitive from the architecture alone: the **residual Conv2d path** (`in_layers.2` / `out_layers.3`) is by far the most sensitive part of a ControlNet, while the **cross-attention value path** (`attn2.to_v` / `attn2.to_out.0`) is the most tolerant - roughly **15x** less sensitive than the Conv2d path.

### 5.3 Whole-distribution comparison

| Statistic | Checkpoint A | Checkpoint B |
| :--- | :--- | :--- |
| Layers measured | 400 | 400 |
| min drift | 0.0122 | 0.0053 |
| median drift | **0.1478** | **0.1119** |
| p90 drift | 0.2512 | 0.2026 |
| max drift | 0.7141 | **1.2090** |
| drift > 0.10 | 356 | 233 |
| **drift > 0.15** | **190** | **117** |
| drift > 0.20 | 86 | 44 |
| drift > 0.25 | 40 | 16 |

The two checkpoints are **not interchangeable**: A is uniformly more sensitive (median 0.148 vs 0.112) while B has a heavier tail (max 1.209 vs 0.714). At the same threshold, A needs 190 protected layers and B needs 117 - a 62 % difference for the same architecture. Any approach that derives the protection set from the architecture alone, from a static score, or from another checkpoint's numbers would get one of them wrong.

### 5.4 The threshold choice

| Threshold | A protected | B protected |
| :--- | :--- | :--- |
| 0.10 | 356 | 233 |
| **0.15 (shipped)** | **190** | **117** |
| 0.20 | 86 | 44 |
| 0.25 | 40 | 16 |

The shipped packs use **0.15**.

### 5.5 The shipped packs

| Property | Checkpoint A | Checkpoint B |
| :--- | :--- | :--- |
| Source | `CN-anytest4_illustrious2_A.safetensors` (2.33 GB, BF16) | `CN-anytest4_illustrious2_B.safetensors` (2.33 GB, BF16) |
| Output | `CN-anytest4_illustrious2_A_hswq_convrot_int8.safetensors` | `CN-anytest4_illustrious2_B_hswq_convrot_int8_t015.safetensors` |
| Size | **1783.1 MB** (1.66 GB) | **1570.3 MB** (1.46 GB) |
| Protected (drift > 0.15) | **174 layers** (measured 190; 16 fall outside the mapped key set) | **105 layers** (measured 117; 12 outside) |
| ConvRot INT8 | 208 Linear + 18 Conv2d | 277 Linear + 18 Conv2d |
| INT8 payload | 720.9 MB | 934.4 MB |
| FP16/BF16 retained | 1060.2 MB | 633.2 MB |
| Load check | `load_torch_file` -> 1298 keys, 227 `comfy_quant` | `load_torch_file` -> 1436 keys, 296 `comfy_quant` |

---

## 6. Why a Full Sweep Was Necessary

An earlier focused measurement covered 29 layers (the static fat-tail group plus the 20 layers just below it) and produced a 13-layer protection set for checkpoint B.

The full sweep of 400 layers showed that this was **unsafe**. The layers that actually matter most - the residual Conv2d path, mean drift 0.43-0.46 - sit **outside** the static fat-tail group, because their weight distributions are near-Gaussian and therefore look mild to a kurtosis screen. The focused measurement did happen to catch the single worst layer (1.131), but it never measured the families at 0.15-0.35 (`proj_in` / `proj_out`, `skip_connection`, `zero_convs`, the `emb_layers.1` family), and the full sweep found **92 additional layers** in that band that the focused run had never looked at.

With a 0.15 threshold, the two answers differ by a factor of nine (13 vs 117 protected layers). A focused measurement can only be trusted for the band it actually covered; for a shipped artifact, the full sweep is the only defensible basis.

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

| | Checkpoint A | Checkpoint B |
| :--- | :--- | :--- |
| Layers measured | 400 | 400 |
| Stoppages | **0** | **0** |
| Wall time | 62 min | 71 min |
| Steady VRAM | ~10.0 GB | ~10.1 GB |
| Peak RSS | ~3.7 GB | ~3.7 GB |
| Speed | ~10 s/layer | ~10 s/layer |

Checkpoints B ran slower because the ComfyUI server and an `iopaint` server were resident on the same GPU during the run; the sweep still completed without interruption.

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

**Do not reuse another checkpoint's protection set.** Two checkpoints of the same architecture differ by 62 % in required protection at the same threshold (Section 5.3).

**The Conv2d scope is a design decision, not an accident.** Some ControlNet converters quantize only 2D Linear weights and leave every Conv2d in FP16. That is a valid choice, but it is a different pack from "everything ConvRot-eligible is quantized, with the measured exceptions". Both exist in the wild; they are not comparable by size, because the protected-set size differs.

**Verify the load path.** A pack with `comfy_quant` stamps is loaded natively by ComfyUI for ControlNet only when the layer set and the `comfy_quant` metadata are consistent. Always confirm with `comfy.utils.load_torch_file` that the pack loads and that the expected number of `comfy_quant` entries is present.

**A measured layer count can exceed the number of protected layers.** The impact JSON counts runtime modules; the pack protects checkpoint keys. A handful of runtime modules (for example the `middle_block_out` / `time_embed` entries) have no directly mappable checkpoint key, so 190 measured layers above the threshold map to 174 protected checkpoint keys for checkpoint A, and 117 to 105 for checkpoint B. The pack's own count is the authoritative one.

---

## 11. Summary

| Aspect | Choice |
| :--- | :--- |
| Protection criterion | Measured trajectory drift of the real ControlNet in a real ComfyUI generation |
| Measurement scope | **Full sweep of all 400 eligible layers, one uninterrupted run per checkpoint** |
| Perturbation | ConvRot INT8 reconstruction (Hadamard rotation + per-channel INT8) |
| Metric | Mean relative MSE of the per-step latent against the pristine trajectory |
| Sampler | Production sampler (`dpmpp_2m` / `karras` / CFG 7.0), 20 steps, seed 42 |
| Threshold | **0.15** |
| Name mapping | ComfyUI's own `unet_to_diffusers`, inverted - no hand-written table |
| Quantized scope | 2D Linear + 4D Conv2d, minus the protected set |
| A pack | 2.33 GB -> 1.66 GB, 208 Linear + 18 Conv2d in INT8, 174 protected |
| B pack | 2.33 GB -> 1.46 GB, 277 Linear + 18 Conv2d in INT8, 105 protected |
| Pack layout | `int8_tensorwise` with `comfy_quant` stamps and `_quantization_metadata` |
| Operational requirement | Detached process, CPU-side weight clone + reconstruction, 10-layer checkpoint with resume |

The essential point: for a ControlNet, the set of layers that must stay in FP16 is a property of *that checkpoint*, discoverable only by measuring *that checkpoint* in the pipeline it will actually run in. The measurement itself is a full sweep, and the engineering around it - detachment, CPU-side reconstruction, checkpointing - is what makes the full sweep a routine operation rather than a multi-hour, multi-restart ordeal.
