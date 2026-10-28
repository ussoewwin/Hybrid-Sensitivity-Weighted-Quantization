# Diag Impact → Forward Top-K Hybrid Quantization (SD1.5, v1.1) — Technical Guide

**Document version:** 1.0  
**Date:** 2026-10-28  
**Scope:** the SD1.5 ConvRot INT8 trajectory-impact hybrid quantization, as implemented by:

| Stage | Script / Module | ComfyUI Node |
|---|---|---|
| Trajectory impact (diag) | `sd1.5/diagimpact_sd15.py` | `HSWQ Diag Impact (ConvRot INT8)` (`model_type="SD1.5"`) |
| Impact profile loader | — | `HSWQ Load Diag Impact Json` |
| Hybrid pack builder | `sd1.5/quantize_sd1.5_hswq_v1.1.py` | `HSWQ Quantize ConvRot INT8` (`model_type="SD1.5"`) |
| Acceptance gate | `benchmark/sd1.5_int8_traj_compare.py` | Embedded Step 3 in `HSWQ Quantize ConvRot INT8` |
| Loader verification | `sd1.5/verify_sd15_loader.py` | `ComfyUI-HSWQ-Loader-and-Tools` |

This document is **SD1.5-specific**. The SDXL reverse hybrid counterpart is documented in [Diag → Reverse Hybrid Quantization (SDXL, v1.1) — Technical Guide](Diag_Reverse_SDXL_v1.1_Technical_Guide.md), the Z-Image Hybrid NVFP4 variant in [How to quantize Z Image - Hybrid NVFP4](How%20to%20quantize%20Z%20Image%20-%20Hybrid%20NVFP4.md), and the universal sensitivity theory in [Trajectory-Sensitivity Impact Ranking — Technical Guide](diag_impact_trajectory_sensitivity_technical_guide.md).

---

## 1. What this pipeline is (one paragraph)

This pipeline is the **trajectory-impact-driven forward top-$K$ hybrid quantizer** for Stable Diffusion 1.5 UNet architectures. Unlike static weight-space selectors (such as the retired V1.0 which relied on kurtosis, outlier ratios, and histogram MSE), this method determines the FP16 protection set dynamically by measuring the actual drift that each layer's ConvRot INT8 reconstruction causes on the **terminal latent state $x_T$** of a full 25-step denoising trajectory. Every ConvRot-eligible 2D `Linear` and 4D `Conv2d` layer (282 candidates on standard SD1.5 UNets) is perturbed in isolation within the original frame (`(q * s) @ H`); the resulting terminal divergence ranks the layers in descending impact order. The **top-$K$ highest-impact layers are preserved in pristine FP16**, while the remaining $282 - K$ layers are converted to **FULL ConvRot INT8** (Hadamard group size 256/64/16/4, per-output-channel FP32 scale, `.comfy_quant` metadata) with optional systematic INT8 bias correction. Checkpoint-embedded CLIP text encoders pass through 100% untouched in original FP16, and the embedded VAE is completely stripped, producing a compact ~1.16 GB hybrid pack that achieves a 25-seed mean trajectory cosine of **0.97622** (compared to native all-INT8 at 0.96265) with **0/25 bifurcated seeds**.

---

## 2. Notation & Definitions

| Term | Mathematical Definition / Meaning |
|---|---|
| **candidate** | A ConvRot-eligible `Linear` (2D) or `Conv2d` (4D) module of the UNet where input channels $C_{\text{in}} \ge 4$ and $C_{\text{in}}$ has a power-of-4 divisor $\le 256$ (§5). Exactly 282 layers in SD1.5 UNet. |
| **terminal latent $x_T$** | The latent tensor obtained at step $T = 25$ under explicit Euler / DDPM integration, representing the actual input to the VAE decoder. |
| **pristine trajectory $x_{\text{ref}, T}$** | The terminal latent produced when all UNet weights remain in unquantized FP16 precision under fixed noise seed $s$. |
| **perturbed latent $x_T(l)$** | The terminal latent produced when candidate layer $l$ is replaced by its ConvRot INT8 reconstruction in the original coordinate frame, holding all other 281 layers pristine. |
| **trajectory impact $\text{impact}(l)$** | Terminal relative MSE divergence: $\frac{\|x_T(l) - x_{\text{ref}, T}\|_F^2}{\|x_{\text{ref}, T}\|_F^2}$, averaged across calibration seeds ($s \in \{42, 137\}$). |
| **top-$K$ protected set** | The $K$ candidate layers exhibiting the highest trajectory impact (or impact weighted by parameter byte cost), retained in original FP16 precision. |
| **converted set** | The remaining $282 - K$ candidate layers quantized to ConvRot INT8. |
| **byte cost $C(l)$** | Memory footprint of layer $l$'s weight tensor: $C(l) = \text{numel}(W_l) \times 2\text{ bytes}$ (FP16). |
| **systematic INT8 bias shift $\delta b$** | First-order output expectation drift: $\delta b = E[(W_q - W)x] \approx (W_q - W) \cdot E[x]$, folded into layer bias as $b \leftarrow b - \delta b$. |
| **gate** | Deterministic 25-seed latent trajectory comparison against pristine FP16 baseline across 25 steps at $512\times 512$ resolution. Production pass requires mean final cosine $\ge 0.97$ and $0/25$ bifurcations. |

---

## 3. Why Final-Step $x_T$ Divergence Instead of Step-Averaging (SD1.5 vs SDXL vs Z-Image)

Although all three architectures utilize the `diagimpact` conceptual lineage, **the numerical processing and trajectory metric in SD1.5 differ fundamentally from both SDXL and Z-Image**. SDXL and Z-Image compute an **arithmetic step-average** of relative MSE across all trajectory steps $t \in [1, T]$:

$$\text{impact}_{\text{SDXL/ZI}}(l) = \frac{1}{T} \sum_{t=1}^T \frac{\|x_t(l) - x_{\text{ref}, t}\|_F^2}{\|x_{\text{ref}, t}\|_F^2}$$

In stark contrast, **SD1.5 completely discards step-averaging** and evaluates exclusively the **terminal latent state divergence at step $T=25$**:

$$\text{impact}_{\text{SD1.5}}(l) = \frac{\|x_T(l) - x_{\text{ref}, T}\|_F^2}{\|x_{\text{ref}, T}\|_F^2} \quad \text{or} \quad 1 - \cos\left(x_T(l), x_{\text{ref}, T}\right)$$

This architectural decision is rooted in empirical validation and the unique dynamics of the SD1.5 UNet.

### 3.1 The SD1.5 Single-Stream UNet & Skip-Connection Dominance

The SD1.5 UNet is a compact, single-stream architecture featuring direct long skip connections that bypass intermediate transformations and connect early downsampling blocks directly into late upsampling blocks (`output_blocks`):

```
x_t ──> input_blocks (Encoder) ─────────────── Long Skip ───────────────> output_blocks (Decoder) ──> eps_t
                 │                                                                 ▲
                 └──> middle_block (Bottleneck) ───────────────────────────────────┘
```

1. **Decoder Skip Connections Inject Error Directly into the Terminal State:**  
   Modules situated on the decoder skip lines—such as `output_blocks.6.0.skip_connection`, `output_blocks.9.0.skip_connection`, and the terminal projection `out.2`—inject their numerical error directly into the output latent. There are no subsequent non-linear layers or contracting blocks downstream to attenuate, filter, or absorb this error. Any perturbation introduced here propagates unmitigated into the final denoised latent $x_T$.
2. **Encoder Errors Undergo Downstream Contraction:**  
   Perturbations introduced in early encoder blocks (e.g. `input_blocks.1.1.proj_in`) affect the intermediate feature representations significantly at early timesteps (high noise ratio). However, as the latent passes through multiple downstream cross-attention blocks, ResNet blocks, and the bottleneck, the contracting nature of the residual maps attenuates a substantial fraction of this initial error before it ever reaches the terminal latent.

### 3.2 The Dilution Effect of Arithmetic Step-Averaging

If one applies arithmetic step-averaging $\frac{1}{T}\sum_{t=1}^T \text{rel\_mse}(x_t, x_{\text{ref}, t})$ to SD1.5, a severe distortion occurs:

* **Early High-Noise Amplification:** At early timesteps ($t \in [1, 10]$), where $\sigma_t \approx 1.0$, the variance of intermediate activations is maximal. An encoder perturbation produces an immediate, high-magnitude intermediate deviation in $x_t$.
* **Sum Domination by Transient Errors:** The arithmetic average is overwhelmingly dominated by these early timesteps, masking whether that error actually survived to corrupt the final image.
* **Dilution of Terminal Skip Divergence:** A decoder skip error that occurs primarily in the final refinement steps ($t \in [18, 25]$) receives an average weight of only $1/25$ per step, despite directly degrading the final visual composition, fine textures, and facial geometry.

By evaluating the terminal latent $x_T$, the metric measures the **accumulated, uncancelled survival error** that actually reaches the VAE decoder.

### 3.3 Trajectory vs Single-Forward vs Step-Average Empirical Ranking Comparison

The distinction is directly observable in empirical ranking data. When comparing single-forward-pass sensitivity (which mimics early-step sensitivity) against full-trajectory terminal evaluation, the two criteria disagree on **38 of the top 80 layers** (only a 42/80 overlap):

| Layer Module Name | Terminal Trajectory Rank ($x_T$) | Single-Forward / Early-Step Rank | Architectural Function |
| :--- | :---: | :---: | :--- |
| `output_blocks.6.0.skip_connection` | **1** | 5 | Decoder skip connection (direct latent injection) |
| `output_blocks.9.0.skip_connection` | **2** | 13 | Decoder skip connection (direct latent injection) |
| `out.2` | **3** | 22 | Terminal UNet output convolution |
| `output_blocks.11.1.proj_in` | **4** | 38 | Late decoder cross-attention input projection |
| `input_blocks.1.1.proj_in` | 58 | **1** | Early encoder projection (absorbed downstream) |
| `input_blocks.2.1.proj_in` | 55 | **2** | Early encoder projection (absorbed downstream) |
| `input_blocks.2.1.attn1.to_k` | 223 | 6 | Early encoder self-attention key projection |

Evaluating $x_T$ correctly identifies decoder skip lines and terminal convolutions as the absolute highest-priority protection targets. Step-averaging falsely elevates `input_blocks.1.1.proj_in` to rank 1, wasting valuable FP16 protection budget on layers whose quantization error is largely dissipated by the sampler.

### 3.4 Direct Mathematical Alignment with the 25-Seed Production Gate

The production acceptance gate evaluates:

$$\text{Final Cosine} = \frac{\langle x_T^{\text{FP16}}, x_T^{\text{INT8}} \rangle}{\|x_T^{\text{FP16}}\|_2 \|x_T^{\text{INT8}}\|_2} \ge 0.97$$

Evaluating $x_T$ during layer diagnosis ensures that the objective function optimized during quantization is mathematically identical to the instrument that judges product acceptance.

---

## 4. Forward Top-$K$ Protection vs SDXL Reverse Hybrid & Z-Image NVFP4

The three architectures implement fundamentally different hybrid construction directions based on their parameter count, error budgets, and initial baseline stability:

| Architectural Property | SDXL Reverse Hybrid (v1.1) | Z-Image Reverse Hybrid (NVFP4) | SD1.5 Forward Top-$K$ (v1.1) |
|---|---|---|---|
| **Base Baseline** | Pristine FP16 Checkpoint | Complete Native ConvRot INT8 | Pristine FP16 Checkpoint |
| **Static Predicate** | V3.1 Predicate (300 MiB budget, SVD $\times$ RMS) | None (Native INT8 already stable) | None (Static metrics fail on SD1.5) |
| **Candidate Pool** | 715 pool layers (788 candidates minus protected) | All eligible DiT blocks | All 282 ConvRot-eligible UNet layers |
| **Sorting Direction** | **Ascending** impact (lowest impact first) | **Ascending** impact (lowest impact first) | **Descending** impact (highest impact first) |
| **Quantization Direction** | Converts $K$ lowest-impact layers to INT8 | Converts $K$ lowest-impact layers to NVFP4 | Protects top-$K$ highest-impact layers in FP16 |
| **Primary Metric** | Step-averaged relative MSE ($\frac{1}{T}\sum_t \text{rel\_mse}_t$) | Step-averaged relative MSE | **Terminal latent state divergence ($x_T$)** |
| **Optimal Target** | $K = 670$ converted layers (out of 715 pool) | $K \approx 60\text{--}110$ converted layers | **$K = 80$ protected layers** (out of 282 candidates) |
| **Payload Footprint** | ~4.92 GB (FP16 base 6.94 GB) | ~60% of INT8 base | **~1.16 GB** (FP16 base 2.03 GB) |

### 4.1 Structural Mechanics

* **SDXL Reverse Hybrid:** In SDXL, a massive 300 MiB static predicate protects the most critical layers first. The remaining 715 pool layers exist in a low-error regime where single-layer perturbations are small. Converting the lowest-impact layers first (reverse direction) ensures that error accumulation begins at the absolute safest mathematical boundary.
* **SD1.5 Forward Top-$K$:** In SD1.5, static predicates completely fail: V1.0's static protection set yielded a score identical to native all-INT8, buying zero quality improvement. Furthermore, because SD1.5 UNet is compact (only ~860M parameters in the UNet), there is no room for a two-tier predicate + pool mechanism. The entire candidate set of 282 layers is diagnosed directly on the trajectory, and protection is applied forward to the top-$K$ most sensitive layers.

---

## 5. Candidate Enumeration & 4D Conv2d / 2D Linear Group Sizing

### 5.1 Mathematical Definition of ConvRot Candidates

Candidate modules are extracted from `model.diffusion_model.` by inspecting every named module:

```python
candidates = []
for name, module in model.named_modules():
    w = getattr(module, "weight", None)
    if w is None or not hasattr(w, "ndim"):
        continue
    if type(module).__name__ in ("Linear", "Conv2d") and w.ndim in (2, 4):
        gs = group_size_for(int(w.shape[1]), preferred=256)
        if gs is not None and int(w.shape[0]) % 4 == 0:
            candidates.append(name)
```

On a standard SD1.5 checkpoint, exactly **282 candidate layers** satisfy these criteria:
* **2D Linear Layers:** Attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`), cross-attention projections, and feed-forward layers (`ff.net.0.proj`, `ff.net.2`).
* **4D Conv2d Layers:** ResNet convolutions (`in_layers.2`, `out_layers.3`), downsamplers (`downsampler.conv`), upsamplers (`upsampler.conv`), and skip connections (`skip_connection`).

### 5.2 Hadamard Group Sizing ($256 \to 64 \to 16 \to 4$)

The normalized regular Hadamard matrix $H_{gs}$ requires that the group size $gs$ be a power of 4:

$$gs \in \{256, 64, 16, 4\}, \quad \log_4(gs) \in \mathbb{Z}^+$$

The helper `group_size_for(n, preferred=256)` selects the largest power-of-4 divisor of the input feature dimension $n = C_{\text{in}}$:

$$gs(n) = \max \left\{ g \in \{256, 64, 16, 4\} \;\middle|\; n \pmod g = 0 \right\}$$

If no divisor $\ge 4$ exists, the layer is marked unconvertible and kept at FP16.

### 5.3 Boundary Layers & Excluded Modules

Boundary and conditioning layers are strictly excluded from INT8 quantization:
* **Time Embeddings:** `time_embed.0.weight`, `time_embed.2.weight` (kept FP16; sensitive to step indexing).
* **Input Convolution:** `input_blocks.0.0.weight` (conv_in; $C_{\text{in}} = 4$, projects 4-channel latent noise).
* **Non-matmul Tensors:** Layer norms, group norms (`in_layers.0`), biases, and timestep schedule buffers remain in original FP16/FP32 precision.

---

## 6. The Measurement Machinery (`diagimpact_sd15.py`)

### 6.1 The Injection Kernel in the Original Frame ($(q \cdot s) @ H$)

Quantization-dequantization must occur **in the original coordinate frame** to reflect the exact error experienced by the runtime:

$$\hat{W} = \text{dequantize}(\text{quantize}(W)) = (q \cdot s) \cdot H$$

```python
def quant_dequant(w, gs):
    hh = build_hadamard(gs).to(device=w.device, dtype=w.dtype)
    if w.dim() == 2:
        of, inf = w.shape
        g = inf // gs
        rot = torch.matmul(w.view(of, g, gs), hh.T).reshape(w.shape)
        amax = rot.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
        s = (amax / 127.0).to(w.dtype)
        q = (rot / s).round().clamp(-127, 127)
        return torch.matmul((q * s).view(of, g, gs), hh).reshape(w.shape)
    if w.dim() == 4:
        oc, ic, kh, kw = w.shape
        flat = w.permute(0, 2, 3, 1).contiguous().view(-1, ic)
        g = ic // gs
        rot = torch.matmul(flat.view(-1, g, gs), hh.T).reshape(flat.shape)
        amax = rot.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
        s = (amax / 127.0).to(w.dtype)
        q = (rot / s).round().clamp(-127, 127)
        back = torch.matmul((q * s).view(-1, g, gs), hh).reshape(flat.shape)
        return back.view(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()
```

> [!CRITICAL]
> **Why the Original Frame Matters:**  
> The shipped safetensors pack stores the rotated weight $W_{\text{rot}} = W \cdot H^T$, and the runtime kernel rotates the incoming activation online ($x_{\text{rot}} = x \cdot H$). The mathematical operation executed is:
> $$y = x_{\text{rot}} \cdot W_{\text{rot}}^T = (x \cdot H) \cdot (W \cdot H^T)^T = x \cdot H \cdot H^T \cdot W^T = x \cdot W^T$$
> Therefore, the physical error entering the computation is $(q \cdot s) \cdot H - W$. Injecting in the rotated space without multiplying by $H$ measures the coordinate transformation rather than the quantization noise, completely inverting layer rankings.

### 6.2 Trajectory Denoising Loop & Explicit Euler / DDPM Formulation

The trajectory simulation uses a deterministic explicit Euler schedule matching SD1.5's 25-step production sampler:

```python
def run_trajectory(model, x0, ctx, steps):
    x = x0.clone()
    sig = torch.linspace(1.0, 0.0, steps + 1, device=DEV, dtype=DT)
    with torch.no_grad():
        for i in range(steps):
            t = (sig[i] * 999.0).reshape(1)
            eps = model(x, t, ctx)
            x = (x - eps * (sig[i] - sig[i + 1])).to(DT)
    return x
```

* `x0`: Fixed initial latent noise $z_0 \sim \mathcal{N}(0, I)$ generated on CPU via fixed seed ($1 \times 4 \times 64 \times 64$).
* `ctx`: Fixed pseudo-prompt conditioning vector ($1 \times 77 \times 768$).
* `steps`: Fixed at 25 (the production step count).

### 6.3 Metric Formulation ($x_T$ Relative MSE & Cosine)

For each candidate layer $l \in [1, 282]$:
1. Module weight $W_l$ is cloned.
2. Injected reconstruction $\hat{W}_l$ is loaded into `module.weight.data`.
3. Trajectory is executed for seeds $s \in \{42, 137\}$, obtaining $x_T(l, s)$.
4. Divergence against pristine reference $x_{\text{ref}, T}(s)$ is computed:

$$\text{rel\_mse}(l, s) = \frac{\|x_T(l, s) - x_{\text{ref}, T}(s)\|_2^2}{\|x_{\text{ref}, T}(s)\|_2^2}$$

5. Module weight is restored in a `finally` block: `module.weight.data = w0`.
6. Aggregate impact is the mean across seeds: $\text{impact}(l) = \frac{1}{|S|}\sum_{s \in S} \text{rel\_mse}(l, s)$.

### 6.4 The 4D Conv2d Channelwise Scale Invariant ($[O, 1, 1, 1]$ vs $[O, 1]$)

For a 4D `Conv2d` tensor $[C_{\text{out}}, C_{\text{in}}, K_H, K_W]$, the quantization scale must reduce across all non-output dimensions:

$$\text{scale} \in \mathbb{R}^{C_{\text{out}} \times 1 \times 1 \times 1}$$

A flattened 1D scale of shape $(C_{\text{out}},)$ does not broadcast identically during mixed-precision execution and induces severe NaN blowups across downstream layers. The pack builder enforces rank preservation: rank 2 for Linear, rank 4 for Conv2d.

### 6.5 JSON Contract & Output Artifacts

The output profile is saved to `<repo>/impact/impact_sd15_traj_<model_stem>.json`:

```json
{
 "impacts": {
  "input_blocks.1.0.in_layers.2": 0.0001423,
  "output_blocks.6.0.skip_connection": 0.0089142,
  "output_blocks.9.0.skip_connection": 0.0076219,
  "out.2": 0.0068415
 },
 "metric": "relmse",
 "seeds": [42, 137],
 "steps": 25
}
```

---

## 7. Parameter Byte-Cost Weighting & Top-$K$ Ranking

In SD1.5, candidate layers exhibit vast differences in parameter volume:
* A $1\times 1$ convolution or Linear projection with $320 \times 320$ has 102,400 parameters (204.8 KB FP16).
* A $3\times 3$ ResNet convolution with $1280 \times 1280 \times 3 \times 3$ has 14,745,600 parameters (29.49 MB FP16) — a **144-fold disparity**.

### 7.1 Marginal Sensitivity per Byte ($\text{Impact} / \text{ByteCost}$)

Allocating an FP16 protection slot to a 29.5 MB layer consumes a substantial portion of the memory savings budget. The pipeline supports weighting trajectory impact by parameter byte-cost:

$$\text{score}(l) = \frac{\text{impact}(l)}{C(l)} = \frac{\|x_T(l) - x_{\text{ref}, T}\|_2^2 / \|x_{\text{ref}, T}\|_2^2}{\text{numel}(W_l) \times 2}$$

This ensures that layers that receive protection yield maximum trajectory stability per megabyte of preserved FP16 payload.

---

## 8. The Pack Builder (`quantize_sd1.5_hswq_v1.1.py`)

### 8.1 Forward Top-$K$ Partitioning

Given the target $K$ (production standard $K=80$):
1. Load impact dictionary from JSON.
2. Sort candidates in **descending** order of impact.
3. Partition candidates:
   $$\text{Protected Set} = \{l_1, l_2, \dots, l_K\} \quad (\text{stored in FP16})$$
   $$\text{Converted Set} = \{l_{K+1}, \dots, l_{282}\} \quad (\text{converted to ConvRot INT8})$$

### 8.2 Untouched CLIP Text Encoder Pass-Through (`cond_stage_model.*`)

In SD1.5, the text encoder (`cond_stage_model.transformer.*`) is exceptionally compact (~246 MB in FP16) and highly sensitive to coordinate rotation. Quantizing the text encoder to INT8 produces zero measurable VRAM gain in ComfyUI while introducing lexical drift.

The builder copies every tensor starting with `cond_stage_model.` **100% byte-identical** to the output pack:
* Identical keys, dtypes (`torch.float16`), and shapes.
* Zero `.weight_scale` or `.comfy_quant` metadata artifacts attached to text encoder keys.

### 8.3 Embedded VAE Removal (`first_stage_model.*`)

Checkpoints typically package an embedded VAE (`first_stage_model.*`, ~335 MB). Modern production workflows universally decode latents using specialized external VAEs (such as `vae-ft-mse-840000-ema-pruned.safetensors`).

The builder unconditionally strips all `first_stage_model.*` keys from the output file, directly reducing file size by ~335 MB without affecting UNet generation fidelity.

### 8.4 Metadata & Serialization (`.comfy_quant`, `weight_scale`)

For each converted layer $m$:
1. $W_{\text{rot}} = \text{rotate\_linear}(W, H, gs)$ or $\text{rotate\_conv}(W, H, gs)$.
2. Quantize: $q = \text{clamp}(\text{round}(W_{\text{rot}} / s), -127, 127) \in \text{int8}$.
3. Scale: $s \in \text{float32}$ stored in rank $(O, 1)$ or $(O, 1, 1, 1)$.
4. Save to safetensors:
   * `model.diffusion_model.<m>.weight` $\leftarrow q$
   * `model.diffusion_model.<m>.weight_scale` $\leftarrow s$
   * `model.diffusion_model.<m>.comfy_quant` $\leftarrow$ UTF-8 JSON bytes:
     `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":gs}`
5. Global metadata:
   `_quantization_metadata`: `{"format_version":"1.0","layers":{...}}`

---

## 9. Systematic Bias Correction & The Optimal $K$ Non-Monotonicity Curve

### 9.1 Mathematical Formulation of Systematic INT8 Shift

Quantizing weights induces a systematic non-zero mean shift in layer activations:

$$\Delta y = (W_q - W) \cdot x$$

Taking the expectation over the calibration data distribution:

$$E[\Delta y] \approx (W_q - W) \cdot E[x]$$

To cancel this drift, the systematic shift $\delta b$ is folded into the layer's additive bias parameter:

$$b_{\text{corrected}} = b_{\text{original}} - \delta b$$

* **Linear (2D):** $\delta b[o] = \sum_i (W_{q,\text{orig}} - W_{\text{orig}})[o, i] \cdot \mu[i]$
* **Conv2d (4D):** $\delta b[o] = \sum_{i, k_h, k_w} (W_{q,\text{orig}} - W_{\text{orig}})[o, i, k_h, k_w] \cdot \mu[i]$

where $\mu = E[x]$ is the mean activation vector across calibration passes, and $W_{q,\text{orig}} = (q \cdot s) \cdot H$ is the dequantized weight in the unrotated frame.

### 9.2 Activation Mean Collection ($E[x]$ via Calibration)

When `--bias_correction` is enabled, the builder runs a 32-sample calibration pass ($512\times 512$, 25 steps, seed 42) through `StableDiffusionPipeline`, attaching forward hooks to capture per-input-channel activation means across all converted layers.

### 9.3 The Error Cancellation Trade-off

A critical architectural invariant governs bias correction:
* **INT8 Layers:** Receive bias correction, actively canceling first-order systematic drift ($E[\Delta y] \to 0$).
* **Protected FP16 Layers:** Kept at pristine weights, but receive **zero bias correction**.

### 9.4 Empirical Proof of Non-Monotonicity ($K=0 \to 40 \to 80 \to 100$)

The interaction between weight precision and bias correction produces a pronounced **non-monotonic quality curve** as $K$ increases:

| Pack Variant | Protected Count $K$ | Mean Final Cosine | Same-Image Seeds | Bifurcated Seeds | File Size | Architectural Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Native INT8** | 0 | 0.96265 | 13 / 25 | 0 / 25 | 1058.6 MB | All layers INT8; all layers receive bias correction. |
| **HSWQ $K=40$** | 40 | 0.96212 | 14 / 25 | 0 / 25 | ~1115 MB | **Buys nothing.** Quality matches native; top 40 layers alone do not separate trajectory sensitivity. |
| **HSWQ $K=80$** | **80** | **0.97622** | **18 / 25** | **0 / 25** | **1167.5 MB** | **Optimal inflection point.** +0.0136 gain over native; captures all high-sensitivity decoder skip connections. |
| **HSWQ $K=100$** | 100 | 0.97400 | 16 / 25 | 0 / 25 | 1193.5 MB | **Performance cliff.** Drops below $K=80$ while costing +26 MB more memory. |

```
Trajectory Cosine
   0.978 ─┐
   0.976 ─┤                  ★ K=80 (0.97622) - Optimal Peak
   0.974 ─┤                                      ▼ K=100 (0.97400) - Degradation
   0.972 ─┤
   0.970 ─┤
   0.968 ─┤
   0.966 ─┤
   0.964 ─┤
   0.962 ─┤   ▲ Native (0.96265)
          └───┴──────────────┴───────────────────┴───────────────> Protected Layers (K)
             K=0            K=40               K=80            K=100
```

> [!IMPORTANT]
> **Why More Protection Degrades Quality:**  
> Moving from $K=80$ to $K=100$ promotes 20 additional layers from INT8 to FP16. However, those 20 layers lose their systematic bias correction. Because these layers possess moderate-to-low trajectory sensitivity, the gain from preserving their weight precision is smaller than the loss incurred by stripping away their systematic error cancellation. Past $K=80$, promoting layers actively harms model fidelity.

---

## 10. ComfyUI In-Workflow Integration (Zero Terminal Commands)

SD1.5 HSWQ v1.1 is fully integrated into ComfyUI custom nodes (`comfyui_nodes/hswq_sd15_quantize.py`), executing the entire 3-step pipeline in-process without manual CLI execution.

### 10.1 `HSWQ Diag Impact (ConvRot INT8)` (Step 1)
* Input: Connected `model` (SD1.5 UNet) and `clip`.
* Parameters: `model_type = "SD1.5"`, `steps = 25`, `seeds = "42,137"`.
* Operation: Diagnoses all 282 ConvRot-eligible UNet layers in-process (~12 min on modern GPU) and outputs `impact_json` file path.

### 10.2 `HSWQ Load Diag Impact Json` & `ImpactSwitch`
* When testing multiple values of $K$ or re-quantizing an existing model, `HSWQ Load Diag Impact Json` lists previously generated `impact_sd15_traj_*.json` files from `<repo>/impact/` in a dropdown.
* An `ImpactSwitch` (`Switch (Any)`) routes either the live diagnosis (Input 1) or precomputed profile (Input 2) to the quantizer.

### 10.3 `HSWQ Quantize ConvRot INT8` (`model_type = "SD1.5"`) (Step 2 & Step 3)
* Consumes `model`, `clip`, `vae`, and `impact_json`.
* Executes Step 2 (Top-$K$ FP16 protection, ConvRot INT8 packing, VAE stripping, CLIP pass-through).
* Immediately triggers Step 3 (Deterministic 25-seed trajectory gate) and streams formatted evaluation tables directly to a `ShowText` node.

---

## 11. The Deterministic 25-Seed Trajectory Gate

Acceptance is judged by `benchmark/sd1.5_int8_traj_compare.py`.

### 11.1 Hardcoded 25-Seed List & Latent Initialization

Seeds are hardcoded to eliminate selection bias:

```
42, 137, 849, 2024, 7391, 18429, 53082, 149206, 382715, 826401,
1938502, 4710928, 8391642, 15820493, 36192847, 71058294, 128491703,
285039184, 491730285, 762019483, 938174026, 1409285713, 2683910547,
3851729406, 4195820371
```

Both FP16 baseline and INT8 pack are initialized with identical CPU-generated noise tensors ($1 \times 4 \times 64 \times 64$) per seed.

### 11.2 Bifurcation Detection Threshold ($\Delta \cos > 0.05$)

Trajectory bifurcation occurs when a quantized model suddenly jumps into a different attractor basin:

$$\Delta \cos_i = \cos(x_{i-1}^{\text{FP16}}, x_{i-1}^{\text{INT8}}) - \cos(x_i^{\text{FP16}}, x_i^{\text{INT8}})$$

$$\text{Bifurcated} \iff \max_{i \in [1, T]} (\Delta \cos_i) > 0.05$$

A bifurcated seed generates a completely different image composition (catastrophic failure), whereas gradual drift ($\Delta \cos_i \le 0.05$) preserves overall semantic content.

### 11.3 Production Gate Criteria

| Metric | Production Threshold |
| :--- | :--- |
| **Mean Final Latent Cosine** | $\ge \mathbf{0.9700}$ (Reference achieves $0.9762$) |
| **Bifurcated Seeds** | $\mathbf{0 / 25}$ ($0.0\%$ tolerance) |
| **Same-Image Verdict Rate** ($\cos \ge 0.98$) | $\ge \mathbf{16 / 25}$ |

---

## 12. Empirical Benchmark Matrix (7 Models / 175 Seeds)

Evaluated across 7 standard SD1.5 checkpoints (175 total trajectory trials):

### 12.1 Cross-Model Summary Table

| Model Checkpoint | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | $\Delta$ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | HSWQ Speedup (↑) | Winner |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.5xxmix9realistic_v40 0.5BRA7** | 1on k80 | **0.97542** | 0.96265 | **+0.01277** | **1.0432** | 1.4793 | **0/25 (0%)** | **+23.2%** | **HSWQ** |
| **0.5xxmix9realistic_v40+0.5chilledReGenericV3** | 1on k80 | **0.95362** | 0.92644 | **+0.02719** | **1.3338** | 2.1379 | **0/25 (0%)** | **+20.1%** | **HSWQ** |
| **0.5yayoiMix_v25+0.5againmix_v22** | 1on k80 | **0.97642** | 0.96010 | **+0.01632** | **1.0225** | 1.7696 | **0/25 (0%)** | **+38.6%** | **HSWQ** |
| **0.5yayoiMix_v25+0.5voidnoisecore_r1733** | 1on k80 | 0.97461 | **0.97518** | −0.00057 | **0.7950** | 0.8165 | **0/25 (0%)** | **+35.1%** | **Native** |
| **0.5yayoiMix_v25+0.5xxmix9realistic_v40** | 1on k80 | **0.96321** | 0.94839 | **+0.01483** | **1.1880** | 1.8368 | **0/25 (0%)** | **+31.3%** | **HSWQ** |
| **0.7_0.6hassaku+0.4dreamshaper+0.3anylora** | 1on k80 | **0.98061** | 0.95456 | **+0.02605** | **0.6839** | 1.7171 | **0/25 (0%)** | **+15.1%** | **HSWQ** |
| **0.7_0.7hassaku+0.3dreamshaper_8_0.3anylora** | 1on k82 | **0.96012** | 0.94130 | **+0.01882** | **1.2980** | 1.9021 | **0/25 (0%)** | **+22.6%** | **HSWQ** |
| **Architecture Family Average** | — | **0.96914** | 0.95266 | **+0.01648** | **1.0521** | 1.6656 | **0/175 (0.0%)** | **+26.6%** | **HSWQ (6/7)** |

* **HSWQ Outperformed Native on 6 out of 7 models**, achieving up to **+0.02719** cosine improvement and **38% latent error reduction**.
* **Zero Bifurcations:** 0 out of 175 seeds suffered trajectory bifurcation under HSWQ.
* **Inference Acceleration:** HSWQ delivered an average **+26.6% speedup** over FP16 baseline due to accelerated fused INT8 convolution execution.

---

## 13. Runtime Loader Architecture & Thread-Local Invariants

### 13.1 `comfy_kitchen` Limitations (Missing INT8 Conv2d Kernel)

ComfyUI's standard `comfy_kitchen` backend implements INT8 GEMM operations for 2D `Linear` layers, but **contains zero kernel implementation for INT8 4D `Conv2d`**. Attempting to load an SD1.5 INT8 pack through ComfyUI's default loader executes Conv2d layers in raw integer mode or unpatched float cast, generating immediate NaNs.

### 13.2 `_patch_ops_decode_and_conv()` & Thread-Local Storage (`_int8_quant_conv_tls`)

The HSWQ runtime patches the execution graph via `_patch_ops_decode_and_conv()` in `ComfyUI-HSWQ-Loader-and-Tools`:
* Intercepts `torch.nn.functional.conv2d` calls.
* Detects when a module's weight is tagged with `.comfy_quant` containing `"convrot": true`.
* Dynamically dequantizes and unrotates the 4D weight tensor in the execution stream.
* Thread-local variable `_int8_quant_conv_tls` gates the execution scope.

### 13.3 Vendored Patch Module Traps & Prestartup Script Requirements

ComfyUI loads custom nodes dynamically. If multiple copies of the patch module exist across custom node directories, each maintains an isolated thread-local storage object. Calling `load_unet_hswq_weight_dtype` ensures that all vendored thread-local flags are synchronized simultaneously before model execution begins.

---

## 14. Forbidden Mistakes & Operational Contracts

| Forbidden Action | Technical Consequence | Enforced Remediation |
|---|---|---|
| **Using Arithmetic Step-Averaging on SD1.5** | Over-weights early encoder blocks; under-weights critical decoder skip lines (`output_blocks.6.0.skip_connection`). | Evaluate exclusively the terminal latent state divergence $x_T$ at step 25 (§3). |
| **Injecting Perturbations in Rotated Space ($q \cdot s$)** | Measures the Hadamard rotation matrix transformation instead of quantization noise, completely inverting layer rankings. | Always inject in the original frame: $(q \cdot s) \cdot H$ (§6.1). |
| **Quantizing CLIP Text Encoder on SD1.5** | Destabilizes prompt conditioning semantics for zero meaningful VRAM savings. | Enforce 100% byte-identical pass-through of all `cond_stage_model.*` tensors (§8.2). |
| **Retaining Embedded VAE (`first_stage_model.*`)** | Bloats safetensors pack size by ~335 MB with redundant weights. | Unconditionally drop `first_stage_model.*` at pack serialization (§8.3). |
| **Treating $K$ as Monotonic (Pushing Past $K=80$)** | Layers lose systematic INT8 bias correction; overall quality drops (0.97622 $\to$ 0.97400 at $K=100$). | Measure $K$ empirically; adopt $K=80$ as the optimal balance point (§9.4). |
| **Storing 4D Conv2d Scale as Flat $(O,)$ Vector** | Breaks PyTorch broadcasting semantics across $(O, 1, 1, 1)$, triggering NaNs during Conv2d execution. | Enforce rank-4 scale tensor $(O, 1, 1, 1)$ for all 4D weights (§6.4). |
| **Using Subprocess Terminal Commands (`run_command`)** | Triggers interactive UI approval prompts, disrupting headless execution. | Drive all quantization, testing, and Git operations via native MCP tools (§10). |

---

## 15. Key Formulas & Function Map

| Concept | Mathematical Formula / Implementation | Primary Function / Script |
|---|---|---|
| **Regular Hadamard Matrix** | $H_{4} = \begin{bmatrix} 1 & 1 & 1 & -1 \\ 1 & 1 & -1 & 1 \\ 1 & -1 & 1 & 1 \\ -1 & 1 & 1 & 1 \end{bmatrix}, \quad H_{gs} = \frac{1}{\sqrt{gs}} \bigotimes H_4$ | `build_hadamard(gs)` in `quantize_sd1.5_hswq_v1.1.py` |
| **2D Linear Rotation** | $W_{\text{rot}} = W \cdot H^T \quad (g = C_{\text{in}} / gs)$ | `rotate_linear(w, h, gs)` |
| **4D Conv2d Rotation** | Flatten $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W} \to [C_{\text{out}} \cdot K_H \cdot K_W, C_{\text{in}}]$, rotate along $C_{\text{in}}$, permute back | `rotate_conv(w, h, gs)` |
| **Channelwise Scale** | $s = \frac{\max_{j} |W_{\text{rot}}[o, j]|}{127.0} \in \mathbb{R}^{C_{\text{out}} \times 1 \times 1 \times 1}$ | `quant_rowwise(w)` |
| **Terminal Divergence Metric** | $\text{impact}(l) = \frac{\|x_T(l) - x_{\text{ref}, T}\|_F^2}{\|x_{\text{ref}, T}\|_F^2}$ at $T = 25$ | `run_trajectory()` in `diagimpact_sd15.py` |
| **Systematic Bias Fold** | $\delta b = (W_{q,\text{orig}} - W_{\text{orig}}) \cdot E[x], \quad b \leftarrow b - \delta b$ | `bias_delta()` in `quantize_sd1.5_hswq_v1.1.py` |
| **Bifurcation Detector** | $\max_{i} (\cos(x_{i-1}^{\text{FP16}}, x_{i-1}^{\text{INT8}}) - \cos(x_i^{\text{FP16}}, x_i^{\text{INT8}})) > 0.05$ | `main()` in `benchmark/sd1.5_int8_traj_compare.py` |
| **ComfyUI In-Workflow Execution** | Step 1 Diag $\to$ Step 2 Pack Builder $\to$ Step 3 Trajectory Gate | `HSWQQuantizeConvRotINT8.quantize()` in `comfyui_nodes/hswq_sd15_quantize.py` |
