# How to Quantize SD1.5 (HSWQ ConvRot INT8, trajectory-impact selection)

This guide describes the **current** SD1.5 method: the FP16 protection set is
chosen from a **measured per-layer trajectory impact**, the same criterion
`Z_Image/diag_impact.py` uses.

> **ComfyUI node coverage is not documented here yet.** This guide covers the
> quantization method and its measurement. The loader / node workflow section
> will be added separately.

---

## Why the old selector failed

The previous `sd1.5/quantize_sd1.5_hswq_v1.0.py` chose its FP16 protection set
from **weight-space** statistics: kurtosis, outlier ratio, absmax, and a
weighted-histogram **MSE** of the quantized weight at the absmax pack point,
screened through a DualMonitor activation profile.

That answers "how much does this layer's *weight* distort", not "how much does
this layer move the *output*". An SD1.5 UNet is a deep series of blocks: a
layer's weight error is amplified, attenuated, or **cancelled** by everything
downstream. A weight-space MSE cannot see that propagation.

The measured consequence: V1.0's protection set scored **the same as native**
(no protection at all). The protection bought nothing. The defect was the
**criterion**, not the quantizer.

---

## The method: trajectory impact

### What is measured

For each ConvRot-eligible layer, one at a time:

1. hold every other layer at its pristine weight;
2. replace **this one layer's weight** by its ConvRot INT8 reconstruction **in
   the original frame** - `dequant(INT8(W @ H^T))` = `(q*s) @ H`;
3. run the **same fixed denoising trajectory** (25 steps, fixed seed, fixed
   latent, fixed context);
4. record the divergence from the pristine trajectory
   (`rel_mse(x_perturbed, x_pristine)`).

That number is the layer's **trajectory impact**: how far the sampler's output
moves when only this layer is quantized.

### Why the trajectory and not a single forward

A single forward measures a layer's sensitivity where the sampler happens to
call it once. A trajectory measures what the sampler experiences: the error
enters at one step, is carried in the latent, and is **amplified or cancelled**
by every later step.

The measured difference is not cosmetic. On this checkpoint the two criteria
disagree on **38 of the top 80 layers** (42/80 overlap):

| layer | trajectory rank | single-forward rank |
| :--- | ---: | ---: |
| `output_blocks.6.0.skip_connection` | **1** | 5 |
| `output_blocks.9.0.skip_connection` | **2** | 13 |
| `out.2` | **3** | 22 |
| `output_blocks.11.1.proj_in` | **4** | 38 |
| `input_blocks.1.1.proj_in` | 58 | **1** |
| `input_blocks.2.1.proj_in` | 55 | **2** |
| `input_blocks.2.1.attn1.to_k` | 223 | 6 |

The trajectory ranks the **decoder-side** `skip_connection` lines and output
layers highest, because their error reaches the final latent directly and is
not diluted by later blocks. A single forward at one timestep instead ranks the
**encoder-side** early `proj_in` transitions, which move that one step the most
but whose error is partly absorbed downstream.

### Why the original frame

The pack stores the **rotated** weight and the kernel rotates the
**activation**, so the error that reaches the output is `(q*s) @ H - W`. Injecting
in the rotated frame (`q*s`) would measure the rotation itself rather than the
error, and inverts the ranking.

---

## The pipeline

```
<base>  (FP16 checkpoint, 2033.8 MB)
  |  Step 1: measure the trajectory impact
  |          sd1.5/measure_sd15_unet_traj.py
  v
impact_sd15_traj_<model>.json   (282 layers)
  |  Step 2: build the pack
  |          sd1.5/build_sd15_unet_convrot_int8.py --top-k <K>
  v
<pack>  (K protected in FP16, the rest ConvRot INT8; CLIP untouched; VAE dropped)
  |  Step 3: the acceptance
  |          benchmark/sd1.5_int8_traj_compare.py (25 seeds)
  v
PASS / FAIL
```

### Step 1 - the trajectory impact

```powershell
python sd1.5/measure_sd15_unet_traj.py `
  --comfy "<ComfyUI root>" `
  --ckpt  "<base>" `
  --seeds "42,137" `
  --metric relmse
```

- 25 steps by default: that is SD1.5's production step count, the same one the
  acceptance gate and the calibration use. 12 is the Z-Image/NVFP4 default and
  is too short here.
- Writes `impact/impact_sd15_traj_<model stem>.json` when `--out` is omitted.
- Measured cost on one consumer GPU: **one 25-step trajectory is 0.98 s**, so
  282 layers x 2 seeds is about **12 minutes** - not hours.
- Progress prints every 10 layers; the ranking is not transferable between
  checkpoints.

### Step 2 - build the pack

```powershell
python sd1.5/build_sd15_unet_convrot_int8.py `
  --input   "<base>" `
  --output  "<pack>" `
  --contrib "impact/impact_sd15_traj_<model stem>.json" `
  --top-k   80 `
  --bias_correction `
  --calib_file "sample/calibration_prompts_128.txt"
```

What the builder writes:

| | |
| :--- | :--- |
| the quantized | every 2D `Linear` **and** 4D `Conv2d` under `model.diffusion_model.`, except the top-K |
| the protected | the top-K layers, plus every 1D tensor - **all stored FP16** |
| CLIP | **copied through unchanged** (`cond_stage_model.`): same keys, same dtypes, zero `.comfy_quant` / `.weight_scale` artifacts |
| VAE | **dropped** (`first_stage_model.` is not written) |
| the scale | INT8 `weight_scale` is FP32 by design; its rank matches the weight (`(O,1)` for 2D, `(O,1,1,1)` for 4D) |

**Bias correction is opt-in** (`--bias_correction`, default OFF). When enabled it
folds the systematic INT8 output shift into the layer bias:

```
E[(W_q - W) x]  ~  (W_q - W) @ E[x]
bias_new = bias_orig - delta
```

with `E[x]` the per-input-channel activation mean collected **in the same run**
from the V1.0 calibration (32 prompts x 25 steps, seed 42, via
`StableDiffusionPipeline`). It applies to **INT8 layers only** - a protected
layer is untouched, weights and bias alike.

The builder reports the honest count:
`the bias-corrected layers: N (skipped: A had no act_mean, B had a shape mismatch)`,
and it **aborts** rather than saving a pack labelled bias-corrected if N is 0.

### Step 3 - the acceptance

```powershell
python benchmark/sd1.5_int8_traj_compare.py `
  --fp16 "<base>" --int8 "<pack>" `
  --comfy_path "<ComfyUI root>" `
  --steps 25 --cfg 7.0 --width 512 --height 512 `
  --seeds "42,137,849,2024,7391,18429,53082,149206,382715,826401,1938502,4710928,8391642,15820493,36192847,71058294,128491703,285039184,491730285,762019483,938174026,1409285713,2683910547,3851729406,4195820371"
```

The gate: `final-cosine mean >= 0.97` and **0/25 bifurcated**. `drifted`
(gradual divergence) is acceptable; `bifurcated` (a sudden trajectory jump) is
not.

---

## Measured results on `0.5xxmix9realistic_v40 0.5BRA7`

25 seeds x 25 steps, 512x512, cfg 7.0.

| pack | the protection criterion | the protected | mean | same-image | bifurcated | size |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| native | none | 0 | 0.96265 | 13/25 | 0/25 | 1058.6 MB |
| k40 | trajectory | 40 | 0.96212 | 14/25 | 0/25 | ~1115 MB |
| **k80** | single forward | **80** | **0.97622** | **18/25** | **0/25** | 1167.5 MB |
| k80 | trajectory | 80 | 0.97542 | 16/25 | 0/25 | 1160.4 MB |
| k100 | single forward | 100 | 0.974 | - | 0/25 | 1193.5 MB |

### What the numbers say

**The protection works, and only above a threshold.** Native sits at 0.96265,
k40 at 0.96212 - identical. Protecting 40 layers buys nothing. The gain appears
between k40 and k80 (+0.0133), so the layers that matter are concentrated in
that band, not in the top 40.

**More protection is not better.** k100 (0.974) is *below* k80 (0.97622) and
26 MB larger. Two effects push against each other:

- a protected layer keeps its FP16 weights - no quantization error - but it
  also **receives no bias correction**;
- an INT8 layer absorbs bias correction, which cancels its systematic error.

Past k80, promoting more layers to FP16 loses more bias correction than it gains
in weight precision.

**The two criteria tie at k=80.** Single-forward 0.97622 vs trajectory 0.97542 -
a difference of 0.0008 against a per-seed spread of 0.80 to 0.999. They agree on
the outcome even though they disagree on 38 of the top 80 layers.

**Size ceiling.** The source is 2033.8 MB and the practical target is about
1.16 GB; the pack values above sit at that limit. There is no room for a
combined multi-metric selection the way SDXL has it - SD1.5 must pick one
criterion and spend its protection budget well.

---

## Honest limits

- **No criterion is globally optimal.** Every method here - static statistics,
  single forward, trajectory - measures **one layer at a time** and ranks. What
  actually happens is tens of layers quantized simultaneously, where errors
  interact. That cannot be recovered from single-layer measurements.
- **Single-layer contributions are not additive.** Two criteria that disagree on
  38 of 80 layers can produce the same final score; two that agree can still
  differ. The score at a given K has to be measured, not predicted.
- **The ranking is checkpoint-specific.** Impact order, K and the trajectory
  numbers must be re-measured for every model. Nothing here transfers.
- **The acceptance uses fixed seeds and prompts**, so it is a relative
  comparison between packs, not an absolute quality claim.

Treat the table as *"among the candidates measured under this criterion, this one
is best"* - not as *"this is the optimum"*.

---

## Files

| file | role |
| :--- | :--- |
| `sd1.5/measure_sd15_unet_traj.py` | **the criterion** - per-layer trajectory impact (25 steps) |
| `sd1.5/measure_sd15_unet_contrib.py` | the single-forward variant, for comparison only |
| `sd1.5/build_sd15_unet_convrot_int8.py` | the builder (top-K protected, bias correction opt-in, VAE dropped) |
| `sd1.5/verify_sd15_loader.py` | the acceptance through the real HSWQ loader |
| `benchmark/sd1.5_int8_traj_compare.py` | the 25-seed trajectory gate |
| `sd1.5/quantize_sd1.5_hswq_v1.0.py` | the **previous** selector (static histogram + MSE + SVD), kept for comparison |
| `sd1.5/native_convert_int8_sd1.5.py` | the plain all-INT8 converter (native baseline) |

---

## The loader trap (read before debugging NaN)

An INT8 Conv2d pack **must** be loaded through the HSWQ loader. Three reasons,
and getting any of them wrong yields NaN (`int8_linear` calls 0, output all NaN):

1. **`comfy_kitchen` has no INT8 conv kernel** - only `int8_linear`. The conv
   path exists solely in the loader's `_patch_ops_decode_and_conv()`.
2. That patch is gated on a thread-local: `_should_inject_int8_conv()` returns
   `bool(_int8_quant_conv_tls.active)`. Calling
   `apply_comfy_quant_int8_patches()` alone does not open it.
3. There are **two vendored copies** of `comfy_quant_int8` (under `patches/` and
   under `nodes/sd1.5 int8/`), each with its own thread-local. The SD1.5 loader
   collects every `_int8_quant_conv_tls` in `sys.modules` and opens them all.

Use `sd1.5/verify_sd15_loader.py`, which calls
`load_checkpoint_sd15_hswq_weight_dtype`.
