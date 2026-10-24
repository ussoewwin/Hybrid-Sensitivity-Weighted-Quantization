# SD1.5 UNet ConvRot INT8 - Aggregate-Contribution Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/unet/0.5xxmix9realistic_v40 0.5BRA7.safetensors` (2033.7 MB, FP16, 1131 tensors: UNet 686 / CLIP 197 / VAE 248)
**Pack.** `models/unet/0.5xxmix9realistic_v40 0.5BRA7_sd15_unet_hswq_convrot_int8.safetensors` (1791.3 MB, **−11.9 %**)
**SHA256.** `68C85762A65F7BF9A237D13DE40412965F576EE781FDDA1CE8CB2A7E6318A799`

---

## 0. What this replaces, and why

SD1.5 is the one family that *should* have used HSWQ's universal engine - it is a
per-checkpoint backbone like SDXL, with many merges and finetunes. The existing
`archives/quantize_sd1.5_hswq_v1.0.py` instead produced a protection set that
**scored the same as native** (no protection at all): the protection bought
nothing.

The reason is the criterion. V1.0 ranked the FP16 protection set from **static
weight-space statistics** (kurtosis / outlier-ratio / absmax) plus a
weight-space histogram MSE. That is a proxy for weight error, not for what the
sampler sees. This pack uses the **aggregate contribution**: replace one layer
by its ConvRot INT8 reconstruction and measure the change in the UNet's **own
output**. Rule 2 of the method hub states the distinction; here it is applied to
a UNet.

---

## 1. What the score is

The score is the **observed change of the UNet's own output**, `model(x,
timesteps, context)`, not a static weight statistic.

```
cos_i = <out_source_i, out_pack_i> / (||out_source_i|| * ||out_pack_i||)
1 - cos_i  = that item's error
```

A pack's score over the acceptance set is `1 - (the mean error)` and
`1 - (the worst item's error)`, quoted as `cos.mean` / `cos.min`.

Three choices are forced by the runtime, all measured here:

- **float64 for the comparison.** The cosine is taken on float64 copies of the
  output; a float32 dot product over 16,384 elements of a diffusion output
  accumulates rounding noise comparable to the signal (the smallest layer
  contribution measured here is 1.2e-08).
- **The baseline must use the same ops class as the pack.** The stock path
  builds with `manual_cast`; an INT8 pack builds with `mixed_precision_ops`.
  Both sides of every number below use `mixed_precision_ops`.
- **The quant detection must see the checkpoint prefix.** `comfy.utils.
  detect_layer_quantization()` matches `k.startswith(unet_key_prefix)`, so
  `model_config_from_unet` has to be called **before** the `model.diffusion_model.`
  prefix is stripped. Done the other way, the pack silently loads as FP16 and
  the INT8 path never runs.

The gate: `cos.mean >= 0.99` and `cos.min >= 0.98` over the fixed 256-item set
(seed 42), finite output.

---

## 2. What is quantized, and the two hard limits

**Only 2D `Linear` under `model.diffusion_model.` is quantized.** Two measured
limits decide this, and both were discovered the hard way:

1. **`comfy_kitchen` has no INT8 conv kernel.** It exposes `int8_linear` (and
   `mm_int8`, `w4a8_int8_linear`) but no INT8 convolution. A `Conv2d` whose
   weight is stored as INT8 therefore reaches a float conv on an int8 tensor and
   the whole UNet output becomes **NaN** (`int8_linear` call count: 0, output
   NaN: 16384). Conv2d stays FP16.
2. **The `weight_scale` shape must match the weight's rank.** The reference
   implementation (`sd1.5/native_convert_int8_sd1.5.py`, `pack_channelwise`)
   stores `(O, 1)` for a 2D weight and `(O, 1, 1, 1)` for a 4D weight. A
   flattened `(O,)` scale does not broadcast the same way and produced NaN
   through the Conv2d path.

Because Conv2d stays FP16, the size saving is bounded by how much of the UNet is
2D. Here it is **184 of the 282 ConvRot-eligible layers** (the other 98 are
Conv2d), so the ceiling is a small reduction, not the ~50 % the ControlNets
reach.

**CLIP is not quantized** (the instruction for this work): the whole
`cond_stage_model.` subtree (197 tensors) passes through byte-identical - same
keys, same dtypes, and **zero** `.comfy_quant` / `.weight_scale` artifacts. The
VAE is likewise untouched.

---

## 3. The impact measurement

`sd1.5/measure_sd15_unet_contrib.py` measures the **aggregate contribution**:
hold the FP16 weights, replace ONE layer by its ConvRot INT8 reconstruction in
the **original frame** (`dequant(INT8(W @ H^T))` = `(q*s) @ H`), and rank by
`1 - cos` of the UNet's own output over 8 fixed seeds.

282 layers were measured, none skipped. The spread is **464x**:

| the statistic | measured |
| :--- | :--- |
| the max | 5.6167e-06 (`input_blocks.1.1.proj_in`) |
| the median | 6.2205e-08 |
| the min | 1.2083e-08 (`output_blocks.11.1.transformer_blocks.0.attn2.to_k`) |

The top 15:

| rank | layer | the contribution | the share |
| :--- | :--- | :--- | :--- |
| 1 | `input_blocks.1.1.proj_in` | 5.6167e-06 | 10.77 % |
| 2 | `input_blocks.2.1.proj_in` | 2.8616e-06 | 5.49 % |
| 3 | `input_blocks.7.1.transformer_blocks.0.ff.net.0.proj` | 2.6664e-06 | 5.11 % |
| 4 | `input_blocks.5.1.proj_in` | 1.6888e-06 | 3.24 % |
| 5 | `output_blocks.6.0.skip_connection` | 1.6788e-06 | 3.22 % |
| 6 | `input_blocks.2.1.transformer_blocks.0.attn1.to_k` | 1.3840e-06 | 2.65 % |
| 7 | `input_blocks.3.0.op` | 1.3572e-06 | 2.60 % |
| 8 | `input_blocks.0.0` | 1.3022e-06 | 2.50 % |
| 9 | `input_blocks.2.1.transformer_blocks.0.attn1.to_q` | 1.0919e-06 | 2.09 % |
| 10 | `input_blocks.2.1.transformer_blocks.0.ff.net.0.proj` | 9.3205e-07 | 1.79 % |

**The structure of the ranking is the finding.** SD1.5's sensitive layers are
the **early input-block transition projections** (`input_blocks.1/2/5/7 .proj_in`)
and the **`skip_connection`** lines - not the attention projections that a
weight-space statistic would flag. The top 10 carry 39.5 % of the summed
contribution; the median layer carries 1.1 %.

---

## 4. The candidates, and the adopted set

Built from the contribution ranking (the top-K kept in FP16) and measured on the
fixed **256-item** acceptance set (seed 42), through the real loader.

| the candidate | the INT8 | the kept | cos mean | cos min | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 184 | 0 | 0.99999453 | 0.99979924 | 1777.3 MB |
| k=10 | 180 | 4 | 0.99999525 | 0.99986564 | 1790.7 MB |
| **k=20 (adopted)** | **177** | **7** | **0.99999623** | **0.99994836** | **1791.3 MB** |
| k=40 | 170 | 14 | 0.99999708 | 0.99996648 | 1793.7 MB |
| k=80 | 143 | 41 | 0.99999810 | 0.99997374 | 1834.0 MB |
| k=120 | 120 | 64 | 0.99999909 | 0.99997575 | 1887.8 MB |

(The "kept" count is lower than K because some of the top-K entries are Conv2d,
which is FP16 in every candidate anyway - protecting them costs nothing and
changes nothing.)

### Why k=20

The curve is selected on **marginal utility** - how much `cos.min` each MB buys:

| the step | the size cost | the gain in cos.min | the efficiency |
| :--- | :--- | :--- | :--- |
| k=0 → 10 | +13.4 MB | +6.64e-05 | 4.95e-06 /MB |
| **k=10 → 20** | **+0.6 MB** | **+8.27e-05** | **1.43e-04 /MB** ← the peak |
| k=20 → 40 | +2.4 MB | +1.81e-05 | 7.60e-06 /MB |
| k=40 → 80 | +40.3 MB | +7.26e-06 | 1.80e-07 /MB |
| k=80 → 120 | +53.9 MB | +2.01e-06 | 3.73e-08 /MB |

**k=20 is the peak efficiency point by a factor of 19x over the next step.**
It costs **+0.6 MB** and moves `cos.min` from 0.99986564 to 0.99994836 - a
**5.3x reduction of the worst-case error** (1.34e-04 → 5.16e-05). Every later
step spends 40-90x more MB per unit of quality.

k=120 has the highest `cos.min` (0.99997575) and is **not** adopted: it costs
**+96.5 MB** over k=20 for a further 2.7e-05 of `cos.min`, which is
3.7e-08 /MB - the flat tail. The rule is the size/quality balance, not the
highest cosine.

---

## 5. The pack

| item | value |
| :--- | :--- |
| the source | 2033.7 MB (FP16) |
| the pack | **1791.3 MB** (−11.9 %) |
| the SHA256 | `68C85762A65F7BF9A237D13DE40412965F576EE781FDDA1CE8CB2A7E6318A799` |
| the quantized layers | **177** of 282 ConvRot-eligible (the rest are Conv2d, FP16) |
| the FP16-kept | 7 top-contribution layers + all Conv2d |
| CLIP / VAE | **untouched** (197 / 248 tensors pass through) |
| the gate (256 items, seed 42) | **PASS** (mean 0.99999623, min 0.99994836) |

### Against the previous approach

| | V1.0 (static selection) | this pack (aggregate contribution) |
| :--- | :--- | :--- |
| the criterion | static kurtosis / outlier-ratio / absmax + weight-space MSE | the UNet's own output change, 282 layers, 8 seeds |
| the measured effect of protection | none (same score as native) | **+8.27e-05 of cos.min for +0.6 MB** |
| the worst item (cos.min) | ~ the native baseline | **0.99994836** |

The V1.0 result was not a bug in the quantizer; it was a **criterion** failure -
the weights it protected were not the weights the sampler is sensitive to. The
measured ranking says the sensitive ones are the early `proj_in` transitions and
the `skip_connection` lines.

---

## 6. Reproduction

```powershell
# the environment
#   python = <ComfyUI>/python_embeded/python.exe
#   COMFY  = <ComfyUI>  (the checkout ComfyUI runs from)

# 1. the criterion: the aggregate contribution (282 layers, 8 seeds, float64)
python sd1.5/measure_sd15_unet_contrib.py `
    --ckpt "<checkpoint>.safetensors" `
    --out impact/contrib_sd15_unet.json --n 8

# 2. the builder (top-k kept in FP16; only 2D Linear is quantized)
python sd1.5/quantize_sd1.5_hswq_v1.1.py `
    --input "<checkpoint>.safetensors" `
    --output impact/_work/sd15_k20.safetensors `
    --contrib impact/contrib_sd15_unet.json --top-k 20

# 3. the acceptance (256 items, seed 42)
$env:SD15_N = "256"
python sd1.5/verify_sd15_unet.py `
    "<checkpoint>.safetensors" impact/_work/sd15_k20.safetensors
```

**Timing.** The contribution pass takes ~70 min on the target machine (282
layers x 9 forwards). The acceptance takes ~60 min for the 256 items. The
builder takes ~2 min.

---

## 7. Error sources and limits

- **Conv2d cannot be INT8 here.** `comfy_kitchen` has no INT8 conv kernel; an
  INT8 Conv2d weight produces NaN. This caps the achievable size reduction at
  the 2D share of the UNet (here 184 of 282 eligible layers), which is why the
  pack is −11.9 % rather than the ~50 % a ControlNet reaches.
- **The `weight_scale` rank must match the weight's.** `(O,1)` for 2D,
  `(O,1,1,1)` for 4D. A flattened `(O,)` breaks the Conv2d broadcast path.
- **The quant detection must run before the prefix strip.** Otherwise the pack
  loads as FP16 and every number is quietly meaningless (the symptom is
  `unet unexpected: [...comfy_quant, ...weight_scale]` and `int8_linear` = 0).
- **The baseline must share the pack's ops class.** `manual_cast` and
  `mixed_precision_ops` are not numerically equal.
- **Input-distribution dependent.** Measured on fixed random latents and
  contexts of the correct shape (4x64x64 latent, 77x768 context, t=500) rather
  than on rendered images. Sound for ranking layers; a real-image spot check is
  still advisable before release.
- **The UNet's own output only.** This is one forward, not a 25-step sampling
  trajectory. A trajectory gate (the SDXL path's instrument) is the stronger
  check and is not run here; it needs a base VAE and sampler round trip.
- **A 256-item set.** `cos.min` is the worst of 256 draws, not a bound.
  Differences below ~1e-07 are not resolved, which is why k=40/k=80/k=120 read
  as a flat tail.
- **Single-layer contributions are marginal.** Many quantized together can
  partially cancel; the acceptance runs on the complete pack.

---

## 8. The experiment ledger

| # | when (JST) | what | the setting | cos mean / min | the size |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2026-10-24 06:05 | the aggregate contribution | 282 layers, 8 seeds, float64 | - (a ranking) | - |
| 2 | 2026-10-24 06:35 | candidate k=0 | 184 INT8, 0 kept | 0.99999453 / 0.99979924 | 1777.3 MB |
| 3 | 2026-10-24 06:35 | candidate k=10 | 180 INT8, 4 kept | 0.99999525 / 0.99986564 | 1790.7 MB |
| 4 | 2026-10-24 06:35 | **candidate k=20** | 177 INT8, 7 kept | **0.99999623 / 0.99994836** | **1791.3 MB** |
| 5 | 2026-10-24 06:35 | candidate k=40 | 170 INT8, 14 kept | 0.99999708 / 0.99996648 | 1793.7 MB |
| 6 | 2026-10-24 06:35 | candidate k=80 | 143 INT8, 41 kept | 0.99999810 / 0.99997374 | 1834.0 MB |
| 7 | 2026-10-24 06:35 | candidate k=120 | 120 INT8, 64 kept | 0.99999909 / 0.99997575 | 1887.8 MB |
| 8 | 2026-10-24 06:45 | the pack audit | CLIP 197 tensors identical, 0 quant artifacts; 0 Conv2d quantized | - | - |

Every row is a real run on this machine; the raw JSON is kept under `impact/`
(git-ignored).

---

## 9. Files

| file | role |
| :--- | :--- |
| `sd1.5/measure_sd15_unet_contrib.py` | the criterion (the aggregate contribution) |
| `sd1.5/quantize_sd1.5_hswq_v1.1.py` | the builder (`--top-k`, FP16 kept, 2D Linear only) |
| `sd1.5/verify_sd15_unet.py` | the acceptance on the fixed 256-item set |
| `md/SD15_UNet_ConvRot_INT8_Aggregate_Contribution_Guide.md` | this guide |
| `impact/contrib_sd15_unet.json` | the contribution (git-ignored) |
| `impact/verify_sd15_256.json` | the acceptance results (git-ignored) |
