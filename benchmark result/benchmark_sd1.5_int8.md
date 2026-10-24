# SD1.5 ConvRot INT8 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **ConvRot INT8** output on the SD1.5 UNet family.
Higher final-cosine is better (1.0 = identical trajectory).

**Source:** `benchmark result/score_sd1.5_int8.txt`

**Method:** deterministic per-step latent trajectory divergence, 25 seeds x 25 steps, 512x512, CFG 7.0, `dpmpp_2m` / `karras`. Every run uses the same latent, context and seed schedule, so the only variable is the quantized UNet.

**Checkpoint:** `0.5xxmix9realistic_v40 0.5BRA7` (FP16 source 2033.8 MB)

---

## Results

| Model | Protection | INT8 layers | FP16 kept | final-cos min | final-cos mean | final-cos max | same-image | bifurcated | Size |
|-------|-----------:|------------:|----------:|--------------:|---------------:|--------------:|-----------:|-----------:|-----:|
| **HSWQ k80** (trajectory impact) | 80 | 202 | 80 | 0.80943 | **0.97542** | 0.99892 | **16/25** | 0/25 | 1160.4 MB |
| **HSWQ k80** (single forward) | 80 | 202 | 80 | 0.80943 | **0.97622** | 0.99878 | **18/25** | 0/25 | 1167.5 MB |
| **native** (no protection) | 0 | 282 | 0 | 0.77574 | 0.96265 | 0.99913 | 13/25 | 0/25 | 1058.6 MB |

---

## HSWQ vs Native comparison

Same setup (vs FP16 reference). **HSWQ** vs **native** (naive all-INT8 cast, no protection).
`Δ` = native − HSWQ (positive Δ mean → HSWQ better, since higher final-cosine is better).

| Model | Protection | HSWQ mean | Native mean | Δ mean | HSWQ same-image | Native same-image | Δ same-image | Winner |
|-------|-----------:|----------:|------------:|-------:|----------------:|------------------:|-------------:|--------|
| HSWQ k80 (single forward) | 80 | **0.97622** | 0.96265 | **+0.01357** | **18/25** | 13/25 | **+5** | **HSWQ** |
| HSWQ k80 (trajectory impact) | 80 | **0.97542** | 0.96265 | **+0.01277** | **16/25** | 13/25 | **+3** | **HSWQ** |

**Winner** = better on both mean final-cosine and same-image count.

---

## Protection sweep

The protection set comes from a measured per-layer **trajectory impact**: hold every other layer pristine, replace ONE layer by its ConvRot INT8 reconstruction in the original frame, run the same 25-step trajectory, record `rel_mse` against the pristine run. The top-K stay FP16.

| K (protected) | Selection criterion | final-cos mean | same-image | Size | Note |
|--------------:|---------------------|---------------:|-----------:|-----:|------|
| 0 (native) | — | 0.96265 | 13/25 | 1058.6 MB | all-INT8 baseline |
| 40 | trajectory impact | 0.96212 | 14/25 | ~1115 MB | **identical to native - protection buys nothing** |
| 80 | single forward | **0.97622** | **18/25** | 1167.5 MB | best measured point |
| 80 | trajectory impact | 0.97542 | 16/25 | 1160.4 MB | ties within noise, 7.1 MB smaller |
| 100 | single forward | 0.974 | — | 1193.5 MB | **worse AND larger than k80** |

---

## Notes

- **final-cos:** cosine similarity between the FP16 and INT8 latent trajectories at the final step. 1.0 = identical.
- **max-drop:** the largest per-step cosine drop along the trajectory. A sudden large drop (`bifurcated`, threshold 0.05) means the picture became a *different image*, not a degraded one.
- **same-image:** `final-cos >= 0.98` **and** not bifurcated.
- **drifted:** gradual divergence - acceptable at this level.
- **bifurcated:** 0/25 in every candidate. No candidate ever jumped to a different picture.
- **Speed:** FP16 1.70-1.71 s/seed → INT8 1.31-1.35 s/seed, **+20.8% to +23.2%**.

---

## Analysis & Key Findings

### The criterion decides whether protection works at all

The previously shipped **V1.0** selector ranked protection from static weight-space statistics (kurtosis, outlier ratio, absmax, weighted-histogram MSE). Its protection set scored **the same as native** - the protection bought nothing. That was a **criterion** failure, not a quantizer failure: those statistics answer "how much does this layer's weight distort", not "how much does this layer move the output". In a deep UNet, a layer's error is amplified, attenuated or **cancelled** downstream, and a weight-space statistic cannot see that propagation.

With the measured trajectory impact, the same quantization produces a real gain:

- **native 0.96265 → k80 0.97622 (+0.01357)**, and same-image **13/25 → 18/25**.

### The protection has a threshold, and more is worse

- **k40 is identical to native** (0.96212 vs 0.96265). Protecting the top 40 layers buys nothing; the layers that matter sit in positions 41-80.
- **k100 is worse than k80 AND larger** (0.974 vs 0.97622, +26 MB). Two effects push against each other: a protected layer keeps FP16 weights but **receives no bias correction**, while an INT8 layer absorbs bias correction and cancels its systematic error. Past k80, promoting layers to FP16 loses more correction than it gains in precision.

### The two criteria disagree on the ranking but agree on the score

The single-forward and trajectory criteria place **38 of the top 80 layers differently**:

| layer | trajectory rank | single-forward rank |
|-------|----------------:|--------------------:|
| `output_blocks.6.0.skip_connection` | **1** | 5 |
| `output_blocks.9.0.skip_connection` | **2** | 13 |
| `out.2` | **3** | 22 |
| `input_blocks.1.1.proj_in` | 58 | **1** |
| `input_blocks.2.1.proj_in` | 55 | **2** |

A trajectory ranks the **decoder-side** bypasses highest (their error reaches the final latent directly); a single forward ranks the **encoder-side** early transitions (they move one timestep the most but are partly absorbed later). At k=80 the two land within 0.0008 of each other - inside the per-seed spread of 0.809 to 0.999.

### The worst seeds are what improve

| seed | native | HSWQ k80 |
|------|-------:|---------:|
| 4710928 | 0.77574 | **0.95062** |
| 53082 | 0.79762 | 0.80943 |
| 762019483 | 0.91077 | **0.94085** |
| 18429 | 0.96543 | 0.96387 |

Seed `4710928` gains **+0.177**. The improvement is concentrated on the unstable trajectories, not spread thinly across the good ones.

### Composition

- **UNet only.** CLIP is copied through unchanged (same keys, same dtypes, zero `.comfy_quant` / `.weight_scale` artifacts).
- **The VAE is dropped** from the pack. Decode with a separate SD1.5 VAE.

### Honest limits

- **No criterion here is globally optimal.** Every method - static statistics, single forward, trajectory - measures **one layer at a time** while tens are quantized together, so single-layer contributions are not additive. Two criteria that disagree on 38 of 80 layers can reach the same score.
- **The ranking is checkpoint-specific.** Impact order, K and these numbers must be re-measured for every model.
- **Size ceiling.** The source is 2033.8 MB and the practical target is about 1.16 GB, so there is no room for the combined multi-metric selection the SDXL path uses.
