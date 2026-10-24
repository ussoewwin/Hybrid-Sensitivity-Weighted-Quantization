# SD1.5 ConvRot INT8 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **ConvRot INT8** output on the SD1.5 UNet family.
Higher final-cosine is better (1.0 = identical trajectory).

**Source:** `benchmark result/score_sd1.5_int8.txt` - every number in the Results
table below is read from that file; nothing is carried over from elsewhere.

**Method:** deterministic per-step latent trajectory divergence, 25 seeds x 25 steps, 512x512, CFG 7.0, `dpmpp_2m` / `karras`. Every run uses the same latent, context and seed schedule, so the only variable is the quantized UNet.

**Checkpoint:** `0.5xxmix9realistic_v40 0.5BRA7`

> **K is not a fixed number.** The protected-layer count is chosen **per checkpoint**
> by measuring a sweep and taking the point where the mean stops paying for the
> extra bytes. The two blocks in this file are the adopted pack and the native
> baseline for this checkpoint; other checkpoints need their own sweep.

---

## Results

| Model | final-cos min | final-cos mean | final-cos max | same-image | bifurcated | FP16 | INT8 | Speed |
|-------|--------------:|---------------:|--------------:|-----------:|-----------:|-----:|-----:|------:|
| **HSWQ k80** (`1on k80`) | 0.80943 | **0.97542** | 0.99892 | **16/25** | 0/25 | 1.70 s | 1.31 s | **+23.2%** |
| **native** (`native convrot int8`) | 0.77574 | 0.96265 | 0.99913 | 13/25 | 0/25 | 1.71 s | 1.35 s | **+20.8%** |

---

## HSWQ vs Native comparison

Same setup (vs FP16 reference). **HSWQ** vs **native** (naive all-INT8 cast, no protection).
`Δ` = native − HSWQ (positive Δ mean → HSWQ better, since higher final-cosine is better).

| Model | HSWQ mean | Native mean | Δ mean | HSWQ same-image | Native same-image | Δ same-image | Δ min | Winner |
|-------|----------:|------------:|-------:|----------------:|------------------:|-------------:|------:|--------|
| HSWQ k80 | **0.97542** | 0.96265 | **+0.01277** | **16/25** | 13/25 | **+3** | +0.03369 | **HSWQ** |

**Winner** = better on both mean final-cosine and same-image count.

---

## Notes

- **final-cos:** cosine similarity between the FP16 and INT8 latent trajectories at the final step. 1.0 = identical.
- **max-drop:** the largest per-step cosine drop along the trajectory (per-seed values are in the source txt). A sudden large drop (`bifurcated`, threshold 0.05) means the picture became a *different image*, not a degraded one.
- **same-image:** `final-cos >= 0.98` **and** not bifurcated.
- **drifted:** gradual divergence - acceptable at this level.
- **bifurcated:** 0/25 in both blocks. Neither candidate ever jumped to a different picture.

---

## Per-seed comparison (from the source txt)

| seed | HSWQ k80 | native | Δ |
|-----:|---------:|-------:|--:|
| 42 | 0.99572 | 0.98524 | +0.01048 |
| 137 | 0.99411 | 0.97665 | +0.01746 |
| 849 | 0.99865 | 0.96341 | +0.03524 |
| 2024 | 0.98348 | 0.98124 | +0.00224 |
| 7391 | 0.98741 | 0.97558 | +0.01183 |
| 18429 | 0.96387 | 0.96543 | -0.00156 |
| 53082 | 0.80943 | 0.79762 | +0.01181 |
| 149206 | 0.98499 | 0.99913 | -0.01414 |
| 382715 | 0.96260 | 0.96209 | +0.00051 |
| 826401 | 0.98585 | 0.97448 | +0.01137 |
| 1938502 | 0.97897 | 0.98035 | -0.00138 |
| 4710928 | 0.95062 | 0.77574 | **+0.17488** |
| 8391642 | 0.97884 | 0.98937 | -0.01053 |
| 15820493 | 0.98530 | 0.97984 | +0.00546 |
| 36192847 | 0.97011 | 0.95999 | +0.01012 |
| 71058294 | 0.95238 | 0.95664 | -0.00426 |
| 128491703 | 0.99093 | 0.98421 | +0.00672 |
| 285039184 | 0.99352 | 0.98210 | +0.01142 |
| 491730285 | 0.99153 | 0.99665 | -0.00512 |
| 762019483 | 0.94085 | 0.91077 | **+0.03008** |
| 938174026 | 0.99892 | 0.99748 | +0.00144 |
| 1409285713 | 0.99875 | 0.99244 | +0.00631 |
| 2683910547 | 0.99177 | 0.98950 | +0.00227 |
| 3851729406 | 0.99851 | 0.99689 | +0.00162 |
| 4195820371 | 0.99840 | 0.99338 | +0.00502 |

**25 seeds: HSWQ better on 18, worse on 6, tied within 0.001 on 1.**

---

## Analysis & Key Findings

### Where the gain comes from

The mean gain is **+0.01277**, and it is not spread evenly. Two seeds carry most of it:

| seed | native | HSWQ k80 | Δ |
|-----:|-------:|---------:|--:|
| 4710928 | 0.77574 | 0.95062 | **+0.17488** |
| 849 | 0.96341 | 0.99865 | +0.03524 |
| 762019483 | 0.91077 | 0.94085 | +0.03008 |

Seed `4710928` alone moves +0.175 - an unstable trajectory rescued rather than a small nudge on every seed. Conversely **6 seeds are slightly worse** (largest: 149206 at −0.01414), so the protection is a trade on individual seeds that nets positive in aggregate.

### The protection has a threshold, and more is not better

The criterion selects layers that the **sampler** is sensitive to, not layers whose weights distort most. That distinction is what makes the protection work at all: the retired V1.0 selector ranked protection from static weight-space statistics (kurtosis, outlier ratio, absmax, weighted-histogram MSE) and its protection set scored **the same as native**.

The sweep that fixes K has two failure modes, which is why K is measured rather than assumed:

- **too few protected layers** - the set misses the layers that actually matter, and the score collapses back to native (the protection buys nothing);
- **too many** - each layer promoted to FP16 keeps its weights exact but **loses the INT8 bias correction** that cancels its systematic output shift, so the score can fall *while the pack grows*.

The adopted K sits between them. For this checkpoint the sweep, the per-K numbers and the reasoning are in the technical guide (`md/How to quantize SD1.5.md`); this file records only the two blocks that were measured and saved here.

### Composition

- **UNet only.** CLIP is copied through unchanged (same keys, same dtypes, zero `.comfy_quant` / `.weight_scale` artifacts).
- **The VAE is dropped** from the pack. Decode with a separate SD1.5 VAE.

### Honest limits

- **No criterion here is globally optimal.** Every method measures **one layer at a time** while tens are quantized together, so single-layer contributions are not additive.
- **The ranking is checkpoint-specific.** Impact order, K and these numbers must be re-measured for every model.
- **25 seeds is a sample.** `min` is the worst of 25 draws, not a bound.
- **This file is one checkpoint, two blocks.** It is a relative comparison (HSWQ vs native on the same gate), not an absolute quality claim.
