## Overview

**v2.3.8 changes what the SDXL ConvRot INT8 benchmark *measures*.**

The benchmark record format was switched from the **decoded-image MSE / SSIM bench** to the **deterministic 25-seed latent-trajectory comparison**. The record file `benchmark result/benchmark_sdxl_int8.md` is now generated from the trajectory score log (`benchmark result/score_sdxl_int8.txt`) and reports **HSWQ ConvRot INT8 vs Native ConvRot INT8** side by side.

No quantization algorithm changed in this release. The change is in the **measurement**, and therefore in the **meaning of every number in the record**.

---

## Before and after

| | Old (up to v2.3.7) | New (v2.3.8) |
| :--- | :--- | :--- |
| Record file | `benchmark result/benchmark_sdxl_int8.md` (decoded-image) | same file, trajectory format |
| Benchmark script | `benchmark/int8bench_sdxl.py` | `benchmark/sdxl_int8_traj_compare.py` |
| What is compared | one decoded RGB image | the per-step latent trajectory |
| Samples | 1 fixed prompt + 1 fixed seed | 25 fixed random seeds x 25 steps |
| Reference | the FP16 image | the FP16 latent (identical initial noise) |
| Metrics | pixel MSE, SSIM | final cosine, max-step drop, latent MSE |
| Pass criterion | none (unbounded MSE, arbitrary SSIM) | mean cosine >= 0.95 **and** 0/25 bifurcated |
| Baseline in the record | Native ConvRot INT8 (decoded image) | Native ConvRot INT8 (same trajectory protocol) |
| VAE decode involved | yes | no (latent space only) |

---

## How the meaning changed

### 1. From "one image looks close" to "the process is reproducible"

The old bench decoded the final latent into an RGB image and measured how close *that one image* was to the FP16 image. A single number described a single sample.

The new bench feeds the **same initial noise and the same seed** to both the FP16 reference and the quantized model and compares the **latent at every sampling step**. The question is no longer "does one picture look similar", but "does the quantized model follow the same denoising trajectory and arrive at the same image as FP16".

### 2. The VAE and the rest of the pipeline leave the measurement

The old metric included the VAE decode in the number, so the score mixed two different things: the error of the quantized UNet and the error introduced by decoding. The new metric stays in **latent space** and measures **only the quantized denoiser**. It is therefore a direct measure of the quantization, and it is independent of the VAE.

### 3. One sample becomes a population (and the worst case becomes visible)

A model can match FP16 on one seed and diverge on another. A single-seed score cannot see that. Running 25 seeds turns the result into a distribution: the mean describes average behaviour, while the minimum and the per-seed table expose the worst case. This is what makes the benchmark usable as a **gate**.

### 4. Gradual drift and catastrophic failure are separated

Two failure modes are now distinguished explicitly:

- **drift** — a gradual, continuous deviation that still keeps a coherent, related image;
- **bifurcation** — a sudden single-step cosine drop (> 0.05) that throws the run into a different picture attractor basin altogether.

A pixel metric cannot tell these apart; a per-step trajectory can. Bifurcation is the dangerous failure mode, and the new record counts it directly (`0/25` is the ship criterion).

### 5. A physical quantity replaces a perceptual proxy, and the gate becomes explicit

- `final-cos` — cosine similarity between the FP16 final latent and the quantized final latent; **1.0 = the same image**.
- `same-image` — per-seed final cosine >= 0.98.
- `bifurcated` — max single-step drop > 0.05.
- **Gate: mean cosine >= 0.95 and 0/25 bifurcated.**

SSIM saturates near 0.97-0.99 and MSE is unbounded and prompt-dependent, so neither gave a fixed pass/fail line. The trajectory metric gives one.

### 6. HSWQ vs Native becomes an apples-to-apples comparison

The record now shows **HSWQ ConvRot INT8** and **Native ConvRot INT8** (naive cast) measured under the *same* 25-seed trajectory protocol, per model, with delta cosine, delta latent MSE, bifurcation counts, and the winner.

---

## What the record file now contains

`benchmark result/benchmark_sdxl_int8.md`:

- **Cross-model summary** — HSWQ vs Native mean cosine, delta, latent MSE, bifurcation rates, speedup and winner for 19 SDXL checkpoints, plus the family average.
- **Per-model detail** — a metric overview and a **side-by-side table for all 25 seeds** (HSWQ cosine / MSE vs Native cosine / MSE, per-seed verdict, winner).
- **Metric definitions** — final cosine, final MSE, max-step drop, verdicts, setup tags (`reNNN`, `1on`).

Removed: the decoded-image MSE / SSIM tables and the 5-seed reverse-hybrid `K` table.

---

## First record under the new format (19 SDXL checkpoints, 25 seeds each)

| Family average | HSWQ ConvRot INT8 | Native ConvRot INT8 |
| :--- | :--- | :--- |
| Mean final cosine | **0.97006** | 0.95356 |
| Bifurcated seeds | **1 / 475** | 3 / 475 |
| Same-image seeds | **260 / 475** | 174 / 475 |
| Mean final latent MSE | **1.4989** | 2.2922 |
| Speedup vs FP16 (avg per seed) | **+14.4%** | — |

---

## Documentation Links

- **Benchmark record (new format)**: [`benchmark result/benchmark_sdxl_int8.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/benchmark%20result/benchmark_sdxl_int8.md)
- **Score log**: [`benchmark result/score_sdxl_int8.txt`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/benchmark%20result/score_sdxl_int8.txt)
- **Quantization guide**: [`md/How to quantize SDXL.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL.md)
- **Changelog**: [CHANGELOG.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/CHANGELOG.md)
