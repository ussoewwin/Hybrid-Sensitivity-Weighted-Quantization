# SDXL ConvRot INT8 Benchmark Test Results

**Production gate: the deterministic 25-seed latent-trajectory comparison**
(`benchmark/sdxl_int8_traj_compare.py`). It samples the FP16 baseline and the quantized model from
identical noise (same seed) and compares the per-step latent trajectory:

- **PASS** = final-cosine **mean ≥ 0.95** and **0/25 bifurcated** (max-step-drop > 0.05 on any seed).
- **same-image** = per-seed final cosine ≥ 0.98.
- Fixed protocol: **25 random seeds** (script default), **25 steps**, 1024×1024, cfg 7.0,
  dpmpp_2m / karras. See [How to quantize SDXL.md](../md/How%20to%20quantize%20SDXL.md).

This replaces the decoded-image (MSE / SSIM) bench as the production metric. The older decoded-image
results are kept below as legacy reference only.

**Sources:** `benchmark result/score_sdxl_reverse_int8.txt` (trajectory, diag-reverse hybrids),
`benchmark result/score_sdxl_int8.txt` (legacy decoded-image bench).

---

## Trajectory benchmark (production protocol)

### Reference points

| Configuration | 25-seed cosine mean | Note |
|---|---|---|
| FP16 baseline | 1.000 | identity (same checkpoint, both branches) |
| native ConvRot INT8 | ≈ 0.93 | all convertible layers INT8; the floor the hybrid must beat |

### Reverse hybrid (diag-reverse) results

`<model>_hswq_rev_int<K>_convrot_int8.safetensors` — K lowest-impact layers ConvRot INT8, everything
else FP16. Checkpoint: `waiIllustriousSDXL_v170` (788 quantifiable layers; FP16 base 6.94 GB;
K=788 (all eligible layers) hybrid ≈ 4.39 GB; GB = decimal).

> **Preliminary:** the rows below were measured with **5 of the 25 seeds** — they are **not gate
> results**. The production verdict requires the full 25-seed set on a quiet GPU (concurrent GPU
> processes change the numbers; ComfyUI picks different attention/offload paths by free VRAM).

| K | size (GB) | seeds | mean cos | min | max | same-image | bifurcated |
|---|---|---|---|---|---|---|---|
| 60 | 6.81 | 5 | 0.98356 | 0.94224 | 0.99944 | 4/5 | 0/5 |
| 75 | 6.78 | 5 | 0.99119 | 0.96872 | 0.99900 | 4/5 | 0/5 |
| 100 | 6.73 | 5 | 0.98968 | 0.96783 | 0.99931 | 4/5 | 0/5 |
| 150 | 6.63 | 5 | 0.97738 | 0.95523 | 0.99904 | 2/5 | 0/5 |
| 300 | 6.06 | 5 | 0.96000 | 0.90721 | 0.99264 | 2/5 | 0/5 |
| 500 | 5.32 | 5 | 0.96796 | 0.93574 | 0.99228 | 2/5 | 0/5 |
| 580 | 4.97 | 5 | 0.94937 | 0.88577 | 0.98800 | 1/5 | 0/5 |
| 620 | 4.82 | 5 | 0.96505 | 0.93282 | 0.99792 | 2/5 | 0/5 |

All rows are above the native ConvRot INT8 reference (≈ 0.93) as expected: the FP16-kept
high-impact layers are the precision reserve that the full pack does not have.

**Re-measurement note:** run-to-run variation was observed at 5 seeds (the same model and seed gave
final-cos 0.97714 in one run and 0.99666 in another while another GPU process was active). Always
use the full 25-seed set, one process at a time, for any verdict.

---

## Legacy results (decoded-image bench — not the gate)

The tables below were measured with the decoded-image fidelity bench (`benchmark/int8bench_sdxl.py`:
one fixed prompt and seed, single image, MSE / SSIM). They are kept for historical comparison only.

### HSWQ ConvRot INT8 (decoded image)

| Model | Bias correction | MSE (↓ better) | SSIM (↑ better) |
|-------|-----------------|----------------|-----------------|
| bluePencilXL_v031 | 1off | 14.80 | 0.9442 |
| epicrealismXL_pureFix | 1off | 7.98 | 0.9763 |
| JANKUTrainedChenkinNoobai_v777 | 1off | 6.31 | 0.9813 |
| koronemixIllustrious_v70 | 1on | 17.99 | 0.9670 |
| koronemixVpred_v20 | 1off | 23.19 | 0.9643 |
| novaAnimeXL_ilV190 | 1on | 7.70 | 0.9620 |
| novaAsianXL_illustriousV70 | 1off | 4.58 | 0.9798 |
| oneObsession_v23 | 1off | 13.36 | 0.9694 |
| perfectionAsianILXL_v10 | 1off | 8.44 | 0.9755 |
| perfectionRealisticILXL_80 | 1on | 2.74 | 0.9865 |
| prefectIllustriousXL_v8 | 1on | 19.96 | 0.9448 |
| realvisxlV30_v30TurboBakedvae | 1on | 8.73 | 0.9711 |
| realvisxlV50_v40Bakedvae | 1on | 4.67 | 0.9837 |
| realvisxlV50_v50Bakedvae | 1on | 5.53 | 0.9735 |
| unholyDesireMixSinister_v80 | 1on | 4.28 | 0.9821 |
| uwazumimixILL_v50 | 1on | 2.40 | 0.9818 |
| waiANIPONYXL_v140 | 1on | 8.27 | 0.9607 |
| waiANIPONYXL_v90 | 1on | 10.23 | 0.9507 |
| waiIllustriousSDXL_v170 | 1off | 8.41 | 0.9712 |
| waiREALCN_v150 | 1on | 6.47 | 0.9672 |
| waiREALISM_v10 | 1on | 9.62 | 0.9527 |

**Labels:** `1on` = bias correction ON, `1off` = bias correction OFF.

### HSWQ ConvRot INT8 vs Native ConvRot INT8 (decoded image)

Same setup (vs FP16 reference). Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better).

| Model | Bias correction | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Winner |
|-------|-----------------|----------|--------------|-------|-----------|---------------|--------|--------|
| bluePencilXL_v031 | 1off | 14.80 | 20.34 | +5.54 | 0.9442 | 0.9365 | −0.0077 | HSWQ |
| epicrealismXL_pureFix | 1off | 7.98 | 8.79 | +0.81 | 0.9763 | 0.9756 | −0.0007 | HSWQ |
| JANKUTrainedChenkinNoobai_v777 | 1off | 6.31 | 21.56 | +15.25 | 0.9813 | 0.9626 | −0.0187 | HSWQ |
| koronemixIllustrious_v70 | 1on | 17.99 | 32.55 | +14.56 | 0.9670 | 0.9330 | −0.0340 | HSWQ |
| koronemixVpred_v20 | 1off | 23.19 | 20.18 | −3.01 | 0.9643 | 0.9754 | +0.0111 | Native |
| novaAnimeXL_ilV190 | 1on | 7.70 | 13.16 | +5.46 | 0.9620 | 0.9350 | −0.0270 | HSWQ |
| novaAsianXL_illustriousV70 | 1off | 4.58 | 5.34 | +0.76 | 0.9798 | 0.9771 | −0.0027 | HSWQ |
| oneObsession_v23 | 1off | 13.36 | 16.49 | +3.13 | 0.9694 | 0.9672 | −0.0022 | HSWQ |
| perfectionAsianILXL_v10 | 1off | 8.44 | 4.38 | −4.06 | 0.9755 | 0.9894 | +0.0139 | Native |
| perfectionRealisticILXL_80 | 1on | 2.74 | 3.00 | +0.26 | 0.9865 | 0.9852 | −0.0013 | HSWQ |
| prefectIllustriousXL_v8 | 1on | 19.96 | 41.23 | +21.27 | 0.9448 | 0.9315 | −0.0133 | HSWQ |
| realvisxlV30_v30TurboBakedvae | 1on | 8.73 | 8.71 | −0.02 | 0.9711 | 0.9683 | −0.0028 | — |
| realvisxlV50_v40Bakedvae | 1on | 4.67 | 5.64 | +0.97 | 0.9837 | 0.9751 | −0.0086 | HSWQ |
| realvisxlV50_v50Bakedvae | 1on | 5.53 | 5.94 | +0.41 | 0.9735 | 0.9728 | −0.0007 | HSWQ |
| unholyDesireMixSinister_v80 | 1on | 4.28 | 7.61 | +3.33 | 0.9821 | 0.9797 | −0.0024 | HSWQ |
| uwazumimixILL_v50 | 1on | 2.40 | 4.95 | +2.55 | 0.9818 | 0.9758 | −0.0060 | HSWQ |
| waiANIPONYXL_v140 | 1on | 8.27 | 8.84 | +0.57 | 0.9607 | 0.9626 | +0.0019 | — |
| waiANIPONYXL_v90 | 1on | 10.23 | 9.60 | −0.63 | 0.9507 | 0.9502 | −0.0005 | — |
| waiIllustriousSDXL_v170 | 1off | 8.41 | 9.02 | +0.61 | 0.9712 | 0.9701 | −0.0011 | HSWQ |
| waiREALCN_v150 | 1on | 6.47 | 12.29 | +5.82 | 0.9672 | 0.9603 | −0.0069 | HSWQ |
| waiREALISM_v10 | 1on | 9.62 | 9.73 | +0.11 | 0.9527 | 0.9522 | −0.0005 | HSWQ |

**Winner** = better on both MSE and SSIM. Native ConvRot INT8 = naive cast ConvRot INT8.

---

## Notes

- **Trajectory protocol** is the production gate: fixed 25 random seeds, 25 steps; report the whole
  per-seed block plus the summary. `drifted` (final-cos < 0.98) is a normal small divergence and is
  not a failure; only **bifurcation** (max-step-drop > 0.05) and **mean < 0.95** fail.
- **MSE / SSIM** (legacy): Mean Squared Error (0 = perfect) / Structural Similarity (1.0 = perfect),
  from the decoded-image bench.
- **Bias correction:** `1on` / `1off` tags apply to the legacy decoded-image rows only; the
  diag-reverse hybrid path does not use bias correction.
