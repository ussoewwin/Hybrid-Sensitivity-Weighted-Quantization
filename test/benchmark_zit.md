# Z Image Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ FP8 quantized** output (Z Image Turbo family).  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_zi.txt`

---

## Results

| Model | Keep ratio | MSE (latent, ↓ better) | SSIM (0–255 view, ↑ better) |
|-------|------------|--------------------------|-----------------------------|
| darkBeastMar2126Latest_dbzit8SDAFOK | r0.05 | 0.0181 | 0.9591 |
| moodyWildMix_v02 | r0.1 | 0.0057 | 0.9582 |
| moodyRealMix_zitV4DPO | r0.1 | 0.0056 | 0.9618 |
| unstableRevolution_V2Fp16 | r0.05 | 0.0069 | 0.9542 |
| beyondREALITY_V30 | r0.05 | 0.0089 | 0.9597 |

---

## HSWQ vs Native FP8 / Official FP8 comparison

Same setup (vs FP16 reference). **HSWQ FP8** vs baseline (see below).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native** = naive cast FP8. **Official FP8** = officially distributed FP8. Native and Official FP8 are not the same.

| Model | Keep | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| darkBeastMar2126Latest_dbzit8SDAFOK | r0.05 | 0.0181 | 0.0253 | +0.0072 | 0.9591 | 0.9177 | −0.0414 | Native FP8 | HSWQ |
| moodyWildMix_v02 | r0.1 | 0.0057 | 0.0188 | +0.0131 | 0.9582 | 0.9297 | −0.0285 | Native FP8 | HSWQ |
| moodyRealMix_zitV4DPO | r0.1 | 0.0056 | 0.0192 | +0.0136 | 0.9618 | 0.9343 | −0.0275 | Official FP8 | HSWQ |
| unstableRevolution_V2Fp16 | r0.05 | 0.0069 | 0.0219 | +0.0150 | 0.9542 | 0.9195 | −0.0347 | Native FP8 | HSWQ |
| beyondREALITY_V30 | r0.05 | 0.0089 | 0.0179 | +0.0090 | 0.9597 | 0.9253 | −0.0344 | Official FP8 | HSWQ |

**Winner** = better on both MSE and SSIM (lower MSE and higher SSIM for HSWQ vs baseline).

---

## Notes

- **MSE (latent):** Mean squared error on raw latent tensors vs FP16 reference; 0 = perfect match.
- **SSIM (0–255 view):** Structural similarity on normalized 0–255 preview images (`zit_bench`); 1.0 = perfect match.
- **Keep ratio:** Fraction of layers kept in FP16 (e.g. r0.1 = 10%, r0.05 = 5%).
- **Test environment (from logs):** Peak VRAM ~12.3 GB FP16 / ~7.3–7.5 GB HSWQ FP8 where applicable; see `score_zi.txt` blocks for per-run VRAM and inference time.

---

## Analysis & Key Findings (Z Image, partial)

For every model in `score_zi.txt` with a Native FP8 or Official FP8 baseline, **HSWQ** shows lower latent MSE and higher SSIM than that baseline, with large gaps vs naive Native FP8 on several checkpoints. **moodyRealMix_zitV4DPO** and **beyondREALITY_V30** use an **Official FP8** baseline; HSWQ still wins on both metrics in those runs. Expand this section as more models and baselines are added to `score_zi.txt`.
