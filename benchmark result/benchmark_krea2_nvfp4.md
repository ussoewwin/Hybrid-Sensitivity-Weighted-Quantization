# Krea2 Hybrid NVFP4 Benchmark Test Results

Benchmark comparison: **BF16 reference** vs **HSWQ hybrid NVFP4 quantized** output (Krea2 family).  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `benchmark result/score_krea2_nvfp4.txt`

**Column labels from the score log:**

| Label | Meaning |
|-------|---------|
| `+N` (e.g. `+239`, `+221`) | Number of **NVFP4 kept layers** in the hybrid output (those layers are kept as NVFP4; the remaining layers stay INT8 shelter) |

---

## Results

| Model | NVFP4 layers (+N) | MSE (↓ better) | SSIM (↑ better) | Latent MSE (↓ better) | Latent Cosine (↑ better) |
|-------|-------------------|----------------|-----------------|-----------------------|--------------------------|
| moodyKrea2Mix_v50BF16 | +239 | 7.7011 | 0.9307 | — | — |
| moodyKrea2Mix_v60BF16 | +221 | 10.2539 | 0.9344 | 0.097993 | 0.965625 |

---

## HSWQ Hybrid NVFP4 vs Native NVFP4 comparison

Same setup (vs BF16 reference). **HSWQ hybrid NVFP4** vs baseline **Native NVFP4** (full-model NVFP4).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; positive Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native NVFP4** = full-model NVFP4 quantize.

| Model | NVFP4 layers | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|--------------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| moodyKrea2Mix_v50BF16 | +239 | 7.7011 | 8.1189 | +0.4178 | 0.9307 | 0.9280 | +0.0027 | Native NVFP4 | HSWQ |
| moodyKrea2Mix_v60BF16 | +221 | 10.2539 | 10.4133 | +0.1594 | 0.9344 | 0.9188 | +0.0156 | Native NVFP4 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **NVFP4 kept layers (`+N`):** Count of layers kept as NVFP4 in the hybrid output (e.g. `+239` = 239 layers in NVFP4; the rest stay INT8). Taken verbatim from each HSWQ run tag in `score_krea2_nvfp4.txt`.
- **Latent MSE / Latent Cosine:** Latent-space metrics (direct, no RGB projection). Only logged for `moodyKrea2Mix_v60BF16`; `moodyKrea2Mix_v50BF16` log contains latent stats only.
- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
