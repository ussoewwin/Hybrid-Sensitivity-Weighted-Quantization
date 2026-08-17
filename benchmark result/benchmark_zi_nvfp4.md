# Z Image ConvRot NVFP4 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot NVFP4 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `benchmark result/score_zi_nvfp4.txt`

---

## Results

| Model | NVFP4 Layers | MSE (latent, ↓ better) | SSIM (decoded, ↑ better) |
|-------|--------------|------------------------|--------------------------|
| moodyProMix_zitV13 | 80 | 0.3484 | 0.9700 |
| moodyProMix_collectorsEdition | 90 | 0.1200 | 0.9813 |
| moodyRealMix_zitV7 | 100 | 0.1445 | 0.9781 |
| moodyRealMix_xhsEdition | 110 | 0.1937 | 0.9909 |

---

## HSWQ ConvRot NVFP4 vs Native NVFP4 comparison

Same setup (vs FP16 reference). **HSWQ ConvRot NVFP4** vs baseline **Native NVFP4**.  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native NVFP4** = naive cast NVFP4.

| Model | NVFP4 Layers | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|--------------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| moodyProMix_zitV13 | 80 | 0.3484 | 0.9634 | +0.6150 | 0.9700 | 0.9548 | −0.0152 | Native NVFP4 | HSWQ |
| moodyProMix_collectorsEdition | 90 | 0.1200 | 0.7916 | +0.6716 | 0.9813 | 0.9094 | −0.0719 | Native NVFP4 | HSWQ |
| moodyRealMix_zitV7 | 100 | 0.1445 | 0.8502 | +0.7057 | 0.9781 | 0.8902 | −0.0879 | Native NVFP4 | HSWQ |
| moodyRealMix_xhsEdition | 110 | 0.1937 | 1.1270 | +0.9333 | 0.9909 | 0.9244 | −0.0665 | Native NVFP4 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **NVFP4 Layers:** Represents the number of layers quantized to NVFP4 (e.g., 80 means 80 NVFP4 layers).
- **MSE (latent):** Mean squared error on raw latent tensors vs FP16 reference; 0 = perfect match.
- **SSIM (decoded):** Structural similarity; 1.0 = perfect match. Target is >=0.9.
