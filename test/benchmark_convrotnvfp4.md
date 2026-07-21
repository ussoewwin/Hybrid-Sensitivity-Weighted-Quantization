# SDXL ConvRot NVFP4 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot NVFP4 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_convrotnvfp4.txt`

---

## Results

| Model | MSE (↓ better) | SSIM (↑ better) | NVFP4 TC (hits/fallbacks) |
|-------|----------------|-----------------|---------------------------|
| animemix_v80 | 6.8081 | 0.9829 | 2450 / 0 |
| waiIllustriousSDXL_v170 | 16.7271 | 0.9593 | 2425 / 0 |
| koronemixIllustrious_v70 | 14.0383 | 0.9648 | 2425 / 0 |

---

## HSWQ ConvRot NVFP4 vs Native NVFP4 comparison

Same setup (vs FP16 reference). **HSWQ ConvRot NVFP4** vs baseline **Native NVFP4**.  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native NVFP4** = naive cast NVFP4.

| Model | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | HSWQ TC | Baseline TC | Baseline | Winner |
|-------|----------|--------------|-------|-----------|---------------|--------|---------|-------------|----------|--------|
| animemix_v80 | 6.8081 | 23.3225 | +16.5144 | 0.9829 | 0.9479 | −0.0350 | 2450 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| waiIllustriousSDXL_v170 | 16.7271 | 36.7770 | +20.0499 | 0.9593 | 0.9346 | −0.0247 | 2425 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| koronemixIllustrious_v70 | 14.0383 | 104.7592 | +90.7209 | 0.9648 | 0.8854 | −0.0794 | 2425 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
- **NVFP4 TC:** NVFP4 Tensor Core matmul `hits` / `fallbacks`.
