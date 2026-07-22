# SDXL ConvRot NVFP4 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot NVFP4 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_convrotnvfp4.txt`

---

## Results

| Model | MSE (↓ better) | SSIM (↑ better) | NVFP4 TC (hits/fallbacks) |
|-------|----------------|-----------------|---------------------------|
| waiIllustriousSDXL_v170 | 16.7271 | 0.9593 | 2425 / 0 |
| koronemixVpred_v20 | 44.9995 | 0.9404 | 2525 / 0 |
| koronemixIllustrious_v70 | 14.0383 | 0.9648 | 2425 / 0 |
| epicrealismXL_pureFix | 11.5932 | 0.9677 | 2550 / 0 |
| ebaraPonyXL_v21 | 13.2105 | 0.9382 | 2600 / 0 |
| animemix_v80 | 6.8081 | 0.9829 | 2450 / 0 |

---

## HSWQ ConvRot NVFP4 vs Native NVFP4 comparison

Same setup (vs FP16 reference). **HSWQ ConvRot NVFP4** vs baseline **Native NVFP4**.  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native NVFP4** = naive cast NVFP4.

| Model | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | HSWQ TC | Baseline TC | Baseline | Winner |
|-------|----------|--------------|-------|-----------|---------------|--------|---------|-------------|----------|--------|
| waiIllustriousSDXL_v170 | 16.7271 | 36.7770 | +20.0499 | 0.9593 | 0.9346 | −0.0247 | 2425 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| koronemixVpred_v20 | 44.9995 | 54.1957 | +9.1962 | 0.9404 | 0.9148 | −0.0256 | 2525 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| koronemixIllustrious_v70 | 14.0383 | 104.7592 | +90.7209 | 0.9648 | 0.8854 | −0.0794 | 2425 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| epicrealismXL_pureFix | 11.5932 | 36.9909 | +25.3977 | 0.9677 | 0.9590 | −0.0087 | 2550 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| ebaraPonyXL_v21 | 13.2105 | 53.4388 | +40.2283 | 0.9382 | 0.8914 | −0.0468 | 2600 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| animemix_v80 | 6.8081 | 23.3225 | +16.5144 | 0.9829 | 0.9479 | −0.0350 | 2450 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
- **NVFP4 TC:** NVFP4 Tensor Core matmul `hits` / `fallbacks`.
