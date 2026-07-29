# SDXL ConvRot NVFP4 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot NVFP4 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_convrotnvfp4.txt`

**Bias correction (column labels from the score log):**

| Label | Meaning |
|-------|---------|
| `1on` | Bias correction **ON** |
| `1off` | Bias correction **OFF** |

---

## Results

| Model | Bias correction | MSE (↓ better) | SSIM (↑ better) | NVFP4 TC (hits/fallbacks) |
|-------|-----------------|----------------|-----------------|---------------------------|
| waiIllustriousSDXL_v170 | 1off | 16.9641 | 0.9596 | 2425 / 0 |
| uwazumimixILL_v50 | 1on | 9.9430 | 0.9530 | 0 / 0 |
| unholyDesireMixSinister_v80 | 1off | 9.5110 | 0.9725 | 0 / 0 |
| realvisxlV50_v50Bakedvae | 1off | 16.5263 | 0.9585 | 0 / 0 |
| realvisxlV50_v40Bakedvae | 1off | 18.3663 | 0.9549 | 0 / 0 |
| realvisxlV30_v30TurboBakedvae | 1on | 30.1016 | 0.9315 | 0 / 0 |
| prefectIllustriousXL_v8 | 1off | 58.1917 | 0.9310 | 2450 / 0 |
| oneObsession_v23 | 1on | 24.7509 | 0.9490 | 0 / 0 |
| novaAsianXL_illustriousV70 | 1on | 20.0107 | 0.9344 | 0 / 0 |
| novaAnimeXL_ilV190 | 1on | 12.8300 | 0.9238 | 2425 / 0 |
| koronemixVpred_v20 | 1on | 29.8958 | 0.9548 | 2525 / 0 |
| koronemixIllustrious_v70 | 1off | 14.0383 | 0.9648 | 2425 / 0 |
| JANKUTrainedChenkinNoobai_v777 | 1on | 25.7451 | 0.9398 | 2525 / 0 |
| epicrealismXL_pureFix | 1off | 11.5932 | 0.9677 | 2550 / 0 |
| ebaraPonyXL_v21 | 1off | 13.2105 | 0.9382 | 2600 / 0 |
| animemix_v80 | 1off | 6.8081 | 0.9829 | 2450 / 0 |

---

## HSWQ ConvRot NVFP4 vs Native NVFP4 comparison

Same setup (vs FP16 reference). **HSWQ ConvRot NVFP4** vs baseline **Native NVFP4**.  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native NVFP4** = naive cast NVFP4.

| Model | Bias correction | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | HSWQ TC | Baseline TC | Baseline | Winner |
|-------|-----------------|----------|--------------|-------|-----------|---------------|--------|---------|-------------|----------|--------|
| waiIllustriousSDXL_v170 | 1off | 16.9641 | 36.7770 | +19.8129 | 0.9596 | 0.9346 | −0.0250 | 2425 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| uwazumimixILL_v50 | 1on | 9.9430 | 38.0450 | +28.1020 | 0.9530 | 0.8909 | −0.0621 | 0 / 0 | 0 / 0 | Native NVFP4 | HSWQ |
| unholyDesireMixSinister_v80 | 1off | 9.5110 | 26.6469 | +17.1359 | 0.9725 | 0.9520 | −0.0205 | 0 / 0 | 0 / 0 | Native NVFP4 | HSWQ |
| realvisxlV50_v50Bakedvae | 1off | 16.5263 | 24.4336 | +7.9073 | 0.9585 | 0.9297 | −0.0288 | 0 / 0 | 0 / 0 | Native NVFP4 | HSWQ |
| realvisxlV50_v40Bakedvae | 1off | 18.3663 | 32.2806 | +13.9143 | 0.9549 | 0.9184 | −0.0365 | 0 / 0 | 0 / 0 | Native NVFP4 | HSWQ |
| realvisxlV30_v30TurboBakedvae | 1on | 30.1016 | 62.4464 | +32.3448 | 0.9315 | 0.9019 | −0.0296 | 0 / 0 | 0 / 0 | Native NVFP4 | HSWQ |
| prefectIllustriousXL_v8 | 1off | 58.1917 | 99.4071 | +41.2154 | 0.9310 | 0.9212 | −0.0098 | 2450 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| oneObsession_v23 | 1on | 24.7509 | 81.3796 | +56.6287 | 0.9490 | 0.9155 | −0.0335 | 0 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| novaAsianXL_illustriousV70 | 1on | 20.0107 | 33.4618 | +13.4511 | 0.9344 | 0.8866 | −0.0478 | 0 / 0 | 0 / 0 | Native NVFP4 | HSWQ |
| novaAnimeXL_ilV190 | 1on | 12.8300 | 55.7887 | +42.9587 | 0.9238 | 0.9119 | −0.0119 | 2425 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| koronemixVpred_v20 | 1on | 29.8958 | 54.1957 | +24.2999 | 0.9548 | 0.9148 | −0.0400 | 2525 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| koronemixIllustrious_v70 | 1off | 14.0383 | 104.7592 | +90.7209 | 0.9648 | 0.8854 | −0.0794 | 2425 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| JANKUTrainedChenkinNoobai_v777 | 1on | 25.7451 | 102.9044 | +77.1593 | 0.9398 | 0.9158 | −0.0240 | 2525 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| epicrealismXL_pureFix | 1off | 11.5932 | 36.9909 | +25.3977 | 0.9677 | 0.9590 | −0.0087 | 2550 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| ebaraPonyXL_v21 | 1off | 13.2105 | 88.0444 | +74.8339 | 0.9382 | 0.8611 | −0.0771 | 2600 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |
| animemix_v80 | 1off | 6.8081 | 23.3225 | +16.5144 | 0.9829 | 0.9479 | −0.0350 | 2450 / 0 | 18575 / 0 | Native NVFP4 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **Bias correction:** Each HSWQ run in `score_convrotnvfp4.txt` is tagged `1on` or `1off`.
  - **`1on`** = bias correction enabled for that convert / bench.
  - **`1off`** = bias correction disabled for that convert / bench.
- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
- **NVFP4 TC:** NVFP4 Tensor Core matmul `hits` / `fallbacks`.
