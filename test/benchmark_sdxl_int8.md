# SDXL ConvRot INT8 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot INT8 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_sdxl_int8.txt`

**Bias correction (column labels from the score log; same meaning as NVFP4):**

| Label | Meaning |
|-------|---------|
| `1on` | Bias correction **ON** |
| `1off` | Bias correction **OFF** |

---

## Results

| Model | Bias correction | MSE (↓ better) | SSIM (↑ better) |
|-------|-----------------|----------------|-----------------|
| waiREALISM_v10 | 1on | 9.6226 | 0.9527 |
| waiREALCN_v150 | 1on | 6.4697 | 0.9672 |
| waiIllustriousSDXL_v170 | 1off | 8.4104 | 0.9712 |
| waiANIPONYXL_v90 | 1on | 10.2265 | 0.9507 |
| waiANIPONYXL_v140 | 1on | 8.2714 | 0.9607 |
| uwazumimixILL_v50 | 1on | 2.4019 | 0.9818 |
| unholyDesireMixSinister_v80 | 1on | 4.2765 | 0.9821 |
| realvisxlV50_v50Bakedvae | 1on | 5.5277 | 0.9735 |
| realvisxlV50_v40Bakedvae | 1on | 4.6653 | 0.9837 |
| realvisxlV30_v30TurboBakedvae | 1on | 8.7272 | 0.9711 |
| prefectIllustriousXL_v8 | 1on | 19.9552 | 0.9448 |
| perfectionRealisticILXL_80 | 1on | 2.7393 | 0.9865 |
| perfectionAsianILXL_v10 | 1off | 8.4370 | 0.9755 |
| oneObsession_v23 | 1off | 13.3607 | 0.9694 |
| novaAsianXL_illustriousV70 | 1off | 4.5833 | 0.9798 |
| novaAnimeXL_ilV190 | 1on | 7.7007 | 0.9620 |
| koronemixVpred_v20 | 1off | 23.1918 | 0.9643 |
| koronemixIllustrious_v70 | 1on | 17.9882 | 0.9670 |
| epicrealismXL_pureFix | 1off | 7.9803 | 0.9763 |
| bluePencilXL_v031 | 1off | 14.8040 | 0.9442 |
| JANKUTrainedChenkinNoobai_v777 | 1off | 6.3061 | 0.9813 |

---

## HSWQ ConvRot INT8 vs Native ConvRot INT8 comparison

Same setup (vs FP16 reference). **HSWQ ConvRot INT8** vs baseline **Native ConvRot INT8**.  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native ConvRot INT8** = naive cast ConvRot INT8.

| Model | Bias correction | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|-----------------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| waiREALISM_v10 | 1on | 9.6226 | 9.7329 | +0.1103 | 0.9527 | 0.9522 | −0.0005 | Native ConvRot INT8 | HSWQ |
| waiREALCN_v150 | 1on | 6.4697 | 12.2876 | +5.8179 | 0.9672 | 0.9603 | −0.0069 | Native ConvRot INT8 | HSWQ |
| waiIllustriousSDXL_v170 | 1off | 8.4104 | 9.0242 | +0.6138 | 0.9712 | 0.9701 | −0.0011 | Native ConvRot INT8 | HSWQ |
| waiANIPONYXL_v90 | 1on | 10.2265 | 9.5987 | −0.6278 | 0.9507 | 0.9502 | −0.0005 | Native ConvRot INT8 | — |
| waiANIPONYXL_v140 | 1on | 8.2714 | 8.8365 | +0.5651 | 0.9607 | 0.9626 | +0.0019 | Native ConvRot INT8 | — |
| uwazumimixILL_v50 | 1on | 2.4019 | 4.9482 | +2.5463 | 0.9818 | 0.9758 | −0.0060 | Native ConvRot INT8 | HSWQ |
| unholyDesireMixSinister_v80 | 1on | 4.2765 | 7.6053 | +3.3288 | 0.9821 | 0.9797 | −0.0024 | Native ConvRot INT8 | HSWQ |
| realvisxlV50_v50Bakedvae | 1on | 5.5277 | 5.9401 | +0.4124 | 0.9735 | 0.9728 | −0.0007 | Native ConvRot INT8 | HSWQ |
| realvisxlV50_v40Bakedvae | 1on | 4.6653 | 5.6398 | +0.9745 | 0.9837 | 0.9751 | −0.0086 | Native ConvRot INT8 | HSWQ |
| realvisxlV30_v30TurboBakedvae | 1on | 8.7272 | 8.7135 | −0.0137 | 0.9711 | 0.9683 | −0.0028 | Native ConvRot INT8 | — |
| prefectIllustriousXL_v8 | 1on | 19.9552 | 41.2338 | +21.2786 | 0.9448 | 0.9315 | −0.0133 | Native ConvRot INT8 | HSWQ |
| perfectionRealisticILXL_80 | 1on | 2.7393 | 3.0018 | +0.2625 | 0.9865 | 0.9852 | −0.0013 | Native ConvRot INT8 | HSWQ |
| perfectionAsianILXL_v10 | 1off | 8.4370 | 4.3816 | −4.0554 | 0.9755 | 0.9894 | +0.0139 | Native ConvRot INT8 | Native |
| oneObsession_v23 | 1off | 13.3607 | 16.4857 | +3.1250 | 0.9694 | 0.9672 | −0.0022 | Native ConvRot INT8 | HSWQ |
| novaAsianXL_illustriousV70 | 1off | 4.5833 | 5.3430 | +0.7597 | 0.9798 | 0.9771 | −0.0027 | Native ConvRot INT8 | HSWQ |
| novaAnimeXL_ilV190 | 1on | 7.7007 | 13.1606 | +5.4599 | 0.9620 | 0.9350 | −0.0270 | Native ConvRot INT8 | HSWQ |
| koronemixVpred_v20 | 1off | 23.1918 | 20.1841 | −3.0077 | 0.9643 | 0.9754 | +0.0111 | Native ConvRot INT8 | Native |
| koronemixIllustrious_v70 | 1on | 17.9882 | 32.5487 | +14.5605 | 0.9670 | 0.9330 | −0.0340 | Native ConvRot INT8 | HSWQ |
| epicrealismXL_pureFix | 1off | 7.9803 | 8.7927 | +0.8124 | 0.9763 | 0.9756 | −0.0007 | Native ConvRot INT8 | HSWQ |
| bluePencilXL_v031 | 1off | 14.8040 | 20.3359 | +5.5319 | 0.9442 | 0.9365 | −0.0077 | Native ConvRot INT8 | HSWQ |
| JANKUTrainedChenkinNoobai_v777 | 1off | 6.3061 | 21.5584 | +15.2523 | 0.9813 | 0.9626 | −0.0187 | Native ConvRot INT8 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **Bias correction:** Each HSWQ run in `score_sdxl_int8.txt` is tagged `1on` or `1off` (same meaning as `test/score_convrotnvfp4.txt` / `test/benchmark_convrotnvfp4.md`).
  - **`1on`** = bias correction enabled for that convert / bench.
  - **`1off`** = bias correction disabled for that convert / bench.
- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
