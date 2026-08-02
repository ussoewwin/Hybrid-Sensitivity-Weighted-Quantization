# SDXL ConvRot INT8 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot INT8 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_sdxl_int8.txt`

**Bias correction (column labels from the score log):**

| Label | Meaning |
|-------|---------|
| `1on` | Bias correction **ON** |
| `1off` | Bias correction **OFF** |

---

## Results

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

---

## HSWQ ConvRot INT8 vs Native ConvRot INT8 comparison

Same setup (vs FP16 reference). **HSWQ ConvRot INT8** vs baseline **Native ConvRot INT8**.  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native ConvRot INT8** = naive cast ConvRot INT8.

| Model | Bias correction | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|-----------------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| bluePencilXL_v031 | 1off | 14.80 | 20.34 | +5.54 | 0.9442 | 0.9365 | −0.0077 | Native ConvRot INT8 | HSWQ |
| epicrealismXL_pureFix | 1off | 7.98 | 8.79 | +0.81 | 0.9763 | 0.9756 | −0.0007 | Native ConvRot INT8 | HSWQ |
| JANKUTrainedChenkinNoobai_v777 | 1off | 6.31 | 21.56 | +15.25 | 0.9813 | 0.9626 | −0.0187 | Native ConvRot INT8 | HSWQ |
| koronemixIllustrious_v70 | 1on | 17.99 | 32.55 | +14.56 | 0.9670 | 0.9330 | −0.0340 | Native ConvRot INT8 | HSWQ |
| koronemixVpred_v20 | 1off | 23.19 | 20.18 | −3.01 | 0.9643 | 0.9754 | +0.0111 | Native ConvRot INT8 | Native |
| novaAnimeXL_ilV190 | 1on | 7.70 | 13.16 | +5.46 | 0.9620 | 0.9350 | −0.0270 | Native ConvRot INT8 | HSWQ |
| novaAsianXL_illustriousV70 | 1off | 4.58 | 5.34 | +0.76 | 0.9798 | 0.9771 | −0.0027 | Native ConvRot INT8 | HSWQ |
| oneObsession_v23 | 1off | 13.36 | 16.49 | +3.13 | 0.9694 | 0.9672 | −0.0022 | Native ConvRot INT8 | HSWQ |
| perfectionAsianILXL_v10 | 1off | 8.44 | 4.38 | −4.06 | 0.9755 | 0.9894 | +0.0139 | Native ConvRot INT8 | Native |
| perfectionRealisticILXL_80 | 1on | 2.74 | 3.00 | +0.26 | 0.9865 | 0.9852 | −0.0013 | Native ConvRot INT8 | HSWQ |
| prefectIllustriousXL_v8 | 1on | 19.96 | 41.23 | +21.27 | 0.9448 | 0.9315 | −0.0133 | Native ConvRot INT8 | HSWQ |
| realvisxlV30_v30TurboBakedvae | 1on | 8.73 | 8.71 | −0.02 | 0.9711 | 0.9683 | −0.0028 | Native ConvRot INT8 | — |
| realvisxlV50_v40Bakedvae | 1on | 4.67 | 5.64 | +0.97 | 0.9837 | 0.9751 | −0.0086 | Native ConvRot INT8 | HSWQ |
| realvisxlV50_v50Bakedvae | 1on | 5.53 | 5.94 | +0.41 | 0.9735 | 0.9728 | −0.0007 | Native ConvRot INT8 | HSWQ |
| unholyDesireMixSinister_v80 | 1on | 4.28 | 7.61 | +3.33 | 0.9821 | 0.9797 | −0.0024 | Native ConvRot INT8 | HSWQ |
| uwazumimixILL_v50 | 1on | 2.40 | 4.95 | +2.55 | 0.9818 | 0.9758 | −0.0060 | Native ConvRot INT8 | HSWQ |
| waiANIPONYXL_v140 | 1on | 8.27 | 8.84 | +0.57 | 0.9607 | 0.9626 | +0.0019 | Native ConvRot INT8 | — |
| waiANIPONYXL_v90 | 1on | 10.23 | 9.60 | −0.63 | 0.9507 | 0.9502 | −0.0005 | Native ConvRot INT8 | — |
| waiIllustriousSDXL_v170 | 1off | 8.41 | 9.02 | +0.61 | 0.9712 | 0.9701 | −0.0011 | Native ConvRot INT8 | HSWQ |
| waiREALCN_v150 | 1on | 6.47 | 12.29 | +5.82 | 0.9672 | 0.9603 | −0.0069 | Native ConvRot INT8 | HSWQ |
| waiREALISM_v10 | 1on | 9.62 | 9.73 | +0.11 | 0.9527 | 0.9522 | −0.0005 | Native ConvRot INT8 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **Bias correction:** Each HSWQ run in `score_sdxl_int8.txt` is tagged `1on` or `1off`.
  - **`1on`** = bias correction enabled for that convert / bench.
  - **`1off`** = bias correction disabled for that convert / bench.
- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
