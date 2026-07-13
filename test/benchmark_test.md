# SDXL Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ FP8 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score.txt`

---

## Results

| Model | Keep ratio | MSE (↓ better) | SSIM (↑ better) |
|-------|------------|----------------|-----------------|
| waiREALISM_v10 | r0.1 | 10.83 | **0.9593** |
| waiREALCN_v150 | r0.15 | 31.20 | 0.9317 |
| waiIllustriousSDXL_v160 | r0.1 | 19.05 | 0.9333 |
| waiIllustriousSDXL_v170 | r0 | 23.87 | 0.9330 |
| waiANIPONYXL_v140 | r0 | 12.88 | **0.9561** |
| waiANIPONYXL_v90 | r0 | 11.89 | **0.9587** |
| uwazumimixILL_v50 | r0 | 13.76 | **0.9641** |
| unholyDesireMixSinister_v60 | r0.15 | 10.29 | 0.9336 |
| unholyDesireMixSinister_v80 | r0 | 10.00 | **0.9553** |
| realvisxlV50_v50Bakedvae | r0.1 | 58.81 | 0.9452 |
| realvisxlV50_v40Bakedvae | r0.1 | 33.54 | **0.9751** |
| realvisxlV30_v30TurboBakedvae | r0.1 | 15.15 | 0.9367 |
| prefectIllustriousXL_v70 | r0.1 | 17.14 | 0.9157 |
| prefectIllustriousXL_v8 | r0.1 | 14.69 | 0.9358 |
| perfectionRealisticILXL_60 | r0.1 | 11.02 | **0.9677** |
| perfectionRealisticILXL_80 | r0.1 | 19.01 | **0.9628** |
| perfectionAsianILXL_v10 | r0.1 | 8.56 | **0.9732** |
| obsessionIllustrious_vPredV20 | r0.1 | 10.23 | **0.9866** |
| novaAsianXL_illustriousV70 | r0.1 | 14.84 | **0.9620** |
| luminarqmixV8Noobaixl_v82 | r0.1 | 10.84 | **0.9683** |
| koronemixVpred_v20 | r0.1 | 13.77 | **0.9622** |
| koronemixIllustrious_v70 | r0.15 | 12.76 | **0.9735** |
| JANKUTrainedNoobaiRouwei_v69 | r0.25 | 10.97 | **0.9614** |
| JANKUTrainedChenkinNoobai_v777 | r0.1 | 19.83 | **0.9575** |
| harukiMIX_ponyV40 | r0.15 | 14.49 | **0.9645** |
| harukiMIX_illustriousV40 | r0.1 | 6.79 | **0.9715** |
| epicrealismXL_pureFix | r0.1 | 6.82 | **0.9783** |
| ebaraPonyXL_v21 | r0.1 | 30.14 | 0.9349 |
| cyberrealistic_v100Redux | r0.1 | 29.09 | **0.9749** |
| cottonnoob_v50 | r0.1 | 6.46 | **0.9877** |
| bluePencilXL_v031 | r0.1 | 24.48 | 0.9006 |
| asianRealismByStable_v30FP16 | r0.1 | 30.26 | 0.9129 |
| animagineXLV31_v30 | r0.1 | 18.25 | 0.9101 |
| animemix_v80 | r0.1 | 17.80 | 0.9297 |
| novaAnimeXL_ilV190 | r0.1 | 18.97 | 0.9315 |
| oneObsession_v21Anime | r0.1 | 23.09 | 0.9109 |
| oneObsession_v22Anime | r0 | 14.15 | **0.9575** |

---

## HSWQ vs Native FP8 comparison (partial)

Same setup (vs FP16 reference). **HSWQ FP8** vs baseline (see below).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native** = naive cast FP8. **Official FP8** = officially distributed FP8. Native and Official FP8 are not the same.

| Model | Keep | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Winner |
|-------|------|----------|--------------|-------|-----------|---------------|--------|--------|
| waiREALISM_v10 | r0.1 | 10.83 | 14.44 | +3.62 | 0.9593 | 0.9340 | −0.0253 | HSWQ |
| waiREALCN_v150 | r0.15 | 31.20 | 27.45 | −3.75 | 0.9317 | 0.9335 | +0.0018 | Native |
| waiIllustriousSDXL_v160 | r0.1 | 19.05 | 46.93 | +27.88 | 0.9333 | 0.8864 | −0.0469 | HSWQ |
| waiIllustriousSDXL_v170 | r0 | 23.87 | 40.11 | +16.24 | 0.9330 | 0.9040 | −0.0290 | HSWQ |
| waiANIPONYXL_v140 | r0 | 12.88 | 18.70 | +5.82 | 0.9561 | 0.9574 | +0.0013 | — |
| waiANIPONYXL_v90 | r0 | 11.89 | 16.86 | +4.97 | 0.9587 | 0.9479 | −0.0108 | HSWQ |
| uwazumimixILL_v50 | r0 | 13.76 | 11.80 | −1.96 | 0.9641 | 0.9542 | −0.0099 | — |
| unholyDesireMixSinister_v60 | r0.15 | 10.29 | 39.70 | +29.41 | 0.9336 | 0.8694 | −0.0642 | HSWQ |
| unholyDesireMixSinister_v80 | r0 | 10.00 | 14.54 | +4.54 | 0.9553 | 0.9425 | −0.0128 | HSWQ |
| realvisxlV50_v50Bakedvae | r0.1 | 58.81 | 69.70 | +10.89 | 0.9452 | 0.9377 | −0.0075 | HSWQ |
| realvisxlV50_v40Bakedvae | r0.1 | 33.54 | 31.09 | −2.45 | 0.9751 | 0.9558 | −0.0193 | — |
| realvisxlV30_v30TurboBakedvae | r0.1 | 15.15 | 44.62 | +29.47 | 0.9367 | 0.8888 | −0.0479 | HSWQ |
| prefectIllustriousXL_v70 | r0.1 | 17.14 | 22.25 | +5.11 | 0.9157 | 0.9096 | −0.0061 | HSWQ |
| prefectIllustriousXL_v8 | r0.1 | 14.69 | 29.27 | +14.58 | 0.9358 | 0.9177 | −0.0181 | HSWQ |
| perfectionRealisticILXL_60 | r0.1 | 11.02 | 34.08 | +23.06 | 0.9677 | 0.9280 | −0.0397 | HSWQ |
| perfectionRealisticILXL_80 | r0.1 | 19.01 | 57.63 | +38.63 | 0.9628 | 0.9427 | −0.0201 | HSWQ |
| perfectionAsianILXL_v10 | r0.1 | 8.56 | 22.37 | +13.81 | 0.9732 | 0.9596 | −0.0136 | HSWQ |
| obsessionIllustrious_vPredV20 | r0.1 | 10.23 | 43.67 | +33.44 | 0.9866 | 0.9626 | −0.0240 | HSWQ |
| novaAsianXL_illustriousV70 | r0.1 | 14.84 | 19.45 | +4.61 | 0.9620 | 0.9445 | −0.0175 | HSWQ |
| luminarqmixV8Noobaixl_v82 | r0.1 | 10.84 | 11.63 | +0.79 | 0.9683 | 0.9604 | −0.0079 | HSWQ |
| koronemixVpred_v20 | r0.1 | 13.77 | 14.55 | +0.78 | 0.9622 | 0.9590 | −0.0032 | HSWQ |
| koronemixIllustrious_v70 | r0.15 | 12.76 | 27.09 | +14.33 | 0.9735 | 0.9610 | −0.0125 | HSWQ |
| JANKUTrainedNoobaiRouwei_v69 | r0.25 | 10.97 | 94.81 | +83.83 | 0.9614 | 0.8872 | −0.0742 | HSWQ |
| JANKUTrainedChenkinNoobai_v777 | r0.1 | 19.83 | 26.37 | +6.54 | 0.9575 | 0.9546 | −0.0029 | HSWQ |
| harukiMIX_ponyV40 | r0.15 | 14.49 | 23.65 | +9.17 | 0.9645 | 0.9301 | −0.0344 | HSWQ |
| harukiMIX_illustriousV40 | r0.1 | 6.79 | 9.32 | +2.53 | 0.9715 | 0.9685 | −0.0030 | HSWQ |
| epicrealismXL_pureFix | r0.1 | 6.82 | 26.79 | +19.96 | 0.9783 | 0.9579 | −0.0204 | HSWQ |
| ebaraPonyXL_v21 | r0.1 | 30.14 | 33.50 | +3.36 | 0.9349 | 0.9203 | −0.0146 | HSWQ |
| cyberrealistic_v100Redux | r0.1 | 29.09 | 79.72 | +50.63 | 0.9749 | 0.9322 | −0.0427 | HSWQ |
| cottonnoob_v50 | r0.1 | 6.46 | 22.28 | +15.83 | 0.9877 | 0.9524 | −0.0353 | HSWQ |
| bluePencilXL_v031 | r0.1 | 24.48 | 41.67 | +17.19 | 0.9006 | 0.8808 | −0.0198 | HSWQ |
| asianRealismByStable_v30FP16 | r0.1 | 30.26 | 12.00 | −18.26 | 0.9129 | 0.9432 | +0.0303 | Official FP8 |
| animagineXLV31_v30 | r0.1 | 18.25 | 51.77 | +33.52 | 0.9101 | 0.8775 | −0.0326 | HSWQ |
| animemix_v80 | r0.1 | 17.80 | 11.15 | −6.66 | 0.9297 | 0.9512 | +0.0215 | Native |
| novaAnimeXL_ilV190 | r0.1 | 18.97 | 25.78 | +6.82 | 0.9315 | 0.9181 | −0.0134 | HSWQ |
| oneObsession_v21Anime | r0.1 | 23.09 | 27.77 | +4.67 | 0.9109 | 0.9084 | −0.0025 | HSWQ |
| oneObsession_v22Anime | r0 | 14.15 | 12.03 | −2.13 | 0.9575 | 0.9629 | +0.0054 | Native |

**Winner** = better on both MSE and SSIM. For asianRealismByStable_v30FP16, the publisher distributes an official FP8 version; that official FP8 outperforms HSWQ.

---

## Notes

- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
- **Keep ratio:** Fraction of layers kept in FP16 (e.g. r0.1 = 10%, r0.15 = 15%). Blank = not recorded in source.
