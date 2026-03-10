# SDXL Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ FP8 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score.txt`

---

## Results

| Model | Keep ratio | MSE (↓ better) | SSIM (↑ better) |
|-------|------------|----------------|-----------------|
| waiREALISM_v10 | r0.1 | 10.72 | **0.9538** |
| waiREALCN_v150 | r0.15 | 31.20 | 0.9317 |
| waiIllustriousSDXL_v160 | r0.1 | 19.05 | 0.9333 |
| waiANIPONYXL_v140 | r0.15 | 15.64 | 0.9361 |
| waiANIPONYXL_v11 | r0.15 | 18.49 | 0.9233 |
| uwazumimixILL_v50 | r0.05 | 14.26 | **0.9679** |
| unholyDesireMixSinister_v60 | r0.15 | 10.29 | 0.9336 |
| realvisxlV50_v50Bakedvae | r0.1 | 58.81 | 0.9452 |
| realvisxlV50_v40Bakedvae | r0.1 | 33.54 | **0.9751** |
| ealvisxlV30_v30TurboBakedvae | r0.1 | 15.15 | 0.9367 |
| prefectIllustriousXL_v70 | r0.1 | 17.14 | 0.9157 |
| perfectionRealisticILXL_60 | r0.1 | 11.02 | 0.9677 |
| perfectionAsianILXL_v10 | r0.1 | 8.56 | 0.9732 |
| obsessionIllustrious_vPredV20 | r0.1 | 10.23 | **0.9866** |
| novaAsianXL_illustriousV70 | r0.1 | 14.84 | 0.9620 |
| luminarqmixV8Noobaixl_v82 | r0.1 | 10.84 | **0.9683** |
| koronemixVpred_v20 | r0.1 | 13.77 | 0.9622 |
| koronemixIllustrious_v70 | r0.15 | 12.76 | **0.9735** |
| JANKUTrainedNoobaiRouwei_v69 | r0.25 | 10.97 | 0.9614 |
| harukiMIX_ponyV40 | r0.15 | 14.49 | 0.9645 |
| harukiMIX_illustriousV40 | r0.1 | 6.79 | **0.9715** |
| epicrealismXL_pureFix | r0.1 | 6.82 | **0.9783** |
| ebaraPonyXL_v21 | r0.1 | 30.14 | 0.9349 |
| cyberrealistic_v100Redux | r0.1 | 29.09 | **0.9749** |
| cottonnoob_v50 | r0.1 | 6.46 | **0.9877** |
| bluePencilXL_v031 | r0.1 | 24.48 | 0.9006 |
| asianRealismByStable_v30FP16 | r0.1 | 30.26 | 0.9129 |

---

## HSWQ vs Native FP8 comparison (partial)

Same setup (vs FP16 reference). **HSWQ FP8** vs baseline (see below).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; positive Δ SSIM ⇒ HSWQ better).  
**Native** = naive cast FP8. **Official FP8** = officially distributed FP8. Native and Official FP8 are not the same.

| Model | Keep | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Winner |
|-------|------|----------|--------------|-------|-----------|---------------|--------|--------|
| waiREALISM_v10 | r0.1 | 10.72 | — | — | 0.9538 | — | — | — |
| waiREALCN_v150 | r0.15 | 31.20 | — | — | 0.9317 | — | — | — |
| waiIllustriousSDXL_v160 | r0.1 | 19.05 | — | — | 0.9333 | — | — | — |
| waiANIPONYXL_v140 | r0.15 | 15.64 | — | — | 0.9361 | — | — | — |
| waiANIPONYXL_v11 | r0.15 | 18.49 | — | — | 0.9233 | — | — | — |
| uwazumimixILL_v50 | r0.05 | 14.26 | — | — | 0.9679 | — | — | — |
| unholyDesireMixSinister_v60 | r0.15 | 10.29 | — | — | 0.9336 | — | — | — |
| realvisxlV50_v50Bakedvae | r0.1 | 58.81 | — | — | 0.9452 | — | — | — |
| realvisxlV50_v40Bakedvae | r0.1 | 33.54 | — | — | 0.9751 | — | — | — |
| ealvisxlV30_v30TurboBakedvae | r0.1 | 15.15 | — | — | 0.9367 | — | — | — |
| prefectIllustriousXL_v70 | r0.1 | 17.14 | — | — | 0.9157 | — | — | — |
| perfectionRealisticILXL_60 | r0.1 | 11.02 | — | — | 0.9677 | — | — | — |
| perfectionAsianILXL_v10 | r0.1 | 8.56 | — | — | 0.9732 | — | — | — |
| obsessionIllustrious_vPredV20 | r0.1 | 10.23 | — | — | 0.9866 | — | — | — |
| novaAsianXL_illustriousV70 | r0.1 | 14.84 | — | — | 0.9620 | — | — | — |
| luminarqmixV8Noobaixl_v82 | r0.1 | 10.84 | 11.63 | +0.79 | 0.9683 | 0.9604 | +0.0079 | HSWQ |
| koronemixVpred_v20 | r0.1 | 13.77 | 14.55 | +0.78 | 0.9622 | 0.9590 | +0.0032 | HSWQ |
| koronemixIllustrious_v70 | r0.15 | 12.76 | 27.09 | +14.33 | 0.9735 | 0.9610 | +0.0125 | HSWQ |
| JANKUTrainedNoobaiRouwei_v69 | r0.25 | 10.97 | 94.81 | +83.84 | 0.9614 | 0.8872 | +0.0742 | HSWQ |
| harukiMIX_ponyV40 | r0.15 | 14.49 | 23.65 | +9.16 | 0.9645 | 0.9301 | +0.0344 | HSWQ |
| harukiMIX_illustriousV40 | r0.1 | 6.79 | 9.32 | +2.53 | 0.9715 | 0.9685 | +0.0030 | HSWQ |
| epicrealismXL_pureFix | r0.1 | 6.82 | 26.79 | +19.97 | 0.9783 | 0.9579 | +0.0204 | HSWQ |
| ebaraPonyXL_v21 | r0.1 | 30.14 | 33.50 | +3.36 | 0.9349 | 0.9203 | +0.0146 | HSWQ |
| cyberrealistic_v100Redux | r0.1 | 29.09 | 79.72 | +50.63 | 0.9749 | 0.9322 | +0.0427 | HSWQ |
| cottonnoob_v50 | r0.1 | 6.46 | 22.28 | +15.82 | 0.9877 | 0.9524 | +0.0353 | HSWQ |
| bluePencilXL_v031 | r0.1 | 24.48 | 41.67 | +17.19 | 0.9006 | 0.8808 | +0.0198 | HSWQ |
| asianRealismByStable_v30FP16 | r0.1 | 30.26 | 12.00 | −18.26 | 0.9129 | 0.9432 | −0.0303 | Official FP8 |

**Winner** = better on both MSE and SSIM. For asianRealismByStable_v30FP16, the publisher distributes an official FP8 version; that official FP8 outperforms HSWQ.

---

## Notes

- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
- **Keep ratio:** Fraction of layers kept in FP16 (e.g. r0.1 = 10%, r0.15 = 15%). Blank = not recorded in source.
- **Test environment:** RTX 5060 Ti 16GB, PyTorch 2.1.0 + CUDA 13.0.
