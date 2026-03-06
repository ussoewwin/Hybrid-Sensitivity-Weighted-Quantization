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
| bluePencilXL_v031 | | 24.48 | 0.9006 |
| asianRealismByStable_v30FP16 | | 30.26 | 0.9129 |

---

## Notes

- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
- **Keep ratio:** Fraction of layers kept in FP16 (e.g. r0.1 = 10%, r0.15 = 15%). Blank = not recorded in source.
