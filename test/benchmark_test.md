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

---

## Notes

- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
- **Keep ratio:** Fraction of layers kept in FP16 (e.g. r0.1 = 10%, r0.15 = 15%). Blank = not recorded in source.
