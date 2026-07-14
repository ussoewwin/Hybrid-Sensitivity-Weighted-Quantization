# SDXL Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ INT8 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_sdxl_int8.txt`

---

## Results

| Model | Keep ratio | MSE (↓ better) | SSIM (↑ better) |
|-------|------------|----------------|-----------------|
| waiIllustriousSDXL_v170 | r0 | 3.62 | **0.9830** |
| unholyDesireMixSinister_v80 | r0 | 16.62 | 0.9586 |
| prefectIllustriousXL_v8 | r0 | 36.10 | 0.9440 |
| novaAnimeXL_ilV190 | r0 | 8.14 | 0.9582 |
| JANKUTrainedChenkinNoobai_v777 | r0 | 11.77 | **0.9801** |

---

## HSWQ vs Native INT8 comparison (partial)

Same setup (vs FP16 reference). **HSWQ INT8** vs baseline (see below).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native** = naive cast INT8. **Official INT8** = officially distributed INT8. Native and Official INT8 are not the same.

| Model | Keep | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Winner |
|-------|------|----------|--------------|-------|-----------|---------------|--------|--------|
| waiIllustriousSDXL_v170 | r0 | 3.62 | 22.58 | +18.96 | 0.9830 | 0.9631 | −0.0199 | HSWQ |
| unholyDesireMixSinister_v80 | r0 | 16.62 | 27.75 | +11.13 | 0.9586 | 0.9383 | −0.0203 | HSWQ |
| prefectIllustriousXL_v8 | r0 | 36.10 | 35.84 | −0.26 | 0.9440 | 0.9161 | −0.0279 | — |
| novaAnimeXL_ilV190 | r0 | 8.14 | 13.16 | +5.02 | 0.9582 | 0.9350 | −0.0232 | HSWQ |
| JANKUTrainedChenkinNoobai_v777 | r0 | 11.77 | 54.32 | +42.55 | 0.9801 | 0.9691 | −0.0110 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
- **Keep ratio:** Fraction of layers kept in FP16 (e.g. r0.1 = 10%, r0.15 = 15%). Blank = not recorded in source.
