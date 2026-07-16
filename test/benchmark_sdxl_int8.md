# SDXL Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ INT8 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_sdxl_int8.txt`

---

## Results

| Model | Keep ratio | MSE (↓ better) | SSIM (↑ better) |
|-------|------------|----------------|-----------------|
| waiIllustriousSDXL_v170 | r0 | 3.60 | 0.9808 |
| prefectIllustriousXL_v8 | r0 | 36.10 | 0.9440 |
| JANKUTrainedChenkinNoobai_v777 | r0 | 11.77 | 0.9801 |
| novaAnimeXL_ilV190 | r0 | 8.14 | 0.9582 |
| unholyDesireMixSinister_v80 | r0 | 16.62 | 0.9586 |
| waiANIPONYXL_v140 | r0 | 6.79 | 0.9669 |
| bluePencilXL_v031 | r0 | 20.31 | 0.9287 |
| perfectionRealisticILXL_80 | r0 | 8.85 | 0.9810 |
| animemix_v80 | r0 | 12.69 | 0.9821 |

---

## HSWQ vs Native INT8 comparison (partial)

Same setup (vs FP16 reference). **HSWQ INT8** vs baseline (see below).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native** = naive cast INT8. **Official INT8** = officially distributed INT8. Native and Official INT8 are not the same.

| Model | Keep | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| waiIllustriousSDXL_v170 | r0 | 3.60 | 43.11 | +39.51 | 0.9808 | 0.9731 | −0.0077 | Native INT8 | HSWQ |
| prefectIllustriousXL_v8 | r0 | 36.10 | 35.84 | −0.26 | 0.9440 | 0.9161 | −0.0279 | Native INT8 | — |
| JANKUTrainedChenkinNoobai_v777 | r0 | 11.77 | 54.32 | +42.55 | 0.9801 | 0.9691 | −0.0110 | Native INT8 | HSWQ |
| novaAnimeXL_ilV190 | r0 | 8.14 | 13.16 | +5.02 | 0.9582 | 0.9350 | −0.0232 | Native INT8 | HSWQ |
| unholyDesireMixSinister_v80 | r0 | 16.62 | 27.75 | +11.13 | 0.9586 | 0.9383 | −0.0203 | Native INT8 | HSWQ |
| waiANIPONYXL_v140 | r0 | 6.79 | 49.61 | +42.82 | 0.9669 | 0.8912 | −0.0757 | Native INT8 | HSWQ |
| bluePencilXL_v031 | r0 | 20.31 | 57.05 | +36.74 | 0.9287 | 0.9036 | −0.0251 | Native INT8 | HSWQ |
| perfectionRealisticILXL_80 | r0 | 8.85 | 12.42 | +3.57 | 0.9810 | 0.9778 | −0.0032 | Native INT8 | HSWQ |
| animemix_v80 | r0 | 12.69 | 50.48 | +37.79 | 0.9821 | 0.9550 | −0.0271 | Native FP8 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.

