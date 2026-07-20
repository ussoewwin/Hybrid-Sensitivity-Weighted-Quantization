# SDXL ConvRot INT8 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot INT8 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_sdxl_int8.txt`

---

## Results

| Model | Keep ratio | MSE (↓ better) | SSIM (↑ better) |
|-------|------------|----------------|-----------------|
| waiIllustriousSDXL_v170 | r0 | 8.4104 | 0.9712 |
| prefectIllustriousXL_v8 | r0 | 36.0997 | 0.9440 |
| novaAnimeXL_ilV190 | r0 | 8.1381 | 0.9582 |
| unholyDesireMixSinister_v80 | r0 | 16.6191 | 0.9586 |
| waiANIPONYXL_v140 | r0 | 6.7943 | 0.9669 |
| perfectionRealisticILXL_80 | r0 | 8.8473 | 0.9810 |
| waiREALCN_v150 | r0 | 6.7854 | 0.9644 |
| oneObsession_v23 | r0 | 13.3607 | 0.9694 |
| bluePencilXL_v031 | r0 | 14.8040 | 0.9442 |
| JANKUTrainedChenkinNoobai_v777 | r0 | 6.3061 | 0.9813 |
| epicrealismXL_pureFix | r0 | 7.9803 | 0.9763 |

---

## HSWQ ConvRot INT8 vs Native ConvRot INT8 comparison (partial)

Same setup (vs FP16 reference). **HSWQ ConvRot INT8** vs baseline (see below).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native** = naive cast INT8. **Official INT8** = officially distributed INT8. Native and Official INT8 are not the same.

| Model | Keep | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| waiIllustriousSDXL_v170 | r0 | 8.4104 | 9.0242 | +0.6138 | 0.9712 | 0.9701 | −0.0011 | Native ConvRot INT8 | HSWQ |
| prefectIllustriousXL_v8 | r0 | 36.0997 | 35.8356 | −0.2641 | 0.9440 | 0.9161 | −0.0279 | Native INT8 | — |
| novaAnimeXL_ilV190 | r0 | 8.1381 | 13.1606 | +5.0225 | 0.9582 | 0.9350 | −0.0232 | Native INT8 | HSWQ |
| unholyDesireMixSinister_v80 | r0 | 16.6191 | 27.7515 | +11.1324 | 0.9586 | 0.9383 | −0.0203 | Native INT8 | HSWQ |
| waiANIPONYXL_v140 | r0 | 6.7943 | 49.6060 | +42.8117 | 0.9669 | 0.8912 | −0.0757 | Native INT8 | HSWQ |
| perfectionRealisticILXL_80 | r0 | 8.8473 | 12.4224 | +3.5751 | 0.9810 | 0.9778 | −0.0032 | Native INT8 | HSWQ |
| waiREALCN_v150 | r0 | 6.7854 | 16.9499 | +10.1645 | 0.9644 | 0.8953 | −0.0691 | Native INT8 | HSWQ |
| oneObsession_v23 | r0 | 13.3607 | 16.4857 | +3.1250 | 0.9694 | 0.9672 | −0.0022 | Native ConvRot INT8 | HSWQ |
| bluePencilXL_v031 | r0 | 14.8040 | 20.3359 | +5.5319 | 0.9442 | 0.9365 | −0.0077 | Native ConvRot INT8 | HSWQ |
| JANKUTrainedChenkinNoobai_v777 | r0 | 6.3061 | 21.5584 | +15.2523 | 0.9813 | 0.9626 | −0.0187 | Native INT8 | HSWQ |
| epicrealismXL_pureFix | r0 | 7.9803 | 8.7927 | +0.8124 | 0.9763 | 0.9756 | −0.0007 | Native ConvRot INT8 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
