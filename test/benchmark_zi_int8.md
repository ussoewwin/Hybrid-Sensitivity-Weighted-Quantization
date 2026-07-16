# Z Image INT8 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ INT8 quantized** output (Z Image Turbo family).  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_zi_int8.txt`

---

## Results

| Model | Keep ratio | MSE (latent, ↓ better) | SSIM (0–255 view, ↑ better) |
|-------|------------|--------------------------|-----------------------------|
| moodyProMix_zitV13 | r0 | 0.0258 | 0.9938 |
| moodyRealMix_zitV7 | r0 | 0.0121 | 0.9983 |
| darkBeastINT8Convrot2_dbzit9DIMRclaw | r0 | 0.0071 | 0.9982 |
| beyondREALITY_V30 | r0 | 0.0078 | 0.9988 |
| unstableRevolution_V3Fp16 | r0 | 0.0117 | 0.9983 |
| Big Love | r0 | 0.0067 | 0.9987 |

---

## HSWQ vs Native INT8 comparison

Same setup (vs FP16 reference). **HSWQ INT8** vs baseline (see below).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native** = naive cast INT8. **Official INT8** = officially distributed INT8. Native and Official INT8 are not the same.

| Model | Keep | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| moodyProMix_zitV13 | r0 | 0.0258 | 0.0269 | +0.0011 | 0.9938 | 0.9798 | −0.0140 | Native INT8 | HSWQ |
| moodyRealMix_zitV7 | r0 | 0.0121 | 0.2344 | +0.2223 | 0.9983 | 0.9597 | −0.0386 | Official INT8 | HSWQ |
| darkBeastINT8Convrot2_dbzit9DIMRclaw | r0 | 0.0071 | 0.1251 | +0.1180 | 0.9982 | 0.9817 | −0.0165 | Native INT8 | HSWQ |
| beyondREALITY_V30 | r0 | 0.0078 | 0.1092 | +0.1014 | 0.9988 | 0.9865 | −0.0123 | Native INT8 | HSWQ |
| unstableRevolution_V3Fp16 | r0 | 0.0117 | 8.3628 | +8.3511 | 0.9983 | 0.9194 | −0.0789 | Native INT8 | HSWQ |
| Big Love | r0 | 0.0067 | 4.8617 | +4.8550 | 0.9987 | 0.9289 | −0.0698 | Native INT8 | HSWQ |

**Winner** = better on both MSE and SSIM (lower MSE and higher SSIM for HSWQ vs baseline).

---

## Notes

- **MSE (latent):** Mean squared error on raw latent tensors vs FP16 reference; 0 = perfect match.
- **SSIM (0–255 view):** Structural similarity on normalized 0–255 preview images (`zit_bench`); 1.0 = perfect match.


---

## Analysis & Key Findings (Z Image INT8, partial)

- **Important VRAM fact (HSWQ):** `12334.8 MB -> 6730.2~7134.3 MB`, saving **5200.4~5604.6 MB (42.2%~45.4%)**.
- **Important VRAM fact (Native INT8):** `12334.8 MB -> 6419.3 MB`, saving **5915.4 MB (48.0%)**.
- **Inference Time (FP16 → INT8):** Inference time improves from ~10.6-11.8s down to **~5.8-6.8s** (nearly 2x speedup).
