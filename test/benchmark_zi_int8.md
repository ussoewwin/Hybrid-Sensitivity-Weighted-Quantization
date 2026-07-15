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

---

## HSWQ vs Native INT8 comparison

Same setup (vs FP16 reference). **HSWQ INT8** vs baseline (see below).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native** = naive cast INT8.

| Model | Keep | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| moodyProMix_zitV13 | r0 | 0.0258 | 0.0269 | +0.0011 | 0.9938 | 0.9798 | −0.0140 | Native INT8 | HSWQ |
| moodyRealMix_zitV7 | r0 | 0.0121 | 7.1742 | +7.1621 | 0.9983 | 0.9354 | −0.0629 | Native INT8 | HSWQ |

**Winner** = better on both MSE and SSIM (lower MSE and higher SSIM for HSWQ vs baseline).

---

## Notes

- **MSE (latent):** Mean squared error on raw latent tensors vs FP16 reference; 0 = perfect match.
- **SSIM (0–255 view):** Structural similarity on normalized 0–255 preview images (`zit_bench`); 1.0 = perfect match.
- **Keep ratio:** Fraction of layers kept in FP16 (e.g. r0.1 = 10%, r0.05 = 5%).

---

## Analysis & Key Findings (Z Image INT8, partial)

- **Important VRAM fact (HSWQ):** `12334.8 MB -> 6730.2~7134.3 MB`, saving **5200.4~5604.6 MB (42.2%~45.4%)**.
- **Important VRAM fact (Native INT8):** `12334.8 MB -> 6419.3 MB`, saving **5915.4 MB (48.0%)**.
- **Inference Time (FP16 → INT8):** Inference time improves from ~10.6-11.8s down to **~5.8-6.8s** (nearly 2x speedup).
