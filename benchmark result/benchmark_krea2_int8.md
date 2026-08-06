# Krea2 ConvRot INT8 Benchmark Test Results

Benchmark comparison: **BF16 reference** vs **HSWQ ConvRot INT8 quantized** output (Krea2 family).  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `benchmark result/score_krea2_int8.txt`

**Column labels from the score log:**

| Label | Meaning |
|-------|---------|
| `+N` (e.g. `+15`, `+20`) | Number of **BF16 protect layers** added from dual-monitor analysis |
| `1on` | Bias correction **ON** |
| `1off` | Bias correction **OFF** |

---

## Results

| Model | BF16 protect layers | Bias correction | MSE (↓ better) | SSIM (↑ better) |
|-------|---------------------|-----------------|----------------|-----------------|
| unstableDissolution_Bf16 | +20 | 1off | 6.84 | 0.9700 |
| moodyKrea2Mix_v50BF16 | +15 | 1off | 2.57 | 0.9869 |
| moodyCutieMixKrea2_v20BF16 | +18 | 1off | 5.30 | 0.9757 |
| gonzalomoKrea2_v20 | +26 | 1off | 6.27 | 0.9573 |
| darkBeast30BF16INT8_darkBeast330 | +20 | 1off | 7.90 | 0.9613 |

---

## HSWQ ConvRot INT8 vs Native ConvRot INT8 comparison

Same setup (vs BF16 reference). **HSWQ ConvRot INT8** vs baseline **Native ConvRot INT8**.  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native ConvRot INT8** = naive cast ConvRot INT8.

| Model | BF16 protect layers | Bias correction | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|---------------------|-----------------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| unstableDissolution_Bf16 | +20 | 1off | 6.84 | 17.13 | +10.29 | 0.9700 | 0.8015 | −0.1685 | Native ConvRot INT8 | HSWQ |
| moodyKrea2Mix_v50BF16 | +15 | 1off | 2.57 | 10.49 | +7.92 | 0.9869 | 0.9039 | −0.0830 | Native ConvRot INT8 | HSWQ |
| moodyCutieMixKrea2_v20BF16 | +18 | 1off | 5.30 | 19.55 | +14.25 | 0.9757 | 0.8538 | −0.1219 | Native ConvRot INT8 | HSWQ |
| gonzalomoKrea2_v20 | +26 | 1off | 6.27 | 11.94 | +5.67 | 0.9573 | 0.8601 | −0.0972 | Native ConvRot INT8 | HSWQ |
| darkBeast30BF16INT8_darkBeast330 | +20 | 1off | 7.90 | 20.85 | +12.95 | 0.9613 | 0.7976 | −0.1637 | Native ConvRot INT8 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **BF16 protect layers (`+N`):** Count of BF16 protect layers added from dual-monitor analysis (e.g. `+15` = 15 layers, `+20` = 20 layers). Taken verbatim from each HSWQ run tag in `score_krea2_int8.txt`.
- **Bias correction:** Each HSWQ run in `score_krea2_int8.txt` is tagged `1on` or `1off`.
  - **`1on`** = bias correction enabled for that convert / bench.
  - **`1off`** = bias correction disabled for that convert / bench.
- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
