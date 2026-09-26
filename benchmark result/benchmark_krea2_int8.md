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
| unstableDissolution_Bf16 | +20 | 1off | 6.52 | 0.9737 |
| moodyKrea2Mix_v50BF16 | +15 | 1off | 2.57 | 0.9869 |
| moodyCutieMixKrea2_v20BF16 | +0 | 1off | 4.79 | 0.9771 |
| gonzalomoKrea2_v20 | +10 | 1off | 6.26 | 0.9573 |
| fasciumKREA2_3MERGE | +10 | 1off | 6.81 | 0.9607 |
| darkBeast30BF16INT8_darkBeast330 | +10 | 1off | 7.98 | 0.9587 |

---

## HSWQ ConvRot INT8 vs Native ConvRot INT8 comparison

Same setup (vs BF16 reference). **HSWQ ConvRot INT8** vs baseline **Native ConvRot INT8**.  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native ConvRot INT8** = naive cast ConvRot INT8.

| Model | BF16 protect layers | Bias correction | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|---------------------|-----------------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| unstableDissolution_Bf16 | +20 | 1off | 6.52 | 10.92 | +4.40 | 0.9737 | 0.9355 | −0.0382 | Native ConvRot INT8 | HSWQ |
| moodyKrea2Mix_v50BF16 | +15 | 1off | 2.57 | 3.72 | +1.15 | 0.9869 | 0.9805 | −0.0064 | Native ConvRot INT8 | HSWQ |
| moodyCutieMixKrea2_v20BF16 | +0 | 1off | 4.79 | 12.83 | +8.04 | 0.9771 | 0.9184 | −0.0587 | Native ConvRot INT8 | HSWQ |
| gonzalomoKrea2_v20 | +10 | 1off | 6.26 | 6.63 | +0.37 | 0.9573 | 0.9531 | −0.0042 | Native ConvRot INT8 | HSWQ |
| fasciumKREA2_3MERGE | +10 | 1off | 6.81 | 11.44 | +4.63 | 0.9607 | 0.9170 | −0.0437 | Native ConvRot INT8 | HSWQ |
| darkBeast30BF16INT8_darkBeast330 | +10 | 1off | 7.98 | 10.74 | +2.76 | 0.9587 | 0.9396 | −0.0191 | Native ConvRot INT8 | HSWQ |

**Winner** = better on both MSE and SSIM.

---

## Notes

- **BF16 protect layers (`+N`):** Count of BF16 protect layers added from dual-monitor analysis (e.g. `+15` = 15 layers, `+20` = 20 layers). Taken verbatim from each HSWQ run tag in `score_krea2_int8.txt`.
- **Bias correction:** Each HSWQ run in `score_krea2_int8.txt` is tagged `1on` or `1off`.
  - **`1on`** = bias correction enabled for that convert / bench.
  - **`1off`** = bias correction disabled for that convert / bench.
- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
