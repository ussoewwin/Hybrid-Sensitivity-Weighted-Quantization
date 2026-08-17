# Z Image ConvRot NVFP4 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot NVFP4 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `benchmark result/score_zi_nvfp4.txt`

---

## Results

| Model | NVFP4 Layers | MSE (latent, ↓ better) | SSIM (decoded, ↑ better) | SSIM target >=0.9 |
|-------|--------------|------------------------|--------------------------|-------------------|
| moodyProMix_zitV13 | 60 | 0.0136 | 0.9769 | PASS |

---

## Performance & Memory

- **VRAM Usage:** `12359.3 MB` (FP16) → `5807.8 MB` (NVFP4+ConvRot)
- **VRAM Saved:** **6551.4 MB (53.0%)**
- **Inference Time:** `34.95s` (FP16) → `18.85s` (NVFP4+ConvRot)

---

## Notes

- **NVFP4 Layers:** Represents the number of layers quantized to NVFP4 (e.g., 60 means 60 NVFP4 layers).
- **MSE (latent):** Mean squared error on raw latent tensors vs FP16 reference; 0 = perfect match.
- **SSIM (decoded):** Structural similarity; 1.0 = perfect match. Target is >=0.9.
