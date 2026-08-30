# Flux1 ConvRot INT8 Benchmark Test Results

Benchmark comparison: **BF16 reference** vs **native ConvRot INT8 quantized** output (Flux1 DiT family).
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `benchmark result/score_flux_int8.txt`

**Conditions:** 1024x1024, 12 steps, euler/simple, guidance=3.5 (fixed), **20 seeds** (42, 137, 5517, 92048, 371506, 5293047, 64820153, 731509284, 8426170395, 9517038246, 210987, 6543210, 98765432, 1357924680, 2468135791, 3579246812, 4680357923, 5791468034, 6802579145, 7913680256), VAE: Ultra-flux1, CLIP: flan_t5_xxl_convrot_int8 + clip_l

---

## Results

| Model | Latent MSE (mean, ↓) | Latent Cos (mean, ↑) | Pixel MSE (mean, ↓) | SSIM (mean, ↑) | SSIM range |
|-------|----------------------|----------------------|---------------------|----------------|------------|
| redcraftHybridH3A2A_realreveal5 | 0.1746 | 0.9880 | 17.44 | 0.9221 | 0.7464–0.9914 |

### Per-seed detail (20 seeds)

| Seed | Latent MSE | Latent Cos | Pixel MSE | SSIM |
|------|------------|------------|-----------|------|
| 42 | 0.1037 | 0.9950 | 10.83 | 0.9485 |
| 137 | 0.5170 | 0.9498 | 44.68 | 0.7875 |
| 5517 | 0.0461 | 0.9975 | 9.46 | 0.9747 |
| 92048 | 0.3609 | 0.9613 | 52.70 | 0.7464 |
| 371506 | 0.1816 | 0.9955 | 5.58 | 0.9743 |
| 5293047 | 0.5389 | 0.9505 | 51.31 | 0.7748 |
| 64820153 | 0.0597 | 0.9980 | 14.10 | 0.9655 |
| 731509284 | 0.3693 | 0.9832 | 20.99 | 0.8853 |
| 8426170395 | 0.3184 | 0.9841 | 27.75 | 0.8746 |
| 9517038246 | 0.0063 | 0.9998 | 2.35 | 0.9914 |
| 210987 | 0.0121 | 0.9994 | 2.76 | 0.9881 |
| 6543210 | 0.0876 | 0.9971 | 8.28 | 0.9664 |
| 98765432 | 0.0575 | 0.9966 | 12.89 | 0.9516 |
| 1357924680 | 0.0343 | 0.9985 | 4.84 | 0.9745 |
| 2468135791 | 0.1143 | 0.9902 | 19.11 | 0.9414 |
| 3579246812 | 0.1555 | 0.9882 | 8.75 | 0.9308 |
| 4680357923 | 0.0488 | 0.9983 | 7.22 | 0.9724 |
| 5791468034 | 0.3558 | 0.9835 | 18.89 | 0.8966 |
| 6802579145 | 0.0665 | 0.9979 | 12.11 | 0.9507 |
| 7913680256 | 0.0570 | 0.9963 | 14.11 | 0.9458 |

---

## Performance

| Metric | FP16/BF16 | INT8 |
|--------|-----------|------|
| Inference time | 22.26 s/seed | 13.96 s/seed |
| Peak VRAM | 13.40 GiB | 12.21 GiB |

---

## Notes

- **native ConvRot INT8** = hswq-free plain compression (Hadamard rotation + per-out-channel INT8 + comfy_quant stamp).
- **Latent MSE / Cos**: computed in latent space before VAE decode (direct UNet quantization error).
- **Pixel MSE / SSIM**: decoded with Ultra-flux1 VAE (grayscale SSIM).
- Updated from the 5-seed run (SSIM mean 0.886) to the 20-seed run (0.922). Only seeds 137 (0.79), 92048 (0.75) and 5293047 (0.77) are poor; most seeds are above 0.93.
- This environment decodes INT8 to BF16 at ComfyUI load time, so VRAM saving is only 8.9% (a true INT8-kept runtime should roughly halve it).
- CLIP uses flan_t5_xxl (non-stock t5xxl). It does not affect fp16/int8 comparison fairness, but absolute quality / prompt understanding may differ from the stock t5xxl.
