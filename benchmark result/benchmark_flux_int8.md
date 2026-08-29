# Flux1 ConvRot INT8 Benchmark Test Results

Benchmark comparison: **BF16 reference** vs **native ConvRot INT8 quantized** output (Flux1 DiT family).  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `benchmark result/score_flux_int8.txt`

**Conditions:** 1024x1024, 12 steps, euler/simple, guidance=3.5 (fixed), 5 seeds (42, 137, 5517, 92048, 371506), VAE: Ultra-flux1, CLIP: flan_t5_xxl_convrot_int8 + clip_l

---

## Results

| Model | Latent MSE (mean, ↓) | Latent Cos (mean, ↑) | Pixel MSE (mean, ↓) | SSIM (mean, ↑) | SSIM range |
|-------|----------------------|----------------------|---------------------|----------------|------------|
| redcraftHybridH3A2A_realreveal5 | 0.2419 | 0.9798 | 24.65 | 0.8863 | 0.7464–0.9747 |

### Per-seed detail

| Seed | Latent MSE | Latent Cos | Pixel MSE | SSIM |
|------|------------|------------|-----------|------|
| 42 | 0.1037 | 0.9950 | 10.83 | 0.9485 |
| 137 | 0.5170 | 0.9498 | 44.68 | 0.7875 |
| 5517 | 0.0461 | 0.9975 | 9.46 | 0.9747 |
| 92048 | 0.3609 | 0.9613 | 52.70 | 0.7464 |
| 371506 | 0.1816 | 0.9955 | 5.58 | 0.9743 |

---

## Performance

| Metric | FP16/BF16 | INT8 |
|--------|-----------|------|
| Inference time | 23.33 s/seed | 16.93 s/seed |
| Peak VRAM | 13.39 GiB | 12.21 GiB |

---

## Notes

- **native ConvRot INT8** = hswq 非使用の単純圧縮（Hadamard 回転 + per-out-channel INT8 + comfy_quant スタンプ）。
- **Latent MSE / Cos**: VAE decode 前の潜在空間で計算（UNet 量子化誤差を直接測定）。
- **Pixel MSE / SSIM**: Ultra-flux1 VAE で decode した画像で計算（grayscale SSIM）。
- シード依存のばらつきが大きい（SSIM 0.75–0.97）。seed 137 / 92048 は逸脱気味。
- この環境では int8 は ComfyUI ロード時に bf16 へデコードされる方式のため、VRAM 節約は 8.8% に留まる（本来の int8 保持実行なら約半減が見込まれる）。
- CLIP に flan_t5_xxl（非純正 t5xxl）を使用。fp16/int8 比較の公平性には影響しないが、絶対的な画質・プロンプト理解は純正 t5xxl と異なる可能性あり。
