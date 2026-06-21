# Z Image Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ FP8 quantized** output (Z Image Turbo family).  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `test/score_zi.txt`

---

## Results

| Model | Keep ratio | MSE (latent, ↓ better) | SSIM (0–255 view, ↑ better) |
|-------|------------|--------------------------|-----------------------------|
| z anime base | r0.05 | 31.60 | 0.9583 |
| unstableRevolution_V3Fp16 | r0.05 | 0.02 | 0.9913 |
| unstableRevolution_V2Fp16 | r0.05 | 0.01 | 0.9542 |
| moodyWildMix_v02 | r0.1 | 0.01 | 0.9582 |
| moodyRealMix_zitV7 | r0.0 | 0.03 | 0.9976 |
| moodyRealMix_zitV6DPO | r0.05 | 0.07 | 0.9919 |
| moodyRealMix_zitV5DPO | r0.05 | 0.01 | 0.9640 |
| moodyRealMix_zitV4DPO | r0.1 | 0.01 | 0.9618 |
| moodyProMix_zitV13 | r0.0 | 0.00 | 0.9964 |
| moodyProMix_zitV12DPO | r0.05 | 0.05 | 0.9918 |
| jibMixZIT_v20 | r0.05 | 0.01 | 0.9577 |
| harukiMIX_zit2603 | r0.05 | 0.01 | 0.9678 |
| darkBeastMar2126Latest_dbzit8SDAFOK | r0.05 | 0.02 | 0.9591 |
| bigLove_zt3 | r0.05 | 0.01 | 0.9607 |
| beyondREALITY_V30 | r0.05 | 0.02 | 0.9917 |
| 2127ZImageAsianUtopian_v36TurboFFV | r0.05 | 0.03 | 0.9493 |

---

## HSWQ vs Native FP8 / Official FP8 comparison

Same setup (vs FP16 reference). **HSWQ FP8** vs baseline (see below).  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native** = naive cast FP8. **Official FP8** = officially distributed FP8. Native and Official FP8 are not the same.

| Model | Keep | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
| z anime base | r0.05 | 31.60 | 50.96 | +19.36 | 0.9583 | 0.9427 | −0.0156 | Official FP8 | HSWQ |
| unstableRevolution_V3Fp16 | r0.05 | 0.02 | 0.06 | +0.04 | 0.9913 | 0.9759 | −0.0154 | Official FP8 | HSWQ |
| unstableRevolution_V2Fp16 | r0.05 | 0.01 | 0.02 | +0.01 | 0.9542 | 0.9195 | −0.0347 | Native FP8 | HSWQ |
| moodyWildMix_v02 | r0.1 | 0.01 | 0.02 | +0.01 | 0.9582 | 0.9297 | −0.0285 | Native FP8 | HSWQ |
| moodyRealMix_zitV7 | r0.0 | 0.03 | 14.84 | +14.81 | 0.9976 | 0.8731 | −0.1245 | Official FP8 | HSWQ |
| moodyRealMix_zitV6DPO | r0.05 | 0.07 | 0.10 | +0.03 | 0.9919 | 0.9899 | −0.0020 | Official FP8 | HSWQ |
| moodyRealMix_zitV5DPO | r0.05 | 0.01 | 0.01 | +0.00 | 0.9640 | 0.9346 | −0.0294 | Official FP8 | HSWQ |
| moodyRealMix_zitV4DPO | r0.1 | 0.01 | 0.02 | +0.01 | 0.9618 | 0.9343 | −0.0275 | Official FP8 | HSWQ |
| moodyProMix_zitV13 | r0.0 | 0.00 | 0.06 | +0.06 | 0.9964 | 0.9799 | −0.0165 | Official FP8 | HSWQ |
| moodyProMix_zitV12DPO | r0.05 | 0.05 | 0.39 | +0.34 | 0.9918 | 0.9603 | −0.0315 | Official FP8 | HSWQ |
| jibMixZIT_v20 | r0.05 | 0.01 | 0.04 | +0.03 | 0.9577 | 0.9269 | −0.0308 | Native FP8 | HSWQ |
| harukiMIX_zit2603 | r0.05 | 0.01 | 0.01 | +0.00 | 0.9678 | 0.9248 | −0.0430 | Native FP8 | HSWQ |
| darkBeastMar2126Latest_dbzit8SDAFOK | r0.05 | 0.02 | 0.03 | +0.01 | 0.9591 | 0.9177 | −0.0414 | Native FP8 | HSWQ |
| bigLove_zt3 | r0.05 | 0.01 | 0.01 | +0.00 | 0.9607 | 0.9230 | −0.0377 | Native FP8 | HSWQ |
| beyondREALITY_V30 | r0.05 | 0.02 | 0.10 | +0.08 | 0.9917 | 0.9748 | −0.0169 | Official FP8 | HSWQ |
| 2127ZImageAsianUtopian_v36TurboFFV | r0.05 | 0.03 | 0.05 | +0.02 | 0.9493 | 0.9226 | −0.0267 | Native FP8 | HSWQ |

**Winner** = better on both MSE and SSIM (lower MSE and higher SSIM for HSWQ vs baseline).

---

## Notes

- **MSE (latent):** Mean squared error on raw latent tensors vs FP16 reference; 0 = perfect match.
- **SSIM (0–255 view):** Structural similarity on normalized 0–255 preview images (`zit_bench`); 1.0 = perfect match.
- **Keep ratio:** Fraction of layers kept in FP16 (e.g. r0.1 = 10%, r0.05 = 5%).
- **Test environment (from logs):** Peak VRAM ~12.3 GB FP16 / ~7.3–7.5 GB HSWQ FP8 where applicable; see `score_zi.txt` blocks for per-run VRAM and inference time.
- **Z-Anime row:** The Z-Anime MSE values in this table are from the 0–255 view block in `score_zi.txt` (not latent-space MSE).

---

## Analysis & Key Findings (Z Image, partial)

- **Important VRAM fact (Z-Anime, HSWQ):** `12335.8 MB -> 9219.3 MB`, so **3116.5 MB (25.3%)** is saved.
- **Important VRAM fact (ZIT, HSWQ):** ZIT rows in the same file save **4825.6–5040.9 MB (39.1%–40.9%)**.
- **Reason in this log (numbers only):** FP16 peaks are almost equal (~`12335.7/12335.8 MB`), but Z-Anime FP8 peak (`9219.3 MB`) is higher than ZIT FP8 peaks (`7294.8–7510.1 MB`), so Z-Anime saved MB/% is lower.
- **Important baseline fact (Z-Anime official FP8):** `-0.0 MB (-0.0%)` saved (`FP8 12335.8 MB`), i.e. no VRAM reduction on the official FP8 baseline.

### Why Z-Anime VRAM saving is lower than ZIT (analysis)

Use the peak decomposition:

`M_peak = M_weights + M_activations + M_workspace + M_overhead`

and saved VRAM:

`saved = M_peak(FP16) - M_peak(FP8)`.

In this log, FP16 peaks are almost identical (Z-Anime `12335.8`, ZIT about `12335.7`), so the gap is determined by FP8 peak only:

- Z-Anime FP8 peak: `9219.3`
- ZIT FP8 peak: `7294.8–7510.1`

Therefore:

- Z-Anime saved: `12335.8 - 9219.3 = 3116.5 MB` (`25.3%`)
- ZIT saved range: `12335.7 - 7510.1 = 4825.6 MB` to `12335.7 - 7294.8 = 5040.9 MB` (`39.1%–40.9%`)

Direct difference in saved MB is exactly the FP8-peak gap:

- vs ZIT min-FP8 case: `9219.3 - 7294.8 = 1924.5 MB`
- vs ZIT max-FP8 case: `9219.3 - 7510.1 = 1709.2 MB`

So, in this benchmark file, Z-Anime saves less VRAM because its FP8 peak remains higher while FP16 peaks are the same scale.

