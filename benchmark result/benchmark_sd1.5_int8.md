# SD1.5 ConvRot INT8 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot INT8 quantized** output (SD1.5 UNet family).  
Higher final-cosine is better (1.0 = identical trajectory).

**Source:** `benchmark result/score_sd1.5_int8.txt`

---

## Results

| Model | Keep ratio | final-cos min | final-cos mean | final-cos max | same-image | bifurcated |
|-------|------------|--------------:|---------------:|--------------:|-----------:|-----------:|
| `0.5xxmix9realistic_v40 0.5BRA7` (HSWQ k80) | k80 | 0.80943 | **0.97542** | 0.99892 | **16/25** | 0/25 |
| `0.5xxmix9realistic_v40 0.5BRA7` (native INT8) | r0 | 0.77574 | 0.96265 | 0.99913 | 13/25 | 0/25 |

---

## HSWQ vs Native INT8 comparison

Same setup (vs FP16 reference). **HSWQ INT8** vs **Native INT8**.  
Higher final-cosine is better. Δ = HSWQ − Native (positive Δ → HSWQ better).  
**Native** = naive cast INT8 (no protection). Native and HSWQ are not the same.

| Model | Keep | HSWQ mean | Native mean | Δ mean | HSWQ min | Native min | Δ min | HSWQ same | Native same | Winner |
|-------|------|----------:|------------:|-------:|---------:|-----------:|------:|----------:|------------:|--------|
| `0.5xxmix9realistic_v40 0.5BRA7` | k80 | **0.97542** | 0.96265 | +0.01277 | **0.80943** | 0.77574 | +0.03369 | **16/25** | 13/25 | **HSWQ** |

**Winner** = better on both mean final-cosine and same-image count.

---

## Notes

- **final-cos:** cosine similarity between the FP16 and INT8 latent trajectories at the final step; 1.0 = perfect match.
- **max-drop:** the largest per-step cosine drop along the trajectory. A sudden large drop (`bifurcated`, threshold 0.05) means the picture became a *different image*, not a degraded one.
- **same-image:** `final-cos >= 0.98` **and** not bifurcated.
- **drifted:** gradual divergence - acceptable at this level.
- **bifurcated:** 0/25 in both blocks.

---

## Analysis & Key Findings (SD1.5 ConvRot INT8)

- **Inference Time (FP16 → INT8):** **1.70s → 1.31s (+23.2%)** for the HSWQ pack; **1.71s → 1.35s (+20.8%)** for native.
- **Mean gain over native:** **+0.01277**, and the worst seed improves from **0.77574 to 0.80943** (+0.03369). Both candidates keep **0/25 bifurcated**.
- **The gain is concentrated, not uniform.** Of 25 seeds, **18 improve, 6 get slightly worse, 1 ties within 0.001**. The largest single gain is seed **4710928** (native 0.77574 → HSWQ 0.95062, **+0.17488**) - an unstable trajectory rescued rather than a small nudge on every seed. The largest loss is seed **149206** (native 0.99913 → HSWQ 0.98499, **−0.01414**).
- **K is not a fixed number.** The protected-layer count is chosen per checkpoint by sweeping it and taking the point where the mean stops paying for the extra bytes. Too few protected layers scores the same as native (the protection buys nothing); too many scores lower *while the pack grows*, because a layer promoted to FP16 keeps exact weights but loses the INT8 bias correction that cancels its systematic output shift. The sweep and its reasoning live in the technical guide, not in this file.
- **Composition:** UNet only. CLIP is copied through unchanged; the VAE is dropped from the pack (decode with a separate SD1.5 VAE).
- **Honest limits:** every criterion here measures one layer at a time while tens are quantized together, so single-layer contributions are not additive. The ranking, K and these numbers are checkpoint-specific. 25 seeds is a sample - `min` is the worst of 25 draws, not a bound.
