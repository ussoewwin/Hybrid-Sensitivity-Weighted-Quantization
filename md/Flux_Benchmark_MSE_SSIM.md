# Flux Benchmark — MSE and SSIM

Only **MSE** was changed: it is now computed in **latent space** (before VAE decode). **SSIM** is unchanged: still computed by `calculate_metrics(img1, img2)` on decoded pixel images (grayscale SSIM). Below: before/after code and rationale.

---

## 1. Problem

- Originally both MSE and SSIM came from **decoded pixel images** via a single `calculate_metrics(img1, img2)`.
- That produced very large MSE (e.g. ~190) because the VAE decoder amplifies small latent differences in pixel space.

---

## 2. Change (MSE only)

- **MSE** → now computed in **latent space** (raw sampler output, before VAE decode) so it reflects UNet quantization error without VAE amplification.
- **SSIM** → **unchanged**. Still from `calculate_metrics(img_fp16, img_fp8)`; that function was not modified. Grading still uses that SSIM value.

---

## 3. Before (MSE/SSIM)

### 3.1 Generator return (before)

```python
    # ... sampler.sample(...) ...
    elapsed = end_time - start_time

    # Decode
    vae_decode = nodes.VAEDecode()
    image_tensor = vae_decode.decode(vae=vae_obj, samples=samples)[0]
    img = Image.fromarray(...)
    return img, elapsed
```

- Latent was not returned; latent MSE could not be computed.

### 3.2 Comparison in main (before)

```python
    img_fp16, t16 = generate_image_comfy(...)
    img_fp8, t8 = generate_image_comfy(...)
    # ...
    mse, score = calculate_metrics(img_fp16, img_fp8)
    print(f"MSE (Error): {mse:.4f} ...")
    print(f"SSIM (Sim) : {score:.4f} ...")
```

- Both MSE and SSIM from `calculate_metrics` on pixel images.

---

## 4. After (MSE only changed)

### 4.1 New: latent MSE

```python
def calculate_latent_mse(l1, l2):
    """MSE on raw latent tensors (before VAE decode)."""
    arr1 = l1[0].cpu().float().numpy()
    arr2 = l2[0].cpu().float().numpy()
    return float(np.mean((arr1 - arr2) ** 2))
```

- Inputs: KSampler latent tensors. MSE is mean squared difference **before** VAE decode.

### 4.2 Generator return (after)

```python
    elapsed = end_time - start_time
    raw_latent = samples["samples"].clone()

    # Decode
    vae_decode = nodes.VAEDecode()
    image_tensor = vae_decode.decode(vae=vae_obj, samples=samples)[0]
    img = Image.fromarray(...)
    return img, raw_latent, elapsed
```

- Returns `raw_latent` so main can compute latent MSE.

### 4.3 Comparison in main (after)

```python
    img_fp16, lat_fp16, t16 = generate_image_comfy(...)
    img_fp8, lat_fp8, t8 = generate_image_comfy(...)
    # ...
    mse = calculate_latent_mse(lat_fp16, lat_fp8)
    _, score = calculate_metrics(img_fp16, img_fp8)
    print(f"MSE (Error): {mse:.6f} ...")
    print(f"SSIM (Sim) : {score:.4f} ...")
```

- **MSE** comes from `calculate_latent_mse(lat_fp16, lat_fp8)` (latent).
- **SSIM** still from `calculate_metrics(img_fp16, img_fp8)`; `calculate_metrics` is unchanged and SSIM is still pixel-space (grayscale).

---

## 5. Why MSE in latent space

- We want to measure **quantization error of the UNet (FP16 vs FP8)**.
- The VAE is shared and unchanged; the decoder is **nonlinear**, so small latent differences are amplified in pixel space.
- **Pixel-space MSE** therefore mixes UNet error and VAE amplification (e.g. ~190).
- **Latent-space MSE** reflects only the UNet output difference and is a proper quantization metric.

---

## 6. SSIM unchanged

- SSIM is still computed by the existing `calculate_metrics(img1, img2)` on decoded images (grayscale).
- No new SSIM function was added; the reported SSIM and the grade logic are unchanged.

---

## 7. Summary

| Metric | Before | After |
|--------|--------|--------|
| **MSE** | From `calculate_metrics` (pixel) | From `calculate_latent_mse(lat_fp16, lat_fp8)` (latent) |
| **SSIM** | From `calculate_metrics` (pixel) | From `calculate_metrics` (pixel, unchanged) |

- **Code changes:** generator returns `(img, raw_latent, elapsed)`; added `calculate_latent_mse`; main uses it for MSE and still uses `calculate_metrics` for SSIM only (`_, score = calculate_metrics(...)`).
