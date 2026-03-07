# Flux Benchmark — MSE and SSIM

How the Flux FP8 benchmark computes **MSE** and **SSIM**, with before/after code and rationale.

---

## 1. Problem

### 1.1 Before

- Both **MSE** and **SSIM** were computed on **decoded pixel images**.
- That led to very large MSE (e.g. ~190) while SDXL/ZIT benchmarks did not.

### 1.2 Approach

- **MSE** — measure quantization error magnitude → compute in **latent space** (before VAE decode).
- **SSIM** — measure perceptual similarity of the final image → keep computing in **pixel space** (after VAE decode).

Below: before/after code and full rationale (MSE/SSIM only).

---

## 2. Before (MSE/SSIM only)

### 2.1 Metric (before)

```python
def calculate_metrics(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    mse = np.mean((arr1.astype(np.float64) - arr2.astype(np.float64)) ** 2)
    score_ssim = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)
    return mse, score_ssim
```

- Inputs: **PIL images** only. MSE and SSIM both on the same pixel images.

### 2.2 Generator return (before)

```python
def generate_image_comfy(unet_name, clip_obj, vae_obj, args, weight_dtype="default"):
    # ... sampling ...
    samples = sampler.sample(...)[0]

    # Decode
    vae_decode = nodes.VAEDecode()
    image_tensor = vae_decode.decode(vae=vae_obj, samples=samples)[0]

    img_array = 255. * image_tensor[0].cpu().numpy()
    img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    return img, elapsed   # image and elapsed time only
```

- **Latent (sampler output) was not returned**, so latent MSE could not be computed.

### 2.3 Comparison in main (before)

```python
    img_fp16, t16 = generate_image_comfy(...)
    img_fp8, t8 = generate_image_comfy(...)

    if img_fp16.size != img_fp8.size:
        print("Error: Image sizes do not match!")
        return

    mse, score = calculate_metrics(img_fp16, img_fp8)

    print(f"MSE (Error): {mse:.4f} \t(0 is perfect match)")
    print(f"SSIM (Sim) : {score:.4f} \t(1.0 is perfect match)")

    if score > 0.98:
        grade = "PERFECT (S)"
    # ...
```

- Comparison used **images only**; one `calculate_metrics` for both MSE and SSIM in pixel space.

---

## 3. After (MSE/SSIM only)

### 3.1 Metrics (after)

**MSE (latent space)**

```python
def calculate_latent_mse(l1, l2):
    """MSE on raw latent tensors (before VAE decode)."""
    arr1 = l1[0].cpu().float().numpy()
    arr2 = l2[0].cpu().float().numpy()
    return float(np.mean((arr1 - arr2) ** 2))
```

- Inputs: **KSampler latent tensors** (batch dim; `l1[0]` is one sample). MSE is the mean squared difference **before** VAE decode.

**SSIM (pixel space)**

```python
def calculate_pixel_ssim(img1, img2):
    """SSIM on decoded pixel images."""
    return float(ssim(np.array(img1), np.array(img2), win_size=3, channel_axis=2, data_range=255))
```

- Inputs: **PIL images** (decoded RGB). SSIM stays in pixel space with `data_range=255`.

### 3.2 Generator return (after)

```python
def generate_image_comfy(unet_name, clip_obj, vae_obj, args, weight_dtype="default"):
    """Generate image using ComfyUI nodes. Returns (PIL image, raw latent tensor, elapsed seconds)."""
    # ...
    sampler = nodes.KSampler()
    samples = sampler.sample(unet, seed, args.steps, ...)[0]

    torch.cuda.synchronize()
    end_time = time.time()
    elapsed = end_time - start_time

    raw_latent = samples["samples"].clone()   # capture before decode

    print("Decoding...")
    vae_decode = nodes.VAEDecode()
    image_tensor = vae_decode.decode(vae=vae_obj, samples=samples)[0]

    img_array = 255. * image_tensor[0].cpu().numpy()
    img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    return img, raw_latent, elapsed   # image, latent, elapsed
```

- Returning **`raw_latent`** from `samples["samples"].clone()` allows main to compute latent MSE.

### 3.3 Comparison in main (after)

```python
    img_fp16, lat_fp16, t16 = generate_image_comfy(...)
    img_fp8, lat_fp8, t8 = generate_image_comfy(...)

    # 3. Comparison
    print("\n=== 3. Calculating Metrics ===")

    latent_mse = calculate_latent_mse(lat_fp16, lat_fp8)
    pixel_ssim = calculate_pixel_ssim(img_fp16, img_fp8)

    print(f"--------------------------------------------------")
    print(f"MSE (Error): {latent_mse:.6f} \t(0 is perfect match)")
    print(f"SSIM (Sim) : {pixel_ssim:.4f} \t(1.0 is perfect match)")
    print(f"--------------------------------------------------")

    if pixel_ssim > 0.98:
        grade = "PERFECT (S)"
    elif pixel_ssim > 0.95:
        grade = "EXCELLENT (A)"
    # ...
```

- **MSE** from `calculate_latent_mse(lat_fp16, lat_fp8)` (latent space).
- **SSIM** from `calculate_pixel_ssim(img_fp16, img_fp8)` (pixel space). Grade uses SSIM only.

---

## 4. Why MSE in latent space

- We want to measure **quantization error of the UNet (FP16 vs FP8)**.
- The **VAE is shared** and unchanged; it does not add quantization error.
- The VAE decoder is **nonlinear**: small latent differences are amplified unevenly in pixel space.
- So **pixel-space MSE** mixes UNet error and VAE amplification (e.g. Flux ~190).
- **Latent-space MSE** reflects only the UNet output difference and is a proper quantization metric, comparable to other latent MSEs (e.g. ZIT).

Summary: **MSE = technical quantization error → measure where the error is (latent).**

---

## 5. Why SSIM stays in pixel space

- SSIM is defined for **structural/perceptual similarity** on **0–255 images**.
- Latents are **16ch, can be negative, different scale**; SSIM on raw latent is ill-conditioned and often low (e.g. ~0.85).
- We care about **how similar the decoded images look**, so SSIM should be on **final pixel output**.
- So **SSIM stays in pixel space** (after decode) and grading is based on it.

Summary: **SSIM = perceptual quality → measure what the user sees (pixel image).**

---

## 6. Summary

| Metric | Before | After | Reason |
|--------|--------|--------|--------|
| **MSE** | Pixel images | **Latent tensors** | Measure quantization error without VAE amplification. |
| **SSIM** | Pixel images | **Pixel images (unchanged)** | Perceptual quality on final image. |

- **Code changes**
  - `generate_image_comfy` returns **`(img, raw_latent, elapsed)`**.
  - **`calculate_latent_mse(lat_fp16, lat_fp8)`** for MSE.
  - **`calculate_pixel_ssim(img_fp16, img_fp8)`** for SSIM (pixel space unchanged).
  - Reported values and grade use **latent MSE** and **pixel SSIM**.
