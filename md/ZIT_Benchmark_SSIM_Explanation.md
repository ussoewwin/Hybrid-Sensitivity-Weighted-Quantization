## ZIT FP8 Benchmark: Why SSIM Was Fixed

### TL;DR
SSIM is supposed to measure *structural similarity of images* (i.e., visual patterns). The previous implementation computed SSIM directly from **latent tensors** by reshaping/transposing them into an “image-like” layout. That made SSIM **not correspond to what the human eye sees**, so the score could be misleading.

This fix changes SSIM to be computed on:
- **Normalized 0–255 views** produced by `latent_to_img()` (default), or
- **VAE-decoded pixel images** when `--vae` is provided.

This aligns the SSIM input with the assumptions of the metric (`data_range=255` and meaningful image channel layout).

---

## 1) What SSIM Measures (and Why That Matters)
**SSIM (Structural Similarity Index)** compares two images by evaluating local luminance/contrast/structure relationships.

For SSIM to reflect “visual difference,” the inputs must behave like real images:
- pixel intensities in a consistent range,
- a channel dimension that corresponds to the image channels (e.g., RGB),
- a spatial layout where neighboring pixels represent neighbors in the rendered image.

If these assumptions are violated, SSIM can still produce a number—but that number no longer means “visual closeness.”

---

## 2) What Was Wrong in the Original Code
The previous implementation (old `calculate_metrics`) computed SSIM like this (conceptually):

1. Take latent tensors `l1` and `l2`
2. Convert to numpy arrays
3. Compute a per-sample `data_range` from latent min/max
4. Do a reshape-like transform:
   - `arr_hwc = arr.transpose(1, 2, 0)`
5. Slice the first 3 channels:
   - `arr_hwc[:, :, :3]`
6. Run SSIM with:
   - `win_size=3`, `channel_axis=2`, `data_range=data_range`

### Why this breaks “SSIM = visual difference”
Latent tensors are **not RGB images**. They are intermediate representations learned by the model.

The original code effectively assumed:
- the latent’s spatial axes correspond to pixel adjacency,
- the latent’s first 3 channels correspond to RGB channels,
- normalizing with latent min/max creates a meaningful intensity scale for SSIM.

None of these assumptions are guaranteed.

As a result:
- The “image” used by SSIM could have a channel meaning that does not map to visual structure.
- The intensity scaling (`data_range`) changes depending on latent min/max, which makes SSIM interpretation inconsistent.
- The computed SSIM may drift away from what you actually observe as visual differences.

That is why, before the fix, SSIM “did not come out accurately” as a proxy for appearance.

---

## 3) How the Fix Works (High-Level)
The fix introduces a conceptually correct pipeline:

1. Convert model outputs to an **actual image-like representation**:
   - Default: `latent_to_img()` produces a normalized 0–255 view.
   - Optional: `--vae` decodes latents into **pixel images**.
2. Compute SSIM between these image-like outputs using a fixed expectation:
   - SSIM uses `data_range=255`.

This ensures SSIM is computed on inputs that better satisfy the metric’s assumptions.

---

## 4) Detailed Code Walkthrough (New Implementation)

### 4.1 `latent_to_img()`: Latents -> Normalized 0–255 View
The helper converts the first latent item into an 8-bit image:

```python
def latent_to_img(l):
    l = l[0].permute(1, 2, 0).cpu().float().numpy()
    l = (l - l.min()) / (l.max() - l.min() + 1e-6) * 255
    return Image.fromarray(l[:, :, :3].astype(np.uint8))
```

Key points:
- It normalizes intensities to **0–255**.
- SSIM later uses `data_range=255`, matching this normalization.
- Output is `uint8`, which is consistent with a “real image” style workflow.

### 4.2 `calculate_ssim_normalized()`: SSIM on 0–255 Images
The SSIM function now takes two image objects (0–255) and runs SSIM with a fixed range:

```python
def calculate_ssim_normalized(img1, img2):
    """SSIM on 0-255 images (e.g. from latent_to_img)."""
    a1 = np.array(img1)
    a2 = np.array(img2)
    return float(ssim(a1, a2, win_size=3, channel_axis=2, data_range=255))
```

Why this is better:
- The inputs are image-like with consistent intensity scaling.
- SSIM’s `data_range` now reflects the actual intensity range of the images (0–255).
- You no longer feed raw latent values directly into SSIM as if they were pixels.

### 4.3 Optional: `--vae` for Pixel-Space SSIM
If you pass `--vae`, the benchmark decodes latents to pixel images via ComfyUI’s VAE loader/decoder and computes SSIM on those decoded images instead of the normalized latent view.

In practice, this typically makes SSIM more aligned with human-perceived differences.

---

## 5) Summary of What Changed

### Before (problematic)
- SSIM computed on reshaped latent tensors
- latent-derived min/max used as `data_range`
- channel slicing assumed latent channels correspond to RGB
- SSIM therefore did not reliably represent visual differences

### After (fixed)
- SSIM computed on **normalized 0–255 views** produced by `latent_to_img()`, or
- SSIM computed on **VAE-decoded pixel images** with `--vae`
- SSIM uses `data_range=255`, matching image intensity scaling
- SSIM becomes a meaningful “appearance similarity” metric

---

## 6) Interpreting the Results

When the benchmark prints:
- `MSE (latent)`:
  - smaller typically means the latent outputs are closer (numerically)
- `SSIM (0-255 view)` or `SSIM (decoded)`:
  - closer to `1.0` means the output images are structurally closer (visually)

If you compare FP16 vs FP8:
- a better SSIM implies the quantized model preserved visual structure more faithfully,
- a better (lower) MSE implies latent-level fidelity improved.

---

## 7) Recommended Validation Steps (Optional but Useful)
To verify that SSIM behavior matches perception:
1. Save and visually inspect `bench_fp16.png` and `bench_fp8.png`.
2. Compare `bench_diff.png` (difference image).
3. Run twice with the same `--seed` and confirm SSIM stability.
4. If possible, try with `--vae` to ensure SSIM corresponds to pixel-space appearance.

---

## 8) Notes / Limitations
Even the default “latent_to_img view” is still a visualization/projection, not a true pixel decode.
For most perceptual fidelity use-cases, supplying `--vae` is the closest alignment to real appearance.

