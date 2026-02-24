# ZIT HSWQ V1.5: Latent Option and Mixed Precision Calibration — Full Guide

This document fully explains the **`--latent` option** and **Mixed Precision calibration** added in `quantize_zit_hswq_v1.5.py`, including the modified code and its meaning.

---

## 1. Latent Option (`--latent`)

### 1.1 Overview

The option sets the **spatial resolution (H × W) of the latent space** used during calibration.  
Default is **128**. You can use values in the 32–256 range: **smaller values are faster and use less memory; larger values yield higher-fidelity statistics**.

### 1.2 Added Code and Flow

#### (1) Command-line argument

```python
# quantize_zit_hswq_v1.5.py, around line 534
parser.add_argument("--latent", type=int, default=128, help="Calibration latent size (H and W). e.g. --latent 256")
```

- **Meaning**: One side of the latent map used during calibration (same value for both H and W).
- **Default 128**: Matches README recommendation. You can also use 32 (faster) or 256 (higher fidelity).

#### (2) Passing into the pipeline

```python
# Around line 614
pipeline = ZITCalibrationPipeline(model, text_encoder, tokenizer, device, latent_size=args.latent)
```

- **Meaning**: Pass `latent_size=args.latent` into the `ZITCalibrationPipeline` constructor so all calibration inference runs at this resolution.

#### (3) Storing and using it inside the pipeline

```python
# ZITCalibrationPipeline.__init__, around lines 319–324
def __init__(self, model, text_encoder, tokenizer, device="cuda", latent_size=128):
    self.model = model
    self.text_encoder = text_encoder
    self.tokenizer = tokenizer
    self.device = device
    self.latent_size = int(latent_size)
```

- **Meaning**: Store `latent_size` as an integer. This value defines the spatial size of the latent tensor per sample.

#### (4) Use in calibration inference (latent map shape)

```python
# ZITCalibrationPipeline.__call__, around lines 371–376
def __call__(self, prompt, num_inference_steps=20, **kwargs):
    """Run calibration inference (real prompts)."""
    batch_size = 1
    latent_h = latent_w = self.latent_size
    latent_c = 16  # Model in_channels
```

- **Meaning**: Set latent map height and width to `latent_h = latent_w = self.latent_size`. Channel count is 16 per ZIT spec.

#### (5) Initial latent noise

```python
# Around lines 418–420
# Initialize latents
x = torch.randn(batch_size, latent_c, latent_h, latent_w,
               device=self.device, dtype=torch.float16)
```

- **Meaning**: The calibration initial noise `x` has shape `(1, 16, latent_size, latent_size)`. With `--latent 32` you get 32×32; with `--latent 256` you get 256×256 spatial resolution for the run.

#### (6) Fallback when no text encoder

```python
# Around lines 378–384
else:
    print("Warning: Text encoder not set. Using random tensor.")
    cap_len = self.latent_size
    cap_feats = torch.randn(batch_size, cap_len, 2560, ...)
    cap_mask = torch.ones(batch_size, cap_len, ...)
```

- **Meaning**: When the text encoder is missing, caption feature length `cap_len` is also set from `latent_size` for consistent behavior.

#### (7) Log output

```python
# Around line 635
print(f"Running calibration ({args.num_calib_samples} samples, {args.num_inference_steps} steps, latent={args.latent})...")
```

- **Meaning**: Logs the resolution used for calibration.

### 1.3 Summary (Latent)

| Item | Description |
|------|-------------|
| **Role** | Specifies the spatial resolution (H, W) of the latent space during calibration. |
| **Usage** | `--latent 32` / `--latent 128` (default) / `--latent 256`, etc. |
| **Small value (e.g. 32)** | Less compute and memory; faster calibration. |
| **Large value (e.g. 256)** | More spatial information; tends to give higher-fidelity statistics. |
| **Recommendation** | Follow README “Latent: 32–256, default 128”; 128 is usually sufficient. |
| **GPU** | For `--latent 256`, RTX 5090 or above recommended; for `--latent 32`, RTX 5060 Ti 16GB is sufficient. |

---

## 2. Mixed Precision Calibration

### 2.1 Overview

During calibration, the **model and main tensors are kept in FP16**, and the **CUDA autocast** context wraps the computation so that statistics (Sensitivity and Importance) are collected in a **mixed-precision** way: lower memory and faster, while remaining numerically stable.

### 2.2 Code and Meaning

#### (1) Model loaded and run in FP16

```python
# Inside load_zit_model, around line 308
model = model.to(device).to(torch.float16)
model.eval()
```

- **Meaning**: The ZIT model is moved to the device and run in **FP16**; calibration is FP16-based from the start.

#### (2) Autocast in the calibration loop

```python
# In main, around lines 640–644
for i, prompt in enumerate(prompts):
    print(f"\nSample {i+1}/{args.num_calib_samples}: {prompt[:50]}...")
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps)
```

- **Meaning**:
  - `torch.no_grad()`: Disables gradient computation to save memory and compute.
  - `torch.amp.autocast("cuda")`: Enables **mixed precision** on CUDA (FP16/BF16/FP32 chosen automatically for stability and speed). Each calibration sample runs inside this context.

#### (3) Explicit FP16 cast in ZITWrapper (avoiding Float/Half mismatch)

```python
# ZITCalibrationPipeline, ZITWrapper.__call__, around lines 396–414
def __call__(self, x, sigma, **kwargs):
    # Ensure inputs are correct dtype for the fp16 model
    dtype = torch.float16

    # ZIT expects t in [0, 1]. Input sigma from sample_euler is appropriate.
    # Cast x and sigma/t to fp16 to avoid "Float and Half" mismatch
    x_in = x.to(dtype=dtype)
    t_in = sigma.to(dtype=dtype)

    try:
        # Ensure features are FP16
        cap_feats_in = self.cap_feats.to(dtype=dtype)
        out = self.model(x_in, t_in, cap_feats_in, None, attention_mask=self.cap_mask)
        if isinstance(out, tuple):
            out = out[0]
        # Output needs to be cast back to x.dtype (likely float32) for k_diffusion
        return out.to(dtype=x.dtype)
```

- **Meaning**:
  - **x**, **sigma**, and **cap_feats** are cast to **FP16** before `self.model(...)` to avoid “Float and Half” runtime errors.
  - Because the model is FP16, all its inputs must be FP16.
  - The output is cast back to `x.dtype` (often float32) for the k_diffusion sampler.

#### (4) Initial latent noise dtype

```python
# Around lines 419–420
x = torch.randn(batch_size, latent_c, latent_h, latent_w,
               device=self.device, dtype=torch.float16)
```

- **Meaning**: The calibration initial noise is created in **FP16** so the whole pipeline stays FP16-based.

### 2.3 Mixed Precision Overview

| Layer | Role |
|--------|------|
| **Model** | Fixed in FP16 via `model.to(torch.float16)`. |
| **Calibration loop** | `torch.amp.autocast("cuda")` enables mixed precision. |
| **ZITWrapper** | Explicitly casts inputs x, sigma, cap_feats to FP16 so dtypes match the model. |
| **Initial latent** | `torch.randn(..., dtype=torch.float16)` for FP16. |
| **Output** | Cast back with `out.to(dtype=x.dtype)` for k_diffusion. |

Together, this implements **mixed precision calibration**: FP16 as the base, with autocast applying higher precision where needed.

---

## 3. Summary

- **`--latent`**: Spatial resolution (H=W) of the latent used in calibration. Smaller → faster and less memory; larger → higher fidelity. Default 128; 32–256 recommended.
- **Mixed precision calibration**: Model and main tensors are FP16; `torch.amp.autocast("cuda")` and explicit casts in ZITWrapper avoid dtype mismatches while running calibration in mixed precision.

With the code and meanings above, you have a complete picture of the V1.5 latent option and mixed precision calibration behavior.
