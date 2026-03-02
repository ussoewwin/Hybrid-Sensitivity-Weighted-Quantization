# Seed and cuDNN Fix for Reproducible Calibration

This document explains **why** HSWQ calibration needed full determinism, **what** the fix achieves, and **how** it is implemented in code (SDXL V1.3 and ZIT V1.6).

---

## 1. Background: Why Calibration Was Non-Reproducible

HSWQ calibration does two things:

1. **Sensitivity** — For each layer, it measures the **variance of the layer’s output** over many calibration samples (many prompts × many denoising steps). Layers with high variance are more “sensitive” and are kept in FP16.
2. **Importance** — Per-channel mean absolute value of **inputs** to each layer, used as weights in the weighted histogram MSE that chooses the optimal clipping (amax) for FP8.

Both depend on the **exact activations** flowing through the model. Those activations depend on:

- **Initial latent noise** — `torch.randn(...)` at the start of each calibration sample.
- **Any other randomness** — e.g. dropout (if enabled), hash order, or third-party libs using `random` / `numpy` / `torch` RNG.
- **GPU computation order** — cuDNN (used by Conv2d and related ops) can pick different algorithms or reduction order for speed, which can change the last few bits of floating-point results.

If any of these change between runs or between machines:

- The **sequence of activations** changes.
- **DualMonitor** (sensitivity and importance) sees different numbers.
- **Layer rankings** (which layers get FP16) and **optimal amax** per layer can change.
- So **Amax and SSIM** can differ run-to-run or machine-to-machine, even with the same prompts and script.

So “reproducibility” here means: **same script, same inputs, same hardware or different hardware → same Amax and same calibration-derived choices (FP16 set, amax dict), and therefore same quantized model and scores.**

---

## 2. What the Fix Means

Fixing **seeds** and **cuDNN behavior** means:

| Goal | How it’s achieved |
|------|-------------------|
| Same initial noise every time | Fix RNG state (Python `random`, `numpy`, `torch`, CUDA) and, where we control it, reset or pass a fixed generator right before generating latents. |
| Same floating-point path on GPU | Force cuDNN to use deterministic algorithms and disable benchmark-driven algorithm selection. |
| Same calibration outputs | With the above, the same prompts and script produce the same activations → same sensitivity/importance → same Amax and FP16 set. |

So after the fix:

- **Same machine, multiple runs** → identical Amax, identical FP16 layer set, identical quantized weights.
- **Different machines** (e.g. different 4090s) → still identical, as long as they run the same code and seeds (no non-deterministic libs or drivers that we don’t control).

The fix is “full” in the sense that we control all **our** sources of non-determinism (RNG and cuDNN). We do not change external libraries (e.g. diffusers internals) beyond what we call; for diffusers we only guarantee the **inputs** we give it (e.g. generator) are fixed.

---

## 3. Code: Global Seed and cuDNN (Script Start)

Used in both **SDXL V1.3** and **ZIT V1.6** (and any other script that needs reproducible calibration). Call this once at the top of the script, after importing `os`, `numpy`, and `torch`, and before any calibration or model code that uses RNG or GPU.

```python
import random
import os
import numpy as np
import torch

def seed_everything(seed=42):
    """Fix all RNG seeds and cuDNN for 100% reproducible calibration (same Amax/scores across runs and machines)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
```

**What each line does:**

| Line | Purpose |
|------|--------|
| `random.seed(seed)` | Python’s `random` module (e.g. any lib that uses it) starts from a fixed state. |
| `os.environ["PYTHONHASHSEED"]` | Makes `hash()` and dict iteration order deterministic (e.g. for string keys). Should be set before process start for full effect; setting in script still helps for any later hashing. |
| `np.random.seed(seed)` | NumPy’s RNG (e.g. any code using `np.random.*`) is fixed. |
| `torch.manual_seed(seed)` | PyTorch CPU RNG (e.g. `torch.randn` on CPU) is fixed. |
| `torch.cuda.manual_seed(seed)` | PyTorch CUDA RNG for the current device is fixed. |
| `torch.cuda.manual_seed_all(seed)` | PyTorch CUDA RNG for **all** devices is fixed (multi-GPU or future use). |
| `torch.backends.cudnn.deterministic = True` | cuDNN is told to choose only deterministic algorithms (no “faster but slightly different” options). |
| `torch.backends.cudnn.benchmark = False` | Disables cuDNN’s auto-tuning of algorithms per input size. That tuning can vary by run/environment and change which algorithm is used, so disabling it keeps the same algorithm for the same op. |

Without this block, different runs or different machines can get different random numbers and different cuDNN paths, and thus different calibration results.

---

## 4. Code: SDXL — Fixed Generator for the Pipeline

In SDXL we use **diffusers**’ `StableDiffusionXLPipeline`. The initial latent is created **inside** `pipeline(...)`. We cannot call `torch.manual_seed` right before that internal `randn`; instead we pass a **fixed generator** so that diffusers uses the same RNG state for latent creation.

Create the generator once before the calibration loop and pass it into every `pipeline(...)` call:

```python
# Before the calibration loop (device is already set, e.g. device = "cuda")
generator = torch.Generator(device=device).manual_seed(42)

for i, prompt in enumerate(prompts):
    with torch.no_grad():
        pipeline(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            output_type="latent",
            generator=generator,
        )
```

**Why this works:**

- `torch.Generator(device=device).manual_seed(42)` creates a generator on the same device as the model and sets its state to a fixed seed.
- Passing `generator=generator` to `pipeline(...)` makes diffusers use this generator for the initial latent (and any other randomness it uses from that generator). The same generator is reused for every sample, so the **sequence** of random numbers is deterministic: sample 1 gets the first draw, sample 2 the next, and so on.
- So the same prompts in the same order always produce the same latents and the same forward pass, hence the same sensitivity/importance and Amax.

**Important:** Do **not** call `manual_seed(42)` again inside the loop for every sample; that would make every sample use the **same** latent and hurt calibration diversity. One generator, reused, gives a deterministic but varied sequence.

---

## 5. Code: ZIT — Seed Right Before Initial Latent

In ZIT we use a **custom** calibration pipeline: we create the initial latent ourselves with `torch.randn(...)`. To guarantee the same latent every time for that call (independent of what happened earlier in the process), we set the seed **immediately before** that line:

```python
# Inside the calibration step (e.g. in ZITCalibrationPipeline.__call__), right before creating the latent
# Fix seed immediately before initial noise so every run yields identical latents (full reproducibility)
torch.manual_seed(42)
x = torch.randn(batch_size, latent_c, latent_h, latent_w,
                device=self.device, dtype=torch.float16)
```

**Why right before `randn`:**

- Between script start and this line, other code may have consumed RNG state (e.g. text encoding, model init, or other ops). So the state of the global RNG at this point can vary by run or environment.
- By calling `torch.manual_seed(42)` immediately before `torch.randn(...)`, we force the **next** random draw to be the same in every run. So the initial latent `x` is identical across runs and machines, and the rest of the calibration (sensitivity, importance, amax) follows from that.

Together with `seed_everything(42)` at script start, this gives full control over the only place we explicitly draw random numbers in the ZIT calibration path.

---

## 6. Summary Table

| Item | SDXL V1.3 | ZIT V1.6 |
|------|-----------|----------|
| Global seed + cuDNN at start | `seed_everything(42)` | `seed_everything(42)` |
| Initial latent | Created inside diffusers | Created in our code |
| How latent is fixed | `generator=torch.Generator(device).manual_seed(42)` passed to `pipeline(..., generator=generator)` | `torch.manual_seed(42)` immediately before `torch.randn(...)` |
| Script location | `quantize_sdxl_hswq_v1.3.py` | `quantize_zit_hswq_v1.6.py` |

With these in place, calibration is **deterministic**: same prompts and same script produce the same Amax and the same FP16 layer set on every run and across supported machines.
