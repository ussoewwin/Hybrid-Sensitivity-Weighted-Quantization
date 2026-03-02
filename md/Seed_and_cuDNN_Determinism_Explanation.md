# Seed and cuDNN Fix for Reproducible Calibration (Technical)

This document gives a **technical** explanation of why HSWQ calibration required full determinism, which **sources of non-determinism** exist, and **exactly how** the fix is implemented in SDXL V1.3, with code and script locations.

---

## 1. Background: What Calibration Does and Why Exact Numbers Matter

### 1.1 Calibration pipeline (SDXL V1.3)

Calibration in `quantize_sdxl_hswq_v1.3.py` runs in this order:

1. Load the UNet and build a diffusers `StableDiffusionXLPipeline` on the target device.
2. Register **forward hooks** on every `Conv2d` and `Linear` in the UNet. Each hook calls `DualMonitor.update(input_tensor, output_tensor)` for that layer.
3. Run **N calibration samples** (e.g. 32). For each sample:
   - Get one prompt from the calibration file.
   - Call `pipeline(prompt=..., num_inference_steps=..., output_type="latent", generator=...)`. This runs **M denoising steps** (e.g. 25); at each step the full UNet forward pass runs, so every hooked layer sees one (input, output) pair per step.
   - Over the run, each layer’s hook is invoked **N × M times** (e.g. 32 × 25 = 800 times).
4. After the loop, **DualMonitor** for each layer holds:
   - `output_sum`, `output_sq_sum`, `count`: running sum of output values and their squares, and call count.
   - `channel_importance`: running per-channel mean of input absolute values.
5. **Sensitivity** per layer = variance of output = `(output_sq_sum/count) - (output_sum/count)^2`. Layers are **ranked by this variance**; the top K% are kept in FP16.
6. **Importance** (per-channel) is used later by the HSWQ optimizer to build a **weighted histogram** of weight values; the optimizer finds the **amax** (clipping threshold) that minimizes weighted MSE for each FP8-quantized layer.

So the **only** inputs to the final quantization are: (a) the **order** of layers by sensitivity, and (b) the **per-layer amax** from the weighted histogram. Both come directly from the **sequence of (input, output) tensors** that each hook saw. If that sequence changes in any way, the order and the amax values change, and the quantized model and its SSIM can change.

### 1.2 Why “exact” activations matter

- **Running statistics:** DualMonitor uses **incremental** mean/variance and importance. The final value for a layer is a function of the **entire sequence** of (input, output) pairs for that layer. Changing even one pair (e.g. one step on one sample) changes the running state and thus the final variance and importance.
- **Propagation:** The initial latent is the only random input we inject per sample. It is fed into the UNet; every layer’s output depends on that latent and all previous layers. So **one different latent** implies **different activations at every layer, at every step**, and therefore different inputs to DualMonitor at every call.
- **Amplification:** Floating-point arithmetic is not associative: `(a + b) + c` and `a + (b + c)` can differ at the last bit. So the **order** in which values are summed (e.g. in cuDNN reductions) can change the result. Over hundreds of layers × many steps × many samples, small per-op differences accumulate into visibly different running means and variances.
- **Consequence:** Without fixing RNG and cuDNN, the same script and same prompts can yield **different layer order** (different layers in FP16) and **different amax** per layer → different quantized weights → **different SSIM** and possible “run-to-run” or “machine-to-machine” variation.

So “reproducibility” here means: **same script, same calibration prompts, same or different machine → identical sensitivity order, identical amax dict, identical quantized model and metrics.**

---

## 2. Sources of Non-Determinism (Technical)

We need to fix every source that can change the sequence of (input, output) tensors or the way they are aggregated.

### 2.1 Python and interpreter

| Source | What it affects | Default behavior |
|--------|-----------------|------------------|
| **`random` module** | Any code using `random.random()`, `random.shuffle()`, etc. (ours or libraries). | Seeded by system time / process id if not set; differs per run. |
| **`hash()` / `PYTHONHASHSEED`** | Hash of objects (e.g. dict keys). Dict iteration order in Python 3.7+ is insertion order, but hashing can affect internal behavior; setting `PYTHONHASHSEED` removes hash randomization. | If unset, Python can randomize hashes (e.g. for strings) for security; iteration over dicts/sets can then vary. |
| **Dict/set iteration** | Any logic that iterates over `state_dict`, `named_modules()`, etc. If order ever depended on hash, it could vary. | With fixed PYTHONHASHSEED and same insertion order, iteration is stable. |

### 2.2 NumPy

| Source | What it affects | Default behavior |
|--------|-----------------|------------------|
| **`np.random`** | Any code using `np.random.rand()`, `np.random.shuffle()`, etc. | Separate RNG from Python and PyTorch; not seeded by `random.seed()` or `torch.manual_seed()`. So it must be seeded explicitly. |

### 2.3 PyTorch (CPU and CUDA)

| Source | What it affects | Default behavior |
|--------|-----------------|------------------|
| **CPU generator** | `torch.randn(..., device='cpu')`, `torch.rand()`, dropout on CPU, etc. | Default global generator; state advances with every call; initial state is undefined unless `torch.manual_seed()` is called. |
| **CUDA generator (per device)** | `torch.randn(..., device='cuda')`, dropout on GPU, and **any library that uses the default CUDA generator** (e.g. diffusers when creating the initial latent). | Each device has its own RNG state; not set by `torch.manual_seed(seed)` alone; must set with `torch.cuda.manual_seed(seed)` and optionally `torch.cuda.manual_seed_all(seed)` for all devices. |
| **Explicit `Generator`** | When we pass `generator=...` to `pipeline()`, diffusers uses **that** generator for the initial latent. Its state is independent of the global CUDA generator until we create it with `manual_seed(42)`. | If we don’t pass a generator, the library uses its default (often the global CUDA generator), whose state we may not control precisely across process/version. |

So we need: (1) global CPU and CUDA seeds at script start, and (2) for SDXL, a **single** `torch.Generator(device).manual_seed(42)` passed to every `pipeline(...)` call so the **sequence** of random numbers consumed by diffusers is fixed.

### 2.4 cuDNN (NVIDIA backend for Conv and related ops)

PyTorch uses **cuDNN** for many ops on GPU (e.g. `Conv2d`, `BatchNorm`, certain reductions). Two settings matter:

| Setting | Meaning | Default | Effect if not fixed |
|---------|---------|---------|---------------------|
| **`torch.backends.cudnn.deterministic`** | When `True`, cuDNN is only allowed to use **deterministic** algorithms for the current op. Some algorithms use atomic adds or different reduction order and can produce slightly different results (last bits) for the same mathematical operation. | `False` | Different runs or different GPUs can pick different algorithms → different floating-point results → different activations. |
| **`torch.backends.cudnn.benchmark`** | When `True`, at the **first** occurrence of each (op, shape) pair, cuDNN **benchmarks** several algorithms and caches the “fastest.” Later runs use the cached choice. | `True` | The “first” run can differ from the next; and the **cached choice** can depend on GPU model, driver, and even other processes. So algorithm selection can vary by run or machine. |

So we set `cudnn.deterministic = True` and `cudnn.benchmark = False` at script start so that (a) only deterministic algorithms are used, and (b) algorithm choice is not time-dependent or environment-dependent.

### 2.5 Summary table: source → fix → effect

| Source | Fix | Effect |
|--------|-----|--------|
| Python `random` | `random.seed(42)` | Same sequence from any code using `random.*`. |
| Hash / dict | `os.environ["PYTHONHASHSEED"] = "42"` | Deterministic hashing; stable behavior for dicts/sets. |
| NumPy | `np.random.seed(42)` | Same sequence from any `np.random.*` usage. |
| PyTorch CPU | `torch.manual_seed(42)` | Same CPU RNG state. |
| PyTorch CUDA (current device) | `torch.cuda.manual_seed(42)` | Same CUDA RNG state for default generator on current device. |
| PyTorch CUDA (all devices) | `torch.cuda.manual_seed_all(42)` | Same CUDA RNG on all devices (multi-GPU or future use). |
| cuDNN algorithm choice | `torch.backends.cudnn.deterministic = True` | Only deterministic algorithms used. |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = False` | No benchmark-based algorithm selection; same algorithm for same op. |
| Diffusers latent | `generator=torch.Generator(device).manual_seed(42)` passed to `pipeline(...)` | Same sequence of initial latents (and any other randomness diffusers takes from that generator). |

---

## 3. Code: Global Seed and cuDNN (Script Start)

**Location in script:** `quantize_sdxl_hswq_v1.3.py`, immediately after importing `numpy` and `torch`, and **before** any code that uses RNG or loads the model (so before `load_unet_from_safetensors` and before the calibration loop).

**Requirement:** `random`, `os`, `numpy`, and `torch` must be imported before calling `seed_everything`. The script already has `import random` and uses `os`, `np`, and `torch`.

**Full code as in the script:**

```python
import argparse
import random
import torch
# ... other imports ...
import os
import numpy as np

# HSWQ module (Fast)
from weighted_histogram_mse_fast import HSWQWeightedHistogramOptimizerFast as HSWQWeightedHistogramOptimizer


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

**Line-by-line:**

| Line | Purpose |
|------|--------|
| `random.seed(seed)` | Sets the state of Python’s `random` module (Mersenne Twister). Any later call to `random.random()`, `random.shuffle()`, etc. (in our code or in libraries) starts from this state. |
| `os.environ["PYTHONHASHSEED"] = str(seed)` | Disables hash randomization when set before or at startup. Ensures `hash()` is deterministic for the same object; helps avoid order-dependent behavior in dicts/sets. |
| `np.random.seed(seed)` | Sets NumPy’s global RNG state. Independent of Python and PyTorch; must be set explicitly. |
| `torch.manual_seed(seed)` | Sets the **default CPU** generator’s state. Affects `torch.randn()`, `torch.rand()`, etc. on CPU. |
| `torch.cuda.manual_seed(seed)` | Sets the **current CUDA device**’s default generator state. Affects `torch.randn(..., device='cuda')` and any CUDA dropout when using the default generator. |
| `torch.cuda.manual_seed_all(seed)` | Sets the default generator state for **all** CUDA devices. Needed if any code runs on a non-current device. |
| `torch.backends.cudnn.deterministic = True` | Tells cuDNN to use only algorithms that produce the same output for the same input (no non-deterministic reductions). May be slower than default. |
| `torch.backends.cudnn.benchmark = False` | Disables “benchmark mode”: cuDNN will not run multiple algorithms and cache the fastest. The same (deterministic) algorithm is used every time for the same op shape. |

**When to call:** Once at import time (or at the very start of `main()` before loading the model). Must run **before** any calibration step so that the first use of RNG and the first cuDNN op see the fixed state.

---

## 4. Code: SDXL — Fixed Generator for the Pipeline

**Location in script:** `quantize_sdxl_hswq_v1.3.py`, inside `main()`, **after** the pipeline is created and prompts are loaded, **before** the `for i, prompt in enumerate(prompts):` loop. The generator is created once and passed into every `pipeline(...)` call.

**Why a generator instead of only global seed:** The initial latent for each calibration sample is created **inside** `StableDiffusionXLPipeline.__call__`. We cannot insert `torch.manual_seed(42)` right before that internal `randn`. The library accepts an optional `generator` argument; when provided, it uses that generator for the initial latent (and possibly other internal randomness). So we create one generator, seed it once, and pass the **same** generator to every call. That way the **sequence** of random numbers (sample 1 → first draw, sample 2 → second draw, …) is fixed and reproducible.

**Full code as in the script:**

```python
    # ... pipeline and prompts are ready ...
    print(f"Running calibration ({args.num_calib_samples} samples, {args.num_inference_steps} steps)...")
    print("Measuring Sensitivity and Importance (input activation) simultaneously...")
    
    pipeline.set_progress_bar_config(disable=False)
    generator = torch.Generator(device=device).manual_seed(42)
    
    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{args.num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad():
            pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                output_type="latent",
                generator=generator,
            )
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
```

**Details:**

- **`torch.Generator(device=device)`** — Creates a new generator object on the same device as the pipeline (usually `"cuda"`). This generator has its own state, separate from the global default.
- **`.manual_seed(42)`** — Sets that generator’s state to a fixed seed and returns the generator. Every subsequent random draw from this generator (e.g. when diffusers calls it for the latent) is deterministic.
- **Reuse in the loop** — We pass the **same** `generator` to every `pipeline(...)` call. Diffusers consumes some number of random values from it for each sample. So sample 1 gets the first N draws, sample 2 the next M draws, etc. The sequence is fixed, so the sequence of latents is fixed.
- **Do not reseed inside the loop** — If we called `generator.manual_seed(42)` (or created a new generator with seed 42) inside the loop, every sample would get the **same** latent. That would hurt diversity of calibration and is wrong. One generator, created once and reused, gives a deterministic but **varied** sequence of latents.

**Device note:** If `device == "cpu"`, `torch.Generator(device="cpu")` is used; the pipeline then runs on CPU and the generator is the CPU generator. The logic is the same: one generator, one seed, reuse for all samples.

---

## 5. Script Location Summary (SDXL V1.3)

| Fix | File | Approximate location |
|-----|------|----------------------|
| `seed_everything` definition and `seed_everything(42)` | `quantize_sdxl_hswq_v1.3.py` | Lines 30–42: after HSWQ import, before C++20 / ComfyUI helpers. |
| `generator = torch.Generator(device=device).manual_seed(42)` | `quantize_sdxl_hswq_v1.3.py` | Inside `main()`, after `pipeline.set_progress_bar_config(disable=False)`, before the `for i, prompt in enumerate(prompts):` loop (line 342). |
| `generator=generator` in `pipeline(...)` | `quantize_sdxl_hswq_v1.3.py` | Inside the same loop, in the `pipeline(...)` call (lines 347–352). |

---

## 6. Before vs After (Expected Behavior)

| Scenario | Without fix | With fix |
|----------|-------------|----------|
| Same machine, same script, two runs | Amax and/or FP16 set can differ; SSIM can differ. | Identical Amax, identical FP16 set, identical quantized model and SSIM. |
| Different machine (e.g. another 4090), same script | Amax and/or FP16 set can differ due to cuDNN and RNG. | Identical Amax, identical FP16 set, identical quantized model and SSIM (assuming same PyTorch/cuDNN and no other non-deterministic drivers). |
| Same run, same calibration data | N/A | Same prompts in same order → same latent sequence → same sensitivity/importance → same amax and layer order. |

The fix does **not** guarantee determinism of code we do not control (e.g. diffusers internals beyond the generator we pass, or system libraries). Within our script and the way we call the pipeline, we fix all **our** sources of non-determinism (RNG and cuDNN) so that calibration is fully reproducible in practice.
