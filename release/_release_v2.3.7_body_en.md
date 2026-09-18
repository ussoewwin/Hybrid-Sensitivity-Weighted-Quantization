## Overview

**v2.3.7 removes the embedded VAE from the SDXL ConvRot INT8 pipeline.**

**Purpose: reduce the size of the quantized files.** The embedded VAE is never actually used.

---

## Why the embedded VAE is removed

- SDXL `.safetensors` checkpoints ship with an embedded VAE (`first_stage_model.*`), and it is carried into the quantized file.
- The ConvRot INT8 pipeline is latent-space only: the conversion and the calibration never decode an image, so the embedded VAE is never used.
- The embedded VAE is therefore payload only. Removing it reduces the size of the quantized file, and the quantized UNet body is untouched.

---

## What v2.3.7 changes

- The reverse converters no longer load the VAE from the checkpoint (they are loaded with `output_vae=False`, and no VAE is kept).
- `sdxl/gen_reverse_int8_sdxl_v1.1.py` drops `first_stage_model.*` right before `save_file`, so the quantized file no longer carries the embedded VAE.
- The V3.1 selector keeps the pipeline VAE off the GPU.
- The trajectory diag already ran without a VAE.

---

## Effect

| Item | Before | After |
| :--- | :--- | :--- |
| Embedded VAE in the quantized file | present - `first_stage_model.*`, 248 tensors, ~160 MiB (fp16) | **removed** |
| Quantized UNet body (`model.diffusion_model.*`) | ConvRot INT8 + FP16 protected layers | unchanged |

Fidelity is unaffected: the embedded VAE takes no part in any measurement of this pipeline, which is done in latent space.
