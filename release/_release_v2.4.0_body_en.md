## Overview

**v2.4.0** marks the official discontinuation of SDXL ConvRot NVFP4 quantization in favor of the high-fidelity **ConvRot INT8 (reverse hybrid)** pipeline, following deterministic latent-trajectory evaluations:

1. **Official Discontinuation of SDXL ConvRot NVFP4 (ConvRot INT8 Exclusively Supported)**:
   - Comprehensive multi-seed latent-trajectory evaluations revealed that 4-bit precision (NVFP4) cannot maintain latent trajectory fidelity on SDXL UNet.
   - Across sampling trajectories, the final latent trajectory cosine consistently fails to reach 0.90, suffering from severe trajectory drift, spatial distortion, and unrecoverable bifurcations.
   - The legacy decoded-image benchmark scores (SSIM ~0.95) were an artifact masked and smoothed by VAE decoding rather than genuine latent representation preservation.
   - Following the identical conclusion reached for Krea2 in v2.3.5, SDXL NVFP4 development has been officially discontinued. SDXL is maintained strictly via the production **ConvRot INT8 (reverse hybrid)** pipeline (`sdxl/diag_impact_sdxl.py` and `sdxl/gen_reverse_int8_sdxl_v1.1.py`), which reliably clears the production quality gate (mean trajectory cosine $\ge 0.95$ and 0/25 bifurcated).

2. **Repository Clean-Up and Deprecated NVFP4 Script Archival**:
   - Deprecated SDXL NVFP4 conversion scripts and benchmark tools have been relocated to `archives/`:
     - `hswq_sdxl_convert_nvfp4_1.0.py` $\rightarrow$ `archives/hswq_sdxl_convert_nvfp4_1.0.py`
     - `hswq_sdxl_convert_nvfp4_1.2.py` $\rightarrow$ `archives/hswq_sdxl_convert_nvfp4_1.2.py`
     - `benchmark/nvfp4bench_sdxl.py` $\rightarrow$ `archives/nvfp4bench_sdxl.py`
     - Trajectory and bake test scripts preserved in `archives/sdxl_nvfp4_traj_compare.py`, `archives/sdxl_w4a8_traj_compare.py`, and `archives/sdxl_w4a8_lora_bake_test.py`.
   - Obsolete NVFP4 links, modes, and benchmark reference rows have been completely excised from `README.md`.

3. **Harmonization of 8-Bit Flagship Architecture**:
   - Reorganized repository overview and documentation to unify **SDXL** and **Krea2** under the high-fidelity **ConvRot INT8** umbrella, establishing deterministic latent-trajectory convergence as the universal quality baseline across all supported architectures.

---

## Technical Details & In-Depth Trajectory Analysis

### 1. Why SDXL Fails Under 4-Bit (NVFP4) Quantization

While static weight-space metrics and superficial post-hoc image decodes previously suggested acceptable output, rigorous evaluation using step-by-step latent trajectory divergence exposed the fundamental structural failure of 4-bit precision on SDXL:

1. **Nonlinear Error Amplification in Iterative Samplers**:
   - In diffusion sampling, quantization errors are injected at every single denoising step. Under 4-bit precision (NVFP4), the coarse quantization grid ($E2M1$ with only 15 representable positive points per block) introduces significant quantization noise into cross-attention and convolutional projections.
   - These errors accumulate across iterative steps and undergo exponential (Lyapunov-style) divergence, driving the latent state away from the FP16 baseline manifold.

2. **Sub-0.90 Latent Trajectory Cosine and Catastrophic Bifurcation**:
   - Even with extensive FP16 protection under a 600 MiB budget and full Hadamard orthogonal rotation (ConvRot), the final latent trajectory cosine fails to reach 0.90.
   - In iterative diffusion systems, a latent trajectory cosine below 0.92 correlates with severe degradation of semantic details, while a cosine below 0.90 indicates irreversible trajectory collapse and bifurcation (generating structurally deformed or entirely divergent imagery).

---

### 2. The VAE Decoding Artifact: Why Legacy SSIM Scores Were Misleading

Previous evaluations of SDXL NVFP4 reported decoded-image SSIM scores in the range of 0.92–0.98. Deterministic trajectory analysis demonstrates why these numbers were deceptive:

- **VAE Decoder Regularization & Spatial Smoothing**:
  The SDXL VAE decoder acts as a highly non-linear low-pass filter with heavy spatial smoothing. High-frequency phase errors, semantic trajectory drifts, and micro-structural collapses occurring in the 4-channel latent space were partially smoothed out during the $8\times$ spatial upsampling and RGB projection.
- **De-masking via Latent Space Measurement**:
  By measuring the unadulterated latent state ($z_t$ and $x_0$ prediction) directly at each step using deterministic initial noise, the true numerical collapse of NVFP4 was uncovered: the model output is unstable and structurally degraded before VAE decoding artificially masks the damage.
- **True Production Gate**:
  The latent trajectory gate (25 seeds $\times$ 25 steps, mean cosine $\ge 0.95$, 0/25 bifurcated) serves as the sole objective benchmark for generative fidelity. NVFP4 cannot clear this gate.

---

### 3. Consolidation on SDXL ConvRot INT8 (Reverse Hybrid)

SDXL quantization is now exclusively concentrated on the proven **ConvRot INT8 (reverse hybrid)** architecture:

- **Empirical Trajectory-Impact Diagnosis (`sdxl/diag_impact_sdxl.py`)**:
  Ranks each eligible Linear and Conv2d layer by directly injecting ConvRot INT8 error on the production sampler and measuring latent drift against the FP16 baseline.
- **Reverse Conversion (`sdxl/gen_reverse_int8_sdxl_v1.1.py`)**:
  Converts the $K$ lowest-impact layers to FULL ConvRot INT8 while preserving the remaining sensitive layers at pristine FP16.
- **Embedded VAE Removal**:
  Drops `first_stage_model.*` (~160 MiB) from the converted pack to eliminate redundant payload, resulting in a streamlined ~4.9 GB file size (~29–40% reduction vs FP16).
- **Verified Fidelity**:
  Consistently achieves deterministic 25-seed latent trajectory cosine $\ge 0.95$ with zero bifurcations.

---

## Active Quantization Architecture Matrix

| Model Architecture | Recommended Mode | Precision & Format | Primary Tool / Script | Production Quality Gate |
| :--- | :--- | :--- | :--- | :--- |
| **SDXL UNet** | **ConvRot INT8 (Reverse Hybrid)** | `int8_tensorwise` + FP16 sensitive | `sdxl/diag_impact_sdxl.py`<br>`sdxl/gen_reverse_int8_sdxl_v1.1.py` | **25-seed trajectory cos $\ge 0.95$, 0/25 bifurcated** |
| **Krea2 DiT** | **ConvRot INT8** | `int8_tensorwise` (1off, blacklist) | `Krea2/hswq_convrot_int8_krea2_v1.5.py` | **25-seed trajectory cos $\ge 0.98$, 0/25 bifurcated** |
| **Z Image Turbo** | **Reverse Hybrid NVFP4** | NVFP4 Linear + INT8 Conv2d | `Z_Image/diag_impact.py`<br>`Z_Image/gen_reverse_nvfp4.py` | **20-seed trajectory cos 0.96–0.97, 0/20 bifurcated** |
| **Qwen Image Edit** | **Native ConvRot INT8** | `int8_tensorwise` | `Qwen Image/native_convert_int8_convrot_qwen.py` | **10-seed trajectory cos $\ge 0.99$** |
| **TE / ControlNet / SAM 3** | **Native ConvRot INT8** | `int8_tensorwise` | `comfyui_nodes/te_controlnet_convrot_int8_convert.py` | **Zero guidance degradation (~50% VRAM reduction)** |

*(Note: Both Krea2 NVFP4 and SDXL NVFP4 are officially discontinued and excluded from active quantization).*

---

## Verification & References

- **Changelog**: [CHANGELOG.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/CHANGELOG.md)
- **SDXL Quantization Guide**: [`md/How to quantize SDXL.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SDXL.md)
- **SDXL Reverse Hybrid Guide**: [`md/Diag_Reverse_SDXL_v1.1_Technical_Guide.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Diag_Reverse_SDXL_v1.1_Technical_Guide.md)
