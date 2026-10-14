## Overview

**v2.4.1** introduces dedicated, high-fidelity **Native ConvRot INT8** quantization support for **Stable Diffusion 1.5 (SD1.5)** across both standalone CLI workflows and direct in-graph execution inside ComfyUI:

1. **Native ConvRot INT8 for SD1.5 (Single-Mode Policy)**:
   - SD1.5 checkpoints (~1.9 GB in FP16) are compressed down to roughly **850–950 MB** via full ConvRot INT8 packing on all eligible Linear and Conv2d weights.
   - Unlike larger diffusion backbones (SDXL, Krea2), HSWQ hybrid FP16 retention is deliberately omitted: because SD1.5 is naturally compact, retaining even a few sensitive layers in FP16 re-inflates the model payload by 200–350 MB without providing measurable gains in generation fidelity.
   - Native ConvRot INT8 reliably achieves a **mean latent trajectory cosine similarity $\ge 0.96$ (~0.97)** with zero trajectory bifurcations, establishing it as the definitive, optimal quantization standard for SD1.5.

2. **Automatic Embedded VAE Removal**:
   - Standard SD1.5 `.safetensors` checkpoints bundle embedded VAE weights (`first_stage_model.*`), adding 160–330 MB of redundant payload.
   - Both the CLI quantizer and ComfyUI node automatically strip the embedded VAE during conversion, ensuring the smallest possible distribution size. At inference time, images are decoded using standard external SD1.5 VAEs.

3. **ComfyUI In-Graph Node Support & Integrated Trajectory Benchmark**:
   - Extended the custom node **`Native ConvRot INT8 Quantize`** (`comfyui_nodes/native_convrot_int8_convert.py`) with a dedicated **`SD1.5`** architecture option.
   - Features fully automated post-convert benchmarking powered by a **25-seed deterministic latent trajectory comparison** (512×512 resolution, `dpmpp_2m` / `karras`, 25 steps, CFG = 7.0), evaluating step-by-step intermediate latents against the FP16 baseline directly in latent space without VAE decoder smoothing bias.

4. **Standalone CLI Converter & Comprehensive Documentation**:
   - Published `sd1.5/native_convert_int8_sd1.5.py` for headless batch execution, supporting custom Hadamard group sizes, per-channel scaling, and optional Card 1 activation bias correction.
   - Added complete technical guide [`md/How to quantize SD1.5.md`](md/How%20to%20quantize%20SD1.5.md) and reference workflow screenshot [`png/sd1.5.png`](png/sd1.5.png).

---

## Technical Details & In-Depth Trajectory Analysis

### 1. Architectural Rationale: Why Native ConvRot INT8 Only for SD1.5

In prior releases (e.g. SDXL in v2.3.6–v2.4.0), HSWQ employed reverse hybrid quantization to keep the highest-impact layers in pristine FP16. For SD1.5, empirical evaluations proved that this approach is counter-productive:

1. **Model Payload vs. Compression Utility**:
   - SD1.5's UNet consists of approximately 860M parameters (~1.72 GB in FP16).
   - In a full INT8 pack (with embedded VAE stripped), the total file size drops to **~850–950 MB** (a ~50% net compression ratio).
   - If an HSWQ selector protects even 10–15 critical ResBlock convolutional layers in FP16, the payload increases by ~300+ MB, degrading the compression ratio from 2.0× to barely 1.35×.
2. **Diminishing Returns on Latent Alignment**:
   - In extensive multi-seed sampling trials on diverse SD1.5 checkpoints (e.g., `chilledReGenericV3_v10`, standard SD1.5 pruned), Native ConvRot INT8 without FP16 protection consistently delivers:
     - **Final latent trajectory cosine**: $\mathbf{0.965 - 0.978}$
     - **Trajectory bifurcation rate**: **0 / 25 seeds** (max step drop $< 0.035 \ll 0.050$)
   - Adding hybrid FP16 protection to the top error-norm layers only shifted the cosine from ~0.971 to ~0.976—a difference undetectable in generative fidelity. Hence, pure Native ConvRot INT8 is the only honest and practical path.

---

### 2. Automatic Embedded VAE Removal

Conventional SD1.5 release files combine the UNet, CLIP Text Encoder, and VAE into a single monolith. In quantized distribution pipelines, embedding the VAE inside the UNet pack creates severe drawbacks:

- **Redundant Storage**: The 248 VAE tensors consume ~160–330 MB of disk space and VRAM.
- **Decoupled Workflow Standards**: Advanced ComfyUI pipelines overwhelmingly employ specialized external VAEs (such as `vae-ft-mse-840000-ema-pruned` or specialized anime VAEs).
- **Quantization Logic**: `sd1.5/native_convert_int8_sd1.5.py` and `comfyui_nodes/native_convrot_int8_convert.py` filter out all `first_stage_model.*` keys prior to serializing the final `.safetensors`. The resulting file contains only the optimized diffusion model and text encoder weights, fully compatible with ComfyUI's standard `Load Checkpoint` / `Load Diffusion Model`.

---

### 3. Integrated Deterministic 25-Seed Latent Trajectory Benchmark

Rather than relying on decoded pixel SSIM (which is smoothed and artificially inflated by the VAE decoder), the built-in benchmark in `Native ConvRot INT8 Quantize` executes deterministic latent-space trajectory tracking:

- **Sampling Parameters**: 512×512, 4 latent channels, `dpmpp_2m` sampler with `karras` scheduler, 25 steps, CFG = 7.0.
- **Evaluation Loop**:
  1. Runs 25 deterministic seeds on the pristine FP16 baseline model, recording step-by-step intermediate latents ($x_t$) and final $x_0$.
  2. Unloads FP16 from VRAM and loads the newly serialized ConvRot INT8 checkpoint.
  3. Re-runs the identical 25 seeds, computing per-step cosine similarity and detecting any abrupt drop $> 0.05$ (bifurcation flag).
  4. Formats a complete multi-seed statistical summary directly in ComfyUI's text output.

---

## Active Quantization Architecture Matrix

| Model Architecture | Recommended Mode | Precision & Format | Primary Tool / Script | Production Quality Gate |
| :--- | :--- | :--- | :--- | :--- |
| **SD1.5 UNet** | **Native ConvRot INT8** | `int8_tensorwise` (embedded VAE stripped) | `sd1.5/native_convert_int8_sd1.5.py`<br>`Native ConvRot INT8 Quantize` (`SD1.5`) | **25-seed trajectory cos $\ge 0.96$ (~0.97), 0/25 bifurcated** |
| **SDXL UNet** | **ConvRot INT8 (Reverse Hybrid)** | `int8_tensorwise` + FP16 sensitive | `sdxl/diag_impact_sdxl.py`<br>`sdxl/gen_reverse_int8_sdxl_v1.1.py` | **25-seed trajectory cos $\ge 0.95$, 0/25 bifurcated** |
| **Krea2 DiT** | **ConvRot INT8** | `int8_tensorwise` (1off, blacklist) | `Krea2/hswq_convrot_int8_krea2_v1.5.py` | **25-seed trajectory cos $\ge 0.98$, 0/25 bifurcated** |
| **Z Image Turbo** | **Reverse Hybrid NVFP4** | NVFP4 Linear + INT8 Conv2d | `Z_Image/diag_impact.py`<br>`Z_Image/gen_reverse_nvfp4.py` | **20-seed trajectory cos 0.96–0.97, 0/20 bifurcated** |
| **Qwen Image Edit** | **Native ConvRot INT8** | `int8_tensorwise` | `Qwen Image/native_convert_int8_convrot_qwen.py` | **10-seed trajectory cos $\ge 0.99$** |
| **TE / ControlNet / SAM 3** | **Native ConvRot INT8** | `int8_tensorwise` | `comfyui_nodes/te_controlnet_convrot_int8_convert.py` | **Zero guidance degradation (~50% VRAM reduction)** |

---

## Verification & Documentation Links

- **SD1.5 Quantization Guide**: [`md/How to quantize SD1.5.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20SD1.5.md)
- **SD1.5 ComfyUI Workflow Screenshot**: [`png/sd1.5.png`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/png/sd1.5.png)
- **CLI Converter Script**: [`sd1.5/native_convert_int8_sd1.5.py`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/sd1.5/native_convert_int8_sd1.5.py)
- **ComfyUI In-Graph Node**: [`comfyui_nodes/native_convrot_int8_convert.py`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/comfyui_nodes/native_convrot_int8_convert.py)
- **Repository Changelog**: [CHANGELOG.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/CHANGELOG.md)
