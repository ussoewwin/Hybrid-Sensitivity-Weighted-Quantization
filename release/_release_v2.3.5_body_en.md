## Overview

**v2.3.5** formalizes the architectural decision on Krea2 quantization, introduces the deterministic SDXL INT8 trajectory comparison benchmark, decouples benchmark scripts to be completely self-contained, and stabilizes trajectory evaluation tools:

1. **Official Discontinuation of Krea2 Hybrid ConvRot NVFP4 (ConvRot INT8 Exclusively Supported)**:
   - Exhaustive empirical evaluation confirmed that 4-bit precision (NVFP4) cannot maintain generative fidelity on Krea2 SingleStreamDiT.
   - Even with HSWQ 4-axis composite sensitivity ranking and extensive layer retention, final latent trajectory cosine fails to reach 0.90 (resulting in severe trajectory drift, spatial distortion, and irreversible bifurcations).
   - Krea2 is officially restricted to **ConvRot INT8** (`Krea2/hswq_convrot_int8_krea2_v1.5.py` and ComfyUI `Native ConvRot INT8 Quantize`), which reliably achieves mean trajectory cosine $\ge 0.98$ with 0 bifurcations.
2. **SDXL Multi-Seed Trajectory Comparator (`benchmark/sdxl_int8_traj_compare.py`)**:
   - Added automated deterministic trajectory evaluation tool for SDXL UNet models comparing baseline FP16/BF16 against ConvRot INT8 checkpoints.
   - Evaluates step-by-step latent progression, final cosine similarity, SSIM, and trajectory divergence across multiple seeds and sampling steps.
3. **Benchmark Script Self-Containment & Zero External Dependencies**:
   - Refactored benchmark scripts (`benchmark/*.py`) to eliminate brittle external imports (such as `common_bench`), making each benchmark script completely self-contained and runnable out-of-the-box.
4. **Trajectory Benchmark Stability & Fixes**:
   - Restored missing `pathlib.Path`, `gc`, and torchaudio stub imports in `benchmark/krea2_traj_compare.py` to resolve startup crash.
   - Preserved `--tc` / `--parity` flags and added `--attention sage2` option for high-throughput attention benchmarking.
   - Removed obsolete legacy directory `benchmark/krea2_nvfp4` (superseded by `krea2_convrot_nvfp4`).
5. **Documentation Updates**:
   - Updated `README.md` and `CHANGELOG.md` with explicit architectural guidelines and deprecation notices. Benchmark raw experimental data (`benchmark result/benchmark_krea2_nvfp4.md`) is retained for technical reference.

---

## Technical Details & Architecture Analysis

### 1. Why Krea2 SingleStreamDiT Fails Under 4-Bit (NVFP4) Quantization

Krea2 utilizes a `SingleStreamDiT` architecture where text tokens and image tokens are concatenated and processed through unified transformer blocks. Empirical evaluations of NVFP4 quantization on this architecture revealed fundamental structural limitations:

1. **Catastrophic Error Propagation in Unified Attention**:
   In two-stream or separate-stream architectures (such as FLUX.1 or Qwen2), cross-attention and separate text/image projections isolate quantization noise. In Krea2's single-stream blocks, quantization noise in weight projections immediately corrupts both multimodal representations simultaneously, accelerating trajectory divergence from step 1.

2. **Sub-0.90 Final Latent Trajectory Cosine**:
   Even when applying HSWQ 4-axis sensitivity weighting ($E[x^2]$ energy, HistCosine V5, NVFP4 pack MSE, and SVD leverage) and selectively retaining high-impact blocks in original precision, the final latent trajectory cosine consistently fails to reach 0.90 across evaluation prompts and seeds.
   - In diffusion models, a latent trajectory cosine below 0.92 indicates observable degradation in fine detail, while a cosine below 0.90 leads to severe structural bifurcations, subject deformity, and prompt misalignment.

3. **Definitive Decision**:
   Because 4-bit precision fundamentally cannot guarantee structural generation integrity on Krea2, all NVFP4 development for Krea2 is discontinued. Users should exclusively use **ConvRot INT8**, which delivers ~50% VRAM / disk savings while maintaining indistinguishable visual quality (mean cosine $\ge 0.98$, 0 bifurcations).

---

### 2. SDXL INT8 Deterministic Trajectory Comparator (`benchmark/sdxl_int8_traj_compare.py`)

To ensure rigorous validation of SDXL ConvRot INT8 models, `benchmark/sdxl_int8_traj_compare.py` provides deterministic, reproducible trajectory benchmarking:

- **Step-by-Step Trajectory Tracking**: Compares intermediate latent vectors $z_t$ against the unquantized baseline at every denoising step.
- **Metrics Computed**:
  - Per-step and final Latent Trajectory Cosine Similarity ($\cos(z_t^{\text{quant}}, z_t^{\text{base}})$).
  - Mean Squared Error (MSE) and Latent SSIM.
  - Trajectory Divergence Index to identify the exact step where numerical drift begins.
- **CLI Usage**:
  ```bash
  python benchmark/sdxl_int8_traj_compare.py \
    --original "models/checkpoints/sdxl_base_bf16.safetensors" \
    --quantized "models/checkpoints/sdxl_convrot_int8.safetensors" \
    --seeds 10 \
    --steps 20 \
    --prompt "A professional studio portrait of an astronaut on Mars"
  ```

---

### 3. Self-Contained Benchmark Architecture

Benchmark scripts in `benchmark/` were decoupled from shared internal modules:
- Removed external module dependencies (e.g. `common_bench`) to eliminate `ModuleNotFoundError` across different execution environments.
- Self-contained CLI parsing, ComfyUI model loader discovery, and device allocation logic.
- Startup crash fix: restored `Path` from `pathlib` in `benchmark/krea2_traj_compare.py`.

---

## Recommended Quantization Matrix

| Target Model | Recommended Architecture / Format | CLI Converter | ComfyUI Custom Node | Expected Cosine |
| :--- | :--- | :--- | :--- | :--- |
| **Krea2 DiT** | **ConvRot INT8** (1off, blacklist protected) | `Krea2/hswq_convrot_int8_krea2_v1.5.py` | `Native ConvRot INT8 Quantize` | **$\ge 0.98$** |
| **SDXL UNet** | **ConvRot INT8** (Card 1 bias correction) | `sdxl_convert/convert_sdxl_convrot_int8.py` | `Native ConvRot INT8 Quantize` | **$\ge 0.99$** |
| **Qwen Image Edit** | **ConvRot INT8** (Hadamard rotation) | `Qwen Image/native_convert_int8_convrot_qwen.py` | `Native ConvRot INT8 Quantize` | **$\ge 0.99$** |
| **TE / ControlNet** | **ConvRot INT8** (FPN/QKV split) | `clip_convert/convert_clip_convrot_int8.py` | `TE / ControlNet ConvRot INT8 Quantize` | **$\ge 0.99$** |
| **Z-Image** | **HSWQ NVFP4** | `Z_Image/` | N/A | **$\ge 0.96$** |

*(Note: Krea2 NVFP4 is discontinued and removed from the active quantization matrix).*

---

## Verification & Documentation

- **Changelog**: [CHANGELOG.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/CHANGELOG.md)
- **Krea2 Quantization Guide**: [`md/How to quantize Krea2.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Krea2.md)
