## Overview

**v2.3.5** formalizes the architectural decision on Krea2 quantization and announces the official discontinuation of 4-bit (NVFP4) development for Krea2:

- **Krea2 Hybrid ConvRot NVFP4 Development Discontinued**: Extensive empirical evaluations confirmed that 4-bit precision (NVFP4) cannot maintain structural generation fidelity on Krea2 SingleStreamDiT. Even with HSWQ sensitivity weighting and layer retention, final latent trajectory cosine fails to reach 0.90 (resulting in severe trajectory drift and frequent bifurcations).
- **Krea2 Supported Strictly via ConvRot INT8**: Krea2 is supported exclusively via **ConvRot INT8** (`Krea2/hswq_convrot_int8_krea2_v1.5.py` and ComfyUI `Native ConvRot INT8 Quantize`), which reliably achieves mean trajectory cosine $\ge 0.98$ with 0 bifurcations, preserving generation fidelity with ~50% VRAM / disk reduction.

---

## Technical Details: Why Krea2 SingleStreamDiT Cannot Support 4-Bit (NVFP4)

Krea2 employs a unified `SingleStreamDiT` architecture where text tokens and image tokens are concatenated into a single sequence and processed simultaneously across unified transformer blocks. Empirical testing across diverse prompts and random seeds revealed fundamental limitations under 4-bit quantization:

### 1. Inherent Vulnerability of Single-Stream Unified Attention to 4-Bit Noise
In dual-stream or cross-attention architectures, text conditioning and image features maintain separate projection pathways, which confines quantization error within individual streams. In Krea2 SingleStreamDiT, any quantization noise introduced in the 4-bit weight projections immediately contaminates both text and image representations at every attention step. This causes quantization errors to compound rapidly from the very first denoising steps.

### 2. Failure to Reach 0.90 Final Latent Trajectory Cosine
Even when deploying HSWQ 4-axis composite sensitivity ranking (dual activation energy, HistCosine V5, NVFP4 pack MSE, and SVD leverage) and selectively retaining high-sensitivity layers in original precision:
- **Final Trajectory Cosine**: Consistently fails to reach **0.90**.
- **Visual Artifacts**: A final cosine below 0.90 leads to severe trajectory drift, prompt misalignment, anatomical distortions, and complete structural bifurcations.
- **Precision Floor**: 4-bit representation lacks the dynamic range necessary to preserve the fine latent representations required by Krea2's single-stream blocks.

### 3. Definitive Architectural Conclusion
Because 4-bit precision fundamentally cannot guarantee structural generation integrity on Krea2 even with HSWQ optimization, **all Krea2 Hybrid ConvRot NVFP4 development is cancelled**.

---

## Supported Quantization Path: Krea2 ConvRot INT8

Krea2 is supported strictly via **ConvRot INT8**:

| Metric / Configuration | ConvRot INT8 | Hybrid ConvRot NVFP4 (Cancelled) |
| :--- | :--- | :--- |
| **Mean Trajectory Cosine** | **$\ge 0.98$** | **$< 0.90$** (Fails fidelity threshold) |
| **Trajectory Bifurcations** | **0** | Frequent |
| **Storage / VRAM Footprint** | **~50%** of original BF16 | ~30% of original BF16 |
| **Visual Fidelity** | Indistinguishable from BF16 | Severe degradation & collapse |
| **Status** | **Production Supported** | **Cancelled** |

### Usage for Krea2 ConvRot INT8

#### Standalone CLI Quantizer
```bash
# Canonical Native ConvRot INT8 conversion (1off bias, structural blacklist protected)
python Krea2/hswq_convrot_int8_krea2_v1.5.py \
  --model "models/diffusion_models/krea2_bf16.safetensors" \
  --output "models/diffusion_models/krea2_convrot_int8.safetensors" \
  --bias_correction 1off
```

#### ComfyUI Custom Node
In ComfyUI workflows, use the **`Native ConvRot INT8 Quantize`** node (`comfyui_nodes/native_convrot_int8_convert.py`) with `model_type` set to `krea2`.

---

## Documentation Links

- **Krea2 Quantization Technical Guide**: [`md/How to quantize Krea2.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Krea2.md)
- **Changelog**: [CHANGELOG.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/CHANGELOG.md)
