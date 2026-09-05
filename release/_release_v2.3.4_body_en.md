## Overview

**v2.3.4** releases complete native and HSWQ **ConvRot INT8 quantization** support for **Krea2 DiT** (SingleStreamDiT):

1. **Structural Blacklist Protection**: Preserves entry/exit embeddings (`first.`, `last.`), adaptive modulation (`mod.`), layer norms (`norm`), patch projectors (`projector`), time/text MLPs (`tmlp`, `txtmlp`, `tproj`), text cross-attention projection (`txtfusion`), and all `bias` tensors in original precision (BF16/FP32), completely eliminating numerical collapse and black latent output.
2. **4-Axis Composite Sensitivity Ranking**: Employs DualMonitor $E[x^2]$ activation energy $\times$ HistCosine V5 distribution alignment $\times$ NVFP4 measured pack MSE $\times$ SVD structural leverage to rank and retain high-impact DiT weights (`--blacklist_keep` and `--keep_sensitive`).
3. **Card 1 Bias Correction Omission (`1off`)**: Standardized on `1off` because all quantized transformer blocks in Krea2 `SingleStreamDiT` (Attention, SwiGLU, TextFusion) are strictly `bias=False`. All biased layers reside in the structural blacklist, rendering bias delta calculation a complete no-op.
4. **ComfyUI Custom Node Integration**: Added Krea2 DiT conversion to the unified **`Native ConvRot INT8 Quantize`** node (`comfyui_nodes/native_convrot_int8_convert.py`), providing 1-click in-graph native ConvRot INT8 model quantization directly in ComfyUI workflows.
5. **Native ConvRot INT8 Recommendation**: For checkpoints where native ConvRot INT8 achieves mean latent trajectory cosine $\ge 0.98$ (with 0 trajectory bifurcations), **using native ConvRot INT8 directly without HSWQ is recommended**, maintaining virtually identical image composition with ~50% model storage size reduction.
6. **Multi-Seed Automated Trajectory & Cosine Benchmark**: Added `benchmark/krea2_int8_traj_compare.py` to evaluate latent fidelity and trajectory stability against the original BF16 baseline across 12 sampling steps over 20 random seeds.
7. **Comprehensive Technical Guide**: Published full documentation in [`md/How to quantize Krea2.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Krea2.md).

---

## Key Features & Highlights

### 1. Structural Blacklist Protection (`_KREA2_BLACKLIST`)
Krea2 SingleStreamDiT models exhibit severe sensitivity at the entry/exit boundaries and adaptive modulation paths. Directly quantizing these layers leads to numerical instability and empty/black latent outputs. The pipeline enforces structural protection on:
- `first.`: Initial patch embedding and condition injection.
- `last.`: Final linear projection to latent velocity/noise.
- `mod.`: Adaptive modulation vectors controlling block residual scaling.
- `norm`: All RMSNorm and LayerNorm weight vectors.
- `projector`: Text-image cross-stream projection bridges.
- `tmlp`, `txtmlp`, `tproj`: Time and text conditioning projection layers.
- `txtfusion`: Text fusion projection heads.
- `bias`: Any tensor containing bias parameters.

All structural blacklist layers remain in original precision (BF16/FP32), safeguarding model convergence.

### 2. 4-Axis Composite Sensitivity Ranking
For high-fidelity calibration when `--keep_sensitive` is enabled, layer retention priority is computed via 4 orthogonal analytical pillars:
$$\text{Score}(L) = E[x^2]_L \times (1 - \text{Cosine}_{\text{HistV5}, L}) \times \text{MSE}_{\text{NVFP4}, L} \times \text{Leverage}_{\text{SVD}, L}$$
- **DualMonitor $E[x^2]$**: Measures forward activation energy and dynamic range expansion.
- **HistCosine V5**: Evaluates directional loss against the optimal physical quantization grid.
- **NVFP4 Pack MSE**: Empirical single-layer reconstruction error on calibration latents.
- **SVD Structural Leverage**: Top singular vector energy distribution reflecting core representation capacity.

### 3. Rationale for `1off` Bias Setting
In SDXL and earlier architectures, Card 1 bias correction compensates for quantization centroid shift. In Krea2 `SingleStreamDiT`:
- All 28 transformer blocks (`SingleStreamBlock`, `Attention`, `SwiGLU`, and `TextFusionTransformer`) are initialized with `bias=False`.
- The only layers containing `bias=True` (`first.`, `last.`, `tmlp`, `txtmlp`, `tproj`) are structurally protected in BF16/FP32.
- 100% of quantized INT8 layers have `bias is None`.
Therefore, Card 1 bias correction has zero mathematical effect on Krea2, and `1off` is established as the canonical, optimal setting.

### 4. Native ConvRot INT8 Recommendation Gate
Extensive benchmark runs across 20 random seeds demonstrate that native ConvRot INT8 with structural blacklist protection achieves:
- Mean latent trajectory cosine $\ge 0.98$.
- 0 trajectory bifurcations (0 instances of cosine $< 0.85$ or drift).
- Virtually indistinguishable image composition and fine structural fidelity.

When a checkpoint satisfies this fidelity gate, using **native ConvRot INT8 directly without HSWQ is recommended**. If a specific fine-tune exhibits drift or bifurcation (mean cosine $< 0.98$), full HSWQ sensitivity layer retention (`--keep_sensitive 10` or `15`) can be selectively applied.

---

## Model Storage Footprint

| Model Architecture | Base Format | Quantized Format | Original Size | ConvRot INT8 Size | File Size Reduction | ComfyUI Native Load |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Krea2 DiT (Native ConvRot INT8)** | BF16 (`safetensors`) | ConvRot INT8 (`int8_tensorwise`) | ~9.5 GB | **~4.8 GB** | **~50%** | Standard `Load Diffusion Model` / `UNetLoader` |
| **Krea2 DiT (HSWQ ConvRot INT8, k10)** | BF16 (`safetensors`) | ConvRot INT8 + BF16 retain | ~9.5 GB | **~5.1 GB** | **~46%** | Standard `Load Diffusion Model` / `UNetLoader` |

---

## Mathematical Formulation & Format Layout

### Hadamard Orthogonal Rotation
Prior to per-channel INT8 scaling, weights and activations are transformed by an orthogonal Sylvester-Hadamard matrix $\mathbf{H}$ ($H \cdot H^T = I$):
$$\mathbf{W}_{\text{rot}} = \mathbf{W} \mathbf{H}^T$$
$$\text{scale}_i = \frac{\max_j(|W_{\text{rot},ij}|)}{127}, \quad W_{q,ij} = \text{round}\left(\frac{W_{\text{rot},ij}}{\text{scale}_i}\right)$$

This eliminates outlier activation peaks and disperses quantization error uniformly across channels, allowing native INT8 Tensor Core execution without outlier degradation.

### Checkpoint Layout & Metadata
```
<layer>.weight           int8
<layer>.weight_scale     float32  [out_features, 1]
<layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata   JSON  {"format_version":"1.0","model_type":"krea2","layers":{...}}
```

---

## Usage Guide

### ComfyUI Custom Node
1. Load a Krea2 checkpoint via standard **`Load Diffusion Model`** (or **`UNetLoader`**).
2. Connect the `MODEL` output to the `model` terminal of **`Native ConvRot INT8 Quantize`** (`comfyui_nodes/native_convrot_int8_convert.py`).
3. Set `model_type` to `krea2` and queue prompt. The quantized model is saved with embedded `comfy_quant` metadata.

### Standalone CLI Quantizer
```bash
# Recommended standard Native ConvRot INT8 conversion (1off, blacklist protected)
python Krea2/hswq_convrot_int8_krea2_v1.5.py \
  --model "models/diffusion_models/krea2_bf16.safetensors" \
  --output "models/diffusion_models/krea2_convrot_int8.safetensors" \
  --bias_correction 1off

# HSWQ ConvRot INT8 with sensitive layer retention (10 DiT layers retained in BF16)
python Krea2/hswq_convrot_int8_krea2_v1.5.py \
  --model "models/diffusion_models/krea2_bf16.safetensors" \
  --output "models/diffusion_models/krea2_convrot_int8_k10.safetensors" \
  --bias_correction 1off \
  --keep_sensitive 10
```

### Automated Trajectory & Cosine Benchmark
Evaluate quantized model fidelity against original BF16 across 12 sampling steps over 20 random seeds:
```bash
python benchmark/krea2_int8_traj_compare.py \
  --original "models/diffusion_models/krea2_bf16.safetensors" \
  --quantized "models/diffusion_models/krea2_convrot_int8.safetensors" \
  --seeds 20 \
  --steps 12
```

---

## Loading Quantized Models in ComfyUI

Place the converted `.safetensors` file in `ComfyUI/models/diffusion_models/` or `ComfyUI/models/unet/`.
- Load directly using the standard ComfyUI **`Load Diffusion Model`** or **`UNetLoader`**.
- ComfyUI automatically detects `comfy_quant` metadata, initializes `comfy_kitchen` online activation rotation, and dispatches native INT8 Tensor Core GEMM operations.

---

## Documentation Links

- **Krea2 Quantization Technical Guide**: [`md/How to quantize Krea2.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Krea2.md)
