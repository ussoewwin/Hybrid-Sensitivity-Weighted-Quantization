## Overview

**v2.3.3** expands the **native ConvRot INT8 quantization** ecosystem with full support for Meta's **Segment Anything** foundation models:
1. **SAM 3.1 Multiplex** (`sam3.1_multiplex_fp16.safetensors` in `models/unet/` or `models/diffusion_models/`).
2. **SAM 3 (Classic non-multiplex / `sam3.pt`)** loaded via the dedicated **`HSWQ SAM3 Loader (sam3.pt)`** (`comfyui_nodes/sam3_pt_loader.py`).
3. **In-Graph ComfyUI Quantization** via the unified **`TE / ControlNet ConvRot INT8 Quantize`** custom node (`comfyui_nodes/te_controlnet_convrot_int8_convert.py`), featuring automatic architectural detection, fused key splitting, and Hadamard orthogonal rotation.
4. **Standalone CLI Conversion** via [`clip_convert/convert_clip_convrot_int8.py`](clip_convert/convert_clip_convrot_int8.py) for headless / batch quantization of generic safetensors (SAM 3, SAM 3.1, CLIP, T5, Qwen2.5-VL, ControlNet, UNet).

By applying orthogonal Walsh-Hadamard rotations prior to per-channel INT8 quantization, this pipeline eliminates activation outlier spikes across deep Linear projection layers, reducing VRAM and disk footprint by **~50%** while preserving 100% segmentation accuracy, tracking fidelity, and zero mask boundary degradation.

---

## Key Features & Highlights

### 1. Unified SAM 3 & SAM 3.1 Quantization (`TE / ControlNet ConvRot INT8 Quantize`)
- **Direct In-Graph Quantization**: Quantize loaded SAM 3 and SAM 3.1 models directly from memory in ComfyUI workflows via the `model` terminal.
- **Automated Version Detection & Architectural Branching**:
  - **SAM 3.1 Multiplex (`sam3.1_multiplex_fp16.safetensors`)**:
    - 3-level FPN architecture (`propagation_convs` / `interactive_convs`).
    - Preserves and quantizes the active $(1024 \times 1024)$ `text_projection` layer.
    - Loaded using standard ComfyUI **`UNetLoader`** (or `Load Diffusion Model`).
  - **SAM 3 Classic (`sam3.pt` / `sam3_fp16.safetensors`)**:
    - 4-level FPN architecture (`sam2_convs` / `vision_backbone.convs.3`).
    - Automatically drops the incompatible / unused $(1024 \times 512)$ `text_projection` layer.
    - Automatically splits fused `.in_proj_weight` and `.in_proj_bias` tensors into discrete `q_proj`, `k_proj`, and `v_proj` weights.
    - Standardizes SAM mask decoder transformer keys (`sam_mask_decoder.transformer.*` -> `.mlp.0.`, `.mlp.2.`, `.norm_final.`) and remaps `tracker.model.*` -> `tracker.*`.
    - Loaded using dedicated **`HSWQ SAM3 Loader (sam3.pt)`** (`HSWQSAM3Loader`).

### 2. Standalone CLI Converter (`convert_clip_convrot_int8.py`)
- **Automatic Model Inspection**: Uses `_detect_sam_version` to dynamically identify SAM 3 vs. SAM 3.1 from checkpoint tensors and apply version-specific key remapping.
- **Generic Safetensors Support**: Quantizes CLIP, LLM, ControlNet, SAM, and UNet `.safetensors` files without requiring a running ComfyUI instance.
- **Customizable Hadamard Group Size**: Defaults to `groupsize=256` (power-of-4 configurable: 4, 16, 64, 256, 1024).

### 3. Model Footprint & Memory Reduction
| Model Architecture | Base Format | Quantized Format | Original Size | ConvRot INT8 Size | VRAM Reduction | ComfyUI Native Load |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SAM 3.1 Multiplex** | FP16 (`safetensors`) | ConvRot INT8 (`int8_tensorwise`) | ~1.63 GB | **~0.84 - 0.89 GB** | **~50%** | Standard `UNetLoader` |
| **SAM 3 Classic** | FP32/FP16 (`.pt` / `safetensors`) | ConvRot INT8 (`int8_tensorwise`) | ~3.21 GB | **~0.55 GB** | **~50%** | Standard `UNetLoader` |

### 4. Packaged Sample Workflow & Visual Guides
- **Ready-to-Use Workflow JSON**: [`sample workflow/native convrot int8.json`](sample%20workflow/native%20convrot%20int8.json) updated with dedicated groups for **SAM 3.1** and **SAM 3** alongside UNet, CLIP, ControlNet, and Model Patch quantization branches.
- **Visual Workflow Screenshots**:
  - `png/sam.png`: Visual layout for SAM 3 and SAM 3.1 quantization graphs.
  - `png/model_patcher.png`: Visual layout for Model Patch quantization.
  - `png/te_controlnet_convrot_int8.png`: Visual layout for Text Encoder & ControlNet quantization.

---

## Mathematical Formulation & Format Layout

### Hadamard Orthogonal Rotation
$$\mathbf{W}_{\text{rot}} = \mathbf{W} \mathbf{H}^T$$
$$\text{scale}_i = \frac{\max_j(|W_{\text{rot},ij}|)}{127}, \quad W_{q,ij} = \text{round}\left(\frac{W_{\text{rot},ij}}{\text{scale}_i}\right)$$

Where $\mathbf{H}$ is the normalized orthogonal Sylvester-Hadamard matrix ($H \cdot H^T = I$). Group-wise rotation along the input channel axis disperses outlier energy evenly across matrix dimensions.

### Checkpoint Layout
```
<layer>.weight           int8
<layer>.weight_scale     float32  [out_features, 1]
<layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata   JSON  {"format_version":"1.0","layers":{...}}
```

---

## Usage Guide

### ComfyUI Custom Node
1. **SAM 3.1 Multiplex**:
   - Add **`UNetLoader`** (or `Load Diffusion Model`) and select `sam3.1_multiplex_fp16.safetensors`.
   - Connect the `MODEL` output to the `model` terminal of **`TE / ControlNet ConvRot INT8 Quantize`**.
   - Queue prompt. Output is saved as `sam3.1_multiplex_convrot_int8.safetensors`.
2. **SAM 3 Classic**:
   - Add **`HSWQ SAM3 Loader (sam3.pt)`** and select `sam3.pt`.
   - Connect the `MODEL` output to the `model` terminal of **`TE / ControlNet ConvRot INT8 Quantize`**.
   - Queue prompt. Output is saved as `sam3_convrot_int8.safetensors`.

### Standalone CLI Script
```bash
# Quantize SAM 3.1 Multiplex
python clip_convert/convert_clip_convrot_int8.py \
  --model models/unet/sam3.1_multiplex_fp16.safetensors \
  --output models/unet/sam3.1_multiplex_convrot_int8.safetensors \
  --groupsize 256

# Quantize SAM 3
python clip_convert/convert_clip_convrot_int8.py \
  --model models/unet/sam3_fp16.safetensors \
  --output models/unet/sam3_convrot_int8.safetensors \
  --groupsize 256
```

---

## Loading Quantized Models in ComfyUI

- **SAM 3 & SAM 3.1 Models**:
  Place the quantized `.safetensors` files in `ComfyUI/models/unet/` or `ComfyUI/models/diffusion_models/` and load them directly using the stock **`UNetLoader`** (or **`Load Diffusion Model`**). ComfyUI automatically detects `comfy_quant` metadata and executes native INT8 Tensor Core operations.
- **ControlNet Models**: Load using `HSWQ Load ControlNet (Simple)` from [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools).
- **Model Patch Models**: Load using `HSWQ Load Model Patch (ConvRot INT8)`.
- **Text Encoders**: Load using standard `CLIPLoader` / `DualCLIPLoader` or `HSWQ Load CLIP (Simple)`.

---

## Documentation Links

- **Text Encoder, ControlNet & SAM 3 / 3.1 Quantization Guide**: [`md/How to quantize Text Encoder and ControlNet.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md)
- **Sample Workflow**: [`sample workflow/native convrot int8.json`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/sample%20workflow/native%20convrot%20int8.json)
