# How to quantize Text Encoder and ControlNet (native ConvRot INT8)

Quantize loaded **Text Encoders (CLIP, T5, Qwen2.5-VL, etc.)** and **ControlNet / ControlNet Union models** directly in ComfyUI using the dedicated custom node **`TE / ControlNet ConvRot INT8 Quantize`** (`comfyui_nodes/te_controlnet_convrot_int8_convert.py`).

By applying orthogonal Hadamard rotations prior to per-channel INT8 quantization, this pipeline eliminates activation outlier spikes in deep Linear layers, reducing VRAM footprint by ~50% to 70% while preserving 100% conditioning accuracy and structural guidance fidelity.

---

## ComfyUI Node Workflow

Quantization is performed directly in-graph from memory without requiring external CLI conversion scripts:

<p align="left">
  <img src="../png/te_controlnet_convrot_int8.png" alt="ComfyUI TE and ControlNet ConvRot INT8 Quantize Workflow" width="600">
</p>

### Pipeline Structure

The workflow consists of two independent, parallel quantization branches:

1. **Text Encoder (CLIP / TE) Branch:**
   - **Loader:** `HSWQ Load CLIP (Simple)` (or standard `CLIPLoader` / `DualCLIPLoader` / `TripleCLIPLoader`).
   - **Quantizer:** `TE / ControlNet ConvRot INT8 Quantize` (connected via the `clip` terminal).
   - **Monitor:** `Show Text` (Custom-Scripts) connected to `report` to display real-time quantization layer statistics and output path.

2. **ControlNet Branch:**
   - **Loader:** `Load ControlNet Model` (or `HSWQ Load ControlNet (Simple)`).
   - **Quantizer:** `TE / ControlNet ConvRot INT8 Quantize` (connected via the `control_net` terminal).
   - **Monitor:** `Show Text` (Custom-Scripts) connected to `report` to display file size and layer breakdown.

---

## Clone the repository

```bash
git clone https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization.git
cd Hybrid-Sensitivity-Weighted-Quantization
```

## Install PyTorch (CUDA)

First, install PyTorch (CUDA).  
In a Windows environment on a local PC, it is advisable to set up a venv virtual environment.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Install other libraries

```bash
pip install -r requirements.txt
pip install -U comfy_kitchen
pip install diffusers accelerate scikit-image
```

`scikit-image` is required for SSIM and benchmark comparisons. `comfy_kitchen` provides the quantization kernels and layout operations.

---

## ComfyUI Custom Node Setup

Copy or link the repository into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization.git
```

---

## Node Parameters & Configuration

### `TE / ControlNet ConvRot INT8 Quantize`

| Input / Widget | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`clip`** | `CLIP` | Optional | Connected Text Encoder / CLIP model instance. |
| **`control_net`** | `CONTROL_NET` | Optional | Connected ControlNet or ControlNet Union model instance. |
| **`group_size`** | `INT` | `256` | Preferred ConvRot Hadamard rotation group size (must be a power of 4: `4`, `16`, `64`, `256`, `1024`). |
| **`convrot`** | `BOOLEAN` | `True` | Enables orthogonal Hadamard rotation prior to INT8 scaling. When disabled, falls back to plain per-channel INT8. |
| **`output_path`** | `STRING` | `""` | Destination `.safetensors` path or directory. If left empty, saves automatically to `ComfyUI/output/` using `<model_name>_convrot_int8.safetensors`. |

> [!NOTE]
> At least one model input (`clip` or `control_net`) must be connected. You can also connect **both** simultaneously to quantize a matching Text Encoder and ControlNet in a single queue execution.

---

## Quantization Mechanics & Format

1. **State-Dict Extraction:**
   The node extracts the unquantized weights directly from the loaded model instance in memory. Upstream loader filenames are automatically traced through the prompt execution graph to derive clear, consistent checkpoint names.

2. **Selective Linear Quantization:**
   - All 2D floating-point Linear weights (`.weight` with `ndim == 2`) are processed with Hadamard rotation:
     $$\mathbf{W}_{\text{rot}} = \mathbf{W} \mathbf{H}^T$$
   - Quantized to signed 8-bit integers with per-channel scaling:
     $$\text{scale} = \frac{\max(|\mathbf{W}_{\text{rot}}|)}{127}, \quad \mathbf{W}_q = \text{round}\left(\frac{\mathbf{W}_{\text{rot}}}{\text{scale}}\right)$$
   - Non-2D weights (embeddings, 1D layer norms, biases) are preserved unquantized in original precision (FP16/BF16/FP32) to prevent precision collapse at model boundaries.

3. **Output Checkpoint Layout:**
   ```
   <layer>.weight           int8
   <layer>.weight_scale     float32  [out_features, 1]
   <layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
   _quantization_metadata   JSON  {"format_version":"1.0","layers":{...}}
   ```

---

## Loading Quantized Models in ComfyUI

### ControlNet Models
ComfyUI does not natively support rotated INT8 tensors for ControlNet models out of the box. To load and execute these weights in ComfyUI workflows, use the dedicated loader from the **[ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)** extension:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools.git
```

Use the **`HSWQ Load ControlNet (Simple)`** node to load the generated `.safetensors` checkpoint directly into your generation graph.

### Text Encoders (CLIP / TE)
Quantized Text Encoders stamped with `comfy_quant` can be loaded using standard ComfyUI `CLIPLoader` / `DualCLIPLoader` or `HSWQ Load CLIP (Simple)`.

---

## Supported Architectures & Verified Checkpoints

| Architecture | Model Family | Verified Models |
| :--- | :--- | :--- |
| **Illustrious-XL / SDXL** | Anytest ControlNet | `CN-anytest4_illustrious2_A`, `CN-anytest4_illustrious2_B` |
| **SDXL 1.0** | ControlNet Union | `controlnet-union-pro-max-sdxl-1.0` (xinsir) |
| **Qwen-Image** | ControlNet Inpainting | `Qwen-Image-ControlNet-Inpainting` (alibaba-pai) |
| **Qwen-Image-2512** | ControlNet Union | `Qwen-Image-2512-Fun-Controlnet-Union-2602` |
| **FLUX.1-dev** | ControlNet Union Pro | `FLUX.1-dev-ControlNet-Union-Pro-2.0` (Shakker Labs) |
| **CLIP / Text Encoders** | CLIP-L, CLIP-G, T5-XXL, Qwen2.5-VL | `CLIP-SAE-ViT-L-14`, `t5xxl_fp16`, `qwen_2.5_vl_7b` |
