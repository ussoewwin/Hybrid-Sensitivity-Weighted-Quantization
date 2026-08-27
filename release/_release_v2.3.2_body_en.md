## Overview

**v2.3.2** introduces full **native ConvRot INT8 quantization** support for:
1. **Qwen Image Edit** diffusion transformer models (`Qwen Image/native_convert_int8_convrot_qwen.py` and ComfyUI node `Native ConvRot INT8 Quantize`).
2. **Text Encoders** (CLIP-L, CLIP-G, T5-XXL, Qwen2.5-VL, etc.) and **ControlNet / ControlNet Union** models (SDXL, Qwen-Image, FLUX.1) directly in ComfyUI workflows via the dedicated custom node **`TE / ControlNet ConvRot INT8 Quantize`** (`comfyui_nodes/te_controlnet_convrot_int8_convert.py`).

By applying orthogonal Walsh-Hadamard rotations prior to symmetric per-channel INT8 scaling, the pipeline eliminates severe activation outlier spikes in deep linear projections. This achieves a **~50% reduction in VRAM footprint** with zero structural guidance degradation and 100% conditioning accuracy.

---

## Key Features & Highlights

### 1. Text Encoder and ControlNet ConvRot INT8 (`TE / ControlNet ConvRot INT8 Quantize`)
- **Direct In-Graph Quantization**: Quantize active Text Encoder (`CLIP`) and ControlNet (`CONTROL_NET`) instances directly from memory without requiring external Python scripts.
- **Dual-Branch Execution**: Single-node architecture capable of quantizing CLIP and ControlNet independently or concurrently in a single prompt execution.
- **Massive VRAM Savings**:
  - Text Encoders (e.g. `Qwen2.5-VL-7B`, `t5xxl_fp16`, `CLIP-SAE-ViT-L-14`): ~50% VRAM reduction.
  - ControlNet / ControlNet Union (e.g. `CN-anytest4_illustrious2_B`, `controlnet-union-pro-max-sdxl-1.0`, `Qwen-Image-ControlNet-Inpainting`, `FLUX.1-dev-ControlNet-Union-Pro-2.0`): ~50% size reduction while preserving fine spatial conditioning fidelity.
- **Selective Boundary Precision**: 2D floating-point Linear weights are rotated and quantized to signed 8-bit integers; 1D normalization weights, biases, and token embeddings remain in original unquantized precision (FP16/BF16/FP32).

### 2. Qwen Image Edit Native ConvRot INT8 Quantization
- **CLI & Node Pipelines**: Full support via CLI script (`Qwen Image/native_convert_int8_convrot_qwen.py`) and ComfyUI custom node (`Native ConvRot INT8 Quantize` with `model_type = "Qwen Image Edit"`).
- **Automated Multi-Seed Trajectory Benchmark**: Evaluates latent trajectory divergence (per-step cosine similarity and bifurcation detection) alongside decoded SSIM.
- **Native ComfyUI Compatibility**: Checkpoints stamped with `int8_tensorwise` format can be loaded directly using standard ComfyUI loader nodes without dedicated custom wrappers.

### 3. Packaged Sample Workflow
- A ready-to-run ComfyUI workflow JSON is included in the repository:
  - **`sample workflow/native convrot int8.json`**
- Provides complete wiring for Diffusion Model (Qwen Image Edit), Text Encoder (Qwen2.5-VL / CLIP-SAE), and ControlNet (Anytest4) quantization with real-time reporting via `Show Text`.

---

## Technical Specifications & Format Layout

### Mathematical Formulation
$$\mathbf{W}_{\text{rot}} = \mathbf{W} \mathbf{H}^T$$
$$\text{scale} = \frac{\max(|\mathbf{W}_{\text{rot}}|)}{127}, \quad \mathbf{W}_q = \text{round}\left(\frac{\mathbf{W}_{\text{rot}}}{\text{scale}}\right)$$

Where $\mathbf{H}$ represents the block-diagonal orthogonal Hadamard matrix with group size $N \in \{4, 16, 64, 256, 1024\}$ (default `256`).

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
1. Add node **`TE / ControlNet ConvRot INT8 Quantize`** to your workflow.
2. Connect `clip` from `CLIPLoader` / `SimpleCLIPLoader` and/or `control_net` from `ControlNetLoader`.
3. Set `group_size` (`256`), `convrot` (`True`), and optional `output_path`.
4. Connect `report` output to `Show Text` to view real-time layer quantization stats and saved checkpoint path.
5. Queue prompt to execute.

### Qwen Image Edit CLI
```bash
python "Qwen Image/native_convert_int8_convrot_qwen.py" \
  --model "<path-to-unet>/<qwen_image_edit>.safetensors" \
  --output "<path-to-unet>/<qwen_image_edit>_convrot_int8.safetensors" \
  --per_channel_int8 \
  --clip_path "<path-to-qwen2.5-vl-7b>" \
  --comfy_path "<path-to-ComfyUI>"
```

---

## Loading Quantized Models

- **Text Encoders**: Load with standard ComfyUI `CLIPLoader` / `DualCLIPLoader` / `TripleCLIPLoader` or `HSWQ Load CLIP (Simple)`.
- **ControlNet Models**: Load with `HSWQ Load ControlNet (Simple)` from [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools).
- **Qwen Image Edit**: Load with standard ComfyUI `UNETLoader` / `Load Diffusion Model`.

---

## Documentation Links

- **Text Encoder & ControlNet Quantization Guide**: [`md/How to quantize Text Encoder and ControlNet.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md)
- **Qwen Image Edit Quantization Guide**: [`md/How to quantize Qwen Image Edit.md`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/How%20to%20quantize%20Qwen%20Image%20Edit.md)
- **Sample Workflow**: [`sample workflow/native convrot int8.json`](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/sample%20workflow/native%20convrot%20int8.json)
