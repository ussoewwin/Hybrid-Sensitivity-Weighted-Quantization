---
license: apache-2.0
tags:
- seedvr2
- seedvr
- vae
- tensorrt
- rtxplan
- blackwell
- rtx-5090
- rtx-5080
- rtx-5070
- rtx-5060
- rtx-5050
- video-upscaling
- video-to-video
- image-to-image
- comfyui
pipeline_tag: video-to-video
---

# SeedVR2 VAE TensorRT Engines for NVIDIA Blackwell

High-performance, pre-compiled **NVIDIA TensorRT** engine plans (`.rtxplan`) built for the official **SeedVR2 VAE** (`seedvr2_ema_vae_fp16`), specifically compiled and optimized for the **NVIDIA Blackwell** GPU architecture (**RTX 5090**, **RTX 5080**, **RTX 5070**, **RTX 5060**, **RTX 5050**, and Blackwell workstation / data center GPUs, Compute Capability `sm_100` / `sm_120`).

---

## 🌟 Overview

**SeedVR2** is a state-of-the-art diffusion-based video restoration and super-resolution model developed by ByteDance. The video VAE decoder component (`seedvr2_ema_vae_fp16`) handles spatial-temporal latent decoding back to RGB frames, which represents a primary computational and memory bottleneck during video upscaling and restoration workflows.

This repository provides pre-compiled TensorRT execution engine plans (`.rtxplan`) optimized for high-throughput, low-latency tiled spatial-temporal inference on NVIDIA Blackwell hardware:

- **Hardware Target:** NVIDIA Blackwell architecture (`sm_100` / `sm_120`, RTX 50-Series: **RTX 5090**, **RTX 5080**, **RTX 5070**, **RTX 5060**, **RTX 5050**).
- **Spatial Configuration:** Optimized for deterministic 256×256 spatial tiling (`tile_256`) to maximize L2 cache utilization, minimize VRAM spikes, and prevent out-of-memory errors on 4K/8K video upscaling.
- **Temporal Configuration:** Comprehensive temporal coverage spanning single-image inference (`1f`) up to 97 frames (`97f`) in standard **4n+1** sequence increments (`1f`, `5f`, `21f`, `25f`, `29f`, `33f`, `37f`, `41f`, `45f`, `49f`, `53f`, `57f`, `61f`, `65f`, `69f`, `73f`, `77f`, `81f`, `85f`, `89f`, `93f`, `97f`).
- **Base Checkpoint:** [Comfy-Org/SeedVR2 (seedvr2_ema_vae_fp16)](https://huggingface.co/Comfy-Org/SeedVR2) / ByteDance SeedVR2.
- **ComfyUI Loader & Upscaler Node:** [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder)

---

## 📦 Available TensorRT Engine Plans

### 🔄 Decoder Engines (`vae_decoder_tile_256_*f.rtxplan`)
Optimized for decoding compressed spatial-temporal latents back to RGB video frame sequences or still images.

| Engine Filename | Module | Tile Size | Temporal Frames | File Size | Target Architecture |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vae_decoder_tile_256_1f.rtxplan` | VAE Decoder | 256×256 | 1 frame (Still Image) | ~306 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_5f.rtxplan` | VAE Decoder | 256×256 | 5 frames | ~308 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_21f.rtxplan` | VAE Decoder | 256×256 | 21 frames | ~308 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_25f.rtxplan` | VAE Decoder | 256×256 | 25 frames | ~308 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_29f.rtxplan` | VAE Decoder | 256×256 | 29 frames | ~308 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_33f.rtxplan` | VAE Decoder | 256×256 | 33 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_37f.rtxplan` | VAE Decoder | 256×256 | 37 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_41f.rtxplan` | VAE Decoder | 256×256 | 41 frames | ~308 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_45f.rtxplan` | VAE Decoder | 256×256 | 45 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_49f.rtxplan` | VAE Decoder | 256×256 | 49 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_53f.rtxplan` | VAE Decoder | 256×256 | 53 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_57f.rtxplan` | VAE Decoder | 256×256 | 57 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_61f.rtxplan` | VAE Decoder | 256×256 | 61 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_65f.rtxplan` | VAE Decoder | 256×256 | 65 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_69f.rtxplan` | VAE Decoder | 256×256 | 69 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_73f.rtxplan` | VAE Decoder | 256×256 | 73 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_77f.rtxplan` | VAE Decoder | 256×256 | 77 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_81f.rtxplan` | VAE Decoder | 256×256 | 81 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_85f.rtxplan` | VAE Decoder | 256×256 | 85 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_89f.rtxplan` | VAE Decoder | 256×256 | 89 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_93f.rtxplan` | VAE Decoder | 256×256 | 93 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |
| `vae_decoder_tile_256_97f.rtxplan` | VAE Decoder | 256×256 | 97 frames | ~307 MB | NVIDIA Blackwell (`sm_100` / `sm_120`) |

> **Note on 4n+1 Temporal Structure:**
> SeedVR2 utilizes a 3D causal temporal downsampling VAE architecture. The temporal latent dimension corresponds to $(T - 1) / 4 + 1$. Consequently, video frame sequences must be compiled and decoded in $4n + 1$ frame blocks (e.g. 5, 21, 25, 29, ..., 97). For single image restoration, use the dedicated `1f` engine.

---

## 🛠️ Performance & Architectural Advantages

1. **Native Blackwell Kernel Tuning:**
   Built and tuned specifically with the TensorRT Blackwell compilation pipeline, fully utilizing Blackwell 5th-Generation Tensor Cores, enhanced memory bandwidth, and fused convolution-activation operators across the entire RTX 50-Series lineup (RTX 5090, RTX 5080, RTX 5070, RTX 5060, RTX 5050).

2. **Deterministic Tiled Processing (Tile 256):**
   Fixed spatial tiling at 256×256 ensures bounded VRAM consumption regardless of full video resolution (1080p, 4K, 8K), eliminating Out-of-Memory (OOM) failures and enabling continuous batching.

3. **Substantial Latency Reduction:**
   Provides significant speedups over standard PyTorch FP16 eager and `torch.compile` execution, removing VAE decode bottlenecks in iterative video generation and upscaling pipelines.

---

## 🚀 Usage in ComfyUI

These TensorRT engine plans (`.rtxplan`) are directly loaded and executed via the dedicated custom node:
- **Loader & Node Repository:** [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder)

### 1. Installation
Clone the custom node repository into your ComfyUI `custom_nodes/` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder.git
```

### 2. Engine Placement
Place the downloaded `.rtxplan` engine files into the `tensorrt_backend/artifacts/` folder:
```
ComfyUI/custom_nodes/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/tensorrt_backend/artifacts/vae_decoder_tile_256_<frames>f.rtxplan
```
Upon launching ComfyUI, all downloaded frame engines will automatically populate the `engine_frames` dropdown list of the **`SeedVR2 Load TensorRT VAE Decoder`** node.

### 3. Workflow Examples

#### Complete Video Upscaler Workflow (TensorRT VAE & Quantized Models)
Workflow JSON: [`example_workflows/SeedVR2_tensorrt_decode.json`](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/blob/main/example_workflows/SeedVR2_tensorrt_decode.json)

![Usage Example - Full Workflow](docs/usage_01.png)

#### TensorRT VAE Decoder Node Integration
Connect the `TRT_VAE` output from **`SeedVR2 Load TensorRT VAE Decoder`** directly into the **`SeedVR2 Video Upscaler`** node:

![Usage Example - TensorRT VAE Decoder](docs/usage_02.png)

---

## 🔧 Building Custom Frame Engines on Demand

The custom node repository also includes a dedicated **`SeedVR2 Build TensorRT VAE Engines`** node to compile custom frame sizes directly inside ComfyUI:

- **Supported Frames:** Any custom frame count following the $4n+1$ format (e.g. 101f, 185f, 205f).
- **Supported Tile Sizes:** `256` (optimal VRAM / long sequences) and `512` (higher spatial patch quality).
- **Supported Targets:** `decoder`, `encoder`, or `both`.

Built engines are saved directly into `tensorrt_backend/artifacts/` and become immediately selectable upon restarting ComfyUI.

---

## 📜 Credits & References

- **ComfyUI TensorRT Loader:** [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder)
- **SeedVR / SeedVR2 Foundation:** ByteDance Seed Team ([ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR))
- **Official VAE Checkpoint:** [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2) (`seedvr2_ema_vae_fp16.safetensors`)
- **ComfyUI Implementation:** [NumZ](https://github.com/numz) & [AInVFX](https://www.youtube.com/@AInVFX) ([ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler))
- **TensorRT VAE Architecture Inspiration:** [VRGDG-SeedVR2-TensorRT-Studio](https://github.com/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio)
- **Acceleration Framework:** [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- **License:** [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
