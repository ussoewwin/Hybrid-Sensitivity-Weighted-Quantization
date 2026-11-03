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

## 🚀 Usage in ComfyUI
These TensorRT engine plans (`.rtxplan`) are directly loaded and executed via the dedicated custom node:
- **Loader & Node Repository:** [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT)

### 1. Installation
Clone the custom node repository into your ComfyUI `custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT.git
```

### 2. Engine Placement
Place the downloaded `.rtxplan` engine files into either of the following supported directories:

```
ComfyUI/custom_nodes/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/tensorrt_backend/artifacts/
```
or
```
ComfyUI/models/tensorrt/seedvr2/
```

Upon launching ComfyUI, all detected `.rtxplan` files will automatically populate the `engine_frames` and `engine_tile` dropdown lists of the respective loader nodes:
- `vae_decoder_tile_*_*f.rtxplan` populates **`SeedVR2 Load TensorRT VAE Decoder`**
- `vae_encoder_*f_tile*.rtxplan` populates **`SeedVR2 Load TensorRT VAE Encoder`**

### 3. Node Integration
Connect both loader nodes directly to the **`SeedVR2 Video Upscaler`** node:
- Connect the `SEEDVR2_VAE` output of **`SeedVR2 Load TensorRT VAE Encoder`** to the **`vae_encode`** input.
- Connect the `SEEDVR2_VAE` output of **`SeedVR2 Load TensorRT VAE Decoder`** to the **`vae_decode`** input.

```
+------------------------------------+
| SeedVR2 Load TensorRT VAE Encoder  |
| [engine_frames: 21 / 45 / 1]       |--(SEEDVR2_VAE)---+
| [engine_tile: 512 / 1024]          |                  |
+------------------------------------+                  |
                                                        v [vae_encode]
                                         +----------------------------+
                                         |   SeedVR2 Video Upscaler   |
                                         +----------------------------+
                                                        ^ [vae_decode]
+------------------------------------+                  |
| SeedVR2 Load TensorRT VAE Decoder  |                  |
| [engine_frames: 21 / 45 / 1]       |--(SEEDVR2_VAE)---+
| [engine_tile: 512 / 1024]          |
+------------------------------------+
```

### 4. Workflow Examples
#### Complete Video Upscaler Workflow (TensorRT VAE & Quantized Models)
Workflow JSON: [`example_workflows/SeedVR2_tensorrt_decode.json`](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/example_workflows/SeedVR2_tensorrt_decode.json)

![Usage Example - Full Workflow](docs/usage_01.png)

#### TensorRT VAE Decoder & Encoder Node Integration
Connect the `SEEDVR2_VAE` outputs from **`SeedVR2 Load TensorRT VAE Decoder`** and **`SeedVR2 Load TensorRT VAE Encoder`** directly into the **`SeedVR2 Video Upscaler`** node:

![Usage Example - TensorRT VAE Nodes](docs/usage_02.png)

## 📜 Credits & References
- **ComfyUI TensorRT Loader:** [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT)
- **SeedVR / SeedVR2 Foundation:** ByteDance Seed Team ([ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR))
- **Official VAE Checkpoint:** [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2) (`seedvr2_ema_vae_fp16.safetensors`)
- **ComfyUI Implementation:** [NumZ](https://github.com/numz) & [AInVFX](https://www.youtube.com/@AInVFX) ([ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler))
- **TensorRT VAE Architecture Inspiration:** [VRGDG-SeedVR2-TensorRT-Studio](https://github.com/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio)
- **Acceleration Framework:** [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- **License:** [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## 📚 Detailed documentation

- [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT)

## 🔒 Security

**Backdoor audit.** Each file's audit status - verdict strength, weight coverage, the record behind it,
and the limits of what that measurement can show - is in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/backdoor/Backdoor_Audit_Verdicts.md). The audit code, the procedure and the raw records are in the same
repository.
