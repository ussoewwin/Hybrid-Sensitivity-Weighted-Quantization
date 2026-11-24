# How to quantize MOSS-Image

Use **native ConvRot INT8** via CLI (`moss/native_convert_int8_moss.py`).

The post-convert fidelity benchmark is **latent-space trajectory divergence** (per-step cosine + bifurcation detection) with the deterministic 25-seed set, which is a stronger measure than decoded SSIM for quantization accuracy.

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
```

`comfy_kitchen` provides the quantization kernels and layout operations. `diffusers accelerate scikit-image` are not needed by the MOSS converter itself.

## Model detection

The converter detects the MOSS-Image family the same way ComfyUI `model_detection` does, through the signature `shared_block_modulation.weight` + `layers.0.proj.ffn_up_gate.weight`. Detection is **fatal on failure**: a file without the signature aborts instead of silently producing a wrong-format pack.

No structure blacklist and no BF16 keep applies on MOSS-Image: every DiT Linear/Conv2d `.weight` (fp16/fp32/bf16) that is ConvRot-eligible is packed, and non-diffusion tensors keep their original dtype as-is. MOSS-Image has no embedded VAE; the text encoder (Qwen3.5-9B) is a separate file.

## Quantize a MOSS-Image model (CLI)

Replace every `<...>` placeholder with a real path on your machine.

Required: **`--model`** (alias `--input`) and **`--output`** only.

```bash
python "moss/native_convert_int8_moss.py" --model "<path-to-unet>/<moss>.safetensors" --output "<path-to-unet>/<moss>_convrot_int8.safetensors"
```

**Notes:**

- **FULL ConvRot** (Linear + Conv2d when `in_dim` / `in_channels` is divisible by a power-of-4 group size) is **ON by default**. Pass `--no-convrot` only for plain INT8 without ConvRot.
- **`--group_size`** (default `256`): preferred ConvRot Hadamard group size; layers whose input width is not divisible by a power-of-4 divisor fall back to a smaller power of 4, and layers with no power-of-4 divisor stay plain tensorwise.
- **`--per_channel_int8`:** per-out-channel amax/scale for plain (non-ConvRot) packs only; format tag stays `int8_tensorwise`.
- **Bias correction (Card 1, optional):** `--bias_correction` + `--calib_file <prompts.txt>` + `--clip_path <qwen3.5-9b pack>` + `--comfy_path <ComfyUI root>` applies Card 1 (`bias += -(W_q - W) @ mu_x`) on all INT8 Linear/Conv, with the prompts encoded through the Comfy CLIP at `CLIPType.MOSS_IMAGE` and the act moments measured by DualMonitor over `--num_calib_samples` (default 32) samples at `--num_inference_steps` (default 25) timesteps. On the reference checkpoint the DualMonitor covered 185 layers and the correction applied to the 16 layers that carry bias (`no_bias=169`, `no_act=0`, `bad_shape=0`).
- The run prints `Converted layers: 185, Kept (other): 193` on the reference checkpoint (reference model 10,164,663,992 B → pack 5,088,552,570 B, ≈ 50% of the FP16 base).

## Validate the pack (25-seed trajectory)

```bash
python "benchmark/moss_int8_traj_compare.py" --fp16 "<path-to-unet>/<moss>.safetensors" --int8 "<path-to-unet>/<moss>_convrot_int8.safetensors" --clip_path "<path-to-qwen3.5-9b-pack>" --comfy_path "<path-to-ComfyUI>"
```

`--seeds` defaults to the deterministic 25-seed set used by every model family in this repository; `--out <file.json>` writes the summary. Production gate: **mean final cosine ≥ 0.95 AND 0/25 bifurcated** (per-seed verdicts: `same-image` = final cosine ≥ 0.98 and no step crossed the threshold; `bifurcated` = single-step cosine drop > 0.05; `drifted` = otherwise).

Reference result on the published checkpoint (`moss_int8_traj_25.json`): mean final cosine **0.97779**, min 0.92547, max 0.99907, same-image 16/25, **bifurcated 0/25** — **GATE PASS**.

## ComfyUI

Load the ConvRot INT8 output with the **standard ComfyUI loader** (`Load Diffusion Model`); the `int8_tensorwise` stamp with the ConvRot marker is read by ComfyUI's mixed-precision ops, so a dedicated HSWQ loader is not required. The text encoder pack loads on the stock CLIP path (`CLIPType.MOSS_IMAGE`).
