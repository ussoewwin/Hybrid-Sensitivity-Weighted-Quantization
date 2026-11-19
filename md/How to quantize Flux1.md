# How to quantize Flux1 ConvRot INT8

Quantize a **FLUX.1 DiT** (`model.diffusion_model.*` layout) to **ConvRot INT8** with the
command-line converter `Flux1/native_convert_int8_convrot_flux.py` (native = hswq-free:
every eligible Linear is rotated and packed INT8, no protection set).

The post-convert fidelity benchmark is **latent-space trajectory divergence** (per-step
cosine + bifurcation detection) on the same seed for both models, which measures the
sampling path itself rather than a single forward pass.

FLUX needs **two text encoders** (`t5xxl` + `clip_l`) for any benchmark run - both are
passed explicitly (`--clip_path` = t5xxl, `--clip_l_path` = clip_l).

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
pip install scikit-image
```

`comfy_kitchen` provides the quantization kernels and layout operations.
`scikit-image` is required for the SSIM / MSE image-space benchmark.

## Quantize a Flux1 model (CLI)

Replace every `<...>` placeholder with a real path on your machine. `--model` and `--input`
are aliases for the same argument.

```bash
python Flux1/native_convert_int8_convrot_flux.py \
  --model "<path-to-unet>/<flux1>.safetensors" \
  --output "<path-to-unet>/<flux1>_native_convrot_int8.safetensors" \
  --no-bench
```

**Notes:**

- **FULL ConvRot** (Linear + Conv2d when `in_dim` is divisible by a power-of-4 group size)
  is **ON by default**. Pass `--no-convrot` only for plain INT8 without ConvRot.
- **Group size:** preferred size is `256`; when `in_features` is not divisible by 256 the
  converter adapts to the largest power-of-4 divisor (>= 4). The stamp written per layer is
  `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}`.
- **`--no-bench`:** pass this to convert and save only. When `--clip_path` / `--clip_l_path` /
  `--comfy_path` (and optionally `--vae`) are given instead, the converter runs the
  image-space benchmark automatically after saving; `--no-bench` skips it.
- The output keeps the source's keys: load it with the **standard ComfyUI loader**
  (`Load Diffusion Model` / `UNetLoader`). No dedicated loader is required, and the FLUX
  text encoders / VAE stay separate files as usual.

## Validation & Benchmarking

### Latent trajectory divergence (deterministic 25-seed)

Compares the FP16 source and the ConvRot INT8 pack step by step with the same initial
noise per seed:

```bash
python benchmark/flux1_convrot_int8_traj_compare.py \
  --fp16 "<path-to-unet>/<flux1>.safetensors" \
  --int8 "<path-to-unet>/<flux1>_native_convrot_int8.safetensors" \
  --clip_path "<path-to-clip>/<t5xxl>.safetensors" \
  --clip_l_path "<path-to-clip>/<clip_l>.safetensors" \
  --comfy_path "<path-to-ComfyUI>" \
  --steps 25 \
  --seeds "42,137,849,2024,7391,18429,53082,149206,382715,826401,1938502,4710928,8391642,15820493,36192847,71058294,128491703,285039184,491730285,762019483,938174026,1409285713,2683910547,3851729406,4195820371"
```

- Same seed -> identical noise for both models (fully deterministic).
- Reports per-step + final latent MSE / cosine, the largest single-step cosine drop, and
  `same-image` (final cosine >= 0.98) / `bifurcated` (a single step drops > 0.05) / `drifted`.
- Both CLIPs are loaded on CPU, encoded, and unloaded before the DiT runs, so the whole GPU
  belongs to the diffusion model.

### Image-space benchmark (MSE / SSIM)

```bash
python benchmark/flux1_convrot_int8_traj_compare.py \
  --fp16 "<path-to-unet>/<flux1>.safetensors" \
  --int8 "<path-to-unet>/<flux1>_native_convrot_int8.safetensors" \
  --clip_path "<path-to-clip>/<t5xxl>.safetensors" \
  --clip_l_path "<path-to-clip>/<clip_l>.safetensors" \
  --comfy_path "<path-to-ComfyUI>"
```

Score files belong in `benchmark result/` in the repository (per the repository's score
convention).

## ComfyUI Deployment

The resulting `.safetensors` is compatible with standard ComfyUI:

1. Place the quantized file into `ComfyUI/models/diffusion_models/` (or `models/unet/`).
2. Load it with **Load Diffusion Model** (or **UNetLoader**).
3. Load the FLUX text encoders as usual: **DualCLIPLoader** with `clip_l` + `t5xxl`
   (type `flux`), and the FLUX VAE with **VAELoader**.
4. `comfy_kitchen` executes the INT8 GEMM with online activation rotation - no extra node
   is needed for the pack itself.
