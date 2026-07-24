# How to quantize SDXL ConvRot NVFP4

The dedicated VRAM for the GPU must be **12GB or more**.

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
pip install diffusers safetensors transformers accelerate tqdm sentencepiece protobuf einops scikit-image
pip install -r requirements.txt
```

`scikit-image` is required for SSIM in `benchmark/nvfp4bench_sdxl.py` (post-convert bench and standalone re-runs).

## Quantize an SDXL model (HSWQ)

Replace every `<...>` placeholder with a real path on your machine (no invented filenames; no machine-local drive hardcoding in published examples).

**Default flow:** convert → save → **clear parent VRAM** → run **`benchmark/nvfp4bench_sdxl.py`** automatically (`--fp16` = `--model`, `--nvfp4` = `--output`). You do **not** need a second manual bench command after a normal HSWQ NVFP4 run.

```bash
python hswq_convert_nvfp4_1.0.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_hswq_nvfp4.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25
```

With bias correction (still includes post-convert bench by default):

```bash
python hswq_convert_nvfp4_1.0.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_hswq_nvfp4.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25 --bias_correction
```

Optional: skip the integrated bench:

```bash
python hswq_convert_nvfp4_1.0.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_hswq_nvfp4.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25 --no-bench
```

**Notes:**

- **Samples:** 32 (recommended).
- **Inference steps:** 25 (as in the example).
- **FULL ConvRot** (Linear → NVFP4, Conv2d → INT8 when `in_dim` is divisible by a power-of-4 group size) is **ON by default**. Pass `--no-convrot` only for plain packs without ConvRot.
- **Bias correction (Card 1):** **OFF by default.** Pass `--bias_correction` to enable (**requires `--calib_file`**). After pack, DualMonitor signed channel means \(\mu_x\) from that calib pass cancel systematic output bias: \(\delta b \approx (W_q - W)\,\mu_x\), written into each corrected layer’s `.bias` (NVFP4 Linear and INT8 Conv paths as packed). The same `--calib_file` run also supplies NVFP4 `.input_scale` / HSWQ sensitivity — Card 1 does **not** need a second calib file. This script has **no** Approach A / `--bias_correction_top_ratio`; when the flag is on, correction covers the packed layers in scope. **Honest:** Card 1 is **model-dependent**. On some SDXL checkpoints, `--bias_correction` **raises** measured scores (MSE / SSIM); on others it **lowers** them. Compare on vs off per model before choosing which pack to ship.
- **Post-convert bench:** **ON by default** (`--bench`). After save, the script **clears parent VRAM** (drop convert tensors + `empty_cache` / `ipc_collect`), then runs `benchmark/nvfp4bench_sdxl.py --fp16 <model> --nvfp4 <output> --prompt "masterpiece, best quality, 1girl, solo, standing, simple background" --seed 123456789` (prompt and seed fixed inside the script). Pass `--no-bench` to skip. A non-zero bench exit code fails the convert process.

## Quantize an SDXL model (native NVFP4)

No calibration file. `--model` and `--input` are aliases for the same argument.

```bash
python native_convert_nvfp4.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_nvfp4.safetensors"
```

**Notes:**

- Plain Kitchen NVFP4 pack (no ConvRot / no calib / no `.input_scale` on this path).
- **Post-convert bench:** **ON by default** (`--bench`). After save, clears parent VRAM then runs the fixed NVFP4 fidelity bench (`--fp16` = `--model`, `--nvfp4` = `--output`, fixed `--prompt` / `--seed`). Pass `--no-bench` to skip.

## HSWQ vs native (honest)

HSWQ ConvRot NVFP4 does **not** always beat native NVFP4 on measured scores (MSE / SSIM). On some SDXL checkpoints, **native scores higher**. Compare both on your own model before choosing which path to ship.

## Benchmark (use this for measurement)

### Integrated (HSWQ NVFP4 — preferred)

`hswq_convert_nvfp4_1.0.py` already chains fidelity measurement after save:

1. Save the NVFP4 pack.
2. Clear parent VRAM (so the bench subprocess does not OOM on a 12GB+ card).
3. Spawn:
   `benchmark/nvfp4bench_sdxl.py --fp16 <model> --nvfp4 <output> --prompt "masterpiece, best quality, 1girl, solo, standing, simple background" --seed 123456789`
   (prompt and seed are fixed inside the convert scripts; not parent CLI overrides).

Pass `--no-bench` to convert only. Do not invent other bench CLI flags on the convert script.

### Standalone (re-run / after `--no-bench`)

A separate bench command is needed when the convert run used `--no-bench`, or to re-measure an existing pack:

```bash
python benchmark/nvfp4bench_sdxl.py --fp16 "<path-to-unet>/<sdxl_unet>.safetensors" --nvfp4 "<path-to-unet>/<sdxl_unet>_nvfp4.safetensors" --prompt "masterpiece, best quality, 1girl, solo, standing, simple background"
```

**Notes:**

- Run this for **both** HSWQ and native outputs against the same FP16 baseline and the same `--prompt` when comparing paths.
- Prefer portable paths (relative under the workspace, or any path you substitute into `<path-to-unet>`) so the same command works on cloud instances.
- Needs a usable `ComfyUI-master` (or `COMFYUI_PATH`) tree for NVFP4 load.
