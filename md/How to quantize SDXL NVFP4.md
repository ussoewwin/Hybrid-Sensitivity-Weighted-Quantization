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
pip install diffusers safetensors transformers accelerate tqdm sentencepiece protobuf einops
pip install -r requirements.txt
```

## Quantize an SDXL model

Replace every `<...>` placeholder with a real path on your machine (no invented filenames; no machine-local drive hardcoding in published examples).

```bash
python hswq_convert_nvfp4_1.0.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_hswq_nvfp4.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25
```

**Notes:**

- **Samples:** 32 (recommended).
- **Inference steps:** 25 (as in the example).
- **FULL ConvRot** (Linear → NVFP4, Conv2d → INT8 when `in_dim` is divisible by a power-of-4 group size) is **ON by default**. Pass `--no-convrot` only for plain packs without ConvRot.
- **Bias correction (Card 1):** **OFF by default.** Pass `--bias_correction` to enable (**requires `--calib_file`**). After pack, DualMonitor signed channel means \(\mu_x\) from that calib pass cancel systematic output bias: \(\delta b \approx (W_q - W)\,\mu_x\), written into each corrected layer’s `.bias` (NVFP4 Linear and INT8 Conv paths as packed). The same `--calib_file` run also supplies NVFP4 `.input_scale` / HSWQ sensitivity — Card 1 does **not** need a second calib file. This script has **no** Approach A / `--bias_correction_top_ratio`; when the flag is on, correction covers the packed layers in scope. **Honest:** Card 1 is **model-dependent**. On some SDXL checkpoints, `--bias_correction` **raises** measured scores (MSE / SSIM); on others it **lowers** them. Compare on vs off per model before choosing which pack to ship.
- **Post-convert bench:** **ON by default.** After save, the convert script **clears parent VRAM** (drop convert tensors + `empty_cache`), then runs `benchmark/nvfp4bench_sdxl.py` with `--fp16` = `--model` / `--input` and `--nvfp4` = `--output` (same paths you passed). Pass `--no-bench` to skip. Optional: `--bench_prompt`, `--bench_seed` (default `123456789`), `--bench_steps` (default `25`). The same post-convert bench flags / VRAM clear apply to **`native_convert_nvfp4.py`**.

## Benchmark (use this for measurement)

**HSWQ** (`hswq_convert_nvfp4_1.0.py`) and **native plain** (`native_convert_nvfp4.py`): post-convert bench is integrated (default ON). A separate bench command is only needed for a re-bench with a custom prompt, or when you used `--no-bench`.

Standalone (re-run):

```bash
python benchmark/nvfp4bench_sdxl.py --fp16 "<path-to-unet>/<sdxl_unet>.safetensors" --nvfp4 "<path-to-unet>/<sdxl_unet>_hswq_nvfp4.safetensors" --prompt "masterpiece, best quality, 1girl, solo, standing, simple background"
```

**Notes:**

- Run this for **both** HSWQ and native outputs against the same FP16 baseline and the same `--prompt` / `--seed` when comparing paths.
- Optional: `--seed` (default `123456789`), `--steps` (default `25`).
- Prefer portable paths (relative under the workspace, or any path you substitute into `<path-to-unet>`) so the same command works on cloud instances.
