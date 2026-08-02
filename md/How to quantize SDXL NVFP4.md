# How to quantize SDXL NVFP4

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

`scikit-image` is required for SSIM in `benchmark/nvfp4bench_sdxl.py` (post-quantize bench and standalone re-runs).

## Quantize an SDXL model (HSWQ)

Replace every `<...>` placeholder with a real path on your machine.

`--model` and `--input` are aliases for the same argument.

**Default flow:** quantize → save → **clear parent VRAM** → run **`benchmark/nvfp4bench_sdxl.py`** automatically (`--fp16` = `--model`/`--input`, `--nvfp4` = `--output`). You do **not** need a second manual bench command after a normal HSWQ NVFP4 run.

```bash
python hswq_sdxl_convert_nvfp4_1.0.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_hswq_nvfp4.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25
```

With bias correction (still includes post-quantize bench by default):

```bash
python hswq_sdxl_convert_nvfp4_1.0.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_hswq_nvfp4.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25 --bias_correction
```

Optional: skip the integrated bench:

```bash
python hswq_sdxl_convert_nvfp4_1.0.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_hswq_nvfp4.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25 --no-bench
```

**Notes:**

- **Samples:** 32 (recommended).
- **Inference steps:** 25 (as in the example).
- **FULL ConvRot** (Linear + Conv2d when `in_dim` is divisible by a power-of-4 group size) is **ON by default** — it is **not** a flag on the example commands above. Eligible Linear → offline Hadamard + **NVFP4**; eligible Conv2d → offline Hadamard + **INT8** `int8_tensorwise` (NVFP4 layout is 2D-only). Pass `--no-convrot` only for plain packs without ConvRot.
- **`--calib_file` / `.input_scale`:** Pass a prompts file as in the examples. Calib writes per-layer NVFP4 **`.input_scale`** and runs DualMonitor + V4 pack-MSE FP16 protect under **`--fp16_budget_mb` hard ceiling 600 MiB** (`--keep_ratio` must stay **0** / r0). Without `--calib_file`, no `.input_scale` is written and inference falls back to ones — that destroys quality. `--bias_correction` also requires `--calib_file`.
- **Bias correction (Card 1):** **OFF by default.** Pass `--bias_correction` to enable. After pack, DualMonitor signed channel means \(\mu_x\) from the **same** `--calib_file` run are used to cancel systematic output bias: \(\delta b \approx (W_q - W)\,\mu_x\) (Linear / Conv2d), written into each layer’s `.bias`. No extra tensors and no format-tag change. **Honest:** Card 1 is **model-dependent**. On some SDXL checkpoints, `--bias_correction` **raises** MSE / SSIM scores; on others it **lowers** them. Treat on vs off as an A/B choice per model — measure both before shipping.
- **Post-quantize bench:** **ON by default** (`--bench`). After save, the script **clears parent VRAM** (drop convert tensors + `empty_cache` / `ipc_collect`), then runs `benchmark/nvfp4bench_sdxl.py --fp16 <input> --nvfp4 <output> --prompt "masterpiece, best quality, 1girl, solo, standing, simple background" --seed 123456789` (prompt and seed fixed inside the script). Pass `--no-bench` to skip. A non-zero bench exit code fails the convert process.

## Quantize an SDXL model (native NVFP4)

No calibration file. `--model` and `--input` are aliases for the same argument.

```bash
python native_convert_nvfp4.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_nvfp4.safetensors"
```

**Notes:**

- **FULL ConvRot** is **not** on this script (plain Kitchen `TensorCoreNVFP4Layout` only). Use **`hswq_sdxl_convert_nvfp4_1.0.py`** for HSWQ + calib. Non-diffusion tensors (CLIP / VAE markers) stay non-4-bit.
- **`--model_type`:** Kitchen DiT profile blacklist / FP8-layer lists (default `Z-Image-Turbo`). Those name prefixes usually do **not** match SDXL UNet keys, so on a UNet-only SDXL file the blacklist is effectively empty; the non-diffusion markers above still apply if CLIP/VAE keys are present.
- **Bias correction (Card 1):** **OFF by default** and **not** available without DualMonitor on this script. Pass `--bias_correction` **and** `--calib_file` on **`hswq_sdxl_convert_nvfp4_1.0.py`** when needed. **Honest:** same as HSWQ — on vs off **depends on the model**; some checkpoints score better with Card 1 on, some score worse. Benchmark both.
- **Post-convert bench:** **ON by default** (`--bench`). After save, clears parent VRAM then runs the fixed NVFP4 fidelity bench (`--fp16` = `--model`, `--nvfp4` = `--output`, fixed `--prompt` / `--seed`). Pass `--no-bench` to skip.

## HSWQ vs native (honest)

HSWQ NVFP4 does **not** always beat native NVFP4 on measured scores (MSE / SSIM). On some SDXL checkpoints, **native scores higher**. Compare both on your own model before choosing which path to ship.

To run HSWQ without ConvRot, use `hswq_sdxl_convert_nvfp4_1.2.py`. Options and usage are identical to the 1.0 version.

```bash
python hswq_sdxl_convert_nvfp4_1.2.py --model "<path-to-unet>/<sdxl_unet>.safetensors" --output "<path-to-unet>/<sdxl_unet>_hswq_nvfp4.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25
```

## Benchmark (use this for measurement)

### Integrated (HSWQ NVFP4 — preferred)

`hswq_sdxl_convert_nvfp4_1.0.py` already chains fidelity measurement after save:

1. Save the NVFP4 pack.
2. Clear parent VRAM (so the bench subprocess does not OOM on a 12GB+ card).
3. Spawn:
   `benchmark/nvfp4bench_sdxl.py --fp16 <input> --nvfp4 <output> --prompt "masterpiece, best quality, 1girl, solo, standing, simple background" --seed 123456789`
   (prompt and seed are fixed inside the quantize/convert scripts; not parent CLI overrides).

Pass `--no-bench` to quantize only. Do not invent other bench CLI flags on the quantize script.

### Standalone (re-run / after `--no-bench`)

A separate bench command is needed when the quantize/convert run used `--no-bench`, or to re-measure an existing pack:

```bash
python benchmark/nvfp4bench_sdxl.py --fp16 "<path-to-unet>/<sdxl_unet>.safetensors" --nvfp4 "<path-to-unet>/<sdxl_unet>_nvfp4.safetensors" --prompt "masterpiece, best quality, 1girl, solo, standing, simple background"
```

**Notes:**

- Run this for **both** HSWQ and native outputs against the same FP16 baseline and the same `--prompt` when comparing paths.
- Prefer portable paths (relative under the workspace, or any path you substitute into `<path-to-unet>`) so the same command works on cloud instances.
- Needs a usable `ComfyUI-master` (or `COMFYUI_PATH`) tree so `nvfp4bench_sdxl.py` can load packs via Comfy `CheckpointLoaderSimple` (Kitchen / scale layout as saved).
