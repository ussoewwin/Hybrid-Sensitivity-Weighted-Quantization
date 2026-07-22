# How to quantize SDXL ConvRot INT8

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

## Quantize an SDXL model (HSWQ)

Example: epicrealismXL_pureFix. Adjust the file paths to your environment.

```bash
python quantize_sdxl_hswq_v3.1.py --input "<path-to-unet>/epicrealismXL_pureFix.safetensors" --output "<path-to-unet>/epicrealismXL_pureFix_hswq_v3.1.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25 --convrot --per_channel_int8
```

**Notes:**

- **Samples:** 32 (recommended).
- **Inference steps:** 25 (as in the example).
- **FULL ConvRot** (Linear + Conv2d when `in_dim` is divisible by a power-of-4 group size) is **ON by default**. Pass `--no-convrot` only for plain INT8 without ConvRot.
- **`--per_channel_int8`:** use per-out-channel amax/scale instead of a single per-tensor scale when packing layers that do **not** go through ConvRot. Under default FULL ConvRot, almost all eligible Linear/Conv2d already use rotate + per-channel scale, so this flag has **little effect** in practice; keep it as **insurance** for any remaining non-ConvRot packs. Format tag stays `int8_tensorwise`.
- **Bias correction (Card 1):** **OFF by default.** Pass `--bias_correction` to enable. After INT8 pack, DualMonitor signed channel means \(\mu_x\) from the **same** `--calib_file` run are used to cancel systematic output bias: \(\delta b \approx (W_q - W)\,\mu_x\) (Linear / Conv2d), written into each layer’s `.bias`. No extra tensors and no format-tag change. Optional `--bias_correction_top_ratio < 1` (Approach A) limits correction to high-sensitivity layers; **full layers (`1.0`) is preferred for SSIM**. `--no-bias_correction` forces off (same as the default). **Honest:** Card 1 is **model-dependent**. On some SDXL checkpoints, `--bias_correction` **raises** MSE / SSIM scores; on others it **lowers** them. Treat on vs off as an A/B choice per model — measure both before shipping.

## Quantize an SDXL model (native ConvRot INT8)

No calibration file. Adjust the file paths to your environment. `--model` and `--input` are aliases for the same argument.

```bash
python native_convert_int8_convrot.py --model "<path-to-unet>/your_sdxl_model.safetensors" --output "<path-to-unet>/your_sdxl_model_convrot_int8.safetensors" --per_channel_int8
```

**Notes:**

- **FULL ConvRot** (Linear + Conv2d when `in_dim` is divisible by a power-of-4 group size) is **ON by default**. Pass `--no-convrot` only for plain INT8 without ConvRot.
- **`--per_channel_int8`:** use per-out-channel amax/scale instead of a single per-tensor scale when packing layers that do **not** go through ConvRot. Under default FULL ConvRot, almost all eligible Linear/Conv2d already use rotate + per-channel scale, so this flag has **little effect** in practice; keep it as **insurance** for any remaining non-ConvRot packs. Format tag stays `int8_tensorwise`.
- **Bias correction (Card 1):** **OFF by default.** Pass `--bias_correction` **and** `--calib_file` (same DualMonitor prompts as HSWQ; samples/steps defaults apply). Applies \(\delta b \approx (W_q - W)\,\mu_x\) on **all** packed INT8 Linear + Conv2d (no top-ratio gate on this script). Without `--bias_correction`, no calibration pass is required for a plain pack. **Honest:** same as HSWQ — on vs off **depends on the model**; some checkpoints score better with Card 1 on, some score worse. Benchmark both.

## HSWQ vs native (honest)

HSWQ ConvRot INT8 does **not** always beat native ConvRot INT8 on measured scores (MSE / SSIM). On some SDXL checkpoints, **native scores higher**. Compare both on your own model before choosing which path to ship.

## Benchmark (use this for measurement)

Use `benchmark/int8bench_sdxl.py` to measure FP16 vs quantized INT8 (MSE / SSIM). Adjust paths and the prompt to your environment. `--fp16` is the baseline checkpoint; `--int8` is the INT8 output from either path above.

```bash
python benchmark/int8bench_sdxl.py --fp16 "<path-to-unet>/your_sdxl_model.safetensors" --int8 "<path-to-unet>/your_sdxl_model_int8.safetensors" --prompt "masterpiece, best quality, 1girl, solo, standing, simple background"
```

**Notes:**

- Run this for **both** HSWQ and native outputs against the same FP16 baseline and the same `--prompt` / `--seed` when comparing paths.
- Optional: `--seed` (default `123456789`), `--steps` (default `25`).
