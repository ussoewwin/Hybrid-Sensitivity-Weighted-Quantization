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

Example: waiIllustriousSDXL_v170. Adjust the file paths to your environment.

```bash
python hswq_convert_nvfp4_1.0.py --model "<path-to-unet>/waiIllustriousSDXL_v170.safetensors" --output "<path-to-unet>/waiIllustriousSDXL_v170_hswq_nvfp4.safetensors" --calib_file "<path-to-calib>/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25
```

**Notes:**

- **Samples:** 32 (recommended).
- **Inference steps:** 25 (as in the example).
- **FULL ConvRot** (Linear → NVFP4, Conv2d → INT8 when `in_dim` is divisible by a power-of-4 group size) is **ON by default**. Pass `--no-convrot` only for plain packs without ConvRot.
- **Bias correction (Card 1):** **OFF by default.** Pass `--bias_correction` to enable (**requires `--calib_file`**). After pack, DualMonitor signed channel means \(\mu_x\) from that calib pass cancel systematic output bias: \(\delta b \approx (W_q - W)\,\mu_x\), written into each corrected layer’s `.bias` (NVFP4 Linear and INT8 Conv paths as packed). The same `--calib_file` run also supplies NVFP4 `.input_scale` / HSWQ sensitivity — Card 1 does **not** need a second calib file. This script has **no** Approach A / `--bias_correction_top_ratio`; when the flag is on, correction covers the packed layers in scope.
