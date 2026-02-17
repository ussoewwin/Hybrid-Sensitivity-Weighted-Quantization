# How to quantize SDXL

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

## Install SageAttention2 (optional, for faster calibration with `--sa2`)

**Windows:**

```bash
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
```

**Linux:**

```bash
pip install sageattention
```

Note: This installs SageAttention 1, not SageAttention2 (SA2). For SA2 on Linux, check the [SageAttention](https://github.com/woct0rdho/SageAttention) repository for a compatible build or wheel.

## Quantize an SDXL model

Example: koronemixVpred_v20. Adjust the file paths to your environment.

```bash
python quantize_sdxl_hswq_v1.2.py --input "D:\USERFILES\ComfyUI\ComfyUI\models\unet\koronemixVpred_v20.safetensors" --output "D:\USERFILES\fp8e4m3\koronemixVpred_v20_hswq_r256_s25_r0.25_v1.safetensors" --calib_file "D:\USERFILES\fp8e4m3\calibration_prompts_256.txt" --num_calib_samples 256 --num_inference_steps 25 --keep_ratio 0.25 --sa2
```

**Notes:**

- The recommended value is **256**; in practice, a sample size of **32** is sufficient to maintain adequate precision.
- The ratio for retaining FP16 can also maintain sufficient quality at **0.1** in the case of SDXL.
