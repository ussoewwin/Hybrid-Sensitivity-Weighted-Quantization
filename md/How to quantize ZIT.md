# How to quantize Z Image Turbo (ZI)

The dedicated VRAM for the GPU must be **24GB or more**.

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

## Install SageAttention2 (optional, for faster calibration with `--sa2`)

**Windows:**

```bash
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
pip install triton-windows
```

**Linux:**

```bash
pip install sageattention triton
```

Note: This installs SageAttention 1, not SageAttention2 (SA2). For SA2 on Linux, check the [SageAttention](https://github.com/woct0rdho/SageAttention) repository for a compatible build or wheel.

## Download text encoder (CLIP)

Download the text encoder and save it in the **`clip`** folder.

- **[ussoewwin/qwen3_4b_abliterated_fp16](https://huggingface.co/ussoewwin/qwen3_4b_abliterated_fp16)** (Hugging Face)

Use the converted safetensors file, e.g. `clip/qwen3_4b_abliterated_fp16_converted.safetensors`, and pass its path to `--clip_path` when quantizing.

## Quantize a ZI model

Adjust the file paths to your environment.

```bash
python quantize_zit_hswq_v1.5.py --input "path/to/your_zit_model.safetensors" --output "path/to/your_zit_model_hswq_L128_r0.25_v1.safetensors" --clip_path "clip/qwen3_4b_abliterated_fp16_converted.safetensors" --calib_file "sample/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25 --keep_ratio 0.25 --latent 128 --sa2
```

**Notes:**

- **Samples:** 32 (recommended).
- **Latent:** 32–256; 128 (recommended).
- **Keep ratio:** 0.25.
- Use `--latent 32` for faster calibration, `--latent 256` for higher fidelity; default is 128.
- **GPU:** For `--latent 256`, RTX 5090 or above is recommended; for `--latent 32`, RTX 5060 Ti 16GB is sufficient.
- Optional `--sa2` enables SageAttention2 for faster calibration.
