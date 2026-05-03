# How to quantize Z Image (ZI)

The dedicated VRAM for the GPU must be **24GB or more**.

**Z Image base models are not recommended for HSWQ quantization. Please select a Z Image turbo model instead.**

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

## `quantize_zib_hswq_v1.92.py` and attention backends

`quantize_zib_hswq_v1.92.py` does **not** define a `--sa2` (SageAttention2) flag. Use only the arguments shown in the example below. You do **not** need a separate “install SageAttention2 for quantization” step for this guide.

## Download text encoder (CLIP)

Download the text encoder and save it in the **`clip`** folder.

- **[ussoewwin/qwen3_4b_abliterated_fp16](https://huggingface.co/ussoewwin/qwen3_4b_abliterated_fp16)** (Hugging Face)

Use the converted safetensors file, e.g. `clip/qwen3_4b_abliterated_fp16_converted.safetensors`, and pass its path to `--clip_path` when quantizing.

## Quantize a ZI model

Adjust the file paths to your environment.

```bash
python quantize_zib_hswq_v1.92.py --input "path/to/your_zit_model.safetensors" --output "path/to/your_zit_model_hswq_r32_r0.25_v1.safetensors" --clip_path "clip/qwen3_4b_abliterated_fp16_converted.safetensors" --calib_file "sample/calibration_prompts_128.txt" --num_calib_samples 32 --num_inference_steps 25 --keep_ratio 0.1
```

**Notes:**

- **Samples:** 32 (recommended).
- **Keep ratio:** 0.1 (as in the example); the valid range is typically `0.05`–`0.25`. For SDXL and ZIT, 0.05–0.10 often gives sufficient quality. Adjust if you want to trade off quality vs. memory/speed.
