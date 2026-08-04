# How to quantize Z Image (ZI)

Use **native ConvRot INT8** with `native_convert_int8_convrot.py`.

**HSWQ Z Image INT8 development and public release ended.** For Z Image 8-bit, use this native path (typically **SSIM > 0.99**). HSWQ INT8 continues for **SDXL** only.

**Prefer a Z Image Turbo (ZIT) checkpoint.** Plain Z Image base models are not recommended.

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
pip install safetensors tqdm
```

## Quantize a ZI model

Replace every `<...>` placeholder with a real path on your machine (no invented filenames). `--model` and `--input` are aliases for the same argument.

```bash
python native_convert_int8_convrot.py --model "<path-to-unet>/<zit_unet>.safetensors" --output "<path-to-unet>/<zit_unet>_convrot_int8.safetensors" --per_channel_int8
```

**Notes:**

- **FULL ConvRot** (Linear + Conv2d when `in_dim` is divisible by a power-of-4 group size) is **ON by default**. Pass `--no-convrot` only for plain INT8 without ConvRot.
- **`--per_channel_int8`:** use per-out-channel amax/scale instead of a single per-tensor scale when packing layers that do **not** go through ConvRot. Under default FULL ConvRot, almost all eligible Linear/Conv2d already use rotate + per-channel scale, so this flag has **little effect** in practice; keep it as **insurance** for any remaining non-ConvRot packs. Format tag stays `int8_tensorwise`.
- **`--groupsize`:** ConvRot Hadamard group size (power of 4; default `256`).
- **ComfyUI:** Load the output with [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools).

## Z-Anime page

- **[SeeSee21/Z-Anime](https://huggingface.co/SeeSee21/Z-Anime)** (Hugging Face)
