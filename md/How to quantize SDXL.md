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

## Quantize an SDXL model

Example: koronemixVpred_v20. Adjust the file paths to your environment.

```bash
python quantize_sdxl_hswq_v1.3.py --input "<path-to-unet>/koronemixVpred_v20.safetensors" --output "<output-dir>/koronemixVpred_v20_hswq_r25_s25_r0.25_v1.safetensors" --calib_file "<output-dir>/calibration_prompts_256.txt" --num_calib_samples 25 --num_inference_steps 25 --keep_ratio 0.25
```

**Notes:**

- **Samples:** 25 (recommended).
- **Keep ratio:** 0.25. 0.1 can maintain sufficient quality for SDXL.
- **SageAttention2 (SA2) is not used for SDXL calibration.** Calibration uses native PyTorch SDPA only. SA2 was found to slightly lower calibration scores (SSIM) and to provide no meaningful speed gain, so it is excluded to keep calibration pure and reproducible.
